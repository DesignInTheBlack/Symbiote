use serde::{Deserialize, Serialize};
use serde_json::Value;
use reqwest::Client;
use url::Url;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use std::sync::Arc;
use once_cell::sync::Lazy;
use tokio::sync::Mutex;
use crate::core::memory::cache;
use crate::core::memory::candidate;
use crate::core::prompt_loader;
use crate::core::system_log;
use crate::core::token_estimator;
use crate::core::kernel::KernelState;
use crate::models::{ConversationSummaryChunk, EpisodicEvent, Settings};
use sha2::{Digest, Sha256};
use uuid::Uuid;
use sqlx::Row;

const PERF_WARN_MEMORY_MS: i64 = 1500;
const DEFAULT_CONTEXT_LIMIT_TOKENS: i32 = 16_384;
const REPAIR_CONFIDENCE_THRESHOLD: f32 = 0.6;

#[derive(Debug, Clone, Copy)]
struct MemoryPassGate {
    allowed: bool,
    reason: &'static str,
}

impl MemoryPassGate {
    fn allow() -> Self {
        Self { allowed: true, reason: "ok" }
    }

    fn deny(reason: &'static str) -> Self {
        Self { allowed: false, reason }
    }
}

pub fn repair_prediction_json(raw: &str) -> (Option<Value>, bool) {
    crate::core::kernel::repair_json_object(raw)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<crate::models::Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>, // "auto", "none"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill: Option<Value>,
    #[serde(skip)]
    pub skip_injection: Option<bool>,
    #[serde(skip)]
    pub skip_memory: Option<bool>,
    #[serde(skip)]
    pub skip_reminders: Option<bool>,
    #[serde(skip)]
    pub memory_expand: Option<bool>,
    #[serde(skip)]
    pub allow_diagnostics: Option<bool>,
    #[serde(skip)]
    pub json_strict: Option<bool>,
    #[serde(skip)]
    pub skip_sanitization: Option<bool>,
    #[serde(skip)]
    pub run_id: Option<String>,
    #[serde(skip)]
    pub request_label: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatResponseMeta {
    pub raw_content: String,
    pub content_no_tags: String,
    pub content: String,
    pub tool_calls: Option<Vec<crate::models::ToolCall>>,
    pub tag_set: crate::core::memory::inject_context::SystemTagSet,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct MemoryPassResult {
    pub success: bool,
    pub error: Option<String>,
    pub conflict_ids: Vec<i64>,
    pub pending_clarify: bool,
    pub written_ids: usize,
    pub facts_written: usize,
    pub rels_written: usize,
}

const MAX_CONFIRM_BELIEFS: usize = 12;
static LAST_RECALLED_BY_CONV: Lazy<Mutex<HashMap<String, Vec<i64>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub data: Vec<EmbeddingData>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponseDelta {
    pub choices: Vec<DeltaChoice>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeltaChoice {
    pub delta: DeltaContent,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeltaContent {
    pub content: Option<String>,
    #[serde(rename = "tool_calls")]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeltaToolCall {
    pub index: Option<u32>,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub r#type: Option<String>,
    pub function: Option<DeltaToolCallFunction>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeltaToolCallFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

use sqlx::{Pool, Sqlite};
use tauri::{AppHandle, Emitter};
use crate::core::reminder_blocks;

const SCAFFOLD_BEGIN: &str = "<<<BEGIN_SECTION:";
const SCAFFOLD_END: &str = "<<<END_SECTION:";
const INTERNAL_BEGIN: &str = "<INTERNAL>";
const INTERNAL_END: &str = "</INTERNAL>";
const STREAM_FLUSH_THRESHOLD: usize = 24;
const STREAM_FLUSH_TAIL: usize = 12;

struct ScaffoldStreamFilter {
    buffer: String,
    in_block: bool,
    line_buffer: String,
}

impl ScaffoldStreamFilter {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            in_block: false,
            line_buffer: String::new(),
        }
    }

    fn filter_chunk(&mut self, chunk: &str) -> String {
        let cleaned = self.strip_scaffold_blocks(chunk);
        self.filter_headings(&cleaned)
    }

    fn finalize(&mut self) -> String {
        if self.in_block {
            self.buffer.clear();
            self.line_buffer.clear();
            self.in_block = false;
            return String::new();
        }

        let mut output = String::new();
        if !self.buffer.is_empty() {
            let remaining = std::mem::take(&mut self.buffer);
            output.push_str(&self.filter_headings(&remaining));
        }
        if !self.line_buffer.is_empty() {
            let tail = std::mem::take(&mut self.line_buffer);
            if !is_scaffold_heading(&tail) {
                output.push_str(&tail);
            }
        }
        output
    }

    fn strip_scaffold_blocks(&mut self, chunk: &str) -> String {
        self.buffer.push_str(chunk);
        let mut output = String::new();

        loop {
            if self.in_block {
                if let Some(end_idx) = self.buffer.find(SCAFFOLD_END) {
                    if let Some(close_rel) = self.buffer[end_idx..].find(">>>") {
                        let close_idx = end_idx + close_rel + 3;
                        self.buffer = self.buffer[close_idx..].to_string();
                        self.in_block = false;
                        continue;
                    }
                }
                self.trim_buffer(SCAFFOLD_END.len().saturating_sub(1));
                break;
            } else if let Some(start_idx) = self.buffer.find(SCAFFOLD_BEGIN) {
                output.push_str(&self.buffer[..start_idx]);
                self.buffer = self.buffer[start_idx + SCAFFOLD_BEGIN.len()..].to_string();
                self.in_block = true;
                continue;
            } else {
                let keep = SCAFFOLD_BEGIN.len().saturating_sub(1);
                if self.buffer.len() > keep {
                    let emit_len = self.buffer.len() - keep;
                    let emit_idx = clamp_char_boundary(&self.buffer, emit_len);
                    if emit_idx > 0 {
                        output.push_str(&self.buffer[..emit_idx]);
                        self.buffer = self.buffer[emit_idx..].to_string();
                    }
                }
                break;
            }
        }

        output
    }

    fn trim_buffer(&mut self, keep: usize) {
        if self.buffer.len() <= keep {
            return;
        }
        let start_idx = clamp_char_boundary(&self.buffer, self.buffer.len().saturating_sub(keep));
        self.buffer = self.buffer[start_idx..].to_string();
    }

    fn filter_headings(&mut self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        self.line_buffer.push_str(text);
        let mut output = String::new();
        while let Some(pos) = self.line_buffer.find('\n') {
            let line = self.line_buffer[..pos].to_string();
            self.line_buffer = self.line_buffer[pos + 1..].to_string();
            if !is_scaffold_heading(&line) {
                output.push_str(&line);
                output.push('\n');
            }
        }
        output.push_str(&self.flush_partial_line());
        output
    }

    fn flush_partial_line(&mut self) -> String {
        if self.line_buffer.len() < STREAM_FLUSH_THRESHOLD {
            return String::new();
        }
        let trimmed = self.line_buffer.trim_start();
        if trimmed.is_empty() || is_scaffold_heading(trimmed) {
            return String::new();
        }
        let flush_len = self.line_buffer.len().saturating_sub(STREAM_FLUSH_TAIL);
        let flush_idx = clamp_char_boundary(&self.line_buffer, flush_len);
        if flush_idx == 0 {
            return String::new();
        }
        let flushed = self.line_buffer[..flush_idx].to_string();
        self.line_buffer = self.line_buffer[flush_idx..].to_string();
        flushed
    }
}

struct InternalStreamFilter {
    buffer: String,
    in_block: bool,
    line_buffer: String,
    allow_diagnostics: bool,
    assistant_label: Option<String>,
    user_label: Option<String>,
}

fn detect_meta_response(text: &str) -> Option<&'static str> {
    let lowered = text.to_lowercase();
    let patterns = [
        ("suggested_response", "here's a suggested response"),
        ("assistant_should", "the assistant should"),
        ("in_response", "in response to the user's input"),
    ];
    for (label, pat) in patterns {
        if lowered.contains(pat) {
            return Some(label);
        }
    }
    None
}

impl InternalStreamFilter {
    fn new(allow_diagnostics: bool, assistant_name: Option<&str>, user_name: Option<&str>) -> Self {
        Self {
            buffer: String::new(),
            in_block: false,
            line_buffer: String::new(),
            allow_diagnostics,
            assistant_label: assistant_name
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty()),
            user_label: user_name
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty()),
        }
    }

    fn filter_chunk(&mut self, chunk: &str) -> String {
        let cleaned = self.strip_internal_blocks(chunk);
        self.filter_lines(&cleaned)
    }

    fn finalize(&mut self) -> String {
        if self.in_block {
            self.buffer.clear();
            self.line_buffer.clear();
            self.in_block = false;
            return String::new();
        }
        let mut output = String::new();
        if !self.buffer.is_empty() {
            let remaining = std::mem::take(&mut self.buffer);
            output.push_str(&self.filter_lines(&remaining));
        }
        if !self.line_buffer.is_empty() {
            let tail = std::mem::take(&mut self.line_buffer);
            if (self.allow_diagnostics || !is_diagnostic_line(&tail))
                && !is_role_label_line(
                    &tail,
                    self.assistant_label.as_deref(),
                    self.user_label.as_deref(),
                )
                && !is_workspace_scaffold_line(&tail)
                && (self.allow_diagnostics
                    || (!is_tool_list_line(&tail)
                        && !is_kv_dump_line(&tail)
                        && !is_identity_anchor_line(&tail)))
            {
                output.push_str(&tail);
            }
        }
        output
    }

    fn strip_internal_blocks(&mut self, chunk: &str) -> String {
        self.buffer.push_str(chunk);
        let mut output = String::new();
        loop {
            if self.in_block {
                if let Some(end_idx) = self.buffer.find(INTERNAL_END) {
                    let close_idx = end_idx + INTERNAL_END.len();
                    self.buffer = self.buffer[close_idx..].to_string();
                    self.in_block = false;
                    continue;
                }
                self.trim_buffer(INTERNAL_END.len().saturating_sub(1));
                break;
            } else if let Some(start_idx) = self.buffer.find(INTERNAL_BEGIN) {
                output.push_str(&self.buffer[..start_idx]);
                self.buffer = self.buffer[start_idx + INTERNAL_BEGIN.len()..].to_string();
                self.in_block = true;
                continue;
            } else {
                let keep = INTERNAL_BEGIN.len().saturating_sub(1);
                if self.buffer.len() > keep {
                    let emit_len = self.buffer.len() - keep;
                    let emit_idx = clamp_char_boundary(&self.buffer, emit_len);
                    if emit_idx > 0 {
                        output.push_str(&self.buffer[..emit_idx]);
                        self.buffer = self.buffer[emit_idx..].to_string();
                    }
                }
                break;
            }
        }
        output
    }

    fn trim_buffer(&mut self, keep: usize) {
        if self.buffer.len() <= keep {
            return;
        }
        let start_idx = clamp_char_boundary(&self.buffer, self.buffer.len().saturating_sub(keep));
        self.buffer = self.buffer[start_idx..].to_string();
    }

    fn filter_lines(&mut self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        self.line_buffer.push_str(text);
        let mut output = String::new();
        while let Some(pos) = self.line_buffer.find('\n') {
            let line = self.line_buffer[..pos].to_string();
            self.line_buffer = self.line_buffer[pos + 1..].to_string();
            if (self.allow_diagnostics || !is_diagnostic_line(&line))
                && !is_role_label_line(
                    &line,
                    self.assistant_label.as_deref(),
                    self.user_label.as_deref(),
                )
                && !is_workspace_scaffold_line(&line)
                && (self.allow_diagnostics
                    || (!is_tool_list_line(&line)
                        && !is_kv_dump_line(&line)
                        && !is_identity_anchor_line(&line)))
            {
                output.push_str(&line);
                output.push('\n');
            }
        }
        output.push_str(&self.flush_partial_line());
        output
    }

    fn flush_partial_line(&mut self) -> String {
        if self.line_buffer.len() < STREAM_FLUSH_THRESHOLD {
            return String::new();
        }
        let trimmed = self.line_buffer.trim_start();
        if trimmed.is_empty() {
            return String::new();
        }
        if self.should_block_prefix(trimmed) {
            return String::new();
        }
        let flush_len = self.line_buffer.len().saturating_sub(STREAM_FLUSH_TAIL);
        let flush_idx = clamp_char_boundary(&self.line_buffer, flush_len);
        if flush_idx == 0 {
            return String::new();
        }
        let flushed = self.line_buffer[..flush_idx].to_string();
        self.line_buffer = self.line_buffer[flush_idx..].to_string();
        flushed
    }

    fn should_block_prefix(&self, line: &str) -> bool {
        if is_role_label_line(line, self.assistant_label.as_deref(), self.user_label.as_deref()) {
            return true;
        }
        if is_workspace_scaffold_line(line) {
            return true;
        }
        if self.allow_diagnostics {
            return false;
        }
        if is_diagnostic_line_prefix(line) {
            return true;
        }
        if is_tool_list_line(line) || is_kv_dump_line(line) || is_identity_anchor_line(line) {
            return true;
        }
        false
    }
}

fn is_scaffold_heading(line: &str) -> bool {
    let trimmed = line.trim().to_lowercase();
    trimmed.starts_with("next steps") || trimmed.starts_with("proposed response")
}

fn is_role_label_line(line: &str, assistant_name: Option<&str>, user_name: Option<&str>) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    let lower = trimmed.to_lowercase();
    let static_labels = ["user", "assistant", "system", "developer", "ergo", "symbiote"];
    if static_labels
        .iter()
        .any(|label| lower == format!("{}:", label) || lower.starts_with(&format!("{}:", label)))
    {
        return true;
    }
    if let Some(name) = assistant_name {
        let label = name.trim().to_lowercase();
        if !label.is_empty()
            && (lower == format!("{}:", label) || lower.starts_with(&format!("{}:", label)))
        {
            return true;
        }
    }
    if let Some(name) = user_name {
        let label = name.trim().to_lowercase();
        if !label.is_empty()
            && (lower == format!("{}:", label) || lower.starts_with(&format!("{}:", label)))
        {
            return true;
        }
    }
    false
}

fn is_workspace_scaffold_line(line: &str) -> bool {
    let lower = line.trim().to_lowercase();
    lower.starts_with("focus:")
        || lower.starts_with("open_questions:")
        || lower.starts_with("active_hypotheses:")
        || lower.starts_with("current_focus:")
        || lower.starts_with("goal_thread:")
        || lower.starts_with("working_set_topics:")
        || lower.starts_with("next_action:")
        || lower.starts_with("confidence:")
        || lower.starts_with("drift_score:")
        || lower.starts_with("initiative_level:")
        || lower.starts_with("uncertainty_level:")
        || lower.starts_with("last_action_outcome:")
        || lower.starts_with("updated_at:")
}

fn is_identity_anchor_line(line: &str) -> bool {
    let lower = line.to_lowercase();
    lower.contains("name:") && lower.contains("role:") && lower.contains("self_model_hash")
}

fn is_tool_list_line(line: &str) -> bool {
    let lower = line.trim().to_lowercase();
    if lower.starts_with("tools:")
        || lower.starts_with("available tools:")
        || lower.starts_with("tool list:")
    {
        return true;
    }
    if lower.starts_with("- ") {
        let entry = lower.trim_start_matches("- ").trim();
        let known_tools = [
            "run_shell",
            "get_current_time",
            "get_system_logs",
            "get_system_capabilities",
            "get_inner_summary",
            "get_workspace_state",
            "get_rolling_summary",
            "save_context",
            "read_context",
        ];
        if known_tools.iter().any(|tool| entry.starts_with(tool)) {
            return true;
        }
    }
    false
}

fn is_kv_dump_line(line: &str) -> bool {
    let lower = line.trim().to_lowercase();
    lower.starts_with("kv:")
        || lower.starts_with("kv ")
        || lower.starts_with("kv memory")
        || lower.starts_with("kv_memory")
        || lower.starts_with("blackboard:")
        || lower.starts_with("context store")
}

fn is_diagnostic_line(line: &str) -> bool {
    let lower = line.trim().trim_end_matches(':').to_lowercase();
    matches!(
        lower.as_str(),
        "tool manifest"
            | "capability manifest"
            | "kv memory"
            | "self-state"
            | "controller state"
            | "telemetry snapshot"
            | "identity anchor"
            | "workspace snapshot"
            | "inner summary"
            | "rolling summary"
            | "introspection summary"
            | "memory context"
            | "episodic context"
            | "capabilities and limitations"
            | "identity thread"
            | "task context"
            | "gate feedback"
            | "user evidence ids"
            | "tool evidence ids"
            | "symbiote system overview"
            | "system overview"
            | "safety rules"
            | "response style"
            | "working memory"
            | "strategy rationale"
            | "strategy_rationale"
            | "self-model signals"
            | "self_model_signals"
    )
}

fn is_diagnostic_line_prefix(line: &str) -> bool {
    let lower = line.trim().to_lowercase();
    let prefixes = [
        "tool manifest",
        "capability manifest",
        "kv memory",
        "self-state",
        "controller state",
        "telemetry snapshot",
        "identity anchor",
        "workspace snapshot",
        "inner summary",
        "rolling summary",
        "introspection summary",
        "memory context",
        "episodic context",
        "capabilities and limitations",
        "identity thread",
        "task context",
        "gate feedback",
        "user evidence ids",
        "tool evidence ids",
        "symbiote system overview",
        "system overview",
        "safety rules",
        "response style",
        "working memory",
        "strategy rationale",
        "strategy_rationale",
        "self-model signals",
        "self_model_signals",
    ];
    prefixes.iter().any(|p| lower.starts_with(p))
}

fn clamp_char_boundary(input: &str, idx: usize) -> usize {
    if idx >= input.len() {
        return input.len();
    }
    if input.is_char_boundary(idx) {
        return idx;
    }
    let mut i = idx;
    while i > 0 && !input.is_char_boundary(i) {
        i -= 1;
    }
    i
}
use crate::db::Db;
use crate::core::memory::snippets;
use crate::core::memory::writer::EmbeddingConfig;

#[derive(Clone)]
pub struct ModelClient {
    client: Client,
    #[allow(dead_code)]
    db_pool: Pool<Sqlite>,
    #[allow(dead_code)]
    app_handle: AppHandle,
}

#[derive(Clone, Debug, Default)]
struct InjectionContext {
    recalled_info: String,
    episodic_context: String,
    semantic_context: String,
}

const EPISODIC_EVENT_ALLOWLIST: [&str; 13] = [
    "message_received",
    "assistant_response_finalized",
    "memory_write_fact",
    "memory_write_rel",
    "memory_write_self_fact",
    "memory_write_self_rel",
    "memory_conflict_resolution",
    "reminder_triggered",
    "episodic_summary",
    "memory_claim_status",
    "memory_claim_created",
    "entity_created",
    "clarify_resolved",
];

const EPISODIC_CONTEXT_MAX_AGE_DAYS: i64 = 30;
const EPISODIC_CONTEXT_MIN_BELIEF_CONFIDENCE: f32 = 0.6;
const EPISODIC_CONTEXT_LINKED_BELIEF_LIMIT: usize = 5;

fn debug_logging_enabled() -> bool {
    static DEBUG_LOGS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DEBUG_LOGS.get_or_init(|| {
        std::env::var("SYMBIOTE_DEBUG_LOGS")
            .ok()
            .map(|value| {
                let normalized = value.trim().to_lowercase();
                normalized == "1" || normalized == "true" || normalized == "yes"
            })
            .unwrap_or(false)
    })
}

fn hash_text(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

fn strip_working_hypothesis_prefix(text: &str) -> (String, bool) {
    let trimmed = text.trim_start();
    let lower = trimmed.to_lowercase();
    let prefix = "working hypothesis";
    if !lower.starts_with(prefix) {
        return (text.to_string(), false);
    }
    let mut rest = trimmed[prefix.len()..].trim_start();
    for marker in [":", "-", "—", "–"] {
        if let Some(stripped) = rest.strip_prefix(marker) {
            rest = stripped.trim_start();
            break;
        }
    }
    (rest.to_string(), true)
}

fn summarize_messages(messages: &[ChatMessage]) -> (usize, String) {
    let mut combined = String::new();
    let mut total = 0usize;
    for msg in messages {
        total += msg.content.len();
        combined.push_str(msg.role.as_str());
        combined.push(':');
        combined.push_str(&msg.content);
        combined.push('\n');
    }
    (total, hash_text(&combined))
}

fn strict_memory_prompt(base: &str) -> String {
    format!(
        "{}\n\nSTRICT OUTPUT RULES:\n- Output ONLY one <memory>...</memory> block and nothing else.\n- Do not include explanations, headers, or blank prose.\n- If no memory should be written, output nothing at all.\n",
        base
    )
}

fn estimate_prompt_tokens(messages: &[ChatMessage]) -> usize {
    token_estimator::estimate_tokens_for_strings(messages.iter().map(|msg| msg.content.as_str()))
}

fn is_explicit_memory_request(query: &str) -> bool {
    let lowered = query.to_lowercase();
    let triggers = [
        "remember",
        "recall",
        "earlier",
        "previous",
        "last time",
        "before",
        "history",
        "we talked",
        "we discussed",
        "you said",
        "i said",
        "what did you",
        "what did i",
        "as we talked",
        "as discussed",
    ];
    triggers.iter().any(|needle| lowered.contains(needle))
}

fn should_expand_memory(
    intent: &crate::core::memory::types::QueryIntent,
    query: &str,
    reduce_injection: bool,
) -> bool {
    if reduce_injection {
        return false;
    }
    matches!(
        intent,
        crate::core::memory::types::QueryIntent::AskHistory
            | crate::core::memory::types::QueryIntent::AskList
            | crate::core::memory::types::QueryIntent::AskExplain
    ) || is_temporal_query(query)
        || is_explicit_memory_request(query)
}

fn trim_to_tokens(input: &str, max_tokens: usize) -> (String, bool) {
    token_estimator::truncate_to_token_budget(input, max_tokens)
}

impl ModelClient {
    pub fn new(db_pool: Pool<Sqlite>, app_handle: AppHandle) -> Self {
        Self {
            client: Client::new(),
            db_pool,
            app_handle,
        }
    }

    /// B2: URL safety and deterministic normalization
    pub fn normalize_url(url_str: &str) -> Result<(String, bool), String> {
        let trimmed = url_str.trim();
        let mut parsed = Url::parse(trimmed).map_err(|e| format!("Invalid URL: {}", e))?;

        if parsed.scheme() != "http" && parsed.scheme() != "https" {
            return Err("Only http and https schemes are allowed".into());
        }

        let mut path = parsed.path().to_string();
        if path.ends_with('/') {
            path.pop();
        }
        if !path.ends_with("/v1") {
            if path.is_empty() {
                path = "/v1".to_string();
            } else {
                path.push_str("/v1");
            }
        }
        parsed.set_path(&path);

        let is_loopback = match parsed.host_str() {
            Some("localhost") | Some("127.0.0.1") | Some("::1") => true,
            _ => false,
        };

        let mut final_url = parsed.to_string();
        if final_url.ends_with('/') {
            final_url.pop();
        }

        Ok((final_url, is_loopback))
    }

    pub async fn test_connection(&self, base_url: &str, api_key: Option<&str>) -> Result<String, String> {
        let url = format!("{}/models", base_url);
        let mut request = self.client.get(&url);
        if let Some(key) = api_key {
            request = request.bearer_auth(key);
        }

        let response = request.send().await.map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            return Err(format!("Server returned error: {}", response.status()));
        }

        let body: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
        let models = body["data"].as_array().ok_or("Invalid response format: missing data array")?;
        
        if models.is_empty() {
            return Err("No models available on this server".into());
        }

        let first_model = models[0]["id"].as_str().ok_or("Invalid model ID format")?;
        Ok(first_model.to_string())
    }

    // Non-streaming chat completion
    pub async fn chat(&self, base_url: &str, api_key: Option<&str>, request: &ChatCompletionRequest) -> Result<(String, Option<Vec<crate::models::ToolCall>>), String> {
        // MEMORY INJECTION
        let mut final_request = request.clone();
        let allow_side_effects = request.skip_memory != Some(true);
        let mut should_reinforce_on_use = false;
        if request.skip_injection != Some(true) {
            let mut query = self.extract_user_query(&final_request.messages);
            if query.as_deref().map(|q| q.trim().is_empty()).unwrap_or(true) {
                let fallback = self
                    .resolve_user_message_for_memory(request.run_id.as_deref(), &final_request.messages)
                    .await;
                if !fallback.trim().is_empty() {
                    query = Some(fallback);
                }
            }
            if let Some(query) = query {
                if allow_side_effects && is_confirmation_request(&query) {
                    let client = self.clone();
                    let run_id = final_request.run_id.clone();
                    tokio::spawn(async move {
                        client.reinforce_last_recalled_on_confirmation(run_id.as_deref()).await;
                    });
                }
                let repair_mode = is_repair_request(&query);
                let reduce_injection = self
                    .is_clarify_mode(&final_request.messages, final_request.run_id.as_deref())
                    .await
                    || repair_mode;
                let base_prompt_tokens = estimate_prompt_tokens(&final_request.messages);
                let force_expand = final_request.memory_expand.unwrap_or(false);
                let injection = self
                    .build_injection_context(
                        &query,
                        final_request.run_id.as_deref(),
                        reduce_injection,
                        repair_mode,
                        allow_side_effects,
                        base_prompt_tokens,
                        force_expand,
                    )
                    .await;
                should_reinforce_on_use = !injection.semantic_context.trim().is_empty();
                self.inject_context_blocks(
                    &mut final_request.messages,
                    &injection.recalled_info,
                    &injection.episodic_context,
                );

                let _ = self.append_pending_clarify(&mut final_request.messages).await;
            }
        }

        self.log_prompt_messages(&final_request, "chat");
        self.log_final_request(&final_request);

        let (raw_content, tool_calls) = self
            .execute_chat_request(base_url, api_key, &final_request)
            .await?;
        let (content_no_tags, tag_set) = crate::core::memory::inject_context::strip_system_tags(&raw_content);
        let memory_triggered = if tag_set.resolve {
            true
        } else if tag_set.clarify {
            false
        } else {
            tag_set.memory
        };
        if memory_triggered {
            let run_id = request.run_id.as_deref().unwrap_or("none");
            eprintln!("[MemoryPass] Trigger detected (<<MEMORY>>) run_id={}", run_id);
        }
        if allow_side_effects && should_reinforce_on_use && !tag_set.clarify {
            let client = self.clone();
            let run_id = request.run_id.clone();
            tokio::spawn(async move {
                client.reinforce_last_recalled_on_use(run_id.as_deref()).await;
            });
        }
        let user_message = self
            .resolve_user_message_for_memory(request.run_id.as_deref(), &final_request.messages)
            .await;
        let memory_gating_enabled = self.memory_evidence_gating_enabled().await;
        let assistant_for_memory = {
            let cleaned = crate::core::memory::inject_context::strip_memory_blocks(&content_no_tags);
            let cleaned = crate::core::reminder_blocks::strip_reminder_blocks(&cleaned);
            cleaned
        };

        let skip_side_effects = self.should_skip_side_effects(request.run_id.as_deref()).await;
        let reminder_specs = if skip_side_effects || request.skip_reminders == Some(true) {
            Vec::new()
        } else {
            reminder_blocks::parse_reminder_blocks(&content_no_tags)
        };

        if !skip_side_effects && (request.skip_reminders != Some(true) || request.skip_memory != Some(true)) {
            let client = self.clone();
            let reminder_specs_bg = reminder_specs.clone();
            let run_id_bg = request.run_id.clone();
            let skip_reminders = request.skip_reminders == Some(true);
            let skip_memory = request.skip_memory == Some(true);
            let memory_triggered_bg = memory_triggered;
            let include_clarify_history_bg = tag_set.resolve;
            let user_message_bg = user_message.clone();
            let assistant_message_bg = assistant_for_memory.clone();
            let memory_gating_enabled_bg = memory_gating_enabled;
            let base_url_bg = base_url.to_string();
            let api_key_bg = api_key.map(|key| key.to_string());
            let model_bg = final_request.model.clone();
            tokio::spawn(async move {
                if client.should_skip_side_effects(run_id_bg.as_deref()).await {
                    return;
                }

                if !skip_reminders {
                    for spec in &reminder_specs_bg {
                        match reminder_blocks::create_reminder(&client.db_pool, &spec.content, &spec.due_in, &spec.reminder_type).await {
                            Ok(id) => {
                                let _ = system_log::log_event(
                                    &client.db_pool,
                                    Some(&client.app_handle),
                                    "info",
                                    "memory",
                                    run_id_bg.as_deref(),
                                    None,
                                    serde_json::json!({
                                        "event": "reminder_created",
                                        "reminder_id": id,
                                        "due_in": spec.due_in,
                                        "reminder_type": spec.reminder_type,
                                    }),
                                )
                                .await;
                            }
                            Err(e) => {
                                eprintln!("[Reminder] Create error: {}", e);
                                let _ = system_log::log_event(
                                    &client.db_pool,
                                    Some(&client.app_handle),
                                    "error",
                                    "memory",
                                    run_id_bg.as_deref(),
                                    None,
                                    serde_json::json!({
                                        "event": "reminder_create_error",
                                        "error": e.to_string(),
                                    }),
                                )
                                .await;
                                let _ = client.app_handle.emit("params", format!("REMINDER_ERROR: {}", e));
                            }
                        }
                    }
                }

                if skip_memory {
                    if memory_triggered_bg {
                        eprintln!(
                            "[MemoryPass] Skipped: skip_memory=true (run_id={})",
                            run_id_bg.as_deref().unwrap_or("none")
                        );
                    }
                    let _ = system_log::log_event(
                        &client.db_pool,
                        Some(&client.app_handle),
                        "info",
                        "memory",
                        run_id_bg.as_deref(),
                        None,
                        serde_json::json!({
                            "event": "memory_pass_skipped",
                            "reason": "skip_memory",
                            "triggered": memory_triggered_bg,
                        }),
                    )
                    .await;
                    return;
                }

                if !memory_triggered_bg {
                    return;
                }

                let gate = client.should_allow_memory_pass(run_id_bg.as_deref()).await;
                if !gate.allowed {
                    eprintln!(
                        "[MemoryPass] Skipped: {} (run_id={})",
                        gate.reason,
                        run_id_bg.as_deref().unwrap_or("none")
                    );
                    let _ = system_log::log_event(
                        &client.db_pool,
                        Some(&client.app_handle),
                        "info",
                        "memory",
                        run_id_bg.as_deref(),
                        None,
                        serde_json::json!({
                            "event": "memory_pass_skipped",
                            "reason": gate.reason,
                        }),
                    )
                    .await;
                    return;
                }

                let user_empty = user_message_bg.trim().is_empty();
                let assistant_empty = assistant_message_bg.trim().is_empty();
                let assistant_hash = hash_payload(&assistant_message_bg);
                if user_empty || assistant_empty {
                    eprintln!(
                        "[MemoryPass] Skipped: empty user/assistant (run_id={}, user_len={}, assistant_len={})",
                        run_id_bg.as_deref().unwrap_or("none"),
                        user_message_bg.trim().len(),
                        assistant_message_bg.trim().len()
                    );
                    let _ = system_log::log_event(
                        &client.db_pool,
                        Some(&client.app_handle),
                        "info",
                        "memory",
                        run_id_bg.as_deref(),
                        None,
                        serde_json::json!({
                            "event": if user_empty { "memory_pass_blocked_empty_user" } else { "memory_pass_blocked_empty_assistant" },
                            "reason": if user_empty { "empty_user_message" } else { "empty_assistant_message" },
                            "user_len": user_message_bg.trim().len(),
                            "assistant_len": assistant_message_bg.trim().len(),
                            "assistant_hash": assistant_hash,
                            "assistant_source": "generated",
                            "evidence_gate": memory_gating_enabled_bg,
                        }),
                    )
                    .await;
                    return;
                }

                let _ = system_log::log_event(
                    &client.db_pool,
                    Some(&client.app_handle),
                    "info",
                    "memory",
                    run_id_bg.as_deref(),
                    None,
                    serde_json::json!({
                        "event": "memory_pass_start",
                        "user_len": user_message_bg.trim().len(),
                        "assistant_len": assistant_message_bg.trim().len(),
                        "assistant_hash": assistant_hash,
                        "assistant_source": "generated",
                        "evidence_gate": memory_gating_enabled_bg,
                    }),
                )
                .await;

                    let prompt_set = match prompt_loader::get_prompts() {
                        Ok(prompts) => prompts,
                        Err(e) => {
                            eprintln!("[MemoryPass] Prompt load error: {}", e);
                            let _ = system_log::log_event(
                                &client.db_pool,
                                Some(&client.app_handle),
                                "error",
                                "memory",
                                run_id_bg.as_deref(),
                                None,
                                serde_json::json!({
                                    "event": "memory_pass_error",
                                    "stage": "load_prompts",
                                    "error": e.to_string(),
                                }),
                            )
                            .await;
                            return;
                        }
                    };
                    let session_id = client
                        .resolve_conversation_id(run_id_bg.as_deref())
                        .await
                        .unwrap_or_else(|| "default".to_string());
                    let injection = client
                        .build_injection_context(
                            &user_message_bg,
                            run_id_bg.as_deref(),
                            false,
                            false,
                            true,
                            user_message_bg.len(),
                            false,
                        )
                        .await;
                    let known_handles = client.fetch_bound_handles_for_session(&session_id).await;
                    let (user_name, assistant_name) = client.fetch_display_names().await;
                    let clarify_context = if include_clarify_history_bg {
                        client
                            .fetch_clarify_context(run_id_bg.as_deref(), &assistant_message_bg, 3)
                            .await
                    } else {
                        String::new()
                    };
                    let clarify_context_opt = if clarify_context.trim().is_empty() {
                        None
                    } else {
                        Some(clarify_context.as_str())
                    };
                    let candidate_block = candidate::format_candidates_for_prompt(
                        &candidate::extract_candidates(&user_message_bg, &assistant_message_bg),
                    );
                    let payload = build_memory_pass_payload(
                        &user_message_bg,
                        &assistant_message_bg,
                        user_name.as_deref(),
                        assistant_name.as_deref(),
                        &injection.semantic_context,
                        clarify_context_opt,
                        is_repair_request(&user_message_bg),
                        &known_handles,
                        Some(candidate_block.as_str()),
                    );
                    let memory_request = ChatCompletionRequest {
                        model: model_bg,
                        messages: vec![
                            ChatMessage {
                                role: "system".to_string(),
                                content: prompt_set.memory_control_prompt,
                            },
                            ChatMessage {
                                role: "user".to_string(),
                                content: payload,
                            },
                        ],
                        stream: false,
                        temperature: None,
                        top_p: None,
                        max_tokens: None,
                        response_format: None,
                        tools: None,
                        tool_choice: None,
                        enable_thinking: None,
                        prefill: None,
                        skip_injection: Some(true),
                        skip_memory: Some(true),
                        skip_reminders: Some(true),
                        memory_expand: None,
                        allow_diagnostics: Some(false),
                        json_strict: None,
                        skip_sanitization: Some(true),
                        run_id: None,
                        request_label: Some("memory_pass_bg".to_string()),
                    };
                    let memory_raw = match client
                        .execute_chat_request(&base_url_bg, api_key_bg.as_deref(), &memory_request)
                        .await
                    {
                        Ok((text, _)) => text,
                        Err(e) => {
                            eprintln!("[MemoryPass] LLM error: {}", e);
                            let _ = system_log::log_event(
                                &client.db_pool,
                                Some(&client.app_handle),
                                "error",
                                "memory",
                                run_id_bg.as_deref(),
                                None,
                                serde_json::json!({
                                    "event": "memory_pass_error",
                                    "stage": "llm_request",
                                    "error": e.to_string(),
                                }),
                            )
                            .await;
                            return;
                        }
                    };
                    eprintln!(
                        "[MemoryPass] Raw response received (run_id={}, chars={})",
                        run_id_bg.as_deref().unwrap_or("none"),
                        memory_raw.chars().count()
                    );

                    client
                        .maybe_capture_memory_raw(
                            run_id_bg.as_deref().unwrap_or("none"),
                            "memory_pass_bg",
                            &memory_raw,
                        )
                        .await;
                    let trimmed = memory_raw.trim();
                    if trimmed.is_empty() {
                        let _ = system_log::log_event(
                            &client.db_pool,
                            Some(&client.app_handle),
                            "warn",
                            "memory",
                            run_id_bg.as_deref(),
                            None,
                            serde_json::json!({
                                "event": "memory_pass_empty_response",
                            }),
                        )
                        .await;
                        return;
                    }

                    let block = match crate::core::memory::inject_context::extract_single_memory_block_strict(&memory_raw) {
                        Some(block) => block,
                        None => {
                            if let Some(block) = Self::extract_memory_block_lenient(&memory_raw) {
                                let _ = system_log::log_event(
                                    &client.db_pool,
                                    Some(&client.app_handle),
                                    "info",
                                    "memory",
                                    run_id_bg.as_deref(),
                                    None,
                                    serde_json::json!({
                                        "event": "memory_pass_fallback_block",
                                        "reason": "lenient_extract",
                                        "request_label": "memory_pass_bg",
                                        "schema": "memory_dsl_v1",
                                    }),
                                )
                                .await;
                                block
                            } else {
                                eprintln!("[MemoryPass] Output not a single <memory> block, skipping.");
                                let _ = system_log::log_event(
                                    &client.db_pool,
                                    Some(&client.app_handle),
                                    "error",
                                    "memory",
                                    run_id_bg.as_deref(),
                                    None,
                                    serde_json::json!({
                                        "event": "memory_pass_invalid_output",
                                        "reason": "missing_memory_block",
                                        "request_label": "memory_pass_bg",
                                        "schema": "memory_dsl_v1",
                                        "raw_len": memory_raw.len(),
                                        "raw_snippet": memory_raw.chars().take(200).collect::<String>(),
                                    }),
                                )
                                .await;
                                return;
                            }
                        }
                    };

                    let api = crate::core::memory::api::MemoryApi::new(
                        client.db_pool.clone(),
                        Some(Arc::new(client.clone())),
                        session_id,
                    )
                    .await;

                    let res = api
                        .parse_and_compile(
                            &block,
                            crate::core::memory::types::Scope::Global,
                            crate::core::memory::types::SourceType::User,
                            None,
                            chrono::Utc::now(),
                        )
                        .await;

                    let error_count = res.errors.len();
                    let _ = system_log::log_event(
                        &client.db_pool,
                        Some(&client.app_handle),
                        if error_count > 0 { "warn" } else { "info" },
                        "memory",
                        run_id_bg.as_deref(),
                        None,
                        serde_json::json!({
                            "event": "memory_pass_result",
                            "written_ids": res.written_ids.len(),
                            "conflict_ids": res.conflict_ids.len(),
                            "pending_writes": res.pending_writes.len(),
                            "pending_clarify": res.pending_clarify.is_some(),
                            "claim_ids": res.claim_ids.len(),
                            "errors": error_count,
                        }),
                    )
                    .await;

                    if !res.errors.is_empty() {
                        let errors = res.errors.clone();
                        eprintln!("[MemoryPass] Compilation errors: {:?}", errors);
                        let now = chrono::Utc::now().to_rfc3339();
                        let error_text = errors.join(" | ");
                        let _ = sqlx::query("INSERT INTO kv_store (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP")
                            .bind("memory_last_error_at")
                            .bind(&now)
                            .execute(&client.db_pool)
                            .await;
                        let _ = sqlx::query("INSERT INTO kv_store (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP")
                            .bind("memory_last_error")
                            .bind(&error_text)
                            .execute(&client.db_pool)
                            .await;
                        let _ = client.app_handle.emit(
                            "memory_error",
                            serde_json::json!({ "errors": errors, "timestamp": now }),
                        );
                    }

                    if res.pending_clarify.is_some() {
                        eprintln!("[MemoryPass] pending_clarify returned; skipping.");
                    }

                    if !res.conflict_ids.is_empty() {
                        let _ = client.app_handle.emit("memory_conflict", res.conflict_ids.clone());
                    }
            });
        }
        
        let mut final_content = crate::core::memory::inject_context::strip_memory_blocks(&content_no_tags);
        if !reminder_specs.is_empty() {
            final_content = reminder_blocks::strip_reminder_blocks(&final_content);
            final_content = reminder_blocks::append_reminder_markers(&final_content, &reminder_specs);
        }

        Ok((final_content, tool_calls))
    }

    pub async fn warm_keep(&self, base_url: &str, api_key: Option<&str>, model: &str, reason: &str) -> Result<(), String> {
        let request = ChatCompletionRequest {
            model: model.to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "ping".to_string(),
            }],
            stream: false,
            temperature: Some(0.0),
            top_p: Some(1.0),
            max_tokens: Some(1),
            response_format: None,
            tools: None,
            tool_choice: Some("none".to_string()),
            enable_thinking: None,
            prefill: None,
            skip_injection: Some(true),
            skip_memory: Some(true),
            skip_reminders: Some(true),
            memory_expand: None,
            allow_diagnostics: Some(false),
            json_strict: None,
            skip_sanitization: None,
            run_id: None,
            request_label: Some("warm_keep".to_string()),
        };
        let started = Instant::now();
        let result = self.execute_chat_request(base_url, api_key, &request).await;
        let duration_ms = started.elapsed().as_millis() as i64;
        let status = if result.is_ok() { "ok" } else { "error" };
        let _ = system_log::log_event(
            &self.db_pool,
            Some(&self.app_handle),
            "info",
            "model",
            None,
            None,
            serde_json::json!({
                "event": "model_warm_keep",
                "model": model,
                "duration_ms": duration_ms,
                "status": status,
                "reason": reason,
            }),
        )
        .await;
        result.map(|_| ())
    }

    pub async fn warm_keep_with_settings(&self, settings: &Settings, reason: &str) -> Result<(), String> {
        let model = settings
            .active_model_id
            .clone()
            .unwrap_or_else(|| "default".to_string());
        self.warm_keep(&settings.api_base_url, settings.api_key.as_deref(), &model, reason)
            .await
    }

    /// Non-streaming chat completion with metadata for kernel orchestration
    pub async fn chat_with_meta(
        &self,
        base_url: &str,
        api_key: Option<&str>,
        request: &ChatCompletionRequest,
    ) -> Result<ChatResponseMeta, String> {
        let mut final_request = request.clone();
        let allow_side_effects = request.skip_memory != Some(true);
        let mut should_reinforce_on_use = false;
        if request.skip_injection != Some(true) {
            let mut query = self.extract_user_query(&final_request.messages);
            if query.as_deref().map(|q| q.trim().is_empty()).unwrap_or(true) {
                let fallback = self
                    .resolve_user_message_for_memory(request.run_id.as_deref(), &final_request.messages)
                    .await;
                if !fallback.trim().is_empty() {
                    query = Some(fallback);
                }
            }
            if let Some(query) = query {
                if allow_side_effects && is_confirmation_request(&query) {
                    let client = self.clone();
                    let run_id = final_request.run_id.clone();
                    tokio::spawn(async move {
                        client.reinforce_last_recalled_on_confirmation(run_id.as_deref()).await;
                    });
                }
                let repair_mode = is_repair_request(&query);
                let reduce_injection = self
                    .is_clarify_mode(&final_request.messages, final_request.run_id.as_deref())
                    .await
                    || repair_mode;
                let base_prompt_tokens = estimate_prompt_tokens(&final_request.messages);
                let force_expand = final_request.memory_expand.unwrap_or(false);
                let injection = self
                    .build_injection_context(
                        &query,
                        final_request.run_id.as_deref(),
                        reduce_injection,
                        repair_mode,
                        allow_side_effects,
                        base_prompt_tokens,
                        force_expand,
                    )
                    .await;
                should_reinforce_on_use = !injection.semantic_context.trim().is_empty();
                self.inject_context_blocks(
                    &mut final_request.messages,
                    &injection.recalled_info,
                    &injection.episodic_context,
                );

                let _ = self.append_pending_clarify(&mut final_request.messages).await;
            }
        }

        self.log_prompt_messages(&final_request, "chat_meta");
        self.log_final_request(&final_request);

        let (raw_content, tool_calls) = self
            .execute_chat_request(base_url, api_key, &final_request)
            .await?;

        let (content_no_tags, tag_set) =
            crate::core::memory::inject_context::strip_system_tags(&raw_content);

        if allow_side_effects && should_reinforce_on_use && !tag_set.clarify {
            let client = self.clone();
            let run_id = request.run_id.clone();
            tokio::spawn(async move {
                client.reinforce_last_recalled_on_use(run_id.as_deref()).await;
            });
        }

        let mut content = crate::core::memory::inject_context::strip_memory_blocks(&content_no_tags);
        let reminder_specs = if request.skip_reminders == Some(true) {
            Vec::new()
        } else {
            reminder_blocks::parse_reminder_blocks(&content_no_tags)
        };
        if !reminder_specs.is_empty() {
            content = reminder_blocks::strip_reminder_blocks(&content);
            content = reminder_blocks::append_reminder_markers(&content, &reminder_specs);
        }

        Ok(ChatResponseMeta {
            raw_content,
            content_no_tags,
            content,
            tool_calls,
            tag_set,
        })
    }

    /// Streaming chat completion with metadata for kernel orchestration
    pub async fn chat_with_meta_stream(
        &self,
        base_url: &str,
        api_key: Option<&str>,
        request: &ChatCompletionRequest,
    ) -> Result<ChatResponseMeta, String> {
        let mut final_request = request.clone();
        final_request.stream = true;
        let allow_side_effects = request.skip_memory != Some(true);
        let mut should_reinforce_on_use = false;
        if request.skip_injection != Some(true) {
            let mut query = self.extract_user_query(&final_request.messages);
            if query.as_deref().map(|q| q.trim().is_empty()).unwrap_or(true) {
                let fallback = self
                    .resolve_user_message_for_memory(request.run_id.as_deref(), &final_request.messages)
                    .await;
                if !fallback.trim().is_empty() {
                    query = Some(fallback);
                }
            }
            if let Some(query) = query {
                if allow_side_effects && is_confirmation_request(&query) {
                    let client = self.clone();
                    let run_id = final_request.run_id.clone();
                    tokio::spawn(async move {
                        client.reinforce_last_recalled_on_confirmation(run_id.as_deref()).await;
                    });
                }
                let repair_mode = is_repair_request(&query);
                let reduce_injection = self
                    .is_clarify_mode(&final_request.messages, final_request.run_id.as_deref())
                    .await
                    || repair_mode;
                let base_prompt_tokens = estimate_prompt_tokens(&final_request.messages);
                let force_expand = final_request.memory_expand.unwrap_or(false);
                let injection = self
                    .build_injection_context(
                        &query,
                        final_request.run_id.as_deref(),
                        reduce_injection,
                        repair_mode,
                        allow_side_effects,
                        base_prompt_tokens,
                        force_expand,
                    )
                    .await;
                should_reinforce_on_use = !injection.semantic_context.trim().is_empty();
                self.inject_context_blocks(
                    &mut final_request.messages,
                    &injection.recalled_info,
                    &injection.episodic_context,
                );

                let _ = self.append_pending_clarify(&mut final_request.messages).await;
            }
        }

        let (empty_retry_max, empty_retry_timeout_ms) = self.empty_response_retry_config().await;
        let mut empty_retry_count: usize = 0;
        let mut empty_payload: Option<(String, Option<Vec<crate::models::ToolCall>>)> = None;
        let (raw_content, tool_calls) = loop {
            let (content, calls) = self
                .stream_chat(self.app_handle.clone(), base_url, api_key, &final_request)
                .await?;
            let has_tool_calls = calls.as_ref().map(|c| !c.is_empty()).unwrap_or(false);
            if content.trim().is_empty() && !has_tool_calls {
                if empty_retry_count < empty_retry_max {
                    empty_retry_count += 1;
                    let _ = system_log::log_event(
                        &self.db_pool,
                        Some(&self.app_handle),
                        "warn",
                        "model",
                        final_request.run_id.as_deref(),
                        None,
                        serde_json::json!({
                            "event": "response_empty_retry",
                            "attempt": empty_retry_count,
                            "max": empty_retry_max,
                            "request_label": final_request.request_label.clone(),
                            "stream": true,
                        }),
                    )
                    .await;
                    if empty_retry_timeout_ms > 0 {
                        tokio::time::sleep(Duration::from_millis(empty_retry_timeout_ms)).await;
                    }
                    continue;
                }
                empty_payload = Some((content, calls));
                break ("".to_string(), None);
            }
            break (content, calls);
        };

        if empty_payload.is_some() {
            let fallback_enabled = self.response_fallback_enabled().await;
            if fallback_enabled {
                let _ = system_log::log_event(
                    &self.db_pool,
                    Some(&self.app_handle),
                    "warn",
                    "model",
                    final_request.run_id.as_deref(),
                    None,
                    serde_json::json!({
                        "event": "response_empty_fallback",
                        "attempt": empty_retry_count,
                        "max": empty_retry_max,
                        "request_label": final_request.request_label.clone(),
                        "stream": true,
                    }),
                )
                .await;
                let mut fallback_request = final_request.clone();
                fallback_request.stream = false;
                if let Ok(fallback) = self
                    .chat_with_meta(base_url, api_key, &fallback_request)
                    .await
                {
                    let has_tool_calls = fallback.tool_calls.as_ref().map(|c| !c.is_empty()).unwrap_or(false);
                    if !fallback.content.trim().is_empty() || has_tool_calls {
                        let _ = system_log::log_event(
                            &self.db_pool,
                            Some(&self.app_handle),
                            "info",
                            "model",
                            final_request.run_id.as_deref(),
                            None,
                            serde_json::json!({
                                "event": "response_empty_recovered",
                                "request_label": final_request.request_label.clone(),
                                "stream": false,
                            }),
                        )
                        .await;
                        return Ok(fallback);
                    }
                }
            }

            let _ = system_log::log_event(
                &self.db_pool,
                Some(&self.app_handle),
                "error",
                "model",
                final_request.run_id.as_deref(),
                None,
                serde_json::json!({
                    "event": "response_empty_error",
                    "attempt": empty_retry_count,
                    "max": empty_retry_max,
                    "request_label": final_request.request_label.clone(),
                    "stream": true,
                }),
            )
            .await;
            self.emit_empty_response_error(&final_request, empty_retry_count, empty_retry_max)
                .await;
            return Err("EMPTY_RESPONSE".to_string());
        }

        let (content_no_tags, tag_set) =
            crate::core::memory::inject_context::strip_system_tags(&raw_content);

        if allow_side_effects && should_reinforce_on_use && !tag_set.clarify {
            let client = self.clone();
            let run_id = request.run_id.clone();
            tokio::spawn(async move {
                client.reinforce_last_recalled_on_use(run_id.as_deref()).await;
            });
        }

        let mut content = crate::core::memory::inject_context::strip_memory_blocks(&content_no_tags);
        let reminder_specs = if request.skip_reminders == Some(true) {
            Vec::new()
        } else {
            reminder_blocks::parse_reminder_blocks(&content_no_tags)
        };
        if !reminder_specs.is_empty() {
            content = reminder_blocks::strip_reminder_blocks(&content);
            content = reminder_blocks::append_reminder_markers(&content, &reminder_specs);
        }

        Ok(ChatResponseMeta {
            raw_content,
            content_no_tags,
            content,
            tool_calls,
            tag_set,
        })
    }

    pub async fn run_memory_pass(
        &self,
        base_url: &str,
        api_key: Option<&str>,
        model: &str,
        run_id: Option<&str>,
        user_message: &str,
        assistant_message: &str,
        include_clarify_history: bool,
    ) -> MemoryPassResult {
        let mut result = MemoryPassResult::default();
        let memory_gating_enabled = self.memory_evidence_gating_enabled().await;
        let mut assistant_message_for_memory = assistant_message.to_string();
        let mut assistant_source = "provided";
        let memory_pass_id = Uuid::new_v4().to_string();
        let run_id = match run_id {
            Some(id) if !id.trim().is_empty() => id,
            _ => {
                result.error = Some("missing_run_id".to_string());
                return result;
            }
        };
        if let Ok(Some(stored_content)) = sqlx::query_scalar::<_, String>(
            "SELECT content FROM messages WHERE run_id = ? AND role = 'assistant' ORDER BY datetime(created_at) DESC LIMIT 1",
        )
        .bind(run_id)
        .fetch_optional(&self.db_pool)
        .await
        {
            let (stored_no_tags, _) =
                crate::core::memory::inject_context::strip_system_tags(&stored_content);
            let mut stored_clean =
                crate::core::memory::inject_context::strip_memory_blocks(&stored_no_tags);
            stored_clean = reminder_blocks::strip_reminder_blocks(&stored_clean);
            let stored_clean = stored_clean.trim();
            if !stored_clean.is_empty() {
                let stored_hash = hash_payload(stored_clean);
                let provided_hash = hash_payload(&assistant_message_for_memory);
                if stored_clean.len() != assistant_message_for_memory.len()
                    || stored_hash != provided_hash
                {
                    let _ = system_log::log_event(
                        &self.db_pool,
                        Some(&self.app_handle),
                        "warn",
                        "memory",
                        Some(run_id),
                        Some(&memory_pass_id),
                        serde_json::json!({
                            "event": "memory_pass_content_mismatch",
                            "memory_pass_id": memory_pass_id.clone(),
                            "input_len": assistant_message_for_memory.len(),
                            "input_hash": provided_hash,
                            "stored_len": stored_clean.len(),
                            "stored_hash": stored_hash,
                        }),
                    )
                    .await;
                }
                assistant_message_for_memory = stored_clean.to_string();
                assistant_source = "stored";
            }
        }
        let assistant_hash = hash_payload(&assistant_message_for_memory);
        let _ = system_log::log_event(
            &self.db_pool,
            Some(&self.app_handle),
            "info",
            "memory",
            Some(run_id),
            Some(&memory_pass_id),
            serde_json::json!({
                "event": "memory_pass_start",
                "memory_pass_id": memory_pass_id.clone(),
                "user_len": user_message.len(),
                "assistant_len": assistant_message_for_memory.len(),
                "assistant_hash": assistant_hash,
                "assistant_source": assistant_source,
                "include_clarify_history": include_clarify_history,
                "evidence_gate": memory_gating_enabled,
            }),
        )
        .await;

        let gate = self.should_allow_memory_pass(Some(run_id)).await;
        if !gate.allowed {
            let _ = system_log::log_event(
                &self.db_pool,
                Some(&self.app_handle),
                "info",
                "memory",
                Some(run_id),
                Some(&memory_pass_id),
                serde_json::json!({
                    "event": "memory_pass_skipped",
                    "reason": gate.reason,
                    "memory_pass_id": memory_pass_id.clone(),
                }),
            )
            .await;
            result.error = Some(gate.reason.to_string());
            return result;
        }

        let db = Db { pool: self.db_pool.clone() };
        let token_ok = db.consume_memory_pass_token(run_id).await.unwrap_or(false);
        if !token_ok {
            let _ = system_log::log_event(
                &self.db_pool,
                Some(&self.app_handle),
                "warn",
                "memory",
                Some(run_id),
                Some(&memory_pass_id),
                serde_json::json!({
                    "event": "memory_pass_skipped",
                    "reason": "token_missing",
                    "memory_pass_id": memory_pass_id.clone(),
                }),
            )
            .await;
            result.error = Some("token_missing".to_string());
            return result;
        }

        let user_empty = user_message.trim().is_empty();
        let assistant_empty = assistant_message_for_memory.trim().is_empty();
        if user_empty || assistant_empty {
            let event = if user_empty {
                "memory_pass_blocked_empty_user"
            } else {
                "memory_pass_blocked_empty_assistant"
            };
            let reason = if user_empty {
                "empty_user_message"
            } else {
                "empty_assistant_message"
            };
            let _ = system_log::log_event(
                &self.db_pool,
                Some(&self.app_handle),
                "info",
                "memory",
                Some(run_id),
                Some(&memory_pass_id),
                serde_json::json!({
                    "event": event,
                    "reason": reason,
                    "memory_pass_id": memory_pass_id.clone(),
                    "user_len": user_message.trim().len(),
                    "assistant_len": assistant_message_for_memory.trim().len(),
                    "assistant_hash": assistant_hash,
                    "evidence_gate": memory_gating_enabled,
                }),
            )
            .await;
            result.error = Some(reason.to_string());
            return result;
        }

        let prompt_set = match prompt_loader::get_prompts() {
            Ok(prompts) => prompts,
            Err(e) => {
                let error_msg = format!("Prompt load error: {}", e);
                let _ = system_log::log_event(
                    &self.db_pool,
                    Some(&self.app_handle),
                    "error",
                    "memory",
                    Some(run_id),
                    Some(&memory_pass_id),
                    serde_json::json!({
                        "event": "memory_pass_error",
                        "stage": "load_prompts",
                        "error": error_msg,
                        "memory_pass_id": memory_pass_id.clone(),
                    }),
                )
                .await;
                result.error = Some(error_msg);
                return result;
            }
        };

        let _ = system_log::log_event(
            &self.db_pool,
            Some(&self.app_handle),
            "info",
            "memory",
            Some(run_id),
            Some(&memory_pass_id),
            serde_json::json!({
                "event": "memory_pass_prompt",
                "memory_pass_id": memory_pass_id.clone(),
                "model": model,
                "memory_prompt_hash": prompt_set.memory_hash,
                "memory_prompt_source": prompt_set.source,
                "memory_prompt_len": prompt_set.memory_control_prompt.len(),
            }),
        )
        .await;

        let session_id = self
            .resolve_conversation_id(Some(run_id))
            .await
            .unwrap_or_else(|| "default".to_string());
        let injection = self
            .build_injection_context(
                user_message,
                Some(run_id),
                false,
                false,
                true,
                user_message.len(),
                false,
            )
            .await;
        let known_handles = self.fetch_bound_handles_for_session(&session_id).await;
        let (user_name, assistant_name) = self.fetch_display_names().await;
        let clarify_context = if include_clarify_history {
            self.fetch_clarify_context(Some(run_id), &assistant_message_for_memory, 3).await
        } else {
            String::new()
        };
        let user_context = self
            .fetch_recent_user_context(Some(run_id), user_message, 4)
            .await;
        let mut context_parts: Vec<String> = Vec::new();
        if !clarify_context.trim().is_empty() {
            context_parts.push(format!(
                "Clarification history:\n{}",
                clarify_context.trim()
            ));
        }
        if !user_context.trim().is_empty() {
            context_parts.push(format!(
                "Recent user context (verbatim):\n{}",
                user_context.trim()
            ));
        }
        let combined_context = context_parts.join("\n\n");
        let clarify_context_opt = if combined_context.trim().is_empty() {
            None
        } else {
            Some(combined_context.as_str())
        };

        let decision_candidates = candidate::extract_candidates(user_message, &assistant_message_for_memory);
        let candidate_block = candidate::format_candidates_for_prompt(&decision_candidates);
        let payload = build_memory_pass_payload(
            user_message,
            &assistant_message_for_memory,
            user_name.as_deref(),
            assistant_name.as_deref(),
            &injection.semantic_context,
            clarify_context_opt,
            is_repair_request(user_message),
            &known_handles,
            Some(candidate_block.as_str()),
        );
        let minimal_user = Self::trim_tail_chars(user_message, 1200);
        let minimal_assistant = Self::trim_tail_chars(&assistant_message_for_memory, 1200);
        let minimal_payload = Self::build_memory_pass_payload_minimal(
            &minimal_user,
            &minimal_assistant,
            user_name.as_deref(),
            assistant_name.as_deref(),
        );
        let user_hash = hash_payload(user_message.trim());
        let assistant_payload_hash = hash_payload(&assistant_message_for_memory);
        let payload_hash = hash_payload(&payload);
        let _ = system_log::log_event(
            &self.db_pool,
            Some(&self.app_handle),
            "info",
            "memory",
            Some(run_id),
            Some(&memory_pass_id),
            serde_json::json!({
                "event": "memory_pass_payload",
                "memory_pass_id": memory_pass_id.clone(),
                "user_len": user_message.trim().len(),
                "assistant_len": assistant_message_for_memory.len(),
                "user_hash": user_hash,
                "assistant_hash": assistant_payload_hash,
                "payload_len": payload.len(),
                "payload_hash": payload_hash,
            }),
        )
        .await;

        let fact_count = decision_candidates
            .iter()
            .filter(|c| matches!(c.kind, candidate::CandidateKind::Fact))
            .count();
        let rel_count = decision_candidates
            .iter()
            .filter(|c| matches!(c.kind, candidate::CandidateKind::Relation))
            .count();
        let candidate_summary: Vec<serde_json::Value> = decision_candidates
            .iter()
            .take(5)
            .map(|c| {
                let kind = match c.kind {
                    candidate::CandidateKind::Fact => "fact",
                    candidate::CandidateKind::Relation => "relation",
                };
                serde_json::json!({
                    "kind": kind,
                    "key": c.key,
                    "value": c.value,
                    "rel_type": c.rel_type,
                    "participants": c.participants.len(),
                    "signals": c.signals,
                })
            })
            .collect();
        let _ = system_log::log_event(
            &self.db_pool,
            Some(&self.app_handle),
            "info",
            "memory",
            Some(run_id),
            Some(&memory_pass_id),
            serde_json::json!({
                "event": "memory_pass_decision_probe",
                "memory_pass_id": memory_pass_id.clone(),
                "candidate_count": decision_candidates.len(),
                "fact_count": fact_count,
                "relation_count": rel_count,
                "candidates": candidate_summary,
            }),
        )
        .await;

        let base_prompt = prompt_set.memory_control_prompt;
        let strict_prompt = strict_memory_prompt(&base_prompt);
        let mut attempt = "base";
        let mut memory_raw_for_attempt: Option<String> = None;

        let res = loop {
            let _ = memory_raw_for_attempt.take();
            let candidate_block = loop {
                let system_prompt = if attempt == "strict" || attempt == "minimal" {
                    strict_prompt.clone()
                } else {
                    base_prompt.clone()
                };
                let memory_payload = if attempt == "minimal" {
                    minimal_payload.clone()
                } else {
                    payload.clone()
                };
                let memory_request = ChatCompletionRequest {
                    model: model.to_string(),
                    messages: vec![
                        ChatMessage {
                            role: "system".to_string(),
                            content: system_prompt,
                        },
                        ChatMessage {
                            role: "user".to_string(),
                            content: memory_payload,
                        },
                    ],
                    stream: false,
                    temperature: None,
                    top_p: None,
                    max_tokens: None,
                    response_format: None,
                    tools: None,
                    tool_choice: None,
                    enable_thinking: None,
                    prefill: None,
                    skip_injection: Some(true),
                    skip_memory: Some(true),
                    skip_reminders: Some(true),
                    memory_expand: None,
                    allow_diagnostics: Some(false),
                    json_strict: None,
                    skip_sanitization: Some(true),
                    run_id: None,
                    request_label: Some(format!("memory_pass_{}", attempt)),
                };

                let memory_raw = match self
                    .execute_chat_request(base_url, api_key, &memory_request)
                    .await
                {
                    Ok((text, _)) => text,
                    Err(e) => {
                        let _ = system_log::log_event(
                            &self.db_pool,
                            Some(&self.app_handle),
                            "error",
                            "memory",
                            Some(run_id),
                            Some(&memory_pass_id),
                            serde_json::json!({
                                "event": "memory_pass_error",
                                "stage": "llm_request",
                                "error": e.to_string(),
                                "memory_pass_id": memory_pass_id.clone(),
                                "attempt": attempt,
                            }),
                        )
                        .await;
                        result.error = Some(e.to_string());
                        return result;
                    }
                };

                memory_raw_for_attempt = Some(memory_raw.clone());
                self
                    .maybe_capture_memory_raw(run_id, &memory_pass_id, &memory_raw)
                    .await;
                let trimmed = memory_raw.trim();
                if trimmed.is_empty() {
                    let _ = system_log::log_event(
                        &self.db_pool,
                        Some(&self.app_handle),
                        "warn",
                        "memory",
                        Some(run_id),
                        Some(&memory_pass_id),
                        serde_json::json!({
                            "event": "memory_pass_empty_response",
                            "memory_pass_id": memory_pass_id.clone(),
                            "attempt": attempt,
                        }),
                    )
                    .await;
                    if attempt == "base" {
                        attempt = "strict";
                        let _ = system_log::log_event(
                            &self.db_pool,
                            Some(&self.app_handle),
                            "info",
                            "memory",
                            Some(run_id),
                            Some(&memory_pass_id),
                            serde_json::json!({
                                "event": "memory_pass_retry_strict",
                                "memory_pass_id": memory_pass_id.clone(),
                            }),
                        )
                        .await;
                        continue;
                    }
                    result.error = Some("empty_response".to_string());
                    return result;
                }

                let candidate_block =
                    crate::core::memory::inject_context::extract_single_memory_block_strict(&memory_raw);
                if let Some(block) = candidate_block {
                    break block;
                }
                if let Some(block) = Self::extract_memory_block_lenient(&memory_raw) {
                    let _ = system_log::log_event(
                        &self.db_pool,
                        Some(&self.app_handle),
                        "info",
                        "memory",
                        Some(run_id),
                        Some(&memory_pass_id),
                        serde_json::json!({
                            "event": "memory_pass_fallback_block",
                            "memory_pass_id": memory_pass_id.clone(),
                            "attempt": attempt,
                            "reason": "lenient_extract",
                            "request_label": format!("memory_pass_{}", attempt),
                        }),
                    )
                    .await;
                    break block;
                }
                if let Some(block) = crate::core::memory::dsl::memory_json_to_dsl(&memory_raw) {
                    let _ = system_log::log_event(
                        &self.db_pool,
                        Some(&self.app_handle),
                        "info",
                        "memory",
                        Some(run_id),
                        Some(&memory_pass_id),
                        serde_json::json!({
                            "event": "memory_pass_json_fallback",
                            "memory_pass_id": memory_pass_id.clone(),
                            "attempt": attempt,
                            "reason": "missing_memory_block",
                            "raw_len": memory_raw.len(),
                            "raw_hash": hash_payload(&memory_raw),
                            "schema": "memory_json_fallback_v1",
                            "request_label": format!("memory_pass_{}", attempt),
                        }),
                    )
                    .await;
                    break block;
                }

                let _ = system_log::log_event(
                    &self.db_pool,
                    Some(&self.app_handle),
                    "error",
                    "memory",
                    Some(run_id),
                    Some(&memory_pass_id),
                    serde_json::json!({
                        "event": "memory_pass_invalid_output",
                        "memory_pass_id": memory_pass_id.clone(),
                        "attempt": attempt,
                        "model": model,
                        "memory_prompt_hash": prompt_set.memory_hash,
                        "memory_prompt_source": prompt_set.source,
                        "raw_len": memory_raw.len(),
                        "raw_hash": hash_payload(&memory_raw),
                        "raw_snippet": memory_raw.chars().take(200).collect::<String>(),
                        "schema": "memory_dsl_v1",
                        "reason": "missing_memory_block",
                        "request_label": format!("memory_pass_{}", attempt),
                    }),
                )
                .await;
                if attempt == "base" {
                    attempt = "strict";
                    let _ = system_log::log_event(
                        &self.db_pool,
                        Some(&self.app_handle),
                        "info",
                        "memory",
                        Some(run_id),
                        Some(&memory_pass_id),
                        serde_json::json!({
                            "event": "memory_pass_retry_strict",
                            "memory_pass_id": memory_pass_id.clone(),
                        }),
                    )
                    .await;
                    continue;
                }
                if attempt == "strict" {
                    attempt = "minimal";
                    let _ = system_log::log_event(
                        &self.db_pool,
                        Some(&self.app_handle),
                        "info",
                        "memory",
                        Some(run_id),
                        Some(&memory_pass_id),
                        serde_json::json!({
                            "event": "memory_pass_minimal_retry",
                            "memory_pass_id": memory_pass_id.clone(),
                        }),
                    )
                    .await;
                    continue;
                }
                result.error = Some("invalid_output".to_string());
                return result;
            };

            let mut block = candidate_block;
            let mut validation = crate::core::memory::dsl::validate_memory_block(&block);
            if !validation.valid {
                let repair_ctx = crate::core::memory::dsl::RepairContext {
                    now: chrono::Utc::now(),
                    assistant_name: assistant_name.clone(),
                };
                let repair = crate::core::memory::dsl::repair_memory_block(&block, &repair_ctx);
                if repair.repaired {
                    let _ = system_log::log_event(
                        &self.db_pool,
                        Some(&self.app_handle),
                        "info",
                        "memory",
                        Some(run_id),
                        Some(&memory_pass_id),
                        serde_json::json!({
                            "event": "memory_pass_repair",
                            "memory_pass_id": memory_pass_id.clone(),
                            "reason": "dsl_validation",
                            "repaired": repair.repaired,
                            "confidence": repair.confidence,
                            "dropped_lines": repair.dropped_lines,
                            "errors": repair.errors.len(),
                        }),
                    )
                    .await;
                    if let Some(repaired_block) = repair.repaired_block {
                        block = repaired_block;
                    } else {
                        result.error = Some("repair_failed".to_string());
                        return result;
                    }
                    if repair.confidence < REPAIR_CONFIDENCE_THRESHOLD {
                        result.error = Some("repair_low_confidence".to_string());
                        return result;
                    }
                }
                validation = crate::core::memory::dsl::validate_memory_block(&block);
                if !validation.valid {
                    if let Some(raw) = memory_raw_for_attempt.as_deref() {
                        if let Some(json_block) = crate::core::memory::dsl::memory_json_to_dsl(raw) {
                            let _ = system_log::log_event(
                                &self.db_pool,
                                Some(&self.app_handle),
                                "info",
                                "memory",
                                Some(run_id),
                                Some(&memory_pass_id),
                                serde_json::json!({
                                    "event": "memory_pass_json_fallback",
                                    "memory_pass_id": memory_pass_id.clone(),
                                    "attempt": attempt,
                                    "reason": "dsl_validation",
                                    "raw_len": raw.len(),
                                    "raw_hash": hash_payload(raw),
                                    "schema": "memory_json_fallback_v1",
                                    "request_label": format!("memory_pass_{}", attempt),
                                }),
                            )
                            .await;
                            block = json_block;
                            validation = crate::core::memory::dsl::validate_memory_block(&block);
                        }
                    }
                }
                if !validation.valid {
                    let _ = system_log::log_event(
                        &self.db_pool,
                        Some(&self.app_handle),
                        "error",
                        "memory",
                        Some(run_id),
                        Some(&memory_pass_id),
                        serde_json::json!({
                            "event": "memory_pass_invalid_output",
                            "memory_pass_id": memory_pass_id.clone(),
                            "attempt": attempt,
                            "model": model,
                            "memory_prompt_hash": prompt_set.memory_hash,
                            "memory_prompt_source": prompt_set.source,
                            "reason": "dsl_validation",
                            "error_count": validation.errors.len(),
                            "statement_count": validation.statement_count,
                            "schema": "memory_dsl_v1",
                            "raw_snippet": block.chars().take(200).collect::<String>(),
                            "request_label": format!("memory_pass_{}", attempt),
                        }),
                    )
                    .await;
                    if attempt == "base" {
                        attempt = "strict";
                        let _ = system_log::log_event(
                            &self.db_pool,
                            Some(&self.app_handle),
                            "info",
                            "memory",
                            Some(run_id),
                            Some(&memory_pass_id),
                            serde_json::json!({
                                "event": "memory_pass_retry_strict",
                                "memory_pass_id": memory_pass_id.clone(),
                            }),
                        )
                        .await;
                        continue;
                    }
                    if attempt == "strict" {
                        attempt = "minimal";
                        let _ = system_log::log_event(
                            &self.db_pool,
                            Some(&self.app_handle),
                            "info",
                            "memory",
                            Some(run_id),
                            Some(&memory_pass_id),
                            serde_json::json!({
                                "event": "memory_pass_minimal_retry",
                                "memory_pass_id": memory_pass_id.clone(),
                            }),
                        )
                        .await;
                        continue;
                    }
                    result.error = Some("invalid_output".to_string());
                    return result;
                }
            }

            let (filtered_block, filter_stats) = filter_interrogative_memory_block(&block, user_message);
            if filter_stats.dropped_total() > 0 {
                let _ = system_log::log_event(
                    &self.db_pool,
                    Some(&self.app_handle),
                    "info",
                    "memory",
                    Some(run_id),
                    Some(&memory_pass_id),
                    serde_json::json!({
                        "event": "memory_pass_interrogative_filter",
                        "memory_pass_id": memory_pass_id.clone(),
                        "attempt": attempt,
                        "user_interrogative": filter_stats.user_interrogative,
                        "total": filter_stats.total,
                        "kept": filter_stats.kept,
                        "dropped_relations": filter_stats.dropped_relations,
                        "dropped_interrogatives": filter_stats.dropped_interrogatives,
                    }),
                )
                .await;
            }
            if filtered_block.trim().is_empty() {
                let _ = system_log::log_event(
                    &self.db_pool,
                    Some(&self.app_handle),
                    "info",
                    "memory",
                    Some(run_id),
                    Some(&memory_pass_id),
                    serde_json::json!({
                        "event": "memory_pass_interrogative_filtered_empty",
                        "memory_pass_id": memory_pass_id.clone(),
                        "attempt": attempt,
                    }),
                )
                .await;
                result.error = Some("interrogative_filtered".to_string());
                return result;
            }
            block = filtered_block;

        let payload_hash = hash_text(&block);
        let _ = db
            .log_memory_write(
                Some(&session_id),
                "memory_pass",
                "model_client",
                "memory_pass",
                Some(run_id),
                None,
                Some(&payload_hash),
                None,
                None,
            )
            .await;

        let api = crate::core::memory::api::MemoryApi::new(
            self.db_pool.clone(),
            Some(Arc::new(self.clone())),
            session_id.clone(),
        )
        .await;

        let res = api
            .parse_and_compile(
                &block,
                crate::core::memory::types::Scope::Global,
                crate::core::memory::types::SourceType::User,
                None,
                chrono::Utc::now(),
            )
            .await;

        if (res.written_ids.is_empty() || !res.errors.is_empty()) && attempt == "base" {
            let _ = system_log::log_event(
                &self.db_pool,
                Some(&self.app_handle),
                "info",
                "memory",
                Some(run_id),
                Some(&memory_pass_id),
                serde_json::json!({
                    "event": "memory_pass_retry_strict",
                    "memory_pass_id": memory_pass_id.clone(),
                    "reason": if res.written_ids.is_empty() { "zero_writes" } else { "compile_errors" },
                }),
            )
            .await;
            attempt = "strict";
            continue;
        }

        break res;
        };

        let error_count = res.errors.len();
        let _ = system_log::log_event(
            &self.db_pool,
            Some(&self.app_handle),
            if error_count > 0 { "warn" } else { "info" },
            "memory",
            Some(run_id),
            Some(&memory_pass_id),
            serde_json::json!({
                "event": "memory_pass_result",
                "written_ids": res.written_ids.len(),
                "conflict_ids": res.conflict_ids.len(),
                "pending_writes": res.pending_writes.len(),
                "pending_clarify": res.pending_clarify.is_some(),
                "claim_ids": res.claim_ids.len(),
                "errors": error_count,
                "memory_pass_id": memory_pass_id.clone(),
            }),
        )
        .await;

        if !res.errors.is_empty() {
            let errors = res.errors.clone();
            let now = chrono::Utc::now().to_rfc3339();
            let error_text = errors.join(" | ");
            let _ = sqlx::query("INSERT INTO kv_store (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP")
                .bind("memory_last_error_at")
                .bind(&now)
                .execute(&self.db_pool)
                .await;
            let _ = sqlx::query("INSERT INTO kv_store (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP")
                .bind("memory_last_error")
                .bind(&error_text)
                .execute(&self.db_pool)
                .await;
            let _ = self.app_handle.emit(
                "memory_error",
                serde_json::json!({ "errors": errors, "timestamp": now }),
            );
            result.error = Some(error_text);
        }
        if res.errors.is_empty() && res.written_ids.is_empty() {
            let _ = system_log::log_event(
                &self.db_pool,
                Some(&self.app_handle),
                "warn",
                "memory",
                Some(run_id),
                Some(&memory_pass_id),
                serde_json::json!({
                    "event": "memory_pass_zero_writes",
                    "memory_pass_id": memory_pass_id.clone(),
                }),
            )
            .await;
            result.error = Some("no_writes".to_string());
        }

        if !res.conflict_ids.is_empty() {
            let _ = self.app_handle.emit("memory_conflict", res.conflict_ids.clone());
        }

        let mut facts_written = 0usize;
        let mut rels_written = 0usize;
        if !res.written_ids.is_empty() {
            let placeholders = res
                .written_ids
                .iter()
                .map(|_| "?")
                .collect::<Vec<_>>()
                .join(",");
            let query = format!(
                "SELECT kind, COUNT(*) as count FROM ics_beliefs WHERE id IN ({}) GROUP BY kind",
                placeholders
            );
            let mut q = sqlx::query(&query);
            for id in res.written_ids.iter() {
                q = q.bind(id);
            }
            if let Ok(rows) = q.fetch_all(&self.db_pool).await {
                for row in rows {
                    let kind: String = row.try_get("kind").unwrap_or_default();
                    let count: i64 = row.try_get("count").unwrap_or(0);
                    if kind == "fact" {
                        facts_written = count.max(0) as usize;
                    } else if kind == "rel" {
                        rels_written = count.max(0) as usize;
                    }
                }
            }
        }

        result.conflict_ids = res.conflict_ids.clone();
        result.pending_clarify = res.pending_clarify.is_some();
        result.written_ids = res.written_ids.len();
        result.facts_written = facts_written;
        result.rels_written = rels_written;
        result.success = res.errors.is_empty() && !res.written_ids.is_empty();
        result
    }

    /// Streaming version of chat - emits token events as chunks arrive
    pub async fn stream_chat(
        &self,
        app_handle: tauri::AppHandle,
        base_url: &str,
        api_key: Option<&str>,
        request: &ChatCompletionRequest,
    ) -> Result<(String, Option<Vec<crate::models::ToolCall>>), String> {
        use eventsource_stream::Eventsource;
        use futures_util::StreamExt;
        use tauri::Emitter;

        let final_request = request.clone();

        self.log_prompt_messages(&final_request, "stream");
        self.log_final_request(&final_request);

        let (prompt_len, prompt_hash) = summarize_messages(&final_request.messages);
        let tool_count = final_request.tools.as_ref().map(|t| t.len()).unwrap_or(0);
        let _ = system_log::log_event(
            &self.db_pool,
            Some(&self.app_handle),
            "info",
            "model",
            final_request.run_id.as_deref(),
            None,
            serde_json::json!({
                "event": "request",
                "model": final_request.model.as_str(),
                "stream": true,
                "message_count": final_request.messages.len(),
                "prompt_len": prompt_len,
                "prompt_hash": prompt_hash,
                "tool_count": tool_count,
            }),
        )
        .await;

        let url = format!("{}/chat/completions", base_url);
        let mut req_builder = self.client.post(&url).json(&final_request);
        if let Some(key) = api_key {
            req_builder = req_builder.bearer_auth(key);
        }

        let call_started = Instant::now();
        let response = req_builder.send().await.map_err(|e| e.to_string())?;
        let call_ms = call_started.elapsed().as_millis() as i64;
        let bucket = if request.run_id.is_some() { "primary" } else { "summary" };
        self.update_latency_avg(bucket, call_ms).await;
        let _ = system_log::log_event(
            &self.db_pool,
            Some(&self.app_handle),
            "info",
            "model",
            request.run_id.as_deref(),
            None,
            serde_json::json!({
                "event": "timing_model_call",
                "duration_ms": call_ms,
                "model": request.model,
                "stream": request.stream,
            }),
        )
        .await;
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            let _ = system_log::log_event(
                &self.db_pool,
                Some(&self.app_handle),
                "error",
                "model",
                final_request.run_id.as_deref(),
                None,
                serde_json::json!({
                    "event": "error",
                    "error": error_text,
                }),
            )
            .await;
            return Err(format!("LLM Error: {}", error_text));
        }

        let mut stream = response.bytes_stream().eventsource();
        let mut full_content = String::new();
        let allow_diagnostics = final_request.allow_diagnostics.unwrap_or(false);
        let (user_name, assistant_name) = self.fetch_display_names().await;
        let mut reminder_filter = reminder_blocks::ReminderStreamFilter::new();
        let mut memory_trigger_filter = crate::core::memory::inject_context::MemoryTriggerStreamFilter::new();
        let mut memory_filter = crate::core::memory::inject_context::MemoryStreamFilter::new();
        let mut scaffold_filter = ScaffoldStreamFilter::new();
        let mut internal_filter = InternalStreamFilter::new(
            allow_diagnostics,
            assistant_name.as_deref(),
            user_name.as_deref(),
        );

        #[derive(Default)]
        struct ToolCallAccumulator {
            id: String,
            name: String,
            arguments: String,
            call_type: String,
        }

        let mut tool_call_acc: HashMap<usize, ToolCallAccumulator> = HashMap::new();
        let mut chunk_counter = 0usize;

        while let Some(event) = stream.next().await {
            match event {
                Ok(event) => {
                    if event.data == "[DONE]" {
                        break;
                    }

                    let chunk: ChatCompletionResponseDelta = match serde_json::from_str(&event.data) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };

                    if let Some(choice) = chunk.choices.get(0) {
                        if let Some(content) = &choice.delta.content {
                            full_content.push_str(content);
                            let filtered_trigger = memory_trigger_filter.filter_chunk(content);
                            let filtered_memory = memory_filter.filter_chunk(&filtered_trigger);
                            let filtered_reminder = reminder_filter.filter_chunk(&filtered_memory);
                            let filtered = scaffold_filter.filter_chunk(&filtered_reminder);
                            let filtered = internal_filter.filter_chunk(&filtered);
                            if !filtered.is_empty() {
                                let _ = app_handle.emit("token", filtered);
                            }
                        }
                        if let Some(tool_calls) = choice.delta.tool_calls.as_ref() {
                            for call in tool_calls {
                                let index = call.index.unwrap_or(0) as usize;
                                let entry = tool_call_acc.entry(index).or_default();
                                if let Some(id) = call.id.as_ref() {
                                    entry.id = id.clone();
                                }
                                if let Some(call_type) = call.r#type.as_ref() {
                                    entry.call_type = call_type.clone();
                                }
                                if let Some(func) = call.function.as_ref() {
                                    if let Some(name) = func.name.as_ref() {
                                        entry.name = name.clone();
                                    }
                                    if let Some(args) = func.arguments.as_ref() {
                                        entry.arguments.push_str(args);
                                    }
                                }
                            }
                        }
                    }
                    if let Some(run_id) = request.run_id.as_deref() {
                        chunk_counter += 1;
                        if chunk_counter % 10 == 0 && self.should_skip_side_effects(Some(run_id)).await {
                            break;
                        }
                    }
                }
                Err(e) => eprintln!("Stream error: {}", e),
            }
        }

        let remaining_trigger = memory_trigger_filter.finalize();
        let mut remaining = String::new();
        if !remaining_trigger.is_empty() {
            let filtered_memory = memory_filter.filter_chunk(&remaining_trigger);
            let filtered_reminder = reminder_filter.filter_chunk(&filtered_memory);
            let filtered = scaffold_filter.filter_chunk(&filtered_reminder);
            let filtered = internal_filter.filter_chunk(&filtered);
            if !filtered.is_empty() {
                remaining.push_str(&filtered);
            }
        }
        let remaining_memory = memory_filter.finalize();
        if !remaining_memory.is_empty() {
            let filtered_reminder = reminder_filter.filter_chunk(&remaining_memory);
            let filtered = scaffold_filter.filter_chunk(&filtered_reminder);
            let filtered = internal_filter.filter_chunk(&filtered);
            remaining.push_str(&filtered);
        }
        let reminder_remaining = reminder_filter.finalize();
        if !reminder_remaining.is_empty() {
            let filtered = scaffold_filter.filter_chunk(&reminder_remaining);
            let filtered = internal_filter.filter_chunk(&filtered);
            remaining.push_str(&filtered);
        }
        let scaffold_remaining = scaffold_filter.finalize();
        if !scaffold_remaining.is_empty() {
            let filtered = internal_filter.filter_chunk(&scaffold_remaining);
            remaining.push_str(&filtered);
        }
        let internal_remaining = internal_filter.finalize();
        if !internal_remaining.is_empty() {
            remaining.push_str(&internal_remaining);
        }
        if !remaining.is_empty() {
            let _ = app_handle.emit("token", remaining);
        }

        let mut indices: Vec<usize> = tool_call_acc.keys().copied().collect();
        indices.sort_unstable();
        let mut tool_calls: Vec<crate::models::ToolCall> = Vec::new();
        for idx in indices {
            if let Some(acc) = tool_call_acc.remove(&idx) {
                if acc.name.trim().is_empty() && acc.arguments.trim().is_empty() {
                    continue;
                }
                let id = if acc.id.trim().is_empty() {
                    Uuid::new_v4().to_string()
                } else {
                    acc.id
                };
                let call_type = if acc.call_type.trim().is_empty() {
                    "function".to_string()
                } else {
                    acc.call_type
                };
                tool_calls.push(crate::models::ToolCall {
                    id,
                    r#type: call_type,
                    function: crate::models::ToolCallFunction {
                        name: acc.name,
                        arguments: acc.arguments,
                    },
                });
            }
        }

        let tool_calls_opt = if tool_calls.is_empty() { None } else { Some(tool_calls) };

        let _ = system_log::log_event(
            &self.db_pool,
            Some(&self.app_handle),
            "info",
            "model",
            final_request.run_id.as_deref(),
            None,
            serde_json::json!({
                "event": "response",
                "content_len": full_content.len(),
                "content_hash": hash_text(&full_content),
                "tool_call_count": tool_calls_opt.as_ref().map(|c| c.len()).unwrap_or(0),
            }),
        )
        .await;

        let _ = app_handle.emit(
            "stream_end",
            serde_json::json!({
                "run_id": final_request.run_id,
                "content_len": full_content.len(),
            }),
        );
        let _ = system_log::log_event(
            &self.db_pool,
            Some(&self.app_handle),
            "info",
            "model",
            final_request.run_id.as_deref(),
            None,
            serde_json::json!({
                "event": "stream_end",
                "content_len": full_content.len(),
            }),
        )
        .await;

        Ok((full_content, tool_calls_opt))
    }

    pub async fn embed(&self, base_url: &str, api_key: Option<&str>, model: &str, input: &str) -> Result<Vec<f32>, String> {
        let url = format!("{}/embeddings", base_url);
        let request = EmbeddingRequest {
            model: model.to_string(),
            input: input.to_string(),
        };

        let mut req_builder = self.client.post(&url).json(&request);
        if let Some(key) = api_key {
            req_builder = req_builder.bearer_auth(key);
        }

        let response = req_builder.send().await.map_err(|e| e.to_string())?;
        if !response.status().is_success() {
             let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
             return Err(format!("Embedding Error: {}", error_text));
        }

        let body: EmbeddingResponse = response.json().await.map_err(|e| e.to_string())?;
        if let Some(first) = body.data.first() {
            Ok(first.embedding.clone())
        } else {
            Err("No embedding returned".to_string())
        }
    }

    async fn append_pending_clarify(&self, messages: &mut Vec<ChatMessage>) -> Result<(), String> {
        if let Ok(Some(row)) = sqlx::query(
            "SELECT id, ref_text, candidates_json FROM ics_pending_clarify WHERE status = 'pending' ORDER BY created_at DESC LIMIT 1"
        )
        .fetch_optional(&self.db_pool)
        .await {
            use sqlx::Row;
            let ref_text: String = row.get("ref_text");
            let candidates_json: String = row.get("candidates_json");

            let candidates: Vec<serde_json::Value> = serde_json::from_str(&candidates_json).unwrap_or_default();
            let options_str: String = candidates.iter().enumerate()
                .map(|(i, c)| {
                    let label = c.get("label").and_then(|v| v.as_str()).unwrap_or("Unknown");
                    let context = c.get("context").and_then(|v| v.as_str()).unwrap_or("");
                    format!("{}. {} ({})", i + 1, label, context)
                })
                .collect::<Vec<_>>()
                .join("\n");

            let clarify_msg = format!(
                "\n\n[CLARIFY] Multiple entities match '{}'. Please ask the user to clarify which one they mean:\n{}\n",
                ref_text.trim_matches(|c| c == '"' || c == '\''),
                options_str
            );

            if let Some(first) = messages.first_mut() {
                if first.role == "system" {
                    first.content.push_str(&clarify_msg);
                }
            }
        }

        Ok(())
    }

    fn is_monologue_request(&self, request: &ChatCompletionRequest) -> bool {
        request
            .messages
            .iter()
            .find(|m| m.role == "system")
            .map(|m| {
                let content = m.content.to_lowercase();
                content.contains("private inner monologue")
                    || content.contains("private internal dialogue")
            })
            .unwrap_or(false)
    }

    fn request_expects_json(request: &ChatCompletionRequest) -> bool {
        request
            .response_format
            .as_ref()
            .and_then(|v| v.get("type"))
            .and_then(|v| v.as_str())
            .map(|kind| matches!(kind, "json_object" | "json_schema"))
            .unwrap_or(false)
    }

    fn request_json_strict(request: &ChatCompletionRequest) -> bool {
        request.json_strict.unwrap_or(false)
    }

    fn response_format_summary(request: &ChatCompletionRequest) -> Value {
        let Some(format) = request.response_format.as_ref() else {
            return Value::Null;
        };
        let format_type = format
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let schema_hash = format
            .get("json_schema")
            .or_else(|| format.get("schema"))
            .and_then(|schema| serde_json::to_string(schema).ok())
            .map(|raw| hash_payload(&raw));
        serde_json::json!({
            "type": format_type,
            "schema_hash": schema_hash,
        })
    }

    fn simplify_json_prompt(messages: &mut Vec<ChatMessage>) {
        let minimal = "Return ONLY valid JSON. Do not include any other text.";
        if let Some(first) = messages.iter_mut().find(|m| m.role == "system") {
            first.content = minimal.to_string();
        } else {
            messages.insert(
                0,
                ChatMessage {
                    role: "system".to_string(),
                    content: minimal.to_string(),
                },
            );
        }
    }

    fn reasoning_json_fallback(reasoning_content: &str) -> Option<String> {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(reasoning_content) {
            if value.is_object() {
                return Some(reasoning_content.to_string());
            }
        }
        let (repaired, _repaired) = crate::core::kernel::repair_json_object(reasoning_content);
        repaired
            .filter(|v| v.is_object())
            .map(|v| v.to_string())
    }

    fn extract_memory_block_lenient(output: &str) -> Option<String> {
        let blocks = crate::core::memory::inject_context::extract_memory_blocks(output);
        blocks
            .into_iter()
            .find(|b| !b.trim().is_empty())
            .map(|b| b.trim().to_string())
    }

    fn trim_tail_chars(input: &str, max_chars: usize) -> String {
        let total = input.chars().count();
        if total <= max_chars {
            return input.to_string();
        }
        input.chars().skip(total - max_chars).collect()
    }

    fn build_memory_pass_payload_minimal(
        user_message: &str,
        assistant_message: &str,
        user_name: Option<&str>,
        assistant_name: Option<&str>,
    ) -> String {
        let mut payload = String::new();
        payload.push_str("USER_HANDLE: $user\n");
        if let Some(name) = user_name {
            payload.push_str(&format!("USER_NAME: {}\n", name));
        }
        payload.push_str("ASSISTANT_HANDLE: $assistant\n");
        if let Some(name) = assistant_name {
            payload.push_str(&format!("ASSISTANT_NAME: {}\n", name));
        }
        payload.push_str("USER_MESSAGE:\n<<<BEGIN_USER_MESSAGE>>>\n");
        payload.push_str(user_message.trim());
        payload.push_str("\n<<<END_USER_MESSAGE>>>\n");
        payload.push_str("ASSISTANT_MESSAGE:\n<<<BEGIN_ASSISTANT_MESSAGE>>>\n");
        payload.push_str(assistant_message.trim());
        payload.push_str("\n<<<END_ASSISTANT_MESSAGE>>>\n");
        payload
    }

    fn reduce_prediction_prompt(messages: &mut Vec<ChatMessage>) {
        let last_user = messages.iter().rev().find(|m| m.role == "user").cloned();
        let mut reduced = Vec::new();
        reduced.push(ChatMessage {
            role: "system".to_string(),
            content: "Return ONLY valid JSON. Do not include any other text.".to_string(),
        });
        if let Some(user) = last_user {
            let trimmed = Self::trim_tail_chars(&user.content, 1200);
            reduced.push(ChatMessage {
                role: "user".to_string(),
                content: trimmed,
            });
        }
        *messages = reduced;
    }

    async fn json_only_disabled_for_model(&self, model: &str) -> bool {
        let db = Db {
            pool: self.db_pool.clone(),
        };
        let settings = db.get_settings().await.ok();
        let raw = settings
            .and_then(|s| s.json_only_disabled_models)
            .unwrap_or_default();
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return false;
        }
        let mut models: Vec<String> = Vec::new();
        if trimmed.starts_with('[') {
            if let Ok(list) = serde_json::from_str::<Vec<String>>(trimmed) {
                models.extend(list);
            }
        } else {
            models.extend(
                trimmed
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty()),
            );
        }
        if models.is_empty() {
            return false;
        }
        let model_lower = model.trim().to_lowercase();
        models.into_iter().any(|entry| {
            let needle = entry.trim().to_lowercase();
            if needle.is_empty() {
                return false;
            }
        model_lower == needle || model_lower.contains(&needle)
        })
    }

    async fn empty_response_retry_config(&self) -> (usize, u64) {
        let settings = Db { pool: self.db_pool.clone() }.get_settings().await.ok();
        empty_response_retry_config_from_settings(settings.as_ref())
    }

    fn should_emit_empty_response_event(request: &ChatCompletionRequest) -> bool {
        request
            .request_label
            .as_deref()
            .map(|label| label == "primary_response")
            .unwrap_or(false)
            && request.run_id.as_deref().map(|id| !id.trim().is_empty()).unwrap_or(false)
    }

    async fn emit_empty_response_error(
        &self,
        request: &ChatCompletionRequest,
        attempt: usize,
        max: usize,
    ) {
        if !Self::should_emit_empty_response_event(request) {
            return;
        }
        let payload = serde_json::json!({
            "run_id": request.run_id.as_deref().unwrap_or(""),
            "request_label": request.request_label.as_deref().unwrap_or(""),
            "attempt": attempt,
            "max": max,
        });
        let _ = self.app_handle.emit("response_empty_error", payload);
    }

    async fn execute_chat_request(
        &self,
        base_url: &str,
        api_key: Option<&str>,
        request: &ChatCompletionRequest,
    ) -> Result<(String, Option<Vec<crate::models::ToolCall>>), String> {
        let mut request = request.clone();
        let mut retried_json_reasoning = false;
        let mut retried_json_strict = false;
        let mut retried_prediction_reduce = false;
        let mut retried_thinking_prefill = false;
        let mut reasoning_only_count: usize = 0;
        let mut reasoning_only_repeat_logged = false;
        let (empty_retry_max, empty_retry_timeout_ms) = self.empty_response_retry_config().await;
        let mut empty_retry_count: usize = 0;
        let mut relaxed_empty_retry = false;

        if request.enable_thinking.is_none() || request.prefill.is_none() {
            if let Ok(settings) = (Db { pool: self.db_pool.clone() }).get_settings().await {
                if let Some(defaults) = settings.request_defaults.as_ref() {
                    if request.enable_thinking.is_none() {
                        request.enable_thinking = defaults.get("enable_thinking").and_then(|v| v.as_bool());
                    }
                    if request.prefill.is_none() {
                        if let Some(value) = defaults.get("prefill") {
                            request.prefill = Some(value.clone());
                        }
                    }
                }
            }
        }

        if request.enable_thinking == Some(true) && request.prefill.is_some() {
            request.prefill = None;
            let _ = system_log::log_event(
                &self.db_pool,
                Some(&self.app_handle),
                "info",
                "model",
                request.run_id.as_deref(),
                None,
                serde_json::json!( {
                    "event": "prefill_disabled_for_thinking",
                    "model": request.model,
                    "request_label": request.request_label.clone(),
                }),
            )
            .await;
        }

        if Self::request_expects_json(&request)
            && self.json_only_disabled_for_model(&request.model).await
        {
            request.response_format = None;
            request.json_strict = Some(false);
            let _ = system_log::log_event(
                &self.db_pool,
                Some(&self.app_handle),
                "info",
                "model",
                request.run_id.as_deref(),
                None,
                serde_json::json!( {
                    "event": "json_mode_disabled_for_model",
                    "model": request.model,
                }),
            )
            .await;
        }

        loop {
            let (prompt_len, prompt_hash) = summarize_messages(&request.messages);
            let tool_count = request.tools.as_ref().map(|t| t.len()).unwrap_or(0);
            let response_format = Self::response_format_summary(&request);
            let _ = system_log::log_event(
                &self.db_pool,
                Some(&self.app_handle),
                "info",
                "model",
                request.run_id.as_deref(),
                None,
                serde_json::json!({
                    "event": "request",
                    "model": request.model,
                    "stream": request.stream,
                    "message_count": request.messages.len(),
                    "prompt_len": prompt_len,
                    "prompt_hash": prompt_hash,
                    "tool_count": tool_count,
                    "request_label": request.request_label.clone(),
                    "response_format": response_format.clone(),
                    "max_tokens": request.max_tokens,
                    "json_strict": request.json_strict,
                    "enable_thinking": request.enable_thinking,
                    "prefill_set": request.prefill.is_some(),
                }),
            )
            .await;

            let url = format!("{}/chat/completions", base_url);
            let mut req_builder = self.client.post(&url).json(&request);
            if let Some(key) = api_key {
                req_builder = req_builder.bearer_auth(key);
            }

            let call_started = Instant::now();
            let response = req_builder.send().await.map_err(|e| e.to_string())?;
            let call_ms = call_started.elapsed().as_millis() as i64;
            let _ = system_log::log_event(
                &self.db_pool,
                Some(&self.app_handle),
                "info",
                "model",
                request.run_id.as_deref(),
                None,
                serde_json::json!({
                    "event": "timing_model_call",
                    "duration_ms": call_ms,
                    "model": request.model,
                    "stream": request.stream,
                }),
            )
            .await;
            if !response.status().is_success() {
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                let error_lower = error_text.to_lowercase();
                if !retried_thinking_prefill
                    && error_lower.contains("prefill")
                    && error_lower.contains("enable_thinking")
                {
                    retried_thinking_prefill = true;
                    request.enable_thinking = Some(false);
                    request.prefill = None;
                    let _ = system_log::log_event(
                        &self.db_pool,
                        Some(&self.app_handle),
                        "warn",
                        "model",
                        request.run_id.as_deref(),
                        None,
                        serde_json::json!( {
                            "event": "prefill_incompatible_retry",
                            "model": request.model,
                            "request_label": request.request_label.clone(),
                            "error": error_text,
                        }),
                    )
                    .await;
                    continue;
                }
                let _ = system_log::log_event(
                    &self.db_pool,
                    Some(&self.app_handle),
                    "error",
                    "model",
                    request.run_id.as_deref(),
                    None,
                    serde_json::json!({
                        "event": "error",
                        "error": error_text,
                    }),
                )
                .await;
                return Err(format!("LLM Error: {}", error_text));
            }

            let json: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
            let choice = &json["choices"][0]["message"];
            let reasoning_content = choice
                .get("reasoning_content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let mut raw_content = choice["content"].as_str().unwrap_or("").to_string();

            let tool_calls = if let Some(calls) = choice["tool_calls"].as_array() {
                let parsed: Vec<crate::models::ToolCall> = serde_json::from_value(serde_json::Value::Array(calls.clone()))
                    .map_err(|e| format!("Tool Call Parse Error: {}", e))?;
                Some(parsed)
            } else {
                None
            };

            let _ = system_log::log_event(
                &self.db_pool,
                Some(&self.app_handle),
                "info",
                "model",
                request.run_id.as_deref(),
                None,
                serde_json::json!({
                    "event": "response",
                    "content_len": raw_content.len(),
                    "content_hash": hash_text(&raw_content),
                    "tool_call_count": tool_calls.as_ref().map(|c| c.len()).unwrap_or(0),
                }),
            )
            .await;
            if raw_content.trim().is_empty() && !reasoning_content.trim().is_empty() {
                let is_monologue = self.is_monologue_request(&request);
                let expects_json = Self::request_expects_json(&request);
                let json_strict = Self::request_json_strict(&request);
                reasoning_only_count = reasoning_only_count.saturating_add(1);
                if reasoning_only_count == 1 {
                    let _ = system_log::log_event(
                        &self.db_pool,
                        Some(&self.app_handle),
                        "warn",
                        "model",
                        request.run_id.as_deref(),
                        None,
                        serde_json::json!({
                            "event": "response_empty_with_reasoning",
                            "reasoning_len": reasoning_content.len(),
                            "reasoning_hash": hash_text(&reasoning_content),
                            "monologue_request": is_monologue,
                            "expects_json": expects_json,
                            "json_strict": json_strict,
                            "request_label": request.request_label.clone(),
                            "response_format": response_format.clone(),
                            "max_tokens": request.max_tokens,
                        }),
                    )
                    .await;
                } else if !reasoning_only_repeat_logged {
                    reasoning_only_repeat_logged = true;
                    let _ = system_log::log_event(
                        &self.db_pool,
                        Some(&self.app_handle),
                        "warn",
                        "model",
                        request.run_id.as_deref(),
                        None,
                        serde_json::json!({
                            "event": "response_empty_with_reasoning_repeat",
                            "count": reasoning_only_count,
                            "monologue_request": is_monologue,
                            "expects_json": expects_json,
                            "json_strict": json_strict,
                            "request_label": request.request_label.clone(),
                            "response_format": response_format.clone(),
                            "max_tokens": request.max_tokens,
                        }),
                    )
                    .await;
                }
                if expects_json {
                    if let Some(fallback) = Self::reasoning_json_fallback(&reasoning_content) {
                        raw_content = fallback;
                        let _ = system_log::log_event(
                            &self.db_pool,
                            Some(&self.app_handle),
                            "info",
                            "model",
                            request.run_id.as_deref(),
                            None,
                            serde_json::json!({
                                "event": "response_reasoning_json_fallback",
                                "content_len": raw_content.len(),
                                "content_hash": hash_text(&raw_content),
                                "monologue_request": is_monologue,
                            }),
                        )
                        .await;
                    } else {
                        let _ = system_log::log_event(
                            &self.db_pool,
                            Some(&self.app_handle),
                            "warn",
                            "model",
                            request.run_id.as_deref(),
                            None,
                            serde_json::json!({
                                "event": "response_reasoning_json_invalid",
                                "reasoning_len": reasoning_content.len(),
                                "reasoning_hash": hash_text(&reasoning_content),
                                "monologue_request": is_monologue,
                                "request_label": request.request_label.clone(),
                                "response_format": response_format.clone(),
                                "max_tokens": request.max_tokens,
                                "json_strict": json_strict,
                            }),
                        )
                        .await;
                            if json_strict && !retried_json_strict {
                                retried_json_strict = true;
                                Self::simplify_json_prompt(&mut request.messages);
                                let _ = system_log::log_event(
                                    &self.db_pool,
                                    Some(&self.app_handle),
                                    "info",
                                    "model",
                                    request.run_id.as_deref(),
                                    None,
                                    serde_json::json!({
                                        "event": "response_reasoning_json_retry",
                                        "mode": "strict_simplified_prompt",
                                        "monologue_request": is_monologue,
                                    }),
                                )
                                .await;
                                continue;
                            }
                            if request.response_format.is_some() && !retried_json_reasoning {
                                retried_json_reasoning = true;
                                request.response_format = None;
                                request.json_strict = Some(false);
                                let _ = system_log::log_event(
                                    &self.db_pool,
                                    Some(&self.app_handle),
                                    "info",
                                    "model",
                                    request.run_id.as_deref(),
                                    None,
                                    serde_json::json!({
                                        "event": "response_reasoning_json_retry",
                                        "mode": "disable_response_format",
                                        "monologue_request": is_monologue,
                                    }),
                                )
                                .await;
                                continue;
                            }
                            let is_prediction = request
                                .request_label
                                .as_deref()
                                .map(|label| label.contains("prediction_generation"))
                                .unwrap_or(false);
                            if is_prediction && !retried_prediction_reduce {
                                retried_prediction_reduce = true;
                                Self::reduce_prediction_prompt(&mut request.messages);
                                request.response_format = None;
                                request.json_strict = Some(false);
                                if request.max_tokens.map(|v| v > 400).unwrap_or(true) {
                                    request.max_tokens = Some(400);
                                }
                                let _ = system_log::log_event(
                                    &self.db_pool,
                                    Some(&self.app_handle),
                                    "info",
                                    "model",
                                    request.run_id.as_deref(),
                                    None,
                                    serde_json::json!({
                                        "event": "response_reasoning_json_retry",
                                        "mode": "prediction_reduced_prompt",
                                        "monologue_request": is_monologue,
                                    }),
                                )
                                .await;
                                continue;
                            }
                            return Err("EMPTY_CONTENT_JSON_MODE".to_string());
                    }
                } else {
                    raw_content = reasoning_content;
                    let _ = system_log::log_event(
                        &self.db_pool,
                        Some(&self.app_handle),
                        "info",
                        "model",
                        request.run_id.as_deref(),
                        None,
                        serde_json::json!({
                            "event": "response_reasoning_fallback",
                            "content_len": raw_content.len(),
                            "content_hash": hash_text(&raw_content),
                            "monologue_request": is_monologue,
                        }),
                    )
                    .await;
                }
            }
            let has_tool_calls = tool_calls.as_ref().map(|c| !c.is_empty()).unwrap_or(false);
            if raw_content.trim().is_empty() && !has_tool_calls {
                if empty_retry_count < empty_retry_max {
                    empty_retry_count += 1;
                    if !relaxed_empty_retry
                        && (request.response_format.is_some() || request.json_strict == Some(true))
                    {
                        relaxed_empty_retry = true;
                        request.response_format = None;
                        request.json_strict = Some(false);
                        Self::simplify_json_prompt(&mut request.messages);
                    }
                    let _ = system_log::log_event(
                        &self.db_pool,
                        Some(&self.app_handle),
                        "warn",
                        "model",
                        request.run_id.as_deref(),
                        None,
                        serde_json::json!({
                            "event": "response_empty_retry",
                            "attempt": empty_retry_count,
                            "max": empty_retry_max,
                            "request_label": request.request_label.clone(),
                            "relaxed_prompt": relaxed_empty_retry,
                        }),
                    )
                    .await;
                    if empty_retry_timeout_ms > 0 {
                        tokio::time::sleep(Duration::from_millis(empty_retry_timeout_ms)).await;
                    }
                    continue;
                }
                let _ = system_log::log_event(
                    &self.db_pool,
                    Some(&self.app_handle),
                    "error",
                    "model",
                    request.run_id.as_deref(),
                    None,
                    serde_json::json!({
                        "event": "response_empty_error",
                        "attempt": empty_retry_count,
                        "max": empty_retry_max,
                        "request_label": request.request_label.clone(),
                    }),
                )
                .await;
                self.emit_empty_response_error(&request, empty_retry_count, empty_retry_max)
                    .await;
                return Err("EMPTY_RESPONSE".to_string());
            }
            let mut content = raw_content;
            let skip_sanitization = request.skip_sanitization.unwrap_or(false);
            if !request.stream && self.non_stream_sanitization_enabled().await && !skip_sanitization {
                let allow_diagnostics = request.allow_diagnostics.unwrap_or(false);
                let (user_name, assistant_name) = self.fetch_display_names().await;
                let (sanitized, removed_chars, modified, meta_reason) =
                    self.sanitize_non_stream_content(
                        &content,
                        allow_diagnostics,
                        assistant_name.as_deref(),
                        user_name.as_deref(),
                    );
                if modified {
                    let _ = system_log::log_event(
                        &self.db_pool,
                        Some(&self.app_handle),
                        "info",
                        "model",
                        request.run_id.as_deref(),
                        None,
                        serde_json::json!({
                            "event": "non_stream_sanitized",
                            "removed_chars": removed_chars,
                            "input_len": content.len(),
                            "output_len": sanitized.len(),
                        }),
                    )
                    .await;
                }
                if let Some(reason) = meta_reason {
                    let _ = system_log::log_event(
                        &self.db_pool,
                        Some(&self.app_handle),
                        "warn",
                        "model",
                        request.run_id.as_deref(),
                        None,
                        serde_json::json!({
                            "event": "meta_response_detected",
                            "reason": reason,
                            "input_len": content.len(),
                            "output_len": sanitized.len(),
                        }),
                    )
                    .await;
                }
                content = sanitized;
            }
            let has_phrase = content.to_lowercase().contains("working hypothesis");
            let (stripped, removed) = strip_working_hypothesis_prefix(&content);
            if removed {
                let _ = system_log::log_event(
                    &self.db_pool,
                    Some(&self.app_handle),
                    "info",
                    "model",
                    request.run_id.as_deref(),
                    None,
                    serde_json::json!({
                        "event": "working_hypothesis_stripped",
                        "input_len": content.len(),
                        "output_len": stripped.len(),
                        "content_hash": hash_text(&content),
                    }),
                )
                .await;
                content = stripped;
            } else if has_phrase {
                let _ = system_log::log_event(
                    &self.db_pool,
                    Some(&self.app_handle),
                    "warn",
                    "model",
                    request.run_id.as_deref(),
                    None,
                    serde_json::json!({
                        "event": "working_hypothesis_detected",
                        "content_len": content.len(),
                        "content_hash": hash_text(&content),
                    }),
                )
                .await;
            }

            return Ok((content, tool_calls));
        }
    }

    fn sanitize_non_stream_content(
        &self,
        raw: &str,
        allow_diagnostics: bool,
        assistant_name: Option<&str>,
        user_name: Option<&str>,
    ) -> (String, usize, bool, Option<&'static str>) {
        let mut memory_trigger_filter =
            crate::core::memory::inject_context::MemoryTriggerStreamFilter::new();
        let mut memory_filter = crate::core::memory::inject_context::MemoryStreamFilter::new();
        let mut reminder_filter = reminder_blocks::ReminderStreamFilter::new();
        let mut scaffold_filter = ScaffoldStreamFilter::new();
        let mut internal_filter =
            InternalStreamFilter::new(allow_diagnostics, assistant_name, user_name);

        let filtered_trigger = memory_trigger_filter.filter_chunk(raw);
        let filtered_memory = memory_filter.filter_chunk(&filtered_trigger);
        let filtered_reminder = reminder_filter.filter_chunk(&filtered_memory);
        let mut output = scaffold_filter.filter_chunk(&filtered_reminder);
        output = internal_filter.filter_chunk(&output);

        let remaining_trigger = memory_trigger_filter.finalize();
        if !remaining_trigger.is_empty() {
            let filtered_memory = memory_filter.filter_chunk(&remaining_trigger);
            let filtered_reminder = reminder_filter.filter_chunk(&filtered_memory);
            let filtered = scaffold_filter.filter_chunk(&filtered_reminder);
            let filtered = internal_filter.filter_chunk(&filtered);
            output.push_str(&filtered);
        }
        let remaining_memory = memory_filter.finalize();
        if !remaining_memory.is_empty() {
            let filtered_reminder = reminder_filter.filter_chunk(&remaining_memory);
            let filtered = scaffold_filter.filter_chunk(&filtered_reminder);
            let filtered = internal_filter.filter_chunk(&filtered);
            output.push_str(&filtered);
        }
        let reminder_remaining = reminder_filter.finalize();
        if !reminder_remaining.is_empty() {
            let filtered = scaffold_filter.filter_chunk(&reminder_remaining);
            let filtered = internal_filter.filter_chunk(&filtered);
            output.push_str(&filtered);
        }
        let scaffold_remaining = scaffold_filter.finalize();
        if !scaffold_remaining.is_empty() {
            let filtered = internal_filter.filter_chunk(&scaffold_remaining);
            output.push_str(&filtered);
        }
        let internal_remaining = internal_filter.finalize();
        if !internal_remaining.is_empty() {
            output.push_str(&internal_remaining);
        }

        let mut meta_reason: Option<&'static str> = None;
        if !allow_diagnostics {
            if let Some(reason) = detect_meta_response(&output) {
                meta_reason = Some(reason);
                let name = user_name.unwrap_or("").trim();
                let prefix = if name.is_empty() {
                    "Understood.".to_string()
                } else {
                    format!("Understood, {}.", name)
                };
                output = format!("{} What would you like to focus on next?", prefix);
            }
        }

        let removed = raw.len().saturating_sub(output.len());
        let modified = removed > 0 || raw != output || meta_reason.is_some();
        (output, removed, modified, meta_reason)
    }

    fn log_final_request(&self, request: &ChatCompletionRequest) {
        if !debug_logging_enabled() {
            return;
        }
        let payload = serde_json::to_string_pretty(request)
            .unwrap_or_else(|_| "<failed to serialize request>".to_string());
        println!("[LLM FINAL REQUEST]\n{}", payload);
    }

    fn log_prompt_messages(&self, request: &ChatCompletionRequest, phase: &str) {
        let payload = serde_json::json!({
            "phase": phase,
            "run_id": request.run_id,
            "model": request.model,
            "messages": request.messages,
        });
        let serialized = serde_json::to_string_pretty(&payload)
            .unwrap_or_else(|_| "<failed to serialize prompt>".to_string());
        println!("[LLM PROMPT:{}]\n{}", phase, serialized);
    }

    fn extract_user_query(&self, messages: &[ChatMessage]) -> Option<String> {
        if let Some(msg) = messages.iter().rev().find(|m| m.role == "user") {
            return Some(msg.content.clone());
        }
        if let Some(sys) = messages.iter().find(|m| m.role == "system") {
            if let Some(user_input) = extract_section(sys.content.as_str(), "User Input") {
                return Some(user_input);
            }
            return extract_section(sys.content.as_str(), "The User Replied");
        }
        None
    }

    fn normalize_interrogative_token(raw: &str) -> Option<String> {
        let trimmed = raw.trim().trim_matches('"').trim_matches('\'').trim_matches('#').trim_matches('$');
        if trimmed.is_empty() {
            return None;
        }
        Some(trimmed.to_lowercase())
    }

    fn ref_is_interrogative(reference: &crate::core::memory::dsl::Ref) -> bool {
        match reference {
            crate::core::memory::dsl::Ref::Handle(_) => false,
            crate::core::memory::dsl::Ref::Label(label) => {
                Self::normalize_interrogative_token(label)
                    .map(|v| is_interrogative_token(&v))
                    .unwrap_or(false)
            }
            crate::core::memory::dsl::Ref::Filter(label, filter) => {
                Self::normalize_interrogative_token(label)
                    .map(|v| is_interrogative_token(&v))
                    .unwrap_or(false)
                    || Self::normalize_interrogative_token(filter)
                        .map(|v| is_interrogative_token(&v))
                        .unwrap_or(false)
            }
            crate::core::memory::dsl::Ref::Name(name) => {
                Self::normalize_interrogative_token(name)
                    .map(|v| is_interrogative_token(&v))
                    .unwrap_or(false)
            }
        }
    }

    async fn resolve_user_message_for_memory(
        &self,
        run_id: Option<&str>,
        messages: &[ChatMessage],
    ) -> String {
        if let Some(message) = self.extract_user_query(messages) {
            return message;
        }

        let Some(run_id) = run_id else {
            return String::new();
        };

        let result = sqlx::query_scalar::<_, String>(
            "SELECT content FROM messages WHERE run_id = ? AND role = 'user' ORDER BY created_at DESC LIMIT 1",
        )
        .bind(run_id)
        .fetch_optional(&self.db_pool)
        .await;

        match result {
            Ok(Some(content)) => content,
            Ok(None) => String::new(),
            Err(e) => {
                eprintln!(
                    "[MemoryPass] Failed to load user message for run_id {}: {}",
                    run_id, e
                );
                String::new()
            }
        }
    }

    async fn resolve_conversation_id(&self, run_id: Option<&str>) -> Option<String> {
        let run_id = run_id?;
        sqlx::query_scalar("SELECT conversation_id FROM runs WHERE run_id = ?")
            .bind(run_id)
            .fetch_optional(&self.db_pool)
            .await
            .ok()
            .flatten()
    }

    async fn is_clarify_mode(&self, messages: &[ChatMessage], run_id: Option<&str>) -> bool {
        if let Some(msg) = messages.iter().rev().find(|m| m.role == "assistant") {
            let (_, tags) = crate::core::memory::inject_context::strip_system_tags(&msg.content);
            return tags.clarify && !tags.resolve;
        }

        let Some(conversation_id) = self.resolve_conversation_id(run_id).await else {
            return false;
        };

        let content = sqlx::query_scalar::<_, String>(
            "SELECT content FROM messages
             WHERE conversation_id = ? AND role = 'assistant'
               AND (metadata IS NULL OR json_extract(metadata, '$.source') != 'monologue')
             ORDER BY created_at DESC LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.db_pool)
        .await
        .ok()
        .flatten();

        let Some(content) = content else {
            return false;
        };

        let (_, tags) = crate::core::memory::inject_context::strip_system_tags(&content);
        tags.clarify && !tags.resolve
    }

    async fn fetch_clarify_context(
        &self,
        run_id: Option<&str>,
        current_assistant: &str,
        max_pairs: usize,
    ) -> String {
        use sqlx::Row;
        if max_pairs == 0 {
            return String::new();
        }

        let Some(conversation_id) = self.resolve_conversation_id(run_id).await else {
            return String::new();
        };

        let rows = sqlx::query(
            "SELECT role, content
             FROM messages
             WHERE conversation_id = ? AND role IN ('user', 'assistant')
               AND (metadata IS NULL OR json_extract(metadata, '$.source') != 'monologue')
             ORDER BY created_at DESC
             LIMIT 120",
        )
        .bind(&conversation_id)
        .fetch_all(&self.db_pool)
        .await;

        let Ok(rows) = rows else {
            return String::new();
        };

        let (current_clean, _) =
            crate::core::memory::inject_context::strip_system_tags(current_assistant);

        let mut pairs: Vec<(String, String)> = Vec::new();
        let mut pending_clarify: Option<String> = None;

        for row in rows.iter().rev() {
            let role: String = row.get("role");
            let content: String = row.get("content");

            if role == "assistant" {
                let (cleaned, tags) =
                    crate::core::memory::inject_context::strip_system_tags(&content);

                if tags.resolve {
                    if cleaned.trim() == current_clean.trim() {
                        break;
                    }
                    pairs.clear();
                    pending_clarify = None;
                    continue;
                }

                if tags.clarify {
                    pending_clarify = Some(cleaned.trim().to_string());
                }
            } else if role == "user" {
                if let Some(prompt) = pending_clarify.take() {
                    let reply = content.trim().to_string();
                    if !prompt.is_empty() && !reply.is_empty() {
                        pairs.push((prompt, reply));
                        if pairs.len() > max_pairs {
                            let overflow = pairs.len() - max_pairs;
                            pairs.drain(0..overflow);
                        }
                    }
                }
            }
        }

        if pairs.is_empty() {
            return String::new();
        }

        let mut lines = Vec::new();
        for (idx, (question, answer)) in pairs.iter().enumerate() {
            lines.push(format!("Turn {}:", idx + 1));
            lines.push(format!("Assistant: {}", question));
            lines.push(format!("User: {}", answer));
        }
        lines.join("\n")
    }

    fn is_trivial_user_context(text: &str) -> bool {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return true;
        }
        let lower = trimmed.to_lowercase();
        let trivial = [
            "ok", "okay", "k", "kk", "thanks", "thank you", "thx", "cool", "great", "nice", "yes",
            "no", "yep", "nope", "sure", "maybe", "hmm", "lol", "lmao", "alright", "got it", "fine",
            "done", "stop",
        ];
        if trivial.iter().any(|t| lower == *t) {
            return true;
        }
        let token_count = lower.split_whitespace().count();
        if token_count <= 1 && lower.len() < 8 {
            return true;
        }
        false
    }

    async fn fetch_recent_user_context(
        &self,
        run_id: Option<&str>,
        current_user_message: &str,
        max_items: usize,
    ) -> String {
        use sqlx::Row;
        if max_items == 0 {
            return String::new();
        }
        let Some(conversation_id) = self.resolve_conversation_id(run_id).await else {
            return String::new();
        };
        let rows = sqlx::query(
            "SELECT content
             FROM messages
             WHERE conversation_id = ? AND role = 'user'
             ORDER BY created_at DESC
             LIMIT 80",
        )
        .bind(&conversation_id)
        .fetch_all(&self.db_pool)
        .await;
        let Ok(rows) = rows else {
            return String::new();
        };

        let current_trimmed = current_user_message.trim();
        let mut collected: Vec<String> = Vec::new();
        for row in rows {
            let content: String = row.get("content");
            let (cleaned, _) = crate::core::memory::inject_context::strip_system_tags(&content);
            let trimmed = cleaned.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed == current_trimmed {
                continue;
            }
            if Self::is_trivial_user_context(trimmed) {
                continue;
            }
            let clipped = if trimmed.len() > 240 {
                format!("{}...", &trimmed[..240])
            } else {
                trimmed.to_string()
            };
            collected.push(clipped);
            if collected.len() >= max_items {
                break;
            }
        }

        if collected.is_empty() {
            return String::new();
        }
        collected.reverse();
        collected
            .into_iter()
            .map(|line| format!("- {}", line))
            .collect::<Vec<_>>()
            .join("\n")
    }

    async fn store_last_recalled_beliefs(
        &self,
        conversation_id: &str,
        packet: &crate::core::memory::types::MemoryPacket,
    ) {
        let mut ids: Vec<i64> = Vec::new();
        for fact in packet.facts.iter().take(MAX_CONFIRM_BELIEFS) {
            ids.push(fact.id);
        }
        for rel in packet.relations.iter().take(MAX_CONFIRM_BELIEFS) {
            if ids.len() >= MAX_CONFIRM_BELIEFS {
                break;
            }
            ids.push(rel.id);
        }

        if ids.is_empty() {
            return;
        }

        let mut guard = LAST_RECALLED_BY_CONV.lock().await;
        guard.insert(conversation_id.to_string(), ids);
    }

    async fn reinforce_last_recalled_on_confirmation(&self, run_id: Option<&str>) {
        use crate::core::memory::attention::evidence::{
            compute_evidence_weight, compute_updated_confidence,
        };
        use crate::core::memory::types::SourceType;
        use sqlx::Row;

        if self.should_skip_side_effects(run_id).await {
            return;
        }
        if !self.should_allow_memory_pass(run_id).await.allowed {
            return;
        }

        let Some(conversation_id) = self.resolve_conversation_id(run_id).await else {
            return;
        };

        let ids = {
            let guard = LAST_RECALLED_BY_CONV.lock().await;
            guard.get(&conversation_id).cloned().unwrap_or_default()
        };

        if ids.is_empty() {
            return;
        }

        let weight = compute_evidence_weight(SourceType::User) as f64;
        for belief_id in ids {
            let row = sqlx::query("SELECT confidence FROM ics_beliefs WHERE id = ? AND status = 'active'")
                .bind(belief_id)
                .fetch_optional(&self.db_pool)
                .await
                .ok()
                .flatten();
            let Some(row) = row else { continue; };
            let confidence: f64 = row.try_get("confidence").unwrap_or(1.0);
            let updated_confidence = compute_updated_confidence(confidence as f32, weight as f32) as f64;

            let _ = sqlx::query(
                "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id)
                 VALUES (?, 'user', ?, ?, ?, NULL)"
            )
            .bind(belief_id)
            .bind("user_confirmation")
            .bind("User confirmed memory")
            .bind(weight)
            .execute(&self.db_pool)
            .await;

            let _ = sqlx::query(
                "UPDATE ics_beliefs
                 SET evidence_weight_total = evidence_weight_total + ?, confidence = ?, last_evidence_at = CURRENT_TIMESTAMP
                 WHERE id = ?"
            )
            .bind(weight)
            .bind(updated_confidence)
            .bind(belief_id)
            .execute(&self.db_pool)
            .await;

            let payload_hash = hash_text(&format!("reinforce_confirmation:{}:{}", conversation_id, belief_id));
            let _ = Db { pool: self.db_pool.clone() }
                .log_memory_write(
                    Some(&conversation_id),
                    "semantic",
                    "model_client",
                    "memory_reinforce",
                    run_id,
                    None,
                    Some(&payload_hash),
                    None,
                    None,
                )
                .await;
        }
    }

    async fn reinforce_last_recalled_on_use(&self, run_id: Option<&str>) {
        use crate::core::memory::attention::evidence::{
            compute_evidence_weight, compute_updated_confidence,
        };
        use crate::core::memory::types::SourceType;
        use sqlx::Row;

        if self.should_skip_side_effects(run_id).await {
            return;
        }
        if !self.should_allow_memory_pass(run_id).await.allowed {
            return;
        }

        let Some(conversation_id) = self.resolve_conversation_id(run_id).await else {
            return;
        };

        let ids = {
            let guard = LAST_RECALLED_BY_CONV.lock().await;
            guard.get(&conversation_id).cloned().unwrap_or_default()
        };

        if ids.is_empty() {
            return;
        }

        let weight = compute_evidence_weight(SourceType::System) as f64;
        for belief_id in ids {
            let row = sqlx::query("SELECT confidence FROM ics_beliefs WHERE id = ? AND status = 'active'")
                .bind(belief_id)
                .fetch_optional(&self.db_pool)
                .await
                .ok()
                .flatten();
            let Some(row) = row else { continue; };
            let confidence: f64 = row.try_get("confidence").unwrap_or(1.0);
            let updated_confidence = compute_updated_confidence(confidence as f32, weight as f32) as f64;

            let _ = sqlx::query(
                "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id)
                 VALUES (?, 'system', ?, ?, ?, NULL)"
            )
            .bind(belief_id)
            .bind("assistant_used")
            .bind("Assistant used memory")
            .bind(weight)
            .execute(&self.db_pool)
            .await;

            let _ = sqlx::query(
                "UPDATE ics_beliefs
                 SET evidence_weight_total = evidence_weight_total + ?, confidence = ?, last_evidence_at = CURRENT_TIMESTAMP
                 WHERE id = ?"
            )
            .bind(weight)
            .bind(updated_confidence)
            .bind(belief_id)
            .execute(&self.db_pool)
            .await;

            let payload_hash = hash_text(&format!("reinforce_use:{}:{}", conversation_id, belief_id));
            let _ = Db { pool: self.db_pool.clone() }
                .log_memory_write(
                    Some(&conversation_id),
                    "semantic",
                    "model_client",
                    "memory_reinforce",
                    run_id,
                    None,
                    Some(&payload_hash),
                    None,
                    None,
                )
                .await;
        }
    }

    async fn execution_mode_for_run(&self, run_id: Option<&str>) -> Option<String> {
        let run_id = run_id?;
        let metadata: Option<String> = sqlx::query_scalar("SELECT metadata FROM runs WHERE run_id = ?")
            .bind(run_id)
            .fetch_optional(&self.db_pool)
            .await
            .ok()
            .flatten();
        let metadata = metadata?;
        let value: serde_json::Value = serde_json::from_str(&metadata).ok()?;
        value
            .get("execution_mode")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    async fn stop_state_allows_memory_write(&self, run_id: &str) -> Option<bool> {
        let conversation_id = self.resolve_conversation_id(Some(run_id)).await?;
        let state_json = Db { pool: self.db_pool.clone() }
            .get_kernel_state(&conversation_id)
            .await
            .ok()
            .flatten()?;
        let state: KernelState = serde_json::from_str(&state_json).ok()?;
        Some(state.stop_state.allowed_capabilities().memory_write)
    }

    async fn should_allow_memory_pass(&self, run_id: Option<&str>) -> MemoryPassGate {
        let run_id = match run_id {
            Some(id) if !id.trim().is_empty() => id,
            _ => {
                eprintln!("[MemoryPass] Missing run_id, skipping");
                return MemoryPassGate::deny("missing_run_id");
            }
        };
        let mode_allowed = match self.execution_mode_for_run(Some(run_id)).await.as_deref() {
            Some("chat") => true,
            Some("direct") => true,
            Some("proactive") => true,
            Some("agentic_graph") => false,
            Some(other) => {
                eprintln!("[MemoryPass] Unknown execution_mode '{}', skipping", other);
                false
            }
            None => {
                eprintln!("[MemoryPass] Missing execution_mode, skipping");
                false
            }
        };

        if !mode_allowed {
            return MemoryPassGate::deny("execution_mode_gate");
        }

        if let Some(allowed) = self.stop_state_allows_memory_write(run_id).await {
            if !allowed {
                return MemoryPassGate::deny("stop_state_blocked");
            }
        }

        let token_ok = Db { pool: self.db_pool.clone() }
            .has_memory_pass_token(run_id)
            .await
            .unwrap_or(false);
        if !token_ok {
            return MemoryPassGate::deny("token_missing");
        }

        MemoryPassGate::allow()
    }

    async fn fetch_display_names(&self) -> (Option<String>, Option<String>) {
        let settings = Db { pool: self.db_pool.clone() }.get_settings().await.ok();
        let user_name = settings
            .as_ref()
            .and_then(|s| s.user_display_name.as_deref())
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(|name| name.to_string());
        let assistant_name = settings
            .as_ref()
            .and_then(|s| s.assistant_display_name.as_deref())
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(|name| name.to_string());
        (user_name, assistant_name)
    }

    async fn memory_evidence_gating_enabled(&self) -> bool {
        Db { pool: self.db_pool.clone() }
            .get_settings()
            .await
            .ok()
            .and_then(|s| s.enable_memory_evidence_gating)
            .unwrap_or(true)
    }

    async fn response_fallback_enabled(&self) -> bool {
        Db { pool: self.db_pool.clone() }
            .get_settings()
            .await
            .ok()
            .and_then(|s| s.response_fallback_enabled)
            .unwrap_or(true)
    }

    async fn maybe_capture_memory_raw(&self, run_id: &str, memory_pass_id: &str, raw: &str) {
        let db = Db { pool: self.db_pool.clone() };
        let flag = match db.get_key("memory_capture_next").await {
            Ok(Some(v)) => v,
            _ => return,
        };
        let enabled = matches!(
            flag.trim().to_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        );
        if !enabled {
            return;
        }
        let truncated: String = raw.chars().take(4000).collect();
        let _ = db.set_key("memory_last_raw", &truncated).await;
        let _ = db
            .set_key("memory_last_raw_len", &raw.len().to_string())
            .await;
        let _ = db.set_key("memory_last_raw_id", memory_pass_id).await;
        let _ = db.set_key("memory_last_raw_run", run_id).await;
        let _ = db
            .set_key("memory_last_raw_at", &chrono::Utc::now().to_rfc3339())
            .await;
        let _ = db.set_key("memory_capture_next", "0").await;
    }

    async fn update_latency_avg(&self, bucket: &str, duration_ms: i64) {
        if duration_ms <= 0 {
            return;
        }
        let db = Db { pool: self.db_pool.clone() };
        let avg_key = format!("latency_{}_avg_ms", bucket);
        let count_key = format!("latency_{}_count", bucket);
        let prev_avg = db
            .get_key(&avg_key)
            .await
            .ok()
            .flatten()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0);
        let prev_count = db
            .get_key(&count_key)
            .await
            .ok()
            .flatten()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        let next_count = prev_count.saturating_add(1).min(1000);
        let next_avg = if prev_count == 0 {
            duration_ms as f64
        } else {
            ((prev_avg * prev_count as f64) + duration_ms as f64) / (next_count as f64)
        };
        let _ = db.set_key(&avg_key, &format!("{:.2}", next_avg)).await;
        let _ = db.set_key(&count_key, &next_count.to_string()).await;
    }

    async fn non_stream_sanitization_enabled(&self) -> bool {
        Db { pool: self.db_pool.clone() }
            .get_settings()
            .await
            .ok()
            .and_then(|s| s.stability_non_stream_sanitization)
            .unwrap_or(true)
    }

    async fn fetch_bound_handles_for_session(&self, session_id: &str) -> Vec<(String, String)> {
        use std::collections::HashSet;
        use sqlx::Row;

        let session_id = session_id.trim();
        let session_id = if session_id.is_empty() { "default" } else { session_id };
        let rows = sqlx::query(
            "SELECT sb.ref_text, e.label, sb.session_id
             FROM ics_session_bindings sb
             JOIN ics_entities e ON e.id = sb.entity_id
             WHERE sb.ref_text IN ('user', 'assistant')
               AND sb.session_id = ?
             ORDER BY sb.created_at DESC",
        )
        .bind(session_id)
        .fetch_all(&self.db_pool)
        .await
        .unwrap_or_default();

        let mut seen = HashSet::new();
        let mut handles: Vec<(String, String)> = Vec::new();
        for row in rows {
            let ref_text: String = row.get("ref_text");
            if !seen.insert(ref_text.clone()) {
                continue;
            }
            let label: String = row.get("label");
            handles.push((format!("${}", ref_text), label));
        }

        if handles.is_empty() && session_id != "default" {
            let fallback = sqlx::query(
                "SELECT sb.ref_text, e.label, sb.session_id
                 FROM ics_session_bindings sb
                 JOIN ics_entities e ON e.id = sb.entity_id
                 WHERE sb.ref_text IN ('user', 'assistant')
                   AND sb.session_id = 'default'
                 ORDER BY sb.created_at DESC",
            )
            .fetch_all(&self.db_pool)
            .await
            .unwrap_or_default();

            let mut seen = HashSet::new();
            for row in fallback {
                let ref_text: String = row.get("ref_text");
                if !seen.insert(ref_text.clone()) {
                    continue;
                }
                let label: String = row.get("label");
                handles.push((format!("${}", ref_text), label));
            }
        }

        handles
    }

    async fn should_skip_side_effects(&self, run_id: Option<&str>) -> bool {
        let run_id = match run_id {
            Some(id) => id,
            None => return false,
        };

        let status: Option<String> = sqlx::query_scalar("SELECT status FROM runs WHERE run_id = ?")
            .bind(run_id)
            .fetch_optional(&self.db_pool)
            .await
            .ok()
            .flatten();

        match status.as_deref() {
            Some("cancelled") => true,
            Some("superseded") => true,
            Some(_) => false,
            None => true, // Treat missing run as cancelled (rollback or wipe)
        }
    }

    fn inject_context_blocks(
        &self,
        messages: &mut Vec<ChatMessage>,
        recalled_info: &str,
        episodic_context: &str,
    ) {
        let recalled_replacement = if recalled_info.trim().is_empty() {
            "None".to_string()
        } else {
            recalled_info.trim().to_string()
        };
        let episodic_replacement = if episodic_context.trim().is_empty() {
            "None".to_string()
        } else {
            episodic_context.trim().to_string()
        };

        if let Some(first) = messages.first_mut() {
            if first.role == "system" {
                if first.content.contains("{{MEMORY_CONTEXT}}") {
                    first.content = first.content.replace("{{MEMORY_CONTEXT}}", &recalled_replacement);
                } else if !recalled_replacement.is_empty() {
                    first.content.push_str("\n\nRecalled Information\n");
                    first.content.push_str(&recalled_replacement);
                }

                if first.content.contains("{{EPISODIC_CONTEXT}}") {
                    first.content = first
                        .content
                        .replace("{{EPISODIC_CONTEXT}}", &episodic_replacement);
                } else if !episodic_replacement.is_empty() {
                    first.content.push_str("\n\nRecent Events\n");
                    first.content.push_str(&episodic_replacement);
                }
                return;
            }
        }

        let mut combined = String::new();
        if !recalled_replacement.is_empty() {
            combined.push_str("Recalled Information\n");
            combined.push_str(&recalled_replacement);
        }
        if !episodic_replacement.is_empty() {
            if !combined.is_empty() {
                combined.push_str("\n\n");
            }
            combined.push_str("Recent Events\n");
            combined.push_str(&episodic_replacement);
        }

        if !combined.is_empty() {
            messages.insert(0, ChatMessage {
                role: "system".to_string(),
                content: combined,
            });
        }
    }

    async fn build_injection_context(
        &self,
        query: &str,
        run_id: Option<&str>,
        reduce_injection: bool,
        repair_mode: bool,
        allow_side_effects: bool,
        base_prompt_tokens: usize,
        force_expand_memory: bool,
    ) -> InjectionContext {
        if query.trim().is_empty() {
            return InjectionContext::default();
        }

        let settings = Db { pool: self.db_pool.clone() }.get_settings().await.ok();
        let injection_policy = settings
            .as_ref()
            .map(|settings| settings.injection_policy.clone())
            .unwrap_or_else(|| "include".to_string());
        if injection_policy.trim().eq_ignore_ascii_case("exclude") {
            return InjectionContext::default();
        }

        let context_limit_tokens = settings
            .as_ref()
            .map(token_estimator::context_limit_tokens)
            .unwrap_or(DEFAULT_CONTEXT_LIMIT_TOKENS as usize);
        let safety_margin_tokens = token_estimator::safety_margin_tokens(context_limit_tokens);
        let available_budget_tokens = context_limit_tokens
            .saturating_sub(base_prompt_tokens.saturating_add(safety_margin_tokens));

        let conversation_id = self.resolve_conversation_id(run_id).await;
        let context_hash = if let Some(conv_id) = conversation_id.as_deref() {
            Db { pool: self.db_pool.clone() }
                .get_memory_context_hash(conv_id)
                .await
                .ok()
                .flatten()
        } else {
            None
        };
        let scopes = crate::core::memory::scope::scopes_for_conversation(conversation_id.as_deref());
        let intent = crate::core::memory::api::infer_query_intent(query);
        let expand_memory = (!reduce_injection && force_expand_memory)
            || should_expand_memory(&intent, query, reduce_injection);
        let mut packet_opt: Option<crate::core::memory::types::MemoryPacket> = None;
        let mut episodic_context = String::new();
        let retrieval_started = Instant::now();
        let mut cache_hit = false;

        if let (Some(conv_id), Some(ctx_hash)) = (conversation_id.as_deref(), context_hash.as_deref()) {
            if let Some((packet, cached_episodic)) =
                cache::get_cached(conv_id, ctx_hash, query, &scopes, &intent).await
            {
                packet_opt = Some(packet);
                episodic_context = cached_episodic;
                cache_hit = true;
            }
        }

        if packet_opt.is_none() {
            if allow_side_effects {
                let session_id = conversation_id.clone().unwrap_or_else(|| "default".to_string());
                let api = crate::core::memory::api::MemoryApi::new(
                    self.db_pool.clone(),
                    Some(Arc::new(self.clone())),
                    session_id,
                )
                .await;
                if let Ok(packet) = api.retrieve(query, &scopes, intent.clone()).await {
                    packet_opt = Some(packet);
                }
            } else {
                let embedding_config = settings
                    .as_ref()
                    .and_then(|s| embedding_config_from_settings(s));
                if let Ok(packet) = crate::core::memory::retrieval::retrieve_with_options(
                    query,
                    &scopes,
                    intent.clone(),
                    &self.db_pool,
                    Some(Arc::new(self.clone())),
                    embedding_config.as_ref(),
                    false,
                )
                .await
                {
                    packet_opt = Some(packet);
                }
            }
            episodic_context = self
                .build_episodic_context(query, packet_opt.as_ref(), conversation_id.as_deref())
                .await
                .unwrap_or_default();

            if let (Some(conv_id), Some(ctx_hash), Some(packet)) =
                (conversation_id.as_deref(), context_hash.as_deref(), packet_opt.as_ref())
            {
                cache::store_cached(
                    conv_id,
                    ctx_hash,
                    query,
                    &scopes,
                    &intent,
                    packet.clone(),
                    episodic_context.clone(),
                )
                .await;
            }
        }

        if repair_mode {
            if let Some(packet) = packet_opt.take() {
                packet_opt = Some(filter_packet_for_repair(packet, query));
            }
        }

        if let Some(run_id) = run_id {
            let duration_ms = retrieval_started.elapsed().as_millis() as i64;
            let _ = system_log::log_event(
                &self.db_pool,
                Some(&self.app_handle),
                "info",
                "memory",
                Some(run_id),
                None,
                serde_json::json!({
                    "event": "timing_memory_retrieval",
                    "duration_ms": duration_ms,
                    "cache_hit": cache_hit,
                    "reduce_injection": reduce_injection,
                    "allow_side_effects": allow_side_effects,
                }),
            )
            .await;
            if duration_ms > PERF_WARN_MEMORY_MS {
                let _ = system_log::log_event(
                    &self.db_pool,
                    Some(&self.app_handle),
                    "warn",
                    "perf",
                    Some(run_id),
                    None,
                    serde_json::json!({
                        "event": "performance_regression",
                        "stage": "memory_retrieval",
                        "duration_ms": duration_ms,
                        "cache_hit": cache_hit,
                    }),
                )
                .await;
            }
        }

        if allow_side_effects {
            if let (Some(packet), Some(conversation_id)) = (packet_opt.as_ref(), conversation_id.as_deref()) {
                self.store_last_recalled_beliefs(conversation_id, packet).await;
            }
        }

        let semantic_context = if reduce_injection || !expand_memory {
            packet_opt
                .as_ref()
                .map(|packet| crate::core::memory::inject_context::format_for_prompt_limited(packet, 3, 2))
                .unwrap_or_default()
        } else {
            packet_opt
                .as_ref()
                .map(crate::core::memory::inject_context::format_for_prompt)
                .unwrap_or_default()
        };

        let (summary_block, episodic_context) = if reduce_injection || !expand_memory {
            ("None".to_string(), String::new())
        } else {
            let conversation_id = conversation_id
                .as_deref()
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .unwrap_or("default");
            let want_recent = matches!(intent, crate::core::memory::types::QueryIntent::AskHistory)
                || is_temporal_query(query);
            let summary_query = if want_recent { None } else { Some(query) };
            let summary_chunks = self
                .fetch_conversation_summary_chunks(conversation_id, summary_query)
                .await
                .unwrap_or_default();
            let summary_block = if summary_chunks.trim().is_empty() {
                "None".to_string()
            } else {
                summary_chunks.trim().to_string()
            };
            (summary_block, episodic_context)
        };

        let semantic_block = if semantic_context.trim().is_empty() {
            "None"
        } else {
            semantic_context.trim()
        };

        let blackboard_block = {
            let db = Db { pool: self.db_pool.clone() };
            match db.search_blackboard(query, 4).await {
                Ok(entries) if !entries.is_empty() => entries
                    .into_iter()
                    .map(|(key, value)| format!("- {}: {}", key, value))
                    .collect::<Vec<_>>()
                    .join("\n"),
                _ => "None".to_string(),
            }
        };

        let recalled_info = format!(
            "Semantic Memory:\n{}\n\nConversation Summary Chunks:\n{}\n\nBlackboard Memory:\n{}",
            semantic_block,
            summary_block,
            blackboard_block
        );
        let mut recalled_info = recalled_info;
        let mut episodic_context = episodic_context;
        let original_recalled_tokens = token_estimator::estimate_tokens(&recalled_info);
        let original_episodic_tokens = token_estimator::estimate_tokens(&episodic_context);
        let mut trimmed_recalled = false;
        let mut trimmed_episodic = false;

        if available_budget_tokens == 0 {
            trimmed_recalled = !recalled_info.trim().is_empty();
            trimmed_episodic = !episodic_context.trim().is_empty();
            recalled_info.clear();
            episodic_context.clear();
        } else {
            let combined_tokens = original_recalled_tokens.saturating_add(original_episodic_tokens);
            if combined_tokens > available_budget_tokens {
                if original_recalled_tokens >= available_budget_tokens {
                    let (trimmed, did_trim) = trim_to_tokens(&recalled_info, available_budget_tokens);
                    recalled_info = trimmed;
                    trimmed_recalled = did_trim;
                    trimmed_episodic = !episodic_context.trim().is_empty();
                    episodic_context.clear();
                } else {
                    let remaining = available_budget_tokens.saturating_sub(original_recalled_tokens);
                    let (trimmed, did_trim) = trim_to_tokens(&episodic_context, remaining);
                    episodic_context = trimmed;
                    trimmed_episodic = did_trim;
                }
            }
        }

        if (trimmed_recalled || trimmed_episodic) && run_id.is_some() {
            let run_id = run_id.unwrap_or("none");
            let _ = system_log::log_event(
                &self.db_pool,
                Some(&self.app_handle),
                "info",
                "memory",
                Some(run_id),
                None,
                serde_json::json!({
                    "event": "memory_injection_trim",
                    "base_prompt_tokens": base_prompt_tokens,
                    "context_limit_tokens": context_limit_tokens,
                    "available_budget_tokens": available_budget_tokens,
                    "original_recalled_tokens": original_recalled_tokens,
                    "original_episodic_tokens": original_episodic_tokens,
                    "trimmed_recalled": trimmed_recalled,
                    "trimmed_episodic": trimmed_episodic,
                }),
            )
            .await;
        }

        if let Some(run_id) = run_id {
            let injection_hash = if recalled_info.trim().is_empty() && episodic_context.trim().is_empty() {
                String::new()
            } else {
                hash_text(&format!("{}||{}", recalled_info.trim(), episodic_context.trim()))
            };
            let _ = system_log::log_event(
                &self.db_pool,
                Some(&self.app_handle),
                "info",
                "memory",
                Some(run_id),
                None,
                serde_json::json!({
                    "event": "binding_hashes",
                    "memory_injection_hash": injection_hash,
                }),
            )
            .await;
        }

        InjectionContext {
            recalled_info,
            episodic_context,
            semantic_context,
        }
    }

    async fn fetch_conversation_summary_chunks(
        &self,
        conversation_id: &str,
        query: Option<&str>,
    ) -> Result<String, String> {
        let db = Db { pool: self.db_pool.clone() };
        let chunks = db
            .search_conversation_summary_chunks(conversation_id, query, 3)
            .await
            .map_err(|e| e.to_string())?;

        Ok(format_conversation_summary_chunks(&chunks))
    }

    async fn build_episodic_context(
        &self,
        query: &str,
        packet: Option<&crate::core::memory::types::MemoryPacket>,
        conversation_id: Option<&str>,
    ) -> Result<String, String> {
        if query.trim().is_empty() {
            return Ok(String::new());
        }

        let db = Db { pool: self.db_pool.clone() };
        let settings = db.get_settings().await.map_err(|e| e.to_string())?;
        if !settings.episodic_injection_enabled.unwrap_or(false) {
            return Ok(String::new());
        }

        let raw_limit = settings
            .episodic_injection_limit
            .unwrap_or(5)
            .max(1)
            .min(50) as usize;
        let limit = raw_limit.max(3).min(8);

        let fetch_limit = ((limit as i64) * 4).max(limit as i64).min(200);
        let conversation_id = conversation_id.map(str::trim).filter(|id| !id.is_empty());
        let mut events: Vec<EpisodicEvent> = db
            .search_episodic_events(
                Some(query),
                conversation_id,
                None,
                None,
                None,
                None,
                fetch_limit,
            )
            .await
            .map_err(|e| e.to_string())?;

        let mut query_event_ids = HashSet::new();
        for event in &events {
            query_event_ids.insert(event.id.clone());
        }

        let mut belief_id_set = HashSet::new();
        if let Some(packet) = packet {
            let mut belief_ids = Vec::new();
            let mut seen_beliefs = HashSet::new();
            for fact in packet
                .facts
                .iter()
                .filter(|f| f.confidence >= EPISODIC_CONTEXT_MIN_BELIEF_CONFIDENCE)
                .take(10)
            {
                if seen_beliefs.insert(fact.id) {
                    belief_ids.push(fact.id);
                    belief_id_set.insert(fact.id);
                }
            }
            for rel in packet
                .relations
                .iter()
                .filter(|r| r.confidence >= EPISODIC_CONTEXT_MIN_BELIEF_CONFIDENCE)
                .take(10)
            {
                if seen_beliefs.insert(rel.id) {
                    belief_ids.push(rel.id);
                    belief_id_set.insert(rel.id);
                }
            }

            if !belief_ids.is_empty() {
                let belief_limit = ((limit as i64) * 2).max(1).min(50);
                if let Ok(mut belief_events) =
                    crate::core::memory::retrieval::fetch_episodic_events_for_beliefs(
                        &self.db_pool,
                        &belief_ids,
                        belief_limit,
                    )
                    .await
                {
                    events.append(&mut belief_events);
                }
            }
        }

        if events.is_empty() {
            return Ok(String::new());
        }

        fn parse_timestamp(ts: &str) -> i64 {
            chrono::DateTime::parse_from_rfc3339(ts)
                .map(|dt| dt.timestamp())
                .unwrap_or(0)
        }

        let now_ts = chrono::Utc::now().timestamp();
        let mut recent_events = Vec::new();
        for event in events.iter() {
            let ts = parse_timestamp(&event.timestamp);
            if ts == 0 {
                recent_events.push(event.clone());
                continue;
            }
            let age_days = (now_ts - ts) / 86_400;
            if age_days <= EPISODIC_CONTEXT_MAX_AGE_DAYS {
                recent_events.push(event.clone());
            }
        }
        if recent_events.is_empty() {
            recent_events = events;
        }

        let mut scored_events: Vec<(f32, EpisodicEvent)> = Vec::new();
        for event in recent_events {
            let allowlisted = EPISODIC_EVENT_ALLOWLIST
                .iter()
                .any(|t| *t == event.event_type);
            if !allowlisted {
                continue;
            }

            let ts = parse_timestamp(&event.timestamp);
            let recency = if ts == 0 {
                0.5
            } else {
                let age_days = (now_ts - ts).max(0) as f32 / 86_400.0;
                let max_age = EPISODIC_CONTEXT_MAX_AGE_DAYS.max(1) as f32;
                (1.0 - (age_days / max_age)).max(0.0)
            };

            let status = event
                .payload
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let mut score = recency;
            if query_event_ids.contains(&event.id) {
                score += 0.75;
            }
            if let Some(belief_id) = event.linked_belief_id {
                if belief_id_set.contains(&belief_id) {
                    score += 0.9;
                }
            }
            match event.event_type.as_str() {
                "memory_claim_status" if status == "promoted" || status == "conflict" => {
                    score += 0.7;
                }
                "memory_claim_status" => {
                    score += 0.2;
                }
                "clarify_resolved" => {
                    score += 0.6;
                }
                "memory_claim_created" => {
                    score -= 0.3;
                }
                "memory_write_rel" => {
                    score += if status == "conflict" { 0.25 } else { 0.05 };
                }
                "memory_write_fact" => {
                    score += if status == "conflict" { 0.2 } else { 0.02 };
                }
                "message_received" | "assistant_response_finalized" => {
                    score += 0.4;
                }
                _ => {}
            }

            if score < 0.0 {
                score = 0.0;
            }

            scored_events.push((score, event));
        }

        scored_events.sort_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| parse_timestamp(&b.1.timestamp).cmp(&parse_timestamp(&a.1.timestamp)))
        });

        let mut lines = Vec::new();
        let mut seen = HashSet::new();
        let mut seen_snippets = HashSet::new();
        let mut linked_belief_ids = Vec::new();

        for (_score, event) in scored_events {
            if !seen.insert(event.id.clone()) {
                continue;
            }

            if let Some(belief_id) = event.linked_belief_id {
                linked_belief_ids.push(belief_id);
            }

            let raw_snippet = event
                .payload
                .get("summary_snippet")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let snippet = snippets::sanitize_episodic_text(raw_snippet)
                .replace('\n', " ")
                .trim()
                .to_string();

            let line = if snippet.is_empty() {
                format!("- {}: {}", event.timestamp, event.event_type)
            } else {
                format!("- {}: {}", event.timestamp, snippet)
            };
            let key_source = if snippet.is_empty() {
                event.event_type.as_str()
            } else {
                snippet.as_str()
            };
            let snippet_key = key_source
                .to_lowercase()
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>();
            if !snippet_key.is_empty() && !seen_snippets.insert(snippet_key) {
                continue;
            }
            lines.push(line);

            if lines.len() >= limit {
                break;
            }
        }

        if lines.is_empty() {
            return Ok(String::new());
        }

        if !linked_belief_ids.is_empty() {
            if let Ok(mut related) = self
                .fetch_belief_snippets(&linked_belief_ids, EPISODIC_CONTEXT_LINKED_BELIEF_LIMIT)
                .await
            {
                let mut related_lines = Vec::new();
                for snippet in related.drain(..) {
                    let cleaned = snippets::sanitize_episodic_text(&snippet)
                        .replace('\n', " ")
                        .trim()
                        .to_string();
                    if cleaned.is_empty() {
                        continue;
                    }
                    related_lines.push(format!("- {}", cleaned));
                }

                if !related_lines.is_empty() {
                    lines.push("Related beliefs:".to_string());
                    lines.extend(related_lines);
                }
            }
        }

        Ok(format!(
            "\n<EPISODIC_CONTEXT>\n{}\n</EPISODIC_CONTEXT>\n",
            lines.join("\n")
        ))
    }

    async fn fetch_belief_snippets(
        &self,
        belief_ids: &[i64],
        limit: usize,
    ) -> Result<Vec<String>, String> {
        use sqlx::Row;
        let mut seen = HashSet::new();
        let mut unique = Vec::new();
        for id in belief_ids {
            if seen.insert(*id) {
                unique.push(*id);
                if unique.len() >= limit {
                    break;
                }
            }
        }

        if unique.is_empty() {
            return Ok(Vec::new());
        }

        let mut belief_snippets = Vec::new();
        for belief_id in unique {
            let row = sqlx::query(
                "SELECT b.kind, b.polarity, fb.key, fb.value_literal, fb.subject_entity_id,
                        e.label AS subject_label, rb.rel_type, rb.direction
                 FROM ics_beliefs b
                 LEFT JOIN ics_fact_beliefs fb ON fb.belief_id = b.id
                 LEFT JOIN ics_entities e ON e.id = fb.subject_entity_id
                 LEFT JOIN ics_rel_beliefs rb ON rb.belief_id = b.id
                 WHERE b.id = ?",
            )
            .bind(belief_id)
            .fetch_optional(&self.db_pool)
            .await
            .map_err(|e| e.to_string())?;

            let Some(row) = row else { continue; };
            let kind: String = row.get("kind");
            let polarity: String = row.get("polarity");

            if kind == "fact" {
                let key: Option<String> = row.try_get("key").ok();
                let value: Option<String> = row.try_get("value_literal").ok();
                let subject_label: Option<String> = row.try_get("subject_label").ok();
                let subject_id: Option<i64> = row.try_get("subject_entity_id").ok();
                if let (Some(k), Some(v)) = (key, value) {
                    let label = subject_label.unwrap_or_else(|| {
                        subject_id
                            .map(|id| format!("entity_{}", id))
                            .unwrap_or_else(|| "entity".to_string())
                    });
                    belief_snippets.push(snippets::render_fact_snippet(&label, &k, &v, &polarity));
                } else {
                    belief_snippets.push(format!("belief {}", belief_id));
                }
            } else {
                let rel_type: Option<String> = row.try_get("rel_type").ok();
                let direction: Option<String> = row.try_get("direction").ok();
                let rel_type = rel_type.unwrap_or_else(|| "relation".to_string());
                let part_rows = sqlx::query(
                    "SELECT rp.role, e.label
                     FROM ics_rel_participants rp
                     JOIN ics_entities e ON e.id = rp.entity_id
                     WHERE rp.belief_id = ?
                     ORDER BY rp.role",
                )
                .bind(belief_id)
                .fetch_all(&self.db_pool)
                .await
                .unwrap_or_default();

                if part_rows.is_empty() {
                    belief_snippets.push(snippets::render_rel_snippet_with_direction(
                        &rel_type,
                        &[],
                        &polarity,
                        direction.as_deref(),
                    ));
                } else {
                    let detail = part_rows
                        .iter()
                        .map(|row| {
                            let role: String = row.get("role");
                            let label: String = row.get("label");
                            (role, label)
                        })
                        .collect::<Vec<_>>()
                        ;
                    belief_snippets.push(snippets::render_rel_snippet_with_direction(
                        &rel_type,
                        &detail,
                        &polarity,
                        direction.as_deref(),
                    ));
                }
            }
        }

        Ok(belief_snippets)
    }
}

fn extract_section(content: &str, title: &str) -> Option<String> {
    let needle = format!("{}\n", title);
    for section in content.split("\n\n") {
        let candidate = section.trim_start();
        if let Some(rest) = candidate.strip_prefix(&needle) {
            let mut trimmed = rest.trim();
            let begin = format!("<<<BEGIN_SECTION:{}>>>", title);
            let end = format!("<<<END_SECTION:{}>>>", title);
            if let Some(after_begin) = trimmed.strip_prefix(&begin) {
                let after_begin = after_begin.trim_start_matches('\n').trim();
                let after_begin = after_begin.trim_end();
                if let Some(without_end) = after_begin.strip_suffix(&end) {
                    trimmed = without_end.trim();
                } else if let Some(pos) = after_begin.rfind(&end) {
                    trimmed = after_begin[..pos].trim();
                } else {
                    trimmed = after_begin;
                }
            }
            if !trimmed.is_empty() && trimmed != "None" {
                return Some(trimmed.to_string());
            }
            return None;
        }
    }
    None
}

fn is_temporal_query(query: &str) -> bool {
    let q = query.to_lowercase();
    q.contains("last week")
        || q.contains("last month")
        || q.contains("last year")
        || q.contains("yesterday")
        || q.contains("earlier")
        || q.contains("previous")
        || q.contains("before")
        || q.contains("past ")
        || q.contains("recent")
        || q.contains("recently")
}

fn is_repair_request(query: &str) -> bool {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return false;
    }
    if q.starts_with("no,") || q.starts_with("no ") {
        return true;
    }
    let signals = [
        "actually",
        "that's wrong",
        "that is wrong",
        "incorrect",
        "not true",
        "correction",
        "i meant",
        "i meant to say",
        "i misspoke",
        "let me correct",
        "to clarify,",
        "to clarify ",
    ];
    signals.iter().any(|s| q.contains(s))
}

fn is_confirmation_request(query: &str) -> bool {
    let trimmed = query.trim().to_lowercase();
    if trimmed.is_empty() {
        return false;
    }
    let tokens: Vec<&str> = trimmed
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();
    if tokens.len() > 4 {
        return false;
    }
    if tokens.iter().any(|t| *t == "but" || *t == "however") {
        return false;
    }
    let signals = [
        "yes",
        "yep",
        "yeah",
        "correct",
        "right",
        "exactly",
        "true",
        "confirmed",
        "affirmative",
    ];
    tokens.iter().any(|t| signals.contains(t))
}

fn filter_packet_for_repair(
    mut packet: crate::core::memory::types::MemoryPacket,
    query: &str,
) -> crate::core::memory::types::MemoryPacket {
    let tokens = tokenize_query(query);
    if tokens.is_empty() {
        return packet;
    }

    let threshold = 0.2f32;
    packet.facts.retain(|fact| {
        let text = format!("{} {} {}", fact.entity_label, fact.key, fact.value);
        token_overlap_ratio(&tokens, &text) >= threshold
    });
    packet.relations.retain(|rel| {
        let mut text = rel.rel_type.clone();
        for participant in &rel.participants {
            text.push(' ');
            text.push_str(&participant.role);
            text.push(' ');
            text.push_str(&participant.entity_label);
        }
        token_overlap_ratio(&tokens, &text) >= threshold
    });

    let mut topic_keys = std::collections::HashSet::new();
    for fact in &packet.facts {
        topic_keys.insert(fact.topic_key.clone());
    }
    packet.conflicts.retain(|conflict| topic_keys.contains(&conflict.topic_key));

    packet
}

fn tokenize_query(query: &str) -> Vec<String> {
    query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn token_overlap_ratio(tokens: &[String], text: &str) -> f32 {
    if tokens.is_empty() {
        return 0.0;
    }
    let lower = text.to_lowercase();
    let mut hit = 0usize;
    for token in tokens {
        if lower.contains(token) {
            hit += 1;
        }
    }
    (hit as f32) / (tokens.len() as f32)
}

fn format_conversation_summary_chunks(chunks: &[ConversationSummaryChunk]) -> String {
    if chunks.is_empty() {
        return String::new();
    }

    let mut lines = Vec::new();
    for chunk in chunks {
        let start = chunk.start_ts.as_deref().map(short_date).unwrap_or_else(|| "unknown".to_string());
        let end = chunk.end_ts.as_deref().map(short_date).unwrap_or_else(|| "unknown".to_string());
        let range = if start == end {
            start
        } else {
            format!("{}..{}", start, end)
        };
        let summary = chunk.summary.trim();
        if summary.is_empty() {
            continue;
        }
        lines.push(format!("- {}: {}", range, summary));
    }
    lines.join("\n")
}

fn short_date(raw: &str) -> String {
    raw.get(0..10).unwrap_or(raw).to_string()
}

fn hash_payload(payload: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(payload.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn build_memory_pass_payload(
    user_message: &str,
    assistant_message: &str,
    user_name: Option<&str>,
    assistant_name: Option<&str>,
    semantic_context: &str,
    clarify_context: Option<&str>,
    repair_mode: bool,
    known_handles: &[(String, String)],
    candidate_block: Option<&str>,
) -> String {
    let mut payload = String::new();
    payload.push_str("USER_HANDLE: $user\n");
    if let Some(name) = user_name {
        payload.push_str(&format!("USER_NAME: {}\n", name));
    }
    payload.push_str("ASSISTANT_HANDLE: $assistant\n");
    if let Some(name) = assistant_name {
        payload.push_str(&format!("ASSISTANT_NAME: {}\n", name));
    }
    let recalled = if semantic_context.trim().is_empty() {
        "None".to_string()
    } else {
        semantic_context.trim().to_string()
    };
    payload.push_str("RECALLED_INFORMATION:\n<<<BEGIN_RECALLED_INFORMATION>>>\n");
    payload.push_str(&recalled);
    payload.push_str("\n<<<END_RECALLED_INFORMATION>>>\n");
    if let Some(ctx) = clarify_context {
        let ctx = ctx.trim();
        if !ctx.is_empty() {
            payload.push_str("CLARIFICATION_CONTEXT:\n<<<BEGIN_CLARIFICATION_CONTEXT>>>\n");
            payload.push_str(ctx);
            payload.push_str("\n<<<END_CLARIFICATION_CONTEXT>>>\n");
        }
    }
    if repair_mode {
        payload.push_str("REPAIR_MODE: true\n");
    }
    payload.push_str("KNOWN_HANDLES:\n<<<BEGIN_KNOWN_HANDLES>>>\n");
    payload.push_str(&format_known_handles(known_handles));
    payload.push_str("\n<<<END_KNOWN_HANDLES>>>\n");
    payload.push_str("MEMORY_CANDIDATES:\n<<<BEGIN_MEMORY_CANDIDATES>>>\n");
    payload.push_str(candidate_block.unwrap_or("None"));
    payload.push_str("\n<<<END_MEMORY_CANDIDATES>>>\n");
    payload.push_str("USER_MESSAGE:\n<<<BEGIN_USER_MESSAGE>>>\n");
    payload.push_str(user_message.trim());
    payload.push_str("\n<<<END_USER_MESSAGE>>>\n");
    payload.push_str("ASSISTANT_MESSAGE:\n<<<BEGIN_ASSISTANT_MESSAGE>>>\n");
    payload.push_str(assistant_message.trim());
    payload.push_str("\n<<<END_ASSISTANT_MESSAGE>>>\n");
    payload
}

fn format_known_handles(handles: &[(String, String)]) -> String {
    if handles.is_empty() {
        return "None".to_string();
    }
    handles
        .iter()
        .map(|(handle, label)| format!("{} = {}", handle, label))
        .collect::<Vec<_>>()
        .join("\n")
}

fn embedding_config_from_settings(settings: &crate::models::Settings) -> Option<EmbeddingConfig> {
    let model = settings
        .embedding_model
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())?
        .to_string();
    let base_url = match ModelClient::normalize_url(&settings.api_base_url) {
        Ok((url, _)) => url,
        Err(_) => settings.api_base_url.clone(),
    };
    Some(EmbeddingConfig {
        base_url,
        model,
        enabled: true,
    })
}

fn empty_response_retry_config_from_settings(
    settings: Option<&crate::models::Settings>,
) -> (usize, u64) {
    let max = settings
        .and_then(|s| s.empty_response_retry_max)
        .unwrap_or(3)
        .max(0) as usize;
    let timeout_ms = settings
        .and_then(|s| s.empty_response_retry_timeout_ms)
        .unwrap_or(4000)
        .max(0) as u64;
    (max, timeout_ms)
}

#[derive(Default, Debug)]
struct InterrogativeFilterStats {
    total: usize,
    kept: usize,
    dropped_relations: usize,
    dropped_interrogatives: usize,
    user_interrogative: bool,
}

impl InterrogativeFilterStats {
    fn dropped_total(&self) -> usize {
        self.dropped_relations + self.dropped_interrogatives
    }
}

fn is_interrogative_token(raw: &str) -> bool {
    matches!(
        raw,
        "what" | "who" | "where" | "when" | "why" | "how" | "which"
    )
}

fn filter_interrogative_memory_block(block: &str, user_message: &str) -> (String, InterrogativeFilterStats) {
    let mut stats = InterrogativeFilterStats::default();
    stats.user_interrogative = crate::core::kernel::is_interrogative_message(user_message);
    let mut lines_out = Vec::new();
    for raw_line in block.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        stats.total += 1;
        let parsed = crate::core::memory::dsl::parse_line(trimmed);
        if let Ok(statement) = parsed {
            if stats.user_interrogative {
                if matches!(statement, crate::core::memory::dsl::DslStatement::Rel(_)) {
                    stats.dropped_relations += 1;
                    continue;
                }
            }
            let mut has_interrogative = false;
            match statement {
                crate::core::memory::dsl::DslStatement::Rel(rel) => {
                    for (_, reference) in rel.participants {
                        if ModelClient::ref_is_interrogative(&reference) {
                            has_interrogative = true;
                            break;
                        }
                    }
                }
                crate::core::memory::dsl::DslStatement::Fact(fact) => {
                    if ModelClient::ref_is_interrogative(&fact.subject) {
                        has_interrogative = true;
                    } else if stats.user_interrogative {
                        if let Some(value_norm) = ModelClient::normalize_interrogative_token(&fact.value) {
                            if is_interrogative_token(&value_norm) {
                                has_interrogative = true;
                            }
                        }
                    }
                }
            }
            if has_interrogative {
                stats.dropped_interrogatives += 1;
                continue;
            }
        }
        lines_out.push(trimmed.to_string());
        stats.kept += 1;
    }
    (lines_out.join("\n"), stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::fs;
    use crate::db::Db;

    #[test]
    fn reasoning_json_fallback_accepts_object() {
        let raw = "{\"ok\": true}";
        let fallback = ModelClient::reasoning_json_fallback(raw);
        assert_eq!(fallback.as_deref(), Some(raw));
    }

    #[test]
    fn reasoning_json_fallback_rejects_non_object() {
        let raw = "[1, 2, 3]";
        assert!(ModelClient::reasoning_json_fallback(raw).is_none());
    }

    #[test]
    fn scaffold_stream_filter_strips_scaffold_across_chunks() {
        let mut filter = ScaffoldStreamFilter::new();
        let mut output = String::new();

        output.push_str(&filter.filter_chunk("Hello\nNext "));
        output.push_str(&filter.filter_chunk("Steps\n<<<BEGIN_SE"));
        output.push_str(&filter.filter_chunk("CTION:Next Steps>>>hidden"));
        output.push_str(&filter.filter_chunk(" content<<<END_SECTION:Next Steps>>>\nProposed "));
        output.push_str(&filter.filter_chunk("Response\n<<<BEGIN_SECTION:Proposed Response>>>"));
        output.push_str(&filter.filter_chunk("\nmore hidden<<<END_SECTION:Proposed Response>>>\nDone"));
        output.push_str(&filter.finalize());

        let lower = output.to_lowercase();
        assert!(lower.contains("hello"));
        assert!(lower.contains("done"));
        assert!(!lower.contains("next steps"));
        assert!(!lower.contains("proposed response"));
        assert!(!lower.contains("hidden"));
        assert!(!lower.contains("<<<begin_section"));
    }

    #[test]
    fn scaffold_stream_filter_flushes_partial_line() {
        let mut filter = ScaffoldStreamFilter::new();
        let chunk = "This is a long streaming line without a newline "
            .repeat(3);
        let output = filter.filter_chunk(&chunk);
        assert!(!output.is_empty());
    }

    #[test]
    fn internal_stream_filter_flushes_partial_line() {
        let mut filter = InternalStreamFilter::new(false, None, None);
        let chunk = "Streaming output should flush before newline "
            .repeat(3);
        let output = filter.filter_chunk(&chunk);
        assert!(!output.is_empty());
    }

    #[test]
    fn internal_stream_filter_blocks_diagnostic_prefix() {
        let mut filter = InternalStreamFilter::new(false, None, None);
        let chunk = "Tool Manifest: should not stream ".repeat(3);
        let output = filter.filter_chunk(&chunk);
        assert!(output.is_empty());
    }

    #[test]
    fn empty_response_retry_config_defaults() {
        let (max, timeout) = empty_response_retry_config_from_settings(None);
        assert_eq!(max, 3);
        assert_eq!(timeout, 4000);
    }

    #[tokio::test]
    async fn empty_response_retry_config_respects_settings() {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("create pool");
        let schema_sql = fs::read_to_string("src/db/schema.sql").expect("schema.sql");
        sqlx::query(&schema_sql)
            .execute(&pool)
            .await
            .expect("apply schema");
        sqlx::query(
            "INSERT INTO settings (id, schema_version, api_base_url, empty_response_retry_max, empty_response_retry_timeout_ms)
             VALUES (1, 1, 'http://localhost', 2, 500)",
        )
        .execute(&pool)
        .await
        .expect("seed settings");
        let db = Db { pool };
        let settings = db.get_settings().await.expect("get settings");
        let (max, timeout) = empty_response_retry_config_from_settings(Some(&settings));
        assert_eq!(max, 2);
        assert_eq!(timeout, 500);
    }

    #[test]
    fn should_emit_empty_response_event_requires_run_and_label() {
        let request = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![],
            stream: false,
            temperature: None,
            top_p: None,
            max_tokens: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            enable_thinking: None,
            prefill: None,
            skip_injection: None,
            skip_memory: None,
            skip_reminders: None,
            memory_expand: None,
            allow_diagnostics: None,
            json_strict: None,
            skip_sanitization: None,
            run_id: Some("run_1".to_string()),
            request_label: Some("primary_response".to_string()),
        };
        assert!(ModelClient::should_emit_empty_response_event(&request));

        let mut request_no_label = request.clone();
        request_no_label.request_label = Some("other".to_string());
        assert!(!ModelClient::should_emit_empty_response_event(&request_no_label));

        let mut request_no_run = request.clone();
        request_no_run.run_id = Some("".to_string());
        assert!(!ModelClient::should_emit_empty_response_event(&request_no_run));
    }

    #[tokio::test]
    async fn empty_stream_response_triggers_fallback() {
        use tauri::Manager;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind server");
        let addr = listener.local_addr().expect("server addr");

        let server = tokio::spawn(async move {
            for idx in 0..2 {
                let (mut socket, _) = listener.accept().await.expect("accept");
                let mut buffer = Vec::new();
                let mut temp = [0u8; 4096];
                loop {
                    let n = socket.read(&mut temp).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    buffer.extend_from_slice(&temp[..n]);
                    if buffer.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                let response_body = if idx == 0 {
                    "data: {\"choices\":[{\"delta\":{}}]}\n\ndata: [DONE]\n\n".to_string()
                } else {
                    "{\"choices\":[{\"message\":{\"content\":\"fallback\"}}]}".to_string()
                };
                let content_type = if idx == 0 {
                    "text/event-stream"
                } else {
                    "application/json"
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\n\r\n{}",
                    content_type,
                    response_body.len(),
                    response_body
                );
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("pool");
        let schema_sql = fs::read_to_string("src/db/schema.sql").expect("schema.sql");
        sqlx::query(&schema_sql)
            .execute(&pool)
            .await
            .expect("apply schema");
        let base_url = format!("http://{}/v1", addr);
        sqlx::query(
            "INSERT INTO settings (id, schema_version, api_base_url, response_fallback_enabled)
             VALUES (1, 1, ?, 1)",
        )
        .bind(&base_url)
        .execute(&pool)
        .await
        .expect("seed settings");

        let app = tauri::Builder::default()
            .build(tauri::generate_context!("tauri.conf.json"))
            .expect("build app");
        let client = ModelClient::new(pool, app.handle());

        let request = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            stream: true,
            temperature: None,
            top_p: None,
            max_tokens: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            enable_thinking: None,
            prefill: None,
            skip_injection: Some(true),
            skip_memory: Some(true),
            skip_reminders: Some(true),
            memory_expand: None,
            allow_diagnostics: None,
            json_strict: None,
            skip_sanitization: Some(true),
            run_id: None,
            request_label: Some("primary_response".to_string()),
        };

        let response = client
            .chat_with_meta_stream(&base_url, None, &request)
            .await
            .expect("fallback response");
        assert_eq!(response.content.trim(), "fallback");

        let _ = server.await;
    }
}
