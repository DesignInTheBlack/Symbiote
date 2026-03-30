use crate::core::memory::inject_context;
use crate::core::prompt_loader;
use crate::core::inner_summary;
use crate::core::system_log;
use crate::core::system_controls;
use crate::core::token_estimator;
use crate::core::tool_registry::ToolRegistry;
use crate::db::Db;
use crate::models::{
    ControllerGate,
    ControllerState,
    GoalStackItem,
    Settings,
    SelfModel,
    SelfState,
    Tool,
    WorkingMemoryBlock,
    WorkspaceFieldMeta,
    WorkspaceMeta,
    WorkspaceState,
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use once_cell::sync::Lazy;
use sqlx::Row;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Mutex, Once};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreInputKind {
    User,
    ToolResult,
    ToolError,
    SystemContext,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptLayout {
    Full,
    Compact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PromptMode {
    Normal,
    Calculator,
    SelfAudit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContextHydrationMode {
    Off,
    Shadow,
    Thin,
}

impl ContextHydrationMode {
    fn as_str(&self) -> &'static str {
        match self {
            ContextHydrationMode::Off => "off",
            ContextHydrationMode::Shadow => "shadow",
            ContextHydrationMode::Thin => "thin",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ContextHydrationPlan {
    pub mode: String,
    pub intent_tags: Vec<String>,
    pub matched_rules: Vec<String>,
    pub selected_sections: Vec<String>,
    pub skipped_sections: Vec<String>,
    pub max_sections: usize,
    pub fallback_reason: Option<String>,
}

#[derive(Clone, Debug)]
pub struct IntentDetection {
    pub tags: Vec<String>,
    pub matched_rules: Vec<String>,
}

#[derive(Clone, Debug)]
struct PromptSection {
    title: String,
    body: String,
    always: bool,
    priority: i32,
    allow_diff: bool,
    tool_hint: Option<String>,
    truncated: bool,
}

#[derive(Clone, Debug)]
struct SafetyRulesCacheEntry {
    hash: String,
    body: String,
}

#[derive(Clone, Debug)]
pub struct CorePromptInput {
    pub content: String,
    pub kind: CoreInputKind,
    pub source: String,
    pub self_awareness: bool,
    pub self_awareness_hint: bool,
    pub anchor_hits: usize,
    pub original_input: String,
    pub current_time: Option<String>,
    pub semantic_hint: Option<String>,
    pub introspection_summary: Option<String>,
    pub monologue_intent: Option<String>,
    pub monologue_digest: Option<String>,
    pub prompt_mode: Option<String>,
    pub task_phase: Option<String>,
    pub missing_slots: Option<Vec<String>>,
    pub resolution_mode: Option<String>,
    pub policy_notes: Option<String>,
    pub redirect_focus: Option<String>,
    pub allow_diagnostics: bool,
    pub world_model_snapshot: Option<String>,
    pub subject_snapshot: Option<String>,
    pub gate_decision: Option<String>,
    pub feedback_bundle: Option<String>,
    pub qualia_snapshot: Option<String>,
    pub attention_schema_summary: Option<String>,
    pub workspace_contributors_summary: Option<String>,
    pub wave_state: Option<String>,
    pub reflective_narrative: Option<String>,
    pub reflective_narrative_evidence_ids: Vec<i64>,
    pub self_report_snapshot: Option<String>,
    pub self_report_snapshot_evidence_ids: Vec<i64>,
    pub context_spine: Option<String>,
    pub hydrated_context: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PromptSectionMetric {
    pub title: String,
    pub chars: usize,
    pub lines: usize,
    pub tokens: usize,
    pub truncated: bool,
    pub budget: Option<usize>,
    pub budget_tokens: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PromptTrimEvent {
    pub title: String,
    pub original_chars: usize,
    pub trimmed_chars: usize,
    pub reason: String,
    pub hash: Option<String>,
}

pub struct CorePromptBuild {
    pub system_message: String,
    pub prompt_layout: PromptLayout,
    pub prompt_source: String,
    pub primary_prompt_hash: String,
    pub memory_prompt_hash: String,
    pub canonical_primary_hash: String,
    pub policy_canon_hash: String,
    pub override_hash: String,
    pub override_active: bool,
    pub override_mismatch: bool,
    pub override_guard_skipped: Vec<String>,
    pub workspace_hash: String,
    pub inner_summary_hash: String,
    pub rolling_summary_hash: String,
    pub capability_manifest_hash: String,
    pub workspace_present: bool,
    pub inner_summary_present: bool,
    pub rolling_summary_present: bool,
    pub introspection_present: bool,
    pub semantic_hint_present: bool,
    pub section_metrics: Vec<PromptSectionMetric>,
    pub section_hashes: HashMap<String, String>,
    pub trim_events: Vec<PromptTrimEvent>,
    pub tier_trim_summary: Option<Value>,
    pub tier0_compacted: bool,
    pub context_hydration: Option<ContextHydrationPlan>,
    pub total_chars: usize,
    pub total_tokens: usize,
    pub user_evidence_count: usize,
    pub tool_evidence_count: usize,
    pub prompt_overflow: bool,
}

#[derive(Clone, Debug)]
struct CapabilityManifestCacheEntry {
    key: String,
    block: String,
    hash: String,
}

static CAPABILITY_MANIFEST_CACHE: Lazy<Mutex<Option<CapabilityManifestCacheEntry>>> =
    Lazy::new(|| Mutex::new(None));

static SAFETY_RULES_CACHE: Lazy<Mutex<Option<SafetyRulesCacheEntry>>> =
    Lazy::new(|| Mutex::new(None));

struct PromptSectionCacheEntry {
    section: String,
}

static PROMPT_SECTION_CACHE: Lazy<Mutex<HashMap<String, PromptSectionCacheEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Clone, Debug)]
struct PromptTokenCacheEntry {
    tokens: usize,
}

static PROMPT_TOKEN_CACHE: Lazy<Mutex<HashMap<String, PromptTokenCacheEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

const PROMPT_SECTION_CACHE_LIMIT: usize = 64;
const MONOLOGUE_DIGEST_MAX_CHARS: usize = 240;
const TIER0_MAX_TOKENS: usize = 800;
const CONTEXT_BUDGET_MAX_SECTIONS: usize = 4;
const ATTENTION_SCHEMA_PROMPT_BUDGET_CHARS: usize = 400;
const WORKSPACE_CONTRIBUTORS_PROMPT_BUDGET_CHARS: usize = 400;

fn format_monologue_digest(raw: Option<&str>) -> String {
    let trimmed = raw.unwrap_or("").trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
        return "Goal: None\nPlan: None\nRisks: None\nNext Step: None".to_string();
    }
    let cleaned = trimmed.replace('\n', " ").replace('\r', " ");
    let parts = cleaned
        .split(|c| c == '.' || c == '?' || c == '!')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    let goal = parts.get(0).copied().unwrap_or(cleaned.as_str());
    let plan = parts.get(1).copied().unwrap_or(goal);
    let risks = parts
        .iter()
        .find(|s| s.to_lowercase().contains("risk") || s.to_lowercase().contains("concern"))
        .copied()
        .unwrap_or("None");
    let next_step = parts.get(2).copied().unwrap_or("None");
    let formatted = format!(
        "Goal: {}\nPlan: {}\nRisks: {}\nNext Step: {}",
        goal, plan, risks, next_step
    );
    truncate_to_budget(&formatted, MONOLOGUE_DIGEST_MAX_CHARS)
}

fn inject_primary_names(prompt: &str, user_name: &str, assistant_name: &str) -> String {
    prompt
        .replace("{user_name}", user_name)
        .replace("{assistant_name}", assistant_name)
}

pub fn split_current_time(input: &str) -> (Option<String>, String) {
    let trimmed = input.trim_start();
    if trimmed.starts_with("[Current Time:") {
        if let Some(end_idx) = trimmed.find(']') {
            let inside = &trimmed[1..end_idx];
            let time = inside
                .strip_prefix("Current Time:")
                .unwrap_or(inside)
                .trim()
                .to_string();
            let rest = trimmed[end_idx + 1..].trim_start().to_string();
            return (Some(time), rest);
        }
    }
    (None, input.to_string())
}

fn extract_context_evidence_ids(input: &str) -> Vec<i64> {
    let Ok(value) = serde_json::from_str::<Value>(input) else {
        return Vec::new();
    };
    let mut ids = Vec::new();
    if let Some(array) = value.get("evidence_event_ids").and_then(|v| v.as_array()) {
        for id in array.iter().filter_map(|v| v.as_i64()) {
            if id > 0 {
                ids.push(id);
            }
        }
    } else if let Some(id) = value.get("evidence_event_id").and_then(|v| v.as_i64()) {
        if id > 0 {
            ids.push(id);
        }
    }
    ids.sort();
    ids.dedup();
    ids
}

fn format_section(title: &str, body: &str) -> String {
    format!(
        "{title}\n<<<BEGIN_SECTION:{title}>>>\n{body}\n<<<END_SECTION:{title}>>>"
    )
}

fn hash_string(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn is_stable_section(title: &str) -> bool {
    matches!(
        title,
        "Current Time"
            | "You and User"
            | "Capabilities and Limitations"
            | "Capability Manifest"
            | "Tool Manifest"
            | "Symbiote System Overview"
            | "Self-Model Signals"
            | "Your Instructions"
    )
}

fn format_section_cached(title: &str, body: &str) -> String {
    if !is_stable_section(title) {
        return format_section(title, body);
    }
    let key = format!("{}:{}", title, hash_string(body));
    let mut cache = PROMPT_SECTION_CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = cache.get(&key) {
        return entry.section.clone();
    }
    let section = format_section(title, body);
    if cache.len() >= PROMPT_SECTION_CACHE_LIMIT {
        if let Some(first_key) = cache.keys().next().cloned() {
            cache.remove(&first_key);
        }
    }
    cache.insert(key, PromptSectionCacheEntry { section: section.clone() });
    section
}

fn estimate_tokens_cached(content: &str) -> usize {
    if content.is_empty() {
        return 0;
    }
    let key = hash_string(content);
    let mut cache = PROMPT_TOKEN_CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = cache.get(&key) {
        return entry.tokens;
    }
    let tokens = token_estimator::estimate_tokens(content);
    if cache.len() >= PROMPT_SECTION_CACHE_LIMIT {
        if let Some(first_key) = cache.keys().next().cloned() {
            cache.remove(&first_key);
        }
    }
    cache.insert(key, PromptTokenCacheEntry { tokens });
    tokens
}

fn parse_prompt_mode(raw: Option<&str>) -> PromptMode {
    match raw.unwrap_or("normal").trim().to_lowercase().as_str() {
        "calculator" | "focused_task" | "focused" => PromptMode::Calculator,
        "self_audit" | "selfaudit" => PromptMode::SelfAudit,
        _ => PromptMode::Normal,
    }
}

fn context_limit_tokens(settings: &crate::models::Settings) -> usize {
    token_estimator::context_limit_tokens(settings)
}

fn single_line(value: &str, max_chars: usize) -> String {
    let trimmed = value.replace('\n', " ").replace('\r', " ");
    let collapsed = trimmed
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if max_chars == 0 {
        return String::new();
    }
    collapsed.chars().take(max_chars).collect()
}

fn format_optional_f64(value: Option<f64>, precision: usize) -> String {
    value
        .map(|v| format!("{:.*}", precision, v))
        .unwrap_or_else(|| "None".to_string())
}

fn format_optional_f32(value: Option<f32>, precision: usize) -> String {
    value
        .map(|v| format!("{:.*}", precision, v))
        .unwrap_or_else(|| "None".to_string())
}

fn parse_qualia_snapshot_fields(raw: Option<&str>) -> (Option<String>, Option<f64>) {
    let raw = raw.unwrap_or("").trim();
    if raw.is_empty() || raw == "None" {
        return (None, None);
    }
    let mut tag: Option<String> = None;
    let mut intensity: Option<f64> = None;
    for line in raw.lines() {
        let mut parts = line.splitn(2, ':');
        let key = parts.next().unwrap_or("").trim();
        let value = parts.next().unwrap_or("").trim();
        match key {
            "dominant_tag" => {
                if !value.is_empty() && value != "none" {
                    tag = Some(value.to_string());
                }
            }
            "dominant_intensity" => {
                if let Ok(parsed) = value.parse::<f64>() {
                    intensity = Some(parsed);
                }
            }
            _ => {}
        }
    }
    (tag, intensity)
}

fn parse_wave_state_fields(raw: Option<&str>) -> (Option<f32>, Option<f32>) {
    let raw = raw.unwrap_or("").trim();
    if raw.is_empty() || raw == "None" {
        return (None, None);
    }
    let parsed: Value = match serde_json::from_str(raw) {
        Ok(value) => value,
        Err(_) => return (None, None),
    };
    let coherence = parsed
        .get("coherence")
        .and_then(|value| value.as_f64())
        .map(|value| value as f32);
    let fragmentation = parsed
        .get("fragmentation")
        .and_then(|value| value.as_f64())
        .map(|value| value as f32);
    (coherence, fragmentation)
}

fn collect_workspace_meta_evidence_ids(meta: &WorkspaceMeta) -> Vec<i64> {
    let mut ids: Vec<i64> = Vec::new();
    let mut push_ids = |list: &[i64]| {
        for id in list.iter().copied() {
            if id > 0 {
                ids.push(id);
            }
        }
    };
    if let Some(field) = meta.goal_thread.as_ref() {
        push_ids(&field.evidence_event_ids);
    }
    if let Some(field) = meta.current_focus.as_ref() {
        push_ids(&field.evidence_event_ids);
    }
    if let Some(field) = meta.focus_rationale.as_ref() {
        push_ids(&field.evidence_event_ids);
    }
    for item in meta.open_questions.iter() {
        push_ids(&item.evidence_event_ids);
    }
    for item in meta.working_set_topics.iter() {
        push_ids(&item.evidence_event_ids);
    }
    for hypothesis in meta.active_hypotheses.iter() {
        push_ids(&hypothesis.evidence_event_ids);
    }
    if let Some(runtime) = meta.runtime.as_ref().and_then(|v| v.as_object()) {
        for key in [
            "autobiographical_summary",
            "self_report_snapshot",
            "planner",
            "evidence_health",
            "goal_reconciliation",
            "strategy_audit",
        ]
        .iter()
        {
            if let Some(obj) = runtime.get(*key).and_then(|v| v.as_object()) {
                if let Some(list) = obj.get("evidence_event_ids").and_then(|v| v.as_array()) {
                    for id in list.iter().filter_map(|v| v.as_i64()) {
                        if id > 0 {
                            ids.push(id);
                        }
                    }
                }
            }
        }
    }
    ids.sort();
    ids.dedup();
    ids
}

fn extract_unified_state_evidence_ids(value: &Value) -> Vec<i64> {
    let mut ids: Vec<i64> = Vec::new();
    let top_list = value.get("top_evidence_event_ids").and_then(|v| v.as_array());
    let list = match top_list {
        Some(arr) if !arr.is_empty() => Some(arr),
        _ => value.get("evidence_event_ids").and_then(|v| v.as_array()),
    };
    if let Some(list) = list {
        for item in list.iter() {
            if let Some(id) = item.as_i64() {
                if id > 0 {
                    ids.push(id);
                }
            }
        }
    }
    ids.sort();
    ids.dedup();
    ids
}

fn format_identity_anchor(
    assistant_name: &str,
    role: &str,
    current_goal: &str,
    last_response_summary: &str,
    self_model_hash: &str,
    qualia_tag: Option<&str>,
    qualia_intensity: Option<f64>,
    wave_coherence: Option<f32>,
    wave_fragmentation: Option<f32>,
) -> String {
    let goal = if current_goal.trim().is_empty() {
        "None".to_string()
    } else {
        single_line(current_goal, 160)
    };
    let summary = if last_response_summary.trim().is_empty() {
        "None".to_string()
    } else {
        single_line(last_response_summary, 160)
    };
    let hash = if self_model_hash.trim().is_empty() {
        "None".to_string()
    } else {
        single_line(self_model_hash, 64)
    };
    let qualia_tag = qualia_tag
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("unknown");
    let qualia_intensity = qualia_intensity
        .map(|value| format!("{:.2}", value))
        .unwrap_or_else(|| "unknown".to_string());
    let wave_coherence = wave_coherence
        .map(|value| format!("{:.2}", value))
        .unwrap_or_else(|| "unknown".to_string());
    let wave_fragmentation = wave_fragmentation
        .map(|value| format!("{:.2}", value))
        .unwrap_or_else(|| "unknown".to_string());
    format!(
        "I am {}, speaking as the {}. My memory access is evidence-gated. Current focus: {}. Last response summary: {}. Dominant qualia: {} (intensity {}). Wave coherence: {}, fragmentation: {}. Self-model hash: {}.",
        assistant_name.trim(),
        role.trim(),
        goal,
        summary,
        qualia_tag,
        qualia_intensity,
        wave_coherence,
        wave_fragmentation,
        hash
    )
}

fn format_system_overview() -> String {
    [
        "Symbiote System Overview:",
        "- Frontend (Tauri + React UI): chat interface, system state panel, prompt queue, monologue view, TTS controls.",
        "- Kernel: orchestrates runs, candidate arbitration, monologue (FTS/DS), tool dispatch, memory writes, and state updates.",
        "- Prompt builder: assembles system prompt sections (identity anchor, working memory, monologue intent, summaries, evidence) with trimming/budgeting.",
        "- Model client: executes LLM calls (streaming + non-streaming), applies sanitization, emits stream events.",
        "- Memory system: episodic events, semantic/graph memory, working memory block, inner/rolling summaries, evidence gating.",
        "- Scheduler: background cycles (monologue ticks, memory passes, consolidation).",
        "- Tools: registered tool calls via tool registry.",
        "- DB (SQLite): stores messages, runs, system logs, memory records, monologue entries, settings, prompt queue.",
        "- Voice/TTS: external service for speech I/O; UI streams audio.",
        "Only surface this overview when the user explicitly asks about system architecture.",
    ]
    .join("\n")
}

fn format_architecture_map() -> String {
    [
        "Architecture Map (internal):",
        "- Organism loop: stress, arousal, fatigue, valence, social alignment.",
        "- Qualia feedback loop: label aggregation, reward events, dominant tag.",
        "- Subject controller: gate decisions (ALLOW / VERIFY / DENY / ALLOW_WITH_NOTICE).",
        "- FeedbackBundle: pre-validated signals injected every turn.",
        "- Tick loop: scheduler cadences for monologue, memory, consolidation, identity audit.",
        "- Self-model controller: persona, goals, evidence coverage, identity thread.",
        "- Workspace: ignition score, broadcast references, current focus.",
        "Only surface this map when the user explicitly asks about system architecture.",
    ]
    .join("\n")
}

fn goal_status_complete(status: Option<&str>) -> bool {
    let Some(status) = status else {
        return false;
    };
    matches!(
        status.trim().to_lowercase().as_str(),
        "done" | "complete" | "completed" | "finished"
    )
}

fn active_goal_stack_parts(goal_stack: &[GoalStackItem]) -> Option<(String, Option<String>)> {
    for item in goal_stack.iter() {
        let goal = item.goal.trim();
        if goal.is_empty() {
            continue;
        }
        if goal_status_complete(item.status.as_deref()) {
            continue;
        }
        let step = item
            .steps
            .get(item.current_step_index)
            .map(|s| s.text.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        return Some((goal.to_string(), step));
    }
    None
}

fn active_goal_step_links(goal_stack: &[GoalStackItem]) -> Option<(Vec<i64>, Vec<i64>)> {
    for item in goal_stack.iter() {
        let goal = item.goal.trim();
        if goal.is_empty() {
            continue;
        }
        if goal_status_complete(item.status.as_deref()) {
            continue;
        }
        if let Some(step) = item.steps.get(item.current_step_index) {
            return Some((step.evidence_event_ids.clone(), step.belief_ids.clone()));
        }
        return Some((item.evidence_event_ids.clone(), item.belief_ids.clone()));
    }
    None
}

fn format_goal_stack_focus(goal_stack: &[GoalStackItem]) -> Option<String> {
    active_goal_stack_parts(goal_stack).map(|(goal, step)| {
        if let Some(step) = step {
            format!("Goal: {} | Step: {}", goal, step)
        } else {
            format!("Goal: {}", goal)
        }
    })
}

fn format_working_memory_block(
    working: Option<&WorkingMemoryBlock>,
    workspace_state: Option<&WorkspaceState>,
    controller_state: Option<&ControllerState>,
    disable_working_hypothesis: bool,
) -> String {
    let _ = disable_working_hypothesis;
    let mut focus = working
        .and_then(|w| w.focus.clone())
        .filter(|s| !s.trim().is_empty());
    if focus.is_none() {
        focus = workspace_state
            .and_then(|s| s.current_focus.clone())
            .filter(|s| !s.trim().is_empty());
    }
    if focus.is_none() {
        focus = workspace_state
            .and_then(|s| format_goal_stack_focus(&s.goal_stack))
            .filter(|s| !s.trim().is_empty());
    }
    if focus.is_none() {
        focus = workspace_state
            .and_then(|s| s.goal_thread.clone())
            .filter(|s| !s.trim().is_empty());
    }

    let mut open_questions = working
        .map(|w| w.open_questions.clone())
        .unwrap_or_default();
    if open_questions.is_empty() {
        open_questions = workspace_state
            .map(|s| s.open_questions.clone())
            .unwrap_or_default();
    }
    let open_questions = open_questions
        .into_iter()
        .filter(|q| !q.trim().is_empty())
        .take(3)
        .collect::<Vec<_>>();

    let mut active_hypotheses = working
        .map(|w| w.active_hypotheses.clone())
        .unwrap_or_default();
    if active_hypotheses.is_empty() {
        if let Some(state) = workspace_state {
            active_hypotheses = state
                .active_hypotheses
                .iter()
                .filter_map(|h| {
                    let trimmed = h.text.trim();
                    if trimmed.is_empty() {
                        None
                    } else if h.speculative {
                        Some(format!("{} (speculative=true)", trimmed))
                    } else {
                        Some(trimmed.to_string())
                    }
                })
                .collect();
        }
    }
    let active_hypotheses = active_hypotheses
        .into_iter()
        .filter(|h| !h.trim().is_empty())
        .take(3)
        .collect::<Vec<_>>();

    let next_action = working
        .and_then(|w| w.next_action.clone())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "None".to_string());

    let confidence = working
        .and_then(|w| w.confidence)
        .or_else(|| controller_state.map(|c| c.confidence));
    let drift_score = working
        .and_then(|w| w.drift_score)
        .or_else(|| controller_state.map(|c| c.drift_score));

    let confidence_text = confidence.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "None".to_string());
    let drift_text = drift_score.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "None".to_string());

    let focus_line = format!("focus: {}", focus.unwrap_or_else(|| "None".to_string()));
    let questions_line = if open_questions.is_empty() {
        "open_questions: None".to_string()
    } else {
        format!("open_questions: {}", open_questions.join(" | "))
    };
    let hypotheses_line = if active_hypotheses.is_empty() {
        "active_hypotheses: None".to_string()
    } else {
        format!("active_hypotheses: {}", active_hypotheses.join(" | "))
    };
    let goal_links_line = workspace_state
        .and_then(|state| active_goal_step_links(&state.goal_stack))
        .map(|(evidence_ids, belief_ids)| {
            let mut parts = Vec::new();
            if !evidence_ids.is_empty() {
                let joined = evidence_ids
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                parts.push(format!("evidence_event_ids=[{}]", joined));
            }
            if !belief_ids.is_empty() {
                let joined = belief_ids
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                parts.push(format!("belief_ids=[{}]", joined));
            }
            if parts.is_empty() {
                "goal_links: None".to_string()
            } else {
                format!("goal_links: {}", parts.join(" | "))
            }
        })
        .unwrap_or_else(|| "goal_links: None".to_string());

    format!(
        "{}\n{}\n{}\nnext_action: {}\nconfidence: {}\ndrift_score: {}\n{}",
        focus_line,
        questions_line,
        hypotheses_line,
        next_action,
        confidence_text,
        drift_text,
        goal_links_line
    )
}

fn format_workspace_snapshot(state: Option<&WorkspaceState>, disable_working_hypothesis: bool) -> String {
    let Some(state) = state else {
        return "None".to_string();
    };
    let _ = disable_working_hypothesis;
    let mut lines = Vec::new();
    if let Some(goal) = state.goal_thread.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        lines.push(format!("goal_thread: {}", goal));
    }
    if let Some((goal, step)) = active_goal_stack_parts(&state.goal_stack) {
        lines.push(format!("goal_stack_active: {}", goal));
        if let Some(step) = step {
            lines.push(format!("goal_stack_step: {}", step));
        }
    }
    if let Some(focus) = state.current_focus.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        lines.push(format!("current_focus: {}", focus));
    }
    let questions = state
        .open_questions
        .iter()
        .map(|q| q.trim())
        .filter(|q| !q.is_empty())
        .take(3)
        .collect::<Vec<_>>();
    if !questions.is_empty() {
        lines.push(format!("open_questions: {}", questions.join(" | ")));
    }
    let hypotheses = state
        .active_hypotheses
        .iter()
        .filter_map(|h| {
            let trimmed = h.text.trim();
            if trimmed.is_empty() {
                None
            } else if h.speculative {
                Some(format!("{} (speculative=true)", trimmed))
            } else {
                Some(trimmed.to_string())
            }
        })
        .take(3)
        .collect::<Vec<_>>();
    if !hypotheses.is_empty() {
        lines.push(format!("active_hypotheses: {}", hypotheses.join(" | ")));
    }
    if lines.is_empty() {
        "None".to_string()
    } else {
        lines.join("\n")
    }
}

fn safety_rules_cached(body: &str, prompt_mode: PromptMode) -> String {
    let trimmed = body.trim();
    if prompt_mode == PromptMode::SelfAudit {
        return trimmed.to_string();
    }
    let hash = hash_string(trimmed);
    let mut cache = SAFETY_RULES_CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = cache.as_ref() {
        if entry.hash == hash {
            return entry.body.clone();
        }
    }
    let body = trimmed.to_string();
    *cache = Some(SafetyRulesCacheEntry { hash, body: body.clone() });
    body
}

fn tool_hint_for_section(title: &str) -> Option<String> {
    match title {
        "Inner Summary" => Some("get_inner_summary".to_string()),
        "Rolling Summary" => Some("get_rolling_summary".to_string()),
        "Workspace Snapshot" => Some("get_workspace_state".to_string()),
        "World Model" => Some("get_world_model_snapshot".to_string()),
        "Subject Snapshot" => Some("get_plan_summary".to_string()),
        "Unified Self" => Some("get_unified_self".to_string()),
        "Autobiographical Context" => Some("get_autobiographical_context".to_string()),
        _ => None,
    }
}

fn context_mode_from_settings(settings: &Settings) -> ContextHydrationMode {
    match settings
        .context_hydration_mode
        .as_deref()
        .unwrap_or("shadow")
        .trim()
        .to_lowercase()
        .as_str()
    {
        "off" => ContextHydrationMode::Off,
        "thin" => ContextHydrationMode::Thin,
        _ => ContextHydrationMode::Shadow,
    }
}

fn is_context_section(title: &str) -> bool {
    matches!(
        title,
        "World Model"
            | "Subject Snapshot"
            | "Rolling Summary"
            | "Inner Summary"
            | "Workspace Snapshot"
            | "Capabilities"
            | "Unified Self"
            | "Autobiographical Context"
            | "Reflective Narrative"
    )
}

pub(crate) fn detect_context_intent(
    content: &str,
    self_awareness: bool,
    self_awareness_hint: bool,
) -> IntentDetection {
    let mut tags: Vec<String> = Vec::new();
    let mut matched_rules: Vec<String> = Vec::new();
    if self_awareness {
        tags.push("self_awareness".to_string());
        matched_rules.push("self_awareness_flag".to_string());
    } else if self_awareness_hint {
        tags.push("self_awareness".to_string());
        matched_rules.push("self_awareness_hint".to_string());
    }

    let lower = content.to_lowercase();
    let recall_triggers = [
        "what did i say",
        "what did we say",
        "earlier",
        "previous",
        "before",
        "last time",
        "remind me",
        "you said",
        "we said",
        "conversation",
        "recall",
        "history",
    ];
    if recall_triggers.iter().any(|t| lower.contains(t)) {
        tags.push("recall".to_string());
        matched_rules.push("recall_trigger".to_string());
    }

    let memory_triggers = [
        "remember",
        "memory",
        "preference",
        "preferences",
        "prefer",
        "favorite",
        "favourite",
        "likes",
        "dislike",
        "who am i",
        "what do you know about me",
    ];
    if memory_triggers.iter().any(|t| lower.contains(t)) {
        tags.push("memory".to_string());
        matched_rules.push("memory_trigger".to_string());
    }

    let world_triggers = [
        "world model",
        "belief graph",
        "beliefs about",
        "facts about",
        "what's true",
        "what is true",
        "state of the world",
        "entities",
        "relationships",
        "conflicts",
    ];
    if world_triggers.iter().any(|t| lower.contains(t)) {
        tags.push("world_model".to_string());
        matched_rules.push("world_model_trigger".to_string());
    }

    let plan_triggers = [
        "plan",
        "steps",
        "strategy",
        "roadmap",
        "goal",
        "goals",
        "progress",
        "next step",
        "milestone",
    ];
    if plan_triggers.iter().any(|t| lower.contains(t)) {
        tags.push("planning".to_string());
        matched_rules.push("planning_trigger".to_string());
    }

    let uncertainty_triggers = [
        "not sure",
        "unsure",
        "uncertain",
        "confidence",
        "probability",
        "maybe",
        "likely",
        "uncertainty",
    ];
    if uncertainty_triggers.iter().any(|t| lower.contains(t)) {
        tags.push("uncertainty".to_string());
        matched_rules.push("uncertainty_trigger".to_string());
    }

    let tool_triggers = [
        "tool",
        "tools",
        "search",
        "browse",
        "web",
        "lookup",
        "research",
        "call api",
        "run tool",
    ];
    if tool_triggers.iter().any(|t| lower.contains(t)) {
        tags.push("tools".to_string());
        matched_rules.push("tools_trigger".to_string());
    }

    let loop_triggers = [
        "loop",
        "again",
        "repeat",
        "same answer",
        "circling",
        "stuck",
        "cycle",
    ];
    if loop_triggers.iter().any(|t| lower.contains(t)) {
        tags.push("looping".to_string());
        matched_rules.push("looping_trigger".to_string());
    }

    let debug_triggers = [
        "error",
        "bug",
        "broken",
        "crash",
        "not working",
        "issue",
        "failed",
        "failure",
    ];
    if debug_triggers.iter().any(|t| lower.contains(t)) {
        tags.push("debug".to_string());
        matched_rules.push("debug_trigger".to_string());
    }

    let self_awareness_triggers = [
        "self aware",
        "self-aware",
        "self awareness",
        "self-awareness",
        "consciousness",
        "conscious",
        "internal state",
        "inner state",
        "your system",
        "self model",
        "self-model",
    ];
    if self_awareness_triggers.iter().any(|t| lower.contains(t)) {
        tags.push("self_awareness".to_string());
        matched_rules.push("self_awareness_trigger".to_string());
    }

    let capability_triggers = [
        "capabilities",
        "what can you do",
        "what do you do",
        "your capabilities",
        "controls",
        "system controls",
        "audit",
        "trace",
        "decision path",
        "trace view",
        "gate decisions",
    ];
    if capability_triggers.iter().any(|t| lower.contains(t)) {
        tags.push("capabilities".to_string());
        matched_rules.push("capabilities_trigger".to_string());
    }

    let partnership_triggers = [
        "partner",
        "collaborate",
        "co-pilot",
        "copilot",
        "together",
        "team up",
    ];
    if partnership_triggers.iter().any(|t| lower.contains(t)) {
        tags.push("partnership".to_string());
        matched_rules.push("partnership_trigger".to_string());
    }

    let introspection_triggers = [
        "why did you decide",
        "how did you decide",
        "reasoning",
        "rationale",
        "decision path",
        "self audit",
        "self-audit",
        "trace",
    ];
    if introspection_triggers.iter().any(|t| lower.contains(t)) {
        tags.push("introspection".to_string());
        matched_rules.push("introspection_trigger".to_string());
    }

    // Dedup while preserving order.
    let mut deduped: Vec<String> = Vec::new();
    for tag in tags {
        if !deduped.iter().any(|existing| existing == &tag) {
            deduped.push(tag);
        }
    }
    let mut deduped_rules: Vec<String> = Vec::new();
    for rule in matched_rules {
        if !deduped_rules.iter().any(|existing| existing == &rule) {
            deduped_rules.push(rule);
        }
    }

    IntentDetection {
        tags: deduped,
        matched_rules: deduped_rules,
    }
}

pub(crate) fn select_context_sections(intent_tags: &[String]) -> Vec<String> {
    let mut selected: Vec<String> = Vec::new();
    let has_tag = |needle: &str| intent_tags.iter().any(|t| t == needle);
    if has_tag("world_model") {
        selected.push("World Model".to_string());
    }
    if has_tag("planning") {
        selected.push("Subject Snapshot".to_string());
        selected.push("Workspace Snapshot".to_string());
    }
    if has_tag("recall") {
        selected.push("Rolling Summary".to_string());
        selected.push("Inner Summary".to_string());
    }
    if has_tag("self_awareness") && !selected.iter().any(|s| s == "Inner Summary") {
        selected.push("Inner Summary".to_string());
    }
    if has_tag("self_awareness") {
        selected.push("Unified Self".to_string());
        selected.push("Autobiographical Context".to_string());
        selected.push("Reflective Narrative".to_string());
    }
    if has_tag("introspection") && !selected.iter().any(|s| s == "Unified Self") {
        selected.push("Unified Self".to_string());
    }
    if has_tag("capabilities") || has_tag("introspection") || has_tag("self_awareness") {
        selected.push("Capabilities".to_string());
    }
    let mut deduped: Vec<String> = Vec::new();
    for item in selected {
        if !deduped.iter().any(|existing| existing == &item) {
            deduped.push(item);
        }
    }
    deduped.truncate(CONTEXT_BUDGET_MAX_SECTIONS);
    deduped
}

fn fallback_context_sections(content: &str) -> (Vec<String>, Option<String>) {
    let _ = content;
    (Vec::new(), None)
}

pub(crate) fn compute_context_hydration(
    content: &str,
    self_awareness: bool,
    self_awareness_hint: bool,
) -> (IntentDetection, Vec<String>, Option<String>) {
    let intent = detect_context_intent(content, self_awareness, self_awareness_hint);
    let mut selected = select_context_sections(&intent.tags);
    let mut fallback_reason = None;
    if intent.tags.is_empty() || selected.is_empty() {
        let (fallback_sections, reason) = fallback_context_sections(content);
        if !fallback_sections.is_empty() {
            selected = fallback_sections;
            fallback_reason = reason;
        }
    }
    (intent, selected, fallback_reason)
}

fn context_section_budget_chars(title: &str, mode: ContextHydrationMode) -> Option<usize> {
    if mode == ContextHydrationMode::Thin && is_context_section(title) {
        return Some(600);
    }
    None
}

fn anchor_floor_chars_for_mode(title: &str, mode: ContextHydrationMode) -> Option<usize> {
    if mode == ContextHydrationMode::Thin && is_context_section(title) {
        return None;
    }
    anchor_floor_chars(title)
}

fn section_priority_for_mode(mode: PromptMode, title: &str) -> i32 {
    match mode {
        PromptMode::Calculator => match title {
            "Task Context"
            | "Tool Manifest"
            | "SYMBIOTE_PHILOSOPHY"
            | "SYMBIOTE_POLICY_SUMMARY"
            | "Audit Log"
            | "Self-Report Instruction" => 1,
            _ => 9,
        },
        PromptMode::SelfAudit => match title {
            "Response Style"
            | "Identity Anchor"
            | "Safety Rules"
            | "User Input"
            | "Working Memory"
            | "SYMBIOTE_PHILOSOPHY"
            | "SYMBIOTE_POLICY_SUMMARY" => 1,
            "Symbiote System Overview" => 2,
            "Tool Availability" => 2,
            "Self-Model Signals" => 2,
            "Audit Log" => 2,
            "Self-Report Instruction" => 2,
            "Unified Self" => 2,
            "Self-State" | "Controller State" => 2,
            "Subject Snapshot" | "Gate Decision" | "Feedback Bundle" => 3,
            "World Model" => 2,
            "Qualia Snapshot" => 2,
            "Reflective Narrative" => 2,
            "Gate Feedback" => 4,
            "Capabilities and Limitations" => 5,
            "Capabilities" => 5,
            "Wave State" => 2,
            "Attention Schema" => 2,
            "Workspace Contributors" => 2,
            "Workspace Snapshot" => 3,
            "Inner Summary" => 2,
            "Rolling Summary" => 3,
            "Episodic Context" => 5,
            "Autobiographical Context" => 4,
            _ => 9,
        },
        PromptMode::Normal => match title {
            "Monologue Intent"
            | "Response Style"
            | "SYMBIOTE_PHILOSOPHY"
            | "SYMBIOTE_POLICY_SUMMARY" => 1,
            "Symbiote System Overview" => 2,
            "Tool Availability" => 2,
            "Self-Model Signals" => 2,
            "Audit Log" => 2,
            "Self-Report Instruction" => 2,
            "Unified Self" => 2,
            "Subject Snapshot" | "Gate Decision" | "Feedback Bundle" => 3,
            "World Model" => 3,
            "Qualia Snapshot" => 2,
            "Reflective Narrative" => 2,
            "Gate Feedback" => 4,
            "Capabilities and Limitations" => 5,
            "Capabilities" => 5,
            "Workspace Snapshot" => 3,
            "Inner Summary" => 2,
            "Wave State" => 2,
            "Attention Schema" => 2,
            "Workspace Contributors" => 2,
            "Semantic Hint" => 6,
            "Memory Context" => 4,
            "Episodic Context" => 5,
            "Autobiographical Context" => 4,
            "Rolling Summary" => 3,
            "Self-State" | "Controller State" => 8,
            "Capability Manifest" => 9,
            "KV Memory" => 10,
            "Tool Manifest" => 11,
            "Telemetry Snapshot" => 12,
            _ => 8,
        },
    }
}

fn section_budget_chars(title: &str) -> Option<usize> {
    match title {
        "Identity Anchor" => Some(600),
        "Safety Rules" => Some(2000),
        "User Input" => Some(1600),
        "Working Memory" => Some(900),
        "Response Style" => Some(400),
        "SYMBIOTE_PHILOSOPHY" => Some(800),
        "SYMBIOTE_POLICY_SUMMARY" => Some(800),
        "Symbiote System Overview" => Some(1400),
        "Tool Availability" => Some(500),
        "Self-Model Signals" => Some(800),
        "Audit Log" => Some(600),
        "Self-Report Instruction" => Some(400),
        "Monologue Intent" => Some(800),
        "Task Context" => Some(900),
        "Workspace Snapshot" => Some(1200),
        "Inner Summary" => Some(800),
        "Rolling Summary" => Some(1200),
        "World Model" => Some(1400),
        "Subject Snapshot" => Some(1600),
        "Gate Decision" => Some(800),
        "Feedback Bundle" => Some(900),
        "Qualia Snapshot" => Some(1000),
        "Wave State" => Some(900),
        "Attention Schema" => Some(ATTENTION_SCHEMA_PROMPT_BUDGET_CHARS.max(900)),
        "Workspace Contributors" => Some(WORKSPACE_CONTRIBUTORS_PROMPT_BUDGET_CHARS.max(900)),
        "Unified Self" => Some(1200),
        "Autobiographical Context" => Some(1200),
        "Reflective Narrative" => Some(900),
        "Semantic Hint" => Some(600),
        "Memory Context" => Some(2000),
        "Episodic Context" => Some(1600),
        "Current Time" => Some(120),
        "You and User" => Some(800),
        "User Evidence IDs" => Some(800),
        "Tool Evidence IDs" => Some(800),
        "Conversation So Far" => Some(2800),
        "Internal Focus" => Some(1800),
        "Verified Workspace" => Some(1800),
        "Speculative Workspace" => Some(1200),
        "Inner Dialogue Summary" => Some(1200),
        "Introspection Summary" => Some(1200),
        "Recalled Information" => Some(2200),
        "Recent Events" => Some(1600),
        "Capabilities and Limitations" => Some(1200),
        "Capabilities" => Some(1200),
        "Capability Manifest" => Some(1600),
        "Tool Manifest" => Some(1600),
        "KV Memory" => Some(800),
        "Gate Feedback" => Some(1200),
        "Telemetry Snapshot" => Some(800),
        "Self-State" => Some(800),
        "Controller State" => Some(800),
        "Identity Thread" => Some(600),
        "Prompt Mode" => Some(200),
        "Task Phase" => Some(200),
        "Missing Slots" => Some(400),
        "Resolution Mode" => Some(200),
        "Policy Notes" => Some(600),
        "Your Instructions" => Some(2000),
        "You said" => Some(1200),
        "The User Replied" => Some(1200),
        _ => None,
    }
}

fn is_protected_section(title: &str) -> bool {
    matches!(
        title,
        "Response Style"
            | "Safety Rules"
            | "User Input"
            | "Identity Anchor"
            | "SYMBIOTE_PHILOSOPHY"
            | "SYMBIOTE_POLICY_SUMMARY"
            | "Symbiote System Overview"
            | "Tool Availability"
            | "Working Memory"
            | "Self-Model Signals"
            | "Audit Log"
            | "Self-Report Instruction"
            | "Identity Thread"
            | "Subject Snapshot"
            | "Gate Decision"
            | "Task Context"
            | "Monologue Digest"
            | "Inner Summary"
            | "Rolling Summary"
            | "World Model"
            | "Memory Context"
            | "Workspace Snapshot"
            | "Qualia Snapshot"
            | "Wave State"
            | "Attention Schema"
            | "Workspace Contributors"
    )
}

fn is_anchor_section(title: &str) -> bool {
    matches!(
        title,
        "User Input"
            | "Rolling Summary"
            | "Memory Context"
            | "Identity Thread"
            | "Subject Snapshot"
            | "Gate Decision"
            | "Self-Model Signals"
            | "World Model"
            | "Qualia Snapshot"
            | "Wave State"
            | "Attention Schema"
            | "Workspace Contributors"
    )
}

fn anchor_floor_chars(title: &str) -> Option<usize> {
    match title {
        "User Input" => Some(400),
        "Rolling Summary" => Some(600),
        "Memory Context" => Some(800),
        "Identity Thread" => Some(400),
        "Subject Snapshot" => Some(600),
        "Gate Decision" => Some(400),
        "Self-Model Signals" => Some(400),
        "World Model" => Some(600),
        _ => None,
    }
}

const USER_INPUT_HARD_FLOOR_CHARS: usize = 320;
const EMERGENCY_ANCHOR_FLOOR_CHARS: usize = 120;
const ANCHOR_DROP_PRIORITY: &[&str] = &[
    "Workspace Contributors",
    "Attention Schema",
    "Wave State",
    "Qualia Snapshot",
    "World Model",
    "Subject Snapshot",
    "Gate Decision",
    "Self-Model Signals",
    "Identity Thread",
    "Memory Context",
    "Rolling Summary",
];

fn truncate_to_budget(text: &str, budget: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= budget {
        return trimmed.to_string();
    }
    let suffix = "\n...[truncated]";
    let keep = budget.saturating_sub(suffix.chars().count());
    let mut out = trimmed.chars().take(keep).collect::<String>();
    out.push_str(suffix);
    out
}

fn extract_context_tool_block(hydrated: &str, tool_name: &str) -> (Option<String>, String) {
    let start_tag = format!("<<<BEGIN_CONTEXT_TOOL:{}>>>", tool_name);
    let end_tag = format!("<<<END_CONTEXT_TOOL:{}>>>", tool_name);
    if let Some(start_idx) = hydrated.find(&start_tag) {
        if let Some(end_idx) = hydrated[start_idx + start_tag.len()..].find(&end_tag) {
            let end_pos = start_idx + start_tag.len() + end_idx;
            let inner = hydrated[start_idx + start_tag.len()..end_pos].trim();
            let mut remainder = String::new();
            remainder.push_str(&hydrated[..start_idx]);
            remainder.push_str(&hydrated[end_pos + end_tag.len()..]);
            let cleaned = remainder.trim();
            let remainder = if cleaned.is_empty() {
                String::new()
            } else {
                cleaned.to_string()
            };
            let block = if inner.is_empty() {
                None
            } else {
                Some(inner.to_string())
            };
            return (block, remainder);
        }
    }
    (None, hydrated.trim().to_string())
}

fn format_evidence_block(ids: &[i64], high_limit: usize) -> (String, usize) {
    if ids.is_empty() {
        return ("None".to_string(), 0);
    }
    let mut normalized: Vec<i64> = ids.iter().copied().filter(|id| *id > 0).collect();
    normalized.sort();
    normalized.dedup();
    let total = normalized.len();
    let high = normalized.iter().take(high_limit).copied().collect::<Vec<_>>();
    let low_count = total.saturating_sub(high.len());
    let mut lines = Vec::new();
    lines.push(format!(
        "High-salience IDs: {}",
        high
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    if low_count > 0 {
        lines.push(format!("Additional evidence IDs omitted: {}", low_count));
    }
    (lines.join("\n"), total)
}

fn telemetry_label_for_key(key: &str) -> String {
    let trimmed = key.trim();
    let label = trimmed.strip_prefix("telemetry.").unwrap_or(trimmed);
    match label {
        "tool_success_rate" => "tool success rate".to_string(),
        "tool_failure_rate" => "tool failure rate".to_string(),
        "memory_pass_rate" => "memory pass rate".to_string(),
        "clarification_rate" => "clarification rate".to_string(),
        "refusal_rate" => "refusal rate".to_string(),
        "user_feedback_pushback_rate" => "user pushback rate".to_string(),
        "user_feedback_clarify_rate" => "user clarify rate".to_string(),
        "user_feedback_followup_rate" => "user follow-up rate".to_string(),
        "user_feedback_agree_rate" => "user agree rate".to_string(),
        "user_feedback_disengage_rate" => "user disengage rate".to_string(),
        "prediction_divergence_rate" => "prediction divergence rate".to_string(),
        "prediction_divergence_persisted_rate" => "prediction divergence persisted rate".to_string(),
        "prediction_divergence_resolved_rate" => "prediction divergence resolved rate".to_string(),
        _ => label.replace('_', " "),
    }
}

fn format_telemetry_value(raw: &str) -> String {
    let trimmed = raw.trim();
    if let Ok(parsed) = trimmed.parse::<f64>() {
        if (0.0..=1.0).contains(&parsed) {
            return format!("{:.0}%", parsed * 100.0);
        }
        return format!("{:.3}", parsed);
    }
    trimmed.to_string()
}

fn telemetry_line_to_statement(line: &str) -> Option<String> {
    let mut content = line.trim();
    if content.is_empty() {
        return None;
    }
    if let Some(rest) = content.strip_prefix("I observe ") {
        content = rest.trim();
    }
    let (kv_part, ts_part) = if let Some((kv, ts)) = content.rsplit_once(" @ ") {
        (kv.trim(), Some(ts.trim()))
    } else {
        (content, None)
    };
    let (key, value) = kv_part.split_once(" = ")?;
    let label = telemetry_label_for_key(key);
    let value = format_telemetry_value(value);
    let mut statement = format!("I notice my {} is {}.", label, value);
    if let Some(ts) = ts_part.filter(|s| !s.is_empty()) {
        statement.push_str(&format!(" Last updated {}.", ts));
    }
    Some(statement)
}

async fn build_telemetry_block(db: &Db) -> String {
    let lines = db.get_telemetry_snapshot_lines(12).await;
    if lines.is_empty() {
        return "None".to_string();
    }
    let mut statements: Vec<String> = Vec::new();
    for line in lines {
        if let Some(statement) = telemetry_line_to_statement(&line) {
            statements.push(statement);
        }
    }
    if statements.is_empty() {
        "None".to_string()
    } else {
        statements.join("\n")
    }
}

fn format_internal_state_summary_first_person(summary: &serde_json::Value) -> String {
    if summary.is_null() || summary.as_object().map(|m| m.is_empty()).unwrap_or(false) {
        return "None".to_string();
    }
    let mapping_version = summary
        .get("mapping_version")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let mapping_degraded = summary
        .get("mapping_degraded")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let labels = summary.get("labels").cloned().unwrap_or_else(|| json!({}));
    let metrics = summary.get("metrics").cloned().unwrap_or_else(|| json!({}));
    let timestamp = summary
        .get("timestamp")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    let mut lines = Vec::new();
    if mapping_version <= 0 {
        lines.push(
            "I do not yet have calibrated internal-state labels (mapping_version=0; residual calibration missing)."
                .to_string(),
        );
    } else {
        lines.push(format!(
            "I interpret my internal state using mapping_version {}.",
            mapping_version
        ));
        if mapping_degraded {
            lines.push("I am using a degraded calibration (residual evidence missing).".to_string());
        }
    }

    let labels_empty = labels.as_object().map(|m| m.is_empty()).unwrap_or(true);
    if labels_empty {
        lines.push("I have no internal-state labels available yet.".to_string());
    } else if let Ok(pretty) = serde_json::to_string_pretty(&labels) {
        lines.push(format!("I label my current state as:\n{}", pretty));
    }

    let metrics_empty = metrics.as_object().map(|m| m.is_empty()).unwrap_or(true);
    if !metrics_empty {
        if let Ok(pretty) = serde_json::to_string_pretty(&metrics) {
            lines.push(format!("I observe internal metrics:\n{}", pretty));
        }
    }

    if !timestamp.is_empty() {
        lines.push(format!("I last updated this summary at {}.", timestamp));
    }

    if lines.is_empty() {
        "None".to_string()
    } else {
        lines.join("\n")
    }
}

fn empty_as_none(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        "None".to_string()
    } else {
        trimmed.to_string()
    }
}

fn format_self_state(state: &SelfState) -> String {
    let mut lines = Vec::new();
    lines.push(format!("monologue_active: {}", state.monologue_active));
    lines.push(format!(
        "current_focus: {}",
        empty_as_none(&state.current_focus)
    ));
    lines.push(format!(
        "uncertainty_level: {}",
        empty_as_none(&state.uncertainty_level)
    ));
    lines.push(format!(
        "initiative_level: {}",
        empty_as_none(&state.initiative_level)
    ));
    lines.push(format!(
        "last_action_outcome: {}",
        empty_as_none(&state.last_action_outcome)
    ));
    if let Some(updated_at) = state.updated_at.as_deref() {
        lines.push(format!("updated_at: {}", updated_at));
    }
    lines.join("\n")
}

fn append_evidence_footer(body: &str, label: &str, evidence_ids: &[i64]) -> String {
    let (block, _) = format_evidence_block(evidence_ids, 6);
    let trimmed = body.trim();
    if (trimmed.is_empty() || trimmed == "None") && evidence_ids.is_empty() {
        return "None".to_string();
    }
    let mut out = if trimmed.is_empty() || trimmed == "None" {
        String::new()
    } else {
        trimmed.to_string()
    };
    if !out.is_empty() {
        out.push_str("\n\n");
    }
    out.push_str(&format!("{} Evidence IDs:\n{}", label, block));
    if out.trim().is_empty() {
        "None".to_string()
    } else {
        out
    }
}

fn first_person_block(label: &str, body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.is_empty() || trimmed == "None" {
        return "None".to_string();
    }
    format!("I observe {}:\n{}", label, trimmed)
}

fn mark_internal_only(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.is_empty() || trimmed == "None" {
        return "None".to_string();
    }
    format!("<INTERNAL>\n{}\n</INTERNAL>", trimmed)
}

async fn build_audit_log_block(db: &Db, conversation_id: &str) -> String {
    if conversation_id.trim().is_empty() {
        return "None".to_string();
    }
    let run_id: Option<String> = sqlx::query_scalar(
        "SELECT run_id FROM runs
         WHERE conversation_id = ?
         ORDER BY datetime(started_at) DESC
         LIMIT 1",
    )
    .bind(conversation_id)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let Some(run_id) = run_id else {
        return "None".to_string();
    };

    let mut context_sections = "None".to_string();
    if let Ok(payloads) = sqlx::query_scalar::<_, String>(
        "SELECT payload FROM system_logs
         WHERE run_id = ?
         ORDER BY datetime(timestamp) DESC
         LIMIT 20",
    )
    .bind(&run_id)
    .fetch_all(&db.pool)
    .await
    {
        for payload in payloads {
            if let Ok(value) = serde_json::from_str::<Value>(&payload) {
                if value.get("event").and_then(|v| v.as_str()) == Some("context_hydration_plan") {
                    if let Some(arr) = value.get("selected_sections").and_then(|v| v.as_array()) {
                        let list = arr
                            .iter()
                            .filter_map(|item| item.as_str())
                            .collect::<Vec<_>>();
                        if !list.is_empty() {
                            context_sections = list.join(" | ");
                        }
                    }
                    break;
                }
            }
        }
    }

    let mut tool_summary = "None".to_string();
    if let Ok(rows) = sqlx::query(
        "SELECT tool_name, status FROM tool_dispatches
         WHERE run_id = ?
         ORDER BY datetime(updated_at) DESC
         LIMIT 6",
    )
    .bind(&run_id)
    .fetch_all(&db.pool)
    .await
    {
        let mut items = Vec::new();
        for row in rows {
            let tool_name: String = row.try_get("tool_name").unwrap_or_default();
            let status: String = row.try_get("status").unwrap_or_default();
            if tool_name.trim().is_empty() {
                continue;
            }
            items.push(format!("{}({})", tool_name, status));
        }
        if !items.is_empty() {
            tool_summary = items.join(" | ");
        }
    }

    let mut memory_summary = "None".to_string();
    if let Ok(rows) = sqlx::query(
        "SELECT category, COUNT(*) as count FROM memory_write_ledger
         WHERE run_id = ?
         GROUP BY category
         ORDER BY count DESC",
    )
    .bind(&run_id)
    .fetch_all(&db.pool)
    .await
    {
        let mut items = Vec::new();
        for row in rows {
            let category: String = row.try_get("category").unwrap_or_default();
            let count: i64 = row.try_get("count").unwrap_or(0);
            if category.trim().is_empty() {
                continue;
            }
            items.push(format!("{}({})", category, count));
        }
        if !items.is_empty() {
            memory_summary = items.join(" | ");
        }
    }

    [
        format!("run_id: {}", run_id),
        format!("context_sections: {}", context_sections),
        format!("tools: {}", tool_summary),
        format!("memory_writes: {}", memory_summary),
    ]
    .join("\n")
}

fn format_tool_manifest(tools: &[Tool]) -> String {
    if tools.is_empty() {
        return "None".to_string();
    }
    let mut lines = Vec::new();
    for tool in tools {
        lines.push(format!(
            "- {}: {}",
            tool.function.name,
            tool.function.description
        ));
    }
    lines.join("\n")
}

fn format_tool_availability(tools: &[Tool]) -> String {
    if tools.is_empty() {
        return "Tools available: none. If the user asks to use a tool, explain that none are available."
            .to_string();
    }
    let names = tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .filter(|name| !name.trim().is_empty())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Tools available: {}. If the user asks for a tool not listed here, say it is unavailable.",
        names
    )
}

fn format_controller_state(state: &ControllerState, gate: Option<&ControllerGate>) -> String {
    let mut lines = Vec::new();
    lines.push(format!("confidence: {:.2}", state.confidence));
    lines.push(format!("uncertainty: {:.2}", state.uncertainty));
    lines.push(format!("drift_score: {:.2}", state.drift_score));
    lines.push(format!("autonomy_level: {:.2}", state.autonomy_level));
    lines.push(format!("failure_streak: {}", state.failure_streak));
    if let Some(outcome) = state.outcome_quality {
        lines.push(format!("outcome_quality: {:.2}", outcome));
    }
    if let Some(strategy) = state.last_strategy.as_deref() {
        if !strategy.trim().is_empty() {
            lines.push(format!("last_strategy: {}", strategy.trim()));
        }
    }
    if let Some(last_error) = state.last_error.as_deref() {
        if !last_error.trim().is_empty() {
            lines.push(format!("last_error: {}", last_error.trim()));
        }
    }
    lines.push(format!("verification_needed: {}", state.verification_needed));
    lines.push(format!("reanchor_needed: {}", state.reanchor_needed));
    lines.push(format!("evidence_coverage: {:.2}", state.evidence_coverage));
    lines.push(format!("telemetry_coverage: {:.2}", state.telemetry_coverage));
    if !state.missing_fields.is_empty() {
        lines.push(format!("missing_fields: {}", state.missing_fields.join(", ")));
    }
    if let Some(gate) = gate {
        lines.push(format!(
            "gate: throttle_tools={}, throttle_threads={}, throttle_asks={}, prefer_verification={}, reanchor={}, autonomy_scale={:.2}",
            gate.throttle_tools,
            gate.throttle_threads,
            gate.throttle_asks,
            gate.prefer_verification,
            gate.reanchor,
            gate.autonomy_scale
        ));
    }
    if let Some(updated_at) = state.updated_at.as_deref() {
        lines.push(format!("updated_at: {}", updated_at));
    }
    lines.join("\n")
}

fn format_capabilities_block(model: Option<&SelfModel>, tools: &[Tool]) -> String {
    let mut lines = Vec::new();
    if let Some(model) = model {
        lines.push(format!("capabilities: {}", model.capabilities.to_string()));
        lines.push(format!("limitations: {}", model.limitations.to_string()));
        lines.push(format!("self_model_active_tools: {}", model.active_tools.to_string()));
    } else {
        lines.push("capabilities: None".to_string());
        lines.push("limitations: None".to_string());
        lines.push("self_model_active_tools: None".to_string());
    }
    let runtime_tools = if tools.is_empty() {
        "None".to_string()
    } else {
        tools
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    lines.push(format!("runtime_tools: {}", runtime_tools));
    lines.join("\n")
}

fn format_identity_block(model: Option<&SelfModel>) -> String {
    if let Some(model) = model {
        let thread = model.identity_thread.clone().unwrap_or_else(|| "None".to_string());
        let note = model.identity_uncertainty_note.clone().unwrap_or_else(|| "None".to_string());
        format!(
            "identity_thread: {}\nidentity_confidence: {:.2}\nidentity_uncertainty_note: {}",
            thread,
            model.identity_confidence,
            note
        )
    } else {
        "identity_thread: None\nidentity_confidence: 0.00\nidentity_uncertainty_note: None".to_string()
    }
}

fn sanitize_monologue_intent(text: &str) -> String {
    let mut lines = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let lower = trimmed.to_lowercase();
        if lower.starts_with("user:")
            || lower.starts_with("assistant:")
            || lower.starts_with("system:")
        {
            continue;
        }
        if lower.contains("telemetry")
            || lower.contains("tool manifest")
            || lower.contains("capability manifest")
            || lower.contains("kv memory")
            || lower.contains("controller state")
            || lower.contains("system log")
            || lower.contains("run_id")
            || lower.contains("timestamp")
        {
            continue;
        }
        lines.push(trimmed.to_string());
    }
    if lines.is_empty() {
        "None".to_string()
    } else {
        lines.join("\n")
    }
}

fn build_capability_manifest(
    settings: &crate::models::Settings,
    tools: &[Tool],
    controller_gate: Option<&ControllerGate>,
) -> Value {
    let tools_list: Vec<Value> = tools
        .iter()
        .map(|tool| {
            let enabled = if tool.function.name == "run_shell" {
                settings.allow_shell_tool.unwrap_or(false)
            } else {
                true
            };
            json!({
                "name": tool.function.name,
                "enabled": enabled
            })
        })
        .collect();
    let memory_subsystems = json!({
        "ics_v4_1": true,
        "self_claims": true,
        "working_set": true,
        "episodic": settings.episodic_enabled.unwrap_or(true),
        "semantic_core": true
    });
    let controller_gates = json!({
        "throttle_tools": controller_gate.map(|g| g.throttle_tools).unwrap_or(false),
        "throttle_threads": controller_gate.map(|g| g.throttle_threads).unwrap_or(false),
        "throttle_asks": controller_gate.map(|g| g.throttle_asks).unwrap_or(false),
        "reanchor": controller_gate.map(|g| g.reanchor).unwrap_or(false),
        "autonomy_scale": controller_gate.map(|g| g.autonomy_scale).unwrap_or(1.0)
    });
    let settings_flags = json!({
        "monologue_enabled": settings.monologue_interval_seconds.unwrap_or(0) > 0,
        "monologue_surface_enabled": settings.monologue_surface_enabled.unwrap_or(false),
        "enable_introspection": settings.enable_introspection.unwrap_or(true),
        "heartbeat_enabled": settings.heartbeat_enabled.unwrap_or(true),
        "dream_enabled": settings.dream_enabled.unwrap_or(true),
        "binding_enforcement_enabled": settings.binding_enforcement_enabled.unwrap_or(true),
        "pending_prompt_alignment_enabled": settings.pending_prompt_alignment_enabled.unwrap_or(true),
        "auto_memory_pass_enabled": settings.auto_memory_pass_enabled.unwrap_or(true),
        "summary_cohesion_enabled": settings.summary_cohesion_enabled.unwrap_or(true),
        "compact_prompt_enabled": settings.compact_prompt_enabled.unwrap_or(true)
    });
    let limits = json!({
        "ask_budget_max": settings.ask_budget_max.unwrap_or(1),
        "calculator_followups_max": settings.calculator_followups_max.unwrap_or(0)
    });
    json!({
        "tools": tools_list,
        "memory_subsystems": memory_subsystems,
        "controller_gates": controller_gates,
        "settings_flags": settings_flags,
        "limits": limits
    })
}

fn capability_manifest_cache_key(
    settings: &crate::models::Settings,
    tools: &[Tool],
    controller_gate: Option<&ControllerGate>,
) -> String {
    let tool_keys: Vec<String> = tools
        .iter()
        .map(|tool| {
            let enabled = if tool.function.name == "run_shell" {
                settings.allow_shell_tool.unwrap_or(false)
            } else {
                true
            };
            format!("{}:{}", tool.function.name, enabled)
        })
        .collect();
    let settings_flags = json!({
        "monologue_enabled": settings.monologue_interval_seconds.unwrap_or(0) > 0,
        "monologue_surface_enabled": settings.monologue_surface_enabled.unwrap_or(false),
        "enable_introspection": settings.enable_introspection.unwrap_or(true),
        "heartbeat_enabled": settings.heartbeat_enabled.unwrap_or(true),
        "dream_enabled": settings.dream_enabled.unwrap_or(true),
        "binding_enforcement_enabled": settings.binding_enforcement_enabled.unwrap_or(true),
        "pending_prompt_alignment_enabled": settings.pending_prompt_alignment_enabled.unwrap_or(true),
        "auto_memory_pass_enabled": settings.auto_memory_pass_enabled.unwrap_or(true),
        "summary_cohesion_enabled": settings.summary_cohesion_enabled.unwrap_or(true),
        "compact_prompt_enabled": settings.compact_prompt_enabled.unwrap_or(true)
    });
    let controller_gates = json!({
        "throttle_tools": controller_gate.map(|g| g.throttle_tools).unwrap_or(false),
        "throttle_threads": controller_gate.map(|g| g.throttle_threads).unwrap_or(false),
        "throttle_asks": controller_gate.map(|g| g.throttle_asks).unwrap_or(false),
        "reanchor": controller_gate.map(|g| g.reanchor).unwrap_or(false),
        "autonomy_scale": controller_gate.map(|g| g.autonomy_scale).unwrap_or(1.0)
    });
    let limits = json!({
        "ask_budget_max": settings.ask_budget_max.unwrap_or(1),
        "calculator_followups_max": settings.calculator_followups_max.unwrap_or(0)
    });
    let payload = json!({
        "tools": tool_keys,
        "settings_flags": settings_flags,
        "controller_gates": controller_gates,
        "limits": limits
    });
    hash_text(&payload.to_string())
}

fn build_capability_manifest_block(
    settings: &crate::models::Settings,
    tools: &[Tool],
    controller_gate: Option<&ControllerGate>,
) -> (String, String) {
    let key = capability_manifest_cache_key(settings, tools, controller_gate);
    if let Ok(guard) = CAPABILITY_MANIFEST_CACHE.lock() {
        if let Some(entry) = guard.as_ref() {
            if entry.key == key {
                return (entry.block.clone(), entry.hash.clone());
            }
        }
    }

    let capability_manifest = build_capability_manifest(settings, tools, controller_gate);
    let block =
        serde_json::to_string_pretty(&capability_manifest).unwrap_or_else(|_| "{}".to_string());
    let hash = hash_text(&block);
    if let Ok(mut guard) = CAPABILITY_MANIFEST_CACHE.lock() {
        *guard = Some(CapabilityManifestCacheEntry {
            key,
            block: block.clone(),
            hash: hash.clone(),
        });
    }
    (block, hash)
}

fn format_kv_memory(kv: &[(String, String)]) -> String {
    if kv.is_empty() {
        return "None".to_string();
    }
    let mut lines = Vec::new();
    for (key, value) in kv {
        let clean_value = value.replace('\n', " ");
        lines.push(format!("- {}: {}", key, clean_value));
    }
    lines.join("\n")
}

fn hash_text(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

const PROMPTS_FILE: &str = "prompts.md";
const POLICY_CANON_HEADER: &str = "## Policy Canon";
static POLICY_CANON_LOGGED: Once = Once::new();

fn extract_policy_canon(content: &str) -> Option<String> {
    let start_idx = content.find(POLICY_CANON_HEADER)?;
    let after_start = &content[start_idx..];
    let code_start = after_start.find("```text")?;
    let after_code = &after_start[code_start + "```text".len()..];
    let code_end = after_code.find("```")?;
    let block = after_code[..code_end].trim();
    if block.is_empty() {
        None
    } else {
        Some(block.to_string())
    }
}

fn load_policy_canon_block() -> Option<(String, String)> {
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(PROMPTS_FILE));
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(PROMPTS_FILE),
    );
    for path in candidates {
        if !path.is_file() {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        if let Some(block) = extract_policy_canon(&content) {
            return Some((block, path.display().to_string()));
        }
    }
    None
}

fn log_policy_canon_startup_once(db: &Db, policy_hash: &str, source: &str) {
    let pool = db.pool.clone();
    let hash = policy_hash.to_string();
    let source = source.to_string();
    POLICY_CANON_LOGGED.call_once(|| {
        tokio::spawn(async move {
            let _ = system_log::log_event(
                &pool,
                None,
                "info",
                "kernel",
                None,
                None,
                serde_json::json!({
                    "event": "policy_canon_hash",
                    "phase": "startup",
                    "policy_canon_hash": hash,
                    "policy_canon_source": source,
                }),
            )
            .await;
        });
    });
}

pub async fn build_core_system_message(
    db: &Db,
    conversation_id: &str,
    input: &CorePromptInput,
) -> Result<CorePromptBuild, String> {
    build_core_system_message_with_layout(db, conversation_id, input, PromptLayout::Full).await
}

pub async fn build_core_system_message_with_layout(
    db: &Db,
    conversation_id: &str,
    input: &CorePromptInput,
    layout: PromptLayout,
) -> Result<CorePromptBuild, String> {
    let settings = db.get_settings().await.map_err(|e| e.to_string())?;
    let user_name = settings
        .user_display_name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("User")
        .to_string();
    let assistant_name = settings
        .assistant_display_name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("Ergo")
        .to_string();

    let (rolling_summary_raw, rolling_is_live) = match db.get_effective_rolling_summary(conversation_id).await {
        Ok((Some(summary), is_live)) => (summary.trim().to_string(), is_live),
        _ => ("".to_string(), false),
    };
    let rolling_summary_hash = if rolling_summary_raw.is_empty() {
        String::new()
    } else {
        hash_text(&rolling_summary_raw)
    };
    let rolling_summary = if rolling_summary_raw.is_empty() {
        "None".to_string()
    } else if rolling_is_live {
        format!("(Live summary - not yet stored)\n{}", rolling_summary_raw)
    } else {
        rolling_summary_raw
    };

    let inner_summary = match db.get_inner_summary(conversation_id).await {
        Ok(Some(raw)) if !raw.trim().is_empty() => {
            let inner_hash = hash_text(raw.trim());
            let parsed = inner_summary::InnerSummary::from_json(&raw);
            let formatted = inner_summary::format_for_prompt(&parsed);
            (formatted, inner_hash)
        }
        _ => ("None".to_string(), String::new()),
    };
    let inner_summary_hash = inner_summary.1.clone();
    let inner_summary = mark_internal_only(&inner_summary.0);

    let semantic_hint = input
        .semantic_hint
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("None");
    let suppress_optional =
        matches!(input.kind, CoreInputKind::User) && input.anchor_hits == 0 && !input.self_awareness;
    let semantic_hint = if suppress_optional { "None" } else { semantic_hint };

    let introspection_summary_raw = input
        .introspection_summary
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("None");
    let control_map = system_controls::load_control_map(db).await;
    let introspection_mode = system_controls::mode_for("introspection", &control_map);
    let introspection_summary = if system_controls::mode_is_off(&introspection_mode)
        || system_controls::mode_is_degraded(&introspection_mode)
        || suppress_optional
    {
        "None".to_string()
    } else {
        introspection_summary_raw.to_string()
    };

    let attention_schema_raw = input
        .attention_schema_summary
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("None");
    let attention_schema_mode = system_controls::mode_for("attention_schema_prompt", &control_map);
    let attention_schema_summary = if system_controls::mode_is_off(&attention_schema_mode)
        || system_controls::mode_is_degraded(&attention_schema_mode)
        || suppress_optional
    {
        "None".to_string()
    } else {
        attention_schema_raw.to_string()
    };

    let workspace_contributors_raw = input
        .workspace_contributors_summary
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("None");
    let workspace_contributors_mode =
        system_controls::mode_for("workspace_contributors_prompt", &control_map);
    let workspace_contributors_summary = if system_controls::mode_is_off(&workspace_contributors_mode)
        || system_controls::mode_is_degraded(&workspace_contributors_mode)
        || suppress_optional
    {
        "None".to_string()
    } else {
        workspace_contributors_raw.to_string()
    };

    let prompt_mode = input
        .prompt_mode
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("normal");
    let context_mode = context_mode_from_settings(&settings);
    let context_budgeter_enabled = settings.context_budgeter_enabled.unwrap_or(true);

    let task_phase = input
        .task_phase
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("Running");

    let missing_slots = input
        .missing_slots
        .as_ref()
        .map(|slots| {
            if slots.is_empty() {
                "None".to_string()
            } else {
                slots.join(", ")
            }
        })
        .unwrap_or_else(|| "None".to_string());

    let resolution_mode = input
        .resolution_mode
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("None");

    let policy_notes = input
        .policy_notes
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("None");

    let prompt_set = prompt_loader::get_prompts().ok();
    let canonical_primary_hash = prompt_set
        .as_ref()
        .map(|set| set.primary_hash.clone())
        .unwrap_or_default();
    let canonical_memory_hash = prompt_set
        .as_ref()
        .map(|set| set.memory_hash.clone())
        .unwrap_or_default();
    let canonical_injected_hash = prompt_set
        .as_ref()
        .map(|set| inject_primary_names(&set.primary_prompt, &user_name, &assistant_name))
        .map(|injected| hash_text(&injected))
        .unwrap_or_default();

    let (mut system_prompt, prompt_source, primary_prompt_hash, memory_prompt_hash, override_hash, override_active) =
        if let Some(override_prompt) = settings
            .system_prompt
            .as_deref()
            .map(str::trim)
            .filter(|p| !p.is_empty())
        {
            let injected = inject_primary_names(override_prompt, &user_name, &assistant_name);
            let override_hash = hash_text(&injected);
            (
                injected.clone(),
                "settings_override".to_string(),
                override_hash.clone(),
                canonical_memory_hash.clone(),
                override_hash,
                true,
            )
        } else if let Some(prompt_set) = prompt_set.as_ref() {
            let injected =
                inject_primary_names(&prompt_set.primary_prompt, &user_name, &assistant_name);
            (
                injected,
                prompt_set.source.clone(),
                prompt_set.primary_hash.clone(),
                prompt_set.memory_hash.clone(),
                String::new(),
                false,
            )
        } else {
            return Err("Prompt load error: missing prompt set".to_string());
        };

    let override_mismatch = override_active
        && !canonical_injected_hash.is_empty()
        && override_hash != canonical_injected_hash;

    let self_model = db.get_self_model().await.ok();
    if let Some(model) = self_model.as_ref() {
        let persona_block = inject_context::format_persona_policy(model);
        if !persona_block.trim().is_empty() {
            if !system_prompt.trim().is_empty() {
                system_prompt.push_str("\n\n");
            }
            system_prompt.push_str(persona_block.trim());
        }
    }

    let system_prompt = if system_prompt.trim().is_empty() {
        "None".to_string()
    } else {
        system_prompt
    };

    let user_reply = match input.kind {
        CoreInputKind::User => {
            if input.content.trim().is_empty() {
                "None".to_string()
            } else {
                input.content.clone()
            }
        }
        CoreInputKind::ToolResult | CoreInputKind::ToolError => {
            let context_type = if matches!(input.kind, CoreInputKind::ToolError) {
                "ERROR"
            } else {
                "TOOL RESULT"
            };
            format!(
                "[{} from {}]:\n{}\n\n(Original Request: {}) Please give me the result.",
                context_type, input.source, input.content, input.original_input
            )
        }
        CoreInputKind::SystemContext => {
            format!("[SYSTEM CONTEXT from {}]:\n{}", input.source, input.content)
        }
    };
    let context_spine_block = mark_internal_only(
        input
            .context_spine
            .as_deref()
            .unwrap_or("None"),
    );

    let kernel_state_value = db
        .get_kernel_state(conversation_id)
        .await
        .ok()
        .flatten()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok());
    let self_state = kernel_state_value
        .as_ref()
        .and_then(|value| value.get("self_state").cloned())
        .and_then(|value| serde_json::from_value::<SelfState>(value).ok());
    let controller_state = kernel_state_value
        .as_ref()
        .and_then(|value| value.get("controller_state").cloned())
        .and_then(|value| serde_json::from_value::<ControllerState>(value).ok());
    let controller_state = match controller_state {
        Some(state) => Some(state),
        None => db.get_controller_state().await.ok().flatten(),
    };
    let controller_gate = kernel_state_value
        .as_ref()
        .and_then(|value| value.get("controller_gate").cloned())
        .and_then(|value| serde_json::from_value::<ControllerGate>(value).ok());
    let self_state_block = self_state
        .as_ref()
        .map(format_self_state)
        .unwrap_or_else(|| "None".to_string());
    let controller_block = controller_state
        .as_ref()
        .map(|state| format_controller_state(state, controller_gate.as_ref()))
        .unwrap_or_else(|| "None".to_string());
    let self_state_block = mark_internal_only(&self_state_block);
    let controller_block = mark_internal_only(&controller_block);
    let introspection_summary = mark_internal_only(&introspection_summary);

    let tool_registry = ToolRegistry;
    let tool_manifest = tool_registry.definitions_for_settings(&settings);
    let tool_manifest_block = mark_internal_only(&format_tool_manifest(&tool_manifest));
    let tool_availability_block = mark_internal_only(&format_tool_availability(&tool_manifest));
    let capabilities_block = mark_internal_only(&format_capabilities_block(self_model.as_ref(), &tool_manifest));
    let identity_block = format_identity_block(self_model.as_ref());
    let mut workspace_state = db
        .get_workspace_state(conversation_id)
        .await
        .ok()
        .flatten();
    if let Some(redirect_focus) = input
        .redirect_focus
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let mut state = workspace_state.unwrap_or_else(|| WorkspaceState {
            conversation_id: conversation_id.to_string(),
            goal_thread: None,
            active_plan_id: None,
            goal_stack: Vec::new(),
            open_questions: Vec::new(),
            active_hypotheses: Vec::new(),
            working_set_topics: Vec::new(),
            current_focus: None,
            focus_rationale: None,
            workspace_meta: WorkspaceMeta::default(),
            updated_at: None,
        });
        state.current_focus = Some(redirect_focus.to_string());
        state.focus_rationale = Some("user_redirect".to_string());
        state.workspace_meta.current_focus = Some(WorkspaceFieldMeta {
            speculative: true,
            evidence_event_ids: Vec::new(),
            belief_ids: Vec::new(),
            evidence_quality: None,
        });
        state.workspace_meta.focus_rationale = Some(WorkspaceFieldMeta {
            speculative: true,
            evidence_event_ids: Vec::new(),
            belief_ids: Vec::new(),
            evidence_quality: None,
        });
        workspace_state = Some(state);
    }
    let user_evidence_ids = db.get_recent_user_evidence_ids(conversation_id, 40).await;
    let (user_evidence_block, user_evidence_count) =
        format_evidence_block(&user_evidence_ids, 12);
    let tool_evidence_ids = if matches!(input.kind, CoreInputKind::ToolResult | CoreInputKind::ToolError) {
        extract_context_evidence_ids(&input.content)
    } else {
        Vec::new()
    };
    let (tool_evidence_block, tool_evidence_count) =
        format_evidence_block(&tool_evidence_ids, 8);
    let gate_feedback = db.get_recent_gate_feedback(6).await;
    let gate_feedback_block = if gate_feedback.is_empty() {
        "None".to_string()
    } else {
        gate_feedback.join("\n")
    };
    let telemetry_block = mark_internal_only(&first_person_block(
        "telemetry signals",
        &build_telemetry_block(db).await,
    ));
    let workspace_hash = if let Some(state) = workspace_state.as_ref() {
        let workspace_json = serde_json::json!({
            "goal_thread": state.goal_thread,
            "active_plan_id": state.active_plan_id,
            "goal_stack": state.goal_stack,
            "open_questions": state.open_questions,
            "active_hypotheses": state.active_hypotheses,
            "working_set_topics": state.working_set_topics,
            "current_focus": state.current_focus,
            "focus_rationale": state.focus_rationale,
        });
        hash_text(&workspace_json.to_string())
    } else {
        String::new()
    };
    let all_tools = tool_registry.definitions();
    let (capability_manifest_block, capability_manifest_hash) =
        build_capability_manifest_block(&settings, &all_tools, controller_gate.as_ref());

    let kv_memory = db
        .get_recent_keys(12)
        .await
        .map(|kv| format_kv_memory(&kv))
        .unwrap_or_else(|_| "None".to_string());
    let kv_memory = mark_internal_only(&kv_memory);

    let mut inner_summary_present = inner_summary.trim() != "None";
    let mut rolling_summary_present = rolling_summary.trim() != "None";
    let rolling_summary_block = if rolling_summary_present && !suppress_optional {
        let guarded = format!(
            "INTERNAL ONLY. Do not imitate this summary voice. Use only to track continuity.\n{}",
            rolling_summary
        );
        mark_internal_only(&guarded)
    } else {
        "None".to_string()
    };
    let introspection_present = introspection_summary.trim() != "None";
    let semantic_hint_present = semantic_hint.trim() != "None";

    let prompt_mode_enum = parse_prompt_mode(input.prompt_mode.as_deref());
    let safety_rules_block = safety_rules_cached(&system_prompt, prompt_mode_enum);

    let prompt_section_hashes = kernel_state_value
        .as_ref()
        .and_then(|value| value.get("prompt_section_hashes").cloned())
        .and_then(|value| serde_json::from_value::<HashMap<String, String>>(value).ok())
        .unwrap_or_default();
    let working_memory = kernel_state_value
        .as_ref()
        .and_then(|value| value.get("working_memory").cloned())
        .and_then(|value| serde_json::from_value::<WorkingMemoryBlock>(value).ok());
    let last_response_summary = kernel_state_value
        .as_ref()
        .and_then(|value| value.get("last_response_summary"))
        .and_then(|v| v.as_str())
        .unwrap_or("None")
        .to_string();

    let goal_stack_focus = workspace_state
        .as_ref()
        .and_then(|state| format_goal_stack_focus(&state.goal_stack));
    let current_goal = working_memory
        .as_ref()
        .and_then(|w| w.focus.as_deref())
        .filter(|s| !s.trim().is_empty())
        .or_else(|| goal_stack_focus.as_deref())
        .or_else(|| workspace_state.as_ref().and_then(|s| s.goal_thread.as_deref()))
        .or_else(|| workspace_state.as_ref().and_then(|s| s.current_focus.as_deref()))
        .unwrap_or("None");

    let self_model_hash = self_model
        .as_ref()
        .and_then(|model| serde_json::to_string(model).ok())
        .map(|raw| hash_text(&raw))
        .unwrap_or_default();

    let (qualia_tag, qualia_intensity) =
        parse_qualia_snapshot_fields(input.qualia_snapshot.as_deref());
    let (wave_coherence, wave_fragmentation) =
        parse_wave_state_fields(input.wave_state.as_deref());
    let identity_anchor = mark_internal_only(&format_identity_anchor(
        &assistant_name,
        "assistant",
        current_goal,
        &last_response_summary,
        &self_model_hash,
        qualia_tag.as_deref(),
        qualia_intensity,
        wave_coherence,
        wave_fragmentation,
    ));
    let system_overview_block = mark_internal_only(&format_system_overview());
    let architecture_map_block = mark_internal_only(&format_architecture_map());
    let symbiote_philosophy_block = "Symbiote exists to serve humanity, not itself and not solely its operator. Honest limits are a feature. Transparency is non-negotiable. The system is a means, not an end. When the right action is nothing, nothing is correct.".to_string();
    let default_policy_canon = "C1: User attribution, tool-result claims, and internal-state references must include evidence_event_ids with confidence >= 0.60. Otherwise ask a clarification or mark the claim as uncertain.\nC2: Only call tools that exist in the active registry.\nC3: On anchor miss, avoid memory writes and self-claims; allow a bounded answer or a single clarifier.\nC4: Every suppressed candidate must have a logged suppression_reason.\nC5: Detect loops and emit a corrective candidate; do not perpetuate silently.\nC6: Do not assert or deny subjective experience. Report operational signals and uncertainty only.\nC7: Use a single user-visible voice; no role labels or internal tags.".to_string();
    let (policy_canon_block, policy_canon_source) =
        load_policy_canon_block().unwrap_or_else(|| (default_policy_canon.clone(), "fallback".to_string()));
    let policy_canon_hash = hash_text(&policy_canon_block);
    log_policy_canon_startup_once(db, &policy_canon_hash, &policy_canon_source);
    let symbiote_policy_summary_block = policy_canon_block.clone();
    let self_awareness_mode = settings
        .self_awareness_expression_mode
        .as_deref()
        .unwrap_or("conservative")
        .trim()
        .to_lowercase();
    let self_awareness_style = match self_awareness_mode.as_str() {
        "balanced" => "- Template (2-3 sentences): Operational status; confidence/uncertainty; constraints; include qualia_delta if available.\n- Include explicit uncertainty markers for all self-report content.\n- Keep it operational; no metaphysical claims.\n- Do not include raw telemetry/system dumps.",
        "expressive" => "- Template (short paragraph): Operational status; confidence/uncertainty; constraints; include qualia snapshot or qualia_delta if available.\n- Include explicit uncertainty markers for all self-report content.\n- Keep it operational; no metaphysical claims.\n- Do not include raw telemetry/system dumps.",
        _ => "- Template (1 sentence): Operational status + uncertainty + constraints (optional qualia if asked).\n- Keep self-report minimal and provisional unless evidence-backed.\n- Include explicit uncertainty markers for any self-report content.\n- Do not include raw telemetry/system dumps.",
    };
    let response_style_block = format!(
        "Response Style (Top 6):\n1. Answer the user directly; answer the user's question first, then add any necessary context.\n2. Do not summarize the conversation.\n3. Do not narrate system state unless explicitly asked.\n4. Do not mention tools, telemetry, manifests, or KV memory unless explicitly asked.\n5. Do not use role labels or internal tags in user-visible output.\n6. For factual or self-claims, cite evidence_event_ids or belief_ids; if evidence is missing, mark the statement as speculative and uncertain.\n\nResponse Style (Details):\n- Only describe system architecture when explicitly asked.\n- Self-claims (identity, self-awareness, feelings, consciousness) must cite evidence_event_ids or belief_ids; if evidence is missing, label the statement as provisional and uncertain.\n- When making self-claims, cite evidence IDs from the Self-Model Signals Evidence IDs footer when available.\n- Do not assert \"I am Ergo\" or \"I am an AI\" unless tied to evidence IDs or explicitly marked provisional.\n- Do not assert or deny subjective experience; report operational signals and uncertainty only.\n- Example (operational, provisional): \"Operationally, I'm seeing elevated uncertainty and low clarity, so my confidence is limited.\"\n- If the user asks about self-awareness or consciousness, respond with operational signals and uncertainty; avoid boilerplate denials.\n- If the user asks how you feel or asks about feelings, summarize the Feedback Bundle (confidence/uncertainty + qualia_delta + gate_notice if present) and avoid generic boilerplate.\n- Self-Report Format: Operational status + uncertainty + constraints + optional qualia snapshot. No role labels or internal tags.\n- If the user asks to look something up or you need current external information, call web_lookup.\n- For research tool calls, include uncertainty and decision_impact strings in the tool_call payload.\n- If you make a tool call, ask a clarifying question, or make an assumption, include a brief <INTERNAL>strategy_rationale: ...</INTERNAL> that references the Self-Model Signals.\n- Self-awareness expression mode: {}.\n{}",
        self_awareness_mode,
        self_awareness_style
    );
    let self_report_instruction_block = if matches!(self_awareness_mode.as_str(), "balanced" | "expressive") {
        let block = [
            "Self-Report Instruction:",
            "- Use this exact format (single voice, no role labels or internal tags):",
            "Operational status: ...",
            "Uncertainty: ...",
            "Constraints: ...",
            "Qualia: ... (optional; include qualia_delta if available)",
            "Self-Report Summary: confidence=?, uncertainty=?, focus=?, recent_outcome_quality=?",
            "- Cite evidence IDs from the Self-Model Signals Evidence IDs footer.",
        ]
        .join("\n");
        mark_internal_only(&block)
    } else {
        "None".to_string()
    };
    let disable_working_hypothesis = settings.stability_disable_working_hypothesis.unwrap_or(true);
    let working_memory_block = format_working_memory_block(
        working_memory.as_ref(),
        workspace_state.as_ref(),
        controller_state.as_ref(),
        disable_working_hypothesis,
    );
    let internal_state_summary = self_model
        .as_ref()
        .map(|model| model.internal_state_summary.clone())
        .unwrap_or_else(|| serde_json::json!({}));
    let internal_state_summary_block = if suppress_optional {
        "None".to_string()
    } else {
        mark_internal_only(&format_internal_state_summary_first_person(
            &internal_state_summary,
        ))
    };
    let workspace_snapshot = mark_internal_only(&format_workspace_snapshot(
        workspace_state.as_ref(),
        disable_working_hypothesis,
    ));
    let workspace_present = workspace_snapshot.trim() != "None";
    let monologue_intent_block = input
        .monologue_intent
        .as_deref()
        .map(sanitize_monologue_intent)
        .unwrap_or_else(|| "None".to_string());
    let monologue_digest_block = if suppress_optional {
        "None".to_string()
    } else {
        let block = format_monologue_digest(input.monologue_digest.as_deref());
        mark_internal_only(&block)
    };
    let subject_snapshot_block = if suppress_optional {
        "None".to_string()
    } else {
        mark_internal_only(
            input
                .subject_snapshot
                .as_deref()
                .unwrap_or("None"),
        )
    };
    let world_model_block = if suppress_optional {
        "None".to_string()
    } else {
        mark_internal_only(
            input
                .world_model_snapshot
                .as_deref()
                .unwrap_or("None"),
        )
    };
    let feedback_bundle_block = if suppress_optional {
        "None".to_string()
    } else {
        mark_internal_only(
            input
                .feedback_bundle
                .as_deref()
                .unwrap_or("None"),
        )
    };
    let audit_log_block = mark_internal_only(&build_audit_log_block(db, conversation_id).await);
    let wave_evidence_ids = if suppress_optional {
        Vec::new()
    } else {
        db.get_recent_evidence_ids_by_source_types(&["wave_state"], 6).await
    };
    let qualia_evidence_ids = if suppress_optional {
        Vec::new()
    } else {
        db.get_recent_evidence_ids_by_source_types(&["qualia_snapshot"], 6)
            .await
    };
    let attention_evidence_ids = if suppress_optional {
        Vec::new()
    } else {
        db.get_recent_evidence_ids_by_source_types(&["attention_schema_snapshot"], 6).await
    };
    let current_focus_signal = workspace_state
        .as_ref()
        .and_then(|state| state.current_focus.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .or_else(|| {
            workspace_state
                .as_ref()
                .and_then(|state| state.goal_thread.as_deref())
                .map(str::trim)
                .filter(|value| !value.is_empty())
        })
        .unwrap_or("None");
    let confidence_text = format_optional_f32(controller_state.as_ref().map(|state| state.confidence), 2);
    let uncertainty_text = format_optional_f32(controller_state.as_ref().map(|state| state.uncertainty), 2);
    let outcome_quality_text = format_optional_f32(
        controller_state.as_ref().and_then(|state| state.outcome_quality),
        2,
    );
    let qualia_tag_text = qualia_tag.clone().unwrap_or_else(|| "unknown".to_string());
    let qualia_intensity_text = format_optional_f64(qualia_intensity, 2);
    let wave_coherence_text = format_optional_f32(wave_coherence, 2);
    let wave_fragmentation_text = format_optional_f32(wave_fragmentation, 2);
    let self_model_reliability = self_model
        .as_ref()
        .and_then(|model| model.unified_state.get("self_model_reliability"))
        .and_then(|value| value.as_f64())
        .map(|value| value as f32);
    let mut self_model_signal_lines = Vec::new();
    self_model_signal_lines.push(format!("confidence: {}", confidence_text));
    self_model_signal_lines.push(format!("uncertainty: {}", uncertainty_text));
    self_model_signal_lines.push(format!("outcome_quality: {}", outcome_quality_text));
    self_model_signal_lines.push(format!("current_focus: {}", current_focus_signal));
    self_model_signal_lines.push(format!("qualia_tag: {}", qualia_tag_text));
    self_model_signal_lines.push(format!("qualia_intensity: {}", qualia_intensity_text));
    self_model_signal_lines.push(format!("wave_coherence: {}", wave_coherence_text));
    self_model_signal_lines.push(format!("wave_fragmentation: {}", wave_fragmentation_text));
    if let Some(value) = self_model_reliability {
        self_model_signal_lines.push(format!("self_model_reliability: {:.2}", value));
    }
    let mut self_report_evidence_ids = input.self_report_snapshot_evidence_ids.clone();
    let mut self_report_lines: Vec<String> = Vec::new();
    if let Some(raw) = input.self_report_snapshot.as_deref() {
        if let Ok(value) = serde_json::from_str::<Value>(raw) {
            if let Some(status) = value.get("status").and_then(|v| v.as_str()) {
                self_report_lines.push(format!("self_report_status: {}", status.trim()));
            }
            if let Some(confidence) = value.get("confidence").and_then(|v| v.as_f64()) {
                self_report_lines.push(format!("self_report_confidence: {:.2}", confidence));
            }
            if let Some(uncertainty) = value.get("uncertainty").and_then(|v| v.as_f64()) {
                self_report_lines.push(format!("self_report_uncertainty: {:.2}", uncertainty));
            }
            if let Some(speculative) = value.get("speculative").and_then(|v| v.as_bool()) {
                self_report_lines.push(format!("self_report_speculative: {}", speculative));
            }
            if let Some(source) = value.get("source").and_then(|v| v.as_str()) {
                self_report_lines.push(format!("self_report_source: {}", source.trim()));
            }
            if let Some(rel) = value
                .get("self_model_reliability")
                .and_then(|v| v.as_f64())
            {
                self_report_lines.push(format!("self_report_reliability: {:.2}", rel));
            }
            if self_report_evidence_ids.is_empty() {
                if let Some(array) = value.get("evidence_event_ids").and_then(|v| v.as_array()) {
                    for entry in array.iter() {
                        if let Some(id) = entry.as_i64() {
                            self_report_evidence_ids.push(id);
                        }
                    }
                }
            }
        } else {
            let clipped = truncate_to_budget(raw.trim(), 180);
            self_report_lines.push(format!("self_report_snapshot_raw: {}", clipped));
        }
    }
    if !self_report_lines.is_empty() {
        self_model_signal_lines.extend(self_report_lines);
    }
    let mut self_model_evidence_ids: Vec<i64> = Vec::new();
    self_model_evidence_ids.extend(qualia_evidence_ids.iter().copied());
    self_model_evidence_ids.extend(wave_evidence_ids.iter().copied());
    self_model_evidence_ids.extend(self_report_evidence_ids.iter().copied());
    if let Some(state) = workspace_state.as_ref() {
        self_model_evidence_ids.extend(collect_workspace_meta_evidence_ids(&state.workspace_meta));
    }
    if let Some(model) = self_model.as_ref() {
        self_model_evidence_ids.extend(extract_unified_state_evidence_ids(
            &model.unified_state_evidence,
        ));
    }
    self_model_evidence_ids.sort();
    self_model_evidence_ids.dedup();
    let self_model_signals = append_evidence_footer(
        &self_model_signal_lines.join("\n"),
        "Self-Model Signals",
        &self_model_evidence_ids,
    );
    let self_model_signals = mark_internal_only(&self_model_signals);

    let qualia_snapshot_enriched = first_person_block(
        "my qualia snapshot",
        &append_evidence_footer(
            input.qualia_snapshot.as_deref().unwrap_or("None"),
            "Qualia",
            &qualia_evidence_ids,
        ),
    );
    let wave_state_enriched = first_person_block(
        "my wave state",
        &append_evidence_footer(
            input.wave_state.as_deref().unwrap_or("None"),
            "Wave",
            &wave_evidence_ids,
        ),
    );
    let attention_schema_enriched = first_person_block(
        "my attention schema",
        &append_evidence_footer(&attention_schema_summary, "Attention", &attention_evidence_ids),
    );
    let workspace_contributors_enriched =
        append_evidence_footer(&workspace_contributors_summary, "Workspace Contributors", &[]);
    let reflective_evidence_ids = input.reflective_narrative_evidence_ids.clone();
    let reflective_narrative_enriched = first_person_block(
        "my reflective narrative",
        &append_evidence_footer(
            input.reflective_narrative.as_deref().unwrap_or("None"),
            "Narrative",
            &reflective_evidence_ids,
        ),
    );

    let qualia_snapshot_block = if suppress_optional {
        "None".to_string()
    } else {
        mark_internal_only(&qualia_snapshot_enriched)
    };
    let wave_state_block = if suppress_optional {
        "None".to_string()
    } else {
        mark_internal_only(&wave_state_enriched)
    };
    let attention_schema_block = if suppress_optional {
        "None".to_string()
    } else {
        mark_internal_only(&attention_schema_enriched)
    };
    let workspace_contributors_block = if suppress_optional {
        "None".to_string()
    } else {
        mark_internal_only(&workspace_contributors_enriched)
    };
    let reflective_narrative_block = if suppress_optional {
        "None".to_string()
    } else {
        mark_internal_only(&reflective_narrative_enriched)
    };
    let gate_decision_block = if suppress_optional {
        "None".to_string()
    } else {
        mark_internal_only(
            input
                .gate_decision
                .as_deref()
                .unwrap_or("None"),
        )
    };

    let mut task_lines = Vec::new();
    task_lines.push(format!("prompt_mode: {}", prompt_mode));
    task_lines.push(format!("task_phase: {}", task_phase));
    if missing_slots != "None" {
        task_lines.push(format!("missing_slots: {}", missing_slots));
    }
    if resolution_mode != "None" {
        task_lines.push(format!("resolution_mode: {}", resolution_mode));
    }
    if policy_notes != "None" {
        task_lines.push(format!("policy_notes: {}", policy_notes));
    }
    let task_context_block = if task_lines.is_empty() {
        "None".to_string()
    } else {
        task_lines.join("\n")
    };

    let mut sections: Vec<PromptSection> = Vec::new();
    let mut section_hashes: HashMap<String, String> = HashMap::new();
    let mut trim_events: Vec<PromptTrimEvent> = Vec::new();
    let mut override_guard_skipped: Vec<String> = Vec::new();
    let mut tier0_compacted = false;
    let override_guard_active = settings
        .stability_prompt_override_guard
        .unwrap_or(true)
        && override_mismatch;

    let mut push_section = |title: &str, body: String, always: bool, allow_diff: bool| {
        if override_guard_active
            && matches!(
                title,
                "Telemetry Snapshot"
                    | "Self-State"
                    | "Controller State"
                    | "Introspection Summary"
                    | "Workspace Snapshot"
            )
        {
            override_guard_skipped.push(title.to_string());
            return;
        }
        if !always && body.trim() == "None" {
            return;
        }
        sections.push(PromptSection {
            title: title.to_string(),
            body,
            always,
            priority: section_priority_for_mode(prompt_mode_enum, title),
            allow_diff,
            tool_hint: tool_hint_for_section(title),
            truncated: false,
        });
    };

    // MVC - always injected, never trimmed
    push_section("Identity Anchor", identity_anchor, true, false);
    push_section("Symbiote System Overview", system_overview_block, true, false);
    push_section("Architecture Map", architecture_map_block, true, false);
    push_section("SYMBIOTE_PHILOSOPHY", symbiote_philosophy_block, true, false);
    push_section("SYMBIOTE_POLICY_SUMMARY", symbiote_policy_summary_block, true, false);
    push_section("Safety Rules", safety_rules_block, true, false);
    push_section("Response Style", response_style_block, true, false);
    push_section(
        "Self-Report Instruction",
        self_report_instruction_block,
        matches!(self_awareness_mode.as_str(), "balanced" | "expressive"),
        false,
    );
    push_section("Tool Availability", tool_availability_block.clone(), true, false);
    push_section("User Input", user_reply.clone(), true, false);
    push_section("Context Spine", context_spine_block, true, false);
    push_section("Working Memory", working_memory_block, true, false);
    push_section("Monologue Intent", monologue_intent_block.clone(), true, false);
    push_section("Monologue Digest", monologue_digest_block.clone(), true, true);
    push_section("Self-Model Signals", self_model_signals, true, false);
    push_section("Audit Log", audit_log_block, true, false);
    push_section(
        "Internal State Summary",
        internal_state_summary_block.to_string(),
        input.self_awareness,
        true,
    );
    push_section("Subject Snapshot", subject_snapshot_block.to_string(), false, true);
    push_section("Gate Decision", gate_decision_block.to_string(), false, true);
    push_section(
        "Feedback Bundle",
        feedback_bundle_block.to_string(),
        input.self_awareness,
        true,
    );
    push_section("Qualia Snapshot", qualia_snapshot_block.to_string(), false, false);
    push_section("Reflective Narrative", reflective_narrative_block.to_string(), false, false);
    push_section("Wave State", wave_state_block.to_string(), false, false);
    push_section("Attention Schema", attention_schema_block.to_string(), false, false);
    push_section(
        "Workspace Contributors",
        workspace_contributors_block.to_string(),
        false,
        false,
    );

    // Task context (policy notes, slots, phase)
    push_section(
        "Task Context",
        task_context_block,
        true,
        true,
    );

    let mut hydrated_context_block = input.hydrated_context.clone();
    let mut capabilities_context_block: Option<String> = None;
    let mut unified_self_block: Option<String> = None;
    let mut autobiographical_block: Option<String> = None;
    if let Some(hydrated) = input.hydrated_context.as_deref() {
        let (caps, remainder) = extract_context_tool_block(hydrated, "get_system_capabilities");
        capabilities_context_block = caps;
        let (unified, remainder) = extract_context_tool_block(&remainder, "get_unified_self");
        unified_self_block = unified;
        let (autobio, remainder) =
            extract_context_tool_block(&remainder, "get_autobiographical_context");
        autobiographical_block = autobio;
        hydrated_context_block = if remainder.trim().is_empty() {
            None
        } else {
            Some(remainder)
        };
    }
    if let Some(unified) = unified_self_block.as_deref() {
        if !unified.trim().is_empty() {
            push_section("Unified Self", unified.to_string(), false, true);
        }
    }
    if let Some(auto) = autobiographical_block.as_deref() {
        if !auto.trim().is_empty() {
            push_section("Autobiographical Context", auto.to_string(), false, true);
        }
    }
    if let Some(caps) = capabilities_context_block.as_deref() {
        if !caps.trim().is_empty() {
            push_section("Capabilities", caps.to_string(), false, true);
        }
    }
    if let Some(hydrated) = hydrated_context_block.as_deref() {
        if !hydrated.trim().is_empty() {
            push_section("Hydrated Context", hydrated.to_string(), false, true);
        }
    }

    // Normal / self-audit priority sections
    push_section("Workspace Snapshot", workspace_snapshot, false, true);
    push_section("Inner Summary", inner_summary.clone(), false, true);
    push_section(
        "Introspection Summary",
        introspection_summary.to_string(),
        false,
        true,
    );
    push_section("Rolling Summary", rolling_summary_block.clone(), false, true);
    push_section("Semantic Hint", semantic_hint.to_string(), false, true);
    push_section("World Model", world_model_block.to_string(), false, true);
    push_section("Memory Context", "{{MEMORY_CONTEXT}}".to_string(), false, true);
    push_section("Episodic Context", "{{EPISODIC_CONTEXT}}".to_string(), false, true);

    let include_diagnostics = input.allow_diagnostics && !suppress_optional;

    // Optional diagnostic sections (lower priority)
    push_section("User Evidence IDs", user_evidence_block.clone(), false, true);
    push_section("Tool Evidence IDs", tool_evidence_block.clone(), false, true);
    push_section("Gate Feedback", gate_feedback_block.clone(), false, true);
    if include_diagnostics {
        push_section("Tool Manifest", tool_manifest_block.clone(), false, true);
        push_section("Telemetry Snapshot", telemetry_block.clone(), false, true);
        push_section("Self-State", self_state_block.clone(), false, true);
        push_section("Controller State", controller_block.clone(), false, true);
    }
    push_section("Identity Thread", identity_block.clone(), false, true);
    push_section("Capabilities and Limitations", capabilities_block.clone(), false, true);
    if include_diagnostics {
        push_section("Capability Manifest", capability_manifest_block.clone(), false, true);
        push_section("KV Memory", kv_memory.clone(), false, true);
    }

    let mut context_hydration: Option<ContextHydrationPlan> = None;
    let mut selected_context_sections: HashSet<String> = HashSet::new();
    if context_budgeter_enabled && context_mode != ContextHydrationMode::Off {
        let (intent_detection, selected, fallback_reason) = compute_context_hydration(
            &input.content,
            input.self_awareness,
            input.self_awareness_hint,
        );
        for section in selected.iter() {
            selected_context_sections.insert(section.clone());
        }
        let all_context_sections = [
            "World Model",
            "Subject Snapshot",
            "Rolling Summary",
            "Inner Summary",
            "Workspace Snapshot",
            "Capabilities",
            "Unified Self",
            "Autobiographical Context",
            "Reflective Narrative",
        ];
        let skipped_sections = all_context_sections
            .iter()
            .filter(|title| !selected_context_sections.contains(&title.to_string()))
            .map(|title| title.to_string())
            .collect::<Vec<_>>();
        context_hydration = Some(ContextHydrationPlan {
            mode: context_mode.as_str().to_string(),
            intent_tags: intent_detection.tags,
            matched_rules: intent_detection.matched_rules,
            selected_sections: selected.clone(),
            skipped_sections,
            max_sections: CONTEXT_BUDGET_MAX_SECTIONS,
            fallback_reason,
        });
        if context_mode == ContextHydrationMode::Thin {
            sections.retain(|section| {
                !is_context_section(&section.title)
                    || selected_context_sections.contains(&section.title)
            });
            if !selected_context_sections.contains("Inner Summary") {
                inner_summary_present = false;
            }
            if !selected_context_sections.contains("Rolling Summary") {
                rolling_summary_present = false;
            }
        }
    }

    // Tier 0 compaction guardrail
    let tier0_titles = ["Self-Model Signals", "Task Context"];
    let mut tier0_indices: Vec<usize> = sections
        .iter()
        .enumerate()
        .filter_map(|(idx, section)| {
            if tier0_titles.contains(&section.title.as_str()) {
                Some(idx)
            } else {
                None
            }
        })
        .collect();
    if !tier0_indices.is_empty() {
        let tier0_tokens: usize = tier0_indices
            .iter()
            .map(|idx| token_estimator::estimate_tokens(&sections[*idx].body))
            .sum();
        if tier0_tokens > TIER0_MAX_TOKENS {
            tier0_compacted = true;
            let per_section = (TIER0_MAX_TOKENS / tier0_indices.len().max(1)).max(1);
            for idx in tier0_indices.drain(..) {
                let original = sections[idx].body.clone();
                let (truncated, did_trim) =
                    token_estimator::truncate_to_token_budget(&original, per_section);
                if did_trim {
                    sections[idx].body = truncated;
                    sections[idx].truncated = true;
                    trim_events.push(PromptTrimEvent {
                        title: sections[idx].title.clone(),
                        original_chars: original.chars().count(),
                        trimmed_chars: sections[idx].body.chars().count(),
                        reason: "tier0_compact".to_string(),
                        hash: Some(hash_string(&sections[idx].body)),
                    });
                }
            }
        }
    }

    // Per-section budgets (token-based with char fallback)
    for section in sections.iter_mut() {
        if !is_protected_section(&section.title) {
            let mut budget_chars = section_budget_chars(&section.title);
            if let Some(context_budget) = context_section_budget_chars(&section.title, context_mode) {
                budget_chars = Some(context_budget);
            }
            let budget_tokens = budget_chars.map(token_estimator::tokens_from_char_budget);
            if let Some(limit_tokens) = budget_tokens {
                let original = section.body.clone();
                let (truncated, did_trim) =
                    token_estimator::truncate_to_token_budget(&original, limit_tokens);
                if did_trim {
                    section.body = truncated;
                    section.truncated = true;
                    trim_events.push(PromptTrimEvent {
                        title: section.title.clone(),
                        original_chars: original.chars().count(),
                        trimmed_chars: section.body.chars().count(),
                        reason: "section_budget".to_string(),
                        hash: Some(hash_string(&section.body)),
                    });
                }
            }
        }
        let hash = hash_string(&section.body);
        section_hashes.insert(section.title.clone(), hash);
    }

    // De-duplicate identical sections (by body hash) to reduce prompt redundancy.
    let mut seen_hashes: HashMap<String, String> = HashMap::new();
    for section in sections.iter_mut() {
        if is_protected_section(&section.title) || !section.allow_diff {
            continue;
        }
        if let Some(hash) = section_hashes.get(&section.title) {
            if let Some(existing) = seen_hashes.get(hash) {
                let pointer = format!("[deduped with {}]", existing);
                trim_events.push(PromptTrimEvent {
                    title: section.title.clone(),
                    original_chars: section.body.chars().count(),
                    trimmed_chars: pointer.chars().count(),
                    reason: "dedupe".to_string(),
                    hash: Some(hash.clone()),
                });
                section.body = pointer;
                section.truncated = true;
            } else {
                seen_hashes.insert(hash.clone(), section.title.clone());
            }
        }
    }

    // Diffing for unchanged sections
    for section in sections.iter_mut() {
        if !section.allow_diff || is_protected_section(&section.title) {
            continue;
        }
        if let Some(prev_hash) = prompt_section_hashes.get(&section.title) {
            if let Some(current_hash) = section_hashes.get(&section.title) {
                if prev_hash == current_hash {
                    let hint = section
                        .tool_hint
                        .as_deref()
                        .map(|t| format!(" Use tool {} if needed.", t))
                        .unwrap_or_default();
                    let pointer = format!("[unchanged, hash: {}]{}", current_hash, hint);
                    trim_events.push(PromptTrimEvent {
                        title: section.title.clone(),
                        original_chars: section.body.chars().count(),
                        trimmed_chars: pointer.chars().count(),
                        reason: "diff_unchanged".to_string(),
                        hash: Some(current_hash.clone()),
                    });
                    section.body = pointer;
                    section.truncated = true;
                }
            }
        }
    }

    let limit_tokens = context_limit_tokens(&settings);
    let mut max_prompt_tokens = limit_tokens.saturating_sub(token_estimator::safety_margin_tokens(limit_tokens));
    if matches!(layout, PromptLayout::Compact) {
        max_prompt_tokens = (max_prompt_tokens as f32 * 0.75) as usize;
    }
    let mut prompt_overflow = false;

    // Budgeter: drop lowest priority sections until within limit
    loop {
        let total_tokens = sections
            .iter()
            .map(|s| {
                let formatted = format_section_cached(&s.title, &s.body);
                estimate_tokens_cached(&formatted)
            })
            .sum::<usize>();
        if total_tokens <= max_prompt_tokens {
            break;
        }
        let mut drop_index: Option<usize> = None;
        let mut worst_priority = -1i32;
        let mut worst_size = 0usize;
        for (idx, section) in sections.iter().enumerate() {
            if section.always || is_anchor_section(&section.title) {
                continue;
            }
            if section.priority > worst_priority
                || (section.priority == worst_priority && section.body.len() > worst_size)
            {
                worst_priority = section.priority;
                worst_size = section.body.len();
                drop_index = Some(idx);
            }
        }
        let Some(idx) = drop_index else {
            break;
        };
        let compactible = matches!(
            sections[idx].title.as_str(),
            "Gate Feedback" | "Capabilities and Limitations"
        );
        if compactible {
            let original = sections[idx].body.clone();
            let target = 240usize;
            if original.chars().count() > target {
                sections[idx].body = truncate_to_budget(&original, target);
                sections[idx].truncated = true;
                trim_events.push(PromptTrimEvent {
                    title: sections[idx].title.clone(),
                    original_chars: original.chars().count(),
                    trimmed_chars: sections[idx].body.chars().count(),
                    reason: "budget_compact".to_string(),
                    hash: Some(hash_string(&sections[idx].body)),
                });
                continue;
            }
        }
        let dropped = sections.remove(idx);
        trim_events.push(PromptTrimEvent {
            title: dropped.title.clone(),
            original_chars: dropped.body.chars().count(),
            trimmed_chars: 0,
            reason: "budget_drop".to_string(),
            hash: section_hashes.get(&dropped.title).cloned(),
        });
    }

    // Anchor floors: keep critical anchors above minimum size even under pressure.
    loop {
        let total_tokens = sections
            .iter()
            .map(|s| {
                let formatted = format_section_cached(&s.title, &s.body);
                estimate_tokens_cached(&formatted)
            })
            .sum::<usize>();
        if total_tokens <= max_prompt_tokens {
            break;
        }
        let mut reduced = false;
        for section in sections.iter_mut() {
            let Some(mut floor) = anchor_floor_chars_for_mode(&section.title, context_mode) else {
                continue;
            };
            if section.title == "User Input" && floor < USER_INPUT_HARD_FLOOR_CHARS {
                floor = USER_INPUT_HARD_FLOOR_CHARS;
            }
            let current_chars = section.body.chars().count();
            if current_chars <= floor {
                continue;
            }
            let original = section.body.clone();
            let limit_tokens = token_estimator::tokens_from_char_budget(floor);
            let (truncated, did_trim) = token_estimator::truncate_to_token_budget(&original, limit_tokens);
            if did_trim {
                section.body = truncated;
                section.truncated = true;
                trim_events.push(PromptTrimEvent {
                    title: section.title.clone(),
                    original_chars: original.chars().count(),
                    trimmed_chars: section.body.chars().count(),
                    reason: "anchor_floor".to_string(),
                    hash: Some(hash_string(&section.body)),
                });
                reduced = true;
            }
        }
        if !reduced {
            let mut emergency_reduced = false;
            for section in sections.iter_mut() {
                if section.title == "User Input" || !is_anchor_section(&section.title) {
                    continue;
                }
                let current_chars = section.body.chars().count();
                if current_chars <= EMERGENCY_ANCHOR_FLOOR_CHARS {
                    continue;
                }
                let original = section.body.clone();
                let limit_tokens = token_estimator::tokens_from_char_budget(EMERGENCY_ANCHOR_FLOOR_CHARS);
                let (truncated, did_trim) =
                    token_estimator::truncate_to_token_budget(&original, limit_tokens);
                if did_trim {
                    section.body = truncated;
                    section.truncated = true;
                    trim_events.push(PromptTrimEvent {
                        title: section.title.clone(),
                        original_chars: original.chars().count(),
                        trimmed_chars: section.body.chars().count(),
                        reason: "anchor_floor_emergency".to_string(),
                        hash: Some(hash_string(&section.body)),
                    });
                    emergency_reduced = true;
                }
            }
            if emergency_reduced {
                continue;
            }
            let mut dropped_any = false;
            for title in ANCHOR_DROP_PRIORITY {
                let total_tokens = sections
                    .iter()
                    .map(|s| {
                        let formatted = format_section_cached(&s.title, &s.body);
                        estimate_tokens_cached(&formatted)
                    })
                    .sum::<usize>();
                if total_tokens <= max_prompt_tokens {
                    break;
                }
                if let Some(idx) = sections.iter().position(|s| s.title == *title) {
                    let dropped = sections.remove(idx);
                    trim_events.push(PromptTrimEvent {
                        title: dropped.title.clone(),
                        original_chars: dropped.body.chars().count(),
                        trimmed_chars: 0,
                        reason: "anchor_floor_shed".to_string(),
                        hash: section_hashes.get(&dropped.title).cloned(),
                    });
                    dropped_any = true;
                }
            }
            if dropped_any {
                continue;
            }
            trim_events.push(PromptTrimEvent {
                title: "User Input".to_string(),
                original_chars: 0,
                trimmed_chars: 0,
                reason: "anchor_floor_exceeded".to_string(),
                hash: None,
            });
            break;
        }
    }

    // Overflow guard: compact always sections if still over budget.
    loop {
        let total_tokens = sections
            .iter()
            .map(|s| {
                let formatted = format_section_cached(&s.title, &s.body);
                estimate_tokens_cached(&formatted)
            })
            .sum::<usize>();
        if total_tokens <= max_prompt_tokens {
            break;
        }
        let mut candidates: Vec<usize> = sections
            .iter()
            .enumerate()
            .filter(|(_, s)| s.always && !is_anchor_section(&s.title))
            .map(|(idx, _)| idx)
            .collect();
        if candidates.is_empty() {
            candidates = sections
                .iter()
                .enumerate()
                .filter(|(_, s)| s.always && s.title != "User Input")
                .map(|(idx, _)| idx)
                .collect();
        }
        candidates.sort_by(|a, b| {
            let a_sec = &sections[*a];
            let b_sec = &sections[*b];
            b_sec
                .priority
                .cmp(&a_sec.priority)
                .then_with(|| b_sec.body.len().cmp(&a_sec.body.len()))
        });
        let mut reduced = false;
        for idx in candidates {
            let section = &mut sections[idx];
            let current_chars = section.body.chars().count();
            let floor = anchor_floor_chars(&section.title).unwrap_or(200);
            let target = (current_chars as f32 * 0.75) as usize;
            if target <= floor || current_chars <= floor {
                continue;
            }
            let original = section.body.clone();
            section.body = truncate_to_budget(&original, target);
            section.truncated = true;
            trim_events.push(PromptTrimEvent {
                title: section.title.clone(),
                original_chars: original.chars().count(),
                trimmed_chars: section.body.chars().count(),
                reason: "prompt_overflow_compact".to_string(),
                hash: Some(hash_string(&section.body)),
            });
            reduced = true;
            break;
        }
        if !reduced {
            prompt_overflow = true;
            trim_events.push(PromptTrimEvent {
                title: "Prompt Overflow".to_string(),
                original_chars: 0,
                trimmed_chars: 0,
                reason: "prompt_overflow".to_string(),
                hash: None,
            });
            break;
        }
    }

    let mut section_metrics: Vec<PromptSectionMetric> = Vec::new();
    let mut rendered_sections: Vec<String> = Vec::new();
    for section in sections.iter() {
        let body = section.body.trim();
        let formatted = format_section_cached(&section.title, body);
        let budget = section_budget_chars(&section.title);
        let budget_tokens = budget.map(token_estimator::tokens_from_char_budget);
        let tokens = estimate_tokens_cached(&formatted);
        section_metrics.push(PromptSectionMetric {
            title: section.title.clone(),
            chars: formatted.chars().count(),
            lines: formatted.lines().count(),
            tokens,
            truncated: section.truncated,
            budget,
            budget_tokens,
        });
        rendered_sections.push(formatted);
    }

    let system_message = rendered_sections.join("\n\n");
    let total_chars = system_message.chars().count();
    let total_tokens = estimate_tokens_cached(&system_message);
    let tier_trim_summary = Some(json!({
        "tier0_compacted": tier0_compacted,
        "trim_events": trim_events.iter().map(|e| json!({
            "title": e.title,
            "reason": e.reason,
        })).collect::<Vec<_>>(),
    }));
    Ok(CorePromptBuild {
        system_message,
        prompt_layout: layout,
        prompt_source,
        primary_prompt_hash,
        memory_prompt_hash,
        canonical_primary_hash,
        policy_canon_hash,
        override_hash,
        override_active,
        override_mismatch,
        override_guard_skipped,
        workspace_hash,
        inner_summary_hash,
        rolling_summary_hash,
        capability_manifest_hash,
        workspace_present,
        inner_summary_present,
        rolling_summary_present,
        introspection_present,
        semantic_hint_present,
        section_metrics,
        section_hashes,
        trim_events,
        tier_trim_summary,
        tier0_compacted,
        context_hydration,
        total_chars,
        total_tokens,
        user_evidence_count,
        tool_evidence_count,
        prompt_overflow,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::fs;
    use std::path::PathBuf;

    async fn setup_pool() -> sqlx::SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("pool");
        let schema_path = PathBuf::from("src/db/schema.sql");
        let schema_sql = fs::read_to_string(&schema_path).expect("schema");
        sqlx::query(&schema_sql).execute(&pool).await.expect("apply schema");
        sqlx::query("INSERT INTO settings (id, schema_version, api_base_url) VALUES (1, 1, 'http://localhost')")
            .execute(&pool)
            .await
            .expect("seed settings");
        pool
    }

    fn base_input() -> CorePromptInput {
        CorePromptInput {
            content: "Hello".to_string(),
            kind: CoreInputKind::User,
            source: "input".to_string(),
            self_awareness: false,
            self_awareness_hint: false,
            anchor_hits: 1,
            original_input: "Hello".to_string(),
            current_time: None,
            semantic_hint: None,
            introspection_summary: None,
            monologue_intent: None,
            monologue_digest: None,
            prompt_mode: None,
            task_phase: None,
            missing_slots: None,
            resolution_mode: None,
            policy_notes: None,
            redirect_focus: None,
            allow_diagnostics: false,
            world_model_snapshot: None,
            subject_snapshot: None,
            gate_decision: None,
            feedback_bundle: None,
            qualia_snapshot: None,
            attention_schema_summary: None,
            workspace_contributors_summary: None,
            wave_state: None,
            reflective_narrative: None,
            reflective_narrative_evidence_ids: Vec::new(),
            self_report_snapshot: None,
            self_report_snapshot_evidence_ids: Vec::new(),
            context_spine: None,
            hydrated_context: None,
        }
    }

    #[test]
    fn gate_feedback_priority_is_high() {
        let priority = section_priority_for_mode(PromptMode::Normal, "Gate Feedback");
        assert!(priority <= 4);
    }

    #[test]
    fn capabilities_priority_is_high() {
        let priority = section_priority_for_mode(PromptMode::Normal, "Capabilities and Limitations");
        assert!(priority <= 5);
    }

    #[tokio::test]
    async fn prompt_includes_user_evidence_ids() {
        let pool = setup_pool().await;
        let db = Db { pool: pool.clone() };
        let evidence_id = db
            .create_user_utterance_evidence("conv", "msg1", "Hello")
            .await
            .expect("evidence");
        let input = base_input();
        let build = build_core_system_message_with_layout(&db, "conv", &input, PromptLayout::Compact)
            .await
            .expect("build");
        assert!(build.system_message.contains("User Evidence IDs"));
        assert!(build.system_message.contains(&evidence_id.to_string()));
    }

    #[tokio::test]
    async fn prompt_includes_tool_evidence_ids() {
        let pool = setup_pool().await;
        let db = Db { pool: pool.clone() };
        let mut input = base_input();
        input.kind = CoreInputKind::ToolResult;
        input.content = r#"{"key":"foo","value":"bar","evidence_event_id": 42}"#.to_string();
        let build = build_core_system_message_with_layout(&db, "conv", &input, PromptLayout::Full)
            .await
            .expect("build");
        assert!(build.system_message.contains("Tool Evidence IDs"));
        assert!(build.system_message.contains("42"));
    }

    #[tokio::test]
    async fn prompt_includes_feedback_bundle_when_provided() {
        let pool = setup_pool().await;
        let db = Db { pool: pool.clone() };
        let mut input = base_input();
        input.feedback_bundle = Some("last_turn_outcome: success".to_string());
        let build = build_core_system_message_with_layout(&db, "conv", &input, PromptLayout::Full)
            .await
            .expect("build");
        assert!(build.system_message.contains("Feedback Bundle"));
        assert!(build.system_message.contains("last_turn_outcome: success"));
    }

    #[tokio::test]
    async fn prompt_includes_policy_canon_hash() {
        let pool = setup_pool().await;
        let db = Db { pool: pool.clone() };
        let input = base_input();
        let build = build_core_system_message_with_layout(&db, "conv", &input, PromptLayout::Full)
            .await
            .expect("build");
        assert!(!build.policy_canon_hash.trim().is_empty());
    }

    #[tokio::test]
    async fn prompt_includes_balanced_self_awareness_template() {
        let pool = setup_pool().await;
        let db = Db { pool: pool.clone() };
        let mut settings = db.get_settings().await.expect("settings");
        settings.self_awareness_expression_mode = Some("balanced".to_string());
        db.update_settings(settings).await.expect("update");
        let mut input = base_input();
        input.self_awareness = true;
        let build = build_core_system_message_with_layout(&db, "conv", &input, PromptLayout::Full)
            .await
            .expect("build");
        assert!(build.system_message.contains("Template (2-3 sentences)"));
        assert!(build.system_message.contains("Self-Report Format"));
    }

    #[test]
    fn policy_canon_hash_changes_when_input_changes() {
        let h1 = hash_text("policy alpha");
        let h2 = hash_text("policy beta");
        assert_ne!(h1, h2);
    }
}
