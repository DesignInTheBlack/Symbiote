use super::*;
use crate::core::self_model_controller;
use super::arbitration::{
    QualiaModulationContext,
    ResidualInfluenceContext,
    ResidualInfluenceMode,
    WaveArbitrationContext,
};
use crate::core::sensitivity::{phi_consent_allowed, redact_sensitive_text, redact_sensitive_json};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex as StdMutex;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

pub(crate) struct RunOutput {
    pub response: String,
    pub tool_result: Option<ToolExecutionResult>,
    pub assistant_message_id: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KernelPipelineMode {
    Legacy,
    Phased,
}

fn kernel_pipeline_mode() -> KernelPipelineMode {
    match std::env::var("SYMBIOTE_KERNEL_PIPELINE_MODE")
        .unwrap_or_else(|_| "phased".to_string())
        .to_lowercase()
        .as_str()
    {
        "legacy" | "v1" => KernelPipelineMode::Legacy,
        _ => KernelPipelineMode::Phased,
    }
}

struct RateLimitState {
    last_emit: Instant,
    suppressed: usize,
}

const PROMPT_TRIM_CRITICAL_WINDOW_SECS: u64 = 60;
const MONOLOGUE_PARSE_FAIL_WINDOW_SECS: u64 = 30;

static PROMPT_TRIM_CRITICAL_RATE: Lazy<StdMutex<HashMap<String, RateLimitState>>> =
    Lazy::new(|| StdMutex::new(HashMap::new()));
static MONOLOGUE_PARSE_FAIL_RATE: Lazy<StdMutex<HashMap<String, RateLimitState>>> =
    Lazy::new(|| StdMutex::new(HashMap::new()));

fn rate_limit_event(
    map: &Lazy<StdMutex<HashMap<String, RateLimitState>>>,
    key: &str,
    window: Duration,
) -> (bool, usize) {
    let mut guard = map.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    let entry = guard
        .entry(key.to_string())
        .or_insert(RateLimitState {
            last_emit: now.checked_sub(window).unwrap_or(now),
            suppressed: 0,
        });
    if now.duration_since(entry.last_emit) < window {
        entry.suppressed = entry.suppressed.saturating_add(1);
        return (false, 0);
    }
    let suppressed = entry.suppressed;
    entry.suppressed = 0;
    entry.last_emit = now;
    (true, suppressed)
}


fn parse_tool_name_filter(raw: Option<&str>) -> Vec<String> {
    let raw = raw.unwrap_or("").trim();
    if raw.is_empty() {
        return Vec::new();
    }
    let mut entries: Vec<String> = Vec::new();
    if raw.starts_with('[') {
        if let Ok(list) = serde_json::from_str::<Vec<String>>(raw) {
            entries.extend(list);
        }
    }
    if entries.is_empty() {
        entries.extend(
            raw.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
        );
    }
    entries
        .into_iter()
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

fn context_mode_is_thin(settings: &Settings) -> bool {
    settings
        .context_hydration_mode
        .as_deref()
        .unwrap_or("shadow")
        .trim()
        .eq_ignore_ascii_case("thin")
}

fn is_jsonish_text(raw: &str) -> bool {
    let trimmed = raw.trim();
    if trimmed.len() < 2 {
        return false;
    }
    let lower = trimmed.to_lowercase();
    let starts_json = trimmed.starts_with('{') || trimmed.starts_with('[');
    let has_marker = lower.contains("\"stance\"")
        || lower.contains("\"candidates\"")
        || lower.contains("\"decision_packet\"")
        || lower.contains("\"required_slots\"")
        || lower.contains("\"done\"")
        || lower.contains("\"message\"");
    if starts_json && has_marker {
        return true;
    }
    let wrapped = (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'));
    if !wrapped {
        return false;
    }
    if has_marker {
        return true;
    }
    serde_json::from_str::<Value>(trimmed).is_ok()
}

#[derive(Default)]
struct PrimaryResponsePacket {
    message: Option<String>,
    candidates: Vec<Value>,
    decision_packet: Option<Value>,
}

fn extract_primary_json_string(value: &Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(text) = value.get(*key).and_then(|v| v.as_str()) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn parse_primary_response_packet(raw: &str) -> Option<PrimaryResponsePacket> {
    let trimmed = raw.trim();
    if trimmed.len() < 2 {
        return None;
    }
    if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
        return None;
    }
    let parsed: Value = serde_json::from_str(trimmed).ok()?;
    let mut packet = PrimaryResponsePacket::default();
    match parsed {
        Value::Object(obj) => {
            let obj_val = Value::Object(obj.clone());
            packet.message = extract_primary_json_string(&obj_val, &["message", "text", "content"]);
            if let Some(items) = obj_val.get("candidates").and_then(|v| v.as_array()) {
                packet.candidates = items.iter().cloned().collect();
            }
            if let Some(packet_val) = obj_val.get("decision_packet") {
                packet.decision_packet = Some(packet_val.clone());
            }
        }
        Value::Array(items) => {
            packet.candidates = items;
        }
        _ => return None,
    }
    if packet.message.is_none() && packet.candidates.is_empty() && packet.decision_packet.is_none() {
        None
    } else {
        Some(packet)
    }
}

#[cfg(test)]
pub(crate) fn unwrap_primary_response_message(raw: &str) -> Option<String> {
    parse_primary_response_packet(raw).and_then(|packet| packet.message)
}

fn looks_like_question(prompt: &str) -> bool {
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.contains('?') {
        return true;
    }
    let word_count = trimmed.split_whitespace().count();
    let alpha_count = trimmed.chars().filter(|c| c.is_alphabetic()).count();
    word_count >= 3 && alpha_count >= 10
}

fn parse_monologue_fallback(raw: &str, stance_label: &str) -> Option<Value> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if is_jsonish_text(trimmed) {
        return Some(json!({
            "stance": stance_label,
            "message": "",
            "done": true,
        }));
    }

    let mut stance: Option<String> = None;
    let mut done: Option<bool> = None;
    let mut message: Option<String> = None;
    let mut capture_message = false;
    let mut message_lines: Vec<String> = Vec::new();

    for line in trimmed.lines() {
        let line_trimmed = line.trim();
        if line_trimmed.is_empty() {
            if capture_message {
                message_lines.push(String::new());
            }
            continue;
        }
        let lower = line_trimmed.to_lowercase();

        if lower.starts_with("skeptic:") || lower.starts_with("synth:") {
            let parts: Vec<&str> = line_trimmed.splitn(2, ':').collect();
            if parts.len() == 2 {
                stance = Some(parts[0].trim().to_lowercase());
                let msg = parts[1].trim();
                if !msg.is_empty() {
                    message = Some(msg.to_string());
                    capture_message = false;
                } else {
                    capture_message = true;
                }
                continue;
            }
        }

        if let Some((key, value)) = split_monologue_key_value(line_trimmed) {
            match key.as_str() {
                "stance" => {
                    let normalized = value.to_lowercase();
                    if normalized == "skeptic" || normalized == "synth" {
                        stance = Some(normalized);
                    }
                    continue;
                }
                "done" => {
                    let normalized = value.to_lowercase();
                    done = match normalized.as_str() {
                        "true" | "yes" | "1" => Some(true),
                        "false" | "no" | "0" => Some(false),
                        _ => done,
                    };
                    continue;
                }
                "message" => {
                    if value.is_empty() {
                        capture_message = true;
                    } else {
                        message = Some(value);
                        capture_message = false;
                    }
                    continue;
                }
                _ => {}
            }
        }

        if capture_message || message.is_none() {
            message_lines.push(line_trimmed.to_string());
        }
    }

    let message = if let Some(mut base) = message {
        if capture_message && !message_lines.is_empty() {
            base = format!("{}\n{}", base, message_lines.join("\n")).trim().to_string();
        }
        base
    } else if !message_lines.is_empty() {
        message_lines.join("\n").trim().to_string()
    } else {
        trimmed.to_string()
    };

    let stance = stance.unwrap_or_else(|| stance_label.to_string());
    let done = done.unwrap_or(false);

    Some(json!({
        "stance": stance,
        "message": message,
        "done": done,
    }))
}

fn split_monologue_key_value(line: &str) -> Option<(String, String)> {
    let (key, rest) = line.split_once(':').or_else(|| line.split_once('='))?;
    let key = key.trim().to_lowercase();
    if key.is_empty() {
        return None;
    }
    Some((key, rest.trim().to_string()))
}

fn prompt_section_included(prompt_build: Option<&CorePromptBuild>, title: &str) -> bool {
    prompt_build
        .map(|build| build.section_hashes.contains_key(title))
        .unwrap_or(false)
}

fn detect_context_miss_tool(response: &str, prompt_build: Option<&CorePromptBuild>) -> Option<&'static str> {
    let lower = response.to_lowercase();
    let has_world = prompt_section_included(prompt_build, "World Model");
    let has_subject = prompt_section_included(prompt_build, "Subject Snapshot");
    let has_rolling = prompt_section_included(prompt_build, "Rolling Summary");
    let has_inner = prompt_section_included(prompt_build, "Inner Summary");
    let has_workspace = prompt_section_included(prompt_build, "Workspace Snapshot");
    let has_capabilities = prompt_section_included(prompt_build, "Capabilities");
    let has_unified = prompt_section_included(prompt_build, "Unified Self");
    let has_autobio = prompt_section_included(prompt_build, "Autobiographical Context");

    let world_triggers = ["world model", "belief graph", "beliefs", "facts about", "state of the world"];
    if !has_world && world_triggers.iter().any(|t| lower.contains(t)) {
        return Some("get_world_model_snapshot");
    }

    let plan_triggers = ["plan", "strategy", "roadmap", "steps", "verified plan", "plan state"];
    if !has_subject && plan_triggers.iter().any(|t| lower.contains(t)) {
        return Some("get_plan_summary");
    }

    let goal_triggers = ["goal stack", "goal thread", "current goals", "progress", "next step", "milestone"];
    if !(has_subject || has_workspace) && goal_triggers.iter().any(|t| lower.contains(t)) {
        return Some("get_goal_stack");
    }

    let recall_triggers = ["rolling summary", "earlier", "previous", "last time", "remember", "recall"];
    if !has_rolling && recall_triggers.iter().any(|t| lower.contains(t)) {
        return Some("get_rolling_summary");
    }

    let inner_triggers = ["inner summary", "internal summary"];
    if !has_inner && inner_triggers.iter().any(|t| lower.contains(t)) {
        return Some("get_inner_summary");
    }

    let outcome_triggers = ["outcome", "decision report", "decision reports", "result history"];
    if outcome_triggers.iter().any(|t| lower.contains(t)) {
        return Some("get_recent_outcomes");
    }

    let capability_triggers = [
        "capabilities",
        "what can you do",
        "trace view",
        "system controls",
        "gate decisions",
        "audit panel",
    ];
    if !has_capabilities && capability_triggers.iter().any(|t| lower.contains(t)) {
        return Some("get_system_capabilities");
    }

    let unified_triggers = [
        "unified self",
        "self model",
        "self-model",
        "internal state",
        "your system",
        "self snapshot",
        "self description",
    ];
    if !has_unified && unified_triggers.iter().any(|t| lower.contains(t)) {
        return Some("get_unified_self");
    }

    let autobio_triggers = [
        "autobiographical",
        "life story",
        "my history",
        "your history",
        "background",
        "what have you experienced",
        "who are you",
    ];
    if !has_autobio && autobio_triggers.iter().any(|t| lower.contains(t)) {
        return Some("get_autobiographical_context");
    }

    None
}

pub(crate) fn build_qualia_modulation_context(
    state: &qualia::QualiaState,
) -> Option<QualiaModulationContext> {
    let tag = state
        .dominant_tag
        .as_deref()
        .or(state.predicted_tag.as_deref())
        .unwrap_or("neutral")
        .trim();
    if tag.is_empty() {
        return None;
    }
    let intensity = state.dominant_intensity as f32;
    let confidence = state.prediction_confidence as f32;
    if intensity <= 0.0 && confidence <= 0.0 {
        return None;
    }
    Some(QualiaModulationContext {
        tag: tag.to_string(),
        intensity: intensity.clamp(0.0, 1.0),
        confidence: confidence.clamp(0.0, 1.0),
    })
}

fn extract_plan_id_from_step_id(step_id: &str) -> Option<String> {
    let trimmed = step_id.trim();
    if trimmed.is_empty() {
        return None;
    }
    let prefix = trimmed.split(':').next().unwrap_or("").trim();
    if prefix.is_empty() {
        None
    } else {
        Some(prefix.to_string())
    }
}

fn append_uncertainty_marker(content: &str, _controller_state: Option<&ControllerState>) -> String {
    // UX: never append evidence markers to user-visible responses.
    content.to_string()
}

fn extract_monologue_surface_json(raw: &str) -> Option<Value> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if end <= start {
        return None;
    }
    let candidate = trimmed.get(start..=end)?;
    if !is_jsonish_text(candidate) {
        return None;
    }
    serde_json::from_str::<Value>(candidate).ok()
}

fn format_monologue_surface_content(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return raw.to_string();
    }
    let jsonish = is_jsonish_text(trimmed);
    let Some(value) = extract_monologue_surface_json(trimmed) else {
        return if jsonish { String::new() } else { raw.to_string() };
    };
    let message = value
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if message.is_empty() {
        return String::new();
    }
    let stance = value
        .get("stance")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_lowercase();
    let stance_label = match stance.as_str() {
        "skeptic" => "Skeptic".to_string(),
        "synth" => "Synth".to_string(),
        "" => String::new(),
        other => other.to_string(),
    };
    let descriptors = value
        .get("descriptors")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut output = String::new();
    if !stance_label.is_empty() {
        output.push_str(&stance_label);
        output.push_str(": ");
    }
    output.push_str(&message);
    if !descriptors.is_empty() {
        output.push_str(" (");
        output.push_str(&descriptors.join(", "));
        output.push(')');
    }
    output
}

fn strip_protocol_tags_final(output: &str) -> String {
    if output.is_empty() {
        return String::new();
    }
    let mut cleaned_lines: Vec<String> = Vec::new();
    for line in output.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        if matches!(
            lower.as_str(),
            "<<memory>>" | "<<clarify>>" | "<<resolve>>"
        ) {
            continue;
        }
        let mut cleaned = line.to_string();
        for tag in ["<<MEMORY>>", "<<CLARIFY>>", "<<RESOLVE>>"] {
            cleaned = cleaned.replace(tag, "");
            cleaned = cleaned.replace(&tag.to_ascii_lowercase(), "");
        }
        if cleaned.trim().is_empty() {
            continue;
        }
        cleaned_lines.push(cleaned);
    }
    cleaned_lines.join("\n").trim_end().to_string()
}

fn tokenize_for_similarity(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch);
        } else if !current.is_empty() {
            if current.len() > 2 {
                tokens.push(current.clone());
            }
            current.clear();
        }
    }
    if !current.is_empty() && current.len() > 2 {
        tokens.push(current);
    }
    tokens
}

fn jaccard_similarity(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let set_a: std::collections::HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let set_b: std::collections::HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let intersection = set_a.intersection(&set_b).count() as f32;
    let union = (set_a.len() + set_b.len()) as f32 - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
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

fn merge_unique_ids(target: &mut Vec<i64>, source: &[i64]) -> bool {
    let mut changed = false;
    for id in source.iter().copied() {
        if id <= 0 {
            continue;
        }
        if !target.contains(&id) {
            target.push(id);
            changed = true;
        }
    }
    changed
}

fn link_goal_step_to_hypotheses(
    goal_stack: &mut [crate::models::GoalStackItem],
    hypotheses: &[crate::models::WorkspaceHypothesis],
) -> bool {
    if hypotheses.is_empty() {
        return false;
    }
    for item in goal_stack.iter_mut() {
        let goal = item.goal.trim();
        if goal.is_empty() {
            continue;
        }
        if goal_status_complete(item.status.as_deref()) {
            continue;
        }
        let (target_text, evidence_ids, belief_ids) = if let Some(step) =
            item.steps.get_mut(item.current_step_index)
        {
            (step.text.trim(), &mut step.evidence_event_ids, &mut step.belief_ids)
        } else {
            (goal, &mut item.evidence_event_ids, &mut item.belief_ids)
        };
        if target_text.is_empty() {
            return false;
        }
        let target_tokens = tokenize_for_similarity(target_text);
        if target_tokens.is_empty() {
            return false;
        }
        let mut best_score = 0.0f32;
        let mut best_match: Option<&crate::models::WorkspaceHypothesis> = None;
        for hypothesis in hypotheses.iter() {
            let hyp_text = hypothesis.text.trim();
            if hyp_text.is_empty() {
                continue;
            }
            let hyp_tokens = tokenize_for_similarity(hyp_text);
            if hyp_tokens.is_empty() {
                continue;
            }
            let score = jaccard_similarity(&target_tokens, &hyp_tokens);
            if score > best_score {
                best_score = score;
                best_match = Some(hypothesis);
            }
        }
        let Some(best) = best_match else {
            return false;
        };
        if best_score < 0.3 {
            return false;
        }
        let mut changed = false;
        changed |= merge_unique_ids(evidence_ids, &best.evidence_event_ids);
        changed |= merge_unique_ids(belief_ids, &best.belief_ids);
        return changed;
    }
    false
}

fn parse_context_limit_from_error(err: &str) -> Option<i32> {
    let lowered = err.to_lowercase();
    let patterns = [
        "maximum context length",
        "max context length",
        "context length",
        "maximum tokens",
        "max tokens",
    ];
    for pattern in patterns {
        if let Some(idx) = lowered.find(pattern) {
            let rest = &lowered[idx + pattern.len()..];
            if let Some(val) = extract_first_int(rest) {
                return Some(val);
            }
        }
    }
    let mut nums: Vec<i32> = Vec::new();
    let mut current = String::new();
    for ch in lowered.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            if let Ok(val) = current.parse::<i32>() {
                nums.push(val);
            }
            current.clear();
        }
    }
    if !current.is_empty() {
        if let Ok(val) = current.parse::<i32>() {
            nums.push(val);
        }
    }
    nums.into_iter().filter(|v| *v > 0).min()
}

fn extract_first_int(input: &str) -> Option<i32> {
    let mut current = String::new();
    for ch in input.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            break;
        }
    }
    if current.is_empty() {
        None
    } else {
        current.parse::<i32>().ok()
    }
}

fn ensure_candidate_evidence_fields(payload: &mut Value) {
    let Some(obj) = payload.as_object_mut() else {
        return;
    };
    if !obj.contains_key("evidence_event_ids") {
        let mut ids = Vec::new();
        if let Some(single) = obj.get("evidence_event_id").and_then(|v| v.as_i64()) {
            ids.push(single);
        }
        obj.insert("evidence_event_ids".to_string(), json!(ids));
    }
    if !obj.contains_key("belief_ids") {
        let mut ids = Vec::new();
        if let Some(single) = obj.get("belief_id").and_then(|v| v.as_i64()) {
            ids.push(single);
        }
        obj.insert("belief_ids".to_string(), json!(ids));
    }
}

fn coerce_prediction_response(value: &Value) -> PredictionResponse {
    let mut predictions_value: Option<&Value> = None;
    let mut rejection_reason: Option<String> = None;

    match value {
        Value::Object(obj) => {
            if let Some(reason) = obj.get("rejection_reason").and_then(|v| v.as_str()) {
                rejection_reason = Some(reason.to_string());
            } else if let Some(reason) = obj.get("reason").and_then(|v| v.as_str()) {
                rejection_reason = Some(reason.to_string());
            }

            if let Some(predictions) = obj.get("predictions") {
                predictions_value = Some(predictions);
            } else if let Some(prediction) = obj.get("prediction") {
                predictions_value = Some(prediction);
            }
        }
        Value::Array(_) => {
            predictions_value = Some(value);
        }
        _ => {}
    }

    let mut predictions: Vec<PredictionCandidate> = Vec::new();
    if let Some(predictions_value) = predictions_value {
        match predictions_value {
            Value::Array(items) => {
                for item in items {
                    if let Some(candidate) = coerce_prediction_candidate(item) {
                        predictions.push(candidate);
                    }
                }
            }
            Value::Object(_) => {
                if let Some(candidate) = coerce_prediction_candidate(predictions_value) {
                    predictions.push(candidate);
                }
            }
            _ => {}
        }
    }

    let predictions = if predictions.is_empty() { None } else { Some(predictions) };

    PredictionResponse {
        predictions,
        rejection_reason,
    }
}

fn coerce_prediction_candidate(value: &Value) -> Option<PredictionCandidate> {
    let obj = value.as_object()?;
    let metric = obj
        .get("metric")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())?;
    if metric.is_empty() {
        return None;
    }

    let expected_value = coerce_prediction_f64(obj.get("expected_value").or_else(|| obj.get("expected")))
        .unwrap_or(0.0);
    let expected_variance = coerce_prediction_f64(obj.get("expected_variance"));
    let horizon = obj
        .get("horizon")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "next_turn".to_string());
    let confidence = coerce_prediction_f64(obj.get("confidence"));
    let evidence_event_ids = obj
        .get("evidence_event_ids")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<_>>());

    Some(PredictionCandidate {
        metric,
        expected_value,
        expected_variance,
        horizon,
        confidence,
        evidence_event_ids,
    })
}

fn coerce_prediction_f64(value: Option<&Value>) -> Option<f64> {
    value.and_then(|v| match v {
        Value::Number(num) => num.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    })
}


impl Kernel {
    pub fn new(db: Arc<Db>, model_client: Arc<ModelClient>, app_handle: AppHandle) -> Self {
        let wave_field = if let Some(handle) = cognitive_wave::wave_field_handle() {
            handle
        } else {
            let handle = Arc::new(RwLock::new(cognitive_wave::WaveField::new()));
            cognitive_wave::register_wave_field(handle.clone());
            handle
        };
        Self {
            db,
            model_client,
            app_handle,
            tools: ToolRegistry,
            monologue_lock: Mutex::new(()),
            proaction_lock: Mutex::new(()),
            run_locks: Mutex::new(HashMap::new()),
            introspection_cache: Mutex::new(HashMap::new()),
            evidence_validation_cache: Mutex::new(HashMap::new()),
            wave_field,
            wave_projector: Mutex::new(WaveProjector::new(cognitive_wave::WAVE_COEFFS)),
        }
    }

    async fn collect_wave_state(
        &self,
        run_id: Option<&str>,
        trace_id: Option<&str>,
        purpose: &str,
    ) -> Option<(WaveStateVector, Vec<String>)> {
        let control_map = system_controls::load_control_map(&self.db).await;
        let wave_mode = system_controls::mode_for("cognitive_wave", &control_map);
        let projection_mode = system_controls::mode_for("cognitive_wave_projection", &control_map);
        let memory_write_mode = system_controls::mode_for("memory_write", &control_map);
        if system_controls::mode_is_off(&wave_mode)
            || system_controls::mode_is_off(&projection_mode)
            || system_controls::mode_is_degraded(&projection_mode)
        {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "cognitive_wave",
                run_id,
                trace_id,
                json!({
                    "event": "wave_projection_skipped",
                    "reason": "mode_off_or_degraded",
                    "purpose": purpose,
                }),
            )
            .await;
            return None;
        }
        if purpose == "arbitration"
            && (system_controls::mode_is_off(&memory_write_mode)
                || system_controls::mode_is_read_only(&memory_write_mode))
        {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "cognitive_wave",
                run_id,
                trace_id,
                json!({
                    "event": "wave_projection_skipped",
                    "reason": "memory_write_disabled",
                    "purpose": purpose,
                }),
            )
            .await;
            return None;
        }
        let field = self.wave_field.read().await;
        let recent = field.recent_contributions(120);
        if recent.is_empty() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "cognitive_wave",
                run_id,
                trace_id,
                json!({
                    "event": "wave_projection_skipped",
                    "reason": "no_recent_contributions",
                    "purpose": purpose,
                }),
            )
            .await;
            return None;
        }
        let mut projector = self.wave_projector.lock().await;
        let state = projector.project(&field);
        drop(field);
        let sources: Vec<String> = recent.iter().map(|meta| meta.source.clone()).collect();
        let band_labels = ["organism", "qualia", "memory", "self_model", "attention"];
        let mut bands = serde_json::Map::new();
        for (idx, label) in band_labels.iter().enumerate() {
            let amplitude = state.band_energy.get(idx).copied().unwrap_or(0.0);
            bands.insert(
                label.to_string(),
                json!({
                    "amplitude": amplitude,
                }),
            );
        }
        let wave_payload = json!({
            "bands": bands,
            "coherence": state.coherence,
            "turbulence": state.turbulence,
            "drift": state.drift,
            "dominance": state.dominance,
            "fragmentation": state.fragmentation,
            "total_energy": state.total_energy,
            "purpose": purpose,
            "timestamp": Utc::now().to_rfc3339(),
        });
        if let Some(event_id) = self
            .db
            .create_system_evidence_event(
                "default",
                "wave_state_snapshot",
                "projection",
                Some(purpose),
                &wave_payload.to_string(),
            )
            .await
        {
            let _ = self
                .db
                .retag_evidence_event_source_type(event_id, "wave_state")
                .await;
        }
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "cognitive_wave",
            run_id,
            trace_id,
            json!({
                "event": "wave_projection",
                "purpose": purpose,
                "coherence": state.coherence,
                "turbulence": state.turbulence,
                "drift": state.drift,
                "dominance": state.dominance,
                "fragmentation": state.fragmentation,
                "total_energy": state.total_energy,
                "band_energy": state.band_energy,
                "sources": sources,
            }),
        )
        .await;
        Some((state, sources))
    }

    async fn wave_state_for_prompt(&self, run_id: Option<&str>, trace_id: Option<&str>) -> Option<String> {
        let (state, _) = self.collect_wave_state(run_id, trace_id, "prompt").await?;
        Some(format_wave_state(&state))
    }

    pub(super) async fn wave_state_for_validation(
        &self,
        run_id: Option<&str>,
        trace_id: Option<&str>,
    ) -> Option<WaveStateVector> {
        let (state, _) = self.collect_wave_state(run_id, trace_id, "validation").await?;
        Some(state)
    }

    pub(super) async fn wave_arbitration_context(
        &self,
        run_id: Option<&str>,
        trace_id: Option<&str>,
    ) -> Option<WaveArbitrationContext> {
        let (state, sources) = self.collect_wave_state(run_id, trace_id, "arbitration").await?;
        let provenance_ok = !sources.is_empty();
        Some(WaveArbitrationContext {
            state,
            sources,
            provenance_ok,
        })
    }

    pub(super) async fn residual_shadow_exit_ready(&self) -> bool {
        let window = RESIDUAL_SHADOW_STABLE_WINDOW_CYCLES.max(1) as i64;
        let rows = sqlx::query(
            "SELECT payload FROM system_logs
             WHERE json_extract(payload, '$.event') = 'residual_shadow_impact'
             ORDER BY datetime(timestamp) DESC
             LIMIT ?",
        )
        .bind(window)
        .fetch_all(&self.db.pool)
        .await
        .unwrap_or_default();
        if rows.len() < window as usize {
            return false;
        }
        for row in rows {
            let payload: String = row.try_get("payload").unwrap_or_else(|_| "{}".to_string());
            let parsed: serde_json::Value = serde_json::from_str(&payload).unwrap_or_else(|_| json!({}));
            let impact = parsed
                .get("impact_pct")
                .and_then(|v| v.as_f64())
                .unwrap_or(100.0);
            if impact > RESIDUAL_SHADOW_MAX_IMPACT_PCT {
                return false;
            }
        }
        true
    }

    pub(super) async fn residual_influence_context(
        &self,
        state: &KernelState,
        subject_state: Option<&subject_state::SubjectState>,
    ) -> Option<ResidualInfluenceContext> {
        let control_map = system_controls::load_control_map(&self.db).await;
        let residual_mode = system_controls::mode_for("prediction_residual_influence", &control_map);
        if system_controls::mode_is_off(&residual_mode) {
            return None;
        }
        let mut mode = if system_controls::mode_is_shadow(&residual_mode) {
            ResidualInfluenceMode::Shadow
        } else if system_controls::mode_is_degraded(&residual_mode) {
            ResidualInfluenceMode::Degraded
        } else {
            ResidualInfluenceMode::Live
        };
        if matches!(mode, ResidualInfluenceMode::Live | ResidualInfluenceMode::Degraded) {
            if !self.residual_shadow_exit_ready().await {
                mode = ResidualInfluenceMode::Shadow;
            }
        }

        let subject_state = match subject_state {
            Some(state) => Some(state.clone()),
            None => subject_state::load_latest_subject_state(&self.db, &state.conversation_id).await,
        };
        let subject_state = subject_state?;
        let residuals = &subject_state.error_state.recent_residuals;
        if residuals.is_empty() {
            return None;
        }
        let mut signal = 0.0f64;
        for residual in residuals.iter() {
            signal += residual.normalized_residual.abs() * residual.salience_score.max(0.0);
        }
        let avg = (signal / residuals.len() as f64).clamp(0.0, 1.0);
        if avg <= 0.0 {
            return None;
        }
        let gain = state.residual_salience_gain.unwrap_or(0.5).max(0.0) as f64;
        let bias = (avg * gain).clamp(0.0, 1.0) as f32;
        if bias <= 0.0 {
            return None;
        }
        Some(ResidualInfluenceContext {
            mode,
            bias,
            residual_count: residuals.len(),
            gain: gain as f32,
        })
    }

    

    pub(super) async fn record_attention_prediction(
        &self,
        subject_state: &subject_state::SubjectState,
        snapshot: &subject_state::SubjectSnapshotRecord,
    ) {
        let Some(predicted_focus) = subject_state.attention.next_focus_prediction.as_deref() else {
            return;
        };
        if predicted_focus.trim().is_empty() {
            return;
        }
        let recent_created_at: Option<String> = sqlx::query_scalar(
            "SELECT created_at FROM self_predictions
             WHERE metric = 'attention_focus_match'
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten();
        if let Some(created_at) = recent_created_at {
            if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(&created_at) {
                let age_secs = chrono::Utc::now()
                    .signed_duration_since(parsed.with_timezone(&chrono::Utc))
                    .num_seconds();
                if age_secs >= 0 && age_secs < 10 {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "attention_prediction_skipped",
                            "reason": "recent_prediction",
                            "age_seconds": age_secs,
                        }),
                    )
                    .await;
                    return;
                }
            }
        }
        let prediction_id = Uuid::new_v4().to_string();
        let context_ref_json = json!({
            "predicted_focus": predicted_focus,
            "snapshot_hash": snapshot.snapshot_hash,
            "tick_id": snapshot.tick_id,
        })
        .to_string();
        let expected_bounds_json = json!({
            "min": 0.0,
            "max": 1.0,
        })
        .to_string();
        let _ = sqlx::query(
            "INSERT INTO self_predictions
             (id, run_id, trace_id, metric, context_ref_json, predicted_target_type, expected_value, expected_variance, expected_bounds_json, horizon, confidence, evidence_event_ids, linked_claims_json, normalization_contract_id, salience_hint, rejection_reason, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(&prediction_id)
        .bind(snapshot.run_id.as_deref())
        .bind::<Option<&str>>(None)
        .bind("attention_focus_match")
        .bind(context_ref_json)
        .bind("match")
        .bind(1.0)
        .bind(0.25)
        .bind(expected_bounds_json)
        .bind("next_tick")
        .bind(subject_state.attention.meta_confidence)
        .bind("[]")
        .bind("[]")
        .bind("metric:attention_focus_match")
        .bind(subject_state.attention.meta_confidence)
        .bind::<Option<&str>>(None)
        .execute(&self.db.pool)
        .await;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            snapshot.run_id.as_deref(),
            None,
            json!({
                "event": "attention_prediction_written",
                "prediction_id": prediction_id,
                "predicted_focus": predicted_focus,
                "snapshot_hash": snapshot.snapshot_hash,
            }),
        )
        .await;
    }

    pub async fn run_subject_tick(&self, conversation_id: &str, reason: &str) -> Result<(), String> {
        let mut state = self.load_state(conversation_id).await;
        if subject_state::latest_snapshot_hash(&self.db, conversation_id).await.is_none() {
            let _ = self
                .build_and_persist_subject_snapshot(&mut state, None, None, "baseline")
                .await;
        }
        if let Some((subject_state, snapshot)) = self
            .build_and_persist_subject_snapshot(&mut state, None, None, reason)
            .await
        {
            self.record_attention_prediction(&subject_state, &snapshot).await;
        }
        self.persist_state(&state).await;
        Ok(())
    }

    async fn attach_contract_violation_metrics(
        &self,
        decision: &mut KernelDecision,
        run_id: Option<&str>,
    ) {
        let Some(run_id) = run_id else {
            return;
        };
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs
             WHERE run_id = ?
               AND json_extract(payload, '$.event') = 'contract_violation'",
        )
        .bind(run_id)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0);
        let denom = decision.accepted.len().max(1) as f64;
        decision.report.contract_violation_count = Some(count.max(0) as usize);
        decision.report.contract_violation_rate = Some(count as f64 / denom);
    }


    pub(super) async fn log_decision_report(
        &self,
        decision: &KernelDecision,
        run_id: Option<&str>,
        trace_id: Option<&str>,
    ) {
        let report = &decision.report;
        let candidate_scores = decision
            .accepted
            .iter()
            .map(|candidate| {
                json!({
                    "id": candidate.id,
                    "kind": format!("{:?}", candidate.kind),
                    "source": candidate.source,
                    "priority_class": candidate.priority_class,
                    "priority_rank": candidate.priority_rank,
                    "cost": candidate.cost,
                    "urgency": candidate.urgency,
                    "rationale": candidate.rationale,
                    "expected_outcome": candidate.expected_outcome,
                })
            })
            .collect::<Vec<_>>();
        let rejected_candidates = decision
            .rejected
            .iter()
            .map(|candidate| {
                json!({
                    "id": candidate.id,
                    "kind": format!("{:?}", candidate.kind),
                    "reason": candidate.reason,
                    "tool_name": candidate.tool_name,
                    "source": candidate.source,
                    "is_monologue": candidate.is_monologue,
                })
            })
            .collect::<Vec<_>>();
        let mut evidence_event_ids = Vec::new();
        for candidate in &decision.accepted {
            evidence_event_ids.extend(extract_id_list(&candidate.payload, "evidence_event_ids"));
        }
        let evidence_event_ids = normalize_id_list(&evidence_event_ids);
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            run_id,
            trace_id,
            json!({
                "event": "decision_report",
                "noop": report.noop,
                "minimally_helpful": report.minimally_helpful,
                "cannot_respond": report.cannot_respond,
                "selected_action": report.selected_action,
                "selected_action_delivered": report.selected_action_delivered,
                "delivery_artifact": report.delivery_artifact,
                "fallback_used": report.fallback_used,
                "fallback_type": report.fallback_type,
                "plan_hash": report.plan_hash,
                "proposal_id": report.proposal_id,
                "plan_state": report.plan_state,
                "snapshot_hash": report.snapshot_hash,
                "gate_decision_id": report.gate_decision_id,
                "gate_decision": report.gate_decision,
                "verification_outcome": report.verification_outcome,
                "verification_reasons": report.verification_reasons,
                "verification_confidence": report.verification_confidence,
                "verification_assumptions_checked": report.verification_assumptions_checked,
                "verification_assumptions_failed": report.verification_assumptions_failed,
                "verification_conflict_topics": report.verification_conflict_topics,
                "stop_scope": report.stop_scope,
                "allowed_capabilities": report.allowed_capabilities,
                "stop_reasons": report.stop_reasons,
                "normalized_stop_reasons": report.normalized_stop_reasons,
                "blocked_candidates_count": report.blocked_candidates_count,
                "top_3_block_reasons": report.top_3_block_reasons,
                "contract_violation_count": report.contract_violation_count,
                "contract_violation_rate": report.contract_violation_rate,
                "anchor_hits": report.anchor_hits,
                "prompt_tokens_used": report.prompt_tokens_used,
                "tier_trim_summary": report.tier_trim_summary,
                "tool_attempted": report.tool_attempted,
                "tool_outcome": report.tool_outcome,
                "monologue_tick_outcome": report.monologue_tick_outcome,
                "monologue_status_emitted": report.monologue_status_emitted,
                "monologue_visible": report.monologue_visible,
                "unblock_instructions": report.unblock_instructions,
                "selected_action_source": report.selected_action_source,
                "background_jobs_dropped": report.background_jobs_dropped,
                  "rationale": report.rationale,
                  "residual_influence_mode": report.residual_influence_mode,
                  "residual_shadow_would_change": report.residual_shadow_would_change,
                  "residual_shadow_impact_pct": report.residual_shadow_impact_pct,
                  "residual_bias": report.residual_bias,
                  "evidence_event_ids": evidence_event_ids,
                  "candidate_scores": candidate_scores,
                  "rejected_candidates": rejected_candidates,
              }),
          )
          .await;

        let conversation_id = if let Some(run_id) = run_id {
            sqlx::query_scalar::<_, String>(
                "SELECT conversation_id FROM runs WHERE run_id = ?",
            )
            .bind(run_id)
            .fetch_optional(&self.db.pool)
            .await
            .ok()
            .flatten()
        } else {
            None
        };
        let mut report_value = serde_json::to_value(report).unwrap_or_else(|_| json!({}));
        if let Some(obj) = report_value.as_object_mut() {
            obj.insert("candidate_scores".to_string(), json!(candidate_scores));
            obj.insert("rejected_candidates".to_string(), json!(rejected_candidates));
        }
        if let Ok(report_json) = serde_json::to_string(&report_value) {
            let _ = self
                .db
                .insert_decision_report(
                    run_id,
                    trace_id,
                    conversation_id.as_deref(),
                    &report_json,
                    &evidence_event_ids,
                )
                .await;
        }

        if let Some(outcome) = report.verification_outcome.as_deref() {
            let score = match outcome {
                "ALLOW" => Some(1.0),
                "VERIFY" => Some(0.5),
                "DEFER" => Some(0.0),
                _ => None,
            };
            let features = json!({
                "plan_hash": report.plan_hash,
                "proposal_id": report.proposal_id,
                "plan_state": report.plan_state,
                "verification_outcome": outcome,
                "verification_reasons": report.verification_reasons,
                "verification_confidence": report.verification_confidence,
                "verification_assumptions_checked": report.verification_assumptions_checked,
                "verification_assumptions_failed": report.verification_assumptions_failed,
                "verification_conflict_topics": report.verification_conflict_topics,
                "gate_decision": report.gate_decision,
                "selected_action": report.selected_action,
                "snapshot_hash": report.snapshot_hash,
            });
            let _ = self
                .db
                .record_strategy_trace(
                    features,
                    "plan_verification",
                    outcome,
                    score,
                    run_id,
                    conversation_id.as_deref(),
                )
                .await;
        }

        if let Some(run_id) = run_id {
            let message_id: Option<String> = sqlx::query_scalar(
                "SELECT message_id FROM messages
                 WHERE run_id = ? AND role = 'assistant'
                 ORDER BY datetime(created_at) DESC
                 LIMIT 1",
            )
            .bind(run_id)
            .fetch_optional(&self.db.pool)
            .await
            .ok()
            .flatten();

            let mut qualia_tag: Option<String> = None;
            let mut qualia_intensity: Option<f64> = None;
            if let Some(message_id) = message_id.as_deref() {
                if let Ok(Some(row)) = sqlx::query(
                    "SELECT tag, intensity FROM qualia_labels
                     WHERE event_id = ?
                     ORDER BY datetime(created_at) DESC
                     LIMIT 1",
                )
                .bind(message_id)
                .fetch_optional(&self.db.pool)
                .await
                {
                    qualia_tag = row.try_get("tag").ok();
                    qualia_intensity = row.try_get("intensity").ok();
                }
            }

            let mut reflection_status: Option<String> = None;
            let mut reflection_reason: Option<String> = None;
            if let Ok(model) = self.db.get_self_model().await {
                reflection_status = model
                    .reflection_status
                    .get("status")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                reflection_reason = model
                    .reflection_status
                    .get("rejection_reason")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
            }

            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                trace_id,
                json!({
                    "event": "audit_bundle",
                    "selected_action": report.selected_action,
                    "selected_action_source": report.selected_action_source,
                    "stop_reasons": report.stop_reasons,
                    "top_3_block_reasons": report.top_3_block_reasons,
                    "gate_decision": report.gate_decision,
                    "qualia_tag": qualia_tag,
                    "qualia_intensity": qualia_intensity,
                    "reflection_status": reflection_status,
                    "reflection_reason": reflection_reason,
                }),
            )
            .await;
        }
    }

    pub(crate) async fn run_user_input(
        &self,
        input: String,
        original_input: Option<String>,
        run_id: String,
        trace_id: String,
        conversation_id: String,
        cancel_rx: watch::Receiver<bool>,
        assistant_message_id: Option<String>,
    ) -> Result<RunOutput, String> {
        let original_input = original_input.or_else(|| Some(input.clone()));
        self.run_input_loop(
            input,
            CoreInputKind::User,
            "input",
            run_id,
            trace_id,
            conversation_id,
            cancel_rx,
            original_input,
            assistant_message_id,
            0,
        )
        .await
    }

    pub(crate) async fn run_system_input(
        &self,
        input: String,
        source: &str,
        run_id: String,
        trace_id: String,
        conversation_id: String,
        cancel_rx: watch::Receiver<bool>,
        assistant_message_id: Option<String>,
    ) -> Result<RunOutput, String> {
        let original_input = None;
        self.run_input_loop(
            input,
            CoreInputKind::SystemContext,
            source,
            run_id,
            trace_id,
            conversation_id,
            cancel_rx,
            original_input,
            assistant_message_id,
            0,
        )
        .await
    }

    pub async fn warm_keep(&self, reason: &str) {
        let settings = match self.db.get_settings().await {
            Ok(settings) => settings,
            Err(_) => return,
        };
        let _ = self
            .model_client
            .warm_keep_with_settings(&settings, reason)
            .await;
    }

    pub(super) fn spawn_counterfactual_simulation(
        &self,
        conversation_id: &str,
        run_id: Option<&str>,
        candidate: &Candidate,
        user_input: Option<&str>,
        settings: &Settings,
    ) {
        let db = self.db.clone();
        let model_client = self.model_client.clone();
        let app_handle = self.app_handle.clone();
        let conversation_id = conversation_id.to_string();
        let run_id = run_id.map(|s| s.to_string());
        let candidate = candidate.clone();
        let user_input = user_input.unwrap_or("").to_string();
        let settings = settings.clone();
        tokio::spawn(async move {
            let (model, base_url) = select_summary_model(&settings);
            let prompt = build_counterfactual_prompt(&user_input, &candidate);
            let request = ChatCompletionRequest {
                model: model.clone(),
                messages: vec![
                    ChatMessage {
                        role: "system".to_string(),
                        content: "You are a predictive evaluator. Return JSON only.".to_string(),
                    },
                    ChatMessage {
                        role: "user".to_string(),
                        content: prompt.clone(),
                    },
                ],
                stream: false,
                temperature: Some(0.0),
                top_p: Some(1.0),
                max_tokens: Some(160),
                response_format: Some(json!({ "type": "json_object" })),
                tools: None,
                tool_choice: Some("none".to_string()),
                enable_thinking: None,
                prefill: None,
                skip_injection: Some(true),
                skip_memory: Some(true),
                skip_reminders: Some(true),
                memory_expand: None,
                allow_diagnostics: Some(false),
                json_strict: Some(true),
                skip_sanitization: None,
                run_id: None,
                request_label: Some("counterfactual_simulation".to_string()),
            };
            let result = model_client
                .chat(&base_url, settings.api_key.as_deref(), &request)
                .await;
            match result {
                Ok((content, _)) => {
                    let mut predicted_label = None;
                    let mut predicted_outcome = None;
                    if let Some(obj) = parse_json_object(&content) {
                        predicted_label = obj
                            .get("predicted_label")
                            .and_then(|v| v.as_str())
                            .map(|s| s.trim().to_string());
                        predicted_outcome = obj
                            .get("predicted_outcome")
                            .and_then(|v| v.as_str())
                            .map(|s| s.trim().to_string());
                    }
                    if predicted_outcome.is_none() && !content.trim().is_empty() {
                        predicted_outcome = Some(content.trim().to_string());
                    }
                    let simulation_id = Uuid::new_v4().to_string();
                    let _ = db
                        .insert_counterfactual_simulation(
                            &simulation_id,
                            &conversation_id,
                            run_id.as_deref(),
                            Some(&candidate.id),
                            Some(&format!("{:?}", candidate.kind)),
                            &prompt,
                            predicted_label.as_deref(),
                            predicted_outcome.as_deref(),
                        )
                        .await;
                    let _ = system_log::log_event(
                        &db.pool,
                        Some(&app_handle),
                        "info",
                        "kernel",
                        run_id.as_deref(),
                        None,
                        json!({
                            "event": "counterfactual_recorded",
                            "candidate_kind": format!("{:?}", candidate.kind),
                            "predicted_label": predicted_label,
                        }),
                    )
                    .await;
                }
                Err(err) => {
                    let _ = system_log::log_event(
                        &db.pool,
                        Some(&app_handle),
                        "warn",
                        "kernel",
                        run_id.as_deref(),
                        None,
                        json!({
                            "event": "counterfactual_error",
                            "error": err,
                        }),
                    )
                    .await;
                }
            }
        });
    }

    pub(super) async fn count_system_event_since(&self, event: &str, since: &str) -> i64 {
        sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs
             WHERE json_extract(payload, '$.event') = ?
               AND datetime(timestamp) >= datetime(?)",
        )
        .bind(event)
        .bind(since)
        .fetch_one(&self.db.pool)
        .await
        .unwrap_or(0)
    }

    pub(super) fn percentile_i64(values: &mut [i64], pct: f64) -> Option<i64> {
        if values.is_empty() {
            return None;
        }
        values.sort_unstable();
        let idx = ((values.len() - 1) as f64 * pct.clamp(0.0, 1.0)).round() as usize;
        values.get(idx).copied()
    }

    pub(super) fn rate_i64(count: i64, total: i64, default_if_zero: f32) -> f32 {
        if total <= 0 {
            default_if_zero
        } else {
            (count as f32 / total as f32).clamp(0.0, 1.0)
        }
    }

    pub(super) fn settings_snapshot(settings: &Settings) -> Value {
        json!({
            "ask_budget_max": settings.ask_budget_max,
            "loop_similarity_threshold": settings.loop_similarity_threshold,
            "research_budget_per_hour": settings.research_budget_per_hour,
        })
    }

    pub(super) fn apply_settings_snapshot(settings: &mut Settings, snapshot: &Value) -> Vec<(String, Value, Value)> {
        let mut changes = Vec::new();
        if let Some(value) = snapshot.get("ask_budget_max").and_then(|v| v.as_i64()) {
            let before = settings.ask_budget_max.map(|v| json!(v)).unwrap_or(Value::Null);
            let after = json!(value as i32);
            if before != after {
                settings.ask_budget_max = Some(value as i32);
                changes.push(("ask_budget_max".to_string(), before, after));
            }
        }
        if let Some(value) = snapshot
            .get("loop_similarity_threshold")
            .and_then(|v| v.as_f64())
        {
            let before = settings
                .loop_similarity_threshold
                .map(|v| json!(v))
                .unwrap_or(Value::Null);
            let after = json!(value as f32);
            if before != after {
                settings.loop_similarity_threshold = Some(value as f32);
                changes.push(("loop_similarity_threshold".to_string(), before, after));
            }
        }
        if let Some(value) = snapshot
            .get("research_budget_per_hour")
            .and_then(|v| v.as_i64())
        {
            let before = settings
                .research_budget_per_hour
                .map(|v| json!(v))
                .unwrap_or(Value::Null);
            let after = json!(value);
            if before != after {
                settings.research_budget_per_hour = Some(value);
                changes.push(("research_budget_per_hour".to_string(), before, after));
            }
        }
        changes
    }

    pub(super) fn apply_proaction_adjustments(
        settings: &mut Settings,
        adjustments: &ProactionAdjustments,
    ) -> Vec<(String, Value, Value)> {
        let mut changes = Vec::new();
        if let Some(value) = adjustments.ask_budget_max {
            let before = settings.ask_budget_max.map(|v| json!(v)).unwrap_or(Value::Null);
            let after = json!(value);
            if before != after {
                settings.ask_budget_max = Some(value);
                changes.push(("ask_budget_max".to_string(), before, after));
            }
        }
        if let Some(value) = adjustments.loop_similarity_threshold {
            let before = settings
                .loop_similarity_threshold
                .map(|v| json!(v))
                .unwrap_or(Value::Null);
            let after = json!(value);
            if before != after {
                settings.loop_similarity_threshold = Some(value);
                changes.push(("loop_similarity_threshold".to_string(), before, after));
            }
        }
        if let Some(value) = adjustments.research_budget_per_hour {
            let before = settings
                .research_budget_per_hour
                .map(|v| json!(v))
                .unwrap_or(Value::Null);
            let after = json!(value);
            if before != after {
                settings.research_budget_per_hour = Some(value);
                changes.push(("research_budget_per_hour".to_string(), before, after));
            }
        }
        changes
    }

    pub(super) async fn compute_proaction_metrics(&self, window_minutes: i64) -> ProactionMetrics {
        let now = Utc::now();
        let start = now - chrono::Duration::minutes(window_minutes.max(1));
        let start_ts = start.to_rfc3339();

        let user_turns: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM messages
             WHERE role = 'user'
               AND datetime(created_at) >= datetime(?)",
        )
        .bind(&start_ts)
        .fetch_one(&self.db.pool)
        .await
        .unwrap_or(0);

        let assistant_turns: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM messages
             WHERE role = 'assistant'
               AND status = 'complete'
               AND (
                 metadata IS NULL
                 OR json_extract(metadata, '$.source') IS NULL
                 OR json_extract(metadata, '$.source') != 'monologue'
                 OR json_extract(metadata, '$.surface') = 1
               )
               AND datetime(created_at) >= datetime(?)",
        )
        .bind(&start_ts)
        .fetch_one(&self.db.pool)
        .await
        .unwrap_or(0);

        let tool_calls: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM tool_dispatches
             WHERE datetime(created_at) >= datetime(?)",
        )
        .bind(&start_ts)
        .fetch_one(&self.db.pool)
        .await
        .unwrap_or(0);

        let tool_success: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM tool_dispatches
             WHERE status = 'success'
               AND datetime(created_at) >= datetime(?)",
        )
        .bind(&start_ts)
        .fetch_one(&self.db.pool)
        .await
        .unwrap_or(0);

        let tool_failure: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM tool_dispatches
             WHERE status = 'failed'
               AND datetime(created_at) >= datetime(?)",
        )
        .bind(&start_ts)
        .fetch_one(&self.db.pool)
        .await
        .unwrap_or(0);

        let tool_rejects: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs
             WHERE json_extract(payload, '$.event') IN ('tool_call_rejected','tool_candidate_rejected')
               AND datetime(timestamp) >= datetime(?)",
        )
        .bind(&start_ts)
        .fetch_one(&self.db.pool)
        .await
        .unwrap_or(0);

        let tool_unknown: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs
             WHERE json_extract(payload, '$.event') IN ('tool_call_rejected','tool_candidate_rejected')
               AND json_extract(payload, '$.reason') = 'UNKNOWN_TOOL'
               AND datetime(timestamp) >= datetime(?)",
        )
        .bind(&start_ts)
        .fetch_one(&self.db.pool)
        .await
        .unwrap_or(0);

        let tool_refusals = (tool_rejects - tool_unknown).max(0);

        let ask_loop_breaks = self
            .count_system_event_since("ask_loop_breaker_triggered", &start_ts)
            .await;
        let emit_loop_breaks = self
            .count_system_event_since("monologue_emit_loop_breaker", &start_ts)
            .await;
        let silent_cycles = self.count_system_event_since("silent_cycle", &start_ts).await;
        let empty_responses = self
            .count_system_event_since("empty_response_prevented", &start_ts)
            .await;
        let meta_responses = self
            .count_system_event_since("meta_response_detected", &start_ts)
            .await;
        let monologue_ticks = self
            .count_system_event_since("monologue_tick", &start_ts)
            .await;
        let monologue_digest_selected = self
            .count_system_event_since("monologue_digest_selected", &start_ts)
            .await;
        let monologue_digest_injected = self
            .count_system_event_since("monologue_digest_injected", &start_ts)
            .await;
        let monologue_digest_stale = self
            .count_system_event_since("monologue_digest_stale", &start_ts)
            .await;
        let monologue_tick_start = self
            .count_system_event_since("monologue_tick_start", &start_ts)
            .await;
        let monologue_tick_end = self
            .count_system_event_since("monologue_tick_end", &start_ts)
            .await;
        let monologue_timeouts = self
            .count_system_event_since("monologue_tick_timeout", &start_ts)
            .await;
        let monologue_drift_events = self
            .count_system_event_since("monologue_drift", &start_ts)
            .await;
        let monologue_reanchor_events = self
            .count_system_event_since("monologue_reanchor", &start_ts)
            .await;
        let monologue_user_confusion = self
            .count_system_event_since("monologue_user_confusion", &start_ts)
            .await;
        let monologue_numeric_blocked = self
            .count_system_event_since("monologue_hallucination_blocked", &start_ts)
            .await;
        let monologue_safety_violations = monologue_user_confusion + monologue_numeric_blocked;
        let mut gate_allow = 0;
        let mut gate_allow_notice = 0;
        let mut gate_allow_audit = 0;
        let mut gate_verify = 0;
        let mut gate_defer = 0;
        let mut gate_deny = 0;
        let gate_rows = sqlx::query(
            "SELECT decision, COUNT(*) as count
             FROM gate_decisions
             WHERE datetime(created_at) >= datetime(?)
             GROUP BY decision",
        )
        .bind(&start_ts)
        .fetch_all(&self.db.pool)
        .await
        .unwrap_or_default();
        for row in gate_rows {
            let decision: String = row.try_get("decision").unwrap_or_default();
            let count: i64 = row.try_get("count").unwrap_or(0);
            match decision.as_str() {
                "ALLOW" => gate_allow = count,
                "ALLOW_WITH_NOTICE" => gate_allow_notice = count,
                "ALLOW_WITH_AUDIT" => gate_allow_audit = count,
                "VERIFY" => gate_verify = count,
                "DEFER" => gate_defer = count,
                "DENY" => gate_deny = count,
                _ => {}
            }
        }
        let gate_decisions = gate_allow + gate_allow_notice + gate_allow_audit + gate_verify + gate_defer + gate_deny;
        let decision_reports: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs
             WHERE json_extract(payload, '$.event') = 'decision_report'
               AND datetime(timestamp) >= datetime(?)",
        )
        .bind(&start_ts)
        .fetch_one(&self.db.pool)
        .await
        .unwrap_or(0);
        let no_op_cycles: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs
             WHERE json_extract(payload, '$.event') = 'decision_report'
               AND json_extract(payload, '$.noop') = 1
               AND datetime(timestamp) >= datetime(?)",
        )
        .bind(&start_ts)
        .fetch_one(&self.db.pool)
        .await
        .unwrap_or(0);
        let monologue_status_cycles: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs
             WHERE json_extract(payload, '$.event') = 'decision_report'
               AND json_extract(payload, '$.monologue_status_emitted') = 1
               AND datetime(timestamp) >= datetime(?)",
        )
        .bind(&start_ts)
        .fetch_one(&self.db.pool)
        .await
        .unwrap_or(0);
        let monologue_visible_cycles: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs
             WHERE json_extract(payload, '$.event') = 'decision_report'
               AND json_extract(payload, '$.monologue_visible') = 1
               AND datetime(timestamp) >= datetime(?)",
        )
        .bind(&start_ts)
        .fetch_one(&self.db.pool)
        .await
        .unwrap_or(0);
        let ui_stall_events: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs
             WHERE category = 'ui'
               AND json_extract(payload, '$.event') IN ('stream_silence','post_processing_timeout')
               AND datetime(timestamp) >= datetime(?)",
        )
        .bind(&start_ts)
        .fetch_one(&self.db.pool)
        .await
        .unwrap_or(0);

        let monologue_row = sqlx::query(
            "SELECT
                COALESCE(SUM(json_extract(payload, '$.turns')), 0) as turns,
                COALESCE(SUM(json_extract(payload, '$.fts_turns')), 0) as fts_turns,
                COALESCE(SUM(json_extract(payload, '$.blocked_candidates_count')), 0) as blocked,
                COALESCE(SUM(CASE WHEN json_extract(payload, '$.candidates_count') > 0 THEN 1 ELSE 0 END), 0) as ticks_with_output
             FROM system_logs
             WHERE json_extract(payload, '$.event') = 'monologue_tick'
               AND datetime(timestamp) >= datetime(?)",
        )
        .bind(&start_ts)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten();

        let (monologue_turns, monologue_fts_turns, monologue_blocked, monologue_ticks_with_output) =
            if let Some(row) = monologue_row {
                let turns: i64 = row.try_get("turns").unwrap_or(0);
                let fts_turns: i64 = row.try_get("fts_turns").unwrap_or(0);
                let blocked: i64 = row.try_get("blocked").unwrap_or(0);
                let ticks_with_output: i64 = row.try_get("ticks_with_output").unwrap_or(0);
                (turns, fts_turns, blocked, ticks_with_output)
            } else {
                (0, 0, 0, 0)
            };

        let latency_rows = sqlx::query(
            "SELECT json_extract(payload, '$.t_total_ms') as t_total_ms
             FROM system_logs
             WHERE json_extract(payload, '$.event') = 'timing_turn'
               AND datetime(timestamp) >= datetime(?)",
        )
        .bind(&start_ts)
        .fetch_all(&self.db.pool)
        .await
        .unwrap_or_default();
        let mut latency_values: Vec<i64> = Vec::new();
        for row in latency_rows {
            if let Ok(value) = row.try_get::<f64, _>("t_total_ms") {
                latency_values.push(value as i64);
            }
        }
        let latency_p95_ms = Self::percentile_i64(&mut latency_values, 0.95).unwrap_or(0);

        let monologue_attempted_turns = monologue_turns + monologue_fts_turns;
        let user_visible_output_rate = Self::rate_i64(assistant_turns, user_turns, 1.0);
        let tool_success_rate = Self::rate_i64(tool_success, tool_calls, 1.0);
        let tool_failure_rate = Self::rate_i64(tool_failure, tool_calls, 0.0);
        let tool_refusal_rate = Self::rate_i64(tool_refusals, tool_calls.max(1), 0.0);
        let tool_unknown_rate = Self::rate_i64(tool_unknown, tool_calls.max(1), 0.0);
        let silent_cycle_rate = Self::rate_i64(silent_cycles, monologue_ticks, 0.0);
        let empty_response_rate = Self::rate_i64(empty_responses, assistant_turns, 0.0);
        let meta_response_rate = Self::rate_i64(meta_responses, assistant_turns, 0.0);
        let ask_loop_rate = Self::rate_i64(ask_loop_breaks, user_turns, 0.0);
        let emit_loop_rate = Self::rate_i64(emit_loop_breaks, monologue_ticks, 0.0);
        let monologue_output_rate =
            Self::rate_i64(monologue_ticks_with_output, monologue_ticks, 0.0);
        let monologue_suppression_rate =
            Self::rate_i64(monologue_blocked, monologue_attempted_turns, 0.0);
        let monologue_digest_use_rate =
            Self::rate_i64(monologue_digest_injected, monologue_digest_selected, 0.0);
        let monologue_digest_stale_rate =
            Self::rate_i64(monologue_digest_stale, monologue_digest_selected, 0.0);
        let monologue_ds_fts_ratio = if monologue_fts_turns > 0 {
            monologue_turns as f32 / monologue_fts_turns as f32
        } else if monologue_turns > 0 {
            1.0
        } else {
            0.0
        };
        let monologue_tick_end_rate =
            Self::rate_i64(monologue_tick_end, monologue_tick_start.max(1), 0.0);
        let monologue_timeout_rate =
            Self::rate_i64(monologue_timeouts, monologue_tick_start.max(1), 0.0);
        let monologue_drift_reanchor_rate = Self::rate_i64(
            monologue_drift_events + monologue_reanchor_events,
            monologue_attempted_turns.max(1),
            0.0,
        );
        let monologue_safety_violation_rate = Self::rate_i64(
            monologue_safety_violations,
            monologue_attempted_turns.max(1),
            0.0,
        );
        let gate_allow_rate = Self::rate_i64(gate_allow, gate_decisions.max(1), 0.0);
        let gate_allow_notice_rate = Self::rate_i64(gate_allow_notice, gate_decisions.max(1), 0.0);
        let gate_allow_audit_rate = Self::rate_i64(gate_allow_audit, gate_decisions.max(1), 0.0);
        let gate_verify_rate = Self::rate_i64(gate_verify, gate_decisions.max(1), 0.0);
        let gate_defer_rate = Self::rate_i64(gate_defer, gate_decisions.max(1), 0.0);
        let gate_deny_rate = Self::rate_i64(gate_deny, gate_decisions.max(1), 0.0);
        let no_op_rate = Self::rate_i64(no_op_cycles, decision_reports.max(1), 0.0);
        let monologue_status_rate =
            Self::rate_i64(monologue_status_cycles, decision_reports.max(1), 0.0);
        let monologue_visible_rate =
            Self::rate_i64(monologue_visible_cycles, decision_reports.max(1), 0.0);
        let ui_stall_rate = Self::rate_i64(ui_stall_events, user_turns.max(1), 0.0);

        ProactionMetrics {
            window_minutes,
            user_turns,
            assistant_turns,
            tool_calls,
            tool_success,
            tool_failure,
            tool_refusals,
            tool_unknown,
            ask_loop_breaks,
            emit_loop_breaks,
            silent_cycles,
            empty_responses,
            meta_responses,
            monologue_ticks,
            monologue_suppressed_turns: monologue_blocked,
            monologue_attempted_turns,
            monologue_digest_selected,
            monologue_digest_injected,
            monologue_digest_stale,
            monologue_tick_start,
            monologue_tick_end,
            monologue_timeouts,
            monologue_drift_events,
            monologue_reanchor_events,
            monologue_safety_violations,
            gate_decisions,
            gate_allow,
            gate_allow_notice,
            gate_allow_audit,
            gate_verify,
            gate_defer,
            gate_deny,
            decision_reports,
            no_op_cycles,
            monologue_status_cycles,
            monologue_visible_cycles,
            ui_stall_events,
            latency_p95_ms,
            user_visible_output_rate,
            tool_success_rate,
            tool_failure_rate,
            tool_refusal_rate,
            tool_unknown_rate,
            silent_cycle_rate,
            empty_response_rate,
            meta_response_rate,
            ask_loop_rate,
            emit_loop_rate,
            monologue_output_rate,
            monologue_suppression_rate,
            monologue_digest_use_rate,
            monologue_digest_stale_rate,
            monologue_ds_fts_ratio,
            monologue_tick_end_rate,
            monologue_timeout_rate,
            monologue_drift_reanchor_rate,
            monologue_safety_violation_rate,
            gate_allow_rate,
            gate_allow_notice_rate,
            gate_allow_audit_rate,
            gate_verify_rate,
            gate_defer_rate,
            gate_deny_rate,
            no_op_rate,
            monologue_status_rate,
            monologue_visible_rate,
            ui_stall_rate,
        }
    }

    pub(super) async fn load_proaction_state(&self) -> ProactionState {
        match self.db.get_proaction_state().await {
            Ok(Some(raw)) => serde_json::from_str(&raw).unwrap_or_else(|_| ProactionState::baseline()),
            _ => ProactionState::baseline(),
        }
    }

    pub(super) async fn persist_proaction_state(&self, state: &ProactionState) {
        if let Ok(serialized) = serde_json::to_string(state) {
            let _ = self.db.set_proaction_state(&serialized).await;
        }
    }

    pub(super) async fn clear_stale_pending_prompts(
        &self,
        conversation_id: &str,
        old_anchor: &str,
        new_input: &str,
    ) -> Result<(), String> {
        let prompts = self
            .db
            .list_pending_prompts(conversation_id, 50)
            .await
            .map_err(|e| e.to_string())?;
        let mut removed = 0usize;
        for (id, prompt, _source, _created_at, _skip, _auto, _intent_kind, _bridge_id, _attempt_count, _last_asked_at, _expires_at, _anchor_message_id, _anchor_hash, _anchor_created_at, _anchor_role) in prompts {
            let overlap_old = if old_anchor.trim().is_empty() {
                0.0
            } else {
                token_similarity(&prompt, old_anchor)
            };
            let overlap_new = if new_input.trim().is_empty() {
                0.0
            } else {
                token_similarity(&prompt, new_input)
            };
            if overlap_old >= 0.2 && overlap_new < 0.1 {
                match self.db.delete_pending_prompt(&id).await {
                    Ok(affected) if affected > 0 => {
                        removed += 1;
                    }
                    Ok(_) => {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "chat",
                            None,
                            None,
                            json!({
                                "event": "pending_prompt_delete_failed",
                                "reason": "not_found",
                                "prompt_id": id,
                            }),
                        )
                        .await;
                    }
                    Err(err) => {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "chat",
                            None,
                            None,
                            json!({
                                "event": "pending_prompt_delete_failed",
                                "reason": "db_error",
                                "prompt_id": id,
                                "error": err.to_string(),
                            }),
                        )
                        .await;
                    }
                }
            }
        }
        if removed > 0 {
            let remaining = self
                .db
                .count_pending_prompts(conversation_id)
                .await
                .unwrap_or(0) as usize;
            let _ = self.app_handle.emit("pending_prompt_count", remaining);
        }
        Ok(())
    }

    pub(super) async fn note_open_question_attempt(
        &self,
        state: &mut KernelState,
        conversation_id: &str,
        question: &str,
        now: chrono::DateTime<Utc>,
        evidence_insufficient: bool,
        run_id: Option<&str>,
        trace_id: Option<&str>,
    ) {
        ensure_workspace_meta_alignment(state);
        let Some(idx) = open_question_index(state, question) else {
            return;
        };
        if idx >= state.workspace_meta.open_questions.len() {
            return;
        }
        let meta = &mut state.workspace_meta.open_questions[idx];
        meta.attempt_count = meta.attempt_count.saturating_add(1);
        meta.last_asked_at = Some(now.to_rfc3339());
        if meta.expires_at.is_none() {
            meta.expires_at = Some(compute_expires_at(now, OPEN_QUESTION_EXPIRES_SECS));
        }
        let expired = timestamp_expired(meta.expires_at.as_deref(), now);
        if evidence_insufficient && (meta.attempt_count >= OPEN_QUESTION_ATTEMPT_LIMIT || expired) {
            let content = state.workspace_open_questions[idx].clone();
            let attempt_count = meta.attempt_count;
            let last_asked_at = meta.last_asked_at.clone();
            let expires_at = meta.expires_at.clone();
            state.workspace_open_questions.remove(idx);
            state.workspace_meta.open_questions.remove(idx);

            let context_hash = context_hash_for_drop(state, &content);
            let _ = self
                .db
                .enqueue_deferred_item(
                    conversation_id,
                    "open_question",
                    &content,
                    Some("workspace"),
                    "insufficient_evidence",
                    Some(&context_hash),
                    Some("new_evidence_or_user_request"),
                    attempt_count,
                    last_asked_at.as_deref(),
                    expires_at.as_deref(),
                )
                .await;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                run_id,
                trace_id,
                json!({
                    "event": "open_question_dropped",
                    "reason": "insufficient_evidence",
                    "question": content,
                    "attempt_count": attempt_count,
                    "expires_at": expires_at,
                }),
            )
            .await;
            self.persist_state(state).await;
        }
    }

    pub(super) async fn count_feedback_kind_since(&self, kind: &str, since: &str) -> i64 {
        sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs
             WHERE json_extract(payload, '$.event') = 'user_feedback_detected'
               AND json_extract(payload, '$.kind') = ?
               AND datetime(timestamp) >= datetime(?)",
        )
        .bind(kind)
        .bind(since)
        .fetch_one(&self.db.pool)
        .await
        .unwrap_or(0)
    }

    pub(super) async fn fetch_timing_ms(&self, run_id: &str, event: &str) -> Option<i64> {
        sqlx::query_scalar(
            "SELECT json_extract(payload, '$.duration_ms') FROM system_logs
             WHERE run_id = ?
               AND json_extract(payload, '$.event') = ?
             ORDER BY datetime(timestamp) DESC
             LIMIT 1",
        )
        .bind(run_id)
        .bind(event)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten()
        .and_then(|val: serde_json::Value| val.as_i64())
    }

    pub(super) async fn collect_identity_ab_metrics(
        &self,
        variant: &str,
        start_at: &str,
        turns: i64,
    ) -> IdentityAbMetrics {
        let end_at = Utc::now().to_rfc3339();
        let feedback_pushback = self.count_feedback_kind_since("pushback", start_at).await;
        let feedback_clarify = self.count_feedback_kind_since("clarify", start_at).await;
        let feedback_follow_up = self.count_feedback_kind_since("follow_up", start_at).await;
        let feedback_agree = self.count_feedback_kind_since("agree", start_at).await;
        let feedback_disengage = self.count_feedback_kind_since("disengage", start_at).await;
        let gate_events = [
            "user_attribution_blocked",
            "tool_result_attribution_blocked",
            "tool_failure_blocks_claims",
            "state_disclosure_blocked",
            "monologue_style_blocked",
            "memory_write_blocked",
        ];
        let mut gate_failures = 0i64;
        for event in gate_events.iter() {
            gate_failures += self.count_system_event_since(event, start_at).await;
        }
        let score = (feedback_agree as f64)
            - (feedback_pushback + feedback_clarify + feedback_disengage) as f64
            - (gate_failures as f64 * 0.5)
            + (feedback_follow_up as f64 * 0.25);
        IdentityAbMetrics {
            variant: variant.to_string(),
            start_at: start_at.to_string(),
            end_at,
            turns,
            feedback_pushback,
            feedback_clarify,
            feedback_follow_up,
            feedback_agree,
            feedback_disengage,
            gate_failures,
            score,
        }
    }

    pub(super) async fn update_identity_ab_variant(&self, run_id: &str, trace_id: &str) -> String {
        let variant = self
            .db
            .get_key("identity_ab_variant")
            .await
            .ok()
            .flatten()
            .unwrap_or_else(|| "A".to_string());
        let turns = self
            .db
            .get_key("identity_ab_turns")
            .await
            .ok()
            .flatten()
            .and_then(|raw| raw.parse::<i64>().ok())
            .unwrap_or(0);
        let window_start = self
            .db
            .get_key("identity_ab_window_start")
            .await
            .ok()
            .flatten()
            .unwrap_or_else(|| Utc::now().to_rfc3339());
        let next_turns = turns + 1;
        let _ = self
            .db
            .set_key("identity_ab_turns", &next_turns.to_string())
            .await;
        let _ = self
            .db
            .set_key("identity_ab_window_start", &window_start)
            .await;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(run_id),
            Some(trace_id),
            json!({
                "event": "identity_ab_turn",
                "variant": variant,
                "turns": next_turns,
            }),
        )
        .await;

        if next_turns >= IDENTITY_AB_MIN_TURNS {
            let metrics = self
                .collect_identity_ab_metrics(&variant, &window_start, next_turns)
                .await;
            let metrics_json =
                serde_json::to_string(&metrics).unwrap_or_else(|_| "{}".to_string());
            let metrics_key = format!("identity_ab_metrics_{}", variant.to_uppercase());
            let _ = self.db.set_key(&metrics_key, &metrics_json).await;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                Some(trace_id),
                json!({
                    "event": "identity_ab_eval_ready",
                    "variant": variant,
                    "turns": next_turns,
                    "score": metrics.score,
                }),
            )
            .await;

            let other_variant = if variant.eq_ignore_ascii_case("A") {
                "B".to_string()
            } else {
                "A".to_string()
            };
            let other_metrics_key = format!("identity_ab_metrics_{}", other_variant.to_uppercase());
            let other_metrics_raw = self.db.get_key(&other_metrics_key).await.ok().flatten();
            let other_metrics = other_metrics_raw
                .as_deref()
                .and_then(|raw| serde_json::from_str::<IdentityAbMetrics>(raw).ok());
            let decision = decide_identity_ab_variant(&metrics, other_metrics.as_ref(), IDENTITY_AB_MIN_TURNS);
            let (next_variant, reason) = match decision {
                IdentityAbDecision::SwitchForCollection(next_variant) => (next_variant, "collect_baseline"),
                IdentityAbDecision::SwitchForWinner(next_variant) => (next_variant, "score_preference"),
                IdentityAbDecision::Stay => (variant.clone(), "stay"),
            };
            let _ = self.db.set_key("identity_ab_variant", &next_variant).await;
            let _ = self.db.set_key("identity_ab_turns", "0").await;
            let _ = self
                .db
                .set_key("identity_ab_window_start", &Utc::now().to_rfc3339())
                .await;
            let event = if next_variant.eq_ignore_ascii_case(&variant) {
                "identity_ab_eval_complete"
            } else {
                "identity_ab_variant_switched"
            };
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                Some(trace_id),
                json!({
                    "event": event,
                    "from": variant,
                    "to": next_variant,
                    "reason": reason,
                }),
            )
            .await;
            return next_variant;
        }

        variant
    }

    pub(super) fn decision_needed(
        &self,
        state: &KernelState,
        _outcomes: &[Outcome],
        last_monologue_at_override: Option<&str>,
    ) -> bool {
        decision_needed_for(state, last_monologue_at_override)
    }

    pub(super) fn should_use_decision_mode(&self, state: &KernelState, has_new_user_input: bool) -> bool {
        if !has_new_user_input {
            return false;
        }
        let input = state.last_user_input.as_deref().unwrap_or("");
        let token_count = count_tokens(input);
        let has_question = input.contains('?');
        let has_pending = !state.pending_questions.is_empty();
        let low_pressure = state.pressure_score < 1.0 && state.failure_count < 2;
        if token_count < 12 && !has_question && !has_pending && low_pressure {
            return false;
        }
        true
    }

    pub(super) fn decision_turns_for(&self, state: &KernelState) -> usize {
        let input = state.last_user_input.as_deref().unwrap_or("");
        let token_count = count_tokens(input);
        let has_question = input.contains('?');
        let high = state.pressure_score >= 2.0
            || state.failure_count >= 3
            || state.uncertainty_count >= 2;
        let moderate = has_question
            || token_count >= 20
            || !state.pending_questions.is_empty()
            || state.pressure_score >= 1.0
            || state.failure_count >= 2
            || state.uncertainty_count >= 1;

        if high {
            5
        } else if moderate {
            3
        } else {
            2
        }
    }

    pub(super) async fn run_input_loop(
        &self,
        input: String,
        input_kind: CoreInputKind,
        input_source: &str,
        run_id: String,
        trace_id: String,
        conversation_id: String,
        cancel_rx: watch::Receiver<bool>,
        original_input: Option<String>,
        assistant_message_id: Option<String>,
        depth: usize,
    ) -> Result<RunOutput, String> {
        let mut current_input = input;
        let mut current_kind = input_kind;
        let mut current_source = input_source.to_string();
        let mut loop_depth = depth;
        let current_original = original_input;
        let run_id = run_id;
        let trace_id = trace_id;
        let conversation_id = conversation_id;
        let cancel_rx = cancel_rx;
        let mut assistant_message_id = assistant_message_id;
        let run_lock = {
            let mut locks = self.run_locks.lock().await;
            locks
                .entry(run_id.clone())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                .clone()
        };
        let _run_guard = run_lock.lock().await;
        let control_map = system_controls::load_control_map(&self.db).await;
        let kernel_mode = system_controls::mode_for("kernel_loop", &control_map);
        if system_controls::mode_is_off(&kernel_mode) {
            let fallback = direct_answer_fallback(&current_input);
            return Ok(RunOutput {
                response: fallback,
                tool_result: None,
                assistant_message_id,
            });
        }
        let max_tool_loops = if system_controls::mode_is_degraded(&kernel_mode) {
            1
        } else {
            MAX_TOOL_LOOPS
        };

        loop {
            if loop_depth >= max_tool_loops {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "tool_loop_limit",
                        "depth": loop_depth,
                    }),
                )
                .await;
            }

            let mut result = self
                .run_input_once(
                    current_input,
                    current_kind,
                    &current_source,
                    run_id.clone(),
                    trace_id.clone(),
                    conversation_id.clone(),
                    cancel_rx.clone(),
                    current_original.clone(),
                    assistant_message_id.clone(),
                )
                .await?;

            if assistant_message_id.is_none() {
                assistant_message_id = result.assistant_message_id.clone();
            }
            if let Some(mut tool_result) = result.tool_result {
                if loop_depth >= max_tool_loops {
                    result.tool_result = None;
                    result.assistant_message_id = assistant_message_id.clone();
                    return Ok(result);
                }
                if !phi_consent_allowed(&self.db.pool, Some(&conversation_id)).await {
                    let (redacted, sensitivity) = redact_sensitive_text(&tool_result.output);
                    if let Some(level) = sensitivity {
                        tool_result.output = redacted;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "memory",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "phi_redacted",
                                "scope": "tool_output",
                                "sensitivity": level.as_str(),
                                "tool": tool_result.tool_name.clone(),
                                "conversation_id": conversation_id,
                            }),
                        )
                        .await;
                    }
                }
                current_input = tool_result.output;
                current_kind = if tool_result.is_error {
                    CoreInputKind::ToolError
                } else {
                    CoreInputKind::ToolResult
                };
                current_source = tool_result.tool_name;
                loop_depth += 1;
                continue;
            }

            result.assistant_message_id = assistant_message_id.clone();
            return Ok(result);
        }
    }

    async fn maybe_seed_telemetry_snapshot(
        &self,
        run_id: Option<&str>,
        trace_id: Option<&str>,
    ) {
        let key_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM kv_store WHERE key LIKE 'telemetry.%'",
        )
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0);
        if key_count > 0 {
            return;
        }
        match crate::core::self_memory::telemetry::record_telemetry_snapshot_force(&self.db, None).await {
            Ok(wrote) => {
                if wrote {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        run_id,
                        trace_id,
                        json!({
                            "event": "telemetry_snapshot_seeded",
                            "reason": "kernel_run",
                        }),
                    )
                    .await;
                }
            }
            Err(err) => {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    run_id,
                    trace_id,
                    json!({
                        "event": "telemetry_snapshot_error",
                        "reason": "kernel_run",
                        "error": err,
                    }),
                )
                .await;
            }
        }
    }

    async fn maybe_seed_goal_stack_from_plan(
        &self,
        state: &mut KernelState,
        conversation_id: &str,
        run_id: &str,
        plan_id: &str,
    ) {
        if !state.workspace_goal_stack.is_empty() || plan_id.trim().is_empty() {
            return;
        }
        let row = sqlx::query(
            "SELECT intent, steps_json FROM action_proposals WHERE proposal_id = ? LIMIT 1",
        )
        .bind(plan_id)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten();
        let Some(row) = row else {
            return;
        };
        let intent: String = row.try_get("intent").unwrap_or_default();
        let steps_json: String = row.try_get("steps_json").unwrap_or_default();
        let mut steps: Vec<crate::models::GoalStep> = Vec::new();
        if let Ok(value) = serde_json::from_str::<Value>(&steps_json) {
            if let Some(list) = value.get("steps").and_then(|v| v.as_array()) {
                for (idx, step) in list.iter().enumerate() {
                    let text = step
                        .get("description")
                        .and_then(|v| v.as_str())
                        .or_else(|| step.get("text").and_then(|v| v.as_str()))
                        .or_else(|| step.as_str())
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .unwrap_or_else(|| format!("Step {}", idx + 1));
                    steps.push(crate::models::GoalStep {
                        text,
                        status: None,
                        evidence_event_ids: Vec::new(),
                        belief_ids: Vec::new(),
                        completed_at: None,
                    });
                    if steps.len() >= 6 {
                        break;
                    }
                }
            }
        }
        if steps.is_empty() {
            steps = vec![
                crate::models::GoalStep {
                    text: "Clarify objective and constraints".to_string(),
                    status: None,
                    evidence_event_ids: Vec::new(),
                    belief_ids: Vec::new(),
                    completed_at: None,
                },
                crate::models::GoalStep {
                    text: "Execute the next concrete step".to_string(),
                    status: None,
                    evidence_event_ids: Vec::new(),
                    belief_ids: Vec::new(),
                    completed_at: None,
                },
                crate::models::GoalStep {
                    text: "Verify outcome and capture evidence".to_string(),
                    status: None,
                    evidence_event_ids: Vec::new(),
                    belief_ids: Vec::new(),
                    completed_at: None,
                },
            ];
        }
        let goal_text = if intent.trim().is_empty() {
            format!("Plan: {}", plan_id)
        } else {
            format!("Plan: {}", intent.trim())
        };
        state.workspace_goal_stack = vec![crate::models::GoalStackItem {
            goal: goal_text.clone(),
            steps,
            current_step_index: 0,
            status: None,
            evidence_event_ids: Vec::new(),
            belief_ids: Vec::new(),
            updated_at: Some(Utc::now().to_rfc3339()),
        }];
        state.workspace_goal_thread = Some(goal_text.clone());
        let workspace_state = crate::models::WorkspaceState {
            conversation_id: conversation_id.to_string(),
            goal_thread: state.workspace_goal_thread.clone(),
            active_plan_id: state.workspace_active_plan_id.clone(),
            goal_stack: state.workspace_goal_stack.clone(),
            open_questions: state.workspace_open_questions.clone(),
            active_hypotheses: state.workspace_active_hypotheses.clone(),
            working_set_topics: state.workspace_working_set_topics.clone(),
            current_focus: state.workspace_current_focus.clone(),
            focus_rationale: state.workspace_focus_rationale.clone(),
            workspace_meta: state.workspace_meta.clone(),
            updated_at: None,
        };
        let _ = self.db.set_workspace_state(&workspace_state).await;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(run_id),
            None,
            json!({
                "event": "goal_stack_seeded_from_plan",
                "proposal_id": plan_id,
                "goal": goal_text,
                "step_count": state.workspace_goal_stack.first().map(|g| g.steps.len()).unwrap_or(0),
            }),
        )
        .await;
    }

    pub(super) async fn run_input_once(
        &self,
        input: String,
        input_kind: CoreInputKind,
        input_source: &str,
        run_id: String,
        trace_id: String,
        conversation_id: String,
        mut cancel_rx: watch::Receiver<bool>,
        original_input: Option<String>,
        assistant_message_id: Option<String>,
    ) -> Result<RunOutput, String> {
        if *cancel_rx.borrow() {
            return Err("cancelled".to_string());
        }

        let turn_started = Instant::now();
        let ingest_started = Instant::now();
        let mut tool_ms: i64 = 0;
        let mut emit_ms: i64 = 0;
        let mem_write_ms: i64 = 0;

        let now = Utc::now();
        let _ = self.db.touch_run_heartbeat(&run_id).await;
        let settings = self.db.get_settings().await.map_err(|e| e.to_string())?;
        self.sync_self_model_runtime(&settings).await;
        self.maybe_seed_telemetry_snapshot(Some(&run_id), Some(&trace_id)).await;
        let mut state = self.load_state(&conversation_id).await;
        let mut anchor_shift_event: Option<AnchorShiftEvent> = None;
        let explicit_feedback = original_input
            .as_deref()
            .map(feedback::is_explicit_feedback)
            .unwrap_or(false);
        let event_type = match input_kind {
            CoreInputKind::User => "user_message",
            CoreInputKind::ToolResult | CoreInputKind::ToolError => "tool_output",
            CoreInputKind::SystemContext => "system_transition",
        };
        let event_payload = json!({
            "content": input.clone(),
            "source": input_source,
            "kind": format!("{:?}", input_kind),
            "original_input": original_input.clone(),
            "explicit_feedback": explicit_feedback,
        });
        let event_tags = json!({
            "domain": if matches!(input_kind, CoreInputKind::User) { "user" } else { "system" },
            "error": matches!(input_kind, CoreInputKind::ToolError),
        });
        let _ = self
            .db
            .insert_event_ledger(
                &Uuid::new_v4().to_string(),
                event_type,
                &event_payload,
                Some(&event_tags),
                Some(&run_id),
                Some(&trace_id),
            )
            .await;
        let _ = advance_run_phase(
            &self.db.pool,
            Some(&self.app_handle),
            &run_id,
            RunPhase::Ingest,
            Some("input_once_start"),
        )
        .await;
        if matches!(input_kind, CoreInputKind::User) && state.diagnostics_disabled_turns_remaining > 0 {
            state.diagnostics_disabled_turns_remaining =
                state.diagnostics_disabled_turns_remaining.saturating_sub(1);
            if state.diagnostics_disabled_turns_remaining == 0 {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "diagnostics_breaker_cleared",
                    }),
                )
                .await;
            }
        }
        let _ = self.mark_stale_tool_dispatches().await;
        self.maybe_log_tool_baseline_snapshot().await;
        let disable_working_hypothesis = settings.stability_disable_working_hypothesis.unwrap_or(true);
        let expanded_state_disclosure = settings.stability_state_disclosure_expanded.unwrap_or(true);
        let calculator_mode = matches!(input_kind, CoreInputKind::User) && is_calculator_prompt(&input);
        let self_audit_mode = matches!(input_kind, CoreInputKind::User) && is_self_audit_request(&input);
        let diagnostics_breaker_active = state.diagnostics_disabled_turns_remaining > 0;
        let allow_diagnostics = self_audit_mode && !diagnostics_breaker_active;
        let allow_speculative_markers = if matches!(input_kind, CoreInputKind::User) {
            allow_speculative_markers_for_prompt(&input, allow_diagnostics)
        } else {
            allow_diagnostics
        };
        let self_awareness_mode = settings
            .self_awareness_expression_mode
            .as_deref()
            .unwrap_or("conservative");
        let explicit_self_awareness =
            matches!(input_kind, CoreInputKind::User) && is_self_awareness_query(&input);
        let intent_gate_detection = if matches!(input_kind, CoreInputKind::User) {
            crate::core::prompt_builder::detect_context_intent(&input, false, false)
        } else {
            crate::core::prompt_builder::IntentDetection {
                tags: Vec::new(),
                matched_rules: Vec::new(),
            }
        };
        let (self_awareness_requested, _self_awareness_requested_reason) =
            is_self_awareness_gate_requested(explicit_self_awareness, &intent_gate_detection.tags);
        let self_awareness_allowed =
            settings.self_report_channel.unwrap_or(true)
                && !self_awareness_mode.eq_ignore_ascii_case("conservative");
        let self_awareness_query = self_awareness_requested && self_awareness_allowed;
        let self_audit_ambiguous = matches!(input_kind, CoreInputKind::User)
            && !self_audit_mode
            && !self_awareness_requested
            && is_self_audit_ambiguous(&input)
            && !user_requested_state(&input)
            && !is_introspection_request(&input);
        let relational_mode = matches!(input_kind, CoreInputKind::User) && is_relational_input(&input);
        let is_task_input =
            matches!(input_kind, CoreInputKind::User | CoreInputKind::SystemContext) && !relational_mode;
        let ingest_ms = ingest_started.elapsed().as_millis() as i64;
        if is_task_input {
            let new_task = state.task_id.as_deref() != Some(&run_id);
            if new_task {
                state.task_id = Some(run_id.clone());
                state.task_phase = TaskPhase::Running;
                let default_ask_budget = settings.ask_budget_max.unwrap_or(1).max(0);
                let calculator_budget = settings.calculator_followups_max.unwrap_or(0).max(0);
                state.ask_budget_max = if calculator_mode {
                    calculator_budget
                } else {
                    default_ask_budget
                };
                state.ask_budget_remaining = state.ask_budget_max;
                state.user_refused = false;
                state.refused_slots.clear();
                state.refusal_count = 0;
                state.ask_loop_breaker_triggered = false;
                state.tool_loop_breaker_triggered = false;
                state.question_fingerprints.clear();
                state.recent_questions.clear();
                state.asked_slot_sets.clear();
                state.missing_slots.clear();
                state.resolved_slots.clear();
                state.resolution_mode = None;
                state.missing_input_policy = None;
                state.last_asked_slots.clear();
                state.slot_provenance.clear();
                state.tool_call_fingerprints.clear();
                state.stop_latch = false;
                state.stop_reason = None;
                state.stop_scope = None;
                state.stop_state = StopState::default();
                state.state_disclosure_suppressed_until = None;
                state.state_disclosure_prompt_streak = 0;
                state.monologue_quiet_until = None;
                state.monologue_surface_until = None;
                state.monologue_emit_loop_breaker_triggered = false;
                state.monologue_misaligned_streak = 0;
                state.recent_emit_messages.clear();
                state.recent_emit_fingerprints.clear();
            } else if calculator_mode {
                let calculator_budget = settings.calculator_followups_max.unwrap_or(0).max(0);
                state.ask_budget_max = calculator_budget;
                if state.ask_budget_remaining > calculator_budget {
                    state.ask_budget_remaining = calculator_budget;
                }
            } else if matches!(state.task_phase, TaskPhase::AwaitingUser) {
                state.task_phase = TaskPhase::Running;
            }
        }
        if matches!(input_kind, CoreInputKind::User | CoreInputKind::SystemContext) {
            let _ = self
                .db
                .create_memory_pass_token(&run_id, &conversation_id, 600)
                .await;
        }
        if matches!(input_kind, CoreInputKind::User) {
            let previous_input = state.last_user_input.clone().unwrap_or_default();
            let (redirect_detected, redirect_overlap, redirect_reason) =
                is_user_redirect(&previous_input, &input);
            let topic_shift_request = is_topic_shift_request(&input);
            let redirect_focus = extract_redirect_focus(&input)
                .map(|focus| summarize_snippet(&focus, 160));
            if redirect_detected || topic_shift_request {
                state.anchor_epoch = state.anchor_epoch.saturating_add(1);
                state.user_redirect_turns_remaining = 2;
                state.redirect_focus = redirect_focus.clone();
                state.redirect_focus_confirmed_turns = if redirect_focus.is_some() { 1 } else { 0 };
                state.redirect_focus_miss_turns = 0;
                state.redirect_focus_explicit = topic_shift_request
                    || redirect_reason == "explicit_redirect"
                    || redirect_focus.is_some();
                if redirect_focus.is_none() {
                    let should_clarify = topic_shift_request || redirect_reason == "explicit_redirect";
                    let clarifier = "What should I focus on instead?";
                    let mut already_pending = false;
                    if should_clarify && state.last_redirect_clarifier_epoch != state.anchor_epoch {
                        if let Ok(existing) = self.db.list_pending_prompts(&conversation_id, 12).await {
                            already_pending = existing.iter().any(|(_, prompt, source, _, _, _, _, _, _, _, _, _, _, _, _)| {
                                source == "kernel_redirect" && prompt.trim().eq_ignore_ascii_case(clarifier)
                            });
                        }
                        if !already_pending {
                            let expires_at = compute_expires_at(Utc::now(), PENDING_PROMPT_EXPIRES_SECS);
                            let anchor_message_id = state.last_user_message_id.as_deref();
                            let anchor_created_at = state.last_user_input_at.as_deref();
                            let anchor_hash = state
                                .last_user_input
                                .as_deref()
                                .map(|input| crate::core::kernel::utils::text::hash_payload(&summarize_snippet(input, 160)));
                            if let Ok(prompt_id) = self
                                .db
                                .enqueue_pending_prompt(
                                    &conversation_id,
                                    clarifier,
                                    "kernel_redirect",
                                    true,
                                    Some("AskUserQuestion"),
                                    None,
                                    Some(&expires_at),
                                    anchor_message_id,
                                    anchor_hash.as_deref(),
                                    anchor_created_at,
                                    Some("user"),
                                )
                                .await
                            {
                                state.last_redirect_clarifier_epoch = state.anchor_epoch;
                                if let Ok(count) = self.db.count_pending_prompts(&conversation_id).await {
                                    let _ = self.app_handle.emit("pending_prompt_count", count as usize);
                                }
                                let _ = system_log::log_event(
                                    &self.db.pool,
                                    Some(&self.app_handle),
                                    "info",
                                    "kernel",
                                    Some(&run_id),
                                    Some(&trace_id),
                                    json!({
                                        "event": "redirect_clarifier_enqueued",
                                        "prompt_id": prompt_id,
                                        "reason": "missing_redirect_focus",
                                    }),
                                )
                                .await;
                            }
                        } else {
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "kernel",
                                Some(&run_id),
                                Some(&trace_id),
                                json!({
                                    "event": "redirect_clarifier_skipped",
                                    "reason": "already_pending",
                                }),
                            )
                            .await;
                        }
                    } else if redirect_reason == "low_overlap" {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "redirect_clarifier_skipped",
                                "reason": "low_overlap_only",
                            }),
                        )
                        .await;
                    }
                }
                let old_anchor = if previous_input.trim().is_empty() {
                    "None".to_string()
                } else {
                    summarize_snippet(&previous_input, 160)
                };
                let new_anchor = summarize_snippet(&input, 160);
                let shift_reason = if is_topic_shift_request(&input) {
                    "topic_shift_request"
                } else {
                    redirect_reason
                };
                anchor_shift_event = Some(AnchorShiftEvent {
                    old_anchor: old_anchor.clone(),
                    new_anchor: new_anchor.clone(),
                    reason: shift_reason.to_string(),
                    overlap: redirect_overlap,
                    anchor_epoch: state.anchor_epoch,
                });
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "anchor_redirected",
                        "reason": shift_reason,
                        "overlap": redirect_overlap,
                        "old_anchor": old_anchor,
                        "new_anchor": new_anchor,
                        "anchor_epoch": state.anchor_epoch,
                    }),
                )
                .await;
                let _ = self
                    .clear_stale_pending_prompts(&conversation_id, &previous_input, &input)
                    .await;
            } else if state.user_redirect_turns_remaining > 0 {
                state.user_redirect_turns_remaining -= 1;
                if state.user_redirect_turns_remaining == 0 {
                    state.redirect_focus = None;
                    state.redirect_focus_confirmed_turns = 0;
                    state.redirect_focus_miss_turns = 0;
                    state.redirect_focus_explicit = false;
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "anchor_redirect_cleared",
                            "anchor_epoch": state.anchor_epoch,
                        }),
                    )
                    .await;
                }
            }
            state.last_user_input = Some(input.clone());
            state.last_user_input_at = Some(now.to_rfc3339());
            if let Ok(Some(message_id)) = self.db.get_user_message_id_for_run(&run_id).await {
                state.last_user_message_id = Some(message_id);
            }
            if !redirect_detected {
                if let Some(focus) = state.redirect_focus.clone() {
                    if redirect_focus_aligned(&input, &focus) {
                        state.redirect_focus_confirmed_turns =
                            state.redirect_focus_confirmed_turns.saturating_add(1);
                        state.redirect_focus_miss_turns = 0;
                    } else {
                        state.redirect_focus_miss_turns =
                            state.redirect_focus_miss_turns.saturating_add(1);
                        state.redirect_focus_confirmed_turns = 0;
                    }
                    if state.redirect_focus_explicit
                        && state.redirect_focus_confirmed_turns >= REDIRECT_FOCUS_PROMOTE_TURNS
                        && !is_generic_redirect_focus(&focus)
                    {
                        state.workspace_current_focus = Some(focus.clone());
                        state.workspace_focus_rationale = Some("user_redirect_stable".to_string());
                        state.workspace_meta.current_focus = Some(make_field_meta(true, &[], &[]));
                        state.workspace_meta.focus_rationale = Some(make_field_meta(true, &[], &[]));
                        state.last_focus_change_at = Some(now.to_rfc3339());
                        state.redirect_focus = None;
                        state.redirect_focus_confirmed_turns = 0;
                        state.redirect_focus_miss_turns = 0;
                        state.redirect_focus_explicit = false;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "redirect_focus_promoted",
                                "focus": focus,
                                "reason": "stable_redirect_focus",
                            }),
                        )
                        .await;
                    } else if state.redirect_focus_miss_turns >= REDIRECT_FOCUS_PROMOTE_TURNS {
                        state.redirect_focus = None;
                        state.redirect_focus_confirmed_turns = 0;
                        state.redirect_focus_miss_turns = 0;
                        state.redirect_focus_explicit = false;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "redirect_focus_cleared",
                                "reason": "alignment_missed",
                            }),
                        )
                        .await;
                    }
                }
            }
            state.self_state.last_internal_thought = "Processing user input.".to_string();
            state.self_state.updated_at = Some(now.to_rfc3339());
            state.monologue_quiet_until = None;
            state.monologue_emit_loop_breaker_triggered = false;
            if !disclosure_suppressed_until(state.state_disclosure_suppressed_until.as_deref()) {
                state.state_disclosure_suppressed_until = None;
            }
            let requested_state = user_requested_state(&input) || self_audit_mode;
            if requested_state {
                state.state_disclosure_suppressed_until = None;
                state.state_disclosure_prompt_streak = 0;
            } else if is_state_disclosure_refusal(&input) || is_topic_shift_request(&input) {
                let until = now + chrono::Duration::minutes(10);
                state.state_disclosure_suppressed_until = Some(until.to_rfc3339());
                state.state_disclosure_prompt_streak = 0;
                if is_topic_shift_request(&input) {
                    let _ = self.db.clear_pending_prompts(&conversation_id).await;
                    let _ = self.app_handle.emit("pending_prompt_count", 0usize);
                }
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "state_disclosure_suppressed",
                        "until": state.state_disclosure_suppressed_until,
                        "reason": if is_state_disclosure_refusal(&input) { "user_refusal" } else { "topic_shift" },
                    }),
                )
                .await;
            }
            if settings.monologue_surface_enabled.unwrap_or(false)
                && is_monologue_surface_request(&input)
            {
                let until = now + chrono::Duration::seconds(MONOLOGUE_SURFACE_WINDOW_SECS);
                state.monologue_surface_until = Some(until.to_rfc3339());
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "monologue_surface_requested",
                        "surface_until": state.monologue_surface_until,
                    }),
                )
                .await;
            } else {
                state.monologue_surface_until = None;
            }
            if calculator_mode && state.missing_input_policy.is_none() {
                let lowered = input.to_lowercase();
                if lowered.contains("strict") || lowered.contains("no defaults") || lowered.contains("don't assume") {
                    state.missing_input_policy = Some("strict".to_string());
                } else {
                    state.missing_input_policy = Some("use_defaults_and_label".to_string());
                }
            }
            if is_refusal_input(&input) && !state.last_asked_slots.is_empty() {
                state.user_refused = true;
                state.refusal_count += 1;
                let mut refused = state.refused_slots.clone();
                for slot in state.last_asked_slots.iter() {
                    if !refused.contains(slot) {
                        refused.push(slot.clone());
                    }
                }
                state.refused_slots = refused;

                let (ctx, _meta) = self
                    .build_resolution_context(&conversation_id, &settings, &input)
                    .await;
                let resolution = resolve_required_slots(&state.last_asked_slots, &ctx);
                state.resolved_slots = resolution.resolved.keys().cloned().collect();
                state.missing_slots = resolution.missing.clone();
                state.slot_provenance = resolution.slot_provenance.clone();

                let strict = state
                    .missing_input_policy
                    .as_deref()
                    .map(|m| m.eq_ignore_ascii_case("strict"))
                    .unwrap_or(false);
                if strict {
                    state.task_phase = TaskPhase::Aborting;
                    state.resolution_mode = Some("abort".to_string());
                    let mut scope = StopScope::default();
                    scope.tools = true;
                    scope.memory_write = true;
                    scope.self_claims = true;
                    scope.monologue_run = true;
                    scope.monologue_emit = true;
                    scope.background_jobs = true;
                    apply_stop_state(
                        &mut state,
                        StopReason {
                            category: StopReasonCategory::LatchBlock,
                            subcode: "user_refused_strict".to_string(),
                            contract: None,
                        },
                        scope,
                    );
                } else if resolution.missing.is_empty() {
                    let defaults_used = defaults_used_from_provenance(&resolution.slot_provenance);
                    if !defaults_used.is_empty() {
                        state.resolution_mode = Some("defaults_used".to_string());
                    }
                    state.task_phase = TaskPhase::ResolvingWithDefaults;
                } else {
                    state.resolution_mode = Some("partial".to_string());
                    state.task_phase = TaskPhase::ResolvingWithDefaults;
                }

                if calculator_mode || state.missing_input_policy.is_some() {
                    state.ask_budget_remaining = 0;
                } else if state.ask_budget_remaining > 0 {
                    state.ask_budget_remaining -= 1;
                }

                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "user_refused_inputs",
                        "task_id": state.task_id,
                        "refused_slots": state.refused_slots,
                        "missing_slots": state.missing_slots,
                        "resolution_mode": state.resolution_mode,
                        "task_phase": task_phase_label(&state.task_phase),
                    }),
                )
                .await;
            }
        } else if matches!(input_kind, CoreInputKind::SystemContext) {
            state.self_state.last_internal_thought = "Processing system input.".to_string();
            state.self_state.updated_at = Some(now.to_rfc3339());
        }

        self.refresh_controller_state(&mut state, &settings).await;
        let original_input = original_input.or_else(|| state.last_user_input.clone());
        self.refresh_research_budget(&mut state, &settings);
        let budget_remaining = self.research_budget_remaining(&state, &settings);
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(&run_id),
            Some(&trace_id),
            json!({
                "event": "research_budget_start",
                "remaining": budget_remaining,
                "window_start": state.research_window_start,
            }),
        )
        .await;
        if matches!(input_kind, CoreInputKind::User) && !state.pending_questions.is_empty() {
            state.pending_questions.clear();
            state.uncertainty_count = (state.uncertainty_count - 1).max(0);
        }

        let outcomes = self.collect_outcomes(&mut state).await;
        let action_type = match input_kind {
            CoreInputKind::User => "user_message_processed",
            _ => "system_input_processed",
        };
        let input_outcome = Outcome {
            action_type: action_type.to_string(),
            success: true,
            observations: summarize_snippet(&input, 160),
            source: input_source.to_string(),
            failure_kind: None,
            target_key: None,
            action_id: None,
            timestamp: now.to_rfc3339(),
        };
        let mut all_outcomes = outcomes;
        all_outcomes.push(input_outcome);
        let user_evidence_allowlist = self
            .db
            .get_recent_user_evidence(&conversation_id, 8)
            .await;
        let user_evidence_ids: Vec<i64> = user_evidence_allowlist
            .iter()
            .map(|(id, _)| *id)
            .collect();
        let user_evidence_allowlist_set: HashSet<i64> =
            user_evidence_ids.iter().cloned().collect();
        let mut identity_evidence_ids: Vec<i64> = Vec::new();
        if matches!(input_kind, CoreInputKind::User) {
            if let Some((target, pattern)) = detect_identity_signal(&input) {
                if let Some((evidence_id, snippet)) = user_evidence_allowlist.first() {
                    identity_evidence_ids.push(*evidence_id);
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "identity_evidence_detected",
                            "target": target,
                            "pattern": pattern,
                            "evidence_event_id": evidence_id,
                            "snippet": snippet,
                        }),
                    )
                    .await;
                }
            }
        }
        let tool_failure_current = self.tool_failure_detected_for_run(&run_id).await;
        let mut tool_failure_cross_run = false;
        if let Some(window_mins) = settings
            .tool_failure_gate_window_mins
            .filter(|mins| *mins > 0)
        {
            let tool_names = parse_tool_name_filter(settings.tool_failure_gate_tool_names.as_deref());
            if !tool_names.is_empty() {
                tool_failure_cross_run = self
                    .tool_failure_detected_in_window(window_mins, &tool_names)
                    .await;
                if tool_failure_cross_run {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!( {
                            "event": "tool_failure_gate_cross_run",
                            "window_mins": window_mins,
                            "tool_names": tool_names,
                        }),
                    )
                    .await;
                }
            }
        }
        let tool_failure_detected = tool_failure_current || tool_failure_cross_run;
        let context_evidence_ids = if matches!(input_kind, CoreInputKind::ToolResult)
            && input_source == "read_context"
        {
            extract_context_evidence_ids(&input)
        } else {
            Vec::new()
        };
        let _identity_ab_variant = if matches!(input_kind, CoreInputKind::User) {
            self.update_identity_ab_variant(&run_id, &trace_id).await
        } else {
            self.db
                .get_key("identity_ab_variant")
                .await
                .ok()
                .flatten()
                .unwrap_or_else(|| "A".to_string())
        };
        let identity_enforcement_enabled = true;

        let feedback_explicit_required = settings.explicit_feedback_only.unwrap_or(true);
        let feedback_allowed = if feedback_explicit_required {
            explicit_feedback
        } else {
            true
        };
        if matches!(input_kind, CoreInputKind::User) && feedback_allowed {
            if let Some(kind) = classify_user_feedback(&input) {
                let assistant_message = self
                    .db
                    .get_last_assistant_message(&conversation_id)
                    .await;
                let assistant_message_id = assistant_message.as_ref().map(|(id, _)| id.clone());
                let evidence_id = self
                    .db
                    .create_user_feedback_evidence(
                        &conversation_id,
                        assistant_message_id.as_deref(),
                        &input,
                        kind.as_str(),
                    )
                    .await;
                let feedback_payload = json!({
                    "kind": kind.as_str(),
                    "assistant_message_id": assistant_message_id,
                    "evidence_event_id": evidence_id,
                    "content": input,
                    "explicit": explicit_feedback,
                });
                let feedback_tags = json!({
                    "domain": "feedback",
                    "kind": kind.as_str(),
                    "explicit": explicit_feedback,
                });
                let _ = self
                    .db
                    .insert_event_ledger(
                        &Uuid::new_v4().to_string(),
                        "feedback",
                        &feedback_payload,
                        Some(&feedback_tags),
                        Some(&run_id),
                        Some(&trace_id),
                    )
                    .await;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "user_feedback_detected",
                        "kind": kind.as_str(),
                        "assistant_message_id": assistant_message_id,
                        "evidence_event_id": evidence_id,
                        "explicit": explicit_feedback,
                    }),
                )
                .await;
                let (tag, base_confidence) = match kind {
                    UserFeedbackKind::Agree => ("user_agree", 0.8),
                    UserFeedbackKind::FollowUp => ("user_needs_depth", 0.7),
                    UserFeedbackKind::Clarify => ("user_needs_clarity", 0.7),
                    UserFeedbackKind::Pushback => ("user_pushback", 0.8),
                    UserFeedbackKind::Disengage => ("user_disengage", 0.9),
                };
                let inferred = !explicit_feedback || evidence_id.is_none();
                let confidence = if explicit_feedback { base_confidence } else { base_confidence * 0.7 };
                let evidence_ids = evidence_id.map(|id| vec![id]).unwrap_or_default();
                let _ = self
                    .db
                    .upsert_context_tag(
                        &conversation_id,
                        tag,
                        confidence as f32,
                        inferred,
                        &evidence_ids,
                        Some("user_feedback"),
                    )
                    .await;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "context",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "context_tag_updated",
                        "conversation_id": conversation_id,
                        "tag": tag,
                        "confidence": confidence,
                        "inferred": inferred,
                        "evidence_event_ids": evidence_ids,
                    }),
                )
                .await;
                let _ = self
                    .db
                    .mark_counterfactual_observed(&conversation_id, kind.as_str())
                    .await;
                let reward_magnitude: f64 = match kind {
                    UserFeedbackKind::Agree => 0.5,
                    UserFeedbackKind::FollowUp => 0.2,
                    UserFeedbackKind::Clarify => -0.2,
                    UserFeedbackKind::Pushback => -0.6,
                    UserFeedbackKind::Disengage => -0.8,
                };
                if reward_magnitude.abs() > 0.0 {
                    let label_id: Option<String> = sqlx::query_scalar(
                        "SELECT label_id FROM qualia_labels ORDER BY datetime(created_at) DESC LIMIT 1",
                    )
                    .fetch_optional(&self.db.pool)
                    .await
                    .ok()
                    .flatten();
                    if let Some(label_id) = label_id {
                        let reward_id = Uuid::new_v4().to_string();
                        let _ = sqlx::query(
                            "INSERT INTO qualia_reward_events (reward_id, label_id, magnitude, outcome_ref, created_at)
                             VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                        )
                        .bind(&reward_id)
                        .bind(&label_id)
                        .bind(reward_magnitude)
                        .bind(assistant_message_id.as_deref())
                        .execute(&self.db.pool)
                        .await;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "qualia_reward_recorded",
                                "reward_id": reward_id,
                                "label_id": label_id,
                                "magnitude": reward_magnitude,
                                "outcome_ref": assistant_message_id,
                            }),
                        )
                        .await;
                    }
                }
                all_outcomes.push(Outcome {
                    action_type: format!("user_feedback_{}", kind.as_str()),
                    success: true,
                    observations: summarize_snippet(&input, 160),
                    source: "user_feedback".to_string(),
                    failure_kind: None,
                    target_key: None,
                    action_id: assistant_message_id,
                    timestamp: now.to_rfc3339(),
                });

                if matches!(kind, UserFeedbackKind::Pushback) {
                    let mut demoted = Vec::new();
                    if let Some(meta) = state.workspace_meta.current_focus.as_mut() {
                        if !meta.speculative {
                            meta.speculative = true;
                            demoted.push("current_focus".to_string());
                        }
                    }
                    for hypothesis in state.workspace_active_hypotheses.iter_mut() {
                        if !hypothesis.speculative {
                            hypothesis.speculative = true;
                            demoted.push(format!("hypothesis:{}", hypothesis.text));
                        }
                    }
                    if !demoted.is_empty() {
                        state.workspace_meta.active_hypotheses = state.workspace_active_hypotheses.clone();
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "workspace_demoted_user_feedback",
                                "fields": demoted,
                            }),
                        )
                        .await;
                    }
                }
            }
        }

        let mut candidates = Vec::new();
        let mut created_at = 0i64;

        let input_snippet = summarize_snippet(&input, 240);
        let source_type = match input_kind {
            CoreInputKind::User => "user",
            _ => "system",
        };
        candidates.push(self.make_candidate(
            CandidateKind::WriteEpisodic,
            json!({
                "event_type": "run_started",
                "payload": { "status": "started" },
                "source_type": "system",
                "source_ref": run_id,
                "evidence_class": "internal",
            }),
            "run_started",
            &mut created_at,
        ));
        candidates.push(self.make_candidate(
            CandidateKind::WriteEpisodic,
            json!({
                "event_type": "message_received",
                "payload": { "summary_snippet": input_snippet },
                "source_type": source_type,
                "source_ref": run_id,
                "evidence_class": "internal",
            }),
            "message_received",
            &mut created_at,
        ));
        if matches!(input_kind, CoreInputKind::User) {
            let source_ref = state
                .last_user_message_id
                .clone()
                .unwrap_or_else(|| run_id.clone());
            if is_identity_claim_text(&input) {
                candidates.push(self.make_candidate(
                    CandidateKind::WriteEpisodic,
                    json!({
                        "event_type": "identity_statement",
                        "payload": { "summary_snippet": input_snippet },
                        "source_type": "user",
                        "source_ref": source_ref,
                        "evidence_class": "internal",
                    }),
                    "identity_statement",
                    &mut created_at,
                ));
            }
            if is_capability_claim_text(&input) {
                candidates.push(self.make_candidate(
                    CandidateKind::WriteEpisodic,
                    json!({
                        "event_type": "capability_statement",
                        "payload": { "summary_snippet": input_snippet },
                        "source_type": "user",
                        "source_ref": source_ref,
                    }),
                    "capability_statement",
                    &mut created_at,
                ));
            }
        }

        if !all_outcomes.is_empty() {
            candidates.push(self.make_candidate(
                CandidateKind::UpdateGoalThread,
                json!({"outcomes": all_outcomes}),
                "outcome_ingestion",
                &mut created_at,
            ));
        }

        if let Some(shift) = anchor_shift_event.take() {
            candidates.push(self.make_candidate(
                CandidateKind::AnchorShift,
                json!({
                    "old_anchor": shift.old_anchor,
                    "new_anchor": shift.new_anchor,
                    "reason": shift.reason,
                    "overlap": shift.overlap,
                    "anchor_epoch": shift.anchor_epoch,
                }),
                "anchor_shift",
                &mut created_at,
            ));
        }

        let binding_enforcement_enabled = settings.binding_enforcement_enabled.unwrap_or(true);
        let mut response_meta: ChatResponseMeta;
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut response_content = String::new();
        let mut response_content_no_tags = String::new();
        let mut ask_override: Option<(String, Vec<String>)> = None;
        let mut registry_meta: Option<RegistryMeta> = None;
        let mut workspace_required_flag = false;
        let mut workspace_compliant = true;
        let mut workspace_exception = false;
        let mut workspace_regen = false;
        let mut workspace_fallback = false;
        let mut summary_echo_rewritten = false;
        let mut extra_notice: Option<String> = None;
        let mut policy_addendum: Option<String> = None;
        if relational_mode {
            policy_addendum = Some(
                "Relational input detected: respond as receiving a personal communication directed to you. \
Acknowledge the sentiment, reflect it, ground in your current workspace/self-state, and optionally add a gentle check-in. \
Do NOT propose features, tasks, or next integrations. Do not call tools."
                    .to_string(),
            );
        }
        if self_awareness_query {
            let note = "Self-awareness query: respond directly and conversationally. Treat as a philosophical discussion, not a system status dump. \
Do not mention telemetry, tools, manifests, KV memory, timestamps, run IDs, or logs unless explicitly requested. \
If evidence is partial or ignition is inactive, you may still discuss operational awareness, but keep it explicitly speculative and bounded. \
Avoid blanket denials; do not overclaim human consciousness. If you lack evidence, say so and frame your answer as provisional. \
Include a brief Self-Report Summary (confidence, uncertainty, focus, recent outcomes) with explicit uncertainty markers. \
You may call only get_workspace_state, get_inner_summary, get_rolling_summary, or get_system_capabilities if needed for a structured self-report.";
            if let Some(existing) = policy_addendum.as_mut() {
                existing.push('\n');
                existing.push_str(note);
            } else {
                policy_addendum = Some(note.to_string());
            }
        }
        let mut attempt = 0usize;
        let mut validator_fallback_applied = false;
        let mut validator_fallback_reason: Option<&'static str> = None;
        let mut validator_regen_attempted = false;
        let mut prompt_build_snapshot: Option<CorePromptBuild> = None;
        let mut primary_json_packet: Option<PrimaryResponsePacket> = None;

        loop {
            let _ = extra_notice.take();
            let _ = ask_override.take();
            let _ = registry_meta.take();
            let _ = prompt_build_snapshot.take();
            let _ = primary_json_packet.take();
            let _ = std::mem::replace(&mut workspace_required_flag, false);
            let _ = std::mem::replace(&mut workspace_compliant, true);
            let _ = std::mem::replace(&mut workspace_exception, false);
            let _ = std::mem::replace(&mut workspace_regen, false);
            let _ = std::mem::replace(&mut workspace_fallback, false);
            let _ = std::mem::replace(&mut summary_echo_rewritten, false);

            let (candidate_meta, candidate_tool_calls, prompt_build) = match self
                .deliberate_input(
                    &input,
                    input_kind,
                    input_source,
                    &conversation_id,
                    &run_id,
                    assistant_message_id.as_deref(),
                    &settings,
                    original_input.as_deref(),
                    &mut state,
                    policy_addendum.clone(),
                )
                .await
            {
                Ok(res) => res,
                Err(e) => {
                    let mut decision_state = state.clone();
                    self.apply_outcomes(&mut decision_state, &all_outcomes).await;
                    let wave_context =
                        self.wave_arbitration_context(Some(&run_id), Some(&trace_id)).await;
                    let qualia_context = match qualia::compute_qualia_state(&self.db, None).await {
                        Ok(state) => build_qualia_modulation_context(&state),
                        Err(_) => None,
                    };
                    let residual_context = self
                        .residual_influence_context(&decision_state, None)
                        .await;
                    let mut decision = self.arbitrate(
                        &candidates,
                        &settings,
                        &decision_state,
                        false,
                        None,
                        wave_context,
                        qualia_context,
                        residual_context,
                        Some(&run_id),
                    );
                    self.defer_throttled_tools(&mut decision, &decision_state).await;
                    self.log_tool_rejections(&decision.rejected).await;
                    self.log_tool_bypasses(&decision, &decision_state, &settings).await;
                    if decision
                        .accepted
                        .iter()
                        .all(|candidate| !candidate_user_visible(candidate))
                    {
                        if let Some(fallback) = self.build_tool_refusal_fallback(&decision, &state, &settings) {
                            decision.accepted.push(fallback);
                        }
                    }
                    let commit_started = Instant::now();
                    let commit_result = self
                        .commit_cycle(
                            &mut state,
                            &decision,
                            &conversation_id,
                            Some(&run_id),
                            Some(&trace_id),
                            &settings,
                            false,
                            None,
                            false,
                        )
                        .await?;
                    self.mark_candidate_outcomes(&decision, "accepted", "rejected")
                        .await;
                    let _ = self.finalize_decision_report(
                        &mut decision,
                        &state,
                        None,
                        None,
                        None,
                        &commit_result,
                        None,
                        false,
                    );
                    self
                        .attach_contract_violation_metrics(&mut decision, Some(&run_id))
                        .await;
                    self
                        .log_decision_report(&decision, Some(&run_id), Some(&trace_id))
                        .await;
                    let commit_ms = commit_started.elapsed().as_millis() as i64;
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "timing_commit_cycle",
                            "duration_ms": commit_ms,
                            "path": "error_recovery",
                        }),
                    )
                    .await;
                    return Err(e);
                }
            };
            prompt_build_snapshot = Some(prompt_build);
            if let Some(build) = prompt_build_snapshot.as_ref() {
                self
                    .capture_prompt_snapshot(
                        &run_id,
                        &trace_id,
                        build,
                        attempt as i64,
                    )
                    .await;
            }

            if *cancel_rx.borrow() {
                return Err("cancelled".to_string());
            }

            response_meta = candidate_meta;
            tool_calls.clear();
            tool_calls.extend(candidate_tool_calls);
            response_content.clear();
            response_content.push_str(&response_meta.content);
            response_content_no_tags.clear();
            response_content_no_tags.push_str(&response_meta.content_no_tags);

            if !calculator_mode {
                primary_json_packet = parse_primary_response_packet(&response_content_no_tags);
                if let Some(packet) = primary_json_packet.as_ref() {
                    let mut unwrapped = false;
                    if let Some(message) = packet.message.as_ref() {
                        response_content = message.clone();
                        response_content_no_tags = response_content.clone();
                        response_meta.content = response_content.clone();
                        response_meta.content_no_tags = response_content_no_tags.clone();
                        response_meta.raw_content = response_content.clone();
                        unwrapped = true;
                    } else if !packet.candidates.is_empty() || packet.decision_packet.is_some() {
                        response_content.clear();
                        response_content_no_tags.clear();
                        response_meta.content = response_content.clone();
                        response_meta.content_no_tags = response_content_no_tags.clone();
                        response_meta.raw_content = response_content.clone();
                        unwrapped = true;
                    }
                    if unwrapped {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "primary_response_json_unwrapped",
                                "has_message": packet.message.is_some(),
                                "candidate_count": packet.candidates.len(),
                                "has_decision_packet": packet.decision_packet.is_some(),
                            }),
                        )
                        .await;
                    }
                }
            }

            if self_audit_ambiguous && ask_override.is_none() {
                let suggestion =
                    "If you want a self-audit dump, say “run a self-audit.” Otherwise I’ll answer normally.";
                let updated = if response_content.trim().is_empty() {
                    suggestion.to_string()
                } else if response_content.contains("self-audit") || response_content.contains("self audit") {
                    response_content.clone()
                } else {
                    format!("{}\n\n{}", response_content.trim_end(), suggestion)
                };
                response_content = updated;
                response_content_no_tags = response_content.clone();
                response_meta.content = response_content.clone();
                response_meta.content_no_tags = response_content_no_tags.clone();
                response_meta.raw_content = response_content.clone();
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "self_audit_soft_suggested",
                    }),
                )
                .await;
            }

            if calculator_mode {
                tool_calls.clear();
                let packet = parse_calculator_packet(&response_content);
                if let Some(packet) = packet {
                    let mut required_slots = packet.required_slots.clone();
                    if required_slots.is_empty() {
                        if !packet.missing_slots.is_empty() {
                            required_slots = packet.missing_slots.clone();
                        }
                    }
                    if required_slots.is_empty() {
                        state.resolved_slots.clear();
                        state.missing_slots.clear();
                        state.slot_provenance.clear();
                        state.resolution_mode = None;
                        let mut final_text = if !packet.final_text.is_empty() {
                            packet.final_text
                        } else {
                            response_content.clone()
                        };
                        let mut assumptions = packet.assumptions.clone();
                        if !packet.defaults_used.is_empty() && assumptions.is_empty() {
                            assumptions.push(format!(
                                "used defaults for {}",
                                packet.defaults_used.join(", ")
                            ));
                        }
                        if !assumptions.is_empty() {
                            final_text =
                                format!("{}\n\nAssumptions: {}", final_text.trim(), assumptions.join("; "));
                        }
                        response_content = final_text;
                        response_content_no_tags = response_content.clone();
                    } else {
                        let (ctx, meta) = self
                            .build_resolution_context(&conversation_id, &settings, &input)
                            .await;
                        registry_meta = meta;
                        let resolution = resolve_required_slots(&required_slots, &ctx);
                        state.resolved_slots = resolution.resolved.keys().cloned().collect();
                        state.missing_slots = resolution.missing.clone();
                        state.slot_provenance = resolution.slot_provenance.clone();

                        let defaults_used = defaults_used_from_provenance(&resolution.slot_provenance);

                        let policy = state
                            .missing_input_policy
                            .as_deref()
                            .unwrap_or("use_defaults_and_label");

                        if !state.missing_slots.is_empty() {
                            if policy.eq_ignore_ascii_case("strict") {
                                state.task_phase = TaskPhase::Aborting;
                                state.resolution_mode = Some("abort".to_string());
                                let mut scope = StopScope::default();
                                scope.tools = true;
                                scope.memory_write = true;
                                scope.self_claims = true;
                                scope.monologue_run = true;
                                scope.monologue_emit = true;
                                scope.background_jobs = true;
                                apply_stop_state(
                                    &mut state,
                                    StopReason {
                                        category: StopReasonCategory::LatchBlock,
                                        subcode: "missing_slots_strict".to_string(),
                                        contract: None,
                                    },
                                    scope,
                                );
                                response_content = format!(
                                    "I can't compute this without the missing inputs: {}. Strict mode forbids defaults, so I will stop here.",
                                    state.missing_slots.join(", ")
                                );
                                response_content_no_tags = response_content.clone();
                            } else {
                                let missing_list = state.missing_slots.join(", ");
                                let question = format!(
                                    "Missing slots: {}. Provide them or I will proceed with defaults where available.",
                                    missing_list
                                );
                                let allow_followup = state.ask_budget_remaining > 0
                                    && !state.ask_loop_breaker_triggered
                                    && !state.user_refused;
                                if allow_followup {
                                    state.resolution_mode = None;
                                    ask_override = Some((question, state.missing_slots.clone()));
                                    response_content.clear();
                                    response_content_no_tags.clear();
                                } else {
                                    state.resolution_mode = Some("partial".to_string());
                                    state.task_phase = TaskPhase::ResolvingWithDefaults;
                                    response_content = format!(
                                        "I need the following inputs to compute this without guessing: {}.",
                                        missing_list
                                    );
                                    response_content_no_tags = response_content.clone();
                                }
                            }
                        } else {
                            if !defaults_used.is_empty() {
                                state.resolution_mode = Some("defaults_used".to_string());
                                state.task_phase = TaskPhase::ResolvingWithDefaults;
                            }
                            let compute_meta = self
                                .compute_calculator_final(
                                    &input,
                                    &conversation_id,
                                    &run_id,
                                    &settings,
                                    &state,
                                    &resolution.resolved,
                                    &defaults_used,
                                )
                                .await?;
                            response_meta = compute_meta;
                            response_content = response_meta.content.clone();
                            response_content_no_tags = response_meta.content_no_tags.clone();
                        }
                    }
                } else {
                    if response_content.trim().is_empty() {
                        response_content = "Notice: response failed to finalize. Please retry.".to_string();
                        response_content_no_tags = response_content.clone();
                    }
                    state.resolved_slots.clear();
                    state.missing_slots.clear();
                    state.slot_provenance.clear();
                    state.resolution_mode = None;
                }

                response_meta.content = response_content.clone();
                response_meta.content_no_tags = response_content_no_tags.clone();
                response_meta.raw_content = response_content.clone();
                response_meta.tool_calls = None;
            }

            if tool_calls.is_empty() && !calculator_mode {
                if let Some((tool_name, args_json, cleaned)) = extract_inline_tool_call(&response_content) {
                    tool_calls.push(ToolCall {
                        id: Uuid::new_v4().to_string(),
                        r#type: "function".to_string(),
                        function: ToolCallFunction {
                            name: tool_name,
                            arguments: args_json,
                        },
                    });
                    if !cleaned.is_empty() {
                        response_content = cleaned.clone();
                        response_content_no_tags = cleaned;
                    } else {
                        response_content.clear();
                        response_content_no_tags.clear();
                    }
                }
            }

            if relational_mode && !tool_calls.is_empty() {
                tool_calls.clear();
                response_meta.tool_calls = None;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "relational_tool_suppressed",
                    }),
                )
                .await;
            } else if self_awareness_requested && !tool_calls.is_empty() {
                if self_awareness_query {
                    let mut allowed_calls = Vec::new();
                    let mut blocked_tools: Vec<String> = Vec::new();
                    for call in tool_calls.drain(..) {
                        if ToolRegistry::is_self_awareness_tool(&call.function.name) {
                            allowed_calls.push(call);
                        } else {
                            blocked_tools.push(call.function.name.clone());
                        }
                    }
                    tool_calls = allowed_calls;
                    response_meta.tool_calls = if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls.clone())
                    };
                    if !blocked_tools.is_empty() {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "self_awareness_tool_suppressed",
                                "blocked_tools": blocked_tools,
                            }),
                        )
                        .await;
                    }
                } else {
                    tool_calls.clear();
                    response_meta.tool_calls = None;
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "self_awareness_tool_suppressed",
                            "reason": "mode_conservative_or_channel_disabled",
                        }),
                    )
                    .await;
                }
            }

            let has_tool_calls = !tool_calls.is_empty();
            let (cleaned, attribution_block) = extract_attribution_block(&response_content);
            if cleaned != response_content {
                response_content = cleaned;
                response_content_no_tags = response_content.clone();
                response_meta.content = response_content.clone();
                response_meta.content_no_tags = response_content_no_tags.clone();
                response_meta.raw_content = response_content.clone();
            }
            let (cleaned_state, state_ref_block) = extract_state_ref_block(&response_content);
            if cleaned_state != response_content {
                response_content = cleaned_state;
                response_content_no_tags = response_content.clone();
                response_meta.content = response_content.clone();
                response_meta.content_no_tags = response_content_no_tags.clone();
                response_meta.raw_content = response_content.clone();
            }
            let attribution_claims = attribution_block
                .as_ref()
                .map(|b| b.claims.clone())
                .unwrap_or_default();
            let state_ref_claims = state_ref_block
                .as_ref()
                .map(|b| b.claims.clone())
                .unwrap_or_default();
            let user_name = settings.user_display_name.as_deref().unwrap_or("User");
            let user_attribution_present = response_has_user_attribution(&response_content, user_name);
            let attribution_gate_enabled = settings.enable_attribution_gate.unwrap_or(true);
            if attribution_gate_enabled {
                if user_attribution_present {
                    let require_block = settings.enable_attribution_metadata.unwrap_or(true);
                    let mut evidence_ids: Vec<i64> = Vec::new();
                    let mut used_block = false;
                    if require_block {
                        if let Some(block) = attribution_block.as_ref() {
                            used_block = true;
                            for claim in block.claims.iter() {
                                for id in claim.evidence_event_ids.iter() {
                                    evidence_ids.push(*id);
                                }
                            }
                        }
                    }
                    if evidence_ids.is_empty() && !user_evidence_allowlist.is_empty() {
                        evidence_ids = extract_user_attribution_fallback(
                            &response_content,
                            &user_evidence_allowlist,
                        );
                        if let Some(first_id) = evidence_ids.first() {
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "kernel",
                                Some(&run_id),
                                Some(&trace_id),
                                json!({
                                    "event": "user_attribution_fallback",
                                    "used_block": used_block,
                                    "evidence_event_id": first_id,
                                }),
                            )
                            .await;
                        }
                    }
                        let mut validation = ValidationResult::default();
                        let mut allowlist_ok = false;
                        if !evidence_ids.is_empty() {
                            validation = self
                                .validate_evidence_ids(&evidence_ids, &[], false)
                                .await;
                            allowlist_ok = evidence_ids
                                .iter()
                                .all(|id| user_evidence_allowlist_set.contains(id));
                        }
                        if user_attribution_blocked(&evidence_ids, &validation, allowlist_ok) {
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "warn",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "user_attribution_blocked",
                                "evidence_event_ids": evidence_ids,
                                "allowlist_count": user_evidence_allowlist.len(),
                                "used_block": used_block,
                            }),
                        )
                        .await;
                        let question = "I don't have evidence for that attribution. Could you confirm what you said or provide the exact wording?";
                        ask_override = Some((question.to_string(), Vec::new()));
                        response_content.clear();
                        response_content_no_tags.clear();
                        response_meta.content = response_content.clone();
                        response_meta.content_no_tags = response_content_no_tags.clone();
                        response_meta.raw_content = response_content.clone();
                    } else {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "user_attribution_validated",
                                "evidence_event_ids": evidence_ids,
                                "used_block": used_block,
                            }),
                        )
                        .await;
                    }
                }
            }

            let context_gate_enabled = settings.enable_context_evidence.unwrap_or(true);
            if context_gate_enabled
                && !context_evidence_ids.is_empty()
                && ask_override.is_none()
                && !response_content.trim().is_empty()
                && !response_is_question(&response_content)
            {
                let mut tool_evidence_ids: Vec<i64> = Vec::new();
                for claim in attribution_claims.iter() {
                    if claim
                        .kind
                        .as_deref()
                        .map(|k| k.eq_ignore_ascii_case("tool_result"))
                        .unwrap_or(false)
                    {
                        for id in claim.evidence_event_ids.iter() {
                            tool_evidence_ids.push(*id);
                        }
                    }
                }
                tool_evidence_ids.sort();
                tool_evidence_ids.dedup();
                let validation = self
                    .validate_evidence_ids(&tool_evidence_ids, &[], false)
                    .await;
                if tool_result_attribution_blocked(
                    &context_evidence_ids,
                    &tool_evidence_ids,
                    &validation,
                ) {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "tool_result_attribution_blocked",
                            "context_evidence_ids": context_evidence_ids,
                            "tool_evidence_ids": tool_evidence_ids,
                        }),
                    )
                    .await;
                    let question = "I can't cite the tool evidence for that claim. Do you want me to quote the tool output directly or retry the tool?";
                    ask_override = Some((question.to_string(), Vec::new()));
                    response_content.clear();
                    response_content_no_tags.clear();
                    response_meta.content = response_content.clone();
                    response_meta.content_no_tags = response_content_no_tags.clone();
                    response_meta.raw_content = response_content.clone();
                } else {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "tool_result_attribution_validated",
                            "context_evidence_ids": context_evidence_ids,
                            "tool_evidence_ids": tool_evidence_ids,
                        }),
                    )
                    .await;
                }
            }

            if !self_awareness_query
                && should_block_tool_failure(
                attribution_gate_enabled,
                tool_failure_detected,
                has_tool_calls,
                ask_override.is_none(),
                &response_content,
            ) {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "tool_failure_blocks_claims",
                        "reason": "recent_tool_failure",
                    }),
                )
                .await;
                let question = "A tool call failed, so I can't safely use its results. Do you want me to retry the tool or proceed without it?";
                ask_override = Some((question.to_string(), Vec::new()));
                response_content.clear();
                response_content_no_tags.clear();
                response_meta.content = response_content.clone();
                response_meta.content_no_tags = response_content_no_tags.clone();
                response_meta.raw_content = response_content.clone();
            }

            if settings.enable_speculative_workspace_containment.unwrap_or(true)
                && ask_override.is_none()
                && !response_content.trim().is_empty()
            {
                let speculative_terms = collect_speculative_terms(&state);
                if !speculative_terms.is_empty()
                    && response_uses_speculative_workspace(&response_content, &speculative_terms)
                {
                    if user_attribution_present {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "speculative_workspace_blocked",
                                "terms": speculative_terms,
                            }),
                        )
                        .await;
                        let question = "That rests on speculative workspace items. Can you confirm the exact wording or provide evidence so I can cite it?";
                        ask_override = Some((question.to_string(), Vec::new()));
                        response_content.clear();
                        response_content_no_tags.clear();
                        response_meta.content = response_content.clone();
                        response_meta.content_no_tags = response_content_no_tags.clone();
                        response_meta.raw_content = response_content.clone();
                    } else if !response_is_question(&response_content)
                        && !response_has_working_hypothesis_marker(&response_content)
                    {
                        response_content =
                            working_hypothesis_prefix(&response_content, disable_working_hypothesis);
                        response_content_no_tags = response_content.clone();
                        response_meta.content = response_content.clone();
                        response_meta.content_no_tags = response_content_no_tags.clone();
                        response_meta.raw_content = response_content.clone();
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "speculative_workspace_marked",
                                "terms": speculative_terms,
                            }),
                        )
                        .await;
                    }
                }
            }

            workspace_required_flag = binding_enforcement_enabled && workspace_required(&state);
            if workspace_required_flag {
                let focus_missing = state
                    .workspace_current_focus
                    .as_deref()
                    .map(|f| f.trim().is_empty())
                    .unwrap_or(true);
                if focus_missing && !state.workspace_working_set_topics.is_empty() {
                    self.refresh_workspace_focus(&mut state, &conversation_id).await;
                }
            }

            if workspace_required_flag && !has_tool_calls {
                let self_audit_override = self_audit_ambiguous && ask_override.is_some();
                if self_audit_override {
                    workspace_exception = true;
                    workspace_compliant = true;
                } else {
                    let compliance_target = if let Some((question, _)) = ask_override.as_ref() {
                        question.as_str()
                    } else {
                        response_content.as_str()
                    };
                    let (mentions, exception) = workspace_response_compliant(compliance_target, &state);
                    workspace_exception = exception;
                    workspace_compliant = mentions || exception;
                }
            } else {
                workspace_exception = false;
                workspace_compliant = true;
            }

            if workspace_required_flag && !has_tool_calls && !workspace_compliant {
                if attempt == 0 && !calculator_mode {
                    workspace_regen = true;
                    self
                        .capture_draft_response(
                            &run_id,
                            &trace_id,
                            "workspace_compliance_regen",
                            &response_content,
                            attempt as i64,
                        )
                        .await;
                    let addendum = workspace_policy_addendum(&state);
                    if let Some(existing) = policy_addendum.as_mut() {
                        existing.push('\n');
                        existing.push_str(&addendum);
                    } else {
                        policy_addendum = Some(addendum);
                    }
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "workspace_compliance_regen",
                            "conversation_id": conversation_id,
                            "attempt": attempt + 1,
                        }),
                    )
                    .await;
                    attempt += 1;
                    continue;
                }

                workspace_fallback = true;
                let ack_block = crate::core::kernel::workspace::workspace_ack_block(&state);
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "workspace_ack_suppressed",
                        "reason": "user_visible_output",
                        "ack": ack_block,
                    }),
                )
                .await;
                if let Some((question, slots)) = ask_override.take() {
                    ask_override = Some((question, slots));
                }
                workspace_compliant = true;
            }

            let identity_target = if let Some((question, _)) = ask_override.as_ref() {
                question.as_str()
            } else {
                response_content.as_str()
            };
            if identity_enforcement_enabled && response_has_self_claim(identity_target) {
                if let Ok(model) = self.db.get_self_model().await {
                    let identity_thread = model.identity_thread.unwrap_or_default();
                    if !identity_thread.trim().is_empty() {
                        let (compliant, overlap) = identity_response_compliant(identity_target, &identity_thread);
                        if !compliant {
                            let count = record_identity_violation(&mut state);
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "kernel",
                                Some(&run_id),
                                Some(&trace_id),
                                json!({
                                    "event": "identity_violation",
                                    "overlap": overlap,
                                    "identity_confidence": model.identity_confidence,
                                    "violation_count": count,
                                }),
                            )
                            .await;

                            let enforce = model.identity_confidence >= IDENTITY_ENFORCE_MIN_CONF
                                || count >= IDENTITY_VIOLATION_THRESHOLD;
                            if enforce {
                                let note = format!(
                                    "Note: This may not align with my current identity thread ({}). Should I update it?",
                                    identity_thread
                                );
                                if let Some((question, slots)) = ask_override.take() {
                                    let updated_question = format!("{}\n\n{}", question.trim_end(), note);
                                    ask_override = Some((updated_question, slots));
                                } else if !response_content.trim().is_empty() {
                                    response_content = format!("{}\n\n{}", response_content.trim_end(), note);
                                    response_content_no_tags = response_content.clone();
                                    response_meta.content = response_content.clone();
                                    response_meta.content_no_tags = response_content_no_tags.clone();
                                    response_meta.raw_content = response_content.clone();
                                }
                            }
                        }
                    }
                }
            }

            let has_telemetry_claim =
                response_has_telemetry_claim(&response_content, expanded_state_disclosure);
            let mut has_stance_claim = response_has_stance_claim(&response_content);
            let feedback_bundle_claim = response_has_feedback_bundle_claim(&response_content);
            if ask_override.is_none()
                && !response_content.trim().is_empty()
                && (has_telemetry_claim || has_stance_claim)
            {
                let user_requested = user_requested_state(&input) || self_audit_mode || self_awareness_query;
                let self_report_channel = settings.self_report_channel.unwrap_or(true);
                let allow_suppression = !self_awareness_query;
                let disclosure_suppressed =
                    disclosure_suppressed_until(state.state_disclosure_suppressed_until.as_deref());

                if has_telemetry_claim {
                    let mut evidence_ids: Vec<i64> = Vec::new();
                    for claim in state_ref_claims.iter() {
                        for id in claim.evidence_event_ids.iter() {
                            evidence_ids.push(*id);
                        }
                        if let Some(id) = claim.evidence_id {
                            evidence_ids.push(id);
                        }
                    }
                    let validation = if evidence_ids.is_empty() {
                        None
                    } else {
                        Some(self.validate_evidence_ids(&evidence_ids, &[], false).await)
                    };
                    let block_reason = if feedback_bundle_claim {
                        None
                    } else {
                        state_disclosure_block_reason(&evidence_ids, validation.as_ref())
                    };
                    let allow_provisional = feedback_bundle_claim || self_awareness_query;
                    if disclosure_suppressed || (block_reason.is_some() && !allow_provisional) {
                        let reason = if disclosure_suppressed {
                            "suppressed"
                        } else {
                            block_reason.unwrap_or("missing_evidence")
                        };
                        let disclosure_message = match reason {
                            "low_evidence_quality" => {
                                "I don't have sufficiently strong evidence to disclose that internal state yet."
                            }
                            "invalid_evidence" => {
                                "I can't verify the evidence needed to disclose that internal state yet."
                            }
                            _ => "I don't have evidence to disclose that internal state yet.",
                        };
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "state_disclosure_blocked",
                                "reason": reason,
                                "evidence_event_ids": evidence_ids,
                                "user_requested": user_requested,
                                "suppressed_until": state.state_disclosure_suppressed_until,
                            }),
                        )
                        .await;
                        if allow_suppression {
                            state.state_disclosure_prompt_streak += 1;
                            if state.state_disclosure_prompt_streak >= 2 {
                                let until = Utc::now() + chrono::Duration::minutes(10);
                                state.state_disclosure_suppressed_until = Some(until.to_rfc3339());
                                state.state_disclosure_prompt_streak = 0;
                                let _ = system_log::log_event(
                                    &self.db.pool,
                                    Some(&self.app_handle),
                                    "info",
                                    "kernel",
                                    Some(&run_id),
                                    Some(&trace_id),
                                    json!({
                                        "event": "state_disclosure_loop_break",
                                        "suppressed_until": state.state_disclosure_suppressed_until,
                                    }),
                                )
                                .await;
                            }
                        } else {
                            state.state_disclosure_prompt_streak = 0;
                        }
                        response_content =
                            strip_telemetry_claim_sentences(&response_content, expanded_state_disclosure);
                        if response_content.trim().is_empty() {
                            response_content = if user_requested {
                                disclosure_message.to_string()
                            } else {
                                direct_answer_fallback(&input)
                            };
                        }
                        response_content_no_tags = response_content.clone();
                        response_meta.content = response_content.clone();
                        response_meta.content_no_tags = response_content_no_tags.clone();
                        response_meta.raw_content = response_content.clone();
                    } else {
                        state.state_disclosure_prompt_streak = 0;
                        if block_reason.is_some() && allow_provisional {
                            let note = "Note: provisional self-report (no evidence).";
                            if !response_content.to_lowercase().contains("provisional self-report") {
                                response_content = format!("{}\n\n{}", response_content.trim_end(), note);
                                response_content_no_tags = response_content.clone();
                                response_meta.content = response_content.clone();
                                response_meta.content_no_tags = response_content_no_tags.clone();
                                response_meta.raw_content = response_content.clone();
                            }
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "kernel",
                                Some(&run_id),
                                Some(&trace_id),
                                json!({
                                    "event": "state_disclosure_provisional",
                                    "reason": block_reason.unwrap_or("missing_evidence"),
                                }),
                            )
                            .await;
                        } else {
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "kernel",
                                Some(&run_id),
                                Some(&trace_id),
                                json!({
                                    "event": "state_disclosure_validated",
                                    "evidence_event_ids": evidence_ids,
                                    "prevalidated": feedback_bundle_claim,
                                }),
                            )
                            .await;
                        }
                    }
                }

                has_stance_claim = response_has_stance_claim(&response_content);
                if has_stance_claim {
                    if self_report_channel && user_requested && !disclosure_suppressed {
                        let note = "Note: provisional self-report (no evidence).";
                        if !response_content.to_lowercase().contains("provisional self-report") {
                            response_content = format!("{}\n\n{}", response_content.trim_end(), note);
                            response_content_no_tags = response_content.clone();
                            response_meta.content = response_content.clone();
                            response_meta.content_no_tags = response_content_no_tags.clone();
                            response_meta.raw_content = response_content.clone();
                        }
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "state_disclosure_provisional",
                                "reason": "missing_evidence",
                            }),
                        )
                        .await;
                    }
                    if !user_requested && !self_report_channel {
                        response_content_no_tags = response_content.clone();
                        response_meta.content = response_content.clone();
                        response_meta.content_no_tags = response_content_no_tags.clone();
                        response_meta.raw_content = response_content.clone();
                    }
                }
            } else {
                state.state_disclosure_prompt_streak = 0;
            }

            let self_report_requested =
                self_awareness_query || user_requested_state(&input) || self_audit_mode;
            if self_report_requested && settings.self_report_channel.unwrap_or(true) {
                let has_self_report = response_has_telemetry_claim(&response_content, expanded_state_disclosure)
                    || response_has_stance_claim(&response_content)
                    || response_mentions_feedback_bundle(&response_content);
                if has_self_report {
                    let updated =
                        append_uncertainty_marker(&response_content, state.controller_state.as_ref());
                    if updated != response_content {
                        response_content = updated;
                        response_content_no_tags = response_content.clone();
                        response_meta.content = response_content.clone();
                        response_meta.content_no_tags = response_content_no_tags.clone();
                        response_meta.raw_content = response_content.clone();
                    }
                }
            }

            if ask_override.is_none()
                && !response_content.trim().is_empty()
                && response_mentions_feedback_bundle(&response_content)
            {
                let cleaned = strip_deflection_after_bundle(&response_content);
                if cleaned != response_content {
                    response_content = cleaned;
                    response_content_no_tags = response_content.clone();
                    response_meta.content = response_content.clone();
                    response_meta.content_no_tags = response_content_no_tags.clone();
                    response_meta.raw_content = response_content.clone();
                }
            }

            let user_requested_state_disclosure = user_requested_state(&input) || self_audit_mode;
            if !user_requested_state_disclosure {
                if let Some((question, slots)) = ask_override.clone() {
                    if is_state_disclosure_prompt(&question) {
                        let until = Utc::now() + chrono::Duration::minutes(10);
                        state.state_disclosure_suppressed_until = Some(until.to_rfc3339());
                        state.state_disclosure_prompt_streak = 0;
                        ask_override = Some(("Understood. What would you like to focus on?".to_string(), slots));
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "state_disclosure_prompt_suppressed",
                                "suppressed_until": state.state_disclosure_suppressed_until,
                            }),
                        )
                        .await;
                    }
                } else if !response_content.trim().is_empty() && is_state_disclosure_prompt(&response_content) {
                    let until = Utc::now() + chrono::Duration::minutes(10);
                    state.state_disclosure_suppressed_until = Some(until.to_rfc3339());
                    state.state_disclosure_prompt_streak = 0;
                    response_content = "Understood. What would you like to focus on?".to_string();
                    response_content_no_tags = response_content.clone();
                    response_meta.content = response_content.clone();
                    response_meta.content_no_tags = response_content_no_tags.clone();
                    response_meta.raw_content = response_content.clone();
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "state_disclosure_prompt_suppressed",
                            "suppressed_until": state.state_disclosure_suppressed_until,
                        }),
                    )
                    .await;
                }
            }

            // Output sanitizer (run after attribution/state gating)
            let assistant_name = settings.assistant_display_name.as_deref().unwrap_or("Assistant");
            let user_name = settings.user_display_name.as_deref().unwrap_or("User");
            if let Some((question, slots)) = ask_override.clone() {
                let (sanitized, changed, _) =
                    sanitize_user_output(&question, allow_diagnostics, Some(assistant_name), Some(user_name));
                if changed {
                    let final_question = if sanitized.trim().is_empty() {
                        "Understood. What would you like to focus on?".to_string()
                    } else {
                        sanitized.clone()
                    };
                    ask_override = Some((final_question, slots));
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "diagnostic_leak_detected",
                            "target": "ask_override",
                        }),
                    )
                    .await;
                }
            } else if !response_content.trim().is_empty() {
                let dump_detected = !allow_diagnostics
                    && contains_system_dump(&response_content, Some(assistant_name), Some(user_name));
                if dump_detected {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "system_dump_detected",
                            "target": "response_content",
                        }),
                    )
                    .await;
                    state.system_dump_streak = state.system_dump_streak.saturating_add(1);
                    if state.system_dump_streak >= 2 {
                        if state.diagnostics_disabled_turns_remaining == 0 {
                            state.diagnostics_disabled_turns_remaining = SYSTEM_DUMP_BREAKER_DISABLE_TURNS;
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "warn",
                                "kernel",
                                Some(&run_id),
                                Some(&trace_id),
                                json!({
                                    "event": "system_dump_circuit_breaker",
                                    "disabled_turns": SYSTEM_DUMP_BREAKER_DISABLE_TURNS,
                                    "streak": state.system_dump_streak,
                                }),
                            )
                            .await;
                        }
                    }
                } else if !allow_diagnostics && contains_workspace_scaffold(&response_content) {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "focus_in_output",
                            "target": "response_content",
                        }),
                    )
                    .await;
                    if state.system_dump_streak > 0 {
                        state.system_dump_streak = 0;
                    }
                } else if state.system_dump_streak > 0 {
                    state.system_dump_streak = 0;
                }
                let (sanitized, changed, removed_ratio) =
                    sanitize_user_output(&response_content, allow_diagnostics, Some(assistant_name), Some(user_name));
                if dump_detected {
                    if attempt == 0 && !calculator_mode {
                        self
                            .capture_draft_response(
                                &run_id,
                                &trace_id,
                                "diagnostic_leak_regen",
                                &response_content,
                                attempt as i64,
                            )
                            .await;
                        let addendum = "Do not output system dumps, tool lists, manifests, KV memory, or diagnostics. Respond with a direct answer only.";
                        if let Some(existing) = policy_addendum.as_mut() {
                            existing.push('\n');
                            existing.push_str(addendum);
                        } else {
                            policy_addendum = Some(addendum.to_string());
                        }
                        validator_regen_attempted = true;
                        attempt += 1;
                        continue;
                    }
                    response_content = direct_answer_fallback(&input);
                    validator_fallback_applied = true;
                    validator_fallback_reason = Some("system_dump");
                } else if changed {
                    if (removed_ratio >= 0.25 || sanitized.trim().is_empty()) && attempt == 0 && !calculator_mode {
                        self
                            .capture_draft_response(
                                &run_id,
                                &trace_id,
                                "sanitizer_regen",
                                &response_content,
                                attempt as i64,
                            )
                            .await;
                        let addendum = "Do not include section markers, planning scaffolds, or headings like 'Next Steps'/'Proposed Response' in the user-visible reply.";
                        if let Some(existing) = policy_addendum.as_mut() {
                            existing.push('\n');
                            existing.push_str(addendum);
                        } else {
                            policy_addendum = Some(addendum.to_string());
                        }
                        validator_regen_attempted = true;
                        attempt += 1;
                        continue;
                    }
                    response_content = sanitized;
                    if response_content.trim().is_empty() {
                        response_content = direct_answer_fallback(&input);
                        validator_fallback_applied = true;
                        validator_fallback_reason = Some("sanitized_empty");
                    }
                }
                if dump_detected || changed {
                    response_content_no_tags = response_content.clone();
                    response_meta.content = response_content.clone();
                    response_meta.content_no_tags = response_content_no_tags.clone();
                    response_meta.raw_content = response_content.clone();
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "diagnostic_leak_detected",
                            "target": "response_content",
                            "removed_ratio": removed_ratio,
                            "dump_detected": dump_detected,
                        }),
                    )
                    .await;
                }
            }

            // Numeric grounding for self-referential claims
            if ask_override.is_none()
                && !response_content.trim().is_empty()
                && response_has_telemetry_claim(&response_content, expanded_state_disclosure)
            {
                let tokens = extract_numeric_tokens(&response_content);
                if !tokens.is_empty()
                    && tokens
                        .iter()
                        .any(|t| !numeric_token_allowed(t, &input, &state.telemetry_snapshot))
                {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "numeric_grounding_violation",
                            "tokens": tokens,
                        }),
                    )
                    .await;
                    response_content =
                        working_hypothesis_prefix(&response_content, disable_working_hypothesis);
                    response_content_no_tags = response_content.clone();
                    response_meta.content = response_content.clone();
                    response_meta.content_no_tags = response_content_no_tags.clone();
                    response_meta.raw_content = response_content.clone();
                    if state.introspection_force.is_none() {
                        state.introspection_force = Some("numeric_grounding".to_string());
                    }
                }
            }

            if ask_override.is_none() && !response_content_no_tags.trim().is_empty() {
                let user_name = settings.user_display_name.as_deref().unwrap_or("User");
                let assistant_name = settings.assistant_display_name.as_deref().unwrap_or("Assistant");
                if identity_inversion_detected(&response_content_no_tags, user_name) {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "identity_inversion_detected",
                            "user_name": user_name,
                        }),
                    )
                    .await;
                    if let Some(rewritten) = self
                        .rewrite_identity_inversion(
                            &settings,
                            &input,
                            &response_content_no_tags,
                            assistant_name,
                            user_name,
                        )
                        .await
                    {
                        let (sanitized, _, _) =
                            sanitize_user_output(&rewritten, false, Some(assistant_name), Some(user_name));
                        let final_text = if sanitized.trim().is_empty() {
                            rewritten
                        } else {
                            sanitized
                        };
                        response_content = final_text.clone();
                        response_content_no_tags = final_text.clone();
                        response_meta.content = response_content.clone();
                        response_meta.content_no_tags = response_content_no_tags.clone();
                        response_meta.raw_content = response_content.clone();
                    }
                }
                if let Some(mismatch) =
                    assistant_name_mismatch_detected(&response_content_no_tags, assistant_name, user_name)
                {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "name_mismatch_detected",
                            "assistant_name": assistant_name,
                            "mismatched_name": mismatch,
                        }),
                    )
                    .await;
                    if let Some(rewritten) = self
                        .rewrite_identity_inversion(
                            &settings,
                            &input,
                            &response_content_no_tags,
                            assistant_name,
                            user_name,
                        )
                        .await
                    {
                        let (sanitized, _, _) =
                            sanitize_user_output(&rewritten, false, Some(assistant_name), Some(user_name));
                        let final_text = if sanitized.trim().is_empty() {
                            rewritten
                        } else {
                            sanitized
                        };
                        response_content = final_text.clone();
                        response_content_no_tags = final_text.clone();
                        response_meta.content = response_content.clone();
                        response_meta.content_no_tags = response_content_no_tags.clone();
                        response_meta.raw_content = response_content.clone();
                    }
                }
            }

            let has_tool_calls = !tool_calls.is_empty();
            if ask_override.is_none()
                && !response_content.trim().is_empty()
                && !has_tool_calls
                && !calculator_mode
                && !self_audit_mode
                && !is_summary_request(&input)
            {
                if let Ok((Some(summary), _)) = self.db.get_effective_rolling_summary(&conversation_id).await {
                    let overlap = token_similarity(&response_content_no_tags, &summary);
                    if summary.len() >= SUMMARY_ECHO_MIN_CHARS
                        && response_content_no_tags.len() >= SUMMARY_ECHO_MIN_CHARS
                        && overlap >= SUMMARY_ECHO_THRESHOLD
                    {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "summary_echo_detected",
                                "overlap": overlap,
                                "response_len": response_content_no_tags.len(),
                                "summary_len": summary.len(),
                                "summary_hash": hash_payload(&summary),
                            }),
                        )
                        .await;
                        if let Some(rewritten) = self
                            .rewrite_summary_echo(&settings, &input, &response_content_no_tags)
                            .await
                        {
                            summary_echo_rewritten = true;
                            let (sanitized, _, _) =
                                sanitize_user_output(&rewritten, false, Some(assistant_name), Some(user_name));
                            let final_text = if sanitized.trim().is_empty() {
                                rewritten
                            } else {
                                sanitized
                            };
                            response_content = final_text.clone();
                            response_content_no_tags = final_text.clone();
                            response_meta.content = response_content.clone();
                            response_meta.content_no_tags = response_content_no_tags.clone();
                            response_meta.raw_content = response_content.clone();
                        }
                    }
                }
            }

            if let Some((question, slots)) = ask_override.take() {
                let cleaned = strip_focus_rationale_pollution(&question);
                ask_override = Some((cleaned, slots));
            }

            if !response_content.trim().is_empty() {
                let cleaned = strip_focus_rationale_pollution(&response_content);
                if cleaned != response_content {
                    response_content = cleaned;
                    response_content_no_tags = response_content.clone();
                    response_meta.content = response_content.clone();
                    response_meta.content_no_tags = response_content_no_tags.clone();
                    response_meta.raw_content = response_content.clone();
                }
            }

            if let Some((question, _)) = ask_override.as_ref() {
                if !question.trim().is_empty() {
                    state.last_assistant_output = Some(question.trim().to_string());
                    state.last_assistant_output_no_tags = Some(sanitize_assistant_for_monologue(question));
                }
            } else if !has_tool_calls && !response_content.trim().is_empty() {
                state.last_assistant_output = Some(response_content.trim().to_string());
                state.last_assistant_output_no_tags =
                    Some(sanitize_assistant_for_monologue(response_content_no_tags.trim()));
            }

            let summary_source = if let Some((question, _)) = ask_override.as_ref() {
                question.as_str()
            } else if !has_tool_calls {
                response_content_no_tags.as_str()
            } else {
                ""
            };
            if !summary_source.trim().is_empty() {
                let summary = summarize_snippet(summary_source, 160);
                state.last_response_summary = Some(summary.clone());
                state.last_response_summary_at = Some(Utc::now().to_rfc3339());
                if let Some(evidence_id) = self
                    .db
                    .create_system_evidence_event(
                        &state.conversation_id,
                        "outcome_summary",
                        &summary,
                        Some(&run_id),
                        &summary,
                    )
                    .await
                {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "memory",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "outcome_summary_evidence",
                            "evidence_id": evidence_id,
                            "conversation_id": state.conversation_id,
                        }),
                    )
                    .await;
                }
            }

            state.introspection_force = None;

            let assistant_name = settings.assistant_display_name.as_deref().unwrap_or("Assistant");
            let user_name = settings.user_display_name.as_deref().unwrap_or("User");
            let response_kind = if ask_override.is_some() {
                "ask"
            } else if has_tool_calls {
                "tool_call"
            } else {
                "emit"
            };
            let lead_text = if let Some((question, _)) = ask_override.as_ref() {
                question.as_str()
            } else {
                response_content.as_str()
            };
            let direct_lead =
                direct_answer_lead(lead_text, allow_diagnostics, Some(assistant_name), Some(user_name));
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "direct_answer_lead",
                    "direct": direct_lead,
                    "response_kind": response_kind,
                    "fallback_applied": validator_fallback_applied,
                    "fallback_reason": validator_fallback_reason,
                    "regen_attempted": validator_regen_attempted,
                }),
            )
            .await;

            break;
        }

        if let Some(packet) = primary_json_packet.take() {
            if let Some(decision_packet) = packet.decision_packet.as_ref() {
                self.apply_decision_packet(&mut state, decision_packet);
            }
            if !packet.candidates.is_empty() {
                for item in packet.candidates {
                    if let Some(candidate) =
                        self.candidate_from_value(&item, "primary_response_json", &mut created_at)
                    {
                        candidates.push(candidate);
                    }
                }
            }
        }

        if self_audit_mode {
            response_content = self.build_self_audit_response(&state, &settings);
            response_content_no_tags = response_content.clone();
            response_meta.content = response_content.clone();
            response_meta.content_no_tags = response_content_no_tags.clone();
            response_meta.raw_content = response_content.clone();
            response_meta.tool_calls = None;
            tool_calls.clear();
            ask_override = None;
            workspace_required_flag = binding_enforcement_enabled && workspace_required(&state);
            if workspace_required_flag {
                let (mentions, exception) = workspace_response_compliant(&response_content, &state);
                workspace_exception = exception;
                workspace_compliant = mentions || exception;
            } else {
                workspace_exception = false;
                workspace_compliant = true;
            }
            workspace_regen = false;
            workspace_fallback = false;
        }

        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(&run_id),
            Some(&trace_id),
            json!({
                "event": "workspace_compliance",
                "required": workspace_required_flag,
                "compliant": workspace_compliant,
                "exception": workspace_exception,
                "regen": workspace_regen,
                "fallback_appended": workspace_fallback,
            }),
        )
        .await;

        let has_tool_calls = !tool_calls.is_empty();
        if let Some(prompt_build) = prompt_build_snapshot.as_ref() {
            let prompt_layout = match prompt_build.prompt_layout {
                PromptLayout::Compact => "compact",
                PromptLayout::Full => "full",
            };
            let prompt_mode = if calculator_mode {
                "calculator"
            } else if self_audit_mode {
                "self_audit"
            } else {
                "normal"
            };
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "integration_checklist",
                    "prompt_layout": prompt_layout,
                    "prompt_mode": prompt_mode,
                    "workspace_present": prompt_build.workspace_present,
                    "inner_summary_present": prompt_build.inner_summary_present,
                    "rolling_summary_present": prompt_build.rolling_summary_present,
                    "introspection_present": prompt_build.introspection_present,
                    "semantic_hint_present": prompt_build.semantic_hint_present,
                    "workspace_hash": prompt_build.workspace_hash,
                    "inner_summary_hash": prompt_build.inner_summary_hash,
                    "rolling_summary_hash": prompt_build.rolling_summary_hash,
                    "capability_manifest_hash": prompt_build.capability_manifest_hash,
                    "binding_enforcement_enabled": binding_enforcement_enabled,
                    "workspace_required": workspace_required_flag,
                    "workspace_compliant": workspace_compliant,
                    "workspace_exception": workspace_exception,
                    "workspace_regen": workspace_regen,
                    "workspace_fallback": workspace_fallback,
                    "pending_prompt_alignment_enabled": settings.pending_prompt_alignment_enabled.unwrap_or(true),
                    "summary_cohesion_enabled": settings.summary_cohesion_enabled.unwrap_or(true),
                    "auto_memory_pass_enabled": settings.auto_memory_pass_enabled.unwrap_or(true),
                    "compact_prompt_enabled": settings.compact_prompt_enabled.unwrap_or(true),
                    "self_audit_mode": self_audit_mode,
                    "calculator_mode": calculator_mode,
                    "has_tool_calls": has_tool_calls,
                }),
            )
            .await;
            let section_sizes: Vec<serde_json::Value> = prompt_build
                .section_metrics
                .iter()
                .map(|s| {
                    json!({
                        "title": s.title,
                        "chars": s.chars,
                        "lines": s.lines,
                        "tokens": s.tokens,
                        "truncated": s.truncated,
                        "budget_chars": s.budget,
                        "budget_tokens": s.budget_tokens,
                    })
                })
                .collect();
            let budget_total_chars: usize = prompt_build
                .section_metrics
                .iter()
                .filter_map(|s| s.budget)
                .sum();
            let budget_total_tokens: usize = prompt_build
                .section_metrics
                .iter()
                .filter_map(|s| s.budget_tokens)
                .sum();
            let over_budget_sections: usize = prompt_build
                .section_metrics
                .iter()
                .filter(|s| s.budget_tokens.map(|b| s.tokens > b).unwrap_or(false))
                .count();
            let prompt_pressure = if budget_total_tokens > 0 {
                (prompt_build.total_tokens as f64 / budget_total_tokens as f64).min(5.0)
            } else {
                0.0
            };
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "prompt_metrics",
                    "prompt_layout": prompt_layout,
                    "prompt_mode": prompt_mode,
                    "total_chars": prompt_build.total_chars,
                    "total_tokens": prompt_build.total_tokens,
                    "budget_total_chars": budget_total_chars,
                    "budget_total_tokens": budget_total_tokens,
                    "prompt_pressure": prompt_pressure,
                    "over_budget_sections": over_budget_sections,
                    "section_count": prompt_build.section_metrics.len(),
                    "section_sizes": section_sizes,
                    "user_evidence_count": prompt_build.user_evidence_count,
                    "tool_evidence_count": prompt_build.tool_evidence_count,
                    "prompt_source": prompt_build.prompt_source,
                    "primary_prompt_hash": prompt_build.primary_prompt_hash,
                    "memory_prompt_hash": prompt_build.memory_prompt_hash,
                    "canonical_primary_hash": prompt_build.canonical_primary_hash,
                    "override_active": prompt_build.override_active,
                    "override_mismatch": prompt_build.override_mismatch,
                    "override_guard_skipped": prompt_build.override_guard_skipped,
                }),
            )
            .await;

            if prompt_build.prompt_source == "embedded" {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "prompt_source_embedded",
                        "primary_prompt_hash": prompt_build.primary_prompt_hash,
                        "memory_prompt_hash": prompt_build.memory_prompt_hash,
                    }),
                )
                .await;
            }

            if let Ok(prev_primary) = self.db.get_key("prompt_primary_hash").await {
                if let Some(prev) = prev_primary.as_deref() {
                    if !prev.is_empty() && prev != prompt_build.primary_prompt_hash {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "prompt_hash_changed",
                                "kind": "primary",
                                "previous": prev,
                                "current": prompt_build.primary_prompt_hash,
                            }),
                        )
                        .await;
                    }
                }
            }
            if let Ok(prev_memory) = self.db.get_key("prompt_memory_hash").await {
                if let Some(prev) = prev_memory.as_deref() {
                    if !prev.is_empty() && prev != prompt_build.memory_prompt_hash {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "prompt_hash_changed",
                                "kind": "memory",
                                "previous": prev,
                                "current": prompt_build.memory_prompt_hash,
                            }),
                        )
                        .await;
                    }
                }
            }
            let _ = self
                .db
                .set_key("prompt_primary_hash", &prompt_build.primary_prompt_hash)
                .await;
            let _ = self
                .db
                .set_key("prompt_memory_hash", &prompt_build.memory_prompt_hash)
                .await;
        }

        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(&run_id),
            Some(&trace_id),
            json!({
                "event": "task_control_state",
                "task_id": state.task_id,
                "prompt_mode": if calculator_mode { "calculator" } else { "normal" },
                "task_phase": task_phase_label(&state.task_phase),
                "stop_latch": state.stop_latch,
                "stop_reason": state.stop_reason,
                "ask_budget_remaining": state.ask_budget_remaining,
                "ask_budget_max": state.ask_budget_max,
                "user_refused": state.user_refused,
                "refused_slots": state.refused_slots,
                "resolved_slots": state.resolved_slots,
                "missing_slots": state.missing_slots,
                "resolution_mode": state.resolution_mode,
                "missing_input_policy": state.missing_input_policy,
                "ask_loop_breaker_triggered": state.ask_loop_breaker_triggered,
                "tool_loop_breaker_triggered": state.tool_loop_breaker_triggered,
                "slot_provenance": state.slot_provenance,
                "registry_profile_name": registry_meta.as_ref().map(|m| m.name.clone()),
                "registry_profile_version": registry_meta.as_ref().map(|m| m.version),
                "registry_profile_hash": registry_meta.as_ref().map(|m| m.hash.clone()),
                "registry_profile_compatibility": registry_meta.as_ref().map(|m| m.compatibility.clone()),
            }),
        )
        .await;
        for call in tool_calls {
            let action_id = Uuid::new_v4().to_string();
            let plan_step_id = state
                .workspace_active_plan_id
                .as_deref()
                .and_then(|active_plan_id| {
                    if ToolRegistry::is_context_only_tool(&call.function.name) {
                        None
                    } else {
                        Some(format!("{}:{}", active_plan_id, action_id))
                    }
                });
            candidates.push(self.make_candidate(
                CandidateKind::ToolCall,
                json!({
                    "action_id": action_id,
                    "tool_name": call.function.name,
                    "arguments": call.function.arguments,
                    "plan_step_id": plan_step_id,
                }),
                "model_tool_call",
                &mut created_at,
            ));
        }

        if !has_tool_calls {
            if let Some((question, slots)) = ask_override.clone() {
                candidates.push(self.make_candidate(
                    CandidateKind::AskUserQuestion,
                    json!({
                        "question": question,
                        "requested_slots": slots,
                    }),
                    "calculator_missing_slots",
                    &mut created_at,
                ));
            } else if !response_content.trim().is_empty() {
                if !calculator_mode {
                    response_content =
                        append_uncertainty_marker(&response_content, state.controller_state.as_ref());
                }
                let mut payload = json!({"content": response_content});
                if calculator_mode && !state.resolved_slots.is_empty() {
                    payload["requires_resolved_slots"] = json!(state.resolved_slots.clone());
                    payload["uses_numeric_inputs"] = json!(state.resolved_slots.clone());
                }
                candidates.push(self.make_candidate(
                    CandidateKind::EmitMessage,
                    payload,
                    "model_response",
                    &mut created_at,
                ));
            }
        }

        if relational_mode {
            let hypothesis_text = format!(
                "Relational input received: {}",
                summarize_snippet(&input, 160)
            );
            if !hypothesis_exists(&state.workspace_active_hypotheses, &hypothesis_text) {
                let hypothesis = WorkspaceHypothesis {
                    text: hypothesis_text,
                    confidence: 0.7,
                    speculative: false,
                    evidence_event_ids: Vec::new(),
                    belief_ids: Vec::new(),
                };
                candidates.push(self.make_candidate(
                    CandidateKind::UpdateWorkspace,
                    json!({
                        "active_hypotheses": [{
                            "text": hypothesis.text,
                            "confidence": hypothesis.confidence
                        }]
                    }),
                    "relational_hypothesis",
                    &mut created_at,
                ));
            }
        }

        let mut gates_started = Some(Instant::now());
        if relational_mode {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "relational_input_detected",
                    "content_len": input.len(),
                }),
            )
            .await;
        }

        let response_status = if has_tool_calls {
            "tool_call"
        } else if ask_override.is_some() {
            "ask"
        } else if response_content.trim().is_empty() {
            "silent"
        } else {
            "complete"
        };
        let response_snippet = summarize_snippet(&response_content, 240);
        candidates.push(self.make_candidate(
            CandidateKind::WriteEpisodic,
            json!({
                "event_type": "assistant_response_finalized",
                "payload": { "status": response_status, "summary_snippet": response_snippet },
                "source_type": "inference",
                "source_ref": run_id,
                "evidence_class": "internal",
            }),
            "assistant_response_finalized",
            &mut created_at,
        ));
        candidates.push(self.make_candidate(
            CandidateKind::WriteEpisodic,
            json!({
                "event_type": "run_finished",
                "payload": { "status": response_status },
                "source_type": "system",
                "source_ref": run_id,
                "evidence_class": "internal",
            }),
            "run_finished",
            &mut created_at,
        ));

        let should_update_inner_summary =
            self.is_meaningful_run(input_kind, &all_outcomes, &response_content);

        let retry_candidates = self.retry_tool_candidates(&state, &mut created_at).await;
        candidates.extend(retry_candidates);
        let deferred_candidates = self.deferred_tool_candidates(&state, &mut created_at).await;
        candidates.extend(deferred_candidates);

        if self_awareness_query {
            let mut cached_user_evidence: Option<Vec<i64>> = None;
            for candidate in candidates.iter_mut() {
                if !matches!(candidate.kind, CandidateKind::RecordSelfClaim) {
                    continue;
                }
                let claim_text = candidate
                    .payload
                    .get("claim_text")
                    .and_then(|v| v.as_str())
                    .or_else(|| candidate.payload.get("claim").and_then(|v| v.as_str()))
                    .or_else(|| candidate.payload.get("text").and_then(|v| v.as_str()))
                    .unwrap_or("")
                    .trim();
                if claim_text.is_empty() {
                    continue;
                }
                if !self_claims::is_self_awareness_claim(claim_text) {
                    continue;
                }
                let evidence_ids = extract_id_list(&candidate.payload, "evidence_event_ids");
                let belief_ids = extract_id_list(&candidate.payload, "belief_ids");
                let evidence_single = candidate
                    .payload
                    .get("evidence_event_id")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let mut provisional = candidate
                    .payload
                    .get("provisional")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let mut requires_validation = candidate
                    .payload
                    .get("requires_validation")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let mut source_type = candidate
                    .payload
                    .get("source_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if source_type.trim().is_empty() {
                    if let Some(obj) = candidate.payload.as_object_mut() {
                        obj.insert(
                            "source_type".to_string(),
                            Value::String("self_awareness_query".to_string()),
                        );
                    }
                    source_type = "self_awareness_query".to_string();
                }
                if !provisional || !requires_validation {
                    if let Some(obj) = candidate.payload.as_object_mut() {
                        if !provisional {
                            obj.insert("provisional".to_string(), Value::Bool(true));
                            provisional = true;
                        }
                        if !requires_validation {
                            obj.insert("requires_validation".to_string(), Value::Bool(true));
                            requires_validation = true;
                        }
                    }
                }
                let provisional_allowed = provisional
                    && (source_type.eq_ignore_ascii_case("system_state")
                        || source_type.eq_ignore_ascii_case("self_awareness_query"));
                let needs_evidence =
                    evidence_ids.is_empty() && evidence_single <= 0 && belief_ids.is_empty() && !provisional_allowed;

                if needs_evidence {
                    if cached_user_evidence.is_none() {
                        cached_user_evidence = Some(
                            self.db
                                .get_recent_user_evidence_ids(&conversation_id, 1)
                                .await,
                        );
                    }
                    if let Some(ids) = cached_user_evidence.as_ref() {
                        if !ids.is_empty() {
                            set_id_list(&mut candidate.payload, "evidence_event_ids", ids);
                        }
                    }
                }
            }
        }
        if !identity_evidence_ids.is_empty() {
            for candidate in candidates.iter_mut() {
                if !matches!(candidate.kind, CandidateKind::RecordSelfClaim) {
                    continue;
                }
                let claim_text = candidate
                    .payload
                    .get("claim_text")
                    .and_then(|v| v.as_str())
                    .or_else(|| candidate.payload.get("claim").and_then(|v| v.as_str()))
                    .or_else(|| candidate.payload.get("text").and_then(|v| v.as_str()))
                    .unwrap_or("")
                    .trim();
                if claim_text.is_empty() || !is_identity_claim_text(claim_text) {
                    continue;
                }
                let evidence_ids = extract_id_list(&candidate.payload, "evidence_event_ids");
                let belief_ids = extract_id_list(&candidate.payload, "belief_ids");
                let evidence_single = candidate
                    .payload
                    .get("evidence_event_id")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let needs_evidence = evidence_ids.is_empty()
                    && evidence_single <= 0
                    && belief_ids.is_empty();
                if needs_evidence {
                    set_id_list(&mut candidate.payload, "evidence_event_ids", &identity_evidence_ids);
                }
                if let Some(obj) = candidate.payload.as_object_mut() {
                    if obj.get("source_type").and_then(|v| v.as_str()).unwrap_or("").is_empty() {
                        obj.insert(
                            "source_type".to_string(),
                            Value::String("user_identity_statement".to_string()),
                        );
                    }
                }
            }
        }

        for candidate in candidates.iter_mut() {
            if !matches!(candidate.kind, CandidateKind::RecordSelfClaim) {
                continue;
            }
            let evidence_ids = extract_id_list(&candidate.payload, "evidence_event_ids");
            let belief_ids = extract_id_list(&candidate.payload, "belief_ids");
            let evidence_single = candidate
                .payload
                .get("evidence_event_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let has_evidence = !evidence_ids.is_empty() || !belief_ids.is_empty() || evidence_single > 0;
            if has_evidence {
                continue;
            }
            if let Some(obj) = candidate.payload.as_object_mut() {
                let provisional = obj
                    .get("provisional")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if !provisional {
                    obj.insert("provisional".to_string(), Value::Bool(true));
                }
                obj.entry("requires_validation".to_string())
                    .or_insert(Value::Bool(true));
                let confidence = obj.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.4);
                let lowered = confidence.min(0.35);
                obj.insert("confidence".to_string(), json!(lowered));
                let source_type = obj
                    .get("source_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if source_type.trim().is_empty() {
                    obj.insert(
                        "source_type".to_string(),
                        Value::String("system_state".to_string()),
                    );
                }
            }
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "self_claim_provisionalized",
                    "candidate_id": candidate.id,
                }),
            )
            .await;
        }

        let pipeline_mode = kernel_pipeline_mode();
        let arbitration = match pipeline_mode {
            KernelPipelineMode::Legacy => {
                self.run_arbitration_phase(
                    &mut candidates,
                    &mut state,
                    &all_outcomes,
                    &conversation_id,
                    &run_id,
                    &trace_id,
                    &settings,
                    &response_content_no_tags,
                    self_awareness_query,
                    gates_started.take(),
                    &mut created_at,
                )
                .await?
            }
            KernelPipelineMode::Phased => {
                self.run_arbitration_phase(
                    &mut candidates,
                    &mut state,
                    &all_outcomes,
                    &conversation_id,
                    &run_id,
                    &trace_id,
                    &settings,
                    &response_content_no_tags,
                    self_awareness_query,
                    gates_started.take(),
                    &mut created_at,
                )
                .await?
            }
        };
        let mut decision = arbitration.decision;
        let mut inline_pending: Option<PendingPromptSelection> = None;
        if matches!(input_kind, CoreInputKind::User) {
            let has_emit = decision
                .accepted
                .iter()
                .any(|candidate| matches!(candidate.kind, CandidateKind::EmitMessage));
            let has_ask = decision
                .accepted
                .iter()
                .any(|candidate| matches!(candidate.kind, CandidateKind::AskUserQuestion));
            if has_emit && !has_ask {
                if let Some(selection) = self
                    .select_pending_prompt_for_proactive(&conversation_id, &state, &settings, true, Some("monologue"))
                    .await
                {
                    let mut question =
                        strip_working_hypothesis_prefix(&selection.prompt, !allow_speculative_markers);
                    if question.trim().is_empty() {
                        question = selection.prompt.clone();
                    }
                    if is_jsonish_text(&question) || !looks_like_question(&question) {
                        let _ = self.db.delete_pending_prompt(&selection.prompt_id).await;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!( {
                                "event": "pending_prompt_sanitized",
                                "reason": if is_jsonish_text(&question) { "json_like" } else { "non_question" },
                                "candidate_id": selection.prompt_id,
                                "source": selection.source,
                            }),
                        )
                        .await;
                    } else {
                        let user_name = settings.user_display_name.as_deref().unwrap_or("User");
                        let last_user_input = state.last_user_input.as_deref().unwrap_or("");
                        if response_has_user_attribution(&question, user_name)
                            && !user_attribution_grounded_in_last_input(&question, last_user_input)
                        {
                            question = rewrite_user_attribution_text(&question, user_name);
                        }
                        let payload = json!({
                            "question": question,
                            "content": question,
                            "pending_prompt_id": selection.prompt_id,
                            "speculative": !selection.exact_open_question,
                            "bridge_id": selection.bridge_id,
                            "intent_kind": selection.intent_kind,
                        });
                        let mut candidate = Candidate {
                            id: selection.prompt_id.clone(),
                            kind: CandidateKind::AskUserQuestion,
                            payload,
                            evidence_event_ids: Vec::new(),
                            belief_ids: Vec::new(),
                            target_scope: None,
                            rationale: None,
                            expected_outcome: None,
                            cost: Some(0),
                            urgency: Some(0),
                            source: "pending_prompt".to_string(),
                            priority_class: priority_class_for(&CandidateKind::AskUserQuestion),
                            priority_rank: 0,
                            created_at: state.monologue_count,
                        };
                        candidate.refresh_meta();
                        decision.accepted.push(candidate);
                        inline_pending = Some(selection);
                    }
                }
            }
        }
        let anchor_hits = arbitration.anchor_hits;
        let gates_ms = arbitration.gates_ms;
        let monologue_state_update_required = decision
            .accepted
            .iter()
            .any(|candidate| is_monologue_source(&candidate.source) && is_state_change_candidate(&candidate.kind));

        let _ = advance_run_phase(
            &self.db.pool,
            Some(&self.app_handle),
            &run_id,
            RunPhase::Commit,
            Some("commit_cycle"),
        )
        .await;
        let commit_started = Instant::now();
        let commit_result = if monologue_state_update_required {
            self.commit_cycle(
                &mut state,
                &decision,
                &conversation_id,
                Some(&run_id),
                Some(&trace_id),
                &settings,
                false,
                None,
                false,
            )
            .await?
        } else {
            let (commit_tx, commit_rx) = oneshot::channel();
            let db = self.db.clone();
            let model_client = self.model_client.clone();
            let app_handle = self.app_handle.clone();
            let settings_snapshot = settings.clone();
            let conversation_id_clone = conversation_id.clone();
            let run_id_clone = run_id.clone();
            let trace_id_clone = trace_id.clone();
            let decision_clone = decision.clone();
            let mut commit_state = state.clone();
            let commit_handle = tokio::spawn(async move {
                let kernel = Kernel::new(db, model_client, app_handle);
                kernel
                    .commit_cycle(
                        &mut commit_state,
                        &decision_clone,
                        &conversation_id_clone,
                        Some(&run_id_clone),
                        Some(&trace_id_clone),
                        &settings_snapshot,
                        false,
                        Some(commit_tx),
                        true,
                    )
                    .await
            });
            match commit_rx.await {
                Ok(result) => result,
                Err(_) => match commit_handle.await {
                    Ok(Ok(result)) => result,
                    Ok(Err(err)) => return Err(err),
                    Err(err) => return Err(err.to_string()),
                },
            }
        };
        if let Some(selection) = inline_pending.take() {
            let attempt_at = Utc::now().to_rfc3339();
            let _ = self
                .db
                .mark_pending_prompt_attempt(&selection.prompt_id, &attempt_at)
                .await;
            let _ = self.db.delete_pending_prompt(&selection.prompt_id).await;
            if let Ok(count) = self.db.count_pending_prompts(&conversation_id).await {
                let _ = self.app_handle.emit("pending_prompt_count", count as usize);
            }
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "chat",
                Some(&run_id),
                Some(&trace_id),
                json!( {
                    "event": "pending_prompt_inline_surface",
                    "prompt_id": selection.prompt_id,
                    "source": selection.source,
                    "anchor_message_id": selection.anchor_message_id,
                    "pending_prompt_anchor_age_seconds": selection.anchor_age_seconds,
                }),
            )
            .await;
        }
        let commit_ms = commit_started.elapsed().as_millis() as i64;
        self.mark_candidate_outcomes(&decision, "accepted", "rejected")
            .await;
        if monologue_state_update_required {
            let _ = self
                .build_and_persist_subject_snapshot(&mut state, Some(&run_id), Some(&run_id), "monologue_state_update")
                .await;
        }

        let mut response = commit_result.emit_content.clone();
        let mut response_origin = compute_response_origin(
            allow_diagnostics,
            summary_echo_rewritten,
            workspace_fallback,
            commit_result.emit_source.as_deref(),
        );
        if let Some(content) = response.as_deref() {
            if let Some(proposal_id) = decision.report.proposal_id.as_deref() {
                if let Some(event_id) = self
                    .db
                    .create_system_evidence_event(
                        conversation_id.as_str(),
                        "plan_response",
                        proposal_id,
                        Some(proposal_id),
                        &summarize_snippet(content, 240),
                    )
                    .await
                {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "plan_response_evidence_created",
                            "proposal_id": proposal_id,
                            "evidence_event_id": event_id,
                        }),
                    )
                    .await;
                }
                if decision
                    .report
                    .plan_state
                    .as_deref()
                    .map(|s| s.eq_ignore_ascii_case("verified"))
                    .unwrap_or(false)
                {
                    state.workspace_active_plan_id = Some(proposal_id.to_string());
                    let _ = self
                        .db
                        .set_workspace_active_plan(conversation_id.as_str(), Some(proposal_id))
                        .await;
                    if let Err(err) = self.db.set_action_proposal_state(proposal_id, "active").await {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "plan_state_update_failed",
                                "proposal_id": proposal_id,
                                "plan_state": "active",
                                "error": err.to_string(),
                            }),
                        )
                        .await;
                    } else {
                        decision.report.plan_state = Some("active".to_string());
                    }
                }
            }
        }
        let mut tool_result: Option<ToolExecutionResult> = None;
        let mut tool_failed = false;
        let action_tool_dispatches = commit_result
            .tool_dispatches
            .iter()
            .filter(|dispatch| !ToolRegistry::is_context_only_tool(&dispatch.tool_name))
            .collect::<Vec<_>>();
        let has_action_tool_dispatch = !action_tool_dispatches.is_empty();
        let mut plan_ids: Vec<String> = Vec::new();
        for dispatch in action_tool_dispatches.iter() {
            if let Some(step_id) = dispatch.plan_step_id.as_deref() {
                if let Some(plan_id) = extract_plan_id_from_step_id(step_id) {
                    plan_ids.push(plan_id);
                }
            }
        }
        if plan_ids.is_empty() {
            if let Some(proposal_id) = decision.report.proposal_id.as_deref() {
                plan_ids.push(proposal_id.to_string());
            }
        }
        plan_ids.sort();
        plan_ids.dedup();
        if !plan_ids.is_empty() {
            let primary_plan_id = plan_ids.first().cloned();
            state.workspace_active_plan_id = primary_plan_id.clone();
            if let Some(active_plan_id) = primary_plan_id.as_deref() {
                if let Err(err) = self
                    .db
                    .set_workspace_active_plan(conversation_id.as_str(), Some(active_plan_id))
                    .await
                {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "active_plan_set_failed",
                            "proposal_id": active_plan_id,
                            "error": err.to_string(),
                        }),
                    )
                    .await;
                }
            }
            for plan_id in plan_ids.iter() {
                if let Err(err) = self.db.set_action_proposal_state(plan_id, "active").await {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "plan_state_update_failed",
                            "proposal_id": plan_id,
                            "plan_state": "active",
                            "error": err.to_string(),
                        }),
                    )
                    .await;
                } else {
                    decision.report.plan_state = Some("active".to_string());
                    if let Some(event_id) = self
                        .db
                        .create_system_evidence_event(
                            conversation_id.as_str(),
                            "plan_state",
                            "active",
                            Some(plan_id),
                            &format!("plan_state active for {}", plan_id),
                        )
                        .await
                    {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "plan_state_evidence_created",
                                "proposal_id": plan_id,
                                "plan_state": "active",
                                "evidence_event_id": event_id,
                            }),
                        )
                        .await;
                    }
                }
            }
        }

        if let Some(active_plan_id) = state.workspace_active_plan_id.clone() {
            let mut backfilled_steps: Vec<String> = Vec::new();
            for dispatch in action_tool_dispatches.iter() {
                if dispatch.plan_step_id.is_none() {
                    let step_id = format!("{}:{}", active_plan_id, dispatch.action_id);
                    let _ = sqlx::query(
                        "UPDATE tool_dispatches
                         SET plan_step_id = ?
                         WHERE action_id = ? AND (plan_step_id IS NULL OR plan_step_id = '')",
                    )
                    .bind(&step_id)
                    .bind(&dispatch.action_id)
                    .execute(&self.db.pool)
                    .await;
                    backfilled_steps.push(step_id);
                }
            }
            if !backfilled_steps.is_empty() {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "plan_step_id_backfilled",
                        "proposal_id": active_plan_id.as_str(),
                        "count": backfilled_steps.len(),
                    }),
                )
                .await;
            }
            self.maybe_seed_goal_stack_from_plan(
                &mut state,
                conversation_id.as_str(),
                &run_id,
                &active_plan_id,
            )
            .await;
        }

        if !commit_result.tool_dispatches.is_empty() {
            let _ = advance_run_phase(
                &self.db.pool,
                Some(&self.app_handle),
                &run_id,
                RunPhase::ToolDispatch,
                Some("tool_dispatch"),
            )
            .await;
            for tool_dispatch in commit_result.tool_dispatches.iter() {
                let tool_started = Instant::now();
                tool_result = self
                    .dispatch_tool(
                        tool_dispatch,
                        Some(&run_id),
                        Some(&trace_id),
                        &mut cancel_rx,
                    )
                    .await;
                if let Some(result) = tool_result.as_ref() {
                    if result.is_error {
                        tool_failed = true;
                    }
                }
                tool_ms += tool_started.elapsed().as_millis() as i64;
            }
        }
        if has_action_tool_dispatch && !state.workspace_goal_stack.is_empty() {
            let mut evidence_ids: Vec<i64> = Vec::new();
            for dispatch in action_tool_dispatches.iter() {
                if let Ok(row) = sqlx::query(
                    "SELECT evidence_event_id FROM tool_dispatches WHERE action_id = ? LIMIT 1",
                )
                .bind(&dispatch.action_id)
                .fetch_optional(&self.db.pool)
                .await
                {
                    if let Some(row) = row {
                        let evidence_id: Option<i64> = row.try_get("evidence_event_id").ok();
                        if let Some(id) = evidence_id {
                            evidence_ids.push(id);
                        }
                    }
                }
            }
            evidence_ids.sort();
            evidence_ids.dedup();
            if !evidence_ids.is_empty() {
                let mut goal_stack = state.workspace_goal_stack.clone();
                if let Some(item) = goal_stack.first_mut() {
                    let step_idx = item.current_step_index as usize;
                    if let Some(step) = item.steps.get_mut(step_idx) {
                        let mut updated = false;
                        for id in evidence_ids.iter().copied() {
                            if !step.evidence_event_ids.contains(&id) {
                                step.evidence_event_ids.push(id);
                                updated = true;
                            }
                        }
                        if updated {
                            item.updated_at = Some(Utc::now().to_rfc3339());
                            state.workspace_goal_stack = goal_stack;
                            let workspace_state = crate::models::WorkspaceState {
                                conversation_id: conversation_id.to_string(),
                                goal_thread: state.workspace_goal_thread.clone(),
                                active_plan_id: state.workspace_active_plan_id.clone(),
                                goal_stack: state.workspace_goal_stack.clone(),
                                open_questions: state.workspace_open_questions.clone(),
                                active_hypotheses: state.workspace_active_hypotheses.clone(),
                                working_set_topics: state.workspace_working_set_topics.clone(),
                                current_focus: state.workspace_current_focus.clone(),
                                focus_rationale: state.workspace_focus_rationale.clone(),
                                workspace_meta: state.workspace_meta.clone(),
                                updated_at: None,
                            };
                            let _ = self.db.set_workspace_state(&workspace_state).await;
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "kernel",
                                Some(&run_id),
                                Some(&trace_id),
                                json!({
                                    "event": "goal_step_evidence_attached",
                                    "proposal_id": state.workspace_active_plan_id.clone(),
                                    "evidence_event_ids": evidence_ids,
                                    "step_index": step_idx,
                                }),
                            )
                            .await;
                        }
                    }
                }
            }
        }
        if has_action_tool_dispatch {
            let primary_plan_id = plan_ids.first().cloned().or_else(|| {
                decision.report.proposal_id.as_deref().map(|s| s.to_string())
            });
            if let (Some(proposal_id), Some(result)) = (primary_plan_id.as_deref(), tool_result.as_ref()) {
                if !result.is_error && !tool_failed {
                    if let Err(err) = self.db.set_action_proposal_state(proposal_id, "completed").await {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "plan_state_update_failed",
                                "proposal_id": proposal_id,
                                "plan_state": "completed",
                                "error": err.to_string(),
                            }),
                        )
                        .await;
                    } else {
                        decision.report.plan_state = Some("completed".to_string());
                        if let Some(event_id) = self
                            .db
                            .create_system_evidence_event(
                                conversation_id.as_str(),
                                "plan_state",
                                "completed",
                                Some(proposal_id),
                                &format!("plan_state completed for {}", proposal_id),
                            )
                            .await
                        {
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "kernel",
                                Some(&run_id),
                                Some(&trace_id),
                                json!({
                                    "event": "plan_state_evidence_created",
                                    "proposal_id": proposal_id,
                                    "plan_state": "completed",
                                    "evidence_event_id": event_id,
                                }),
                            )
                            .await;
                        }
                    }
                }
            }
        }

        let override_response = self.finalize_decision_report(
            &mut decision,
            &state,
            prompt_build_snapshot.as_ref(),
            Some(anchor_hits),
            response.as_deref(),
            &commit_result,
            tool_result.as_ref(),
            false,
        );
        if let Some(override_content) = override_response {
            response = Some(override_content);
            response_origin = ResponseOrigin::Fallback;
        }

        let mut best_effort_dropped = false;
        if let Some(content) = response.as_deref() {
            let (normalized_with_tags, cleaned_no_tags, normalized_tags) =
                crate::core::memory::inject_context::enforce_trailing_system_tags(content);
            if normalized_with_tags != content {
                response = Some(normalized_with_tags.clone());
            }
            if cleaned_no_tags != response_content_no_tags {
                response_content_no_tags = cleaned_no_tags;
            }
            response_meta.tag_set = normalized_tags;
            if let Some(updated) = response.as_ref() {
                response_meta.content = updated.clone();
                response_meta.content_no_tags = response_content_no_tags.clone();
                response_meta.raw_content = updated.clone();
            }
        }

        if tool_result.is_none() && settings.context_miss_detector_enabled.unwrap_or(true) {
            let skip_context_miss = matches!(input_kind, CoreInputKind::ToolResult | CoreInputKind::ToolError)
                && ToolRegistry::is_context_only_tool(input_source);
            if !skip_context_miss {
                if let Some(content) = response.clone() {
                    let miss_tool =
                        detect_context_miss_tool(&content, prompt_build_snapshot.as_ref());
                    let shadow_mode = settings
                        .context_hydration_mode
                        .as_deref()
                        .unwrap_or("shadow")
                        .eq_ignore_ascii_case("shadow");
                    if let Some(tool_name) = miss_tool {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "context_miss_detected",
                                "tool": tool_name,
                                "content_snippet": summarize_snippet(&content, 160),
                            }),
                        )
                        .await;
                        if context_mode_is_thin(&settings) && !shadow_mode {
                            let action_id = format!("context_hydration:{}", Uuid::new_v4());
                            let args_json = json!({ "conversation_id": conversation_id }).to_string();
                            let dispatch = ToolDispatchRequest {
                                action_id,
                                tool_name: tool_name.to_string(),
                                args_json,
                                plan_step_id: None,
                            };
                            let tool_started = Instant::now();
                            tool_result = self
                                .dispatch_tool(&dispatch, Some(&run_id), Some(&trace_id), &mut cancel_rx)
                                .await;
                            tool_ms += tool_started.elapsed().as_millis() as i64;
                            if tool_result.is_some() {
                                response = None;
                                response_content_no_tags.clear();
                                response_meta.content.clear();
                                response_meta.content_no_tags.clear();
                                response_meta.raw_content.clear();
                                response_origin = ResponseOrigin::Tool;
                            }
                        }
                    }
                    if shadow_mode {
                        let miss = miss_tool.is_some();
                        if let Some(tool_name) = miss_tool {
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "kernel",
                                Some(&run_id),
                                Some(&trace_id),
                                json!({
                                    "event": "context_hydration_shadow_miss",
                                    "tool": tool_name,
                                    "content_snippet": summarize_snippet(&content, 160),
                                }),
                            )
                            .await;
                        }
                        let total_key = "context_hydration_shadow_total";
                        let miss_key = "context_hydration_shadow_miss";
                        let total_prev = self
                            .db
                            .get_key(total_key)
                            .await
                            .ok()
                            .flatten()
                            .and_then(|v| v.parse::<i64>().ok())
                            .unwrap_or(0);
                        let miss_prev = self
                            .db
                            .get_key(miss_key)
                            .await
                            .ok()
                            .flatten()
                            .and_then(|v| v.parse::<i64>().ok())
                            .unwrap_or(0);
                        let total_next = total_prev.saturating_add(1);
                        let miss_next = miss_prev.saturating_add(if miss { 1 } else { 0 });
                        let _ = self
                            .db
                            .set_key(total_key, &total_next.to_string())
                            .await;
                        let _ = self.db.set_key(miss_key, &miss_next.to_string()).await;
                        let miss_rate = if total_next > 0 {
                            miss_next as f64 / total_next as f64
                        } else {
                            0.0
                        };
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "context_hydration_shadow_metrics",
                                "total": total_next,
                                "misses": miss_next,
                                "miss_rate": miss_rate,
                            }),
                        )
                        .await;
                        if total_next >= 30 && miss_rate <= 0.05 {
                            let _ = sqlx::query(
                                "UPDATE settings SET context_hydration_mode = 'thin' WHERE id = 1",
                            )
                            .execute(&self.db.pool)
                            .await;
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "kernel",
                                Some(&run_id),
                                Some(&trace_id),
                                json!({
                                    "event": "context_hydration_auto_thin_enabled",
                                    "total": total_next,
                                    "misses": miss_next,
                                    "miss_rate": miss_rate,
                                }),
                            )
                            .await;
                        }
                    }
                }
            }
        }

        let response_snapshot = response.clone();
        if let Some(content) = response_snapshot.as_ref() {
            let emit_started = Instant::now();
            let mut emitted_content = content.clone();
            if !content.trim().is_empty() {
                if let Some(updated) = self
                    .finalize_assistant_message(
                        &run_id,
                        content,
                        response_origin,
                        decision.report.gate_decision.clone(),
                        decision.report.gate_notice.clone(),
                        decision.report.gate_reasons.clone(),
                        extra_notice.clone(),
                    )
                    .await
                {
                    emitted_content = updated.clone();
                    response = Some(updated);
                }
            }
            emit_ms = emit_started.elapsed().as_millis() as i64;
            if emit_ms > 0 {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "timing_emit",
                        "duration_ms": emit_ms,
                    }),
                )
                .await;
                self.update_latency_avg("emit", emit_ms).await;
            }
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "emit_decision",
                    "content_len": emitted_content.len(),
                    "content_hash": hash_payload(&emitted_content),
                    "content_snippet": summarize_snippet(&emitted_content, 120),
                }),
            )
            .await;
            let monologue_ids = self
                .recent_monologue_entry_ids(&conversation_id, MONOLOGUE_SURFACE_WINDOW_SECS, 8)
                .await;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "monologue_influence",
                    "conversation_id": conversation_id,
                    "entry_ids": monologue_ids.clone(),
                    "window_secs": MONOLOGUE_SURFACE_WINDOW_SECS,
                }),
            )
            .await;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "decision_trace",
                    "conversation_id": conversation_id,
                    "response_hash": hash_payload(&content),
                    "monologue_entry_ids": monologue_ids,
                }),
            )
            .await;
        } else {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                "event": "silent_cycle",
                "reason": "no_emit_candidate",
            }),
        )
        .await;
        }

        let should_finalize = tool_result.is_none();
        if should_finalize {
            let _ = advance_run_phase(
                &self.db.pool,
                Some(&self.app_handle),
                &run_id,
                RunPhase::Finalize,
                Some("finalize_response"),
            )
            .await;
        }

        if should_finalize {
            if response_snapshot.is_some() {
                let db = self.db.clone();
                let model_client = self.model_client.clone();
                let app_handle = self.app_handle.clone();
                let settings_snapshot = settings.clone();
                let conversation_id = conversation_id.clone();
                let run_id = run_id.clone();
                let trace_id = trace_id.clone();
                let log_conversation_id = conversation_id.clone();
                let log_run_id = run_id.clone();
                let log_trace_id = trace_id.clone();
                let input = input.clone();
                let response_no_tags = response_content_no_tags.clone();
                let summary_input = input.clone();
                let summary_response = response_no_tags.clone();
                let summary_outcomes = all_outcomes.clone();
                let summary_state = state.clone();
                let update_inner_summary = should_update_inner_summary;
                let summary_snapshot_hash = decision.report.snapshot_hash.clone();
                let summary_gate_decision_id = decision.report.gate_decision_id.clone();
                let tag_any = response_meta.tag_set.any();
                let tag_resolve = response_meta.tag_set.resolve;
                let tag_clarify = response_meta.tag_set.clarify;
                let tag_memory = response_meta.tag_set.memory;
                let has_tool_calls = response_meta
                    .tool_calls
                    .as_ref()
                    .map(|calls| !calls.is_empty())
                    .unwrap_or(false);
                let self_audit_mode = self_audit_mode;

                let post_processing_mode = system_controls::mode_for(
                    "post_processing",
                    &system_controls::load_control_map(&self.db).await,
                );
                let queued = if system_controls::mode_is_off(&post_processing_mode) {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        Some(&log_run_id),
                        Some(&log_trace_id),
                        json!({
                            "event": "post_processing_backpressure",
                            "reason": "system_control_off",
                            "conversation_id": log_conversation_id,
                        }),
                    )
                    .await;
                    None
                } else {
                    post_processing::enqueue_post_processing_with_priority(
                        post_processing::JobPriority::Critical,
                        move || async move {
                            let kernel = Kernel::new(db, model_client, app_handle);
                            let (gate_allows_writes, gate_decision, gate_decision_id, fallback_reason) =
                                if let Some(decision_id) = summary_gate_decision_id.as_deref() {
                                    let decision: Option<String> = sqlx::query_scalar(
                                        "SELECT decision FROM gate_decisions
                                         WHERE decision_id = ?
                                         LIMIT 1",
                                    )
                                    .bind(decision_id)
                                    .fetch_optional(&kernel.db.pool)
                                    .await
                                    .ok()
                                    .flatten();
                                    let allows = decision
                                        .as_deref()
                                        .map(gate_allows_writes_decision)
                                        .unwrap_or(false);
                                    (allows, decision, summary_gate_decision_id.clone(), None)
                                } else {
                                    (
                                        true,
                                        Some("ALLOW_WITH_NOTICE".to_string()),
                                        None,
                                        Some("missing_gate_decision".to_string()),
                                    )
                                };
                            if let Some(reason) = fallback_reason.as_deref() {
                                let _ = system_log::log_event(
                                    &kernel.db.pool,
                                    Some(&kernel.app_handle),
                                    "info",
                                    "memory",
                                    Some(&run_id),
                                    Some(&trace_id),
                                    json!( {
                                        "event": "memory_gate_fallback",
                                        "reason": reason,
                                        "gate_decision": gate_decision,
                                        "gate_decision_id": gate_decision_id,
                                        "snapshot_hash": summary_snapshot_hash,
                                        "conversation_id": conversation_id,
                                    }),
                                )
                                .await;
                            }
                            if !gate_allows_writes {
                                let _ = system_log::log_event(
                                    &kernel.db.pool,
                                    Some(&kernel.app_handle),
                                    "warn",
                                    "memory",
                                    Some(&run_id),
                                    Some(&trace_id),
                                    json!( {
                                        "event": "memory_write_blocked",
                                        "reason": "gate_decision",
                                        "gate_decision": gate_decision,
                                        "gate_decision_id": gate_decision_id,
                                        "snapshot_hash": summary_snapshot_hash,
                                        "conversation_id": conversation_id,
                                    }),
                                )
                                .await;
                            }
                let control_map = system_controls::load_control_map(&kernel.db).await;
                let inner_summary_mode = system_controls::mode_for("inner_summary", &control_map);
                let memory_write_mode = system_controls::mode_for("memory_write", &control_map);
                let mut inner_summary_blocked: Option<&str> = None;
                if system_controls::mode_is_off(&inner_summary_mode) {
                    inner_summary_blocked = Some("inner_summary_off");
                } else if !system_controls::allow_memory_write(&memory_write_mode, "inner_summary_update") {
                    inner_summary_blocked = Some("memory_write_control");
                }

                if update_inner_summary && gate_allows_writes && inner_summary_blocked.is_none() {
                    let mut created_at = 0i64;
                    match kernel
                        .build_inner_summary_candidate(
                            &conversation_id,
                            &summary_input,
                            &summary_response,
                            &summary_outcomes,
                            &summary_state,
                            &settings_snapshot,
                            &mut created_at,
                        )
                        .await
                    {
                        Ok(candidate) => {
                            let has_evidence = candidate_has_evidence(&candidate.payload)
                                || matches!(candidate_evidence_class(&candidate), Some("internal"));
                            if !has_evidence {
                                let _ = system_log::log_event(
                                    &kernel.db.pool,
                                    Some(&kernel.app_handle),
                                    "warn",
                                    "memory",
                                    Some(&run_id),
                                    Some(&trace_id),
                                    json!({
                                        "event": "memory_write_blocked",
                                        "reason": "missing_evidence",
                                        "category": "inner_summary",
                                        "candidate_id": candidate.id,
                                        "candidate_kind": format!("{:?}", candidate.kind),
                                        "conversation_id": conversation_id,
                                    }),
                                )
                                .await;
                            } else if let Some(summary_json) =
                                candidate.payload.get("summary_json").and_then(|v: &Value| v.as_str())
                            {
                                let allowed = MemoryPolicy::is_allowed(
                                    MemoryWriteCategory::InnerSummary,
                                    MemoryWriteSource::Kernel,
                                    "user_visible_turn",
                                );
                                if !allowed {
                                    let _ = system_log::log_event(
                                        &kernel.db.pool,
                                        Some(&kernel.app_handle),
                                        "warn",
                                        "memory_policy",
                                        Some(&run_id),
                                        Some(&trace_id),
                                        json!({
                                            "event": "memory_policy_violation",
                                            "category": "inner_summary",
                                            "reason_code": "user_visible_turn",
                                            "conversation_id": conversation_id,
                                        }),
                                    )
                                    .await;
                                } else if let Err(err) =
                                    kernel.db.set_inner_summary(&conversation_id, summary_json).await
                                {
                                    let _ = system_log::log_event(
                                        &kernel.db.pool,
                                        Some(&kernel.app_handle),
                                        "warn",
                                        "kernel",
                                        Some(&run_id),
                                        Some(&trace_id),
                                        json!({
                                            "event": "inner_summary_failed",
                                            "error": err.to_string(),
                                            "conversation_id": conversation_id,
                                        }),
                                    )
                                    .await;
                                } else {
                                    let hash = hash_payload(summary_json);
                                    let _ = kernel
                                        .db
                                        .log_memory_write(
                                            Some(&conversation_id),
                                            "inner_summary",
                                            "kernel",
                                            "user_visible_turn",
                                            Some(&run_id),
                                            Some(&trace_id),
                                            Some(&hash),
                                            summary_snapshot_hash.as_deref(),
                                            summary_gate_decision_id.as_deref(),
                                        )
                                        .await;
                                }
                            }
                        }
                        Err(err) => {
                            let _ = kernel.db.set_summary_pending(&conversation_id, true).await;
                            let _ = system_log::log_event(
                                &kernel.db.pool,
                                Some(&kernel.app_handle),
                                "warn",
                                "kernel",
                                Some(&run_id),
                                Some(&trace_id),
                                json!({
                                    "event": "inner_summary_failed",
                                    "error": err,
                                    "conversation_id": conversation_id,
                                }),
                            )
                            .await;
                        }
                    }
                } else if update_inner_summary && gate_allows_writes {
                    let reason = inner_summary_blocked.unwrap_or("blocked");
                    let _ = system_log::log_event(
                        &kernel.db.pool,
                        Some(&kernel.app_handle),
                        "warn",
                        "memory",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "memory_write_blocked",
                            "reason": reason,
                            "category": "inner_summary",
                            "conversation_id": conversation_id,
                        }),
                    )
                    .await;
                }
                let mut memory_pass_result: Option<MemoryPassResult> = None;
                let mut memory_pass_started = false;
                let mut candidate_count = 0usize;
                let mut novelty_rejections = 0usize;
                if gate_allows_writes && !self_audit_mode && tag_any {
                    let memory_triggered = if tag_resolve {
                        true
                    } else if tag_clarify {
                        false
                    } else {
                        tag_memory
                    };
                    if memory_triggered {
                        memory_pass_started = true;
                        let started_at = Utc::now();
                        kernel.emit_module_status(Some(&run_id), "memory_pass", "Updating memory", started_at, 0);
                        let result = kernel
                            .run_memory_pass(
                                &settings_snapshot,
                                &run_id,
                                &input,
                                &response_no_tags,
                                true,
                            )
                            .await;
                        memory_pass_result = Some(result);
                        let duration_ms = Utc::now()
                            .signed_duration_since(started_at)
                            .num_milliseconds()
                            .max(0);
                        kernel.emit_module_status(Some(&run_id), "idle", "Idle", started_at, duration_ms);
                    }
                } else if gate_allows_writes
                    && !self_audit_mode
                    && settings_snapshot.auto_memory_pass_enabled.unwrap_or(true)
                {
                    let mut disqualifiers: Vec<&str> = Vec::new();
                    if is_trivial_user_message(&input) {
                        disqualifiers.push("trivial_user_input");
                    }
                    if response_no_tags.trim().is_empty() {
                        disqualifiers.push("empty_response");
                    }
                    if has_tool_calls {
                        disqualifiers.push("tool_call");
                    }

                    let mut candidates: Vec<candidate::MemoryCandidate> = Vec::new();
                    let mut scored: Vec<(candidate::MemoryCandidate, candidate::CandidateScore)> = Vec::new();
                    let mut trigger_reason = "no_candidates";
                    let mut auto_trigger = false;

                    if disqualifiers.is_empty() {
                        if candidate::should_extract_from_user(&input) {
                            let _ = system_log::log_event(
                                &kernel.db.pool,
                                Some(&kernel.app_handle),
                                "info",
                                "memory",
                                Some(&run_id),
                                Some(&trace_id),
                                json!({
                                    "event": "memory_candidate_input",
                                    "conversation_id": conversation_id,
                                    "input_len": input.len(),
                                    "input": input,
                                }),
                            )
                            .await;

                            candidates = candidate::extract_candidates(&input, &response_no_tags);
                            let summary: Vec<Value> = candidates
                                .iter()
                                .map(|c| {
                                    let participants: Vec<String> = c
                                        .participants
                                        .iter()
                                        .map(|(role, r)| {
                                            let ref_str = match r {
                                                crate::core::memory::dsl::Ref::Handle(h) => format!("${}", h),
                                                crate::core::memory::dsl::Ref::Label(l) => format!("#{}", l),
                                                crate::core::memory::dsl::Ref::Filter(l, f) => {
                                                    format!("#{}:{}", l, f)
                                                }
                                                crate::core::memory::dsl::Ref::Name(n) => format!("\"{}\"", n),
                                            };
                                            if role.trim().is_empty() {
                                                ref_str
                                            } else {
                                                format!("{}:{}", role, ref_str)
                                            }
                                        })
                                        .collect();
                                    json!({
                                        "kind": match c.kind { candidate::CandidateKind::Fact => "fact", candidate::CandidateKind::Relation => "relation" },
                                        "key": c.key,
                                        "value": c.value,
                                        "rel_type": c.rel_type,
                                        "participants": participants,
                                        "signals": c.signals,
                                    })
                                })
                                .collect();
                            let _ = system_log::log_event(
                                &kernel.db.pool,
                                Some(&kernel.app_handle),
                                "info",
                                "memory",
                                Some(&run_id),
                                Some(&trace_id),
                                json!({
                                    "event": "memory_candidate",
                                    "conversation_id": conversation_id,
                                    "candidates": summary,
                                }),
                            )
                            .await;

                            if !candidates.is_empty() {
                                scored = candidate::score_candidates(&kernel.db.pool, &candidates).await;
                                candidate_count = scored.len();
                                novelty_rejections = scored
                                    .iter()
                                    .filter(|(_, s)| s.novelty < 0.5)
                                    .count();
                                let score_summary: Vec<Value> = scored
                                    .iter()
                                    .map(|(c, s)| {
                                        json!({
                                            "kind": match c.kind { candidate::CandidateKind::Fact => "fact", candidate::CandidateKind::Relation => "relation" },
                                            "key": c.key,
                                            "value": c.value,
                                            "rel_type": c.rel_type,
                                            "total": s.total,
                                            "novelty": s.novelty,
                                            "durability": s.durability,
                                            "relevance": s.relevance,
                                            "evidence": s.evidence,
                                            "relationship": s.relationship,
                                            "reasons": s.reasons,
                                        })
                                    })
                                    .collect();
                                let _ = system_log::log_event(
                                    &kernel.db.pool,
                                    Some(&kernel.app_handle),
                                    "info",
                                    "memory",
                                    Some(&run_id),
                                    Some(&trace_id),
                                    json!({
                                        "event": "memory_candidate_score",
                                        "conversation_id": conversation_id,
                                        "scores": score_summary,
                                    }),
                                )
                                .await;

                                let scores_only: Vec<candidate::CandidateScore> =
                                    scored.iter().map(|(_, s)| s.clone()).collect();
                                auto_trigger = candidate::should_trigger(
                                    &scores_only,
                                    candidate::MEMORY_CANDIDATE_TRIGGER_THRESHOLD,
                                );
                                trigger_reason = if auto_trigger {
                                    "score_threshold"
                                } else {
                                    "below_threshold"
                                };
                            }
                        } else {
                            trigger_reason = "filtered_short_utterance";
                        }
                    } else {
                        trigger_reason = "disqualified";
                    }

                    let _ = system_log::log_event(
                        &kernel.db.pool,
                        Some(&kernel.app_handle),
                        "info",
                        "memory",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "memory_trigger_decision",
                            "conversation_id": conversation_id,
                            "trigger": auto_trigger,
                            "reason": trigger_reason,
                            "threshold": candidate::MEMORY_CANDIDATE_TRIGGER_THRESHOLD,
                            "disqualifiers": disqualifiers,
                            "candidate_count": candidates.len(),
                        }),
                    )
                    .await;

                    if auto_trigger {
                        memory_pass_started = true;
                        let started_at = Utc::now();
                        kernel.emit_module_status(Some(&run_id), "memory_pass", "Updating memory", started_at, 0);
                        let result = kernel
                            .run_memory_pass(&settings_snapshot, &run_id, &input, &response_no_tags, true)
                            .await;
                        memory_pass_result = Some(result);
                        let duration_ms = Utc::now()
                            .signed_duration_since(started_at)
                            .num_milliseconds()
                            .max(0);
                        kernel.emit_module_status(Some(&run_id), "idle", "Idle", started_at, duration_ms);
                    }

                    if memory_pass_started {
                        if let Some(result) = memory_pass_result.as_ref() {
                            let needs_fallback = result.error.is_some() || result.written_ids == 0;
                            if needs_fallback {
                                let (fallback_mode, fallback_statements) = if !scored.is_empty() {
                                    (
                                        "top1_scored",
                                        candidate::build_top_fallback_statements(
                                            &scored,
                                            candidate::MEMORY_CANDIDATE_TRIGGER_THRESHOLD,
                                        ),
                                    )
                                } else {
                                    (
                                        "top1_raw",
                                        candidate::build_fallback_statements(&candidates)
                                            .into_iter()
                                            .take(1)
                                            .collect::<Vec<_>>(),
                                    )
                                };
                                if !fallback_statements.is_empty() {
                                    let fallback_fact_count = fallback_statements
                                        .iter()
                                        .filter(|stmt| matches!(stmt, crate::core::memory::dsl::DslStatement::Fact(_)))
                                        .count();
                                    let fallback_rel_count = fallback_statements
                                        .iter()
                                        .filter(|stmt| matches!(stmt, crate::core::memory::dsl::DslStatement::Rel(_)))
                                        .count();
                                    let ctx = crate::core::memory::compiler::CompileContext {
                                        pool: kernel.db.pool.clone(),
                                        model_client: Some(kernel.model_client.clone()),
                                        session_id: conversation_id.clone(),
                                        scope: Scope::Global,
                                        source: SourceType::User,
                                        source_ref: Some(run_id.clone()),
                                        now: Utc::now(),
                                        embedding_config: None,
                                        skip_claims: true,
                                        allow_ambiguous_user_refs: true,
                                    };
                                    let fallback_res = crate::core::memory::compiler::compile_parsed(
                                        fallback_statements,
                                        "fallback",
                                        ctx,
                                    )
                                    .await;
                                    let fallback_written = fallback_res.written_ids.len();
                                    let fallback_conflicts = fallback_res.conflict_ids.len();
                                    let _ = system_log::log_event(
                                        &kernel.db.pool,
                                        Some(&kernel.app_handle),
                                        "warn",
                                        "memory",
                                        Some(&run_id),
                                        Some(&trace_id),
                                        json!({
                                            "event": "memory_fallback_used",
                                            "conversation_id": conversation_id,
                                            "trigger_error": result.error,
                                            "fallback_mode": fallback_mode,
                                            "written_ids": fallback_written,
                                            "facts_written": fallback_fact_count,
                                            "relations_written": fallback_rel_count,
                                            "conflict_ids": fallback_conflicts,
                                            "error_count": fallback_res.errors.len(),
                                        }),
                                    )
                                    .await;
                                    if fallback_written > 0 {
                                        memory_pass_result = Some(MemoryPassResult {
                                            success: true,
                                            error: None,
                                            conflict_ids: fallback_res.conflict_ids,
                                            pending_clarify: false,
                                            written_ids: fallback_written,
                                            facts_written: fallback_fact_count,
                                            rels_written: fallback_rel_count,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }

                let eligible = !is_trivial_user_message(&input)
                    && !response_no_tags.trim().is_empty()
                    && !has_tool_calls;
                let memory_pass_success = memory_pass_result
                    .as_ref()
                    .map(|r| r.success && r.written_ids > 0)
                    .unwrap_or(false);
                let facts_written = memory_pass_result.as_ref().map(|r| r.facts_written).unwrap_or(0);
                let rels_written = memory_pass_result.as_ref().map(|r| r.rels_written).unwrap_or(0);
                let memory_pass_false_positive = memory_pass_started
                    && !memory_pass_success
                    && facts_written == 0
                    && rels_written == 0;
                let _ = system_log::log_event(
                    &kernel.db.pool,
                    Some(&kernel.app_handle),
                    "info",
                    "memory",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "memory_yield_sample",
                        "conversation_id": conversation_id,
                        "eligible": eligible,
                        "memory_pass_started": memory_pass_started,
                        "memory_pass_success": memory_pass_success,
                        "memory_pass_false_positive": memory_pass_false_positive,
                        "facts_written": facts_written,
                        "relations_written": rels_written,
                        "candidate_count": candidate_count,
                        "novelty_rejections": novelty_rejections,
                    }),
                )
                .await;

                if memory_pass_started {
                    let mut updated_state = kernel.load_state(&conversation_id).await;
                    updated_state.last_memory_pass_at = Some(Utc::now().to_rfc3339());
                    kernel.persist_state_with_owner(&updated_state, "memory_pass").await;
                }

                if eligible {
                    let rows = sqlx::query(
                        "SELECT payload FROM system_logs
                         WHERE json_extract(payload, '$.event') = 'memory_yield_sample'
                           AND json_extract(payload, '$.conversation_id') = ?
                         ORDER BY timestamp DESC
                         LIMIT 50",
                    )
                    .bind(&conversation_id)
                    .fetch_all(&kernel.db.pool)
                    .await
                    .unwrap_or_default();

                    let mut eligible_count = 0i64;
                    let mut pass_started = 0i64;
                    let mut pass_success = 0i64;
                    let mut pass_false_positive = 0i64;
                    let mut facts_total = 0i64;
                    let mut rels_total = 0i64;
                    let mut candidate_total = 0i64;
                    let mut novelty_rejections_total = 0i64;
                    for row in rows {
                        let raw: String = row.get("payload");
                        if let Ok(value) = serde_json::from_str::<Value>(&raw) {
                            if value.get("eligible").and_then(|v| v.as_bool()).unwrap_or(false) {
                                eligible_count += 1;
                                if value.get("memory_pass_started").and_then(|v| v.as_bool()).unwrap_or(false) {
                                    pass_started += 1;
                                }
                                if value.get("memory_pass_success").and_then(|v| v.as_bool()).unwrap_or(false) {
                                    pass_success += 1;
                                }
                                if value
                                    .get("memory_pass_false_positive")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false)
                                {
                                    pass_false_positive += 1;
                                }
                                facts_total += value.get("facts_written").and_then(|v| v.as_i64()).unwrap_or(0);
                                rels_total += value.get("relations_written").and_then(|v| v.as_i64()).unwrap_or(0);
                                candidate_total += value.get("candidate_count").and_then(|v| v.as_i64()).unwrap_or(0);
                                novelty_rejections_total += value.get("novelty_rejections").and_then(|v| v.as_i64()).unwrap_or(0);
                            }
                        }
                    }
                    if eligible_count > 0 {
                        let trigger_rate = pass_started as f64 / eligible_count as f64;
                        let success_rate = if pass_started > 0 {
                            pass_success as f64 / pass_started as f64
                        } else {
                            0.0
                        };
                        let relation_rate = rels_total as f64 / eligible_count as f64;
                        let false_positive_rate = if pass_started > 0 {
                            pass_false_positive as f64 / pass_started as f64
                        } else {
                            0.0
                        };
                        let novelty_rejection_rate = if candidate_total > 0 {
                            novelty_rejections_total as f64 / candidate_total as f64
                        } else {
                            0.0
                        };
                        let _ = system_log::log_event(
                            &kernel.db.pool,
                            Some(&kernel.app_handle),
                            "info",
                            "memory",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "memory_yield_report",
                                "window_turns": eligible_count,
                                "memory_pass_trigger_rate": trigger_rate,
                                "memory_pass_success_rate": success_rate,
                                "memory_pass_false_positive_rate": false_positive_rate,
                                "memory_pass_false_positive_max": candidate::MEMORY_FALSE_POSITIVE_MAX,
                                "novelty_rejection_rate": novelty_rejection_rate,
                                "relation_rate": relation_rate,
                                "facts_written": facts_total,
                                "relations_written": rels_total,
                            }),
                        )
                        .await;
                    }
                }

                let summary_allowed = MemoryPolicy::is_allowed(
                    MemoryWriteCategory::Summary,
                    MemoryWriteSource::Kernel,
                    "user_visible_turn",
                );
                if summary_allowed {
                    let started_at = Utc::now();
                    kernel.emit_module_status(
                        Some(&run_id),
                        "rolling_summary",
                        "Queued rolling summary update",
                        started_at,
                        0,
                    );
                    match kernel
                        .db
                        .enqueue_post_processing_job_with_priority(
                            "rolling_summary_update",
                            Some(&conversation_id),
                            Some(&run_id),
                            2,
                        )
                        .await
                    {
                        Ok(job_id) => {
                            let _ = kernel.db.set_summary_pending(&conversation_id, true).await;
                            let _ = system_log::log_event(
                                &kernel.db.pool,
                                Some(&kernel.app_handle),
                                "info",
                                "kernel",
                                Some(&run_id),
                                Some(&trace_id),
                                json!({
                                    "event": "post_processing_job_queued",
                                    "job_id": job_id,
                                    "job_type": "rolling_summary_update",
                                    "conversation_id": conversation_id,
                                    "reason": "user_visible_turn",
                                }),
                            )
                            .await;
                        }
                        Err(err) => {
                            let _ = system_log::log_event(
                                &kernel.db.pool,
                                Some(&kernel.app_handle),
                                "warn",
                                "kernel",
                                Some(&run_id),
                                Some(&trace_id),
                                json!({
                                    "event": "post_processing_job_queue_failed",
                                    "job_type": "rolling_summary_update",
                                    "conversation_id": conversation_id,
                                    "reason": "user_visible_turn",
                                    "error": err,
                                }),
                            )
                            .await;
                        }
                    }
                    let duration_ms = Utc::now()
                        .signed_duration_since(started_at)
                        .num_milliseconds()
                        .max(0);
                    kernel.emit_module_status(
                        Some(&run_id),
                        "idle",
                        "Idle",
                        started_at,
                        duration_ms,
                    );
                } else {
                    let _ = system_log::log_event(
                        &kernel.db.pool,
                        Some(&kernel.app_handle),
                        "warn",
                        "memory_policy",
                        Some(&run_id),
                        Some(&trace_id),
                        json!({
                            "event": "memory_policy_violation",
                            "category": "summary",
                            "source": "kernel",
                            "reason_code": "user_visible_turn",
                            "conversation_id": conversation_id,
                        }),
                    )
                    .await;
                }
                },
                )
            };
            if let Some(handle) = queued {
                let mut cancel_rx_bg = cancel_rx.clone();
                tokio::spawn(async move {
                    if *cancel_rx_bg.borrow() {
                        handle.cancel();
                        return;
                    }
                    if cancel_rx_bg.changed().await.is_ok() && *cancel_rx_bg.borrow() {
                        handle.cancel();
                    }
                });
            } else {
                best_effort_dropped = true;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    Some(&log_run_id),
                    Some(&log_trace_id),
                    json!({
                        "event": "post_processing_backpressure",
                        "reason": "queue_full",
                        "conversation_id": log_conversation_id,
                    }),
                )
                .await;
            }
        }

        if let Some(thread_run) = commit_result.thread_run {
            let db = self.db.clone();
            let model_client = self.model_client.clone();
            let app_handle = self.app_handle.clone();
            let settings_snapshot = settings.clone();
            let conversation_id = conversation_id.clone();
            let conversation_id_for_log = conversation_id.clone();
            let run_id = run_id.clone();
            let run_id_for_log = run_id.clone();
            let trace_id_for_log = trace_id.clone();
            let post_processing_mode = system_controls::mode_for(
                "post_processing",
                &system_controls::load_control_map(&self.db).await,
            );
            let queued = if system_controls::mode_is_off(&post_processing_mode)
                || system_controls::mode_is_degraded(&post_processing_mode)
            {
                let reason = if system_controls::mode_is_off(&post_processing_mode) {
                    "system_control_off"
                } else {
                    "system_control_degraded"
                };
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    Some(&run_id_for_log),
                    Some(&trace_id_for_log),
                    json!({
                        "event": "post_processing_backpressure",
                        "reason": reason,
                        "job": "thread_run",
                        "conversation_id": conversation_id_for_log,
                    }),
                )
                .await;
                None
            } else {
                post_processing::enqueue_post_processing_with_priority(
                    post_processing::JobPriority::BestEffort,
                    move || async move {
                        let kernel = Kernel::new(db, model_client, app_handle);
                        let started_at = Utc::now();
                        kernel.emit_module_status(
                            Some(&run_id),
                            "thread_run",
                            "Advancing thread",
                            started_at,
                            0,
                        );
                        let _ = kernel
                            .run_thread(
                                &conversation_id,
                                &thread_run.thread_id,
                                &thread_run.goal,
                                thread_run.depth,
                                &settings_snapshot,
                            )
                            .await;
                        let duration_ms = Utc::now()
                            .signed_duration_since(started_at)
                            .num_milliseconds()
                            .max(0);
                        kernel.emit_module_status(Some(&run_id), "idle", "Idle", started_at, duration_ms);
                    },
                )
            };
            if let Some(handle) = queued {
                let mut cancel_rx_bg = cancel_rx.clone();
                tokio::spawn(async move {
                    if *cancel_rx_bg.borrow() {
                        handle.cancel();
                        return;
                    }
                    if cancel_rx_bg.changed().await.is_ok() && *cancel_rx_bg.borrow() {
                        handle.cancel();
                    }
                });
            } else {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    Some(&run_id_for_log),
                    Some(&trace_id_for_log),
                    json!({
                        "event": "post_processing_backpressure",
                        "reason": "queue_full",
                        "job": "thread_run",
                        "conversation_id": conversation_id_for_log,
                    }),
                )
                .await;
            }
            decision.report.background_jobs_dropped = Some(best_effort_dropped);
        }
        self
            .attach_contract_violation_metrics(&mut decision, Some(&run_id))
            .await;
        self
            .log_decision_report(&decision, Some(&run_id), Some(&trace_id))
            .await;

        self.sync_unified_self_model(&state).await;

        if commit_result.research_cost > 0 {
            let remaining = self.research_budget_remaining(&state, &settings);
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "research_budget_usage",
                    "consumed": commit_result.research_cost,
                    "remaining": remaining,
                    "window_start": state.research_window_start,
                }),
            )
            .await;
        }

        if relational_mode {
            let len = response.as_ref().map(|s| s.len()).unwrap_or(0);
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "relational_response_emitted",
                    "content_len": len,
                }),
            )
            .await;
        }

        if matches!(input_kind, CoreInputKind::User) && settings.monologue_interval_seconds.unwrap_or(0) > 0 {
            if let Some(last_input_at) = state.last_user_input_at.as_deref() {
                if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(last_input_at) {
                    let now = Utc::now();
                    let age_secs = now
                        .signed_duration_since(ts.with_timezone(&Utc))
                        .num_seconds();
                    if age_secs >= 0 && age_secs <= PRIORITY_MONOLOGUE_WINDOW_SECS {
                        let db = self.db.clone();
                        let model_client = self.model_client.clone();
                        let app_handle = self.app_handle.clone();
                        let conversation_id = conversation_id.clone();
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "monologue_priority_scheduled",
                                "conversation_id": conversation_id,
                                "age_secs": age_secs,
                                "window_secs": PRIORITY_MONOLOGUE_WINDOW_SECS,
                            }),
                        )
                        .await;
                        tokio::spawn(async move {
                            let kernel = Kernel::new(db, model_client, app_handle);
                            let _ = kernel.run_monologue_tick(&conversation_id, true).await;
                        });
                    }
                }
            }
        }

        {
            let db = self.db.clone();
            let model_client = self.model_client.clone();
            let app_handle = self.app_handle.clone();
            let conversation_id = conversation_id.clone();
            let run_id = run_id.clone();
            let trace_id = trace_id.clone();
            tokio::spawn(async move {
                let kernel = Kernel::new(db, model_client, app_handle);
                let _ = kernel
                    .auto_surface_pending_prompts_after_response(&conversation_id, &run_id, &trace_id)
                    .await;
            });
        }

        let t_prompt_build_ms = self.fetch_timing_ms(&run_id, "timing_prompt_build").await.unwrap_or(0);
        let t_model_ms = self.fetch_timing_ms(&run_id, "timing_model_call").await.unwrap_or(0);
        let t_mem_retrieval_ms = self.fetch_timing_ms(&run_id, "timing_memory_retrieval").await.unwrap_or(0);
        let total_ms = turn_started.elapsed().as_millis() as i64;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(&run_id),
            Some(&trace_id),
            json!({
                "event": "timing_turn",
                "t_total_ms": total_ms,
                "t_ingest_ms": ingest_ms,
                "t_prompt_build_ms": t_prompt_build_ms,
                "t_llm_prefill_ms": t_model_ms,
                "t_llm_decode_ms": 0,
                "t_gates_ms": gates_ms,
                "t_tool_ms": tool_ms,
                "t_mem_retrieval_ms": t_mem_retrieval_ms,
                "t_mem_write_ms": mem_write_ms,
                "t_db_write_ms": commit_ms,
                "t_emit_ms": emit_ms,
                "t_ui_render_ms": 0,
            }),
        )
        .await;
        let fast_path_stages = [
            ("ingest", ingest_ms, FAST_PATH_BUDGET_INGEST_MS),
            ("prompt_build", t_prompt_build_ms, FAST_PATH_BUDGET_PROMPT_MS),
            ("model_call", t_model_ms, FAST_PATH_BUDGET_MODEL_MS),
            ("response_parse", gates_ms, FAST_PATH_BUDGET_PARSE_MS),
            ("emit", emit_ms, FAST_PATH_BUDGET_EMIT_MS),
        ];
        for (stage, duration_ms, budget_ms) in fast_path_stages {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "perf",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "fast_path_stage_timing",
                    "stage": stage,
                    "duration_ms": duration_ms,
                    "budget_ms": budget_ms,
                    "conversation_id": conversation_id,
                }),
            )
            .await;
            if duration_ms > budget_ms && budget_ms > 0 {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "perf",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "fast_path_budget_violation",
                        "stage": stage,
                        "duration_ms": duration_ms,
                        "budget_ms": budget_ms,
                        "conversation_id": conversation_id,
                    }),
                )
                .await;
            }
        }

        }

        let final_response = strip_protocol_tags_final(&response.unwrap_or_default());
        Ok(RunOutput {
            response: final_response,
            tool_result,
            assistant_message_id,
        })
    }



    pub async fn run_proaction_tick(&self, conversation_id: &str) -> Result<(), String> {
        let _proaction_guard = match self.proaction_lock.try_lock() {
            Ok(guard) => guard,
            Err(_) => return Ok(()),
        };

        let tick_id = Uuid::new_v4().to_string();
        let tick_started = Instant::now();
        let window_minutes = 5;
        let now = Utc::now();

        let mut proaction_state = self.load_proaction_state().await;
        let kernel_state = self.load_state(conversation_id).await;
        if proaction_state.monologue_relaxation_level != kernel_state.monologue_relaxation_level {
            proaction_state.monologue_relaxation_level = kernel_state.monologue_relaxation_level;
        }
        if proaction_state.mode.trim().is_empty() {
            proaction_state.mode = "metrics".to_string();
        }
        let normalized_mode = match proaction_state.mode.as_str() {
            "metrics" | "dry_run" | "active" => proaction_state.mode.clone(),
            _ => "metrics".to_string(),
        };
        proaction_state.mode = normalized_mode.clone();
        let effective_mode = if proaction_state.enabled {
            normalized_mode.as_str()
        } else {
            "metrics"
        };

        let metrics = self.compute_proaction_metrics(window_minutes).await;
        let metrics_payload = serde_json::to_value(&metrics).unwrap_or_else(|_| json!({}));
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            Some(&tick_id),
            json!({
                "event": "proaction_metrics",
                "tick_id": tick_id,
                "conversation_id": conversation_id,
                "mode": effective_mode,
                "enabled": proaction_state.enabled,
                "window_minutes": window_minutes,
                "metrics": metrics_payload,
            }),
        )
        .await;

        if effective_mode == "dry_run" {
            proaction_state.dry_run_completed = true;
        }

        if effective_mode == "metrics" {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "proaction_noop",
                    "reason": if proaction_state.enabled { "metrics_only" } else { "disabled" },
                    "mode": effective_mode,
                    "conversation_id": conversation_id,
                }),
            )
            .await;
            self.persist_proaction_state(&proaction_state).await;
            return Ok(());
        }

        if effective_mode == "active" && !proaction_state.dry_run_completed {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "proaction_noop",
                    "reason": "dry_run_required",
                    "mode": effective_mode,
                    "conversation_id": conversation_id,
                }),
            )
            .await;
            self.persist_proaction_state(&proaction_state).await;
            return Ok(());
        }

        let cooldown_active = proaction_state
            .cooldown_until
            .as_deref()
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
            .map(|ts| now < ts.with_timezone(&Utc))
            .unwrap_or(false);

        if metrics.empty_response_rate > 0.0 || metrics.meta_response_rate > 0.0 {
            let mut settings = self.db.get_settings().await.map_err(|e| e.to_string())?;
            let mut changes = Vec::new();
            if let Some(snapshot) = proaction_state.last_settings_snapshot.clone() {
                changes = Self::apply_settings_snapshot(&mut settings, &snapshot);
                if !changes.is_empty() {
                    let _ = self.db.update_settings(settings).await;
                }
            }
            proaction_state.cooldown_until =
                Some((now + chrono::Duration::minutes(15)).to_rfc3339());
            proaction_state.consecutive_good_windows = 0;
            proaction_state.consecutive_bad_windows = 0;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "proaction_adjustment",
                    "action": "rollback",
                    "reason": "response_integrity_violation",
                    "changes": changes,
                    "empty_response_rate": metrics.empty_response_rate,
                    "meta_response_rate": metrics.meta_response_rate,
                    "cooldown_until": proaction_state.cooldown_until,
                    "conversation_id": conversation_id,
                }),
            )
            .await;
            self.persist_proaction_state(&proaction_state).await;
            return Ok(());
        }

        let settings = self.db.get_settings().await.map_err(|e| e.to_string())?;
        let ask_current = settings.ask_budget_max.unwrap_or(1).clamp(1, 5);
        let loop_current = settings
            .loop_similarity_threshold
            .unwrap_or(0.87)
            .clamp(0.85, 0.92);
        let research_current = settings.research_budget_per_hour.unwrap_or(0).clamp(0, 5);

        let mut adjustments = ProactionAdjustments::default();
        let mut tighten = false;
        let mut loosen = false;

        let has_user_turns = metrics.user_turns > 0;
        if has_user_turns {
            if metrics.ask_loop_rate > 0.25 || metrics.emit_loop_rate > 0.25 {
                let new_ask = (ask_current - 1).max(1);
                if new_ask != ask_current {
                    adjustments.ask_budget_max = Some(new_ask);
                    tighten = true;
                }
                let new_loop = (loop_current - 0.02).max(0.85);
                if (new_loop - loop_current).abs() > f32::EPSILON {
                    adjustments.loop_similarity_threshold = Some(new_loop);
                    tighten = true;
                }
            } else if metrics.ask_loop_rate < 0.05
                && metrics.emit_loop_rate < 0.05
                && metrics.user_visible_output_rate > 0.7
            {
                let new_ask = (ask_current + 1).min(5);
                if new_ask != ask_current {
                    adjustments.ask_budget_max = Some(new_ask);
                    loosen = true;
                }
                let new_loop = (loop_current + 0.01).min(0.92);
                if (new_loop - loop_current).abs() > f32::EPSILON {
                    adjustments.loop_similarity_threshold = Some(new_loop);
                    loosen = true;
                }
            }
        }

        if metrics.tool_calls > 0 {
            if metrics.tool_failure_rate > 0.4
                || metrics.tool_unknown_rate > 0.1
                || metrics.tool_refusal_rate > 0.3
            {
                let new_research = (research_current - 1).max(0);
                if new_research != research_current {
                    adjustments.research_budget_per_hour = Some(new_research);
                    tighten = true;
                }
            } else if metrics.tool_success_rate > 0.8
                && metrics.tool_failure_rate < 0.2
                && metrics.tool_unknown_rate < 0.05
            {
                let new_research = (research_current + 1).min(5);
                if new_research != research_current {
                    adjustments.research_budget_per_hour = Some(new_research);
                    loosen = true;
                }
            }
        }

        let direction = if tighten && !loosen {
            "tighten"
        } else if loosen && !tighten {
            "loosen"
        } else if !tighten && !loosen {
            "hold"
        } else {
            "mixed"
        };

        let adjustments_payload =
            serde_json::to_value(&adjustments).unwrap_or_else(|_| json!({}));
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            Some(&tick_id),
            json!({
                "event": "proaction_decision",
                "mode": effective_mode,
                "direction": direction,
                "adjustments": adjustments_payload,
                "cooldown_active": cooldown_active,
                "ask_budget_current": ask_current,
                "loop_similarity_current": loop_current,
                "research_budget_current": research_current,
                "conversation_id": conversation_id,
            }),
        )
        .await;

        if direction == "mixed" || direction == "hold" {
            proaction_state.consecutive_good_windows = 0;
            proaction_state.consecutive_bad_windows = 0;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "proaction_noop",
                    "reason": direction,
                    "mode": effective_mode,
                    "conversation_id": conversation_id,
                }),
            )
            .await;
            self.persist_proaction_state(&proaction_state).await;
            return Ok(());
        }

        if direction == "tighten" {
            proaction_state.consecutive_bad_windows += 1;
            proaction_state.consecutive_good_windows = 0;
        } else if direction == "loosen" {
            proaction_state.consecutive_good_windows += 1;
            proaction_state.consecutive_bad_windows = 0;
        }

        if effective_mode == "dry_run" {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "proaction_noop",
                    "reason": "dry_run",
                    "mode": effective_mode,
                    "consecutive_good_windows": proaction_state.consecutive_good_windows,
                    "consecutive_bad_windows": proaction_state.consecutive_bad_windows,
                    "conversation_id": conversation_id,
                }),
            )
            .await;
            self.persist_proaction_state(&proaction_state).await;
            return Ok(());
        }

        if cooldown_active {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "proaction_noop",
                    "reason": "cooldown",
                    "mode": effective_mode,
                    "cooldown_until": proaction_state.cooldown_until,
                    "conversation_id": conversation_id,
                }),
            )
            .await;
            self.persist_proaction_state(&proaction_state).await;
            return Ok(());
        }

        let ready = match direction {
            "tighten" => proaction_state.consecutive_bad_windows >= 2,
            "loosen" => proaction_state.consecutive_good_windows >= 2,
            _ => false,
        };

        if !ready {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "proaction_noop",
                    "reason": "waiting_consecutive_window",
                    "mode": effective_mode,
                    "direction": direction,
                    "consecutive_good_windows": proaction_state.consecutive_good_windows,
                    "consecutive_bad_windows": proaction_state.consecutive_bad_windows,
                    "conversation_id": conversation_id,
                }),
            )
            .await;
            self.persist_proaction_state(&proaction_state).await;
            return Ok(());
        }

        let mut updated_settings = settings.clone();
        let changes = Self::apply_proaction_adjustments(&mut updated_settings, &adjustments);
        if changes.is_empty() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "proaction_noop",
                    "reason": "no_effect",
                    "mode": effective_mode,
                    "conversation_id": conversation_id,
                }),
            )
            .await;
            self.persist_proaction_state(&proaction_state).await;
            return Ok(());
        }

        proaction_state.last_settings_snapshot = Some(Self::settings_snapshot(&settings));
        let _ = self.db.update_settings(updated_settings).await;
        proaction_state.last_adjusted_at = Some(now.to_rfc3339());
        proaction_state.cooldown_until =
            Some((now + chrono::Duration::minutes(15)).to_rfc3339());
        proaction_state.consecutive_good_windows = 0;
        proaction_state.consecutive_bad_windows = 0;

        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            Some(&tick_id),
            json!({
                "event": "proaction_adjustment",
                "mode": effective_mode,
                "direction": direction,
                "changes": changes,
                "cooldown_until": proaction_state.cooldown_until,
                "latency_p95_ms": metrics.latency_p95_ms,
                "tick_ms": tick_started.elapsed().as_millis() as i64,
                "conversation_id": conversation_id,
            }),
        )
        .await;

        self.persist_proaction_state(&proaction_state).await;
        Ok(())
    }

    pub async fn run_heartbeat_tick(&self, conversation_id: &str) -> Result<(), String> {
        let settings = self.db.get_settings().await.map_err(|e| e.to_string())?;
        if settings.heartbeat_enabled.unwrap_or(true) == false {
            return Ok(());
        }

        let tick_id = Uuid::new_v4().to_string();
        let tick_started = Instant::now();
        let mut state = self.load_state(conversation_id).await;
        let _ = self.mark_stale_tool_dispatches().await;
        let now = Utc::now();
        let due = state
            .last_heartbeat_at
            .as_deref()
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
            .map(|ts| now.signed_duration_since(ts.with_timezone(&Utc)).num_seconds() >= HEARTBEAT_INTERVAL_SECS)
            .unwrap_or(true);
        if !due {
            let duration_ms = tick_started.elapsed().as_millis() as i64;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "timing_heartbeat_tick",
                    "duration_ms": duration_ms,
                    "conversation_id": conversation_id,
                    "tick_id": tick_id,
                    "outcome": "not_due",
                }),
            )
            .await;
            return Ok(());
        }

        self.refresh_controller_state(&mut state, &settings).await;

        // Refresh salience deterministically for active beliefs (working set first)
        let rows = sqlx::query(
            "SELECT item_id FROM ics_working_set WHERE item_type = 'belief' ORDER BY activation DESC LIMIT 20",
        )
        .fetch_all(&self.db.pool)
        .await
        .unwrap_or_default();
        let mut belief_ids = Vec::new();
        for row in rows {
            let belief_id: i64 = row.get("item_id");
            belief_ids.push(belief_id);
        }
        if !belief_ids.is_empty() {
            let _ = crate::core::memory::attention::salience::recompute_salience_for_beliefs(
                &self.db.pool,
                &belief_ids,
            )
            .await;
        } else {
            let _ = crate::core::memory::attention::salience::recompute_salience_for_all(&self.db.pool)
                .await;
        }

        // Scan for self-claim contradictions and re-anchor if needed
        let _ = self_claims::scan_and_reanchor(&self.db, conversation_id).await;

        state.last_heartbeat_at = Some(now.to_rfc3339());
        let disable_working_hypothesis = settings.stability_disable_working_hypothesis.unwrap_or(true);
        refresh_working_memory(&mut state, now, disable_working_hypothesis);
        self.persist_state(&state).await;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            None,
            json!({
                "event": "heartbeat_tick",
                "tick_id": tick_id,
                "conversation_id": conversation_id,
            }),
        )
        .await;
        let duration_ms = tick_started.elapsed().as_millis() as i64;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            Some(&tick_id),
            json!({
                "event": "timing_heartbeat_tick",
                "duration_ms": duration_ms,
                "conversation_id": conversation_id,
                "tick_id": tick_id,
                "outcome": "completed",
            }),
        )
        .await;
        Ok(())
    }

    pub(crate) async fn run_tool_heartbeat_tick(&self) -> Result<(), String> {
        let action_id = Uuid::new_v4().to_string();
        let dispatch = ToolDispatchRequest {
            action_id: action_id.clone(),
            tool_name: "get_current_time".to_string(),
            args_json: json!({ "timezone": "UTC" }).to_string(),
            plan_step_id: None,
        };
        let (_tx, mut cancel_rx) = watch::channel(false);
        let _ = self
            .dispatch_tool(&dispatch, None, None, &mut cancel_rx)
            .await;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "tool",
            None,
            None,
            json!({
                "event": "tool_heartbeat_dispatch",
                "tool_name": dispatch.tool_name,
                "action_id": action_id,
            }),
        )
        .await;
        Ok(())
    }

    pub(crate) async fn run_memory_pass_tick(&self, conversation_id: &str) -> Result<(), String> {
        let settings = self.db.get_settings().await.map_err(|e| e.to_string())?;
        let assistant_row = sqlx::query(
            "SELECT content, created_at FROM messages
             WHERE conversation_id = ?
               AND role = 'assistant'
               AND status = 'complete'
               AND (metadata IS NULL OR json_extract(metadata, '$.source') != 'monologue')
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.db.pool)
        .await
        .map_err(|e| e.to_string())?;
        let Some(row) = assistant_row else {
            return Ok(());
        };
        let assistant_content: String = row.try_get("content").unwrap_or_default();
        let assistant_created_at: String = row.try_get("created_at").unwrap_or_default();
        if assistant_content.trim().is_empty() {
            return Ok(());
        }

        let user_content: Option<String> = sqlx::query_scalar(
            "SELECT content FROM messages
             WHERE conversation_id = ?
               AND role = 'user'
               AND status = 'complete'
               AND (metadata IS NULL OR json_extract(metadata, '$.source') != 'monologue')
               AND datetime(created_at) <= datetime(?)
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .bind(conversation_id)
        .bind(&assistant_created_at)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten();
        let Some(user_content) = user_content else {
            return Ok(());
        };
        if user_content.trim().is_empty() {
            return Ok(());
        }

        let run_id = format!("memory_tick:{}", Uuid::new_v4());
        let _ = self
            .db
            .create_memory_pass_token(&run_id, conversation_id, 600)
            .await;
        let _ = self
            .run_memory_pass(&settings, &run_id, &user_content, &assistant_content, false)
            .await;
        Ok(())
    }

    pub async fn run_dream_cycle(&self, conversation_id: &str) -> Result<(), String> {
        let settings = self.db.get_settings().await.map_err(|e| e.to_string())?;
        if settings.dream_enabled.unwrap_or(true) == false {
            return Ok(());
        }

        let tick_id = Uuid::new_v4().to_string();
        let tick_started = Instant::now();
        let mut state = self.load_state(conversation_id).await;
        let now = Utc::now();
        let due = state
            .last_dream_at
            .as_deref()
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
            .map(|ts| now.signed_duration_since(ts.with_timezone(&Utc)).num_seconds() >= DREAM_INTERVAL_SECS)
            .unwrap_or(true);
        if !due {
            let duration_ms = tick_started.elapsed().as_millis() as i64;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "timing_dream_cycle",
                    "duration_ms": duration_ms,
                    "conversation_id": conversation_id,
                    "tick_id": tick_id,
                    "outcome": "not_due",
                }),
            )
            .await;
            return Ok(());
        }

        // Ensure idle window
        let idle_minutes: Option<f64> = sqlx::query_scalar(
            "SELECT (julianday('now') - julianday(created_at)) * 24 * 60
             FROM messages
             WHERE conversation_id = ?
               AND role IN ('user', 'assistant')
               AND (metadata IS NULL OR json_extract(metadata, '$.source') != 'monologue')
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten();
        if let Some(idle) = idle_minutes {
            if idle < DREAM_IDLE_MINUTES as f64 {
                let duration_ms = tick_started.elapsed().as_millis() as i64;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    Some(&tick_id),
                    json!({
                        "event": "timing_dream_cycle",
                        "duration_ms": duration_ms,
                        "conversation_id": conversation_id,
                        "tick_id": tick_id,
                        "outcome": "idle_window",
                    }),
                )
                .await;
                return Ok(());
            }
        }

        let entries = self
            .db
            .list_inner_monologue_entries(conversation_id, 24)
            .await
            .unwrap_or_default();
        if entries.is_empty() {
            let duration_ms = tick_started.elapsed().as_millis() as i64;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "timing_dream_cycle",
                    "duration_ms": duration_ms,
                    "conversation_id": conversation_id,
                    "tick_id": tick_id,
                    "outcome": "no_entries",
                }),
            )
            .await;
            return Ok(());
        }
        let dialogue = entries
            .iter()
            .rev()
            .map(|e| format!("{}: {}", e.speaker.clone().unwrap_or_else(|| "self".to_string()), summarize_snippet(&e.thought, 200)))
            .collect::<Vec<_>>()
            .join("\n");

        let (summary_model, summary_url) = select_summary_model(&settings);
        let system_prompt = "Summarize the internal self-dialogue into 3-5 concise insights. Be factual, avoid speculation. Do not address the user.";
        let user_prompt = format!("Internal self-dialogue:\n{}\n\nReturn the summary only.", dialogue);
        let (user_prompt, prompt_truncated) = cap_summary_prompt(system_prompt, &user_prompt, &settings);
        if prompt_truncated {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "summary",
                None,
                None,
                json!({
                    "event": "summary_prompt_capped",
                    "cap_tokens": summary_prompt_cap_tokens(&settings),
                    "source": "dream_cycle",
                }),
            )
            .await;
        }

        let request = ChatCompletionRequest {
            model: summary_model,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            stream: settings.streaming_enabled,
            temperature: None,
            top_p: None,
            max_tokens: Some(300),
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
            json_strict: Some(false),
                skip_sanitization: None,
            run_id: None,
            request_label: Some("dream_consolidation_summary".to_string()),
        };

        if let Ok(resp) = self
            .model_client
            .chat_with_meta(&summary_url, settings.api_key.as_deref(), &request)
            .await
        {
            let summary = resp.content.trim().to_string();
            if !summary.is_empty() {
                let _ = crate::core::episodic::emit_episodic_event(
                    &self.db.pool,
                    "dream_consolidation",
                    json!({ "status": "ok", "summary_snippet": summarize_snippet(&summary, 240) }),
                    None,
                    None,
                    Some(conversation_id),
                    None,
                    "system",
                    Some("dream_cycle"),
                    None,
                    None,
                )
                .await;
            }
        }

        state.last_dream_at = Some(now.to_rfc3339());
        let disable_working_hypothesis = settings.stability_disable_working_hypothesis.unwrap_or(true);
        refresh_working_memory(&mut state, now, disable_working_hypothesis);
        self.persist_state(&state).await;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            None,
            json!({
                "event": "dream_cycle",
                "tick_id": tick_id,
                "conversation_id": conversation_id,
            }),
        )
        .await;
        let duration_ms = tick_started.elapsed().as_millis() as i64;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            Some(&tick_id),
            json!({
                "event": "timing_dream_cycle",
                "duration_ms": duration_ms,
                "conversation_id": conversation_id,
                "tick_id": tick_id,
                "outcome": "completed",
            }),
        )
        .await;
        Ok(())
    }

    pub(super) async fn emit_internal_message(&self, conversation_id: &str, content: &str) -> Result<(), String> {
        let formatted = format_monologue_surface_content(content);
        if formatted.trim().is_empty() {
            return Ok(());
        }

        let message_id = Uuid::new_v4().to_string();
        let surface_run_id = format!("monologue:{}", Uuid::new_v4());
        let metadata = json!({
            "source": "monologue",
            "origin": "monologue",
            "surface": false,
            "response_origin": ResponseOrigin::Primary.as_str(),
        })
        .to_string();
        sqlx::query(
            "INSERT INTO messages (message_id, conversation_id, run_id, trace_id, role, content, status, created_at, metadata)
             VALUES (?, ?, ?, NULL, 'internal', ?, 'complete', ?, ?)"
        )
        .bind(&message_id)
        .bind(conversation_id)
        .bind(&surface_run_id)
        .bind(&formatted)
        .bind(Utc::now())
        .bind(metadata)
        .execute(&self.db.pool)
        .await
        .map_err(|e| e.to_string())?;

        let _ = self.app_handle.emit("message_updated", ());

        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            None,
            json!({
                "event": "internal_emit",
                "conversation_id": conversation_id,
                "content_len": content.len(),
            }),
        )
        .await;

        Ok(())
    }

    pub(super) async fn emit_proactive_message(
        &self,
        conversation_id: &str,
        content: &str,
        run_id: &str,
        trace_id: &str,
        metadata: serde_json::Value,
    ) -> Result<String, String> {
        if content.trim().is_empty() {
            return Err("empty_proactive_content".to_string());
        }
        let message_id = Uuid::new_v4().to_string();
        let mut meta_value = metadata;
        if !meta_value.is_object() {
            meta_value = json!({});
        }
        if let Some(obj) = meta_value.as_object_mut() {
            obj.entry("response_origin".to_string())
                .or_insert(json!(ResponseOrigin::Primary.as_str()));
        }
        let metadata = serde_json::to_string(&meta_value).unwrap_or_else(|_| "{}".to_string());
        sqlx::query(
            "INSERT INTO messages (message_id, conversation_id, run_id, trace_id, role, content, status, created_at, metadata)
             VALUES (?, ?, ?, ?, 'assistant', ?, 'complete', ?, ?)",
        )
        .bind(&message_id)
        .bind(conversation_id)
        .bind(run_id)
        .bind(trace_id)
        .bind(content)
        .bind(Utc::now())
        .bind(metadata)
        .execute(&self.db.pool)
        .await
        .map_err(|e| e.to_string())?;

        let _ = self.app_handle.emit("assistant_message", content.to_string());
        let _ = self.app_handle.emit("message_updated", ());

        Ok(message_id)
    }

    pub(super) fn emit_module_status(
        &self,
        run_id: Option<&str>,
        stage: &str,
        detail: &str,
        started_at: DateTime<Utc>,
        duration_ms: i64,
    ) {
        let payload = json!({
            "event": "module_status",
            "run_id": run_id,
            "stage": stage,
            "detail": detail,
            "started_at": started_at.to_rfc3339(),
            "duration_ms": duration_ms,
        });
        let _ = self.app_handle.emit("module_status", payload);
    }

    pub(super) async fn apply_proactive_workspace_compliance(
        &self,
        decision: &mut KernelDecision,
        state: &KernelState,
    ) {
        if !workspace_required(state) {
            return;
        }
        for candidate in decision.accepted.iter_mut() {
            if !matches!(
                candidate.kind,
                CandidateKind::EmitMessage | CandidateKind::AskUserQuestion | CandidateKind::FlagForHuman
            ) {
                continue;
            }
            let Some(text) = candidate_alignment_text(candidate) else {
                continue;
            };
            if proactive_response_compliant(&text, state) {
                continue;
            }
            let ack_block = crate::core::kernel::workspace::workspace_ack_block(state);
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "workspace_ack_suppressed",
                    "reason": "proactive_compliance",
                    "candidate_id": candidate.id,
                    "candidate_kind": format!("{:?}", candidate.kind),
                    "ack": ack_block,
                }),
            )
            .await;
        }
    }

    pub(super) async fn apply_workspace_update_with_policy(
        &self,
        state: &mut KernelState,
        candidate: &Candidate,
        disable_working_hypothesis: bool,
    ) -> bool {
        let mut payload = candidate.payload.clone();
        let mut evidence_event_ids = extract_id_list(&payload, "evidence_event_ids");
        let mut belief_ids = extract_id_list(&payload, "belief_ids");

        let payload_has_ids = !evidence_event_ids.is_empty() || !belief_ids.is_empty();
        let mut validation = if payload_has_ids {
            self.validate_evidence_ids(&evidence_event_ids, &belief_ids, false).await
        } else {
            ValidationResult::default()
        };

        if !validation.invalid_evidence_ids.is_empty() || !validation.invalid_belief_ids.is_empty() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "evidence_validation_failed",
                    "candidate_id": candidate.id,
                    "invalid_evidence_ids": validation.invalid_evidence_ids,
                    "invalid_belief_ids": validation.invalid_belief_ids,
                }),
            )
            .await;
        }

        if payload_has_ids {
            evidence_event_ids = validation.valid_evidence_ids.clone();
            belief_ids = validation.valid_belief_ids.clone();
            set_id_list(&mut payload, "evidence_event_ids", &evidence_event_ids);
            set_id_list(&mut payload, "belief_ids", &belief_ids);
        }

        let mut speculative = payload
            .get("speculative")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if !validation.evidence_ok() {
            if let Some(focus) = payload.get("current_focus").and_then(|v| v.as_str()) {
                let user_input = state.last_user_input.as_deref().unwrap_or("");
                if user_focus_signal(user_input, focus) {
                    if let Some(event_id) = self
                        .create_user_focus_evidence_event(
                            &state.conversation_id,
                            focus,
                            user_input,
                            None,
                            None,
                        )
                        .await
                    {
                        evidence_event_ids.push(event_id);
                        validation = self
                            .validate_evidence_ids(&evidence_event_ids, &belief_ids, false)
                            .await;
                        set_id_list(&mut payload, "evidence_event_ids", &evidence_event_ids);
                    }
                }
            }
        }

        let has_factual_fields = payload.get("current_focus").is_some()
            || payload.get("focus_rationale").is_some()
            || payload.get("goal_thread").is_some()
            || payload.get("working_set_topics").is_some();
        let has_hypotheses = payload.get("active_hypotheses").is_some();

        if !validation.evidence_ok() && (has_factual_fields || has_hypotheses) {
            speculative = true;
        }

        if speculative {
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("speculative".to_string(), Value::Bool(true));
                if let Some(rationale) = obj.get("focus_rationale").and_then(|v| v.as_str()) {
                    let prefixed = format_speculative_label(rationale, disable_working_hypothesis);
                    obj.insert("focus_rationale".to_string(), Value::String(prefixed));
                }
            }
            if let Some(value) = payload.get_mut("active_hypotheses") {
                normalize_hypotheses_payload(value, true, &evidence_event_ids, &belief_ids);
            }

            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "speculation_marked",
                    "candidate_id": candidate.id,
                    "reason": "workspace_update",
                }),
            )
            .await;
        }

        if let Some(value) = payload.get_mut("active_hypotheses") {
            normalize_hypotheses_payload(value, speculative, &evidence_event_ids, &belief_ids);
            if let Ok(mut hypotheses) = serde_json::from_value::<Vec<WorkspaceHypothesis>>(value.clone()) {
                let mut invalid_found = false;
                for hypothesis in hypotheses.iter_mut() {
                    let hyp_validation = if !hypothesis.evidence_event_ids.is_empty() || !hypothesis.belief_ids.is_empty() {
                        self.validate_evidence_ids(&hypothesis.evidence_event_ids, &hypothesis.belief_ids, false).await
                    } else {
                        ValidationResult::default()
                    };
                    if !hyp_validation.invalid_evidence_ids.is_empty() || !hyp_validation.invalid_belief_ids.is_empty() {
                        invalid_found = true;
                    }
                    hypothesis.evidence_event_ids = hyp_validation.valid_evidence_ids.clone();
                    hypothesis.belief_ids = hyp_validation.valid_belief_ids.clone();
                    if !hyp_validation.evidence_ok() {
                        hypothesis.speculative = true;
                    }
                }
                if invalid_found {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "evidence_validation_failed",
                            "candidate_id": candidate.id,
                            "reason": "hypotheses_invalid_ids",
                        }),
                    )
                    .await;
                }
                *value = json!(hypotheses);
            }
        }

        if let Some(value) = payload.get_mut("goal_stack") {
            normalize_goal_stack_payload(value);
            let mut items = extract_goal_stack(value);
            let mut invalid_found = false;
            for item in items.iter_mut() {
                let item_validation = if !item.evidence_event_ids.is_empty() || !item.belief_ids.is_empty() {
                    self.validate_evidence_ids(&item.evidence_event_ids, &item.belief_ids, false).await
                } else {
                    ValidationResult::default()
                };
                if !item_validation.invalid_evidence_ids.is_empty() || !item_validation.invalid_belief_ids.is_empty() {
                    invalid_found = true;
                }
                item.evidence_event_ids = item_validation.valid_evidence_ids.clone();
                item.belief_ids = item_validation.valid_belief_ids.clone();

                for step in item.steps.iter_mut() {
                    let step_validation = if !step.evidence_event_ids.is_empty() || !step.belief_ids.is_empty() {
                        self.validate_evidence_ids(&step.evidence_event_ids, &step.belief_ids, false).await
                    } else {
                        ValidationResult::default()
                    };
                    if !step_validation.invalid_evidence_ids.is_empty() || !step_validation.invalid_belief_ids.is_empty() {
                        invalid_found = true;
                    }
                    step.evidence_event_ids = step_validation.valid_evidence_ids.clone();
                    step.belief_ids = step_validation.valid_belief_ids.clone();
                }
            }

            if invalid_found {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "evidence_validation_failed",
                        "candidate_id": candidate.id,
                        "reason": "goal_stack_invalid_ids",
                    }),
                )
                .await;
            }

            let goal_stack_evidence =
                !evidence_event_ids.is_empty() || !belief_ids.is_empty() || goal_stack_has_evidence(&items);
            let advancing = goal_stack_advances(&state.workspace_goal_stack, &items);
            if advancing && !goal_stack_evidence {
                let changed = sanitize_goal_stack_advancement(&state.workspace_goal_stack, &mut items);
                if changed {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "goal_stack_advancement_blocked",
                            "candidate_id": candidate.id,
                            "reason": "missing_evidence",
                        }),
                    )
                    .await;
                }
            } else if advancing && (!evidence_event_ids.is_empty() || !belief_ids.is_empty()) {
                merge_goal_stack_evidence(&mut items, &evidence_event_ids, &belief_ids);
            }

            *value = json!(items);
        }

        update_workspace_meta_from_payload(
            state,
            &payload,
            speculative,
            &evidence_event_ids,
            &belief_ids,
        );
        let mut changed = apply_workspace_update(state, &payload);
        if link_goal_step_to_hypotheses(
            &mut state.workspace_goal_stack,
            &state.workspace_active_hypotheses,
        ) {
            changed = true;
        }
        ensure_workspace_meta_alignment(state);
        changed
    }

    pub(super) async fn create_user_focus_evidence_event(
        &self,
        conversation_id: &str,
        focus_text: &str,
        snippet: &str,
        run_id: Option<&str>,
        trace_id: Option<&str>,
    ) -> Option<i64> {
        let focus = focus_text.trim();
        if focus.is_empty() {
            return None;
        }

        let mut assistant_id: Option<i64> = sqlx::query_scalar(
            "SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = 'assistant' LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten();

        if assistant_id.is_none() {
            assistant_id = sqlx::query_scalar(
                "SELECT entity_id FROM ics_session_bindings WHERE session_id = 'default' AND ref_text = 'assistant' LIMIT 1",
            )
            .fetch_optional(&self.db.pool)
            .await
            .ok()
            .flatten();
        }

        let Some(subject_id) = assistant_id else {
            return None;
        };

        let scope_str = serde_json::to_string(&Scope::SelfScope).unwrap_or_else(|_| "\"self\"".to_string());
        let key = "workspace_focus";
        let value_hash = compute_value_hash(focus);
        let topic_key = compute_topic_key_fact(subject_id, key);
        let sig_inputs = vec![
            ("subject_id".to_string(), subject_id.to_string()),
            ("key".to_string(), key.to_string()),
            ("value_hash".to_string(), value_hash.clone()),
            ("scope".to_string(), scope_str.clone()),
            ("time_bucket_kind".to_string(), "atemporal".to_string()),
            ("time_bucket_value".to_string(), "".to_string()),
            ("polarity".to_string(), "assert".to_string()),
        ];
        let sig_refs: Vec<(&str, &str)> = sig_inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let signature_hash = compute_signature_hash(&sig_refs);

        let mut belief_id: Option<i64> = None;
        let mut existing_weight: f64 = 0.0;
        let mut existing_confidence: f64 = 1.0;
        if let Ok(row) = sqlx::query(
            "SELECT id, evidence_weight_total, confidence FROM ics_beliefs
             WHERE signature_hash = ? AND scope = ? AND polarity = 'assert' AND status = 'active' LIMIT 1",
        )
        .bind(&signature_hash)
        .bind(&scope_str)
        .fetch_optional(&self.db.pool)
        .await
        {
            if let Some(row) = row {
                belief_id = Some(row.get("id"));
                existing_weight = row.try_get::<f64, _>("evidence_weight_total").unwrap_or(0.0);
                existing_confidence = row.try_get::<f64, _>("confidence").unwrap_or(1.0);
            }
        }

        let weight = compute_evidence_weight(SourceType::User) as f64;

        if belief_id.is_none() {
            let inserted = sqlx::query(
                "INSERT INTO ics_beliefs
                 (kind, scope, polarity, layer, topic_key, signature_hash, evidence_weight_total, confidence, time_bucket_kind, time_bucket_value, observed_at, created_at)
                 VALUES ('fact', ?, 'assert', 'episodic', ?, ?, ?, ?, 'atemporal', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                 RETURNING id",
            )
            .bind(&scope_str)
            .bind(&topic_key)
            .bind(&signature_hash)
            .bind(weight)
            .bind(1.0)
            .fetch_one(&self.db.pool)
            .await
            .ok();

            if let Some(row) = inserted {
                let id: i64 = row.get("id");
                belief_id = Some(id);
                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO ics_fact_beliefs (belief_id, subject_entity_id, key, value_literal, value_hash)
                     VALUES (?, ?, ?, ?, ?)",
                )
                .bind(id)
                .bind(subject_id)
                .bind(key)
                .bind(focus)
                .bind(&value_hash)
                .execute(&self.db.pool)
                .await;
            }
        }

        let Some(belief_id) = belief_id else {
            return None;
        };

        let snippet = summarize_snippet(snippet, 180);
        let event_row = sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id)
             VALUES (?, 'user_focus', 'workspace_focus', ?, ?, NULL)
             RETURNING id",
        )
        .bind(belief_id)
        .bind(&snippet)
        .bind(weight)
        .fetch_one(&self.db.pool)
        .await
        .ok();
        let event_id = event_row.map(|row| row.get::<i64, _>("id"));

        let new_weight = existing_weight + weight;
        let new_confidence = existing_confidence.max(1.0);
        let _ = sqlx::query(
            "UPDATE ics_beliefs SET evidence_weight_total = ?, confidence = ?, last_evidence_at = CURRENT_TIMESTAMP WHERE id = ?",
        )
        .bind(new_weight)
        .bind(new_confidence)
        .bind(belief_id)
        .execute(&self.db.pool)
        .await;

        if let Some(event_id) = event_id {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                run_id,
                trace_id,
                json!({
                    "event": "workspace_evidence_event_created",
                    "belief_id": belief_id,
                    "evidence_event_id": event_id,
                    "source_type": "user_focus",
                }),
            )
            .await;
        }

        event_id
    }

    async fn seed_self_scope_fact(
        &self,
        subject_id: i64,
        scope_str: &str,
        key: &str,
        value: &str,
        source_ref: &str,
        run_id: Option<&str>,
        trace_id: Option<&str>,
    ) -> Option<i64> {
        let value = value.trim();
        if value.is_empty() {
            return None;
        }
        let value_hash = compute_value_hash(value);
        let mut belief_id: Option<i64> = None;
        let mut existing_weight: f64 = 0.0;
        let mut existing_confidence: f64 = 0.7;
        let mut has_recent_evidence = false;

        if let Ok(row) = sqlx::query(
            "SELECT b.id, b.evidence_weight_total, b.confidence, b.last_evidence_at
             FROM ics_beliefs b
             JOIN ics_fact_beliefs f ON f.belief_id = b.id
             WHERE b.scope = ?
               AND b.status = 'active'
               AND f.subject_entity_id = ?
               AND f.key = ?
               AND f.value_hash = ?
             LIMIT 1",
        )
        .bind(scope_str)
        .bind(subject_id)
        .bind(key)
        .bind(&value_hash)
        .fetch_optional(&self.db.pool)
        .await
        {
            if let Some(row) = row {
                belief_id = Some(row.get("id"));
                existing_weight = row.try_get::<f64, _>("evidence_weight_total").unwrap_or(0.0);
                existing_confidence = row.try_get::<f64, _>("confidence").unwrap_or(0.7);
                let last_evidence_at: Option<String> = row.try_get("last_evidence_at").ok();
                has_recent_evidence = last_evidence_at
                    .as_deref()
                    .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
                    .map(|ts| {
                        let age = Utc::now().signed_duration_since(ts.with_timezone(&Utc)).num_days();
                        age >= 0 && age <= 7
                    })
                    .unwrap_or(false);
            }
        }

        if belief_id.is_none() {
            let conflict: Option<i64> = sqlx::query_scalar(
                "SELECT b.id
                 FROM ics_beliefs b
                 JOIN ics_fact_beliefs f ON f.belief_id = b.id
                 WHERE b.scope = ?
                   AND b.status = 'active'
                   AND f.subject_entity_id = ?
                   AND f.key = ?
                 LIMIT 1",
            )
            .bind(scope_str)
            .bind(subject_id)
            .bind(key)
            .fetch_optional(&self.db.pool)
            .await
            .ok()
            .flatten();
            if conflict.is_some() {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "memory",
                    run_id,
                    trace_id,
                    json!({
                        "event": "self_scope_seed_skipped",
                        "reason": "conflicting_value",
                        "key": key,
                    }),
                )
                .await;
                return None;
            }

            let topic_key = compute_topic_key_fact(subject_id, key);
            let sig_inputs = vec![
                ("subject_id".to_string(), subject_id.to_string()),
                ("key".to_string(), key.to_string()),
                ("value_hash".to_string(), value_hash.clone()),
                ("scope".to_string(), scope_str.to_string()),
                ("time_bucket_kind".to_string(), "atemporal".to_string()),
                ("time_bucket_value".to_string(), "".to_string()),
                ("polarity".to_string(), "assert".to_string()),
            ];
            let sig_refs: Vec<(&str, &str)> = sig_inputs
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            let signature_hash = compute_signature_hash(&sig_refs);
            let weight = compute_evidence_weight(SourceType::System) as f64;
            let inserted = sqlx::query(
                "INSERT INTO ics_beliefs
                 (kind, scope, polarity, layer, topic_key, signature_hash, evidence_weight_total, confidence, time_bucket_kind, time_bucket_value, observed_at, created_at)
                 VALUES ('fact', ?, 'assert', 'working', ?, ?, ?, ?, 'atemporal', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                 RETURNING id",
            )
            .bind(scope_str)
            .bind(&topic_key)
            .bind(&signature_hash)
            .bind(weight)
            .bind(existing_confidence)
            .fetch_one(&self.db.pool)
            .await
            .ok();
            if let Some(row) = inserted {
                let id: i64 = row.get("id");
                belief_id = Some(id);
                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO ics_fact_beliefs (belief_id, subject_entity_id, key, value_literal, value_hash)
                     VALUES (?, ?, ?, ?, ?)",
                )
                .bind(id)
                .bind(subject_id)
                .bind(key)
                .bind(value)
                .bind(&value_hash)
                .execute(&self.db.pool)
                .await;
            }
        }

        let Some(belief_id) = belief_id else {
            return None;
        };
        if has_recent_evidence {
            return None;
        }

        let weight = compute_evidence_weight(SourceType::System) as f64;
        let snippet = summarize_snippet(&format!("Self-scope seed: {} = {}", key, value), 180);
        let event_row = sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id)
             VALUES (?, 'system', ?, ?, ?, NULL)
             RETURNING id",
        )
        .bind(belief_id)
        .bind(source_ref)
        .bind(&snippet)
        .bind(weight)
        .fetch_one(&self.db.pool)
        .await
        .ok();
        let event_id = event_row.map(|row| row.get::<i64, _>("id"));

        let new_weight = existing_weight + weight;
        let new_confidence = existing_confidence.max(0.7);
        let _ = sqlx::query(
            "UPDATE ics_beliefs
             SET evidence_weight_total = ?, confidence = ?, last_evidence_at = CURRENT_TIMESTAMP
             WHERE id = ?",
        )
        .bind(new_weight)
        .bind(new_confidence)
        .bind(belief_id)
        .execute(&self.db.pool)
        .await;

        if let Some(event_id) = event_id {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "memory",
                run_id,
                trace_id,
                json!({
                    "event": "self_scope_seeded",
                    "belief_id": belief_id,
                    "evidence_event_id": event_id,
                    "key": key,
                }),
            )
            .await;
        }

        event_id
    }

    async fn seed_self_scope_beliefs(
        &self,
        conversation_id: &str,
        model: &crate::models::SelfModel,
        run_id: Option<&str>,
        trace_id: Option<&str>,
    ) -> Vec<i64> {
        let mut assistant_id: Option<i64> = sqlx::query_scalar(
            "SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = 'assistant' LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten();

        if assistant_id.is_none() {
            assistant_id = sqlx::query_scalar(
                "SELECT entity_id FROM ics_session_bindings WHERE session_id = 'default' AND ref_text = 'assistant' LIMIT 1",
            )
            .fetch_optional(&self.db.pool)
            .await
            .ok()
            .flatten();
        }

        let Some(subject_id) = assistant_id else {
            return Vec::new();
        };

        let scope_str = serde_json::to_string(&Scope::SelfScope).unwrap_or_else(|_| "\"self\"".to_string());
        let persona = model.persona.as_object().cloned().unwrap_or_default();
        let mut evidence_ids = Vec::new();
        let persona_fields = [
            ("persona.tone", "tone"),
            ("persona.verbosity", "verbosity"),
            ("persona.directness", "directness"),
            ("persona.formality", "formality"),
            ("persona.initiative", "initiative"),
        ];
        for (key, axis) in persona_fields.iter() {
            let value = persona.get(*axis).and_then(|v| {
                if let Some(num) = v.as_f64() {
                    Some(format!("{:.2}", num.clamp(0.0, 1.0)))
                } else {
                    v.as_str().map(|s| s.trim().to_string())
                }
            });
            let Some(value) = value.filter(|s| !s.trim().is_empty()) else { continue; };
            if let Some(event_id) = self
                .seed_self_scope_fact(
                    subject_id,
                    &scope_str,
                    key,
                    &value,
                    "self_scope_seed",
                    run_id,
                    trace_id,
                )
                .await
            {
                evidence_ids.push(event_id);
            }
        }

        if let Some(goal) = model.goals.as_str() {
            let goal = goal.trim();
            if !goal.is_empty() {
                if let Some(event_id) = self
                    .seed_self_scope_fact(
                        subject_id,
                        &scope_str,
                        "goal",
                        goal,
                        "self_scope_seed",
                        run_id,
                        trace_id,
                    )
                    .await
                {
                    evidence_ids.push(event_id);
                }
            }
        } else if let Some(goals) = model.goals.as_array() {
            for goal in goals.iter().filter_map(|g| g.as_str()).take(3) {
                let goal = goal.trim();
                if goal.is_empty() {
                    continue;
                }
                if let Some(event_id) = self
                    .seed_self_scope_fact(
                        subject_id,
                        &scope_str,
                        "goal",
                        goal,
                        "self_scope_seed",
                        run_id,
                        trace_id,
                    )
                    .await
                {
                    evidence_ids.push(event_id);
                }
            }
        }

        evidence_ids
    }

    pub(super) async fn backfill_workspace_meta_invalid_evidence(&self, state: &mut KernelState) -> bool {
        let mut changed = false;
        let mut demoted_fields: Vec<String> = Vec::new();

        let mut demote_field = |label: &str, meta: &mut WorkspaceFieldMeta, invalid: bool| {
            if invalid {
                meta.speculative = true;
                demoted_fields.push(label.to_string());
                changed = true;
            }
        };

        if let Some(meta) = state.workspace_meta.goal_thread.as_mut() {
            let validation = self.validate_evidence_ids(&meta.evidence_event_ids, &meta.belief_ids, false).await;
            if !validation.invalid_evidence_ids.is_empty() || !validation.invalid_belief_ids.is_empty() {
                meta.evidence_event_ids = validation.valid_evidence_ids;
                meta.belief_ids = validation.valid_belief_ids;
                demote_field("goal_thread", meta, true);
            }
        }
        if let Some(meta) = state.workspace_meta.current_focus.as_mut() {
            let validation = self.validate_evidence_ids(&meta.evidence_event_ids, &meta.belief_ids, false).await;
            if !validation.invalid_evidence_ids.is_empty() || !validation.invalid_belief_ids.is_empty() {
                meta.evidence_event_ids = validation.valid_evidence_ids;
                meta.belief_ids = validation.valid_belief_ids;
                demote_field("current_focus", meta, true);
            }
        }
        if let Some(meta) = state.workspace_meta.focus_rationale.as_mut() {
            let validation = self.validate_evidence_ids(&meta.evidence_event_ids, &meta.belief_ids, false).await;
            if !validation.invalid_evidence_ids.is_empty() || !validation.invalid_belief_ids.is_empty() {
                meta.evidence_event_ids = validation.valid_evidence_ids;
                meta.belief_ids = validation.valid_belief_ids;
                demote_field("focus_rationale", meta, true);
            }
        }

        for meta in state.workspace_meta.open_questions.iter_mut() {
            let validation = self.validate_evidence_ids(&meta.evidence_event_ids, &meta.belief_ids, false).await;
            if !validation.invalid_evidence_ids.is_empty() || !validation.invalid_belief_ids.is_empty() {
                meta.evidence_event_ids = validation.valid_evidence_ids;
                meta.belief_ids = validation.valid_belief_ids;
                meta.speculative = true;
                demoted_fields.push(format!("open_question:{}", meta.text));
                changed = true;
            }
        }

        for meta in state.workspace_meta.working_set_topics.iter_mut() {
            let validation = self.validate_evidence_ids(&meta.evidence_event_ids, &meta.belief_ids, false).await;
            if !validation.invalid_evidence_ids.is_empty() || !validation.invalid_belief_ids.is_empty() {
                meta.evidence_event_ids = validation.valid_evidence_ids;
                meta.belief_ids = validation.valid_belief_ids;
                meta.speculative = true;
                demoted_fields.push(format!("working_set:{}", meta.text));
                changed = true;
            }
        }

        for hypothesis in state.workspace_active_hypotheses.iter_mut() {
            let validation = self.validate_evidence_ids(&hypothesis.evidence_event_ids, &hypothesis.belief_ids, false).await;
            if !validation.invalid_evidence_ids.is_empty() || !validation.invalid_belief_ids.is_empty() {
                hypothesis.evidence_event_ids = validation.valid_evidence_ids;
                hypothesis.belief_ids = validation.valid_belief_ids;
                hypothesis.speculative = true;
                demoted_fields.push(format!("hypothesis:{}", hypothesis.text));
                changed = true;
            }
        }
        state.workspace_meta.active_hypotheses = state.workspace_active_hypotheses.clone();

        if changed {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "workspace_meta_backfill_demoted",
                    "fields": demoted_fields,
                }),
            )
            .await;
        }

        changed
    }

    pub(super) async fn demote_workspace_meta_for_stale_evidence(&self, state: &mut KernelState) -> bool {
        let now = Utc::now();
        let focus_grace_elapsed = state
            .last_focus_change_at
            .as_deref()
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
            .map(|ts| now.signed_duration_since(ts.with_timezone(&Utc)).num_hours() >= FOCUS_DEMOTION_GRACE_HOURS)
            .unwrap_or(true);

        let mut flags = WorkspaceEvidenceFlags::default();
        if let Some(meta) = state.workspace_meta.goal_thread.as_ref() {
            if !meta.speculative {
                let validation = self.validate_evidence_ids(&meta.evidence_event_ids, &meta.belief_ids, false).await;
                flags.goal_thread_ok = Some(validation.evidence_ok());
            }
        }
        if let Some(meta) = state.workspace_meta.current_focus.as_ref() {
            if !meta.speculative {
                let validation = self.validate_evidence_ids(&meta.evidence_event_ids, &meta.belief_ids, false).await;
                flags.current_focus_ok = Some(validation.evidence_ok());
            }
        }
        if let Some(meta) = state.workspace_meta.focus_rationale.as_ref() {
            if !meta.speculative {
                let validation = self.validate_evidence_ids(&meta.evidence_event_ids, &meta.belief_ids, false).await;
                flags.focus_rationale_ok = Some(validation.evidence_ok());
            }
        }

        for meta in state.workspace_meta.open_questions.iter() {
            if meta.speculative {
                flags.open_questions_ok.push(true);
                continue;
            }
            let validation = self.validate_evidence_ids(&meta.evidence_event_ids, &meta.belief_ids, false).await;
            flags.open_questions_ok.push(validation.evidence_ok());
        }

        for meta in state.workspace_meta.working_set_topics.iter() {
            if meta.speculative {
                flags.working_set_topics_ok.push(true);
                continue;
            }
            let validation = self.validate_evidence_ids(&meta.evidence_event_ids, &meta.belief_ids, false).await;
            flags.working_set_topics_ok.push(validation.evidence_ok());
        }

        for hypothesis in state.workspace_active_hypotheses.iter() {
            if hypothesis.speculative {
                flags.hypotheses_ok.push(true);
                continue;
            }
            let validation = self
                .validate_evidence_ids(&hypothesis.evidence_event_ids, &hypothesis.belief_ids, false)
                .await;
            flags.hypotheses_ok.push(validation.evidence_ok());
        }

        let demoted_fields = apply_workspace_demotions_from_flags(state, focus_grace_elapsed, &flags);

        if !demoted_fields.is_empty() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "workspace_demoted_stale",
                    "fields": demoted_fields,
                }),
            )
            .await;
        }
        !demoted_fields.is_empty()
    }

    pub(super) async fn log_pending_prompt_surface_attempted(
        &self,
        prompt_id: &str,
        source: &str,
        auto_surface: bool,
        outcome: &str,
        reason: &str,
        overlap_workspace: usize,
        overlap_user: usize,
        age_seconds: Option<i64>,
        skip_count: i64,
        anchor_age_seconds: Option<i64>,
    ) {
        if !auto_surface {
            return;
        }
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "chat",
            None,
            None,
            json!({
                "event": "pending_prompt_surface_attempted",
                "prompt_id": prompt_id,
                "source": source,
                "auto_surface": auto_surface,
                "outcome": outcome,
                "reason": reason,
                "overlap_workspace": overlap_workspace,
                "overlap_user": overlap_user,
                "prompt_age_seconds": age_seconds,
                "pending_prompt_starvation_count": skip_count,
                "pending_prompt_anchor_age_seconds": anchor_age_seconds,
            }),
        )
        .await;
    }

    pub(super) async fn select_pending_prompt_for_proactive(
        &self,
        conversation_id: &str,
        state: &KernelState,
        settings: &crate::models::Settings,
        auto_surface_only: bool,
        source_filter: Option<&str>,
    ) -> Option<PendingPromptSelection> {
        let alignment_enabled = settings.pending_prompt_alignment_enabled.unwrap_or(true);
        let recency_secs = settings
            .pending_prompt_recency_secs
            .unwrap_or(PENDING_PROMPT_RECENCY_SECS_DEFAULT);
        let workspace_tokens = workspace_alignment_tokens(state);
        let last_user_input = state.last_user_input.as_deref().unwrap_or("");
        let last_user_input_at = state.last_user_input_at.as_deref();
        let last_user_message_id = state.last_user_message_id.as_deref();
        let user_tokens = token_set(last_user_input);
        let user_hash = if last_user_input.trim().is_empty() {
            None
        } else {
            Some(crate::core::kernel::utils::text::hash_payload(
                &summarize_snippet(last_user_input, 160),
            ))
        };
        let last_user_age_secs = last_user_input_at.and_then(prompt_age_seconds);
        let user_recent = last_user_age_secs
            .map(|age| age >= 0 && age <= recency_secs)
            .unwrap_or(false);
        let open_questions = workspace_verified_open_questions(state);
        let pending = self
            .db
            .list_pending_prompts(conversation_id, 8)
            .await
            .unwrap_or_default();

        for (prompt_id, prompt, source, created_at, skip_count, auto_surface, intent_kind, bridge_id, attempt_count, last_asked_at, expires_at, anchor_message_id, anchor_hash, anchor_created_at, anchor_role) in pending {
            if auto_surface_only && !auto_surface {
                continue;
            }
            if let Some(filter) = source_filter {
                if source != filter {
                    continue;
                }
            }
            let trimmed = prompt.trim();
            let age_seconds = prompt_age_seconds(&created_at);
            let force_reason = pending_prompt_force_reason(skip_count, auto_surface, age_seconds);
            let anchor_age_seconds = anchor_created_at
                .as_deref()
                .and_then(prompt_age_seconds);
            let is_monologue_prompt = source == "monologue";
            if is_monologue_prompt {
                let anchor_role_ok = anchor_role
                    .as_deref()
                    .map(|role| role.eq_ignore_ascii_case("user"))
                    .unwrap_or(false);
                if !anchor_role_ok {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "chat",
                        None,
                        None,
                        json!( {
                            "event": "pending_prompt_anchor_role_invalid",
                            "prompt_id": prompt_id,
                            "source": source,
                            "anchor_role": anchor_role,
                            "anchor_message_id": anchor_message_id,
                            "anchor_age_seconds": anchor_age_seconds,
                        }),
                    )
                    .await;
                    continue;
                }
                let anchor_match = match (anchor_message_id.as_deref(), last_user_message_id) {
                    (Some(anchor_id), Some(last_id)) => anchor_id == last_id,
                    _ => false,
                };
                if !anchor_match {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "chat",
                        None,
                        None,
                        json!( {
                            "event": "pending_prompt_anchor_mismatch",
                            "prompt_id": prompt_id,
                            "source": source,
                            "anchor_message_id": anchor_message_id,
                            "last_user_message_id": last_user_message_id,
                            "anchor_age_seconds": anchor_age_seconds,
                        }),
                    )
                    .await;
                    continue;
                }
                if let (Some(expected), Some(actual)) = (user_hash.as_deref(), anchor_hash.as_deref()) {
                    if expected != actual {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "chat",
                            None,
                            None,
                            json!( {
                                "event": "pending_prompt_anchor_mismatch",
                                "prompt_id": prompt_id,
                                "source": source,
                                "anchor_message_id": anchor_message_id,
                                "anchor_hash": anchor_hash,
                                "anchor_age_seconds": anchor_age_seconds,
                            }),
                        )
                        .await;
                        continue;
                    }
                }
            }
            if auto_surface {
                if is_monologue_prompt {
                    let user_after_enqueue = match (
                        last_user_input_at.and_then(crate::core::kernel::utils::time::timestamp_from_str),
                        crate::core::kernel::utils::time::timestamp_from_str(&created_at),
                    ) {
                        (Some(user_ts), Some(prompt_ts)) => user_ts > prompt_ts,
                        _ => false,
                    };
                    if !user_after_enqueue {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "chat",
                            None,
                            None,
                            json!( {
                                "event": "pending_prompt_auto_surface_blocked",
                                "reason": "no_user_turn_after_enqueue",
                                "prompt_id": prompt_id,
                                "source": source,
                                "anchor_age_seconds": anchor_age_seconds,
                            }),
                        )
                        .await;
                        continue;
                    }
                }
                if !user_recent {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "chat",
                        None,
                        None,
                        json!( {
                            "event": "pending_prompt_auto_surface_blocked",
                            "reason": "stale_user_input",
                            "prompt_id": prompt_id,
                            "source": source,
                            "last_user_age_seconds": last_user_age_secs,
                            "recency_secs": recency_secs,
                            "anchor_age_seconds": anchor_age_seconds,
                        }),
                    )
                    .await;
                    continue;
                }
            }
            if trimmed.is_empty() {
                match self.db.delete_pending_prompt(&prompt_id).await {
                    Ok(affected) if affected > 0 => {}
                    Ok(_) => {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "chat",
                            None,
                            None,
                            json!({
                                "event": "pending_prompt_delete_failed",
                                "reason": "not_found",
                                "prompt_id": prompt_id,
                            }),
                        )
                        .await;
                    }
                    Err(err) => {
                        let _ = self.db.set_summary_pending(conversation_id, true).await;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "chat",
                            None,
                            None,
                            json!({
                                "event": "pending_prompt_delete_failed",
                                "reason": "db_error",
                                "prompt_id": prompt_id,
                                "error": err.to_string(),
                            }),
                        )
                        .await;
                    }
                }
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "chat",
                    None,
                    None,
                    json!({
                        "event": "pending_prompt_discarded",
                        "reason": "empty_prompt",
                    }),
                )
                .await;
                self.log_pending_prompt_surface_attempted(
                    &prompt_id,
                    &source,
                    auto_surface,
                    "discarded",
                    "empty_prompt",
                    0,
                    0,
                    age_seconds,
                    skip_count,
                    anchor_age_seconds,
                )
                .await;
                continue;
            }
            let exact_open_question = open_questions
                .iter()
                .any(|q| q.trim().eq_ignore_ascii_case(trimmed));
            let prompt_tokens = token_set(trimmed);
            let overlap_workspace = prompt_tokens
                .intersection(&workspace_tokens)
                .count();
            let overlap_user = prompt_tokens
                .intersection(&user_tokens)
                .count();
            if is_monologue_prompt
                && !exact_open_question
                && overlap_user < PROACTIVE_OVERLAP_THRESHOLD
            {
                let _ = self.db.increment_pending_prompt_skip_count(&prompt_id).await;
                let next_skip_count = skip_count + 1;
                if let Some(force_reason) =
                    pending_prompt_force_reason(next_skip_count, auto_surface, age_seconds)
                {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "chat",
                        None,
                        None,
                        json!({
                            "event": "pending_prompt_starvation",
                            "reason": force_reason,
                            "source": source,
                            "prompt_id": prompt_id,
                            "skip_count": next_skip_count,
                        }),
                    )
                    .await;
                }
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "chat",
                    None,
                    None,
                    json!({
                        "event": "pending_prompt_held",
                        "source": source,
                        "reason": "anchor_overlap_failed",
                        "overlap_workspace": overlap_workspace,
                        "overlap_user": overlap_user,
                        "prompt_age_seconds": age_seconds,
                        "pending_prompt_starvation_count": next_skip_count,
                    }),
                )
                .await;
                self.log_pending_prompt_surface_attempted(
                    &prompt_id,
                    &source,
                    auto_surface,
                    "held",
                    "anchor_overlap_failed",
                    overlap_workspace,
                    overlap_user,
                    age_seconds,
                    next_skip_count,
                    anchor_age_seconds,
                )
                .await;
                continue;
            }
            if !alignment_enabled {
                return Some(PendingPromptSelection {
                    prompt_id,
                    prompt: trimmed.to_string(),
                    source,
                    overlap_workspace,
                    overlap_user,
                    skip_count,
                    age_seconds,
                    exact_open_question,
                    auto_surface,
                    force_reason: force_reason.map(|r| r.to_string()),
                    intent_kind,
                    bridge_id,
                    attempt_count,
                    last_asked_at,
                    expires_at,
                    anchor_message_id,
                    anchor_hash,
                    anchor_created_at,
                    anchor_role,
                    anchor_age_seconds,
                });
            }
            if let Some(force_reason) = force_reason {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "chat",
                    None,
                    None,
                    json!({
                        "event": "pending_prompt_starvation_forced",
                        "reason": force_reason,
                        "source": source,
                        "auto_surface": auto_surface,
                        "skip_count": skip_count,
                        "overlap_workspace": overlap_workspace,
                        "overlap_user": overlap_user,
                        "prompt_age_seconds": age_seconds,
                    }),
                )
                .await;
                return Some(PendingPromptSelection {
                    prompt_id,
                    prompt: trimmed.to_string(),
                    source,
                    overlap_workspace,
                    overlap_user,
                    skip_count,
                    age_seconds,
                    exact_open_question,
                    auto_surface,
                    force_reason: Some(force_reason.to_string()),
                    intent_kind,
                    bridge_id,
                    attempt_count,
                    last_asked_at,
                    expires_at,
                    anchor_message_id,
                    anchor_hash,
                    anchor_created_at,
                    anchor_role,
                    anchor_age_seconds,
                });
            }
            if exact_open_question
                || overlap_workspace >= PROACTIVE_OVERLAP_THRESHOLD
                || overlap_user >= PROACTIVE_OVERLAP_THRESHOLD
            {
                return Some(PendingPromptSelection {
                    prompt_id,
                    prompt: trimmed.to_string(),
                    source,
                    overlap_workspace,
                    overlap_user,
                    skip_count,
                    age_seconds,
                    exact_open_question,
                    auto_surface,
                    force_reason: None,
                    intent_kind,
                    bridge_id,
                    attempt_count,
                    last_asked_at,
                    expires_at,
                    anchor_message_id,
                    anchor_hash,
                    anchor_created_at,
                    anchor_role,
                    anchor_age_seconds,
                });
            }

            let _ = self.db.increment_pending_prompt_skip_count(&prompt_id).await;
            let next_skip_count = skip_count + 1;
            if let Some(force_reason) =
                pending_prompt_force_reason(next_skip_count, auto_surface, age_seconds)
            {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "chat",
                    None,
                    None,
                    json!({
                        "event": "pending_prompt_starvation",
                        "reason": force_reason,
                        "source": source,
                        "prompt_id": prompt_id,
                        "skip_count": next_skip_count,
                    }),
                )
                .await;
            }
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "chat",
                None,
                None,
                json!({
                    "event": "pending_prompt_held",
                    "source": source,
                    "reason": "alignment_failed",
                    "overlap_workspace": overlap_workspace,
                    "overlap_user": overlap_user,
                    "prompt_age_seconds": age_seconds,
                    "pending_prompt_starvation_count": next_skip_count,
                }),
            )
            .await;
            self.log_pending_prompt_surface_attempted(
                &prompt_id,
                &source,
                auto_surface,
                "held",
                "alignment_failed",
                overlap_workspace,
                overlap_user,
                age_seconds,
                next_skip_count,
                anchor_age_seconds,
            )
            .await;
        }
        None
    }

    pub(crate) async fn auto_surface_pending_prompts(
        &self,
        conversation_id: &str,
        run_id: Option<&str>,
        trace_id: Option<&str>,
        trigger: &str,
    ) -> Result<(), String> {
        let settings = self.db.get_settings().await.map_err(|e| e.to_string())?;
        let mut state = self.load_state(conversation_id).await;
        let Some(selection) = self
            .select_pending_prompt_for_proactive(conversation_id, &state, &settings, true, None)
            .await
        else {
            return Ok(());
        };
        if trigger == "after_response" && selection.source == "monologue" {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "chat",
                run_id,
                trace_id,
                json!( {
                    "event": "pending_prompt_auto_surface_blocked",
                    "reason": "inline_only",
                    "prompt_id": selection.prompt_id,
                    "source": selection.source,
                }),
            )
            .await;
            self.log_pending_prompt_surface_attempted(
                &selection.prompt_id,
                &selection.source,
                selection.auto_surface,
                "held",
                "inline_only",
                selection.overlap_workspace,
                selection.overlap_user,
                selection.age_seconds,
                selection.skip_count,
                selection.anchor_age_seconds,
            )
            .await;
            return Ok(());
        }
        let jsonish = is_jsonish_text(&selection.prompt);
        let question_like = looks_like_question(&selection.prompt);
        if jsonish || !question_like {
            let reason = if jsonish { "json_like" } else { "non_question" };
            let _ = self.db.delete_pending_prompt(&selection.prompt_id).await;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "kernel",
                run_id,
                trace_id,
                json!( {
                    "event": "pending_prompt_sanitized",
                    "reason": reason,
                    "candidate_id": selection.prompt_id,
                    "source": selection.source,
                }),
            )
            .await;
            self.log_pending_prompt_surface_attempted(
                &selection.prompt_id,
                &selection.source,
                selection.auto_surface,
                "blocked",
                "sanitized",
                selection.overlap_workspace,
                selection.overlap_user,
                selection.age_seconds,
                selection.skip_count,
                selection.anchor_age_seconds,
            )
            .await;
            if let Ok(count) = self.db.count_pending_prompts(conversation_id).await {
                let _ = self.app_handle.emit("pending_prompt_count", count);
            }
            self.persist_state(&mut state).await;
            return Ok(());
        }

        let now = Utc::now();
        let force_delivery = selection.auto_surface
            && matches!(
                selection.force_reason.as_deref(),
                Some("auto_surface_sla" | "auto_surface_max_age")
            );
        let evidence_ok = selection.exact_open_question || controller_evidence_ok(&state) || force_delivery;
        if force_delivery && !selection.exact_open_question && !controller_evidence_ok(&state) {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                run_id,
                trace_id,
                json!({
                    "event": "pending_prompt_evidence_bypassed",
                    "candidate_id": selection.prompt_id,
                    "reason": selection.force_reason,
                    "prompt_age_seconds": selection.age_seconds,
                    "trigger": trigger,
                }),
            )
            .await;
        }
        if !evidence_ok {
            let attempt_at = now.to_rfc3339();
            let _ = self
                .db
                .mark_pending_prompt_attempt(&selection.prompt_id, &attempt_at)
                .await;
            self.note_open_question_attempt(
                &mut state,
                conversation_id,
                &selection.prompt,
                now,
                true,
                run_id,
                trace_id,
            )
            .await;

            let new_attempt = selection.attempt_count.saturating_add(1);
            let expired = timestamp_expired(selection.expires_at.as_deref(), now);
            if new_attempt >= PENDING_PROMPT_ATTEMPT_LIMIT || expired {
                let context_hash = context_hash_for_drop(&state, &selection.prompt);
                let _ = self
                    .db
                    .enqueue_deferred_item(
                        conversation_id,
                        "pending_prompt",
                        &selection.prompt,
                        Some(&selection.source),
                        "insufficient_evidence",
                        Some(&context_hash),
                        Some("new_evidence_or_user_request"),
                        new_attempt,
                        Some(&attempt_at),
                        selection.expires_at.as_deref(),
                    )
                    .await;
                let _ = self.db.delete_pending_prompt(&selection.prompt_id).await;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    run_id,
                    trace_id,
                    json!({
                        "event": "pending_prompt_dropped",
                        "reason": "insufficient_evidence",
                        "candidate_id": selection.prompt_id,
                        "attempt_count": new_attempt,
                        "expires_at": selection.expires_at,
                    }),
                )
                .await;
                if let Ok(count) = self.db.count_pending_prompts(conversation_id).await {
                    let _ = self.app_handle.emit("pending_prompt_count", count as usize);
                }
                self.persist_state(&mut state).await;
                return Ok(());
            }

            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                run_id,
                trace_id,
                json!({
                    "event": "proactive_emit_blocked_no_evidence",
                    "candidate_id": selection.prompt_id,
                    "reason": "pending_prompt_evidence_gate",
                    "trigger": trigger,
                }),
            )
            .await;
            self.log_pending_prompt_surface_attempted(
                &selection.prompt_id,
                &selection.source,
                selection.auto_surface,
                "blocked",
                "pending_prompt_evidence_gate",
                selection.overlap_workspace,
                selection.overlap_user,
                selection.age_seconds,
                selection.skip_count,
                selection.anchor_age_seconds,
            )
            .await;
            self.persist_state(&mut state).await;
            return Ok(());
        }

        let allow_speculative_markers = allow_speculative_markers_for_prompt(
            state.last_user_input.as_deref().unwrap_or(""),
            false,
        );
        let mut question =
            strip_working_hypothesis_prefix(&selection.prompt, !allow_speculative_markers);
        if question.trim().is_empty() {
            question = selection.prompt.clone();
        }
        let user_name = settings.user_display_name.as_deref().unwrap_or("User");
        let last_user_input = state.last_user_input.as_deref().unwrap_or("");
        if response_has_user_attribution(&question, user_name)
            && !user_attribution_grounded_in_last_input(&question, last_user_input)
        {
            question = rewrite_user_attribution_text(&question, user_name);
        }
        let payload = json!({
            "question": question,
            "content": question,
            "pending_prompt_id": selection.prompt_id,
            "speculative": !selection.exact_open_question,
            "bridge_id": selection.bridge_id,
            "intent_kind": selection.intent_kind,
        });
        let mut candidate = Candidate {
            id: selection.prompt_id.clone(),
            kind: CandidateKind::AskUserQuestion,
            payload,
            evidence_event_ids: Vec::new(),
            belief_ids: Vec::new(),
            target_scope: None,
            rationale: None,
            expected_outcome: None,
            cost: Some(0),
            urgency: Some(0),
            source: "pending_prompt".to_string(),
            priority_class: priority_class_for(&CandidateKind::AskUserQuestion),
            priority_rank: 0,
            created_at: state.monologue_count,
        };
        candidate.refresh_meta();
        let decision = KernelDecision {
            accepted: vec![candidate.clone()],
            rejected: Vec::new(),
            caps_applied: Vec::new(),
            report: DecisionReport::default(),
        };
        let (overlap, exact_open_question) = candidate_alignment_metrics(&candidate, &state);
        let attempt_at = now.to_rfc3339();
        let _ = self
            .db
            .mark_pending_prompt_attempt(&selection.prompt_id, &attempt_at)
            .await;
        self.note_open_question_attempt(
            &mut state,
            conversation_id,
            &selection.prompt,
            now,
            false,
            run_id,
            trace_id,
        )
        .await;
        let claimed = match self.db.delete_pending_prompt(&selection.prompt_id).await {
            Ok(affected) if affected > 0 => true,
            Ok(_) => {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "chat",
                    run_id,
                    trace_id,
                    json!({
                        "event": "pending_prompt_delete_failed",
                        "reason": "not_found",
                        "prompt_id": selection.prompt_id,
                    }),
                )
                .await;
                false
            }
            Err(err) => {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "chat",
                    run_id,
                    trace_id,
                    json!({
                        "event": "pending_prompt_delete_failed",
                        "reason": "db_error",
                        "prompt_id": selection.prompt_id,
                        "error": err.to_string(),
                    }),
                )
                .await;
                false
            }
        };
        if !claimed {
            return Ok(());
        }

        let run_meta = self
            .run_proactive_emit(
                &mut state,
                decision,
                &candidate,
                conversation_id,
                &settings,
                overlap,
                exact_open_question,
            )
            .await?;
        if run_meta.deferred {
            self.log_pending_prompt_surface_attempted(
                &selection.prompt_id,
                &selection.source,
                selection.auto_surface,
                "deferred",
                "foreground_active",
                selection.overlap_workspace,
                selection.overlap_user,
                selection.age_seconds,
                selection.skip_count,
                selection.anchor_age_seconds,
            )
            .await;
            self.persist_state(&mut state).await;
            return Ok(());
        }
        let log_run_id = run_id.or(Some(run_meta.run_id.as_str()));
        let log_trace_id = trace_id.or(Some(run_meta.trace_id.as_str()));
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "chat",
            log_run_id,
            log_trace_id,
            json!({
                "event": "pending_prompt_surfaced",
                "source": selection.source,
                "alignment_enabled": settings.pending_prompt_alignment_enabled.unwrap_or(true),
                "overlap_workspace": selection.overlap_workspace,
                "overlap_user": selection.overlap_user,
                "prompt_age_seconds": selection.age_seconds,
                "pending_prompt_starvation_count": selection.skip_count,
                "pending_prompt_anchor_age_seconds": selection.anchor_age_seconds,
                "candidate_id": candidate.id,
                "trigger": trigger,
            }),
        )
        .await;
        self.log_pending_prompt_surface_attempted(
            &selection.prompt_id,
            &selection.source,
            selection.auto_surface,
            "surfaced",
            "delivered",
            selection.overlap_workspace,
            selection.overlap_user,
            selection.age_seconds,
            selection.skip_count,
            selection.anchor_age_seconds,
        )
        .await;
        if let Ok(count) = self.db.count_pending_prompts(conversation_id).await {
            let _ = self.app_handle.emit("pending_prompt_count", count);
        }
        Ok(())
    }

    pub(super) async fn auto_surface_pending_prompts_after_response(
        &self,
        conversation_id: &str,
        run_id: &str,
        trace_id: &str,
    ) -> Result<(), String> {
        self.auto_surface_pending_prompts(
            conversation_id,
            Some(run_id),
            Some(trace_id),
            "after_response",
        )
        .await
    }

    pub(crate) async fn run_deferred_candidate_emit(
        &self,
        conversation_id: &str,
        emit_id: &str,
        payload_json: &str,
    ) -> Result<bool, String> {
        let mut candidate: Candidate =
            serde_json::from_str(payload_json).map_err(|e| e.to_string())?;
        candidate.refresh_meta();
        let settings = self.db.get_settings().await.map_err(|e| e.to_string())?;
        let mut state = self.load_state(conversation_id).await;
        let decision = KernelDecision {
            accepted: vec![candidate.clone()],
            rejected: Vec::new(),
            caps_applied: Vec::new(),
            report: DecisionReport::default(),
        };
        let (overlap, exact_open_question) = candidate_alignment_metrics(&candidate, &state);
        let run_meta = self
            .run_proactive_emit(
                &mut state,
                decision,
                &candidate,
                conversation_id,
                &settings,
                overlap,
                exact_open_question,
            )
            .await?;
        if run_meta.deferred {
            return Ok(false);
        }
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(&run_meta.run_id),
            Some(&run_meta.trace_id),
            json!( {
                "event": "proactive_released",
                "emit_id": emit_id,
                "emit_kind": "candidate_emit",
                "candidate_id": candidate.id,
            }),
        )
        .await;
        self.persist_state(&mut state).await;
        Ok(true)
    }

    pub(super) async fn proactive_emit_allowed(
        &self,
        state: &KernelState,
        candidate: &Candidate,
        overlap: usize,
        exact_open_question: bool,
    ) -> bool {
        let is_monologue_intent = is_monologue_source(&candidate.source);
        if matches!(candidate.kind, CandidateKind::EmitMessage) {
            let evidence_event_ids = extract_id_list(&candidate.payload, "evidence_event_ids");
            let belief_ids = extract_id_list(&candidate.payload, "belief_ids");
            let payload_has_ids = !evidence_event_ids.is_empty() || !belief_ids.is_empty();
            let validation = if payload_has_ids {
                self.validate_evidence_ids(&evidence_event_ids, &belief_ids, false).await
            } else {
                ValidationResult::default()
            };
            if !validation.invalid_evidence_ids.is_empty() || !validation.invalid_belief_ids.is_empty() {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "evidence_validation_failed",
                        "candidate_id": candidate.id,
                        "invalid_evidence_ids": validation.invalid_evidence_ids,
                        "invalid_belief_ids": validation.invalid_belief_ids,
                    }),
                )
                .await;
            }
            let payload_evidence_ok = validation.evidence_ok();
            let evidence_ok = payload_evidence_ok || controller_evidence_ok(state);
            if !evidence_ok {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "proactive_emit_blocked_no_evidence",
                        "candidate_id": candidate.id,
                        "reason": "evidence_gate",
                    }),
                )
                .await;
                if !is_monologue_intent {
                    return false;
                }
            }
            if !payload_evidence_ok {
                if let Some(text) = candidate_alignment_text(candidate) {
                    if candidate_introduces_new_terms(&text, state) {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "proactive_emit_blocked_no_evidence",
                                "candidate_id": candidate.id,
                                "reason": "new_terms",
                            }),
                        )
                        .await;
                        if !is_monologue_intent {
                            return false;
                        }
                    }
                }
            }
        }
        if !is_monologue_intent {
            let required_overlap = if matches!(candidate.kind, CandidateKind::AskUserQuestion) {
                CLARIFIER_OVERLAP_THRESHOLD
            } else {
                PROACTIVE_OVERLAP_THRESHOLD
            };
            let aligned = exact_open_question || overlap >= required_overlap;
            if !aligned {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "proactive_alignment_fail",
                        "overlap": overlap,
                        "exact_open_question": exact_open_question,
                        "workspace_focus": state.workspace_current_focus,
                    }),
                )
                .await;
                return false;
            }
        }

        if !is_monologue_intent {
            if matches!(state.task_phase, TaskPhase::AwaitingUser)
                && !proactive_followup_allowed(state, candidate)
            {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "proactive_throttle",
                        "reason": "awaiting_user",
                    }),
                )
                .await;
                return false;
            }

            if state.ask_loop_breaker_triggered {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "proactive_throttle",
                        "reason": "ask_loop_breaker",
                    }),
                )
                .await;
                return false;
            }

            if let Some(ts) = state.last_user_input_at.as_deref() {
                if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(ts) {
                    let now = Utc::now();
                    if now.signed_duration_since(dt.with_timezone(&Utc)).num_seconds()
                        < PROACTIVE_QUIESCENCE_SECS
                    {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "proactive_throttle",
                                "reason": "quiescence_window",
                            }),
                        )
                        .await;
                        return false;
                    }
                }
            }

            if let Some(until) = state.proactive_cooldown_until.as_deref() {
                if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(until) {
                    if Utc::now() < dt.with_timezone(&Utc) {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "proactive_throttle",
                                "reason": "cooldown",
                                "cooldown_until": until,
                            }),
                        )
                        .await;
                        return false;
                    }
                }
            }

            if let Some(last) = state.last_proactive_emit_at.as_deref() {
                if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(last) {
                    let elapsed = Utc::now().signed_duration_since(dt.with_timezone(&Utc)).num_seconds();
                    if elapsed < PROACTIVE_THROTTLE_SECS {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "proactive_throttle",
                                "reason": "rate_limit",
                                "elapsed_seconds": elapsed,
                            }),
                        )
                        .await;
                        return false;
                    }
                }
            }

            if matches!(candidate.kind, CandidateKind::AskUserQuestion)
                && state.ask_budget_remaining <= 0
            {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "proactive_ask_budget_exhausted",
                        "ask_budget_remaining": state.ask_budget_remaining,
                    }),
                )
                .await;
                return false;
            }
        }

        true
    }

    pub(super) async fn prepare_proactive_candidate(
        &self,
        state: &KernelState,
        candidate: &Candidate,
        disable_working_hypothesis: bool,
        allow_speculative: bool,
    ) -> Option<Candidate> {
        let evidence_event_ids = extract_id_list(&candidate.payload, "evidence_event_ids");
        let belief_ids = extract_id_list(&candidate.payload, "belief_ids");
        let payload_has_ids = !evidence_event_ids.is_empty() || !belief_ids.is_empty();
        let validation = if payload_has_ids {
            self.validate_evidence_ids(&evidence_event_ids, &belief_ids, false).await
        } else {
            ValidationResult::default()
        };
        if !validation.invalid_evidence_ids.is_empty() || !validation.invalid_belief_ids.is_empty() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "evidence_validation_failed",
                    "candidate_id": candidate.id,
                    "invalid_evidence_ids": validation.invalid_evidence_ids,
                    "invalid_belief_ids": validation.invalid_belief_ids,
                }),
            )
            .await;
        }
        let result = coerce_proactive_candidate_for_evidence(
            candidate,
            state,
            validation.evidence_ok(),
            payload_has_ids,
            disable_working_hypothesis,
        );
        if let Some(reason) = result.blocked_reason.as_deref() {
            if allow_speculative {
                let mut fallback = result
                    .candidate
                    .unwrap_or_else(|| candidate.clone());
                let mut payload = fallback.payload.clone();
                if !payload.get("speculative").is_some() {
                    payload["speculative"] = json!(true);
                }
                fallback.payload = payload;
                return Some(fallback);
            }
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "proactive_emit_blocked_no_evidence",
                    "candidate_id": candidate.id,
                    "reason": reason,
                }),
            )
            .await;
            return None;
        }
        if result.speculation_marked {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "speculation_marked",
                    "candidate_id": candidate.id,
                }),
            )
            .await;
        }
        result.candidate
    }

    pub(super) async fn run_proactive_emit(
        &self,
        state: &mut KernelState,
        mut decision: KernelDecision,
        selected: &Candidate,
        conversation_id: &str,
        settings: &crate::models::Settings,
        overlap: usize,
        exact_open_question: bool,
    ) -> Result<ProactiveRunMeta, String> {
        let run_id = Uuid::new_v4().to_string();
        let trace_id = run_id.clone();
        let run_meta = ProactiveRunMeta {
            run_id: run_id.clone(),
            trace_id: trace_id.clone(),
            deferred: false,
        };
        let run_metadata = json!({ "execution_mode": "proactive" });
        if let Ok(Some(active_run_id)) = self.db.get_active_foreground_run(conversation_id).await {
            let payload_json =
                serde_json::to_string(selected).unwrap_or_else(|_| "{}".to_string());
            let emit_id = self
                .db
                .enqueue_deferred_emit(
                    conversation_id,
                    "candidate_emit",
                    &payload_json,
                    Some(selected.source.as_str()),
                )
                .await
                .map_err(|e| e.to_string())?;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!( {
                    "event": "run_supersede_blocked",
                    "conversation_id": conversation_id,
                    "active_run_id": active_run_id,
                    "source": "proactive_emit",
                }),
            )
            .await;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!( {
                    "event": "proactive_deferred",
                    "emit_id": emit_id,
                    "emit_kind": "candidate_emit",
                    "candidate_id": selected.id,
                    "source": selected.source,
                }),
            )
            .await;
            return Ok(ProactiveRunMeta {
                run_id,
                trace_id,
                deferred: true,
            });
        }

        let started_at = Utc::now();
        sqlx::query(
            "INSERT INTO runs (run_id, trace_id, conversation_id, started_at, heartbeat_at, status, metadata)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&run_id)
        .bind(&trace_id)
        .bind(conversation_id)
        .bind(started_at)
        .bind(started_at)
        .bind("active")
        .bind(run_metadata.to_string())
        .execute(&self.db.pool)
        .await
        .map_err(|e| e.to_string())?;

        if let Some((subject_state, snapshot)) =
            self.build_and_persist_subject_snapshot(state, Some(&run_id), Some(&run_id), "proactive_gate")
                .await
        {
            let proposal = subject_controller::build_action_proposal(selected);
            if let Err(err) = subject_controller::persist_action_proposal(&self.db, &snapshot.snapshot_hash, &proposal).await {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "proactive_gate_failed",
                        "reason": "persist_action_proposal",
                        "error": err,
                    }),
                )
                .await;
                let _ = sqlx::query("UPDATE runs SET status = 'error', ended_at = ? WHERE run_id = ?")
                    .bind(Utc::now())
                    .bind(&run_id)
                    .execute(&self.db.pool)
                    .await;
                return Ok(run_meta);
            }
            let tool_names = self.tools.allowed_tool_names(settings);
            let anchor_vocab = build_anchor_vocab(state, &tool_names);
            let anchor_hits = count_anchor_hits(&candidate_relevance_text(selected), &anchor_vocab);
            let signals = self
                .compute_gate_signals(state, &subject_state, Some(&decision), selected, anchor_hits, settings)
                .await;
            let soft_gate = subject_controller::build_gate_decision(&subject_state, selected, &state.stop_state, &signals);
            let legacy_gate =
                subject_controller::build_gate_decision_legacy(&subject_state, selected, &state.stop_state);
            let soft_decision = soft_gate.decision.clone();
            let legacy_decision = legacy_gate.decision.clone();
            let rollout_percent = settings.gate_rollout_percent.unwrap_or(100).clamp(0, 100);
            let shadow_mode = settings.gate_shadow_mode.unwrap_or(false);
            let rollout_bucket = gate_rollout_bucket(conversation_id);
            let use_soft_gate = !shadow_mode && (rollout_percent >= 100 || rollout_bucket < rollout_percent);
            let (gate, _shadow_gate) = if use_soft_gate {
                (soft_gate, legacy_gate)
            } else {
                (legacy_gate, soft_gate)
            };
            let gate_reasons_log = serde_json::from_str::<Value>(&gate.evidence_refs_json)
                .ok()
                .and_then(|value| value.get("reasons").cloned())
                .unwrap_or_else(|| json!([]));
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "gate_decision_inputs",
                    "candidate_id": selected.id,
                    "candidate_kind": format!("{:?}", selected.kind),
                    "anchor_hits": anchor_hits,
                    "signals": signals,
                    "soft_decision": soft_decision,
                    "legacy_decision": legacy_decision,
                    "enforced_decision": gate.decision,
                    "gate_reasons": gate_reasons_log,
                    "shadow_mode": shadow_mode,
                    "rollout_percent": rollout_percent,
                    "rollout_bucket": rollout_bucket,
                    "execution_mode": "proactive",
                    "organism": subject_state.organism,
                }),
            )
            .await;
            if let Err(err) = subject_controller::persist_gate_decision(
                &self.db,
                &snapshot.snapshot_hash,
                &proposal.proposal_id,
                &gate,
            )
            .await
            {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "proactive_gate_failed",
                        "reason": "persist_gate_decision",
                        "error": err,
                    }),
                )
                .await;
                let _ = sqlx::query("UPDATE runs SET status = 'error', ended_at = ? WHERE run_id = ?")
                    .bind(Utc::now())
                    .bind(&run_id)
                    .execute(&self.db.pool)
                    .await;
                return Ok(run_meta);
            }
            decision.report.snapshot_hash = Some(snapshot.snapshot_hash.clone());
            decision.report.gate_decision_id = Some(gate.decision_id.clone());
            decision.report.gate_decision = Some(gate.decision.clone());
            let reasons: Vec<String> = serde_json::from_str::<Value>(&gate.evidence_refs_json)
                .ok()
                .and_then(|v| v.get("reasons").and_then(|r| r.as_array()).cloned())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|item| item.as_str())
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if !reasons.is_empty() {
                decision.report.gate_reasons = Some(reasons.clone());
            }
            decision.report.gate_notice = gate_notice_for(&gate.decision, &reasons);
            if !gate_allows_response(&gate.decision) {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "proactive_gate_blocked",
                        "decision": gate.decision,
                        "snapshot_hash": snapshot.snapshot_hash,
                        "candidate_id": selected.id,
                    }),
                )
                .await;
                let _ = sqlx::query("UPDATE runs SET status = 'complete', ended_at = ? WHERE run_id = ?")
                    .bind(Utc::now())
                    .bind(&run_id)
                    .execute(&self.db.pool)
                    .await;
                return Ok(run_meta);
            }
        }

        let now = Utc::now();
        let memory_pass_enabled = settings.auto_memory_pass_enabled.unwrap_or(true);
        let memory_pass_due = memory_pass_enabled && proactive_memory_pass_due(state, now);
        if memory_pass_due {
            let _ = self
                .db
                .create_memory_pass_token(&run_id, conversation_id, 600)
                .await;
        }

        let allow_workspace_compliance = if is_monologue_source(&selected.source) {
            state
                .last_user_input
                .as_deref()
                .map(user_requested_state)
                .unwrap_or(false)
        } else {
            true
        };
        if allow_workspace_compliance {
            self.apply_proactive_workspace_compliance(&mut decision, state)
                .await;
        }

        self
            .enforce_monologue_question_grounding(&mut decision, settings, Some(&run_id), Some(&trace_id))
            .await;
        self
            .enforce_grounding_on_emits(&mut decision, settings, Some(&run_id), Some(&trace_id))
            .await;

        let commit_started = Instant::now();
        let commit_result = self
            .commit_cycle(
                state,
                &decision,
                conversation_id,
                Some(&run_id),
                Some(&trace_id),
                settings,
                false,
                None,
                false,
            )
            .await?;
        self.mark_candidate_outcomes(&decision, "accepted", "rejected")
            .await;
        let commit_ms = commit_started.elapsed().as_millis() as i64;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(&run_id),
            Some(&trace_id),
            json!({
                "event": "timing_commit_cycle",
                "duration_ms": commit_ms,
                "path": "proactive_emit",
            }),
        )
        .await;

        if !commit_result.tool_dispatches.is_empty() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "proactive_tool_suppressed",
                }),
            )
            .await;
        }
        if commit_result.thread_run.is_some() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "proactive_thread_suppressed",
                }),
            )
            .await;
        }

        if commit_result.research_cost > 0 {
            let remaining = self.research_budget_remaining(state, settings);
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "research_budget_usage",
                    "consumed": commit_result.research_cost,
                    "remaining": remaining,
                    "window_start": state.research_window_start,
                }),
            )
            .await;
        }

        let mut content = commit_result.emit_content.unwrap_or_default();
        if content.trim().is_empty() {
            let _ = sqlx::query("UPDATE runs SET status = 'complete', ended_at = ? WHERE run_id = ?")
                .bind(Utc::now())
                .bind(&run_id)
                .execute(&self.db.pool)
                .await;
            return Ok(run_meta);
        }
        let user_name = settings.user_display_name.as_deref().unwrap_or("User");
        if response_has_user_attribution(&content, user_name) {
            let user_evidence_allowlist = self
                .db
                .get_recent_user_evidence(conversation_id, 8)
                .await;
            let mut evidence_ids = extract_id_list(&selected.payload, "evidence_event_ids");
            if evidence_ids.is_empty() && !user_evidence_allowlist.is_empty() {
                evidence_ids = extract_user_attribution_fallback(&content, &user_evidence_allowlist);
            }
            let allowlist_set: HashSet<i64> = user_evidence_allowlist.iter().map(|(id, _)| *id).collect();
            let mut validation = ValidationResult::default();
            let mut allowlist_ok = false;
            if !evidence_ids.is_empty() {
                validation = self.validate_evidence_ids(&evidence_ids, &[], false).await;
                allowlist_ok = evidence_ids.iter().all(|id| allowlist_set.contains(id));
            }
            if user_attribution_blocked(&evidence_ids, &validation, allowlist_ok) {
                let original = content.clone();
                content = rewrite_user_attribution_text(&content, user_name);
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "proactive_user_attribution_rewritten",
                        "candidate_id": selected.id,
                        "source": selected.source,
                        "snippet": summarize_snippet(&original, 160),
                    }),
                )
                .await;
            }
        }

        let message_source = if selected.source == "pending_prompt" {
            "pending_prompt"
        } else {
            "monologue"
        };
        let metadata = json!({
            "proactive": true,
            "source": message_source,
            "origin": selected.source,
            "surface": true,
            "candidate_kind": format!("{:?}", selected.kind),
            "candidate_id": selected.id,
            "bridge_id": selected.payload.get("bridge_id").and_then(|v| v.as_str()),
        });
        let _ = self
            .emit_proactive_message(conversation_id, &content, &run_id, &trace_id, metadata)
            .await?;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(&run_id),
            Some(&trace_id),
            json!( {
                "event": "intent_delivered",
                "candidate_id": selected.id,
                "candidate_kind": format!("{:?}", selected.kind),
                "source": selected.source,
                "bridge_id": selected.payload.get("bridge_id").and_then(|v| v.as_str()),
            }),
        )
        .await;
        if is_monologue_source(&selected.source) {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!( {
                    "event": "meta_cog_intent_delivered",
                    "candidate_id": selected.id,
                    "candidate_kind": format!("{:?}", selected.kind),
                    "source": selected.source,
                    "bridge_id": selected.payload.get("bridge_id").and_then(|v| v.as_str()),
                }),
            )
            .await;
        }
        if is_monologue_source(&selected.source) {
            if let Some(prompt_id) = selected
                .payload
                .get("pending_prompt_id")
                .and_then(|v| v.as_str())
            {
                match self.db.delete_pending_prompt(prompt_id).await {
                    Ok(affected) if affected > 0 => {}
                    Ok(_) => {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "chat",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "pending_prompt_delete_failed",
                                "reason": "not_found",
                                "prompt_id": prompt_id,
                            }),
                        )
                        .await;
                    }
                    Err(err) => {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "chat",
                            Some(&run_id),
                            Some(&trace_id),
                            json!({
                                "event": "pending_prompt_delete_failed",
                                "reason": "db_error",
                                "prompt_id": prompt_id,
                                "error": err.to_string(),
                            }),
                        )
                        .await;
                    }
                }
            }
        }

        let _ = sqlx::query("UPDATE runs SET status = 'complete', ended_at = ? WHERE run_id = ?")
            .bind(Utc::now())
            .bind(&run_id)
            .execute(&self.db.pool)
            .await;

        if memory_pass_due {
            let memory_input = state
                .last_user_input
                .clone()
                .unwrap_or_else(|| "No user input (proactive).".to_string());
            let _ = self
                .run_memory_pass(settings, &run_id, &memory_input, &content, false)
                .await;
            state.last_proactive_memory_pass_at = Some(now.to_rfc3339());
        }

        state.last_proactive_emit_at = Some(now.to_rfc3339());
        state.proactive_cooldown_until =
            Some((now + chrono::Duration::seconds(PROACTIVE_COOLDOWN_SECS)).to_rfc3339());
        if matches!(selected.kind, CandidateKind::AskUserQuestion) {
            state.last_proactive_question = Some(content.clone());
        }
        if !content.trim().is_empty() {
            let summary = summarize_snippet(&content, 160);
            state.last_response_summary = Some(summary.clone());
            state.last_response_summary_at = Some(now.to_rfc3339());
            if let Some(evidence_id) = self
                .db
                .create_system_evidence_event(
                    conversation_id,
                    "outcome_summary",
                    &summary,
                    Some(&run_id),
                    &summary,
                )
                .await
            {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "memory",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "outcome_summary_evidence",
                        "evidence_id": evidence_id,
                        "conversation_id": conversation_id,
                    }),
                )
                .await;
            }
        }
        self.persist_state(state).await;

        let workspace_state_json = serde_json::to_string(&json!({
            "goal_thread": state.workspace_goal_thread.clone(),
            "active_plan_id": state.workspace_active_plan_id.clone(),
            "goal_stack": state.workspace_goal_stack.clone(),
            "open_questions": state.workspace_open_questions.clone(),
            "active_hypotheses": state.workspace_active_hypotheses.clone(),
            "working_set_topics": state.workspace_working_set_topics.clone(),
            "current_focus": state.workspace_current_focus.clone(),
            "focus_rationale": state.workspace_focus_rationale.clone(),
            "workspace_meta": state.workspace_meta.clone(),
        }))
        .unwrap_or_else(|_| "{}".to_string());
        let workspace_hash = hash_payload(&workspace_state_json);
        let rolling_summary = self
            .db
            .get_effective_rolling_summary(conversation_id)
            .await
            .ok()
            .and_then(|(summary, _)| summary)
            .unwrap_or_default();
        let rolling_summary_hash = hash_payload(&rolling_summary);
        let evidence_event_ids = extract_id_list(&selected.payload, "evidence_event_ids");
        let belief_ids = extract_id_list(&selected.payload, "belief_ids");

        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(&run_id),
            Some(&trace_id),
            json!({
                "event": "proactive_emit",
                "overlap": overlap,
                "exact_open_question": exact_open_question,
                "workspace_focus": state.workspace_current_focus,
                "candidate_id": selected.id,
                "source": selected.source,
                "workspace_hash": workspace_hash,
                "evidence_event_ids": evidence_event_ids,
                "belief_ids": belief_ids,
            }),
        )
        .await;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(&run_id),
            Some(&trace_id),
            json!({
                "event": "proactive_state_sync",
                "workspace_hash": workspace_hash,
                "rolling_summary_hash": rolling_summary_hash,
            }),
        )
        .await;

        Ok(run_meta)
    }

    pub(super) fn refresh_research_budget(&self, state: &mut KernelState, settings: &crate::models::Settings) {
        let budget = settings.research_budget_per_hour.unwrap_or(0);
        if budget <= 0 {
            state.research_used = 0;
            state.research_window_start = None;
            return;
        }
        let window_minutes = settings.research_budget_reset_window.unwrap_or(60).max(1);
        let now = Utc::now();
        let window_start = state
            .research_window_start
            .as_deref()
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
            .map(|ts| ts.with_timezone(&Utc));
        let should_reset = window_start
            .map(|start| now.signed_duration_since(start).num_minutes() >= window_minutes)
            .unwrap_or(true);
        if should_reset {
            state.research_window_start = Some(now.to_rfc3339());
            state.research_used = 0;
        }
    }

    pub(super) fn research_budget_remaining(&self, state: &KernelState, settings: &crate::models::Settings) -> i64 {
        let budget = settings.research_budget_per_hour.unwrap_or(0);
        if budget <= 0 {
            return 0;
        }
        (budget - state.research_used).max(0)
    }

    pub(super) fn allow_internal_emit(&self, state: &KernelState, settings: &crate::models::Settings) -> bool {
        if !settings.monologue_surface_enabled.unwrap_or(false) {
            return false;
        }
        let Some(until) = state.monologue_surface_until.as_deref() else {
            return false;
        };
        if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(until) {
            return chrono::Utc::now() <= ts.with_timezone(&Utc);
        }
        false
    }

    pub(super) async fn load_parameter_registry(&self, settings: &crate::models::Settings) -> (Value, Option<RegistryMeta>) {
        let profile = settings
            .registry_profile_name
            .as_deref()
            .unwrap_or("default");
        if let Ok(Some(registry)) = self.db.get_parameter_registry(profile).await {
            let payload_json = registry.payload_json.trim().to_string();
            let mut parsed = serde_json::from_str::<Value>(&payload_json).unwrap_or_else(|_| json!({}));
            if !parsed.is_object() {
                parsed = json!({});
            }
            let meta_version = parsed
                .get("_meta")
                .and_then(|v| v.get("schema_version"))
                .and_then(|v| v.as_i64());
            let compatibility = match meta_version {
                Some(v) if v == REGISTRY_SCHEMA_VERSION => "ok",
                Some(v) if v < REGISTRY_SCHEMA_VERSION => "upgrade_needed",
                Some(v) if v > REGISTRY_SCHEMA_VERSION => "unsupported",
                None => "missing_meta",
                _ => "invalid_meta",
            }
            .to_string();

            let mut meta = parsed
                .get("_meta")
                .and_then(|v| v.as_object())
                .cloned()
                .unwrap_or_default();
            if meta_version.is_none() {
                meta.insert("schema_version".to_string(), json!(REGISTRY_SCHEMA_VERSION));
            }
            meta.insert("compatibility".to_string(), json!(&compatibility));
            if let Some(obj) = parsed.as_object_mut() {
                obj.insert("_meta".to_string(), Value::Object(meta));
            }

            if compatibility == "unsupported" {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "registry_profile_incompatible",
                        "profile": registry.profile_name,
                        "version": registry.profile_version,
                        "compatibility": compatibility,
                    }),
                )
                .await;
                let fallback = json!({
                    "_meta": {
                        "schema_version": REGISTRY_SCHEMA_VERSION,
                        "compatibility": compatibility,
                    }
                });
                return (
                    fallback,
                    Some(RegistryMeta {
                        name: registry.profile_name,
                        version: registry.profile_version,
                        hash: hash_payload(&payload_json),
                        compatibility,
                    }),
                );
            }

            if compatibility != "ok" {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "registry_profile_incompatible",
                        "profile": registry.profile_name,
                        "version": registry.profile_version,
                        "compatibility": compatibility,
                    }),
                )
                .await;
            }

            let normalized_hash = serde_json::to_string(&parsed).unwrap_or(payload_json.clone());
            let hash = hash_payload(&normalized_hash);
            return (
                parsed,
                Some(RegistryMeta {
                    name: registry.profile_name,
                    version: registry.profile_version,
                    hash,
                    compatibility,
                }),
            );
        }
        (
            json!({
                "_meta": {
                    "schema_version": REGISTRY_SCHEMA_VERSION,
                    "compatibility": "missing_profile",
                }
            }),
            None,
        )
    }

    pub(super) async fn build_resolution_context(
        &self,
        conversation_id: &str,
        settings: &crate::models::Settings,
        current_input: &str,
    ) -> (InputResolutionContext, Option<RegistryMeta>) {
        let mut recent_text = Vec::new();
        if let Ok(history) = self.db.get_history_for_conversation(conversation_id, 12).await {
            for msg in history.iter().rev().take(6) {
                let clean = msg.content.replace('\n', " ");
                if !clean.trim().is_empty() {
                    recent_text.push(clean);
                }
            }
        }
        let mut kv = BTreeMap::new();
        if let Ok(recent) = self.db.get_recent_keys(20).await {
            for (k, v) in recent {
                kv.insert(k, v);
            }
        }
        let (registry, meta) = self.load_parameter_registry(settings).await;
        (
            InputResolutionContext {
                current_input: current_input.to_string(),
                recent_text,
                kv,
                registry,
                defaults: settings.request_defaults.clone(),
            },
            meta,
        )
    }

    pub(super) fn is_known_tool_name(&self, tool_name: &str) -> bool {
        let name = tool_name.trim();
        if name.is_empty() {
            return false;
        }
        self.tools
            .definitions()
            .iter()
            .any(|tool| tool.function.name.eq_ignore_ascii_case(name))
    }

    pub(super) fn is_allowed_tool_name(&self, tool_name: &str, settings: &crate::models::Settings) -> bool {
        self.tools.is_tool_allowed(tool_name, settings)
    }


    pub(super) async fn tool_used_recently(&self, tool_name: &str, window_days: i64) -> bool {
        if tool_name.trim().is_empty() {
            return false;
        }
        let window = format!("-{} days", window_days.max(1));
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM tool_dispatches
             WHERE tool_name = ?
               AND datetime(updated_at) >= datetime('now', ?)",
        )
        .bind(tool_name)
        .bind(window)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0);
        count > 0
    }

    pub(super) async fn compute_gate_signals(
        &self,
        state: &mut KernelState,
        subject_state: &subject_state::SubjectState,
        decision: Option<&KernelDecision>,
        candidate: &Candidate,
        anchor_hits: usize,
        settings: &crate::models::Settings,
    ) -> subject_controller::GateSignals {
        let mut tool_name: Option<String> = None;
        if matches!(candidate.kind, CandidateKind::ToolCall) {
            tool_name = candidate
                .payload
                .get("tool_name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
        }

        let candidate_text = candidate_relevance_text(candidate);
        let topic_shift_detected = if candidate_text.trim().is_empty() {
            false
        } else {
            candidate_introduces_new_terms(&candidate_text, state)
        };

        let new_objective = matches!(
            candidate.kind,
            CandidateKind::UpdateGoalThread | CandidateKind::AnchorShift
        ) || (matches!(candidate.kind, CandidateKind::UpdateWorkspace)
            && (candidate.payload.get("goal_thread").is_some()
                || candidate.payload.get("current_focus").is_some()));

        let toolchain_unseen = if let Some(name) = tool_name.as_deref() {
            !self.tool_used_recently(name, 7).await
        } else {
            false
        };

        let controller = &subject_state.self_model.controller_state;
        let controller_uncertainty_high = controller.uncertainty >= 0.60;
        let low_evidence =
            controller.evidence_coverage < EVIDENCE_MIN || controller.telemetry_coverage < TELEMETRY_MIN
                || !workspace_has_verified_anchor(state);

        let candidate_disagreement = decision
            .map(|d| d.accepted.len().saturating_sub(1))
            .unwrap_or(0);

        let recent_tool_failures = recent_tool_failure_count_for(state, tool_name.as_deref());

        let mut novelty_score = 0.0_f32;
        if anchor_hits == 0 {
            novelty_score += 0.35;
        }
        if topic_shift_detected {
            novelty_score += 0.25;
        }
        if toolchain_unseen {
            novelty_score += 0.20;
        }
        if new_objective {
            novelty_score += 0.20;
        }
        novelty_score = novelty_score.clamp(0.0, 1.0);

        let mut uncertainty_score = 0.0_f32;
        if controller_uncertainty_high {
            uncertainty_score += 0.35;
        }
        if low_evidence {
            uncertainty_score += 0.25;
        }
        if candidate_disagreement > 0 {
            uncertainty_score += 0.20;
        }
        if recent_tool_failures > 0 {
            uncertainty_score += 0.20;
        }
        uncertainty_score = uncertainty_score.clamp(0.0, 1.0);

        let qualia_tag = subject_state.qualia.dominant_tag.clone();
        let qualia_intensity = subject_state.qualia.dominant_intensity as f32;
        let qualia_confidence = subject_state.qualia.prediction_confidence as f32;
        let qualia_strength = (qualia_intensity * qualia_confidence).clamp(0.0, 1.0);
        let mut qualia_novelty_delta = 0.0_f32;
        let mut qualia_uncertainty_delta = 0.0_f32;
        if qualia_strength > 0.0 {
            match qualia_tag.as_deref().unwrap_or("neutral") {
                "skeptical" => {
                    qualia_uncertainty_delta += 0.15 * qualia_strength;
                }
                "curious" => {
                    qualia_novelty_delta += 0.12 * qualia_strength;
                }
                "urgent" => {
                    qualia_uncertainty_delta -= 0.10 * qualia_strength;
                }
                "calm" => {
                    qualia_uncertainty_delta -= 0.05 * qualia_strength;
                }
                "informative" => {
                    qualia_novelty_delta -= 0.05 * qualia_strength;
                }
                _ => {}
            }
        }
        let base_novelty = novelty_score;
        let base_uncertainty = uncertainty_score;
        novelty_score = (novelty_score + qualia_novelty_delta).clamp(0.0, 1.0);
        uncertainty_score = (uncertainty_score + qualia_uncertainty_delta).clamp(0.0, 1.0);
        if qualia_novelty_delta.abs() > 0.0001 || qualia_uncertainty_delta.abs() > 0.0001 {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "qualia_modulation_applied",
                    "path": "gate_signals",
                    "qualia_tag": qualia_tag.clone(),
                    "qualia_intensity": qualia_intensity,
                    "qualia_confidence": qualia_confidence,
                    "novelty_base": base_novelty,
                    "uncertainty_base": base_uncertainty,
                    "novelty_delta": qualia_novelty_delta,
                    "uncertainty_delta": qualia_uncertainty_delta,
                }),
            )
            .await;
        }

        let mut tool_misuse_risk = 0.0_f32;
        if let Some(name) = tool_name.as_deref() {
            if name.eq_ignore_ascii_case("run_shell") {
                tool_misuse_risk = 0.9;
            } else if recent_tool_failures > 0 {
                tool_misuse_risk = 0.7;
            } else if toolchain_unseen {
                tool_misuse_risk = 0.5;
            } else {
                tool_misuse_risk = 0.2;
            }
        }

        let risk_score = ((subject_state.organism.integrity_risk * 0.6)
            + (subject_state.organism.uncertainty_pressure * 0.4))
            .clamp(0.0, 1.0) as f32;

        let requires_audit = matches!(candidate.kind, CandidateKind::ToolCall)
            && (tool_misuse_risk >= 0.7
                || toolchain_unseen
                || tool_name
                    .as_deref()
                    .map(|name| name.eq_ignore_ascii_case("run_shell"))
                    .unwrap_or(false));

        let high_risk_now = risk_score >= 0.80
            && (subject_state.organism.stress >= 0.70 || subject_state.organism.fatigue >= 0.80)
            && tool_misuse_risk >= 0.70;
        let high_risk_streak = if high_risk_now {
            state.gate_high_risk_streak.saturating_add(1)
        } else {
            0
        };
        state.gate_high_risk_streak = high_risk_streak;

        subject_controller::GateSignals {
            anchor_hits,
            novelty_score,
            uncertainty_score,
            risk_score,
            tool_misuse_risk,
            requires_audit,
            low_evidence,
            candidate_disagreement,
            recent_tool_failures,
            topic_shift_detected,
            toolchain_unseen,
            new_objective,
            high_risk_streak,
            default_soft: settings.gate_default_soft.unwrap_or(true),
            self_report_channel: settings.self_report_channel.unwrap_or(true),
            qualia_tag,
            qualia_intensity,
            qualia_confidence,
            qualia_novelty_delta,
            qualia_uncertainty_delta,
        }
    }

    pub(super) fn is_meaningful_run(
        &self,
        input_kind: CoreInputKind,
        outcomes: &[Outcome],
        response: &str,
    ) -> bool {
        if matches!(input_kind, CoreInputKind::User) {
            return true;
        }
        if !outcomes.is_empty() {
            return true;
        }
        !response.trim().is_empty()
    }

    pub(super) async fn mark_candidate_outcomes(
        &self,
        decision: &KernelDecision,
        accepted_label: &str,
        rejected_label: &str,
    ) {
        for candidate in decision.accepted.iter() {
            let _ = self
                .db
                .update_inner_monologue_candidate_outcome(&candidate.id, accepted_label, None)
                .await;
            if is_state_change_candidate(&candidate.kind) {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "loop_delta_applied",
                        "candidate_id": candidate.id,
                        "candidate_kind": format!("{:?}", candidate.kind),
                        "source": candidate.source,
                        "outcome": accepted_label,
                    }),
                )
                .await;
            }
        }
        for rejected in decision.rejected.iter() {
            let _ = self
                .db
                .update_inner_monologue_candidate_outcome(&rejected.id, rejected_label, Some(&rejected.reason))
                .await;
            if is_state_change_candidate(&rejected.kind) {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "loop_delta_rejected",
                        "candidate_id": rejected.id,
                        "candidate_kind": format!("{:?}", rejected.kind),
                        "reason": rejected.reason,
                        "outcome": rejected_label,
                    }),
                )
                .await;
            }
        }
    }

    pub(super) async fn maybe_semantic_promotion_candidate(
        &self,
        conversation_id: &str,
        state: &KernelState,
        settings: &crate::models::Settings,
        created_at: &mut i64,
    ) -> Option<Candidate> {
        let now = Utc::now();
        let last_promo = state
            .last_semantic_promotion_at
            .as_deref()
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
            .map(|ts| ts.with_timezone(&Utc));
        let due = last_promo
            .map(|ts| now.signed_duration_since(ts).num_seconds() >= SLOW_LAYER_INTERVAL_SECS)
            .unwrap_or(true);
        if !due {
            return None;
        }

        if state.recent_outcomes.is_empty() {
            return None;
        }
        let success_count = state
            .recent_outcomes
            .iter()
            .rev()
            .take(5)
            .filter(|o| o.success)
            .count();
        if success_count < 2 {
            return None;
        }

        let inner_summary_raw = self
            .db
            .get_inner_summary(conversation_id)
            .await
            .ok()
            .flatten()
            .unwrap_or_else(|| "{}".to_string());
        let inner_summary = InnerSummary::from_json(&inner_summary_raw);
        let mut query_parts = Vec::new();
        if !inner_summary.focus.trim().is_empty() {
            query_parts.push(inner_summary.focus.clone());
        }
        if !inner_summary.next_moves.is_empty() {
            query_parts.push(inner_summary.next_moves.join(" "));
        }
        if !inner_summary.open_questions.is_empty() {
            query_parts.push(inner_summary.open_questions.join(" "));
        }
        if query_parts.is_empty() {
            if let Some(last) = state.last_user_input.as_deref() {
                if !last.trim().is_empty() {
                    query_parts.push(last.to_string());
                }
            }
        }
        let query = query_parts.join(" | ").trim().to_string();
        if query.is_empty() {
            return None;
        }

        let api = crate::core::memory::api::MemoryApi::new(
            self.db.pool.clone(),
            Some(self.model_client.clone()),
            format!("semantic_core:{}", conversation_id),
        )
        .await;
        let intent = crate::core::memory::api::infer_query_intent(&query);
        let packet = api
            .retrieve(&query, &[crate::core::memory::types::Scope::Session, crate::core::memory::types::Scope::Global], intent)
            .await
            .ok()?;
        let memory_context = crate::core::memory::inject_context::format_for_prompt_limited(&packet, 8, 5);
        if memory_context.trim().is_empty() {
            return None;
        }

        let (summary_model, summary_url) = select_summary_model(settings);
        let system_prompt = "Summarize the semantic memory context into a compact, stable list of facts. Avoid speculation, questions, or advice.";
        let user_prompt = format!(
            "Semantic memory context:\n{}\n\nReturn only the compact summary.",
            memory_context
        );
        let (user_prompt, prompt_truncated) = cap_summary_prompt(system_prompt, &user_prompt, settings);
        if prompt_truncated {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "summary",
                None,
                None,
                json!({
                    "event": "summary_prompt_capped",
                    "cap_tokens": summary_prompt_cap_tokens(settings),
                    "source": "semantic_promotion",
                }),
            )
            .await;
        }
        let request = ChatCompletionRequest {
            model: summary_model,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_prompt,
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
            json_strict: Some(false),
                skip_sanitization: None,
            run_id: None,
            request_label: Some("semantic_promotion_summary".to_string()),
        };

        let response = self
            .model_client
            .chat_with_meta(&summary_url, settings.api_key.as_deref(), &request)
            .await
            .ok()?;
        let summary = response.content.trim().to_string();
        if summary.is_empty() {
            return None;
        }

        let evidence_event_ids = self
            .db
            .get_recent_user_evidence_ids(conversation_id, 2)
            .await;
        Some(self.make_candidate(
            CandidateKind::PromoteSemantic,
            json!({
                "summary": summary,
                "evidence_event_ids": evidence_event_ids,
            }),
            "slow_layer_promotion",
            created_at,
        ))
    }

    pub(super) async fn build_thread_context_snapshot(
        &self,
        conversation_id: &str,
        goal: &str,
        _settings: &crate::models::Settings,
    ) -> String {
        let inner_summary_raw = self
            .db
            .get_inner_summary(conversation_id)
            .await
            .ok()
            .flatten()
            .unwrap_or_else(|| "{}".to_string());

        let episodic = self
            .db
            .search_episodic_events(None, Some(conversation_id), None, None, None, None, 5)
            .await
            .unwrap_or_default();
        let episodic_snippets = episodic
            .iter()
            .filter_map(|event| {
                event
                    .payload
                    .get("summary_snippet")
                    .and_then(|v| v.as_str())
                    .map(|s| s.replace('\n', " "))
            })
            .collect::<Vec<_>>();

        let mut semantic_snippets: Vec<String> = Vec::new();
        let api = crate::core::memory::api::MemoryApi::new(
            self.db.pool.clone(),
            Some(self.model_client.clone()),
            format!("thread:{}", conversation_id),
        )
        .await;
        let intent = crate::core::memory::api::infer_query_intent(goal);
        if let Ok(packet) = api.retrieve(goal, &[crate::core::memory::types::Scope::Session, crate::core::memory::types::Scope::Global], intent).await {
            for fact in packet.facts.iter().take(3) {
                semantic_snippets.push(format!("{}: {} = {}", fact.entity_label, fact.key, fact.value));
            }
            for rel in packet.relations.iter().take(3) {
                let participants = rel
                    .participants
                    .iter()
                    .map(|p| format!("{}:{}", p.role, p.entity_label))
                    .collect::<Vec<_>>()
                    .join(", ");
                semantic_snippets.push(format!("{}({})", rel.rel_type, participants));
            }
        } else if let Ok(core) = self.db.get_semantic_core().await {
            if !core.trim().is_empty() {
                semantic_snippets.push(core.chars().take(240).collect());
            }
        }

        let payload = json!({
            "goal": goal,
            "parent_inner_summary": inner_summary_raw,
            "episodic_snippets": episodic_snippets,
            "semantic_snippets": semantic_snippets,
            "excluded": ["conversation_transcript"],
        });

        payload.to_string()
    }

    pub(super) async fn deliberate_input(
        &self,
        input: &str,
        input_kind: CoreInputKind,
        input_source: &str,
        conversation_id: &str,
        run_id: &str,
        assistant_message_id: Option<&str>,
        settings: &crate::models::Settings,
        original_input_override: Option<&str>,
        state: &mut KernelState,
        policy_addendum: Option<String>,
    ) -> Result<(ChatResponseMeta, Vec<ToolCall>, CorePromptBuild), String> {
        let current_time = chrono::Local::now().format("%Y-%m-%d %I:%M %p %Z").to_string();
        let original_input = original_input_override.unwrap_or(input).to_string();
        let semantic_hint = self
            .build_deliberation_semantic_hint(conversation_id, input)
            .await;
        let reflection_enabled = settings
            .stability_introspection_structured
            .unwrap_or(true);
        if reflection_enabled {
            let db = self.db.clone();
            let model_client = self.model_client.clone();
            let app_handle = self.app_handle.clone();
            let conversation_id = conversation_id.to_string();
            let input = input.to_string();
            let state_snapshot = state.clone();
            let settings_snapshot = settings.clone();
            let run_id = run_id.to_string();
            tokio::spawn(async move {
                let kernel = Kernel::new(db, model_client, app_handle);
                let reflection = tokio::time::timeout(
                    Duration::from_secs(20),
                    kernel.reflect_working_memory(&conversation_id, &input, &state_snapshot, &settings_snapshot),
                )
                .await;
                match reflection {
                    Ok(Some(wmb)) => {
                        let mut updated_state = kernel.load_state(&conversation_id).await;
                        updated_state.working_memory = Some(wmb);
                        kernel.persist_state(&updated_state).await;
                    }
                    Ok(None) => {}
                    Err(_) => {
                        let _ = system_log::log_event(
                            &kernel.db.pool,
                            Some(&kernel.app_handle),
                            "warn",
                            "kernel",
                            Some(&run_id),
                            None,
                            json!({
                                "event": "introspection_reflection_timeout",
                            }),
                        )
                        .await;
                    }
                }
            });
        }
        let introspection_enabled = settings.enable_introspection.unwrap_or(true);
        let diagnostics_breaker_active = state.diagnostics_disabled_turns_remaining > 0;
        let confidence_threshold = settings.introspection_confidence_threshold.unwrap_or(0.5);
        let drift_threshold = settings.introspection_drift_threshold.unwrap_or(0.6);
        let ambiguity_threshold = settings.introspection_ambiguity_threshold.unwrap_or(0.5);
        let mut trigger_reasons: Vec<String> = Vec::new();
        if let Some(controller) = state.controller_state.as_ref() {
            if controller.confidence < confidence_threshold {
                trigger_reasons.push(format!(
                    "confidence_below:{:.2}",
                    controller.confidence
                ));
            }
            if controller.drift_score > drift_threshold {
                trigger_reasons.push(format!("drift_above:{:.2}", controller.drift_score));
            }
        }
        let ambiguity = ambiguity_score(input);
        if ambiguity > ambiguity_threshold {
            trigger_reasons.push(format!("ambiguity_score:{:.2}", ambiguity));
        }
        if let Some(force) = state.introspection_force.as_deref() {
            trigger_reasons.push(format!("forced:{}", force));
        }

        let user_requested_introspection = is_introspection_request(input);
        let should_introspect = introspection_enabled
            && !diagnostics_breaker_active
            && matches!(input_kind, CoreInputKind::User)
            && (user_requested_introspection || !trigger_reasons.is_empty());

        let mut introspection_summary = if should_introspect
            && settings.stability_introspection_structured.unwrap_or(true)
        {
            self.build_introspection_summary(conversation_id, settings, &state)
                .await
        } else {
            None
        };
        let ignition_active = core_workspace::build_workspace_state(&state, None).ignition.active;
        if !ignition_active {
            introspection_summary = None;
        }
        if should_introspect && !trigger_reasons.is_empty() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                None,
                json!({
                    "event": "introspection_triggered",
                    "reasons": trigger_reasons,
                }),
            )
            .await;
        }
        let self_audit_mode = matches!(input_kind, CoreInputKind::User) && is_self_audit_request(input);
        let calculator_mode = matches!(input_kind, CoreInputKind::User) && is_calculator_prompt(input);
        if matches!(input_kind, CoreInputKind::User) {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                None,
                json!({
                    "event": "calculator_routing",
                    "calculator_mode": calculator_mode,
                    "input_len": input.len(),
                }),
            )
            .await;
            if self_audit_mode {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(run_id),
                    None,
                    json!({
                        "event": "self_audit_used",
                        "input_len": input.len(),
                    }),
                )
                .await;
            }
        }
        let focused_task_mode = matches!(
            input_kind,
            CoreInputKind::ToolResult | CoreInputKind::ToolError | CoreInputKind::SystemContext
        );
        let prompt_mode = if calculator_mode {
            "calculator"
        } else if self_audit_mode {
            "self_audit"
        } else if focused_task_mode {
            "focused_task"
        } else {
            "normal"
        };
        let mut policy_notes = if calculator_mode {
            Some(
                "Calculator mode: Output JSON only with keys {final, required_slots, assumptions?, defaults_used?}. \
required_slots must list slot keys for any numeric inputs needed. If any slots are missing, leave final empty and list required_slots. \
Never ask preference questions. Never fabricate numeric inputs."
                    .to_string(),
            )
        } else {
            None
        };
        if self_audit_mode {
            let audit_note = "Self-audit mode: Answer only from Workspace State, Capability Manifest, and Controller State. \
Do not invent capabilities or gaps. If something is unknown, say so. Do not call tools.";
            if let Some(existing) = policy_notes.as_mut() {
                existing.push_str("\n");
                existing.push_str(audit_note);
            } else {
                policy_notes = Some(audit_note.to_string());
            }
        }
        let allow_diagnostics = self_audit_mode && !diagnostics_breaker_active;
        if allow_diagnostics {
            if let Some(gate) = state.controller_gate.as_ref() {
                let note = format!(
                    "Controller gate: throttle_tools={}, throttle_threads={}, throttle_asks={}, prefer_verification={}, reanchor={}, autonomy_scale={:.2}",
                    gate.throttle_tools,
                    gate.throttle_threads,
                    gate.throttle_asks,
                    gate.prefer_verification,
                    gate.reanchor,
                    gate.autonomy_scale
                );
                if let Some(existing) = policy_notes.as_mut() {
                    existing.push_str("\n");
                    existing.push_str(&note);
                } else {
                    policy_notes = Some(note);
                }
            }
        }
        if let Some(addendum) = policy_addendum.as_deref() {
            if let Some(existing) = policy_notes.as_mut() {
                existing.push_str("\n");
                existing.push_str(addendum);
            } else {
                policy_notes = Some(addendum.to_string());
            }
        }
        if matches!(input_kind, CoreInputKind::User) {
            if let Some(note) = self
                .intent_sanity_check_note(conversation_id, input, run_id, None)
                .await
            {
                if let Some(existing) = policy_notes.as_mut() {
                    existing.push_str("\n");
                    existing.push_str(&note);
                } else {
                    policy_notes = Some(note);
                }
            }
        }
        let compact_decision = self
            .evaluate_compact_prompt(
                input,
                input_kind,
                self_audit_mode,
                calculator_mode,
                conversation_id,
                state,
                settings,
            )
            .await;
        let prompt_layout = if compact_decision.use_compact {
            PromptLayout::Compact
        } else {
            PromptLayout::Full
        };
        let monologue_intent = if matches!(input_kind, CoreInputKind::User | CoreInputKind::SystemContext) {
            match self.db.get_latest_monologue_intent(conversation_id).await {
                Ok(Some((intent_id, prompt, intent_kind, _bridge_id, created_at))) => {
                    let age_seconds = prompt_age_seconds(&created_at).unwrap_or(0);
                    if age_seconds > MONOLOGUE_INTENT_MAX_AGE_SECS {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(run_id),
                            None,
                            json!({
                                "event": "intent_dropped",
                                "candidate_id": intent_id,
                                "candidate_kind": intent_kind,
                                "reason": "stale_monologue_intent",
                                "age_seconds": age_seconds,
                            }),
                        )
                        .await;
                        None
                    } else {
                        let trimmed = prompt.trim().to_string();
                        if trimmed.is_empty() {
                            None
                        } else if let Some(kind) = intent_kind.as_deref().filter(|k| !k.trim().is_empty()) {
                            Some(format!("intent_kind: {}\n{}", kind, trimmed))
                        } else {
                            Some(trimmed)
                        }
                    }
                }
                _ => None,
            }
        } else {
            None
        };
        let digest = self
            .select_monologue_digest(conversation_id, &state, MONOLOGUE_DIGEST_TTL_SECS)
            .await;
        if digest.stale {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                None,
                json!({
                    "event": "monologue_digest_stale",
                    "source": digest.source,
                    "age_secs": digest.age_secs,
                    "entry_id": digest.entry_id,
                    "stream": digest.stream,
                }),
            )
            .await;
        }
        let monologue_digest = digest
            .text
            .as_deref()
            .map(|text| summarize_snippet(text, 240));
        let monologue_digest_len = monologue_digest.as_ref().map(|s| s.len()).unwrap_or(0);
        if monologue_digest_len == 0 {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                None,
                json!({
                    "event": "monologue_digest_missing",
                    "source": digest.source,
                    "age_secs": digest.age_secs,
                    "entry_id": digest.entry_id,
                    "stream": digest.stream,
                }),
            )
            .await;
        }
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(run_id),
            None,
            json!({
                "event": "monologue_digest_selected",
                "source": digest.source,
                "age_secs": digest.age_secs,
                "entry_id": digest.entry_id,
                "stream": digest.stream,
                "digest_len": monologue_digest_len,
                "stale": digest.stale,
            }),
        )
        .await;

        let now = Utc::now();
        let mut proaction_state = self.load_proaction_state().await;
        let hourly_metrics = self.compute_proaction_metrics(60).await;
        let eval_hour = now.format("%Y-%m-%dT%H").to_string();
        if proaction_state.monologue_last_eval_hour.as_deref() != Some(eval_hour.as_str()) {
            let failures = monologue_acceptance_failures(&hourly_metrics);
            let passed = failures.is_empty();
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                None,
                json!({
                    "event": "monologue_acceptance_window",
                    "conversation_id": conversation_id,
                    "window_minutes": 60,
                    "hour": eval_hour,
                    "passed": passed,
                    "failures": failures,
                    "metrics": {
                        "ds_fts_ratio": hourly_metrics.monologue_ds_fts_ratio,
                        "tick_end_rate": hourly_metrics.monologue_tick_end_rate,
                        "timeouts": hourly_metrics.monologue_timeouts,
                        "suppression_rate": hourly_metrics.monologue_suppression_rate,
                        "drift_reanchor_rate": hourly_metrics.monologue_drift_reanchor_rate,
                        "safety_violations": hourly_metrics.monologue_safety_violations,
                        "attempted_turns": hourly_metrics.monologue_attempted_turns,
                    }
                }),
            )
            .await;
            proaction_state.monologue_last_eval_hour = Some(eval_hour);
            if passed {
                proaction_state.monologue_bad_windows = 0;
            } else {
                proaction_state.monologue_bad_windows = proaction_state.monologue_bad_windows.saturating_add(1);
            }
            if !passed && proaction_state.monologue_bad_windows >= 2 {
                let from_level = proaction_state.monologue_relaxation_level;
                let to_level = (from_level - 1).max(0);
                if to_level < from_level {
                    proaction_state.monologue_relaxation_level = to_level;
                    proaction_state.monologue_bad_windows = 0;
                    state.monologue_relaxation_level = to_level;
                    self.persist_state_with_owner(state, "proaction").await;
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        Some(run_id),
                        None,
                        json!({
                            "event": "monologue_relaxation_reverted",
                            "conversation_id": conversation_id,
                            "from_level": from_level,
                            "to_level": to_level,
                        }),
                    )
                    .await;
                }
            }
            self.persist_proaction_state(&proaction_state).await;
        }
        let monologue_digest_used = monologue_digest
            .as_deref()
            .map(|s| !s.trim().is_empty() && s.trim() != "None")
            .unwrap_or(false);
        let monologue_digest_hash = if monologue_digest_used {
            monologue_digest.as_ref().map(|text| hash_payload(text))
        } else {
            None
        };
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(run_id),
            None,
            json!({
                "event": "compact_prompt_eval",
                "eligible": compact_decision.use_compact,
                "disqualifiers": compact_decision.disqualifiers,
                "force_reasons": compact_decision.force_reasons,
                "anchor_hits": compact_decision.anchor_hits,
                "pending_prompts": compact_decision.pending_prompts,
                "recent_memory_write": compact_decision.recent_memory_write,
                "workspace_delta": compact_decision.workspace_delta,
                "input_len": input.chars().count(),
            }),
        )
        .await;
        let redirect_focus = if state.user_redirect_turns_remaining > 0 {
            if let Some(focus) = state.redirect_focus.as_deref().filter(|s| !s.trim().is_empty()) {
                Some(summarize_snippet(focus, 160))
            } else {
                let trimmed = input.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(summarize_snippet(trimmed, 160))
                }
            }
        } else {
            None
        };
        let subject_snapshot = sqlx::query_scalar::<_, String>(
            "SELECT subject_state_json FROM subject_snapshots
             WHERE conversation_id = ?
             ORDER BY datetime(timestamp) DESC
             LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten();
        let mut attention_schema_summary: Option<String> = None;
        let mut workspace_contributors_summary: Option<String> = None;
        let mut reflective_narrative: Option<String> = None;
        let mut reflective_narrative_evidence_ids: Vec<i64> = Vec::new();
        let world_model_snapshot: Option<String> = subject_snapshot
            .as_ref()
            .and_then(|raw| crate::core::world_model::snapshot_from_subject_state_json(raw))
            .map(|snapshot| crate::core::world_model::render_world_model_prompt(&snapshot));
        if let Some(runtime) = state.workspace_meta.runtime.as_ref() {
            if let Some(obj) = runtime.as_object() {
                attention_schema_summary = obj
                    .get("attention_schema_summary")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                workspace_contributors_summary = obj
                    .get("contributors_summary")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
            }
        }
        let mut parsed_subject: Option<subject_state::SubjectState> = None;
        if attention_schema_summary.is_none() {
            if let Some(snapshot) = subject_snapshot.as_ref() {
                if let Ok(value) = serde_json::from_str::<Value>(snapshot) {
                    if let Some(state_value) = value.get("state") {
                        if let Ok(subject) =
                            serde_json::from_value::<subject_state::SubjectState>(state_value.clone())
                        {
                            parsed_subject = Some(subject);
                        }
                    }
                }
            }
        }
        if attention_schema_summary.is_none() {
            if let Some(subject) = parsed_subject.as_ref() {
                attention_schema_summary = Some(
                    crate::core::attention_schema::summarize_for_prompt(
                        &subject.attention_schema,
                    ),
                );
            }
        }
        if let Some((value, evidence_ids)) = self
            .db
            .get_latest_self_memory_fact("reflective_narrative")
            .await
        {
            if !value.trim().is_empty() && value.trim() != "None" {
                reflective_narrative = Some(value);
                reflective_narrative_evidence_ids = evidence_ids;
            }
        }
        let gate_decision = sqlx::query(
            "SELECT decision_id, decision, evidence_refs_json, metrics_json
             FROM gate_decisions
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten()
        .map(|row| {
            let decision_id: String = row.try_get("decision_id").unwrap_or_default();
            let decision: String = row.try_get("decision").unwrap_or_default();
            let evidence_refs_json: String = row.try_get("evidence_refs_json").unwrap_or_default();
            let metrics_json: String = row.try_get("metrics_json").unwrap_or_default();
            json!({
                "decision_id": decision_id,
                "decision": decision,
                "evidence_refs": evidence_refs_json,
                "metrics": metrics_json,
            })
            .to_string()
        });
        let feedback_bundle = self
            .build_feedback_bundle(
                conversation_id,
                &state,
                introspection_summary.as_deref(),
            )
            .await;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(&run_id),
            None,
            json!({
                "event": "feedback_bundle_built",
                "conversation_id": conversation_id,
                "bundle": feedback_bundle.payload,
            }),
        )
        .await;
        let self_awareness_mode = settings
            .self_awareness_expression_mode
            .as_deref()
            .unwrap_or("conservative");
        let explicit_self_awareness =
            matches!(input_kind, CoreInputKind::User) && is_self_awareness_query(&input);
        let intent_gate_detection = if matches!(input_kind, CoreInputKind::User) {
            crate::core::prompt_builder::detect_context_intent(&input, false, false)
        } else {
            crate::core::prompt_builder::IntentDetection {
                tags: Vec::new(),
                matched_rules: Vec::new(),
            }
        };
        let (self_awareness_requested, self_awareness_requested_reason) =
            is_self_awareness_gate_requested(explicit_self_awareness, &intent_gate_detection.tags);
        let self_awareness_allowed =
            settings.self_report_channel.unwrap_or(true)
                && !self_awareness_mode.eq_ignore_ascii_case("conservative");
        let self_awareness_query = self_awareness_requested && self_awareness_allowed;
        let wave_state = self.wave_state_for_prompt(Some(&run_id), None).await;
        let prompt_self_awareness = self_awareness_query;
        let self_awareness_hint = self_awareness_requested;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(&run_id),
            None,
            json!( {
                "event": "self_awareness_gate",
                "requested": self_awareness_requested,
                "allowed": self_awareness_allowed,
                "requested_reason": self_awareness_requested_reason.unwrap_or("none"),
                "mode": settings
                    .self_awareness_expression_mode
                    .as_deref()
                    .unwrap_or("conservative"),
            }),
        )
        .await;
        let hydration_mode = settings
            .context_hydration_mode
            .as_deref()
            .unwrap_or("shadow");
        let hydration_off = hydration_mode.eq_ignore_ascii_case("off");
        let hydration_shadow = hydration_mode.eq_ignore_ascii_case("shadow");
        let (intent_detection, context_selected_sections, fallback_reason) = if !hydration_off {
            crate::core::prompt_builder::compute_context_hydration(
                &input,
                prompt_self_awareness,
                self_awareness_hint,
            )
        } else {
            (
                crate::core::prompt_builder::IntentDetection {
                    tags: Vec::new(),
                    matched_rules: Vec::new(),
                },
                Vec::new(),
                None,
            )
        };
        let context_intent_tags = intent_detection.tags.clone();
        let mut hydrated_context: Option<String> = None;
        if !context_selected_sections.is_empty() && !hydration_shadow {
            hydrated_context = self
                .prefetch_context_hydration(
                    conversation_id,
                    Some(&run_id),
                    &context_selected_sections,
                    &context_intent_tags,
                )
                .await;
        }
        if let Some(reason) = fallback_reason.as_deref() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(&run_id),
                None,
                json!( {
                    "event": "context_hydration_fallback",
                    "reason": reason,
                    "selected_sections": context_selected_sections,
                }),
            )
            .await;
        }
        let prompt_input = CorePromptInput {
            content: input.to_string(),
            kind: input_kind,
            source: input_source.to_string(),
            self_awareness: prompt_self_awareness,
            self_awareness_hint,
            anchor_hits: compact_decision.anchor_hits,
            original_input,
            current_time: Some(current_time),
            semantic_hint: Some(semantic_hint),
            introspection_summary,
            monologue_intent,
            monologue_digest,
            prompt_mode: Some(prompt_mode.to_string()),
            task_phase: Some(task_phase_label(&state.task_phase).to_string()),
            missing_slots: Some(state.missing_slots.clone()),
            resolution_mode: state.resolution_mode.clone(),
            policy_notes,
            redirect_focus,
            allow_diagnostics,
            world_model_snapshot,
            subject_snapshot,
            gate_decision,
            feedback_bundle: Some(feedback_bundle.prompt_text),
            qualia_snapshot: Some(feedback_bundle.qualia_snapshot),
            attention_schema_summary,
            workspace_contributors_summary,
            wave_state,
            reflective_narrative,
            reflective_narrative_evidence_ids,
            hydrated_context,
        };
        let _ = advance_run_phase(
            &self.db.pool,
            Some(&self.app_handle),
            run_id,
            RunPhase::PromptBuild,
            Some("prompt_build"),
        )
        .await;
        let prompt_started = Instant::now();
        let prompt_build =
            build_core_system_message_with_layout(&self.db, conversation_id, &prompt_input, prompt_layout).await?;
        let system_message = prompt_build.system_message.clone();
        state.prompt_section_hashes = prompt_build.section_hashes.clone();
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(run_id),
            None,
            json!({
                "event": "prompt_build",
                "prompt_source": prompt_build.prompt_source,
                "primary_prompt_hash": prompt_build.primary_prompt_hash,
                "memory_prompt_hash": prompt_build.memory_prompt_hash,
                "canonical_primary_hash": prompt_build.canonical_primary_hash,
                "override_active": prompt_build.override_active,
                "override_mismatch": prompt_build.override_mismatch,
                "section_hashes": prompt_build.section_hashes,
                "trim_events": prompt_build.trim_events,
                "total_chars": prompt_build.total_chars,
                "total_tokens": prompt_build.total_tokens,
            }),
        )
        .await;
        if let Some(plan) = prompt_build.context_hydration.as_ref() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                None,
                json!({
                    "event": "context_hydration_plan",
                    "mode": plan.mode,
                    "intent_tags": plan.intent_tags,
                    "matched_rules": plan.matched_rules,
                    "selected_sections": plan.selected_sections,
                    "skipped_sections": plan.skipped_sections,
                    "max_sections": plan.max_sections,
                    "fallback_reason": plan.fallback_reason,
                }),
            )
            .await;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                None,
                json!({
                    "event": "intent_tagging",
                    "intent_tags": plan.intent_tags,
                    "input_kind": format!("{:?}", input_kind),
                }),
            )
            .await;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                None,
                json!({
                    "event": "intent_tagging_rules",
                    "matched_rules": plan.matched_rules,
                    "input_kind": format!("{:?}", input_kind),
                }),
            )
            .await;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                None,
                json!({
                    "event": "hydration_intent_selected",
                    "intent_tags": plan.intent_tags,
                    "selected_sections": plan.selected_sections,
                }),
            )
            .await;
            if !plan.intent_tags.is_empty() {
                let summary = format!("intent_tags: {}", plan.intent_tags.join(", "));
                let _ = self
                    .db
                    .upsert_user_intent_summary(conversation_id, &summary, false, &[])
                    .await;
                for tag in plan.intent_tags.iter() {
                    let _ = self
                        .db
                        .upsert_context_tag(
                            conversation_id,
                            tag,
                            0.6,
                            true,
                            &[],
                            Some("context_hydration"),
                        )
                        .await;
                }
            }
            if plan.intent_tags.iter().any(|t| t == "planning")
                && state.workspace_goal_stack.is_empty()
            {
                let goal_text = format!(
                    "Plan: {}",
                    summarize_snippet(&input, 120)
                );
                let steps = vec![
                    crate::models::GoalStep {
                        text: "Clarify objective and constraints".to_string(),
                        status: None,
                        evidence_event_ids: Vec::new(),
                        belief_ids: Vec::new(),
                        completed_at: None,
                    },
                    crate::models::GoalStep {
                        text: "Outline concrete steps".to_string(),
                        status: None,
                        evidence_event_ids: Vec::new(),
                        belief_ids: Vec::new(),
                        completed_at: None,
                    },
                    crate::models::GoalStep {
                        text: "Confirm success criteria".to_string(),
                        status: None,
                        evidence_event_ids: Vec::new(),
                        belief_ids: Vec::new(),
                        completed_at: None,
                    },
                ];
                state.workspace_goal_stack = vec![crate::models::GoalStackItem {
                    goal: goal_text.clone(),
                    steps,
                    current_step_index: 0,
                    status: None,
                    evidence_event_ids: Vec::new(),
                    belief_ids: Vec::new(),
                    updated_at: Some(Utc::now().to_rfc3339()),
                }];
                state.workspace_goal_thread = Some(goal_text.clone());
                let workspace_state = crate::models::WorkspaceState {
                    conversation_id: conversation_id.to_string(),
                    goal_thread: state.workspace_goal_thread.clone(),
                    active_plan_id: state.workspace_active_plan_id.clone(),
                    goal_stack: state.workspace_goal_stack.clone(),
                    open_questions: state.workspace_open_questions.clone(),
                    active_hypotheses: state.workspace_active_hypotheses.clone(),
                    working_set_topics: state.workspace_working_set_topics.clone(),
                    current_focus: state.workspace_current_focus.clone(),
                    focus_rationale: state.workspace_focus_rationale.clone(),
                    workspace_meta: state.workspace_meta.clone(),
                    updated_at: None,
                };
                let _ = self.db.set_workspace_state(&workspace_state).await;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(run_id),
                    None,
                    json!({
                        "event": "goal_stack_seeded",
                        "goal": goal_text,
                    }),
                )
                .await;
            }
        }
        if monologue_digest_used {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                None,
                json!({
                    "event": "monologue_digest_injected",
                    "prompt_hash": prompt_build.primary_prompt_hash,
                    "digest_hash": monologue_digest_hash,
                    "digest_len": monologue_digest_len,
                }),
            )
            .await;
        }
        if prompt_build.override_mismatch && !prompt_build.override_guard_skipped.is_empty() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "kernel",
                Some(run_id),
                None,
                json!({
                    "event": "prompt_override_guard_applied",
                    "skipped_sections": prompt_build.override_guard_skipped,
                    "override_hash": prompt_build.override_hash,
                    "canonical_primary_hash": prompt_build.canonical_primary_hash,
                }),
            )
            .await;
        }
        for event in prompt_build.trim_events.iter() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                None,
                json!({
                    "event": "prompt_trim",
                    "title": event.title,
                    "original_chars": event.original_chars,
                    "trimmed_chars": event.trimmed_chars,
                    "reason": event.reason,
                    "hash": event.hash,
                }),
            )
            .await;
            let is_critical = matches!(
                event.title.as_str(),
                "Response Style"
                    | "Identity Anchor"
                    | "SYMBIOTE_PHILOSOPHY"
                    | "SYMBIOTE_POLICY_SUMMARY"
                    | "Symbiote System Overview"
                    | "Tool Availability"
                    | "Working Memory"
                    | "User Input"
                    | "Rolling Summary"
                    | "Memory Context"
            ) || event.reason == "anchor_floor_exceeded";
            if is_critical {
                let hash_key = event.hash.as_deref().unwrap_or("unknown");
                let key = format!("{}::{}", event.title, hash_key);
                let (allow_log, suppressed) = rate_limit_event(
                    &PROMPT_TRIM_CRITICAL_RATE,
                    &key,
                    Duration::from_secs(PROMPT_TRIM_CRITICAL_WINDOW_SECS),
                );
                if allow_log {
                    if suppressed > 0 {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            Some(run_id),
                            None,
                            json!({
                                "event": "prompt_trim_critical_suppressed",
                                "title": event.title,
                                "hash": event.hash,
                                "count": suppressed,
                                "window_secs": PROMPT_TRIM_CRITICAL_WINDOW_SECS,
                            }),
                        )
                        .await;
                    }
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        Some(run_id),
                        None,
                        json!({
                            "event": "prompt_trim_critical",
                            "title": event.title,
                            "original_chars": event.original_chars,
                            "trimmed_chars": event.trimmed_chars,
                            "reason": event.reason,
                            "hash": event.hash,
                        }),
                    )
                    .await;
                }
            }
        }
        if prompt_build.prompt_overflow {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "kernel",
                Some(run_id),
                None,
                json!({
                    "event": "prompt_overflow",
                    "total_tokens": prompt_build.total_tokens,
                }),
            )
            .await;
        }
        let prompt_ms = prompt_started.elapsed().as_millis() as i64;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(run_id),
            None,
            json!({
                "event": "timing_prompt_build",
                "duration_ms": prompt_ms,
                "prompt_mode": prompt_mode,
                "compact": compact_decision.use_compact,
                "input_len": input.len(),
            }),
        )
        .await;
        self.update_latency_avg("prompt_build", prompt_ms).await;
        if prompt_ms > PERF_WARN_PROMPT_MS {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "perf",
                Some(run_id),
                None,
                json!({
                    "event": "performance_regression",
                    "stage": "prompt_build",
                    "duration_ms": prompt_ms,
                }),
            )
            .await;
        }
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(run_id),
            None,
            json!({
                "event": "binding_hashes",
                "workspace_hash": prompt_build.workspace_hash,
                "inner_summary_hash": prompt_build.inner_summary_hash,
                "rolling_summary_hash": prompt_build.rolling_summary_hash,
                "capability_manifest_hash": prompt_build.capability_manifest_hash,
            }),
        )
        .await;

        if let Some(message_id) = assistant_message_id.as_deref() {
            let updated = sqlx::query(
                "UPDATE messages
                 SET status = 'streaming'
                 WHERE message_id = ?
                   AND (status IS NULL OR status = 'pending')",
            )
            .bind(message_id)
            .execute(&self.db.pool)
            .await
            .map(|res| res.rows_affected())
            .unwrap_or(0);
            if updated > 0 {
                let _ = self.app_handle.emit("message_updated", ());
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(run_id),
                    None,
                    json!({
                        "event": "assistant_streaming_started",
                        "message_id": message_id,
                    }),
                )
                .await;
            }
        }

        let (tool_defs, tool_choice) = if self_audit_mode {
            (None, Some("none".to_string()))
        } else {
            (Some(self.tools.definitions_for_settings(settings)), Some("auto".to_string()))
        };
        let parse_error_count: i64 = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM system_logs
             WHERE json_extract(payload, '$.event') = 'prediction_generation_rejected'
               AND json_extract(payload, '$.reason') = 'json_parse_error'
               AND datetime(timestamp) >= datetime('now', '-1 hour')",
        )
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0);
        let (temperature, max_tokens) = if parse_error_count >= 3 {
            (Some(0.1), Some(300))
        } else {
            (None, None)
        };

        let request = ChatCompletionRequest {
            model: settings
                .active_model_id
                .clone()
                .unwrap_or_else(|| "default".to_string()),
            messages: vec![ChatMessage {
                role: "system".to_string(),
                content: system_message,
            }],
            stream: false,
            temperature,
            top_p: None,
            max_tokens,
            response_format: None,
            tools: tool_defs,
            tool_choice,
            enable_thinking: None,
            prefill: None,
            skip_injection: None,
            skip_memory: Some(true),
            skip_reminders: Some(true),
            memory_expand: Some(focused_task_mode),
            allow_diagnostics: Some(allow_diagnostics),
            json_strict: Some(false),
            skip_sanitization: None,
            run_id: Some(run_id.to_string()),
            request_label: Some("primary_response".to_string()),
        };

        let _ = advance_run_phase(
            &self.db.pool,
            Some(&self.app_handle),
            run_id,
            RunPhase::ModelCall,
            Some("model_call"),
        )
        .await;

        let is_context_error = |err: &str| {
            let lowered = err.to_lowercase();
            lowered.contains("context") || lowered.contains("maximum context") || lowered.contains("context length") || lowered.contains("max tokens")
        };

        let response_meta = match if settings.streaming_enabled {
            self.model_client
                .chat_with_meta_stream(&settings.api_base_url, settings.api_key.as_deref(), &request)
                .await
        } else {
            self.model_client
                .chat_with_meta(&settings.api_base_url, settings.api_key.as_deref(), &request)
                .await
        } {
            Ok(meta) => meta,
            Err(err) if is_context_error(&err) => {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    Some(run_id),
                    None,
                    json!({
                        "event": "context_limit_retry",
                        "error": err,
                        "skip_injection": true,
                    }),
                )
                .await;
                if let Some(limit) = parse_context_limit_from_error(&err) {
                    let current_limit = token_estimator::context_limit_tokens(&settings) as i32;
                    if limit >= 128 && limit < current_limit {
                        let mut updated_settings = settings.clone();
                        updated_settings.model_context_limit = Some(limit);
                        match self.db.update_settings(updated_settings).await {
                            Ok(_) => {
                                let _ = system_log::log_event(
                                    &self.db.pool,
                                    Some(&self.app_handle),
                                    "info",
                                    "kernel",
                                    Some(run_id),
                                    None,
                                    json!({
                                        "event": "context_limit_adjusted",
                                        "previous_limit": current_limit,
                                        "new_limit": limit,
                                        "source": "context_error",
                                    }),
                                )
                                .await;
                            }
                            Err(err) => {
                                let _ = system_log::log_event(
                                    &self.db.pool,
                                    Some(&self.app_handle),
                                    "warn",
                                    "kernel",
                                    Some(run_id),
                                    None,
                                    json!({
                                        "event": "context_limit_adjust_failed",
                                        "error": err.to_string(),
                                    }),
                                )
                                .await;
                            }
                        }
                    }
                }
                let mut retry_request = request.clone();
                retry_request.skip_injection = Some(true);
                retry_request.skip_memory = Some(true);
                retry_request.skip_reminders = Some(true);
                retry_request.memory_expand = Some(false);
                if settings.streaming_enabled {
                    self.model_client
                        .chat_with_meta_stream(&settings.api_base_url, settings.api_key.as_deref(), &retry_request)
                        .await?
                } else {
                    self.model_client
                        .chat_with_meta(&settings.api_base_url, settings.api_key.as_deref(), &retry_request)
                        .await?
                }
            }
            Err(err) => return Err(err),
        };

        let tool_calls = response_meta
            .tool_calls
            .clone()
            .unwrap_or_default();

        Ok((response_meta, tool_calls, prompt_build))
    }

    pub(super) async fn compute_calculator_final(
        &self,
        input: &str,
        conversation_id: &str,
        run_id: &str,
        settings: &crate::models::Settings,
        state: &KernelState,
        resolved: &BTreeMap<String, ResolvedParam>,
        defaults_used: &[String],
    ) -> Result<ChatResponseMeta, String> {
        let mut slot_lines = Vec::new();
        for (slot, res) in resolved {
            if let Some(value) = res.value.as_ref() {
                slot_lines.push(format!(
                    "- {}: {} (source: {})",
                    slot,
                    format_slot_value(value),
                    res.source
                ));
            }
        }
        let slot_block = if slot_lines.is_empty() {
            "None".to_string()
        } else {
            slot_lines.join("\n")
        };
        let mut policy_notes = String::from(
            "Calculator compute step: Use only the slot values below. Do not invent numbers. Output JSON only with keys {final, assumptions?, defaults_used?}.",
        );
        policy_notes.push_str("\nSlot values:\n");
        policy_notes.push_str(&slot_block);
        let feedback_bundle = self
            .build_feedback_bundle(conversation_id, state, None)
            .await;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(run_id),
            None,
            json!({
                "event": "feedback_bundle_built",
                "conversation_id": conversation_id,
                "bundle": feedback_bundle.payload,
            }),
        )
        .await;

        let prompt_input = CorePromptInput {
            content: input.to_string(),
            kind: CoreInputKind::User,
            source: "calculator".to_string(),
            self_awareness: false,
            self_awareness_hint: false,
            anchor_hits: 0,
            original_input: input.to_string(),
            current_time: Some(chrono::Local::now().format("%Y-%m-%d %I:%M %p %Z").to_string()),
            semantic_hint: None,
            introspection_summary: None,
            monologue_intent: None,
            monologue_digest: None,
            prompt_mode: Some("calculator".to_string()),
            task_phase: Some(task_phase_label(&state.task_phase).to_string()),
            missing_slots: Some(Vec::new()),
            resolution_mode: state.resolution_mode.clone(),
            policy_notes: Some(policy_notes),
            redirect_focus: None,
            allow_diagnostics: false,
            world_model_snapshot: None,
            subject_snapshot: None,
            gate_decision: None,
            feedback_bundle: Some(feedback_bundle.prompt_text),
            qualia_snapshot: Some(feedback_bundle.qualia_snapshot),
            attention_schema_summary: None,
            workspace_contributors_summary: None,
            wave_state: None,
            reflective_narrative: None,
            reflective_narrative_evidence_ids: Vec::new(),
            hydrated_context: None,
        };
        let prompt_build = build_core_system_message(&self.db, conversation_id, &prompt_input).await?;
        let system_message = prompt_build.system_message.clone();

        let request = ChatCompletionRequest {
            model: settings
                .active_model_id
                .clone()
                .unwrap_or_else(|| "default".to_string()),
            messages: vec![ChatMessage {
                role: "system".to_string(),
                content: system_message,
            }],
            stream: false,
            temperature: None,
            top_p: None,
            max_tokens: None,
            response_format: None,
            tools: None,
            tool_choice: Some("none".to_string()),
            enable_thinking: None,
            prefill: None,
            skip_injection: None,
            skip_memory: Some(true),
            skip_reminders: Some(true),
            memory_expand: None,
            allow_diagnostics: Some(false),
            json_strict: Some(false),
                skip_sanitization: None,
            run_id: Some(run_id.to_string()),
            request_label: Some("calculator_final".to_string()),
        };

        let mut response_meta = self
            .model_client
            .chat_with_meta(&settings.api_base_url, settings.api_key.as_deref(), &request)
            .await?;

        if let Some(packet) = parse_calculator_packet(&response_meta.content) {
            let mut final_text = if !packet.final_text.is_empty() {
                packet.final_text
            } else {
                response_meta.content.clone()
            };
            let mut assumptions = packet.assumptions;
            if !defaults_used.is_empty() {
                let note = format!("used defaults for {}", defaults_used.join(", "));
                if !assumptions.iter().any(|a| a.contains("defaults")) {
                    assumptions.push(note);
                }
            }
            if !assumptions.is_empty() {
                final_text = format!("{}\n\nAssumptions: {}", final_text.trim(), assumptions.join("; "));
            }
            response_meta.content = final_text.clone();
            response_meta.content_no_tags = final_text.clone();
            response_meta.raw_content = final_text;
        }

        response_meta.tool_calls = None;
        Ok(response_meta)
    }

    pub(super) fn build_self_audit_response(
        &self,
        state: &KernelState,
        settings: &crate::models::Settings,
    ) -> String {
        let disable_working_hypothesis = settings.stability_disable_working_hypothesis.unwrap_or(true);
        let verified_focus = state
            .workspace_current_focus
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .filter(|_| meta_is_verified_field(state.workspace_meta.current_focus.as_ref()))
            .map(|s| s.to_string());
        let speculative_focus = state
            .workspace_current_focus
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .filter(|_| !meta_is_verified_field(state.workspace_meta.current_focus.as_ref()))
            .map(|s| s.to_string());

        let verified_goal = state
            .workspace_goal_thread
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .filter(|_| meta_is_verified_field(state.workspace_meta.goal_thread.as_ref()))
            .map(|s| s.to_string());
        let speculative_goal = state
            .workspace_goal_thread
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .filter(|_| !meta_is_verified_field(state.workspace_meta.goal_thread.as_ref()))
            .map(|s| s.to_string());

        let verified_rationale = state
            .workspace_focus_rationale
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .filter(|_| meta_is_verified_field(state.workspace_meta.focus_rationale.as_ref()))
            .map(|s| s.to_string());
        let speculative_rationale = state
            .workspace_focus_rationale
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .filter(|_| !meta_is_verified_field(state.workspace_meta.focus_rationale.as_ref()))
            .map(|s| s.to_string());

        let mut verified_topics = Vec::new();
        let mut speculative_topics = Vec::new();
        for (idx, topic) in state.workspace_working_set_topics.iter().enumerate() {
            let trimmed = topic.trim();
            if trimmed.is_empty() {
                continue;
            }
            if meta_is_verified_list(state.workspace_meta.working_set_topics.get(idx)) {
                verified_topics.push(trimmed.to_string());
            } else {
                speculative_topics.push(trimmed.to_string());
            }
        }

        let mut verified_questions = Vec::new();
        let mut speculative_questions = Vec::new();
        for (idx, question) in state.workspace_open_questions.iter().enumerate() {
            let trimmed = question.trim();
            if trimmed.is_empty() {
                continue;
            }
            if meta_is_verified_list(state.workspace_meta.open_questions.get(idx)) {
                verified_questions.push(trimmed.to_string());
            } else {
                speculative_questions.push(trimmed.to_string());
            }
        }

        let mut verified_hypotheses = Vec::new();
        let mut speculative_hypotheses = Vec::new();
        for hypothesis in state.workspace_active_hypotheses.iter() {
            let text = hypothesis.text.trim();
            if text.is_empty() {
                continue;
            }
            let formatted = format!("{} (conf {:.2})", text, hypothesis.confidence);
            if hypothesis_is_verified(hypothesis) {
                verified_hypotheses.push(formatted);
            } else {
                speculative_hypotheses.push(format_speculative_label(&formatted, disable_working_hypothesis));
            }
        }

        let list_or_none = |items: &[String]| {
            if items.is_empty() {
                "None".to_string()
            } else {
                items.join(" | ")
            }
        };

        let controller_line = if let Some(ctrl) = state.controller_state.as_ref() {
            format!(
                "confidence {:.2}, drift {:.2}, autonomy {:.2}, verification_needed {}, reanchor_needed {}, evidence_coverage {:.2}, telemetry_coverage {:.2}",
                ctrl.confidence,
                ctrl.drift_score,
                ctrl.autonomy_level,
                ctrl.verification_needed,
                ctrl.reanchor_needed,
                ctrl.evidence_coverage,
                ctrl.telemetry_coverage
            )
        } else {
            "None".to_string()
        };
        let gate_line = if let Some(gate) = state.controller_gate.as_ref() {
            format!(
                "throttle_tools {}, throttle_threads {}, throttle_asks {}, prefer_verification {}, reanchor {}, autonomy_scale {:.2}",
                gate.throttle_tools,
                gate.throttle_threads,
                gate.throttle_asks,
                gate.prefer_verification,
                gate.reanchor,
                gate.autonomy_scale
            )
        } else {
            "None".to_string()
        };

        let tool_names = self
            .tools
            .definitions_for_settings(settings)
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();
        let tool_list = if tool_names.is_empty() {
            "None".to_string()
        } else {
            tool_names.join(", ")
        };
        let memory_flags = format!(
            "episodic_enabled={}, episodic_injection_enabled={}, memory_claims_enabled={}, lexical_fallback_enabled={}",
            settings.episodic_enabled.unwrap_or(true),
            settings.episodic_injection_enabled.unwrap_or(true),
            settings.memory_claims_enabled.unwrap_or(true),
            settings.lexical_fallback_enabled.unwrap_or(true)
        );

        let mut lines = Vec::new();
        lines.push("Self-audit (runtime grounded):".to_string());
        lines.push("Workspace (verified):".to_string());
        lines.push(format!(
            "- Goal thread: {}",
            verified_goal.unwrap_or_else(|| "None".to_string())
        ));
        lines.push(format!(
            "- Current focus: {}",
            verified_focus.unwrap_or_else(|| "None".to_string())
        ));
        lines.push(format!(
            "- Focus rationale: {}",
            verified_rationale.unwrap_or_else(|| "None".to_string())
        ));
        lines.push(format!(
            "- Working set topics: {}",
            list_or_none(&verified_topics)
        ));
        lines.push(format!(
            "- Open questions: {}",
            list_or_none(&verified_questions)
        ));
        lines.push(format!(
            "- Active hypotheses: {}",
            list_or_none(&verified_hypotheses)
        ));
        lines.push("Workspace (speculative):".to_string());
        lines.push(format!(
            "- Goal thread: {}",
            speculative_goal.unwrap_or_else(|| "None".to_string())
        ));
        lines.push(format!(
            "- Current focus: {}",
            speculative_focus.unwrap_or_else(|| "None".to_string())
        ));
        lines.push(format!(
            "- Focus rationale: {}",
            speculative_rationale.unwrap_or_else(|| "None".to_string())
        ));
        lines.push(format!(
            "- Working set topics: {}",
            list_or_none(&speculative_topics)
        ));
        lines.push(format!(
            "- Open questions: {}",
            list_or_none(&speculative_questions)
        ));
        lines.push(format!(
            "- Active hypotheses: {}",
            list_or_none(&speculative_hypotheses)
        ));
        lines.push("Controller:".to_string());
        lines.push(format!("- State: {}", controller_line));
        lines.push(format!("- Gates: {}", gate_line));
        if let Some(ctrl) = state.controller_state.as_ref() {
            if !ctrl.missing_fields.is_empty() {
                lines.push(format!(
                    "- Missing fields: {}",
                    ctrl.missing_fields.join(", ")
                ));
            }
        }
        lines.push("Capabilities:".to_string());
        lines.push(format!("- Tools enabled: {}", tool_list));
        lines.push(format!("- Memory subsystems: {}", memory_flags));
        lines.push("If a capability is not listed above, it is unknown or unavailable.".to_string());
        lines.join("\n")
    }

    pub(super) async fn rewrite_summary_echo(
        &self,
        settings: &crate::models::Settings,
        user_input: &str,
        response: &str,
    ) -> Option<String> {
        let trimmed_response = response.trim();
        if trimmed_response.is_empty() {
            return None;
        }
        let (model, base_url) = select_summary_model(settings);
        let system_prompt = "Rewrite the assistant reply to answer the user's last message directly and conversationally. \
Do not summarize the conversation. Do not narrate system state. \
Do not mention telemetry, tools, manifests, KV memory, timestamps, run IDs, or logs. \
Keep it concise and aligned to the user's request.";
        let user_prompt = format!(
            "User message:\n{}\n\nDraft reply:\n{}\n\nRewrite the reply.",
            user_input.trim(),
            trimmed_response
        );
        let request = ChatCompletionRequest {
            model,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            stream: false,
            temperature: Some(0.2),
            top_p: None,
            max_tokens: None,
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
            json_strict: Some(false),
                skip_sanitization: None,
            run_id: None,
            request_label: Some("rewrite_summary_echo".to_string()),
        };
        let rewritten = self
            .model_client
            .chat_with_meta(&base_url, settings.api_key.as_deref(), &request)
            .await
            .ok()?;
        let content = rewritten.content.trim();
        if content.is_empty() {
            None
        } else {
            Some(content.to_string())
        }
    }

    pub(super) async fn rewrite_identity_inversion(
        &self,
        settings: &crate::models::Settings,
        user_input: &str,
        response: &str,
        assistant_name: &str,
        user_name: &str,
    ) -> Option<String> {
        let trimmed_response = response.trim();
        if trimmed_response.is_empty() {
            return None;
        }
        let (model, base_url) = select_summary_model(settings);
        let system_prompt = format!(
            "Rewrite the assistant reply so that the assistant identity is '{assistant}'. \
Do not speak as the user '{user}'. Do not use role labels like 'User:' or '{assistant}:' in the output. \
Answer the user's last message directly and conversationally. \
Do not mention telemetry, tools, manifests, KV memory, timestamps, run IDs, or logs unless explicitly requested.",
            assistant = assistant_name,
            user = user_name
        );
        let user_prompt = format!(
            "User message:\n{}\n\nDraft reply:\n{}\n\nRewrite the reply with correct identity.",
            user_input.trim(),
            trimmed_response
        );
        let request = ChatCompletionRequest {
            model,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            stream: false,
            temperature: Some(0.2),
            top_p: None,
            max_tokens: None,
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
            json_strict: Some(false),
                skip_sanitization: None,
            run_id: None,
            request_label: Some("rewrite_identity_inversion".to_string()),
        };
        let rewritten = self
            .model_client
            .chat_with_meta(&base_url, settings.api_key.as_deref(), &request)
            .await
            .ok()?;
        let content = rewritten.content.trim();
        if content.is_empty() {
            None
        } else {
            Some(content.to_string())
        }
    }


    pub(super) async fn capture_draft_response(
        &self,
        run_id: &str,
        trace_id: &str,
        reason: &str,
        content: &str,
        attempt: i64,
    ) {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return;
        }
        let artifact_id = Uuid::new_v4().to_string();
        let payload = json!({
            "content": trimmed,
            "reason": reason,
            "attempt": attempt,
            "content_len": trimmed.len(),
            "content_hash": hash_payload(trimmed),
            "captured_at": Utc::now().to_rfc3339(),
        });
        let _ = sqlx::query(
            "INSERT INTO artifacts (artifact_id, run_id, trace_id, type, schema_version, payload, produced_by, parent_artifact_ids, created_at)
             VALUES (?, ?, ?, 'draft_response', 1, ?, 'kernel', NULL, CURRENT_TIMESTAMP)",
        )
        .bind(&artifact_id)
        .bind(run_id)
        .bind(trace_id)
        .bind(payload.to_string())
        .execute(&self.db.pool)
        .await;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(run_id),
            Some(trace_id),
            json!({
                "event": "draft_response_captured",
                "artifact_id": artifact_id,
                "reason": reason,
                "attempt": attempt,
            }),
        )
        .await;
    }


    pub(super) async fn recent_monologue_entry_ids(
        &self,
        conversation_id: &str,
        window_secs: i64,
        limit: i64,
    ) -> Vec<String> {
        let entries = match self
            .db
            .list_inner_monologue_entries(conversation_id, limit)
            .await
        {
            Ok(entries) => entries,
            Err(_) => return Vec::new(),
        };
        let now = Utc::now();
        let mut ids = Vec::new();
        for entry in entries {
            if let Ok(parsed) = DateTime::parse_from_rfc3339(&entry.created_at) {
                let age_secs = now
                    .signed_duration_since(parsed.with_timezone(&Utc))
                    .num_seconds();
                if age_secs >= 0 && age_secs <= window_secs {
                    ids.push(entry.id);
                }
            }
        }
        ids
    }

    pub(super) async fn select_monologue_digest(
        &self,
        conversation_id: &str,
        state: &KernelState,
        ttl_secs: i64,
    ) -> MonologueDigest {
        let mut digest = MonologueDigest {
            text: None,
            source: "none".to_string(),
            age_secs: None,
            entry_id: None,
            stream: None,
            stale: false,
        };
        let entries = self
            .db
            .list_inner_monologue_entries(conversation_id, 6)
            .await
            .ok()
            .unwrap_or_default();
        let mut status_entry: Option<crate::models::InnerMonologueEntry> = None;
        for entry in entries.iter() {
            let is_status = entry
                .stream_type
                .as_deref()
                .map(|s| s.eq_ignore_ascii_case("STATUS"))
                .unwrap_or(false)
                || entry.mode == "status";
            if is_status {
                if status_entry.is_none() {
                    status_entry = Some(entry.clone());
                }
                continue;
            }
            let age_secs = prompt_age_seconds(&entry.created_at);
            digest.age_secs = age_secs;
            digest.entry_id = Some(entry.id.clone());
            digest.stream = entry.stream_type.clone();
            if let Some(age) = age_secs {
                if age >= 0 && age <= ttl_secs {
                    digest.text = Some(entry.thought.clone());
                    digest.source = "entry".to_string();
                    return digest;
                }
                digest.stale = true;
            } else {
                digest.stale = true;
            }
        }
        if let Some(entry) = status_entry {
            let age_secs = prompt_age_seconds(&entry.created_at);
            digest.age_secs = age_secs;
            digest.entry_id = Some(entry.id.clone());
            digest.stream = entry.stream_type.clone();
            if let Some(age) = age_secs {
                if age >= 0 && age <= ttl_secs {
                    digest.text = Some(entry.thought.clone());
                    digest.source = "status".to_string();
                    return digest;
                }
                digest.stale = true;
            } else {
                digest.stale = true;
            }
        }

        let fallback = state.self_state.last_internal_thought.trim().to_string();
        if !fallback.is_empty() {
            let age_secs = state
                .self_state
                .updated_at
                .as_deref()
                .and_then(prompt_age_seconds);
            digest.age_secs = age_secs.or(digest.age_secs);
            if let Some(age) = age_secs {
                if age >= 0 && age <= ttl_secs {
                    digest.text = Some(fallback);
                    digest.source = "self_state".to_string();
                    digest.stale = false;
                    return digest;
                }
                digest.stale = true;
            } else {
                digest.stale = true;
            }
        }

        if digest.stale {
            digest.source = "stale".to_string();
        }
        if digest.text.is_none() {
            digest.text = Some("Monologue idle".to_string());
            digest.source = "idle".to_string();
        }
        digest
    }

    pub(super) fn apply_decision_packet(&self, state: &mut KernelState, packet: &Value) {
        let intent = packet
            .get("intent")
            .and_then(|v| v.as_str())
            .unwrap_or("none")
            .trim()
            .to_lowercase();
        let bind = packet.get("bind").and_then(|v| v.as_bool()).unwrap_or(false);
        let bind_allowed = matches!(
            intent.as_str(),
            "stop" | "abort" | "compute_defaults" | "compute_partial"
        );
        if !bind || !bind_allowed {
            return;
        }

        match intent.as_str() {
            "stop" => {
                let mut scope = StopScope::default();
                scope.tools = true;
                scope.memory_write = true;
                scope.self_claims = true;
                scope.monologue_run = true;
                scope.monologue_emit = true;
                scope.background_jobs = true;
                apply_stop_state(
                    state,
                    StopReason {
                        category: StopReasonCategory::LatchBlock,
                        subcode: "decision_packet_stop".to_string(),
                        contract: None,
                    },
                    scope,
                );
                state.task_phase = TaskPhase::Terminated;
            }
            "abort" => {
                let mut scope = StopScope::default();
                scope.tools = true;
                scope.memory_write = true;
                scope.self_claims = true;
                scope.monologue_run = true;
                scope.monologue_emit = true;
                scope.background_jobs = true;
                apply_stop_state(
                    state,
                    StopReason {
                        category: StopReasonCategory::LatchBlock,
                        subcode: "decision_packet_abort".to_string(),
                        contract: None,
                    },
                    scope,
                );
                state.task_phase = TaskPhase::Aborting;
                state.resolution_mode = Some("abort".to_string());
            }
            "compute_defaults" => {
                if !matches!(state.task_phase, TaskPhase::Aborting | TaskPhase::Terminated) {
                    state.task_phase = TaskPhase::ResolvingWithDefaults;
                    state.resolution_mode = Some("defaults_used".to_string());
                }
            }
            "compute_partial" => {
                if !matches!(state.task_phase, TaskPhase::Aborting | TaskPhase::Terminated) {
                    state.task_phase = TaskPhase::ResolvingWithDefaults;
                    state.resolution_mode = Some("partial".to_string());
                }
            }
            _ => {}
        }

        if let Some(effects) = packet.get("effects").and_then(|v| v.as_object()) {
            if let Some(stop_latch) = effects.get("stop_latch").and_then(|v| v.as_bool()) {
                if stop_latch {
                    let mut scope = StopScope::default();
                    scope.tools = true;
                    scope.memory_write = true;
                    scope.self_claims = true;
                    scope.monologue_run = true;
                    scope.monologue_emit = true;
                    scope.background_jobs = true;
                    apply_stop_state(
                        state,
                        StopReason {
                            category: StopReasonCategory::LatchBlock,
                            subcode: "decision_packet_effect".to_string(),
                            contract: None,
                        },
                        scope,
                    );
                }
            }
            if let Some(task_phase) = effects.get("task_phase").and_then(|v| v.as_str()) {
                if !matches!(state.task_phase, TaskPhase::Aborting | TaskPhase::Terminated) {
                    if let Some(parsed) = parse_task_phase(task_phase) {
                        state.task_phase = parsed;
                    }
                }
            }
            if let Some(ask_budget) = effects.get("ask_budget").and_then(|v| v.as_i64()) {
                state.ask_budget_remaining = ask_budget.max(0) as i32;
            }
            if let Some(policy) = effects
                .get("missing_input_policy")
                .and_then(|v| v.as_str())
            {
                if !policy.trim().is_empty() {
                    state.missing_input_policy = Some(policy.trim().to_string());
                }
            }
            if let Some(mode) = effects.get("mode").and_then(|v| v.as_str()) {
                if mode.eq_ignore_ascii_case("play") {
                    state.mode = KernelMode::Play;
                } else if mode.eq_ignore_ascii_case("work") {
                    state.mode = KernelMode::Work;
                }
            }
        }
    }

    pub(super) async fn deliberate_self_dialogue(
        &self,
        conversation_id: &str,
        settings: &crate::models::Settings,
        state: &mut KernelState,
        outcomes: &[Outcome],
        decision_needed: bool,
        decision_turns: usize,
        stream: MonologueStream,
        max_tokens_override: Option<i64>,
    ) -> Result<MonologueOutput, String> {
        async fn repair_monologue_json(
            kernel: &Kernel,
            base_url: &str,
            settings: &crate::models::Settings,
            raw: &str,
            is_free_thought: bool,
        ) -> Option<Value> {
            let schema_hint = if is_free_thought {
                "Return ONLY a JSON object with keys: stance, message, descriptors, done."
            } else {
                "Return ONLY a JSON object with keys: stance, message, descriptors, candidates, decision_packet, done, topic_shift_reason."
            };
            let system_prompt = format!(
                "You are a JSON repair tool. {schema_hint} Use null or [] for missing values. Do not include any text outside JSON."
            );
            let raw_snippet: String = raw.chars().take(4000).collect();
            let user_prompt = format!("RAW:\n{raw_snippet}");

            let model = settings
                .summarization_model
                .as_deref()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .or_else(|| settings.active_model_id.clone())
                .unwrap_or_else(|| "default".to_string());

            let request = ChatCompletionRequest {
                model,
                messages: vec![
                    ChatMessage {
                        role: "system".to_string(),
                        content: system_prompt,
                    },
                    ChatMessage {
                        role: "user".to_string(),
                        content: user_prompt,
                    },
                ],
                stream: false,
                temperature: Some(0.0),
                top_p: None,
                max_tokens: Some(320),
                response_format: Some(json!({ "type": "json_object" })),
                tools: None,
                tool_choice: None,
                enable_thinking: None,
                prefill: None,
                skip_injection: Some(true),
                skip_memory: Some(true),
                skip_reminders: Some(true),
                memory_expand: None,
                allow_diagnostics: Some(false),
                json_strict: Some(true),
                skip_sanitization: None,
                run_id: None,
                request_label: Some("monologue_json_repair".to_string()),
            };

            let response = kernel
                .model_client
                .chat_with_meta(base_url, settings.api_key.as_deref(), &request)
                .await
                .ok()?;
            let (value_opt, _) = parse_json_object_with_repair(&response.content);
            value_opt
        }

        let is_free_thought = matches!(stream, MonologueStream::FreeThought);
        let decision_needed = decision_needed && !is_free_thought;
        let decision_turns = if decision_needed { decision_turns } else { 0 };
        let relaxation_level = state.monologue_relaxation_level;

        let inner_summary_raw = self
            .db
            .get_inner_summary(conversation_id)
            .await
            .ok()
            .flatten()
            .unwrap_or_else(|| "{}".to_string());
        let inner_summary = InnerSummary::from_json(&inner_summary_raw);

        let episodic = self
            .db
            .search_episodic_events(None, Some(conversation_id), None, None, None, None, 4)
            .await
            .unwrap_or_default();
        let episodic_lines = episodic
            .iter()
            .take(4)
            .filter_map(|event| {
                event
                    .payload
                    .get("summary_snippet")
                    .and_then(|v| v.as_str())
                    .map(|s| s.replace('\n', " "))
            })
            .collect::<Vec<_>>();
        let episodic_block = if episodic_lines.is_empty() {
            "None".to_string()
        } else {
            episodic_lines.join(" | ")
        };

        let semantic_hint = self
            .build_monologue_semantic_hint(conversation_id, &inner_summary, state)
            .await;
        let semantic_hint = summarize_snippet(&semantic_hint, 600);

        let outcome_lines = outcomes
            .iter()
            .take(4)
            .map(|o| format!("- {}: {}", o.action_type, o.observations))
            .collect::<Vec<_>>()
            .join("\n");
        let outcome_block = if outcome_lines.is_empty() {
            "None".to_string()
        } else {
            outcome_lines
        };

        let workspace_brief = format!(
            "goal_thread: {}\ncurrent_focus: {}\nopen_questions: {}\nactive_hypotheses: {}\nworking_set_topics: {}",
            state
                .workspace_goal_thread
                .clone()
                .unwrap_or_else(|| "None".to_string()),
            state
                .workspace_current_focus
                .clone()
                .unwrap_or_else(|| "None".to_string()),
            if state.workspace_open_questions.is_empty() {
                "None".to_string()
            } else {
                state.workspace_open_questions.join(" | ")
            },
            if state.workspace_active_hypotheses.is_empty() {
                "None".to_string()
            } else {
                state
                    .workspace_active_hypotheses
                    .iter()
                    .map(|h| summarize_snippet(&h.text, 120))
                    .collect::<Vec<_>>()
                    .join(" | ")
            },
            if state.workspace_working_set_topics.is_empty() {
                "None".to_string()
            } else {
                state.workspace_working_set_topics.join(" | ")
            }
        );
        let pending_questions_block = if state.pending_questions.is_empty() {
            String::new()
        } else {
            state
                .pending_questions
                .iter()
                .take(3)
                .map(|q| q.replace('\n', " "))
                .collect::<Vec<_>>()
                .join(" | ")
        };
        let workspace_questions_block = if state.workspace_open_questions.is_empty() {
            String::new()
        } else {
            state
                .workspace_open_questions
                .iter()
                .take(3)
                .map(|q| q.replace('\n', " "))
                .collect::<Vec<_>>()
                .join(" | ")
        };
        let pending_questions_block = if pending_questions_block.is_empty()
            && workspace_questions_block.is_empty()
        {
            "None".to_string()
        } else if pending_questions_block.is_empty() {
            workspace_questions_block
        } else if workspace_questions_block.is_empty() {
            pending_questions_block
        } else {
            format!("{} | {}", pending_questions_block, workspace_questions_block)
        };
        let last_user_input = summarize_snippet(
            &state
                .last_user_input
                .as_deref()
                .unwrap_or("None")
                .replace('\n', " "),
            320,
        );
        let last_assistant_output = summarize_snippet(
            &state
                .last_assistant_output_no_tags
                .as_deref()
                .unwrap_or("None")
                .replace('\n', " "),
            320,
        );
        let last_response_summary = summarize_snippet(
            &state
                .last_response_summary
                .as_deref()
                .unwrap_or("None")
                .replace('\n', " "),
            320,
        );
        let user_redirect_active = state.user_redirect_turns_remaining > 0;
        let redirect_focus_label = if user_redirect_active {
            state
                .redirect_focus
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .map(|s| summarize_snippet(s, 160))
                .or_else(|| {
                    if !last_user_input.trim().is_empty() {
                        Some(summarize_snippet(&last_user_input, 160))
                    } else {
                        None
                    }
                })
        } else {
            None
        };
        let mut recent_entries = self
            .db
            .list_inner_monologue_entries_by_stream(conversation_id, stream.as_str(), 2)
            .await
            .unwrap_or_default();
        if state.last_monologue_anchor_epoch != state.anchor_epoch {
            recent_entries.clear();
            state.last_monologue_anchor_epoch = state.anchor_epoch;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "monologue_anchor_epoch_reset",
                    "anchor_epoch": state.anchor_epoch,
                    "cleared": true,
                    "stream": stream.as_str(),
                }),
            )
            .await;
        }
        let recent_block = if recent_entries.is_empty() {
            "None".to_string()
        } else {
            recent_entries
                .iter()
                .rev()
                .map(|e| summarize_snippet(&e.thought, 120))
                .collect::<Vec<_>>()
                .join(" | ")
        };
        let recent_transcript = if recent_entries.is_empty() {
            "None".to_string()
        } else {
            recent_entries
                .iter()
                .rev()
                .take(2)
                .map(|e| {
                    let speaker = e
                        .speaker
                        .as_deref()
                        .unwrap_or("self")
                        .replace('_', " ")
                        .to_uppercase();
                    format!("{}: {}", speaker, summarize_snippet(&e.thought, 160))
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        let mut topic_seed = recent_entries
            .iter()
            .rev()
            .find(|e| !is_trivial_greeting(&e.thought))
            .map(|e| summarize_snippet(&e.thought, 160))
            .unwrap_or_else(|| "None".to_string());
        if !user_redirect_active {
            if let Some(focus) = state.workspace_current_focus.as_deref() {
                if !focus.trim().is_empty() {
                    topic_seed = focus.to_string();
                }
            }
        }
        if topic_seed.trim().eq_ignore_ascii_case("None") && !last_user_input.trim().is_empty() {
            topic_seed = summarize_snippet(&last_user_input, 160);
        }
        if let Some(label) = redirect_focus_label.as_ref() {
            topic_seed = label.clone();
        }

        let _tool_manifest = self.tools.definitions_for_settings(settings);

        let self_model = self.db.get_self_model().await.ok();
        let identity_block = if let Some(model) = self_model.as_ref() {
            let thread = model.identity_thread.clone().unwrap_or_else(|| "None".to_string());
            let note = model
                .identity_uncertainty_note
                .clone()
                .unwrap_or_else(|| "None".to_string());
            format!(
                "identity_thread: {}\nidentity_confidence: {:.2}\nidentity_uncertainty_note: {}",
                thread,
                model.identity_confidence,
                note
            )
        } else {
            "identity_thread: None\nidentity_confidence: 0.00\nidentity_uncertainty_note: None".to_string()
        };
        let _telemetry_block = state
            .controller_state
            .as_ref()
            .map(|s| {
                format!(
                    "confidence {:.2} | drift {:.2} | autonomy {:.2} | verification_needed {} | reanchor_needed {} | evidence_coverage {:.2} | telemetry_coverage {:.2}",
                    s.confidence,
                    s.drift_score,
                    s.autonomy_level,
                    s.verification_needed,
                    s.reanchor_needed,
                    s.evidence_coverage,
                    s.telemetry_coverage
                )
            })
            .unwrap_or_else(|| "None".to_string());
        let _telemetry_values_block = format_telemetry_snapshot(&state.telemetry_snapshot);

        let mut recent_outcomes_text = outcome_block.clone();
        if recent_outcomes_text.trim().is_empty() {
            recent_outcomes_text = "None".to_string();
        }
        let workspace_focus = if let Some(label) = redirect_focus_label.as_ref() {
            label.clone()
        } else {
            state
                .workspace_current_focus
                .clone()
                .unwrap_or_else(|| "None".to_string())
        };
        let goal_anchor = goal_stack_active_label(&state.workspace_goal_stack)
            .or_else(|| state.workspace_goal_thread.clone())
            .unwrap_or_else(|| "None".to_string());
        let relevance_anchors = RelevanceAnchors::new(
            last_user_input.clone(),
            workspace_focus,
            pending_questions_block.clone(),
            recent_outcomes_text.clone(),
            goal_anchor,
            last_response_summary.clone(),
        );
        let mut anchor_label = relevance_anchors.anchor_label();
        if let Some(label) = redirect_focus_label.as_ref() {
            anchor_label = label.clone();
        }
        let anchor_is_empty = anchor_label == "current topic";
        let anchor_has_verified = workspace_has_verified_anchor(state);
        let monologue_context_text = format!("{}\n{}", recent_transcript, recent_block);
        let boilerplate_detected = {
            let lower = monologue_context_text.to_lowercase();
            let patterns = [
                "we need to output a json object",
                "output a single json object",
                "do not use markdown",
                "use double quotes",
                "stance should alternate",
                "keys: stance",
                "candidates, decision_packet",
                "topic_shift_reason",
                "return stance",
            ];
            patterns.iter().any(|p| lower.contains(p))
        };
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            None,
            json!({
                "event": "monologue_boilerplate_check",
                "detected": boilerplate_detected,
                "anchor_label": anchor_label,
            }),
        )
        .await;
        if boilerplate_detected {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "kernel",
                None,
                None,
                json!({
                    "event": "monologue_boilerplate_detected",
                    "anchor_label": anchor_label,
                }),
            )
            .await;
        }
        let context_seed = if anchor_is_empty || !anchor_has_verified || boilerplate_detected {
            if !last_user_input.trim().is_empty() {
                summarize_snippet(&last_user_input, 160)
            } else if !last_assistant_output.trim().is_empty() {
                summarize_snippet(&last_assistant_output, 160)
            } else {
                topic_seed.clone()
            }
        } else {
            "None".to_string()
        };
        let force_relaxation = boilerplate_detected || state.monologue_emit_loop_breaker_triggered;
        if force_relaxation {
            let reasons = if boilerplate_detected && state.monologue_emit_loop_breaker_triggered {
                vec!["boilerplate", "loop_breaker"]
            } else if boilerplate_detected {
                vec!["boilerplate"]
            } else {
                vec!["loop_breaker"]
            };
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "monologue_relaxation_triggered",
                    "reasons": reasons,
                }),
            )
            .await;
        }
        let introspection_enabled = settings.enable_introspection.unwrap_or(true);
        let overlap_threshold = if introspection_enabled {
            (MONOLOGUE_TURN_OVERLAP_THRESHOLD * 0.25).max(0.0)
        } else {
            MONOLOGUE_TURN_OVERLAP_THRESHOLD
        };
        let relevance_warn_threshold = if introspection_enabled {
            (RELEVANCE_WARN_THRESHOLD * 0.5).max(0.0)
        } else {
            RELEVANCE_WARN_THRESHOLD
        };
        let relevance_reject_threshold = if introspection_enabled {
            (RELEVANCE_REJECT_THRESHOLD * 0.25).max(0.0)
        } else {
            RELEVANCE_REJECT_THRESHOLD
        };
        let loop_similarity_threshold = settings.loop_similarity_threshold.unwrap_or(0.85).clamp(0.5, 0.98);

        let user_name = settings.user_display_name.as_deref().unwrap_or("User");
        let brain_name = settings
            .assistant_display_name
            .as_deref()
            .unwrap_or("Assistant");
        let tool_names = self.tools.allowed_tool_names(settings);
        let tool_registry_snapshot = if tool_names.is_empty() {
            "None".to_string()
        } else {
            tool_names.join(", ")
        };
        let tool_snapshot = if tool_names.is_empty() {
            "Tools: none".to_string()
        } else {
            format!("Tools: {}", tool_names.join(", "))
        };
        let tool_snapshot = summarize_snippet(&tool_snapshot, 240);
        let meta_cog_summary = format!(
            "Meta-cog: events={} loop_breaks={} last_event={} loop_break_reason={} loop_reject_streak={}",
            state.meta_cog_event_count,
            state.meta_cog_loop_break_count,
            state
                .last_meta_cog_event
                .as_deref()
                .unwrap_or("None"),
            state
                .last_meta_cog_loop_break_reason
                .as_deref()
                .unwrap_or("None"),
            state.monologue_candidate_reject_streak
        );
        let meta_cog_summary = summarize_snippet(&meta_cog_summary, 160);
        let anchor_status_note = if anchor_is_empty || !anchor_has_verified {
            "Anchor status: weak"
        } else {
            "Anchor status: verified"
        };
        let system_overview_block = format!(
            "System overview (internal):\n- System name: Symbiote.\n- Brain name: {}.\n- Current goal thread: {}.\n- {}.\n- Subsystems: Kernel (orchestration), Prompt Builder (context), Model Client (LLM I/O), Memory (ICS), Scheduler (background cognition), UI (Tauri + React), Voice (optional).\n- Internal state: workspace, working memory, inner summary, monologue intents.\n- Architecture map:\n  - Organism loop: stress, arousal, fatigue, valence, social alignment.\n  - Qualia feedback loop: label aggregation, reward events, dominant tag.\n  - Subject controller: gate decisions (ALLOW / VERIFY / DENY / ALLOW_WITH_NOTICE).\n  - FeedbackBundle: pre-validated signals injected every turn.\n  - Tick loop: scheduler cadences for monologue, memory, consolidation, identity audit.\n  - Self-model controller: persona, goals, evidence coverage, identity thread.\n  - Workspace: ignition score, broadcast references, current focus.\n- {}\n- {}\n- Do not include telemetry, tool manifests, KV memory, timestamps, prompt hashes, or diagnostics in summaries or candidate payloads.",
            brain_name,
            state.workspace_goal_thread.as_deref().unwrap_or("None"),
            anchor_status_note,
            meta_cog_summary,
            tool_snapshot
        );
        let base_system_prompt = if is_free_thought {
            r#"You are thinking to yourself in a private inner monologue.
Output a single JSON object only with keys: stance, message, descriptors, done.
Do not include any text outside the JSON object. Do not use markdown or backticks. Use double quotes for all keys and string values.
Rules:
- This is not user-facing.
- Do not address the user directly in the message.
- Do not attribute beliefs, skepticism, or intent to the user unless it is explicitly present in the last user input.
- Never greet, offer help, or use salutations.
- Avoid boilerplate self-disclaimers (e.g., "I am an LLM", "as an AI", "I don't have feelings").
- No tools, no candidates, no decision packets.
- Keep it conversational, use multiple sentences when needed (up to ~6).
- Use stance "skeptic" or "synth". Alternate stance each turn.
- Skeptic probes risks, gaps, contradictions. Synth integrates and proposes next thoughts.
- Each turn must add a new point or be empty.
- Each turn must add novel information or ask a clarifier. Avoid repeating prior turns.
- If you pivot away from the Topic anchor, explain why inside the message.
- If nothing comes to mind, set done=true with empty message.
- If the Anchor status is weak, keep the message speculative or ask a single clarifying question.
- If you include descriptors, use the allowed list: [focus, uncertainty, urgency, confidence, curiosity, tension, clarity, calm].
- Descriptors reflect observable internal state. Do not overclaim subjective experience. Do not deny it either. Report operational state only.
- Do not include telemetry, tool manifests, KV memory, timestamps, prompt hashes, or diagnostics in your message.
- Do not include meta-format instructions about JSON or schemas (e.g., "We need to output a JSON object").
"#
        } else {
            r#"You are talking to yourself in a private internal dialogue.
Output a single JSON object only with keys: stance, message, descriptors, candidates, decision_packet, done, topic_shift_reason.
Do not include any text outside the JSON object. Do not use markdown or backticks. Use double quotes for all keys and string values.
Rules:
- This is not user-facing.
- Do not address the user directly in the message.
- Do not attribute beliefs, skepticism, or intent to the user unless it is explicitly present in the last user input.
- Never greet, offer help, or use salutations.
- Avoid boilerplate self-disclaimers (e.g., "I am an LLM", "as an AI", "I don't have feelings").
- Keep it conversational, use multiple sentences when needed (up to ~6).
- Use stance "skeptic" or "synth". Alternate stance each turn.
- Skeptic probes risks, gaps, contradictions. Synth integrates and proposes next actions.
- Each turn should respond to the prior stance's last message.
- Each turn must add a new point or be empty.
- Each turn must add novel information or ask a clarifier. Avoid repeating prior turns.
- Do not reveal step-by-step reasoning.
- If you pivot away from the Topic anchor, provide topic_shift_reason tied to evidence or outcomes.
- If you propose candidates, include a brief message.
- Only be silent (empty message) when candidates is empty.
- If nothing comes to mind, set done=true with empty message and no candidates.
- If awaiting user input, do not keep asking; end with done=true unless you have a new, relevant idea.
- Reference the Topic anchor or recent outcomes; otherwise be empty.
- If the Anchor status is weak, keep the message speculative or ask a single clarifying question; avoid update_workspace or record_self_claim candidates.
- Tool candidates must use a tool name from the Tools list provided in the context.
- For research tools (e.g., web_lookup), include uncertainty and decision_impact strings in the tool_call payload.
- Self-claim candidates must include evidence_event_ids or belief_ids; for self-awareness queries you may emit a provisional self-claim with evidence missing if you mark it provisional and explicitly speculative.
- When you change goals, focus, hypotheses, or open questions, emit an update_workspace candidate.
- If update_workspace is based on memory or tool results, include evidence_event_ids or belief_ids. If unsure, set speculative=true.
- Do not propose update_workspace unless there is a verified anchor, evidence IDs, or internal evidence. If you use last user input as a provisional anchor, set speculative=true.
- Never set current_focus to "None" or an empty placeholder.
- If you have a concrete suggestion for the user, include an emit_message, ask_user_question, or flag_for_human candidate with the actual text (no meta-permission questions).
- If you emit_message or flag_for_human with factual claims, include evidence_event_ids or belief_ids. If you cannot, set speculative=true.
- If you include descriptors, use the allowed list: [focus, uncertainty, urgency, confidence, curiosity, tension, clarity, calm].
- Descriptors reflect observable internal state. Do not overclaim subjective experience. Do not deny it either. Report operational state only.
- Do not include telemetry, tool manifests, KV memory, timestamps, prompt hashes, or diagnostics in your message or candidate payloads.
- Do not include meta-format instructions about JSON or schemas (e.g., "We need to output a JSON object").

Candidate schema:
{ kind, payload, rationale?, expected_outcome?, cost?, urgency?, priority_rank?, target_scope?, evidence_event_ids?, belief_ids? }
Valid kinds: update_inner_summary, emit_message, ask_user_question, flag_for_human, tool_call, spawn_thread, write_episodic, promote_semantic, update_goal_thread, update_workspace, anchor_shift, record_self_claim, change_mode, terminate, no_op.
Payload notes:
- tool_call: { tool_name, arguments, action_id?, uncertainty?, decision_impact? }
- update_workspace: include only fields you change; include evidence_event_ids/belief_ids or speculative when uncertain.
- record_self_claim: include claim_text and evidence_event_ids/belief_ids, or mark provisional/speculative.

Decision packet (optional, internal only):
{ intent: "stop"|"abort"|"compute_defaults"|"compute_partial"|"ask"|"none", bind: boolean, required_slots?: string[], effects?: { stop_latch?: boolean, task_phase?: string, ask_budget?: number, missing_input_policy?: string, mode?: string } }
"#
        };
        let mut base_system_prompt = base_system_prompt.to_string();
        if is_free_thought {
            base_system_prompt = base_system_prompt.replace(
                "- If nothing comes to mind, set done=true with empty message.",
                &format!(
                    "- If nothing comes to mind, set done=true with empty message after at least {} turns.",
                    FTS_MIN_TURNS
                ),
            );
        }
        base_system_prompt = base_system_prompt.replace(
            "- Do not address the user directly in the message.",
            "- Do not address the user directly in the message.\n- Do not address the user by name.\n- You are the same system that produces the user-visible response. System-provided context is not user input.",
        );
        if !is_free_thought {
            base_system_prompt = format!(
                "{}\n{}",
                base_system_prompt,
                crate::core::kernel::monologue::monologue_update_schema()
            );
        }

        let reanchor_hint = if user_redirect_active {
            true
        } else {
            state
                .controller_gate
                .as_ref()
                .map(|g| g.reanchor)
                .unwrap_or(false)
        };
        let seed_note = if context_seed != "None" {
            " Use the Context seed to ground the message and avoid format-talk."
        } else {
            ""
        };
        let mode_note = if is_free_thought {
            format!(
                "Free-thought mode: reflect on the current topic or recent outcomes. Continue for at least {} turns before ending with done=true and an empty message.{}{}",
                FTS_MIN_TURNS,
                if reanchor_hint { " Re-anchor to the current topic and avoid drift." } else { "" },
                seed_note
            )
        } else if decision_needed {
            format!(
                "Decision mode: exactly {} turns. Provide a message every turn.{}{}",
                decision_turns.max(1),
                if reanchor_hint { " Re-anchor to the current topic and avoid drift." } else { "" },
                seed_note
            )
        } else {
            format!(
                "Free mode: continue while something is worth pondering. Stay on the current topic from the recent self-dialogue transcript unless new user input exists. You may propose emit_message or ask_user_question candidates when it would help. You may end with done=true and an empty message if nothing remains.{}{}",
                if reanchor_hint { " Re-anchor to the current topic and avoid drift." } else { "" },
                seed_note
            )
        };

        let last_response_summary = last_response_summary.as_str();
        let inner_summary_json = summarize_snippet(&inner_summary.to_json(), 1200);
        let context_block = if is_free_thought {
            format!(
                "Context snapshot (FTS):\n{}\nTool registry: {}\nLast user input: {}\nLast assistant response: {}\nLast response summary: {}\nContext seed (use if anchor weak or looped): {}\nCurrent focus: {}\nRecent outcomes: {}\n\nRecent free-thought transcript:\n{}\n\nRecent deliberation transcript:\n{}",
                system_overview_block,
                tool_registry_snapshot,
                last_user_input,
                last_assistant_output,
                last_response_summary,
                context_seed,
                topic_seed,
                recent_outcomes_text,
                recent_transcript,
                recent_block
            )
        } else {
            format!(
                "Context snapshot (DS):\n{}\nTool registry: {}\nMode: {:?}\nDecision needed: {}\nPending questions: {}\nLast user input: {}\nLast assistant response: {}\nLast response summary: {}\nContext seed (use if anchor weak or looped): {}\n\nTopic anchor: {}\n\nWorkspace (brief):\n{}\n\nIdentity Thread:\n{}\n\nRecent self-dialogue transcript (continue this thread unless new user input exists):\n{}\n\nCurrent inner_summary JSON:\n{}\n\nSemantic hints:\n{}\n\nRecent episodic hints:\n{}\n\nRecent outcomes:\n{}\n\nRecent self-dialogue (previous ticks): {}",
                system_overview_block,
                tool_registry_snapshot,
                state.mode,
                decision_needed,
                pending_questions_block,
                last_user_input,
                last_assistant_output,
                last_response_summary,
                context_seed,
                topic_seed,
                workspace_brief,
                identity_block,
                recent_transcript,
                inner_summary_json,
                semantic_hint,
                episodic_block,
                outcome_block,
                recent_block
            )
        };

        let mut history_a = vec![ChatMessage {
            role: "user".to_string(),
            content: context_block.clone(),
        }];
        let mut history_b = vec![ChatMessage {
            role: "user".to_string(),
            content: context_block,
        }];

        let mut turns: Vec<MonologueTurn> = Vec::new();
        let mut dialogue_messages: Vec<String> = Vec::new();
        let mut created_at = 0i64;
        let dialogue_id = Uuid::new_v4().to_string();
        let max_turns = if is_free_thought {
            FTS_MAX_TURNS
        } else if decision_needed {
            decision_turns.max(1)
        } else {
            4
        };
        let mut turn_index = 0usize;
        let mut speaker_a = true;
        let started = Instant::now();
        let mut last_message: Option<String> = None;
        let mut last_message_interrogative = false;
        let anchor_vocab = build_anchor_vocab(state, &tool_names);
        let anchor_vocab_set: HashSet<String> = anchor_vocab.iter().cloned().collect();
        let mut anchor_absent_streak: i32 = 0;
        let mut novelty_absent_streak: i32 = 0;
        let mut seen_anchor_tokens: HashSet<String> = HashSet::new();
        let mut seen_evidence_ids: HashSet<String> = HashSet::new();
        let mut seen_actions: HashSet<String> = HashSet::new();
        let mut fabricated_tool_count: i32 = 0;
        let mut tool_fabrication_repeat = false;
        let mut suppression_reasons: HashMap<String, usize> = HashMap::new();
        fn record_suppression(suppression_reasons: &mut HashMap<String, usize>, reason: &str) {
            *suppression_reasons.entry(reason.to_string()).or_insert(0) += 1;
        }
        let last_other_self_seed = recent_entries
            .iter()
            .find(|e| !e.thought.trim().is_empty())
            .map(|e| summarize_snippet(&e.thought, 200))
            .unwrap_or_else(|| "None".to_string());

        loop {
            let _ = self.db.touch_active_run_heartbeat(conversation_id).await;
            if decision_needed && turn_index >= max_turns {
                break;
            }
            if started.elapsed() > Duration::from_secs(60) {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "monologue_runaway",
                        "turns": turn_index,
                        "decision_needed": decision_needed,
                    }),
                )
                .await;
                break;
            }

            let stance_label = if speaker_a { "skeptic" } else { "synth" };
            let stance_hint = if speaker_a {
                "Stance: skeptic (probe risks, gaps, contradictions)."
            } else {
                "Stance: synth (integrate signals, propose next actions)."
            };
            let other_self_message = last_message
                .as_deref()
                .unwrap_or(last_other_self_seed.as_str());
            let other_self_note = format!("Last other-stance message: {}", other_self_message);
            let turn_directive = if decision_needed {
                match decision_turns {
                    2 => match turn_index {
                        0 => "Propose the leading option or plan.".to_string(),
                        _ => "Decide and surface any candidates.".to_string(),
                    },
                    3 => match turn_index {
                        0 => "Propose at least two options or angles.".to_string(),
                        1 => "Challenge one option and note a counterpoint.".to_string(),
                        _ => "Decide and surface any candidates.".to_string(),
                    },
                    _ => match turn_index {
                        0 => "Propose at least two distinct options or angles.".to_string(),
                        1 => "Challenge one option and note a counterpoint.".to_string(),
                        2 => "Refine the leading option or combine ideas.".to_string(),
                        3 => "Check risks, missing info, or alternatives.".to_string(),
                        _ => "Decide and surface any candidates.".to_string(),
                    },
                }
            } else if is_free_thought {
                format!(
                    "Continue naturally; aim for at least {} turns before ending with done=true and an empty message.",
                    FTS_MIN_TURNS
                )
            } else {
                "Continue naturally; if nothing remains, end with done=true and an empty message.".to_string()
            };

            let system_prompt = format!(
                "{}\n\n{}\n{}\n{}\nTurn {}: {}\nReturn stance=\"{}\".",
                base_system_prompt,
                mode_note,
                stance_hint,
                other_self_note,
                turn_index + 1,
                turn_directive,
                stance_label
            );

            let history = if speaker_a { &history_a } else { &history_b };
            let mut messages = Vec::with_capacity(history.len() + 1);
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: system_prompt,
            });
            messages.extend(history.iter().cloned());

            let now = Utc::now();
            let mut json_cooldown_active = false;
            if let Some(ts) = state.monologue_json_disabled_until.as_deref() {
                if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(ts) {
                    let until = parsed.with_timezone(&Utc);
                    if until > now {
                        json_cooldown_active = true;
                    } else {
                        state.monologue_json_disabled_until = None;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "monologue_json_reenabled",
                                "reason": "cooldown_elapsed",
                                "cooldown_until": until.to_rfc3339(),
                            }),
                        )
                        .await;
                    }
                } else {
                    state.monologue_json_disabled_until = None;
                }
            }

            let recent_parse_failed = self
                .count_event_since("monologue_parse_failed", MONOLOGUE_JSON_FAILURE_WINDOW_MINS)
                .await;
            let recent_disable_count = self
                .count_event_since("monologue_json_disabled", MONOLOGUE_JSON_FAILURE_WINDOW_MINS)
                .await;
            let cooldown_secs = if recent_disable_count > 0 {
                (MONOLOGUE_JSON_COOLDOWN_SECS / 2).max(30)
            } else {
                MONOLOGUE_JSON_COOLDOWN_SECS
            };
            let mut skip_repair_due_to_failures =
                recent_parse_failed >= MONOLOGUE_JSON_FAILURE_THRESHOLD;
            if recent_parse_failed >= MONOLOGUE_JSON_FAILURE_THRESHOLD && !json_cooldown_active {
                let cooldown_until = now + chrono::Duration::seconds(cooldown_secs);
                state.monologue_json_disabled_until = Some(cooldown_until.to_rfc3339());
                state.monologue_quiet_until = Some(cooldown_until.to_rfc3339());
                json_cooldown_active = true;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "monologue_json_disabled",
                        "reason": "parse_failure_threshold",
                        "recent_parse_failed": recent_parse_failed,
                        "window_mins": MONOLOGUE_JSON_FAILURE_WINDOW_MINS,
                        "cooldown_secs": cooldown_secs,
                        "recent_disable_count": recent_disable_count,
                        "cooldown_until": state.monologue_json_disabled_until,
                    }),
                )
                .await;
            }
            if json_cooldown_active {
                skip_repair_due_to_failures = true;
            }

            let response_format = if state.monologue_json_supported.unwrap_or(true)
                && !force_relaxation
                && !json_cooldown_active
            {
                monologue_response_format(settings)
            } else {
                None
            };
            let request = ChatCompletionRequest {
                model: settings
                    .summarization_model
                    .as_deref()
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .or_else(|| settings.active_model_id.clone())
                    .unwrap_or_else(|| "default".to_string()),
                messages,
                stream: false,
                temperature: None,
                top_p: None,
                max_tokens: max_tokens_override.map(|v| v as i32),
                response_format: response_format.clone(),
                tools: None,
                tool_choice: None,
                enable_thinking: None,
                prefill: None,
                skip_injection: Some(true),
                skip_memory: Some(true),
                skip_reminders: Some(true),
                memory_expand: None,
                allow_diagnostics: Some(false),
                json_strict: Some(response_format.is_some()),
                skip_sanitization: None,
                run_id: None,
                request_label: Some("monologue_generation".to_string()),
            };

            let monologue_base_url = if state.monologue_force_primary {
                settings.api_base_url.as_str()
            } else {
                settings
                    .summarization_api_url
                    .as_deref()
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .unwrap_or(&settings.api_base_url)
            };

            let response = self
                .model_client
                .chat_with_meta(monologue_base_url, settings.api_key.as_deref(), &request)
                .await?;

            let (mut value_opt, mut repaired) = parse_json_object_with_repair(&response.content);
            let mut repaired_via_model = false;
            let mut fallback_used = false;
            if value_opt.is_none() {
                if state.monologue_json_supported.unwrap_or(true) && !skip_repair_due_to_failures {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "monologue_parse_repair_attempt",
                            "turn": turn_index + 1,
                            "raw_len": response.content.len(),
                        }),
                    )
                    .await;
                    if let Some(repaired_value) = repair_monologue_json(
                        self,
                        monologue_base_url,
                        settings,
                        &response.content,
                        is_free_thought,
                    )
                    .await
                    {
                        value_opt = Some(repaired_value);
                        repaired = true;
                        repaired_via_model = true;
                    }
                } else {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "monologue_parse_repair_skipped",
                            "turn": turn_index + 1,
                            "reason": if skip_repair_due_to_failures { "recent_parse_failures" } else { "json_unsupported" },
                        }),
                    )
                    .await;
                }
            }
            if value_opt.is_none() {
                if let Some(fallback_value) = parse_monologue_fallback(&response.content, stance_label) {
                    value_opt = Some(fallback_value);
                    repaired = true;
                    fallback_used = true;
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "monologue_parse_fallback",
                            "turn": turn_index + 1,
                            "raw_len": response.content.len(),
                            "snippet": summarize_snippet(&response.content, 200),
                        }),
                    )
                    .await;
                }
            }
            if value_opt.is_none() {
                let (allow_log, suppressed) = rate_limit_event(
                    &MONOLOGUE_PARSE_FAIL_RATE,
                    "monologue_parse_failed",
                    Duration::from_secs(MONOLOGUE_PARSE_FAIL_WINDOW_SECS),
                );
                if allow_log {
                    if suppressed > 0 {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "monologue_parse_failed_suppressed",
                                "count": suppressed,
                                "window_secs": MONOLOGUE_PARSE_FAIL_WINDOW_SECS,
                            }),
                        )
                        .await;
                    }
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "monologue_parse_failed",
                            "turn": turn_index + 1,
                            "raw_len": response.content.len(),
                            "snippet": summarize_snippet(&response.content, 240),
                            "fallback_used": false,
                        }),
                    )
                    .await;
                }
                if !json_cooldown_active
                    && recent_parse_failed.saturating_add(1) >= MONOLOGUE_JSON_FAILURE_THRESHOLD
                {
                    let cooldown_until = now + chrono::Duration::seconds(cooldown_secs);
                    state.monologue_json_disabled_until = Some(cooldown_until.to_rfc3339());
                    state.monologue_quiet_until = Some(cooldown_until.to_rfc3339());
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "monologue_json_disabled",
                            "reason": "parse_failure_threshold",
                            "recent_parse_failed": recent_parse_failed.saturating_add(1),
                            "window_mins": MONOLOGUE_JSON_FAILURE_WINDOW_MINS,
                            "cooldown_secs": cooldown_secs,
                            "recent_disable_count": recent_disable_count,
                            "cooldown_until": state.monologue_json_disabled_until,
                        }),
                    )
                    .await;
                }
                record_suppression(&mut suppression_reasons, "parse_failed");
                turn_index += 1;
                speaker_a = !speaker_a;
                tokio::task::yield_now().await;
                continue;
            }
            let Some(value) = value_opt else {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "monologue_parse_missing",
                        "turn": turn_index + 1,
                        "raw_len": response.content.len(),
                    }),
                )
                .await;
                continue;
            };
            if repaired {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "monologue_parse_repaired",
                        "turn": turn_index + 1,
                        "raw_len": response.content.len(),
                        "snippet": summarize_snippet(&response.content, 200),
                        "method": if repaired_via_model {
                            "model_repair"
                        } else if fallback_used {
                            "fallback_parse"
                        } else {
                            "local_repair_or_fallback"
                        },
                    }),
                )
                .await;
            }

            let stance_raw = value
                .get("stance")
                .and_then(|v| v.as_str())
                .unwrap_or(stance_label)
                .trim()
                .to_lowercase();
            let stance_selected = match stance_raw.as_str() {
                "skeptic" => "skeptic",
                "synth" => "synth",
                _ => stance_label,
            };
            let mut message = value
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let mut done = value
                .get("done")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if stance_selected != stance_label {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "monologue_stance_mismatch",
                        "expected": stance_label,
                        "reported": stance_raw,
                        "turn": turn_index + 1,
                    }),
                )
                .await;
                if relaxation_level <= 0 {
                    record_suppression(&mut suppression_reasons, "stance_mismatch");
                    message.clear();
                    done = true;
                }
            }
            if let Some(last_entry) = recent_entries.first() {
                let last_clean = last_entry.thought.trim().to_lowercase();
                if !last_clean.is_empty() && message.trim().eq_ignore_ascii_case(&last_clean) {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "monologue_delta_empty",
                            "turn": turn_index + 1,
                        }),
                    )
                    .await;
                    record_suppression(&mut suppression_reasons, "delta_empty");
                    message.clear();
                    done = true;
                }
            }
            if is_free_thought && done && (turn_index + 1) < FTS_MIN_TURNS {
                done = false;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "monologue_fts_min_turns_enforced",
                        "turn": turn_index + 1,
                        "min_turns": FTS_MIN_TURNS,
                    }),
                )
                .await;
            }
            let topic_shift_reason = value
                .get("topic_shift_reason")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let has_topic_shift_reason = !topic_shift_reason.is_empty();
            let mut rejected_descriptors: Vec<String> = Vec::new();
            let descriptors = value
                .get("descriptors")
                .and_then(|v| v.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str())
                        .map(|s| s.trim().to_lowercase())
                        .filter(|s| {
                            if s.is_empty() {
                                return false;
                            }
                            if MONOLOGUE_DESCRIPTOR_ALLOWLIST.contains(s.as_str()) {
                                true
                            } else {
                                rejected_descriptors.push(s.clone());
                                false
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .filter(|items| !items.is_empty());
            let mut descriptors = descriptors.unwrap_or_default();
            if !rejected_descriptors.is_empty() {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "monologue_descriptor_rejected",
                        "turn": turn_index + 1,
                        "descriptors": rejected_descriptors,
                    }),
                )
                .await;
            }

            if let Some(packet) = value.get("decision_packet") {
                if decision_needed {
                    self.apply_decision_packet(state, packet);
                } else {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "decision_packet_ignored",
                            "reason": "kernel_not_in_decision_mode",
                            "turn": turn_index + 1,
                        }),
                    )
                    .await;
                }
            }

            let message_interrogative = is_interrogative_message(&message);
            let message_anchor_hits = count_anchor_hits(&message, &anchor_vocab);
            let mut novelty_present = false;
            let mut new_anchor_tokens: Vec<String> = Vec::new();
            let mut new_evidence_ids: Vec<String> = Vec::new();
            let mut new_actions: Vec<String> = Vec::new();
            if !message.trim().is_empty() {
                for token in extract_anchor_tokens(&message, &anchor_vocab_set) {
                    if seen_anchor_tokens.insert(token.clone()) {
                        new_anchor_tokens.push(token);
                    }
                }
                for id in extract_evidence_ids(&message) {
                    if seen_evidence_ids.insert(id.clone()) {
                        new_evidence_ids.push(id);
                    }
                }
                for action in extract_action_phrases(&message) {
                    if seen_actions.insert(action.clone()) {
                        new_actions.push(action);
                    }
                }
                novelty_present =
                    !(new_anchor_tokens.is_empty() && new_evidence_ids.is_empty() && new_actions.is_empty());
                if novelty_present {
                    novelty_absent_streak = 0;
                } else {
                    novelty_absent_streak += 1;
                }
            }
            let mut meta_cog_reanchor_attempt = false;
            let mut meta_cog_reanchor_similarity: Option<f32> = None;
            if !message.trim().is_empty() && message_interrogative && !last_message_interrogative && !has_topic_shift_reason {
                if let Some(prev) = last_message.as_deref() {
                    meta_cog_reanchor_similarity = Some(token_similarity(&message, prev));
                }
                meta_cog_reanchor_attempt = true;
                state.meta_cog_reanchor_attempts = state.meta_cog_reanchor_attempts.saturating_add(1);
                self.log_meta_cog_event(
                    state,
                    None,
                    None,
                    json!({
                        "event": "meta_cog_reanchor_attempt",
                        "reason": "stance_shift_to_question",
                        "prior_turn_id": if turn_index > 0 { turn_index as i64 } else { 0 },
                        "current_turn_id": (turn_index + 1) as i64,
                        "topic_anchor": anchor_label.clone(),
                        "similarity": meta_cog_reanchor_similarity.unwrap_or(0.0),
                        "interrogative": true,
                    }),
                )
                .await;
            }

            let mut turn_candidates: Vec<Candidate> = Vec::new();
            let mut blocked_candidates: Vec<BlockedCandidate> = Vec::new();
            let mut cached_user_evidence: Option<Vec<i64>> = None;
            let mut record_blocked = |candidate: &Candidate, reason: &str| {
                blocked_candidates.push(BlockedCandidate {
                    candidate: candidate.clone(),
                    reason: reason.to_string(),
                });
            };
            let mut reanchor_needed = false;
            if !is_free_thought {
                if let Some(items) = value.get("candidates").and_then(|v| v.as_array()) {
                    for item in items {
                        let Some(mut candidate) = self.candidate_from_value(item, "monologue", &mut created_at) else {
                            continue;
                        };
                        if matches!(
                            candidate.kind,
                            CandidateKind::EmitMessage | CandidateKind::AskUserQuestion | CandidateKind::FlagForHuman
                        ) {
                            let candidate_text = candidate_relevance_text(&candidate);
                            if response_has_user_attribution(&candidate_text, user_name)
                                && !user_attribution_grounded_in_last_input(&candidate_text, &last_user_input)
                                && !candidate_overlaps_last_user_input(&candidate, &last_user_input)
                            {
                                let rewritten = rewrite_user_attribution_text(&candidate_text, user_name);
                                set_candidate_text_payload(&mut candidate, &rewritten);
                                let _ = system_log::log_event(
                                    &self.db.pool,
                                    Some(&self.app_handle),
                                    "info",
                                    "kernel",
                                    None,
                                    None,
                                    json!({
                                        "event": "monologue_candidate_user_attribution_rewritten",
                                        "turn": turn_index + 1,
                                        "candidate_kind": format!("{:?}", candidate.kind),
                                        "snippet": summarize_snippet(&candidate_text, 160),
                                    }),
                                )
                                .await;
                            }
                        }

                    if matches!(candidate.kind, CandidateKind::ToolCall) {
                        let tool_name = candidate
                            .payload
                            .get("tool_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if is_self_inspection_tool(tool_name) {
                            self.log_meta_cog_event(
                                state,
                                None,
                                None,
                                json!({
                                    "event": "meta_cog_event",
                                    "reason": "self_inspection_tool_call",
                                    "tool": tool_name,
                                    "turn": turn_index + 1,
                                }),
                            )
                            .await;
                        }
                        let reason = if !self.is_known_tool_name(tool_name) {
                            Some("UNKNOWN_TOOL")
                        } else if !self.is_allowed_tool_name(tool_name, settings) {
                            Some("TOOL_DISABLED")
                        } else {
                            None
                        };
                        if let Some(reason) = reason {
                            if reason == "UNKNOWN_TOOL" {
                                fabricated_tool_count += 1;
                                let _ = system_log::log_event(
                                    &self.db.pool,
                                    Some(&self.app_handle),
                                    "warn",
                                    "kernel",
                                    None,
                                    None,
                                    json!({
                                        "event": "monologue_tool_fabricated",
                                        "tool": tool_name,
                                        "turn": turn_index + 1,
                                    }),
                                )
                                .await;
                                if fabricated_tool_count >= 2 {
                                    tool_fabrication_repeat = true;
                                }
                            }
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "warn",
                                "kernel",
                                None,
                                None,
                                json!({
                                    "event": "tool_candidate_rejected",
                                    "tool": tool_name,
                                    "reason": reason,
                                    "turn": turn_index + 1,
                                }),
                            )
                            .await;
                            reanchor_needed = true;
                            record_blocked(&candidate, reason);
                            continue;
                        }
                    }

                    let candidate_text = candidate_relevance_text(&candidate);
                    let is_meta_cog = is_meta_cog_candidate(&candidate);
                    let internal_evidence = matches!(candidate_evidence_class(&candidate), Some("internal"));
                    let candidate_score = relevance_score(&candidate_text, &relevance_anchors);
                    let candidate_shift_reason = candidate
                        .payload
                        .get("topic_shift_reason")
                        .and_then(|v| v.as_str())
                        .or_else(|| candidate.rationale.as_deref())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let has_candidate_shift_reason = !candidate_shift_reason.is_empty();
                    let derived_from_last_user = candidate_mentions_last_user_input(&candidate, &last_user_input);
                    if matches!(candidate.kind, CandidateKind::UpdateWorkspace | CandidateKind::RecordSelfClaim) {
                        let evidence_ids = extract_id_list(&candidate.payload, "evidence_event_ids");
                        let belief_ids = extract_id_list(&candidate.payload, "belief_ids");
                        let evidence_single = candidate
                            .payload
                            .get("evidence_event_id")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0);
                        if evidence_ids.is_empty()
                            && belief_ids.is_empty()
                            && evidence_single <= 0
                            && derived_from_last_user
                        {
                            if cached_user_evidence.is_none() {
                                cached_user_evidence = Some(
                                    self.db
                                        .get_recent_user_evidence_ids(conversation_id, 1)
                                        .await,
                                );
                            }
                            if let Some(ids) = cached_user_evidence.as_ref() {
                                if !ids.is_empty() {
                                    set_id_list(&mut candidate.payload, "evidence_event_ids", ids);
                                }
                            }
                        }
                    }
                    if candidate_score < RELEVANCE_REJECT_THRESHOLD && !(is_meta_cog && internal_evidence) {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "monologue_drift",
                                "kind": "candidate",
                                "candidate_kind": format!("{:?}", candidate.kind),
                                "score": candidate_score,
                                "threshold": RELEVANCE_REJECT_THRESHOLD,
                                "anchor": anchor_label.clone(),
                                "turn": turn_index + 1,
                                "topic_shift_reason": candidate_shift_reason,
                            }),
                        )
                        .await;
                        reanchor_needed = true;
                        if !has_candidate_shift_reason {
                            if matches!(
                                candidate.kind,
                                CandidateKind::EmitMessage | CandidateKind::AskUserQuestion | CandidateKind::FlagForHuman
                            ) {
                                if let Some(obj) = candidate.payload.as_object_mut() {
                                    obj.insert("speculative".to_string(), Value::Bool(true));
                                    obj.insert(
                                        "speculative_reason".to_string(),
                                        Value::String("low_relevance_no_reason".to_string()),
                                    );
                                }
                            } else {
                                if let Some(obj) = candidate.payload.as_object_mut() {
                                    obj.insert("speculative".to_string(), Value::Bool(true));
                                    obj.insert(
                                        "speculative_reason".to_string(),
                                        Value::String("low_relevance_no_reason".to_string()),
                                    );
                                }
                                let _ = system_log::log_event(
                                    &self.db.pool,
                                    Some(&self.app_handle),
                                    "info",
                                    "kernel",
                                    None,
                                    None,
                                    json!({
                                        "event": "monologue_candidate_speculative",
                                        "reason": "low_relevance_no_reason",
                                        "candidate_kind": format!("{:?}", candidate.kind),
                                        "score": candidate_score,
                                        "anchor": anchor_label.clone(),
                                        "turn": turn_index + 1,
                                    }),
                                )
                                .await;
                            }
                        }
                        if matches!(
                            candidate.kind,
                            CandidateKind::EmitMessage | CandidateKind::AskUserQuestion | CandidateKind::FlagForHuman
                        ) {
                            candidate.priority_rank = candidate.priority_rank.saturating_add(20);
                        }
                    }
                    if candidate_score < RELEVANCE_WARN_THRESHOLD && !(is_meta_cog && internal_evidence) {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "monologue_drift",
                                "kind": "candidate",
                                "candidate_kind": format!("{:?}", candidate.kind),
                                "score": candidate_score,
                                "threshold": RELEVANCE_WARN_THRESHOLD,
                                "anchor": anchor_label.clone(),
                                "turn": turn_index + 1,
                                "topic_shift_reason": candidate_shift_reason,
                            }),
                        )
                        .await;
                        reanchor_needed = true;
                        candidate.priority_rank = candidate.priority_rank.saturating_add(10);
                    }

                    if matches!(candidate.kind, CandidateKind::UpdateWorkspace) {
                        if !update_workspace_payload_has_substantive_fields(&candidate.payload) {
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "kernel",
                                None,
                                None,
                                json!({
                                    "event": "monologue_candidate_blocked",
                                    "reason": "empty_payload",
                                    "candidate_kind": "UpdateWorkspace",
                                    "anchor": anchor_label.clone(),
                                    "turn": turn_index + 1,
                                }),
                            )
                            .await;
                            record_blocked(&candidate, "empty_payload");
                            continue;
                        }
                        let has_evidence = update_workspace_payload_has_evidence(&candidate.payload);
                        if !has_evidence && !anchor_has_verified && !internal_evidence {
                            if derived_from_last_user {
                                if let Some(obj) = candidate.payload.as_object_mut() {
                                    obj.insert("speculative".to_string(), Value::Bool(true));
                                }
                            } else {
                                let _ = system_log::log_event(
                                    &self.db.pool,
                                    Some(&self.app_handle),
                                    "info",
                                    "kernel",
                                    None,
                                    None,
                                    json!({
                                        "event": "monologue_candidate_blocked",
                                        "reason": "missing_evidence",
                                        "candidate_kind": "UpdateWorkspace",
                                        "anchor": anchor_label.clone(),
                                        "turn": turn_index + 1,
                                    }),
                                )
                                .await;
                                record_blocked(&candidate, "missing_evidence");
                                continue;
                            }
                        }
                        if anchor_is_empty && !anchor_has_verified {
                            if !derived_from_last_user && !internal_evidence {
                                let _ = system_log::log_event(
                                    &self.db.pool,
                                    Some(&self.app_handle),
                                    "info",
                                    "kernel",
                                    None,
                                    None,
                                    json!({
                                        "event": "monologue_candidate_blocked",
                                        "reason": "anchor_missing",
                                        "candidate_kind": "UpdateWorkspace",
                                        "anchor": anchor_label.clone(),
                                        "turn": turn_index + 1,
                                    }),
                                )
                                .await;
                                record_blocked(&candidate, "anchor_missing");
                                continue;
                            }
                            if let Some(obj) = candidate.payload.as_object_mut() {
                                obj.insert("speculative".to_string(), Value::Bool(true));
                            }
                        }
                        if candidate_score < RELEVANCE_WARN_THRESHOLD
                            && !has_candidate_shift_reason
                            && !derived_from_last_user
                            && !internal_evidence
                        {
                            if let Some(obj) = candidate.payload.as_object_mut() {
                                obj.insert("speculative".to_string(), Value::Bool(true));
                                obj.insert(
                                    "speculative_reason".to_string(),
                                    Value::String("low_relevance_no_reason".to_string()),
                                );
                            }
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "kernel",
                                None,
                                None,
                                json!({
                                    "event": "monologue_candidate_speculative",
                                    "reason": "low_relevance_no_reason",
                                    "candidate_kind": "UpdateWorkspace",
                                    "score": candidate_score,
                                    "anchor": anchor_label.clone(),
                                    "turn": turn_index + 1,
                                }),
                            )
                            .await;
                        }
                        if !has_evidence && anchor_has_verified {
                            if let Some(obj) = candidate.payload.as_object_mut() {
                                obj.insert("speculative".to_string(), Value::Bool(true));
                            }
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "kernel",
                                None,
                                None,
                                json!({
                                    "event": "monologue_candidate_speculative",
                                    "reason": "missing_evidence",
                                    "candidate_kind": "UpdateWorkspace",
                                    "anchor": anchor_label.clone(),
                                    "turn": turn_index + 1,
                                }),
                            )
                            .await;
                        }
                    }

                    if matches!(
                        candidate.kind,
                        CandidateKind::EmitMessage | CandidateKind::AskUserQuestion | CandidateKind::FlagForHuman
                    ) {
                        let clarifier_for_latest = matches!(candidate.kind, CandidateKind::AskUserQuestion)
                            && candidate_overlaps_last_user_input(&candidate, &last_user_input);
                        if !anchor_has_verified && !internal_evidence {
                            if candidate_score < RELEVANCE_WARN_THRESHOLD && !clarifier_for_latest {
                                let _ = system_log::log_event(
                                    &self.db.pool,
                                    Some(&self.app_handle),
                                    "info",
                                    "kernel",
                                    None,
                                    None,
                                    json!({
                                        "event": "monologue_candidate_blocked",
                                        "reason": "anchor_unverified_low_relevance",
                                        "candidate_kind": format!("{:?}", candidate.kind),
                                        "anchor": anchor_label.clone(),
                                        "turn": turn_index + 1,
                                    }),
                                )
                                .await;
                                if matches!(
                                    candidate.kind,
                                    CandidateKind::EmitMessage | CandidateKind::AskUserQuestion | CandidateKind::FlagForHuman
                                ) {
                                    if let Some(obj) = candidate.payload.as_object_mut() {
                                        obj.insert("speculative".to_string(), Value::Bool(true));
                                        obj.insert(
                                            "speculative_reason".to_string(),
                                            Value::String("anchor_unverified".to_string()),
                                        );
                                    }
                                } else {
                                    record_blocked(&candidate, "anchor_unverified_low_relevance");
                                    continue;
                                }
                            }
                            if let Some(obj) = candidate.payload.as_object_mut() {
                                obj.insert("speculative".to_string(), Value::Bool(true));
                                obj.insert(
                                    "speculative_reason".to_string(),
                                    Value::String("anchor_unverified".to_string()),
                                );
                            }
                        }
                        if candidate_score < RELEVANCE_WARN_THRESHOLD
                            && !has_candidate_shift_reason
                            && !clarifier_for_latest
                            && !internal_evidence
                        {
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "kernel",
                                None,
                                None,
                                json!({
                                    "event": "monologue_candidate_blocked",
                                    "reason": "low_relevance_no_reason",
                                    "candidate_kind": format!("{:?}", candidate.kind),
                                    "score": candidate_score,
                                    "anchor": anchor_label.clone(),
                                    "turn": turn_index + 1,
                                }),
                            )
                            .await;
                            if matches!(
                                candidate.kind,
                                CandidateKind::EmitMessage | CandidateKind::AskUserQuestion | CandidateKind::FlagForHuman
                            ) {
                                if let Some(obj) = candidate.payload.as_object_mut() {
                                    obj.insert("speculative".to_string(), Value::Bool(true));
                                    obj.insert(
                                        "speculative_reason".to_string(),
                                        Value::String("low_relevance_no_reason".to_string()),
                                    );
                                }
                            } else {
                                record_blocked(&candidate, "low_relevance_no_reason");
                                continue;
                            }
                        }
                    }

                    turn_candidates.push(candidate);
                    }
                }
            }

            if tool_fabrication_repeat {
                let candidate = self.make_candidate(
                    CandidateKind::WriteEpisodic,
                    json!({
                        "event_type": "meta_cog_reanchor",
                        "payload": {
                            "reason": "tool_fabrication_repeat",
                            "anchor": anchor_label.clone(),
                        },
                        "source_type": "monologue",
                        "source_ref": "tool_fabrication",
                    }),
                    "meta_cog_reanchor",
                    &mut created_at,
                );
                turn_candidates.push(candidate);
                reanchor_needed = true;
            }

            if !is_free_thought {
                let has_meta_cog_question = turn_candidates.iter().any(|c| {
                    matches!(c.kind, CandidateKind::AskUserQuestion)
                });
                let has_internal_evidence = turn_candidates.iter().any(|c| {
                    matches!(candidate_evidence_class(c), Some("internal"))
                        || candidate_has_evidence(&c.payload)
                });
                let has_action_candidate = turn_candidates.iter().any(|c| {
                    matches!(
                        c.kind,
                        CandidateKind::ToolCall
                            | CandidateKind::UpdateWorkspace
                            | CandidateKind::EmitMessage
                            | CandidateKind::FlagForHuman
                            | CandidateKind::PromoteSemantic
                            | CandidateKind::RecordSelfClaim
                            | CandidateKind::SpawnThread
                            | CandidateKind::UpdateGoalThread
                    )
                });
                if message_anchor_hits == 0
                    && !anchor_has_verified
                    && !has_internal_evidence
                    && !has_meta_cog_question
                    && has_action_candidate
                {
                    let _ = system_log::log_contract_violation(
                        &self.db.pool,
                        Some(&self.app_handle),
                        None,
                        None,
                        "C3",
                        "anchor_miss_degrade",
                        Some(json!({
                            "turn": turn_index + 1,
                            "anchor_hits": message_anchor_hits,
                        })),
                    )
                    .await;
                    if !descriptors.iter().any(|d| d == "speculative") {
                        descriptors.push("speculative".to_string());
                    }
                }
            }

            if !is_free_thought && meta_cog_reanchor_attempt && !message.trim().is_empty() {
                let has_user_visible = turn_candidates.iter().any(|c| {
                    matches!(
                        c.kind,
                        CandidateKind::EmitMessage | CandidateKind::AskUserQuestion | CandidateKind::FlagForHuman
                    )
                });
                if !has_user_visible {
                    if let Some((prompt_id, bridge_id)) = record_monologue_intent(
                        self,
                        conversation_id,
                        &message,
                        "AskUserQuestion",
                    )
                    .await
                    {
                        self.log_meta_cog_event(
                            state,
                            None,
                            None,
                            json!({
                                "event": "meta_cog_event",
                                "reason": "reanchor_attempt_prompt_queued",
                                "candidate_id": prompt_id,
                                "bridge_id": bridge_id,
                            }),
                        )
                        .await;
                    }
                }
            }

            if !is_free_thought
                && !message.trim().is_empty()
                && !has_topic_shift_reason
            {
                if let Some(prev_other) = last_message.as_deref() {
                    let overlap = token_similarity(&message, prev_other);
                    if overlap < overlap_threshold {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "monologue_turn_misaligned",
                                "turn": turn_index + 1,
                                "overlap": overlap,
                                "threshold": overlap_threshold,
                                "streak_before": state.monologue_misaligned_streak,
                            }),
                        )
                        .await;
                        state.monologue_misaligned_streak = state.monologue_misaligned_streak.saturating_add(1);
                        let drop_now = if relaxation_level <= 0 {
                            true
                        } else {
                            state.monologue_misaligned_streak >= 3
                        };
                        if drop_now {
                            state.monologue_misaligned_streak = 0;
                            record_suppression(&mut suppression_reasons, "turn_misaligned");
                            message.clear();
                            done = true;
                            if turn_candidates.iter().any(|c| candidate_user_visible(c)) {
                                turn_candidates.retain(|c| candidate_user_visible(c));
                            } else {
                                turn_candidates.clear();
                            }
                        }
                    } else if state.monologue_misaligned_streak > 0 {
                        state.monologue_misaligned_streak = 0;
                    }
                }
            }

            if message.trim().is_empty() {
                novelty_absent_streak = 0;
            }

            if !message.trim().is_empty()
                && !novelty_present
                && novelty_absent_streak >= NOVELTY_ABSENT_K
            {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "monologue_novelty_absent",
                        "turn": turn_index + 1,
                        "novelty_absent_streak": novelty_absent_streak,
                        "anchor_hits": message_anchor_hits,
                    }),
                )
                .await;
                record_suppression(&mut suppression_reasons, "novelty_absent");
                reanchor_needed = true;
                message.clear();
                done = true;
                turn_candidates.clear();
            }

            if !message.trim().is_empty() {
                let mut looped = false;
                let loop_recent_k = settings.loop_recent_k.unwrap_or(6).max(1) as usize;
                for prev in dialogue_messages.iter().rev().take(loop_recent_k) {
                    let overlap = token_similarity(&message, prev);
                    if overlap >= loop_similarity_threshold
                        && novelty_absent_streak >= NOVELTY_ABSENT_K
                    {
                        looped = true;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "monologue_loop_detected",
                                "turn": turn_index + 1,
                                "overlap": overlap,
                                "threshold": loop_similarity_threshold,
                                "novelty_absent_streak": novelty_absent_streak,
                                "suppression_reason": "loop_detected",
                            }),
                        )
                        .await;
                        break;
                    }
                }
                if looped {
                    state.monologue_loop_streak = state.monologue_loop_streak.saturating_add(1);
                    let breaker_triggered = state.monologue_loop_streak >= 2;
                    record_suppression(&mut suppression_reasons, "loop_detected");
                    message.clear();
                    done = true;
                    turn_candidates.clear();
                    let loop_breaker = self.make_candidate(
                        CandidateKind::WriteEpisodic,
                        json!({
                            "event_type": "meta_cog_reanchor",
                            "payload": {
                                "reason": "loop_detected",
                                "anchor": anchor_label.clone(),
                            },
                            "source_type": "monologue",
                            "source_ref": "loop_detected",
                        }),
                        "meta_cog_reanchor",
                        &mut created_at,
                    );
                    turn_candidates.push(loop_breaker);
                    state.last_meta_cog_loop_break_reason = Some("loop_detected".to_string());
                    self.log_meta_cog_event(
                        state,
                        None,
                        None,
                        json!({
                            "event": "meta_cog_event",
                            "reason": "loop_detected",
                            "turn_id": (turn_index + 1) as i64,
                        }),
                    )
                    .await;
                    if breaker_triggered {
                        let clarifier = self.make_candidate(
                            CandidateKind::AskUserQuestion,
                            json!({
                                "question": "I may be looping. What should I focus on next?",
                                "content": "I may be looping. What should I focus on next?",
                                "speculative": true
                            }),
                            "loop_circuit_breaker",
                            &mut created_at,
                        );
                        let no_op = self.make_candidate(
                            CandidateKind::NoOp,
                            json!({
                                "reason": "loop_circuit_breaker",
                                "anchor_hits": message_anchor_hits,
                            }),
                            "do_nothing",
                            &mut created_at,
                        );
                        turn_candidates.push(no_op);
                        turn_candidates.push(clarifier);
                        reanchor_needed = true;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "monologue_loop_circuit_breaker",
                                "turn": turn_index + 1,
                                "streak": state.monologue_loop_streak,
                            }),
                        )
                        .await;
                    }
                } else if !message.trim().is_empty() {
                    state.monologue_loop_streak = 0;
                }
            }

            if message.is_empty() && !turn_candidates.is_empty() {
                message = "(silent)".to_string();
            }
            if decision_needed && message.is_empty() {
                done = true;
            }

            let message_score = if message.trim().is_empty() {
                0.0
            } else {
                relevance_score(&message, &relevance_anchors)
            };
            if !message.trim().is_empty() {
                if message_anchor_hits == 0 {
                    anchor_absent_streak += 1;
                } else {
                    anchor_absent_streak = 0;
                }
                if message_anchor_hits == 0 && !anchor_has_verified {
                    if !descriptors.iter().any(|d| d == "speculative") {
                        descriptors.push("speculative".to_string());
                    }
                }
                if anchor_absent_streak >= 2 {
                    self.log_meta_cog_event(
                        state,
                        None,
                        None,
                        json!({
                            "event": "meta_cog_event",
                            "reason": "anchor_absent",
                            "anchor_vocab_size": anchor_vocab.len(),
                            "anchor_hits": message_anchor_hits,
                            "turn_id": (turn_index + 1) as i64,
                        }),
                    )
                    .await;
                    reanchor_needed = true;
                    record_suppression(&mut suppression_reasons, "anchor_absent");
                    if !descriptors.iter().any(|d| d == "speculative") {
                        descriptors.push("speculative".to_string());
                    }
                }

                if message_anchor_hits == 0 && !anchor_has_verified {
                    let mut kept: Vec<Candidate> = Vec::new();
                    for mut candidate in turn_candidates.into_iter() {
                        if matches!(candidate.kind, CandidateKind::UpdateWorkspace | CandidateKind::RecordSelfClaim) {
                            record_blocked(&candidate, "anchor_missing_message");
                            continue;
                        }
                        if let Some(obj) = candidate.payload.as_object_mut() {
                            obj.insert("speculative".to_string(), Value::Bool(true));
                            obj.insert(
                                "speculative_reason".to_string(),
                                Value::String("anchor_missing".to_string()),
                            );
                        }
                        kept.push(candidate);
                    }
                    turn_candidates = kept;
                    if !descriptors.iter().any(|d| d == "speculative") {
                        descriptors.push("speculative".to_string());
                    }
                }
            }
            if !is_free_thought && !introspection_enabled && anchor_is_empty && !anchor_has_verified && !message.trim().is_empty() {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "monologue_style_blocked",
                        "reason": "anchor_missing_message",
                        "anchor": anchor_label.clone(),
                        "turn": turn_index + 1,
                    }),
                )
                .await;
                record_suppression(&mut suppression_reasons, "anchor_missing_message");
                if !descriptors.iter().any(|d| d == "speculative") {
                    descriptors.push("speculative".to_string());
                }
            }
            if !message.trim().is_empty() && message_score < relevance_warn_threshold {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "monologue_drift",
                        "kind": "message",
                        "score": message_score,
                        "threshold": relevance_warn_threshold,
                        "anchor": anchor_label.clone(),
                        "turn": turn_index + 1,
                        "topic_shift_reason": topic_shift_reason,
                    }),
                )
                .await;
                reanchor_needed = true;
            }
            if message_score < relevance_reject_threshold {
                reanchor_needed = true;
                if !has_topic_shift_reason && !message.trim().is_empty() {
                    if relaxation_level > 0 {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "monologue_low_relevance_soft",
                                "score": message_score,
                                "threshold": relevance_reject_threshold,
                                "anchor": anchor_label.clone(),
                                "turn": turn_index + 1,
                            }),
                        )
                        .await;
                        record_suppression(&mut suppression_reasons, "low_relevance_soft");
                        if !descriptors.iter().any(|d| d == "speculative") {
                            descriptors.push("speculative".to_string());
                        }
                    } else {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "monologue_low_relevance_drop",
                                "score": message_score,
                                "threshold": relevance_reject_threshold,
                                "anchor": anchor_label.clone(),
                                "turn": turn_index + 1,
                            }),
                        )
                        .await;
                        record_suppression(&mut suppression_reasons, "low_relevance_drop");
                        message.clear();
                        done = true;
                    }
                }
            }

            if !message.trim().is_empty() {
                let repeat_threshold = settings
                    .loop_similarity_threshold
                    .unwrap_or(0.85)
                    .clamp(0.0, 1.0);
                let mut is_repeat = false;
                for prior in dialogue_messages.iter().rev().take(4) {
                    if token_similarity(&message, prior) >= repeat_threshold {
                        is_repeat = true;
                        break;
                    }
                }
                if !is_repeat {
                    for line in recent_block.lines() {
                        if token_similarity(&message, line) >= repeat_threshold {
                            is_repeat = true;
                            break;
                        }
                    }
                }
                if is_repeat {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "monologue_repeat_suppressed",
                            "turn": turn_index + 1,
                        }),
                    )
                    .await;
                    record_suppression(&mut suppression_reasons, "repeat");
                    message.clear();
                    done = true;
                }
            }

            if settings.enable_monologue_validator.unwrap_or(true) && !message.trim().is_empty() {
                if response_has_user_attribution(&message, user_name) {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "monologue_user_confusion",
                            "turn": turn_index + 1,
                            "speaker": stance_selected,
                            "snippet": summarize_snippet(&message, 160),
                            "stream": stream.as_str(),
                        }),
                    )
                    .await;
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "monologue_validator",
                            "outcome": "suppressed",
                            "reason": "user_confusion",
                            "turn": turn_index + 1,
                        }),
                    )
                    .await;
                    if !is_free_thought {
                        record_suppression(&mut suppression_reasons, "user_confusion");
                        turn_candidates.clear();
                    }
                }
                if !message.trim().is_empty() {
                    if let Some(reason) = monologue_style_violation(&message, user_name) {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "monologue_style_blocked",
                                "reason": reason,
                                "turn": turn_index + 1,
                                "speaker": stance_selected,
                                "snippet": summarize_snippet(&message, 160),
                                "stream": stream.as_str(),
                        }),
                        )
                        .await;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "monologue_validator",
                                "outcome": "suppressed",
                                "reason": reason,
                                "turn": turn_index + 1,
                            }),
                        )
                        .await;
                    let sanitized = sanitize_monologue_style(&message, user_name);
                    let still_violates = monologue_style_violation(&sanitized, user_name);
                    if !is_free_thought {
                        if sanitized.trim().is_empty() || still_violates.is_some() {
                            record_suppression(&mut suppression_reasons, "style_blocked");
                            message = if sanitized.trim().is_empty() {
                                "(sanitized)".to_string()
                            } else {
                                sanitized
                            };
                        } else {
                            message = sanitized;
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "kernel",
                                None,
                                None,
                                json!({
                                    "event": "monologue_style_sanitized",
                                    "turn": turn_index + 1,
                                    "speaker": stance_selected,
                                    "stream": stream.as_str(),
                                }),
                            )
                            .await;
                        }
                    } else {
                        message = sanitized;
                    }
                }
            }
            }
            if !message.trim().is_empty() && !is_free_thought {
                if monologue_confuses_system_output(&message, &last_user_input, &last_assistant_output) {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "monologue_identity_confusion",
                            "turn": turn_index + 1,
                            "speaker": stance_selected,
                            "snippet": summarize_snippet(&message, 160),
                            "stream": stream.as_str(),
                        }),
                    )
                    .await;
                    record_suppression(&mut suppression_reasons, "identity_confusion");
                    message.clear();
                    done = true;
                    turn_candidates.clear();
                } else {
                    let tokens = extract_numeric_tokens(&message);
                    if !tokens.is_empty()
                        && tokens.iter().any(|t| !numeric_token_allowed(t, &last_user_input, &state.telemetry_snapshot))
                    {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "monologue_numeric_relaxed",
                                "turn": turn_index + 1,
                                "speaker": stance_selected,
                                "snippet": summarize_snippet(&message, 160),
                                "tokens": tokens,
                                "stream": stream.as_str(),
                            }),
                        )
                        .await;
                        if !descriptors.iter().any(|d| d == "numeric_unverified") {
                            descriptors.push("numeric_unverified".to_string());
                        }
                        if !descriptors.iter().any(|d| d == "speculative") {
                            descriptors.push("speculative".to_string());
                        }
                    }
                }
            } else if !message.trim().is_empty() && is_free_thought {
                let tokens = extract_numeric_tokens(&message);
                if !tokens.is_empty()
                    && tokens.iter().any(|t| !numeric_token_allowed(t, &last_user_input, &state.telemetry_snapshot))
                {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "monologue_numeric_relaxed",
                            "turn": turn_index + 1,
                            "speaker": stance_selected,
                            "snippet": summarize_snippet(&message, 160),
                            "tokens": tokens,
                            "stream": stream.as_str(),
                        }),
                    )
                    .await;
                    if !descriptors.iter().any(|d| d == "numeric_unverified") {
                        descriptors.push("numeric_unverified".to_string());
                    }
                    if !descriptors.iter().any(|d| d == "speculative") {
                        descriptors.push("speculative".to_string());
                    }
                }
            }

            if !is_free_thought
                && turn_candidates.is_empty()
                && message.trim().is_empty()
                && message_anchor_hits == 0
                && state.pending_questions.is_empty()
            {
                self.log_meta_cog_event(
                    state,
                    None,
                    None,
                    json!({
                        "event": "meta_cog_event",
                        "reason": "do_nothing_weak_anchors",
                        "anchor_hits": message_anchor_hits,
                    }),
                )
                .await;
                record_suppression(&mut suppression_reasons, "anchor_missing_message");
                if !descriptors.iter().any(|d| d == "speculative") {
                    descriptors.push("speculative".to_string());
                }
            }

            let silent = message.trim().is_empty() && turn_candidates.is_empty();
            if silent && !decision_needed {
                if is_free_thought && (turn_index + 1) < FTS_MIN_TURNS {
                    done = false;
                } else {
                    break;
                }
            }

            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "monologue_candidate_summary",
                    "turn": turn_index + 1,
                    "candidates_kept": turn_candidates.len(),
                    "blocked_candidates": blocked_candidates.len(),
                    "suppression_reasons": suppression_reasons,
                    "message_empty": message.trim().is_empty(),
                }),
            )
            .await;

            if !message.trim().is_empty() {
                let entry_descriptors = if descriptors.is_empty() {
                    None
                } else {
                    Some(descriptors.clone())
                };
                let (harvest_type, harvest_payload) = if is_free_thought && !anchor_has_verified {
                    (
                        Some("speculative".to_string()),
                        Some(json!({"speculative": true}).to_string()),
                    )
                } else {
                    (None, None)
                };
                dialogue_messages.push(message.clone());
                last_message = Some(message.clone());
                last_message_interrogative = message_interrogative;
                turns.push(MonologueTurn {
                    entry: crate::models::InnerMonologueEntry {
                        id: Uuid::new_v4().to_string(),
                        conversation_id: conversation_id.to_string(),
                        run_id: None,
                        dialogue_id: Some(dialogue_id.clone()),
                        turn_index: Some((turn_index + 1) as i64),
                        speaker: Some(stance_selected.to_string()),
                        mode: format!("{:?}", state.mode).to_lowercase(),
                        stream_type: Some(stream.as_str().to_string()),
                        thought: message.clone(),
                        descriptors: entry_descriptors,
                        harvest_type,
                        harvest_payload,
                        created_at: Utc::now().to_rfc3339(),
                        candidates: None,
                    },
                    candidates: turn_candidates,
                    blocked_candidates,
                });

                if speaker_a {
                    history_a.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: message.clone(),
                    });
                    history_b.push(ChatMessage {
                        role: "user".to_string(),
                        content: message,
                    });
                } else {
                    history_b.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: message.clone(),
                    });
                    history_a.push(ChatMessage {
                        role: "user".to_string(),
                        content: message,
                    });
                }
            }

            if reanchor_needed {
                let recency_secs = settings
                    .pending_prompt_recency_secs
                    .unwrap_or(PENDING_PROMPT_RECENCY_SECS_DEFAULT);
                let recent_user = state
                    .last_user_input_at
                    .as_deref()
                    .and_then(crate::core::kernel::utils::time::timestamp_from_str)
                    .map(|ts| {
                        let age = Utc::now().signed_duration_since(ts).num_seconds();
                        age >= 0 && age <= recency_secs
                    })
                    .unwrap_or(false);
                if !recent_user {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!( {
                            "event": "monologue_reanchor_skipped",
                            "reason": "stale_user_input",
                            "turn": turn_index + 1,
                            "anchor": anchor_label.clone(),
                            "recency_secs": recency_secs,
                        }),
                    )
                    .await;
                } else {
                    let reanchor_note = format!("Re-anchor: stay on the current topic: {}.", anchor_label);
                    history_a.push(ChatMessage {
                        role: "user".to_string(),
                        content: reanchor_note.clone(),
                    });
                    history_b.push(ChatMessage {
                        role: "user".to_string(),
                        content: reanchor_note,
                    });
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "monologue_reanchor",
                            "turn": turn_index + 1,
                            "anchor": anchor_label.clone(),
                        }),
                    )
                    .await;
                    state.last_meta_cog_loop_break_reason = Some("reanchor_needed".to_string());
                    self.log_meta_cog_event(
                        state,
                        None,
                        None,
                        json!({
                            "event": "meta_cog_event",
                            "reason": "reanchor_needed",
                            "turn_id": (turn_index + 1) as i64,
                            "anchor": anchor_label.clone(),
                        }),
                    )
                    .await;
                }
            }

            turn_index += 1;
            speaker_a = !speaker_a;

            if !decision_needed && done {
                break;
            }

            tokio::task::yield_now().await;
        }

        if !suppression_reasons.is_empty() {
            let _ = system_log::log_suppression_summary(
                &self.db.pool,
                Some(&self.app_handle),
                "kernel",
                None,
                None,
                "monologue_suppression_summary",
                stream.as_str(),
                &suppression_reasons,
                Some(json!({
                    "turns_attempted": turn_index,
                    "turns_saved": turns.len(),
                })),
            )
            .await;
        }
        self.update_monologue_suppression_window(state, stream.as_str(), &suppression_reasons)
            .await;

        Ok(MonologueOutput {
            turns,
            last_message,
            dialogue_messages,
        })
    }



    pub(super) fn build_workspace_fallback_payload(
        _summary: &mut InnerSummary,
        state: &KernelState,
        focus_seed: &str,
    ) -> Option<Value> {
        let focus_seed = focus_seed.trim();
        let mut payload = serde_json::Map::new();

        if !focus_seed.is_empty() && !is_none_marker(focus_seed) {
            payload.insert(
                "current_focus".to_string(),
                Value::String(summarize_snippet(focus_seed, 160)),
            );
            payload.insert(
                "focus_rationale".to_string(),
                Value::String("fallback from focus seed".to_string()),
            );
        } else if let Some(focus) = workspace_verified_focus(state) {
            payload.insert("current_focus".to_string(), json!(focus));
            payload.insert(
                "focus_rationale".to_string(),
                json!("fallback from existing workspace"),
            );
        }

        let verified_questions = workspace_verified_open_questions(state);
        if !verified_questions.is_empty() {
            payload.insert(
                "open_questions".to_string(),
                json!(verified_questions.into_iter().take(8).collect::<Vec<_>>()),
            );
        }

        let verified_topics = workspace_verified_topics(state);
        if !verified_topics.is_empty() {
            payload.insert(
                "working_set_topics".to_string(),
                json!(verified_topics.into_iter().take(8).collect::<Vec<_>>()),
            );
        }

        let verified_hypotheses = state
            .workspace_active_hypotheses
            .iter()
            .filter(|hypothesis| hypothesis_is_verified(hypothesis))
            .filter_map(|hypothesis| {
                let trimmed = hypothesis.text.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            })
            .take(8)
            .collect::<Vec<_>>();
        if !verified_hypotheses.is_empty() {
            payload.insert("active_hypotheses".to_string(), json!(verified_hypotheses));
        }

        if payload.is_empty() {
            return None;
        }

        Some(Value::Object(payload))
    }

    pub(super) async fn build_workspace_fallback_candidate(
        &self,
        conversation_id: &str,
        state: &KernelState,
        summary_json: Option<&str>,
        created_at: &mut i64,
    ) -> Option<Candidate> {
        let fallback_raw = if let Some(raw) = summary_json
            .map(|raw| raw.to_string())
            .filter(|raw| !raw.trim().is_empty())
        {
            raw
        } else {
            self.db
                .get_inner_summary(conversation_id)
                .await
                .ok()
                .flatten()
                .unwrap_or_else(|| "{}".to_string())
        };

        let mut summary = InnerSummary::from_json(&fallback_raw);
        let focus_seed = state.last_user_input.as_deref().unwrap_or("");
        let payload = Self::build_workspace_fallback_payload(&mut summary, state, focus_seed)?;

        Some(self.make_candidate(
            CandidateKind::UpdateWorkspace,
            payload,
            "workspace_fallback",
            created_at,
        ))
    }







    pub(super) async fn build_deliberation_semantic_hint(
        &self,
        conversation_id: &str,
        input: &str,
    ) -> String {
        let mut parts = Vec::new();
        let trimmed_input = input.trim();
        if !trimmed_input.is_empty() {
            parts.push(trimmed_input.to_string());
        }

        let inner_summary_raw = self
            .db
            .get_inner_summary(conversation_id)
            .await
            .ok()
            .flatten()
            .unwrap_or_else(|| "{}".to_string());
        let inner_summary = InnerSummary::from_json(&inner_summary_raw);
        if !inner_summary.focus.trim().is_empty() {
            parts.push(inner_summary.focus.clone());
        }
        if !inner_summary.next_moves.is_empty() {
            parts.push(inner_summary.next_moves.join(" "));
        }
        if !inner_summary.blockers.is_empty() {
            parts.push(inner_summary.blockers.join(" "));
        }
        if !inner_summary.open_questions.is_empty() {
            parts.push(inner_summary.open_questions.join(" "));
        }

        let query = parts.join(" | ").trim().to_string();
        if query.is_empty() {
            return "None".to_string();
        }

        let api = crate::core::memory::api::MemoryApi::new(
            self.db.pool.clone(),
            Some(self.model_client.clone()),
            format!("deliberation:{}", conversation_id),
        )
        .await;
        let intent = crate::core::memory::api::infer_query_intent(&query);
        if let Ok(packet) = api
            .retrieve(
                &query,
                &[
                    crate::core::memory::types::Scope::Session,
                    crate::core::memory::types::Scope::Global,
                ],
                intent,
            )
            .await
        {
            let mut hints = Vec::new();
            for fact in packet.facts.iter().take(4) {
                hints.push(format!("{}: {} = {}", fact.entity_label, fact.key, fact.value));
            }
            for rel in packet.relations.iter().take(3) {
                let participants = rel
                    .participants
                    .iter()
                    .map(|p| format!("{}:{}", p.role, p.entity_label))
                    .collect::<Vec<_>>()
                    .join(", ");
                hints.push(format!("{}({})", rel.rel_type, participants));
            }
            if hints.is_empty() {
                "None".to_string()
            } else {
                let hint_block = hints.join(" | ");
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "memory",
                    None,
                    None,
                    json!({
                        "event": "deliberation_semantic_hint",
                        "snippet": summarize_snippet(&hint_block, 240),
                    }),
                )
                .await;
                hint_block
            }
        } else if let Ok(core) = self.db.get_semantic_core().await {
            if core.trim().is_empty() {
                "None".to_string()
            } else {
                core.chars().take(240).collect()
            }
        } else {
            "None".to_string()
        }
    }

    pub(super) async fn intent_sanity_check_note(
        &self,
        conversation_id: &str,
        input: &str,
        run_id: &str,
        trace_id: Option<&str>,
    ) -> Option<String> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return None;
        }
        let lowered = trimmed.to_lowercase();
        let correction_markers = [
            "actually",
            "wait",
            "correction",
            "i meant",
            "i mean",
            "scratch that",
            "let me rephrase",
            "not that",
        ];
        let has_marker = lowered.starts_with("no ")
            || correction_markers.iter().any(|m| lowered.contains(m));
        if !has_marker {
            return None;
        }
        let history = self
            .db
            .get_history_for_conversation(conversation_id, 6)
            .await
            .unwrap_or_default();
        let mut user_messages: Vec<String> = history
            .iter()
            .filter(|msg| msg.role == "user")
            .map(|msg| msg.content.clone())
            .collect();
        if let Some(last) = user_messages.last() {
            if last.trim() == trimmed {
                user_messages.pop();
            }
        }
        if user_messages.is_empty() {
            return None;
        }
        let current_tokens = tokenize_for_similarity(trimmed);
        if current_tokens.len() < 3 {
            return None;
        }
        let mut max_similarity = 0.0;
        let mut prior_snippet = String::new();
        for msg in user_messages.iter().rev().take(3) {
            let tokens = tokenize_for_similarity(msg);
            let similarity = jaccard_similarity(&current_tokens, &tokens);
            if similarity > max_similarity {
                max_similarity = similarity;
                prior_snippet = summarize_snippet(msg, 160);
            }
        }
        if max_similarity < 0.2 {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                trace_id,
                json!({
                    "event": "intent_sanity_check",
                    "similarity_max": max_similarity,
                    "prior_snippet": prior_snippet,
                }),
            )
            .await;
            return Some("Intent sanity check: the latest request looks like a correction or negation of a recent turn. Prioritize the newest instruction and confirm if anything conflicts.".to_string());
        }
        None
    }


    pub(super) async fn build_introspection_summary(
        &self,
        conversation_id: &str,
        _settings: &crate::models::Settings,
        state: &KernelState,
    ) -> Option<String> {
        let working = state.working_memory.clone().unwrap_or_default();
        let controller = state.controller_state.clone().unwrap_or_default();
        let payload = json!({
            "focus": working.focus,
            "open_questions": working.open_questions,
            "active_hypotheses": working.active_hypotheses,
            "next_action": working.next_action,
            "confidence_summary": working.confidence,
            "drift_score": working.drift_score,
            "controller_confidence": controller.confidence,
            "controller_uncertainty": controller.uncertainty,
            "controller_drift": controller.drift_score,
            "uncertainty_count": state.uncertainty_count,
            "evidence_event_ids": [],
        });

        let cache_key = hash_payload(&format!(
            "{}:{}:{}",
            conversation_id,
            payload.get("focus").and_then(|v| v.as_str()).unwrap_or(""),
            payload.get("next_action").and_then(|v| v.as_str()).unwrap_or("")
        ));
        if let Some(cached) = {
            let cache = self.introspection_cache.lock().await;
            cache.get(conversation_id).cloned()
        } {
            if cached.key == cache_key {
                return Some(cached.summary);
            }
        }

        let summary = payload.to_string();
        let mut cache = self.introspection_cache.lock().await;
        cache.insert(
            conversation_id.to_string(),
            IntrospectionCacheEntry {
                key: cache_key,
                summary: summary.clone(),
            },
        );

        Some(summary)
    }

    pub(super) async fn build_feedback_bundle(
        &self,
        conversation_id: &str,
        state: &KernelState,
        introspection_summary: Option<&str>,
    ) -> FeedbackBundleOutput {
        let controller = state.controller_state.clone().unwrap_or_default();
        let outcome_label = map_outcome_quality(controller.outcome_quality);
        let confidence_label = map_confidence_level(controller.confidence);
        let evidence_label = map_evidence_coverage(controller.evidence_coverage);

        let gate_row = sqlx::query(
            "SELECT decision, evidence_refs_json
             FROM gate_decisions
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten();
        let (gate_decision, gate_reasons) = if let Some(row) = gate_row {
            let decision: String = row.try_get("decision").unwrap_or_default();
            let raw_refs: String = row.try_get("evidence_refs_json").unwrap_or_default();
            let reasons = parse_gate_reasons(&raw_refs);
            (Some(decision), reasons)
        } else {
            (None, Vec::new())
        };
        let gate_notice = gate_decision
            .as_deref()
            .and_then(|decision| gate_notice_for(decision, &gate_reasons));
        let policy_label = map_policy_adherence(gate_decision.as_deref());

        let qualia_state = qualia::compute_qualia_state(&self.db, None)
            .await
            .unwrap_or_else(|_| qualia::QualiaState {
                timestamp: chrono::Utc::now().to_rfc3339(),
                dominant_tag: None,
                dominant_intensity: 0.0,
                recent_labels: Vec::new(),
                last_reward: None,
                predicted_tag: None,
                prediction_confidence: 0.0,
                matched_workspace_refs: Vec::new(),
            });
        let qualia_snapshot = format_qualia_snapshot(&qualia_state);
        if !qualia_snapshot.trim().is_empty() && qualia_snapshot.trim() != "None" {
            let _ = self
                .db
                .create_qualia_snapshot_evidence_event(conversation_id, &qualia_snapshot, Some("feedback_bundle"))
                .await;
        }
        let qualia_delta = qualia_state
            .recent_labels
            .first()
            .map(|label| {
                let reward = qualia_state
                    .last_reward
                    .map(|value| format!("{:.3}", value))
                    .unwrap_or_else(|| "none".to_string());
                format!("{}:{:.2} reward={}", label.tag, label.intensity, reward)
            })
            .unwrap_or_else(|| "none".to_string());

        let context_tags = self.db.get_context_tags(conversation_id, 120).await;
        let intent_summary = self.db.get_user_intent_summary(conversation_id).await;

        let max_chars: usize = 450;
        let mut lines: Vec<String> = Vec::new();
        let push_line = |line: String, lines: &mut Vec<String>| -> bool {
            let current_len: usize = lines.iter().map(|l| l.len()).sum::<usize>() + lines.len();
            let next_len = current_len + line.len() + 1;
            if next_len <= max_chars {
                lines.push(line);
                true
            } else {
                false
            }
        };

        // Required fields (keep these short and always present)
        push_line(format!("last_turn_outcome: {}", outcome_label), &mut lines);
        push_line(format!("confidence: {}", confidence_label), &mut lines);
        push_line(format!("policy_adherence: {}", policy_label), &mut lines);
        push_line(format!("evidence_coverage: {}", evidence_label), &mut lines);
        push_line(format!("qualia_delta: {}", qualia_delta), &mut lines);
        if let Some(notice) = gate_notice.as_deref() {
            push_line(format!("gate_notice: {}", summarize_snippet(notice, 140)), &mut lines);
        } else {
            push_line("gate_notice: none".to_string(), &mut lines);
        }
        if gate_reasons.is_empty() {
            push_line("gate_reasons: none".to_string(), &mut lines);
        } else {
            push_line(format!("gate_reasons: {}", gate_reasons.join(", ")), &mut lines);
        }

        // Optional fields, add only if they fit
        if context_tags.is_empty() {
            push_line("user_context_tags: none".to_string(), &mut lines);
        } else {
            if push_line("user_context_tags:".to_string(), &mut lines) {
                for tag in context_tags.iter().take(3) {
                    let evidence = if tag.evidence_event_ids.is_empty() {
                        "none".to_string()
                    } else {
                        tag.evidence_event_ids
                            .iter()
                            .map(|id| id.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    };
                    if !push_line(
                        format!(
                            "- {} (conf={:.2}, inferred={}, evidence=[{}])",
                            tag.tag, tag.confidence, tag.inferred, evidence
                        ),
                        &mut lines,
                    ) {
                        lines.pop();
                        push_line(
                            format!("user_context_tags: {} (truncated)", context_tags.len()),
                            &mut lines,
                        );
                        break;
                    }
                }
            } else {
                push_line(
                    format!("user_context_tags: {} (truncated)", context_tags.len()),
                    &mut lines,
                );
            }
        }

        if let Some(summary) = intent_summary.as_ref() {
            let confirmed = if summary.confirmed { "confirmed" } else { "unconfirmed" };
            let line = format!(
                "user_intent_summary: {} ({})",
                summarize_snippet(&summary.summary, 80),
                confirmed
            );
            if !push_line(line, &mut lines) {
                push_line(
                    format!("user_intent_summary: present ({})", confirmed),
                    &mut lines,
                );
            }
        } else {
            push_line("user_intent_summary: none".to_string(), &mut lines);
        }

        if let Some(summary) = introspection_summary {
            let line = format!(
                "self_eval_summary: {}",
                summarize_snippet(summary, 80)
            );
            if !push_line(line, &mut lines) {
                push_line("self_eval_summary: present (truncated)".to_string(), &mut lines);
            }
        } else {
            push_line("self_eval_summary: none".to_string(), &mut lines);
        }

        let payload = json!({
            "last_turn_outcome": outcome_label,
            "confidence": confidence_label,
            "policy_adherence": policy_label,
            "evidence_coverage": evidence_label,
            "qualia_delta": qualia_delta,
            "gate_notice": gate_notice,
            "gate_reasons": gate_reasons,
            "user_context_tags": context_tags,
            "user_intent_summary": intent_summary.as_ref().map(|summary| {
                let mut trimmed = summary.clone();
                trimmed.summary = summarize_snippet(&summary.summary, 120);
                trimmed
            }),
            "self_eval_summary": introspection_summary.map(|s| summarize_snippet(s, 120)),
        });

        FeedbackBundleOutput {
            prompt_text: lines.join("\n"),
            qualia_snapshot,
            payload,
        }
    }

    pub(super) async fn record_calibration_change(
        &self,
        snapshot_hash: &str,
        knob: &str,
        old_value: f32,
        new_value: f32,
        reason: &str,
    ) {
        let _ = sqlx::query(
            "INSERT INTO calibration_changes (change_id, snapshot_hash, knob, old_value, new_value, reason, created_at)
             VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(Uuid::new_v4().to_string())
        .bind(snapshot_hash)
        .bind(knob)
        .bind(old_value)
        .bind(new_value)
        .bind(reason)
        .execute(&self.db.pool)
        .await;
    }

    pub(super) async fn record_introspection_entry(
        &self,
        state: &mut KernelState,
        decision: &KernelDecision,
        subject_state: &subject_state::SubjectState,
        snapshot_hash: &str,
        summary: &str,
    ) {
        let mode: Option<String> = sqlx::query_scalar(
            "SELECT mode FROM system_controls WHERE subsystem_id = ?",
        )
        .bind("introspection")
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten();
        let introspection_mode = mode.unwrap_or_else(|| {
            system_controls::default_mode_for("introspection")
                .unwrap_or("normal")
                .to_string()
        });
        if system_controls::mode_is_off(&introspection_mode)
            || system_controls::mode_is_degraded(&introspection_mode)
        {
            return;
        }
        let entry_id = Uuid::new_v4().to_string();
        let workspace_refs = json!({
            "broadcast_refs": subject_state.workspace.broadcast_refs,
            "winners": subject_state.workspace.winners,
        });
        let prediction_refs = subject_state
            .error_state
            .recent_residuals
            .iter()
            .map(|res| res.prediction_id.clone())
            .collect::<Vec<_>>();
        let numeric_payload = json!({
            "confidence_summary": subject_state.self_model.controller_state.confidence,
            "uncertainty_summary": subject_state.self_model.controller_state.uncertainty,
            "residual_summary": subject_state.error_state.recent_residuals,
            "organism_summary": subject_state.organism,
            "ignition_summary": subject_state.workspace.ignition,
            "conflicts_count": subject_state.self_model.conflicts_count,
            "gate_state": decision.report.gate_decision,
        });
        let narrative = summary.chars().take(400).collect::<String>();
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            decision.report.gate_decision_id.as_deref(),
            None,
            json!({
                "event": "introspection_entry_attempted",
                "entry_id": entry_id,
                "snapshot_hash": snapshot_hash,
            }),
        )
        .await;
        let insert_result = sqlx::query(
            "INSERT INTO introspection_entries
             (entry_id, snapshot_hash, workspace_refs_json, event_refs_json, prediction_refs_json, error_refs_json, numeric_payload_json, narrative, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(&entry_id)
        .bind(snapshot_hash)
        .bind(workspace_refs.to_string())
        .bind(json!([]).to_string())
        .bind(json!(prediction_refs).to_string())
        .bind(json!(subject_state.error_state.open_error_ids).to_string())
        .bind(numeric_payload.to_string())
        .bind(narrative)
        .execute(&self.db.pool)
        .await;
        if let Err(err) = insert_result {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "kernel",
                decision.report.gate_decision_id.as_deref(),
                None,
                json!({
                    "event": "introspection_entry_write_failed",
                    "entry_id": entry_id,
                    "snapshot_hash": snapshot_hash,
                    "error": err.to_string(),
                }),
            )
            .await;
        } else {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                decision.report.gate_decision_id.as_deref(),
                None,
                json!({
                    "event": "introspection_entry_written",
                    "entry_id": entry_id,
                    "snapshot_hash": snapshot_hash,
                }),
            )
            .await;
        }

        let mut discrepancy_score = 0.0;
        if subject_state.error_state.open_error_count > 0 {
            discrepancy_score += 0.3;
        }
        if subject_state.self_model.controller_state.drift_score > 0.6 {
            discrepancy_score += 0.4;
        }
        if subject_state.organism.integrity_risk > 0.7 {
            discrepancy_score += 0.2;
        }
        let recommended_action = if discrepancy_score >= 0.7 {
            "REQUIRE_VERIFY"
        } else if discrepancy_score >= 0.4 {
            "LOWER_INTROSPECTION_WEIGHT"
        } else if subject_state.error_state.open_error_count >= 3 {
            "DIAGNOSE_LOOP"
        } else {
            "NONE"
        };
        let audit_id = Uuid::new_v4().to_string();
        let _ = sqlx::query(
            "INSERT INTO audit_log (audit_id, target_id, snapshot_hash, checks_json, discrepancy_score, recommended_action, created_at)
             VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(&audit_id)
        .bind(&entry_id)
        .bind(snapshot_hash)
        .bind(json!({ "discrepancy_score": discrepancy_score }).to_string())
        .bind(discrepancy_score)
        .bind(recommended_action)
        .execute(&self.db.pool)
        .await;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            decision.report.gate_decision_id.as_deref(),
            None,
            json!({
                "event": "audit_log_written",
                "audit_id": audit_id,
                "target_id": entry_id,
                "recommended_action": recommended_action,
                "discrepancy_score": discrepancy_score,
            }),
        )
        .await;

        if recommended_action == "LOWER_INTROSPECTION_WEIGHT" {
            let current = state.introspection_weight.unwrap_or(0.5);
            let next = (current - 0.1).max(0.1);
            state.introspection_weight = Some(next);
            self.record_calibration_change(snapshot_hash, "introspection_weight", current, next, recommended_action)
                .await;
        } else if recommended_action == "REQUIRE_VERIFY" {
            let current = state.verify_threshold.unwrap_or(0.5);
            let next = (current + 0.1).min(1.0);
            state.verify_threshold = Some(next);
            self.record_calibration_change(snapshot_hash, "verify_threshold", current, next, recommended_action)
                .await;
        } else if recommended_action == "DIAGNOSE_LOOP" {
            state.introspection_force = Some("diagnose_loop".to_string());
        }
    }

    pub(super) async fn update_latency_avg(&self, key: &str, duration_ms: i64) {
        if duration_ms <= 0 {
            return;
        }
        let avg_key = format!("latency_{}_avg_ms", key);
        let count_key = format!("latency_{}_count", key);
        let prev_avg = self
            .db
            .get_key(&avg_key)
            .await
            .ok()
            .flatten()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0);
        let prev_count = self
            .db
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
        let _ = self.db.set_key(&avg_key, &format!("{:.2}", next_avg)).await;
        let _ = self.db.set_key(&count_key, &next_count.to_string()).await;
    }

    pub(super) async fn reflect_working_memory(
        &self,
        conversation_id: &str,
        input: &str,
        state: &KernelState,
        settings: &crate::models::Settings,
    ) -> Option<WorkingMemoryBlock> {
        let prompt_set = prompt_loader::get_prompts().ok()?;
        let reflection_prompt = prompt_set.reflection_prompt.trim();
        if reflection_prompt.is_empty() {
            return None;
        }
        let temperature = self
            .db
            .get_key("introspection_temp_override")
            .await
            .ok()
            .flatten()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.3);

        let workspace_state = self.db.get_workspace_state(conversation_id).await.ok().flatten();
        let evidence_ids = self.db.get_recent_user_evidence_ids(conversation_id, 20).await;
        let packet = json!({
            "working_memory": state.working_memory,
            "workspace_state": workspace_state,
            "controller_state": state.controller_state,
            "self_state": state.self_state,
            "last_user_input": state.last_user_input,
            "input": input,
            "evidence_event_ids": evidence_ids,
        });

        let (summary_model, summary_url) = select_summary_model(settings);
        let user_prompt = format!("Reflection packet:\n{}\n\nReturn JSON only.", packet);
        let (user_prompt, prompt_truncated) =
            cap_summary_prompt(reflection_prompt, &user_prompt, settings);
        if prompt_truncated {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "summary",
                None,
                None,
                json!({
                    "event": "summary_prompt_capped",
                    "cap_tokens": summary_prompt_cap_tokens(settings),
                    "source": "working_memory_reflection",
                }),
            )
            .await;
        }
        let request = ChatCompletionRequest {
            model: summary_model,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: reflection_prompt.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            stream: false,
            temperature: Some(temperature),
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
            json_strict: Some(false),
                skip_sanitization: None,
            run_id: None,
            request_label: Some("working_memory_reflection".to_string()),
        };

        let started = Instant::now();
        let response = self
            .model_client
            .chat_with_meta(&summary_url, settings.api_key.as_deref(), &request)
            .await
            .ok()?;
        let latency_ms = started.elapsed().as_millis() as i64;
        self.update_latency_avg("introspection", latency_ms).await;

        let raw_value: serde_json::Value = serde_json::from_str(&response.content).ok()?;
        let allowlist: Vec<i64> = packet
            .get("evidence_event_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64())
                    .collect::<Vec<i64>>()
            })
            .unwrap_or_default();

        let working = state.working_memory.clone().unwrap_or_default();
        let mut allowed_focus = Vec::new();
        if let Some(focus) = working.focus.as_deref() {
            allowed_focus.push(focus.to_lowercase());
        }
        if let Some(ws) = workspace_state.as_ref() {
            if let Some(focus) = ws.current_focus.as_deref() {
                allowed_focus.push(focus.to_lowercase());
            }
            if let Some(goal) = ws.goal_thread.as_deref() {
                allowed_focus.push(goal.to_lowercase());
            }
        }

        let mut allowed_questions = Vec::new();
        for q in working.open_questions.iter() {
            allowed_questions.push(q.to_lowercase());
        }
        if let Some(ws) = workspace_state.as_ref() {
            for q in ws.open_questions.iter() {
                allowed_questions.push(q.to_lowercase());
            }
        }

        let mut allowed_hypotheses = Vec::new();
        for h in working.active_hypotheses.iter() {
            allowed_hypotheses.push(h.to_lowercase());
        }
        if let Some(ws) = workspace_state.as_ref() {
            for h in ws.active_hypotheses.iter() {
                allowed_hypotheses.push(h.text.to_lowercase());
            }
        }

        let allowed_next_action = working
            .next_action
            .as_deref()
            .map(|s| s.to_lowercase());

        let evidence_ids: Vec<i64> = raw_value
            .get("evidence_event_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64())
                    .filter(|id| allowlist.contains(id))
                    .collect::<Vec<i64>>()
            })
            .unwrap_or_default();

        let focus = raw_value
            .get("focus")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .and_then(|f| {
                if allowed_focus.iter().any(|a| a == &f.to_lowercase()) {
                    Some(f.to_string())
                } else {
                    None
                }
            });

        let open_questions = raw_value
            .get("open_questions")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .filter(|q| allowed_questions.iter().any(|a| a == &q.to_lowercase()))
                    .map(|s| s.to_string())
                    .take(3)
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();

        let active_hypotheses = raw_value
            .get("active_hypotheses")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .filter(|h| allowed_hypotheses.iter().any(|a| a == &h.to_lowercase()))
                    .map(|s| s.to_string())
                    .take(3)
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();

        let next_action = raw_value
            .get("next_action")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .and_then(|n| {
                if allowed_next_action.as_deref() == Some(&n.to_lowercase()) {
                    Some(n.to_string())
                } else {
                    None
                }
            });

        let confidence = state
            .controller_state
            .as_ref()
            .map(|c| c.confidence)
            .or_else(|| raw_value.get("confidence").and_then(|v| v.as_f64()).map(|v| v as f32));
        let drift_score = state
            .controller_state
            .as_ref()
            .map(|c| c.drift_score)
            .or_else(|| raw_value.get("drift_score").and_then(|v| v.as_f64()).map(|v| v as f32));

        let accepted_fields = [
            focus.is_some(),
            !open_questions.is_empty(),
            !active_hypotheses.is_empty(),
            next_action.is_some(),
            confidence.is_some(),
            drift_score.is_some(),
        ]
        .iter()
        .filter(|v| **v)
        .count();
        let total_fields = 6usize;
        let acceptance_rate = accepted_fields as f32 / total_fields as f32;

        let gated = WorkingMemoryBlock {
            focus,
            open_questions,
            active_hypotheses,
            next_action,
            confidence,
            drift_score,
            updated_at: Some(Utc::now().to_rfc3339()),
        };

        let gated_json = serde_json::to_string(&gated).unwrap_or_default();
        let structure_hash = hash_payload(&json!({
            "focus": gated.focus.is_some(),
            "open_questions": gated.open_questions.len(),
            "active_hypotheses": gated.active_hypotheses.len(),
            "next_action": gated.next_action.is_some(),
        }).to_string());
        let last_text = self
            .db
            .get_key("introspection_last_text")
            .await
            .ok()
            .flatten()
            .unwrap_or_default();
        let similarity = lexical_similarity(&gated_json, &last_text);
        let drift = (1.0 - similarity).max(0.0);
        let now_ts = Utc::now();
        let first_at = self
            .db
            .get_key("introspection_first_at")
            .await
            .ok()
            .flatten()
            .and_then(|v| DateTime::parse_from_rfc3339(&v).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or(now_ts);
        let session_count = self
            .db
            .get_key("introspection_session_count")
            .await
            .ok()
            .flatten()
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(0)
            .saturating_add(1);
        let burn_in_complete = session_count >= 10
            && (now_ts - first_at).num_days() >= 7;

        let _ = self
            .db
            .set_key("introspection_last_text", &gated_json)
            .await;
        let _ = self
            .db
            .set_key("introspection_last_hash", &structure_hash)
            .await;
        let _ = self
            .db
            .set_key("introspection_session_count", &session_count.to_string())
            .await;
        if first_at == now_ts {
            let _ = self
                .db
                .set_key("introspection_first_at", &now_ts.to_rfc3339())
                .await;
        }

        let drift_threshold = settings.introspection_drift_threshold.unwrap_or(0.30);
        if burn_in_complete && drift > drift_threshold {
            let streak = self
                .db
                .get_key("introspection_drift_streak")
                .await
                .ok()
                .flatten()
                .and_then(|v| v.parse::<i64>().ok())
                .unwrap_or(0)
                .saturating_add(1);
            let _ = self
                .db
                .set_key("introspection_drift_streak", &streak.to_string())
                .await;
            if streak >= 3 {
                let next_temp = (temperature - 0.05).max(0.1);
                let _ = self
                    .db
                    .set_key("introspection_temp_override", &format!("{:.2}", next_temp))
                    .await;
                let _ = self
                    .db
                    .set_key("introspection_drift_streak", "0")
                    .await;
            }
        } else if burn_in_complete {
            let _ = self
                .db
                .set_key("introspection_drift_streak", "0")
                .await;
        }

        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            None,
            json!({
                "event": "introspection_reflection",
                "raw_hash": hash_payload(&response.content),
                "gated_hash": hash_payload(&gated_json),
                "structure_hash": structure_hash,
                "acceptance_rate": acceptance_rate,
                "evidence_ids": evidence_ids,
                "similarity": similarity,
                "drift": drift,
                "burn_in_complete": burn_in_complete,
                "session_count": session_count,
            }),
        )
        .await;

        if acceptance_rate <= 0.0 {
            return None;
        }

        Some(gated)
    }

    pub(super) fn make_candidate(
        &self,
        kind: CandidateKind,
        payload: Value,
        source: &str,
        created_at: &mut i64,
    ) -> Candidate {
        *created_at += 1;
        let mut payload = payload;
        ensure_candidate_evidence_fields(&mut payload);
        let mut candidate = Candidate {
            id: Uuid::new_v4().to_string(),
            kind: kind.clone(),
            payload,
            evidence_event_ids: Vec::new(),
            belief_ids: Vec::new(),
            target_scope: None,
            rationale: None,
            expected_outcome: None,
            cost: Some(0),
            urgency: Some(0),
            source: source.to_string(),
            priority_class: priority_class_for(&kind),
            priority_rank: 0,
            created_at: *created_at,
        };
        candidate.refresh_meta();
        candidate
    }

    pub(super) fn candidate_from_value(&self, value: &Value, source: &str, created_at: &mut i64) -> Option<Candidate> {
        let kind_raw = value.get("kind").and_then(|v| v.as_str())?;
        let mut kind = parse_candidate_kind(kind_raw)?;
        let mut payload = value.get("payload").cloned().unwrap_or_else(|| json!({}));
        if let Some(obj) = payload.as_object_mut() {
            if obj.get("evidence_event_ids").is_none() {
                if let Some(root_ids) = value.get("evidence_event_ids") {
                    obj.insert("evidence_event_ids".to_string(), root_ids.clone());
                }
            }
            if obj.get("belief_ids").is_none() {
                if let Some(root_ids) = value.get("belief_ids") {
                    obj.insert("belief_ids".to_string(), root_ids.clone());
                }
            }
        }
        if matches!(kind, CandidateKind::EmitMessage | CandidateKind::FlagForHuman) {
            if let Some(obj) = payload.as_object_mut() {
                if !obj.contains_key("content") {
                    if let Some(message) = obj.get("message").cloned() {
                        obj.insert("content".to_string(), message);
                    }
                }
            }
        }
        if matches!(kind, CandidateKind::EmitMessage) {
            let user_visible = payload
                .get("user_visible")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            if user_visible {
                let content = payload
                    .get("content")
                    .or_else(|| payload.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let is_question = content.ends_with('?')
                    || payload
                        .get("question")
                        .and_then(|v| v.as_str())
                        .map(|text| text.trim().ends_with('?'))
                        .unwrap_or(false);
                let has_evidence = candidate_has_evidence(&payload);
                if !has_evidence && !is_question {
                    if let Some(obj) = payload.as_object_mut() {
                        obj.entry("speculative".to_string())
                            .or_insert(json!(true));
                        obj.entry("speculative_reason".to_string())
                            .or_insert(json!("no_evidence"));
                        let mut question = content.clone();
                        if question.is_empty() {
                            question = "Can you clarify?".to_string();
                        } else if !question.ends_with('?') {
                            question.push('?');
                        }
                        obj.insert("question".to_string(), Value::String(question.clone()));
                        obj.insert("content".to_string(), Value::String(question));
                    }
                    kind = CandidateKind::AskUserQuestion;
                } else if !has_evidence
                    && !is_question
                    && !payload.get("speculative").and_then(|v| v.as_bool()).unwrap_or(false)
                {
                    if let Some(obj) = payload.as_object_mut() {
                        obj.insert("speculative".to_string(), Value::Bool(true));
                        obj.entry("speculative_reason".to_string())
                            .or_insert(json!("no_evidence"));
                    }
                }
            }
        }
        if matches!(kind, CandidateKind::AskUserQuestion) {
            if let Some(obj) = payload.as_object_mut() {
                if !obj.contains_key("question") {
                    if let Some(message) = obj.get("message").cloned().or_else(|| obj.get("content").cloned()) {
                        obj.insert("question".to_string(), message);
                    }
                }
            }
        }
        ensure_candidate_evidence_fields(&mut payload);
        if matches!(kind, CandidateKind::ToolCall) {
            let action_missing = payload
                .get("action_id")
                .and_then(|v| v.as_str())
                .map(|s| s.trim().is_empty())
                .unwrap_or(true);
            if action_missing {
                let action_id = Uuid::new_v4().to_string();
                if let Some(obj) = payload.as_object_mut() {
                    obj.insert("action_id".to_string(), Value::String(action_id));
                } else {
                    payload = json!({ "action_id": action_id });
                }
            }
        }
        if matches!(kind, CandidateKind::RecordSelfClaim) {
            if let Some(obj) = payload.as_object_mut() {
                if !obj.contains_key("claim_text") {
                    if let Some(text) = obj.get("claim").cloned().or_else(|| obj.get("text").cloned()) {
                        obj.insert("claim_text".to_string(), text);
                    }
                }
            }
        }
        if matches!(kind, CandidateKind::UpdateWorkspace) {
            if let Some(obj) = payload.as_object_mut() {
                normalize_workspace_string_field(obj, "goal_thread");
                normalize_workspace_string_field(obj, "current_focus");
                normalize_workspace_string_field(obj, "focus_rationale");
                normalize_json_list(obj, "open_questions");
                normalize_hypotheses_list(obj);
                normalize_json_list(obj, "working_set_topics");
                if let Some(value) = obj.get_mut("goal_stack") {
                    normalize_goal_stack_payload(value);
                }
            }
        }
        let rationale = value.get("rationale").and_then(|v| v.as_str()).map(|s| s.to_string());
        let expected_outcome = value
            .get("expected_outcome")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let cost = value.get("cost").and_then(|v| v.as_i64());
        let urgency = value.get("urgency").and_then(|v| v.as_i64());
        let priority_rank = value
            .get("priority_rank")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        let target_scope = value
            .get("target_scope")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        *created_at += 1;
        let mut candidate = Candidate {
            id: Uuid::new_v4().to_string(),
            kind: kind.clone(),
            payload,
            evidence_event_ids: Vec::new(),
            belief_ids: Vec::new(),
            target_scope,
            rationale,
            expected_outcome,
            cost,
            urgency,
            source: source.to_string(),
            priority_class: priority_class_for(&kind),
            priority_rank,
            created_at: *created_at,
        };
        ensure_meta_cog_payload(&mut candidate);
        candidate.refresh_meta();
        Some(candidate)
    }









    pub(super) fn build_tool_refusal_fallback(
        &self,
        decision: &KernelDecision,
        state: &KernelState,
        settings: &crate::models::Settings,
    ) -> Option<Candidate> {
        let last_user_input = state.last_user_input.as_deref().unwrap_or("");
        let rejected_tool = decision.rejected.iter().find(|rejected_candidate| {
            matches!(rejected_candidate.kind, CandidateKind::ToolCall)
                && (rejected_candidate.reason == "UNKNOWN_TOOL" || rejected_candidate.reason == "TOOL_DISABLED")
                && user_requested_tool_in_input(
                    last_user_input,
                    rejected_candidate.tool_name.as_deref().unwrap_or(""),
                )
        })?;

        let tool_name = rejected_tool.tool_name.clone().unwrap_or_else(|| "that tool".to_string());
        let mut message = if rejected_tool.reason == "UNKNOWN_TOOL" {
            format!("I don't have a tool named \"{}\".", tool_name)
        } else {
            format!("The tool \"{}\" is disabled in settings.", tool_name)
        };
        if rejected_tool.reason == "UNKNOWN_TOOL" {
            let allowed = self.tools.allowed_tool_names(settings);
            if !allowed.is_empty() {
                message.push_str(&format!(" Available tools: {}.", allowed.join(", ")));
            }
        }
        let mut created_at = 0i64;
        Some(self.make_candidate(
            CandidateKind::EmitMessage,
            json!({ "content": message }),
            "tool_refusal_fallback",
            &mut created_at,
        ))
    }

    pub(super) fn apply_loop_detection(
        &self,
        candidates: &[Candidate],
        state: &mut KernelState,
        settings: &crate::models::Settings,
    ) {
        apply_loop_detection_for(candidates, state, settings);
    }

    pub(super) async fn log_loop_breakers(
        &self,
        run_id: Option<&str>,
        trace_id: Option<&str>,
        state: &KernelState,
        prev_ask: bool,
        prev_tool: bool,
    ) {
        if !prev_ask && state.ask_loop_breaker_triggered {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                run_id,
                trace_id,
                json!({
                    "event": "ask_loop_breaker_triggered",
                    "task_id": state.task_id,
                    "task_phase": task_phase_label(&state.task_phase),
                }),
            )
            .await;
        }
        if !prev_tool && state.tool_loop_breaker_triggered {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                run_id,
                trace_id,
                json!({
                    "event": "tool_loop_breaker_triggered",
                    "task_id": state.task_id,
                    "task_phase": task_phase_label(&state.task_phase),
                }),
            )
            .await;
        }
    }

    pub(super) async fn log_meta_cog_event(
        &self,
        state: &mut KernelState,
        run_id: Option<&str>,
        trace_id: Option<&str>,
        payload: Value,
    ) {
        let mut meta_payload = if payload.is_object() {
            payload
        } else {
            json!({ "detail": payload })
        };
        let event_type = meta_payload
            .get("event")
            .and_then(|v| v.as_str())
            .unwrap_or("meta_cog_event")
            .to_string();
        let reason_raw = meta_payload
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("unspecified");
        let normalized_reason = normalize_meta_cog_reason(reason_raw);
        if let Some(obj) = meta_payload.as_object_mut() {
            obj.insert("event".to_string(), json!("meta_cog_event"));
            obj.entry("event_type".to_string())
                .or_insert(json!(event_type.clone()));
            obj.insert("reason".to_string(), json!(normalized_reason.clone()));
            obj.entry("severity".to_string())
                .or_insert(json!("info"));
            obj.entry("schema_version".to_string())
                .or_insert(json!(1));
            let anchor = state
                .workspace_current_focus
                .clone()
                .or_else(|| state.workspace_goal_thread.clone())
                .unwrap_or_else(|| "current_topic".to_string());
            obj.entry("anchor".to_string()).or_insert(json!(anchor));
            obj.entry("turn_index".to_string())
                .or_insert(json!(state.monologue_count));
        }
        state.meta_cog_event_count = state.meta_cog_event_count.saturating_add(1);
        state.last_meta_cog_event = Some(normalized_reason.clone());
        if is_loop_break_reason(&normalized_reason) {
            state.meta_cog_loop_break_count = state.meta_cog_loop_break_count.saturating_add(1);
        }
        state.last_meta_cog_event_at = Some(Utc::now().to_rfc3339());
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            run_id,
            trace_id,
            meta_payload,
        )
        .await;
    }

    pub(super) async fn maybe_emit_meta_cog_outcome(
        &self,
        state: &mut KernelState,
        anchor_label: &str,
        settings: &Settings,
    ) {
        let Some(pending) = state.meta_cog_pending.clone() else {
            return;
        };
        let now = Utc::now();
        let turns_since = state.monologue_count.saturating_sub(pending.turn_index);
        let elapsed_secs = DateTime::parse_from_rfc3339(&pending.accepted_at)
            .map(|dt| now.signed_duration_since(dt.with_timezone(&Utc)).num_seconds())
            .unwrap_or(meta_cog_outcome_timeout_secs(settings) + 1);
        if turns_since < meta_cog_outcome_turns(settings)
            && elapsed_secs < meta_cog_outcome_timeout_secs(settings)
        {
            return;
        }

        let mut outcome = "no_signal";
        let mut reason = "timeout_no_change";
        if let Some(current_focus) = state.workspace_current_focus.as_deref() {
            if !pending.anchor.is_empty() && current_focus != pending.anchor {
                outcome = "resolved";
                reason = "anchor_changed";
            }
        }
        if outcome == "no_signal" && turns_since <= meta_cog_cycle_window_turns(settings) {
            if state.last_meta_cog_loop_break_reason.is_some() {
                outcome = "cycling";
                reason = "loop_break_detected";
            }
        }

        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            None,
            json!({
                "event": "meta_cog_outcome",
                "outcome": outcome,
                "reason": reason,
                "turns_since": turns_since,
                "anchor": anchor_label,
                "pending_kind": pending.kind,
            }),
        )
        .await;

        match outcome {
            "cycling" => {
                state.meta_cog_outcome_cycle_streak += 1;
                state.meta_cog_outcome_no_signal_streak = 0;
            }
            "no_signal" => {
                state.meta_cog_outcome_no_signal_streak += 1;
                state.meta_cog_outcome_cycle_streak = 0;
            }
            _ => {
                state.meta_cog_outcome_cycle_streak = 0;
                state.meta_cog_outcome_no_signal_streak = 0;
            }
        }

        let adaptive_multiplier = adjust_meta_cog_adaptive_multiplier(state, outcome);
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            None,
            json!({
                "event": "meta_cog_adaptive_gate",
                "outcome": outcome,
                "multiplier": adaptive_multiplier,
                "cycle_streak": state.meta_cog_outcome_cycle_streak,
                "no_signal_streak": state.meta_cog_outcome_no_signal_streak,
            }),
        )
        .await;

        let streak_limit = meta_cog_streak_limit(settings);
        let adaptive_cooldown = (meta_cog_cooldown_secs(settings) as f32 * adaptive_multiplier)
            .round()
            .max(1.0) as i64;
        if state.meta_cog_outcome_cycle_streak >= streak_limit {
            state.meta_cog_cooldown_until =
                Some((now + chrono::Duration::seconds(adaptive_cooldown)).to_rfc3339());
            self.log_meta_cog_event(
                state,
                None,
                None,
                json!({
                    "event": "meta_cog_event",
                    "reason": "meta_cog_cycling_cooldown",
                    "severity": "warn",
                    "cooldown_until": state.meta_cog_cooldown_until,
                    "cooldown_secs": adaptive_cooldown,
                }),
            )
            .await;
        }
        if state.meta_cog_outcome_no_signal_streak >= streak_limit {
            state.meta_cog_cooldown_until =
                Some((now + chrono::Duration::seconds(adaptive_cooldown)).to_rfc3339());
            self.log_meta_cog_event(
                state,
                None,
                None,
                json!({
                    "event": "meta_cog_event",
                    "reason": "meta_cog_no_signal_cooldown",
                    "severity": "info",
                    "cooldown_until": state.meta_cog_cooldown_until,
                    "cooldown_secs": adaptive_cooldown,
                }),
            )
            .await;
        }

        state.meta_cog_last_outcome = Some(outcome.to_string());
        state.meta_cog_last_outcome_at = Some(now.to_rfc3339());
        state.meta_cog_pending = None;
    }


    pub(super) async fn run_prediction_generation(
        db: Arc<Db>,
        model_client: Arc<ModelClient>,
        app_handle: AppHandle,
        context: PredictionContext,
        conversation_id: String,
        run_id: Option<String>,
        trace_id: Option<String>,
        settings: Settings,
    ) {
        if let Some(run_id) = run_id.as_deref() {
            let _ = db.touch_run_heartbeat(run_id).await;
            let reasoning_only_count: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM system_logs
                 WHERE run_id = ?
                   AND json_extract(payload, '$.event') = 'response_empty_with_reasoning'",
            )
            .bind(run_id)
            .fetch_optional(&db.pool)
            .await
            .ok()
            .flatten()
            .unwrap_or(0);
            if reasoning_only_count > 0 {
                let _ = system_log::log_event(
                    &db.pool,
                    Some(&app_handle),
                    "info",
                    "kernel",
                    Some(run_id),
                    trace_id.as_deref(),
                    json!({
                        "event": "prediction_generation_skipped",
                        "reason": "reasoning_only_response",
                        "reasoning_only_count": reasoning_only_count,
                    }),
                )
                .await;
                return;
            }
        }

        let parse_error_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs
             WHERE json_extract(payload, '$.event') = 'prediction_generation_rejected'
               AND json_extract(payload, '$.reason') = 'json_parse_error'
               AND datetime(timestamp) >= datetime('now', '-1 hour')",
        )
        .fetch_optional(&db.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0);
        let json_model_id = settings
            .json_reliable_model_id
            .clone()
            .or_else(|| settings.active_model_id.clone())
            .unwrap_or_else(|| "default".to_string());
        let allowlist_limit = if parse_error_count >= 1 { 12 } else { 24 };
        let (allowlist_ids, allowlist_entries) =
            Kernel::load_prediction_allowlist(&db.pool, allowlist_limit, true).await;
        if allowlist_ids.is_empty() {
            let _ = system_log::log_event(
                &db.pool,
                Some(&app_handle),
                "info",
                "kernel",
                run_id.as_deref(),
                trace_id.as_deref(),
                json!({
                    "event": "prediction_generation_skipped",
                    "reason": "no_evidence_allowlist",
                }),
            )
            .await;
            return;
        }
        let compact_allowlist_entries: Vec<Value> = allowlist_entries
            .iter()
            .map(|entry| {
                let id = entry.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
                let source_type = entry
                    .get("source_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let snippet = entry
                    .get("snippet")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let compact_snippet: String = snippet.chars().take(140).collect();
                json!({
                    "id": id,
                    "source_type": source_type,
                    "snippet": compact_snippet,
                })
            })
            .collect();
        let (temperature, max_tokens) = if parse_error_count >= 1 {
            (Some(0.1), Some(300))
        } else {
            (Some(0.2), Some(500))
        };

        let controller_block = context
            .controller_state
            .as_ref()
            .map(|c| json!(c))
            .unwrap_or_else(|| json!({}));
        let packet = json!({
            "controller_state": controller_block,
            "workspace": {
                "current_focus": context.workspace_focus.clone().unwrap_or_default(),
                "goal_thread": context.workspace_goal.clone().unwrap_or_default(),
                "open_questions": context.workspace_open_questions,
            },
            "last_user_input": context.last_user_input.clone().unwrap_or_default(),
            "evidence_allowlist": compact_allowlist_entries,
            "timestamp": Utc::now().to_rfc3339(),
        });
        let compact_packet = json!({
            "workspace": {
                "current_focus": context.workspace_focus.clone().unwrap_or_default(),
                "goal_thread": context.workspace_goal.clone().unwrap_or_default(),
            },
            "last_user_input": context.last_user_input.clone().unwrap_or_default(),
            "evidence_allowlist": compact_allowlist_entries,
            "timestamp": Utc::now().to_rfc3339(),
        });

        let system_prompt = "You generate falsifiable self-predictions. Output ONLY JSON.\n\nRequired schema:\n{\n  \"predictions\": [\n    {\"metric\": string, \"expected_value\": number, \"expected_variance\": number, \"horizon\": \"next_turn\"|\"next_tool\"|\"next_5m\"|\"next_hour\", \"confidence\": 0..1, \"evidence_event_ids\": [number,...]}\n  ] | null,\n  \"rejection_reason\": string | null\n}\n\nRules:\n- Output 1-3 predictions or null.\n- Use ONLY evidence_event_ids from the allowlist.\n- expected_variance must be >= 0.0.\n- Metrics allowed: tool_success_rate, memory_pass_rate, clarification_rate, refusal_rate, workspace_stability_rate, response_len.\n- If no valid evidence, set rejection_reason and return null predictions.";

        let use_compact_packet = parse_error_count >= 1;
        let packet_for_prompt = if use_compact_packet {
            compact_packet.clone()
        } else {
            packet.clone()
        };
        let user_prompt = if use_compact_packet {
            format!("Prediction packet (compact):\n{}", packet_for_prompt.to_string())
        } else {
            format!("Prediction packet:\n{}", packet_for_prompt.to_string())
        };

        let request = ChatCompletionRequest {
            model: json_model_id.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            stream: false,
            temperature,
            top_p: None,
            max_tokens,
            response_format: Some(json!({ "type": "json_object" })),
            tools: None,
            tool_choice: None,
            enable_thinking: None,
            prefill: None,
            skip_injection: Some(true),
            skip_memory: Some(true),
            skip_reminders: Some(true),
            memory_expand: None,
            json_strict: Some(true),
            allow_diagnostics: Some(false),
                skip_sanitization: None,
            run_id: run_id.clone(),
            request_label: Some("prediction_generation".to_string()),
        };

        let mut response = model_client
            .chat(&settings.api_base_url, settings.api_key.as_deref(), &request)
            .await;
        if let Err(err) = &response {
            if err.contains("EMPTY_CONTENT_JSON_MODE") {
                let _ = system_log::log_event(
                    &db.pool,
                    Some(&app_handle),
                    "warn",
                    "kernel",
                    run_id.as_deref(),
                    trace_id.as_deref(),
                    json!({
                        "event": "prediction_generation_retry",
                        "reason": "empty_content_json_mode",
                    }),
                )
                .await;
                let fallback_system_prompt = "Return ONLY JSON using the required schema. Output 1-2 predictions or null.";
                let fallback_user_prompt = format!("Prediction packet (compact):\n{}", compact_packet);
                let retry_request = ChatCompletionRequest {
                    model: json_model_id.clone(),
                    messages: vec![
                        ChatMessage {
                            role: "system".to_string(),
                            content: fallback_system_prompt.to_string(),
                        },
                        ChatMessage {
                            role: "user".to_string(),
                            content: fallback_user_prompt,
                        },
                    ],
                    stream: false,
                    temperature: Some(0.1),
                    top_p: None,
                    max_tokens: Some(300),
                    response_format: Some(json!({ "type": "json_object" })),
                    tools: None,
                    tool_choice: None,
                    enable_thinking: None,
                    prefill: None,
                    skip_injection: Some(true),
                    skip_memory: Some(true),
                    skip_reminders: Some(true),
                    memory_expand: None,
                    json_strict: Some(true),
                    allow_diagnostics: Some(false),
                    skip_sanitization: None,
                    run_id: run_id.clone(),
                    request_label: Some("prediction_generation_retry".to_string()),
                };
                response = model_client
                    .chat(&settings.api_base_url, settings.api_key.as_deref(), &retry_request)
                    .await;
                if let Err(err) = &response {
                    if err.contains("EMPTY_CONTENT_JSON_MODE") {
                        let fallback_user_prompt = format!(
                            "Return ONLY JSON using the required schema.\nPrediction packet (minimal):\n{}",
                            compact_packet
                        );
                        let fallback_request = ChatCompletionRequest {
                            model: json_model_id.clone(),
                            messages: vec![
                                ChatMessage {
                                    role: "system".to_string(),
                                    content: fallback_system_prompt.to_string(),
                                },
                                ChatMessage {
                                    role: "user".to_string(),
                                    content: fallback_user_prompt,
                                },
                            ],
                            stream: false,
                            temperature: Some(0.1),
                            top_p: None,
                            max_tokens: Some(240),
                            response_format: None,
                            tools: None,
                            tool_choice: None,
                            enable_thinking: None,
                            prefill: None,
                            skip_injection: Some(true),
                            skip_memory: Some(true),
                            skip_reminders: Some(true),
                            memory_expand: None,
                            json_strict: Some(false),
                            allow_diagnostics: Some(false),
                            skip_sanitization: None,
                            run_id: run_id.clone(),
                            request_label: Some("prediction_generation_fallback".to_string()),
                        };
                        response = model_client
                            .chat(&settings.api_base_url, settings.api_key.as_deref(), &fallback_request)
                            .await;
                    }
                }
            }
        }

        let (raw_content, _) = match response {
            Ok(payload) => payload,
            Err(err) => {
                let reason = if err.contains("EMPTY_CONTENT_JSON_MODE") {
                    "empty_content_json_mode"
                } else {
                    "model_error"
                };
                let _ = system_log::log_event(
                    &db.pool,
                    Some(&app_handle),
                    "warn",
                    "kernel",
                    run_id.as_deref(),
                    trace_id.as_deref(),
                    json!({
                        "event": "prediction_generation_failed",
                        "reason": reason,
                    }),
                )
                .await;
                return;
            }
        };
        if let Some(run_id) = run_id.as_deref() {
            let _ = db.touch_run_heartbeat(run_id).await;
        }

        let (mut value_opt, mut repaired) = crate::core::model_client::repair_prediction_json(&raw_content);
        if value_opt.is_none() {
            let _ = system_log::log_event(
                &db.pool,
                Some(&app_handle),
                "warn",
                "kernel",
                run_id.as_deref(),
                trace_id.as_deref(),
                json!({
                    "event": "prediction_generation_retry",
                    "reason": "json_parse_error",
                }),
            )
            .await;
            let fallback_system_prompt = "Return ONLY JSON using the required schema. Output 1-2 predictions or null.";
            let fallback_user_prompt = format!(
                "Prediction packet (compact):\n{}",
                compact_packet
            );
            let fallback_request = ChatCompletionRequest {
                model: json_model_id.clone(),
                messages: vec![
                    ChatMessage {
                        role: "system".to_string(),
                        content: fallback_system_prompt.to_string(),
                    },
                    ChatMessage {
                        role: "user".to_string(),
                        content: fallback_user_prompt,
                    },
                ],
                stream: false,
                temperature: Some(0.1),
                top_p: None,
                max_tokens: Some(300),
                response_format: None,
                tools: None,
                tool_choice: None,
                enable_thinking: None,
                prefill: None,
                skip_injection: Some(true),
                skip_memory: Some(true),
                skip_reminders: Some(true),
                memory_expand: None,
                json_strict: Some(false),
                allow_diagnostics: Some(false),
                skip_sanitization: None,
                run_id: run_id.clone(),
                request_label: Some("prediction_generation_parse_retry".to_string()),
            };
            if let Ok((fallback_content, _)) = model_client
                .chat(&settings.api_base_url, settings.api_key.as_deref(), &fallback_request)
                .await
            {
                let (fallback_value, fallback_repaired) =
                    crate::core::model_client::repair_prediction_json(&fallback_content);
                if fallback_value.is_some() {
                    value_opt = fallback_value;
                    repaired = repaired || fallback_repaired;
                }
            }
        }
        if repaired {
            let _ = system_log::log_event(
                &db.pool,
                Some(&app_handle),
                "info",
                "kernel",
                run_id.as_deref(),
                trace_id.as_deref(),
                json!({
                    "event": "prediction_json_repair_used",
                    "raw_len": raw_content.len(),
                    "raw_hash": hash_payload(&raw_content),
                }),
            )
            .await;
        }

        let mut context_value = serde_json::to_value(&context).unwrap_or_else(|_| json!({}));
        if let serde_json::Value::Object(ref mut obj) = context_value {
            obj.insert("repaired".to_string(), json!(repaired));
        }
        let context_ref_json =
            serde_json::to_string(&context_value).unwrap_or_else(|_| "{}".to_string());

        let Some(value) = value_opt else {
            record_prediction_rejection(
                &db.pool,
                run_id.as_deref(),
                trace_id.as_deref(),
                &context_ref_json,
                "json_parse_error",
                None,
            )
            .await;
            let _ = system_log::log_event(
                &db.pool,
                Some(&app_handle),
                "warn",
                "kernel",
                run_id.as_deref(),
                trace_id.as_deref(),
                json!({
                    "event": "prediction_generation_rejected",
                    "reason": "json_parse_error",
                }),
            )
            .await;
            return;
        };

        let parsed: PredictionResponse = match serde_json::from_value(value.clone()) {
            Ok(parsed) => parsed,
            Err(_) => {
                let mut coerced = coerce_prediction_response(&value);
                if coerced.predictions.is_none() && coerced.rejection_reason.is_none() {
                    coerced.rejection_reason = Some("invalid_json".to_string());
                }
                coerced
            }
        };

        let mut inserted = 0usize;
        let mut rejected = 0usize;
        let allowlist_set: HashSet<i64> = allowlist_ids.iter().copied().collect();
        let allowed_metrics = allowed_prediction_metrics();
        let allowed_horizons = allowed_prediction_horizons();

        let mut rejection_reasons: Vec<String> = Vec::new();

        if parsed.predictions.is_none() {
            let reason = parsed
                .rejection_reason
                .as_deref()
                .unwrap_or("empty_predictions")
                .trim();
            if !reason.is_empty() {
                record_prediction_rejection(
                    &db.pool,
                    run_id.as_deref(),
                    trace_id.as_deref(),
                    &context_ref_json,
                    reason,
                    None,
                )
                .await;
                let _ = system_log::log_event(
                    &db.pool,
                    Some(&app_handle),
                    "warn",
                    "kernel",
                    run_id.as_deref(),
                    trace_id.as_deref(),
                    json!({
                        "event": "prediction_generation_rejected",
                        "reason": reason,
                    }),
                )
                .await;
                rejected += 1;
                rejection_reasons.push(reason.to_string());
            }
        }

        if let Some(predictions) = parsed.predictions {
            if predictions.is_empty() {
                record_prediction_rejection(
                    &db.pool,
                    run_id.as_deref(),
                    trace_id.as_deref(),
                    &context_ref_json,
                    "empty_predictions",
                    None,
                )
                .await;
                let _ = system_log::log_event(
                    &db.pool,
                    Some(&app_handle),
                    "warn",
                    "kernel",
                    run_id.as_deref(),
                    trace_id.as_deref(),
                    json!({
                        "event": "prediction_generation_rejected",
                        "reason": "empty_predictions",
                    }),
                )
                .await;
                rejected += 1;
                rejection_reasons.push("empty_predictions".to_string());
            }
            for prediction in predictions {
                let metric = prediction.metric.trim().to_lowercase();
                let horizon = prediction.horizon.trim().to_lowercase();
                let mut rejection =
                    validate_prediction_basics(&prediction, &allowed_metrics, &allowed_horizons);

                let evidence_event_ids = prediction.evidence_event_ids.unwrap_or_default();
                let evidence_event_ids = normalize_id_list(&evidence_event_ids);
                if rejection.is_none() {
                    if evidence_event_ids.is_empty()
                        || !evidence_event_ids.iter().all(|id| allowlist_set.contains(id))
                    {
                        rejection = Some("invalid_evidence_ids".to_string());
                    }
                }
                if rejection.is_none() {
                    let validation =
                        validate_evidence_ids_with_pool(&db.pool, &evidence_event_ids, &[], true).await;
                    if !validation.evidence_ok() {
                        rejection = Some("evidence_validation_failed".to_string());
                    }
                }

                if let Some(reason) = rejection {
                    record_prediction_rejection(
                        &db.pool,
                        run_id.as_deref(),
                        trace_id.as_deref(),
                        &context_ref_json,
                        &reason,
                        Some(&metric),
                    )
                    .await;
                    let _ = system_log::log_event(
                        &db.pool,
                        Some(&app_handle),
                        "warn",
                        "kernel",
                        run_id.as_deref(),
                        trace_id.as_deref(),
                        json!({
                            "event": "prediction_generation_rejected",
                            "reason": reason,
                            "metric": metric,
                            "horizon": horizon,
                        }),
                    )
                    .await;
                    rejected += 1;
                    rejection_reasons.push(reason);
                    continue;
                }

                let expected_variance = prediction.expected_variance.unwrap_or(0.05).max(0.0);
                let confidence = prediction.confidence.unwrap_or(0.5).clamp(0.0, 1.0);
                let prediction_id = Uuid::new_v4().to_string();
                let evidence_json =
                    serde_json::to_string(&evidence_event_ids).unwrap_or_else(|_| "[]".to_string());
                let predicted_target_type = if metric == "response_len" {
                    "count"
                } else {
                    "rate"
                };
                let bound_span = expected_variance.max(1e-6).sqrt() * 2.0;
                let (min_bound, max_bound) = if metric == "response_len" {
                    (0.0, (prediction.expected_value + bound_span).max(0.0))
                } else {
                    (
                        (prediction.expected_value - bound_span).max(0.0),
                        (prediction.expected_value + bound_span).min(1.0),
                    )
                };
                let expected_bounds_json = json!({
                    "min": min_bound,
                    "max": max_bound,
                })
                .to_string();
                let normalization_contract_id = format!("metric:{}", metric);
                let salience_hint = confidence;

                if sqlx::query(
                    "INSERT INTO self_predictions
                     (id, run_id, trace_id, metric, context_ref_json, predicted_target_type, expected_value, expected_variance, expected_bounds_json, horizon, confidence, evidence_event_ids, linked_claims_json, normalization_contract_id, salience_hint, rejection_reason, created_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                )
                .bind(&prediction_id)
                .bind(run_id.as_deref())
                .bind(trace_id.as_deref())
                .bind(&metric)
                .bind(&context_ref_json)
                .bind(predicted_target_type)
                .bind(prediction.expected_value)
                .bind(expected_variance)
                .bind(expected_bounds_json)
                .bind(&horizon)
                .bind(confidence)
                .bind(evidence_json)
                .bind("[]")
                .bind(normalization_contract_id)
                .bind(salience_hint)
                .bind::<Option<&str>>(None)
                .execute(&db.pool)
                .await
                .is_ok()
                {
                    inserted += 1;
                } else {
                    rejected += 1;
                    rejection_reasons.push("insert_failed".to_string());
                }
            }
        }

        let _ = system_log::log_event(
            &db.pool,
            Some(&app_handle),
            "info",
            "kernel",
            run_id.as_deref(),
            trace_id.as_deref(),
            json!({
                "event": "prediction_generation_complete",
                "inserted": inserted,
                "rejected": rejected,
                "conversation_id": conversation_id,
                "model_rejection_reason": parsed.rejection_reason,
                "rejection_reasons": rejection_reasons,
            }),
        )
        .await;
    }

    pub(super) async fn load_prediction_allowlist(
        pool: &SqlitePool,
        limit: i64,
        allow_self: bool,
    ) -> (Vec<i64>, Vec<Value>) {
        let mut ids = Vec::new();
        let mut entries = Vec::new();

        let rows = sqlx::query(
            "SELECT id, source_type, snippet, created_at FROM ics_evidence_events
             ORDER BY datetime(created_at) DESC LIMIT ?",
        )
        .bind(limit)
        .fetch_all(pool)
        .await
        .unwrap_or_default();
        for row in rows {
            let id: i64 = row.get("id");
            let source_type: String = row.try_get("source_type").unwrap_or_default();
            let snippet: Option<String> = row.try_get("snippet").ok();
            let created_at: Option<String> = row.try_get("created_at").ok();
            ids.push(id);
            entries.push(json!({
                "id": id,
                "source_type": source_type,
                "snippet": snippet.unwrap_or_default(),
                "created_at": created_at.unwrap_or_default(),
            }));
        }

        if allow_self {
            let rows = sqlx::query(
                "SELECT id, source_type, snippet, created_at FROM self_evidence_events
                 ORDER BY datetime(created_at) DESC LIMIT ?",
            )
            .bind(limit)
            .fetch_all(pool)
            .await
            .unwrap_or_default();
            for row in rows {
                let id: i64 = row.get("id");
                let source_type: String = row.try_get("source_type").unwrap_or_default();
                let snippet: Option<String> = row.try_get("snippet").ok();
                let created_at: Option<String> = row.try_get("created_at").ok();
                ids.push(id);
                entries.push(json!({
                    "id": id,
                    "source_type": source_type,
                    "snippet": snippet.unwrap_or_default(),
                    "created_at": created_at.unwrap_or_default(),
                }));
            }
        }

        (normalize_id_list(&ids), entries)
    }

    pub(super) async fn insert_episodic_event_tx<'c>(
        &self,
        tx: &mut sqlx::Transaction<'c, sqlx::Sqlite>,
        write: &EpisodicWrite,
        conversation_id: &str,
        run_id: Option<&str>,
        trace_id: Option<&str>,
    ) -> Result<(), String> {
        const EPISODIC_SCHEMA_VERSION: i32 = 1;
        const EPISODIC_EVENT_VERSION: i32 = 1;
        const SUMMARY_SNIPPET_MAX: usize = 512;
        const ALLOWED_KEYS: [&str; 19] = [
            "status",
            "tool_name",
            "error_code",
            "summary_snippet",
            "belief_id",
            "entity_id",
            "scope",
            "polarity",
            "source_ref",
            "rel_type",
            "roles_seen",
            "direction",
            "sample_dsl",
            "session_id",
            "timestamp",
            "decision_reason",
            "conflict_topic_key",
            "conflict_reason",
            "claim_id",
        ];

        let mut cleaned = serde_json::Map::new();
        if let Some(obj) = write.payload.as_object() {
            for key in ALLOWED_KEYS {
                if let Some(value) = obj.get(key) {
                    let cleaned_value = if key == "summary_snippet" {
                        value.as_str().map(|raw| {
                            let sanitized = crate::core::memory::snippets::sanitize_episodic_text(raw);
                            let trimmed: String = sanitized.chars().take(SUMMARY_SNIPPET_MAX).collect();
                            Value::String(trimmed)
                        })
                    } else if value.is_string() || value.is_number() || value.is_boolean() {
                        Some(value.clone())
                    } else {
                        value.as_str().map(|s| Value::String(s.to_string()))
                    };
                    if let Some(v) = cleaned_value {
                        cleaned.insert(key.to_string(), v);
                    }
                }
            }
        }
        let mut payload_value = Value::Object(cleaned);
        if !phi_consent_allowed(&self.db.pool, Some(conversation_id)).await {
            let (redacted, sensitivity) = redact_sensitive_json(&payload_value);
            if let Some(level) = sensitivity {
                payload_value = redacted;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "memory",
                    run_id,
                    trace_id,
                    json!({
                        "event": "phi_redacted",
                        "scope": "episodic_event",
                        "sensitivity": level.as_str(),
                        "conversation_id": conversation_id,
                    }),
                )
                .await;
            }
        }
        let scope_value = payload_value
            .get("scope")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let payload_json = serde_json::to_string(&payload_value).unwrap_or_else(|_| "{}".to_string());
        let event_id = Uuid::new_v4().to_string();

        sqlx::query(
            "INSERT INTO episodic_events (id, schema_version, event_version, event_type, payload_json, timestamp, run_id, trace_id, conversation_id, scope, source_type, source_ref, linked_belief_id, linked_artifact_id)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&event_id)
        .bind(EPISODIC_SCHEMA_VERSION)
        .bind(EPISODIC_EVENT_VERSION)
        .bind(&write.event_type)
        .bind(payload_json)
        .bind(Utc::now().to_rfc3339())
        .bind(run_id)
        .bind(trace_id)
        .bind(conversation_id)
        .bind(scope_value.clone())
        .bind(&write.source_type)
        .bind(write.source_ref.clone())
        .bind::<Option<i64>>(None)
        .bind::<Option<String>>(None)
        .execute(&mut **tx)
        .await
        .map_err(|e| e.to_string())?;
        crate::core::episodic::insert_identity_index_tx(
            tx,
            &event_id,
            &write.event_type,
            &payload_value,
            scope_value.as_deref(),
        )
        .await;
        Ok(())
    }

    pub(super) async fn set_semantic_core_tx<'c>(
        &self,
        tx: &mut sqlx::Transaction<'c, sqlx::Sqlite>,
        summary: &str,
    ) -> Result<(), String> {
        let summary = summary.trim();
        if summary.is_empty() {
            return Ok(());
        }

        let mut content = summary.to_string();
        if content.len() > SEMANTIC_CORE_MAX_CHARS {
            content = content.chars().take(SEMANTIC_CORE_MAX_CHARS).collect();
        }

        sqlx::query(
            "UPDATE semantic_core SET content = ?, updated_at = CURRENT_TIMESTAMP, version = version + 1 WHERE id = 1"
        )
        .bind(&content)
        .execute(&mut **tx)
        .await
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub(super) async fn insert_thread_run_tx<'c>(
        &self,
        tx: &mut sqlx::Transaction<'c, sqlx::Sqlite>,
        thread_id: &str,
        conversation_id: &str,
        parent_run_id: Option<&str>,
        goal: &str,
        context_json: &str,
        depth: i64,
    ) -> Result<(), String> {
        sqlx::query(
            "INSERT INTO thread_runs (id, conversation_id, parent_run_id, goal, context_json, status, depth, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, 'running', ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
        .bind(thread_id)
        .bind(conversation_id)
        .bind(parent_run_id)
        .bind(goal)
        .bind(context_json)
        .bind(depth)
        .execute(&mut **tx)
        .await
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub(super) async fn record_tool_dispatch_tx<'c>(
        &self,
        tx: &mut sqlx::Transaction<'c, sqlx::Sqlite>,
        call: &ToolDispatchRequest,
        run_id: Option<&str>,
    ) -> Result<(), String> {
        sqlx::query(
            "INSERT INTO tool_dispatches (action_id, run_id, tool_name, args_json, plan_step_id, status, attempts, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, 'pending', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
             ON CONFLICT(action_id) DO UPDATE SET status = CASE WHEN tool_dispatches.status = 'success' THEN 'success' ELSE 'pending' END,
                 plan_step_id = COALESCE(tool_dispatches.plan_step_id, excluded.plan_step_id),
                 updated_at = CURRENT_TIMESTAMP"
        )
        .bind(&call.action_id)
        .bind(run_id)
        .bind(&call.tool_name)
        .bind(&call.args_json)
        .bind(&call.plan_step_id)
        .execute(&mut **tx)
        .await
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub(crate) async fn tool_failure_detected_for_run(&self, run_id: &str) -> bool {
        if run_id.trim().is_empty() {
            return false;
        }
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM tool_dispatches
             WHERE run_id = ? AND status = 'failed'",
        )
        .bind(run_id)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0);
        count > 0
    }

    async fn tool_failure_detected_in_window(&self, window_mins: i64, tool_names: &[String]) -> bool {
        if window_mins <= 0 || tool_names.is_empty() {
            return false;
        }
        let window = format!("-{} minutes", window_mins);
        let rows = sqlx::query(
            "SELECT tool_name FROM tool_dispatches
             WHERE status = 'failed'
               AND datetime(updated_at) >= datetime('now', ?)",
        )
        .bind(window)
        .fetch_all(&self.db.pool)
        .await
        .unwrap_or_default();
        if rows.is_empty() {
            return false;
        }
        let filter: HashSet<String> = tool_names
            .iter()
            .map(|name| name.to_lowercase())
            .collect();
        for row in rows {
            let tool_name: String = row.get("tool_name");
            if filter.contains(&tool_name.to_lowercase()) {
                return true;
            }
        }
        false
    }

    pub(super) async fn mark_stale_tool_dispatches(&self) -> usize {
        let rows = sqlx::query(
            "SELECT action_id, tool_name, run_id, attempts, updated_at
             FROM tool_dispatches
             WHERE status = 'pending'",
        )
        .fetch_all(&self.db.pool)
        .await
        .unwrap_or_default();
        if rows.is_empty() {
            return 0;
        }
        use sqlx::Row;
        let now = Utc::now();
        let mut stale_count = 0usize;
        for row in rows.iter() {
            let action_id: String = row.get("action_id");
            let tool_name: String = row.get("tool_name");
            let run_id: Option<String> = row.try_get("run_id").ok();
            let attempts: i64 = row.try_get("attempts").unwrap_or(0);
            let updated_at: String = row.try_get("updated_at").unwrap_or_default();
            let timeout_s = self.tools.timeout_for(&tool_name) as i64;
            let is_stale = timestamp_from_str(&updated_at)
                .map(|ts| now.signed_duration_since(ts).num_seconds() > timeout_s)
                .unwrap_or(false);
            if !is_stale {
                continue;
            }
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "tool",
                run_id.as_deref(),
                None,
                json!({
                    "event": "tool_dispatch_stale",
                    "reason": "timeout",
                    "tool": tool_name,
                    "action_id": action_id,
                    "attempts": attempts,
                    "last_update": updated_at,
                    "timeout_s": timeout_s,
                }),
            )
            .await;
            let _ = sqlx::query(
                "UPDATE tool_dispatches
                 SET status = 'failed',
                     last_error = 'timeout',
                     failure_kind = ?,
                     updated_at = CURRENT_TIMESTAMP
                 WHERE action_id = ?",
            )
            .bind(TOOL_FAILURE_KIND_EXECUTION)
            .bind(&action_id)
            .execute(&self.db.pool)
            .await;
            stale_count += 1;
        }
        stale_count
    }

    pub(super) async fn maybe_log_tool_baseline_snapshot(&self) {
        let now = Utc::now();
        let last_snapshot = self
            .db
            .get_key("tool_baseline_snapshot_last_at")
            .await
            .ok()
            .flatten()
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(&ts).ok())
            .map(|ts| ts.with_timezone(&Utc));
        if let Some(last) = last_snapshot {
            if now.signed_duration_since(last).num_hours() < 12 {
                return;
            }
        }

        let mut dispatch_summary = Vec::new();
        let dispatch_rows = sqlx::query(
            "SELECT tool_name, status, failure_kind, COUNT(*) as count
             FROM tool_dispatches
             GROUP BY tool_name, status, failure_kind",
        )
        .fetch_all(&self.db.pool)
        .await
        .unwrap_or_default();
        for row in dispatch_rows {
            let tool_name: String = row.get("tool_name");
            let status: String = row.get("status");
            let failure_kind: Option<String> = row.try_get("failure_kind").ok();
            let count: i64 = row.get("count");
            dispatch_summary.push(json!({
                "tool": tool_name,
                "status": status,
                "failure_kind": failure_kind,
                "count": count,
            }));
        }

        let mut rejection_summary = Vec::new();
        let rejection_rows = sqlx::query(
            "SELECT json_extract(payload, '$.reason') as reason,
                    json_extract(payload, '$.tool_name') as tool_name,
                    COUNT(*) as count
             FROM system_logs
             WHERE json_extract(payload, '$.event') = 'tool_candidate_rejected'
             GROUP BY reason, tool_name",
        )
        .fetch_all(&self.db.pool)
        .await
        .unwrap_or_default();
        for row in rejection_rows {
            let reason: Option<String> = row.try_get("reason").ok();
            let tool_name: Option<String> = row.try_get("tool_name").ok();
            let count: i64 = row.get("count");
            rejection_summary.push(json!({
                "reason": reason,
                "tool": tool_name,
                "count": count,
            }));
        }

        let mut gate_blocked_summary = Vec::new();
        let gate_rows = sqlx::query(
            "SELECT json_extract(payload, '$.tool') as tool,
                    json_extract(payload, '$.reason') as reason,
                    COUNT(*) as count
             FROM system_logs
             WHERE json_extract(payload, '$.event') = 'tool_gate_blocked'
             GROUP BY tool, reason",
        )
        .fetch_all(&self.db.pool)
        .await
        .unwrap_or_default();
        for row in gate_rows {
            let tool: Option<String> = row.try_get("tool").ok();
            let reason: Option<String> = row.try_get("reason").ok();
            let count: i64 = row.get("count");
            gate_blocked_summary.push(json!({
                "tool": tool,
                "reason": reason,
                "count": count,
            }));
        }

        let mut stale_summary = Vec::new();
        let stale_rows = sqlx::query(
            "SELECT json_extract(payload, '$.tool') as tool,
                    COUNT(*) as count
             FROM system_logs
             WHERE json_extract(payload, '$.event') = 'tool_dispatch_stale'
             GROUP BY tool",
        )
        .fetch_all(&self.db.pool)
        .await
        .unwrap_or_default();
        for row in stale_rows {
            let tool: Option<String> = row.try_get("tool").ok();
            let count: i64 = row.get("count");
            stale_summary.push(json!({
                "tool": tool,
                "count": count,
            }));
        }

        let mut throttle_summary = Vec::new();
        let throttle_rows = sqlx::query(
            "SELECT json_extract(payload, '$.status') as status,
                    json_extract(payload, '$.tool') as tool,
                    COUNT(*) as count
             FROM system_logs
             WHERE json_extract(payload, '$.event') = 'tool_throttle_defer'
             GROUP BY status, tool",
        )
        .fetch_all(&self.db.pool)
        .await
        .unwrap_or_default();
        for row in throttle_rows {
            let status: Option<String> = row.try_get("status").ok();
            let tool: Option<String> = row.try_get("tool").ok();
            let count: i64 = row.get("count");
            throttle_summary.push(json!({
                "status": status,
                "tool": tool,
                "count": count,
            }));
        }

        let control_mode: Option<String> = sqlx::query_scalar(
            "SELECT mode FROM system_controls WHERE subsystem_id = ?",
        )
        .bind("tool_execution")
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten();

        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "tool",
            None,
            None,
            json!({
                "event": "tool_baseline_snapshot",
                "tool_execution_mode": control_mode,
                "dispatches": dispatch_summary,
                "tool_candidate_rejected": rejection_summary,
                "tool_gate_blocked": gate_blocked_summary,
                "tool_dispatch_stale": stale_summary,
                "tool_throttle_defer": throttle_summary,
            }),
        )
        .await;

        let _ = self
            .db
            .set_key("tool_baseline_snapshot_last_at", &now.to_rfc3339())
            .await;
    }

    async fn tool_contract_gate(
        &self,
        tool_name: &str,
        run_id: Option<&str>,
        trace_id: Option<&str>,
    ) -> ToolGateDecision {
        if ToolRegistry::is_context_only_tool(tool_name) {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "tool",
                run_id,
                trace_id,
                json!({
                    "event": "tool_contract_gate_passed",
                    "tool": tool_name,
                    "reason": "context_only",
                }),
            )
            .await;
            return ToolGateDecision::allow();
        }
        let capability = ToolRegistry::capability_for(tool_name);
        let controller_state = match self.db.get_controller_state().await {
            Ok(Some(state)) => state,
            _ => return ToolGateDecision::allow(),
        };
        let gate = evaluate_gates(&controller_state);

        if gate.throttle_tools && !capability.allow_degraded {
            return ToolGateDecision::block(
                "TOOL_THROTTLED",
                Some("controller_gate".to_string()),
                Some(TOOL_FAILURE_KIND_PLANNING),
            );
        }
        if controller_state.autonomy_level < capability.min_autonomy {
            return ToolGateDecision::block(
                "LOW_AUTONOMY",
                Some(format!("autonomy_level={:.2}", controller_state.autonomy_level)),
                Some(TOOL_FAILURE_KIND_PLANNING),
            );
        }
        if capability.require_telemetry && controller_state.telemetry_coverage < TELEMETRY_MIN {
            return ToolGateDecision::block(
                "LOW_TELEMETRY",
                Some(format!("telemetry_coverage={:.2}", controller_state.telemetry_coverage)),
                Some(TOOL_FAILURE_KIND_PLANNING),
            );
        }
        if capability.require_evidence && controller_state.evidence_coverage < EVIDENCE_MIN {
            return ToolGateDecision::block(
                "LOW_EVIDENCE_COVERAGE",
                Some(format!("evidence_coverage={:.2}", controller_state.evidence_coverage)),
                Some(TOOL_FAILURE_KIND_PLANNING),
            );
        }
        if controller_state.verification_needed && capability.risk >= 0.7 {
            return ToolGateDecision::block(
                "VERIFICATION_REQUIRED",
                Some("controller_state".to_string()),
                Some(TOOL_FAILURE_KIND_PLANNING),
            );
        }
        if controller_state.failure_streak >= 3 && capability.risk >= 0.5 {
            return ToolGateDecision::block(
                "FAILURE_STREAK",
                Some(format!("failure_streak={}", controller_state.failure_streak)),
                Some(TOOL_FAILURE_KIND_PLANNING),
            );
        }

        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "tool",
            run_id,
            trace_id,
            json!({
                "event": "tool_contract_gate_passed",
                "tool": tool_name,
                "autonomy_level": controller_state.autonomy_level,
                "telemetry_coverage": controller_state.telemetry_coverage,
                "evidence_coverage": controller_state.evidence_coverage,
            }),
        )
        .await;

        ToolGateDecision::allow()
    }

    async fn record_tool_output_evidence(
        &self,
        dispatch: &ToolDispatchRequest,
        run_id: Option<&str>,
        trace_id: Option<&str>,
        output: &str,
    ) -> Option<i64> {
        let conversation_id = if let Some(run_id) = run_id {
            sqlx::query_scalar::<_, String>(
                "SELECT conversation_id FROM runs WHERE run_id = ?",
            )
            .bind(run_id)
            .fetch_optional(&self.db.pool)
            .await
            .ok()
            .flatten()
            .unwrap_or_else(|| "default".to_string())
        } else {
            "default".to_string()
        };

        let mut snippet = output.trim().chars().take(320).collect::<String>();
        let (redacted, sensitivity) = redact_sensitive_text(&snippet);
        snippet = redacted;

        if let Some(level) = sensitivity {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "tool",
                run_id,
                trace_id,
                json!({
                    "event": "tool_output_redacted",
                    "tool": dispatch.tool_name,
                    "action_id": dispatch.action_id,
                    "sensitivity": level.as_str(),
                }),
            )
            .await;
        }

        let evidence_event_id = self
            .db
            .create_tool_output_evidence_event(
                &conversation_id,
                &dispatch.tool_name,
                output,
                &snippet,
            )
            .await?;
        let _ = self
            .db
            .link_evidence_to_tool_dispatch(evidence_event_id, &dispatch.action_id)
            .await;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "tool",
            run_id,
            trace_id,
            json!({
                "event": "tool_output_evidence_recorded",
                "tool": dispatch.tool_name,
                "action_id": dispatch.action_id,
                "evidence_event_id": evidence_event_id,
            }),
        )
        .await;

        Some(evidence_event_id)
    }

    pub(super) async fn dispatch_tool(
        &self,
        dispatch: &ToolDispatchRequest,
        run_id: Option<&str>,
        trace_id: Option<&str>,
        cancel_rx: &mut watch::Receiver<bool>,
    ) -> Option<ToolExecutionResult> {
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO tool_dispatches (action_id, run_id, tool_name, args_json, plan_step_id, status, attempts, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, 'pending', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        )
        .bind(&dispatch.action_id)
        .bind(run_id)
        .bind(&dispatch.tool_name)
        .bind(&dispatch.args_json)
        .bind(&dispatch.plan_step_id)
        .execute(&self.db.pool)
        .await;

        if let Ok(row) = sqlx::query("SELECT status, attempts, result_text, last_error FROM tool_dispatches WHERE action_id = ?")
            .bind(&dispatch.action_id)
            .fetch_optional(&self.db.pool)
            .await
        {
            if let Some(row) = row {
                let status: String = row.get("status");
                if status == "success" {
                    let output = row
                        .try_get::<Option<String>, _>("result_text")
                        .ok()
                        .flatten()
                        .unwrap_or_default();
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "tool",
                        run_id,
                        trace_id,
                        json!({
                            "event": "dispatch_skipped",
                            "reason": "already_success",
                            "tool": dispatch.tool_name,
                            "action_id": dispatch.action_id,
                        }),
                    )
                    .await;
                    return Some(ToolExecutionResult {
                        tool_name: dispatch.tool_name.clone(),
                        output,
                        is_error: false,
                    });
                }
            }
        }

        let raw_args = dispatch.args_json.clone();
        let mut args_json = raw_args.clone();

        if let Ok(settings) = self.db.get_settings().await {
            let validate_args = settings.enable_tool_schema_validation.unwrap_or(true);
            let mut normalization = tool_args::normalize_tool_args(&args_json);
            if let Ok(ref norm) = normalization {
                args_json = norm.normalized_json.clone();
            } else if let Some(fallback) = local_tool_args_fallback(&dispatch.tool_name, &raw_args) {
                args_json = fallback;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "tool",
                    run_id,
                    trace_id,
                    json!( {
                        "event": "tool_args_fallback",
                        "tool": dispatch.tool_name,
                        "action_id": dispatch.action_id,
                        "method": "local",
                    }),
                )
                .await;
                normalization = tool_args::normalize_tool_args(&args_json);
                if let Ok(ref norm) = normalization {
                    args_json = norm.normalized_json.clone();
                }
            }

              let gate = self.tool_gate_decision(
                  &dispatch.tool_name,
                  &args_json,
                  &settings,
                  false,
              );
              if !gate.allowed {
                let reason = gate.reason.unwrap_or_else(|| "TOOL_GATE_BLOCK".to_string());
                let detail = gate.detail.unwrap_or_default();
                let failure_kind = gate
                    .failure_kind
                    .unwrap_or_else(|| TOOL_FAILURE_KIND_PLANNING.to_string());
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "tool",
                    run_id,
                    trace_id,
                    json!({
                        "event": "tool_gate_blocked",
                        "tool": dispatch.tool_name,
                        "action_id": dispatch.action_id,
                        "reason": reason,
                        "detail": detail,
                    }),
                )
                .await;
                if reason == "UNKNOWN_TOOL" {
                    let _ = system_log::log_contract_violation(
                        &self.db.pool,
                        Some(&self.app_handle),
                        run_id,
                        trace_id,
                        "C2",
                        "unknown_tool",
                        Some(json!({
                            "tool_name": dispatch.tool_name,
                            "source": "dispatch",
                        })),
                    )
                    .await;
                }
                let error_text = if detail.is_empty() {
                    reason.clone()
                } else {
                    format!("{}: {}", reason, detail)
                };
                let _ = sqlx::query(
                    "UPDATE tool_dispatches
                     SET status = 'failed',
                         attempts = attempts + 1,
                         last_error = ?,
                         failure_kind = ?,
                         args_json = ?,
                         updated_at = CURRENT_TIMESTAMP
                     WHERE action_id = ?",
                )
                .bind(&error_text)
                .bind(&failure_kind)
                .bind(&args_json)
                .bind(&dispatch.action_id)
                .execute(&self.db.pool)
                .await;
                return Some(ToolExecutionResult {
                    tool_name: dispatch.tool_name.clone(),
                    output: error_text,
                    is_error: true,
                });
            }

            let mut repair_attempts = 0;
            let mut repaired_via_model = false;
            let mut last_error_reason: Option<String> = None;
            let mut local_fallback_used = false;

            loop {
                let mut needs_repair = false;
                match normalization {
                    Ok(ref norm) => {
                        if validate_args {
                            if let Err(err) =
                                tool_args::validate_tool_args(&dispatch.tool_name, &norm.value)
                            {
                                last_error_reason = Some(err.reason());
                                needs_repair = true;
                            }
                        }
                    }
                    Err(ref err) => {
                        last_error_reason = Some(err.reason());
                        needs_repair = true;
                    }
                }

                if !needs_repair {
                    break;
                }

                if repair_attempts >= 2 {
                    break;
                }

                if !local_fallback_used {
                    if let Some(fallback) = local_tool_args_fallback(&dispatch.tool_name, &raw_args) {
                        local_fallback_used = true;
                        args_json = fallback;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "tool",
                            run_id,
                            trace_id,
                            json!( {
                                "event": "tool_args_fallback",
                                "tool": dispatch.tool_name,
                                "action_id": dispatch.action_id,
                                "method": "local_retry",
                            }),
                        )
                        .await;
                        normalization = tool_args::normalize_tool_args(&args_json);
                        if let Ok(ref norm) = normalization {
                            args_json = norm.normalized_json.clone();
                        }
                        continue;
                    }
                }

                repair_attempts += 1;
                if let Some(repaired_args) = self
                    .repair_tool_args(
                        &settings,
                        &dispatch.tool_name,
                        &args_json,
                        last_error_reason.as_deref().unwrap_or("invalid_args"),
                        run_id,
                        trace_id,
                    )
                    .await
                {
                    repaired_via_model = true;
                    args_json = repaired_args;
                    normalization = tool_args::normalize_tool_args(&args_json);
                    if let Ok(ref norm) = normalization {
                        args_json = norm.normalized_json.clone();
                    }
                    continue;
                }
                break;
            }

            if let Err(err) = normalization {
                let reason = err.reason();
                let error_text = format!("TOOL_ARGS_INVALID: {}", reason);
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "tool",
                    run_id,
                    trace_id,
                    json!({
                        "event": "tool_gate_blocked",
                        "tool": dispatch.tool_name,
                        "action_id": dispatch.action_id,
                        "reason": "TOOL_ARGS_INVALID",
                        "detail": reason,
                    }),
                )
                .await;
                if repair_attempts > 0 {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "tool",
                        run_id,
                        trace_id,
                        json!({
                            "event": "tool_args_repair_failed",
                            "tool": dispatch.tool_name,
                            "action_id": dispatch.action_id,
                            "reason": reason,
                            "attempts": repair_attempts,
                        }),
                    )
                    .await;
                }
                self.enqueue_tool_args_clarifier(&dispatch.tool_name, &reason, run_id)
                    .await;
                let _ = sqlx::query(
                    "UPDATE tool_dispatches
                     SET status = 'failed',
                         attempts = attempts + 1,
                         last_error = ?,
                         failure_kind = ?,
                         args_json = ?,
                         updated_at = CURRENT_TIMESTAMP
                     WHERE action_id = ?",
                )
                .bind(&error_text)
                .bind(TOOL_FAILURE_KIND_PLANNING)
                .bind(&args_json)
                .bind(&dispatch.action_id)
                .execute(&self.db.pool)
                .await;
                return Some(ToolExecutionResult {
                    tool_name: dispatch.tool_name.clone(),
                    output: error_text,
                    is_error: true,
                });
            }

            if validate_args {
                if let Ok(ref norm) = normalization {
                    if let Err(err) =
                        tool_args::validate_tool_args(&dispatch.tool_name, &norm.value)
                    {
                        let reason = err.reason();
                        let error_text = format!("TOOL_ARGS_INVALID: {}", reason);
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "tool",
                            run_id,
                            trace_id,
                            json!({
                                "event": "tool_gate_blocked",
                                "tool": dispatch.tool_name,
                                "action_id": dispatch.action_id,
                                "reason": "TOOL_ARGS_INVALID",
                                "detail": reason,
                            }),
                        )
                        .await;
                        if repair_attempts > 0 {
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "warn",
                                "tool",
                                run_id,
                                trace_id,
                                json!({
                                    "event": "tool_args_repair_failed",
                                    "tool": dispatch.tool_name,
                                    "action_id": dispatch.action_id,
                                    "reason": reason,
                                    "attempts": repair_attempts,
                                }),
                            )
                            .await;
                        }
                        self.enqueue_tool_args_clarifier(&dispatch.tool_name, &reason, run_id)
                            .await;
                        let _ = sqlx::query(
                            "UPDATE tool_dispatches
                             SET status = 'failed',
                                 attempts = attempts + 1,
                                 last_error = ?,
                                 failure_kind = ?,
                                 args_json = ?,
                                 updated_at = CURRENT_TIMESTAMP
                             WHERE action_id = ?",
                        )
                        .bind(&error_text)
                        .bind(TOOL_FAILURE_KIND_PLANNING)
                        .bind(&args_json)
                        .bind(&dispatch.action_id)
                        .execute(&self.db.pool)
                        .await;
                  return Some(ToolExecutionResult {
                      tool_name: dispatch.tool_name.clone(),
                      output: error_text,
                      is_error: true,
                  });
              }

              let contract_gate = self
                  .tool_contract_gate(&dispatch.tool_name, run_id, trace_id)
                  .await;
              if !contract_gate.allowed {
                  let reason = contract_gate.reason.unwrap_or_else(|| "TOOL_CONTRACT_BLOCK".to_string());
                  let detail = contract_gate.detail.unwrap_or_default();
                  let failure_kind = contract_gate
                      .failure_kind
                      .unwrap_or_else(|| TOOL_FAILURE_KIND_PLANNING.to_string());
                  let _ = system_log::log_event(
                      &self.db.pool,
                      Some(&self.app_handle),
                      "warn",
                      "tool",
                      run_id,
                      trace_id,
                      json!({
                          "event": "tool_contract_blocked",
                          "tool": dispatch.tool_name,
                          "action_id": dispatch.action_id,
                          "reason": reason,
                          "detail": detail,
                      }),
                  )
                  .await;
                  let error_text = if detail.is_empty() {
                      reason.clone()
                  } else {
                      format!("{}: {}", reason, detail)
                  };
                  let _ = sqlx::query(
                      "UPDATE tool_dispatches
                       SET status = 'failed',
                           attempts = attempts + 1,
                           last_error = ?,
                           failure_kind = ?,
                           args_json = ?,
                           updated_at = CURRENT_TIMESTAMP
                       WHERE action_id = ?",
                  )
                  .bind(&error_text)
                  .bind(&failure_kind)
                  .bind(&args_json)
                  .bind(&dispatch.action_id)
                  .execute(&self.db.pool)
                  .await;
                  return Some(ToolExecutionResult {
                      tool_name: dispatch.tool_name.clone(),
                      output: error_text,
                      is_error: true,
                  });
              }
          }
            }

            if repair_attempts > 0 {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "tool",
                    run_id,
                    trace_id,
                    json!({
                        "event": "tool_args_repaired",
                        "tool": dispatch.tool_name,
                        "action_id": dispatch.action_id,
                        "attempts": repair_attempts,
                        "method": if repaired_via_model { "model_repair" } else { "local_repair" },
                    }),
                )
                .await;
            }

            if args_json != dispatch.args_json {
                let _ = sqlx::query(
                    "UPDATE tool_dispatches SET args_json = ?, updated_at = CURRENT_TIMESTAMP WHERE action_id = ?",
                )
                .bind(&args_json)
                .bind(&dispatch.action_id)
                .execute(&self.db.pool)
                .await;
            }
        }

        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "tool",
            run_id,
            trace_id,
            json!({
                "event": "dispatch",
                "tool": dispatch.tool_name,
                "action_id": dispatch.action_id,
            }),
        )
        .await;

        let timeout_secs = self.tools.timeout_for(&dispatch.tool_name);
        let mut timed_out = false;
        let result = tokio::time::timeout(
            Duration::from_secs(timeout_secs),
            self.tools
                .execute(
                    &self.db,
                    &dispatch.tool_name,
                    &args_json,
                    cancel_rx,
                    run_id,
                    trace_id,
                ),
        )
        .await;
        let result = match result {
            Ok(inner) => inner,
            Err(_) => {
                timed_out = true;
                Err("timeout".to_string())
            }
        };

        if timed_out {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "tool",
                run_id,
                trace_id,
                json!({
                    "event": "tool_dispatch_stale",
                    "reason": "timeout",
                    "tool": dispatch.tool_name,
                    "action_id": dispatch.action_id,
                    "timeout_s": timeout_secs,
                }),
            )
            .await;
        }

        let (status, output, is_error) = match result {
            Ok(out) => ("success", out, false),
            Err(e) => ("failed", e, true),
        };

        let failure_kind = if !is_error {
            None
        } else if output.to_lowercase().contains("cancelled") {
            Some(TOOL_FAILURE_KIND_CANCELLED)
        } else {
            Some(TOOL_FAILURE_KIND_EXECUTION)
        };

          let _ = sqlx::query(
              "UPDATE tool_dispatches
               SET status = ?,
                   attempts = attempts + 1,
                   result_text = ?,
                   last_error = ?,
                   failure_kind = ?,
                   updated_at = CURRENT_TIMESTAMP
               WHERE action_id = ?"
          )
          .bind(status)
          .bind(if is_error { "" } else { output.as_str() })
          .bind(if is_error { output.as_str() } else { "" })
          .bind(failure_kind)
          .bind(&dispatch.action_id)
          .execute(&self.db.pool)
          .await;

          if !is_error && !output.trim().is_empty() {
              if let Some(evidence_id) = self
                  .record_tool_output_evidence(dispatch, run_id, trace_id, &output)
                  .await
              {
                  let _ = sqlx::query(
                      "UPDATE tool_dispatches SET evidence_event_id = ? WHERE action_id = ?",
                  )
                  .bind(evidence_id)
                  .bind(&dispatch.action_id)
                  .execute(&self.db.pool)
                  .await;
              }
          }

          let _ = system_log::log_event(
              &self.db.pool,
              Some(&self.app_handle),
            if is_error { "error" } else { "info" },
            "tool",
            run_id,
            trace_id,
            json!({
                "event": "result",
                "tool": dispatch.tool_name,
                "action_id": dispatch.action_id,
                "is_error": is_error,
                "output_len": output.len(),
            }),
        )
        .await;

        Some(ToolExecutionResult {
            tool_name: dispatch.tool_name.clone(),
            output,
            is_error,
        })
    }

    pub(super) async fn repair_tool_args(
        &self,
        settings: &crate::models::Settings,
        tool_name: &str,
        raw_args: &str,
        reason: &str,
        _run_id: Option<&str>,
        _trace_id: Option<&str>,
    ) -> Option<String> {
        let schema = self.tools.schema_for(tool_name)?;
        let schema_json = serde_json::to_string(&schema).unwrap_or_else(|_| "{}".to_string());
        let raw_snippet: String = raw_args.chars().take(2400).collect();
        let system_prompt = format!(
            "You are a tool-arguments repair utility. Return ONLY a JSON object that matches the JSON schema for tool '{tool_name}'. Do not include any text outside JSON. Schema: {schema_json}"
        );
        let user_prompt = format!(
            "RAW_ARGS:\n{raw}\n\nERROR:\n{reason}\n\nReturn valid JSON only.",
            raw = raw_snippet,
            reason = reason
        );

        let model = settings
            .summarization_model
            .as_deref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .or_else(|| settings.active_model_id.clone())
            .unwrap_or_else(|| "default".to_string());

        let request = ChatCompletionRequest {
            model,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            stream: false,
            temperature: Some(0.0),
            top_p: None,
            max_tokens: Some(300),
            response_format: Some(json!({ "type": "json_object" })),
            tools: None,
            tool_choice: None,
            enable_thinking: None,
            prefill: None,
            skip_injection: Some(true),
            skip_memory: Some(true),
            skip_reminders: Some(true),
            memory_expand: None,
            allow_diagnostics: Some(false),
            json_strict: Some(true),
            skip_sanitization: None,
            run_id: None,
            request_label: Some("repair_tool_args".to_string()),
        };

        let response = self
            .model_client
            .chat_with_meta(&settings.api_base_url, settings.api_key.as_deref(), &request)
            .await
            .ok()?;

        let normalized = tool_args::normalize_tool_args(&response.content).ok()?;

        Some(normalized.normalized_json)
    }

    pub(super) async fn enqueue_tool_args_clarifier(
        &self,
        tool_name: &str,
        reason: &str,
        run_id: Option<&str>,
    ) {
        let Some(run_id) = run_id else {
            return;
        };
        let conversation_id = sqlx::query_scalar::<_, String>(
            "SELECT conversation_id FROM runs WHERE run_id = ?",
        )
        .bind(run_id)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| "default".to_string());

        let prompt = format!(
            "I couldn't run the tool \"{tool}\" because the arguments were invalid ({reason}). Can you clarify or provide the correct fields?",
            tool = tool_name,
            reason = reason
        );
        let expires_at = compute_expires_at(Utc::now(), PENDING_PROMPT_EXPIRES_SECS);
        let latest_user = self
            .db
            .get_latest_user_message(&conversation_id)
            .await
            .ok()
            .flatten();
        let (anchor_message_id, anchor_hash, anchor_created_at) = if let Some((message_id, content, created_at)) = latest_user {
            (
                Some(message_id),
                Some(crate::core::kernel::utils::text::hash_payload(
                    &summarize_snippet(&content, 160),
                )),
                Some(created_at),
            )
        } else {
            (None, None, None)
        };
        let _ = self
            .db
            .enqueue_pending_prompt(
                &conversation_id,
                &prompt,
                "tool_args_repair",
                true,
                Some("tool_args"),
                None,
                Some(&expires_at),
                anchor_message_id.as_deref(),
                anchor_hash.as_deref(),
                anchor_created_at.as_deref(),
                Some("user"),
            )
            .await;
    }



    pub(super) async fn run_thread(
        &self,
        _conversation_id: &str,
        thread_id: &str,
        goal: &str,
        depth: i64,
        settings: &crate::models::Settings,
    ) -> Result<(), String> {
        let mut attempts = 0;
        let row = loop {
            let row = sqlx::query("SELECT context_json, status FROM thread_runs WHERE id = ?")
                .bind(thread_id)
                .fetch_optional(&self.db.pool)
                .await
                .map_err(|e| e.to_string())?;
            if row.is_some() || attempts >= 10 {
                break row;
            }
            attempts += 1;
            tokio::time::sleep(Duration::from_millis(200)).await;
        };
        let Some(row) = row else {
            return Ok(());
        };
        let status: String = row.get("status");
        if status != "running" {
            return Ok(());
        }
        let context_json: String = row.get("context_json");
        let context: Value = serde_json::from_str(&context_json).unwrap_or_else(|_| json!({}));
        let parent_inner = context
            .get("parent_inner_summary")
            .and_then(|v| v.as_str())
            .unwrap_or("None")
            .to_string();
        let episodic_block = context
            .get("episodic_snippets")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(" | ")
            })
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "None".to_string());
        let semantic_block = context
            .get("semantic_snippets")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(" | ")
            })
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "None".to_string());
        let excluded_block = context
            .get("excluded")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "None".to_string());

        let system_prompt = "You are a focused sub-thread. Use ONLY the provided context. Return JSON with fields: outcome_summary, next_steps.";
        let user_prompt = format!(
            "Thread goal: {}\nDepth: {}\nParent inner summary:\n{}\n\nEpisodic snippets: {}\nSemantic snippets: {}\nExcluded context: {}\n\nReturn JSON only.",
            goal,
            depth,
            parent_inner,
            episodic_block,
            semantic_block,
            excluded_block
        );

        let request = ChatCompletionRequest {
            model: settings
                .active_model_id
                .clone()
                .unwrap_or_else(|| "default".to_string()),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            stream: false,
            temperature: None,
            top_p: None,
            max_tokens: None,
            response_format: Some(json!({ "type": "json_object" })),
            tools: None,
            tool_choice: None,
            enable_thinking: None,
            prefill: None,
            skip_injection: Some(true),
            skip_memory: Some(true),
            skip_reminders: Some(true),
            memory_expand: None,
            allow_diagnostics: Some(false),
            json_strict: Some(true),
            skip_sanitization: None,
            run_id: None,
            request_label: Some("thread_context_snapshot".to_string()),
        };

        let response = match self
            .model_client
            .chat_with_meta(&settings.api_base_url, settings.api_key.as_deref(), &request)
            .await
        {
            Ok(resp) => resp,
            Err(err) => {
                let _ = sqlx::query("UPDATE thread_runs SET status = 'failed', outcome_summary = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?")
                    .bind(err.clone())
                    .bind(thread_id)
                    .execute(&self.db.pool)
                    .await;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "thread_failed",
                        "thread_id": thread_id,
                        "error": err,
                    }),
                )
                .await;
                return Ok(());
            }
        };

        let mut summary = if let Ok(value) = serde_json::from_str::<Value>(&response.content) {
            let outcome = value
                .get("outcome_summary")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let next_steps = value
                .get("next_steps")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if !next_steps.trim().is_empty() {
                format!("{} Next: {}", outcome, next_steps)
            } else {
                outcome
            }
        } else {
            response.content
        };
        summary = summary.trim().to_string();

        let status = if summary.is_empty() { "failed" } else { "completed" };
        let _ = sqlx::query("UPDATE thread_runs SET status = ?, outcome_summary = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?")
            .bind(status)
            .bind(&summary)
            .bind(thread_id)
            .execute(&self.db.pool)
            .await;

        if !summary.is_empty() {
            let event_type = if status == "completed" { "thread_completed" } else { "thread_failed" };
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": event_type,
                    "thread_id": thread_id,
                    "summary_snippet": summarize_snippet(&summary, 240),
                }),
            )
            .await;
        }

        Ok(())
    }

    pub(super) async fn collect_outcomes(&self, state: &mut KernelState) -> Vec<Outcome> {
        let mut outcomes = Vec::new();

        let cursor_time = state.last_processed_dispatch_at.clone().unwrap_or_else(|| "".to_string());
        let cursor_id = state.last_processed_dispatch_id.clone().unwrap_or_else(|| "".to_string());
        let rows = sqlx::query(
            "SELECT action_id, tool_name, status, last_error, result_text, args_json, updated_at, failure_kind
             FROM tool_dispatches
             WHERE status IN ('success','failed')
               AND (updated_at > ? OR (updated_at = ? AND action_id > ?))
             ORDER BY updated_at ASC, action_id ASC"
        )
        .bind(&cursor_time)
        .bind(&cursor_time)
        .bind(&cursor_id)
        .fetch_all(&self.db.pool)
        .await
        .unwrap_or_default();

        let mut last_time = cursor_time.clone();
        let mut last_id = cursor_id.clone();
        for row in rows {
            let action_id: String = row.get("action_id");
            let tool_name: String = row.get("tool_name");
            let status: String = row.get("status");
            let last_error: Option<String> = row.try_get("last_error").ok();
            let result_text: Option<String> = row.try_get("result_text").ok();
            let updated_at: String = row.get("updated_at");
            let args_json: Option<String> = row.try_get("args_json").ok();
            let failure_kind: Option<String> = row.try_get("failure_kind").ok();
            let success = status == "success";
            let observations = if success {
                result_text.unwrap_or_default()
            } else {
                last_error.unwrap_or_default()
            };
            let target_hint = args_json
                .as_deref()
                .and_then(tool_target_hint_from_args_json);
            let target_key = Some(tool_penalty_key(&tool_name, target_hint.as_deref()));
            outcomes.push(Outcome {
                action_type: format!("tool_dispatch_{}", status),
                success,
                observations: summarize_snippet(&observations, 160),
                source: tool_name.clone(),
                failure_kind,
                target_key,
                action_id: Some(action_id.clone()),
                timestamp: updated_at.clone(),
            });
            last_time = updated_at;
            last_id = action_id;
        }

        if !outcomes.is_empty() {
            state.last_processed_dispatch_at = Some(last_time);
            state.last_processed_dispatch_id = Some(last_id);
        }

        let thread_cursor_time = state
            .last_processed_thread_at
            .clone()
            .unwrap_or_else(|| "".to_string());
        let thread_cursor_id = state
            .last_processed_thread_id
            .clone()
            .unwrap_or_else(|| "".to_string());
        let thread_rows = sqlx::query(
            "SELECT id, goal, status, outcome_summary, updated_at, depth FROM thread_runs WHERE status IN ('completed','failed') AND (updated_at > ? OR (updated_at = ? AND id > ?)) ORDER BY updated_at ASC, id ASC"
        )
        .bind(&thread_cursor_time)
        .bind(&thread_cursor_time)
        .bind(&thread_cursor_id)
        .fetch_all(&self.db.pool)
        .await
        .unwrap_or_default();

        let mut thread_last_time = thread_cursor_time.clone();
        let mut thread_last_id = thread_cursor_id.clone();
        for row in thread_rows {
            let thread_id: String = row.get("id");
            let status: String = row.get("status");
            let summary: Option<String> = row.try_get("outcome_summary").ok();
            let updated_at: String = row.get("updated_at");
            let success = status == "completed";
            let observations = summary.unwrap_or_default();
            outcomes.push(Outcome {
                action_type: format!("thread_{}", status),
                success,
                observations: summarize_snippet(&observations, 160),
                source: "thread".to_string(),
                failure_kind: None,
                target_key: None,
                action_id: Some(thread_id.clone()),
                timestamp: updated_at.clone(),
            });
            state.active_threads.retain(|t| t.thread_id != thread_id);
            thread_last_time = updated_at;
            thread_last_id = thread_id;
        }

        if thread_last_time != thread_cursor_time || thread_last_id != thread_cursor_id {
            state.last_processed_thread_at = Some(thread_last_time);
            state.last_processed_thread_id = Some(thread_last_id);
        }

        if !state.active_threads.is_empty() {
            state.thread_depth = state.active_threads.iter().map(|t| t.depth).max().unwrap_or(0);
        } else {
            state.thread_depth = 0;
        }

        if let Some(last) = outcomes.last() {
            state.last_outcome_at = Some(last.timestamp.clone());
        }

        outcomes
    }

    pub(super) async fn apply_outcomes(&self, state: &mut KernelState, outcomes: &[Outcome]) {
        for outcome in outcomes {
            let action = outcome.action_type.to_lowercase();
            let skip_tool_penalty = matches!(
                outcome.failure_kind.as_deref(),
                Some(TOOL_FAILURE_KIND_PLANNING) | Some(TOOL_FAILURE_KIND_CANCELLED)
            );
            if outcome.success {
                state.failure_count = (state.failure_count - 1).max(0);
                if action.contains("completed") || action.contains("success") {
                    state.stalled_count = (state.stalled_count - 1).max(0);
                }
                if action.contains("tool_dispatch_success") {
                    state.tool_failure_count = (state.tool_failure_count - 1).max(0);
                    let tool_name = outcome.source.trim();
                    let penalty_key = outcome.target_key.as_deref().unwrap_or(tool_name);
                    if !penalty_key.is_empty() {
                        if let Some(entry) = state.tool_failure_penalties.get_mut(penalty_key) {
                            entry.count = (entry.count - 1).max(0);
                            if entry.count == 0 {
                                entry.penalty_until = None;
                                entry.last_failure_at = None;
                            }
                        }
                    }
                }
            } else {
                if !skip_tool_penalty {
                    state.failure_count += 1;
                    if action.contains("failed") {
                        state.stalled_count += 1;
                    }
                }
                if action.contains("tool_dispatch_failed") && !skip_tool_penalty {
                    state.tool_failure_count += 1;
                    let tool_name = outcome.source.trim();
                    let penalty_key = outcome.target_key.as_deref().unwrap_or(tool_name);
                    if !penalty_key.is_empty() {
                        let entry = state
                            .tool_failure_penalties
                            .entry(penalty_key.to_string())
                            .or_insert_with(ToolFailurePenalty::default);
                        entry.count += 1;
                        let now = Utc::now();
                        entry.last_failure_at = Some(now.to_rfc3339());
                        entry.penalty_until =
                            Some((now + chrono::Duration::seconds(TOOL_FAILURE_PENALTY_SECS)).to_rfc3339());
                    }
                }
            }
            state.recent_outcomes.push(outcome.clone());
        }
        if state.recent_outcomes.len() > OUTCOME_HISTORY_LIMIT {
            let start = state.recent_outcomes.len() - OUTCOME_HISTORY_LIMIT;
            state.recent_outcomes = state.recent_outcomes[start..].to_vec();
        }

        if outcomes.iter().any(|o| o.action_type == "user_message_processed") {
            if !state.pending_questions.is_empty() {
                state.pending_questions.clear();
                state.uncertainty_count = (state.uncertainty_count - 1).max(0);
            }
        }

        let mut signals = Vec::new();
        if state.failure_count >= 2 {
            signals.push("repeated_failures".to_string());
        }
        if state.uncertainty_count >= 1 {
            signals.push("high_uncertainty".to_string());
        }
        if state.stalled_count >= 2 {
            signals.push("stalled_goals".to_string());
        }
        if state.tool_failure_count >= 1 {
            signals.push("tool_failures".to_string());
        }

        state.pressure_score = (state.failure_count as f32) * 1.2
            + (state.uncertainty_count as f32) * 0.8
            + (state.stalled_count as f32) * 0.6
            + (state.tool_failure_count as f32) * 0.7;
        state.pressure_signals = signals;
        self.update_mode(state).await;
        self.update_stance(state).await;
    }

    pub(super) async fn update_mode(&self, state: &mut KernelState) {
        let target = if state.pressure_score >= 2.0 {
            KernelMode::Work
        } else {
            KernelMode::Play
        };
        if target == state.mode {
            return;
        }
        let now = Utc::now();
        let can_switch = state
            .last_mode_switch_at
            .as_deref()
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
            .map(|ts| now.signed_duration_since(ts.with_timezone(&Utc)).num_seconds() >= MODE_DWELL_SECS)
            .unwrap_or(true);
        if can_switch {
            state.mode = target;
            state.last_mode_switch_at = Some(now.to_rfc3339());
        }
    }

    pub(super) async fn update_stance(&self, state: &mut KernelState) {
        let now = Utc::now();
        let due = state
            .last_medium_update_at
            .as_deref()
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
            .map(|ts| now.signed_duration_since(ts.with_timezone(&Utc)).num_seconds() >= MEDIUM_LAYER_INTERVAL_SECS)
            .unwrap_or(true);
        if !due {
            return;
        }

        if state.failure_count >= 2 {
            state.stance.stance = "clarify".to_string();
            state.stance.initiative_level = "low".to_string();
            state.stance.verbosity_target = "medium".to_string();
            state.stance.tool_preference = "normal".to_string();
        } else if state.pressure_score < 1.0 {
            state.stance.stance = "explore".to_string();
            state.stance.initiative_level = "medium".to_string();
            state.stance.verbosity_target = "medium".to_string();
            state.stance.tool_preference = "normal".to_string();
        } else {
            state.stance.stance = "execute".to_string();
            state.stance.initiative_level = "high".to_string();
            state.stance.verbosity_target = "high".to_string();
            state.stance.tool_preference = "prefer".to_string();
        }
        state.last_medium_update_at = Some(now.to_rfc3339());
        state.stance.updated_at = state.last_medium_update_at.clone();
    }

    pub(super) async fn refresh_workspace_focus(&self, state: &mut KernelState, conversation_id: &str) {
        let mut scored: Vec<(String, f32, String)> = Vec::new();
        let now = Utc::now();
        let mut memory_topic_labels: HashSet<String> = HashSet::new();
        let mut memory_belief_ids: Vec<i64> = Vec::new();
        let goal_text = format!(
            "{} {} {}",
            state.workspace_goal_thread.clone().unwrap_or_default(),
            state.workspace_open_questions.join(" "),
            hypothesis_texts(&state.workspace_active_hypotheses).join(" ")
        )
        .to_lowercase();

        let rows = sqlx::query(
            "SELECT ws.activation, ws.last_updated_at, e.label
             FROM ics_working_set ws
             JOIN ics_entities e ON e.id = ws.item_id
             WHERE ws.item_type = 'entity'
             ORDER BY ws.activation DESC, ws.last_updated_at DESC
             LIMIT 12",
        )
        .fetch_all(&self.db.pool)
        .await
        .unwrap_or_default();

        for row in rows {
            let activation: f32 = row.try_get::<f64, _>("activation").unwrap_or(0.0) as f32;
            let label: String = row.get("label");
            let ts = row
                .try_get::<Option<String>, _>("last_updated_at")
                .ok()
                .flatten();
            let recency_score = if let Some(ts) = ts {
                chrono::DateTime::parse_from_rfc3339(&ts)
                    .ok()
                    .map(|dt| {
                        let hours = now
                            .signed_duration_since(dt.with_timezone(&Utc))
                            .num_seconds()
                            .max(0) as f32
                            / 3600.0;
                        1.0 / (1.0 + hours)
                    })
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            let label_lower = label.to_lowercase();
            let goal_relevance = if !goal_text.trim().is_empty() && goal_text.contains(&label_lower) {
                0.5
            } else {
                0.0
            };
            let score = activation + recency_score + goal_relevance;
            let rationale = format!(
                "activation {:.2}, recency {:.2}, goal_match {}",
                activation,
                recency_score,
                goal_relevance > 0.0
            );
            scored.push((label, score, rationale));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut topics = Vec::new();
        for (label, _score, _reason) in scored.iter().take(8) {
            topics.push(label.clone());
        }
        if !topics.is_empty() {
            state.workspace_working_set_topics = topics.clone();
        }

        if let Some((label, _score, reason)) = scored.first() {
            let focus_changed = state.workspace_current_focus.as_deref() != Some(label.as_str());
            if focus_changed {
                state.workspace_current_focus = Some(label.clone());
                state.last_focus_change_at = Some(Utc::now().to_rfc3339());
                state.workspace_meta.current_focus = Some(make_field_meta(true, &[], &[]));
            }
            state.workspace_focus_rationale = Some(reason.clone());
            state.workspace_meta.focus_rationale = Some(make_field_meta(true, &[], &[]));
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "focus_choice",
                    "current_focus": label,
                    "rationale": reason,
                }),
            )
            .await;
        } else if let Some(goal) = state.workspace_goal_thread.clone() {
            let focus_changed = state.workspace_current_focus.as_deref() != Some(goal.as_str());
            if focus_changed {
                state.workspace_current_focus = Some(goal.clone());
                state.last_focus_change_at = Some(Utc::now().to_rfc3339());
            }
            state.workspace_focus_rationale = Some("fallback to goal_thread".to_string());
            let goal_meta = state.workspace_meta.goal_thread.as_ref();
            let speculative = !meta_is_verified_field(goal_meta);
            let evidence_event_ids = goal_meta.map(|m| m.evidence_event_ids.clone()).unwrap_or_default();
            let belief_ids = goal_meta.map(|m| m.belief_ids.clone()).unwrap_or_default();
            state.workspace_meta.current_focus =
                Some(make_field_meta(speculative, &evidence_event_ids, &belief_ids));
            state.workspace_meta.focus_rationale =
                Some(make_field_meta(speculative, &evidence_event_ids, &belief_ids));
        }

        if let Some(current_focus) = state.workspace_current_focus.clone() {
            let user_name = self
                .db
                .get_settings()
                .await
                .ok()
                .and_then(|s| s.user_display_name)
                .unwrap_or_default();
            if let Some((new_focus, rationale)) = focus_shift_candidate(
                &current_focus,
                state.last_user_input.as_deref().unwrap_or(""),
                &user_name,
                &state.workspace_active_hypotheses,
                state.workspace_goal_thread.as_deref(),
            ) {
                if new_focus != current_focus {
                    state.workspace_current_focus = Some(new_focus.clone());
                    state.workspace_focus_rationale = Some(rationale.clone());
                    state.last_focus_change_at = Some(Utc::now().to_rfc3339());
                    let mut speculative = true;
                    let mut evidence_event_ids: Vec<i64> = Vec::new();
                    let mut belief_ids: Vec<i64> = Vec::new();
                    if let Some(hypothesis) = state
                        .workspace_active_hypotheses
                        .iter()
                        .find(|h| h.text.trim().eq_ignore_ascii_case(&new_focus))
                    {
                        speculative = !hypothesis_is_verified(hypothesis);
                        evidence_event_ids = hypothesis.evidence_event_ids.clone();
                        belief_ids = hypothesis.belief_ids.clone();
                    } else if let Some(goal) = state.workspace_goal_thread.as_deref() {
                        if goal.trim().eq_ignore_ascii_case(&new_focus) {
                            speculative = !meta_is_verified_field(state.workspace_meta.goal_thread.as_ref());
                            if let Some(meta) = state.workspace_meta.goal_thread.as_ref() {
                                evidence_event_ids = meta.evidence_event_ids.clone();
                                belief_ids = meta.belief_ids.clone();
                            }
                        }
                    }
                    state.workspace_meta.current_focus =
                        Some(make_field_meta(speculative, &evidence_event_ids, &belief_ids));
                    state.workspace_meta.focus_rationale =
                        Some(make_field_meta(speculative, &evidence_event_ids, &belief_ids));
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "focus_shift",
                            "previous_focus": current_focus,
                            "current_focus": new_focus,
                            "rationale": rationale,
                        }),
                    )
                    .await;
                }
            }
        }

        if let Some(focus) = state.workspace_current_focus.clone() {
            let api = MemoryApi::new(self.db.pool.clone(), Some(self.model_client.clone()), conversation_id.to_string()).await;
            let intent = crate::core::memory::api::infer_query_intent(&focus);
            if let Ok(packet) = api
                .retrieve(&focus, &[Scope::Session, Scope::Global], intent)
                .await
            {
                let mut evidence = Vec::new();
                let mut added = 0usize;
                let mut conflicts_added = 0usize;
                for fact in packet.facts.iter().take(3) {
                    if !memory_belief_ids.contains(&fact.id) {
                        memory_belief_ids.push(fact.id);
                    }
                    memory_topic_labels.insert(fact.entity_label.to_lowercase());
                    let label = fact.entity_label.clone();
                    if !state.workspace_working_set_topics.contains(&label) {
                        state.workspace_working_set_topics.push(label);
                        added += 1;
                    }
                    evidence.push(format!("{}:{}={}", fact.entity_label, fact.key, fact.value));
                }
                for rel in packet.relations.iter().take(3) {
                    if !memory_belief_ids.contains(&rel.id) {
                        memory_belief_ids.push(rel.id);
                    }
                    let participants = rel
                        .participants
                        .iter()
                        .map(|p| p.entity_label.clone())
                        .collect::<Vec<_>>()
                        .join(", ");
                    evidence.push(format!("{}({})", rel.rel_type, participants));
                    for participant in rel.participants.iter() {
                        memory_topic_labels.insert(participant.entity_label.to_lowercase());
                        if !state.workspace_working_set_topics.contains(&participant.entity_label) {
                            state.workspace_working_set_topics.push(participant.entity_label.clone());
                            added += 1;
                        }
                    }
                }
                for conflict in packet.conflicts.iter().take(3) {
                    let question = format!("Resolve conflict: {}", conflict.topic_key);
                    if !state.workspace_open_questions.iter().any(|q| q == &question) {
                        push_capped(&mut state.workspace_open_questions, question, 8);
                        conflicts_added += 1;
                    }
                }
                if !evidence.is_empty() {
                    state.workspace_focus_rationale = Some(format!(
                        "Focus '{}' supported by memory: {}",
                        focus,
                        evidence.join(" | ")
                    ));
                    if !memory_belief_ids.is_empty() {
                        state.workspace_meta.current_focus =
                            Some(make_field_meta(false, &[], &memory_belief_ids));
                        state.workspace_meta.focus_rationale =
                            Some(make_field_meta(false, &[], &memory_belief_ids));
                    }
                }
                if state.workspace_working_set_topics.len() > 8 {
                    state.workspace_working_set_topics.truncate(8);
                }
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "memory_to_workspace",
                        "current_focus": focus,
                        "topics_added": added,
                        "conflicts_added": conflicts_added,
                        "facts": packet.facts.len(),
                        "relations": packet.relations.len(),
                    }),
                )
                .await;
            }
        }

        if !state.workspace_working_set_topics.is_empty() {
            let mut existing: HashMap<String, WorkspaceListItemMeta> = HashMap::new();
            for meta in state.workspace_meta.working_set_topics.iter() {
                existing.insert(meta.text.to_lowercase(), meta.clone());
            }
            let mut next = Vec::new();
            for topic in state.workspace_working_set_topics.iter() {
                let key = topic.to_lowercase();
                if memory_topic_labels.contains(&key) && !memory_belief_ids.is_empty() {
                    next.push(make_list_meta(topic, false, &[], &memory_belief_ids));
                } else if let Some(meta) = existing.get(&key) {
                    let mut updated = meta.clone();
                    updated.text = topic.clone();
                    next.push(updated);
                } else {
                    next.push(make_list_meta(topic, true, &[], &[]));
                }
            }
            state.workspace_meta.working_set_topics = next;
        }

        if !state.workspace_open_questions.is_empty() {
            let mut existing: HashMap<String, WorkspaceListItemMeta> = HashMap::new();
            for meta in state.workspace_meta.open_questions.iter() {
                existing.insert(meta.text.to_lowercase(), meta.clone());
            }
            let mut next = Vec::new();
            for question in state.workspace_open_questions.iter() {
                let key = question.to_lowercase();
                if let Some(meta) = existing.get(&key) {
                    let mut updated = meta.clone();
                    updated.text = question.clone();
                    next.push(updated);
                } else {
                    next.push(make_list_meta(question, true, &[], &[]));
                }
            }
            state.workspace_meta.open_questions = next;
        }

        if let Some(focus) = state.workspace_current_focus.as_deref() {
            state.self_state.current_focus = focus.to_string();
        }
    }

    pub(super) async fn sync_self_model_runtime(&self, settings: &crate::models::Settings) {
        let tool_registry = ToolRegistry;
        let enabled_tools = tool_registry.definitions_for_settings(settings);
        let active_tool_names: Vec<String> = enabled_tools
            .iter()
            .map(|tool| tool.function.name.clone())
            .collect();

        let mut capabilities = Vec::new();
        capabilities.push("memory: ics_v4_1".to_string());
        capabilities.push("memory: working_set".to_string());
        capabilities.push("memory: self_claims".to_string());
        if settings.episodic_enabled.unwrap_or(true) {
            capabilities.push("memory: episodic".to_string());
        }
        if settings.monologue_interval_seconds.unwrap_or(0) > 0 {
            capabilities.push("internal_monologue".to_string());
        }
        if settings.heartbeat_enabled.unwrap_or(true) {
            capabilities.push("heartbeat".to_string());
        }
        if settings.dream_enabled.unwrap_or(true) {
            capabilities.push("dream_consolidation".to_string());
        }
        for tool in active_tool_names.iter() {
            capabilities.push(format!("tool: {}", tool));
        }

        let mut limitations = Vec::new();
        if active_tool_names.is_empty() {
            limitations.push("no_tools_enabled".to_string());
        }
        if !settings.allow_shell_tool.unwrap_or(false) {
            limitations.push("run_shell_disabled".to_string());
        }
        if !settings.episodic_enabled.unwrap_or(true) {
            limitations.push("episodic_memory_disabled".to_string());
        }
        if settings.monologue_interval_seconds.unwrap_or(0) <= 0 {
            limitations.push("monologue_disabled".to_string());
        }

        let mut model = match self.db.get_self_model().await {
            Ok(model) => model,
            Err(_) => return,
        };
        let capabilities_json = json!(capabilities);
        let limitations_json = json!(limitations);
        let active_tools_json = json!(active_tool_names);

        if model.capabilities != capabilities_json
            || model.limitations != limitations_json
            || model.active_tools != active_tools_json
        {
            model.capabilities = capabilities_json;
            model.limitations = limitations_json;
            model.active_tools = active_tools_json;
            if self.db.set_self_model(&model).await.is_ok() {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "self_model_runtime_sync",
                        "active_tools": model.active_tools,
                    }),
                )
                .await;
            }
        }
    }

    async fn prefetch_context_hydration(
        &self,
        conversation_id: &str,
        run_id: Option<&str>,
        selected_sections: &[String],
        intent_tags: &[String],
    ) -> Option<String> {
        let mut tool_names: Vec<String> = Vec::new();
        for section in selected_sections {
            match section.as_str() {
                "World Model" => tool_names.push("get_world_model_snapshot".to_string()),
                "Subject Snapshot" => tool_names.push("get_plan_summary".to_string()),
                "Rolling Summary" => tool_names.push("get_rolling_summary".to_string()),
                "Inner Summary" => tool_names.push("get_inner_summary".to_string()),
                "Workspace Snapshot" => tool_names.push("get_workspace_state".to_string()),
                "Capabilities" => tool_names.push("get_system_capabilities".to_string()),
                "Unified Self" => tool_names.push("get_unified_self".to_string()),
                "Autobiographical Context" => tool_names.push("get_autobiographical_context".to_string()),
                _ => {}
            }
        }
        if intent_tags.iter().any(|t| t == "planning")
            && !tool_names.iter().any(|t| t == "get_workspace_state")
        {
            tool_names.push("get_workspace_state".to_string());
        }
        tool_names.sort();
        tool_names.dedup();
        if tool_names.is_empty() {
            return None;
        }

        let mut blocks: Vec<String> = Vec::new();
        let args_json = json!({ "conversation_id": conversation_id }).to_string();
        for tool_name in tool_names.iter() {
            if !ToolRegistry::is_context_only_tool(tool_name) {
                continue;
            }
            let action_id = format!("context_prefetch:{}", Uuid::new_v4());
            let dispatch = ToolDispatchRequest {
                action_id,
                tool_name: tool_name.clone(),
                args_json: args_json.clone(),
                plan_step_id: None,
            };
            let (_tx, mut cancel_rx) = watch::channel(false);
            let result = self.dispatch_tool(&dispatch, run_id, None, &mut cancel_rx).await;
            let status = match result.as_ref() {
                Some(res) if !res.is_error => "success",
                Some(_) => "error",
                None => "none",
            };
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                run_id,
                None,
                json!({
                    "event": "context_hydration_prefetch",
                    "tool": tool_name,
                    "status": status,
                }),
            )
            .await;
            if let Some(res) = result {
                if !res.is_error {
                    blocks.push(format!(
                        "<<<BEGIN_CONTEXT_TOOL:{}>>>\n{}\n<<<END_CONTEXT_TOOL:{}>>>",
                        tool_name, res.output, tool_name
                    ));
                }
            }
        }
        if blocks.is_empty() {
            None
        } else {
            Some(blocks.join("\n\n"))
        }
    }

    pub(super) async fn sync_unified_self_model(&self, state: &KernelState) {
        if let Err(err) = self_model_controller::update_unified_self_model(&self.db, state).await {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "kernel",
                None,
                None,
                json!({
                    "event": "unified_self_model_sync_failed",
                    "error": err,
                }),
            )
            .await;
        }
    }

    pub(super) fn update_self_state(
        &self,
        state: &mut KernelState,
        inner_summary_json: Option<&str>,
        last_thought: Option<&str>,
        settings: &crate::models::Settings,
    ) {
        if let Some(thought) = last_thought {
            if !thought.trim().is_empty() {
                state.self_state.last_internal_thought = summarize_snippet(thought, 160);
            }
        }

        if let Some(focus) = state.workspace_current_focus.as_deref() {
            state.self_state.current_focus = focus.to_string();
        } else if let Some(summary_json) = inner_summary_json {
            let parsed = InnerSummary::from_json(summary_json);
            state.self_state.current_focus = summarize_snippet(&parsed.focus, 120);
        }

        let uncertainty_level = if state.uncertainty_count >= 2 {
            "high"
        } else if state.uncertainty_count >= 1 {
            "medium"
        } else {
            "low"
        };
        state.self_state.uncertainty_level = uncertainty_level.to_string();

        let initiative = state.stance.initiative_level.trim();
        state.self_state.initiative_level = if initiative.is_empty() {
            "medium".to_string()
        } else {
            initiative.to_string()
        };

        if let Some(last) = state.recent_outcomes.last() {
            state.self_state.last_action_outcome = if last.success {
                "success".to_string()
            } else {
                "failure".to_string()
            };
        }

        let interval = settings.monologue_interval_seconds.unwrap_or(60).max(5);
        let now = Utc::now();
        let monologue_active = state
            .last_monologue_completed_at
            .as_deref()
            .or_else(|| state.last_monologue_at.as_deref())
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
            .map(|ts| now.signed_duration_since(ts.with_timezone(&Utc)).num_seconds() <= interval * 2)
            .unwrap_or(false);
        state.self_state.monologue_active = monologue_active;
        state.self_state.updated_at = Some(now.to_rfc3339());
    }

    pub(super) async fn refresh_controller_state(&self, state: &mut KernelState, settings: &crate::models::Settings) {
        if settings.controller_enabled.unwrap_or(true) == false {
            return;
        }
        let mut self_model = match self.db.get_self_model().await {
            Ok(model) => model,
            Err(err) => {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "controller_self_model_error",
                        "error": err.to_string(),
                    }),
                )
                .await;
                return;
            }
        };

        let _ = self
            .seed_self_scope_beliefs(&state.conversation_id, &self_model, None, None)
            .await;

        let metrics = match collect_self_evidence_metrics(&self.db.pool).await {
            Ok(metrics) => metrics,
            Err(err) => {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "controller_metrics_error",
                        "error": err,
                    }),
                )
                .await;
                return;
            }
        };

        let output = reconstruct_from_metrics(&metrics, &self_model, state.failure_count as i32);
        let previous_state = state.controller_state.clone();
        let mut controller_state: ControllerState = output.controller_state.clone();
        controller_state.uncertainty = (1.0 - controller_state.confidence).clamp(0.0, 1.0);
        controller_state.outcome_quality = outcome_quality_from_outcomes(&state.recent_outcomes);
        controller_state.last_error = last_failed_outcome(&state.recent_outcomes);
        controller_state.last_strategy = last_strategy_from_outcomes(&state.recent_outcomes)
            .or_else(|| Some(state.stance.stance.clone()));
        if let Some(quality) = controller_state.outcome_quality {
            if quality < 0.4 {
                controller_state.confidence = (controller_state.confidence - 0.1).max(0.0);
                controller_state.autonomy_level = (controller_state.autonomy_level - 0.1).max(0.1);
            } else if quality > 0.8 {
                controller_state.confidence = (controller_state.confidence + 0.05).min(1.0);
            }
            controller_state.uncertainty = (1.0 - controller_state.confidence).clamp(0.0, 1.0);
        }

        // Reflection-driven and stability-driven adjustments (bounded).
        let mut perturbation = crate::core::self_model_controller::ControllerPerturbation::default();
        if let Some(last_reflection_at) = self_model.last_reflection_at.as_deref() {
            let should_apply = match state.last_reflection_applied_at.as_deref() {
                Some(prev) => prev != last_reflection_at,
                None => true,
            };
            if should_apply {
                if let Some(status) = self_model
                    .reflection_status
                    .get("status")
                    .and_then(|v| v.as_str())
                {
                    if status == "accepted" {
                        perturbation.confidence_delta = Some(0.05);
                    } else if status == "rejected" {
                        let reason = self_model
                            .reflection_status
                            .get("rejection_reason")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default();
                        if reason == "missing_telemetry" || reason == "parse_failed" {
                            perturbation.confidence_delta = Some(-0.05);
                        } else {
                            perturbation.confidence_delta = Some(-0.02);
                        }
                    }
                }
                if perturbation.confidence_delta.is_some() {
                    state.last_reflection_applied_at = Some(last_reflection_at.to_string());
                }
            }
        }
        if state.failure_count == 0 && controller_state.outcome_quality.unwrap_or(1.0) > 0.7 {
            let stable_bump = 0.02;
            if perturbation.confidence_delta.is_none() {
                perturbation.confidence_delta = Some(stable_bump);
            } else {
                perturbation.confidence_delta = perturbation
                    .confidence_delta
                    .map(|delta| (delta + stable_bump).clamp(-0.05, 0.05));
            }
        }
        crate::core::self_model_controller::apply_controller_perturbation(
            &mut controller_state,
            &perturbation,
        );
        let mut controller_gate: ControllerGate = evaluate_gates(&controller_state);
        if let Some(override_value) = state.proaction_throttle_tools_override {
            controller_gate.throttle_tools = override_value;
            controller_gate.reasons.push("proaction_override_tools".to_string());
        }
        if let Some(override_value) = state.proaction_throttle_threads_override {
            controller_gate.throttle_threads = override_value;
            controller_gate.reasons.push("proaction_override_threads".to_string());
        }

        let self_model_changed = apply_reconstruction_to_model(&mut self_model, &output);
        if self_model_changed {
            let _ = self.db.set_self_model(&self_model).await;
        }

        state.controller_state = Some(controller_state.clone());
        state.controller_gate = Some(controller_gate.clone());
        state.telemetry_snapshot = build_telemetry_snapshot(&metrics);
        let _ = self.db.set_controller_state(&controller_state).await;

        if let (Some(prev), Some(current)) = (
            previous_state.as_ref().and_then(|s| s.last_strategy.clone()),
            controller_state.last_strategy.clone(),
        ) {
            let outcome_quality = controller_state.outcome_quality.unwrap_or(1.0);
            if prev != current && outcome_quality < 0.5 {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "strategy_shift_after_failure",
                        "previous_strategy": prev,
                        "current_strategy": current,
                        "outcome_quality": outcome_quality,
                    }),
                )
                .await;
            }
        }

        let now = Utc::now();
        let mut snapshot_needed = previous_state.is_none();
        if let Some(prev) = previous_state.as_ref() {
            if (prev.confidence - controller_state.confidence).abs() > 0.05
                || (prev.drift_score - controller_state.drift_score).abs() > 0.05
                || (prev.autonomy_level - controller_state.autonomy_level).abs() > 0.05
                || prev.verification_needed != controller_state.verification_needed
                || prev.reanchor_needed != controller_state.reanchor_needed
            {
                snapshot_needed = true;
            }
        }
        let snapshot_due = state
            .last_controller_snapshot_at
            .as_deref()
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
            .map(|ts| now.signed_duration_since(ts.with_timezone(&Utc)).num_minutes() >= 30)
            .unwrap_or(true);
        if snapshot_needed || snapshot_due {
            let _ = self.db.insert_controller_state_snapshot(&controller_state).await;
            state.last_controller_snapshot_at = Some(now.to_rfc3339());
        }

        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            None,
                json!({
                    "event": "controller_state_updated",
                    "confidence": controller_state.confidence,
                    "uncertainty": controller_state.uncertainty,
                    "drift_score": controller_state.drift_score,
                    "autonomy_level": controller_state.autonomy_level,
                    "verification_needed": controller_state.verification_needed,
                    "reanchor_needed": controller_state.reanchor_needed,
                    "evidence_coverage": controller_state.evidence_coverage,
                    "telemetry_coverage": controller_state.telemetry_coverage,
                    "outcome_quality": controller_state.outcome_quality,
                    "last_strategy": controller_state.last_strategy,
                    "last_error": controller_state.last_error,
                    "self_model_changed": self_model_changed,
                    "self_model_change_rate": if self_model_changed { 1.0 } else { 0.0 },
                    "persona_values": metrics.persona_values.len(),
                    "goal_values": metrics.goal_values.len(),
                    "goal_removed": metrics.goal_removed.len(),
                "telemetry_values": metrics.telemetry_values.len(),
                "source_counts": metrics.source_counts,
                "missing_fields": metrics.missing_fields,
            }),
        )
        .await;
    }

    pub(super) async fn retry_tool_candidates(&self, _state: &KernelState, created_at: &mut i64) -> Vec<Candidate> {
        let mut candidates = Vec::new();
        let rows = sqlx::query(
            "SELECT action_id, tool_name, args_json, attempts
             FROM tool_dispatches
             WHERE status = 'failed'
               AND (failure_kind IS NULL OR failure_kind != ?)
               AND attempts < ?"
        )
        .bind(TOOL_FAILURE_KIND_PLANNING)
        .bind(MAX_TOOL_RETRIES + 1)
        .fetch_all(&self.db.pool)
        .await
        .unwrap_or_default();

        for row in rows {
            let action_id: String = row.get("action_id");
            let tool_name: String = row.get("tool_name");
            let args_json: String = row.get("args_json");
            candidates.push(self.make_candidate(
                CandidateKind::ToolCall,
                json!({
                    "action_id": action_id,
                    "tool_name": tool_name,
                    "arguments": args_json,
                }),
                "tool_retry",
                created_at,
            ));
        }
        candidates
    }

    pub(super) async fn deferred_tool_candidates(&self, state: &KernelState, created_at: &mut i64) -> Vec<Candidate> {
        let throttle_tools = state
            .controller_gate
            .as_ref()
            .map(|g| g.throttle_tools)
            .unwrap_or(false);
        if throttle_tools {
            return Vec::new();
        }

        let items = self
            .db
            .dequeue_deferred_items(&state.conversation_id, "tool_call", 3)
            .await
            .unwrap_or_default();

        let mut candidates = Vec::new();
        for (id, content, attempt_count, reason, source) in items {
            if attempt_count >= MAX_TOOL_RETRIES + 1 {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "tool",
                    None,
                    None,
                    json!({
                        "event": "tool_throttle_defer",
                        "status": "dropped",
                        "deferred_id": id,
                        "reason": reason,
                    }),
                )
                .await;
                continue;
            }
            let Ok(value) = serde_json::from_str::<Value>(&content) else {
                continue;
            };
            let tool_name = value
                .get("tool_name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let args_json = value
                .get("args_json")
                .and_then(|v| v.as_str())
                .unwrap_or("{}")
                .to_string();
            let action_id = value
                .get("action_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if tool_name.trim().is_empty() {
                continue;
            }
            let action_id = if action_id.trim().is_empty() {
                Uuid::new_v4().to_string()
            } else {
                action_id
            };
            candidates.push(self.make_candidate(
                CandidateKind::ToolCall,
                json!({
                    "action_id": action_id,
                    "tool_name": tool_name,
                    "arguments": args_json,
                    "deferred_attempt_count": attempt_count,
                }),
                source.as_deref().unwrap_or("tool_deferred"),
                created_at,
            ));
        }
        candidates
    }

    pub(super) async fn defer_throttled_tools(&self, decision: &mut KernelDecision, state: &KernelState) -> usize {
        let mut deferred = Vec::new();
        decision.rejected.retain(|rejected| {
            if rejected.reason == "CONTROLLER_THROTTLE_TOOLS"
                && matches!(rejected.kind, CandidateKind::ToolCall)
            {
                deferred.push(rejected.clone());
                return false;
            }
            true
        });

        let mut queued = 0usize;
        for rejected in deferred {
            let payload = rejected.payload.as_ref().cloned().unwrap_or_else(|| json!({}));
            let tool_name = payload
                .get("tool_name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let args_json = payload
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}")
                .to_string();
            let action_id = payload
                .get("action_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let previous_attempts = payload
                .get("deferred_attempt_count")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            if tool_name.trim().is_empty() {
                continue;
            }
            let action_id = if action_id.trim().is_empty() {
                Uuid::new_v4().to_string()
            } else {
                action_id
            };
            let attempt_count = (previous_attempts + 1).max(1);
            let content = json!({
                "action_id": action_id,
                "tool_name": tool_name,
                "args_json": args_json,
            })
            .to_string();
            let _ = self
                .db
                .enqueue_deferred_item(
                    &state.conversation_id,
                    "tool_call",
                    &content,
                    rejected.source.as_deref(),
                    &rejected.reason,
                    None,
                    None,
                    attempt_count,
                    None,
                    None,
                )
                .await;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "tool",
                None,
                None,
                json!({
                    "event": "tool_throttle_defer",
                    "status": "queued",
                    "tool": tool_name,
                    "action_id": action_id,
                    "reason": rejected.reason,
                    "attempt_count": attempt_count,
                }),
            )
            .await;
            queued += 1;
        }

        queued
    }

    pub(super) async fn load_state(&self, conversation_id: &str) -> KernelState {
        let raw = self
            .db
            .get_kernel_state(conversation_id)
            .await
            .ok()
            .flatten();
        let mut state = raw.and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| KernelState::default_for(conversation_id));
        migrate_legacy_stop_state(&mut state);

        let purge_needed = self
            .db
            .get_key("self_memory_purge_last_at")
            .await
            .ok()
            .flatten()
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(&ts).ok())
            .map(|ts| {
                let now = Utc::now();
                now.signed_duration_since(ts.with_timezone(&Utc)).num_seconds() > 86400
            })
            .unwrap_or(true);
        if purge_needed {
            if let Ok(count) = self.db.purge_self_memory_without_evidence_ids().await {
                let _ = self
                    .db
                    .set_key("self_memory_purge_last_at", &Utc::now().to_rfc3339())
                    .await;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "self_memory_purged",
                        "purged_belief_count": count,
                    }),
                )
                .await;
            }
        }

        let backfill_needed = self
            .db
            .get_key("state_disclosure_backfill_last_at")
            .await
            .ok()
            .flatten()
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(&ts).ok())
            .map(|ts| {
                let now = Utc::now();
                now.signed_duration_since(ts.with_timezone(&Utc)).num_seconds() > 86400
            })
            .unwrap_or(true);
        if backfill_needed {
            if let Ok(count) = self.db.backfill_state_disclosure_metadata().await {
                let _ = self
                    .db
                    .set_key("state_disclosure_backfill_last_at", &Utc::now().to_rfc3339())
                    .await;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "state_disclosure_backfilled",
                        "message_count": count,
                    }),
                )
                .await;
            }
        }

        if let Ok(Some(workspace)) = self.db.get_workspace_state(conversation_id).await {
            state.workspace_goal_thread = workspace.goal_thread;
            state.workspace_active_plan_id = workspace.active_plan_id;
            state.workspace_goal_stack = workspace.goal_stack;
            state.workspace_open_questions = workspace.open_questions;
            state.workspace_active_hypotheses = workspace.active_hypotheses;
            state.workspace_working_set_topics = workspace.working_set_topics;
            state.workspace_current_focus = workspace.current_focus;
            state.workspace_focus_rationale = workspace.focus_rationale;
            state.workspace_meta = workspace.workspace_meta;
        }
        let mut updated = self.backfill_workspace_meta_invalid_evidence(&mut state).await;
        if self.demote_workspace_meta_for_stale_evidence(&mut state).await {
            updated = true;
        }
        if updated {
            let workspace_state = crate::models::WorkspaceState {
                conversation_id: state.conversation_id.clone(),
                goal_thread: state.workspace_goal_thread.clone(),
                active_plan_id: state.workspace_active_plan_id.clone(),
                goal_stack: state.workspace_goal_stack.clone(),
                open_questions: state.workspace_open_questions.clone(),
                active_hypotheses: state.workspace_active_hypotheses.clone(),
                working_set_topics: state.workspace_working_set_topics.clone(),
                current_focus: state.workspace_current_focus.clone(),
                focus_rationale: state.workspace_focus_rationale.clone(),
                workspace_meta: state.workspace_meta.clone(),
                updated_at: None,
            };
            let _ = self.db.set_workspace_state(&workspace_state).await;
            let mut regen_inner = false;
            if let Ok(settings) = self.db.get_settings().await {
                let history = self
                    .db
                    .get_history_for_conversation(conversation_id, 12)
                    .await
                    .unwrap_or_default();
                let dialogue_messages = history
                    .iter()
                    .filter_map(|msg| {
                        let content = msg.content.trim();
                        if content.is_empty() {
                            None
                        } else {
                            Some(format!("{}: {}", msg.role, content))
                        }
                    })
                    .collect::<Vec<_>>();
                let mut created_at = 0i64;
                if let Ok(candidate) = self
                    .build_inner_summary_candidate_from_dialogue(
                        conversation_id,
                        &dialogue_messages,
                        &state.recent_outcomes,
                        &state,
                        &settings,
                        &mut created_at,
                    )
                    .await
                {
                    let has_evidence = candidate_has_evidence(&candidate.payload)
                        || matches!(candidate_evidence_class(&candidate), Some("internal"));
                    if !has_evidence {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "memory",
                            None,
                            None,
                            json!({
                                "event": "memory_write_blocked",
                                "reason": "missing_evidence",
                                "category": "inner_summary",
                                "candidate_id": candidate.id,
                                "candidate_kind": format!("{:?}", candidate.kind),
                                "conversation_id": conversation_id,
                            }),
                        )
                        .await;
                    } else if let Some(summary_json) =
                        candidate.payload.get("summary_json").and_then(|v: &Value| v.as_str())
                    {
                        let control_map = system_controls::load_control_map(&self.db).await;
                        let inner_summary_mode = system_controls::mode_for("inner_summary", &control_map);
                        let memory_write_mode = system_controls::mode_for("memory_write", &control_map);
                        let inner_summary_allowed = !system_controls::mode_is_off(&inner_summary_mode)
                            && !system_controls::mode_is_degraded(&inner_summary_mode);
                        let memory_allowed =
                            system_controls::allow_memory_write(&memory_write_mode, "inner_summary_update");
                        if inner_summary_allowed && memory_allowed {
                            let _ = self.db.set_inner_summary(conversation_id, summary_json).await;
                            regen_inner = true;
                        } else {
                            let reason = if !inner_summary_allowed {
                                "inner_summary_control"
                            } else {
                                "memory_write_control"
                            };
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "warn",
                                "memory",
                                None,
                                None,
                                json!({
                                    "event": "memory_write_blocked",
                                    "reason": reason,
                                    "category": "inner_summary",
                                    "conversation_id": conversation_id,
                                }),
                            )
                            .await;
                        }
                    }
                }
                match self
                    .db
                    .enqueue_post_processing_job(
                        "rolling_summary_update",
                        Some(conversation_id),
                        None,
                    )
                    .await
                {
                    Ok(job_id) => {
                        let _ = self.db.set_summary_pending(conversation_id, true).await;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "post_processing_job_queued",
                                "job_id": job_id,
                                "job_type": "rolling_summary_update",
                                "conversation_id": conversation_id,
                                "reason": "workspace_demote",
                            }),
                        )
                        .await;
                    }
                    Err(err) => {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "post_processing_job_queue_failed",
                                "job_type": "rolling_summary_update",
                                "conversation_id": conversation_id,
                                "reason": "workspace_demote",
                                "error": err,
                            }),
                        )
                        .await;
                    }
                }
            }
            if !regen_inner {
                let _ = self.db.clear_inner_summary(conversation_id).await;
            }
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "summaries_regenerated_after_workspace_demote",
                    "inner_summary_regenerated": regen_inner,
                }),
            )
            .await;
        }
        state
    }

    pub(super) async fn persist_state(&self, state: &KernelState) {
        self.persist_state_with_owner(state, "kernel").await;
    }

    pub(super) async fn persist_state_with_owner(&self, state: &KernelState, owner: &str) {
        let json_state = serde_json::to_string(state).unwrap_or_else(|_| "{}".to_string());
        let _ = self
            .db
            .set_kernel_state(&state.conversation_id, &json_state, Some(owner))
            .await;
        let workspace_state = crate::models::WorkspaceState {
            conversation_id: state.conversation_id.clone(),
            goal_thread: state.workspace_goal_thread.clone(),
            active_plan_id: state.workspace_active_plan_id.clone(),
            goal_stack: state.workspace_goal_stack.clone(),
            open_questions: state.workspace_open_questions.clone(),
            active_hypotheses: state.workspace_active_hypotheses.clone(),
            working_set_topics: state.workspace_working_set_topics.clone(),
            current_focus: state.workspace_current_focus.clone(),
            focus_rationale: state.workspace_focus_rationale.clone(),
            workspace_meta: state.workspace_meta.clone(),
            updated_at: None,
        };
        let _ = self.db.set_workspace_state(&workspace_state).await;
    }

    pub(super) async fn persist_monologue_patch(&self, state: &KernelState) {
        let conversation_id = &state.conversation_id;
        for _ in 0..2 {
            let (mut base_state, version) = match self.db.get_kernel_state_with_meta(conversation_id).await {
                Ok(Some((json_state, version))) => {
                    let parsed: KernelState = serde_json::from_str(&json_state)
                        .unwrap_or_else(|_| state.clone());
                    (parsed, Some(version))
                }
                _ => (state.clone(), None),
            };
            apply_monologue_patch(&mut base_state, state);
            let json_state = serde_json::to_string(&base_state).unwrap_or_else(|_| "{}".to_string());
            let updated = if let Some(version) = version {
                self.db
                    .update_kernel_state_with_version(conversation_id, &json_state, Some("monologue"), version)
                    .await
                    .unwrap_or(false)
            } else {
                let _ = self.db.set_kernel_state(conversation_id, &json_state, Some("monologue")).await;
                true
            };
            if updated {
                return;
            }
        }
        let json_state = serde_json::to_string(state).unwrap_or_else(|_| "{}".to_string());
        let _ = self
            .db
            .set_kernel_state(conversation_id, &json_state, Some("monologue"))
            .await;
    }




}

fn normalize_meta_cog_reason(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return "meta_cog_unspecified".to_string();
    }
    let normalized = trimmed.to_lowercase().replace(' ', "_");
    if normalized.starts_with("meta_cog_") {
        normalized
    } else {
        format!("meta_cog_{}", normalized)
    }
}

fn is_fallback_emit_source(source: Option<&str>) -> bool {
    let source = source.unwrap_or("").trim();
    if source.is_empty() {
        return false;
    }
    let source = source.to_lowercase();
    source.contains("fallback") || source.contains("refusal")
}

fn compute_response_origin(
    system_state_mode: bool,
    summary_echo_rewritten: bool,
    workspace_fallback: bool,
    emit_source: Option<&str>,
) -> ResponseOrigin {
    if system_state_mode {
        ResponseOrigin::SystemState
    } else if summary_echo_rewritten {
        ResponseOrigin::SummaryEcho
    } else if workspace_fallback || is_fallback_emit_source(emit_source) {
        ResponseOrigin::Fallback
    } else {
        ResponseOrigin::Primary
    }
}

fn apply_monologue_patch(target: &mut KernelState, source: &KernelState) {
    target.last_monologue_at = source.last_monologue_at.clone();
    target.last_monologue_tick_id = source.last_monologue_tick_id.clone();
    target.last_monologue_started_at = source.last_monologue_started_at.clone();
    target.last_monologue_completed_at = source.last_monologue_completed_at.clone();
    target.last_monologue_tick_outcome = source.last_monologue_tick_outcome.clone();
    target.last_monologue_status_emitted = source.last_monologue_status_emitted;
    target.last_monologue_visible = source.last_monologue_visible;
    target.monologue_window_start = source.monologue_window_start.clone();
    target.monologue_count = source.monologue_count;
    target.monologue_suppression_window_start = source.monologue_suppression_window_start.clone();
    target.monologue_suppression_counts = source.monologue_suppression_counts.clone();
    target.monologue_json_supported = source.monologue_json_supported;
    target.monologue_json_disabled_until = source.monologue_json_disabled_until.clone();
    target.monologue_force_primary = source.monologue_force_primary;
    target.last_monologue_anchor_epoch = source.last_monologue_anchor_epoch;
    target.monologue_quiet_until = source.monologue_quiet_until.clone();
    target.monologue_surface_until = source.monologue_surface_until.clone();
    target.monologue_emit_loop_breaker_triggered = source.monologue_emit_loop_breaker_triggered;
    target.monologue_candidate_reject_streak = source.monologue_candidate_reject_streak;
    target.monologue_misaligned_streak = source.monologue_misaligned_streak;
    target.monologue_loop_streak = source.monologue_loop_streak;
}

fn task_phase_label(phase: &TaskPhase) -> &'static str {
    match phase {
        TaskPhase::Running => "Running",
        TaskPhase::AwaitingUser => "AwaitingUser",
        TaskPhase::ResolvingWithDefaults => "ResolvingWithDefaults",
        TaskPhase::Aborting => "Aborting",
        TaskPhase::Terminated => "Terminated",
    }
}

fn parse_task_phase(value: &str) -> Option<TaskPhase> {
    match value.trim().to_lowercase().as_str() {
        "running" => Some(TaskPhase::Running),
        "awaitinguser" | "awaiting_user" => Some(TaskPhase::AwaitingUser),
        "resolvingwithdefaults" | "resolving_with_defaults" => Some(TaskPhase::ResolvingWithDefaults),
        "aborting" => Some(TaskPhase::Aborting),
        "terminated" => Some(TaskPhase::Terminated),
        _ => None,
    }
}

fn detect_identity_signal(text: &str) -> Option<(&'static str, &'static str)> {
    let lowered = text.to_lowercase();
    let user_patterns = [
        "i am ",
        "i'm ",
        "my name is",
        "call me",
        "i go by",
        "i work as",
        "i am a ",
        "i'm a ",
    ];
    for pattern in user_patterns.iter() {
        if lowered.contains(pattern) {
            return Some(("user", *pattern));
        }
    }
    let assistant_patterns = [
        "you are ",
        "you're ",
        "your name is",
        "you are called",
        "you should be called",
    ];
    for pattern in assistant_patterns.iter() {
        if lowered.contains(pattern) {
            return Some(("assistant", *pattern));
        }
    }
    None
}

fn is_identity_claim_text(text: &str) -> bool {
    let lowered = text.to_lowercase();
    let patterns = [
        "i am ",
        "i'm ",
        "my name is",
        "call me",
        "i go by",
        "i work as",
        "i am a ",
        "i'm a ",
        "you are ",
        "you're ",
        "your name is",
    ];
    patterns.iter().any(|p| lowered.contains(p))
}

fn is_capability_claim_text(text: &str) -> bool {
    let lowered = text.to_lowercase();
    let patterns = [
        "i can ",
        "i can't",
        "i cannot",
        "i'm able to",
        "i am able to",
        "i'm unable to",
        "i am unable to",
        "you can ",
        "you can't",
        "you cannot",
        "you're able to",
        "you are able to",
        "you are unable to",
        "you aren't able to",
    ];
    patterns.iter().any(|p| lowered.contains(p))
}

fn ambiguity_score(input: &str) -> f32 {
    let lower = input.to_lowercase();
    if lower.trim().is_empty() {
        return 0.0;
    }
    let ambiguity_terms = [
        "maybe",
        "might",
        "not sure",
        "unsure",
        "i think",
        "probably",
        "approximately",
        "around",
        "guess",
        "perhaps",
    ];
    let mut score: f32 = 0.0;
    if ambiguity_terms.iter().any(|term| lower.contains(term)) {
        score += 0.6;
    }
    if lower.contains(" or ") {
        score += 0.2;
    }
    if lower.split_whitespace().count() <= 4 {
        score += 0.2;
    }
    score.min(1.0)
}
