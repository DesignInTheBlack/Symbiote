use super::*;
pub(crate) fn is_research_tool_candidate(tool_name: &str) -> bool {
    let name = tool_name.trim().to_lowercase();
    if RESEARCH_TOOLS.iter().any(|t| t.eq_ignore_ascii_case(&name)) {
        return true;
    }
    name.starts_with("web_") || name.contains("search") || name.contains("browse")
}

pub(crate) fn is_research_tool(candidate: &Candidate) -> bool {
    let tool_name = candidate
        .payload
        .get("tool_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    is_research_tool_candidate(tool_name)
}

pub(crate) fn is_local_tool_name(tool_name: &str) -> bool {
    matches!(
        tool_name.trim().to_lowercase().as_str(),
        "save_context"
            | "read_context"
            | "get_current_time"
            | "get_system_logs"
            | "get_inner_summary"
            | "get_workspace_state"
            | "get_rolling_summary"
            | "get_system_capabilities"
            | "get_unified_self"
            | "get_autobiographical_context"
    )
}

pub(crate) fn local_tool_args_fallback(tool_name: &str, raw_args: &str) -> Option<String> {
    let raw = raw_args.trim();
    if raw.is_empty() {
        return None;
    }
    let tool = tool_name.trim().to_lowercase();
    match tool.as_str() {
        "web_lookup" => Some(serde_json::json!({ "query": raw }).to_string()),
        "get_system_logs" => {
            let lower = raw.to_lowercase();
            let mut obj = serde_json::Map::new();
            for cat in [
                "kernel", "memory", "tool", "scheduler", "pipeline", "system", "ui", "model", "audit",
            ] {
                if lower.contains(cat) {
                    obj.insert("category".to_string(), Value::String(cat.to_string()));
                    break;
                }
            }
            Some(Value::Object(obj).to_string())
        }
        "get_inner_summary"
        | "get_workspace_state"
        | "get_rolling_summary"
        | "get_current_time"
        | "get_system_capabilities"
        | "get_unified_self"
        | "get_autobiographical_context" => {
            Some("{}".to_string())
        }
        "read_context" => Some(serde_json::json!({ "key": raw }).to_string()),
        _ => None,
    }
}

pub(crate) fn user_requested_tool_in_input(input: &str, tool_name: &str) -> bool {
    let input = input.to_lowercase();
    let tool = tool_name.trim().to_lowercase();
    if tool.is_empty() {
        return false;
    }
    let tool_spaced = tool.replace('_', " ");
    input.contains(&tool)
        || (!tool_spaced.is_empty() && input.contains(&tool_spaced))
        || input.contains(&format!("use {}", tool))
        || input.contains(&format!("run {}", tool))
        || input.contains(&format!("call {}", tool))
        || (is_research_tool_candidate(&tool)
            && [
                "look up",
                "lookup",
                "search",
                "google",
                "browse",
                "web",
                "internet",
                "online",
                "find out",
            ]
            .iter()
            .any(|phrase| input.contains(phrase)))
}

pub(crate) fn extract_inline_tool_call(raw: &str) -> Option<(String, String, String)> {
    fn extract_from_value(value: &Value) -> Option<(String, String)> {
        let name = value
            .get("name")
            .or_else(|| value.get("tool_name"))
            .or_else(|| value.get("tool"))
            .and_then(|v| v.as_str())?
            .trim()
            .to_string();
        if name.is_empty() {
            return None;
        }
        let args_value = value.get("arguments").or_else(|| value.get("args"));
        let args = match args_value {
            Some(Value::String(s)) => s.trim().to_string(),
            Some(other) => serde_json::to_string(other).ok()?,
            None => "{}".to_string(),
        };
        Some((name, args))
    }

    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
            if let Some((name, args)) = extract_from_value(&value) {
                return Some((name, args, String::new()));
            }
        }
    }

    let start = clamp_char_boundary(raw, raw.rfind('{')?);
    let mut end = clamp_char_boundary(raw, raw.rfind('}')?);
    if end >= raw.len() {
        end = raw.len().saturating_sub(1);
    }
    if end <= start {
        return None;
    }
    let json_slice = &raw[start..=end];
    let Ok(value) = serde_json::from_str::<Value>(json_slice) else {
        return None;
    };
    let (name, args) = extract_from_value(&value)?;
    let cleaned = raw[..start].trim_end().to_string();
    Some((name, args, cleaned))
}

pub(crate) fn has_research_budget(state: &KernelState, settings: &crate::models::Settings) -> bool {
    let budget = settings.research_budget_per_hour.unwrap_or(0);
    if budget <= 0 {
        return false;
    }
    let cost = settings.research_cost_per_call.unwrap_or(1).max(1);
    state.research_used + cost <= budget
}

pub(crate) fn normalize_slot_key(key: &str) -> String {
    key.trim().to_lowercase()
}

pub(crate) fn normalize_slot_set(slots: &[String]) -> Vec<String> {
    let mut items: Vec<String> = slots.iter().map(|s| normalize_slot_key(s)).collect();
    items.sort();
    items.dedup();
    items
}

pub(crate) fn hash_slot_set(slots: &[String]) -> String {
    let normalized = normalize_slot_set(slots);
    if normalized.is_empty() {
        return String::new();
    }
    hash_payload(&normalized.join("|"))
}

pub(crate) fn is_slot_subset(slots: &[String], asked_sets: &[Vec<String>]) -> bool {
    if slots.is_empty() {
        return false;
    }
    let target = normalize_slot_set(slots);
    for asked in asked_sets {
        let asked_norm = normalize_slot_set(asked);
        if asked_norm.is_empty() {
            continue;
        }
        if target.iter().all(|slot| asked_norm.contains(slot)) {
            return true;
        }
    }
    false
}

pub(crate) fn extract_requested_slots(payload: &Value) -> Vec<String> {
    let mut slots = Vec::new();
    if let Some(arr) = payload.get("requested_slots").and_then(|v| v.as_array()) {
        for item in arr {
            if let Some(slot) = item.as_str() {
                let clean = slot.trim();
                if !clean.is_empty() {
                    slots.push(clean.to_string());
                }
            }
        }
    } else if let Some(arr) = payload.get("missing_slots").and_then(|v| v.as_array()) {
        for item in arr {
            if let Some(slot) = item.as_str() {
                let clean = slot.trim();
                if !clean.is_empty() {
                    slots.push(clean.to_string());
                }
            }
        }
    }
    slots
}

pub(crate) fn extract_id_list(payload: &Value, key: &str) -> Vec<i64> {
    payload
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    if let Some(n) = item.as_i64() {
                        Some(n)
                    } else if let Some(s) = item.as_str() {
                        s.trim().parse::<i64>().ok()
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

pub(crate) fn set_id_list(payload: &mut Value, key: &str, ids: &[i64]) {
    if let Some(obj) = payload.as_object_mut() {
        if ids.is_empty() {
            obj.remove(key);
        } else {
            obj.insert(key.to_string(), json!(ids));
        }
    }
}

pub(crate) fn format_slot_value(value: &Value) -> String {
    match value {
        Value::Number(num) => num.to_string(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        _ => value.to_string(),
    }
}

pub(crate) fn defaults_used_from_provenance(provenance: &BTreeMap<String, String>) -> Vec<String> {
    provenance
        .iter()
        .filter_map(|(slot, source)| {
            if matches!(source.as_str(), "registry" | "seed_default" | "inferred_default") {
                Some(slot.clone())
            } else {
                None
            }
        })
        .collect()
}

pub(crate) fn is_calculator_prompt(input: &str) -> bool {
    let lower = input.to_lowercase();
    let has_numbers = input.chars().any(|c| c.is_ascii_digit());
    let math_signal = lower.contains("calculate")
        || lower.contains("compute")
        || lower.contains("average")
        || lower.contains("mean")
        || lower.contains("variance")
        || lower.contains("std")
        || lower.contains("weighted")
        || lower.contains("sum")
        || lower.contains("ratio")
        || lower.contains("percent")
        || lower.contains("percentage");
    let looks_like_request = lower.contains('?')
        || lower.starts_with("calculate")
        || lower.starts_with("compute")
        || lower.starts_with("sum")
        || lower.starts_with("average")
        || lower.starts_with("mean")
        || lower.starts_with("what")
        || lower.starts_with("how")
        || lower.starts_with("give me")
        || lower.starts_with("find");
    let explicit_intent = math_signal && looks_like_request;
    let has_ops = input.chars().any(|c| "+*/=".contains(c));
    let has_equals = input.contains('=');
    let has_table = input
        .lines()
        .any(|l| l.contains('|') && l.contains('-') && l.contains('|'));
    let num_count = input
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .count();
    let has_formula = has_numbers && (has_equals || (has_ops && num_count >= 2));
    let numeric_signal = has_numbers && (explicit_intent || has_formula);
    let table_signal = has_table && has_numbers;
    numeric_signal || has_formula || table_signal
}

pub(crate) fn tool_fingerprint(tool_name: &str, args_json: &str) -> String {
    let payload = format!("{}::{}", tool_name.trim().to_lowercase(), args_json.trim());
    hash_payload(&payload)
}

pub(crate) fn tool_failure_active(penalty: &ToolFailurePenalty) -> bool {
    let now = Utc::now();
    if let Some(until) = penalty.penalty_until.as_deref() {
        if let Some(ts) = timestamp_from_str(until) {
            return ts > now;
        }
    }
    if let Some(last) = penalty.last_failure_at.as_deref() {
        if let Some(ts) = timestamp_from_str(last) {
            return now.signed_duration_since(ts).num_minutes() <= 30;
        }
    }
    false
}

pub(crate) fn recent_tool_failure_count_for(state: &KernelState, tool_name: Option<&str>) -> usize {
    let Some(tool_name) = tool_name else {
        return 0;
    };
    let needle = tool_name.trim().to_lowercase();
    if needle.is_empty() {
        return 0;
    }
    state
        .tool_failure_penalties
        .iter()
        .filter(|(key, penalty)| {
            tool_failure_active(penalty)
                && key
                    .to_lowercase()
                    .starts_with(&needle)
        })
        .count()
}

pub(crate) fn validate_tool_call_args(tool_name: &str, args_json: &str) -> Result<(), String> {
    let normalized = tool_args::normalize_tool_args(args_json).map_err(|e| e.reason())?;
    tool_args::validate_tool_args(tool_name, &normalized.value).map_err(|e| e.reason())?;
    Ok(())
}

pub(crate) fn tool_outcome_from_result(result: &ToolExecutionResult) -> ToolOutcome {
    if result.is_error {
        let lower = result.output.to_lowercase();
        if lower.contains("timeout") {
            ToolOutcome::Timeout
        } else {
            ToolOutcome::HardFail
        }
    } else {
        if result.output.trim().is_empty() {
            ToolOutcome::SoftFail
        } else {
            ToolOutcome::Success
        }
    }
}

pub(crate) fn tool_penalty_key(tool_name: &str, target_hint: Option<&str>) -> String {
    let tool = tool_name.trim();
    let Some(target_hint) = target_hint else {
        return tool.to_string();
    };
    let normalized = normalize_tool_target_hint(target_hint);
    if normalized.is_empty() {
        tool.to_string()
    } else {
        format!("{}::{}", tool, normalized)
    }
}

pub(crate) fn tool_target_hint_from_payload(payload: &Value) -> Option<String> {
    if let Some(target) = payload.get("target").and_then(|v| v.as_str()) {
        return Some(target.to_string());
    }
    if let Some(target) = payload.get("domain").and_then(|v| v.as_str()) {
        return Some(target.to_string());
    }
    if let Some(url) = payload.get("url").and_then(|v| v.as_str()) {
        return Some(url.to_string());
    }
    if let Some(query) = payload.get("query").and_then(|v| v.as_str()) {
        return Some(query.to_string());
    }
    if let Some(query) = payload.get("q").and_then(|v| v.as_str()) {
        return Some(query.to_string());
    }
    if let Some(args) = payload.get("arguments") {
        match args {
            Value::String(raw) => {
                if let Some(target) = tool_target_hint_from_args_json(raw) {
                    return Some(target);
                }
                if !raw.trim().is_empty() {
                    return Some(raw.trim().to_string());
                }
            }
            Value::Object(_) => {
                if let Some(target) = tool_target_hint_from_value(args) {
                    return Some(target);
                }
            }
            _ => {}
        }
    }
    None
}

pub(crate) fn tool_target_hint_from_args_json(args_json: &str) -> Option<String> {
    let parsed = serde_json::from_str::<Value>(args_json).ok()?;
    tool_target_hint_from_value(&parsed)
}

pub(crate) fn tool_target_hint_from_value(value: &Value) -> Option<String> {
    let obj = value.as_object()?;
    for key in ["target", "domain", "url", "query", "q"] {
        if let Some(raw) = obj.get(key).and_then(|v| v.as_str()) {
            if !raw.trim().is_empty() {
                return Some(raw.to_string());
            }
        }
    }
    None
}

pub(crate) fn normalize_tool_target_hint(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let host_hint = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        let without_scheme = trimmed.split("://").nth(1).unwrap_or(trimmed);
        without_scheme.split('/').next().unwrap_or(without_scheme)
    } else {
        trimmed
    };
    let collapsed = host_hint
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    collapsed.chars().take(64).collect::<String>()
}
