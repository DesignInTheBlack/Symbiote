use std::collections::HashMap;

use jsonschema::{Draft, JSONSchema};
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;

use crate::core::tool_registry::ToolRegistry;

#[derive(Debug, Clone)]
pub enum ToolArgErrorKind {
    InvalidJson,
    RootNotObject,
    SchemaInvalid,
    RangeInvalid,
}

#[derive(Debug, Clone)]
pub struct ToolArgError {
    pub kind: ToolArgErrorKind,
    pub message: String,
}

impl ToolArgError {
    pub fn new(kind: ToolArgErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn reason(&self) -> String {
        match self.kind {
            ToolArgErrorKind::InvalidJson => "invalid_json".to_string(),
            ToolArgErrorKind::RootNotObject => "root_not_object".to_string(),
            ToolArgErrorKind::SchemaInvalid => {
                if self.message.trim().is_empty() {
                    "schema_invalid".to_string()
                } else {
                    format!("schema_invalid: {}", self.message)
                }
            }
            ToolArgErrorKind::RangeInvalid => self.message.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolArgNormalization {
    pub value: Value,
    pub normalized_json: String,
    pub repaired: bool,
    pub extracted: bool,
}

static LEADING_LABEL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^(assistant|assistant_message|assistant_response|arguments|args|tool arguments|tool_args|payload|json)\s*[:=]\s*")
        .expect("label regex")
});

static SCHEMA_CACHE: Lazy<HashMap<String, JSONSchema>> = Lazy::new(|| {
    let registry = ToolRegistry;
    let mut map = HashMap::new();
    for tool in registry.definitions().into_iter() {
        let name = tool.function.name.to_lowercase();
        let schema = tool.function.parameters;
        if let Ok(compiled) =
            JSONSchema::options().with_draft(Draft::Draft7).compile(&schema)
        {
            map.insert(name, compiled);
        }
    }
    map
});

pub fn normalize_tool_args(raw: &str) -> Result<ToolArgNormalization, ToolArgError> {
    let mut cleaned = raw.trim().to_string();
    if cleaned.is_empty() {
        return Err(ToolArgError::new(
            ToolArgErrorKind::InvalidJson,
            "invalid_json",
        ));
    }

    cleaned = strip_markdown_fence(&cleaned);
    cleaned = LEADING_LABEL_RE.replace(cleaned.trim(), "").to_string();

    let mut extracted = false;
    if let Some(snippet) = extract_first_json_object(&cleaned) {
        cleaned = snippet.to_string();
        extracted = true;
    } else if !cleaned.contains('{') && cleaned.contains(':') {
        cleaned = format!("{{{}}}", cleaned);
        extracted = true;
    }

    cleaned = normalize_json_quotes(&cleaned);
    cleaned = strip_trailing_commas(&cleaned);

    let mut repaired = false;
    let mut value = match serde_json::from_str::<Value>(&cleaned) {
        Ok(value) => value,
        Err(_) => {
            match json5::from_str::<Value>(&cleaned) {
                Ok(value) => {
                    repaired = true;
                    value
                }
                Err(_) => {
                    return Err(ToolArgError::new(
                        ToolArgErrorKind::InvalidJson,
                        "invalid_json",
                    ))
                }
            }
        }
    };

    if !value.is_object() {
        return Err(ToolArgError::new(
            ToolArgErrorKind::RootNotObject,
            "root_not_object",
        ));
    }

    if value.as_object().map(|obj| obj.is_empty()).unwrap_or(false) {
        // normalize empty objects to ensure consistent JSON output
        value = serde_json::json!({});
    }

    let normalized_json = serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string());

    Ok(ToolArgNormalization {
        value,
        normalized_json,
        repaired,
        extracted,
    })
}

pub fn validate_tool_args(tool_name: &str, value: &Value) -> Result<(), ToolArgError> {
    if !value.is_object() {
        return Err(ToolArgError::new(
            ToolArgErrorKind::RootNotObject,
            "root_not_object",
        ));
    }

    let key = tool_name.trim().to_lowercase();
    let schema = SCHEMA_CACHE.get(&key).ok_or_else(|| {
        ToolArgError::new(ToolArgErrorKind::SchemaInvalid, "unknown_tool_schema")
    })?;

    if let Err(mut errors) = schema.validate(value) {
        if let Some(err) = errors.next() {
            return Err(ToolArgError::new(
                ToolArgErrorKind::SchemaInvalid,
                err.to_string(),
            ));
        }
        return Err(ToolArgError::new(
            ToolArgErrorKind::SchemaInvalid,
            "schema_invalid",
        ));
    }

    if let Some(obj) = value.as_object() {
        if let Some(max_results) = obj.get("max_results") {
            let max = max_results
                .as_i64()
                .ok_or_else(|| ToolArgError::new(ToolArgErrorKind::RangeInvalid, "max_results_invalid"))?;
            if !(1..=10).contains(&max) {
                return Err(ToolArgError::new(
                    ToolArgErrorKind::RangeInvalid,
                    "max_results_out_of_range",
                ));
            }
        }
        if let Some(recency_days) = obj.get("recency_days") {
            let recency = recency_days
                .as_i64()
                .ok_or_else(|| ToolArgError::new(ToolArgErrorKind::RangeInvalid, "recency_days_invalid"))?;
            if recency < 0 || recency > 365 {
                return Err(ToolArgError::new(
                    ToolArgErrorKind::RangeInvalid,
                    "recency_days_out_of_range",
                ));
            }
        }
        if let Some(limit) = obj.get("limit") {
            let limit = limit
                .as_i64()
                .ok_or_else(|| ToolArgError::new(ToolArgErrorKind::RangeInvalid, "limit_invalid"))?;
            if !(1..=100).contains(&limit) {
                return Err(ToolArgError::new(
                    ToolArgErrorKind::RangeInvalid,
                    "limit_out_of_range",
                ));
            }
        }

        let name = tool_name.trim().to_lowercase();
        if name == "read_context" {
            let key = obj.get("key").and_then(|v| v.as_str()).unwrap_or("").trim();
            if key.is_empty() {
                return Err(ToolArgError::new(
                    ToolArgErrorKind::RangeInvalid,
                    "read_context_requires_key",
                ));
            }
        }
        if name == "save_context" {
            let key = obj.get("key").and_then(|v| v.as_str()).unwrap_or("").trim();
            let value_str = obj.get("value").and_then(|v| v.as_str()).unwrap_or("").trim();
            if key.is_empty() {
                return Err(ToolArgError::new(
                    ToolArgErrorKind::RangeInvalid,
                    "save_context_requires_key",
                ));
            }
            if value_str.is_empty() {
                return Err(ToolArgError::new(
                    ToolArgErrorKind::RangeInvalid,
                    "save_context_requires_value",
                ));
            }
        }
        if name == "run_shell" {
            let cmd = obj.get("command").and_then(|v| v.as_str()).unwrap_or("").trim();
            if cmd.is_empty() {
                return Err(ToolArgError::new(
                    ToolArgErrorKind::RangeInvalid,
                    "run_shell_requires_command",
                ));
            }
        }
        if name == "web_lookup" {
            let query = obj.get("query").and_then(|v| v.as_str()).unwrap_or("").trim();
            if query.is_empty() {
                return Err(ToolArgError::new(
                    ToolArgErrorKind::RangeInvalid,
                    "web_lookup_requires_query",
                ));
            }
        }
    }

    Ok(())
}

pub fn normalize_and_validate(
    tool_name: &str,
    raw_args: &str,
) -> Result<ToolArgNormalization, ToolArgError> {
    let normalized = normalize_tool_args(raw_args)?;
    validate_tool_args(tool_name, &normalized.value)?;
    Ok(normalized)
}

fn strip_markdown_fence(raw: &str) -> String {
    let trimmed = raw.trim();
    if !trimmed.starts_with("```") {
        return trimmed.to_string();
    }
    let mut lines = trimmed.lines();
    let _ = lines.next();
    let mut content: Vec<&str> = Vec::new();
    for line in lines {
        if line.trim_start().starts_with("```") {
            break;
        }
        content.push(line);
    }
    if content.is_empty() {
        trimmed.to_string()
    } else {
        content.join("\n")
    }
}

fn extract_first_json_object(raw: &str) -> Option<&str> {
    let mut in_string = false;
    let mut escape = false;
    let mut depth = 0usize;
    let mut start: Option<usize> = None;
    for (idx, ch) in raw.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if ch == '{' {
            if depth == 0 {
                start = Some(idx);
            }
            depth = depth.saturating_add(1);
        } else if ch == '}' {
            if depth == 0 {
                continue;
            }
            depth = depth.saturating_sub(1);
            if depth == 0 {
                let s = start?;
                return raw.get(s..=idx);
            }
        }
    }
    None
}

fn normalize_json_quotes(raw: &str) -> String {
    raw.replace('\u{201C}', "\"")
        .replace('\u{201D}', "\"")
        .replace('\u{2018}', "'")
        .replace('\u{2019}', "'")
        .replace('\u{00AB}', "\"")
        .replace('\u{00BB}', "\"")
}

fn strip_trailing_commas(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    let mut in_string = false;
    let mut escape = false;

    while let Some(ch) = chars.next() {
        if escape {
            out.push(ch);
            escape = false;
            continue;
        }
        if ch == '\\' && in_string {
            out.push(ch);
            escape = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            out.push(ch);
            continue;
        }
        if !in_string && ch == ',' {
            let mut iter = chars.clone();
            let mut next_non_ws = None;
            while let Some(next) = iter.next() {
                if !next.is_whitespace() {
                    next_non_ws = Some(next);
                    break;
                }
            }
            if matches!(next_non_ws, Some('}') | Some(']')) {
                continue;
            }
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{normalize_tool_args, validate_tool_args};
    use serde_json::json;

    #[test]
    fn normalize_tool_args_repairs_trailing_comma_and_fence() {
        let raw = "```json\n{'query': 'alpha',}\n```";
        let normalized = normalize_tool_args(raw).expect("normalized");
        assert!(normalized.repaired);
        assert_eq!(
            normalized.value.get("query").and_then(|v| v.as_str()),
            Some("alpha")
        );
    }

    #[test]
    fn normalize_tool_args_wraps_missing_braces() {
        let raw = "query: \"alpha\"";
        let normalized = normalize_tool_args(raw).expect("normalized");
        assert_eq!(
            normalized.value.get("query").and_then(|v| v.as_str()),
            Some("alpha")
        );
    }

    #[test]
    fn validate_tool_args_accepts_minimum_payloads() {
        let mut cases = Vec::new();
        cases.push(("run_shell", json!({"command": "dir"})));
        cases.push(("get_current_time", json!({})));
        cases.push(("get_system_logs", json!({})));
        cases.push(("get_system_capabilities", json!({})));
        cases.push(("get_inner_summary", json!({})));
        cases.push(("get_workspace_state", json!({})));
        cases.push(("get_rolling_summary", json!({})));
        cases.push(("save_context", json!({"key": "k", "value": "v"})));
        cases.push(("read_context", json!({"key": "k"})));
        cases.push(("web_lookup", json!({"query": "alpha"})));

        for (name, value) in cases {
            assert!(
                validate_tool_args(name, &value).is_ok(),
                "schema should accept minimal args for {name}"
            );
        }
    }

    #[test]
    fn validate_tool_args_rejects_range_violations() {
        let value = json!({"query": "alpha", "max_results": 25});
        assert!(validate_tool_args("web_lookup", &value).is_err());
        let value = json!({"limit": 0});
        assert!(validate_tool_args("get_system_logs", &value).is_err());
    }
}
