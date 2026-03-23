use serde_json::{json, Value};

use super::text::clamp_char_boundary;
use crate::models::Settings;

fn extract_first_json_object(raw: &str) -> Option<&str> {
    let mut depth = 0usize;
    let mut start: Option<usize> = None;
    for (idx, ch) in raw.char_indices() {
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
                let end = clamp_char_boundary(raw, idx);
                return raw.get(s..=end);
            }
        }
    }
    None
}

pub(crate) fn parse_json_object(raw: &str) -> Option<Value> {
    if let Ok(value) = serde_json::from_str::<Value>(raw) {
        return Some(value);
    }
    if let Some(snippet) = extract_first_json_object(raw) {
        return serde_json::from_str::<Value>(snippet).ok();
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

pub(crate) fn parse_json_object_repair(raw: &str) -> Option<Value> {
    let snippet = extract_first_json_object(raw).unwrap_or(raw);
    let normalized = normalize_json_quotes(snippet);
    let cleaned = strip_trailing_commas(&normalized);
    serde_json::from_str::<Value>(&cleaned).ok()
}

pub(crate) fn parse_json_object_with_repair(raw: &str) -> (Option<Value>, bool) {
    if let Some(value) = parse_json_object(raw) {
        return (Some(value), false);
    }
    if let Some(value) = parse_json_object_repair(raw) {
        return (Some(value), true);
    }
    (None, false)
}

pub(crate) fn repair_json_object(raw: &str) -> (Option<Value>, bool) {
    let (value_opt, repaired) = parse_json_object_with_repair(raw);
    let value = value_opt.and_then(|v| if v.is_object() { Some(v) } else { None });
    (value, repaired)
}

pub(crate) fn monologue_response_format(settings: &Settings) -> Option<Value> {
    if let Some(defaults) = settings.request_defaults.as_ref() {
        if let Some(fmt) = defaults.get("response_format") {
            return Some(fmt.clone());
        }
        if defaults
            .get("json_mode")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            return Some(json!({ "type": "json_object" }));
        }
    }
    Some(json!({ "type": "json_object" }))
}

#[derive(Debug, Clone)]
pub(crate) struct CalculatorPacket {
    pub final_text: String,
    pub required_slots: Vec<String>,
    pub assumptions: Vec<String>,
    pub defaults_used: Vec<String>,
    pub missing_slots: Vec<String>,
}

fn read_string_array(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

pub(crate) fn parse_calculator_packet(raw: &str) -> Option<CalculatorPacket> {
    let value = parse_json_object(raw)?;
    let final_text = value
        .get("final")
        .or_else(|| value.get("message"))
        .or_else(|| value.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let mut required_slots = read_string_array(&value, "required_slots");
    let missing_slots = read_string_array(&value, "missing_slots");
    if required_slots.is_empty() && !missing_slots.is_empty() {
        required_slots = missing_slots.clone();
    }
    let assumptions = read_string_array(&value, "assumptions");
    let defaults_used = read_string_array(&value, "defaults_used");
    Some(CalculatorPacket {
        final_text,
        required_slots,
        assumptions,
        defaults_used,
        missing_slots,
    })
}
