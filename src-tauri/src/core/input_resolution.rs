use serde_json::Value;
use std::collections::BTreeMap;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct ResolvedParam {
    pub value: Option<Value>,
    pub source: String,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct ResolutionResult {
    pub resolved: BTreeMap<String, ResolvedParam>,
    pub missing: Vec<String>,
    pub slot_provenance: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct InputResolutionContext {
    pub current_input: String,
    pub recent_text: Vec<String>,
    pub kv: BTreeMap<String, String>,
    pub registry: Value,
    pub defaults: Option<Value>,
}

fn lookup_json_path(root: &Value, path: &str) -> Option<Value> {
    let mut current = root;
    for part in path.split('.') {
        if part.is_empty() {
            continue;
        }
        match current {
            Value::Object(map) => {
                current = map.get(part)?;
            }
            _ => return None,
        }
    }
    Some(current.clone())
}

fn extract_numeric_for_label(text: &str, label: &str) -> Option<Value> {
    if label.is_empty() {
        return None;
    }
    let escaped = regex::escape(label);
    let pattern = format!(r"(?i){}\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", escaped);
    if let Ok(re) = Regex::new(&pattern) {
        if let Some(cap) = re.captures(text) {
            if let Some(raw) = cap.get(1) {
                if let Ok(num) = raw.as_str().parse::<f64>() {
                    return Some(Value::from(num));
                }
            }
        }
    }
    None
}

fn extract_value_from_text(text: &str, key: &str) -> Option<Value> {
    if let Some(val) = extract_numeric_for_label(text, key) {
        return Some(val);
    }
    if let Some(last) = key.split('.').last() {
        if last != key {
            if let Some(val) = extract_numeric_for_label(text, last) {
                return Some(val);
            }
        }
    }
    None
}

pub fn resolve_param(key: &str, ctx: &InputResolutionContext) -> ResolvedParam {
    if let Some(val) = extract_value_from_text(&ctx.current_input, key) {
        return ResolvedParam { value: Some(val), source: "explicit_user".to_string(), confidence: 1.0 };
    }

    for snippet in &ctx.recent_text {
        if let Some(val) = extract_value_from_text(snippet, key) {
            return ResolvedParam { value: Some(val), source: "prior_turn".to_string(), confidence: 0.85 };
        }
    }

    if let Some(val) = ctx.kv.get(key) {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            if let Ok(num) = trimmed.parse::<f64>() {
                return ResolvedParam { value: Some(Value::from(num)), source: "memory".to_string(), confidence: 0.7 };
            }
            return ResolvedParam { value: Some(Value::from(trimmed)), source: "memory".to_string(), confidence: 0.7 };
        }
    }

    if let Some(val) = lookup_json_path(&ctx.registry, key) {
        return ResolvedParam { value: Some(val), source: "registry".to_string(), confidence: 0.6 };
    }

    if let Some(defaults) = ctx.defaults.as_ref() {
        if let Some(val) = lookup_json_path(defaults, key) {
            return ResolvedParam { value: Some(val), source: "seed_default".to_string(), confidence: 0.5 };
        }
    }

    ResolvedParam { value: None, source: "missing".to_string(), confidence: 0.0 }
}

pub fn resolve_required_slots(keys: &[String], ctx: &InputResolutionContext) -> ResolutionResult {
    let mut resolved = BTreeMap::new();
    let mut missing = Vec::new();
    let mut provenance = BTreeMap::new();

    for key in keys {
        let res = resolve_param(key, ctx);
        if res.value.is_some() {
            provenance.insert(key.clone(), res.source.clone());
            resolved.insert(key.clone(), res);
        } else {
            missing.push(key.clone());
        }
    }

    ResolutionResult { resolved, missing, slot_provenance: provenance }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn resolve_param_prefers_current_input() {
        let mut kv = BTreeMap::new();
        kv.insert("alpha".to_string(), "0.2".to_string());
        let ctx = InputResolutionContext {
            current_input: "alpha: 0.9".to_string(),
            recent_text: vec!["alpha: 0.4".to_string()],
            kv,
            registry: json!({"alpha": 0.1}),
            defaults: Some(json!({"alpha": 0.05})),
        };
        let res = resolve_param("alpha", &ctx);
        assert_eq!(res.source, "explicit_user");
        assert_eq!(res.value, Some(json!(0.9)));
    }

    #[test]
    fn resolve_required_slots_tracks_missing() {
        let ctx = InputResolutionContext {
            current_input: "".to_string(),
            recent_text: vec![],
            kv: BTreeMap::new(),
            registry: json!({}),
            defaults: Some(json!({"beta": 1.2})),
        };
        let keys = vec!["alpha".to_string(), "beta".to_string()];
        let res = resolve_required_slots(&keys, &ctx);
        assert!(res.resolved.contains_key("beta"));
        assert!(res.missing.contains(&"alpha".to_string()));
        assert_eq!(res.slot_provenance.get("beta").map(String::as_str), Some("seed_default"));
    }
}
