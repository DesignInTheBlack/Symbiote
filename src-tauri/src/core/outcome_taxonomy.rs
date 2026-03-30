use once_cell::sync::Lazy;
use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct OutcomeTaxonomy {
    verdicts: HashSet<String>,
    target_types: HashSet<String>,
    sources: HashSet<String>,
    version: i64,
}

static OUTCOME_TAXONOMY: Lazy<OutcomeTaxonomy> = Lazy::new(|| {
    let raw = include_str!("../../../shared/outcome_taxonomy.json");
    let parsed: Value = serde_json::from_str(raw).unwrap_or_else(|_| serde_json::json!({}));

    let verdicts = parsed
        .get("verdicts")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default();

    let target_types = parsed
        .get("target_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default();

    let sources = parsed
        .get("sources")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default();

    let version = parsed.get("version").and_then(|v| v.as_i64()).unwrap_or(0);

    OutcomeTaxonomy {
        verdicts,
        target_types,
        sources,
        version,
    }
});

pub fn is_valid_verdict(verdict: &str) -> bool {
    OUTCOME_TAXONOMY.verdicts.contains(verdict)
}

pub fn is_valid_target_type(target_type: &str) -> bool {
    OUTCOME_TAXONOMY.target_types.contains(target_type)
}

pub fn is_valid_outcome(target_type: &str, verdict: &str) -> bool {
    is_valid_target_type(target_type) && is_valid_verdict(verdict)
}

pub fn allowed_verdicts() -> Vec<String> {
    let mut out: Vec<String> = OUTCOME_TAXONOMY.verdicts.iter().cloned().collect();
    out.sort();
    out
}

pub fn allowed_target_types() -> Vec<String> {
    let mut out: Vec<String> = OUTCOME_TAXONOMY.target_types.iter().cloned().collect();
    out.sort();
    out
}

pub fn allowed_sources() -> Vec<String> {
    let mut out: Vec<String> = OUTCOME_TAXONOMY.sources.iter().cloned().collect();
    out.sort();
    out
}

pub fn version() -> i64 {
    OUTCOME_TAXONOMY.version
}
