use once_cell::sync::Lazy;
use std::collections::HashSet;

static SYSTEM_LOG_EVENTS: Lazy<HashSet<String>> = Lazy::new(|| {
    let raw = include_str!("../../../shared/system_log_events.json");
    let parsed: serde_json::Value = serde_json::from_str(raw).unwrap_or_else(|_| serde_json::json!({}));
    parsed
        .get("events")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default()
});

pub fn is_known_event(event: &str) -> bool {
    SYSTEM_LOG_EVENTS.contains(event)
}

#[cfg(test)]
mod tests {
    use super::is_known_event;

    #[test]
    fn schema_contains_core_events() {
        assert!(is_known_event("memory_pass_start"));
        assert!(is_known_event("memory_pass_result"));
        assert!(is_known_event("memory_yield_report"));
        assert!(is_known_event("identity_snapshot_written"));
        assert!(is_known_event("identity_audit"));
        assert!(is_known_event("fast_path_stage_timing"));
        assert!(is_known_event("web_lookup_start"));
        assert!(is_known_event("web_lookup_result"));
    }
}
