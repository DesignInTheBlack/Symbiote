use symbiote_lib::core::system_controls::{registry, ControlChangeRequest, ControlState, validate_change, mode_for};
use std::collections::HashMap;

#[test]
fn registry_includes_core_subsystems() {
    let ids: Vec<&str> = registry().iter().map(|def| def.id).collect();
    assert!(ids.contains(&"kernel_loop"));
    assert!(ids.contains(&"scheduler_tick"));
    assert!(ids.contains(&"prompt_loader"));
    assert!(ids.contains(&"feedback_loop"));
}

#[test]
fn validate_change_requires_reason() {
    let mut states = HashMap::new();
    states.insert(
        "kernel_loop".to_string(),
        ControlState {
            subsystem_id: "kernel_loop".to_string(),
            mode: "normal".to_string(),
            value_json: None,
            updated_at: None,
            updated_by: None,
            reason: Some("test".to_string()),
        },
    );
    let request = ControlChangeRequest {
        subsystem_id: "kernel_loop".to_string(),
        mode: "degraded".to_string(),
        value_json: None,
        actor: None,
        reason: Some("".to_string()),
        override_critical: false,
    };
    assert!(validate_change(&request, &states).is_err());
}

#[test]
fn validate_change_blocks_critical_disable_without_override() {
    let mut states = HashMap::new();
    states.insert(
        "kernel_loop".to_string(),
        ControlState {
            subsystem_id: "kernel_loop".to_string(),
            mode: "normal".to_string(),
            value_json: None,
            updated_at: None,
            updated_by: None,
            reason: Some("test".to_string()),
        },
    );
    let request = ControlChangeRequest {
        subsystem_id: "kernel_loop".to_string(),
        mode: "off".to_string(),
        value_json: None,
        actor: None,
        reason: Some("maintenance".to_string()),
        override_critical: false,
    };
    assert!(validate_change(&request, &states).is_err());
}

#[test]
fn mode_for_cascades_dependency_off() {
    let mut states = HashMap::new();
    states.insert(
        "kernel_loop".to_string(),
        ControlState {
            subsystem_id: "kernel_loop".to_string(),
            mode: "off".to_string(),
            value_json: None,
            updated_at: None,
            updated_by: None,
            reason: Some("test".to_string()),
        },
    );
    states.insert(
        "monologue_loop".to_string(),
        ControlState {
            subsystem_id: "monologue_loop".to_string(),
            mode: "normal".to_string(),
            value_json: None,
            updated_at: None,
            updated_by: None,
            reason: Some("test".to_string()),
        },
    );
    let effective = mode_for("monologue_loop", &states);
    assert_eq!(effective, "off");
}
