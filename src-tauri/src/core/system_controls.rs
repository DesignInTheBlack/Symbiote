use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use crate::models::SystemControlEntry;
use crate::models::Settings;
use crate::db::Db;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubsystemClass {
    Critical,
    Core,
    Optional,
}

#[derive(Clone, Debug, Serialize)]
pub struct SubsystemDefinition {
    pub id: &'static str,
    pub label: &'static str,
    pub class: SubsystemClass,
    pub default_mode: &'static str,
    pub supported_modes: &'static [&'static str],
    pub depends_on: &'static [&'static str],
    pub enforcement_notes: &'static str,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ControlState {
    pub subsystem_id: String,
    pub mode: String,
    #[serde(default)]
    pub value_json: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
    #[serde(default)]
    pub updated_by: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ControlChangeRequest {
    pub subsystem_id: String,
    pub mode: String,
    #[serde(default)]
    pub value_json: Option<String>,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub override_critical: bool,
}

const MODE_NORMAL: &str = "normal";
const MODE_DEGRADED: &str = "degraded";
const MODE_OFF: &str = "off";
const MODE_READ_ONLY: &str = "read_only";
const MODE_SHADOW: &str = "shadow";

static REGISTRY: &[SubsystemDefinition] = &[
    SubsystemDefinition {
        id: "kernel_loop",
        label: "Kernel loop",
        class: SubsystemClass::Critical,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &[],
        enforcement_notes: "Primary execution loop. Off disables response generation.",
    },
    SubsystemDefinition {
        id: "scheduler_tick",
        label: "Scheduler tick",
        class: SubsystemClass::Critical,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &[],
        enforcement_notes: "Scheduler cadence and tick execution.",
    },
    SubsystemDefinition {
        id: "prediction_generation",
        label: "Prediction generation",
        class: SubsystemClass::Optional,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop", "scheduler_tick"],
        enforcement_notes: "Self-prediction generation and validation.",
    },
    SubsystemDefinition {
        id: "safety_contracts",
        label: "Safety contracts",
        class: SubsystemClass::Critical,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL],
        depends_on: &[],
        enforcement_notes: "Safety contracts cannot be disabled.",
    },
    SubsystemDefinition {
        id: "monologue_loop",
        label: "Monologue loop",
        class: SubsystemClass::Core,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Internal monologue scheduling and emission.",
    },
    SubsystemDefinition {
        id: "monologue_recovery",
        label: "Monologue recovery",
        class: SubsystemClass::Core,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_OFF],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Controls unhalt behavior and recovery gating for monologue.",
    },
    SubsystemDefinition {
        id: "workspace_broadcast",
        label: "Workspace broadcast",
        class: SubsystemClass::Core,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Workspace state broadcast into prompts.",
    },
    SubsystemDefinition {
        id: "workspace_contributors",
        label: "Workspace contributors",
        class: SubsystemClass::Optional,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["workspace_broadcast"],
        enforcement_notes: "Workspace contributor snapshots and missing contributor alerts.",
    },
    SubsystemDefinition {
        id: "workspace_contributors_prompt",
        label: "Workspace contributors prompt",
        class: SubsystemClass::Optional,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["workspace_contributors"],
        enforcement_notes: "Injects workspace contributor summaries into prompts.",
    },
    SubsystemDefinition {
        id: "memory_write",
        label: "Memory write",
        class: SubsystemClass::Core,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF, MODE_READ_ONLY],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Long-term memory writes and persistence.",
    },
    SubsystemDefinition {
        id: "memory_writer_evidence_relax",
        label: "Memory writer evidence relax",
        class: SubsystemClass::Optional,
        default_mode: MODE_OFF,
        supported_modes: &[MODE_NORMAL, MODE_OFF],
        depends_on: &["memory_write"],
        enforcement_notes: "Temporarily relax evidence gating for memory writer/reflection (debug only).",
    },
    SubsystemDefinition {
        id: "memory_retrieval",
        label: "Memory retrieval",
        class: SubsystemClass::Core,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF, MODE_READ_ONLY],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Memory retrieval and context injection.",
    },
    SubsystemDefinition {
        id: "tool_execution",
        label: "Tool execution",
        class: SubsystemClass::Core,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Tool calls, dispatch, and responses.",
    },
    SubsystemDefinition {
        id: "qualia_loop",
        label: "Qualia loop",
        class: SubsystemClass::Optional,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Qualia label processing and reward updates.",
    },
    SubsystemDefinition {
        id: "qualia_auto",
        label: "Qualia auto-label",
        class: SubsystemClass::Optional,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_OFF],
        depends_on: &["qualia_loop"],
        enforcement_notes: "Auto-labels neutral qualia when no recent user labels exist.",
    },
    SubsystemDefinition {
        id: "organism_loop",
        label: "Organism loop",
        class: SubsystemClass::Optional,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Organism signal updates and decay.",
    },
    SubsystemDefinition {
        id: "cognitive_wave",
        label: "Cognitive wave field",
        class: SubsystemClass::Optional,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Fourier-domain wave field contributions and decay.",
    },
    SubsystemDefinition {
        id: "cognitive_wave_projection",
        label: "Cognitive wave projection",
        class: SubsystemClass::Optional,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["cognitive_wave"],
        enforcement_notes: "Wave projection metrics for prompts and arbitration.",
    },
    SubsystemDefinition {
        id: "attention_schema",
        label: "Attention schema",
        class: SubsystemClass::Optional,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Attention schema computation and persistence.",
    },
    SubsystemDefinition {
        id: "attention_schema_prompt",
        label: "Attention schema prompt",
        class: SubsystemClass::Optional,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["attention_schema"],
        enforcement_notes: "Injects attention schema summaries into prompts.",
    },
    SubsystemDefinition {
        id: "introspection",
        label: "Introspection",
        class: SubsystemClass::Core,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Introspection prompt and entry writes.",
    },
    SubsystemDefinition {
        id: "prediction_residual_influence",
        label: "Prediction residual influence",
        class: SubsystemClass::Optional,
        default_mode: MODE_SHADOW,
        supported_modes: &[MODE_NORMAL, MODE_SHADOW, MODE_DEGRADED, MODE_OFF],
        depends_on: &["prediction_generation"],
        enforcement_notes: "Residual influence on arbitration and wave; shadow mode by default.",
    },
    SubsystemDefinition {
        id: "audits",
        label: "Audits",
        class: SubsystemClass::Core,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Audit passes and discrepancy scoring.",
    },
    SubsystemDefinition {
        id: "voice_output",
        label: "Voice output",
        class: SubsystemClass::Optional,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_OFF],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Voice output and playback.",
    },
    SubsystemDefinition {
        id: "telemetry_sampling",
        label: "Telemetry sampling",
        class: SubsystemClass::Optional,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Telemetry and health sampling.",
    },
    SubsystemDefinition {
        id: "ui_live_refresh",
        label: "UI live refresh",
        class: SubsystemClass::Optional,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &[],
        enforcement_notes: "UI timers and polling cadence.",
    },
    SubsystemDefinition {
        id: "rolling_summary",
        label: "Rolling summary",
        class: SubsystemClass::Core,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop", "memory_write"],
        enforcement_notes: "Rolling summary updates.",
    },
    SubsystemDefinition {
        id: "inner_summary",
        label: "Inner summary",
        class: SubsystemClass::Core,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Inner summary updates.",
    },
    SubsystemDefinition {
        id: "self_memory",
        label: "Self memory",
        class: SubsystemClass::Core,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF, MODE_READ_ONLY],
        depends_on: &["kernel_loop", "memory_write"],
        enforcement_notes: "Self memory write pipeline.",
    },
    SubsystemDefinition {
        id: "memory_consolidation",
        label: "Memory consolidation",
        class: SubsystemClass::Core,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop", "memory_write"],
        enforcement_notes: "Memory consolidation passes.",
    },
    SubsystemDefinition {
        id: "episodic",
        label: "Episodic memory",
        class: SubsystemClass::Optional,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop", "memory_write"],
        enforcement_notes: "Episodic capture and compaction.",
    },
    SubsystemDefinition {
        id: "post_processing",
        label: "Post processing",
        class: SubsystemClass::Core,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Post-processing passes and sanitization.",
    },
    SubsystemDefinition {
        id: "prompt_loader",
        label: "Prompt loader",
        class: SubsystemClass::Core,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &[],
        enforcement_notes: "Prompt reloads and registry refresh.",
    },
    SubsystemDefinition {
        id: "feedback_loop",
        label: "Feedback loop",
        class: SubsystemClass::Core,
        default_mode: MODE_NORMAL,
        supported_modes: &[MODE_NORMAL, MODE_DEGRADED, MODE_OFF],
        depends_on: &["kernel_loop"],
        enforcement_notes: "Feedback detection and alignment signals.",
    },
];

pub fn registry() -> &'static [SubsystemDefinition] {
    REGISTRY
}

pub fn registry_map() -> HashMap<&'static str, &'static SubsystemDefinition> {
    let mut map = HashMap::new();
    for def in REGISTRY {
        map.insert(def.id, def);
    }
    map
}

pub fn default_mode_for(subsystem_id: &str) -> Option<&'static str> {
    registry().iter().find(|def| def.id == subsystem_id).map(|def| def.default_mode)
}

pub fn validate_change(
    request: &ControlChangeRequest,
    current_states: &HashMap<String, ControlState>,
) -> Result<(), String> {
    let registry = registry_map();
    let def = registry
        .get(request.subsystem_id.as_str())
        .ok_or_else(|| format!("Unknown subsystem: {}", request.subsystem_id))?;

    let reason = request.reason.as_deref().unwrap_or("").trim();
    if reason.is_empty() {
        return Err("Reason is required for control changes.".to_string());
    }

    let requested_mode = request.mode.to_lowercase();
    if !def.supported_modes.iter().any(|mode| mode == &requested_mode.as_str()) {
        return Err(format!(
            "Mode '{}' not supported for subsystem '{}'.",
            requested_mode, def.id
        ));
    }

    if def.class == SubsystemClass::Critical && requested_mode == MODE_OFF && !request.override_critical {
        return Err(format!(
            "Subsystem '{}' is critical and cannot be disabled without override.",
            def.id
        ));
    }

    if requested_mode != MODE_OFF {
        for dependency in def.depends_on.iter() {
            let dep_mode = current_states
                .get(&dependency.to_string())
                .map(|state| state.mode.to_lowercase())
                .unwrap_or_else(|| default_mode_for(dependency).unwrap_or(MODE_NORMAL).to_string());
            if dep_mode == MODE_OFF {
                return Err(format!(
                    "Subsystem '{}' depends on '{}' which is off.",
                    def.id, dependency
                ));
            }
        }
    }

    Ok(())
}

pub fn normalize_mode(mode: &str) -> String {
    match mode.trim().to_lowercase().as_str() {
        MODE_DEGRADED => MODE_DEGRADED.to_string(),
        MODE_OFF => MODE_OFF.to_string(),
        MODE_READ_ONLY => MODE_READ_ONLY.to_string(),
        MODE_SHADOW => MODE_SHADOW.to_string(),
        _ => MODE_NORMAL.to_string(),
    }
}

pub fn mode_is_off(mode: &str) -> bool {
    mode.trim().eq_ignore_ascii_case(MODE_OFF)
}

pub fn mode_is_degraded(mode: &str) -> bool {
    mode.trim().eq_ignore_ascii_case(MODE_DEGRADED)
}

pub fn mode_is_shadow(mode: &str) -> bool {
    mode.trim().eq_ignore_ascii_case(MODE_SHADOW)
}

pub fn is_subsystem_enabled(subsystem_id: &str, controls: &HashMap<String, ControlState>) -> bool {
    let mode = mode_for(subsystem_id, controls);
    !mode_is_off(&mode)
}

pub fn mode_is_read_only(mode: &str) -> bool {
    mode.trim().eq_ignore_ascii_case(MODE_READ_ONLY)
}

pub fn is_high_confidence_reason(reason_code: &str) -> bool {
    let lowered = reason_code.to_lowercase();
    lowered.contains("evidence")
        || lowered.contains("critical")
        || lowered.contains("safety")
        || lowered.contains("high_confidence")
        || lowered.contains("verified")
        || lowered.contains("tool_outcome")
        || lowered.contains("memory_writer_evidence")
        || lowered.contains("memory_api_write")
}

pub fn allow_memory_write(mode: &str, reason_code: &str) -> bool {
    if mode_is_off(mode) || mode_is_read_only(mode) {
        return false;
    }
    if mode_is_degraded(mode) {
        return is_high_confidence_reason(reason_code);
    }
    true
}

pub fn mode_for(subsystem_id: &str, controls: &HashMap<String, ControlState>) -> String {
    let mut visited = HashSet::new();
    resolve_mode(subsystem_id, controls, &mut visited)
}

fn resolve_mode(
    subsystem_id: &str,
    controls: &HashMap<String, ControlState>,
    visited: &mut HashSet<String>,
) -> String {
    if visited.contains(subsystem_id) {
        return controls
            .get(subsystem_id)
            .map(|state| normalize_mode(&state.mode))
            .unwrap_or_else(|| {
                default_mode_for(subsystem_id)
                    .unwrap_or(MODE_NORMAL)
                    .to_string()
            });
    }
    visited.insert(subsystem_id.to_string());

    let mode = controls
        .get(subsystem_id)
        .map(|state| normalize_mode(&state.mode))
        .unwrap_or_else(|| {
            default_mode_for(subsystem_id)
                .unwrap_or(MODE_NORMAL)
                .to_string()
        });
    if mode_is_off(&mode) {
        return mode;
    }

    if let Some(def) = registry_map().get(subsystem_id) {
        for dependency in def.depends_on.iter() {
            let dep_mode = resolve_mode(dependency, controls, visited);
            if mode_is_off(&dep_mode) {
                return MODE_OFF.to_string();
            }
        }
    }

    mode
}

pub fn map_from_entries(entries: &[SystemControlEntry]) -> HashMap<String, ControlState> {
    let mut map = HashMap::new();
    for entry in entries {
        map.insert(
            entry.subsystem_id.clone(),
            ControlState {
                subsystem_id: entry.subsystem_id.clone(),
                mode: entry.mode.clone(),
                value_json: entry.value_json.clone(),
                updated_at: Some(entry.updated_at.clone()),
                updated_by: entry.updated_by.clone(),
                reason: entry.reason.clone(),
            },
        );
    }
    map
}

pub async fn load_control_map(db: &Db) -> HashMap<String, ControlState> {
    match db.get_system_controls().await {
        Ok(entries) => map_from_entries(&entries),
        Err(_) => HashMap::new(),
    }
}

pub async fn apply_recommendation_action(
    db: &Db,
    action: &Value,
    actor: &str,
) -> Result<String, String> {
    let action_type = action
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "recommendation_action_missing_type".to_string())?;
    match action_type {
        "set_system_control" => {
            let subsystem_id = action
                .get("subsystem_id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "recommendation_action_missing_subsystem".to_string())?;
            let mode = action
                .get("mode")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "recommendation_action_missing_mode".to_string())?;
            let reason = action
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("recommendation");
            let override_critical = action
                .get("override_critical")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let existing = db.get_system_controls().await.map_err(|e| e.to_string())?;
            let current_states = map_from_entries(&existing);
            let request = ControlChangeRequest {
                subsystem_id: subsystem_id.to_string(),
                mode: mode.to_string(),
                value_json: action.get("value_json").and_then(|v| v.as_str()).map(|s| s.to_string()),
                actor: Some(actor.to_string()),
                reason: Some(reason.to_string()),
                override_critical,
            };
            validate_change(&request, &current_states)?;
            db.insert_system_control_event(
                subsystem_id,
                current_states.get(subsystem_id).map(|s| s.mode.clone()),
                &normalize_mode(mode),
                request.value_json.clone(),
                Some(actor.to_string()),
                Some(reason.to_string()),
                "accepted",
            )
            .await
            .map_err(|e| e.to_string())?;
            db.set_system_control(
                subsystem_id,
                &normalize_mode(mode),
                request.value_json.clone(),
                Some(actor.to_string()),
                Some(reason.to_string()),
            )
            .await
            .map_err(|e| e.to_string())?;
            Ok(format!("system_control:{}={}", subsystem_id, normalize_mode(mode)))
        }
        "update_settings" => {
            let patch = action
                .get("settings")
                .ok_or_else(|| "recommendation_action_missing_settings".to_string())?;
            let mut current = db.get_settings().await.map_err(|e| e.to_string())?;
            apply_settings_patch(&mut current, patch)?;
            db.update_settings(current)
                .await
                .map_err(|e| e.to_string())?;
            Ok("settings_updated".to_string())
        }
        _ => Err(format!("recommendation_action_unknown: {}", action_type)),
    }
}

fn apply_settings_patch(settings: &mut Settings, patch: &Value) -> Result<(), String> {
    let obj = patch
        .as_object()
        .ok_or_else(|| "settings_patch_invalid".to_string())?;
    for (key, value) in obj.iter() {
        match key.as_str() {
            "evidence_auto_capture" => {
                settings.evidence_auto_capture = value.as_bool();
            }
            "evidence_emit_budget" => {
                settings.evidence_emit_budget = value.as_i64().map(|v| v as i32);
            }
            "planner_enabled" => {
                settings.planner_enabled = value.as_bool();
            }
            "confidence_calibration" => {
                settings.confidence_calibration = value.as_bool();
            }
            "gate_penalty_integration" => {
                settings.gate_penalty_integration = value.as_bool();
            }
            "scheduler_cognition" => {
                settings.scheduler_cognition = value.as_bool();
            }
            "learning_feedback" => {
                settings.learning_feedback = value.as_bool();
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_state(id: &str, mode: &str) -> ControlState {
        ControlState {
            subsystem_id: id.to_string(),
            mode: mode.to_string(),
            value_json: None,
            updated_at: None,
            updated_by: None,
            reason: Some("test".to_string()),
        }
    }

    #[test]
    fn validate_change_rejects_missing_reason() {
        let mut states = HashMap::new();
        states.insert("kernel_loop".to_string(), base_state("kernel_loop", "normal"));
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
    fn validate_change_blocks_critical_off_without_override() {
        let mut states = HashMap::new();
        states.insert("kernel_loop".to_string(), base_state("kernel_loop", "normal"));
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
    fn validate_change_allows_critical_off_with_override() {
        let mut states = HashMap::new();
        states.insert("kernel_loop".to_string(), base_state("kernel_loop", "normal"));
        let request = ControlChangeRequest {
            subsystem_id: "kernel_loop".to_string(),
            mode: "off".to_string(),
            value_json: None,
            actor: None,
            reason: Some("maintenance".to_string()),
            override_critical: true,
        };
        assert!(validate_change(&request, &states).is_ok());
    }

    #[test]
    fn validate_change_blocks_dependency_off() {
        let mut states = HashMap::new();
        states.insert("kernel_loop".to_string(), base_state("kernel_loop", "off"));
        let request = ControlChangeRequest {
            subsystem_id: "monologue_loop".to_string(),
            mode: "normal".to_string(),
            value_json: None,
            actor: None,
            reason: Some("test".to_string()),
            override_critical: false,
        };
        assert!(validate_change(&request, &states).is_err());
    }
}
