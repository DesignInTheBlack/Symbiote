use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TelemetrySnapshotEntry {
    pub value: f32,
    pub belief_id: Option<i64>,
    pub last_evidence_at: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct FeedbackBundleOutput {
    pub prompt_text: String,
    pub qualia_snapshot: String,
    pub payload: Value,
}

pub(crate) fn build_telemetry_snapshot(
    metrics: &crate::core::self_model_controller::SelfEvidenceMetrics,
) -> BTreeMap<String, TelemetrySnapshotEntry> {
    let mut snapshot = BTreeMap::new();
    for (key, point) in metrics.telemetry_values.iter() {
        if let Ok(value) = point.value.trim().parse::<f32>() {
            let belief_id = if point.belief_id > 0 {
                Some(point.belief_id)
            } else {
                None
            };
            snapshot.insert(
                key.to_string(),
                TelemetrySnapshotEntry {
                    value,
                    belief_id,
                    last_evidence_at: point.last_evidence_at.clone(),
                },
            );
        }
    }
    snapshot
}

pub(crate) fn format_telemetry_snapshot(snapshot: &BTreeMap<String, TelemetrySnapshotEntry>) -> String {
    if snapshot.is_empty() {
        return "None".to_string();
    }
    snapshot
        .iter()
        .map(|(k, entry)| {
            if let Some(belief_id) = entry.belief_id {
                format!("{}={:.3} (belief_id={})", k, entry.value, belief_id)
            } else {
                format!("{}={:.3}", k, entry.value)
            }
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

pub(crate) fn map_outcome_quality(outcome: Option<f32>) -> &'static str {
    let value = outcome.unwrap_or(0.0);
    if value >= 0.75 {
        "success"
    } else if value >= 0.4 {
        "partial"
    } else {
        "fail"
    }
}

pub(crate) fn map_confidence_level(confidence: f32) -> &'static str {
    if confidence >= 0.75 {
        "high"
    } else if confidence >= 0.45 {
        "medium"
    } else {
        "low"
    }
}

pub(crate) fn map_evidence_coverage(coverage: f32) -> &'static str {
    if coverage >= 0.7 {
        "full"
    } else if coverage >= 0.3 {
        "partial"
    } else {
        "none"
    }
}

pub(crate) fn map_policy_adherence(decision: Option<&str>) -> &'static str {
    match decision.unwrap_or("") {
        "DENY" => "fail",
        "VERIFY" | "DEFER" | "ALLOW_WITH_NOTICE" | "ALLOW_WITH_AUDIT" => "warn",
        "ALLOW" => "ok",
        _ => "unknown",
    }
}

pub(crate) fn parse_gate_reasons(raw: &str) -> Vec<String> {
    serde_json::from_str::<Value>(raw)
        .ok()
        .and_then(|value| {
            value.get("reasons").and_then(|reasons| {
                reasons.as_array().map(|arr| {
                    arr.iter()
                        .filter_map(|item| item.as_str().map(|s| s.to_string()))
                        .collect::<Vec<_>>()
                })
            })
        })
        .unwrap_or_default()
}

pub(crate) fn format_qualia_snapshot(state: &qualia::QualiaState) -> String {
    let mut lines = Vec::new();
    lines.push(format!("timestamp: {}", state.timestamp));
    lines.push(format!(
        "dominant_tag: {}",
        state.dominant_tag.as_deref().unwrap_or("none")
    ));
    lines.push(format!("dominant_intensity: {:.3}", state.dominant_intensity));
    lines.push(format!(
        "last_reward: {}",
        state
            .last_reward
            .map(|value| format!("{:.3}", value))
            .unwrap_or_else(|| "none".to_string())
    ));
    lines.push(format!(
        "predicted_tag: {}",
        state.predicted_tag.as_deref().unwrap_or("none")
    ));
    lines.push(format!("prediction_confidence: {:.3}", state.prediction_confidence));
    if !state.matched_workspace_refs.is_empty() {
        lines.push(format!(
            "matched_workspace_refs: {}",
            state.matched_workspace_refs.join(", ")
        ));
    }
    if !state.recent_labels.is_empty() {
        let recent = state
            .recent_labels
            .iter()
            .take(5)
            .map(|label| format!("{}:{:.2}@{}", label.tag, label.intensity, label.created_at))
            .collect::<Vec<_>>()
            .join(" | ");
        lines.push(format!("recent_labels: {}", recent));
    }
    lines.join("\n")
}
