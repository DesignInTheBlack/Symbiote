use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{Row, SqlitePool};
use std::collections::{HashMap, HashSet};

use crate::models::{
    ControllerGate,
    ControllerState,
    GoalStackItem,
    SelfModel,
    WorkspaceFieldMeta,
    WorkspaceListItemMeta,
    WorkspaceHypothesis,
    WorkspaceMeta,
};
use crate::db::Db;
use crate::core::kernel::KernelState;
use crate::core::memory::types::Scope;
use crate::core::cognitive_wave::{AmplitudeBounds, DecayProfile, WaveBand, WaveContributionInput};
use crate::core::memory::retrieval;
use crate::core::qualia;
use num_complex::Complex32;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SelfEvidenceMetrics {
    pub persona_values: HashMap<String, EvidencePoint>,
    pub goal_values: Vec<EvidencePoint>,
    pub goal_removed: Vec<EvidencePoint>,
    pub telemetry_values: HashMap<String, EvidencePoint>,
    pub source_counts: HashMap<String, i64>,
    pub avg_confidence: f32,
    pub evidence_coverage: f32,
    pub telemetry_coverage: f32,
    pub missing_fields: Vec<String>,
    pub conflict_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidencePoint {
    pub belief_id: i64,
    pub key: String,
    pub value: String,
    pub confidence: f32,
    pub last_evidence_at: Option<String>,
    pub source_quality: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReconstructionOutput {
    pub controller_state: ControllerState,
    pub reconstructed_persona: Value,
    pub reconstructed_goals: Value,
}

const PERSONA_KEYS: [&str; 5] = [
    "persona.tone",
    "persona.verbosity",
    "persona.directness",
    "persona.formality",
    "persona.initiative",
];
const TELEMETRY_KEYS: [&str; 13] = [
    "telemetry.tool_success_rate",
    "telemetry.tool_failure_rate",
    "telemetry.memory_pass_rate",
    "telemetry.clarification_rate",
    "telemetry.refusal_rate",
    "telemetry.user_feedback_pushback_rate",
    "telemetry.user_feedback_clarify_rate",
    "telemetry.user_feedback_followup_rate",
    "telemetry.user_feedback_agree_rate",
    "telemetry.user_feedback_disengage_rate",
    "telemetry.prediction_divergence_rate",
    "telemetry.prediction_divergence_persisted_rate",
    "telemetry.prediction_divergence_resolved_rate",
];
const PERSONA_MAX_AGE_DAYS: i64 = 30;
const TELEMETRY_MAX_AGE_DAYS: i64 = 7;
const TELEMETRY_MIN_COVERAGE: f32 = 0.5;
const EVIDENCE_MIN_COVERAGE: f32 = 0.6;

pub async fn telemetry_keys_present(pool: &SqlitePool) -> Result<bool, String> {
    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM kv_store WHERE key LIKE 'telemetry.%'",
    )
    .fetch_one(pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(count > 0)
}

pub async fn collect_self_evidence_metrics(pool: &SqlitePool) -> Result<SelfEvidenceMetrics, String> {
    let scope = serde_json::to_string(&Scope::SelfScope).unwrap_or_else(|_| "\"self\"".to_string());
    let assistant_id: Option<i64> = sqlx::query_scalar(
        "SELECT entity_id FROM ics_session_bindings WHERE session_id = 'default' AND ref_text = 'assistant' LIMIT 1"
    )
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    let assistant_id = assistant_id.unwrap_or(0);

    let rows = sqlx::query(
        "SELECT b.id as belief_id, b.confidence, b.last_evidence_at, f.key, f.value_literal
         FROM ics_beliefs b
         JOIN ics_fact_beliefs f ON b.id = f.belief_id
         WHERE b.scope = ? AND b.status = 'active' AND f.subject_entity_id = ?"
    )
    .bind(&scope)
    .bind(assistant_id)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    let telemetry_rows = sqlx::query(
        "SELECT key, value, strftime('%Y-%m-%dT%H:%M:%SZ', updated_at) as updated_at
         FROM kv_store
         WHERE key LIKE 'telemetry.%'"
    )
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    struct RawPoint {
        belief_id: i64,
        key: String,
        value: String,
        confidence: f32,
        last_evidence_at: Option<String>,
    }
    let mut raw_points = Vec::new();
    for row in rows {
        let belief_id: i64 = row.get("belief_id");
        raw_points.push(RawPoint {
            belief_id,
            key: row.get("key"),
            value: row.get("value_literal"),
            confidence: row.try_get::<f64, _>("confidence").unwrap_or(0.0) as f32,
            last_evidence_at: row.get("last_evidence_at"),
        });
    }

    let mut source_counts: HashMap<String, i64> = HashMap::new();
    let mut source_quality_map: HashMap<i64, f32> = HashMap::new();
    if !raw_points.is_empty() {
        let placeholders = raw_points.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let query = format!(
            "SELECT belief_id, source_type, weight FROM ics_evidence_events WHERE belief_id IN ({})",
            placeholders
        );
        let mut stmt = sqlx::query(&query);
        for point in raw_points.iter() {
            stmt = stmt.bind(point.belief_id);
        }
        let evidence_rows = stmt.fetch_all(pool).await.unwrap_or_default();
        for row in evidence_rows {
            let belief_id: i64 = row.get("belief_id");
            let source_raw: String = row.try_get("source_type").unwrap_or_default();
            let weight: f32 = row.try_get::<f64, _>("weight").unwrap_or(1.0) as f32;
            let normalized = source_raw.trim().to_lowercase();
            *source_counts.entry(normalized.clone()).or_insert(0) += 1;
            let quality = (source_quality_for(&normalized) * weight.clamp(0.0, 1.0)).clamp(0.0, 1.0);
            source_quality_map
                .entry(belief_id)
                .and_modify(|v| {
                    if quality > *v {
                        *v = quality;
                    }
                })
                .or_insert(quality);
        }
    }

    let mut persona_values: HashMap<String, EvidencePoint> = HashMap::new();
    let mut goal_values = Vec::new();
    let mut goal_removed = Vec::new();
    let mut confidences = Vec::new();

    for raw in raw_points {
        let key = raw.key.clone();
        let value = raw.value.clone();
        let confidence = raw.confidence;
        let last_evidence_at = raw.last_evidence_at.clone();
        let source_quality = source_quality_map.get(&raw.belief_id).copied().unwrap_or(0.5);
        let point = EvidencePoint {
            belief_id: raw.belief_id,
            key: key.clone(),
            value: value.clone(),
            confidence,
            last_evidence_at,
            source_quality,
        };

        if PERSONA_KEYS.iter().any(|k| k.eq_ignore_ascii_case(&key)) {
            let replace = match persona_values.get(&key) {
                None => true,
                Some(existing) => newer_than(&point.last_evidence_at, &existing.last_evidence_at),
            };
            if replace {
                persona_values.insert(key, point);
            }
            confidences.push(confidence);
            continue;
        }

        if key.eq_ignore_ascii_case("goal") {
            goal_values.push(point);
            confidences.push(confidence);
            continue;
        }

        if key.eq_ignore_ascii_case("goal_removed") {
            goal_removed.push(point);
            confidences.push(confidence);
            continue;
        }
    }

    let mut telemetry_values: HashMap<String, EvidencePoint> = HashMap::new();
    for row in telemetry_rows {
        let key: String = row.get("key");
        let value: Option<String> = row.try_get("value").ok();
        let last_evidence_at: Option<String> = row
            .try_get::<String, _>("updated_at")
            .ok();
        let mut confidence = 1.0;
        if let Some(ts) = last_evidence_at.as_ref() {
            let recency = recency_score(&Some(ts.clone()), TELEMETRY_MAX_AGE_DAYS);
            if recency <= 0.0 {
                continue;
            }
            confidence = (0.5 + 0.5 * recency).clamp(0.0, 1.0);
        }
        let belief_id: i64 = 0;
        let point = EvidencePoint {
            belief_id,
            key: key.clone(),
            value: value.unwrap_or_default(),
            confidence,
            last_evidence_at,
            source_quality: 0.7,
        };
        let replace = match telemetry_values.get(&key) {
            None => true,
            Some(existing) => newer_than(&point.last_evidence_at, &existing.last_evidence_at),
        };
        if replace {
            telemetry_values.insert(key, point);
        }
    }

    let avg_confidence = if confidences.is_empty() {
        0.2
    } else {
        let sum: f32 = confidences.iter().sum();
        (sum / confidences.len() as f32).clamp(0.0, 1.0)
    };

    let mut missing_fields = Vec::new();
    let mut evidence_score = 0.0;
    for key in PERSONA_KEYS {
        if let Some(point) = persona_values.get(key) {
            evidence_score += recency_score(&point.last_evidence_at, PERSONA_MAX_AGE_DAYS) * point.source_quality;
        } else {
            missing_fields.push(key.to_string());
        }
    }

    let goals_present = !goal_values.is_empty();
    if goals_present {
        let latest_goal = goal_values
            .iter()
            .max_by(|a, b| a.last_evidence_at.cmp(&b.last_evidence_at));
        if let Some(point) = latest_goal {
            evidence_score += recency_score(&point.last_evidence_at, PERSONA_MAX_AGE_DAYS) * point.source_quality;
        }
    } else {
        missing_fields.push("goals".to_string());
    }

    let total_required = PERSONA_KEYS.len() + 1;
    let evidence_coverage = if total_required == 0 {
        0.0
    } else {
        (evidence_score / total_required as f32).clamp(0.0, 1.0)
    };

    let mut telemetry_missing = Vec::new();
    let mut telemetry_score = 0.0;
    for key in TELEMETRY_KEYS {
        if let Some(point) = telemetry_values.get(key) {
            telemetry_score += recency_score(&point.last_evidence_at, TELEMETRY_MAX_AGE_DAYS);
        } else {
            telemetry_missing.push(key.to_string());
        }
    }
    let telemetry_coverage = if TELEMETRY_KEYS.is_empty() {
        0.0
    } else {
        (telemetry_score / TELEMETRY_KEYS.len() as f32).clamp(0.0, 1.0)
    };
    missing_fields.extend(telemetry_missing);

    let conflict_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(DISTINCT cs.id)
         FROM ics_conflict_sets cs
         JOIN ics_conflict_set_members m ON m.conflict_set_id = cs.id
         JOIN ics_beliefs b ON b.id = m.belief_id
         WHERE cs.status = 'open' AND b.scope = ?"
    )
    .bind(&scope)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?
    .unwrap_or(0);

    Ok(SelfEvidenceMetrics {
        persona_values,
        goal_values,
        goal_removed,
        telemetry_values,
        source_counts,
        avg_confidence,
        evidence_coverage,
        telemetry_coverage,
        missing_fields,
        conflict_count,
    })
}

pub fn wave_contribution_from_controller_state(state: &ControllerState) -> WaveContributionInput {
    let confidence = state.confidence.clamp(0.0, 1.0);
    let uncertainty = state.uncertainty.clamp(0.0, 1.0);
    let drift = state.drift_score.clamp(0.0, 1.0);
    let autonomy = state.autonomy_level.clamp(0.0, 1.0);
    let evidence = state.evidence_coverage.clamp(0.0, 1.0);
    let telemetry = state.telemetry_coverage.clamp(0.0, 1.0);
    let failure = (state.failure_streak.max(0) as f32 / 10.0).clamp(0.0, 1.0);
    let coeffs = vec![
        Complex32::new(confidence, uncertainty),
        Complex32::new(drift, failure),
        Complex32::new(autonomy, evidence),
        Complex32::new(telemetry, drift),
    ];
    let amplitude = (0.2 + (confidence * 0.5) - (uncertainty * 0.2)).clamp(0.05, 0.8);
    WaveContributionInput {
        source: "self_model_controller",
        band: WaveBand::SelfModel,
        coeffs,
        amplitude,
        amplitude_bounds: AmplitudeBounds::new(0.05, 0.8),
        decay_profile: DecayProfile::for_band(WaveBand::SelfModel),
    }
}

fn telemetry_value(metrics: &SelfEvidenceMetrics, key: &str) -> f32 {
    metrics
        .telemetry_values
        .get(key)
        .and_then(|point| point.value.parse::<f32>().ok())
        .unwrap_or(0.0)
}

fn feedback_signal(metrics: &SelfEvidenceMetrics) -> f32 {
    let pushback = telemetry_value(metrics, "telemetry.user_feedback_pushback_rate");
    let clarify = telemetry_value(metrics, "telemetry.user_feedback_clarify_rate");
    let follow_up = telemetry_value(metrics, "telemetry.user_feedback_followup_rate");
    let agree = telemetry_value(metrics, "telemetry.user_feedback_agree_rate");
    let disengage = telemetry_value(metrics, "telemetry.user_feedback_disengage_rate");
    (agree + 0.5 * follow_up - pushback - clarify - disengage).clamp(-1.0, 1.0)
}

pub fn reconstruct_from_metrics(
    metrics: &SelfEvidenceMetrics,
    current: &SelfModel,
    failure_streak: i32,
) -> ReconstructionOutput {
    let mut persona = current.persona.clone();
    let mut updated_persona = false;
    for key in PERSONA_KEYS {
        if let Some(point) = metrics.persona_values.get(key) {
            if let Ok(val) = point.value.trim().parse::<f32>() {
                let axis = key.split('.').last().unwrap_or(key);
                if let Some(obj) = persona.as_object_mut() {
                    obj.insert(axis.to_string(), Value::from(val.clamp(0.0, 1.0)));
                    updated_persona = true;
                }
            }
        }
    }

    if !updated_persona && persona.is_null() {
        persona = serde_json::json!({
            "tone": 0.55,
            "verbosity": 0.45,
            "directness": 0.7,
            "formality": 0.5,
            "initiative": 0.5,
        });
    }

    let mut removed = HashSet::new();
    for point in &metrics.goal_removed {
        removed.insert(point.value.clone());
    }
    let mut goals = Vec::new();
    let mut seen = HashSet::new();
    let mut goal_values = metrics.goal_values.clone();
    goal_values.sort_by(|a, b| b.last_evidence_at.cmp(&a.last_evidence_at));
    for point in goal_values {
        if removed.contains(&point.value) {
            continue;
        }
        if seen.insert(point.value.clone()) {
            goals.push(point.value);
        }
        if goals.len() >= 3 {
            break;
        }
    }

    let reconstructed_goals = if goals.is_empty() {
        current.goals.clone()
    } else {
        Value::from(goals)
    };

    let divergence = telemetry_value(metrics, "telemetry.prediction_divergence_rate");
    let divergence_persisted = telemetry_value(metrics, "telemetry.prediction_divergence_persisted_rate");
    let divergence_resolved = telemetry_value(metrics, "telemetry.prediction_divergence_resolved_rate");
    let divergence_pressure = (divergence + 0.5 * divergence_persisted - 0.5 * divergence_resolved)
        .clamp(0.0, 1.0);

    let drift_score = (metrics.conflict_count as f32 / 3.0 + divergence_pressure).clamp(0.0, 1.0);
    let feedback_adjustment = feedback_signal(&metrics).clamp(-0.2, 0.2);
    let confidence = (metrics.avg_confidence + feedback_adjustment - 0.2 * divergence_pressure).clamp(0.0, 1.0);
    let mut autonomy_level = (confidence - 0.5 * drift_score - 0.05 * failure_streak as f32).clamp(0.1, 1.0);
    let telemetry_ok = metrics.telemetry_coverage >= TELEMETRY_MIN_COVERAGE;
    let evidence_ok = metrics.evidence_coverage >= EVIDENCE_MIN_COVERAGE;
    let mut notes = Vec::new();
    if !evidence_ok {
        autonomy_level = autonomy_level.min(0.25);
        notes.push("autonomy_clamped:evidence_low".to_string());
    }
    if !telemetry_ok {
        autonomy_level = autonomy_level.min(0.3);
        notes.push("autonomy_clamped:telemetry_low".to_string());
    }
    if confidence < 0.5 {
        autonomy_level = autonomy_level.min(0.3);
        notes.push("autonomy_clamped:confidence_low".to_string());
    }
    if evidence_ok && telemetry_ok && confidence > 0.7 && drift_score < 0.3 {
        autonomy_level = (autonomy_level + 0.05).min(1.0);
        notes.push("autonomy_boost:high_evidence".to_string());
    }
    let verification_needed =
        confidence < 0.5 || drift_score > 0.6 || !telemetry_ok || !evidence_ok;
    let reanchor_needed = drift_score > 0.5;

    let controller_state = ControllerState {
        confidence,
        uncertainty: (1.0 - confidence).clamp(0.0, 1.0),
        drift_score,
        failure_streak,
        autonomy_level,
        verification_needed,
        reanchor_needed,
        evidence_coverage: metrics.evidence_coverage,
        telemetry_coverage: metrics.telemetry_coverage,
        last_error: None,
        last_strategy: None,
        outcome_quality: None,
        missing_fields: metrics.missing_fields.clone(),
        updated_at: Some(Utc::now().to_rfc3339()),
        notes,
    };

    ReconstructionOutput {
        controller_state,
        reconstructed_persona: persona,
        reconstructed_goals,
    }
}

pub fn evaluate_gates(state: &ControllerState) -> ControllerGate {
    let mut reasons = Vec::new();
    let throttle_tools = state.autonomy_level < 0.25 || state.failure_streak >= 3;
    if throttle_tools {
        reasons.push("low_autonomy_throttle_tools".to_string());
    }
    let throttle_threads = state.autonomy_level < 0.35;
    if throttle_threads {
        reasons.push("low_autonomy_throttle_threads".to_string());
    }
    let throttle_asks = false;
    let prefer_verification = state.verification_needed;
    if prefer_verification {
        reasons.push("verification_needed".to_string());
    }
    let reanchor = state.reanchor_needed;
    if reanchor {
        reasons.push("reanchor_needed".to_string());
    }

    ControllerGate {
        throttle_tools,
        throttle_threads,
        throttle_asks,
        prefer_verification,
        reanchor,
        autonomy_scale: state.autonomy_level,
        reasons,
    }
}

#[derive(Debug, Clone, Default)]
pub struct ControllerPerturbation {
    pub confidence_delta: Option<f32>,
    pub autonomy_delta: Option<f32>,
    pub strategy: Option<String>,
}

pub fn apply_controller_perturbation(state: &mut ControllerState, perturb: &ControllerPerturbation) {
    if let Some(delta) = perturb.confidence_delta {
        state.confidence = (state.confidence + delta).clamp(0.0, 1.0);
        state.uncertainty = (1.0 - state.confidence).clamp(0.0, 1.0);
    }
    if let Some(delta) = perturb.autonomy_delta {
        state.autonomy_level = (state.autonomy_level + delta).clamp(0.0, 1.0);
    }
    if let Some(strategy) = perturb.strategy.as_ref() {
        state.last_strategy = Some(strategy.clone());
    }
}

fn collect_field_meta_evidence(
    meta: &Option<WorkspaceFieldMeta>,
    evidence_event_ids: &mut Vec<i64>,
    belief_ids: &mut Vec<i64>,
) {
    if let Some(item) = meta.as_ref() {
        evidence_event_ids.extend(item.evidence_event_ids.iter().copied());
        belief_ids.extend(item.belief_ids.iter().copied());
    }
}

fn collect_list_meta_evidence(
    items: &[WorkspaceListItemMeta],
    evidence_event_ids: &mut Vec<i64>,
    belief_ids: &mut Vec<i64>,
) {
    for item in items.iter() {
        evidence_event_ids.extend(item.evidence_event_ids.iter().copied());
        belief_ids.extend(item.belief_ids.iter().copied());
    }
}

fn collect_hypothesis_evidence(
    items: &[WorkspaceHypothesis],
    evidence_event_ids: &mut Vec<i64>,
    belief_ids: &mut Vec<i64>,
) {
    for item in items.iter() {
        evidence_event_ids.extend(item.evidence_event_ids.iter().copied());
        belief_ids.extend(item.belief_ids.iter().copied());
    }
}

fn collect_workspace_meta_evidence(meta: &WorkspaceMeta) -> (Vec<i64>, Vec<i64>) {
    let mut evidence_event_ids = Vec::new();
    let mut belief_ids = Vec::new();

    collect_field_meta_evidence(&meta.goal_thread, &mut evidence_event_ids, &mut belief_ids);
    collect_field_meta_evidence(&meta.current_focus, &mut evidence_event_ids, &mut belief_ids);
    collect_field_meta_evidence(&meta.focus_rationale, &mut evidence_event_ids, &mut belief_ids);
    collect_list_meta_evidence(&meta.open_questions, &mut evidence_event_ids, &mut belief_ids);
    collect_list_meta_evidence(&meta.working_set_topics, &mut evidence_event_ids, &mut belief_ids);
    collect_hypothesis_evidence(&meta.active_hypotheses, &mut evidence_event_ids, &mut belief_ids);

    evidence_event_ids.sort();
    evidence_event_ids.dedup();
    belief_ids.sort();
    belief_ids.dedup();

    (evidence_event_ids, belief_ids)
}

fn status_is_complete(status: Option<&str>) -> bool {
    let Some(status) = status else {
        return false;
    };
    matches!(
        status.trim().to_lowercase().as_str(),
        "done" | "complete" | "completed" | "finished"
    )
}

fn goal_stack_summary(goal_stack: &[GoalStackItem]) -> Vec<String> {
    let mut summaries = Vec::new();
    for item in goal_stack.iter() {
        if summaries.len() >= 3 {
            break;
        }
        let goal = item.goal.trim();
        if goal.is_empty() {
            continue;
        }
        if status_is_complete(item.status.as_deref()) {
            continue;
        }
        let step = item
            .steps
            .get(item.current_step_index)
            .map(|s| s.text.trim())
            .filter(|s| !s.is_empty());
        let summary = if let Some(step) = step {
            format!("{} :: {}", goal, step)
        } else {
            goal.to_string()
        };
        summaries.push(summary);
    }
    summaries
}

async fn memory_confidence_metrics(pool: &SqlitePool) -> Result<Value, String> {
    let avg_conf: Option<f64> = sqlx::query_scalar(
        "SELECT AVG(confidence) FROM ics_beliefs WHERE status = 'active'",
    )
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;
    let low_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ics_beliefs WHERE status = 'active' AND confidence < 0.5",
    )
    .fetch_one(pool)
    .await
    .map_err(|e| e.to_string())?;
    let belief_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ics_beliefs WHERE status = 'active'",
    )
    .fetch_one(pool)
    .await
    .map_err(|e| e.to_string())?;
    let last_validated_at: Option<String> = sqlx::query_scalar(
        "SELECT MAX(last_validated_at) FROM ics_beliefs WHERE status = 'active'",
    )
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?
    .flatten();

    Ok(json!({
        "avg_confidence": avg_conf.unwrap_or(0.0),
        "low_confidence_count": low_count,
        "belief_count": belief_count,
        "last_validated_at": last_validated_at,
    }))
}

fn render_qualia_snapshot(state: &qualia::QualiaState) -> String {
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
    lines.push(format!(
        "prediction_confidence: {:.3}",
        state.prediction_confidence
    ));
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

async fn latest_evidence_snippet(pool: &SqlitePool, source_type: &str) -> Option<String> {
    let source_type = source_type.trim();
    if source_type.is_empty() {
        return None;
    }
    sqlx::query_scalar(
        "SELECT snippet FROM ics_evidence_events
         WHERE source_type = ?
         ORDER BY datetime(created_at) DESC
         LIMIT 1",
    )
    .bind(source_type)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()
}

pub async fn build_unified_self_model(
    db: &Db,
    state: &KernelState,
) -> Result<(Value, Value), String> {
    let controller_state = db
        .get_controller_state()
        .await
        .map_err(|e| e.to_string())?
        .unwrap_or_default();
    let model = db.get_self_model().await.map_err(|e| e.to_string())?;
    let memory_metrics = memory_confidence_metrics(&db.pool).await?;
    let qualia_state = qualia::compute_qualia_state(db, None).await?;
    let qualia_snapshot = render_qualia_snapshot(&qualia_state);
    let wave_state = latest_evidence_snippet(&db.pool, "wave_state").await;
    let autobiographical_summary =
        retrieval::render_autobiographical_context(&db.pool, Some(&state.conversation_id), 6)
            .await;

    let workspace = json!({
        "current_focus": state.workspace_current_focus.clone(),
        "focus_rationale": state.workspace_focus_rationale.clone(),
        "goal_thread": state.workspace_goal_thread.clone(),
        "goal_stack": state.workspace_goal_stack.clone(),
        "open_questions": state.workspace_open_questions.clone(),
        "working_set_topics": state.workspace_working_set_topics.clone(),
        "active_hypotheses": state.workspace_active_hypotheses.clone(),
    });

    let (evidence_event_ids, belief_ids) = collect_workspace_meta_evidence(&state.workspace_meta);
    let qualia_evidence_ids = db
        .get_recent_evidence_ids_by_source_types(&["qualia_snapshot"], 4)
        .await;
    let wave_evidence_ids = db
        .get_recent_evidence_ids_by_source_types(&["wave_state"], 4)
        .await;
    let mut all_evidence_ids = evidence_event_ids.clone();
    all_evidence_ids.extend(qualia_evidence_ids.iter().copied());
    all_evidence_ids.extend(wave_evidence_ids.iter().copied());
    all_evidence_ids.sort();
    all_evidence_ids.dedup();
    let evidence = json!({
        "evidence_event_ids": all_evidence_ids,
        "belief_ids": belief_ids,
        "qualia_evidence_ids": qualia_evidence_ids,
        "wave_evidence_ids": wave_evidence_ids,
    });

    let unified_state = json!({
        "workspace": workspace,
        "workspace_meta": state.workspace_meta.clone(),
        "controller_state": controller_state,
        "internal_state_summary": model.internal_state_summary,
        "internal_state_map_version": model.internal_state_map_version,
        "memory": memory_metrics,
        "qualia_snapshot": qualia_snapshot,
        "wave_state": wave_state.unwrap_or_else(|| "None".to_string()),
        "autobiographical_summary": autobiographical_summary,
        "updated_at": Utc::now().to_rfc3339(),
    });

    Ok((unified_state, evidence))
}

pub async fn update_unified_self_model(db: &Db, state: &KernelState) -> Result<(), String> {
    let (unified_state, unified_evidence) = build_unified_self_model(db, state).await?;
    let mut model = db.get_self_model().await.map_err(|e| e.to_string())?;
    model.unified_state = unified_state;
    model.unified_state_evidence = unified_evidence;
    model.unified_state_updated_at = Some(Utc::now().to_rfc3339());
    let goal_summary = goal_stack_summary(&state.workspace_goal_stack);
    if !goal_summary.is_empty() {
        model.goals = json!(goal_summary);
    } else if let Some(goal) = state
        .workspace_goal_thread
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        model.goals = json!([goal]);
    }
    db.set_self_model(&model).await.map_err(|e| e.to_string())?;
    Ok(())
}

pub async fn collect_unified_self_evidence_ids(db: &Db) -> Result<Vec<i64>, String> {
    let model = db.get_self_model().await.map_err(|e| e.to_string())?;
    let mut evidence_ids = Vec::new();
    if let Some(list) = model
        .unified_state_evidence
        .get("evidence_event_ids")
        .and_then(|v| v.as_array())
    {
        for item in list.iter() {
            if let Some(id) = item.as_i64() {
                if id > 0 {
                    evidence_ids.push(id);
                }
            }
        }
    }
    evidence_ids.sort();
    evidence_ids.dedup();
    Ok(evidence_ids)
}

#[cfg(test)]
mod tests_reconstruction {
    use super::*;
    use chrono::Utc;

    #[test]
    fn divergence_pressure_increases_drift() {
        let mut telemetry_values = HashMap::new();
        telemetry_values.insert(
            "telemetry.prediction_divergence_rate".to_string(),
            EvidencePoint {
                belief_id: 0,
                key: "telemetry.prediction_divergence_rate".to_string(),
                value: "1.0".to_string(),
                confidence: 0.9,
                last_evidence_at: Some(Utc::now().to_rfc3339()),
                source_quality: 1.0,
            },
        );
        telemetry_values.insert(
            "telemetry.prediction_divergence_persisted_rate".to_string(),
            EvidencePoint {
                belief_id: 0,
                key: "telemetry.prediction_divergence_persisted_rate".to_string(),
                value: "1.0".to_string(),
                confidence: 0.9,
                last_evidence_at: Some(Utc::now().to_rfc3339()),
                source_quality: 1.0,
            },
        );
        telemetry_values.insert(
            "telemetry.prediction_divergence_resolved_rate".to_string(),
            EvidencePoint {
                belief_id: 0,
                key: "telemetry.prediction_divergence_resolved_rate".to_string(),
                value: "0.0".to_string(),
                confidence: 0.9,
                last_evidence_at: Some(Utc::now().to_rfc3339()),
                source_quality: 1.0,
            },
        );

        let metrics = SelfEvidenceMetrics {
            persona_values: HashMap::new(),
            goal_values: Vec::new(),
            goal_removed: Vec::new(),
            telemetry_values,
            source_counts: HashMap::new(),
            avg_confidence: 0.8,
            evidence_coverage: 1.0,
            telemetry_coverage: 1.0,
            missing_fields: Vec::new(),
            conflict_count: 0,
        };

        let current = SelfModel {
            capabilities: serde_json::json!({}),
            limitations: serde_json::json!({}),
            active_tools: serde_json::json!({}),
            memory_health: serde_json::json!({}),
            persona: serde_json::json!({}),
            persona_daily_delta: serde_json::json!({}),
            persona_last_delta_date: None,
            goals: serde_json::json!([]),
            identity_thread: None,
            identity_confidence: 0.0,
            identity_uncertainty_note: None,
            identity_updated_at: None,
            reflection_status: serde_json::json!({}),
            reflection_frozen: false,
            last_reflection_at: None,
            internal_state_summary: serde_json::json!({}),
            internal_state_map_version: None,
            unified_state: serde_json::json!({}),
            unified_state_evidence: serde_json::json!({}),
            unified_state_updated_at: None,
            updated_at: Utc::now().to_rfc3339(),
        };

        let output = reconstruct_from_metrics(&metrics, &current, 0);
        assert!(output.controller_state.drift_score > 0.5);
        assert!(output.controller_state.verification_needed);
    }
}

pub fn apply_reconstruction_to_model(model: &mut SelfModel, output: &ReconstructionOutput) -> bool {
    let mut changed = false;
    if model.persona != output.reconstructed_persona {
        model.persona = output.reconstructed_persona.clone();
        changed = true;
    }
    if model.goals != output.reconstructed_goals {
        model.goals = output.reconstructed_goals.clone();
        changed = true;
    }
    changed
}

fn newer_than(a: &Option<String>, b: &Option<String>) -> bool {
    match (a, b) {
        (Some(a_ts), Some(b_ts)) => {
            let a_dt = parse_timestamp(a_ts);
            let b_dt = parse_timestamp(b_ts);
            match (a_dt, b_dt) {
                (Some(a_dt), Some(b_dt)) => a_dt > b_dt,
                (Some(_), None) => true,
                _ => false,
            }
        }
        (Some(_), None) => true,
        _ => false,
    }
}

fn parse_timestamp(raw: &str) -> Option<DateTime<Utc>> {
    if let Ok(parsed) = DateTime::parse_from_rfc3339(raw) {
        return Some(parsed.with_timezone(&Utc));
    }
    if let Ok(parsed) = NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S") {
        return Some(DateTime::<Utc>::from_naive_utc_and_offset(parsed, Utc));
    }
    if let Ok(parsed) = NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S%.f") {
        return Some(DateTime::<Utc>::from_naive_utc_and_offset(parsed, Utc));
    }
    None
}

fn recency_score(last_evidence_at: &Option<String>, max_age_days: i64) -> f32 {
    let Some(ts) = last_evidence_at.as_deref() else {
        return 0.0;
    };
    let Some(parsed) = parse_timestamp(ts) else {
        return 0.0;
    };
    let age_days = (Utc::now() - parsed.with_timezone(&Utc)).num_days().max(0) as f32;
    if max_age_days <= 0 {
        return 0.0;
    }
    if age_days >= max_age_days as f32 {
        return 0.0;
    }
    (1.0 - (age_days / max_age_days as f32)).clamp(0.0, 1.0)
}

fn source_quality_for(raw: &str) -> f32 {
    match raw.trim().to_lowercase().as_str() {
        "user" => 1.0,
        "user_focus" => 1.0,
        "tool" => 0.85,
        "system" => 0.75,
        "inference" => 0.5,
        _ => 0.6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reconstruction_uses_persona_evidence_and_goals() {
        let current = SelfModel {
            capabilities: json!([]),
            limitations: json!([]),
            active_tools: json!([]),
            memory_health: json!({}),
            persona: json!({
                "tone": 0.5,
                "verbosity": 0.5,
                "directness": 0.5,
                "formality": 0.5,
                "initiative": 0.5,
            }),
            persona_daily_delta: json!({}),
            persona_last_delta_date: None,
            goals: json!(["old_goal"]),
            identity_thread: None,
            identity_confidence: 0.5,
            identity_uncertainty_note: None,
            identity_updated_at: None,
            reflection_status: json!({}),
            reflection_frozen: false,
            last_reflection_at: None,
            internal_state_summary: serde_json::json!({}),
            internal_state_map_version: None,
            unified_state: serde_json::json!({}),
            unified_state_evidence: serde_json::json!({}),
            unified_state_updated_at: None,
            updated_at: "now".to_string(),
        };

        let metrics = SelfEvidenceMetrics {
            persona_values: HashMap::from([
                ("persona.tone".to_string(), EvidencePoint {
                    belief_id: 1,
                    key: "persona.tone".to_string(),
                    value: "0.7".to_string(),
                    confidence: 0.9,
                    last_evidence_at: Some("2026-02-20T00:00:00Z".to_string()),
                    source_quality: 0.9,
                }),
            ]),
            goal_values: vec![EvidencePoint {
                belief_id: 2,
                key: "goal".to_string(),
                value: "new_goal".to_string(),
                confidence: 0.8,
                last_evidence_at: Some("2026-02-20T00:00:00Z".to_string()),
                source_quality: 0.8,
            }],
            goal_removed: vec![],
            telemetry_values: HashMap::new(),
            source_counts: HashMap::new(),
            avg_confidence: 0.85,
            evidence_coverage: 0.5,
            telemetry_coverage: 1.0,
            missing_fields: vec![],
            conflict_count: 0,
        };

        let output = reconstruct_from_metrics(&metrics, &current, 0);
        let tone = output.reconstructed_persona["tone"].as_f64().unwrap_or_default();
        assert!((tone - 0.7).abs() < 0.001);
        assert_eq!(output.reconstructed_goals, json!(["new_goal"]));
        assert!(output.controller_state.confidence > 0.8);
    }

    #[test]
    fn continuity_small_telemetry_change_limits_autonomy_shift() {
        let current = SelfModel {
            capabilities: json!([]),
            limitations: json!([]),
            active_tools: json!([]),
            memory_health: json!({}),
            persona: json!({
                "tone": 0.5,
                "verbosity": 0.5,
                "directness": 0.5,
                "formality": 0.5,
                "initiative": 0.5,
            }),
            persona_daily_delta: json!({}),
            persona_last_delta_date: None,
            goals: json!(["goal"]),
            identity_thread: None,
            identity_confidence: 0.5,
            identity_uncertainty_note: None,
            identity_updated_at: None,
            reflection_status: json!({}),
            reflection_frozen: false,
            last_reflection_at: None,
            internal_state_summary: serde_json::json!({}),
            internal_state_map_version: None,
            unified_state: serde_json::json!({}),
            unified_state_evidence: serde_json::json!({}),
            unified_state_updated_at: None,
            updated_at: "now".to_string(),
        };

        let mut telemetry_values = HashMap::new();
        telemetry_values.insert(
            "telemetry.user_feedback_pushback_rate".to_string(),
            EvidencePoint {
                belief_id: 0,
                key: "telemetry.user_feedback_pushback_rate".to_string(),
                value: "0.20".to_string(),
                confidence: 0.9,
                last_evidence_at: Some("2026-02-20T00:00:00Z".to_string()),
                source_quality: 0.7,
            },
        );
        telemetry_values.insert(
            "telemetry.user_feedback_followup_rate".to_string(),
            EvidencePoint {
                belief_id: 0,
                key: "telemetry.user_feedback_followup_rate".to_string(),
                value: "0.30".to_string(),
                confidence: 0.9,
                last_evidence_at: Some("2026-02-20T00:00:00Z".to_string()),
                source_quality: 0.7,
            },
        );

        let base_metrics = SelfEvidenceMetrics {
            persona_values: HashMap::from([(
                "persona.tone".to_string(),
                EvidencePoint {
                    belief_id: 1,
                    key: "persona.tone".to_string(),
                    value: "0.6".to_string(),
                    confidence: 0.8,
                    last_evidence_at: Some("2026-02-20T00:00:00Z".to_string()),
                    source_quality: 0.8,
                },
            )]),
            goal_values: vec![EvidencePoint {
                belief_id: 2,
                key: "goal".to_string(),
                value: "goal".to_string(),
                confidence: 0.8,
                last_evidence_at: Some("2026-02-20T00:00:00Z".to_string()),
                source_quality: 0.8,
            }],
            goal_removed: vec![],
            telemetry_values: telemetry_values.clone(),
            source_counts: HashMap::new(),
            avg_confidence: 0.8,
            evidence_coverage: 0.7,
            telemetry_coverage: 1.0,
            missing_fields: vec![],
            conflict_count: 0,
        };

        let mut perturbed = base_metrics.clone();
        if let Some(point) = perturbed.telemetry_values.get_mut("telemetry.user_feedback_pushback_rate") {
            point.value = "0.25".to_string();
        }
        let base_out = reconstruct_from_metrics(&base_metrics, &current, 0);
        let perturbed_out = reconstruct_from_metrics(&perturbed, &current, 0);
        let delta = (base_out.controller_state.autonomy_level - perturbed_out.controller_state.autonomy_level).abs();
        assert!(delta <= 0.15, "autonomy delta too large: {}", delta);
    }

    #[test]
    fn controller_perturbation_adjusts_state() {
        let mut state = ControllerState {
            confidence: 0.7,
            uncertainty: 0.3,
            drift_score: 0.2,
            failure_streak: 0,
            autonomy_level: 0.8,
            verification_needed: false,
            reanchor_needed: false,
            evidence_coverage: 0.8,
            telemetry_coverage: 0.8,
            last_error: None,
            last_strategy: None,
            outcome_quality: None,
            missing_fields: Vec::new(),
            updated_at: None,
            notes: Vec::new(),
        };
        let perturb = ControllerPerturbation {
            confidence_delta: Some(-0.2),
            autonomy_delta: Some(-0.3),
            strategy: Some("ask_user".to_string()),
        };
        apply_controller_perturbation(&mut state, &perturb);
        assert!(state.confidence < 0.7);
        assert!(state.autonomy_level < 0.8);
        assert_eq!(state.last_strategy.as_deref(), Some("ask_user"));
    }
}
