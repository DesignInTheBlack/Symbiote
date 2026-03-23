use serde::{Deserialize, Serialize};

use crate::models::ControllerState;
use super::*;

#[derive(Debug, Deserialize, Clone)]
pub(crate) struct PredictionCandidate {
    pub metric: String,
    pub expected_value: f64,
    pub expected_variance: Option<f64>,
    pub horizon: String,
    pub confidence: Option<f64>,
    pub evidence_event_ids: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PredictionContext {
    pub controller_state: Option<ControllerState>,
    pub workspace_focus: Option<String>,
    pub workspace_goal: Option<String>,
    pub workspace_open_questions: Vec<String>,
    pub last_user_input: Option<String>,
}

pub(crate) fn build_prediction_context(state: &KernelState) -> PredictionContext {
    PredictionContext {
        controller_state: state.controller_state.clone(),
        workspace_focus: state.workspace_current_focus.clone(),
        workspace_goal: state.workspace_goal_thread.clone(),
        workspace_open_questions: state.workspace_open_questions.clone(),
        last_user_input: state.last_user_input.clone(),
    }
}

pub fn allowed_prediction_metrics() -> HashSet<String> {
    [
        "tool_success_rate",
        "memory_pass_rate",
        "clarification_rate",
        "refusal_rate",
        "workspace_stability_rate",
        "response_len",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

pub fn allowed_prediction_horizons() -> HashSet<String> {
    ["next_turn", "next_tool", "next_5m", "next_hour"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

pub fn validate_prediction_fields(
    metric: &str,
    expected_value: f64,
    expected_variance: Option<f64>,
    horizon: &str,
    allowed_metrics: &HashSet<String>,
    allowed_horizons: &HashSet<String>,
) -> Option<String> {
    let metric = metric.trim().to_lowercase();
    let horizon = horizon.trim().to_lowercase();
    if !allowed_metrics.contains(&metric) {
        return Some("invalid_metric".to_string());
    }
    if !allowed_horizons.contains(&horizon) {
        return Some("invalid_horizon".to_string());
    }
    if expected_variance.unwrap_or(0.0) < 0.0 {
        return Some("invalid_variance".to_string());
    }
    if metric != "response_len"
        && (expected_value < 0.0 || expected_value > 1.0)
    {
        return Some("invalid_expected_value".to_string());
    }
    None
}

pub(crate) fn validate_prediction_basics(
    candidate: &PredictionCandidate,
    allowed_metrics: &HashSet<String>,
    allowed_horizons: &HashSet<String>,
) -> Option<String> {
    validate_prediction_fields(
        &candidate.metric,
        candidate.expected_value,
        candidate.expected_variance,
        &candidate.horizon,
        allowed_metrics,
        allowed_horizons,
    )
}

pub async fn record_prediction_rejection(
    pool: &SqlitePool,
    run_id: Option<&str>,
    trace_id: Option<&str>,
    context_ref_json: &str,
    reason: &str,
    metric_hint: Option<&str>,
) {
    let rejection_id = Uuid::new_v4().to_string();
    let _ = sqlx::query(
        "INSERT INTO self_predictions
         (id, run_id, trace_id, metric, context_ref_json, predicted_target_type, expected_value, expected_variance, expected_bounds_json, horizon, confidence, evidence_event_ids, linked_claims_json, normalization_contract_id, salience_hint, rejection_reason, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
    )
    .bind(&rejection_id)
    .bind(run_id)
    .bind(trace_id)
    .bind(metric_hint.unwrap_or("rejection"))
    .bind(context_ref_json)
    .bind("none")
    .bind(0.0_f64)
    .bind(0.0_f64)
    .bind::<Option<String>>(None)
    .bind("next_turn")
    .bind(0.0_f64)
    .bind("[]")
    .bind::<Option<String>>(None)
    .bind::<Option<String>>(None)
    .bind(0.0_f64)
    .bind(reason)
    .execute(pool)
    .await;
}
