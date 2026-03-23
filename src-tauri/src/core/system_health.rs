use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::Serialize;
use serde_json::{json, Value};
use sqlx::Row;
use std::collections::HashMap;
use std::sync::Arc;
use tauri::AppHandle;
use uuid::Uuid;

use crate::core::{system_controls, system_log, workspace as core_workspace, self_model_controller};
use crate::core::kernel::constants::{EVIDENCE_MIN, TELEMETRY_MIN};
use crate::db::Db;

pub const HEALTH_WINDOW_MINUTES: i64 = 60;
const GATE_INPUT_LIMIT: i64 = 6;
pub const UNITY_WINDOW_MINUTES: i64 = 30;
const UNITY_GREEN_WINDOW_HOURS: i64 = 6;
const AUTO_DEGRADE_MEMORY_DRIFT: i64 = 3;
const AUTO_DEGRADE_TOOL_FAILURE_RATE: f64 = 0.25;
const OUTCOME_REMINDER_RUN_THRESHOLD: i64 = 10;
const OUTCOME_REMINDER_COOLDOWN_MINUTES: i64 = 60;
const MIN_EVIDENCE_EVENTS: i64 = 6;
const WARN_RATE_TARGET_PER_MIN: f64 = 8.0;
const BENIGN_WARN_EVENTS: &[&str] = &[
    "prompt_trim_critical",
    "contract_violation",
    "response_empty_retry",
    "response_reasoning_json_retry",
    "response_reasoning_json_invalid",
    "response_empty_with_reasoning",
    "response_empty_with_reasoning_repeat",
    "memory_pass_retry_strict",
    "monologue_parse_repair_attempt",
    "monologue_parse_repair_skipped",
    "monologue_tick_retry",
    "monologue_tick_retry_skipped",
    "prediction_generation_rejected",
];

#[derive(Debug, Serialize, Clone)]
pub struct SystemHealthSnapshotPayload {
    pub snapshot_id: String,
    pub timestamp: String,
    pub run_id: Option<String>,
    pub trace_id: Option<String>,
    pub metrics: Value,
    pub subsystem_states: Value,
}

pub struct HealthAggregator {
    db: Arc<Db>,
}

impl HealthAggregator {
    pub fn new(db: Arc<Db>) -> Self {
        Self { db }
    }

    async fn apply_control_mode(
        &self,
        control_map: &HashMap<String, system_controls::ControlState>,
        subsystem_id: &str,
        mode: &str,
        reason: &str,
        app_handle: Option<&AppHandle>,
    ) {
        let current = control_map.get(subsystem_id);
        let current_mode = current.map(|c| c.mode.as_str()).unwrap_or("normal");
        let current_reason = current.and_then(|c| c.reason.as_deref()).unwrap_or("");
        let auto_reason = current_reason.starts_with("auto:health");
        if current_mode != "normal" && current_mode != "degraded" {
            return;
        }
        if mode == "degraded" {
            if current_mode == "degraded" && auto_reason && current_reason == reason {
                return;
            }
            if current_mode != "normal" && !auto_reason {
                return;
            }
        } else if mode == "normal" {
            if !auto_reason || current_mode != "degraded" {
                return;
            }
        }
        if let Ok(entry) = self
            .db
            .set_system_control(subsystem_id, mode, None, Some("health_auto".to_string()), Some(reason.to_string()))
            .await
        {
            let _ = system_log::log_event(
                &self.db.pool,
                app_handle,
                "info",
                "system",
                None,
                None,
                json!({
                    "event": "auto_degrade_control",
                    "subsystem_id": entry.subsystem_id,
                    "mode": entry.mode,
                    "reason": entry.reason,
                }),
            )
            .await;
        }
    }

    async fn apply_auto_degradation(
        &self,
        control_map: &HashMap<String, system_controls::ControlState>,
        evidence_low: bool,
        memory_drift: bool,
        tool_unreliable: bool,
        app_handle: Option<&AppHandle>,
    ) {
        let mut targets: HashMap<&str, &str> = HashMap::new();
        if evidence_low {
            targets.entry("memory_write").or_insert("auto:health:evidence_low");
            targets.entry("self_memory").or_insert("auto:health:evidence_low");
            targets.entry("memory_consolidation").or_insert("auto:health:evidence_low");
        }
        if memory_drift {
            targets.entry("memory_write").or_insert("auto:health:memory_drift");
            targets.entry("memory_consolidation").or_insert("auto:health:memory_drift");
        }
        if tool_unreliable {
            targets.entry("tool_execution").or_insert("auto:health:tool_failure_rate");
        }

        let mut to_recover: Vec<&str> = Vec::new();
        for id in ["memory_write", "self_memory", "memory_consolidation", "tool_execution"] {
            let still_needed = targets.contains_key(id);
            if !still_needed {
                to_recover.push(id);
            }
        }

        for (target, reason) in targets.iter() {
            self.apply_control_mode(control_map, target, "degraded", reason, app_handle)
                .await;
        }

        for target in to_recover {
            self.apply_control_mode(control_map, target, "normal", "auto:health:recovered", app_handle)
                .await;
        }
    }

    pub async fn capture_snapshot(
        &self,
        run_id: Option<&str>,
        trace_id: Option<&str>,
        app_handle: Option<&AppHandle>,
    ) -> Result<SystemHealthSnapshotPayload, String> {
        let now = Utc::now();
        let since = now - ChronoDuration::minutes(HEALTH_WINDOW_MINUTES);
        let since_str = since.to_rfc3339();

        let control_map = system_controls::load_control_map(&self.db).await;
        let subsystem_states = build_subsystem_states(&control_map);
        let cockpit_write_enabled = self
            .db
            .get_settings()
            .await
            .ok()
            .and_then(|s| s.cockpit_write_enabled)
            .unwrap_or(false);

        let gate_counts = fetch_gate_counts(&self.db.pool, &since_str).await;
        let gate_inputs = fetch_gate_inputs(&self.db.pool).await;
        let (module_status, module_detail, module_status_at) =
            fetch_latest_module_status(&self.db.pool).await;

        let (controller_state, organism_state, error_state) =
            fetch_latest_subject_state(&self.db.pool).await;

        let memory_pass_count =
            count_system_log_event(&self.db.pool, "memory_pass_result", &since_str).await;
        let memory_pass_skipped =
            count_system_log_event(&self.db.pool, "memory_pass_skipped", &since_str).await;
        let memory_pass_zero_writes =
            count_system_log_event(&self.db.pool, "memory_pass_zero_writes", &since_str).await;
        let kernel_cycle_count =
            count_system_log_event(&self.db.pool, "kernel_cycle", &since_str).await;
        let memory_promotion_low_confidence =
            count_system_log_event(&self.db.pool, "memory_promotion_low_confidence", &since_str).await;
        let memory_validation_runs =
            count_system_log_event(&self.db.pool, "memory_validation_run", &since_str).await;
        let memory_validation_drift =
            count_system_log_event(&self.db.pool, "memory_validation_drift", &since_str).await;
        let memory_validation_last_at =
            last_system_log_timestamp(&self.db.pool, None, None, Some("memory_validation_run")).await;
        let telemetry_calibration_runs =
            count_system_log_event(&self.db.pool, "telemetry_calibration_run", &since_str).await;
        let telemetry_calibration_drift =
            count_system_log_event(&self.db.pool, "telemetry_calibration_drift", &since_str).await;
        let telemetry_calibration_last_at =
            last_system_log_timestamp(&self.db.pool, None, None, Some("telemetry_calibration_run")).await;
        let outcome_total =
            count_table_in_window(&self.db.pool, "outcome_events", "created_at", &since_str, None).await;
        let outcome_confirm = count_table_in_window(
            &self.db.pool,
            "outcome_events",
            "created_at",
            &since_str,
            Some("verdict = 'confirm'"),
        )
        .await;
        let outcome_disconfirm = count_table_in_window(
            &self.db.pool,
            "outcome_events",
            "created_at",
            &since_str,
            Some("verdict = 'disconfirm'"),
        )
        .await;
        let outcome_inconclusive = count_table_in_window(
            &self.db.pool,
            "outcome_events",
            "created_at",
            &since_str,
            Some("verdict = 'inconclusive'"),
        )
        .await;
        let outcome_last_at =
            last_table_timestamp(&self.db.pool, "outcome_events", "created_at", None).await;
        let outcome_accuracy = if (outcome_confirm + outcome_disconfirm) > 0 {
            outcome_confirm as f64 / (outcome_confirm + outcome_disconfirm) as f64
        } else {
            0.0
        };
        let outcome_accuracy_opt = if outcome_total > 0 {
            Some(outcome_accuracy)
        } else {
            None
        };
        if outcome_total == 0 && kernel_cycle_count >= OUTCOME_REMINDER_RUN_THRESHOLD {
            let reminder_due = self
                .db
                .get_key("outcome_reminder_last_at")
                .await
                .ok()
                .flatten()
                .and_then(|ts| parse_timestamp(&ts))
                .map(|ts| now.signed_duration_since(ts).num_minutes() >= OUTCOME_REMINDER_COOLDOWN_MINUTES)
                .unwrap_or(true);
            if reminder_due {
                let _ = self
                    .db
                    .set_key("outcome_reminder_last_at", &now.to_rfc3339())
                    .await;
                let _ = system_log::log_event(
                    &self.db.pool,
                    app_handle,
                    "info",
                    "system",
                    None,
                    None,
                    json!({
                        "event": "outcome_reminder",
                        "window_minutes": HEALTH_WINDOW_MINUTES,
                        "run_count": kernel_cycle_count,
                        "outcome_total": outcome_total,
                    }),
                )
                .await;
            }
        }
        let drift_penalty = (memory_validation_drift + telemetry_calibration_drift) as f64 * 0.02;
        let combined_score = outcome_accuracy_opt.map(|acc| (acc - drift_penalty).clamp(0.0, 1.0));
        let memory_write_count =
            count_table_in_window(&self.db.pool, "memory_write_ledger", "created_at", &since_str, None).await;
        let memory_error_at = last_system_log_timestamp(
            &self.db.pool,
            Some("memory"),
            Some("error"),
            None,
        )
        .await
        .unwrap_or_default();

        let rolling_summary_updates =
            count_system_log_event(&self.db.pool, "rolling_summary_updated", &since_str).await;
        let rolling_summary_failures =
            count_system_log_event(&self.db.pool, "rolling_summary_failed", &since_str).await;
        let summary_chunk_count = count_table_in_window(
            &self.db.pool,
            "conversation_summary_chunks",
            "created_at",
            &since_str,
            None,
        )
        .await;
        let inner_summary_updates = count_table_in_window(
            &self.db.pool,
            "memory_write_ledger",
            "created_at",
            &since_str,
            Some("category = 'inner_summary'"),
        )
        .await;
        let inner_summary_failures =
            count_system_log_event(&self.db.pool, "inner_summary_failed", &since_str).await;
        let last_inner_summary_at = last_table_timestamp(
            &self.db.pool,
            "memory_write_ledger",
            "created_at",
            Some("category = 'inner_summary'"),
        )
        .await;
        let last_rolling_summary_at =
            last_system_log_timestamp(&self.db.pool, None, None, Some("rolling_summary_updated")).await;
        let rolling_age_seconds =
            diff_seconds_from_now(last_rolling_summary_at.as_deref()).unwrap_or(0);
        let inner_age_seconds =
            diff_seconds_from_now(last_inner_summary_at.as_deref()).unwrap_or(0);

        let self_memory_writes =
            count_table_in_window(&self.db.pool, "self_beliefs", "created_at", &since_str, None).await;
        let self_memory_denies = count_system_log_event_filtered(
            &self.db.pool,
            "memory_write_blocked",
            &since_str,
            Some("json_extract(payload, '$.reason') LIKE '%self_memory%' OR json_extract(payload, '$.action') = 'self_memory_write'"),
        )
        .await;
        let last_self_memory_at =
            last_table_timestamp(&self.db.pool, "self_beliefs", "created_at", None).await;

        let consolidation_runs =
            count_system_log_event(&self.db.pool, "consolidation_complete", &since_str).await;
        let consolidation_errors =
            count_system_log_event(&self.db.pool, "consolidation_error", &since_str).await;
        let consolidation_last_at =
            last_system_log_timestamp(&self.db.pool, None, None, Some("consolidation_complete")).await;

        let episodic_events =
            count_table_in_window(&self.db.pool, "episodic_events", "timestamp", &since_str, None).await;
        let episodic_compactions =
            count_system_log_event(&self.db.pool, "episodic_compaction", &since_str).await;
        let episodic_last_at =
            last_table_timestamp(&self.db.pool, "episodic_events", "timestamp", None).await;

        let monologue_entries =
            count_table_in_window(&self.db.pool, "inner_monologue_entries", "created_at", &since_str, None).await;
        let (monologue_suppressed, monologue_total) =
            count_monologue_suppression(&self.db.pool, &since_str).await;
        let monologue_tick_start =
            count_system_log_event(&self.db.pool, "monologue_tick_start", &since_str).await;
        let monologue_tick_end =
            count_system_log_event(&self.db.pool, "monologue_tick_end", &since_str).await;
        let monologue_tick_timeout =
            count_system_log_event(&self.db.pool, "monologue_tick_timeout", &since_str).await;
        let loop_outcomes =
            count_system_log_event(&self.db.pool, "monologue_loop_outcome", &since_str).await;
        let loop_noop = count_system_log_event_filtered(
            &self.db.pool,
            "monologue_loop_outcome",
            &since_str,
            Some("json_extract(payload, '$.state_change_candidates') = 0"),
        )
        .await;
        let loop_delta_applied =
            count_system_log_event(&self.db.pool, "loop_delta_applied", &since_str).await;
        let loop_delta_rejected =
            count_system_log_event(&self.db.pool, "loop_delta_rejected", &since_str).await;
        let loop_noop_streak = fetch_latest_kernel_loop_streak(&self.db.pool).await;
        let monologue_success_rate = if monologue_tick_start > 0 {
            monologue_tick_end as f64 / monologue_tick_start as f64
        } else {
            0.0
        };
        let monologue_suppression_rate = if monologue_total > 0 {
            monologue_suppressed as f64 / monologue_total as f64
        } else {
            0.0
        };
        let last_monologue_suppression =
            last_table_timestamp(&self.db.pool, "inner_monologue_candidates", "created_at", Some("suppression_reason IS NOT NULL AND suppression_reason != ''")).await;
        let last_monologue_reason = fetch_latest_suppression_reason(&self.db.pool).await;

        let wave_contributions =
            count_system_log_event(&self.db.pool, "wave_contribution", &since_str).await;
        let wave_projections =
            count_system_log_event(&self.db.pool, "wave_projection", &since_str).await;
        let wave_modulations =
            count_system_log_event(&self.db.pool, "wave_arbitration_modulation", &since_str).await;
        let wave_source_counts =
            fetch_grouped_event_counts(&self.db.pool, "wave_contribution", "source", &since_str).await;

        let audits_count =
            count_table_in_window(&self.db.pool, "audit_log", "created_at", &since_str, None).await;
        let audits_mean_score =
            average_table_value(&self.db.pool, "audit_log", "discrepancy_score", &since_str, None).await;

        let qualia_labels =
            count_table_in_window(&self.db.pool, "qualia_labels", "created_at", &since_str, None).await;
        let qualia_rewards =
            count_table_in_window(&self.db.pool, "qualia_reward_events", "created_at", &since_str, None).await;
        let qualia_mean_intensity =
            average_table_value(&self.db.pool, "qualia_labels", "intensity", &since_str, None).await;
        let qualia_mean_reward =
            average_table_value(&self.db.pool, "qualia_reward_events", "magnitude", &since_str, None).await;

        let feedback_breakdown = fetch_feedback_breakdown(&self.db.pool, &since_str).await;
        let prompt_trim_counts =
            fetch_grouped_event_counts(&self.db.pool, "prompt_trim", "title", &since_str).await;
        let prompt_overflow_count =
            count_system_log_event(&self.db.pool, "prompt_overflow", &since_str).await;
        let attention_schema_updates =
            count_system_log_event(&self.db.pool, "attention_schema_updated", &since_str).await;
        let attention_schema_stability =
            average_system_log_metric(&self.db.pool, "attention_schema_updated", "stability", &since_str).await;
        let (attention_schema_last_at, attention_schema_last_payload) =
            fetch_latest_log_payload(&self.db.pool, "attention_schema_updated").await;
        let attention_schema_capacity = attention_schema_last_payload
            .as_ref()
            .and_then(|p| p.get("capacity_usage"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let attention_schema_policy = attention_schema_last_payload
            .as_ref()
            .and_then(|p| p.get("selection_policy"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let attention_schema_last_stability = attention_schema_last_payload
            .as_ref()
            .and_then(|p| p.get("stability"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let workspace_snapshot_count =
            count_system_log_event(&self.db.pool, "workspace_snapshot", &since_str).await;
        let workspace_missing_count =
            count_system_log_event(&self.db.pool, "workspace_missing_contributors", &since_str).await;
        let (workspace_snapshot_at, workspace_snapshot_payload) =
            fetch_latest_log_payload(&self.db.pool, "workspace_snapshot").await;
        let workspace_summary = workspace_snapshot_payload
            .as_ref()
            .and_then(|payload| payload.get("contributors"))
            .and_then(|value| serde_json::from_value::<crate::models::WorkspaceContributors>(value.clone()).ok())
            .map(|contributors| core_workspace::summarize_contributors(&contributors))
            .unwrap_or_else(|| "".to_string());
        let prediction_complete =
            count_system_log_event(&self.db.pool, "prediction_generation_complete", &since_str).await;
        let prediction_rejected =
            count_system_log_event(&self.db.pool, "prediction_generation_rejected", &since_str).await;
        let prediction_failed =
            count_system_log_event(&self.db.pool, "prediction_generation_failed", &since_str).await;
        let prediction_repair =
            count_system_log_event(&self.db.pool, "prediction_json_repair_used", &since_str).await;
        let prediction_repair_rate = if prediction_complete > 0 {
            prediction_repair as f64 / prediction_complete as f64
        } else {
            0.0
        };
        let prediction_skipped =
            count_system_log_event(&self.db.pool, "prediction_generation_skipped", &since_str).await;
        let residual_shadow_impact_pct =
            average_system_log_metric(&self.db.pool, "residual_shadow_impact", "impact_pct", &since_str).await;
        let residual_shadow_would_change =
            count_system_log_event_filtered(
                &self.db.pool,
                "residual_shadow_impact",
                &since_str,
                Some("json_extract(payload, '$.would_change_winner') = 1"),
            )
            .await;
        let (self_reflection_at, self_reflection_frozen, self_reflection_status) =
            fetch_self_reflection_status(&self.db.pool).await;
        let self_reflection_skipped =
            count_system_log_event(&self.db.pool, "self_reflection_skipped", &since_str).await;
        let self_reflection_missing =
            count_system_log_event(&self.db.pool, "self_reflection_missing_evidence", &since_str).await;
        let self_claim_rejected =
            count_system_log_event(&self.db.pool, "self_claim_rejected", &since_str).await;
        let self_claim_missing =
            count_system_log_event(&self.db.pool, "self_claim_missing_evidence", &since_str).await;
        let self_claim_stale =
            count_system_log_event(&self.db.pool, "self_claim_stale_evidence", &since_str).await;
        let (identity_thread_present, identity_confidence) =
            fetch_identity_state(&self.db.pool).await;
        let phi_write_blocked =
            count_system_log_event(&self.db.pool, "phi_write_blocked", &since_str).await;
        let phi_redacted =
            count_system_log_event(&self.db.pool, "phi_redacted", &since_str).await;
        let relation_beliefs =
            count_table_in_window(&self.db.pool, "ics_beliefs", "created_at", &since_str, Some("kind = 'rel'")).await;
        let relation_shape_missing =
            count_system_log_event(&self.db.pool, "relation_shape_missing", &since_str).await;
        let relation_shape_mismatch =
            count_system_log_event(&self.db.pool, "relation_shape_mismatch", &since_str).await;
        let relation_promotions_created =
            count_system_log_event(&self.db.pool, "relation_promotion_created", &since_str).await;
        let relation_promotions_skipped =
            count_system_log_event(&self.db.pool, "relation_promotion_skipped", &since_str).await;
        let relation_promotion_precision = if relation_promotions_created + relation_promotions_skipped > 0 {
            relation_promotions_created as f64
                / (relation_promotions_created + relation_promotions_skipped) as f64
        } else {
            0.0
        };
        let summary_archive_skipped =
            count_system_log_event(&self.db.pool, "summary_archive_skipped", &since_str).await;

        let (error_counts, error_open) = fetch_error_event_counts(&self.db.pool, &since_str).await;
        let error_by_classification =
            fetch_grouped_counts(&self.db.pool, "error_events", "classification", "created_at", &since_str).await;
        let error_by_status =
            fetch_grouped_counts(&self.db.pool, "error_events", "status", "created_at", &since_str).await;

        let (tool_successes, tool_failures) =
            fetch_tool_dispatch_counts(&self.db.pool, &since_str).await;
        let (tool_planning_errors, tool_execution_errors, tool_cancelled) =
            fetch_tool_failure_kind_counts(&self.db.pool, &since_str).await;
        let tool_throttle_defers = count_system_log_event_filtered(
            &self.db.pool,
            "tool_throttle_defer",
            &since_str,
            Some("json_extract(payload, '$.status') = 'queued'"),
        )
        .await;

        let (log_warn_count, log_error_count, log_by_category) =
            fetch_log_level_counts(&self.db.pool, &since_str).await;
        let mut benign_warn_counts =
            fetch_event_counts_for_list(&self.db.pool, BENIGN_WARN_EVENTS, &since_str).await;
        let benign_workspace_missing = count_benign_workspace_missing(
            &self.db.pool,
            &since_str,
            &control_map,
            cockpit_write_enabled,
        )
        .await;
        if benign_workspace_missing > 0 {
            benign_warn_counts.insert(
                "workspace_missing_contributors".to_string(),
                benign_workspace_missing,
            );
        }
        let benign_warn_total = benign_warn_counts.values().copied().sum::<i64>();
        let warn_effective = (log_warn_count - benign_warn_total).max(0);
        let warn_rate = warn_effective as f64 / HEALTH_WINDOW_MINUTES as f64;
        let warn_penalty = (warn_rate / WARN_RATE_TARGET_PER_MIN).min(1.0);
        let warn_top = fetch_top_log_events(&self.db.pool, "warn", &since_str, 6).await;
        let warn_top_json: Vec<Value> = warn_top
            .iter()
            .map(|(event, count)| json!({ "event": event, "count": count }))
            .collect();
        let mut warn_samples: Vec<Value> = Vec::new();
        for (event, _count) in warn_top.iter().take(3) {
            if let Some(sample) =
                fetch_log_sample_payload(&self.db.pool, "warn", event, &since_str).await
            {
                warn_samples.push(json!({
                    "event": event,
                    "sample": sample,
                }));
            }
        }
        let tool_log_failures = log_by_category
            .get("tool")
            .and_then(|value| value.as_object())
            .map(|obj| {
                obj.get("warn").and_then(|v| v.as_i64()).unwrap_or(0)
                    + obj.get("error").and_then(|v| v.as_i64()).unwrap_or(0)
            })
            .unwrap_or(0);

        let pending_prompt_count = count_table_in_window(
            &self.db.pool,
            "pending_user_prompts",
            "created_at",
            "1970-01-01T00:00:00Z",
            None,
        )
        .await;
        let pending_prompt_oldest = first_table_timestamp(
            &self.db.pool,
            "pending_user_prompts",
            "created_at",
            None,
        )
        .await;
        let pending_prompt_age_seconds =
            diff_seconds_from_now(pending_prompt_oldest.as_deref()).unwrap_or(0);

        let (active_run_id, last_run_id) = fetch_run_ids(&self.db.pool).await;

        let scheduler_cadence_ms =
            fetch_scheduler_cadence(&self.db.pool).await.unwrap_or(0);
        let scheduler_avg_tick_ms =
            average_system_log_metric(&self.db.pool, "timing_heartbeat_tick", "duration_ms", &since_str).await;
        let scheduler_idle_downshift = system_controls::mode_is_degraded(
            &system_controls::mode_for("scheduler_tick", &control_map),
        );

        let gate_total = gate_counts.values().copied().sum::<i64>();
        let gate_verify = *gate_counts.get("VERIFY").unwrap_or(&0);
        let gate_defer = *gate_counts.get("DEFER").unwrap_or(&0);
        let gate_deny = *gate_counts.get("DENY").unwrap_or(&0);
        let gate_allow = *gate_counts.get("ALLOW").unwrap_or(&0);
        let gate_notice = *gate_counts.get("ALLOW_WITH_NOTICE").unwrap_or(&0);
        let gate_audit = *gate_counts.get("ALLOW_WITH_AUDIT").unwrap_or(&0);
        let gate_activity = if gate_total > 0 {
            (gate_verify + gate_defer + gate_deny) as f64 / gate_total as f64
        } else {
            0.0
        };

        let error_rate = if gate_total > 0 {
            error_counts as f64 / gate_total as f64
        } else {
            0.0
        };
        let health_score = (1.0 - (error_rate + warn_penalty)).clamp(0.0, 1.0);

        let tool_dispatch_total = tool_successes + tool_failures;
        let tool_planning_error_rate = if tool_dispatch_total > 0 {
            tool_planning_errors as f64 / tool_dispatch_total as f64
        } else {
            0.0
        };
        let tool_execution_error_rate = if tool_dispatch_total > 0 {
            tool_execution_errors as f64 / tool_dispatch_total as f64
        } else {
            0.0
        };
        let tool_throttle_defer_rate = if (tool_dispatch_total + tool_throttle_defers) > 0 {
            tool_throttle_defers as f64 / (tool_dispatch_total + tool_throttle_defers) as f64
        } else {
            0.0
        };

        let memory_activity = ((memory_pass_count + rolling_summary_updates + inner_summary_updates
            + consolidation_runs + episodic_events) as f64
            / 12.0)
            .min(1.0);

        let controller_confidence = controller_state
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);
        let controller_evidence = controller_state
            .get("evidence_coverage")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let controller_telemetry = controller_state
            .get("telemetry_coverage")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let evidence_events = count_table_in_window(
            &self.db.pool,
            "ics_evidence_events",
            "created_at",
            &since_str,
            None,
        )
        .await;
        let self_evidence_events = count_table_in_window(
            &self.db.pool,
            "self_evidence_events",
            "created_at",
            &since_str,
            None,
        )
        .await;
        let evidence_event_total = evidence_events + self_evidence_events;
        let evidence_pipeline_ready = evidence_event_total >= MIN_EVIDENCE_EVENTS;
        let telemetry_keys_present = self_model_controller::telemetry_keys_present(&self.db.pool)
            .await
            .unwrap_or(false);
        let telemetry_missing = !telemetry_keys_present && controller_telemetry <= 0.0;
        if telemetry_missing {
            match crate::core::self_memory::telemetry::record_telemetry_snapshot_force(&self.db, None).await {
                Ok(wrote) => {
                    if wrote {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            app_handle,
                            "info",
                            "health",
                            run_id,
                            trace_id,
                            json!({
                                "event": "telemetry_snapshot_seeded",
                                "reason": "telemetry_missing",
                                "controller_telemetry": controller_telemetry,
                            }),
                        )
                        .await;
                    }
                }
                Err(err) => {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        app_handle,
                        "warn",
                        "health",
                        run_id,
                        trace_id,
                        json!({
                            "event": "telemetry_snapshot_error",
                            "reason": "telemetry_missing",
                            "error": err,
                        }),
                    )
                    .await;
                }
            }
        }
        let mut evidence_low = controller_telemetry < TELEMETRY_MIN as f64
            || (!evidence_pipeline_ready && controller_evidence < EVIDENCE_MIN as f64);
        if telemetry_missing {
            evidence_low = false;
        }
        let memory_drift = memory_validation_drift >= AUTO_DEGRADE_MEMORY_DRIFT;
        let tool_failure_rate = if tool_dispatch_total > 0 {
            tool_failures as f64 / tool_dispatch_total as f64
        } else {
            0.0
        };
        let tool_unreliable = tool_failure_rate >= AUTO_DEGRADE_TOOL_FAILURE_RATE;

        if app_handle.is_some() {
            self.apply_auto_degradation(
                &control_map,
                evidence_low,
                memory_drift,
                tool_unreliable,
                app_handle,
            )
            .await;
        }
        if evidence_low {
            let _ = system_log::log_event(
                &self.db.pool,
                app_handle,
                "info",
                "health",
                run_id,
                trace_id,
                json!({
                    "event": "evidence_low_gate",
                    "controller_evidence": controller_evidence,
                    "controller_telemetry": controller_telemetry,
                    "evidence_event_total": evidence_event_total,
                    "min_evidence_events": MIN_EVIDENCE_EVENTS,
                }),
            )
            .await;
        } else if evidence_pipeline_ready {
            let updated = sqlx::query(
                "UPDATE settings
                 SET world_model_reconcile_mode = 'active'
                 WHERE id = 1 AND (world_model_reconcile_mode IS NULL OR world_model_reconcile_mode != 'active')",
            )
            .execute(&self.db.pool)
            .await
            .map(|result| result.rows_affected() > 0)
            .unwrap_or(false);
            if updated {
                let _ = system_log::log_event(
                    &self.db.pool,
                    app_handle,
                    "info",
                    "health",
                    run_id,
                    trace_id,
                    json!({
                        "event": "world_model_reconcile_mode_activated",
                        "mode": "active",
                        "evidence_event_total": evidence_event_total,
                    }),
                )
                .await;
            }
        }

        let metrics = json!({
            "window_minutes": HEALTH_WINDOW_MINUTES,
            "scheduler": {
                "cadence_ms": scheduler_cadence_ms,
                "avg_tick_duration_ms": scheduler_avg_tick_ms,
                "idle_downshift": scheduler_idle_downshift,
            },
            "kernel": {
                "run_count": count_table_in_window(&self.db.pool, "runs", "started_at", &since_str, None).await,
                "cycle_count": kernel_cycle_count,
                "decision_outcomes": gate_counts,
                "tool_usage_rate": if gate_total > 0 {
                    (tool_successes + tool_failures) as f64 / gate_total as f64
                } else {
                    0.0
                },
            },
            "gate": {
                "counts": {
                    "ALLOW": gate_allow,
                    "ALLOW_WITH_NOTICE": gate_notice,
                    "ALLOW_WITH_AUDIT": gate_audit,
                    "VERIFY": gate_verify,
                    "DEFER": gate_defer,
                    "DENY": gate_deny,
                },
                "inputs": gate_inputs,
                "total": gate_total,
                "verify_rate": if gate_total > 0 { gate_verify as f64 / gate_total as f64 } else { 0.0 },
            },
            "controller": controller_state,
            "organism": organism_state,
            "memory": {
                "memory_pass_count": memory_pass_count,
                "memory_pass_skipped": memory_pass_skipped,
                "memory_pass_zero_writes": memory_pass_zero_writes,
                "write_count": memory_write_count,
                "promotion_low_confidence": memory_promotion_low_confidence,
                "validation_runs": memory_validation_runs,
                "drift_events": memory_validation_drift,
                "last_validation_at": memory_validation_last_at,
                "phi_write_blocked": phi_write_blocked,
                "last_error_at": memory_error_at,
            },
            "telemetry": {
                "calibration_runs": telemetry_calibration_runs,
                "drift_events": telemetry_calibration_drift,
                "last_calibration_at": telemetry_calibration_last_at,
            },
            "outcomes": {
                "total": outcome_total,
                "confirm": outcome_confirm,
                "disconfirm": outcome_disconfirm,
                "inconclusive": outcome_inconclusive,
                "accuracy": outcome_accuracy_opt,
                "last_outcome_at": outcome_last_at,
            },
            "scorecard": {
                "combined_score": combined_score,
                "drift_penalty": drift_penalty,
                "outcome_accuracy": outcome_accuracy_opt,
            },
            "summaries": {
                "rolling_updates": rolling_summary_updates,
                "rolling_failures": rolling_summary_failures,
                "summary_chunk_count": summary_chunk_count,
                "summary_archive_skipped": summary_archive_skipped,
                "inner_updates": inner_summary_updates,
                "inner_failures": inner_summary_failures,
                "rolling_last_at": last_rolling_summary_at,
                "inner_last_at": last_inner_summary_at,
                "rolling_age_seconds": rolling_age_seconds,
                "inner_age_seconds": inner_age_seconds,
            },
            "self_memory": {
                "writes": self_memory_writes,
                "denies": self_memory_denies,
                "last_write_at": last_self_memory_at,
            },
            "self_reflection": {
                "last_reflection_at": self_reflection_at,
                "reflection_frozen": self_reflection_frozen,
                "skipped": self_reflection_skipped,
                "missing_evidence": self_reflection_missing,
                "status": self_reflection_status,
            },
            "self_claims": {
                "rejected": self_claim_rejected,
                "missing_evidence": self_claim_missing,
                "stale_evidence": self_claim_stale,
            },
            "identity": {
                "thread_present": identity_thread_present,
                "confidence": identity_confidence,
            },
            "consolidation": {
                "runs": consolidation_runs,
                "errors": consolidation_errors,
                "last_at": consolidation_last_at,
            },
            "episodic": {
                "events": episodic_events,
                "compactions": episodic_compactions,
                "last_at": episodic_last_at,
            },
            "monologue": {
                "entries": monologue_entries,
                "tick_start": monologue_tick_start,
                "tick_end": monologue_tick_end,
                "tick_timeout": monologue_tick_timeout,
                "success_rate": monologue_success_rate,
                "suppression_rate": monologue_suppression_rate,
                "loop_outcomes": loop_outcomes,
                "loop_noop": loop_noop,
                "loop_noop_rate": if loop_outcomes > 0 { loop_noop as f64 / loop_outcomes as f64 } else { 0.0 },
                "loop_delta_applied": loop_delta_applied,
                "loop_delta_rejected": loop_delta_rejected,
                "loop_state_change_rate": if loop_outcomes > 0 { loop_delta_applied as f64 / loop_outcomes as f64 } else { 0.0 },
                "loop_noop_streak": loop_noop_streak,
                "last_suppression_at": last_monologue_suppression,
                "last_suppression_reason": last_monologue_reason,
            },
            "wave": {
                "contributions": wave_contributions,
                "projections": wave_projections,
                "modulations": wave_modulations,
                "sources": wave_source_counts,
            },
            "prediction": {
                "generation_complete": prediction_complete,
                "rejected": prediction_rejected,
                "failed": prediction_failed,
                "skipped": prediction_skipped,
                "json_repair": prediction_repair,
                "json_repair_rate": prediction_repair_rate,
                "residual_shadow_impact_pct": residual_shadow_impact_pct,
                "residual_shadow_would_change": residual_shadow_would_change,
            },
            "attention_schema": {
                "updates": attention_schema_updates,
                "avg_stability": attention_schema_stability,
                "capacity_usage": attention_schema_capacity,
                "selection_policy": attention_schema_policy,
                "stability": attention_schema_last_stability,
                "last_updated_at": attention_schema_last_at,
            },
            "workspace_contributors": {
                "snapshots": workspace_snapshot_count,
                "missing": workspace_missing_count,
                "missing_rate": if workspace_snapshot_count > 0 {
                    workspace_missing_count as f64 / workspace_snapshot_count as f64
                } else {
                    0.0
                },
                "last_snapshot_at": workspace_snapshot_at,
                "summary": workspace_summary,
            },
            "audits": {
                "count": audits_count,
                "mean_discrepancy": audits_mean_score,
            },
            "qualia": {
                "labels": qualia_labels,
                "rewards": qualia_rewards,
                "mean_intensity": qualia_mean_intensity,
                "mean_reward": qualia_mean_reward,
            },
            "prompt_trim": prompt_trim_counts,
            "prompt_overflow": prompt_overflow_count,
            "phi": {
                "write_blocked": phi_write_blocked,
                "redacted": phi_redacted,
            },
            "relations": {
                "beliefs_24h": relation_beliefs,
                "shape_missing": relation_shape_missing,
                "shape_mismatch": relation_shape_mismatch,
                "promotions_created": relation_promotions_created,
                "promotions_skipped": relation_promotions_skipped,
                "promotion_precision": relation_promotion_precision,
            },
            "feedback": feedback_breakdown,
            "errors": {
                "total": error_counts,
                "open": error_open,
                "by_classification": error_by_classification,
                "by_status": error_by_status,
            },
            "tools": {
                "dispatches": tool_successes + tool_failures,
                "successes": tool_successes,
                "failures": tool_failures,
                "log_failures": tool_log_failures,
                "planning_errors": tool_planning_errors,
                "execution_errors": tool_execution_errors,
                "cancelled": tool_cancelled,
                "planning_error_rate": tool_planning_error_rate,
                "execution_error_rate": tool_execution_error_rate,
                "throttle_defers": tool_throttle_defers,
                "throttle_defer_rate": tool_throttle_defer_rate,
            },
            "logs": {
                "warn": log_warn_count,
                "warn_effective": warn_effective,
                "warn_rate_per_min": warn_rate,
                "warn_penalty": warn_penalty,
                "warn_excluded": benign_warn_counts,
                "error": log_error_count,
                "by_category": log_by_category,
            },
            "health_debug": {
                "warn_top": warn_top_json,
                "warn_samples": warn_samples,
            },
            "pending_prompts": {
                "count": pending_prompt_count,
                "oldest_age_seconds": pending_prompt_age_seconds,
                "oldest_at": pending_prompt_oldest.unwrap_or_default(),
            },
            "run": {
                "active_run_id": active_run_id,
                "last_run_id": last_run_id,
                "module_stage": module_status,
                "module_detail": module_detail,
                "module_updated_at": module_status_at,
            },
            "avatar": {
                "processing_phase": module_status,
                "certainty": controller_confidence,
                "health": health_score,
                "memory_activity": memory_activity,
                "organism": organism_state,
                "gate_activity": gate_activity,
                "pending_prompts": pending_prompt_count,
            },
            "error_state": error_state,
        });

        let snapshot_id = Uuid::new_v4().to_string();
        let timestamp = now.to_rfc3339();
        let metrics_json = serde_json::to_string(&metrics).map_err(|e| e.to_string())?;
        let subsystem_states_json =
            serde_json::to_string(&subsystem_states).map_err(|e| e.to_string())?;

        self.db
            .insert_system_health_snapshot(
                &snapshot_id,
                &timestamp,
                run_id,
                trace_id,
                &metrics_json,
                &subsystem_states_json,
            )
            .await
            .map_err(|e| e.to_string())?;

        let _ = system_log::log_event(
            &self.db.pool,
            app_handle,
            "info",
            "system",
            run_id,
            trace_id,
            json!({
                "event": "system_health_snapshot",
                "snapshot_id": snapshot_id,
                "window_minutes": HEALTH_WINDOW_MINUTES,
                "gate_total": gate_total,
                "error_total": error_counts,
            }),
        )
        .await;

        Ok(SystemHealthSnapshotPayload {
            snapshot_id,
            timestamp,
            run_id: run_id.map(|v| v.to_string()),
            trace_id: trace_id.map(|v| v.to_string()),
            metrics,
            subsystem_states,
        })
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct UnityHealthSnapshot {
    pub timestamp: String,
    pub window_minutes: i64,
    pub pass: bool,
    pub metrics: Value,
}

pub async fn capture_unity_snapshot(
    db: Arc<Db>,
    app_handle: Option<&AppHandle>,
) -> Result<UnityHealthSnapshot, String> {
    let now = Utc::now();
    let since = now - ChronoDuration::minutes(UNITY_WINDOW_MINUTES);
    let since_str = since.to_rfc3339();

    let prediction_failed = count_system_log_event(&db.pool, "prediction_generation_failed", &since_str).await;
    let prediction_reasons = fetch_grouped_event_counts(&db.pool, "prediction_generation_failed", "reason", &since_str).await;

    let monologue_parse_failed = count_system_log_event(&db.pool, "monologue_parse_failed", &since_str).await;
    let monologue_tick_timeout = count_system_log_event(&db.pool, "monologue_tick_timeout", &since_str).await;
    let monologue_tick_retry = count_system_log_event(&db.pool, "monologue_tick_retry", &since_str).await;

    let contract_total = count_system_log_event(&db.pool, "contract_violation", &since_str).await;
    let contract_policy = fetch_grouped_event_counts(&db.pool, "contract_violation", "policy_id", &since_str).await;
    let contract_reason = fetch_grouped_event_counts(&db.pool, "contract_violation", "reason", &since_str).await;

    let memory_blocked_total = count_system_log_event(&db.pool, "memory_write_blocked", &since_str).await;
    let memory_blocked_reason = fetch_grouped_event_counts(&db.pool, "memory_write_blocked", "reason", &since_str).await;
    let memory_blocked_source = fetch_grouped_event_counts(&db.pool, "memory_write_blocked", "source", &since_str).await;

    let subject_snapshots = count_table_in_window(&db.pool, "subject_snapshots", "timestamp", &since_str, None).await;
    let gate_decisions = count_table_in_window(&db.pool, "gate_decisions", "created_at", &since_str, None).await;
    let snapshots_with_gate = count_snapshots_with_gate(&db.pool, &since_str).await;

    let c1_violations = contract_policy.get("C1").copied().unwrap_or(0);
    let gate_blocks = memory_blocked_reason.get("gate_decision").copied().unwrap_or(0);

    let pass = prediction_failed == 0
        && monologue_tick_timeout < 2
        && monologue_parse_failed < 1
        && c1_violations == 0
        && gate_blocks == 0;

    let metrics = json!({
        "window_minutes": UNITY_WINDOW_MINUTES,
        "prediction_generation_failed": {
            "total": prediction_failed,
            "reasons": prediction_reasons,
        },
        "monologue": {
            "parse_failed": monologue_parse_failed,
            "tick_timeout": monologue_tick_timeout,
            "tick_retry": monologue_tick_retry,
        },
        "contract_violation": {
            "total": contract_total,
            "policy": contract_policy,
            "reason": contract_reason,
        },
        "memory_write_blocked": {
            "total": memory_blocked_total,
            "reason": memory_blocked_reason,
            "source": memory_blocked_source,
        },
        "snapshots": {
            "subject_snapshots": subject_snapshots,
            "gate_decisions": gate_decisions,
            "snapshots_with_gate": snapshots_with_gate,
        },
    });

    let _ = system_log::log_event(
        &db.pool,
        app_handle,
        "info",
        "system",
        None,
        None,
        json!({
            "event": "unity_health_snapshot",
            "window_minutes": UNITY_WINDOW_MINUTES,
            "pass": pass,
            "metrics": metrics,
        }),
    )
    .await;

    let _ = log_continuity_health(&db, app_handle).await;

    Ok(UnityHealthSnapshot {
        timestamp: now.to_rfc3339(),
        window_minutes: UNITY_WINDOW_MINUTES,
        pass,
        metrics,
    })
}

pub async fn capture_unity_diagnostic(
    db: Arc<Db>,
    app_handle: Option<&AppHandle>,
) -> Result<(), String> {
    let mut events = Vec::new();
    if let Ok(rows) = sqlx::query(
        "SELECT event_id, timestamp, type, payload
         FROM event_ledger
         WHERE type IN (
             'prediction_generation_failed',
             'monologue_parse_failed',
             'monologue_tick_timeout',
             'monologue_tick_retry',
             'contract_violation',
             'memory_write_blocked'
         )
         ORDER BY datetime(timestamp) DESC, rowid DESC
         LIMIT 20",
    )
    .fetch_all(&db.pool)
    .await
    {
        for row in rows {
            let payload_raw: String = row.get("payload");
            let payload = serde_json::from_str::<Value>(&payload_raw).unwrap_or_else(|_| json!({ "raw": payload_raw }));
            events.push(json!({
                "event_id": row.get::<String, _>("event_id"),
                "timestamp": row.get::<String, _>("timestamp"),
                "type": row.get::<String, _>("type"),
                "payload": payload,
            }));
        }
    }

    let settings = db.get_settings().await.ok();
    let settings_snapshot = settings.map(|s| {
        json!({
            "api_base_url": s.api_base_url,
            "active_model_id": s.active_model_id,
            "summarization_api_url": s.summarization_api_url,
            "summarization_model": s.summarization_model,
            "monologue_interval_seconds": s.monologue_interval_seconds,
            "monologue_timeout_secs": s.monologue_timeout_secs,
            "monologue_retry_timeout_secs": s.monologue_retry_timeout_secs,
            "monologue_max_per_hour": s.monologue_max_per_hour,
        })
    });

    let _ = system_log::log_event(
        &db.pool,
        app_handle,
        "warn",
        "system",
        None,
        None,
        json!({
            "event": "unity_diagnostic_snapshot",
            "events": events,
            "settings": settings_snapshot,
        }),
    )
    .await;

    Ok(())
}

pub async fn unity_green_window_ready(pool: &sqlx::SqlitePool) -> bool {
    let rows = sqlx::query(
        "SELECT timestamp, json_extract(payload, '$.pass') as pass
         FROM system_logs
         WHERE json_extract(payload, '$.event') = 'unity_health_snapshot'
         ORDER BY datetime(timestamp) DESC, rowid DESC
         LIMIT 6",
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    if rows.len() < 6 {
        return false;
    }
    let mut passes = Vec::new();
    let mut timestamps = Vec::new();
    for row in rows {
        let pass: i64 = row.try_get("pass").unwrap_or(0);
        let ts: String = row.try_get("timestamp").unwrap_or_default();
        passes.push(pass > 0);
        timestamps.push(ts);
    }
    if passes.iter().any(|p| !*p) {
        return false;
    }
    let newest = timestamps.first().cloned().unwrap_or_default();
    let oldest = timestamps.last().cloned().unwrap_or_default();
    let Some(newest_dt) = parse_timestamp(&newest) else { return false; };
    let Some(oldest_dt) = parse_timestamp(&oldest) else { return false; };
    let span_hours = (newest_dt - oldest_dt).num_hours();
    span_hours >= (UNITY_GREEN_WINDOW_HOURS - 1)
}

pub async fn last_unity_green_at(pool: &sqlx::SqlitePool) -> Option<DateTime<Utc>> {
    last_system_log_timestamp(pool, None, None, Some("unity_green_window")).await
        .and_then(|ts| parse_timestamp(&ts))
}

fn build_subsystem_states(controls: &HashMap<String, system_controls::ControlState>) -> Value {
    let mut list = Vec::new();
    for def in system_controls::registry() {
        let state = controls.get(def.id);
        let mode = state
            .map(|s| s.mode.clone())
            .unwrap_or_else(|| def.default_mode.to_string());
        list.push(json!({
            "id": def.id,
            "label": def.label,
            "class": format!("{:?}", def.class).to_lowercase(),
            "default_mode": def.default_mode,
            "supported_modes": def.supported_modes,
            "depends_on": def.depends_on,
            "enforcement_notes": def.enforcement_notes,
            "mode": mode,
            "updated_at": state.and_then(|s| s.updated_at.clone()).unwrap_or_default(),
            "updated_by": state.and_then(|s| s.updated_by.clone()).unwrap_or_default(),
            "reason": state.and_then(|s| s.reason.clone()).unwrap_or_default(),
            "value_json": state.and_then(|s| s.value_json.clone()),
        }));
    }
    json!(list)
}

async fn fetch_gate_counts(pool: &sqlx::SqlitePool, since: &str) -> HashMap<String, i64> {
    let mut map: HashMap<String, i64> = HashMap::new();
    if let Ok(rows) = sqlx::query(
        "SELECT decision, COUNT(*) as count FROM gate_decisions
         WHERE datetime(created_at) >= datetime(?)
         GROUP BY decision",
    )
    .bind(since)
    .fetch_all(pool)
    .await
    {
        for row in rows {
            let decision: String = row.get("decision");
            let count: i64 = row.get("count");
            map.insert(decision.to_uppercase(), count);
        }
    }
    map
}

async fn fetch_gate_inputs(pool: &sqlx::SqlitePool) -> Vec<Value> {
    let mut entries = Vec::new();
    if let Ok(rows) = sqlx::query(
        "SELECT id, timestamp, payload FROM system_logs
         WHERE json_extract(payload, '$.event') = 'gate_decision_inputs'
         ORDER BY datetime(timestamp) DESC, rowid DESC
         LIMIT ?",
    )
    .bind(GATE_INPUT_LIMIT)
    .fetch_all(pool)
    .await
    {
        for row in rows {
            let raw: String = row.get("payload");
            let payload = serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| json!({ "raw": raw }));
            entries.push(json!({
                "id": row.get::<String, _>("id"),
                "timestamp": row.get::<String, _>("timestamp"),
                "payload": payload,
            }));
        }
    }
    entries
}

async fn fetch_latest_module_status(pool: &sqlx::SqlitePool) -> (String, String, String) {
    if let Ok(row) = sqlx::query(
        "SELECT payload, timestamp FROM system_logs
         WHERE json_extract(payload, '$.event') = 'module_status'
         ORDER BY datetime(timestamp) DESC, rowid DESC
         LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    {
        if let Some(row) = row {
            let raw: String = row.get("payload");
            if let Ok(payload) = serde_json::from_str::<Value>(&raw) {
                let stage = payload.get("stage").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let detail = payload.get("detail").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let ts: String = row.get("timestamp");
                return (stage, detail, ts);
            }
        }
    }
    ("".to_string(), "".to_string(), "".to_string())
}

async fn fetch_latest_subject_state(
    pool: &sqlx::SqlitePool,
) -> (Value, Value, Value) {
    let mut controller = json!({
        "confidence": 0.5,
        "uncertainty": 0.5,
        "failure_streak": 0,
        "verification_needed": false,
        "reanchor_needed": false,
        "autonomy_level": 0.5,
        "evidence_coverage": 0.0,
        "telemetry_coverage": 0.0,
        "last_error": "",
        "last_strategy": "",
    });
    let mut organism = json!({
        "stress": 0.0,
        "fatigue": 0.0,
        "social_alignment": 0.5,
        "uncertainty_pressure": 0.0,
        "integrity_risk": 0.0,
        "arousal": 0.5,
    });
    let mut error_state = json!({
        "open_error_count": 0,
        "pattern_flags": [],
        "diagnosis_flags": [],
    });

    if let Ok(row) = sqlx::query(
        "SELECT subject_state_json FROM subject_snapshots
         ORDER BY datetime(timestamp) DESC
         LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    {
        if let Some(row) = row {
            let raw: String = row.get("subject_state_json");
            if let Ok(value) = serde_json::from_str::<Value>(&raw) {
                if let Some(ctrl) = value
                    .get("state")
                    .and_then(|v| v.get("self_model"))
                    .and_then(|v| v.get("controller_state"))
                {
                    controller = ctrl.clone();
                }
                if let Some(org) = value.get("state").and_then(|v| v.get("organism")) {
                    organism = org.clone();
                }
                if let Some(err) = value.get("state").and_then(|v| v.get("error_state")) {
                    error_state = err.clone();
                }
            }
        }
    }

    (controller, organism, error_state)
}

async fn fetch_latest_kernel_loop_streak(pool: &sqlx::SqlitePool) -> i64 {
    let raw: Option<String> = sqlx::query_scalar(
        "SELECT state_json FROM kernel_states
         ORDER BY datetime(updated_at) DESC LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    let Some(raw) = raw else { return 0; };
    let Ok(value) = serde_json::from_str::<Value>(&raw) else { return 0; };
    value
        .get("monologue_loop_streak")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
}

async fn fetch_latest_log_payload(
    pool: &sqlx::SqlitePool,
    event_name: &str,
) -> (Option<String>, Option<Value>) {
    let row = sqlx::query(
        "SELECT timestamp, payload FROM system_logs
         WHERE json_extract(payload, '$.event') = ?
         ORDER BY datetime(timestamp) DESC
         LIMIT 1",
    )
    .bind(event_name)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    let Some(row) = row else {
        return (None, None);
    };
    let timestamp: Option<String> = row.try_get("timestamp").ok();
    let payload_raw: Option<String> = row.try_get("payload").ok();
    let payload = payload_raw
        .as_deref()
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok());
    (timestamp, payload)
}

async fn fetch_scheduler_cadence(pool: &sqlx::SqlitePool) -> Option<i64> {
    if let Ok(row) = sqlx::query(
        "SELECT payload FROM system_logs
         WHERE json_extract(payload, '$.event') = 'scheduler_cadence_updated'
         ORDER BY datetime(timestamp) DESC, rowid DESC
         LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    {
        if let Some(row) = row {
            let raw: String = row.get("payload");
            if let Ok(value) = serde_json::from_str::<Value>(&raw) {
                if let Some(val) = value.get("cadence_ms").and_then(|v| v.as_i64()) {
                    return Some(val);
                }
            }
        }
    }
    None
}

async fn latest_conversation_id(pool: &sqlx::SqlitePool) -> Option<String> {
    sqlx::query_scalar(
        "SELECT conversation_id FROM messages
         ORDER BY datetime(created_at) DESC
         LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()
}

async fn log_continuity_health(db: &Db, app_handle: Option<&AppHandle>) -> Result<(), String> {
    let now = Utc::now();
    let since = now - ChronoDuration::hours(24);
    let since_str = since.to_rfc3339();

    let tick_start = count_system_log_event(&db.pool, "monologue_tick_start", &since_str).await;
    let tick_end = count_system_log_event(&db.pool, "monologue_tick_end", &since_str).await;
    let tick_timeout = count_system_log_event(&db.pool, "monologue_tick_timeout", &since_str).await;

    let conversation_id = latest_conversation_id(&db.pool)
        .await
        .unwrap_or_else(|| "default".to_string());
    let summary_text: Option<String> = sqlx::query_scalar(
        "SELECT summary FROM conversation_summaries WHERE conversation_id = ? LIMIT 1",
    )
    .bind(&conversation_id)
    .fetch_optional(&db.pool)
    .await
    .map_err(|e| e.to_string())?;
    let rolling_summary_present = summary_text
        .as_deref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);

    let summary_trim_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'prompt_trim'
           AND json_extract(payload, '$.title') IN ('Rolling Summary', 'Inner Summary')
           AND datetime(timestamp) >= datetime('now','-24 hours')",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);

    let empty_assistant_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM messages
         WHERE conversation_id = ?
           AND role = 'assistant'
           AND (content IS NULL OR trim(content) = '')
           AND status IN ('streaming','error','pending')",
    )
    .bind(&conversation_id)
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);

    let success_rate = if tick_start > 0 {
        (tick_end as f64) / (tick_start as f64)
    } else {
        0.0
    };

    let _ = system_log::log_event(
        &db.pool,
        app_handle,
        "info",
        "system",
        None,
        None,
        json!({
            "event": "continuity_health",
            "conversation_id": conversation_id,
            "window_hours": 24,
            "monologue_tick_start": tick_start,
            "monologue_tick_end": tick_end,
            "monologue_tick_timeout": tick_timeout,
            "monologue_success_rate": success_rate,
            "rolling_summary_present": rolling_summary_present,
            "summary_trim_count": summary_trim_count,
            "empty_assistant_count": empty_assistant_count,
        }),
    )
    .await;

    Ok(())
}

async fn count_system_log_event(pool: &sqlx::SqlitePool, event: &str, since: &str) -> i64 {
    sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = ?
           AND datetime(timestamp) >= datetime(?)",
    )
    .bind(event)
    .bind(since)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0)
}

async fn count_system_log_event_filtered(
    pool: &sqlx::SqlitePool,
    event: &str,
    since: &str,
    extra_where: Option<&str>,
) -> i64 {
    let mut sql = String::from(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = ?
           AND datetime(timestamp) >= datetime(?)",
    );
    if let Some(extra) = extra_where {
        sql.push_str(" AND ");
        sql.push_str(extra);
    }
    sqlx::query_scalar(&sql)
        .bind(event)
        .bind(since)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0)
}

async fn fetch_grouped_event_counts(
    pool: &sqlx::SqlitePool,
    event: &str,
    field: &str,
    since: &str,
) -> HashMap<String, i64> {
    let mut out: HashMap<String, i64> = HashMap::new();
    let sql = format!(
        "SELECT json_extract(payload, '$.{field}') as key, COUNT(*) as count
         FROM system_logs
         WHERE json_extract(payload, '$.event') = ?
           AND datetime(timestamp) >= datetime(?)
         GROUP BY key"
    );
    if let Ok(rows) = sqlx::query(&sql)
        .bind(event)
        .bind(since)
        .fetch_all(pool)
        .await
    {
        for row in rows {
            let key: Option<String> = row.try_get("key").ok();
            let count: i64 = row.try_get("count").unwrap_or(0);
            if let Some(key) = key {
                out.insert(key, count);
            }
        }
    }
    out
}

async fn fetch_event_counts_for_list(
    pool: &sqlx::SqlitePool,
    events: &[&str],
    since: &str,
) -> HashMap<String, i64> {
    let mut out: HashMap<String, i64> = HashMap::new();
    if events.is_empty() {
        return out;
    }
    let placeholders = events.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT json_extract(payload, '$.event') as event, COUNT(*) as count
         FROM system_logs
         WHERE level = 'warn'
           AND json_extract(payload, '$.event') IN ({})
           AND datetime(timestamp) >= datetime(?)
         GROUP BY event",
        placeholders
    );
    let mut builder = sqlx::query(&sql);
    for event in events.iter() {
        builder = builder.bind(event);
    }
    builder = builder.bind(since);
    if let Ok(rows) = builder.fetch_all(pool).await {
        for row in rows {
            let event: Option<String> = row.try_get("event").ok();
            let count: i64 = row.try_get("count").unwrap_or(0);
            if let Some(event) = event {
                out.insert(event, count);
            }
        }
    }
    out
}

async fn fetch_top_log_events(
    pool: &sqlx::SqlitePool,
    level: &str,
    since: &str,
    limit: i64,
) -> Vec<(String, i64)> {
    let mut out = Vec::new();
    if let Ok(rows) = sqlx::query(
        "SELECT json_extract(payload, '$.event') as event, COUNT(*) as count
         FROM system_logs
         WHERE level = ? AND datetime(timestamp) >= datetime(?)
         GROUP BY event
         ORDER BY count DESC
         LIMIT ?",
    )
    .bind(level)
    .bind(since)
    .bind(limit.max(1))
    .fetch_all(pool)
    .await
    {
        for row in rows {
            let event: Option<String> = row.try_get("event").ok();
            let count: i64 = row.try_get("count").unwrap_or(0);
            if let Some(event) = event {
                out.push((event, count));
            }
        }
    }
    out
}

async fn fetch_log_sample_payload(
    pool: &sqlx::SqlitePool,
    level: &str,
    event: &str,
    since: &str,
) -> Option<Value> {
    let row: Option<String> = sqlx::query_scalar(
        "SELECT payload FROM system_logs
         WHERE level = ?
           AND json_extract(payload, '$.event') = ?
           AND datetime(timestamp) >= datetime(?)
         ORDER BY datetime(timestamp) DESC
         LIMIT 1",
    )
    .bind(level)
    .bind(event)
    .bind(since)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    row.and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
}

fn subsystem_expected_missing(
    subsystem: &str,
    control_map: &HashMap<String, system_controls::ControlState>,
    cockpit_write_enabled: bool,
) -> bool {
    match subsystem {
        "memory" => {
            let memory_write = system_controls::mode_for("memory_write", control_map);
            let memory_retrieval = system_controls::mode_for("memory_retrieval", control_map);
            !cockpit_write_enabled
                || system_controls::mode_is_off(&memory_write)
                || system_controls::mode_is_read_only(&memory_write)
                || system_controls::mode_is_degraded(&memory_write)
                || system_controls::mode_is_off(&memory_retrieval)
                || system_controls::mode_is_read_only(&memory_retrieval)
        }
        "prediction" => {
            let prediction = system_controls::mode_for("prediction_generation", control_map);
            system_controls::mode_is_off(&prediction) || system_controls::mode_is_degraded(&prediction)
        }
        "attention" => {
            let attention = system_controls::mode_for("attention_schema", control_map);
            system_controls::mode_is_off(&attention) || system_controls::mode_is_degraded(&attention)
        }
        "self_model" => {
            let self_memory = system_controls::mode_for("self_memory", control_map);
            system_controls::mode_is_off(&self_memory)
                || system_controls::mode_is_degraded(&self_memory)
                || system_controls::mode_is_read_only(&self_memory)
        }
        "qualia" => {
            let qualia = system_controls::mode_for("qualia_loop", control_map);
            system_controls::mode_is_off(&qualia) || system_controls::mode_is_degraded(&qualia)
        }
        "tools" => {
            let tools = system_controls::mode_for("tool_execution", control_map);
            system_controls::mode_is_off(&tools) || system_controls::mode_is_degraded(&tools)
        }
        _ => false,
    }
}

async fn count_benign_workspace_missing(
    pool: &sqlx::SqlitePool,
    since: &str,
    control_map: &HashMap<String, system_controls::ControlState>,
    cockpit_write_enabled: bool,
) -> i64 {
    let rows = sqlx::query(
        "SELECT payload FROM system_logs
         WHERE json_extract(payload, '$.event') = 'workspace_missing_contributors'
           AND datetime(timestamp) >= datetime(?)",
    )
    .bind(since)
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let mut benign_count = 0i64;
    for row in rows {
        let payload_raw: String = row.get("payload");
        let payload = serde_json::from_str::<Value>(&payload_raw).unwrap_or_else(|_| json!({}));
        let missing = payload.get("missing").and_then(|v| v.as_array());
        let Some(missing) = missing else {
            continue;
        };
        if missing.is_empty() {
            continue;
        }
        let all_expected = missing.iter().all(|item| {
            item.as_str()
                .map(|name| subsystem_expected_missing(name, control_map, cockpit_write_enabled))
                .unwrap_or(false)
        });
        if all_expected {
            benign_count += 1;
        }
    }

    benign_count
}

async fn fetch_self_reflection_status(
    pool: &sqlx::SqlitePool,
) -> (Option<String>, bool, Value) {
    if let Ok(Some(row)) = sqlx::query(
        "SELECT last_reflection_at, reflection_frozen, reflection_status_json FROM self_model LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    {
        let last_reflection_at: Option<String> = row.try_get("last_reflection_at").ok();
        let reflection_frozen: bool = row.try_get::<i64, _>("reflection_frozen").unwrap_or(0) != 0;
        let status_raw: Option<String> = row.try_get("reflection_status_json").ok();
        let status = status_raw
            .as_deref()
            .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
            .unwrap_or(Value::Null);
        return (last_reflection_at, reflection_frozen, status);
    }
    (None, false, Value::Null)
}

async fn fetch_identity_state(pool: &sqlx::SqlitePool) -> (bool, Option<f64>) {
    if let Ok(Some(row)) = sqlx::query(
        "SELECT identity_thread, identity_confidence FROM self_model LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    {
        let thread: Option<String> = row.try_get("identity_thread").ok();
        let confidence: Option<f64> = row.try_get("identity_confidence").ok();
        let present = thread.as_deref().map(|t| !t.trim().is_empty()).unwrap_or(false);
        return (present, confidence);
    }
    (false, None)
}

async fn count_snapshots_with_gate(pool: &sqlx::SqlitePool, since: &str) -> i64 {
    sqlx::query_scalar(
        "SELECT COUNT(*) FROM subject_snapshots s
         WHERE datetime(s.timestamp) >= datetime(?)
           AND EXISTS (SELECT 1 FROM gate_decisions g WHERE g.snapshot_hash = s.snapshot_hash)",
    )
    .bind(since)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0)
}

async fn average_system_log_metric(
    pool: &sqlx::SqlitePool,
    event: &str,
    field: &str,
    since: &str,
) -> f64 {
    let sql = format!(
        "SELECT AVG(CAST(json_extract(payload, '$.{}') AS REAL)) FROM system_logs
         WHERE json_extract(payload, '$.event') = ?
           AND datetime(timestamp) >= datetime(?)",
        field
    );
    sqlx::query_scalar(&sql)
        .bind(event)
        .bind(since)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0.0)
}

async fn count_table_in_window(
    pool: &sqlx::SqlitePool,
    table: &str,
    timestamp_col: &str,
    since: &str,
    extra_where: Option<&str>,
) -> i64 {
    let mut sql = format!(
        "SELECT COUNT(*) FROM {} WHERE datetime({}) >= datetime(?)",
        table, timestamp_col
    );
    if let Some(extra) = extra_where {
        sql.push_str(" AND ");
        sql.push_str(extra);
    }
    sqlx::query_scalar(&sql)
        .bind(since)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0)
}

async fn fetch_latest_suppression_reason(pool: &sqlx::SqlitePool) -> String {
    sqlx::query_scalar(
        "SELECT suppression_reason FROM inner_monologue_candidates
         WHERE suppression_reason IS NOT NULL AND suppression_reason != ''
         ORDER BY datetime(created_at) DESC
         LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()
    .unwrap_or_default()
}

async fn average_table_value(
    pool: &sqlx::SqlitePool,
    table: &str,
    column: &str,
    since: &str,
    extra_where: Option<&str>,
) -> f64 {
    let mut sql = format!(
        "SELECT AVG({}) as avg_value FROM {} WHERE datetime(created_at) >= datetime(?)",
        column, table
    );
    if let Some(extra) = extra_where {
        sql.push_str(" AND ");
        sql.push_str(extra);
    }
    sqlx::query_scalar(&sql)
        .bind(since)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0.0)
}

async fn last_system_log_timestamp(
    pool: &sqlx::SqlitePool,
    category: Option<&str>,
    level: Option<&str>,
    event: Option<&str>,
) -> Option<String> {
    let mut sql = "SELECT timestamp FROM system_logs WHERE 1=1".to_string();
    if category.is_some() {
        sql.push_str(" AND category = ?");
    }
    if level.is_some() {
        sql.push_str(" AND level = ?");
    }
    if event.is_some() {
        sql.push_str(" AND json_extract(payload, '$.event') = ?");
    }
    sql.push_str(" ORDER BY datetime(timestamp) DESC, rowid DESC LIMIT 1");

    let mut query = sqlx::query(&sql);
    if let Some(cat) = category {
        query = query.bind(cat);
    }
    if let Some(lvl) = level {
        query = query.bind(lvl);
    }
    if let Some(ev) = event {
        query = query.bind(ev);
    }

    query
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
        .map(|row| row.get::<String, _>("timestamp"))
}

async fn last_table_timestamp(
    pool: &sqlx::SqlitePool,
    table: &str,
    timestamp_col: &str,
    extra_where: Option<&str>,
) -> Option<String> {
    let mut sql = format!(
        "SELECT {} FROM {}",
        timestamp_col, table
    );
    if let Some(extra) = extra_where {
        sql.push_str(" WHERE ");
        sql.push_str(extra);
    }
    sql.push_str(&format!(" ORDER BY datetime({}) DESC LIMIT 1", timestamp_col));
    sqlx::query(&sql)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
        .map(|row| row.get::<String, _>(timestamp_col))
}

async fn first_table_timestamp(
    pool: &sqlx::SqlitePool,
    table: &str,
    timestamp_col: &str,
    extra_where: Option<&str>,
) -> Option<String> {
    let mut sql = format!(
        "SELECT {} FROM {}",
        timestamp_col, table
    );
    if let Some(extra) = extra_where {
        sql.push_str(" WHERE ");
        sql.push_str(extra);
    }
    sql.push_str(&format!(" ORDER BY datetime({}) ASC LIMIT 1", timestamp_col));
    sqlx::query(&sql)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
        .map(|row| row.get::<String, _>(timestamp_col))
}

async fn count_monologue_suppression(
    pool: &sqlx::SqlitePool,
    since: &str,
) -> (i64, i64) {
    let suppressed: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM inner_monologue_candidates
         WHERE suppression_reason IS NOT NULL AND suppression_reason != ''
           AND datetime(created_at) >= datetime(?)",
    )
    .bind(since)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    let total: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM inner_monologue_candidates
         WHERE datetime(created_at) >= datetime(?)",
    )
    .bind(since)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    (suppressed, total)
}

async fn fetch_feedback_breakdown(pool: &sqlx::SqlitePool, since: &str) -> Value {
    let mut counts: HashMap<String, i64> = HashMap::new();
    if let Ok(rows) = sqlx::query(
        "SELECT payload FROM system_logs
         WHERE json_extract(payload, '$.event') = 'user_feedback_detected'
           AND datetime(timestamp) >= datetime(?)",
    )
    .bind(since)
    .fetch_all(pool)
    .await
    {
        for row in rows {
            let raw: String = row.get("payload");
            let kind = serde_json::from_str::<Value>(&raw)
                .ok()
                .and_then(|v| v.get("kind").and_then(|k| k.as_str()).map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown".to_string());
            *counts.entry(kind).or_insert(0) += 1;
        }
    }
    let total: i64 = counts.values().sum();
    let positive = *counts.get("agree").unwrap_or(&0);
    let neutral = *counts.get("followup").unwrap_or(&0);
    let negative = counts
        .get("clarify")
        .unwrap_or(&0)
        + counts.get("pushback").unwrap_or(&0)
        + counts.get("disengage").unwrap_or(&0);
    json!({
        "total": total,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "by_kind": counts,
    })
}

async fn fetch_error_event_counts(pool: &sqlx::SqlitePool, since: &str) -> (i64, i64) {
    let total: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM error_events
         WHERE datetime(created_at) >= datetime(?)",
    )
    .bind(since)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    let open: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM error_events
         WHERE status = 'OPEN' AND datetime(created_at) >= datetime(?)",
    )
    .bind(since)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    (total, open)
}

async fn fetch_tool_dispatch_counts(pool: &sqlx::SqlitePool, since: &str) -> (i64, i64) {
    let mut success = 0;
    let mut failure = 0;
    if let Ok(rows) = sqlx::query(
        "SELECT status, COUNT(*) as count FROM tool_dispatches
         WHERE datetime(updated_at) >= datetime(?)
         GROUP BY status",
    )
    .bind(since)
    .fetch_all(pool)
    .await
    {
        for row in rows {
            let status: String = row.get("status");
            let count: i64 = row.get("count");
            match status.as_str() {
                "success" => success += count,
                "failed" => failure += count,
                _ => {}
            }
        }
    }
    (success, failure)
}

async fn fetch_tool_failure_kind_counts(pool: &sqlx::SqlitePool, since: &str) -> (i64, i64, i64) {
    let mut planning = 0;
    let mut execution = 0;
    let mut cancelled = 0;
    if let Ok(rows) = sqlx::query(
        "SELECT failure_kind, COUNT(*) as count FROM tool_dispatches
         WHERE status = 'failed' AND datetime(updated_at) >= datetime(?)
         GROUP BY failure_kind",
    )
    .bind(since)
    .fetch_all(pool)
    .await
    {
        for row in rows {
            let failure_kind: Option<String> = row.try_get("failure_kind").ok();
            let count: i64 = row.get("count");
            match failure_kind.as_deref() {
                Some("planning_error") => planning += count,
                Some("cancelled") => cancelled += count,
                _ => execution += count,
            }
        }
    }
    (planning, execution, cancelled)
}

async fn fetch_grouped_counts(
    pool: &sqlx::SqlitePool,
    table: &str,
    column: &str,
    timestamp_col: &str,
    since: &str,
) -> HashMap<String, i64> {
    let sql = format!(
        "SELECT {}, COUNT(*) as count FROM {} WHERE datetime({}) >= datetime(?) GROUP BY {}",
        column, table, timestamp_col, column
    );
    let mut map = HashMap::new();
    if let Ok(rows) = sqlx::query(&sql).bind(since).fetch_all(pool).await {
        for row in rows {
            let key: String = row.get(column);
            let count: i64 = row.get("count");
            map.insert(key, count);
        }
    }
    map
}

async fn fetch_log_level_counts(
    pool: &sqlx::SqlitePool,
    since: &str,
) -> (i64, i64, HashMap<String, Value>) {
    let warn_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs WHERE level = 'warn' AND datetime(timestamp) >= datetime(?)",
    )
    .bind(since)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    let error_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs WHERE level = 'error' AND datetime(timestamp) >= datetime(?)",
    )
    .bind(since)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);

    let mut map: HashMap<String, Value> = HashMap::new();
    if let Ok(rows) = sqlx::query(
        "SELECT category, level, COUNT(*) as count
         FROM system_logs
         WHERE level IN ('warn', 'error') AND datetime(timestamp) >= datetime(?)
         GROUP BY category, level",
    )
    .bind(since)
    .fetch_all(pool)
    .await
    {
        for row in rows {
            let category: String = row.get("category");
            let level: String = row.get("level");
            let count: i64 = row.get("count");
            let entry = map.entry(category).or_insert_with(|| json!({ "warn": 0, "error": 0 }));
            if let Some(obj) = entry.as_object_mut() {
                if level == "warn" {
                    obj.insert("warn".to_string(), json!(count));
                } else if level == "error" {
                    obj.insert("error".to_string(), json!(count));
                }
            }
        }
    }

    (warn_count, error_count, map)
}

async fn fetch_run_ids(pool: &sqlx::SqlitePool) -> (String, String) {
    let active_run: Option<String> = sqlx::query_scalar(
        "SELECT run_id FROM runs WHERE ended_at IS NULL ORDER BY datetime(started_at) DESC LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    let last_run: Option<String> = sqlx::query_scalar(
        "SELECT run_id FROM runs ORDER BY datetime(started_at) DESC LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    (active_run.unwrap_or_default(), last_run.unwrap_or_default())
}

fn diff_seconds_from_now(timestamp: Option<&str>) -> Option<i64> {
    let Some(ts) = timestamp else { return None; };
    let parsed = parse_timestamp(ts)?;
    Some((Utc::now() - parsed).num_seconds())
}

fn parse_timestamp(raw: &str) -> Option<DateTime<Utc>> {
    chrono::DateTime::parse_from_rfc3339(raw)
        .map(|dt| dt.with_timezone(&Utc))
        .or_else(|_| {
            chrono::NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S")
                .map(|naive| chrono::DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc))
        })
        .ok()
}
