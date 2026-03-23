use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use once_cell::sync::Lazy;
use chrono::{DateTime, Utc, Duration as ChronoDuration};
use crate::db::Db;
use crate::core::kernel::Kernel;
use crate::models::Reminder;
use crate::core::episodic;
use crate::core::memory::claims;
use crate::core::memory::compiler;
use crate::core::memory::types::SourceType;
use crate::core::rolling_summary;
use crate::core::self_memory::write_self_fact;
use crate::core::system_log;
use crate::core::system_controls;
use crate::core::system_health;
use crate::core::self_model_controller;
use crate::core::cognitive_wave;
use crate::core::qualia;
use crate::core::kernel::constants::INTERNAL_STATE_MAP_MIN_OBSERVATIONS;
use crate::core::kernel::KernelState;
use crate::core::subject_state;
use crate::core::sensitivity::detect_sensitivity;
use tauri::{AppHandle, Emitter};
use uuid::Uuid;
use sqlx::Row;
use serde_json::{json, Value};

/// Thought trigger interval: 5 minutes
const THOUGHT_INTERVAL_SECS: i64 = 300;
/// Thought trigger interval when user is active: 30 seconds
const ACTIVE_THOUGHT_INTERVAL_SECS: i64 = 30;
/// Proaction interval: 5 minutes
const PROACTION_INTERVAL_SECS: i64 = 300;
/// Pending-claims evaluation interval: 1 minute
const CLAIMS_EVAL_INTERVAL_SECS: i64 = 60;
/// Working set decay interval: 1 minute
const DECAY_INTERVAL_SECS: i64 = 60;
const MEMORY_VALIDATION_INTERVAL_SECS: i64 = 15 * 60;
/// Prediction evaluation interval: 1 minute
const PREDICTION_EVAL_INTERVAL_SECS: i64 = 60;
const IDENTITY_AUDIT_INTERVAL_SECS: i64 = 600;
/// Idle threshold for archiving rolling summaries
const SUMMARY_IDLE_MINUTES: f64 = 15.0;
/// Version delta to archive rolling summaries during active sessions
const SUMMARY_ARCHIVE_VERSION_DELTA: i64 = 2;
/// Visible turn count threshold for archiving summaries
const SUMMARY_ARCHIVE_TURN_THRESHOLD: i64 = 6;
/// Heartbeat interval: 1 minute
const HEARTBEAT_INTERVAL_SECS: i64 = 60;
/// System health snapshot interval: 10 seconds
const SYSTEM_HEALTH_INTERVAL_SECS: i64 = 10;
/// Unity health snapshot interval: 60 minutes
const UNITY_HEALTH_INTERVAL_SECS: i64 = 3600;
const UNITY_GREEN_WINDOW_HOURS: i64 = 6;
const PHI_BACKFILL_INTERVAL_SECS: i64 = 6 * 60 * 60;
const EPISODIC_IDENTITY_BACKFILL_INTERVAL_SECS: i64 = 6 * 60 * 60;
/// Dream interval: 10 minutes
const DREAM_INTERVAL_SECS: i64 = 600;
/// Self-memory bridge interval: 5 minutes
const SELF_MEMORY_BRIDGE_INTERVAL_SECS: i64 = 300;

const ROLLING_SUMMARY_RETRY_BASE_SECS: u64 = 5;
const ROLLING_SUMMARY_RETRY_MAX_SECS: u64 = 120;
const ROLLING_SUMMARY_RETRY_LIMIT: u32 = 5;

static ROLLING_SUMMARY_RETRY_ATTEMPTS: Lazy<Mutex<HashMap<String, u32>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
/// Warm-keep interval: 3 minutes
const WARM_KEEP_INTERVAL_SECS: i64 = 180;
/// Policy scan interval (outer loop enforcement)
const POLICY_SCAN_INTERVAL_SECS: i64 = 120;
const POLICY_SCAN_WINDOW_SECS: i64 = 900;
const POLICY_ESCALATION_THRESHOLD: i64 = 3;
/// Background gate window (seconds) after a user-visible run completes
const BACKGROUND_GATE_WINDOW_SECS: i64 = 15;
const AUTO_SURFACE_CHECK_INTERVAL_SECS: i64 = 5;
const STALE_RUN_CLEANUP_INTERVAL_SECS: i64 = 60;
const STALE_RUN_SUSPECT_SECS: i64 = 90;
const STALE_RUN_HARD_SECS: i64 = 240;
const STALE_RUN_TOOL_GRACE_SECS: i64 = 300;
const STALE_RUN_POST_PROCESSING_GRACE_SECS: i64 = 300;
const STALE_STREAMING_MESSAGE_SECS: i64 = 240;
const MONOLOGUE_ACTIVE_WINDOW_SECS: i64 = 60;
const ACTIVE_CONVERSATION_WINDOW_MINUTES: i64 = 24 * 60;
const CONTRIBUTOR_WINDOW_MINUTES: i64 = 60;
const TOOL_HEARTBEAT_WINDOW_MINUTES: i64 = 10;
const TELEMETRY_TOOL_HEARTBEAT_WINDOW_MINUTES: i64 = 60;
const SELF_MODEL_REFRESH_DEBOUNCE_SECS: i64 = 60;
const CADENCE_FLOOR_MS: u64 = 15_000;
const CADENCE_CEILING_MS: u64 = 300_000;
const SUBJECT_TICK_ACTIVE_MS: u64 = 250;
const SUBJECT_TICK_IDLE_MS: u64 = 2000;
const PREDICTION_MIN_SIGMA: f64 = 2.0;
const PREDICTION_MIN_ABS_DELTA: f64 = 0.1;
const PREDICTION_DEFAULT_VARIANCE: f64 = 0.05;
const INTERNAL_STATE_MAP_CALIBRATION_INTERVAL_SECS: i64 = 1800;
const INTERNAL_STATE_MAP_WINDOW_HOURS: i64 = 24;
const INTERNAL_STATE_MAP_DEGRADED_WINDOW_HOURS: i64 = 6;
const INTERNAL_STATE_MAP_BOOTSTRAP_MIN: i64 = 25;
const TELEMETRY_CALIBRATION_INTERVAL_SECS: i64 = 3600;
const WAVE_SNAPSHOT_INTERVAL_SECS: i64 = 60;

async fn background_gate_open(db: &Db) -> bool {
    let active: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM runs WHERE status = 'active'")
        .fetch_one(&db.pool)
        .await
        .unwrap_or(0);
    if active > 0 {
        return false;
    }
    let recent: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM runs WHERE ended_at IS NOT NULL AND (julianday('now') - julianday(ended_at)) * 86400 < ?",
    )
    .bind(BACKGROUND_GATE_WINDOW_SECS)
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    recent == 0
}

async fn recent_user_activity(db: &Db, conversation_id: &str, window_secs: i64) -> bool {
    let latest: Option<String> = sqlx::query_scalar(
        "SELECT created_at FROM messages
         WHERE conversation_id = ?
           AND role = 'user'
         ORDER BY datetime(created_at) DESC
         LIMIT 1",
    )
    .bind(conversation_id)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let Some(latest) = latest else { return false; };
    let Ok(parsed) = DateTime::parse_from_rfc3339(&latest)
        .or_else(|_| DateTime::parse_from_str(&latest, "%Y-%m-%d %H:%M:%S"))
    else {
        return false;
    };
    let now = Utc::now();
    let age = now.signed_duration_since(parsed.with_timezone(&Utc)).num_seconds();
    age >= 0 && age <= window_secs
}

async fn pending_auto_surface_exists(db: &Db, conversation_id: &str) -> bool {
    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM pending_user_prompts
         WHERE conversation_id = ?
           AND auto_surface = 1",
    )
    .bind(conversation_id)
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    count > 0
}

async fn list_active_conversations(db: &Db) -> Vec<String> {
    match db.list_conversation_ids(Some(ACTIVE_CONVERSATION_WINDOW_MINUTES)).await {
        Ok(ids) => ids,
        Err(err) => {
            let _ = system_log::log_event(
                &db.pool,
                None,
                "warn",
                "scheduler",
                None,
                None,
                json!({
                    "event": "list_conversations_failed",
                    "error": err.to_string(),
                }),
            )
            .await;
            Vec::new()
        }
    }
}

async fn cleanup_stale_runs_and_messages(db: &Db, app: &AppHandle) {
    let now = Utc::now();
    let cutoff = (now - chrono::Duration::seconds(STALE_RUN_SUSPECT_SECS)).to_rfc3339();
    let run_rows = sqlx::query(
        "SELECT run_id, trace_id, started_at, heartbeat_at, conversation_id
         FROM runs
         WHERE status = 'active'
           AND datetime(COALESCE(heartbeat_at, started_at)) <= datetime(?)",
    )
    .bind(&cutoff)
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();

    for row in run_rows {
        let run_id: String = row.get("run_id");
        let trace_id: String = row.get("trace_id");
        let started_at: String = row.get("started_at");
        let heartbeat_at: Option<String> = row.try_get("heartbeat_at").ok();
        let conversation_id: Option<String> = row.try_get("conversation_id").ok();
        let last_activity = heartbeat_at.as_deref().unwrap_or(&started_at);
        let last_dt = chrono::DateTime::parse_from_rfc3339(last_activity)
            .map(|dt| dt.with_timezone(&Utc))
            .or_else(|_| {
                chrono::NaiveDateTime::parse_from_str(last_activity, "%Y-%m-%d %H:%M:%S")
                    .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
            })
            .ok();
        let age_secs = last_dt
            .map(|dt| now.signed_duration_since(dt).num_seconds())
            .unwrap_or(STALE_RUN_SUSPECT_SECS);

        let pending_tool_dispatches: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM tool_dispatches
             WHERE run_id = ? AND status = 'pending'",
        )
        .bind(&run_id)
        .fetch_optional(&db.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0);

        let pending_post_processing: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM post_processing_jobs
             WHERE run_id = ?
               AND status IN ('queued','running')",
        )
        .bind(&run_id)
        .fetch_optional(&db.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0);

        if age_secs < STALE_RUN_SUSPECT_SECS {
            continue;
        }

        if age_secs < STALE_RUN_HARD_SECS {
            let _ = system_log::log_event(
                &db.pool,
                Some(app),
                "warn",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "run_stale_suspect",
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "age_secs": age_secs,
                    "suspect_threshold_secs": STALE_RUN_SUSPECT_SECS,
                    "hard_threshold_secs": STALE_RUN_HARD_SECS,
                    "pending_tool_dispatches": pending_tool_dispatches,
                    "pending_post_processing": pending_post_processing,
                    "heartbeat_at": heartbeat_at,
                }),
            )
            .await;
            continue;
        }

        if pending_tool_dispatches > 0 || pending_post_processing > 0 {
            let tool_grace = if pending_tool_dispatches > 0 {
                STALE_RUN_TOOL_GRACE_SECS
            } else {
                0
            };
            let post_processing_grace = if pending_post_processing > 0 {
                STALE_RUN_POST_PROCESSING_GRACE_SECS
            } else {
                0
            };
            let grace_limit = STALE_RUN_HARD_SECS + tool_grace.max(post_processing_grace);
            if age_secs < grace_limit {
                let _ = system_log::log_event(
                    &db.pool,
                    Some(app),
                    "warn",
                    "kernel",
                    Some(&run_id),
                    Some(&trace_id),
                    json!({
                        "event": "run_stale_cleanup_deferred",
                        "run_id": run_id,
                        "conversation_id": conversation_id,
                        "age_secs": age_secs,
                        "hard_threshold_secs": STALE_RUN_HARD_SECS,
                        "grace_threshold_secs": grace_limit,
                        "pending_tool_dispatches": pending_tool_dispatches,
                        "pending_post_processing": pending_post_processing,
                        "heartbeat_at": heartbeat_at,
                    }),
                )
                .await;
                continue;
            }
            let _ = system_log::log_event(
                &db.pool,
                Some(app),
                "warn",
                "kernel",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "run_stale_cleanup_forced",
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "age_secs": age_secs,
                    "hard_threshold_secs": STALE_RUN_HARD_SECS,
                    "grace_threshold_secs": grace_limit,
                    "pending_tool_dispatches": pending_tool_dispatches,
                    "pending_post_processing": pending_post_processing,
                    "heartbeat_at": heartbeat_at,
                }),
            )
            .await;
        }

        let run_rows = sqlx::query(
            "UPDATE runs SET status = 'error', ended_at = ? WHERE run_id = ? AND status = 'active'",
        )
        .bind(now)
        .bind(&run_id)
        .execute(&db.pool)
        .await
        .map(|res| res.rows_affected())
        .unwrap_or(0);

        let msg_rows = sqlx::query(
            "UPDATE messages
             SET status = 'error',
                 error = ?,
                 content = CASE
                     WHEN content IS NULL OR trim(content) = '' THEN ?
                     ELSE content
                 END
             WHERE run_id = ?
               AND status IN ('streaming', 'pending', 'active')",
        )
        .bind("stale_run_cleanup")
        .bind("Notice: response failed to finalize. Please retry.")
        .bind(&run_id)
        .execute(&db.pool)
        .await
        .map(|res| res.rows_affected())
        .unwrap_or(0);

        if msg_rows > 0 {
            let _ = app.emit("message_updated", ());
        }

        let _ = system_log::log_event(
            &db.pool,
            Some(app),
            "warn",
            "kernel",
            Some(&run_id),
            Some(&trace_id),
            json!({
                "event": "run_stale_cleanup",
                "run_id": run_id,
                "started_at": started_at,
                "heartbeat_at": heartbeat_at,
                "age_secs": age_secs,
                "cutoff": cutoff,
                "runs_updated": run_rows,
                "messages_updated": msg_rows,
            }),
        )
        .await;
    }

    let msg_cutoff = (now - chrono::Duration::seconds(STALE_STREAMING_MESSAGE_SECS)).to_rfc3339();
    let message_rows = sqlx::query(
        "SELECT message_id, run_id, created_at
         FROM messages
         WHERE status = 'streaming'
           AND datetime(created_at) <= datetime(?)",
    )
    .bind(&msg_cutoff)
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();
    for row in message_rows {
        let message_id: String = row.get("message_id");
        let run_id: Option<String> = row.try_get("run_id").ok();
        let created_at: String = row.get("created_at");
        let fallback = "Notice: response failed to finalize. Please retry.";
        let msg_rows = sqlx::query(
            "UPDATE messages
             SET status = 'error',
                 error = ?,
                 content = CASE
                     WHEN content IS NULL OR trim(content) = '' THEN ?
                     ELSE content
                 END
             WHERE message_id = ? AND status = 'streaming'",
        )
        .bind("stale_streaming_cleanup")
        .bind(fallback)
        .bind(&message_id)
        .execute(&db.pool)
        .await
        .map(|res| res.rows_affected())
        .unwrap_or(0);
        if let Some(run_id) = run_id.as_deref() {
            let run_rows = sqlx::query(
                "UPDATE runs SET status = 'error', ended_at = ? WHERE run_id = ? AND status = 'active'",
            )
            .bind(now)
            .bind(run_id)
            .execute(&db.pool)
            .await
            .map(|res| res.rows_affected())
            .unwrap_or(0);
            let _ = system_log::log_event(
                &db.pool,
                Some(app),
                "warn",
                "kernel",
                Some(run_id),
                None,
                json!({
                    "event": "message_stale_run_update",
                    "message_id": message_id,
                    "run_id": run_id,
                    "run_rows": run_rows,
                }),
            )
            .await;
        }
        if msg_rows > 0 {
            let _ = app.emit("message_updated", ());
        }
        let _ = system_log::log_event(
            &db.pool,
            Some(app),
            "warn",
            "kernel",
            run_id.as_deref(),
            None,
            json!({
                "event": "message_stale_cleanup",
                "message_id": message_id,
                "created_at": created_at,
                "cutoff": msg_cutoff,
                "messages_updated": msg_rows,
            }),
        )
        .await;
    }
}

async fn requeue_failed_rolling_summary_jobs(db: &Db, app: &AppHandle) {
    let updated = sqlx::query(
        "UPDATE post_processing_jobs
         SET status = 'queued', started_at = NULL, ended_at = NULL, error = NULL
         WHERE status = 'failed'
           AND job_type = 'rolling_summary_update'
           AND error = 'memory_policy_blocked'",
    )
    .execute(&db.pool)
    .await
    .map(|res| res.rows_affected())
    .unwrap_or(0);

    if updated > 0 {
        let _ = system_log::log_event(
            &db.pool,
            Some(app),
            "info",
            "scheduler",
            None,
            None,
            json!( {
                "event": "rolling_summary_requeued",
                "count": updated,
            }),
        )
        .await;
    }
}

fn record_rolling_summary_retry_attempt(conversation_id: &str) -> u32 {
    let mut attempts = ROLLING_SUMMARY_RETRY_ATTEMPTS
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let entry = attempts.entry(conversation_id.to_string()).or_insert(0);
    *entry = entry.saturating_add(1);
    *entry
}

fn clear_rolling_summary_retry_attempts(conversation_id: &str) {
    let mut attempts = ROLLING_SUMMARY_RETRY_ATTEMPTS
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    attempts.remove(conversation_id);
}

fn rolling_summary_retry_delay_secs(attempt: u32) -> u64 {
    let exp = 2u64.saturating_pow(attempt.saturating_sub(1).min(6));
    (ROLLING_SUMMARY_RETRY_BASE_SECS.saturating_mul(exp))
        .min(ROLLING_SUMMARY_RETRY_MAX_SECS)
}

async fn fetch_latency_ms(db: &Db, key: &str) -> Option<f64> {
    db.get_key(key)
        .await
        .ok()
        .flatten()
        .and_then(|value| value.parse::<f64>().ok())
}

async fn compute_desired_cadence_ms(db: &Db) -> u64 {
    let primary = fetch_latency_ms(db, "latency_primary_avg_ms").await.unwrap_or(0.0);
    let summary = fetch_latency_ms(db, "latency_summary_avg_ms").await.unwrap_or(0.0);
    let introspection = fetch_latency_ms(db, "latency_introspection_avg_ms").await.unwrap_or(0.0);
    let max_latency = primary.max(summary).max(introspection);
    let base = if max_latency <= 0.0 { CADENCE_FLOOR_MS } else { (max_latency * 3.0) as u64 };
    base.clamp(CADENCE_FLOOR_MS, CADENCE_CEILING_MS)
}

async fn latest_evidence_timestamp(db: &Db) -> Option<DateTime<Utc>> {
    let latest_ics: Option<String> = sqlx::query_scalar(
        "SELECT MAX(created_at) FROM ics_evidence_events",
    )
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let latest_self: Option<String> = sqlx::query_scalar(
        "SELECT MAX(created_at) FROM self_evidence_events",
    )
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();

    let mut latest: Option<DateTime<Utc>> = None;
    for raw in [latest_ics, latest_self].into_iter().flatten() {
        if let Some(parsed) = parse_timestamp(&raw) {
            if latest.map(|dt| parsed > dt).unwrap_or(true) {
                latest = Some(parsed);
            }
        }
    }
    latest
}

async fn run_policy_scan(db: &Db, app: &AppHandle) {
    let since = (Utc::now() - chrono::Duration::seconds(POLICY_SCAN_WINDOW_SECS)).to_rfc3339();
    let rows: Vec<String> = sqlx::query_scalar(
        "SELECT payload FROM system_logs WHERE timestamp > ?",
    )
    .bind(&since)
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();

    let mut loop_detected = 0i64;
    let mut loop_ack = 0i64;
    let mut violations_by_policy: HashMap<String, i64> = HashMap::new();
    let mut seen_violation_keys: HashSet<String> = HashSet::new();
    let mut direct_answer_total = 0i64;
    let mut direct_answer_true = 0i64;
    let mut validator_fallback_true = 0i64;

    for raw in rows.iter() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) else { continue; };
        let event = value.get("event").and_then(|v| v.as_str()).unwrap_or("");
        if event == "direct_answer_lead" {
            direct_answer_total += 1;
            if value.get("direct").and_then(|v| v.as_bool()).unwrap_or(false) {
                direct_answer_true += 1;
            }
            if value
                .get("fallback_applied")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                validator_fallback_true += 1;
            }
        }
        if event == "contract_violation" {
            if let Some(policy) = value.get("policy_id").and_then(|v| v.as_str()) {
                *violations_by_policy.entry(policy.to_string()).or_insert(0) += 1;
                if let Some(ctx) = value.get("context").and_then(|v| v.as_object()) {
                    if let Some(candidate_id) = ctx.get("candidate_id").and_then(|v| v.as_str()) {
                        seen_violation_keys.insert(format!("{}:{}", policy, candidate_id));
                    }
                    if let Some(turn) = ctx.get("turn").and_then(|v| v.as_i64()) {
                        seen_violation_keys.insert(format!("{}:turn:{}", policy, turn));
                    }
                }
            }
        }
    }

    for raw in rows.iter() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) else { continue; };
        let event = value.get("event").and_then(|v| v.as_str()).unwrap_or("");
        if event == "monologue_loop_detected" {
            loop_detected += 1;
        }
        if event == "meta_cog_event" {
            if let Some(reason) = value.get("reason").and_then(|v| v.as_str()) {
                if reason.contains("loop_detected") || reason.contains("loop_break") {
                    loop_ack += 1;
                }
            }
        }
        if event == "kernel_cycle" {
            if let Some(accepted) = value.get("accepted").and_then(|v| v.as_array()) {
                for item in accepted {
                    let kind = item.get("kind").and_then(|v| v.as_str()).unwrap_or("");
                    if kind != "EmitMessage" && kind != "FlagForHuman" {
                        continue;
                    }
                    let payload = item.get("payload").cloned().unwrap_or_else(|| json!({}));
                    let candidate_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
                    let speculative = payload.get("speculative").and_then(|v| v.as_bool()).unwrap_or(false);
                    let evidence_event_ids = payload.get("evidence_event_ids").and_then(|v| v.as_array()).map(|a| !a.is_empty()).unwrap_or(false);
                    let belief_ids = payload.get("belief_ids").and_then(|v| v.as_array()).map(|a| !a.is_empty()).unwrap_or(false);
                    let evidence_class = payload.get("evidence_class").and_then(|v| v.as_str()).unwrap_or("");
                    if !speculative && !evidence_event_ids && !belief_ids && evidence_class != "internal" {
                        let key = format!("C1:{}", candidate_id);
                        if !seen_violation_keys.contains(&key) {
                            let _ = system_log::log_contract_violation(
                                &db.pool,
                                Some(app),
                                None,
                                None,
                                "C1",
                                "ungrounded_assertion",
                                Some(json!({
                                    "candidate_id": candidate_id,
                                })),
                            )
                            .await;
                        }
                    }
                }
            }
            if let Some(rejected) = value.get("rejected").and_then(|v| v.as_array()) {
                for item in rejected {
                    let reason = item.get("reason").and_then(|v| v.as_str()).unwrap_or("");
                    if reason.trim().is_empty() {
                        let candidate_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
                        let key = format!("C4:{}", candidate_id);
                        if !seen_violation_keys.contains(&key) {
                            let _ = system_log::log_contract_violation(
                                &db.pool,
                                Some(app),
                                None,
                                None,
                                "C4",
                                "missing_suppression_reason",
                                Some(json!({
                                    "candidate_id": candidate_id,
                                })),
                            )
                            .await;
                        }
                    }
                }
            }
        }
    }

    if loop_detected > loop_ack && !seen_violation_keys.contains("C5:loop_window") {
        let _ = system_log::log_contract_violation(
            &db.pool,
            Some(app),
            None,
            None,
            "C5",
            "unacknowledged_loop",
            Some(json!({
                "loop_detected": loop_detected,
                "loop_ack": loop_ack,
            })),
        )
        .await;
    }

    if direct_answer_total > 0 {
        let direct_rate = (direct_answer_true as f64) / (direct_answer_total as f64);
        let fallback_rate = (validator_fallback_true as f64) / (direct_answer_total as f64);
        let _ = system_log::log_event(
            &db.pool,
            Some(app),
            "info",
            "kernel",
            None,
            None,
            json!({
                "event": "response_quality_metrics",
                "window_secs": POLICY_SCAN_WINDOW_SECS,
                "direct_answer_rate": direct_rate,
                "validator_fallback_rate": fallback_rate,
                "direct_answer_total": direct_answer_total,
                "direct_answer_true": direct_answer_true,
                "validator_fallback_true": validator_fallback_true,
            }),
        )
        .await;
    }

    let escalations: Vec<(String, i64)> = violations_by_policy
        .into_iter()
        .filter(|(_, count)| *count >= POLICY_ESCALATION_THRESHOLD)
        .collect();
    for (policy, count) in escalations {
        let already_escalated: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs WHERE timestamp > ? AND json_extract(payload, '$.event') = 'contract_violation_escalated' AND json_extract(payload, '$.policy_id') = ?"
        )
        .bind(&since)
        .bind(&policy)
        .fetch_one(&db.pool)
        .await
        .unwrap_or(0);
        if already_escalated == 0 {
            let _ = system_log::log_contract_violation_escalated(
                &db.pool,
                Some(app),
                None,
                None,
                &policy,
                count,
                Some(json!({ "window_seconds": POLICY_SCAN_WINDOW_SECS })),
            )
            .await;
        }
    }
}

pub struct Scheduler {
    db: Arc<Db>,
    app_handle: AppHandle,
    kernel: Arc<Kernel>,
}

impl Scheduler {
    pub fn new(db: Arc<Db>, app_handle: AppHandle, kernel: Arc<Kernel>) -> Self {
        Self { db, app_handle, kernel }
    }

    pub fn start(&self) {
        let db = self.db.clone();
        let app = self.app_handle.clone();
        let kernel = self.kernel.clone();

        // Use a dedicated OS thread with its own tokio runtime
        // This guarantees the scheduler runs independently of Tauri's runtime management
        std::thread::spawn(move || {
            println!("[Scheduler] Creating dedicated runtime...");
            
            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    eprintln!("[Scheduler] FATAL: Failed to create runtime: {}", e);
                    return;
                }
            };

            rt.block_on(async move {
                println!("[Scheduler] Background thread running!");
                let _ = system_log::log_event(
                    &db.pool,
                    Some(&app),
                    "info",
                    "scheduler",
                    None,
                    None,
                    json!({
                        "event": "scheduler_started",
                    }),
                )
                .await;

                let telemetry_key_count: i64 = sqlx::query_scalar(
                    "SELECT COUNT(*) FROM kv_store WHERE key LIKE 'telemetry.%'",
                )
                .fetch_optional(&db.pool)
                .await
                .ok()
                .flatten()
                .unwrap_or(0);
                if telemetry_key_count == 0 {
                    match crate::core::self_memory::telemetry::record_telemetry_snapshot_force(&db, None).await {
                        Ok(wrote) => {
                            if wrote {
                                let _ = system_log::log_event(
                                    &db.pool,
                                    Some(&app),
                                    "info",
                                    "scheduler",
                                    None,
                                    None,
                                    json!({
                                        "event": "telemetry_snapshot_seeded",
                                        "reason": "scheduler_start",
                                    }),
                                )
                                .await;
                            }
                        }
                        Err(err) => {
                            let _ = system_log::log_event(
                                &db.pool,
                                Some(&app),
                                "warn",
                                "scheduler",
                                None,
                                None,
                                json!({
                                    "event": "telemetry_snapshot_error",
                                    "error": err,
                                    "reason": "scheduler_start",
                                }),
                            )
                            .await;
                        }
                    }
                }

                requeue_failed_rolling_summary_jobs(&db, &app).await;

                let db_clone = db.clone();
                let app_clone = app.clone();
                tokio::spawn(async move {
                    let _ = system_log::log_event(
                        &db_clone.pool,
                        Some(&app_clone),
                        "info",
                        "scheduler",
                        None,
                        None,
                        json!({
                            "event": "prediction_outcome_backfill_start",
                        }),
                    )
                    .await;
                    if let Err(err) = evaluate_prediction_outcomes(&db_clone, &app_clone).await {
                        let _ = system_log::log_event(
                            &db_clone.pool,
                            Some(&app_clone),
                            "warn",
                            "scheduler",
                            None,
                            None,
                            json!({
                                "event": "prediction_outcome_backfill_error",
                                "error": err,
                            }),
                        )
                        .await;
                    } else {
                        let _ = system_log::log_event(
                            &db_clone.pool,
                            Some(&app_clone),
                            "info",
                            "scheduler",
                            None,
                            None,
                            json!({
                                "event": "prediction_outcome_backfill_complete",
                            }),
                        )
                        .await;
                    }
                });

                let subject_db = db.clone();
                let subject_kernel = kernel.clone();
                tokio::spawn(async move {
                    loop {
                        let control_map = system_controls::load_control_map(&subject_db).await;
                        let scheduler_mode = system_controls::mode_for("scheduler_tick", &control_map);
                        if system_controls::mode_is_off(&scheduler_mode) {
                            tokio::time::sleep(Duration::from_millis(SUBJECT_TICK_IDLE_MS)).await;
                            continue;
                        }
                        let conversation_ids = list_active_conversations(&subject_db).await;
                        let mut activity: Vec<(String, bool)> = Vec::with_capacity(conversation_ids.len());
                        let mut any_active = false;
                        for cid in conversation_ids.iter() {
                            let active =
                                recent_user_activity(&subject_db, cid, MONOLOGUE_ACTIVE_WINDOW_SECS).await;
                            if active {
                                any_active = true;
                            }
                            activity.push((cid.clone(), active));
                        }
                        let mut interval_ms = if any_active {
                            SUBJECT_TICK_ACTIVE_MS
                        } else {
                            SUBJECT_TICK_IDLE_MS
                        };
                        if system_controls::mode_is_degraded(&scheduler_mode) {
                            interval_ms = interval_ms.saturating_mul(2).min(10_000);
                        }
                        tokio::time::sleep(Duration::from_millis(interval_ms)).await;
                        for (cid, active) in activity.into_iter() {
                            let reason = if active { "active_tick" } else { "idle_tick" };
                            let _ = subject_kernel.run_subject_tick(&cid, reason).await;
                        }
                    }
                });

                let mut cadence_ms: u64 = CADENCE_FLOOR_MS;
                let mut pending_cadence_ms: u64 = cadence_ms;
                let mut pending_cadence_samples: u8 = 0;
                let mut last_thought = Utc::now();
                let mut last_proaction = Utc::now();
                let mut last_heartbeat = Utc::now();
                let mut last_dream = Utc::now();
                let mut last_self_bridge = Utc::now();
                let mut last_claims = Utc::now();
                let mut last_decay = Utc::now();
                let mut last_memory_validation =
                    Utc::now() - ChronoDuration::seconds(MEMORY_VALIDATION_INTERVAL_SECS);
                let mut last_predictions = Utc::now();
                let mut last_identity_audit = Utc::now() - ChronoDuration::seconds(IDENTITY_AUDIT_INTERVAL_SECS);
                let mut last_warm_keep = Utc::now();
                let mut last_policy_scan = Utc::now();
                let mut last_auto_surface = Utc::now();
                let mut last_stale_cleanup = Utc::now();
                let mut last_health_snapshot = Utc::now() - ChronoDuration::seconds(SYSTEM_HEALTH_INTERVAL_SECS);
                let mut last_unity_snapshot = Utc::now() - ChronoDuration::seconds(UNITY_HEALTH_INTERVAL_SECS);
                let mut last_internal_state_map = Utc::now()
                    - ChronoDuration::seconds(INTERNAL_STATE_MAP_CALIBRATION_INTERVAL_SECS);
                let mut last_telemetry_calibration = Utc::now()
                    - ChronoDuration::seconds(TELEMETRY_CALIBRATION_INTERVAL_SECS);

                loop {
                    tokio::time::sleep(Duration::from_millis(cadence_ms)).await;

                    let control_map = system_controls::load_control_map(&db).await;
                    let scheduler_mode = system_controls::mode_for("scheduler_tick", &control_map);
                    if system_controls::mode_is_off(&scheduler_mode) {
                        continue;
                    }
                    let scheduler_degraded = system_controls::mode_is_degraded(&scheduler_mode);
                    let memory_consolidation_mode =
                        system_controls::mode_for("memory_consolidation", &control_map);
                    let self_memory_mode = system_controls::mode_for("self_memory", &control_map);
                    let memory_retrieval_mode = system_controls::mode_for("memory_retrieval", &control_map);
                    let memory_write_mode = system_controls::mode_for("memory_write", &control_map);
                    let episodic_mode = system_controls::mode_for("episodic", &control_map);
                    let telemetry_mode = system_controls::mode_for("telemetry_sampling", &control_map);
                    let prediction_generation_mode =
                        system_controls::mode_for("prediction_generation", &control_map);
                    let qualia_loop_mode = system_controls::mode_for("qualia_loop", &control_map);
                    let qualia_auto_mode = system_controls::mode_for("qualia_auto", &control_map);
                    let tool_execution_mode = system_controls::mode_for("tool_execution", &control_map);

                    let now = Utc::now();
                    let now_ts = now.timestamp();
                    let background_allowed = background_gate_open(&db).await;
                    let conversation_ids = list_active_conversations(&db).await;
                    let mut activity_map: HashMap<String, bool> = HashMap::new();
                    let mut any_active = false;
                    for cid in conversation_ids.iter() {
                        let active = recent_user_activity(&db, cid, MONOLOGUE_ACTIVE_WINDOW_SECS).await;
                        activity_map.insert(cid.clone(), active);
                        if active {
                            any_active = true;
                        }
                    }
                    let thought_interval = if any_active {
                        ACTIVE_THOUGHT_INTERVAL_SECS
                    } else {
                        THOUGHT_INTERVAL_SECS
                    };
                    let mut thought_due = (now - last_thought).num_seconds() >= thought_interval;
                    let mut proaction_due = (now - last_proaction).num_seconds() >= PROACTION_INTERVAL_SECS;
                    let heartbeat_due = (now - last_heartbeat).num_seconds() >= HEARTBEAT_INTERVAL_SECS;
                    let mut dream_due = (now - last_dream).num_seconds() >= DREAM_INTERVAL_SECS;
                    let mut self_bridge_due = (now - last_self_bridge).num_seconds() >= SELF_MEMORY_BRIDGE_INTERVAL_SECS;
                    let mut claims_due = (now - last_claims).num_seconds() >= CLAIMS_EVAL_INTERVAL_SECS;
                    let mut decay_due = (now - last_decay).num_seconds() >= DECAY_INTERVAL_SECS;
                    let mut memory_validation_due =
                        (now - last_memory_validation).num_seconds() >= MEMORY_VALIDATION_INTERVAL_SECS;
                    let mut predictions_due = (now - last_predictions).num_seconds() >= PREDICTION_EVAL_INTERVAL_SECS;
                    let mut identity_audit_due = (now - last_identity_audit).num_seconds() >= IDENTITY_AUDIT_INTERVAL_SECS;
                    let mut warm_keep_due = (now - last_warm_keep).num_seconds() >= WARM_KEEP_INTERVAL_SECS;
                    let mut policy_scan_due = (now - last_policy_scan).num_seconds() >= POLICY_SCAN_INTERVAL_SECS;
                    let mut auto_surface_due =
                        (now - last_auto_surface).num_seconds() >= AUTO_SURFACE_CHECK_INTERVAL_SECS;
                    let stale_cleanup_due =
                        (now - last_stale_cleanup).num_seconds() >= STALE_RUN_CLEANUP_INTERVAL_SECS;
                    let health_interval = if system_controls::mode_is_degraded(&telemetry_mode) {
                        SYSTEM_HEALTH_INTERVAL_SECS * 3
                    } else {
                        SYSTEM_HEALTH_INTERVAL_SECS
                    };
                    let health_due = (now - last_health_snapshot).num_seconds() >= health_interval;
                    let mut unity_due = (now - last_unity_snapshot).num_seconds() >= UNITY_HEALTH_INTERVAL_SECS;
                    let mut internal_state_map_due =
                        (now - last_internal_state_map).num_seconds() >= INTERNAL_STATE_MAP_CALIBRATION_INTERVAL_SECS;
                    let mut telemetry_calibration_due =
                        (now - last_telemetry_calibration).num_seconds() >= TELEMETRY_CALIBRATION_INTERVAL_SECS;
                    let mut self_model_refresh_due = false;
                    let mut latest_evidence_at: Option<DateTime<Utc>> = None;
                    let mut internal_state_map_ready_count: Option<i64> = None;

                    if scheduler_degraded {
                        // Skip non-critical scheduler work in degraded mode.
                        thought_due = false;
                        proaction_due = false;
                        dream_due = false;
                        self_bridge_due = false;
                        claims_due = false;
                        decay_due = false;
                        memory_validation_due = false;
                        predictions_due = false;
                        identity_audit_due = false;
                        warm_keep_due = false;
                        policy_scan_due = false;
                        auto_surface_due = false;
                        unity_due = false;
                        internal_state_map_due = false;
                        telemetry_calibration_due = false;
                        self_model_refresh_due = false;
                    }
                    if background_allowed && !scheduler_degraded {
                        internal_state_map_ready_count = internal_state_map_bootstrap_ready(&db).await;
                        if internal_state_map_ready_count.is_some() {
                            internal_state_map_due = true;
                        }
                        if !conversation_ids.is_empty() {
                            latest_evidence_at = latest_evidence_timestamp(&db).await;
                            if let Some(latest_ts) = latest_evidence_at {
                                let last_refresh_raw = db.get_key("self_model_refresh_evidence_at")
                                    .await
                                    .ok()
                                    .flatten();
                                let last_refresh = last_refresh_raw
                                    .as_deref()
                                    .and_then(parse_timestamp);
                                let has_new = last_refresh.map(|ts| latest_ts > ts).unwrap_or(true);
                                let debounce_ok = last_refresh
                                    .map(|ts| now.signed_duration_since(ts).num_seconds() >= SELF_MODEL_REFRESH_DEBOUNCE_SECS)
                                    .unwrap_or(true);
                                if has_new && debounce_ok {
                                    self_model_refresh_due = true;
                                }
                            }
                        }
                    }
                    if thought_due {
                        last_thought = now;
                    }
                    if proaction_due {
                        last_proaction = now;
                    }
                    if heartbeat_due {
                        last_heartbeat = now;
                    }
                    if dream_due {
                        last_dream = now;
                    }
                    if self_bridge_due {
                        last_self_bridge = now;
                    }
                    if claims_due {
                        last_claims = now;
                    }
                    if decay_due {
                        last_decay = now;
                    }
                    if memory_validation_due {
                        last_memory_validation = now;
                    }
                    if predictions_due {
                        last_predictions = now;
                    }
                    if warm_keep_due {
                        last_warm_keep = now;
                    }
                    if policy_scan_due {
                        last_policy_scan = now;
                    }
                    if auto_surface_due {
                        last_auto_surface = now;
                    }
                    if stale_cleanup_due {
                        last_stale_cleanup = now;
                    }
                    if internal_state_map_due {
                        last_internal_state_map = now;
                    }
                    if telemetry_calibration_due {
                        last_telemetry_calibration = now;
                    }

                    // ============================================================
                    // INTERNAL MONOLOGUE TICKS
                    // ============================================================
                    if thought_due {
                        let kernel_clone = kernel.clone();
                        let ids = conversation_ids.clone();
                        tokio::spawn(async move {
                            for cid in ids {
                                let _ = kernel_clone.run_monologue_tick(&cid, false).await;
                            }
                        });
                    }
                    let _ = db.purge_expired_memory_pass_tokens().await;
                    if stale_cleanup_due {
                        let db_clone = db.clone();
                        let app_clone = app.clone();
                        tokio::spawn(async move {
                            cleanup_stale_runs_and_messages(&db_clone, &app_clone).await;
                        });
                    }
                    if background_allowed && !scheduler_degraded {
                        let db_clone = db.clone();
                        let app_clone = app.clone();
                        let kernel_clone = kernel.clone();
                        tokio::spawn(async move {
                            process_post_processing_jobs(db_clone, app_clone, kernel_clone).await;
                        });
                    }
                    if auto_surface_due {
                        let db_clone = db.clone();
                        let kernel_clone = kernel.clone();
                        let ids = conversation_ids.clone();
                        tokio::spawn(async move {
                            for cid in ids {
                                if pending_auto_surface_exists(&db_clone, &cid).await {
                                    let _ = kernel_clone
                                        .auto_surface_pending_prompts(&cid, None, None, "auto_surface_check")
                                        .await;
                                }
                            }
                        });
                    }

                    if proaction_due {
                        let kernel_clone = kernel.clone();
                        let ids = conversation_ids.clone();
                        tokio::spawn(async move {
                            for cid in ids {
                                let _ = kernel_clone.run_proaction_tick(&cid).await;
                            }
                        });
                    }

                    // ============================================================
                    // HEARTBEAT + DREAM (summary model)
                    // ============================================================
                    if background_allowed && heartbeat_due {
                        let kernel_clone = kernel.clone();
                        let ids = conversation_ids.clone();
                        tokio::spawn(async move {
                            for cid in ids {
                                let _ = kernel_clone.run_heartbeat_tick(&cid).await;
                            }
                        });
                    }
                    if background_allowed && heartbeat_due {
                        let db_clone = db.clone();
                        let app_clone = app.clone();
                        let ids = conversation_ids.clone();
                        let qualia_loop_mode = qualia_loop_mode.clone();
                        let qualia_auto_mode = qualia_auto_mode.clone();
                        tokio::spawn(async move {
                            if system_controls::mode_is_off(&qualia_loop_mode)
                                || system_controls::mode_is_off(&qualia_auto_mode)
                            {
                                return;
                            }
                            let since = (Utc::now() - ChronoDuration::minutes(CONTRIBUTOR_WINDOW_MINUTES)).to_rfc3339();
                            for cid in ids {
                                let latest_message_id: Option<String> = sqlx::query_scalar(
                                    "SELECT message_id FROM messages
                                     WHERE conversation_id = ? AND role = 'assistant' AND status = 'complete'
                                     ORDER BY datetime(created_at) DESC
                                     LIMIT 1",
                                )
                                .bind(&cid)
                                .fetch_optional(&db_clone.pool)
                                .await
                                .ok()
                                .flatten();
                                let Some(message_id) = latest_message_id else { continue; };

                                let label_count: i64 = sqlx::query_scalar(
                                    "SELECT COUNT(*) FROM qualia_labels
                                     WHERE event_id = ? AND datetime(created_at) >= datetime(?)",
                                )
                                .bind(&message_id)
                                .bind(&since)
                                .fetch_optional(&db_clone.pool)
                                .await
                                .ok()
                                .flatten()
                                .unwrap_or(0);

                                let reward_count: i64 = sqlx::query_scalar(
                                    "SELECT COUNT(*) FROM qualia_reward_events r
                                     JOIN qualia_labels l ON r.label_id = l.label_id
                                     WHERE l.event_id = ? AND datetime(r.created_at) >= datetime(?)",
                                )
                                .bind(&message_id)
                                .bind(&since)
                                .fetch_optional(&db_clone.pool)
                                .await
                                .ok()
                                .flatten()
                                .unwrap_or(0);

                                if label_count + reward_count == 0 {
                                    let _ = qualia::maybe_auto_label_for_recent_message(&db_clone, Some(&app_clone), &cid, None).await;
                                }
                            }
                        });
                    }
                    if background_allowed && heartbeat_due {
                        let db_clone = db.clone();
                        let kernel_clone = kernel.clone();
                        let tool_execution_mode = tool_execution_mode.clone();
                        tokio::spawn(async move {
                            if system_controls::mode_is_off(&tool_execution_mode) {
                                return;
                            }
                            let since =
                                (Utc::now() - ChronoDuration::minutes(TOOL_HEARTBEAT_WINDOW_MINUTES))
                                    .to_rfc3339();
                            let recent_dispatches: i64 = sqlx::query_scalar(
                                "SELECT COUNT(*) FROM tool_dispatches
                                 WHERE datetime(updated_at) >= datetime(?)",
                            )
                            .bind(&since)
                            .fetch_optional(&db_clone.pool)
                            .await
                            .ok()
                            .flatten()
                            .unwrap_or(0);
                            if recent_dispatches == 0 {
                                let _ = kernel_clone.run_tool_heartbeat_tick().await;
                            }
                        });
                    }
                    if background_allowed && heartbeat_due {
                        let db_clone = db.clone();
                        let kernel_clone = kernel.clone();
                        let ids = conversation_ids.clone();
                        let memory_write_mode = memory_write_mode.clone();
                        let memory_retrieval_mode = memory_retrieval_mode.clone();
                        tokio::spawn(async move {
                            if system_controls::mode_is_off(&memory_write_mode)
                                || system_controls::mode_is_read_only(&memory_write_mode)
                                || system_controls::mode_is_off(&memory_retrieval_mode)
                            {
                                return;
                            }
                            let since = (Utc::now() - ChronoDuration::minutes(CONTRIBUTOR_WINDOW_MINUTES)).to_rfc3339();
                            for cid in ids {
                                let recent_message_count: i64 = sqlx::query_scalar(
                                    "SELECT COUNT(*) FROM messages
                                     WHERE conversation_id = ?
                                       AND role IN ('user','assistant')
                                       AND status = 'complete'
                                       AND (metadata IS NULL OR json_extract(metadata, '$.source') != 'monologue')
                                       AND datetime(created_at) >= datetime(?)",
                                )
                                .bind(&cid)
                                .bind(&since)
                                .fetch_optional(&db_clone.pool)
                                .await
                                .ok()
                                .flatten()
                                .unwrap_or(0);
                                if recent_message_count == 0 {
                                    continue;
                                }

                                let memory_recent_writes: i64 = sqlx::query_scalar(
                                    "SELECT COUNT(*) FROM memory_write_ledger
                                     WHERE conversation_id = ? AND datetime(created_at) >= datetime(?)",
                                )
                                .bind(&cid)
                                .bind(&since)
                                .fetch_optional(&db_clone.pool)
                                .await
                                .ok()
                                .flatten()
                                .unwrap_or(0);
                                if memory_recent_writes == 0 {
                                    let _ = kernel_clone.run_memory_pass_tick(&cid).await;
                                }
                            }
                        });
                    }
                    if background_allowed && heartbeat_due {
                        let db_clone = db.clone();
                        let app_clone = app.clone();
                        tokio::spawn(async move {
                            cognitive_wave::maybe_emit_wave_state_snapshot(
                                &db_clone,
                                Some(&app_clone),
                                WAVE_SNAPSHOT_INTERVAL_SECS,
                            )
                            .await;
                        });
                    }
                    if background_allowed && dream_due {
                        let kernel_clone = kernel.clone();
                        let ids = conversation_ids.clone();
                        tokio::spawn(async move {
                            for cid in ids {
                                let _ = kernel_clone.run_dream_cycle(&cid).await;
                            }
                        });
                    }
                    if background_allowed && policy_scan_due {
                        let db_clone = db.clone();
                        let app_clone = app.clone();
                        tokio::spawn(async move {
                            run_policy_scan(&db_clone, &app_clone).await;
                        });
                    }
                    if background_allowed && warm_keep_due {
                        let kernel_clone = kernel.clone();
                        tokio::spawn(async move {
                            kernel_clone.warm_keep("scheduler").await;
                        });
                    }
                    if background_allowed && internal_state_map_due {
                        if let Some(count) = internal_state_map_ready_count {
                            let _ = system_log::log_event(
                                &db.pool,
                                Some(&app),
                                "info",
                                "scheduler",
                                None,
                                None,
                                json!({
                                    "event": "internal_state_map_bootstrap_due",
                                    "observed_count": count,
                                }),
                            )
                            .await;
                        }
                        let db_clone = db.clone();
                        let app_clone = app.clone();
                        tokio::spawn(async move {
                            run_internal_state_map_calibration(&db_clone, &app_clone).await;
                        });
                    }
                    if background_allowed && telemetry_calibration_due {
                        let db_clone = db.clone();
                        let app_clone = app.clone();
                        let kernel_clone = kernel.clone();
                        let tool_execution_mode = tool_execution_mode.clone();
                        tokio::spawn(async move {
                            if !system_controls::mode_is_off(&tool_execution_mode) {
                                let since = (Utc::now()
                                    - ChronoDuration::minutes(TELEMETRY_TOOL_HEARTBEAT_WINDOW_MINUTES))
                                .to_rfc3339();
                                let recent_dispatches: i64 = sqlx::query_scalar(
                                    "SELECT COUNT(*) FROM tool_dispatches
                                     WHERE datetime(updated_at) >= datetime(?)",
                                )
                                .bind(&since)
                                .fetch_optional(&db_clone.pool)
                                .await
                                .ok()
                                .flatten()
                                .unwrap_or(0);
                                if recent_dispatches == 0 {
                                    let _ = kernel_clone.run_tool_heartbeat_tick().await;
                                    let _ = system_log::log_event(
                                        &db_clone.pool,
                                        Some(&app_clone),
                                        "info",
                                        "scheduler",
                                        None,
                                        None,
                                        json!({
                                            "event": "telemetry_calibration_tool_heartbeat",
                                            "window_minutes": TELEMETRY_TOOL_HEARTBEAT_WINDOW_MINUTES,
                                            "recent_dispatches": recent_dispatches,
                                        }),
                                    )
                                    .await;
                                }
                            }
                            let _ = crate::core::telemetry_calibration::run_telemetry_calibration(
                                &db_clone,
                                Some(&app_clone),
                            )
                            .await;
                        });
                    }
                    if background_allowed && self_model_refresh_due {
                        let db_clone = db.clone();
                        let app_clone = app.clone();
                        let ids = conversation_ids.clone();
                        let latest_ts = latest_evidence_at.map(|ts| ts.to_rfc3339());
                        tokio::spawn(async move {
                            let mut refreshed = 0usize;
                            for cid in ids {
                                let raw = db_clone.get_kernel_state(&cid).await.ok().flatten();
                                let kernel_state: KernelState = raw
                                    .and_then(|s| serde_json::from_str(&s).ok())
                                    .unwrap_or_else(|| KernelState::default_for(&cid));
                                match self_model_controller::update_unified_self_model(&db_clone, &kernel_state).await {
                                    Ok(_) => refreshed += 1,
                                    Err(err) => {
                                        let _ = system_log::log_event(
                                            &db_clone.pool,
                                            Some(&app_clone),
                                            "warn",
                                            "scheduler",
                                            None,
                                            None,
                                            json!({
                                                "event": "self_model_refresh_failed",
                                                "conversation_id": cid,
                                                "error": err,
                                            }),
                                        )
                                        .await;
                                    }
                                }
                            }
                            if let Some(ts) = latest_ts.as_deref() {
                                let _ = db_clone.set_key("self_model_refresh_evidence_at", ts).await;
                            }
                            let _ = system_log::log_event(
                                &db_clone.pool,
                                Some(&app_clone),
                                "info",
                                "scheduler",
                                None,
                                None,
                                json!({
                                    "event": "self_model_refresh_debounced",
                                    "conversation_count": refreshed,
                                    "latest_evidence_at": latest_ts,
                                }),
                            )
                            .await;
                        });
                    }
                    
                    // ============================================================
                    // REMINDERS (Using Unix timestamp comparison)
                    // ============================================================
                    let all_pending: Vec<Reminder> = match sqlx::query_as::<_, Reminder>(
                        r#"SELECT id, content, due_at, type, status, created_at FROM reminders WHERE status = 'PENDING'"#
                    )
                    .fetch_all(&db.pool)
                    .await 
                    {
                        Ok(rows) => {
                            if !rows.is_empty() {
                                println!("[Scheduler] Found {} pending reminders", rows.len());
                            }
                            rows
                        },
                        Err(e) => {
                            eprintln!("[Scheduler] Error polling reminders: {}", e);
                            Vec::new()
                        }
                    };

                    for reminder in all_pending {
                        if reminder.due_at <= now_ts {
                            println!("[Scheduler] TRIGGERING reminder: '{}' (due_at: {} <= now: {})", 
                                reminder.content, reminder.due_at, now_ts);

                            let _ = episodic::emit_episodic_event(
                                &db.pool,
                                "reminder_triggered",
                                json!({ "status": "triggered", "summary_snippet": reminder.content }),
                                None,
                                None,
                                Some("default"),
                                None,
                                "system",
                                Some(&reminder.id),
                                None,
                                None,
                            )
                            .await;

                            let _ = system_log::log_event(
                                &db.pool,
                                Some(&app),
                                "info",
                                "scheduler",
                                None,
                                None,
                                json!({
                                    "event": "reminder_triggered",
                                    "reminder_id": reminder.id.clone(),
                                    "due_at": reminder.due_at,
                                }),
                            )
                            .await;
                            
                            if let Err(e) = app.emit("reminder_triggered", &reminder) {
                                eprintln!("[Scheduler] Failed to emit event: {}", e);
                            }

                            let _ = sqlx::query("UPDATE reminders SET status = 'COMPLETED' WHERE id = ?")
                                .bind(&reminder.id)
                                .execute(&db.pool)
                                .await;
                        }
                    }

                    if background_allowed {
                    // ============================================================
                    // ROLLING SUMMARY ARCHIVE (idle detection)
                    // ============================================================
                    let active_runs: Option<i64> = sqlx::query_scalar(
                        "SELECT COUNT(*) FROM runs WHERE status = 'active'"
                    )
                    .fetch_optional(&db.pool)
                    .await
                    .ok()
                    .flatten();

                    let recent_summary_blocks: i64 = sqlx::query_scalar(
                        "SELECT COUNT(*) FROM system_logs
                         WHERE json_extract(payload, '$.event') = 'memory_write_blocked'
                           AND json_extract(payload, '$.category') = 'summary'
                           AND datetime(timestamp) >= datetime('now','-15 minutes')",
                    )
                    .fetch_one(&db.pool)
                    .await
                    .unwrap_or(0);

                    for conversation_id in conversation_ids.iter() {
                        if active_runs.unwrap_or(0) == 0 {
                            let idle_minutes: Option<f64> = sqlx::query_scalar(
                                "SELECT (julianday('now') - julianday(created_at)) * 24 * 60
                                 FROM messages
                                 WHERE conversation_id = ?
                                   AND role IN ('user', 'assistant')
                                   AND (metadata IS NULL OR json_extract(metadata, '$.source') != 'monologue')
                                 ORDER BY datetime(created_at) DESC
                                 LIMIT 1"
                            )
                            .bind(conversation_id)
                            .fetch_optional(&db.pool)
                            .await
                            .ok()
                            .flatten();

                            if let Some(idle_minutes) = idle_minutes {
                                if idle_minutes >= SUMMARY_IDLE_MINUTES {
                                    let now = Utc::now();
                                    let summary_updated_at: Option<String> = sqlx::query_scalar(
                                        "SELECT updated_at FROM conversation_summaries WHERE conversation_id = ? LIMIT 1",
                                    )
                                    .bind(conversation_id)
                                    .fetch_optional(&db.pool)
                                    .await
                                    .ok()
                                    .flatten();
                                    let summary_stale = summary_updated_at
                                        .as_deref()
                                        .and_then(parse_timestamp)
                                        .map(|ts| now.signed_duration_since(ts).num_minutes() as f64 >= SUMMARY_IDLE_MINUTES)
                                        .unwrap_or(false);

                                    if summary_stale && recent_summary_blocks == 0 {
                                        let _ = rolling_summary::archive_rolling_summary(
                                            db.clone(),
                                            conversation_id,
                                            "scheduler",
                                            "summary_archive",
                                        )
                                        .await;
                                        let _ = system_log::log_event(
                                            &db.pool,
                                            Some(&app),
                                            "info",
                                            "scheduler",
                                            None,
                                            None,
                                            json!({
                                                "event": "summary_archived_idle",
                                                "conversation_id": conversation_id,
                                                "idle_minutes": idle_minutes,
                                                "summary_stale": summary_stale,
                                                "recent_summary_blocks": recent_summary_blocks,
                                            }),
                                        )
                                        .await;
                                    }
                                }
                            }
                        }

                        // ============================================================
                        // Rolling Summary: Version-based Archiving (active sessions)
                        // ============================================================
                        let summary_version: Option<i64> = sqlx::query_scalar(
                            "SELECT version FROM conversation_summaries WHERE conversation_id = ? LIMIT 1",
                        )
                        .bind(conversation_id)
                        .fetch_optional(&db.pool)
                        .await
                        .ok()
                        .flatten();
                        if let Some(current_version) = summary_version {
                            let last_version: Option<i64> = sqlx::query_scalar(
                                "SELECT source_summary_version FROM conversation_summary_chunks
                                 WHERE conversation_id = ?
                                 ORDER BY datetime(created_at) DESC
                                 LIMIT 1",
                            )
                            .bind(conversation_id)
                            .fetch_optional(&db.pool)
                            .await
                            .ok()
                            .flatten();
                            let delta = current_version - last_version.unwrap_or(0);
                            if delta >= SUMMARY_ARCHIVE_VERSION_DELTA && recent_summary_blocks == 0 {
                                let _ = rolling_summary::archive_rolling_summary(
                                    db.clone(),
                                    conversation_id,
                                    "scheduler",
                                    "summary_archive_turn",
                                )
                                .await;
                                let _ = system_log::log_event(
                                    &db.pool,
                                    Some(&app),
                                    "info",
                                    "scheduler",
                                    None,
                                    None,
                                    json!({
                                        "event": "summary_archived_turn",
                                        "conversation_id": conversation_id,
                                        "current_version": current_version,
                                        "last_version": last_version,
                                        "delta": delta,
                                    }),
                                )
                                .await;
                            }
                        }

                        // ============================================================
                        // Rolling Summary: Turn-count Archiving (active sessions)
                        // ============================================================
                        let recent_turns: Option<i64> = sqlx::query_scalar(
                            "SELECT COUNT(*) FROM messages
                             WHERE conversation_id = ?
                               AND role IN ('user', 'assistant')
                               AND (metadata IS NULL OR json_extract(metadata, '$.source') != 'monologue')
                               AND datetime(created_at) > COALESCE((
                                   SELECT end_ts FROM conversation_summary_chunks
                                   WHERE conversation_id = ?
                                   ORDER BY datetime(created_at) DESC
                                   LIMIT 1
                               ), '1970-01-01')",
                        )
                        .bind(conversation_id)
                        .bind(conversation_id)
                        .fetch_optional(&db.pool)
                        .await
                        .ok()
                        .flatten();
                        if let Some(recent_turns) = recent_turns {
                            if recent_turns >= SUMMARY_ARCHIVE_TURN_THRESHOLD && recent_summary_blocks == 0 {
                                let _ = rolling_summary::archive_rolling_summary(
                                    db.clone(),
                                    conversation_id,
                                    "scheduler",
                                    "summary_archive_turn_count",
                                )
                                .await;
                                let _ = system_log::log_event(
                                    &db.pool,
                                    Some(&app),
                                    "info",
                                    "scheduler",
                                    None,
                                    None,
                                    json!({
                                        "event": "summary_archived_turn_count",
                                        "conversation_id": conversation_id,
                                        "recent_turns": recent_turns,
                                        "threshold": SUMMARY_ARCHIVE_TURN_THRESHOLD,
                                    }),
                                )
                                .await;
                            }
                        }
                    }

                    // ============================================================
                    // PHI Backfill (every 6 hours)
                    // ============================================================
                    let last_backfill: Option<String> = sqlx::query_scalar(
                        "SELECT value FROM kv_store WHERE key = 'phi_backfill_last_at' LIMIT 1",
                    )
                    .fetch_optional(&db.pool)
                    .await
                    .ok()
                    .flatten();
                    let backfill_due = last_backfill
                        .as_deref()
                        .and_then(parse_timestamp)
                        .map(|ts| {
                            Utc::now()
                                .signed_duration_since(ts)
                                .num_seconds()
                                >= PHI_BACKFILL_INTERVAL_SECS
                        })
                        .unwrap_or(true);
                    if backfill_due {
                        match run_phi_backfill(&db.pool, 200).await {
                            Ok(processed) => {
                                let _ = sqlx::query(
                                    "INSERT INTO kv_store (key, value, updated_at)
                                     VALUES ('phi_backfill_last_at', ?, CURRENT_TIMESTAMP)
                                     ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
                                )
                                .bind(Utc::now().to_rfc3339())
                                .execute(&db.pool)
                                .await;
                                let _ = system_log::log_event(
                                    &db.pool,
                                    Some(&app),
                                    "info",
                                    "memory",
                                    None,
                                    None,
                                    json!({
                                        "event": "phi_backfill_complete",
                                        "processed": processed,
                                    }),
                                )
                                .await;
                            }
                            Err(e) => {
                                let _ = system_log::log_event(
                                    &db.pool,
                                    Some(&app),
                                    "warn",
                                    "memory",
                                    None,
                                    None,
                                    json!({
                                        "event": "phi_backfill_error",
                                        "error": e,
                                    }),
                                )
                                .await;
                            }
                        }
                    }

                    // ============================================================
                    // Episodic Identity Index Backfill (every 6 hours)
                    // ============================================================
                    let last_epi_backfill: Option<String> = sqlx::query_scalar(
                        "SELECT value FROM kv_store WHERE key = 'episodic_identity_backfill_last_at' LIMIT 1",
                    )
                    .fetch_optional(&db.pool)
                    .await
                    .ok()
                    .flatten();
                    let epi_backfill_due = last_epi_backfill
                        .as_deref()
                        .and_then(parse_timestamp)
                        .map(|ts| {
                            Utc::now()
                                .signed_duration_since(ts)
                                .num_seconds()
                                >= EPISODIC_IDENTITY_BACKFILL_INTERVAL_SECS
                        })
                        .unwrap_or(true);
                    if epi_backfill_due {
                        match episodic::backfill_identity_index(&db.pool, 250).await {
                            Ok(processed) => {
                                let _ = sqlx::query(
                                    "INSERT INTO kv_store (key, value, updated_at)
                                     VALUES ('episodic_identity_backfill_last_at', ?, CURRENT_TIMESTAMP)
                                     ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
                                )
                                .bind(Utc::now().to_rfc3339())
                                .execute(&db.pool)
                                .await;
                                let _ = system_log::log_event(
                                    &db.pool,
                                    Some(&app),
                                    "info",
                                    "memory",
                                    None,
                                    None,
                                    json!({
                                        "event": "episodic_identity_backfill_complete",
                                        "processed": processed,
                                    }),
                                )
                                .await;
                            }
                            Err(e) => {
                                let _ = system_log::log_event(
                                    &db.pool,
                                    Some(&app),
                                    "warn",
                                    "memory",
                                    None,
                                    None,
                                    json!({
                                        "event": "episodic_identity_backfill_error",
                                        "error": e,
                                    }),
                                )
                                .await;
                            }
                        }
                    }

                    // ============================================================
                    // META-COGNITIVE: Autonomous Thought Triggers
                    // ============================================================
                    
                    // Run thought triggers on a dynamic cadence (active user: 30s, otherwise ~5 minutes)
                    if thought_due {
                        println!("[Meta-Cognitive] Running autonomous thought triggers...");
                        
                        // Trigger 1: Unresolved high-priority conflicts
                        if let Ok(row) = sqlx::query(
                            "SELECT COUNT(*) as c FROM ics_conflict_sets 
                             WHERE status = 'open' AND priority = 'high'"
                        )
                        .fetch_one(&db.pool)
                        .await {
                            let conflict_count: i32 = row.try_get("c").unwrap_or(0);
                            if conflict_count > 0 {
                                let thought = format!(
                                    "I have {} unresolved high-priority conflict(s) I'd like to clarify with you when you have a moment.",
                                    conflict_count
                                );
                                let _ = sqlx::query(
                                    "INSERT OR REPLACE INTO kv_store (key, value, updated_at)
                                     VALUES ('pending_thought', ?, datetime('now'))"
                                )
                                .bind(&thought)
                                .execute(&db.pool)
                                .await;
                                println!("[Meta-Cognitive] Queued conflict-resolution thought");
                                let _ = system_log::log_event(
                                    &db.pool,
                                    Some(&app),
                                    "info",
                                    "scheduler",
                                    None,
                                    None,
                                    json!({
                                        "event": "thought_triggered",
                                        "kind": "conflict_resolution",
                                        "count": conflict_count,
                                    }),
                                )
                                .await;
                            }
                        }
                        
                        // Trigger 2: Stale high-confidence beliefs (not reinforced in 7+ days)
                        if let Ok(row) = sqlx::query(
                            "SELECT COUNT(*) as c FROM ics_beliefs 
                             WHERE status = 'active'
                             AND last_evidence_at < datetime('now', '-7 days')
                             AND confidence > 0.8
                             AND valid_to IS NULL"
                        )
                        .fetch_one(&db.pool)
                        .await {
                            let stale_count: i32 = row.try_get("c").unwrap_or(0);
                            if stale_count > 3 {
                                // Mark that a belief review is needed (surfaced via epistemic)
                                let _ = sqlx::query(
                                    "INSERT OR REPLACE INTO kv_store (key, value, updated_at)
                                     VALUES ('belief_review_needed', 'true', datetime('now'))"
                                )
                                .execute(&db.pool)
                                .await;
                                println!("[Meta-Cognitive] Flagged {} stale high-confidence beliefs for review", stale_count);
                                let _ = system_log::log_event(
                                    &db.pool,
                                    Some(&app),
                                    "info",
                                    "scheduler",
                                    None,
                                    None,
                                    json!({
                                        "event": "thought_triggered",
                                        "kind": "belief_review",
                                        "count": stale_count,
                                    }),
                                )
                                .await;
                            }
                        }
                    }

                    // ============================================================
                    // SELF-REFLECTION (every 5 minutes)
                    // ============================================================
                    if thought_due {
                        let reflection_started = Instant::now();
                        let _ = system_log::log_event(
                            &db.pool,
                            Some(&app),
                            "info",
                            "scheduler",
                            None,
                            None,
                            json!({
                                "event": "self_reflection_attempted",
                            }),
                        )
                        .await;
                        if let Err(e) = crate::core::self_reflection::run_reflection(&db, &app).await {
                            eprintln!("[Self-Reflection] Error: {}", e);
                            let _ = system_log::log_event(
                                &db.pool,
                                Some(&app),
                                "error",
                                "scheduler",
                                None,
                                None,
                                json!({
                                    "event": "self_reflection_error",
                                    "error": e.to_string(),
                                }),
                            )
                            .await;
                            let _ = system_log::log_event(
                                &db.pool,
                                Some(&app),
                                "warn",
                                "scheduler",
                                None,
                                None,
                                json!({
                                    "event": "self_reflection_completed",
                                    "success": false,
                                    "duration_ms": reflection_started.elapsed().as_millis() as i64,
                                    "error": e.to_string(),
                                }),
                            )
                            .await;
                        } else {
                            let _ = system_log::log_event(
                                &db.pool,
                                Some(&app),
                                "info",
                                "scheduler",
                                None,
                                None,
                                json!({
                                    "event": "self_reflection_completed",
                                    "success": true,
                                    "duration_ms": reflection_started.elapsed().as_millis() as i64,
                                }),
                            )
                            .await;
                        }
                    }

                    // ============================================================
                    // SELF-MEMORY BRIDGE + PARITY CHECK (every 5 minutes)
                    // ============================================================
                    if self_bridge_due {
                        if system_controls::mode_is_off(&self_memory_mode)
                            || system_controls::mode_is_read_only(&self_memory_mode)
                        {
                            // Skip bridge when self-memory is disabled or read-only.
                        } else {
                            let bridge_limit = if system_controls::mode_is_degraded(&self_memory_mode) {
                                10
                            } else {
                                50
                            };
                            match crate::core::self_memory::bridge::bridge_pending_events(&db.pool, bridge_limit).await {
                                Ok(report) => {
                                    let _ = system_log::log_event(
                                        &db.pool,
                                        Some(&app),
                                        "info",
                                        "memory",
                                        None,
                                        None,
                                        json!({
                                            "event": "self_memory_bridge",
                                            "bridge_type": "self_evidence",
                                            "processed": report.processed,
                                            "skipped": report.skipped,
                                            "errors": report.errors,
                                        }),
                                    )
                                    .await;
                                }
                                Err(e) => {
                                    let _ = system_log::log_event(
                                        &db.pool,
                                        Some(&app),
                                        "warn",
                                        "memory",
                                        None,
                                        None,
                                        json!({
                                            "event": "self_memory_bridge_error",
                                            "error": e,
                                        }),
                                    )
                                    .await;
                                }
                            }
                            match crate::core::self_memory::bridge::bridge_identity_evidence_events(&db.pool, bridge_limit).await {
                                Ok(report) => {
                                    let _ = system_log::log_event(
                                        &db.pool,
                                        Some(&app),
                                        "info",
                                        "memory",
                                        None,
                                        None,
                                        json!({
                                            "event": "self_memory_bridge",
                                            "bridge_type": "identity_evidence",
                                            "processed": report.processed,
                                            "skipped": report.skipped,
                                            "errors": report.errors,
                                        }),
                                    )
                                    .await;
                                }
                                Err(e) => {
                                    let _ = system_log::log_event(
                                        &db.pool,
                                        Some(&app),
                                        "warn",
                                        "memory",
                                        None,
                                        None,
                                        json!({
                                            "event": "self_memory_bridge_error",
                                            "bridge_type": "identity_evidence",
                                            "error": e,
                                        }),
                                    )
                                    .await;
                                }
                            }
                            crate::core::self_memory::bridge::log_parity(&db.pool).await;
                        }
                    }

                    // ============================================================
                    // EPISODIC RETENTION + COMPACTION (every 5 minutes)
                    // ============================================================
                    if thought_due {
                        let episodic_blocked = system_controls::mode_is_off(&episodic_mode)
                            || system_controls::mode_is_degraded(&episodic_mode)
                            || system_controls::mode_is_off(&memory_write_mode)
                            || system_controls::mode_is_read_only(&memory_write_mode);
                        if episodic_blocked {
                            // Skip episodic compaction when disabled or memory writes are blocked.
                        } else {
                        let episodic_compaction_enabled: Option<i32> = sqlx::query_scalar(
                            "SELECT episodic_compaction_enabled FROM settings WHERE id = 1"
                        )
                        .fetch_optional(&db.pool)
                        .await
                        .ok()
                        .flatten();

                        if episodic_compaction_enabled.unwrap_or(0) != 0 {
                        const EPISODIC_RETENTION_DAYS: i64 = 45;
                        const EPISODIC_SUMMARY_LIMIT: i64 = 200;
                        const EPISODIC_SUMMARY_MAX: usize = 512;
                        const EPISODIC_SUMMARY_MIN_PARTS: usize = 3;

                        let cutoff = chrono::Utc::now() - chrono::Duration::days(EPISODIC_RETENTION_DAYS);
                        let cutoff_str = cutoff.to_rfc3339();

                        let conversation_ids: Vec<Option<String>> = sqlx::query_scalar(
                            "SELECT DISTINCT conversation_id FROM episodic_events WHERE julianday(timestamp) < julianday(?)"
                        )
                        .bind(&cutoff_str)
                        .fetch_all(&db.pool)
                        .await
                        .unwrap_or_default();

                        let has_null = conversation_ids.iter().any(|c| c.is_none());

                        for conversation_id in conversation_ids.into_iter().flatten() {
                            let rows = sqlx::query(
                                "SELECT event_type, payload_json, source_type
                                 FROM episodic_events
                                 WHERE conversation_id = ? AND julianday(timestamp) < julianday(?)
                                 ORDER BY timestamp ASC
                                 LIMIT ?"
                            )
                            .bind(&conversation_id)
                            .bind(&cutoff_str)
                            .bind(EPISODIC_SUMMARY_LIMIT)
                            .fetch_all(&db.pool)
                            .await
                            .unwrap_or_default();

                            if rows.is_empty() {
                                continue;
                            }

                            let mut summary = String::new();
                            let mut has_non_tool = false;
                            let mut parts_count = 0;
                            for row in rows {
                                let event_type: String = row.get("event_type");
                                let source_type: String = row.get("source_type");
                                let payload_raw: String = row.get("payload_json");
                                let payload: serde_json::Value = serde_json::from_str(&payload_raw).unwrap_or_default();
                                let snippet = payload
                                    .get("summary_snippet")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .trim();
                                let part = if snippet.is_empty() {
                                    event_type.clone()
                                } else {
                                    format!("{}: {}", event_type, snippet)
                                };

                                if !summary.is_empty() {
                                    summary.push_str(" | ");
                                }
                                summary.push_str(&part);
                                parts_count += 1;

                                if source_type != "tool" || !event_type.starts_with("tool_") {
                                    has_non_tool = true;
                                }

                                if summary.chars().count() >= EPISODIC_SUMMARY_MAX {
                                    summary = summary.chars().take(EPISODIC_SUMMARY_MAX).collect();
                                    break;
                                }
                            }

                            let summary_event_id = crate::core::episodic::emit_episodic_event(
                                &db.pool,
                                "episodic_summary",
                                serde_json::json!({ "status": "compacted", "summary_snippet": summary }),
                                None,
                                None,
                                Some(&conversation_id),
                                None,
                                "system",
                                Some("compaction"),
                                None,
                                None,
                            )
                            .await
                            .unwrap_or_default();

                            if has_non_tool && parts_count >= EPISODIC_SUMMARY_MIN_PARTS {
                                let event_id = if summary_event_id.is_empty() {
                                    None
                                } else {
                                    Some(summary_event_id.as_str())
                                };
                                let _ = claims::create_summary_claim(
                                    &db.pool,
                                    &summary,
                                    Some(&conversation_id),
                                    Some("compaction"),
                                    event_id,
                                )
                                .await;
                            }

                            let _ = sqlx::query(
                                "DELETE FROM episodic_events WHERE conversation_id = ? AND julianday(timestamp) < julianday(?)"
                            )
                            .bind(&conversation_id)
                            .bind(&cutoff_str)
                            .execute(&db.pool)
                            .await;

                            let _ = system_log::log_event(
                                &db.pool,
                                Some(&app),
                                "info",
                                "scheduler",
                                None,
                                None,
                                json!({
                                    "event": "episodic_compaction",
                                    "conversation_id": conversation_id,
                                    "parts": parts_count,
                                    "summary_len": summary.len(),
                                }),
                            )
                            .await;
                        }

                        if has_null {
                            let rows = sqlx::query(
                                "SELECT event_type, payload_json, source_type
                                 FROM episodic_events
                                 WHERE conversation_id IS NULL AND julianday(timestamp) < julianday(?)
                                 ORDER BY timestamp ASC
                                 LIMIT ?"
                            )
                            .bind(&cutoff_str)
                            .bind(EPISODIC_SUMMARY_LIMIT)
                            .fetch_all(&db.pool)
                            .await
                            .unwrap_or_default();

                            if !rows.is_empty() {
                                let mut summary = String::new();
                                let mut has_non_tool = false;
                                let mut parts_count = 0;
                                for row in rows {
                                    let event_type: String = row.get("event_type");
                                    let source_type: String = row.get("source_type");
                                    let payload_raw: String = row.get("payload_json");
                                    let payload: serde_json::Value = serde_json::from_str(&payload_raw).unwrap_or_default();
                                    let snippet = payload
                                        .get("summary_snippet")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .trim();
                                    let part = if snippet.is_empty() {
                                        event_type.clone()
                                    } else {
                                        format!("{}: {}", event_type, snippet)
                                    };

                                    if !summary.is_empty() {
                                        summary.push_str(" | ");
                                    }
                                    summary.push_str(&part);
                                    parts_count += 1;

                                    if source_type != "tool" || !event_type.starts_with("tool_") {
                                        has_non_tool = true;
                                    }

                                    if summary.chars().count() >= EPISODIC_SUMMARY_MAX {
                                        summary = summary.chars().take(EPISODIC_SUMMARY_MAX).collect();
                                        break;
                                    }
                                }

                                let summary_event_id = crate::core::episodic::emit_episodic_event(
                                    &db.pool,
                                    "episodic_summary",
                                    serde_json::json!({ "status": "compacted", "summary_snippet": summary }),
                                    None,
                                    None,
                                    None,
                                    None,
                                    "system",
                                    Some("compaction"),
                                    None,
                                    None,
                                )
                                .await
                                .unwrap_or_default();

                                if has_non_tool && parts_count >= EPISODIC_SUMMARY_MIN_PARTS {
                                    let event_id = if summary_event_id.is_empty() {
                                        None
                                    } else {
                                        Some(summary_event_id.as_str())
                                    };
                                    let _ = claims::create_summary_claim(
                                        &db.pool,
                                        &summary,
                                        None,
                                        Some("compaction"),
                                        event_id,
                                    )
                                    .await;
                                }

                                let _ = sqlx::query(
                                    "DELETE FROM episodic_events WHERE conversation_id IS NULL AND julianday(timestamp) < julianday(?)"
                                )
                                .bind(&cutoff_str)
                                .execute(&db.pool)
                                .await;

                                let _ = system_log::log_event(
                                    &db.pool,
                                    Some(&app),
                                    "info",
                                    "scheduler",
                                    None,
                                    None,
                                    json!({
                                        "event": "episodic_compaction",
                                        "conversation_id": null,
                                        "parts": parts_count,
                                        "summary_len": summary.len(),
                                    }),
                                )
                                .await;
                            }
                        }

                        let _ = sqlx::query(
                            "DELETE FROM messages WHERE julianday(created_at) < julianday(?)"
                        )
                        .bind(&cutoff_str)
                        .execute(&db.pool)
                        .await;

                        let _ = sqlx::query(
                            "DELETE FROM artifacts WHERE run_id IN (SELECT run_id FROM runs WHERE julianday(started_at) < julianday(?))"
                        )
                        .bind(&cutoff_str)
                        .execute(&db.pool)
                        .await;

                        let _ = sqlx::query(
                            "DELETE FROM runs WHERE julianday(started_at) < julianday(?)"
                        )
                        .bind(&cutoff_str)
                        .execute(&db.pool)
                        .await;

                        let _ = sqlx::query(
                            "DELETE FROM conversation_summary_chunks
                             WHERE julianday(COALESCE(end_ts, created_at)) < julianday(?)"
                        )
                        .bind(&cutoff_str)
                        .execute(&db.pool)
                        .await;

                        if let Ok(promoted) = claims::auto_promote_summary_claims(&db.pool, None).await {
                            if promoted > 0 {
                                println!("[Scheduler] Promoted {} summary claim(s).", promoted);
                            }
                        }
                    }
                    }

                    // ============================================================
                    // MEMORY MAINTENANCE: Consolidation (every 5 minutes)
                    // ============================================================
                    if thought_due {
                        let consolidation_blocked = system_controls::mode_is_off(&memory_consolidation_mode)
                            || system_controls::mode_is_degraded(&memory_consolidation_mode)
                            || system_controls::mode_is_off(&memory_write_mode)
                            || system_controls::mode_is_read_only(&memory_write_mode);
                        if !consolidation_blocked {
                            println!("[Memory] Running consolidation...");
                            use crate::core::memory::api::MemoryApi;
                            let api = MemoryApi::new(db.pool.clone(), None, "scheduler".to_string()).await;
                            match api.consolidate().await {
                                Ok(result) => {
                                    println!("[Memory] Consolidation complete: aliases={}, sketches={}, stale={}, archived={}",
                                        result.aliases_promoted, result.sketches_updated,
                                        result.stale_deactivated, result.conflicts_archived);
                                    let _ = system_log::log_event(
                                        &db.pool,
                                        Some(&app),
                                        "info",
                                        "memory",
                                        None,
                                        None,
                                        json!({
                                            "event": "consolidation_complete",
                                            "aliases_promoted": result.aliases_promoted,
                                            "sketches_updated": result.sketches_updated,
                                            "stale_deactivated": result.stale_deactivated,
                                            "conflicts_archived": result.conflicts_archived,
                                        }),
                                    )
                                    .await;
                                }
                                Err(e) => {
                                    eprintln!("[Memory] Consolidation error: {}", e);
                                    let _ = system_log::log_event(
                                        &db.pool,
                                        Some(&app),
                                        "error",
                                        "memory",
                                        None,
                                        None,
                                        json!({
                                            "event": "consolidation_error",
                                            "error": e.to_string(),
                                        }),
                                    )
                                    .await;
                                }
                            }

                            if let Err(e) = crate::core::memory::rel_type_catalog::curate_rel_types(&db.pool).await {
                                eprintln!("[Memory] Rel type curator error: {}", e);
                                let _ = system_log::log_event(
                                    &db.pool,
                                    Some(&app),
                                    "error",
                                    "memory",
                                    None,
                                    None,
                                    json!({
                                        "event": "rel_type_curator_error",
                                        "error": e.to_string(),
                                    }),
                                )
                                .await;
                            }
                        }
                        // ============================================================
                        // SELF-MEMORY MAINTENANCE: Decay (every 5 minutes)
                        // ============================================================
                        if !(system_controls::mode_is_off(&self_memory_mode)
                            || system_controls::mode_is_read_only(&self_memory_mode))
                        {
                            let decay_limit = if system_controls::mode_is_degraded(&self_memory_mode) {
                                15
                            } else {
                                30
                            };
                            let decay_strength = if system_controls::mode_is_degraded(&self_memory_mode) {
                                0.1
                            } else {
                                0.2
                            };
                            if let Err(e) = crate::core::self_memory::decay::decay_self_memory(&db.pool, decay_limit, decay_strength).await {
                                eprintln!("[Self-Memory] Decay error: {}", e);
                                let _ = system_log::log_event(
                                    &db.pool,
                                    Some(&app),
                                    "error",
                                    "memory",
                                    None,
                                    None,
                                    json!({
                                        "event": "self_memory_decay_error",
                                        "error": e.to_string(),
                                    }),
                                )
                                .await;
                            }
                        }

                        // ============================================================
                        // MEMORY MAINTENANCE: Salience recompute (every 5 minutes)
                        // ============================================================
                        if !consolidation_blocked {
                            let updated = crate::core::memory::attention::salience::recompute_salience_for_all(&db.pool)
                                .await
                                .unwrap_or(0);
                            println!("[Memory] Salience recompute applied ({} beliefs).", updated);
                            let _ = system_log::log_event(
                                &db.pool,
                                Some(&app),
                                "info",
                                "memory",
                                None,
                                None,
                                json!({
                                    "event": "salience_recomputed",
                                    "beliefs": updated,
                                }),
                            )
                            .await;
                        }

                        // ============================================================
                        // MEMORY VALIDATION: Confidence decay + drift detection
                        // ============================================================
                        if memory_validation_due {
                            let validation_blocked = system_controls::mode_is_off(&memory_write_mode)
                                || system_controls::mode_is_read_only(&memory_write_mode);
                            if !validation_blocked {
                                let mut config = crate::core::memory::validation::MemoryValidationConfig::default();
                                if system_controls::mode_is_degraded(&memory_write_mode) {
                                    config.max_beliefs = 150;
                                    config.decay_per_day = 0.995;
                                    config.drift_threshold = 0.08;
                                }
                                match crate::core::memory::validation::validate_memory_beliefs(
                                    &db.pool,
                                    &config,
                                    Some(&app),
                                )
                                .await
                                {
                                    Ok((ics, self_mem)) => {
                                        println!(
                                            "[Memory] Validation complete: ics_scanned={}, self_scanned={}, drift_events={}",
                                            ics.scanned,
                                            self_mem.scanned,
                                            ics.drift_events + self_mem.drift_events
                                        );
                                    }
                                    Err(e) => {
                                        eprintln!("[Memory] Validation error: {}", e);
                                        let _ = system_log::log_event(
                                            &db.pool,
                                            Some(&app),
                                            "error",
                                            "memory",
                                            None,
                                            None,
                                            json!({
                                                "event": "memory_validation_error",
                                                "error": e.to_string(),
                                            }),
                                        )
                                        .await;
                                    }
                                }
                            }
                        }

                        if health_due && !system_controls::mode_is_off(&telemetry_mode) {
                            last_health_snapshot = now;
                            let db_clone = db.clone();
                            let app_clone = app.clone();
                            tokio::spawn(async move {
                                let aggregator = system_health::HealthAggregator::new(db_clone.clone());
                                if let Err(err) = aggregator
                                    .capture_snapshot(None, None, Some(&app_clone))
                                    .await
                                {
                                    let _ = system_log::log_event(
                                        &db_clone.pool,
                                        Some(&app_clone),
                                        "warn",
                                        "scheduler",
                                        None,
                                        None,
                                        json!({
                                            "event": "system_health_snapshot",
                                            "status": "error",
                                            "error": err,
                                        }),
                                    )
                                    .await;
                                }
                            });
                        }

                        if unity_due && !system_controls::mode_is_off(&telemetry_mode) {
                            last_unity_snapshot = now;
                            let db_clone = db.clone();
                            let app_clone = app.clone();
                            tokio::spawn(async move {
                                match system_health::capture_unity_snapshot(db_clone.clone(), Some(&app_clone)).await {
                                    Ok(snapshot) => {
                                        if snapshot.pass {
                                            let ready = system_health::unity_green_window_ready(&db_clone.pool).await;
                                            if ready {
                                                let should_log = match system_health::last_unity_green_at(&db_clone.pool).await {
                                                    Some(ts) => (Utc::now() - ts).num_hours() >= UNITY_GREEN_WINDOW_HOURS,
                                                    None => true,
                                                };
                                                if should_log {
                                                    let _ = system_log::log_event(
                                                        &db_clone.pool,
                                                        Some(&app_clone),
                                                        "info",
                                                        "system",
                                                        None,
                                                        None,
                                                        json!({
                                                            "event": "unity_green_window",
                                                            "window_hours": UNITY_GREEN_WINDOW_HOURS,
                                                        }),
                                                    )
                                                    .await;
                                                }
                                            }
                                        } else {
                                            let _ = system_health::capture_unity_diagnostic(db_clone.clone(), Some(&app_clone)).await;
                                        }
                                    }
                                    Err(err) => {
                                        let _ = system_log::log_event(
                                            &db_clone.pool,
                                            Some(&app_clone),
                                            "warn",
                                            "scheduler",
                                            None,
                                            None,
                                            json!({
                                                "event": "unity_health_snapshot",
                                                "status": "error",
                                                "error": err,
                                            }),
                                        )
                                        .await;
                                    }
                                }
                            });
                        }

                        if !system_controls::mode_is_off(&telemetry_mode) {
                            match crate::core::self_memory::telemetry::record_telemetry_snapshot(&db, None).await {
                                Ok(wrote) => {
                                    let _ = wrote;
                                }
                                Err(e) => {
                                    let _ = system_log::log_event(
                                        &db.pool,
                                        Some(&app),
                                        "warn",
                                        "scheduler",
                                        None,
                                        None,
                                        json!({
                                            "event": "telemetry_snapshot_error",
                                            "error": e,
                                        }),
                                    )
                                    .await;
                                }
                            }
                        }
                    }

                    // ============================================================
                    // PREDICTION OUTCOMES (every minute)
                    // ============================================================
                    if predictions_due {
                        if !system_controls::mode_is_off(&prediction_generation_mode)
                            && !system_controls::mode_is_degraded(&prediction_generation_mode)
                        {
                            let last_prediction_at: Option<String> = sqlx::query_scalar(
                                "SELECT created_at FROM self_predictions
                                 ORDER BY datetime(created_at) DESC LIMIT 1",
                            )
                            .fetch_optional(&db.pool)
                            .await
                            .ok()
                            .flatten();
                            let age_secs = last_prediction_at
                                .as_deref()
                                .and_then(|raw| chrono::DateTime::parse_from_rfc3339(raw).ok())
                                .map(|dt| now.signed_duration_since(dt.with_timezone(&Utc)).num_seconds())
                                .unwrap_or(PREDICTION_EVAL_INTERVAL_SECS + 1);
                            if age_secs >= PREDICTION_EVAL_INTERVAL_SECS {
                                let _ = system_log::log_event(
                                    &db.pool,
                                    Some(&app),
                                    "info",
                                    "scheduler",
                                    None,
                                    None,
                                    json!({
                                        "event": "prediction_generation_skipped",
                                        "reason": "cadence_miss",
                                        "age_seconds": age_secs,
                                    }),
                                )
                                .await;
                            }
                        }
                        let db_clone = db.clone();
                        let app_clone = app.clone();
                        tokio::spawn(async move {
                            if let Err(e) = evaluate_prediction_outcomes(&db_clone, &app_clone).await {
                                let _ = system_log::log_event(
                                    &db_clone.pool,
                                    Some(&app_clone),
                                    "warn",
                                    "scheduler",
                                    None,
                                    None,
                                    json!({
                                        "event": "prediction_outcome_error",
                                        "error": e,
                                    }),
                                )
                                .await;
                            }
                        });
                    }

                    if identity_audit_due {
                        last_identity_audit = now;
                        let db_clone = db.clone();
                        let app_clone = app.clone();
                        tokio::spawn(async move {
                            if let Err(e) = audit_identity_snapshot(&db_clone, &app_clone).await {
                                let _ = system_log::log_event(
                                    &db_clone.pool,
                                    Some(&app_clone),
                                    "warn",
                                    "scheduler",
                                    None,
                                    None,
                                    json!({
                                        "event": "identity_audit_error",
                                        "error": e,
                                    }),
                                )
                                .await;
                            }
                        });
                    }

                    // ============================================================
                    // MEMORY CLAIMS: Evaluate pending claims (every minute)
                    // ============================================================
                    if claims_due {
                        if compiler::memory_claims_enabled(&db.pool).await {
                            if let Ok(processed) = claims::evaluate_pending_claims(&db.pool, None, 25).await {
                                if processed > 0 {
                                    println!("[Memory] Evaluated {} pending claim(s).", processed);
                                    let _ = system_log::log_event(
                                        &db.pool,
                                        Some(&app),
                                        "info",
                                        "memory",
                                        None,
                                        None,
                                        json!({
                                            "event": "claims_evaluated",
                                            "processed": processed,
                                        }),
                                    )
                                    .await;
                                }
                            }
                        }
                    }
                    
                    // ============================================================
                    // MEMORY MAINTENANCE: Working Set Decay (every minute)
                    // ============================================================
                    if decay_due {
                        let _ = cognitive_wave::decay_tick(
                            &db.pool,
                            Some(&app),
                            DECAY_INTERVAL_SECS as f32,
                            None,
                            None,
                        )
                        .await;
                        // Decay activation gradually
                        }
                        let _ = sqlx::query(
                            "UPDATE ics_working_set SET activation = activation * 0.95 WHERE activation > 0.01"
                        ).execute(&db.pool).await;
                        
                        // Prune items below threshold
                        let _ = sqlx::query(
                            "DELETE FROM ics_working_set WHERE activation < 0.01"
                        ).execute(&db.pool).await;

                        let base_half_life = db
                            .get_settings()
                            .await
                            .ok()
                            .and_then(|s| s.memory_half_life_hours)
                            .unwrap_or(168.0)
                            .max(1.0);
                        if let Ok((promoted_working, promoted_semantic, promoted_world)) =
                            apply_layer_promotion(&db).await
                        {
                        if let Ok(deactivated) = apply_layer_decay(&db, base_half_life as f64).await {
                                let _ = system_log::log_event(
                                    &db.pool,
                                    Some(&app),
                                    "info",
                                    "memory",
                                    None,
                                    None,
                                    json!({
                                        "event": "memory_layer_maintenance",
                                        "promoted_working_to_episodic": promoted_working,
                                        "promoted_to_semantic": promoted_semantic,
                                        "promoted_to_world": promoted_world,
                                        "deactivated": deactivated,
                                    }),
                                )
                                .await;
                            }
                        }
                    }

                    }

                    if background_allowed {
                        let now_key = Utc::now().to_rfc3339();
                        let last_at = db
                            .get_key("memory_write_ledger_last_at")
                            .await
                            .ok()
                            .flatten()
                            .unwrap_or_else(|| "1970-01-01T00:00:00Z".to_string());
                        let write_count: i64 = sqlx::query_scalar(
                            "SELECT COUNT(*) FROM memory_write_ledger WHERE datetime(created_at) > datetime(?)",
                        )
                        .bind(&last_at)
                        .fetch_optional(&db.pool)
                        .await
                        .ok()
                        .flatten()
                        .unwrap_or(0);
                        let _ = db.set_key("memory_write_ledger_last_at", &now_key).await;

                        let belief_count: i64 = sqlx::query_scalar(
                            "SELECT COUNT(*) FROM ics_beliefs WHERE status = 'active'",
                        )
                        .fetch_optional(&db.pool)
                        .await
                        .ok()
                        .flatten()
                        .unwrap_or(0);
                        let prev_belief_count = db
                            .get_key("memory_belief_count_last")
                            .await
                            .ok()
                            .flatten()
                            .and_then(|v| v.parse::<i64>().ok())
                            .unwrap_or(belief_count);
                        let graph_delta = belief_count.saturating_sub(prev_belief_count);
                        let _ = db
                            .set_key("memory_belief_count_last", &belief_count.to_string())
                            .await;

                        let promoted_count: i64 = sqlx::query_scalar(
                            "SELECT COUNT(*) FROM memory_claims WHERE status = 'promoted'",
                        )
                        .fetch_optional(&db.pool)
                        .await
                        .ok()
                        .flatten()
                        .unwrap_or(0);
                        let prev_promoted = db
                            .get_key("memory_claims_promoted_last")
                            .await
                            .ok()
                            .flatten()
                            .and_then(|v| v.parse::<i64>().ok())
                            .unwrap_or(promoted_count);
                        let promoted_delta = promoted_count.saturating_sub(prev_promoted);
                        let _ = db
                            .set_key("memory_claims_promoted_last", &promoted_count.to_string())
                            .await;

                        let conflict_resolved: i64 = sqlx::query_scalar(
                            "SELECT COUNT(*) FROM ics_conflict_sets WHERE status = 'resolved'",
                        )
                        .fetch_optional(&db.pool)
                        .await
                        .ok()
                        .flatten()
                        .unwrap_or(0);
                        let prev_conflict_resolved = db
                            .get_key("memory_conflicts_resolved_last")
                            .await
                            .ok()
                            .flatten()
                            .and_then(|v| v.parse::<i64>().ok())
                            .unwrap_or(conflict_resolved);
                        let conflict_delta = conflict_resolved.saturating_sub(prev_conflict_resolved);
                        let _ = db
                            .set_key("memory_conflicts_resolved_last", &conflict_resolved.to_string())
                            .await;

                        let evidence_count: i64 = sqlx::query_scalar(
                            "SELECT COUNT(*) FROM ics_evidence_events",
                        )
                        .fetch_optional(&db.pool)
                        .await
                        .ok()
                        .flatten()
                        .unwrap_or(0);
                        let prev_evidence_count = db
                            .get_key("memory_evidence_count_last")
                            .await
                            .ok()
                            .flatten()
                            .and_then(|v| v.parse::<i64>().ok())
                            .unwrap_or(evidence_count);
                        let evidence_delta = evidence_count.saturating_sub(prev_evidence_count);
                        let _ = db
                            .set_key("memory_evidence_count_last", &evidence_count.to_string())
                            .await;

                        let _ = system_log::log_event(
                            &db.pool,
                            Some(&app),
                            "info",
                            "memory",
                            None,
                            None,
                            json!({
                                "event": "memory_cycle_metrics",
                                "write_count": write_count,
                                "graph_delta": graph_delta,
                                "promoted_delta": promoted_delta,
                                "conflict_delta": conflict_delta,
                                "evidence_delta": evidence_delta,
                            }),
                        )
                        .await;

                        if graph_delta > 20 && evidence_delta == 0 {
                            let _ = system_log::log_event(
                                &db.pool,
                                Some(&app),
                                "warn",
                                "memory",
                                None,
                                None,
                                json!({
                                    "event": "memory_write_noise",
                                    "graph_delta": graph_delta,
                                    "evidence_delta": evidence_delta,
                                }),
                            )
                            .await;
                        }
                    }

                    let mut desired_cadence = compute_desired_cadence_ms(&db).await;
                    if scheduler_degraded {
                        desired_cadence = desired_cadence.saturating_mul(2).min(CADENCE_CEILING_MS);
                    }
                    if desired_cadence != cadence_ms {
                        if desired_cadence == pending_cadence_ms {
                            pending_cadence_samples = pending_cadence_samples.saturating_add(1);
                        } else {
                            pending_cadence_ms = desired_cadence;
                            pending_cadence_samples = 1;
                        }
                        if pending_cadence_samples >= 3 {
                            cadence_ms = pending_cadence_ms;
                            pending_cadence_samples = 0;
                            let _ = system_log::log_event(
                                &db.pool,
                                Some(&app),
                                "info",
                                "scheduler",
                                None,
                                None,
                                json!({
                                    "event": "scheduler_cadence_updated",
                                    "cadence_ms": cadence_ms,
                                }),
                            )
                            .await;
                        }
                    } else {
                        pending_cadence_samples = 0;
                    }
                }
                }
            );
        });
    }
}

fn parse_timestamp(raw: &str) -> Option<DateTime<Utc>> {
    if let Ok(ts) = DateTime::parse_from_rfc3339(raw) {
        return Some(ts.with_timezone(&Utc));
    }
    if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S") {
        return Some(DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc));
    }
    None
}

async fn run_phi_backfill(pool: &sqlx::SqlitePool, batch: i64) -> Result<i64, String> {
    let rows = sqlx::query(
        "SELECT b.id, b.kind, f.value_literal
         FROM ics_beliefs b
         LEFT JOIN ics_fact_beliefs f ON f.belief_id = b.id
         LEFT JOIN memory_sensitivity ms ON ms.belief_id = b.id
         WHERE ms.belief_id IS NULL
         LIMIT ?",
    )
    .bind(batch)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    let mut processed = 0i64;
    for row in rows {
        let belief_id: i64 = row.try_get("id").unwrap_or(0);
        if belief_id <= 0 {
            continue;
        }
        let kind: String = row
            .try_get::<String, _>("kind")
            .unwrap_or_else(|_| "fact".to_string());
        let value_literal: String = row
            .try_get::<String, _>("value_literal")
            .unwrap_or_default();
        let sensitivity = detect_sensitivity(&value_literal)
            .map(|level| level.as_str().to_string())
            .unwrap_or_else(|| "none".to_string());
        let _ = sqlx::query(
            "INSERT OR REPLACE INTO memory_sensitivity (belief_id, kind, sensitivity, created_at, updated_at)
             VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        )
        .bind(belief_id)
        .bind(&kind)
        .bind(&sensitivity)
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;
        processed += 1;
    }
    Ok(processed)
}

async fn next_assistant_message_after(
    pool: &sqlx::SqlitePool,
    after: &DateTime<Utc>,
) -> Option<(DateTime<Utc>, String)> {
    let row = sqlx::query(
        "SELECT content, created_at FROM messages
         WHERE role = 'assistant' AND status = 'complete'
           AND (metadata IS NULL OR json_extract(metadata, '$.source') != 'monologue')
           AND datetime(created_at) > datetime(?)
         ORDER BY datetime(created_at) ASC LIMIT 1",
    )
    .bind(after.to_rfc3339())
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()?;
    let content: String = row.get("content");
    let created_at: String = row.get("created_at");
    let observed_at = parse_timestamp(&created_at)?;
    Some((observed_at, content))
}

async fn next_tool_dispatch_after(
    pool: &sqlx::SqlitePool,
    after: &DateTime<Utc>,
) -> Option<(DateTime<Utc>, String)> {
    let row = sqlx::query(
        "SELECT status, created_at FROM tool_dispatches
         WHERE status IN ('success', 'failed') AND datetime(created_at) > datetime(?)
         ORDER BY datetime(created_at) ASC LIMIT 1",
    )
    .bind(after.to_rfc3339())
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()?;
    let status: String = row.get("status");
    let created_at: String = row.get("created_at");
    let observed_at = parse_timestamp(&created_at)?;
    Some((observed_at, status))
}

async fn next_subject_snapshot_after(
    pool: &sqlx::SqlitePool,
    after: &DateTime<Utc>,
) -> Option<(DateTime<Utc>, String)> {
    let row = sqlx::query(
        "SELECT subject_state_json, timestamp FROM subject_snapshots
         WHERE julianday(timestamp) > julianday(?)
         ORDER BY julianday(timestamp) ASC
         LIMIT 1",
    )
    .bind(after.to_rfc3339())
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()?;
    let subject_state_json: String = row.get("subject_state_json");
    let timestamp: String = row.get("timestamp");
    let observed_at = parse_timestamp(&timestamp)?;
    Some((observed_at, subject_state_json))
}

fn rate(count: i64, total: i64, default_if_zero: f64) -> f64 {
    if total <= 0 {
        default_if_zero
    } else {
        (count as f64 / total as f64).clamp(0.0, 1.0)
    }
}

async fn tool_success_rate(pool: &sqlx::SqlitePool, start: &DateTime<Utc>, end: &DateTime<Utc>) -> f64 {
    let success: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM tool_dispatches
         WHERE status = 'success' AND datetime(created_at) > datetime(?) AND datetime(created_at) <= datetime(?)",
    )
    .bind(start.to_rfc3339())
    .bind(end.to_rfc3339())
    .fetch_one(pool)
    .await
    .unwrap_or(0);
    let failed: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM tool_dispatches
         WHERE status = 'failed' AND datetime(created_at) > datetime(?) AND datetime(created_at) <= datetime(?)",
    )
    .bind(start.to_rfc3339())
    .bind(end.to_rfc3339())
    .fetch_one(pool)
    .await
    .unwrap_or(0);
    let total = success + failed;
    rate(success, total, 1.0)
}

async fn memory_pass_rate(pool: &sqlx::SqlitePool, start: &DateTime<Utc>, end: &DateTime<Utc>) -> f64 {
    let memory_passes: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_write_ledger
         WHERE category = 'memory_pass' AND datetime(created_at) > datetime(?) AND datetime(created_at) <= datetime(?)",
    )
    .bind(start.to_rfc3339())
    .bind(end.to_rfc3339())
    .fetch_one(pool)
    .await
    .unwrap_or(0);
    let runs: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM runs
         WHERE datetime(started_at) > datetime(?) AND datetime(started_at) <= datetime(?)",
    )
    .bind(start.to_rfc3339())
    .bind(end.to_rfc3339())
    .fetch_one(pool)
    .await
    .unwrap_or(0);
    rate(memory_passes, runs, 0.0)
}

async fn clarification_rate(pool: &sqlx::SqlitePool, start: &DateTime<Utc>, end: &DateTime<Utc>) -> f64 {
    let clarifications: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ics_pending_clarify
         WHERE datetime(created_at) > datetime(?) AND datetime(created_at) <= datetime(?)",
    )
    .bind(start.to_rfc3339())
    .bind(end.to_rfc3339())
    .fetch_one(pool)
    .await
    .unwrap_or(0);
    let runs: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM runs
         WHERE datetime(started_at) > datetime(?) AND datetime(started_at) <= datetime(?)",
    )
    .bind(start.to_rfc3339())
    .bind(end.to_rfc3339())
    .fetch_one(pool)
    .await
    .unwrap_or(0);
    rate(clarifications, runs, 0.0)
}

async fn refusal_rate(pool: &sqlx::SqlitePool, start: &DateTime<Utc>, end: &DateTime<Utc>) -> f64 {
    let refusals: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'user_refused_inputs'
           AND datetime(timestamp) > datetime(?) AND datetime(timestamp) <= datetime(?)",
    )
    .bind(start.to_rfc3339())
    .bind(end.to_rfc3339())
    .fetch_one(pool)
    .await
    .unwrap_or(0);
    let runs: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM runs
         WHERE datetime(started_at) > datetime(?) AND datetime(started_at) <= datetime(?)",
    )
    .bind(start.to_rfc3339())
    .bind(end.to_rfc3339())
    .fetch_one(pool)
    .await
    .unwrap_or(0);
    rate(refusals, runs, 0.0)
}

async fn response_len_avg(pool: &sqlx::SqlitePool, start: &DateTime<Utc>, end: &DateTime<Utc>) -> f64 {
    let rows = sqlx::query(
        "SELECT content FROM messages
         WHERE role = 'assistant' AND status = 'complete'
           AND (metadata IS NULL OR json_extract(metadata, '$.source') != 'monologue')
           AND datetime(created_at) > datetime(?) AND datetime(created_at) <= datetime(?)",
    )
    .bind(start.to_rfc3339())
    .bind(end.to_rfc3339())
    .fetch_all(pool)
    .await
    .unwrap_or_default();
    if rows.is_empty() {
        return 0.0;
    }
    let mut total = 0usize;
    for row in rows.iter() {
        let content: String = row.get("content");
        total += content.chars().count();
    }
    total as f64 / rows.len() as f64
}

async fn workspace_stability_rate(
    pool: &sqlx::SqlitePool,
    start: &DateTime<Utc>,
    end: &DateTime<Utc>,
) -> f64 {
    let updates: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'workspace_update'
           AND datetime(timestamp) > datetime(?) AND datetime(timestamp) <= datetime(?)",
    )
    .bind(start.to_rfc3339())
    .bind(end.to_rfc3339())
    .fetch_one(pool)
    .await
    .unwrap_or(0);

    let demotions: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') IN ('workspace_demoted_stale','workspace_meta_backfill_demoted','workspace_demoted_user_feedback')
           AND datetime(timestamp) > datetime(?) AND datetime(timestamp) <= datetime(?)",
    )
    .bind(start.to_rfc3339())
    .bind(end.to_rfc3339())
    .fetch_one(pool)
    .await
    .unwrap_or(0);

    if updates <= 0 {
        return if demotions == 0 { 1.0 } else { 0.0 };
    }
    let rate = 1.0 - (demotions as f64 / updates as f64);
    rate.clamp(0.0, 1.0)
}

fn layer_rank(layer: &str) -> i32 {
    match layer.trim().to_lowercase().as_str() {
        "working" => 0,
        "episodic" => 1,
        "semantic" => 2,
        "world" => 3,
        _ => 0,
    }
}

async fn apply_layer_promotion(db: &Db) -> Result<(u64, u64, u64), String> {
    let rows = sqlx::query(
        "SELECT b.id, b.layer,
                COUNT(e.id) AS evidence_count,
                COUNT(DISTINCT e.source_type) AS source_type_count,
                SUM(CASE WHEN e.source_type = 'user' THEN 1 ELSE 0 END) AS user_count,
                SUM(CASE WHEN e.source_type = 'system' THEN 1 ELSE 0 END) AS system_count,
                COUNT(DISTINCT CASE WHEN e.source_type = 'tool' AND e.source_ref LIKE 'http%' THEN e.source_ref END) AS web_sources
         FROM ics_beliefs b
         LEFT JOIN ics_evidence_events e ON e.belief_id = b.id
         WHERE b.status = 'active'
         GROUP BY b.id"
    )
    .fetch_all(&db.pool)
    .await
    .map_err(|e| e.to_string())?;

    let mut episodic = 0u64;
    let mut semantic = 0u64;
    let mut world = 0u64;

    for row in rows {
        let id: i64 = row.get("id");
        let layer: String = row.get("layer");
        let evidence_count: i64 = row.get("evidence_count");
        if evidence_count <= 0 {
            continue;
        }
        let source_type_count: i64 = row.get("source_type_count");
        let user_count: i64 = row.get("user_count");
        let system_count: i64 = row.get("system_count");
        let web_sources: i64 = row.get("web_sources");

        let has_user_or_system = user_count > 0 || system_count > 0;
        let corroborated = source_type_count >= 2 || web_sources >= 2 || has_user_or_system;

        let target_layer = if evidence_count >= 3 && corroborated {
            "world"
        } else if evidence_count >= 2 && corroborated {
            "semantic"
        } else if evidence_count >= 1 {
            "episodic"
        } else {
            "working"
        };

        let current_rank = layer_rank(&layer);
        let target_rank = layer_rank(target_layer);
        if target_rank > current_rank {
            let _ = sqlx::query("UPDATE ics_beliefs SET layer = ? WHERE id = ?")
                .bind(target_layer)
                .bind(id)
                .execute(&db.pool)
                .await
                .map_err(|e| e.to_string())?;
            match target_layer {
                "episodic" => episodic += 1,
                "semantic" => semantic += 1,
                "world" => world += 1,
                _ => {}
            }
        }
    }

    Ok((episodic, semantic, world))
}

async fn apply_layer_decay(db: &Db, base_half_life_hours: f64) -> Result<u64, String> {
    let now = Utc::now();
    let working_hours = (base_half_life_hours / 4.0).max(6.0);
    let episodic_hours = base_half_life_hours.max(12.0);
    let semantic_hours = (base_half_life_hours * 4.0).max(48.0);

    let working_cutoff = (now - chrono::Duration::hours(working_hours as i64)).to_rfc3339();
    let episodic_cutoff = (now - chrono::Duration::hours(episodic_hours as i64)).to_rfc3339();
    let semantic_cutoff = (now - chrono::Duration::hours(semantic_hours as i64)).to_rfc3339();

    let mut deactivated = 0u64;
    deactivated += sqlx::query(
        "UPDATE ics_beliefs SET status = 'inactive'
         WHERE status = 'active' AND layer = 'working' AND last_evidence_at < ?"
    )
    .bind(&working_cutoff)
    .execute(&db.pool)
    .await
    .map_err(|e| e.to_string())?
    .rows_affected();

    deactivated += sqlx::query(
        "UPDATE ics_beliefs SET status = 'inactive'
         WHERE status = 'active' AND layer = 'episodic' AND last_evidence_at < ?"
    )
    .bind(&episodic_cutoff)
    .execute(&db.pool)
    .await
    .map_err(|e| e.to_string())?
    .rows_affected();

    deactivated += sqlx::query(
        "UPDATE ics_beliefs SET status = 'inactive'
         WHERE status = 'active' AND layer = 'semantic' AND last_evidence_at < ?"
    )
    .bind(&semantic_cutoff)
    .execute(&db.pool)
    .await
    .map_err(|e| e.to_string())?
    .rows_affected();

    Ok(deactivated)
}

async fn audit_identity_snapshot(db: &Db, app: &AppHandle) -> Result<(), String> {
    let model = db.get_self_model().await.map_err(|e| e.to_string())?;
    let snapshot = db.get_latest_identity_snapshot().await.map_err(|e| e.to_string())?;
    let Some((snapshot_id, snapshot_json, evidence_ids, _invariants)) = snapshot else {
        if let Some(snapshot_id) = bootstrap_identity_snapshot(db, app).await? {
            system_log::log_event(
                &db.pool,
                Some(app),
                "info",
                "system",
                None,
                None,
                json!({
                    "event": "identity_audit",
                    "status": "bootstrapped",
                    "snapshot_id": snapshot_id,
                }),
            )
            .await
            .map_err(|e| e.to_string())?;
            return Ok(());
        }
        system_log::log_event(
            &db.pool,
            Some(app),
            "warn",
            "system",
            None,
            None,
            json!({
                "event": "identity_audit",
                "status": "missing_snapshot",
            }),
        )
        .await
        .map_err(|e| e.to_string())?;
        return Ok(());
    };

    let snapshot_thread = snapshot_json
        .get("identity_thread")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let snapshot_confidence = snapshot_json
        .get("identity_confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(model.identity_confidence as f64) as f32;
    let snapshot_note = snapshot_json
        .get("identity_uncertainty_note")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let snapshot_updated_at = snapshot_json
        .get("identity_updated_at")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let current_thread = model.identity_thread.clone().unwrap_or_default();
    let current_note = model.identity_uncertainty_note.clone().unwrap_or_default();
    let mut drift = false;
    let mut reasons = Vec::new();
    if snapshot_thread.trim().is_empty() {
        reasons.push("missing_identity_thread");
    }
    if snapshot_updated_at.trim().is_empty() {
        reasons.push("missing_identity_updated_at");
    }
    if current_thread.trim() != snapshot_thread {
        drift = true;
        reasons.push("identity_thread_mismatch");
    }
    if (model.identity_confidence - snapshot_confidence).abs() > 0.01 {
        drift = true;
        reasons.push("identity_confidence_mismatch");
    }
    if current_note.trim() != snapshot_note {
        drift = true;
        reasons.push("identity_note_mismatch");
    }
    if evidence_ids.is_empty() {
        reasons.push("missing_evidence_ids");
    }

    let stale_days = 30;
    if let Ok(parsed) = DateTime::parse_from_rfc3339(&snapshot_updated_at) {
        let age_days = (Utc::now() - parsed.with_timezone(&Utc)).num_days();
        if age_days > stale_days {
            reasons.push("stale_snapshot");
        }
    }

    system_log::log_event(
        &db.pool,
        Some(app),
        if drift { "warn" } else { "info" },
        "system",
        None,
        None,
        json!({
            "event": "identity_audit",
            "status": if drift { "drift" } else { "ok" },
            "snapshot_id": snapshot_id,
            "reasons": reasons,
            "evidence_count": evidence_ids.len(),
        }),
    )
    .await
    .map_err(|e| e.to_string())?;

    Ok(())
}

async fn bootstrap_identity_snapshot(db: &Db, app: &AppHandle) -> Result<Option<String>, String> {
    let settings = db.get_settings().await.map_err(|e| e.to_string())?;
    let assistant_name = settings
        .assistant_display_name
        .unwrap_or_else(|| "Ergo".to_string());
    let user_name = settings
        .user_display_name
        .unwrap_or_else(|| "User".to_string());
    let identity_thread = format!(
        "{} is Symbiote, a governed assistant with evidence-gated memory, tool registry, and audit logs. Outputs are model candidates gated by a kernel; subjective experience is an open question.",
        assistant_name
    );
    let evidence_id = db
        .create_system_evidence_event(
            "default",
            "system_identity",
            &identity_thread,
            Some("bootstrap_identity"),
            "Bootstrap identity snapshot",
        )
        .await;
    let Some(evidence_id) = evidence_id else {
        system_log::log_event(
            &db.pool,
            Some(app),
            "warn",
            "system",
            None,
            None,
            json!({
                "event": "identity_audit",
                "status": "bootstrap_failed",
                "reason": "missing_evidence_id",
            }),
        )
        .await
        .map_err(|e| e.to_string())?;
        return Ok(None);
    };

    let now = Utc::now().to_rfc3339();
    let snapshot = json!({
        "identity_thread": identity_thread,
        "identity_confidence": 0.7,
        "identity_uncertainty_note": "bootstrap_snapshot",
        "identity_updated_at": now,
    });
    let invariants = json!({
        "assistant_display_name": assistant_name,
        "user_display_name": user_name,
        "contract": "symbiote_contract_v1",
    });
    let snapshot_id = db
        .create_identity_snapshot(
            &snapshot,
            &[evidence_id],
            Some(&invariants),
            Some("bootstrap"),
            Some("scheduler_bootstrap"),
        )
        .await
        .map_err(|e| e.to_string())?;

    if let Ok(mut model) = db.get_self_model().await {
        model.identity_thread = snapshot
            .get("identity_thread")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        model.identity_confidence = snapshot
            .get("identity_confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(model.identity_confidence as f64) as f32;
        model.identity_uncertainty_note = snapshot
            .get("identity_uncertainty_note")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        model.identity_updated_at = snapshot
            .get("identity_updated_at")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let _ = db.set_self_model(&model).await;
    }

    let _ = system_log::log_event(
        &db.pool,
        Some(app),
        "info",
        "system",
        None,
        None,
        json!({
            "event": "identity_snapshot_written",
            "snapshot_id": snapshot_id,
            "evidence_count": 1,
            "reason": "bootstrap",
        }),
    )
    .await;

    Ok(Some(snapshot_id))
}

async fn compute_prediction_observation(
    db: &Db,
    app: &AppHandle,
    run_id: Option<&str>,
    trace_id: Option<&str>,
    metric: &str,
    horizon: &str,
    created_at: &DateTime<Utc>,
    context_ref_json: Option<&str>,
) -> Option<(f64, DateTime<Utc>)> {
    let context_value = context_ref_json
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok());
    let snapshot_hash = context_value
        .as_ref()
        .and_then(|value| value.get("snapshot_hash").and_then(|v| v.as_str()))
        .map(|s| s.to_string());
    match horizon {
        "next_tick" => {
            if metric != "attention_focus_match" {
                let _ = log_prediction_observation_event(
                    db,
                    app,
                    run_id,
                    trace_id,
                    "prediction_observation_skipped",
                    json!({
                        "metric": metric,
                        "horizon": horizon,
                        "reason": "unsupported_metric",
                    }),
                )
                .await;
                return None;
            }
            let predicted_focus = context_value
                .as_ref()
                .and_then(|value| value.get("predicted_focus").and_then(|v| v.as_str()).map(|s| s.to_string()));
            let Some(predicted_focus) = predicted_focus else {
                let _ = log_prediction_observation_event(
                    db,
                    app,
                    run_id,
                    trace_id,
                    "prediction_observation_skipped",
                    json!({
                        "metric": metric,
                        "horizon": horizon,
                        "reason": "missing_predicted_focus",
                    }),
                )
                .await;
                return None;
            };
            let mut anchor_time = created_at.clone();
            if let Some(hash) = snapshot_hash.as_deref() {
                let snapshot_ts: Option<String> = sqlx::query_scalar(
                    "SELECT timestamp FROM subject_snapshots WHERE snapshot_hash = ? LIMIT 1",
                )
                .bind(hash)
                .fetch_optional(&db.pool)
                .await
                .ok()
                .flatten();
                if let Some(snapshot_ts) = snapshot_ts {
                    if let Some(parsed) = parse_timestamp(&snapshot_ts) {
                        anchor_time = parsed;
                    } else {
                        let _ = log_prediction_observation_event(
                            db,
                            app,
                            run_id,
                            trace_id,
                            "prediction_observation_missing_snapshot_hash",
                            json!({
                                "metric": metric,
                                "horizon": horizon,
                                "snapshot_hash": hash,
                                "reason": "timestamp_parse_failed",
                            }),
                        )
                        .await;
                    }
                } else {
                    let _ = log_prediction_observation_event(
                        db,
                        app,
                        run_id,
                        trace_id,
                        "prediction_observation_missing_snapshot_hash",
                        json!({
                            "metric": metric,
                            "horizon": horizon,
                            "snapshot_hash": hash,
                            "reason": "snapshot_not_found",
                        }),
                    )
                    .await;
                }
            }
            let Some((observed_at, subject_state_json)) =
                next_subject_snapshot_after(&db.pool, &anchor_time).await
            else {
                let _ = log_prediction_observation_event(
                    db,
                    app,
                    run_id,
                    trace_id,
                    "prediction_observation_missing_snapshot",
                    json!({
                        "metric": metric,
                        "horizon": horizon,
                        "anchor_time": anchor_time.to_rfc3339(),
                        "created_at": created_at.to_rfc3339(),
                        "snapshot_hash": snapshot_hash,
                    }),
                )
                .await;
                return None;
            };
            let observed_value = match serde_json::from_str::<serde_json::Value>(&subject_state_json) {
                Ok(value) => {
                    let refs = value
                        .get("state")
                        .and_then(|s| s.get("attention"))
                        .and_then(|a| a.get("current_focus_refs"))
                        .and_then(|v| v.as_array())
                        .cloned()
                        .unwrap_or_default();
                    if refs.iter().any(|v| v.as_str().map(|s| s == predicted_focus).unwrap_or(false)) {
                        1.0
                    } else {
                        0.0
                    }
                }
                Err(_) => 0.0,
            };
            Some((observed_value, observed_at))
        }
        "next_tool" => {
            if metric != "tool_success_rate" {
                let _ = log_prediction_observation_event(
                    db,
                    app,
                    run_id,
                    trace_id,
                    "prediction_observation_skipped",
                    json!({
                        "metric": metric,
                        "horizon": horizon,
                        "reason": "unsupported_metric",
                    }),
                )
                .await;
                return None;
            }
            let (observed_at, status) = next_tool_dispatch_after(&db.pool, created_at).await?;
            let observed_value = if status == "success" { 1.0 } else { 0.0 };
            Some((observed_value, observed_at))
        }
        "next_turn" => {
            let (observed_at, content) = next_assistant_message_after(&db.pool, created_at).await?;
            let window_start = created_at.clone();
            let window_end = observed_at.clone();
            let observed_value = match metric {
                "response_len" => content.chars().count() as f64,
                "tool_success_rate" => tool_success_rate(&db.pool, &window_start, &window_end).await,
                "memory_pass_rate" => memory_pass_rate(&db.pool, &window_start, &window_end).await,
                "clarification_rate" => clarification_rate(&db.pool, &window_start, &window_end).await,
                "refusal_rate" => refusal_rate(&db.pool, &window_start, &window_end).await,
                "workspace_stability_rate" => workspace_stability_rate(&db.pool, &window_start, &window_end).await,
                _ => {
                    let _ = log_prediction_observation_event(
                        db,
                        app,
                        run_id,
                        trace_id,
                        "prediction_observation_skipped",
                        json!({
                            "metric": metric,
                            "horizon": horizon,
                            "reason": "unsupported_metric",
                        }),
                    )
                    .await;
                    return None;
                }
            };
            Some((observed_value, observed_at))
        }
        "next_5m" | "next_hour" => {
            let horizon_minutes = if horizon == "next_5m" { 5 } else { 60 };
            let end_time = *created_at + chrono::Duration::minutes(horizon_minutes);
            if Utc::now() < end_time {
                return None;
            }
            let observed_value = match metric {
                "response_len" => response_len_avg(&db.pool, created_at, &end_time).await,
                "tool_success_rate" => tool_success_rate(&db.pool, created_at, &end_time).await,
                "memory_pass_rate" => memory_pass_rate(&db.pool, created_at, &end_time).await,
                "clarification_rate" => clarification_rate(&db.pool, created_at, &end_time).await,
                "refusal_rate" => refusal_rate(&db.pool, created_at, &end_time).await,
                "workspace_stability_rate" => workspace_stability_rate(&db.pool, created_at, &end_time).await,
                _ => {
                    let _ = log_prediction_observation_event(
                        db,
                        app,
                        run_id,
                        trace_id,
                        "prediction_observation_skipped",
                        json!({
                            "metric": metric,
                            "horizon": horizon,
                            "reason": "unsupported_metric",
                        }),
                    )
                    .await;
                    return None;
                }
            };
            Some((observed_value, end_time))
        }
        _ => {
            let _ = log_prediction_observation_event(
                db,
                app,
                run_id,
                trace_id,
                "prediction_observation_skipped",
                json!({
                    "metric": metric,
                    "horizon": horizon,
                    "reason": "unsupported_horizon",
                }),
            )
            .await;
            None
        }
    }
}

async fn log_prediction_observation_event(
    db: &Db,
    app: &AppHandle,
    run_id: Option<&str>,
    trace_id: Option<&str>,
    event: &str,
    mut payload: Value,
) {
    if let Some(obj) = payload.as_object_mut() {
        obj.insert("event".to_string(), json!(event));
    } else {
        payload = json!({
            "event": event,
            "detail": payload,
        });
    }
    let _ = system_log::log_event(
        &db.pool,
        Some(app),
        "info",
        "scheduler",
        run_id,
        trace_id,
        payload,
    )
    .await;
}

async fn process_post_processing_jobs(db: Arc<Db>, app: AppHandle, kernel: Arc<Kernel>) {
    let Some(job) = db.claim_post_processing_job().await else {
        return;
    };
    let job_id = job.job_id.clone();
    let job_type = job.job_type.clone();
    let conversation_id = job.conversation_id.clone();
    let run_id = job.run_id.clone();

    let _ = system_log::log_event(
        &db.pool,
        Some(&app),
        "info",
        "scheduler",
        run_id.as_deref(),
        None,
        json!({
            "event": "post_processing_job_started",
            "job_id": job_id,
            "job_type": job_type,
            "conversation_id": conversation_id,
        }),
    )
    .await;

    let result = match job_type.as_str() {
        "rolling_summary_update" => {
            if let Some(cid) = conversation_id.as_deref() {
                rolling_summary::update_rolling_summary(
                    db.clone(),
                    kernel.model_client(),
                    cid,
                    Some(&app),
                    "scheduler",
                    "scheduler_compaction",
                    run_id.as_deref(),
                    None,
                )
                .await
                .map(|_| ())
            } else {
                Err("missing_conversation_id".to_string())
            }
        }
        _ => Err("unknown_job_type".to_string()),
    };

    match result {
        Ok(_) => {
            let _ = db.mark_post_processing_job_completed(&job_id).await;
            if job_type == "rolling_summary_update" {
                if let Some(cid) = conversation_id.as_deref() {
                    clear_rolling_summary_retry_attempts(cid);
                    let _ = db.set_summary_pending(cid, false).await;
                }
            }
            let _ = system_log::log_event(
                &db.pool,
                Some(&app),
                "info",
                "scheduler",
                run_id.as_deref(),
                None,
                json!({
                    "event": "post_processing_job_completed",
                    "job_id": job_id,
                    "job_type": job_type,
                    "conversation_id": conversation_id,
                }),
            )
            .await;
        }
        Err(err) => {
            let _ = db.mark_post_processing_job_failed(&job_id, &err).await;
            let mut retry_scheduled = false;
            if job_type == "rolling_summary_update" {
                if let Some(cid) = conversation_id.as_deref() {
                    if err != "memory_policy_blocked" {
                        let attempt = record_rolling_summary_retry_attempt(cid);
                        if attempt <= ROLLING_SUMMARY_RETRY_LIMIT {
                            let delay_secs = rolling_summary_retry_delay_secs(attempt);
                            retry_scheduled = true;
                            let db_clone = db.clone();
                            let app_clone = app.clone();
                            let cid_owned = cid.to_string();
                            let run_id_owned = run_id.as_ref().map(|s| s.to_string());
                            tokio::spawn(async move {
                                tokio::time::sleep(Duration::from_secs(delay_secs)).await;
                                let enqueue_result = db_clone
                                    .enqueue_post_processing_job_with_priority(
                                        "rolling_summary_update",
                                        Some(&cid_owned),
                                        run_id_owned.as_deref(),
                                        2,
                                    )
                                    .await;
                                let _ = system_log::log_event(
                                    &db_clone.pool,
                                    Some(&app_clone),
                                    "info",
                                    "scheduler",
                                    run_id_owned.as_deref(),
                                    None,
                                    json!( {
                                        "event": "rolling_summary_retry_enqueued",
                                        "conversation_id": cid_owned,
                                        "attempt": attempt,
                                        "delay_secs": delay_secs,
                                        "ok": enqueue_result.is_ok(),
                                    }),
                                )
                                .await;
                                if enqueue_result.is_err() {
                                    let _ = db_clone.set_summary_pending(&cid_owned, true).await;
                                }
                            });
                            let _ = db.set_summary_pending(cid, true).await;
                        } else {
                            clear_rolling_summary_retry_attempts(cid);
                        }
                    } else {
                        retry_scheduled = true;
                        let _ = db.set_summary_pending(cid, true).await;
                    }
                    if !retry_scheduled {
                        let _ = db.set_summary_pending(cid, false).await;
                    }
                }
            }
            if job_type == "rolling_summary_update" && err == "memory_policy_blocked" {
                let _ = system_log::log_event(
                    &db.pool,
                    Some(&app),
                    "info",
                    "scheduler",
                    run_id.as_deref(),
                    None,
                    json!( {
                        "event": "rolling_summary_skipped",
                        "reason": "memory_policy_blocked",
                        "job_id": job_id,
                        "conversation_id": conversation_id,
                    }),
                )
                .await;
            }
            let _ = system_log::log_event(
                &db.pool,
                Some(&app),
                "warn",
                "scheduler",
                run_id.as_deref(),
                None,
                json!({
                    "event": "post_processing_job_failed",
                    "job_id": job_id,
                    "job_type": job_type,
                    "conversation_id": conversation_id,
                    "error": err,
                }),
            )
            .await;
        }
    }
}

fn classify_error_for_metric(metric: &str) -> (&'static str, Vec<&'static str>) {
    let metric = metric.to_lowercase();
    if metric.contains("attention") {
        ("ATTENTION_DRIFT", vec!["REANCHOR", "REQUIRE_VERIFY"])
    } else if metric.contains("memory") || metric.contains("workspace") {
        ("RETRIEVAL_DRIFT", vec!["REANCHOR", "REQUIRE_VERIFY"])
    } else if metric.contains("tool") {
        ("ACTION_MISMATCH", vec!["REQUIRE_VERIFY"])
    } else if metric.contains("refusal") || metric.contains("clarification") {
        ("USER_ALIGNMENT", vec!["REQUEST_USER_CLARIFICATION"])
    } else {
        ("PREDICTION_DIVERGENCE", vec!["REQUIRE_VERIFY"])
    }
}

async fn evaluate_prediction_outcomes(db: &Db, app: &AppHandle) -> Result<(), String> {
    let rows = sqlx::query(
        "SELECT p.id, p.metric, p.expected_value, p.expected_variance, p.horizon, p.created_at, p.run_id, p.trace_id,
                p.linked_claims_json, p.salience_hint, p.context_ref_json
         FROM self_predictions p
         LEFT JOIN self_prediction_outcomes o ON o.prediction_id = p.id
         WHERE o.id IS NULL AND (p.rejection_reason IS NULL OR p.rejection_reason = '')
         ORDER BY datetime(p.created_at) ASC
         LIMIT 50",
    )
    .fetch_all(&db.pool)
    .await
    .map_err(|e| e.to_string())?;

    if rows.is_empty() {
        let _ = system_log::log_event(
            &db.pool,
            Some(app),
            "info",
            "scheduler",
            None,
            None,
            json!({
                "event": "prediction_outcome_batch_empty",
            }),
        )
        .await;
        return Ok(());
    }
    let _ = system_log::log_event(
        &db.pool,
        Some(app),
        "info",
        "scheduler",
        None,
        None,
        json!({
            "event": "prediction_outcome_batch_start",
            "count": rows.len(),
        }),
    )
    .await;

    for row in rows {
        let prediction_id: String = row.get("id");
        let metric: String = row
            .try_get::<String, _>("metric")
            .unwrap_or_default()
            .to_lowercase();
        let horizon: String = row
            .try_get::<String, _>("horizon")
            .unwrap_or_default()
            .to_lowercase();
        let run_id: Option<String> = row.try_get("run_id").ok();
        let trace_id: Option<String> = row.try_get("trace_id").ok();
        let created_at_raw: String = row.try_get("created_at").unwrap_or_default();
        let Some(created_at) = parse_timestamp(&created_at_raw) else {
            let _ = system_log::log_event(
                &db.pool,
                Some(app),
                "warn",
                "scheduler",
                run_id.as_deref(),
                trace_id.as_deref(),
                json!({
                    "event": "prediction_observation_timestamp_parse_failed",
                    "prediction_id": prediction_id,
                    "metric": metric,
                    "horizon": horizon,
                    "created_at_raw": created_at_raw,
                }),
            )
            .await;
            let _ = sqlx::query("UPDATE self_predictions SET rejection_reason = ? WHERE id = ?")
                .bind("timestamp_parse_failed")
                .bind(&prediction_id)
                .execute(&db.pool)
                .await;
            continue;
        };
        let expected_value: f64 = row.try_get::<f64, _>("expected_value").unwrap_or(0.0);
        let expected_variance: f64 = row.try_get::<f64, _>("expected_variance").unwrap_or(PREDICTION_DEFAULT_VARIANCE);
        let linked_claims_json: String = row
            .try_get::<String, _>("linked_claims_json")
            .unwrap_or_else(|_| "[]".to_string());
        let context_ref_json: Option<String> = row.try_get("context_ref_json").ok();
        let salience_hint: f64 = row
            .try_get::<f64, _>("salience_hint")
            .unwrap_or(1.0)
            .max(0.1);
        let context_value = context_ref_json
            .as_deref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok());
        let snapshot_hash = context_value
            .as_ref()
            .and_then(|value| value.get("snapshot_hash").and_then(|v| v.as_str()))
            .map(|s| s.to_string());
        let predicted_focus = context_value
            .as_ref()
            .and_then(|value| value.get("predicted_focus").and_then(|v| v.as_str()))
            .map(|s| s.to_string());

        let Some((observed_value, observed_at)) =
            compute_prediction_observation(
                db,
                app,
                run_id.as_deref(),
                trace_id.as_deref(),
                &metric,
                &horizon,
                &created_at,
                context_ref_json.as_deref(),
            )
            .await
        else {
            let age_secs = Utc::now()
                .signed_duration_since(created_at)
                .num_seconds()
                .max(0);
            let _ = system_log::log_event(
                &db.pool,
                Some(app),
                "info",
                "scheduler",
                run_id.as_deref(),
                trace_id.as_deref(),
                json!({
                    "event": "prediction_observation_pending",
                    "prediction_id": prediction_id,
                    "metric": metric,
                    "horizon": horizon,
                    "age_seconds": age_secs,
                    "snapshot_hash": snapshot_hash,
                    "predicted_focus": predicted_focus,
                }),
            )
            .await;
            if age_secs >= 600 {
                let _ = sqlx::query("UPDATE self_predictions SET rejection_reason = ? WHERE id = ?")
                    .bind("observation_timeout")
                    .bind(&prediction_id)
                    .execute(&db.pool)
                    .await;
                let _ = system_log::log_event(
                    &db.pool,
                    Some(app),
                    "warn",
                    "scheduler",
                    run_id.as_deref(),
                    trace_id.as_deref(),
                    json!({
                        "event": "prediction_observation_timeout",
                        "prediction_id": prediction_id,
                        "metric": metric,
                        "horizon": horizon,
                        "age_seconds": age_secs,
                    }),
                )
                .await;
            }
            continue;
        };

        let variance = if expected_variance <= 0.0 {
            PREDICTION_DEFAULT_VARIANCE
        } else {
            expected_variance
        };
        let sigma = variance.sqrt().max(1e-6);
        let delta = observed_value - expected_value;
        let z_score = delta / sigma;
        let significant = delta.abs() >= PREDICTION_MIN_ABS_DELTA && z_score.abs() >= PREDICTION_MIN_SIGMA;
        let previous_significant: bool = sqlx::query_scalar(
            "SELECT o.significant
             FROM self_prediction_outcomes o
             JOIN self_predictions p ON o.prediction_id = p.id
             WHERE p.metric = ? AND datetime(o.observed_at) < datetime(?)
             ORDER BY datetime(o.observed_at) DESC
             LIMIT 1",
        )
        .bind(&metric)
        .bind(observed_at.to_rfc3339())
        .fetch_optional(&db.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0)
        > 0;

        let outcome_id = Uuid::new_v4().to_string();
        let evidence_refs_json = json!({
            "metric": metric,
            "horizon": horizon,
            "observed_at": observed_at.to_rfc3339(),
        })
        .to_string();
        let outcome_result = sqlx::query(
            "INSERT INTO self_prediction_outcomes
             (id, prediction_id, observed_value, delta, z_score, significant, evidence_refs_json, observed_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&outcome_id)
        .bind(&prediction_id)
        .bind(observed_value)
        .bind(delta)
        .bind(z_score)
        .bind(if significant { 1 } else { 0 })
        .bind(&evidence_refs_json)
        .bind(observed_at.to_rfc3339())
        .execute(&db.pool)
        .await;
        if let Err(err) = outcome_result {
            let _ = system_log::log_event(
                &db.pool,
                Some(app),
                "warn",
                "scheduler",
                run_id.as_deref(),
                trace_id.as_deref(),
                json!({
                    "event": "prediction_outcome_insert_failed",
                    "prediction_id": prediction_id,
                    "error": err.to_string(),
                }),
            )
            .await;
            continue;
        }
        let _ = system_log::log_event(
            &db.pool,
            Some(app),
            "info",
            "scheduler",
            run_id.as_deref(),
            trace_id.as_deref(),
            json!({
                "event": "prediction_outcome_written",
                "prediction_id": prediction_id,
                "outcome_id": outcome_id,
                "observed_value": observed_value,
                "observed_at": observed_at.to_rfc3339(),
                "delta": delta,
                "z_score": z_score,
                "significant": significant,
                "previous_significant": previous_significant,
            }),
        )
        .await;

        let residual_id = Uuid::new_v4().to_string();
        let normalized_residual = z_score;
        let salience_score = normalized_residual.abs() * salience_hint;
        let residual_result = sqlx::query(
            "INSERT INTO residual_vectors
             (residual_id, prediction_id, outcome_id, residual_value, normalized_residual, salience_score, created_at)
             VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(&residual_id)
        .bind(&prediction_id)
        .bind(&outcome_id)
        .bind(delta)
        .bind(normalized_residual)
        .bind(salience_score)
        .execute(&db.pool)
        .await;
        if let Err(err) = residual_result {
            let _ = system_log::log_event(
                &db.pool,
                Some(app),
                "warn",
                "scheduler",
                run_id.as_deref(),
                trace_id.as_deref(),
                json!({
                    "event": "residual_vector_error",
                    "prediction_id": prediction_id,
                    "error": err.to_string(),
                }),
            )
            .await;
            continue;
        }

        let residual_snapshot = json!({
            "normalized_residual": normalized_residual,
            "salience_score": salience_score,
            "prediction_id": prediction_id,
            "timestamp": observed_at.to_rfc3339(),
        });
        if let Some(event_id) = db
            .create_system_evidence_event(
                "default",
                "prediction_residual_snapshot",
                &residual_id,
                Some(&prediction_id),
                &residual_snapshot.to_string(),
            )
            .await
        {
            let _ = db
                .retag_evidence_event_source_type(event_id, "prediction_residual_snapshot")
                .await;
            let _ = system_log::log_event(
                &db.pool,
                Some(app),
                "info",
                "scheduler",
                run_id.as_deref(),
                trace_id.as_deref(),
                json!({
                    "event": "prediction_residual_snapshot_recorded",
                    "prediction_id": prediction_id,
                    "residual_id": residual_id,
                    "evidence_event_id": event_id,
                }),
            )
            .await;
        }

        let _ = system_log::log_event(
            &db.pool,
            Some(app),
            "info",
            "scheduler",
            run_id.as_deref(),
            trace_id.as_deref(),
            json!({
                "event": "residual_vector_written",
                "residual_id": residual_id,
                "prediction_id": prediction_id,
                "normalized_residual": normalized_residual,
                "salience_score": salience_score,
            }),
        )
        .await;

        if significant {
            let (classification, recommended_actions) = classify_error_for_metric(&metric);
            let error_event_id = Uuid::new_v4().to_string();
            let recommended_actions_json =
                serde_json::to_string(&recommended_actions).unwrap_or_else(|_| "[]".to_string());
            let _ = sqlx::query(
                "INSERT INTO error_events
                 (error_event_id, residual_id, linked_claims_json, classification, status, recommended_actions_json, created_at)
                 VALUES (?, ?, ?, ?, 'OPEN', ?, CURRENT_TIMESTAMP)",
            )
            .bind(&error_event_id)
            .bind(&residual_id)
            .bind(&linked_claims_json)
            .bind(classification)
            .bind(&recommended_actions_json)
            .execute(&db.pool)
            .await;

            let _ = system_log::log_event(
                &db.pool,
                Some(app),
                "info",
                "scheduler",
                run_id.as_deref(),
                trace_id.as_deref(),
                json!({
                    "event": "error_event_opened",
                    "error_event_id": error_event_id,
                    "residual_id": residual_id,
                    "classification": classification,
                }),
            )
            .await;

            let snippet = format!(
                "prediction divergence {} expected {:.3} observed {:.3} z {:.2}",
                metric, expected_value, observed_value, z_score
            );
            let evidence_id = db
                .create_system_evidence_event(
                    "default",
                    &format!("prediction.divergence.{}", metric),
                    &format!("{:.3}", delta),
                    Some(&prediction_id),
                    &snippet,
                )
                .await;
            if let Some(evidence_id) = evidence_id {
                let evidence_ids = vec![evidence_id];
                let _ = write_self_fact(
                    &db.pool,
                    &format!("prediction.divergence.{}", metric),
                    &format!("{:.3}", delta),
                    &snippet,
                    Some(observed_at),
                    SourceType::System,
                    Some(&evidence_ids),
                )
                .await;
            }
            let _ = system_log::log_event(
                &db.pool,
                Some(app),
                "info",
                "scheduler",
                run_id.as_deref(),
                trace_id.as_deref(),
                json!({
                    "event": "prediction_divergence",
                    "prediction_id": prediction_id,
                    "metric": metric,
                    "expected_value": expected_value,
                    "observed_value": observed_value,
                    "delta": delta,
                    "z_score": z_score,
                }),
            )
            .await;

            let _ = system_log::log_event(
                &db.pool,
                Some(app),
                "info",
                "scheduler",
                run_id.as_deref(),
                trace_id.as_deref(),
                json!({
                    "event": if previous_significant { "prediction_divergence_persisted" } else { "prediction_divergence_started" },
                    "prediction_id": prediction_id,
                    "metric": metric,
                    "expected_value": expected_value,
                    "observed_value": observed_value,
                    "delta": delta,
                    "z_score": z_score,
                }),
            )
            .await;
        } else if previous_significant {
            let resolved = sqlx::query(
                "UPDATE error_events
                 SET status = 'RESOLVED'
                 WHERE error_event_id IN (
                    SELECT e.error_event_id
                    FROM error_events e
                    JOIN residual_vectors r ON e.residual_id = r.residual_id
                    WHERE r.prediction_id = ? AND e.status = 'OPEN'
                    ORDER BY datetime(e.created_at) DESC
                    LIMIT 1
                 )",
            )
            .bind(&prediction_id)
            .execute(&db.pool)
            .await
            .ok()
            .map(|res| res.rows_affected())
            .unwrap_or(0);

            if resolved > 0 {
                let _ = system_log::log_event(
                    &db.pool,
                    Some(app),
                    "info",
                    "scheduler",
                    run_id.as_deref(),
                    trace_id.as_deref(),
                    json!({
                        "event": "error_event_resolved",
                        "prediction_id": prediction_id,
                    }),
                )
                .await;
            }

            let _ = system_log::log_event(
                &db.pool,
                Some(app),
                "info",
                "scheduler",
                run_id.as_deref(),
                trace_id.as_deref(),
                json!({
                    "event": "prediction_divergence_resolved",
                    "prediction_id": prediction_id,
                    "metric": metric,
                    "expected_value": expected_value,
                    "observed_value": observed_value,
                    "delta": delta,
                    "z_score": z_score,
                }),
            )
            .await;
        }
    }

    Ok(())
}

async fn internal_state_map_bootstrap_ready(db: &Db) -> Option<i64> {
    let previous_version: i64 =
        sqlx::query_scalar("SELECT COALESCE(MAX(version), 0) FROM internal_state_map")
            .fetch_one(&db.pool)
            .await
            .unwrap_or(0);
    if previous_version > 0 {
        return None;
    }
    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ics_evidence_events
         WHERE source_type IN ('wave_state', 'attention_schema_snapshot', 'prediction_residual_snapshot')",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    if count >= INTERNAL_STATE_MAP_BOOTSTRAP_MIN {
        Some(count)
    } else {
        None
    }
}

async fn run_internal_state_map_calibration(db: &Db, app: &AppHandle) {
    let previous_version: i64 = sqlx::query_scalar("SELECT COALESCE(MAX(version), 0) FROM internal_state_map")
        .fetch_one(&db.pool)
        .await
        .unwrap_or(0);
    let required_min = if previous_version == 0 { 25 } else { INTERNAL_STATE_MAP_MIN_OBSERVATIONS };
    let cutoff = (Utc::now() - ChronoDuration::hours(INTERNAL_STATE_MAP_WINDOW_HOURS)).to_rfc3339();
    let rows = sqlx::query(
        "SELECT source_type, snippet FROM ics_evidence_events
         WHERE source_type IN ('wave_state', 'attention_schema_snapshot', 'prediction_residual_snapshot')
           AND datetime(created_at) >= datetime(?)",
    )
    .bind(&cutoff)
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();

    let mut metrics: HashMap<String, Vec<f64>> = HashMap::new();
    for row in rows {
        let source_type: String = row.try_get("source_type").unwrap_or_default();
        let snippet: Option<String> = row.try_get("snippet").ok();
        let Some(snippet) = snippet else { continue; };
        let parsed: Value = serde_json::from_str(&snippet).unwrap_or_else(|_| json!({}));
        match source_type.as_str() {
            "wave_state" => {
                if let Some(bands) = parsed.get("bands").and_then(|v| v.as_object()) {
                    for (band, entry) in bands.iter() {
                        if let Some(amplitude) = entry.get("amplitude").and_then(|v| v.as_f64()) {
                            metrics
                                .entry(format!("wave.band.{}.amplitude", band))
                                .or_default()
                                .push(amplitude);
                        }
                    }
                }
            }
            "attention_schema_snapshot" => {
                if let Some(value) = parsed.get("capacity_usage").and_then(|v| v.as_f64()) {
                    metrics
                        .entry("attention.capacity_usage".to_string())
                        .or_default()
                        .push(value);
                }
                if let Some(value) = parsed.get("stability").and_then(|v| v.as_f64()) {
                    metrics
                        .entry("attention.stability".to_string())
                        .or_default()
                        .push(value);
                }
            }
            "prediction_residual_snapshot" => {
                if let Some(value) = parsed.get("normalized_residual").and_then(|v| v.as_f64()) {
                    metrics
                        .entry("residual.normalized_residual".to_string())
                        .or_default()
                        .push(value);
                }
                if let Some(value) = parsed.get("salience_score").and_then(|v| v.as_f64()) {
                    metrics
                        .entry("residual.salience_score".to_string())
                        .or_default()
                        .push(value);
                }
            }
            _ => {}
        }
    }

    let mut required_metrics: Vec<String> = cognitive_wave::WaveBand::all()
        .iter()
        .map(|band| format!("wave.band.{}.amplitude", band.label()))
        .collect();
    required_metrics.push("attention.capacity_usage".to_string());
    required_metrics.push("attention.stability".to_string());
    required_metrics.push("residual.normalized_residual".to_string());
    required_metrics.push("residual.salience_score".to_string());

    let mut counts = serde_json::Map::new();
    let mut insufficient = false;
    let mut residual_missing = false;
    let mut base_missing = false;
    for metric in required_metrics.iter() {
        let count = metrics.get(metric).map(|v| v.len() as i64).unwrap_or(0);
        counts.insert(metric.clone(), json!(count));
        if count < required_min {
            insufficient = true;
            if metric.starts_with("residual.") {
                residual_missing = true;
            } else {
                base_missing = true;
            }
        }
    }

    let mut degraded = false;
    if insufficient && !base_missing && residual_missing {
        let last_residual_raw: Option<String> = sqlx::query_scalar(
            "SELECT MAX(created_at) FROM ics_evidence_events WHERE source_type = 'prediction_residual_snapshot'",
        )
        .fetch_optional(&db.pool)
        .await
        .ok()
        .flatten();
        let residual_stale = last_residual_raw
            .as_deref()
            .and_then(parse_timestamp)
            .map(|ts| {
                Utc::now()
                    .signed_duration_since(ts)
                    .num_hours()
                    >= INTERNAL_STATE_MAP_DEGRADED_WINDOW_HOURS
            })
            .unwrap_or(true);
        if residual_stale {
            degraded = true;
        }
    }

    if insufficient && !degraded {
        let _ = system_log::log_event(
            &db.pool,
            Some(app),
            "info",
            "scheduler",
            None,
            None,
            json!({
                "event": "internal_state_map_insufficient_data",
                "required_min": required_min,
                "window_hours": INTERNAL_STATE_MAP_WINDOW_HOURS,
                "metric_counts": counts,
            }),
        )
        .await;
        return;
    }

    if degraded {
        required_metrics.retain(|metric| !metric.starts_with("residual."));
        let _ = system_log::log_event(
            &db.pool,
            Some(app),
            "warn",
            "scheduler",
            None,
            None,
            json!({
                "event": "internal_state_map_degraded_mode",
                "required_min": required_min,
                "window_hours": INTERNAL_STATE_MAP_WINDOW_HOURS,
                "residual_missing_window_hours": INTERNAL_STATE_MAP_DEGRADED_WINDOW_HOURS,
                "metric_counts": counts,
            }),
        )
        .await;
    }

    fn percentile(sorted: &[f64], p: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let p = p.clamp(0.0, 1.0);
        let rank = p * (sorted.len().saturating_sub(1)) as f64;
        let idx = rank.floor() as usize;
        let frac = rank - idx as f64;
        if idx + 1 < sorted.len() {
            sorted[idx] + (sorted[idx + 1] - sorted[idx]) * frac
        } else {
            sorted[idx]
        }
    }

    let new_version = previous_version + 1;

    let mut tx = match db.pool.begin().await {
        Ok(tx) => tx,
        Err(_) => return,
    };
    let author = "scheduler";
    let rationale = if degraded {
        format!(
            "quantile_calibration_window_{}h_degraded",
            INTERNAL_STATE_MAP_WINDOW_HOURS
        )
    } else {
        format!("quantile_calibration_window_{}h", INTERNAL_STATE_MAP_WINDOW_HOURS)
    };
    let mut ranges_log = serde_json::Map::new();

    for metric in required_metrics.iter() {
        let Some(mut values) = metrics.get(metric).cloned() else { continue; };
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min = *values.first().unwrap_or(&0.0);
        let max = *values.last().unwrap_or(&0.0);
        let p20 = percentile(&values, 0.2);
        let p80 = percentile(&values, 0.8);

        let ranges = vec![
            ("low", min, p20),
            ("normal", p20, p80),
            ("elevated", p80, max),
        ];
        let mut metric_ranges = serde_json::Map::new();
        for (label, range_min, range_max) in ranges {
            metric_ranges.insert(
                label.to_string(),
                json!({ "min": range_min, "max": range_max }),
            );
            let _ = sqlx::query(
                "INSERT INTO internal_state_map
                 (version, metric, range_min, range_max, label, author, rationale, degraded, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            )
            .bind(new_version)
            .bind(metric)
            .bind(range_min)
            .bind(range_max)
            .bind(label)
            .bind(author)
            .bind(&rationale)
            .bind(if degraded { 1 } else { 0 })
            .execute(&mut *tx)
            .await;
        }
        ranges_log.insert(metric.to_string(), json!(metric_ranges));
    }

    let _ = tx.commit().await;

    let _ = system_log::log_event(
        &db.pool,
        Some(app),
        "info",
        "scheduler",
        None,
        None,
        json!({
            "event": "internal_state_map_changed",
            "previous_version": previous_version,
            "version": new_version,
            "author": author,
            "window_hours": INTERNAL_STATE_MAP_WINDOW_HOURS,
            "required_min": required_min,
            "degraded": degraded,
            "metric_counts": counts,
            "ranges": ranges_log,
        }),
    )
    .await;

    refresh_subject_snapshots_after_map(db, app, new_version).await;
}

async fn refresh_subject_snapshots_after_map(db: &Db, app: &AppHandle, map_version: i64) {
    let conv_rows = sqlx::query("SELECT conversation_id FROM conversations ORDER BY datetime(updated_at) DESC LIMIT 10")
        .fetch_all(&db.pool)
        .await
        .unwrap_or_default();
    for row in conv_rows {
        let conversation_id: String = row.try_get("conversation_id").unwrap_or_else(|_| "default".to_string());
        let raw = db.get_kernel_state(&conversation_id).await.ok().flatten();
        let kernel_state: KernelState = raw
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| KernelState::default_for(&conversation_id));
        let previous = subject_state::load_latest_subject_state(db, &conversation_id).await;
        let subject_state = match subject_state::build_subject_state(db, &kernel_state, previous.as_ref()).await {
            Ok(state) => state,
            Err(err) => {
                let _ = system_log::log_event(
                    &db.pool,
                    Some(app),
                    "warn",
                    "scheduler",
                    None,
                    None,
                    json!({
                        "event": "internal_state_map_snapshot_refresh_failed",
                        "reason": "build_subject_state",
                        "conversation_id": conversation_id,
                        "error": err,
                    }),
                )
                .await;
                continue;
            }
        };
        let tick_id = Uuid::new_v4().to_string();
        let snapshot = match subject_state::snapshot_subject_state(&subject_state, &tick_id, &conversation_id, None) {
            Ok(record) => record,
            Err(err) => {
                let _ = system_log::log_event(
                    &db.pool,
                    Some(app),
                    "warn",
                    "scheduler",
                    None,
                    None,
                    json!({
                        "event": "internal_state_map_snapshot_refresh_failed",
                        "reason": "snapshot_subject_state",
                        "conversation_id": conversation_id,
                        "error": err,
                    }),
                )
                .await;
                continue;
            }
        };
        if let Err(err) = subject_state::persist_subject_snapshot(db, &snapshot).await {
            let _ = system_log::log_event(
                &db.pool,
                Some(app),
                "warn",
                "scheduler",
                None,
                None,
                json!({
                    "event": "internal_state_map_snapshot_refresh_failed",
                    "reason": "persist_subject_snapshot",
                    "conversation_id": conversation_id,
                    "error": err,
                }),
            )
            .await;
            continue;
        }
        let _ = system_log::log_event(
            &db.pool,
            Some(app),
            "info",
            "scheduler",
            None,
            None,
            json!({
                "event": "internal_state_map_snapshot_refresh",
                "conversation_id": conversation_id,
                "snapshot_hash": snapshot.snapshot_hash,
                "tick_id": snapshot.tick_id,
                "map_version": map_version,
            }),
        )
        .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::fs;
    use std::path::PathBuf;
    use crate::core::memory::types::Scope;
    use crate::core::memory::canonical::compute_value_hash;

    #[test]
    fn scheduler_intervals_are_positive() {
        assert!(THOUGHT_INTERVAL_SECS > 0);
        assert!(CLAIMS_EVAL_INTERVAL_SECS > 0);
        assert!(HEARTBEAT_INTERVAL_SECS > 0);
        assert!(DREAM_INTERVAL_SECS > 0);
        assert!(SELF_MEMORY_BRIDGE_INTERVAL_SECS > 0);
        assert!(DECAY_INTERVAL_SECS > 0);
        assert!(PREDICTION_EVAL_INTERVAL_SECS > 0);
    }

    async fn setup_db() -> Db {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("pool");
        let schema_path = PathBuf::from("src/db/schema.sql");
        let schema_sql = fs::read_to_string(&schema_path).expect("schema");
        sqlx::query(&schema_sql).execute(&pool).await.expect("apply schema");
        sqlx::query("INSERT INTO settings (id, schema_version, api_base_url) VALUES (1, 1, 'http://localhost')")
            .execute(&pool)
            .await
            .expect("seed settings");
        Db { pool }
    }

    #[tokio::test]
    async fn apply_layer_promotion_advances_with_corroboration() {
        let db = setup_db().await;
        let entity_id: i64 = sqlx::query(
            "INSERT INTO ics_entities (label, label_canonical, aliases, aliases_canonical, keys, resolution_state)
             VALUES ('Ergo', 'ergo', '[]', '[]', '[]', 'normal')
             RETURNING id",
        )
        .fetch_one(&db.pool)
        .await
        .map(|row| row.get::<i64, _>("id"))
        .expect("entity");

        let scope = serde_json::to_string(&Scope::Session).unwrap_or_else(|_| "\"session\"".to_string());
        let value_hash = compute_value_hash("value");
        let belief_id: i64 = sqlx::query(
            "INSERT INTO ics_beliefs
             (kind, scope, status, layer, polarity, confidence, salience, topic_key, signature_hash, evidence_weight_total, last_evidence_at, created_at)
             VALUES ('fact', ?, 'active', 'working', 'assert', 1.0, 1.0, 'fact:test', 'sig:test', 0.0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
             RETURNING id",
        )
        .bind(&scope)
        .fetch_one(&db.pool)
        .await
        .map(|row| row.get::<i64, _>("id"))
        .expect("belief");

        let _ = sqlx::query(
            "INSERT INTO ics_fact_beliefs (belief_id, subject_entity_id, key, value_literal, value_hash)
             VALUES (?, ?, 'test.key', 'value', ?)",
        )
        .bind(belief_id)
        .bind(entity_id)
        .bind(&value_hash)
        .execute(&db.pool)
        .await;

        let _ = sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id)
             VALUES (?, 'tool', 'http://example.com', 'snippet', 0.6, NULL)",
        )
        .bind(belief_id)
        .execute(&db.pool)
        .await;
        let _ = sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id)
             VALUES (?, 'user', 'user', 'snippet', 0.7, NULL)",
        )
        .bind(belief_id)
        .execute(&db.pool)
        .await;

        let _ = apply_layer_promotion(&db).await.expect("promotion");
        let layer: String = sqlx::query_scalar("SELECT layer FROM ics_beliefs WHERE id = ?")
            .bind(belief_id)
            .fetch_one(&db.pool)
            .await
            .expect("layer");
        assert_eq!(layer, "semantic");
    }
}

