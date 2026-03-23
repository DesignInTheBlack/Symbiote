use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use crate::db::Db;

use crate::core::system_log;

const DEFAULT_WINDOW_HOURS: i64 = 24;
const TELEMETRY_ACTIVE_INTERVAL_SECS: i64 = 120;
const TELEMETRY_IDLE_INTERVAL_SECS: i64 = 900;
const TELEMETRY_ACTIVE_WINDOW_SECS: i64 = 600;

const KEY_TOOL_SUCCESS_RATE: &str = "telemetry.tool_success_rate";
const KEY_TOOL_FAILURE_RATE: &str = "telemetry.tool_failure_rate";
const KEY_MEMORY_PASS_RATE: &str = "telemetry.memory_pass_rate";
const KEY_CLARIFICATION_RATE: &str = "telemetry.clarification_rate";
const KEY_REFUSAL_RATE: &str = "telemetry.refusal_rate";
const KEY_FEEDBACK_PUSHBACK_RATE: &str = "telemetry.user_feedback_pushback_rate";
const KEY_FEEDBACK_CLARIFY_RATE: &str = "telemetry.user_feedback_clarify_rate";
const KEY_FEEDBACK_FOLLOWUP_RATE: &str = "telemetry.user_feedback_followup_rate";
const KEY_FEEDBACK_AGREE_RATE: &str = "telemetry.user_feedback_agree_rate";
const KEY_FEEDBACK_DISENGAGE_RATE: &str = "telemetry.user_feedback_disengage_rate";
const KEY_PREDICTION_DIVERGENCE_RATE: &str = "telemetry.prediction_divergence_rate";
const KEY_PREDICTION_DIVERGENCE_PERSISTED_RATE: &str = "telemetry.prediction_divergence_persisted_rate";
const KEY_PREDICTION_DIVERGENCE_RESOLVED_RATE: &str = "telemetry.prediction_divergence_resolved_rate";

fn parse_timestamp(raw: &str) -> Option<DateTime<Utc>> {
    if let Ok(ts) = DateTime::parse_from_rfc3339(raw) {
        return Some(ts.with_timezone(&Utc));
    }
    if let Ok(naive) = NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S") {
        return Some(Utc.from_utc_datetime(&naive));
    }
    None
}

fn rate(count: i64, total: i64) -> f32 {
    if total <= 0 {
        0.0
    } else {
        (count as f32 / total as f32).clamp(0.0, 1.0)
    }
}

pub async fn record_telemetry_snapshot(
    db: &Db,
    window_hours: Option<i64>,
) -> Result<bool, String> {
    record_telemetry_snapshot_internal(db, window_hours, false).await
}

pub async fn record_telemetry_snapshot_force(
    db: &Db,
    window_hours: Option<i64>,
) -> Result<bool, String> {
    record_telemetry_snapshot_internal(db, window_hours, true).await
}

async fn record_telemetry_snapshot_internal(
    db: &Db,
    window_hours: Option<i64>,
    force: bool,
) -> Result<bool, String> {
    let pool = &db.pool;
    let now = Utc::now();
    let window_hours = window_hours.unwrap_or(DEFAULT_WINDOW_HOURS).max(1);

    let last_at: Option<String> = sqlx::query_scalar(
        "SELECT value FROM kv_store WHERE key = 'telemetry_snapshot_last_at'",
    )
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?
    .flatten();
    let last_user_at: Option<String> = sqlx::query_scalar(
        "SELECT created_at FROM messages WHERE role = 'user' ORDER BY datetime(created_at) DESC LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?
    .flatten();
    let active_use = last_user_at
        .as_deref()
        .and_then(parse_timestamp)
        .map(|ts| now.signed_duration_since(ts).num_seconds().max(0) <= TELEMETRY_ACTIVE_WINDOW_SECS)
        .unwrap_or(false);
    let min_interval = if active_use {
        TELEMETRY_ACTIVE_INTERVAL_SECS
    } else {
        TELEMETRY_IDLE_INTERVAL_SECS
    };
    if !force {
        if let Some(last_at) = last_at.as_deref() {
            if let Some(parsed) = parse_timestamp(last_at) {
                let elapsed = now
                    .signed_duration_since(parsed)
                    .num_seconds()
                    .max(0);
                if elapsed < min_interval {
                    return Ok(false);
                }
            }
        }
    }

    let cutoff = now - chrono::Duration::hours(window_hours);
    let cutoff_str = cutoff.to_rfc3339();

    let tool_success: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM tool_dispatches
         WHERE status = 'success' AND julianday(updated_at) >= julianday(?)",
    )
    .bind(&cutoff_str)
    .fetch_one(pool)
    .await
    .unwrap_or(0);
    let tool_failed: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM tool_dispatches
         WHERE status = 'failed' AND julianday(updated_at) >= julianday(?)",
    )
    .bind(&cutoff_str)
    .fetch_one(pool)
    .await
    .unwrap_or(0);
    let tool_total = tool_success + tool_failed;

    let memory_passes: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_write_ledger
         WHERE category = 'memory_pass' AND julianday(created_at) >= julianday(?)",
    )
    .bind(&cutoff_str)
    .fetch_one(pool)
    .await
    .unwrap_or(0);

    let clarify_events: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ics_pending_clarify
         WHERE julianday(created_at) >= julianday(?)",
    )
    .bind(&cutoff_str)
    .fetch_one(pool)
    .await
    .unwrap_or(0);

    let refusal_events: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'user_refused_inputs'
           AND julianday(timestamp) >= julianday(?)",
    )
    .bind(&cutoff_str)
    .fetch_one(pool)
    .await
    .unwrap_or(0);

    let feedback_pushback: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'user_feedback_detected'
           AND json_extract(payload, '$.kind') = 'pushback'
           AND julianday(timestamp) >= julianday(?)",
    )
    .bind(&cutoff_str)
    .fetch_one(pool)
    .await
    .unwrap_or(0);
    let feedback_clarify: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'user_feedback_detected'
           AND json_extract(payload, '$.kind') = 'clarify'
           AND julianday(timestamp) >= julianday(?)",
    )
    .bind(&cutoff_str)
    .fetch_one(pool)
    .await
    .unwrap_or(0);
    let feedback_followup: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'user_feedback_detected'
           AND json_extract(payload, '$.kind') = 'follow_up'
           AND julianday(timestamp) >= julianday(?)",
    )
    .bind(&cutoff_str)
    .fetch_one(pool)
    .await
    .unwrap_or(0);
    let feedback_agree: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'user_feedback_detected'
           AND json_extract(payload, '$.kind') = 'agree'
           AND julianday(timestamp) >= julianday(?)",
    )
    .bind(&cutoff_str)
    .fetch_one(pool)
    .await
    .unwrap_or(0);
    let feedback_disengage: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'user_feedback_detected'
           AND json_extract(payload, '$.kind') = 'disengage'
           AND julianday(timestamp) >= julianday(?)",
    )
    .bind(&cutoff_str)
    .fetch_one(pool)
    .await
    .unwrap_or(0);

    let divergence_events: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'prediction_divergence'
           AND julianday(timestamp) >= julianday(?)",
    )
    .bind(&cutoff_str)
    .fetch_one(pool)
    .await
    .unwrap_or(0);
    let divergence_persisted: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'prediction_divergence_persisted'
           AND julianday(timestamp) >= julianday(?)",
    )
    .bind(&cutoff_str)
    .fetch_one(pool)
    .await
    .unwrap_or(0);
    let divergence_resolved: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'prediction_divergence_resolved'
           AND julianday(timestamp) >= julianday(?)",
    )
    .bind(&cutoff_str)
    .fetch_one(pool)
    .await
    .unwrap_or(0);

    let run_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM runs WHERE julianday(started_at) >= julianday(?)",
    )
    .bind(&cutoff_str)
    .fetch_one(pool)
    .await
    .unwrap_or(0);

    let tool_success_rate = rate(tool_success, tool_total);
    let tool_failure_rate = rate(tool_failed, tool_total);
    let memory_pass_rate = rate(memory_passes, run_count);
    let clarification_rate = rate(clarify_events, run_count);
    let refusal_rate = rate(refusal_events, run_count);
    let feedback_pushback_rate = rate(feedback_pushback, run_count);
    let feedback_clarify_rate = rate(feedback_clarify, run_count);
    let feedback_followup_rate = rate(feedback_followup, run_count);
    let feedback_agree_rate = rate(feedback_agree, run_count);
    let feedback_disengage_rate = rate(feedback_disengage, run_count);
    let divergence_rate = rate(divergence_events, run_count);
    let divergence_persisted_rate = rate(divergence_persisted, run_count);
    let divergence_resolved_rate = rate(divergence_resolved, run_count);

    let tool_snippet = format!(
        "window {}h: tool dispatches success {}, failed {}, total {}",
        window_hours, tool_success, tool_failed, tool_total
    );
    let memory_snippet = format!(
        "window {}h: memory passes {}, runs {}",
        window_hours, memory_passes, run_count
    );
    let clarification_snippet = format!(
        "window {}h: clarifications {}, runs {}",
        window_hours, clarify_events, run_count
    );
    let refusal_snippet = format!(
        "window {}h: refusals {}, runs {}",
        window_hours, refusal_events, run_count
    );
    let feedback_snippet = format!(
        "window {}h: feedback pushback {}, clarify {}, follow_up {}, agree {}, disengage {}, runs {}",
        window_hours,
        feedback_pushback,
        feedback_clarify,
        feedback_followup,
        feedback_agree,
        feedback_disengage,
        run_count
    );
    let divergence_snippet = format!(
        "window {}h: divergence {}, persisted {}, resolved {}, runs {}",
        window_hours,
        divergence_events,
        divergence_persisted,
        divergence_resolved,
        run_count
    );

    let metrics = vec![
        (KEY_TOOL_SUCCESS_RATE, tool_success_rate, tool_snippet.clone()),
        (KEY_TOOL_FAILURE_RATE, tool_failure_rate, tool_snippet),
        (KEY_MEMORY_PASS_RATE, memory_pass_rate, memory_snippet),
        (KEY_CLARIFICATION_RATE, clarification_rate, clarification_snippet),
        (KEY_REFUSAL_RATE, refusal_rate, refusal_snippet),
        (KEY_FEEDBACK_PUSHBACK_RATE, feedback_pushback_rate, feedback_snippet.clone()),
        (KEY_FEEDBACK_CLARIFY_RATE, feedback_clarify_rate, feedback_snippet.clone()),
        (KEY_FEEDBACK_FOLLOWUP_RATE, feedback_followup_rate, feedback_snippet.clone()),
        (KEY_FEEDBACK_AGREE_RATE, feedback_agree_rate, feedback_snippet.clone()),
        (KEY_FEEDBACK_DISENGAGE_RATE, feedback_disengage_rate, feedback_snippet),
        (KEY_PREDICTION_DIVERGENCE_RATE, divergence_rate, divergence_snippet.clone()),
        (KEY_PREDICTION_DIVERGENCE_PERSISTED_RATE, divergence_persisted_rate, divergence_snippet.clone()),
        (KEY_PREDICTION_DIVERGENCE_RESOLVED_RATE, divergence_resolved_rate, divergence_snippet),
    ];
    for (key, value, snippet) in metrics {
        let _ = db
            .set_key_with_keywords(key, &format!("{:.3}", value), &snippet, false)
            .await;
    }

    let _ = system_log::log_event(
        pool,
        None,
        "info",
        "system",
        None,
        None,
        serde_json::json!({
            "event": "telemetry_snapshot",
            "window_hours": window_hours,
            "tool_success_rate": format!("{:.3}", tool_success_rate),
            "tool_failure_rate": format!("{:.3}", tool_failure_rate),
            "memory_pass_rate": format!("{:.3}", memory_pass_rate),
            "clarification_rate": format!("{:.3}", clarification_rate),
            "refusal_rate": format!("{:.3}", refusal_rate),
            "feedback_pushback_rate": format!("{:.3}", feedback_pushback_rate),
            "feedback_clarify_rate": format!("{:.3}", feedback_clarify_rate),
            "feedback_followup_rate": format!("{:.3}", feedback_followup_rate),
            "feedback_agree_rate": format!("{:.3}", feedback_agree_rate),
            "feedback_disengage_rate": format!("{:.3}", feedback_disengage_rate),
            "prediction_divergence_rate": format!("{:.3}", divergence_rate),
            "prediction_divergence_persisted_rate": format!("{:.3}", divergence_persisted_rate),
            "prediction_divergence_resolved_rate": format!("{:.3}", divergence_resolved_rate),
            "run_count": run_count,
        }),
    )
    .await;

    let _ = sqlx::query(
        "INSERT OR REPLACE INTO kv_store (key, value, updated_at)
         VALUES ('telemetry_snapshot_last_at', ?, CURRENT_TIMESTAMP)",
    )
    .bind(now.to_rfc3339())
    .execute(pool)
    .await;

    let _ = system_log::log_event(
        pool,
        None,
        "info",
        "system",
        None,
        None,
        serde_json::json!({
            "event": "telemetry_snapshot_written",
            "window_hours": window_hours,
            "force": force,
            "run_count": run_count,
            "active_use": active_use,
        }),
    )
    .await;

    Ok(true)
}
