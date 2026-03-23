use serde_json::{json, Value};
use sqlx::SqlitePool;
use tauri::{AppHandle, Emitter};
use uuid::Uuid;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use once_cell::sync::Lazy;

use crate::models::SystemLogEntry;
use crate::core::system_log_schema;
use crate::core::system_controls;
use crate::core::sensitivity::{phi_consent_allowed, redact_sensitive_json};

static TELEMETRY_ALWAYS_EVENTS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "system_control_changed",
        "system_control_rejected",
        "system_health_snapshot",
        "error_event_opened",
        "error_event_resolved",
        "contract_violation",
        "contract_violation_escalated",
        "run_error",
        "safe_halt",
        "system_reset",
    ]
    .into_iter()
    .collect()
});

struct RateLimitState {
    last_emit: Instant,
    suppressed: usize,
}

const CONTRACT_VIOLATION_WINDOW_SECS: u64 = 60;
static CONTRACT_VIOLATION_RATE: Lazy<Mutex<HashMap<String, RateLimitState>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn rate_limit_contract_violation(policy_id: &str, reason: &str) -> (bool, usize) {
    let key = format!("{}::{}", policy_id, reason);
    let mut guard = CONTRACT_VIOLATION_RATE.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    let entry = guard
        .entry(key)
        .or_insert(RateLimitState {
            last_emit: now.checked_sub(Duration::from_secs(CONTRACT_VIOLATION_WINDOW_SECS)).unwrap_or(now),
            suppressed: 0,
        });
    if now.duration_since(entry.last_emit) < Duration::from_secs(CONTRACT_VIOLATION_WINDOW_SECS) {
        entry.suppressed = entry.suppressed.saturating_add(1);
        return (false, 0);
    }
    let suppressed = entry.suppressed;
    entry.suppressed = 0;
    entry.last_emit = now;
    (true, suppressed)
}

fn conversation_id_from_payload(payload: &Value) -> Option<&str> {
    payload
        .get("conversation_id")
        .and_then(|v| v.as_str())
        .or_else(|| payload.get("conversation").and_then(|v| v.as_str()))
        .or_else(|| {
            payload
                .get("context")
                .and_then(|ctx| ctx.get("conversation_id"))
                .and_then(|v| v.as_str())
        })
}

pub async fn log_event(
    pool: &SqlitePool,
    app_handle: Option<&AppHandle>,
    level: &str,
    category: &str,
    run_id: Option<&str>,
    trace_id: Option<&str>,
    payload: Value,
) -> Result<SystemLogEntry, String> {
    let mut payload = payload;
    let event_name = payload.get("event").and_then(|v| v.as_str()).map(|s| s.to_string());
    if let Some(event_name) = event_name.as_deref() {
        let event_name = event_name.to_string();
        if !system_log_schema::is_known_event(&event_name) {
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("schema_unknown_event".to_string(), json!(event_name));
            }
        }
    }
    let conversation_id = conversation_id_from_payload(&payload);
    if !phi_consent_allowed(pool, conversation_id).await {
        let (redacted, sensitivity) = redact_sensitive_json(&payload);
        if let Some(level) = sensitivity {
            payload = redacted;
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("phi_redacted".to_string(), json!(true));
                obj.insert("phi_sensitivity".to_string(), json!(level.as_str()));
            }
        }
    }
    let id = Uuid::new_v4().to_string();
    let timestamp = chrono::Utc::now().to_rfc3339();
    let payload_json = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
    let event_type = event_name.clone().unwrap_or_else(|| category.to_string());

    let telemetry_mode = fetch_telemetry_mode(pool).await;
    if !telemetry_allows_event(&telemetry_mode, level, event_name.as_deref()) {
        return Ok(SystemLogEntry {
            id,
            timestamp,
            level: level.to_string(),
            category: category.to_string(),
            run_id: run_id.map(|s| s.to_string()),
            trace_id: trace_id.map(|s| s.to_string()),
            payload,
        });
    }

    sqlx::query(
        "INSERT INTO system_logs (id, timestamp, level, category, run_id, trace_id, payload)
         VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&id)
    .bind(&timestamp)
    .bind(level)
    .bind(category)
    .bind(run_id)
    .bind(trace_id)
    .bind(&payload_json)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    let entry = SystemLogEntry {
        id,
        timestamp,
        level: level.to_string(),
        category: category.to_string(),
        run_id: run_id.map(|s| s.to_string()),
        trace_id: trace_id.map(|s| s.to_string()),
        payload,
    };

    if let Some(app) = app_handle {
        let _ = app.emit("system_log", entry.clone());
    }

    let tags_json = serde_json::json!({
        "category": category,
        "level": level,
    });
    let tags_payload = serde_json::to_string(&tags_json).unwrap_or_else(|_| "{}".to_string());
    let _ = sqlx::query(
        "INSERT INTO event_ledger (event_id, timestamp, type, payload, tags, run_id, trace_id)
         VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&entry.id)
    .bind(&entry.timestamp)
    .bind(&event_type)
    .bind(&payload_json)
    .bind(&tags_payload)
    .bind(run_id)
    .bind(trace_id)
    .execute(pool)
    .await;

    Ok(entry)
}

pub async fn log_suppression_summary(
    pool: &SqlitePool,
    app_handle: Option<&AppHandle>,
    category: &str,
    run_id: Option<&str>,
    trace_id: Option<&str>,
    event: &str,
    stream: &str,
    counts: &HashMap<String, usize>,
    context: Option<Value>,
) -> Result<SystemLogEntry, String> {
    let mut payload = json!({
        "event": event,
        "stream": stream,
        "suppression_counts": counts,
    });
    if let Some(context) = context {
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("context".to_string(), context);
        }
    }
    log_event(pool, app_handle, "info", category, run_id, trace_id, payload).await
}

pub async fn log_contract_violation(
    pool: &SqlitePool,
    app_handle: Option<&AppHandle>,
    run_id: Option<&str>,
    trace_id: Option<&str>,
    policy_id: &str,
    reason: &str,
    context: Option<Value>,
) -> Result<SystemLogEntry, String> {
    let (allow_log, suppressed) = rate_limit_contract_violation(policy_id, reason);
    if !allow_log {
        return Ok(SystemLogEntry {
            id: Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            level: "warn".to_string(),
            category: "policy".to_string(),
            run_id: run_id.map(|s| s.to_string()),
            trace_id: trace_id.map(|s| s.to_string()),
            payload: json!({
                "event": "contract_violation_rate_limited",
                "policy_id": policy_id,
                "reason": reason,
            }),
        });
    }
    if suppressed > 0 {
        let _ = log_event(
            pool,
            app_handle,
            "info",
            "policy",
            run_id,
            trace_id,
            json!({
                "event": "contract_violation_suppressed",
                "policy_id": policy_id,
                "reason": reason,
                "count": suppressed,
                "window_secs": CONTRACT_VIOLATION_WINDOW_SECS,
            }),
        )
        .await;
    }
    let mut payload = json!({
        "event": "contract_violation",
        "policy_id": policy_id,
        "reason": reason,
    });
    if let Some(ctx) = context {
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("context".to_string(), ctx);
        }
    }
    log_event(pool, app_handle, "warn", "policy", run_id, trace_id, payload).await
}

pub async fn log_contract_violation_escalated(
    pool: &SqlitePool,
    app_handle: Option<&AppHandle>,
    run_id: Option<&str>,
    trace_id: Option<&str>,
    policy_id: &str,
    count: i64,
    context: Option<Value>,
) -> Result<SystemLogEntry, String> {
    let mut payload = json!({
        "event": "contract_violation_escalated",
        "policy_id": policy_id,
        "count": count,
        "severity": "warn",
    });
    if let Some(ctx) = context {
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("context".to_string(), ctx);
        }
    }
    log_event(pool, app_handle, "warn", "policy", run_id, trace_id, payload).await
}

pub async fn log_monologue_state_update(
    pool: &SqlitePool,
    app_handle: Option<&AppHandle>,
    run_id: Option<&str>,
    trace_id: Option<&str>,
    conversation_id: &str,
    candidate_id: &str,
    candidate_kind: &str,
    target_scope: Option<&str>,
    evidence_event_ids: &[i64],
    belief_ids: &[i64],
    summary: Option<&str>,
    before: Option<Value>,
    after: Option<Value>,
) -> Result<SystemLogEntry, String> {
    let mut payload = json!({
        "event": "monologue_state_update",
        "conversation_id": conversation_id,
        "candidate_id": candidate_id,
        "candidate_kind": candidate_kind,
        "target_scope": target_scope,
        "evidence_event_ids": evidence_event_ids,
        "belief_ids": belief_ids,
    });
    if let Some(summary) = summary {
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("summary".to_string(), json!(summary));
        }
    }
    if let Some(before) = before {
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("before".to_string(), before);
        }
    }
    if let Some(after) = after {
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("after".to_string(), after);
        }
    }
    log_event(pool, app_handle, "info", "kernel", run_id, trace_id, payload).await
}

pub async fn flush_logs(pool: &SqlitePool) -> Result<(), String> {
    sqlx::query("PRAGMA wal_checkpoint(TRUNCATE)")
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

async fn fetch_telemetry_mode(pool: &SqlitePool) -> String {
    let mode: Option<String> = sqlx::query_scalar(
        "SELECT mode FROM system_controls WHERE subsystem_id = ?",
    )
    .bind("telemetry_sampling")
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    mode.unwrap_or_else(|| {
        system_controls::default_mode_for("telemetry_sampling")
            .unwrap_or("normal")
            .to_string()
    })
}

fn telemetry_allows_event(mode: &str, level: &str, event: Option<&str>) -> bool {
    if level.eq_ignore_ascii_case("error") || level.eq_ignore_ascii_case("warn") {
        return true;
    }
    if let Some(event) = event {
        if TELEMETRY_ALWAYS_EVENTS.contains(event) {
            return true;
        }
    }
    if system_controls::mode_is_off(mode) {
        return false;
    }
    if system_controls::mode_is_degraded(mode) {
        return event
            .map(|name| TELEMETRY_ALWAYS_EVENTS.contains(name))
            .unwrap_or(false);
    }
    true
}
