use chrono::{DateTime, Utc};
use sqlx::{Row, SqlitePool};

use crate::core::memory::api::MemoryApi;
use crate::core::memory::types::{Scope, SourceType};
use crate::core::system_log;

#[derive(Debug, Clone)]
pub struct BridgeReport {
    pub processed: usize,
    pub skipped: usize,
    pub errors: usize,
}

#[derive(Debug, Clone)]
pub struct SelfMemoryParity {
    pub self_beliefs_active: i64,
    pub ics_self_beliefs_active: i64,
}

fn quote_dsl_value(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"");
    format!("\"{}\"", escaped)
}

fn parse_ts(raw: Option<String>) -> Option<DateTime<Utc>> {
    raw.and_then(|ts| DateTime::parse_from_rfc3339(&ts).ok())
        .map(|dt| dt.with_timezone(&Utc))
}

async fn resolve_bridge_conversation_id(pool: &SqlitePool) -> String {
    let conversation_id: Option<String> = sqlx::query_scalar(
        "SELECT conversation_id FROM conversations ORDER BY datetime(updated_at) DESC LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    conversation_id.unwrap_or_else(|| "default".to_string())
}

fn is_internal_state_key(input: &str) -> bool {
    let lowered = input.to_lowercase();
    lowered.starts_with("telemetry.")
        || lowered.contains("telemetry")
        || lowered.contains("controller")
        || lowered.contains("gate")
        || lowered.contains("self_state")
        || lowered.contains("self_model")
        || lowered.contains("runtime_state")
}

fn detect_identity_target(snippet: &str) -> Option<&'static str> {
    let lowered = snippet.to_lowercase();
    let user_patterns = [
        "i am ",
        "i'm ",
        "my name is",
        "call me",
        "i go by",
        "i work as",
    ];
    if user_patterns.iter().any(|p| lowered.contains(p)) {
        return Some("user");
    }
    let assistant_patterns = [
        "you are ",
        "you're ",
        "your name is",
        "you are called",
        "you should be called",
    ];
    if assistant_patterns.iter().any(|p| lowered.contains(p)) {
        return Some("assistant");
    }
    None
}

fn detect_capability_target(snippet: &str) -> Option<&'static str> {
    let lowered = snippet.to_lowercase();
    let user_patterns = [
        "i can ",
        "i can't",
        "i cannot",
        "i'm able to",
        "i am able to",
        "i'm unable to",
        "i am unable to",
    ];
    if user_patterns.iter().any(|p| lowered.contains(p)) {
        return Some("user");
    }
    let assistant_patterns = [
        "you can ",
        "you can't",
        "you cannot",
        "you're able to",
        "you are able to",
        "you are unable to",
        "you aren't able to",
    ];
    if assistant_patterns.iter().any(|p| lowered.contains(p)) {
        return Some("assistant");
    }
    None
}

async fn already_bridged(pool: &SqlitePool, source_ref: &str) -> bool {
    let existing: Option<i64> = sqlx::query_scalar(
        "SELECT id FROM ics_evidence_events WHERE source_ref = ? LIMIT 1",
    )
    .bind(source_ref)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    existing.is_some()
}

async fn update_bridged_event(pool: &SqlitePool, source_ref: &str, snippet: &str, weight: f32) {
    let _ = sqlx::query(
        "UPDATE ics_evidence_events SET snippet = ?, weight = ? WHERE source_ref = ?",
    )
    .bind(snippet)
    .bind(weight)
    .bind(source_ref)
    .execute(pool)
    .await;
}

async fn evidence_already_in_self(pool: &SqlitePool, evidence_id: i64) -> bool {
    let existing: Option<i64> = sqlx::query_scalar(
        "SELECT 1 FROM self_evidence_events
         WHERE source_evidence_ids IS NOT NULL
           AND EXISTS (
             SELECT 1 FROM json_each(self_evidence_events.source_evidence_ids)
             WHERE json_each.value = ?
           )
         LIMIT 1",
    )
    .bind(evidence_id)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    existing.is_some()
}

pub async fn bridge_self_event(pool: &SqlitePool, event_id: i64) -> Result<bool, String> {
    let row = sqlx::query(
        "SELECT se.id, se.snippet, se.weight, se.created_at, se.source_type, sb.kind, sb.observed_at, sb.id as belief_id,
                sf.key, sf.value_literal, sr.rel_type
         FROM self_evidence_events se
         JOIN self_beliefs sb ON sb.id = se.belief_id
         LEFT JOIN self_fact_beliefs sf ON sf.belief_id = sb.id
         LEFT JOIN self_rel_beliefs sr ON sr.belief_id = sb.id
         WHERE se.id = ?"
    )
    .bind(event_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    let Some(row) = row else {
        return Ok(false);
    };

    let snippet: String = row.get("snippet");
    let source_type: String = row.try_get("source_type").unwrap_or_else(|_| "system".to_string());
    if source_type.trim().eq_ignore_ascii_case("system_state")
        || source_type.trim().eq_ignore_ascii_case("telemetry")
    {
        return Ok(false);
    }
    let weight: f32 = row.try_get::<f64, _>("weight").unwrap_or(1.0) as f32;
    let kind: String = row.get("kind");
    let observed_at = row.try_get::<String, _>("observed_at").ok();
    let created_at = row.try_get::<String, _>("created_at").ok();
    let belief_id: i64 = row.get("belief_id");
    let source_ref = format!("self_memory_event:{}", event_id);

    if already_bridged(pool, &source_ref).await {
        return Ok(false);
    }

    let mut dsl_line = String::new();
    if kind == "fact" {
        let key: String = row.get("key");
        let value: String = row.get("value_literal");
        if is_internal_state_key(&key) {
            return Ok(false);
        }
        if key.trim().is_empty() || value.trim().is_empty() {
            return Ok(false);
        }
        dsl_line = format!("$assistant:{} = {}", key, quote_dsl_value(&value));
    } else if kind == "rel" {
        let rel_type: String = row.get("rel_type");
        if is_internal_state_key(&rel_type) {
            return Ok(false);
        }
        if rel_type.trim().is_empty() {
            return Ok(false);
        }
        let participants_rows = sqlx::query(
            "SELECT role, label FROM self_rel_participants WHERE belief_id = ? ORDER BY role, label",
        )
        .bind(belief_id)
        .fetch_all(pool)
        .await
        .map_err(|e| e.to_string())?;
        if participants_rows.is_empty() {
            return Ok(false);
        }
        let participants = participants_rows
            .iter()
            .map(|r| {
                let role: String = r.get("role");
                let label: String = r.get("label");
                format!("{}: {}", role, quote_dsl_value(&label))
            })
            .collect::<Vec<_>>()
            .join(", ");
        dsl_line = format!("{}({})", rel_type, participants);
    }

    if dsl_line.trim().is_empty() {
        return Ok(false);
    }

    let now = parse_ts(observed_at)
        .or_else(|| parse_ts(created_at))
        .unwrap_or_else(Utc::now);

    let session_id = resolve_bridge_conversation_id(pool).await;
    let api = MemoryApi::new(pool.clone(), None, session_id).await;
    let result = api
        .parse_and_compile(
            &dsl_line,
            Scope::SelfScope,
            SourceType::System,
            Some(source_ref.clone()),
            now,
        )
        .await;

    if !result.errors.is_empty() {
        return Err(result.errors.join("; "));
    }

    update_bridged_event(pool, &source_ref, &snippet, weight).await;
    Ok(true)
}

pub async fn bridge_pending_events(pool: &SqlitePool, limit: i64) -> Result<BridgeReport, String> {
    let cursor: Option<i64> = sqlx::query_scalar::<_, String>(
        "SELECT value FROM kv_store WHERE key = 'self_memory_bridge_cursor'",
    )
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()
    .and_then(|v| v.parse::<i64>().ok());

    let cursor = cursor.unwrap_or(0);
    let rows = sqlx::query(
        "SELECT id FROM self_evidence_events WHERE id > ? ORDER BY id ASC LIMIT ?",
    )
    .bind(cursor)
    .bind(limit.max(1))
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    let mut processed = 0usize;
    let mut skipped = 0usize;
    let mut errors = 0usize;
    let mut last_id = cursor;

    for row in rows {
        let event_id: i64 = row.get("id");
        last_id = event_id.max(last_id);
        match bridge_self_event(pool, event_id).await {
            Ok(true) => processed += 1,
            Ok(false) => skipped += 1,
            Err(_) => errors += 1,
        }
    }

    if last_id > cursor {
        let _ = sqlx::query(
            "INSERT INTO kv_store (key, value, updated_at)
             VALUES ('self_memory_bridge_cursor', ?, CURRENT_TIMESTAMP)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
        )
        .bind(last_id.to_string())
        .execute(pool)
        .await;
    }

    Ok(BridgeReport {
        processed,
        skipped,
        errors,
    })
}

pub async fn bridge_identity_evidence_events(
    pool: &SqlitePool,
    limit: i64,
) -> Result<BridgeReport, String> {
    let cursor: Option<i64> = sqlx::query_scalar::<_, String>(
        "SELECT value FROM kv_store WHERE key = 'self_memory_identity_cursor'",
    )
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()
    .and_then(|v| v.parse::<i64>().ok());

    let cursor = cursor.unwrap_or(0);
    let rows = sqlx::query(
        "SELECT id, source_type, snippet, created_at
         FROM ics_evidence_events
         WHERE id > ?
         ORDER BY id ASC
         LIMIT ?",
    )
    .bind(cursor)
    .bind(limit.max(1))
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    let mut processed = 0usize;
    let mut skipped = 0usize;
    let mut errors = 0usize;
    let mut last_id = cursor;

    for row in rows {
        let id: i64 = row.get("id");
        last_id = id.max(last_id);
        if evidence_already_in_self(pool, id).await {
            skipped += 1;
            continue;
        }
        let source_type: String = row.try_get("source_type").unwrap_or_default();
        if source_type != "user" && source_type != "system" {
            skipped += 1;
            continue;
        }
        let snippet_raw: Option<String> = row.try_get("snippet").ok();
        let snippet = snippet_raw.unwrap_or_default().trim().to_string();
        if snippet.is_empty() {
            skipped += 1;
            continue;
        }
        let created_at: Option<String> = row.try_get("created_at").ok();
        let observed_at = parse_ts(created_at).map(|dt| dt.with_timezone(&Utc));

        let identity_target = detect_identity_target(&snippet);
        let capability_target = if identity_target.is_none() {
            detect_capability_target(&snippet)
        } else {
            None
        };

        let (kind, target) = if let Some(target) = identity_target {
            ("identity_statement", target)
        } else if let Some(target) = capability_target {
            ("capability_statement", target)
        } else {
            skipped += 1;
            continue;
        };

        let key = format!("{}_{}", kind, target);
        let trimmed: String = snippet.chars().take(240).collect();
        let result = crate::core::self_memory::write_self_fact(
            pool,
            &key,
            &trimmed,
            &trimmed,
            observed_at,
            SourceType::System,
            Some(&[id]),
        )
        .await;
        match result {
            Ok(_) => processed += 1,
            Err(_) => errors += 1,
        }
    }

    if last_id > cursor {
        let _ = sqlx::query(
            "INSERT INTO kv_store (key, value, updated_at)
             VALUES ('self_memory_identity_cursor', ?, CURRENT_TIMESTAMP)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
        )
        .bind(last_id.to_string())
        .execute(pool)
        .await;
    }

    Ok(BridgeReport {
        processed,
        skipped,
        errors,
    })
}

pub async fn check_parity(pool: &SqlitePool) -> Result<SelfMemoryParity, String> {
    let self_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM self_beliefs WHERE status = 'active'",
    )
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?
    .unwrap_or(0);

    let scope = serde_json::to_string(&Scope::SelfScope).unwrap_or_else(|_| "\"self\"".to_string());
    let ics_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ics_beliefs WHERE status = 'active' AND scope = ?",
    )
    .bind(&scope)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?
    .unwrap_or(0);

    Ok(SelfMemoryParity {
        self_beliefs_active: self_count,
        ics_self_beliefs_active: ics_count,
    })
}

pub async fn log_parity(pool: &SqlitePool) {
    if let Ok(parity) = check_parity(pool).await {
        let _ = system_log::log_event(
            pool,
            None,
            "info",
            "memory",
            None,
            None,
            serde_json::json!({
                "event": "self_memory_parity",
                "self_beliefs_active": parity.self_beliefs_active,
                "ics_self_beliefs_active": parity.ics_self_beliefs_active,
            }),
        )
        .await;
    }
}
