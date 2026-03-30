use chrono::{DateTime, Utc};
use sqlx::{SqlitePool, Row};
use crate::core::memory::canonical::compute_value_hash;
use crate::core::memory::types::SourceType;
use crate::core::episodic;
use crate::core::sensitivity::{detect_sensitivity, phi_consent_allowed};
use crate::core::memory::retrieval;
use crate::core::kernel::KernelState;
use crate::core::system_controls;
use crate::core::system_log;
use crate::db::Db;
use serde_json::json;
use uuid::Uuid;
pub mod decay;
pub mod config;
pub mod bridge;
pub mod telemetry;

#[derive(Debug, Clone)]
pub struct SelfMemoryWriteResult {
    pub belief_id: i64,
    pub evidence_event_id: Option<i64>,
    pub episodic_event_id: Option<String>,
}

pub async fn compact_autobiographical(
    db: &Db,
    state: &KernelState,
    limit: i64,
) -> String {
    let recent = retrieval::render_autobiographical_context(
        &db.pool,
        Some(&state.conversation_id),
        limit,
    )
    .await;
    let stable = state
        .workspace_meta
        .runtime
        .as_ref()
        .and_then(|runtime| runtime.get("autobiographical_summary"))
        .and_then(|value| value.get("summary"))
        .and_then(|value| value.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("none"));
    match stable {
        Some(stable_summary) if stable_summary != recent => {
            if recent.trim().is_empty() {
                stable_summary
            } else {
                format!("Stable thread: {}\nRecent thread: {}", stable_summary, recent.trim())
            }
        }
        Some(stable_summary) => stable_summary,
        None => recent,
    }
}

async fn control_mode(pool: &SqlitePool, subsystem_id: &str) -> String {
    let mode: Option<String> = sqlx::query_scalar(
        "SELECT mode FROM system_controls WHERE subsystem_id = ?",
    )
    .bind(subsystem_id)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    mode.unwrap_or_else(|| system_controls::default_mode_for(subsystem_id).unwrap_or("normal").to_string())
}

async fn ensure_self_memory_write_allowed(pool: &SqlitePool) -> Result<(), String> {
    let self_mode = control_mode(pool, "self_memory").await;
    if system_controls::mode_is_off(&self_mode)
        || system_controls::mode_is_read_only(&self_mode)
        || system_controls::mode_is_degraded(&self_mode)
    {
        let _ = system_log::log_event(
            pool,
            None,
            "warn",
            "memory",
            None,
            None,
            json!({
                "event": "memory_write_blocked",
                "reason": "system_control_self_memory",
                "mode": self_mode,
            }),
        )
        .await;
        return Err("SelfMemoryWriteDisabled".to_string());
    }
    let memory_mode = control_mode(pool, "memory_write").await;
    if !system_controls::allow_memory_write(&memory_mode, "self_memory_write") {
        let _ = system_log::log_event(
            pool,
            None,
            "warn",
            "memory",
            None,
            None,
            json!({
                "event": "memory_write_blocked",
                "reason": "system_control",
                "mode": memory_mode,
            }),
        )
        .await;
        return Err("SelfMemoryWriteDisabled".to_string());
    }
    let recovered_logged: Option<String> = sqlx::query_scalar(
        "SELECT value FROM kv_store WHERE key = 'self_memory_recovered' LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    if recovered_logged.is_none() {
        let _ = system_log::log_event(
            pool,
            None,
            "info",
            "memory",
            None,
            None,
            json!({
                "event": "self_memory_recovered",
                "self_memory_mode": self_mode,
                "memory_write_mode": memory_mode,
            }),
        )
        .await;
        let _ = sqlx::query(
            "INSERT INTO kv_store (key, value, updated_at)
             VALUES ('self_memory_recovered', ?, CURRENT_TIMESTAMP)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
        )
        .bind(Utc::now().to_rfc3339())
        .execute(pool)
        .await;
    }
    Ok(())
}

async fn write_self_fact_core(
    pool: &SqlitePool,
    key: &str,
    value: &str,
    evidence_snippet: &str,
    observed_at: Option<DateTime<Utc>>,
    source: SourceType,
    source_evidence_ids: Option<&[i64]>,
) -> Result<SelfMemoryWriteResult, String> {
    ensure_self_memory_write_allowed(pool).await?;
    let snippet = evidence_snippet.trim();
    if snippet.is_empty() {
        return Err("EmptyEvidenceSnippet".to_string());
    }
    let sensitivity = detect_sensitivity(&format!("{} {}", key, value))
        .or_else(|| detect_sensitivity(snippet));
    if let Some(level) = sensitivity {
        if !phi_consent_allowed(pool, None).await {
            let _ = system_log::log_event(
                pool,
                None,
                "warn",
                "memory",
                None,
                None,
                json!({
                    "event": "phi_write_blocked",
                    "kind": "self_fact",
                    "sensitivity": level.as_str(),
                }),
            )
            .await;
            return Err("PhiBlocked".to_string());
        }
    }
    if source != SourceType::System {
        return Err("SelfMemorySourceMustBeSystem".to_string());
    }
    let mut source_ids: Vec<i64> = source_evidence_ids
        .unwrap_or(&[])
        .iter()
        .copied()
        .filter(|id| *id > 0)
        .collect();
    source_ids.sort();
    source_ids.dedup();
    if source_ids.is_empty() {
        return Err("MissingEvidenceIds".to_string());
    }
    let source_ids_json =
        serde_json::to_string(&source_ids).unwrap_or_else(|_| "[]".to_string());

    let now = observed_at.unwrap_or_else(Utc::now);
    let value_hash = compute_value_hash(value);

    let mut tx = pool.begin().await.map_err(|e| e.to_string())?;
    let row = sqlx::query(
        "INSERT INTO self_beliefs (kind, scope, status, confidence, evidence_weight_total, observed_at, last_evidence_at, created_at)
         VALUES ('fact', 'self', 'active', 1.0, 1.0, ?, ?, ?) RETURNING id"
    )
    .bind(now.to_rfc3339())
    .bind(now.to_rfc3339())
    .bind(now.to_rfc3339())
    .fetch_one(&mut *tx)
    .await
    .map_err(|e| e.to_string())?;
    let belief_id: i64 = row.get("id");

    sqlx::query(
        "INSERT INTO self_fact_beliefs (belief_id, key, value_literal, value_hash)
         VALUES (?, ?, ?, ?)"
    )
    .bind(belief_id)
    .bind(key)
    .bind(value)
    .bind(value_hash)
    .execute(&mut *tx)
    .await
    .map_err(|e| e.to_string())?;

    let episodic_on = episodic::episodic_enabled(pool).await;
    let episodic_event_id = if episodic_on { Some(Uuid::new_v4().to_string()) } else { None };
    let evidence_row = sqlx::query(
        "INSERT INTO self_evidence_events (belief_id, source_type, snippet, weight, source_evidence_ids, episodic_event_id, created_at)
         VALUES (?, 'system', ?, 1.0, ?, ?, ?) RETURNING id"
    )
    .bind(belief_id)
    .bind(snippet)
    .bind(&source_ids_json)
    .bind(episodic_event_id.as_deref())
    .bind(now.to_rfc3339())
    .fetch_one(&mut *tx)
    .await
    .map_err(|e| e.to_string())?;
    let evidence_event_id: i64 = evidence_row.get("id");

    tx.commit().await.map_err(|e| e.to_string())?;
    if let Some(event_id) = episodic_event_id.as_deref() {
        let _ = episodic::emit_episodic_event_with_id(
            pool,
            event_id,
            "memory_write_self_fact",
            json!({ "status": "inserted", "summary_snippet": snippet }),
            None,
            None,
            None,
            Some("self"),
            "system",
            None,
            Some(belief_id),
            None,
        )
        .await;
    }
    Ok(SelfMemoryWriteResult {
        belief_id,
        evidence_event_id: Some(evidence_event_id),
        episodic_event_id,
    })
}

pub async fn write_self_fact(
    pool: &SqlitePool,
    key: &str,
    value: &str,
    evidence_snippet: &str,
    observed_at: Option<DateTime<Utc>>,
    source: SourceType,
    source_evidence_ids: Option<&[i64]>,
) -> Result<SelfMemoryWriteResult, String> {
    let result = write_self_fact_core(
        pool,
        key,
        value,
        evidence_snippet,
        observed_at,
        source,
        source_evidence_ids,
    )
    .await?;
    if let Some(event_id) = result.evidence_event_id {
        let _ = bridge::bridge_self_event(pool, event_id).await;
    }
    Ok(result)
}

pub async fn write_self_fact_unbridged(
    pool: &SqlitePool,
    key: &str,
    value: &str,
    evidence_snippet: &str,
    observed_at: Option<DateTime<Utc>>,
    source: SourceType,
    source_evidence_ids: Option<&[i64]>,
) -> Result<SelfMemoryWriteResult, String> {
    write_self_fact_core(
        pool,
        key,
        value,
        evidence_snippet,
        observed_at,
        source,
        source_evidence_ids,
    )
    .await
}

pub async fn write_self_rel(
    pool: &SqlitePool,
    rel_type: &str,
    participants: &[(String, String)],
    evidence_snippet: &str,
    observed_at: Option<DateTime<Utc>>,
    source: SourceType,
    source_evidence_ids: Option<&[i64]>,
) -> Result<SelfMemoryWriteResult, String> {
    ensure_self_memory_write_allowed(pool).await?;
    let snippet = evidence_snippet.trim();
    if snippet.is_empty() {
        return Err("EmptyEvidenceSnippet".to_string());
    }
    let sensitivity = detect_sensitivity(snippet);
    if let Some(level) = sensitivity {
        if !phi_consent_allowed(pool, None).await {
            let _ = system_log::log_event(
                pool,
                None,
                "warn",
                "memory",
                None,
                None,
                json!({
                    "event": "phi_write_blocked",
                    "kind": "self_rel",
                    "sensitivity": level.as_str(),
                }),
            )
            .await;
            return Err("PhiBlocked".to_string());
        }
    }
    if source != SourceType::System {
        return Err("SelfMemorySourceMustBeSystem".to_string());
    }
    let mut source_ids: Vec<i64> = source_evidence_ids
        .unwrap_or(&[])
        .iter()
        .copied()
        .filter(|id| *id > 0)
        .collect();
    source_ids.sort();
    source_ids.dedup();
    if source_ids.is_empty() {
        return Err("MissingEvidenceIds".to_string());
    }
    let source_ids_json =
        serde_json::to_string(&source_ids).unwrap_or_else(|_| "[]".to_string());

    let now = observed_at.unwrap_or_else(Utc::now);
    let participants_canonical = canonicalize_participants(participants);

    let mut tx = pool.begin().await.map_err(|e| e.to_string())?;
    let row = sqlx::query(
        "INSERT INTO self_beliefs (kind, scope, status, confidence, evidence_weight_total, observed_at, last_evidence_at, created_at)
         VALUES ('rel', 'self', 'active', 1.0, 1.0, ?, ?, ?) RETURNING id"
    )
    .bind(now.to_rfc3339())
    .bind(now.to_rfc3339())
    .bind(now.to_rfc3339())
    .fetch_one(&mut *tx)
    .await
    .map_err(|e| e.to_string())?;
    let belief_id: i64 = row.get("id");

    sqlx::query(
        "INSERT INTO self_rel_beliefs (belief_id, rel_type, participants_canonical, anchor_signature)
         VALUES (?, ?, ?, ?)"
    )
    .bind(belief_id)
    .bind(rel_type)
    .bind(&participants_canonical)
    .bind(&participants_canonical)
    .execute(&mut *tx)
    .await
    .map_err(|e| e.to_string())?;

    for (role, label) in participants {
        sqlx::query(
            "INSERT INTO self_rel_participants (belief_id, role, label)
             VALUES (?, ?, ?)"
        )
        .bind(belief_id)
        .bind(role)
        .bind(label)
        .execute(&mut *tx)
        .await
        .map_err(|e| e.to_string())?;
    }

    let episodic_on = episodic::episodic_enabled(pool).await;
    let episodic_event_id = if episodic_on { Some(Uuid::new_v4().to_string()) } else { None };
    let evidence_row = sqlx::query(
        "INSERT INTO self_evidence_events (belief_id, source_type, snippet, weight, source_evidence_ids, episodic_event_id, created_at)
         VALUES (?, 'system', ?, 1.0, ?, ?, ?) RETURNING id"
    )
    .bind(belief_id)
    .bind(snippet)
    .bind(&source_ids_json)
    .bind(episodic_event_id.as_deref())
    .bind(now.to_rfc3339())
    .fetch_one(&mut *tx)
    .await
    .map_err(|e| e.to_string())?;
    let evidence_event_id: i64 = evidence_row.get("id");

    tx.commit().await.map_err(|e| e.to_string())?;
    if let Some(event_id) = episodic_event_id.as_deref() {
        let _ = episodic::emit_episodic_event_with_id(
            pool,
            event_id,
            "memory_write_self_rel",
            json!({ "status": "inserted", "summary_snippet": snippet }),
            None,
            None,
            None,
            Some("self"),
            "system",
            None,
            Some(belief_id),
            None,
        )
        .await;
    }
    let _ = bridge::bridge_self_event(pool, evidence_event_id).await;
    Ok(SelfMemoryWriteResult {
        belief_id,
        evidence_event_id: Some(evidence_event_id),
        episodic_event_id,
    })
}

fn canonicalize_participants(participants: &[(String, String)]) -> String {
    let mut items = participants.to_vec();
    items.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    items
        .iter()
        .map(|(role, label)| format!("{}:{}", role, label))
        .collect::<Vec<_>>()
        .join("|")
}
