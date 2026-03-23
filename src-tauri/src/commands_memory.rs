use tauri::State;
use chrono::Utc;
use sqlx::Row;
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use std::sync::Arc;
use crate::db::Db;
use crate::core::memory::api::MemoryApi;
use crate::core::memory::claims;
use crate::core::memory::cache;
use crate::core::memory::types::{ClaimOutcome, Scope, SourceType, MemoryPacket};
use crate::core::model_client::ModelClient;
use crate::core::episodic;
use crate::core::memory::debug::MemoryDebugLog;
use crate::core::system_log;
use crate::core::system_controls;
use crate::models::SelfInspection;
use crate::models::SelfModel;
use crate::models::EpisodicEvent;
use crate::models::{MemoryClaim, PolicyVersion, StrategyTrace, ReflectionStagingEntry};
use serde_json::json;
use uuid::Uuid;
use sqlx::QueryBuilder;

async fn gate_allows_memory_write(pool: &SqlitePool, conversation_id: &str, action: &str) -> Result<(), String> {
    let cockpit_write_enabled: Option<i32> = sqlx::query_scalar(
        "SELECT cockpit_write_enabled FROM settings LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    if cockpit_write_enabled.unwrap_or(0) == 0 {
        let _ = system_log::log_event(
            pool,
            None,
            "warn",
            "memory",
            None,
            None,
            json!({
                "event": "memory_write_blocked",
                "reason": "memory_write_control",
                "action": action,
                "conversation_id": conversation_id,
            }),
        )
        .await;
        return Err(format!("{} blocked by cockpit_write_enabled", action));
    }

    let snapshot_hash: Option<String> = sqlx::query_scalar(
        "SELECT snapshot_hash FROM subject_snapshots
         WHERE conversation_id = ?
         ORDER BY datetime(timestamp) DESC
         LIMIT 1",
    )
    .bind(conversation_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?
    .flatten();
    let gate_decision: Option<String> = if let Some(hash) = snapshot_hash.as_deref() {
        sqlx::query_scalar(
            "SELECT decision FROM gate_decisions
             WHERE snapshot_hash = ?
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .bind(hash)
        .fetch_optional(pool)
        .await
        .map_err(|e| e.to_string())?
        .flatten()
    } else {
        None
    };
    if !matches!(gate_decision.as_deref(), Some("ALLOW")) {
        let _ = system_log::log_event(
            pool,
            None,
            "warn",
            "memory",
            None,
            None,
            json!({
                "event": "memory_write_blocked",
                "reason": "gate_decision",
                "action": action,
                "gate_decision": gate_decision,
                "snapshot_hash": snapshot_hash,
                "conversation_id": conversation_id,
            }),
        )
        .await;
        return Err(format!("{} blocked by GateDecision", action));
    }

    let mode: Option<String> = sqlx::query_scalar(
        "SELECT mode FROM system_controls WHERE subsystem_id = ?",
    )
    .bind("memory_write")
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    let memory_mode = mode.unwrap_or_else(|| {
        system_controls::default_mode_for("memory_write")
            .unwrap_or("normal")
            .to_string()
    });
    if system_controls::mode_is_off(&memory_mode) || system_controls::mode_is_read_only(&memory_mode) {
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
                "action": action,
                "mode": memory_mode,
                "conversation_id": conversation_id,
            }),
        )
        .await;
        return Err(format!("{} blocked by system_controls", action));
    }
    if system_controls::mode_is_degraded(&memory_mode)
        && !system_controls::allow_memory_write(&memory_mode, action)
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
                "reason": "system_control_degraded",
                "action": action,
                "mode": memory_mode,
                "conversation_id": conversation_id,
            }),
        )
        .await;
        return Err(format!("{} blocked by system_controls (degraded)", action));
    }
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct MemoryHealthCheck {
    pub episodic_enabled: bool,
    pub episodic_injection_enabled: bool,
    pub episodic_compaction_enabled: bool,
    pub embedding_model: Option<String>,
    pub embedding_ready: bool,
    pub memory_claims_enabled: bool,
}

#[tauri::command]
pub async fn memory_health_check(
    db: State<'_, Arc<Db>>,
) -> Result<MemoryHealthCheck, String> {
    let settings = db.get_settings().await.map_err(|e| e.to_string())?;
    let embedding_model = settings
        .embedding_model
        .clone()
        .map(|m| m.trim().to_string())
        .filter(|m| !m.is_empty());
    let embedding_ready = embedding_model.is_some() && !settings.api_base_url.trim().is_empty();
    Ok(MemoryHealthCheck {
        episodic_enabled: settings.episodic_enabled.unwrap_or(false),
        episodic_injection_enabled: settings.episodic_injection_enabled.unwrap_or(false),
        episodic_compaction_enabled: settings.episodic_compaction_enabled.unwrap_or(false),
        embedding_model,
        embedding_ready,
        memory_claims_enabled: settings.memory_claims_enabled.unwrap_or(true),
    })
}


#[tauri::command]
pub async fn memory_write(
    pool: State<'_, SqlitePool>,
    model_client: State<'_, Arc<ModelClient>>,
    input: String,
    source: Option<String>, // "user", "tool", "system"
    evidence_event_ids: Option<Vec<i64>>,
    admin_override: Option<bool>,
) -> Result<crate::core::memory::compiler::CompileResult, String> {
    gate_allows_memory_write(&pool, "default", "memory_write").await?;
    let evidence_ids = normalize_evidence_ids(evidence_event_ids.as_deref());
    let admin_override = admin_override.unwrap_or(false);
    let manual_untrusted = admin_override && evidence_ids.is_empty();
    if memory_write_requires_evidence(&evidence_ids, admin_override) {
        let _ = system_log::log_event(
            &pool,
            None,
            "warn",
            "memory",
            None,
            None,
            json!({
                "event": "memory_write_blocked",
                "reason": "missing_evidence",
            }),
        )
        .await;
        return Err("memory_write requires evidence_event_ids or admin_override".to_string());
    }
    if !evidence_ids.is_empty() && !evidence_ids_exist(&pool, &evidence_ids).await {
        let _ = system_log::log_event(
            &pool,
            None,
            "warn",
            "memory",
            None,
            None,
            json!({
                "event": "memory_write_blocked",
                "reason": "invalid_evidence",
                "evidence_event_ids": evidence_ids,
            }),
        )
        .await;
        return Err("memory_write evidence_event_ids were not found".to_string());
    }
    if admin_override {
        let _ = system_log::log_event(
            &pool,
            None,
            "info",
            "memory",
            None,
            None,
            json!({
                "event": "memory_write_admin_override",
            }),
        )
        .await;
    }
    if manual_untrusted {
        let _ = system_log::log_event(
            &pool,
            None,
            "warn",
            "memory",
            None,
            None,
            json!({
                "event": "memory_write_untrusted",
                "reason": "admin_override_no_evidence",
            }),
        )
        .await;
    }
    let api = MemoryApi::new((*pool).clone(), Some((*model_client).clone()), "default".to_string()).await; // TODO: Session handling
    
    let source_type = resolve_memory_write_source_type(source.as_deref(), admin_override);
    
    // Default scope global for now or pass in?
    let now = chrono::Utc::now();
    let source_ref = if !evidence_ids.is_empty() {
        Some(format!(
            "evidence_event_ids:{}",
            evidence_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ))
    } else if manual_untrusted {
        Some("manual_untrusted".to_string())
    } else if admin_override {
        Some("admin_override".to_string())
    } else {
        None
    };
    let result = api
        .parse_and_compile(&input, Scope::Global, source_type, source_ref, now)
        .await;
    if !result.errors.is_empty() {
        let now = chrono::Utc::now().to_rfc3339();
        let error_text = result.errors.join(" | ");
        let _ = sqlx::query("INSERT INTO kv_store (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP")
            .bind("memory_last_error_at")
            .bind(&now)
            .execute(&*pool)
            .await;
        let _ = sqlx::query("INSERT INTO kv_store (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP")
            .bind("memory_last_error")
            .bind(&error_text)
            .execute(&*pool)
            .await;
    }
    Ok(result)
}

fn normalize_evidence_ids(ids: Option<&[i64]>) -> Vec<i64> {
    let mut out: Vec<i64> = ids
        .unwrap_or(&[])
        .iter()
        .copied()
        .filter(|id| *id > 0)
        .collect();
    out.sort();
    out.dedup();
    out
}

fn memory_write_requires_evidence(evidence_ids: &[i64], admin_override: bool) -> bool {
    evidence_ids.is_empty() && !admin_override
}

fn resolve_memory_write_source_type(source: Option<&str>, admin_override: bool) -> SourceType {
    if admin_override {
        return SourceType::System;
    }
    match source {
        Some("user") => SourceType::User,
        Some("tool") => SourceType::Tool,
        Some("system") => SourceType::System,
        _ => SourceType::User,
    }
}

async fn evidence_ids_exist(pool: &SqlitePool, ids: &[i64]) -> bool {
    if ids.is_empty() {
        return false;
    }
    let mut builder = QueryBuilder::new("SELECT COUNT(*) as count FROM ics_evidence_events WHERE id IN (");
    let mut separated = builder.separated(", ");
    for id in ids.iter() {
        separated.push_bind(id);
    }
    separated.push_unseparated(")");
    let query = builder.build_query_scalar::<i64>();
    let count = query.fetch_one(pool).await.unwrap_or(0);
    count >= ids.len() as i64
}

#[cfg(test)]
mod tests {
    use super::{normalize_evidence_ids, memory_write_requires_evidence, resolve_memory_write_source_type, evidence_ids_exist};
    use crate::core::memory::types::SourceType;
    use sqlx::sqlite::SqlitePoolOptions;

    #[test]
    fn normalize_evidence_ids_dedups_and_filters() {
        let ids = vec![3, 1, 0, -4, 3, 2];
        let normalized = normalize_evidence_ids(Some(ids.as_slice()));
        assert_eq!(normalized, vec![1, 2, 3]);
    }

    #[test]
    fn memory_write_gate_requires_evidence_or_admin() {
        assert!(memory_write_requires_evidence(&[], false));
        assert!(!memory_write_requires_evidence(&[1], false));
        assert!(!memory_write_requires_evidence(&[], true));
    }

    #[test]
    fn memory_write_admin_override_forces_system_source() {
        assert!(matches!(resolve_memory_write_source_type(Some("user"), true), SourceType::System));
        assert!(matches!(resolve_memory_write_source_type(None, true), SourceType::System));
        assert!(matches!(resolve_memory_write_source_type(Some("tool"), false), SourceType::Tool));
    }

    #[tokio::test]
    async fn evidence_ids_exist_detects_missing() {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("pool");
        sqlx::query("CREATE TABLE ics_evidence_events (id INTEGER PRIMARY KEY)")
            .execute(&pool)
            .await
            .expect("create table");
        sqlx::query("INSERT INTO ics_evidence_events (id) VALUES (1), (2)")
            .execute(&pool)
            .await
            .expect("seed");
        assert!(evidence_ids_exist(&pool, &[1, 2]).await);
        assert!(!evidence_ids_exist(&pool, &[1, 3]).await);
    }
}

#[tauri::command]
pub async fn memory_retrieve(
    pool: State<'_, SqlitePool>,
    model_client: State<'_, Arc<ModelClient>>,
    query: String,
) -> Result<MemoryPacket, String> {
    let api = MemoryApi::new((*pool).clone(), Some((*model_client).clone()), "default".to_string()).await;
    // Intent default to AskCurrent
    api.retrieve(&query, &[Scope::Global], crate::core::memory::api::infer_query_intent(&query)).await
}

#[derive(Debug, Serialize)]
pub struct MemoryRetrievalDebugResponse {
    pub summary: String,
    pub log: MemoryDebugLog,
}

#[tauri::command]
pub async fn memory_retrieval_debug(
    pool: State<'_, SqlitePool>,
    model_client: State<'_, Arc<ModelClient>>,
    query: String,
) -> Result<MemoryRetrievalDebugResponse, String> {
    let api = MemoryApi::new((*pool).clone(), Some((*model_client).clone()), "default".to_string()).await;
    let packet = api
        .retrieve_with_debug(&query, &[Scope::Global], crate::core::memory::api::infer_query_intent(&query))
        .await?;

    let mut log = MemoryDebugLog::new();
    if let Some(ret) = packet.debug_log.clone() {
        log = log.with_retrieval(ret);
    }
    let summary = log.format_summary();

    if let Ok(log_json) = serde_json::to_string(&log) {
        let _ = sqlx::query(
            "INSERT INTO kv_store (key, value, updated_at)
             VALUES (?, ?, CURRENT_TIMESTAMP)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP"
        )
        .bind("last_memory_debug")
        .bind(&log_json)
        .execute(&*pool)
        .await;
        let _ = sqlx::query(
            "INSERT INTO kv_store (key, value, updated_at)
             VALUES (?, ?, CURRENT_TIMESTAMP)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP"
        )
        .bind("memory_last_retrieval_debug")
        .bind(log_json)
        .execute(&*pool)
        .await;
    }

    Ok(MemoryRetrievalDebugResponse { summary, log })
}

#[tauri::command]
pub async fn memory_get_last_debug(
    pool: State<'_, SqlitePool>,
) -> Result<Option<MemoryRetrievalDebugResponse>, String> {
    let raw: Option<String> = sqlx::query_scalar("SELECT value FROM kv_store WHERE key = ?")
        .bind("last_memory_debug")
        .fetch_optional(&*pool)
        .await
        .map_err(|e| e.to_string())?
        .flatten();

    let Some(raw) = raw else {
        return Ok(None);
    };

    let log: MemoryDebugLog = serde_json::from_str(&raw).unwrap_or_default();
    let summary = log.format_summary();
    Ok(Some(MemoryRetrievalDebugResponse { summary, log }))
}

#[tauri::command]
pub async fn memory_get_scopes(
    conversation_id: Option<String>,
) -> Result<Vec<String>, String> {
    let scopes = crate::core::memory::scope::scopes_for_conversation(conversation_id.as_deref());
    let formatted = scopes
        .into_iter()
        .map(|scope| match scope {
            Scope::Global => "global".to_string(),
            Scope::Session => "session".to_string(),
            Scope::Project(id) => format!("project:{}", id),
            Scope::Context(id) => format!("context:{}", id),
            Scope::SelfScope => "self".to_string(),
        })
        .collect::<Vec<_>>();
    Ok(formatted)
}

#[tauri::command]
pub async fn memory_consolidate(
    pool: State<'_, SqlitePool>,
) -> Result<crate::core::memory::api::ConsolidationResult, String> {
    gate_allows_memory_write(&pool, "default", "memory_consolidate").await?;
    // Consolidation might not strictly need model client yet, but good to have
    let api = MemoryApi::new((*pool).clone(), None, "default".to_string()).await;
    api.consolidate().await
}

#[tauri::command]
pub async fn memory_resolve_conflict(
    pool: State<'_, SqlitePool>,
    conflict_id: i64,
    action: String,  // "resolve" | "archive" | "pick_winner" | "keep_both"
    winner_belief_id: Option<i64>,
    resolution_note: Option<String>,
    user_reply: Option<String>,
) -> Result<bool, String> {
    gate_allows_memory_write(&pool, "default", "memory_resolve_conflict").await?;
    use sqlx::Row;
    
    let reply_snippet = user_reply.as_deref().unwrap_or("").trim();
    match action.as_str() {
        "archive" => {
            sqlx::query("UPDATE ics_conflict_sets SET status = 'archived', updated_at = CURRENT_TIMESTAMP WHERE id = ?")
                .bind(conflict_id)
                .execute(&*pool)
                .await
                .map_err(|e| e.to_string())?;
        },
        "resolve" => {
            sqlx::query("UPDATE ics_conflict_sets SET status = 'resolved', resolution_note = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?")
                .bind(&resolution_note)
                .bind(conflict_id)
                .execute(&*pool)
                .await
                .map_err(|e| e.to_string())?;
        },
        "pick_winner" => {
            let winner_id = winner_belief_id.ok_or("winner_belief_id required for pick_winner")?;
            
            // Deactivate all other beliefs in conflict
            let member_ids: Vec<i64> = sqlx::query(
                "SELECT belief_id FROM ics_conflict_set_members WHERE conflict_set_id = ? AND belief_id != ?"
            )
            .bind(conflict_id)
            .bind(winner_id)
            .fetch_all(&*pool)
            .await
            .map_err(|e| e.to_string())?
            .iter()
            .map(|row| row.get::<i64, _>("belief_id"))
            .collect();
            
            for id in member_ids {
                sqlx::query("UPDATE ics_beliefs SET status = 'inactive' WHERE id = ?")
                    .bind(id)
                    .execute(&*pool)
                    .await
                    .map_err(|e| e.to_string())?;
                    
                // Create supersedes link
                sqlx::query("INSERT OR IGNORE INTO ics_belief_links (from_id, to_id, link_type) VALUES (?, ?, 'supersedes')")
                    .bind(winner_id)
                    .bind(id)
                    .execute(&*pool)
                    .await
                    .map_err(|e| e.to_string())?;
            }
            
            if !reply_snippet.is_empty() {
                let episodic_on = episodic::episodic_enabled(&*pool).await;
                let episodic_event_id = if episodic_on { Some(Uuid::new_v4().to_string()) } else { None };
                let _ = sqlx::query(
                    "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id) VALUES (?, 'user', 'conflict_resolution', ?, 1.0, ?)"
                )
                .bind(winner_id)
                .bind(reply_snippet)
                .bind(episodic_event_id.as_deref())
                .execute(&*pool)
                .await;

                if let Some(event_id) = episodic_event_id {
                    let _ = episodic::emit_episodic_event_with_id(
                        &*pool,
                        &event_id,
                        "memory_conflict_resolution",
                        json!({ "status": "inserted", "summary_snippet": reply_snippet }),
                        None,
                        None,
                        Some("default"),
                        None,
                        "user",
                        Some("conflict_resolution"),
                        Some(winner_id),
                        None,
                    )
                    .await;
                }
            }

            sqlx::query("UPDATE ics_conflict_sets SET status = 'resolved', resolution_note = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?")
                .bind(&resolution_note)
                .bind(conflict_id)
                .execute(&*pool)
                .await
                .map_err(|e| e.to_string())?;
        },
        "keep_both" => {
            let mut should_resolve = true;
            let member_rows = sqlx::query(
                "SELECT fb.key
                 FROM ics_conflict_set_members csm
                 JOIN ics_fact_beliefs fb ON fb.belief_id = csm.belief_id
                 WHERE csm.conflict_set_id = ?"
            )
            .bind(conflict_id)
            .fetch_all(&*pool)
            .await
            .map_err(|e| e.to_string())?;

            for row in member_rows {
                let key: String = row.get("key");
                if let Ok(Some(cardinality)) = sqlx::query_scalar::<_, String>(
                    "SELECT cardinality FROM ics_token_policies WHERE token = ?"
                )
                .bind(key)
                .fetch_optional(&*pool)
                .await {
                    if cardinality.eq_ignore_ascii_case("one") {
                        should_resolve = false;
                        break;
                    }
                }
            }

            let status = if should_resolve { "resolved" } else { "open" };
            sqlx::query("UPDATE ics_conflict_sets SET status = ?, resolution_note = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?")
                .bind(status)
                .bind(&resolution_note)
                .bind(conflict_id)
                .execute(&*pool)
                .await
                .map_err(|e| e.to_string())?;

            if !reply_snippet.is_empty() {
                let belief_rows = sqlx::query(
                    "SELECT belief_id FROM ics_conflict_set_members WHERE conflict_set_id = ?"
                )
                .bind(conflict_id)
                .fetch_all(&*pool)
                .await
                .map_err(|e| e.to_string())?;

                for row in belief_rows {
                    let belief_id: i64 = row.get("belief_id");
                    let episodic_on = episodic::episodic_enabled(&*pool).await;
                    let episodic_event_id = if episodic_on { Some(Uuid::new_v4().to_string()) } else { None };
                    let _ = sqlx::query(
                        "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id) VALUES (?, 'user', 'conflict_resolution', ?, 1.0, ?)"
                    )
                    .bind(belief_id)
                    .bind(reply_snippet)
                    .bind(episodic_event_id.as_deref())
                    .execute(&*pool)
                    .await;

                    if let Some(event_id) = episodic_event_id {
                        let _ = episodic::emit_episodic_event_with_id(
                            &*pool,
                            &event_id,
                            "memory_conflict_resolution",
                            json!({ "status": "inserted", "summary_snippet": reply_snippet }),
                            None,
                            None,
                            Some("default"),
                            None,
                            "user",
                            Some("conflict_resolution"),
                            Some(belief_id),
                            None,
                        )
                        .await;
                    }
                }
            }
        },
        _ => return Err("Invalid action".to_string()),
    }
    
    cache::bump_cache_version();
    Ok(true)
}

#[tauri::command]
pub async fn memory_resolve_clarify(
    pool: State<'_, SqlitePool>,
    model_client: State<'_, Arc<ModelClient>>,
    pending_id: i64,
    reply: String,
    source: Option<String>,
) -> Result<crate::core::memory::clarify::ClarifyResult, String> {
    let api = MemoryApi::new((*pool).clone(), Some((*model_client).clone()), "default".to_string()).await;
    let source_type = match source.as_deref() {
        Some("tool") => SourceType::Tool,
        Some("system") => SourceType::System,
        Some("inference") => SourceType::Inference,
        _ => SourceType::User,
    };

    let result = crate::core::memory::clarify::resolve_clarify(
        pending_id,
        reply.trim(),
        &*pool,
        Some((*model_client).clone()),
        Scope::Global,
        source_type,
        api.embedding_config().cloned(),
    )
    .await;
    Ok(result)
}

#[derive(Debug, Serialize)]
pub struct ConflictMemberView {
    pub belief_id: i64,
    pub kind: String,
    pub polarity: String,
    pub confidence: f64,
    pub status: String,
    pub preview: String,
    pub observed_at: Option<String>,
    pub evidence_snippet: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ConflictView {
    pub id: i64,
    pub topic_key: String,
    pub status: String,
    pub priority: String,
    pub resolution_note: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub members: Vec<ConflictMemberView>,
}

#[derive(Debug, Deserialize)]
pub struct BeliefProvenanceArgs {
    #[serde(alias = "beliefId")]
    pub belief_id: i64,
    pub kind: String,
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct EntityProvenanceArgs {
    #[serde(alias = "entityId")]
    pub entity_id: i64,
    pub limit: Option<i64>,
}

#[tauri::command]
pub async fn memory_list_conflicts(
    pool: State<'_, SqlitePool>,
) -> Result<Vec<ConflictView>, String> {
    use sqlx::Row;

    let conflict_rows = sqlx::query(
        "SELECT id, topic_key, status, priority, resolution_note, created_at, updated_at
         FROM ics_conflict_sets
         WHERE status = 'open'
         ORDER BY updated_at DESC"
    )
    .fetch_all(&*pool)
    .await
    .map_err(|e| e.to_string())?;

    let mut results = Vec::new();
    for row in conflict_rows {
        let conflict_id: i64 = row.get("id");
        let member_rows = sqlx::query(
            "SELECT b.id as belief_id, b.kind, b.polarity, b.confidence, b.status,
                    b.observed_at,
                    fb.key, fb.value_literal, rb.rel_type
             FROM ics_conflict_set_members csm
             JOIN ics_beliefs b ON b.id = csm.belief_id
             LEFT JOIN ics_fact_beliefs fb ON fb.belief_id = b.id
             LEFT JOIN ics_rel_beliefs rb ON rb.belief_id = b.id
             WHERE csm.conflict_set_id = ?"
        )
        .bind(conflict_id)
        .fetch_all(&*pool)
        .await
        .map_err(|e| e.to_string())?;

        let mut members = Vec::new();
        for mrow in member_rows {
            let kind: String = mrow.get("kind");
            let belief_id: i64 = mrow.get("belief_id");
            let preview = if kind == "fact" {
                let key: Option<String> = mrow.try_get("key").ok();
                let value: Option<String> = mrow.try_get("value_literal").ok();
                match (key, value) {
                    (Some(k), Some(v)) => format!("{} = {}", k, v),
                    _ => "fact".to_string(),
                }
            } else {
                let rel_type: Option<String> = mrow.try_get("rel_type").ok();
                let rel_type = rel_type.unwrap_or_else(|| "relation".to_string());
                let parts = sqlx::query(
                    "SELECT rp.role, e.label
                     FROM ics_rel_participants rp
                     JOIN ics_entities e ON e.id = rp.entity_id
                     WHERE rp.belief_id = ?
                     ORDER BY rp.role"
                )
                .bind(belief_id)
                .fetch_all(&*pool)
                .await
                .unwrap_or_default();

                if parts.is_empty() {
                    rel_type
                } else {
                    let detail = parts
                        .iter()
                        .map(|row| {
                            let role: String = row.get("role");
                            let label: String = row.get("label");
                            format!("{}: {}", role, label)
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("{}({})", rel_type, detail)
                }
            };

            let evidence_row = sqlx::query(
                "SELECT snippet, created_at FROM ics_evidence_events
                 WHERE belief_id = ?
                 ORDER BY created_at DESC
                 LIMIT 1"
            )
            .bind(belief_id)
            .fetch_optional(&*pool)
            .await
            .unwrap_or(None);

            let evidence_snippet = if let Some(row) = evidence_row {
                let snippet: Option<String> = row.try_get("snippet").ok();
                snippet
            } else {
                None
            };

            members.push(ConflictMemberView {
                belief_id,
                kind,
                polarity: mrow.get("polarity"),
                confidence: mrow.get("confidence"),
                status: mrow.get("status"),
                preview,
                observed_at: mrow.try_get("observed_at").ok(),
                evidence_snippet,
            });
        }

        results.push(ConflictView {
            id: conflict_id,
            topic_key: row.get("topic_key"),
            status: row.get("status"),
            priority: row.get("priority"),
            resolution_note: row.get("resolution_note"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
            members,
        });
    }

    Ok(results)
}

#[derive(Debug, Serialize)]
pub struct SelfChangeView {
    pub belief_id: i64,
    pub snippet: String,
    pub observed_at: String,
}

#[tauri::command]
pub async fn self_memory_list_changes(
    pool: State<'_, SqlitePool>,
    limit: Option<i64>,
) -> Result<Vec<SelfChangeView>, String> {
    let limit = limit.unwrap_or(50).max(1).min(200);
    let rows = sqlx::query(
        "SELECT belief_id, snippet, created_at
         FROM self_evidence_events
         ORDER BY created_at DESC
         LIMIT ?"
    )
    .bind(limit)
    .fetch_all(&*pool)
    .await
    .map_err(|e| e.to_string())?;

    let mut results = Vec::new();
    for row in rows {
        results.push(SelfChangeView {
            belief_id: row.get("belief_id"),
            snippet: row.get::<String, _>("snippet"),
            observed_at: row.get::<String, _>("created_at"),
        });
    }
    Ok(results)
}

#[tauri::command]
pub async fn self_memory_rollback(
    pool: State<'_, SqlitePool>,
    belief_id: i64,
) -> Result<bool, String> {
    gate_allows_memory_write(&pool, "default", "self_memory_rollback").await?;
    sqlx::query("DELETE FROM self_evidence_events WHERE belief_id = ?")
        .bind(belief_id)
        .execute(&*pool)
        .await
        .map_err(|e| e.to_string())?;
    sqlx::query("DELETE FROM self_rel_participants WHERE belief_id = ?")
        .bind(belief_id)
        .execute(&*pool)
        .await
        .map_err(|e| e.to_string())?;
    sqlx::query("DELETE FROM self_rel_beliefs WHERE belief_id = ?")
        .bind(belief_id)
        .execute(&*pool)
        .await
        .map_err(|e| e.to_string())?;
    sqlx::query("DELETE FROM self_fact_beliefs WHERE belief_id = ?")
        .bind(belief_id)
        .execute(&*pool)
        .await
        .map_err(|e| e.to_string())?;
    sqlx::query("DELETE FROM self_beliefs WHERE id = ?")
        .bind(belief_id)
        .execute(&*pool)
        .await
        .map_err(|e| e.to_string())?;
    Ok(true)
}

#[tauri::command]
pub async fn self_inspect(
    pool: State<'_, SqlitePool>,
) -> Result<SelfInspection, String> {
    crate::db::get_self_inspection(&*pool).await
}

#[tauri::command]
pub async fn set_reflection_frozen(
    pool: State<'_, SqlitePool>,
    frozen: bool,
) -> Result<bool, String> {
    sqlx::query("UPDATE self_model SET reflection_frozen = ? WHERE id = 1")
        .bind(if frozen { 1 } else { 0 })
        .execute(&*pool)
        .await
        .map_err(|e| e.to_string())?;
    Ok(true)
}

#[tauri::command]
pub async fn list_reflection_staging(
    db: State<'_, Arc<Db>>,
    status: Option<String>,
    limit: Option<i64>,
) -> Result<Vec<ReflectionStagingEntry>, String> {
    let limit = limit.unwrap_or(20).max(1);
    db.list_reflection_staging(status.as_deref(), limit)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn approve_reflection_staging(
    db: State<'_, Arc<Db>>,
    stage_id: String,
    reviewer: Option<String>,
) -> Result<serde_json::Value, String> {
    db.update_reflection_staging_status(&stage_id, "approved", reviewer.as_deref())
        .await
        .map_err(|e| e.to_string())?;
    crate::core::self_reflection::apply_reflection_staged(
        &db,
        &stage_id,
        reviewer.as_deref(),
    )
    .await
}

#[tauri::command]
pub async fn reject_reflection_staging(
    db: State<'_, Arc<Db>>,
    stage_id: String,
    reviewer: Option<String>,
) -> Result<(), String> {
    db.update_reflection_staging_status(&stage_id, "rejected", reviewer.as_deref())
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn self_model_rollback(
    pool: State<'_, SqlitePool>,
) -> Result<bool, String> {
    gate_allows_memory_write(&pool, "default", "self_model_rollback").await?;
    use sqlx::Row;
    let row = sqlx::query(
        "SELECT snapshot_json FROM self_model_checkpoints ORDER BY id DESC LIMIT 1"
    )
    .fetch_optional(&*pool)
    .await
    .map_err(|e| e.to_string())?;

    let Some(row) = row else {
        return Ok(false);
    };

    let snapshot: String = row.get("snapshot_json");
    let parsed: serde_json::Value = serde_json::from_str(&snapshot).unwrap_or_default();
    let mut reflection_status = parsed.get("reflection_status").cloned().unwrap_or_else(|| serde_json::json!({}));
    if let Some(obj) = reflection_status.as_object_mut() {
        obj.insert("rolled_back".to_string(), serde_json::json!(true));
        obj.insert("rolled_back_at".to_string(), serde_json::json!(Utc::now().to_rfc3339()));
    }

    let model = SelfModel {
        capabilities: parsed.get("capabilities").cloned().unwrap_or_else(|| serde_json::json!([])),
        limitations: parsed.get("limitations").cloned().unwrap_or_else(|| serde_json::json!([])),
        active_tools: parsed.get("active_tools").cloned().unwrap_or_else(|| serde_json::json!([])),
        memory_health: parsed.get("memory_health").cloned().unwrap_or_else(|| serde_json::json!({})),
        persona: parsed.get("persona").cloned().unwrap_or_else(|| serde_json::json!({})),
        persona_daily_delta: parsed.get("persona_daily_delta").cloned().unwrap_or_else(|| serde_json::json!({})),
        persona_last_delta_date: parsed.get("persona_last_delta_date").and_then(|v| v.as_str().map(|s| s.to_string())),
        goals: parsed.get("goals").cloned().unwrap_or_else(|| serde_json::json!([])),
        identity_thread: parsed.get("identity_thread").and_then(|v| v.as_str().map(|s| s.to_string())),
        identity_confidence: parsed.get("identity_confidence").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32,
        identity_uncertainty_note: parsed.get("identity_uncertainty_note").and_then(|v| v.as_str().map(|s| s.to_string())),
        identity_updated_at: parsed.get("identity_updated_at").and_then(|v| v.as_str().map(|s| s.to_string())),
        reflection_status,
        reflection_frozen: parsed.get("reflection_frozen").and_then(|v| v.as_bool()).unwrap_or(false),
        last_reflection_at: parsed.get("last_reflection_at").and_then(|v| v.as_str().map(|s| s.to_string())),
        internal_state_summary: parsed.get("internal_state_summary").cloned().unwrap_or_else(|| serde_json::json!({})),
        internal_state_map_version: parsed.get("internal_state_map_version").and_then(|v| v.as_i64()),
        unified_state: parsed.get("unified_state").cloned().unwrap_or_else(|| serde_json::json!({})),
        unified_state_evidence: parsed.get("unified_state_evidence").cloned().unwrap_or_else(|| serde_json::json!({})),
        unified_state_updated_at: parsed.get("unified_state_updated_at").and_then(|v| v.as_str().map(|s| s.to_string())),
        updated_at: parsed.get("updated_at").and_then(|v| v.as_str().map(|s| s.to_string())).unwrap_or_else(|| Utc::now().to_rfc3339()),
    };

    sqlx::query(
        "UPDATE self_model SET
            capabilities_json = ?,
            limitations_json = ?,
            active_tools_json = ?,
            memory_health_json = ?,
            persona_json = ?,
            persona_daily_delta_json = ?,
            persona_last_delta_date = ?,
            goals_json = ?,
            identity_thread = ?,
            identity_confidence = ?,
            identity_uncertainty_note = ?,
            identity_updated_at = ?,
            reflection_status_json = ?,
            reflection_frozen = ?,
            last_reflection_at = ?,
            internal_state_summary_json = ?,
            internal_state_map_version = ?,
            unified_state_json = ?,
            unified_state_evidence_json = ?,
            unified_state_updated_at = ?,
            updated_at = CURRENT_TIMESTAMP
         WHERE id = 1"
    )
    .bind(model.capabilities.to_string())
    .bind(model.limitations.to_string())
    .bind(model.active_tools.to_string())
    .bind(model.memory_health.to_string())
    .bind(model.persona.to_string())
    .bind(model.persona_daily_delta.to_string())
    .bind(model.persona_last_delta_date.clone())
    .bind(model.goals.to_string())
    .bind(model.identity_thread.clone())
    .bind(model.identity_confidence as f64)
    .bind(model.identity_uncertainty_note.clone())
    .bind(model.identity_updated_at.clone())
    .bind(model.reflection_status.to_string())
    .bind(if model.reflection_frozen { 1 } else { 0 })
    .bind(model.last_reflection_at.clone())
    .bind(model.internal_state_summary.to_string())
    .bind(model.internal_state_map_version)
    .bind(model.unified_state.to_string())
    .bind(model.unified_state_evidence.to_string())
    .bind(model.unified_state_updated_at.clone())
    .execute(&*pool)
    .await
    .map_err(|e| e.to_string())?;

    Ok(true)
}

#[tauri::command]
pub async fn identity_rollback(
    db: State<'_, Arc<Db>>,
    snapshot_id: Option<String>,
    reason: Option<String>,
) -> Result<bool, String> {
    gate_allows_memory_write(&db.pool, "default", "identity_rollback").await?;
    let snapshot_meta = if let Some(id) = snapshot_id.as_deref() {
        db.get_identity_snapshot(id).await.map_err(|e| e.to_string())?
    } else {
        db.get_latest_identity_snapshot().await.map_err(|e| e.to_string())?
    };
    let (snapshot_id_log, evidence_event_ids) = if let Some((id, _snapshot, evidence_ids, _)) = snapshot_meta.as_ref() {
        (Some(id.clone()), evidence_ids.clone())
    } else {
        (snapshot_id.clone(), Vec::new())
    };
    let restored = db
        .restore_identity_snapshot(snapshot_id.as_deref())
        .await
        .map_err(|e| e.to_string())?;
    let success = restored.is_some();
    let reason_value = reason.unwrap_or_else(|| "manual".to_string());
    let _ = system_log::log_event(
        &db.pool,
        None,
        if success { "info" } else { "warn" },
        "system",
        None,
        None,
        json!({
            "event": "identity_rollback",
            "status": if success { "restored" } else { "missing_snapshot" },
            "snapshot_id": snapshot_id_log,
            "reason": reason_value,
            "evidence_event_ids": evidence_event_ids,
        }),
    )
    .await;
    Ok(success)
}

#[tauri::command]
pub async fn memory_get_provenance(
    db: State<'_, Arc<Db>>,
    args: BeliefProvenanceArgs,
) -> Result<Vec<EpisodicEvent>, String> {
    let limit = args.limit.unwrap_or(50);
    match args.kind.as_str() {
        "self" => db
            .list_episodic_events_for_self_belief(args.belief_id, limit)
            .await
            .map_err(|e| e.to_string()),
        _ => db
            .list_episodic_events_for_ics_belief(args.belief_id, limit)
            .await
            .map_err(|e| e.to_string()),
    }
}

#[tauri::command]
pub async fn memory_get_entity_provenance(
    db: State<'_, Arc<Db>>,
    args: EntityProvenanceArgs,
) -> Result<Vec<EpisodicEvent>, String> {
    let limit = args.limit.unwrap_or(50);
    db.list_episodic_events_for_entity(args.entity_id, limit)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn memory_list_relation_shape_missing(
    db: State<'_, Arc<Db>>,
    limit: Option<i64>,
) -> Result<Vec<EpisodicEvent>, String> {
    let limit = limit.unwrap_or(50);
    db.list_relation_shape_missing(limit)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn record_strategy_trace(
    db: State<'_, Arc<Db>>,
    features_json: String,
    strategy_label: String,
    outcome: String,
    success_score: Option<f64>,
    run_id: Option<String>,
    conversation_id: Option<String>,
) -> Result<String, String> {
    let features = serde_json::from_str(&features_json).unwrap_or_else(|_| serde_json::json!({}));
    db.record_strategy_trace(
        features,
        &strategy_label,
        &outcome,
        success_score,
        run_id.as_deref(),
        conversation_id.as_deref(),
    )
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn list_strategy_traces(
    db: State<'_, Arc<Db>>,
    limit: Option<i64>,
) -> Result<Vec<StrategyTrace>, String> {
    let limit = limit.unwrap_or(50);
    db.list_strategy_traces(limit)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn create_policy_version(
    db: State<'_, Arc<Db>>,
    label: String,
    payload_json: String,
    parent_id: Option<String>,
    reason: Option<String>,
) -> Result<String, String> {
    let payload = serde_json::from_str(&payload_json).unwrap_or_else(|_| serde_json::json!({}));
    db.create_policy_version(&label, payload, parent_id.as_deref(), reason.as_deref())
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn list_policy_versions(
    db: State<'_, Arc<Db>>,
    limit: Option<i64>,
) -> Result<Vec<PolicyVersion>, String> {
    let limit = limit.unwrap_or(50);
    db.list_policy_versions(limit)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn create_memory_claim(
    db: State<'_, Arc<Db>>,
    kind: String,
    scope: String,
    claim_text: String,
    source_type: String,
    source_ref: Option<String>,
    episodic_event_id: Option<String>,
) -> Result<String, String> {
    gate_allows_memory_write(&db.pool, "default", "create_memory_claim").await?;
    let claim_id = db.create_memory_claim(
        &kind,
        &scope,
        &claim_text,
        &source_type,
        source_ref.as_deref(),
        episodic_event_id.as_deref(),
    )
    .await
    .map_err(|e| e.to_string())?;

    claims::record_claim_outcome(&db.pool, &claim_id, "pending", Some("pending")).await;
    Ok(claim_id)
}

#[tauri::command]
pub async fn list_memory_claims(
    db: State<'_, Arc<Db>>,
    status: Option<String>,
    limit: Option<i64>,
) -> Result<Vec<MemoryClaim>, String> {
    let limit = limit.unwrap_or(50);
    db.list_memory_claims(status.as_deref(), limit)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn memory_get_claim_outcomes(
    pool: State<'_, SqlitePool>,
    limit: Option<i64>,
) -> Result<Vec<ClaimOutcome>, String> {
    let raw: Option<String> = sqlx::query_scalar("SELECT value FROM kv_store WHERE key = ?")
        .bind("last_claim_outcomes")
        .fetch_optional(&*pool)
        .await
        .map_err(|e| e.to_string())?
        .flatten();

    let mut outcomes: Vec<ClaimOutcome> = raw
        .and_then(|payload| serde_json::from_str(&payload).ok())
        .unwrap_or_default();

    if let Some(limit) = limit {
        let limit = limit.max(1).min(200) as usize;
        if outcomes.len() > limit {
            outcomes = outcomes.split_off(outcomes.len() - limit);
        }
    }

    Ok(outcomes)
}

#[tauri::command]
pub async fn memory_evaluate_claims(
    db: State<'_, Arc<Db>>,
    model_client: State<'_, Arc<ModelClient>>,
    limit: Option<i64>,
) -> Result<i64, String> {
    let limit = limit.unwrap_or(20).max(1).min(200) as usize;
    let processed = claims::evaluate_pending_claims(
        &db.pool,
        Some((*model_client).clone()),
        limit,
    )
    .await?;
    Ok(processed as i64)
}

#[tauri::command]
pub async fn memory_backfill_relation_shape_claims(
    db: State<'_, Arc<Db>>,
    model_client: State<'_, Arc<ModelClient>>,
    limit: Option<i64>,
) -> Result<i64, String> {
    gate_allows_memory_write(&db.pool, "default", "memory_backfill_relation_shape_claims").await?;
    let limit = limit.unwrap_or(200).max(1).min(2000) as i64;
    let ids = sqlx::query(
        "SELECT id FROM memory_claims
         WHERE status = 'rejected'
           AND decision_reason IS NOT NULL
           AND (decision_reason LIKE '%relation_shape_%' OR decision_reason LIKE '%RelationShape%')
         ORDER BY created_at DESC
         LIMIT ?"
    )
    .bind(limit)
    .fetch_all(&db.pool)
    .await
    .map_err(|e| e.to_string())?;

    if ids.is_empty() {
        return Ok(0);
    }

    let mut tx = db.pool.begin().await.map_err(|e| e.to_string())?;
    for row in ids.iter() {
        let id: String = row.get("id");
        let _ = sqlx::query(
            "UPDATE memory_claims
             SET status = 'pending',
                 decision_reason = NULL,
                 conflict_topic_key = NULL,
                 conflict_reason = NULL,
                 evaluated_at = NULL,
                 updated_at = CURRENT_TIMESTAMP
             WHERE id = ?"
        )
        .bind(id)
        .execute(&mut *tx)
        .await;
    }
    let _ = tx.commit().await;

    let processed = claims::evaluate_pending_claims(
        &db.pool,
        Some((*model_client).clone()),
        limit as usize,
    )
    .await?;
    Ok(processed as i64)
}

#[tauri::command]
pub async fn memory_backfill_rel_type_claims(
    db: State<'_, Arc<Db>>,
    model_client: State<'_, Arc<ModelClient>>,
    limit: Option<i64>,
) -> Result<i64, String> {
    gate_allows_memory_write(&db.pool, "default", "memory_backfill_rel_type_claims").await?;
    let limit = limit.unwrap_or(200).max(1).min(2000) as i64;
    let ids = sqlx::query(
        "SELECT id FROM memory_claims
         WHERE status = 'rejected'
           AND decision_reason IS NOT NULL
           AND (decision_reason LIKE '%ResolvedRelTypeIdEmpty%' OR decision_reason LIKE '%MissingRelTypeId%')
         ORDER BY created_at DESC
         LIMIT ?"
    )
    .bind(limit)
    .fetch_all(&db.pool)
    .await
    .map_err(|e| e.to_string())?;

    if ids.is_empty() {
        return Ok(0);
    }

    let mut tx = db.pool.begin().await.map_err(|e| e.to_string())?;
    for row in ids.iter() {
        let id: String = row.get("id");
        let _ = sqlx::query(
            "UPDATE memory_claims
             SET status = 'pending',
                 decision_reason = NULL,
                 conflict_topic_key = NULL,
                 conflict_reason = NULL,
                 evaluated_at = NULL,
                 updated_at = CURRENT_TIMESTAMP
             WHERE id = ?"
        )
        .bind(id)
        .execute(&mut *tx)
        .await;
    }
    let _ = tx.commit().await;

    let processed = claims::evaluate_pending_claims(
        &db.pool,
        Some((*model_client).clone()),
        limit as usize,
    )
    .await?;
    Ok(processed as i64)
}

#[tauri::command]
pub async fn update_memory_claim_status(
    db: State<'_, Arc<Db>>,
    model_client: State<'_, Arc<ModelClient>>,
    claim_id: String,
    status: String,
) -> Result<(), String> {
    gate_allows_memory_write(&db.pool, "default", "update_memory_claim_status").await?;
    if status == "promoted" {
        let promotion = claims::promote_claim(&db.pool, Some((*model_client).clone()), &claim_id).await?;
        let linked_belief_id = promotion.written_ids.first().copied();
        let _ = episodic::emit_claim_status_event(
            &db.pool,
            &claim_id,
            &status,
            linked_belief_id,
            "system",
            Some("memory_claims"),
            Some("promoted"),
            None,
            None,
        )
        .await;
        return Ok(());
    }

    let decision_reason = "manual_update";
    db.update_memory_claim_status(&claim_id, &status, Some(decision_reason))
        .await
        .map_err(|e| e.to_string())?;
    claims::record_claim_outcome(&db.pool, &claim_id, &status, Some(decision_reason)).await;
    let _ = episodic::emit_claim_status_event(
        &db.pool,
        &claim_id,
        &status,
        None,
        "system",
        Some("memory_claims"),
        Some(decision_reason),
        None,
        None,
    )
    .await;
    Ok(())
}
