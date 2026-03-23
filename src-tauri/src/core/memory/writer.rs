use crate::core::memory::types::{Scope, SourceType, RelationShape, Cardinality};
use crate::core::memory::dsl::{FactStmt, RelStmt, RelDirection};
use crate::core::memory::config::{SOURCE_WEIGHT_USER, SOURCE_WEIGHT_TOOL, SOURCE_WEIGHT_SYSTEM, SOURCE_WEIGHT_INFERENCE};
use crate::core::memory::canonical::{compute_topic_key_fact, compute_value_hash, compute_signature_hash};
use crate::core::memory::embedding_index;
use crate::core::memory::cache;
use crate::core::memory::snippets;
use crate::core::identity;
use crate::core::system_controls;
use crate::core::system_log;
use crate::core::sensitivity::{SensitivityLevel, detect_sensitivity, phi_consent_allowed};
use sqlx::SqlitePool; // Owns connection
use sqlx::Row;

use std::collections::HashMap;
use std::sync::Arc;
use crate::core::model_client::ModelClient;
use crate::core::episodic;
use crate::db::Db;
use serde_json::json;
use uuid::Uuid;
use sha2::{Digest, Sha256};

use chrono::{DateTime, Utc};

/// Configuration for embedding generation (§10.3)
#[derive(Clone, Debug)]
pub struct EmbeddingConfig {
    pub base_url: String,
    pub model: String,
    pub enabled: bool,
}

pub struct WriteContext {
    pub pool: SqlitePool,
    pub model_client: Option<Arc<ModelClient>>,
    pub scope: Scope,
    pub source: SourceType,
    pub source_ref: Option<String>,
    pub now: DateTime<Utc>,
    pub embedding_config: Option<EmbeddingConfig>,
    pub conversation_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct RelationWriteOptions {
    pub sort_participants: bool,
}

#[derive(Debug)]
pub enum WriteResult {
    Inserted(i64),
    Updated(i64),
    Conflict { belief_id: i64, conflict_set_id: i64 },
    Ignored(String),
    Error(String),
}

async fn upsert_memory_sensitivity(
    pool: &SqlitePool,
    belief_id: i64,
    kind: &str,
    sensitivity: SensitivityLevel,
) {
    let _ = sqlx::query(
        "INSERT INTO memory_sensitivity (belief_id, kind, sensitivity, created_at, updated_at)
         VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
         ON CONFLICT(belief_id) DO UPDATE SET
            kind = excluded.kind,
            sensitivity = excluded.sensitivity,
            updated_at = CURRENT_TIMESTAMP",
    )
    .bind(belief_id)
    .bind(kind)
    .bind(sensitivity.as_str())
    .execute(pool)
    .await;
}

fn hash_text(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

async fn log_phi_write_blocked(
    ctx: &WriteContext,
    kind: &str,
    sensitivity: SensitivityLevel,
    value_hash: Option<&str>,
    rel_type: Option<&str>,
) {
    let _ = system_log::log_event(
        &ctx.pool,
        None,
        "warn",
        "memory",
        None,
        None,
        json!({
            "event": "phi_write_blocked",
            "kind": kind,
            "sensitivity": sensitivity.as_str(),
            "value_hash": value_hash,
            "rel_type": rel_type,
            "source": serialize_source(ctx.source),
            "conversation_id": ctx.conversation_id,
        }),
    )
    .await;
}

async fn log_memory_write_blocked(
    ctx: &WriteContext,
    kind: &str,
    reason: &str,
    details: serde_json::Value,
) {
    let _ = system_log::log_event(
        &ctx.pool,
        None,
        "warn",
        "memory",
        None,
        None,
        json!({
            "event": "memory_write_blocked",
            "kind": kind,
            "reason": reason,
            "details": details,
            "source": serialize_source(ctx.source),
            "conversation_id": ctx.conversation_id,
        }),
    )
    .await;
}

/// Create a new entity (Spec §6.2)
/// New entities start tentative unless strongly anchored (§1.1)
pub async fn create_entity(label: &str, entity_type: Option<&str>, ctx: &WriteContext) -> Result<i64, String> {
    use crate::core::memory::canonical::canonicalize_label;
    
    let canon = canonicalize_label(label);
    let mut attempts = 0;
    
    loop {
        attempts += 1;
        
        // 1. Try find existing
        let existing = sqlx::query("SELECT id FROM ics_entities WHERE label_canonical = ?")
            .bind(&canon)
            .fetch_optional(&ctx.pool)
            .await;
            
        match existing {
            Ok(Some(row)) => return Ok(row.get("id")),
            Ok(None) => {}, // Proceed to create
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("database is locked") || err_str.contains("code: 5") {
                    if attempts <= 5 {
                        tokio::time::sleep(std::time::Duration::from_millis(50 * (1 << attempts))).await;
                        continue;
                    }
                }
                return Err(err_str);
            }
        }
        
        // 2. Create as tentative (per §1.1)
        let res = sqlx::query("INSERT INTO ics_entities (label, label_canonical, aliases, resolution_state, entity_type) VALUES (?, ?, '[]', 'tentative', ?) RETURNING id")
            .bind(label)
            .bind(&canon)
            .bind(entity_type)
            .fetch_one(&ctx.pool)
            .await;
            
        match res {
            Ok(row) => {
                let id: i64 = row.get("id");
                let source_type = serialize_source(ctx.source);
                let _ = episodic::emit_episodic_event(
                    &ctx.pool,
                    "entity_created",
                    json!({ "status": "created", "summary_snippet": label, "entity_id": id }),
                    None,
                    None,
                    ctx.conversation_id.as_deref(),
                    None,
                    &source_type,
                    ctx.source_ref.as_deref(),
                    None,
                    None,
                )
                .await;
                let payload = format!("entity_created:{}:{}", id, label);
                log_semantic_write(ctx, "memory_writer_evidence", &payload).await;
                cache::bump_cache_version();
                return Ok(id);
            },
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("database is locked") || err_str.contains("code: 5") {
                     if attempts <= 5 {
                        tokio::time::sleep(std::time::Duration::from_millis(50 * (1 << attempts))).await;
                        continue;
                    }
                }
                return Err(err_str);
            }
        }
    }
}

/// Promote entity from tentative to normal (§1.1)
/// Called when: 2+ sources, key/handle assigned, cross-session usage
pub async fn promote_entity_to_normal(entity_id: i64, pool: &SqlitePool) -> Result<(), String> {
    // Check if already normal
    let current: Option<String> = sqlx::query_scalar("SELECT resolution_state FROM ics_entities WHERE id = ?")
        .bind(entity_id)
        .fetch_optional(pool)
        .await
        .map_err(|e| e.to_string())?;
    
    if current.as_deref() == Some("normal") {
        return Ok(());
    }
    
    // Promote
    sqlx::query("UPDATE ics_entities SET resolution_state = 'normal' WHERE id = ? AND resolution_state = 'tentative'")
        .bind(entity_id)
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;
    
    Ok(())
}

use crate::core::memory::dsl::TimeExpr;

/// Compute time bucket from DSL time expression (Spec §4.3)
fn compute_time_bucket(time_expr: &Option<TimeExpr>) -> (String, Option<String>) {
    match time_expr {
        None => ("atemporal".to_string(), None),
        Some(t) => match t.kind.as_str() {
            "instant" => {
                // Check if it's a full timestamp (>10 chars) or just a date
                if t.value.len() > 10 {
                    ("exact".to_string(), Some(t.value.clone()))
                } else {
                    ("day".to_string(), Some(t.value.clone()))
                }
            },
            "range" => ("range".to_string(), Some(t.value.clone())),
            "relative" => {
                // Expand relative keywords to actual dates
                let now = chrono::Utc::now();
                match t.value.as_str() {
                    "today" => ("day".to_string(), Some(now.format("%Y-%m-%d").to_string())),
                    "yesterday" => {
                        let yesterday = now - chrono::Duration::days(1);
                        ("day".to_string(), Some(yesterday.format("%Y-%m-%d").to_string()))
                    },
                    "this_week" => {
                        use chrono::Datelike;
                        let start = now - chrono::Duration::days(now.weekday().num_days_from_monday() as i64);
                        let end = start + chrono::Duration::days(6);
                        ("range".to_string(), Some(format!("{}..{}", 
                            start.format("%Y-%m-%d"), end.format("%Y-%m-%d"))))
                    },
                    _ => ("atemporal".to_string(), None),
                }
            },
            _ => ("atemporal".to_string(), None),
        }
    }
}

/// Write a resolved Fact (Spec §6.4)
pub async fn write_fact(stmt: FactStmt, subject_id: i64, ctx: &WriteContext) -> WriteResult {
    if ctx.source == SourceType::System {
        if let Some(source_ref) = ctx.source_ref.as_deref() {
            if is_internal_state_ref(source_ref) {
                let value_hash = compute_value_hash(&stmt.value);
                log_memory_write_blocked(
                    ctx,
                    "fact",
                    "internal_state_blocked",
                    json!({
                        "source_ref": source_ref,
                        "key": stmt.key,
                        "value_hash": value_hash,
                    }),
                )
                .await;
                return WriteResult::Ignored("internal_state_blocked".to_string());
            }
        }
        if let Some(certainty) = stmt.certainty {
            if certainty < 0.5 {
                let value_hash = compute_value_hash(&stmt.value);
                log_memory_write_blocked(
                    ctx,
                    "fact",
                    "low_confidence_system",
                    json!({
                        "key": stmt.key,
                        "value_hash": value_hash,
                        "certainty": certainty,
                    }),
                )
                .await;
                return WriteResult::Ignored("low_confidence_system".to_string());
            }
        }
    }
    if matches!(ctx.source, SourceType::System | SourceType::Tool | SourceType::Inference) {
        if contains_diagnostic_marker(&stmt.key) || contains_diagnostic_marker(&stmt.value) {
            let value_hash = compute_value_hash(&stmt.value);
            log_memory_write_blocked(
                ctx,
                "fact",
                "telemetry_filtered",
                json!({
                    "key": stmt.key,
                    "value_hash": value_hash,
                }),
            )
            .await;
            return WriteResult::Ignored("telemetry_filtered".to_string());
        }
    }
    let value_hash = compute_value_hash(&stmt.value);
    let scope_str = serde_json::to_string(&ctx.scope).unwrap_or_default();
    let (time_bucket_kind, time_bucket_value) = compute_time_bucket(&stmt.time_expr);
    let time_bucket_value_sig = time_bucket_value.clone().unwrap_or_default();
    let topic_key = compute_topic_key_fact(subject_id, &stmt.key);

    let sig_inputs = vec![
        ("subject_id".to_string(), subject_id.to_string()),
        ("key".to_string(), stmt.key.clone()),
        ("value_hash".to_string(), value_hash.clone()),
        ("scope".to_string(), scope_str.clone()),
        ("time_bucket_kind".to_string(), time_bucket_kind.clone()),
        ("time_bucket_value".to_string(), time_bucket_value_sig),
        ("polarity".to_string(), stmt.polarity.clone()),
    ];
    let sig_refs: Vec<(&str, &str)> = sig_inputs
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let signature_hash = compute_signature_hash(&sig_refs);
    let subject_label: String = sqlx::query_scalar("SELECT label FROM ics_entities WHERE id = ?")
        .bind(subject_id)
        .fetch_optional(&ctx.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| format!("entity_{}", subject_id));
    let snippet = snippets::render_fact_snippet(&subject_label, &stmt.key, &stmt.value, &stmt.polarity);
    if snippet.trim().is_empty() {
        return WriteResult::Error("EmptyEvidenceSnippet".to_string());
    }
    let identity_hit = identity::is_identity_fact_key(&stmt.key)
        || identity::contains_identity_statement(&stmt.value);
    let capability_hit = identity::is_capability_fact_key(&stmt.key)
        || identity::contains_capability_statement(&stmt.value);
    let sensitivity = detect_sensitivity(&format!("{} {}", stmt.key, stmt.value))
        .or_else(|| detect_sensitivity(&snippet));
    if let Some(level) = sensitivity {
        if !phi_consent_allowed(&ctx.pool, ctx.conversation_id.as_deref()).await {
            let value_hash = compute_value_hash(&stmt.value);
            log_phi_write_blocked(ctx, "fact", level, Some(&value_hash), None).await;
            log_memory_write_blocked(
                ctx,
                "fact",
                "phi_blocked",
                json!({
                    "value_hash": value_hash,
                    "sensitivity": level.as_str(),
                    "key": stmt.key,
                }),
            )
            .await;
            return WriteResult::Ignored("phi_blocked".to_string());
        }
    }
    let episodic_on = episodic::episodic_enabled(&ctx.pool).await;
    
    let mut attempts = 0;
    loop {
        attempts += 1;
        
        // Use a block to control transaction lifetime
        let result = async {
             // START TRANSACTION
            let mut tx = match ctx.pool.begin().await {
                Ok(t) => t,
                Err(e) => return (WriteResult::Error(format!("Failed to start transaction: {}", e)), None),
            };

            // 2. Check for existing identical belief
            let existing: Option<sqlx::sqlite::SqliteRow> = sqlx::query("SELECT id, evidence_weight_total, confidence FROM ics_beliefs WHERE signature_hash = ? AND scope = ? AND polarity = ? AND status = 'active'")
                .bind(&signature_hash)
                .bind(&scope_str)
                .bind(&stmt.polarity)
                .fetch_optional(&mut *tx)
                .await
                .unwrap_or(None);
                
            if let Some(row) = existing {
                let id: i64 = row.get("id");
                let weight_total: f64 = row.get("evidence_weight_total");
                let current_confidence: f64 = row.try_get("confidence").unwrap_or(1.0);
                
                // Reinforce
                let weight = get_source_weight(ctx.source);
                let new_weight = weight_total as f32 + weight;
                let new_confidence = current_confidence.max(stmt.certainty.unwrap_or(1.0) as f64);
                
                let episodic_event_id = if episodic_on { Some(Uuid::new_v4().to_string()) } else { None };
                let _ = sqlx::query("INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id) VALUES (?, ?, ?, ?, ?, ?)")
                    .bind(id)
                    .bind(serialize_source(ctx.source))
                    .bind(stmt.source_ref.as_ref().or(ctx.source_ref.as_ref()))
                    .bind(&snippet)
                    .bind(weight)
                    .bind(episodic_event_id.as_deref())
                    .execute(&mut *tx)
                    .await;

                let identity_event_id = if identity_hit {
                    sqlx::query(
                        "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id)
                         VALUES (?, 'identity_statement', ?, ?, 0.0, ?)
                         RETURNING id",
                    )
                    .bind(id)
                    .bind(stmt.source_ref.as_ref().or(ctx.source_ref.as_ref()))
                    .bind(&snippet)
                    .bind(episodic_event_id.as_deref())
                    .fetch_one(&mut *tx)
                    .await
                    .ok()
                    .map(|row| row.get::<i64, _>("id"))
                } else {
                    None
                };
                let capability_event_id = if capability_hit {
                    sqlx::query(
                        "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id)
                         VALUES (?, 'capability_statement', ?, ?, 0.0, ?)
                         RETURNING id",
                    )
                    .bind(id)
                    .bind(stmt.source_ref.as_ref().or(ctx.source_ref.as_ref()))
                    .bind(&snippet)
                    .bind(episodic_event_id.as_deref())
                    .fetch_one(&mut *tx)
                    .await
                    .ok()
                    .map(|row| row.get::<i64, _>("id"))
                } else {
                    None
                };
                
                let _ = sqlx::query("UPDATE ics_beliefs SET evidence_weight_total = ?, confidence = ?, last_evidence_at = CURRENT_TIMESTAMP WHERE id = ?")
                    .bind(new_weight)
                    .bind(new_confidence)
                    .bind(id)
                    .execute(&mut *tx)
                    .await;
                    
                if let Err(e) = tx.commit().await {
                    return (WriteResult::Error(format!("Failed to commit update: {}", e)), None);
                }

                // Post-commit compaction (can fail without rollback)
                let pool = ctx.pool.clone();
                let belief_id = id;
                spawn_compaction(pool, belief_id);

                let _ = crate::core::memory::attention::salience::recompute_salience_for_beliefs(
                    &ctx.pool,
                    &[id],
                )
                .await;

                if let Some(identity_event_id) = identity_event_id {
                    let _ = system_log::log_event(
                        &ctx.pool,
                        None,
                        "info",
                        "memory",
                        None,
                        None,
                        json!({
                            "event": "identity_evidence_attached",
                            "belief_id": id,
                            "evidence_event_id": identity_event_id,
                            "conversation_id": ctx.conversation_id,
                            "source": "memory_writer",
                        }),
                    )
                    .await;
                }

                if matches!(ctx.scope, Scope::SelfScope) && ctx.source != SourceType::System {
                    let mut source_ids: Vec<i64> = Vec::new();
                    if let Some(id) = identity_event_id {
                        source_ids.push(id);
                    }
                    if let Some(id) = capability_event_id {
                        source_ids.push(id);
                    }
                    if !source_ids.is_empty() {
                        let pool = ctx.pool.clone();
                        let key = stmt.key.clone();
                        let value = stmt.value.clone();
                        let snippet = snippet.clone();
                        let now = ctx.now;
                        let source_ids = source_ids.clone();
                        tokio::spawn(async move {
                            let _ = crate::core::self_memory::write_self_fact_unbridged(
                                &pool,
                                &key,
                                &value,
                                &snippet,
                                Some(now),
                                SourceType::System,
                                Some(&source_ids),
                            )
                            .await;
                        });
                    }
                }
                    
                return (WriteResult::Updated(id), episodic_event_id);
            }
            
            // 3. Cardinality Logic (Spec I8: Default MANY_SET)
            let cardinality: Option<String> = sqlx::query_scalar("SELECT cardinality FROM ics_token_policies WHERE token = ?")
                .bind(&stmt.key)
                .fetch_optional(&mut *tx)
                .await
                .unwrap_or(None);

            let is_one = match cardinality.as_deref() {
                Some("ONE") => true,
                _ => false, 
            };
            let mut superseded_ids: Vec<i64> = Vec::new();
            
            if is_one {
                let collision: Option<sqlx::sqlite::SqliteRow> = sqlx::query("SELECT id FROM ics_beliefs WHERE topic_key = ? AND scope = ? AND polarity = ? AND status = 'active'")
                    .bind(&topic_key)
                    .bind(&scope_str)
                    .bind(&stmt.polarity)
                    .fetch_optional(&mut *tx)
                    .await
                    .unwrap_or(None);
                    
                if let Some(row) = collision {
                    let old_id: i64 = row.get("id");
                    superseded_ids.push(old_id);
                    // Supersede - mark old as inactive
                    let _ = sqlx::query("UPDATE ics_beliefs SET status = 'inactive' WHERE id = ?")
                        .bind(old_id)
                        .execute(&mut *tx)
                        .await;
                }
            }
            
            // 4. Insert New Belief
            let weight = get_source_weight(ctx.source);
            let certainty = stmt.certainty.unwrap_or(1.0);
            
            let layer = initial_layer_for_source(ctx.source);
            let res = sqlx::query("INSERT INTO ics_beliefs (kind, scope, polarity, layer, topic_key, signature_hash, evidence_weight_total, confidence, time_bucket_kind, time_bucket_value, observed_at, created_at) VALUES ('fact', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id")
                .bind(&scope_str)
                .bind(&stmt.polarity)
                .bind(layer)
                .bind(&topic_key)
                .bind(&signature_hash)
                .bind(weight)
                .bind(certainty)
                .bind(&time_bucket_kind)
                .bind(&time_bucket_value)
                .bind(ctx.now.to_rfc3339())
                .bind(ctx.now.to_rfc3339())
                .fetch_one(&mut *tx)
                .await;
                
            let id = match res {
                Ok(row) => row.get::<i64, _>("id"),
                Err(e) => {
                    let _ = tx.rollback().await;
                    return (WriteResult::Error(format!("Failed to insert belief: {}", e)), None);
                }
            };
            
            // Create supersedes link (using captured collision.old_id if ONE cardinality)
            if is_one {
                // Find the belief we just marked inactive and link
                let old_ids: Vec<i64> = sqlx::query_scalar("SELECT id FROM ics_beliefs WHERE topic_key = ? AND scope = ? AND status = 'inactive' AND id != ?")
                    .bind(&topic_key)
                    .bind(&scope_str)
                    .bind(id)
                    .fetch_all(&mut *tx)
                    .await
                    .unwrap_or_default();
                
                for old_id in old_ids {
                    let _ = sqlx::query("INSERT INTO ics_belief_links (from_id, to_id, link_type) VALUES (?, ?, 'supersedes')")
                        .bind(id)
                        .bind(old_id)
                        .execute(&mut *tx)
                        .await;
                }
            }
            
            let _ = sqlx::query("INSERT INTO ics_fact_beliefs (belief_id, subject_entity_id, key, value_literal, value_hash) VALUES (?, ?, ?, ?, ?)")
                .bind(id)
                .bind(subject_id)
                .bind(&stmt.key)
                .bind(&stmt.value)
                .bind(&value_hash)
                .execute(&mut *tx)
                .await;
                
            // Add initial evidence
            let episodic_event_id = if episodic_on { Some(Uuid::new_v4().to_string()) } else { None };
            let _ = sqlx::query("INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id) VALUES (?, ?, ?, ?, ?, ?)")
                .bind(id)
                .bind(serialize_source(ctx.source))
                .bind(stmt.source_ref.as_ref().or(ctx.source_ref.as_ref()))
                .bind(&snippet)
                .bind(weight)
                .bind(episodic_event_id.as_deref())
                .execute(&mut *tx)
                .await;

            let identity_event_id = if identity_hit {
                sqlx::query(
                    "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id)
                     VALUES (?, 'identity_statement', ?, ?, 0.0, ?)
                     RETURNING id",
                )
                .bind(id)
                .bind(stmt.source_ref.as_ref().or(ctx.source_ref.as_ref()))
                .bind(&snippet)
                .bind(episodic_event_id.as_deref())
                .fetch_one(&mut *tx)
                .await
                .ok()
                .map(|row| row.get::<i64, _>("id"))
            } else {
                None
            };
            let capability_event_id = if capability_hit {
                sqlx::query(
                    "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id)
                     VALUES (?, 'capability_statement', ?, ?, 0.0, ?)
                     RETURNING id",
                )
                .bind(id)
                .bind(stmt.source_ref.as_ref().or(ctx.source_ref.as_ref()))
                .bind(&snippet)
                .bind(episodic_event_id.as_deref())
                .fetch_one(&mut *tx)
                .await
                .ok()
                .map(|row| row.get::<i64, _>("id"))
            } else {
                None
            };
            
            if let Err(e) = tx.commit().await {
                return (WriteResult::Error(format!("Failed to commit new belief: {}", e)), None);
            }
            
            // Post-commit
            let pool = ctx.pool.clone();
            let belief_id = id;
            spawn_compaction(pool, belief_id);
            
            let _ = crate::core::memory::attention::salience::recompute_salience_for_beliefs(
                &ctx.pool,
                &[id],
            )
            .await;

            if let Some(identity_event_id) = identity_event_id {
                let _ = system_log::log_event(
                    &ctx.pool,
                    None,
                    "info",
                    "memory",
                    None,
                    None,
                    json!({
                        "event": "identity_evidence_attached",
                        "belief_id": id,
                        "evidence_event_id": identity_event_id,
                        "conversation_id": ctx.conversation_id,
                        "source": "memory_writer",
                    }),
                )
                .await;
            }
            if matches!(ctx.scope, Scope::SelfScope) && ctx.source != SourceType::System {
                let mut source_ids: Vec<i64> = Vec::new();
                if let Some(id) = identity_event_id {
                    source_ids.push(id);
                }
                if let Some(id) = capability_event_id {
                    source_ids.push(id);
                }
                if !source_ids.is_empty() {
                    let pool = ctx.pool.clone();
                    let key = stmt.key.clone();
                    let value = stmt.value.clone();
                    let snippet = snippet.clone();
                    let now = ctx.now;
                    let source_ids = source_ids.clone();
                    tokio::spawn(async move {
                        let _ = crate::core::self_memory::write_self_fact_unbridged(
                            &pool,
                            &key,
                            &value,
                            &snippet,
                            Some(now),
                            SourceType::System,
                            Some(&source_ids),
                        )
                        .await;
                    });
                }
            }
            
            // Generate and store embedding (§10.3)
            // Using EmbeddingConfig from context
            if let (Some(client), Some(config)) = (&ctx.model_client, &ctx.embedding_config) {
                if config.enabled {
                    let embedding_text = format!("{}: {}", &stmt.key, &stmt.value);
                    let client = client.clone();
                    let pool = ctx.pool.clone();
                    let belief_id = id;
                    let base_url = config.base_url.clone();
                    let model = config.model.clone();
                    
                    tokio::spawn(async move {
                        if let Ok(embedding) = client.embed(&base_url, None, &model, &embedding_text).await {
                             let embedding_bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
                             let signature = embedding_index::embedding_signature(&embedding) as i64;
                             let _ = sqlx::query(
                                "INSERT INTO ics_embeddings (id, assertion_id, embedding, created_at) VALUES (?, ?, ?, datetime('now'))"
                            )
                            .bind(uuid::Uuid::new_v4().to_string())
                            .bind(belief_id)
                            .bind(&embedding_bytes)
                            .execute(&pool)
                            .await;

                             let _ = sqlx::query(
                                "INSERT OR IGNORE INTO ics_embedding_lsh (assertion_id, bucket, created_at) VALUES (?, ?, datetime('now'))"
                             )
                             .bind(belief_id)
                             .bind(signature)
                             .execute(&pool)
                             .await;
                        }
                    });
                }
            }
            
            // §7.6: Conflict detection
            // Check for polarity conflicts (assert vs deny with same topic_key)
            let polarity = &stmt.polarity;
            let opposite_polarity = if polarity == "assert" { "deny" } else { "assert" };
            
            let conflicting_ids: Vec<i64> = sqlx::query_scalar(
                "SELECT id FROM ics_beliefs WHERE topic_key = ? AND scope = ? AND status = 'active' AND polarity = ? AND id != ?"
            )
            .bind(&topic_key)
            .bind(&scope_str)
            .bind(opposite_polarity)
            .bind(id)
            .fetch_all(&ctx.pool)
            .await
            .unwrap_or_default();

            let collision_ids: Vec<i64> = sqlx::query_scalar(
                "SELECT id FROM ics_beliefs WHERE topic_key = ? AND scope = ? AND status = 'active' AND polarity = ? AND id != ? AND signature_hash != ?"
            )
            .bind(&topic_key)
            .bind(&scope_str)
            .bind(&stmt.polarity)
            .bind(id)
            .bind(&signature_hash)
            .fetch_all(&ctx.pool)
            .await
            .unwrap_or_default();
            
            let mut conflict_set_id: Option<i64> = None;
            let mut conflict_candidates: Vec<i64> = Vec::new();
            conflict_candidates.extend(conflicting_ids.iter().cloned());
            conflict_candidates.extend(collision_ids.iter().cloned());
            conflict_candidates.extend(superseded_ids.iter().cloned());
            conflict_candidates.sort();
            conflict_candidates.dedup();
            if !conflict_candidates.is_empty() {
                // Create or update conflict set
                let existing_set: Option<i64> = sqlx::query_scalar(
                    "SELECT id FROM ics_conflict_sets WHERE topic_key = ? AND status = 'open'"
                )
                .bind(&topic_key)
                .fetch_optional(&ctx.pool)
                .await
                .ok()
                .flatten();
                
                let set_id = match existing_set {
                    Some(sid) => sid,
                    None => {
                        let row = sqlx::query("INSERT INTO ics_conflict_sets (topic_key, status) VALUES (?, 'open') RETURNING id")
                            .bind(&topic_key)
                            .fetch_one(&ctx.pool)
                            .await;
                        match row {
                            Ok(r) => r.get("id"),
                            Err(_) => 0,
                        }
                    }
                };
                
                if set_id != 0 {
                    conflict_set_id = Some(set_id);
                }

                // Create contradicts links
                for cid in &conflict_candidates {
                    let _ = sqlx::query("INSERT OR IGNORE INTO ics_belief_links (from_id, to_id, link_type) VALUES (?, ?, 'contradicts')")
                        .bind(id)
                        .bind(cid)
                        .execute(&ctx.pool)
                        .await;
                    let _ = sqlx::query("INSERT OR IGNORE INTO ics_belief_links (from_id, to_id, link_type) VALUES (?, ?, 'contradicts')")
                        .bind(cid)
                        .bind(id)
                        .execute(&ctx.pool)
                        .await;
                    
                    // Add to conflict set members
                    let _ = sqlx::query("INSERT OR IGNORE INTO ics_conflict_set_members (conflict_set_id, belief_id) VALUES (?, ?)")
                        .bind(set_id)
                        .bind(id)
                        .execute(&ctx.pool)
                        .await;
                    let _ = sqlx::query("INSERT OR IGNORE INTO ics_conflict_set_members (conflict_set_id, belief_id) VALUES (?, ?)")
                        .bind(set_id)
                        .bind(cid)
                        .execute(&ctx.pool)
                        .await;
                }
            }

            if let Some(conflict_set_id) = conflict_set_id {
                (
                    WriteResult::Conflict {
                        belief_id: id,
                        conflict_set_id,
                    },
                    episodic_event_id,
                )
            } else {
                (WriteResult::Inserted(id), episodic_event_id)
            }
        }.await;

        match result {
            (WriteResult::Error(e), _) if (e.contains("database is locked") || e.contains("code: 5")) && attempts <= 5 => {
                 tokio::time::sleep(std::time::Duration::from_millis(50 * (1 << attempts))).await;
                 continue;
            },
            (WriteResult::Inserted(id), Some(event_id)) => {
                let source_ref = stmt.source_ref.as_deref().or(ctx.source_ref.as_deref());
                emit_memory_event(
                    ctx,
                    &event_id,
                    "memory_write_fact",
                    "inserted",
                    &snippet,
                    &scope_str,
                    source_ref,
                    Some(id),
                    Some(subject_id),
                    &stmt.polarity,
                )
                .await;
                if let Some(level) = sensitivity {
                    upsert_memory_sensitivity(&ctx.pool, id, "fact", level).await;
                }
                cache::bump_cache_version();
                return WriteResult::Inserted(id);
            }
            (WriteResult::Inserted(id), None) => {
                let payload = format!("memory_write_fact:{}:{}", id, snippet);
                log_semantic_write(ctx, "memory_writer_evidence", &payload).await;
                if let Some(level) = sensitivity {
                    upsert_memory_sensitivity(&ctx.pool, id, "fact", level).await;
                }
                cache::bump_cache_version();
                return WriteResult::Inserted(id);
            }
            (WriteResult::Updated(id), Some(event_id)) => {
                let source_ref = stmt.source_ref.as_deref().or(ctx.source_ref.as_deref());
                emit_memory_event(
                    ctx,
                    &event_id,
                    "memory_write_fact",
                    "updated",
                    &snippet,
                    &scope_str,
                    source_ref,
                    Some(id),
                    Some(subject_id),
                    &stmt.polarity,
                )
                .await;
                if let Some(level) = sensitivity {
                    upsert_memory_sensitivity(&ctx.pool, id, "fact", level).await;
                }
                cache::bump_cache_version();
                return WriteResult::Updated(id);
            }
            (WriteResult::Updated(id), None) => {
                let payload = format!("memory_write_fact:{}:{}", id, snippet);
                log_semantic_write(ctx, "memory_writer_evidence", &payload).await;
                if let Some(level) = sensitivity {
                    upsert_memory_sensitivity(&ctx.pool, id, "fact", level).await;
                }
                cache::bump_cache_version();
                return WriteResult::Updated(id);
            }
            (WriteResult::Conflict { belief_id, conflict_set_id }, Some(event_id)) => {
                let source_ref = stmt.source_ref.as_deref().or(ctx.source_ref.as_deref());
                emit_memory_event(
                    ctx,
                    &event_id,
                    "memory_write_fact",
                    "conflict",
                    &snippet,
                    &scope_str,
                    source_ref,
                    Some(belief_id),
                    Some(subject_id),
                    &stmt.polarity,
                )
                .await;
                if let Some(level) = sensitivity {
                    upsert_memory_sensitivity(&ctx.pool, belief_id, "fact", level).await;
                }
                cache::bump_cache_version();
                return WriteResult::Conflict { belief_id, conflict_set_id };
            }
            (WriteResult::Conflict { belief_id, conflict_set_id }, None) => {
                let payload = format!("memory_write_fact:{}:{}", belief_id, snippet);
                log_semantic_write(ctx, "memory_writer_evidence", &payload).await;
                if let Some(level) = sensitivity {
                    upsert_memory_sensitivity(&ctx.pool, belief_id, "fact", level).await;
                }
                cache::bump_cache_version();
                return WriteResult::Conflict { belief_id, conflict_set_id };
            }
            (other, _) => return other,
        }
    }
}

pub async fn write_rel(
    stmt: RelStmt,
    participants: Vec<(String, i64)>,
    shape: &RelationShape,
    options: RelationWriteOptions,
    ctx: &WriteContext,
    rel_type_raw: Option<&str>,
    rel_type_id: Option<&str>,
) -> WriteResult {
    if ctx.source == SourceType::System {
        if let Some(source_ref) = ctx.source_ref.as_deref() {
            if is_internal_state_ref(source_ref) {
                log_memory_write_blocked(
                    ctx,
                    "rel",
                    "internal_state_blocked",
                    json!({
                        "source_ref": source_ref,
                        "rel_type": stmt.rel_type,
                        "participant_count": participants.len(),
                    }),
                )
                .await;
                return WriteResult::Ignored("internal_state_blocked".to_string());
            }
        }
        if let Some(certainty) = stmt.certainty {
            if certainty < 0.5 {
                log_memory_write_blocked(
                    ctx,
                    "rel",
                    "low_confidence_system",
                    json!({
                        "rel_type": stmt.rel_type,
                        "participant_count": participants.len(),
                        "certainty": certainty,
                    }),
                )
                .await;
                return WriteResult::Ignored("low_confidence_system".to_string());
            }
        }
    }
    if matches!(ctx.source, SourceType::System | SourceType::Tool | SourceType::Inference) {
        if contains_diagnostic_marker(&stmt.rel_type) {
            log_memory_write_blocked(
                ctx,
                "rel",
                "telemetry_filtered",
                json!({
                    "rel_type": stmt.rel_type,
                    "participant_count": participants.len(),
                }),
            )
            .await;
            return WriteResult::Ignored("telemetry_filtered".to_string());
        }
    }
    use crate::core::memory::canonical::{serialize_participant_ids, serialize_participants};
    
    let ordered_participants = participants.clone();
    let mut canonical_participants = participants.clone();
    let is_directional_pair = stmt.direction.is_some() && canonical_participants.len() == 2;
    if options.sort_participants && stmt.direction.is_none() {
        canonical_participants.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    } else if is_directional_pair {
        canonical_participants.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    }
    let rel_type_norm = stmt.rel_type.clone();
    let rel_type_raw = rel_type_raw.unwrap_or(rel_type_norm.as_str());
    let mut rel_type_id = match rel_type_id.or_else(|| stmt.rel_type_id.as_deref()) {
        Some(id) if !id.trim().is_empty() => id.to_string(),
        _ => String::new(),
    };
    if rel_type_id.trim().is_empty() {
        let canonicalization_on = crate::core::memory::compiler::relation_canonicalization_enabled(&ctx.pool).await;
        let roles_seen: Vec<String> = canonical_participants.iter().map(|(role, _)| role.clone()).collect();
        let resolved = crate::core::memory::rel_type_catalog::resolve_rel_type(
            &ctx.pool,
            &rel_type_norm,
            &roles_seen,
            canonicalization_on,
        )
        .await
        .unwrap_or_else(|_| crate::core::memory::rel_type_catalog::RelTypeResolution {
            rel_type_id: Uuid::new_v4().to_string(),
            rel_type_norm: rel_type_norm.clone(),
            rel_type_raw: rel_type_norm.clone(),
        });
        rel_type_id = resolved.rel_type_id;
    }
    if rel_type_id.trim().is_empty() {
        rel_type_id = Uuid::new_v4().to_string();
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO rel_type (rel_type_id, canonical_name, status, created_at)
             VALUES (?, ?, 'provisional', CURRENT_TIMESTAMP)"
        )
        .bind(&rel_type_id)
        .bind(&rel_type_norm)
        .execute(&ctx.pool)
        .await;
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO rel_type_alias (alias, rel_type_id, confidence, status, created_at)
             VALUES (?, ?, 1.0, 'confirmed', CURRENT_TIMESTAMP)"
        )
        .bind(&rel_type_norm)
        .bind(&rel_type_id)
        .execute(&ctx.pool)
        .await;
    }
    // 1. Compute Signatures
    let scope_str = serde_json::to_string(&ctx.scope).unwrap_or_default();
    let (time_bucket_kind, time_bucket_value) = compute_time_bucket(&stmt.time_expr);
    let time_bucket_value_sig = time_bucket_value.clone().unwrap_or_default();

    let participant_ids: Vec<i64> = participants.iter().map(|(_, id)| *id).collect();
    let labels = fetch_entity_labels(&ctx.pool, &participant_ids).await.unwrap_or_default();
    let snippet = build_relation_snippet(
        &rel_type_norm,
        &participants,
        &labels,
        &stmt.polarity,
        stmt.direction.as_ref(),
    );
    if snippet.trim().is_empty() {
        return WriteResult::Error("EmptyEvidenceSnippet".to_string());
    }
    let sensitivity = detect_sensitivity(&snippet);
    if let Some(level) = sensitivity {
        if !phi_consent_allowed(&ctx.pool, ctx.conversation_id.as_deref()).await {
            let snippet_hash = hash_text(&snippet);
            log_phi_write_blocked(ctx, "rel", level, Some(&snippet_hash), Some(&rel_type_norm)).await;
            log_memory_write_blocked(
                ctx,
                "rel",
                "phi_blocked",
                json!({
                    "snippet_hash": snippet_hash,
                    "sensitivity": level.as_str(),
                    "rel_type": rel_type_norm,
                }),
            )
            .await;
            return WriteResult::Ignored("phi_blocked".to_string());
        }
    }
    
    let is_one = matches!(shape.cardinality_override, Some(Cardinality::One));
    if is_one && shape.anchor_roles.is_empty() {
        // Fail-open: allow write even if shape is missing anchor_roles.
        // Anchor signature will fall back to all participants.
    }
    
    let anchor_sig = crate::core::memory::canonical::compute_anchor_signature(&shape.anchor_roles, &canonical_participants, false);
    // Rel topic_key anchored (Spec 7.0)
    // (kind=rel, rel_type, scope, role_sig(all), anchor_sig)
    // Wait, role_sig is ALL participants. Topic key differentiates same anchors but different other roles?
    // Spec 7.0: "Rel topic_key (anchored): (kind=rel, rel_type, scope, ... role_signature... anchor_signature)"
    // Actually, usually topic_key is (rel_type + anchors). 
    // Standard: topic_key = "rel:<type>:<anchor_sig>"
    let topic_key = format!("rel:{}:{}", rel_type_id, anchor_sig); 
    
    // Signature hash for specific belief (includes ALL participants)
    let participants_canonical = serialize_participants(&canonical_participants);
    let participants_ordered = serialize_participants(&ordered_participants);
    let participants_id_sig = serialize_participant_ids(&canonical_participants);
    let mut sig_inputs = vec![
        ("rel_type_id".to_string(), rel_type_id.to_string()),
        ("participants".to_string(), participants_id_sig),
        ("arity".to_string(), canonical_participants.len().to_string()),
        ("scope".to_string(), scope_str.clone()),
        ("time_bucket_kind".to_string(), time_bucket_kind.clone()),
        ("time_bucket_value".to_string(), time_bucket_value_sig),
        ("polarity".to_string(), stmt.polarity.clone()),
    ];
    if let Some(direction) = direction_label(stmt.direction.as_ref()) {
        sig_inputs.push(("direction".to_string(), direction.to_string()));
    }
    let sig_refs: Vec<(&str, &str)> = sig_inputs
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let signature_hash = crate::core::memory::canonical::compute_signature_hash(&sig_refs);
    let episodic_on = episodic::episodic_enabled(&ctx.pool).await;
    
    // START TRANSACTION RETRY LOOP
    let mut attempts = 0;
    loop {
        attempts += 1;
        
        let result = async {
            let mut tx = match ctx.pool.begin().await {
                Ok(t) => t,
                Err(e) => return (WriteResult::Error(format!("Failed to start transaction: {}", e)), None),
            };
            
            // 2. Check Existing
            let existing: Option<sqlx::sqlite::SqliteRow> = sqlx::query("SELECT id, evidence_weight_total, confidence FROM ics_beliefs WHERE signature_hash = ? AND scope = ? AND polarity = ? AND status = 'active'")
                .bind(&signature_hash)
                .bind(&scope_str)
                .bind(&stmt.polarity)
                .fetch_optional(&mut *tx)
                .await
                .unwrap_or(None);

            if let Some(row) = existing {
                let id: i64 = row.get("id");
                let weight_total: f64 = row.get("evidence_weight_total");
                let current_confidence: f64 = row.try_get("confidence").unwrap_or(1.0);
                
                let weight = get_source_weight(ctx.source);
                let new_weight = weight_total as f32 + weight;
                let new_confidence = current_confidence.max(stmt.certainty.unwrap_or(1.0) as f64);
                
                // Add Evidence
                 let episodic_event_id = if episodic_on { Some(Uuid::new_v4().to_string()) } else { None };
                 let _ = sqlx::query("INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id) VALUES (?, ?, ?, ?, ?, ?)")
                    .bind(id)
                    .bind(serialize_source(ctx.source))
                    .bind(stmt.source_ref.as_ref().or(ctx.source_ref.as_ref()))
                    .bind(&snippet)
                    .bind(weight)
                    .bind(episodic_event_id.as_deref())
                    .execute(&mut *tx)
                    .await;
                    
                let _ = sqlx::query("UPDATE ics_beliefs SET evidence_weight_total = ?, confidence = ?, last_evidence_at = CURRENT_TIMESTAMP WHERE id = ?")
                    .bind(new_weight)
                    .bind(new_confidence)
                    .bind(id)
                    .execute(&mut *tx)
                    .await;
                    
                if let Err(e) = tx.commit().await {
                    return (WriteResult::Error(format!("Failed to commit update: {}", e)), None);
                }

                // Post-commit compaction (I5)
                let pool = ctx.pool.clone();
                let belief_id = id;
                spawn_compaction(pool, belief_id);

                let _ = crate::core::memory::attention::salience::recompute_salience_for_beliefs(
                    &ctx.pool,
                    &[id],
                )
                .await;

                return (WriteResult::Updated(id), episodic_event_id);
            }
            
            // 3. Cardinality Checks (ONE)
            if is_one {
                // Enforce: only one ACTIVE belief per topic_key (anchors)
                // If same anchors, supersede old one.
                // Difference is other participants or polarity.
                
                 let collision: Option<sqlx::sqlite::SqliteRow> = sqlx::query("SELECT id FROM ics_beliefs WHERE topic_key = ? AND scope = ? AND polarity = ? AND status = 'active'")
                    .bind(&topic_key)
                    .bind(&scope_str)
                    .bind(&stmt.polarity)
                    .fetch_optional(&mut *tx)
                    .await
                    .unwrap_or(None);
                    
                if let Some(row) = collision {
                    let old_id: i64 = row.get("id");
                     let _ = sqlx::query("UPDATE ics_beliefs SET status = 'inactive' WHERE id = ?")
                        .bind(old_id)
                        .execute(&mut *tx)
                        .await;
                }
            }
            
            // 4. Insert New
            let weight = get_source_weight(ctx.source);
            let certainty = stmt.certainty.unwrap_or(1.0);
            
            let layer = initial_layer_for_source(ctx.source);
            let res = sqlx::query("INSERT INTO ics_beliefs (kind, scope, polarity, layer, topic_key, signature_hash, evidence_weight_total, confidence, time_bucket_kind, time_bucket_value, observed_at, created_at) VALUES ('rel', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id")
                .bind(&scope_str)
                .bind(&stmt.polarity)
                .bind(layer)
                .bind(&topic_key)
                .bind(&signature_hash)
                .bind(weight)
                .bind(certainty)
                .bind(&time_bucket_kind)
                .bind(&time_bucket_value)
                .bind(ctx.now.to_rfc3339())
                .bind(ctx.now.to_rfc3339())
                .fetch_one(&mut *tx)
                .await;
                
            let id = match res {
                Ok(row) => row.get::<i64, _>("id"),
                Err(e) => {
                    let _ = tx.rollback().await;
                    return (WriteResult::Error(format!("Failed to insert rel belief: {}", e)), None);
                }
            };
            
            // Rel Belief Table
            let direction_str = direction_label(stmt.direction.as_ref());
            let _ = sqlx::query("INSERT INTO ics_rel_beliefs (belief_id, rel_type_id, rel_type, rel_type_norm, rel_type_raw, participants_canonical, participants_ordered, anchor_signature, direction) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)")
                .bind(id)
                .bind(&rel_type_id)
                .bind(&rel_type_norm)
                .bind(&rel_type_norm)
                .bind(rel_type_raw)
                .bind(&participants_canonical)
                .bind(&participants_ordered)
                .bind(&anchor_sig)
                .bind(direction_str)
                .execute(&mut *tx)
                .await;
                
            // Rel Participants Table (Role-indexed)
            for (role, pid) in &participants {
                let _ = sqlx::query("INSERT INTO ics_rel_participants (belief_id, entity_id, role) VALUES (?, ?, ?)")
                    .bind(id)
                    .bind(pid)
                    .bind(role)
                    .execute(&mut *tx)
                    .await;
                
                // Infer entity type from role name
                let inferred_type = match role.to_lowercase().as_str() {
                    "person" | "user" | "owner" | "author" | "creator" | "subject" | "actor"
                    | "parent" | "child" | "mother" | "father" | "daughter" | "son" | "spouse"
                    | "sibling" | "brother" | "sister" | "partner" | "husband" | "wife" => Some("person"),
                    "place" | "location" | "city" | "country" | "venue" => Some("place"),
                    "work" | "project" | "product" | "book" | "movie" | "song" | "company" => Some("work"),
                    "event" | "meeting" | "appointment" => Some("event"),
                    "concept" | "idea" | "topic" | "category" | "object" | "thing" | "item" => Some("concept"),
                    _ => None,
                };
                
                if let Some(entity_type) = inferred_type {
                    // Only update if entity_type is currently NULL (don't overwrite explicit types)
                    let _ = sqlx::query(
                        "UPDATE ics_entities SET entity_type = ? WHERE id = ? AND entity_type IS NULL"
                    )
                    .bind(entity_type)
                    .bind(pid)
                    .execute(&mut *tx)
                    .await;
                }
            }

            let roles = participants
                .iter()
                .map(|(role, _)| role.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let _ = sqlx::query("INSERT INTO ics_rel_fts(rowid, rel_type, roles) VALUES (?, ?, ?)")
                .bind(id)
                .bind(&rel_type_norm)
                .bind(&roles)
                .execute(&mut *tx)
                .await;
            
            // Evidence
            let episodic_event_id = if episodic_on { Some(Uuid::new_v4().to_string()) } else { None };
            let _ = sqlx::query("INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id) VALUES (?, ?, ?, ?, ?, ?)")
                .bind(id)
                .bind(serialize_source(ctx.source))
                .bind(stmt.source_ref.as_ref().or(ctx.source_ref.as_ref()))
                .bind(&snippet)
                .bind(weight)
                .bind(episodic_event_id.as_deref())
                .execute(&mut *tx)
                .await;
                
            if let Err(e) = tx.commit().await {
                return (WriteResult::Error(format!("Failed to commit new rel: {}", e)), None);
            }
            
            // Post-commit compaction (I5)
            let pool = ctx.pool.clone();
            let belief_id = id;
            spawn_compaction(pool, belief_id);

            // Â§7.6: Conflict detection (relations)
            let polarity = &stmt.polarity;
            let opposite_polarity = if polarity == "assert" { "deny" } else { "assert" };

            let conflicting_ids: Vec<i64> = sqlx::query_scalar(
                "SELECT id FROM ics_beliefs WHERE topic_key = ? AND scope = ? AND status = 'active' AND polarity = ? AND id != ?"
            )
            .bind(&topic_key)
            .bind(&scope_str)
            .bind(opposite_polarity)
            .bind(id)
            .fetch_all(&ctx.pool)
            .await
            .unwrap_or_default();

            let collision_ids: Vec<i64> = sqlx::query_scalar(
                "SELECT id FROM ics_beliefs WHERE topic_key = ? AND scope = ? AND status = 'active' AND polarity = ? AND id != ? AND signature_hash != ?"
            )
            .bind(&topic_key)
            .bind(&scope_str)
            .bind(&stmt.polarity)
            .bind(id)
            .bind(&signature_hash)
            .fetch_all(&ctx.pool)
            .await
            .unwrap_or_default();

            let mut conflict_set_id: Option<i64> = None;
            let mut conflict_candidates: Vec<i64> = Vec::new();
            conflict_candidates.extend(conflicting_ids.iter().cloned());
            conflict_candidates.extend(collision_ids.iter().cloned());
            conflict_candidates.sort();
            conflict_candidates.dedup();
            if !conflict_candidates.is_empty() {
                let existing_set: Option<i64> = sqlx::query_scalar(
                    "SELECT id FROM ics_conflict_sets WHERE topic_key = ? AND status = 'open'"
                )
                .bind(&topic_key)
                .fetch_optional(&ctx.pool)
                .await
                .ok()
                .flatten();

                let set_id = match existing_set {
                    Some(sid) => sid,
                    None => {
                        let row = sqlx::query("INSERT INTO ics_conflict_sets (topic_key, status) VALUES (?, 'open') RETURNING id")
                            .bind(&topic_key)
                            .fetch_one(&ctx.pool)
                            .await;
                        match row {
                            Ok(r) => r.get("id"),
                            Err(_) => 0,
                        }
                    }
                };

                if set_id != 0 {
                    conflict_set_id = Some(set_id);
                }

                for cid in &conflict_candidates {
                    let _ = sqlx::query("INSERT OR IGNORE INTO ics_belief_links (from_id, to_id, link_type) VALUES (?, ?, 'contradicts')")
                        .bind(id)
                        .bind(cid)
                        .execute(&ctx.pool)
                        .await;
                    let _ = sqlx::query("INSERT OR IGNORE INTO ics_belief_links (from_id, to_id, link_type) VALUES (?, ?, 'contradicts')")
                        .bind(cid)
                        .bind(id)
                        .execute(&ctx.pool)
                        .await;

                    let _ = sqlx::query("INSERT OR IGNORE INTO ics_conflict_set_members (conflict_set_id, belief_id) VALUES (?, ?)")
                        .bind(set_id)
                        .bind(id)
                        .execute(&ctx.pool)
                        .await;
                    let _ = sqlx::query("INSERT OR IGNORE INTO ics_conflict_set_members (conflict_set_id, belief_id) VALUES (?, ?)")
                        .bind(set_id)
                        .bind(cid)
                        .execute(&ctx.pool)
                        .await;
                }
            }

            if let Some(conflict_set_id) = conflict_set_id {
                (
                    WriteResult::Conflict {
                        belief_id: id,
                        conflict_set_id,
                    },
                    episodic_event_id,
                )
            } else {
                (WriteResult::Inserted(id), episodic_event_id)
            }
        }.await;
        
        match result {
            (WriteResult::Error(e), _) if (e.contains("database is locked") || e.contains("code: 5")) && attempts <= 5 => {
                 tokio::time::sleep(std::time::Duration::from_millis(50 * (1 << attempts))).await;
                 continue;
            },
            (WriteResult::Inserted(id), Some(event_id)) => {
                let source_ref = stmt.source_ref.as_deref().or(ctx.source_ref.as_deref());
                emit_memory_event(
                    ctx,
                    &event_id,
                    "memory_write_rel",
                    "inserted",
                    &snippet,
                    &scope_str,
                    source_ref,
                    Some(id),
                    None,
                    &stmt.polarity,
                )
                .await;
                if let Some(level) = sensitivity {
                    upsert_memory_sensitivity(&ctx.pool, id, "rel", level).await;
                }
                cache::bump_cache_version();
                return WriteResult::Inserted(id);
            }
            (WriteResult::Inserted(id), None) => {
                let payload = format!("memory_write_rel:{}:{}", id, snippet);
                log_semantic_write(ctx, "memory_writer_evidence", &payload).await;
                if let Some(level) = sensitivity {
                    upsert_memory_sensitivity(&ctx.pool, id, "rel", level).await;
                }
                cache::bump_cache_version();
                return WriteResult::Inserted(id);
            }
            (WriteResult::Updated(id), Some(event_id)) => {
                let source_ref = stmt.source_ref.as_deref().or(ctx.source_ref.as_deref());
                emit_memory_event(
                    ctx,
                    &event_id,
                    "memory_write_rel",
                    "updated",
                    &snippet,
                    &scope_str,
                    source_ref,
                    Some(id),
                    None,
                    &stmt.polarity,
                )
                .await;
                if let Some(level) = sensitivity {
                    upsert_memory_sensitivity(&ctx.pool, id, "rel", level).await;
                }
                cache::bump_cache_version();
                return WriteResult::Updated(id);
            }
            (WriteResult::Updated(id), None) => {
                let payload = format!("memory_write_rel:{}:{}", id, snippet);
                log_semantic_write(ctx, "memory_writer_evidence", &payload).await;
                if let Some(level) = sensitivity {
                    upsert_memory_sensitivity(&ctx.pool, id, "rel", level).await;
                }
                cache::bump_cache_version();
                return WriteResult::Updated(id);
            }
            (WriteResult::Conflict { belief_id, conflict_set_id }, Some(event_id)) => {
                let source_ref = stmt.source_ref.as_deref().or(ctx.source_ref.as_deref());
                emit_memory_event(
                    ctx,
                    &event_id,
                    "memory_write_rel",
                    "conflict",
                    &snippet,
                    &scope_str,
                    source_ref,
                    Some(belief_id),
                    None,
                    &stmt.polarity,
                )
                .await;
                if let Some(level) = sensitivity {
                    upsert_memory_sensitivity(&ctx.pool, belief_id, "rel", level).await;
                }
                cache::bump_cache_version();
                return WriteResult::Conflict { belief_id, conflict_set_id };
            }
            (WriteResult::Conflict { belief_id, conflict_set_id }, None) => {
                let payload = format!("memory_write_rel:{}:{}", belief_id, snippet);
                log_semantic_write(ctx, "memory_writer_evidence", &payload).await;
                if let Some(level) = sensitivity {
                    upsert_memory_sensitivity(&ctx.pool, belief_id, "rel", level).await;
                }
                cache::bump_cache_version();
                return WriteResult::Conflict { belief_id, conflict_set_id };
            }
            (other, _) => return other,
        }
    }
}

async fn fetch_entity_labels(pool: &SqlitePool, ids: &[i64]) -> Result<HashMap<i64, String>, String> {
    if ids.is_empty() {
        return Ok(HashMap::new());
    }

    let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let query = format!("SELECT id, label FROM ics_entities WHERE id IN ({})", placeholders);
    let mut q = sqlx::query(&query);
    for id in ids {
        q = q.bind(id);
    }
    let rows = q.fetch_all(pool).await.map_err(|e| e.to_string())?;
    let mut labels = HashMap::new();
    for row in rows {
        let id: i64 = row.get("id");
        let label: String = row.get("label");
        labels.insert(id, label);
    }
    Ok(labels)
}

fn direction_label(direction: Option<&RelDirection>) -> Option<&'static str> {
    match direction {
        Some(RelDirection::Directed) => Some("directed"),
        Some(RelDirection::Bidirectional) => Some("bidirectional"),
        None => None,
    }
}

fn build_relation_snippet(
    rel_type: &str,
    participants: &[(String, i64)],
    labels: &HashMap<i64, String>,
    polarity: &str,
    direction: Option<&RelDirection>,
) -> String {
    let parts = participants
        .iter()
        .map(|(role, id)| {
            let label = labels
                .get(id)
                .cloned()
                .unwrap_or_else(|| format!("entity_{}", id));
            (role.clone(), label)
        })
        .collect::<Vec<_>>();

    snippets::render_rel_snippet_with_direction(rel_type, &parts, polarity, direction_label(direction))
}


fn get_source_weight(s: SourceType) -> f32 {
    match s {
        SourceType::User => SOURCE_WEIGHT_USER,
        SourceType::Tool => SOURCE_WEIGHT_TOOL,
        SourceType::System => SOURCE_WEIGHT_SYSTEM,
        SourceType::Inference => SOURCE_WEIGHT_INFERENCE,
    }
}

async fn compaction_allowed(pool: &SqlitePool) -> bool {
    let mode: Option<String> = sqlx::query_scalar(
        "SELECT mode FROM system_controls WHERE subsystem_id = ?",
    )
    .bind("memory_consolidation")
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    let mode = mode.unwrap_or_else(|| {
        system_controls::default_mode_for("memory_consolidation")
            .unwrap_or("normal")
            .to_string()
    });
    !(system_controls::mode_is_off(&mode) || system_controls::mode_is_degraded(&mode))
}

fn spawn_compaction(pool: SqlitePool, belief_id: i64) {
    tokio::spawn(async move {
        if compaction_allowed(&pool).await {
            let _ = crate::core::memory::consolidation::compaction::compact_evidence(belief_id, &pool).await;
        }
    });
}

fn initial_layer_for_source(source: SourceType) -> &'static str {
    match source {
        SourceType::User | SourceType::Tool => "episodic",
        SourceType::System | SourceType::Inference => "working",
    }
}

fn is_internal_state_ref(source_ref: &str) -> bool {
    let lowered = source_ref.to_lowercase();
    lowered.starts_with("telemetry.")
        || lowered.contains("telemetry")
        || lowered.contains("controller")
        || lowered.contains("gate")
        || lowered.contains("self_state")
        || lowered.contains("self_model")
        || lowered.contains("runtime_state")
}

fn contains_diagnostic_marker(value: &str) -> bool {
    let lowered = value.to_lowercase();
    let markers = [
        "telemetry",
        "tool manifest",
        "tool list",
        "controller state",
        "kv memory",
        "prompt hash",
        "run_id",
        "trace_id",
        "timestamp",
        "latency",
        "module_status",
        "system log",
    ];
    markers.iter().any(|marker| lowered.contains(marker))
}

fn serialize_source(s: SourceType) -> String {
    match s {
        SourceType::User => "user".to_string(),
        SourceType::Tool => "tool".to_string(),
        SourceType::System => "system".to_string(),
        SourceType::Inference => "inference".to_string(),
    }
}

fn hash_payload(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

async fn log_semantic_write(ctx: &WriteContext, reason: &str, payload: &str) {
    let hash = hash_payload(payload);
    let db = Db { pool: ctx.pool.clone() };
    let _ = db
        .log_memory_write(
            ctx.conversation_id.as_deref(),
            "semantic",
            "memory_writer",
            reason,
            None,
            None,
            Some(&hash),
            None,
            None,
        )
        .await;
}

async fn emit_memory_event(
    ctx: &WriteContext,
    event_id: &str,
    event_type: &str,
    status: &str,
    snippet: &str,
    scope_str: &str,
    source_ref: Option<&str>,
    belief_id: Option<i64>,
    entity_id: Option<i64>,
    polarity: &str,
) {
    let source_type = serialize_source(ctx.source);
    let _ = episodic::emit_episodic_event_with_id(
        &ctx.pool,
        event_id,
        event_type,
        json!({
            "status": status,
            "summary_snippet": snippet,
            "belief_id": belief_id,
            "entity_id": entity_id,
            "scope": scope_str,
            "polarity": polarity,
            "source_ref": source_ref,
        }),
        None,
        None,
        ctx.conversation_id.as_deref(),
        Some(scope_str),
        &source_type,
        source_ref,
        belief_id,
        None,
    )
    .await;

    if let Some(belief_id) = belief_id {
        let payload = format!("{}:{}:{}", event_type, belief_id, snippet);
        log_semantic_write(ctx, "memory_writer_evidence", &payload).await;
    }
}
