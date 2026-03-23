use chrono::Utc;
use sqlx::{Row, SqlitePool};
use std::collections::HashMap;
use std::sync::Arc;

use crate::core::episodic;
use crate::core::memory::attention::working_set;
use crate::core::memory::canonical::{canonicalize_role_token, compute_anchor_signature, compute_topic_key_fact, normalize_rel_type, normalize_role_token};
use crate::core::memory::compiler::{self, ClarifyCandidate, CompileContext};
use crate::core::memory::dsl::{self, DslStatement, Ref, RelDirection};
use crate::core::memory::rel_type_catalog;
use crate::core::memory::resolver::{self, ResolveContext, ResolutionResult};
use crate::core::memory::scope::parse_scope;
use crate::core::memory::types::{Cardinality, ClaimOutcome, Scope, SourceType};
use crate::core::model_client::ModelClient;
use crate::core::memory::api::MemoryApi;
use crate::core::system_log;

const SUMMARY_PROMOTION_WINDOW_DAYS: i64 = 14;
const SUMMARY_PROMOTION_MIN_OCCURRENCES: usize = 2;
const SUMMARY_PROMOTION_MIN_PARTS: usize = 3;
const SUMMARY_PROMOTION_MAX_BATCH: usize = 5;
const CLAIM_OUTCOME_BUFFER_LIMIT: usize = 50;

pub struct ClaimPromotionResult {
    pub claim_id: String,
    pub written_ids: Vec<i64>,
    pub conflict_ids: Vec<i64>,
}

pub(crate) async fn record_claim_outcome(
    pool: &SqlitePool,
    claim_id: &str,
    status: &str,
    reason: Option<&str>,
) {
    let row = sqlx::query("SELECT scope, session_id FROM memory_claims WHERE id = ?")
        .bind(claim_id)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();

    let (scope, session_id) = if let Some(row) = row {
        let scope: String = row.get("scope");
        let session_id: Option<String> = row.try_get("session_id").ok();
        (scope, session_id)
    } else {
        ("\"global\"".to_string(), None)
    };

    let outcome = ClaimOutcome {
        claim_id: claim_id.to_string(),
        status: status.to_string(),
        reason: reason.map(|r| r.to_string()),
        scope,
        session_id,
        created_at: Utc::now().to_rfc3339(),
    };

    let _ = append_claim_outcome(pool, outcome).await;
}

async fn append_claim_outcome(pool: &SqlitePool, outcome: ClaimOutcome) -> Result<(), String> {
    let key = "last_claim_outcomes";
    let existing: Option<String> = sqlx::query_scalar("SELECT value FROM kv_store WHERE key = ?")
        .bind(key)
        .fetch_optional(pool)
        .await
        .map_err(|e| e.to_string())?
        .flatten();

    let mut outcomes: Vec<ClaimOutcome> = existing
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default();
    outcomes.push(outcome);
    if outcomes.len() > CLAIM_OUTCOME_BUFFER_LIMIT {
        let overflow = outcomes.len() - CLAIM_OUTCOME_BUFFER_LIMIT;
        outcomes.drain(0..overflow);
    }

    let serialized = serde_json::to_string(&outcomes).unwrap_or_else(|_| "[]".to_string());
    sqlx::query(
        "INSERT INTO kv_store (key, value, updated_at)
         VALUES (?, ?, CURRENT_TIMESTAMP)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP"
    )
    .bind(key)
    .bind(serialized)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    Ok(())
}

fn summarize_candidates(candidates_json: &str, limit: usize) -> Option<String> {
    let candidates: Vec<ClarifyCandidate> = serde_json::from_str(candidates_json).unwrap_or_default();
    if candidates.is_empty() {
        return None;
    }
    let mut parts = Vec::new();
    for candidate in candidates.iter().take(limit) {
        let context = candidate.context.trim();
        if context.is_empty() {
            parts.push(format!("{} (#{})", candidate.label, candidate.entity_id));
        } else {
            parts.push(format!(
                "{} (#{}; {})",
                candidate.label,
                candidate.entity_id,
                context
            ));
        }
    }
    let mut summary = parts.join(", ");
    if candidates.len() > limit {
        summary.push_str(&format!(" +{} more", candidates.len() - limit));
    }
    Some(summary)
}

pub async fn create_summary_claim(
    pool: &SqlitePool,
    summary: &str,
    conversation_id: Option<&str>,
    source_ref: Option<&str>,
    episodic_event_id: Option<&str>,
) -> Result<Option<String>, String> {
    if !compiler::memory_claims_enabled(pool).await {
        return Ok(None);
    }

    let summary = summary.trim();
    if summary.is_empty() {
        return Ok(None);
    }

    let scope = if let Some(id) = conversation_id {
        Scope::Context(id.to_string())
    } else {
        Scope::Global
    };
    let scope_str = serde_json::to_string(&scope).unwrap_or_else(|_| "\"global\"".to_string());
    let claim_text = build_summary_dsl(summary, conversation_id);
    let session_id = conversation_id.map(str::trim).filter(|id| !id.is_empty());

    let claim_id = uuid::Uuid::new_v4().to_string();
    let _ = sqlx::query(
        "INSERT INTO memory_claims (id, kind, scope, session_id, claim_text, status, source_type, source_ref, episodic_event_id, created_at, updated_at)
         VALUES (?, 'summary', ?, ?, ?, 'pending', 'system', ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
    )
    .bind(&claim_id)
    .bind(&scope_str)
    .bind(session_id)
    .bind(&claim_text)
    .bind(source_ref)
    .bind(episodic_event_id)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    record_claim_outcome(pool, &claim_id, "pending", Some("pending")).await;

    let _ = episodic::emit_episodic_event(
        pool,
        "memory_claim_created",
        serde_json::json!({
            "status": "pending",
            "summary_snippet": summary,
            "claim_id": claim_id.as_str(),
        }),
        None,
        None,
        conversation_id,
        Some(&scope_str),
        "system",
        source_ref,
        None,
        None,
    )
    .await;

    Ok(Some(claim_id))
}

pub async fn promote_claim(
    pool: &SqlitePool,
    model_client: Option<Arc<ModelClient>>,
    claim_id: &str,
) -> Result<ClaimPromotionResult, String> {
    let row = sqlx::query(
        "SELECT id, status, scope, session_id, claim_text, source_type, source_ref, rel_type_id
         FROM memory_claims WHERE id = ?"
    )
    .bind(claim_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?
    .ok_or_else(|| "Memory claim not found".to_string())?;

    let status: String = row.get("status");
    if status == "promoted" {
        return Ok(ClaimPromotionResult {
            claim_id: claim_id.to_string(),
            written_ids: Vec::new(),
            conflict_ids: Vec::new(),
        });
    }
    if status != "pending" && status != "evaluating" {
        return Err(format!("Claim status '{}' is not promotable", status));
    }

    let scope_raw: String = row.get("scope");
    let claim_session_id: Option<String> = row.try_get("session_id").ok();
    let claim_text: String = row.get("claim_text");
    let source_type_raw: String = row.get("source_type");
    let source_ref: Option<String> = row.try_get("source_ref").ok();
    let claim_rel_type_id: Option<String> = row.try_get("rel_type_id").ok();

    if contains_diagnostic_marker(&claim_text) {
        let _ = system_log::log_event(
            pool,
            None,
            "warn",
            "memory_policy",
            None,
            None,
            serde_json::json!({
                "event": "memory_claim_rejected_telemetry",
                "claim_id": claim_id,
                "source_type": source_type_raw,
                "source_ref": source_ref.clone(),
            }),
        )
        .await;
        return Err("ClaimRejected: telemetry_markers".to_string());
    }

    let scope = parse_scope_from_claim(&scope_raw).unwrap_or(Scope::Global);
    let source_type = parse_source_type(&source_type_raw);
    let trimmed_claim = claim_text.trim_start();
    let parsed_stmt = serde_json::from_str::<DslStatement>(&claim_text).ok();
    if parsed_stmt.is_none() && trimmed_claim.starts_with('{') {
        return Err("InvalidClaimJson: failed to parse claim_text as JSON".to_string());
    }
    let dsl_block = claim_text_to_dsl(&claim_text)?;
    if dsl_block.trim().is_empty() && parsed_stmt.is_none() {
        return Err("Claim text is empty".to_string());
    }

    let promotion_session = claim_session_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(|id| id.to_string())
        .unwrap_or_else(|| "claim_promotion".to_string());
    let api = MemoryApi::new(pool.clone(), model_client.clone(), promotion_session.clone()).await;
    let ctx = CompileContext {
        pool: pool.clone(),
        model_client,
        session_id: promotion_session,
        scope,
        source: source_type,
        source_ref: source_ref.clone(),
        now: Utc::now(),
        embedding_config: api.embedding_config().cloned(),
        skip_claims: true,
        allow_ambiguous_user_refs: false,
    };

    let result = if let Some(mut stmt) = parsed_stmt {
        if let DslStatement::Rel(ref mut rel_stmt) = stmt {
            let canonicalization_on = compiler::relation_canonicalization_enabled(pool).await;
            let mut has_rel_type_id = rel_stmt
                .rel_type_id
                .as_deref()
                .map(|id| !id.trim().is_empty())
                .unwrap_or(false);
            if !has_rel_type_id {
                if let Some(ref rel_type_id) = claim_rel_type_id {
                    if !rel_type_id.trim().is_empty() {
                        rel_stmt.rel_type_id = Some(rel_type_id.clone());
                        has_rel_type_id = true;
                    }
                }
            }
            if has_rel_type_id {
                if let Some(ref rel_type_id) = rel_stmt.rel_type_id {
                    if let Ok(resolved) = rel_type_catalog::resolve_rel_type_from_id(
                        pool,
                        rel_type_id,
                        &rel_stmt.rel_type,
                        canonicalization_on,
                    )
                    .await
                    {
                        rel_stmt.rel_type_id = Some(resolved.rel_type_id.clone());
                        let res = sqlx::query(
                            "UPDATE memory_claims SET rel_type_raw = ?, rel_type_norm = ?, rel_type_id = ? WHERE id = ?"
                        )
                        .bind(&resolved.rel_type_raw)
                        .bind(&resolved.rel_type_norm)
                        .bind(&resolved.rel_type_id)
                        .bind(claim_id)
                        .execute(pool)
                        .await
                        .map_err(|e| e.to_string())?;
                        if res.rows_affected() == 0 {
                            return Err("ClaimUpdateFailed: rel_type_id not persisted".to_string());
                        }
                    }
                }
            } else {
                let alias_map = if canonicalization_on {
                    compiler::load_role_alias_map(pool).await
                } else {
                    HashMap::new()
                };
                let roles_seen: Vec<String> = rel_stmt
                    .participants
                    .iter()
                    .map(|(role, _)| {
                        if canonicalization_on {
                            canonicalize_role_token(role, &alias_map)
                        } else {
                            normalize_role_token(role)
                        }
                    })
                    .collect();
                if let Ok(resolved) = rel_type_catalog::resolve_rel_type(
                    pool,
                    &rel_stmt.rel_type,
                    &roles_seen,
                    canonicalization_on,
                )
                .await
                {
                    rel_stmt.rel_type_id = Some(resolved.rel_type_id.clone());
                    let res = sqlx::query(
                        "UPDATE memory_claims SET rel_type_raw = ?, rel_type_norm = ?, rel_type_id = ? WHERE id = ?"
                    )
                    .bind(&resolved.rel_type_raw)
                    .bind(&resolved.rel_type_norm)
                    .bind(&resolved.rel_type_id)
                    .bind(claim_id)
                    .execute(pool)
                    .await
                    .map_err(|e| e.to_string())?;
                    if res.rows_affected() == 0 {
                        return Err("ClaimUpdateFailed: rel_type_id not persisted".to_string());
                    }
                }
            }
        }
        compiler::compile_parsed(vec![stmt], &claim_text, ctx).await
    } else {
        compiler::compile(&dsl_block, ctx).await
    };
    if !result.errors.is_empty() {
        return Err(format!("Claim promotion failed: {:?}", result.errors));
    }
    if result.pending_clarify.is_some() {
        return Err("Claim promotion requires clarification".to_string());
    }

    sqlx::query(
        "UPDATE memory_claims
         SET status = 'promoted',
             evaluated_at = CURRENT_TIMESTAMP,
             decision_reason = 'promoted',
             conflict_topic_key = NULL,
             conflict_reason = NULL,
             updated_at = CURRENT_TIMESTAMP
         WHERE id = ?"
    )
    .bind(claim_id)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    record_claim_outcome(pool, claim_id, "promoted", Some("promoted")).await;

    Ok(ClaimPromotionResult {
        claim_id: claim_id.to_string(),
        written_ids: result.written_ids,
        conflict_ids: result.conflict_ids,
    })
}

pub async fn auto_promote_summary_claims(
    pool: &SqlitePool,
    model_client: Option<Arc<ModelClient>>,
) -> Result<usize, String> {
    if !compiler::memory_claims_enabled(pool).await {
        return Ok(0);
    }

    let cutoff = format!("-{} days", SUMMARY_PROMOTION_WINDOW_DAYS);
    let rows = sqlx::query(
        "SELECT id, claim_text, scope, created_at
         FROM memory_claims
         WHERE status = 'pending' AND kind = 'summary'
           AND created_at >= datetime('now', ?)
         ORDER BY created_at ASC"
    )
    .bind(&cutoff)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    if rows.is_empty() {
        return Ok(0);
    }

    let mut grouped: HashMap<(String, String), Vec<String>> = HashMap::new();
    for row in rows {
        let claim_text: String = row.get("claim_text");
        let scope: String = row.get("scope");
        let id: String = row.get("id");
        grouped.entry((claim_text, scope)).or_default().push(id);
    }

    let mut promoted = 0;
    for ((claim_text, _scope), ids) in grouped {
        if promoted >= SUMMARY_PROMOTION_MAX_BATCH {
            break;
        }
        if ids.len() < SUMMARY_PROMOTION_MIN_OCCURRENCES {
            continue;
        }
        if summary_parts_count(&claim_text) < SUMMARY_PROMOTION_MIN_PARTS {
            continue;
        }

        let promote_id = match ids.last() {
            Some(id) => id.clone(),
            None => continue,
        };

        if let Ok(result) = promote_claim(pool, model_client.clone(), &promote_id).await {
            promoted += 1;
            let linked_belief_id = result.written_ids.first().copied();
            let _ = episodic::emit_claim_status_event(
                pool,
                &promote_id,
                "promoted",
                linked_belief_id,
                "system",
                Some("compaction"),
                Some("promoted"),
                None,
                None,
            )
            .await;

            let superseded: Vec<String> = ids.into_iter().filter(|id| id != &promote_id).collect();
            if !superseded.is_empty() {
                let placeholders = superseded.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                let query = format!(
                    "UPDATE memory_claims SET status = 'superseded', updated_at = CURRENT_TIMESTAMP WHERE id IN ({})",
                    placeholders
                );
                let mut q = sqlx::query(&query);
                for id in &superseded {
                    q = q.bind(id);
                }
                let _ = q.execute(pool).await;
            }
        }
    }

    Ok(promoted)
}

pub async fn evaluate_pending_claims(
    pool: &SqlitePool,
    model_client: Option<Arc<ModelClient>>,
    limit: usize,
) -> Result<usize, String> {
    if !compiler::memory_claims_enabled(pool).await {
        return Ok(0);
    }

    let limit = limit.max(1).min(200) as i64;
    let rows = sqlx::query(
        "SELECT id FROM memory_claims
         WHERE status = 'pending' AND kind IN ('fact', 'rel')
         ORDER BY created_at ASC
         LIMIT ?"
    )
    .bind(limit)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    let mut processed = 0;
    for row in rows {
        let claim_id: String = row.get("id");
        if evaluate_claim(pool, model_client.clone(), &claim_id).await.is_ok() {
            processed += 1;
        }
    }

    Ok(processed)
}

pub async fn evaluate_claim(
    pool: &SqlitePool,
    model_client: Option<Arc<ModelClient>>,
    claim_id: &str,
) -> Result<(), String> {
    if !compiler::memory_claims_enabled(pool).await {
        return Ok(());
    }

    let updated = sqlx::query(
        "UPDATE memory_claims
         SET status = 'evaluating', updated_at = CURRENT_TIMESTAMP
         WHERE id = ? AND status = 'pending'"
    )
    .bind(claim_id)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    if updated.rows_affected() == 0 {
        return Err(format!("Claim '{}' is not pending", claim_id));
    }

    let row = sqlx::query(
        "SELECT kind, scope, session_id, claim_text, source_type, source_ref
         FROM memory_claims WHERE id = ?"
    )
    .bind(claim_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?
    .ok_or_else(|| "Memory claim not found".to_string())?;

    let scope_raw: String = row.get("scope");
    let claim_session_id: Option<String> = row.try_get("session_id").ok();
    let claim_text: String = row.get("claim_text");
    let source_type_raw: String = row.get("source_type");
    let source_ref: Option<String> = row.try_get("source_ref").ok();

    let stmt = match parse_claim_statement(&claim_text) {
        Ok(stmt) => stmt,
        Err(err) => {
            finalize_claim(
                pool,
                claim_id,
                "rejected",
                Some(&format!("ParseError: {}", err)),
                None,
                None,
                source_type_str(&parse_source_type(&source_type_raw)),
                source_ref.as_deref(),
                None,
            )
            .await?;
            return Ok(());
        }
    };

    let (scope_expr, stmt_source_ref) = match &stmt {
        DslStatement::Fact(f) => (f.scope_expr.as_deref(), f.source_ref.as_deref()),
        DslStatement::Rel(r) => (r.scope_expr.as_deref(), r.source_ref.as_deref()),
    };

    let effective_scope = if let Some(scope_raw) = scope_expr {
        match parse_scope(scope_raw) {
            Ok(scope) => scope,
            Err(e) => {
                finalize_claim(
                    pool,
                    claim_id,
                    "rejected",
                    Some(&format!("InvalidScope: {}", e.0)),
                    None,
                    None,
                    source_type_str(&parse_source_type(&source_type_raw)),
                    source_ref.as_deref(),
                    None,
                )
                .await?;
                return Ok(());
            }
        }
    } else {
        parse_scope_from_claim(&scope_raw).unwrap_or(Scope::Global)
    };

    let scope_str = serde_json::to_string(&effective_scope).unwrap_or_default();
    let effective_source_ref = stmt_source_ref
        .map(|s| s.to_string())
        .or_else(|| source_ref.clone());
    let session_id = claim_session_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(|id| id.to_string())
        .unwrap_or_else(|| session_id_from_scope(&effective_scope));
    let source_type = parse_source_type(&source_type_raw);

    let anchor_entity_ids = resolver::get_working_set_anchors(pool, 10)
        .await
        .unwrap_or_default();
    let resolve_ctx = ResolveContext {
        pool: pool.clone(),
        session_id: session_id.clone(),
        expected_type: None,
        anchor_entity_ids,
    };

    match stmt {
        DslStatement::Fact(fact) => {
            evaluate_fact_claim(
                pool,
                model_client,
                claim_id,
                fact,
                &scope_str,
                &session_id,
                &resolve_ctx,
                &source_type,
                effective_source_ref.as_deref(),
            )
            .await
        }
        DslStatement::Rel(rel) => {
            evaluate_rel_claim(
                pool,
                model_client,
                claim_id,
                rel,
                &scope_str,
                &effective_scope,
                &session_id,
                &resolve_ctx,
                &source_type,
                effective_source_ref.as_deref(),
            )
            .await
        }
    }
}

async fn evaluate_fact_claim(
    pool: &SqlitePool,
    model_client: Option<Arc<ModelClient>>,
    claim_id: &str,
    fact: dsl::FactStmt,
    scope_str: &str,
    session_id: &str,
    resolve_ctx: &ResolveContext,
    source_type: &SourceType,
    source_ref: Option<&str>,
) -> Result<(), String> {
    let original_dsl = dsl_statement_to_line(&DslStatement::Fact(fact.clone()));
    let value_trimmed = fact.value.trim();
    if value_trimmed.starts_with('#') || value_trimmed.starts_with('$') {
        finalize_claim(
            pool,
            claim_id,
            "rejected",
            Some("InvalidFactValue: entity refs not allowed in fact values"),
            None,
            None,
            source_type_str(source_type),
            source_ref,
            None,
        )
        .await?;
        return Ok(());
    }

    let subject_resolution = match resolver::resolve_ref(&fact.subject, resolve_ctx).await {
        Ok(resolution) => resolution,
        Err(e) => {
            finalize_claim(
                pool,
                claim_id,
                "rejected",
                Some(&format!("ResolutionError: {}", e)),
                None,
                None,
                source_type_str(source_type),
                source_ref,
                None,
            )
            .await?;
            return Ok(());
        }
    };

    match subject_resolution {
        ResolutionResult::Resolved(subject_id) => {
            let topic_key = compute_topic_key_fact(subject_id, &fact.key);
            let conflict = match has_conflict(pool, &topic_key, scope_str, &fact.polarity).await {
                Ok(conflict) => conflict,
                Err(e) => {
                    finalize_claim(
                        pool,
                        claim_id,
                        "rejected",
                        Some(&format!("ConflictCheckFailed: {}", e)),
                        None,
                        None,
                        source_type_str(source_type),
                        source_ref,
                        None,
                    )
                    .await?;
                    return Ok(());
                }
            };
            if conflict {
                finalize_claim(
                    pool,
                    claim_id,
                    "conflict",
                    Some("ConflictDetected"),
                    Some(&topic_key),
                    Some("opposite_polarity_active"),
                    source_type_str(source_type),
                    source_ref,
                    None,
                )
                .await?;
                return Ok(());
            }

            let promotion = match promote_claim(pool, model_client.clone(), claim_id).await {
                Ok(promotion) => promotion,
                Err(e) => {
                    finalize_claim(
                        pool,
                        claim_id,
                        "rejected",
                        Some(&format!("PromotionFailed: {}", e)),
                        None,
                        None,
                        source_type_str(source_type),
                        source_ref,
                        None,
                    )
                    .await?;
                    return Ok(());
                }
            };
            if !promotion.written_ids.is_empty() {
                update_working_set_for_beliefs(pool, &promotion.written_ids).await?;
            }
            let linked_belief_id = promotion.written_ids.first().copied();
            let _ = episodic::emit_claim_status_event(
                pool,
                claim_id,
                "promoted",
                linked_belief_id,
                source_type_str(source_type),
                source_ref,
                Some("promoted"),
                None,
                None,
            )
            .await;
            Ok(())
        }
        ResolutionResult::Ambiguous(pending) => {
            let ref_text = format!("{:?}", &fact.subject);
            let pending_id = create_pending_clarify_for_claim(
                pool,
                claim_id,
                session_id.to_string(),
                &original_dsl,
                &ref_text,
                &pending.candidates_json,
            )
            .await?;
            let candidates_summary = summarize_candidates(&pending.candidates_json, 3);
            let reason = match candidates_summary {
                Some(summary) => format!("AmbiguousRef: {} | Candidates: {}", ref_text, summary),
                None => format!("AmbiguousRef: {}", ref_text),
            };
            finalize_claim(
                pool,
                claim_id,
                "needs_clarify",
                Some(&reason),
                None,
                None,
                source_type_str(source_type),
                source_ref,
                Some(pending_id),
            )
            .await
        }
        ResolutionResult::NewEntity(label) => {
            if claim_allows_entity_create(source_type) {
                let promotion = match promote_claim(pool, model_client.clone(), claim_id).await {
                    Ok(promotion) => promotion,
                    Err(e) => {
                        finalize_claim(
                            pool,
                            claim_id,
                            "rejected",
                            Some(&format!("PromotionFailed: {}", e)),
                            None,
                            None,
                            source_type_str(source_type),
                            source_ref,
                            None,
                        )
                        .await?;
                        return Ok(());
                    }
                };
                if !promotion.written_ids.is_empty() {
                    update_working_set_for_beliefs(pool, &promotion.written_ids).await?;
                }
                let linked_belief_id = promotion.written_ids.first().copied();
                let _ = episodic::emit_claim_status_event(
                    pool,
                    claim_id,
                    "promoted",
                    linked_belief_id,
                    source_type_str(source_type),
                    source_ref,
                    Some("promoted"),
                    None,
                    None,
                )
                .await;
                Ok(())
            } else {
                let ref_text = label.trim().to_string();
                let pending_id = create_pending_clarify_for_claim(
                    pool,
                    claim_id,
                    session_id.to_string(),
                    &original_dsl,
                    &ref_text,
                    "[]",
                )
                .await?;
                finalize_claim(
                    pool,
                    claim_id,
                    "needs_clarify",
                    Some(&format!("EntityCreationNotAllowed: {}", ref_text)),
                    None,
                    None,
                    source_type_str(source_type),
                    source_ref,
                    Some(pending_id),
                )
                .await
            }
        }
    }
}

async fn evaluate_rel_claim(
    pool: &SqlitePool,
    model_client: Option<Arc<ModelClient>>,
    claim_id: &str,
    rel: dsl::RelStmt,
    scope_str: &str,
    scope: &Scope,
    session_id: &str,
    resolve_ctx: &ResolveContext,
    source_type: &SourceType,
    source_ref: Option<&str>,
) -> Result<(), String> {
    let original_dsl = dsl_statement_to_line(&DslStatement::Rel(rel.clone()));
    let dsl::RelStmt {
        rel_type,
        participants,
        direction,
        polarity,
        rel_type_id,
        ..
    } = rel;
    let canonicalization_on = compiler::relation_canonicalization_enabled(pool).await;
    let _rel_type_raw = rel_type.clone();
    let alias_map = if canonicalization_on {
        compiler::load_role_alias_map(pool).await
    } else {
        HashMap::new()
    };

    let mut roles_seen = Vec::new();
    let mut canonicalized = Vec::new();
    for (role, p_ref) in participants {
        let canonical_role = if canonicalization_on {
            canonicalize_role_token(&role, &alias_map)
        } else {
            role
        };
        roles_seen.push(canonical_role.clone());
        canonicalized.push((canonical_role, p_ref));
    }

    let mut fallback_used = false;
    let rel_type_resolved = if let Some(provided) = rel_type_id.clone().filter(|id| !id.trim().is_empty()) {
        rel_type_catalog::resolve_rel_type_from_id(
            pool,
            &provided,
            &_rel_type_raw,
            canonicalization_on,
        )
        .await
    } else {
        rel_type_catalog::resolve_rel_type(pool, &_rel_type_raw, &roles_seen, canonicalization_on).await
    };
    let mut rel_type_resolved = match rel_type_resolved {
        Ok(resolved) => resolved,
        Err(_) => {
            fallback_used = true;
            rel_type_catalog::RelTypeResolution {
                rel_type_id: uuid::Uuid::new_v4().to_string(),
                rel_type_norm: normalize_rel_type(&_rel_type_raw),
                rel_type_raw: _rel_type_raw.clone(),
            }
        }
    };
    if rel_type_resolved.rel_type_id.trim().is_empty() {
        rel_type_resolved.rel_type_id = uuid::Uuid::new_v4().to_string();
        fallback_used = true;
    }
    if fallback_used {
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO rel_type (rel_type_id, canonical_name, status, created_at)
             VALUES (?, ?, 'provisional', CURRENT_TIMESTAMP)"
        )
        .bind(&rel_type_resolved.rel_type_id)
        .bind(&rel_type_resolved.rel_type_norm)
        .execute(pool)
        .await;
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO rel_type_alias (alias, rel_type_id, confidence, status, created_at)
             VALUES (?, ?, 1.0, 'confirmed', CURRENT_TIMESTAMP)"
        )
        .bind(&rel_type_resolved.rel_type_norm)
        .bind(&rel_type_resolved.rel_type_id)
        .execute(pool)
        .await;
    }
    let rel_type_norm = rel_type_resolved.rel_type_norm.clone();
    let rel_type_id = rel_type_resolved.rel_type_id.clone();

    let res = sqlx::query(
        "UPDATE memory_claims SET rel_type_raw = ?, rel_type_norm = ?, rel_type_id = ? WHERE id = ?"
    )
    .bind(&_rel_type_raw)
    .bind(&rel_type_norm)
    .bind(&rel_type_id)
    .bind(claim_id)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;
    if res.rows_affected() == 0 {
        finalize_claim(
            pool,
            claim_id,
            "rejected",
            Some("ClaimUpdateFailed: rel_type_id not persisted"),
            None,
            None,
            source_type_str(source_type),
            source_ref,
            None,
        )
        .await?;
        return Ok(());
    }

    let sample_dsl = compiler::format_rel_sample(&rel_type_norm, &canonicalized, direction.as_ref());
    let shape = match compiler::resolve_relation_shape(
        &CompileContext {
            pool: pool.clone(),
            model_client: None,
            session_id: session_id.to_string(),
            scope: scope.clone(),
            source: *source_type,
            source_ref: source_ref.map(|s| s.to_string()),
            now: Utc::now(),
            embedding_config: None,
            skip_claims: true,
            allow_ambiguous_user_refs: false,
        },
        &rel_type_id,
        &rel_type_norm,
        &roles_seen,
        &direction,
        &sample_dsl,
        scope,
        canonicalization_on,
    )
    .await
    {
        Ok(shape) => shape,
        Err(err) => {
            finalize_claim(
                pool,
                claim_id,
                "rejected",
                Some(&err),
                None,
                None,
                source_type_str(source_type),
                source_ref,
                None,
            )
            .await?;
            return Ok(());
        }
    };

    if matches!(shape.cardinality_override, Some(Cardinality::One)) && shape.anchor_roles.is_empty() {
        // Fail-open: allow claim evaluation to proceed even if anchor_roles are missing.
    }

    let mut resolved_participants = Vec::new();
    let mut needs_new_entity = false;
    for (role, p_ref) in canonicalized {
        let resolution = match resolver::resolve_ref(&p_ref, resolve_ctx).await {
            Ok(resolution) => resolution,
            Err(e) => {
                finalize_claim(
                    pool,
                    claim_id,
                    "rejected",
                    Some(&format!("ResolutionError: {}", e)),
                    None,
                    None,
                    source_type_str(source_type),
                    source_ref,
                    None,
                )
                .await?;
                return Ok(());
            }
        };
        match resolution {
            ResolutionResult::Resolved(id) => resolved_participants.push((role, id)),
            ResolutionResult::Ambiguous(pending) => {
                let ref_text = format!("{:?}", p_ref);
                let pending_id = create_pending_clarify_for_claim(
                    pool,
                    claim_id,
                    session_id.to_string(),
                    &original_dsl,
                    &ref_text,
                    &pending.candidates_json,
                )
                .await?;
                let candidates_summary = summarize_candidates(&pending.candidates_json, 3);
                let reason = match candidates_summary {
                    Some(summary) => format!("AmbiguousRef: {} | Candidates: {}", ref_text, summary),
                    None => format!("AmbiguousRef: {}", ref_text),
                };
                finalize_claim(
                    pool,
                    claim_id,
                    "needs_clarify",
                    Some(&reason),
                    None,
                    None,
                    source_type_str(source_type),
                    source_ref,
                    Some(pending_id),
                )
                .await?;
                return Ok(());
            }
            ResolutionResult::NewEntity(label) => {
                if claim_allows_entity_create(source_type) {
                    needs_new_entity = true;
                } else {
                    let ref_text = label.trim().to_string();
                    let pending_id = create_pending_clarify_for_claim(
                        pool,
                        claim_id,
                        session_id.to_string(),
                        &original_dsl,
                        &ref_text,
                        "[]",
                    )
                    .await?;
                    finalize_claim(
                        pool,
                        claim_id,
                        "needs_clarify",
                        Some(&format!("EntityCreationNotAllowed: {}", ref_text)),
                        None,
                        None,
                        source_type_str(source_type),
                        source_ref,
                        Some(pending_id),
                    )
                    .await?;
                    return Ok(());
                }
            }
        }
    }

    if needs_new_entity {
        let promotion = match promote_claim(pool, model_client.clone(), claim_id).await {
            Ok(promotion) => promotion,
            Err(e) => {
                finalize_claim(
                    pool,
                    claim_id,
                    "rejected",
                    Some(&format!("PromotionFailed: {}", e)),
                    None,
                    None,
                    source_type_str(source_type),
                    source_ref,
                    None,
                )
                .await?;
                return Ok(());
            }
        };
        if !promotion.written_ids.is_empty() {
            update_working_set_for_beliefs(pool, &promotion.written_ids).await?;
        }
        let linked_belief_id = promotion.written_ids.first().copied();
        let _ = episodic::emit_claim_status_event(
            pool,
            claim_id,
            "promoted",
            linked_belief_id,
            source_type_str(source_type),
            source_ref,
            Some("promoted"),
            None,
            None,
        )
        .await;
        return Ok(());
    }

    if canonicalization_on && shape.commutative {
        resolved_participants.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    } else if !canonicalization_on && direction.is_none() {
        resolved_participants.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    }

    let anchor_sig = compute_anchor_signature(&shape.anchor_roles, &resolved_participants, false);
    let topic_key = format!("rel:{}:{}", rel_type_id, anchor_sig);
    let conflict = match has_conflict(pool, &topic_key, scope_str, &polarity).await {
        Ok(conflict) => conflict,
        Err(e) => {
            finalize_claim(
                pool,
                claim_id,
                "rejected",
                Some(&format!("ConflictCheckFailed: {}", e)),
                None,
                None,
                source_type_str(source_type),
                source_ref,
                None,
            )
            .await?;
            return Ok(());
        }
    };
    if conflict {
        finalize_claim(
            pool,
            claim_id,
            "conflict",
            Some("ConflictDetected"),
            Some(&topic_key),
            Some("opposite_polarity_active"),
            source_type_str(source_type),
            source_ref,
            None,
        )
        .await?;
        return Ok(());
    }

    let promotion = match promote_claim(pool, model_client.clone(), claim_id).await {
        Ok(promotion) => promotion,
        Err(e) => {
            finalize_claim(
                pool,
                claim_id,
                "rejected",
                Some(&format!("PromotionFailed: {}", e)),
                None,
                None,
                source_type_str(source_type),
                source_ref,
                None,
            )
            .await?;
            return Ok(());
        }
    };
    if !promotion.written_ids.is_empty() {
        update_working_set_for_beliefs(pool, &promotion.written_ids).await?;
    }
    let linked_belief_id = promotion.written_ids.first().copied();
    let _ = episodic::emit_claim_status_event(
        pool,
        claim_id,
        "promoted",
        linked_belief_id,
        source_type_str(source_type),
        source_ref,
        Some("promoted"),
        None,
        None,
    )
    .await;
    Ok(())
}

async fn finalize_claim(
    pool: &SqlitePool,
    claim_id: &str,
    status: &str,
    decision_reason: Option<&str>,
    conflict_topic_key: Option<&str>,
    conflict_reason: Option<&str>,
    source_type: &str,
    source_ref: Option<&str>,
    pending_clarify_id: Option<i64>,
) -> Result<(), String> {
    sqlx::query(
        "UPDATE memory_claims
         SET status = ?,
             decision_reason = ?,
             conflict_topic_key = ?,
             conflict_reason = ?,
             evaluated_at = CURRENT_TIMESTAMP,
             updated_at = CURRENT_TIMESTAMP
         WHERE id = ?"
    )
    .bind(status)
    .bind(decision_reason)
    .bind(conflict_topic_key)
    .bind(conflict_reason)
    .bind(claim_id)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    record_claim_outcome(pool, claim_id, status, decision_reason).await;

    let linked_belief_id = None;
    let _ = episodic::emit_claim_status_event(
        pool,
        claim_id,
        status,
        linked_belief_id,
        source_type,
        source_ref,
        decision_reason,
        conflict_topic_key,
        conflict_reason,
    )
    .await;

    if let Some(pending_id) = pending_clarify_id {
        let _ = sqlx::query("UPDATE ics_pending_clarify SET status = 'pending' WHERE id = ?")
            .bind(pending_id)
            .execute(pool)
            .await;
    }

    Ok(())
}

async fn create_pending_clarify_for_claim(
    pool: &SqlitePool,
    claim_id: &str,
    session_id: String,
    original_dsl: &str,
    ref_text: &str,
    candidates_json: &str,
) -> Result<i64, String> {
    let existing: Option<i64> = sqlx::query_scalar(
        "SELECT id FROM ics_pending_clarify
         WHERE claim_id = ? AND status = 'pending'
         ORDER BY created_at DESC
         LIMIT 1"
    )
    .bind(claim_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?
    .flatten();

    if let Some(id) = existing {
        return Ok(id);
    }

    let row = sqlx::query(
        "INSERT INTO ics_pending_clarify (claim_id, session_id, original_dsl, ref_text, candidates_json, status)
         VALUES (?, ?, ?, ?, ?, 'pending')
         RETURNING id"
    )
    .bind(claim_id)
    .bind(&session_id)
    .bind(original_dsl)
    .bind(ref_text)
    .bind(candidates_json)
    .fetch_one(pool)
    .await
    .map_err(|e| e.to_string())?;

    Ok(row.get("id"))
}

async fn has_conflict(
    pool: &SqlitePool,
    topic_key: &str,
    scope_str: &str,
    polarity: &str,
) -> Result<bool, String> {
    let opposite = if polarity == "assert" { "deny" } else { "assert" };
    let existing: Option<i64> = sqlx::query_scalar(
        "SELECT id FROM ics_beliefs
         WHERE topic_key = ? AND scope = ? AND status = 'active' AND polarity = ?
         LIMIT 1"
    )
    .bind(topic_key)
    .bind(scope_str)
    .bind(opposite)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?
    .flatten();

    Ok(existing.is_some())
}

async fn update_working_set_for_beliefs(pool: &SqlitePool, belief_ids: &[i64]) -> Result<(), String> {
    if belief_ids.is_empty() {
        return Ok(());
    }

    let placeholders = belief_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let mut entity_ids = Vec::new();

    let fact_query = format!(
        "SELECT DISTINCT subject_entity_id AS entity_id FROM ics_fact_beliefs WHERE belief_id IN ({})",
        placeholders
    );
    let mut fact_stmt = sqlx::query(&fact_query);
    for id in belief_ids {
        fact_stmt = fact_stmt.bind(id);
    }
    if let Ok(rows) = fact_stmt.fetch_all(pool).await {
        for row in rows {
            if let Ok(id) = row.try_get::<i64, _>("entity_id") {
                if !entity_ids.contains(&id) {
                    entity_ids.push(id);
                }
            }
        }
    }

    let rel_query = format!(
        "SELECT DISTINCT entity_id FROM ics_rel_participants WHERE belief_id IN ({})",
        placeholders
    );
    let mut rel_stmt = sqlx::query(&rel_query);
    for id in belief_ids {
        rel_stmt = rel_stmt.bind(id);
    }
    if let Ok(rows) = rel_stmt.fetch_all(pool).await {
        for row in rows {
            if let Ok(id) = row.try_get::<i64, _>("entity_id") {
                if !entity_ids.contains(&id) {
                    entity_ids.push(id);
                }
            }
        }
    }

    working_set::update_working_set(pool, &entity_ids, belief_ids).await
}

fn parse_claim_statement(claim_text: &str) -> Result<DslStatement, String> {
    if let Ok(stmt) = serde_json::from_str::<DslStatement>(claim_text) {
        return Ok(stmt);
    }

    let parsed = dsl::parse_memory_block(claim_text);
    if parsed.is_empty() {
        return Err("Empty claim".to_string());
    }

    let mut statements = Vec::new();
    for line in parsed {
        match line {
            Ok(stmt) => statements.push(stmt),
            Err(err) => return Err(err),
        }
    }

    if statements.len() != 1 {
        return Err("Claim must contain exactly one statement".to_string());
    }

    Ok(statements.remove(0))
}

fn session_id_from_scope(scope: &Scope) -> String {
    match scope {
        Scope::Context(id) => id.clone(),
        _ => "default".to_string(),
    }
}

fn claim_allows_entity_create(source_type: &SourceType) -> bool {
    matches!(source_type, SourceType::User | SourceType::Inference)
}

fn source_type_str(source: &SourceType) -> &'static str {
    match source {
        SourceType::User => "user",
        SourceType::Tool => "tool",
        SourceType::System => "system",
        SourceType::Inference => "inference",
    }
}

fn build_summary_dsl(summary: &str, conversation_id: Option<&str>) -> String {
    let label = if let Some(id) = conversation_id {
        format!("Conversation {}", id)
    } else {
        "Conversation global".to_string()
    };
    let scope_expr = if let Some(id) = conversation_id {
        format!("context:{}", id)
    } else {
        "global".to_string()
    };
    let date = Utc::now().format("%Y-%m-%d").to_string();
    let summary_literal = format_literal(summary);

    format!(
        "\"{}\":episodic_summary = {} ^{} @{}",
        escape_quotes(&label),
        summary_literal,
        date,
        scope_expr
    )
}

fn contains_diagnostic_marker(text: &str) -> bool {
    let lowered = text.to_lowercase();
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

fn claim_text_to_dsl(claim_text: &str) -> Result<String, String> {
    if let Ok(stmt) = serde_json::from_str::<DslStatement>(claim_text) {
        Ok(dsl_statement_to_line(&stmt))
    } else {
        let trimmed = claim_text.trim_start();
        if trimmed.starts_with('{') {
            Err("InvalidClaimJson: failed to parse claim_text as JSON".to_string())
        } else {
            Ok(claim_text.to_string())
        }
    }
}

fn parse_scope_from_claim(raw: &str) -> Option<Scope> {
    if let Ok(scope) = serde_json::from_str::<Scope>(raw) {
        return Some(scope);
    }
    parse_scope(raw).ok()
}

fn parse_source_type(raw: &str) -> SourceType {
    match raw.trim().to_lowercase().as_str() {
        "user" => SourceType::User,
        "user_focus" => SourceType::User,
        "tool" => SourceType::Tool,
        "system" => SourceType::System,
        "inference" => SourceType::Inference,
        _ => SourceType::Inference,
    }
}

fn dsl_statement_to_line(stmt: &DslStatement) -> String {
    match stmt {
        DslStatement::Fact(fact) => {
            let subject = ref_to_string(&fact.subject);
            let value = format_literal(&fact.value);
            let mut line = format!("{}:{} = {}", subject, fact.key, value);
            if let Some(certainty) = fact.certainty {
                line.push_str(&format!(" ~{}", certainty));
            }
            if let Some(time) = &fact.time_expr {
                line.push_str(&format!(" ^{}", time.value));
            }
            if let Some(scope_expr) = &fact.scope_expr {
                line.push_str(&format!(" @{}", scope_expr));
            }
            if let Some(source_ref) = &fact.source_ref {
                line.push_str(&format!(" <{}>", source_ref));
            }
            if fact.polarity == "deny" {
                line.push_str(" !deny");
            }
            line
        }
        DslStatement::Rel(rel) => {
            let args = format_rel_args(&rel.participants, rel.direction);
            let mut line = format!("{}({})", rel.rel_type, args);
            if let Some(certainty) = rel.certainty {
                line.push_str(&format!(" ~{}", certainty));
            }
            if let Some(time) = &rel.time_expr {
                line.push_str(&format!(" ^{}", time.value));
            }
            if let Some(scope_expr) = &rel.scope_expr {
                line.push_str(&format!(" @{}", scope_expr));
            }
            if let Some(source_ref) = &rel.source_ref {
                line.push_str(&format!(" <{}>", source_ref));
            }
            if rel.polarity == "deny" {
                line.push_str(" !deny");
            }
            line
        }
    }
}

fn format_rel_args(participants: &[(String, Ref)], direction: Option<RelDirection>) -> String {
    if let Some(dir) = direction {
        if participants.len() == 2 {
            let left = format_participant(&participants[0]);
            let right = format_participant(&participants[1]);
            let arrow = match dir {
                RelDirection::Directed => "->",
                RelDirection::Bidirectional => "<->",
            };
            return format!("{} {} {}", left, arrow, right);
        }
    }

    participants
        .iter()
        .map(format_participant)
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_participant(participant: &(String, Ref)) -> String {
    format!("{}: {}", participant.0, ref_to_string(&participant.1))
}

fn ref_to_string(r: &Ref) -> String {
    match r {
        Ref::Handle(handle) => format!("${}", handle),
        Ref::Label(label) => format!("#{}", label),
        Ref::Filter(label, _key) => format!("#{}", label),
        Ref::Name(name) => format!("\"{}\"", escape_quotes(name)),
    }
}

fn format_literal(value: &str) -> String {
    let cleaned = value.replace('\n', " ").replace('\r', " ");
    format!("\"{}\"", escape_quotes(cleaned.trim()))
}

fn escape_quotes(input: &str) -> String {
    input.replace('\\', "\\\\").replace('"', "\\\"")
}

fn summary_parts_count(summary: &str) -> usize {
    summary
        .split(" | ")
        .filter(|part| !part.trim().is_empty())
        .count()
}
