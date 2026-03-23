use sqlx::SqlitePool;
use sqlx::Row;
use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use crate::core::memory::types::{Scope, SourceType, PendingWrite, RelationShape, Cardinality};
use crate::core::memory::dsl::{self, DslStatement, RelStmt, RelDirection};
use crate::core::memory::resolver::{self, ResolveContext, ResolutionResult};
use crate::core::memory::writer::{self, WriteContext, WriteResult, RelationWriteOptions};
use crate::core::memory::scope::parse_scope;
use crate::core::memory::inject_context::extract_memory_blocks;
use crate::core::memory::canonical::{canonicalize_label, canonicalize_role_token, normalize_role_token};
use crate::core::memory::claims;
use crate::core::memory::rel_type_catalog;
use crate::core::model_client::ModelClient;
use crate::core::episodic;
use crate::core::system_controls;
use crate::core::system_log;
use serde_json::json;
use uuid::Uuid;
use chrono::{Utc, NaiveDateTime};

pub struct CompileContext {
    pub pool: SqlitePool,
    pub model_client: Option<Arc<ModelClient>>,
    pub session_id: String,
    pub scope: Scope,
    pub source: SourceType,
    pub source_ref: Option<String>,
    pub now: chrono::DateTime<chrono::Utc>,
    pub embedding_config: Option<crate::core::memory::writer::EmbeddingConfig>,
    pub skip_claims: bool,
    pub allow_ambiguous_user_refs: bool,
}

async fn memory_write_allowed_for_context(ctx: &CompileContext) -> bool {
    let mode: Option<String> = sqlx::query_scalar(
        "SELECT mode FROM system_controls WHERE subsystem_id = ?",
    )
    .bind("memory_write")
    .fetch_optional(&ctx.pool)
    .await
    .ok()
    .flatten();
    let mode = mode
        .unwrap_or_else(|| system_controls::default_mode_for("memory_write").unwrap_or("normal").to_string());
    let reason = match ctx.source {
        SourceType::Tool => "tool_outcome",
        SourceType::System => "memory_pass",
        SourceType::Inference => "inference",
        SourceType::User => "user_input",
    };
    if system_controls::allow_memory_write(&mode, reason) {
        return true;
    }
    let _ = system_log::log_event(
        &ctx.pool,
        None,
        "warn",
        "memory",
        None,
        None,
        json!({
            "event": "memory_write_blocked",
            "reason": "system_control",
            "mode": mode,
            "conversation_id": ctx.session_id,
        }),
    )
    .await;
    false
}

/// Clarification candidate info for LLM injection
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClarifyCandidate {
    pub entity_id: i64,
    pub label: String,
    pub context: String, // e.g., "sister, last mentioned 3 days ago"
}

/// Pending clarification for ambiguous resolution
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PendingClarify {
    pub id: i64,
    pub ref_text: String,
    pub candidates: Vec<ClarifyCandidate>,
    pub original_dsl: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CompileResult {
    pub written_ids: Vec<i64>,
    pub conflict_ids: Vec<i64>,
    pub pending_writes: Vec<PendingWrite>,
    pub pending_clarify: Option<PendingClarify>,
    pub claim_ids: Vec<String>,
    pub errors: Vec<String>,
}

const CLARIFY_MAX_ATTEMPTS: i64 = 2;
const CLARIFY_ATTEMPT_WINDOW_HOURS: i64 = 24;

async fn should_drop_clarify(pool: &SqlitePool, session_id: &str, ref_text: &str) -> Result<bool, String> {
    let key = format!("clarify_attempts:{}:{}", session_id, canonicalize_label(ref_text));
    let row = sqlx::query("SELECT value, updated_at FROM kv_store WHERE key = ?")
        .bind(&key)
        .fetch_optional(pool)
        .await
        .map_err(|e| e.to_string())?;
    let Some(row) = row else {
        return Ok(false);
    };
    let count_raw: Option<String> = row.try_get("value").ok();
    let count = count_raw
        .as_deref()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(0);
    let updated_at: Option<String> = row.try_get("updated_at").ok();
    if let Some(ts) = updated_at.as_deref() {
        if let Ok(parsed) = NaiveDateTime::parse_from_str(ts, "%Y-%m-%d %H:%M:%S") {
            let age_hours = (Utc::now().naive_utc() - parsed).num_hours();
            if age_hours > CLARIFY_ATTEMPT_WINDOW_HOURS {
                return Ok(false);
            }
        }
    }
    Ok(count >= CLARIFY_MAX_ATTEMPTS)
}

async fn bump_clarify_attempt(pool: &SqlitePool, session_id: &str, ref_text: &str) -> Result<i64, String> {
    let key = format!("clarify_attempts:{}:{}", session_id, canonicalize_label(ref_text));
    let row = sqlx::query("SELECT value, updated_at FROM kv_store WHERE key = ?")
        .bind(&key)
        .fetch_optional(pool)
        .await
        .map_err(|e| e.to_string())?;
    let mut count = 0i64;
    if let Some(row) = row {
        let count_raw: Option<String> = row.try_get("value").ok();
        count = count_raw
            .as_deref()
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(0);
    }
    count += 1;
    sqlx::query(
        "INSERT INTO kv_store (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
    )
    .bind(&key)
    .bind(count.to_string())
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(count)
}

/// Infer entity type from role name in relations
fn infer_type_from_role(role: &str) -> Option<&'static str> {
    match role.to_lowercase().as_str() {
        "person" | "user" | "owner" | "author" | "creator" | "subject" | "actor"
        | "parent" | "child" | "mother" | "father" | "daughter" | "son" | "spouse"
        | "sibling" | "brother" | "sister" | "partner" | "husband" | "wife" => Some("person"),
        "place" | "location" | "city" | "country" | "venue" => Some("place"),
        "work" | "project" | "product" | "book" | "movie" | "song" | "company" => Some("work"),
        "event" | "meeting" | "appointment" => Some("event"),
        "concept" | "idea" | "topic" | "category" | "object" | "thing" | "item" => Some("concept"),
        _ => None,
    }
}

fn label_from_ref(r: &dsl::Ref) -> Option<String> {
    match r {
        dsl::Ref::Handle(h) => Some(h.to_string()),
        dsl::Ref::Label(l) => Some(l.to_string()),
        dsl::Ref::Filter(l, _key) => Some(l.to_string()),
        dsl::Ref::Name(n) => Some(n.to_string()),
    }
}

fn low_confidence(current: Option<f32>) -> Option<f32> {
    let low = 0.4;
    match current {
        Some(v) if v <= low => Some(v),
        _ => Some(low),
    }
}

fn is_identity_key(key: &str) -> bool {
    matches!(key, "name" | "full_name" | "preferred_name" | "display_name")
}

fn identity_handle(subject: &dsl::Ref, key: &str, polarity: &str) -> Option<String> {
    if polarity != "assert" || !is_identity_key(key) {
        return None;
    }

    match subject {
        dsl::Ref::Handle(handle) if handle == "user" || handle == "assistant" => Some(handle.to_string()),
        _ => None,
    }
}

fn parse_json_vec(raw: Option<&str>) -> Vec<String> {
    raw.and_then(|v| serde_json::from_str::<Vec<String>>(v).ok())
        .unwrap_or_default()
}

fn sanitize_memory_input(input: &str) -> Option<(String, &'static str)> {
    let blocks = extract_memory_blocks(input);
    if let Some(block) = blocks.into_iter().find(|b| !b.trim().is_empty()) {
        return Some((block.trim().to_string(), "extracted_block"));
    }
    let mut lines = Vec::new();
    for line in input.lines() {
        if crate::core::memory::dsl::is_dsl_line(line) {
            lines.push(line.trim());
        }
    }
    if lines.is_empty() {
        return None;
    }
    Some((lines.join("\n"), "filtered_lines"))
}

fn ensure_alias(aliases: &mut Vec<String>, aliases_canon: &mut Vec<String>, alias: &str) {
    let canon = canonicalize_label(alias);
    if aliases_canon.iter().any(|existing| existing == &canon) {
        return;
    }
    aliases.push(alias.to_string());
    aliases_canon.push(canon);
}

fn ensure_key(keys: &mut Vec<String>, key: &str) {
    if keys.iter().any(|k| k == key) {
        return;
    }
    keys.push(key.to_string());
}

fn remove_key(keys: &mut Vec<String>, key: &str) -> bool {
    let original_len = keys.len();
    keys.retain(|k| k != key);
    keys.len() != original_len
}

async fn upsert_session_binding(
    pool: &SqlitePool,
    session_id: &str,
    ref_text: &str,
    entity_id: i64,
) -> Result<(), String> {
    sqlx::query(
        "INSERT INTO ics_session_bindings (session_id, ref_text, entity_id, created_at)
         VALUES (?, ?, ?, CURRENT_TIMESTAMP)
         ON CONFLICT(session_id, ref_text) DO UPDATE SET entity_id = excluded.entity_id"
    )
    .bind(session_id)
    .bind(ref_text)
    .bind(entity_id)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(())
}

async fn maybe_upsert_name_binding(
    pool: &SqlitePool,
    session_id: &str,
    ref_text: &str,
    entity_id: i64,
) -> Result<(), String> {
    let existing: Option<i64> = sqlx::query_scalar(
        "SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = ?"
    )
    .bind(session_id)
    .bind(ref_text)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?
    .flatten();

    if existing.is_some() && existing != Some(entity_id) {
        return Ok(());
    }

    upsert_session_binding(pool, session_id, ref_text, entity_id).await
}

async fn update_display_name(
    pool: &SqlitePool,
    handle: &str,
    label: &str,
) -> Result<(), String> {
    let column = match handle {
        "user" => "user_display_name",
        "assistant" => "assistant_display_name",
        _ => return Ok(()),
    };
    let query = format!("UPDATE settings SET {} = ? WHERE id = 1", column);
    sqlx::query(&query)
        .bind(label)
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

async fn load_entity_state(pool: &SqlitePool, entity_id: i64) -> Result<(String, Vec<String>, Vec<String>, Vec<String>), String> {
    let row = sqlx::query(
        "SELECT label, keys, aliases, aliases_canonical FROM ics_entities WHERE id = ?"
    )
    .bind(entity_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    let Some(row) = row else {
        return Err(format!("Entity {} not found", entity_id));
    };

    let label: String = row.get("label");
    let keys = parse_json_vec(row.try_get::<Option<String>, _>("keys").ok().flatten().as_deref());
    let aliases = parse_json_vec(row.try_get::<Option<String>, _>("aliases").ok().flatten().as_deref());
    let mut aliases_canon = parse_json_vec(row.try_get::<Option<String>, _>("aliases_canonical").ok().flatten().as_deref());
    if aliases.is_empty() {
        aliases_canon.clear();
    } else if aliases_canon.len() != aliases.len() {
        aliases_canon = aliases.iter().map(|alias| canonicalize_label(alias)).collect();
    }
    Ok((label, keys, aliases, aliases_canon))
}

async fn save_entity_state(
    pool: &SqlitePool,
    entity_id: i64,
    label: &str,
    keys: &[String],
    aliases: &[String],
    aliases_canon: &[String],
) -> Result<(), String> {
    let keys_json = serde_json::to_string(keys).unwrap_or_else(|_| "[]".to_string());
    let aliases_json = serde_json::to_string(aliases).unwrap_or_else(|_| "[]".to_string());
    let aliases_canon_json = serde_json::to_string(aliases_canon).unwrap_or_else(|_| "[]".to_string());
    let label_canon = canonicalize_label(label);

    sqlx::query(
        "UPDATE ics_entities
         SET label = ?, label_canonical = ?, aliases = ?, aliases_canonical = ?, keys = ?
         WHERE id = ?"
    )
    .bind(label)
    .bind(label_canon)
    .bind(aliases_json)
    .bind(aliases_canon_json)
    .bind(keys_json)
    .bind(entity_id)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(())
}

async fn bind_identity_name(
    pool: &SqlitePool,
    session_id: &str,
    handle: &str,
    subject_id: i64,
    desired_label: &str,
) -> Result<(), String> {
    let desired_label = desired_label.trim();
    if desired_label.is_empty() {
        return Ok(());
    }

    let session_id = if session_id.trim().is_empty() {
        "default"
    } else {
        session_id
    };

    let handle_key = format!("sys:{}", handle);
    let desired_canon = canonicalize_label(desired_label);
    let key_pattern = format!("%\"{}\"%", handle_key);

    let target_row = sqlx::query(
        "SELECT id FROM ics_entities
         WHERE label_canonical = ?
         ORDER BY CASE WHEN keys LIKE ? THEN 0 ELSE 1 END, access_count DESC
         LIMIT 1"
    )
    .bind(&desired_canon)
    .bind(&key_pattern)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    let target_id = target_row.map(|row| row.get::<i64, _>("id")).unwrap_or(subject_id);

    if target_id != subject_id {
        let (subject_label, mut subject_keys, subject_aliases, subject_aliases_canon) =
            load_entity_state(pool, subject_id).await?;
        let (mut target_label, mut target_keys, mut target_aliases, mut target_aliases_canon) =
            load_entity_state(pool, target_id).await?;

        if target_label != desired_label {
            ensure_alias(&mut target_aliases, &mut target_aliases_canon, &target_label);
            target_label = desired_label.to_string();
        }

        if subject_label != desired_label {
            ensure_alias(&mut target_aliases, &mut target_aliases_canon, &subject_label);
        }

        ensure_key(&mut target_keys, &handle_key);
        save_entity_state(pool, target_id, &target_label, &target_keys, &target_aliases, &target_aliases_canon).await?;

        if remove_key(&mut subject_keys, &handle_key) {
            save_entity_state(
                pool,
                subject_id,
                &subject_label,
                &subject_keys,
                &subject_aliases,
                &subject_aliases_canon,
            )
            .await?;
        }

        upsert_session_binding(pool, session_id, handle, target_id).await?;
        let _ = maybe_upsert_name_binding(pool, session_id, desired_label, target_id).await;
        let _ = update_display_name(pool, handle, desired_label).await;
        return Ok(());
    }

    let (mut subject_label, mut subject_keys, mut subject_aliases, mut subject_aliases_canon) =
        load_entity_state(pool, subject_id).await?;

    if subject_label != desired_label {
        ensure_alias(&mut subject_aliases, &mut subject_aliases_canon, &subject_label);
        subject_label = desired_label.to_string();
    }

    ensure_key(&mut subject_keys, &handle_key);

    save_entity_state(
        pool,
        subject_id,
        &subject_label,
        &subject_keys,
        &subject_aliases,
        &subject_aliases_canon,
    )
    .await?;

    upsert_session_binding(pool, session_id, handle, subject_id).await?;
    let _ = maybe_upsert_name_binding(pool, session_id, desired_label, subject_id).await;
    let _ = update_display_name(pool, handle, desired_label).await;

    Ok(())
}

pub async fn compile(input: &str, ctx: CompileContext) -> CompileResult {
    let mut result = CompileResult {
        written_ids: vec![],
        conflict_ids: vec![],
        pending_writes: vec![],
        pending_clarify: None,
        claim_ids: vec![],
        errors: vec![],
    };
    if !memory_write_allowed_for_context(&ctx).await {
        result.errors.push("memory_write_blocked: system_control".to_string());
        return result;
    }

    let mut fact_count = 0usize;
    let mut rel_count = 0usize;
    let mut parse_errors = 0usize;
    let mut write_errors = 0usize;

    // 1. Parse
    let mut parsed = dsl::parse_memory_block(input);
    let mut sanitized_input: Option<String> = None;
    if !parsed.iter().any(|res| res.is_ok()) {
        if let Some((sanitized, reason)) = sanitize_memory_input(input) {
            let original_len = input.len();
            let sanitized_len = sanitized.len();
            sanitized_input = Some(sanitized);
            parsed = dsl::parse_memory_block(sanitized_input.as_deref().unwrap_or(input));
            let _ = system_log::log_event(
                &ctx.pool,
                None,
                "info",
                "memory",
                None,
                None,
                serde_json::json!({
                    "event": "memory_dsl_fallback_used",
                    "reason": reason,
                    "original_len": original_len,
                    "sanitized_len": sanitized_len,
                }),
            )
            .await;
        }
    }
    let input_for_parse = sanitized_input.as_deref().unwrap_or(input);

    // 2. Process Statements
    for (i, line_res) in parsed.into_iter().enumerate() {
        match line_res {
            Ok(stmt) => {
                match &stmt {
                    DslStatement::Fact(_) => fact_count += 1,
                    DslStatement::Rel(_) => rel_count += 1,
                }
                match process_statement(stmt, input_for_parse, &ctx, &mut result).await {
                    Ok(res) => {
                        match res {
                            WriteResult::Inserted(id) | WriteResult::Updated(id) => result.written_ids.push(id),
                            WriteResult::Conflict { belief_id, conflict_set_id } => {
                                result.written_ids.push(belief_id);
                                result.conflict_ids.push(conflict_set_id);
                            },
                            WriteResult::Ignored(_) => {},
                            WriteResult::Error(e) => {
                                write_errors += 1;
                                result.errors.push(format!("Line {}: Write error: {}", i, e));
                            }
                        }
                    },
                    Err(e) => {
                        // Check if this is a pending clarify error (not a real error)
                        if !e.starts_with("PENDING_CLARIFY:") {
                            write_errors += 1;
                            result.errors.push(format!("Line {}: Resolution error: {}", i, e));
                        }
                    },
                }
            },
            Err(e) => {
                parse_errors += 1;
                result.errors.push(format!("Line {}: Parse error: {}", i, e))
            }
        }
    }

    let _ = system_log::log_event(
        &ctx.pool,
        None,
        "info",
        "memory",
        None,
        None,
        serde_json::json!({
            "event": "memory_compile_summary",
            "session_id": ctx.session_id,
            "source": format!("{:?}", ctx.source),
            "fact_count": fact_count,
            "rel_count": rel_count,
            "parse_errors": parse_errors,
            "write_errors": write_errors,
            "written_count": result.written_ids.len(),
            "conflict_count": result.conflict_ids.len(),
        }),
    )
    .await;

    result
}

pub async fn compile_parsed(
    statements: Vec<DslStatement>,
    original_dsl: &str,
    ctx: CompileContext,
) -> CompileResult {
    let mut result = CompileResult {
        written_ids: vec![],
        conflict_ids: vec![],
        pending_writes: vec![],
        pending_clarify: None,
        claim_ids: vec![],
        errors: vec![],
    };
    if !memory_write_allowed_for_context(&ctx).await {
        result.errors.push("memory_write_blocked: system_control".to_string());
        return result;
    }

    let mut fact_count = 0usize;
    let mut rel_count = 0usize;
    let mut write_errors = 0usize;

    for (i, stmt) in statements.into_iter().enumerate() {
        match &stmt {
            DslStatement::Fact(_) => fact_count += 1,
            DslStatement::Rel(_) => rel_count += 1,
        }
        match process_statement(stmt, original_dsl, &ctx, &mut result).await {
            Ok(res) => match res {
                WriteResult::Inserted(id) | WriteResult::Updated(id) => {
                    result.written_ids.push(id)
                }
                WriteResult::Conflict {
                    belief_id,
                    conflict_set_id,
                } => {
                    result.written_ids.push(belief_id);
                    result.conflict_ids.push(conflict_set_id);
                }
                WriteResult::Ignored(_) => {}
                WriteResult::Error(e) => {
                    write_errors += 1;
                    result.errors.push(format!("Line {}: Write error: {}", i, e))
                }
            },
            Err(e) => {
                if !e.starts_with("PENDING_CLARIFY:") {
                    write_errors += 1;
                    result
                        .errors
                        .push(format!("Line {}: Resolution error: {}", i, e));
                }
            }
        }
    }

    let _ = system_log::log_event(
        &ctx.pool,
        None,
        "info",
        "memory",
        None,
        None,
        serde_json::json!({
            "event": "memory_compile_summary",
            "session_id": ctx.session_id,
            "source": format!("{:?}", ctx.source),
            "fact_count": fact_count,
            "rel_count": rel_count,
            "parse_errors": 0,
            "write_errors": write_errors,
            "written_count": result.written_ids.len(),
            "conflict_count": result.conflict_ids.len(),
        }),
    )
    .await;

    result
}

async fn process_statement(stmt: DslStatement, original_dsl: &str, ctx: &CompileContext, result: &mut CompileResult) -> Result<WriteResult, String> {
    let (scope_expr, stmt_source_ref) = match &stmt {
        DslStatement::Fact(f) => (f.scope_expr.as_deref(), f.source_ref.as_deref()),
        DslStatement::Rel(r) => (r.scope_expr.as_deref(), r.source_ref.as_deref()),
    };

    let effective_scope = if let Some(scope_raw) = scope_expr {
        parse_scope(scope_raw).map_err(|e| format!("Invalid scope '{}': {}", scope_raw, e.0))?
    } else {
        ctx.scope.clone()
    };

    let effective_source_ref = stmt_source_ref
        .map(|s| s.to_string())
        .or_else(|| ctx.source_ref.clone());

    let conversation_id = ctx.session_id.trim();
    let conversation_id = if conversation_id.is_empty() {
        None
    } else {
        Some(conversation_id.to_string())
    };

    if !ctx.skip_claims && memory_claims_enabled(&ctx.pool).await && claims_gate_source(&ctx.source) {
        let kind = match &stmt {
            DslStatement::Fact(_) => "fact",
            DslStatement::Rel(_) => "rel",
        };

        let (rel_type_raw, rel_type_norm, rel_type_id) = match &stmt {
            DslStatement::Rel(r) => {
                let canonicalization_on = relation_canonicalization_enabled(&ctx.pool).await;
                let raw = r.rel_type.clone();
                let alias_map = if canonicalization_on {
                    load_role_alias_map(&ctx.pool).await
                } else {
                    HashMap::new()
                };
                let roles_seen: Vec<String> = r
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
                let resolved = rel_type_catalog::resolve_rel_type(&ctx.pool, &raw, &roles_seen, canonicalization_on)
                    .await
                    .map_err(|e| format!("RelTypeResolve: {}", e))?;
                (Some(raw), Some(resolved.rel_type_norm), Some(resolved.rel_type_id))
            }
            _ => (None, None, None),
        };

        let mut claim_stmt = stmt.clone();
        if let DslStatement::Rel(ref mut rel_stmt) = claim_stmt {
            if let Some(ref rel_type_id) = rel_type_id {
                rel_stmt.rel_type_id = Some(rel_type_id.clone());
            }
        }
        let claim_text = serde_json::to_string(&claim_stmt).unwrap_or_else(|_| original_dsl.to_string());
        let scope_str = serde_json::to_string(&effective_scope).unwrap_or_default();
        let source_type = source_type_str(&ctx.source);
        let source_ref = effective_source_ref.as_deref();
        let claim_id = Uuid::new_v4().to_string();
        let claim_session_id = ctx.session_id.trim();
        let claim_session_id = if claim_session_id.is_empty() {
            None
        } else {
            Some(claim_session_id)
        };

        let episodic_event_id = if episodic::episodic_enabled(&ctx.pool).await {
            let id = episodic::emit_episodic_event(
                &ctx.pool,
                "memory_claim_created",
                json!({
                    "status": "pending",
                    "summary_snippet": format!("claim {}", kind),
                    "claim_id": claim_id.as_str(),
                }),
                None,
                None,
                conversation_id.as_deref(),
                Some(&scope_str),
                source_type,
                source_ref,
                None,
                None,
            )
            .await
            .unwrap_or_default();
            if id.is_empty() { None } else { Some(id) }
        } else {
            None
        };

        let _ = sqlx::query(
            "INSERT INTO memory_claims (id, kind, scope, session_id, claim_text, rel_type_raw, rel_type_norm, rel_type_id, status, source_type, source_ref, episodic_event_id, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
        .bind(&claim_id)
        .bind(kind)
        .bind(scope_str)
        .bind(claim_session_id)
        .bind(claim_text)
        .bind(rel_type_raw)
        .bind(rel_type_norm)
        .bind(rel_type_id)
        .bind(source_type)
        .bind(source_ref)
        .bind(episodic_event_id.as_deref())
        .execute(&ctx.pool)
        .await;

        claims::record_claim_outcome(&ctx.pool, &claim_id, "pending", Some("pending")).await;

        result.claim_ids.push(claim_id);
        return Ok(WriteResult::Ignored("ClaimCreated".to_string()));
    }

    // Fetch anchor entities from working set for neighborhood overlap scoring
    let anchor_entity_ids = resolver::get_working_set_anchors(&ctx.pool, 10)
        .await
        .unwrap_or_default();
    
    let resolve_ctx = ResolveContext {
        pool: ctx.pool.clone(),
        session_id: ctx.session_id.clone(),
        expected_type: None, // Could be inferred from context in future
        anchor_entity_ids,
    };
    
    let write_ctx = WriteContext {
        pool: ctx.pool.clone(),
        model_client: ctx.model_client.clone(),
        scope: effective_scope.clone(),
        source: ctx.source.clone(),
        source_ref: effective_source_ref.clone(),
        now: ctx.now,
        embedding_config: ctx.embedding_config.clone(),
        conversation_id,
    };


    match stmt {
        DslStatement::Fact(mut fact) => {
            let value_trimmed = fact.value.trim();
            let value_is_entity_ref = value_trimmed.starts_with('#') || value_trimmed.starts_with('$');
            let identity_handle = identity_handle(&fact.subject, &fact.key, &fact.polarity);
            let identity_label = identity_handle
                .as_ref()
                .map(|_| fact.value.trim().to_string())
                .filter(|label| !label.is_empty());
            let ref_text = format!("{:?}", fact.subject);
            match resolver::resolve_ref(&fact.subject, &resolve_ctx).await.map_err(|e| e.to_string())? {
                ResolutionResult::Resolved(subject_id) => {
                    // §6.3: Try FACT→REL promotion before writing fact
                    if let Some(promo) = crate::core::memory::promotion::try_promote_fact(&fact, subject_id, &resolve_ctx, &write_ctx).await {
                        let mut rel_stmt = RelStmt {
                            rel_type: promo.rel_type.clone(),
                            rel_type_id: None,
                            participants: vec![],
                            direction: promo.direction,
                            certainty: fact.certainty,
                            time_expr: fact.time_expr.clone(),
                            scope_expr: fact.scope_expr.clone(),
                            source_ref: fact.source_ref.clone(),
                            polarity: fact.polarity.clone(),
                        };
                        if promo.low_confidence {
                            rel_stmt.certainty = low_confidence(fact.certainty);
                        }
                        let rel_result = write_rel_from_resolved(ctx, &write_ctx, rel_stmt, promo.participants, original_dsl).await?;
                        return Ok(rel_result);
                    }

                    if value_is_entity_ref {
                        return Err(format!(
                            "InvalidFactValue: fact values cannot be entity references (#/$). Use a relation for key '{}'.",
                            fact.key
                        ));
                    }
                    
                    // No promotion - write as fact
                    let write_result = writer::write_fact(fact, subject_id, &write_ctx).await;
                    if let (Some(handle), Some(label)) = (identity_handle.as_deref(), identity_label.as_deref()) {
                        if matches!(write_result, WriteResult::Inserted(_) | WriteResult::Updated(_) | WriteResult::Conflict { .. }) {
                            if let Err(e) = bind_identity_name(&ctx.pool, &ctx.session_id, handle, subject_id, label).await {
                                result.errors.push(format!("IdentityBinding: {}", e));
                            }
                        }
                    }
                    Ok(write_result)
                },
                ResolutionResult::NewEntity(label) => {
                    match writer::create_entity(&label, None, &write_ctx).await {
                        Ok(new_id) => {
                            // Try promotion for new entity too
                            if let Some(promo) = crate::core::memory::promotion::try_promote_fact(&fact, new_id, &resolve_ctx, &write_ctx).await {
                                let mut rel_stmt = RelStmt {
                                    rel_type: promo.rel_type.clone(),
                                    rel_type_id: None,
                                    participants: vec![],
                                    direction: promo.direction,
                                    certainty: fact.certainty,
                                    time_expr: fact.time_expr.clone(),
                                    scope_expr: fact.scope_expr.clone(),
                                    source_ref: fact.source_ref.clone(),
                                    polarity: fact.polarity.clone(),
                                };
                                if promo.low_confidence {
                                    rel_stmt.certainty = low_confidence(fact.certainty);
                                }
                                let rel_result = write_rel_from_resolved(ctx, &write_ctx, rel_stmt, promo.participants, original_dsl).await?;
                                return Ok(rel_result);
                            }

                            if value_is_entity_ref {
                                return Err(format!(
                                    "InvalidFactValue: fact values cannot be entity references (#/$). Use a relation for key '{}'.",
                                    fact.key
                                ));
                            }
                            let write_result = writer::write_fact(fact, new_id, &write_ctx).await;
                            if let (Some(handle), Some(label)) = (identity_handle.as_deref(), identity_label.as_deref()) {
                                if matches!(write_result, WriteResult::Inserted(_) | WriteResult::Updated(_) | WriteResult::Conflict { .. }) {
                                    if let Err(e) = bind_identity_name(&ctx.pool, &ctx.session_id, handle, new_id, label).await {
                                        result.errors.push(format!("IdentityBinding: {}", e));
                                    }
                                }
                            }
                            Ok(write_result)
                        },
                        Err(e) => Err(format!("Failed to create entity '{}': {}", label, e)),
                    }
                },
                ResolutionResult::Ambiguous(pending) => {
                    if ctx.allow_ambiguous_user_refs && matches!(ctx.source, SourceType::User) {
                        if let Some(label) = label_from_ref(&fact.subject) {
                            fact.certainty = low_confidence(fact.certainty);
                            match writer::create_entity(&label, None, &write_ctx).await {
                                Ok(new_id) => {
                                    if let Some(promo) = crate::core::memory::promotion::try_promote_fact(&fact, new_id, &resolve_ctx, &write_ctx).await {
                                        let mut rel_stmt = RelStmt {
                                            rel_type: promo.rel_type.clone(),
                                            rel_type_id: None,
                                            participants: vec![],
                                            direction: promo.direction,
                                            certainty: fact.certainty,
                                            time_expr: fact.time_expr.clone(),
                                            scope_expr: fact.scope_expr.clone(),
                                            source_ref: fact.source_ref.clone(),
                                            polarity: fact.polarity.clone(),
                                        };
                                        if promo.low_confidence {
                                            rel_stmt.certainty = low_confidence(fact.certainty);
                                        }
                                        let rel_result = write_rel_from_resolved(ctx, &write_ctx, rel_stmt, promo.participants, original_dsl).await?;
                                        return Ok(rel_result);
                                    }

                                    if value_is_entity_ref {
                                        return Err(format!(
                                            "InvalidFactValue: fact values cannot be entity references (#/$). Use a relation for key '{}'.",
                                            fact.key
                                        ));
                                    }
                                    let write_result = writer::write_fact(fact, new_id, &write_ctx).await;
                                    if let (Some(handle), Some(label)) = (identity_handle.as_deref(), identity_label.as_deref()) {
                                        if matches!(write_result, WriteResult::Inserted(_) | WriteResult::Updated(_) | WriteResult::Conflict { .. }) {
                                            if let Err(e) = bind_identity_name(&ctx.pool, &ctx.session_id, handle, new_id, label).await {
                                                result.errors.push(format!("IdentityBinding: {}", e));
                                            }
                                        }
                                    }
                                    return Ok(write_result);
                                }
                                Err(e) => {
                                    return Err(format!("Failed to create entity '{}': {}", label, e));
                                }
                            }
                        }
                    }
                    if should_drop_clarify(&ctx.pool, &ctx.session_id, &ref_text)
                        .await
                        .unwrap_or(false)
                    {
                        result.errors.push("ClarifyDropped: attempts_exceeded".to_string());
                        return Err("PENDING_CLARIFY_DROPPED".to_string());
                    }
                    let _ = bump_clarify_attempt(&ctx.pool, &ctx.session_id, &ref_text).await;
                    // Store PendingClarify in DB
                    let candidates_json = &pending.candidates_json;
                    let insert_res = sqlx::query(
                        "INSERT INTO ics_pending_clarify (session_id, original_dsl, ref_text, candidates_json, status) 
                         VALUES (?, ?, ?, ?, 'pending') RETURNING id"
                    )
                    .bind(&ctx.session_id)
                    .bind(original_dsl)
                    .bind(&ref_text)
                    .bind(candidates_json)
                    .fetch_one(&ctx.pool)
                    .await;
                    
                    if let Ok(row) = insert_res {
                        let pending_id: i64 = row.get("id");
                        
                        // Parse candidates for result
                        let candidates: Vec<ClarifyCandidate> = serde_json::from_str(candidates_json)
                            .unwrap_or_default();
                        
                        result.pending_clarify = Some(PendingClarify {
                            id: pending_id,
                            ref_text: ref_text.clone(),
                            candidates,
                            original_dsl: original_dsl.to_string(),
                        });
                    }
                    
                    Err("PENDING_CLARIFY: Awaiting user clarification".to_string())
                }
            }
        },
        DslStatement::Rel(rel) => {
            let RelStmt {
                rel_type,
                participants,
                direction,
                certainty,
                time_expr,
                scope_expr,
                source_ref,
                polarity,
                rel_type_id,
            } = rel;
            let mut certainty = certainty;

            let canonicalization_on = relation_canonicalization_enabled(&ctx.pool).await;
            let rel_type_raw = rel_type.clone();
            let alias_map = if canonicalization_on {
                load_role_alias_map(&ctx.pool).await
            } else {
                HashMap::new()
            };

            let mut roles_seen = Vec::new();
            let mut canonicalized_participants = Vec::new();
            for (role, p_ref) in participants {
                let canonical_role = if canonicalization_on {
                    canonicalize_role_token(&role, &alias_map)
                } else {
                    role
                };
                roles_seen.push(canonical_role.clone());
                canonicalized_participants.push((canonical_role, p_ref));
            }

            let rel_type_resolved = if let Some(provided) = rel_type_id.clone().filter(|id| !id.trim().is_empty()) {
                rel_type_catalog::resolve_rel_type_from_id(
                    &ctx.pool,
                    &provided,
                    &rel_type_raw,
                    canonicalization_on,
                )
                .await?
            } else {
                rel_type_catalog::resolve_rel_type(
                    &ctx.pool,
                    &rel_type_raw,
                    &roles_seen,
                    canonicalization_on,
                )
                .await?
            };
            let rel_type_norm = rel_type_resolved.rel_type_norm.clone();
            let rel_type_id = rel_type_resolved.rel_type_id.clone();

            let sample_dsl = format_rel_sample(&rel_type_norm, &canonicalized_participants, direction.as_ref());
            let shape = resolve_relation_shape(
                ctx,
                &rel_type_id,
                &rel_type_norm,
                &roles_seen,
                &direction,
                &sample_dsl,
                &effective_scope,
                canonicalization_on,
            )
            .await?;

            let mut resolved_participants = vec![];
            let mut used_ambiguous_fallback = false;
            for (role, p_ref) in canonicalized_participants {
                 match resolver::resolve_ref(&p_ref, &resolve_ctx).await.map_err(|e| e.to_string())? {
                    ResolutionResult::Resolved(id) => resolved_participants.push((role, id)),
                    ResolutionResult::NewEntity(label) => {
                        let inferred_type = infer_type_from_role(&role);
                        match writer::create_entity(&label, inferred_type, &write_ctx).await {
                            Ok(id) => resolved_participants.push((role, id)),
                            Err(e) => return Err(format!("Failed to create rel participant '{}': {}", label, e)),
                        }
                    },
                    ResolutionResult::Ambiguous(pending) => {
                        if ctx.allow_ambiguous_user_refs && matches!(ctx.source, SourceType::User) {
                            if let Some(label) = label_from_ref(&p_ref) {
                                let inferred_type = infer_type_from_role(&role);
                                match writer::create_entity(&label, inferred_type, &write_ctx).await {
                                    Ok(id) => {
                                        resolved_participants.push((role, id));
                                        used_ambiguous_fallback = true;
                                        continue;
                                    }
                                    Err(e) => {
                                        return Err(format!("Failed to create rel participant '{}': {}", label, e));
                                    }
                                }
                            }
                        }
                        let ref_text = format!("{:?}", p_ref);
                        if should_drop_clarify(&ctx.pool, &ctx.session_id, &ref_text)
                            .await
                            .unwrap_or(false)
                        {
                            result.errors.push("ClarifyDropped: attempts_exceeded".to_string());
                            return Err("PENDING_CLARIFY_DROPPED".to_string());
                        }
                        let _ = bump_clarify_attempt(&ctx.pool, &ctx.session_id, &ref_text).await;
                        let candidates_json = &pending.candidates_json;
                        let insert_res = sqlx::query(
                            "INSERT INTO ics_pending_clarify (session_id, original_dsl, ref_text, candidates_json, status) 
                             VALUES (?, ?, ?, ?, 'pending') RETURNING id"
                        )
                        .bind(&ctx.session_id)
                        .bind(original_dsl)
                        .bind(&ref_text)
                        .bind(candidates_json)
                        .fetch_one(&ctx.pool)
                        .await;

                        if let Ok(row) = insert_res {
                            let pending_id: i64 = row.get("id");
                            let candidates: Vec<ClarifyCandidate> = serde_json::from_str(candidates_json)
                                .unwrap_or_default();

                            result.pending_clarify = Some(PendingClarify {
                                id: pending_id,
                                ref_text: ref_text.clone(),
                                candidates,
                                original_dsl: original_dsl.to_string(),
                            });
                        }

                        return Err("PENDING_CLARIFY: Awaiting user clarification".to_string());
                    },
                 }
            }

            if canonicalization_on && shape.commutative {
                resolved_participants.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            }

            if used_ambiguous_fallback {
                certainty = low_confidence(certainty);
            }

            let options = RelationWriteOptions {
                sort_participants: !canonicalization_on,
            };
            Ok(writer::write_rel(RelStmt {
                rel_type: rel_type_norm,
                rel_type_id: Some(rel_type_id.clone()),
                participants: vec![],
                direction,
                certainty,
                time_expr,
                scope_expr,
                source_ref,
                polarity,
            }, resolved_participants, &shape, options, &write_ctx, Some(&rel_type_raw), Some(&rel_type_id)).await)
        }
    }
}

pub(crate) async fn memory_claims_enabled(pool: &SqlitePool) -> bool {
    let enabled: Option<i32> = sqlx::query_scalar("SELECT memory_claims_enabled FROM settings WHERE id = 1")
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();
    enabled.unwrap_or(0) != 0
}

fn source_type_str(source: &SourceType) -> &'static str {
    match source {
        SourceType::User => "user",
        SourceType::Tool => "tool",
        SourceType::System => "system",
        SourceType::Inference => "inference",
    }
}

fn claims_gate_source(source: &SourceType) -> bool {
    matches!(source, SourceType::User | SourceType::Inference)
}

pub(crate) async fn relation_canonicalization_enabled(pool: &SqlitePool) -> bool {
    let key = "feature.relation_canonicalization_v1";
    let stored: Option<String> = sqlx::query_scalar("SELECT value FROM kv_store WHERE key = ?")
        .bind(key)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();
    if let Some(value) = stored {
        let normalized = value.trim().to_lowercase();
        return normalized == "true" || normalized == "1" || normalized == "yes";
    }

    let existing: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ics_rel_beliefs")
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0);
    let enabled = existing == 0;
    let _ = sqlx::query(
        "INSERT INTO kv_store (key, value, updated_at)
         VALUES (?, ?, CURRENT_TIMESTAMP)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP"
    )
    .bind(key)
    .bind(if enabled { "true" } else { "false" })
    .execute(pool)
    .await;
    enabled
}

pub(crate) async fn load_role_alias_map(pool: &SqlitePool) -> HashMap<String, String> {
    let rows = sqlx::query("SELECT from_role, to_role FROM ics_role_aliases WHERE status = 'confirmed'")
        .fetch_all(pool)
        .await
        .unwrap_or_default();
    let mut map = HashMap::new();
    for row in rows {
        let from_role: String = row.get("from_role");
        let to_role: String = row.get("to_role");
        let from_norm = normalize_role_token(&from_role);
        let to_norm = normalize_role_token(&to_role);
        if from_norm.is_empty() || to_norm.is_empty() {
            continue;
        }
        if from_norm == to_norm {
            continue;
        }
        map.insert(from_norm, to_norm);
    }
    map
}

fn normalize_role_list(raw: Vec<String>) -> Vec<String> {
    let mut roles = Vec::new();
    let mut seen = HashSet::new();
    for role in raw {
        let normalized = normalize_role_token(&role);
        if normalized.is_empty() {
            continue;
        }
        if seen.insert(normalized.clone()) {
            roles.push(normalized);
        }
    }
    roles
}

fn parse_cardinality(raw: Option<String>) -> Option<Cardinality> {
    let normalized = raw?.trim().to_uppercase();
    match normalized.as_str() {
        "ONE" => Some(Cardinality::One),
        "MANY" | "MANY_SET" => Some(Cardinality::ManySet),
        "TIME_SERIES" | "TIMESERIES" => Some(Cardinality::TimeSeries),
        _ => None,
    }
}

async fn load_relation_shape(pool: &SqlitePool, rel_type_id: &str, rel_type: &str) -> Result<Option<RelationShape>, String> {
    let row = sqlx::query(
        "SELECT roles, anchor_roles, cardinality_override, commutative, expected_arity, status
         FROM rel_shape WHERE rel_type_id = ?"
    )
    .bind(rel_type_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    if let Some(row) = row {
        let roles_raw: Option<String> = row.try_get("roles").ok();
        let anchor_raw: Option<String> = row.try_get("anchor_roles").ok();
        let cardinality_raw: Option<String> = row.try_get("cardinality_override").ok();
        let commutative: i64 = row.try_get("commutative").unwrap_or(0);
        let expected_arity: Option<i64> = row.try_get("expected_arity").ok();
        let status: Option<String> = row.try_get("status").ok();

        let roles_vec: Vec<String> = if let Some(raw) = roles_raw {
            serde_json::from_str(&raw)
                .map_err(|e| format!("RelationShapeParseError: rel_type '{}' roles: {}", rel_type, e))?
        } else {
            Vec::new()
        };
        let anchor_vec: Vec<String> = if let Some(raw) = anchor_raw {
            serde_json::from_str(&raw)
                .map_err(|e| format!("RelationShapeParseError: rel_type '{}' anchor_roles: {}", rel_type, e))?
        } else {
            Vec::new()
        };

        return Ok(Some(RelationShape {
            rel_type_id: Some(rel_type_id.to_string()),
            rel_type: rel_type.to_string(),
            roles: normalize_role_list(roles_vec),
            anchor_roles: normalize_role_list(anchor_vec),
            cardinality_override: parse_cardinality(cardinality_raw),
            commutative: commutative != 0,
            expected_arity,
            status: status.unwrap_or_else(|| "seeded".to_string()),
        }));
    }

    // Legacy fallback: load from ics_relation_shapes if rel_shape is missing.
    let legacy = sqlx::query(
        "SELECT roles, anchor_roles, cardinality_override, commutative, expected_arity, status
         FROM ics_relation_shapes WHERE rel_type = ?"
    )
    .bind(rel_type)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    let Some(row) = legacy else {
        return Ok(None);
    };

    let roles_raw: Option<String> = row.try_get("roles").ok();
    let anchor_raw: Option<String> = row.try_get("anchor_roles").ok();
    let cardinality_raw: Option<String> = row.try_get("cardinality_override").ok();
    let commutative: i64 = row.try_get("commutative").unwrap_or(0);
    let expected_arity: Option<i64> = row.try_get("expected_arity").ok();
    let status: Option<String> = row.try_get("status").ok();

    let roles_vec: Vec<String> = if let Some(raw) = roles_raw.clone() {
        serde_json::from_str(&raw)
            .map_err(|e| format!("RelationShapeParseError: rel_type '{}' roles: {}", rel_type, e))?
    } else {
        Vec::new()
    };
    let anchor_vec: Vec<String> = if let Some(raw) = anchor_raw.clone() {
        serde_json::from_str(&raw)
            .map_err(|e| format!("RelationShapeParseError: rel_type '{}' anchor_roles: {}", rel_type, e))?
    } else {
        Vec::new()
    };

    // Best-effort sync into rel_shape to keep tables aligned.
    let roles_json = roles_raw.unwrap_or_else(|| "[]".to_string());
    let anchor_json = anchor_raw.unwrap_or_else(|| "[]".to_string());
    let cardinality_raw_bind = cardinality_raw.clone();
    let _ = sqlx::query(
        "INSERT OR IGNORE INTO rel_shape
         (rel_type_id, roles, anchor_roles, cardinality_override, commutative, expected_arity, status, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)"
    )
    .bind(rel_type_id)
    .bind(&roles_json)
    .bind(&anchor_json)
    .bind(cardinality_raw_bind)
    .bind(commutative)
    .bind(expected_arity)
    .bind(status.clone().unwrap_or_else(|| "seeded".to_string()))
    .execute(pool)
    .await;

    Ok(Some(RelationShape {
        rel_type_id: Some(rel_type_id.to_string()),
        rel_type: rel_type.to_string(),
        roles: normalize_role_list(roles_vec),
        anchor_roles: normalize_role_list(anchor_vec),
        cardinality_override: parse_cardinality(cardinality_raw),
        commutative: commutative != 0,
        expected_arity,
        status: status.unwrap_or_else(|| "seeded".to_string()),
    }))
}

fn default_relation_shape(rel_type_id: &str, rel_type: &str) -> RelationShape {
    RelationShape {
        rel_type_id: Some(rel_type_id.to_string()),
        rel_type: rel_type.to_string(),
        roles: Vec::new(),
        anchor_roles: Vec::new(),
        cardinality_override: None,
        commutative: false,
        expected_arity: None,
        status: "provisional".to_string(),
    }
}

fn validate_relation_shape(
    rel_type: &str,
    roles_seen: &[String],
    direction: &Option<RelDirection>,
    shape: &RelationShape,
) -> Result<(), String> {
    if shape.roles.is_empty() {
        return Err(format!("RelationShapeInvalid: rel_type '{}' has no roles", rel_type));
    }

    if shape.commutative && direction.is_some() {
        return Err(format!(
            "RelationDirectionInvalid: rel_type '{}' is commutative and cannot be directed",
            rel_type
        ));
    }

    if shape.commutative {
        let unique_roles: HashSet<_> = shape.roles.iter().cloned().collect();
        if unique_roles.len() > 1 {
            return Err(format!(
                "RelationShapeInvalid: rel_type '{}' is commutative but roles are not symmetric",
                rel_type
            ));
        }
    }

    let shape_set: HashSet<String> = shape.roles.iter().cloned().collect();
    let seen_set: HashSet<String> = roles_seen.iter().cloned().collect();

    let mut missing: Vec<String> = shape_set.difference(&seen_set).cloned().collect();
    let mut extra: Vec<String> = seen_set.difference(&shape_set).cloned().collect();
    missing.sort();
    extra.sort();

    if !extra.is_empty() {
        return Err(format!(
            "RelationRoleInvalid: rel_type '{}' unexpected roles {:?}",
            rel_type, extra
        ));
    }
    if !missing.is_empty() {
        return Err(format!(
            "RelationRoleMissing: rel_type '{}' missing roles {:?}",
            rel_type, missing
        ));
    }

    if !shape.anchor_roles.is_empty() {
        let anchor_set: HashSet<String> = shape.anchor_roles.iter().cloned().collect();
        let mut anchor_extra: Vec<String> = anchor_set.difference(&shape_set).cloned().collect();
        anchor_extra.sort();
        if !anchor_extra.is_empty() {
            return Err(format!(
                "RelationShapeInvalid: rel_type '{}' anchor_roles not in roles {:?}",
                rel_type, anchor_extra
            ));
        }
    }

    Ok(())
}

fn direction_label(direction: &Option<RelDirection>) -> &'static str {
    match direction {
        Some(RelDirection::Directed) => "directed",
        Some(RelDirection::Bidirectional) => "bidirectional",
        None => "none",
    }
}

pub(crate) fn format_rel_sample(
    rel_type: &str,
    participants: &[(String, dsl::Ref)],
    direction: Option<&RelDirection>,
) -> String {
    if let Some(dir) = direction {
        if participants.len() == 2 {
            let left = format_rel_participant(&participants[0]);
            let right = format_rel_participant(&participants[1]);
            let arrow = match dir {
                RelDirection::Directed => "->",
                RelDirection::Bidirectional => "<->",
            };
            return format!("{}({} {} {})", rel_type, left, arrow, right);
        }
    }

    let args = participants
        .iter()
        .map(format_rel_participant)
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}({})", rel_type, args)
}

fn format_rel_participant(participant: &(String, dsl::Ref)) -> String {
    format!("{}: {}", participant.0, format_rel_ref(&participant.1))
}

fn format_rel_ref(r: &dsl::Ref) -> String {
    match r {
        dsl::Ref::Handle(handle) => format!("${}", handle),
        dsl::Ref::Label(label) => format!("#{}", label),
        dsl::Ref::Filter(label, _key) => format!("#{}", label),
        dsl::Ref::Name(name) => format!("\"{}\"", escape_quotes(name)),
    }
}

fn escape_quotes(input: &str) -> String {
    input.replace('\\', "\\\\").replace('"', "\\\"")
}

async fn emit_relation_shape_missing(
    ctx: &CompileContext,
    rel_type: &str,
    roles_seen: &[String],
    direction: &Option<RelDirection>,
    sample_dsl: &str,
    scope: &Scope,
) {
    if !episodic::episodic_enabled(&ctx.pool).await {
        return;
    }

    let scope_str = serde_json::to_string(scope).unwrap_or_default();
    let summary_snippet = if sample_dsl.trim().is_empty() {
        format!("relation_shape_missing: {}", rel_type)
    } else {
        sample_dsl.trim().to_string()
    };

    let conversation_id = ctx.session_id.trim();
    let conversation_id = if conversation_id.is_empty() {
        None
    } else {
        Some(conversation_id)
    };

    let _ = episodic::emit_episodic_event(
        &ctx.pool,
        "relation_shape_missing",
        json!({
            "status": "missing",
            "summary_snippet": summary_snippet,
            "rel_type": rel_type,
            "roles_seen": roles_seen.join(", "),
            "direction": direction_label(direction),
            "sample_dsl": sample_dsl,
            "session_id": ctx.session_id.as_str(),
            "timestamp": ctx.now.to_rfc3339(),
            "source_ref": ctx.source_ref.as_deref(),
        }),
        None,
        None,
        conversation_id,
        Some(&scope_str),
        source_type_str(&ctx.source),
        ctx.source_ref.as_deref(),
        None,
        None,
    )
    .await;
}

async fn emit_relation_shape_mismatch(
    ctx: &CompileContext,
    rel_type: &str,
    roles_seen: &[String],
    direction: &Option<RelDirection>,
    sample_dsl: &str,
    scope: &Scope,
    error: &str,
) {
    if !episodic::episodic_enabled(&ctx.pool).await {
        return;
    }

    let scope_str = serde_json::to_string(scope).unwrap_or_default();
    let summary_snippet = if sample_dsl.trim().is_empty() {
        format!("relation_shape_mismatch: {}", rel_type)
    } else {
        sample_dsl.trim().to_string()
    };

    let conversation_id = ctx.session_id.trim();
    let conversation_id = if conversation_id.is_empty() {
        None
    } else {
        Some(conversation_id)
    };

    let _ = episodic::emit_episodic_event(
        &ctx.pool,
        "relation_shape_mismatch",
        json!({
            "status": "mismatch",
            "summary_snippet": summary_snippet,
            "rel_type": rel_type,
            "roles_seen": roles_seen.join(", "),
            "direction": direction_label(direction),
            "error": error,
            "sample_dsl": sample_dsl,
            "session_id": ctx.session_id.as_str(),
            "timestamp": ctx.now.to_rfc3339(),
            "source_ref": ctx.source_ref.as_deref(),
        }),
        None,
        None,
        conversation_id,
        Some(&scope_str),
        source_type_str(&ctx.source),
        ctx.source_ref.as_deref(),
        None,
        None,
    )
    .await;
}

async fn try_repair_relation_shape(
    ctx: &CompileContext,
    rel_type_id: &str,
    rel_type: &str,
    roles_seen: &[String],
) -> bool {
    if rel_type_id.trim().is_empty() {
        return false;
    }
    if roles_seen.is_empty() {
        return false;
    }

    let roles_norm = normalize_role_list(roles_seen.to_vec());
    if roles_norm.is_empty() {
        return false;
    }

    let roles_json = serde_json::to_string(&roles_norm).unwrap_or_else(|_| "[]".to_string());
    let expected_arity = roles_norm.len() as i64;

    let mut tx = match ctx.pool.begin().await {
        Ok(tx) => tx,
        Err(_) => return false,
    };

    let _ = sqlx::query(
        "UPDATE rel_shape
         SET roles = ?, expected_arity = ?, status = 'provisional'
         WHERE rel_type_id = ?"
    )
    .bind(&roles_json)
    .bind(expected_arity)
    .bind(rel_type_id)
    .execute(&mut *tx)
    .await;

    let _ = sqlx::query(
        "UPDATE ics_relation_shapes
         SET roles = ?, expected_arity = ?, status = 'provisional'
         WHERE rel_type = ?"
    )
    .bind(&roles_json)
    .bind(expected_arity)
    .bind(rel_type)
    .execute(&mut *tx)
    .await;

    let _ = tx.commit().await;
    true
}

pub(crate) async fn resolve_relation_shape(
    ctx: &CompileContext,
    rel_type_id: &str,
    rel_type: &str,
    roles_seen: &[String],
    direction: &Option<RelDirection>,
    sample_dsl: &str,
    scope: &Scope,
    canonicalization_on: bool,
) -> Result<RelationShape, String> {
    let shape = load_relation_shape(&ctx.pool, rel_type_id, rel_type).await?;
    match shape {
            Some(shape) => {
                if canonicalization_on {
                    if let Err(err) = validate_relation_shape(rel_type, roles_seen, direction, &shape) {
                        emit_relation_shape_mismatch(ctx, rel_type, roles_seen, direction, sample_dsl, scope, &err).await;
                        if shape.status != "seeded" && try_repair_relation_shape(ctx, rel_type_id, rel_type, roles_seen).await {
                            if let Some(repaired) = load_relation_shape(&ctx.pool, rel_type_id, rel_type).await? {
                                return Ok(repaired);
                            }
                        }
                    }
                    if let Some(expected) = shape.expected_arity {
                        if expected as usize != roles_seen.len() {
                            emit_relation_shape_mismatch(
                                ctx,
                                rel_type,
                                roles_seen,
                                direction,
                                sample_dsl,
                                scope,
                                &format!("RelationArityMismatch: expected {}, saw {}", expected, roles_seen.len()),
                            )
                            .await;
                            if shape.status != "seeded" && try_repair_relation_shape(ctx, rel_type_id, rel_type, roles_seen).await {
                                if let Some(repaired) = load_relation_shape(&ctx.pool, rel_type_id, rel_type).await? {
                                    return Ok(repaired);
                                }
                            }
                        }
                    }
                }
                Ok(shape)
            }
            None => {
                if canonicalization_on {
                    emit_relation_shape_missing(ctx, rel_type, roles_seen, direction, sample_dsl, scope).await;
                    let roles_norm = normalize_role_list(roles_seen.to_vec());
                    let roles_json = serde_json::to_string(&roles_norm).unwrap_or_else(|_| "[]".to_string());
                    let expected_arity = roles_seen.len() as i64;

                    let mut tx = ctx.pool.begin().await.map_err(|e| e.to_string())?;
                    let _ = sqlx::query(
                        "INSERT INTO rel_shape (rel_type_id, roles, anchor_roles, commutative, expected_arity, status, created_at)
                         VALUES (?, ?, '[]', 0, ?, 'provisional', CURRENT_TIMESTAMP)
                         ON CONFLICT(rel_type_id) DO NOTHING"
                    )
                    .bind(rel_type_id)
                    .bind(&roles_json)
                    .bind(expected_arity)
                    .execute(&mut *tx)
                    .await;

                    let _ = sqlx::query(
                        "INSERT INTO ics_relation_shapes (rel_type, roles, anchor_roles, cardinality_override, commutative, expected_arity, status, created_at)
                         VALUES (?, ?, '[]', NULL, 0, ?, 'provisional', CURRENT_TIMESTAMP)
                         ON CONFLICT(rel_type) DO NOTHING"
                    )
                    .bind(rel_type)
                    .bind(&roles_json)
                    .bind(expected_arity)
                    .execute(&mut *tx)
                    .await;

                    let _ = tx.commit().await;

                    if let Some(shape) = load_relation_shape(&ctx.pool, rel_type_id, rel_type).await? {
                        return Ok(shape);
                }
            }
            Ok(default_relation_shape(rel_type_id, rel_type))
        }
    }
}

async fn write_rel_from_resolved(
    ctx: &CompileContext,
    write_ctx: &WriteContext,
    rel_stmt: RelStmt,
    participants: Vec<(String, i64)>,
    sample_dsl: &str,
) -> Result<WriteResult, String> {
    let canonicalization_on = relation_canonicalization_enabled(&ctx.pool).await;
    let rel_type_raw = rel_stmt.rel_type.clone();
    let alias_map = if canonicalization_on {
        load_role_alias_map(&ctx.pool).await
    } else {
        HashMap::new()
    };

    let mut roles_seen = Vec::new();
    let mut canonicalized_participants = Vec::new();
    for (role, id) in participants {
        let canonical_role = if canonicalization_on {
            canonicalize_role_token(&role, &alias_map)
        } else {
            role
        };
        roles_seen.push(canonical_role.clone());
        canonicalized_participants.push((canonical_role, id));
    }

    let rel_type_resolved = rel_type_catalog::resolve_rel_type(
        &ctx.pool,
        &rel_type_raw,
        &roles_seen,
        canonicalization_on,
    )
    .await?;
    let rel_type_norm = rel_type_resolved.rel_type_norm.clone();
    let rel_type_id = rel_type_resolved.rel_type_id.clone();

    let mut rel_stmt_norm = rel_stmt;
    rel_stmt_norm.rel_type = rel_type_norm.clone();
    rel_stmt_norm.rel_type_id = Some(rel_type_id.clone());

    let shape = resolve_relation_shape(
        ctx,
        &rel_type_id,
        &rel_type_norm,
        &roles_seen,
        &rel_stmt_norm.direction,
        sample_dsl,
        &write_ctx.scope,
        canonicalization_on,
    )
    .await?;

    if canonicalization_on && shape.commutative {
        canonicalized_participants.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    }

    let options = RelationWriteOptions {
        sort_participants: !canonicalization_on,
    };
    Ok(writer::write_rel(
        rel_stmt_norm,
        canonicalized_participants,
        &shape,
        options,
        write_ctx,
        Some(&rel_type_raw),
        Some(&rel_type_id),
    )
    .await)
}
