//! Disambiguation Resolution Module (ICS v4.1 §6.2)
//! Handles user clarification responses and commits pending writes.

use sqlx::SqlitePool;
use sqlx::Row;
use crate::core::memory::types::{Scope, SourceType};
use crate::core::memory::compiler::{self, CompileContext, CompileResult, ClarifyCandidate};
use crate::core::memory::claims;
use crate::core::memory::writer::EmbeddingConfig;
use crate::core::memory::canonical::canonicalize_label;
use crate::core::episodic;
use std::sync::Arc;
use crate::core::model_client::ModelClient;
use std::collections::HashSet;

/// Result of resolving a clarification
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ClarifyResult {
    pub success: bool,
    pub selected_entity_id: Option<i64>,
    pub selected_label: Option<String>,
    pub compile_result: Option<CompileResult>,
    pub error: Option<String>,
}

/// User's choice from clarification options
#[derive(Debug)]
pub enum ClarifyChoice {
    /// User selected a specific candidate by ID
    SelectById(i64),
    /// User wants to create a new entity
    CreateNew(String),
    /// User cancelled/skipped the clarification
    Cancel,
}

/// Parse user reply to determine their choice
/// Supports multiple formats per ICS v4.1 §6.2:
/// - Option numbers: "1", "option 1", "#1"
/// - Exact label matches: "Alice", "Sarah (sister)"
/// - Create new keywords: "none", "neither", "create new", "new"
/// - Cancel keywords: "cancel", "skip", "nevermind"
pub fn parse_user_reply(reply: &str, candidates: &[ClarifyCandidate]) -> ClarifyChoice {
    let reply_lower = reply.trim().to_lowercase();
    
    // Check for cancel keywords
    if matches!(reply_lower.as_str(), "cancel" | "skip" | "nevermind" | "abort") {
        return ClarifyChoice::Cancel;
    }
    
    // Check for create new keywords
    if reply_lower.starts_with("create new")
        || reply_lower.starts_with("new ")
        || matches!(reply_lower.as_str(), "none" | "neither" | "none of these" | "new")
    {
        // Extract custom label if provided after "create new:" or "new:"
        let label = if reply_lower.starts_with("create new:") {
            reply.trim()[11..].trim().to_string()
        } else if reply_lower.starts_with("new:") {
            reply.trim()[4..].trim().to_string()
        } else {
            String::new() // Will use original ref_text
        };
        return ClarifyChoice::CreateNew(label);
    }
    
    // Check for option number
    // Patterns: "1", "2", "#1", "option 1", "opt 1"
    let num_str = reply_lower
        .replace("option", "")
        .replace("opt", "")
        .replace("#", "")
        .trim()
        .to_string();
    
    if let Ok(num) = num_str.parse::<usize>() {
        // 1-indexed to 0-indexed
        if num >= 1 && num <= candidates.len() {
            return ClarifyChoice::SelectById(candidates[num - 1].entity_id);
        }
    }
    
    // Check for exact label match (case-insensitive)
    for candidate in candidates {
        if candidate.label.to_lowercase() == reply_lower {
            return ClarifyChoice::SelectById(candidate.entity_id);
        }
        // Also check if reply contains the label
        if reply_lower.contains(&candidate.label.to_lowercase()) {
            return ClarifyChoice::SelectById(candidate.entity_id);
        }
    }
    
    // Default: treat as create new with the reply as the label
    ClarifyChoice::CreateNew(reply.trim().to_string())
}

/// Resolve a pending clarification with the user's choice
/// Returns the result of committing the pending write
pub async fn resolve_clarify(
    pending_id: i64,
    user_reply: &str,
    pool: &SqlitePool,
    model_client: Option<Arc<ModelClient>>,
    scope: Scope,
    source: SourceType,
    embedding_config: Option<EmbeddingConfig>,
) -> ClarifyResult {
    // 1. Load pending clarification from DB
    let pending_row = match sqlx::query(
        "SELECT claim_id, session_id, original_dsl, ref_text, candidates_json, status
         FROM ics_pending_clarify WHERE id = ?"
    )
    .bind(pending_id)
    .fetch_optional(pool)
    .await
    {
        Ok(Some(row)) => row,
        Ok(None) => return ClarifyResult {
            success: false,
            selected_entity_id: None,
            selected_label: None,
            compile_result: None,
            error: Some(format!("Pending clarification {} not found", pending_id)),
        },
        Err(e) => return ClarifyResult {
            success: false,
            selected_entity_id: None,
            selected_label: None,
            compile_result: None,
            error: Some(format!("Database error: {}", e)),
        },
    };
    
    let status: String = pending_row.get("status");
    if status != "pending" {
        return ClarifyResult {
            success: false,
            selected_entity_id: None,
            selected_label: None,
            compile_result: None,
            error: Some(format!("Clarification {} already resolved (status: {})", pending_id, status)),
        };
    }
    
    let claim_id: Option<String> = pending_row.try_get("claim_id").ok();
    let session_id: String = pending_row.get("session_id");
    let original_dsl: String = pending_row.get("original_dsl");
    let ref_text: String = pending_row.get("ref_text");
    let candidates_json: String = pending_row.get("candidates_json");
    let conversation_id = session_id.trim();
    let conversation_id = if conversation_id.is_empty() {
        None
    } else {
        Some(conversation_id)
    };
    
    // 2. Parse candidates
    let candidates: Vec<ClarifyCandidate> = serde_json::from_str(&candidates_json)
        .unwrap_or_default();
    
    // 3. Parse user reply
    let choice = parse_user_reply(user_reply, &candidates);
    
    // 4. Handle the choice
    match choice {
        ClarifyChoice::Cancel => {
            // Mark as cancelled
            let _ = sqlx::query("UPDATE ics_pending_clarify SET status = 'cancelled' WHERE id = ?")
                .bind(pending_id)
                .execute(pool)
                .await;
            
            ClarifyResult {
                success: true,
                selected_entity_id: None,
                selected_label: None,
                compile_result: None,
                error: None,
            }
        }
        
        ClarifyChoice::SelectById(entity_id) => {
            let claims_enabled = compiler::memory_claims_enabled(pool).await;
            let claim_id = claim_id
                .as_deref()
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(|id| id.to_string());

            // Find the selected candidate's label
            let selected_label = candidates.iter()
                .find(|c| c.entity_id == entity_id)
                .map(|c| c.label.clone());
            
            // Create session binding for future resolution
            let _ = sqlx::query(
                "INSERT INTO ics_session_bindings (session_id, ref_text, entity_id, created_at)
                 VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                 ON CONFLICT(session_id, ref_text) DO UPDATE SET entity_id = excluded.entity_id"
            )
            .bind(&session_id)
            .bind(&ref_text)
            .bind(entity_id)
            .execute(pool)
            .await;

            let _ = maybe_add_entity_alias(pool, entity_id, &ref_text).await;
            
            let mut compile_result = None;
            if let Some(claim_id) = claim_id.as_deref() {
                if claims_enabled {
                    let _ = sqlx::query(
                        "UPDATE memory_claims
                         SET status = 'pending',
                             decision_reason = NULL,
                             conflict_topic_key = NULL,
                             conflict_reason = NULL,
                             updated_at = CURRENT_TIMESTAMP
                         WHERE id = ?"
                    )
                    .bind(claim_id)
                    .execute(pool)
                    .await;
                    if let Err(e) = claims::evaluate_claim(pool, model_client.clone(), claim_id).await {
                        return ClarifyResult {
                            success: false,
                            selected_entity_id: Some(entity_id),
                            selected_label,
                            compile_result: None,
                            error: Some(format!("Claim evaluation failed: {}", e)),
                        };
                    }
                } else {
                    // Re-compile the original DSL (now with binding in place)
                    let ctx = CompileContext {
                        pool: pool.clone(),
                        model_client,
                        session_id: session_id.clone(),
                        scope,
                        source,
                        source_ref: Some(format!("clarify:{}", pending_id)),
                        now: chrono::Utc::now(),
                        embedding_config: embedding_config.clone(),
                        skip_claims: false,
                        allow_ambiguous_user_refs: false,
                    };

                    compile_result = Some(compiler::compile(&original_dsl, ctx).await);
                }
            } else {
                // Re-compile the original DSL (now with binding in place)
                let ctx = CompileContext {
                    pool: pool.clone(),
                    model_client,
                    session_id: session_id.clone(),
                    scope,
                    source,
                    source_ref: Some(format!("clarify:{}", pending_id)),
                    now: chrono::Utc::now(),
                    embedding_config: embedding_config.clone(),
                    skip_claims: false,
                    allow_ambiguous_user_refs: false,
                };

                compile_result = Some(compiler::compile(&original_dsl, ctx).await);
            }

            // Mark as resolved
            let _ = sqlx::query("UPDATE ics_pending_clarify SET status = 'resolved' WHERE id = ?")
                .bind(pending_id)
                .execute(pool)
                .await;

            let source_ref = format!("clarify:{}", pending_id);
            let _ = episodic::emit_episodic_event(
                pool,
                "clarify_resolved",
                serde_json::json!({
                    "status": "resolved",
                    "summary_snippet": selected_label.clone().unwrap_or_else(|| ref_text.clone()),
                    "entity_id": entity_id,
                    "claim_id": claim_id.as_deref(),
                }),
                None,
                None,
                conversation_id,
                None,
                source_type_str(&source),
                Some(source_ref.as_str()),
                None,
                None,
            )
            .await;
            
            ClarifyResult {
                success: true,
                selected_entity_id: Some(entity_id),
                selected_label,
                compile_result,
                error: None,
            }
        }
        
        ClarifyChoice::CreateNew(label) => {
            let claims_enabled = compiler::memory_claims_enabled(pool).await;
            let claim_id = claim_id
                .as_deref()
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(|id| id.to_string());

            // Use provided label or original ref_text
            let new_label = if label.is_empty() { ref_text.clone() } else { label };
            let label_canonical = canonicalize_label(&new_label);
            
            // Create the new entity
            let entity_result = sqlx::query(
                "INSERT INTO ics_entities (label, label_canonical, resolution_state, created_at, last_accessed_at, access_count)
                 VALUES (?, ?, 'tentative', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0)
                 RETURNING id"
            )
            .bind(&new_label)
            .bind(label_canonical)
            .fetch_one(pool)
            .await;
            
            match entity_result {
                Ok(row) => {
                    let new_entity_id: i64 = row.get("id");
                    
                    // Create session binding
                    let _ = sqlx::query(
                        "INSERT INTO ics_session_bindings (session_id, ref_text, entity_id, created_at)
                         VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                         ON CONFLICT(session_id, ref_text) DO UPDATE SET entity_id = excluded.entity_id"
                    )
                    .bind(&session_id)
                    .bind(&ref_text)
                    .bind(new_entity_id)
                    .execute(pool)
                    .await;
                    
                    let mut compile_result = None;
                    if let Some(claim_id) = claim_id.as_deref() {
                        if claims_enabled {
                            let _ = sqlx::query(
                                "UPDATE memory_claims
                                 SET status = 'pending',
                                     decision_reason = NULL,
                                     conflict_topic_key = NULL,
                                     conflict_reason = NULL,
                                     updated_at = CURRENT_TIMESTAMP
                                 WHERE id = ?"
                            )
                            .bind(claim_id)
                            .execute(pool)
                            .await;
                            if let Err(e) = claims::evaluate_claim(pool, model_client.clone(), claim_id).await {
                                return ClarifyResult {
                                    success: false,
                                    selected_entity_id: Some(new_entity_id),
                                    selected_label: Some(new_label),
                                    compile_result: None,
                                    error: Some(format!("Claim evaluation failed: {}", e)),
                                };
                            }
                        } else {
                            let ctx = CompileContext {
                                pool: pool.clone(),
                                model_client,
                                session_id: session_id.clone(),
                                scope,
                                source,
                                source_ref: Some(format!("clarify:{}", pending_id)),
                                now: chrono::Utc::now(),
                                embedding_config: embedding_config.clone(),
                                skip_claims: false,
                                allow_ambiguous_user_refs: false,
                            };

                            compile_result = Some(compiler::compile(&original_dsl, ctx).await);
                        }
                    } else {
                        let ctx = CompileContext {
                            pool: pool.clone(),
                            model_client,
                            session_id: session_id.clone(),
                            scope,
                            source,
                            source_ref: Some(format!("clarify:{}", pending_id)),
                            now: chrono::Utc::now(),
                            embedding_config: embedding_config.clone(),
                            skip_claims: false,
                            allow_ambiguous_user_refs: false,
                        };

                        compile_result = Some(compiler::compile(&original_dsl, ctx).await);
                    }
                    
                    // Mark as resolved
                    let _ = sqlx::query("UPDATE ics_pending_clarify SET status = 'resolved' WHERE id = ?")
                        .bind(pending_id)
                        .execute(pool)
                        .await;

                    let source_ref = format!("clarify:{}", pending_id);
                    let _ = episodic::emit_episodic_event(
                        pool,
                        "clarify_resolved",
                        serde_json::json!({
                            "status": "resolved",
                            "summary_snippet": new_label.clone(),
                            "entity_id": new_entity_id,
                            "claim_id": claim_id.as_deref(),
                        }),
                        None,
                        None,
                        conversation_id,
                        None,
                        source_type_str(&source),
                        Some(source_ref.as_str()),
                        None,
                        None,
                    )
                    .await;
                    
                    ClarifyResult {
                        success: true,
                        selected_entity_id: Some(new_entity_id),
                        selected_label: Some(new_label),
                        compile_result,
                        error: None,
                    }
                }
                Err(e) => ClarifyResult {
                    success: false,
                    selected_entity_id: None,
                    selected_label: None,
                    compile_result: None,
                    error: Some(format!("Failed to create entity: {}", e)),
                }
            }
        }
    }
}

fn source_type_str(source: &SourceType) -> &'static str {
    match source {
        SourceType::User => "user",
        SourceType::Tool => "tool",
        SourceType::System => "system",
        SourceType::Inference => "inference",
    }
}

fn parse_json_vec(raw: Option<&str>) -> Vec<String> {
    raw.and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
        .unwrap_or_default()
}

async fn maybe_add_entity_alias(pool: &SqlitePool, entity_id: i64, alias: &str) -> Result<(), String> {
    let alias = alias.trim();
    if alias.is_empty() {
        return Ok(());
    }
    let alias_lower = alias.to_lowercase();
    let banned = [
        "he", "she", "they", "them", "his", "her", "their", "it", "this", "that",
        "these", "those", "you", "me", "my", "mine", "our", "ours",
    ];
    if alias.len() <= 2 || banned.contains(&alias_lower.as_str()) {
        return Ok(());
    }

    let row = sqlx::query(
        "SELECT label, aliases, aliases_canonical FROM ics_entities WHERE id = ?"
    )
    .bind(entity_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    let Some(row) = row else {
        return Ok(());
    };

    let label: String = row.get("label");
    let label_canon = canonicalize_label(&label);
    let alias_canon = canonicalize_label(alias);

    if alias_canon == label_canon {
        return Ok(());
    }

    let mut aliases = parse_json_vec(row.try_get::<Option<String>, _>("aliases").ok().flatten().as_deref());
    let mut aliases_canon = parse_json_vec(row.try_get::<Option<String>, _>("aliases_canonical").ok().flatten().as_deref());
    if aliases.is_empty() {
        aliases_canon.clear();
    } else if aliases_canon.len() != aliases.len() {
        aliases_canon = aliases.iter().map(|a| canonicalize_label(a)).collect();
    }

    let mut seen: HashSet<String> = aliases_canon.iter().cloned().collect();
    if !seen.insert(alias_canon.clone()) {
        return Ok(());
    }

    aliases.push(alias.to_string());
    aliases_canon.push(alias_canon);

    let aliases_json = serde_json::to_string(&aliases).unwrap_or_else(|_| "[]".to_string());
    let aliases_canon_json = serde_json::to_string(&aliases_canon).unwrap_or_else(|_| "[]".to_string());

    sqlx::query(
        "UPDATE ics_entities
         SET aliases = ?, aliases_canonical = ?
         WHERE id = ?"
    )
    .bind(aliases_json)
    .bind(aliases_canon_json)
    .bind(entity_id)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    Ok(())
}

/// Format pending clarification for LLM prompt injection
pub fn format_clarification_prompt(pending: &crate::core::memory::compiler::PendingClarify) -> String {
    let mut lines = vec![
        format!("\n[DISAMBIGUATION NEEDED]"),
        format!("The reference \"{}\" is ambiguous. Please clarify which one you mean:", pending.ref_text),
    ];
    
    for (i, candidate) in pending.candidates.iter().enumerate() {
        let context = if candidate.context.is_empty() {
            String::new()
        } else {
            format!(" ({})", candidate.context)
        };
        lines.push(format!("  {}. {}{}", i + 1, candidate.label, context));
    }
    
    lines.push(format!("  Or say \"create new\" to create a new entity."));
    lines.push(format!(""));
    
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_option_number() {
        let candidates = vec![
            ClarifyCandidate { entity_id: 1, label: "Alice".into(), context: "".into() },
            ClarifyCandidate { entity_id: 2, label: "Bob".into(), context: "".into() },
        ];
        
        assert!(matches!(parse_user_reply("1", &candidates), ClarifyChoice::SelectById(1)));
        assert!(matches!(parse_user_reply("option 2", &candidates), ClarifyChoice::SelectById(2)));
        assert!(matches!(parse_user_reply("#1", &candidates), ClarifyChoice::SelectById(1)));
    }
    
    #[test]
    fn test_parse_label_match() {
        let candidates = vec![
            ClarifyCandidate { entity_id: 1, label: "Alice".into(), context: "".into() },
            ClarifyCandidate { entity_id: 2, label: "Bob".into(), context: "".into() },
        ];
        
        assert!(matches!(parse_user_reply("alice", &candidates), ClarifyChoice::SelectById(1)));
        assert!(matches!(parse_user_reply("Bob", &candidates), ClarifyChoice::SelectById(2)));
    }
    
    #[test]
    fn test_parse_create_new() {
        let candidates = vec![];
        
        assert!(matches!(parse_user_reply("none", &candidates), ClarifyChoice::CreateNew(_)));
        assert!(matches!(parse_user_reply("create new", &candidates), ClarifyChoice::CreateNew(_)));
        assert!(matches!(parse_user_reply("new", &candidates), ClarifyChoice::CreateNew(_)));
    }
    
    #[test]
    fn test_parse_cancel() {
        let candidates = vec![];
        
        assert!(matches!(parse_user_reply("cancel", &candidates), ClarifyChoice::Cancel));
        assert!(matches!(parse_user_reply("skip", &candidates), ClarifyChoice::Cancel));
    }
}
