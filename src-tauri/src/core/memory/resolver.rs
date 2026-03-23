use crate::core::memory::types::{PendingWrite, EntityType};
use crate::core::memory::dsl::Ref;
use crate::core::memory::config::{RESOLVE_THRESHOLD, MARGIN_THRESHOLD};
use crate::core::memory::canonical::canonicalize_label;
use sqlx::SqlitePool;
use sqlx::sqlite::SqliteRow;
use sqlx::Row;
use std::collections::HashSet;

// Context for resolution (session, existing bindings)
pub struct ResolveContext {
    pub pool: SqlitePool,
    pub session_id: String,
    pub expected_type: Option<EntityType>,  // Type hint for scoring
    pub anchor_entity_ids: Vec<i64>,        // From working set for neighborhood overlap
}

impl ResolveContext {
    /// Create a basic context without type hint or anchors (backward compatible)
    pub fn basic(pool: SqlitePool, session_id: String) -> Self {
        Self {
            pool,
            session_id,
            expected_type: None,
            anchor_entity_ids: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub enum ResolutionResult {
    Resolved(i64), // EntityId
    Ambiguous(PendingWrite), // Should contain candidates
    NewEntity(String), // Create new with label
}

/// Candidate with multi-factor scoring breakdown
struct ScoredCandidate {
    entity_id: i64,
    label: String,
    entity_type: Option<String>,
    label_score: f32,      // weak: 0.2 weight
    type_score: f32,       // medium: 0.2 weight  
    recency_score: f32,    // weak: 0.1 weight
    neighborhood_score: f32, // strong: 0.5 weight
}

impl ScoredCandidate {
    fn total_score(&self) -> f32 {
        // Weights per ICS v4.1 §6.1
        const LABEL_WEIGHT: f32 = 0.2;
        const TYPE_WEIGHT: f32 = 0.2;
        const RECENCY_WEIGHT: f32 = 0.1;
        const NEIGHBORHOOD_WEIGHT: f32 = 0.5;
        
        self.label_score * LABEL_WEIGHT
            + self.type_score * TYPE_WEIGHT
            + self.recency_score * RECENCY_WEIGHT
            + self.neighborhood_score * NEIGHBORHOOD_WEIGHT
    }
}

fn is_alias_candidate(alias: &str) -> bool {
    let alias = alias.trim();
    if alias.is_empty() || alias.len() <= 2 {
        return false;
    }

    if alias.chars().all(|c| !c.is_alphanumeric()) {
        return false;
    }

    let alias_lower = alias.to_lowercase();
    let banned = [
        "he", "she", "they", "them", "his", "her", "their", "it", "this", "that",
        "these", "those", "you", "me", "my", "mine", "our", "ours", "i", "we",
    ];
    if banned.contains(&alias_lower.as_str()) {
        return false;
    }

    if alias_lower.starts_with('$') || alias_lower.starts_with('#') {
        return false;
    }

    true
}

async fn record_entity_alias_evidence(
    pool: &SqlitePool,
    entity_id: i64,
    alias: &str,
    label: &str,
) -> Result<(), String> {
    let alias = alias.trim();
    if !is_alias_candidate(alias) {
        return Ok(());
    }

    let alias_canon = canonicalize_label(alias);
    if alias_canon.is_empty() {
        return Ok(());
    }
    let label_canon = canonicalize_label(label);
    if alias_canon == label_canon {
        return Ok(());
    }

    let existing_aliases: Option<String> = sqlx::query("SELECT aliases_canonical FROM ics_entities WHERE id = ?")
        .bind(entity_id)
        .fetch_optional(pool)
        .await
        .map_err(map_err)?
        .and_then(|row| row.try_get::<String, _>("aliases_canonical").ok());
    if let Some(raw) = existing_aliases {
        let aliases: Vec<String> = serde_json::from_str::<Vec<String>>(&raw).unwrap_or_default();
        if aliases.iter().any(|c| c == &alias_canon) {
            return Ok(());
        }
    }

    let _ = sqlx::query(
        "INSERT INTO ics_entity_aliases (entity_id, alias, alias_canonical, status, evidence_count, created_at, updated_at)
         VALUES (?, ?, ?, 'proposed', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
         ON CONFLICT(entity_id, alias_canonical)
         DO UPDATE SET evidence_count = evidence_count + 1, updated_at = CURRENT_TIMESTAMP"
    )
    .bind(entity_id)
    .bind(alias)
    .bind(&alias_canon)
    .execute(pool)
    .await;

    Ok(())
}

/// Main resolution entry point (Spec §6.1)
pub async fn resolve_ref(r: &Ref, ctx: &ResolveContext) -> Result<ResolutionResult, String> {
    match r {
        Ref::Handle(h) => resolve_handle(h, ctx).await,
        Ref::Label(l) => resolve_label(l, ctx).await,
        Ref::Name(n) => resolve_name(n, ctx).await,
        Ref::Filter(l, _key) => resolve_label(l, ctx).await,
    }
}

// Result alias for easier error handling
type Res = Result<ResolutionResult, String>;
use std::fmt::Display;
fn map_err<E: Display>(e: E) -> String { e.to_string() }

async fn resolve_handle(handle: &str, ctx: &ResolveContext) -> Res {
    // 1. Check session bindings (from disambiguation)
    let session_id = ctx.session_id.trim();
    let session_id = if session_id.is_empty() { "default" } else { session_id };
    let binding = sqlx::query("SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = ?")
        .bind(session_id)
        .bind(handle)
        .fetch_optional(&ctx.pool)
        .await
        .map_err(map_err)?;
        
    if let Some(row) = binding {
        let id: i64 = row.get("entity_id");
        return Ok(ResolutionResult::Resolved(id));
    }

    // Fallback: for core handles, reuse the default binding to avoid fragmenting $user/$assistant.
    if handle == "user" || handle == "assistant" {
        let default_binding = sqlx::query(
            "SELECT entity_id FROM ics_session_bindings WHERE session_id = 'default' AND ref_text = ?"
        )
        .bind(handle)
        .fetch_optional(&ctx.pool)
        .await
        .map_err(map_err)?;

        if let Some(row) = default_binding {
            let id: i64 = row.get("entity_id");
            if session_id != "default" {
                let _ = sqlx::query(
                    "INSERT INTO ics_session_bindings (session_id, ref_text, entity_id, created_at)
                     VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                     ON CONFLICT(session_id, ref_text) DO UPDATE SET entity_id = excluded.entity_id"
                )
                .bind(session_id)
                .bind(handle)
                .bind(id)
                .execute(&ctx.pool)
                .await;
            }
            return Ok(ResolutionResult::Resolved(id));
        }
    }
    
    Ok(ResolutionResult::NewEntity(handle.to_string()))
}

async fn resolve_label(label: &str, ctx: &ResolveContext) -> Res {
    // Exact match on label or key (per §6.1 priority 2)
    let canonical_label = canonicalize_label(label);
    if canonical_label == "user" {
        if let Ok(ResolutionResult::Resolved(id)) = resolve_handle("user", ctx).await {
            return Ok(ResolutionResult::Resolved(id));
        }
    } else if canonical_label == "assistant" {
        if let Ok(ResolutionResult::Resolved(id)) = resolve_handle("assistant", ctx).await {
            return Ok(ResolutionResult::Resolved(id));
        }
    }
    let exact = sqlx::query("SELECT id FROM ics_entities WHERE label = ? OR label_canonical = ? OR keys LIKE ? OR aliases LIKE ? OR aliases_canonical LIKE ?")
        .bind(label)
        .bind(&canonical_label)
        .bind(format!("%\"{}\"%" , label))
        .bind(format!("%\"{}\"%" , label))
        .bind(format!("%\"{}\"%" , canonical_label))
        .fetch_optional(&ctx.pool)
        .await
        .map_err(map_err)?;
        
    if let Some(row) = exact {
        let id: i64 = row.get("id");
        return Ok(ResolutionResult::Resolved(id));
    }
    
    resolve_fuzzy(label, ctx).await
}

async fn resolve_name(name: &str, ctx: &ResolveContext) -> Res {
    // 1. Check session bindings first (from disambiguation)
    let binding = sqlx::query("SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = ?")
        .bind(&ctx.session_id)
        .bind(name)
        .fetch_optional(&ctx.pool)
        .await
        .map_err(map_err)?;
        
    if let Some(row) = binding {
        let id: i64 = row.get("entity_id");
        return Ok(ResolutionResult::Resolved(id));
    }
    
    // 2. Fall back to label resolution
    resolve_label(name, ctx).await
}

/// Compute neighborhood overlap score per ICS v4.1 §6.1
/// Uses hub-resistant weighting: w(n) = 1/log2(degree(n)+2)
async fn compute_neighborhood_score(
    candidate_id: i64,
    anchor_ids: &[i64],
    pool: &SqlitePool,
) -> Result<f32, String> {
    if anchor_ids.is_empty() {
        return Ok(0.0);
    }
    
    // Get candidate's neighbors from sketch cache
    let sketch_row = sqlx::query("SELECT neighbors_top FROM ics_entity_sketches WHERE entity_id = ?")
        .bind(candidate_id)
        .fetch_optional(pool)
        .await
        .map_err(map_err)?;
    
    let candidate_neighbors: HashSet<i64> = match sketch_row {
        Some(row) => {
            let neighbors_json: String = row.try_get("neighbors_top").unwrap_or_default();
            serde_json::from_str::<Vec<i64>>(&neighbors_json)
                .unwrap_or_default()
                .into_iter()
                .collect()
        }
        None => HashSet::new(),
    };
    
    if candidate_neighbors.is_empty() {
        return Ok(0.0);
    }
    
    // Compute intersection with anchors
    let anchor_set: HashSet<i64> = anchor_ids.iter().cloned().collect();
    let intersection: Vec<i64> = candidate_neighbors
        .intersection(&anchor_set)
        .cloned()
        .collect();
    
    if intersection.is_empty() {
        return Ok(0.0);
    }
    
    // Apply hub-resistant weighting
    let mut weighted_overlap = 0.0f32;
    
    for neighbor_id in intersection {
        // Get degree from cache
        let degree: i64 = sqlx::query("SELECT degree FROM ics_entity_degrees WHERE entity_id = ?")
            .bind(neighbor_id)
            .fetch_optional(pool)
            .await
            .map_err(map_err)?
            .map(|r| r.get::<i64, _>("degree"))
            .unwrap_or(1);
        
        // Hub penalty: w(n) = 1/log2(degree+2)
        let weight = 1.0 / ((degree as f32 + 2.0).log2());
        weighted_overlap += weight;
    }
    
    // Normalize by min(|A|,|B|) for stability
    let normalizer = (candidate_neighbors.len().min(anchor_ids.len())).max(1) as f32;
    let normalized = weighted_overlap / normalizer;
    
    // Clamp to [0, 1]
    Ok(normalized.min(1.0))
}

/// Compute label similarity score
fn compute_label_score(candidate_label: &str, query_text: &str) -> f32 {
    let candidate_canon = canonicalize_label(candidate_label);
    let query_canon = canonicalize_label(query_text);

    if candidate_canon == query_canon {
        1.0 // Exact match (case-insensitive)
    } else if candidate_canon.contains(&query_canon) {
        0.7 // Substring match
    } else if query_canon.contains(&candidate_canon) {
        0.6 // Query contains label
    } else {
        0.3 // FTS matched but no direct similarity
    }
}

/// Compute type fit score
fn compute_type_score(candidate_type: Option<&str>, expected_type: Option<&EntityType>) -> f32 {
    match (candidate_type, expected_type) {
        (Some(ct), Some(et)) => {
            let et_str = format!("{:?}", et).to_lowercase();
            if ct.to_lowercase() == et_str {
                1.0 // Exact type match
            } else {
                0.3 // Type exists but doesn't match
            }
        }
        (Some(_), None) => 0.5, // Candidate has type, no expectation
        (None, Some(_)) => 0.2, // Expected type, candidate untyped
        (None, None) => 0.5,    // No type info either way
    }
}

/// Compute recency score based on last_accessed_at
fn compute_recency_score(last_accessed: Option<&str>) -> f32 {
    let Some(accessed_str) = last_accessed else {
        return 0.3; // No access info
    };
    
    let Ok(accessed) = chrono::DateTime::parse_from_rfc3339(accessed_str) else {
        return 0.3;
    };
    
    let now = chrono::Utc::now();
    let hours_ago = (now - accessed.with_timezone(&chrono::Utc)).num_hours();
    
    if hours_ago < 1 {
        1.0 // Very recent
    } else if hours_ago < 24 {
        0.8 // Within a day
    } else if hours_ago < 168 {
        0.5 // Within a week
    } else {
        0.2 // Older
    }
}

/// Full fuzzy resolution with multi-factor scoring per ICS v4.1 §6.1
async fn resolve_fuzzy(query_text: &str, ctx: &ResolveContext) -> Res {
    let clean_query = query_text.replace("\"", "");
    let fts_query = format!("\"{}\"*", clean_query);
    
    // Fetch candidates with additional data for scoring
    let rows: Vec<SqliteRow> = sqlx::query(
        "SELECT e.id, e.label, e.entity_type, e.last_accessed_at 
         FROM ics_entities_fts f 
         JOIN ics_entities e ON e.id = f.rowid 
         WHERE ics_entities_fts MATCH ? 
         ORDER BY rank 
         LIMIT 20"
    )
        .bind(&fts_query)
        .fetch_all(&ctx.pool)
        .await
        .map_err(map_err)?;
    
    if rows.is_empty() {
        return Ok(ResolutionResult::NewEntity(query_text.to_string()));
    }
    
    // Build scored candidates with multi-factor scores
    let mut candidates: Vec<ScoredCandidate> = Vec::new();
    
    for row in &rows {
        let entity_id: i64 = row.get("id");
        let label: String = row.get("label");
        let entity_type: Option<String> = row.try_get("entity_type").ok();
        let last_accessed: Option<String> = row.try_get("last_accessed_at").ok();
        
        // Compute individual factor scores
        let label_score = compute_label_score(&label, query_text);
        let type_score = compute_type_score(entity_type.as_deref(), ctx.expected_type.as_ref());
        let recency_score = compute_recency_score(last_accessed.as_deref());
        
        // Compute neighborhood overlap (async)
        let neighborhood_score = compute_neighborhood_score(
            entity_id,
            &ctx.anchor_entity_ids,
            &ctx.pool,
        ).await.unwrap_or(0.0);
        
        candidates.push(ScoredCandidate {
            entity_id,
            label,
            entity_type,
            label_score,
            type_score,
            recency_score,
            neighborhood_score,
        });
    }
    
    // Sort by total score descending
    candidates.sort_by(|a, b| {
        b.total_score()
            .partial_cmp(&a.total_score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    
    if let Some(top) = candidates.first() {
        let top_score = top.total_score();
        
        if top_score >= RESOLVE_THRESHOLD {
            if candidates.len() > 1 {
                let second_score = candidates[1].total_score();
                
                if top_score - second_score < MARGIN_THRESHOLD {
                    // Ambiguous - include score breakdown in candidates_json
                    let candidates_json: Vec<serde_json::Value> = candidates.iter()
                        .take(5)
                        .map(|c| {
                            serde_json::json!({
                                "entity_id": c.entity_id,
                                "label": c.label,
                                "entity_type": c.entity_type,
                                "score": c.total_score(),
                                "breakdown": {
                                    "label": c.label_score,
                                    "type": c.type_score,
                                    "recency": c.recency_score,
                                    "neighborhood": c.neighborhood_score
                                }
                            })
                        })
                        .collect();
                    
                    return Ok(ResolutionResult::Ambiguous(PendingWrite {
                        id: 0,
                        parsed_lines: query_text.to_string(),
                        candidates_json: serde_json::to_string(&candidates_json).unwrap_or_default(),
                        status: "pending".to_string(),
                        created_at: chrono::Utc::now().to_rfc3339(),
                    }));
                }
            }
            let _ = record_entity_alias_evidence(
                &ctx.pool,
                top.entity_id,
                query_text,
                &top.label,
            )
            .await;
            return Ok(ResolutionResult::Resolved(top.entity_id));
        }
    }
    
    Ok(ResolutionResult::NewEntity(query_text.to_string()))
}

/// Get anchor entity IDs from the working set (top activated entities)
pub async fn get_working_set_anchors(pool: &SqlitePool, limit: usize) -> Result<Vec<i64>, String> {
    let rows = sqlx::query(
        "SELECT item_id FROM ics_working_set 
         WHERE item_type = 'entity' 
         ORDER BY activation DESC 
         LIMIT ?"
    )
        .bind(limit as i64)
        .fetch_all(pool)
        .await
        .map_err(map_err)?;
    
    Ok(rows.iter().map(|r| r.get("item_id")).collect())
}

