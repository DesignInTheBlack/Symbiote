use sqlx::SqlitePool;
use sqlx::Row;
use crate::core::memory::config::FTS_CANDIDATE_LIMIT;
use std::collections::{HashSet, HashMap};

pub struct Anchor {
    pub entity_id: i64,
    pub score: f32,
    pub reason: String,
}

use std::sync::Arc;
use crate::core::model_client::ModelClient;
use crate::core::memory::writer::EmbeddingConfig;
use crate::core::memory::embedding_index;

const EMBEDDING_BRUTE_FORCE_LIMIT: i64 = 2000;
const SEMANTIC_SCORE_SCALE: f32 = 10.0;

/// Find anchors based on query text using FTS and Vector Search (Spec §9.1 + §10.3)
pub async fn find_anchors(query: &str, pool: &SqlitePool, model_client: Option<Arc<ModelClient>>, embedding_config: Option<&EmbeddingConfig>) -> Result<Vec<Anchor>, String> {
    // 1. Tokenize and Filter (FTS Prep)
    // Standard English stop words + common query verbs
    // Standard English stop words + common query verbs
    let stop_words: HashSet<&str> = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", 
        "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", 
        "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", 
        "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "is", "can", 
        "cant", "are", "am", "was", "were"
    ].iter().cloned().collect();

    let mut tokens: Vec<String> = query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .filter(|s| !stop_words.contains(*s))
        .map(|s| s.to_string())
        .collect();

    if tokens.is_empty() {
        return Ok(vec![]);
    }
    
    // Expand tokens with aliases per ICS v4.1 §2.5 guardrails
    use crate::core::memory::config::MIN_PROPOSED_EXPAND;
    const MAX_ALIAS_EXPANSIONS: usize = 6;
    
    let original_len = tokens.len();
    let mut total_expansions = 0;
    
    for token in tokens.clone().iter().take(original_len) {
        if total_expansions >= MAX_ALIAS_EXPANSIONS {
            break;
        }
        
        // 1. Confirmed aliases first (highest priority)
        let confirmed_aliases = sqlx::query(
            "SELECT to_token FROM ics_token_aliases 
             WHERE from_token = ? AND status = 'confirmed' 
             ORDER BY evidence_count DESC 
             LIMIT 3"
        )
        .bind(token)
        .fetch_all(pool)
        .await
        .unwrap_or_default();
        
        for row in confirmed_aliases {
            if total_expansions >= MAX_ALIAS_EXPANSIONS { break; }
            let alias: String = row.get("to_token");
            if !tokens.contains(&alias) {
                tokens.push(alias);
                total_expansions += 1;
            }
        }
        
        // 2. Proposed aliases only if they meet guardrails
        let proposed_aliases = sqlx::query(
            "SELECT ta.to_token, ta.evidence_count,
                    (SELECT COUNT(*) FROM ics_fact_beliefs fb 
                     WHERE fb.key = ta.to_token OR fb.value_literal LIKE '%' || ta.to_token || '%') as co_usage
             FROM ics_token_aliases ta
             WHERE ta.from_token = ? AND ta.status = 'proposed' AND ta.evidence_count >= ?
             ORDER BY ta.evidence_count DESC
             LIMIT 3"
        )
        .bind(token)
        .bind(MIN_PROPOSED_EXPAND as i64)
        .fetch_all(pool)
        .await
        .unwrap_or_default();
        
        for row in proposed_aliases {
            if total_expansions >= MAX_ALIAS_EXPANSIONS { break; }
            
            let co_usage: i64 = row.try_get("co_usage").unwrap_or(0);
            if co_usage == 0 {
                continue; // Guardrail: must have co-usage signal
            }
            
            let alias: String = row.get("to_token");
            if !tokens.contains(&alias) {
                tokens.push(alias);
                total_expansions += 1;
            }
        }
    }

    // 2. Construct FTS Query (OR logic)
    // Use prefix matching for each token to handle partials/plurals roughly
    let fts_parts: Vec<String> = tokens.iter()
        .map(|t| format!("\"{}\"*", t)) 
        .collect();
    
    let fts_query = fts_parts.join(" OR ");
    
    // 3. FTS Search - Entities
    let mut anchors_map: HashMap<i64, Anchor> = HashMap::new();

    let entity_rows = sqlx::query("SELECT rowid, rank FROM ics_entities_fts WHERE ics_entities_fts MATCH ? ORDER BY rank LIMIT ?")
        .bind(&fts_query)
        .bind(FTS_CANDIDATE_LIMIT as i64)
        .fetch_all(pool)
        .await
        .map_err(|e| e.to_string())?;

    for row in entity_rows {
        let id: i64 = row.get("rowid");
        let rank: f64 = row.get("rank");
        anchors_map.insert(id, Anchor {
            entity_id: id,
            score: rank as f32, // Lower is better in FTS5
            reason: format!("entity_match:{}", fts_query),
        });
    }

    // 4. FTS Search - Facts (Attributes)
    // Find facts matching the query, then link back to their subject entity.
    let fact_rows = sqlx::query(
        r#"
        SELECT fb.subject_entity_id, fts.rank 
        FROM ics_facts_fts fts 
        JOIN ics_fact_beliefs fb ON fb.belief_id = fts.rowid 
        WHERE ics_facts_fts MATCH ? 
        ORDER BY fts.rank 
        LIMIT ?
        "#
    )
    .bind(&fts_query)
    .bind(FTS_CANDIDATE_LIMIT as i64)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    for row in fact_rows {
        let id: i64 = row.get("subject_entity_id");
        let rank: f64 = row.get("rank");
        
        // If already found via name, keep the better score (lower rank)
        if let Some(existing) = anchors_map.get(&id) {
             if (rank as f32) < existing.score {
                 anchors_map.insert(id, Anchor {
                    entity_id: id,
                    score: rank as f32,
                    reason: format!("fact_match:{}", fts_query),
                 });
             }
        } else {
             anchors_map.insert(id, Anchor {
                entity_id: id,
                score: rank as f32,
                reason: format!("fact_match:{}", fts_query),
             });
        }
    }

    // 5. FTS Search - Relations (types + roles)
    if let Ok(rel_rows) = sqlx::query(
        "SELECT f.rowid, f.rank
         FROM ics_rel_fts f
         JOIN ics_beliefs b ON b.id = f.rowid
         WHERE ics_rel_fts MATCH ? AND b.status = 'active'
         ORDER BY f.rank
         LIMIT ?"
    )
        .bind(&fts_query)
        .bind(FTS_CANDIDATE_LIMIT as i64)
        .fetch_all(pool)
        .await
    {
        let mut rel_scores: HashMap<i64, f32> = HashMap::new();
        for row in &rel_rows {
            let belief_id: i64 = row.get("rowid");
            let rank: f64 = row.get("rank");
            rel_scores.insert(belief_id, rank as f32);
        }

        if !rel_scores.is_empty() {
            let placeholders: String = rel_scores.keys().map(|_| "?").collect::<Vec<_>>().join(",");
            let query = format!(
                "SELECT rp.belief_id, rp.entity_id, rp.role, rb.rel_type
                 FROM ics_rel_participants rp
                 JOIN ics_rel_beliefs rb ON rb.belief_id = rp.belief_id
                 WHERE rp.belief_id IN ({})",
                placeholders
            );
            let mut q = sqlx::query(&query);
            for belief_id in rel_scores.keys() {
                q = q.bind(belief_id);
            }

            let rel_part_rows = q.fetch_all(pool).await.unwrap_or_default();
            for row in rel_part_rows {
                let belief_id: i64 = row.get("belief_id");
                let entity_id: i64 = row.get("entity_id");
                let role: String = row.get("role");
                let rel_type: String = row.get("rel_type");
                let rank = *rel_scores.get(&belief_id).unwrap_or(&0.0);
                let reason = format!("rel_match:{}:{}", rel_type, role);

                if let Some(existing) = anchors_map.get(&entity_id) {
                    if rank < existing.score {
                        anchors_map.insert(entity_id, Anchor {
                            entity_id,
                            score: rank,
                            reason,
                        });
                    }
                } else {
                    anchors_map.insert(entity_id, Anchor {
                        entity_id,
                        score: rank,
                        reason,
                    });
                }
            }
        }
    }
    
    // 6. Vector Search - Semantic Fallback (§10.3)
    // If ModelClient is available and embeddings enabled, use embeddings to find related concepts
    if let (Some(client), Some(config)) = (model_client, embedding_config) {
        if config.enabled {
            if let Ok(query_vec) = client.embed(&config.base_url, None, &config.model, query).await {
                let signature = embedding_index::embedding_signature(&query_vec);
                let buckets = embedding_index::candidate_buckets(signature);
                let rows = if buckets.is_empty() {
                    Vec::new()
                } else {
                    let placeholders: String = buckets.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                    let query = format!(
                        "SELECT e.assertion_id, e.embedding
                         FROM ics_embeddings e
                         JOIN ics_embedding_lsh l ON l.assertion_id = e.assertion_id
                         WHERE l.bucket IN ({})
                         LIMIT 800",
                        placeholders
                    );
                    let mut q = sqlx::query(&query);
                    for bucket in &buckets {
                        q = q.bind(bucket);
                    }
                    match q.fetch_all(pool).await {
                        Ok(rows) => rows,
                        Err(_) => {
                            let total: Option<i64> = sqlx::query_scalar("SELECT COUNT(*) FROM ics_embeddings")
                                .fetch_optional(pool)
                                .await
                                .ok()
                                .flatten();
                            if total.unwrap_or(0) <= EMBEDDING_BRUTE_FORCE_LIMIT {
                                sqlx::query("SELECT assertion_id, embedding FROM ics_embeddings LIMIT ?")
                                    .bind(EMBEDDING_BRUTE_FORCE_LIMIT)
                                    .fetch_all(pool)
                                    .await
                                    .unwrap_or_default()
                            } else {
                                Vec::new()
                            }
                        }
                    }
                };
                
                let mut semantic_matches: Vec<(i64, f32)> = vec![];
                
                for row in rows {
                    let belief_id: i64 = row.get("assertion_id");
                    let blob: Vec<u8> = row.get("embedding");
                    
                    // Convert blob back to f32 vec
                    // Assumes f32 (4 bytes) little endian
                    if blob.len() % 4 == 0 {
                        let vec: Vec<f32> = blob
                            .chunks(4)
                            .filter_map(|chunk| {
                                let arr: [u8; 4] = chunk.try_into().ok()?;
                                Some(f32::from_le_bytes(arr))
                            })
                            .collect();
                            
                        if vec.len() == query_vec.len() {
                            let sim = cosine_similarity(&query_vec, &vec);
                            if sim > 0.65 { // Threshold for "relevant"
                                semantic_matches.push((belief_id, sim));
                            }
                        }
                    }
                }
                
                // Sort by similarity desc
                semantic_matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                // Link back to subjects (Embeddings are on facts/beliefs)
                // We need to find the subject_entity_id for these beliefs.
                for (belief_id, sim) in semantic_matches.into_iter().take(5) {
                    // Get subject ID from belief
                    let subject_row = sqlx::query("SELECT subject_entity_id FROM ics_fact_beliefs WHERE belief_id = ?")
                        .bind(belief_id)
                        .fetch_optional(pool)
                        .await
                        .unwrap_or(None);
                        
                    if let Some(row) = subject_row {
                        let subject_id: i64 = row.get("subject_entity_id");
                        // Map cosine similarity to the same "lower is better" scale as FTS.
                        let semantic_score = (1.0 - sim).max(0.0) * SEMANTIC_SCORE_SCALE;

                        anchors_map.entry(subject_id)
                            .and_modify(|a| {
                                if semantic_score < a.score {
                                    a.score = semantic_score;
                                }
                                a.reason.push_str(&format!("; semantic({:.2})", sim));
                            })
                            .or_insert(Anchor {
                                entity_id: subject_id,
                                score: semantic_score,
                                reason: format!("semantic_match:{:.2}", sim),
                            });
                    }
                }
            }
        }
    }

    Ok(anchors_map.into_values().collect())
}

pub async fn find_anchors_lexical_fallback(
    query: &str,
    pool: &SqlitePool,
    entity_limit: i64,
    fact_limit: i64,
    rel_limit: i64,
) -> Result<Vec<Anchor>, String> {
    let mut tokens: Vec<String> = query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    if tokens.is_empty() {
        return Ok(vec![]);
    }

    if tokens.len() > 12 {
        tokens.truncate(12);
    }

    let fts_parts: Vec<String> = tokens.iter()
        .map(|t| format!("\"{}\"*", t))
        .collect();
    let fts_query = fts_parts.join(" OR ");

    let mut anchors_map: HashMap<i64, Anchor> = HashMap::new();

    let entity_rows = sqlx::query(
        "SELECT rowid, rank FROM ics_entities_fts WHERE ics_entities_fts MATCH ? ORDER BY rank LIMIT ?"
    )
    .bind(&fts_query)
    .bind(entity_limit.max(1))
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    for row in entity_rows {
        let id: i64 = row.get("rowid");
        let rank: f64 = row.get("rank");
        anchors_map.insert(id, Anchor {
            entity_id: id,
            score: rank as f32,
            reason: "fallback:lexical:entity_match".to_string(),
        });
    }

    let fact_rows = sqlx::query(
        r#"
        SELECT fb.subject_entity_id, fts.rank
        FROM ics_facts_fts fts
        JOIN ics_fact_beliefs fb ON fb.belief_id = fts.rowid
        WHERE ics_facts_fts MATCH ?
        ORDER BY fts.rank
        LIMIT ?
        "#
    )
    .bind(&fts_query)
    .bind(fact_limit.max(1))
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    for row in fact_rows {
        let id: i64 = row.get("subject_entity_id");
        let rank: f64 = row.get("rank");
        if let Some(existing) = anchors_map.get(&id) {
            if (rank as f32) < existing.score {
                anchors_map.insert(id, Anchor {
                    entity_id: id,
                    score: rank as f32,
                    reason: "fallback:lexical:fact_match".to_string(),
                });
            }
        } else {
            anchors_map.insert(id, Anchor {
                entity_id: id,
                score: rank as f32,
                reason: "fallback:lexical:fact_match".to_string(),
            });
        }
    }

    if let Ok(rel_rows) = sqlx::query(
        "SELECT f.rowid, f.rank
         FROM ics_rel_fts f
         JOIN ics_beliefs b ON b.id = f.rowid
         WHERE ics_rel_fts MATCH ? AND b.status = 'active'
         ORDER BY f.rank
         LIMIT ?"
    )
    .bind(&fts_query)
    .bind(rel_limit.max(1))
    .fetch_all(pool)
    .await
    {
        let mut rel_scores: HashMap<i64, f32> = HashMap::new();
        for row in &rel_rows {
            let belief_id: i64 = row.get("rowid");
            let rank: f64 = row.get("rank");
            rel_scores.insert(belief_id, rank as f32);
        }

        if !rel_scores.is_empty() {
            let placeholders: String = rel_scores.keys().map(|_| "?").collect::<Vec<_>>().join(",");
            let query = format!(
                "SELECT rp.belief_id, rp.entity_id
                 FROM ics_rel_participants rp
                 WHERE rp.belief_id IN ({})",
                placeholders
            );
            let mut q = sqlx::query(&query);
            for belief_id in rel_scores.keys() {
                q = q.bind(belief_id);
            }

            let rel_part_rows = q.fetch_all(pool).await.unwrap_or_default();
            for row in rel_part_rows {
                let belief_id: i64 = row.get("belief_id");
                let entity_id: i64 = row.get("entity_id");
                let rank = *rel_scores.get(&belief_id).unwrap_or(&0.0);
                let reason = "fallback:lexical:rel_match".to_string();

                if let Some(existing) = anchors_map.get(&entity_id) {
                    if rank < existing.score {
                        anchors_map.insert(entity_id, Anchor {
                            entity_id,
                            score: rank,
                            reason,
                        });
                    }
                } else {
                    anchors_map.insert(entity_id, Anchor {
                        entity_id,
                        score: rank,
                        reason,
                    });
                }
            }
        }
    }

    Ok(anchors_map.into_values().collect())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }
    
    dot_product / (magnitude_a * magnitude_b)
}
