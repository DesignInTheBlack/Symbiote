use sqlx::SqlitePool;
use sqlx::Row;
use crate::core::memory::config::{SKETCH_MAX_NEIGHBORS, SKETCH_MAX_TOKENS};
// use crate::core::memory::types::EntitySketch; // internal struct usage in SQL

/// Update EntitySketch caches (Spec §12.2.1)
pub async fn update_entity_sketches(pool: &SqlitePool) -> Result<usize, String> {
    // 1. Get all active entities (simplistic: all entities. Optimization: only those with recent activity)
    // For full correctness, we iterate all.
    let entities = sqlx::query("SELECT id FROM ics_entities WHERE resolution_state != 'merged'")
        .fetch_all(pool)
        .await
        .map_err(|e| e.to_string())?;
    
    let mut update_count = 0;
    
    for row in entities {
        let entity_id: i64 = row.get("id");
        
        // 2. Compute Tokens (Top K facts by weight)
        let token_rows = sqlx::query(
            "SELECT fb.value_literal 
             FROM ics_fact_beliefs fb
             JOIN ics_beliefs b ON b.id = fb.belief_id
             WHERE fb.subject_entity_id = ? AND b.status = 'active'
             ORDER BY b.evidence_weight_total DESC
             LIMIT ?"
        )
        .bind(entity_id)
        .bind(SKETCH_MAX_TOKENS as i64)
        .fetch_all(pool)
        .await
        .map_err(|e| e.to_string())?;
        
        let tokens: Vec<String> = token_rows.iter().map(|r| r.get("value_literal")).collect();
        let tokens_json = serde_json::to_string(&tokens).map_err(|e| e.to_string())?;
        
        // 3. Compute Neighbors (Top K entities connected by shared relations)
        // Join rel_participants -> rel_beliefs -> rel_participants (other)
        let neighbor_rows = sqlx::query(
            "SELECT rp2.entity_id, b.evidence_weight_total
             FROM ics_rel_participants rp1
             JOIN ics_rel_beliefs rb ON rb.belief_id = rp1.belief_id
             JOIN ics_beliefs b ON b.id = rb.belief_id
             JOIN ics_rel_participants rp2 ON rp2.belief_id = rb.belief_id
             WHERE rp1.entity_id = ? AND rp2.entity_id != ? AND b.status = 'active'
             ORDER BY b.evidence_weight_total DESC
             LIMIT ?"
        )
        .bind(entity_id)
        .bind(entity_id)
        .bind(SKETCH_MAX_NEIGHBORS as i64)
        .fetch_all(pool)
        .await
        .map_err(|e| e.to_string())?;
        
        let mut neighbors: Vec<i64> = neighbor_rows.iter().map(|r| r.get("entity_id")).collect();
        neighbors.sort();
        neighbors.dedup(); // Basic dedup, potentially simplified if multiple rels connect same entities
        let neighbors_json = serde_json::to_string(&neighbors).map_err(|e| e.to_string())?;
        
        // 4. Upsert into ics_entity_sketches
        let _ = sqlx::query(
            "INSERT INTO ics_entity_sketches (entity_id, neighbors_top, tokens_top, updated_at)
             VALUES (?, ?, ?, CURRENT_TIMESTAMP)
             ON CONFLICT(entity_id) DO UPDATE SET 
             neighbors_top = excluded.neighbors_top,
             tokens_top = excluded.tokens_top,
             updated_at = CURRENT_TIMESTAMP"
        )
        .bind(entity_id)
        .bind(neighbors_json)
        .bind(tokens_json)
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;
        
        update_count += 1;
    }
    
    Ok(update_count)
}
