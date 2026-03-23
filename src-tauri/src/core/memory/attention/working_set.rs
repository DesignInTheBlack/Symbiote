use sqlx::SqlitePool;

/// Spreading Activation Config (Spec §14)
const SPREAD_FACTOR: f64 = 0.3; // How much activation spreads to neighbors
const MAX_ACTIVATION: f64 = 1.0;
const OUTCOME_BOOST_CAP: f64 = 1.0;

/// Update working set with accessed/written items (Spec §14)
pub async fn update_working_set(pool: &SqlitePool, accessed_entities: &[i64], written_beliefs: &[i64]) -> Result<(), String> {
    // 1. Boost accessed entities AND spread activation to neighbors
    for &id in accessed_entities {
        // Upsert activation = 1.0 (full boost)
        let _ = sqlx::query(
            "INSERT INTO ics_working_set (item_id, item_type, activation, last_updated_at) 
             VALUES (?, 'entity', 1.0, CURRENT_TIMESTAMP)
             ON CONFLICT(item_id, item_type) DO UPDATE SET activation = 1.0, last_updated_at = CURRENT_TIMESTAMP"
        )
        .bind(id)
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;
        
        // Spread activation to related entities
        let _ = spread_activation(id, 1.0, pool).await;
    }
    
    // 2. Boost written beliefs
    for &id in written_beliefs {
         let _ = sqlx::query(
            "INSERT INTO ics_working_set (item_id, item_type, activation, last_updated_at) 
             VALUES (?, 'belief', 1.0, CURRENT_TIMESTAMP)
             ON CONFLICT(item_id, item_type) DO UPDATE SET activation = 1.0, last_updated_at = CURRENT_TIMESTAMP"
        )
        .bind(id)
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;
    }
    
    Ok(())
}

/// Spread activation to related entities (Spec §14 - Working Set)
/// When an entity is accessed, related entities gain partial activation.
pub async fn spread_activation(entity_id: i64, source_activation: f64, pool: &SqlitePool) -> Result<(), String> {
    // Get 1-hop neighbors via relations (entities that share a belief with this entity)
    let neighbors: Vec<i64> = sqlx::query_scalar(
        "SELECT DISTINCT rp2.entity_id 
         FROM ics_rel_participants rp1
         JOIN ics_rel_participants rp2 ON rp2.belief_id = rp1.belief_id
         WHERE rp1.entity_id = ? AND rp2.entity_id != ?
         LIMIT 20"  // Cap to prevent runaway spreading on highly-connected entities
    )
    .bind(entity_id)
    .bind(entity_id)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;
    
    if neighbors.is_empty() {
        return Ok(());
    }
    
    let spread_amount = source_activation * SPREAD_FACTOR;
    
    for neighbor_id in neighbors {
        let _ = sqlx::query(
            "INSERT INTO ics_working_set (item_id, item_type, activation, last_updated_at)
             VALUES (?, 'entity', ?, CURRENT_TIMESTAMP)
             ON CONFLICT(item_id, item_type) DO UPDATE SET 
                activation = MIN(?, activation + ?),
                last_updated_at = CURRENT_TIMESTAMP"
        )
        .bind(neighbor_id)
        .bind(spread_amount)
        .bind(MAX_ACTIVATION)
        .bind(spread_amount)
        .execute(pool)
        .await;
    }
    
    Ok(())
}

/// Apply outcome-weighted boosts from episodic signals (Phase 5).
pub async fn apply_outcome_boost(pool: &SqlitePool, belief_id: i64, boost: f64) -> Result<(), String> {
    if boost <= 0.0 {
        return Ok(());
    }

    let _ = sqlx::query(
        "INSERT INTO ics_working_set (item_id, item_type, activation, last_updated_at)
         VALUES (?, 'belief', ?, CURRENT_TIMESTAMP)
         ON CONFLICT(item_id, item_type) DO UPDATE SET
            activation = MIN(?, activation + ?),
            last_updated_at = CURRENT_TIMESTAMP"
    )
    .bind(belief_id)
    .bind(boost)
    .bind(OUTCOME_BOOST_CAP)
    .bind(boost)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    Ok(())
}
