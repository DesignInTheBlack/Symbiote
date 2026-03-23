use num_complex::Complex32;
use sqlx::SqlitePool;
use sqlx::Row;
use crate::core::cognitive_wave::{AmplitudeBounds, DecayProfile, WaveBand, WaveContributionInput};
use crate::core::memory::config::{MAX_EVIDENCE_EVENTS_PER_BELIEF, KEEP_NEWEST_EVIDENCE, KEEP_TOP_WEIGHTED_EVIDENCE, KEEP_UNIQUE_SNIPPETS};

/// Enforce I5: Evidence Growth Bounds
/// Deletes excess evidence events for a belief, prioritizing:
/// 1. Age (Keep Newest)
/// 2. Weight (Keep Top)
/// 3. Unique Snippets (Keep Diversity)
pub async fn compact_evidence(belief_id: i64, pool: &SqlitePool) -> Result<(), String> {
    // 1. Check count
    let count: i64 = sqlx::query("SELECT COUNT(*) as c FROM ics_evidence_events WHERE belief_id = ?")
        .bind(belief_id)
        .fetch_one(pool)
        .await
        .map_err(|e| e.to_string())?
        .get("c");

    if count <= MAX_EVIDENCE_EVENTS_PER_BELIEF as i64 {
        return Ok(());
    }

    // 2. Identification of IDs to KEEP
    // Strategy: Collect IDs in Rust to avoid complex SQL nested queries (sqlite limits) 
    // or use a temporary table. Simple collection is fine for small N (~50).
    
    // a) Get all events
    let rows = sqlx::query("SELECT id, weight, created_at, snippet_hash FROM ics_evidence_events WHERE belief_id = ? ORDER BY created_at DESC")
        .bind(belief_id)
        .fetch_all(pool)
        .await
        .map_err(|e| e.to_string())?;

    let mut keep_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut processed_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // b) Keep newest N
    for i in 0..KEEP_NEWEST_EVIDENCE {
        if i >= rows.len() { break; }
        let id: i64 = rows[i].get("id");
        keep_ids.insert(id);
        processed_indices.insert(i);
    }

    // c) From remaining, keep top N by weight
    let mut remaining: Vec<(usize, f64)> = rows.iter().enumerate()
        .filter(|(i, _)| !processed_indices.contains(i))
        .map(|(i, r)| (i, r.get::<f64, _>("weight")))
        .collect();
    
    // Sort desc by weight
    remaining.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut weight_kept_count = 0;
    for (idx, _w) in &remaining {
        if weight_kept_count >= KEEP_TOP_WEIGHTED_EVIDENCE { break; }
        let id: i64 = rows[*idx].get("id");
        keep_ids.insert(id);
        processed_indices.insert(*idx);
        weight_kept_count += 1;
    }

    // d) From remaining, keep unique snippet hashes
    // Collect snippets we already have in kept set? No, we want *additional* unique snippets.
    let mut known_hashes: std::collections::HashSet<String> = std::collections::HashSet::new();
    
    // Fill known hashes from already kept
    for idx in &processed_indices {
         if let Ok(h) = rows[*idx].try_get::<String, _>("snippet_hash") {
             known_hashes.insert(h);
         }
    }

    let mut snippet_kept_count = 0;
    // Iterate original rows by age (desc) to prefer newer unique snippets?
    // "remaining" is sorted by weight. Let's re-sort remaining valid indices by age?
    // Actually, simple iteration over original rows check is easier.
    for (i, row) in rows.iter().enumerate() {
        if processed_indices.contains(&i) { continue; }
        if snippet_kept_count >= KEEP_UNIQUE_SNIPPETS { break; }
        
        if let Ok(h) = row.try_get::<String, _>("snippet_hash") {
            if !known_hashes.contains(&h) {
                let id: i64 = row.get("id");
                keep_ids.insert(id);
                known_hashes.insert(h.clone());
                snippet_kept_count += 1;
            }
        }
    }
    
    // 3. Delete everything NOT in keep_ids
    if keep_ids.is_empty() { return Ok(()); } // Safety

    let ids_str = keep_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(",");
    let query = format!("DELETE FROM ics_evidence_events WHERE belief_id = ? AND id NOT IN ({})", ids_str);
    
    sqlx::query(&query)
        .bind(belief_id)
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;

    let removed = (count - keep_ids.len() as i64).max(0);
    if let Some(contribution) = contribution_from_compaction(removed, count) {
        let _ = crate::core::cognitive_wave::try_contribute(pool, None, &contribution, None, None).await;
    }

    Ok(())
}

fn contribution_from_compaction(removed: i64, total: i64) -> Option<WaveContributionInput> {
    if removed <= 0 || total <= 0 {
        return None;
    }
    let ratio = (removed as f32 / total as f32).clamp(0.0, 1.0);
    let amplitude = (0.05 + ratio * 0.35).clamp(0.05, 0.4);
    let mut coeffs = Vec::new();
    for idx in 0..8 {
        let phase = (idx as f32 * 0.35).sin();
        coeffs.push(Complex32::new(ratio, phase * ratio));
    }
    Some(WaveContributionInput {
        source: "memory_compaction",
        band: WaveBand::Memory,
        coeffs,
        amplitude,
        amplitude_bounds: AmplitudeBounds::new(0.05, 0.4),
        decay_profile: DecayProfile::for_band(WaveBand::Memory),
    })
}
