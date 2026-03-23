use serde_json::json;
use sqlx::SqlitePool;
use num_complex::Complex32;
use crate::core::cognitive_wave::{AmplitudeBounds, DecayProfile, WaveBand, WaveContributionInput};

use crate::core::system_log;

pub struct MergeResult {
    pub success: bool,
    pub beliefs_updated: usize,
}

/// Merge two entities (Spec §12.1)
pub async fn merge_entities(pool: &SqlitePool, from_id: i64, to_id: i64, reason: &str) -> Result<MergeResult, String> {
    // 1. Log event
    let _ = sqlx::query("INSERT INTO ics_merge_events (from_id, to_id, reason) VALUES (?, ?, ?)")
        .bind(from_id)
        .bind(to_id)
        .bind(reason)
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;
        
    // 2. Re-point Facts
    let res_facts = sqlx::query("UPDATE ics_fact_beliefs SET subject_entity_id = ? WHERE subject_entity_id = ?")
        .bind(to_id)
        .bind(from_id)
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;
        
    // 3. Re-point Relation Participants
    let res_rels = sqlx::query("UPDATE ics_rel_participants SET entity_id = ? WHERE entity_id = ?")
        .bind(to_id)
        .bind(from_id)
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;
        
    // 4. Mark old Entity as merged (Soft Delete / Redirect)
    let _ = sqlx::query("UPDATE ics_entities SET resolution_state = 'do_not_merge', label = label || ' (merged)' WHERE id = ?")
        .bind(from_id)
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;

    let _ = system_log::log_event(
        pool,
        None,
        "warn",
        "memory",
        None,
        None,
        json!({
            "event": "merge_no_rollback",
            "from_id": from_id,
            "to_id": to_id,
            "reason": reason,
        }),
    )
    .await;
        
    // Also remove from FTS to prevent anchor matching?
    // Triggers might handle this, or we rely on 'active' checks. 
    // Anchors uses `ics_entities_fts`. If we update label, FTS updates.
    
    let beliefs_updated = (res_facts.rows_affected() + res_rels.rows_affected()) as usize;
    if let Some(contribution) = contribution_from_merge(beliefs_updated) {
        let _ = crate::core::cognitive_wave::try_contribute(pool, None, &contribution, None, None).await;
    }

    Ok(MergeResult {
        success: true,
        beliefs_updated,
    })
}

pub async fn rollback_merge(_pool: &SqlitePool, _event_id: i64) -> Result<bool, String> {
    Err("Rollback not implemented".to_string())
}

fn contribution_from_merge(updated: usize) -> Option<WaveContributionInput> {
    if updated == 0 {
        return None;
    }
    let scaled = (updated as f32 / 10.0).clamp(0.0, 1.0);
    let amplitude = (0.05 + scaled * 0.4).clamp(0.05, 0.45);
    let mut coeffs = Vec::new();
    for idx in 0..8 {
        let phase = (idx as f32 * 0.25).cos();
        coeffs.push(Complex32::new(scaled, phase * scaled));
    }
    Some(WaveContributionInput {
        source: "memory_merge",
        band: WaveBand::Memory,
        coeffs,
        amplitude,
        amplitude_bounds: AmplitudeBounds::new(0.05, 0.45),
        decay_profile: DecayProfile::for_band(WaveBand::Memory),
    })
}
