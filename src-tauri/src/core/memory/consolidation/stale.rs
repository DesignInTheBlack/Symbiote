//! Stale Inference Deactivation (ICS v4.1 §12.2.4)
//! Marks low-confidence inferences that haven't been reinforced as inactive.

use sqlx::SqlitePool;
use sqlx::Row;

/// Configuration for stale inference detection
pub struct StaleInferenceConfig {
    /// Confidence threshold below which inferences are considered weak
    pub min_confidence: f32,
    /// Days since last evidence before considered stale
    pub stale_days: i64,
    /// Confidence threshold for provisional items
    pub provisional_confidence: f32,
}

impl Default for StaleInferenceConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            stale_days: 45,
            provisional_confidence: 0.5,
        }
    }
}

/// Deactivate stale inferences per §12.2.4
/// Finds inferred beliefs with:
/// - Low confidence (< min_confidence)
/// - No reinforcement in stale_days
/// - Not supported by user/tool evidence
/// Marks as inactive (never deletes)
pub async fn deactivate_stale_inferences(
    pool: &SqlitePool,
    config: &StaleInferenceConfig,
) -> Result<usize, String> {
    // Find stale inference beliefs
    // Criteria:
    // 1. Source type is 'inference' (check evidence_events)
    // 2. Confidence < threshold
    // 3. Last evidence older than stale_days
    // 4. No user/tool evidence supporting it
    
    let cutoff_date = chrono::Utc::now() - chrono::Duration::days(config.stale_days);
    let cutoff_str = cutoff_date.to_rfc3339();
    
    // Get beliefs that are potentially stale (weak inferences)
    let stale_beliefs = sqlx::query(
        "SELECT b.id, b.confidence, b.last_evidence_at
         FROM ics_beliefs b
         WHERE b.status = 'active'
           AND b.confidence < ?
           AND b.last_evidence_at < ?
           AND NOT EXISTS (
               SELECT 1 FROM ics_evidence_events e 
               WHERE e.belief_id = b.id 
               AND e.source_type IN ('user', 'tool')
           )"
    )
    .bind(config.min_confidence)
    .bind(&cutoff_str)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;
    
    let mut deactivated = 0;
    
    for row in stale_beliefs {
        let belief_id: i64 = row.get("id");
        
        // Mark as inactive (never delete per spec)
        let _ = sqlx::query("UPDATE ics_beliefs SET status = 'inactive' WHERE id = ?")
            .bind(belief_id)
            .execute(pool)
            .await
            .map_err(|e| e.to_string())?;
        
        deactivated += 1;
    }

    // Deactivate provisional (low-confidence) beliefs that were not reinforced
    let provisional_beliefs = sqlx::query(
        "SELECT b.id
         FROM ics_beliefs b
         WHERE b.status = 'active'
           AND b.confidence < ?
           AND b.last_evidence_at < ?"
    )
    .bind(config.provisional_confidence)
    .bind(&cutoff_str)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    for row in provisional_beliefs {
        let belief_id: i64 = row.get("id");
        let _ = sqlx::query("UPDATE ics_beliefs SET status = 'inactive' WHERE id = ?")
            .bind(belief_id)
            .execute(pool)
            .await
            .map_err(|e| e.to_string())?;
        deactivated += 1;
    }

    Ok(deactivated)
}
