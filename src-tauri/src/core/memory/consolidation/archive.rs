//! Conflict Archiving (ICS v4.1 §12.2.5)
//! Archives old resolved conflicts to keep the active set manageable.

use sqlx::SqlitePool;

/// Configuration for conflict archiving
pub struct ArchiveConfig {
    /// Days after resolution before archiving
    pub archive_after_days: i64,
}

impl Default for ArchiveConfig {
    fn default() -> Self {
        Self {
            archive_after_days: 30,
        }
    }
}

/// Archive old resolved conflicts per §12.2.5
/// Updates status from 'resolved' to 'archived' for old conflicts
pub async fn archive_old_conflicts(
    pool: &SqlitePool,
    config: &ArchiveConfig,
) -> Result<usize, String> {
    let cutoff_date = chrono::Utc::now() - chrono::Duration::days(config.archive_after_days);
    let cutoff_str = cutoff_date.to_rfc3339();
    
    let result = sqlx::query(
        "UPDATE ics_conflict_sets 
         SET status = 'archived', updated_at = CURRENT_TIMESTAMP
         WHERE status = 'resolved' AND updated_at < ?"
    )
    .bind(&cutoff_str)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;
    
    Ok(result.rows_affected() as usize)
}
