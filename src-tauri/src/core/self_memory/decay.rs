use sqlx::{SqlitePool, Row};

pub async fn decay_self_memory(pool: &SqlitePool, max_age_days: i64, min_weight: f32) -> Result<usize, String> {
    let threshold = format!("-{} days", max_age_days);

    let rows = sqlx::query(
        "SELECT id FROM self_beliefs
         WHERE evidence_weight_total < ?
           AND created_at < datetime('now', ?)"
    )
    .bind(min_weight)
    .bind(threshold)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    if rows.is_empty() {
        return Ok(0);
    }

    let ids: Vec<i64> = rows
        .iter()
        .filter_map(|r| r.try_get::<i64, _>("id").ok())
        .collect();

    let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let query = format!("DELETE FROM self_fact_beliefs WHERE belief_id IN ({})", placeholders);
    let mut q = sqlx::query(&query);
    for id in &ids {
        q = q.bind(id);
    }
    let _ = q.execute(pool).await;

    let query = format!("DELETE FROM self_rel_participants WHERE belief_id IN ({})", placeholders);
    let mut q = sqlx::query(&query);
    for id in &ids {
        q = q.bind(id);
    }
    let _ = q.execute(pool).await;

    let query = format!("DELETE FROM self_rel_beliefs WHERE belief_id IN ({})", placeholders);
    let mut q = sqlx::query(&query);
    for id in &ids {
        q = q.bind(id);
    }
    let _ = q.execute(pool).await;

    let query = format!("DELETE FROM self_evidence_events WHERE belief_id IN ({})", placeholders);
    let mut q = sqlx::query(&query);
    for id in &ids {
        q = q.bind(id);
    }
    let _ = q.execute(pool).await;

    let query = format!("DELETE FROM self_beliefs WHERE id IN ({})", placeholders);
    let mut q = sqlx::query(&query);
    for id in &ids {
        q = q.bind(id);
    }
    let deleted = q.execute(pool).await.map_err(|e| e.to_string())?.rows_affected();
    Ok(deleted as usize)
}
