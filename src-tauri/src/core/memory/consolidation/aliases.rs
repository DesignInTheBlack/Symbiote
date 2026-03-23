use sqlx::SqlitePool;
use crate::core::memory::config::CONFIRMED_EVIDENCE_THRESHOLD;
use crate::core::memory::canonical::canonicalize_label;
use sqlx::Row;

/// Promote eligible aliases (Spec §12.2.5)
pub async fn promote_eligible_aliases(pool: &SqlitePool) -> Result<usize, String> {
    // 1. Find Proposed aliases with evidence > threshold
    let token_result = sqlx::query(
        "UPDATE ics_token_aliases 
         SET status = 'confirmed' 
         WHERE status = 'proposed' AND evidence_count >= ?"
    )
    .bind(CONFIRMED_EVIDENCE_THRESHOLD as i64)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    let entity_promoted = promote_entity_aliases(pool).await?;
    Ok(token_result.rows_affected() as usize + entity_promoted)
}

async fn promote_entity_aliases(pool: &SqlitePool) -> Result<usize, String> {
    let rows = sqlx::query(
        "SELECT entity_id, alias, alias_canonical
         FROM ics_entity_aliases
         WHERE status = 'proposed' AND evidence_count >= ?"
    )
    .bind(CONFIRMED_EVIDENCE_THRESHOLD as i64)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    if rows.is_empty() {
        return Ok(0);
    }

    let mut tx = pool.begin().await.map_err(|e| e.to_string())?;
    let mut promoted = 0usize;

    for row in rows {
        let entity_id: i64 = row.get("entity_id");
        let alias: String = row.get("alias");
        let alias_canon: String = row.get("alias_canonical");

        let entity_row = sqlx::query(
            "SELECT aliases, aliases_canonical, label FROM ics_entities WHERE id = ?"
        )
        .bind(entity_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| e.to_string())?;

        let Some(entity_row) = entity_row else { continue; };
        let label: String = entity_row.get("label");
        let label_canon = canonicalize_label(&label);
        if alias_canon == label_canon {
            let _ = sqlx::query(
                "UPDATE ics_entity_aliases
                 SET status = 'confirmed', updated_at = CURRENT_TIMESTAMP
                 WHERE entity_id = ? AND alias_canonical = ?"
            )
            .bind(entity_id)
            .bind(&alias_canon)
            .execute(&mut *tx)
            .await;
            promoted += 1;
            continue;
        }

        let aliases_raw: Option<String> = entity_row.try_get("aliases").ok();
        let aliases_canon_raw: Option<String> = entity_row.try_get("aliases_canonical").ok();
        let mut aliases: Vec<String> = aliases_raw
            .as_deref()
            .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
            .unwrap_or_default();
        let mut aliases_canon: Vec<String> = aliases_canon_raw
            .as_deref()
            .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
            .unwrap_or_default();
        if aliases.is_empty() {
            aliases_canon.clear();
        } else if aliases_canon.len() != aliases.len() {
            aliases_canon = aliases.iter().map(|a| canonicalize_label(a)).collect();
        }

        if !aliases_canon.iter().any(|c| c == &alias_canon) {
            aliases.push(alias.clone());
            aliases_canon.push(alias_canon.clone());
            let aliases_json = serde_json::to_string(&aliases).unwrap_or_else(|_| "[]".to_string());
            let aliases_canon_json = serde_json::to_string(&aliases_canon).unwrap_or_else(|_| "[]".to_string());
            let _ = sqlx::query(
                "UPDATE ics_entities SET aliases = ?, aliases_canonical = ? WHERE id = ?"
            )
            .bind(aliases_json)
            .bind(aliases_canon_json)
            .bind(entity_id)
            .execute(&mut *tx)
            .await;
        }

        let _ = sqlx::query(
            "UPDATE ics_entity_aliases
             SET status = 'confirmed', updated_at = CURRENT_TIMESTAMP
             WHERE entity_id = ? AND alias_canonical = ?"
        )
        .bind(entity_id)
        .bind(&alias_canon)
        .execute(&mut *tx)
        .await;
        promoted += 1;
    }

    let _ = tx.commit().await;
    Ok(promoted)
}
