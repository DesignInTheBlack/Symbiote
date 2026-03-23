use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use sqlx::{Row, SqlitePool};

use crate::core::memory::config::{
    SALIENCE_EVIDENCE_SCALE, SALIENCE_HALF_LIFE_HOURS, SALIENCE_MAX, SALIENCE_MIN,
    SALIENCE_WEIGHT_INFERENCE, SALIENCE_WEIGHT_OBSERVATION, SALIENCE_WEIGHT_SELF_CORE,
    SALIENCE_WEIGHT_USER,
};
use crate::core::memory::scope::parse_scope_str;
use crate::core::memory::types::{Scope, SourceType};

fn parse_source_type(raw: &str) -> Option<SourceType> {
    match raw.trim().to_lowercase().as_str() {
        "user" => Some(SourceType::User),
        "user_focus" => Some(SourceType::User),
        "tool" => Some(SourceType::Tool),
        "system" => Some(SourceType::System),
        "inference" => Some(SourceType::Inference),
        _ => None,
    }
}

fn weight_for_source(source: Option<SourceType>) -> f32 {
    match source {
        Some(SourceType::User) => SALIENCE_WEIGHT_USER,
        Some(SourceType::Tool) | Some(SourceType::System) => SALIENCE_WEIGHT_OBSERVATION,
        Some(SourceType::Inference) => SALIENCE_WEIGHT_INFERENCE,
        None => SALIENCE_WEIGHT_INFERENCE,
    }
}

fn base_weight_for(scope: Option<Scope>, sources: &[SourceType]) -> f32 {
    if matches!(scope, Some(Scope::SelfScope)) {
        return SALIENCE_WEIGHT_SELF_CORE;
    }
    let mut max_weight = SALIENCE_WEIGHT_INFERENCE;
    for source in sources {
        max_weight = max_weight.max(weight_for_source(Some(*source)));
    }
    max_weight
}

fn parse_timestamp(raw: &str) -> Option<DateTime<Utc>> {
    if let Ok(ts) = DateTime::parse_from_rfc3339(raw) {
        return Some(ts.with_timezone(&Utc));
    }
    if let Ok(naive) = NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S") {
        return Some(Utc.from_utc_datetime(&naive));
    }
    None
}

fn recency_factor(last_evidence_at: Option<&str>) -> f32 {
    let Some(raw) = last_evidence_at else {
        return 1.0;
    };
    let Some(ts) = parse_timestamp(raw) else {
        return 1.0;
    };
    let now = Utc::now();
    let age_seconds = (now - ts).num_seconds().max(0) as f32;
    let age_hours = age_seconds / 3600.0;
    if SALIENCE_HALF_LIFE_HOURS <= 0.0 {
        return 1.0;
    }
    0.5_f32.powf(age_hours / SALIENCE_HALF_LIFE_HOURS)
}

fn compute_salience(
    scope: Option<Scope>,
    sources: &[SourceType],
    evidence_weight_total: f32,
    last_evidence_at: Option<&str>,
) -> f32 {
    let base_weight = base_weight_for(scope, sources);
    let evidence_boost = 1.0 + (evidence_weight_total.max(0.0)).ln_1p() * SALIENCE_EVIDENCE_SCALE;
    let recency = recency_factor(last_evidence_at);
    (base_weight * evidence_boost * recency).clamp(SALIENCE_MIN, SALIENCE_MAX)
}

/// Recompute salience for a specific list of belief IDs using deterministic weights.
pub async fn recompute_salience_for_beliefs(
    pool: &SqlitePool,
    belief_ids: &[i64],
) -> Result<usize, String> {
    if belief_ids.is_empty() {
        return Ok(0);
    }
    let placeholders = belief_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let query = format!(
        "SELECT id, scope, evidence_weight_total, last_evidence_at
         FROM ics_beliefs
         WHERE id IN ({})",
        placeholders
    );
    let mut stmt = sqlx::query(&query);
    for id in belief_ids {
        stmt = stmt.bind(id);
    }
    let belief_rows = stmt.fetch_all(pool).await.map_err(|e| e.to_string())?;

    let evidence_query = format!(
        "SELECT belief_id, source_type
         FROM ics_evidence_events
         WHERE belief_id IN ({})",
        placeholders
    );
    let mut evidence_stmt = sqlx::query(&evidence_query);
    for id in belief_ids {
        evidence_stmt = evidence_stmt.bind(id);
    }
    let evidence_rows = evidence_stmt.fetch_all(pool).await.map_err(|e| e.to_string())?;

    let mut sources_map: std::collections::HashMap<i64, Vec<SourceType>> = std::collections::HashMap::new();
    for row in evidence_rows {
        let belief_id: i64 = row.get("belief_id");
        let source_raw: String = row.try_get("source_type").unwrap_or_default();
        if let Some(source_type) = parse_source_type(&source_raw) {
            sources_map.entry(belief_id).or_default().push(source_type);
        }
    }

    let mut updated = 0;
    for row in belief_rows {
        let belief_id: i64 = row.get("id");
        let scope_raw: String = row.try_get("scope").unwrap_or_default();
        let evidence_weight_total: f32 = row
            .try_get::<f64, _>("evidence_weight_total")
            .unwrap_or(0.0) as f32;
        let last_evidence_at: Option<String> = row.try_get("last_evidence_at").ok();
        let scope = parse_scope_str(&scope_raw);
        let sources = sources_map.get(&belief_id).cloned().unwrap_or_default();
        let salience = compute_salience(
            scope,
            &sources,
            evidence_weight_total,
            last_evidence_at.as_deref(),
        );
        sqlx::query("UPDATE ics_beliefs SET salience = ? WHERE id = ?")
            .bind(salience)
            .bind(belief_id)
            .execute(pool)
            .await
            .map_err(|e| e.to_string())?;
        updated += 1;
    }

    Ok(updated)
}

/// Recompute salience for all active beliefs.
pub async fn recompute_salience_for_all(pool: &SqlitePool) -> Result<usize, String> {
    let ids: Vec<i64> = sqlx::query_scalar(
        "SELECT id FROM ics_beliefs WHERE status = 'active'",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    let mut updated = 0;
    for chunk in ids.chunks(200) {
        updated += recompute_salience_for_beliefs(pool, chunk).await?;
    }
    Ok(updated)
}

/// Legacy helper retained for compatibility; uses recompute path for deterministic behavior.
pub async fn boost_salience_for_beliefs(
    pool: &SqlitePool,
    belief_ids: &[i64],
    _amount: f32,
) -> Result<(), String> {
    let _ = recompute_salience_for_beliefs(pool, belief_ids).await?;
    Ok(())
}

pub fn calculate_decayed_salience(current_salience: f32, decay_rate: f32, comparisons: f32) -> f32 {
    current_salience * (1.0 - decay_rate).powf(comparisons)
}
