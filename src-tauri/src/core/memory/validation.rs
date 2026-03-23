use chrono::{DateTime, Utc};
use sqlx::{Row, SqlitePool};
use tauri::AppHandle;
use crate::core::system_log;

#[derive(Debug, Clone)]
pub struct MemoryValidationConfig {
    pub max_beliefs: i64,
    pub min_interval_minutes: i64,
    pub decay_per_day: f32,
    pub min_confidence: f32,
    pub drift_threshold: f32,
}

impl Default for MemoryValidationConfig {
    fn default() -> Self {
        Self {
            max_beliefs: 400,
            min_interval_minutes: 60,
            decay_per_day: 0.99,
            min_confidence: 0.05,
            drift_threshold: 0.05,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryValidationResult {
    pub scanned: i64,
    pub updated: i64,
    pub drift_events: i64,
    pub max_drop: f32,
}

fn parse_ts(raw: Option<String>) -> Option<DateTime<Utc>> {
    let raw = raw?;
    DateTime::parse_from_rfc3339(&raw)
        .map(|dt| dt.with_timezone(&Utc))
        .or_else(|_| {
            chrono::NaiveDateTime::parse_from_str(&raw, "%Y-%m-%d %H:%M:%S")
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
        })
        .ok()
}

async fn validate_table(
    pool: &SqlitePool,
    table: &str,
    config: &MemoryValidationConfig,
) -> Result<MemoryValidationResult, String> {
    let now = Utc::now();
    let cutoff = now - chrono::Duration::minutes(config.min_interval_minutes);
    let cutoff_str = cutoff.to_rfc3339();
    let limit = config.max_beliefs.max(1);

    let rows = sqlx::query(&format!(
        "SELECT id, confidence, last_evidence_at, last_validated_at
         FROM {table}
         WHERE status = 'active'
           AND (last_validated_at IS NULL OR datetime(last_validated_at) <= datetime(?))
         ORDER BY datetime(last_evidence_at) ASC
         LIMIT ?",
        table = table
    ))
    .bind(&cutoff_str)
    .bind(limit)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    let mut scanned = 0i64;
    let mut updated = 0i64;
    let mut drift_events = 0i64;
    let mut max_drop: f32 = 0.0;

    for row in rows {
        scanned += 1;
        let id: i64 = row.get("id");
        let confidence: f32 = row.try_get::<f64, _>("confidence").ok().unwrap_or(1.0) as f32;
        let last_evidence_at = parse_ts(row.try_get("last_evidence_at").ok());
        let age_days = last_evidence_at
            .map(|ts| (now - ts).num_seconds().max(0) as f32 / 86_400.0)
            .unwrap_or(0.0);
        let decay = config.decay_per_day.powf(age_days.max(0.0));
        let new_confidence = (confidence * decay).clamp(config.min_confidence, 1.0);
        let drop = (confidence - new_confidence).max(0.0);
        if drop >= config.drift_threshold {
            drift_events += 1;
            if drop > max_drop {
                max_drop = drop;
            }
        }

        if (new_confidence - confidence).abs() > 0.0001 {
            updated += 1;
        }

        let _ = sqlx::query(&format!(
            "UPDATE {table}
             SET confidence = ?, last_validated_at = CURRENT_TIMESTAMP
             WHERE id = ?",
            table = table
        ))
        .bind(new_confidence as f64)
        .bind(id)
        .execute(pool)
        .await;
    }

    Ok(MemoryValidationResult {
        scanned,
        updated,
        drift_events,
        max_drop,
    })
}

pub async fn validate_memory_beliefs(
    pool: &SqlitePool,
    config: &MemoryValidationConfig,
    app_handle: Option<&AppHandle>,
) -> Result<(MemoryValidationResult, MemoryValidationResult), String> {
    let ics = validate_table(pool, "ics_beliefs", config).await?;
    let self_mem = validate_table(pool, "self_beliefs", config).await?;

    let _ = system_log::log_event(
        pool,
        app_handle,
        "info",
        "memory",
        None,
        None,
        serde_json::json!({
            "event": "memory_validation_run",
            "ics": {
                "scanned": ics.scanned,
                "updated": ics.updated,
                "drift_events": ics.drift_events,
                "max_drop": ics.max_drop,
            },
            "self": {
                "scanned": self_mem.scanned,
                "updated": self_mem.updated,
                "drift_events": self_mem.drift_events,
                "max_drop": self_mem.max_drop,
            }
        }),
    )
    .await;

    if ics.drift_events > 0 || self_mem.drift_events > 0 {
        let _ = system_log::log_event(
            pool,
            app_handle,
            "warn",
            "memory",
            None,
            None,
            serde_json::json!({
                "event": "memory_validation_drift",
                "ics_drift_events": ics.drift_events,
                "self_drift_events": self_mem.drift_events,
                "ics_max_drop": ics.max_drop,
                "self_max_drop": self_mem.max_drop,
            }),
        )
        .await;
    }

    Ok((ics, self_mem))
}
