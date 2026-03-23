use std::collections::HashMap;

use chrono::Utc;
use serde_json::json;
use sqlx::Row;

use crate::core::cognitive_wave::{wave_field_handle, WaveBand};
use crate::core::subject_state::ErrorState;
use crate::db::Db;
use crate::models::AttentionSchemaState;

#[derive(Debug, Clone)]
struct MapRange {
    label: String,
    min: f64,
    max: f64,
}

async fn load_active_mapping(
    db: &Db,
) -> Option<(i64, bool, HashMap<String, Vec<MapRange>>)> {
    let version: Option<i64> = sqlx::query_scalar(
        "SELECT MAX(version) FROM internal_state_map",
    )
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let Some(version) = version else { return None; };

    let rows = sqlx::query(
        "SELECT metric, range_min, range_max, label, degraded
         FROM internal_state_map
         WHERE version = ?",
    )
    .bind(version)
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();

    let mut map: HashMap<String, Vec<MapRange>> = HashMap::new();
    let mut degraded = false;
    for row in rows {
        let metric: String = row.try_get("metric").unwrap_or_default();
        let range_min: f64 = row.try_get("range_min").unwrap_or(0.0);
        let range_max: f64 = row.try_get("range_max").unwrap_or(0.0);
        let label: String = row.try_get("label").unwrap_or_else(|_| "unknown".to_string());
        let row_degraded: i64 = row.try_get("degraded").unwrap_or(0);
        if row_degraded > 0 {
            degraded = true;
        }
        map.entry(metric).or_default().push(MapRange {
            label,
            min: range_min,
            max: range_max,
        });
    }

    Some((version, degraded, map))
}

fn apply_mapping(
    metrics: &HashMap<String, f64>,
    mapping: &HashMap<String, Vec<MapRange>>,
) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    for (metric, value) in metrics.iter() {
        let label = mapping
            .get(metric)
            .and_then(|ranges| {
                ranges.iter().find(|range| *value >= range.min && *value <= range.max)
            })
            .map(|range| range.label.clone())
            .unwrap_or_else(|| "unknown".to_string());
        labels.insert(metric.clone(), label);
    }
    labels
}

pub async fn compute_internal_state_summary(
    db: &Db,
    attention_schema: &AttentionSchemaState,
    error_state: &ErrorState,
) -> (serde_json::Value, Option<i64>) {
    let mut metrics: HashMap<String, f64> = HashMap::new();

    if let Some(handle) = wave_field_handle() {
        let field = handle.read().await;
        for band in WaveBand::all() {
            let amplitude = field.band_energy(band) as f64;
            metrics.insert(format!("wave.band.{}.amplitude", band.label()), amplitude);
        }
    }

    metrics.insert("attention.capacity_usage".to_string(), attention_schema.capacity_usage);
    metrics.insert("attention.stability".to_string(), attention_schema.stability);

    if !error_state.recent_residuals.is_empty() {
        let mut norm_sum = 0.0f64;
        let mut salience_sum = 0.0f64;
        let count = error_state.recent_residuals.len() as f64;
        for residual in error_state.recent_residuals.iter() {
            norm_sum += residual.normalized_residual.abs();
            salience_sum += residual.salience_score;
        }
        metrics.insert("residual.normalized_residual".to_string(), norm_sum / count);
        metrics.insert("residual.salience_score".to_string(), salience_sum / count);
    }

    let mapping = load_active_mapping(db).await;
    let (labels, mapping_version, mapping_degraded) = if let Some((version, degraded, map)) = mapping {
        (apply_mapping(&metrics, &map), Some(version), degraded)
    } else {
        (HashMap::new(), None, false)
    };

    let summary = json!({
        "mapping_version": mapping_version,
        "mapping_degraded": mapping_degraded,
        "metrics": metrics,
        "labels": labels,
        "timestamp": Utc::now().to_rfc3339(),
    });

    (summary, mapping_version)
}
