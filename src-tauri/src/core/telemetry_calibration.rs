use serde_json::json;
use sqlx::Row;
use tauri::AppHandle;

use crate::core::system_log;
use crate::db::Db;

const TELEMETRY_CALIBRATION_WINDOW_MINUTES: i64 = 60;
const DRIFT_THRESHOLD: f64 = 0.15;

async fn fetch_kv_rate(db: &Db, key: &str) -> Option<f64> {
    let row = sqlx::query("SELECT value FROM kv_store WHERE key = ?")
        .bind(key)
        .fetch_optional(&db.pool)
        .await
        .ok()
        .flatten()?;
    let raw: String = row.get("value");
    raw.parse::<f64>().ok()
}

async fn upsert_kv_rate(db: &Db, key: &str, value: f64) {
    let _ = sqlx::query(
        "INSERT INTO kv_store (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
    )
    .bind(key)
    .bind(value.to_string())
    .execute(&db.pool)
    .await;
}

async fn record_calibration(
    db: &Db,
    metric: &str,
    observed_rate: f64,
    expected_rate: Option<f64>,
    drift: f64,
    sample_count: i64,
    window_minutes: i64,
) {
    let _ = sqlx::query(
        "INSERT INTO telemetry_calibrations (metric, observed_rate, expected_rate, drift, sample_count, window_minutes, created_at)
         VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
    )
    .bind(metric)
    .bind(observed_rate)
    .bind(expected_rate)
    .bind(drift)
    .bind(sample_count)
    .bind(window_minutes)
    .execute(&db.pool)
    .await;
}

pub async fn run_telemetry_calibration(db: &Db, app: Option<&AppHandle>) -> Result<(), String> {
    let window_expr = format!("-{} minutes", TELEMETRY_CALIBRATION_WINDOW_MINUTES);
    let rows = sqlx::query(
        "SELECT status, COUNT(*) as count
         FROM tool_dispatches
         WHERE datetime(created_at) >= datetime('now', ?)
         GROUP BY status",
    )
    .bind(&window_expr)
    .fetch_all(&db.pool)
    .await
    .map_err(|e| e.to_string())?;

    let mut success = 0i64;
    let mut failed = 0i64;
    for row in rows {
        let status: String = row.get("status");
        let count: i64 = row.try_get("count").unwrap_or(0);
        match status.as_str() {
            "success" => success = count,
            "failed" => failed = count,
            _ => {}
        }
    }
    let total = success + failed;
    if total == 0 {
        let _ = system_log::log_event(
            &db.pool,
            app,
            "info",
            "telemetry",
            None,
            None,
            json!({
                "event": "telemetry_calibration_skipped",
                "reason": "no_tool_samples",
                "window_minutes": TELEMETRY_CALIBRATION_WINDOW_MINUTES,
            }),
        )
        .await;
        return Ok(());
    }

    let observed_success = success as f64 / total as f64;
    let observed_failure = failed as f64 / total as f64;

    let metrics = [
        ("telemetry.tool_success_rate", observed_success),
        ("telemetry.tool_failure_rate", observed_failure),
    ];

    for (metric, observed) in metrics {
        let expected = fetch_kv_rate(db, metric).await;
        let drift = expected.map(|val| observed - val).unwrap_or(0.0);
        record_calibration(
            db,
            metric,
            observed,
            expected,
            drift,
            total,
            TELEMETRY_CALIBRATION_WINDOW_MINUTES,
        )
        .await;

        if let Some(expected_rate) = expected {
            if drift.abs() >= DRIFT_THRESHOLD {
                let _ = system_log::log_event(
                    &db.pool,
                    app,
                    "warn",
                    "telemetry",
                    None,
                    None,
                    json!({
                        "event": "telemetry_calibration_drift",
                        "metric": metric,
                        "observed_rate": observed,
                        "expected_rate": expected_rate,
                        "drift": drift,
                        "sample_count": total,
                        "window_minutes": TELEMETRY_CALIBRATION_WINDOW_MINUTES,
                    }),
                )
                .await;
                upsert_kv_rate(db, metric, observed).await;
            }
        }
    }

    let _ = system_log::log_event(
        &db.pool,
        app,
        "info",
        "telemetry",
        None,
        None,
        json!({
            "event": "telemetry_calibration_run",
            "success_count": success,
            "failed_count": failed,
            "window_minutes": TELEMETRY_CALIBRATION_WINDOW_MINUTES,
        }),
    )
    .await;

    Ok(())
}
