use sqlx::sqlite::SqlitePoolOptions;

use symbiote_lib::core::telemetry_calibration::run_telemetry_calibration;
use symbiote_lib::db::Db;

async fn setup_db() -> Db {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .expect("pool");
    let db = Db { pool };
    db.init().await.expect("init");
    db
}

#[tokio::test]
async fn telemetry_calibration_writes_rows() {
    let db = setup_db().await;
    let pool = &db.pool;

    sqlx::query(
        "INSERT INTO tool_dispatches (action_id, run_id, tool_name, args_json, status, attempts, created_at, updated_at)
         VALUES ('a1', 'run1', 'get_current_time', '{}', 'success', 1, datetime('now','-5 minutes'), datetime('now','-5 minutes'))",
    )
    .execute(pool)
    .await
    .expect("insert success");

    sqlx::query(
        "INSERT INTO tool_dispatches (action_id, run_id, tool_name, args_json, status, attempts, created_at, updated_at)
         VALUES ('a2', 'run1', 'web_lookup', '{}', 'failed', 1, datetime('now','-4 minutes'), datetime('now','-4 minutes'))",
    )
    .execute(pool)
    .await
    .expect("insert failed");

    sqlx::query(
        "INSERT INTO kv_store (key, value, updated_at) VALUES ('telemetry.tool_success_rate', '0.0', CURRENT_TIMESTAMP)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
    )
    .execute(pool)
    .await
    .expect("seed success rate");

    run_telemetry_calibration(&db, None)
        .await
        .expect("calibration run");

    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM telemetry_calibrations",
    )
    .fetch_one(pool)
    .await
    .unwrap_or(0);

    assert!(count >= 1);
}

