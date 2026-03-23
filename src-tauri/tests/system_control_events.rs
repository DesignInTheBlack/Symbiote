use symbiote_lib::core::system_log;
use symbiote_lib::db::Db;
use serde_json::json;
use sqlx::sqlite::SqlitePoolOptions;

async fn setup_db() -> Db {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory pool");
    let db = Db { pool };
    db.init().await.expect("Failed to init db");
    db
}

#[tokio::test]
async fn control_change_writes_event_and_log() {
    let db = setup_db().await;

    let _entry = db
        .set_system_control(
            "tool_execution",
            "off",
            None,
            Some("test".to_string()),
            Some("testing".to_string()),
        )
        .await
        .expect("Failed to set control");

    let _telemetry = db
        .set_system_control(
            "telemetry_sampling",
            "off",
            None,
            Some("test".to_string()),
            Some("disable telemetry".to_string()),
        )
        .await
        .expect("Failed to set telemetry control");

    let event_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM system_control_events")
        .fetch_one(&db.pool)
        .await
        .unwrap_or(0);

    assert!(event_count > 0);

    system_log::log_event(
        &db.pool,
        None,
        "info",
        "system",
        None,
        None,
        json!({
            "event": "system_control_changed",
            "subsystem_id": "tool_execution",
            "mode": "off",
        }),
    )
    .await
    .expect("Failed to log control change");

    let log_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs WHERE json_extract(payload, '$.event') = 'system_control_changed'"
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);

    assert!(log_count > 0);
}
