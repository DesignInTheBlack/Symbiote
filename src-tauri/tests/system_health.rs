use symbiote_lib::core::system_health::HealthAggregator;
use symbiote_lib::core::system_log;
use symbiote_lib::db::Db;
use chrono::Utc;
use serde_json::json;
use sqlx::sqlite::SqlitePoolOptions;
use std::sync::Arc;
use uuid::Uuid;

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
async fn health_snapshot_includes_gate_inputs_and_organism() {
    let db = setup_db().await;

    let subject_state = json!({
        "state": {
            "self_model": {
                "controller_state": {
                    "confidence": 0.7,
                    "uncertainty": 0.3,
                    "failure_streak": 1,
                    "verification_needed": false,
                    "reanchor_needed": false
                }
            },
            "organism": {
                "stress": 0.2,
                "fatigue": 0.3,
                "social_alignment": 0.6
            },
            "error_state": { "open_error_count": 0 }
        }
    });

    let snapshot_hash = Uuid::new_v4().to_string();
    sqlx::query(
        "INSERT INTO subject_snapshots (snapshot_hash, snapshot_version, tick_id, conversation_id, run_id, timestamp, subject_state_json)
         VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)"
    )
    .bind(&snapshot_hash)
    .bind("v1")
    .bind("tick_1")
    .bind("default")
    .bind("run_1")
    .bind(subject_state.to_string())
    .execute(&db.pool)
    .await
    .expect("Failed to insert subject snapshot");

    system_log::log_event(
        &db.pool,
        None,
        "info",
        "kernel",
        None,
        None,
        json!({
            "event": "gate_decision_inputs",
            "signals": { "organism_social_alignment_low": 1 },
            "enforced_decision": "ALLOW"
        })
    )
    .await
    .expect("Failed to log gate inputs");

    let aggregator = HealthAggregator::new(Arc::new(db));
    let snapshot = aggregator
        .capture_snapshot(None, None, None)
        .await
        .expect("Failed to capture snapshot");

    let gate_inputs_len = snapshot
        .metrics
        .get("gate")
        .and_then(|g| g.get("inputs"))
        .and_then(|v| v.as_array())
        .map(|arr| arr.len())
        .unwrap_or(0);

    assert!(gate_inputs_len > 0);

    let fatigue = snapshot
        .metrics
        .get("organism")
        .and_then(|o| o.get("fatigue"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    assert!((fatigue - 0.3).abs() < 0.01);
}

#[tokio::test]
async fn workspace_snapshot_logged_in_health_metrics() {
    let db = setup_db().await;

    system_log::log_event(
        &db.pool,
        None,
        "info",
        "kernel",
        None,
        None,
        json!({
            "event": "workspace_snapshot",
            "contributors": {
                "kernel": {
                    "cycle_id": "tick_1",
                    "broadcast_refs": [],
                    "ignition_active": false,
                    "timestamp": Utc::now().to_rfc3339()
                },
                "missing": []
            }
        }),
    )
    .await
    .expect("log workspace snapshot");

    let aggregator = HealthAggregator::new(Arc::new(db));
    let snapshot = aggregator
        .capture_snapshot(None, None, None)
        .await
        .expect("Failed to capture snapshot");

    let snapshots = snapshot
        .metrics
        .get("workspace_contributors")
        .and_then(|v| v.get("snapshots"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    assert!(snapshots > 0);
}

#[tokio::test]
async fn combined_score_null_when_no_outcomes() {
    let db = setup_db().await;
    let aggregator = HealthAggregator::new(Arc::new(db));
    let snapshot = aggregator
        .capture_snapshot(None, None, None)
        .await
        .expect("capture snapshot");
    let combined = snapshot
        .metrics
        .get("scorecard")
        .and_then(|v| v.get("combined_score"));
    assert!(combined.is_none() || combined.unwrap().is_null());
}
