use chrono::Utc;
use serde_json::json;
use sqlx::sqlite::SqlitePoolOptions;
use std::sync::Arc;
use uuid::Uuid;

use symbiote_lib::core::kernel::MONOLOGUE_STATE_CHANGE_WINDOW_TICKS;
use symbiote_lib::core::memory::compiler::{self, CompileContext};
use symbiote_lib::core::memory::types::{Scope, SourceType};
use symbiote_lib::core::qualia::maybe_auto_label_for_message;
use symbiote_lib::core::rolling_summary::archive_rolling_summary;
use symbiote_lib::core::system_health::HealthAggregator;
use symbiote_lib::db::Db;

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
async fn summary_chunk_created() {
    let db = setup_db().await;
    let pool = db.pool.clone();

    sqlx::query(
        "INSERT INTO messages (message_id, conversation_id, role, content, status, created_at)
         VALUES ('m1', 'default', 'user', 'hello', 'complete', datetime('now','-2 minutes'))",
    )
    .execute(&pool)
    .await
    .expect("insert message");

    sqlx::query(
        "INSERT INTO conversation_summaries (conversation_id, summary, updated_at, version)
         VALUES ('default', 'User greeted the system.', datetime('now'), 1)",
    )
    .execute(&pool)
    .await
    .expect("insert summary");

    let _ = archive_rolling_summary(Arc::new(db), "default", "test", "summary_archive_turn_count")
        .await
        .expect("archive summary");

    let chunk_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM conversation_summary_chunks WHERE conversation_id = 'default'",
    )
    .fetch_one(&pool)
    .await
    .unwrap_or(0);

    assert!(chunk_count > 0);
}

#[tokio::test]
async fn relation_write_creates_belief() {
    let db = setup_db().await;
    let ctx = CompileContext {
        pool: db.pool.clone(),
        model_client: None,
        session_id: "default".to_string(),
        scope: Scope::Session,
        source: SourceType::User,
        source_ref: None,
        now: Utc::now(),
        embedding_config: None,
        skip_claims: true,
        allow_ambiguous_user_refs: true,
    };

    let dsl = "works_at(employee: #User -> employer: #Org)";
    let result = compiler::compile(dsl, ctx).await;
    assert!(result.errors.is_empty());

    let rel_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ics_rel_beliefs")
        .fetch_one(&db.pool)
        .await
        .unwrap_or(0);

    assert!(rel_count > 0);
}

#[tokio::test]
async fn qualia_auto_label_creates_label() {
    let db = setup_db().await;
    let pool = &db.pool;

    sqlx::query(
        "INSERT OR REPLACE INTO system_controls (subsystem_id, mode, updated_at)
         VALUES ('qualia_loop', 'normal', CURRENT_TIMESTAMP)",
    )
    .execute(pool)
    .await
    .expect("enable qualia_loop");
    sqlx::query(
        "INSERT OR REPLACE INTO system_controls (subsystem_id, mode, updated_at)
         VALUES ('qualia_auto', 'normal', CURRENT_TIMESTAMP)",
    )
    .execute(pool)
    .await
    .expect("enable qualia_auto");

    let snapshot_hash = Uuid::new_v4().to_string();
    sqlx::query(
        "INSERT INTO subject_snapshots (snapshot_hash, snapshot_version, tick_id, conversation_id, run_id, timestamp, subject_state_json)
         VALUES (?, 'v1', 'tick_1', 'default', NULL, CURRENT_TIMESTAMP, '{}')",
    )
    .bind(&snapshot_hash)
    .execute(pool)
    .await
    .expect("insert snapshot");

    sqlx::query(
        "INSERT INTO messages (message_id, conversation_id, role, content, status, created_at)
         VALUES ('m2', 'default', 'assistant', 'ready', 'complete', CURRENT_TIMESTAMP)",
    )
    .execute(pool)
    .await
    .expect("insert assistant message");

    let label_id = maybe_auto_label_for_message(&db, None, "default", "m2", Some("run_1"), false)
        .await
        .expect("auto label")
        .expect("label created");

    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM qualia_labels WHERE label_id = ?",
    )
    .bind(&label_id)
    .fetch_one(pool)
    .await
    .unwrap_or(0);

    assert_eq!(count, 1);
}

#[tokio::test]
async fn monologue_loop_closure_has_state_change_within_window() {
    let db = setup_db().await;
    let pool = &db.pool;

    for idx in 0..MONOLOGUE_STATE_CHANGE_WINDOW_TICKS {
        let state_change = if idx == MONOLOGUE_STATE_CHANGE_WINDOW_TICKS - 1 {
            1
        } else {
            0
        };
        let payload = json!({
            "event": "monologue_loop_outcome",
            "state_change_candidates": state_change,
            "non_state_change_candidates": 1,
            "suppressed_candidates": 0,
            "no_op_reason": if state_change == 0 { "no_state_change".to_string() } else { "".to_string() },
        });
        sqlx::query(
            "INSERT INTO system_logs (id, timestamp, level, category, run_id, trace_id, payload)
             VALUES (?, ?, 'info', 'kernel', NULL, NULL, ?)",
        )
        .bind(Uuid::new_v4().to_string())
        .bind(Utc::now().to_rfc3339())
        .bind(payload.to_string())
        .execute(pool)
        .await
        .expect("insert monologue_loop_outcome");
    }

    let delta_payload = json!({
        "event": "loop_delta_applied",
        "candidate_kind": "record_self_claim",
    });
    sqlx::query(
        "INSERT INTO system_logs (id, timestamp, level, category, run_id, trace_id, payload)
         VALUES (?, ?, 'info', 'kernel', NULL, NULL, ?)",
    )
    .bind(Uuid::new_v4().to_string())
    .bind(Utc::now().to_rfc3339())
    .bind(delta_payload.to_string())
    .execute(pool)
    .await
    .expect("insert loop_delta_applied");

    let aggregator = HealthAggregator::new(Arc::new(db));
    let snapshot = aggregator
        .capture_snapshot(None, None, None)
        .await
        .expect("capture snapshot");
    let loop_rate = snapshot
        .metrics
        .get("monologue")
        .and_then(|m| m.get("loop_state_change_rate"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    assert!(loop_rate > 0.0);
}
