use chrono::{Duration as ChronoDuration, Utc};
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::Row;
use symbiote_lib::core::memory::validation::{validate_memory_beliefs, MemoryValidationConfig};
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
async fn outcome_disconfirm_reduces_memory_confidence() {
    let db = setup_db().await;
    let pool = &db.pool;

    let scope_str = serde_json::to_string(&symbiote_lib::core::memory::types::Scope::Session).unwrap();
    let row = sqlx::query(
        "INSERT INTO ics_beliefs (kind, scope, polarity, layer, topic_key, signature_hash, evidence_weight_total, confidence, time_bucket_kind)
         VALUES ('fact', ?, 'assert', 'episodic', 'topic', 'sig', 1.0, 0.9, 'atemporal')
         RETURNING id",
    )
    .bind(&scope_str)
    .fetch_one(pool)
    .await
    .expect("insert belief");
    let belief_id: i64 = row.get("id");

    let row = sqlx::query(
        "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, created_at)
         VALUES (?, 'user', 'test', 'evidence', 1.0, CURRENT_TIMESTAMP)
         RETURNING id",
    )
    .bind(belief_id)
    .fetch_one(pool)
    .await
    .expect("insert evidence");
    let evidence_id: i64 = row.get("id");

    let before: f64 = sqlx::query_scalar("SELECT confidence FROM ics_beliefs WHERE id = ?")
        .bind(belief_id)
        .fetch_one(pool)
        .await
        .unwrap_or(0.0);

    db.record_outcome_event(
        Some("run_1"),
        None,
        Some("msg_1"),
        "message",
        "disconfirm",
        0.8,
        "test",
        None,
        &[evidence_id],
    )
    .await
    .expect("record outcome");

    let after: f64 = sqlx::query_scalar("SELECT confidence FROM ics_beliefs WHERE id = ?")
        .bind(belief_id)
        .fetch_one(pool)
        .await
        .unwrap_or(0.0);

    assert!(after < before, "expected confidence to drop after disconfirm");
}

#[tokio::test]
async fn memory_validation_decays_confidence_over_time() {
    let db = setup_db().await;
    let pool = &db.pool;

    let scope_str = serde_json::to_string(&symbiote_lib::core::memory::types::Scope::Session).unwrap();
    let old_ts = (Utc::now() - ChronoDuration::days(30)).to_rfc3339();
    let row = sqlx::query(
        "INSERT INTO ics_beliefs (kind, scope, polarity, layer, topic_key, signature_hash, evidence_weight_total, confidence, time_bucket_kind, last_evidence_at)
         VALUES ('fact', ?, 'assert', 'episodic', 'topic', 'sig_old', 1.0, 0.9, 'atemporal', ?)
         RETURNING id",
    )
    .bind(&scope_str)
    .bind(&old_ts)
    .fetch_one(pool)
    .await
    .expect("insert belief");
    let belief_id: i64 = row.get("id");

    let mut config = MemoryValidationConfig::default();
    config.max_beliefs = 10;
    config.min_interval_minutes = 0;
    config.decay_per_day = 0.98;
    config.drift_threshold = 0.01;

    let (ics, _) = validate_memory_beliefs(&db.pool, &config, None)
        .await
        .expect("validation run");

    let after: f64 = sqlx::query_scalar("SELECT confidence FROM ics_beliefs WHERE id = ?")
        .bind(belief_id)
        .fetch_one(pool)
        .await
        .unwrap_or(0.0);

    assert!(after < 0.9, "expected confidence decay after validation");
    assert!(ics.drift_events >= 0, "validation result should return drift metrics");
}
