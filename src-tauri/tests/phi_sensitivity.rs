use chrono::Utc;
use sqlx::sqlite::SqlitePoolOptions;
use std::sync::Arc;

use symbiote_lib::core::memory::dsl::{FactStmt, Ref};
use symbiote_lib::core::memory::types::{Scope, SourceType};
use symbiote_lib::core::memory::writer::{write_fact, WriteContext, WriteResult};
use symbiote_lib::core::rolling_summary::archive_rolling_summary;
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
async fn phi_fact_write_blocked_without_consent() {
    let db = setup_db().await;

    let subject_id: i64 = sqlx::query_scalar(
        "SELECT entity_id FROM ics_session_bindings
         WHERE session_id = 'default' AND ref_text = 'user' LIMIT 1",
    )
    .fetch_one(&db.pool)
    .await
    .expect("missing user entity");

    let ctx = WriteContext {
        pool: db.pool.clone(),
        model_client: None,
        scope: Scope::Session,
        source: SourceType::User,
        source_ref: None,
        now: Utc::now(),
        embedding_config: None,
        conversation_id: Some("default".to_string()),
    };

    let stmt = FactStmt {
        subject: Ref::Handle("user".to_string()),
        key: "email".to_string(),
        value: "user@example.com".to_string(),
        value_quoted: true,
        certainty: Some(0.9),
        time_expr: None,
        scope_expr: None,
        source_ref: None,
        polarity: "assert".to_string(),
    };

    let result = write_fact(stmt, subject_id, &ctx).await;
    assert!(matches!(result, WriteResult::Ignored(reason) if reason == "phi_blocked"));

    let blocked: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs WHERE json_extract(payload, '$.event') = 'phi_write_blocked'",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);

    assert!(blocked > 0);
}

#[tokio::test]
async fn summary_archive_redacts_phi_without_consent() {
    let db = setup_db().await;
    let pool = db.pool.clone();

    sqlx::query(
        "INSERT INTO messages (message_id, conversation_id, role, content, status, created_at)
         VALUES ('m1', 'default', 'user', 'hello', 'complete', datetime('now','-1 minutes'))",
    )
    .execute(&pool)
    .await
    .expect("insert message");

    sqlx::query(
        "INSERT INTO conversation_summaries (conversation_id, summary, updated_at, version)
         VALUES ('default', 'Call me at 555-123-4567', datetime('now'), 1)",
    )
    .execute(&pool)
    .await
    .expect("insert summary");

    let _ = archive_rolling_summary(Arc::new(db), "default", "test", "summary_archive")
        .await
        .expect("archive summary");

    let redacted: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs WHERE json_extract(payload, '$.event') = 'phi_redacted'",
    )
    .fetch_one(&pool)
    .await
    .unwrap_or(0);

    assert!(redacted > 0);
}
