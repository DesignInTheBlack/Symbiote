use symbiote_lib::core::cognitive_checks::run_cognitive_checks;
use symbiote_lib::db::Db;
use sqlx::sqlite::SqlitePoolOptions;
use std::fs;
use std::path::PathBuf;

async fn setup_pool() -> sqlx::SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .expect("pool");
    let schema_path = PathBuf::from("src/db/schema.sql");
    let schema_sql = fs::read_to_string(&schema_path).expect("schema");
    sqlx::query(&schema_sql).execute(&pool).await.expect("apply schema");
    pool
}

#[tokio::test]
async fn cognitive_checks_flag_orphaned_evidence() {
    let pool = setup_pool().await;
    sqlx::query("INSERT INTO ics_evidence_events (belief_id, evidence_type, source_type, weight, created_at) VALUES (9999, 'test', 'system', 1.0, CURRENT_TIMESTAMP)")
        .execute(&pool)
        .await
        .expect("insert orphan evidence");

    let db = Db { pool };
    let results = run_cognitive_checks(&db, "default").await;
    let evidence_check = results
        .iter()
        .find(|r| r.name == "evidence_integrity")
        .expect("evidence check");
    assert!(!evidence_check.passed);
}
