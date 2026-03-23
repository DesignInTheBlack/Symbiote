
use chrono::{Duration as ChronoDuration, Utc};
use serde_json::json;
use sqlx::{sqlite::SqlitePoolOptions, Row};
use uuid::Uuid;

use symbiote_lib::core::self_claims::{record_self_claim, SelfClaimInput, evidence_is_stale, stale_evidence_ttl_seconds};
use symbiote_lib::core::self_memory::config::SELF_EVIDENCE_STALE_AFTER_HOURS;
use symbiote_lib::core::self_reflection;
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

async fn seed_gate_allow(db: &Db) -> String {
    let snapshot_hash = Uuid::new_v4().to_string();
    sqlx::query(
        "INSERT INTO subject_snapshots (snapshot_hash, snapshot_version, tick_id, conversation_id, run_id, timestamp, subject_state_json)
         VALUES (?, 'v1', 'tick_1', 'default', 'run_1', CURRENT_TIMESTAMP, ?)",
    )
    .bind(&snapshot_hash)
    .bind("{}")
    .execute(&db.pool)
    .await
    .expect("insert subject snapshot");

    let proposal_id = Uuid::new_v4().to_string();
    sqlx::query(
        "INSERT INTO action_proposals
         (proposal_id, snapshot_hash, intent, steps_json, risk_level, required_claims_json, required_error_bounds_json, verification_plan_json, success_criteria_json, created_at)
         VALUES (?, ?, 'test', '[]', 'low', '[]', NULL, NULL, '[]', CURRENT_TIMESTAMP)",
    )
    .bind(&proposal_id)
    .bind(&snapshot_hash)
    .execute(&db.pool)
    .await
    .expect("insert action proposal");

    let decision_id = Uuid::new_v4().to_string();
    sqlx::query(
        "INSERT INTO gate_decisions (decision_id, proposal_id, snapshot_hash, decision, evidence_refs_json, metrics_json, created_at)
         VALUES (?, ?, ?, 'ALLOW', '[]', '{}', CURRENT_TIMESTAMP)",
    )
    .bind(&decision_id)
    .bind(&proposal_id)
    .bind(&snapshot_hash)
    .execute(&db.pool)
    .await
    .expect("insert gate decision");

    snapshot_hash
}

async fn seed_stale_evidence(db: &Db) -> i64 {
    let evidence_id = db
        .create_system_evidence_event(
            "default",
            "identity.thread",
            "stale evidence",
            Some("test"),
            "stale evidence",
        )
        .await
        .expect("evidence event");

    let stale_at = (Utc::now() - ChronoDuration::hours(SELF_EVIDENCE_STALE_AFTER_HOURS + 1))
        .to_rfc3339();
    sqlx::query("UPDATE ics_evidence_events SET created_at = ? WHERE id = ?")
        .bind(&stale_at)
        .bind(evidence_id)
        .execute(&db.pool)
        .await
        .expect("update stale evidence");

    evidence_id
}

#[tokio::test]
async fn apply_reflection_staged_updates_self_model() {
    let db = setup_db().await;

    let evidence_id = db
        .create_system_evidence_event(
            "default",
            "reflection_test",
            "identity_update",
            Some("test"),
            "identity evidence",
        )
        .await
        .expect("evidence event");

    let proposal = json!({
        "persona_delta": null,
        "persona_reason": null,
        "persona_observed_at": null,
        "persona_evidence_event_ids": null,
        "goals": null,
        "goals_reason": null,
        "goals_observed_at": null,
        "goals_evidence_event_ids": null,
        "identity_thread": "Test identity thread.",
        "identity_confidence": 0.6,
        "identity_uncertainty_note": null,
        "identity_evidence_event_ids": [evidence_id],
        "self_memory_writes": null,
        "rejection_reason": null
    })
    .to_string();

    let stage_id = db
        .insert_reflection_staging(&proposal, &vec![evidence_id])
        .await
        .expect("staging insert");

    db.update_reflection_staging_status(&stage_id, "approved", Some("test"))
        .await
        .expect("staging approve");

    self_reflection::apply_reflection_staged(&db, &stage_id, Some("test"))
        .await
        .expect("apply reflection");

    let last_reflection: Option<String> =
        sqlx::query_scalar("SELECT last_reflection_at FROM self_model LIMIT 1")
            .fetch_optional(&db.pool)
            .await
            .ok()
            .flatten();

    assert!(last_reflection.is_some());
}

#[tokio::test]
async fn self_reflection_missing_evidence_is_skipped() {
    let db = setup_db().await;

    let proposal = json!({
        "persona_delta": null,
        "persona_reason": null,
        "persona_observed_at": null,
        "persona_evidence_event_ids": null,
        "goals": null,
        "goals_reason": null,
        "goals_observed_at": null,
        "goals_evidence_event_ids": null,
        "identity_thread": "Missing evidence.",
        "identity_confidence": 0.4,
        "identity_uncertainty_note": null,
        "identity_evidence_event_ids": [],
        "self_memory_writes": null,
        "rejection_reason": null
    })
    .to_string();

    let stage_id = db
        .insert_reflection_staging(&proposal, &vec![])
        .await
        .expect("staging insert");

    db.update_reflection_staging_status(&stage_id, "approved", Some("test"))
        .await
        .expect("staging approve");

    self_reflection::apply_reflection_staged(&db, &stage_id, Some("test"))
        .await
        .expect("apply reflection");

    let last_reflection: Option<String> =
        sqlx::query_scalar("SELECT last_reflection_at FROM self_model LIMIT 1")
            .fetch_optional(&db.pool)
            .await
            .ok()
            .flatten();

    assert!(last_reflection.is_none());
}

#[tokio::test]
async fn self_claim_rejects_missing_evidence() {
    let db = setup_db().await;
    seed_gate_allow(&db).await;

    let input = SelfClaimInput {
        claim_text: "I am helpful".to_string(),
        claim_key: "identity.helpful".to_string(),
        evidence_event_ids: Vec::new(),
        belief_ids: Vec::new(),
        confidence: 0.7,
        polarity: "assert".to_string(),
        source_run_id: Some("run_1".to_string()),
        conversation_id: Some("default".to_string()),
        provisional: false,
        source_type: Some("system_state".to_string()),
        requires_validation: false,
        ttl_seconds: None,
        promotion_rule: None,
        eviction_rule: None,
    };

    let result = record_self_claim(&db, input)
        .await
        .expect("record self claim");
    assert!(result.is_none());

    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM self_claims")
        .fetch_one(&db.pool)
        .await
        .unwrap_or(0);
    assert_eq!(count, 0);
}

#[tokio::test]
async fn stale_evidence_forces_provisional_self_claim() {
    let db = setup_db().await;
    seed_gate_allow(&db).await;
    let evidence_id = seed_stale_evidence(&db).await;

    let stale_at = evidence_is_stale(&db, &[evidence_id]).await;
    assert!(stale_at.is_some());

    let input = SelfClaimInput {
        claim_text: "I can help".to_string(),
        claim_key: "capability.help".to_string(),
        evidence_event_ids: vec![evidence_id],
        belief_ids: Vec::new(),
        confidence: 0.6,
        polarity: "assert".to_string(),
        source_run_id: Some("run_1".to_string()),
        conversation_id: Some("default".to_string()),
        provisional: true,
        source_type: Some("system_state".to_string()),
        requires_validation: true,
        ttl_seconds: Some(stale_evidence_ttl_seconds()),
        promotion_rule: None,
        eviction_rule: None,
    };

    let _ = record_self_claim(&db, input)
        .await
        .expect("record self claim");

    let row = sqlx::query(
        "SELECT provisional, requires_validation, ttl_seconds FROM self_claims ORDER BY created_at DESC LIMIT 1",
    )
    .fetch_one(&db.pool)
    .await
    .expect("self claim row");

    let provisional: i64 = row.get("provisional");
    let requires_validation: i64 = row.get("requires_validation");
    let ttl_seconds: Option<i64> = row.try_get("ttl_seconds").ok();

    assert_eq!(provisional, 1);
    assert_eq!(requires_validation, 1);
    assert!(ttl_seconds.unwrap_or(0) > 0);
}
