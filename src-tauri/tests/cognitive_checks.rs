
use chrono::Utc;
use sqlx::sqlite::SqlitePoolOptions;
use symbiote_lib::core::attention_model::{AttentionModel, AttentionReason};
use symbiote_lib::core::attention_schema;
use symbiote_lib::core::cognitive_checks::run_cognitive_checks;
use symbiote_lib::core::kernel::{CandidateKind, is_state_change_candidate};
use symbiote_lib::core::kernel::{
    allowed_prediction_horizons,
    allowed_prediction_metrics,
    record_prediction_rejection,
    validate_prediction_fields,
};
use symbiote_lib::core::memory::dsl::{FactStmt, Ref};
use symbiote_lib::core::memory::types::{Scope, SourceType};
use symbiote_lib::core::memory::writer::{write_fact, WriteContext, WriteResult};
use symbiote_lib::core::model_client::repair_prediction_json;
use symbiote_lib::core::qualia::QualiaState;
use symbiote_lib::core::self_claims::{record_self_claim, SelfClaimInput};
use symbiote_lib::core::self_memory::bridge::bridge_identity_evidence_events;
use symbiote_lib::core::workspace::build_workspace_state;
use symbiote_lib::db::Db;
use tokio::time::{timeout, Duration};
use uuid::Uuid;

const TEST_TIMEOUT_SECS: u64 = 20;

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

async fn enable_episodic(db: &Db) {
    sqlx::query("UPDATE settings SET episodic_enabled = 1 WHERE id = 1")
        .execute(&db.pool)
        .await
        .expect("enable episodic settings");

    for subsystem in ["episodic", "memory_write"] {
        sqlx::query(
            "INSERT OR REPLACE INTO system_controls (subsystem_id, mode, updated_at)
             VALUES (?, 'normal', CURRENT_TIMESTAMP)",
        )
        .bind(subsystem)
        .execute(&db.pool)
        .await
        .expect("enable system control");
    }
}

async fn seed_identity_evidence(db: &Db) -> i64 {
    enable_episodic(db).await;

    let subject_id: i64 = match sqlx::query_scalar(
        "SELECT id FROM ics_entities WHERE label = 'User' LIMIT 1",
    )
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    {
        Some(id) => id,
        None => sqlx::query_scalar(
            "INSERT INTO ics_entities (label, label_canonical, entity_type, aliases, aliases_canonical, keys, resolution_state)
             VALUES ('User', 'user', 'Person', '[]', '[]', '[]', 'normal') RETURNING id",
        )
        .fetch_one(&db.pool)
        .await
        .expect("insert entity"),
    };

    let ctx = WriteContext {
        pool: db.pool.clone(),
        model_client: None,
        scope: Scope::Session,
        source: SourceType::User,
        source_ref: Some("user".to_string()),
        now: Utc::now(),
        embedding_config: None,
        conversation_id: Some("default".to_string()),
    };

    let stmt = FactStmt {
        subject: Ref::Handle("$user".to_string()),
        key: "identity.role".to_string(),
        value: "I am a test agent".to_string(),
        value_quoted: false,
        certainty: None,
        time_expr: None,
        scope_expr: None,
        source_ref: None,
        polarity: "assert".to_string(),
    };

    let result = write_fact(stmt, subject_id, &ctx).await;
    assert!(matches!(result, WriteResult::Inserted(_) | WriteResult::Updated(_)));

    sqlx::query_scalar(
        "SELECT id FROM ics_evidence_events WHERE source_type = 'user' ORDER BY id DESC LIMIT 1",
    )
    .fetch_one(&db.pool)
    .await
    .expect("fetch evidence id")
}

#[tokio::test]
async fn cognitive_checks_fail_on_orphaned_evidence() {
    timeout(Duration::from_secs(TEST_TIMEOUT_SECS), async {
        let db = setup_db().await;

        sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, created_at)
             VALUES (9999, 'user', 'msg', 'orphan', 1.0, CURRENT_TIMESTAMP)"
        )
        .execute(&db.pool)
        .await
        .expect("insert orphan evidence");

        let results = run_cognitive_checks(&db, "default").await;
        let evidence_check = results
            .iter()
            .find(|r| r.name == "evidence_integrity")
            .expect("missing evidence_integrity check");

        assert!(!evidence_check.passed, "evidence_integrity should fail on orphaned evidence");
    })
    .await
    .expect("cognitive_checks test timed out");
}

#[test]
fn prediction_json_repair_extracts_object() {
    let raw = r#"noise {"predictions": null, "rejection_reason": "no_evidence"} trailing"#;
    let (value_opt, repaired) = repair_prediction_json(raw);
    assert!(repaired);
    let value = value_opt.expect("value");
    assert!(value.get("predictions").is_some());
}

#[test]
fn prediction_validation_rejects_invalid_metric() {
    let allowed_metrics = allowed_prediction_metrics();
    let allowed_horizons = allowed_prediction_horizons();
    let rejection = validate_prediction_fields(
        "invalid_metric",
        0.5,
        Some(0.1),
        "next_turn",
        &allowed_metrics,
        &allowed_horizons,
    );
    assert_eq!(rejection.as_deref(), Some("invalid_metric"));
}

#[tokio::test]
async fn prediction_rejection_logging_inserts_row() {
    let db = setup_db().await;
    record_prediction_rejection(
        &db.pool,
        Some("run_1"),
        Some("trace_1"),
        "{}",
        "invalid_metric",
        Some("invalid_metric"),
    )
    .await;

    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM self_predictions WHERE rejection_reason = 'invalid_metric'",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);

    assert_eq!(count, 1);
}

#[tokio::test]
async fn attention_schema_computes_fields() {
    let db = setup_db().await;
    let kernel_state = symbiote_lib::core::kernel::KernelState::default_for("default");
    let workspace = build_workspace_state(&kernel_state, None);
    let now = Utc::now().to_rfc3339();
    let attention = AttentionModel {
        timestamp: now.clone(),
        current_focus_refs: vec!["focus".to_string()],
        why_focused: vec![AttentionReason {
            source: "workspace_broadcast".to_string(),
            weight: 0.8,
        }],
        meta_confidence: 0.8,
        next_focus_prediction: Some("focus".to_string()),
    };
    let qualia = QualiaState {
        timestamp: now,
        dominant_tag: None,
        dominant_intensity: 0.0,
        recent_labels: Vec::new(),
        last_reward: None,
        predicted_tag: None,
        prediction_confidence: 0.0,
        matched_workspace_refs: Vec::new(),
    };

    let schema = attention_schema::compute_attention_schema(
        &db,
        &workspace,
        &attention,
        &qualia,
        None,
    )
    .await
    .expect("compute schema");

    assert!((0.0..=1.0).contains(&schema.capacity_usage));
    assert!(!schema.selection_policy.is_empty());
    assert!(!schema.last_updated_at.is_empty());
}

#[test]
fn state_change_candidate_classification() {
    assert!(is_state_change_candidate(&CandidateKind::UpdateWorkspace));
    assert!(is_state_change_candidate(&CandidateKind::RecordSelfClaim));
    assert!(is_state_change_candidate(&CandidateKind::ToolCall));
    assert!(!is_state_change_candidate(&CandidateKind::EmitMessage));
    assert!(!is_state_change_candidate(&CandidateKind::AskUserQuestion));
    assert!(!is_state_change_candidate(&CandidateKind::NoOp));
}

#[tokio::test]
async fn episodic_events_create_evidence_and_episodic() {
    let db = setup_db().await;
    let evidence_id = seed_identity_evidence(&db).await;

    let episodic_event_id: Option<String> = sqlx::query_scalar(
        "SELECT episodic_event_id FROM ics_evidence_events WHERE id = ?",
    )
    .bind(evidence_id)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();

    assert!(episodic_event_id.as_deref().unwrap_or("").len() > 0);

    let exists: Option<String> = sqlx::query_scalar(
        "SELECT id FROM episodic_events WHERE id = ?",
    )
    .bind(episodic_event_id.unwrap())
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();

    assert!(exists.is_some());
}

#[tokio::test]
async fn evidence_bridges_to_self() {
    let db = setup_db().await;
    let evidence_id = seed_identity_evidence(&db).await;

    let report = bridge_identity_evidence_events(&db.pool, 10)
        .await
        .expect("bridge identity evidence");
    assert!(report.processed > 0 || report.skipped > 0);

    let source_ids: Option<String> = sqlx::query_scalar(
        "SELECT source_evidence_ids FROM self_evidence_events ORDER BY id DESC LIMIT 1",
    )
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let source_ids = source_ids.unwrap_or_else(|| "[]".to_string());
    let parsed: Vec<i64> = serde_json::from_str(&source_ids).unwrap_or_default();

    assert!(parsed.contains(&evidence_id));
}

#[tokio::test]
async fn self_evidence_supports_self_claim() {
    let db = setup_db().await;
    seed_gate_allow(&db).await;

    let _ = seed_identity_evidence(&db).await;
    let _ = bridge_identity_evidence_events(&db.pool, 10)
        .await
        .expect("bridge identity evidence");

    let self_evidence_id: i64 = sqlx::query_scalar(
        "SELECT id FROM self_evidence_events ORDER BY id DESC LIMIT 1",
    )
    .fetch_one(&db.pool)
    .await
    .expect("self evidence id");

    let input = SelfClaimInput {
        claim_text: "I am a test agent".to_string(),
        claim_key: "identity.role".to_string(),
        evidence_event_ids: vec![self_evidence_id],
        belief_ids: Vec::new(),
        confidence: 0.7,
        polarity: "assert".to_string(),
        source_run_id: None,
        conversation_id: Some("default".to_string()),
        provisional: false,
        source_type: Some("system_state".to_string()),
        requires_validation: false,
        ttl_seconds: None,
        promotion_rule: None,
        eviction_rule: None,
    };

    let claim_id = record_self_claim(&db, input)
        .await
        .expect("record self claim")
        .expect("claim stored");

    let stored: Option<String> = sqlx::query_scalar(
        "SELECT id FROM self_claims WHERE id = ?",
    )
    .bind(&claim_id)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();

    assert!(stored.is_some());
}
