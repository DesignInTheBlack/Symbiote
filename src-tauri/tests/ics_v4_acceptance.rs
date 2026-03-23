use symbiote_lib::core::memory::api::MemoryApi;
use symbiote_lib::core::memory::claims;
use symbiote_lib::core::memory::types::{Scope, SourceType, QueryIntent};
use symbiote_lib::core::episodic;
use symbiote_lib::core::kernel::sanitize_user_output;
use symbiote_lib::core::cognitive_checks::run_cognitive_checks;
use symbiote_lib::db::Db;
use serde_json::json;
use sqlx::{sqlite::SqlitePoolOptions, SqlitePool, Row};
use std::path::PathBuf;
use std::fs;
use chrono::Utc;

async fn setup_pool() -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory pool");

    let schema_path = PathBuf::from("src/db/schema.sql");
    let schema_sql = fs::read_to_string(&schema_path).expect("Failed to read schema.sql");
    sqlx::query(&schema_sql).execute(&pool).await.expect("Failed to apply schema");

    sqlx::query("CREATE VIRTUAL TABLE IF NOT EXISTS ics_entities_fts USING fts5(label, aliases, entity_id UNINDEXED)")
        .execute(&pool)
        .await
        .expect("Failed fts entities");
    sqlx::query("CREATE TRIGGER IF NOT EXISTS ics_ent_ai AFTER INSERT ON ics_entities BEGIN INSERT INTO ics_entities_fts(rowid, label, aliases, entity_id) VALUES (new.rowid, new.label, new.aliases, new.id); END")
        .execute(&pool)
        .await
        .expect("Failed trigger entities");
    sqlx::query("CREATE TRIGGER IF NOT EXISTS ics_ent_ad AFTER DELETE ON ics_entities BEGIN DELETE FROM ics_entities_fts WHERE rowid = old.rowid; END")
        .execute(&pool)
        .await
        .expect("Failed trigger entities delete");
    sqlx::query("CREATE TRIGGER IF NOT EXISTS ics_ent_au AFTER UPDATE ON ics_entities BEGIN DELETE FROM ics_entities_fts WHERE rowid = old.rowid; INSERT INTO ics_entities_fts(rowid, label, aliases, entity_id) VALUES (new.rowid, new.label, new.aliases, new.id); END")
        .execute(&pool)
        .await
        .expect("Failed trigger entities update");

    sqlx::query("CREATE VIRTUAL TABLE IF NOT EXISTS ics_facts_fts USING fts5(key, value_literal, content='ics_fact_beliefs', content_rowid='belief_id')")
        .execute(&pool)
        .await
        .expect("Failed fts facts");
    sqlx::query("CREATE TRIGGER IF NOT EXISTS ics_fact_ai AFTER INSERT ON ics_fact_beliefs BEGIN INSERT INTO ics_facts_fts(rowid, key, value_literal) VALUES (new.belief_id, new.key, new.value_literal); END")
        .execute(&pool)
        .await
        .expect("Failed trigger facts");
    sqlx::query("CREATE TRIGGER IF NOT EXISTS ics_fact_ad AFTER DELETE ON ics_fact_beliefs BEGIN INSERT INTO ics_facts_fts(ics_facts_fts, rowid, key, value_literal) VALUES('delete', old.belief_id, old.key, old.value_literal); END")
        .execute(&pool)
        .await
        .expect("Failed trigger facts delete");
    sqlx::query("CREATE TRIGGER IF NOT EXISTS ics_fact_au AFTER UPDATE ON ics_fact_beliefs BEGIN INSERT INTO ics_facts_fts(ics_facts_fts, rowid, key, value_literal) VALUES('delete', old.belief_id, old.key, old.value_literal); INSERT INTO ics_facts_fts(rowid, key, value_literal) VALUES (new.belief_id, new.key, new.value_literal); END")
        .execute(&pool)
        .await
        .expect("Failed trigger facts update");

    sqlx::query("CREATE VIRTUAL TABLE IF NOT EXISTS ics_rel_fts USING fts5(rel_type, roles)")
        .execute(&pool)
        .await
        .expect("Failed fts rels");

    sqlx::query("INSERT INTO settings (id, schema_version, api_base_url) VALUES (1, 1, 'http://localhost:11434/v1')")
        .execute(&pool)
        .await
        .expect("Failed to seed settings");

    let shapes = vec![
        ("writes", "[\"subject\",\"object\"]", 0),
        ("father_of", "[\"father\",\"child\"]", 0),
        ("works_with", "[\"person\"]", 1),
    ];
    for (rel_type, roles_json, commutative) in shapes {
        sqlx::query(
            "INSERT INTO ics_relation_shapes (rel_type, roles, anchor_roles, commutative, created_at)
             VALUES (?, ?, '[]', ?, CURRENT_TIMESTAMP)"
        )
        .bind(rel_type)
        .bind(roles_json)
        .bind(commutative)
        .execute(&pool)
        .await
        .expect("Failed to seed relation shapes");
    }

    pool
}

#[test]
fn test_output_sanitizer_strips_scaffold() {
    let input = "Next Steps\n<<<BEGIN_SECTION:Next Steps>>>\n1. Do X\n<<<END_SECTION:Next Steps>>>\nProposed Response\n<<<BEGIN_SECTION:Proposed Response>>>\nHello\n<<<END_SECTION:Proposed Response>>>\n";
    let (cleaned, changed, _) = sanitize_user_output(input, false, None, None);
    assert!(changed, "Expected sanitizer to mark changes");
    let lower = cleaned.to_lowercase();
    assert!(!lower.contains("<<<begin_section"), "Scaffold markers should be removed");
    assert!(!lower.contains("next steps"), "Planning headers should be removed");
    assert!(!lower.contains("proposed response"), "Planning headers should be removed");
}

#[tokio::test]
async fn test_cognitive_checks_scaffold_leakage_detects_markers() {
    let pool = setup_pool().await;
    let conversation_id = "conv_scaffold";
    sqlx::query(
        "INSERT INTO conversations (conversation_id, schema_version, created_at, updated_at)
         VALUES (?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
    )
    .bind(conversation_id)
    .execute(&pool)
    .await
    .expect("insert conversation");

    sqlx::query(
        "INSERT INTO messages (message_id, conversation_id, role, content, status, created_at)
         VALUES (?, ?, 'assistant', ?, 'complete', CURRENT_TIMESTAMP)",
    )
    .bind("msg_scaffold")
    .bind(conversation_id)
    .bind("<<<BEGIN_SECTION:Next Steps>>>Do X<<<END_SECTION:Next Steps>>>")
    .execute(&pool)
    .await
    .expect("insert message");

    let db = Db { pool: pool.clone() };
    let results = run_cognitive_checks(&db, conversation_id).await;
    let scaffold_check = results
        .iter()
        .find(|r| r.name == "output_scaffold_leakage")
        .expect("scaffold check present");
    assert!(!scaffold_check.passed, "Expected scaffold leakage to be detected");
}

#[tokio::test]
async fn test_cognitive_checks_numeric_grounding_flags_invalid() {
    let pool = setup_pool().await;
    let conversation_id = "conv_numeric";
    sqlx::query(
        "INSERT INTO conversations (conversation_id, schema_version, created_at, updated_at)
         VALUES (?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
    )
    .bind(conversation_id)
    .execute(&pool)
    .await
    .expect("insert conversation");

    sqlx::query(
        "INSERT INTO messages (message_id, conversation_id, role, content, status, created_at)
         VALUES (?, ?, 'user', ?, 'complete', CURRENT_TIMESTAMP)",
    )
    .bind("msg_user")
    .bind(conversation_id)
    .bind("Please check telemetry.")
    .execute(&pool)
    .await
    .expect("insert user message");

    let entity_row = sqlx::query(
        "INSERT INTO ics_entities (label, label_canonical)
         VALUES ('telemetry', 'telemetry')
         RETURNING id",
    )
    .fetch_one(&pool)
    .await
    .expect("insert entity");
    let entity_id: i64 = entity_row.get("id");

    let belief_row = sqlx::query(
        "INSERT INTO ics_beliefs (kind, scope, polarity, topic_key, signature_hash, evidence_weight_total, confidence, time_bucket_kind)
         VALUES ('fact', 'global', 'assert', 'telemetry', 'sig', 1.0, 1.0, 'atemporal')
         RETURNING id",
    )
    .fetch_one(&pool)
    .await
    .expect("insert belief");
    let belief_id: i64 = belief_row.get("id");

    sqlx::query(
        "INSERT INTO ics_fact_beliefs (belief_id, subject_entity_id, key, value_literal, value_hash)
         VALUES (?, ?, 'telemetry.memory_pass_rate', '0.0', 'hash')",
    )
    .bind(belief_id)
    .bind(entity_id)
    .execute(&pool)
    .await
    .expect("insert telemetry fact");

    sqlx::query(
        "INSERT INTO inner_monologue_entries (id, conversation_id, mode, stream_type, thought, created_at)
         VALUES (?, ?, 'work', 'DS', ?, CURRENT_TIMESTAMP)",
    )
    .bind("monologue1")
    .bind(conversation_id)
    .bind("Telemetry shows memory pass rate 0.8 right now.")
    .execute(&pool)
    .await
    .expect("insert monologue");

    let db = Db { pool: pool.clone() };
    let results = run_cognitive_checks(&db, conversation_id).await;
    let numeric_check = results
        .iter()
        .find(|r| r.name == "monologue_numeric_grounding")
        .expect("numeric grounding check present");
    assert!(!numeric_check.passed, "Expected numeric grounding to fail for invalid numbers");
}

#[tokio::test]
async fn test_cognitive_checks_fts_visibility_passes() {
    let pool = setup_pool().await;
    let conversation_id = "conv_fts";
    sqlx::query(
        "INSERT INTO conversations (conversation_id, schema_version, created_at, updated_at)
         VALUES (?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
    )
    .bind(conversation_id)
    .execute(&pool)
    .await
    .expect("insert conversation");

    sqlx::query(
        "INSERT INTO inner_monologue_entries (id, conversation_id, mode, stream_type, thought, created_at)
         VALUES (?, ?, 'work', 'FTS', ?, CURRENT_TIMESTAMP)",
    )
    .bind("fts_entry")
    .bind(conversation_id)
    .bind("Free thought checkpoint.")
    .execute(&pool)
    .await
    .expect("insert fts entry");

    let db = Db { pool: pool.clone() };
    let results = run_cognitive_checks(&db, conversation_id).await;
    let fts_check = results
        .iter()
        .find(|r| r.name == "fts_visibility")
        .expect("fts visibility check present");
    assert!(fts_check.passed, "Expected FTS visibility to pass when entries exist");
}

#[tokio::test]
async fn test_monologue_multi_turn_entries_persist() {
    let pool = setup_pool().await;
    let conversation_id = "conv_monologue_multi";
    sqlx::query(
        "INSERT INTO conversations (conversation_id, schema_version, created_at, updated_at)
         VALUES (?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
    )
    .bind(conversation_id)
    .execute(&pool)
    .await
    .expect("insert conversation");

    let ds_dialogue = "dialogue_ds";
    for turn in 1..=2 {
        sqlx::query(
            "INSERT INTO inner_monologue_entries (id, conversation_id, dialogue_id, turn_index, speaker, mode, stream_type, thought, created_at)
             VALUES (?, ?, ?, ?, 'self_a', 'work', 'DS', ?, CURRENT_TIMESTAMP)",
        )
        .bind(format!("ds_turn_{}", turn))
        .bind(conversation_id)
        .bind(ds_dialogue)
        .bind(turn)
        .bind(format!("DS turn {}", turn))
        .execute(&pool)
        .await
        .expect("insert DS monologue turn");
    }

    let fts_dialogue = "dialogue_fts";
    for turn in 1..=2 {
        sqlx::query(
            "INSERT INTO inner_monologue_entries (id, conversation_id, dialogue_id, turn_index, speaker, mode, stream_type, thought, created_at)
             VALUES (?, ?, ?, ?, 'self_b', 'work', 'FTS', ?, CURRENT_TIMESTAMP)",
        )
        .bind(format!("fts_turn_{}", turn))
        .bind(conversation_id)
        .bind(fts_dialogue)
        .bind(turn)
        .bind(format!("FTS turn {}", turn))
        .execute(&pool)
        .await
        .expect("insert FTS monologue turn");
    }

    let ds_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM inner_monologue_entries WHERE conversation_id = ? AND stream_type = 'DS' AND dialogue_id = ?",
    )
    .bind(conversation_id)
    .bind(ds_dialogue)
    .fetch_one(&pool)
    .await
    .expect("count DS turns");
    assert_eq!(ds_count, 2, "Expected two DS turns for the same dialogue");

    let fts_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM inner_monologue_entries WHERE conversation_id = ? AND stream_type = 'FTS' AND dialogue_id = ?",
    )
    .bind(conversation_id)
    .bind(fts_dialogue)
    .fetch_one(&pool)
    .await
    .expect("count FTS turns");
    assert_eq!(fts_count, 2, "Expected two FTS turns for the same dialogue");

    let db = Db { pool: pool.clone() };
    let results = run_cognitive_checks(&db, conversation_id).await;
    let fts_check = results
        .iter()
        .find(|r| r.name == "fts_visibility")
        .expect("fts visibility check present");
    assert!(fts_check.passed, "Expected FTS visibility to pass for multi-turn entries");
}

#[tokio::test]
async fn test_ics_v4_end_to_end() {
    // 1. Setup
    let pool = setup_pool().await;

    // Initialize API
    let api = MemoryApi::new(pool.clone(), None, "test_session".to_string()).await;

    // 2. Write Data (Phase 2)
    let dsl_input = r#"
    $user:name = 'Desig' ~1.0
    #Alice:role = 'Assistant'
    #Alice:location = 'Server'
    writes(subject: $user, object: #Alice) ~0.9
    father_of(father: #Bob -> child: $user)
    father_of(father: #Carl -> child: #Hannah)
    "#;

    println!("Compiling memory block...");
    let compile_res = api.parse_and_compile(
        dsl_input, 
        Scope::Global, 
        SourceType::User, 
        None,
        Utc::now()
    ).await;

    assert!(compile_res.errors.is_empty(), "Compilation errors: {:?}", compile_res.errors);
    assert!(compile_res.written_ids.len() >= 3, "Expected at least 3 writes (2 facts + 1 rel + maybe more)");
    println!("Compilation success. Written IDs: {:?}", compile_res.written_ids);

    // 3. Retrieval (Phase 3)
    println!("Retrieving 'Alice'...");
    let packet = api.retrieve(
        "Alice", 
        &[Scope::Global], 
        QueryIntent::AskCurrent
    ).await.expect("Retrieval failed");

    println!("Packet Facts: {:?}", packet.facts);
    println!("Packet Rels: {:?}", packet.relations);

    // Verify Facts
    let has_role = packet.facts.iter().any(|f| f.key == "role" && f.value == "Assistant");
    assert!(has_role, "Failed to retrieve Alice's role fact");

    // Verify Rel
    let has_rel = packet.relations.iter().any(|r| r.rel_type == "writes");
    assert!(has_rel, "Failed to retrieve 'writes' relation");

    // Relation-only query with $user
    let father_packet = api.retrieve(
        "who is my father",
        &[Scope::Global],
        QueryIntent::AskCurrent
    ).await.expect("Retrieval failed");
    let has_father_rel = father_packet.relations.iter().any(|r| r.rel_type == "father_of");
    assert!(has_father_rel, "Failed to retrieve father_of relation for $user");

    // Relation-only query with named entity
    let hannah_packet = api.retrieve(
        "who is Hannah's father",
        &[Scope::Global],
        QueryIntent::AskCurrent
    ).await.expect("Retrieval failed");
    let has_hannah_father = hannah_packet.relations.iter().any(|r| r.rel_type == "father_of");
    assert!(has_hannah_father, "Failed to retrieve father_of relation for Hannah");

    // Verify Scoring (Basic)
    let role_fact = packet.facts.iter().find(|f| f.key == "role").unwrap();
    assert!(role_fact.score > 0.0, "Score should be positive");
    
    // 4. Attention/Working Set (Phase 4)
    // Check if Alice is in working set
    let working_row = sqlx::query("SELECT activation FROM ics_working_set WHERE item_type = 'entity' AND activation > 0")
        .fetch_optional(&pool)
        .await
        .unwrap();
    assert!(working_row.is_some(), "Working set should contain activated entity");

    // 5. Consolidation (Phase 5)
    // Create an alias via evidence? (Hard to simulate in integration without lower level access)
    // Let's just run consolidate and ensure it doesn't crash
    println!("Running consolidation...");
    let consol_res = api.consolidate().await.expect("Consolidation failed");
    println!("Consolidation result: {:?}", consol_res);
    
    println!("Test Complete: Success");
}

#[tokio::test]
async fn test_fact_dedupe_is_entity_scoped() {
    let pool = setup_pool().await;
    let api = MemoryApi::new(pool.clone(), None, "test_session".to_string()).await;

    let dsl_input = r#"
    #Alice:favorite_color = 'blue'
    #Bob:favorite_color = 'blue'
    "#;

    let compile_res = api.parse_and_compile(
        dsl_input,
        Scope::Global,
        SourceType::User,
        None,
        Utc::now(),
    ).await;

    assert!(compile_res.errors.is_empty(), "Compilation errors: {:?}", compile_res.errors);

    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ics_fact_beliefs WHERE key = ? AND value_literal = ?"
    )
    .bind("favorite_color")
    .bind("blue")
    .fetch_one(&pool)
    .await
    .expect("Failed to count fact beliefs");

    assert_eq!(count, 2, "Expected two distinct fact beliefs for separate entities");
}

#[tokio::test]
async fn test_provenance_queries() {
    let pool = setup_pool().await;
    let api = MemoryApi::new(pool.clone(), None, "test_session".to_string()).await;

    let dsl_input = r#"
    #Alice:role = 'Assistant'
    works_with(person: #Alice, person: #Bob)
    "#;

    let compile_res = api.parse_and_compile(
        dsl_input,
        Scope::Global,
        SourceType::User,
        None,
        Utc::now(),
    ).await;

    assert!(compile_res.errors.is_empty(), "Compilation errors: {:?}", compile_res.errors);

    let belief_id: i64 = sqlx::query_scalar(
        "SELECT belief_id FROM ics_fact_beliefs WHERE key = 'role' AND value_literal = 'Assistant' LIMIT 1"
    )
    .fetch_one(&pool)
    .await
    .expect("Failed to fetch belief_id");

    let entity_id: i64 = sqlx::query_scalar(
        "SELECT id FROM ics_entities WHERE label = 'Alice' LIMIT 1"
    )
    .fetch_one(&pool)
    .await
    .expect("Failed to fetch entity_id");

    let db = Db { pool: pool.clone() };
    let belief_events = db
        .list_episodic_events_for_ics_belief(belief_id, 10)
        .await
        .expect("Failed to fetch belief provenance");
    assert!(!belief_events.is_empty(), "Expected episodic events for belief provenance");

    let entity_events = db
        .list_episodic_events_for_entity(entity_id, 10)
        .await
        .expect("Failed to fetch entity provenance");
    assert!(!entity_events.is_empty(), "Expected episodic events for entity provenance");
}

#[tokio::test]
async fn test_claim_gating_and_promotion() {
    let pool = setup_pool().await;
    sqlx::query("UPDATE settings SET memory_claims_enabled = 1 WHERE id = 1")
        .execute(&pool)
        .await
        .expect("Failed to enable memory claims");

    let api = MemoryApi::new(pool.clone(), None, "test_session".to_string()).await;
    let dsl_input = r#"#Alice:favorite_color = 'blue'"#;
    let compile_res = api
        .parse_and_compile(dsl_input, Scope::Global, SourceType::User, None, Utc::now())
        .await;

    assert!(compile_res.errors.is_empty(), "Compilation errors: {:?}", compile_res.errors);
    assert!(compile_res.written_ids.is_empty(), "Expected claims-only path with no direct writes");
    assert_eq!(compile_res.claim_ids.len(), 1, "Expected a single claim_id");

    let claim_id = compile_res.claim_ids[0].clone();
    let processed = claims::evaluate_pending_claims(&pool, None, 10)
        .await
        .expect("Claim evaluation failed");
    assert_eq!(processed, 1, "Expected the pending claim to be evaluated");

    let status: Option<String> = sqlx::query_scalar("SELECT status FROM memory_claims WHERE id = ?")
        .bind(&claim_id)
        .fetch_optional(&pool)
        .await
        .expect("Failed to fetch claim status");
    assert_eq!(status.as_deref(), Some("promoted"));

    let belief_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ics_beliefs")
        .fetch_one(&pool)
        .await
        .expect("Failed to count beliefs");
    assert!(belief_count >= 1, "Expected at least one belief after promotion");
}

#[tokio::test]
async fn test_rel_claims_failopen_rel_type_id() {
    let pool = setup_pool().await;
    sqlx::query("UPDATE settings SET memory_claims_enabled = 1 WHERE id = 1")
        .execute(&pool)
        .await
        .expect("Failed to enable memory claims");

    let api = MemoryApi::new(pool.clone(), None, "test_session".to_string()).await;
    let dsl_input = r#"mother(child: $user -> mother: #Mom)"#;
    let compile_res = api
        .parse_and_compile(dsl_input, Scope::Global, SourceType::User, None, Utc::now())
        .await;

    assert!(compile_res.errors.is_empty(), "Compilation errors: {:?}", compile_res.errors);
    assert_eq!(compile_res.claim_ids.len(), 1, "Expected a single relation claim");

    let processed = claims::evaluate_pending_claims(&pool, None, 10)
        .await
        .expect("Claim evaluation failed");
    assert_eq!(processed, 1, "Expected the pending relation claim to be evaluated");

    let rel_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ics_rel_beliefs")
        .fetch_one(&pool)
        .await
        .expect("Failed to count relation beliefs");
    assert!(rel_count >= 1, "Expected at least one relation belief after promotion");

    let rel_type_id: Option<String> = sqlx::query_scalar("SELECT rel_type_id FROM ics_rel_beliefs LIMIT 1")
        .fetch_optional(&pool)
        .await
        .expect("Failed to fetch rel_type_id");
    assert!(rel_type_id.unwrap_or_default().trim().len() > 0, "Expected rel_type_id to be populated");
}

#[tokio::test]
async fn test_episodic_event_sanitizes_snippets() {
    let pool = setup_pool().await;
    sqlx::query("UPDATE settings SET episodic_enabled = 1 WHERE id = 1")
        .execute(&pool)
        .await
        .expect("Failed to enable episodic");

    let raw_snippet = "<MEMORY_CONTEXT>keep this</MEMORY_CONTEXT>\n#Alice:age = \"30\"\n<<MEMORY>>";
    let event_id = episodic::emit_episodic_event(
        &pool,
        "memory_write_fact",
        json!({ "summary_snippet": raw_snippet }),
        None,
        None,
        Some("test_session"),
        None,
        "user",
        None,
        None,
        None,
    )
    .await
    .expect("Failed to emit episodic event");

    let payload_raw: String = sqlx::query_scalar("SELECT payload_json FROM episodic_events WHERE id = ?")
        .bind(&event_id)
        .fetch_one(&pool)
        .await
        .expect("Failed to fetch episodic payload");
    let payload: serde_json::Value =
        serde_json::from_str(&payload_raw).expect("Failed to parse episodic payload");
    let snippet = payload
        .get("summary_snippet")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    assert_eq!(snippet, "keep this");
}

#[tokio::test]
async fn test_memory_write_snippets_are_natural_language() {
    let pool = setup_pool().await;
    sqlx::query("UPDATE settings SET episodic_enabled = 1, memory_claims_enabled = 0 WHERE id = 1")
        .execute(&pool)
        .await
        .expect("Failed to update settings");

    let api = MemoryApi::new(pool.clone(), None, "test_session".to_string()).await;
    let dsl_input = r#"
    #Alice:role = 'Assistant'
    works_with(person: #Alice, person: #Bob)
    "#;
    let compile_res = api
        .parse_and_compile(dsl_input, Scope::Global, SourceType::User, None, Utc::now())
        .await;
    assert!(compile_res.errors.is_empty(), "Compilation errors: {:?}", compile_res.errors);

    let fact_payload: String = sqlx::query_scalar(
        "SELECT payload_json FROM episodic_events
         WHERE event_type = 'memory_write_fact'
         ORDER BY rowid DESC
         LIMIT 1",
    )
    .fetch_one(&pool)
    .await
    .expect("Failed to fetch fact episodic payload");
    let fact_json: serde_json::Value =
        serde_json::from_str(&fact_payload).expect("Failed to parse fact payload");
    let fact_snippet = fact_json
        .get("summary_snippet")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    assert!(fact_snippet.starts_with("Fact"), "Expected fact snippet to be natural language");
    assert!(!fact_snippet.contains("#"), "Fact snippet should not contain DSL refs");
    assert!(!fact_snippet.contains(" = "), "Fact snippet should not contain DSL equals");
    assert!(!fact_snippet.to_lowercase().contains("<memory"), "Fact snippet should not contain memory markers");

    let rel_payload: String = sqlx::query_scalar(
        "SELECT payload_json FROM episodic_events
         WHERE event_type = 'memory_write_rel'
         ORDER BY rowid DESC
         LIMIT 1",
    )
    .fetch_one(&pool)
    .await
    .expect("Failed to fetch rel episodic payload");
    let rel_json: serde_json::Value =
        serde_json::from_str(&rel_payload).expect("Failed to parse rel payload");
    let rel_snippet = rel_json
        .get("summary_snippet")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    assert!(rel_snippet.starts_with("Relation"), "Expected relation snippet to be natural language");
    assert!(!rel_snippet.contains("("), "Relation snippet should not contain DSL parentheses");
    assert!(!rel_snippet.contains("->"), "Relation snippet should not contain DSL arrows");
    assert!(!rel_snippet.to_lowercase().contains("<memory"), "Relation snippet should not contain memory markers");
}
