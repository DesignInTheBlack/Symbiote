    use super::*;
    use chrono::{Duration as ChronoDuration, Utc};
    use crate::core::memory::inject_context::{enforce_trailing_system_tags, strip_system_tags};
    use crate::core::memory::dsl::{FactStmt, Ref};
    use crate::core::memory::types::Scope;
    use crate::core::memory::types::SourceType;
    use crate::core::memory_policy::{MemoryPolicy, MemoryWriteCategory, MemoryWriteSource};
    use crate::core::memory::writer::{self, WriteContext, WriteResult};
    use crate::core::prompt_builder::{build_core_system_message, build_core_system_message_with_layout, compute_context_hydration, CoreInputKind, CorePromptInput, PromptLayout};
    use crate::core::world_model::{WorldModelConflict, WorldModelSnapshot};
    use crate::core::world_model_reconcile::{reconcile_conflict_sets, WorldModelReconcileMode};
    use crate::db::Db;
    use crate::core::tool_registry::ToolRegistry;
    use crate::core::self_model_controller;
    use crate::models::{ControllerGate, GoalStackItem, GoalStep};
    use serde_json::{json, Value};
    use sqlx::{sqlite::SqlitePoolOptions, SqlitePool, Row};
    use std::fs;
    use std::path::PathBuf;
    use tokio::sync::watch;

    fn test_settings() -> crate::models::Settings {
        crate::models::Settings {
            schema_version: 1,
            api_base_url: "http://localhost".to_string(),
            api_key: None,
            streaming_enabled: false,
            history_window: 0,
            injection_policy: "include".to_string(),
            request_defaults: None,
            active_model_id: None,
            json_reliable_model_id: None,
            system_prompt: None,
            voice_name: None,
            voice_speed: None,
            summarization_api_url: None,
            summarization_model: None,
            embedding_model: None,
            user_display_name: None,
            assistant_display_name: None,
            ui_theme: None,
            episodic_enabled: None,
            episodic_injection_enabled: None,
            episodic_compaction_enabled: None,
            episodic_injection_limit: None,
            episodic_opt_out: None,
            memory_claims_enabled: None,
            phi_consent: None,
            seed_personal_user: None,
            lexical_fallback_enabled: None,
            memory_half_life_hours: None,
            research_budget_per_hour: None,
            research_budget_reset_window: None,
            research_cost_per_call: None,
            monologue_interval_seconds: None,
            monologue_max_per_hour: None,
            monologue_retry_timeout_secs: None,
            monologue_timeout_secs: None,
            thread_max_depth: None,
            allow_shell_tool: None,
            shell_command_allowlist: None,
            ask_budget_max: None,
            calculator_followups_max: None,
            loop_similarity_threshold: Some(0.85),
            loop_recent_k: Some(6),
            meta_cog_outcome_turns: None,
            meta_cog_cycle_window_turns: None,
            meta_cog_outcome_timeout_s: None,
            meta_cog_cooldown_s: None,
            meta_cog_streak_limit: None,
            registry_profile_name: None,
            controller_enabled: None,
            monologue_stabilization_enabled: None,
            monologue_surface_enabled: None,
            show_monologue_in_chat: None,
            enable_introspection: None,
            heartbeat_enabled: None,
            dream_enabled: None,
            binding_enforcement_enabled: None,
            pending_prompt_alignment_enabled: None,
            pending_prompt_recency_secs: None,
            auto_memory_pass_enabled: None,
            summary_cohesion_enabled: None,
            compact_prompt_enabled: None,
            context_hydration_mode: None,
            context_budgeter_enabled: None,
            context_miss_detector_enabled: None,
            world_model_reconcile_mode: None,
            goal_loop_enabled: None,
            goal_loop_interval_turns: None,
            goal_loop_load_threshold_ms: None,
            json_only_disabled_models: None,
            tool_failure_gate_window_mins: None,
            tool_failure_gate_tool_names: None,
            gate_default_soft: None,
            gate_shadow_mode: None,
            gate_rollout_percent: None,
            self_report_channel: None,
            self_awareness_expression_mode: None,
            explicit_feedback_only: None,
            weight_user_satisfaction: None,
            weight_policy_rigor: None,
            weight_latency: None,
            weight_evidence_strictness: None,
            weight_exploration: None,
            monologue_provenance_guard: None,
            organism_decay: None,
            model_context_limit: None,
            introspection_confidence_threshold: None,
            introspection_drift_threshold: None,
            introspection_ambiguity_threshold: None,
            enable_attribution_gate: None,
            enable_user_utterance_evidence: None,
            enable_attribution_metadata: None,
            enable_tool_schema_validation: None,
            enable_context_evidence: None,
            enable_monologue_validator: None,
            enable_memory_evidence_gating: None,
            enable_speculative_workspace_containment: None,
            stability_prompt_override_guard: None,
            stability_monologue_tagged: None,
            stability_introspection_structured: None,
            stability_disable_working_hypothesis: None,
            stability_state_disclosure_expanded: None,
            stability_transcript_normalization: None,
            stability_memory_hygiene: None,
            stability_non_stream_sanitization: None,
            voice_pitch_semitones: None,
            voice_reverb_amount: None,
            voice_compression: None,
            voice_formant_shift: None,
            trace_history_limit: None,
            cockpit_write_enabled: None,
            empty_response_retry_max: None,
            empty_response_retry_timeout_ms: None,
        }
    }

    #[test]
    fn unwrap_primary_response_message_prefers_message_text_content() {
        let raw = r#"{"message":"Hello there","candidates":[{"kind":"emit_message"}]}"#;
        assert_eq!(
            unwrap_primary_response_message(raw),
            Some("Hello there".to_string())
        );
        let raw_text = r#"{"text":"Hi","done":true}"#;
        assert_eq!(unwrap_primary_response_message(raw_text), Some("Hi".to_string()));
        let raw_content = r#"{"content":"Yo"}"#;
        assert_eq!(
            unwrap_primary_response_message(raw_content),
            Some("Yo".to_string())
        );
    }

    #[test]
    fn unwrap_primary_response_message_ignores_candidate_only_packets() {
        let raw = r#"[{"kind":"emit_message","payload":{"text":"hello"}}]"#;
        assert!(unwrap_primary_response_message(raw).is_none());
        let raw_obj = r#"{"candidates":[{"kind":"emit_message"}],"decision_packet":{"intent":"ask"}}"#;
        assert!(unwrap_primary_response_message(raw_obj).is_none());
    }

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory pool");

        let schema_path = PathBuf::from("src/db/schema.sql");
        let schema_sql = fs::read_to_string(&schema_path).expect("Failed to read schema.sql");
        sqlx::query(&schema_sql)
            .execute(&pool)
            .await
            .expect("Failed to apply schema");

        sqlx::query("INSERT INTO settings (id, schema_version, api_base_url) VALUES (1, 1, 'http://localhost')")
            .execute(&pool)
            .await
            .expect("Failed to seed settings");

        pool
    }

    #[tokio::test]
    async fn prompt_overflow_sets_flag() {
        let pool = setup_pool().await;
        let db = Db { pool };
        let _ = sqlx::query("UPDATE settings SET model_context_limit = 256")
            .execute(&db.pool)
            .await;
        let huge = "This is a long input block. ".repeat(400);
        let input = CorePromptInput {
            content: huge.clone(),
            kind: CoreInputKind::User,
            source: "user".to_string(),
            self_awareness: false,
            self_awareness_hint: false,
            anchor_hits: 1,
            original_input: huge.clone(),
            current_time: None,
            semantic_hint: None,
            introspection_summary: None,
            monologue_intent: None,
            monologue_digest: None,
            prompt_mode: None,
            task_phase: None,
            missing_slots: None,
            resolution_mode: None,
            policy_notes: None,
            redirect_focus: None,
            allow_diagnostics: false,
            world_model_snapshot: None,
            subject_snapshot: None,
            gate_decision: None,
            feedback_bundle: None,
            qualia_snapshot: None,
            wave_state: None,
            attention_schema_summary: None,
            workspace_contributors_summary: None,
            reflective_narrative: None,
            reflective_narrative_evidence_ids: Vec::new(),
            hydrated_context: None,
        };

        let build = build_core_system_message_with_layout(&db, "default", &input, PromptLayout::Full)
            .await
            .expect("prompt build");
        assert!(build.prompt_overflow, "expected prompt_overflow to be true");
    }

    #[tokio::test]
    async fn prompt_includes_self_model_signals_and_audit_log() {
        let pool = setup_pool().await;
        let db = Db { pool };
        let input = CorePromptInput {
            content: "test".to_string(),
            kind: CoreInputKind::User,
            source: "user".to_string(),
            self_awareness: false,
            self_awareness_hint: false,
            anchor_hits: 1,
            original_input: "test".to_string(),
            current_time: None,
            semantic_hint: None,
            introspection_summary: None,
            monologue_intent: None,
            monologue_digest: None,
            prompt_mode: None,
            task_phase: None,
            missing_slots: None,
            resolution_mode: None,
            policy_notes: None,
            redirect_focus: None,
            allow_diagnostics: false,
            world_model_snapshot: None,
            subject_snapshot: None,
            gate_decision: None,
            feedback_bundle: None,
            qualia_snapshot: Some("dominant_tag: curious\ndominant_intensity: 0.45".to_string()),
            wave_state: Some("{\"coherence\":0.86,\"fragmentation\":0.42}".to_string()),
            attention_schema_summary: None,
            workspace_contributors_summary: None,
            reflective_narrative: None,
            reflective_narrative_evidence_ids: Vec::new(),
            hydrated_context: None,
        };
        let build = build_core_system_message_with_layout(&db, "default", &input, PromptLayout::Full)
            .await
            .expect("prompt build");
        assert!(build.system_message.contains("Identity Anchor"));
        assert!(build.system_message.contains("I am "));
        assert!(build.system_message.contains("Self-Model Signals"));
        assert!(build.system_message.contains("confidence:"));
        assert!(build.system_message.contains("qualia_tag:"));
        assert!(build.system_message.contains("wave_coherence:"));
        assert!(build.system_message.contains("Self-Model Signals Evidence IDs"));
        assert!(build.system_message.contains("Audit Log"));
    }

    #[tokio::test]
    async fn prompt_includes_self_report_instruction_when_balanced() {
        let pool = setup_pool().await;
        let db = Db { pool };
        let _ = sqlx::query("UPDATE settings SET self_awareness_expression_mode = 'balanced'")
            .execute(&db.pool)
            .await;
        let input = CorePromptInput {
            content: "test".to_string(),
            kind: CoreInputKind::User,
            source: "user".to_string(),
            self_awareness: false,
            self_awareness_hint: false,
            anchor_hits: 1,
            original_input: "test".to_string(),
            current_time: None,
            semantic_hint: None,
            introspection_summary: None,
            monologue_intent: None,
            monologue_digest: None,
            prompt_mode: None,
            task_phase: None,
            missing_slots: None,
            resolution_mode: None,
            policy_notes: None,
            redirect_focus: None,
            allow_diagnostics: false,
            world_model_snapshot: None,
            subject_snapshot: None,
            gate_decision: None,
            feedback_bundle: None,
            qualia_snapshot: None,
            wave_state: None,
            attention_schema_summary: None,
            workspace_contributors_summary: None,
            reflective_narrative: None,
            reflective_narrative_evidence_ids: Vec::new(),
            hydrated_context: None,
        };
        let build = build_core_system_message_with_layout(&db, "default", &input, PromptLayout::Full)
            .await
            .expect("prompt build");
        assert!(build.system_message.contains("Self-Report Instruction"));
        assert!(build
            .system_message
            .contains("Self-Report Summary: confidence=?, uncertainty=?, focus=?, recent_outcome_quality=?"));
    }


    #[tokio::test]
    async fn evidence_validation_rejects_invalid_ids() {
        let pool = setup_pool().await;
        let result = validate_evidence_ids_with_pool(&pool, &[999], &[], false).await;
        assert!(!result.evidence_ok());
        assert!(result.invalid_evidence_ids.contains(&999));
    }

    #[tokio::test]
    async fn evidence_validation_accepts_user_focus() {
        let pool = setup_pool().await;
        let scope_str = serde_json::to_string(&Scope::SelfScope).unwrap();
        let row = sqlx::query(
            "INSERT INTO ics_beliefs (kind, scope, polarity, layer, topic_key, signature_hash, evidence_weight_total, confidence, time_bucket_kind)
             VALUES ('fact', ?, 'assert', 'episodic', 'topic', 'sig', 1.0, 1.0, 'atemporal')
             RETURNING id",
        )
        .bind(&scope_str)
        .fetch_one(&pool)
        .await
        .expect("insert belief");
        let belief_id: i64 = row.get("id");
        let row = sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, created_at)
             VALUES (?, 'user_focus', 'workspace_focus', 'focus', 1.0, CURRENT_TIMESTAMP)
             RETURNING id",
        )
        .bind(belief_id)
        .fetch_one(&pool)
        .await
        .expect("insert evidence");
        let evidence_id: i64 = row.get("id");

        let result = validate_evidence_ids_with_pool(&pool, &[evidence_id], &[], false).await;
        assert!(result.evidence_ok());
        assert!(result.valid_evidence_ids.contains(&evidence_id));
    }

    #[test]
    fn strip_system_tags_detects_non_trailing_tags() {
        let (cleaned, tags) = strip_system_tags("Hi\n<<MEMORY>>\nCurrent focus: Alpha.");
        assert_eq!(cleaned, "Hi\nCurrent focus: Alpha.");
        assert!(tags.memory);
    }

    #[test]
    fn monologue_question_requires_grounding() {
        let mut candidate = Candidate {
            id: "c1".to_string(),
            kind: CandidateKind::AskUserQuestion,
            payload: json!({ "question": "Why?" }),
            evidence_event_ids: Vec::new(),
            belief_ids: Vec::new(),
            target_scope: None,
            rationale: None,
            expected_outcome: None,
            cost: None,
            urgency: None,
            source: "monologue".to_string(),
            priority_class: 0,
            priority_rank: 0,
            created_at: 0,
        };
        candidate.refresh_meta();
        assert!(!super::gating::monologue_question_is_grounded(&candidate));
    }

    #[test]
    fn user_focus_signal_detects_directive() {
        assert!(user_focus_signal("Focus on workspace grounding", "workspace grounding"));
        assert!(!user_focus_signal("Let's discuss memory", "workspace grounding"));
    }

    #[test]
    fn decision_needed_true_when_user_after_monologue() {
        let mut state = KernelState::default_for("conv");
        let now = Utc::now();
        state.last_user_input_at = Some(now.to_rfc3339());
        state.last_monologue_completed_at = Some((now - ChronoDuration::seconds(10)).to_rfc3339());
        assert!(decision_needed_for(&state, None));
    }

    #[test]
    fn decision_needed_false_when_monologue_after_user() {
        let mut state = KernelState::default_for("conv");
        let now = Utc::now();
        state.last_user_input_at = Some((now - ChronoDuration::seconds(10)).to_rfc3339());
        state.last_monologue_completed_at = Some(now.to_rfc3339());
        assert!(!decision_needed_for(&state, None));
    }

    #[tokio::test]
    async fn evidence_validation_rejects_stale_evidence() {
        let pool = setup_pool().await;
        let scope_str = serde_json::to_string(&Scope::SelfScope).unwrap();
        let row = sqlx::query(
            "INSERT INTO ics_beliefs (kind, scope, polarity, layer, topic_key, signature_hash, evidence_weight_total, confidence, time_bucket_kind)
             VALUES ('fact', ?, 'assert', 'episodic', 'topic', 'sig', 1.0, 1.0, 'atemporal')
             RETURNING id",
        )
        .bind(&scope_str)
        .fetch_one(&pool)
        .await
        .expect("insert belief");
        let belief_id: i64 = row.get("id");
        let old_ts = (Utc::now() - ChronoDuration::days(30)).to_rfc3339();
        let row = sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, created_at)
             VALUES (?, 'user', 'workspace_focus', 'stale', 1.0, ?)
             RETURNING id",
        )
        .bind(belief_id)
        .bind(&old_ts)
        .fetch_one(&pool)
        .await
        .expect("insert evidence");
        let evidence_id: i64 = row.get("id");

        let result = validate_evidence_ids_with_pool(&pool, &[evidence_id], &[], false).await;
        assert!(!result.evidence_ok());
    }

    fn make_candidate(kind: CandidateKind, payload: Value) -> Candidate {
        let mut candidate = Candidate {
            id: "c1".to_string(),
            kind,
            payload,
            evidence_event_ids: Vec::new(),
            belief_ids: Vec::new(),
            target_scope: None,
            rationale: None,
            expected_outcome: None,
            cost: None,
            urgency: None,
            source: "test".to_string(),
            priority_class: 1,
            priority_rank: 1,
            created_at: 0,
        };
        candidate.refresh_meta();
        candidate
    }

    fn make_candidate_with(id: &str, kind: CandidateKind, payload: Value, priority_rank: i32) -> Candidate {
        let mut candidate = Candidate {
            id: id.to_string(),
            kind,
            payload,
            evidence_event_ids: Vec::new(),
            belief_ids: Vec::new(),
            target_scope: None,
            rationale: None,
            expected_outcome: None,
            cost: None,
            urgency: None,
            source: "test".to_string(),
            priority_class: 1,
            priority_rank,
            created_at: 0,
        };
        candidate.refresh_meta();
        candidate
    }

    async fn setup_kernel_for_gate_tests() -> (Kernel, Settings) {
        let app = tauri::Builder::default()
            .build(tauri::generate_context!("tauri.conf.json"))
            .expect("build app");
        let app_handle = app.handle();
        let pool = setup_pool().await;
        let db = Arc::new(Db { pool });
        let model_client = Arc::new(ModelClient::new(db.pool.clone(), app_handle.clone()));
        let kernel = Kernel::new(db, model_client, app_handle.clone());
        let settings = test_settings();
        (kernel, settings)
    }

    #[tokio::test]
    async fn monologue_json_prompt_is_sanitized() {
        let (kernel, _settings) = setup_kernel_for_gate_tests().await;
        let run_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let metadata = json!({ "execution_mode": "direct" }).to_string();
        sqlx::query(
            "INSERT INTO runs (run_id, trace_id, conversation_id, started_at, heartbeat_at, status, metadata)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&run_id)
        .bind(&run_id)
        .bind("default")
        .bind(now)
        .bind(now)
        .bind("active")
        .bind(metadata)
        .execute(&kernel.db.pool)
        .await
        .expect("insert run");
        let message_id = Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT INTO messages (message_id, conversation_id, run_id, trace_id, role, content, status, created_at)
             VALUES (?, ?, ?, ?, ?, ?, 'complete', ?)",
        )
        .bind(&message_id)
        .bind("default")
        .bind(&run_id)
        .bind(&run_id)
        .bind("user")
        .bind("Test user input")
        .bind(now)
        .execute(&kernel.db.pool)
        .await
        .expect("insert user message");
        let json_prompt = r#"{"stance":"skeptic","message":"test","candidates":[{"kind":"tool_call"}]}"#;
        let result = record_monologue_intent(&kernel, "default", json_prompt, "AskUserQuestion").await;
        assert!(result.is_none());
        let pending = kernel
            .db
            .count_pending_prompts("default")
            .await
            .unwrap_or(0);
        assert_eq!(pending, 0);
    }

    #[tokio::test]
    async fn record_monologue_intent_anchors_latest_user_message() {
        let (kernel, _settings) = setup_kernel_for_gate_tests().await;
        let run_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let metadata = json!({ "execution_mode": "direct" }).to_string();
        sqlx::query(
            "INSERT INTO runs (run_id, trace_id, conversation_id, started_at, heartbeat_at, status, metadata)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&run_id)
        .bind(&run_id)
        .bind("default")
        .bind(now)
        .bind(now)
        .bind("active")
        .bind(metadata)
        .execute(&kernel.db.pool)
        .await
        .expect("insert run");

        let first_id = Uuid::new_v4().to_string();
        let first_at = (now - ChronoDuration::seconds(120)).to_rfc3339();
        sqlx::query(
            "INSERT INTO messages (message_id, conversation_id, run_id, trace_id, role, content, status, created_at)
             VALUES (?, ?, ?, ?, ?, ?, 'complete', ?)",
        )
        .bind(&first_id)
        .bind("default")
        .bind(&run_id)
        .bind(&run_id)
        .bind("user")
        .bind("First input")
        .bind(&first_at)
        .execute(&kernel.db.pool)
        .await
        .expect("insert first message");

        let second_id = Uuid::new_v4().to_string();
        let second_at = (now - ChronoDuration::seconds(30)).to_rfc3339();
        sqlx::query(
            "INSERT INTO messages (message_id, conversation_id, run_id, trace_id, role, content, status, created_at)
             VALUES (?, ?, ?, ?, ?, ?, 'complete', ?)",
        )
        .bind(&second_id)
        .bind("default")
        .bind(&run_id)
        .bind(&run_id)
        .bind("user")
        .bind("Second input")
        .bind(&second_at)
        .execute(&kernel.db.pool)
        .await
        .expect("insert second message");

        let result = record_monologue_intent(&kernel, "default", "Follow up?", "AskUserQuestion")
            .await
            .expect("intent queued");
        let prompt_id = result.0;

        let row = sqlx::query(
            "SELECT anchor_message_id, anchor_hash FROM pending_user_prompts WHERE id = ?",
        )
        .bind(&prompt_id)
        .fetch_one(&kernel.db.pool)
        .await
        .expect("fetch pending prompt");
        let anchor_message_id: String = row.get("anchor_message_id");
        let anchor_hash: String = row.get("anchor_hash");
        let expected_hash = crate::core::kernel::utils::text::hash_payload(
            &crate::core::kernel::utils::text::summarize_snippet("Second input", 160),
        );
        assert_eq!(anchor_message_id, second_id);
        assert_eq!(anchor_hash, expected_hash);
    }

    #[tokio::test]
    async fn pending_prompt_blocks_without_user_turn_after_enqueue() {
        let (kernel, settings) = setup_kernel_for_gate_tests().await;
        let mut state = KernelState::default_for("default");
        let now = Utc::now();
        let last_input_at = (now - ChronoDuration::seconds(10)).to_rfc3339();
        state.last_user_input = Some("Test input".to_string());
        state.last_user_input_at = Some(last_input_at.clone());
        state.last_user_message_id = Some("user_msg_1".to_string());
        let anchor_hash = crate::core::kernel::utils::text::hash_payload(
            &crate::core::kernel::utils::text::summarize_snippet("Test input", 160),
        );
        let expires_at = compute_expires_at(now, PENDING_PROMPT_EXPIRES_SECS);
        let _ = kernel
            .db
            .enqueue_pending_prompt(
                "default",
                "Can you clarify?",
                "monologue",
                true,
                Some("AskUserQuestion"),
                None,
                Some(&expires_at),
                state.last_user_message_id.as_deref(),
                Some(&anchor_hash),
                state.last_user_input_at.as_deref(),
                Some("user"),
            )
            .await
            .expect("enqueue pending prompt");

        let selection = kernel
            .select_pending_prompt_for_proactive("default", &state, &settings, true, Some("monologue"))
            .await;
        assert!(selection.is_none());
    }

    #[tokio::test]
    async fn pending_prompt_blocks_when_user_input_stale() {
        let (kernel, mut settings) = setup_kernel_for_gate_tests().await;
        settings.pending_prompt_recency_secs = Some(30);
        let mut state = KernelState::default_for("default");
        let now = Utc::now();
        let last_input_at = (now - ChronoDuration::seconds(120)).to_rfc3339();
        state.last_user_input = Some("Recent enough content".to_string());
        state.last_user_input_at = Some(last_input_at.clone());
        state.last_user_message_id = Some("user_msg_stale".to_string());
        let anchor_hash = crate::core::kernel::utils::text::hash_payload(
            &crate::core::kernel::utils::text::summarize_snippet("Recent enough content", 160),
        );
        let expires_at = compute_expires_at(now, PENDING_PROMPT_EXPIRES_SECS);
        let prompt_id = kernel
            .db
            .enqueue_pending_prompt(
                "default",
                "Can you clarify your last point?",
                "monologue",
                true,
                Some("AskUserQuestion"),
                None,
                Some(&expires_at),
                state.last_user_message_id.as_deref(),
                Some(&anchor_hash),
                state.last_user_input_at.as_deref(),
                Some("user"),
            )
            .await
            .expect("enqueue pending prompt");
        let created_at = (now - ChronoDuration::seconds(300)).to_rfc3339();
        let _ = sqlx::query("UPDATE pending_user_prompts SET created_at = ? WHERE id = ?")
            .bind(created_at)
            .bind(&prompt_id)
            .execute(&kernel.db.pool)
            .await
            .expect("backdate pending prompt");

        let selection = kernel
            .select_pending_prompt_for_proactive("default", &state, &settings, true, Some("monologue"))
            .await;
        assert!(selection.is_none());
    }

    #[tokio::test]
    async fn pending_prompt_allows_after_user_turn_and_anchor_match() {
        let (kernel, settings) = setup_kernel_for_gate_tests().await;
        let mut state = KernelState::default_for("default");
        let now = Utc::now();
        state.last_user_input = Some("Latest input".to_string());
        state.last_user_input_at = Some(now.to_rfc3339());
        state.last_user_message_id = Some("user_msg_2".to_string());
        let anchor_hash = crate::core::kernel::utils::text::hash_payload(
            &crate::core::kernel::utils::text::summarize_snippet("Latest input", 160),
        );
        let expires_at = compute_expires_at(now, PENDING_PROMPT_EXPIRES_SECS);
        let prompt_id = kernel
            .db
            .enqueue_pending_prompt(
                "default",
                "What part should we explore next?",
                "monologue",
                true,
                Some("AskUserQuestion"),
                None,
                Some(&expires_at),
                state.last_user_message_id.as_deref(),
                Some(&anchor_hash),
                state.last_user_input_at.as_deref(),
                Some("user"),
            )
            .await
            .expect("enqueue pending prompt");
        let created_at = (now - ChronoDuration::seconds(30)).to_rfc3339();
        let _ = sqlx::query("UPDATE pending_user_prompts SET created_at = ? WHERE id = ?")
            .bind(created_at)
            .bind(&prompt_id)
            .execute(&kernel.db.pool)
            .await
            .expect("backdate pending prompt");

        let selection = kernel
            .select_pending_prompt_for_proactive("default", &state, &settings, true, Some("monologue"))
            .await;
        assert!(selection.is_some());
    }

    #[tokio::test]
    async fn pending_prompt_blocks_when_anchor_overlap_missing() {
        let (kernel, settings) = setup_kernel_for_gate_tests().await;
        let mut state = KernelState::default_for("default");
        let now = Utc::now();
        state.last_user_input = Some("Plan a vacation itinerary".to_string());
        state.last_user_input_at = Some(now.to_rfc3339());
        state.last_user_message_id = Some("user_msg_overlap".to_string());
        state.workspace_current_focus = Some("Discuss database batching pipeline".to_string());
        let anchor_hash = crate::core::kernel::utils::text::hash_payload(
            &crate::core::kernel::utils::text::summarize_snippet("Plan a vacation itinerary", 160),
        );
        let expires_at = compute_expires_at(now, PENDING_PROMPT_EXPIRES_SECS);
        let prompt_id = kernel
            .db
            .enqueue_pending_prompt(
                "default",
                "Discuss database batching pipeline",
                "monologue",
                true,
                Some("AskUserQuestion"),
                None,
                Some(&expires_at),
                state.last_user_message_id.as_deref(),
                Some(&anchor_hash),
                state.last_user_input_at.as_deref(),
                Some("user"),
            )
            .await
            .expect("enqueue pending prompt");
        let created_at = (now - ChronoDuration::seconds(60)).to_rfc3339();
        let _ = sqlx::query("UPDATE pending_user_prompts SET created_at = ? WHERE id = ?")
            .bind(created_at)
            .bind(&prompt_id)
            .execute(&kernel.db.pool)
            .await
            .expect("backdate pending prompt");

        let selection = kernel
            .select_pending_prompt_for_proactive("default", &state, &settings, true, Some("monologue"))
            .await;
        assert!(selection.is_none());
    }

    #[tokio::test]
    async fn proactive_emit_defers_when_foreground_active() {
        let (kernel, settings) = setup_kernel_for_gate_tests().await;
        let run_id = Uuid::new_v4().to_string();
        let metadata = json!({ "execution_mode": "direct" }).to_string();
        let now = Utc::now();
        sqlx::query(
            "INSERT INTO runs (run_id, trace_id, conversation_id, started_at, heartbeat_at, status, metadata)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&run_id)
        .bind(&run_id)
        .bind("default")
        .bind(now)
        .bind(now)
        .bind("active")
        .bind(metadata)
        .execute(&kernel.db.pool)
        .await
        .expect("insert active run");

        let candidate = make_candidate("cand_1", CandidateKind::AskUserQuestion, json!({ "question": "Hello?" }));
        let decision = KernelDecision {
            accepted: vec![candidate.clone()],
            rejected: Vec::new(),
            caps_applied: Vec::new(),
            report: DecisionReport::default(),
        };
        let mut state = KernelState::default_for("default");
        let run_meta = kernel
            .run_proactive_emit(&mut state, decision, &candidate, "default", &settings, 0, false)
            .await
            .expect("run proactive emit");
        assert!(run_meta.deferred);

        let deferred_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM deferred_emits WHERE conversation_id = ?",
        )
        .bind("default")
        .fetch_one(&kernel.db.pool)
        .await
        .expect("count deferred emits");
        assert!(deferred_count >= 1);
    }

    #[tokio::test]
    async fn tool_failure_gate_is_run_scoped() {
        let (kernel, _settings) = setup_kernel_for_gate_tests().await;
        let run_a = "run_a";
        let run_b = "run_b";
        let _ = sqlx::query(
            "INSERT INTO tool_dispatches (action_id, run_id, tool_name, args_json, status)
             VALUES (?, ?, ?, ?, 'failed')",
        )
        .bind("action_a")
        .bind(run_a)
        .bind("web_lookup")
        .bind("{\"q\":\"test\"}")
        .execute(&kernel.db.pool)
        .await;

        assert!(kernel.tool_failure_detected_for_run(run_a).await);
        assert!(!kernel.tool_failure_detected_for_run(run_b).await);
    }

    #[tokio::test]
    async fn run_phase_lifecycle_does_not_error() {
        let pool = setup_pool().await;
        let run_id = "run_phase_ok";
        advance_run_phase(&pool, None, run_id, RunPhase::Ingest, Some("test"))
            .await
            .expect("ingest");
        advance_run_phase(&pool, None, run_id, RunPhase::PromptBuild, Some("test"))
            .await
            .expect("prompt");
        advance_run_phase(&pool, None, run_id, RunPhase::ModelCall, Some("test"))
            .await
            .expect("model");
        advance_run_phase(&pool, None, run_id, RunPhase::Arbitration, Some("test"))
            .await
            .expect("arb");
        advance_run_phase(&pool, None, run_id, RunPhase::Commit, Some("test"))
            .await
            .expect("commit");
        advance_run_phase(&pool, None, run_id, RunPhase::Finalize, Some("test"))
            .await
            .expect("finalize");
        advance_run_phase(&pool, None, run_id, RunPhase::Complete, Some("test"))
            .await
            .expect("complete");
    }

    #[test]
    fn scheduler_compaction_allowed_by_memory_policy() {
        assert!(MemoryPolicy::is_allowed(
            MemoryWriteCategory::Summary,
            MemoryWriteSource::Scheduler,
            "scheduler_compaction",
        ));
    }

    #[tokio::test]
    async fn pipeline_smoke_prompt_memory_gate() {
        let app = tauri::Builder::default()
            .build(tauri::generate_context!("tauri.conf.json"))
            .expect("build app");
        let app_handle = app.handle();
        let pool = setup_pool().await;
        let db = Arc::new(Db { pool });
        let model_client = Arc::new(ModelClient::new(db.pool.clone(), app_handle.clone()));
        let kernel = Kernel::new(db.clone(), model_client, app_handle.clone());

        let input = CorePromptInput {
            content: "Hello".to_string(),
            kind: CoreInputKind::User,
            source: "test".to_string(),
            self_awareness: false,
            self_awareness_hint: false,
            anchor_hits: 1,
            original_input: "Hello".to_string(),
            current_time: None,
            semantic_hint: None,
            introspection_summary: None,
            monologue_intent: None,
            monologue_digest: None,
            prompt_mode: None,
            task_phase: None,
            missing_slots: None,
            resolution_mode: None,
            policy_notes: None,
            redirect_focus: None,
            allow_diagnostics: false,
            world_model_snapshot: None,
            subject_snapshot: None,
            gate_decision: None,
            feedback_bundle: None,
            qualia_snapshot: None,
            wave_state: None,
            attention_schema_summary: None,
            workspace_contributors_summary: None,
            reflective_narrative: None,
            reflective_narrative_evidence_ids: Vec::new(),
            hydrated_context: None,
        };
        let build = build_core_system_message(&db, "default", &input)
            .await
            .expect("build prompt");
        assert!(build.system_message.contains("<<<BEGIN_SECTION:Identity Anchor>>>"));
        assert!(build.system_message.contains("<<<BEGIN_SECTION:User Input>>>"));
        assert!(build.system_message.contains("<<<BEGIN_SECTION:Rolling Summary>>>"));
        assert!(build.system_message.contains("<<<BEGIN_SECTION:Memory Context>>>"));

        db.create_memory_pass_token("run_smoke", "default", 60)
            .await
            .expect("create memory pass token");
        let has_token = db
            .has_memory_pass_token("run_smoke")
            .await
            .unwrap_or(false);
        assert!(has_token);

        let mut state = KernelState::default_for("default");
        let snapshot = kernel
            .build_and_persist_subject_snapshot(&mut state, None, None, "test_smoke")
            .await;
        assert!(snapshot.is_some());
        let snapshot_hash = state
            .last_subject_snapshot_hash
            .clone()
            .expect("snapshot hash");
        let gate_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM gate_decisions WHERE snapshot_hash = ?",
        )
        .bind(&snapshot_hash)
        .fetch_one(&db.pool)
        .await
        .unwrap_or(0);
        assert!(gate_count > 0);
    }

    #[tokio::test]
    async fn gate_order_table_driven() {
        let (kernel, base_settings) = setup_kernel_for_gate_tests().await;

        struct Scenario {
            name: &'static str,
            candidates: Vec<Candidate>,
            state: KernelState,
            settings: Settings,
            expected_reject_id: &'static str,
            expected_reason: &'static str,
            expected_action: &'static str,
            expected_emit: bool,
            expected_tools: bool,
        }

        let mut base_state = KernelState::default_for("conv");
        base_state.ask_budget_remaining = 2;

        let mut scenarios = Vec::new();

        // 1) Policy block
        {
            let mut state = base_state.clone();
            let settings = base_settings.clone();
            let candidates = vec![
                make_candidate_with(
                    "blocked",
                    CandidateKind::EmitMessage,
                    json!({"content": "hi", "policy_block": "C1"}),
                    1,
                ),
                make_candidate_with(
                    "ask",
                    CandidateKind::AskUserQuestion,
                    json!({"question": "Need more detail?"}),
                    2,
                ),
            ];
            scenarios.push(Scenario {
                name: "policy_block",
                candidates,
                state,
                settings,
                expected_reject_id: "blocked",
                expected_reason: "POLICY_BLOCK_C1",
                expected_action: "clarifying_question",
                expected_emit: true,
                expected_tools: true,
            });
        }

        // 2) Stop state (tools blocked)
        {
            let mut state = base_state.clone();
            let settings = base_settings.clone();
            let mut scope = StopScope::default();
            scope.tools = true;
            state.stop_state.apply_reason(
                StopReason {
                    category: StopReasonCategory::LatchBlock,
                    subcode: "stop_tools".to_string(),
                    contract: None,
                },
                scope,
            );
            let candidates = vec![
                make_candidate_with(
                    "tool",
                    CandidateKind::ToolCall,
                    json!({"tool_name": "get_current_time", "arguments": "{}"}),
                    1,
                ),
                make_candidate_with(
                    "ask",
                    CandidateKind::AskUserQuestion,
                    json!({"question": "Need more detail?"}),
                    2,
                ),
            ];
            scenarios.push(Scenario {
                name: "stop_state_tools",
                candidates,
                state,
                settings,
                expected_reject_id: "tool",
                expected_reason: "STOP_STATE_TOOLS",
                expected_action: "clarifying_question",
                expected_emit: true,
                expected_tools: false,
            });
        }

        // 3) Budget cap (tool calls)
        {
            let mut state = base_state.clone();
            let settings = base_settings.clone();
            let candidates = vec![
                make_candidate_with(
                    "tool1",
                    CandidateKind::ToolCall,
                    json!({"tool_name": "get_current_time", "arguments": "{\"tz\":\"UTC\"}"}),
                    1,
                ),
                make_candidate_with(
                    "tool2",
                    CandidateKind::ToolCall,
                    json!({"tool_name": "get_current_time", "arguments": "{\"tz\":\"EST\"}"}),
                    2,
                ),
            ];
            scenarios.push(Scenario {
                name: "budget_tool_cap",
                candidates,
                state,
                settings,
                expected_reject_id: "tool2",
                expected_reason: "CAP_REACHED_TOOLCALL",
                expected_action: "tool_attempt",
                expected_emit: true,
                expected_tools: true,
            });
        }

        // 4) Phase requirement
        {
            let mut state = base_state.clone();
            state.task_phase = TaskPhase::Aborting;
            let settings = base_settings.clone();
            let candidates = vec![
                make_candidate_with(
                    "tool",
                    CandidateKind::ToolCall,
                    json!({"tool_name": "get_current_time", "arguments": "{}"}),
                    1,
                ),
                make_candidate_with(
                    "ask",
                    CandidateKind::AskUserQuestion,
                    json!({"question": "Need more detail?"}),
                    2,
                ),
            ];
            scenarios.push(Scenario {
                name: "phase_halts",
                candidates,
                state,
                settings,
                expected_reject_id: "tool",
                expected_reason: "TASK_PHASE_HALTED",
                expected_action: "explain_blockers",
                expected_emit: true,
                expected_tools: true,
            });
        }

        // 5) Evidence gating
        {
            let mut state = base_state.clone();
            let settings = base_settings.clone();
            let candidates = vec![
                make_candidate_with(
                    "self_claim",
                    CandidateKind::RecordSelfClaim,
                    json!({"claim": "system is stable"}),
                    1,
                ),
                make_candidate_with(
                    "ask",
                    CandidateKind::AskUserQuestion,
                    json!({"question": "Need more detail?"}),
                    2,
                ),
            ];
            scenarios.push(Scenario {
                name: "evidence_missing",
                candidates,
                state,
                settings,
                expected_reject_id: "self_claim",
                expected_reason: "SELF_CLAIM_MISSING_EVIDENCE",
                expected_action: "clarifying_question",
                expected_emit: true,
                expected_tools: true,
            });
        }

        // 6) Tool duplication
        {
            let mut state = base_state.clone();
            let fingerprint = tool_fingerprint("get_current_time", "{}");
            state.tool_call_fingerprints.push(fingerprint);
            let settings = base_settings.clone();
            let candidates = vec![
                make_candidate_with(
                    "tool",
                    CandidateKind::ToolCall,
                    json!({"tool_name": "get_current_time", "arguments": "{}"}),
                    1,
                ),
                make_candidate_with(
                    "ask",
                    CandidateKind::AskUserQuestion,
                    json!({"question": "Need more detail?"}),
                    2,
                ),
            ];
            scenarios.push(Scenario {
                name: "tool_repeat",
                candidates,
                state,
                settings,
                expected_reject_id: "tool",
                expected_reason: "TOOL_CALL_REPEAT",
                expected_action: "clarifying_question",
                expected_emit: true,
                expected_tools: true,
            });
        }

        // 7) Tool penalty
        {
            let mut state = base_state.clone();
            let until = (Utc::now() + ChronoDuration::seconds(60)).to_rfc3339();
            state.tool_failure_penalties.insert(
                "get_current_time".to_string(),
                ToolFailurePenalty {
                    count: 1,
                    last_failure_at: Some(Utc::now().to_rfc3339()),
                    penalty_until: Some(until),
                },
            );
            let settings = base_settings.clone();
            let candidates = vec![
                make_candidate_with(
                    "tool",
                    CandidateKind::ToolCall,
                    json!({"tool_name": "get_current_time", "arguments": "{}"}),
                    1,
                ),
                make_candidate_with(
                    "ask",
                    CandidateKind::AskUserQuestion,
                    json!({"question": "Need more detail?"}),
                    2,
                ),
            ];
            scenarios.push(Scenario {
                name: "tool_penalty",
                candidates,
                state,
                settings,
                expected_reject_id: "tool",
                expected_reason: "TOOL_PENALTY_ACTIVE",
                expected_action: "clarifying_question",
                expected_emit: true,
                expected_tools: true,
            });
        }

        // 8) Candidate-level restriction (prefer ask)
        {
            let mut state = base_state.clone();
            state.uncertainty_count = 1;
            let settings = base_settings.clone();
            let candidates = vec![
                make_candidate_with(
                    "emit",
                    CandidateKind::EmitMessage,
                    json!({"content": "Answer"}),
                    1,
                ),
                make_candidate_with(
                    "ask",
                    CandidateKind::AskUserQuestion,
                    json!({"question": "Need more detail?"}),
                    2,
                ),
            ];
            scenarios.push(Scenario {
                name: "prefer_ask_user",
                candidates,
                state,
                settings,
                expected_reject_id: "emit",
                expected_reason: "POLICY_PREFER_ASK_USER",
                expected_action: "clarifying_question",
                expected_emit: true,
                expected_tools: true,
            });
        }

        for scenario in scenarios {
            let decision = kernel.arbitrate(
                &scenario.candidates,
                &scenario.settings,
                &scenario.state,
                false,
                None,
                None,
                None,
                None,
                None,
            );
            let rejected = decision
                .rejected
                .iter()
                .find(|r| r.id == scenario.expected_reject_id)
                .expect(&format!("{}: missing rejected candidate", scenario.name));
            assert_eq!(
                rejected.reason,
                scenario.expected_reason,
                "{}: wrong gate reason",
                scenario.name
            );
            assert_eq!(
                decision.report.selected_action,
                scenario.expected_action,
                "{}: wrong selected action",
                scenario.name
            );
            assert_eq!(
                decision.report.allowed_capabilities.emit,
                scenario.expected_emit,
                "{}: emit capability mismatch",
                scenario.name
            );
            assert_eq!(
                decision.report.allowed_capabilities.tools,
                scenario.expected_tools,
                "{}: tools capability mismatch",
                scenario.name
            );
        }
    }

    #[test]
    fn action_gate_blocks_reasked_slot_sets() {
        let mut state = KernelState::default_for("conv");
        state.ask_budget_remaining = 1;
        state.asked_slot_sets.push(vec!["alpha".to_string(), "beta".to_string()]);
        let candidate = make_candidate(
            CandidateKind::AskUserQuestion,
            json!({
                "question": "Provide alpha",
                "requested_slots": ["alpha"]
            }),
        );
        let reason = action_gate_reason_for(&candidate, &state, false);
        assert_eq!(reason.as_deref(), Some("ASKED_SLOT_SET_REPEAT"));
    }

    #[test]
    fn action_gate_blocks_refused_slots() {
        let mut state = KernelState::default_for("conv");
        state.ask_budget_remaining = 1;
        state.refused_slots = vec!["gamma".to_string()];
        let candidate = make_candidate(
            CandidateKind::AskUserQuestion,
            json!({
                "question": "Provide gamma",
                "requested_slots": ["gamma"]
            }),
        );
        let reason = action_gate_reason_for(&candidate, &state, false);
        assert_eq!(reason.as_deref(), Some("REFUSED_SLOTS"));
    }

    #[test]
    fn action_gate_blocks_missing_slots_in_strict_mode() {
        let mut state = KernelState::default_for("conv");
        state.missing_slots = vec!["alpha".to_string()];
        state.missing_input_policy = Some("strict".to_string());
        let candidate = make_candidate(
            CandidateKind::EmitMessage,
            json!({
                "content": "Result",
                "requires_resolved_slots": ["alpha"]
            }),
        );
        let reason = action_gate_reason_for(&candidate, &state, false);
        assert_eq!(reason.as_deref(), Some("MISSING_SLOTS_STRICT"));
    }

    #[test]
    fn loop_breaker_triggers_on_repeat_question() {
        let mut state = KernelState::default_for("conv");
        state.recent_questions = vec!["What would you like to do next?".to_string()];
        let candidate = make_candidate(
            CandidateKind::AskUserQuestion,
            json!({"question": "What would you like to do next?"}),
        );
        let settings = test_settings();
        apply_loop_detection_for(&[candidate], &mut state, &settings);
        assert!(state.ask_loop_breaker_triggered);
        assert_eq!(state.task_phase, TaskPhase::ResolvingWithDefaults);
    }

    #[test]
    fn action_gate_blocks_repeat_emit() {
        let mut state = KernelState::default_for("conv");
        let fingerprint = emit_fingerprint("Repeat message");
        state.recent_emit_fingerprints.push(fingerprint);
        let candidate = make_candidate(
            CandidateKind::EmitMessage,
            json!({"content": "Repeat message"})
        );
        let reason = action_gate_reason_for(&candidate, &state, false);
        assert_eq!(reason.as_deref(), Some("EMIT_REPEAT"));
    }

    #[test]
    fn emit_loop_breaker_triggers_on_repeat_message() {
        let mut state = KernelState::default_for("conv");
        state.recent_emit_messages = vec!["Same thought".to_string()];
        let candidate = make_candidate(
            CandidateKind::EmitMessage,
            json!({"content": "Same thought"})
        );
        let settings = test_settings();
        apply_emit_loop_detection_for(&[candidate], &mut state, &settings);
        assert!(state.monologue_emit_loop_breaker_triggered);
    }

    #[test]
    fn calculator_prompt_requires_numeric_signal() {
        assert!(!is_calculator_prompt("Would you say you are self aware at this point?"));
        assert!(!is_calculator_prompt("I am 2 years old"));
        assert!(is_calculator_prompt("calculate 2+2"));
        assert!(is_calculator_prompt("mean of 2, 3, 5"));
    }

    #[test]
    fn monologue_surface_requires_explicit_request() {
        assert!(!is_monologue_surface_request("What are you thinking right now?"));
        assert!(is_monologue_surface_request("Show your inner monologue."));
    }

    #[test]
    fn action_gate_blocks_self_claim_missing_evidence() {
        let state = KernelState::default_for("conv");
        let candidate = make_candidate(
            CandidateKind::RecordSelfClaim,
            json!({"claim_text": "I prefer concise replies."}),
        );
        let reason = action_gate_reason_for(&candidate, &state, false);
        assert_eq!(reason.as_deref(), Some("SELF_CLAIM_MISSING_EVIDENCE"));
    }

    #[test]
    fn self_audit_detection_triggers() {
        assert!(is_self_audit_request("Run a self-audit."));
        assert!(is_self_audit_request("Show runtime state."));
        assert!(is_self_audit_request("System status, please."));
        assert!(is_self_audit_request("Capabilities dump."));
        assert!(!is_self_audit_request("What can you do?"));
    }

    #[test]
    fn self_audit_detection_ambiguous() {
        assert!(is_self_audit_ambiguous("What can you do?"));
        assert!(is_self_audit_ambiguous("What are you thinking about?"));
        assert!(!is_self_audit_ambiguous("Run a self-audit."));
    }

    #[test]
    fn self_awareness_detection() {
        assert!(is_self_awareness_query("Are you self-aware?"));
        assert!(is_self_awareness_query("What is your internal state right now?"));
        assert!(is_self_awareness_query("Do you feel conflicted about this?"));
        assert!(!is_self_awareness_query("Are you aware of the latest update?"));
        assert!(!is_self_awareness_query("Can you do X?"));
        assert!(!is_self_awareness_query("Do you remember what I said yesterday?"));
        assert!(!is_self_awareness_query("Run a self-audit."));
    }

    #[test]
    fn context_hydration_detects_capabilities_and_introspection() {
        let (intent, sections, fallback) = compute_context_hydration(
            "What would you change about your system? How did you decide?",
            false,
            false,
        );
        assert!(intent.tags.iter().any(|t| t == "self_awareness"));
        assert!(intent.tags.iter().any(|t| t == "capabilities"));
        assert!(intent.tags.iter().any(|t| t == "introspection"));
        assert!(sections.iter().any(|s| s == "Capabilities"));
        assert!(fallback.is_none());
    }

    #[test]
    fn context_hydration_fallback_when_empty() {
        let (intent, sections, fallback) = compute_context_hydration("Hello there.", false, false);
        assert!(intent.tags.is_empty());
        assert!(sections.iter().any(|s| s == "Rolling Summary"));
        assert!(sections.iter().any(|s| s == "Workspace Snapshot"));
        assert_eq!(fallback.as_deref(), Some("fallback_empty_intent"));
    }

    #[test]
    fn json_repair_recovers_object() {
        let raw = r#"{"stance":"synth","message":"ok"} trailing"#;
        let (value, repaired) = parse_json_object_with_repair(raw);
        assert!(value.is_some());
        assert!(repaired);
    }

    #[test]
    fn clarifier_overlap_detects_latest_user_input() {
        let candidate = make_candidate(
            CandidateKind::AskUserQuestion,
            json!({"question": "Do you want to use the API key now?"}),
        );
        assert!(candidate_overlaps_last_user_input(
            &candidate,
            "We need to confirm the API key."
        ));
        assert!(!candidate_overlaps_last_user_input(
            &candidate,
            "Let's discuss pricing instead."
        ));
    }

    #[test]
    fn pending_prompt_starvation_forces_after_limit() {
        assert!(pending_prompt_force_reason(
            PENDING_PROMPT_STARVATION_LIMIT - 1,
            false,
            None
        )
        .is_none());
        assert_eq!(
            pending_prompt_force_reason(PENDING_PROMPT_STARVATION_LIMIT, false, None),
            Some("starvation")
        );
        assert_eq!(
            pending_prompt_force_reason(
                0,
                true,
                Some(AUTO_SURFACE_MAX_AGE_SECS + 1)
            ),
            Some("auto_surface_max_age")
        );
        assert_eq!(
            pending_prompt_force_reason(0, true, Some(AUTO_SURFACE_SLA_SECS + 1)),
            Some("auto_surface_sla")
        );
        assert!(pending_prompt_force_reason(0, true, Some(1)).is_none());
        assert!(pending_prompt_force_reason(0, true, None).is_none());
        assert!(pending_prompt_force_reason(0, false, Some(100)).is_none());
    }

    #[test]
    fn auto_surface_max_age_forces_even_without_starvation() {
        assert_eq!(
            pending_prompt_force_reason(0, true, Some(AUTO_SURFACE_MAX_AGE_SECS)),
            Some("auto_surface_max_age")
        );
    }

    #[test]
    fn internal_asks_ignore_budget_gate() {
        let mut state = KernelState::default_for("conv");
        state.ask_budget_remaining = 0;
        let candidate = make_candidate(
            CandidateKind::AskUserQuestion,
            json!({"question": "Provide alpha", "requested_slots": ["alpha"]}),
        );
        let reason = action_gate_reason_for(&candidate, &state, true);
        assert_ne!(reason.as_deref(), Some("ASK_BUDGET_EXHAUSTED"));
    }

    #[test]
    fn update_workspace_applies_fields() {
        let mut state = KernelState::default_for("conv");
        let payload = json!({
            "goal_thread": "Improve cognition",
            "goal_stack": [{
                "goal": "Improve cognition",
                "steps": ["Audit modules"],
                "current_step_index": 0
            }],
            "open_questions": ["What is missing?"],
            "current_focus": "Workspace persistence",
            "focus_rationale": "High activation"
        });
        let changed = apply_workspace_update(&mut state, &payload);
        assert!(changed);
        assert_eq!(state.workspace_goal_thread.as_deref(), Some("Improve cognition"));
        assert_eq!(state.workspace_goal_stack.len(), 1);
        assert_eq!(state.workspace_goal_stack[0].goal, "Improve cognition");
        assert_eq!(state.workspace_open_questions, vec!["What is missing?".to_string()]);
        assert_eq!(state.workspace_current_focus.as_deref(), Some("Workspace persistence"));
        assert_eq!(state.workspace_focus_rationale.as_deref(), Some("High activation"));
    }

    #[test]
    fn auto_memory_decision_triggers_on_identity() {
        let decision = auto_memory_decision("My name is Alice.", "Got it, I'll remember.");
        assert!(decision.trigger);
        assert!(!decision.ambiguity);
    }

    #[test]
    fn auto_memory_decision_blocks_on_ambiguity() {
        let decision = auto_memory_decision("I might live in Paris.", "Noted.");
        assert!(!decision.trigger);
        assert!(decision.ambiguity);
    }

    #[test]
    fn auto_memory_decision_requires_high_confidence() {
        let decision = auto_memory_decision("I like jazz music.", "Noted.");
        assert!(!decision.trigger);
        assert!(!decision.ambiguity);
        assert!(decision.score < AUTO_MEMORY_CONFIDENCE_THRESHOLD);
    }

    #[test]
    fn workspace_response_requires_focus_or_exception() {
        let mut state = KernelState::default_for("conv");
        state.workspace_current_focus = Some("Workspace persistence".to_string());
        let (compliant, exception) = workspace_response_compliant("We should improve Workspace persistence.", &state);
        assert!(compliant);
        assert!(!exception);

        let (compliant, exception) = workspace_response_compliant("This is unrelated.", &state);
        assert!(!compliant);
        assert!(!exception);

        let (compliant, exception) =
            workspace_response_compliant("Not relevant to the current workspace focus.", &state);
        assert!(compliant);
        assert!(exception);
    }

    #[test]
    fn workspace_delta_fields_detects_changes() {
        let mut state = KernelState::default_for("conv");
        let snapshot = WorkspaceSnapshot::from_state(&state);
        state.workspace_current_focus = Some("New focus".to_string());
        let delta = workspace_delta_fields(&snapshot, &state);
        assert_eq!(delta.len(), 1);
        assert!(delta.contains(&"current_focus".to_string()));
    }

    #[test]
    fn workspace_anchor_applies_focus() {
        let mut state = KernelState::default_for("conv");
        state.workspace_current_focus = Some("Anchored focus".to_string());
        let mut summary = InnerSummary::default();
        apply_workspace_anchor(&mut summary, &state, &[]);
        assert_eq!(summary.focus, "Anchored focus");
    }

    #[test]
    fn coerce_candidate_converts_emit_without_evidence() {
        let mut state = KernelState::default_for("conv");
        state.workspace_working_set_topics = vec!["Ken".to_string()];
        let candidate = make_candidate(
            CandidateKind::EmitMessage,
            json!({"content": "Ken built a radio transmitter."}),
        );
        let result = coerce_proactive_candidate_for_evidence(&candidate, &state, false, false, false);
        let coerced = result.candidate.expect("coerced");
        assert!(matches!(coerced.kind, CandidateKind::AskUserQuestion));
        let question = coerced
            .payload
            .get("question")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(question.to_lowercase().starts_with("working hypothesis:"));
    }

    #[test]
    fn coerce_candidate_allows_emit_with_evidence_ids() {
        let state = KernelState::default_for("conv");
        let candidate = make_candidate(
            CandidateKind::EmitMessage,
            json!({"content": "Evidence backed.", "evidence_event_ids": [1]}),
        );
        let result = coerce_proactive_candidate_for_evidence(&candidate, &state, true, true, false);
        let coerced = result.candidate.expect("candidate");
        assert!(matches!(coerced.kind, CandidateKind::EmitMessage));
    }

    #[test]
    fn coerce_candidate_uses_open_question_without_prefix() {
        let mut state = KernelState::default_for("conv");
        state.workspace_open_questions = vec!["What evidence supports this?".to_string()];
        let candidate = make_candidate(
            CandidateKind::EmitMessage,
            json!({"content": "What evidence supports this?"}),
        );
        let result = coerce_proactive_candidate_for_evidence(&candidate, &state, false, false, false);
        let coerced = result.candidate.expect("candidate");
        assert!(matches!(coerced.kind, CandidateKind::AskUserQuestion));
        let question = coerced
            .payload
            .get("question")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert_eq!(question, "What evidence supports this?");
    }

    #[test]
    fn normalize_hypotheses_payload_wraps_text() {
        let mut payload = json!({
            "active_hypotheses": ["Hypothesis A"]
        });
        let value = payload
            .get_mut("active_hypotheses")
            .expect("active_hypotheses");
        normalize_hypotheses_payload(value, true, &[], &[]);
        let list = payload
            .get("active_hypotheses")
            .and_then(|v| v.as_array())
            .expect("array");
        let obj = list[0].as_object().expect("object");
        assert_eq!(obj.get("speculative").and_then(|v| v.as_bool()), Some(true));
    }

    #[test]
    fn candidate_new_terms_detection() {
        let mut state = KernelState::default_for("conv");
        state.workspace_working_set_topics = vec!["Ken".to_string()];
        let has_new = candidate_introduces_new_terms("Ken built a radio", &state);
        assert!(has_new);
    }

    #[test]
    fn workspace_fallback_payload_ignores_summary_focus() {
        let state = KernelState::default_for("conv");
        let mut summary = InnerSummary::default();
        summary.focus = "Workspace persistence".to_string();
        summary.open_questions = vec!["What is missing?".to_string()];
        summary.blockers = vec!["Latency drift".to_string()];
        summary.active_threads = vec!["Binding".to_string()];
        summary.next_moves = vec!["Add checklist".to_string()];

        let payload = Kernel::build_workspace_fallback_payload(&mut summary, &state, "");

        assert!(payload.is_none());
    }

    #[test]
    fn workspace_fallback_payload_fills_focus_from_seed() {
        let state = KernelState::default_for("conv");
        let mut summary = InnerSummary::default();
        summary.focus = "Summary focus should be ignored".to_string();
        let focus_seed = "Strengthen workspace compliance";
        let expected = summarize_snippet(focus_seed, 160);

        let payload = Kernel::build_workspace_fallback_payload(&mut summary, &state, focus_seed)
            .expect("payload");

        assert_eq!(
            payload.get("current_focus").and_then(|v| v.as_str()),
            Some(expected.as_str())
        );
        assert_eq!(
            payload.get("focus_rationale").and_then(|v| v.as_str()),
            Some("fallback from focus seed")
        );
    }

    #[test]
    fn workspace_fallback_payload_uses_existing_workspace_focus() {
        let mut state = KernelState::default_for("conv");
        state.workspace_current_focus = Some("Existing focus".to_string());
        state.workspace_meta.current_focus = Some(make_field_meta(false, &[1], &[]));
        let mut summary = InnerSummary::default();

        let payload = Kernel::build_workspace_fallback_payload(&mut summary, &state, "")
            .expect("payload");

        assert_eq!(
            payload.get("current_focus").and_then(|v| v.as_str()),
            Some("Existing focus")
        );
        assert_eq!(
            payload.get("focus_rationale").and_then(|v| v.as_str()),
            Some("fallback from existing workspace")
        );
    }

    #[test]
    fn workspace_fallback_payload_returns_none_without_focus_or_seed() {
        let state = KernelState::default_for("conv");
        let mut summary = InnerSummary::default();

        let payload = Kernel::build_workspace_fallback_payload(&mut summary, &state, "");

        assert!(payload.is_none());
    }

    #[test]
    fn monologue_style_violation_blocks_greeting_and_help() {
        assert_eq!(monologue_style_violation("Hello there.", "Ken"), Some("greeting"));
        assert_eq!(
            monologue_style_violation("How can I help today?", "Ken"),
            Some("help_offer")
        );
        assert_eq!(
            monologue_style_violation("Ken, we should refocus.", "Ken"),
            Some("user_address")
        );
        assert_eq!(
            monologue_style_violation("I am an LLM and I don't have feelings.", "Ken"),
            Some("self_disclaimer")
        );
    }

    #[test]
    fn update_workspace_payload_requires_substantive_fields() {
        let empty = json!({"current_focus": "None"});
        assert!(!update_workspace_payload_has_substantive_fields(&empty));
        let payload = json!({"current_focus": "Focus", "open_questions": ["Why?"]});
        assert!(update_workspace_payload_has_substantive_fields(&payload));
        let goal_stack = json!({"goal_stack": [{"goal": "Ship"}]});
        assert!(update_workspace_payload_has_substantive_fields(&goal_stack));
    }

    #[test]
    fn update_workspace_payload_detects_evidence() {
        let no_evidence = json!({"current_focus": "Focus"});
        assert!(!update_workspace_payload_has_evidence(&no_evidence));
        let evidence = json!({"current_focus": "Focus", "evidence_event_ids": [1]});
        assert!(update_workspace_payload_has_evidence(&evidence));
        let hypothesis_evidence = json!({
            "active_hypotheses": [{"text": "Hyp", "evidence_event_ids": [2]}]
        });
        assert!(update_workspace_payload_has_evidence(&hypothesis_evidence));
        let goal_stack_evidence = json!({
            "goal_stack": [{
                "goal": "Ship",
                "steps": [{"text": "Build", "evidence_event_ids": [3]}]
            }]
        });
        assert!(update_workspace_payload_has_evidence(&goal_stack_evidence));
    }

    #[test]
    fn prediction_metrics_include_reasoning_targets() {
        let metrics = allowed_prediction_metrics();
        assert!(metrics.contains("clarification_rate"));
        assert!(metrics.contains("workspace_stability_rate"));
    }

    #[test]
    fn workspace_demotion_respects_focus_grace_window() {
        let mut state = KernelState::default_for("conv");
        state.workspace_current_focus = Some("Focus".to_string());
        state.workspace_meta.current_focus = Some(make_field_meta(false, &[1], &[]));

        let mut flags = WorkspaceEvidenceFlags::default();
        flags.current_focus_ok = Some(false);

        let demoted = apply_workspace_demotions_from_flags(&mut state, false, &flags);
        assert!(demoted.is_empty());
        assert_eq!(
            state.workspace_meta.current_focus.as_ref().map(|m| m.speculative),
            Some(false)
        );

        let demoted = apply_workspace_demotions_from_flags(&mut state, true, &flags);
        assert!(demoted.contains(&"current_focus".to_string()));
        assert_eq!(
            state.workspace_meta.current_focus.as_ref().map(|m| m.speculative),
            Some(true)
        );
    }

    #[test]
    fn workspace_demotion_marks_open_questions() {
        let mut state = KernelState::default_for("conv");
        state.workspace_open_questions = vec!["Why?".to_string()];
        state.workspace_meta.open_questions = vec![make_list_meta("Why?", false, &[1], &[])];

        let mut flags = WorkspaceEvidenceFlags::default();
        flags.open_questions_ok = vec![false];

        let demoted = apply_workspace_demotions_from_flags(&mut state, true, &flags);
        assert!(demoted.iter().any(|d| d.contains("open_question:Why?")));
        assert_eq!(state.workspace_meta.open_questions[0].speculative, true);
    }

    #[test]
    fn workspace_required_ignores_speculative_focus() {
        let mut state = KernelState::default_for("conv");
        state.workspace_current_focus = Some("Speculative focus".to_string());
        state.workspace_meta.current_focus = Some(make_field_meta(true, &[], &[]));
        assert!(!workspace_required(&state));
    }

    #[test]
    fn workspace_alignment_tokens_use_verified_only() {
        let mut state = KernelState::default_for("conv");
        state.workspace_current_focus = Some("Speculative focus".to_string());
        state.workspace_meta.current_focus = Some(make_field_meta(true, &[], &[]));
        state.workspace_working_set_topics = vec!["Verified topic".to_string()];
        state.workspace_meta.working_set_topics =
            vec![make_list_meta("Verified topic", false, &[], &[1])];

        let tokens = workspace_alignment_tokens(&state);
        assert!(tokens.contains("verified"));
        assert!(!tokens.contains("speculative"));
    }

    #[test]
    fn extract_context_evidence_ids_parses_payload() {
        let single = r#"{"key":"foo","value":"bar","evidence_event_id": 42}"#;
        assert_eq!(extract_context_evidence_ids(single), vec![42]);
        let multiple = r#"{"evidence_event_ids":[3,2,2,1]}"#;
        assert_eq!(extract_context_evidence_ids(multiple), vec![1, 2, 3]);
        let invalid = "not json";
        assert!(extract_context_evidence_ids(invalid).is_empty());
    }

    #[test]
    fn speculative_workspace_detection_flags_terms() {
        let mut state = KernelState::default_for("conv");
        state.workspace_current_focus = Some("Speculative focus".to_string());
        state.workspace_meta.current_focus = Some(make_field_meta(true, &[], &[]));
        let terms = collect_speculative_terms(&state);
        assert!(response_uses_speculative_workspace(
            "Speculative focus might be relevant.",
            &terms
        ));
    }

    #[test]
    fn response_question_detection() {
        assert!(response_is_question("What is the next step?"));
        assert!(response_is_question("Could you clarify?"));
        assert!(!response_is_question("This is a statement."));
    }

    #[test]
    fn candidate_new_terms_uses_verified_tokens() {
        let mut state = KernelState::default_for("conv");
        state.workspace_current_focus = Some("Ken".to_string());
        state.workspace_meta.current_focus = Some(make_field_meta(true, &[], &[]));
        state.workspace_working_set_topics = vec!["Ergo".to_string()];
        state.workspace_meta.working_set_topics =
            vec![make_list_meta("Ergo", false, &[], &[2])];

        assert!(candidate_introduces_new_terms("Ken built a radio", &state));
        assert!(!candidate_introduces_new_terms("Ergo is relevant", &state));
    }

    #[test]
    fn proactive_memory_pass_due_respects_last_timestamp() {
        let mut state = KernelState::default_for("conv");
        let now = Utc::now();
        state.last_proactive_memory_pass_at = Some(now.to_rfc3339());
        assert!(!proactive_memory_pass_due(&state, now));
        let earlier = now - chrono::Duration::seconds(PROACTIVE_MEMORY_PASS_MIN_SECS + 1);
        state.last_proactive_memory_pass_at = Some(earlier.to_rfc3339());
        assert!(proactive_memory_pass_due(&state, now));
    }

    #[test]
    fn action_gate_defers_ask_after_hypothesis_promotion() {
        let mut state = KernelState::default_for("conv");
        state.hypothesis_defer_until = Some(5);
        state.monologue_count = 3;
        let candidate = make_candidate(
            CandidateKind::AskUserQuestion,
            json!({"question": "Clarify details?"}),
        );
        let reason = action_gate_reason_for(&candidate, &state, true);
        assert_eq!(reason.as_deref(), Some("ASK_DEFERRED"));
    }

    #[test]
    fn action_gate_allows_clarifier_when_ask_budget_exhausted() {
        let mut state = KernelState::default_for("conv");
        state.ask_budget_remaining = 0;
        state.last_user_input = Some("Memory system diagnostics are failing.".to_string());
        let candidate = make_candidate(
            CandidateKind::AskUserQuestion,
            json!({"question": "Which memory system diagnostic failed?"}),
        );
        let reason = action_gate_reason_for(&candidate, &state, false);
        assert!(reason.is_none(), "Clarifier tied to latest input should bypass ask budget");
    }

    #[test]
    fn action_gate_allows_clarifier_when_internal_cycle_deferred() {
        let mut state = KernelState::default_for("conv");
        state.hypothesis_defer_until = Some(5);
        state.monologue_count = 2;
        state.last_user_input = Some("Memory system diagnostics are failing.".to_string());
        let candidate = make_candidate(
            CandidateKind::AskUserQuestion,
            json!({"question": "Which memory system diagnostic failed?"}),
        );
        let reason = action_gate_reason_for(&candidate, &state, true);
        assert!(reason.is_none(), "Clarifier tied to latest input should bypass ASK_DEFERRED");
    }

    #[test]
    fn proactive_followup_allows_slot_match() {
        let mut state = KernelState::default_for("conv");
        state.last_proactive_question = Some("Provide alpha".to_string());
        state.last_asked_slots = vec!["alpha".to_string()];
        let candidate = make_candidate(
            CandidateKind::AskUserQuestion,
            json!({"question": "Provide alpha", "requested_slots": ["alpha"]}),
        );
        assert!(proactive_followup_allowed(&state, &candidate));
    }

    #[test]
    fn proactive_followup_allows_overlap() {
        let mut state = KernelState::default_for("conv");
        state.last_proactive_question = Some("Provide architecture summary".to_string());
        let candidate = make_candidate(
            CandidateKind::AskUserQuestion,
            json!({"question": "Can you provide a summary of the architecture?"}),
        );
        assert!(proactive_followup_allowed(&state, &candidate));
    }

    fn make_identity_metrics(variant: &str, turns: i64, score: f64) -> IdentityAbMetrics {
        IdentityAbMetrics {
            variant: variant.to_string(),
            start_at: "start".to_string(),
            end_at: "end".to_string(),
            turns,
            feedback_pushback: 0,
            feedback_clarify: 0,
            feedback_follow_up: 0,
            feedback_agree: 0,
            feedback_disengage: 0,
            gate_failures: 0,
            score,
        }
    }

    #[test]
    fn identity_ab_decision_requires_min_turns() {
        let current = make_identity_metrics("A", IDENTITY_AB_MIN_TURNS - 1, 0.2);
        let decision = decide_identity_ab_variant(&current, None, IDENTITY_AB_MIN_TURNS);
        assert_eq!(decision, IdentityAbDecision::Stay);
    }

    #[test]
    fn identity_ab_decision_switches_for_collection_when_other_missing() {
        let current = make_identity_metrics("A", IDENTITY_AB_MIN_TURNS, 0.2);
        let decision = decide_identity_ab_variant(&current, None, IDENTITY_AB_MIN_TURNS);
        assert_eq!(
            decision,
            IdentityAbDecision::SwitchForCollection("B".to_string())
        );
    }

    #[test]
    fn identity_ab_decision_switches_for_collection_when_other_under_min() {
        let current = make_identity_metrics("A", IDENTITY_AB_MIN_TURNS, 0.2);
        let other = make_identity_metrics("B", IDENTITY_AB_MIN_TURNS - 2, 0.8);
        let decision = decide_identity_ab_variant(&current, Some(&other), IDENTITY_AB_MIN_TURNS);
        assert_eq!(
            decision,
            IdentityAbDecision::SwitchForCollection("B".to_string())
        );
    }

    #[test]
    fn identity_ab_decision_prefers_higher_score_when_both_ready() {
        let current = make_identity_metrics("A", IDENTITY_AB_MIN_TURNS, 0.2);
        let other = make_identity_metrics("B", IDENTITY_AB_MIN_TURNS, 0.5);
        let decision = decide_identity_ab_variant(&current, Some(&other), IDENTITY_AB_MIN_TURNS);
        assert_eq!(
            decision,
            IdentityAbDecision::SwitchForWinner("B".to_string())
        );
    }

    #[test]
    fn identity_ab_decision_stays_when_current_score_higher() {
        let current = make_identity_metrics("B", IDENTITY_AB_MIN_TURNS, 0.6);
        let other = make_identity_metrics("A", IDENTITY_AB_MIN_TURNS, 0.2);
        let decision = decide_identity_ab_variant(&current, Some(&other), IDENTITY_AB_MIN_TURNS);
        assert_eq!(decision, IdentityAbDecision::Stay);
    }

    #[test]
    fn focus_shift_moves_from_user_to_hypothesis() {
        let hypotheses = vec![WorkspaceHypothesis {
            text: "Qualia framework".to_string(),
            confidence: 0.7,
            speculative: false,
            evidence_event_ids: Vec::new(),
            belief_ids: Vec::new(),
        }];
        let shift = focus_shift_candidate(
            "Ken",
            "Discuss qualia in AI systems.",
            "Ken",
            &hypotheses,
            Some("Goal thread"),
        )
        .expect("shift");
        assert_eq!(shift.0, "Qualia framework");
    }

    #[test]
    fn focus_shift_uses_goal_when_no_hypothesis() {
        let hypotheses = Vec::new();
        let shift = focus_shift_candidate(
            "Ken",
            "Discuss qualia in AI systems.",
            "Ken",
            &hypotheses,
            Some("Goal thread"),
        )
        .expect("shift");
        assert_eq!(shift.0, "Goal thread");
    }

    #[test]
    fn focus_shift_skips_relational_input() {
        let hypotheses = vec![WorkspaceHypothesis {
            text: "Qualia framework".to_string(),
            confidence: 0.7,
            speculative: false,
            evidence_event_ids: Vec::new(),
            belief_ids: Vec::new(),
        }];
        let shift = focus_shift_candidate(
            "Ken",
            "I built you and I want you to know.",
            "Ken",
            &hypotheses,
            Some("Goal thread"),
        );
        assert!(shift.is_none());
    }

    #[test]
    fn relational_input_detection_requires_affective_signal() {
        assert!(is_relational_input("I built you and I want you to know this."));
        assert!(!is_relational_input("Explain the architecture of your memory system."));
    }

    #[test]
    fn telemetry_claim_detection_handles_common_phrases() {
        assert!(response_has_telemetry_claim("Current focus: memory hygiene.", false));
        assert!(response_has_telemetry_claim("My self-state is stable.", false));
        assert!(!response_has_telemetry_claim("This is a general state-of-the-art note.", false));
        assert!(response_has_telemetry_claim("Task phase: running.", true));
        assert!(response_has_telemetry_claim("Gate decision is VERIFY.", true));
        assert!(response_has_telemetry_claim("I have a pending prompt.", true));
    }

    #[test]
    fn stance_claim_detection_handles_common_phrases() {
        assert!(response_has_stance_claim("I'm thinking about it."));
        assert!(response_has_stance_claim("I don't have independent thoughts."));
        assert!(response_has_stance_claim("I'm trying to answer based on learned patterns."));
        assert!(!response_has_stance_claim("This is a general state-of-the-art note."));
    }

    #[test]
    fn state_ref_block_extraction_strips_and_parses() {
        let input = "Status update. <state_ref>{\"claims\":[{\"kind\":\"self_state\",\"evidence_event_ids\":[12]}]}</state_ref>";
        let (cleaned, block) = extract_state_ref_block(input);
        assert_eq!(cleaned, "Status update.");
        let block = block.expect("state_ref block");
        assert_eq!(block.claims.len(), 1);
        assert_eq!(block.claims[0].evidence_event_ids, vec![12]);
    }

    #[test]
    fn user_requested_state_detection() {
        assert!(user_requested_state("What is your current focus?"));
        assert!(user_requested_state("Show workspace state"));
        assert!(user_requested_state("Having any interesting thoughts?"));
        assert!(user_requested_state("How are you feeling today?"));
        assert!(!user_requested_state("Summarize the architecture."));
    }

    #[test]
    fn response_has_user_attribution_detects_patterns() {
        assert!(response_has_user_attribution("Ken said the system is stable.", "Ken"));
        assert!(response_has_user_attribution("As Ken mentioned, the logs changed.", "Ken"));
        assert!(response_has_user_attribution("I appreciate your assessment here.", "Ken"));
        assert!(!response_has_user_attribution("The system said it is stable.", "Ken"));
    }

    #[test]
    fn user_attribution_rewrite_handles_assessment() {
        let text = "While I appreciate your assessment, I want to ask a question.";
        let rewritten = rewrite_user_attribution_text(text, "Ken");
        assert!(!rewritten.to_lowercase().contains("your assessment"));
        assert!(rewritten.to_lowercase().contains("the assessment"));
    }

    #[test]
    fn user_attribution_fallback_prefers_quote_match() {
        let allowlist = vec![
            (10, "Hello world".to_string()),
            (11, "Different snippet".to_string()),
        ];
        let ids = extract_user_attribution_fallback("Ken said \"hello world\" today.", &allowlist);
        assert_eq!(ids, vec![10]);
    }

    #[test]
    fn user_attribution_blocked_without_evidence() {
        let validation = ValidationResult::default();
        assert!(user_attribution_blocked(&[], &validation, false));
    }

    #[test]
    fn user_attribution_allows_valid_evidence() {
        let mut validation = ValidationResult::default();
        validation.fresh_ok = true;
        validation.valid_evidence_ids = vec![1];
        assert!(!user_attribution_blocked(&[1], &validation, true));
    }

    #[test]
    fn tool_failure_gate_blocks_when_triggered() {
        assert!(should_block_tool_failure(true, true, false, true, "Result"));
        assert!(!should_block_tool_failure(true, true, true, true, "Result"));
        assert!(!should_block_tool_failure(true, false, false, true, "Result"));
    }

    #[test]
    fn tool_result_attribution_requires_context_evidence() {
        let mut validation = ValidationResult::default();
        validation.fresh_ok = true;
        validation.valid_evidence_ids = vec![2];
        assert!(tool_result_attribution_blocked(&[1], &[2], &validation));
        assert!(!tool_result_attribution_blocked(&[2], &[2], &validation));
    }

    #[test]
    fn validate_tool_call_args_rejects_missing_fields() {
        assert!(validate_tool_call_args("read_context", "{}").is_err());
        assert!(validate_tool_call_args("save_context", "{\"key\":\"\"}").is_err());
        assert!(validate_tool_call_args("save_context", "{\"key\":\"k\",\"value\":\"\"}").is_err());
        assert!(validate_tool_call_args("save_context", "{\"key\":\"k\",\"value\":\"v\"}").is_ok());
    }

    #[test]
    fn working_hypothesis_prefix_adds_marker() {
        let prefixed = working_hypothesis_prefix("Speculative response", false);
        assert_eq!(prefixed, "Working hypothesis: Speculative response");
        let already = working_hypothesis_prefix("Working hypothesis: ok", false);
        assert_eq!(already, "Working hypothesis: ok");
        let disabled = working_hypothesis_prefix("Speculative response", true);
        assert_eq!(disabled, "Speculative response (speculative=true)");
    }

    #[test]
    fn state_disclosure_block_reason_requires_evidence() {
        let mut validation = ValidationResult::default();
        validation.valid_evidence_ids = vec![1];
        assert_eq!(
            state_disclosure_block_reason(&[], Some(&validation)),
            Some("missing_evidence")
        );
        let mut invalid = ValidationResult::default();
        invalid.invalid_evidence_ids = vec![3];
        assert_eq!(
            state_disclosure_block_reason(&[3], Some(&invalid)),
            Some("invalid_evidence")
        );
        assert_eq!(
            state_disclosure_block_reason(&[1], Some(&validation)),
            None
        );
    }

    #[test]
    fn inject_gate_notice_prepends_notice_once() {
        let notice = "Notice: responding under uncertainty.";
        let content = "Hello there.";
        let updated = inject_gate_notice(content, notice);
        assert!(updated.starts_with(notice));
        assert!(updated.contains(content));

        let already = format!("{}\n\n{}", notice, content);
        let unchanged = inject_gate_notice(&already, notice);
        assert_eq!(unchanged, already);
    }

    #[test]
    fn workspace_has_verified_anchor_when_focus_verified() {
        let mut state = KernelState::default_for("conv");
        state.workspace_current_focus = Some("Focus".to_string());
        state.workspace_meta.current_focus = Some(WorkspaceFieldMeta {
            speculative: false,
            evidence_event_ids: vec![1],
            belief_ids: Vec::new(),
        });
        assert!(workspace_has_verified_anchor(&state));
    }

    #[test]
    fn outcome_quality_penalizes_user_feedback() {
        let outcomes = vec![
            Outcome {
                action_type: "user_feedback_pushback".to_string(),
                success: true,
                observations: "nope".to_string(),
                source: "user_feedback".to_string(),
                failure_kind: None,
                target_key: None,
                action_id: None,
                timestamp: "now".to_string(),
            },
            Outcome {
                action_type: "user_feedback_agree".to_string(),
                success: true,
                observations: "ok".to_string(),
                source: "user_feedback".to_string(),
                failure_kind: None,
                target_key: None,
                action_id: None,
                timestamp: "now".to_string(),
            },
        ];
        let quality = outcome_quality_from_outcomes(&outcomes).unwrap_or(0.0);
        assert!(quality < 1.0, "Expected mixed feedback to reduce quality");
        assert!(quality > 0.0, "Expected mixed feedback to keep quality above 0");
    }

    #[test]
    fn assistant_name_mismatch_detects_wrong_name() {
        let text = "I am Ergo, here to help.";
        let mismatch = assistant_name_mismatch_detected(text, "Nova", "Ken");
        assert_eq!(mismatch.as_deref(), Some("ergo"));
        let mismatch_ok = assistant_name_mismatch_detected(text, "Ergo", "Ken");
        assert!(mismatch_ok.is_none());
    }

    #[test]
    fn tool_penalty_key_scopes_target_hint() {
        let key = tool_penalty_key("web.run", Some("https://example.com/path?q=1"));
        assert_eq!(key, "web.run::example.com");

        let key = tool_penalty_key("web.run", Some("example.com"));
        assert_eq!(key, "web.run::example.com");

        let key = tool_penalty_key("web.run", Some("   "));
        assert_eq!(key, "web.run");
    }

    #[tokio::test]
    async fn planning_error_does_not_increment_tool_penalties() {
        let (kernel, _settings) = setup_kernel_for_gate_tests().await;
        let mut state = KernelState::default_for("conv");
        let outcome = Outcome {
            action_type: "tool_dispatch_failed".to_string(),
            success: false,
            observations: "invalid args".to_string(),
            source: "web_lookup".to_string(),
            failure_kind: Some(TOOL_FAILURE_KIND_PLANNING.to_string()),
            target_key: Some(tool_penalty_key("web_lookup", None)),
            action_id: None,
            timestamp: Utc::now().to_rfc3339(),
        };

        kernel.apply_outcomes(&mut state, &[outcome]).await;

        assert_eq!(state.tool_failure_count, 0);
        assert_eq!(state.failure_count, 0);
        assert!(state.tool_failure_penalties.is_empty());
    }

    #[tokio::test]
    async fn deferred_tool_call_replays_after_throttle_clears() {
        let (kernel, _settings) = setup_kernel_for_gate_tests().await;
        let content = json!({
            "action_id": "action-1",
            "tool_name": "get_current_time",
            "args_json": "{}"
        })
        .to_string();

        kernel
            .db
            .enqueue_deferred_item(
                "conv",
                "tool_call",
                &content,
                Some("test"),
                "CONTROLLER_THROTTLE_TOOLS",
                None,
                None,
                0,
                None,
                None,
            )
            .await
            .expect("enqueue deferred");

        let mut state = KernelState::default_for("conv");
        state.controller_gate = Some(ControllerGate {
            throttle_tools: true,
            ..Default::default()
        });

        let mut created_at = 0i64;
        let candidates = kernel
            .deferred_tool_candidates(&state, &mut created_at)
            .await;
        assert!(candidates.is_empty());

        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM deferred_queue WHERE conversation_id = ? AND item_type = 'tool_call'",
        )
        .bind("conv")
        .fetch_one(&kernel.db.pool)
        .await
        .expect("count deferred");
        assert_eq!(count, 1);

        state.controller_gate = Some(ControllerGate {
            throttle_tools: false,
            ..Default::default()
        });

        let candidates = kernel
            .deferred_tool_candidates(&state, &mut created_at)
            .await;
        assert_eq!(candidates.len(), 1);
        assert_eq!(
            candidates[0]
                .payload
                .get("tool_name")
                .and_then(|v| v.as_str()),
            Some("get_current_time")
        );

        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM deferred_queue WHERE conversation_id = ? AND item_type = 'tool_call'",
        )
        .bind("conv")
        .fetch_one(&kernel.db.pool)
        .await
        .expect("count deferred after replay");
        assert_eq!(count, 0);
    }

    #[test]
    fn goal_loop_advances_only_with_evidence() {
        let mut goal_stack = vec![GoalStackItem {
            goal: "Ship the prototype".to_string(),
            steps: vec![
                GoalStep {
                    text: "Draft spec".to_string(),
                    ..Default::default()
                },
                GoalStep {
                    text: "Review spec".to_string(),
                    ..Default::default()
                },
            ],
            current_step_index: 0,
            status: None,
            evidence_event_ids: Vec::new(),
            belief_ids: Vec::new(),
            updated_at: None,
        }];
        let now = "2026-03-19T00:00:00Z";

        assert!(apply_goal_loop_tick(&mut goal_stack, now).is_none());
        assert_eq!(goal_stack[0].current_step_index, 0);

        goal_stack[0].steps[0].evidence_event_ids.push(1);
        assert!(apply_goal_loop_tick(&mut goal_stack, now).is_some());
        assert_eq!(goal_stack[0].current_step_index, 1);
        assert_eq!(
            goal_stack[0].steps[0].status.as_deref(),
            Some("completed")
        );

        goal_stack[0].steps[1].evidence_event_ids.push(2);
        assert!(apply_goal_loop_tick(&mut goal_stack, now).is_some());
        assert_eq!(goal_stack[0].current_step_index, 2);
        assert_eq!(goal_stack[0].status.as_deref(), Some("completed"));
    }

    #[test]
    fn plan_state_revised_on_world_model_conflict() {
        let candidate = make_candidate(
            CandidateKind::ToolCall,
            json!({ "assumptions": ["foo dependency"] }),
        );
        let proposal = subject_controller::build_action_proposal(&candidate);
        let mut snapshot = WorldModelSnapshot::default();
        snapshot.conflict_count = 1;
        snapshot.conflicts = vec![WorldModelConflict {
            id: 1,
            topic_key: "foo".to_string(),
            status: "open".to_string(),
            priority: "high".to_string(),
            resolution_note: None,
            member_belief_ids: Vec::new(),
            updated_at: Utc::now().to_rfc3339(),
        }];
        let verification = subject_controller::verify_action_proposal(&proposal, &snapshot);
        assert_eq!(verification.outcome, "DEFER");
        assert_eq!(
            super::arbitration::plan_state_for_verification(&verification),
            "revised"
        );
    }

    #[test]
    fn enforce_trailing_system_tags_moves_tags_to_end() {
        let input = "Hello\n<<MEMORY>>\nMore\n<<CLARIFY>>";
        let (with_tags, cleaned, tags) = enforce_trailing_system_tags(input);
        assert!(tags.memory);
        assert!(tags.clarify);
        assert!(!cleaned.contains("<<MEMORY>>"));
        assert!(!cleaned.contains("<<CLARIFY>>"));
        let trimmed = with_tags.trim_end();
        assert!(
            trimmed.ends_with("<<MEMORY>>\n<<CLARIFY>>")
                || trimmed.ends_with("<<MEMORY>>\r\n<<CLARIFY>>")
        );
    }

    #[tokio::test]
    async fn qualia_snapshot_creates_evidence_event() {
        let pool = setup_pool().await;
        let db = Db { pool };
        let _ = db.ensure_self_model_row().await;
        let qualia_state = qualia::compute_qualia_state(&db, None).await.unwrap();
        let snapshot = format_qualia_snapshot(&qualia_state);
        let event_id = db
            .create_qualia_snapshot_evidence_event("default", &snapshot, Some("test"))
            .await
            .expect("qualia evidence event");
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM ics_evidence_events WHERE id = ? AND source_type = 'qualia_snapshot'",
        )
        .bind(event_id)
        .fetch_one(&db.pool)
        .await
        .unwrap_or(0);
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn reflective_narrative_injected_into_prompt() {
        let pool = setup_pool().await;
        let db = Db { pool };
        let input = CorePromptInput {
            content: "test".to_string(),
            kind: CoreInputKind::User,
            source: "user".to_string(),
            self_awareness: false,
            self_awareness_hint: false,
            anchor_hits: 1,
            original_input: "test".to_string(),
            current_time: None,
            semantic_hint: None,
            introspection_summary: None,
            monologue_intent: None,
            monologue_digest: None,
            prompt_mode: None,
            task_phase: None,
            missing_slots: None,
            resolution_mode: None,
            policy_notes: None,
            redirect_focus: None,
            allow_diagnostics: false,
            world_model_snapshot: None,
            subject_snapshot: None,
            gate_decision: None,
            feedback_bundle: None,
            qualia_snapshot: None,
            wave_state: None,
            attention_schema_summary: None,
            workspace_contributors_summary: None,
            reflective_narrative: Some("I notice a skeptical posture.".to_string()),
            reflective_narrative_evidence_ids: vec![42],
            hydrated_context: None,
        };
        let build = build_core_system_message_with_layout(&db, "default", &input, PromptLayout::Full)
            .await
            .expect("prompt build");
        assert!(build.system_message.contains("Reflective Narrative"));
        assert!(build.system_message.contains("skeptical posture"));
    }

    #[tokio::test]
    async fn unified_self_tool_returns_snapshot() {
        let pool = setup_pool().await;
        let db = Db { pool };
        let _ = db.ensure_self_model_row().await;
        let state = KernelState::default_for("default");
        self_model_controller::update_unified_self_model(&db, &state)
            .await
            .expect("update unified self");
        let mut cancel_rx = watch::channel(false).1;
        let registry = ToolRegistry;
        let result = registry
            .execute(
                &db,
                "get_unified_self",
                "{\"conversation_id\":\"default\"}",
                &mut cancel_rx,
                None,
                None,
            )
            .await
            .expect("tool exec");
        let parsed: Value = serde_json::from_str(&result).expect("json parse");
        assert!(parsed.get("unified_state").is_some());
        let unified = parsed.get("unified_state").unwrap();
        assert!(unified.get("qualia_snapshot").is_some());
        assert!(unified.get("autobiographical_summary").is_some());
    }

    #[tokio::test]
    async fn reconcile_emits_world_model_events_after_conflict_injection() {
        let pool = setup_pool().await;
        let ctx = WriteContext {
            pool: pool.clone(),
            model_client: None,
            scope: Scope::SelfScope,
            source: SourceType::System,
            source_ref: Some("test_conflict".to_string()),
            now: Utc::now(),
            embedding_config: None,
            conversation_id: Some("conv".to_string()),
        };
        let subject_id = writer::create_entity("assistant", Some("system"), &ctx)
            .await
            .expect("entity");

        let stmt_a = FactStmt {
            subject: Ref::Handle("assistant".to_string()),
            key: "system_status".to_string(),
            value: "online".to_string(),
            value_quoted: false,
            certainty: Some(0.9),
            time_expr: None,
            scope_expr: None,
            source_ref: None,
            polarity: "assert".to_string(),
        };
        let stmt_b = FactStmt {
            subject: Ref::Handle("assistant".to_string()),
            key: "system_status".to_string(),
            value: "offline".to_string(),
            value_quoted: false,
            certainty: Some(0.2),
            time_expr: None,
            scope_expr: None,
            source_ref: None,
            polarity: "deny".to_string(),
        };

        let belief_a = match writer::write_fact(stmt_a, subject_id, &ctx).await {
            WriteResult::Inserted(id) | WriteResult::Updated(id) => id,
            WriteResult::Conflict { belief_id, .. } => belief_id,
            WriteResult::Ignored(reason) => panic!("unexpected ignored: {}", reason),
            WriteResult::Error(err) => panic!("write_fact failed: {}", err),
        };
        let belief_b = match writer::write_fact(stmt_b, subject_id, &ctx).await {
            WriteResult::Inserted(id) | WriteResult::Updated(id) => id,
            WriteResult::Conflict { belief_id, .. } => belief_id,
            WriteResult::Ignored(reason) => panic!("unexpected ignored: {}", reason),
            WriteResult::Error(err) => panic!("write_fact failed: {}", err),
        };

        let conflict_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ics_conflict_sets")
            .fetch_one(&pool)
            .await
            .unwrap_or(0);
        assert!(conflict_count > 0, "expected conflict set creation");

        let _ = sqlx::query(
            "UPDATE ics_beliefs SET confidence = ?, evidence_weight_total = ? WHERE id = ?",
        )
        .bind(0.9f64)
        .bind(1.8f64)
        .bind(belief_a)
        .execute(&pool)
        .await;
        let _ = sqlx::query(
            "UPDATE ics_beliefs SET confidence = ?, evidence_weight_total = ? WHERE id = ?",
        )
        .bind(0.2f64)
        .bind(0.2f64)
        .bind(belief_b)
        .execute(&pool)
        .await;

        let _ = sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, created_at)
             VALUES (?, 'user', 'test', 'evidence_a', 1.0, CURRENT_TIMESTAMP)",
        )
        .bind(belief_a)
        .execute(&pool)
        .await;
        let _ = sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, created_at)
             VALUES (?, 'tool', 'test', 'evidence_b', 1.0, CURRENT_TIMESTAMP)",
        )
        .bind(belief_a)
        .execute(&pool)
        .await;

        let report = reconcile_conflict_sets(&pool, "conv", WorldModelReconcileMode::Active)
            .await
            .expect("reconcile");
        assert!(report.scanned > 0);

        let events_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM world_model_events")
            .fetch_one(&pool)
            .await
            .unwrap_or(0);
        assert!(events_count > 0, "expected world_model_events after reconcile");
    }
