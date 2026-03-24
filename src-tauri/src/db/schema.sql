-- symbiote schema v0

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    started_at DATETIME NOT NULL,
    heartbeat_at DATETIME,
    ended_at DATETIME,
    status TEXT NOT NULL, -- 'active', 'complete', 'error', 'cancelled', 'superseded'
    superseded_by_run_id TEXT,
    metadata TEXT, -- JSON (includes response_origin for assistant messages)
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    run_id TEXT,
    trace_id TEXT,
    role TEXT NOT NULL, -- 'system', 'user', 'assistant', 'internal'
    content TEXT NOT NULL,
    status TEXT NOT NULL, -- 'complete', 'streaming', 'error', 'cancelled'
    error TEXT, -- JSON
    created_at DATETIME NOT NULL,
    metadata TEXT, -- JSON
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TRIGGER IF NOT EXISTS trg_messages_touch_conversation
AFTER INSERT ON messages
BEGIN
    UPDATE conversations
    SET updated_at = CURRENT_TIMESTAMP
    WHERE conversation_id = NEW.conversation_id;
END;

CREATE TABLE IF NOT EXISTS conversation_summaries (
    conversation_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL DEFAULT '',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 0,
    pending INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    last_error_at DATETIME,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS conversation_live_summaries (
    conversation_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL DEFAULT '',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 0,
    pending INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    last_error_at DATETIME,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS conversation_weekly_summaries (
    conversation_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL DEFAULT '',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS conversation_summary_chunks (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    start_ts DATETIME,
    end_ts DATETIME,
    start_message_id TEXT,
    end_message_id TEXT,
    source_summary_version INTEGER,
    summary_hash TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    type TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    payload TEXT NOT NULL, -- JSON
    produced_by TEXT NOT NULL,
    parent_artifact_ids TEXT, -- JSON
    created_at DATETIME NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS system_logs (
    id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    level TEXT NOT NULL, -- info, warn, error, debug
    category TEXT NOT NULL, -- model, tool, memory, scheduler, pipeline, system, ui
    run_id TEXT,
    trace_id TEXT,
    payload TEXT NOT NULL -- JSON
);

CREATE TABLE IF NOT EXISTS system_controls (
    control_id TEXT PRIMARY KEY,
    subsystem_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    value_json TEXT,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by TEXT,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS system_control_events (
    event_id TEXT PRIMARY KEY,
    subsystem_id TEXT NOT NULL,
    previous_mode TEXT,
    new_mode TEXT NOT NULL,
    value_json TEXT,
    actor TEXT,
    reason TEXT,
    status TEXT,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS system_health_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    run_id TEXT,
    trace_id TEXT,
    metrics_json TEXT NOT NULL,
    subsystem_states_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_ledger (
    event_id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    type TEXT NOT NULL, -- user_message | tool_output | system_transition | feedback
    payload TEXT NOT NULL, -- JSON
    tags TEXT, -- JSON
    run_id TEXT,
    trace_id TEXT
);

CREATE TABLE IF NOT EXISTS memory_write_ledger (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    category TEXT NOT NULL, -- summary|inner_summary|episodic|semantic|semantic_core|memory_pass
    source TEXT NOT NULL, -- kernel|scheduler|model_client|memory_writer|self_reflection
    reason_code TEXT NOT NULL,
    run_id TEXT,
    trace_id TEXT,
    payload_hash TEXT,
    snapshot_hash TEXT,
    gate_decision_id TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memory_pass_tokens (
    run_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS post_processing_jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    conversation_id TEXT,
    run_id TEXT,
    priority INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'queued', -- queued | running | completed | failed
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    ended_at DATETIME,
    error TEXT
);

CREATE TABLE IF NOT EXISTS deferred_emits (
    emit_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    emit_kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    source TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS kernel_states (
    conversation_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    state_write_owner TEXT,
    monologue_write_version INTEGER NOT NULL DEFAULT 0,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS proaction_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state_json TEXT NOT NULL,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS inner_summaries (
    conversation_id TEXT PRIMARY KEY,
    summary_json TEXT NOT NULL DEFAULT '{}',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    last_error_at DATETIME,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS inner_monologue_entries (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    run_id TEXT,
    dialogue_id TEXT,
    turn_index INTEGER,
    speaker TEXT, -- self_a | self_b
    mode TEXT NOT NULL, -- play | work
    stream_type TEXT NOT NULL DEFAULT 'DS', -- FTS | DS
    thought TEXT NOT NULL,
    descriptors_json TEXT,
    harvest_type TEXT,
    harvest_payload TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS inner_monologue_candidates (
    id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL,
    candidate_id TEXT,
    outcome TEXT, -- accepted | rejected | executed
    suppression_reason TEXT,
    candidate_json TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (entry_id) REFERENCES inner_monologue_entries(id)
);

CREATE TABLE IF NOT EXISTS subject_snapshots (
    snapshot_hash TEXT PRIMARY KEY,
    snapshot_version TEXT NOT NULL,
    tick_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    run_id TEXT,
    timestamp DATETIME NOT NULL,
    subject_state_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS action_proposals (
    proposal_id TEXT PRIMARY KEY,
    snapshot_hash TEXT NOT NULL,
    intent TEXT NOT NULL,
    steps_json TEXT NOT NULL,
    plan_hash TEXT NOT NULL DEFAULT '',
    plan_state TEXT NOT NULL DEFAULT 'draft', -- draft | verified | active | revised | completed
    risk_level TEXT NOT NULL,
    required_claims_json TEXT NOT NULL,
    required_error_bounds_json TEXT,
    verification_plan_json TEXT,
    success_criteria_json TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (snapshot_hash) REFERENCES subject_snapshots(snapshot_hash)
);

CREATE TABLE IF NOT EXISTS gate_decisions (
    decision_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL,
    decision TEXT NOT NULL, -- ALLOW | VERIFY | DEFER | DENY
    evidence_refs_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (proposal_id) REFERENCES action_proposals(proposal_id),
    FOREIGN KEY (snapshot_hash) REFERENCES subject_snapshots(snapshot_hash)
);

-- ============================================================
-- Evidence Lineage (Phase 1)
-- ============================================================

CREATE TABLE IF NOT EXISTS evidence_sources (
    evidence_id TEXT PRIMARY KEY,
    source_table TEXT NOT NULL, -- ics_evidence_events | self_evidence_events | other
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL, -- user | tool | system | inference | identity_statement | capability_statement
    source_ref TEXT,
    snippet TEXT,
    weight REAL,
    confidence REAL,
    run_id TEXT,
    trace_id TEXT,
    conversation_id TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_sources_origin ON evidence_sources(source_table, source_id);
CREATE INDEX IF NOT EXISTS idx_evidence_sources_time ON evidence_sources(created_at DESC);

CREATE TABLE IF NOT EXISTS evidence_links (
    link_id TEXT PRIMARY KEY,
    evidence_id TEXT NOT NULL,
    target_type TEXT NOT NULL, -- belief | self_belief | decision_report | message | tool_dispatch
    target_id TEXT NOT NULL,
    relation TEXT DEFAULT 'supports',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (evidence_id) REFERENCES evidence_sources(evidence_id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_links_target ON evidence_links(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_evidence_links_evidence ON evidence_links(evidence_id);

CREATE TABLE IF NOT EXISTS decision_reports (
    report_id TEXT PRIMARY KEY,
    run_id TEXT,
    trace_id TEXT,
    conversation_id TEXT,
    report_json TEXT NOT NULL,
    evidence_event_ids TEXT NOT NULL DEFAULT '[]',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_decision_reports_run ON decision_reports(run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decision_reports_conv ON decision_reports(conversation_id, created_at DESC);


CREATE TABLE IF NOT EXISTS residual_vectors (
    residual_id TEXT PRIMARY KEY,
    prediction_id TEXT NOT NULL,
    outcome_id TEXT NOT NULL,
    residual_value REAL NOT NULL,
    normalized_residual REAL NOT NULL,
    salience_score REAL NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES self_predictions(id),
    FOREIGN KEY (outcome_id) REFERENCES self_prediction_outcomes(id)
);

CREATE TABLE IF NOT EXISTS error_events (
    error_event_id TEXT PRIMARY KEY,
    residual_id TEXT NOT NULL,
    linked_claims_json TEXT NOT NULL,
    classification TEXT NOT NULL,
    status TEXT NOT NULL, -- OPEN | MITIGATED | RESOLVED
    recommended_actions_json TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (residual_id) REFERENCES residual_vectors(residual_id)
);

CREATE TABLE IF NOT EXISTS introspection_entries (
    entry_id TEXT PRIMARY KEY,
    snapshot_hash TEXT NOT NULL,
    workspace_refs_json TEXT NOT NULL,
    event_refs_json TEXT NOT NULL,
    prediction_refs_json TEXT,
    error_refs_json TEXT,
    numeric_payload_json TEXT NOT NULL,
    narrative TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (snapshot_hash) REFERENCES subject_snapshots(snapshot_hash)
);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id TEXT PRIMARY KEY,
    target_id TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL,
    checks_json TEXT NOT NULL,
    discrepancy_score REAL NOT NULL,
    recommended_action TEXT NOT NULL, -- NONE | LOWER_INTROSPECTION_WEIGHT | REQUIRE_VERIFY | REQUEST_USER_CLARIFICATION | REANCHOR | DIAGNOSE_LOOP
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (snapshot_hash) REFERENCES subject_snapshots(snapshot_hash)
);

CREATE TABLE IF NOT EXISTS calibration_changes (
    change_id TEXT PRIMARY KEY,
    snapshot_hash TEXT NOT NULL,
    knob TEXT NOT NULL,
    old_value REAL NOT NULL,
    new_value REAL NOT NULL,
    reason TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (snapshot_hash) REFERENCES subject_snapshots(snapshot_hash)
);

CREATE TABLE IF NOT EXISTS qualia_labels (
    label_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL,
    tag TEXT NOT NULL,
    intensity REAL NOT NULL,
    context_json TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (snapshot_hash) REFERENCES subject_snapshots(snapshot_hash)
);

CREATE TABLE IF NOT EXISTS qualia_reward_events (
    reward_id TEXT PRIMARY KEY,
    label_id TEXT NOT NULL,
    magnitude REAL NOT NULL,
    outcome_ref TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (label_id) REFERENCES qualia_labels(label_id)
);

CREATE TABLE IF NOT EXISTS counterfactual_simulations (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    run_id TEXT,
    candidate_id TEXT,
    candidate_kind TEXT,
    prompt TEXT NOT NULL,
    predicted_label TEXT,
    predicted_outcome TEXT,
    observed_label TEXT,
    matched INTEGER,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    observed_at DATETIME,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS tool_dispatches (
    action_id TEXT PRIMARY KEY,
    run_id TEXT,
    tool_name TEXT NOT NULL,
    args_json TEXT NOT NULL,
    plan_step_id TEXT, -- proposal_id:step_index (context-only tools omit)
    status TEXT NOT NULL, -- pending | success | failed
    attempts INTEGER NOT NULL DEFAULT 0,
    failure_kind TEXT, -- planning_error | execution_error | cancelled
    last_error TEXT,
    result_text TEXT,
    evidence_event_id INTEGER,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS outcome_events (
    outcome_id TEXT PRIMARY KEY,
    run_id TEXT,
    trace_id TEXT,
    candidate_id TEXT,
    target_type TEXT NOT NULL DEFAULT 'decision_report', -- decision_report | message | tool_dispatch
    verdict TEXT NOT NULL, -- confirm | disconfirm | inconclusive
    confidence REAL NOT NULL DEFAULT 0.5,
    source TEXT NOT NULL DEFAULT 'operator',
    note TEXT,
    evidence_event_ids TEXT NOT NULL DEFAULT '[]',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_outcome_events_run ON outcome_events(run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_outcome_events_candidate ON outcome_events(candidate_id, created_at DESC);

CREATE TABLE IF NOT EXISTS thread_runs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    parent_run_id TEXT,
    goal TEXT NOT NULL,
    context_json TEXT NOT NULL,
    outcome_summary TEXT,
    status TEXT NOT NULL DEFAULT 'running', -- running | completed | failed
    depth INTEGER NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version INTEGER NOT NULL,
    api_base_url TEXT NOT NULL,
    api_key TEXT,
    streaming_enabled BOOLEAN NOT NULL DEFAULT 1,
    history_window INTEGER NOT NULL DEFAULT 3,
    injection_policy TEXT NOT NULL DEFAULT 'include',
    request_defaults TEXT, -- JSON
    active_model_id TEXT,
    json_reliable_model_id TEXT,
    system_prompt TEXT,
    user_display_name TEXT,
    assistant_display_name TEXT,
    onboarding_completed BOOLEAN NOT NULL DEFAULT 0,
    ui_theme TEXT DEFAULT 'builtin:utopia',
    voice_name TEXT DEFAULT 'bf_isabella',
    voice_speed REAL DEFAULT 1.0,
    voice_reference_audio TEXT,
    voice_reference_text TEXT,
    voice_quality_preset TEXT,
    voice_cfg_value REAL,
    voice_denoiser_enabled BOOLEAN,
    voice_temperature REAL,
    voice_pitch_semitones REAL DEFAULT 1,
    voice_reverb_amount REAL DEFAULT 0.15,
    voice_compression REAL DEFAULT 0.05,
    voice_formant_shift REAL,
    summarization_api_url TEXT,
    summarization_model TEXT DEFAULT 'summarizer',
    embedding_model TEXT,
    trace_history_limit INTEGER DEFAULT 10,
    cockpit_write_enabled BOOLEAN NOT NULL DEFAULT 0,
    episodic_enabled BOOLEAN NOT NULL DEFAULT 1,
    episodic_injection_enabled BOOLEAN NOT NULL DEFAULT 1,
    episodic_compaction_enabled BOOLEAN NOT NULL DEFAULT 1,
    episodic_injection_limit INTEGER NOT NULL DEFAULT 5,
    episodic_opt_out BOOLEAN NOT NULL DEFAULT 0,
    memory_claims_enabled BOOLEAN NOT NULL DEFAULT 1,
    phi_consent BOOLEAN NOT NULL DEFAULT 0,
    seed_personal_user BOOLEAN NOT NULL DEFAULT 1,
    lexical_fallback_enabled BOOLEAN NOT NULL DEFAULT 1,
    memory_half_life_hours REAL NOT NULL DEFAULT 168,
    research_budget_per_hour INTEGER NOT NULL DEFAULT 6,
    research_budget_reset_window INTEGER NOT NULL DEFAULT 60,
    research_cost_per_call INTEGER NOT NULL DEFAULT 1,
    monologue_interval_seconds INTEGER NOT NULL DEFAULT 20,
    monologue_timeout_secs INTEGER NOT NULL DEFAULT 75,
    monologue_retry_timeout_secs INTEGER NOT NULL DEFAULT 25,
    empty_response_retry_max INTEGER NOT NULL DEFAULT 1,
    empty_response_retry_timeout_ms INTEGER NOT NULL DEFAULT 2000,
    monologue_max_per_hour INTEGER NOT NULL DEFAULT 360,
    thread_max_depth INTEGER NOT NULL DEFAULT 4,
    allow_shell_tool BOOLEAN NOT NULL DEFAULT 0,
    shell_command_allowlist TEXT,
    ask_budget_max INTEGER NOT NULL DEFAULT 6,
    calculator_followups_max INTEGER NOT NULL DEFAULT 0,
    loop_similarity_threshold REAL NOT NULL DEFAULT 0.90,
    loop_recent_k INTEGER NOT NULL DEFAULT 6,
    meta_cog_outcome_turns INTEGER NOT NULL DEFAULT 3,
    meta_cog_cycle_window_turns INTEGER NOT NULL DEFAULT 2,
    meta_cog_outcome_timeout_s INTEGER NOT NULL DEFAULT 120,
    meta_cog_cooldown_s INTEGER NOT NULL DEFAULT 60,
    meta_cog_streak_limit INTEGER NOT NULL DEFAULT 3,
    registry_profile_name TEXT DEFAULT 'default',
    controller_enabled BOOLEAN NOT NULL DEFAULT 1,
    monologue_stabilization_enabled BOOLEAN NOT NULL DEFAULT 1,
    monologue_surface_enabled BOOLEAN NOT NULL DEFAULT 1,
    show_monologue_in_chat BOOLEAN NOT NULL DEFAULT 1,
    enable_introspection BOOLEAN NOT NULL DEFAULT 1,
    heartbeat_enabled BOOLEAN NOT NULL DEFAULT 1,
    dream_enabled BOOLEAN NOT NULL DEFAULT 1,
    binding_enforcement_enabled BOOLEAN NOT NULL DEFAULT 1,
    pending_prompt_alignment_enabled BOOLEAN NOT NULL DEFAULT 1,
    pending_prompt_recency_secs INTEGER NOT NULL DEFAULT 90,
    auto_memory_pass_enabled BOOLEAN NOT NULL DEFAULT 1,
    summary_cohesion_enabled BOOLEAN NOT NULL DEFAULT 1,
    compact_prompt_enabled BOOLEAN NOT NULL DEFAULT 0,
    context_hydration_mode TEXT NOT NULL DEFAULT 'shadow',
    context_budgeter_enabled BOOLEAN NOT NULL DEFAULT 1,
    context_miss_detector_enabled BOOLEAN NOT NULL DEFAULT 1,
    world_model_reconcile_mode TEXT NOT NULL DEFAULT 'shadow',
    goal_loop_enabled BOOLEAN NOT NULL DEFAULT 1,
    goal_loop_interval_turns INTEGER NOT NULL DEFAULT 3,
    goal_loop_load_threshold_ms INTEGER NOT NULL DEFAULT 650,
    json_only_disabled_models TEXT,
    tool_failure_gate_window_mins INTEGER,
    tool_failure_gate_tool_names TEXT,
    gate_default_soft BOOLEAN NOT NULL DEFAULT 1,
    gate_shadow_mode BOOLEAN NOT NULL DEFAULT 0,
    gate_rollout_percent INTEGER NOT NULL DEFAULT 100,
    self_report_channel BOOLEAN NOT NULL DEFAULT 1,
    self_awareness_expression_mode TEXT NOT NULL DEFAULT 'balanced',
    explicit_feedback_only BOOLEAN NOT NULL DEFAULT 1,
    weight_user_satisfaction REAL NOT NULL DEFAULT 0.5,
    weight_policy_rigor REAL NOT NULL DEFAULT 0.5,
    weight_latency REAL NOT NULL DEFAULT 0.5,
    weight_evidence_strictness REAL NOT NULL DEFAULT 0.5,
    weight_exploration REAL NOT NULL DEFAULT 0.5,
    monologue_provenance_guard BOOLEAN NOT NULL DEFAULT 1,
    organism_decay BOOLEAN NOT NULL DEFAULT 1,
    model_context_limit INTEGER NOT NULL DEFAULT 16384,
    introspection_confidence_threshold REAL NOT NULL DEFAULT 0.5,
    introspection_drift_threshold REAL NOT NULL DEFAULT 0.6,
    introspection_ambiguity_threshold REAL NOT NULL DEFAULT 0.5,
    enable_attribution_gate BOOLEAN NOT NULL DEFAULT 1,
    enable_user_utterance_evidence BOOLEAN NOT NULL DEFAULT 1,
    enable_attribution_metadata BOOLEAN NOT NULL DEFAULT 1,
    enable_tool_schema_validation BOOLEAN NOT NULL DEFAULT 1,
    enable_context_evidence BOOLEAN NOT NULL DEFAULT 1,
    enable_monologue_validator BOOLEAN NOT NULL DEFAULT 1,
    enable_memory_evidence_gating BOOLEAN NOT NULL DEFAULT 1,
    enable_speculative_workspace_containment BOOLEAN NOT NULL DEFAULT 1,
    stability_prompt_override_guard BOOLEAN NOT NULL DEFAULT 1,
    stability_monologue_tagged BOOLEAN NOT NULL DEFAULT 1,
    stability_introspection_structured BOOLEAN NOT NULL DEFAULT 1,
    stability_disable_working_hypothesis BOOLEAN NOT NULL DEFAULT 0,
    stability_state_disclosure_expanded BOOLEAN NOT NULL DEFAULT 1,
    stability_transcript_normalization BOOLEAN NOT NULL DEFAULT 1,
    stability_memory_hygiene BOOLEAN NOT NULL DEFAULT 1,
    stability_non_stream_sanitization BOOLEAN NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS phi_consent_scopes (
    conversation_id TEXT PRIMARY KEY,
    enabled BOOLEAN NOT NULL DEFAULT 0,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS context_tags (
    tag_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    inferred BOOLEAN NOT NULL DEFAULT 0,
    evidence_event_ids TEXT NOT NULL DEFAULT '[]',
    source TEXT,
    last_seen_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (conversation_id, tag)
);

CREATE TABLE IF NOT EXISTS user_intent_summaries (
    summary_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    confirmed BOOLEAN NOT NULL DEFAULT 0,
    evidence_event_ids TEXT NOT NULL DEFAULT '[]',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id),
    UNIQUE (conversation_id)
);

CREATE TABLE IF NOT EXISTS parameter_registry (
    id INTEGER PRIMARY KEY,
    profile_name TEXT NOT NULL,
    profile_version INTEGER NOT NULL DEFAULT 1,
    payload_json TEXT NOT NULL,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS self_model (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    capabilities_json TEXT NOT NULL DEFAULT '[]',
    limitations_json TEXT NOT NULL DEFAULT '[]',
    active_tools_json TEXT NOT NULL DEFAULT '[]',
    memory_health_json TEXT NOT NULL DEFAULT '{}',
    persona_json TEXT NOT NULL DEFAULT '{}',
    persona_daily_delta_json TEXT NOT NULL DEFAULT '{}',
    persona_last_delta_date TEXT,
    goals_json TEXT NOT NULL DEFAULT '[]',
    identity_thread TEXT,
    identity_confidence REAL NOT NULL DEFAULT 0.5,
    identity_uncertainty_note TEXT,
    identity_updated_at TEXT,
    reflection_status_json TEXT NOT NULL DEFAULT '{}',
    reflection_frozen INTEGER NOT NULL DEFAULT 0,
    last_reflection_at TEXT,
    internal_state_summary_json TEXT NOT NULL DEFAULT '{}',
    internal_state_map_version INTEGER,
    unified_state_json TEXT NOT NULL DEFAULT '{}',
    unified_state_evidence_json TEXT NOT NULL DEFAULT '{}',
    unified_state_updated_at TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS internal_state_map (
    id INTEGER PRIMARY KEY,
    version INTEGER NOT NULL,
    metric TEXT NOT NULL,
    range_min REAL NOT NULL,
    range_max REAL NOT NULL,
    label TEXT NOT NULL,
    author TEXT,
    rationale TEXT,
    degraded INTEGER NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_internal_state_map_version ON internal_state_map(version);
CREATE INDEX IF NOT EXISTS idx_internal_state_map_metric ON internal_state_map(metric);

CREATE TABLE IF NOT EXISTS telemetry_calibrations (
    id INTEGER PRIMARY KEY,
    metric TEXT NOT NULL,
    observed_rate REAL NOT NULL,
    expected_rate REAL,
    drift REAL NOT NULL,
    sample_count INTEGER NOT NULL,
    window_minutes INTEGER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS identity_snapshots (
    id TEXT PRIMARY KEY,
    snapshot_json TEXT NOT NULL,
    evidence_event_ids TEXT NOT NULL, -- JSON array
    invariants_json TEXT, -- JSON
    reason TEXT,
    source TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS self_model_controller_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS self_model_controller_snapshots (
    id INTEGER PRIMARY KEY,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS internal_state_snapshots (
    id INTEGER PRIMARY KEY,
    run_id TEXT,
    message_id TEXT,
    conversation_id TEXT,
    confidence REAL NOT NULL,
    uncertainty REAL NOT NULL,
    qualia_tag TEXT,
    qualia_intensity REAL,
    internal_state_summary_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS self_model_checkpoints (
    id INTEGER PRIMARY KEY,
    snapshot_json TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS self_reflection_staging (
    id TEXT PRIMARY KEY,
    proposal_json TEXT NOT NULL,
    evidence_event_ids TEXT, -- JSON array
    status TEXT NOT NULL DEFAULT 'pending', -- pending|approved|rejected|applied
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TEXT,
    reviewed_by TEXT
);

CREATE TABLE IF NOT EXISTS self_goal_evidence (
    id INTEGER PRIMARY KEY,
    goal TEXT NOT NULL,
    evidence_snippet TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kv_store (
    key TEXT PRIMARY KEY,
    value TEXT,
    keywords TEXT,
    is_critical BOOLEAN DEFAULT 0,
    evidence_event_id INTEGER,
    updated_at DATETIME NOT NULL
);

-- ============================================================
-- Self Predictions (Phase 7)
-- ============================================================

CREATE TABLE IF NOT EXISTS self_predictions (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    trace_id TEXT,
    metric TEXT NOT NULL,
    context_ref_json TEXT,
    predicted_target_type TEXT,
    expected_value REAL NOT NULL,
    expected_variance REAL NOT NULL DEFAULT 0.0,
    expected_bounds_json TEXT,
    horizon TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    evidence_event_ids TEXT NOT NULL DEFAULT '[]',
    linked_claims_json TEXT,
    normalization_contract_id TEXT,
    salience_hint REAL,
    rejection_reason TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_self_predictions_created ON self_predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_self_predictions_metric ON self_predictions(metric);

CREATE TABLE IF NOT EXISTS self_prediction_outcomes (
    id TEXT PRIMARY KEY,
    prediction_id TEXT NOT NULL,
    observed_value REAL NOT NULL,
    delta REAL NOT NULL,
    z_score REAL NOT NULL,
    significant INTEGER NOT NULL DEFAULT 0,
    evidence_refs_json TEXT,
    observed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES self_predictions(id)
);

CREATE INDEX IF NOT EXISTS idx_self_prediction_outcomes_pred ON self_prediction_outcomes(prediction_id);

-- Semantic Core: Single row containing compressed user model
CREATE TABLE IF NOT EXISTS semantic_core (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    content TEXT NOT NULL DEFAULT '',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 0
);

-- Consolidation State: Tracks when to run summarization
CREATE TABLE IF NOT EXISTS consolidation_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_run DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    entries_since INTEGER NOT NULL DEFAULT 0,
    summary TEXT
);

CREATE TABLE IF NOT EXISTS reminders (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    due_at INTEGER NOT NULL,  -- Unix timestamp
    type TEXT NOT NULL, -- 'ALARM', 'REMINDER', 'CONVERSATION'
    status TEXT NOT NULL, -- 'PENDING', 'COMPLETED', 'SNOOZED'
    created_at INTEGER NOT NULL  -- Unix timestamp
);

-- ============================================================
-- Self Memory (Internal)
-- ============================================================

CREATE TABLE IF NOT EXISTS self_beliefs (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL, -- fact, rel
    scope TEXT NOT NULL DEFAULT 'self',
    status TEXT NOT NULL DEFAULT 'active',
    confidence REAL NOT NULL DEFAULT 1.0,
    evidence_weight_total REAL NOT NULL DEFAULT 0.0,
    observed_at TEXT NOT NULL,
    last_evidence_at TEXT NOT NULL,
    last_validated_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS self_fact_beliefs (
    belief_id INTEGER PRIMARY KEY,
    key TEXT NOT NULL,
    value_literal TEXT NOT NULL,
    value_hash TEXT NOT NULL,
    FOREIGN KEY (belief_id) REFERENCES self_beliefs(id)
);

CREATE TABLE IF NOT EXISTS self_rel_beliefs (
    belief_id INTEGER PRIMARY KEY,
    rel_type TEXT NOT NULL,
    participants_canonical TEXT NOT NULL,
    anchor_signature TEXT NOT NULL,
    FOREIGN KEY (belief_id) REFERENCES self_beliefs(id)
);

CREATE TABLE IF NOT EXISTS self_rel_participants (
    belief_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    label TEXT NOT NULL,
    FOREIGN KEY (belief_id) REFERENCES self_beliefs(id)
);

CREATE TABLE IF NOT EXISTS self_evidence_events (
    id INTEGER PRIMARY KEY,
    belief_id INTEGER NOT NULL,
    source_type TEXT NOT NULL, -- system only
    snippet TEXT NOT NULL,
    weight REAL NOT NULL,
    source_evidence_ids TEXT, -- JSON array of ics_evidence_events ids
    episodic_event_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (belief_id) REFERENCES self_beliefs(id)
);

-- ============================================================
-- Self Claims (Audit Trail)
-- ============================================================

CREATE TABLE IF NOT EXISTS self_claims (
    id TEXT PRIMARY KEY,
    claim_text TEXT NOT NULL,
    claim_key TEXT NOT NULL,
    evidence_event_ids TEXT, -- JSON array of numeric IDs
    belief_ids TEXT, -- JSON array of numeric IDs
    confidence REAL NOT NULL DEFAULT 1.0,
    polarity TEXT NOT NULL DEFAULT 'assert',
    provisional INTEGER NOT NULL DEFAULT 0,
    source_type TEXT,
    requires_validation INTEGER NOT NULL DEFAULT 0,
    ttl_seconds INTEGER,
    promotion_rule TEXT,
    eviction_rule TEXT,
    expires_at DATETIME,
    source_run_id TEXT,
    conversation_id TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_self_claims_key ON self_claims(claim_key);
CREATE INDEX IF NOT EXISTS idx_self_claims_created ON self_claims(created_at DESC);

-- ============================================================
-- ICS v4.1 Tables (Spec v4.1)
-- ============================================================

-- 1.1 Entities
CREATE TABLE IF NOT EXISTS ics_entities (
    id INTEGER PRIMARY KEY,
    label TEXT NOT NULL,
    label_canonical TEXT NOT NULL,
    entity_type TEXT, -- Person, Place, Work, etc.
    aliases TEXT, -- JSON array of strings
    aliases_canonical TEXT, -- JSON array
    keys TEXT, -- JSON array
    resolution_state TEXT NOT NULL DEFAULT 'normal', -- normal, tentative, do_not_merge
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER NOT NULL DEFAULT 0
);

-- 1.2 Beliefs (Base)
CREATE TABLE IF NOT EXISTS ics_beliefs (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL, -- fact, rel
    scope TEXT NOT NULL, -- global, session, project:x, context:x
    status TEXT NOT NULL DEFAULT 'active', -- active, inactive
    reconcile_state TEXT NOT NULL DEFAULT 'active', -- active | contested | retiring | retired
    reconcile_reason TEXT,
    reconcile_updated_at DATETIME,
    reconcile_run_id TEXT,
    reconcile_demoted_runs INTEGER NOT NULL DEFAULT 0,
    layer TEXT NOT NULL DEFAULT 'episodic', -- working, episodic, semantic, world
    polarity TEXT NOT NULL DEFAULT 'assert', -- assert, deny
    confidence REAL NOT NULL DEFAULT 1.0,
    salience REAL NOT NULL DEFAULT 1.0,
    topic_key TEXT NOT NULL,
    signature_hash TEXT NOT NULL,
    evidence_weight_total REAL NOT NULL DEFAULT 0.0,
    last_evidence_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_validated_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER NOT NULL DEFAULT 0,
    -- Time Model (§4)
    time_bucket_kind TEXT NOT NULL DEFAULT 'atemporal',
    time_bucket_value TEXT,
    observed_at DATETIME,
    valid_from DATETIME,
    valid_to DATETIME
);

-- 1.2.1 Fact Beliefs
CREATE TABLE IF NOT EXISTS ics_fact_beliefs (
    belief_id INTEGER PRIMARY KEY,
    subject_entity_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value_literal TEXT NOT NULL,
    value_hash TEXT NOT NULL,
    FOREIGN KEY (belief_id) REFERENCES ics_beliefs(id),
    FOREIGN KEY (subject_entity_id) REFERENCES ics_entities(id)
);

-- 1.2.2 Rel Beliefs
CREATE TABLE IF NOT EXISTS ics_rel_beliefs (
    belief_id INTEGER PRIMARY KEY,
    rel_type_id TEXT,
    rel_type TEXT NOT NULL,
    rel_type_norm TEXT NOT NULL,
    rel_type_raw TEXT,
    participants_canonical TEXT NOT NULL, -- Serialized string for hashing
    participants_ordered TEXT,           -- Preserves original participant order for directed relations
    anchor_signature TEXT NOT NULL,       -- Serialized string for ONE uniqueness
    direction TEXT,
    FOREIGN KEY (belief_id) REFERENCES ics_beliefs(id)
);

-- 1.1.1 Entity Alias Proposals (Untrusted)
CREATE TABLE IF NOT EXISTS ics_entity_aliases (
    entity_id INTEGER NOT NULL,
    alias TEXT NOT NULL,
    alias_canonical TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'proposed', -- proposed, confirmed, rejected
    evidence_count INTEGER NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_id, alias_canonical),
    FOREIGN KEY (entity_id) REFERENCES ics_entities(id)
);

-- Rel Participants (N-ary)
CREATE TABLE IF NOT EXISTS ics_rel_participants (
    belief_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    FOREIGN KEY (belief_id) REFERENCES ics_beliefs(id),
    FOREIGN KEY (entity_id) REFERENCES ics_entities(id)
);

-- 1.3 Belief Links
CREATE TABLE IF NOT EXISTS ics_belief_links (
    from_id INTEGER NOT NULL,
    to_id INTEGER NOT NULL,
    link_type TEXT NOT NULL, -- supports, contradicts, supersedes, derived_from
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (from_id, to_id, link_type),
    FOREIGN KEY (from_id) REFERENCES ics_beliefs(id),
    FOREIGN KEY (to_id) REFERENCES ics_beliefs(id)
);

-- 1.4 Conflict Sets
CREATE TABLE IF NOT EXISTS ics_conflict_sets (
    id INTEGER PRIMARY KEY,
    topic_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open', -- open, resolved, archived
    priority TEXT NOT NULL DEFAULT 'normal', -- low, normal, high
    resolution_note TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ics_conflict_set_members (
    conflict_set_id INTEGER NOT NULL,
    belief_id INTEGER NOT NULL,
    PRIMARY KEY (conflict_set_id, belief_id),
    FOREIGN KEY (conflict_set_id) REFERENCES ics_conflict_sets(id),
    FOREIGN KEY (belief_id) REFERENCES ics_beliefs(id)
);

-- 1.4.1 World Model Events (belief reconciliation lineage)
CREATE TABLE IF NOT EXISTS world_model_events (
    event_id TEXT PRIMARY KEY,
    belief_id INTEGER NOT NULL,
    conflict_set_id INTEGER,
    event_type TEXT NOT NULL, -- promote | demote | retire | contest | restore
    prev_state TEXT,
    new_state TEXT,
    reason TEXT,
    evidence_event_ids TEXT NOT NULL DEFAULT '[]', -- JSON array
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (belief_id) REFERENCES ics_beliefs(id),
    FOREIGN KEY (conflict_set_id) REFERENCES ics_conflict_sets(id)
);

CREATE INDEX IF NOT EXISTS idx_world_model_events_belief ON world_model_events(belief_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_world_model_events_conflict ON world_model_events(conflict_set_id, created_at DESC);

-- 1.5 Token Aliases
CREATE TABLE IF NOT EXISTS ics_token_aliases (
    token_kind TEXT NOT NULL, -- fact_key, rel_type
    from_token TEXT NOT NULL,
    to_token TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'proposed', -- proposed, confirmed
    evidence_count INTEGER NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (token_kind, from_token, to_token)
);

-- 1.5.1 Rel Type Aliases (Phase 1)
CREATE TABLE IF NOT EXISTS ics_rel_type_aliases (
    alias TEXT PRIMARY KEY,
    rel_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'confirmed', -- confirmed | provisional | rejected
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 1.5.2 Relation Type Catalog (Phase 2)
CREATE TABLE IF NOT EXISTS rel_type (
    rel_type_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'provisional', -- canonical | provisional | deprecated
    embedding TEXT, -- JSON or vector payload
    merged_into TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rel_type_alias (
    alias TEXT PRIMARY KEY,
    rel_type_id TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'confirmed', -- confirmed | provisional | rejected
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (rel_type_id) REFERENCES rel_type(rel_type_id)
);

CREATE TABLE IF NOT EXISTS rel_shape (
    rel_type_id TEXT PRIMARY KEY,
    roles TEXT NOT NULL, -- JSON array
    anchor_roles TEXT NOT NULL, -- JSON array
    cardinality_override TEXT,
    commutative BOOLEAN NOT NULL DEFAULT 0,
    expected_arity INTEGER,
    status TEXT NOT NULL DEFAULT 'seeded', -- seeded | provisional
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (rel_type_id) REFERENCES rel_type(rel_type_id)
);

-- 1.6 Role Aliases
CREATE TABLE IF NOT EXISTS ics_role_aliases (
    from_role TEXT NOT NULL,
    to_role TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'proposed', -- proposed, confirmed
    evidence_count INTEGER NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (from_role, to_role)
);

-- 4.4 Token Policies
CREATE TABLE IF NOT EXISTS ics_token_policies (
    token TEXT PRIMARY KEY,
    cardinality TEXT NOT NULL DEFAULT 'MANY_SET',
    read_policy TEXT NOT NULL DEFAULT 'LIST',
    allow_future_in_current BOOLEAN NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 2.1 Predicate Registry (Legacy / Upgrade Support)
CREATE TABLE IF NOT EXISTS ics_predicate_registry (
    predicate_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    subject_type TEXT NOT NULL,
    object_mode TEXT NOT NULL, -- 'unary' | 'entity' | 'literal'
    value_type TEXT, -- 'bool' | 'int' | 'float' | 'string' | 'enum' | 'date' | 'datetime' | 'duration' | 'json'
    partition_key TEXT,
    conflict_policy TEXT NOT NULL, -- 'latest_wins_in_scope' | 'highest_confidence_wins' | 'validity_window_based' | 'must_resolve_manually'
    is_traversal_relevant INTEGER NOT NULL DEFAULT 1,
    is_deprecated INTEGER NOT NULL DEFAULT 0,
    replaced_by TEXT,
    created_at TEXT NOT NULL
);

-- 8.1 Relation Shapes 
CREATE TABLE IF NOT EXISTS ics_relation_shapes (
    rel_type TEXT PRIMARY KEY,
    roles TEXT NOT NULL, -- JSON array
    anchor_roles TEXT NOT NULL, -- JSON array
    cardinality_override TEXT,
    commutative BOOLEAN NOT NULL DEFAULT 0,
    expected_arity INTEGER,
    status TEXT NOT NULL DEFAULT 'seeded', -- seeded | provisional
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 6.3 Promotion Mappings
CREATE TABLE IF NOT EXISTS ics_promotion_maps (
    id INTEGER PRIMARY KEY,
    from_fact_key TEXT NOT NULL UNIQUE,
    to_rel_type TEXT NOT NULL,
    subject_role TEXT NOT NULL,
    value_role TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 1.7 Evidence Events
CREATE TABLE IF NOT EXISTS ics_evidence_events (
    id INTEGER PRIMARY KEY,
    belief_id INTEGER NOT NULL,
    source_type TEXT NOT NULL, -- user, tool, system, inference
    source_ref TEXT,
    snippet TEXT,
    snippet_hash TEXT,
    weight REAL NOT NULL,
    episodic_event_id TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (belief_id) REFERENCES ics_beliefs(id)
);

-- Evidence lineage triggers (defined after evidence tables exist)
CREATE TRIGGER IF NOT EXISTS trg_ics_evidence_events_lineage
AFTER INSERT ON ics_evidence_events
BEGIN
    INSERT OR IGNORE INTO evidence_sources
        (evidence_id, source_table, source_id, source_type, source_ref, snippet, weight, created_at)
    VALUES
        ('ics:' || NEW.id, 'ics_evidence_events', CAST(NEW.id AS TEXT), NEW.source_type, NEW.source_ref, NEW.snippet, NEW.weight, COALESCE(NEW.created_at, CURRENT_TIMESTAMP));

    INSERT OR IGNORE INTO evidence_links
        (link_id, evidence_id, target_type, target_id, relation, created_at)
    VALUES
        ('link:ics:' || NEW.id || ':belief:' || NEW.belief_id, 'ics:' || NEW.id, 'belief', CAST(NEW.belief_id AS TEXT), 'supports', COALESCE(NEW.created_at, CURRENT_TIMESTAMP));
END;

CREATE TRIGGER IF NOT EXISTS trg_self_evidence_events_lineage
AFTER INSERT ON self_evidence_events
BEGIN
    INSERT OR IGNORE INTO evidence_sources
        (evidence_id, source_table, source_id, source_type, source_ref, snippet, weight, created_at)
    VALUES
        ('self:' || NEW.id, 'self_evidence_events', CAST(NEW.id AS TEXT), NEW.source_type, NULL, NEW.snippet, NEW.weight, COALESCE(NEW.created_at, CURRENT_TIMESTAMP));

    INSERT OR IGNORE INTO evidence_links
        (link_id, evidence_id, target_type, target_id, relation, created_at)
    VALUES
        ('link:self:' || NEW.id || ':belief:' || NEW.belief_id, 'self:' || NEW.id, 'self_belief', CAST(NEW.belief_id AS TEXT), 'supports', COALESCE(NEW.created_at, CURRENT_TIMESTAMP));
END;

-- ============================================================
-- Episodic Events (Additive, Instrumentation Only)
-- ============================================================

CREATE TABLE IF NOT EXISTS episodic_events (
    id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    event_version INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    run_id TEXT,
    trace_id TEXT,
    conversation_id TEXT,
    scope TEXT,
    source_type TEXT NOT NULL, -- user, tool, system, inference
    source_ref TEXT,
    linked_belief_id INTEGER,
    linked_artifact_id TEXT
);

CREATE TABLE IF NOT EXISTS episodic_identity_index (
    episodic_event_id TEXT PRIMARY KEY,
    identity_relevance REAL NOT NULL DEFAULT 0.0,
    valence_tag TEXT,
    valence_intensity REAL,
    qualia_evidence_ids TEXT, -- JSON array
    narrative_thread_id TEXT,
    narrative_position INTEGER,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (episodic_event_id) REFERENCES episodic_events(id)
);

CREATE INDEX IF NOT EXISTS idx_epi_identity_relevance ON episodic_identity_index(identity_relevance DESC);
CREATE INDEX IF NOT EXISTS idx_epi_identity_thread ON episodic_identity_index(narrative_thread_id, narrative_position);

-- ============================================================
-- Strategy Traces + Procedural Memory (Phase 6)
-- ============================================================

CREATE TABLE IF NOT EXISTS strategy_traces (
    id TEXT PRIMARY KEY,
    features_json TEXT NOT NULL,
    strategy_label TEXT NOT NULL,
    outcome TEXT NOT NULL,
    success_score REAL,
    run_id TEXT,
    conversation_id TEXT,
    created_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS policy_versions (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    parent_id TEXT,
    reason TEXT,
    created_at DATETIME NOT NULL
);

-- ============================================================
-- Memory Claims (Phase 7)
-- ============================================================

CREATE TABLE IF NOT EXISTS memory_claims (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL, -- fact, rel
    scope TEXT NOT NULL,
    session_id TEXT,
    claim_text TEXT NOT NULL, -- DSL line or summary
    rel_type_raw TEXT,
    rel_type_norm TEXT,
    rel_type_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending', -- pending, promoted, rejected
    source_type TEXT NOT NULL,
    source_ref TEXT,
    conflict_topic_key TEXT,
    conflict_reason TEXT,
    evaluated_at DATETIME,
    decision_reason TEXT,
    episodic_event_id TEXT,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_sensitivity (
    belief_id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL, -- fact | rel
    sensitivity TEXT NOT NULL, -- none | pii | phi
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (belief_id) REFERENCES ics_beliefs(id)
);

-- Embeddings (Spec 3.1)
CREATE TABLE IF NOT EXISTS ics_embeddings (
    id TEXT PRIMARY KEY,
    assertion_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    created_at DATETIME NOT NULL,
    FOREIGN KEY (assertion_id) REFERENCES ics_fact_beliefs(belief_id)
);

CREATE INDEX IF NOT EXISTS idx_ics_embeddings_assertion ON ics_embeddings(assertion_id);

-- Embedding LSH Index (Vector Buckets)
CREATE TABLE IF NOT EXISTS ics_embedding_lsh (
    assertion_id INTEGER NOT NULL,
    bucket INTEGER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (assertion_id, bucket),
    FOREIGN KEY (assertion_id) REFERENCES ics_fact_beliefs(belief_id)
);

CREATE INDEX IF NOT EXISTS idx_ics_embedding_lsh_bucket ON ics_embedding_lsh(bucket);

-- 1.8 Entity Sketch Cache
CREATE TABLE IF NOT EXISTS ics_entity_sketches (
    entity_id INTEGER PRIMARY KEY,
    neighbors_top TEXT, -- JSON array of IDs
    tokens_top TEXT, -- JSON array of strings
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (entity_id) REFERENCES ics_entities(id)
);

-- Hub Detection Cache
CREATE TABLE IF NOT EXISTS ics_entity_degrees (
    entity_id INTEGER PRIMARY KEY,
    degree INTEGER NOT NULL DEFAULT 0,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (entity_id) REFERENCES ics_entities(id)
);

-- Legacy Entity Degree Cache (Migration Compatibility)
CREATE TABLE IF NOT EXISTS ics_entity_degree_cache (
    entity_id TEXT PRIMARY KEY,
    out_degree INTEGER NOT NULL DEFAULT 0,
    in_degree INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

-- 6.2 Session Bindings / 6.2 Pending Writes / 12.1 Merge Events
CREATE TABLE IF NOT EXISTS ics_pending_clarify (
    id INTEGER PRIMARY KEY,
    claim_id TEXT,
    session_id TEXT NOT NULL,
    original_dsl TEXT NOT NULL,
    ref_text TEXT NOT NULL,
    candidates_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ics_session_bindings (
    session_id TEXT NOT NULL, -- conversation_id
    ref_text TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    handle TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, ref_text),
    FOREIGN KEY (entity_id) REFERENCES ics_entities(id)
);

CREATE TABLE IF NOT EXISTS ics_pending_writes (
    id INTEGER PRIMARY KEY,
    parsed_lines TEXT NOT NULL,
    candidates_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ics_merge_events (
    id INTEGER PRIMARY KEY,
    from_id INTEGER NOT NULL,
    to_id INTEGER NOT NULL,
    reason TEXT NOT NULL,
    is_rolled_back BOOLEAN NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (from_id) REFERENCES ics_entities(id),
    FOREIGN KEY (to_id) REFERENCES ics_entities(id)
);

CREATE TABLE IF NOT EXISTS ics_working_set (
    item_id INTEGER NOT NULL, -- Entity or Belief ID (context dependent usage, or separate tables? simpler to keep separate usually, but spec says one set. Let's assume entities for now as anchors)
    item_type TEXT NOT NULL, -- entity, belief
    activation REAL NOT NULL DEFAULT 0.0,
    last_updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (item_id, item_type)
);

CREATE VIEW IF NOT EXISTS claim_ledger_index AS
SELECT
    CAST(id AS TEXT) AS claim_id,
    scope,
    confidence,
    evidence_event_ids AS provenance_refs_json,
    CASE
        WHEN expires_at IS NOT NULL AND datetime(expires_at) <= datetime('now') THEN 'DEPRECATED'
        ELSE 'ACTIVE'
    END AS status
FROM self_claims
UNION ALL
SELECT
    CAST(b.id AS TEXT) AS claim_id,
    b.scope,
    b.confidence,
    COALESCE(
        (SELECT json_group_array(e.id) FROM ics_evidence_events e WHERE e.belief_id = b.id),
        '[]'
    ) AS provenance_refs_json,
    CASE
        WHEN b.status = 'active' THEN 'ACTIVE'
        ELSE 'DEPRECATED'
    END AS status
FROM ics_beliefs b;

-- ============================================================
-- Cognitive Workspace + Pending Prompts
-- ============================================================

CREATE TABLE IF NOT EXISTS workspace_state (
    conversation_id TEXT PRIMARY KEY,
    goal_thread TEXT,
    active_plan_id TEXT,
    goal_stack_json TEXT,
    open_questions_json TEXT,
    active_hypotheses_json TEXT,
    working_set_topics_json TEXT,
    current_focus TEXT,
    focus_rationale TEXT,
    workspace_meta_json TEXT,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pending_user_prompts (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'monologue',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    skip_count INTEGER NOT NULL DEFAULT 0,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_asked_at DATETIME,
    expires_at DATETIME,
    auto_surface INTEGER NOT NULL DEFAULT 0,
    intent_kind TEXT,
    bridge_id TEXT,
    anchor_message_id TEXT,
    anchor_hash TEXT,
    anchor_created_at DATETIME,
    anchor_role TEXT
);

CREATE TABLE IF NOT EXISTS deferred_queue (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    item_type TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    reason TEXT NOT NULL,
    last_context_hash TEXT,
    reopen_trigger TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_asked_at DATETIME,
    expires_at DATETIME,
    dropped_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- FTS5 Virtual Tables & Triggers
-- NOTE: These are managed by db/mod.rs migrations to avoid conflicts.
-- The mod.rs file drops and recreates these with updated schemas.
-- ============================================================

-- (Managed in mod.rs migrations)

-- ============================================================
-- Indexes
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_beliefs_topic_key ON ics_beliefs(topic_key);
CREATE INDEX IF NOT EXISTS idx_beliefs_status ON ics_beliefs(status);
CREATE INDEX IF NOT EXISTS idx_beliefs_reconcile_state ON ics_beliefs(reconcile_state);
CREATE INDEX IF NOT EXISTS idx_beliefs_scope ON ics_beliefs(scope);
CREATE INDEX IF NOT EXISTS idx_beliefs_signature ON ics_beliefs(signature_hash);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON ics_fact_beliefs(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_facts_key ON ics_fact_beliefs(key);
CREATE INDEX IF NOT EXISTS idx_rels_type ON ics_rel_beliefs(rel_type);
CREATE INDEX IF NOT EXISTS idx_rels_type_id ON ics_rel_beliefs(rel_type_id);
CREATE INDEX IF NOT EXISTS idx_rel_type_alias_rel_type_id ON rel_type_alias(rel_type_id);
CREATE INDEX IF NOT EXISTS idx_evidence_belief ON ics_evidence_events(belief_id);
CREATE INDEX IF NOT EXISTS idx_links_from ON ics_belief_links(from_id);
CREATE INDEX IF NOT EXISTS idx_links_to ON ics_belief_links(to_id);
CREATE INDEX IF NOT EXISTS idx_working_activation ON ics_working_set(activation DESC);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias ON ics_entity_aliases(alias_canonical);

CREATE INDEX IF NOT EXISTS idx_episodic_run_time ON episodic_events(run_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_conv_time ON episodic_events(conversation_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_type_time ON episodic_events(event_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_traces_created ON strategy_traces(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_traces_label ON strategy_traces(strategy_label);
CREATE INDEX IF NOT EXISTS idx_policy_versions_created ON policy_versions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_deferred_queue_conv_time ON deferred_queue(conversation_id, dropped_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_claims_status ON memory_claims(status);
CREATE INDEX IF NOT EXISTS idx_memory_claims_created ON memory_claims(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_sensitivity_level ON memory_sensitivity(sensitivity);
CREATE INDEX IF NOT EXISTS idx_conv_summary_chunks_conv_time ON conversation_summary_chunks(conversation_id, end_ts DESC);
CREATE INDEX IF NOT EXISTS idx_conv_summary_chunks_hash ON conversation_summary_chunks(summary_hash);
CREATE INDEX IF NOT EXISTS idx_system_logs_time ON system_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_logs_run ON system_logs(run_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_logs_category ON system_logs(category, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_controls_subsystem ON system_controls(subsystem_id);
CREATE INDEX IF NOT EXISTS idx_system_control_events_time ON system_control_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_health_snapshots_time ON system_health_snapshots(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_event_ledger_time ON event_ledger(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_event_ledger_type ON event_ledger(type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_messages_time ON messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_write_ledger_time ON memory_write_ledger(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_write_ledger_conv ON memory_write_ledger(conversation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_pass_tokens_expiry ON memory_pass_tokens(expires_at);
CREATE INDEX IF NOT EXISTS idx_post_processing_jobs_status ON post_processing_jobs(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_post_processing_jobs_conv ON post_processing_jobs(conversation_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_deferred_emits_conv_time ON deferred_emits(conversation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_inner_monologue_conv_time ON inner_monologue_entries(conversation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_inner_monologue_candidates_entry ON inner_monologue_candidates(entry_id);
CREATE INDEX IF NOT EXISTS idx_counterfactual_conv_time ON counterfactual_simulations(conversation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reflection_staging_status ON self_reflection_staging(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tool_dispatches_run ON tool_dispatches(run_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_thread_runs_conv_time ON thread_runs(conversation_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_workspace_state_updated ON workspace_state(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_pending_prompts_conv_time ON pending_user_prompts(conversation_id, created_at DESC);

