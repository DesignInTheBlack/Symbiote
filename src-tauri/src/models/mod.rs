use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};


#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Conversation {
    pub conversation_id: String,
    pub schema_version: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum RunStatus {
    #[serde(rename = "active")]
    Active,
    #[serde(rename = "complete")]
    Complete,
    #[serde(rename = "error")]
    Error,
    #[serde(rename = "cancelled")]
    Cancelled,
    #[serde(rename = "superseded")]
    Superseded,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Run {
    pub run_id: String,
    pub trace_id: String,
    pub conversation_id: String,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub status: String, // 'active', 'complete', 'error', 'cancelled', 'superseded'
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum MessageRole {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "developer")]
    Developer,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum MessageStatus {
    #[serde(rename = "complete")]
    Complete,
    #[serde(rename = "streaming")]
    Streaming,
    #[serde(rename = "error")]
    Error,
    #[serde(rename = "cancelled")]
    Cancelled,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub message_id: String,
    pub conversation_id: String,
    pub run_id: Option<String>,
    pub trace_id: Option<String>,
    pub role: String,
    pub content: String,
    pub status: String,
    pub error: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseOrigin {
    Primary,
    Fallback,
    Tool,
    SystemState,
    SummaryEcho,
}

impl ResponseOrigin {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResponseOrigin::Primary => "primary",
            ResponseOrigin::Fallback => "fallback",
            ResponseOrigin::Tool => "tool",
            ResponseOrigin::SystemState => "system_state",
            ResponseOrigin::SummaryEcho => "summary_echo",
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Artifact {
    pub artifact_id: String,
    pub run_id: String,
    pub trace_id: String,
    pub artifact_type: String,
    pub schema_version: i32,
    pub payload: serde_json::Value,
    pub produced_by: String,
    pub parent_artifact_ids: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemLogEntry {
    pub id: String,
    pub timestamp: String,
    pub level: String,
    pub category: String,
    pub run_id: Option<String>,
    pub trace_id: Option<String>,
    pub payload: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemControlEntry {
    pub control_id: String,
    pub subsystem_id: String,
    pub mode: String,
    pub value_json: Option<String>,
    pub updated_at: String,
    pub updated_by: Option<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemControlEvent {
    pub event_id: String,
    pub subsystem_id: String,
    pub previous_mode: Option<String>,
    pub new_mode: String,
    pub value_json: Option<String>,
    pub actor: Option<String>,
    pub reason: Option<String>,
    pub status: Option<String>,
    pub timestamp: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ContextTagEntry {
    pub tag: String,
    pub confidence: f32,
    pub inferred: bool,
    pub evidence_event_ids: Vec<i64>,
    pub last_seen_at: String,
    pub source: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UserIntentSummary {
    pub summary: String,
    pub confirmed: bool,
    pub evidence_event_ids: Vec<i64>,
    pub updated_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemHealthSnapshot {
    pub snapshot_id: String,
    pub timestamp: String,
    pub run_id: Option<String>,
    pub trace_id: Option<String>,
    pub metrics_json: String,
    pub subsystem_states_json: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InnerMonologueCandidate {
    pub id: String,
    pub entry_id: String,
    pub candidate_id: Option<String>,
    pub outcome: Option<String>,
    pub suppression_reason: Option<String>,
    pub candidate_json: String,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InnerMonologueEntry {
    pub id: String,
    pub conversation_id: String,
    pub run_id: Option<String>,
    pub dialogue_id: Option<String>,
    pub turn_index: Option<i64>,
    pub speaker: Option<String>,
    pub mode: String,
    pub stream_type: Option<String>,
    pub thought: String,
    pub descriptors: Option<Vec<String>>,
    pub harvest_type: Option<String>,
    pub harvest_payload: Option<String>,
    pub created_at: String,
    pub candidates: Option<Vec<InnerMonologueCandidate>>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct SelfState {
    pub monologue_active: bool,
    pub last_internal_thought: String,
    pub current_focus: String,
    pub uncertainty_level: String,
    pub initiative_level: String,
    pub last_action_outcome: String,
    pub updated_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct AttentionSourceAttribution {
    pub source: String,
    pub weight: f64,
    pub share: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct AttentionSchemaState {
    pub capacity_usage: f64,
    pub selection_policy: String,
    pub suppressed_items: Vec<String>,
    pub stability: f64,
    pub source_attribution: Vec<AttentionSourceAttribution>,
    pub last_updated_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct WorkingMemoryBlock {
    pub focus: Option<String>,
    pub open_questions: Vec<String>,
    pub active_hypotheses: Vec<String>,
    pub next_action: Option<String>,
    pub confidence: Option<f32>,
    pub drift_score: Option<f32>,
    pub updated_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MemoryWriteLedgerEntry {
    pub id: String,
    pub conversation_id: Option<String>,
    pub category: String,
    pub source: String,
    pub reason_code: String,
    pub run_id: Option<String>,
    pub trace_id: Option<String>,
    pub payload_hash: Option<String>,
    pub snapshot_hash: Option<String>,
    pub gate_decision_id: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EvidenceSource {
    pub evidence_id: String,
    pub source_table: String,
    pub source_id: String,
    pub source_type: String,
    pub source_ref: Option<String>,
    pub snippet: Option<String>,
    pub weight: Option<f64>,
    pub confidence: Option<f64>,
    pub run_id: Option<String>,
    pub trace_id: Option<String>,
    pub conversation_id: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EvidenceLink {
    pub link_id: String,
    pub evidence_id: String,
    pub target_type: String,
    pub target_id: String,
    pub relation: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DecisionReportRecord {
    pub report_id: String,
    pub run_id: Option<String>,
    pub trace_id: Option<String>,
    pub conversation_id: Option<String>,
    pub report_json: String,
    pub evidence_event_ids: Vec<i64>,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EvidenceLineageEntry {
    pub source: EvidenceSource,
    pub links: Vec<EvidenceLink>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OutcomeEvent {
    pub outcome_id: String,
    pub run_id: Option<String>,
    pub trace_id: Option<String>,
    pub candidate_id: Option<String>,
    pub target_type: String,
    pub verdict: String,
    pub confidence: f32,
    pub source: String,
    pub note: Option<String>,
    pub evidence_event_ids: Vec<i64>,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EventLedgerEntry {
    pub event_id: String,
    pub timestamp: String,
    #[serde(rename = "type")]
    pub r#type: String,
    pub payload: serde_json::Value,
    pub tags: Option<serde_json::Value>,
    pub run_id: Option<String>,
    pub trace_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SubjectSnapshot {
    pub snapshot_hash: String,
    pub snapshot_version: String,
    pub tick_id: String,
    pub conversation_id: String,
    pub run_id: Option<String>,
    pub timestamp: String,
    pub subject_state_json: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ActionProposal {
    pub proposal_id: String,
    pub snapshot_hash: String,
    pub intent: String,
    pub steps_json: String,
    pub plan_hash: String,
    pub plan_state: String,
    pub risk_level: String,
    pub required_claims_json: String,
    pub required_error_bounds_json: Option<String>,
    pub verification_plan_json: Option<String>,
    pub success_criteria_json: String,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GateDecision {
    pub decision_id: String,
    pub proposal_id: String,
    pub snapshot_hash: String,
    pub decision: String,
    pub evidence_refs_json: String,
    pub metrics_json: String,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResidualVector {
    pub residual_id: String,
    pub prediction_id: String,
    pub outcome_id: String,
    pub residual_value: f64,
    pub normalized_residual: f64,
    pub salience_score: f64,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ErrorEvent {
    pub error_event_id: String,
    pub residual_id: String,
    pub linked_claims_json: String,
    pub classification: String,
    pub status: String,
    pub recommended_actions_json: String,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IntrospectionEntry {
    pub entry_id: String,
    pub snapshot_hash: String,
    pub workspace_refs_json: String,
    pub event_refs_json: String,
    pub prediction_refs_json: Option<String>,
    pub error_refs_json: Option<String>,
    pub numeric_payload_json: String,
    pub narrative: String,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AuditLogEntry {
    pub audit_id: String,
    pub target_id: String,
    pub snapshot_hash: String,
    pub checks_json: String,
    pub discrepancy_score: f64,
    pub recommended_action: String,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CalibrationChange {
    pub change_id: String,
    pub snapshot_hash: String,
    pub knob: String,
    pub old_value: f64,
    pub new_value: f64,
    pub reason: String,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QualiaLabel {
    pub label_id: String,
    pub event_id: String,
    pub snapshot_hash: String,
    pub tag: String,
    pub intensity: f64,
    pub context_json: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QualiaRewardEvent {
    pub reward_id: String,
    pub label_id: String,
    pub magnitude: f64,
    pub outcome_ref: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CognitiveReadinessReport {
    pub generated_at: String,
    pub kernel_state_present: bool,
    pub last_monologue_at: Option<String>,
    pub last_inner_summary_at: Option<String>,
    pub last_conversation_summary_at: Option<String>,
    pub last_semantic_core_at: Option<String>,
    pub last_memory_pass_at: Option<String>,
    pub recent_memory_policy_violations: i64,
    pub tool_dispatch_successes: i64,
    pub tool_dispatch_failures: i64,
    pub notes: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CognitiveCheckResult {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ParameterRegistry {
    pub profile_name: String,
    pub profile_version: i64,
    pub payload_json: String,
    pub updated_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct WorkspaceHypothesis {
    pub text: String,
    pub confidence: f32,
    #[serde(default)]
    pub speculative: bool,
    #[serde(default)]
    pub evidence_event_ids: Vec<i64>,
    #[serde(default)]
    pub belief_ids: Vec<i64>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct GoalStep {
    pub text: String,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub evidence_event_ids: Vec<i64>,
    #[serde(default)]
    pub belief_ids: Vec<i64>,
    #[serde(default)]
    pub completed_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct GoalStackItem {
    pub goal: String,
    #[serde(default)]
    pub steps: Vec<GoalStep>,
    #[serde(default)]
    pub current_step_index: usize,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub evidence_event_ids: Vec<i64>,
    #[serde(default)]
    pub belief_ids: Vec<i64>,
    #[serde(default)]
    pub updated_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct WorkspaceFieldMeta {
    #[serde(default)]
    pub speculative: bool,
    #[serde(default)]
    pub evidence_event_ids: Vec<i64>,
    #[serde(default)]
    pub belief_ids: Vec<i64>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct WorkspaceListItemMeta {
    pub text: String,
    #[serde(default)]
    pub speculative: bool,
    #[serde(default)]
    pub evidence_event_ids: Vec<i64>,
    #[serde(default)]
    pub belief_ids: Vec<i64>,
    #[serde(default)]
    pub attempt_count: i64,
    #[serde(default)]
    pub last_asked_at: Option<String>,
    #[serde(default)]
    pub expires_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct WorkspaceMeta {
    #[serde(default)]
    pub goal_thread: Option<WorkspaceFieldMeta>,
    #[serde(default)]
    pub current_focus: Option<WorkspaceFieldMeta>,
    #[serde(default)]
    pub focus_rationale: Option<WorkspaceFieldMeta>,
    #[serde(default)]
    pub open_questions: Vec<WorkspaceListItemMeta>,
    #[serde(default)]
    pub working_set_topics: Vec<WorkspaceListItemMeta>,
    #[serde(default)]
    pub active_hypotheses: Vec<WorkspaceHypothesis>,
    #[serde(default)]
    pub runtime: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct WorkspaceKernelContributor {
    pub cycle_id: String,
    pub broadcast_refs: Vec<String>,
    pub ignition_active: bool,
    pub timestamp: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct WorkspaceMemoryContributor {
    pub recent_writes: i64,
    pub top_topics: Vec<String>,
    pub last_write_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct WorkspacePredictionContributor {
    pub last_prediction_at: Option<String>,
    pub divergence_count: i64,
    pub residual_count: i64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct WorkspaceAttentionContributor {
    pub focus_refs: Vec<String>,
    pub meta_confidence: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct WorkspaceSelfModelContributor {
    pub identity_confidence: f64,
    pub last_reflection_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct WorkspaceQualiaContributor {
    pub dominant_tag: Option<String>,
    pub intensity: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct WorkspaceToolsContributor {
    pub success_rate: f64,
    pub failure_rate: f64,
    pub last_failure_kind: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct WorkspaceOrganismContributor {
    pub integrity_risk: f64,
    pub drift_score: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct WorkspaceErrorContributor {
    pub open_error_count: usize,
    pub pattern_flags: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct WorkspaceContributors {
    pub kernel: Option<WorkspaceKernelContributor>,
    pub memory: Option<WorkspaceMemoryContributor>,
    pub prediction: Option<WorkspacePredictionContributor>,
    pub attention: Option<WorkspaceAttentionContributor>,
    pub self_model: Option<WorkspaceSelfModelContributor>,
    pub qualia: Option<WorkspaceQualiaContributor>,
    pub tools: Option<WorkspaceToolsContributor>,
    pub organism: Option<WorkspaceOrganismContributor>,
    pub error_state: Option<WorkspaceErrorContributor>,
    pub missing: Vec<String>,
    pub updated_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WorkspaceState {
    pub conversation_id: String,
    pub goal_thread: Option<String>,
    pub active_plan_id: Option<String>,
    #[serde(default)]
    pub goal_stack: Vec<GoalStackItem>,
    pub open_questions: Vec<String>,
    pub active_hypotheses: Vec<WorkspaceHypothesis>,
    pub working_set_topics: Vec<String>,
    pub current_focus: Option<String>,
    pub focus_rationale: Option<String>,
    #[serde(default)]
    pub workspace_meta: WorkspaceMeta,
    pub updated_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Settings {
    pub schema_version: i32,
    pub api_base_url: String,
    pub api_key: Option<String>,
    pub streaming_enabled: bool,
    pub history_window: i32,
    pub injection_policy: String, // 'include', 'exclude'
    pub request_defaults: Option<serde_json::Value>,
    pub active_model_id: Option<String>,
    pub system_prompt: Option<String>,
    pub voice_name: Option<String>,
    pub voice_speed: Option<f32>,
    pub summarization_api_url: Option<String>,
    pub summarization_model: Option<String>,
    pub embedding_model: Option<String>,
    pub user_display_name: Option<String>,
    pub assistant_display_name: Option<String>,
    pub onboarding_completed: Option<bool>,
    pub ui_theme: Option<String>,
    pub episodic_enabled: Option<bool>,
    pub episodic_injection_enabled: Option<bool>,
    pub episodic_compaction_enabled: Option<bool>,
    pub episodic_injection_limit: Option<i32>,
    pub episodic_opt_out: Option<bool>,
    pub memory_claims_enabled: Option<bool>,
    pub phi_consent: Option<bool>,
    pub seed_personal_user: Option<bool>,
    pub lexical_fallback_enabled: Option<bool>,
    pub memory_half_life_hours: Option<f32>,
    pub research_budget_per_hour: Option<i64>,
    pub research_budget_reset_window: Option<i64>,
    pub research_cost_per_call: Option<i64>,
    pub monologue_interval_seconds: Option<i64>,
    pub monologue_timeout_secs: Option<i64>,
    pub monologue_retry_timeout_secs: Option<i64>,
    pub empty_response_retry_max: Option<i32>,
    pub empty_response_retry_timeout_ms: Option<i64>,
    pub monologue_max_per_hour: Option<i64>,
    pub thread_max_depth: Option<i64>,
    pub allow_shell_tool: Option<bool>,
    pub shell_command_allowlist: Option<String>,

    pub ask_budget_max: Option<i32>,
    pub calculator_followups_max: Option<i32>,
    pub loop_similarity_threshold: Option<f32>,
    pub loop_recent_k: Option<i32>,
    pub meta_cog_outcome_turns: Option<i64>,
    pub meta_cog_cycle_window_turns: Option<i64>,
    pub meta_cog_outcome_timeout_s: Option<i64>,
    pub meta_cog_cooldown_s: Option<i64>,
    pub meta_cog_streak_limit: Option<i32>,
    pub registry_profile_name: Option<String>,
    pub controller_enabled: Option<bool>,
    pub monologue_stabilization_enabled: Option<bool>,
    pub monologue_surface_enabled: Option<bool>,
    pub show_monologue_in_chat: Option<bool>,
    pub enable_introspection: Option<bool>,
    pub heartbeat_enabled: Option<bool>,
    pub dream_enabled: Option<bool>,
    pub binding_enforcement_enabled: Option<bool>,
    pub pending_prompt_alignment_enabled: Option<bool>,
    pub auto_memory_pass_enabled: Option<bool>,
    pub summary_cohesion_enabled: Option<bool>,
    pub compact_prompt_enabled: Option<bool>,
    pub context_hydration_mode: Option<String>,
    pub context_budgeter_enabled: Option<bool>,
    pub context_miss_detector_enabled: Option<bool>,
    pub world_model_reconcile_mode: Option<String>,
    pub goal_loop_enabled: Option<bool>,
    pub goal_loop_interval_turns: Option<i32>,
    pub goal_loop_load_threshold_ms: Option<i64>,
    pub json_only_disabled_models: Option<String>,
    pub tool_failure_gate_window_mins: Option<i64>,
    pub tool_failure_gate_tool_names: Option<String>,
    pub gate_default_soft: Option<bool>,
    pub gate_shadow_mode: Option<bool>,
    pub gate_rollout_percent: Option<i32>,
    pub self_report_channel: Option<bool>,
    pub self_awareness_expression_mode: Option<String>,
    pub explicit_feedback_only: Option<bool>,
    pub weight_user_satisfaction: Option<f32>,
    pub weight_policy_rigor: Option<f32>,
    pub weight_latency: Option<f32>,
    pub weight_evidence_strictness: Option<f32>,
    pub weight_exploration: Option<f32>,
    pub monologue_provenance_guard: Option<bool>,
    pub organism_decay: Option<bool>,
    pub model_context_limit: Option<i32>,
    pub introspection_confidence_threshold: Option<f32>,
    pub introspection_drift_threshold: Option<f32>,
    pub introspection_ambiguity_threshold: Option<f32>,
    pub enable_attribution_gate: Option<bool>,
    pub enable_user_utterance_evidence: Option<bool>,
    pub enable_attribution_metadata: Option<bool>,
    pub enable_tool_schema_validation: Option<bool>,
    pub enable_context_evidence: Option<bool>,
    pub enable_monologue_validator: Option<bool>,
    pub enable_memory_evidence_gating: Option<bool>,
    pub enable_speculative_workspace_containment: Option<bool>,
    pub stability_prompt_override_guard: Option<bool>,
    pub stability_monologue_tagged: Option<bool>,
    pub stability_introspection_structured: Option<bool>,
    pub stability_disable_working_hypothesis: Option<bool>,
    pub stability_state_disclosure_expanded: Option<bool>,
    pub stability_transcript_normalization: Option<bool>,
    pub stability_memory_hygiene: Option<bool>,
    pub stability_non_stream_sanitization: Option<bool>,
    
    // Voice Effects
    pub voice_pitch_semitones: Option<f32>,
    pub voice_reverb_amount: Option<f32>,
    pub voice_compression: Option<f32>,
    pub voice_formant_shift: Option<f32>,
    
    // Protocol Settings
    pub trace_history_limit: Option<i32>,  // Default 10, max 50
    pub cockpit_write_enabled: Option<bool>,
}

impl Settings {
    pub fn validate(&mut self) -> Vec<String> {
        let mut adjustments = Vec::new();

        if self.history_window < 0 {
            adjustments.push("history_window: clamped to 0".to_string());
            self.history_window = 0;
        }

        clamp_opt_i64(
            &mut self.monologue_interval_seconds,
            0,
            None,
            "monologue_interval_seconds",
            &mut adjustments,
        );
        clamp_opt_i64(
            &mut self.monologue_timeout_secs,
            5,
            None,
            "monologue_timeout_secs",
            &mut adjustments,
        );
        clamp_opt_i64(
            &mut self.monologue_retry_timeout_secs,
            5,
            None,
            "monologue_retry_timeout_secs",
            &mut adjustments,
        );
        clamp_opt_i32(
            &mut self.empty_response_retry_max,
            0,
            None,
            "empty_response_retry_max",
            &mut adjustments,
        );
        clamp_opt_i64(
            &mut self.empty_response_retry_timeout_ms,
            0,
            None,
            "empty_response_retry_timeout_ms",
            &mut adjustments,
        );
        clamp_opt_i64(
            &mut self.monologue_max_per_hour,
            0,
            None,
            "monologue_max_per_hour",
            &mut adjustments,
        );
        clamp_opt_i64(
            &mut self.thread_max_depth,
            0,
            None,
            "thread_max_depth",
            &mut adjustments,
        );
        clamp_opt_i64(
            &mut self.tool_failure_gate_window_mins,
            0,
            None,
            "tool_failure_gate_window_mins",
            &mut adjustments,
        );
        clamp_opt_i64(
            &mut self.research_budget_per_hour,
            0,
            None,
            "research_budget_per_hour",
            &mut adjustments,
        );
        clamp_opt_i64(
            &mut self.research_budget_reset_window,
            0,
            None,
            "research_budget_reset_window",
            &mut adjustments,
        );
        clamp_opt_i64(
            &mut self.research_cost_per_call,
            0,
            None,
            "research_cost_per_call",
            &mut adjustments,
        );
        clamp_opt_i64(
            &mut self.meta_cog_outcome_turns,
            0,
            None,
            "meta_cog_outcome_turns",
            &mut adjustments,
        );
        clamp_opt_i64(
            &mut self.meta_cog_cycle_window_turns,
            0,
            None,
            "meta_cog_cycle_window_turns",
            &mut adjustments,
        );
        clamp_opt_i64(
            &mut self.meta_cog_outcome_timeout_s,
            0,
            None,
            "meta_cog_outcome_timeout_s",
            &mut adjustments,
        );
        clamp_opt_i64(
            &mut self.meta_cog_cooldown_s,
            0,
            None,
            "meta_cog_cooldown_s",
            &mut adjustments,
        );

        clamp_opt_i32(&mut self.ask_budget_max, 0, None, "ask_budget_max", &mut adjustments);
        clamp_opt_i32(
            &mut self.calculator_followups_max,
            0,
            None,
            "calculator_followups_max",
            &mut adjustments,
        );
        clamp_opt_i32(&mut self.loop_recent_k, 1, None, "loop_recent_k", &mut adjustments);
        clamp_opt_i32(
            &mut self.meta_cog_streak_limit,
            0,
            None,
            "meta_cog_streak_limit",
            &mut adjustments,
        );

        clamp_opt_f32(
            &mut self.loop_similarity_threshold,
            0.0,
            Some(1.0),
            "loop_similarity_threshold",
            &mut adjustments,
        );
        clamp_opt_f32(
            &mut self.introspection_confidence_threshold,
            0.0,
            Some(1.0),
            "introspection_confidence_threshold",
            &mut adjustments,
        );
        clamp_opt_f32(
            &mut self.introspection_drift_threshold,
            0.0,
            Some(1.0),
            "introspection_drift_threshold",
            &mut adjustments,
        );
        clamp_opt_f32(
            &mut self.introspection_ambiguity_threshold,
            0.0,
            Some(1.0),
            "introspection_ambiguity_threshold",
            &mut adjustments,
        );
        clamp_opt_i32(
            &mut self.gate_rollout_percent,
            0,
            Some(100),
            "gate_rollout_percent",
            &mut adjustments,
        );

        if let Some(value) = self.model_context_limit {
            if value < 128 {
                adjustments.push(format!("model_context_limit: {} -> 128", value));
                self.model_context_limit = Some(128);
            }
        }

        if let Some(value) = self.memory_half_life_hours {
            if value <= 0.0 {
                adjustments.push(format!("memory_half_life_hours: {} -> None", value));
                self.memory_half_life_hours = None;
            }
        }

        if let Some(raw) = self.self_awareness_expression_mode.clone() {
            let normalized = raw.trim().to_lowercase();
            let clamped = match normalized.as_str() {
                "conservative" | "balanced" | "expressive" => normalized,
                _ => "conservative".to_string(),
            };
            if clamped != raw {
                adjustments.push(format!(
                    "self_awareness_expression_mode: {} -> {}",
                    raw, clamped
                ));
                self.self_awareness_expression_mode = Some(clamped);
            }
        }

        adjustments
    }
}

fn clamp_opt_i64(
    value: &mut Option<i64>,
    min: i64,
    max: Option<i64>,
    name: &str,
    adjustments: &mut Vec<String>,
) {
    if let Some(raw) = *value {
        let mut updated = raw;
        if raw < min {
            updated = min;
        }
        if let Some(max_val) = max {
            if updated > max_val {
                updated = max_val;
            }
        }
        if updated != raw {
            adjustments.push(format!("{}: {} -> {}", name, raw, updated));
            *value = Some(updated);
        }
    }
}

fn clamp_opt_i32(
    value: &mut Option<i32>,
    min: i32,
    max: Option<i32>,
    name: &str,
    adjustments: &mut Vec<String>,
) {
    if let Some(raw) = *value {
        let mut updated = raw;
        if raw < min {
            updated = min;
        }
        if let Some(max_val) = max {
            if updated > max_val {
                updated = max_val;
            }
        }
        if updated != raw {
            adjustments.push(format!("{}: {} -> {}", name, raw, updated));
            *value = Some(updated);
        }
    }
}

fn clamp_opt_f32(
    value: &mut Option<f32>,
    min: f32,
    max: Option<f32>,
    name: &str,
    adjustments: &mut Vec<String>,
) {
    if let Some(raw) = *value {
        let mut updated = raw;
        if raw < min {
            updated = min;
        }
        if let Some(max_val) = max {
            if updated > max_val {
                updated = max_val;
            }
        }
        if (updated - raw).abs() > f32::EPSILON {
            adjustments.push(format!("{}: {} -> {}", name, raw, updated));
            *value = Some(updated);
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RollingSummaryStatus {
    pub summary: Option<String>,
    pub last_error: Option<String>,
    pub last_error_at: Option<String>,
    #[serde(default)]
    pub pending: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConversationSummaryChunk {
    pub id: String,
    pub chunk_id: String,
    pub summary: String,
    pub start_ts: Option<String>,
    pub end_ts: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SelfModel {
    pub capabilities: serde_json::Value,
    pub limitations: serde_json::Value,
    pub active_tools: serde_json::Value,
    pub memory_health: serde_json::Value,
    pub persona: serde_json::Value,
    pub persona_daily_delta: serde_json::Value,
    pub persona_last_delta_date: Option<String>,
    pub goals: serde_json::Value,
    #[serde(default)]
    pub identity_thread: Option<String>,
    #[serde(default)]
    pub identity_confidence: f32,
    #[serde(default)]
    pub identity_uncertainty_note: Option<String>,
    #[serde(default)]
    pub identity_updated_at: Option<String>,
    pub reflection_status: serde_json::Value,
    pub reflection_frozen: bool,
    pub last_reflection_at: Option<String>,
    #[serde(default)]
    pub internal_state_summary: serde_json::Value,
    #[serde(default)]
    pub internal_state_map_version: Option<i64>,
    #[serde(default)]
    pub unified_state: serde_json::Value,
    #[serde(default)]
    pub unified_state_evidence: serde_json::Value,
    #[serde(default)]
    pub unified_state_updated_at: Option<String>,
    pub updated_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ReflectionStagingEntry {
    pub id: String,
    pub proposal_json: String,
    pub evidence_event_ids: Option<Vec<i64>>,
    pub status: String,
    pub created_at: String,
    pub reviewed_at: Option<String>,
    pub reviewed_by: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ControllerState {
    pub confidence: f32,
    #[serde(default)]
    pub uncertainty: f32,
    pub drift_score: f32,
    pub failure_streak: i32,
    pub autonomy_level: f32,
    pub verification_needed: bool,
    pub reanchor_needed: bool,
    pub evidence_coverage: f32,
    #[serde(default)]
    pub telemetry_coverage: f32,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default)]
    pub last_strategy: Option<String>,
    #[serde(default)]
    pub outcome_quality: Option<f32>,
    pub missing_fields: Vec<String>,
    pub updated_at: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ControllerGate {
    pub throttle_tools: bool,
    pub throttle_threads: bool,
    pub throttle_asks: bool,
    pub prefer_verification: bool,
    pub reanchor: bool,
    pub autonomy_scale: f32,
    pub reasons: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tool {
    pub r#type: String, // "function"
    pub function: ToolFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema
    #[serde(default, skip_serializing)]
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String, // "function"
    pub function: ToolCallFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String, // JSON string
}

#[derive(Debug, Serialize, Deserialize, Clone, sqlx::FromRow)]
pub struct Reminder {
    pub id: String,
    pub content: String,
    pub due_at: i64,  // Unix timestamp for reliable SQLite comparison
    pub r#type: String, // 'ALARM', 'REMINDER', 'CONVERSATION'
    pub status: String,
    pub created_at: i64,  // Unix timestamp
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SelfInspection {
    pub tables: Vec<(String, i64)>,
    pub last_user_memory_write: Option<String>,
    pub last_self_memory_write: Option<String>,
    pub last_memory_error_at: Option<String>,
    pub open_conflicts: i64,
    pub error_count: i64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SelfClaim {
    pub id: String,
    pub claim_text: String,
    pub claim_key: String,
    pub evidence_event_ids: Option<serde_json::Value>,
    pub belief_ids: Option<serde_json::Value>,
    pub confidence: f32,
    pub polarity: String,
    pub source_run_id: Option<String>,
    pub conversation_id: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EpisodicEvent {
    pub id: String,
    pub event_type: String,
    pub payload: serde_json::Value,
    pub timestamp: String,
    pub run_id: Option<String>,
    pub trace_id: Option<String>,
    pub conversation_id: Option<String>,
    pub scope: Option<String>,
    pub source_type: String,
    pub source_ref: Option<String>,
    pub linked_belief_id: Option<i64>,
    pub linked_artifact_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StrategyTrace {
    pub id: String,
    pub features: serde_json::Value,
    pub strategy_label: String,
    pub outcome: String,
    pub success_score: Option<f64>,
    pub run_id: Option<String>,
    pub conversation_id: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PolicyVersion {
    pub id: String,
    pub label: String,
    pub payload: serde_json::Value,
    pub parent_id: Option<String>,
    pub reason: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MemoryClaim {
    pub id: String,
    pub kind: String,
    pub scope: String,
    pub session_id: Option<String>,
    pub claim_text: String,
    pub rel_type_raw: Option<String>,
    pub rel_type_norm: Option<String>,
    pub rel_type_id: Option<String>,
    pub status: String,
    pub source_type: String,
    pub source_ref: Option<String>,
    pub conflict_topic_key: Option<String>,
    pub conflict_reason: Option<String>,
    pub evaluated_at: Option<String>,
    pub decision_reason: Option<String>,
    pub episodic_event_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}
