use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{KernelMode, TaskPhase, StanceState, MetaCogPending, Outcome, ThreadHandle, StopState, ToolFailurePenalty};
use crate::core::kernel::telemetry::TelemetrySnapshotEntry;
use crate::models::{SelfState, WorkingMemoryBlock, WorkspaceHypothesis};

fn default_monologue_relaxation_level() -> i32 {
    1
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct AutoMemoryDecision {
    pub trigger: bool,
    pub score: f32,
    pub ambiguity: bool,
    #[allow(dead_code)]
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct IntrospectionCacheEntry {
    pub key: String,
    pub summary: String,
}

#[derive(Debug, Clone)]
pub(crate) struct ValidationResult {
    pub valid_evidence_ids: Vec<i64>,
    pub invalid_evidence_ids: Vec<i64>,
    pub valid_belief_ids: Vec<i64>,
    pub invalid_belief_ids: Vec<i64>,
    pub source_types: Vec<String>,
    pub max_quality: f32,
    pub quality_ok: bool,
    pub fresh_ok: bool,
    pub belief_fresh_ok: bool,
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self {
            valid_evidence_ids: Vec::new(),
            invalid_evidence_ids: Vec::new(),
            valid_belief_ids: Vec::new(),
            invalid_belief_ids: Vec::new(),
            source_types: Vec::new(),
            max_quality: 0.0,
            quality_ok: true,
            fresh_ok: false,
            belief_fresh_ok: false,
        }
    }
}

impl ValidationResult {
    pub(crate) fn evidence_ok(&self) -> bool {
        (self.fresh_ok || self.belief_fresh_ok) && self.quality_ok
    }
}

#[derive(Debug, Clone)]
pub(crate) struct EvidenceValidationCacheEntry {
    pub result: ValidationResult,
    pub cached_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub(crate) struct CompactPromptDecision {
    pub use_compact: bool,
    pub disqualifiers: Vec<String>,
    pub pending_prompts: i64,
    pub recent_memory_write: bool,
    pub workspace_delta: i32,
    pub force_reasons: Vec<String>,
    pub anchor_hits: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelState {
    pub conversation_id: String,
    pub mode: KernelMode,
    #[serde(default)]
    pub task_id: Option<String>,
    #[serde(default)]
    pub task_phase: TaskPhase,
    pub pressure_score: f32,
    pub pressure_signals: Vec<String>,
    pub stance: StanceState,
    pub last_mode_switch_at: Option<String>,
    pub last_medium_update_at: Option<String>,
    pub last_monologue_at: Option<String>,
    #[serde(default)]
    pub last_monologue_tick_id: Option<String>,
    #[serde(default)]
    pub last_monologue_started_at: Option<String>,
    #[serde(default)]
    pub last_monologue_completed_at: Option<String>,
    #[serde(default)]
    pub last_monologue_tick_outcome: Option<String>,
    #[serde(default)]
    pub last_monologue_status_emitted: Option<bool>,
    #[serde(default)]
    pub last_monologue_visible: Option<bool>,
    #[serde(default)]
    pub last_reflection_applied_at: Option<String>,
    pub monologue_window_start: Option<String>,
    pub monologue_count: i64,
    #[serde(default)]
    pub monologue_suppression_window_start: Option<String>,
    #[serde(default)]
    pub monologue_suppression_counts: HashMap<String, i64>,
    #[serde(default = "default_monologue_relaxation_level")]
    pub monologue_relaxation_level: i32,
    #[serde(default)]
    pub monologue_json_supported: Option<bool>,
    #[serde(default)]
    pub monologue_json_disabled_until: Option<String>,
    #[serde(default)]
    pub monologue_force_primary: bool,
    #[serde(default)]
    pub hypothesis_defer_until: Option<i64>,
    #[serde(default)]
    pub last_hypothesis_promoted_at: Option<String>,
    pub research_window_start: Option<String>,
    pub research_used: i64,
    pub thread_depth: i64,
    pub last_processed_dispatch_at: Option<String>,
    pub last_processed_dispatch_id: Option<String>,
    pub last_processed_thread_at: Option<String>,
    pub last_processed_thread_id: Option<String>,
    pub last_outcome_at: Option<String>,
    pub last_semantic_promotion_at: Option<String>,
    pub recent_outcomes: Vec<Outcome>,
    pub pending_questions: Vec<String>,
    #[serde(default)]
    pub asked_slot_sets: Vec<Vec<String>>,
    #[serde(default)]
    pub question_fingerprints: Vec<String>,
    #[serde(default)]
    pub recent_questions: Vec<String>,
    #[serde(default)]
    pub refused_slots: Vec<String>,
    #[serde(default)]
    pub refusal_count: i32,
    #[serde(default)]
    pub identity_violation_count: i64,
    #[serde(default)]
    pub identity_violation_window_start: Option<String>,
    #[serde(default)]
    pub pending_actions: Vec<String>,
    #[serde(default)]
    pub pending_reframes: Vec<String>,
    #[serde(default)]
    pub pending_angles: Vec<String>,
    #[serde(default)]
    pub last_user_input: Option<String>,
    #[serde(default)]
    pub last_user_input_at: Option<String>,
    #[serde(default)]
    pub last_user_message_id: Option<String>,
    #[serde(default)]
    pub last_input_evidence_event_ids: Vec<i64>,
    #[serde(default)]
    pub anchor_epoch: i64,
    #[serde(default)]
    pub user_redirect_turns_remaining: i32,
    #[serde(default)]
    pub redirect_focus: Option<String>,
    #[serde(default)]
    pub last_redirect_clarifier_epoch: i64,
    #[serde(default)]
    pub redirect_focus_confirmed_turns: i32,
    #[serde(default)]
    pub redirect_focus_miss_turns: i32,
    #[serde(default)]
    pub redirect_focus_explicit: bool,
    #[serde(default)]
    pub last_monologue_anchor_epoch: i64,
    #[serde(default)]
    pub last_assistant_output: Option<String>,
    #[serde(default)]
    pub last_assistant_output_no_tags: Option<String>,
    #[serde(default)]
    pub last_proactive_emit_at: Option<String>,
    #[serde(default)]
    pub proactive_cooldown_until: Option<String>,
    #[serde(default)]
    pub last_proactive_question: Option<String>,
    #[serde(default)]
    pub last_proactive_memory_pass_at: Option<String>,
    #[serde(default)]
    pub last_memory_pass_at: Option<String>,
    pub failure_count: i64,
    #[serde(default)]
    pub tool_failure_count: i64,
    #[serde(default)]
    pub tool_failure_penalties: HashMap<String, ToolFailurePenalty>,
    pub uncertainty_count: i64,
    pub stalled_count: i64,
    pub halted: bool,
    #[serde(default)]
    pub monologue_unhalted_at: Option<String>,
    #[serde(default)]
    pub monologue_probation_until: Option<String>,
    #[serde(default)]
    pub monologue_health_last_alert_at: Option<String>,
    pub active_threads: Vec<ThreadHandle>,
    #[serde(default)]
    pub self_state: SelfState,
    #[serde(default)]
    pub working_memory: Option<WorkingMemoryBlock>,
    #[serde(default)]
    pub last_response_summary: Option<String>,
    #[serde(default)]
    pub last_response_summary_at: Option<String>,
    #[serde(default)]
    pub prompt_section_hashes: HashMap<String, String>,
    #[serde(default)]
    pub introspection_force: Option<String>,
    #[serde(default)]
    pub workspace_goal_thread: Option<String>,
    #[serde(default)]
    pub workspace_active_plan_id: Option<String>,
    #[serde(default)]
    pub workspace_goal_stack: Vec<crate::models::GoalStackItem>,
    #[serde(default)]
    pub goal_loop_turn_count: i64,
    #[serde(default)]
    pub workspace_open_questions: Vec<String>,
    #[serde(default)]
    pub workspace_active_hypotheses: Vec<WorkspaceHypothesis>,
    #[serde(default)]
    pub workspace_working_set_topics: Vec<String>,
    #[serde(default)]
    pub workspace_current_focus: Option<String>,
    #[serde(default)]
    pub workspace_focus_rationale: Option<String>,
    #[serde(default)]
    pub workspace_meta: crate::models::WorkspaceMeta,
    #[serde(default)]
    pub last_workspace_delta_fields: Vec<String>,
    #[serde(default)]
    pub last_workspace_delta_count: i32,
    #[serde(default)]
    pub last_workspace_update_at: Option<String>,
    #[serde(default)]
    pub last_focus_change_at: Option<String>,
    #[serde(default)]
    pub last_subject_snapshot_hash: Option<String>,
    #[serde(default)]
    pub last_subject_snapshot_at: Option<String>,
    #[serde(default)]
    pub last_plan_hash: Option<String>,
    #[serde(skip, default)]
    pub last_persisted_state_hash: Option<String>,
    #[serde(skip, default)]
    pub last_persisted_workspace_hash: Option<String>,
    #[serde(default)]
    pub controller_state: Option<crate::models::ControllerState>,
    #[serde(default)]
    pub controller_gate: Option<crate::models::ControllerGate>,
    #[serde(default)]
    pub self_model_version: i64,
    #[serde(default)]
    pub self_model_updated_at: Option<String>,
    #[serde(default)]
    pub self_model_unified: Option<serde_json::Value>,
    #[serde(default)]
    pub self_model_unified_evidence: Option<serde_json::Value>,
    #[serde(default)]
    pub last_self_report_at: Option<String>,
    #[serde(default)]
    pub self_report_snapshot: Option<serde_json::Value>,
    #[serde(default)]
    pub proaction_throttle_tools_override: Option<bool>,
    #[serde(default)]
    pub proaction_throttle_threads_override: Option<bool>,
    #[serde(default)]
    pub last_controller_snapshot_at: Option<String>,
    #[serde(default)]
    pub(crate) telemetry_snapshot: BTreeMap<String, TelemetrySnapshotEntry>,
    #[serde(default)]
    pub monologue_quiet_until: Option<String>,
    #[serde(default)]
    pub monologue_surface_until: Option<String>,
    #[serde(default)]
    pub recent_emit_fingerprints: Vec<String>,
    #[serde(default)]
    pub recent_emit_messages: Vec<String>,
    #[serde(default)]
    pub monologue_emit_loop_breaker_triggered: bool,
    #[serde(default)]
    pub monologue_candidate_reject_streak: i32,
    #[serde(default)]
    pub monologue_misaligned_streak: i32,
    #[serde(default)]
    pub last_heartbeat_at: Option<String>,
    #[serde(default)]
    pub last_dream_at: Option<String>,
    #[serde(default)]
    pub stop_latch: bool,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub stop_scope: Option<String>,
    #[serde(default)]
    pub stop_state: StopState,
    #[serde(default)]
    pub gate_high_risk_streak: i32,
    #[serde(skip)]
    pub evidence_emit_budget_remaining: i32,
    #[serde(skip)]
    pub evidence_emit_dedup: HashSet<String>,
    #[serde(default)]
    pub meta_cog_event_count: i64,
    #[serde(default)]
    pub last_meta_cog_event: Option<String>,
    #[serde(default)]
    pub last_meta_cog_event_at: Option<String>,
    #[serde(default)]
    pub last_meta_cog_loop_break_reason: Option<String>,
    #[serde(default)]
    pub meta_cog_loop_break_count: i64,
    #[serde(default)]
    pub meta_cog_pending: Option<MetaCogPending>,
    #[serde(default)]
    pub meta_cog_outcome_cycle_streak: i32,
    #[serde(default)]
    pub meta_cog_outcome_no_signal_streak: i32,
    #[serde(default)]
    pub meta_cog_last_outcome: Option<String>,
    #[serde(default)]
    pub meta_cog_last_outcome_at: Option<String>,
    #[serde(default)]
    pub meta_cog_cooldown_until: Option<String>,
    #[serde(default)]
    pub meta_cog_adaptive_multiplier: f32,
    #[serde(default)]
    pub meta_cog_reanchor_attempts: i32,
    #[serde(default)]
    pub ask_budget_remaining: i32,
    #[serde(default)]
    pub ask_budget_max: i32,
    #[serde(default)]
    pub user_refused: bool,
    #[serde(default)]
    pub state_disclosure_suppressed_until: Option<String>,
    #[serde(default)]
    pub state_disclosure_prompt_streak: i32,
    #[serde(default)]
    pub diagnostics_disabled_turns_remaining: i32,
    #[serde(default)]
    pub system_dump_streak: i32,
    #[serde(default)]
    pub monologue_loop_streak: i32,
    #[serde(default)]
    pub ask_loop_breaker_triggered: bool,
    #[serde(default)]
    pub tool_loop_breaker_triggered: bool,
    #[serde(default)]
    pub tool_call_fingerprints: Vec<String>,
    #[serde(default)]
    pub resolved_slots: Vec<String>,
    #[serde(default)]
    pub missing_slots: Vec<String>,
    #[serde(default)]
    pub resolution_mode: Option<String>,
    #[serde(default)]
    pub missing_input_policy: Option<String>,
    #[serde(default)]
    pub last_asked_slots: Vec<String>,
    #[serde(default)]
    pub slot_provenance: BTreeMap<String, String>,
    #[serde(default)]
    pub self_identity_claim_id: Option<String>,
    #[serde(default)]
    pub introspection_verbosity: Option<f32>,
    #[serde(default)]
    pub confirmation_frequency: Option<f32>,
    #[serde(default)]
    pub verify_threshold: Option<f32>,
    #[serde(default)]
    pub drift_sensitivity: Option<f32>,
    #[serde(default)]
    pub introspection_weight: Option<f32>,
    #[serde(default)]
    pub residual_salience_gain: Option<f32>,
    #[serde(default)]
    pub organism_influence_gain: Option<f32>,
    #[serde(default)]
    pub workspace_verbosity: Option<f32>,
}

impl KernelState {
    pub fn default_for(conversation_id: &str) -> Self {
        let initial_goal_thread = "Introduce myself and learn about the user.".to_string();
        Self {
            conversation_id: conversation_id.to_string(),
            mode: KernelMode::Work,
            task_id: None,
            task_phase: TaskPhase::default(),
            pressure_score: 0.0,
            pressure_signals: Vec::new(),
            stance: StanceState::default(),
            last_mode_switch_at: None,
            last_medium_update_at: None,
            last_monologue_at: None,
            last_monologue_tick_id: None,
            last_monologue_started_at: None,
            last_monologue_completed_at: None,
            last_monologue_tick_outcome: None,
            last_monologue_status_emitted: None,
            last_monologue_visible: None,
            last_reflection_applied_at: None,
            monologue_window_start: None,
            monologue_count: 0,
            monologue_suppression_window_start: None,
            monologue_suppression_counts: HashMap::new(),
            monologue_relaxation_level: 1,
            monologue_json_supported: None,
            monologue_json_disabled_until: None,
            monologue_force_primary: false,
            hypothesis_defer_until: None,
            last_hypothesis_promoted_at: None,
            research_window_start: None,
            research_used: 0,
            thread_depth: 0,
            last_processed_dispatch_at: None,
            last_processed_dispatch_id: None,
            last_processed_thread_at: None,
            last_processed_thread_id: None,
            last_outcome_at: None,
            last_semantic_promotion_at: None,
            recent_outcomes: Vec::new(),
            pending_questions: Vec::new(),
            asked_slot_sets: Vec::new(),
            question_fingerprints: Vec::new(),
            recent_questions: Vec::new(),
            refused_slots: Vec::new(),
            refusal_count: 0,
            identity_violation_count: 0,
            identity_violation_window_start: None,
            pending_actions: Vec::new(),
            pending_reframes: Vec::new(),
            pending_angles: Vec::new(),
            last_user_input: None,
            last_user_input_at: None,
            last_user_message_id: None,
            last_input_evidence_event_ids: Vec::new(),
            anchor_epoch: 0,
            user_redirect_turns_remaining: 0,
            redirect_focus: None,
            last_redirect_clarifier_epoch: 0,
            redirect_focus_confirmed_turns: 0,
            redirect_focus_miss_turns: 0,
            redirect_focus_explicit: false,
            last_monologue_anchor_epoch: 0,
            last_assistant_output: None,
            last_assistant_output_no_tags: None,
            last_proactive_emit_at: None,
            proactive_cooldown_until: None,
            last_proactive_question: None,
            last_proactive_memory_pass_at: None,
            last_memory_pass_at: None,
            failure_count: 0,
            tool_failure_count: 0,
            tool_failure_penalties: HashMap::new(),
            uncertainty_count: 0,
            stalled_count: 0,
            halted: false,
            monologue_unhalted_at: None,
            monologue_probation_until: None,
            monologue_health_last_alert_at: None,
            active_threads: Vec::new(),
            self_state: SelfState::default(),
            working_memory: None,
            last_response_summary: None,
            last_response_summary_at: None,
            prompt_section_hashes: HashMap::new(),
            introspection_force: None,
            workspace_goal_thread: Some(initial_goal_thread),
            workspace_active_plan_id: None,
            workspace_goal_stack: Vec::new(),
            goal_loop_turn_count: 0,
            workspace_open_questions: Vec::new(),
            workspace_active_hypotheses: Vec::new(),
            workspace_working_set_topics: Vec::new(),
            workspace_current_focus: None,
            workspace_focus_rationale: None,
            workspace_meta: crate::models::WorkspaceMeta::default(),
            last_workspace_delta_fields: Vec::new(),
            last_workspace_delta_count: 0,
            last_workspace_update_at: None,
            last_focus_change_at: None,
            last_subject_snapshot_hash: None,
            last_subject_snapshot_at: None,
            last_plan_hash: None,
            last_persisted_state_hash: None,
            last_persisted_workspace_hash: None,
            controller_state: None,
            controller_gate: None,
            self_model_version: 0,
            self_model_updated_at: None,
            self_model_unified: None,
            self_model_unified_evidence: None,
            last_self_report_at: None,
            self_report_snapshot: None,
            proaction_throttle_tools_override: None,
            proaction_throttle_threads_override: None,
            last_controller_snapshot_at: None,
            telemetry_snapshot: BTreeMap::new(),
            monologue_quiet_until: None,
            monologue_surface_until: None,
            recent_emit_fingerprints: Vec::new(),
            recent_emit_messages: Vec::new(),
            monologue_emit_loop_breaker_triggered: false,
            monologue_candidate_reject_streak: 0,
            monologue_misaligned_streak: 0,
            last_heartbeat_at: None,
            last_dream_at: None,
            stop_latch: false,
            stop_reason: None,
            stop_scope: None,
            stop_state: StopState::default(),
            gate_high_risk_streak: 0,
            evidence_emit_budget_remaining: 0,
            evidence_emit_dedup: HashSet::new(),
            meta_cog_event_count: 0,
            last_meta_cog_event: None,
            last_meta_cog_event_at: None,
            last_meta_cog_loop_break_reason: None,
            meta_cog_loop_break_count: 0,
            meta_cog_pending: None,
            meta_cog_outcome_cycle_streak: 0,
            meta_cog_outcome_no_signal_streak: 0,
            meta_cog_last_outcome: None,
            meta_cog_last_outcome_at: None,
            meta_cog_cooldown_until: None,
            meta_cog_adaptive_multiplier: 1.0,
            meta_cog_reanchor_attempts: 0,
            ask_budget_remaining: 0,
            ask_budget_max: 0,
            user_refused: false,
            state_disclosure_suppressed_until: None,
            state_disclosure_prompt_streak: 0,
            diagnostics_disabled_turns_remaining: 0,
            system_dump_streak: 0,
            monologue_loop_streak: 0,
            ask_loop_breaker_triggered: false,
            tool_loop_breaker_triggered: false,
            tool_call_fingerprints: Vec::new(),
            resolved_slots: Vec::new(),
            missing_slots: Vec::new(),
            resolution_mode: None,
            missing_input_policy: None,
            last_asked_slots: Vec::new(),
            slot_provenance: BTreeMap::new(),
            self_identity_claim_id: None,
            introspection_verbosity: Some(0.5),
            confirmation_frequency: Some(0.5),
            verify_threshold: Some(0.5),
            drift_sensitivity: Some(0.5),
            introspection_weight: Some(0.5),
            residual_salience_gain: Some(0.5),
            organism_influence_gain: Some(0.5),
            workspace_verbosity: Some(0.5),
        }
    }
}

