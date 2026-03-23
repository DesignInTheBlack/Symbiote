use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ProactionState {
    pub enabled: bool,
    pub mode: String, // metrics | dry_run | active
    pub last_adjusted_at: Option<String>,
    pub cooldown_until: Option<String>,
    pub last_settings_snapshot: Option<Value>,
    pub consecutive_good_windows: i32,
    pub consecutive_bad_windows: i32,
    pub dry_run_completed: bool,
    #[serde(default)]
    pub monologue_relaxation_level: i32,
    #[serde(default)]
    pub monologue_bad_windows: i32,
    #[serde(default)]
    pub monologue_last_eval_hour: Option<String>,
}

impl ProactionState {
    pub fn baseline() -> Self {
        ProactionState {
            enabled: false,
            mode: "metrics".to_string(),
            last_adjusted_at: None,
            cooldown_until: None,
            last_settings_snapshot: None,
            consecutive_good_windows: 0,
            consecutive_bad_windows: 0,
            dry_run_completed: false,
            monologue_relaxation_level: 1,
            monologue_bad_windows: 0,
            monologue_last_eval_hour: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProactionAdjustments {
    pub ask_budget_max: Option<i32>,
    pub loop_similarity_threshold: Option<f32>,
    pub research_budget_per_hour: Option<i64>,
    pub monologue_surface_enabled: Option<bool>,
    pub pending_prompt_alignment_enabled: Option<bool>,
    pub monologue_interval_seconds: Option<i64>,
    pub monologue_max_per_hour: Option<i64>,
    pub tool_preference: Option<String>,
    pub throttle_tools: Option<bool>,
    pub throttle_threads: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProactionMetrics {
    pub window_minutes: i64,
    pub user_turns: i64,
    pub assistant_turns: i64,
    pub tool_calls: i64,
    pub tool_success: i64,
    pub tool_failure: i64,
    pub tool_refusals: i64,
    pub tool_unknown: i64,
    pub ask_loop_breaks: i64,
    pub emit_loop_breaks: i64,
    pub silent_cycles: i64,
    pub empty_responses: i64,
    pub meta_responses: i64,
    pub monologue_ticks: i64,
    pub monologue_suppressed_turns: i64,
    pub monologue_attempted_turns: i64,
    pub monologue_digest_selected: i64,
    pub monologue_digest_injected: i64,
    pub monologue_digest_stale: i64,
    pub monologue_tick_start: i64,
    pub monologue_tick_end: i64,
    pub monologue_timeouts: i64,
    pub monologue_drift_events: i64,
    pub monologue_reanchor_events: i64,
    pub monologue_safety_violations: i64,
    pub gate_decisions: i64,
    pub gate_allow: i64,
    pub gate_allow_notice: i64,
    pub gate_allow_audit: i64,
    pub gate_verify: i64,
    pub gate_defer: i64,
    pub gate_deny: i64,
    pub decision_reports: i64,
    pub no_op_cycles: i64,
    pub monologue_status_cycles: i64,
    pub monologue_visible_cycles: i64,
    pub ui_stall_events: i64,
    pub latency_p95_ms: i64,
    pub user_visible_output_rate: f32,
    pub tool_success_rate: f32,
    pub tool_failure_rate: f32,
    pub tool_refusal_rate: f32,
    pub tool_unknown_rate: f32,
    pub silent_cycle_rate: f32,
    pub empty_response_rate: f32,
    pub meta_response_rate: f32,
    pub ask_loop_rate: f32,
    pub emit_loop_rate: f32,
    pub monologue_output_rate: f32,
    pub monologue_suppression_rate: f32,
    pub monologue_digest_use_rate: f32,
    pub monologue_digest_stale_rate: f32,
    pub monologue_ds_fts_ratio: f32,
    pub monologue_tick_end_rate: f32,
    pub monologue_timeout_rate: f32,
    pub monologue_drift_reanchor_rate: f32,
    pub monologue_safety_violation_rate: f32,
    pub gate_allow_rate: f32,
    pub gate_allow_notice_rate: f32,
    pub gate_allow_audit_rate: f32,
    pub gate_verify_rate: f32,
    pub gate_defer_rate: f32,
    pub gate_deny_rate: f32,
    pub no_op_rate: f32,
    pub monologue_status_rate: f32,
    pub monologue_visible_rate: f32,
    pub ui_stall_rate: f32,
}

#[derive(Debug, Clone)]
pub struct PendingPromptSelection {
    pub prompt_id: String,
    pub prompt: String,
    pub source: String,
    pub overlap_workspace: usize,
    pub overlap_user: usize,
    pub skip_count: i64,
    pub age_seconds: Option<i64>,
    pub exact_open_question: bool,
    pub auto_surface: bool,
    pub force_reason: Option<String>,
    pub intent_kind: Option<String>,
    pub bridge_id: Option<String>,
    pub attempt_count: i64,
    #[allow(dead_code)]
    pub last_asked_at: Option<String>,
    pub expires_at: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ProactiveRunMeta {
    pub run_id: String,
    pub trace_id: String,
    pub deferred: bool,
}
