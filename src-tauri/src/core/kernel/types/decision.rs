use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{AllowedCapabilities, Candidate, RejectedCandidate, StopReason};
use crate::core::kernel::prediction::PredictionCandidate;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SelectedAction {
    DirectAnswer,
    BoundedAnswer,
    ClarifyingQuestion,
    ToolAttempt,
    ExplainBlockers,
}

impl SelectedAction {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            SelectedAction::DirectAnswer => "direct_answer",
            SelectedAction::BoundedAnswer => "bounded_answer",
            SelectedAction::ClarifyingQuestion => "clarifying_question",
            SelectedAction::ToolAttempt => "tool_attempt",
            SelectedAction::ExplainBlockers => "explain_blockers",
        }
    }

    pub(crate) fn from_str(raw: &str) -> Self {
        match raw.trim() {
            "bounded_answer" => SelectedAction::BoundedAnswer,
            "clarifying_question" => SelectedAction::ClarifyingQuestion,
            "tool_attempt" => SelectedAction::ToolAttempt,
            "explain_blockers" => SelectedAction::ExplainBlockers,
            _ => SelectedAction::DirectAnswer,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeliveryArtifact {
    QuestionMark,
    ToolCall,
    BoundedAnswerBlock,
    ExplainBlockers,
    None,
}

impl DeliveryArtifact {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            DeliveryArtifact::QuestionMark => "question_mark",
            DeliveryArtifact::ToolCall => "tool_call",
            DeliveryArtifact::BoundedAnswerBlock => "bounded_answer_block",
            DeliveryArtifact::ExplainBlockers => "explain_blockers",
            DeliveryArtifact::None => "none",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolOutcome {
    Success,
    SoftFail,
    HardFail,
    Timeout,
    Blocked,
}

impl ToolOutcome {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            ToolOutcome::Success => "success",
            ToolOutcome::SoftFail => "soft_fail",
            ToolOutcome::HardFail => "hard_fail",
            ToolOutcome::Timeout => "timeout",
            ToolOutcome::Blocked => "blocked",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionReport {
    pub noop: bool,
    pub minimally_helpful: bool,
    pub cannot_respond: bool,
    pub selected_action: String,
    pub selected_action_delivered: bool,
    pub delivery_artifact: String,
    pub fallback_used: bool,
    #[serde(default)]
    pub fallback_type: Option<String>,
    #[serde(default)]
    pub plan_hash: Option<String>,
    #[serde(default)]
    pub proposal_id: Option<String>,
    #[serde(default)]
    pub plan_state: Option<String>,
    #[serde(default)]
    pub snapshot_hash: Option<String>,
    #[serde(default)]
    pub gate_decision_id: Option<String>,
    #[serde(default)]
    pub gate_decision: Option<String>,
    #[serde(default)]
    pub gate_reasons: Option<Vec<String>>,
    #[serde(default)]
    pub gate_notice: Option<String>,
    #[serde(default)]
    pub gate_penalty: Option<f64>,
    #[serde(default)]
    pub gate_penalty_reasons: Option<Vec<String>>,
    #[serde(default)]
    pub soft_gate_decision: Option<String>,
    #[serde(default)]
    pub soft_gate_reasons: Option<Vec<String>>,
    #[serde(default)]
    pub verification_outcome: Option<String>,
    #[serde(default)]
    pub verification_reasons: Option<Vec<String>>,
    #[serde(default)]
    pub verification_confidence: Option<f64>,
    #[serde(default)]
    pub verification_assumptions_checked: Option<usize>,
    #[serde(default)]
    pub verification_assumptions_failed: Option<usize>,
    #[serde(default)]
    pub verification_conflict_topics: Option<Vec<String>>,
    #[serde(default)]
    pub stop_scope: Option<String>,
    pub allowed_capabilities: AllowedCapabilities,
    #[serde(default)]
    pub stop_reasons: Vec<StopReason>,
    #[serde(default)]
    pub normalized_stop_reasons: Vec<String>,
    pub blocked_candidates_count: usize,
    #[serde(alias = "top_block_reasons")]
    pub top_3_block_reasons: Vec<String>,
    #[serde(default)]
    pub contract_violation_count: Option<usize>,
    #[serde(default)]
    pub contract_violation_rate: Option<f64>,
    #[serde(default)]
    pub anchor_hits: Option<usize>,
    #[serde(default)]
    pub prompt_tokens_used: Option<usize>,
    #[serde(default)]
    pub tier_trim_summary: Option<Value>,
    #[serde(default)]
    pub tool_attempted: Option<String>,
    #[serde(default)]
    pub tool_outcome: Option<String>,
    #[serde(default)]
    pub monologue_tick_outcome: Option<String>,
    #[serde(default)]
    pub monologue_status_emitted: Option<bool>,
    #[serde(default)]
    pub monologue_visible: Option<bool>,
    #[serde(default)]
    pub unblock_instructions: Option<String>,
    #[serde(default)]
    pub selected_action_source: Option<String>,
    #[serde(default)]
    pub background_jobs_dropped: Option<bool>,
    #[serde(default)]
    pub rationale: Option<String>,
    #[serde(default)]
    pub residual_influence_mode: Option<String>,
    #[serde(default)]
    pub residual_shadow_would_change: Option<bool>,
    #[serde(default)]
    pub residual_shadow_impact_pct: Option<f64>,
    #[serde(default)]
    pub residual_bias: Option<f64>,
}

impl Default for DecisionReport {
    fn default() -> Self {
        Self {
            noop: false,
            minimally_helpful: false,
            cannot_respond: false,
            selected_action: SelectedAction::DirectAnswer.as_str().to_string(),
            selected_action_delivered: false,
            delivery_artifact: DeliveryArtifact::None.as_str().to_string(),
            fallback_used: false,
            fallback_type: None,
            plan_hash: None,
            proposal_id: None,
            plan_state: None,
            snapshot_hash: None,
            gate_decision_id: None,
            gate_decision: None,
            gate_reasons: None,
            gate_notice: None,
            gate_penalty: None,
            gate_penalty_reasons: None,
            soft_gate_decision: None,
            soft_gate_reasons: None,
            verification_outcome: None,
            verification_reasons: None,
            verification_confidence: None,
            verification_assumptions_checked: None,
            verification_assumptions_failed: None,
            verification_conflict_topics: None,
            stop_scope: None,
            allowed_capabilities: AllowedCapabilities::default(),
            stop_reasons: Vec::new(),
            normalized_stop_reasons: Vec::new(),
            blocked_candidates_count: 0,
            top_3_block_reasons: Vec::new(),
            contract_violation_count: None,
            contract_violation_rate: None,
            anchor_hits: None,
            prompt_tokens_used: None,
            tier_trim_summary: None,
            tool_attempted: None,
            tool_outcome: None,
            monologue_tick_outcome: None,
            monologue_status_emitted: None,
            monologue_visible: None,
            unblock_instructions: None,
            selected_action_source: None,
            background_jobs_dropped: None,
            rationale: None,
            residual_influence_mode: None,
            residual_shadow_would_change: None,
            residual_shadow_impact_pct: None,
            residual_bias: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelDecision {
    pub accepted: Vec<Candidate>,
    pub rejected: Vec<RejectedCandidate>,
    pub caps_applied: Vec<String>,
    #[serde(default)]
    pub report: DecisionReport,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PredictionResponse {
    pub predictions: Option<Vec<PredictionCandidate>>,
    pub rejection_reason: Option<String>,
}
