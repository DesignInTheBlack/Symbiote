use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CandidateKind {
    UpdateInnerSummary,
    EmitMessage,
    AskUserQuestion,
    FlagForHuman,
    ToolCall,
    SpawnThread,
    WriteEpisodic,
    PromoteSemantic,
    UpdateGoalThread,
    UpdateWorkspace,
    AnchorShift,
    RecordSelfClaim,
    ChangeMode,
    Terminate,
    NoOp,
}

pub fn is_state_change_candidate(kind: &CandidateKind) -> bool {
    matches!(
        kind,
        CandidateKind::UpdateWorkspace
            | CandidateKind::UpdateGoalThread
            | CandidateKind::RecordSelfClaim
            | CandidateKind::UpdateInnerSummary
            | CandidateKind::PromoteSemantic
            | CandidateKind::WriteEpisodic
            | CandidateKind::AnchorShift
            | CandidateKind::ChangeMode
            | CandidateKind::ToolCall
            | CandidateKind::SpawnThread
            | CandidateKind::Terminate
    )
}

pub(crate) fn parse_candidate_kind(raw: &str) -> Option<CandidateKind> {
    match raw.trim().to_lowercase().as_str() {
        "update_inner_summary" => Some(CandidateKind::UpdateInnerSummary),
        "emit_message" => Some(CandidateKind::EmitMessage),
        "ask_user_question" => Some(CandidateKind::AskUserQuestion),
        "flag_for_human" => Some(CandidateKind::FlagForHuman),
        "tool_call" => Some(CandidateKind::ToolCall),
        "spawn_thread" => Some(CandidateKind::SpawnThread),
        "write_episodic" => Some(CandidateKind::WriteEpisodic),
        "promote_semantic" => Some(CandidateKind::PromoteSemantic),
        "update_goal_thread" => Some(CandidateKind::UpdateGoalThread),
        "update_workspace" => Some(CandidateKind::UpdateWorkspace),
        "anchor_shift" => Some(CandidateKind::AnchorShift),
        "record_self_claim" => Some(CandidateKind::RecordSelfClaim),
        "change_mode" => Some(CandidateKind::ChangeMode),
        "terminate" => Some(CandidateKind::Terminate),
        "no_op" | "noop" => Some(CandidateKind::NoOp),
        _ => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub kind: CandidateKind,
    pub payload: Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_event_ids: Vec<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub belief_ids: Vec<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_scope: Option<String>,
    pub rationale: Option<String>,
    pub expected_outcome: Option<String>,
    pub cost: Option<i64>,
    pub urgency: Option<i64>,
    pub source: String,
    pub priority_class: i32,
    pub priority_rank: i32,
    pub created_at: i64,
}

impl Candidate {
    pub fn refresh_meta(&mut self) {
        self.evidence_event_ids = extract_id_list(&self.payload, "evidence_event_ids");
        self.belief_ids = extract_id_list(&self.payload, "belief_ids");
        if self.target_scope.is_none() {
            self.target_scope = extract_target_scope(&self.payload);
        }
    }
}

fn extract_id_list(payload: &Value, key: &str) -> Vec<i64> {
    payload
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    if let Some(n) = item.as_i64() {
                        Some(n)
                    } else if let Some(s) = item.as_str() {
                        s.trim().parse::<i64>().ok()
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn extract_target_scope(payload: &Value) -> Option<String> {
    payload
        .get("target_scope")
        .and_then(|v| v.as_str())
        .or_else(|| payload.get("scope").and_then(|v| v.as_str()))
        .or_else(|| payload.get("target").and_then(|v| v.as_str()))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectedCandidate {
    pub id: String,
    pub kind: CandidateKind,
    pub reason: String,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub is_monologue: Option<bool>,
    #[serde(default)]
    pub payload: Option<Value>,
}

#[derive(Debug, Clone)]
pub(crate) struct CoercedCandidate {
    pub candidate: Option<Candidate>,
    pub speculation_marked: bool,
    pub blocked_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_change_candidate_classification() {
        assert!(is_state_change_candidate(&CandidateKind::UpdateWorkspace));
        assert!(is_state_change_candidate(&CandidateKind::RecordSelfClaim));
        assert!(is_state_change_candidate(&CandidateKind::ToolCall));
        assert!(!is_state_change_candidate(&CandidateKind::EmitMessage));
        assert!(!is_state_change_candidate(&CandidateKind::AskUserQuestion));
        assert!(!is_state_change_candidate(&CandidateKind::NoOp));
    }
}
