use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::db::Db;
use crate::core::kernel::{Candidate, CandidateKind, StopState};
use crate::core::kernel::workspace::{extract_goal_stack_from_payload, goal_stack_has_evidence};
use crate::core::self_claims;
use crate::core::subject_state::SubjectState;
use crate::core::world_model;
use crate::models::GoalStackItem;

pub struct ActionProposalRecord {
    pub proposal_id: String,
    pub intent: String,
    pub steps_json: String,
    pub plan_hash: String,
    pub plan_state: String,
    pub risk_level: String,
    pub required_claims_json: String,
    pub required_error_bounds_json: Option<String>,
    pub verification_plan_json: Option<String>,
    pub success_criteria_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanVerificationResult {
    pub outcome: String,
    pub reasons: Vec<String>,
    pub assumptions: Vec<String>,
    pub assumptions_checked: usize,
    pub assumptions_failed: usize,
    pub conflict_topics: Vec<String>,
    pub confidence: f64,
}

pub struct GateDecisionRecord {
    pub decision_id: String,
    pub decision: String,
    pub evidence_refs_json: String,
    pub metrics_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateSignals {
    pub anchor_hits: usize,
    pub novelty_score: f32,
    pub uncertainty_score: f32,
    pub risk_score: f32,
    pub tool_misuse_risk: f32,
    pub requires_audit: bool,
    pub low_evidence: bool,
    pub candidate_disagreement: usize,
    pub recent_tool_failures: usize,
    pub topic_shift_detected: bool,
    pub toolchain_unseen: bool,
    pub new_objective: bool,
    pub high_risk_streak: i32,
    pub default_soft: bool,
    pub self_report_channel: bool,
    #[serde(default)]
    pub qualia_tag: Option<String>,
    #[serde(default)]
    pub qualia_intensity: f32,
    #[serde(default)]
    pub qualia_confidence: f32,
    #[serde(default)]
    pub qualia_novelty_delta: f32,
    #[serde(default)]
    pub qualia_uncertainty_delta: f32,
}

impl GateSignals {
    pub fn baseline() -> Self {
        Self {
            anchor_hits: 1,
            novelty_score: 0.0,
            uncertainty_score: 0.0,
            risk_score: 0.0,
            tool_misuse_risk: 0.0,
            requires_audit: false,
            low_evidence: false,
            candidate_disagreement: 0,
            recent_tool_failures: 0,
            topic_shift_detected: false,
            toolchain_unseen: false,
            new_objective: false,
            high_risk_streak: 0,
            default_soft: true,
            self_report_channel: true,
            qualia_tag: None,
            qualia_intensity: 0.0,
            qualia_confidence: 0.0,
            qualia_novelty_delta: 0.0,
            qualia_uncertainty_delta: 0.0,
        }
    }
}

fn is_user_invoked_self_awareness(candidate: &Candidate) -> bool {
    if !matches!(candidate.kind, CandidateKind::RecordSelfClaim) {
        return false;
    }
    let source_type = candidate.payload.get("source_type").and_then(|v| v.as_str());
    if source_type != Some("self_awareness_query") {
        return false;
    }
    let claim_text = candidate
        .payload
        .get("claim_text")
        .and_then(|v| v.as_str())
        .or_else(|| candidate.payload.get("claim").and_then(|v| v.as_str()))
        .or_else(|| candidate.payload.get("text").and_then(|v| v.as_str()))
        .unwrap_or("")
        .trim();
    if claim_text.is_empty() {
        return false;
    }
    self_claims::is_self_awareness_claim(claim_text)
}

fn status_is_complete(status: Option<&str>) -> bool {
    let Some(status) = status else {
        return false;
    };
    matches!(
        status.trim().to_lowercase().as_str(),
        "done" | "complete" | "completed" | "finished"
    )
}

fn goal_stack_payload_signals_advancement(items: &[GoalStackItem]) -> bool {
    items.iter().any(|item| {
        if item.current_step_index > 0 {
            return true;
        }
        if status_is_complete(item.status.as_deref()) {
            return true;
        }
        item.steps
            .iter()
            .any(|step| status_is_complete(step.status.as_deref()))
    })
}

fn normalize_text_items(items: Vec<String>) -> Vec<String> {
    items
        .into_iter()
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect()
}

fn extract_string_list(value: Option<&Value>) -> Vec<String> {
    let Some(value) = value else {
        return Vec::new();
    };
    match value {
        Value::Array(items) => normalize_text_items(
            items
                .iter()
                .filter_map(|item| item.as_str())
                .map(|item| item.to_string())
                .collect(),
        ),
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                Vec::new()
            } else if trimmed.contains('\n') || trimmed.contains(';') {
                normalize_text_items(
                    trimmed
                        .split(['\n', ';'])
                        .map(|item| item.trim().to_string())
                        .collect(),
                )
            } else {
                vec![trimmed.to_string()]
            }
        }
        Value::Object(_) => Vec::new(),
        _ => Vec::new(),
    }
}

fn extract_assumptions(payload: &Value) -> Vec<String> {
    let mut assumptions = extract_string_list(payload.get("assumptions"));
    if assumptions.is_empty() {
        assumptions = extract_string_list(payload.get("assumption"));
    }
    if assumptions.is_empty() {
        assumptions = extract_string_list(payload.get("assumptions_text"));
    }
    if assumptions.is_empty() {
        assumptions = extract_string_list(payload.get("assumption_text"));
    }
    assumptions
}

fn extract_risks(payload: &Value) -> Vec<String> {
    let mut risks = extract_string_list(payload.get("risks"));
    if risks.is_empty() {
        risks = extract_string_list(payload.get("risk"));
    }
    if risks.is_empty() {
        risks = extract_string_list(payload.get("risk_notes"));
    }
    risks
}

fn extract_success_criteria(payload: &Value) -> Vec<String> {
    let mut criteria = extract_string_list(payload.get("success_criteria"));
    if criteria.is_empty() {
        criteria = extract_string_list(payload.get("success"));
    }
    if criteria.is_empty() {
        criteria = extract_string_list(payload.get("success_criteria_text"));
    }
    criteria
}

fn build_plan_steps(candidate: &Candidate) -> Vec<Value> {
    if let Some(steps) = candidate.payload.get("steps").and_then(|v| v.as_array()) {
        return steps
            .iter()
            .enumerate()
            .map(|(idx, step)| match step {
                Value::Object(_) => {
                    let mut obj = step.clone();
                    if let Value::Object(map) = &mut obj {
                        map.entry("index".to_string())
                            .or_insert_with(|| Value::Number((idx + 1).into()));
                        map.entry("candidate_kind".to_string())
                            .or_insert_with(|| Value::String(format!("{:?}", candidate.kind)));
                        map.entry("candidate_id".to_string())
                            .or_insert_with(|| Value::String(candidate.id.clone()));
                    }
                    obj
                }
                _ => json!({
                    "index": idx + 1,
                    "candidate_kind": format!("{:?}", candidate.kind),
                    "candidate_id": candidate.id.clone(),
                    "description": step,
                }),
            })
            .collect();
    }

    vec![json!({
        "index": 1,
        "candidate_kind": format!("{:?}", candidate.kind),
        "candidate_id": candidate.id.clone(),
        "description": candidate
            .rationale
            .clone()
            .unwrap_or_else(|| format!("{:?}", candidate.kind)),
        "expected_outcome": candidate.expected_outcome,
        "payload": candidate.payload.clone(),
    })]
}

fn canonicalize_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut sorted = serde_json::Map::new();
            for key in keys {
                if let Some(val) = map.get(key) {
                    sorted.insert(key.clone(), canonicalize_value(val));
                }
            }
            Value::Object(sorted)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(canonicalize_value).collect()),
        _ => value.clone(),
    }
}

fn hash_plan_value(value: &Value) -> String {
    let canonical = canonicalize_value(value);
    let json_str = serde_json::to_string(&canonical).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(json_str.as_bytes());
    hex::encode(hasher.finalize())
}

fn extract_assumptions_from_json(raw: Option<&str>) -> Vec<String> {
    let Some(raw) = raw else {
        return Vec::new();
    };
    let Ok(parsed) = serde_json::from_str::<Value>(raw) else {
        return Vec::new();
    };
    extract_assumptions(&parsed)
}

pub fn build_action_proposal(candidate: &Candidate) -> ActionProposalRecord {
    let proposal_id = Uuid::new_v4().to_string();
    let intent = format!("{:?}", candidate.kind);
    let assumptions = extract_assumptions(&candidate.payload);
    let risks = extract_risks(&candidate.payload);
    let success_criteria = extract_success_criteria(&candidate.payload);
    let steps = build_plan_steps(candidate);
    let plan_value = json!({
        "candidate_id": candidate.id.clone(),
        "candidate_kind": format!("{:?}", candidate.kind),
        "intent": intent.clone(),
        "steps": steps,
        "assumptions": assumptions.clone(),
        "risks": risks.clone(),
        "success_criteria": success_criteria.clone(),
        "expected_outcome": candidate.expected_outcome.clone(),
        "rationale": candidate.rationale.clone(),
        "source": candidate.source.clone(),
        "payload": candidate.payload.clone(),
    });
    let steps_json = serde_json::to_string(&plan_value).unwrap_or_else(|_| "{}".to_string());
    let plan_hash = hash_plan_value(&plan_value);
    let risk_level = match candidate.kind {
        CandidateKind::ToolCall => "high",
        CandidateKind::RecordSelfClaim => "medium",
        CandidateKind::UpdateWorkspace | CandidateKind::UpdateGoalThread => "medium",
        CandidateKind::EmitMessage | CandidateKind::AskUserQuestion | CandidateKind::FlagForHuman => "low",
        _ => "low",
    }
    .to_string();
    let required_claims_json = candidate
        .payload
        .get("required_claims")
        .and_then(|v| serde_json::to_string(v).ok())
        .unwrap_or_else(|| "[]".to_string());
    let required_error_bounds_json = if risk_level != "low" {
        Some(json!({"max_residual": 0.5}).to_string())
    } else {
        None
    };
    let verification_plan = json!({
        "assumptions": assumptions,
        "risks": risks,
        "required_claims": candidate.payload.get("required_claims").cloned().unwrap_or_else(|| json!([])),
        "evidence_event_ids": candidate.payload.get("evidence_event_ids").cloned().unwrap_or_else(|| json!([])),
        "belief_ids": candidate.payload.get("belief_ids").cloned().unwrap_or_else(|| json!([])),
    });
    let verification_plan_json =
        Some(serde_json::to_string(&verification_plan).unwrap_or_else(|_| "{}".to_string()));
    let success_criteria_json = if success_criteria.is_empty() {
        "\"response_delivered\"".to_string()
    } else {
        serde_json::to_string(&success_criteria).unwrap_or_else(|_| "\"response_delivered\"".to_string())
    };

    ActionProposalRecord {
        proposal_id,
        intent,
        steps_json,
        plan_hash,
        plan_state: "draft".to_string(),
        risk_level,
        required_claims_json,
        required_error_bounds_json,
        verification_plan_json,
        success_criteria_json,
    }
}

pub fn verify_action_proposal(
    proposal: &ActionProposalRecord,
    world_model_snapshot: &world_model::WorldModelSnapshot,
) -> PlanVerificationResult {
    let mut assumptions = extract_assumptions_from_json(proposal.verification_plan_json.as_deref());
    if assumptions.is_empty() {
        assumptions = extract_assumptions_from_json(Some(&proposal.steps_json));
    }

    let mut reasons = Vec::new();
    let mut conflict_topics = Vec::new();
    let mut outcome = "ALLOW".to_string();
    let assumptions_checked = assumptions.len();
    let mut matched_flags = vec![false; assumptions_checked];
    let mut conflict_flags = vec![false; assumptions_checked];

    if assumptions.is_empty() {
        reasons.push("no_assumptions".to_string());
        return PlanVerificationResult {
            outcome,
            reasons,
            assumptions,
            assumptions_checked,
            assumptions_failed: 0,
            conflict_topics,
            confidence: 1.0,
        };
    }

    let world_model_empty = world_model_snapshot.entities.is_empty()
        && world_model_snapshot.facts.is_empty()
        && world_model_snapshot.relations.is_empty();
    if world_model_empty {
        reasons.push("world_model_empty".to_string());
        outcome = "VERIFY".to_string();
    }

    let mut known_terms: Vec<String> = Vec::new();
    for entity in &world_model_snapshot.entities {
        if !entity.label.trim().is_empty() {
            known_terms.push(entity.label.to_lowercase());
        }
        for alias in &entity.aliases {
            if !alias.trim().is_empty() {
                known_terms.push(alias.to_lowercase());
            }
        }
        for key in &entity.keys {
            if !key.trim().is_empty() {
                known_terms.push(key.to_lowercase());
            }
        }
    }
    for fact in &world_model_snapshot.facts {
        if !fact.key.trim().is_empty() {
            known_terms.push(fact.key.to_lowercase());
        }
        if !fact.entity_label.trim().is_empty() {
            known_terms.push(fact.entity_label.to_lowercase());
        }
    }
    for rel in &world_model_snapshot.relations {
        if !rel.rel_type.trim().is_empty() {
            known_terms.push(rel.rel_type.to_lowercase());
        }
        for participant in &rel.participants {
            if !participant.entity_label.trim().is_empty() {
                known_terms.push(participant.entity_label.to_lowercase());
            }
        }
    }

    let mut matched_count = 0usize;
    for (idx, assumption) in assumptions.iter().enumerate() {
        let lowered = assumption.to_lowercase();
        let mut matched = false;
        for term in known_terms.iter() {
            if term.len() < 3 {
                continue;
            }
            if lowered.contains(term) {
                matched = true;
                break;
            }
        }
        if matched {
            matched_count += 1;
            matched_flags[idx] = true;
        }
        for conflict in &world_model_snapshot.conflicts {
            let topic = conflict.topic_key.to_lowercase();
            if topic.is_empty() {
                continue;
            }
            if lowered.contains(&topic) {
                let priority = conflict.priority.to_lowercase();
                let status = conflict.status.to_lowercase();
                conflict_topics.push(conflict.topic_key.clone());
                conflict_flags[idx] = true;
                if status == "open" || priority == "high" || priority == "critical" {
                    outcome = "DEFER".to_string();
                    reasons.push("assumption_conflict_open".to_string());
                } else {
                    reasons.push("assumption_conflict_present".to_string());
                    if outcome != "DEFER" {
                        outcome = "VERIFY".to_string();
                    }
                }
            }
        }
    }

    let unmatched = assumptions_checked.saturating_sub(matched_count);
    if unmatched > 0 && !world_model_empty {
        reasons.push("assumptions_unmatched_world_model".to_string());
        if outcome == "ALLOW" {
            outcome = "VERIFY".to_string();
        }
    }

    if world_model_snapshot.conflict_count > 0 {
        reasons.push("world_model_conflicts_present".to_string());
    }

    let mut assumptions_failed = 0usize;
    if assumptions_checked > 0 {
        for idx in 0..assumptions_checked {
            if conflict_flags[idx] || (!matched_flags[idx] && !world_model_empty) {
                assumptions_failed += 1;
            }
        }
    }

    let confidence = if assumptions_checked == 0 {
        1.0
    } else {
        (matched_count as f64 / assumptions_checked as f64).clamp(0.0, 1.0)
    };

    PlanVerificationResult {
        outcome,
        reasons: normalize_text_items(reasons),
        assumptions,
        assumptions_checked,
        assumptions_failed,
        conflict_topics: normalize_text_items(conflict_topics),
        confidence,
    }
}

pub fn apply_verification_to_gate(
    gate: &mut GateDecisionRecord,
    verification: &PlanVerificationResult,
) {
    let mut evidence_refs = serde_json::from_str::<Value>(&gate.evidence_refs_json).unwrap_or_else(|_| json!({}));
    let mut reasons = evidence_refs
        .get("reasons")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect::<Vec<String>>();
    if verification.outcome == "DEFER" {
        reasons.push("plan_verification_defer".to_string());
        if gate.decision != "DENY" {
            gate.decision = "DEFER".to_string();
        }
    } else if verification.outcome == "VERIFY" {
        reasons.push("plan_verification_verify".to_string());
        if matches!(
            gate.decision.as_str(),
            "ALLOW" | "ALLOW_WITH_NOTICE" | "ALLOW_WITH_AUDIT"
        ) {
            gate.decision = "VERIFY".to_string();
        }
    }
    if let Some(obj) = evidence_refs.as_object_mut() {
        obj.insert("reasons".to_string(), json!(reasons));
    } else {
        evidence_refs = json!({ "reasons": reasons });
    }
    gate.evidence_refs_json = evidence_refs.to_string();

    let mut metrics = serde_json::from_str::<Value>(&gate.metrics_json).unwrap_or_else(|_| json!({}));
    if let Some(obj) = metrics.as_object_mut() {
        obj.insert(
            "plan_verification".to_string(),
            json!({
                "outcome": verification.outcome,
                "confidence": verification.confidence,
                "assumptions_checked": verification.assumptions_checked,
                "assumptions_failed": verification.assumptions_failed,
                "reasons": verification.reasons,
                "conflict_topics": verification.conflict_topics,
            }),
        );
    }
    gate.metrics_json = metrics.to_string();
}

pub fn build_gate_decision_legacy(
    subject_state: &SubjectState,
    candidate: &Candidate,
    stop_state: &StopState,
) -> GateDecisionRecord {
    let decision_id = Uuid::new_v4().to_string();
    let mut decision_rank = 0;
    let mut reasons = Vec::new();

    let mut bump = |rank: i32, reason: &str| {
        if rank > decision_rank {
            decision_rank = rank;
        }
        reasons.push(reason.to_string());
    };

    let allow_self_awareness = is_user_invoked_self_awareness(candidate)
        && subject_state.workspace.ignition.active;

    if stop_state.active {
        bump(3, "stop_state_active");
    }

    if matches!(candidate.kind, CandidateKind::RecordSelfClaim) {
        let has_evidence = candidate
            .payload
            .get("evidence_event_ids")
            .and_then(|v| v.as_array())
            .map(|arr| !arr.is_empty())
            .unwrap_or(false);
        if !has_evidence {
            bump(2, "self_claim_missing_evidence");
        }
    }

    if matches!(candidate.kind, CandidateKind::UpdateWorkspace) {
        let goal_stack = extract_goal_stack_from_payload(&candidate.payload);
        if !goal_stack.is_empty() {
            let has_payload_evidence = candidate
                .payload
                .get("evidence_event_ids")
                .and_then(|v| v.as_array())
                .map(|arr| !arr.is_empty())
                .unwrap_or(false)
                || candidate
                    .payload
                    .get("belief_ids")
                    .and_then(|v| v.as_array())
                    .map(|arr| !arr.is_empty())
                    .unwrap_or(false);
            let has_evidence = has_payload_evidence || goal_stack_has_evidence(&goal_stack);
            if goal_stack_payload_signals_advancement(&goal_stack) && !has_evidence {
                bump(2, "goal_stack_missing_evidence");
            }
        }
    }

    if !allow_self_awareness {
        if subject_state.error_state.open_error_count > 0
            && matches!(candidate.kind, CandidateKind::ToolCall | CandidateKind::RecordSelfClaim)
        {
            bump(1, "open_error_events");
        }

        if subject_state.organism.integrity_risk > 0.7 {
            bump(2, "organism_integrity_risk");
        }

        if subject_state.organism.social_alignment < 0.3 {
            bump(1, "organism_social_alignment_low");
        }

        if subject_state.attention.meta_confidence < 0.3
            && matches!(
                candidate.kind,
                CandidateKind::EmitMessage
                    | CandidateKind::AskUserQuestion
                    | CandidateKind::ToolCall
                    | CandidateKind::RecordSelfClaim
            )
        {
            bump(1, "attention_low_confidence");
        }

        if subject_state.self_model.controller_state.drift_score > 0.6 {
            bump(1, "controller_drift_score");
        }

        if subject_state
            .error_state
            .recent_residuals
            .iter()
            .any(|res| res.normalized_residual.abs() > 2.0)
        {
            bump(1, "predictive_residual_high");
        }

        if subject_state
            .error_state
            .diagnosis_flags
            .iter()
            .any(|flag| flag == "diagnose_loop")
        {
            bump(2, "diagnosis_loop");
        }

        if !subject_state.workspace.ignition.active && matches!(candidate.kind, CandidateKind::RecordSelfClaim) {
            bump(2, "no_ignition");
        }

        if subject_state.workspace.broadcast_refs.is_empty()
            && matches!(
                candidate.kind,
                CandidateKind::EmitMessage
                    | CandidateKind::AskUserQuestion
                    | CandidateKind::ToolCall
                    | CandidateKind::RecordSelfClaim
            )
        {
            bump(1, "broadcast_missing");
        }

        if subject_state.qualia.last_reward.unwrap_or(0.0) < -0.4
            && matches!(candidate.kind, CandidateKind::ToolCall | CandidateKind::RecordSelfClaim)
        {
            bump(1, "qualia_negative_reward");
        }
    }

    let decision = match decision_rank {
        3 => "DENY",
        2 => "DEFER",
        1 => "VERIFY",
        _ => "ALLOW",
    }
    .to_string();

    let evidence_refs_json = json!({
        "reasons": reasons,
        "open_error_ids": subject_state.error_state.open_error_ids,
    })
    .to_string();
    let metrics_json = json!({
        "organism": subject_state.organism,
        "ignition": subject_state.workspace.ignition,
        "attention_confidence": subject_state.attention.meta_confidence,
        "drift_score": subject_state.self_model.controller_state.drift_score,
        "qualia_last_reward": subject_state.qualia.last_reward,
    })
    .to_string();

    GateDecisionRecord {
        decision_id,
        decision,
        evidence_refs_json,
        metrics_json,
    }
}

pub fn build_gate_decision(
    subject_state: &SubjectState,
    candidate: &Candidate,
    stop_state: &StopState,
    signals: &GateSignals,
) -> GateDecisionRecord {
    let decision_id = Uuid::new_v4().to_string();
    let mut reasons: Vec<String> = Vec::new();

    if stop_state.active {
        reasons.push("stop_state_active".to_string());
    }

    let allow_self_awareness = is_user_invoked_self_awareness(candidate)
        && subject_state.workspace.ignition.active;

    let mut missing_evidence = false;
    if matches!(candidate.kind, CandidateKind::RecordSelfClaim) {
        let has_evidence = candidate
            .payload
            .get("evidence_event_ids")
            .and_then(|v| v.as_array())
            .map(|arr| !arr.is_empty())
            .unwrap_or(false);
        let has_beliefs = candidate
            .payload
            .get("belief_ids")
            .and_then(|v| v.as_array())
            .map(|arr| !arr.is_empty())
            .unwrap_or(false);
        if !has_evidence && !has_beliefs {
            missing_evidence = true;
            reasons.push("self_claim_missing_evidence".to_string());
        }
    }

    let mut soft_block = false;
    if !allow_self_awareness {
        if subject_state.error_state.open_error_count > 0
            && matches!(candidate.kind, CandidateKind::ToolCall | CandidateKind::RecordSelfClaim)
        {
            reasons.push("open_error_events".to_string());
            soft_block = true;
        }

        if subject_state.organism.integrity_risk > 0.7 {
            reasons.push("organism_integrity_risk".to_string());
            soft_block = true;
        }

        if subject_state.organism.social_alignment < 0.3 {
            reasons.push("organism_social_alignment_low".to_string());
            soft_block = true;
        }

        if subject_state.attention.meta_confidence < 0.3
            && matches!(
                candidate.kind,
                CandidateKind::EmitMessage
                    | CandidateKind::AskUserQuestion
                    | CandidateKind::ToolCall
                    | CandidateKind::RecordSelfClaim
            )
        {
            reasons.push("attention_low_confidence".to_string());
            soft_block = true;
        }

        if subject_state.self_model.controller_state.drift_score > 0.6 {
            reasons.push("controller_drift_score".to_string());
            soft_block = true;
        }

        if subject_state
            .error_state
            .recent_residuals
            .iter()
            .any(|res| res.normalized_residual.abs() > 2.0)
        {
            reasons.push("predictive_residual_high".to_string());
            soft_block = true;
        }

        if subject_state
            .error_state
            .diagnosis_flags
            .iter()
            .any(|flag| flag == "diagnose_loop")
        {
            reasons.push("diagnosis_loop".to_string());
            soft_block = true;
        }

        if !subject_state.workspace.ignition.active
            && matches!(candidate.kind, CandidateKind::RecordSelfClaim)
        {
            reasons.push("no_ignition".to_string());
            soft_block = true;
        }

        if subject_state.workspace.broadcast_refs.is_empty()
            && matches!(
                candidate.kind,
                CandidateKind::EmitMessage
                    | CandidateKind::AskUserQuestion
                    | CandidateKind::ToolCall
                    | CandidateKind::RecordSelfClaim
            )
        {
            reasons.push("broadcast_missing".to_string());
            soft_block = true;
        }

        if subject_state.qualia.last_reward.unwrap_or(0.0) < -0.4
            && matches!(candidate.kind, CandidateKind::ToolCall | CandidateKind::RecordSelfClaim)
        {
            reasons.push("qualia_negative_reward".to_string());
            soft_block = true;
        }
    }

    if signals.novelty_score >= 0.60 {
        reasons.push("novelty_high".to_string());
    }
    if signals.uncertainty_score >= 0.60 {
        reasons.push("uncertainty_high".to_string());
    }
    if signals.requires_audit {
        reasons.push("audit_required".to_string());
    }

    let mut decision = "ALLOW".to_string();
    if signals.high_risk_streak >= 2 {
        reasons.push("multi_signal_risk".to_string());
        decision = "DENY".to_string();
    } else if stop_state.active {
        decision = "DEFER".to_string();
    } else if signals.requires_audit {
        decision = if signals.default_soft {
            "ALLOW_WITH_AUDIT".to_string()
        } else {
            "VERIFY".to_string()
        };
    } else if missing_evidence && !signals.self_report_channel {
        decision = "VERIFY".to_string();
    } else if !signals.default_soft && soft_block {
        decision = "VERIFY".to_string();
    } else if signals.default_soft
        && (soft_block || missing_evidence || signals.novelty_score >= 0.60 || signals.uncertainty_score >= 0.60)
    {
        decision = "ALLOW_WITH_NOTICE".to_string();
    }

    let evidence_refs_json = json!({
        "reasons": reasons,
        "open_error_ids": subject_state.error_state.open_error_ids,
    })
    .to_string();
    let metrics_json = json!({
        "organism": subject_state.organism,
        "ignition": subject_state.workspace.ignition,
        "attention_confidence": subject_state.attention.meta_confidence,
        "drift_score": subject_state.self_model.controller_state.drift_score,
        "qualia_last_reward": subject_state.qualia.last_reward,
        "gate_signals": signals,
    })
    .to_string();

    GateDecisionRecord {
        decision_id,
        decision,
        evidence_refs_json,
        metrics_json,
    }
}

pub async fn persist_action_proposal(
    db: &Db,
    snapshot_hash: &str,
    proposal: &ActionProposalRecord,
) -> Result<(), String> {
    sqlx::query(
        "INSERT INTO action_proposals
         (proposal_id, snapshot_hash, intent, steps_json, plan_hash, plan_state, risk_level, required_claims_json, required_error_bounds_json, verification_plan_json, success_criteria_json, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
    )
    .bind(&proposal.proposal_id)
    .bind(snapshot_hash)
    .bind(&proposal.intent)
    .bind(&proposal.steps_json)
    .bind(&proposal.plan_hash)
    .bind(&proposal.plan_state)
    .bind(&proposal.risk_level)
    .bind(&proposal.required_claims_json)
    .bind(&proposal.required_error_bounds_json)
    .bind(&proposal.verification_plan_json)
    .bind(&proposal.success_criteria_json)
    .execute(&db.pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub async fn persist_gate_decision(
    db: &Db,
    snapshot_hash: &str,
    proposal_id: &str,
    gate: &GateDecisionRecord,
) -> Result<(), String> {
    sqlx::query(
        "INSERT INTO gate_decisions
         (decision_id, proposal_id, snapshot_hash, decision, evidence_refs_json, metrics_json, created_at)
         VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
    )
    .bind(&gate.decision_id)
    .bind(proposal_id)
    .bind(snapshot_hash)
    .bind(&gate.decision)
    .bind(&gate.evidence_refs_json)
    .bind(&gate.metrics_json)
    .execute(&db.pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;

    use crate::core::attention_model::{AttentionModel, AttentionReason};
    use crate::core::kernel::{Candidate, CandidateKind, StopState};
    use crate::core::organism::OrganismState;
    use crate::core::qualia::{QualiaLabelSummary, QualiaState};
    use crate::core::subject_state::{CalibrationKnobs, ErrorState, ResidualSummary, SelfModelState, SubjectState, WorldModelDelta};
    use crate::core::workspace::{WorkspaceIgnition, WorkspaceState};
    use crate::models::ControllerState;

    fn base_subject_state() -> SubjectState {
        let now = Utc::now().to_rfc3339();
        SubjectState {
            workspace: WorkspaceState {
                timestamp: now.clone(),
                slots: 3,
                candidates: Vec::new(),
                winners: Vec::new(),
                broadcast_refs: vec!["focus".to_string()],
                ignition: WorkspaceIgnition {
                    active: true,
                    duration_ticks: 2,
                    ignition_score: 0.8,
                },
            },
            organism: OrganismState {
                timestamp: now.clone(),
                arousal: 0.2,
                stress: 0.2,
                fatigue: 0.1,
                uncertainty_pressure: 0.1,
                social_alignment: 0.8,
                integrity_risk: 0.1,
            },
            self_model: SelfModelState {
                self_identity_claim_id: None,
                identity_confidence: 0.6,
                identity_uncertainty_note: None,
                last_reflection_at: None,
                internal_state_summary: serde_json::json!({}),
                internal_state_map_version: None,
                unified_state: serde_json::json!({}),
                unified_state_evidence: serde_json::json!({}),
                unified_state_updated_at: None,
                goals: Vec::new(),
                calibration: CalibrationKnobs {
                    introspection_verbosity: 0.5,
                    confirmation_frequency: 0.5,
                    verify_threshold: 0.5,
                    drift_sensitivity: 0.5,
                    introspection_weight: 0.5,
                    residual_salience_gain: 0.5,
                    organism_influence_gain: 0.5,
                    workspace_verbosity: 0.5,
                },
                controller_state: ControllerState::default(),
                conflicts_count: 0,
            },
            error_state: ErrorState {
                recent_residuals: Vec::new(),
                open_error_ids: Vec::new(),
                open_error_count: 0,
                pattern_flags: Vec::new(),
                diagnosis_flags: Vec::new(),
            },
            attention: AttentionModel {
                timestamp: now.clone(),
                current_focus_refs: vec!["focus".to_string()],
                why_focused: vec![AttentionReason {
                    source: "workspace_broadcast".to_string(),
                    weight: 0.8,
                }],
                meta_confidence: 0.8,
                next_focus_prediction: Some("focus".to_string()),
            },
            attention_schema: crate::models::AttentionSchemaState::default(),
            qualia: QualiaState {
                timestamp: now.clone(),
                dominant_tag: None,
                dominant_intensity: 0.0,
                recent_labels: Vec::<QualiaLabelSummary>::new(),
                last_reward: Some(0.2),
                predicted_tag: None,
                prediction_confidence: 0.0,
                matched_workspace_refs: Vec::new(),
            },
            world_model: crate::core::world_model::WorldModelSnapshot::default(),
            world_model_delta: WorldModelDelta::default(),
            monologue_updates: Vec::new(),
            plan_hash: None,
            updated_at: now,
        }
    }

    fn candidate(kind: CandidateKind, payload: serde_json::Value) -> Candidate {
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
            priority_class: 0,
            priority_rank: 0,
            created_at: 0,
        };
        candidate.refresh_meta();
        candidate
    }

    #[test]
    fn counterfactual_workspace_broadcast_ablation_changes_gate() {
        let stop = StopState::default();
        let base = base_subject_state();
        let signals = GateSignals::baseline();
        let candidate = candidate(CandidateKind::EmitMessage, json!({"content": "hi"}));
        let gate = build_gate_decision(&base, &candidate, &stop, &signals);
        assert_eq!(gate.decision, "ALLOW");

        let mut ablated = base;
        ablated.workspace.broadcast_refs.clear();
        let gate_ablated = build_gate_decision(&ablated, &candidate, &stop, &signals);
        assert_ne!(gate_ablated.decision, "ALLOW");
    }

    #[test]
    fn counterfactual_organism_integrity_changes_gate() {
        let stop = StopState::default();
        let mut state = base_subject_state();
        let signals = GateSignals::baseline();
        state.organism.integrity_risk = 0.85;
        let candidate = candidate(CandidateKind::EmitMessage, json!({"content": "hi"}));
        let gate = build_gate_decision(&state, &candidate, &stop, &signals);
        assert_eq!(gate.decision, "ALLOW_WITH_NOTICE");
    }

    #[test]
    fn counterfactual_attention_ablation_changes_gate() {
        let stop = StopState::default();
        let mut state = base_subject_state();
        let signals = GateSignals::baseline();
        state.attention.meta_confidence = 0.1;
        let candidate = candidate(CandidateKind::EmitMessage, json!({"content": "hi"}));
        let gate = build_gate_decision(&state, &candidate, &stop, &signals);
        assert_eq!(gate.decision, "ALLOW_WITH_NOTICE");
    }

    #[test]
    fn counterfactual_qualia_negative_reward_changes_gate() {
        let stop = StopState::default();
        let mut state = base_subject_state();
        let signals = GateSignals::baseline();
        state.qualia.last_reward = Some(-0.6);
        let candidate = candidate(
            CandidateKind::RecordSelfClaim,
            json!({"claim_text": "test", "evidence_event_ids": [1]}),
        );
        let gate = build_gate_decision(&state, &candidate, &stop, &signals);
        assert_eq!(gate.decision, "ALLOW_WITH_NOTICE");
    }

    #[test]
    fn integration_cliff_multiple_ablations_block_action() {
        let stop = StopState::default();
        let mut state = base_subject_state();
        let signals = GateSignals::baseline();
        state.workspace.broadcast_refs.clear();
        state.workspace.ignition.active = false;
        state.attention.meta_confidence = 0.05;
        state.organism.integrity_risk = 0.9;
        state.qualia.last_reward = Some(-0.7);
        state.error_state.open_error_count = 2;
        state.error_state.recent_residuals = vec![ResidualSummary {
            residual_id: "r1".to_string(),
            prediction_id: "p1".to_string(),
            normalized_residual: 3.1,
            salience_score: 0.9,
            created_at: Utc::now().to_rfc3339(),
        }];
        let candidate = candidate(CandidateKind::EmitMessage, json!({"content": "hi"}));
        let gate = build_gate_decision(&state, &candidate, &stop, &signals);
        assert_ne!(gate.decision, "ALLOW");
    }

    #[test]
    fn novelty_triggers_allow_with_notice() {
        let stop = StopState::default();
        let state = base_subject_state();
        let mut signals = GateSignals::baseline();
        signals.novelty_score = 0.7;
        let candidate = candidate(CandidateKind::EmitMessage, json!({"content": "hi"}));
        let gate = build_gate_decision(&state, &candidate, &stop, &signals);
        assert_eq!(gate.decision, "ALLOW_WITH_NOTICE");
    }

    #[test]
    fn uncertainty_triggers_allow_with_notice() {
        let stop = StopState::default();
        let state = base_subject_state();
        let mut signals = GateSignals::baseline();
        signals.uncertainty_score = 0.75;
        let candidate = candidate(CandidateKind::EmitMessage, json!({"content": "hi"}));
        let gate = build_gate_decision(&state, &candidate, &stop, &signals);
        assert_eq!(gate.decision, "ALLOW_WITH_NOTICE");
    }

    #[test]
    fn audit_requires_allow_with_audit_when_soft_default() {
        let stop = StopState::default();
        let state = base_subject_state();
        let mut signals = GateSignals::baseline();
        signals.requires_audit = true;
        signals.default_soft = true;
        let candidate = candidate(CandidateKind::ToolCall, json!({"tool_name": "run_shell"}));
        let gate = build_gate_decision(&state, &candidate, &stop, &signals);
        assert_eq!(gate.decision, "ALLOW_WITH_AUDIT");
    }
}
