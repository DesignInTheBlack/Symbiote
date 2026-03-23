use super::*;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(super) struct WaveArbitrationContext {
    pub state: WaveStateVector,
    pub sources: Vec<String>,
    pub provenance_ok: bool,
}

#[derive(Debug, Clone)]
pub(super) struct QualiaModulationContext {
    pub tag: String,
    pub intensity: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResidualInfluenceMode {
    Shadow,
    Live,
    Degraded,
}

#[derive(Debug, Clone)]
pub(super) struct ResidualInfluenceContext {
    pub mode: ResidualInfluenceMode,
    pub bias: f32,
    pub residual_count: usize,
    pub gain: f32,
}

#[derive(Debug, Clone)]
struct WaveScoreEntry {
    base_score: f32,
    factor: f32,
    qualia_factor: f32,
    adjusted_score: f32,
    residual_bias: f32,
    reason: String,
    qualia_reason: String,
}

#[derive(Debug, Clone)]
struct WaveCalibration {
    quality_proxy: f32,
    outcome_samples: usize,
    task_success_rate: f32,
}

struct WeightProfile {
    user_satisfaction: f32,
    policy_rigor: f32,
    latency: f32,
    evidence_strictness: f32,
    exploration: f32,
}

fn weight_profile(settings: &crate::models::Settings) -> WeightProfile {
    WeightProfile {
        user_satisfaction: settings.weight_user_satisfaction.unwrap_or(0.5),
        policy_rigor: settings.weight_policy_rigor.unwrap_or(0.5),
        latency: settings.weight_latency.unwrap_or(0.5),
        evidence_strictness: settings.weight_evidence_strictness.unwrap_or(0.5),
        exploration: settings.weight_exploration.unwrap_or(0.5),
    }
}

fn candidate_weighted_score(candidate: &Candidate, settings: &crate::models::Settings) -> f32 {
    let weights = weight_profile(settings);
    let satisfaction_signal = match candidate.kind {
        CandidateKind::EmitMessage | CandidateKind::AskUserQuestion => 1.0,
        CandidateKind::ToolCall => 0.3,
        CandidateKind::FlagForHuman => 0.2,
        _ => 0.0,
    };
    let rigor_signal = if candidate_has_evidence(&candidate.payload) {
        1.0
    } else if candidate_is_question(candidate) {
        0.6
    } else {
        0.0
    };
    let evidence_signal = if candidate_has_evidence(&candidate.payload) {
        1.0
    } else {
        -1.0
    };
    let latency_signal = match candidate.kind {
        CandidateKind::ToolCall | CandidateKind::SpawnThread => -1.0,
        _ => 0.3,
    };
    let exploration_signal = match candidate.kind {
        CandidateKind::ToolCall
        | CandidateKind::SpawnThread
        | CandidateKind::PromoteSemantic
        | CandidateKind::UpdateWorkspace => 1.0,
        _ => 0.0,
    };

    (weights.user_satisfaction * satisfaction_signal)
        + (weights.policy_rigor * rigor_signal)
        + (weights.evidence_strictness * evidence_signal)
        + (weights.latency * latency_signal)
        + (weights.exploration * exploration_signal)
}

fn residual_bias_for_candidate(candidate: &Candidate, ctx: &ResidualInfluenceContext) -> f32 {
    let mut bias = ctx.bias.max(0.0);
    if matches!(ctx.mode, ResidualInfluenceMode::Degraded) {
        bias *= 0.5;
    }
    if bias <= 0.0 {
        return 0.0;
    }
    let multiplier = match candidate.kind {
        CandidateKind::AskUserQuestion => 0.45,
        CandidateKind::ToolCall => 0.35,
        CandidateKind::EmitMessage => -0.45,
        CandidateKind::FlagForHuman => -0.25,
        CandidateKind::SpawnThread => 0.2,
        _ => 0.0,
    };
    (bias * multiplier).clamp(-0.5, 0.5)
}

fn wave_modulation_factor(
    ctx: &WaveArbitrationContext,
    candidate: &Candidate,
    calibration: &WaveCalibration,
) -> (f32, String) {
    let mut non_organism_sources = std::collections::HashSet::new();
    for source in ctx.sources.iter() {
        if source != "organism" {
            non_organism_sources.insert(source.as_str());
        }
    }
    if non_organism_sources.len() < 2 {
        return (1.0, "wave_insufficient_sources".to_string());
    }
    if ctx.state.coherence < 0.55 || ctx.state.turbulence > 0.7 {
        return (1.0, "wave_unstable".to_string());
    }
    if calibration.outcome_samples < 3 {
        return (1.0, "wave_uncalibrated".to_string());
    }
    if calibration.quality_proxy < -0.3 || calibration.task_success_rate < 0.4 {
        return (1.0, "wave_quality_low".to_string());
    }

    let mut delta = (ctx.state.coherence - ctx.state.turbulence - ctx.state.fragmentation * 0.5)
        .clamp(-1.0, 1.0)
        * 0.2;
    let mut reason = "baseline".to_string();

    if ctx.state.turbulence > 0.6 {
        match candidate.kind {
            CandidateKind::AskUserQuestion => {
                delta += 0.05;
                reason = "turbulence_bias_question".to_string();
            }
            CandidateKind::ToolCall | CandidateKind::SpawnThread => {
                delta -= 0.05;
                reason = "turbulence_penalty_tools".to_string();
            }
            _ => {}
        }
    }

    if ctx.state.coherence > 0.6 && matches!(candidate.kind, CandidateKind::EmitMessage) {
        delta += 0.05;
        reason = "coherence_bias_emit".to_string();
    }

    delta = delta.clamp(-0.2, 0.2);
    (1.0 + delta, reason)
}

fn qualia_modulation_factor(
    ctx: &QualiaModulationContext,
    candidate: &Candidate,
) -> (f32, String) {
    let tag = ctx.tag.trim().to_lowercase();
    if tag.is_empty() || tag == "neutral" {
        return (1.0, "qualia_neutral".to_string());
    }
    let strength = (ctx.intensity * ctx.confidence).clamp(0.0, 1.0);
    if strength <= 0.01 {
        return (1.0, "qualia_low_confidence".to_string());
    }

    let mut delta = 0.0f32;
    let mut reason = "qualia_baseline".to_string();
    match tag.as_str() {
        "skeptical" => {
            match candidate.kind {
                CandidateKind::AskUserQuestion => {
                    delta += 0.12 * strength;
                    reason = "qualia_skeptical_bias_question".to_string();
                }
                CandidateKind::ToolCall => {
                    delta -= 0.08 * strength;
                    reason = "qualia_skeptical_penalty_tool".to_string();
                }
                CandidateKind::EmitMessage => {
                    delta -= 0.10 * strength;
                    reason = "qualia_skeptical_penalty_emit".to_string();
                }
                _ => {
                    delta -= 0.02 * strength;
                    reason = "qualia_skeptical_soft".to_string();
                }
            }
        }
        "curious" => {
            match candidate.kind {
                CandidateKind::AskUserQuestion => {
                    delta += 0.14 * strength;
                    reason = "qualia_curious_bias_question".to_string();
                }
                CandidateKind::ToolCall => {
                    delta += 0.06 * strength;
                    reason = "qualia_curious_bias_tool".to_string();
                }
                _ => {
                    delta += 0.02 * strength;
                    reason = "qualia_curious_soft".to_string();
                }
            }
        }
        "urgent" => {
            match candidate.kind {
                CandidateKind::ToolCall => {
                    delta += 0.10 * strength;
                    reason = "qualia_urgent_bias_tool".to_string();
                }
                CandidateKind::EmitMessage => {
                    delta += 0.08 * strength;
                    reason = "qualia_urgent_bias_emit".to_string();
                }
                CandidateKind::AskUserQuestion => {
                    delta -= 0.05 * strength;
                    reason = "qualia_urgent_penalty_question".to_string();
                }
                _ => {
                    delta += 0.02 * strength;
                    reason = "qualia_urgent_soft".to_string();
                }
            }
        }
        "calm" => {
            match candidate.kind {
                CandidateKind::ToolCall => {
                    delta -= 0.05 * strength;
                    reason = "qualia_calm_penalty_tool".to_string();
                }
                CandidateKind::AskUserQuestion => {
                    delta += 0.03 * strength;
                    reason = "qualia_calm_bias_question".to_string();
                }
                _ => {
                    delta += 0.01 * strength;
                    reason = "qualia_calm_soft".to_string();
                }
            }
        }
        "informative" => {
            match candidate.kind {
                CandidateKind::EmitMessage => {
                    delta += 0.06 * strength;
                    reason = "qualia_informative_bias_emit".to_string();
                }
                _ => {
                    delta -= 0.02 * strength;
                    reason = "qualia_informative_soft".to_string();
                }
            }
        }
        _ => {}
    }

    delta = delta.clamp(-0.2, 0.2);
    (1.0 + delta, reason)
}

fn derive_wave_calibration(state: &KernelState) -> WaveCalibration {
    let mut outcome_samples = 0usize;
    let mut successes = 0usize;
    let mut score = 0.0f32;
    let mut weight = 0.0f32;

    for outcome in state.recent_outcomes.iter().rev().take(12) {
        outcome_samples += 1;
        if outcome.success {
            successes += 1;
        }

        if let Some(kind) = outcome.action_type.strip_prefix("user_feedback_") {
            let delta = match kind {
                "agree" => 0.6,
                "follow_up" => 0.2,
                "clarify" => -0.2,
                "pushback" => -0.6,
                "disengage" => -0.8,
                _ => 0.0,
            };
            score += delta;
            weight += 1.0;
            continue;
        }

        let delta = if outcome.success { 0.2 } else { -0.2 };
        score += delta;
        weight += 0.4;
    }

    let quality_proxy = if weight > 0.0 {
        (score / weight).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let task_success_rate = if outcome_samples > 0 {
        successes as f32 / outcome_samples as f32
    } else {
        0.0
    };

    WaveCalibration {
        quality_proxy,
        outcome_samples,
        task_success_rate,
    }
}

fn required_capability_for_candidate(candidate: &Candidate, is_monologue_intent: bool) -> Option<&'static str> {
    match candidate.kind {
        CandidateKind::EmitMessage | CandidateKind::AskUserQuestion | CandidateKind::FlagForHuman => {
            if is_monologue_intent {
                Some("monologue_emit")
            } else {
                Some("emit")
            }
        }
        CandidateKind::ToolCall => Some("tools"),
        CandidateKind::RecordSelfClaim => Some("self_claims"),
        CandidateKind::UpdateInnerSummary
        | CandidateKind::WriteEpisodic
        | CandidateKind::PromoteSemantic
        | CandidateKind::UpdateGoalThread
        | CandidateKind::UpdateWorkspace
        | CandidateKind::AnchorShift => Some("memory_write"),
        CandidateKind::SpawnThread => Some("background_jobs"),
        _ => None,
    }
}

fn capability_allowed(allowed: &AllowedCapabilities, capability: &str) -> bool {
    match capability {
        "emit" => allowed.emit,
        "tools" => allowed.tools,
        "memory_write" => allowed.memory_write,
        "self_claims" => allowed.self_claims,
        "monologue_run" => allowed.monologue_run,
        "monologue_emit" => allowed.monologue_emit,
        "background_jobs" => allowed.background_jobs,
        _ => true,
    }
}

fn candidate_question_text(candidate: &Candidate) -> Option<String> {
    candidate
        .payload
        .get("payload")
        .and_then(|v| v.as_str())
        .or_else(|| candidate.payload.get("question").and_then(|v| v.as_str()))
        .or_else(|| candidate.payload.get("content").and_then(|v| v.as_str()))
        .map(|s| s.to_string())
}

fn candidate_is_question(candidate: &Candidate) -> bool {
    if matches!(candidate.kind, CandidateKind::AskUserQuestion) {
        return true;
    }
    candidate_question_text(candidate)
        .map(|text| text.trim().ends_with('?'))
        .unwrap_or(false)
}

fn candidate_is_bounded(candidate: &Candidate) -> bool {
    let bounded_flag = candidate
        .payload
        .get("bounded")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let speculative = candidate
        .payload
        .get("speculative")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let assumptions = candidate
        .payload
        .get("assumptions")
        .and_then(|v| v.as_str())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    if bounded_flag || speculative || assumptions {
        return true;
    }
    candidate_alignment_text(candidate)
        .map(|text| text.to_lowercase().contains("assumption"))
        .unwrap_or(false)
}

fn mark_candidate_bounded(candidate: &mut Candidate) {
    if let Some(obj) = candidate.payload.as_object_mut() {
        let content = obj
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if !content.to_lowercase().contains("assumption") {
            let updated = if content.is_empty() {
                "Assumptions: Based on current context only.".to_string()
            } else {
                format!("Assumptions: Based on current context only.\n\n{}", content)
            };
            obj.insert("content".to_string(), Value::String(updated));
        }
        obj.insert("bounded".to_string(), Value::Bool(true));
    }
}

#[derive(Debug, Clone)]
enum GateOutcome {
    Allow,
    BlockWithFallbacks(String),
}

fn policy_block_reason(candidate: &Candidate) -> Option<String> {
    if let Some(reason) = candidate.payload.get("policy_block").and_then(|v| v.as_str()) {
        if !reason.trim().is_empty() {
            return Some(format!("POLICY_BLOCK_{}", reason.trim()));
        }
    }
    if let Some(reason) = candidate.payload.get("contract_block").and_then(|v| v.as_str()) {
        if !reason.trim().is_empty() {
            return Some(format!("CONTRACT_BLOCK_{}", reason.trim()));
        }
    }
    None
}

fn gate_policy(candidate: &Candidate) -> GateOutcome {
    match policy_block_reason(candidate) {
        Some(reason) => GateOutcome::BlockWithFallbacks(reason),
        None => GateOutcome::Allow,
    }
}

fn gate_stop_state(
    candidate: &Candidate,
    allowed_capabilities: &AllowedCapabilities,
    is_monologue_intent: bool,
) -> GateOutcome {
    if let Some(capability) = required_capability_for_candidate(candidate, is_monologue_intent) {
        if !capability_allowed(allowed_capabilities, capability) {
            return GateOutcome::BlockWithFallbacks(format!("STOP_STATE_{}", capability.to_uppercase()));
        }
    }
    GateOutcome::Allow
}

fn gate_action_gate(
    candidate: &Candidate,
    state: &KernelState,
    internal_cycle: bool,
) -> GateOutcome {
    match action_gate_reason_for(candidate, state, internal_cycle) {
        Some(reason) => GateOutcome::BlockWithFallbacks(reason),
        None => GateOutcome::Allow,
    }
}

fn gate_matrix_reason(
    candidate: &Candidate,
    state: &KernelState,
    allowed_capabilities: &AllowedCapabilities,
    internal_cycle: bool,
    is_monologue_intent: bool,
) -> Option<String> {
    if let GateOutcome::BlockWithFallbacks(reason) = gate_policy(candidate) {
        return Some(reason);
    }
    if let GateOutcome::BlockWithFallbacks(reason) =
        gate_stop_state(candidate, allowed_capabilities, is_monologue_intent)
    {
        return Some(reason);
    }
    if let GateOutcome::BlockWithFallbacks(reason) = gate_action_gate(candidate, state, internal_cycle) {
        return Some(reason);
    }
    None
}

fn tool_penalty_active(state: &KernelState, penalty_key: &str) -> bool {
    let entry = match state.tool_failure_penalties.get(penalty_key) {
        Some(entry) => entry,
        None => return false,
    };
    let until = entry.penalty_until.as_deref().unwrap_or("");
    if until.is_empty() {
        return false;
    }
    chrono::DateTime::parse_from_rfc3339(until)
        .ok()
        .map(|ts| chrono::Utc::now() < ts.with_timezone(&Utc))
        .unwrap_or(false)
}

fn is_cheap_tool_name(tool_name: &str) -> bool {
    matches!(
        tool_name.trim().to_lowercase().as_str(),
        "get_current_time"
            | "get_system_logs"
            | "get_system_capabilities"
            | "get_inner_summary"
            | "get_workspace_state"
            | "get_rolling_summary"
            | "read_context"
            | "save_context"
    )
}

fn unsafe_to_assume(state: &KernelState, candidate: &Candidate) -> bool {
    if state
        .missing_input_policy
        .as_deref()
        .map(|p| p.eq_ignore_ascii_case("strict"))
        .unwrap_or(false)
    {
        return true;
    }
    if let Some(required) = candidate
        .payload
        .get("requires_resolved_slots")
        .and_then(|v| v.as_array())
    {
        let required_slots = required
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| normalize_slot_key(s))
            .collect::<Vec<_>>();
        if !required_slots.is_empty() {
            let missing: std::collections::HashSet<String> = state
                .missing_slots
                .iter()
                .map(|s| normalize_slot_key(s))
                .collect();
            if required_slots.iter().any(|slot| missing.contains(slot)) {
                return true;
            }
        }
    }
    false
}

fn stop_reason_from_rejection(reason: &str) -> StopReason {
    let mut category = StopReasonCategory::UnknownBlock;
    let mut contract: Option<String> = None;
    if reason.starts_with("POLICY_") || reason.starts_with("CONTRACT_") {
        category = StopReasonCategory::PolicyBlock;
        if reason.contains("C1") {
            contract = Some("C1".to_string());
        } else if reason.contains("C3") {
            contract = Some("C3".to_string());
        } else if reason.contains("C4") {
            contract = Some("C4".to_string());
        } else if reason.contains("C5") {
            contract = Some("C5".to_string());
        }
    } else if reason.starts_with("CAP_")
        || reason.starts_with("BUDGET_")
        || reason.starts_with("THREAD_")
        || reason.starts_with("PLAY_")
        || reason.starts_with("INTERNAL_")
    {
        category = StopReasonCategory::BudgetBlock;
    } else if reason.starts_with("TASK_PHASE") {
        category = StopReasonCategory::PhaseBlock;
    } else if reason.starts_with("SELF_CLAIM")
        || reason.starts_with("MISSING_SLOTS")
        || reason.starts_with("REFUSED_SLOTS")
    {
        category = StopReasonCategory::EvidenceBlock;
    } else if reason.starts_with("TOOL_") || reason.starts_with("RESEARCH_") {
        category = StopReasonCategory::ToolBlock;
    } else if reason.starts_with("ASK_") || reason.starts_with("EMIT_") || reason.starts_with("STOP_") {
        category = StopReasonCategory::LatchBlock;
    }
    StopReason {
        category,
        subcode: reason.to_string(),
        contract,
    }
}

fn build_unblock_instructions(stop_reasons: &[StopReason]) -> String {
    if stop_reasons.is_empty() {
        return "Provide more context or rephrase the request.".to_string();
    }
    let mut notes = Vec::new();
    for reason in stop_reasons {
        let note = match reason.category {
            StopReasonCategory::PolicyBlock => {
                if let Some(contract) = reason.contract.as_deref() {
                    format!("Address contract {} constraints or request a permitted action.", contract)
                } else {
                    "Adjust the request to align with policy constraints.".to_string()
                }
            }
            StopReasonCategory::BudgetBlock => "Reduce scope, wait for budget reset, or increase limits.".to_string(),
            StopReasonCategory::PhaseBlock => "Complete the current phase or reset the task.".to_string(),
            StopReasonCategory::LatchBlock => "Clear stop/quiet latches or allow emission.".to_string(),
            StopReasonCategory::EvidenceBlock => "Provide evidence or missing details to proceed safely.".to_string(),
            StopReasonCategory::ToolBlock => "Allow the tool or wait for tool penalty cooldown.".to_string(),
            StopReasonCategory::TimeoutBlock => "Retry after timeout or provide a smaller request.".to_string(),
            StopReasonCategory::UnknownBlock => "Provide more context or rephrase the request.".to_string(),
        };
        if !notes.contains(&note) {
            notes.push(note);
        }
    }
    notes.join(" ")
}

fn build_explain_blockers_message(stop_reasons: &[StopReason], unblock: &str) -> String {
    let mut parts = Vec::new();
    if !stop_reasons.is_empty() {
        let reasons = stop_reasons
            .iter()
            .map(|r| match r.contract.as_deref() {
                Some(contract) => format!("{}({})", r.subcode, contract),
                None => r.subcode.clone(),
            })
            .collect::<Vec<_>>()
            .join(", ");
        parts.push(format!("Blockers: {}", reasons));
    }
    if !unblock.trim().is_empty() {
        parts.push(format!("Unblock: {}", unblock.trim()));
    }
    if parts.is_empty() {
        "Blocked without a specific reason. Provide more context.".to_string()
    } else {
        parts.join(" ")
    }
}

pub(crate) fn apply_plan_verification_report(
    report: &mut DecisionReport,
    verification: &subject_controller::PlanVerificationResult,
) {
    report.verification_outcome = Some(verification.outcome.clone());
    report.verification_reasons = Some(verification.reasons.clone());
    report.verification_confidence = Some(verification.confidence);
    report.verification_assumptions_checked = Some(verification.assumptions_checked);
    report.verification_assumptions_failed = Some(verification.assumptions_failed);
    report.verification_conflict_topics = Some(verification.conflict_topics.clone());
}

pub(crate) fn plan_state_for_verification(
    verification: &subject_controller::PlanVerificationResult,
) -> &'static str {
    if !verification.conflict_topics.is_empty() {
        return "revised";
    }
    match verification.outcome.as_str() {
        "ALLOW" => "verified",
        "DEFER" => "revised",
        "VERIFY" => "draft",
        _ => "draft",
    }
}

pub(crate) fn apply_plan_verification_to_gate_decision(
    gate: &mut subject_controller::GateDecisionRecord,
    verification: &subject_controller::PlanVerificationResult,
) {
    subject_controller::apply_verification_to_gate(gate, verification);
}

impl Kernel {
    pub(super) fn arbitrate(
        &self,
        candidates: &[Candidate],
        settings: &crate::models::Settings,
        state: &KernelState,
        internal_cycle: bool,
        anchor_hits: Option<usize>,
        wave_context: Option<WaveArbitrationContext>,
        qualia_context: Option<QualiaModulationContext>,
        residual_context: Option<ResidualInfluenceContext>,
        run_id: Option<&str>,
    ) -> KernelDecision {
        let mut list = candidates.to_vec();
        let wave_context = wave_context.filter(|ctx| ctx.provenance_ok);
        let residual_context = residual_context
            .filter(|ctx| ctx.bias.abs() > 0.0 && ctx.residual_count > 0);
        let calibration = derive_wave_calibration(state);
        let mut score_cache: HashMap<String, WaveScoreEntry> = HashMap::new();
        let residual_live = residual_context
            .as_ref()
            .map(|ctx| matches!(ctx.mode, ResidualInfluenceMode::Live | ResidualInfluenceMode::Degraded))
            .unwrap_or(false);
        for candidate in list.iter() {
            let base_score = candidate_weighted_score(candidate, settings);
            let (factor, reason) = if let Some(ctx) = wave_context.as_ref() {
                wave_modulation_factor(ctx, candidate, &calibration)
            } else {
                (1.0, "wave_disabled".to_string())
            };
            let (qualia_factor, qualia_reason) = if let Some(ctx) = qualia_context.as_ref() {
                qualia_modulation_factor(ctx, candidate)
            } else {
                (1.0, "qualia_disabled".to_string())
            };
            let residual_bias = if residual_live {
                residual_context
                    .as_ref()
                    .map(|ctx| residual_bias_for_candidate(candidate, ctx))
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            let adjusted_score = base_score * factor * qualia_factor + residual_bias;
            score_cache.insert(
                candidate.id.clone(),
                WaveScoreEntry {
                    base_score,
                    factor,
                    qualia_factor,
                    adjusted_score,
                    residual_bias,
                    reason,
                    qualia_reason,
                },
            );
        }
        let sort_with = |entries: &mut Vec<Candidate>, score_fn: &dyn Fn(&Candidate) -> f32| {
            entries.sort_by(|a, b| {
                let a_meta = if is_meta_cog_candidate(a) && !matches!(a.kind, CandidateKind::NoOp) { 0 } else { 1 };
                let b_meta = if is_meta_cog_candidate(b) && !matches!(b.kind, CandidateKind::NoOp) { 0 } else { 1 };
                let a_mono = if is_monologue_intent_candidate(a) { 0 } else { 1 };
                let b_mono = if is_monologue_intent_candidate(b) { 0 } else { 1 };
                a_meta
                    .cmp(&b_meta)
                    .then_with(|| a_mono.cmp(&b_mono))
                    .then_with(|| a.priority_class.cmp(&b.priority_class))
                    .then_with(|| {
                        let a_score = score_fn(a);
                        let b_score = score_fn(b);
                        b_score.partial_cmp(&a_score).unwrap_or(Ordering::Equal)
                    })
                    .then_with(|| a.priority_rank.cmp(&b.priority_rank))
                    .then_with(|| b.urgency.unwrap_or(0).cmp(&a.urgency.unwrap_or(0)))
                    .then_with(|| a.cost.unwrap_or(0).cmp(&b.cost.unwrap_or(0)))
                    .then_with(|| a.created_at.cmp(&b.created_at))
            });
        };
        let score_for_actual = |candidate: &Candidate| {
            score_cache
                .get(&candidate.id)
                .map(|entry| entry.adjusted_score)
                .unwrap_or_else(|| candidate_weighted_score(candidate, settings))
        };
        sort_with(&mut list, &score_for_actual);

        let base_winner_id = list.first().map(|c| c.id.clone());
        let mut shadow_winner_id: Option<String> = None;
        let mut residual_shadow_would_change: Option<bool> = None;
        let mut residual_shadow_impact_pct: Option<f64> = None;
        if let Some(ctx) = residual_context.as_ref() {
            if matches!(ctx.mode, ResidualInfluenceMode::Shadow) {
                let mut shadow_list = list.clone();
                let score_for_shadow = |candidate: &Candidate| {
                    let base = score_cache
                        .get(&candidate.id)
                        .map(|entry| entry.base_score * entry.factor * entry.qualia_factor)
                        .unwrap_or_else(|| candidate_weighted_score(candidate, settings));
                    base + residual_bias_for_candidate(candidate, ctx)
                };
                sort_with(&mut shadow_list, &score_for_shadow);
                shadow_winner_id = shadow_list.first().map(|c| c.id.clone());
                let would_change = shadow_winner_id != base_winner_id;
                residual_shadow_would_change = Some(would_change);
                residual_shadow_impact_pct = Some(if would_change { 100.0 } else { 0.0 });
            }
        }

        if let (Some(ctx), Some(run_id)) = (wave_context.as_ref(), run_id) {
            if let Some(top) = list.first() {
                if let Some(entry) = score_cache.get(&top.id) {
                    let db = self.db.clone();
                    let app = self.app_handle.clone();
                    let run_id = Some(run_id.to_string());
                    let payload = json!({
                        "event": "wave_arbitration_modulation",
                        "candidate_id": top.id,
                        "candidate_kind": format!("{:?}", top.kind),
                        "base_score": entry.base_score,
                        "factor": entry.factor,
                        "qualia_factor": entry.qualia_factor,
                        "adjusted_score": entry.adjusted_score,
                        "residual_bias": entry.residual_bias,
                        "reason": entry.reason,
                        "qualia_reason": entry.qualia_reason,
                        "coherence": ctx.state.coherence,
                        "turbulence": ctx.state.turbulence,
                        "drift": ctx.state.drift,
                        "dominance": ctx.state.dominance,
                        "fragmentation": ctx.state.fragmentation,
                        "quality_proxy": calibration.quality_proxy,
                        "outcome_samples": calibration.outcome_samples,
                        "task_success_rate": calibration.task_success_rate,
                        "sources": ctx.sources,
                    });
                    tokio::spawn(async move {
                        let _ = system_log::log_event(
                            &db.pool,
                            Some(&app),
                            "info",
                            "cognitive_wave",
                            run_id.as_deref(),
                            None,
                            payload,
                        )
                        .await;
                    });
                }
            }
        }
        if let (Some(ctx), Some(run_id)) = (qualia_context.as_ref(), run_id) {
            if let Some(top) = list.first() {
                if let Some(entry) = score_cache.get(&top.id) {
                    let db = self.db.clone();
                    let app = self.app_handle.clone();
                    let run_id = Some(run_id.to_string());
                    let payload = json!({
                        "event": "qualia_modulation_applied",
                        "path": "arbitration",
                        "candidate_id": top.id,
                        "candidate_kind": format!("{:?}", top.kind),
                        "base_score": entry.base_score,
                        "wave_factor": entry.factor,
                        "qualia_factor": entry.qualia_factor,
                        "adjusted_score": entry.adjusted_score,
                        "residual_bias": entry.residual_bias,
                        "qualia_reason": entry.qualia_reason,
                        "qualia_tag": ctx.tag.clone(),
                        "qualia_intensity": ctx.intensity,
                        "qualia_confidence": ctx.confidence,
                    });
                    tokio::spawn(async move {
                        let _ = system_log::log_event(
                            &db.pool,
                            Some(&app),
                            "info",
                            "kernel",
                            run_id.as_deref(),
                            None,
                            payload,
                        )
                        .await;
                    });
                }
            }
        }
        if let (Some(ctx), Some(run_id)) = (residual_context.as_ref(), run_id) {
            if matches!(ctx.mode, ResidualInfluenceMode::Shadow) {
                let db = self.db.clone();
                let app = self.app_handle.clone();
                let run_id = Some(run_id.to_string());
                let payload = json!({
                    "event": "residual_shadow_impact",
                    "impact_pct": residual_shadow_impact_pct.unwrap_or(0.0),
                    "would_change_winner": residual_shadow_would_change.unwrap_or(false),
                    "base_winner_id": base_winner_id,
                    "shadow_winner_id": shadow_winner_id,
                    "bias": ctx.bias,
                    "gain": ctx.gain,
                    "residual_count": ctx.residual_count,
                });
                tokio::spawn(async move {
                    let _ = system_log::log_event(
                        &db.pool,
                        Some(&app),
                        "info",
                        "kernel",
                        run_id.as_deref(),
                        None,
                        payload,
                    )
                    .await;
                });
            }
        }

        let mut accepted = Vec::new();
        let mut rejected = Vec::new();
        let mut caps_applied: BTreeSet<String> = BTreeSet::new();
        let mut tool_calls = 0;
        let mut thread_spawns = 0;
        let mut emissions = 0;
        let mut flagged = 0;
        let mut summary_updates = 0;
        let mut creative = 0;
        let mut effective_stop_state = state.stop_state.clone();

        let has_internal_evidence = list.iter().any(|c| {
            candidate_has_evidence(&c.payload) || matches!(candidate_evidence_class(c), Some("internal"))
        });
        if anchor_hits == Some(0) && !has_internal_evidence {
            let mut scope = StopScope::default();
            scope.memory_write = true;
            scope.self_claims = true;
            let reason = StopReason {
                category: StopReasonCategory::EvidenceBlock,
                subcode: "anchor_miss_no_evidence".to_string(),
                contract: Some("C3".to_string()),
            };
            effective_stop_state.apply_reason(reason, scope);
        }

        let allow_internal_emit = self.allow_internal_emit(state, settings);
        if internal_cycle && !allow_internal_emit {
            let mut scope = StopScope::default();
            scope.emit = true;
            effective_stop_state.apply_reason(
                StopReason {
                    category: StopReasonCategory::LatchBlock,
                    subcode: "internal_emit_disabled".to_string(),
                    contract: None,
                },
                scope,
            );
        }

        let allowed_capabilities = effective_stop_state.allowed_capabilities();
        let stop_reasons = effective_stop_state.reasons.clone();
        let has_ask_candidate = list
            .iter()
            .any(|c| matches!(c.kind, CandidateKind::AskUserQuestion));
        let has_spawn_candidate = list
            .iter()
            .any(|c| matches!(c.kind, CandidateKind::SpawnThread));
        let controller_gate = state.controller_gate.as_ref();
        let throttle_tools = controller_gate.map(|g| g.throttle_tools).unwrap_or(false);
        let throttle_threads = controller_gate.map(|g| g.throttle_threads).unwrap_or(false);
        let throttle_asks = controller_gate.map(|g| g.throttle_asks).unwrap_or(false);
        let prefer_verification = controller_gate.map(|g| g.prefer_verification).unwrap_or(false);
        let prefer_ask_user = state.uncertainty_count >= 1
            || state.failure_count >= 2
            || state.stance.stance == "clarify"
            || prefer_verification;
        let prefer_decompose = state.stalled_count >= 2 && has_spawn_candidate;
        let avoid_tools = state.stance.tool_preference == "avoid";
        let last_user_input = state.last_user_input.as_deref().unwrap_or("").to_lowercase();
        let allow_multi_tool = list.iter().any(is_research_tool) && has_research_budget(state, settings);
        let max_tool_calls = if allow_multi_tool { 2 } else { 1 };

        for mut candidate in list {
            let mut reason = None;
            let is_monologue_intent = is_monologue_intent_candidate(&candidate);
            let is_monologue_src = is_monologue_source(&candidate.source);
            let tool_name = if matches!(candidate.kind, CandidateKind::ToolCall) {
                candidate
                    .payload
                    .get("tool_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            } else {
                String::new()
            };
            let is_context_only_tool = !tool_name.is_empty()
                && crate::core::tool_registry::ToolRegistry::is_context_only_tool(&tool_name);
            let allow_context_toolcap = is_monologue_intent && is_context_only_tool;
            let user_requested_tool = if matches!(candidate.kind, CandidateKind::ToolCall) {
                user_requested_tool_in_input(&last_user_input, &tool_name)
            } else {
                false
            };
            if reason.is_none() {
                if let Some(block_reason) = gate_matrix_reason(
                    &candidate,
                    state,
                    &allowed_capabilities,
                    internal_cycle,
                    is_monologue_intent,
                ) {
                    reason = Some(block_reason);
                }
            }
            if reason.is_none() {
                match candidate.kind {
                    CandidateKind::ToolCall => {
                        if tool_calls >= max_tool_calls && !allow_context_toolcap {
                            reason = Some("CAP_REACHED_TOOLCALL".to_string());
                        } else if is_research_tool(&candidate) {
                            let mut has_uncertainty = candidate
                                .payload
                                .get("uncertainty")
                                .and_then(|v| v.as_str())
                                .map(|s| !s.trim().is_empty())
                                .unwrap_or(false);
                            let mut has_impact = candidate
                                .payload
                                .get("decision_impact")
                                .and_then(|v| v.as_str())
                                .map(|s| !s.trim().is_empty())
                                .unwrap_or(false);
                            if !has_uncertainty || !has_impact {
                                let cheap_attempt = is_cheap_tool_name(&tool_name);
                                let auto_justify = user_requested_tool
                                    || prefer_verification
                                    || state.uncertainty_count >= 1
                                    || (cheap_attempt && candidate.urgency.unwrap_or(0) >= 7);
                                if auto_justify {
                                    let reason_label = if user_requested_tool {
                                        "user_request"
                                    } else if cheap_attempt {
                                        "cheap_tool"
                                    } else if prefer_verification || state.uncertainty_count >= 1 {
                                        "uncertainty"
                                    } else {
                                        "auto"
                                    };
                                    if !candidate.payload.is_object() {
                                        candidate.payload = json!({});
                                    }
                                    if let Some(obj) = candidate.payload.as_object_mut() {
                                        if !has_uncertainty {
                                            obj.insert(
                                                "uncertainty".to_string(),
                                                Value::String(format!(
                                                    "External lookup requested/needed ({})",
                                                    reason_label
                                                )),
                                            );
                                            has_uncertainty = true;
                                        }
                                        if !has_impact {
                                            obj.insert(
                                                "decision_impact".to_string(),
                                                Value::String(
                                                    "Lookup required to answer with current evidence.".to_string(),
                                                ),
                                            );
                                            has_impact = true;
                                        }
                                        obj.insert("auto_justified".to_string(), Value::Bool(true));
                                        obj.insert(
                                            "auto_justification_reason".to_string(),
                                            Value::String(reason_label.to_string()),
                                        );
                                        obj.insert(
                                            "auto_justification_tool".to_string(),
                                            Value::String(tool_name.clone()),
                                        );
                                    }
                                } else {
                                    reason = Some("RESEARCH_MISSING_JUSTIFICATION".to_string());
                                }
                            }
                            if reason.is_none() && !has_research_budget(state, settings) {
                                reason = Some("BUDGET_EXCEEDED_RESEARCH".to_string());
                            }
                        }
                    }
                    CandidateKind::SpawnThread => {
                        if thread_spawns >= 1 {
                            reason = Some("CAP_REACHED_THREAD".to_string());
                        } else {
                            let max_depth = settings.thread_max_depth.unwrap_or(2).max(0);
                            if state.thread_depth >= max_depth {
                                reason = Some("THREAD_DEPTH_MAX".to_string());
                            }
                        }
                    }
                    CandidateKind::EmitMessage => {
                        if emissions >= 1 && !is_monologue_intent {
                            reason = Some("CAP_REACHED_EMIT".to_string());
                        }
                    }
                    CandidateKind::FlagForHuman => {
                        if flagged >= 1 && !is_monologue_intent {
                            reason = Some("CAP_REACHED_FLAG_FOR_HUMAN".to_string());
                        }
                    }
                    CandidateKind::AskUserQuestion => {
                        if emissions >= 1 && !is_monologue_intent {
                            reason = Some("CAP_REACHED_EMIT".to_string());
                        }
                    }
                    CandidateKind::UpdateInnerSummary => {
                        if summary_updates >= 1 {
                            reason = Some("CAP_REACHED_SUMMARY".to_string());
                        }
                    }
                    _ => {}
                }
            }

            if reason.is_none() && matches!(candidate.kind, CandidateKind::ToolCall) {
                let penalty_key =
                    super::tool_penalty_key(&tool_name, super::tool_target_hint_from_payload(&candidate.payload).as_deref());
                if tool_penalty_active(state, &penalty_key) {
                    reason = Some("TOOL_PENALTY_ACTIVE".to_string());
                }
            }

            if reason.is_none() {
                if !is_monologue_src && matches!(candidate.kind, CandidateKind::ToolCall) {
                    let auto_justified = candidate
                        .payload
                        .get("auto_justified")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let allow_for_verification = prefer_verification || auto_justified;
                    let allow_for_local = is_local_tool_name(&tool_name);
                    let throttle_bypass = user_requested_tool || allow_for_verification || allow_for_local;

                    if avoid_tools && !user_requested_tool && !allow_for_local && !allow_for_verification {
                        reason = Some("POLICY_TOOL_AVOID".to_string());
                    } else if throttle_tools && candidate.urgency.unwrap_or(0) < 7 && !throttle_bypass {
                        reason = Some("CONTROLLER_THROTTLE_TOOLS".to_string());
                    }
                } else if !is_monologue_src && throttle_threads && matches!(candidate.kind, CandidateKind::SpawnThread) {
                    if candidate.urgency.unwrap_or(0) < 7 {
                        reason = Some("CONTROLLER_THROTTLE_THREADS".to_string());
                    }
                } else if !is_monologue_intent && throttle_asks && matches!(candidate.kind, CandidateKind::AskUserQuestion) {
                    if candidate.urgency.unwrap_or(0) < 7 {
                        reason = Some("CONTROLLER_THROTTLE_ASKS".to_string());
                    }
                }
            }

            if reason.is_none()
                && prefer_ask_user
                && has_ask_candidate
                && matches!(candidate.kind, CandidateKind::EmitMessage)
                && !is_monologue_intent
            {
                reason = Some("POLICY_PREFER_ASK_USER".to_string());
            }

            if reason.is_none()
                && prefer_decompose
                && !is_monologue_intent
                && matches!(candidate.kind, CandidateKind::EmitMessage)
            {
                reason = Some("POLICY_PREFER_DECOMPOSE".to_string());
            }

            if reason.is_none() && state.mode == KernelMode::Play {
                if matches!(candidate.kind, CandidateKind::ToolCall | CandidateKind::SpawnThread) {
                    if tool_calls + thread_spawns >= 1 {
                        reason = Some("PLAY_ACTION_LIMIT".to_string());
                    }
                }
                if matches!(candidate.kind, CandidateKind::UpdateGoalThread) {
                    if creative >= 1 {
                        reason = Some("PLAY_CREATIVE_LIMIT".to_string());
                    }
                }
            }

            if let Some(reason) = reason {
                if reason.starts_with("CAP_")
                    || reason.starts_with("PLAY_")
                    || reason.starts_with("INTERNAL_")
                    || reason.starts_with("BUDGET_")
                    || reason.starts_with("THREAD_")
                {
                    caps_applied.insert(reason.clone());
                }
                let kind = candidate.kind.clone();
                rejected.push(RejectedCandidate {
                    id: candidate.id,
                    kind: kind.clone(),
                    reason,
                    tool_name: candidate
                        .payload
                        .get("tool_name")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                    source: Some(candidate.source.clone()),
                    is_monologue: Some(is_monologue_src),
                    payload: if matches!(kind, CandidateKind::ToolCall) {
                        Some(candidate.payload.clone())
                    } else {
                        None
                    },
                });
                continue;
            }

            match candidate.kind {
                CandidateKind::ToolCall => {
                    if !allow_context_toolcap {
                        tool_calls += 1;
                    }
                }
                CandidateKind::SpawnThread => thread_spawns += 1,
                CandidateKind::EmitMessage | CandidateKind::AskUserQuestion => {
                    emissions += 1
                }
                CandidateKind::FlagForHuman => {
                    emissions += 1;
                    flagged += 1;
                }
                CandidateKind::UpdateInnerSummary => summary_updates += 1,
                CandidateKind::UpdateGoalThread => creative += 1,
                _ => {}
            }

            accepted.push(candidate);
        }

        let blocked_candidates_count = rejected.len();
        let mut reason_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for rejected_candidate in rejected.iter() {
            *reason_counts.entry(rejected_candidate.reason.clone()).or_insert(0) += 1;
        }
        let mut top_block_reasons = reason_counts
            .iter()
            .map(|(reason, count)| (reason.clone(), *count))
            .collect::<Vec<_>>();
        top_block_reasons.sort_by(|a, b| b.1.cmp(&a.1));
        let top_block_reasons: Vec<String> = top_block_reasons
            .into_iter()
            .take(3)
            .map(|(reason, _)| reason)
            .collect();

        let tools_blocked = !allowed_capabilities.tools;
        let user_requested_tool = accepted.iter().any(|c| {
            if !matches!(c.kind, CandidateKind::ToolCall) {
                return false;
            }
            let name = c
                .payload
                .get("tool_name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            user_requested_tool_in_input(&last_user_input, name)
        });

        let unsafe_assume = accepted.iter().any(|c| unsafe_to_assume(state, c));

        let direct_idx = accepted.iter().position(|c| {
            matches!(c.kind, CandidateKind::EmitMessage | CandidateKind::FlagForHuman) && !candidate_is_question(c)
        });
        let bounded_idx = accepted.iter().position(|c| {
            matches!(c.kind, CandidateKind::EmitMessage | CandidateKind::FlagForHuman) && candidate_is_bounded(c)
        });
        let question_idx = accepted.iter().position(|c| candidate_is_question(c));
        let tool_idx = accepted.iter().position(|c| matches!(c.kind, CandidateKind::ToolCall));

        let mut selected_action = SelectedAction::DirectAnswer;
        let mut selected_source: Option<String> = None;
        let mut fallback_used = false;
        let mut fallback_type: Option<String> = None;
        let mut selected_index: Option<usize> = None;

        if let Some(idx) = direct_idx {
            selected_action = SelectedAction::DirectAnswer;
            selected_source = Some(accepted[idx].source.clone());
            selected_index = Some(idx);
        } else {
            fallback_used = true;
            let skip_bounded = unsafe_assume || (user_requested_tool && tools_blocked);
            if !skip_bounded {
                if let Some(idx) = bounded_idx.or(direct_idx) {
                    selected_action = SelectedAction::BoundedAnswer;
                    selected_source = Some(accepted[idx].source.clone());
                    selected_index = Some(idx);
                }
            }
            if selected_index.is_none() {
                if let Some(idx) = question_idx {
                    selected_action = SelectedAction::ClarifyingQuestion;
                    selected_source = Some(accepted[idx].source.clone());
                    selected_index = Some(idx);
                }
            }
            if selected_index.is_none() {
                if let Some(idx) = tool_idx {
                    selected_action = SelectedAction::ToolAttempt;
                    selected_source = Some(accepted[idx].source.clone());
                    selected_index = Some(idx);
                }
            }
            if selected_index.is_none() {
                selected_action = SelectedAction::ExplainBlockers;
            }
            fallback_type = Some(selected_action.as_str().to_string());
        }

        if selected_action == SelectedAction::BoundedAnswer {
            if let Some(idx) = selected_index {
                mark_candidate_bounded(&mut accepted[idx]);
            }
        }

        let mut stop_reason_set: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut stop_reasons_report = Vec::new();
        for reason in stop_reasons.iter() {
            let key = format!("{:?}:{}", reason.category, reason.subcode);
            if stop_reason_set.insert(key) {
                stop_reasons_report.push(reason.clone());
            }
        }
        for reason in rejected.iter().map(|r| r.reason.clone()) {
            let mapped = stop_reason_from_rejection(&reason);
            let key = format!("{:?}:{}", mapped.category, mapped.subcode);
            if stop_reason_set.insert(key) {
                stop_reasons_report.push(mapped);
            }
        }

        let mut cannot_respond = false;
        let fallback_allowed = allowed_capabilities.emit || !accepted.is_empty();
        if !allowed_capabilities.emit && !fallback_allowed {
            cannot_respond = true;
            if stop_reasons_report.is_empty() {
                stop_reasons_report.push(StopReason {
                    category: StopReasonCategory::UnknownBlock,
                    subcode: "emit_blocked".to_string(),
                    contract: None,
                });
            }
        }

        let unblock_instructions = build_unblock_instructions(&stop_reasons_report);

        if selected_action == SelectedAction::ExplainBlockers && allowed_capabilities.emit {
            let mut created_at = Utc::now().timestamp();
            let content = build_explain_blockers_message(&stop_reasons_report, &unblock_instructions);
            let candidate = self.make_candidate(
                CandidateKind::EmitMessage,
                json!({"content": content, "blocked": true}),
                "system_explain_blockers",
                &mut created_at,
            );
            accepted.push(candidate);
        }

        if !allowed_capabilities.emit && selected_action == SelectedAction::ExplainBlockers {
            cannot_respond = true;
        }

        let mut report = DecisionReport::default();
        report.allowed_capabilities = allowed_capabilities.clone();
        report.stop_reasons = stop_reasons_report;
        report.blocked_candidates_count = blocked_candidates_count;
        report.top_3_block_reasons = top_block_reasons;
        report.selected_action = selected_action.as_str().to_string();
        report.fallback_used = fallback_used;
        report.fallback_type = fallback_type;
        report.selected_action_source = selected_source;
        report.cannot_respond = cannot_respond;
        if report.cannot_respond || selected_action == SelectedAction::ExplainBlockers {
            report.unblock_instructions = Some(unblock_instructions);
        }
        if let Some(ctx) = residual_context.as_ref() {
            report.residual_influence_mode = Some(match ctx.mode {
                ResidualInfluenceMode::Shadow => "shadow".to_string(),
                ResidualInfluenceMode::Live => "live".to_string(),
                ResidualInfluenceMode::Degraded => "degraded".to_string(),
            });
            report.residual_bias = Some(ctx.bias as f64);
            if matches!(ctx.mode, ResidualInfluenceMode::Shadow) {
                report.residual_shadow_would_change = residual_shadow_would_change;
                report.residual_shadow_impact_pct = residual_shadow_impact_pct;
            }
        }

        KernelDecision {
            accepted,
            rejected,
            caps_applied: caps_applied.into_iter().collect(),
            report,
        }
    }

    pub(super) async fn log_tool_rejections(&self, rejected: &[RejectedCandidate]) {
        for rejected_candidate in rejected.iter() {
            if !matches!(rejected_candidate.kind, CandidateKind::ToolCall) {
                continue;
            }
            let tool_name = rejected_candidate.tool_name.clone().unwrap_or_default();
            let source = rejected_candidate.source.clone().unwrap_or_default();
            let is_monologue = rejected_candidate.is_monologue.unwrap_or(false);
            let reason = rejected_candidate.reason.clone();
            if reason == "CONTROLLER_THROTTLE_TOOLS" {
                continue;
            }
            if reason == "UNKNOWN_TOOL" {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "tool",
                    None,
                    None,
                    json!({
                        "event": "tool_unknown_name",
                        "tool_name": tool_name,
                        "source": source,
                        "is_monologue": is_monologue,
                    }),
                )
                .await;
                let _ = system_log::log_contract_violation(
                    &self.db.pool,
                    Some(&self.app_handle),
                    None,
                    None,
                    "C2",
                    "unknown_tool",
                    Some(json!({
                        "tool_name": tool_name,
                        "source": source,
                        "is_monologue": is_monologue,
                    })),
                )
                .await;
            }
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "tool",
                None,
                None,
                json!({
                    "event": "tool_candidate_rejected",
                    "reason": reason,
                    "tool_name": tool_name,
                    "source": source,
                    "is_monologue": is_monologue,
                }),
            )
            .await;
        }
    }

    pub(super) async fn log_tool_bypasses(
        &self,
        decision: &KernelDecision,
        state: &KernelState,
        settings: &crate::models::Settings,
    ) {
        let controller_gate = state.controller_gate.as_ref();
        let throttle_tools = controller_gate.map(|g| g.throttle_tools).unwrap_or(false);
        let avoid_tools = state.stance.tool_preference == "avoid";
        let last_user_input = state.last_user_input.as_deref().unwrap_or("").to_lowercase();

        for candidate in decision.accepted.iter() {
            if !matches!(candidate.kind, CandidateKind::ToolCall) {
                continue;
            }
            if is_monologue_source(&candidate.source) {
                continue;
            }
            let tool_name = candidate
                .payload
                .get("tool_name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if tool_name.is_empty() {
                continue;
            }
            if !user_requested_tool_in_input(&last_user_input, tool_name) {
                continue;
            }
            if !self.is_known_tool_name(tool_name) || !self.is_allowed_tool_name(tool_name, settings) {
                continue;
            }

            let mut reason: Option<&'static str> = None;
            if avoid_tools && !is_local_tool_name(tool_name) {
                reason = Some("POLICY_TOOL_AVOID");
            } else if throttle_tools && candidate.urgency.unwrap_or(0) < 7 {
                reason = Some("CONTROLLER_THROTTLE_TOOLS");
            }
            if let Some(reason) = reason {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "tool",
                    None,
                    None,
                    json!({
                        "event": "tool_bypass_user_requested",
                        "reason": reason,
                        "tool_name": tool_name,
                    }),
                )
                .await;
            }
        }
    }

    pub(super) async fn log_decision(
        &self,
        decision: &KernelDecision,
        run_id: Option<&str>,
        trace_id: Option<&str>,
    ) {
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            run_id,
            trace_id,
            json!({
                "event": "kernel_cycle",
                "caps_applied": decision.caps_applied,
                "accepted": decision.accepted,
                "rejected": decision.rejected,
            }),
        )
        .await;

        for candidate in decision.accepted.iter() {
            if !matches!(candidate.kind, CandidateKind::EmitMessage | CandidateKind::FlagForHuman) {
                continue;
            }
            let speculative = candidate
                .payload
                .get("speculative")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let has_evidence = candidate_has_evidence(&candidate.payload)
                || matches!(candidate_evidence_class(&candidate), Some("internal"));
            let text = candidate_alignment_text(candidate).unwrap_or_default();
            let looks_like_question = text.trim().ends_with('?');
            if !speculative && !has_evidence && !looks_like_question {
                let _ = system_log::log_contract_violation(
                    &self.db.pool,
                    Some(&self.app_handle),
                    run_id,
                    trace_id,
                    "C1",
                    "ungrounded_assertion",
                    Some(json!({
                        "candidate_id": candidate.id,
                        "candidate_kind": format!("{:?}", candidate.kind),
                        "snippet": summarize_snippet(&text, 160),
                    })),
                )
                .await;
            }
        }

        let unknown_tool_rejections = decision
            .rejected
            .iter()
            .filter(|r| r.reason == "UNKNOWN_TOOL" || r.reason == "TOOL_DISABLED")
            .count();
        if unknown_tool_rejections > 0 {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "kernel",
                run_id,
                trace_id,
                json!({
                    "event": "unknown_tool_rejections",
                    "count": unknown_tool_rejections,
                }),
            )
            .await;
        }

        let emission_suppressed = decision
            .rejected
            .iter()
            .filter(|r| {
                matches!(
                    r.kind,
                    CandidateKind::EmitMessage | CandidateKind::AskUserQuestion | CandidateKind::FlagForHuman
                )
            })
            .count();
        if emission_suppressed > 0 {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                run_id,
                trace_id,
                json!({
                    "event": "emission_suppressed",
                    "count": emission_suppressed,
                }),
            )
            .await;
        }

        for rejected in decision.rejected.iter() {
            if rejected.reason.trim().is_empty() {
                let _ = system_log::log_contract_violation(
                    &self.db.pool,
                    Some(&self.app_handle),
                    run_id,
                    trace_id,
                    "C4",
                    "missing_suppression_reason",
                    Some(json!({
                        "candidate_id": rejected.id,
                        "candidate_kind": format!("{:?}", rejected.kind),
                    })),
                )
                .await;
            }
        }

        let deferred = decision
            .rejected
            .iter()
            .filter(|r| r.reason == "ASK_DEFERRED")
            .count();
        if deferred > 0 {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                run_id,
                trace_id,
                json!({
                    "event": "ask_deferred",
                    "count": deferred,
                }),
            )
            .await;
        }
    }
}
