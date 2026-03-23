use super::*;

pub(crate) struct EpisodicWrite {
    pub event_type: String,
    pub payload: Value,
    pub source_type: String,
    pub source_ref: Option<String>,
}

#[derive(Clone)]
pub(crate) struct ThreadRunRequest {
    pub thread_id: String,
    pub goal: String,
    pub depth: i64,
}

#[derive(Clone)]
pub(crate) struct CommitResult {
    pub emit_content: Option<String>,
    pub emit_source: Option<String>,
    pub tool_dispatches: Vec<ToolDispatchRequest>,
    pub thread_run: Option<ThreadRunRequest>,
    pub research_cost: i64,
    pub ask_question: Option<String>,
    pub ask_slots: Vec<String>,
}

pub(crate) struct AnchorShiftEvent {
    pub old_anchor: String,
    pub new_anchor: String,
    pub reason: String,
    pub overlap: f32,
    pub anchor_epoch: i64,
}

pub(crate) fn question_fingerprint(question: &str, slots: &[String]) -> String {
    let mut parts = Vec::new();
    let norm_q = normalize_question(question);
    if !norm_q.is_empty() {
        parts.push(norm_q);
    }
    let slot_hash = hash_slot_set(slots);
    if !slot_hash.is_empty() {
        parts.push(slot_hash);
    }
    if parts.is_empty() {
        return String::new();
    }
    hash_payload(&parts.join("::"))
}

pub(crate) fn normalize_emit_message(text: &str) -> String {
    text.trim()
        .trim_end_matches(|c: char| c == '.' || c == '!' || c == '?')
        .to_lowercase()
}

pub(crate) fn emit_fingerprint(message: &str) -> String {
    let norm = normalize_emit_message(message);
    if norm.is_empty() {
        return String::new();
    }
    hash_payload(&norm)
}

pub(crate) fn is_loop_break_reason(reason: &str) -> bool {
    let normalized = reason.trim_start_matches("meta_cog_");
    matches!(
        normalized,
        "loop_detected"
            | "reanchor_needed"
            | "emit_loop_breaker"
            | "anchor_absent"
            | "tool_fabrication_repeat"
    )
}

pub(crate) fn decision_needed_for(state: &KernelState, last_monologue_at_override: Option<&str>) -> bool {
    let user_ts = state
        .last_user_input_at
        .as_deref()
        .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok());
    let mono_ts = last_monologue_at_override
        .or_else(|| state.last_monologue_completed_at.as_deref())
        .or_else(|| state.last_monologue_at.as_deref())
        .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok());
    match (user_ts, mono_ts) {
        (Some(user), Some(mono)) => user > mono,
        (Some(_), None) => true,
        _ => false,
    }
}

pub(crate) fn action_gate_reason_for(candidate: &Candidate, state: &KernelState, internal_cycle: bool) -> Option<String> {
    let is_monologue_intent = is_monologue_source(&candidate.source)
        && matches!(
            candidate.kind,
            CandidateKind::EmitMessage | CandidateKind::AskUserQuestion | CandidateKind::FlagForHuman
        );

    if matches!(state.task_phase, TaskPhase::Aborting | TaskPhase::Terminated) {
        return match candidate.kind {
            CandidateKind::EmitMessage | CandidateKind::Terminate => None,
            _ => Some("TASK_PHASE_HALTED".to_string()),
        };
    }

    if is_meta_cog_candidate(candidate) {
        if let Some(until) = state.meta_cog_cooldown_until.as_deref() {
            if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(until) {
                if chrono::Utc::now() < ts.with_timezone(&Utc) {
                    return Some("META_COG_COOLDOWN".to_string());
                }
            }
        }
    }

    if matches!(state.task_phase, TaskPhase::ResolvingWithDefaults) {
        if matches!(candidate.kind, CandidateKind::AskUserQuestion) {
            return Some("TASK_PHASE_DEFAULTS_NO_ASK".to_string());
        }
    }

    if matches!(candidate.kind, CandidateKind::RecordSelfClaim) {
        let evidence_ids = extract_id_list(&candidate.payload, "evidence_event_ids");
        let belief_ids = extract_id_list(&candidate.payload, "belief_ids");
        let provisional = candidate
            .payload
            .get("provisional")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let source_type = candidate
            .payload
            .get("source_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let provisional_allowed = provisional && source_type.eq_ignore_ascii_case("system_state");
        if evidence_ids.is_empty() && belief_ids.is_empty() && !provisional_allowed {
            return Some("SELF_CLAIM_MISSING_EVIDENCE".to_string());
        }
    }

    if matches!(candidate.kind, CandidateKind::UpdateWorkspace) {
        let goal_stack = extract_goal_stack_from_payload(&candidate.payload);
        if !goal_stack.is_empty() {
            let has_evidence = !extract_id_list(&candidate.payload, "evidence_event_ids").is_empty()
                || !extract_id_list(&candidate.payload, "belief_ids").is_empty()
                || goal_stack_has_evidence(&goal_stack);
            if goal_stack_advances(&state.workspace_goal_stack, &goal_stack) && !has_evidence {
                return Some("GOAL_STACK_MISSING_EVIDENCE".to_string());
            }
        }
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
            let missing: HashSet<String> = state
                .missing_slots
                .iter()
                .map(|s| normalize_slot_key(s))
                .collect();
            if required_slots.iter().any(|slot| missing.contains(slot)) {
                let policy = state
                    .missing_input_policy
                    .as_deref()
                    .unwrap_or("use_defaults_and_label");
                if policy.eq_ignore_ascii_case("strict") {
                    return Some("MISSING_SLOTS_STRICT".to_string());
                }
            }
        }
    }

    if matches!(candidate.kind, CandidateKind::ToolCall) {
        let tool_name = candidate
            .payload
            .get("tool_name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let args = candidate
            .payload
            .get("arguments")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !tool_name.is_empty() {
            let fingerprint = tool_fingerprint(tool_name, args);
            if state.tool_call_fingerprints.iter().any(|f| f == &fingerprint) {
                return Some("TOOL_CALL_REPEAT".to_string());
            }
        }
    }

    if matches!(candidate.kind, CandidateKind::AskUserQuestion) {
        let last_user_input = state.last_user_input.as_deref().unwrap_or("");
        let clarifier_for_latest = candidate_overlaps_last_user_input(candidate, last_user_input);
        if !internal_cycle && state.ask_budget_remaining <= 0 && !clarifier_for_latest && !is_monologue_intent {
            return Some("ASK_BUDGET_EXHAUSTED".to_string());
        }
        if state.ask_loop_breaker_triggered {
            return Some("ASK_LOOP_BREAKER".to_string());
        }
        if internal_cycle {
            if let Some(until) = state.hypothesis_defer_until {
                let stalled = state.stalled_count >= 2 || state.failure_count >= 2;
                if state.monologue_count < until && !stalled && !clarifier_for_latest {
                    return Some("ASK_DEFERRED".to_string());
                }
            }
        }
        let requested_slots = extract_requested_slots(&candidate.payload);
        if state.missing_input_policy.is_some() && requested_slots.is_empty() {
            return Some("ASK_MISSING_SLOT_LIST".to_string());
        }
        if !requested_slots.is_empty() && is_slot_subset(&requested_slots, &state.asked_slot_sets) {
            return Some("ASKED_SLOT_SET_REPEAT".to_string());
        }
        if !requested_slots.is_empty() {
            let refused: HashSet<String> = state
                .refused_slots
                .iter()
                .map(|s| normalize_slot_key(s))
                .collect();
            if requested_slots
                .iter()
                .any(|slot| refused.contains(&normalize_slot_key(slot)))
            {
                return Some("REFUSED_SLOTS".to_string());
            }
        }
    }

    if internal_cycle && matches!(candidate.kind, CandidateKind::EmitMessage) {
        if let Some(until) = state.monologue_quiet_until.as_deref() {
            if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(until) {
                if chrono::Utc::now() < ts.with_timezone(&Utc) {
                    return Some("MONOLOGUE_QUIET_LATCH".to_string());
                }
            }
        }
    }

    if matches!(candidate.kind, CandidateKind::EmitMessage) {
        if state.monologue_emit_loop_breaker_triggered {
            return Some("EMIT_LOOP_BREAKER".to_string());
        }
        let content = candidate
            .payload
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if !content.is_empty() {
            let fingerprint = emit_fingerprint(&content);
            if !fingerprint.is_empty()
                && state
                    .recent_emit_fingerprints
                    .iter()
                    .any(|f| f == &fingerprint)
            {
                return Some("EMIT_REPEAT".to_string());
            }
        }
    }

    None
}

pub(crate) fn apply_loop_detection_for(
    candidates: &[Candidate],
    state: &mut KernelState,
    settings: &crate::models::Settings,
) {
    let k = settings.loop_recent_k.unwrap_or(6).max(1) as usize;
    let threshold = settings
        .loop_similarity_threshold
        .unwrap_or(0.85)
        .clamp(0.0, 1.0);

    for candidate in candidates {
        if matches!(candidate.kind, CandidateKind::AskUserQuestion) {
            let question = candidate
                .payload
                .get("payload")
                .and_then(|v| v.as_str())
                .or_else(|| candidate.payload.get("question").and_then(|v| v.as_str()))
                .or_else(|| candidate.payload.get("content").and_then(|v| v.as_str()))
                .unwrap_or("")
                .trim()
                .to_string();
            if question.is_empty() {
                continue;
            }
            let slots = extract_requested_slots(&candidate.payload);
            let fingerprint = question_fingerprint(&question, &slots);
            if !fingerprint.is_empty() && state.question_fingerprints.iter().any(|f| f == &fingerprint) {
                state.ask_loop_breaker_triggered = true;
            } else {
                for prior in state.recent_questions.iter().rev().take(k) {
                    let sim = token_similarity(&question, prior);
                    if sim >= threshold {
                        state.ask_loop_breaker_triggered = true;
                        break;
                    }
                }
            }
            if state.ask_loop_breaker_triggered {
                if !matches!(state.task_phase, TaskPhase::Aborting | TaskPhase::Terminated) {
                    state.task_phase = TaskPhase::ResolvingWithDefaults;
                }
                if state.resolution_mode.is_none() {
                    state.resolution_mode = Some("defaults_used".to_string());
                }
                break;
            }
        }
        if matches!(candidate.kind, CandidateKind::ToolCall) {
            let tool_name = candidate
                .payload
                .get("tool_name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let args = candidate
                .payload
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if tool_name.is_empty() {
                continue;
            }
            let fingerprint = tool_fingerprint(tool_name, args);
            if state.tool_call_fingerprints.iter().any(|f| f == &fingerprint) {
                state.tool_loop_breaker_triggered = true;
            }
        }
    }
}

pub(crate) fn apply_emit_loop_detection_for(
    candidates: &[Candidate],
    state: &mut KernelState,
    settings: &crate::models::Settings,
) {
    let k = settings.loop_recent_k.unwrap_or(6).max(1) as usize;
    let threshold = settings
        .loop_similarity_threshold
        .unwrap_or(0.85)
        .clamp(0.0, 1.0);

    for candidate in candidates {
        if !matches!(candidate.kind, CandidateKind::EmitMessage | CandidateKind::FlagForHuman) {
            continue;
        }
        let message = candidate
            .payload
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if message.is_empty() {
            continue;
        }
        let fingerprint = emit_fingerprint(&message);
        if !fingerprint.is_empty()
            && state
                .recent_emit_fingerprints
                .iter()
                .any(|f| f == &fingerprint)
        {
            state.monologue_emit_loop_breaker_triggered = true;
            break;
        }
        for prior in state.recent_emit_messages.iter().rev().take(k) {
            let sim = token_similarity(&message, prior);
            if sim >= threshold {
                state.monologue_emit_loop_breaker_triggered = true;
                break;
            }
        }
        if state.monologue_emit_loop_breaker_triggered {
            break;
        }
    }
}
