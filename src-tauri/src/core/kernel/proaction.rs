use super::*;

fn pending_prompt_looks_jsonish(prompt: &str) -> bool {
    let trimmed = prompt.trim();
    if trimmed.len() < 2 {
        return false;
    }
    let wrapped = (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'));
    if !wrapped {
        return false;
    }
    let lower = trimmed.to_lowercase();
    if lower.contains("\"stance\"")
        || lower.contains("\"candidates\"")
        || lower.contains("\"done\"")
        || lower.contains("\"message\"")
    {
        return true;
    }
    serde_json::from_str::<Value>(trimmed).is_ok()
}

pub(crate) fn prompt_age_seconds(created_at: &str) -> Option<i64> {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(created_at) {
        let now = Utc::now();
        return Some(now.signed_duration_since(dt.with_timezone(&Utc)).num_seconds());
    }
    if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(created_at, "%Y-%m-%d %H:%M:%S") {
        let now = Utc::now();
        let dt = chrono::DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc);
        return Some(now.signed_duration_since(dt).num_seconds());
    }
    None
}

pub(crate) fn response_mentions_open_questions(response: &str, state: &KernelState) -> bool {
    if response.trim().is_empty() {
        return false;
    }
    let verified_questions = workspace_verified_open_questions(state);
    if verified_questions.is_empty() {
        return false;
    }
    let response_tokens = token_set(response);
    for question in verified_questions.iter() {
        let q_tokens = token_set(question);
        if response_tokens.intersection(&q_tokens).count() >= PROACTIVE_OVERLAP_THRESHOLD {
            return true;
        }
    }
    false
}

pub(crate) fn proactive_response_compliant(response: &str, state: &KernelState) -> bool {
    response_mentions_workspace(response, state) || response_mentions_open_questions(response, state)
}

pub(crate) fn workspace_alignment_tokens(state: &KernelState) -> HashSet<String> {
    let mut combined = String::new();
    if let Some(focus) = workspace_verified_focus(state) {
        combined.push_str(&focus);
        combined.push(' ');
    }
    for topic in workspace_verified_topics(state) {
        combined.push_str(&topic);
        combined.push(' ');
    }
    for question in workspace_verified_open_questions(state) {
        combined.push_str(&question);
        combined.push(' ');
    }
    token_set(&combined)
}

pub(crate) fn meta_cog_outcome_turns(settings: &Settings) -> i64 {
    settings.meta_cog_outcome_turns.unwrap_or(META_COG_OUTCOME_TURNS).max(1)
}

pub(crate) fn meta_cog_cycle_window_turns(settings: &Settings) -> i64 {
    settings
        .meta_cog_cycle_window_turns
        .unwrap_or(META_COG_CYCLE_WINDOW_TURNS)
        .max(1)
}

pub(crate) fn meta_cog_outcome_timeout_secs(settings: &Settings) -> i64 {
    settings
        .meta_cog_outcome_timeout_s
        .unwrap_or(META_COG_OUTCOME_TIMEOUT_SECS)
        .max(30)
}

pub(crate) fn meta_cog_cooldown_secs(settings: &Settings) -> i64 {
    settings
        .meta_cog_cooldown_s
        .unwrap_or(META_COG_COOLDOWN_SECS)
        .max(10)
}

pub(crate) fn meta_cog_streak_limit(settings: &Settings) -> i32 {
    settings
        .meta_cog_streak_limit
        .unwrap_or(META_COG_OUTCOME_STREAK_LIMIT)
        .max(1)
}

pub(crate) fn adjust_meta_cog_adaptive_multiplier(state: &mut KernelState, outcome: &str) -> f32 {
    let mut multiplier = state.meta_cog_adaptive_multiplier;
    match outcome {
        "resolved" => {
            multiplier = (multiplier - 0.5).max(1.0);
        }
        "cycling" | "no_signal" => {
            multiplier = (multiplier + 0.5).min(4.0);
        }
        _ => {}
    }
    state.meta_cog_adaptive_multiplier = multiplier;
    multiplier
}

pub(crate) fn is_monologue_source(source: &str) -> bool {
    matches!(source, "monologue" | "self_dialogue")
}

pub(crate) fn is_monologue_intent_candidate(candidate: &Candidate) -> bool {
    is_monologue_source(&candidate.source)
        && matches!(
            candidate.kind,
            CandidateKind::EmitMessage | CandidateKind::AskUserQuestion | CandidateKind::FlagForHuman
        )
}

pub(crate) fn candidate_evidence_class(candidate: &Candidate) -> Option<&str> {
    candidate
        .payload
        .get("evidence_class")
        .and_then(|v| v.as_str())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
}

pub(crate) fn is_meta_cog_candidate(candidate: &Candidate) -> bool {
    if !is_monologue_source(&candidate.source) {
        return false;
    }
    if matches!(candidate.kind, CandidateKind::NoOp) {
        return true;
    }
    if !matches!(
        candidate.kind,
        CandidateKind::UpdateGoalThread
            | CandidateKind::UpdateWorkspace
            | CandidateKind::AskUserQuestion
            | CandidateKind::EmitMessage
            | CandidateKind::ToolCall
            | CandidateKind::FlagForHuman
    ) {
        return false;
    }
    if candidate
        .payload
        .get("meta_cog")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return true;
    }
    matches!(candidate_evidence_class(candidate), Some("internal"))
}

pub(crate) fn ensure_meta_cog_payload(candidate: &mut Candidate) {
    if !is_monologue_source(&candidate.source) {
        return;
    }
    if !matches!(
        candidate.kind,
        CandidateKind::UpdateGoalThread
            | CandidateKind::UpdateWorkspace
            | CandidateKind::AskUserQuestion
            | CandidateKind::EmitMessage
            | CandidateKind::ToolCall
            | CandidateKind::FlagForHuman
    ) {
        return;
    }
    let mut obj = candidate
        .payload
        .as_object()
        .cloned()
        .unwrap_or_default();
    obj.entry("meta_cog".to_string())
        .or_insert(Value::Bool(true));
    obj.entry("evidence_class".to_string())
        .or_insert(Value::String("internal".to_string()));
    candidate.payload = Value::Object(obj);
}

pub(crate) fn pending_prompt_force_reason(
    skip_count: i64,
    auto_surface: bool,
    age_seconds: Option<i64>,
) -> Option<&'static str> {
    if skip_count >= PENDING_PROMPT_STARVATION_LIMIT {
        return Some("starvation");
    }
    if auto_surface {
        if let Some(age) = age_seconds {
            if age >= AUTO_SURFACE_MAX_AGE_SECS {
                return Some("auto_surface_max_age");
            }
            if age >= AUTO_SURFACE_SLA_SECS {
                return Some("auto_surface_sla");
            }
        }
    }
    None
}

pub(crate) fn coerce_proactive_candidate_for_evidence(
    candidate: &Candidate,
    state: &KernelState,
    evidence_ok: bool,
    payload_has_ids: bool,
    disable_working_hypothesis: bool,
) -> CoercedCandidate {
    if !matches!(candidate.kind, CandidateKind::EmitMessage) {
        return CoercedCandidate {
            candidate: Some(candidate.clone()),
            speculation_marked: false,
            blocked_reason: None,
        };
    }
    let Some(text) = candidate_alignment_text(candidate) else {
        return CoercedCandidate {
            candidate: None,
            speculation_marked: false,
            blocked_reason: Some("empty_text".to_string()),
        };
    };
    let payload_has_evidence = payload_has_ids;
    let evidence_ok = evidence_ok || controller_evidence_ok(state);
    let exact_open_question = candidate_exact_open_question(&text, state);
    let introduces_new_terms = !payload_has_evidence && candidate_introduces_new_terms(&text, state);

    if exact_open_question {
        let mut payload = candidate.payload.clone();
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("question".to_string(), Value::String(text.clone()));
            obj.insert("content".to_string(), Value::String(text.clone()));
            obj.insert("speculative".to_string(), Value::Bool(false));
        } else {
            payload = json!({
                "question": text,
                "content": text,
                "speculative": false
            });
        }
        let mut next = candidate.clone();
        next.kind = CandidateKind::AskUserQuestion;
        next.payload = payload;
        return CoercedCandidate {
            candidate: Some(next),
            speculation_marked: false,
            blocked_reason: None,
        };
    }

    if evidence_ok && !introduces_new_terms {
        return CoercedCandidate {
            candidate: Some(candidate.clone()),
            speculation_marked: false,
            blocked_reason: None,
        };
    }

    let question = if exact_open_question {
        text.clone()
    } else {
        working_hypothesis_prefix(&text, disable_working_hypothesis)
    };
    if question.trim().is_empty() {
        return CoercedCandidate {
            candidate: None,
            speculation_marked: false,
            blocked_reason: Some("empty_question".to_string()),
        };
    }

    let mut payload = candidate.payload.clone();
    if let Some(obj) = payload.as_object_mut() {
        obj.insert("question".to_string(), Value::String(question.clone()));
        obj.insert("content".to_string(), Value::String(question.clone()));
        obj.insert("speculative".to_string(), Value::Bool(!exact_open_question));
    } else {
        payload = json!({
            "question": question,
            "content": question,
            "speculative": !exact_open_question
        });
    }

    let mut next = candidate.clone();
    next.kind = CandidateKind::AskUserQuestion;
    next.payload = payload;

    CoercedCandidate {
        candidate: Some(next),
        speculation_marked: !exact_open_question,
        blocked_reason: None,
    }
}

pub(crate) fn candidate_alignment_metrics(candidate: &Candidate, state: &KernelState) -> (usize, bool) {
    let Some(text) = candidate_alignment_text(candidate) else {
        return (0, false);
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return (0, false);
    }
    let exact_open_question = workspace_verified_open_questions(state)
        .iter()
        .any(|q| q.trim().eq_ignore_ascii_case(trimmed));
    let candidate_tokens = token_set(trimmed);
    let workspace_tokens = workspace_alignment_tokens(state);
    let overlap = candidate_tokens
        .intersection(&workspace_tokens)
        .count();
    (overlap, exact_open_question)
}

pub(crate) fn proactive_followup_allowed(state: &KernelState, candidate: &Candidate) -> bool {
    let Some(last_question) = state.last_proactive_question.as_deref() else {
        return false;
    };
    let requested = extract_requested_slots(&candidate.payload);
    if !requested.is_empty() && !state.last_asked_slots.is_empty() {
        if normalize_slot_set(&requested) == normalize_slot_set(&state.last_asked_slots) {
            return true;
        }
    }
    let Some(candidate_text) = candidate_alignment_text(candidate) else {
        return false;
    };
    let candidate_tokens = token_set(&candidate_text);
    let last_tokens = token_set(last_question);
    candidate_tokens.intersection(&last_tokens).count() >= PROACTIVE_OVERLAP_THRESHOLD
}

pub(crate) fn proactive_memory_pass_due(state: &KernelState, now: DateTime<Utc>) -> bool {
    let Some(last_ts) = state.last_proactive_memory_pass_at.as_deref() else {
        return true;
    };
    let Ok(parsed) = DateTime::parse_from_rfc3339(last_ts) else {
        return true;
    };
    let elapsed = now.signed_duration_since(parsed.with_timezone(&Utc)).num_seconds();
    elapsed >= PROACTIVE_MEMORY_PASS_MIN_SECS
}

pub(crate) fn monologue_acceptance_failures(metrics: &ProactionMetrics) -> Vec<String> {
    let mut failures = Vec::new();
    if metrics.monologue_attempted_turns == 0 {
        return failures;
    }
    if metrics.monologue_ds_fts_ratio < 0.60 {
        failures.push("ds_fts_ratio".to_string());
    }
    if metrics.monologue_tick_end_rate < 0.95 {
        failures.push("tick_end_rate".to_string());
    }
    if metrics.monologue_timeouts > 2 {
        failures.push("timeouts_per_hour".to_string());
    }
    if metrics.monologue_suppression_rate > 0.25 {
        failures.push("suppression_rate".to_string());
    }
    if metrics.monologue_drift_reanchor_rate > 0.03 {
        failures.push("drift_reanchor_rate".to_string());
    }
    if metrics.monologue_safety_violations > 1 {
        failures.push("safety_violations".to_string());
    }
    failures
}

pub(crate) async fn clear_monologue_intents(kernel: &Kernel, conversation_id: &str) {
    let rows = sqlx::query(
        "SELECT id, intent_kind, bridge_id
         FROM pending_user_prompts
         WHERE conversation_id = ?
           AND auto_surface = 1
           AND source = 'monologue'",
    )
    .bind(conversation_id)
    .fetch_all(&kernel.db.pool)
    .await
    .unwrap_or_default();
    for row in rows {
        let id: String = row.get("id");
        let intent_kind: Option<String> = row.try_get("intent_kind").ok();
        let bridge_id: Option<String> = row.try_get("bridge_id").ok();
        let _ = system_log::log_event(
            &kernel.db.pool,
            Some(&kernel.app_handle),
            "info",
            "kernel",
            None,
            None,
            json!({
                "event": "intent_dropped",
                "candidate_id": id,
                "candidate_kind": intent_kind,
                "bridge_id": bridge_id,
                "reason": "replaced_by_new_intent",
            }),
        )
        .await;
    }
    let _ = sqlx::query(
        "DELETE FROM pending_user_prompts WHERE conversation_id = ? AND auto_surface = 1 AND source = 'monologue'",
    )
    .bind(conversation_id)
    .execute(&kernel.db.pool)
    .await;
}

pub(crate) async fn record_monologue_intent(
    kernel: &Kernel,
    conversation_id: &str,
    content: &str,
    intent_kind: &str,
) -> Option<(String, String)> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }
    if pending_prompt_looks_jsonish(trimmed) {
        let _ = system_log::log_event(
            &kernel.db.pool,
            Some(&kernel.app_handle),
            "warn",
            "kernel",
            None,
            None,
            json!( {
                "event": "pending_prompt_sanitized",
                "reason": "json_like",
                "candidate_kind": intent_kind,
                "source": "monologue",
            }),
        )
        .await;
        return None;
    }
    clear_monologue_intents(kernel, conversation_id).await;
    let bridge_id = Uuid::new_v4().to_string();
    let expires_at = compute_expires_at(Utc::now(), PENDING_PROMPT_EXPIRES_SECS);
    let prompt_id = kernel
        .db
        .enqueue_pending_prompt(
            conversation_id,
            trimmed,
            "monologue",
            true,
            Some(intent_kind),
            Some(&bridge_id),
            Some(&expires_at),
        )
        .await
        .ok()?;
    let _ = system_log::log_event(
        &kernel.db.pool,
        Some(&kernel.app_handle),
        "info",
        "kernel",
        None,
        None,
        json!({
            "event": "intent_queued",
            "candidate_kind": intent_kind,
            "candidate_id": prompt_id,
            "bridge_id": bridge_id,
        }),
    )
    .await;
    let _ = system_log::log_event(
        &kernel.db.pool,
        Some(&kernel.app_handle),
        "info",
        "kernel",
        None,
        None,
        json!({
            "event": "meta_cog_intent_queued",
            "candidate_kind": intent_kind,
            "candidate_id": prompt_id,
            "bridge_id": bridge_id,
        }),
    )
    .await;
    {
        let db = kernel.db.clone();
        let model_client = kernel.model_client.clone();
        let app_handle = kernel.app_handle.clone();
        let conversation_id = conversation_id.to_string();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(AUTO_SURFACE_SLA_SECS as u64)).await;
            let kernel = Kernel::new(db, model_client, app_handle);
            let _ = kernel
                .auto_surface_pending_prompts(&conversation_id, None, None, "auto_surface_sla")
                .await;
        });
    }
    Some((prompt_id, bridge_id))
}
