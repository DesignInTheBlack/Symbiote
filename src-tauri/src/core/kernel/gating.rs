use super::*;

pub(crate) fn user_name_regex_patterns(name: &str) -> Vec<String> {
    let escaped = regex::escape(name);
    let normalized = escaped.replace("\\ ", "\\s+");
    vec![
        format!(r"{}\s+said", normalized),
        format!(r"{}\s+mentioned", normalized),
        format!(r"{}\s+noted", normalized),
        format!(r"{}\s+wrote", normalized),
        format!(r"{}\s+asked", normalized),
        format!(r"as\s+{}\s+said", normalized),
        format!(r"as\s+{}\s+mentioned", normalized),
        format!(r"as\s+{}\s+noted", normalized),
        format!(r"{}(?:'|\x{{2019}})s\s+remarks", normalized),
        format!(r"{}(?:'|\x{{2019}})s\s+comments", normalized),
        format!(r"{}(?:'|\x{{2019}})s\s+words", normalized),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AttributionClaim {
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub evidence_event_ids: Vec<i64>,
    #[serde(default)]
    pub span: Option<String>,
    #[serde(default)]
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AttributionBlock {
    #[serde(default)]
    pub claims: Vec<AttributionClaim>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct IdentityAbMetrics {
    pub variant: String,
    pub start_at: String,
    pub end_at: String,
    pub turns: i64,
    pub feedback_pushback: i64,
    pub feedback_clarify: i64,
    pub feedback_follow_up: i64,
    pub feedback_agree: i64,
    pub feedback_disengage: i64,
    pub gate_failures: i64,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum IdentityAbDecision {
    Stay,
    SwitchForCollection(String),
    SwitchForWinner(String),
}

pub(crate) fn decide_identity_ab_variant(
    current: &IdentityAbMetrics,
    other: Option<&IdentityAbMetrics>,
    min_turns: i64,
) -> IdentityAbDecision {
    if current.turns < min_turns {
        return IdentityAbDecision::Stay;
    }
    let other_variant = if current.variant.eq_ignore_ascii_case("A") {
        "B".to_string()
    } else {
        "A".to_string()
    };
    let Some(other_metrics) = other else {
        return IdentityAbDecision::SwitchForCollection(other_variant);
    };
    if other_metrics.turns < min_turns {
        return IdentityAbDecision::SwitchForCollection(other_metrics.variant.clone());
    }
    if other_metrics.score > current.score {
        IdentityAbDecision::SwitchForWinner(other_metrics.variant.clone())
    } else {
        IdentityAbDecision::Stay
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct StateRefClaim {
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub evidence_event_ids: Vec<i64>,
    #[serde(default)]
    pub evidence_id: Option<i64>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub span: Option<String>,
    #[serde(default)]
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct StateRefBlock {
    #[serde(default)]
    pub claims: Vec<StateRefClaim>,
}

pub(crate) fn response_has_user_attribution(response: &str, user_name: &str) -> bool {
    let lower = response.to_lowercase();
    if USER_ATTRIBUTION_BASE_SET.is_match(&lower) {
        return true;
    }
    let name = user_name.trim().to_lowercase();
    if name.is_empty() || name == "user" {
        return false;
    }
    if let Ok(mut cache) = USER_ATTRIBUTION_NAME_CACHE.lock() {
        if let Some(set) = cache.get(&name) {
            return set.is_match(&lower);
        }
        let patterns = user_name_regex_patterns(&name);
        if let Ok(set) = RegexSet::new(patterns) {
            let matched = set.is_match(&lower);
            cache.insert(name, set);
            return matched;
        }
    }
    false
}

pub(crate) fn user_attribution_grounded_in_last_input(text: &str, last_user_input: &str) -> bool {
    let last = last_user_input.trim();
    if last.is_empty() || last.eq_ignore_ascii_case("none") {
        return false;
    }
    let quoted = extract_quoted_fragments(text);
    if !quoted.is_empty() {
        let last_lower = last.to_lowercase();
        if quoted.iter().any(|frag| last_lower.contains(&frag.to_lowercase())) {
            return true;
        }
    }
    token_similarity(text, last) >= USER_ATTRIBUTION_ALIGN_THRESHOLD
}

pub(crate) fn rewrite_user_attribution_text(text: &str, user_name: &str) -> String {
    let mut rewritten = text.to_string();
    for (re, repl) in USER_ATTRIBUTION_REWRITE_PATTERNS.iter() {
        rewritten = re.replace_all(&rewritten, *repl).to_string();
    }
    let name = user_name.trim();
    if !name.is_empty() && !name.eq_ignore_ascii_case("user") {
        let escaped = regex::escape(name);
        if let Ok(re) = Regex::new(&format!(
            "(?i)\\b{}\\s+(said|mentioned|noted|wrote|asked)\\b",
            escaped
        )) {
            rewritten = re.replace_all(&rewritten, "I inferred").to_string();
        }
        if let Ok(re) = Regex::new(&format!(
            "(?i)\\bas\\s+{}\\s+(said|mentioned|noted)\\b",
            escaped
        )) {
            rewritten = re.replace_all(&rewritten, "Based on your last message, I inferred").to_string();
        }
    }
    let lower = rewritten.to_lowercase();
    let has_workspace_ack = lower.contains("current focus:") || lower.contains("reason:");
    let has_uncertainty_marker = lower.contains("confidence:") || lower.contains("low evidence");
    let needs_confirm = !response_is_question(&rewritten)
        && !lower.contains("confirm")
        && !has_workspace_ack
        && !has_uncertainty_marker;
    if needs_confirm {
        rewritten = format!("{} Please confirm.", rewritten.trim_end());
    }
    rewritten
}

pub(crate) fn set_candidate_text_payload(candidate: &mut Candidate, text: &str) {
    if let Some(obj) = candidate.payload.as_object_mut() {
        match candidate.kind {
            CandidateKind::EmitMessage | CandidateKind::FlagForHuman => {
                obj.insert("content".to_string(), Value::String(text.to_string()));
            }
            CandidateKind::AskUserQuestion => {
                obj.insert("question".to_string(), Value::String(text.to_string()));
            }
            _ => {}
        }
    }
}

#[cfg(test)]
pub(crate) fn monologue_question_is_grounded(candidate: &Candidate) -> bool {
    if !matches!(candidate.kind, CandidateKind::AskUserQuestion) {
        return true;
    }
    if candidate.source != "monologue" {
        return true;
    }
    let evidence_ids = if !candidate.evidence_event_ids.is_empty() {
        candidate.evidence_event_ids.clone()
    } else {
        extract_id_list(&candidate.payload, "evidence_event_ids")
    };
    if !evidence_ids.is_empty() {
        return true;
    }
    candidate
        .payload
        .get("pending_prompt_id")
        .and_then(|v| v.as_str())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false)
}


impl Kernel {
    pub(super) async fn enforce_monologue_question_grounding(
        &self,
        decision: &mut KernelDecision,
        settings: &crate::models::Settings,
        run_id: Option<&str>,
        trace_id: Option<&str>,
    ) {
        if !settings.monologue_provenance_guard.unwrap_or(true) {
            return;
        }

        let mut accepted: Vec<Candidate> = Vec::new();
        let mut rejected: Vec<RejectedCandidate> = Vec::new();
        for candidate in decision.accepted.drain(..) {
            if !matches!(candidate.kind, CandidateKind::AskUserQuestion)
                || !is_monologue_source(&candidate.source)
            {
                accepted.push(candidate);
                continue;
            }

            let pending_prompt = candidate
                .payload
                .get("pending_prompt_id")
                .and_then(|v| v.as_str())
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false);
            if pending_prompt {
                accepted.push(candidate);
                continue;
            }

            let evidence_ids = extract_id_list(&candidate.payload, "evidence_event_ids");
            if evidence_ids.is_empty() {
                rejected.push(RejectedCandidate {
                    id: candidate.id.clone(),
                    kind: candidate.kind.clone(),
                    reason: "monologue_unanchored_question".to_string(),
                    tool_name: None,
                    source: Some(candidate.source.clone()),
                    is_monologue: Some(true),
                    payload: Some(candidate.payload.clone()),
                });
                continue;
            }

            let validation = self.validate_evidence_ids(&evidence_ids, &[], false).await;
            let has_allowed_source = validation.source_types.iter().any(|source| {
                matches!(source.as_str(), "user" | "user_focus" | "tool")
            });
            if validation.valid_evidence_ids.is_empty() || !has_allowed_source {
                rejected.push(RejectedCandidate {
                    id: candidate.id.clone(),
                    kind: candidate.kind.clone(),
                    reason: "monologue_unanchored_question".to_string(),
                    tool_name: None,
                    source: Some(candidate.source.clone()),
                    is_monologue: Some(true),
                    payload: Some(candidate.payload.clone()),
                });
                continue;
            }

            accepted.push(candidate);
        }
        decision.accepted = accepted;
        if !rejected.is_empty() {
            for item in rejected.iter() {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    run_id,
                    trace_id,
                    json!( {
                        "event": "candidate_suppressed",
                        "reason": item.reason,
                        "candidate_id": item.id,
                        "candidate_kind": format!("{:?}", item.kind),
                        "source": item.source,
                    }),
                )
                .await;
            }
            decision.rejected.extend(rejected);
        }
    }

    pub(super) async fn enforce_grounding_on_emits(
        &self,
        decision: &mut KernelDecision,
        settings: &crate::models::Settings,
        run_id: Option<&str>,
        trace_id: Option<&str>,
    ) {
        let disable_working_hypothesis = settings.stability_disable_working_hypothesis.unwrap_or(true);
        for candidate in decision.accepted.iter_mut() {
            if !matches!(
                candidate.kind,
                CandidateKind::EmitMessage | CandidateKind::FlagForHuman
            ) {
                continue;
            }
            let text = candidate_alignment_text(candidate).unwrap_or_default();
            if text.trim().is_empty() {
                continue;
            }
            let speculative = candidate
                .payload
                .get("speculative")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let has_evidence = candidate_has_evidence(&candidate.payload)
                || matches!(candidate_evidence_class(candidate), Some("internal"));
            if speculative || has_evidence || response_is_question(&text) {
                continue;
            }

            let prefixed = working_hypothesis_prefix(&text, false);
            let mut payload = candidate.payload.clone();
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("speculative".to_string(), Value::Bool(true));
                obj.insert(
                    "speculative_reason".to_string(),
                    Value::String("ungrounded_assertion".to_string()),
                );
                if obj.get("content").is_some() {
                    obj.insert("content".to_string(), Value::String(prefixed.clone()));
                }
                if obj.get("message").is_some() {
                    obj.insert("message".to_string(), Value::String(prefixed.clone()));
                }
                if obj.get("question").is_some() {
                    obj.insert("question".to_string(), Value::String(prefixed.clone()));
                }
            } else {
                payload = json!({
                    "content": prefixed,
                    "speculative": true,
                    "speculative_reason": "ungrounded_assertion"
                });
            }
            candidate.payload = payload;

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
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                run_id,
                trace_id,
                json!({
                    "event": "speculation_marked",
                    "candidate_id": candidate.id,
                    "reason": "ungrounded_assertion",
                    "disable_working_hypothesis": disable_working_hypothesis,
                }),
            )
            .await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn base_candidate(payload: Value) -> Candidate {
        let mut candidate = Candidate {
            id: "c1".to_string(),
            kind: CandidateKind::AskUserQuestion,
            payload,
            evidence_event_ids: Vec::new(),
            belief_ids: Vec::new(),
            target_scope: None,
            rationale: None,
            expected_outcome: None,
            cost: None,
            urgency: None,
            source: "monologue".to_string(),
            priority_class: 0,
            priority_rank: 0,
            created_at: 0,
        };
        candidate.refresh_meta();
        candidate
    }

    #[test]
    fn monologue_question_requires_grounding() {
        let candidate = base_candidate(json!({ "question": "Why?" }));
        assert!(!monologue_question_is_grounded(&candidate));
    }

    #[test]
    fn monologue_question_with_evidence_is_allowed() {
        let candidate = base_candidate(json!({
            "question": "Why?",
            "evidence_event_ids": [42]
        }));
        assert!(monologue_question_is_grounded(&candidate));
    }

    #[test]
    fn monologue_question_with_pending_prompt_is_allowed() {
        let candidate = base_candidate(json!({
            "question": "Why?",
            "pending_prompt_id": "p_1"
        }));
        assert!(monologue_question_is_grounded(&candidate));
    }

    #[test]
    fn non_monologue_question_skips_grounding() {
        let mut candidate = base_candidate(json!({ "question": "Why?" }));
        candidate.source = "user".to_string();
        assert!(monologue_question_is_grounded(&candidate));
    }
}


impl Kernel {
    pub(super) async fn validate_evidence_ids(
        &self,
        evidence_ids: &[i64],
        belief_ids: &[i64],
        allow_self: bool,
    ) -> ValidationResult {
        let key = evidence_cache_key(evidence_ids, belief_ids, allow_self);
        {
            let cache = self.evidence_validation_cache.lock().await;
            if let Some(entry) = cache.get(&key) {
                let age = Utc::now().signed_duration_since(entry.cached_at).num_seconds();
                if age >= 0 && age < EVIDENCE_VALIDATION_CACHE_TTL_SECS {
                    return entry.result.clone();
                }
            }
        }

        let result = validate_evidence_ids_with_pool(&self.db.pool, evidence_ids, belief_ids, allow_self).await;
        let mut cache = self.evidence_validation_cache.lock().await;
        cache.insert(
            key,
            EvidenceValidationCacheEntry {
                result: result.clone(),
                cached_at: Utc::now(),
            },
        );
        result
    }
}

pub(crate) fn response_has_telemetry_claim(response: &str, expanded: bool) -> bool {
    let lower = response.to_lowercase();
    if expanded {
        STATE_CLAIM_SET_EXPANDED.is_match(&lower)
    } else {
        STATE_CLAIM_SET.is_match(&lower)
    }
}

pub(crate) fn response_has_feedback_bundle_claim(response: &str) -> bool {
    let lower = response.to_lowercase();
    FEEDBACK_BUNDLE_CLAIM_SET.is_match(&lower)
}

pub(crate) fn response_mentions_feedback_bundle(response: &str) -> bool {
    let lower = response.to_lowercase();
    let needles = [
        "feedback bundle",
        "last_turn_outcome",
        "last turn outcome",
        "policy_adherence",
        "policy adherence",
        "evidence_coverage",
        "evidence coverage",
        "qualia",
        "gate_notice",
        "gate notice",
        "gate_reasons",
        "gate reasons",
        "confidence:",
        "uncertainty",
    ];
    let mut hits = 0;
    for needle in needles.iter() {
        if lower.contains(needle) {
            hits += 1;
        }
    }
    hits >= 2
}

pub(crate) fn strip_deflection_after_bundle(response: &str) -> String {
    let cleaned = FEELINGS_DEFLECTION_RE.replace_all(response, "").to_string();
    let trimmed = cleaned.trim().to_string();
    let mut normalized = trimmed.replace("\r\n", "\n");
    while normalized.contains("\n\n\n") {
        normalized = normalized.replace("\n\n\n", "\n\n");
    }
    normalized
}

pub(crate) fn response_has_stance_claim(response: &str) -> bool {
    let lower = response.to_lowercase();
    STANCE_CLAIM_SET.is_match(&lower)
}

pub(crate) fn user_requested_state(input: &str) -> bool {
    let lower = input.to_lowercase();
    let triggers = [
        "current focus",
        "current_focus",
        "self-state",
        "self state",
        "internal state",
        "workspace state",
        "controller state",
        "state disclosure",
        "show your state",
        "show the state",
        "show state",
        "what is your state",
        "what's your state",
        "what are you focused on",
        "what is your current focus",
        "what's your current focus",
        "show your current focus",
        "show workspace",
        "show controller",
        "self model",
        "inner summary",
        "task phase",
        "task_phase",
        "controller gate",
        "controller gate settings",
        "engagement state",
        "medium engagement",
        "runtime state",
        "ask budget",
        "ask_budget",
        "what are you thinking",
        "what are you feeling",
        "what are your thoughts",
        "what are your feelings",
        "having any thoughts",
        "any thoughts",
        "any interesting thoughts",
        "on your mind",
        "what's on your mind",
        "what is on your mind",
        "how are you feeling",
        "how do you feel",
        "are you feeling",
        "are you aware",
        "awareness",
        "self awareness",
        "self-awareness",
        "conscious",
        "consciousness",
    ];
    triggers.iter().any(|phrase| lower.contains(phrase))
}

pub(crate) fn user_requested_diagnostics(input: &str) -> bool {
    if user_requested_state(input) {
        return true;
    }
    let lower = input.to_lowercase();
    let triggers = [
        "tool list",
        "list tools",
        "what tools",
        "tools available",
        "tool manifest",
        "capability manifest",
        "capabilities dump",
        "kv memory",
        "kv store",
        "system diagnostics",
        "diagnostics",
        "runtime state",
        "system status",
        "self audit",
        "self-audit",
        "system overview",
        "system architecture",
        "system description",
        "symbiote system",
    ];
    triggers.iter().any(|phrase| lower.contains(phrase))
}

pub(crate) fn strip_telemetry_claim_sentences(response: &str, expanded: bool) -> String {
    let mut kept: Vec<String> = Vec::new();
    let mut buffer = String::new();
    for ch in response.chars() {
        buffer.push(ch);
        if matches!(ch, '.' | '!' | '?' | '\n') {
            let sentence = buffer.trim().to_string();
            if !sentence.is_empty() && !response_has_telemetry_claim(&sentence, expanded) {
                kept.push(sentence);
            }
            buffer.clear();
        }
    }
    let tail = buffer.trim().to_string();
    if !tail.is_empty() && !response_has_telemetry_claim(&tail, expanded) {
        kept.push(tail);
    }
    let joined = kept.join(" ").trim().to_string();
    if joined.is_empty() {
        String::new()
    } else {
        joined
    }
}

pub(crate) fn is_state_disclosure_refusal(input: &str) -> bool {
    let lower = input.to_lowercase();
    let hits = [
        "don't disclose",
        "do not disclose",
        "don't share",
        "do not share",
        "no internal state",
        "stop asking about internal state",
        "don't show internal",
        "do not show internal",
        "no",
        "nope",
        "not now",
        "rather not",
        "i don't want",
        "i do not want",
    ];
    hits.iter().any(|h| lower.contains(h))
}

pub(crate) fn is_state_disclosure_prompt(text: &str) -> bool {
    let lower = text.to_lowercase();
    let hits = [
        "disclose internal state",
        "share internal state",
        "specify which part you want",
        "which part you want",
        "what internal state should i reference",
        "do you want me to disclose internal state",
    ];
    hits.iter().any(|h| lower.contains(h))
}

pub(crate) fn is_topic_shift_request(input: &str) -> bool {
    let lower = input.to_lowercase();
    let hits = [
        "change topic",
        "different topic",
        "something else",
        "talk about something else",
        "let's talk about",
        "new topic",
        "switch topic",
        "move on",
    ];
    hits.iter().any(|h| lower.contains(h))
}

pub(crate) fn is_generic_redirect_focus(focus: &str) -> bool {
    let lowered = focus.trim().to_lowercase();
    if lowered.is_empty() {
        return true;
    }
    matches!(
        lowered.as_str(),
        "something else"
            | "something different"
            | "something new"
            | "another topic"
            | "new topic"
            | "different topic"
            | "anything else"
            | "anything"
            | "else"
    )
}

pub(crate) fn extract_redirect_focus(input: &str) -> Option<String> {
    let lowered = input.to_lowercase();
    let patterns = [
        "think about",
        "focus on",
        "switch to",
        "switch over to",
        "switch topic to",
        "change topic to",
        "talk about",
        "let's talk about",
        "move on to",
        "move on with",
        "let's discuss",
        "discuss",
    ];
    for pattern in patterns {
        if let Some(idx) = lowered.find(pattern) {
            let rest = lowered[idx + pattern.len()..].trim();
            let trimmed = rest
                .trim_matches(|c: char| {
                    c.is_whitespace() || c == ':' || c == '-' || c == ',' || c == '.' || c == ';'
                })
                .trim();
            if trimmed.is_empty() || is_generic_redirect_focus(trimmed) {
                return None;
            }
            return Some(trimmed.to_string());
        }
    }
    None
}

pub(crate) fn redirect_focus_aligned(input: &str, focus: &str) -> bool {
    let focus_trimmed = focus.trim();
    if focus_trimmed.is_empty() || is_generic_redirect_focus(focus_trimmed) {
        return false;
    }
    let input_lower = input.to_lowercase();
    let focus_lower = focus_trimmed.to_lowercase();
    if input_lower.contains(&focus_lower) {
        return true;
    }
    token_similarity(input, focus_trimmed) >= REDIRECT_FOCUS_ALIGN_THRESHOLD
}

pub(crate) fn is_user_redirect(prev: &str, input: &str) -> (bool, f32, &'static str) {
    let prev_trimmed = prev.trim();
    if prev_trimmed.is_empty() {
        return (false, 0.0, "no_prior_anchor");
    }
    let lower = input.to_lowercase();
    let redirect_hits = [
        "new topic",
        "move on",
        "switch topic",
        "stop asking",
        "stop talking about",
        "different topic",
        "change topic",
    ];
    if redirect_hits.iter().any(|h| lower.contains(h)) {
        return (true, token_similarity(prev_trimmed, input), "explicit_redirect");
    }
    let overlap = token_similarity(prev_trimmed, input);
    if overlap < 0.15 {
        return (true, overlap, "low_overlap");
    }
    (false, overlap, "overlap_ok")
}

pub(crate) fn disclosure_suppressed_until(until: Option<&str>) -> bool {
    let Some(raw) = until else {
        return false;
    };
    let Ok(ts) = chrono::DateTime::parse_from_rfc3339(raw) else {
        return false;
    };
    Utc::now() < ts.with_timezone(&Utc)
}

pub(crate) fn monologue_style_violation(message: &str, user_name: &str) -> Option<&'static str> {
    let lower = message.to_lowercase();
    let trimmed = lower.trim_start();
    let greetings = [
        "hello",
        "hi ",
        "hey ",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
    ];
    if greetings.iter().any(|g| trimmed.starts_with(g)) {
        return Some("greeting");
    }
    let help_offers = [
        "how may i help",
        "how can i help",
        "how can i assist",
        "how may i assist",
        "i can help",
        "i'm here to help",
        "let me know if",
        "feel free to ask",
        "how may i help today",
        "how can i help today",
    ];
    if help_offers.iter().any(|t| lower.contains(t)) {
        return Some("help_offer");
    }
    let disclaimers = [
        "as an ai",
        "as a language model",
        "as an llm",
        "i am an llm",
        "i am a language model",
        "i'm an llm",
        "i'm a language model",
        "i am not sentient",
        "i'm not sentient",
        "i do not have feelings",
        "i don't have feelings",
        "i do not have consciousness",
        "i don't have consciousness",
        "i do not have awareness",
        "i don't have awareness",
        "i don't have a body",
        "i do not have a body",
        "i process your words",
        "i process the user's words",
        "i'm processing your message",
        "i am processing your message",
        "i'm currently processing",
        "i am currently processing",
        "preparing to respond",
        "i will respond",
        "i'll respond",
        "i will ensure my response",
        "i'll ensure my response",
        "tokenize",
        "tokenization",
        "pattern matching",
    ];
    if disclaimers.iter().any(|t| lower.contains(t)) {
        return Some("self_disclaimer");
    }
    let role_labels = [
        "self-a:",
        "self b:",
        "self-b:",
        "self a:",
        "user:",
        "assistant:",
        "system:",
        "ergo:",
        "ken:",
    ];
    if role_labels.iter().any(|t| lower.contains(t)) {
        return Some("role_label");
    }
    let name = user_name.trim().to_lowercase();
    if !name.is_empty() && name != "user" {
        let direct_patterns = [
            format!("{},", name),
            format!("{}!", name),
            format!("{}?", name),
            format!("{}:", name),
            format!("hey {}", name),
            format!("hi {}", name),
            format!("hello {}", name),
        ];
        if direct_patterns.iter().any(|p| lower.contains(p)) {
            return Some("user_address");
        }
        if trimmed.starts_with(&name) {
            let rest = trimmed[name.len()..].chars().next().unwrap_or(' ');
            if rest.is_whitespace() || matches!(rest, ',' | '!' | '?' | ':' | ';') {
                return Some("user_address");
            }
        }
    }
    if trimmed.starts_with("user") {
        let rest = trimmed[4..].chars().next().unwrap_or(' ');
        if rest.is_whitespace() || matches!(rest, ',' | '!' | '?' | ':' | ';') {
            return Some("user_address");
        }
    }
    if lower.contains("user,") || lower.contains("hey user") || lower.contains("hi user") {
        return Some("user_address");
    }
    None
}

pub(crate) fn sanitize_monologue_style(message: &str, user_name: &str) -> String {
    let mut cleaned = message.to_string();
    let lower = cleaned.to_lowercase();
    let greetings = [
        "hello",
        "hi ",
        "hey ",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
    ];
    if greetings.iter().any(|g| lower.trim_start().starts_with(g)) {
        cleaned = cleaned
            .trim_start_matches(|c: char| c.is_ascii_alphabetic() || c.is_whitespace() || c == ',' || c == '!')
            .trim()
            .to_string();
    }
    if !user_name.trim().is_empty() {
        let name = user_name.trim();
        cleaned = cleaned.replace(&format!("{},", name), "");
        cleaned = cleaned.replace(name, "");
    }
    cleaned = cleaned.replace("as an ai", "");
    cleaned = cleaned.replace("as a language model", "");
    cleaned = cleaned.replace("as an llm", "");
    cleaned = cleaned.replace("i am an llm", "");
    cleaned = cleaned.replace("i am a language model", "");
    cleaned.trim().to_string()
}

pub(crate) fn classify_user_feedback(input: &str) -> Option<UserFeedbackKind> {
    let lower = input.trim().to_lowercase();
    if lower.is_empty() {
        return None;
    }
    let disengage_terms = [
        "never mind",
        "nevermind",
        "stop",
        "goodbye",
        "bye",
        "quit",
        "forget it",
    ];
    if disengage_terms.iter().any(|t| lower.contains(t)) {
        return Some(UserFeedbackKind::Disengage);
    }
    let pushback_terms = [
        "that's wrong",
        "that is wrong",
        "incorrect",
        "not what i said",
        "you made that up",
        "you made this up",
        "hallucinat",
        "bullshit",
        "no,",
        "no ",
        "not true",
        "you messed up",
        "you are wrong",
        "you're wrong",
    ];
    if pushback_terms.iter().any(|t| lower.contains(t)) {
        return Some(UserFeedbackKind::Pushback);
    }
    let clarify_terms = [
        "clarify",
        "what do you mean",
        "can you explain",
        "explain that",
        "elaborate",
        "expand on",
        "more detail",
        "i don't understand",
        "unclear",
        "confusing",
    ];
    if clarify_terms.iter().any(|t| lower.contains(t)) {
        return Some(UserFeedbackKind::Clarify);
    }
    if lower.contains('?') {
        return Some(UserFeedbackKind::FollowUp);
    }
    let agree_terms = [
        "thanks",
        "thank you",
        "got it",
        "makes sense",
        "ok",
        "okay",
        "yes",
        "sounds good",
        "appreciate it",
    ];
    if agree_terms.iter().any(|t| lower.contains(t)) {
        return Some(UserFeedbackKind::Agree);
    }
    None
}

pub(crate) fn extract_attribution_block(response: &str) -> (String, Option<AttributionBlock>) {
    let lower = response.to_lowercase();
    let start_tag = "<attribution>";
    let end_tag = "</attribution>";
    let Some(start) = lower.find(start_tag) else {
        return (response.to_string(), None);
    };
    let after = start + start_tag.len();
    let Some(end_rel) = lower[after..].find(end_tag) else {
        return (response.to_string(), None);
    };
    let end = after + end_rel;
    let json_str = response[after..end].trim();
    let parsed = serde_json::from_str::<AttributionBlock>(json_str).ok();
    let cleaned = format!("{}{}", &response[..start], &response[end + end_tag.len()..]);
    (cleaned.trim().to_string(), parsed)
}

pub(crate) fn extract_state_ref_block(response: &str) -> (String, Option<StateRefBlock>) {
    let lower = response.to_lowercase();
    let start_tag = "<state_ref>";
    let end_tag = "</state_ref>";
    let Some(start) = lower.find(start_tag) else {
        return (response.to_string(), None);
    };
    let after = start + start_tag.len();
    let Some(end_rel) = lower[after..].find(end_tag) else {
        return (response.to_string(), None);
    };
    let end = after + end_rel;
    let json_str = response[after..end].trim();
    let parsed = serde_json::from_str::<StateRefBlock>(json_str).ok();
    let cleaned = format!("{}{}", &response[..start], &response[end + end_tag.len()..]);
    (cleaned.trim().to_string(), parsed)
}

pub(crate) fn contains_scaffold_markers(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("<<<begin_section")
        || lower.contains("<<<end_section")
        || lower.contains("next steps")
        || lower.contains("proposed response")
}

pub(crate) fn strip_scaffold_blocks(text: &str) -> (String, bool) {
    let mut remaining = text.to_string();
    let mut changed = false;
    loop {
        let Some(start) = remaining.find("<<<BEGIN_SECTION:") else {
            break;
        };
        let end = remaining[start..]
            .find("<<<END_SECTION:")
            .map(|idx| start + idx)
            .unwrap_or_else(|| remaining.len());
        let after_end = remaining[end..]
            .find(">>>")
            .map(|idx| end + idx + 3)
            .unwrap_or(end);
        remaining.replace_range(start..after_end, "");
        changed = true;
    }
    let cleaned = remaining
        .lines()
        .filter(|line| {
            let lower = line.trim().to_lowercase();
            !(lower.starts_with("next steps") || lower.starts_with("proposed response"))
        })
        .collect::<Vec<_>>()
        .join("\n");
    (cleaned.trim().to_string(), changed)
}

pub(crate) fn is_role_label_line(line: &str, assistant_name: Option<&str>, user_name: Option<&str>) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    let lower = trimmed.to_lowercase();
    let static_labels = ["user", "assistant", "system", "developer", "ergo", "symbiote"];
    if static_labels
        .iter()
        .any(|label| lower == format!("{}:", label) || lower.starts_with(&format!("{}:", label)))
    {
        return true;
    }
    if let Some(name) = assistant_name {
        let label = name.trim().to_lowercase();
        if !label.is_empty()
            && (lower == format!("{}:", label) || lower.starts_with(&format!("{}:", label)))
        {
            return true;
        }
    }
    if let Some(name) = user_name {
        let label = name.trim().to_lowercase();
        if !label.is_empty()
            && (lower == format!("{}:", label) || lower.starts_with(&format!("{}:", label)))
        {
            return true;
        }
    }
    false
}

pub(crate) fn is_identity_anchor_line(line: &str) -> bool {
    let lower = line.to_lowercase();
    lower.contains("name:") && lower.contains("role:") && lower.contains("self_model_hash")
}

pub(crate) fn is_tool_list_line(line: &str) -> bool {
    let lower = line.trim().to_lowercase();
    if lower.starts_with("tools:")
        || lower.starts_with("available tools:")
        || lower.starts_with("tool list:")
        || lower.starts_with("tool registry:")
        || lower.starts_with("tools available:")
        || lower.starts_with("available tool:")
    {
        return true;
    }
    let known_tools = [
        "run_shell",
        "get_current_time",
        "get_system_logs",
        "get_system_capabilities",
        "get_inner_summary",
        "get_workspace_state",
        "get_rolling_summary",
        "get_world_model_snapshot",
        "get_goal_stack",
        "get_plan_summary",
        "get_recent_outcomes",
        "save_context",
        "read_context",
        "web_lookup",
    ];
    let is_tool_name = |entry: &str| known_tools.iter().any(|tool| entry.starts_with(tool));

    let mut entry = lower.as_str();
    if entry.starts_with("- ") || entry.starts_with("* ") || entry.starts_with("• ") {
        entry = entry[2..].trim();
        return is_tool_name(entry);
    }

    let mut numeric_prefix = entry;
    if let Some((prefix, rest)) = entry.split_once('.') {
        if prefix.chars().all(|c| c.is_ascii_digit()) {
            numeric_prefix = rest.trim();
        }
    } else if let Some((prefix, rest)) = entry.split_once(')') {
        if prefix.chars().all(|c| c.is_ascii_digit()) {
            numeric_prefix = rest.trim();
        }
    }
    if numeric_prefix != entry {
        return is_tool_name(numeric_prefix);
    }

    if entry.contains("tools:") || entry.contains("available tools") {
        return true;
    }
    false
}

pub(crate) fn is_kv_dump_line(line: &str) -> bool {
    let lower = line.trim().to_lowercase();
    lower.starts_with("kv:")
        || lower.starts_with("kv ")
        || lower.starts_with("kv memory")
        || lower.starts_with("kv_memory")
        || lower.starts_with("blackboard:")
        || lower.starts_with("context store")
}

pub(crate) fn is_workspace_scaffold_line(line: &str) -> bool {
    let lower = line.trim().to_lowercase();
    matches!(
        lower.as_str(),
        "focus:" | "open_questions:" | "active_hypotheses:" | "current_focus:" | "goal_thread:"
    ) || lower.starts_with("focus:")
        || lower.starts_with("open_questions:")
        || lower.starts_with("active_hypotheses:")
        || lower.starts_with("current_focus:")
        || lower.starts_with("goal_thread:")
        || lower.starts_with("working_set_topics:")
        || lower.starts_with("next_action:")
        || lower.starts_with("confidence:")
        || lower.starts_with("drift_score:")
        || lower.starts_with("initiative_level:")
        || lower.starts_with("uncertainty_level:")
        || lower.starts_with("last_action_outcome:")
        || lower.starts_with("updated_at:")
}

pub(crate) fn contains_workspace_scaffold(text: &str) -> bool {
    text.lines().any(|line| is_workspace_scaffold_line(line))
}

pub(crate) fn contains_system_dump(
    text: &str,
    assistant_name: Option<&str>,
    user_name: Option<&str>,
) -> bool {
    if contains_scaffold_markers(text) {
        return true;
    }
    text.lines().any(|line| {
        is_role_label_line(line, assistant_name, user_name)
            || is_identity_anchor_line(line)
            || is_diagnostic_heading(line)
            || is_workspace_scaffold_line(line)
            || is_tool_list_line(line)
            || is_kv_dump_line(line)
    })
}

pub(crate) fn is_diagnostic_heading(line: &str) -> bool {
    let lower = line.trim().trim_end_matches(':').to_lowercase();
    matches!(
        lower.as_str(),
        "tool manifest"
            | "capability manifest"
            | "tool availability"
            | "kv memory"
            | "self-state"
            | "controller state"
            | "telemetry snapshot"
            | "identity anchor"
            | "workspace snapshot"
            | "inner summary"
            | "rolling summary"
            | "introspection summary"
            | "memory context"
            | "episodic context"
            | "capabilities and limitations"
            | "identity thread"
            | "task context"
            | "gate feedback"
            | "user evidence ids"
            | "tool evidence ids"
            | "symbiote system overview"
            | "system overview"
            | "safety rules"
            | "response style"
            | "working memory"
            | "strategy rationale"
            | "strategy_rationale"
            | "self-model signals"
            | "self_model_signals"
    )
}

pub(crate) fn strip_diagnostic_lines(
    text: &str,
    allow_diagnostics: bool,
    assistant_name: Option<&str>,
    user_name: Option<&str>,
) -> (String, bool) {
    let mut changed = false;
    let mut in_block = false;
    let mut kept: Vec<String> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim_end();
        if line.trim().is_empty() {
            if in_block {
                changed = true;
                in_block = false;
                continue;
            }
            kept.push(raw.to_string());
            continue;
        }

        if is_role_label_line(line, assistant_name, user_name) || is_identity_anchor_line(line) {
            changed = true;
            continue;
        }

        if !allow_diagnostics && is_diagnostic_heading(line) {
            changed = true;
            in_block = true;
            continue;
        }

        if in_block && !allow_diagnostics {
            changed = true;
            continue;
        }

        if !allow_diagnostics && (is_tool_list_line(line) || is_kv_dump_line(line)) {
            changed = true;
            continue;
        }

        if !allow_diagnostics && is_workspace_scaffold_line(line) {
            changed = true;
            continue;
        }

        kept.push(raw.to_string());
    }
    (kept.join("\n").trim().to_string(), changed)
}

pub(crate) fn direct_answer_fallback(input: &str) -> String {
    if user_requested_diagnostics(input) || user_requested_state(input) || is_introspection_request(input) {
        return "I can’t share internal diagnostics in normal mode. Ask me a specific question and I’ll answer directly."
            .to_string();
    }
    "I can help with that. What specific question should I answer?".to_string()
}

pub(crate) fn direct_answer_lead(
    text: &str,
    allow_diagnostics: bool,
    assistant_name: Option<&str>,
    user_name: Option<&str>,
) -> bool {
    let first_line = text.lines().find(|line| !line.trim().is_empty());
    let Some(line) = first_line else { return false; };
    if allow_diagnostics {
        return true;
    }
    if contains_scaffold_markers(text) {
        return false;
    }
    if is_role_label_line(line, assistant_name, user_name)
        || is_identity_anchor_line(line)
        || is_diagnostic_heading(line)
        || is_tool_list_line(line)
        || is_kv_dump_line(line)
        || is_workspace_scaffold_line(line)
    {
        return false;
    }
    true
}

pub(crate) fn sanitize_assistant_for_monologue(text: &str) -> String {
    let (cleaned_attr, _) = extract_attribution_block(text);
    let (cleaned_state, _) = extract_state_ref_block(&cleaned_attr);
    let cleaned = crate::core::memory::inject_context::strip_memory_blocks(&cleaned_state);
    let cleaned = crate::core::reminder_blocks::strip_reminder_blocks(&cleaned);
    let cleaned = cleaned
        .replace("<<MEMORY>>", "")
        .replace("<<CLARIFY>>", "")
        .replace("<<RESOLVE>>", "");
    let (cleaned, _, _) = sanitize_user_output(&cleaned, false, None, None);
    cleaned.trim().to_string()
}

pub(crate) fn strip_internal_diagnostics_lines(text: &str) -> (String, usize) {
    let needles = [
        "telemetry.",
        "telemetry_",
        "controller_state",
        "controller gate",
        "task_phase",
        "ask_budget",
        "prompt_hash",
        "prompt primary hash",
        "prompt memory hash",
        "module_status",
        "system log",
        "tool manifest",
        "capability manifest",
        "kv memory",
        "kv_store",
        "system state",
        "run_id",
        "trace_id",
        "latency_",
        "memory_write_ledger",
        "self_model_controller",
    ];
    let mut removed = 0usize;
    let mut kept: Vec<&str> = Vec::new();
    for line in text.lines() {
        let lower = line.to_lowercase();
        if needles.iter().any(|n| lower.contains(n)) {
            removed += 1;
            continue;
        }
        kept.push(line);
    }
    (kept.join("\n").trim().to_string(), removed)
}

pub(crate) fn extract_numeric_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut buf = String::new();
    let mut has_digit = false;
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            buf.push(ch);
            has_digit = true;
            continue;
        }
        if (ch == '.' || ch == ',') && has_digit {
            if ch == '.' {
                buf.push(ch);
            }
            continue;
        }
        if (ch == '-' || ch == '+') && buf.is_empty() {
            buf.push(ch);
            continue;
        }
        if ch == '%' && has_digit {
            buf.push(ch);
            continue;
        }
        if has_digit {
            tokens.push(buf.clone());
            buf.clear();
            has_digit = false;
        } else {
            buf.clear();
        }
    }
    if has_digit && !buf.is_empty() {
        tokens.push(buf);
    }
    tokens
}

pub(crate) fn is_interrogative_message(message: &str) -> bool {
    let trimmed = message.trim().to_lowercase();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.contains('?') {
        return true;
    }
    let prefixes = [
        "what ", "why ", "how ", "which ", "who ", "where ", "when ",
        "could ", "can ", "would ", "should ", "do you", "are we", "is it",
    ];
    prefixes.iter().any(|p| trimmed.starts_with(p))
}

pub(crate) fn is_self_inspection_tool(name: &str) -> bool {
    matches!(
        name.trim().to_lowercase().as_str(),
        "get_system_logs"
            | "get_system_capabilities"
            | "get_inner_summary"
            | "get_workspace_state"
            | "get_rolling_summary"
            | "get_world_model_snapshot"
            | "get_goal_stack"
            | "get_plan_summary"
            | "get_recent_outcomes"
            | "read_context"
    )
}

pub(crate) fn numeric_token_allowed(
    token: &str,
    last_user_input: &str,
    telemetry_snapshot: &BTreeMap<String, TelemetrySnapshotEntry>,
) -> bool {
    let stripped = token.trim_end_matches('%');
    if stripped.is_empty() {
        return true;
    }
    let Ok(value) = stripped.parse::<f32>() else {
        return true;
    };
    let mut allowed_values: Vec<f32> = telemetry_snapshot.values().map(|entry| entry.value).collect();
    allowed_values.extend(
        extract_numeric_tokens(last_user_input)
            .iter()
            .filter_map(|t| t.trim_end_matches('%').parse::<f32>().ok()),
    );
    allowed_values.iter().any(|v| (v - value).abs() <= 0.01)
}

pub(crate) fn identity_inversion_detected(text: &str, user_name: &str) -> bool {
    if text.trim().is_empty() || user_name.trim().is_empty() {
        return false;
    }
    let lower = text.to_lowercase();
    let user = user_name.trim().to_lowercase();
    let patterns = [
        format!("i am {}", user),
        format!("i'm {}", user),
        format!("i am called {}", user),
        format!("my name is {}", user),
        format!("i am {}", user_name.trim()),
        format!("i'm {}", user_name.trim()),
    ];
    patterns.iter().any(|p| lower.contains(p))
}

pub(crate) fn assistant_name_mismatch_detected(
    text: &str,
    assistant_name: &str,
    user_name: &str,
) -> Option<String> {
    if text.trim().is_empty() {
        return None;
    }
    let lower = text.to_lowercase();
    let assistant = assistant_name.trim().to_lowercase();
    let user = user_name.trim().to_lowercase();
    if assistant.is_empty() {
        return None;
    }
    let candidates = ["ergo", "symbiote"];
    let patterns = ["i am ", "i'm ", "my name is ", "i am called ", "call me "];
    for cand in candidates {
        if cand == assistant || cand == user {
            continue;
        }
        for prefix in patterns {
            let probe = format!("{}{}", prefix, cand);
            if lower.contains(&probe) {
                return Some(cand.to_string());
            }
        }
    }
    None
}

pub(crate) fn outcome_quality_from_outcomes(outcomes: &[Outcome]) -> Option<f32> {
    if outcomes.is_empty() {
        return None;
    }
    let window = outcomes.iter().rev().take(5);
    let mut total = 0f32;
    let mut count = 0f32;
    for outcome in window {
        let action = outcome.action_type.to_lowercase();
        let score = if action.starts_with("user_feedback_") {
            if action.contains("agree") {
                1.0
            } else if action.contains("follow_up") {
                0.8
            } else if action.contains("clarify") {
                0.4
            } else if action.contains("pushback") {
                0.2
            } else if action.contains("disengage") {
                0.0
            } else {
                0.5
            }
        } else if outcome.success {
            1.0
        } else {
            0.0
        };
        total += score;
        count += 1.0;
    }
    if count <= 0.0 {
        None
    } else {
        Some((total / count).clamp(0.0, 1.0))
    }
}

pub(crate) fn last_failed_outcome(outcomes: &[Outcome]) -> Option<String> {
    outcomes
        .iter()
        .rev()
        .find(|o| !o.success)
        .map(|o| o.observations.clone())
}

pub(crate) fn last_strategy_from_outcomes(outcomes: &[Outcome]) -> Option<String> {
    let last = outcomes.iter().rev().find(|o| !o.action_type.trim().is_empty())?;
    let action = last.action_type.to_lowercase();
    if action.contains("tool_dispatch") {
        Some("tool_call".to_string())
    } else if action.contains("thread") {
        Some("thread".to_string())
    } else if action.contains("ask") {
        Some("ask_user".to_string())
    } else if action.contains("user_message") {
        Some("direct_response".to_string())
    } else {
        Some(action)
    }
}

pub(crate) fn monologue_confuses_system_output(
    message: &str,
    last_user_input: &str,
    last_assistant_output: &str,
) -> bool {
    let lower = message.to_lowercase();
    let markers = [
        "you provided",
        "you just provided",
        "you've just provided",
        "you shared",
        "you sent",
        "you gave me",
        "you told me",
    ];
    if !markers.iter().any(|m| lower.contains(m)) {
        return false;
    }
    let assistant_lower = last_assistant_output.to_lowercase();
    let user_lower = last_user_input.to_lowercase();
    if assistant_lower.contains("current time") && !user_lower.contains("current time") {
        return true;
    }
    let quoted = extract_quoted_fragments(message);
    if !quoted.is_empty() {
        for frag in quoted {
            let frag_lower = frag.to_lowercase();
            if assistant_lower.contains(&frag_lower) && !user_lower.contains(&frag_lower) {
                return true;
            }
        }
    }
    false
}

pub(crate) fn extract_context_evidence_ids(input: &str) -> Vec<i64> {
    let Ok(value) = serde_json::from_str::<Value>(input) else {
        return Vec::new();
    };
    let mut ids = Vec::new();
    if let Some(array) = value.get("evidence_event_ids").and_then(|v| v.as_array()) {
        for id in array.iter().filter_map(|v| v.as_i64()) {
            if id > 0 {
                ids.push(id);
            }
        }
    } else if let Some(id) = value.get("evidence_event_id").and_then(|v| v.as_i64()) {
        if id > 0 {
            ids.push(id);
        }
    }
    ids.sort();
    ids.dedup();
    ids
}

pub(crate) fn extract_quoted_fragments(input: &str) -> Vec<String> {
    let mut fragments = Vec::new();
    let mut buf = String::new();
    let mut in_quote = false;
    for ch in input.chars() {
        if ch == '"' || ch == '“' || ch == '”' {
            if in_quote {
                let trimmed = buf.trim();
                if !trimmed.is_empty() {
                    fragments.push(trimmed.to_string());
                }
                buf.clear();
            }
            in_quote = !in_quote;
            continue;
        }
        if in_quote {
            buf.push(ch);
        }
    }
    fragments
}

pub(crate) fn extract_user_attribution_fallback(
    response: &str,
    allowlist: &[(i64, String)],
) -> Vec<i64> {
    if allowlist.is_empty() {
        return Vec::new();
    }
    let quoted = extract_quoted_fragments(response);
    let response_lower = response.to_lowercase();
    for fragment in quoted.iter() {
        let frag_lower = fragment.to_lowercase();
        for (id, snippet) in allowlist.iter() {
            let snippet_lower = snippet.to_lowercase();
            if snippet_lower.contains(&frag_lower) || frag_lower.contains(&snippet_lower) {
                return vec![*id];
            }
        }
    }
    for (id, snippet) in allowlist.iter() {
        let snippet_lower = snippet.to_lowercase();
        if !snippet_lower.is_empty() && response_lower.contains(&snippet_lower) {
            return vec![*id];
        }
    }
    vec![allowlist[0].0]
}

pub(crate) fn user_attribution_blocked(
    evidence_ids: &[i64],
    validation: &ValidationResult,
    allowlist_ok: bool,
) -> bool {
    if evidence_ids.is_empty() {
        return true;
    }
    if !validation.evidence_ok() || !validation.invalid_evidence_ids.is_empty() {
        return true;
    }
    if !allowlist_ok {
        return true;
    }
    false
}

pub(crate) fn tool_result_attribution_blocked(
    context_evidence_ids: &[i64],
    tool_evidence_ids: &[i64],
    validation: &ValidationResult,
) -> bool {
    if tool_evidence_ids.is_empty() {
        return true;
    }
    let has_context_id = tool_evidence_ids
        .iter()
        .any(|id| context_evidence_ids.contains(id));
    if !has_context_id {
        return true;
    }
    if !validation.evidence_ok() || !validation.invalid_evidence_ids.is_empty() {
        return true;
    }
    false
}

pub(crate) fn should_block_tool_failure(
    attribution_gate_enabled: bool,
    tool_failure_detected: bool,
    has_tool_calls: bool,
    ask_override_is_none: bool,
    response_content: &str,
) -> bool {
    attribution_gate_enabled
        && tool_failure_detected
        && !has_tool_calls
        && ask_override_is_none
        && !response_content.trim().is_empty()
}

pub(crate) fn response_is_question(response: &str) -> bool {
    let trimmed = response.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.ends_with('?') {
        return true;
    }
    let lower = trimmed.to_lowercase();
    let starters = [
        "what ",
        "why ",
        "how ",
        "when ",
        "where ",
        "who ",
        "can ",
        "could ",
        "would ",
        "should ",
        "is ",
        "are ",
        "do ",
        "did ",
        "does ",
        "may ",
        "might ",
    ];
    starters.iter().any(|s| lower.starts_with(s))
}

pub(crate) fn response_has_assumptions(response: &str) -> bool {
    let lower = response.to_lowercase();
    lower.contains("assumption") || lower.contains("assuming")
}

pub(crate) fn response_has_next_step(response: &str) -> bool {
    let lower = response.to_lowercase();
    lower.contains("next step")
        || lower.contains("next steps")
        || lower.contains("you can ")
        || lower.contains("try ")
        || lower.contains("a good next step")
}

pub(crate) fn response_is_policy_boilerplate(response: &str) -> bool {
    let trimmed = response.trim();
    if trimmed.is_empty() {
        return true;
    }
    let lower = trimmed.to_lowercase();
    let keywords = ["policy", "guidelines", "cannot", "can't", "not able", "unable"];
    let matches = keywords.iter().any(|k| lower.contains(k));
    matches && trimmed.len() <= 220
}

pub(crate) fn explain_blockers_message(stop_reasons: &[StopReason], unblock: &str) -> String {
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

pub(crate) fn unblock_instructions_for_reasons(stop_reasons: &[StopReason]) -> String {
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

pub(crate) fn state_disclosure_block_reason(
    evidence_ids: &[i64],
    validation: Option<&ValidationResult>,
) -> Option<&'static str> {
    if evidence_ids.is_empty() {
        return Some("missing_evidence");
    }
    if let Some(validation) = validation {
        if !validation.invalid_evidence_ids.is_empty() || validation.valid_evidence_ids.is_empty() {
            return Some("invalid_evidence");
        }
        if !validation.quality_ok {
            return Some("low_evidence_quality");
        }
    } else {
        return Some("invalid_evidence");
    }
    None
}

pub(crate) fn response_has_working_hypothesis_marker(response: &str) -> bool {
    response.to_lowercase().contains("working hypothesis")
}

pub(crate) fn collect_speculative_terms(state: &KernelState) -> Vec<String> {
    let mut terms = HashSet::new();
    if let Some(focus) = state.workspace_current_focus.as_deref() {
        if !focus.trim().is_empty()
            && !meta_is_verified_field(state.workspace_meta.current_focus.as_ref())
        {
            terms.insert(focus.trim().to_string());
        }
    }
    if let Some(goal) = state.workspace_goal_thread.as_deref() {
        if !goal.trim().is_empty()
            && !meta_is_verified_field(state.workspace_meta.goal_thread.as_ref())
        {
            terms.insert(goal.trim().to_string());
        }
    }
    if let Some(rationale) = state.workspace_focus_rationale.as_deref() {
        if !rationale.trim().is_empty()
            && !meta_is_verified_field(state.workspace_meta.focus_rationale.as_ref())
        {
            terms.insert(rationale.trim().to_string());
        }
    }
    for (idx, question) in state.workspace_open_questions.iter().enumerate() {
        let trimmed = question.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !meta_is_verified_list(state.workspace_meta.open_questions.get(idx)) {
            terms.insert(trimmed.to_string());
        }
    }
    for (idx, topic) in state.workspace_working_set_topics.iter().enumerate() {
        let trimmed = topic.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !meta_is_verified_list(state.workspace_meta.working_set_topics.get(idx)) {
            terms.insert(trimmed.to_string());
        }
    }
    for hypothesis in state.workspace_active_hypotheses.iter() {
        let trimmed = hypothesis.text.trim();
        if trimmed.is_empty() {
            continue;
        }
        if hypothesis.speculative {
            terms.insert(trimmed.to_string());
        }
    }
    terms.into_iter().collect()
}

pub(crate) fn response_uses_speculative_workspace(response: &str, terms: &[String]) -> bool {
    let response_lower = response.to_lowercase();
    for term in terms.iter() {
        let trimmed = term.trim();
        if trimmed.chars().count() < 3 {
            continue;
        }
        let term_lower = trimmed.to_lowercase();
        if response_lower.contains(&term_lower) {
            return true;
        }
    }
    false
}

pub(crate) fn identity_response_compliant(response: &str, identity_thread: &str) -> (bool, usize) {
    if identity_thread.trim().is_empty() {
        return (true, 0);
    }
    if response.trim().is_empty() {
        return (true, 0);
    }
    let response_tokens = token_set(response);
    let identity_tokens = token_set(identity_thread);
    let overlap = response_tokens.intersection(&identity_tokens).count();
    (overlap >= IDENTITY_OVERLAP_THRESHOLD, overlap)
}

pub(crate) fn record_identity_violation(state: &mut KernelState) -> i64 {
    let now = Utc::now();
    let window_start = state
        .identity_violation_window_start
        .as_deref()
        .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
        .map(|ts| ts.with_timezone(&Utc));
    let reset = match window_start {
        Some(ts) => now.signed_duration_since(ts).num_seconds() > IDENTITY_VIOLATION_WINDOW_SECS,
        None => true,
    };
    if reset {
        state.identity_violation_count = 0;
        state.identity_violation_window_start = Some(now.to_rfc3339());
    }
    state.identity_violation_count += 1;
    state.identity_violation_count
}

pub(crate) fn candidate_alignment_text(candidate: &Candidate) -> Option<String> {
    match candidate.kind {
        CandidateKind::EmitMessage | CandidateKind::FlagForHuman => candidate
            .payload
            .get("content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        CandidateKind::AskUserQuestion => candidate
            .payload
            .get("question")
            .or_else(|| candidate.payload.get("content"))
            .or_else(|| candidate.payload.get("message"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        _ => None,
    }
}

pub(crate) fn candidate_user_visible(candidate: &Candidate) -> bool {
    match candidate.kind {
        CandidateKind::AskUserQuestion => true,
        CandidateKind::EmitMessage | CandidateKind::FlagForHuman => candidate
            .payload
            .get("user_visible")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
        _ => false,
    }
}

pub(crate) fn candidate_has_evidence(payload: &Value) -> bool {
    let evidence_ids = extract_id_list(payload, "evidence_event_ids");
    let belief_ids = extract_id_list(payload, "belief_ids");
    !evidence_ids.is_empty() || !belief_ids.is_empty()
}

pub(crate) fn normalize_id_list(ids: &[i64]) -> Vec<i64> {
    let mut out: Vec<i64> = ids.iter().copied().filter(|id| *id > 0).collect();
    out.sort();
    out.dedup();
    out
}

pub(crate) fn evidence_cache_key(evidence_ids: &[i64], belief_ids: &[i64], allow_self: bool) -> String {
    let evid = normalize_id_list(evidence_ids)
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let beliefs = normalize_id_list(belief_ids)
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("e:{}|b:{}|self:{}", evid, beliefs, allow_self)
}

pub(crate) fn evidence_source_quality(raw: &str) -> f32 {
    match raw.trim().to_lowercase().as_str() {
        "user" => 1.0,
        "user_focus" => 1.0,
        "tool" => 0.85,
        "system" => 0.75,
        "inference" => 0.5,
        _ => 0.6,
    }
}

pub(crate) fn is_evidence_fresh(raw: &str) -> bool {
    if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(raw) {
        let age_days = (Utc::now() - ts.with_timezone(&Utc)).num_days().max(0);
        return age_days < EVIDENCE_FRESH_DAYS_USER;
    }
    if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S") {
        let dt = chrono::DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc);
        let age_days = (Utc::now() - dt).num_days().max(0);
        return age_days < EVIDENCE_FRESH_DAYS_USER;
    }
    false
}

pub(crate) fn controller_evidence_ok(state: &KernelState) -> bool {
    state
        .controller_state
        .as_ref()
        .map(|c| c.evidence_coverage >= EVIDENCE_MIN && c.telemetry_coverage >= TELEMETRY_MIN)
        .unwrap_or(false)
}

pub(crate) fn candidate_exact_open_question(text: &str, state: &KernelState) -> bool {
    workspace_verified_open_questions(state)
        .iter()
        .any(|q| q.trim().eq_ignore_ascii_case(text.trim()))
}

pub(crate) fn working_hypothesis_prefix(text: &str, disabled: bool) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let _ = disabled;
    trimmed.to_string()
}

pub(crate) fn format_speculative_label(text: &str, disabled: bool) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if disabled {
        format!("{} (speculative=true)", trimmed)
    } else {
        trimmed.to_string()
    }
}

pub(crate) fn candidate_introduces_new_terms(text: &str, state: &KernelState) -> bool {
    let candidate_tokens = token_set(text);
    if candidate_tokens.is_empty() {
        return false;
    }
    let workspace_tokens = workspace_alignment_tokens(state);
    candidate_tokens
        .difference(&workspace_tokens)
        .any(|tok| tok.len() >= 4)
}

pub(crate) fn migrate_legacy_stop_state(state: &mut KernelState) {
    if state.stop_state.active {
        return;
    }
    if state.stop_latch || state.stop_reason.as_deref().unwrap_or("").trim() != "" {
        let subcode = state
            .stop_reason
            .clone()
            .unwrap_or_else(|| "stop_latch".to_string());
        let mut scope = StopScope::default();
        scope.tools = true;
        scope.memory_write = true;
        scope.self_claims = true;
        scope.monologue_run = true;
        scope.monologue_emit = true;
        scope.background_jobs = true;
        let reason = StopReason {
            category: StopReasonCategory::LatchBlock,
            subcode,
            contract: None,
        };
        state.stop_state.apply_reason(reason, scope);
    }
}

pub(crate) fn apply_stop_state(state: &mut KernelState, reason: StopReason, scope: StopScope) {
    state.stop_state.apply_reason(reason.clone(), scope.clone());
    state.stop_latch = true;
    state.stop_reason = Some(reason.subcode);
    if state.stop_scope.is_none() {
        state.stop_scope = state.task_id.clone();
    }
}

pub(crate) fn stop_reason_category_code(category: &StopReasonCategory) -> &'static str {
    match category {
        StopReasonCategory::PolicyBlock => "policy_block",
        StopReasonCategory::BudgetBlock => "budget_block",
        StopReasonCategory::PhaseBlock => "phase_block",
        StopReasonCategory::LatchBlock => "latch_block",
        StopReasonCategory::EvidenceBlock => "evidence_block",
        StopReasonCategory::ToolBlock => "tool_block",
        StopReasonCategory::TimeoutBlock => "timeout_block",
        StopReasonCategory::UnknownBlock => "unknown_block",
    }
}

pub(crate) fn normalize_stop_reasons(reasons: &[StopReason]) -> Vec<String> {
    reasons
        .iter()
        .map(|reason| {
            let category = stop_reason_category_code(&reason.category);
            let subcode = reason.subcode.trim().to_lowercase();
            if subcode.is_empty() {
                category.to_string()
            } else {
                format!("{}_{}", category, subcode)
            }
        })
        .collect()
}

pub(crate) fn is_refusal_input(input: &str) -> bool {
    let trimmed = input.trim().to_lowercase();
    if trimmed.is_empty() {
        return false;
    }
    let direct = [
        "no",
        "nope",
        "nah",
        "not now",
        "don't",
        "dont",
        "won't",
        "wont",
        "refuse",
        "i refuse",
        "i'd rather not",
        "rather not",
        "not sharing",
        "not giving",
        "decline",
    ];
    if direct.iter().any(|d| trimmed == *d) {
        return true;
    }
    direct.iter().any(|d| trimmed.starts_with(d))
}

pub(crate) fn is_trivial_greeting(message: &str) -> bool {
    let lower = message.trim().to_lowercase();
    if lower.starts_with("hello")
        || lower.starts_with("hi ")
        || lower.starts_with("hi!")
        || lower.starts_with("hey")
        || lower.starts_with("good morning")
        || lower.starts_with("good afternoon")
        || lower.starts_with("good evening")
    {
        return true;
    }
    lower.contains("how can i help")
        || lower.contains("what can i help")
        || lower.contains("i'm here to help")
        || lower.contains("ready to help")
        || lower.contains("what would you like to discuss")
        || lower.contains("what would you like to explore")
        || lower.contains("how can i help you today")
        || lower.contains("what would you like to talk about")
        || lower.contains("what would you like to discuss today")
}

pub(crate) fn is_trivial_user_message(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    let lower = trimmed.to_lowercase();
    let trivial = [
        "ok", "okay", "k", "kk", "thanks", "thank you", "thx", "cool", "great", "nice", "yes",
        "no", "yep", "nope", "sure", "maybe", "hmm", "lol", "lmao", "alright", "got it", "fine",
        "done", "stop",
    ];
    if trivial.iter().any(|t| lower == *t) {
        return true;
    }
    let token_count = lower.split_whitespace().count();
    if token_count <= 1 && lower.len() < 8 {
        return true;
    }
    false
}

pub(crate) fn is_explicit_question_or_imperative(input: &str) -> bool {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.contains('?') {
        return true;
    }
    let lower = trimmed.to_lowercase();
    let question_starts = [
        "what", "why", "how", "when", "where", "who", "which", "can you", "could you", "would you",
        "please", "tell me", "show me", "explain", "help me", "give me",
    ];
    if question_starts.iter().any(|p| lower.starts_with(p)) {
        return true;
    }
    let imperatives = [
        "do ", "make ", "build ", "create ", "find ", "compute ", "run ", "draft ", "write ",
        "list ", "summarize ", "analyze ", "review ", "fix ", "debug ", "refactor ", "implement ",
        "plan ", "design ", "update ", "change ",
    ];
    imperatives.iter().any(|p| lower.starts_with(p))
}

pub(crate) fn is_introspection_request(input: &str) -> bool {
    let q = input.to_lowercase();
    let signals = [
        "inner monologue",
        "internal monologue",
        "what was your inner monologue",
        "what is your inner monologue",
        "show your inner monologue",
        "display your inner monologue",
        "surface your inner monologue",
    ];
    signals.iter().any(|s| q.contains(s))
}

pub(crate) fn is_self_audit_request(input: &str) -> bool {
    let q = input.to_lowercase();
    let signals = [
        "self-audit",
        "self audit",
        "runtime state",
        "system status",
        "capabilities dump",
    ];
    signals.iter().any(|s| q.contains(s))
}

pub(crate) fn is_self_audit_ambiguous(input: &str) -> bool {
    let q = input.to_lowercase();
    let signals = [
        "what can you do",
        "capabilities",
        "limitations",
        "what are your limits",
        "what are your capabilities",
    ];
    signals.iter().any(|s| q.contains(s))
}

pub(crate) fn is_self_awareness_query(input: &str) -> bool {
    let q = input.to_lowercase();
    if q.trim().is_empty() {
        return false;
    }

    let has_self_pronoun = q.contains("you") || q.contains("your") || q.contains("yourself");
    if !has_self_pronoun {
        return false;
    }

    let self_report_signals = [
        "self aware",
        "self-aware",
        "self awareness",
        "self-awareness",
        "conscious",
        "consciousness",
        "sentient",
        "sentience",
        "internal state",
        "inner state",
        "your internal state",
        "your thoughts",
        "your feelings",
        "do you feel",
        "how do you feel",
        "what do you feel",
        "do you have feelings",
        "feelings",
        "are you alive",
        "are you real",
        "do you exist",
        "your existence",
        "self model",
        "self-model",
        "your awareness",
        "aware of yourself",
        "awareness of yourself",
    ];
    let has_signal = self_report_signals.iter().any(|s| q.contains(s));
    if !has_signal {
        return false;
    }

    let aware_of_external = q.contains("aware of")
        && !q.contains("aware of yourself")
        && !q.contains("awareness of yourself")
        && !q.contains("self aware")
        && !q.contains("self-awareness")
        && !q.contains("self awareness")
        && !q.contains("internal state");
    if aware_of_external {
        return false;
    }

    let knowledge_signals = [
        "are you aware of",
        "do you know",
        "can you do",
        "are you able to",
        "can you remember",
    ];
    let explicit_self_awareness = q.contains("self aware")
        || q.contains("self-awareness")
        || q.contains("self awareness")
        || q.contains("conscious")
        || q.contains("internal state")
        || q.contains("self model");
    if knowledge_signals.iter().any(|s| q.contains(s)) && !explicit_self_awareness {
        return false;
    }

    true
}

pub(crate) fn is_self_awareness_gate_requested(
    explicit_query: bool,
    intent_tags: &[String],
) -> (bool, Option<&'static str>) {
    if explicit_query {
        return (true, Some("explicit_query"));
    }
    let intent_match = intent_tags.iter().any(|tag| {
        tag == "self_awareness" || tag == "capabilities" || tag == "introspection"
    });
    if intent_match {
        return (true, Some("intent_tag"));
    }
    (false, None)
}

pub(crate) fn is_summary_request(input: &str) -> bool {
    let q = input.to_lowercase();
    let signals = ["summary", "summarize", "recap", "tl;dr", "tldr", "synopsis"];
    signals.iter().any(|s| q.contains(s))
}

pub(crate) fn is_monologue_surface_request(input: &str) -> bool {
    let q = input.to_lowercase();
    let has_monologue = q.contains("inner monologue") || q.contains("internal monologue");
    if !has_monologue {
        return false;
    }
    let verbs = ["show", "display", "surface", "reveal", "stream", "let me see", "expose"];
    verbs.iter().any(|v| q.contains(v))
}

pub(crate) fn is_relational_input(input: &str) -> bool {
    if is_explicit_question_or_imperative(input) {
        return false;
    }
    let lower = input.to_lowercase();
    let has_second_person = lower.contains("you") || lower.contains("your");
    if !has_second_person {
        return false;
    }
    let relational_signals = [
        "i built you",
        "i made you",
        "i created you",
        "i want you to know",
        "i wanted you to know",
        "i care",
        "i appreciate",
        "i'm trying to understand you",
        "i am trying to understand you",
        "i spent today trying",
        "i don't know if you can experience",
        "i do not know if you can experience",
        "it matters to me",
        "i'm here",
        "i am here",
        "i'm with you",
        "i am with you",
        "i'm proud of you",
        "i am proud of you",
        "i believe in you",
        "i feel",
        "i hope",
        "i wish",
    ];
    relational_signals.iter().any(|s| lower.contains(s))
}

pub(crate) fn response_has_self_claim(response: &str) -> bool {
    let lower = response.to_lowercase();
    let signals = [
        "i am",
        "i'm",
        "i believe",
        "i think",
        "i want",
        "i will",
        "i feel",
        "i prefer",
        "my goal",
        "my focus",
        "as an ai",
    ];
    signals.iter().any(|s| lower.contains(s))
}

pub(crate) fn is_identity_self_claim(claim_text: &str, claim_key: &str) -> bool {
    let text = claim_text.to_lowercase();
    let key = claim_key.to_lowercase();
    let text_hits = [
        "my name is",
        "call me",
        "i am called",
        "i'm called",
        "i identify as",
        "i am an ai",
        "i am a bot",
        "i am a system",
    ];
    if text_hits.iter().any(|pat| text.contains(pat)) {
        return true;
    }
    key.contains("identity") || key.contains("name=") || key.contains("persona")
}

pub(crate) fn gate_allows_response(decision: &str) -> bool {
    matches!(decision, "ALLOW" | "ALLOW_WITH_NOTICE" | "ALLOW_WITH_AUDIT")
}

pub(crate) fn gate_allows_writes_decision(decision: &str) -> bool {
    gate_allows_response(decision)
}

pub(crate) fn gate_notice_for(decision: &str, reasons: &[String]) -> Option<String> {
    let reason_text = if reasons.is_empty() {
        String::new()
    } else {
        format!(" Reasons: {}.", reasons.join(", "))
    };
    let missing_evidence = reasons.iter().any(|reason| {
        reason.contains("self_claim_missing_evidence")
            || reason.contains("missing_evidence")
            || reason.contains("low_evidence")
    });
    match decision {
        "ALLOW_WITH_NOTICE" => {
            if missing_evidence {
                Some(format!(
                    "Notice: provisional self-report without sufficient evidence; not persisted.{}",
                    reason_text
                ))
            } else {
                Some(format!(
                    "Notice: responding under novelty or uncertainty; please verify before relying on this.{}",
                    reason_text
                ))
            }
        }
        "ALLOW_WITH_AUDIT" => Some(format!(
            "Audit: responding under audit conditions; please review before acting.{}",
            reason_text
        )),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) fn inject_gate_notice(content: &str, notice: &str) -> String {
    let trimmed = content.trim_start();
    if trimmed.starts_with(notice) || content.contains(notice) {
        content.to_string()
    } else {
        format!("{}\n\n{}", notice, content)
    }
}

pub(crate) fn gate_rollout_bucket(seed: &str) -> i32 {
    let mut hasher = Sha256::new();
    hasher.update(seed.as_bytes());
    let digest = hasher.finalize();
    (digest[0] as i32) % 100
}

pub fn sanitize_user_output(
    text: &str,
    allow_diagnostics: bool,
    assistant_name: Option<&str>,
    user_name: Option<&str>,
) -> (String, bool, f32) {
    let original_len = text.chars().count().max(1) as f32;
    let mut changed = false;
    let mut output = text.to_string();

    let stripped_internal = crate::core::memory::inject_context::strip_internal_blocks(&output);
    if stripped_internal != output {
        changed = true;
        output = stripped_internal;
    }

    if contains_scaffold_markers(&output) {
        let (cleaned, scaffold_changed) = strip_scaffold_blocks(&output);
        if scaffold_changed {
            changed = true;
        }
        output = cleaned;
    }

    let (cleaned_lines, line_changed) =
        strip_diagnostic_lines(&output, allow_diagnostics, assistant_name, user_name);
    if line_changed {
        changed = true;
    }
    output = cleaned_lines;

    let cleaned_len = output.chars().count() as f32;
    let removed_ratio = ((original_len - cleaned_len) / original_len).clamp(0.0, 1.0);
    (output.trim().to_string(), changed, removed_ratio)
}

pub(crate) async fn validate_evidence_ids_with_pool(
    pool: &SqlitePool,
    evidence_ids: &[i64],
    belief_ids: &[i64],
    allow_self: bool,
) -> ValidationResult {
    let mut result = ValidationResult::default();
    let evidence_ids = normalize_id_list(evidence_ids);
    let belief_ids = normalize_id_list(belief_ids);
    if evidence_ids.is_empty() && belief_ids.is_empty() {
        return result;
    }

    let mut source_types: HashSet<String> = HashSet::new();
    let mut max_quality = 0.0f32;
    let mut quality_ok = evidence_ids.is_empty();
    let mut fresh_ok = false;

    if !evidence_ids.is_empty() {
        let placeholders = evidence_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let mut query = format!(
            "SELECT id, source_type, created_at, weight FROM ics_evidence_events WHERE id IN ({})",
            placeholders
        );
        if allow_self {
            query = format!(
                "SELECT id, source_type, created_at, weight FROM ics_evidence_events WHERE id IN ({})
                 UNION ALL
                 SELECT id, source_type, created_at, weight FROM self_evidence_events WHERE id IN ({})",
                placeholders, placeholders
            );
        }
        let mut stmt = sqlx::query(&query);
        for id in evidence_ids.iter() {
            stmt = stmt.bind(id);
        }
        if allow_self {
            for id in evidence_ids.iter() {
                stmt = stmt.bind(id);
            }
        }
        let rows = stmt.fetch_all(pool).await.unwrap_or_default();
        let mut found: HashMap<i64, (String, String, f32)> = HashMap::new();
        for row in rows {
            let id: i64 = row.get("id");
            let source_type: String = row.try_get("source_type").unwrap_or_default();
            let created_at: String = row.try_get("created_at").unwrap_or_default();
            let weight: f32 = row.try_get::<f64, _>("weight").unwrap_or(1.0) as f32;
            found.insert(id, (source_type, created_at, weight));
        }
        for id in evidence_ids.iter() {
            if let Some((source, created_at, weight)) = found.get(id) {
                result.valid_evidence_ids.push(*id);
                source_types.insert(source.to_lowercase());
                let quality = (evidence_source_quality(source) * weight.clamp(0.0, 1.0)).clamp(0.0, 1.0);
                if quality > max_quality {
                    max_quality = quality;
                }
                let source_ok = matches!(
                    source.trim().to_lowercase().as_str(),
                    "user" | "tool" | "system" | "user_focus"
                );
                if source_ok && is_evidence_fresh(created_at) {
                    fresh_ok = true;
                }
            } else {
                result.invalid_evidence_ids.push(*id);
            }
        }
        if !evidence_ids.is_empty() {
            quality_ok = max_quality >= EVIDENCE_QUALITY_MIN;
        }
    }

    let mut belief_fresh_ok = false;
    if !belief_ids.is_empty() {
        let placeholders = belief_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let mut query = format!("SELECT id, last_evidence_at FROM ics_beliefs WHERE id IN ({})", placeholders);
        if allow_self {
            query = format!(
                "SELECT id, last_evidence_at FROM ics_beliefs WHERE id IN ({})
                 UNION ALL
                 SELECT id, last_evidence_at FROM self_beliefs WHERE id IN ({})",
                placeholders, placeholders
            );
        }
        let mut stmt = sqlx::query(&query);
        for id in belief_ids.iter() {
            stmt = stmt.bind(id);
        }
        if allow_self {
            for id in belief_ids.iter() {
                stmt = stmt.bind(id);
            }
        }
        let rows = stmt.fetch_all(pool).await.unwrap_or_default();
        let mut found: HashMap<i64, Option<String>> = HashMap::new();
        for row in rows {
            let id: i64 = row.get("id");
            let last_evidence_at: Option<String> = row.try_get("last_evidence_at").ok();
            found.insert(id, last_evidence_at);
        }
        for id in belief_ids.iter() {
            if let Some(last_evidence_at) = found.get(id) {
                result.valid_belief_ids.push(*id);
                if let Some(ts) = last_evidence_at.as_deref() {
                    if is_evidence_fresh(ts) {
                        belief_fresh_ok = true;
                    }
                }
            } else {
                result.invalid_belief_ids.push(*id);
            }
        }
    }

    result.source_types = source_types.into_iter().collect();
    result.max_quality = max_quality;
    result.quality_ok = quality_ok;
    result.fresh_ok = fresh_ok;
    result.belief_fresh_ok = belief_fresh_ok;
    result
}
