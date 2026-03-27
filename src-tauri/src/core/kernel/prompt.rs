use std::collections::HashSet;

use super::*;

#[derive(Clone)]
pub(crate) struct RelevanceAnchors {
    pub last_user_input: String,
    pub inner_focus: String,
    pub pending_questions: String,
    pub recent_outcomes: String,
    pub goal_thread: String,
    pub response_summary: String,
    pub last_user_tokens: HashSet<String>,
    pub inner_focus_tokens: HashSet<String>,
    pub pending_tokens: HashSet<String>,
    pub outcomes_tokens: HashSet<String>,
    pub goal_tokens: HashSet<String>,
    pub response_tokens: HashSet<String>,
}

impl RelevanceAnchors {
    pub(crate) fn new(
        last_user_input: String,
        inner_focus: String,
        pending_questions: String,
        recent_outcomes: String,
        goal_thread: String,
        response_summary: String,
    ) -> Self {
        let last_user_tokens = token_set(&last_user_input);
        let inner_focus_tokens = token_set(&inner_focus);
        let pending_tokens = token_set(&pending_questions);
        let outcomes_tokens = token_set(&recent_outcomes);
        let goal_tokens = token_set(&goal_thread);
        let response_tokens = token_set(&response_summary);
        Self {
            last_user_input,
            inner_focus,
            pending_questions,
            recent_outcomes,
            goal_thread,
            response_summary,
            last_user_tokens,
            inner_focus_tokens,
            pending_tokens,
            outcomes_tokens,
            goal_tokens,
            response_tokens,
        }
    }

    pub(crate) fn anchor_label(&self) -> String {
        if !self.last_user_input.trim().is_empty() && self.last_user_input.trim() != "None" {
            return summarize_snippet(&self.last_user_input, 160);
        }
        if !self.inner_focus.trim().is_empty() && self.inner_focus.trim() != "None" {
            return summarize_snippet(&self.inner_focus, 160);
        }
        if !self.goal_thread.trim().is_empty() && self.goal_thread.trim() != "None" {
            return summarize_snippet(&self.goal_thread, 160);
        }
        if !self.pending_questions.trim().is_empty() && self.pending_questions.trim() != "None" {
            return summarize_snippet(&self.pending_questions, 160);
        }
        if !self.response_summary.trim().is_empty() && self.response_summary.trim() != "None" {
            return summarize_snippet(&self.response_summary, 160);
        }
        if !self.recent_outcomes.trim().is_empty() && self.recent_outcomes.trim() != "None" {
            return summarize_snippet(&self.recent_outcomes, 160);
        }
        "current topic".to_string()
    }
}

pub(crate) fn relevance_score(text: &str, anchors: &RelevanceAnchors) -> f32 {
    let tokens = token_set(text);
    if tokens.is_empty() {
        return 0.0;
    }
    let mut score: f32 = 0.0;
    score += 0.30 * overlap_ratio(&tokens, &anchors.last_user_tokens);
    score += 0.20 * overlap_ratio(&tokens, &anchors.inner_focus_tokens);
    score += 0.15 * overlap_ratio(&tokens, &anchors.goal_tokens);
    score += 0.15 * overlap_ratio(&tokens, &anchors.pending_tokens);
    score += 0.10 * overlap_ratio(&tokens, &anchors.response_tokens);
    score += 0.10 * overlap_ratio(&tokens, &anchors.outcomes_tokens);
    score.min(1.0)
}

pub(crate) fn priority_class_for(kind: &CandidateKind) -> i32 {
    match kind {
        CandidateKind::UpdateGoalThread => 1,
        CandidateKind::UpdateWorkspace => 1,
        CandidateKind::AnchorShift => 1,
        CandidateKind::UpdateInnerSummary => 2,
        CandidateKind::WriteEpisodic => 3,
        CandidateKind::RecordSelfClaim => 3,
        CandidateKind::SpawnThread => 4,
        CandidateKind::ToolCall => 5,
        CandidateKind::EmitMessage | CandidateKind::AskUserQuestion | CandidateKind::FlagForHuman => 6,
        CandidateKind::PromoteSemantic => 7,
        CandidateKind::NoOp => 8,
        CandidateKind::ChangeMode => 0,
        CandidateKind::Terminate => 9,
    }
}

pub(crate) fn strip_focus_rationale_pollution(text: &str) -> String {
    if text.trim().is_empty() {
        return String::new();
    }

    let mut cleaned_lines = Vec::new();
    for line in text.lines() {
        let lower = line.to_lowercase();
        if lower.contains("reason: focus")
            && lower.contains("supported by memory")
        {
            if let Some(idx) = lower.find("reason: focus") {
                let prefix = line[..idx].trim_end();
                if !prefix.is_empty() {
                    cleaned_lines.push(prefix.to_string());
                }
            }
            continue;
        }
        cleaned_lines.push(line.to_string());
    }

    cleaned_lines.join("\n").trim().to_string()
}

pub(crate) fn build_counterfactual_prompt(user_input: &str, candidate: &Candidate) -> String {
    let payload_raw = candidate.payload.to_string();
    let payload_cleaned = strip_focus_rationale_pollution(&payload_raw);
    let payload_snippet = summarize_snippet(&payload_cleaned, 400);
    format!(
        "You are a predictive evaluator. Output JSON only with keys: predicted_label, predicted_outcome.\n\
predicted_label must be one of: agree, followup, clarify, pushback, disengage.\n\n\
User input:\n{}\n\nCandidate kind: {:?}\nCandidate payload:\n{}\n\nPredict likely user feedback and brief outcome.",
        summarize_snippet(user_input, 240),
        candidate.kind,
        payload_snippet
    )
}

pub(crate) fn candidate_relevance_text(candidate: &Candidate) -> String {
    match candidate.kind {
        CandidateKind::EmitMessage => candidate
            .payload
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        CandidateKind::AskUserQuestion => candidate
            .payload
            .get("question")
            .or_else(|| candidate.payload.get("content"))
            .or_else(|| candidate.payload.get("message"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        CandidateKind::ToolCall => {
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
            format!("{} {}", tool_name, args)
        }
        CandidateKind::WriteEpisodic => candidate
            .payload
            .get("event_type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        CandidateKind::UpdateGoalThread => candidate
            .payload
            .get("payload")
            .or_else(|| candidate.payload.get("outcomes"))
            .map(|v| v.to_string())
            .unwrap_or_default(),
        CandidateKind::UpdateInnerSummary => candidate
            .payload
            .get("summary_json")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        CandidateKind::RecordSelfClaim => candidate
            .payload
            .get("claim_text")
            .or_else(|| candidate.payload.get("claim"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        CandidateKind::UpdateWorkspace => {
            let mut parts = Vec::new();
            let focus = candidate
                .payload
                .get("current_focus")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            if !focus.is_empty() {
                parts.push(focus.to_string());
            }
            let goal = candidate
                .payload
                .get("goal_thread")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            if !goal.is_empty() {
                parts.push(goal.to_string());
            }
            if let Some(stack_value) = candidate.payload.get("goal_stack") {
                let items = extract_goal_stack(stack_value);
                if let Some(label) = goal_stack_active_label(&items) {
                    parts.push(label);
                }
            }
            let rationale = candidate
                .payload
                .get("focus_rationale")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            if !rationale.is_empty() {
                parts.push(rationale.to_string());
            }
            if let Some(value) = candidate.payload.get("open_questions") {
                parts.extend(extract_string_list(value));
            }
            if let Some(value) = candidate.payload.get("working_set_topics") {
                parts.extend(extract_string_list(value));
            }
            if let Some(value) = candidate.payload.get("active_hypotheses") {
                let hypotheses = extract_hypotheses(value);
                parts.extend(hypotheses.into_iter().map(|h| h.text));
            }
            parts.join(" ").trim().to_string()
        }
        CandidateKind::AnchorShift => {
            let old_anchor = candidate
                .payload
                .get("old_anchor")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            let new_anchor = candidate
                .payload
                .get("new_anchor")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            format!("{} {}", old_anchor, new_anchor).trim().to_string()
        }
        _ => String::new(),
    }
}

pub(crate) fn candidate_overlaps_last_user_input(candidate: &Candidate, last_user_input: &str) -> bool {
    let last = last_user_input.trim();
    if last.is_empty() || last.eq_ignore_ascii_case("none") {
        return false;
    }
    let candidate_text = candidate_relevance_text(candidate);
    if candidate_text.trim().is_empty() {
        return false;
    }
    let candidate_tokens = token_set(&candidate_text);
    if candidate_tokens.is_empty() {
        return false;
    }
    let user_tokens = token_set(last);
    if user_tokens.is_empty() {
        return false;
    }
    candidate_tokens
        .intersection(&user_tokens)
        .count()
        >= CLARIFIER_OVERLAP_THRESHOLD
}

pub(crate) fn candidate_mentions_last_user_input(candidate: &Candidate, last_user_input: &str) -> bool {
    let last = last_user_input.trim();
    if last.is_empty() || last.eq_ignore_ascii_case("none") {
        return false;
    }
    let candidate_text = candidate_relevance_text(candidate);
    if candidate_text.trim().is_empty() {
        return false;
    }
    let candidate_tokens = token_set(&candidate_text);
    if candidate_tokens.is_empty() {
        return false;
    }
    let user_tokens = token_set(last);
    if user_tokens.is_empty() {
        return false;
    }
    candidate_tokens.intersection(&user_tokens).count() >= 1
}

pub(crate) fn anchor_tokens_from_text(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            buf.push(ch.to_ascii_lowercase());
        } else if !buf.is_empty() {
            out.push(buf.clone());
            buf.clear();
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

pub(crate) fn build_anchor_vocab(
    state: &KernelState,
    tool_names: &[String],
) -> Vec<String> {
    let mut vocab: Vec<String> = Vec::new();
    if let Some(last_input) = state.last_user_input.as_deref() {
        vocab.extend(anchor_tokens_from_text(last_input));
    }
    if let Some(summary) = state.last_response_summary.as_deref() {
        vocab.extend(anchor_tokens_from_text(summary));
    }
    if let Some(goal) = state.workspace_goal_thread.as_deref() {
        vocab.extend(anchor_tokens_from_text(goal));
    }
    for question in state.workspace_open_questions.iter() {
        if !question.trim().is_empty() {
            vocab.extend(anchor_tokens_from_text(question));
        }
    }
    for outcome in state.recent_outcomes.iter().take(5) {
        if !outcome.action_type.trim().is_empty() {
            vocab.extend(anchor_tokens_from_text(&outcome.action_type));
        }
        if !outcome.observations.trim().is_empty() {
            vocab.extend(anchor_tokens_from_text(&outcome.observations));
        }
    }
    if state.user_redirect_turns_remaining <= 0 || state.redirect_focus.is_some() {
        if let Some(focus) = state.workspace_current_focus.as_deref() {
            vocab.extend(anchor_tokens_from_text(focus));
        }
    }
    if let Some(focus) = state.redirect_focus.as_deref() {
        vocab.extend(anchor_tokens_from_text(focus));
    }
    for name in tool_names {
        vocab.push(name.to_lowercase());
    }
    let symbiote_terms = [
        "symbiote",
        "kernel",
        "prompt",
        "prompt_builder",
        "model",
        "model_client",
        "memory",
        "scheduler",
        "monologue",
        "workspace",
        "inner_summary",
    ];
    vocab.extend(symbiote_terms.iter().map(|s| s.to_string()));
    vocab.retain(|t| !t.trim().is_empty());
    vocab.sort();
    vocab.dedup();
    vocab
}

pub(crate) fn count_anchor_hits(message: &str, vocab: &[String]) -> usize {
    if message.trim().is_empty() || vocab.is_empty() {
        return 0;
    }
    let tokens = anchor_tokens_from_text(message);
    let vocab_set: HashSet<&str> = vocab.iter().map(|s| s.as_str()).collect();
    tokens.iter().filter(|t| vocab_set.contains(t.as_str())).count()
}

pub(crate) fn extract_anchor_tokens(message: &str, vocab: &HashSet<String>) -> Vec<String> {
    if message.trim().is_empty() || vocab.is_empty() {
        return Vec::new();
    }
    let mut out: HashSet<String> = HashSet::new();
    for token in anchor_tokens_from_text(message) {
        if vocab.contains(&token) {
            out.insert(token);
        }
    }
    out.into_iter().collect()
}

pub(crate) fn extract_evidence_ids(message: &str) -> Vec<String> {
    let lower = message.to_lowercase();
    if !(lower.contains("evidence")
        || lower.contains("event id")
        || lower.contains("event_id")
        || lower.contains("evidence id")
        || lower.contains("evidence_id"))
    {
        return Vec::new();
    }
    let mut out: HashSet<String> = HashSet::new();
    for part in message.split(|c: char| !c.is_ascii_digit()) {
        if !part.is_empty() {
            out.insert(part.to_string());
        }
    }
    out.into_iter().collect()
}

pub(crate) fn extract_action_phrases(message: &str) -> Vec<String> {
    let mut out: HashSet<String> = HashSet::new();
    let verbs = [
        "action",
        "next",
        "do",
        "implement",
        "fix",
        "update",
        "add",
        "remove",
        "build",
        "tune",
        "verify",
        "investigate",
        "measure",
        "test",
        "run",
        "ship",
        "deploy",
    ];
    for raw in message.lines() {
        let mut line = raw.trim();
        if line.is_empty() {
            continue;
        }
        line = line
            .trim_start_matches('-')
            .trim_start_matches('*')
            .trim_start_matches('•')
            .trim();
        if line.is_empty() {
            continue;
        }
        let lower = line.to_lowercase();
        if verbs.iter().any(|verb| {
            lower == *verb
                || lower.starts_with(&format!("{}:", verb))
                || lower.starts_with(&format!("{} ", verb))
        }) {
            out.insert(lower);
        }
    }
    out.into_iter().collect()
}

pub(crate) fn normalize_question(text: &str) -> String {
    text.trim()
        .trim_end_matches(|c: char| c == '?' || c == '.' || c == '!')
        .to_lowercase()
}

pub(crate) fn normalize_question_key(input: &str) -> String {
    input.trim().to_lowercase()
}

pub(crate) fn open_question_index(state: &KernelState, question: &str) -> Option<usize> {
    let needle = normalize_question_key(question);
    if needle.is_empty() {
        return None;
    }
    state
        .workspace_open_questions
        .iter()
        .position(|q| normalize_question_key(q) == needle)
}

pub(crate) fn context_hash_for_drop(state: &KernelState, fallback: &str) -> String {
    let basis = state
        .last_user_input
        .as_deref()
        .unwrap_or(fallback);
    hash_payload(basis)
}

pub(crate) fn summary_prompt_cap_tokens(settings: &Settings) -> usize {
    let limit = token_estimator::context_limit_tokens(settings);
    let cap = ((limit as f32) * 0.2).floor() as usize;
    cap.min(2000).max(1)
}

pub(crate) fn truncate_tail_to_token_budget(text: &str, max_tokens: usize) -> String {
    if text.trim().is_empty() || max_tokens == 0 {
        return String::new();
    }
    let current_tokens = token_estimator::estimate_tokens_for_strings([text]);
    if current_tokens <= max_tokens {
        return text.to_string();
    }
    let char_count = text.chars().count().max(1);
    let ratio = max_tokens as f32 / current_tokens as f32;
    let keep_chars = ((char_count as f32) * ratio).floor().max(1.0) as usize;
    let skip = char_count.saturating_sub(keep_chars);
    text.chars().skip(skip).collect()
}

pub(crate) fn cap_summary_prompt(system_prompt: &str, user_prompt: &str, settings: &Settings) -> (String, bool) {
    let cap_tokens = summary_prompt_cap_tokens(settings);
    let system_tokens = token_estimator::estimate_tokens_for_strings([system_prompt]);
    let available = cap_tokens.saturating_sub(system_tokens);
    let user_tokens = token_estimator::estimate_tokens_for_strings([user_prompt]);
    if user_tokens <= available {
        return (user_prompt.to_string(), false);
    }
    (truncate_tail_to_token_budget(user_prompt, available), true)
}

pub(crate) fn select_summary_model(settings: &crate::models::Settings) -> (String, String) {
    if let Some(url) = settings
        .summarization_api_url
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        let model = settings
            .summarization_model
            .clone()
            .unwrap_or_else(|| "default".to_string());
        return (model, url.to_string());
    }
    (
        settings
            .active_model_id
            .clone()
            .unwrap_or_else(|| "default".to_string()),
        settings.api_base_url.clone(),
    )
}

pub(crate) fn push_capped(list: &mut Vec<String>, item: String, cap: usize) {
    if item.trim().is_empty() {
        return;
    }
    list.push(item);
    if list.len() > cap {
        let start = list.len() - cap;
        *list = list[start..].to_vec();
    }
}
