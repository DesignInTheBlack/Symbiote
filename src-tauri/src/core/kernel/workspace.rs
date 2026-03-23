use super::*;
use crate::models::GoalStackItem;

#[derive(Debug, Default)]
pub(crate) struct WorkspaceEvidenceFlags {
    pub goal_thread_ok: Option<bool>,
    pub current_focus_ok: Option<bool>,
    pub focus_rationale_ok: Option<bool>,
    pub open_questions_ok: Vec<bool>,
    pub working_set_topics_ok: Vec<bool>,
    pub hypotheses_ok: Vec<bool>,
}

pub(crate) fn update_workspace_payload_has_substantive_fields(payload: &Value) -> bool {
    if let Some(text) = payload.get("goal_thread").and_then(|v| v.as_str()) {
        if !is_none_marker(text) {
            return true;
        }
    }
    if let Some(text) = payload.get("current_focus").and_then(|v| v.as_str()) {
        if !is_none_marker(text) {
            return true;
        }
    }
    if let Some(text) = payload.get("focus_rationale").and_then(|v| v.as_str()) {
        if !is_none_marker(text) {
            return true;
        }
    }
    if let Some(value) = payload.get("open_questions") {
        if !extract_string_list(value).is_empty() {
            return true;
        }
    }
    if let Some(value) = payload.get("working_set_topics") {
        if !extract_string_list(value).is_empty() {
            return true;
        }
    }
    if let Some(value) = payload.get("active_hypotheses") {
        if !extract_hypotheses(value).is_empty() {
            return true;
        }
    }
    if let Some(value) = payload.get("goal_stack") {
        if !extract_goal_stack(value).is_empty() {
            return true;
        }
    }
    false
}

pub(crate) fn update_workspace_payload_has_evidence(payload: &Value) -> bool {
    if !extract_id_list(payload, "evidence_event_ids").is_empty()
        || !extract_id_list(payload, "belief_ids").is_empty()
    {
        return true;
    }
    let Some(hypotheses) = payload.get("active_hypotheses").and_then(|v| v.as_array()) else {
        return false;
    };
    for item in hypotheses.iter() {
        if let Some(obj) = item.as_object() {
            if let Some(array) = obj.get("evidence_event_ids").and_then(|v| v.as_array()) {
                if array.iter().any(|v| v.as_i64().unwrap_or(0) > 0) {
                    return true;
                }
            }
            if let Some(array) = obj.get("belief_ids").and_then(|v| v.as_array()) {
                if array.iter().any(|v| v.as_i64().unwrap_or(0) > 0) {
                    return true;
                }
            }
        }
    }
    if let Some(value) = payload.get("goal_stack") {
        let items = extract_goal_stack(value);
        if goal_stack_has_evidence(&items) {
            return true;
        }
    }
    false
}

#[derive(Clone, Debug)]
pub(crate) struct WorkspaceSnapshot {
    pub goal_thread: Option<String>,
    pub goal_stack: Vec<GoalStackItem>,
    pub open_questions: Vec<String>,
    pub active_hypotheses: Vec<WorkspaceHypothesis>,
    pub working_set_topics: Vec<String>,
    pub current_focus: Option<String>,
    pub focus_rationale: Option<String>,
    pub workspace_meta: crate::models::WorkspaceMeta,
}

impl WorkspaceSnapshot {
    pub(crate) fn from_state(state: &KernelState) -> Self {
        Self {
            goal_thread: state.workspace_goal_thread.clone(),
            goal_stack: state.workspace_goal_stack.clone(),
            open_questions: state.workspace_open_questions.clone(),
            active_hypotheses: state.workspace_active_hypotheses.clone(),
            working_set_topics: state.workspace_working_set_topics.clone(),
            current_focus: state.workspace_current_focus.clone(),
            focus_rationale: state.workspace_focus_rationale.clone(),
            workspace_meta: state.workspace_meta.clone(),
        }
    }
}

pub(crate) fn update_workspace_runtime_meta(
    state: &mut KernelState,
    workspace_state: &core_workspace::WorkspaceState,
    contributors: &crate::models::WorkspaceContributors,
    contributors_summary: &str,
    attention_schema: &crate::models::AttentionSchemaState,
    attention_schema_summary: &str,
) {
    let mut runtime_meta = core_workspace::workspace_state_to_meta(workspace_state);
    if let Some(obj) = runtime_meta.as_object_mut() {
        obj.insert("contributors".to_string(), serde_json::json!(contributors));
        obj.insert(
            "contributors_summary".to_string(),
            serde_json::Value::String(contributors_summary.to_string()),
        );
        obj.insert(
            "attention_schema".to_string(),
            serde_json::json!(attention_schema.clone()),
        );
        obj.insert(
            "attention_schema_summary".to_string(),
            serde_json::Value::String(attention_schema_summary.to_string()),
        );
    }
    state.workspace_meta.runtime = Some(runtime_meta);
}

pub(crate) fn workspace_delta_fields(before: &WorkspaceSnapshot, state: &KernelState) -> Vec<String> {
    let mut fields = Vec::new();
    if before.goal_thread != state.workspace_goal_thread {
        fields.push("goal_thread".to_string());
    }
    if before.goal_stack != state.workspace_goal_stack {
        fields.push("goal_stack".to_string());
    }
    if before.open_questions != state.workspace_open_questions {
        fields.push("open_questions".to_string());
    }
    if before.active_hypotheses != state.workspace_active_hypotheses {
        fields.push("active_hypotheses".to_string());
    }
    if before.working_set_topics != state.workspace_working_set_topics {
        fields.push("working_set_topics".to_string());
    }
    if before.current_focus != state.workspace_current_focus {
        fields.push("current_focus".to_string());
    }
    if before.focus_rationale != state.workspace_focus_rationale {
        fields.push("focus_rationale".to_string());
    }
    let mut before_meta = before.workspace_meta.clone();
    before_meta.runtime = None;
    let mut state_meta = state.workspace_meta.clone();
    state_meta.runtime = None;
    if before_meta != state_meta {
        fields.push("workspace_meta".to_string());
    }
    fields
}
pub(crate) fn is_none_marker(text: &str) -> bool {
    let lower = text.trim().to_lowercase();
    lower.is_empty() || lower == "none" || lower == "null" || lower == "n/a" || lower == "na"
}

pub(crate) fn normalize_workspace_string_field(obj: &mut serde_json::Map<String, Value>, key: &str) {
    let Some(value) = obj.get(key).cloned() else {
        return;
    };
    if value.is_null() {
        obj.remove(key);
        return;
    }
    if let Some(text) = value.as_str() {
        let cleaned = text.trim();
        if cleaned.is_empty() || is_none_marker(cleaned) {
            obj.remove(key);
        } else {
            obj.insert(key.to_string(), Value::String(cleaned.to_string()));
        }
    }
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

fn extract_id_list_value(value: Option<&Value>) -> Vec<i64> {
    value
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

fn normalize_goal_step_value(value: &Value) -> Option<Value> {
    match value {
        Value::String(raw) => {
            let text = raw.trim();
            if text.is_empty() {
                return None;
            }
            Some(json!({ "text": text }))
        }
        Value::Object(obj) => {
            let text = obj
                .get("text")
                .or_else(|| obj.get("step"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            if text.is_empty() {
                return None;
            }
            let mut out = serde_json::Map::new();
            out.insert("text".to_string(), Value::String(text.to_string()));
            if let Some(status) = obj
                .get("status")
                .and_then(|v| v.as_str())
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            {
                out.insert("status".to_string(), Value::String(status.to_string()));
            }
            let evidence_event_ids = extract_id_list_value(obj.get("evidence_event_ids"));
            if !evidence_event_ids.is_empty() {
                out.insert("evidence_event_ids".to_string(), json!(evidence_event_ids));
            }
            let belief_ids = extract_id_list_value(obj.get("belief_ids"));
            if !belief_ids.is_empty() {
                out.insert("belief_ids".to_string(), json!(belief_ids));
            }
            if let Some(completed_at) = obj
                .get("completed_at")
                .and_then(|v| v.as_str())
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            {
                out.insert("completed_at".to_string(), Value::String(completed_at.to_string()));
            }
            Some(Value::Object(out))
        }
        _ => None,
    }
}

pub(crate) fn normalize_goal_stack_payload(value: &mut Value) {
    if !value.is_array() {
        let single = value.clone();
        *value = Value::Array(vec![single]);
    }
    let Some(arr) = value.as_array() else {
        return;
    };
    let mut normalized = Vec::new();
    for item in arr.iter() {
        match item {
            Value::String(raw_goal) => {
                let goal = raw_goal.trim();
                if goal.is_empty() {
                    continue;
                }
                normalized.push(json!({
                    "goal": goal,
                    "steps": [],
                    "current_step_index": 0
                }));
            }
            Value::Object(obj) => {
                let goal = obj
                    .get("goal")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();
                if goal.is_empty() {
                    continue;
                }
                let steps_value = obj.get("steps");
                let mut steps = Vec::new();
                if let Some(Value::Array(step_arr)) = steps_value {
                    for step in step_arr {
                        if let Some(normalized_step) = normalize_goal_step_value(step) {
                            steps.push(normalized_step);
                        }
                    }
                }
                let mut current_step_index = obj
                    .get("current_step_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                if current_step_index > steps.len() {
                    current_step_index = steps.len();
                }
                let status = obj
                    .get("status")
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let evidence_event_ids = extract_id_list_value(obj.get("evidence_event_ids"));
                let belief_ids = extract_id_list_value(obj.get("belief_ids"));
                let updated_at = obj
                    .get("updated_at")
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let mut out = serde_json::Map::new();
                out.insert("goal".to_string(), Value::String(goal.to_string()));
                out.insert("steps".to_string(), Value::Array(steps));
                out.insert(
                    "current_step_index".to_string(),
                    Value::Number(serde_json::Number::from(current_step_index as u64)),
                );
                if let Some(status) = status {
                    out.insert("status".to_string(), Value::String(status));
                }
                if !evidence_event_ids.is_empty() {
                    out.insert("evidence_event_ids".to_string(), json!(evidence_event_ids));
                }
                if !belief_ids.is_empty() {
                    out.insert("belief_ids".to_string(), json!(belief_ids));
                }
                if let Some(updated_at) = updated_at {
                    out.insert("updated_at".to_string(), Value::String(updated_at));
                }
                normalized.push(Value::Object(out));
            }
            _ => {}
        }
    }
    *value = Value::Array(normalized);
}

pub(crate) fn extract_goal_stack(value: &Value) -> Vec<GoalStackItem> {
    let mut normalized = value.clone();
    normalize_goal_stack_payload(&mut normalized);
    serde_json::from_value::<Vec<GoalStackItem>>(normalized).unwrap_or_default()
}

pub(crate) fn extract_goal_stack_from_payload(payload: &Value) -> Vec<GoalStackItem> {
    payload
        .get("goal_stack")
        .map(extract_goal_stack)
        .unwrap_or_default()
}

pub(crate) fn goal_stack_has_evidence(items: &[GoalStackItem]) -> bool {
    items.iter().any(|item| {
        !item.evidence_event_ids.is_empty()
            || !item.belief_ids.is_empty()
            || item
                .steps
                .iter()
                .any(|step| !step.evidence_event_ids.is_empty() || !step.belief_ids.is_empty())
    })
}

pub(crate) fn goal_stack_advances(prev: &[GoalStackItem], next: &[GoalStackItem]) -> bool {
    let max_len = next.len().max(prev.len());
    for idx in 0..max_len {
        let prev_item = prev.get(idx);
        let next_item = next.get(idx);
        let Some(next_item) = next_item else {
            continue;
        };
        let prev_index = prev_item.map(|item| item.current_step_index).unwrap_or(0);
        if next_item.current_step_index > prev_index {
            return true;
        }
        let prev_status = prev_item.as_ref().and_then(|item| item.status.as_deref());
        if status_is_complete(next_item.status.as_deref()) && !status_is_complete(prev_status) {
            return true;
        }
        for (step_idx, step) in next_item.steps.iter().enumerate() {
            let prev_step_status = prev_item
                .and_then(|item| item.steps.get(step_idx))
                .and_then(|step| step.status.as_deref());
            if status_is_complete(step.status.as_deref()) && !status_is_complete(prev_step_status) {
                return true;
            }
        }
    }
    false
}

pub(crate) fn sanitize_goal_stack_advancement(
    prev: &[GoalStackItem],
    next: &mut Vec<GoalStackItem>,
) -> bool {
    let mut changed = false;
    for (idx, next_item) in next.iter_mut().enumerate() {
        let Some(prev_item) = prev.get(idx) else {
            if next_item.current_step_index > 0 {
                next_item.current_step_index = 0;
                changed = true;
            }
            for step in next_item.steps.iter_mut() {
                if status_is_complete(step.status.as_deref()) {
                    step.status = None;
                    step.completed_at = None;
                    changed = true;
                }
            }
            if status_is_complete(next_item.status.as_deref()) {
                next_item.status = None;
                changed = true;
            }
            continue;
        };

        if next_item.current_step_index > prev_item.current_step_index {
            next_item.current_step_index = prev_item.current_step_index;
            changed = true;
        }

        let prev_status = prev_item.status.as_deref();
        if status_is_complete(next_item.status.as_deref()) && !status_is_complete(prev_status) {
            next_item.status = prev_item.status.clone();
            changed = true;
        }

        for (step_idx, step) in next_item.steps.iter_mut().enumerate() {
            let prev_step_status = prev_item
                .steps
                .get(step_idx)
                .and_then(|prev_step| prev_step.status.as_deref());
            if status_is_complete(step.status.as_deref()) && !status_is_complete(prev_step_status) {
                step.status = prev_item
                    .steps
                    .get(step_idx)
                    .and_then(|prev_step| prev_step.status.clone());
                step.completed_at = prev_item
                    .steps
                    .get(step_idx)
                    .and_then(|prev_step| prev_step.completed_at.clone());
                changed = true;
            }
        }
    }
    changed
}

pub(crate) fn merge_goal_stack_evidence(
    items: &mut [GoalStackItem],
    evidence_event_ids: &[i64],
    belief_ids: &[i64],
) {
    if evidence_event_ids.is_empty() && belief_ids.is_empty() {
        return;
    }
    for item in items.iter_mut() {
        let item_has_progress = item.current_step_index > 0
            || status_is_complete(item.status.as_deref())
            || item
                .steps
                .iter()
                .any(|step| status_is_complete(step.status.as_deref()));
        if item_has_progress {
            merge_id_lists(&mut item.evidence_event_ids, evidence_event_ids);
            merge_id_lists(&mut item.belief_ids, belief_ids);
        }
        for step in item.steps.iter_mut() {
            if status_is_complete(step.status.as_deref()) {
                merge_id_lists(&mut step.evidence_event_ids, evidence_event_ids);
                merge_id_lists(&mut step.belief_ids, belief_ids);
            }
        }
    }
}

pub(crate) fn goal_stack_active_label(goal_stack: &[GoalStackItem]) -> Option<String> {
    for item in goal_stack.iter() {
        let goal = item.goal.trim();
        if goal.is_empty() {
            continue;
        }
        if status_is_complete(item.status.as_deref()) {
            continue;
        }
        let step = item
            .steps
            .get(item.current_step_index)
            .map(|s| s.text.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        if let Some(step) = step {
            return Some(format!("{} :: {}", goal, step));
        }
        return Some(goal.to_string());
    }
    None
}

pub(crate) fn apply_goal_loop_tick(goal_stack: &mut Vec<GoalStackItem>, now: &str) -> Option<Value> {
    if goal_stack.is_empty() {
        return None;
    }
    for item in goal_stack.iter_mut() {
        if status_is_complete(item.status.as_deref()) {
            continue;
        }
        let goal_label = item.goal.clone();
        if item.steps.is_empty() {
            let has_evidence = !item.evidence_event_ids.is_empty() || !item.belief_ids.is_empty();
            if has_evidence {
                item.status = Some("completed".to_string());
                item.updated_at = Some(now.to_string());
                return Some(json!({
                    "goal": goal_label,
                    "reason": "goal_evidence",
                    "status": "completed"
                }));
            }
            continue;
        }

        if item.current_step_index >= item.steps.len() {
            let has_evidence = !item.evidence_event_ids.is_empty() || !item.belief_ids.is_empty();
            if has_evidence {
                item.status = Some("completed".to_string());
                item.updated_at = Some(now.to_string());
                return Some(json!({
                    "goal": goal_label,
                    "reason": "goal_evidence",
                    "status": "completed"
                }));
            }
            continue;
        }

        let step_idx = item.current_step_index;
        let steps_len = item.steps.len();
        let (step_complete, step_text) = {
            let step = &mut item.steps[step_idx];
            let has_evidence = !step.evidence_event_ids.is_empty() || !step.belief_ids.is_empty();
            if !has_evidence {
                continue;
            }
            let step_complete = status_is_complete(step.status.as_deref());
            if !step_complete {
                step.status = Some("completed".to_string());
                step.completed_at = Some(now.to_string());
            }
            (step_complete, step.text.clone())
        };
        let prev_index = item.current_step_index;
        item.current_step_index = (item.current_step_index + 1).min(steps_len);
        item.updated_at = Some(now.to_string());
        if item.current_step_index >= steps_len {
            item.status = Some("completed".to_string());
        }
        return Some(json!({
            "goal": goal_label,
            "step": step_text,
            "previous_index": prev_index,
            "new_index": item.current_step_index,
            "reason": if step_complete { "advance_completed_step" } else { "step_completed" }
        }));
    }
    None
}

pub(crate) fn meta_is_verified_field(meta: Option<&WorkspaceFieldMeta>) -> bool {
    meta.map(|m| !m.speculative && ( !m.evidence_event_ids.is_empty() || !m.belief_ids.is_empty()))
        .unwrap_or(false)
}

pub(crate) fn meta_is_verified_list(meta: Option<&WorkspaceListItemMeta>) -> bool {
    meta.map(|m| !m.speculative && (!m.evidence_event_ids.is_empty() || !m.belief_ids.is_empty()))
        .unwrap_or(false)
}

pub(crate) fn hypothesis_is_verified(hypothesis: &WorkspaceHypothesis) -> bool {
    !hypothesis.speculative
        && (!hypothesis.evidence_event_ids.is_empty() || !hypothesis.belief_ids.is_empty())
}

pub(crate) fn workspace_verified_focus(state: &KernelState) -> Option<String> {
    if let Some(focus) = state.workspace_current_focus.as_deref() {
        let trimmed = focus.trim();
        if !trimmed.is_empty() && meta_is_verified_field(state.workspace_meta.current_focus.as_ref()) {
            return Some(trimmed.to_string());
        }
    }
    if let Some(goal) = state.workspace_goal_thread.as_deref() {
        let trimmed = goal.trim();
        if !trimmed.is_empty() && meta_is_verified_field(state.workspace_meta.goal_thread.as_ref()) {
            return Some(trimmed.to_string());
        }
    }
    for (idx, topic) in state.workspace_working_set_topics.iter().enumerate() {
        let trimmed = topic.trim();
        if trimmed.is_empty() {
            continue;
        }
        if meta_is_verified_list(state.workspace_meta.working_set_topics.get(idx)) {
            return Some(trimmed.to_string());
        }
    }
    None
}

pub(crate) fn workspace_verified_topics(state: &KernelState) -> Vec<String> {
    state
        .workspace_working_set_topics
        .iter()
        .enumerate()
        .filter_map(|(idx, topic)| {
            let trimmed = topic.trim();
            if trimmed.is_empty() {
                return None;
            }
            if meta_is_verified_list(state.workspace_meta.working_set_topics.get(idx)) {
                Some(trimmed.to_string())
            } else {
                None
            }
        })
        .collect()
}

pub(crate) fn workspace_verified_open_questions(state: &KernelState) -> Vec<String> {
    state
        .workspace_open_questions
        .iter()
        .enumerate()
        .filter_map(|(idx, question)| {
            let trimmed = question.trim();
            if trimmed.is_empty() {
                return None;
            }
            if meta_is_verified_list(state.workspace_meta.open_questions.get(idx)) {
                Some(trimmed.to_string())
            } else {
                None
            }
        })
        .collect()
}

pub(crate) fn workspace_has_verified_anchor(state: &KernelState) -> bool {
    workspace_verified_focus(state).is_some()
        || !workspace_verified_topics(state).is_empty()
        || !workspace_verified_open_questions(state).is_empty()
}

pub(crate) fn apply_workspace_demotions_from_flags(
    state: &mut KernelState,
    focus_grace_elapsed: bool,
    flags: &WorkspaceEvidenceFlags,
) -> Vec<String> {
    let mut demoted_fields: Vec<String> = Vec::new();
    let mut demote_field = |label: &str, meta: &mut WorkspaceFieldMeta, guard_focus: bool| {
        if guard_focus && !focus_grace_elapsed {
            return;
        }
        meta.speculative = true;
        demoted_fields.push(label.to_string());
    };

    if let Some(meta) = state.workspace_meta.goal_thread.as_mut() {
        if !meta.speculative && flags.goal_thread_ok == Some(false) {
            demote_field("goal_thread", meta, false);
        }
    }
    if let Some(meta) = state.workspace_meta.current_focus.as_mut() {
        if !meta.speculative && flags.current_focus_ok == Some(false) {
            demote_field("current_focus", meta, true);
        }
    }
    if let Some(meta) = state.workspace_meta.focus_rationale.as_mut() {
        if !meta.speculative && flags.focus_rationale_ok == Some(false) {
            demote_field("focus_rationale", meta, true);
        }
    }

    for (idx, meta) in state.workspace_meta.open_questions.iter_mut().enumerate() {
        if meta.speculative {
            continue;
        }
        let ok = flags.open_questions_ok.get(idx).copied().unwrap_or(true);
        if !ok {
            meta.speculative = true;
            demoted_fields.push(format!("open_question:{}", meta.text));
        }
    }

    for (idx, meta) in state.workspace_meta.working_set_topics.iter_mut().enumerate() {
        if meta.speculative {
            continue;
        }
        let ok = flags.working_set_topics_ok.get(idx).copied().unwrap_or(true);
        if !ok {
            meta.speculative = true;
            demoted_fields.push(format!("working_set:{}", meta.text));
        }
    }

    for (idx, hypothesis) in state.workspace_active_hypotheses.iter_mut().enumerate() {
        if hypothesis.speculative {
            continue;
        }
        let ok = flags.hypotheses_ok.get(idx).copied().unwrap_or(true);
        if !ok {
            hypothesis.speculative = true;
            demoted_fields.push(format!("hypothesis:{}", hypothesis.text));
        }
    }
    state.workspace_meta.active_hypotheses = state.workspace_active_hypotheses.clone();

    demoted_fields
}

pub(crate) fn workspace_meta_counts(state: &KernelState) -> (usize, usize) {
    let mut verified = 0usize;
    let mut speculative = 0usize;

    if let Some(focus) = state.workspace_current_focus.as_deref() {
        if !focus.trim().is_empty() {
            if meta_is_verified_field(state.workspace_meta.current_focus.as_ref()) {
                verified += 1;
            } else {
                speculative += 1;
            }
        }
    }
    if let Some(goal) = state.workspace_goal_thread.as_deref() {
        if !goal.trim().is_empty() {
            if meta_is_verified_field(state.workspace_meta.goal_thread.as_ref()) {
                verified += 1;
            } else {
                speculative += 1;
            }
        }
    }
    if let Some(rationale) = state.workspace_focus_rationale.as_deref() {
        if !rationale.trim().is_empty() {
            if meta_is_verified_field(state.workspace_meta.focus_rationale.as_ref()) {
                verified += 1;
            } else {
                speculative += 1;
            }
        }
    }
    for (idx, topic) in state.workspace_working_set_topics.iter().enumerate() {
        if topic.trim().is_empty() {
            continue;
        }
        if meta_is_verified_list(state.workspace_meta.working_set_topics.get(idx)) {
            verified += 1;
        } else {
            speculative += 1;
        }
    }
    for (idx, question) in state.workspace_open_questions.iter().enumerate() {
        if question.trim().is_empty() {
            continue;
        }
        if meta_is_verified_list(state.workspace_meta.open_questions.get(idx)) {
            verified += 1;
        } else {
            speculative += 1;
        }
    }
    for hypothesis in state.workspace_active_hypotheses.iter() {
        if hypothesis.text.trim().is_empty() {
            continue;
        }
        if hypothesis_is_verified(hypothesis) {
            verified += 1;
        } else {
            speculative += 1;
        }
    }

    (verified, speculative)
}

pub(crate) fn workspace_required(state: &KernelState) -> bool {
    if workspace_verified_focus(state).is_some() {
        return true;
    }
    !workspace_verified_topics(state).is_empty()
}

pub(crate) fn workspace_focus_label(state: &KernelState) -> Option<String> {
    workspace_verified_focus(state)
}

pub(crate) fn workspace_rationale_label(state: &KernelState) -> Option<String> {
    if let Some(reason) = state.workspace_focus_rationale.as_deref() {
        let trimmed = reason.trim();
        if !trimmed.is_empty() && meta_is_verified_field(state.workspace_meta.focus_rationale.as_ref()) {
            return Some(trimmed.to_string());
        }
    }
    if meta_is_verified_field(state.workspace_meta.goal_thread.as_ref()) {
        Some("fallback to goal_thread".to_string())
    } else {
        None
    }
}

pub(crate) fn is_user_focus_label(focus: &str, user_name: &str) -> bool {
    let focus_lower = focus.trim().to_lowercase();
    if focus_lower.is_empty() {
        return false;
    }
    if !user_name.trim().is_empty() && focus_lower == user_name.trim().to_lowercase() {
        return true;
    }
    matches!(focus_lower.as_str(), "user" | "me" | "you")
}

pub(crate) fn user_focus_signal(input: &str, focus: &str) -> bool {
    let input_lower = input.to_lowercase();
    let focus_lower = focus.trim().to_lowercase();
    if focus_lower.is_empty() {
        return false;
    }
    let focus_markers = [
        "focus:",
        "focus on",
        "focus is",
        "set focus",
        "current focus",
        "let's focus",
        "primary focus",
    ];
    let has_marker = focus_markers.iter().any(|m| input_lower.contains(m));
    if !has_marker {
        return false;
    }
    input_lower.contains(&focus_lower)
}

pub(crate) fn focus_shift_candidate(
    current_focus: &str,
    last_user_input: &str,
    user_name: &str,
    hypotheses: &[WorkspaceHypothesis],
    goal_thread: Option<&str>,
) -> Option<(String, String)> {
    if !is_user_focus_label(current_focus, user_name) {
        return None;
    }
    if last_user_input.trim().is_empty() {
        return None;
    }
    if is_relational_input(last_user_input) {
        return None;
    }
    let lower_input = last_user_input.to_lowercase();
    if !user_name.trim().is_empty() && lower_input.contains(&user_name.trim().to_lowercase()) {
        return None;
    }
    let hypothesis = hypotheses.iter().find_map(|h| {
        let text = h.text.trim();
        if text.is_empty() {
            None
        } else {
            Some(text.to_string())
        }
    });
    if let Some(text) = hypothesis {
        return Some((text, "shift from user focus to hypothesis".to_string()));
    }
    if let Some(goal) = goal_thread {
        let trimmed = goal.trim();
        if !trimmed.is_empty() {
            return Some((trimmed.to_string(), "shift from user focus to goal_thread".to_string()));
        }
    }
    None
}

pub(crate) fn workspace_ack_block(state: &KernelState) -> String {
    let focus = workspace_focus_label(state).unwrap_or_else(|| "unspecified".to_string());
    let rationale = workspace_rationale_label(state).unwrap_or_else(|| "no rationale provided".to_string());
    format!("Current focus: {}.\nReason: {}.", focus, rationale)
}

#[allow(dead_code)]
pub(crate) fn append_workspace_ack(content: &str, state: &KernelState) -> String {
    let block = workspace_ack_block(state);
    if content.trim().is_empty() {
        block
    } else {
        let mut body_lines: Vec<&str> = Vec::new();
        let mut tag_lines: Vec<&str> = Vec::new();
        for line in content.lines() {
            let lower = line.trim().to_ascii_lowercase();
            if matches!(
                lower.as_str(),
                "<<memory>>" | "<<clarify>>" | "<<resolve>>"
            ) {
                tag_lines.push(line);
            } else {
                body_lines.push(line);
            }
        }
        while matches!(body_lines.last(), Some(line) if {
            let trimmed = line.trim().to_ascii_lowercase();
            trimmed.starts_with("confidence:") || trimmed == "low evidence."
        }) {
            body_lines.pop();
        }
        while matches!(body_lines.last(), Some(line) if line.trim().is_empty()) {
            body_lines.pop();
        }
        let mut out = if body_lines.is_empty() {
            block.clone()
        } else {
            format!("{}\n\n{}", body_lines.join("\n").trim_end(), block)
        };
        if !tag_lines.is_empty() {
            out = format!("{}\n{}", out.trim_end(), tag_lines.join("\n"));
        }
        out
    }
}

pub(crate) fn workspace_policy_addendum(state: &KernelState) -> String {
    let mut note = String::from("Workspace binding required: ");
    if let Some(focus) = workspace_focus_label(state) {
        note.push_str(&format!("Current focus: {}. ", focus));
    }
    let topics = workspace_verified_topics(state);
    if !topics.is_empty() {
        note.push_str(&format!("Working set topics: {}. ", topics.join(", ")));
    }
    note.push_str(
        "Your response must explicitly reference the current focus or a working set topic, or state why the workspace is not relevant.",
    );
    note
}

pub(crate) fn response_mentions_workspace_term(response: &str, term: &str) -> bool {
    let needle = term.trim().to_lowercase();
    if needle.is_empty() {
        return false;
    }
    let haystack = response.to_lowercase();
    if needle.len() <= 3 {
        return haystack
            .split(|c: char| !c.is_alphanumeric())
            .any(|tok| tok == needle);
    }
    haystack.contains(&needle)
}

pub(crate) fn response_mentions_workspace(response: &str, state: &KernelState) -> bool {
    if response.trim().is_empty() {
        return false;
    }
    if let Some(focus) = workspace_verified_focus(state) {
        if response_mentions_workspace_term(response, &focus) {
            return true;
        }
    }
    for topic in workspace_verified_topics(state) {
        if response_mentions_workspace_term(response, &topic) {
            return true;
        }
    }
    false
}

pub(crate) fn response_has_workspace_exception(response: &str) -> bool {
    let lower = response.to_lowercase();
    let relevance_terms = [
        "not relevant",
        "not applicable",
        "not related",
        "does not relate",
        "doesn't relate",
        "outside the current focus",
        "outside current focus",
        "outside the workspace",
        "unrelated to",
        "no direct relation",
        "not connected",
        "does not apply",
    ];
    let workspace_terms = ["workspace", "current focus", "working set", "focus"];
    let mentions_workspace = workspace_terms.iter().any(|term| lower.contains(term));
    if !mentions_workspace {
        return false;
    }
    relevance_terms.iter().any(|term| lower.contains(term))
}

pub(crate) fn workspace_response_compliant(response: &str, state: &KernelState) -> (bool, bool) {
    let mentions = response_mentions_workspace(response, state);
    if mentions {
        return (true, false);
    }
    let exception = response_has_workspace_exception(response);
    (false, exception)
}

pub(crate) fn normalize_json_list(obj: &mut serde_json::Map<String, Value>, key: &str) {
    if let Some(value) = obj.get(key) {
        if value.is_array() {
            return;
        }
    }
    if let Some(value) = obj.get(key).and_then(|v| v.as_str()) {
        let cleaned = value.trim();
        if cleaned.is_empty() {
            obj.remove(key);
        } else {
            obj.insert(key.to_string(), json!([cleaned]));
        }
        return;
    }
    if let Some(value) = obj.get(key) {
        if value.is_null() {
            obj.remove(key);
        }
    }
}

pub(crate) fn normalize_hypotheses_list(obj: &mut serde_json::Map<String, Value>) {
    let Some(value) = obj.get("active_hypotheses").cloned() else {
        return;
    };
    if value.is_array() {
        return;
    }
    if value.is_object() {
        obj.insert("active_hypotheses".to_string(), Value::Array(vec![value]));
        return;
    }
    if let Some(text) = value.as_str() {
        let cleaned = text.trim();
        if cleaned.is_empty() {
            obj.remove("active_hypotheses");
        } else {
            obj.insert("active_hypotheses".to_string(), json!([cleaned]));
        }
        return;
    }
    if value.is_null() {
        obj.remove("active_hypotheses");
    }
}

pub(crate) fn extract_string_list(value: &Value) -> Vec<String> {
    if let Some(arr) = value.as_array() {
        return arr
            .iter()
            .filter_map(|item| item.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }
    if let Some(text) = value.as_str() {
        let cleaned = text.trim();
        if cleaned.is_empty() {
            Vec::new()
        } else {
            vec![cleaned.to_string()]
        }
    } else {
        Vec::new()
    }
}

pub(crate) fn clamp_confidence(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.7
    }
}

pub(crate) fn extract_hypotheses(value: &Value) -> Vec<WorkspaceHypothesis> {
    if let Some(arr) = value.as_array() {
        let mut out = Vec::new();
        for item in arr {
            if let Some(text) = item.as_str() {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    continue;
                }
                out.push(WorkspaceHypothesis {
                    text: trimmed.to_string(),
                    confidence: 0.7,
                    speculative: false,
                    evidence_event_ids: Vec::new(),
                    belief_ids: Vec::new(),
                });
                continue;
            }
            if let Some(obj) = item.as_object() {
                let text = obj
                    .get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();
                if text.is_empty() {
                    continue;
                }
                let confidence = obj
                    .get("confidence")
                    .and_then(|v| v.as_f64())
                    .map(|v| clamp_confidence(v as f32))
                    .unwrap_or(0.7);
                let speculative = obj
                    .get("speculative")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let evidence_event_ids = obj
                    .get("evidence_event_ids")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|item| item.as_i64().or_else(|| item.as_str().and_then(|s| s.parse::<i64>().ok())))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let belief_ids = obj
                    .get("belief_ids")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|item| item.as_i64().or_else(|| item.as_str().and_then(|s| s.parse::<i64>().ok())))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                out.push(WorkspaceHypothesis {
                    text,
                    confidence,
                    speculative,
                    evidence_event_ids,
                    belief_ids,
                });
            }
        }
        return out;
    }
    if let Some(text) = value.as_str() {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return vec![WorkspaceHypothesis {
                text: trimmed.to_string(),
                confidence: 0.7,
                speculative: false,
                evidence_event_ids: Vec::new(),
                belief_ids: Vec::new(),
            }];
        }
    }
    Vec::new()
}

pub(crate) fn hypothesis_texts(hypotheses: &[WorkspaceHypothesis]) -> Vec<String> {
    hypotheses
        .iter()
        .filter_map(|h| {
            let text = h.text.trim();
            if text.is_empty() {
                None
            } else {
                Some(text.to_string())
            }
        })
        .collect()
}

pub(crate) fn hypothesis_exists(hypotheses: &[WorkspaceHypothesis], text: &str) -> bool {
    let needle = text.trim().to_lowercase();
    if needle.is_empty() {
        return true;
    }
    hypotheses.iter().any(|h| h.text.trim().to_lowercase() == needle)
}

pub(crate) fn merge_id_lists(target: &mut Vec<i64>, incoming: &[i64]) {
    for id in incoming {
        if !target.contains(id) {
            target.push(*id);
        }
    }
}

pub(crate) fn make_field_meta(speculative: bool, evidence_event_ids: &[i64], belief_ids: &[i64]) -> WorkspaceFieldMeta {
    WorkspaceFieldMeta {
        speculative,
        evidence_event_ids: evidence_event_ids.to_vec(),
        belief_ids: belief_ids.to_vec(),
    }
}

pub(crate) fn make_list_meta(text: &str, speculative: bool, evidence_event_ids: &[i64], belief_ids: &[i64]) -> WorkspaceListItemMeta {
    WorkspaceListItemMeta {
        text: text.to_string(),
        speculative,
        evidence_event_ids: evidence_event_ids.to_vec(),
        belief_ids: belief_ids.to_vec(),
        attempt_count: 0,
        last_asked_at: None,
        expires_at: None,
    }
}

pub(crate) fn ensure_workspace_meta_alignment(state: &mut KernelState) {
    let meta = &mut state.workspace_meta;
    if state.workspace_goal_thread.is_none() {
        meta.goal_thread = None;
    } else if meta.goal_thread.is_none() {
        meta.goal_thread = Some(make_field_meta(true, &[], &[]));
    }
    if state.workspace_current_focus.is_none() {
        meta.current_focus = None;
    } else if meta.current_focus.is_none() {
        meta.current_focus = Some(make_field_meta(true, &[], &[]));
    }
    if state.workspace_focus_rationale.is_none() {
        meta.focus_rationale = None;
    } else if meta.focus_rationale.is_none() {
        meta.focus_rationale = Some(make_field_meta(true, &[], &[]));
    }
    // Open questions
    if meta.open_questions.len() > state.workspace_open_questions.len() {
        meta.open_questions.truncate(state.workspace_open_questions.len());
    }
    for (idx, question) in state.workspace_open_questions.iter().enumerate() {
        if idx >= meta.open_questions.len() {
            meta.open_questions.push(make_list_meta(question, true, &[], &[]));
        } else {
            meta.open_questions[idx].text = question.clone();
        }
    }
    // Working set topics
    if meta.working_set_topics.len() > state.workspace_working_set_topics.len() {
        meta.working_set_topics.truncate(state.workspace_working_set_topics.len());
    }
    for (idx, topic) in state.workspace_working_set_topics.iter().enumerate() {
        if idx >= meta.working_set_topics.len() {
            meta.working_set_topics.push(make_list_meta(topic, true, &[], &[]));
        } else {
            meta.working_set_topics[idx].text = topic.clone();
        }
    }
    meta.active_hypotheses = state.workspace_active_hypotheses.clone();
}

pub(crate) fn normalize_hypotheses_payload(
    value: &mut Value,
    speculative: bool,
    evidence_event_ids: &[i64],
    belief_ids: &[i64],
) {
    let mut hypotheses: Vec<WorkspaceHypothesis> = if let Ok(list) = serde_json::from_value::<Vec<WorkspaceHypothesis>>(value.clone()) {
        list
    } else if let Ok(list) = serde_json::from_value::<Vec<String>>(value.clone()) {
        list.into_iter()
            .filter_map(|text| {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(WorkspaceHypothesis {
                        text: trimmed.to_string(),
                        confidence: 0.7,
                        speculative: false,
                        evidence_event_ids: Vec::new(),
                        belief_ids: Vec::new(),
                    })
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    for hypothesis in hypotheses.iter_mut() {
        if speculative {
            hypothesis.speculative = true;
        }
        if !evidence_event_ids.is_empty() {
            merge_id_lists(&mut hypothesis.evidence_event_ids, evidence_event_ids);
        }
        if !belief_ids.is_empty() {
            merge_id_lists(&mut hypothesis.belief_ids, belief_ids);
        }
    }

    *value = json!(hypotheses);
}

pub(crate) fn update_workspace_meta_from_payload(
    state: &mut KernelState,
    payload: &Value,
    speculative: bool,
    evidence_event_ids: &[i64],
    belief_ids: &[i64],
) {
    if payload.get("goal_thread").and_then(|v| v.as_str()).is_some() {
        state.workspace_meta.goal_thread = Some(make_field_meta(speculative, evidence_event_ids, belief_ids));
    }
    if payload.get("current_focus").and_then(|v| v.as_str()).is_some() {
        state.workspace_meta.current_focus = Some(make_field_meta(speculative, evidence_event_ids, belief_ids));
    }
    if payload.get("focus_rationale").and_then(|v| v.as_str()).is_some() {
        state.workspace_meta.focus_rationale = Some(make_field_meta(speculative, evidence_event_ids, belief_ids));
    }
    if let Some(open_questions) = payload.get("open_questions") {
        let mut next = Vec::new();
        if let Some(arr) = open_questions.as_array() {
            for item in arr {
                if let Some(text) = item.as_str() {
                    let trimmed = text.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    next.push(make_list_meta(trimmed, speculative, evidence_event_ids, belief_ids));
                }
            }
        }
        if !next.is_empty() {
            state.workspace_meta.open_questions = next;
        }
    }
    if let Some(topics) = payload.get("working_set_topics") {
        let mut next = Vec::new();
        if let Some(arr) = topics.as_array() {
            for item in arr {
                if let Some(text) = item.as_str() {
                    let trimmed = text.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    next.push(make_list_meta(trimmed, speculative, evidence_event_ids, belief_ids));
                }
            }
        }
        if !next.is_empty() {
            state.workspace_meta.working_set_topics = next;
        }
    }
}

pub(crate) fn structured_framework_detected(text: &str) -> bool {
    let lower = text.to_lowercase();
    let step_markers = [
        "step 1",
        "step one",
        "step 2",
        "step two",
        "step 3",
        "step three",
        "first",
        "second",
        "third",
    ];
    let mut hits = 0usize;
    for marker in step_markers.iter() {
        if lower.contains(marker) {
            hits += 1;
        }
    }
    if hits >= 3 {
        return true;
    }
    let segments = text
        .split(|c| c == '\n' || c == ';')
        .filter(|s| !s.trim().is_empty())
        .count();
    if segments >= 3 {
        return true;
    }
    let keywords = ["framework", "model", "architecture", "pipeline"];
    if keywords.iter().any(|k| lower.contains(k)) {
        let clause_count = text.matches(',').count()
            + text.matches(';').count()
            + text.matches(':').count();
        if clause_count >= 2 {
            return true;
        }
    }
    false
}

pub(crate) fn extract_structured_framework_text(monologue: &MonologueOutput) -> Option<String> {
    for message in monologue.dialogue_messages.iter().rev() {
        if structured_framework_detected(message) {
            return Some(message.clone());
        }
    }
    for turn in monologue.turns.iter().rev() {
        for candidate in turn.candidates.iter().rev() {
            if !matches!(
                candidate.kind,
                CandidateKind::EmitMessage | CandidateKind::AskUserQuestion | CandidateKind::FlagForHuman
            ) {
                continue;
            }
            if let Some(text) = candidate_alignment_text(candidate) {
                if structured_framework_detected(&text) {
                    return Some(text);
                }
            }
        }
    }
    None
}

pub(crate) fn apply_workspace_update(state: &mut KernelState, payload: &Value) -> bool {
    let mut changed = false;
    if let Some(goal) = payload.get("goal_thread").and_then(|v| v.as_str()) {
        let cleaned = goal.trim();
        let next = if cleaned.is_empty() { None } else { Some(cleaned.to_string()) };
        if state.workspace_goal_thread != next {
            state.workspace_goal_thread = next;
            changed = true;
        }
    }
    if let Some(value) = payload.get("open_questions") {
        let list = extract_string_list(value);
        if state.workspace_open_questions != list {
            state.workspace_open_questions = list;
            changed = true;
        }
    }
    if let Some(value) = payload.get("active_hypotheses") {
        let list = extract_hypotheses(value);
        if state.workspace_active_hypotheses != list {
            state.workspace_active_hypotheses = list;
            changed = true;
        }
    }
    if let Some(value) = payload.get("working_set_topics") {
        let list = extract_string_list(value);
        if state.workspace_working_set_topics != list {
            state.workspace_working_set_topics = list;
            changed = true;
        }
    }
    if let Some(focus) = payload.get("current_focus").and_then(|v| v.as_str()) {
        let cleaned = focus.trim();
        let next = if cleaned.is_empty() { None } else { Some(cleaned.to_string()) };
        if state.workspace_current_focus != next {
            state.workspace_current_focus = next;
            state.last_focus_change_at = Some(Utc::now().to_rfc3339());
            changed = true;
        }
    }
    if let Some(rationale) = payload.get("focus_rationale").and_then(|v| v.as_str()) {
        let cleaned = rationale.trim();
        let next = if cleaned.is_empty() { None } else { Some(cleaned.to_string()) };
        if state.workspace_focus_rationale != next {
            state.workspace_focus_rationale = next;
            changed = true;
        }
    }
    if let Some(value) = payload.get("goal_stack") {
        let list = extract_goal_stack(value);
        if state.workspace_goal_stack != list {
            state.workspace_goal_stack = list;
            changed = true;
        }
    }
    changed
}

pub(crate) fn format_workspace_anchor(state: &KernelState) -> String {
    let focus = workspace_verified_focus(state).unwrap_or_else(|| "None".to_string());
    let topics = {
        let verified = workspace_verified_topics(state);
        if verified.is_empty() {
            "None".to_string()
        } else {
            verified.join(" | ")
        }
    };
    let questions = {
        let verified = workspace_verified_open_questions(state);
        if verified.is_empty() {
            "None".to_string()
        } else {
            verified.join(" | ")
        }
    };
    format!(
        "focus: {}\nworking_set_topics: {}\nopen_questions: {}",
        focus, topics, questions
    )
}

pub(crate) fn apply_workspace_anchor(summary: &mut InnerSummary, state: &KernelState, outcomes: &[Outcome]) {
    if let Some(focus) = workspace_verified_focus(state) {
        let trimmed = focus.trim();
        if !trimmed.is_empty() {
            summary.focus = trimmed.to_string();
        }
    }
    let verified_topics = workspace_verified_topics(state);
    if !verified_topics.is_empty() {
        summary.active_threads = verified_topics.into_iter().take(3).collect();
    }
    let verified_questions = workspace_verified_open_questions(state);
    if !verified_questions.is_empty() {
        summary.open_questions = verified_questions.into_iter().take(3).collect();
    }
    if !outcomes.is_empty() {
        let mut recent = Vec::new();
        for outcome in outcomes.iter().rev().take(3).rev() {
            recent.push(format!("{}: {}", outcome.action_type, outcome.observations));
        }
        if !recent.is_empty() {
            summary.recent_outcomes = recent;
        }
    }
}

pub(crate) fn refresh_working_memory(state: &mut KernelState, now: DateTime<Utc>, disable_working_hypothesis: bool) {
    let mut block = WorkingMemoryBlock::default();
    let focus = state
        .workspace_current_focus
        .clone()
        .or_else(|| state.workspace_goal_thread.clone())
        .or_else(|| {
            if state.self_state.current_focus.trim().is_empty() {
                None
            } else {
                Some(state.self_state.current_focus.clone())
            }
        });
    block.focus = focus.filter(|s| !s.trim().is_empty());

    block.open_questions = state
        .workspace_open_questions
        .iter()
        .filter_map(|q| {
            let trimmed = q.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
        .take(3)
        .collect();

    block.active_hypotheses = state
        .workspace_active_hypotheses
        .iter()
        .filter_map(|h| {
            let trimmed = h.text.trim();
            if trimmed.is_empty() {
                None
            } else if h.speculative {
                Some(format_speculative_label(trimmed, disable_working_hypothesis))
            } else {
                Some(trimmed.to_string())
            }
        })
        .take(3)
        .collect();

    block.next_action = state
        .pending_actions
        .iter()
        .find(|a| !a.trim().is_empty())
        .map(|a| a.trim().to_string());
    block.confidence = state.controller_state.as_ref().map(|c| c.confidence);
    block.drift_score = state.controller_state.as_ref().map(|c| c.drift_score);
    block.updated_at = Some(now.to_rfc3339());
    state.working_memory = Some(block);
}

#[cfg(test)]
pub(crate) fn auto_memory_decision(user_message: &str, assistant_message: &str) -> AutoMemoryDecision {
    let user_lower = user_message.to_lowercase();
    let assistant_lower = assistant_message.to_lowercase();
    let mut score: f32 = 0.0;
    let mut reasons = Vec::new();

    let identity_hits = ["my name is", "call me", "i am called", "i'm called"]
        .iter()
        .any(|pat| user_lower.contains(pat));
    if identity_hits {
        score += 0.8;
        reasons.push("identity".to_string());
    }

    let preference_hits = ["i like", "i love", "i prefer", "my favorite", "i hate", "i dislike"]
        .iter()
        .any(|pat| user_lower.contains(pat));
    if preference_hits {
        score += 0.4;
        reasons.push("preference".to_string());
    }

    let profile_hits = ["i live in", "i'm from", "i am from", "i work at", "i'm a", "i am a"]
        .iter()
        .any(|pat| user_lower.contains(pat));
    if profile_hits {
        score += 0.3;
        reasons.push("profile".to_string());
    }

    let assistant_ack = ["i'll remember", "i will remember", "noted", "got it", "understood"]
        .iter()
        .any(|pat| assistant_lower.contains(pat));
    if assistant_ack {
        score += 0.1;
        reasons.push("assistant_ack".to_string());
    }

    let has_anchor = user_lower
        .split(|c: char| !c.is_alphanumeric())
        .any(|tok| tok == "i" || tok == "my");
    if !has_anchor {
        reasons.push("missing_anchor".to_string());
    }

    let ambiguity_terms = [
        "maybe",
        "might",
        "not sure",
        "unsure",
        "i think",
        "probably",
        "approximately",
        "around",
        "guess",
        "perhaps",
    ];
    let ambiguity = ambiguity_terms
        .iter()
        .any(|term| user_lower.contains(term) || assistant_lower.contains(term))
        || !has_anchor;

    let score = score.min(1.0);
    let trigger = score >= AUTO_MEMORY_CONFIDENCE_THRESHOLD && !ambiguity;
    AutoMemoryDecision {
        trigger,
        score,
        ambiguity,
        reasons,
    }
}
