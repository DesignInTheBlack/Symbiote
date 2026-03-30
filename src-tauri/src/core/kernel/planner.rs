use serde_json::Value;
use std::collections::HashSet;

use crate::core::kernel::prompt::candidate_relevance_text;
use crate::core::kernel::utils::text::{hash_payload, summarize_snippet};
use crate::models::{GoalStep, GoalStackItem};

use super::{Candidate, CandidateKind, KernelState, RejectedCandidate};

fn goal_status_complete(status: Option<&str>) -> bool {
    let Some(status) = status else {
        return false;
    };
    matches!(
        status.trim().to_lowercase().as_str(),
        "done" | "complete" | "completed" | "finished"
    )
}

fn sanitize_plan_step_text(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut cleaned = trimmed
        .trim_end_matches(|c: char| c == '.' || c == ';' || c == ',')
        .trim()
        .to_string();
    if cleaned.len() > 160 {
        cleaned = summarize_snippet(&cleaned, 160);
    }
    cleaned
}

fn extract_bulleted_steps(input: &str) -> Vec<String> {
    let mut steps = Vec::new();
    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut content: Option<&str> = None;
        if let Some(rest) = trimmed.strip_prefix("- ") {
            content = Some(rest);
        } else if let Some(rest) = trimmed.strip_prefix("* ") {
            content = Some(rest);
        } else {
            let mut digits = String::new();
            for ch in trimmed.chars() {
                if ch.is_ascii_digit() {
                    digits.push(ch);
                } else {
                    break;
                }
            }
            if !digits.is_empty() {
                let rest = trimmed[digits.len()..].trim_start();
                if rest.starts_with('.') || rest.starts_with(')') {
                    let rest = rest[1..].trim_start();
                    if !rest.is_empty() {
                        content = Some(rest);
                    }
                }
            }
        }
        if let Some(content) = content {
            let cleaned = sanitize_plan_step_text(content);
            if !cleaned.is_empty() {
                steps.push(cleaned);
            }
        }
    }
    steps
}

fn extract_sentence_steps(input: &str) -> Vec<String> {
    let mut steps = Vec::new();
    for segment in input.split(|c| c == '.' || c == '!' || c == '?') {
        let cleaned = sanitize_plan_step_text(segment);
        if !cleaned.is_empty() {
            steps.push(cleaned);
        }
    }
    steps
}

pub fn parse_plan_steps_from_input(input: &str) -> Vec<String> {
    let mut steps = extract_bulleted_steps(input);
    if steps.is_empty() {
        let lowered = input.to_lowercase();
        let mut normalized = input.to_string();
        for marker in [" and then ", " then ", " next ", " after that ", " after ", " afterward ", " lastly "] {
            if lowered.contains(marker) {
                normalized = normalized.replace(marker, "\n");
            }
        }
        steps = normalized
            .lines()
            .map(sanitize_plan_step_text)
            .filter(|s| !s.is_empty())
            .collect();
    }
    if steps.is_empty() {
        steps = extract_sentence_steps(input);
    }
    if steps.is_empty() {
        steps = vec![
            "Clarify objective and constraints".to_string(),
            "Execute the next concrete step".to_string(),
            "Verify outcome and capture evidence".to_string(),
        ];
    }
    steps.truncate(6);
    steps
}

pub fn build_goal_steps(step_texts: Vec<String>) -> Vec<GoalStep> {
    let mut steps = Vec::new();
    for (idx, text) in step_texts.into_iter().enumerate() {
        if text.trim().is_empty() {
            continue;
        }
        let preconditions = if idx == 0 {
            vec!["input_parsed".to_string()]
        } else {
            vec![format!("step_{}_complete", idx)]
        };
        let postconditions = vec![format!("step_{}_complete", idx + 1)];
        steps.push(GoalStep {
            text,
            status: None,
            preconditions,
            postconditions,
            failure_count: 0,
            last_failure_at: None,
            evidence_event_ids: Vec::new(),
            belief_ids: Vec::new(),
            completed_at: None,
        });
    }
    if steps.is_empty() {
        steps.push(GoalStep {
            text: "Clarify objective and constraints".to_string(),
            status: None,
            preconditions: vec!["input_parsed".to_string()],
            postconditions: vec!["step_1_complete".to_string()],
            failure_count: 0,
            last_failure_at: None,
            evidence_event_ids: Vec::new(),
            belief_ids: Vec::new(),
            completed_at: None,
        });
    }
    steps
}

pub fn plan_goal_text(input: &str) -> String {
    let summary = summarize_snippet(input, 120);
    let cleaned = summary.trim().trim_end_matches(|c: char| c == '.' || c == ';' || c == ',');
    if cleaned.is_empty() {
        "Active plan".to_string()
    } else {
        cleaned.to_string()
    }
}

pub fn plan_hash(goal: &str, steps: &[GoalStep]) -> String {
    let payload = serde_json::json!({
        "goal": goal,
        "steps": steps,
    });
    hash_payload(&payload.to_string())
}

pub fn tokenize_plan_text(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(|s| s.trim().to_lowercase())
        .filter(|s| s.len() >= 3)
        .collect()
}

pub fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut intersection = 0usize;
    for item in a.iter() {
        if b.contains(item) {
            intersection += 1;
        }
    }
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

pub fn payload_step_index(payload: &Value) -> Option<usize> {
    if let Some(idx) = payload.get("step_index").and_then(|v| v.as_i64()) {
        if idx > 0 {
            return Some((idx - 1) as usize);
        }
    }
    if let Some(step_id) = payload.get("plan_step_id").and_then(|v| v.as_str()) {
        let trimmed = step_id.trim();
        if let Some((_, last)) = trimmed.rsplit_once(':') {
            if let Ok(idx) = last.trim().parse::<i64>() {
                if idx > 0 {
                    return Some((idx - 1) as usize);
                }
            }
        }
    }
    None
}

pub fn assign_plan_step_indices(candidates: &mut [Candidate], state: &KernelState) {
    let Some(active_goal) = state
        .workspace_goal_stack
        .iter()
        .find(|item| !goal_status_complete(item.status.as_deref()))
    else {
        return;
    };
    if active_goal.steps.is_empty() {
        return;
    }
    let current_step_index = active_goal.current_step_index.min(active_goal.steps.len().saturating_sub(1));
    let step_tokens: Vec<HashSet<String>> = active_goal
        .steps
        .iter()
        .map(|step| tokenize_plan_text(&step.text))
        .collect();
    let plan_hash_value = state.last_plan_hash.clone();

    for candidate in candidates.iter_mut() {
        if payload_step_index(&candidate.payload).is_some() {
            continue;
        }
        match candidate.kind {
            CandidateKind::EmitMessage
            | CandidateKind::AskUserQuestion
            | CandidateKind::ToolCall
            | CandidateKind::UpdateWorkspace
            | CandidateKind::UpdateGoalThread
            | CandidateKind::RecordSelfClaim
            | CandidateKind::SpawnThread => {}
            _ => continue,
        }
        let candidate_text = candidate_relevance_text(candidate);
        let candidate_tokens = tokenize_plan_text(&candidate_text);
        let mut best_idx: Option<usize> = None;
        let mut best_score = 0.0_f32;
        for (idx, tokens) in step_tokens.iter().enumerate() {
            let score = jaccard_similarity(&candidate_tokens, tokens);
            if score > best_score {
                best_score = score;
                best_idx = Some(idx);
            }
        }
        let selected_idx = if best_score >= 0.2 {
            best_idx
        } else {
            Some(current_step_index)
        };
        if let Some(step_idx) = selected_idx {
            if let Some(obj) = candidate.payload.as_object_mut() {
                obj.entry("step_index".to_string())
                    .or_insert(Value::Number(((step_idx + 1) as i64).into()));
                if let Some(step) = active_goal.steps.get(step_idx) {
                    obj.entry("plan_step_text".to_string())
                        .or_insert(Value::String(step.text.clone()));
                }
                if let Some(hash) = plan_hash_value.as_ref() {
                    obj.entry("plan_hash".to_string())
                        .or_insert(Value::String(hash.clone()));
                }
                let step_id = if let Some(active_plan_id) = state.workspace_active_plan_id.as_deref() {
                    format!("{}:{}", active_plan_id, step_idx + 1)
                } else if let Some(hash) = plan_hash_value.as_ref() {
                    format!("{}:{}", hash, step_idx + 1)
                } else {
                    String::new()
                };
                if !step_id.is_empty() {
                    obj.insert("plan_step_id".to_string(), Value::String(step_id));
                }
            }
            candidate.refresh_meta();
        }
    }
}

pub fn filter_candidates_for_active_step(
    state: &KernelState,
    candidates: Vec<Candidate>,
) -> (Vec<Candidate>, Vec<RejectedCandidate>) {
    let Some(active_goal) = state
        .workspace_goal_stack
        .iter()
        .find(|item| !goal_status_complete(item.status.as_deref()))
    else {
        return (candidates, Vec::new());
    };
    if active_goal.steps.is_empty() {
        return (candidates, Vec::new());
    }
    let current_step_index = active_goal.current_step_index.min(active_goal.steps.len().saturating_sub(1));
    let mut kept = Vec::new();
    let mut rejected = Vec::new();

    for candidate in candidates {
        if matches!(
            candidate.kind,
            CandidateKind::Terminate | CandidateKind::ChangeMode | CandidateKind::NoOp
        ) {
            kept.push(candidate);
            continue;
        }
        let step_idx = payload_step_index(&candidate.payload);
        if step_idx == Some(current_step_index) {
            kept.push(candidate);
            continue;
        }
        let tool_name = candidate
            .payload
            .get("tool_name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        rejected.push(RejectedCandidate {
            id: candidate.id,
            kind: candidate.kind,
            reason: if step_idx.is_none() {
                "plan_step_missing".to_string()
            } else {
                "plan_step_mismatch".to_string()
            },
            tool_name,
            source: Some(candidate.source),
            is_monologue: None,
            payload: Some(candidate.payload),
        });
    }
    (kept, rejected)
}

pub fn active_goal(state: &KernelState) -> Option<GoalStackItem> {
    state
        .workspace_goal_stack
        .iter()
        .find(|item| !goal_status_complete(item.status.as_deref()))
        .cloned()
}
