use serde::{Deserialize, Serialize};

const MAX_ITEMS: usize = 3;
const DEFAULT_MAX_CHARS: usize = 1000;
const MAX_ITEM_CHARS: usize = 180;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InnerSummary {
    pub focus: String,
    pub active_threads: Vec<String>,
    pub blockers: Vec<String>,
    pub next_moves: Vec<String>,
    pub open_questions: Vec<String>,
    pub recent_outcomes: Vec<String>,
}

impl InnerSummary {
    pub fn from_json(raw: &str) -> Self {
        serde_json::from_str(raw).unwrap_or_default()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

pub fn sanitize_inner_summary(mut summary: InnerSummary, max_chars: usize) -> (InnerSummary, bool) {
    let mut trimmed = false;

    summary.focus = sanitize_item(&summary.focus, &mut trimmed);
    summary.active_threads = trim_list(summary.active_threads, &mut trimmed);
    summary.blockers = trim_list(summary.blockers, &mut trimmed);
    summary.next_moves = trim_list(summary.next_moves, &mut trimmed);
    summary.open_questions = trim_list(summary.open_questions, &mut trimmed);
    summary.recent_outcomes = trim_list(summary.recent_outcomes, &mut trimmed);

    let max_chars = if max_chars == 0 { DEFAULT_MAX_CHARS } else { max_chars };
    let mut json = summary.to_json();
    if json.len() > max_chars {
        trimmed = true;
        // Aggressive trimming: drop from the longest lists until under cap.
        loop {
            if json.len() <= max_chars {
                break;
            }
            let mut changed = false;
            if !summary.recent_outcomes.is_empty() {
                summary.recent_outcomes.pop();
                changed = true;
            } else if !summary.open_questions.is_empty() {
                summary.open_questions.pop();
                changed = true;
            } else if !summary.next_moves.is_empty() {
                summary.next_moves.pop();
                changed = true;
            } else if !summary.blockers.is_empty() {
                summary.blockers.pop();
                changed = true;
            } else if !summary.active_threads.is_empty() {
                summary.active_threads.pop();
                changed = true;
            } else if summary.focus.len() > 0 {
                summary.focus = summary.focus.chars().take(MAX_ITEM_CHARS).collect();
                changed = true;
            }
            if !changed {
                break;
            }
            json = summary.to_json();
            if summary.focus.is_empty() && summary.active_threads.is_empty() && summary.blockers.is_empty()
                && summary.next_moves.is_empty() && summary.open_questions.is_empty() && summary.recent_outcomes.is_empty() {
                break;
            }
        }
    }

    (summary, trimmed)
}

pub fn format_for_prompt(summary: &InnerSummary) -> String {
    summary.to_json()
}

fn trim_list(items: Vec<String>, trimmed: &mut bool) -> Vec<String> {
    let mut out = Vec::new();
    for item in items.into_iter().take(MAX_ITEMS) {
        let cleaned = sanitize_item(&item, trimmed);
        if !cleaned.is_empty() {
            out.push(cleaned);
        }
    }
    out
}

fn contains_diagnostic_marker(text: &str) -> bool {
    let lower = text.to_lowercase();
    let markers = [
        "telemetry",
        "tool manifest",
        "tool list",
        "controller state",
        "kv memory",
        "prompt hash",
        "run_id",
        "trace_id",
        "timestamp",
        "latency",
        "module_status",
        "system log",
    ];
    markers.iter().any(|marker| lower.contains(marker))
}

fn sanitize_item(item: &str, trimmed: &mut bool) -> String {
    let cleaned = trim_item(item, trimmed);
    if cleaned.is_empty() {
        return cleaned;
    }
    if contains_diagnostic_marker(&cleaned) {
        *trimmed = true;
        return "".to_string();
    }
    cleaned
}

fn trim_item(item: &str, trimmed: &mut bool) -> String {
    let mut s = item.trim().to_string();
    if s.len() > MAX_ITEM_CHARS {
        s = s.chars().take(MAX_ITEM_CHARS).collect();
        *trimmed = true;
    }
    s
}
