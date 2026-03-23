use std::collections::HashMap;
use std::panic::Location;
use std::sync::Mutex;

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::SqlitePool;
use tauri::{AppHandle, Emitter};

use crate::core::system_log;
use chrono::Utc;
use uuid::Uuid;

fn module_stage_for_phase(phase: RunPhase) -> Option<(&'static str, &'static str)> {
    match phase {
        RunPhase::Created => Some(("queued", "Queued for processing")),
        RunPhase::Ingest => Some(("ingest_input", "Ingesting input")),
        RunPhase::PromptBuild => Some(("prompt_build", "Building prompt")),
        RunPhase::ModelCall => Some(("llm_wait", "Waiting on LLM")),
        RunPhase::Arbitration => Some(("arbitration", "Arbitrating candidates")),
        RunPhase::Commit => Some(("commit_cycle", "Finalizing updates")),
        RunPhase::ToolDispatch => Some(("tool_call", "Calling tool")),
        RunPhase::Finalize => Some(("finalize", "Finalizing response")),
        RunPhase::Complete => Some(("idle", "Idle")),
        RunPhase::Error => Some(("error", "Error")),
        RunPhase::Cancelled => Some(("cancelled", "Cancelled")),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunPhase {
    Created,
    Ingest,
    PromptBuild,
    ModelCall,
    Arbitration,
    Commit,
    ToolDispatch,
    Finalize,
    Complete,
    Error,
    Cancelled,
}

impl RunPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            RunPhase::Created => "created",
            RunPhase::Ingest => "ingest",
            RunPhase::PromptBuild => "prompt_build",
            RunPhase::ModelCall => "model_call",
            RunPhase::Arbitration => "arbitration",
            RunPhase::Commit => "commit",
            RunPhase::ToolDispatch => "tool_dispatch",
            RunPhase::Finalize => "finalize",
            RunPhase::Complete => "complete",
            RunPhase::Error => "error",
            RunPhase::Cancelled => "cancelled",
        }
    }
}

static RUN_PHASES: Lazy<Mutex<HashMap<String, RunPhase>>> = Lazy::new(|| Mutex::new(HashMap::new()));

fn is_terminal(phase: RunPhase) -> bool {
    matches!(phase, RunPhase::Complete | RunPhase::Error | RunPhase::Cancelled)
}

fn allowed_transition(current: RunPhase, next: RunPhase) -> bool {
    if current == next {
        return true;
    }
    if is_terminal(current) {
        return false;
    }
    if matches!(next, RunPhase::Error | RunPhase::Cancelled) {
        return true;
    }
    match current {
        RunPhase::Created => matches!(next, RunPhase::Ingest),
        RunPhase::Ingest => matches!(next, RunPhase::PromptBuild),
        RunPhase::PromptBuild => matches!(next, RunPhase::ModelCall),
        RunPhase::ModelCall => matches!(next, RunPhase::Arbitration | RunPhase::PromptBuild),
        RunPhase::Arbitration => matches!(next, RunPhase::Commit),
        RunPhase::Commit => matches!(next, RunPhase::ToolDispatch | RunPhase::Finalize | RunPhase::Complete),
        RunPhase::ToolDispatch => matches!(
            next,
            RunPhase::Ingest | RunPhase::Arbitration | RunPhase::Commit | RunPhase::Finalize | RunPhase::Complete
        ),
        RunPhase::Finalize => matches!(next, RunPhase::Complete),
        RunPhase::Complete | RunPhase::Error | RunPhase::Cancelled => false,
    }
}

pub async fn advance_run_phase(
    pool: &SqlitePool,
    app_handle: Option<&AppHandle>,
    run_id: &str,
    next: RunPhase,
    detail: Option<&str>,
) -> Result<RunPhase, String> {
    let (prev, allowed, should_clear) = {
        let mut guard = RUN_PHASES.lock().map_err(|_| "run_phase_lock_poisoned")?;
        let current = guard.get(run_id).copied().unwrap_or(RunPhase::Created);
        let allowed = allowed_transition(current, next);
        if allowed {
            if is_terminal(next) {
                guard.remove(run_id);
            } else {
                guard.insert(run_id.to_string(), next);
            }
        }
        (current, allowed, is_terminal(next))
    };

    if !allowed {
        let caller = Location::caller();
        let ignore_reason = if is_terminal(prev) {
            Some("terminal_phase")
        } else if prev == RunPhase::Commit {
            Some("commit_locked")
        } else {
            None
        };
        if let Some(reason) = ignore_reason {
            let _ = system_log::log_event(
                pool,
                app_handle,
                "info",
                "kernel",
                Some(run_id),
                None,
                json!({
                    "event": "run_phase_transition_ignored",
                    "from": prev.as_str(),
                    "to": next.as_str(),
                    "detail": detail.unwrap_or(""),
                    "reason": reason,
                    "caller_file": caller.file(),
                    "caller_line": caller.line(),
                    "caller_column": caller.column(),
                }),
            )
            .await;
            return Ok(prev);
        }

        let _ = system_log::log_event(
            pool,
            app_handle,
            "warn",
            "kernel",
            Some(run_id),
            None,
            json!({
                "event": "run_phase_invalid_transition",
                "from": prev.as_str(),
                "to": next.as_str(),
                "detail": detail.unwrap_or(""),
                "caller_file": caller.file(),
                "caller_line": caller.line(),
                "caller_column": caller.column(),
            }),
        )
        .await;
        return Err(format!(
            "Invalid run phase transition: {} -> {} at {}:{}",
            prev.as_str(),
            next.as_str(),
            caller.file(),
            caller.line()
        ));
    }

    let _ = system_log::log_event(
        pool,
        app_handle,
        "info",
        "kernel",
        Some(run_id),
        None,
        json!({
            "event": "run_phase_transition",
            "from": prev.as_str(),
            "to": next.as_str(),
            "detail": detail.unwrap_or(""),
            "terminal": should_clear,
        }),
    )
    .await;
    let _ = sqlx::query(
        "INSERT INTO event_ledger (event_id, timestamp, type, payload, tags, run_id, trace_id)
         VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, NULL)",
    )
    .bind(Uuid::new_v4().to_string())
    .bind("system_transition")
    .bind(json!({
        "from": prev.as_str(),
        "to": next.as_str(),
        "detail": detail.unwrap_or(""),
        "terminal": should_clear,
    }).to_string())
    .bind(json!({ "phase": next.as_str() }).to_string())
    .bind(run_id)
    .execute(pool)
    .await;

    if let Some(handle) = app_handle {
        if let Some((stage, default_detail)) = module_stage_for_phase(next) {
            let payload = json!({
                "event": "module_status",
                "run_id": run_id,
                "stage": stage,
                "detail": if let Some(d) = detail { d } else { default_detail },
                "started_at": Utc::now().to_rfc3339(),
                "duration_ms": 0,
            });
            let _ = handle.emit("module_status", payload);
        }
    }

    Ok(next)
}
