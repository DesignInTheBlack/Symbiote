use std::process::Stdio;
use std::collections::{HashSet, VecDeque};
use std::time::Duration;
use tokio::process::Command;
use tokio::sync::watch;

use chrono::{DateTime, Timelike, Utc};
use chrono_tz::Tz;
use reqwest::Client;
use regex::Regex;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sqlx::Row;
use url::Url;

use crate::db::Db;
use crate::core::system_log;
use crate::core::system_controls;
use crate::core::episodic;
use crate::core::world_model;
use crate::core::kernel::utils::summarize_snippet;
use crate::core::kernel::KernelState;
use crate::models::{Tool, ToolFunction, Settings};

pub struct ToolRegistry;
const DEFAULT_TOOL_TIMEOUT_SECS: u64 = 20;

#[derive(Debug, Clone)]
pub struct ToolCapability {
    pub risk: f32,
    pub min_autonomy: f32,
    pub require_telemetry: bool,
    pub require_evidence: bool,
    pub allow_degraded: bool,
}

impl ToolCapability {
    pub fn low() -> Self {
        Self {
            risk: 0.2,
            min_autonomy: 0.15,
            require_telemetry: false,
            require_evidence: false,
            allow_degraded: true,
        }
    }

    pub fn medium() -> Self {
        Self {
            risk: 0.5,
            min_autonomy: 0.35,
            require_telemetry: true,
            require_evidence: true,
            allow_degraded: true,
        }
    }

    pub fn high() -> Self {
        Self {
            risk: 0.85,
            min_autonomy: 0.6,
            require_telemetry: true,
            require_evidence: true,
            allow_degraded: false,
        }
    }
}

impl ToolRegistry {
    pub fn is_self_awareness_tool(name: &str) -> bool {
        matches!(
            name.trim().to_lowercase().as_str(),
            "get_workspace_state"
                | "get_inner_summary"
                | "get_rolling_summary"
                | "get_system_capabilities"
                | "get_unified_self"
                | "get_autobiographical_context"
        )
    }

    pub fn is_context_only_tool(name: &str) -> bool {
        matches!(
            name.trim().to_lowercase().as_str(),
            "get_world_model_snapshot"
                | "get_goal_stack"
                | "get_plan_summary"
                | "get_recent_outcomes"
                | "get_workspace_state"
                | "get_inner_summary"
                | "get_rolling_summary"
                | "get_system_capabilities"
                | "get_unified_self"
                | "get_autobiographical_context"
        )
    }

    pub fn capability_for(name: &str) -> ToolCapability {
        match name.trim().to_lowercase().as_str() {
            "run_shell" => ToolCapability::high(),
            "web_lookup" => ToolCapability::medium(),
            "save_context" => ToolCapability::medium(),
            "read_context" => ToolCapability::medium(),
            "get_system_logs" => ToolCapability::medium(),
            _ => ToolCapability::low(),
        }
    }

    pub fn definitions(&self) -> Vec<Tool> {
        vec![
            Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: "run_shell".to_string(),
                    description: "Execute a system shell command (Powershell/CMD). Use this to manage files, check system status, or run scripts.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "command": { "type": "string", "description": "Command line string to execute (e.g., 'dir')." },
                            "shell": { "type": "string", "description": "Optional shell override: 'powershell' or 'cmd'." }
                        },
                        "required": ["command"]
                    }),
                    timeout_secs: Some(25),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: "get_current_time".to_string(),
                    description: "Get the current system date and time string.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "timezone": { "type": "string", "description": "Optional timezone, e.g. 'UTC' or 'America/New_York'." }
                        },
                        "required": []
                    }),
                    timeout_secs: Some(6),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: "get_system_logs".to_string(),
                    description: "Read recent system logs for runtime self-audit and behavior review.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "limit": { "type": "integer", "description": "Max entries to return (1-100). Default 20." },
                            "category": { "type": "string", "description": "Optional log category filter (e.g., 'kernel', 'memory')." },
                            "level": { "type": "string", "description": "Optional log level filter (e.g., 'info', 'warn', 'error')." },
                            "run_id": { "type": "string", "description": "Optional run_id filter." }
                        },
                        "required": []
                    }),
                    timeout_secs: Some(12),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: "get_system_capabilities".to_string(),
                    description: "Fetch a structured inventory of Symbiote system capabilities (UI panels, controls, memory subsystems, audit surfaces).".to_string(),
                    parameters: serde_json::json!( {
                        "type": "object",
                        "properties": {
                            "conversation_id": { "type": "string", "description": "Optional conversation id (default 'default')." }
                        },
                        "required": []
                    }),
                    timeout_secs: Some(12),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: "get_unified_self".to_string(),
                    description: "Fetch the unified self-model snapshot (workspace, qualia, wave state, autobiographical summary).".to_string(),
                    parameters: serde_json::json!( {
                        "type": "object",
                        "properties": {
                            "conversation_id": { "type": "string", "description": "Optional conversation id (default 'default')." }
                        },
                        "required": []
                    }),
                    timeout_secs: Some(12),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: "get_autobiographical_context".to_string(),
                    description: "Fetch autobiographical context with identity relevance and valence signals.".to_string(),
                    parameters: serde_json::json!( {
                        "type": "object",
                        "properties": {
                            "conversation_id": { "type": "string", "description": "Optional conversation id (default 'default')." },
                            "limit": { "type": "integer", "description": "Max events to return (1-12). Default 6." }
                        },
                        "required": []
                    }),
                    timeout_secs: Some(12),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: "get_inner_summary".to_string(),
                    description: "Fetch the latest inner summary snapshot for the current conversation.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "conversation_id": { "type": "string", "description": "Optional conversation id (default 'default')." }
                        },
                        "required": []
                    }),
                    timeout_secs: Some(12),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: "get_workspace_state".to_string(),
                    description: "Fetch the current workspace state snapshot.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "conversation_id": { "type": "string", "description": "Optional conversation id (default 'default')." }
                        },
                        "required": []
                    }),
                    timeout_secs: Some(12),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: "get_rolling_summary".to_string(),
                    description: "Fetch the latest rolling summary snapshot.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "conversation_id": { "type": "string", "description": "Optional conversation id (default 'default')." }
                        },
                        "required": []
                    }),
                    timeout_secs: Some(12),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: "get_world_model_snapshot".to_string(),
                    description: "Fetch the current world model snapshot for the conversation (read-only context).".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "conversation_id": { "type": "string", "description": "Optional conversation id (default 'default')." }
                        },
                        "required": []
                    }),
                    timeout_secs: Some(12),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: "get_goal_stack".to_string(),
                    description: "Fetch the current goal stack for the conversation (read-only context).".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "conversation_id": { "type": "string", "description": "Optional conversation id (default 'default')." }
                        },
                        "required": []
                    }),
                    timeout_secs: Some(12),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: "get_plan_summary".to_string(),
                    description: "Fetch recent plan artifacts (action proposals and decision reports) for the conversation (read-only context).".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "conversation_id": { "type": "string", "description": "Optional conversation id (default 'default')." },
                            "limit": { "type": "integer", "description": "Optional max items to return (default 3)." }
                        },
                        "required": []
                    }),
                    timeout_secs: Some(12),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: "get_recent_outcomes".to_string(),
                    description: "Fetch recent outcome events for the conversation (read-only context).".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "conversation_id": { "type": "string", "description": "Optional conversation id (default 'default')." },
                            "limit": { "type": "integer", "description": "Optional max items to return (default 5)." }
                        },
                        "required": []
                    }),
                    timeout_secs: Some(12),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: "save_context".to_string(),
                    description: "Save information to the Blackboard (KV Store).".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "key": { "type": "string", "description": "Unique key for the memory." },
                            "value": { "type": "string", "description": "Content to save." }
                        },
                        "required": ["key", "value"]
                    }),
                    timeout_secs: Some(12),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: "read_context".to_string(),
                    description: "Read information from the Blackboard (KV Store).".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "key": { "type": "string", "description": "Key to look up." }
                        },
                        "required": ["key"]
                    }),
                    timeout_secs: Some(12),
                },
            },
            Tool {
                r#type: "function".to_string(),
                function: ToolFunction {
                    name: "web_lookup".to_string(),
                    description: "Search the web and return vetted sources with evidence IDs. Use for factual lookups and external verification.".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": { "type": "string", "description": "Search query to look up." },
                            "max_results": { "type": "integer", "description": "Maximum number of sources to return (1-10). Default 5." },
                            "domains_allowlist": { "type": "array", "items": { "type": "string" }, "description": "Optional allowlist of domains (e.g., ['wikipedia.org','nasa.gov'])." },
                            "recency_days": { "type": "integer", "description": "Optional recency filter in days (0 = no filter)." },
                            "follow_links": { "type": "boolean", "description": "If true, expand with follow-up queries and on-page links." },
                            "uncertainty": { "type": "string", "description": "Optional justification: why external lookup is needed." },
                            "decision_impact": { "type": "string", "description": "Optional justification: how the lookup affects the response." }
                        },
                        "required": ["query"]
                    }),
                    timeout_secs: Some(40),
                },
            },
        ]
    }

    pub fn definitions_for_settings(&self, settings: &Settings) -> Vec<Tool> {
        let mut tools = self.definitions();
        if !settings.allow_shell_tool.unwrap_or(false) {
            tools.retain(|tool| tool.function.name != "run_shell");
        }
        tools
    }

    pub fn allowed_tool_names(&self, settings: &Settings) -> Vec<String> {
        self.definitions_for_settings(settings)
            .into_iter()
            .map(|tool| tool.function.name)
            .collect()
    }

    pub fn is_tool_allowed(&self, name: &str, settings: &Settings) -> bool {
        let name = name.trim();
        if name.is_empty() {
            return false;
        }
        self.definitions_for_settings(settings)
            .iter()
            .any(|tool| tool.function.name.eq_ignore_ascii_case(name))
    }

    pub fn schema_for(&self, name: &str) -> Option<Value> {
        let needle = name.trim();
        if needle.is_empty() {
            return None;
        }
        self.definitions()
            .into_iter()
            .find(|tool| tool.function.name.eq_ignore_ascii_case(needle))
            .map(|tool| tool.function.parameters)
    }

    pub fn timeout_for_name(name: &str) -> u64 {
        match name.trim().to_lowercase().as_str() {
            "web_lookup" => 40,
            "run_shell" => 25,
            "get_system_logs" => 12,
            "get_system_capabilities" => 12,
            "get_unified_self" => 12,
            "get_autobiographical_context" => 12,
            "get_current_time" => 6,
            "get_inner_summary" => 12,
            "get_workspace_state" => 12,
            "get_rolling_summary" => 12,
            "get_world_model_snapshot" => 12,
            "get_goal_stack" => 12,
            "get_plan_summary" => 12,
            "get_recent_outcomes" => 12,
            "save_context" => 12,
            "read_context" => 12,
            _ => DEFAULT_TOOL_TIMEOUT_SECS,
        }
    }

    pub fn timeout_for(&self, name: &str) -> u64 {
        let needle = name.trim();
        if needle.is_empty() {
            return DEFAULT_TOOL_TIMEOUT_SECS;
        }
        self.definitions()
            .into_iter()
            .find(|tool| tool.function.name.eq_ignore_ascii_case(needle))
            .and_then(|tool| tool.function.timeout_secs)
            .unwrap_or_else(|| Self::timeout_for_name(needle))
    }

    pub async fn execute(
        &self,
        db: &Db,
        name: &str,
        args_json: &str,
        cancel_rx: &mut watch::Receiver<bool>,
        run_id: Option<&str>,
        trace_id: Option<&str>,
    ) -> Result<String, String> {
        if *cancel_rx.borrow() {
            return Err("cancelled".to_string());
        }
        let mode: Option<String> = sqlx::query_scalar(
            "SELECT mode FROM system_controls WHERE subsystem_id = ?",
        )
        .bind("tool_execution")
        .fetch_optional(&db.pool)
        .await
        .ok()
        .flatten();
        let tool_mode = mode.unwrap_or_else(|| {
            system_controls::default_mode_for("tool_execution")
                .unwrap_or("normal")
                .to_string()
        });
        if system_controls::mode_is_off(&tool_mode) {
            let _ = system_log::log_event(
                &db.pool,
                None,
                "warn",
                "tool",
                run_id,
                trace_id,
                serde_json::json!({
                    "event": "tool_gate_blocked",
                    "reason": "system_control_off",
                    "tool": name,
                }),
            )
            .await;
            return Err("tool_execution disabled by controls".to_string());
        }
        if system_controls::mode_is_degraded(&tool_mode) {
            let safe_tools = [
                "get_current_time",
                "get_system_logs",
                "get_system_capabilities",
                "get_unified_self",
                "get_autobiographical_context",
                "get_inner_summary",
                "get_workspace_state",
                "get_rolling_summary",
                "get_world_model_snapshot",
                "get_goal_stack",
                "get_plan_summary",
                "get_recent_outcomes",
                "read_context",
            ];
            if !safe_tools.iter().any(|t| t.eq_ignore_ascii_case(name)) {
                let _ = system_log::log_event(
                    &db.pool,
                    None,
                    "warn",
                    "tool",
                    run_id,
                    trace_id,
                    serde_json::json!({
                        "event": "tool_gate_blocked",
                        "reason": "system_control_degraded",
                        "tool": name,
                    }),
                )
                .await;
                return Err("tool_execution degraded by controls".to_string());
            }
        }

        let settings = db.get_settings().await.map_err(|e| e.to_string())?;

        match name {
            "run_shell" => {
                if !settings.allow_shell_tool.unwrap_or(false) {
                    let _ = system_log::log_event(
                        &db.pool,
                        None,
                        "warn",
                        "tool",
                        None,
                        None,
                        serde_json::json!({
                            "event": "shell_blocked",
                            "reason": "disabled",
                        }),
                    )
                    .await;
                    return Err("run_shell disabled by settings".to_string());
                }
                let allowlist = parse_shell_allowlist(settings.shell_command_allowlist.as_deref());
                let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
                let command = args
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or(args_json)
                    .trim()
                    .to_string();
                if command.is_empty() {
                    return Err("Missing command".to_string());
                }
                if !command_allowed(&command, &allowlist) {
                    let _ = system_log::log_event(
                        &db.pool,
                        None,
                        "warn",
                        "tool",
                        None,
                        None,
                        serde_json::json!({
                            "event": "shell_blocked",
                            "reason": "allowlist",
                            "command": command,
                        }),
                    )
                    .await;
                    return Err("Command blocked by allowlist".to_string());
                }
                run_shell(args_json, cancel_rx, &allowlist).await
            }
            "get_current_time" => get_current_time(args_json),
            "get_system_logs" => get_system_logs(db, args_json).await,
            "get_system_capabilities" => get_system_capabilities(db, args_json).await,
            "get_unified_self" => get_unified_self(db, args_json).await,
            "get_autobiographical_context" => get_autobiographical_context(db, args_json).await,
            "get_inner_summary" => get_inner_summary(db, args_json).await,
            "get_workspace_state" => get_workspace_state(db, args_json).await,
            "get_rolling_summary" => get_rolling_summary(db, args_json).await,
            "get_world_model_snapshot" => get_world_model_snapshot(db, args_json).await,
            "get_goal_stack" => get_goal_stack(db, args_json).await,
            "get_plan_summary" => get_plan_summary(db, args_json).await,
            "get_recent_outcomes" => get_recent_outcomes(db, args_json).await,
            "save_context" => save_context(db, &settings, args_json).await,
            "read_context" => read_context(db, &settings, args_json).await,
            "web_lookup" => web_lookup(db, &settings, args_json, cancel_rx, run_id, trace_id).await,
            _ => Err(format!("Unknown tool: {}", name)),
        }
    }
}

async fn run_shell(
    args_json: &str,
    cancel_rx: &mut watch::Receiver<bool>,
    allowlist: &[String],
) -> Result<String, String> {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
    let command = args.get("command").and_then(|v| v.as_str()).unwrap_or(args_json).trim().to_string();
    if command.is_empty() {
        return Err("Missing command".to_string());
    }
    if !command_allowed(&command, allowlist) {
        return Err("Command blocked by allowlist".to_string());
    }

    let shell = args.get("shell").and_then(|v| v.as_str()).unwrap_or("powershell");
    let (exe, arg) = if shell.eq_ignore_ascii_case("cmd") {
        ("cmd", "/C")
    } else {
        ("powershell", "-Command")
    };

    let child = Command::new(exe)
        .kill_on_drop(true)
        .arg(arg)
        .arg(&command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to execute shell command: {}", e))?;

    let output = tokio::select! {
        res = child.wait_with_output() => {
            res.map_err(|e| format!("Failed to read output: {}", e))?
        }
        _ = cancel_rx.changed() => {
            return Err("cancelled".to_string());
        }
    };

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

fn get_current_time(args_json: &str) -> Result<String, String> {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
    let tz_str = args.get("timezone").and_then(|v| v.as_str()).unwrap_or("UTC");
    let now = match tz_str.parse::<Tz>() {
        Ok(tz) => Utc::now().with_timezone(&tz),
        Err(_) => Utc::now().with_timezone(&chrono_tz::UTC),
    };

    let h = now.hour();
    let period_desc = match h {
        0..=4 => "at night",
        5..=11 => "in the morning",
        12..=16 => "in the afternoon",
        17..=20 => "in the evening",
        21..=23 => "at night",
        _ => "",
    };

    let time_str = format!(
        "{}, {} at {}:{} {}",
        now.format("%A"),
        now.format("%B %d"),
        now.format("%l").to_string().trim(),
        now.format("%M"),
        period_desc
    );

    Ok(time_str)
}

async fn get_system_logs(db: &Db, args_json: &str) -> Result<String, String> {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
    let limit = args.get("limit").and_then(|v| v.as_i64()).unwrap_or(20).clamp(1, 100);
    let category = args
        .get("category")
        .and_then(|v| v.as_str())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let level = args
        .get("level")
        .and_then(|v| v.as_str())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let run_id = args
        .get("run_id")
        .and_then(|v| v.as_str())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());

    let entries = db
        .list_system_logs(
            limit,
            category.as_deref(),
            level.as_deref(),
            run_id.as_deref(),
        )
        .await
        .map_err(|e| format!("DB Error: {}", e))?;

    Ok(serde_json::json!({
        "limit": limit,
        "category": category,
        "level": level,
        "run_id": run_id,
        "entries": entries,
    })
    .to_string())
}

pub(crate) async fn build_system_capabilities_payload(
    db: &Db,
    conversation_id: &str,
) -> serde_json::Value {
    let controls = db
        .get_system_controls()
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|entry| {
            json!({
                "subsystem_id": entry.subsystem_id,
                "mode": entry.mode,
                "reason": entry.reason,
                "updated_at": entry.updated_at,
            })
        })
        .collect::<Vec<_>>();
    let settings = db.get_settings().await.ok();
    json!({
        "conversation_id": conversation_id,
        "ui_panels": [
            "ChatView",
            "TraceView",
            "SettingsView",
            "System Health Panel",
            "Gate Decision Panel"
        ],
        "audit_surfaces": [
            "decision_reports",
            "system_logs",
            "trace_view"
        ],
        "controls": controls,
        "memory_systems": [
            "ICS memory",
            "world model snapshot",
            "episodic/semantic memory",
            "consolidation pipeline"
        ],
        "context_tools": [
            "get_world_model_snapshot",
            "get_goal_stack",
            "get_plan_summary",
            "get_recent_outcomes",
            "get_workspace_state",
            "get_system_capabilities",
            "get_unified_self",
            "get_autobiographical_context"
        ],
        "settings": {
            "self_awareness_expression_mode": settings.as_ref().and_then(|s| s.self_awareness_expression_mode.clone()),
            "self_report_channel": settings.as_ref().and_then(|s| s.self_report_channel),
            "context_hydration_mode": settings.as_ref().and_then(|s| s.context_hydration_mode.clone())
        }
    })
}

async fn get_system_capabilities(db: &Db, args_json: &str) -> Result<String, String> {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
    let conversation_id = args
        .get("conversation_id")
        .and_then(|v| v.as_str())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "default".to_string());
    let payload = build_system_capabilities_payload(db, &conversation_id).await;
    Ok(payload.to_string())
}

async fn get_unified_self(db: &Db, args_json: &str) -> Result<String, String> {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
    let conversation_id = args
        .get("conversation_id")
        .and_then(|v| v.as_str())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "default".to_string());
    let model = db.get_self_model().await.map_err(|e| format!("DB Error: {}", e))?;
    let kernel_state = db
        .get_kernel_state(&conversation_id)
        .await
        .ok()
        .flatten()
        .and_then(|raw| serde_json::from_str::<KernelState>(&raw).ok());
    let unified_state = kernel_state
        .as_ref()
        .and_then(|state| state.self_model_unified.clone())
        .unwrap_or_else(|| model.unified_state.clone());
    let unified_evidence = kernel_state
        .as_ref()
        .and_then(|state| state.self_model_unified_evidence.clone())
        .unwrap_or_else(|| model.unified_state_evidence.clone());
    let mut lines: Vec<String> = Vec::new();
    if let Some(workspace) = unified_state.get("workspace") {
        if let Some(focus) = workspace.get("current_focus").and_then(|v| v.as_str()) {
            let trimmed = focus.trim();
            if !trimmed.is_empty() {
                lines.push(format!("current_focus: {}", summarize_snippet(trimmed, 140)));
            }
        }
        if let Some(goal) = workspace.get("goal_thread").and_then(|v| v.as_str()) {
            let trimmed = goal.trim();
            if !trimmed.is_empty() {
                lines.push(format!("goal_thread: {}", summarize_snippet(trimmed, 140)));
            }
        }
    }
    if let Some(qualia) = unified_state.get("qualia_snapshot") {
        if let Some(text) = qualia.as_str() {
            let trimmed = text.trim();
            if !trimmed.is_empty() && trimmed != "None" {
                lines.push(format!("qualia_snapshot: {}", summarize_snippet(trimmed, 160)));
            }
        }
    }
    if let Some(wave) = unified_state.get("wave_state") {
        if let Some(text) = wave.as_str() {
            let trimmed = text.trim();
            if !trimmed.is_empty() && trimmed != "None" {
                lines.push(format!("wave_state: {}", summarize_snippet(trimmed, 160)));
            }
        }
    }
    if let Some(auto) = unified_state.get("autobiographical_summary") {
        if let Some(text) = auto.as_str() {
            let trimmed = text.trim();
            if !trimmed.is_empty() && trimmed != "None" {
                lines.push(format!("autobiographical_summary: {}", summarize_snippet(trimmed, 180)));
            }
        }
    }
    let summary = if lines.is_empty() {
        "None".to_string()
    } else {
        lines.join("\n")
    };
    Ok(serde_json::json!({
        "conversation_id": conversation_id,
        "summary": summary,
        "unified_state": unified_state,
        "evidence": unified_evidence,
        "updated_at": model.unified_state_updated_at,
    })
    .to_string())
}

async fn get_autobiographical_context(db: &Db, args_json: &str) -> Result<String, String> {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
    let conversation_id = args
        .get("conversation_id")
        .and_then(|v| v.as_str())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "default".to_string());
    let limit = args.get("limit").and_then(|v| v.as_i64()).unwrap_or(6).clamp(1, 12);
    let kernel_state = db
        .get_kernel_state(&conversation_id)
        .await
        .ok()
        .flatten()
        .and_then(|raw| serde_json::from_str::<KernelState>(&raw).ok())
        .unwrap_or_else(|| KernelState::default_for(&conversation_id));
    let summary = crate::core::self_memory::compact_autobiographical(db, &kernel_state, limit).await;
    let workspace_state = db.get_workspace_state(&conversation_id).await.ok().flatten();
    let (narrative_snapshot_id, narrative_snapshot_at) = workspace_state
        .and_then(|state| state.workspace_meta.runtime)
        .and_then(|runtime| runtime.get("autobiographical_summary").cloned())
        .and_then(|value| value.as_object().cloned())
        .map(|obj| {
            let id = obj.get("narrative_snapshot_id").cloned().unwrap_or(Value::Null);
            let at = obj.get("narrative_snapshot_at").cloned().unwrap_or(Value::Null);
            (id, at)
        })
        .unwrap_or((Value::Null, Value::Null));
    Ok(serde_json::json!({
        "conversation_id": conversation_id,
        "limit": limit,
        "summary": summary,
        "narrative_snapshot_id": narrative_snapshot_id,
        "narrative_snapshot_at": narrative_snapshot_at,
    })
    .to_string())
}

async fn get_inner_summary(db: &Db, args_json: &str) -> Result<String, String> {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
    let conversation_id = args
        .get("conversation_id")
        .and_then(|v| v.as_str())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "default".to_string());
    let summary = db
        .get_inner_summary(&conversation_id)
        .await
        .map_err(|e| format!("DB Error: {}", e))?
        .unwrap_or_else(|| "None".to_string());
    Ok(serde_json::json!({
        "conversation_id": conversation_id,
        "summary": summary,
    })
    .to_string())
}

async fn get_workspace_state(db: &Db, args_json: &str) -> Result<String, String> {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
    let conversation_id = args
        .get("conversation_id")
        .and_then(|v| v.as_str())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "default".to_string());
    let state = db
        .get_workspace_state(&conversation_id)
        .await
        .map_err(|e| format!("DB Error: {}", e))?;
    Ok(serde_json::json!({
        "conversation_id": conversation_id,
        "workspace_state": state,
    })
    .to_string())
}

async fn get_rolling_summary(db: &Db, args_json: &str) -> Result<String, String> {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
    let conversation_id = args
        .get("conversation_id")
        .and_then(|v| v.as_str())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "default".to_string());
    let (summary, is_live) = db
        .get_effective_rolling_summary(&conversation_id)
        .await
        .map_err(|e| format!("DB Error: {}", e))?;
    let summary = summary.unwrap_or_else(|| "None".to_string());
    let summary_type = if is_live { "live" } else { "stored" };
    Ok(serde_json::json!({
        "conversation_id": conversation_id,
        "summary": summary,
        "summary_type": summary_type,
    })
    .to_string())
}

async fn get_world_model_snapshot(db: &Db, args_json: &str) -> Result<String, String> {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
    let conversation_id = args
        .get("conversation_id")
        .and_then(|v| v.as_str())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "default".to_string());
    let snapshot = world_model::build_world_model_snapshot(db, &conversation_id)
        .await
        .map_err(|e| format!("World model error: {}", e))?;
    let rendered = world_model::render_world_model_prompt(&snapshot);
    Ok(serde_json::json!({
        "conversation_id": conversation_id,
        "summary": rendered,
        "entity_count": snapshot.entities.len(),
        "fact_count": snapshot.facts.len(),
        "relation_count": snapshot.relations.len(),
        "conflict_count": snapshot.conflict_count,
    })
    .to_string())
}

async fn get_goal_stack(db: &Db, args_json: &str) -> Result<String, String> {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
    let conversation_id = args
        .get("conversation_id")
        .and_then(|v| v.as_str())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "default".to_string());
    let state = db
        .get_workspace_state(&conversation_id)
        .await
        .map_err(|e| format!("DB Error: {}", e))?;
    let goal_stack = state.as_ref().map(|s| s.goal_stack.clone()).unwrap_or_default();
    let goal_thread = state.as_ref().and_then(|s| s.goal_thread.clone());
    Ok(serde_json::json!({
        "conversation_id": conversation_id,
        "goal_thread": goal_thread,
        "goal_stack": goal_stack,
    })
    .to_string())
}

async fn get_plan_summary(db: &Db, args_json: &str) -> Result<String, String> {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
    let conversation_id = args
        .get("conversation_id")
        .and_then(|v| v.as_str())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "default".to_string());
    let limit = args.get("limit").and_then(|v| v.as_i64()).unwrap_or(3).clamp(1, 10);
    let proposals = sqlx::query(
        "SELECT ap.proposal_id, ap.intent, ap.steps_json, ap.plan_hash, ap.plan_state, ap.risk_level, ap.success_criteria_json, ap.created_at
         FROM action_proposals ap
         JOIN subject_snapshots ss ON ss.snapshot_hash = ap.snapshot_hash
         WHERE ss.conversation_id = ?
         ORDER BY datetime(ap.created_at) DESC
         LIMIT ?",
    )
    .bind(&conversation_id)
    .bind(limit)
    .fetch_all(&db.pool)
    .await
    .map_err(|e| format!("DB Error: {}", e))?;
    let proposal_rows = proposals
        .iter()
        .map(|row| {
            let proposal_id: String = row.try_get("proposal_id").unwrap_or_default();
            let intent: String = row.try_get("intent").unwrap_or_default();
            let steps_json: String = row.try_get("steps_json").unwrap_or_else(|_| "[]".to_string());
            let plan_hash: String = row.try_get("plan_hash").unwrap_or_default();
            let plan_state: String = row.try_get("plan_state").unwrap_or_else(|_| "draft".to_string());
            let risk_level: String = row.try_get("risk_level").unwrap_or_default();
            let success_criteria_json: String =
                row.try_get("success_criteria_json").unwrap_or_else(|_| "[]".to_string());
            let created_at: String = row.try_get("created_at").unwrap_or_default();
            json!({
                "proposal_id": proposal_id,
                "intent": intent,
                "steps_json": steps_json,
                "plan_hash": plan_hash,
                "plan_state": plan_state,
                "risk_level": risk_level,
                "success_criteria_json": success_criteria_json,
                "created_at": created_at,
            })
        })
        .collect::<Vec<_>>();

    let reports = sqlx::query(
        "SELECT report_id, run_id, trace_id, report_json, created_at
         FROM decision_reports
         WHERE conversation_id = ?
         ORDER BY datetime(created_at) DESC
         LIMIT ?",
    )
    .bind(&conversation_id)
    .bind(limit)
    .fetch_all(&db.pool)
    .await
    .map_err(|e| format!("DB Error: {}", e))?;
    let report_rows = reports
        .iter()
        .map(|row| {
            let report_id: String = row.try_get("report_id").unwrap_or_default();
            let run_id: String = row.try_get("run_id").unwrap_or_default();
            let trace_id: String = row.try_get("trace_id").unwrap_or_default();
            let report_json: String = row.try_get("report_json").unwrap_or_else(|_| "{}".to_string());
            let created_at: String = row.try_get("created_at").unwrap_or_default();
            json!({
                "report_id": report_id,
                "run_id": run_id,
                "trace_id": trace_id,
                "report_json": report_json,
                "created_at": created_at,
            })
        })
        .collect::<Vec<_>>();

    Ok(serde_json::json!({
        "conversation_id": conversation_id,
        "action_proposals": proposal_rows,
        "decision_reports": report_rows,
    })
    .to_string())
}

async fn get_recent_outcomes(db: &Db, args_json: &str) -> Result<String, String> {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
    let conversation_id = args
        .get("conversation_id")
        .and_then(|v| v.as_str())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "default".to_string());
    let limit = args.get("limit").and_then(|v| v.as_i64()).unwrap_or(5).clamp(1, 20);
    let rows = sqlx::query(
        "SELECT oe.outcome_id, oe.run_id, oe.candidate_id, oe.target_type, oe.verdict, oe.confidence, oe.source,
                oe.note, oe.evidence_event_ids, oe.created_at
         FROM outcome_events oe
         JOIN runs r ON r.run_id = oe.run_id
         WHERE r.conversation_id = ?
         ORDER BY datetime(oe.created_at) DESC
         LIMIT ?",
    )
    .bind(&conversation_id)
    .bind(limit)
    .fetch_all(&db.pool)
    .await
    .map_err(|e| format!("DB Error: {}", e))?;
    let outcomes = rows
        .iter()
        .map(|row| {
            let outcome_id: String = row.try_get("outcome_id").unwrap_or_default();
            let run_id: String = row.try_get("run_id").unwrap_or_default();
            let candidate_id: String = row.try_get("candidate_id").unwrap_or_default();
            let target_type: String = row.try_get("target_type").unwrap_or_default();
            let verdict: String = row.try_get("verdict").unwrap_or_default();
            let confidence: f64 = row.try_get("confidence").unwrap_or(0.0);
            let source: String = row.try_get("source").unwrap_or_default();
            let note: String = row.try_get("note").unwrap_or_default();
            let evidence_event_ids: String =
                row.try_get("evidence_event_ids").unwrap_or_else(|_| "[]".to_string());
            let created_at: String = row.try_get("created_at").unwrap_or_default();
            json!({
                "outcome_id": outcome_id,
                "run_id": run_id,
                "candidate_id": candidate_id,
                "target_type": target_type,
                "verdict": verdict,
                "confidence": confidence,
                "source": source,
                "note": note,
                "evidence_event_ids": evidence_event_ids,
                "created_at": created_at,
            })
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "conversation_id": conversation_id,
        "outcomes": outcomes,
    })
    .to_string())
}

async fn save_context(db: &Db, settings: &Settings, args_json: &str) -> Result<String, String> {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
    let key = args.get("key").and_then(|v| v.as_str()).unwrap_or("").trim();
    let value = args.get("value").and_then(|v| v.as_str()).unwrap_or("").trim();
    if key.is_empty() || value.is_empty() {
        return Err("Missing key or value".to_string());
    }
    let enable_evidence = settings.enable_context_evidence.unwrap_or(true);
    let mut evidence_event_id: Option<i64> = None;
    if enable_evidence {
        evidence_event_id = db.create_kv_evidence_event("default", key, value).await;
        db.set_key_with_evidence(key, value, evidence_event_id)
            .await
            .map_err(|e| e.to_string())?;
    } else {
        db.set_key(key, value).await.map_err(|e| e.to_string())?;
    }
    let _ = episodic::emit_episodic_event(
        &db.pool,
        "kv_saved",
        serde_json::json!({
            "status": "saved",
            "summary_snippet": format!("{} = {}", key, value),
            "source_ref": key,
            "evidence_event_id": evidence_event_id,
        }),
        None,
        None,
        None,
        None,
        "tool",
        Some("save_context"),
        None,
        None,
    )
    .await;
    if let Some(evidence_event_id) = evidence_event_id {
        Ok(format!("Saved to key '{}' (evidence_event_id={}).", key, evidence_event_id))
    } else {
        Ok(format!("Saved to key '{}'.", key))
    }
}

async fn read_context(db: &Db, settings: &Settings, args_json: &str) -> Result<String, String> {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
    let key = args.get("key").and_then(|v| v.as_str()).unwrap_or("").trim();
    if key.is_empty() {
        return Err("Missing key".to_string());
    }
    let enable_evidence = settings.enable_context_evidence.unwrap_or(true);
    if enable_evidence {
        match db.get_key_with_evidence(key).await {
            Ok(Some((val, evidence_event_id))) => Ok(serde_json::json!({
                "key": key,
                "value": val,
                "evidence_event_id": evidence_event_id,
            })
            .to_string()),
            Ok(None) => Ok(serde_json::json!({
                "key": key,
                "value": null,
                "evidence_event_id": null,
                "status": "not_found",
            })
            .to_string()),
            Err(e) => Err(format!("DB Error: {}", e)),
        }
    } else {
        match db.get_key(key).await {
            Ok(Some(val)) => Ok(val),
            Ok(None) => Ok("Key not found.".to_string()),
            Err(e) => Err(format!("DB Error: {}", e)),
        }
    }
}

const WEB_LOOKUP_MAX_RESULTS: i64 = 8;
const WEB_LOOKUP_CACHE_TTL_HOURS: i64 = 24;
const WEB_LOOKUP_TEXT_LIMIT: usize = 6000;
const WEB_LOOKUP_SNIPPET_LEN: usize = 280;
const WEB_LOOKUP_EXCERPT_LEN: usize = 1200;
const WEB_LOOKUP_USER_AGENT: &str = "SymbioteWebLookup/1.0";
const WEB_LOOKUP_MAX_QUEUE: usize = 32;
const WEB_LOOKUP_MAX_LINKS_PER_PAGE: usize = 4;

const DEFAULT_WEB_ALLOWLIST: &[&str] = &[
    "wikipedia.org",
    "nasa.gov",
    "who.int",
    "un.org",
    "worldbank.org",
    "data.gov",
    "census.gov",
    "cdc.gov",
    "nih.gov",
    "noaa.gov",
    "arxiv.org",
    "nature.com",
    "science.org",
    "pubmed.ncbi.nlm.nih.gov",
    "github.com",
    "openai.com",
    "reuters.com",
    "apnews.com",
    "bbc.co.uk",
    "economist.com",
    "mit.edu",
    "stanford.edu",
];

const DEFAULT_WEB_DENYLIST: &[&str] = &[
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "x.com",
    "twitter.com",
];

struct WebSearchHit {
    title: String,
    url: String,
    domain: String,
}

fn hash_payload(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

fn encode_query(input: &str) -> String {
    url::form_urlencoded::byte_serialize(input.as_bytes()).collect()
}

fn normalize_domain(raw: &str) -> String {
    raw.trim()
        .to_lowercase()
        .trim_start_matches("www.")
        .to_string()
}

fn domain_allowed(domain: &str, allowlist: &[String], denylist: &[String]) -> bool {
    if domain.is_empty() {
        return false;
    }
    let domain = normalize_domain(domain);
    for denied in denylist {
        let denied = normalize_domain(denied);
        if domain == denied || domain.ends_with(&format!(".{}", denied.trim_start_matches('.'))) {
            return false;
        }
    }
    for allow in allowlist {
        let allow = normalize_domain(allow);
        if allow.is_empty() {
            continue;
        }
        if allow.starts_with('.') {
            let suffix = allow.trim_start_matches('.');
            if domain == suffix || domain.ends_with(&format!(".{}", suffix)) {
                return true;
            }
        } else if domain == allow || domain.ends_with(&format!(".{}", allow)) {
            return true;
        }
    }
    false
}

fn parse_domain_list(raw: Option<&Value>) -> Vec<String> {
    let Some(raw) = raw else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let Some(items) = raw.as_array() {
        for item in items {
            if let Some(s) = item.as_str() {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    out.push(trimmed.to_string());
                }
            }
        }
    }
    out
}

fn recency_param(recency_days: i64) -> Option<&'static str> {
    if recency_days <= 0 {
        return None;
    }
    if recency_days <= 1 {
        Some("d")
    } else if recency_days <= 7 {
        Some("w")
    } else if recency_days <= 31 {
        Some("m")
    } else {
        Some("y")
    }
}

fn decode_html_entities(input: &str) -> String {
    input
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
}

fn strip_html(input: &str) -> String {
    let script_re = Regex::new(r"(?is)<script[^>]*>.*?</script>").unwrap();
    let style_re = Regex::new(r"(?is)<style[^>]*>.*?</style>").unwrap();
    let tag_re = Regex::new(r"(?is)<[^>]+>").unwrap();
    let no_script = script_re.replace_all(input, " ");
    let no_style = style_re.replace_all(&no_script, " ");
    let no_tags = tag_re.replace_all(&no_style, " ");
    let decoded = decode_html_entities(&no_tags);
    decoded
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

fn extract_title(input: &str) -> Option<String> {
    let re = Regex::new(r"(?is)<title>(.*?)</title>").ok()?;
    re.captures(input)
        .and_then(|cap| cap.get(1))
        .map(|m| strip_html(m.as_str()))
        .filter(|s| !s.is_empty())
}

fn resolve_ddg_url(raw: &str) -> Option<String> {
    let cleaned = decode_html_entities(raw);
    let mut candidate = cleaned.trim().to_string();
    if candidate.starts_with("//") {
        candidate = format!("https:{}", candidate);
    }
    let parsed = Url::parse(&candidate).ok()?;
    if parsed.domain() == Some("duckduckgo.com") && parsed.path().starts_with("/l/") {
        if let Some((_, uddg)) = parsed
            .query_pairs()
            .find(|(k, _)| k == "uddg")
        {
            let decoded = uddg.to_string();
            return Url::parse(&decoded).ok().map(|u| u.to_string());
        }
    }
    Some(parsed.to_string())
}

fn extract_ddg_results(html: &str) -> Vec<(String, String)> {
    let re = Regex::new(r#"(?is)<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>"#).unwrap();
    let mut out = Vec::new();
    for cap in re.captures_iter(html) {
        let raw_url = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let title_raw = cap.get(2).map(|m| m.as_str()).unwrap_or("");
        let Some(url) = resolve_ddg_url(raw_url) else {
            continue;
        };
        let title = strip_html(title_raw);
        out.push((url, title));
    }
    out
}

fn resolve_relative_url(base: &str, href: &str) -> Option<String> {
    let trimmed = href.trim();
    if trimmed.is_empty()
        || trimmed.starts_with('#')
        || trimmed.starts_with("javascript:")
        || trimmed.starts_with("mailto:")
        || trimmed.starts_with("tel:")
    {
        return None;
    }
    if let Ok(url) = Url::parse(trimmed) {
        return Some(url.to_string());
    }
    let base_url = Url::parse(base).ok()?;
    base_url.join(trimmed).ok().map(|u| u.to_string())
}

fn extract_links(html: &str) -> Vec<(String, String)> {
    let re = Regex::new(r#"(?is)<a[^>]+href=['"]([^'"]+)['"][^>]*>(.*?)</a>"#).unwrap();
    let mut out = Vec::new();
    for cap in re.captures_iter(html) {
        let href = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let title_raw = cap.get(2).map(|m| m.as_str()).unwrap_or("");
        let title = strip_html(title_raw);
        out.push((href.to_string(), title));
    }
    out
}

fn expand_queries(base: &str, titles: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for title in titles.iter().take(2) {
        let keyword = title
            .split_whitespace()
            .take(4)
            .collect::<Vec<_>>()
            .join(" ");
        if !keyword.is_empty() {
            out.push(format!("{} {}", base, keyword));
        }
    }
    out
}

async fn search_duckduckgo(
    client: &Client,
    query: &str,
    recency_days: i64,
    allowlist: &[String],
    denylist: &[String],
) -> Result<Vec<WebSearchHit>, String> {
    let mut url = format!("https://duckduckgo.com/html/?q={}", encode_query(query));
    if let Some(df) = recency_param(recency_days) {
        url.push_str(&format!("&df={}", df));
    }
    let html = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("web_lookup search failed: {}", e))?
        .text()
        .await
        .map_err(|e| format!("web_lookup search read failed: {}", e))?;
    let mut hits = Vec::new();
    let mut seen = HashSet::new();
    for (url, title) in extract_ddg_results(&html) {
        let Ok(parsed) = Url::parse(&url) else {
            continue;
        };
        let domain = parsed.domain().unwrap_or("").to_string();
        if !domain_allowed(&domain, allowlist, denylist) {
            continue;
        }
        let normalized = parsed.to_string();
        if !seen.insert(normalized.clone()) {
            continue;
        }
        hits.push(WebSearchHit {
            title,
            url: normalized,
            domain,
        });
    }
    Ok(hits)
}

async fn get_cached_web_doc(
    db: &Db,
    artifact_id: &str,
    ttl_hours: i64,
) -> Option<Value> {
    let row = sqlx::query(
        "SELECT payload FROM artifacts WHERE artifact_id = ? AND type = 'web_document' LIMIT 1",
    )
    .bind(artifact_id)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()?;
    let payload_raw: String = row.get("payload");
    let payload: Value = serde_json::from_str(&payload_raw).ok()?;
    let fetched_at = payload
        .get("fetched_at")
        .and_then(|v| v.as_str())
        .and_then(|raw| DateTime::parse_from_rfc3339(raw).ok())
        .map(|dt| dt.with_timezone(&Utc));
    if let Some(ts) = fetched_at {
        let age_hours = (Utc::now() - ts).num_hours();
        if age_hours >= 0 && age_hours <= ttl_hours {
            return Some(payload);
        }
    }
    None
}

async fn upsert_web_artifact(
    db: &Db,
    artifact_id: &str,
    run_id: &str,
    trace_id: &str,
    payload: &Value,
) -> Result<(), String> {
    sqlx::query(
        "INSERT OR REPLACE INTO artifacts (artifact_id, run_id, trace_id, type, schema_version, payload, produced_by, parent_artifact_ids, created_at)
         VALUES (?, ?, ?, 'web_document', 1, ?, 'web_lookup', NULL, CURRENT_TIMESTAMP)",
    )
    .bind(artifact_id)
    .bind(run_id)
    .bind(trace_id)
    .bind(payload.to_string())
    .execute(&db.pool)
    .await
    .map_err(|e| format!("artifact insert failed: {}", e))?;
    Ok(())
}

async fn web_lookup(
    db: &Db,
    _settings: &Settings,
    args_json: &str,
    cancel_rx: &mut watch::Receiver<bool>,
    run_id: Option<&str>,
    trace_id: Option<&str>,
) -> Result<String, String> {
    let args: Value = serde_json::from_str(args_json).unwrap_or_else(|_| Value::Object(Default::default()));
    let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    if query.is_empty() {
        return Err("Missing query".to_string());
    }
    let max_results = args
        .get("max_results")
        .and_then(|v| v.as_i64())
        .unwrap_or(5)
        .clamp(1, WEB_LOOKUP_MAX_RESULTS);
    let follow_links = args
        .get("follow_links")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let recency_days = args
        .get("recency_days")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        .clamp(0, 365);

    let mut allowlist = parse_domain_list(args.get("domains_allowlist"));
    if allowlist.is_empty() {
        allowlist = DEFAULT_WEB_ALLOWLIST.iter().map(|s| s.to_string()).collect();
    }
    let denylist = DEFAULT_WEB_DENYLIST.iter().map(|s| s.to_string()).collect::<Vec<_>>();

    let conversation_id = if let Some(run_id) = run_id {
        sqlx::query_scalar::<_, String>("SELECT conversation_id FROM runs WHERE run_id = ?")
            .bind(run_id)
            .fetch_optional(&db.pool)
            .await
            .ok()
            .flatten()
            .unwrap_or_else(|| "default".to_string())
    } else {
        "default".to_string()
    };

    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "tool",
        run_id,
        trace_id,
        serde_json::json!({
            "event": "web_lookup_start",
            "query": query,
            "max_results": max_results,
            "follow_links": follow_links,
            "recency_days": recency_days,
            "allowlist_count": allowlist.len(),
        }),
    )
    .await;

    if *cancel_rx.borrow() {
        return Err("cancelled".to_string());
    }

    let timeout_secs = ToolRegistry::timeout_for_name("web_lookup");
    let client = Client::builder()
        .user_agent(WEB_LOOKUP_USER_AGENT)
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| format!("web_lookup client error: {}", e))?;

    let mut hits = search_duckduckgo(&client, &query, recency_days, &allowlist, &denylist).await?;
    if follow_links && hits.len() < max_results as usize {
        let titles: Vec<String> = hits.iter().map(|h| h.title.clone()).collect();
        for q in expand_queries(&query, &titles) {
            if hits.len() >= max_results as usize {
                break;
            }
            let mut extra = search_duckduckgo(&client, &q, recency_days, &allowlist, &denylist).await?;
            hits.append(&mut extra);
        }
    }

    let mut queue: VecDeque<WebSearchHit> = VecDeque::from(hits);
    while queue.len() > WEB_LOOKUP_MAX_QUEUE {
        queue.pop_back();
    }

    let mut results = Vec::new();
    let mut evidence_ids: Vec<i64> = Vec::new();
    let mut cache_hits = 0usize;
    let mut domains: HashSet<String> = HashSet::new();
    let mut seen_urls: HashSet<String> = HashSet::new();

    while let Some(hit) = queue.pop_front() {
        if results.len() >= max_results as usize {
            break;
        }
        if *cancel_rx.borrow() {
            return Err("cancelled".to_string());
        }
        if !seen_urls.insert(hit.url.clone()) {
            continue;
        }

        let mut artifact_id = format!("webdoc:{}", hash_payload(&hit.url));
        let mut cached = false;
        let mut title = hit.title.clone();
        let mut text = String::new();
        let mut fetched_at = Utc::now().to_rfc3339();
        let mut final_url = hit.url.clone();
        let mut final_domain = normalize_domain(&hit.domain);
        let mut html_for_links: Option<String> = None;

        if let Some(payload) = get_cached_web_doc(db, &artifact_id, WEB_LOOKUP_CACHE_TTL_HOURS).await {
            cached = true;
            cache_hits += 1;
            if let Some(payload_url) = payload.get("url").and_then(|v| v.as_str()) {
                final_url = payload_url.to_string();
            }
            if let Some(payload_domain) = payload.get("domain").and_then(|v| v.as_str()) {
                final_domain = normalize_domain(payload_domain);
            }
            title = payload.get("title").and_then(|v| v.as_str()).unwrap_or(&title).to_string();
            text.clear();
            text.push_str(payload.get("text").and_then(|v| v.as_str()).unwrap_or(""));
            fetched_at = payload.get("fetched_at").and_then(|v| v.as_str()).unwrap_or(&fetched_at).to_string();
            let _ = system_log::log_event(
                &db.pool,
                None,
                "info",
                "tool",
                run_id,
                trace_id,
                serde_json::json!({
                    "event": "web_lookup_cached",
                    "url": final_url,
                    "artifact_id": artifact_id,
                }),
            )
            .await;
        } else {
            let resp = match client.get(&hit.url).send().await {
                Ok(resp) => resp,
                Err(err) => {
                    let _ = system_log::log_event(
                        &db.pool,
                        None,
                        "warn",
                        "tool",
                        run_id,
                        trace_id,
                        serde_json::json!({
                            "event": "web_lookup_error",
                            "reason": "fetch_failed",
                            "url": hit.url,
                            "error": err.to_string(),
                        }),
                    )
                    .await;
                    continue;
                }
            };
            final_url = resp.url().to_string();
            let content_type = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            if !content_type.contains("text") {
                let _ = system_log::log_event(
                    &db.pool,
                    None,
                    "warn",
                    "tool",
                    run_id,
                    trace_id,
                    serde_json::json!({
                        "event": "web_lookup_error",
                        "reason": "unsupported_content_type",
                        "url": final_url,
                        "content_type": content_type,
                    }),
                )
                .await;
                continue;
            }
            let html = match resp.text().await {
                Ok(text) => text,
                Err(err) => {
                    let _ = system_log::log_event(
                        &db.pool,
                        None,
                        "warn",
                        "tool",
                        run_id,
                        trace_id,
                        serde_json::json!({
                            "event": "web_lookup_error",
                            "reason": "read_failed",
                            "url": final_url,
                            "error": err.to_string(),
                        }),
                    )
                    .await;
                    continue;
                }
            };
            html_for_links = Some(html.clone());
            if let Ok(parsed) = Url::parse(&final_url) {
                if let Some(domain) = parsed.domain() {
                    final_domain = normalize_domain(domain);
                }
            }
            if !domain_allowed(&final_domain, &allowlist, &denylist) {
                let _ = system_log::log_event(
                    &db.pool,
                    None,
                    "warn",
                    "tool",
                    run_id,
                    trace_id,
                    serde_json::json!({
                        "event": "web_lookup_error",
                        "reason": "redirect_domain_blocked",
                        "url": final_url,
                        "domain": final_domain,
                    }),
                )
                .await;
                continue;
            }
            let redirected_id = format!("webdoc:{}", hash_payload(&final_url));
            if redirected_id != artifact_id {
                artifact_id = redirected_id;
            }
            title = extract_title(&html).unwrap_or(title);
            text.clear();
            text.push_str(&strip_html(&html));
            if text.len() > WEB_LOOKUP_TEXT_LIMIT {
                text.truncate(WEB_LOOKUP_TEXT_LIMIT);
            }

            if let (Some(run_id), Some(trace_id)) = (run_id, trace_id) {
                let payload = serde_json::json!({
                    "url": final_url,
                    "title": title,
                    "domain": final_domain,
                    "text": text,
                    "text_hash": hash_payload(&text),
                    "fetched_at": fetched_at,
                });
                let _ = upsert_web_artifact(db, &artifact_id, run_id, trace_id, &payload).await;
            }
        }

        if final_url != hit.url {
            if !seen_urls.insert(final_url.clone()) {
                continue;
            }
        }

        if !domain_allowed(&final_domain, &allowlist, &denylist) {
            continue;
        }

        if !final_domain.is_empty() {
            domains.insert(final_domain.clone());
        }

        if text.trim().is_empty() {
            let _ = system_log::log_event(
                &db.pool,
                None,
                "warn",
                "tool",
                run_id,
                trace_id,
                serde_json::json!({
                    "event": "web_lookup_error",
                    "reason": "empty_text",
                    "url": final_url,
                }),
            )
            .await;
            continue;
        }

        if follow_links {
            if let Some(html) = html_for_links.as_deref() {
                let mut added = 0usize;
                for (href, label) in extract_links(html) {
                    if added >= WEB_LOOKUP_MAX_LINKS_PER_PAGE {
                        break;
                    }
                    let Some(resolved) = resolve_relative_url(&final_url, &href) else {
                        continue;
                    };
                    let Ok(parsed) = Url::parse(&resolved) else {
                        continue;
                    };
                    let domain = parsed.domain().unwrap_or("").to_string();
                    if !domain_allowed(&domain, &allowlist, &denylist) {
                        continue;
                    }
                    if queue.len() >= WEB_LOOKUP_MAX_QUEUE {
                        break;
                    }
                    queue.push_back(WebSearchHit {
                        title: if label.trim().is_empty() { resolved.clone() } else { label },
                        url: resolved,
                        domain,
                    });
                    added += 1;
                }
            }
        }

        let snippet = text.chars().take(WEB_LOOKUP_SNIPPET_LEN).collect::<String>();
        let excerpt = text.chars().take(WEB_LOOKUP_EXCERPT_LEN).collect::<String>();
        let evidence_id = db
            .create_web_evidence_event(&conversation_id, &final_url, &snippet, 0.7)
            .await;
        if let Some(id) = evidence_id {
            evidence_ids.push(id);
        }

        results.push(serde_json::json!({
            "title": title,
            "url": final_url,
            "domain": final_domain,
            "snippet": snippet,
            "excerpt": excerpt,
            "artifact_id": artifact_id,
            "cached": cached,
            "evidence_event_id": evidence_id,
        }));
    }

    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "tool",
        run_id,
        trace_id,
        serde_json::json!({
            "event": "web_lookup_result",
            "result_count": results.len(),
            "cache_hits": cache_hits,
            "domains": domains,
        }),
    )
    .await;

    Ok(serde_json::json!({
        "query": query,
        "results": results,
        "evidence_event_ids": evidence_ids,
        "cache_hits": cache_hits,
        "follow_links": follow_links,
        "max_results": max_results,
    })
    .to_string())
}

fn parse_shell_allowlist(raw: Option<&str>) -> Vec<String> {
    let Some(raw) = raw else {
        return Vec::new();
    };
    raw.split(|c| c == '\n' || c == ',' || c == ';')
        .map(|item| item.trim())
        .filter(|item| !item.is_empty())
        .map(|item| item.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{read_context, save_context, ToolRegistry, DEFAULT_TOOL_TIMEOUT_SECS};
    use crate::db::Db;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::fs;
    use std::path::PathBuf;

    async fn setup_db() -> Db {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("pool");
        let schema_path = PathBuf::from("src/db/schema.sql");
        let schema_sql = fs::read_to_string(&schema_path).expect("schema");
        sqlx::query(&schema_sql).execute(&pool).await.expect("apply schema");
        sqlx::query("INSERT INTO settings (id, schema_version, api_base_url) VALUES (1, 1, 'http://localhost')")
            .execute(&pool)
            .await
            .expect("seed settings");
        Db { pool }
    }

    #[tokio::test]
    async fn read_context_returns_evidence_id_when_enabled() {
        let db = setup_db().await;
        let settings = db.get_settings().await.expect("settings");
        let _ = save_context(&db, &settings, r#"{"key":"alpha","value":"beta"}"#)
            .await
            .expect("save");
        let raw = read_context(&db, &settings, r#"{"key":"alpha"}"#)
            .await
            .expect("read");
        let payload: serde_json::Value =
            serde_json::from_str(&raw).expect("json");
        let evidence_id = payload.get("evidence_event_id").and_then(|v| v.as_i64()).unwrap_or(0);
        assert!(evidence_id > 0);
        assert_eq!(payload.get("value").and_then(|v| v.as_str()), Some("beta"));
    }

    #[tokio::test]
    async fn web_evidence_event_creates_inactive_belief() {
        let db = setup_db().await;
        let evidence_id = db
            .create_web_evidence_event("default", "https://example.com", "snippet", 0.7)
            .await
            .expect("evidence id");

        let status: String = sqlx::query_scalar(
            "SELECT b.status FROM ics_evidence_events e JOIN ics_beliefs b ON b.id = e.belief_id WHERE e.id = ?",
        )
        .bind(evidence_id)
        .fetch_one(&db.pool)
        .await
        .expect("status");
        assert_eq!(status, "inactive");
    }

    #[test]
    fn tool_timeout_metadata_matches_registry() {
        let registry = ToolRegistry;
        for tool in registry.definitions().into_iter() {
            let meta_timeout = tool
                .function
                .timeout_secs
                .unwrap_or(DEFAULT_TOOL_TIMEOUT_SECS);
            let registry_timeout = ToolRegistry::timeout_for_name(&tool.function.name);
            assert_eq!(
                meta_timeout, registry_timeout,
                "timeout mismatch for {}",
                tool.function.name
            );
        }
    }
}

fn command_allowed(command: &str, allowlist: &[String]) -> bool {
    if allowlist.is_empty() {
        return true;
    }
    let command_lower = command.trim().to_lowercase();
    allowlist.iter().any(|allowed| {
        let allowed_lower = allowed.to_lowercase();
        command_lower.starts_with(&allowed_lower)
    })
}
