pub mod model_client;
pub mod prompt_builder;
pub mod tool_registry;
pub mod tool_args;
pub mod system_log;
pub mod system_log_schema;
pub mod system_controls;
pub mod system_health;
pub mod outcome_taxonomy;
pub mod sensitivity;
pub mod identity;
pub mod kernel;
pub mod inner_summary;
pub mod prompt_loader;
pub mod input_resolution;
pub mod feedback;
pub mod voice_manager_v2;
pub mod scheduler;
pub mod memory;
pub mod self_memory;
pub mod self_reflection;
pub mod self_model_controller;
pub mod self_claims;
pub mod reminder_blocks;
pub mod rolling_summary;
pub mod episodic;
pub mod memory_policy;
pub mod cognitive_checks;
pub mod token_estimator;
pub mod run_phase;
pub mod post_processing;
pub mod subject_state;
pub mod workspace;
pub mod organism;
pub mod attention_model;
pub mod attention_schema;
pub mod qualia;
pub mod subject_controller;
pub mod cognitive_wave;
pub mod cognitive_wave_projection;
pub mod internal_state_map;
pub mod telemetry_calibration;
pub mod world_model;
pub mod world_model_reconcile;

use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::{Mutex, watch};
use tauri::{AppHandle, Emitter};
use crate::db::Db;
use crate::models::{Message, Run};
#[cfg(test)]
use crate::models::WorkspaceState;
use uuid::Uuid;
use chrono::Utc;
#[cfg(test)]
use chrono::{DateTime, NaiveDateTime};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use self::model_client::ModelClient;
use self::kernel::Kernel;
use crate::core::run_phase::{advance_run_phase, RunPhase};

/// Stores state when execution is paused waiting for user clarification
#[derive(Clone)]
pub struct PendingClarification {
    pub run_id: String,
    pub node_id: String,
    pub original_input: String,
    pub accumulated_context: String,
}

#[cfg(test)]
const PROMPT_OVERLAP_THRESHOLD: usize = 3;
#[cfg(test)]
const PROMPT_STOPWORDS: [&str; 22] = [
    "the", "and", "for", "with", "this", "that", "from", "what", "when", "where", "have", "your",
    "you", "are", "was", "were", "will", "can", "should", "would", "could", "about",
];

#[cfg(test)]
fn tokenize_for_overlap(text: &str) -> HashSet<String> {
    let mut tokens = HashSet::new();
    for token in text
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 3)
    {
        if PROMPT_STOPWORDS.contains(&token) {
            continue;
        }
        tokens.insert(token.to_string());
    }
    tokens
}

#[cfg(test)]
fn count_overlap(a: &HashSet<String>, b: &HashSet<String>) -> usize {
    a.intersection(b).count()
}

fn message_hash(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn safe_snippet(input: &str, max_len: usize) -> String {
    if max_len == 0 {
        return String::new();
    }
    let cleaned = input.replace('\n', " ").replace('\r', " ");
    cleaned.chars().take(max_len).collect()
}

fn normalize_transcript_input(input: &str, labels: &[String]) -> (String, bool, Vec<String>) {
    if input.trim().is_empty() {
        return (input.to_string(), false, Vec::new());
    }
    let mut normalized_lines: Vec<String> = Vec::new();
    let mut removed_labels: Vec<String> = Vec::new();
    let mut normalized = false;
    let mut label_set: HashSet<String> = labels
        .iter()
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    label_set.extend(
        [
            "user",
            "assistant",
            "system",
            "self-a",
            "self-b",
            "self a",
            "self b",
            "ai",
            "ergo",
            "ken",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    for line in input.lines() {
        let trimmed = line.trim_start();
        let mut matched_label: Option<String> = None;
        let mut remainder = trimmed;
        for label in label_set.iter() {
            let needle = format!("{}:", label);
            if trimmed.to_lowercase().starts_with(&needle) {
                matched_label = Some(label.clone());
                remainder = trimmed[needle.len()..].trim_start();
                break;
            }
        }
        if let Some(label) = matched_label {
            normalized = true;
            removed_labels.push(label);
            normalized_lines.push(remainder.to_string());
        } else {
            normalized_lines.push(line.to_string());
        }
    }
    if !normalized {
        return (input.to_string(), false, Vec::new());
    }
    let normalized_text = normalized_lines.join("\n").trim().to_string();
    (normalized_text, true, removed_labels)
}

async fn allow_empty_response_fallback(db: &Db, run_id: &str) -> bool {
    let payload: Option<String> = sqlx::query_scalar(
        "SELECT payload FROM system_logs
         WHERE run_id = ?
           AND json_extract(payload, '$.event') = 'decision_report'
         ORDER BY datetime(timestamp) DESC
         LIMIT 1",
    )
    .bind(run_id)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let Some(payload) = payload else {
        return true;
    };
    let Ok(value) = serde_json::from_str::<Value>(&payload) else {
        return true;
    };
    let cannot_respond = value
        .get("cannot_respond")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    !cannot_respond
}

#[cfg(test)]
#[allow(dead_code)]
fn workspace_token_set(workspace: &WorkspaceState) -> HashSet<String> {
    let mut combined = String::new();
    if let Some(focus) = workspace.current_focus.as_deref() {
        combined.push_str(focus);
        combined.push(' ');
    }
    for topic in workspace.working_set_topics.iter() {
        combined.push_str(topic);
        combined.push(' ');
    }
    for question in workspace.open_questions.iter() {
        combined.push_str(question);
        combined.push(' ');
    }
    tokenize_for_overlap(&combined)
}

#[cfg(test)]
#[allow(dead_code)]
fn prompt_age_seconds(created_at: &str) -> Option<i64> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(created_at) {
        let now = Utc::now();
        return Some(now.signed_duration_since(dt.with_timezone(&Utc)).num_seconds());
    }
    if let Ok(naive) = NaiveDateTime::parse_from_str(created_at, "%Y-%m-%d %H:%M:%S") {
        let now = Utc::now();
        let dt = DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc);
        return Some(now.signed_duration_since(dt).num_seconds());
    }
    None
}

async fn mark_run_cancelled_if_active(
    db: &Db,
    app_handle: &AppHandle,
    run_id: &str,
    trace_id: Option<&str>,
    assistant_msg_id: &str,
    source: Option<&str>,
) -> bool {
    let run_status: Option<String> = sqlx::query_scalar("SELECT status FROM runs WHERE run_id = ?")
        .bind(run_id)
        .fetch_optional(&db.pool)
        .await
        .ok()
        .flatten();
    let message_status: Option<String> =
        sqlx::query_scalar("SELECT status FROM messages WHERE message_id = ?")
            .bind(assistant_msg_id)
            .fetch_optional(&db.pool)
            .await
            .ok()
            .flatten();

    let message_active = matches!(
        message_status.as_deref(),
        Some("streaming") | Some("pending") | Some("active")
    );
    let should_cancel = run_status.as_deref() == Some("active") && message_active;
    if should_cancel {
        let mut payload = json!({
            "event": "run_cancelled",
            "reason": "user_abort",
        });
        if let Some(src) = source {
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("source".to_string(), json!(src));
            }
        }
        let _ = system_log::log_event(
            &db.pool,
            Some(app_handle),
            "warn",
            "chat",
            Some(run_id),
            trace_id,
            payload,
        )
        .await;
        let _ = advance_run_phase(
            &db.pool,
            Some(app_handle),
            run_id,
            RunPhase::Cancelled,
            Some("user_abort"),
        )
        .await;
        let _ = db.mark_run_cancelled(run_id, "user_abort").await;
        true
    } else {
        let mut payload = json!({
            "event": "run_cancelled_ignored",
            "reason": "late_abort",
            "run_status": run_status,
            "message_status": message_status,
        });
        if let Some(src) = source {
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("source".to_string(), json!(src));
            }
        }
        let _ = system_log::log_event(
            &db.pool,
            Some(app_handle),
            "info",
            "chat",
            Some(run_id),
            trace_id,
            payload,
        )
        .await;
        false
    }
}

pub struct ChatManager {
    pub db: Arc<Db>,
    pub model_client: Arc<ModelClient>,
    pub app_handle: AppHandle,
    pub current_abort_controller: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
    pub current_cancel_tx: Arc<Mutex<Option<watch::Sender<bool>>>>,
    pub current_run_id: Arc<Mutex<Option<String>>>,
    pub kernel: Arc<Kernel>,
    pub pending_clarification: Arc<Mutex<Option<PendingClarification>>>,
}

#[derive(Clone)]
struct DeferredEmitContext {
    db: Arc<Db>,
    kernel: Arc<Kernel>,
    app_handle: AppHandle,
    current_abort_controller: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
    current_cancel_tx: Arc<Mutex<Option<watch::Sender<bool>>>>,
    current_run_id: Arc<Mutex<Option<String>>>,
}

impl Clone for ChatManager {
    fn clone(&self) -> Self {
        Self {
            db: self.db.clone(),
            model_client: self.model_client.clone(),
            app_handle: self.app_handle.clone(),
            current_abort_controller: self.current_abort_controller.clone(),
            current_cancel_tx: self.current_cancel_tx.clone(),
            current_run_id: self.current_run_id.clone(),
            kernel: self.kernel.clone(),
            pending_clarification: self.pending_clarification.clone(),
        }
    }
}

async fn process_deferred_emits_with_context(
    ctx: DeferredEmitContext,
    conversation_id: String,
) {
    let active = ctx
        .db
        .get_active_foreground_run(&conversation_id)
        .await
        .ok()
        .flatten();
    if active.is_some() {
        return;
    }
    let Some(deferred) = ctx.db.claim_deferred_emit(&conversation_id).await else {
        return;
    };
    if ctx
        .db
        .get_active_foreground_run(&conversation_id)
        .await
        .ok()
        .flatten()
        .is_some()
    {
        let _ = ctx
            .db
            .enqueue_deferred_emit(
                &conversation_id,
                &deferred.emit_kind,
                &deferred.payload_json,
                deferred.source.as_deref(),
            )
            .await;
        return;
    }
    match deferred.emit_kind.as_str() {
        "candidate_emit" => {
            if let Err(err) = ctx
                .kernel
                .run_deferred_candidate_emit(
                    &conversation_id,
                    &deferred.emit_id,
                    &deferred.payload_json,
                )
                .await
            {
                let _ = system_log::log_event(
                    &ctx.db.pool,
                    Some(&ctx.app_handle),
                    "warn",
                    "chat",
                    None,
                    None,
                    json!( {
                        "event": "proactive_release_failed",
                        "emit_id": deferred.emit_id,
                        "emit_kind": deferred.emit_kind,
                        "error": err,
                    }),
                )
                .await;
            }
        }
        "pending_prompt_send" => {
            let prompt_id = serde_json::from_str::<serde_json::Value>(&deferred.payload_json)
                .ok()
                .and_then(|value| {
                    value
                        .get("prompt_id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                })
                .unwrap_or_else(|| deferred.payload_json.clone());
            if let Err(err) = deliver_pending_prompt(&ctx.db, &ctx.app_handle, &prompt_id).await {
                let _ = system_log::log_event(
                    &ctx.db.pool,
                    Some(&ctx.app_handle),
                    "warn",
                    "chat",
                    None,
                    None,
                    json!( {
                        "event": "pending_prompt_deferred_failed",
                        "emit_id": deferred.emit_id,
                        "prompt_id": prompt_id,
                        "error": err,
                    }),
                )
                .await;
            }
        }
        "proactive_instruction" => {
            let instructions =
                serde_json::from_str::<serde_json::Value>(&deferred.payload_json)
                    .ok()
                    .and_then(|value| {
                        value
                            .get("instructions")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .unwrap_or_else(|| deferred.payload_json.clone());
            let emit_id = deferred.emit_id.clone();
            let emit_kind = deferred.emit_kind.clone();
            let ctx_for_spawn = ctx.clone();
            let conversation_id_for_spawn = conversation_id.clone();
            tokio::task::spawn_blocking(move || {
                let result = tauri::async_runtime::block_on(run_proactive_event_with_context(
                    ctx_for_spawn.clone(),
                    instructions,
                    conversation_id_for_spawn,
                ));
                let (level, payload) = match result {
                    Ok(_) => (
                        "info",
                        json!( {
                            "event": "proactive_released",
                            "emit_id": emit_id,
                            "emit_kind": emit_kind,
                        }),
                    ),
                    Err(err) => (
                        "warn",
                        json!( {
                            "event": "proactive_release_failed",
                            "emit_id": emit_id,
                            "emit_kind": emit_kind,
                            "error": err.to_string(),
                        }),
                    ),
                };
                let _ = tauri::async_runtime::block_on(system_log::log_event(
                    &ctx_for_spawn.db.pool,
                    Some(&ctx_for_spawn.app_handle),
                    level,
                    "chat",
                    None,
                    None,
                    payload,
                ));
            });
        }
        _ => {
            let _ = system_log::log_event(
                &ctx.db.pool,
                Some(&ctx.app_handle),
                "warn",
                "chat",
                None,
                None,
                json!( {
                    "event": "proactive_release_failed",
                    "emit_id": deferred.emit_id,
                    "emit_kind": deferred.emit_kind,
                    "error": "unknown_emit_kind",
                }),
            )
            .await;
        }
    }
}

pub(crate) async fn deliver_pending_prompt(
    db: &Db,
    app_handle: &AppHandle,
    prompt_id: &str,
) -> Result<(), String> {
    let Some((pending_id, prompt, source, conversation_id, _created_at, _skip_count, auto_surface, intent_kind, bridge_id, _attempt_count, _last_asked_at, _expires_at, _anchor_message_id, _anchor_hash, _anchor_created_at, _anchor_role)) = db
        .get_pending_prompt_by_id(prompt_id)
        .await
        .map_err(|e| e.to_string())?
    else {
        return Err("Pending prompt not found".to_string());
    };

    let run_id = Uuid::new_v4().to_string();
    let trace_id = run_id.clone();
    let run_metadata = json!({ "execution_mode": "pending_prompt" });
    let started_at = Utc::now();
    sqlx::query(
        "INSERT INTO runs (run_id, trace_id, conversation_id, started_at, heartbeat_at, status, metadata)
         VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&run_id)
    .bind(&trace_id)
    .bind(&conversation_id)
    .bind(started_at)
    .bind(started_at)
    .bind("active")
    .bind(run_metadata.to_string())
    .execute(&db.pool)
    .await
    .map_err(|e| e.to_string())?;

    let message_id = Uuid::new_v4().to_string();
    let origin = if auto_surface { "monologue" } else { source.as_str() };
    let surface = origin != "monologue";
    let role = if origin == "monologue" { "internal" } else { "assistant" };
    let metadata = json!({
        "proactive": true,
        "source": source,
        "pending_prompt_id": pending_id,
        "surface": surface,
        "origin": origin,
        "response_origin": "primary",
        "candidate_id": pending_id,
        "candidate_kind": intent_kind,
        "bridge_id": bridge_id,
    })
    .to_string();
    sqlx::query(
        "INSERT INTO messages (message_id, conversation_id, run_id, trace_id, role, content, status, created_at, metadata)
         VALUES (?, ?, ?, ?, ?, ?, 'complete', ?, ?)",
    )
    .bind(&message_id)
    .bind(&conversation_id)
    .bind(&run_id)
    .bind(&trace_id)
    .bind(role)
    .bind(&prompt)
    .bind(Utc::now())
    .bind(metadata)
    .execute(&db.pool)
    .await
    .map_err(|e| e.to_string())?;

    let _ = sqlx::query("UPDATE runs SET status = 'complete', ended_at = ? WHERE run_id = ?")
        .bind(Utc::now())
        .bind(&run_id)
        .execute(&db.pool)
        .await;

    let attempt_at = Utc::now().to_rfc3339();
    let _ = db.mark_pending_prompt_attempt(&pending_id, &attempt_at).await;

    match db.delete_pending_prompt(&pending_id).await {
        Ok(affected) if affected > 0 => {}
        Ok(_) => {
            let _ = system_log::log_event(
                &db.pool,
                Some(app_handle),
                "warn",
                "chat",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "pending_prompt_delete_failed",
                    "reason": "not_found",
                    "prompt_id": pending_id,
                }),
            )
            .await;
        }
        Err(err) => {
            let _ = system_log::log_event(
                &db.pool,
                Some(app_handle),
                "warn",
                "chat",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "pending_prompt_delete_failed",
                    "reason": "db_error",
                    "prompt_id": pending_id,
                    "error": err.to_string(),
                }),
            )
            .await;
        }
    }

    if surface {
        let _ = app_handle.emit("assistant_message", prompt.clone());
    }
    let _ = app_handle.emit("message_updated", ());
    if let Ok(count) = db.count_pending_prompts("default").await {
        let _ = app_handle.emit("pending_prompt_count", count as usize);
    }
    let _ = system_log::log_event(
        &db.pool,
        Some(app_handle),
        "info",
        "chat",
        Some(&run_id),
        Some(&trace_id),
        json!({
            "event": "pending_prompt_sent",
            "prompt_id": pending_id,
            "message_id": message_id,
        }),
    )
    .await;

    Ok(())
}

async fn run_proactive_event_with_context(
    ctx: DeferredEmitContext,
    instructions: String,
    conversation_id: String,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Ok(Some(active_run_id)) = ctx.db.get_active_foreground_run(&conversation_id).await {
        let payload_json = json!({ "instructions": instructions.clone() }).to_string();
        let emit_id = ctx
            .db
            .enqueue_deferred_emit(
                &conversation_id,
                "proactive_instruction",
                &payload_json,
                Some("proactive"),
            )
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let _ = system_log::log_event(
            &ctx.db.pool,
            Some(&ctx.app_handle),
            "info",
            "chat",
            None,
            None,
            json!( {
                "event": "run_supersede_blocked",
                "conversation_id": conversation_id.as_str(),
                "active_run_id": active_run_id,
                "source": "proactive",
            }),
        )
        .await;
        let _ = system_log::log_event(
            &ctx.db.pool,
            Some(&ctx.app_handle),
            "info",
            "chat",
            None,
            None,
            json!( {
                "event": "proactive_deferred",
                "emit_id": emit_id,
                "emit_kind": "proactive_instruction",
                "source": "proactive",
            }),
        )
        .await;
        return Ok(());
    }
    let run_id = Uuid::new_v4().to_string();
    let trace_id = run_id.clone();
    let run_metadata = json!({ "execution_mode": "proactive" });

    // 1. Create Run
    let run = Run {
        run_id: run_id.clone(),
        trace_id: trace_id.clone(),
        conversation_id: conversation_id.to_string(),
        started_at: Utc::now(),
        ended_at: None,
        status: "active".to_string(),
        metadata: Some(run_metadata.clone()),
    };
    sqlx::query("INSERT INTO runs (run_id, trace_id, conversation_id, started_at, heartbeat_at, status, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)")
        .bind(&run.run_id)
        .bind(&run.trace_id)
        .bind(&run.conversation_id)
        .bind(run.started_at)
        .bind(run.started_at)
        .bind(&run.status)
        .bind(run_metadata.to_string())
        .execute(&ctx.db.pool)
        .await?;

    let _ = advance_run_phase(
        &ctx.db.pool,
        Some(&ctx.app_handle),
        &run_id,
        RunPhase::Created,
        Some("run_created"),
    )
    .await;

    // 2. Create system message (hidden trigger)
    let sys_msg_id = Uuid::new_v4().to_string();
    sqlx::query("INSERT INTO messages (message_id, conversation_id, run_id, trace_id, role, content, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
        .bind(&sys_msg_id)
        .bind(&conversation_id)
        .bind(&run_id)
        .bind(&trace_id)
        .bind("system")
        .bind(&instructions)
        .bind("complete")
        .bind(Utc::now())
        .execute(&ctx.db.pool)
        .await?;

    let _ = system_log::log_event(
        &ctx.db.pool,
        Some(&ctx.app_handle),
        "info",
        "chat",
        Some(&run_id),
        Some(&trace_id),
        json!({
            "event": "message_received",
            "role": "system",
            "content_len": instructions.len(),
            "content_hash": message_hash(&instructions),
            "content_snippet": safe_snippet(&instructions, 120),
            "source": "proactive",
        }),
    )
    .await;

    // 3. Create assistant placeholder
    let assistant_msg_id = Uuid::new_v4().to_string();
    sqlx::query("INSERT INTO messages (message_id, conversation_id, run_id, trace_id, role, content, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
        .bind(&assistant_msg_id)
        .bind(&conversation_id)
        .bind(&run_id)
        .bind(&trace_id)
        .bind("assistant")
        .bind("")
        .bind("pending")
        .bind(Utc::now())
        .execute(&ctx.db.pool)
        .await?;
    let _ = ctx.app_handle.emit("message_updated", ());

    *ctx.current_run_id.lock().await = Some(run_id.clone());

    // 4. Setup Abort + Cancellation
    let (abort_tx, mut abort_rx) = tokio::sync::oneshot::channel();
    *ctx.current_abort_controller.lock().await = Some(abort_tx);
    let (cancel_tx, cancel_rx) = watch::channel(false);
    *ctx.current_cancel_tx.lock().await = Some(cancel_tx.clone());

    let app_handle = ctx.app_handle.clone();
    let db = ctx.db.clone();
    let kernel = ctx.kernel.clone();
    let deferred_ctx = ctx.clone();
    let assistant_msg_id_clone = assistant_msg_id.clone();
    let run_id_clone = run_id.clone();
    let trace_id_clone = trace_id.clone();
    let current_run_id = ctx.current_run_id.clone();
    let current_cancel_tx = ctx.current_cancel_tx.clone();
    let current_abort_controller = ctx.current_abort_controller.clone();
    let input = instructions.clone();
    let conversation_id_owned = conversation_id.to_string();

    tokio::spawn(async move {
        let cancel_state_rx = cancel_tx.subscribe();
        let run_id_for_exec = run_id_clone.clone();
        let trace_id_for_exec = trace_id_clone.clone();
        let conversation_id_for_exec = conversation_id_owned.clone();
        let conversation_id_for_release = conversation_id_owned.clone();
        let assistant_msg_id_for_exec = assistant_msg_id_clone.clone();
        let mut exec_handle = tokio::spawn(async move {
            kernel
                .run_system_input(
                    input,
                    "proactive",
                    run_id_for_exec,
                    trace_id_for_exec,
                    conversation_id_for_exec,
                    cancel_rx,
                    Some(assistant_msg_id_for_exec),
                )
                .await
        });

            let completion = tokio::select! {
                res = &mut exec_handle => Some(res),
                abort_result = &mut abort_rx => {
                    if abort_result.is_ok() {
                        let _ = system_log::log_event(
                            &db.pool,
                            Some(&app_handle),
                            "warn",
                            "chat",
                            Some(&run_id_clone),
                            Some(&trace_id_clone),
                            json!({
                                "event": "abort_signal_received",
                                "reason": "abort_signal",
                                "source": "proactive",
                            }),
                        )
                        .await;
                        let _ = cancel_tx.send(true);
                        if tokio::time::timeout(tokio::time::Duration::from_secs(2), &mut exec_handle).await.is_err() {
                            exec_handle.abort();
                        }
                        None
                    } else {
                        let _ = system_log::log_event(
                            &db.pool,
                            Some(&app_handle),
                            "warn",
                            "chat",
                            Some(&run_id_clone),
                            Some(&trace_id_clone),
                            json!({
                                "event": "abort_signal_received",
                                "reason": "abort_sender_dropped",
                                "source": "proactive",
                            }),
                        )
                        .await;
                        Some(exec_handle.await)
                    }
                }
            };

        if let Some(joined) = completion {
            let result = match joined {
                Ok(inner) => inner,
                Err(e) => Err(e.to_string()),
            };
            match result {
                Ok(run_output) => {
                    let mut final_response = run_output.response;
                    let mut suppress_surface = false;
                    if final_response.trim().is_empty() {
                        if allow_empty_response_fallback(&db, &run_id_clone).await {
                            final_response =
                                "Understood. What would you like to focus on next?".to_string();
                            let _ = system_log::log_event(
                                &db.pool,
                                Some(&app_handle),
                                "warn",
                                "chat",
                                Some(&run_id_clone),
                                Some(&trace_id_clone),
                                json!({
                                    "event": "empty_response_prevented",
                                    "message_id": assistant_msg_id_clone,
                                    "source": "proactive",
                                }),
                            )
                            .await;
                        } else {
                            suppress_surface = true;
                        }
                    }
                    let metadata = json!({
                        "source": "proactive",
                        "origin": "assistant",
                        "response_origin": if suppress_surface { "cannot_respond" } else { "primary" },
                        "surface": !suppress_surface,
                        "candidate_kind": "EmitMessage",
                        "candidate_id": assistant_msg_id_clone.clone(),
                        "bridge_id": null
                    })
                    .to_string();
                    let _ = sqlx::query("UPDATE messages SET status = 'complete', content = ?, metadata = ? WHERE message_id = ?")
                        .bind(&final_response)
                        .bind(metadata)
                        .bind(&assistant_msg_id_clone)
                        .execute(&db.pool)
                        .await;
                    let _ = sqlx::query("UPDATE runs SET status = 'complete', ended_at = ? WHERE run_id = ?")
                        .bind(Utc::now())
                        .bind(&run_id_clone)
                        .execute(&db.pool)
                        .await;

                    let _ = advance_run_phase(
                        &db.pool,
                        Some(&app_handle),
                        &run_id_clone,
                        RunPhase::Complete,
                        Some("run_completed"),
                    )
                    .await;

                    let _ = system_log::log_event(
                        &db.pool,
                        Some(&app_handle),
                        "info",
                        "chat",
                        Some(&run_id_clone),
                        Some(&trace_id_clone),
                        json!({
                            "event": "run_completed",
                            "output_len": final_response.len(),
                            "source": "proactive",
                        }),
                    )
                    .await;

                    let _ = app_handle.emit("message_updated", ());
                    process_deferred_emits_with_context(
                        deferred_ctx,
                        conversation_id_for_release,
                    )
                    .await;
                }
                Err(e) => {
                    let _ = system_log::log_event(
                        &db.pool,
                        Some(&app_handle),
                        "error",
                        "chat",
                        Some(&run_id_clone),
                        Some(&trace_id_clone),
                        json!({
                            "event": "run_error",
                            "error": e.clone(),
                            "source": "proactive",
                        }),
                    )
                    .await;

                    let _ = advance_run_phase(
                        &db.pool,
                        Some(&app_handle),
                        &run_id_clone,
                        RunPhase::Error,
                        Some("run_error"),
                    )
                    .await;

                    if *cancel_state_rx.borrow() {
                        let _ = mark_run_cancelled_if_active(
                            &db,
                            &app_handle,
                            &run_id_clone,
                            Some(&trace_id_clone),
                            &assistant_msg_id_clone,
                            Some("proactive"),
                        )
                        .await;
                    } else {
                        let fallback = "Notice: response failed to finalize. Please retry.";
                        let _ = sqlx::query(
                            "UPDATE messages
                             SET status = 'error',
                                 error = ?,
                                 content = CASE
                                     WHEN content IS NULL OR trim(content) = '' THEN ?
                                     ELSE content
                                 END
                             WHERE message_id = ?",
                        )
                        .bind(serde_json::to_string(&e).unwrap_or_default())
                        .bind(fallback)
                        .bind(&assistant_msg_id_clone)
                        .execute(&db.pool)
                        .await;

                        let _ = app_handle.emit("message_updated", ());
                    }
                }
            }
        } else {
            let _ = mark_run_cancelled_if_active(
                &db,
                &app_handle,
                &run_id_clone,
                Some(&trace_id_clone),
                &assistant_msg_id_clone,
                Some("proactive"),
            )
            .await;
        }

        let _ = app_handle.emit("message_updated", ());
        *current_run_id.lock().await = None;
        *current_cancel_tx.lock().await = None;
        *current_abort_controller.lock().await = None;
    });

    Ok(())
}

impl ChatManager {
    pub async fn new(db: Arc<Db>, model_client: Arc<ModelClient>, app_handle: AppHandle) -> Self {
        let kernel = Arc::new(Kernel::new(
            db.clone(),
            model_client.clone(),
            app_handle.clone(),
        ));

        let control_map = system_controls::load_control_map(&db).await;
        let prompt_mode = system_controls::mode_for("prompt_loader", &control_map);
        if !system_controls::mode_is_off(&prompt_mode)
            && !system_controls::mode_is_degraded(&prompt_mode)
        {
            if let Err(e) = prompt_loader::reload_prompts() {
                eprintln!("[Prompt] Load error: {}", e);
            }
        } else {
            eprintln!("[Prompt] Reload skipped (prompt_loader: {}).", prompt_mode);
        }

        Self {
            db,
            model_client,
            app_handle,
            current_abort_controller: Arc::new(Mutex::new(None)),
            current_cancel_tx: Arc::new(Mutex::new(None)),
            current_run_id: Arc::new(Mutex::new(None)),
            kernel,
            pending_clarification: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn send_message(&self, content: String) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let conversation_id = "default";
        let run_id = Uuid::new_v4().to_string();
        let trace_id = run_id.clone();
        let run_metadata = json!({ "execution_mode": "direct" });
        let raw_content = content;
        let control_map = system_controls::load_control_map(&self.db).await;
        let feedback_mode = system_controls::mode_for("feedback_loop", &control_map);
        let (content, explicit_feedback) = if system_controls::mode_is_off(&feedback_mode)
            || system_controls::mode_is_degraded(&feedback_mode)
        {
            (raw_content.clone(), false)
        } else {
            feedback::extract_explicit_feedback(&raw_content)
        };

        // 1. Create Run (supersede any existing active runs)
        let superseded = self
            .db
            .supersede_active_runs(conversation_id, &run_id, "superseded_by_new_run")
            .await?;
        if !superseded.is_empty() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "chat",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "run_superseded",
                    "superseded_by": run_id,
                    "superseded_runs": superseded,
                    "conversation_id": conversation_id,
                }),
            )
            .await;
        }

        // 1. Create Run
        let run = Run {
            run_id: run_id.clone(),
            trace_id: trace_id.clone(),
            conversation_id: conversation_id.to_string(),
            started_at: Utc::now(),
            ended_at: None,
            status: "active".to_string(),
            metadata: Some(run_metadata.clone()),
        };
        sqlx::query("INSERT INTO runs (run_id, trace_id, conversation_id, started_at, heartbeat_at, status, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)")
            .bind(&run.run_id)
            .bind(&run.trace_id)
            .bind(&run.conversation_id)
            .bind(run.started_at)
            .bind(run.started_at)
            .bind(&run.status)
            .bind(run_metadata.to_string())
            .execute(&self.db.pool)
            .await?;

        let _ = advance_run_phase(
            &self.db.pool,
            Some(&self.app_handle),
            &run_id,
            RunPhase::Created,
            Some("run_created"),
        )
        .await;

        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "chat",
            Some(&run_id),
            Some(&trace_id),
            json!({
                "event": "run_started",
                "conversation_id": conversation_id,
            }),
        )
        .await;

        // 2. Create User Message
        let user_msg_id = Uuid::new_v4().to_string();
        sqlx::query("INSERT INTO messages (message_id, conversation_id, run_id, trace_id, role, content, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
            .bind(&user_msg_id)
            .bind(conversation_id)
            .bind(&run_id)
            .bind(&trace_id)
            .bind("user")
            .bind(&content)
            .bind("complete")
            .bind(Utc::now())
            .execute(&self.db.pool)
            .await?;

        let mut user_metadata = json!({ "source": "user", "explicit_feedback": explicit_feedback });
        if let Some(evidence_id) = self
            .db
            .create_user_utterance_evidence(conversation_id, &user_msg_id, &content)
            .await
        {
            user_metadata["evidence_event_ids"] = json!([evidence_id]);
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "chat",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "user_utterance_evidence_created",
                    "evidence_event_id": evidence_id,
                    "message_id": user_msg_id,
                }),
            )
            .await;
        }
        let _ = sqlx::query("UPDATE messages SET metadata = ? WHERE message_id = ?")
            .bind(user_metadata.to_string())
            .bind(&user_msg_id)
            .execute(&self.db.pool)
            .await;

        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "chat",
            Some(&run_id),
            Some(&trace_id),
            json!({
                "event": "message_received",
                "role": "user",
                "content_len": content.len(),
                "content_hash": message_hash(&content),
                "content_snippet": safe_snippet(&content, 120),
                "explicit_feedback": explicit_feedback,
            }),
        )
        .await;

        // 3. Create Assistant Placeholder
        let assistant_msg_id = Uuid::new_v4().to_string();
        sqlx::query("INSERT INTO messages (message_id, conversation_id, run_id, trace_id, role, content, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
            .bind(&assistant_msg_id)
            .bind(conversation_id)
            .bind(&run_id)
            .bind(&trace_id)
            .bind("assistant")
            .bind("")
            .bind("pending")
            .bind(Utc::now())
            .execute(&self.db.pool)
            .await?;
        self.app_handle.emit("message_updated", ())?;

        *self.current_run_id.lock().await = Some(run_id.clone());

        // 4. Setup Abort + Cancellation
        let (abort_tx, mut abort_rx) = tokio::sync::oneshot::channel();
        *self.current_abort_controller.lock().await = Some(abort_tx);
        let (cancel_tx, cancel_rx) = watch::channel(false);
        *self.current_cancel_tx.lock().await = Some(cancel_tx.clone());

        let app_handle = self.app_handle.clone();
        let db = self.db.clone();
        let kernel = self.kernel.clone();
        let assistant_msg_id_clone = assistant_msg_id.clone();
        let run_id_clone = run_id.clone();
        let trace_id_clone = trace_id.clone();
        let current_run_id = self.current_run_id.clone();
        let current_cancel_tx = self.current_cancel_tx.clone();
        let current_abort_controller = self.current_abort_controller.clone();

        let settings = self.db.get_settings().await.ok();
        let normalize_enabled = settings
            .as_ref()
            .and_then(|s| s.stability_transcript_normalization)
            .unwrap_or(true);
        let mut label_candidates: Vec<String> = Vec::new();
        if let Some(name) = settings
            .as_ref()
            .and_then(|s| s.user_display_name.as_deref())
            .map(str::trim)
            .filter(|name| !name.is_empty())
        {
            label_candidates.push(name.to_string());
        }
        if let Some(name) = settings
            .as_ref()
            .and_then(|s| s.assistant_display_name.as_deref())
            .map(str::trim)
            .filter(|name| !name.is_empty())
        {
            label_candidates.push(name.to_string());
        }
        let (normalized_input, was_normalized, removed_labels) = if normalize_enabled {
            normalize_transcript_input(&content, &label_candidates)
        } else {
            (content.clone(), false, Vec::new())
        };
        if was_normalized {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "chat",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "transcript_normalized",
                    "input_len": content.len(),
                    "output_len": normalized_input.len(),
                    "input_hash": message_hash(&content),
                    "output_hash": message_hash(&normalized_input),
                    "labels_removed": removed_labels,
                }),
            )
            .await;
        }
        let input = normalized_input;
        let original_input = raw_content.clone();
        let conversation_id_owned = conversation_id.to_string();

        tokio::spawn(async move {
            let cancel_state_rx = cancel_tx.subscribe();
            let run_id_for_exec = run_id_clone.clone();
            let trace_id_for_exec = trace_id_clone.clone();
            let conversation_id_for_exec = conversation_id_owned.clone();
            let conversation_id_for_release = conversation_id_owned.clone();
            let assistant_msg_id_for_exec = assistant_msg_id_clone.clone();
            let kernel_for_exec = kernel.clone();
            let mut exec_handle = tokio::spawn(async move {
                kernel_for_exec
                    .run_user_input(
                        input,
                        Some(original_input),
                        run_id_for_exec,
                        trace_id_for_exec,
                        conversation_id_for_exec,
                        cancel_rx,
                        Some(assistant_msg_id_for_exec),
                    )
                    .await
            });

            let completion = tokio::select! {
                res = &mut exec_handle => Some(res),
                abort_result = &mut abort_rx => {
                    if abort_result.is_ok() {
                        let _ = system_log::log_event(
                            &db.pool,
                            Some(&app_handle),
                            "warn",
                            "chat",
                            Some(&run_id_clone),
                            Some(&trace_id_clone),
                            json!({
                                "event": "abort_signal_received",
                                "reason": "abort_signal",
                            }),
                        )
                        .await;
                        let _ = cancel_tx.send(true);
                        if tokio::time::timeout(tokio::time::Duration::from_secs(2), &mut exec_handle).await.is_err() {
                            exec_handle.abort();
                        }
                        None
                    } else {
                        let _ = system_log::log_event(
                            &db.pool,
                            Some(&app_handle),
                            "warn",
                            "chat",
                            Some(&run_id_clone),
                            Some(&trace_id_clone),
                            json!({
                                "event": "abort_signal_received",
                                "reason": "abort_sender_dropped",
                            }),
                        )
                        .await;
                        Some(exec_handle.await)
                    }
                }
            };

            if let Some(joined) = completion {
                let result = match joined {
                    Ok(inner) => inner,
                    Err(e) => Err(e.to_string()),
                };
                match result {
                    Ok(run_output) => {
                        let mut final_response = run_output.response;
                        let mut suppress_surface = false;
                        if final_response.trim().is_empty() {
                            if allow_empty_response_fallback(&db, &run_id_clone).await {
                                final_response =
                                    "Understood. What would you like to focus on next?".to_string();
                                let _ = system_log::log_event(
                                    &db.pool,
                                    Some(&app_handle),
                                    "warn",
                                    "chat",
                                    Some(&run_id_clone),
                                    Some(&trace_id_clone),
                                    json!({
                                        "event": "empty_response_prevented",
                                        "message_id": assistant_msg_id_clone,
                                    }),
                                )
                                .await;
                            } else {
                                suppress_surface = true;
                            }
                        }
                        {
                            let existing_meta: Option<String> = sqlx::query_scalar(
                                "SELECT metadata FROM messages WHERE message_id = ?",
                            )
                            .bind(&assistant_msg_id_clone)
                            .fetch_optional(&db.pool)
                            .await
                            .ok()
                            .flatten();
                            let mut meta_value = existing_meta
                                .as_deref()
                                .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
                                .unwrap_or_else(|| json!({}));
                            if !meta_value.is_object() {
                                meta_value = json!({});
                            }
                            if let Some(obj) = meta_value.as_object_mut() {
                                obj.entry("source".to_string()).or_insert(json!("assistant"));
                                obj.entry("origin".to_string()).or_insert(json!("assistant"));
                                obj.insert(
                                    "response_origin".to_string(),
                                    json!(if suppress_surface { "cannot_respond" } else { "primary" }),
                                );
                                obj.insert("surface".to_string(), json!(!suppress_surface));
                                obj.entry("candidate_kind".to_string())
                                    .or_insert(json!("EmitMessage"));
                                obj.entry("candidate_id".to_string())
                                    .or_insert(json!(&assistant_msg_id_clone));
                                obj.entry("bridge_id".to_string()).or_insert(json!(null));
                            }
                            let metadata = serde_json::to_string(&meta_value).unwrap_or_else(|_| "{}".to_string());
                            let _ = sqlx::query("UPDATE messages SET status = 'complete', content = ?, metadata = ? WHERE message_id = ?")
                                .bind(&final_response)
                                .bind(metadata)
                                .bind(&assistant_msg_id_clone)
                                .execute(&db.pool)
                                .await;
                        }
                        let _ = sqlx::query("UPDATE runs SET status = 'complete', ended_at = ? WHERE run_id = ?")
                            .bind(Utc::now())
                            .bind(&run_id_clone)
                            .execute(&db.pool)
                            .await;

        let _ = advance_run_phase(
            &db.pool,
            Some(&app_handle),
            &run_id_clone,
            RunPhase::Complete,
            Some("run_completed"),
        )
        .await;

                        let _ = system_log::log_event(
                            &db.pool,
                            Some(&app_handle),
                            "info",
                            "chat",
                            Some(&run_id_clone),
                            Some(&trace_id_clone),
                            json!({
                                "event": "run_completed",
                                "output_len": final_response.len(),
                            }),
                        )
                        .await;

                        let _ = app_handle.emit("message_updated", ());
                        let deferred_ctx = DeferredEmitContext {
                            db: db.clone(),
                            kernel: kernel.clone(),
                            app_handle: app_handle.clone(),
                            current_abort_controller: current_abort_controller.clone(),
                            current_cancel_tx: current_cancel_tx.clone(),
                            current_run_id: current_run_id.clone(),
                        };
                        process_deferred_emits_with_context(
                            deferred_ctx,
                            conversation_id_for_release,
                        )
                        .await;
                    }
                    Err(e) => {
                        let _ = system_log::log_event(
                            &db.pool,
                            Some(&app_handle),
                            "error",
                            "chat",
                            Some(&run_id_clone),
                            Some(&trace_id_clone),
                            json!({
                                "event": "run_error",
                                "error": e.clone(),
                            }),
                        )
                        .await;

        let _ = advance_run_phase(
            &db.pool,
            Some(&app_handle),
            &run_id_clone,
            RunPhase::Error,
            Some("run_error"),
        )
        .await;

                        if *cancel_state_rx.borrow() {
                            let _ = mark_run_cancelled_if_active(
                                &db,
                                &app_handle,
                                &run_id_clone,
                                Some(&trace_id_clone),
                                &assistant_msg_id_clone,
                                None,
                            )
                            .await;
                        } else {
                            let fallback = if e == "EMPTY_RESPONSE" {
                                "Notice: the model returned an empty response. Please retry."
                            } else {
                                "Notice: response failed to finalize. Please retry."
                            };
                            let _ = sqlx::query(
                                "UPDATE messages
                                 SET status = 'error',
                                     error = ?,
                                     content = CASE
                                         WHEN content IS NULL OR trim(content) = '' THEN ?
                                         ELSE content
                                     END
                                 WHERE message_id = ?",
                            )
                            .bind(serde_json::to_string(&e).unwrap_or_default())
                            .bind(fallback)
                            .bind(&assistant_msg_id_clone)
                            .execute(&db.pool)
                            .await;

                            let _ = app_handle.emit("message_updated", ());
                        }
                    }
                }
            } else {
                let _ = mark_run_cancelled_if_active(
                    &db,
                    &app_handle,
                    &run_id_clone,
                    Some(&trace_id_clone),
                    &assistant_msg_id_clone,
                    None,
                )
                .await;
            }

            let _ = app_handle.emit("message_updated", ());
            *current_run_id.lock().await = None;
            *current_cancel_tx.lock().await = None;
            *current_abort_controller.lock().await = None;
        });

        Ok(())
    }

    pub async fn abort(&self, run_id: Option<&str>, source: Option<&str>) {
        let current_run = self.current_run_id.lock().await.clone();
        let should_abort = match run_id {
            Some(target) => current_run.as_deref() == Some(target),
            None => current_run.is_some(),
        };
        if !should_abort {
            return;
        }
        let current_run_id = current_run.clone();
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "chat",
            current_run.as_deref(),
            current_run.as_deref(),
            json!({
                "event": "abort_called",
                "source": source.unwrap_or("unknown"),
                "run_id": run_id,
                "current_run_id": current_run_id,
            }),
        )
        .await;
        if let Some(cancel_tx) = self.current_cancel_tx.lock().await.as_ref() {
            let _ = cancel_tx.send(true);
        }
        if let Some(control) = self.current_abort_controller.lock().await.take() {
            let _ = control.send(());
        }
        *self.pending_clarification.lock().await = None;
    }

    pub async fn get_messages(&self) -> Result<Vec<Message>, Box<dyn std::error::Error + Send + Sync>> {
        self.db.get_history(50).await 
    }

    pub async fn clear_history(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let conversation_ids = self
            .db
            .list_conversation_ids(None)
            .await
            .unwrap_or_else(|_| vec!["default".to_string()]);
        for conversation_id in conversation_ids {
            let _ = rolling_summary::archive_rolling_summary(
                self.db.clone(),
                &conversation_id,
                "kernel",
                "summary_archive",
            )
            .await;
        }
        self.db.clear_messages().await?;
        let _ = self.app_handle.emit("message_updated", ());
        Ok(())
    }

    /// Triggers a proactive response from the agent based on a system instruction (e.g. Reminder)
    /// Triggers a proactive response from the agent based on a system instruction (e.g. Reminder)
    pub async fn trigger_proactive_event(&self, instructions: String) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let ctx = DeferredEmitContext {
            db: self.db.clone(),
            kernel: self.kernel.clone(),
            app_handle: self.app_handle.clone(),
            current_abort_controller: self.current_abort_controller.clone(),
            current_cancel_tx: self.current_cancel_tx.clone(),
            current_run_id: self.current_run_id.clone(),
        };
        run_proactive_event_with_context(ctx, instructions, "default".to_string()).await
    }

    pub async fn submit_clarification(
        &self,
        answer: String,
        original_input: String,
        original_run_id: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let conversation_id = if let Some(run_id) = original_run_id.as_deref() {
            sqlx::query_scalar::<_, String>("SELECT conversation_id FROM runs WHERE run_id = ?")
                .bind(run_id)
                .fetch_optional(&self.db.pool)
                .await
                .ok()
                .flatten()
                .unwrap_or_else(|| "default".to_string())
        } else {
            "default".to_string()
        };
        let run_id = Uuid::new_v4().to_string();
        let trace_id = run_id.clone();
        let run_metadata = json!({ "execution_mode": "direct" });

        // 1. Create Run (supersede any existing active runs)
        let superseded = self
            .db
            .supersede_active_runs(&conversation_id, &run_id, "superseded_by_clarification")
            .await?;
        if !superseded.is_empty() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "chat",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "run_superseded",
                    "superseded_by": run_id,
                    "superseded_runs": superseded,
                    "conversation_id": conversation_id,
                    "source": "clarification",
                }),
            )
            .await;
        }

        // 1. Create Run
        let run = Run {
            run_id: run_id.clone(),
            trace_id: trace_id.clone(),
            conversation_id: conversation_id.to_string(),
            started_at: Utc::now(),
            ended_at: None,
            status: "active".to_string(),
            metadata: Some(run_metadata.clone()),
        };
        sqlx::query("INSERT INTO runs (run_id, trace_id, conversation_id, started_at, heartbeat_at, status, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)")
            .bind(&run.run_id)
            .bind(&run.trace_id)
            .bind(&run.conversation_id)
            .bind(run.started_at)
            .bind(run.started_at)
            .bind(&run.status)
            .bind(run_metadata.to_string())
            .execute(&self.db.pool)
            .await?;

        let _ = advance_run_phase(
            &self.db.pool,
            Some(&self.app_handle),
            &run_id,
            RunPhase::Created,
            Some("run_created"),
        )
        .await;

        // 2. Create User Message
        let user_msg_id = Uuid::new_v4().to_string();
        sqlx::query("INSERT INTO messages (message_id, conversation_id, run_id, trace_id, role, content, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
            .bind(&user_msg_id)
            .bind(conversation_id.as_str())
            .bind(&run_id)
            .bind(&trace_id)
            .bind("user")
            .bind(&answer)
            .bind("complete")
            .bind(Utc::now())
            .execute(&self.db.pool)
            .await?;

        let mut user_metadata = json!({ "source": "user" });
        if let Some(evidence_id) = self
            .db
            .create_user_utterance_evidence(conversation_id.as_str(), &user_msg_id, &answer)
            .await
        {
            user_metadata["evidence_event_ids"] = json!([evidence_id]);
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "chat",
                Some(&run_id),
                Some(&trace_id),
                json!({
                    "event": "user_utterance_evidence_created",
                    "evidence_event_id": evidence_id,
                    "message_id": user_msg_id,
                }),
            )
            .await;
        }
        let _ = sqlx::query("UPDATE messages SET metadata = ? WHERE message_id = ?")
            .bind(user_metadata.to_string())
            .bind(&user_msg_id)
            .execute(&self.db.pool)
            .await;

        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "chat",
            Some(&run_id),
            Some(&trace_id),
            json!({
                "event": "message_received",
                "role": "user",
                "content_len": answer.len(),
                "content_hash": message_hash(&answer),
                "content_snippet": safe_snippet(&answer, 120),
                "source": "clarification",
            }),
        )
        .await;

        // 3. Build enriched input for LLM (includes original context)
        let enriched_input = format!(
            "Original question: {}

User's clarification: {}",
            original_input,
            answer
        );

        // 4. Create Assistant Placeholder
        let assistant_msg_id = Uuid::new_v4().to_string();
        sqlx::query("INSERT INTO messages (message_id, conversation_id, run_id, trace_id, role, content, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
            .bind(&assistant_msg_id)
            .bind(conversation_id.as_str())
            .bind(&run_id)
            .bind(&trace_id)
            .bind("assistant")
            .bind("")
            .bind("pending")
            .bind(Utc::now())
            .execute(&self.db.pool)
            .await?;
        self.app_handle.emit("message_updated", ())?;

        *self.current_run_id.lock().await = Some(run_id.clone());

        // 5. Setup Abort + Cancellation
        let (abort_tx, mut abort_rx) = tokio::sync::oneshot::channel();
        *self.current_abort_controller.lock().await = Some(abort_tx);
        let (cancel_tx, cancel_rx) = watch::channel(false);
        *self.current_cancel_tx.lock().await = Some(cancel_tx.clone());

        let app_handle = self.app_handle.clone();
        let db = self.db.clone();
        let kernel = self.kernel.clone();
        let assistant_msg_id_clone = assistant_msg_id.clone();
        let run_id_clone = run_id.clone();
        let trace_id_clone = trace_id.clone();
        let current_run_id = self.current_run_id.clone();
        let current_cancel_tx = self.current_cancel_tx.clone();
        let current_abort_controller = self.current_abort_controller.clone();
        let input = enriched_input.clone();
        let conversation_id_owned = conversation_id.to_string();

        tokio::spawn(async move {
            let cancel_state_rx = cancel_tx.subscribe();
            let run_id_for_exec = run_id_clone.clone();
            let trace_id_for_exec = trace_id_clone.clone();
            let conversation_id_for_exec = conversation_id_owned.clone();
            let conversation_id_for_release = conversation_id_owned.clone();
            let assistant_msg_id_for_exec = assistant_msg_id_clone.clone();
            let kernel_for_exec = kernel.clone();
            let mut exec_handle = tokio::spawn(async move {
                kernel_for_exec
                    .run_user_input(
                        input,
                        None,
                        run_id_for_exec,
                        trace_id_for_exec,
                        conversation_id_for_exec,
                        cancel_rx,
                        Some(assistant_msg_id_for_exec),
                    )
                    .await
            });

            let completion = tokio::select! {
                res = &mut exec_handle => Some(res),
                abort_result = &mut abort_rx => {
                    if abort_result.is_ok() {
                        let _ = system_log::log_event(
                            &db.pool,
                            Some(&app_handle),
                            "warn",
                            "chat",
                            Some(&run_id_clone),
                            Some(&trace_id_clone),
                            json!({
                                "event": "abort_signal_received",
                                "reason": "abort_signal",
                                "source": "clarification",
                            }),
                        )
                        .await;
                        let _ = cancel_tx.send(true);
                        if tokio::time::timeout(tokio::time::Duration::from_secs(2), &mut exec_handle).await.is_err() {
                            exec_handle.abort();
                        }
                        None
                    } else {
                        let _ = system_log::log_event(
                            &db.pool,
                            Some(&app_handle),
                            "warn",
                            "chat",
                            Some(&run_id_clone),
                            Some(&trace_id_clone),
                            json!({
                                "event": "abort_signal_received",
                                "reason": "abort_sender_dropped",
                                "source": "clarification",
                            }),
                        )
                        .await;
                        Some(exec_handle.await)
                    }
                }
            };

            if let Some(joined) = completion {
                let result = match joined {
                    Ok(inner) => inner,
                    Err(e) => Err(e.to_string()),
                };
                match result {
                    Ok(run_output) => {
                        let mut final_response = run_output.response;
                        let mut suppress_surface = false;
                        if final_response.trim().is_empty() {
                            if allow_empty_response_fallback(&db, &run_id_clone).await {
                                final_response =
                                    "Understood. What would you like to focus on next?".to_string();
                                let _ = system_log::log_event(
                                    &db.pool,
                                    Some(&app_handle),
                                    "warn",
                                    "chat",
                                    Some(&run_id_clone),
                                    Some(&trace_id_clone),
                                    json!({
                                        "event": "empty_response_prevented",
                                        "message_id": assistant_msg_id_clone,
                                        "source": "clarification",
                                    }),
                                )
                                .await;
                            } else {
                                suppress_surface = true;
                            }
                        }
                        let metadata = json!({
                            "source": "assistant",
                            "origin": "assistant",
                            "response_origin": if suppress_surface { "cannot_respond" } else { "primary" },
                            "surface": !suppress_surface,
                            "candidate_kind": "EmitMessage",
                            "candidate_id": assistant_msg_id_clone.clone(),
                            "bridge_id": null
                        })
                        .to_string();
                        let _ = sqlx::query("UPDATE messages SET status = 'complete', content = ?, metadata = ? WHERE message_id = ?")
                            .bind(&final_response)
                            .bind(metadata)
                            .bind(&assistant_msg_id_clone)
                            .execute(&db.pool)
                            .await;
                        let _ = sqlx::query("UPDATE runs SET status = 'complete', ended_at = ? WHERE run_id = ?")
                            .bind(Utc::now())
                            .bind(&run_id_clone)
                            .execute(&db.pool)
                            .await;

                        let _ = advance_run_phase(
                            &db.pool,
                            Some(&app_handle),
                            &run_id_clone,
                            RunPhase::Complete,
                            Some("run_completed"),
                        )
                        .await;

                        let _ = system_log::log_event(
                            &db.pool,
                            Some(&app_handle),
                            "info",
                            "chat",
                            Some(&run_id_clone),
                            Some(&trace_id_clone),
                            json!({
                                "event": "run_completed",
                                "output_len": final_response.len(),
                                "source": "clarification",
                            }),
                        )
                        .await;

                        let _ = app_handle.emit("message_updated", ());
                        let deferred_ctx = DeferredEmitContext {
                            db: db.clone(),
                            kernel: kernel.clone(),
                            app_handle: app_handle.clone(),
                            current_abort_controller: current_abort_controller.clone(),
                            current_cancel_tx: current_cancel_tx.clone(),
                            current_run_id: current_run_id.clone(),
                        };
                        process_deferred_emits_with_context(
                            deferred_ctx,
                            conversation_id_for_release,
                        )
                        .await;
                    }
                    Err(e) => {
                        let _ = system_log::log_event(
                            &db.pool,
                            Some(&app_handle),
                            "error",
                            "chat",
                            Some(&run_id_clone),
                            Some(&trace_id_clone),
                            json!({
                                "event": "run_error",
                                "error": e.clone(),
                                "source": "clarification",
                            }),
                        )
                        .await;

                        let _ = advance_run_phase(
                            &db.pool,
                            Some(&app_handle),
                            &run_id_clone,
                            RunPhase::Error,
                            Some("run_error"),
                        )
                        .await;

                        if *cancel_state_rx.borrow() {
                            let _ = mark_run_cancelled_if_active(
                                &db,
                                &app_handle,
                                &run_id_clone,
                                Some(&trace_id_clone),
                                &assistant_msg_id_clone,
                                Some("clarification"),
                            )
                            .await;
                        } else {
                            let fallback = "Notice: response failed to finalize. Please retry.";
                            let _ = sqlx::query(
                                "UPDATE messages
                                 SET status = 'error',
                                     error = ?,
                                     content = CASE
                                         WHEN content IS NULL OR trim(content) = '' THEN ?
                                         ELSE content
                                     END
                                 WHERE message_id = ?",
                            )
                            .bind(serde_json::to_string(&e).unwrap_or_default())
                            .bind(fallback)
                            .bind(&assistant_msg_id_clone)
                            .execute(&db.pool)
                            .await;

                            let _ = app_handle.emit("message_updated", ());
                        }
                    }
                }
            } else {
                let _ = mark_run_cancelled_if_active(
                    &db,
                    &app_handle,
                    &run_id_clone,
                    Some(&trace_id_clone),
                    &assistant_msg_id_clone,
                    Some("clarification"),
                )
                .await;
            }

            let _ = app_handle.emit("message_updated", ());
            *current_run_id.lock().await = None;
            *current_cancel_tx.lock().await = None;
            *current_abort_controller.lock().await = None;
        });

        Ok(())
    }

    pub async fn set_pending_clarification(&self, pending: PendingClarification) {
        *self.pending_clarification.lock().await = Some(pending);
    }

    pub async fn has_pending_clarification(&self) -> bool {
        self.pending_clarification.lock().await.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_prompt_alignment_accepts_overlap() {
        let prompt_tokens = tokenize_for_overlap("Review workspace focus and memory plan");
        let user_tokens = tokenize_for_overlap("We should review memory plan");
        let overlap = count_overlap(&prompt_tokens, &user_tokens);
        assert!(overlap >= PROMPT_OVERLAP_THRESHOLD);
    }

    #[test]
    fn pending_prompt_alignment_rejects_low_overlap() {
        let prompt_tokens = tokenize_for_overlap("Discuss database batching");
        let user_tokens = tokenize_for_overlap("Plan a vacation itinerary");
        let overlap = count_overlap(&prompt_tokens, &user_tokens);
        assert!(overlap < PROMPT_OVERLAP_THRESHOLD);
    }
}

