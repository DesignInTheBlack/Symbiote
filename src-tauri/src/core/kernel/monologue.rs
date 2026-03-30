use super::*;
use crate::core::system_controls;
use crate::core::self_memory;
use crate::core::cognitive_wave_projection::WaveStateVector;
use crate::core::kernel::run::build_qualia_modulation_context;

const MONOLOGUE_FRESHNESS_LOG_MAX_MS: i64 = 60_000;
const MONOLOGUE_LATENCY_BUDGET_MS: i64 = 1200;
const MONOLOGUE_LLM_TIMEOUT_SECS: u64 = 90;
const MONOLOGUE_RETRY_TIMEOUT_SECS: u64 = 25;
const MONOLOGUE_RETRY_MAX_TOKENS: i64 = 140;
const MONOLOGUE_MAX_TOKENS_DELIBERATION: i64 = 900;
const MONOLOGUE_MAX_TOKENS_FREE_THOUGHT: i64 = 700;
const MONOLOGUE_LOCK_BUSY_COOLDOWN_SECS: i64 = 120;
const SAFE_UNHALT_WINDOW_MINS: i64 = 20;
const SAFE_UNHALT_MIN_SINCE_HALT_SECS: i64 = 300;
const SAFE_UNHALT_MAX_PARSE_FAILED: i64 = 3;
const SAFE_UNHALT_MAX_TIMEOUTS: i64 = 2;
const SAFE_UNHALT_MAX_RUNAWAY: i64 = 1;
const SAFE_UNHALT_MAX_TOOL_SCHEMA_ERRORS: i64 = 1;
const SAFE_UNHALT_MAX_UNKNOWN_TOOL: i64 = 1;
const SAFE_UNHALT_MAX_EMPTY_REASONING: i64 = 2;
const SAFE_UNHALT_MAX_PREDICTION_FAILED: i64 = 2;
const MONOLOGUE_PROBATION_SECS: i64 = 15 * 60;
const MONOLOGUE_PROBATION_WINDOW_MINS: i64 = 10;
const MONOLOGUE_HEALTH_ALERT_COOLDOWN_SECS: i64 = 300;
const MONOLOGUE_TIMEOUT_SAMPLE_WINDOW_MINS: i64 = 60;
const MONOLOGUE_TIMEOUT_SAMPLE_LIMIT: i64 = 20;
const MONOLOGUE_TIMEOUT_MIN_SECS: u64 = 15;
const MONOLOGUE_TIMEOUT_MAX_SECS: u64 = 180;
const MONOLOGUE_STALE_USER_INPUT_GRACE_SECS: i64 = 30;

const MONOLOGUE_UPDATE_SCHEMA: &str = r#"
State update format:
- If you intend to change workspace, goal thread, semantic core, or world model, emit a candidate in `candidates[]` with kind `update_workspace`, `update_goal_thread`, or `promote_semantic`.
- Do not encode state updates only in the `message` field.
- You may include top-level fields on a candidate: `target_scope` (workspace|goal_thread|semantic|world_model), `evidence_event_ids`, `belief_ids`.
"#;

pub(crate) fn monologue_update_schema() -> &'static str {
    MONOLOGUE_UPDATE_SCHEMA
}

fn monologue_timeout_secs(settings: &crate::models::Settings) -> u64 {
    settings
        .monologue_timeout_secs
        .unwrap_or(MONOLOGUE_LLM_TIMEOUT_SECS as i64)
        .max(5) as u64
}

fn monologue_retry_timeout_secs(settings: &crate::models::Settings) -> u64 {
    settings
        .monologue_retry_timeout_secs
        .unwrap_or(MONOLOGUE_RETRY_TIMEOUT_SECS as i64)
        .max(5) as u64
}

fn sanitize_monologue_user_attribution(
    message: &str,
    last_user_input: &str,
    user_name: &str,
) -> (String, bool) {
    if !super::response_has_user_attribution(message, user_name) {
        return (message.to_string(), false);
    }
    if last_user_input.trim().is_empty() {
        let cleaned = message
            .split(|c| c == '.' || c == '?' || c == '!' || c == '\n')
            .filter(|sentence| !super::response_has_user_attribution(sentence, user_name))
            .map(|sentence| sentence.trim())
            .filter(|sentence| !sentence.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        return (cleaned, true);
    }
    let cleaned = message
        .split(|c| c == '.' || c == '?' || c == '!' || c == '\n')
        .filter(|sentence| !super::response_has_user_attribution(sentence, user_name))
        .map(|sentence| sentence.trim())
        .filter(|sentence| !sentence.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    (cleaned, true)
}

fn monologue_followup_allowed(last_user_input: &str) -> bool {
    let trimmed = last_user_input.trim();
    if trimmed.is_empty() {
        return false;
    }
    if super::is_trivial_greeting(trimmed) || super::is_trivial_user_message(trimmed) {
        return false;
    }
    true
}

#[derive(Debug, Clone, Default)]
struct MonologueHealthCounts {
    parse_failed: i64,
    timeouts: i64,
    runaway: i64,
    tool_schema_errors: i64,
    unknown_tool: i64,
    empty_reasoning: i64,
    prediction_failed: i64,
}

impl MonologueHealthCounts {
    fn within_unhalt_limits(&self) -> bool {
        self.parse_failed <= SAFE_UNHALT_MAX_PARSE_FAILED
            && self.timeouts <= SAFE_UNHALT_MAX_TIMEOUTS
            && self.runaway <= SAFE_UNHALT_MAX_RUNAWAY
            && self.tool_schema_errors <= SAFE_UNHALT_MAX_TOOL_SCHEMA_ERRORS
            && self.unknown_tool <= SAFE_UNHALT_MAX_UNKNOWN_TOOL
            && self.empty_reasoning <= SAFE_UNHALT_MAX_EMPTY_REASONING
            && self.prediction_failed <= SAFE_UNHALT_MAX_PREDICTION_FAILED
    }

    fn as_json(&self) -> Value {
        json!({
            "parse_failed": self.parse_failed,
            "timeouts": self.timeouts,
            "runaway": self.runaway,
            "tool_schema_errors": self.tool_schema_errors,
            "unknown_tool": self.unknown_tool,
            "empty_reasoning": self.empty_reasoning,
            "prediction_failed": self.prediction_failed,
        })
    }
}

#[derive(Debug, Clone)]
struct MonologueTimeoutConfig {
    llm_timeout_secs: u64,
    retry_timeout_secs: u64,
    adaptive: bool,
    sample_count: usize,
    p90_ms: i64,
    avg_ms: i64,
}

impl Kernel {
    async fn ensure_monologue_json_support(
        &self,
        settings: &crate::models::Settings,
        state: &mut KernelState,
    ) {
        if state.monologue_json_supported.is_some() {
            return;
        }

        let summarizer_url = settings
            .summarization_api_url
            .as_deref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty());
        let primary_url = settings.api_base_url.as_str();

        let probe = |base_url: String, model: String| async move {
            let request = ChatCompletionRequest {
                model,
                messages: vec![
                    ChatMessage {
                        role: "system".to_string(),
                        content: "Return ONLY valid JSON in the form {\"ok\": true}.".to_string(),
                    },
                    ChatMessage {
                        role: "user".to_string(),
                        content: "Probe JSON mode.".to_string(),
                    },
                ],
                stream: false,
                temperature: Some(0.0),
                top_p: None,
                max_tokens: Some(32),
                response_format: Some(json!({ "type": "json_object" })),
                tools: None,
                tool_choice: None,
                enable_thinking: None,
                prefill: None,
                skip_injection: Some(true),
                skip_memory: Some(true),
                skip_reminders: Some(true),
                memory_expand: None,
                allow_diagnostics: Some(false),
                json_strict: Some(true),
                skip_sanitization: None,
                run_id: None,
                request_label: Some("monologue_json_probe".to_string()),
            };

            match self
                .model_client
                .chat_with_meta(&base_url, settings.api_key.as_deref(), &request)
                .await
            {
                Ok(response) => (parse_json_object_with_repair(&response.content).0.is_some(), None),
                Err(err) => (false, Some(err)),
            }
        };

        if let Some(base_url) = summarizer_url {
            let model = settings
                .summarization_model
                .as_deref()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .or_else(|| settings.active_model_id.clone())
                .unwrap_or_else(|| "default".to_string());
            let (supported, error) = probe(base_url.to_string(), model.clone()).await;
            if supported {
                state.monologue_json_supported = Some(true);
                state.monologue_force_primary = false;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "monologue_json_probe",
                        "supported": true,
                        "base_url": base_url,
                        "model": model,
                        "error": error,
                    }),
                )
                .await;
                return;
            }

            let primary_model = settings
                .active_model_id
                .clone()
                .unwrap_or_else(|| "default".to_string());
            let (primary_supported, primary_error) = probe(primary_url.to_string(), primary_model.clone()).await;
            state.monologue_json_supported = Some(primary_supported);
            state.monologue_force_primary = true;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "monologue_json_probe",
                    "supported": primary_supported,
                    "base_url": primary_url,
                    "model": primary_model,
                    "error": primary_error,
                    "fallback_from_summarizer": true,
                }),
            )
            .await;
            return;
        }

        let primary_model = settings
            .active_model_id
            .clone()
            .unwrap_or_else(|| "default".to_string());
        let (supported, error) = probe(primary_url.to_string(), primary_model.clone()).await;
        state.monologue_json_supported = Some(supported);
        state.monologue_force_primary = false;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            None,
            json!({
                "event": "monologue_json_probe",
                "supported": supported,
                "base_url": primary_url,
                "model": primary_model,
                "error": error,
            }),
        )
        .await;
    }

    pub(super) async fn count_event_since(&self, event: &str, window_mins: i64) -> i64 {
        let window = format!("-{} minutes", window_mins);
        sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs
             WHERE json_extract(payload, '$.event') = ?
               AND datetime(timestamp) >= datetime('now', ?)"
        )
        .bind(event)
        .bind(window)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0)
    }

    async fn recent_monologue_durations(&self, window_mins: i64, limit: i64) -> Vec<i64> {
        let window = format!("-{} minutes", window_mins);
        let rows = sqlx::query(
            "SELECT json_extract(payload, '$.duration_ms') as duration_ms
             FROM system_logs
             WHERE json_extract(payload, '$.event') = 'timing_monologue_tick'
               AND datetime(timestamp) >= datetime('now', ?)
             ORDER BY datetime(timestamp) DESC
             LIMIT ?",
        )
        .bind(window)
        .bind(limit.max(1))
        .fetch_all(&self.db.pool)
        .await
        .unwrap_or_default();

        let mut durations: Vec<i64> = Vec::new();
        for row in rows {
            if let Ok(Some(ms)) = row.try_get::<Option<i64>, _>("duration_ms") {
                if ms > 0 {
                    durations.push(ms);
                }
            }
        }
        durations
    }

    async fn adaptive_monologue_timeouts(
        &self,
        settings: &crate::models::Settings,
    ) -> MonologueTimeoutConfig {
        let base_timeout = monologue_timeout_secs(settings);
        let base_retry = monologue_retry_timeout_secs(settings);
        let samples = self
            .recent_monologue_durations(
                MONOLOGUE_TIMEOUT_SAMPLE_WINDOW_MINS,
                MONOLOGUE_TIMEOUT_SAMPLE_LIMIT,
            )
            .await;
        if samples.len() < 6 {
            return MonologueTimeoutConfig {
                llm_timeout_secs: base_timeout,
                retry_timeout_secs: base_retry,
                adaptive: false,
                sample_count: samples.len(),
                p90_ms: 0,
                avg_ms: 0,
            };
        }
        let mut sorted = samples.clone();
        sorted.sort();
        let idx = ((sorted.len() as f64) * 0.9).ceil().max(1.0) as usize - 1;
        let p90_ms = *sorted.get(idx).unwrap_or_else(|| sorted.last().unwrap_or(&0));
        let avg_ms = (samples.iter().sum::<i64>() / samples.len() as i64).max(0);
        let scaled = ((p90_ms as f64) / 1000.0 * 1.8).ceil() as u64;
        let adaptive_timeout = scaled
            .max(MONOLOGUE_TIMEOUT_MIN_SECS)
            .min(MONOLOGUE_TIMEOUT_MAX_SECS);
        let llm_timeout_secs = adaptive_timeout.max(base_timeout);
        let retry_floor = ((llm_timeout_secs as f64) * 0.35).ceil() as u64;
        let retry_timeout_secs = base_retry.max(retry_floor).min(llm_timeout_secs);

        MonologueTimeoutConfig {
            llm_timeout_secs,
            retry_timeout_secs,
            adaptive: llm_timeout_secs != base_timeout || retry_timeout_secs != base_retry,
            sample_count: samples.len(),
            p90_ms,
            avg_ms,
        }
    }

    async fn count_tool_schema_errors_since(&self, window_mins: i64) -> i64 {
        let window = format!("-{} minutes", window_mins);
        sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs
             WHERE json_extract(payload, '$.event') = 'tool_gate_blocked'
               AND json_extract(payload, '$.reason') = 'TOOL_ARGS_INVALID'
               AND datetime(timestamp) >= datetime('now', ?)"
        )
        .bind(window)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0)
    }

    async fn count_unknown_tool_since(&self, window_mins: i64) -> i64 {
        let window = format!("-{} minutes", window_mins);
        sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs
             WHERE json_extract(payload, '$.event') = 'tool_candidate_rejected'
               AND json_extract(payload, '$.reason') = 'UNKNOWN_TOOL'
               AND datetime(timestamp) >= datetime('now', ?)"
        )
        .bind(window)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0)
    }

    async fn monologue_health_counts(&self, window_mins: i64) -> MonologueHealthCounts {
        MonologueHealthCounts {
            parse_failed: self.count_event_since("monologue_parse_failed", window_mins).await,
            timeouts: self.count_event_since("monologue_tick_timeout", window_mins).await,
            runaway: self.count_event_since("monologue_runaway", window_mins).await,
            tool_schema_errors: self.count_tool_schema_errors_since(window_mins).await,
            unknown_tool: self.count_unknown_tool_since(window_mins).await,
            empty_reasoning: self.count_event_since("response_empty_with_reasoning", window_mins).await,
            prediction_failed: self.count_event_since("prediction_generation_failed", window_mins).await,
        }
    }

    async fn last_safe_halt_at(&self) -> Option<DateTime<Utc>> {
        sqlx::query_scalar(
            "SELECT timestamp FROM system_logs
             WHERE json_extract(payload, '$.event') = 'safe_halt'
             ORDER BY datetime(timestamp) DESC
             LIMIT 1"
        )
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten()
        .and_then(|ts: String| chrono::DateTime::parse_from_rfc3339(&ts).ok())
        .map(|ts| ts.with_timezone(&Utc))
    }

    async fn persist_monologue_status(
        &self,
        state: &mut KernelState,
        conversation_id: &str,
        tick_id: &str,
        outcome: &str,
        reason: &str,
        emit_allowed: bool,
    ) -> Result<(), String> {
        let now = Utc::now().to_rfc3339();
        state.last_monologue_tick_outcome = Some(outcome.to_string());
        state.last_monologue_status_emitted = Some(true);
        state.last_monologue_visible = Some(emit_allowed);
        let advance_timestamps = !(outcome == "skipped" && reason == "interval");
        if advance_timestamps {
            state.last_monologue_completed_at = Some(now.clone());
            state.last_monologue_at = Some(now.clone());
        }

        if emit_allowed {
            let entry = crate::models::InnerMonologueEntry {
                id: Uuid::new_v4().to_string(),
                conversation_id: conversation_id.to_string(),
                run_id: Some(tick_id.to_string()),
                dialogue_id: Some(format!("status:{}", tick_id)),
                turn_index: Some(0),
                speaker: Some("system".to_string()),
                mode: "status".to_string(),
                stream_type: Some("STATUS".to_string()),
                thought: format!("Status: {}", reason),
                descriptors: None,
                harvest_type: Some("status".to_string()),
                harvest_payload: Some(
                    json!({
                        "reason": reason,
                        "outcome": outcome,
                        "tick_id": tick_id,
                    })
                    .to_string(),
                ),
                created_at: now.clone(),
                candidates: None,
            };
            let _ = self.db.insert_inner_monologue_entry(&entry).await;
            let _ = self.app_handle.emit("inner_monologue", entry);
        } else {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(tick_id),
                json!({
                    "event": "monologue_status_blocked",
                    "conversation_id": conversation_id,
                    "tick_id": tick_id,
                    "reason": reason,
                    "outcome": outcome,
                }),
            )
            .await;
        }
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            Some(tick_id),
            json!({
                "event": "monologue_tick_result",
                "conversation_id": conversation_id,
                "tick_id": tick_id,
                "outcome": outcome,
                "suppressed_reason": reason,
                "emitted_entries": if emit_allowed { 1 } else { 0 },
                "status_only": emit_allowed,
            }),
        )
        .await;
        Ok(())
    }

    pub(super) async fn update_monologue_suppression_window(
        &self,
        state: &mut KernelState,
        stream: &str,
        suppression_reasons: &HashMap<String, usize>,
    ) {
        let now = Utc::now();
        let window_start = state
            .monologue_suppression_window_start
            .as_deref()
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
            .map(|ts| ts.with_timezone(&Utc))
            .unwrap_or(now);

        if now.signed_duration_since(window_start).num_seconds() >= 3600 {
            if !state.monologue_suppression_counts.is_empty() {
                let mut summary_counts: HashMap<String, usize> = HashMap::new();
                for (reason, count) in state.monologue_suppression_counts.iter() {
                    if *count > 0 {
                        summary_counts.insert(reason.clone(), (*count).max(0) as usize);
                    }
                }
            if !summary_counts.is_empty() {
                let _ = system_log::log_suppression_summary(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "kernel",
                        None,
                        None,
                        "monologue_suppression_summary",
                        "all",
                        &summary_counts,
                        Some(json!({
                            "window": "hour",
                            "window_start": window_start.to_rfc3339(),
                            "window_end": now.to_rfc3339(),
                            "latest_stream": stream,
                        })),
                    )
                    .await;
                }
            }
            state.monologue_suppression_counts.clear();
            state.monologue_suppression_window_start = Some(now.to_rfc3339());
        } else if state.monologue_suppression_window_start.is_none() {
            state.monologue_suppression_window_start = Some(window_start.to_rfc3339());
        }

        if !suppression_reasons.is_empty() {
            for (reason, count) in suppression_reasons.iter() {
                let entry = state.monologue_suppression_counts.entry(reason.clone()).or_insert(0);
                *entry = entry.saturating_add(*count as i64);
            }
            if state.monologue_suppression_window_start.is_none() {
                state.monologue_suppression_window_start = Some(now.to_rfc3339());
            }
        }
    }

    pub async fn run_monologue_tick(
        &self,
        conversation_id: &str,
        priority: bool,
        origin_run_id: Option<String>,
        origin_trace_id: Option<String>,
    ) -> Result<(), String> {
        let controls = system_controls::load_control_map(&self.db).await;
        let mode = system_controls::mode_for("monologue_loop", &controls);
        let inner_summary_mode = system_controls::mode_for("inner_summary", &controls);
        let memory_write_mode = system_controls::mode_for("memory_write", &controls);
        let recovery_mode = system_controls::mode_for("monologue_recovery", &controls);
        let recovery_allowed = !system_controls::mode_is_off(&recovery_mode);
        let mut unhalt_requested = false;
        let mut force_unhalt = false;
        if let Some(state) = controls.get("monologue_recovery") {
            if let Some(raw) = state.value_json.as_deref() {
                if let Ok(value) = serde_json::from_str::<Value>(raw) {
                    unhalt_requested = value.get("request_unhalt").and_then(|v| v.as_bool()).unwrap_or(false);
                    force_unhalt = value.get("force_unhalt").and_then(|v| v.as_bool()).unwrap_or(false);
                }
            }
        }
        if system_controls::mode_is_off(&mode) {
            return Ok(());
        }
        if system_controls::mode_is_degraded(&mode) && !priority {
            return Ok(());
        }
        let tick_id = Uuid::new_v4().to_string();
        let entry_run_id = origin_run_id
            .clone()
            .unwrap_or_else(|| tick_id.clone());
        let log_run_id = origin_run_id.as_deref();
        let log_trace_id = origin_trace_id
            .as_deref()
            .or_else(|| Some(tick_id.as_str()));
        let tick_started_at = Utc::now();
        let tick_user_snapshot_at = self
            .db
            .get_latest_user_message_at(conversation_id)
            .await
            .ok()
            .flatten()
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(&ts).ok())
            .map(|ts| ts.with_timezone(&Utc));
        let tick_started = Instant::now();
        let settings = self.db.get_settings().await.map_err(|e| e.to_string())?;
        self.sync_self_model_runtime(&settings).await;
        let timeout_config = self.adaptive_monologue_timeouts(&settings).await;
        let llm_timeout_secs = timeout_config.llm_timeout_secs;
        let retry_timeout_secs = timeout_config.retry_timeout_secs;
        if timeout_config.adaptive {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                log_run_id,
                log_trace_id,
                json!({
                    "event": "monologue_timeout_adaptive",
                    "conversation_id": conversation_id,
                    "llm_timeout_secs": llm_timeout_secs,
                    "retry_timeout_secs": retry_timeout_secs,
                    "sample_count": timeout_config.sample_count,
                    "p90_ms": timeout_config.p90_ms,
                    "avg_ms": timeout_config.avg_ms,
                }),
            )
            .await;
        }
        let lock_wait_started = Instant::now();
        let mut _monologue_guard = match self.monologue_lock.try_lock() {
            Ok(guard) => guard,
            Err(_) => {
                let wait_ms = lock_wait_started.elapsed().as_millis() as i64;
                let mut state = self.load_state(conversation_id).await;
                let now = Utc::now();
                let suppress_status = state
                    .last_monologue_at
                    .as_deref()
                    .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
                    .map(|ts| now.signed_duration_since(ts.with_timezone(&Utc)).num_seconds())
                    .map(|delta| delta < MONOLOGUE_LOCK_BUSY_COOLDOWN_SECS)
                    .unwrap_or(false);
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    log_run_id,
                    log_trace_id,
                    json!({
                        "event": "monologue_tick_lock_wait_ms",
                        "conversation_id": conversation_id,
                        "tick_id": tick_id,
                        "wait_ms": wait_ms,
                        "acquired": false,
                        "stage": "snapshot",
                    }),
                )
                .await;
                if suppress_status {
                    return Ok(());
                }
                let allowed = state.stop_state.allowed_capabilities();
                let _ = self
                    .persist_monologue_status(
                        &mut state,
                        conversation_id,
                        &tick_id,
                        "skipped",
                        "lock_busy",
                        allowed.monologue_emit,
                    )
                    .await;
                self.persist_monologue_patch(&state).await;
                return Ok(());
            }
        };
        let lock_wait_ms = lock_wait_started.elapsed().as_millis() as i64;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            log_run_id,
            log_trace_id,
            json!({
                "event": "monologue_tick_lock_wait_ms",
                "conversation_id": conversation_id,
                "tick_id": tick_id,
                "wait_ms": lock_wait_ms,
                "acquired": true,
                "stage": "snapshot",
            }),
        )
        .await;
        let lock_hold_started = Instant::now();
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            log_run_id,
            log_trace_id,
            json!({
                "event": "monologue_tick_start",
                "conversation_id": conversation_id,
                "tick_id": tick_id,
                "started_at": tick_started_at.to_rfc3339(),
                "priority": priority,
            }),
        )
        .await;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            log_run_id,
            log_trace_id,
            json!({
                "event": "monologue_tick_stage",
                "conversation_id": conversation_id,
                "tick_id": tick_id,
                "stage": "snapshot",
            }),
        )
        .await;
        let mut state = self.load_state(conversation_id).await;
        if let Ok(Some((message_id, content, created_at))) =
            self.db.get_latest_user_message(conversation_id).await
        {
            state.last_user_message_id = Some(message_id);
            state.last_user_input = Some(content);
            state.last_user_input_at = Some(created_at);
        }
        state.last_monologue_tick_id = Some(tick_id.clone());
        self.ensure_monologue_json_support(&settings, &mut state)
            .await;
        self.refresh_research_budget(&mut state, &settings);

        let mut effective_stop_state = state.stop_state.clone();
        if settings.monologue_stabilization_enabled.unwrap_or(true) {
            if let Some(until) = state.monologue_quiet_until.as_deref() {
                if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(until) {
                    if Utc::now() < ts.with_timezone(&Utc) {
                        let mut scope = StopScope::default();
                        scope.monologue_run = true;
                        scope.monologue_emit = true;
                        effective_stop_state.apply_reason(
                            StopReason {
                                category: StopReasonCategory::LatchBlock,
                                subcode: "monologue_quiet_until".to_string(),
                                contract: None,
                            },
                            scope,
                        );
                    } else {
                        state.monologue_quiet_until = None;
                    }
                }
            }
        }
        let mut allowed_capabilities = effective_stop_state.allowed_capabilities();
        let mut monologue_emit_allowed = allowed_capabilities.monologue_emit;
        let monologue_run_allowed = allowed_capabilities.monologue_run;

        if !monologue_run_allowed {
            self.persist_monologue_status(
                &mut state,
                conversation_id,
                &tick_id,
                "suppressed",
                "stop_state",
                monologue_emit_allowed,
            )
            .await?;
            self.persist_monologue_patch(&state).await;
            let hold_ms = lock_hold_started.elapsed().as_millis() as i64;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "monologue_tick_lock_hold_ms",
                    "conversation_id": conversation_id,
                    "tick_id": tick_id,
                    "hold_ms": hold_ms,
                    "stage": "snapshot",
                }),
            )
            .await;
            return Ok(());
        }

        if !priority {
            let active_runs: Option<i64> = sqlx::query_scalar(
                "SELECT COUNT(*) FROM runs WHERE status = 'active' AND conversation_id = ?",
            )
            .bind(conversation_id)
            .fetch_optional(&self.db.pool)
            .await
            .ok()
            .flatten();
            if active_runs.unwrap_or(0) > 0 {
                let elapsed_ms = tick_started.elapsed().as_millis() as i64;
                let remaining_ms = MONOLOGUE_LATENCY_BUDGET_MS.saturating_sub(elapsed_ms);
                if remaining_ms <= 0 {
                    self.persist_monologue_status(
                        &mut state,
                        conversation_id,
                        &tick_id,
                        "skipped",
                        "latency_budget_exceeded",
                        monologue_emit_allowed,
                    )
                    .await?;
                    let duration_ms = tick_started.elapsed().as_millis() as i64;
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        Some(&tick_id),
                        json!({
                            "event": "timing_monologue_tick",
                            "duration_ms": duration_ms,
                            "conversation_id": conversation_id,
                            "tick_id": tick_id,
                            "outcome": "latency_budget_exceeded",
                        }),
                    )
                    .await;
                    self.persist_monologue_patch(&state).await;
                    let hold_ms = lock_hold_started.elapsed().as_millis() as i64;
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        Some(&tick_id),
                        json!({
                            "event": "monologue_tick_lock_hold_ms",
                            "conversation_id": conversation_id,
                            "tick_id": tick_id,
                            "hold_ms": hold_ms,
                            "stage": "snapshot",
                        }),
                    )
                    .await;
                    return Ok(());
                }
            }
        }

        let budget_remaining = self.research_budget_remaining(&state, &settings);
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            None,
            json!({
                "event": "research_budget_start",
                "remaining": budget_remaining,
                "window_start": state.research_window_start,
            }),
        )
        .await;

        if state.halted && recovery_allowed {
            let now = Utc::now();
            let last_safe_halt = self.last_safe_halt_at().await;
            let safe_cooldown_ok = last_safe_halt
                .map(|ts| now.signed_duration_since(ts).num_seconds() >= SAFE_UNHALT_MIN_SINCE_HALT_SECS)
                .unwrap_or(true);
            let counts = self.monologue_health_counts(SAFE_UNHALT_WINDOW_MINS).await;
            if unhalt_requested || force_unhalt {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    Some(&tick_id),
                    json!({
                        "event": "monologue_unhalt_requested",
                        "conversation_id": conversation_id,
                        "tick_id": tick_id,
                        "force": force_unhalt,
                    }),
                )
                .await;
            }
            let criteria_ok = safe_cooldown_ok && counts.within_unhalt_limits();
            if force_unhalt || criteria_ok {
                state.halted = false;
                state.monologue_unhalted_at = Some(now.to_rfc3339());
                state.monologue_probation_until = Some((now + chrono::Duration::seconds(MONOLOGUE_PROBATION_SECS)).to_rfc3339());
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    None,
                    Some(&tick_id),
                    json!({
                        "event": "monologue_unhalted",
                        "conversation_id": conversation_id,
                        "tick_id": tick_id,
                        "counts": counts.as_json(),
                        "safe_cooldown_ok": safe_cooldown_ok,
                        "criteria_ok": criteria_ok,
                        "requested": unhalt_requested,
                        "forced": force_unhalt,
                    }),
                )
                .await;
            } else {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    Some(&tick_id),
                    json!({
                        "event": "monologue_unhalt_blocked",
                        "conversation_id": conversation_id,
                        "tick_id": tick_id,
                        "counts": counts.as_json(),
                        "safe_cooldown_ok": safe_cooldown_ok,
                        "criteria_ok": criteria_ok,
                        "requested": unhalt_requested,
                        "forced": force_unhalt,
                    }),
                )
                .await;
            }
        }

        if let Some(until) = state.monologue_probation_until.as_deref() {
            if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(until) {
                let now = Utc::now();
                if now < ts.with_timezone(&Utc) && !state.halted {
                    let counts = self.monologue_health_counts(MONOLOGUE_PROBATION_WINDOW_MINS).await;
                    if !counts.within_unhalt_limits() {
                        state.halted = true;
                        state.monologue_probation_until = None;
                        state.monologue_quiet_until =
                            Some((now + chrono::Duration::seconds(300)).to_rfc3339());
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            None,
                            Some(&tick_id),
                            json!({
                                "event": "monologue_probation_halt",
                                "conversation_id": conversation_id,
                                "tick_id": tick_id,
                                "counts": counts.as_json(),
                            }),
                        )
                        .await;
                    }
                } else if now >= ts.with_timezone(&Utc) {
                    state.monologue_probation_until = None;
                }
            } else {
                state.monologue_probation_until = None;
            }
        }

        if !state.halted {
            let now = Utc::now();
            let should_alert = state
                .monologue_health_last_alert_at
                .as_deref()
                .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
                .map(|ts| now.signed_duration_since(ts.with_timezone(&Utc)).num_seconds() >= MONOLOGUE_HEALTH_ALERT_COOLDOWN_SECS)
                .unwrap_or(true);
            if should_alert {
                let counts = self.monologue_health_counts(MONOLOGUE_PROBATION_WINDOW_MINS).await;
                if !counts.within_unhalt_limits() {
                    state.monologue_health_last_alert_at = Some(now.to_rfc3339());
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        None,
                        Some(&tick_id),
                        json!({
                            "event": "monologue_health_alert",
                            "conversation_id": conversation_id,
                            "tick_id": tick_id,
                            "counts": counts.as_json(),
                        }),
                    )
                    .await;
                }
            }
        }

        if state.halted {
            self.persist_monologue_status(
                &mut state,
                conversation_id,
                &tick_id,
                "skipped",
                "state_halted",
                monologue_emit_allowed,
            )
            .await?;
            let duration_ms = tick_started.elapsed().as_millis() as i64;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "timing_monologue_tick",
                    "duration_ms": duration_ms,
                    "conversation_id": conversation_id,
                    "tick_id": tick_id,
                    "outcome": "halted",
                }),
            )
            .await;
            self.persist_monologue_patch(&state).await;
            let hold_ms = lock_hold_started.elapsed().as_millis() as i64;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "monologue_tick_lock_hold_ms",
                    "conversation_id": conversation_id,
                    "tick_id": tick_id,
                    "hold_ms": hold_ms,
                    "stage": "snapshot",
                }),
            )
            .await;
            return Ok(());
        }

        if settings.monologue_stabilization_enabled.unwrap_or(true) {
            if state.monologue_quiet_until.is_some() && state.monologue_quiet_until.as_deref().unwrap_or("").is_empty() {
                state.monologue_quiet_until = None;
            }
        }

        self.refresh_controller_state(&mut state, &settings).await;

        let last_monologue_at_snapshot = state
            .last_monologue_completed_at
            .clone()
            .or_else(|| state.last_monologue_at.clone());
        match self.monologue_due(&mut state, &settings) {
            MonologueDue::Due => {
                state.last_monologue_started_at = Some(tick_started_at.to_rfc3339());
            }
            MonologueDue::Skipped(reason) => {
                self.persist_monologue_status(
                    &mut state,
                    conversation_id,
                    &tick_id,
                    "skipped",
                    reason,
                    monologue_emit_allowed,
                )
                .await?;
                let duration_ms = tick_started.elapsed().as_millis() as i64;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    Some(&tick_id),
                    json!({
                        "event": "timing_monologue_tick",
                        "duration_ms": duration_ms,
                        "conversation_id": conversation_id,
                        "tick_id": tick_id,
                        "outcome": "not_due",
                        "reason": reason,
                    }),
                )
                .await;
                self.persist_monologue_patch(&state).await;
                let hold_ms = lock_hold_started.elapsed().as_millis() as i64;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    Some(&tick_id),
                    json!({
                        "event": "monologue_tick_lock_hold_ms",
                        "conversation_id": conversation_id,
                        "tick_id": tick_id,
                        "hold_ms": hold_ms,
                        "stage": "snapshot",
                    }),
                )
                .await;
                return Ok(());
            }
        }

        let outcomes = self.collect_outcomes(&mut state).await;
        self.refresh_workspace_focus(&mut state, conversation_id).await;
        let has_new_user_input = self.decision_needed(&state, &outcomes, last_monologue_at_snapshot.as_deref());
        let mut decision_mode = self.should_use_decision_mode(&state, has_new_user_input);
        let decision_turns = if decision_mode {
            self.decision_turns_for(&state)
        } else {
            0
        };
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            None,
            json!({
                "event": "decision_mode_selection",
                "has_new_user_input": has_new_user_input,
                "decision_mode": decision_mode,
                "decision_turns": decision_turns,
                "path": if decision_mode {
                    "decision"
                } else if has_new_user_input {
                    "direct"
                } else {
                    "free"
                },
            }),
        )
        .await;
        let mut candidates = Vec::new();
        let mut status_written = false;
        let mut created_at = 0i64;
        let mut inner_summary_seed: Option<String> = None;

        if !outcomes.is_empty() {
            candidates.push(self.make_candidate(
                CandidateKind::UpdateGoalThread,
                json!({"outcomes": outcomes}),
                "outcome_ingestion",
                &mut created_at,
            ));
        }

        let mut snapshot_state = state.clone();
        let snapshot_anchor_epoch = snapshot_state.anchor_epoch;
        self.persist_monologue_patch(&state).await;
        let hold_ms = lock_hold_started.elapsed().as_millis() as i64;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            Some(&tick_id),
            json!({
                "event": "monologue_tick_lock_hold_ms",
                "conversation_id": conversation_id,
                "tick_id": tick_id,
                "hold_ms": hold_ms,
                "stage": "snapshot",
            }),
        )
        .await;
        drop(_monologue_guard);
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            Some(&tick_id),
            json!({
                "event": "monologue_tick_stage",
                "conversation_id": conversation_id,
                "tick_id": tick_id,
                "stage": "llm_call",
            }),
        )
        .await;

        let mut llm_error: Option<String> = None;
        let mut llm_timed_out = false;
        let mut deliberation = match tokio::time::timeout(
            std::time::Duration::from_secs(llm_timeout_secs),
            self.deliberate_self_dialogue(
                conversation_id,
                &settings,
                &mut snapshot_state,
                &outcomes,
                decision_mode,
                decision_turns,
                MonologueStream::Deliberation,
                Some(MONOLOGUE_MAX_TOKENS_DELIBERATION),
            ),
        )
        .await
        {
            Ok(Ok(output)) => output,
            Ok(Err(err)) => {
                llm_error = Some(err);
                MonologueOutput {
                    turns: Vec::new(),
                    last_message: None,
                    dialogue_messages: Vec::new(),
                }
            }
            Err(_) => {
                llm_timed_out = true;
                MonologueOutput {
                    turns: Vec::new(),
                    last_message: None,
                    dialogue_messages: Vec::new(),
                }
            }
        };

        if llm_timed_out {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "monologue_tick_timeout",
                    "conversation_id": conversation_id,
                    "tick_id": tick_id,
                    "timeout_secs": llm_timeout_secs,
                }),
            )
            .await;
            let counts = self.monologue_health_counts(MONOLOGUE_PROBATION_WINDOW_MINS).await;
            if counts.timeouts >= 1 {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    None,
                    Some(&tick_id),
                    json!({
                        "event": "monologue_tick_retry_skipped",
                        "conversation_id": conversation_id,
                        "tick_id": tick_id,
                        "reason": "recent_timeouts",
                        "timeouts": counts.timeouts,
                    }),
                )
                .await;
            } else {
                let retry = tokio::time::timeout(
                    std::time::Duration::from_secs(retry_timeout_secs),
                    self.deliberate_self_dialogue(
                        conversation_id,
                        &settings,
                        &mut snapshot_state,
                        &outcomes,
                        decision_mode,
                        decision_turns,
                        MonologueStream::Deliberation,
                        Some(MONOLOGUE_RETRY_MAX_TOKENS),
                    ),
                )
                .await;
                match retry {
                    Ok(Ok(output)) => {
                        deliberation = output;
                        llm_timed_out = false;
                        llm_error = None;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            Some(&tick_id),
                            json!({
                                "event": "monologue_tick_retry",
                                "conversation_id": conversation_id,
                                "tick_id": tick_id,
                                "stream": "DS",
                                "status": "ok",
                                "max_tokens": MONOLOGUE_RETRY_MAX_TOKENS,
                            }),
                        )
                        .await;
                    }
                    Ok(Err(err)) => {
                        llm_error = Some(err);
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            None,
                            Some(&tick_id),
                            json!({
                                "event": "monologue_tick_retry",
                                "conversation_id": conversation_id,
                                "tick_id": tick_id,
                                "stream": "DS",
                                "status": "error",
                            }),
                        )
                        .await;
                    }
                    Err(_) => {
                        llm_timed_out = true;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            None,
                            Some(&tick_id),
                            json!({
                                "event": "monologue_tick_retry",
                                "conversation_id": conversation_id,
                                "tick_id": tick_id,
                                "stream": "DS",
                                "status": "timeout",
                                "timeout_secs": retry_timeout_secs,
                            }),
                        )
                        .await;
                    }
                }
            }
        } else if let Some(err) = llm_error.as_deref() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "error",
                    "context": "monologue_deliberation",
                    "error": err,
                }),
            )
            .await;
        }

        if llm_timed_out || llm_error.is_some() {
            let commit_wait_started = Instant::now();
            _monologue_guard = self.monologue_lock.lock().await;
            let commit_wait_ms = commit_wait_started.elapsed().as_millis() as i64;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "monologue_tick_lock_wait_ms",
                    "conversation_id": conversation_id,
                    "tick_id": tick_id,
                    "wait_ms": commit_wait_ms,
                    "acquired": true,
                    "stage": "commit",
                }),
            )
            .await;
            let commit_hold_started = Instant::now();
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "monologue_tick_stage",
                    "conversation_id": conversation_id,
                    "tick_id": tick_id,
                    "stage": "commit",
                }),
            )
            .await;
            state = self.load_state(conversation_id).await;
            let allowed = state.stop_state.allowed_capabilities();
            let reason = if llm_timed_out { "timeout" } else { "llm_error" };
            let _ = self
                .persist_monologue_status(
                    &mut state,
                    conversation_id,
                    &tick_id,
                    "skipped",
                    reason,
                    allowed.monologue_emit,
                )
                .await;
            self.persist_monologue_patch(&state).await;
            let duration_ms = tick_started.elapsed().as_millis() as i64;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "timing_monologue_tick",
                    "duration_ms": duration_ms,
                    "conversation_id": conversation_id,
                    "tick_id": tick_id,
                    "outcome": reason,
                }),
            )
            .await;
            let hold_ms = commit_hold_started.elapsed().as_millis() as i64;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "monologue_tick_lock_hold_ms",
                    "conversation_id": conversation_id,
                    "tick_id": tick_id,
                    "hold_ms": hold_ms,
                    "stage": "commit",
                }),
            )
            .await;
            return Ok(());
        }

        let mut free_thought_timed_out = false;
        let mut free_thought = match tokio::time::timeout(
            std::time::Duration::from_secs(llm_timeout_secs),
            self.deliberate_self_dialogue(
                conversation_id,
                &settings,
                &mut snapshot_state,
                &outcomes,
                false,
                0,
                MonologueStream::FreeThought,
                Some(MONOLOGUE_MAX_TOKENS_FREE_THOUGHT),
            ),
        )
        .await
        {
            Ok(Ok(output)) => Some(output),
            Ok(Err(err)) => {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "monologue_fts_error",
                        "error": err,
                    }),
                )
                .await;
                None
            }
            Err(_) => {
                free_thought_timed_out = true;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    None,
                    Some(&tick_id),
                    json!({
                        "event": "monologue_tick_timeout",
                        "conversation_id": conversation_id,
                        "tick_id": tick_id,
                        "stream": "free_thought",
                        "timeout_secs": llm_timeout_secs,
                    }),
                )
                .await;
                None
            }
        };

        if free_thought_timed_out {
            let retry = tokio::time::timeout(
                std::time::Duration::from_secs(retry_timeout_secs),
                self.deliberate_self_dialogue(
                    conversation_id,
                    &settings,
                    &mut snapshot_state,
                    &outcomes,
                    false,
                    0,
                    MonologueStream::FreeThought,
                    Some(MONOLOGUE_RETRY_MAX_TOKENS),
                ),
            )
            .await;
            match retry {
                Ok(Ok(output)) => {
                    free_thought = Some(output);
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        Some(&tick_id),
                        json!({
                            "event": "monologue_tick_retry",
                            "conversation_id": conversation_id,
                            "tick_id": tick_id,
                            "stream": "FTS",
                            "status": "ok",
                            "max_tokens": MONOLOGUE_RETRY_MAX_TOKENS,
                        }),
                    )
                    .await;
                }
                Ok(Err(err)) => {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        None,
                        Some(&tick_id),
                        json!({
                            "event": "monologue_tick_retry",
                            "conversation_id": conversation_id,
                            "tick_id": tick_id,
                            "stream": "FTS",
                            "status": "error",
                            "error": err,
                        }),
                    )
                    .await;
                }
                Err(_) => {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        None,
                        Some(&tick_id),
                        json!({
                            "event": "monologue_tick_retry",
                            "conversation_id": conversation_id,
                            "tick_id": tick_id,
                            "stream": "FTS",
                            "status": "timeout",
                            "timeout_secs": retry_timeout_secs,
                        }),
                    )
                    .await;
                }
            }
        }

        let snapshot_delta_last_monologue_anchor_epoch = snapshot_state.last_monologue_anchor_epoch;
        let snapshot_delta_meta_cog_reanchor_attempts = snapshot_state.meta_cog_reanchor_attempts;
        let snapshot_delta_monologue_misaligned_streak = snapshot_state.monologue_misaligned_streak;
        let snapshot_delta_monologue_loop_streak = snapshot_state.monologue_loop_streak;
        let snapshot_delta_last_meta_cog_loop_break_reason =
            snapshot_state.last_meta_cog_loop_break_reason.clone();

        let commit_wait_started = Instant::now();
        _monologue_guard = self.monologue_lock.lock().await;
        let commit_wait_ms = commit_wait_started.elapsed().as_millis() as i64;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            Some(&tick_id),
            json!({
                "event": "monologue_tick_lock_wait_ms",
                "conversation_id": conversation_id,
                "tick_id": tick_id,
                "wait_ms": commit_wait_ms,
                "acquired": true,
                "stage": "commit",
            }),
        )
        .await;
        let commit_hold_started = Instant::now();
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            Some(&tick_id),
            json!({
                "event": "monologue_tick_stage",
                "conversation_id": conversation_id,
                "tick_id": tick_id,
                "stage": "commit",
            }),
        )
        .await;
        state = self.load_state(conversation_id).await;

        let mut stale_monologue = false;
        let mut stale_reasons: Vec<&str> = Vec::new();
        if let Some(latest) = tick_user_snapshot_at {
            let age_secs = tick_started_at
                .signed_duration_since(latest)
                .num_seconds()
                .max(0);
            if age_secs > MONOLOGUE_STALE_USER_INPUT_GRACE_SECS {
                stale_monologue = true;
                stale_reasons.push("stale_user_input");
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "monologue_stale",
                        "tick_id": tick_id,
                        "age_secs": age_secs,
                        "grace_secs": MONOLOGUE_STALE_USER_INPUT_GRACE_SECS,
                        "reason": "stale_user_input",
                        "tick_user_snapshot_at": latest.to_rfc3339(),
                    }),
                )
                .await;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "stale_monologue_dropped",
                        "tick_id": tick_id,
                        "latest_user_message_at": latest.to_rfc3339(),
                        "tick_started_at": tick_started_at.to_rfc3339(),
                    }),
                )
                .await;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "monologue_stale_dropped",
                        "tick_id": tick_id,
                        "latest_user_message_at": latest.to_rfc3339(),
                        "tick_started_at": tick_started_at.to_rfc3339(),
                    }),
                )
                .await;
            }
        }
        if state.anchor_epoch != snapshot_anchor_epoch {
            stale_monologue = true;
            stale_reasons.push("anchor_epoch_changed");
        }
        let stale_reason_payload = if stale_monologue {
            Some(json!({ "stale_reasons": stale_reasons.clone() }))
        } else {
            None
        };
        if stale_monologue {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "monologue_commit_skipped",
                    "conversation_id": conversation_id,
                    "tick_id": tick_id,
                    "reasons": stale_reasons,
                }),
            )
            .await;
        } else {
            state.last_monologue_anchor_epoch = snapshot_delta_last_monologue_anchor_epoch;
            state.meta_cog_reanchor_attempts = snapshot_delta_meta_cog_reanchor_attempts;
            state.monologue_misaligned_streak = snapshot_delta_monologue_misaligned_streak;
            state.monologue_loop_streak = snapshot_delta_monologue_loop_streak;
            state.last_meta_cog_loop_break_reason = snapshot_delta_last_meta_cog_loop_break_reason;
        }
        allowed_capabilities = state.stop_state.allowed_capabilities();
        monologue_emit_allowed = allowed_capabilities.monologue_emit;

        if stale_monologue {
            decision_mode = false;
        }

        if let Some(message) = free_thought
            .as_ref()
            .and_then(|m| m.last_message.as_deref())
            .or_else(|| deliberation.last_message.as_deref())
        {
            if !message.trim().is_empty() {
                state.self_state.last_internal_thought = summarize_snippet(message, 160);
                state.self_state.updated_at = Some(Utc::now().to_rfc3339());
            }
        }

        let (self_a_turns, self_b_turns) =
            deliberation
                .turns
                .iter()
                .fold((0usize, 0usize), |acc, turn| {
            let speaker = turn.entry.speaker.as_deref().unwrap_or("");
            if speaker == "self_a" {
                (acc.0 + 1, acc.1)
            } else if speaker == "self_b" {
                (acc.0, acc.1 + 1)
            } else {
                acc
            }
        });
        let monologue_candidate_count: usize = deliberation.turns.iter().map(|t| t.candidates.len()).sum();
        let monologue_blocked_count: usize = deliberation
            .turns
            .iter()
            .map(|t| t.blocked_candidates.len())
            .sum();
        let mut blocked_reason_counts: HashMap<String, usize> = HashMap::new();
        for turn in deliberation.turns.iter() {
            for blocked in turn.blocked_candidates.iter() {
                *blocked_reason_counts
                    .entry(blocked.reason.clone())
                    .or_insert(0) += 1;
            }
        }
        if !blocked_reason_counts.is_empty() {
            let _ = system_log::log_suppression_summary(
                &self.db.pool,
                Some(&self.app_handle),
                "kernel",
                None,
                None,
                "monologue_candidate_suppressed",
                MonologueStream::Deliberation.as_str(),
                &blocked_reason_counts,
                Some(json!({
                    "turns": deliberation.turns.len(),
                    "blocked_candidates": monologue_blocked_count,
                })),
            )
            .await;
            let _ = system_log::log_suppression_summary(
                &self.db.pool,
                Some(&self.app_handle),
                "kernel",
                None,
                None,
                "monologue_suppression_summary",
                MonologueStream::Deliberation.as_str(),
                &blocked_reason_counts,
                Some(json!({
                    "turns": deliberation.turns.len(),
                    "blocked_candidates": monologue_blocked_count,
                    "window": "tick",
                })),
            )
            .await;
            self.update_monologue_suppression_window(
                &mut state,
                MonologueStream::Deliberation.as_str(),
                &blocked_reason_counts,
            )
            .await;
        }
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            None,
            json!({
                "event": "monologue_tick",
                "tick_id": tick_id,
                "turns": deliberation.turns.len(),
                "fts_turns": free_thought.as_ref().map(|m| m.turns.len()).unwrap_or(0),
                "speaker_counts": {
                    "self_a": self_a_turns,
                    "self_b": self_b_turns,
                },
                "candidates_count": monologue_candidate_count,
                "blocked_candidates_count": monologue_blocked_count,
                "monologue_count": state.monologue_count,
                "reason": "scheduler",
            }),
        )
        .await;

        if monologue_candidate_count > 0 && !stale_monologue {
            for turn in deliberation.turns.iter() {
                candidates.extend(turn.candidates.clone());
            }
        }
        if stale_monologue {
            candidates.clear();
        }
        if !stale_monologue && candidates.is_empty() && monologue_blocked_count > 0 {
            let mut seed_source = "default";
            let seed = if let Some(focus) = state
                .workspace_current_focus
                .as_deref()
                .filter(|s| !s.trim().is_empty())
            {
                seed_source = "workspace_focus";
                focus
            } else if let Some(goal) = state
                .workspace_goal_thread
                .as_deref()
                .filter(|s| !s.trim().is_empty())
            {
                seed_source = "workspace_goal";
                goal
            } else if let Some(summary) = inner_summary_seed
                .as_deref()
                .filter(|s| !s.trim().is_empty())
            {
                seed_source = "inner_summary";
                summary
            } else if let Some(last_input) = state
                .last_user_input
                .as_deref()
                .filter(|s| !s.trim().is_empty())
            {
                seed_source = "last_user_input";
                last_input
            } else {
                "current topic"
            };
            let seed_snippet = summarize_snippet(seed, 160);
            let content = if seed_snippet.trim().is_empty() {
                "Review current context and prepare the next step.".to_string()
            } else {
                format!("Next focus: {}", seed_snippet)
            };
            candidates.push(self.make_candidate(
                CandidateKind::EmitMessage,
                json!({
                    "content": content,
                    "user_visible": false,
                    "speculative": true,
                    "speculative_reason": "monologue_fallback",
                }),
                "monologue_fallback_candidate",
                &mut created_at,
            ));
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "monologue_fallback_candidate",
                    "blocked_candidates": monologue_blocked_count,
                    "seed_source": seed_source,
                }),
            )
            .await;
        }

        if !stale_monologue && !deliberation.dialogue_messages.is_empty() {
            let inner_summary_allowed = !system_controls::mode_is_off(&inner_summary_mode)
                && !system_controls::mode_is_degraded(&inner_summary_mode);
            let memory_allowed =
                system_controls::allow_memory_write(&memory_write_mode, "inner_summary_update");
            if !inner_summary_allowed || !memory_allowed {
                let reason = if !inner_summary_allowed {
                    "inner_summary_control"
                } else {
                    "memory_write_control"
                };
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "memory",
                    None,
                    None,
                    json!({
                        "event": "memory_write_blocked",
                        "reason": reason,
                        "category": "inner_summary",
                        "conversation_id": conversation_id,
                    }),
                )
                .await;
            } else {
            if let Ok(candidate) = self
                .build_inner_summary_candidate_from_dialogue(
                    conversation_id,
                    &deliberation.dialogue_messages,
                    &outcomes,
                    &state,
                    &settings,
                    &mut created_at,
                )
                .await
            {
                if let Some(summary_json) = candidate.payload.get("summary_json").and_then(|v: &Value| v.as_str()) {
                    inner_summary_seed = Some(summary_json.to_string());
                }
                candidates.push(candidate);
            }
            }
        }

        let binding_enforcement_enabled = settings.binding_enforcement_enabled.unwrap_or(true);
        let monologue_has_content = deliberation
            .turns
            .iter()
            .any(|turn| !turn.entry.thought.trim().is_empty());
        if !stale_monologue {
            if monologue_candidate_count == 0 && monologue_blocked_count > 0 {
                state.monologue_candidate_reject_streak += 1;
            } else {
                state.monologue_candidate_reject_streak = 0;
            }
            if state.monologue_candidate_reject_streak >= 3 {
                let fallback = format!(
                    "Would you like me to surface a concrete next step related to: {}?",
                    summarize_snippet(
                        &state
                            .last_user_input
                            .clone()
                            .unwrap_or_else(|| "your request".to_string()),
                        120
                    )
                );
                candidates.push(self.make_candidate(
                    CandidateKind::AskUserQuestion,
                    json!({ "question": fallback }),
                    "monologue_reject_loop_breaker",
                    &mut created_at,
                ));
                state.monologue_candidate_reject_streak = 0;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "monologue_candidate_reject_loop_breaker",
                        "blocked_count": monologue_blocked_count,
                    }),
                )
                .await;
            }
        }
        let has_workspace_candidate = candidates
            .iter()
            .any(|candidate| matches!(candidate.kind, CandidateKind::UpdateWorkspace));
        if !stale_monologue && binding_enforcement_enabled && monologue_has_content && !has_workspace_candidate {
            if let Some(candidate) = self
                .build_workspace_fallback_candidate(
                    conversation_id,
                    &state,
                    inner_summary_seed.as_deref(),
                    &mut created_at,
                )
                .await
            {
                candidates.push(candidate);
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "workspace_fallback_enqueued",
                        "conversation_id": conversation_id,
                    }),
                )
                .await;
            }
        }

        let mut decision_state = state.clone();
        self.apply_outcomes(&mut decision_state, &outcomes).await;
        if !stale_monologue {
            if let Some(promo) = self
                .maybe_semantic_promotion_candidate(conversation_id, &decision_state, &settings, &mut created_at)
                .await
            {
                candidates.push(promo);
            }
        }

        if !stale_monologue && decision_mode && candidates.is_empty() {
            candidates.push(
                self.fallback_inner_summary_candidate(conversation_id, &state, &mut created_at)
                    .await,
            );
        }

        if !stale_monologue {
            if let Some(candidate) = self
                .build_meta_cognitive_observation_candidate(conversation_id, &mut created_at)
                .await
            {
                candidates.push(candidate);
            }
        }

        let state_change_candidates = candidates
            .iter()
            .filter(|candidate| is_state_change_candidate(&candidate.kind))
            .count();
        let non_state_change_candidates = candidates.len().saturating_sub(state_change_candidates);
        let loop_noop_reason = if state_change_candidates == 0 {
            if stale_monologue {
                Some("stale_monologue".to_string())
            } else if candidates.is_empty() && monologue_blocked_count > 0 {
                Some("blocked_candidates".to_string())
            } else if candidates.is_empty() {
                Some("no_candidates".to_string())
            } else {
                Some("non_state_change_only".to_string())
            }
        } else {
            None
        };

        let mut emitted_entries = false;
        if !deliberation.turns.is_empty() {
            let mut entry_batch: Vec<crate::models::InnerMonologueEntry> = Vec::new();
            let mut candidate_batch: Vec<crate::models::InnerMonologueCandidate> = Vec::new();
            let mut emit_entries: Vec<crate::models::InnerMonologueEntry> = Vec::new();
            for turn in deliberation.turns.iter() {
                let mut entry = turn.entry.clone();
                if settings.monologue_provenance_guard.unwrap_or(true) {
                    let user_name = settings.user_display_name.as_deref().unwrap_or("User");
                    let last_user_input = state.last_user_input.as_deref().unwrap_or("");
                    let original_len = entry.thought.len();
                    let (cleaned, changed) =
                        sanitize_monologue_user_attribution(&entry.thought, last_user_input, user_name);
                    if changed {
                        entry.thought = cleaned.clone();
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            Some(&tick_id),
                            json!({
                                "event": "monologue_provenance_stripped",
                                "entry_id": entry.id,
                                "stream": entry.stream_type,
                                "original_len": original_len,
                                "cleaned_len": cleaned.len(),
                            }),
                        )
                        .await;
                    }
                }
                entry.run_id = Some(entry_run_id.clone());
                if let Some(stale_payload) = stale_reason_payload.as_ref() {
                    let mut payload = entry
                        .harvest_payload
                        .as_deref()
                        .and_then(|p| serde_json::from_str::<Value>(p).ok())
                        .unwrap_or_else(|| json!({}));
                    if let Some(obj) = payload.as_object_mut() {
                        obj.insert(
                            "stale_reasons".to_string(),
                            stale_payload.get("stale_reasons").cloned().unwrap_or(json!([])),
                        );
                    }
                    entry.harvest_payload = Some(payload.to_string());
                }
                let entry_id = entry.id.clone();
                let mut candidate_entries = Vec::new();
                if stale_monologue {
                    for candidate in turn.candidates.iter() {
                        let mut candidate_value =
                            serde_json::to_value(candidate).unwrap_or_else(|_| json!({}));
                        if let Some(obj) = candidate_value.as_object_mut() {
                            obj.insert("run_id".to_string(), Value::String(tick_id.clone()));
                            obj.insert(
                                "blocked_reason".to_string(),
                                Value::String("stale_monologue".to_string()),
                            );
                        }
                        let candidate_json =
                            serde_json::to_string(&candidate_value).unwrap_or_else(|_| "{}".to_string());
                        let candidate_entry = crate::models::InnerMonologueCandidate {
                            id: Uuid::new_v4().to_string(),
                            entry_id: entry_id.clone(),
                            candidate_id: Some(candidate.id.clone()),
                            outcome: Some("rejected".to_string()),
                            suppression_reason: Some("stale_monologue".to_string()),
                            candidate_json,
                            created_at: Utc::now().to_rfc3339(),
                        };
                        candidate_entries.push(candidate_entry.clone());
                        candidate_batch.push(candidate_entry);
                    }
                } else {
                    for candidate in turn.candidates.iter() {
                        let mut candidate_value =
                            serde_json::to_value(candidate).unwrap_or_else(|_| json!({}));
                        if let Some(obj) = candidate_value.as_object_mut() {
                            obj.insert("run_id".to_string(), Value::String(tick_id.clone()));
                        }
                        let candidate_json =
                            serde_json::to_string(&candidate_value).unwrap_or_else(|_| "{}".to_string());
                        let candidate_entry = crate::models::InnerMonologueCandidate {
                            id: Uuid::new_v4().to_string(),
                            entry_id: entry_id.clone(),
                            candidate_id: Some(candidate.id.clone()),
                            outcome: Some("proposed".to_string()),
                            suppression_reason: None,
                            candidate_json,
                            created_at: Utc::now().to_rfc3339(),
                        };
                        candidate_entries.push(candidate_entry.clone());
                        candidate_batch.push(candidate_entry);
                    }
                    for blocked in turn.blocked_candidates.iter() {
                        let mut candidate_value =
                            serde_json::to_value(&blocked.candidate).unwrap_or_else(|_| json!({}));
                        if let Some(obj) = candidate_value.as_object_mut() {
                            obj.insert("run_id".to_string(), Value::String(entry_run_id.clone()));
                            obj.insert("tick_id".to_string(), Value::String(tick_id.clone()));
                            obj.insert("blocked_reason".to_string(), Value::String(blocked.reason.clone()));
                        }
                        let candidate_json =
                            serde_json::to_string(&candidate_value).unwrap_or_else(|_| "{}".to_string());
                        let candidate_entry = crate::models::InnerMonologueCandidate {
                            id: Uuid::new_v4().to_string(),
                            entry_id: entry_id.clone(),
                            candidate_id: Some(blocked.candidate.id.clone()),
                            outcome: Some("rejected".to_string()),
                            suppression_reason: Some(blocked.reason.clone()),
                            candidate_json,
                            created_at: Utc::now().to_rfc3339(),
                        };
                        candidate_entries.push(candidate_entry.clone());
                        candidate_batch.push(candidate_entry);
                    }
                }
                if !candidate_entries.is_empty() {
                    entry.candidates = Some(candidate_entries);
                }
                entry.run_id = Some(entry_run_id.clone());
                emit_entries.push(entry.clone());
                entry_batch.push(entry);
            }
            let _ = self.db.insert_inner_monologue_entries_batch(&entry_batch).await;
            let _ = self
                .db
                .insert_inner_monologue_candidates_batch(&candidate_batch)
                .await;
            emitted_entries = true;
            if let Some(last_user_at) = state
                .last_user_input_at
                .as_deref()
                .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
            {
                let freshness_ms = Utc::now()
                    .signed_duration_since(last_user_at.with_timezone(&Utc))
                    .num_milliseconds();
                if freshness_ms <= MONOLOGUE_FRESHNESS_LOG_MAX_MS {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        Some(&tick_id),
                        json!({
                            "event": "monologue_freshness_ms",
                            "conversation_id": conversation_id,
                            "tick_id": tick_id,
                            "stream": MonologueStream::Deliberation.as_str(),
                            "freshness_ms": freshness_ms,
                        }),
                    )
                    .await;
                } else {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        Some(&tick_id),
                        json!({
                            "event": "monologue_freshness_skipped",
                            "conversation_id": conversation_id,
                            "tick_id": tick_id,
                            "stream": MonologueStream::Deliberation.as_str(),
                            "freshness_ms": freshness_ms,
                            "reason": "stale_user_input",
                        }),
                    )
                    .await;
                }
            }
            for entry in emit_entries {
                let _ = self.app_handle.emit("inner_monologue", entry);
            }
        }

            if let Some(free_thought) = free_thought.as_ref() {
                if !free_thought.turns.is_empty() {
                    let mut entry_batch: Vec<crate::models::InnerMonologueEntry> = Vec::new();
                    let mut emit_entries: Vec<crate::models::InnerMonologueEntry> = Vec::new();
                    for turn in free_thought.turns.iter() {
                        let mut entry = turn.entry.clone();
                        if settings.monologue_provenance_guard.unwrap_or(true) {
                            let user_name = settings.user_display_name.as_deref().unwrap_or("User");
                            let last_user_input = state.last_user_input.as_deref().unwrap_or("");
                            let original_len = entry.thought.len();
                            let (cleaned, changed) =
                                sanitize_monologue_user_attribution(&entry.thought, last_user_input, user_name);
                            if changed {
                                entry.thought = cleaned.clone();
                                let _ = system_log::log_event(
                                    &self.db.pool,
                                    Some(&self.app_handle),
                                    "info",
                                    "kernel",
                                    None,
                                    Some(&tick_id),
                                    json!({
                                        "event": "monologue_provenance_stripped",
                                        "entry_id": entry.id,
                                        "stream": entry.stream_type,
                                        "original_len": original_len,
                                        "cleaned_len": cleaned.len(),
                                    }),
                                )
                                .await;
                            }
                        }
                        entry.run_id = Some(entry_run_id.clone());
                        if let Some(stale_payload) = stale_reason_payload.as_ref() {
                            let mut payload = entry
                                .harvest_payload
                                .as_deref()
                                .and_then(|p| serde_json::from_str::<Value>(p).ok())
                                .unwrap_or_else(|| json!({}));
                            if let Some(obj) = payload.as_object_mut() {
                                obj.insert(
                                    "stale_reasons".to_string(),
                                    stale_payload.get("stale_reasons").cloned().unwrap_or(json!([])),
                                );
                            }
                            entry.harvest_payload = Some(payload.to_string());
                        }
                        emit_entries.push(entry.clone());
                        entry_batch.push(entry);
                    }
                let _ = self.db.insert_inner_monologue_entries_batch(&entry_batch).await;
                emitted_entries = true;
                if let Some(last_user_at) = state
                    .last_user_input_at
                    .as_deref()
                    .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
                {
                    let freshness_ms = Utc::now()
                        .signed_duration_since(last_user_at.with_timezone(&Utc))
                        .num_milliseconds();
                    if freshness_ms <= MONOLOGUE_FRESHNESS_LOG_MAX_MS {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            Some(&tick_id),
                            json!({
                                "event": "monologue_freshness_ms",
                                "conversation_id": conversation_id,
                                "tick_id": tick_id,
                                "stream": MonologueStream::FreeThought.as_str(),
                                "freshness_ms": freshness_ms,
                            }),
                        )
                        .await;
                    } else {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            Some(&tick_id),
                            json!({
                                "event": "monologue_freshness_skipped",
                                "conversation_id": conversation_id,
                                "tick_id": tick_id,
                                "stream": MonologueStream::FreeThought.as_str(),
                                "freshness_ms": freshness_ms,
                                "reason": "stale_user_input",
                            }),
                        )
                        .await;
                    }
                }
                for entry in emit_entries {
                    let _ = self.app_handle.emit("inner_monologue", entry);
                }
            }
        }

        if !emitted_entries {
            self.persist_monologue_status(
                &mut state,
                conversation_id,
                &tick_id,
                "status_only",
                "no_entries",
                monologue_emit_allowed,
            )
            .await?;
            status_written = true;
        }

        let mut no_candidates_logged = false;
        if !decision_mode && candidates.is_empty() {
            self.update_self_state(&mut state, None, None, &settings);
            let disable_working_hypothesis = settings.stability_disable_working_hypothesis.unwrap_or(true);
            refresh_working_memory(&mut state, Utc::now(), disable_working_hypothesis);
            if !status_written {
                self.persist_monologue_status(
                    &mut state,
                    conversation_id,
                    &tick_id,
                    "no_candidates",
                    "no_candidates",
                    monologue_emit_allowed,
                )
                .await?;
            }
            self.persist_monologue_patch(&state).await;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "silent_cycle",
                    "reason": "monologue_tick",
                }),
            )
            .await;
            let duration_ms = tick_started.elapsed().as_millis() as i64;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "timing_monologue_tick",
                    "duration_ms": duration_ms,
                    "conversation_id": conversation_id,
                    "tick_id": tick_id,
                    "outcome": "no_candidates",
                }),
            )
            .await;
            let hold_ms = commit_hold_started.elapsed().as_millis() as i64;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "monologue_tick_lock_hold_ms",
                    "conversation_id": conversation_id,
                    "tick_id": tick_id,
                    "hold_ms": hold_ms,
                    "stage": "commit",
                }),
            )
            .await;
            no_candidates_logged = true;
        }
        if candidates.is_empty() {
            let reason = if no_candidates_logged {
                "no_candidates_logged"
            } else {
                "no_candidates"
            };
            candidates.push(self.make_candidate(
                CandidateKind::NoOp,
                json!({ "reason": reason }),
                "no_op",
                &mut created_at,
            ));
        }

        if let Some(framework_text) = extract_structured_framework_text(&deliberation) {
            let hypothesis_text = summarize_snippet(&framework_text, 220);
            if !hypothesis_exists(&state.workspace_active_hypotheses, &hypothesis_text) {
                let hypothesis = WorkspaceHypothesis {
                    text: hypothesis_text.clone(),
                    confidence: 0.7,
                    speculative: false,
                    evidence_event_ids: Vec::new(),
                    belief_ids: Vec::new(),
                    evidence_quality: None,
                };
                candidates.push(self.make_candidate(
                    CandidateKind::UpdateWorkspace,
                    json!({
                        "active_hypotheses": [{
                            "text": hypothesis.text,
                            "confidence": hypothesis.confidence
                        }]
                    }),
                    "hypothesis_promoted",
                    &mut created_at,
                ));
                state.hypothesis_defer_until = Some(state.monologue_count + 2);
                state.last_hypothesis_promoted_at = Some(Utc::now().to_rfc3339());
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "hypothesis_promoted",
                        "confidence": hypothesis.confidence,
                        "text": hypothesis.text,
                    }),
                )
                .await;
            }
        }

        let prev_ask_loop = state.ask_loop_breaker_triggered;
        let prev_tool_loop = state.tool_loop_breaker_triggered;
        let prev_emit_loop = state.monologue_emit_loop_breaker_triggered;
        self.apply_loop_detection(&candidates, &mut state, &settings);
        self.log_loop_breakers(None, None, &state, prev_ask_loop, prev_tool_loop)
            .await;
        let anchor_label_now = state
            .workspace_current_focus
            .clone()
            .unwrap_or_else(|| "current topic".to_string());
        self.maybe_emit_meta_cog_outcome(&mut state, &anchor_label_now, &settings)
            .await;
        self.apply_loop_detection(&candidates, &mut decision_state, &settings);
        if settings.monologue_stabilization_enabled.unwrap_or(true) {
            apply_emit_loop_detection_for(&candidates, &mut state, &settings);
            apply_emit_loop_detection_for(&candidates, &mut decision_state, &settings);
            if !prev_emit_loop && state.monologue_emit_loop_breaker_triggered {
                let quiet_until = Utc::now() + chrono::Duration::seconds(MONOLOGUE_QUIET_SECS);
                state.monologue_quiet_until = Some(quiet_until.to_rfc3339());
                decision_state.monologue_quiet_until = state.monologue_quiet_until.clone();
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "monologue_emit_loop_breaker",
                        "quiet_until": state.monologue_quiet_until,
                    }),
                )
                .await;
                state.last_meta_cog_loop_break_reason = Some("emit_loop_breaker".to_string());
                let quiet_until = state.monologue_quiet_until.clone();
                self.log_meta_cog_event(
                    &mut state,
                    None,
                    None,
                    json!({
                        "event": "meta_cog_event",
                        "reason": "emit_loop_breaker",
                        "quiet_until": quiet_until,
                    }),
                )
                .await;
            }
        }

        let wave_context = self.wave_arbitration_context(None, None).await;
        let narrative_wave_state = wave_context.as_ref().map(|ctx| ctx.state.clone());
        let qualia_state = qualia::compute_qualia_state(&self.db, None).await.ok();
        let qualia_context = qualia_state
            .as_ref()
            .and_then(|state| build_qualia_modulation_context(state));
        let residual_context = self
            .residual_influence_context(&decision_state, None)
            .await;
        let mut decision_internal = self.arbitrate(
            &candidates,
            &settings,
            &decision_state,
            true,
            None,
            wave_context.clone(),
            qualia_context.clone(),
            residual_context.clone(),
            None,
            None,
        );
        let mut decision_proactive = self.arbitrate(
            &candidates,
            &settings,
            &decision_state,
            false,
            None,
            wave_context,
            qualia_context,
            residual_context,
            None,
            None,
        );
        self.defer_throttled_tools(&mut decision_internal, &decision_state).await;
        self.defer_throttled_tools(&mut decision_proactive, &decision_state).await;
        self.log_tool_rejections(&decision_internal.rejected).await;
        self.log_tool_rejections(&decision_proactive.rejected).await;
        self.log_tool_bypasses(&decision_internal, &decision_state, &settings).await;
        self.log_tool_bypasses(&decision_proactive, &decision_state, &settings).await;

        let proactive_candidate = decision_proactive
            .accepted
            .iter()
            .find(|c| candidate_user_visible(c))
            .cloned();
        let proactive_has_tool = decision_proactive
            .accepted
            .iter()
            .any(|c| matches!(c.kind, CandidateKind::ToolCall));
        let mut monologue_intent_record: Option<(String, String)> = None;

        if let Some(candidate) = proactive_candidate {
            let mut candidate = candidate;
            let is_monologue_intent = is_monologue_source(&candidate.source)
                && matches!(
                    candidate.kind,
                    CandidateKind::EmitMessage | CandidateKind::AskUserQuestion | CandidateKind::FlagForHuman
                );
            let is_monologue_question = is_monologue_intent && matches!(candidate.kind, CandidateKind::AskUserQuestion);
            let allow_monologue_question = if is_monologue_question {
                let last_user_input = state.last_user_input.as_deref().unwrap_or("");
                monologue_followup_allowed(last_user_input)
            } else {
                true
            };
            if is_monologue_intent {
                if !allow_monologue_question {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "monologue_question_suppressed",
                            "reason": "trivial_user_input",
                            "candidate_kind": format!("{:?}", candidate.kind),
                            "source": candidate.source,
                        }),
                    )
                    .await;
                } else if let Some((prompt_id, bridge_id)) = record_monologue_intent(
                    self,
                    conversation_id,
                    candidate_alignment_text(&candidate).as_deref().unwrap_or(""),
                    &format!("{:?}", candidate.kind),
                )
                .await
                {
                    candidate.payload["bridge_id"] = json!(bridge_id);
                    candidate.payload["pending_prompt_id"] = json!(prompt_id);
                    monologue_intent_record = Some((prompt_id, bridge_id));
                }
            }
            if proactive_has_tool && !is_monologue_intent {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "proactive_throttle",
                        "reason": "tool_call_present",
                    }),
                )
                .await;
            } else {
                let disable_working_hypothesis = settings.stability_disable_working_hypothesis.unwrap_or(true);
                if let Some(prepared) =
                    self.prepare_proactive_candidate(&state, &candidate, disable_working_hypothesis, is_monologue_intent)
                        .await
                {
                    let mut decision_for_emit = decision_proactive.clone();
                    for accepted in decision_for_emit.accepted.iter_mut() {
                        if accepted.id == prepared.id {
                            *accepted = prepared.clone();
                            break;
                        }
                    }
                    let (overlap, exact_open_question) =
                        candidate_alignment_metrics(&prepared, &state);
                    let allow_emit = if is_monologue_intent {
                        if is_monologue_question {
                            allow_monologue_question
                        } else {
                            true
                        }
                    } else {
                        self.proactive_emit_allowed(&state, &prepared, overlap, exact_open_question).await
                    };
                    if allow_emit {
                        let _run_meta = self.run_proactive_emit(
                            &mut state,
                            decision_for_emit,
                            &prepared,
                            conversation_id,
                            &settings,
                            overlap,
                            exact_open_question,
                        )
                        .await?;
                        self.mark_candidate_outcomes(&decision_proactive, "executed", "rejected")
                            .await;
                        let hold_ms = commit_hold_started.elapsed().as_millis() as i64;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            Some(&tick_id),
                            json!({
                                "event": "monologue_tick_lock_hold_ms",
                                "conversation_id": conversation_id,
                                "tick_id": tick_id,
                                "hold_ms": hold_ms,
                                "stage": "commit",
                            }),
                        )
                        .await;
                        return Ok(());
                    } else if is_monologue_intent {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "intent_blocked",
                                "candidate_id": prepared.id,
                                "candidate_kind": format!("{:?}", prepared.kind),
                                "source": prepared.source,
                                "bridge_id": prepared.payload.get("bridge_id").and_then(|v| v.as_str()),
                            }),
                        )
                        .await;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "meta_cog_intent_blocked",
                                "candidate_id": prepared.id,
                                "candidate_kind": format!("{:?}", prepared.kind),
                                "source": prepared.source,
                                "bridge_id": prepared.payload.get("bridge_id").and_then(|v| v.as_str()),
                            }),
                        )
                        .await;
                    }
                }
            }
        }

        if let Some(selection) = self
            .select_pending_prompt_for_proactive(conversation_id, &state, &settings, false, None)
            .await
        {
            let now = Utc::now();
            let evidence_ok = selection.exact_open_question || controller_evidence_ok(&state);
            if !evidence_ok {
                let attempt_at = now.to_rfc3339();
                let _ = self
                    .db
                    .mark_pending_prompt_attempt(&selection.prompt_id, &attempt_at)
                    .await;
                self.note_open_question_attempt(
                    &mut state,
                    conversation_id,
                    &selection.prompt,
                    now,
                    true,
                    None,
                    None,
                )
                .await;

                let new_attempt = selection.attempt_count.saturating_add(1);
                let expired = timestamp_expired(selection.expires_at.as_deref(), now);
                if new_attempt >= PENDING_PROMPT_ATTEMPT_LIMIT || expired {
                    let context_hash = context_hash_for_drop(&state, &selection.prompt);
                    let _ = self
                        .db
                        .enqueue_deferred_item(
                            conversation_id,
                            "pending_prompt",
                            &selection.prompt,
                            Some(&selection.source),
                            "insufficient_evidence",
                            Some(&context_hash),
                            Some("new_evidence_or_user_request"),
                            new_attempt,
                            Some(&attempt_at),
                            selection.expires_at.as_deref(),
                        )
                        .await;
                    let _ = self.db.delete_pending_prompt(&selection.prompt_id).await;
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "pending_prompt_dropped",
                            "reason": "insufficient_evidence",
                            "candidate_id": selection.prompt_id,
                            "attempt_count": new_attempt,
                            "expires_at": selection.expires_at,
                        }),
                    )
                    .await;
                    if let Ok(count) = self.db.count_pending_prompts(conversation_id).await {
                        let _ = self.app_handle.emit("pending_prompt_count", count as usize);
                    }
                    self.persist_monologue_patch(&state).await;
                    let hold_ms = commit_hold_started.elapsed().as_millis() as i64;
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        Some(&tick_id),
                        json!({
                            "event": "monologue_tick_lock_hold_ms",
                            "conversation_id": conversation_id,
                            "tick_id": tick_id,
                            "hold_ms": hold_ms,
                            "stage": "commit",
                        }),
                    )
                    .await;
                    return Ok(());
                }
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "proactive_emit_blocked_no_evidence",
                        "candidate_id": selection.prompt_id,
                        "reason": "pending_prompt_evidence_gate",
                    }),
                )
                .await;
                self.log_pending_prompt_surface_attempted(
                    &selection.prompt_id,
                    &selection.source,
                    selection.auto_surface,
                    "blocked",
                    "pending_prompt_evidence_gate",
                    selection.overlap_workspace,
                    selection.overlap_user,
                    selection.age_seconds,
                    selection.skip_count,
                    selection.anchor_age_seconds,
                )
                .await;
                self.persist_monologue_patch(&state).await;
            } else {
                let allow_speculative_markers = allow_speculative_markers_for_prompt(
                    state.last_user_input.as_deref().unwrap_or(""),
                    false,
                );
                let mut question =
                    strip_working_hypothesis_prefix(&selection.prompt, !allow_speculative_markers);
                if question.trim().is_empty() {
                    question = selection.prompt.clone();
                }
                let user_name = settings.user_display_name.as_deref().unwrap_or("User");
                let last_user_input = state.last_user_input.as_deref().unwrap_or("");
                if response_has_user_attribution(&question, user_name)
                    && !user_attribution_grounded_in_last_input(&question, last_user_input)
                {
                    question = rewrite_user_attribution_text(&question, user_name);
                }
                if !selection.exact_open_question {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!( {
                            "event": "speculation_marked",
                            "candidate_id": selection.prompt_id,
                            "reason": "pending_prompt",
                        }),
                    )
                    .await;
                }
                let payload = json!({
                    "question": question,
                    "content": question,
                    "pending_prompt_id": selection.prompt_id,
                    "speculative": !selection.exact_open_question,
                    "bridge_id": selection.bridge_id,
                    "intent_kind": selection.intent_kind,
                });
                let mut candidate = Candidate {
                    id: selection.prompt_id.clone(),
                    kind: CandidateKind::AskUserQuestion,
                    payload,
                    evidence_event_ids: Vec::new(),
                    belief_ids: Vec::new(),
                    target_scope: None,
                    rationale: None,
                    expected_outcome: None,
                    cost: Some(0),
                    urgency: Some(0),
                    source: "pending_prompt".to_string(),
                    priority_class: priority_class_for(&CandidateKind::AskUserQuestion),
                    priority_rank: 0,
                    created_at: state.monologue_count,
                };
                candidate.refresh_meta();
                let decision = KernelDecision {
                    accepted: vec![candidate.clone()],
                    rejected: Vec::new(),
                    caps_applied: Vec::new(),
                    report: DecisionReport::default(),
                };
                let (overlap, exact_open_question) =
                    candidate_alignment_metrics(&candidate, &state);
                let attempt_at = now.to_rfc3339();
                let _ = self
                    .db
                    .mark_pending_prompt_attempt(&selection.prompt_id, &attempt_at)
                    .await;
                self.note_open_question_attempt(
                    &mut state,
                    conversation_id,
                    &selection.prompt,
                    now,
                    false,
                    None,
                    None,
                )
                .await;
                let allow_emit = true;
                if allow_emit {
                    let claimed = match self.db.delete_pending_prompt(&selection.prompt_id).await {
                        Ok(affected) if affected > 0 => true,
                        Ok(_) => {
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "warn",
                                "chat",
                                None,
                                None,
                                json!({
                                    "event": "pending_prompt_delete_failed",
                                    "reason": "not_found",
                                    "prompt_id": selection.prompt_id,
                                }),
                            )
                            .await;
                            false
                        }
                        Err(err) => {
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "warn",
                                "chat",
                                None,
                                None,
                                json!({
                                    "event": "pending_prompt_delete_failed",
                                    "reason": "db_error",
                                    "prompt_id": selection.prompt_id,
                                    "error": err.to_string(),
                                }),
                            )
                            .await;
                            false
                        }
                    };
                    if !claimed {
                        let hold_ms = commit_hold_started.elapsed().as_millis() as i64;
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            Some(&tick_id),
                            json!({
                                "event": "monologue_tick_lock_hold_ms",
                                "conversation_id": conversation_id,
                                "tick_id": tick_id,
                                "hold_ms": hold_ms,
                                "stage": "commit",
                            }),
                        )
                        .await;
                        return Ok(());
                    }
                    let _run_meta = self.run_proactive_emit(
                        &mut state,
                        decision,
                        &candidate,
                        conversation_id,
                        &settings,
                        overlap,
                        exact_open_question,
                    )
                    .await?;
                    self.mark_candidate_outcomes(&decision_internal, "skipped", "skipped")
                        .await;
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "chat",
                        None,
                        None,
                        json!({
                            "event": "pending_prompt_surfaced",
                            "source": selection.source,
                            "alignment_enabled": settings.pending_prompt_alignment_enabled.unwrap_or(true),
                            "overlap_workspace": selection.overlap_workspace,
                            "overlap_user": selection.overlap_user,
                            "prompt_age_seconds": selection.age_seconds,
                            "pending_prompt_starvation_count": selection.skip_count,
                            "candidate_id": candidate.id,
                        }),
                    )
                    .await;
                    self.log_pending_prompt_surface_attempted(
                        &selection.prompt_id,
                        &selection.source,
                        selection.auto_surface,
                        "surfaced",
                        "delivered",
                        selection.overlap_workspace,
                        selection.overlap_user,
                        selection.age_seconds,
                        selection.skip_count,
                        selection.anchor_age_seconds,
                    )
                    .await;
                    if let Ok(count) = self.db.count_pending_prompts(conversation_id).await {
                        let _ = self.app_handle.emit("pending_prompt_count", count);
                    }
                    return Ok(());
                } else {
                    self.log_pending_prompt_surface_attempted(
                        &selection.prompt_id,
                        &selection.source,
                        selection.auto_surface,
                        "blocked",
                        "emit_blocked",
                        selection.overlap_workspace,
                        selection.overlap_user,
                        selection.age_seconds,
                        selection.skip_count,
                        selection.anchor_age_seconds,
                    )
                    .await;
                }
            }
        }

        if let Some(primary_candidate) = decision_internal.accepted.first().cloned() {
            if let Some((subject_state, snapshot)) = self
                .build_and_persist_subject_snapshot(
                    &mut state,
                    log_run_id,
                    Some(tick_id.as_str()),
                    "monologue_gate",
                )
                .await
            {
                let proposal = subject_controller::build_action_proposal(&primary_candidate);
                if let Err(err) = subject_controller::persist_action_proposal(&self.db, &snapshot.snapshot_hash, &proposal).await {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "monologue_gate_failed",
                            "reason": "persist_action_proposal",
                            "error": err,
                        }),
                    )
                    .await;
                    let allowed = state.stop_state.allowed_capabilities();
                    self.persist_monologue_status(
                        &mut state,
                        conversation_id,
                        &tick_id,
                        "suppressed",
                        "gate_failed",
                        allowed.monologue_emit,
                    )
                    .await?;
                    self.persist_monologue_patch(&state).await;
                    return Ok(());
                }
                let tool_names = self.tools.allowed_tool_names(&settings);
                let anchor_vocab = build_anchor_vocab(&state, &tool_names);
                let anchor_hits = count_anchor_hits(&candidate_relevance_text(&primary_candidate), &anchor_vocab);
                let signals = self
                    .compute_gate_signals_score(&state, &subject_state, Some(&decision_internal), &primary_candidate, anchor_hits, &settings)
                    .await;
                self.commit_gate_signal_state(&mut state, &signals);

                let soft_gate = subject_controller::build_gate_decision(
                    &subject_state,
                    &primary_candidate,
                    &state.stop_state,
                    &signals,
                );
                let legacy_gate = subject_controller::build_gate_decision_legacy(
                    &subject_state,
                    &primary_candidate,
                    &state.stop_state,
                );
                let soft_decision = soft_gate.decision.clone();
                let legacy_decision = legacy_gate.decision.clone();
                let rollout_percent = settings.gate_rollout_percent.unwrap_or(100).clamp(0, 100);
                let shadow_mode = settings.gate_shadow_mode.unwrap_or(false);
                let rollout_bucket = gate_rollout_bucket(conversation_id);
                let gate_penalty_enabled = settings.gate_penalty_integration.unwrap_or(true);
                let use_soft_gate = gate_penalty_enabled
                    && !shadow_mode
                    && (rollout_percent >= 100 || rollout_bucket < rollout_percent);

                let soft_reasons: Vec<String> = serde_json::from_str::<Value>(&soft_gate.evidence_refs_json)
                    .ok()
                    .and_then(|v| v.get("reasons").and_then(|r| r.as_array()).cloned())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|item| item.as_str())
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();

                let soft_penalty = if use_soft_gate {
                    crate::core::subject_controller::gate_penalty_for_candidate(
                        &subject_state,
                        &primary_candidate,
                        &state.stop_state,
                        &signals,
                        settings.learning_feedback.unwrap_or(true),
                    )
                } else {
                    crate::core::subject_controller::GatePenalty {
                        penalty: 0.0,
                        reasons: Vec::new(),
                    }
                };

                let mut hard_decision: Option<String> = None;
                let mut hard_reasons: Vec<String> = Vec::new();
                let mut tool_gate_detail: Option<String> = None;
                if use_soft_gate {
                    if state.stop_state.active {
                        hard_decision = Some("DEFER".to_string());
                        hard_reasons.push("stop_state_active".to_string());
                    }
                    if !self.plan_preconditions_met(&state, &primary_candidate) {
                        hard_decision = Some("VERIFY".to_string());
                        hard_reasons.push("plan_precondition_unmet".to_string());
                    }
                    if matches!(primary_candidate.kind, CandidateKind::ToolCall) {
                        let tool_name = primary_candidate
                            .payload
                            .get("tool_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        if !tool_name.is_empty() {
                            let args_json = primary_candidate
                                .payload
                                .get("arguments")
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "{}".to_string());
                            let tool_gate = self.tool_gate_decision(&tool_name, &args_json, &settings, true);
                            if !tool_gate.allowed {
                                hard_decision = Some("DENY".to_string());
                                let reason = tool_gate
                                    .reason
                                    .unwrap_or_else(|| "TOOL_BLOCK".to_string());
                                hard_reasons.push(format!("tool_contract_{}", reason.to_lowercase()));
                                tool_gate_detail = tool_gate.detail.clone();
                            }
                        }
                    }
                }

                let hard_gate = hard_decision.clone().map(|decision_label| subject_controller::GateDecisionRecord {
                    decision_id: Uuid::new_v4().to_string(),
                    decision: decision_label,
                    evidence_refs_json: json!({ "reasons": hard_reasons }).to_string(),
                    metrics_json: json!({
                        "hard_gate": true,
                        "tool_gate_detail": tool_gate_detail,
                    })
                    .to_string(),
                });

                let mut gate = if !use_soft_gate {
                    legacy_gate
                } else if let Some(hard_gate) = hard_gate {
                    hard_gate
                } else {
                    let mut gate = soft_gate.clone();
                    if !gate_allows_response(&gate.decision) {
                        gate.decision = "ALLOW_WITH_NOTICE".to_string();
                    }
                    gate
                };

                let mut metrics = serde_json::from_str::<Value>(&gate.metrics_json).unwrap_or_else(|_| json!({}));
                if let Some(obj) = metrics.as_object_mut() {
                    obj.insert("soft_gate_decision".to_string(), json!(soft_gate.decision));
                    obj.insert("soft_gate_reasons".to_string(), json!(soft_reasons.clone()));
                    obj.insert("gate_penalty".to_string(), json!(soft_penalty.penalty));
                    obj.insert("gate_penalty_reasons".to_string(), json!(soft_penalty.reasons.clone()));
                    obj.insert("hard_gate_triggered".to_string(), json!(use_soft_gate && hard_decision.is_some()));
                }
                gate.metrics_json = metrics.to_string();
                let gate_reasons_log = serde_json::from_str::<Value>(&gate.evidence_refs_json)
                    .ok()
                    .and_then(|value| value.get("reasons").cloned())
                    .unwrap_or_else(|| json!([]));
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    Some(&tick_id),
                    json!({
                        "event": "gate_decision_inputs",
                        "candidate_id": primary_candidate.id,
                        "candidate_kind": format!("{:?}", primary_candidate.kind),
                        "anchor_hits": anchor_hits,
                        "signals": signals,
                        "soft_decision": soft_decision,
                        "legacy_decision": legacy_decision,
                        "enforced_decision": gate.decision,
                        "gate_reasons": gate_reasons_log,
                        "gate_penalty": soft_penalty.penalty,
                        "gate_penalty_reasons": soft_penalty.reasons,
                        "hard_gate_decision": hard_decision,
                        "shadow_mode": shadow_mode,
                        "rollout_percent": rollout_percent,
                        "rollout_bucket": rollout_bucket,
                        "execution_mode": "monologue",
                        "organism": subject_state.organism,
                    }),
                )
                .await;
                if let Err(err) = subject_controller::persist_gate_decision(
                    &self.db,
                    &snapshot.snapshot_hash,
                    &proposal.proposal_id,
                    &gate,
                )
                .await
                {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "monologue_gate_failed",
                            "reason": "persist_gate_decision",
                            "error": err,
                        }),
                    )
                    .await;
                    let allowed = state.stop_state.allowed_capabilities();
                    self.persist_monologue_status(
                        &mut state,
                        conversation_id,
                        &tick_id,
                        "suppressed",
                        "gate_failed",
                        allowed.monologue_emit,
                    )
                    .await?;
                    self.persist_monologue_patch(&state).await;
                    return Ok(());
                }
                decision_internal.report.snapshot_hash = Some(snapshot.snapshot_hash.clone());
                decision_internal.report.gate_decision_id = Some(gate.decision_id.clone());
                decision_internal.report.gate_decision = Some(gate.decision.clone());
                decision_internal.report.gate_penalty = Some(soft_penalty.penalty as f64);
                if !soft_penalty.reasons.is_empty() {
                    decision_internal.report.gate_penalty_reasons = Some(soft_penalty.reasons.clone());
                }
                decision_internal.report.soft_gate_decision = Some(soft_decision);
                if !soft_reasons.is_empty() {
                    decision_internal.report.soft_gate_reasons = Some(soft_reasons.clone());
                }
                let reasons: Vec<String> = serde_json::from_str::<Value>(&gate.evidence_refs_json)
                    .ok()
                    .and_then(|v| v.get("reasons").and_then(|r| r.as_array()).cloned())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|item| item.as_str())
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                if !reasons.is_empty() {
                    decision_internal.report.gate_reasons = Some(reasons.clone());
                }
                decision_internal.report.gate_notice = gate_notice_for(&gate.decision, &reasons);
                if !gate_allows_response(&gate.decision) {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        None,
                        None,
                        json!({
                            "event": "monologue_gate_blocked",
                            "decision": gate.decision,
                            "snapshot_hash": snapshot.snapshot_hash,
                            "candidate_id": primary_candidate.id,
                        }),
                    )
                    .await;
                    let mut gate_rejections = Vec::new();
                    for candidate in decision_internal.accepted.iter() {
                        gate_rejections.push(RejectedCandidate {
                            id: candidate.id.clone(),
                            kind: candidate.kind.clone(),
                            reason: "gate_decision".to_string(),
                            tool_name: None,
                            source: Some(candidate.source.clone()),
                            is_monologue: Some(is_monologue_source(&candidate.source)),
                            payload: if matches!(candidate.kind, CandidateKind::ToolCall) {
                                Some(candidate.payload.clone())
                            } else {
                                None
                            },
                        });
                    }
                    decision_internal.rejected.extend(gate_rejections);
                    decision_internal.accepted.clear();
                }
            }
        }

        let commit_started = Instant::now();
        let completed_at = Utc::now().to_rfc3339();
        state.last_monologue_completed_at = Some(completed_at.clone());
        state.last_monologue_at = Some(completed_at);
        let commit_result = self
            .commit_cycle(
                &mut state,
                &decision_internal,
                conversation_id,
                None,
                None,
                &settings,
                true,
                None,
                false,
            )
            .await?;
        let commit_ms = commit_started.elapsed().as_millis() as i64;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            None,
            json!({
                "event": "timing_commit_cycle",
                "duration_ms": commit_ms,
                "path": "monologue_tick",
            }),
        )
        .await;
        self.mark_candidate_outcomes(&decision_internal, "executed", "rejected")
            .await;

        if let Some(question) = commit_result.ask_question.as_deref() {
            let trimmed = question.trim();
            if !trimmed.is_empty() {
                let last_user_input = state.last_user_input.as_deref().unwrap_or("");
                let allow_followup = monologue_followup_allowed(last_user_input);
                let mut pending_prompt_id: Option<String> = None;
                let mut bridge_id: Option<String> = None;
                if let Some((prompt_id, bridge)) = monologue_intent_record.take() {
                    if allow_followup {
                        pending_prompt_id = Some(prompt_id);
                        bridge_id = Some(bridge);
                    }
                } else {
                    let mut derived_from_user = false;
                    if allow_followup && !last_user_input.trim().is_empty() {
                        let question_tokens = super::token_set(trimmed);
                        let user_tokens = super::token_set(last_user_input);
                        derived_from_user = !question_tokens.is_empty()
                            && !user_tokens.is_empty()
                            && question_tokens.intersection(&user_tokens).count()
                                >= super::CLARIFIER_OVERLAP_THRESHOLD;
                    }
                    if derived_from_user {
                        if let Some((prompt_id, bridge)) =
                            record_monologue_intent(self, conversation_id, trimmed, "AskUserQuestion").await
                        {
                            pending_prompt_id = Some(prompt_id);
                            bridge_id = Some(bridge);
                        }
                    }
                }
                let payload = json!({
                    "question": trimmed,
                    "content": trimmed,
                    "requested_slots": commit_result.ask_slots.clone(),
                    "pending_prompt_id": pending_prompt_id,
                    "bridge_id": bridge_id,
                    "evidence_event_ids": [],
                });
                let mut candidate = Candidate {
                    id: pending_prompt_id.clone().unwrap_or_else(|| Uuid::new_v4().to_string()),
                    kind: CandidateKind::AskUserQuestion,
                    payload,
                    evidence_event_ids: Vec::new(),
                    belief_ids: Vec::new(),
                    target_scope: None,
                    rationale: None,
                    expected_outcome: None,
                    cost: Some(0),
                    urgency: Some(0),
                    source: "monologue".to_string(),
                    priority_class: priority_class_for(&CandidateKind::AskUserQuestion),
                    priority_rank: 0,
                    created_at: state.monologue_count,
                };
                candidate.refresh_meta();
                let evidence_ids = super::tools::extract_id_list(&candidate.payload, "evidence_event_ids");
                let derived_from_user = if allow_followup {
                    super::prompt::candidate_overlaps_last_user_input(&candidate, last_user_input)
                        || super::prompt::candidate_mentions_last_user_input(&candidate, last_user_input)
                } else {
                    false
                };
                let grounded = if settings.monologue_provenance_guard.unwrap_or(true) {
                    if pending_prompt_id.is_some() {
                        true
                    } else if evidence_ids.is_empty() {
                        false
                    } else {
                        let validation = self.validate_evidence_ids(&evidence_ids, &[], false).await;
                        let has_allowed_source = validation.source_types.iter().any(|source| {
                            matches!(source.as_str(), "user" | "user_focus" | "tool")
                        });
                        !validation.valid_evidence_ids.is_empty() && has_allowed_source
                    }
                } else {
                    derived_from_user || !evidence_ids.is_empty()
                };
                if !grounded {
                    let reason = if !allow_followup {
                        "trivial_user_input"
                    } else {
                        "monologue_unanchored_question"
                    };
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        None,
                        None,
                        json!( {
                            "event": "monologue_question_suppressed",
                            "reason": reason,
                            "candidate_kind": "AskUserQuestion",
                            "source": "monologue",
                        }),
                    )
                    .await;
                } else {
                    let decision = KernelDecision {
                        accepted: vec![candidate.clone()],
                        rejected: Vec::new(),
                        caps_applied: Vec::new(),
                        report: DecisionReport::default(),
                    };
                    let (overlap, exact_open_question) = candidate_alignment_metrics(&candidate, &state);
                    if self
                        .proactive_emit_allowed(&state, &candidate, overlap, exact_open_question)
                        .await
                    {
                        let _run_meta = self.run_proactive_emit(
                            &mut state,
                            decision,
                            &candidate,
                            conversation_id,
                            &settings,
                            overlap,
                            exact_open_question,
                        )
                        .await?;
                    } else {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "intent_blocked",
                                "candidate_id": candidate.id,
                                "candidate_kind": "AskUserQuestion",
                                "source": "monologue",
                                "bridge_id": bridge_id,
                            }),
                        )
                        .await;
                    }
                }
            }
        }

        if !commit_result.tool_dispatches.is_empty() {
            let (_tx, mut cancel_rx) = watch::channel(false);
            for tool_dispatch in commit_result.tool_dispatches.iter() {
                let _ = self
                    .dispatch_tool(tool_dispatch, None, None, &mut cancel_rx)
                    .await;
            }
        }

        if let Some(thread_run) = commit_result.thread_run {
            let _ = self
                .run_thread(conversation_id, &thread_run.thread_id, &thread_run.goal, thread_run.depth, &settings)
                .await;
        }

        if commit_result.research_cost > 0 {
            let remaining = self.research_budget_remaining(&state, &settings);
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "research_budget_usage",
                    "consumed": commit_result.research_cost,
                    "remaining": remaining,
                    "window_start": state.research_window_start,
                }),
            )
            .await;
        }

        if let Some(content) = commit_result.emit_content.as_deref() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "monologue_emit",
                    "decision_mode": decision_mode,
                    "content_len": content.len(),
                }),
            )
            .await;
            if self.allow_internal_emit(&state, &settings) {
                let _ = self.emit_internal_message(conversation_id, content).await;
            } else {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "internal_emit_blocked",
                        "conversation_id": conversation_id,
                    }),
                )
                .await;
            }
        } else {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "silent_cycle",
                    "reason": "monologue_tick",
                }),
            )
            .await;
        }

        self
            .maybe_write_reflective_narrative(
                conversation_id,
                qualia_state.as_ref(),
                state.controller_state.as_ref(),
                narrative_wave_state.as_ref(),
            )
            .await;

        let self_model_update_needed = emitted_entries || state_change_candidates > 0;
        if self_model_update_needed {
            self.sync_unified_self_model(&mut state).await;
        } else {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                None,
                Some(&tick_id),
                json!({
                    "event": "self_model_update_skipped",
                    "reason": "monologue_no_state_change",
                    "conversation_id": conversation_id,
                    "tick_id": tick_id,
                }),
            )
            .await;
        }

        let outcome = if emitted_entries {
            "saved"
        } else if status_written {
            "status_only"
        } else {
            "no_candidates"
        };
        state.last_monologue_tick_outcome = Some(outcome.to_string());
        state.last_monologue_status_emitted = Some(status_written || emitted_entries);
        state.last_monologue_visible = Some(monologue_emit_allowed && (status_written || emitted_entries));
        self.persist_monologue_patch(&state).await;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            Some(&tick_id),
            json!({
                "event": "monologue_tick_result",
                "conversation_id": conversation_id,
                "tick_id": tick_id,
                "outcome": outcome,
                "suppressed_reason": Value::Null,
                "emitted_entries": if emitted_entries { 1 } else { 0 },
                "status_only": status_written && !emitted_entries,
            }),
        )
        .await;

        let suppressed_candidates = monologue_blocked_count + decision_internal.rejected.len();
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            Some(&tick_id),
            json!({
                "event": "monologue_loop_outcome",
                "conversation_id": conversation_id,
                "tick_id": tick_id,
                "state_change_candidates": state_change_candidates,
                "non_state_change_candidates": non_state_change_candidates,
                "suppressed_candidates": suppressed_candidates,
                "no_op_reason": loop_noop_reason,
            }),
        )
        .await;

        let duration_ms = tick_started.elapsed().as_millis() as i64;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            Some(&tick_id),
            json!({
                "event": "monologue_tick_end",
                "conversation_id": conversation_id,
                "tick_id": tick_id,
                "duration_ms": duration_ms,
                "outcome": outcome,
            }),
        )
        .await;

        let hold_ms = commit_hold_started.elapsed().as_millis() as i64;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            Some(&tick_id),
            json!({
                "event": "monologue_tick_lock_hold_ms",
                "conversation_id": conversation_id,
                "tick_id": tick_id,
                "hold_ms": hold_ms,
                "stage": "commit",
            }),
        )
        .await;

        Ok(())
    }

    async fn maybe_write_reflective_narrative(
        &self,
        conversation_id: &str,
        qualia_state: Option<&qualia::QualiaState>,
        controller_state: Option<&crate::models::ControllerState>,
        wave_state: Option<&WaveStateVector>,
    ) {
        let Some(narrative) =
            build_reflective_narrative_text(qualia_state, controller_state, wave_state)
        else {
            return;
        };

        let mut evidence_ids = self
            .db
            .get_recent_evidence_ids_by_source_types(
                &[
                    "qualia_snapshot",
                    "wave_state",
                    "attention_schema_snapshot",
                    "prediction_residual_snapshot",
                ],
                6,
            )
            .await;

        if evidence_ids.is_empty() {
            if let Some(qualia_state) = qualia_state {
                let snapshot = format_qualia_snapshot(qualia_state);
                if !snapshot.trim().is_empty() && snapshot.trim() != "None" {
                    if let Some(event_id) = self
                        .db
                        .create_qualia_snapshot_evidence_event(
                            conversation_id,
                            &snapshot,
                            Some("reflective_narrative"),
                        )
                        .await
                    {
                        evidence_ids.push(event_id);
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            None,
                            None,
                            json!({
                                "event": "reflective_narrative_evidence_fallback",
                                "evidence_event_id": event_id,
                            }),
                        )
                        .await;
                    }
                }
            }
        }

        if evidence_ids.is_empty() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "kernel",
                None,
                None,
                json!({
                    "event": "reflective_narrative_written",
                    "status": "skipped",
                    "reason": "missing_evidence_ids",
                }),
            )
            .await;
            return;
        }

        let snippet = summarize_snippet(&narrative, 220);
        match self_memory::write_self_fact(
            &self.db.pool,
            "reflective_narrative",
            &narrative,
            &snippet,
            Some(Utc::now()),
            crate::core::memory::types::SourceType::System,
            Some(&evidence_ids),
        )
        .await
        {
            Ok(result) => {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "reflective_narrative_written",
                        "belief_id": result.belief_id,
                        "evidence_event_id": result.evidence_event_id,
                        "evidence_event_ids": evidence_ids,
                    }),
                )
                .await;
            }
            Err(err) => {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    None,
                    None,
                    json!({
                        "event": "reflective_narrative_written",
                        "status": "failed",
                        "error": err,
                    }),
                )
                .await;
            }
        }
    }

    async fn build_meta_cognitive_observation_candidate(
        &self,
        conversation_id: &str,
        created_at: &mut i64,
    ) -> Option<Candidate> {
        let row = sqlx::query(
            "SELECT g.decision, g.evidence_refs_json, g.metrics_json, g.created_at
             FROM gate_decisions g
             JOIN subject_snapshots s ON s.snapshot_hash = g.snapshot_hash
             WHERE s.conversation_id = ?
             ORDER BY datetime(g.created_at) DESC
             LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten()?;

        let decision: String = row.try_get("decision").unwrap_or_default();
        if decision.trim().is_empty() {
            return None;
        }

        let evidence_refs_raw: String = row.try_get("evidence_refs_json").unwrap_or_else(|_| "{}".to_string());
        let metrics_raw: String = row.try_get("metrics_json").unwrap_or_else(|_| "{}".to_string());
        let evidence_refs: Value = serde_json::from_str(&evidence_refs_raw).unwrap_or_else(|_| json!({}));
        let metrics: Value = serde_json::from_str(&metrics_raw).unwrap_or_else(|_| json!({}));

        let reasons: Vec<String> = evidence_refs
            .get("reasons")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let gate_signals = metrics.get("gate_signals").cloned().unwrap_or_else(|| json!({}));

        let mut clauses: Vec<String> = Vec::new();
        if decision != "ALLOW" {
            clauses.push(format!("the gate decision was {}", decision));
        }
        if gate_signals
            .get("uncertainty_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
            >= 0.6
        {
            clauses.push("uncertainty was high".to_string());
        }
        if gate_signals
            .get("novelty_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
            >= 0.6
        {
            clauses.push("novelty was high".to_string());
        }
        if gate_signals
            .get("requires_audit")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            clauses.push("an audit was required".to_string());
        }
        if clauses.is_empty() && reasons.is_empty() {
            return None;
        }

        let observation = if clauses.is_empty() {
            format!("I adjusted my response because the gate flagged: {}.", reasons.join(", "))
        } else if reasons.is_empty() {
            format!("I adjusted my response because {}.", clauses.join(" and "))
        } else {
            format!(
                "I adjusted my response because {} (signals: {}).",
                clauses.join(" and "),
                reasons.join(", ")
            )
        };

        let evidence_ids = self
            .db
            .get_recent_evidence_ids_by_source_types(
                &[
                    "qualia_snapshot",
                    "wave_state",
                    "attention_schema_snapshot",
                    "prediction_residual_snapshot",
                ],
                4,
            )
            .await;
        if evidence_ids.is_empty() {
            return None;
        }

        let decision_for_payload = decision.clone();
        let reasons_for_payload = reasons.clone();
        let payload = json!({
            "event_type": "meta_cognitive_observation",
            "payload": {
                "decision": decision_for_payload,
                "reasons": reasons_for_payload,
                "observation": observation,
                "gate_signals": gate_signals,
            },
            "source_type": "kernel",
            "source_ref": "gate_decision",
            "evidence_event_ids": evidence_ids,
        });

        let candidate = self.make_candidate(
            CandidateKind::WriteEpisodic,
            payload,
            "meta_cognitive_observation",
            created_at,
        );
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            None,
            None,
            json!({
                "event": "meta_cognitive_observation",
                "decision": decision,
                "reasons": reasons,
            }),
        )
        .await;
        Some(candidate)
    }

    pub(super) async fn build_monologue_semantic_hint(
        &self,
        conversation_id: &str,
        summary: &InnerSummary,
        state: &KernelState,
    ) -> String {
        let mut parts = Vec::new();
        if !summary.focus.trim().is_empty() {
            parts.push(summary.focus.clone());
        }
        if !summary.next_moves.is_empty() {
            parts.push(summary.next_moves.join(" "));
        }
        if !summary.blockers.is_empty() {
            parts.push(summary.blockers.join(" "));
        }
        if !summary.open_questions.is_empty() {
            parts.push(summary.open_questions.join(" "));
        }
        if parts.is_empty() {
            if let Some(last_input) = state.last_user_input.as_deref() {
                if !last_input.trim().is_empty() {
                    parts.push(last_input.to_string());
                }
            }
        }
        let query = parts.join(" | ").trim().to_string();
        if query.is_empty() {
            return "None".to_string();
        }

        let api = crate::core::memory::api::MemoryApi::new(
            self.db.pool.clone(),
            Some(self.model_client.clone()),
            format!("monologue:{}", conversation_id),
        )
        .await;
        let intent = crate::core::memory::api::infer_query_intent(&query);
        if let Ok(packet) = api
            .retrieve_fast(
                &query,
                &[crate::core::memory::types::Scope::Session, crate::core::memory::types::Scope::Global],
                intent,
            )
            .await
        {
            let mut hints = Vec::new();
            for fact in packet.facts.iter().take(4) {
                hints.push(format!("{}: {} = {}", fact.entity_label, fact.key, fact.value));
            }
            for rel in packet.relations.iter().take(3) {
                let participants = rel
                    .participants
                    .iter()
                    .map(|p| format!("{}:{}", p.role, p.entity_label))
                    .collect::<Vec<_>>()
                    .join(", ");
                hints.push(format!("{}({})", rel.rel_type, participants));
            }
            if hints.is_empty() {
                "None".to_string()
            } else {
                let hint_block = summarize_snippet(&hints.join(" | "), 280);
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "memory",
                    None,
                    None,
                    json!({
                        "event": "monologue_semantic_hint",
                        "snippet": summarize_snippet(&hint_block, 240),
                    }),
                )
                .await;
                hint_block
            }
        } else if let Ok(core) = self.db.get_semantic_core().await {
            if core.trim().is_empty() {
                "None".to_string()
            } else {
                core.chars().take(240).collect()
            }
        } else {
            "None".to_string()
        }
    }

    fn monologue_due(&self, state: &mut KernelState, settings: &crate::models::Settings) -> MonologueDue {
        let interval = settings.monologue_interval_seconds.unwrap_or(60).max(5);
        let max_per_hour = settings.monologue_max_per_hour.unwrap_or(0).max(0);
        let now = Utc::now();
        let priority_window_secs: i64 = 30;
        let freshness_override_secs: i64 = 60;
        let last_user_at = state
            .last_user_input_at
            .as_deref()
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
            .map(|ts| ts.with_timezone(&Utc));

        let mut last_at: Option<chrono::DateTime<Utc>> = None;
        for ts in [
            state.last_monologue_completed_at.as_deref(),
            state.last_monologue_at.as_deref(),
            state.last_monologue_started_at.as_deref(),
        ] {
            if let Some(parsed) = ts.and_then(|v| chrono::DateTime::parse_from_rfc3339(v).ok()) {
                let parsed = parsed.with_timezone(&Utc);
                if last_at.map(|current| parsed > current).unwrap_or(true) {
                    last_at = Some(parsed);
                }
            }
        }
        if let Some(last) = last_at {
            if now.signed_duration_since(last).num_seconds() < interval {
                let recent_user = last_user_at
                    .map(|user_ts| now.signed_duration_since(user_ts).num_seconds() <= priority_window_secs)
                    .unwrap_or(false);
                if !recent_user {
                    return MonologueDue::Skipped("interval");
                }
            }
        }

        let window_start = state
            .monologue_window_start
            .as_deref()
            .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
            .map(|ts| ts.with_timezone(&Utc))
            .unwrap_or(now);
        if now.signed_duration_since(window_start).num_seconds() >= 3600 {
            state.monologue_window_start = Some(now.to_rfc3339());
            state.monologue_count = 0;
        } else if max_per_hour > 0 && state.monologue_count >= max_per_hour {
            let freshness_override = last_user_at
                .and_then(|user_ts| {
                    last_at.map(|mono_ts| (user_ts > mono_ts, now.signed_duration_since(mono_ts).num_seconds()))
                })
                .map(|(is_newer, delta)| is_newer && delta >= freshness_override_secs)
                .unwrap_or(false);
            if freshness_override {
                state.monologue_count = 0;
                state.monologue_window_start = Some(now.to_rfc3339());
            } else {
            let until = window_start + chrono::Duration::seconds(3600);
            state.monologue_quiet_until = Some(until.to_rfc3339());
            return MonologueDue::Skipped("max_per_hour");
            }
        }

        state.monologue_count += 1;
        if state.monologue_window_start.is_none() {
            state.monologue_window_start = Some(now.to_rfc3339());
        }
        if let Some(until) = state.hypothesis_defer_until {
            if state.monologue_count >= until {
                state.hypothesis_defer_until = None;
            }
        }
        MonologueDue::Due
    }
}

fn build_reflective_narrative_text(
    qualia_state: Option<&qualia::QualiaState>,
    controller_state: Option<&crate::models::ControllerState>,
    wave_state: Option<&WaveStateVector>,
) -> Option<String> {
    let mut sentences: Vec<String> = Vec::new();

    if let Some(qualia) = qualia_state {
        let tag = qualia
            .dominant_tag
            .as_deref()
            .unwrap_or("neutral")
            .trim();
        let intensity = qualia.dominant_intensity;
        if tag != "neutral" && intensity >= 0.15 {
            sentences.push(format!(
                "I notice a {} posture because the dominant qualia tag is '{}' (intensity {:.2}).",
                tag,
                tag,
                intensity
            ));
        } else if tag != "neutral" && intensity > 0.0 {
            sentences.push(format!(
                "I notice a slight '{}' bias in my current signals.",
                tag
            ));
        }
    }

    if let Some(controller) = controller_state {
        if controller.uncertainty >= 0.6 {
            sentences.push(
                "My confidence is low, so I'm biasing toward verification.".to_string(),
            );
        } else if controller.uncertainty <= 0.3 {
            sentences.push(
                "My confidence is steady, so I'm comfortable proceeding.".to_string(),
            );
        }
    }

    if let Some(wave) = wave_state {
        if wave.turbulence > 0.7 {
            sentences.push(
                "Wave turbulence is elevated, so I'm reducing commitment.".to_string(),
            );
        } else if wave.coherence > 0.7 {
            sentences.push(
                "Wave coherence is high, so I'm keeping the plan direct.".to_string(),
            );
        }
    }

    if sentences.is_empty() {
        None
    } else {
        Some(sentences.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::sanitize_monologue_user_attribution;

    #[test]
    fn strips_user_attribution_without_input() {
        let user_name = "User";
        let message = "User said they were skeptical. I should verify the claims.";
        let (cleaned, stripped) = sanitize_monologue_user_attribution(message, "", user_name);
        assert!(stripped);
        assert_eq!(cleaned, "I should verify the claims");
    }

    #[test]
    fn keeps_user_attribution_with_overlap() {
        let user_name = "User";
        let last_input = "I am skeptical about this plan.";
        let message = "User said they were skeptical about this plan.";
        let (cleaned, stripped) = sanitize_monologue_user_attribution(message, last_input, user_name);
        assert!(!stripped);
        assert_eq!(cleaned, message);
    }

    #[test]
    fn ignores_messages_without_attribution() {
        let user_name = "User";
        let message = "We should consider alternative hypotheses.";
        let (cleaned, stripped) = sanitize_monologue_user_attribution(message, "", user_name);
        assert!(!stripped);
        assert_eq!(cleaned, message);
    }
}
