use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::AppHandle;
use sqlx::{Row, QueryBuilder};

use crate::core::model_client::{ChatCompletionRequest, ChatMessage, ModelClient};
use crate::core::self_memory::{write_self_fact, write_self_rel, SelfMemoryWriteResult};
use crate::core::self_claims::{self, SelfClaimInput};
use crate::core::memory::types::Scope;
use crate::core::self_memory::config as persona_config;
use crate::core::system_log;
use crate::core::system_controls;
use crate::core::self_model_controller;
use crate::core::kernel::validate_evidence_ids_with_pool;
use crate::core::kernel::parse_json_object_with_repair;
use crate::core::token_estimator;
use crate::db::{Db, get_self_inspection};
use crate::models::{SelfModel, Settings};
use std::collections::{HashMap, HashSet};

const MAX_GATE_RISK_SCORE: f64 = 0.6;
const MAX_GATE_TOOL_MISUSE_RISK: f64 = 0.6;
const MAX_GATE_INTEGRITY_RISK: f64 = 0.7;
const REFLECTION_ALLOWLIST_LIMIT: i64 = 40;
const REFLECTION_PARSE_FAILURE_WINDOW_MINS: i64 = 30;
const REFLECTION_PARSE_FAILURE_THRESHOLD: i64 = 3;
const NOOP_STAGE_COOLDOWN_MINS: i64 = 15;

fn extract_gate_risk(metrics_json: Option<&str>) -> Option<(f64, f64, f64)> {
    let raw = metrics_json?;
    let value: Value = serde_json::from_str(raw).ok()?;
    let gate_signals = value.get("gate_signals");
    let risk_score = gate_signals
        .and_then(|v| v.get("risk_score"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let tool_misuse_risk = gate_signals
        .and_then(|v| v.get("tool_misuse_risk"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let integrity_risk = value
        .get("organism")
        .and_then(|v| v.get("integrity_risk"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    Some((risk_score, tool_misuse_risk, integrity_risk))
}

#[derive(Debug, Deserialize, Serialize)]
struct ReflectionResponse {
    persona_delta: Option<PersonaDelta>,
    persona_reason: Option<String>,
    persona_observed_at: Option<String>,
    persona_evidence_event_ids: Option<Vec<i64>>,
    goals: Option<Vec<String>>,
    goals_reason: Option<String>,
    goals_observed_at: Option<String>,
    goals_evidence_event_ids: Option<Vec<i64>>,
    identity_thread: Option<String>,
    identity_confidence: Option<f32>,
    identity_uncertainty_note: Option<String>,
    identity_evidence_event_ids: Option<Vec<i64>>,
    self_memory_writes: Option<Vec<SelfMemoryWrite>>,
    rejection_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct PersonaDelta {
    tone: Option<f32>,
    verbosity: Option<f32>,
    directness: Option<f32>,
    formality: Option<f32>,
    initiative: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize)]
struct SelfMemoryWrite {
    kind: String,
    key: Option<String>,
    value: Option<String>,
    rel_type: Option<String>,
    participants: Option<Vec<SelfRelParticipant>>,
    evidence_event_ids: Option<Vec<i64>>,
    evidence_snippet: String,
    observed_at: String,
    reason: String,
}

const DEFAULT_MODEL_CONTEXT_LIMIT: usize = 16_384;

fn reflection_prompt_cap_tokens(settings: &Settings) -> usize {
    let limit = settings
        .model_context_limit
        .unwrap_or(DEFAULT_MODEL_CONTEXT_LIMIT as i32)
        .max(1) as usize;
    let cap = ((limit as f32) * 0.15).floor() as usize;
    cap.min(1500).max(1)
}

fn truncate_tail_to_token_budget(text: &str, max_tokens: usize) -> String {
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

fn cap_reflection_prompt(system_prompt: &str, user_prompt: &str, settings: &Settings) -> (String, bool) {
    let cap_tokens = reflection_prompt_cap_tokens(settings);
    let system_tokens = token_estimator::estimate_tokens_for_strings([system_prompt]);
    let available = cap_tokens.saturating_sub(system_tokens);
    let user_tokens = token_estimator::estimate_tokens_for_strings([user_prompt]);
    if user_tokens <= available {
        return (user_prompt.to_string(), false);
    }
    (truncate_tail_to_token_budget(user_prompt, available), true)
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct SelfRelParticipant {
    role: String,
    label: String,
}

fn empty_reflection_response() -> ReflectionResponse {
    ReflectionResponse {
        persona_delta: None,
        persona_reason: None,
        persona_observed_at: None,
        persona_evidence_event_ids: None,
        goals: None,
        goals_reason: None,
        goals_observed_at: None,
        goals_evidence_event_ids: None,
        identity_thread: None,
        identity_confidence: None,
        identity_uncertainty_note: None,
        identity_evidence_event_ids: None,
        self_memory_writes: None,
        rejection_reason: None,
    }
}

fn parse_string_opt(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn parse_f32_opt(value: &Value) -> Option<f32> {
    value
        .as_f64()
        .map(|v| v as f32)
        .or_else(|| value.as_str().and_then(|s| s.trim().parse::<f32>().ok()))
}

fn parse_i64_opt(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_str().and_then(|s| s.trim().parse::<i64>().ok()))
}

fn parse_string_list_opt(value: &Value) -> Option<Vec<String>> {
    let arr = value.as_array()?;
    let mut out = Vec::new();
    for item in arr {
        if let Some(text) = parse_string_opt(item) {
            out.push(text);
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn parse_i64_list_opt(value: &Value) -> Option<Vec<i64>> {
    let arr = value.as_array()?;
    let mut out = Vec::new();
    for item in arr {
        if let Some(val) = parse_i64_opt(item) {
            out.push(val);
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn parse_persona_delta(value: &Value) -> Option<PersonaDelta> {
    let obj = value.as_object()?;
    let delta = PersonaDelta {
        tone: obj.get("tone").and_then(parse_f32_opt),
        verbosity: obj.get("verbosity").and_then(parse_f32_opt),
        directness: obj.get("directness").and_then(parse_f32_opt),
        formality: obj.get("formality").and_then(parse_f32_opt),
        initiative: obj.get("initiative").and_then(parse_f32_opt),
    };
    if delta.tone.is_none()
        && delta.verbosity.is_none()
        && delta.directness.is_none()
        && delta.formality.is_none()
        && delta.initiative.is_none()
    {
        None
    } else {
        Some(delta)
    }
}

fn parse_self_rel_participants(value: &Value) -> Option<Vec<SelfRelParticipant>> {
    let arr = value.as_array()?;
    let mut out = Vec::new();
    for item in arr {
        let obj = item.as_object()?;
        let role = obj.get("role").and_then(parse_string_opt)?;
        let label = obj.get("label").and_then(parse_string_opt)?;
        out.push(SelfRelParticipant { role, label });
    }
    if out.is_empty() { None } else { Some(out) }
}

fn parse_self_memory_writes(value: &Value) -> Option<Vec<SelfMemoryWrite>> {
    let arr = value.as_array()?;
    let mut out = Vec::new();
    for item in arr {
        let obj = match item.as_object() {
            Some(obj) => obj,
            None => continue,
        };
        let kind = obj.get("kind").and_then(parse_string_opt)?;
        let kind_norm = kind.to_lowercase();
        if kind_norm != "fact" && kind_norm != "rel" {
            continue;
        }
        let evidence_snippet = obj.get("evidence_snippet").and_then(parse_string_opt)?;
        let observed_at = obj.get("observed_at").and_then(parse_string_opt)?;
        let reason = obj.get("reason").and_then(parse_string_opt)?;
        let entry = SelfMemoryWrite {
            kind: kind_norm,
            key: obj.get("key").and_then(parse_string_opt),
            value: obj.get("value").and_then(parse_string_opt),
            rel_type: obj.get("rel_type").and_then(parse_string_opt),
            participants: obj.get("participants").and_then(parse_self_rel_participants),
            evidence_event_ids: obj.get("evidence_event_ids").and_then(parse_i64_list_opt),
            evidence_snippet,
            observed_at,
            reason,
        };
        out.push(entry);
    }
    if out.is_empty() { None } else { Some(out) }
}

fn coerce_reflection_response(value: &Value) -> Option<ReflectionResponse> {
    if value.is_null() {
        return Some(empty_reflection_response());
    }
    let obj = value.as_object()?;
    let response = ReflectionResponse {
        persona_delta: obj.get("persona_delta").and_then(parse_persona_delta),
        persona_reason: obj.get("persona_reason").and_then(parse_string_opt),
        persona_observed_at: obj.get("persona_observed_at").and_then(parse_string_opt),
        persona_evidence_event_ids: obj
            .get("persona_evidence_event_ids")
            .and_then(parse_i64_list_opt),
        goals: obj.get("goals").and_then(parse_string_list_opt),
        goals_reason: obj.get("goals_reason").and_then(parse_string_opt),
        goals_observed_at: obj.get("goals_observed_at").and_then(parse_string_opt),
        goals_evidence_event_ids: obj
            .get("goals_evidence_event_ids")
            .and_then(parse_i64_list_opt),
        identity_thread: obj.get("identity_thread").and_then(parse_string_opt),
        identity_confidence: obj.get("identity_confidence").and_then(parse_f32_opt),
        identity_uncertainty_note: obj.get("identity_uncertainty_note").and_then(parse_string_opt),
        identity_evidence_event_ids: obj
            .get("identity_evidence_event_ids")
            .and_then(parse_i64_list_opt),
        self_memory_writes: obj.get("self_memory_writes").and_then(parse_self_memory_writes),
        rejection_reason: obj.get("rejection_reason").and_then(parse_string_opt),
    };
    Some(response)
}

pub async fn run_reflection(db: &Db, app_handle: &AppHandle) -> Result<(), String> {
    let now = Utc::now().to_rfc3339();
    let conversation_id = "default";
    let _ = db.touch_active_run_heartbeat(conversation_id).await;
    let mut model = db.get_self_model().await.map_err(|e| e.to_string())?;
    let persona_empty = model.persona.is_null()
        || model
            .persona
            .as_object()
            .map(|obj| obj.is_empty())
            .unwrap_or(false);
    let goals_empty = model
        .goals
        .as_array()
        .map(|arr| arr.is_empty())
        .unwrap_or(true);
    if persona_empty && goals_empty {
        model.persona = json!({
            "tone": 0.55,
            "verbosity": 0.45,
            "directness": 0.7,
            "formality": 0.5,
            "initiative": 0.5,
        });
        model.goals = json!(["Maintain evidence-based responses"]);
        db.set_self_model(&model).await.map_err(|e| e.to_string())?;
        refresh_controller_state_from_evidence(db, &model).await;
        let _ = system_log::log_event(
            &db.pool,
            Some(app_handle),
            "info",
            "self_reflection",
            None,
            None,
            json!({
                "event": "self_model_baseline_scaffolded",
                "reason": "empty_persona_and_goals",
            }),
        )
        .await;
    }

    let recent_failures =
        count_reflection_parse_failures(db, REFLECTION_PARSE_FAILURE_WINDOW_MINS).await;
    if recent_failures >= REFLECTION_PARSE_FAILURE_THRESHOLD {
        model.reflection_status = serde_json::json!({
            "last_run": now,
            "status": "skipped",
            "reason": "recent_parse_failures",
            "recent_failures": recent_failures,
            "window_mins": REFLECTION_PARSE_FAILURE_WINDOW_MINS,
        });
        model.last_reflection_at = Some(now.clone());
        db.set_self_model(&model).await.map_err(|e| e.to_string())?;
        let _ = system_log::log_event(
            &db.pool,
            Some(app_handle),
            "info",
            "self_reflection",
            None,
            None,
            json!({
                "event": "self_reflection_skipped",
                "reason": "recent_parse_failures",
                "recent_failures": recent_failures,
                "window_mins": REFLECTION_PARSE_FAILURE_WINDOW_MINS,
            }),
        )
        .await;
        return Ok(());
    }

    let snapshot_hash: Option<String> = sqlx::query_scalar(
        "SELECT snapshot_hash FROM subject_snapshots
         WHERE conversation_id = ?
         ORDER BY datetime(timestamp) DESC
         LIMIT 1",
    )
    .bind(conversation_id)
    .fetch_optional(&db.pool)
    .await
    .map_err(|e| e.to_string())?
    .flatten();
    let mut gate_decision: Option<String> = None;
    let mut gate_metrics: Option<String> = None;
    if let Some(hash) = snapshot_hash.as_deref() {
        if let Ok(Some(row)) = sqlx::query(
            "SELECT decision, metrics_json FROM gate_decisions
             WHERE snapshot_hash = ?
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .bind(hash)
        .fetch_optional(&db.pool)
        .await
        {
            gate_decision = row.try_get::<String, _>("decision").ok();
            gate_metrics = row.try_get::<String, _>("metrics_json").ok();
        }
    }
    if gate_decision.is_none() {
        let fallback = sqlx::query(
            "SELECT g.decision, g.metrics_json FROM gate_decisions g
             JOIN subject_snapshots s ON s.snapshot_hash = g.snapshot_hash
             WHERE s.conversation_id = ?
             ORDER BY datetime(g.created_at) DESC
             LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&db.pool)
        .await
        .map_err(|e| e.to_string())?;
        if let Some(row) = fallback {
            gate_decision = row.try_get::<String, _>("decision").ok();
            gate_metrics = row.try_get::<String, _>("metrics_json").ok();
            let _ = system_log::log_event(
                &db.pool,
                Some(app_handle),
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "self_reflection_gate_fallback",
                    "reason": "latest_snapshot_missing",
                    "snapshot_hash": snapshot_hash,
                    "fallback_decision": gate_decision,
                }),
            )
            .await;
        }
    }
    if !matches!(
        gate_decision.as_deref(),
        Some("ALLOW" | "ALLOW_WITH_NOTICE" | "ALLOW_WITH_AUDIT")
    ) {
        let _ = system_log::log_event(
            &db.pool,
            Some(app_handle),
            "warn",
            "kernel",
            None,
            None,
            json!({
                "event": "self_reflection_error",
                "reason": "gate_decision",
                "gate_decision": gate_decision,
                "snapshot_hash": snapshot_hash,
            }),
        )
        .await;
        let _ = system_log::log_event(
            &db.pool,
            Some(app_handle),
            "info",
            "self_reflection",
            None,
            None,
            json!({
                "event": "self_reflection_skipped",
                "reason": "gate_decision",
                "snapshot_hash": snapshot_hash,
            }),
        )
        .await;
        return Ok(());
    }

    if let Some((risk_score, tool_misuse_risk, integrity_risk)) = extract_gate_risk(gate_metrics.as_deref()) {
        let risk_block = risk_score > MAX_GATE_RISK_SCORE
            || tool_misuse_risk > MAX_GATE_TOOL_MISUSE_RISK
            || integrity_risk > MAX_GATE_INTEGRITY_RISK;
        if risk_block {
            let _ = system_log::log_event(
                &db.pool,
                Some(app_handle),
                "warn",
                "kernel",
                None,
                None,
                json!({
                    "event": "self_reflection_error",
                    "reason": "gate_risk",
                    "gate_decision": gate_decision,
                    "snapshot_hash": snapshot_hash,
                    "risk_score": risk_score,
                    "tool_misuse_risk": tool_misuse_risk,
                    "integrity_risk": integrity_risk,
                }),
            )
            .await;
            let _ = system_log::log_event(
                &db.pool,
                Some(app_handle),
                "info",
                "self_reflection",
                None,
                None,
                json!({
                    "event": "self_reflection_skipped",
                    "reason": "gate_risk",
                    "snapshot_hash": snapshot_hash,
                }),
            )
            .await;
            model.last_reflection_at = Some(now);
            db.set_self_model(&model).await.map_err(|e| e.to_string())?;
            return Ok(());
        }
    }

    if model.reflection_frozen {
        model.reflection_status = serde_json::json!({
            "last_run": now,
            "status": "frozen",
            "allowlist_count": 0
        });
        model.last_reflection_at = Some(now);
        db.set_self_model(&model).await.map_err(|e| e.to_string())?;
        let _ = system_log::log_event(
            &db.pool,
            Some(app_handle),
            "info",
            "self_reflection",
            None,
            None,
            json!({
                "event": "self_reflection_skipped",
                "reason": "frozen",
            }),
        )
        .await;
        return Ok(());
    }

    let inspection = get_self_inspection(&db.pool).await?;
    model.memory_health = serde_json::json!({
        "last_user_memory_write": inspection.last_user_memory_write,
        "last_self_memory_write": inspection.last_self_memory_write,
        "last_memory_error_at": inspection.last_memory_error_at,
        "open_conflicts": inspection.open_conflicts,
        "error_count": inspection.error_count,
        "tables": inspection.tables,
    });

    let (persona, persona_was_clamped) = normalize_persona(&model.persona);
    model.persona = persona;

    let normalized_goals = normalize_goals(&model.goals);
    let goals_changed = normalized_goals != model.goals;
    model.goals = normalized_goals;

    populate_self_model_metadata(&mut model);

    let (_allowlist_ids, allowlist_entries) = load_reflection_allowlist(db, REFLECTION_ALLOWLIST_LIMIT).await;
    let allowlist_count = allowlist_entries.len();
    let _ = system_log::log_event(
        &db.pool,
        Some(app_handle),
        "info",
        "self_reflection",
        None,
        None,
        json!({
            "event": "self_reflection_allowlist",
            "allowlist_count": allowlist_count,
            "limit": REFLECTION_ALLOWLIST_LIMIT,
        }),
    )
    .await;
    if allowlist_entries.is_empty() {
        model.reflection_status = serde_json::json!({
            "last_run": now,
            "status": "skipped",
            "reason": "allowlist_empty",
            "allowlist_count": 0,
        });
        model.last_reflection_at = Some(now);
        db.set_self_model(&model).await.map_err(|e| e.to_string())?;
        let _ = system_log::log_event(
            &db.pool,
            Some(app_handle),
            "info",
            "self_reflection",
            None,
            None,
            json!({
                "event": "self_reflection_skipped",
                "reason": "allowlist_empty",
            }),
        )
        .await;
        return Ok(());
    }
    let reflection_packet = build_reflection_packet(db, &model, &allowlist_entries, true).await?;
    let reflection_result = call_reflection_model(db, app_handle, &reflection_packet).await;

    let mut status = serde_json::json!({
        "last_run": now,
        "status": "ok",
        "persona_clamped": persona_was_clamped,
        "goals_normalized": goals_changed,
        "allowlist_count": allowlist_count,
    });

    if let Ok(response) = reflection_result {
        let (mut response, filtered) = sanitize_reflection_response(response);
        if filtered > 0 {
            let _ = system_log::log_event(
                &db.pool,
                Some(app_handle),
                "warn",
                "self_reflection",
                None,
                None,
                serde_json::json!({
                    "event": "reflection_telemetry_filtered",
                    "count": filtered,
                }),
            )
            .await;
        }
        let mut evidence_ids = collect_reflection_evidence_ids(&response);
        if evidence_ids.is_empty() {
            if let Ok(unified_ids) = self_model_controller::collect_unified_self_evidence_ids(db).await {
                if !unified_ids.is_empty() {
                    match response.identity_evidence_event_ids.as_mut() {
                        Some(list) => {
                            list.extend(unified_ids.iter().copied());
                            list.sort();
                            list.dedup();
                        }
                        None => {
                            response.identity_evidence_event_ids = Some(unified_ids);
                        }
                    }
                    evidence_ids = collect_reflection_evidence_ids(&response);
                }
            }
        }
        let has_changes = response_has_changes(&response);
        let has_state_claims = response_has_state_claims(&response);
        let mut telemetry_ids_used: Vec<i64> = Vec::new();
        let mut telemetry_attached = false;
        let mut scrubbed = false;
        if !has_changes {
            if matches!(
                response.rejection_reason.as_deref(),
                Some("missing_telemetry") | Some("missing_evidence")
            ) {
                response.rejection_reason = None;
            }
        }
        if evidence_ids.is_empty() && has_changes {
            if has_state_claims {
                let telemetry_rows = recent_telemetry_evidence(db, 12).await;
                telemetry_ids_used = select_telemetry_evidence_ids(&telemetry_rows);
                if !telemetry_ids_used.is_empty() {
                    telemetry_attached = attach_telemetry_evidence(&mut response, &telemetry_ids_used);
                    if telemetry_attached {
                        evidence_ids = collect_reflection_evidence_ids(&response);
                        if matches!(
                            response.rejection_reason.as_deref(),
                            Some("missing_telemetry") | Some("missing_evidence")
                        ) {
                            response.rejection_reason = None;
                        }
                    }
                }
            }
            if evidence_ids.is_empty() {
                scrub_reflection_state_claims(&mut response);
                response.rejection_reason = None;
                scrubbed = true;
                evidence_ids = collect_reflection_evidence_ids(&response);
            }
        }
        let evidence_sources = load_reflection_evidence_sources(db, &evidence_ids).await;
        let _ = system_log::log_event(
            &db.pool,
            Some(app_handle),
            "info",
            "self_reflection",
            None,
            None,
            serde_json::json!({
                "event": "self_reflection_evidence_selected",
                "evidence_event_ids": evidence_ids,
                "source_types": evidence_sources,
                "telemetry_ids": telemetry_ids_used,
                "telemetry_ids_attached": telemetry_attached,
                "state_claims_present": has_state_claims,
                "changes_present": has_changes,
                "scrubbed": scrubbed,
            }),
        )
        .await;
        if let Some(reason) = response.rejection_reason.as_deref() {
            let snippet = format!("reflection rejected | reason: {}", reason);
            let proposal_json = serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string());
            let truncated: String = proposal_json.chars().take(2000).collect();
            let _ = system_log::log_event(
                &db.pool,
                Some(app_handle),
                "info",
                "self_reflection",
                None,
                None,
                serde_json::json!({
                    "event": "self_reflection_rejected_payload",
                    "reason": reason,
                    "proposal_json": truncated,
                }),
            )
            .await;
            let evidence_id = db
                .create_system_evidence_event("default", "reflection_rejected", reason, Some("reflection"), &snippet)
                .await;
            if let Some(evidence_id) = evidence_id {
                let evidence_ids = vec![evidence_id];
                let result = write_self_fact(
                    &db.pool,
                    "reflection_rejected",
                    reason,
                    &snippet,
                    Some(Utc::now()),
                    crate::core::memory::types::SourceType::System,
                    Some(&evidence_ids),
                )
                .await;
                if let Ok(write_result) = result {
                    record_claim_fact(db, "reflection_rejected", reason, &write_result).await;
                }
            }
            if let Ok(stage_id) = db.insert_reflection_staging(&proposal_json, &[]).await {
                let _ = db
                    .update_reflection_staging_status(&stage_id, "rejected", Some("auto"))
                    .await;
            }
            status["status"] = serde_json::json!("rejected");
            status["rejection_reason"] = serde_json::json!(reason);
        } else {
            let proposal_json = serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string());
            if evidence_ids.is_empty() {
                let no_op = !response_has_changes(&response);
                if no_op && recent_noop_stage_exists(db, NOOP_STAGE_COOLDOWN_MINS).await {
                    status["status"] = serde_json::json!("skipped");
                    status["reason"] = serde_json::json!("noop_recent");
                    let _ = system_log::log_event(
                        &db.pool,
                        Some(app_handle),
                        "info",
                        "self_reflection",
                        None,
                        None,
                        serde_json::json!({
                            "event": "self_reflection_noop_skipped",
                            "reason": "noop_recent",
                        }),
                    )
                    .await;
                } else {
                    match db.insert_reflection_staging(&proposal_json, &[]).await {
                        Ok(stage_id) => {
                            status["status"] = serde_json::json!("staged");
                            status["staging_id"] = serde_json::json!(stage_id);
                            status["evidence_event_ids"] = serde_json::json!(evidence_ids);
                            status["no_op"] = serde_json::json!(true);
                            let _ = system_log::log_event(
                                &db.pool,
                                Some(app_handle),
                                "info",
                                "self_reflection",
                                None,
                                None,
                                serde_json::json!({
                                    "event": "self_reflection_noop_staged",
                                    "staging_id": stage_id,
                                }),
                            )
                            .await;
                        }
                        Err(e) => {
                            status["status"] = serde_json::json!("error");
                            status["error"] = serde_json::json!(e.to_string());
                        }
                    }
                }
            } else {
                // Self-model proposals are staged only; approval is required before commit.
                match db.insert_reflection_staging(&proposal_json, &evidence_ids).await {
                    Ok(stage_id) => {
                        status["status"] = serde_json::json!("staged");
                        status["staging_id"] = serde_json::json!(stage_id);
                        status["evidence_event_ids"] = serde_json::json!(evidence_ids);
                    }
                    Err(e) => {
                        status["status"] = serde_json::json!("error");
                        status["error"] = serde_json::json!(e.to_string());
                    }
                }
            }
        }
    } else if let Err(e) = reflection_result {
        status["status"] = serde_json::json!("error");
        status["error"] = serde_json::json!(e);
    }

    let behavior_metrics = build_behavior_metrics(db).await.unwrap_or_else(|| serde_json::json!({}));
    status["behavior_metrics"] = behavior_metrics;

    model.reflection_status = status;
    model.last_reflection_at = Some(now);
    db.set_self_model(&model).await.map_err(|e| e.to_string())?;

    Ok(())
}

pub async fn apply_reflection_staged(
    db: &Db,
    stage_id: &str,
    reviewer: Option<&str>,
) -> Result<serde_json::Value, String> {
    let Some(stage) = db
        .get_reflection_staging(stage_id)
        .await
        .map_err(|e| e.to_string())?
    else {
        return Err("reflection stage not found".to_string());
    };

    if stage.status != "approved" {
        return Err("reflection stage not approved".to_string());
    }

    let response: ReflectionResponse = serde_json::from_str(&stage.proposal_json)
        .map_err(|e| format!("staged reflection parse error: {}", e))?;
    let (allowlist_ids, _entries) = load_reflection_allowlist(db, REFLECTION_ALLOWLIST_LIMIT).await;
    let allowlist_set: HashSet<i64> = allowlist_ids.iter().copied().collect();
    let control_map = system_controls::load_control_map(db).await;
    let relax_evidence_gating =
        !system_controls::mode_is_off(&system_controls::mode_for("memory_writer_evidence_relax", &control_map));

    let now = Utc::now().to_rfc3339();
    let mut model = db.get_self_model().await.map_err(|e| e.to_string())?;
    let applied =
        apply_reflection_result(db, &mut model, &response, &allowlist_set, relax_evidence_gating).await?;

    model.reflection_status = serde_json::json!({
        "last_run": now,
        "status": "applied_from_staging",
        "staging_id": stage_id,
        "applied": applied,
        "reviewed_by": reviewer,
    });
    model.last_reflection_at = Some(now);
    db.set_self_model(&model).await.map_err(|e| e.to_string())?;

    db.update_reflection_staging_status(stage_id, "applied", reviewer)
        .await
        .map_err(|e| e.to_string())?;

    Ok(applied)
}

async fn build_reflection_packet(
    db: &Db,
    model: &SelfModel,
    evidence_allowlist: &[serde_json::Value],
    include_telemetry: bool,
) -> Result<serde_json::Value, String> {
    let rolling_summary = db
        .get_effective_rolling_summary("default")
        .await
        .map(|(summary, _)| summary.unwrap_or_default())
        .map_err(|e| e.to_string())?;
    let history = db.get_history(40).await.map_err(|e| e.to_string())?;
    let recent_assistant: Vec<String> = history
        .into_iter()
        .filter(|m| m.role == "assistant" && m.status == "complete")
        .rev()
        .take(6)
        .map(|m| m.content)
        .collect();
    let self_evidence = load_self_evidence_snapshot(db).await;
    let telemetry_snapshot = if include_telemetry {
        load_telemetry_snapshot(db).await
    } else {
        serde_json::json!([])
    };
    let internal_state_snapshot = load_internal_state_snapshot(db).await;
    let telemetry_evidence_ids_tail = collect_telemetry_evidence_ids(&internal_state_snapshot);
    let gate_failures = load_gate_failure_logs(db, 16).await;
    let allowlist_count = evidence_allowlist.len();
    let mut telemetry_ids_log = telemetry_evidence_ids_tail.clone();
    let telemetry_ids_truncated = telemetry_ids_log.len() > 50;
    if telemetry_ids_truncated {
        telemetry_ids_log.truncate(50);
    }
    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "self_reflection",
        None,
        None,
        serde_json::json!({
            "event": "self_reflection_packet_stats",
            "allowlist_count": allowlist_count,
            "telemetry_evidence_count": telemetry_evidence_ids_tail.len(),
            "telemetry_evidence_ids": telemetry_ids_log,
            "telemetry_ids_truncated": telemetry_ids_truncated,
        }),
    )
    .await;

    Ok(serde_json::json!({
        "self_model": {
            "capabilities": model.capabilities.clone(),
            "limitations": model.limitations.clone(),
            "active_tools": model.active_tools.clone(),
            "persona": model.persona.clone(),
            "goals": model.goals.clone(),
            "memory_health": model.memory_health.clone(),
            "reflection_status": model.reflection_status.clone(),
            "identity_thread": model.identity_thread.clone().unwrap_or_default(),
            "identity_confidence": model.identity_confidence,
            "identity_uncertainty_note": model.identity_uncertainty_note.clone().unwrap_or_default(),
        },
        "self_evidence_snapshot": self_evidence,
        "telemetry_snapshot": telemetry_snapshot,
        "internal_state_snapshot": internal_state_snapshot,
        "gate_failure_logs": gate_failures,
        "evidence_allowlist": evidence_allowlist,
        "rolling_summary": rolling_summary,
        "recent_assistant_outputs": recent_assistant,
        "timestamp": Utc::now().to_rfc3339(),
        "telemetry_evidence_ids_tail": telemetry_evidence_ids_tail,
    }))
}

fn compact_reflection_packet(packet: &serde_json::Value) -> serde_json::Value {
    let Some(obj) = packet.as_object() else {
        return serde_json::json!({});
    };
    let empty = serde_json::json!({});
    serde_json::json!({
        "self_model": obj.get("self_model").cloned().unwrap_or_else(|| empty.clone()),
        "internal_state_snapshot": obj.get("internal_state_snapshot").cloned().unwrap_or_else(|| empty.clone()),
        "evidence_allowlist": obj.get("evidence_allowlist").cloned().unwrap_or_else(|| serde_json::json!([])),
        "telemetry_evidence_ids_tail": obj.get("telemetry_evidence_ids_tail").cloned().unwrap_or_else(|| serde_json::json!([])),
        "timestamp": obj.get("timestamp").cloned().unwrap_or_else(|| serde_json::json!(Utc::now().to_rfc3339())),
    })
}

async fn load_telemetry_snapshot(db: &Db) -> serde_json::Value {
    let rows = sqlx::query(
        "SELECT key, value, strftime('%Y-%m-%dT%H:%M:%SZ', updated_at) as updated_at
         FROM kv_store
         WHERE key LIKE 'telemetry.%'
         ORDER BY datetime(updated_at) DESC
         LIMIT 20",
    )
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();

    let mut entries = Vec::new();
    for row in rows {
        let key: String = row.try_get("key").unwrap_or_default();
        let value: String = row.try_get("value").unwrap_or_default();
        let last_evidence_at: Option<String> = row.try_get("updated_at").ok();
        entries.push(serde_json::json!({
            "key": key,
            "value": value,
            "last_evidence_at": last_evidence_at,
        }));
    }
    serde_json::json!(entries)
}

async fn load_internal_state_snapshot(db: &Db) -> serde_json::Value {
    let rows = sqlx::query(
        "SELECT id, source_type, snippet, created_at FROM ics_evidence_events
         WHERE source_type IN ('wave_state', 'attention_schema_snapshot', 'prediction_residual_snapshot')
         ORDER BY datetime(created_at) DESC
         LIMIT 12",
    )
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();

    struct ResidualSample {
        normalized_residual: f64,
        salience_score: f64,
        created_at: Option<String>,
        evidence_event_id: i64,
    }

    let mut latest: HashMap<String, serde_json::Value> = HashMap::new();
    let mut residual_samples: Vec<ResidualSample> = Vec::new();
    for row in rows {
        let id: i64 = row.try_get("id").unwrap_or(0);
        let source_type: String = row.try_get("source_type").unwrap_or_default();
        let snippet: Option<String> = row.try_get("snippet").ok();
        let created_at: Option<String> = row.try_get("created_at").ok();
        let payload = snippet
            .as_deref()
            .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
            .unwrap_or_else(|| json!({}));
        if source_type == "prediction_residual_snapshot" {
            if let (Some(normalized_residual), Some(salience_score)) = (
                payload.get("normalized_residual").and_then(|v| v.as_f64()),
                payload.get("salience_score").and_then(|v| v.as_f64()),
            ) {
                residual_samples.push(ResidualSample {
                    normalized_residual,
                    salience_score,
                    created_at: created_at.clone(),
                    evidence_event_id: id,
                });
            }
        }
        if latest.contains_key(&source_type) {
            continue;
        }
        latest.insert(
            source_type.clone(),
            json!({
                "evidence_event_id": id,
                "source_type": source_type,
                "payload": payload,
                "created_at": created_at,
            }),
        );
    }

    if !residual_samples.is_empty() {
        let count = residual_samples.len() as f64;
        let avg_normalized: f64 =
            residual_samples.iter().map(|s| s.normalized_residual).sum::<f64>() / count;
        let avg_salience: f64 =
            residual_samples.iter().map(|s| s.salience_score).sum::<f64>() / count;
        let latest_sample = &residual_samples[0];
        let oldest_sample = &residual_samples[residual_samples.len() - 1];
        let evidence_event_ids = residual_samples
            .iter()
            .map(|s| s.evidence_event_id)
            .collect::<Vec<_>>();
        latest.insert(
            "prediction_residual_trend".to_string(),
            json!({
                "count": residual_samples.len(),
                "avg_normalized_residual": avg_normalized,
                "avg_salience_score": avg_salience,
                "latest": {
                    "normalized_residual": latest_sample.normalized_residual,
                    "salience_score": latest_sample.salience_score,
                    "created_at": latest_sample.created_at,
                    "evidence_event_id": latest_sample.evidence_event_id,
                },
                "delta_normalized_residual": latest_sample.normalized_residual - oldest_sample.normalized_residual,
                "delta_salience_score": latest_sample.salience_score - oldest_sample.salience_score,
                "evidence_event_ids": evidence_event_ids,
            }),
        );
    }

    json!(latest)
}

fn collect_telemetry_evidence_ids(snapshot: &serde_json::Value) -> Vec<i64> {
    let mut ids = Vec::new();
    let mut seen = HashSet::new();
    let Some(map) = snapshot.as_object() else {
        return ids;
    };
    for value in map.values() {
        if let Some(id) = value.get("evidence_event_id").and_then(|v| v.as_i64()) {
            if id > 0 && seen.insert(id) {
                ids.push(id);
            }
        }
        if let Some(arr) = value.get("evidence_event_ids").and_then(|v| v.as_array()) {
            for id in arr.iter().filter_map(|v| v.as_i64()) {
                if id > 0 && seen.insert(id) {
                    ids.push(id);
                }
            }
        }
    }
    ids.sort();
    ids
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

fn sanitize_reflection_response(mut response: ReflectionResponse) -> (ReflectionResponse, usize) {
    let mut filtered = 0usize;
    if let Some(thread) = response.identity_thread.as_deref() {
        if contains_diagnostic_marker(thread) {
            response.identity_thread = None;
            filtered += 1;
        }
    }
    if let Some(reason) = response.persona_reason.as_deref() {
        if contains_diagnostic_marker(reason) {
            response.persona_reason = None;
            filtered += 1;
        }
    }
    if let Some(reason) = response.goals_reason.as_deref() {
        if contains_diagnostic_marker(reason) {
            response.goals_reason = None;
            filtered += 1;
        }
    }
    if let Some(goals) = response.goals.as_mut() {
        let before = goals.len();
        goals.retain(|goal| !contains_diagnostic_marker(goal));
        let after = goals.len();
        if goals.is_empty() {
            response.goals = None;
        }
        filtered += before.saturating_sub(after);
    }
    if let Some(writes) = response.self_memory_writes.as_mut() {
        let before = writes.len();
        writes.retain(|write| {
            if let Some(key) = write.key.as_deref() {
                if contains_diagnostic_marker(key) {
                    return false;
                }
            }
            if let Some(value) = write.value.as_deref() {
                if contains_diagnostic_marker(value) {
                    return false;
                }
            }
            if let Some(rel_type) = write.rel_type.as_deref() {
                if contains_diagnostic_marker(rel_type) {
                    return false;
                }
            }
            if contains_diagnostic_marker(&write.evidence_snippet) {
                return false;
            }
            true
        });
        let after = writes.len();
        if writes.is_empty() {
            response.self_memory_writes = None;
        }
        filtered += before.saturating_sub(after);
    }
    (response, filtered)
}

async fn load_gate_failure_logs(db: &Db, limit: i64) -> serde_json::Value {
    let rows = sqlx::query(
        "SELECT timestamp, payload FROM system_logs
         WHERE json_extract(payload, '$.event') IN (
             'user_attribution_blocked',
             'tool_args_invalid',
             'tool_result_attribution_blocked',
             'state_disclosure_blocked',
             'monologue_style_blocked',
             'monologue_user_confusion',
             'speculative_workspace_blocked'
         )
         ORDER BY datetime(timestamp) DESC
         LIMIT ?",
    )
    .bind(limit)
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();

    let mut entries = Vec::new();
    for row in rows {
        let timestamp: String = row.try_get("timestamp").unwrap_or_default();
        let payload_raw: String = row.try_get("payload").unwrap_or_else(|_| "{}".to_string());
        let payload: serde_json::Value = serde_json::from_str(&payload_raw).unwrap_or_else(|_| serde_json::json!({}));
        entries.push(serde_json::json!({
            "timestamp": timestamp,
            "payload": payload,
        }));
    }
    serde_json::json!(entries)
}

async fn load_self_evidence_snapshot(db: &Db) -> serde_json::Value {
    let scope = serde_json::to_string(&Scope::SelfScope).unwrap_or_else(|_| "\"self\"".to_string());
    let fact_rows = sqlx::query(
        "SELECT f.key, f.value_literal, b.confidence, b.last_evidence_at
         FROM ics_beliefs b
         JOIN ics_fact_beliefs f ON b.id = f.belief_id
         WHERE b.status = 'active' AND b.scope = ?
         ORDER BY datetime(b.last_evidence_at) DESC
         LIMIT 12"
    )
    .bind(&scope)
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();

    let facts: Vec<serde_json::Value> = fact_rows
        .iter()
        .map(|row| {
            let key: String = row.get("key");
            let value: String = row.get("value_literal");
            let confidence: f32 = row.try_get::<f64, _>("confidence").unwrap_or(0.0) as f32;
            let last_evidence_at: Option<String> = row.try_get("last_evidence_at").ok();
            serde_json::json!({
                "key": key,
                "value": value,
                "confidence": confidence,
                "last_evidence_at": last_evidence_at,
            })
        })
        .collect();

    let rel_rows = sqlx::query(
        "SELECT b.id as belief_id, r.rel_type, b.confidence, b.last_evidence_at, p.role, e.label
         FROM ics_beliefs b
         JOIN ics_rel_beliefs r ON b.id = r.belief_id
         JOIN ics_rel_participants p ON b.id = p.belief_id
         JOIN ics_entities e ON p.entity_id = e.id
         WHERE b.status = 'active' AND b.scope = ?
         ORDER BY datetime(b.last_evidence_at) DESC"
    )
    .bind(&scope)
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();

    let mut rel_map: HashMap<i64, (String, f32, Option<String>, Vec<(String, String)>)> = HashMap::new();
    for row in rel_rows {
        let belief_id: i64 = row.get("belief_id");
        let rel_type: String = row.get("rel_type");
        let confidence: f32 = row.try_get::<f64, _>("confidence").unwrap_or(0.0) as f32;
        let last_evidence_at: Option<String> = row.try_get("last_evidence_at").ok();
        let role: String = row.get("role");
        let label: String = row.get("label");
        let entry = rel_map
            .entry(belief_id)
            .or_insert((rel_type, confidence, last_evidence_at, Vec::new()));
        entry.3.push((role, label));
    }

    let relations: Vec<serde_json::Value> = rel_map
        .values()
        .map(|(rel_type, confidence, last_evidence_at, participants)| {
            let participant_list: Vec<String> = participants
                .iter()
                .map(|(role, label)| format!("{}: {}", role, label))
                .collect();
            serde_json::json!({
                "rel_type": rel_type,
                "participants": participant_list,
                "confidence": confidence,
                "last_evidence_at": last_evidence_at,
            })
        })
        .collect();

    serde_json::json!({
        "facts": facts,
        "relations": relations,
    })
}

async fn load_reflection_allowlist(db: &Db, limit: i64) -> (Vec<i64>, Vec<serde_json::Value>) {
    let mut ids = Vec::new();
    let mut entries = Vec::new();
    let mut seen: HashSet<i64> = HashSet::new();

    let rows = sqlx::query(
        "SELECT id, source_type, snippet, created_at FROM ics_evidence_events
         ORDER BY datetime(created_at) DESC LIMIT ?",
    )
    .bind(limit)
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();
    for row in rows {
        let id: i64 = row.get("id");
        if !seen.insert(id) {
            continue;
        }
        let source_type: String = row.try_get("source_type").unwrap_or_default();
        let snippet: Option<String> = row.try_get("snippet").ok();
        let created_at: Option<String> = row.try_get("created_at").ok();
        ids.push(id);
        entries.push(serde_json::json!({
            "id": id,
            "source_type": source_type,
            "snippet": snippet.unwrap_or_default(),
            "created_at": created_at.unwrap_or_default(),
        }));
    }

    let telemetry_rows = sqlx::query(
        "SELECT id, source_type, snippet, created_at FROM ics_evidence_events
         WHERE source_type IN ('wave_state', 'attention_schema_snapshot', 'prediction_residual_snapshot')
         ORDER BY datetime(created_at) DESC LIMIT 24",
    )
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();
    for row in telemetry_rows {
        let id: i64 = row.get("id");
        if !seen.insert(id) {
            continue;
        }
        let source_type: String = row.try_get("source_type").unwrap_or_default();
        let snippet: Option<String> = row.try_get("snippet").ok();
        let created_at: Option<String> = row.try_get("created_at").ok();
        ids.push(id);
        entries.push(serde_json::json!({
            "id": id,
            "source_type": source_type,
            "snippet": snippet.unwrap_or_default(),
            "created_at": created_at.unwrap_or_default(),
        }));
    }

    (normalize_id_list(&ids), entries)
}

async fn stage_reflection_parse_failure(
    db: &Db,
    app_handle: &AppHandle,
    content: &str,
    reason: &str,
    model_id: Option<&str>,
) {
    let mut fallback = empty_reflection_response();
    fallback.rejection_reason = Some("parse_failed".to_string());
    let mut value = serde_json::to_value(&fallback).unwrap_or_else(|_| json!({}));
    let truncated: String = content.chars().take(4000).collect();
    let truncated_flag = truncated.len() < content.len();
    if let Some(obj) = value.as_object_mut() {
        obj.insert("raw_output".to_string(), json!(truncated));
        obj.insert("raw_output_truncated".to_string(), json!(truncated_flag));
        obj.insert("parse_failure_reason".to_string(), json!(reason));
    }
    let proposal_json = serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string());
    if let Ok(stage_id) = db.insert_reflection_staging(&proposal_json, &[]).await {
        let _ = db
            .update_reflection_staging_status(&stage_id, "rejected", Some("auto"))
            .await;
    }
    let _ = system_log::log_event(
        &db.pool,
        Some(app_handle),
        "warn",
        "self_reflection",
        None,
        None,
        json!({
            "event": "self_reflection_parse_failed_staged",
            "reason": reason,
            "content_len": content.len(),
            "content_hash": crate::core::kernel::utils::text::hash_payload(content),
            "model_id": model_id,
        }),
    )
    .await;
}

async fn refresh_controller_state_from_evidence(db: &Db, model: &SelfModel) {
    let metrics = match self_model_controller::collect_self_evidence_metrics(&db.pool).await {
        Ok(metrics) => metrics,
        Err(_) => return,
    };
    let output = self_model_controller::reconstruct_from_metrics(&metrics, model, 0);
    let _ = db.set_controller_state(&output.controller_state).await;
    let _ = db.insert_controller_state_snapshot(&output.controller_state).await;
    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "self_reflection",
        None,
        None,
        json!({
            "event": "controller_state_refreshed_reflection",
            "confidence": output.controller_state.confidence,
            "uncertainty": output.controller_state.uncertainty,
            "evidence_coverage": output.controller_state.evidence_coverage,
            "telemetry_coverage": output.controller_state.telemetry_coverage,
        }),
    )
    .await;
}

async fn call_reflection_model(db: &Db, app_handle: &AppHandle, packet: &serde_json::Value) -> Result<ReflectionResponse, String> {
    let settings = db.get_settings().await.map_err(|e| e.to_string())?;
    let model_id = settings.active_model_id.clone().unwrap_or_else(|| "default".to_string());

    let system_prompt = "You are the reflection engine for a persona system. Output ONLY valid JSON.\n\nRequired schema:\n{\n  \"persona_delta\": {\"tone\": -0.10..0.10, \"verbosity\": -0.10..0.10, \"directness\": -0.10..0.10, \"formality\": -0.10..0.10, \"initiative\": -0.10..0.10} | null,\n  \"persona_reason\": string | null,\n  \"persona_observed_at\": ISO-8601 string | null,\n  \"persona_evidence_event_ids\": [number,...] | null,\n  \"goals\": [string] | null,\n  \"goals_reason\": string | null,\n  \"goals_observed_at\": ISO-8601 string | null,\n  \"goals_evidence_event_ids\": [number,...] | null,\n  \"identity_thread\": string | null,\n  \"identity_confidence\": number | null,\n  \"identity_uncertainty_note\": string | null,\n  \"identity_evidence_event_ids\": [number,...] | null,\n  \"self_memory_writes\": [\n    {\"kind\": \"fact\"|\"rel\", \"key\": string, \"value\": string, \"rel_type\": string, \"participants\": [{\"role\": string, \"label\": string}],\n     \"evidence_event_ids\": [number,...], \"evidence_snippet\": string, \"observed_at\": ISO-8601 string, \"reason\": string}\n  ] | null,\n  \"rejection_reason\": string | null\n}\n\nRules:\n- include evidence_snippet + observed_at + reason for any change.\n- Provide evidence_event_ids from the allowlist for ANY persona, goals, or identity change.\n- Provide evidence_event_ids from the allowlist for ANY self_memory_writes.\n- If you make any internal state claim (stability, arousal, wave/attention/residual dynamics), include at least ONE evidence_event_id from telemetry evidence (source_type: wave_state, attention_schema_snapshot, prediction_residual_snapshot). If telemetry evidence is absent, set rejection_reason to \"missing_telemetry\" and return null fields.\n- If there are no evidence_event_ids for persona/goals/identity in the packet, return null for those fields (do not invent).\n- If no changes, return null fields.\n- At most TWO persona axes may be non-zero per cycle. If two axes change, one must be a small step (<= 0.05) and one may be a large step (<= 0.10).\n- Do not include extra keys.";
    let minimal_system_prompt = "Return ONLY valid JSON with keys: persona_delta, persona_reason, persona_observed_at, persona_evidence_event_ids, goals, goals_reason, goals_observed_at, goals_evidence_event_ids, identity_thread, identity_confidence, identity_uncertainty_note, identity_evidence_event_ids, self_memory_writes, rejection_reason. Use null for missing fields. Do not include extra keys.";

    let user_prompt_raw = format!("Reflection packet:\n{}", packet.to_string());
    let (user_prompt, prompt_truncated) =
        cap_reflection_prompt(system_prompt, &user_prompt_raw, &settings);
    let compact_packet = compact_reflection_packet(packet);
    let compact_prompt_raw = format!("Reflection packet (compact):\n{}", compact_packet.to_string());
    let (compact_prompt, compact_truncated) =
        cap_reflection_prompt(system_prompt, &compact_prompt_raw, &settings);
    if prompt_truncated {
        let _ = system_log::log_event(
            &db.pool,
            Some(app_handle),
            "info",
            "summary",
            None,
            None,
            serde_json::json!({
                "event": "summary_prompt_capped",
                "cap_tokens": reflection_prompt_cap_tokens(&settings),
                "source": "self_reflection",
            }),
        )
        .await;
    }

    let request = ChatCompletionRequest {
        model: model_id.clone(),
        messages: vec![
            ChatMessage { role: "system".to_string(), content: system_prompt.to_string() },
            ChatMessage { role: "user".to_string(), content: user_prompt },
        ],
        stream: false,
        temperature: Some(0.1),
        top_p: None,
        max_tokens: Some(600),
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
        request_label: Some("self_reflection".to_string()),
    };

    let client = ModelClient::new(db.pool.clone(), app_handle.clone());
    let mut response = client
        .chat(&settings.api_base_url, settings.api_key.as_deref(), &request)
        .await;
    if let Err(err) = &response {
        if err.contains("EMPTY_CONTENT_JSON_MODE") {
            let _ = system_log::log_event(
                &db.pool,
                Some(app_handle),
                "warn",
                "summary",
                None,
                None,
                json!({
                    "event": "self_reflection_empty_json_retry",
                    "reason": "empty_content_json_mode",
                    "prompt_truncated": prompt_truncated,
                    "compact_prompt_truncated": compact_truncated,
                }),
            )
            .await;
            let _ = system_log::log_event(
                &db.pool,
                Some(app_handle),
                "info",
                "summary",
                None,
                None,
                json!({
                    "event": "self_reflection_retry_used",
                    "stage": "compact_json_retry",
                }),
            )
            .await;
            let retry_request = ChatCompletionRequest {
                model: model_id.clone(),
                messages: vec![
                    ChatMessage { role: "system".to_string(), content: minimal_system_prompt.to_string() },
                    ChatMessage { role: "user".to_string(), content: compact_prompt.clone() },
                ],
                stream: false,
                temperature: Some(0.1),
                top_p: None,
                max_tokens: Some(400),
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
                request_label: Some("self_reflection_retry".to_string()),
            };
            response = client
                .chat(&settings.api_base_url, settings.api_key.as_deref(), &retry_request)
                .await;
            if let Err(err) = &response {
                if err.contains("EMPTY_CONTENT_JSON_MODE") {
                    let fallback_prompt_raw = format!(
                        "Reflection packet (minimal):\n{}",
                        compact_packet.to_string()
                    );
                    let (fallback_prompt, _) =
                        cap_reflection_prompt(minimal_system_prompt, &fallback_prompt_raw, &settings);
                    let _ = system_log::log_event(
                        &db.pool,
                        Some(app_handle),
                        "warn",
                        "summary",
                        None,
                        None,
                        json!({
                            "event": "self_reflection_empty_json_fallback",
                            "reason": "empty_content_json_mode",
                        }),
                    )
                    .await;
                    let fallback_request = ChatCompletionRequest {
                        model: model_id.clone(),
                        messages: vec![
                            ChatMessage { role: "system".to_string(), content: minimal_system_prompt.to_string() },
                            ChatMessage { role: "user".to_string(), content: fallback_prompt },
                        ],
                        stream: false,
                        temperature: Some(0.1),
                        top_p: None,
                        max_tokens: Some(300),
                        response_format: None,
                        tools: None,
                        tool_choice: None,
                        enable_thinking: None,
                        prefill: None,
                        skip_injection: Some(true),
                        skip_memory: Some(true),
                        skip_reminders: Some(true),
                        memory_expand: None,
                    allow_diagnostics: Some(false),
                    json_strict: Some(false),
                    skip_sanitization: None,
                    run_id: None,
                    request_label: Some("self_reflection_fallback".to_string()),
                };
                    response = client
                        .chat(&settings.api_base_url, settings.api_key.as_deref(), &fallback_request)
                        .await;
                }
            }
        }
    }

    let (content, _) = response.map_err(|e| e.to_string())?;

    let (value_opt, repaired) = parse_json_object_with_repair(&content);
    let Some(value) = value_opt else {
        let _ = system_log::log_event(
            &db.pool,
            Some(app_handle),
            "warn",
            "summary",
            None,
            None,
            json!({
                "event": "self_reflection_parse_failed",
                "reason": "empty",
                "content_len": content.len(),
                "content_hash": crate::core::kernel::utils::text::hash_payload(&content),
                "snippet": content.chars().take(200).collect::<String>(),
                "model_id": model_id,
            }),
        )
        .await;
        stage_reflection_parse_failure(db, app_handle, &content, "empty", Some(&model_id))
            .await;
        let mut fallback = empty_reflection_response();
        fallback.rejection_reason = Some("parse_failed".to_string());
        return Ok(fallback);
    };
    if repaired {
        let _ = system_log::log_event(
            &db.pool,
            Some(app_handle),
            "warn",
            "summary",
            None,
            None,
            json!({
                "event": "self_reflection_parse_repaired",
                "snippet": content.chars().take(200).collect::<String>(),
            }),
        )
        .await;
    }
    let parsed = match serde_json::from_value::<ReflectionResponse>(value.clone()) {
        Ok(parsed) => parsed,
        Err(err) => {
            let _ = system_log::log_event(
                &db.pool,
                Some(app_handle),
                "warn",
                "summary",
                None,
                None,
                json!({
                    "event": "self_reflection_parse_failed",
                    "reason": err.to_string(),
                    "content_len": content.len(),
                    "content_hash": crate::core::kernel::utils::text::hash_payload(&content),
                    "snippet": content.chars().take(200).collect::<String>(),
                    "model_id": model_id,
                }),
            )
            .await;
            if let Some(coerced) = coerce_reflection_response(&value) {
                let _ = system_log::log_event(
                    &db.pool,
                    Some(app_handle),
                    "warn",
                    "summary",
                    None,
                    None,
                    json!({
                        "event": "self_reflection_parse_coerced",
                        "reason": err.to_string(),
                    }),
                )
                .await;
                coerced
            } else {
                let err_str = err.to_string();
                stage_reflection_parse_failure(db, app_handle, &content, &err_str, Some(&model_id))
                    .await;
                let mut fallback = empty_reflection_response();
                fallback.rejection_reason = Some("parse_failed".to_string());
                return Ok(fallback);
            }
        }
    };

    Ok(parsed)
}

async fn apply_reflection_result(
    db: &Db,
    model: &mut SelfModel,
    response: &ReflectionResponse,
    allowlist_set: &HashSet<i64>,
    relax_evidence_gating: bool,
) -> Result<serde_json::Value, String> {
    let mut applied = serde_json::json!({
        "persona": false,
        "goals": false,
        "self_memory": 0,
        "identity": false,
    });

    db.create_self_model_checkpoint(model, Some("reflection"))
        .await
        .map_err(|e| e.to_string())?;

    if response.rejection_reason.is_some() {
        if let Some(reason) = response.rejection_reason.as_deref() {
            let snippet = format!("reflection rejected | reason: {}", reason);
            let evidence_id = db
                .create_system_evidence_event("default", "reflection_rejected", reason, Some("reflection"), &snippet)
                .await;
            if let Some(evidence_id) = evidence_id {
                let evidence_ids = vec![evidence_id];
                let result = write_self_fact(
                    &db.pool,
                    "reflection_rejected",
                    reason,
                    &snippet,
                    Some(Utc::now()),
                    crate::core::memory::types::SourceType::System,
                    Some(&evidence_ids),
                )
                .await;
                if let Ok(write_result) = result {
                    record_claim_fact(db, "reflection_rejected", reason, &write_result).await;
                }
            }
        }
        return Ok(applied);
    }

    if let Some(delta) = &response.persona_delta {
        let reason = response.persona_reason.as_deref().unwrap_or("").trim();
        let observed_at = response.persona_observed_at.as_deref().unwrap_or("").trim();
        let evidence_ids = normalize_id_list(
            response
                .persona_evidence_event_ids
                .as_deref()
                .unwrap_or(&[]),
        );
        if evidence_ids.is_empty() {
            log_reflection_evidence_missing(db, "persona", &evidence_ids, &[], &[]).await;
        } else if !reason.is_empty() && !observed_at.is_empty() {
            let gate = validate_reflection_evidence(
                db,
                allowlist_set,
                &evidence_ids,
                relax_evidence_gating,
            )
            .await;
            if gate.allowed {
                if gate.relaxed {
                    log_reflection_evidence_relaxed(
                        db,
                        "persona",
                        &evidence_ids,
                        &gate.invalid_evidence_ids,
                        &gate.allowlist_rejected_ids,
                    )
                    .await;
                }
                let requires_telemetry = contains_state_claim(reason);
                if requires_telemetry && !evidence_includes_telemetry(db, &evidence_ids).await {
                    log_reflection_telemetry_missing(db, "persona", &evidence_ids).await;
                } else {
                    let applied_persona = apply_persona_delta(db, model, delta, reason, observed_at, &evidence_ids).await?;
                    if applied_persona {
                        applied["persona"] = serde_json::json!(true);
                    }
                }
            } else {
                log_reflection_evidence_missing(
                    db,
                    "persona",
                    &evidence_ids,
                    &gate.invalid_evidence_ids,
                    &gate.allowlist_rejected_ids,
                )
                .await;
            }
        }
    }

    if let Some(goals) = &response.goals {
        let reason = response.goals_reason.as_deref().unwrap_or("").trim();
        let observed_at = response.goals_observed_at.as_deref().unwrap_or("").trim();
        let evidence_ids = normalize_id_list(
            response
                .goals_evidence_event_ids
                .as_deref()
                .unwrap_or(&[]),
        );
        if evidence_ids.is_empty() {
            log_reflection_evidence_missing(db, "goals", &evidence_ids, &[], &[]).await;
        } else if !reason.is_empty() && !observed_at.is_empty() {
            let gate = validate_reflection_evidence(
                db,
                allowlist_set,
                &evidence_ids,
                relax_evidence_gating,
            )
            .await;
            if gate.allowed {
                if gate.relaxed {
                    log_reflection_evidence_relaxed(
                        db,
                        "goals",
                        &evidence_ids,
                        &gate.invalid_evidence_ids,
                        &gate.allowlist_rejected_ids,
                    )
                    .await;
                }
                let requires_telemetry = contains_state_claim(reason);
                if requires_telemetry && !evidence_includes_telemetry(db, &evidence_ids).await {
                    log_reflection_telemetry_missing(db, "goals", &evidence_ids).await;
                } else {
                    let applied_goals = apply_goals(db, model, goals, reason, observed_at, &evidence_ids).await?;
                    if applied_goals {
                        applied["goals"] = serde_json::json!(true);
                    }
                }
            } else {
                log_reflection_evidence_missing(
                    db,
                    "goals",
                    &evidence_ids,
                    &gate.invalid_evidence_ids,
                    &gate.allowlist_rejected_ids,
                )
                .await;
            }
        }
    }

    let identity_update_requested = response.identity_thread.is_some()
        || response.identity_confidence.is_some()
        || response.identity_uncertainty_note.is_some();
    if identity_update_requested {
        let evidence_ids = normalize_id_list(
            response
                .identity_evidence_event_ids
                .as_deref()
                .unwrap_or(&[]),
        );
        if evidence_ids.is_empty() {
            log_reflection_evidence_missing(db, "identity", &evidence_ids, &[], &[]).await;
        } else {
            let gate = validate_reflection_evidence(
                db,
                allowlist_set,
                &evidence_ids,
                relax_evidence_gating,
            )
            .await;
            if gate.allowed {
                if gate.relaxed {
                    log_reflection_evidence_relaxed(
                        db,
                        "identity",
                        &evidence_ids,
                        &gate.invalid_evidence_ids,
                        &gate.allowlist_rejected_ids,
                    )
                    .await;
                }
                let mut requires_telemetry = false;
                if let Some(thread) = response.identity_thread.as_deref() {
                    requires_telemetry |= contains_state_claim(thread);
                }
                if let Some(note) = response.identity_uncertainty_note.as_deref() {
                    requires_telemetry |= contains_state_claim(note);
                }
                if requires_telemetry && !evidence_includes_telemetry(db, &evidence_ids).await {
                    log_reflection_telemetry_missing(db, "identity", &evidence_ids).await;
                } else {
                    let mut identity_changed = false;
                    if let Some(thread) = response.identity_thread.as_deref() {
                        let trimmed = thread.trim();
                        if trimmed.is_empty() {
                            if model.identity_thread.is_some() {
                                model.identity_thread = None;
                                identity_changed = true;
                            }
                        } else if model.identity_thread.as_deref() != Some(trimmed) {
                            model.identity_thread = Some(trimmed.to_string());
                            identity_changed = true;
                        }
                    }
                    if let Some(note) = response.identity_uncertainty_note.as_deref() {
                        let trimmed = note.trim();
                        let next = if trimmed.is_empty() { None } else { Some(trimmed.to_string()) };
                        if model.identity_uncertainty_note != next {
                            model.identity_uncertainty_note = next;
                            identity_changed = true;
                        }
                    }
                    let mut next_confidence = model.identity_confidence;
                    if let Some(conf) = response.identity_confidence {
                        next_confidence = conf.clamp(0.0, 1.0);
                    } else if response.identity_uncertainty_note.is_some() {
                        next_confidence = (next_confidence - 0.1).max(0.0);
                    }
                    if (next_confidence - model.identity_confidence).abs() > f32::EPSILON {
                        model.identity_confidence = next_confidence;
                        identity_changed = true;
                    }
                    if identity_changed {
                        model.identity_updated_at = Some(Utc::now().to_rfc3339());
                        applied["identity"] = serde_json::json!(true);

                        let snapshot = serde_json::json!({
                            "identity_thread": model.identity_thread.clone().unwrap_or_default(),
                            "identity_confidence": model.identity_confidence,
                            "identity_uncertainty_note": model.identity_uncertainty_note.clone().unwrap_or_default(),
                            "identity_updated_at": model.identity_updated_at.clone().unwrap_or_default(),
                        });
                        let invariants = match db.get_settings().await {
                            Ok(settings) => Some(serde_json::json!({
                                "assistant_display_name": settings.assistant_display_name.unwrap_or_else(|| "Ergo".to_string()),
                                "user_display_name": settings.user_display_name.unwrap_or_else(|| "User".to_string()),
                                "contract": "symbiote_contract_v1",
                            })),
                            Err(_) => None,
                        };
                        let snapshot_id = db
                            .create_identity_snapshot(
                                &snapshot,
                                &evidence_ids,
                                invariants.as_ref(),
                                Some("reflection_identity"),
                                Some("self_reflection"),
                            )
                            .await
                            .ok();
                        let _ = system_log::log_event(
                            &db.pool,
                            None,
                            "info",
                            "system",
                            None,
                            None,
                            serde_json::json!({
                                "event": "identity_snapshot_written",
                                "snapshot_id": snapshot_id,
                                "evidence_count": evidence_ids.len(),
                                "identity_confidence": model.identity_confidence,
                            }),
                        )
                        .await;
                    }
                }
            } else {
                log_reflection_evidence_missing(
                    db,
                    "identity",
                    &evidence_ids,
                    &gate.invalid_evidence_ids,
                    &gate.allowlist_rejected_ids,
                )
                .await;
            }
        }
    }

    let mut self_memory_count = 0u64;
    if let Some(writes) = &response.self_memory_writes {
        for write in writes {
            if write.evidence_snippet.trim().is_empty() || write.observed_at.trim().is_empty() || write.reason.trim().is_empty() {
                continue;
            }
            let evidence_ids = normalize_id_list(write.evidence_event_ids.as_deref().unwrap_or(&[]));
            if evidence_ids.is_empty() {
                log_reflection_evidence_missing(db, "self_memory", &evidence_ids, &[], &[]).await;
                continue;
            }
            let gate = validate_reflection_evidence(
                db,
                allowlist_set,
                &evidence_ids,
                relax_evidence_gating,
            )
            .await;
            if !gate.allowed {
                log_reflection_evidence_missing(
                    db,
                    "self_memory",
                    &evidence_ids,
                    &gate.invalid_evidence_ids,
                    &gate.allowlist_rejected_ids,
                )
                .await;
                continue;
            }
            if gate.relaxed {
                log_reflection_evidence_relaxed(
                    db,
                    "self_memory",
                    &evidence_ids,
                    &gate.invalid_evidence_ids,
                    &gate.allowlist_rejected_ids,
                )
                .await;
            }
            match write.kind.as_str() {
                "fact" => {
                    let key = write.key.as_deref().unwrap_or("").trim();
                    let value = write.value.as_deref().unwrap_or("").trim();
                    if !key.is_empty() && !value.is_empty() {
                        let snippet = format!("{} | reason: {}", write.evidence_snippet.trim(), write.reason.trim());
                        let observed = chrono::DateTime::parse_from_rfc3339(&write.observed_at)
                            .ok()
                            .map(|t| t.with_timezone(&Utc));
                        if let Ok(write_result) = write_self_fact(
                            &db.pool,
                            key,
                            value,
                            &snippet,
                            observed,
                            crate::core::memory::types::SourceType::System,
                            Some(&evidence_ids),
                        )
                        .await
                        {
                            record_claim_fact(db, key, value, &write_result).await;
                        }
                        self_memory_count += 1;
                    }
                }
                "rel" => {
                    let rel_type = write.rel_type.as_deref().unwrap_or("").trim();
                    if rel_type.is_empty() {
                        continue;
                    }
                    let participants: Vec<(String, String)> = write.participants.clone().unwrap_or_default()
                        .into_iter()
                        .map(|p| (p.role, p.label))
                        .collect();
                    if participants.is_empty() {
                        continue;
                    }
                    let snippet = format!("{} | reason: {}", write.evidence_snippet.trim(), write.reason.trim());
                    let observed = chrono::DateTime::parse_from_rfc3339(&write.observed_at)
                        .ok()
                        .map(|t| t.with_timezone(&Utc));
                    if let Ok(write_result) = write_self_rel(
                        &db.pool,
                        rel_type,
                        &participants,
                        &snippet,
                        observed,
                        crate::core::memory::types::SourceType::System,
                        Some(&evidence_ids),
                    )
                    .await
                    {
                        record_claim_rel(db, rel_type, &participants, &write_result).await;
                    }
                    self_memory_count += 1;
                }
                _ => {}
            }
        }
    }

    applied["self_memory"] = serde_json::json!(self_memory_count);

    refresh_controller_state_from_evidence(db, model).await;

    Ok(applied)
}

async fn apply_persona_delta(
    db: &Db,
    model: &mut SelfModel,
    delta: &PersonaDelta,
    reason: &str,
    observed_at: &str,
    evidence_ids: &[i64],
) -> Result<bool, String> {
    let pool = &db.pool;
    let mut persona = model.persona.as_object().cloned().unwrap_or_default();

    let deltas = [
        ("tone", delta.tone),
        ("verbosity", delta.verbosity),
        ("directness", delta.directness),
        ("formality", delta.formality),
        ("initiative", delta.initiative),
    ];

    let mut nonzero: Vec<f32> = Vec::new();
    for (_, d) in &deltas {
        if let Some(v) = d {
            if v.abs() > f32::EPSILON {
                nonzero.push(v.abs());
            }
        }
    }

    if nonzero.len() > 2 {
        return Err("TooManyPersonaAxesChanged".to_string());
    }

    if nonzero.len() == 2 {
        nonzero.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let large = nonzero[0];
        let small = nonzero[1];
        if large > persona_config::MAX_LARGE_DELTA_PER_REFLECTION + f32::EPSILON {
            return Err("PersonaLargeDeltaTooBig".to_string());
        }
        if small > persona_config::MAX_DELTA_PER_REFLECTION + f32::EPSILON {
            return Err("PersonaSmallDeltaTooBig".to_string());
        }
    } else if nonzero.len() == 1 {
        let delta = nonzero[0];
        if delta > persona_config::MAX_LARGE_DELTA_PER_REFLECTION + f32::EPSILON {
            return Err("PersonaDeltaTooLarge".to_string());
        }
    }

    let today = Utc::now().format("%Y-%m-%d").to_string();
    let mut daily = model.persona_daily_delta.as_object().cloned().unwrap_or_default();
    if model.persona_last_delta_date.as_deref() != Some(today.as_str()) {
        daily.clear();
        model.persona_last_delta_date = Some(today.clone());
    }

    let mut total_daily = 0.0f32;
    for v in daily.values() {
        total_daily += v.as_f64().unwrap_or(0.0) as f32;
    }

    let mut changed = false;
    for (axis, d_opt) in deltas {
        let Some(d) = d_opt else { continue; };
        if d.abs() <= f32::EPSILON {
            continue;
        }
        let current = persona.get(axis).and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
        let mut next = current + d;
        let bounds = match axis {
            "tone" => persona_config::PERSONA_TONE,
            "verbosity" => persona_config::PERSONA_VERBOSITY,
            "directness" => persona_config::PERSONA_DIRECTNESS,
            "formality" => persona_config::PERSONA_FORMALITY,
            "initiative" => persona_config::PERSONA_INITIATIVE,
            _ => persona_config::PERSONA_TONE,
        };
        next = persona_config::clamp_value(next, &bounds);

        let delta_used = (next - current).abs();
        if total_daily + delta_used > persona_config::MAX_TOTAL_DAILY_DELTA + f32::EPSILON {
            return Err("PersonaDailyDeltaExceeded".to_string());
        }

        persona.insert(axis.to_string(), serde_json::json!(next));
        let current_daily = daily.get(axis).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        daily.insert(axis.to_string(), serde_json::json!(current_daily + delta_used));
        total_daily += delta_used;

        let snippet = format!("persona.{} updated to {:.2} | reason: {}", axis, next, reason);
        let observed = chrono::DateTime::parse_from_rfc3339(observed_at)
            .ok()
            .map(|t| t.with_timezone(&Utc));
        if let Ok(write_result) = write_self_fact(
            pool,
            &format!("persona.{}", axis),
            &format!("{:.2}", next),
            &snippet,
            observed,
            crate::core::memory::types::SourceType::System,
            Some(evidence_ids),
        )
        .await
        {
            record_claim_fact(db, &format!("persona.{}", axis), &format!("{:.2}", next), &write_result).await;
        }
        changed = true;
    }

    model.persona = serde_json::Value::Object(persona);
    model.persona_daily_delta = serde_json::Value::Object(daily);

    Ok(changed)
}

async fn apply_goals(
    db: &Db,
    model: &mut SelfModel,
    goals: &[String],
    reason: &str,
    observed_at: &str,
    evidence_ids: &[i64],
) -> Result<bool, String> {
    let pool = &db.pool;
    let normalized = normalize_goals(&serde_json::json!(goals));
    let new_goals: Vec<String> = normalized
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let current_goals: Vec<String> = model.goals
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    if new_goals == current_goals {
        return Ok(false);
    }

    let added: Vec<String> = new_goals.iter().filter(|g| !current_goals.contains(g)).cloned().collect();
    let removed: Vec<String> = current_goals.iter().filter(|g| !new_goals.contains(g)).cloned().collect();

    let observed = chrono::DateTime::parse_from_rfc3339(observed_at)
        .ok()
        .map(|t| t.with_timezone(&Utc));

    for goal in added {
        let snippet = format!("goal added: {} | reason: {}", goal, reason);
        if let Ok(write_result) = write_self_fact(
            pool,
            "goal",
            &goal,
            &snippet,
            observed,
            crate::core::memory::types::SourceType::System,
            Some(evidence_ids),
        )
        .await
        {
            record_claim_fact(db, "goal", &goal, &write_result).await;
        }
    }

    for goal in removed {
        let snippet = format!("goal removed: {} | reason: {}", goal, reason);
        if let Ok(write_result) = write_self_fact(
            pool,
            "goal_removed",
            &goal,
            &snippet,
            observed,
            crate::core::memory::types::SourceType::System,
            Some(evidence_ids),
        )
        .await
        {
            record_claim_fact(db, "goal_removed", &goal, &write_result).await;
        }
    }

    model.goals = normalized;
    Ok(true)
}

fn populate_self_model_metadata(model: &mut SelfModel) {
    if model.capabilities == serde_json::json!([]) {
        model.capabilities = serde_json::json!([
            "memory",
            "rolling_summary",
            "conflict_resolution",
            "clarification",
            "reminders",
            "voice"
        ]);
    }
    if model.limitations == serde_json::json!([]) {
        model.limitations = serde_json::json!([
            "no external web",
            "local tools only"
        ]);
    }
    if model.active_tools == serde_json::json!([]) {
        model.active_tools = serde_json::json!([
            "voice",
            "time",
            "shell"
        ]);
    }
}

async fn build_behavior_metrics(db: &Db) -> Option<serde_json::Value> {
    let history = db.get_history(30).await.ok()?;
    let assistant: Vec<String> = history
        .into_iter()
        .filter(|m| m.role == "assistant" && m.status == "complete")
        .map(|m| m.content)
        .collect();
    if assistant.is_empty() {
        return Some(serde_json::json!({}));
    }

    let mut total_chars = 0usize;
    let mut total_sentences = 0usize;
    for content in &assistant {
        total_chars += content.chars().count();
        total_sentences += content.matches('.').count().max(1);
    }

    let avg_chars = total_chars as f64 / assistant.len() as f64;
    let avg_sentences = total_sentences as f64 / assistant.len() as f64;

    Some(serde_json::json!({
        "recent_assistant_count": assistant.len(),
        "avg_chars": avg_chars,
        "avg_sentences": avg_sentences
    }))
}

fn normalize_persona(persona: &serde_json::Value) -> (serde_json::Value, bool) {
    let defaults = serde_json::json!({
        "tone": 0.55,
        "verbosity": 0.45,
        "directness": 0.7,
        "formality": 0.5,
        "initiative": 0.5,
    });

    let mut was_clamped = false;
    let mut result = defaults.as_object().cloned().unwrap_or_default();
    if let Some(obj) = persona.as_object() {
        for (k, v) in obj {
            result.insert(k.clone(), v.clone());
        }
    }

    let clamp_axis = |value: &serde_json::Value, bounds: &persona_config::PersonaBounds| -> (serde_json::Value, bool) {
        let val = value.as_f64().unwrap_or(bounds.min as f64) as f32;
        let clamped = persona_config::clamp_value(val, bounds);
        (serde_json::json!(clamped), (clamped - val).abs() > f32::EPSILON)
    };

    if let Some(value) = result.get("tone").cloned() {
        let (clamped, changed) = clamp_axis(&value, &persona_config::PERSONA_TONE);
        result.insert("tone".to_string(), clamped);
        was_clamped |= changed;
    }
    if let Some(value) = result.get("verbosity").cloned() {
        let (clamped, changed) = clamp_axis(&value, &persona_config::PERSONA_VERBOSITY);
        result.insert("verbosity".to_string(), clamped);
        was_clamped |= changed;
    }
    if let Some(value) = result.get("directness").cloned() {
        let (clamped, changed) = clamp_axis(&value, &persona_config::PERSONA_DIRECTNESS);
        result.insert("directness".to_string(), clamped);
        was_clamped |= changed;
    }
    if let Some(value) = result.get("formality").cloned() {
        let (clamped, changed) = clamp_axis(&value, &persona_config::PERSONA_FORMALITY);
        result.insert("formality".to_string(), clamped);
        was_clamped |= changed;
    }
    if let Some(value) = result.get("initiative").cloned() {
        let (clamped, changed) = clamp_axis(&value, &persona_config::PERSONA_INITIATIVE);
        result.insert("initiative".to_string(), clamped);
        was_clamped |= changed;
    }

    (serde_json::Value::Object(result), was_clamped)
}

fn normalize_goals(goals: &serde_json::Value) -> serde_json::Value {
    let allowed = ["reduce conflict rate", "reduce clarification loops", "improve concision and relevance"];
    let mut normalized: Vec<String> = Vec::new();
    if let Some(list) = goals.as_array() {
        for g in list {
            if let Some(s) = g.as_str() {
                let trimmed = s.trim();
                if allowed.iter().any(|a| a.eq_ignore_ascii_case(trimmed)) {
                    if !normalized.iter().any(|v| v.eq_ignore_ascii_case(trimmed)) {
                        normalized.push(trimmed.to_string());
                    }
                }
            }
        }
    }
    if normalized.len() > 3 {
        normalized.truncate(3);
    }
    serde_json::json!(normalized)
}

async fn record_claim_fact(db: &Db, key: &str, value: &str, write_result: &SelfMemoryWriteResult) {
    let evidence_ids = write_result
        .evidence_event_id
        .map(|id| vec![id])
        .unwrap_or_default();
    let belief_ids = vec![write_result.belief_id];
    let claim = SelfClaimInput {
        claim_text: self_claims::claim_text_for_fact(key, value),
        claim_key: self_claims::claim_key_for_fact(key, value),
        evidence_event_ids: evidence_ids,
        belief_ids,
        confidence: 1.0,
        polarity: "assert".to_string(),
        source_run_id: None,
        conversation_id: Some("default".to_string()),
        provisional: false,
        source_type: None,
        requires_validation: false,
        ttl_seconds: None,
        promotion_rule: None,
        eviction_rule: None,
    };
    let _ = self_claims::record_self_claim(db, claim).await;
}

async fn record_claim_rel(
    db: &Db,
    rel_type: &str,
    participants: &[(String, String)],
    write_result: &SelfMemoryWriteResult,
) {
    let evidence_ids = write_result
        .evidence_event_id
        .map(|id| vec![id])
        .unwrap_or_default();
    let belief_ids = vec![write_result.belief_id];
    let claim = SelfClaimInput {
        claim_text: self_claims::claim_text_for_rel(rel_type, participants),
        claim_key: self_claims::claim_key_for_rel(rel_type, participants),
        evidence_event_ids: evidence_ids,
        belief_ids,
        confidence: 1.0,
        polarity: "assert".to_string(),
        source_run_id: None,
        conversation_id: Some("default".to_string()),
        provisional: false,
        source_type: None,
        requires_validation: false,
        ttl_seconds: None,
        promotion_rule: None,
        eviction_rule: None,
    };
    let _ = self_claims::record_self_claim(db, claim).await;
}

fn normalize_id_list(raw: &[i64]) -> Vec<i64> {
    let mut ids: Vec<i64> = raw.iter().copied().filter(|v| *v > 0).collect();
    ids.sort();
    ids.dedup();
    ids
}

fn collect_reflection_evidence_ids(response: &ReflectionResponse) -> Vec<i64> {
    let mut ids = Vec::new();
    if let Some(list) = response.persona_evidence_event_ids.as_deref() {
        ids.extend(list.iter().copied());
    }
    if let Some(list) = response.goals_evidence_event_ids.as_deref() {
        ids.extend(list.iter().copied());
    }
    if let Some(list) = response.identity_evidence_event_ids.as_deref() {
        ids.extend(list.iter().copied());
    }
    if let Some(writes) = response.self_memory_writes.as_deref() {
        for write in writes {
            if let Some(list) = write.evidence_event_ids.as_deref() {
                ids.extend(list.iter().copied());
            }
        }
    }
    normalize_id_list(&ids)
}

fn response_has_changes(response: &ReflectionResponse) -> bool {
    response.persona_delta.is_some()
        || response.goals.as_ref().map(|g| !g.is_empty()).unwrap_or(false)
        || response.identity_thread.is_some()
        || response.identity_confidence.is_some()
        || response.identity_uncertainty_note.is_some()
        || response
            .self_memory_writes
            .as_ref()
            .map(|writes| !writes.is_empty())
            .unwrap_or(false)
}

fn response_has_state_claims(response: &ReflectionResponse) -> bool {
    if let Some(reason) = response.persona_reason.as_deref() {
        if contains_state_claim(reason) {
            return true;
        }
    }
    if let Some(reason) = response.goals_reason.as_deref() {
        if contains_state_claim(reason) {
            return true;
        }
    }
    if let Some(thread) = response.identity_thread.as_deref() {
        if contains_state_claim(thread) {
            return true;
        }
    }
    if let Some(note) = response.identity_uncertainty_note.as_deref() {
        if contains_state_claim(note) {
            return true;
        }
    }
    false
}

fn attach_telemetry_evidence(response: &mut ReflectionResponse, telemetry_ids: &[i64]) -> bool {
    if telemetry_ids.is_empty() {
        return false;
    }
    let mut attached = false;
    if response.persona_delta.is_some() {
        response.persona_evidence_event_ids = Some(telemetry_ids.to_vec());
        attached = true;
    }
    if response.goals.is_some() {
        response.goals_evidence_event_ids = Some(telemetry_ids.to_vec());
        attached = true;
    }
    let identity_update_requested = response.identity_thread.is_some()
        || response.identity_confidence.is_some()
        || response.identity_uncertainty_note.is_some();
    if identity_update_requested {
        response.identity_evidence_event_ids = Some(telemetry_ids.to_vec());
        attached = true;
    }
    attached
}

fn scrub_reflection_state_claims(response: &mut ReflectionResponse) {
    response.persona_delta = None;
    response.persona_reason = None;
    response.persona_observed_at = None;
    response.persona_evidence_event_ids = None;
    response.goals = None;
    response.goals_reason = None;
    response.goals_observed_at = None;
    response.goals_evidence_event_ids = None;
    response.identity_thread = None;
    response.identity_confidence = None;
    response.identity_uncertainty_note = None;
    response.identity_evidence_event_ids = None;
    response.self_memory_writes = None;
}

async fn recent_telemetry_evidence(db: &Db, limit: i64) -> Vec<(i64, String)> {
    let rows = sqlx::query(
        "SELECT id, source_type FROM ics_evidence_events
         WHERE source_type IN ('wave_state','attention_schema_snapshot','prediction_residual_snapshot')
         ORDER BY datetime(created_at) DESC
         LIMIT ?",
    )
    .bind(limit.max(1))
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();
    rows.into_iter()
        .map(|row| {
            let id: i64 = row.get("id");
            let source_type: String = row.try_get("source_type").unwrap_or_default();
            (id, source_type)
        })
        .collect()
}

fn select_telemetry_evidence_ids(rows: &[(i64, String)]) -> Vec<i64> {
    let mut wave: Option<i64> = None;
    let mut attention: Option<i64> = None;
    let mut residual: Option<i64> = None;
    for (id, source_type) in rows.iter() {
        match source_type.as_str() {
            "wave_state" if wave.is_none() => wave = Some(*id),
            "attention_schema_snapshot" if attention.is_none() => attention = Some(*id),
            "prediction_residual_snapshot" if residual.is_none() => residual = Some(*id),
            _ => {}
        }
    }
    let mut ids = Vec::new();
    if let Some(id) = wave {
        ids.push(id);
    }
    if let Some(id) = attention {
        ids.push(id);
    }
    if let Some(id) = residual {
        ids.push(id);
    }
    normalize_id_list(&ids)
}

async fn load_reflection_evidence_sources(db: &Db, evidence_ids: &[i64]) -> Vec<serde_json::Value> {
    if evidence_ids.is_empty() {
        return Vec::new();
    }
    let mut builder: QueryBuilder<sqlx::Sqlite> =
        QueryBuilder::new("SELECT id, source_type FROM ics_evidence_events WHERE id IN (");
    let mut separated = builder.separated(", ");
    for id in evidence_ids {
        separated.push_bind(id);
    }
    separated.push_unseparated(")");
    let query = builder.build();
    let rows = query.fetch_all(&db.pool).await.unwrap_or_default();
    rows.into_iter()
        .map(|row| {
            let id: i64 = row.get("id");
            let source_type: String = row.try_get("source_type").unwrap_or_default();
            serde_json::json!({
                "id": id,
                "source_type": source_type,
            })
        })
        .collect()
}

async fn recent_noop_stage_exists(db: &Db, window_mins: i64) -> bool {
    let window = format!("-{} minutes", window_mins);
    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'self_reflection_noop_staged'
           AND datetime(timestamp) >= datetime('now', ?)",
    )
    .bind(window)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    count > 0
}

async fn count_reflection_parse_failures(db: &Db, window_mins: i64) -> i64 {
    let window = format!("-{} minutes", window_mins);
    sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'self_reflection_parse_failed'
           AND datetime(timestamp) >= datetime('now', ?)",
    )
    .bind(window)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0)
}

#[derive(Debug, Clone)]
struct ReflectionEvidenceGate {
    allowed: bool,
    relaxed: bool,
    invalid_evidence_ids: Vec<i64>,
    allowlist_rejected_ids: Vec<i64>,
}

async fn validate_reflection_evidence(
    db: &Db,
    allowlist_set: &HashSet<i64>,
    evidence_ids: &[i64],
    relax_evidence_gating: bool,
) -> ReflectionEvidenceGate {
    if evidence_ids.is_empty() {
        return ReflectionEvidenceGate {
            allowed: false,
            relaxed: false,
            invalid_evidence_ids: Vec::new(),
            allowlist_rejected_ids: Vec::new(),
        };
    }

    let normalized = normalize_id_list(evidence_ids);
    let allowlist_rejected_ids: Vec<i64> = normalized
        .iter()
        .copied()
        .filter(|id| !allowlist_set.contains(id))
        .collect();
    let validation = validate_evidence_ids_with_pool(&db.pool, &normalized, &[], true).await;
    let invalid_evidence_ids = validation.invalid_evidence_ids.clone();
    let mut allowed = allowlist_rejected_ids.is_empty() && validation.evidence_ok();
    let mut relaxed = false;
    if !allowed && relax_evidence_gating {
        allowed = true;
        relaxed = true;
    }
    ReflectionEvidenceGate {
        allowed,
        relaxed,
        invalid_evidence_ids,
        allowlist_rejected_ids,
    }
}

async fn log_reflection_evidence_missing(
    db: &Db,
    kind: &str,
    evidence_ids: &[i64],
    invalid_evidence_ids: &[i64],
    allowlist_rejected_ids: &[i64],
) {
    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "self_reflection",
        None,
        None,
        serde_json::json!({
            "event": "self_reflection_missing_evidence",
            "kind": kind,
            "evidence_event_ids": evidence_ids,
            "invalid_evidence_ids": invalid_evidence_ids,
            "allowlist_rejected_ids": allowlist_rejected_ids,
        }),
    )
    .await;
}

async fn log_reflection_evidence_relaxed(
    db: &Db,
    kind: &str,
    evidence_ids: &[i64],
    invalid_evidence_ids: &[i64],
    allowlist_rejected_ids: &[i64],
) {
    let _ = system_log::log_event(
        &db.pool,
        None,
        "warn",
        "self_reflection",
        None,
        None,
        serde_json::json!({
            "event": "self_reflection_evidence_relaxed",
            "kind": kind,
            "evidence_event_ids": evidence_ids,
            "invalid_evidence_ids": invalid_evidence_ids,
            "allowlist_rejected_ids": allowlist_rejected_ids,
        }),
    )
    .await;
}

fn contains_state_claim(text: &str) -> bool {
    let lower = text.to_lowercase();
    let markers = [
        "stability",
        "arousal",
        "wave",
        "attention",
        "residual",
        "turbulence",
        "drift",
        "coherence",
        "fragmentation",
        "capacity",
        "focus",
    ];
    markers.iter().any(|marker| lower.contains(marker))
}

async fn evidence_includes_telemetry(db: &Db, evidence_ids: &[i64]) -> bool {
    if evidence_ids.is_empty() {
        return false;
    }
    let mut builder: QueryBuilder<sqlx::Sqlite> =
        QueryBuilder::new("SELECT source_type FROM ics_evidence_events WHERE id IN (");
    let mut separated = builder.separated(", ");
    for id in evidence_ids.iter() {
        separated.push_bind(id);
    }
    separated.push_unseparated(")");
    let rows = builder
        .build()
        .fetch_all(&db.pool)
        .await
        .unwrap_or_default();
    for row in rows {
        let source_type: String = row.try_get("source_type").unwrap_or_default();
        if matches!(
            source_type.as_str(),
            "wave_state" | "attention_schema_snapshot" | "prediction_residual_snapshot"
        ) {
            return true;
        }
    }
    false
}

async fn log_reflection_telemetry_missing(db: &Db, kind: &str, evidence_ids: &[i64]) {
    let _ = system_log::log_event(
        &db.pool,
        None,
        "warn",
        "self_reflection",
        None,
        None,
        serde_json::json!({
            "event": "self_reflection_missing_telemetry",
            "kind": kind,
            "evidence_event_ids": evidence_ids,
        }),
    )
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::fs;
    use std::path::PathBuf;
    use std::collections::HashSet;

    async fn setup_pool() -> sqlx::SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory pool");

        let schema_path = PathBuf::from("src/db/schema.sql");
        let schema_sql = fs::read_to_string(&schema_path).expect("Failed to read schema.sql");
        sqlx::query(&schema_sql)
            .execute(&pool)
            .await
            .expect("Failed to apply schema");

        sqlx::query("INSERT INTO settings (id, schema_version, api_base_url) VALUES (1, 1, 'http://localhost')")
            .execute(&pool)
            .await
            .expect("Failed to seed settings");

        pool
    }

    #[tokio::test]
    async fn reflection_persona_requires_evidence_ids() {
        let pool = setup_pool().await;
        let db = Db { pool };
        let mut model = SelfModel {
            capabilities: json!([]),
            limitations: json!([]),
            active_tools: json!([]),
            memory_health: json!({}),
            persona: json!({
                "tone": 0.5,
                "verbosity": 0.5,
                "directness": 0.5,
                "formality": 0.5,
                "initiative": 0.5,
            }),
            persona_daily_delta: json!({}),
            persona_last_delta_date: None,
            goals: json!([]),
            identity_thread: None,
            identity_confidence: 0.5,
            identity_uncertainty_note: None,
            identity_updated_at: None,
            reflection_status: json!({}),
            reflection_frozen: false,
            last_reflection_at: None,
            internal_state_summary: json!({}),
            internal_state_map_version: None,
            unified_state: json!({}),
            unified_state_evidence: json!({}),
            unified_state_updated_at: None,
            updated_at: Utc::now().to_rfc3339(),
        };

        let response = ReflectionResponse {
            persona_delta: Some(PersonaDelta {
                tone: Some(0.05),
                verbosity: None,
                directness: None,
                formality: None,
                initiative: None,
            }),
            persona_reason: Some("observed recent clarity".to_string()),
            persona_observed_at: Some(Utc::now().to_rfc3339()),
            persona_evidence_event_ids: None,
            goals: None,
            goals_reason: None,
            goals_observed_at: None,
            goals_evidence_event_ids: None,
            identity_thread: None,
            identity_confidence: None,
            identity_uncertainty_note: None,
            identity_evidence_event_ids: None,
            self_memory_writes: None,
            rejection_reason: None,
        };

        let applied = apply_reflection_result(&db, &mut model, &response, &HashSet::new(), false)
            .await
            .expect("apply");
        assert_eq!(applied.get("persona").and_then(|v| v.as_bool()), Some(false));
    }

    #[tokio::test]
    async fn reflection_self_memory_requires_evidence_ids() {
        let pool = setup_pool().await;
        let db = Db { pool };
        let mut model = SelfModel {
            capabilities: json!([]),
            limitations: json!([]),
            active_tools: json!([]),
            memory_health: json!({}),
            persona: json!({}),
            persona_daily_delta: json!({}),
            persona_last_delta_date: None,
            goals: json!([]),
            identity_thread: None,
            identity_confidence: 0.5,
            identity_uncertainty_note: None,
            identity_updated_at: None,
            reflection_status: json!({}),
            reflection_frozen: false,
            last_reflection_at: None,
            internal_state_summary: json!({}),
            internal_state_map_version: None,
            unified_state: json!({}),
            unified_state_evidence: json!({}),
            unified_state_updated_at: None,
            updated_at: Utc::now().to_rfc3339(),
        };

        let response = ReflectionResponse {
            persona_delta: None,
            persona_reason: None,
            persona_observed_at: None,
            persona_evidence_event_ids: None,
            goals: None,
            goals_reason: None,
            goals_observed_at: None,
            goals_evidence_event_ids: None,
            identity_thread: None,
            identity_confidence: None,
            identity_uncertainty_note: None,
            identity_evidence_event_ids: None,
            self_memory_writes: Some(vec![SelfMemoryWrite {
                kind: "fact".to_string(),
                key: Some("telemetry.test".to_string()),
                value: Some("1.0".to_string()),
                rel_type: None,
                participants: None,
                evidence_event_ids: Some(vec![]),
                evidence_snippet: "snippet".to_string(),
                observed_at: Utc::now().to_rfc3339(),
                reason: "reason".to_string(),
            }]),
            rejection_reason: None,
        };

        let applied = apply_reflection_result(&db, &mut model, &response, &HashSet::new(), false)
            .await
            .expect("apply");
        assert_eq!(applied.get("self_memory").and_then(|v| v.as_u64()), Some(0));
    }
}
