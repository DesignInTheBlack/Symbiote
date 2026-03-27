use std::sync::Arc;
use std::collections::HashMap;
use tauri::State;
use crate::db::Db;
use crate::core::ChatManager;
use crate::core::reminder_blocks;
use crate::core::episodic;
use crate::core::system_log;
use crate::core::system_controls;
use crate::core::system_health;
use crate::core::model_client::ModelClient;
use crate::core::prompt_loader;
use crate::core::subject_state;
use crate::core::kernel::KernelState;
use crate::models::{
    Message,
    Settings,
    SystemControlEntry,
    SystemControlEvent,
    SystemHealthSnapshot,
    ContextTagEntry,
    UserIntentSummary,
    EvidenceLineageEntry,
    OutcomeEvent,
    SystemLogEntry,
};
use crate::core::voice_manager_v2::VoiceManager; // Import
use tauri::{AppHandle, Manager, Emitter};
use serde_json::json;
use std::fs;
use std::path::Path;
use chrono::Utc;
use uuid::Uuid;
use sha2::{Digest, Sha256};
use serde_json::Value;
use sqlx::Row;
use serde::Serialize;

#[derive(Serialize)]
pub struct PromptStatus {
    pub prompt_source: String,
    pub primary_prompt_hash: String,
    pub memory_prompt_hash: String,
    pub canonical_primary_hash: String,
    pub override_hash: Option<String>,
    pub override_active: bool,
    pub override_mismatch: bool,
}

#[derive(Serialize, Default)]
pub struct WaveStatus {
    pub coherence: Option<f64>,
    pub dominance: Option<f64>,
    pub turbulence: Option<f64>,
    pub drift: Option<f64>,
    pub fragmentation: Option<f64>,
    pub total_energy: Option<f64>,
    pub band_energy: Option<Value>,
    pub last_projection_at: Option<String>,
    pub last_contribution_at: Option<String>,
    pub projection_age_seconds: Option<i64>,
    pub contribution_age_seconds: Option<i64>,
}

#[derive(Serialize)]
pub struct CandidateEvidenceSummary {
    pub candidate_kind: String,
    pub candidate_count: i64,
    pub with_evidence: i64,
    pub evidence_event_ids: Vec<i64>,
    pub belief_ids: Vec<i64>,
}

#[derive(Serialize)]
pub struct DiagnosticsSnapshot {
    pub system_controls: Vec<SystemControlEntry>,
    pub candidate_evidence: Vec<CandidateEvidenceSummary>,
    pub memory_write_blocked: Vec<SystemLogEntry>,
    pub memory_pass_invalid_output: Vec<SystemLogEntry>,
}

fn extract_id_list_from_value(payload: &Value, key: &str) -> Vec<i64> {
    let mut ids = Vec::new();
    if let Some(array) = payload.get(key).and_then(|v| v.as_array()) {
        for item in array.iter().filter_map(|v| v.as_i64()) {
            if item > 0 {
                ids.push(item);
            }
        }
    } else if let Some(id) = payload.get(key).and_then(|v| v.as_i64()) {
        if id > 0 {
            ids.push(id);
        }
    }
    ids.sort();
    ids.dedup();
    ids
}

async fn load_system_logs_by_event(
    pool: &sqlx::SqlitePool,
    event: &str,
    limit: i64,
) -> Vec<SystemLogEntry> {
    let rows = sqlx::query(
        "SELECT id, timestamp, level, category, run_id, trace_id, payload
         FROM system_logs
         WHERE json_extract(payload, '$.event') = ?
         ORDER BY datetime(timestamp) DESC, rowid DESC
         LIMIT ?",
    )
    .bind(event)
    .bind(limit.max(1))
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let mut out = Vec::new();
    for row in rows {
        let payload_raw: String = row.try_get("payload").unwrap_or_else(|_| "{}".to_string());
        let payload = serde_json::from_str(&payload_raw).unwrap_or_else(|_| json!({}));
        out.push(SystemLogEntry {
            id: row.get("id"),
            timestamp: row.get("timestamp"),
            level: row.get("level"),
            category: row.get("category"),
            run_id: row.try_get("run_id").ok(),
            trace_id: row.try_get("trace_id").ok(),
            payload,
        });
    }
    out
}

fn inject_primary_names(prompt: &str, user_name: &str, assistant_name: &str) -> String {
    prompt
        .replace("{user_name}", user_name)
        .replace("{assistant_name}", assistant_name)
}

fn hash_prompt(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

// --- Voice Commands ---
#[tauri::command]
pub async fn start_voice_service(
    app: AppHandle,
    voice_manager: State<'_, VoiceManager>,
    db: State<'_, Arc<Db>>,
) -> Result<(), String> {
    println!("[COMMAND] Starting Voice Service V2...");
    let control_map = system_controls::load_control_map(&db).await;
    let voice_mode = system_controls::mode_for("voice_output", &control_map);
    if system_controls::mode_is_off(&voice_mode) {
        return Ok(());
    }
    voice_manager.start(&app).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn restart_voice_service(
    app: AppHandle,
    voice_manager: State<'_, VoiceManager>,
    db: State<'_, Arc<Db>>,
) -> Result<(), String> {
    println!("[COMMAND] Restarting Voice Service V2...");
    let control_map = system_controls::load_control_map(&db).await;
    let voice_mode = system_controls::mode_for("voice_output", &control_map);
    if system_controls::mode_is_off(&voice_mode) {
        return Ok(());
    }
    voice_manager.stop(Some(&app));
    voice_manager.start(&app).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn log_ui_timing(
    db: State<'_, Arc<Db>>,
    event: String,
    duration_ms: i64,
    detail: Option<String>,
    run_id: Option<String>,
    message_id: Option<String>,
) -> Result<(), String> {
    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "ui",
        run_id.as_deref(),
        None,
        json!({
            "event": event,
            "duration_ms": duration_ms,
            "detail": detail,
            "run_id": run_id,
            "message_id": message_id,
        }),
    )
    .await;
    Ok(())
}

#[tauri::command]
pub async fn log_tts_event(
    db: State<'_, Arc<Db>>,
    event: String,
    duration_ms: i64,
    detail: Option<String>,
    run_id: Option<String>,
    message_id: Option<String>,
) -> Result<(), String> {
    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "ui",
        run_id.as_deref(),
        None,
        json!({
            "event": event,
            "duration_ms": duration_ms,
            "detail": detail,
            "run_id": run_id,
            "message_id": message_id,
        }),
    )
    .await;
    Ok(())
}

#[tauri::command]
pub async fn get_messages(db: State<'_, Arc<Db>>, _settings: State<'_, Arc<Settings>>) -> Result<Vec<Message>, String> {
    db.get_history(50).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_settings(db: State<'_, Arc<Db>>) -> Result<Settings, String> {
    db.get_settings().await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_prompt_status(db: State<'_, Arc<Db>>) -> Result<PromptStatus, String> {
    let settings = db.get_settings().await.map_err(|e| e.to_string())?;
    let user_name = settings
        .user_display_name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("User");
    let assistant_name = settings
        .assistant_display_name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("Ergo");

    let prompt_set = prompt_loader::get_prompts().map_err(|e| e.to_string())?;
    let canonical_injected = inject_primary_names(&prompt_set.primary_prompt, user_name, assistant_name);
    let canonical_primary_hash = hash_prompt(&canonical_injected);

    let override_active = settings
        .system_prompt
        .as_deref()
        .map(|p| !p.trim().is_empty())
        .unwrap_or(false);
    let (override_hash, prompt_source, primary_prompt_hash) = if override_active {
        let injected = inject_primary_names(
            settings.system_prompt.as_deref().unwrap_or(""),
            user_name,
            assistant_name,
        );
        let override_hash = hash_prompt(&injected);
        (
            Some(override_hash.clone()),
            "settings_override".to_string(),
            override_hash,
        )
    } else {
        (
            None,
            prompt_set.source.clone(),
            prompt_set.primary_hash.clone(),
        )
    };

    let override_mismatch = override_active
        && !canonical_primary_hash.is_empty()
        && override_hash
            .as_ref()
            .map(|h| h != &canonical_primary_hash)
            .unwrap_or(false);

    Ok(PromptStatus {
        prompt_source,
        primary_prompt_hash,
        memory_prompt_hash: prompt_set.memory_hash.clone(),
        canonical_primary_hash,
        override_hash,
        override_active,
        override_mismatch,
    })
}

#[tauri::command]
pub async fn get_rolling_summary(db: State<'_, Arc<Db>>) -> Result<Option<String>, String> {
    db.get_rolling_summary("default").await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_rolling_summary_status(db: State<'_, Arc<Db>>) -> Result<crate::models::RollingSummaryStatus, String> {
    db.get_rolling_summary_status("default")
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_live_summary(db: State<'_, Arc<Db>>) -> Result<Option<String>, String> {
    db.get_live_summary("default").await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_live_summary_status(db: State<'_, Arc<Db>>) -> Result<crate::models::RollingSummaryStatus, String> {
    db.get_live_summary_status("default")
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_inner_monologue_entries(
    db: State<'_, Arc<Db>>,
    limit: Option<i64>,
) -> Result<Vec<crate::models::InnerMonologueEntry>, String> {
    let limit = limit.unwrap_or(50);
    db.list_inner_monologue_entries_with_candidates("default", limit)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn record_qualia_label(
    db: State<'_, Arc<Db>>,
    event_id: String,
    tag: String,
    intensity: f64,
    context: Option<Value>,
) -> Result<String, String> {
    let mut snapshot_hash = subject_state::latest_snapshot_hash(&db, "default").await;
    if snapshot_hash.is_none() {
        let mut state = KernelState::default_for("default");
        if let Ok(subject) = subject_state::build_subject_state(&db, &state, None).await {
            if let Ok(snapshot) = subject_state::snapshot_subject_state(
                &subject,
                &Uuid::new_v4().to_string(),
                "default",
                None,
            ) {
                let _ = subject_state::persist_subject_snapshot(&db, &snapshot).await;
                state.last_subject_snapshot_hash = Some(snapshot.snapshot_hash.clone());
                state.last_subject_snapshot_at = Some(snapshot.timestamp.clone());
                snapshot_hash = Some(snapshot.snapshot_hash);
            }
        }
    }
    let Some(snapshot_hash) = snapshot_hash else {
        return Err("no_subject_snapshot_available".to_string());
    };
    let label_id = Uuid::new_v4().to_string();
    let context_value = context.unwrap_or_else(|| json!({}));
    let mut broadcast_refs: Vec<String> = Vec::new();
    let mut ignition_active = false;
    if let Ok(Some(raw_state)) = sqlx::query_scalar::<_, String>(
        "SELECT subject_state_json FROM subject_snapshots WHERE snapshot_hash = ? LIMIT 1",
    )
    .bind(&snapshot_hash)
    .fetch_optional(&db.pool)
    .await
    {
        if let Ok(value) = serde_json::from_str::<Value>(&raw_state) {
            if let Some(state) = value.get("state") {
                if let Some(arr) = state
                    .get("workspace")
                    .and_then(|w| w.get("broadcast_refs"))
                    .and_then(|v| v.as_array())
                {
                    broadcast_refs = arr
                        .iter()
                        .filter_map(|v| v.as_str())
                        .map(|s| s.to_string())
                        .collect();
                }
                ignition_active = state
                    .get("workspace")
                    .and_then(|w| w.get("ignition"))
                    .and_then(|i| i.get("active"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
            }
        }
    }
    let mut enriched = match context_value {
        Value::Object(mut map) => {
            map.insert("workspace_refs".to_string(), json!(broadcast_refs));
            map.insert("ignition_active".to_string(), json!(ignition_active));
            map.insert("snapshot_hash".to_string(), json!(snapshot_hash));
            Value::Object(map)
        }
        other => json!({
            "context": other,
            "workspace_refs": broadcast_refs,
            "ignition_active": ignition_active,
            "snapshot_hash": snapshot_hash,
        }),
    };
    let context_json = Some(enriched.to_string());
    sqlx::query(
        "INSERT INTO qualia_labels (label_id, event_id, snapshot_hash, tag, intensity, context_json, created_at)
         VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
    )
    .bind(&label_id)
    .bind(&event_id)
    .bind(&snapshot_hash)
    .bind(tag.trim())
    .bind(intensity)
    .bind(context_json.as_deref())
    .execute(&db.pool)
    .await
    .map_err(|e| e.to_string())?;
    let mut episodic_event_id: Option<String> = None;
    if let Ok(Some(row)) = sqlx::query(
        "SELECT conversation_id, timestamp FROM subject_snapshots
         WHERE snapshot_hash = ?
         LIMIT 1",
    )
    .bind(&snapshot_hash)
    .fetch_optional(&db.pool)
    .await
    {
        let conversation_id: String = row.get("conversation_id");
        let snapshot_ts: String = row.get("timestamp");
        episodic_event_id = sqlx::query_scalar(
            "SELECT id FROM episodic_events
             WHERE id = ? AND conversation_id = ?
             LIMIT 1",
        )
        .bind(&event_id)
        .bind(&conversation_id)
        .fetch_optional(&db.pool)
        .await
        .ok()
        .flatten();
        if episodic_event_id.is_none() {
            episodic_event_id = sqlx::query_scalar(
                "SELECT id FROM episodic_events
                 WHERE conversation_id = ?
                   AND datetime(timestamp) BETWEEN datetime(?, '-2 minutes') AND datetime(?, '+2 minutes')
                 ORDER BY abs(strftime('%s', timestamp) - strftime('%s', ?)) ASC
                 LIMIT 1",
            )
            .bind(&conversation_id)
            .bind(&snapshot_ts)
            .bind(&snapshot_ts)
            .bind(&snapshot_ts)
            .fetch_optional(&db.pool)
            .await
            .ok()
            .flatten();
        }
    }
    if let Some(ref episodic_id) = episodic_event_id {
        enriched = match enriched {
            Value::Object(mut map) => {
                map.insert("episodic_event_id".to_string(), json!(episodic_id));
                Value::Object(map)
            }
            other => json!({
                "context": other,
                "episodic_event_id": episodic_id,
            }),
        };
        let updated_context = enriched.to_string();
        let _ = sqlx::query("UPDATE qualia_labels SET context_json = ? WHERE label_id = ?")
            .bind(updated_context)
            .bind(&label_id)
            .execute(&db.pool)
            .await;
        let qualia_evidence_ids = db
            .get_recent_evidence_ids_by_source_types(&["qualia_snapshot"], 6)
            .await;
        episodic::upsert_identity_index_for_qualia_label(
            &db.pool,
            episodic_id,
            tag.trim(),
            intensity,
            &qualia_evidence_ids,
        )
        .await;
    }
    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "kernel",
        None,
        None,
        json!({
            "event": "qualia_label_recorded",
            "label_id": label_id,
            "event_id": event_id,
            "episodic_event_id": episodic_event_id,
            "tag": tag,
            "intensity": intensity,
            "snapshot_hash": snapshot_hash,
        }),
    )
    .await;
    Ok(label_id)
}

#[tauri::command]
pub async fn record_qualia_reward(
    db: State<'_, Arc<Db>>,
    label_id: String,
    magnitude: f64,
    outcome_ref: Option<String>,
) -> Result<String, String> {
    let reward_id = Uuid::new_v4().to_string();
    sqlx::query(
        "INSERT INTO qualia_reward_events (reward_id, label_id, magnitude, outcome_ref, created_at)
         VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
    )
    .bind(&reward_id)
    .bind(&label_id)
    .bind(magnitude)
    .bind(outcome_ref.as_deref())
    .execute(&db.pool)
    .await
    .map_err(|e| e.to_string())?;
    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "kernel",
        None,
        None,
        json!({
            "event": "qualia_reward_recorded",
            "reward_id": reward_id,
            "label_id": label_id,
            "magnitude": magnitude,
        }),
    )
    .await;
    Ok(reward_id)
}

#[tauri::command]
pub async fn record_outcome(
    db: State<'_, Arc<Db>>,
    run_id: Option<String>,
    trace_id: Option<String>,
    candidate_id: Option<String>,
    target_type: Option<String>,
    verdict: String,
    confidence: Option<f64>,
    source: Option<String>,
    note: Option<String>,
    evidence_event_ids: Option<Vec<i64>>,
) -> Result<String, String> {
    let control_map = system_controls::load_control_map(&db).await;
    let feedback_mode = system_controls::mode_for("feedback_loop", &control_map);
    if system_controls::mode_is_off(&feedback_mode) || system_controls::mode_is_read_only(&feedback_mode) {
        let _ = system_log::log_event(
            &db.pool,
            None,
            "warn",
            "outcome",
            run_id.as_deref(),
            trace_id.as_deref(),
            json!({
                "event": "outcome_write_blocked",
                "reason": "feedback_loop_control",
                "mode": feedback_mode,
                "candidate_id": candidate_id,
                "verdict": verdict,
            }),
        )
        .await;
        return Err("feedback_loop_control".to_string());
    }
    if system_controls::mode_is_degraded(&feedback_mode) {
        let _ = system_log::log_event(
            &db.pool,
            None,
            "info",
            "outcome",
            run_id.as_deref(),
            trace_id.as_deref(),
            json!({
                "event": "outcome_write_degraded",
                "mode": feedback_mode,
                "candidate_id": candidate_id,
                "verdict": verdict,
            }),
        )
        .await;
    }
    let outcome_id = db
        .record_outcome_event(
            run_id.as_deref(),
            trace_id.as_deref(),
            candidate_id.as_deref(),
            target_type.as_deref().unwrap_or("decision_report"),
            &verdict,
            confidence.unwrap_or(0.5),
            source.as_deref().unwrap_or("operator"),
            note.as_deref(),
            evidence_event_ids.as_deref().unwrap_or(&[]),
        )
        .await?;
    Ok(outcome_id)
}

#[tauri::command]
pub async fn list_outcomes(
    db: State<'_, Arc<Db>>,
    limit: Option<i64>,
) -> Result<Vec<OutcomeEvent>, String> {
    db.list_outcome_events(limit.unwrap_or(50)).await
}

#[derive(Serialize)]
pub struct SubjectSnapshotView {
    pub snapshot_hash: String,
    pub tick_id: String,
    pub timestamp: String,
    pub run_id: Option<String>,
    pub subject_state_json: String,
}

#[derive(Serialize)]
pub struct GateDecisionView {
    pub decision_id: String,
    pub proposal_id: String,
    pub snapshot_hash: String,
    pub decision: String,
    pub evidence_refs_json: String,
    pub metrics_json: String,
    pub created_at: String,
}

#[derive(Serialize)]
pub struct IntrospectionEntryView {
    pub entry_id: String,
    pub snapshot_hash: String,
    pub numeric_payload_json: String,
    pub narrative: String,
    pub created_at: String,
}

#[derive(Serialize)]
pub struct AuditLogView {
    pub audit_id: String,
    pub target_id: String,
    pub snapshot_hash: String,
    pub discrepancy_score: f64,
    pub recommended_action: String,
    pub created_at: String,
}

#[derive(Serialize)]
pub struct ErrorEventView {
    pub error_event_id: String,
    pub residual_id: String,
    pub classification: String,
    pub status: String,
    pub created_at: String,
}

#[derive(Serialize)]
pub struct QualiaLabelView {
    pub label_id: String,
    pub event_id: String,
    pub snapshot_hash: String,
    pub tag: String,
    pub intensity: f64,
    pub created_at: String,
}

#[tauri::command]
pub async fn get_subject_snapshots(
    db: State<'_, Arc<Db>>,
    limit: Option<i64>,
) -> Result<Vec<SubjectSnapshotView>, String> {
    let limit = limit.unwrap_or(20).max(1);
    let rows = sqlx::query(
        "SELECT snapshot_hash, tick_id, timestamp, run_id, subject_state_json
         FROM subject_snapshots
         ORDER BY datetime(timestamp) DESC
         LIMIT ?",
    )
    .bind(limit)
    .fetch_all(&db.pool)
    .await
    .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        out.push(SubjectSnapshotView {
            snapshot_hash: row.try_get("snapshot_hash").unwrap_or_default(),
            tick_id: row.try_get("tick_id").unwrap_or_default(),
            timestamp: row.try_get("timestamp").unwrap_or_default(),
            run_id: row.try_get("run_id").ok(),
            subject_state_json: row.try_get("subject_state_json").unwrap_or_default(),
        });
    }
    Ok(out)
}

#[tauri::command]
pub async fn get_gate_decisions(
    db: State<'_, Arc<Db>>,
    limit: Option<i64>,
) -> Result<Vec<GateDecisionView>, String> {
    let limit = limit.unwrap_or(20).max(1);
    let rows = sqlx::query(
        "SELECT decision_id, proposal_id, snapshot_hash, decision, evidence_refs_json, metrics_json, created_at
         FROM gate_decisions
         ORDER BY datetime(created_at) DESC
         LIMIT ?",
    )
    .bind(limit)
    .fetch_all(&db.pool)
    .await
    .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        out.push(GateDecisionView {
            decision_id: row.try_get("decision_id").unwrap_or_default(),
            proposal_id: row.try_get("proposal_id").unwrap_or_default(),
            snapshot_hash: row.try_get("snapshot_hash").unwrap_or_default(),
            decision: row.try_get("decision").unwrap_or_default(),
            evidence_refs_json: row.try_get("evidence_refs_json").unwrap_or_default(),
            metrics_json: row.try_get("metrics_json").unwrap_or_default(),
            created_at: row.try_get("created_at").unwrap_or_default(),
        });
    }
    Ok(out)
}

#[tauri::command]
pub async fn get_context_tags(
    db: State<'_, Arc<Db>>,
    conversation_id: Option<String>,
    ttl_minutes: Option<i64>,
) -> Result<Vec<ContextTagEntry>, String> {
    let conversation_id = conversation_id.unwrap_or_else(|| "default".to_string());
    let ttl = ttl_minutes.unwrap_or(120);
    Ok(db.get_context_tags(&conversation_id, ttl).await)
}

#[tauri::command]
pub async fn get_user_intent_summary(
    db: State<'_, Arc<Db>>,
    conversation_id: Option<String>,
) -> Result<Option<UserIntentSummary>, String> {
    let conversation_id = conversation_id.unwrap_or_else(|| "default".to_string());
    Ok(db.get_user_intent_summary(&conversation_id).await)
}

#[tauri::command]
pub async fn update_user_intent_summary(
    db: State<'_, Arc<Db>>,
    conversation_id: Option<String>,
    summary: String,
    confirmed: bool,
    evidence_event_ids: Option<Vec<i64>>,
) -> Result<(), String> {
    let conversation_id = conversation_id.unwrap_or_else(|| "default".to_string());
    let evidence_ids = evidence_event_ids.unwrap_or_default();
    db.upsert_user_intent_summary(&conversation_id, summary.trim(), confirmed, &evidence_ids)
        .await?;
    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "context",
        None,
        None,
        json!({
            "event": "intent_summary_updated",
            "conversation_id": conversation_id,
            "confirmed": confirmed,
            "evidence_event_ids": evidence_ids,
        }),
    )
    .await;
    Ok(())
}

#[tauri::command]
pub async fn get_introspection_entries(
    db: State<'_, Arc<Db>>,
    limit: Option<i64>,
) -> Result<Vec<IntrospectionEntryView>, String> {
    let limit = limit.unwrap_or(20).max(1);
    let rows = sqlx::query(
        "SELECT entry_id, snapshot_hash, numeric_payload_json, narrative, created_at
         FROM introspection_entries
         ORDER BY datetime(created_at) DESC
         LIMIT ?",
    )
    .bind(limit)
    .fetch_all(&db.pool)
    .await
    .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        out.push(IntrospectionEntryView {
            entry_id: row.try_get("entry_id").unwrap_or_default(),
            snapshot_hash: row.try_get("snapshot_hash").unwrap_or_default(),
            numeric_payload_json: row.try_get("numeric_payload_json").unwrap_or_default(),
            narrative: row.try_get("narrative").unwrap_or_default(),
            created_at: row.try_get("created_at").unwrap_or_default(),
        });
    }
    Ok(out)
}

#[tauri::command]
pub async fn get_audit_log(
    db: State<'_, Arc<Db>>,
    limit: Option<i64>,
) -> Result<Vec<AuditLogView>, String> {
    let limit = limit.unwrap_or(20).max(1);
    let rows = sqlx::query(
        "SELECT audit_id, target_id, snapshot_hash, discrepancy_score, recommended_action, created_at
         FROM audit_log
         ORDER BY datetime(created_at) DESC
         LIMIT ?",
    )
    .bind(limit)
    .fetch_all(&db.pool)
    .await
    .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        out.push(AuditLogView {
            audit_id: row.try_get("audit_id").unwrap_or_default(),
            target_id: row.try_get("target_id").unwrap_or_default(),
            snapshot_hash: row.try_get("snapshot_hash").unwrap_or_default(),
            discrepancy_score: row.try_get::<f64, _>("discrepancy_score").unwrap_or(0.0),
            recommended_action: row.try_get("recommended_action").unwrap_or_default(),
            created_at: row.try_get("created_at").unwrap_or_default(),
        });
    }
    Ok(out)
}

#[tauri::command]
pub async fn get_error_events(
    db: State<'_, Arc<Db>>,
    limit: Option<i64>,
) -> Result<Vec<ErrorEventView>, String> {
    let limit = limit.unwrap_or(20).max(1);
    let rows = sqlx::query(
        "SELECT error_event_id, residual_id, classification, status, created_at
         FROM error_events
         ORDER BY datetime(created_at) DESC
         LIMIT ?",
    )
    .bind(limit)
    .fetch_all(&db.pool)
    .await
    .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        out.push(ErrorEventView {
            error_event_id: row.try_get("error_event_id").unwrap_or_default(),
            residual_id: row.try_get("residual_id").unwrap_or_default(),
            classification: row.try_get("classification").unwrap_or_default(),
            status: row.try_get("status").unwrap_or_default(),
            created_at: row.try_get("created_at").unwrap_or_default(),
        });
    }
    Ok(out)
}

#[tauri::command]
pub async fn get_qualia_labels(
    db: State<'_, Arc<Db>>,
    limit: Option<i64>,
) -> Result<Vec<QualiaLabelView>, String> {
    let limit = limit.unwrap_or(20).max(1);
    let rows = sqlx::query(
        "SELECT label_id, event_id, snapshot_hash, tag, intensity, created_at
         FROM qualia_labels
         ORDER BY datetime(created_at) DESC
         LIMIT ?",
    )
    .bind(limit)
    .fetch_all(&db.pool)
    .await
    .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        out.push(QualiaLabelView {
            label_id: row.try_get("label_id").unwrap_or_default(),
            event_id: row.try_get("event_id").unwrap_or_default(),
            snapshot_hash: row.try_get("snapshot_hash").unwrap_or_default(),
            tag: row.try_get("tag").unwrap_or_default(),
            intensity: row.try_get::<f64, _>("intensity").unwrap_or(0.0),
            created_at: row.try_get("created_at").unwrap_or_default(),
        });
    }
    Ok(out)
}

#[tauri::command]
pub async fn get_cognitive_readiness_report(
    db: State<'_, Arc<Db>>,
    conversation_id: Option<String>,
) -> Result<crate::models::CognitiveReadinessReport, String> {
    let conversation_id = conversation_id
        .as_deref()
        .filter(|id| !id.trim().is_empty())
        .unwrap_or("default");
    db.cognitive_readiness_report(conversation_id)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn run_cognitive_checks(
    db: State<'_, Arc<Db>>,
    conversation_id: Option<String>,
) -> Result<Vec<crate::models::CognitiveCheckResult>, String> {
    let conversation_id = conversation_id
        .as_deref()
        .filter(|id| !id.trim().is_empty())
        .unwrap_or("default");
    Ok(crate::core::cognitive_checks::run_cognitive_checks(&db, conversation_id).await)
}

#[tauri::command]
pub async fn get_system_logs(
    db: State<'_, Arc<Db>>,
    limit: Option<i64>,
    category: Option<String>,
    level: Option<String>,
    run_id: Option<String>,
) -> Result<Vec<crate::models::SystemLogEntry>, String> {
    db.list_system_logs(
        limit.unwrap_or(200),
        category.as_deref(),
        level.as_deref(),
        run_id.as_deref(),
    )
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_system_capabilities(
    db: State<'_, Arc<Db>>,
    conversation_id: Option<String>,
) -> Result<serde_json::Value, String> {
    let conversation_id = conversation_id
        .as_deref()
        .filter(|id| !id.trim().is_empty())
        .unwrap_or("default");
    let payload = crate::core::tool_registry::build_system_capabilities_payload(&db, conversation_id).await;
    Ok(payload)
}

#[tauri::command]
pub async fn get_evidence_lineage(
    db: State<'_, Arc<Db>>,
    limit: Option<i64>,
) -> Result<Vec<EvidenceLineageEntry>, String> {
    db.list_evidence_lineage(limit.unwrap_or(200))
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_system_controls(
    db: State<'_, Arc<Db>>,
) -> Result<Vec<SystemControlEntry>, String> {
    db.get_system_controls().await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn set_system_control(
    app: AppHandle,
    voice_manager: State<'_, VoiceManager>,
    db: State<'_, Arc<Db>>,
    subsystem_id: String,
    mode: String,
    value_json: Option<String>,
    reason: String,
    override_critical: bool,
    actor: Option<String>,
) -> Result<SystemControlEntry, String> {
    let settings = db.get_settings().await.map_err(|e| e.to_string())?;
    let cockpit_write_enabled = settings.cockpit_write_enabled.unwrap_or(false);
    if !cockpit_write_enabled {
        let previous_mode = db
            .get_system_controls()
            .await
            .ok()
            .and_then(|entries| {
                entries
                    .into_iter()
                    .find(|entry| entry.subsystem_id == subsystem_id)
                    .map(|entry| entry.mode)
            });
        let _ = db
            .insert_system_control_event(
                &subsystem_id,
                previous_mode,
                &system_controls::normalize_mode(&mode),
                value_json.clone(),
                actor.clone(),
                Some(reason.clone()),
                "rejected",
            )
            .await;
        let _ = system_log::log_event(
            &db.pool,
            None,
            "warn",
            "system",
            None,
            None,
            json!({
                "event": "system_control_rejected",
                "subsystem_id": subsystem_id,
                "mode": mode,
                "reason": reason,
                "actor": actor,
                "error": "cockpit_write_disabled",
            }),
        )
        .await;
        return Err("Cockpit write mode is disabled in Settings.".to_string());
    }
    let existing = db.get_system_controls().await.map_err(|e| e.to_string())?;
    let current_states = system_controls::map_from_entries(&existing);
    let request = system_controls::ControlChangeRequest {
        subsystem_id: subsystem_id.clone(),
        mode: mode.clone(),
        value_json: value_json.clone(),
        actor: actor.clone(),
        reason: Some(reason.clone()),
        override_critical,
    };
    if let Err(err) = system_controls::validate_change(&request, &current_states) {
        let previous_mode = current_states
            .get(&subsystem_id)
            .map(|state| state.mode.clone());
        let _ = db
            .insert_system_control_event(
                &subsystem_id,
                previous_mode,
                &system_controls::normalize_mode(&mode),
                value_json.clone(),
                actor.clone(),
                Some(reason.clone()),
                "rejected",
            )
            .await;
        let _ = system_log::log_event(
            &db.pool,
            None,
            "warn",
            "system",
            None,
            None,
            json!({
                "event": "system_control_rejected",
                "subsystem_id": subsystem_id,
                "mode": mode,
                "reason": reason,
                "actor": actor,
                "error": err,
            }),
        )
        .await;
        return Err(err);
    }

    let updated = db
        .set_system_control(
            &subsystem_id,
            &system_controls::normalize_mode(&mode),
            value_json.clone(),
            actor.clone(),
            Some(reason.clone()),
        )
        .await
        .map_err(|e| e.to_string())?;

    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "system",
        None,
        None,
        json!({
            "event": "system_control_changed",
            "subsystem_id": subsystem_id,
            "mode": updated.mode,
            "reason": reason,
            "actor": actor,
            "value_json": value_json,
        }),
    )
    .await;

    if subsystem_id == "voice_output" && system_controls::mode_is_off(&updated.mode) {
        voice_manager.stop(Some(&app));
    }

    Ok(updated)
}

#[tauri::command]
pub async fn get_system_control_events(
    db: State<'_, Arc<Db>>,
    limit: Option<i64>,
) -> Result<Vec<SystemControlEvent>, String> {
    db.list_system_control_events(limit.unwrap_or(200))
        .await
        .map_err(|e| e.to_string())
}

#[derive(Serialize)]
pub struct SystemHealthSnapshotView {
    pub snapshot_id: String,
    pub timestamp: String,
    pub run_id: Option<String>,
    pub trace_id: Option<String>,
    pub metrics: Value,
    pub subsystem_states: Value,
}

fn parse_snapshot(snapshot: SystemHealthSnapshot) -> SystemHealthSnapshotView {
    let metrics = serde_json::from_str::<Value>(&snapshot.metrics_json)
        .unwrap_or_else(|_| json!({ "raw": snapshot.metrics_json }));
    let subsystem_states = serde_json::from_str::<Value>(&snapshot.subsystem_states_json)
        .unwrap_or_else(|_| json!({ "raw": snapshot.subsystem_states_json }));
    SystemHealthSnapshotView {
        snapshot_id: snapshot.snapshot_id,
        timestamp: snapshot.timestamp,
        run_id: snapshot.run_id,
        trace_id: snapshot.trace_id,
        metrics,
        subsystem_states,
    }
}

#[tauri::command]
pub async fn get_system_health_snapshot(
    db: State<'_, Arc<Db>>,
    app: AppHandle,
) -> Result<SystemHealthSnapshotView, String> {
    let aggregator = system_health::HealthAggregator::new(db.inner().clone());
    let snapshot = aggregator
        .capture_snapshot(None, None, Some(&app))
        .await
        .map_err(|e| e.to_string())?;
    Ok(SystemHealthSnapshotView {
        snapshot_id: snapshot.snapshot_id,
        timestamp: snapshot.timestamp,
        run_id: snapshot.run_id,
        trace_id: snapshot.trace_id,
        metrics: snapshot.metrics,
        subsystem_states: snapshot.subsystem_states,
    })
}

#[tauri::command]
pub async fn get_system_health_history(
    db: State<'_, Arc<Db>>,
    limit: Option<i64>,
) -> Result<Vec<SystemHealthSnapshotView>, String> {
    let snapshots = db
        .list_system_health_snapshots(limit.unwrap_or(50))
        .await
        .map_err(|e| e.to_string())?;
    Ok(snapshots.into_iter().map(parse_snapshot).collect())
}

#[tauri::command]
pub async fn get_diagnostics_snapshot(
    db: State<'_, Arc<Db>>,
) -> Result<DiagnosticsSnapshot, String> {
    let system_controls = db.get_system_controls().await.map_err(|e| e.to_string())?;

    let kernel_cycle_payload: Option<String> = sqlx::query_scalar(
        "SELECT payload FROM system_logs
         WHERE json_extract(payload, '$.event') = 'kernel_cycle'
         ORDER BY datetime(timestamp) DESC, rowid DESC
         LIMIT 1",
    )
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();

    let mut evidence_map: HashMap<String, CandidateEvidenceSummary> = HashMap::new();
    if let Some(raw) = kernel_cycle_payload.as_deref() {
        if let Ok(payload) = serde_json::from_str::<Value>(raw) {
            if let Some(accepted) = payload.get("accepted").and_then(|v| v.as_array()) {
                for candidate in accepted {
                    let kind = candidate
                        .get("kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let payload = candidate.get("payload").cloned().unwrap_or_else(|| json!({}));
                    let mut evidence_ids = extract_id_list_from_value(&payload, "evidence_event_ids");
                    if evidence_ids.is_empty() {
                        evidence_ids = extract_id_list_from_value(&payload, "evidence_event_id");
                    }
                    let belief_ids = extract_id_list_from_value(&payload, "belief_ids");
                    let has_evidence = !evidence_ids.is_empty() || !belief_ids.is_empty();
                    let entry = evidence_map.entry(kind.clone()).or_insert(CandidateEvidenceSummary {
                        candidate_kind: kind,
                        candidate_count: 0,
                        with_evidence: 0,
                        evidence_event_ids: Vec::new(),
                        belief_ids: Vec::new(),
                    });
                    entry.candidate_count += 1;
                    if has_evidence {
                        entry.with_evidence += 1;
                    }
                    entry.evidence_event_ids.extend(evidence_ids);
                    entry.belief_ids.extend(belief_ids);
                }
            }
        }
    }
    for entry in evidence_map.values_mut() {
        entry.evidence_event_ids.sort();
        entry.evidence_event_ids.dedup();
        entry.belief_ids.sort();
        entry.belief_ids.dedup();
    }
    let mut candidate_evidence: Vec<CandidateEvidenceSummary> = evidence_map.into_values().collect();
    candidate_evidence.sort_by(|a, b| a.candidate_kind.cmp(&b.candidate_kind));

    let memory_write_blocked = load_system_logs_by_event(&db.pool, "memory_write_blocked", 20).await;
    let memory_pass_invalid_output =
        load_system_logs_by_event(&db.pool, "memory_pass_invalid_output", 20).await;

    Ok(DiagnosticsSnapshot {
        system_controls,
        candidate_evidence,
        memory_write_blocked,
        memory_pass_invalid_output,
    })
}

#[tauri::command]
pub async fn get_wave_status(db: State<'_, Arc<Db>>) -> Result<WaveStatus, String> {
    let now = Utc::now();
    let mut status = WaveStatus::default();

    if let Ok(Some(row)) = sqlx::query(
        "SELECT payload, timestamp FROM system_logs
         WHERE json_extract(payload, '$.event') = 'wave_projection'
         ORDER BY datetime(timestamp) DESC, rowid DESC
         LIMIT 1",
    )
    .fetch_optional(&db.pool)
    .await
    {
        let payload: String = row.get("payload");
        let timestamp: String = row.get("timestamp");
        if let Ok(value) = serde_json::from_str::<Value>(&payload) {
            status.coherence = value.get("coherence").and_then(|v| v.as_f64());
            status.turbulence = value.get("turbulence").and_then(|v| v.as_f64());
            status.drift = value.get("drift").and_then(|v| v.as_f64());
            status.dominance = value.get("dominance").and_then(|v| v.as_f64());
            status.fragmentation = value.get("fragmentation").and_then(|v| v.as_f64());
            status.total_energy = value.get("total_energy").and_then(|v| v.as_f64());
            status.band_energy = value.get("band_energy").cloned();
        }
        status.last_projection_at = Some(timestamp.clone());
        if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(&timestamp) {
            status.projection_age_seconds = Some(
                now.signed_duration_since(parsed.with_timezone(&Utc)).num_seconds().max(0),
            );
        }
    }

    if let Ok(Some(row)) = sqlx::query(
        "SELECT timestamp FROM system_logs
         WHERE json_extract(payload, '$.event') = 'wave_contribution'
         ORDER BY datetime(timestamp) DESC, rowid DESC
         LIMIT 1",
    )
    .fetch_optional(&db.pool)
    .await
    {
        let timestamp: String = row.get("timestamp");
        status.last_contribution_at = Some(timestamp.clone());
        if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(&timestamp) {
            status.contribution_age_seconds = Some(
                now.signed_duration_since(parsed.with_timezone(&Utc)).num_seconds().max(0),
            );
        }
    }

    Ok(status)
}

#[tauri::command]
pub async fn get_self_model(db: State<'_, Arc<Db>>) -> Result<crate::models::SelfModel, String> {
    db.get_self_model().await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_parameter_registry(
    db: State<'_, Arc<Db>>,
    profile_name: Option<String>,
) -> Result<crate::models::ParameterRegistry, String> {
    let profile = profile_name.unwrap_or_else(|| "default".to_string());
    db.get_parameter_registry(&profile)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "Parameter registry not found".to_string())
}

#[tauri::command]
pub async fn update_parameter_registry(
    db: State<'_, Arc<Db>>,
    profile_name: Option<String>,
    payload_json: String,
) -> Result<crate::models::ParameterRegistry, String> {
    let profile = profile_name.unwrap_or_else(|| "default".to_string());
    db.upsert_parameter_registry(&profile, &payload_json)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn update_settings(db: State<'_, Arc<Db>>, settings: Settings) -> Result<(), String> {
    db.update_settings(settings).await.map_err(|e| e.to_string())?;
    let control_map = system_controls::load_control_map(&db).await;
    let prompt_mode = system_controls::mode_for("prompt_loader", &control_map);
    if !system_controls::mode_is_off(&prompt_mode)
        && !system_controls::mode_is_degraded(&prompt_mode)
    {
        prompt_loader::reload_prompts().map_err(|e| format!("Prompt reload failed: {}", e))?;
    }
    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "system",
        None,
        None,
        json!({
            "event": "settings_updated",
        }),
    )
    .await;
    Ok(())
}

#[tauri::command]
pub async fn get_phi_consent_scope(
    db: State<'_, Arc<Db>>,
    conversation_id: String,
) -> Result<Option<bool>, String> {
    db.get_phi_consent_scope(&conversation_id)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn set_phi_consent_scope(
    db: State<'_, Arc<Db>>,
    conversation_id: String,
    enabled: bool,
) -> Result<(), String> {
    db.set_phi_consent_scope(&conversation_id, enabled)
        .await
        .map_err(|e| e.to_string())?;
    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "system",
        None,
        None,
        json!({
            "event": "phi_consent_scope_updated",
            "conversation_id": conversation_id,
            "enabled": enabled,
        }),
    )
    .await;
    Ok(())
}

#[derive(Serialize)]
pub struct ThemeList {
    pub themes: Vec<String>,
    pub dir: String,
}

#[derive(Serialize)]
pub struct PendingPromptView {
    pub id: String,
    pub prompt: String,
    pub source: String,
    pub created_at: String,
    pub skip_count: i64,
    pub attempt_count: i64,
    pub last_asked_at: Option<String>,
    pub expires_at: Option<String>,
    pub auto_surface: bool,
    pub intent_kind: Option<String>,
    pub bridge_id: Option<String>,
    pub anchor_message_id: Option<String>,
    pub anchor_hash: Option<String>,
    pub anchor_created_at: Option<String>,
    pub anchor_role: Option<String>,
}

fn sanitize_theme_name(name: &str) -> Result<&str, String> {
    if name.contains("..") || name.contains('/') || name.contains('\\') {
        return Err("Invalid theme name".to_string());
    }
    let file_name = Path::new(name)
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "Invalid theme name".to_string())?;
    if file_name.is_empty() {
        return Err("Invalid theme name".to_string());
    }
    Ok(file_name)
}

#[tauri::command]
pub async fn list_themes(app: AppHandle) -> Result<ThemeList, String> {
    let app_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let theme_dir = app_dir.join("themes");
    if !theme_dir.exists() {
        fs::create_dir_all(&theme_dir).map_err(|e| e.to_string())?;
    }

    let mut themes = Vec::new();
    if let Ok(entries) = fs::read_dir(&theme_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    if ext.eq_ignore_ascii_case("css") {
                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                            themes.push(stem.to_string());
                        }
                    }
                }
            }
        }
    }

    themes.sort();
    Ok(ThemeList {
        themes,
        dir: theme_dir.to_string_lossy().to_string(),
    })
}

#[tauri::command]
pub async fn read_theme_file(app: AppHandle, name: String) -> Result<String, String> {
    let safe = sanitize_theme_name(&name)?;
    let app_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let theme_dir = app_dir.join("themes");
    if !theme_dir.exists() {
        fs::create_dir_all(&theme_dir).map_err(|e| e.to_string())?;
    }
    let filename = if safe.to_lowercase().ends_with(".css") {
        safe.to_string()
    } else {
        format!("{}.css", safe)
    };
    let path = theme_dir.join(filename);
    fs::read_to_string(&path).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn send_message(chat: State<'_, Arc<ChatManager>>, content: String) -> Result<(), String> {
    chat.send_message(content).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn list_pending_prompts(
    db: State<'_, Arc<Db>>,
    limit: Option<i64>,
) -> Result<Vec<PendingPromptView>, String> {
    let limit = limit.unwrap_or(8);
    let rows = db
        .list_pending_prompts("default", limit)
        .await
        .map_err(|e| e.to_string())?;
    Ok(rows
        .into_iter()
        .map(|(id, prompt, source, created_at, skip_count, auto_surface, intent_kind, bridge_id, attempt_count, last_asked_at, expires_at, anchor_message_id, anchor_hash, anchor_created_at, anchor_role)| PendingPromptView {
            id,
            prompt,
            source,
            created_at,
            skip_count,
            attempt_count,
            last_asked_at,
            expires_at,
            auto_surface,
            intent_kind,
            bridge_id,
            anchor_message_id,
            anchor_hash,
            anchor_created_at,
            anchor_role,
        })
        .collect())
}

#[tauri::command]
pub async fn get_pending_prompt_count(
    db: State<'_, Arc<Db>>,
) -> Result<i64, String> {
    db.count_pending_prompts("default")
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn dismiss_pending_prompt(
    chat: State<'_, Arc<ChatManager>>,
    prompt_id: String,
) -> Result<(), String> {
    let deleted = chat
        .db
        .delete_pending_prompt(&prompt_id)
        .await
        .map_err(|e| e.to_string())?;
    if deleted == 0 {
        let _ = system_log::log_event(
            &chat.db.pool,
            Some(&chat.app_handle),
            "warn",
            "chat",
            None,
            None,
            json!({
                "event": "pending_prompt_delete_failed",
                "reason": "not_found",
                "prompt_id": prompt_id,
            }),
        )
        .await;
    }
    if let Ok(count) = chat.db.count_pending_prompts("default").await {
        let _ = chat.app_handle.emit("pending_prompt_count", count as usize);
    }
    Ok(())
}

#[tauri::command]
pub async fn rephrase_pending_prompt(
    chat: State<'_, Arc<ChatManager>>,
    prompt_id: String,
    prompt: String,
) -> Result<(), String> {
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return Err("Prompt cannot be empty".to_string());
    }
    chat.db
        .update_pending_prompt(&prompt_id, trimmed)
        .await
        .map_err(|e| e.to_string())?;
    if let Ok(count) = chat.db.count_pending_prompts("default").await {
        let _ = chat.app_handle.emit("pending_prompt_count", count as usize);
    }
    Ok(())
}

#[tauri::command]
pub async fn send_pending_prompt(
    chat: State<'_, Arc<ChatManager>>,
    prompt_id: String,
) -> Result<(), String> {
    let Some((pending_id, _prompt, source, conversation_id, _created_at, _skip_count, _auto_surface, _intent_kind, _bridge_id, _attempt_count, _last_asked_at, _expires_at, _anchor_message_id, _anchor_hash, _anchor_created_at, _anchor_role)) = chat
        .db
        .get_pending_prompt_by_id(&prompt_id)
        .await
        .map_err(|e| e.to_string())?
    else {
        return Err("Pending prompt not found".to_string());
    };

    if let Ok(Some(active_run_id)) = chat.db.get_active_foreground_run(&conversation_id).await {
        let payload = json!({ "prompt_id": pending_id }).to_string();
        let _ = chat
            .db
            .enqueue_deferred_emit(&conversation_id, "pending_prompt_send", &payload, Some(source.as_str()))
            .await
            .map_err(|e| e.to_string())?;
        let _ = system_log::log_event(
            &chat.db.pool,
            Some(&chat.app_handle),
            "info",
            "chat",
            None,
            None,
            json!({
                "event": "pending_prompt_deferred",
                "prompt_id": pending_id,
                "active_run_id": active_run_id,
                "source": source,
            }),
        )
        .await;
        if let Ok(count) = chat.db.count_pending_prompts("default").await {
            let _ = chat.app_handle.emit("pending_prompt_count", count as usize);
        }
        return Ok(());
    }

    crate::core::deliver_pending_prompt(chat.db.as_ref(), &chat.app_handle, &pending_id).await?;
    Ok(())
}

#[tauri::command]
pub async fn abort_generation(
    chat: State<'_, Arc<ChatManager>>,
    run_id: Option<String>,
    source: Option<String>,
) -> Result<(), String> {
    let source_label = source.as_deref().unwrap_or("unknown");
    let active_run_id = if run_id.is_none() {
        chat.db.get_active_foreground_run("default").await.ok().flatten()
    } else {
        None
    };
    let _ = system_log::log_event(
        &chat.db.pool,
        Some(&chat.app_handle),
        "info",
        "chat",
        None,
        None,
        json!({
            "event": "abort_generation_requested",
            "source": source_label,
            "run_id": run_id,
            "active_run_id": active_run_id,
        }),
    )
    .await;
    chat.abort(run_id.as_deref(), Some(source_label)).await;
    Ok(())
}

#[tauri::command]
pub async fn submit_clarification(
    chat: State<'_, Arc<ChatManager>>,
    answer: String,
    original_input: String,
    original_run_id: Option<String>,
) -> Result<(), String> {
    chat.submit_clarification(answer, original_input, original_run_id)
        .await
        .map_err(|e| e.to_string())
}


#[tauri::command]
pub async fn test_connection(db: State<'_, Arc<Db>>, app: tauri::AppHandle, url: String, api_key: Option<String>) -> Result<String, String> {
    let client = ModelClient::new(db.pool.clone(), app);
    client.test_connection(&url, api_key.as_deref()).await
}

#[tauri::command]
pub async fn clear_history(chat: State<'_, Arc<ChatManager>>) -> Result<(), String> {
    chat.clear_history().await.map_err(|e| e.to_string())?;
    let _ = system_log::log_event(
        &chat.db.pool,
        Some(&chat.app_handle),
        "info",
        "system",
        None,
        None,
        json!({
            "event": "history_cleared",
        }),
    )
    .await;
    Ok(())
}

#[tauri::command]
pub async fn reset_conversation_data(
    db: State<'_, Arc<Db>>,
    chat: State<'_, Arc<ChatManager>>,
    app: tauri::AppHandle
) -> Result<(), String> {
    let _ = system_log::log_event(
        &db.pool,
        Some(&app),
        "info",
        "chat",
        None,
        None,
        json!({
            "event": "abort_generation_requested",
            "source": "reset_conversation_data",
            "run_id": None::<String>,
            "active_run_id": chat.db.get_active_foreground_run("default").await.ok().flatten(),
        }),
    )
    .await;
    chat.abort(None, Some("reset_conversation_data")).await;
    let conversation_ids = db
        .inner()
        .list_conversation_ids(None)
        .await
        .unwrap_or_else(|_| vec!["default".to_string()]);
    for conversation_id in conversation_ids {
        let _ = crate::core::rolling_summary::archive_rolling_summary(
            db.inner().clone(),
            &conversation_id,
            "kernel",
            "summary_archive",
        )
        .await;
    }
    db.reset_conversation_data().await.map_err(|e| e.to_string())?;
    use tauri::Emitter;
    let _ = app.emit("message_updated", ());
    let _ = app.emit("memory_updated", ()); // Fix: Refresh Memory Panel
    let _ = system_log::log_event(
        &db.pool,
        Some(&app),
        "info",
        "system",
        None,
        None,
        json!({
            "event": "conversation_reset",
        }),
    )
    .await;
    Ok(())
}

#[tauri::command]
pub async fn reset_all_data(
    db: State<'_, Arc<Db>>,
    chat: State<'_, Arc<ChatManager>>,
    app: tauri::AppHandle
) -> Result<(), String> {
    use tauri::Emitter;
    let _ = system_log::log_event(
        &db.pool,
        Some(&app),
        "info",
        "chat",
        None,
        None,
        json!({
            "event": "abort_generation_requested",
            "source": "reset_all_data",
            "run_id": None::<String>,
            "active_run_id": chat.db.get_active_foreground_run("default").await.ok().flatten(),
        }),
    )
    .await;
    chat.abort(None, Some("reset_all_data")).await;
    db.reset_all_data().await.map_err(|e| e.to_string())?;
    let _ = app.emit("message_updated", ());
    let _ = app.emit("memory_updated", ());
    let _ = system_log::log_event(
        &db.pool,
        Some(&app),
        "info",
        "system",
        None,
        None,
        json!({
            "event": "system_reset",
        }),
    )
    .await;
    Ok(())
}

#[tauri::command]
pub async fn get_episodic_events(
    db: State<'_, Arc<Db>>,
    conversation_id: Option<String>,
    limit: Option<i64>,
) -> Result<Vec<crate::models::EpisodicEvent>, String> {
    let limit = limit.unwrap_or(50);
    db.list_episodic_events(conversation_id.as_deref(), limit)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn search_episodic_events(
    db: State<'_, Arc<Db>>,
    query: Option<String>,
    conversation_id: Option<String>,
    run_id: Option<String>,
    event_type: Option<String>,
    start_time: Option<String>,
    end_time: Option<String>,
    limit: Option<i64>,
) -> Result<Vec<crate::models::EpisodicEvent>, String> {
    let limit = limit.unwrap_or(50);
    db.search_episodic_events(
        query.as_deref(),
        conversation_id.as_deref(),
        run_id.as_deref(),
        event_type.as_deref(),
        start_time.as_deref(),
        end_time.as_deref(),
        limit,
    )
    .await
    .map_err(|e| e.to_string())
}
#[tauri::command]
pub fn normalize_url(url: String) -> Result<(String, bool), String> {
    ModelClient::normalize_url(&url)
}


#[tauri::command]
pub async fn trigger_reminder_response(
    chat_manager: State<'_, Arc<ChatManager>>,
    _reminder_id: String,
    content: String,
) -> Result<(), String> {
    let trimmed = content.trim();
    let instructions = format!(
        "Reminder triggered. Draft a concise, friendly reminder. Apologize for any interruption. Reminder: {}",
        trimmed
    );
    chat_manager
        .trigger_proactive_event(instructions)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn create_reminder(db: State<'_, Arc<Db>>, content: String, due_in: String, reminder_type: String) -> Result<String, String> {
    let reminder_id = reminder_blocks::create_reminder(&db.pool, &content, &due_in, &reminder_type).await?;
    let _ = episodic::emit_episodic_event(
        &db.pool,
        "reminder_created",
        json!({ "status": "created", "summary_snippet": content }),
        None,
        None,
        Some("default"),
        None,
        "user",
        Some(&reminder_id),
        None,
        None,
    )
    .await;
    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "system",
        None,
        None,
        json!({
            "event": "reminder_created",
            "reminder_id": reminder_id.clone(),
            "due_in": due_in,
            "reminder_type": reminder_type,
        }),
    )
    .await;
    Ok(reminder_id)
}
