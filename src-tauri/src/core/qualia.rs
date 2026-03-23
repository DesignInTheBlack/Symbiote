use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::Row;
use uuid::Uuid;

use crate::core::workspace::WorkspaceState;
use crate::core::system_controls;
use crate::core::system_log;
use crate::db::Db;
use crate::core::cognitive_wave::{AmplitudeBounds, DecayProfile, WaveBand, WaveContributionInput};
use crate::core::model_client::{ChatCompletionRequest, ChatMessage, ModelClient};
use crate::core::kernel::parse_json_object_with_repair;
use num_complex::Complex32;
use tauri::AppHandle;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualiaState {
    pub timestamp: String,
    pub dominant_tag: Option<String>,
    pub dominant_intensity: f64,
    pub recent_labels: Vec<QualiaLabelSummary>,
    pub last_reward: Option<f64>,
    pub predicted_tag: Option<String>,
    pub prediction_confidence: f64,
    pub matched_workspace_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualiaLabelSummary {
    pub tag: String,
    pub intensity: f64,
    pub created_at: String,
}

fn heuristic_qualia_tag(message: &str) -> (String, f64, String) {
    let trimmed = message.trim();
    if trimmed.is_empty() {
        return ("neutral".to_string(), 0.2, "empty_message".to_string());
    }
    let lower = trimmed.to_lowercase();
    if lower.contains("urgent") || lower.contains("asap") || lower.contains("immediately") {
        return ("urgent".to_string(), 0.5, "urgency_terms".to_string());
    }
    if lower.contains("however")
        || lower.contains("but ")
        || lower.contains("not ")
        || lower.contains("doubt")
        || lower.contains("skeptic")
    {
        return ("skeptical".to_string(), 0.45, "contrast_or_negation".to_string());
    }
    if trimmed.contains('?') {
        return ("curious".to_string(), 0.45, "question_marks".to_string());
    }
    if trimmed.len() > 280
        || lower.contains("therefore")
        || lower.contains("overall")
        || lower.contains("in summary")
    {
        return ("informative".to_string(), 0.35, "expository".to_string());
    }
    if lower.contains("calm") || lower.contains("steady") || lower.contains("relaxed") {
        return ("calm".to_string(), 0.3, "calm_terms".to_string());
    }
    ("neutral".to_string(), 0.2, "default".to_string())
}

async fn infer_qualia_tag(
    db: &Db,
    app_handle: Option<&AppHandle>,
    message: &str,
    run_id: Option<&str>,
    message_id: &str,
) -> (String, f64, String, bool) {
    let trimmed = message.trim();
    if trimmed.is_empty() {
        let (tag, intensity, reason) = heuristic_qualia_tag(trimmed);
        return (tag, intensity, reason, true);
    }
    let Some(app_handle) = app_handle else {
        let (tag, intensity, reason) = heuristic_qualia_tag(trimmed);
        return (tag, intensity, reason, true);
    };

    let settings = match db.get_settings().await {
        Ok(settings) => settings,
        Err(_) => {
            let (tag, intensity, reason) = heuristic_qualia_tag(trimmed);
            return (tag, intensity, reason, true);
        }
    };
    let model_id = settings
        .active_model_id
        .clone()
        .unwrap_or_else(|| "default".to_string());

    let allowed = ["curious", "skeptical", "informative", "calm", "urgent", "neutral"];
    let system_prompt = "Assign a single qualia tag and intensity for the assistant message. Output ONLY JSON.\n\nSchema:\n{\n  \"tag\": \"curious|skeptical|informative|calm|urgent|neutral\",\n  \"intensity\": 0.0-1.0,\n  \"reason\": \"short reason\"\n}\n\nRules:\n- Pick exactly one tag from the allowed list.\n- Intensity between 0.0 and 1.0.\n- Keep reason under 8 words.\n- Do not include extra keys.";
    let user_prompt = format!(
        "Assistant message:\n{}\n\nAllowed tags: {}",
        trimmed.chars().take(800).collect::<String>(),
        allowed.join(", ")
    );
    let request = ChatCompletionRequest {
        model: model_id,
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_prompt,
            },
        ],
        stream: false,
        temperature: Some(0.2),
        top_p: None,
        max_tokens: Some(80),
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
        run_id: run_id.map(|v| v.to_string()),
        request_label: Some("qualia_auto_label".to_string()),
    };

    let client = ModelClient::new(db.pool.clone(), app_handle.clone());
    let response = client
        .chat(&settings.api_base_url, settings.api_key.as_deref(), &request)
        .await;
    let (content, _) = match response {
        Ok(payload) => payload,
        Err(_) => {
            let (tag, intensity, reason) = heuristic_qualia_tag(trimmed);
            return (tag, intensity, reason, true);
        }
    };
    let (value_opt, _) = parse_json_object_with_repair(&content);
    let Some(value) = value_opt else {
        let (tag, intensity, reason) = heuristic_qualia_tag(trimmed);
        return (tag, intensity, reason, true);
    };
    let tag = value
        .get("tag")
        .and_then(|v| v.as_str())
        .unwrap_or("neutral")
        .to_string();
    let intensity = value
        .get("intensity")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.2)
        .clamp(0.0, 1.0);
    let reason = value
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("model")
        .to_string();

    if !allowed.iter().any(|allowed_tag| *allowed_tag == tag) {
        let (fallback_tag, fallback_intensity, fallback_reason) = heuristic_qualia_tag(trimmed);
        let _ = system_log::log_event(
            &db.pool,
            None,
            "warn",
            "kernel",
            run_id,
            None,
            json!({
                "event": "qualia_auto_label_fallback",
                "message_id": message_id,
                "reason": "invalid_tag",
                "tag": tag,
            }),
        )
        .await;
        return (fallback_tag, fallback_intensity, fallback_reason, true);
    }

    (tag, intensity, reason, false)
}

pub async fn compute_qualia_state(
    db: &Db,
    workspace: Option<&WorkspaceState>,
) -> Result<QualiaState, String> {
    let mode: Option<String> = sqlx::query_scalar(
        "SELECT mode FROM system_controls WHERE subsystem_id = ?",
    )
    .bind("qualia_loop")
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let qualia_mode = mode.unwrap_or_else(|| {
        system_controls::default_mode_for("qualia_loop")
            .unwrap_or("normal")
            .to_string()
    });
    if system_controls::mode_is_off(&qualia_mode) {
        return Ok(QualiaState {
            timestamp: chrono::Utc::now().to_rfc3339(),
            dominant_tag: None,
            dominant_intensity: 0.0,
            recent_labels: Vec::new(),
            last_reward: None,
            predicted_tag: None,
            prediction_confidence: 0.0,
            matched_workspace_refs: Vec::new(),
        });
    }
    let rows = sqlx::query(
        "SELECT tag, intensity, created_at, context_json
         FROM qualia_labels
         ORDER BY datetime(created_at) DESC
         LIMIT 10",
    )
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();

    let mut recent_labels = Vec::new();
    let mut accum: std::collections::HashMap<String, (f64, i64)> = std::collections::HashMap::new();
    let broadcast_refs = workspace
        .map(|w| w.broadcast_refs.clone())
        .unwrap_or_default();
    let mut predicted_tag: Option<String> = None;
    let mut prediction_confidence = 0.0;
    let mut matched_workspace_refs: Vec<String> = Vec::new();
    let mut best_score = 0.0;

    for row in rows {
        let tag: String = row.try_get("tag").unwrap_or_default();
        let intensity: f64 = row.try_get("intensity").unwrap_or(0.0);
        let created_at: String = row.try_get("created_at").unwrap_or_default();
        let context_json: Option<String> = row.try_get("context_json").ok();
        recent_labels.push(QualiaLabelSummary { tag: tag.clone(), intensity, created_at });
        let entry = accum.entry(tag.clone()).or_insert((0.0, 0));
        entry.0 += intensity;
        entry.1 += 1;

        if !broadcast_refs.is_empty() {
            let mut label_refs: Vec<String> = Vec::new();
            if let Some(raw) = context_json.as_deref() {
                if let Ok(value) = serde_json::from_str::<Value>(raw) {
                    if let Some(arr) = value.get("workspace_refs").and_then(|v| v.as_array()) {
                        label_refs = arr
                            .iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect();
                    } else if let Some(arr) = value.get("broadcast_refs").and_then(|v| v.as_array()) {
                        label_refs = arr
                            .iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect();
                    }
                }
            }

            if !label_refs.is_empty() {
                let mut overlap = 0usize;
                let mut overlaps = Vec::new();
                for label_ref in label_refs.iter() {
                    if broadcast_refs.iter().any(|r| r == label_ref) {
                        overlap += 1;
                        overlaps.push(label_ref.clone());
                    }
                }
                let similarity = if broadcast_refs.is_empty() {
                    0.0
                } else {
                    overlap as f64 / broadcast_refs.len().max(1) as f64
                };
                let score = intensity * (1.0 + similarity);
                if score > best_score {
                    best_score = score;
                    predicted_tag = Some(tag.clone());
                    matched_workspace_refs = overlaps;
                    prediction_confidence = (score / 1.5).clamp(0.0, 1.0);
                }
            }
        }
    }

    let mut dominant_tag: Option<String> = None;
    let mut dominant_intensity = 0.0;
    if !system_controls::mode_is_degraded(&qualia_mode) {
        for (tag, (sum, count)) in accum.iter() {
            let avg = if *count > 0 { sum / *count as f64 } else { 0.0 };
            if avg > dominant_intensity {
                dominant_intensity = avg;
                dominant_tag = Some(tag.clone());
            }
        }
    }

    if system_controls::mode_is_degraded(&qualia_mode) {
        predicted_tag = None;
        prediction_confidence = 0.0;
    }

    let last_reward: Option<f64> = sqlx::query_scalar(
        "SELECT magnitude FROM qualia_reward_events ORDER BY datetime(created_at) DESC LIMIT 1",
    )
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();

    Ok(QualiaState {
        timestamp: chrono::Utc::now().to_rfc3339(),
        dominant_tag,
        dominant_intensity,
        recent_labels,
        last_reward,
        predicted_tag,
        prediction_confidence,
        matched_workspace_refs,
    })
}

pub fn wave_contribution(state: &QualiaState) -> Option<WaveContributionInput> {
    let intensity = state.dominant_intensity as f32;
    let reward = state.last_reward.unwrap_or(0.0) as f32;
    let confidence = state.prediction_confidence as f32;
    if intensity <= 0.0 && reward.abs() <= 0.01 && confidence <= 0.01 {
        return None;
    }
    let mut coeffs = Vec::new();
    coeffs.push(Complex32::new(intensity, reward));
    coeffs.push(Complex32::new(confidence, intensity));
    coeffs.push(Complex32::new(reward, confidence));
    let amplitude = (0.15 + intensity.abs()).clamp(0.05, 0.9);
    Some(WaveContributionInput {
        source: "qualia",
        band: WaveBand::Qualia,
        coeffs,
        amplitude,
        amplitude_bounds: AmplitudeBounds::new(0.05, 0.9),
        decay_profile: DecayProfile::for_band(WaveBand::Qualia),
    })
}

pub async fn maybe_auto_label_for_recent_message(
    db: &Db,
    app_handle: Option<&AppHandle>,
    conversation_id: &str,
    run_id: Option<&str>,
) -> Result<Option<String>, String> {
    let latest_id: Option<String> = sqlx::query_scalar(
        "SELECT message_id FROM messages
         WHERE conversation_id = ? AND role = 'assistant' AND status = 'complete'
         ORDER BY datetime(created_at) DESC
         LIMIT 1",
    )
    .bind(conversation_id)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let Some(message_id) = latest_id else {
        return Ok(None);
    };
    maybe_auto_label_for_message(db, app_handle, conversation_id, &message_id, run_id, false).await
}

pub async fn maybe_auto_label_for_message(
    db: &Db,
    app_handle: Option<&AppHandle>,
    conversation_id: &str,
    message_id: &str,
    run_id: Option<&str>,
    allow_non_latest: bool,
) -> Result<Option<String>, String> {
    let control_map = system_controls::load_control_map(db).await;
    let qualia_loop_mode = system_controls::mode_for("qualia_loop", &control_map);
    if system_controls::mode_is_off(&qualia_loop_mode) {
        log_qualia_auto_skip(db, app_handle, conversation_id, Some(message_id), "qualia_loop_off", run_id).await;
        return Ok(None);
    }
    let auto_mode = system_controls::mode_for("qualia_auto", &control_map);
    if system_controls::mode_is_off(&auto_mode) {
        log_qualia_auto_skip(db, app_handle, conversation_id, Some(message_id), "qualia_auto_off", run_id).await;
        return Ok(None);
    }

    let latest_id: Option<String> = sqlx::query_scalar(
        "SELECT message_id FROM messages
         WHERE conversation_id = ? AND role = 'assistant' AND status = 'complete'
         ORDER BY datetime(created_at) DESC
         LIMIT 1",
    )
    .bind(conversation_id)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    if latest_id.as_deref() != Some(message_id) {
        if !allow_non_latest {
            log_qualia_auto_skip(db, app_handle, conversation_id, Some(message_id), "not_latest", run_id).await;
            return Ok(None);
        }
    }
    if latest_id.is_none() {
        log_qualia_auto_skip(db, app_handle, conversation_id, Some(message_id), "no_latest_message", run_id).await;
        return Ok(None);
    }

    let existing: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM qualia_labels WHERE event_id = ?",
    )
    .bind(message_id)
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    if existing > 0 {
        log_qualia_auto_skip(db, app_handle, conversation_id, Some(message_id), "already_labeled", run_id).await;
        return Ok(None);
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
    .ok()
    .flatten();
    let Some(snapshot_hash) = snapshot_hash else {
        log_qualia_auto_skip(db, app_handle, conversation_id, Some(message_id), "missing_subject_snapshot", run_id).await;
        return Ok(None);
    };

    let message_text: String = sqlx::query_scalar(
        "SELECT content FROM messages WHERE message_id = ? LIMIT 1",
    )
    .bind(message_id)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or_default();

    let label_id = Uuid::new_v4().to_string();
    let (tag, intensity, reason, fallback_used) =
        infer_qualia_tag(db, app_handle, &message_text, run_id, message_id).await;
    let context_json = json!({
        "auto": true,
        "source": "qualia_auto",
        "conversation_id": conversation_id,
        "message_id": message_id,
        "run_id": run_id,
        "snapshot_hash": snapshot_hash,
        "reason": reason,
        "fallback_used": fallback_used,
    })
    .to_string();

    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "kernel",
        run_id,
        None,
        json!({
            "event": "qualia_auto_label_inferred",
            "label_id": label_id,
            "event_id": message_id,
            "tag": tag,
            "intensity": intensity,
            "fallback_used": fallback_used,
            "reason": reason,
            "snapshot_hash": snapshot_hash,
        }),
    )
    .await;

    sqlx::query(
        "INSERT INTO qualia_labels (label_id, event_id, snapshot_hash, tag, intensity, context_json, created_at)
         VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
    )
    .bind(&label_id)
    .bind(message_id)
    .bind(&snapshot_hash)
    .bind(&tag)
    .bind(intensity)
    .bind(&context_json)
    .execute(&db.pool)
    .await
    .map_err(|e| e.to_string())?;

    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "kernel",
        run_id,
        None,
        json!({
            "event": "qualia_auto_label_recorded",
            "label_id": label_id,
            "event_id": message_id,
            "tag": tag,
            "intensity": intensity,
            "snapshot_hash": snapshot_hash,
        }),
    )
    .await;

    Ok(Some(label_id))
}

async fn log_qualia_auto_skip(
    db: &Db,
    app_handle: Option<&AppHandle>,
    conversation_id: &str,
    message_id: Option<&str>,
    reason: &str,
    run_id: Option<&str>,
) {
    let _ = system_log::log_event(
        &db.pool,
        app_handle,
        "info",
        "qualia",
        run_id,
        None,
        json!({
            "event": "qualia_auto_label_skipped",
            "conversation_id": conversation_id,
            "message_id": message_id,
            "reason": reason,
        }),
    )
    .await;
}
