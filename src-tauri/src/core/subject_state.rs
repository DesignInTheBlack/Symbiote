use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use chrono::Utc;
use sqlx::Row;

use crate::db::Db;
use crate::core::kernel::KernelState;
use crate::core::{attention_model, attention_schema, internal_state_map, organism, qualia, workspace, system_controls, system_log, world_model};
use crate::core::cognitive_wave;
use crate::core::self_model_controller;
use crate::models::{ControllerState, SelfModel};

pub const SUBJECT_SNAPSHOT_VERSION: &str = "1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectState {
    pub workspace: workspace::WorkspaceState,
    pub organism: organism::OrganismState,
    pub self_model: SelfModelState,
    pub error_state: ErrorState,
    pub attention: attention_model::AttentionModel,
    #[serde(default)]
    pub attention_schema: crate::models::AttentionSchemaState,
    pub qualia: qualia::QualiaState,
    #[serde(default)]
    pub world_model: world_model::WorldModelSnapshot,
    #[serde(default)]
    pub world_model_delta: WorldModelDelta,
    #[serde(default)]
    pub monologue_updates: Vec<MonologueStateUpdate>,
    #[serde(default)]
    pub plan_hash: Option<String>,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorldModelDelta {
    #[serde(default)]
    pub added: Vec<WorldModelDeltaItem>,
    #[serde(default)]
    pub removed: Vec<WorldModelDeltaItem>,
    #[serde(default)]
    pub changed: Vec<WorldModelDeltaItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelDeltaItem {
    pub belief_id: i64,
    #[serde(default)]
    pub topic_key: Option<String>,
    #[serde(default)]
    pub prev_state: Option<String>,
    #[serde(default)]
    pub new_state: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorState {
    pub recent_residuals: Vec<ResidualSummary>,
    pub open_error_ids: Vec<String>,
    pub open_error_count: usize,
    pub pattern_flags: Vec<String>,
    pub diagnosis_flags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualSummary {
    pub residual_id: String,
    pub prediction_id: String,
    pub normalized_residual: f64,
    pub salience_score: f64,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModelState {
    pub self_identity_claim_id: Option<String>,
    pub identity_confidence: f64,
    pub identity_uncertainty_note: Option<String>,
    #[serde(default)]
    pub last_reflection_at: Option<String>,
    #[serde(default)]
    pub internal_state_summary: Value,
    #[serde(default)]
    pub internal_state_map_version: Option<i64>,
    #[serde(default)]
    pub unified_state: Value,
    #[serde(default)]
    pub unified_state_evidence: Value,
    #[serde(default)]
    pub unified_state_updated_at: Option<String>,
    pub goals: Vec<String>,
    pub calibration: CalibrationKnobs,
    pub controller_state: ControllerState,
    pub conflicts_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationKnobs {
    pub introspection_verbosity: f64,
    pub confirmation_frequency: f64,
    pub verify_threshold: f64,
    pub drift_sensitivity: f64,
    pub introspection_weight: f64,
    pub residual_salience_gain: f64,
    pub organism_influence_gain: f64,
    pub workspace_verbosity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonologueStateUpdate {
    pub timestamp: String,
    pub candidate_id: Option<String>,
    pub candidate_kind: Option<String>,
    pub target_scope: Option<String>,
    #[serde(default)]
    pub evidence_event_ids: Vec<i64>,
    #[serde(default)]
    pub belief_ids: Vec<i64>,
    pub summary: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SubjectSnapshotRecord {
    pub snapshot_hash: String,
    pub snapshot_version: String,
    pub tick_id: String,
    pub conversation_id: String,
    pub run_id: Option<String>,
    pub timestamp: String,
    pub subject_state_json: String,
}

pub async fn build_subject_state(
    db: &Db,
    kernel_state: &KernelState,
    previous: Option<&SubjectState>,
) -> Result<SubjectState, String> {
    let control_map = system_controls::load_control_map(db).await;
    let workspace_mode = system_controls::mode_for("workspace_broadcast", &control_map);
    let mut workspace_state = workspace::build_workspace_state(kernel_state, previous.map(|s| &s.workspace));
    if system_controls::mode_is_off(&workspace_mode) {
        workspace_state.winners.clear();
        workspace_state.broadcast_refs.clear();
        workspace_state.ignition.active = false;
        workspace_state.ignition.duration_ticks = 0;
        workspace_state.ignition.ignition_score = 0.0;
    } else if system_controls::mode_is_degraded(&workspace_mode) {
        if workspace_state.winners.len() > 1 {
            workspace_state.winners.truncate(1);
            workspace_state.broadcast_refs = workspace_state
                .winners
                .iter()
                .map(|c| c.content_ref.clone())
                .collect::<Vec<_>>();
            let ignition_score = if workspace_state.winners.is_empty() {
                0.0
            } else {
                workspace_state.winners.iter().map(|c| c.salience_score).sum::<f64>()
                    / workspace_state.winners.len() as f64
            };
            workspace_state.ignition.active = ignition_score >= 0.5;
            workspace_state.ignition.ignition_score = ignition_score;
        }
    }
    let error_state = load_error_state(db).await?;
    let qualia_state = qualia::compute_qualia_state(db, Some(&workspace_state)).await?;
    let attention_state =
        attention_model::compute_attention_model(db, &kernel_state.conversation_id, &workspace_state, &qualia_state)
            .await?;
    let previous_focus_refs = previous.map(|state| state.attention.current_focus_refs.as_slice());
    let attention_schema_mode = system_controls::mode_for("attention_schema", &control_map);
    let attention_schema_state = if system_controls::mode_is_off(&attention_schema_mode) {
        previous
            .map(|state| state.attention_schema.clone())
            .unwrap_or_default()
    } else {
        attention_schema::compute_attention_schema(
            db,
            &workspace_state,
            &attention_state,
            &qualia_state,
            previous_focus_refs,
        )
        .await
        .unwrap_or_default()
    };
    let organism_state =
        organism::compute_organism_state(db, kernel_state, &error_state, &attention_state, &qualia_state).await?;
    let (internal_state_summary, internal_state_map_version) =
        internal_state_map::compute_internal_state_summary(db, &attention_schema_state, &error_state).await;
    let self_model_state = build_self_model_state(
        db,
        kernel_state,
        internal_state_summary,
        internal_state_map_version,
    )
    .await?;
    let world_model_state = match world_model::build_world_model_snapshot(db, &kernel_state.conversation_id).await {
        Ok(snapshot) => snapshot,
        Err(err) => {
            let _ = system_log::log_event(
                &db.pool,
                None,
                "warn",
                "memory",
                None,
                None,
                json!({
                    "event": "world_model_snapshot_failed",
                    "conversation_id": kernel_state.conversation_id.as_str(),
                    "error": err,
                }),
            )
            .await;
            world_model::WorldModelSnapshot::default()
        }
    };
    let world_model_delta = compute_world_model_delta(previous.map(|s| &s.world_model), &world_model_state);
    let monologue_updates = load_monologue_updates(db, &kernel_state.conversation_id, 8).await;

    let mut organism_contribution = organism::wave_contribution(&organism_state);
    let qualia_strength = (qualia_state.dominant_intensity * qualia_state.prediction_confidence)
        .clamp(0.0, 1.0) as f32;
    if qualia_strength > 0.0 {
        let modulation = 1.0 + (qualia_strength * 0.25);
        let before = organism_contribution.amplitude;
        organism_contribution.amplitude = organism_contribution
            .amplitude_bounds
            .clamp_value(before * modulation);
        let _ = system_log::log_event(
            &db.pool,
            None,
            "info",
            "cognitive_wave",
            None,
            None,
            json!({
                "event": "qualia_modulated_organism_wave",
                "qualia_strength": qualia_strength,
                "amplitude_before": before,
                "amplitude_after": organism_contribution.amplitude,
            }),
        )
        .await;
    }
    let _ = cognitive_wave::try_contribute(&db.pool, None, &organism_contribution, None, None).await;
    if let Some(qualia_contribution) = qualia::wave_contribution(&qualia_state) {
        let _ = cognitive_wave::try_contribute(&db.pool, None, &qualia_contribution, None, None).await;
    }
    let self_model_contribution =
        self_model_controller::wave_contribution_from_controller_state(&self_model_state.controller_state);
    let _ = cognitive_wave::try_contribute(&db.pool, None, &self_model_contribution, None, None).await;
    let residual_mode = system_controls::mode_for("prediction_residual_influence", &control_map);
    if !system_controls::mode_is_off(&residual_mode) && !system_controls::mode_is_shadow(&residual_mode) {
        if let Some(contribution) = residual_wave_contribution(&error_state) {
            let mut contribution = contribution;
            if system_controls::mode_is_degraded(&residual_mode) {
                contribution.amplitude *= 0.5;
            }
            let _ = cognitive_wave::try_contribute(&db.pool, None, &contribution, None, None).await;
        }
    }

    if !system_controls::mode_is_off(&attention_schema_mode) {
        let focus_refs = attention_state
            .current_focus_refs
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>();
        let _ = system_log::log_event(
            &db.pool,
            None,
            "info",
            "attention",
            None,
            None,
            json!({
                "event": "attention_schema_updated",
                "capacity_usage": attention_schema_state.capacity_usage,
                "selection_policy": attention_schema_state.selection_policy,
                "stability": attention_schema_state.stability,
                "suppressed_count": attention_schema_state.suppressed_items.len(),
                "focus_ref_count": focus_refs.len(),
            }),
        )
        .await;

        let snapshot_payload = json!({
            "capacity_usage": attention_schema_state.capacity_usage,
            "stability": attention_schema_state.stability,
            "selection_policy": attention_schema_state.selection_policy,
            "suppressed_count": attention_schema_state.suppressed_items.len(),
            "focus_refs": focus_refs,
            "timestamp": attention_schema_state.last_updated_at,
        });
        if let Some(event_id) = db
            .create_system_evidence_event(
                "default",
                "attention_schema_snapshot",
                &attention_schema_state.last_updated_at,
                Some("attention_schema_snapshot"),
                &snapshot_payload.to_string(),
            )
            .await
        {
            let _ = db
                .retag_evidence_event_source_type(event_id, "attention_schema_snapshot")
                .await;
        }
    }

    Ok(SubjectState {
        workspace: workspace_state,
        organism: organism_state,
        self_model: self_model_state,
        error_state,
        attention: attention_state,
        attention_schema: attention_schema_state,
        qualia: qualia_state,
        world_model: world_model_state,
        world_model_delta,
        monologue_updates,
        plan_hash: kernel_state.last_plan_hash.clone(),
        updated_at: Utc::now().to_rfc3339(),
    })
}

fn compute_world_model_delta(
    previous: Option<&world_model::WorldModelSnapshot>,
    current: &world_model::WorldModelSnapshot,
) -> WorldModelDelta {
    const DELTA_LIMIT: usize = 24;

    fn snapshot_map(
        snapshot: &world_model::WorldModelSnapshot,
    ) -> std::collections::HashMap<i64, (String, String)> {
        let mut map = std::collections::HashMap::new();
        for fact in snapshot.facts.iter() {
            map.insert(
                fact.belief_id,
                (fact.topic_key.clone(), fact.reconcile_state.clone()),
            );
        }
        for rel in snapshot.relations.iter() {
            map.insert(
                rel.belief_id,
                (rel.topic_key.clone(), rel.reconcile_state.clone()),
            );
        }
        map
    }

    let current_map = snapshot_map(current);
    let previous_map = previous.map(snapshot_map).unwrap_or_default();

    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();

    for (belief_id, (topic_key, state)) in current_map.iter() {
        if let Some((_, prev_state)) = previous_map.get(belief_id) {
            if prev_state != state {
                changed.push(WorldModelDeltaItem {
                    belief_id: *belief_id,
                    topic_key: Some(topic_key.clone()),
                    prev_state: Some(prev_state.clone()),
                    new_state: Some(state.clone()),
                });
            }
        } else {
            added.push(WorldModelDeltaItem {
                belief_id: *belief_id,
                topic_key: Some(topic_key.clone()),
                prev_state: None,
                new_state: Some(state.clone()),
            });
        }
    }

    for (belief_id, (topic_key, state)) in previous_map.iter() {
        if !current_map.contains_key(belief_id) {
            removed.push(WorldModelDeltaItem {
                belief_id: *belief_id,
                topic_key: Some(topic_key.clone()),
                prev_state: Some(state.clone()),
                new_state: None,
            });
        }
    }

    if added.len() > DELTA_LIMIT {
        added.truncate(DELTA_LIMIT);
    }
    if removed.len() > DELTA_LIMIT {
        removed.truncate(DELTA_LIMIT);
    }
    if changed.len() > DELTA_LIMIT {
        changed.truncate(DELTA_LIMIT);
    }

    WorldModelDelta { added, removed, changed }
}

async fn load_monologue_updates(db: &Db, conversation_id: &str, limit: i64) -> Vec<MonologueStateUpdate> {
    let rows = sqlx::query(
        "SELECT timestamp, payload FROM system_logs
         WHERE json_extract(payload, '$.event') = 'monologue_state_update'
           AND json_extract(payload, '$.conversation_id') = ?
         ORDER BY datetime(timestamp) DESC
         LIMIT ?",
    )
    .bind(conversation_id)
    .bind(limit.max(1))
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();

    let mut updates = Vec::new();
    for row in rows {
        let timestamp: String = row.try_get("timestamp").unwrap_or_else(|_| Utc::now().to_rfc3339());
        let payload_raw: String = row.try_get("payload").unwrap_or_else(|_| "{}".to_string());
        let payload: Value = serde_json::from_str(&payload_raw).unwrap_or_else(|_| json!({}));
        let candidate_id = payload
            .get("candidate_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let candidate_kind = payload
            .get("candidate_kind")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let target_scope = payload
            .get("target_scope")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let evidence_event_ids = payload
            .get("evidence_event_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item.as_i64())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let belief_ids = payload
            .get("belief_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item.as_i64())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let summary = payload
            .get("summary")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        updates.push(MonologueStateUpdate {
            timestamp,
            candidate_id,
            candidate_kind,
            target_scope,
            evidence_event_ids,
            belief_ids,
            summary,
        });
    }
    updates
}

pub fn snapshot_subject_state(
    state: &SubjectState,
    tick_id: &str,
    conversation_id: &str,
    run_id: Option<&str>,
) -> Result<SubjectSnapshotRecord, String> {
    let envelope = json!({
        "snapshot_version": SUBJECT_SNAPSHOT_VERSION,
        "state": state,
    });
    let canonical = canonicalize_value(&envelope);
    let json_str = serde_json::to_string(&canonical).map_err(|e| e.to_string())?;
    let hash = hash_payload(&json_str);
    Ok(SubjectSnapshotRecord {
        snapshot_hash: hash,
        snapshot_version: SUBJECT_SNAPSHOT_VERSION.to_string(),
        tick_id: tick_id.to_string(),
        conversation_id: conversation_id.to_string(),
        run_id: run_id.map(|s| s.to_string()),
        timestamp: Utc::now().to_rfc3339(),
        subject_state_json: json_str,
    })
}

pub async fn persist_subject_snapshot(db: &Db, record: &SubjectSnapshotRecord) -> Result<(), String> {
    sqlx::query(
        "INSERT INTO subject_snapshots
         (snapshot_hash, snapshot_version, tick_id, conversation_id, run_id, timestamp, subject_state_json)
         VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&record.snapshot_hash)
    .bind(&record.snapshot_version)
    .bind(&record.tick_id)
    .bind(&record.conversation_id)
    .bind(record.run_id.as_deref())
    .bind(&record.timestamp)
    .bind(&record.subject_state_json)
    .execute(&db.pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub async fn latest_snapshot_hash(db: &Db, conversation_id: &str) -> Option<String> {
    sqlx::query_scalar(
        "SELECT snapshot_hash FROM subject_snapshots
         WHERE conversation_id = ?
         ORDER BY datetime(timestamp) DESC
         LIMIT 1",
    )
    .bind(conversation_id)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
}

pub async fn load_latest_subject_state(db: &Db, conversation_id: &str) -> Option<SubjectState> {
    let raw: Option<String> = sqlx::query_scalar(
        "SELECT subject_state_json FROM subject_snapshots
         WHERE conversation_id = ?
         ORDER BY datetime(timestamp) DESC
         LIMIT 1",
    )
    .bind(conversation_id)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let raw = raw?;
    let value: Value = serde_json::from_str(&raw).ok()?;
    let state_value = value.get("state")?.clone();
    serde_json::from_value::<SubjectState>(state_value).ok()
}

async fn build_self_model_state(
    db: &Db,
    kernel_state: &KernelState,
    internal_state_summary: Value,
    internal_state_map_version: Option<i64>,
) -> Result<SelfModelState, String> {
    let mut model: SelfModel = db.get_self_model().await.map_err(|e| e.to_string())?;
    let goals: Vec<String> = serde_json::from_value(model.goals.clone()).unwrap_or_default();
    let identity_confidence = model.identity_confidence as f64;
    let identity_uncertainty_note = model.identity_uncertainty_note.clone();
    let last_reflection_at = model.last_reflection_at.clone();
    let controller_state = kernel_state.controller_state.clone().unwrap_or_default();
    let conflicts_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'self_claim_contradiction'
           AND datetime(timestamp) >= datetime('now', '-7 days')"
    )
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    let calibration = CalibrationKnobs {
        introspection_verbosity: kernel_state.introspection_verbosity.unwrap_or(0.5) as f64,
        confirmation_frequency: kernel_state.confirmation_frequency.unwrap_or(0.5) as f64,
        verify_threshold: kernel_state.verify_threshold.unwrap_or(0.5) as f64,
        drift_sensitivity: kernel_state.drift_sensitivity.unwrap_or(0.5) as f64,
        introspection_weight: kernel_state.introspection_weight.unwrap_or(0.5) as f64,
        residual_salience_gain: kernel_state.residual_salience_gain.unwrap_or(0.5) as f64,
        organism_influence_gain: kernel_state.organism_influence_gain.unwrap_or(0.5) as f64,
        workspace_verbosity: kernel_state.workspace_verbosity.unwrap_or(0.5) as f64,
    };

    let mut needs_update = false;
    if model.internal_state_summary != internal_state_summary {
        model.internal_state_summary = internal_state_summary.clone();
        needs_update = true;
    }
    if model.internal_state_map_version != internal_state_map_version {
        model.internal_state_map_version = internal_state_map_version;
        needs_update = true;
    }
    if needs_update {
        let _ = db.set_self_model(&model).await;
    }

    Ok(SelfModelState {
        self_identity_claim_id: kernel_state.self_identity_claim_id.clone(),
        identity_confidence,
        identity_uncertainty_note,
        last_reflection_at,
        internal_state_summary,
        internal_state_map_version,
        unified_state: model.unified_state.clone(),
        unified_state_evidence: model.unified_state_evidence.clone(),
        unified_state_updated_at: model.unified_state_updated_at.clone(),
        goals,
        calibration,
        controller_state,
        conflicts_count,
    })
}

async fn load_error_state(db: &Db) -> Result<ErrorState, String> {
    let residual_rows = sqlx::query(
        "SELECT residual_id, prediction_id, normalized_residual, salience_score, created_at
         FROM residual_vectors
         ORDER BY datetime(created_at) DESC
         LIMIT 10",
    )
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();
    let mut recent_residuals = Vec::new();
    for row in residual_rows {
        let residual_id: String = row.try_get("residual_id").unwrap_or_default();
        let prediction_id: String = row.try_get("prediction_id").unwrap_or_default();
        let normalized_residual: f64 = row.try_get("normalized_residual").unwrap_or(0.0);
        let salience_score: f64 = row.try_get("salience_score").unwrap_or(0.0);
        let created_at: String = row.try_get("created_at").unwrap_or_default();
        recent_residuals.push(ResidualSummary {
            residual_id,
            prediction_id,
            normalized_residual,
            salience_score,
            created_at,
        });
    }

    let error_rows = sqlx::query(
        "SELECT error_event_id, classification, status
         FROM error_events
         WHERE status = 'OPEN'
         ORDER BY datetime(created_at) DESC
         LIMIT 10",
    )
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();
    let mut open_error_ids = Vec::new();
    let mut pattern_flags = Vec::new();
    for row in error_rows {
        let error_id: String = row.try_get("error_event_id").unwrap_or_default();
        let classification: String = row.try_get("classification").unwrap_or_default();
        open_error_ids.push(error_id);
        if classification == "CLAIM_WRONG" {
            pattern_flags.push("claim_wrong".to_string());
        } else if classification == "RETRIEVAL_DRIFT" {
            pattern_flags.push("retrieval_drift".to_string());
        } else if classification == "ATTENTION_DRIFT" {
            pattern_flags.push("attention_drift".to_string());
        }
    }

    let open_error_count = open_error_ids.len();
    let diagnosis_flags = if open_error_count >= 3 {
        vec!["diagnose_loop".to_string()]
    } else {
        Vec::new()
    };

    Ok(ErrorState {
        recent_residuals,
        open_error_ids,
        open_error_count,
        pattern_flags,
        diagnosis_flags,
    })
}

fn hash_payload(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

fn canonicalize_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut sorted = serde_json::Map::new();
            for key in keys {
                if let Some(val) = map.get(key) {
                    sorted.insert(key.clone(), canonicalize_value(val));
                }
            }
            Value::Object(sorted)
        }
        Value::Array(arr) => {
            Value::Array(arr.iter().map(canonicalize_value).collect())
        }
        _ => value.clone(),
    }
}

fn residual_wave_contribution(error_state: &ErrorState) -> Option<cognitive_wave::WaveContributionInput> {
    if error_state.recent_residuals.is_empty() {
        return None;
    }
    let mut signal = 0.0f64;
    for residual in error_state.recent_residuals.iter() {
        signal += residual.normalized_residual.abs() * residual.salience_score.max(0.0);
    }
    let avg = (signal / error_state.recent_residuals.len() as f64).clamp(0.0, 1.0);
    if avg <= 0.0 {
        return None;
    }
    let coeffs = vec![
        num_complex::Complex32::new(avg as f32, 1.0 - avg as f32),
        num_complex::Complex32::new(avg as f32 * 0.6, avg as f32 * 0.4),
    ];
    let amplitude = (0.1 + avg as f32 * 0.5).clamp(0.05, 0.9);
    Some(cognitive_wave::WaveContributionInput {
        source: "prediction_residual",
        band: cognitive_wave::WaveBand::Attention,
        coeffs,
        amplitude,
        amplitude_bounds: cognitive_wave::AmplitudeBounds::new(0.05, 0.9),
        decay_profile: cognitive_wave::DecayProfile::for_band(cognitive_wave::WaveBand::Attention),
    })
}
