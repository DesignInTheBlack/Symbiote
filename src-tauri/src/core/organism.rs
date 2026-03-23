use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::Row;

use crate::db::Db;
use crate::core::kernel::KernelState;
use crate::core::subject_state::ErrorState;
use crate::core::attention_model::AttentionModel;
use crate::core::qualia::QualiaState;
use crate::core::system_controls;
use crate::core::cognitive_wave::{AmplitudeBounds, DecayProfile, WaveBand, WaveContributionInput};
use num_complex::Complex32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismState {
    pub timestamp: String,
    pub arousal: f64,
    pub stress: f64,
    pub fatigue: f64,
    pub uncertainty_pressure: f64,
    pub social_alignment: f64,
    pub integrity_risk: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValenceSignal {
    pub timestamp: String,
    pub goal_progress: f64,
    pub uncertainty_change: f64,
    pub integrity_change: f64,
    pub alignment_change: f64,
    pub net_valence: f64,
}

async fn load_previous_organism_state(db: &Db, conversation_id: &str) -> Option<OrganismState> {
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
    let Ok(value) = serde_json::from_str::<Value>(&raw) else {
        return None;
    };
    let organism_value = value.get("state")?.get("organism")?.clone();
    serde_json::from_value::<OrganismState>(organism_value).ok()
}

pub async fn compute_organism_state(
    db: &Db,
    kernel_state: &KernelState,
    error_state: &ErrorState,
    attention: &AttentionModel,
    qualia: &QualiaState,
) -> Result<OrganismState, String> {
    let now = chrono::Utc::now().to_rfc3339();
    let mode: Option<String> = sqlx::query_scalar(
        "SELECT mode FROM system_controls WHERE subsystem_id = ?",
    )
    .bind("organism_loop")
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let organism_mode = mode.unwrap_or_else(|| {
        system_controls::default_mode_for("organism_loop")
            .unwrap_or("normal")
            .to_string()
    });
    if system_controls::mode_is_off(&organism_mode) || system_controls::mode_is_degraded(&organism_mode) {
        if let Some(previous) = load_previous_organism_state(db, &kernel_state.conversation_id).await {
            return Ok(OrganismState { timestamp: now, ..previous });
        }
        return Ok(OrganismState {
            timestamp: now,
            arousal: 0.5,
            stress: 0.0,
            fatigue: 0.0,
            uncertainty_pressure: 0.0,
            social_alignment: 0.5,
            integrity_risk: 0.0,
        });
    }
    let drift = kernel_state
        .controller_state
        .as_ref()
        .map(|c| c.drift_score as f64)
        .unwrap_or(0.0);
    let uncertainty = kernel_state
        .controller_state
        .as_ref()
        .map(|c| c.uncertainty as f64)
        .unwrap_or(0.0);
    let failure_streak = kernel_state
        .controller_state
        .as_ref()
        .map(|c| c.failure_streak as f64)
        .unwrap_or(0.0);

    let settings = db.get_settings().await.ok();
    let explicit_feedback_only = settings
        .as_ref()
        .and_then(|s| s.explicit_feedback_only)
        .unwrap_or(true);
    let organism_decay = settings
        .as_ref()
        .and_then(|s| s.organism_decay)
        .unwrap_or(true);

    let recent_runs: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM runs WHERE datetime(started_at) >= datetime('now', '-1 hour')",
    )
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);

    let mut feedback_pushback: i64 = 0;
    let mut feedback_clarify: i64 = 0;
    let mut feedback_disengage: i64 = 0;
    let mut feedback_agree: i64 = 0;
    let mut feedback_follow_up: i64 = 0;
    let feedback_rows = if explicit_feedback_only {
        sqlx::query(
            "SELECT json_extract(payload, '$.kind') AS kind, COUNT(*) AS count
             FROM system_logs
             WHERE json_extract(payload, '$.event') = 'user_feedback_detected'
               AND json_extract(payload, '$.explicit') = 1
               AND datetime(timestamp) >= datetime('now', '-1 day')
             GROUP BY kind",
        )
        .fetch_all(&db.pool)
        .await
        .unwrap_or_default()
    } else {
        sqlx::query(
            "SELECT json_extract(payload, '$.kind') AS kind, COUNT(*) AS count
             FROM system_logs
             WHERE json_extract(payload, '$.event') = 'user_feedback_detected'
               AND datetime(timestamp) >= datetime('now', '-1 day')
             GROUP BY kind",
        )
        .fetch_all(&db.pool)
        .await
        .unwrap_or_default()
    };
    for row in feedback_rows {
        let kind: String = row.try_get("kind").unwrap_or_default();
        let count: i64 = row.try_get("count").unwrap_or(0);
        match kind.as_str() {
            "pushback" => feedback_pushback = count,
            "clarify" => feedback_clarify = count,
            "disengage" => feedback_disengage = count,
            "agree" => feedback_agree = count,
            "follow_up" => feedback_follow_up = count,
            _ => {}
        }
    }

    let feedback_score = (feedback_agree as f64 + 0.25 * feedback_follow_up as f64)
        - (feedback_pushback + feedback_clarify + feedback_disengage) as f64;
    let qualia_reward = qualia.last_reward.unwrap_or(0.0).clamp(-1.0, 1.0);
    let social_alignment = (((feedback_score + 5.0) / 10.0) + qualia_reward * 0.1).clamp(0.0, 1.0);

    let open_error_pressure = (error_state.open_error_count as f64 / 5.0).clamp(0.0, 1.0);
    let attention_pressure = attention.meta_confidence;
    let arousal = (0.2 + open_error_pressure + attention_pressure * 0.3 + qualia_reward.abs() * 0.2).clamp(0.0, 1.0);
    let stress = (open_error_pressure + drift * 0.5 + uncertainty * 0.5 + (-qualia_reward).max(0.0) * 0.3).clamp(0.0, 1.0);
    let mut fatigue = (recent_runs as f64 / 30.0).clamp(0.0, 1.0);
    let uncertainty_pressure = (uncertainty + open_error_pressure * 0.4 + (-qualia_reward).max(0.0) * 0.2).clamp(0.0, 1.0);
    let integrity_risk = (failure_streak / 10.0 + open_error_pressure * 0.5).clamp(0.0, 1.0);
    let mut social_alignment = social_alignment;

    if organism_decay {
        if let Some(prev) = load_previous_organism_state(db, &kernel_state.conversation_id).await {
            let alpha = 0.6;
            fatigue = (alpha * fatigue) + ((1.0 - alpha) * prev.fatigue);
            social_alignment = (alpha * social_alignment) + ((1.0 - alpha) * prev.social_alignment);
        }
    }

    Ok(OrganismState {
        timestamp: now,
        arousal,
        stress,
        fatigue,
        uncertainty_pressure,
        social_alignment,
        integrity_risk,
    })
}

pub fn compute_valence_signal(state: &OrganismState) -> ValenceSignal {
    let goal_progress = (1.0 - state.uncertainty_pressure).clamp(0.0, 1.0);
    let uncertainty_change = -state.uncertainty_pressure;
    let integrity_change = -state.integrity_risk;
    let alignment_change = state.social_alignment;
    let net_valence = (goal_progress + uncertainty_change + integrity_change + alignment_change) / 4.0;
    ValenceSignal {
        timestamp: chrono::Utc::now().to_rfc3339(),
        goal_progress,
        uncertainty_change,
        integrity_change,
        alignment_change,
        net_valence,
    }
}

pub fn wave_contribution(state: &OrganismState) -> WaveContributionInput {
    let values = [
        state.arousal,
        state.stress,
        state.fatigue,
        state.uncertainty_pressure,
        state.social_alignment,
        state.integrity_risk,
    ];
    let mut coeffs = Vec::with_capacity(values.len());
    for value in values.iter() {
        let real = (*value).clamp(0.0, 1.0) as f32;
        let imag = (1.0 - *value).clamp(0.0, 1.0) as f32;
        coeffs.push(Complex32::new(real, imag));
    }
    let amplitude = (0.2 + state.arousal as f32).clamp(0.05, 0.8);
    WaveContributionInput {
        source: "organism",
        band: WaveBand::Organism,
        coeffs,
        amplitude,
        amplitude_bounds: AmplitudeBounds::new(0.05, 0.8),
        decay_profile: DecayProfile::for_band(WaveBand::Organism),
    }
}
