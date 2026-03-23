use std::collections::{HashMap, VecDeque};
use std::ops::Range;
use std::sync::Arc;

use chrono::Utc;
use num_complex::Complex32;
use once_cell::sync::{Lazy, OnceCell};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::SqlitePool;
use tauri::AppHandle;
use tokio::sync::RwLock;

use crate::core::kernel::utils::text::hash_payload;
use crate::core::system_controls;
use crate::core::system_log;
use crate::db::Db;

pub const WAVE_COEFFS: usize = 256;
pub const BAND_COUNT: usize = 5;
const MAX_CONTRIBUTIONS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WaveBand {
    Organism,
    Qualia,
    Memory,
    SelfModel,
    Attention,
}

impl WaveBand {
    pub fn index(self) -> usize {
        match self {
            WaveBand::Organism => 0,
            WaveBand::Qualia => 1,
            WaveBand::Memory => 2,
            WaveBand::SelfModel => 3,
            WaveBand::Attention => 4,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            WaveBand::Organism => "organism",
            WaveBand::Qualia => "qualia",
            WaveBand::Memory => "memory",
            WaveBand::SelfModel => "self_model",
            WaveBand::Attention => "attention",
        }
    }

    pub fn all() -> [WaveBand; BAND_COUNT] {
        [
            WaveBand::Organism,
            WaveBand::Qualia,
            WaveBand::Memory,
            WaveBand::SelfModel,
            WaveBand::Attention,
        ]
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DecayProfile {
    pub per_band: [f32; BAND_COUNT],
}

impl DecayProfile {
    pub fn clamp(self) -> Self {
        let mut per_band = self.per_band;
        for value in per_band.iter_mut() {
            *value = value.clamp(0.0, 1.0);
        }
        Self { per_band }
    }

    pub fn blend(self, other: DecayProfile, alpha: f32) -> Self {
        let mut per_band = [0.0f32; BAND_COUNT];
        let alpha = alpha.clamp(0.0, 1.0);
        for idx in 0..BAND_COUNT {
            per_band[idx] = (self.per_band[idx] * (1.0 - alpha)) + (other.per_band[idx] * alpha);
        }
        Self { per_band }
    }

    pub fn slow() -> Self {
        Self {
            per_band: [0.002, 0.004, 0.008, 0.002, 0.006],
        }
    }

    pub fn medium() -> Self {
        Self {
            per_band: [0.004, 0.008, 0.014, 0.004, 0.010],
        }
    }

    pub fn fast() -> Self {
        Self {
            per_band: [0.008, 0.014, 0.024, 0.010, 0.018],
        }
    }

    pub fn for_band(band: WaveBand) -> Self {
        match band {
            WaveBand::Organism => Self::slow(),
            WaveBand::Qualia => Self::medium(),
            WaveBand::Memory => Self::fast(),
            WaveBand::SelfModel => Self::slow(),
            WaveBand::Attention => Self::medium(),
        }
    }
}

impl Default for DecayProfile {
    fn default() -> Self {
        Self::slow()
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AmplitudeBounds {
    pub min: f32,
    pub max: f32,
}

impl AmplitudeBounds {
    pub fn new(min: f32, max: f32) -> Self {
        if min <= max {
            Self { min, max }
        } else {
            Self { min: max, max: min }
        }
    }

    pub fn clamp_value(&self, value: f32) -> f32 {
        value.clamp(self.min, self.max)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct WaveContributionMeta {
    pub source: String,
    pub band: WaveBand,
    pub amplitude: f32,
    pub amplitude_bounds: AmplitudeBounds,
    pub decay_profile: DecayProfile,
    pub timestamp: String,
    pub timestamp_unix: i64,
}

#[derive(Clone, Debug)]
pub struct WaveContributionInput {
    pub source: &'static str,
    pub band: WaveBand,
    pub coeffs: Vec<Complex32>,
    pub amplitude: f32,
    pub amplitude_bounds: AmplitudeBounds,
    pub decay_profile: DecayProfile,
}

#[derive(Clone, Debug, Serialize)]
pub struct WaveDecaySummary {
    pub dt_secs: f32,
    pub energy_before: f32,
    pub energy_after: f32,
    pub band_energy_after: [f32; BAND_COUNT],
}

pub struct WaveField {
    coeffs: Vec<Complex32>,
    band_ranges: [Range<usize>; BAND_COUNT],
    decay_profile: DecayProfile,
    contributions: VecDeque<WaveContributionMeta>,
}

impl WaveField {
    pub fn new() -> Self {
        let coeffs = vec![Complex32::new(0.0, 0.0); WAVE_COEFFS];
        let band_ranges = split_ranges(WAVE_COEFFS, BAND_COUNT);
        Self {
            coeffs,
            band_ranges,
            decay_profile: DecayProfile::default(),
            contributions: VecDeque::with_capacity(MAX_CONTRIBUTIONS),
        }
    }

    pub fn decay(&mut self, dt_secs: f32) -> WaveDecaySummary {
        let dt = dt_secs.max(0.0);
        let energy_before = self.total_energy();
        for (idx, range) in self.band_ranges.iter().enumerate() {
            let rate = self.decay_profile.per_band[idx].clamp(0.0, 1.0);
            let factor = (-rate * dt).exp();
            for coeff in &mut self.coeffs[range.clone()] {
                *coeff = *coeff * factor;
            }
        }
        let energy_after = self.total_energy();
        let mut band_energy_after = [0.0f32; BAND_COUNT];
        for band in WaveBand::all() {
            band_energy_after[band.index()] = self.band_energy(band);
        }
        WaveDecaySummary {
            dt_secs: dt,
            energy_before,
            energy_after,
            band_energy_after,
        }
    }

    pub fn contribute(
        &mut self,
        band: WaveBand,
        coeffs: &[Complex32],
        amplitude: f32,
        bounds: AmplitudeBounds,
        decay_profile: DecayProfile,
        source: &str,
    ) -> WaveContributionMeta {
        let bounds = AmplitudeBounds::new(bounds.min, bounds.max);
        let amplitude = bounds.clamp_value(amplitude);
        let range = self.band_ranges[band.index()].clone();
        let len = range.end.saturating_sub(range.start).max(1);
        let mut buffer = vec![Complex32::new(0.0, 0.0); len];
        for idx in 0..len {
            if let Some(value) = coeffs.get(idx) {
                buffer[idx] = *value;
            }
        }
        let mut max_mag = 0.0f32;
        for value in buffer.iter() {
            let mag = value.norm();
            if mag > max_mag {
                max_mag = mag;
            }
        }
        let scale = if max_mag > 0.0 {
            amplitude / max_mag
        } else {
            0.0
        };
        for (idx, value) in buffer.iter().enumerate() {
            let target = range.start + idx;
            if target < self.coeffs.len() {
                self.coeffs[target] += *value * scale;
            }
        }
        let decay_profile = decay_profile.clamp();
        self.decay_profile = self.decay_profile.blend(decay_profile, 0.2);

        let timestamp = Utc::now();
        let meta = WaveContributionMeta {
            source: source.to_string(),
            band,
            amplitude,
            amplitude_bounds: bounds,
            decay_profile,
            timestamp: timestamp.to_rfc3339(),
            timestamp_unix: timestamp.timestamp(),
        };
        self.contributions.push_back(meta.clone());
        while self.contributions.len() > MAX_CONTRIBUTIONS {
            self.contributions.pop_front();
        }
        meta
    }

    pub fn sample_band(&self, band: WaveBand) -> Vec<Complex32> {
        let range = self.band_ranges[band.index()].clone();
        self.coeffs[range].to_vec()
    }

    pub fn coeffs(&self) -> &[Complex32] {
        &self.coeffs
    }

    pub fn band_ranges(&self) -> &[Range<usize>; BAND_COUNT] {
        &self.band_ranges
    }

    pub fn decay_profile(&self) -> DecayProfile {
        self.decay_profile
    }

    pub fn band_energy(&self, band: WaveBand) -> f32 {
        let range = self.band_ranges[band.index()].clone();
        self.coeffs[range]
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f32>()
    }

    pub fn total_energy(&self) -> f32 {
        self.coeffs.iter().map(|c| c.norm_sqr()).sum::<f32>()
    }

    pub fn recent_contributions(&self, window_secs: i64) -> Vec<WaveContributionMeta> {
        let cutoff = Utc::now().timestamp().saturating_sub(window_secs.max(0));
        self.contributions
            .iter()
            .filter(|meta| meta.timestamp_unix >= cutoff)
            .cloned()
            .collect()
    }
}

fn split_ranges(total: usize, bands: usize) -> [Range<usize>; BAND_COUNT] {
    let base = total / bands.max(1);
    let remainder = total % bands.max(1);
    let mut ranges = Vec::with_capacity(bands);
    let mut start = 0usize;
    for idx in 0..bands {
        let extra = if idx < remainder { 1 } else { 0 };
        let end = (start + base + extra).min(total);
        ranges.push(start..end);
        start = end;
    }
    while ranges.len() < BAND_COUNT {
        ranges.push(total..total);
    }
    [
        ranges[0].clone(),
        ranges[1].clone(),
        ranges[2].clone(),
        ranges[3].clone(),
        ranges[4].clone(),
    ]
}

static WAVE_FIELD_HANDLE: OnceCell<Arc<RwLock<WaveField>>> = OnceCell::new();
static WAVE_GAIN: Lazy<std::sync::Mutex<f32>> = Lazy::new(|| std::sync::Mutex::new(1.0));

pub fn register_wave_field(handle: Arc<RwLock<WaveField>>) {
    let _ = WAVE_FIELD_HANDLE.set(handle);
}

pub fn wave_field_handle() -> Option<Arc<RwLock<WaveField>>> {
    WAVE_FIELD_HANDLE.get().cloned()
}

pub fn current_gain() -> f32 {
    *WAVE_GAIN.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn set_gain(value: f32) {
    let mut guard = WAVE_GAIN.lock().unwrap_or_else(|e| e.into_inner());
    *guard = value.clamp(0.1, 2.0);
}

pub async fn update_gain_from_attention(
    pool: &SqlitePool,
    app_handle: Option<&AppHandle>,
    meta_confidence: f64,
    sources: &[&str],
) {
    let gain = (0.5 + meta_confidence as f32).clamp(0.2, 1.5);
    set_gain(gain);
    let _ = system_log::log_event(
        pool,
        app_handle,
        "info",
        "cognitive_wave",
        None,
        None,
        serde_json::json!({
            "event": "wave_gain_updated",
            "gain": gain,
            "meta_confidence": meta_confidence,
            "sources": sources,
        }),
    )
    .await;
}

pub async fn try_contribute(
    pool: &SqlitePool,
    app_handle: Option<&AppHandle>,
    input: &WaveContributionInput,
    run_id: Option<&str>,
    trace_id: Option<&str>,
) -> Option<WaveContributionMeta> {
    let mode = mode_for_pool(pool, "cognitive_wave").await;
    if system_controls::mode_is_off(&mode) {
        let _ = system_log::log_event(
            pool,
            app_handle,
            "info",
            "cognitive_wave",
            run_id,
            trace_id,
            serde_json::json!({
                "event": "wave_contribution_skipped",
                "reason": "mode_off",
                "source": input.source,
                "band": input.band.label(),
            }),
        )
        .await;
        return None;
    }
    let handle = match wave_field_handle() {
        Some(handle) => handle,
        None => {
            let _ = system_log::log_event(
                pool,
                app_handle,
                "warn",
                "cognitive_wave",
                run_id,
                trace_id,
                serde_json::json!({
                    "event": "wave_contribution_skipped",
                    "reason": "missing_field_handle",
                    "source": input.source,
                    "band": input.band.label(),
                }),
            )
            .await;
            return None;
        }
    };

    let mut amplitude = input.amplitude;
    if system_controls::mode_is_degraded(&mode) {
        amplitude *= 0.5;
    }
    amplitude *= current_gain();

    let mut field = handle.write().await;
    let meta = field.contribute(
        input.band,
        &input.coeffs,
        amplitude,
        input.amplitude_bounds,
        input.decay_profile,
        input.source,
    );
    drop(field);

    let _ = system_log::log_event(
        pool,
        app_handle,
        "info",
        "cognitive_wave",
        run_id,
        trace_id,
        serde_json::json!({
            "event": "wave_contribution",
            "source": meta.source,
            "band": meta.band.label(),
            "amplitude": meta.amplitude,
            "amplitude_bounds": meta.amplitude_bounds,
            "decay_profile": meta.decay_profile,
            "timestamp": meta.timestamp,
        }),
    )
    .await;

    Some(meta)
}

pub async fn decay_tick(
    pool: &SqlitePool,
    app_handle: Option<&AppHandle>,
    dt_secs: f32,
    run_id: Option<&str>,
    trace_id: Option<&str>,
) {
    let mode = mode_for_pool(pool, "cognitive_wave").await;
    if system_controls::mode_is_off(&mode) {
        return;
    }
    let Some(handle) = wave_field_handle() else { return; };
    let mut field = handle.write().await;
    let summary = field.decay(dt_secs);
    drop(field);
    let _ = system_log::log_event(
        pool,
        app_handle,
        "info",
        "cognitive_wave",
        run_id,
        trace_id,
        serde_json::json!({
            "event": "wave_decay",
            "dt_secs": summary.dt_secs,
            "energy_before": summary.energy_before,
            "energy_after": summary.energy_after,
            "band_energy_after": summary.band_energy_after,
        }),
    )
    .await;
}

pub async fn maybe_emit_wave_state_snapshot(
    db: &Db,
    app_handle: Option<&AppHandle>,
    cadence_secs: i64,
) {
    let mode = mode_for_pool(&db.pool, "cognitive_wave").await;
    if system_controls::mode_is_off(&mode) {
        return;
    }
    let now = Utc::now();
    let last_emitted = db
        .get_key("wave_state_snapshot_last_at")
        .await
        .ok()
        .flatten()
        .and_then(|raw| chrono::DateTime::parse_from_rfc3339(&raw).ok())
        .map(|ts| ts.with_timezone(&Utc));
    if let Some(last) = last_emitted {
        if now.signed_duration_since(last).num_seconds() < cadence_secs {
            return;
        }
    }

    let Some(handle) = wave_field_handle() else { return; };
    let field = handle.read().await;
    let decay_profile = field.decay_profile();
    let recent = field.recent_contributions(120);
    let mut sources_by_band: HashMap<String, Vec<String>> = HashMap::new();
    for meta in recent {
        sources_by_band
            .entry(meta.band.label().to_string())
            .or_default()
            .push(meta.source);
    }
    let mut bands = serde_json::Map::new();
    for band in WaveBand::all() {
        let label = band.label();
        let amplitude = field.band_energy(band);
        let sources = sources_by_band.remove(label).unwrap_or_default();
        bands.insert(
            label.to_string(),
            json!({
                "amplitude": amplitude,
                "sources": sources,
            }),
        );
    }
    let decay_map = json!({
        "organism": decay_profile.per_band[WaveBand::Organism.index()],
        "qualia": decay_profile.per_band[WaveBand::Qualia.index()],
        "memory": decay_profile.per_band[WaveBand::Memory.index()],
        "self_model": decay_profile.per_band[WaveBand::SelfModel.index()],
        "attention": decay_profile.per_band[WaveBand::Attention.index()],
    });
    let payload = json!({
        "bands": bands,
        "decay_profile": decay_map,
        "timestamp": now.to_rfc3339(),
    });
    let payload_json = payload.to_string();
    let snapshot_hash = hash_payload(&payload_json);

    let _ = system_log::log_event(
        &db.pool,
        app_handle,
        "info",
        "cognitive_wave",
        None,
        None,
        json!({
            "event": "wave_state_snapshot",
            "bands": payload.get("bands").cloned().unwrap_or_else(|| json!({})),
            "decay_profile": payload.get("decay_profile").cloned().unwrap_or_else(|| json!({})),
            "timestamp": payload.get("timestamp").cloned().unwrap_or_else(|| json!(now.to_rfc3339())),
        }),
    )
    .await;

    if let Some(event_id) = db
        .create_system_evidence_event(
            "default",
            "wave_state_snapshot",
            &snapshot_hash,
            Some("wave_state"),
            &payload_json,
        )
        .await
    {
        let _ = db
            .retag_evidence_event_source_type(event_id, "wave_state")
            .await;
    }

    let _ = db
        .set_key("wave_state_snapshot_last_at", &now.to_rfc3339())
        .await;
}

async fn mode_for_pool(pool: &SqlitePool, subsystem_id: &str) -> String {
    let mode: Option<String> = sqlx::query_scalar(
        "SELECT mode FROM system_controls WHERE subsystem_id = ?",
    )
    .bind(subsystem_id)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    mode.unwrap_or_else(|| {
        system_controls::default_mode_for(subsystem_id)
            .unwrap_or("normal")
            .to_string()
    })
}
