use chrono::Utc;
use num_complex::Complex32;
use serde::Serialize;

use crate::core::cognitive_wave::{WaveBand, WaveField, BAND_COUNT};

#[derive(Clone, Debug, Serialize)]
pub struct WaveStateVector {
    pub timestamp: String,
    pub coherence: f32,
    pub turbulence: f32,
    pub drift: f32,
    pub dominance: String,
    pub fragmentation: f32,
    pub total_energy: f32,
    pub band_energy: [f32; BAND_COUNT],
}

pub struct WaveProjector {
    prev_coeffs: Vec<Complex32>,
    prev_sum: Complex32,
    prev_sum_abs: f32,
}

impl WaveProjector {
    pub fn new(size: usize) -> Self {
        Self {
            prev_coeffs: vec![Complex32::new(0.0, 0.0); size],
            prev_sum: Complex32::new(0.0, 0.0),
            prev_sum_abs: 0.0,
        }
    }

    pub fn project(&mut self, field: &WaveField) -> WaveStateVector {
        let coeffs = field.coeffs();
        let mut sum = Complex32::new(0.0, 0.0);
        let mut sum_abs = 0.0f32;
        for coeff in coeffs.iter() {
            sum += *coeff;
            sum_abs += coeff.norm();
        }
        let coherence = if sum_abs > 0.0 {
            (sum.norm() / sum_abs).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let turbulence = if self.prev_coeffs.len() == coeffs.len() {
            let mut delta_sum = 0.0f32;
            for (idx, coeff) in coeffs.iter().enumerate() {
                delta_sum += (*coeff - self.prev_coeffs[idx]).norm();
            }
            let mean_delta = delta_sum / coeffs.len().max(1) as f32;
            let mean_mag = sum_abs / coeffs.len().max(1) as f32;
            if mean_mag > 0.0 {
                (mean_delta / mean_mag).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        let drift = if self.prev_sum_abs > 0.0 || sum_abs > 0.0 {
            let denom = (sum.norm() + self.prev_sum.norm()).max(0.0001);
            ((sum - self.prev_sum).norm() / denom).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let mut band_energy = [0.0f32; BAND_COUNT];
        for band in WaveBand::all() {
            band_energy[band.index()] = field.band_energy(band);
        }
        let total_energy: f32 = band_energy.iter().sum();
        let dominance = if total_energy > 0.0 {
            let mut best_band = WaveBand::Organism;
            let mut best_energy = band_energy[0];
            for band in WaveBand::all() {
                let energy = band_energy[band.index()];
                if energy > best_energy {
                    best_energy = energy;
                    best_band = band;
                }
            }
            best_band.label().to_string()
        } else {
            "none".to_string()
        };

        let fragmentation = if total_energy > 0.0 {
            let mut entropy = 0.0f32;
            for energy in band_energy.iter() {
                if *energy <= 0.0 {
                    continue;
                }
                let p = *energy / total_energy;
                entropy -= p * p.ln();
            }
            let norm = (BAND_COUNT as f32).ln().max(0.0001);
            (entropy / norm).clamp(0.0, 1.0)
        } else {
            0.0
        };

        self.prev_coeffs = coeffs.to_vec();
        self.prev_sum = sum;
        self.prev_sum_abs = sum_abs;

        WaveStateVector {
            timestamp: Utc::now().to_rfc3339(),
            coherence,
            turbulence,
            drift,
            dominance,
            fragmentation,
            total_energy,
            band_energy,
        }
    }
}

pub fn format_wave_state(state: &WaveStateVector) -> String {
    serde_json::to_string(state).unwrap_or_else(|_| "{}".to_string())
}
