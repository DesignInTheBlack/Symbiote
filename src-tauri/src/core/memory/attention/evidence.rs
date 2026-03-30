use crate::core::memory::config::{
    SOURCE_WEIGHT_INFERENCE,
    SOURCE_WEIGHT_SYSTEM,
    SOURCE_WEIGHT_TOOL,
    SOURCE_WEIGHT_USER,
    CONFIDENCE_ALPHA,
};
use crate::core::memory::types::SourceType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceQualityTier {
    VeryLow,
    Low,
    Medium,
    High,
}

impl EvidenceQualityTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            EvidenceQualityTier::VeryLow => "very_low",
            EvidenceQualityTier::Low => "low",
            EvidenceQualityTier::Medium => "medium",
            EvidenceQualityTier::High => "high",
        }
    }
}

pub fn compute_evidence_weight(source: SourceType) -> f32 {
    match source {
        SourceType::User => SOURCE_WEIGHT_USER,
        SourceType::Tool => SOURCE_WEIGHT_TOOL,
        SourceType::System => SOURCE_WEIGHT_SYSTEM,
        SourceType::Inference => SOURCE_WEIGHT_INFERENCE,
    }
}

pub fn compute_evidence_quality(
    source: SourceType,
    weight: f32,
    strength: Option<f32>,
    age_days: f32,
) -> f32 {
    let base = match source {
        SourceType::User => 1.0,
        SourceType::Tool => 0.85,
        SourceType::System => 0.75,
        SourceType::Inference => 0.5,
    };
    let weight_factor = weight.clamp(0.0, 1.0);
    let strength_factor = strength
        .map(|value| (0.4 + 0.6 * value.clamp(0.0, 1.0)).clamp(0.2, 1.0))
        .unwrap_or(0.6);
    let recency = if age_days <= 1.0 {
        1.0
    } else {
        (1.0 / (1.0 + (age_days / 14.0))).clamp(0.35, 1.0)
    };
    (base * weight_factor * strength_factor * recency).clamp(0.0, 1.0)
}

pub fn evidence_quality_tier(score: f32) -> EvidenceQualityTier {
    let score = score.clamp(0.0, 1.0);
    if score >= 0.75 {
        EvidenceQualityTier::High
    } else if score >= 0.55 {
        EvidenceQualityTier::Medium
    } else if score >= 0.35 {
        EvidenceQualityTier::Low
    } else {
        EvidenceQualityTier::VeryLow
    }
}

pub fn quality_floor_for_memory(strictness: f32) -> f32 {
    let strictness = strictness.clamp(0.0, 1.0);
    (0.35 + 0.3 * strictness).clamp(0.2, 0.85)
}

pub fn quality_floor_for_self_claim(strictness: f32) -> f32 {
    let strictness = strictness.clamp(0.0, 1.0);
    (0.45 + 0.35 * strictness).clamp(0.25, 0.9)
}

/// Update confidence based on new evidence weight (Spec 9.3)
/// C_new = C_old + alpha * weight * (1 - C_old)
pub fn compute_updated_confidence(current_confidence: f32, weight: f32) -> f32 {
    current_confidence + CONFIDENCE_ALPHA * weight * (1.0 - current_confidence)
}
