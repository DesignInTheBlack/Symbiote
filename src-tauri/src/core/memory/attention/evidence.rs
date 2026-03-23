use crate::core::memory::types::SourceType;
use crate::core::memory::config::{SOURCE_WEIGHT_USER, SOURCE_WEIGHT_TOOL, SOURCE_WEIGHT_SYSTEM, SOURCE_WEIGHT_INFERENCE, CONFIDENCE_ALPHA};

pub fn compute_evidence_weight(source: SourceType) -> f32 {
    match source {
        SourceType::User => SOURCE_WEIGHT_USER,
        SourceType::Tool => SOURCE_WEIGHT_TOOL,
        SourceType::System => SOURCE_WEIGHT_SYSTEM,
        SourceType::Inference => SOURCE_WEIGHT_INFERENCE,
    }
}

/// Update confidence based on new evidence weight (Spec §9.3)
/// C_new = C_old + alpha * weight * (1 - C_old)
pub fn compute_updated_confidence(current_confidence: f32, weight: f32) -> f32 {
    current_confidence + CONFIDENCE_ALPHA * weight * (1.0 - current_confidence)
}
