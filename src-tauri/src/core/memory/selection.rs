use crate::core::memory::config::ANSWER_CONFIDENCE_K;

/// Spec §I4: Support Formula
/// S(b) = ln(1 + W_b) * (0.5 + 0.5 * C_b)
pub fn compute_support(weight: f32, confidence: f32) -> f32 {
    (1.0 + weight).ln() * (0.5 + 0.5 * confidence)
}

/// Spec §I4: Reliability Formula
/// reliability(b) = 1 / (1 + contradicted_support(topic_key, b))
/// Reduces effective support when contradicting beliefs exist
pub fn compute_reliability(contradicted_support: f32) -> f32 {
    1.0 / (1.0 + contradicted_support)
}

/// Spec §I4: Effective Support
/// eff_support(b) = support(b) * reliability(b)
pub fn compute_effective_support(support: f32, reliability: f32) -> f32 {
    support * reliability
}

/// Spec §I4: Answer Confidence Formula
/// AC(b) = eff_support(b) / (eff_support(b) + Sum(eff_support(competitors)) + K)
pub fn compute_answer_confidence(eff_support: f32, sum_competitor_eff_support: f32) -> f32 {
    eff_support / (eff_support + sum_competitor_eff_support + ANSWER_CONFIDENCE_K)
}

// Struct for scoring
pub struct ScoredBelief {
    pub belief_id: i64,
    pub support: f32,
    pub effective_support: f32,
    pub answer_confidence: f32,
}

/// Select best belief using full I4 formula with reliability
/// contradicted_map: belief_id -> sum of contradicting beliefs' support
pub fn select_best_belief(candidates: &[(i64, f32, f32)], contradicted_map: &std::collections::HashMap<i64, f32>) -> Option<ScoredBelief> {
    // candidates: (id, weight, confidence)
    if candidates.is_empty() { return None; }
    
    // 1. Calculate Effective Support for all (with reliability)
    let eff_supports: Vec<(i64, f32, f32)> = candidates.iter().map(|(id, w, c)| {
        let support = compute_support(*w, *c);
        let contradicted = contradicted_map.get(id).copied().unwrap_or(0.0);
        let reliability = compute_reliability(contradicted);
        let eff = compute_effective_support(support, reliability);
        (*id, support, eff)
    }).collect();
    
    // 2. Select Max Effective Support
    let (best_id, best_support, best_eff) = eff_supports.iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))?;
        
    // 3. Sum others' effective support
    let sum_others: f32 = eff_supports.iter()
        .filter(|(id, _, _)| id != best_id)
        .map(|(_, _, eff)| *eff)
        .sum();
        
    let ac = compute_answer_confidence(*best_eff, sum_others);
    
    Some(ScoredBelief {
        belief_id: *best_id,
        support: *best_support,
        effective_support: *best_eff,
        answer_confidence: ac,
    })
}

