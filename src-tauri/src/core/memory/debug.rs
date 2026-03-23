//! Memory System Observability Module (ICS v4.1 §13)
//! Provides debug logging and explanation data for memory operations.

use serde::{Serialize, Deserialize};

/// Debug log for a memory retrieval operation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetrievalDebugLog {
    /// Query that triggered retrieval
    pub query: String,
    /// Anchors found with their scores
    pub anchors: Vec<AnchorDebug>,
    /// Graph traversal steps
    pub traversal: TraversalDebug,
    /// Selection/ranking breakdown
    pub selection: SelectionDebug,
    /// Scope shadowing decisions
    pub shadowing: ShadowingDebug,
    /// Total time in milliseconds
    pub duration_ms: u64,
}

/// Debug info for an anchor entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorDebug {
    pub entity_id: i64,
    pub label: String,
    pub source: String,        // "fts", "vector", "exact"
    pub score: f32,
}

/// Debug info for graph traversal
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraversalDebug {
    pub starting_anchors: usize,
    pub hops_taken: usize,
    pub entities_visited: usize,
    pub beliefs_collected: usize,
    pub frontier_max_size: usize,
    pub was_bounded: bool,     // true if hit limits
}

/// Debug info for selection/ranking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SelectionDebug {
    pub facts_before_filter: usize,
    pub facts_after_negation: usize,
    pub facts_after_shadowing: usize,
    pub facts_final: usize,
    /// Top 5 facts with their ranking breakdown
    pub top_facts: Vec<FactRankingDebug>,
}

/// Ranking breakdown for a single fact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactRankingDebug {
    pub id: i64,
    pub topic_key: String,
    pub value_preview: String, // First 50 chars
    pub score: f32,
    pub components: RankingComponents,
}

/// Individual components of the ranking score
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RankingComponents {
    pub evidence_weight: f32,
    pub confidence: f32,
    pub salience: f32,
    pub time_decay: f32,
    pub assoc_weight: f32,
    pub i4_support: f32,
}

/// Debug info for scope shadowing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShadowingDebug {
    pub topic_groups_count: usize,
    pub beliefs_shadowed: usize,
    /// Details of shadowed beliefs
    pub shadowed_details: Vec<ShadowedBelief>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowedBelief {
    pub id: i64,
    pub topic_key: String,
    pub scope: String,
    pub shadowed_by_scope: String,
}

/// Debug log for entity resolution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResolutionDebugLog {
    /// Reference text being resolved
    pub ref_text: String,
    /// Candidates considered with scores
    pub candidates: Vec<CandidateDebug>,
    /// Final decision
    pub decision: String, // "resolved", "ambiguous", "new_entity"
    /// Selected entity ID if resolved
    pub selected_id: Option<i64>,
    /// Margin between top candidates (for ambiguity)
    pub margin: Option<f32>,
    /// Why disambiguation was triggered (if applicable)
    pub disambiguation_reason: Option<String>,
}

/// Debug info for a resolution candidate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateDebug {
    pub entity_id: i64,
    pub label: String,
    pub total_score: f32,
    pub score_breakdown: CandidateScoreBreakdown,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CandidateScoreBreakdown {
    pub label_score: f32,
    pub type_score: f32,
    pub recency_score: f32,
    pub neighborhood_score: f32,
    /// Neighbor IDs that contributed to overlap
    pub neighborhood_contributors: Vec<i64>,
}

/// Compile (write) operation debug log
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompileDebugLog {
    /// Statements parsed
    pub statements_parsed: usize,
    /// Resolutions performed
    pub resolutions: Vec<ResolutionDebugLog>,
    /// Beliefs written
    pub beliefs_written: usize,
    /// Beliefs reinforced (duplicate signature)
    pub beliefs_reinforced: usize,
    /// Beliefs superseded
    pub beliefs_superseded: usize,
    /// Errors encountered
    pub errors: Vec<String>,
}

/// Master debug log container
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryDebugLog {
    pub retrieval: Option<RetrievalDebugLog>,
    pub resolution: Option<ResolutionDebugLog>,
    pub compile: Option<CompileDebugLog>,
    pub timestamp: String,
}

impl MemoryDebugLog {
    pub fn new() -> Self {
        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            ..Default::default()
        }
    }
    
    pub fn with_retrieval(mut self, log: RetrievalDebugLog) -> Self {
        self.retrieval = Some(log);
        self
    }
    
    pub fn with_resolution(mut self, log: ResolutionDebugLog) -> Self {
        self.resolution = Some(log);
        self
    }
    
    pub fn with_compile(mut self, log: CompileDebugLog) -> Self {
        self.compile = Some(log);
        self
    }
    
    /// Format as human-readable summary
    pub fn format_summary(&self) -> String {
        let mut lines = vec![];
        
        if let Some(ref ret) = self.retrieval {
            lines.push(format!("[Retrieval] query=\"{}\" anchors={} facts={} duration={}ms",
                ret.query,
                ret.anchors.len(),
                ret.selection.facts_final,
                ret.duration_ms,
            ));
            
            if !ret.anchors.is_empty() {
                lines.push("  Anchors:".to_string());
                for a in &ret.anchors {
                    lines.push(format!("    - {} (id={}, src={}, score={:.2})", 
                        a.label, a.entity_id, a.source, a.score));
                }
            }
            
            if ret.shadowing.beliefs_shadowed > 0 {
                lines.push(format!("  Shadowing: {} beliefs shadowed", ret.shadowing.beliefs_shadowed));
            }
        }
        
        if let Some(ref res) = self.resolution {
            lines.push(format!("[Resolution] ref=\"{}\" decision={} candidates={}",
                res.ref_text,
                res.decision,
                res.candidates.len(),
            ));
            
            if let Some(margin) = res.margin {
                lines.push(format!("  Margin: {:.3}", margin));
            }
            
            if let Some(ref reason) = res.disambiguation_reason {
                lines.push(format!("  Disambiguation: {}", reason));
            }
        }
        
        if let Some(ref comp) = self.compile {
            lines.push(format!("[Compile] parsed={} written={} reinforced={} superseded={}",
                comp.statements_parsed,
                comp.beliefs_written,
                comp.beliefs_reinforced,
                comp.beliefs_superseded,
            ));
            
            if !comp.errors.is_empty() {
                lines.push(format!("  Errors: {:?}", comp.errors));
            }
        }
        
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_debug_log_creation() {
        let log = MemoryDebugLog::new();
        assert!(log.retrieval.is_none());
        assert!(!log.timestamp.is_empty());
    }
    
    #[test]
    fn test_format_summary() {
        let mut log = MemoryDebugLog::new();
        log.retrieval = Some(RetrievalDebugLog {
            query: "test query".to_string(),
            anchors: vec![AnchorDebug {
                entity_id: 1,
                label: "TestEntity".to_string(),
                source: "fts".to_string(),
                score: 0.95,
            }],
            selection: SelectionDebug {
                facts_final: 5,
                ..Default::default()
            },
            duration_ms: 42,
            ..Default::default()
        });
        
        let summary = log.format_summary();
        assert!(summary.contains("test query"));
        assert!(summary.contains("TestEntity"));
    }
}
