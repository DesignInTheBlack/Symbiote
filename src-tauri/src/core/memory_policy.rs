#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryWriteCategory {
    Summary,
    InnerSummary,
    Episodic,
    Semantic,
    SemanticCore,
    MemoryPass,
    Unknown,
}

impl MemoryWriteCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryWriteCategory::Summary => "summary",
            MemoryWriteCategory::InnerSummary => "inner_summary",
            MemoryWriteCategory::Episodic => "episodic",
            MemoryWriteCategory::Semantic => "semantic",
            MemoryWriteCategory::SemanticCore => "semantic_core",
            MemoryWriteCategory::MemoryPass => "memory_pass",
            MemoryWriteCategory::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryWriteSource {
    Kernel,
    Scheduler,
    ModelClient,
    MemoryWriter,
    SelfReflection,
    Unknown,
}

impl MemoryWriteSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryWriteSource::Kernel => "kernel",
            MemoryWriteSource::Scheduler => "scheduler",
            MemoryWriteSource::ModelClient => "model_client",
            MemoryWriteSource::MemoryWriter => "memory_writer",
            MemoryWriteSource::SelfReflection => "self_reflection",
            MemoryWriteSource::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryPolicy;

impl MemoryPolicy {
    pub fn is_allowed(category: MemoryWriteCategory, source: MemoryWriteSource, reason_code: &str) -> bool {
        match category {
            MemoryWriteCategory::Summary => {
                match source {
                    MemoryWriteSource::Kernel => matches!(
                        reason_code,
                        "user_visible_turn"
                            | "internal_emit"
                            | "summary_archive"
                            | "summary_archive_turn"
                            | "summary_archive_turn_count"
                    ),
                    MemoryWriteSource::Scheduler => matches!(
                        reason_code,
                        "scheduler_compaction"
                            | "summary_archive"
                            | "summary_archive_turn"
                            | "summary_archive_turn_count"
                    ),
                    _ => false,
                }
            }
            MemoryWriteCategory::InnerSummary => {
                source == MemoryWriteSource::Kernel
                    && matches!(reason_code, "user_visible_turn" | "internal_tick" | "monologue_tick")
            }
            MemoryWriteCategory::Episodic => {
                match source {
                    MemoryWriteSource::Kernel => matches!(reason_code, "meaningful_run" | "tool_outcome" | "thread_outcome"),
                    MemoryWriteSource::Scheduler => matches!(reason_code, "scheduler_compaction" | "scheduler_maintenance"),
                    MemoryWriteSource::MemoryWriter => reason_code == "memory_writer_evidence",
                    _ => false,
                }
            }
            MemoryWriteCategory::Semantic => {
                match source {
                    MemoryWriteSource::Kernel => reason_code == "kernel_slow_promotion",
                    MemoryWriteSource::ModelClient => matches!(reason_code, "memory_pass" | "memory_reinforce"),
                    MemoryWriteSource::MemoryWriter => matches!(reason_code, "memory_writer_evidence" | "memory_api_write"),
                    _ => false,
                }
            }
            MemoryWriteCategory::SemanticCore => {
                source == MemoryWriteSource::Kernel && reason_code == "kernel_slow_promotion"
            }
            MemoryWriteCategory::MemoryPass => {
                source == MemoryWriteSource::ModelClient && reason_code == "memory_pass"
            }
            MemoryWriteCategory::Unknown => false,
        }
    }
}
