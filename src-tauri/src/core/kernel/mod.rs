use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use regex::{Regex, RegexSet};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sqlx::{Row, SqlitePool};
use tauri::{AppHandle, Emitter};
use tokio::sync::{oneshot, watch, Mutex, RwLock};
use uuid::Uuid;

use crate::core::inner_summary::{sanitize_inner_summary, InnerSummary};
use crate::core::input_resolution::{InputResolutionContext, ResolvedParam, resolve_required_slots};
use crate::core::feedback;
use crate::core::memory::api::MemoryApi;
use crate::core::memory::canonical::{compute_signature_hash, compute_topic_key_fact, compute_value_hash};
use crate::core::memory::candidate;
use crate::core::memory::attention::evidence::compute_evidence_weight;
use crate::core::memory::types::{Scope, SourceType};
use crate::core::memory_policy::{MemoryPolicy, MemoryWriteCategory, MemoryWriteSource};
use crate::core::model_client::{ChatCompletionRequest, ChatMessage, ChatResponseMeta, MemoryPassResult, ModelClient};
use crate::core::organism;
use crate::core::qualia;
use crate::core::cognitive_wave::{self, WaveField};
use crate::core::cognitive_wave_projection::{WaveProjector, WaveStateVector, format_wave_state};
use crate::core::prompt_builder::{
    build_core_system_message,
    build_core_system_message_with_layout,
    CoreInputKind,
    CorePromptBuild,
    CorePromptInput,
    PromptLayout,
};
use crate::core::prompt_loader;
use crate::core::post_processing;
use crate::core::run_phase::{advance_run_phase, RunPhase};
use crate::core::self_model_controller::{apply_reconstruction_to_model, collect_self_evidence_metrics, evaluate_gates, reconstruct_from_metrics};
use crate::core::self_claims::{self, SelfClaimInput};
use crate::core::system_controls;
use crate::core::{subject_controller, subject_state, workspace as core_workspace};
use crate::core::system_log;
use crate::core::token_estimator;
use crate::core::tool_args;

mod arbitration;
mod monologue;
mod memory_orch;
pub(crate) mod constants;
mod types;
pub(crate) mod utils;
mod prediction;
mod telemetry;
mod gating;
mod tools;
pub(crate) mod workspace;
mod commit;
mod run;
mod prompt;
mod proaction;
mod subject;
mod pipeline;
#[cfg(test)]
mod tests;

use constants::*;
use types::*;
use utils::*;
use prediction::*;
use telemetry::*;
use gating::*;
use tools::*;
use workspace::*;
use commit::*;
use prompt::*;
use proaction::*;

pub use types::{
    AllowedCapabilities,
    Candidate,
    CandidateKind,
    DecisionReport,
    is_state_change_candidate,
    KernelDecision,
    KernelMode,
    KernelState,
    MetaCogPending,
    Outcome,
    ProactionAdjustments,
    ProactionMetrics,
    ProactionState,
    RejectedCandidate,
    StanceState,
    StopReason,
    StopReasonCategory,
    StopScope,
    StopState,
    TaskPhase,
    ThreadHandle,
    ToolFailurePenalty,
};
pub use gating::sanitize_user_output;
pub(crate) use gating::validate_evidence_ids_with_pool;
pub(crate) use utils::json::{parse_json_object_with_repair, repair_json_object};
pub(crate) use types::ValidationResult;
pub use prediction::{
    allowed_prediction_horizons,
    allowed_prediction_metrics,
    record_prediction_rejection,
    validate_prediction_fields,
};
pub(crate) use prediction::validate_prediction_basics;
pub use constants::MONOLOGUE_STATE_CHANGE_WINDOW_TICKS;
pub(super) use tools::{tool_penalty_key, tool_target_hint_from_args_json, tool_target_hint_from_payload};
use crate::core::tool_registry::ToolRegistry;
use crate::db::Db;
use crate::models::{
    ControllerGate,
    ControllerState,
    ResponseOrigin,
    Settings,
    ToolCall,
    ToolCallFunction,
    WorkingMemoryBlock,
    WorkspaceHypothesis,
    WorkspaceFieldMeta,
    WorkspaceListItemMeta,
};

pub struct Kernel {
    db: Arc<Db>,
    model_client: Arc<ModelClient>,
    app_handle: AppHandle,
    tools: ToolRegistry,
    monologue_lock: Mutex<()>,
    proaction_lock: Mutex<()>,
    run_locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    introspection_cache: Mutex<HashMap<String, IntrospectionCacheEntry>>,
    evidence_validation_cache: Mutex<HashMap<String, EvidenceValidationCacheEntry>>,
    wave_field: Arc<RwLock<WaveField>>,
    wave_projector: Mutex<WaveProjector>,
}

impl Kernel {
    pub(crate) fn model_client(&self) -> Arc<ModelClient> {
        self.model_client.clone()
    }
}
