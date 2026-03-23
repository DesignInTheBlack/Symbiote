use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Enums

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    Person,
    Place,
    Work,
    Event,
    Concept,
    Project,
    System,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionState {
    Normal,
    Tentative,
    DoNotMerge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeliefKind {
    Fact,
    Rel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeliefStatus {
    Active,
    Inactive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Polarity {
    Assert,
    Deny,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cardinality {
    One,
    ManySet,
    TimeSeries,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadPolicy {
    Current,
    List,
    History,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Global,
    Session, // Should probably carry session ID if needed, but for now simple
    Project(String),
    Context(String),
    #[serde(rename = "self")]
    SelfScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeBucketKind {
    Atemporal,
    Day,
    Range,
    Exact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkType {
    Supports,
    Contradicts,
    Supersedes,
    DerivedFrom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictStatus {
    Open,
    Resolved,
    Archived,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    User,
    Tool,
    System,
    Inference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AliasStatus {
    Proposed,
    Confirmed,
}

// Structs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: i64,
    pub label: String,
    pub label_canonical: String,
    pub entity_type: Option<EntityType>,
    pub aliases: Vec<String>,
    pub aliases_canonical: Vec<String>,
    pub keys: Vec<String>,
    pub resolution_state: ResolutionState,
    pub created_at: String, // ISO8601
    pub last_accessed_at: String,
    pub access_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactBelief {
    pub belief_id: i64,
    pub subject_entity_id: i64,
    pub key: String,
    pub value_literal: String,
    pub value_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelBelief {
    pub belief_id: i64,
    pub rel_type_id: Option<String>,
    pub rel_type: String,
    pub participants_canonical: String,
    pub anchor_signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Participant {
    pub role: String,
    pub entity_id: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefLink {
    pub from_id: i64,
    pub to_id: i64,
    pub link_type: LinkType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictSet {
    pub id: i64,
    pub topic_key: String,
    pub status: ConflictStatus,
    pub priority: String, // low, normal, high
    pub resolution_note: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefEvidenceEvent {
    pub id: i64,
    pub belief_id: i64,
    pub source_type: SourceType,
    pub source_ref: Option<String>,
    pub snippet: Option<String>,
    pub snippet_hash: Option<String>,
    pub weight: f32,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAlias {
    pub token_kind: String, // fact_key | rel_type
    pub from_token: String,
    pub to_token: String,
    pub status: AliasStatus,
    pub evidence_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleAlias {
    pub from_role: String,
    pub to_role: String,
    pub status: AliasStatus,
    pub evidence_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPolicy {
    pub token: String,
    pub cardinality: Cardinality,
    pub read_policy: ReadPolicy,
    pub allow_future_in_current: bool,
    // defaults...
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationShape {
    pub rel_type_id: Option<String>,
    pub rel_type: String,
    pub roles: Vec<String>,
    pub anchor_roles: Vec<String>,
    pub cardinality_override: Option<Cardinality>,
    pub commutative: bool,
    pub expected_arity: Option<i64>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionMapping {
    pub from_fact_key: String,
    pub to_rel_type: String,
    pub subject_role: String,
    pub value_role: String,
    pub status: String, // active | retired
    pub mapping_version: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySketch {
    pub entity_id: i64,
    pub neighbors_top: Vec<i64>,
    pub tokens_top: Vec<String>,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingSetEntry {
    pub id: i64, // EntityId or BeliefId
    pub activation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingWrite {
    pub id: i64,
    pub parsed_lines: String,
    pub candidates_json: String,
    pub status: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeEvent {
    pub id: i64,
    pub from_id: i64,
    pub to_id: i64,
    pub reason: String,
    pub is_rolled_back: bool,
    pub created_at: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Belief { // Joined representation for retrieval
    pub id: i64,
    pub kind: BeliefKind,
    pub status: BeliefStatus,
    pub score: f32,
    pub confidence: f32,
    // Fact fields
    pub key: Option<String>,
    pub value: Option<String>,
    // Rel fields
    pub rel_type: Option<String>,
    pub participants_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryIntent {
    AskCurrent,
    AskList,
    AskHistory,
    AskExplain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPacket {
    pub facts: Vec<ScoredFact>,
    pub relations: Vec<ScoredRel>,
    pub conflicts: Vec<ConflictSet>,
    pub bound_handles: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadowed_by_scope_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dropped_by_scope_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub episodic_events: Option<Vec<crate::models::EpisodicEvent>>,
    /// Debug log (only populated when debug=true in retrieval)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug_log: Option<crate::core::memory::debug::RetrievalDebugLog>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimOutcome {
    pub claim_id: String,
    pub status: String,
    pub reason: Option<String>,
    pub scope: String,
    pub session_id: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredFact {
    pub id: i64,
    pub entity_id: i64, // NEW: Added for efficient history lookup
    pub entity_label: String,
    pub key: String,
    pub topic_key: String,
    pub value: String,
    pub confidence: f32,
    pub score: f32,
    // Time Model (§4)
    pub time_bucket_kind: String,
    pub time_bucket_value: Option<String>,
    // For I7 tie-break and negation
    pub signature_hash: String,
    pub polarity: String,
    // For §3.1 Scope Shadowing
    pub scope: String,
    // NEW: System time injection
    pub observed_at: Option<String>,
    pub observed_at_formatted: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredRel {
    pub id: i64,
    pub rel_type: String,
    pub participants: Vec<ScoredRelParticipant>,
    pub direction: Option<String>,
    pub order_is_trusted: bool,
    pub confidence: f32,
    pub score: f32,
    // Time Model (§4)
    pub time_bucket_kind: String,
    pub time_bucket_value: Option<String>,
    // For I7 tie-break and negation
    pub signature_hash: String,
    pub polarity: String,
    // For §3.1 Scope Shadowing
    pub scope: String,
    // NEW: System time injection
    pub observed_at: Option<String>,
    pub observed_at_formatted: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredRelParticipant {
    pub role: String,
    pub entity_id: i64,
    pub entity_label: String,
}
