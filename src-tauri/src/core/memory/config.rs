// ICS v4.1 Configuration and Invariants

// I5: Evidence growth bounds
pub const MAX_EVIDENCE_EVENTS_PER_BELIEF: usize = 50;
pub const KEEP_NEWEST_EVIDENCE: usize = 20;
pub const KEEP_TOP_WEIGHTED_EVIDENCE: usize = 20;
pub const KEEP_UNIQUE_SNIPPETS: usize = 10;

// I4: Answer confidence parameters
pub const ANSWER_CONFIDENCE_K: f32 = 1.0;

// I3: Time bucket comparisons (ordering implied in code)

// §6.1 Entity Resolution
pub const RESOLVE_THRESHOLD: f32 = 0.6;
pub const MARGIN_THRESHOLD: f32 = 0.15;
pub const FTS_CANDIDATE_LIMIT: usize = 20;

// §10.4 Retrieval Bounds
pub const MAX_ANCHORS: usize = 5;
pub const MAX_HOPS: usize = 4;
pub const ABOLUTE_MAX_HOPS: usize = 5;
pub const MAX_NODES: usize = 50;
pub const MAX_EXPANSIONS_PER_NODE: usize = 3;
pub const MAX_EXPANSIONS_FOR_ANCHOR: usize = 8;
pub const MAX_FRONTIER: usize = 64;
pub const MAX_RECALLED_FACTS: usize = 38;
pub const MAX_RECALLED_RELS: usize = 25;
pub const MAX_MEMORY_CONTEXT_CHARS: usize = 12000;

// §10.7 Selection
pub const UNCERTAIN_MARGIN: f32 = 0.10;
pub const RELATION_SCORE_THRESHOLD: f32 = 0.15;

// §1.5.1 Token/Role Alias Guardrails
pub const MIN_PROPOSED_EXPAND: usize = 3;
pub const MAX_ALIAS_EXPANSIONS: usize = 6;
pub const CONFIRMED_EVIDENCE_THRESHOLD: usize = 10;

// Rel type alias guardrails (Phase 1)
pub const REL_TYPE_ALIAS_MIN_CONFIDENCE: f32 = 0.85;
// Rel type catalog (Phase 2)
pub const REL_TYPE_SIMILARITY_THRESHOLD: f32 = 0.92;
pub const REL_TYPE_PROMOTE_MIN_EDGES: usize = 3;

// §1.8 Entity Sketch Cache
pub const SKETCH_MAX_NEIGHBORS: usize = 50;
pub const SKETCH_MAX_TOKENS: usize = 30;
pub const HUB_DEGREE_THRESHOLD: u32 = 500;

// §14 Working Set
pub const WORKING_SET_ENTITY_SCORE_BOOST: f32 = 0.2;
pub const WORKING_SET_BELIEF_SCORE_BOOST: f32 = 0.1;
pub const WORKING_SET_MAX_ENTITIES: usize = 75;
pub const WORKING_SET_MAX_BELIEFS: usize = 120;

// §9.3 Evidence Weights
pub const SOURCE_WEIGHT_USER: f32 = 1.0;
pub const SOURCE_WEIGHT_TOOL: f32 = 0.9;
pub const SOURCE_WEIGHT_SYSTEM: f32 = 0.7;
pub const SOURCE_WEIGHT_INFERENCE: f32 = 0.4;
pub const CONFIDENCE_ALPHA: f32 = 0.15;

// Salience weighting (deterministic)
pub const SALIENCE_WEIGHT_SELF_CORE: f32 = 1.0;
pub const SALIENCE_WEIGHT_OBSERVATION: f32 = 0.8;
pub const SALIENCE_WEIGHT_USER: f32 = 0.6;
pub const SALIENCE_WEIGHT_INFERENCE: f32 = 0.4;
pub const SALIENCE_EVIDENCE_SCALE: f32 = 0.15;
pub const SALIENCE_MIN: f32 = 0.1;
pub const SALIENCE_MAX: f32 = 3.0;
pub const SALIENCE_HALF_LIFE_HOURS: f32 = 168.0; // 7 days
