use std::collections::{HashMap, HashSet};
use std::sync::Mutex as StdMutex;

use once_cell::sync::Lazy;
use regex::{Regex, RegexSet};

pub const MODE_DWELL_SECS: i64 = 120;
pub const MEDIUM_LAYER_INTERVAL_SECS: i64 = 600;
pub const SLOW_LAYER_INTERVAL_SECS: i64 = 3600;
pub const MAX_TOOL_RETRIES: i64 = 1;
pub const MAX_TOOL_LOOPS: usize = 3;
pub const TOOL_FAILURE_KIND_PLANNING: &str = "planning_error";
pub const TOOL_FAILURE_KIND_EXECUTION: &str = "execution_error";
pub const TOOL_FAILURE_KIND_CANCELLED: &str = "cancelled";
pub const OUTCOME_HISTORY_LIMIT: usize = 12;
pub const DEFAULT_INNER_SUMMARY_CAP: usize = 1000;
pub const SAFE_HALT_FAILURE_THRESHOLD: i64 = 6;
pub const SEMANTIC_CORE_MAX_CHARS: usize = 4000;
pub const RELEVANCE_REJECT_THRESHOLD: f32 = 0.20;
pub const RELEVANCE_WARN_THRESHOLD: f32 = 0.35;
pub const HEARTBEAT_INTERVAL_SECS: i64 = 60;
pub const PERF_WARN_PROMPT_MS: i64 = 1500;
pub const PERF_WARN_COMMIT_MS: i64 = 1500;
pub const FAST_PATH_BUDGET_INGEST_MS: i64 = 250;
pub const FAST_PATH_BUDGET_PROMPT_MS: i64 = 1500;
pub const FAST_PATH_BUDGET_MODEL_MS: i64 = 8000;
pub const TOOL_FAILURE_PENALTY_SECS: i64 = 120;
pub const FAST_PATH_BUDGET_PARSE_MS: i64 = 800;
pub const FAST_PATH_BUDGET_EMIT_MS: i64 = 400;
pub const DREAM_INTERVAL_SECS: i64 = 900;
pub const DREAM_IDLE_MINUTES: i64 = 10;
pub const MONOLOGUE_QUIET_SECS: i64 = 300;
pub const MONOLOGUE_SURFACE_WINDOW_SECS: i64 = 120;
pub const PRIORITY_MONOLOGUE_WINDOW_SECS: i64 = 30;
pub const MONOLOGUE_DIGEST_TTL_SECS: i64 = 240;
pub const MONOLOGUE_JSON_FAILURE_WINDOW_MINS: i64 = 30;
pub const MONOLOGUE_JSON_FAILURE_THRESHOLD: i64 = 2;
pub const MONOLOGUE_JSON_COOLDOWN_SECS: i64 = 15 * 60;
pub const PROACTIVE_OVERLAP_THRESHOLD: usize = 3;
pub const CLARIFIER_OVERLAP_THRESHOLD: usize = 1;
pub const PENDING_PROMPT_STARVATION_LIMIT: i64 = 3;
pub const AUTO_SURFACE_SLA_SECS: i64 = 10;
pub const AUTO_SURFACE_MAX_AGE_SECS: i64 = 30;
pub const PENDING_PROMPT_ATTEMPT_LIMIT: i64 = 3;
pub const PENDING_PROMPT_EXPIRES_SECS: i64 = 3600;
pub const OPEN_QUESTION_ATTEMPT_LIMIT: i64 = 3;
pub const OPEN_QUESTION_EXPIRES_SECS: i64 = 3600;
pub const REDIRECT_FOCUS_PROMOTE_TURNS: i32 = 2;
pub const REDIRECT_FOCUS_ALIGN_THRESHOLD: f32 = 0.20;
pub const MONOLOGUE_INTENT_MAX_AGE_SECS: i64 = 3600;
pub const SUMMARY_ECHO_THRESHOLD: f32 = 0.72;
pub const SUMMARY_ECHO_MIN_CHARS: usize = 120;
pub const EVIDENCE_MIN: f32 = 0.60;
pub const EVIDENCE_QUALITY_MIN: f32 = 0.60;
pub const TELEMETRY_MIN: f32 = 0.50;
pub const PROACTIVE_THROTTLE_SECS: i64 = 60;
pub const PROACTIVE_COOLDOWN_SECS: i64 = 90;
pub const PROACTIVE_QUIESCENCE_SECS: i64 = 10;
pub const PROACTIVE_MEMORY_PASS_MIN_SECS: i64 = 600;
pub const SYSTEM_DUMP_BREAKER_DISABLE_TURNS: i32 = 3;
pub const EVIDENCE_FRESH_DAYS_USER: i64 = 14;
pub const FOCUS_DEMOTION_GRACE_HOURS: i64 = 48;
pub const EVIDENCE_VALIDATION_CACHE_TTL_SECS: i64 = 60;
pub const IDENTITY_OVERLAP_THRESHOLD: usize = 2;
pub const IDENTITY_VIOLATION_THRESHOLD: i64 = 3;
pub const IDENTITY_VIOLATION_WINDOW_SECS: i64 = 3600;
pub const IDENTITY_AB_MIN_TURNS: i64 = 50;
pub const IDENTITY_ENFORCE_MIN_CONF: f32 = 0.7;
pub const MONOLOGUE_TURN_OVERLAP_THRESHOLD: f32 = 0.12;
pub const FTS_MIN_TURNS: usize = 3;
pub const FTS_MAX_TURNS: usize = 6;
pub const MONOLOGUE_STATE_CHANGE_WINDOW_TICKS: i64 = 5;
pub const NOVELTY_ABSENT_K: i32 = 2;
pub const META_COG_OUTCOME_TURNS: i64 = 3;
pub const META_COG_CYCLE_WINDOW_TURNS: i64 = 2;
pub const META_COG_OUTCOME_TIMEOUT_SECS: i64 = 120;
pub const META_COG_OUTCOME_STREAK_LIMIT: i32 = 3;
pub const META_COG_COOLDOWN_SECS: i64 = 60;
pub const USER_ATTRIBUTION_ALIGN_THRESHOLD: f32 = 0.22;
pub const RESEARCH_TOOLS: [&str; 1] = ["web_lookup"];
pub const REGISTRY_SCHEMA_VERSION: i64 = 1;
#[cfg(test)]
pub const AUTO_MEMORY_CONFIDENCE_THRESHOLD: f32 = 0.75;
pub const RESIDUAL_SHADOW_MAX_IMPACT_PCT: f64 = 5.0;
pub const RESIDUAL_SHADOW_STABLE_WINDOW_CYCLES: i64 = 100;
pub const INTERNAL_STATE_MAP_MIN_OBSERVATIONS: i64 = 100;

pub static USER_ATTRIBUTION_BASE_SET: Lazy<RegexSet> = Lazy::new(|| {
    RegexSet::new([
        r"you said",
        r"you mentioned",
        r"you noted",
        r"you wrote",
        r"you told",
        r"you asked",
        r"you explained",
        r"as you said",
        r"as you mentioned",
        r"as you noted",
        r"user said",
        r"user mentioned",
        r"user noted",
        r"user wrote",
        r"user asked",
    ])
    .expect("user attribution regex set")
});

pub static STATE_CLAIM_SET: Lazy<RegexSet> = Lazy::new(|| {
    RegexSet::new([
        r"current\s*focus",
        r"current_focus",
        r"self[-_ ]?state",
        r"internal\s*state",
        r"internal\s*metrics",
        r"workspace\s*state",
        r"controller\s*state",
        r"controller\s*confidence",
        r"gate\s*decision",
        r"pending\s*prompt",
        r"last[_ ]internal[_ ]thought",
        r"inner\s*summary",
    ])
    .expect("state claim regex set")
});

pub static STANCE_CLAIM_SET: Lazy<RegexSet> = Lazy::new(|| {
    RegexSet::new([
        r"\bi\s+think\b",
        r"\bi\s+am\s+thinking\b",
        r"\bi'?m\s+thinking\b",
        r"\bi\s+feel\b",
        r"\bi\s+am\s+feeling\b",
        r"\bi'?m\s+feeling\b",
        r"\bi\s+notice\b",
        r"\bi\s+am\s+aware\b",
        r"\bi'?m\s+aware\b",
        r"\bi\s+am\s+uncertain\b",
        r"\bi'?m\s+uncertain\b",
        r"\bi\s+am\s+unsure\b",
        r"\bi'?m\s+unsure\b",
        r"\bi\s+do\s+not\s+have\s+thoughts\b",
        r"\bi\s+don'?t\s+have\s+thoughts\b",
        r"\bi\s+do\s+not\s+have\s+feelings\b",
        r"\bi\s+don'?t\s+have\s+feelings\b",
        r"\bi\s+do\s+not\s+have\s+consciousness\b",
        r"\bi\s+don'?t\s+have\s+consciousness\b",
    ])
    .expect("stance claim regex set")
});

pub static STATE_CLAIM_SET_EXPANDED: Lazy<RegexSet> = Lazy::new(|| {
    RegexSet::new([
        r"internal\s*signal",
        r"internal\s*module",
        r"internal\s*phase",
        r"internal\s*policy",
        r"internal\s*rule",
        r"model\s*state",
        r"memory\s*state",
        r"memory\s*control",
        r"memory\s*gate",
        r"telemetry\s*state",
        r"telemetry\s*signal",
        r"diagnostic\s*state",
        r"system\s*state",
        r"kernel\s*state",
        r"kernel\s*signal",
        r"kernel\s*policy",
        r"kernel\s*decision",
    ])
    .expect("state claim regex set expanded")
});

pub static FEEDBACK_BUNDLE_CLAIM_SET: Lazy<RegexSet> = Lazy::new(|| {
    RegexSet::new([
        r"feedback\s*bundle",
        r"last\s*turn\s*outcome",
        r"policy\s*adherence",
        r"evidence\s*coverage",
        r"\bconfidence\b",
        r"\buncertainty\b",
        r"qualia",
        r"gate\s*notice",
        r"gate\s*reasons",
        r"context\s*tag",
        r"user\s*intent\s*summary",
    ])
    .expect("feedback bundle claim regex set")
});

pub static FEELINGS_DEFLECTION_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)(^|[\.\?!])\s*i\s+(do\s+not|don't|donâ€™t)\s+(have|experience)\s+(subjective\s+)?feelings[^\.\?!]*(\.|\?|!|$)",
    )
    .expect("feelings deflection regex")
});

pub static USER_ATTRIBUTION_NAME_CACHE: Lazy<StdMutex<HashMap<String, RegexSet>>> =
    Lazy::new(|| StdMutex::new(HashMap::new()));

pub static USER_ATTRIBUTION_REWRITE_PATTERNS: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
    vec![
        (Regex::new("(?i)\\byou said\\b").unwrap(), "I inferred"),
        (Regex::new("(?i)\\byou mentioned\\b").unwrap(), "I inferred"),
        (Regex::new("(?i)\\byou noted\\b").unwrap(), "I inferred"),
        (Regex::new("(?i)\\byou wrote\\b").unwrap(), "I inferred"),
        (Regex::new("(?i)\\byou told\\b").unwrap(), "I inferred"),
        (Regex::new("(?i)\\byou asked\\b").unwrap(), "I inferred"),
        (Regex::new("(?i)\\bas you said\\b").unwrap(), "Based on your last message, I inferred"),
        (Regex::new("(?i)\\bas you mentioned\\b").unwrap(), "Based on your last message, I inferred"),
        (Regex::new("(?i)\\bas you noted\\b").unwrap(), "Based on your last message, I inferred"),
        (Regex::new("(?i)\\buser said\\b").unwrap(), "I inferred"),
        (Regex::new("(?i)\\buser mentioned\\b").unwrap(), "I inferred"),
        (Regex::new("(?i)\\buser noted\\b").unwrap(), "I inferred"),
        (Regex::new("(?i)\\buser wrote\\b").unwrap(), "I inferred"),
        (Regex::new("(?i)\\buser asked\\b").unwrap(), "I inferred"),
    ]
});

pub static MONOLOGUE_DESCRIPTOR_ALLOWLIST: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "focus",
        "uncertainty",
        "urgency",
        "confidence",
        "curiosity",
        "tension",
        "clarity",
        "calm",
        "speculative",
    ]
    .into_iter()
    .collect()
});

pub static NON_IGNITION_LANGUAGE_PATTERNS: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
    vec![
        (Regex::new("(?i)\\bi noticed\\b").unwrap(), "I inferred"),
        (Regex::new("(?i)\\bi am aware\\b").unwrap(), "I infer"),
        (Regex::new("(?i)\\bi was aware\\b").unwrap(), "I inferred"),
        (Regex::new("(?i)\\bi feel\\b").unwrap(), "I register"),
        (Regex::new("(?i)\\bi felt\\b").unwrap(), "I registered"),
        (Regex::new("(?i)\\bi experience\\b").unwrap(), "I infer"),
        (Regex::new("(?i)\\bi experienced\\b").unwrap(), "I inferred"),
        (Regex::new("(?i)\\bi am conscious\\b").unwrap(), "I am operational"),
        (Regex::new("(?i)\\bqualia\\b").unwrap(), "tags"),
    ]
});
