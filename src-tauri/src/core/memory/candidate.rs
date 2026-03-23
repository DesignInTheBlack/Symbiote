use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use std::collections::HashSet;

use crate::core::memory::canonical::normalize_rel_type;
use crate::core::memory::rel_vocab::is_canonical_relation;
use crate::core::memory::dsl::{DslStatement, FactStmt, Ref, RelDirection, RelStmt};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CandidateKind {
    Fact,
    Relation,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CandidateSource {
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCandidate {
    pub kind: CandidateKind,
    pub key: Option<String>,
    pub value: Option<String>,
    pub rel_type: Option<String>,
    pub participants: Vec<(String, Ref)>,
    pub source: CandidateSource,
    pub signals: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateScore {
    pub total: f32,
    pub novelty: f32,
    pub durability: f32,
    pub relevance: f32,
    pub evidence: f32,
    pub relationship: f32,
    pub reasons: Vec<String>,
}

pub const MEMORY_CANDIDATE_TRIGGER_THRESHOLD: f32 = 0.60;
pub const MEMORY_FALSE_POSITIVE_MAX: f32 = 0.25;

// Canonical relation vocabulary lives in rel_vocab.rs

fn is_commutative_relation(rel_type: &str) -> bool {
    matches!(
        rel_type,
        "works_with" | "collaborates_with" | "friends" | "spouse_of" | "sibling_of"
    )
}

static ACK_TERMS: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        "ok",
        "okay",
        "yes",
        "yep",
        "yeah",
        "thanks",
        "thank you",
        "sure",
        "alright",
        "right",
        "cool",
        "got it",
        "understood",
        "fine",
        "k",
    ]
});

static NAME_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:my name is|call me|i am called|i'm called)\s+([^.!?\n]+)").unwrap()
});
static LIKE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:i\s+(?:really\s+)?(?:like|love))\s+([^.!?\n]+)").unwrap()
});
static PREFER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:i\s+(?:really\s+)?prefer)\s+([^.!?\n]+)").unwrap()
});
static DISLIKE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:i\s+(?:really\s+)?(?:dislike|hate))\s+([^.!?\n]+)").unwrap()
});
static ROLE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:i\s+am\s+a|i\s+am\s+an|i'm\s+a|i'm\s+an)\s+([^.!?\n]+)").unwrap()
});
static WORK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:i\s+work\s+at|i\s+work\s+for)\s+([^.!?\n]+)").unwrap()
});
static LIVE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:i\s+live\s+in|i'm\s+from|i\s+am\s+from)\s+([^.!?\n]+)").unwrap()
});
static CREATE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:i\s+(?:created|built|made|designed)\s+you)\b").unwrap()
});
static OWN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:i\s+own(?:\s+the)?)\s+([^.!?\n]+)").unwrap()
});
static PROJECT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:i\s+am\s+working\s+on|i'm\s+working\s+on|i\s+am\s+building|i'm\s+building|i\s+am\s+making|i'm\s+making)\s+([^.!?\n]+)").unwrap()
});
static PROJECT_IS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:my|our)\s+project\s+(?:is|called|named)\s+([^.!?\n]+)").unwrap()
});
static COLLAB_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:i\s+(?:work|collaborate)\s+with)\s+([^.!?\n]+)").unwrap()
});
static FOUNDED_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:i\s+(?:founded|started|created|built|made|designed))\s+([^.!?\n]+)").unwrap()
});
static CREATOR_ROLE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:i\s+am|i'm)\s+(?:the\s+)?(?:creator|founder|builder)\s+of\s+([^.!?\n]+)").unwrap()
});
static MEMBER_OF_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:i\s+am\s+part\s+of|i'm\s+part\s+of|i\s+belong\s+to)\s+([^.!?\n]+)").unwrap()
});
static COMPANY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\bmy\s+(?:company|startup|organization|org|team)\s+(?:is|called|named)\s+([^.!?\n]+)").unwrap()
});
static ROLE_AT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:i\s+am|i'm)\s+the\s+([^.!?\n]+?)\s+at\s+([^.!?\n]+)").unwrap()
});
static GOAL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:my\s+goal\s+is\s+to|i\s+(?:want|plan|need|intend)\s+to)\s+([^.!?\n]+)").unwrap()
});
static COMMIT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:i\s+will|i\s+promise\s+to|i\s+commit\s+to)\s+([^.!?\n]+)").unwrap()
});
static PET_NAME_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\bmy\s+(dog|cat|pet|puppy|kitten)\s*(?:'s)?\s*name\s+is\s+([^.!?\n]+)").unwrap()
});
static FAMILY_NAME_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\bmy\s+(mother|mom|father|dad|parent|sister|brother|sibling|wife|husband|spouse|partner|son|daughter|child)\s*(?:'s)?\s*name\s+is\s+([^.!?\n]+)")
        .unwrap()
});
static FAMILY_IS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b([A-Z][a-zA-Z'\\-]+)\s+is\s+my\s+(mother|mom|father|dad|parent|sister|brother|sibling|wife|husband|spouse|partner|son|daughter|child)\b")
        .unwrap()
});
static SYSTEM_KEYWORD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(symbiote|kernel|scheduler|monologue|memory|tool|tools|arbitration|prompt|gate|gating|anchor|self-model|telemetry|ui|frontend|backend|db|database)\b")
        .unwrap()
});
static PROPER_NOUN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b[A-Z][a-zA-Z0-9_\\-]{2,}\b").unwrap());
static RELATION_VERB_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(uses|depends on|depends_on|is|are|requires|blocks|improves)\b").unwrap()
});

fn normalize_family_role(role: &str) -> String {
    match role.to_lowercase().as_str() {
        "mom" => "mother".to_string(),
        "dad" => "father".to_string(),
        "spouse" | "partner" => "spouse".to_string(),
        "sister" | "brother" => "sibling".to_string(),
        other => other.to_string(),
    }
}

fn build_family_candidate(role_raw: &str, name: &str, user_ref: &Ref) -> Option<MemoryCandidate> {
    let role = normalize_family_role(role_raw);
    let name = trimmed_capture(name)?;
    let relative = Ref::Label(name);
    match role.as_str() {
        "mother" => Some(MemoryCandidate {
            kind: CandidateKind::Relation,
            key: None,
            value: None,
            rel_type: Some("mother_of".to_string()),
            participants: vec![
                ("mother".to_string(), relative),
                ("child".to_string(), user_ref.clone()),
            ],
            source: CandidateSource::User,
            signals: vec!["family".to_string()],
        }),
        "father" => Some(MemoryCandidate {
            kind: CandidateKind::Relation,
            key: None,
            value: None,
            rel_type: Some("father_of".to_string()),
            participants: vec![
                ("father".to_string(), relative),
                ("child".to_string(), user_ref.clone()),
            ],
            source: CandidateSource::User,
            signals: vec!["family".to_string()],
        }),
        "parent" => Some(MemoryCandidate {
            kind: CandidateKind::Relation,
            key: None,
            value: None,
            rel_type: Some("parent_of".to_string()),
            participants: vec![
                ("parent".to_string(), relative),
                ("child".to_string(), user_ref.clone()),
            ],
            source: CandidateSource::User,
            signals: vec!["family".to_string()],
        }),
        "spouse" | "husband" | "wife" => Some(MemoryCandidate {
            kind: CandidateKind::Relation,
            key: None,
            value: None,
            rel_type: Some("spouse_of".to_string()),
            participants: vec![
                ("spouse".to_string(), relative),
                ("spouse".to_string(), user_ref.clone()),
            ],
            source: CandidateSource::User,
            signals: vec!["family".to_string()],
        }),
        "sibling" => Some(MemoryCandidate {
            kind: CandidateKind::Relation,
            key: None,
            value: None,
            rel_type: Some("sibling_of".to_string()),
            participants: vec![
                ("sibling".to_string(), relative),
                ("sibling".to_string(), user_ref.clone()),
            ],
            source: CandidateSource::User,
            signals: vec!["family".to_string()],
        }),
        "son" | "daughter" | "child" => Some(MemoryCandidate {
            kind: CandidateKind::Relation,
            key: None,
            value: None,
            rel_type: Some("parent_of".to_string()),
            participants: vec![
                ("parent".to_string(), user_ref.clone()),
                ("child".to_string(), relative),
            ],
            source: CandidateSource::User,
            signals: vec!["family".to_string()],
        }),
        _ => None,
    }
}

fn trimmed_capture(input: &str) -> Option<String> {
    let trimmed = input.trim().trim_matches('"').trim_matches('\'').trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn is_high_signal_short(input: &str) -> bool {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return false;
    }
    let has_system_keyword = SYSTEM_KEYWORD_RE.is_match(trimmed);
    let has_proper_noun = PROPER_NOUN_RE.is_match(trimmed);
    let has_relation_verb = RELATION_VERB_RE.is_match(trimmed);
    (has_system_keyword && has_proper_noun) || has_relation_verb
}

pub fn should_extract_from_user(input: &str) -> bool {
    let trimmed_raw = input.trim();
    let trimmed = trimmed_raw.to_lowercase();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.len() < 6 {
        return is_high_signal_short(trimmed_raw);
    }
    if ACK_TERMS.iter().any(|term| trimmed == *term) {
        return false;
    }
    let words = trimmed.split_whitespace().count();
    if words < 3 {
        return is_high_signal_short(trimmed_raw);
    }
    true
}

fn should_extract_from_assistant(input: &str) -> bool {
    let trimmed = input.trim().to_lowercase();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.len() < 12 {
        return false;
    }
    if !(trimmed.contains("remember") || trimmed.contains("noted") || trimmed.contains("got it")) {
        return false;
    }
    true
}

fn format_ref_for_prompt(r: &Ref) -> String {
    match r {
        Ref::Handle(h) => format!("${}", h),
        Ref::Label(l) => format!("#{}", l),
        Ref::Filter(l, f) => format!("#{}:{}", l, f),
        Ref::Name(n) => format!("\"{}\"", n),
    }
}

pub fn format_candidates_for_prompt(candidates: &[MemoryCandidate]) -> String {
    if candidates.is_empty() {
        return "None".to_string();
    }
    let mut lines = Vec::new();
    for candidate in candidates.iter().take(8) {
        match candidate.kind {
            CandidateKind::Fact => {
                let key = candidate.key.as_deref().unwrap_or("fact");
                let value = candidate.value.as_deref().unwrap_or("");
                let value = value.replace('"', "'");
                let subject = candidate
                    .participants
                    .first()
                    .map(|p| p.1.clone())
                    .unwrap_or_else(|| Ref::Handle("user".to_string()));
                lines.push(format!("{}:{} = \"{}\"", format_ref_for_prompt(&subject), key, value));
            }
            CandidateKind::Relation => {
                let Some(rel_type_raw) = candidate.rel_type.as_ref() else { continue };
                let rel_type = normalize_rel_type(rel_type_raw);
                if !is_canonical_relation(&rel_type) {
                    continue;
                }
                let parts: Vec<String> = candidate
                    .participants
                    .iter()
                    .map(|(role, r)| format!("{}: {}", role, format_ref_for_prompt(r)))
                    .collect();
                if candidate.participants.len() == 2 {
                    let arrow = if is_commutative_relation(&rel_type) { "<->" } else { "->" };
                    let left = parts.get(0).cloned().unwrap_or_default();
                    let right = parts.get(1).cloned().unwrap_or_default();
                    lines.push(format!("{}({} {} {})", rel_type, left, arrow, right));
                } else {
                    lines.push(format!("{}({})", rel_type, parts.join(", ")));
                }
            }
        }
    }
    if lines.is_empty() {
        "None".to_string()
    } else {
        lines.join("\n")
    }
}

fn extract_from_text(text: &str, source: CandidateSource, user_ref: &Ref, assistant_ref: &Ref) -> Vec<MemoryCandidate> {
    let mut out = Vec::new();

    if let Some(cap) = NAME_RE.captures(text) {
        if let Some(name) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Fact,
                key: Some("name".to_string()),
                value: Some(name),
                rel_type: None,
                participants: vec![(String::new(), user_ref.clone())],
                source,
                signals: vec!["identity".to_string()],
            });
        }
    }

    if let Some(cap) = LIKE_RE.captures(text) {
        if let Some(pref) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Relation,
                key: None,
                value: None,
                rel_type: Some("likes".to_string()),
                participants: vec![
                    ("person".to_string(), user_ref.clone()),
                    ("object".to_string(), Ref::Label(pref)),
                ],
                source,
                signals: vec!["preference".to_string()],
            });
        }
    }

    if let Some(cap) = PREFER_RE.captures(text) {
        if let Some(pref) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Relation,
                key: None,
                value: None,
                rel_type: Some("prefers".to_string()),
                participants: vec![
                    ("person".to_string(), user_ref.clone()),
                    ("object".to_string(), Ref::Label(pref)),
                ],
                source,
                signals: vec!["preference".to_string()],
            });
        }
    }

    if let Some(cap) = DISLIKE_RE.captures(text) {
        if let Some(pref) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Relation,
                key: None,
                value: None,
                rel_type: Some("dislikes".to_string()),
                participants: vec![
                    ("person".to_string(), user_ref.clone()),
                    ("object".to_string(), Ref::Label(pref)),
                ],
                source,
                signals: vec!["preference".to_string()],
            });
        }
    }

    if let Some(cap) = ROLE_RE.captures(text) {
        if let Some(role) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Fact,
                key: Some("role".to_string()),
                value: Some(role),
                rel_type: None,
                participants: vec![(String::new(), user_ref.clone())],
                source,
                signals: vec!["profile".to_string()],
            });
        }
    }

    if let Some(cap) = WORK_RE.captures(text) {
        if let Some(work) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Relation,
                key: None,
                value: None,
                rel_type: Some("works_at".to_string()),
                participants: vec![
                    ("person".to_string(), user_ref.clone()),
                    ("work".to_string(), Ref::Label(work)),
                ],
                source,
                signals: vec!["work".to_string()],
            });
        }
    }

    if let Some(cap) = LIVE_RE.captures(text) {
        if let Some(place) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Relation,
                key: None,
                value: None,
                rel_type: Some("lives_in".to_string()),
                participants: vec![
                    ("person".to_string(), user_ref.clone()),
                    ("place".to_string(), Ref::Label(place)),
                ],
                source,
                signals: vec!["location".to_string()],
            });
        }
    }

    if CREATE_RE.is_match(text) {
        out.push(MemoryCandidate {
            kind: CandidateKind::Relation,
            key: None,
            value: None,
            rel_type: Some("created_by".to_string()),
            participants: vec![
                ("creator".to_string(), user_ref.clone()),
                ("created".to_string(), assistant_ref.clone()),
            ],
            source,
            signals: vec!["relationship".to_string(), "creator".to_string()],
        });
    }

    for cap in FAMILY_NAME_RE.captures_iter(text) {
        if let (Some(role), Some(name)) = (cap.get(1), cap.get(2)) {
            if let Some(candidate) = build_family_candidate(role.as_str(), name.as_str(), &user_ref) {
                let mut candidate = candidate;
                candidate.source = source;
                out.push(candidate);
            }
        }
    }

    for cap in FAMILY_IS_RE.captures_iter(text) {
        if let (Some(name), Some(role)) = (cap.get(1), cap.get(2)) {
            if let Some(candidate) = build_family_candidate(role.as_str(), name.as_str(), &user_ref) {
                let mut candidate = candidate;
                candidate.source = source;
                out.push(candidate);
            }
        }
    }

    for cap in PET_NAME_RE.captures_iter(text) {
        if let Some(name) = cap.get(2).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Relation,
                key: None,
                value: None,
                rel_type: Some("owns".to_string()),
                participants: vec![
                    ("owner".to_string(), user_ref.clone()),
                    ("object".to_string(), Ref::Label(name)),
                ],
                source,
                signals: vec!["pet".to_string()],
            });
        }
    }

    if let Some(cap) = OWN_RE.captures(text) {
        if let Some(obj) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Relation,
                key: None,
                value: None,
                rel_type: Some("owns".to_string()),
                participants: vec![
                    ("owner".to_string(), user_ref.clone()),
                    ("object".to_string(), Ref::Label(obj)),
                ],
                source,
                signals: vec!["ownership".to_string()],
            });
        }
    }

    if let Some(cap) = PROJECT_RE.captures(text) {
        if let Some(project) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Relation,
                key: None,
                value: None,
                rel_type: Some("project_member_of".to_string()),
                participants: vec![
                    ("member".to_string(), user_ref.clone()),
                    ("project".to_string(), Ref::Label(project)),
                ],
                source,
                signals: vec!["project".to_string()],
            });
        }
    }

    if let Some(cap) = PROJECT_IS_RE.captures(text) {
        if let Some(project) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Relation,
                key: None,
                value: None,
                rel_type: Some("project_member_of".to_string()),
                participants: vec![
                    ("member".to_string(), user_ref.clone()),
                    ("project".to_string(), Ref::Label(project)),
                ],
                source,
                signals: vec!["project".to_string()],
            });
        }
    }

    if let Some(cap) = COLLAB_RE.captures(text) {
        if let Some(person) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Relation,
                key: None,
                value: None,
                rel_type: Some("collaborates_with".to_string()),
                participants: vec![
                    ("person".to_string(), user_ref.clone()),
                    ("person".to_string(), Ref::Label(person)),
                ],
                source,
                signals: vec!["collaboration".to_string()],
            });
        }
    }

    if let Some(cap) = FOUNDED_RE.captures(text) {
        if let Some(target) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Relation,
                key: None,
                value: None,
                rel_type: Some("created_by".to_string()),
                participants: vec![
                    ("creator".to_string(), user_ref.clone()),
                    ("created".to_string(), Ref::Label(target)),
                ],
                source,
                signals: vec!["creator".to_string(), "project".to_string()],
            });
        }
    }

    if let Some(cap) = CREATOR_ROLE_RE.captures(text) {
        if let Some(target) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Relation,
                key: None,
                value: None,
                rel_type: Some("created_by".to_string()),
                participants: vec![
                    ("creator".to_string(), user_ref.clone()),
                    ("created".to_string(), Ref::Label(target)),
                ],
                source,
                signals: vec!["creator".to_string()],
            });
        }
    }

    if let Some(cap) = MEMBER_OF_RE.captures(text) {
        if let Some(group) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Relation,
                key: None,
                value: None,
                rel_type: Some("member_of".to_string()),
                participants: vec![
                    ("member".to_string(), user_ref.clone()),
                    ("group".to_string(), Ref::Label(group)),
                ],
                source,
                signals: vec!["membership".to_string()],
            });
        }
    }

    if let Some(cap) = COMPANY_RE.captures(text) {
        if let Some(org) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Relation,
                key: None,
                value: None,
                rel_type: Some("member_of".to_string()),
                participants: vec![
                    ("member".to_string(), user_ref.clone()),
                    ("group".to_string(), Ref::Label(org)),
                ],
                source,
                signals: vec!["organization".to_string()],
            });
        }
    }

    if let Some(cap) = ROLE_AT_RE.captures(text) {
        if let (Some(role), Some(org)) = (cap.get(1), cap.get(2)) {
            if let Some(role) = trimmed_capture(role.as_str()) {
                out.push(MemoryCandidate {
                    kind: CandidateKind::Fact,
                    key: Some("role".to_string()),
                    value: Some(role),
                    rel_type: None,
                    participants: vec![(String::new(), user_ref.clone())],
                    source,
                    signals: vec!["profile".to_string()],
                });
            }
            if let Some(org) = trimmed_capture(org.as_str()) {
                out.push(MemoryCandidate {
                    kind: CandidateKind::Relation,
                    key: None,
                    value: None,
                    rel_type: Some("works_at".to_string()),
                    participants: vec![
                        ("person".to_string(), user_ref.clone()),
                        ("work".to_string(), Ref::Label(org)),
                    ],
                    source,
                    signals: vec!["work".to_string()],
                });
            }
        }
    }

    if let Some(cap) = GOAL_RE.captures(text) {
        if let Some(goal) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Fact,
                key: Some("goal".to_string()),
                value: Some(goal),
                rel_type: None,
                participants: vec![(String::new(), user_ref.clone())],
                source,
                signals: vec!["goal".to_string()],
            });
        }
    }

    if let Some(cap) = COMMIT_RE.captures(text) {
        if let Some(commitment) = cap.get(1).and_then(|m| trimmed_capture(m.as_str())) {
            out.push(MemoryCandidate {
                kind: CandidateKind::Fact,
                key: Some("commitment".to_string()),
                value: Some(commitment),
                rel_type: None,
                participants: vec![(String::new(), user_ref.clone())],
                source,
                signals: vec!["commitment".to_string()],
            });
        }
    }

    out
}

fn dedupe_candidates(candidates: Vec<MemoryCandidate>) -> Vec<MemoryCandidate> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for candidate in candidates {
        let key = format!(
            "{:?}|{:?}|{:?}|{:?}",
            candidate.kind,
            candidate.key,
            candidate.value,
            candidate.rel_type
        );
        if seen.insert(key) {
            out.push(candidate);
        }
    }
    out
}

pub fn extract_candidates(user_text: &str, assistant_text: &str) -> Vec<MemoryCandidate> {
    let mut out = Vec::new();
    let user_ref = Ref::Handle("user".to_string());
    let assistant_ref = Ref::Handle("assistant".to_string());

    if should_extract_from_user(user_text) {
        out.extend(extract_from_text(user_text, CandidateSource::User, &user_ref, &assistant_ref));
    }
    if should_extract_from_assistant(assistant_text) {
        out.extend(extract_from_text(assistant_text, CandidateSource::Assistant, &user_ref, &assistant_ref));
    }

    dedupe_candidates(out)
}

pub async fn score_candidates(pool: &SqlitePool, candidates: &[MemoryCandidate]) -> Vec<(MemoryCandidate, CandidateScore)> {
    let mut scored = Vec::new();
    for candidate in candidates.iter().cloned() {
        let novelty = novelty_score(pool, &candidate).await;
        let durability = durability_score(&candidate);
        let relevance = relevance_score(&candidate);
        let evidence = evidence_score(candidate.source);
        let relationship = relationship_score(&candidate);
        let total = (novelty * 0.30) + (durability * 0.25) + (relevance * 0.20) + (evidence * 0.15) + (relationship * 0.10);
        let mut reasons = Vec::new();
        if novelty < 0.5 {
            reasons.push("low_novelty".to_string());
        }
        if durability < 0.5 {
            reasons.push("low_durability".to_string());
        }
        if relevance > 0.7 {
            reasons.push("high_relevance".to_string());
        }
        if relationship > 0.7 {
            reasons.push("relationship_signal".to_string());
        }
        scored.push((
            candidate,
            CandidateScore {
                total,
                novelty,
                durability,
                relevance,
                evidence,
                relationship,
                reasons,
            },
        ));
    }
    scored
}

pub fn should_trigger(scores: &[CandidateScore], threshold: f32) -> bool {
    scores.iter().any(|s| s.total >= threshold)
}

pub fn build_fallback_statements(candidates: &[MemoryCandidate]) -> Vec<DslStatement> {
    let mut stmts = Vec::new();
    for candidate in candidates {
        let base_certainty = match candidate.source {
            CandidateSource::User => 0.6,
            CandidateSource::Assistant => 0.4,
            CandidateSource::Tool => 0.8,
        };
        match candidate.kind {
            CandidateKind::Fact => {
                let Some(key) = candidate.key.as_ref() else { continue };
                let Some(value) = candidate.value.as_ref() else { continue };
                let subject = candidate
                    .participants
                    .first()
                    .map(|p| p.1.clone())
                    .unwrap_or_else(|| Ref::Handle("user".to_string()));
                stmts.push(DslStatement::Fact(FactStmt {
                    subject,
                    key: key.clone(),
                    value: value.clone(),
                    value_quoted: true,
                    certainty: Some(base_certainty),
                    time_expr: None,
                    scope_expr: None,
                    source_ref: None,
                    polarity: "assert".to_string(),
                }));
            }
            CandidateKind::Relation => {
                let Some(rel_type_raw) = candidate.rel_type.as_ref() else { continue };
                let rel_type = normalize_rel_type(rel_type_raw);
                if !is_canonical_relation(&rel_type) {
                    continue;
                }
                if candidate.participants.is_empty() {
                    continue;
                }
                let relation_certainty = if matches!(candidate.source, CandidateSource::User) {
                    0.5
                } else {
                    base_certainty
                };
                let direction = if is_commutative_relation(&rel_type) {
                    Some(RelDirection::Bidirectional)
                } else {
                    Some(RelDirection::Directed)
                };
                stmts.push(DslStatement::Rel(RelStmt {
                    rel_type,
                    rel_type_id: None,
                    participants: candidate.participants.clone(),
                    direction,
                    certainty: Some(relation_certainty),
                    time_expr: None,
                    scope_expr: None,
                    source_ref: None,
                    polarity: "assert".to_string(),
                }));
            }
        }
    }
    stmts
}

pub fn select_top_candidate(
    scored: &[(MemoryCandidate, CandidateScore)],
    min_score: f32,
) -> Option<(MemoryCandidate, CandidateScore)> {
    scored
        .iter()
        .filter(|(_, s)| s.total >= min_score)
        .max_by(|a, b| a.1.total.partial_cmp(&b.1.total).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(c, s)| (c.clone(), s.clone()))
}

pub fn build_top_fallback_statements(
    scored: &[(MemoryCandidate, CandidateScore)],
    min_score: f32,
) -> Vec<DslStatement> {
    if let Some((candidate, _score)) = select_top_candidate(scored, min_score) {
        return build_fallback_statements(&[candidate]);
    }
    Vec::new()
}

async fn novelty_score(pool: &SqlitePool, candidate: &MemoryCandidate) -> f32 {
    match candidate.kind {
        CandidateKind::Fact => {
            let (Some(key), Some(value)) = (candidate.key.as_ref(), candidate.value.as_ref()) else {
                return 0.5;
            };
            let existing: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM ics_fact_beliefs WHERE key = ? AND value_literal = ? LIMIT 1",
            )
            .bind(key)
            .bind(value)
            .fetch_optional(pool)
            .await
            .ok()
            .flatten();
            if existing.is_some() {
                0.0
            } else {
                1.0
            }
        }
        CandidateKind::Relation => {
            let Some(rel_type) = candidate.rel_type.as_ref() else {
                return 0.5;
            };
            let rel_norm = normalize_rel_type(rel_type);
            let existing: Option<i64> = sqlx::query_scalar(
                "SELECT 1 FROM ics_rel_beliefs WHERE rel_type_norm = ? LIMIT 1",
            )
            .bind(rel_norm)
            .fetch_optional(pool)
            .await
            .ok()
            .flatten();
            if existing.is_some() {
                0.3
            } else {
                1.0
            }
        }
    }
}

fn durability_score(candidate: &MemoryCandidate) -> f32 {
    let text = candidate
        .value
        .as_deref()
        .or_else(|| candidate.rel_type.as_deref())
        .unwrap_or("");
    let lower = text.to_lowercase();
    if lower.contains("today")
        || lower.contains("yesterday")
        || lower.contains("right now")
        || lower.contains("currently")
        || lower.contains("this week")
        || lower.contains("tonight")
    {
        return 0.2;
    }
    match candidate.kind {
        CandidateKind::Relation => 0.9,
        CandidateKind::Fact => {
            if let Some(key) = candidate.key.as_ref() {
                if key == "name" || key == "role" || key == "goal" || key == "commitment" {
                    return 0.9;
                }
            }
            0.7
        }
    }
}

fn relevance_score(candidate: &MemoryCandidate) -> f32 {
    match candidate.kind {
        CandidateKind::Relation => 0.9,
        CandidateKind::Fact => {
            if let Some(key) = candidate.key.as_ref() {
                if key == "name" || key == "role" || key == "goal" || key == "commitment" {
                    return 0.8;
                }
            }
            0.6
        }
    }
}

fn evidence_score(source: CandidateSource) -> f32 {
    match source {
        CandidateSource::User => 0.8,
        CandidateSource::Assistant => 0.5,
        CandidateSource::Tool => 0.9,
    }
}

fn relationship_score(candidate: &MemoryCandidate) -> f32 {
    if candidate.kind == CandidateKind::Relation {
        if let Some(rel_type) = candidate.rel_type.as_ref() {
            let rel_norm = normalize_rel_type(rel_type);
            if matches!(
                rel_norm.as_str(),
                "created_by"
                    | "owns"
                    | "works_at"
                    | "project_member_of"
                    | "member_of"
                    | "mother_of"
                    | "father_of"
                    | "parent_of"
                    | "spouse_of"
                    | "sibling_of"
                    | "collaborates_with"
                    | "works_with"
                    | "friends"
                    | "lives_in"
            ) {
                return 1.0;
            }
        }
        return 0.6;
    }
    0.0
}
