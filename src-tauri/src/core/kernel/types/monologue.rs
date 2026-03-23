use super::Candidate;

#[derive(Debug, Clone)]
pub(crate) struct MonologueOutput {
    pub turns: Vec<MonologueTurn>,
    pub last_message: Option<String>,
    pub dialogue_messages: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MonologueStream {
    FreeThought,
    Deliberation,
}

pub(crate) enum MonologueDue {
    Due,
    Skipped(&'static str),
}

impl MonologueStream {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            MonologueStream::FreeThought => "FTS",
            MonologueStream::Deliberation => "DS",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MonologueTurn {
    pub entry: crate::models::InnerMonologueEntry,
    pub candidates: Vec<Candidate>,
    pub blocked_candidates: Vec<BlockedCandidate>,
}

#[derive(Debug, Clone)]
pub(crate) struct BlockedCandidate {
    pub candidate: Candidate,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub(crate) struct MonologueDigest {
    pub text: Option<String>,
    pub source: String,
    pub age_secs: Option<i64>,
    pub entry_id: Option<String>,
    pub stream: Option<String>,
    pub stale: bool,
}
