use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReasonCategory {
    PolicyBlock,
    BudgetBlock,
    PhaseBlock,
    LatchBlock,
    EvidenceBlock,
    ToolBlock,
    TimeoutBlock,
    UnknownBlock,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StopReason {
    pub category: StopReasonCategory,
    pub subcode: String,
    #[serde(default)]
    pub contract: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StopScope {
    #[serde(default)]
    pub emit: bool,
    #[serde(default)]
    pub tools: bool,
    #[serde(default)]
    pub memory_write: bool,
    #[serde(default)]
    pub self_claims: bool,
    #[serde(default)]
    pub monologue_run: bool,
    #[serde(default)]
    pub monologue_emit: bool,
    #[serde(default)]
    pub background_jobs: bool,
}

impl StopScope {
    pub fn merge(&mut self, other: &StopScope) {
        self.emit |= other.emit;
        self.tools |= other.tools;
        self.memory_write |= other.memory_write;
        self.self_claims |= other.self_claims;
        self.monologue_run |= other.monologue_run;
        self.monologue_emit |= other.monologue_emit;
        self.background_jobs |= other.background_jobs;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopState {
    pub active: bool,
    #[serde(default)]
    pub reasons: Vec<StopReason>,
    #[serde(default)]
    pub scope: StopScope,
}

impl Default for StopState {
    fn default() -> Self {
        Self {
            active: false,
            reasons: Vec::new(),
            scope: StopScope::default(),
        }
    }
}

impl StopState {
    pub fn allow_all() -> Self {
        Self::default()
    }

    pub fn apply_reason(&mut self, reason: StopReason, scope: StopScope) {
        self.active = true;
        self.reasons.push(reason);
        self.scope.merge(&scope);
    }

    pub fn allowed_capabilities(&self) -> AllowedCapabilities {
        let mut allowed = AllowedCapabilities::default();
        if self.scope.emit {
            allowed.emit = false;
        }
        if self.scope.tools {
            allowed.tools = false;
        }
        if self.scope.memory_write {
            allowed.memory_write = false;
        }
        if self.scope.self_claims {
            allowed.self_claims = false;
        }
        if self.scope.monologue_run {
            allowed.monologue_run = false;
        }
        if self.scope.monologue_emit {
            allowed.monologue_emit = false;
        }
        if self.scope.background_jobs {
            allowed.background_jobs = false;
        }
        allowed
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllowedCapabilities {
    pub emit: bool,
    pub tools: bool,
    pub memory_write: bool,
    pub self_claims: bool,
    pub monologue_run: bool,
    pub monologue_emit: bool,
    pub background_jobs: bool,
}

impl Default for AllowedCapabilities {
    fn default() -> Self {
        Self {
            emit: true,
            tools: true,
            memory_write: true,
            self_claims: true,
            monologue_run: true,
            monologue_emit: true,
            background_jobs: true,
        }
    }
}
