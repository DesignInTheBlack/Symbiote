use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub(crate) struct RegistryMeta {
    pub name: String,
    pub version: i64,
    pub hash: String,
    pub compatibility: String,
}

#[derive(Clone)]
pub(crate) struct ToolDispatchRequest {
    pub action_id: String,
    pub tool_name: String,
    pub args_json: String,
    pub plan_step_id: Option<String>,
}

pub(crate) struct ToolExecutionResult {
    pub tool_name: String,
    pub output: String,
    pub is_error: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ToolGateDecision {
    pub allowed: bool,
    pub reason: Option<String>,
    pub detail: Option<String>,
    pub failure_kind: Option<String>,
}

impl ToolGateDecision {
    pub fn allow() -> Self {
        Self {
            allowed: true,
            reason: None,
            detail: None,
            failure_kind: None,
        }
    }

    pub fn block(reason: &str, detail: Option<String>, failure_kind: Option<&str>) -> Self {
        Self {
            allowed: false,
            reason: Some(reason.to_string()),
            detail,
            failure_kind: failure_kind.map(|s| s.to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolFailurePenalty {
    #[serde(default)]
    pub count: i64,
    #[serde(default)]
    pub last_failure_at: Option<String>,
    #[serde(default)]
    pub penalty_until: Option<String>,
}
