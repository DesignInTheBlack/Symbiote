use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum KernelMode {
    Play,
    Work,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskPhase {
    Running,
    AwaitingUser,
    ResolvingWithDefaults,
    Aborting,
    Terminated,
}

impl Default for TaskPhase {
    fn default() -> Self {
        TaskPhase::Running
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StanceState {
    pub stance: String,
    pub verbosity_target: String,
    pub initiative_level: String,
    pub tool_preference: String,
    pub updated_at: Option<String>,
}

impl Default for StanceState {
    fn default() -> Self {
        Self {
            stance: "execute".to_string(),
            verbosity_target: "medium".to_string(),
            initiative_level: "medium".to_string(),
            tool_preference: "normal".to_string(),
            updated_at: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub action_type: String,
    pub success: bool,
    pub observations: String,
    pub source: String,
    #[serde(default)]
    pub failure_kind: Option<String>,
    #[serde(default)]
    pub target_key: Option<String>,
    pub action_id: Option<String>,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadHandle {
    pub thread_id: String,
    pub goal: String,
    pub status: String,
    pub spawned_at: String,
    pub depth: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetaCogPending {
    pub kind: String,
    pub anchor: String,
    pub accepted_at: String,
    pub turn_index: i64,
    pub source: String,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum UserFeedbackKind {
    Clarify,
    Pushback,
    FollowUp,
    Agree,
    Disengage,
}

impl UserFeedbackKind {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            UserFeedbackKind::Clarify => "clarify",
            UserFeedbackKind::Pushback => "pushback",
            UserFeedbackKind::FollowUp => "follow_up",
            UserFeedbackKind::Agree => "agree",
            UserFeedbackKind::Disengage => "disengage",
        }
    }
}
