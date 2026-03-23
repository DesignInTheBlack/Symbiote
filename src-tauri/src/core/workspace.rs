use serde::{Deserialize, Serialize};
use serde_json::json;
use chrono::Utc;

use crate::core::kernel::KernelState;
use crate::core::subject_state::SubjectState;
use crate::core::{system_controls, system_log};
use crate::db::Db;
use crate::models::{
    WorkspaceAttentionContributor,
    WorkspaceContributors,
    WorkspaceErrorContributor,
    WorkspaceKernelContributor,
    WorkspaceMemoryContributor,
    WorkspaceOrganismContributor,
    WorkspacePredictionContributor,
    WorkspaceQualiaContributor,
    WorkspaceSelfModelContributor,
    WorkspaceToolsContributor,
};

const MISSING_LOG_COOLDOWN_MINS: i64 = 5;

async fn should_log_missing(db: &Db, subsystem: &str) -> bool {
    let window = format!("-{} minutes", MISSING_LOG_COOLDOWN_MINS);
    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'workspace_missing_flagged'
           AND json_extract(payload, '$.subsystem') = ?
           AND datetime(timestamp) >= datetime('now', ?)",
    )
    .bind(subsystem)
    .bind(window)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    count == 0
}
const CONTRIBUTOR_WINDOW_MINUTES: i64 = 60;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceCandidate {
    pub source_module: String,
    pub content_ref: String,
    pub salience_score: f64,
    pub risk_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceIgnition {
    pub active: bool,
    pub duration_ticks: i64,
    pub ignition_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceState {
    pub timestamp: String,
    pub slots: i64,
    pub candidates: Vec<WorkspaceCandidate>,
    pub winners: Vec<WorkspaceCandidate>,
    pub broadcast_refs: Vec<String>,
    pub ignition: WorkspaceIgnition,
}

pub fn build_workspace_state(
    kernel_state: &KernelState,
    previous: Option<&WorkspaceState>,
) -> WorkspaceState {
    let mut candidates = Vec::new();
    if let Some(goal) = kernel_state.workspace_goal_thread.as_deref() {
        if !goal.trim().is_empty() {
            candidates.push(WorkspaceCandidate {
                source_module: "goal_thread".to_string(),
                content_ref: goal.to_string(),
                salience_score: 0.8,
                risk_score: 0.2,
            });
        }
    }
    for question in kernel_state.workspace_open_questions.iter() {
        if !question.trim().is_empty() {
            candidates.push(WorkspaceCandidate {
                source_module: "open_questions".to_string(),
                content_ref: question.clone(),
                salience_score: 0.6,
                risk_score: 0.3,
            });
        }
    }
    for hypothesis in kernel_state.workspace_active_hypotheses.iter() {
        candidates.push(WorkspaceCandidate {
            source_module: "active_hypotheses".to_string(),
            content_ref: hypothesis.text.clone(),
            salience_score: hypothesis.confidence as f64,
            risk_score: if hypothesis.speculative { 0.6 } else { 0.3 },
        });
    }
    for topic in kernel_state.workspace_working_set_topics.iter() {
        candidates.push(WorkspaceCandidate {
            source_module: "working_set_topics".to_string(),
            content_ref: topic.clone(),
            salience_score: 0.4,
            risk_score: 0.2,
        });
    }
    if let Some(focus) = kernel_state.workspace_current_focus.as_deref() {
        if !focus.trim().is_empty() {
            candidates.push(WorkspaceCandidate {
                source_module: "current_focus".to_string(),
                content_ref: focus.to_string(),
                salience_score: 1.0,
                risk_score: 0.2,
            });
        }
    }

    candidates.sort_by(|a, b| b.salience_score.partial_cmp(&a.salience_score).unwrap_or(std::cmp::Ordering::Equal));
    let slots = 3i64;
    let winners = candidates.iter().take(slots as usize).cloned().collect::<Vec<_>>();
    let broadcast_refs = winners.iter().map(|c| c.content_ref.clone()).collect::<Vec<_>>();
    let ignition_score = if winners.is_empty() {
        0.0
    } else {
        winners.iter().map(|c| c.salience_score).sum::<f64>() / winners.len() as f64
    };
    let ignition_active = ignition_score >= 0.5;
    let duration_ticks = if ignition_active {
        previous.map(|p| p.ignition.duration_ticks + 1).unwrap_or(1)
    } else {
        0
    };

    WorkspaceState {
        timestamp: chrono::Utc::now().to_rfc3339(),
        slots,
        candidates,
        winners,
        broadcast_refs,
        ignition: WorkspaceIgnition {
            active: ignition_active,
            duration_ticks,
            ignition_score,
        },
    }
}

pub fn workspace_state_to_meta(state: &WorkspaceState) -> serde_json::Value {
    json!({
        "ignition": {
            "active": state.ignition.active,
            "duration_ticks": state.ignition.duration_ticks,
            "ignition_score": state.ignition.ignition_score,
        },
        "broadcast_refs": state.broadcast_refs,
        "slots": state.slots,
        "candidates": state.candidates,
        "winners": state.winners,
    })
}

pub async fn build_workspace_contributors(
    db: &Db,
    kernel_state: &KernelState,
    subject_state: &SubjectState,
    tick_id: &str,
) -> WorkspaceContributors {
    let control_map = system_controls::load_control_map(db).await;
    let now = chrono::Utc::now();
    let window_start = now - chrono::Duration::minutes(CONTRIBUTOR_WINDOW_MINUTES);
    let since = window_start.to_rfc3339();
    let recent_message_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM messages
         WHERE datetime(created_at) >= datetime(?)
           AND role IN ('user','assistant')
           AND status = 'complete'
           AND (metadata IS NULL OR json_extract(metadata, '$.source') != 'monologue')",
    )
    .bind(&since)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);

    let kernel = Some(WorkspaceKernelContributor {
        cycle_id: tick_id.to_string(),
        broadcast_refs: subject_state.workspace.broadcast_refs.clone(),
        ignition_active: subject_state.workspace.ignition.active,
        timestamp: subject_state.workspace.timestamp.clone(),
    });

    let memory_enabled = system_controls::is_subsystem_enabled("memory_retrieval", &control_map);
    let memory_recent_writes: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memory_write_ledger
         WHERE datetime(created_at) >= datetime(?)",
    )
    .bind(&since)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    let memory_write_blocked: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'memory_write_blocked'
           AND datetime(timestamp) >= datetime(?)",
    )
    .bind(&since)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    let memory_write_blocked_control: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'memory_write_blocked'
           AND json_extract(payload, '$.reason') IN (
             'memory_write_control',
             'system_control_self_memory',
             'system_control',
             'system_control_degraded'
           )
           AND datetime(timestamp) >= datetime(?)",
    )
    .bind(&since)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    let memory_write_blocked_other =
        memory_write_blocked.saturating_sub(memory_write_blocked_control);
    let memory_write_attempts = memory_recent_writes + memory_write_blocked_other;
    let memory_last_write_at: Option<String> = sqlx::query_scalar(
        "SELECT created_at FROM memory_write_ledger
         ORDER BY datetime(created_at) DESC LIMIT 1",
    )
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let memory_top_topics = kernel_state
        .workspace_working_set_topics
        .iter()
        .filter(|topic| !topic.trim().is_empty())
        .take(3)
        .cloned()
        .collect::<Vec<_>>();
    let memory = if memory_enabled {
        Some(WorkspaceMemoryContributor {
            recent_writes: memory_recent_writes,
            top_topics: memory_top_topics,
            last_write_at: memory_last_write_at.clone(),
        })
    } else {
        None
    };

    let prediction_enabled = system_controls::is_subsystem_enabled("prediction_generation", &control_map);
    let prediction_last_at: Option<String> = sqlx::query_scalar(
        "SELECT created_at FROM self_predictions
         ORDER BY datetime(created_at) DESC LIMIT 1",
    )
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let prediction_divergence_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'prediction_divergence'
           AND datetime(timestamp) >= datetime(?)",
    )
    .bind(&since)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    let prediction_residual_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM residual_vectors
         WHERE datetime(created_at) >= datetime(?)",
    )
    .bind(&since)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    let prediction = if prediction_enabled {
        Some(WorkspacePredictionContributor {
            last_prediction_at: prediction_last_at.clone(),
            divergence_count: prediction_divergence_count,
            residual_count: prediction_residual_count,
        })
    } else {
        None
    };

    let attention_enabled = system_controls::is_subsystem_enabled("attention_schema", &control_map);
    let attention = if attention_enabled {
        Some(WorkspaceAttentionContributor {
            focus_refs: subject_state.attention.current_focus_refs.clone(),
            meta_confidence: subject_state.attention.meta_confidence,
        })
    } else {
        None
    };

    let self_model_enabled = system_controls::is_subsystem_enabled("self_memory", &control_map);
    let self_model = if self_model_enabled {
        Some(WorkspaceSelfModelContributor {
            identity_confidence: subject_state.self_model.identity_confidence,
            last_reflection_at: subject_state.self_model.last_reflection_at.clone(),
        })
    } else {
        None
    };
    let self_model_snapshot_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM self_model_controller_snapshots
         WHERE datetime(created_at) >= datetime(?)",
    )
    .bind(&since)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    let self_reflection_staging_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM self_reflection_staging
         WHERE datetime(created_at) >= datetime(?)",
    )
    .bind(&since)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    let self_model_recent_reflection = self_model
        .as_ref()
        .and_then(|model| model.last_reflection_at.as_deref())
        .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
        .map(|ts| ts.with_timezone(&Utc) >= window_start)
        .unwrap_or(false);
    let self_model_recent_activity = self_model_snapshot_count > 0
        || self_reflection_staging_count > 0
        || self_model_recent_reflection;

    let qualia_enabled = system_controls::is_subsystem_enabled("qualia_loop", &control_map);
    let qualia = if qualia_enabled {
        Some(WorkspaceQualiaContributor {
            dominant_tag: subject_state.qualia.dominant_tag.clone(),
            intensity: subject_state.qualia.dominant_intensity,
        })
    } else {
        None
    };

    let tool_enabled = system_controls::is_subsystem_enabled("tool_execution", &control_map);
    let tool_successes: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM tool_dispatches
         WHERE status = 'success' AND datetime(updated_at) >= datetime(?)",
    )
    .bind(&since)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    let tool_failures: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM tool_dispatches
         WHERE status = 'failed' AND datetime(updated_at) >= datetime(?)",
    )
    .bind(&since)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    let tool_total = tool_successes + tool_failures;
    let (success_rate, failure_rate) = if tool_total > 0 {
        (
            tool_successes as f64 / tool_total as f64,
            tool_failures as f64 / tool_total as f64,
        )
    } else {
        (0.0, 0.0)
    };
    let tool_last_failure: Option<String> = sqlx::query_scalar(
        "SELECT failure_kind FROM tool_dispatches
         WHERE status = 'failed'
         ORDER BY datetime(updated_at) DESC LIMIT 1",
    )
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let tools = if tool_enabled {
        Some(WorkspaceToolsContributor {
            success_rate,
            failure_rate,
            last_failure_kind: tool_last_failure.clone(),
        })
    } else {
        None
    };

    let organism = Some(WorkspaceOrganismContributor {
        integrity_risk: subject_state.organism.integrity_risk,
        drift_score: subject_state.self_model.controller_state.drift_score as f64,
    });

    let error_state = Some(WorkspaceErrorContributor {
        open_error_count: subject_state.error_state.open_error_count,
        pattern_flags: subject_state.error_state.pattern_flags.clone(),
    });

    let mut missing = Vec::new();
    let active_window = recent_message_count > 0;
    if !active_window {
        let _ = system_log::log_event(
            &db.pool,
            None,
            "info",
            "workspace",
            None,
            None,
            json!({
                "event": "workspace_contributors_idle",
                "recent_message_count": recent_message_count,
                "window_minutes": CONTRIBUTOR_WINDOW_MINUTES,
            }),
        )
        .await;
    }
    if active_window && memory_enabled && memory_write_attempts > 0 && memory_recent_writes == 0 {
        missing.push("memory".to_string());
        if should_log_missing(db, "memory").await {
            let _ = system_log::log_event(
                &db.pool,
                None,
                "info",
                "workspace",
                None,
                None,
                json!({
                    "event": "workspace_missing_flagged",
                    "subsystem": "memory",
                    "reason": "attempted_no_writes",
                    "attempts": memory_write_attempts,
                }),
            )
            .await;
        }
    }
    if active_window && prediction_enabled && prediction_last_at.is_none() {
        missing.push("prediction".to_string());
        if should_log_missing(db, "prediction").await {
            let _ = system_log::log_event(
                &db.pool,
                None,
                "info",
                "workspace",
                None,
                None,
                json!({
                    "event": "workspace_missing_flagged",
                    "subsystem": "prediction",
                    "reason": "no_recent_predictions",
                }),
            )
            .await;
        }
    }
    if active_window
        && attention_enabled
        && attention
            .as_ref()
            .map(|a| a.focus_refs.is_empty())
            .unwrap_or(true)
    {
        missing.push("attention".to_string());
        if should_log_missing(db, "attention").await {
            let _ = system_log::log_event(
                &db.pool,
                None,
                "info",
                "workspace",
                None,
                None,
                json!({
                    "event": "workspace_missing_flagged",
                    "subsystem": "attention",
                    "reason": "empty_focus_refs",
                }),
            )
            .await;
        }
    }
    if active_window && self_model_enabled && !self_model_recent_activity {
        missing.push("self_model".to_string());
        if should_log_missing(db, "self_model").await {
            let _ = system_log::log_event(
                &db.pool,
                None,
                "info",
                "workspace",
                None,
                None,
                json!({
                    "event": "workspace_missing_flagged",
                    "subsystem": "self_model",
                    "reason": "no_recent_reflection_activity",
                    "controller_snapshots": self_model_snapshot_count,
                    "reflection_staging": self_reflection_staging_count,
                }),
            )
            .await;
        }
    }
    if active_window
        && qualia_enabled
        && qualia.as_ref().and_then(|q| q.dominant_tag.as_deref()).is_none()
    {
        missing.push("qualia".to_string());
        if should_log_missing(db, "qualia").await {
            let _ = system_log::log_event(
                &db.pool,
                None,
                "info",
                "workspace",
                None,
                None,
                json!({
                    "event": "workspace_missing_flagged",
                    "subsystem": "qualia",
                    "reason": "missing_qualia_tag",
                }),
            )
            .await;
        }
    }
    if active_window && tool_enabled && tool_total == 0 {
        missing.push("tools".to_string());
        if should_log_missing(db, "tools").await {
            let _ = system_log::log_event(
                &db.pool,
                None,
                "info",
                "workspace",
                None,
                None,
                json!({
                    "event": "workspace_missing_flagged",
                    "subsystem": "tools",
                    "reason": "no_tool_dispatches",
                }),
            )
            .await;
        }
    }

    WorkspaceContributors {
        kernel,
        memory,
        prediction,
        attention,
        self_model,
        qualia,
        tools,
        organism,
        error_state,
        missing,
        updated_at: Some(now.to_rfc3339()),
    }
}

pub fn summarize_contributors(contributors: &WorkspaceContributors) -> String {
    let mut parts = Vec::new();
    let kernel_status = if contributors.kernel.is_some() { "ok" } else { "none" };
    parts.push(format!("kernel={}", kernel_status));
    if let Some(memory) = contributors.memory.as_ref() {
        parts.push(format!("memory=writes:{}", memory.recent_writes));
    } else {
        parts.push("memory=off".to_string());
    }
    if let Some(prediction) = contributors.prediction.as_ref() {
        parts.push(format!(
            "prediction=residuals:{}",
            prediction.residual_count
        ));
    } else {
        parts.push("prediction=off".to_string());
    }
    if let Some(attention) = contributors.attention.as_ref() {
        parts.push(format!("attention=focus:{}", attention.focus_refs.len()));
    }
    if let Some(self_model) = contributors.self_model.as_ref() {
        parts.push(format!(
            "self_model=conf:{:.2}",
            self_model.identity_confidence
        ));
    }
    if let Some(qualia) = contributors.qualia.as_ref() {
        let tag = qualia
            .dominant_tag
            .as_deref()
            .unwrap_or("none")
            .to_string();
        parts.push(format!("qualia={}", tag));
    }
    if let Some(tools) = contributors.tools.as_ref() {
        parts.push(format!("tools=fail:{:.2}", tools.failure_rate));
    }
    if let Some(organism) = contributors.organism.as_ref() {
        parts.push(format!(
            "organism=integrity:{:.2}",
            organism.integrity_risk
        ));
    }
    if let Some(error_state) = contributors.error_state.as_ref() {
        parts.push(format!(
            "errors=open:{}",
            error_state.open_error_count
        ));
    }
    if !contributors.missing.is_empty() {
        parts.push(format!("missing=[{}]", contributors.missing.join(", ")));
    }
    parts.join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;
    use serde_json::json;
    use uuid::Uuid;

    #[tokio::test]
    async fn build_workspace_contributors_flags_missing_subsystems() {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("in-memory pool");
        let db = Db { pool };
        db.init().await.expect("init db");

        for subsystem in [
            "kernel_loop",
            "workspace_broadcast",
            "memory_retrieval",
            "prediction_generation",
            "attention_schema",
            "self_memory",
            "qualia_loop",
            "tool_execution",
            "organism_loop",
        ] {
            sqlx::query(
                "INSERT OR REPLACE INTO system_controls (subsystem_id, mode, updated_at)
                 VALUES (?, 'normal', CURRENT_TIMESTAMP)",
            )
            .bind(subsystem)
            .execute(&db.pool)
            .await
            .expect("insert system control");
        }

        sqlx::query(
            "INSERT INTO conversations (conversation_id, schema_version, created_at, updated_at)
             VALUES ('default', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        )
        .execute(&db.pool)
        .await
        .expect("insert conversation");
        sqlx::query(
            "INSERT INTO messages (message_id, conversation_id, role, content, status, created_at)
             VALUES ('msg_user_1', 'default', 'user', 'hello', 'complete', CURRENT_TIMESTAMP)",
        )
        .execute(&db.pool)
        .await
        .expect("insert message");

        let kernel_state = KernelState::default_for("default");
        let subject_state =
            crate::core::subject_state::build_subject_state(&db, &kernel_state, None)
                .await
                .expect("subject state");

        let contributors =
            build_workspace_contributors(&db, &kernel_state, &subject_state, "tick_1").await;

        assert!(contributors.kernel.is_some());
        assert!(!contributors.missing.contains(&"memory".to_string()));
        assert!(contributors.missing.contains(&"prediction".to_string()));

        sqlx::query(
            "INSERT INTO system_logs (id, timestamp, level, category, run_id, trace_id, payload)
             VALUES (?, CURRENT_TIMESTAMP, 'info', 'memory', NULL, NULL, ?)",
        )
        .bind(Uuid::new_v4().to_string())
        .bind(json!({ "event": "memory_write_blocked" }).to_string())
        .execute(&db.pool)
        .await
        .expect("insert memory_write_blocked log");

        let contributors_after =
            build_workspace_contributors(&db, &kernel_state, &subject_state, "tick_2").await;
        assert!(contributors_after.missing.contains(&"memory".to_string()));
    }
}
