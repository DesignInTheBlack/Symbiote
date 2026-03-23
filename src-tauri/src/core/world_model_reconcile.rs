use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use crate::core::system_log;

const MIN_CONFIDENCE: f64 = 0.6;
const MIN_SCORE_GAP: f64 = 0.2;
const MIN_WIN_SCORE: f64 = 0.75;
const MIN_EVIDENCE_WEIGHT: f64 = 0.6;
const MIN_EVIDENCE_SOURCES: i64 = 2;
const RETIRE_DEMOTION_RUNS: i64 = 3;
const RETIRE_STALE_DAYS: i64 = 7;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum WorldModelReconcileMode {
    Off,
    Shadow,
    Active,
}

impl WorldModelReconcileMode {
    pub fn from_str(raw: Option<&str>) -> Self {
        match raw.unwrap_or("shadow").trim().to_lowercase().as_str() {
            "off" => WorldModelReconcileMode::Off,
            "active" => WorldModelReconcileMode::Active,
            _ => WorldModelReconcileMode::Shadow,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            WorldModelReconcileMode::Off => "off",
            WorldModelReconcileMode::Shadow => "shadow",
            WorldModelReconcileMode::Active => "active",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelReconcileReport {
    pub scanned: usize,
    pub resolved: usize,
    pub unresolved: usize,
    pub outcomes: Vec<WorldModelConflictOutcome>,
    pub mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelConflictOutcome {
    pub conflict_set_id: i64,
    pub topic_key: String,
    pub status: String,
    pub winner_belief_id: Option<i64>,
    pub score_gap: Option<f64>,
    pub member_count: usize,
    pub reason: String,
}

#[derive(Debug, Clone)]
struct BeliefScore {
    belief_id: i64,
    confidence: f64,
    evidence_weight: f64,
    score: f64,
    evidence_sources: i64,
    reconcile_state: String,
    reconcile_demoted_runs: i64,
    last_evidence_at: Option<String>,
}

pub async fn reconcile_conflict_sets(
    pool: &SqlitePool,
    conversation_id: &str,
    mode: WorldModelReconcileMode,
) -> Result<WorldModelReconcileReport, String> {
    if mode == WorldModelReconcileMode::Off {
        let _ = system_log::log_event(
            pool,
            None,
            "info",
            "memory",
            None,
            None,
            json!({
                "event": "world_model_reconcile_skipped",
                "conversation_id": conversation_id,
                "mode": mode.as_str(),
            }),
        )
        .await;
        return Ok(WorldModelReconcileReport {
            scanned: 0,
            resolved: 0,
            unresolved: 0,
            outcomes: Vec::new(),
            mode: mode.as_str().to_string(),
        });
    }

    let rows = sqlx::query(
        "SELECT id, topic_key
         FROM ics_conflict_sets
         WHERE status = 'open'
         ORDER BY CASE priority WHEN 'high' THEN 2 WHEN 'normal' THEN 1 ELSE 0 END DESC,
                  datetime(updated_at) DESC,
                  id ASC",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    let run_id = Uuid::new_v4().to_string();
    let mut report = WorldModelReconcileReport {
        scanned: rows.len(),
        resolved: 0,
        unresolved: 0,
        outcomes: Vec::new(),
        mode: mode.as_str().to_string(),
    };

    for row in rows {
        let conflict_set_id: i64 = row.get("id");
        let topic_key: String = row.get("topic_key");

        let member_rows = sqlx::query(
            "SELECT b.id, b.confidence, b.evidence_weight_total, b.reconcile_state,
                    b.reconcile_demoted_runs, b.last_evidence_at
             FROM ics_conflict_set_members m
             JOIN ics_beliefs b ON b.id = m.belief_id
             WHERE m.conflict_set_id = ? AND b.status = 'active' AND b.reconcile_state != 'retired'
             ORDER BY b.id ASC",
        )
        .bind(conflict_set_id)
        .fetch_all(pool)
        .await
        .map_err(|e| e.to_string())?;

        let mut scores: Vec<BeliefScore> = Vec::new();
        for member in member_rows {
            let belief_id = member.get::<i64, _>("id");
            let confidence = member.try_get::<f64, _>("confidence").unwrap_or(0.0);
            let evidence_weight = member.try_get::<f64, _>("evidence_weight_total").unwrap_or(0.0);
            let evidence_sources = sqlx::query_scalar::<_, i64>(
                "SELECT COUNT(DISTINCT source_type) FROM ics_evidence_events WHERE belief_id = ?",
            )
            .bind(belief_id)
            .fetch_optional(pool)
            .await
            .ok()
            .flatten()
            .unwrap_or(0);
            let reconcile_state = member
                .try_get::<String, _>("reconcile_state")
                .unwrap_or_else(|_| "active".to_string());
            let reconcile_demoted_runs = member
                .try_get::<i64, _>("reconcile_demoted_runs")
                .unwrap_or(0);
            let last_evidence_at = member.try_get::<String, _>("last_evidence_at").ok();
            let score = score_belief(confidence, evidence_weight);
            scores.push(BeliefScore {
                belief_id,
                confidence,
                evidence_weight,
                score,
                evidence_sources,
                reconcile_state,
                reconcile_demoted_runs,
                last_evidence_at,
            });
        }

        if scores.is_empty() {
            report.unresolved += 1;
            report.outcomes.push(WorldModelConflictOutcome {
                conflict_set_id,
                topic_key: topic_key.clone(),
                status: "open".to_string(),
                winner_belief_id: None,
                score_gap: None,
                member_count: 0,
                reason: "no_active_members".to_string(),
            });
            let _ = system_log::log_event(
                pool,
                None,
                "info",
                "memory",
                None,
                None,
                json!({
                    "event": "world_model_conflict_unresolved",
                    "conversation_id": conversation_id,
                    "conflict_set_id": conflict_set_id,
                    "topic_key": topic_key,
                    "reason": "no_active_members",
                    "mode": mode.as_str(),
                }),
            )
            .await;
            continue;
        }

        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let winner = &scores[0];
        let runner_up = scores.get(1);
        let score_gap = runner_up.map(|other| winner.score - other.score);
        let winner_ok = if scores.len() == 1 {
            winner.confidence >= MIN_CONFIDENCE
                && winner.score >= MIN_WIN_SCORE
                && winner.evidence_weight >= MIN_EVIDENCE_WEIGHT
                && winner.evidence_sources >= MIN_EVIDENCE_SOURCES
        } else {
            winner.confidence >= MIN_CONFIDENCE
                && winner.score >= MIN_WIN_SCORE
                && score_gap.unwrap_or(0.0) >= MIN_SCORE_GAP
                && winner.evidence_weight >= MIN_EVIDENCE_WEIGHT
                && winner.evidence_sources >= MIN_EVIDENCE_SOURCES
        };

        if winner_ok {
            let note = format!(
                "auto_resolved:winner={}; score={:.2}; gap={:.2}; conf={:.2}; evidence={:.2}; sources={}",
                winner.belief_id,
                winner.score,
                score_gap.unwrap_or(0.0),
                winner.confidence,
                winner.evidence_weight,
                winner.evidence_sources,
            );

            if mode == WorldModelReconcileMode::Active {
                let _ = sqlx::query(
                    "UPDATE ics_conflict_sets
                     SET status = 'resolved', resolution_note = ?, updated_at = CURRENT_TIMESTAMP
                     WHERE id = ? AND status = 'open'",
                )
                .bind(&note)
                .bind(conflict_set_id)
                .execute(pool)
                .await;
            }

            let mut updates = Vec::new();
            for belief in scores.iter() {
                if belief.belief_id == winner.belief_id {
                    updates.push((belief, "active".to_string(), "winner".to_string()));
                } else {
                    let next_state = if belief.confidence < 0.5 && belief.evidence_weight < 0.5 {
                        "retiring"
                    } else {
                        "contested"
                    };
                    updates.push((belief, next_state.to_string(), "conflict_loser".to_string()));
                }
            }

            if mode == WorldModelReconcileMode::Active {
                for (belief, next_state, reason) in updates.iter() {
                    apply_belief_state(pool, belief, &next_state, &reason, &run_id).await?;
                }
                // Retire beliefs that have been demoted repeatedly and are stale.
                for belief in scores.iter() {
                    let retire = belief.reconcile_demoted_runs >= RETIRE_DEMOTION_RUNS
                        && belief
                            .last_evidence_at
                            .as_deref()
                            .map(|ts| is_stale(ts, RETIRE_STALE_DAYS))
                            .unwrap_or(false);
                    if retire {
                        apply_belief_state(pool, belief, "retired", "stale_retire", &run_id).await?;
                    }
                }
            }

            report.resolved += 1;
            report.outcomes.push(WorldModelConflictOutcome {
                conflict_set_id,
                topic_key: topic_key.clone(),
                status: if mode == WorldModelReconcileMode::Active {
                    "resolved".to_string()
                } else {
                    "shadow_resolved".to_string()
                },
                winner_belief_id: Some(winner.belief_id),
                score_gap,
                member_count: scores.len(),
                reason: "auto_resolved".to_string(),
            });

            let _ = system_log::log_event(
                pool,
                None,
                "info",
                "memory",
                None,
                None,
                json!({
                    "event": "world_model_conflict_resolved",
                    "conversation_id": conversation_id,
                    "conflict_set_id": conflict_set_id,
                    "topic_key": topic_key,
                    "winner_belief_id": winner.belief_id,
                    "winner_score": winner.score,
                    "score_gap": score_gap,
                    "member_count": scores.len(),
                    "resolution_note": note,
                    "mode": mode.as_str(),
                }),
            )
            .await;
        } else {
            report.unresolved += 1;
            report.outcomes.push(WorldModelConflictOutcome {
                conflict_set_id,
                topic_key: topic_key.clone(),
                status: "open".to_string(),
                winner_belief_id: None,
                score_gap,
                member_count: scores.len(),
                reason: "insufficient_gap".to_string(),
            });

            let _ = system_log::log_event(
                pool,
                None,
                "info",
                "memory",
                None,
                None,
                json!({
                    "event": "world_model_conflict_unresolved",
                    "conversation_id": conversation_id,
                    "conflict_set_id": conflict_set_id,
                    "topic_key": topic_key,
                    "top_score": winner.score,
                    "score_gap": score_gap,
                    "member_count": scores.len(),
                    "reason": "insufficient_gap",
                    "mode": mode.as_str(),
                }),
            )
            .await;
        }
    }

    Ok(report)
}

async fn apply_belief_state(
    pool: &SqlitePool,
    belief: &BeliefScore,
    next_state: &str,
    reason: &str,
    run_id: &str,
) -> Result<(), String> {
    if belief.reconcile_state == next_state {
        return Ok(());
    }
    let now = Utc::now().to_rfc3339();
    let mut demoted_runs = belief.reconcile_demoted_runs;
    if matches!(next_state, "contested" | "retiring" | "retired") {
        demoted_runs = demoted_runs.saturating_add(1);
    } else if next_state == "active" {
        demoted_runs = 0;
    }
    sqlx::query(
        "UPDATE ics_beliefs
         SET reconcile_state = ?, reconcile_reason = ?, reconcile_updated_at = ?, reconcile_run_id = ?, reconcile_demoted_runs = ?
         WHERE id = ?",
    )
    .bind(next_state)
    .bind(reason)
    .bind(&now)
    .bind(run_id)
    .bind(demoted_runs)
    .bind(belief.belief_id)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    let event_id = format!("wm:{}:{}", run_id, belief.belief_id);
    let event_type = match next_state {
        "retired" => "retire",
        "retiring" => "demote",
        "contested" => "contest",
        "active" => "restore",
        _ => "state_change",
    };
    let _ = sqlx::query(
        "INSERT INTO world_model_events (event_id, belief_id, event_type, prev_state, new_state, reason, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(event_id)
    .bind(belief.belief_id)
    .bind(event_type)
    .bind(&belief.reconcile_state)
    .bind(next_state)
    .bind(reason)
    .bind(now)
    .execute(pool)
    .await;

    Ok(())
}

fn score_belief(confidence: f64, evidence_weight: f64) -> f64 {
    let evidence_component = (evidence_weight / 3.0).min(1.0);
    (confidence * 0.7) + (evidence_component * 0.3)
}

fn is_stale(raw_ts: &str, days: i64) -> bool {
    let parsed = DateTime::parse_from_rfc3339(raw_ts)
        .map(|dt| dt.with_timezone(&Utc))
        .or_else(|_| {
            chrono::NaiveDateTime::parse_from_str(raw_ts, "%Y-%m-%d %H:%M:%S")
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
        })
        .ok();
    parsed
        .map(|dt| Utc::now().signed_duration_since(dt).num_days() >= days)
        .unwrap_or(false)
}
