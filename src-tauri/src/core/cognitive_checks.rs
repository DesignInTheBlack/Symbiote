use crate::db::Db;
use crate::core::system_controls;
use crate::models::CognitiveCheckResult;
use chrono::{DateTime, Utc, TimeZone};
use sqlx::Row;
use std::collections::HashMap;

fn parse_db_ts(raw: &str) -> Option<DateTime<Utc>> {
    chrono::NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S")
        .map(|dt| Utc.from_utc_datetime(&dt))
        .ok()
        .or_else(|| chrono::DateTime::parse_from_rfc3339(raw).map(|dt| dt.with_timezone(&Utc)).ok())
}

fn extract_numeric_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut buf = String::new();
    let mut has_digit = false;
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            buf.push(ch);
            has_digit = true;
            continue;
        }
        if (ch == '.' || ch == ',') && has_digit {
            if ch == '.' {
                buf.push(ch);
            }
            continue;
        }
        if (ch == '-' || ch == '+') && buf.is_empty() {
            buf.push(ch);
            continue;
        }
        if ch == '%' && has_digit {
            buf.push(ch);
            continue;
        }
        if has_digit {
            tokens.push(buf.clone());
            buf.clear();
            has_digit = false;
        } else {
            buf.clear();
        }
    }
    if has_digit && !buf.is_empty() {
        tokens.push(buf);
    }
    tokens
}

fn numeric_token_allowed(token: &str, allowed: &[f32], last_user_input: &str) -> bool {
    let stripped = token.trim_end_matches('%');
    if stripped.is_empty() {
        return true;
    }
    let Ok(value) = stripped.parse::<f32>() else {
        return true;
    };
    if allowed.iter().any(|v| (v - value).abs() <= 0.01) {
        return true;
    }
    let user_numbers = extract_numeric_tokens(last_user_input);
    user_numbers
        .iter()
        .filter_map(|n| n.trim_end_matches('%').parse::<f32>().ok())
        .any(|v| (v - value).abs() <= 0.01)
}

const PERF_WARN_MODEL_MS: i64 = 8000;
const PERF_WARN_MONOLOGUE_MS: i64 = 4000;
const PERF_WARN_HEARTBEAT_MS: i64 = 3000;
const PERF_WARN_DREAM_MS: i64 = 6000;
const LATENCY_TARGET_P50_MS: i64 = 1500;
const LATENCY_TARGET_P95_MS: i64 = 4000;
const LATENCY_TARGET_TTFB_MS: i64 = 500;

fn percentile(values: &mut [i64], pct: f64) -> Option<i64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let rank = ((values.len() as f64 - 1.0) * pct.clamp(0.0, 1.0)).round() as usize;
    values.get(rank).copied()
}

pub async fn run_cognitive_checks(db: &Db, conversation_id: &str) -> Vec<CognitiveCheckResult> {
    let mode: Option<String> = sqlx::query_scalar(
        "SELECT mode FROM system_controls WHERE subsystem_id = ?",
    )
    .bind("audits")
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let audit_mode = mode.unwrap_or_else(|| {
        system_controls::default_mode_for("audits")
            .unwrap_or("normal")
            .to_string()
    });
    if system_controls::mode_is_off(&audit_mode) || system_controls::mode_is_degraded(&audit_mode) {
        return Vec::new();
    }
    let mut results = Vec::new();

    // Tool call -> tool result -> response
    let tool_row = sqlx::query(
        "SELECT action_id, updated_at FROM tool_dispatches WHERE status = 'success' ORDER BY datetime(updated_at) DESC LIMIT 1"
    )
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    if let Some(row) = tool_row {
        let updated_at: String = row.get("updated_at");
        let message_row = sqlx::query(
            "SELECT message_id FROM messages
             WHERE role = 'assistant' AND conversation_id = ?
               AND datetime(created_at) >= datetime(?)
               AND (metadata IS NULL OR json_extract(metadata, '$.source') != 'monologue')
             ORDER BY datetime(created_at) DESC LIMIT 1"
        )
        .bind(conversation_id)
        .bind(&updated_at)
        .fetch_optional(&db.pool)
        .await
        .ok()
        .flatten();
        let passed = message_row.is_some();
        results.push(CognitiveCheckResult {
            name: "tool_loop".to_string(),
            passed,
            detail: if passed {
                "Tool dispatch success followed by assistant response.".to_string()
            } else {
                "Tool dispatch success found but no assistant response after it.".to_string()
            },
        });
    } else {
        results.push(CognitiveCheckResult {
            name: "tool_loop".to_string(),
            passed: false,
            detail: "No successful tool dispatches recorded.".to_string(),
        });
    }

    // Monologue tick updates inner summary
    let last_monologue_at: Option<String> = sqlx::query_scalar(
        "SELECT created_at FROM inner_monologue_entries WHERE conversation_id = ? ORDER BY datetime(created_at) DESC LIMIT 1"
    )
    .bind(conversation_id)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let last_monologue_at_copy = last_monologue_at.clone();
    let last_inner_summary_at: Option<String> = sqlx::query_scalar(
        "SELECT updated_at FROM inner_summaries WHERE conversation_id = ? LIMIT 1"
    )
    .bind(conversation_id)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let monologue_check = match (last_monologue_at, last_inner_summary_at) {
        (Some(monologue_at), Some(inner_at)) => {
            let mono_ts = chrono::DateTime::parse_from_rfc3339(&monologue_at).ok();
            let inner_ts = chrono::DateTime::parse_from_rfc3339(&inner_at).ok();
            if let (Some(mono_ts), Some(inner_ts)) = (mono_ts, inner_ts) {
                if inner_ts >= mono_ts {
                    CognitiveCheckResult {
                        name: "monologue_updates_summary".to_string(),
                        passed: true,
                        detail: "Inner summary updated after monologue tick.".to_string(),
                    }
                } else {
                    CognitiveCheckResult {
                        name: "monologue_updates_summary".to_string(),
                        passed: false,
                        detail: "Inner summary older than last monologue tick.".to_string(),
                    }
                }
            } else {
                CognitiveCheckResult {
                    name: "monologue_updates_summary".to_string(),
                    passed: false,
                    detail: "Unable to parse monologue/summary timestamps.".to_string(),
                }
            }
        }
        (None, _) => CognitiveCheckResult {
            name: "monologue_updates_summary".to_string(),
            passed: false,
            detail: "No monologue entries recorded.".to_string(),
        },
        (_, None) => CognitiveCheckResult {
            name: "monologue_updates_summary".to_string(),
            passed: false,
            detail: "No inner summary updates recorded.".to_string(),
        },
    };
    results.push(monologue_check);

    // Monologue tick updates workspace
    let last_workspace_at: Option<String> = sqlx::query_scalar(
        "SELECT updated_at FROM workspace_state WHERE conversation_id = ? LIMIT 1"
    )
    .bind(conversation_id)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let workspace_check = match (last_monologue_at_copy, last_workspace_at) {
        (Some(monologue_at), Some(workspace_at)) => {
            let mono_ts = parse_db_ts(&monologue_at);
            let workspace_ts = parse_db_ts(&workspace_at);
            if let (Some(mono_ts), Some(workspace_ts)) = (mono_ts, workspace_ts) {
                if workspace_ts >= mono_ts {
                    CognitiveCheckResult {
                        name: "monologue_updates_workspace".to_string(),
                        passed: true,
                        detail: "Workspace updated after monologue tick.".to_string(),
                    }
                } else {
                    CognitiveCheckResult {
                        name: "monologue_updates_workspace".to_string(),
                        passed: false,
                        detail: "Workspace older than last monologue tick.".to_string(),
                    }
                }
            } else {
                CognitiveCheckResult {
                    name: "monologue_updates_workspace".to_string(),
                    passed: false,
                    detail: "Unable to parse monologue/workspace timestamps.".to_string(),
                }
            }
        }
        (None, _) => CognitiveCheckResult {
            name: "monologue_updates_workspace".to_string(),
            passed: false,
            detail: "No monologue entries recorded.".to_string(),
        },
        (_, None) => CognitiveCheckResult {
            name: "monologue_updates_workspace".to_string(),
            passed: false,
            detail: "Workspace state missing.".to_string(),
        },
    };
    results.push(workspace_check);

    // Semantic hint usage in monologue
    let hint_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs WHERE category = 'memory' AND json_extract(payload, '$.event') = 'monologue_semantic_hint'"
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "semantic_hint_usage".to_string(),
        passed: hint_count > 0,
        detail: if hint_count > 0 {
            format!("Semantic hint events recorded: {}", hint_count)
        } else {
            "No semantic hint events recorded.".to_string()
        },
    });

    // Policy violation logging check (write a known violation)
    let snapshot_hash: Option<String> = sqlx::query_scalar(
        "SELECT snapshot_hash FROM subject_snapshots
         WHERE conversation_id = ?
         ORDER BY datetime(timestamp) DESC
         LIMIT 1",
    )
    .bind(conversation_id)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let gate_decision: Option<String> = if let Some(hash) = snapshot_hash.as_deref() {
        sqlx::query_scalar(
            "SELECT decision FROM gate_decisions
             WHERE snapshot_hash = ?
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .bind(hash)
        .fetch_optional(&db.pool)
        .await
        .ok()
        .flatten()
    } else {
        None
    };
    if matches!(
        gate_decision.as_deref(),
        Some("ALLOW" | "ALLOW_WITH_NOTICE" | "ALLOW_WITH_AUDIT")
    ) {
        let before_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs WHERE category = 'memory_policy'",
        )
        .fetch_one(&db.pool)
        .await
        .unwrap_or(0);
        let allowed = db
            .log_memory_write(
                Some(conversation_id),
                "semantic",
                "unknown",
                "test_violation",
                None,
                None,
                None,
                snapshot_hash.as_deref(),
                None,
            )
            .await
            .unwrap_or(true);
        let after_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs WHERE category = 'memory_policy'",
        )
        .fetch_one(&db.pool)
        .await
        .unwrap_or(0);
        let policy_passed = !allowed && after_count > before_count;
        results.push(CognitiveCheckResult {
            name: "policy_violation_logging".to_string(),
            passed: policy_passed,
            detail: if policy_passed {
                "Policy violation logged successfully.".to_string()
            } else {
                "Policy violation log was not observed.".to_string()
            },
        });
    } else {
        results.push(CognitiveCheckResult {
            name: "policy_violation_logging".to_string(),
            passed: true,
            detail: "Skipped: gate decision not ALLOW.".to_string(),
        });
    }

    let drift_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs WHERE json_extract(payload, '$.event') = 'monologue_drift'"
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "monologue_drift_events".to_string(),
        passed: drift_count == 0,
        detail: format!("Drift events recorded: {}", drift_count),
    });

    let unknown_tool_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs WHERE json_extract(payload, '$.event') IN ('unknown_tool_rejections', 'tool_candidate_rejected', 'tool_call_rejected')"
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "unknown_tool_events".to_string(),
        passed: unknown_tool_count == 0,
        detail: format!("Unknown/disabled tool events recorded: {}", unknown_tool_count),
    });

    // Self-audit grounding check
    let self_audit_total: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs WHERE json_extract(payload, '$.event') = 'self_audit_used'"
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    let self_audit_failures: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs WHERE json_extract(payload, '$.event') IN ('self_audit_validation_failed','self_audit_validation_hard_fail')"
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "self_audit_grounding".to_string(),
        passed: self_audit_failures == 0,
        detail: if self_audit_total == 0 {
            "No self-audit turns recorded.".to_string()
        } else if self_audit_failures == 0 {
            "Self-audit responses validated against runtime state.".to_string()
        } else {
            format!("Self-audit validation failures recorded: {}", self_audit_failures)
        },
    });

    // Memory evidence integrity check
    let memory_integrity_violations: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'memory_pass_result'
           AND json_extract(payload, '$.pending_clarify') = 1
           AND json_extract(payload, '$.written_ids') > 0"
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "memory_evidence_integrity".to_string(),
        passed: memory_integrity_violations == 0,
        detail: if memory_integrity_violations == 0 {
            "No memory writes occurred while clarifications were pending.".to_string()
        } else {
            format!("Memory writes occurred with pending clarifications: {}", memory_integrity_violations)
        },
    });

    // Attribution trace audit
    let attribution_blocked: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'user_attribution_blocked'
           AND datetime(timestamp) >= datetime('now', '-7 day')",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "attribution_trace".to_string(),
        passed: attribution_blocked == 0,
        detail: if attribution_blocked == 0 {
            "No blocked attributions in last 7 days.".to_string()
        } else {
            format!("Blocked attributions in last 7 days: {}", attribution_blocked)
        },
    });

    // Role safety audit
    let monologue_confusion: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'monologue_user_confusion'
           AND datetime(timestamp) >= datetime('now', '-7 day')",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "role_safety".to_string(),
        passed: monologue_confusion == 0,
        detail: if monologue_confusion == 0 {
            "No monologue user-impersonation events in last 7 days.".to_string()
        } else {
            format!("Monologue user-impersonation events in last 7 days: {}", monologue_confusion)
        },
    });

    let monologue_style_blocked: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'monologue_style_blocked'
           AND datetime(timestamp) >= datetime('now', '-7 day')",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "monologue_style_violations".to_string(),
        passed: monologue_style_blocked == 0,
        detail: if monologue_style_blocked == 0 {
            "No monologue style violations in last 7 days.".to_string()
        } else {
            format!("Monologue style violations in last 7 days: {}", monologue_style_blocked)
        },
    });

    let state_disclosure_blocked: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'state_disclosure_blocked'
           AND datetime(timestamp) >= datetime('now', '-7 day')",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "state_disclosure_guard".to_string(),
        passed: state_disclosure_blocked == 0,
        detail: if state_disclosure_blocked == 0 {
            "No blocked state disclosures in last 7 days.".to_string()
        } else {
            format!("Blocked state disclosures in last 7 days: {}", state_disclosure_blocked)
        },
    });

    let memory_write_blocked: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'memory_write_blocked'
           AND datetime(timestamp) >= datetime('now', '-7 day')",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "memory_write_gate".to_string(),
        passed: memory_write_blocked == 0,
        detail: if memory_write_blocked == 0 {
            "No blocked manual memory writes in last 7 days.".to_string()
        } else {
            format!("Blocked manual memory writes in last 7 days: {}", memory_write_blocked)
        },
    });

    // Self-memory evidence integrity
    let self_missing_evidence: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM self_evidence_events
         WHERE source_evidence_ids IS NULL OR source_evidence_ids = '' OR source_evidence_ids = '[]'",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "self_memory_evidence".to_string(),
        passed: self_missing_evidence == 0,
        detail: if self_missing_evidence == 0 {
            "All self-memory evidence events include source evidence IDs.".to_string()
        } else {
            format!("Self-memory evidence events missing IDs: {}", self_missing_evidence)
        },
    });

    // Summary provenance audit
    let summary = db
        .get_effective_rolling_summary(conversation_id)
        .await
        .ok()
        .and_then(|(summary, _)| summary)
        .unwrap_or_default();
    let summary_lower = summary.to_lowercase();
    let contains_monologue = summary_lower.contains("self a")
        || summary_lower.contains("self b")
        || summary_lower.contains("inner monologue")
        || summary_lower.contains("internal monologue");
    results.push(CognitiveCheckResult {
        name: "summary_provenance".to_string(),
        passed: !contains_monologue,
        detail: if contains_monologue {
            "Rolling summary appears to contain monologue content.".to_string()
        } else {
            "Rolling summary free of monologue markers.".to_string()
        },
    });

    // Identity stability audit
    let identity_updated_at: Option<String> = sqlx::query_scalar(
        "SELECT identity_updated_at FROM self_model WHERE id = 1",
    )
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let identity_supported = if let Some(updated_at) = identity_updated_at.as_deref() {
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM self_evidence_events
             WHERE source_evidence_ids IS NOT NULL AND source_evidence_ids != '' AND source_evidence_ids != '[]'
               AND datetime(created_at) >= datetime(?)",
        )
        .bind(updated_at)
        .fetch_one(&db.pool)
        .await
        .unwrap_or(0);
        count > 0
    } else {
        true
    };
    results.push(CognitiveCheckResult {
        name: "identity_stability".to_string(),
        passed: identity_supported,
        detail: if identity_supported {
            "Identity updates are supported by evidence events.".to_string()
        } else {
            "Identity updated without corresponding evidence events.".to_string()
        },
    });

    let emission_suppressed: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs WHERE json_extract(payload, '$.event') = 'emission_suppressed'"
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "emission_suppressed_events".to_string(),
        passed: emission_suppressed == 0,
        detail: format!("Emission suppression events recorded: {}", emission_suppressed),
    });

    let ask_loop_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs WHERE json_extract(payload, '$.event') = 'ask_loop_breaker_triggered'"
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "ask_loop_breaker_events".to_string(),
        passed: true,
        detail: format!("Ask loop breaker events recorded: {}", ask_loop_count),
    });

    let tool_loop_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs WHERE json_extract(payload, '$.event') = 'tool_loop_breaker_triggered'"
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "tool_loop_breaker_events".to_string(),
        passed: true,
        detail: format!("Tool loop breaker events recorded: {}", tool_loop_count),
    });

    let emit_loop_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs WHERE json_extract(payload, '$.event') = 'monologue_emit_loop_breaker'"
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "monologue_emit_loop_breaker_events".to_string(),
        passed: true,
        detail: format!("Emit loop breaker events recorded: {}", emit_loop_count),
    });

    let controller_state_present: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM self_model_controller_state WHERE id = 1"
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "controller_state_present".to_string(),
        passed: controller_state_present > 0,
        detail: if controller_state_present > 0 {
            "Controller state recorded.".to_string()
        } else {
            "Controller state missing.".to_string()
        },
    });

    let refusal_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs WHERE json_extract(payload, '$.event') = 'user_refused_inputs'"
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "refusal_events".to_string(),
        passed: true,
        detail: format!("User refusal events recorded: {}", refusal_count),
    });

    let perf_regression_events: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'performance_regression'
           AND json_extract(payload, '$.stage') IN ('prompt_build','commit_cycle','memory_retrieval','model_call')"
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    let slow_model_calls: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'timing_model_call'
           AND json_extract(payload, '$.duration_ms') > ?"
    )
    .bind(PERF_WARN_MODEL_MS)
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    let user_perf_passed = perf_regression_events == 0 && slow_model_calls == 0;
    results.push(CognitiveCheckResult {
        name: "performance_regression_user_turns".to_string(),
        passed: user_perf_passed,
        detail: format!(
            "performance_regression events: {}, slow_model_calls: {}",
            perf_regression_events, slow_model_calls
        ),
    });

    let monologue_slow: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'timing_monologue_tick'
           AND json_extract(payload, '$.duration_ms') > ?"
    )
    .bind(PERF_WARN_MONOLOGUE_MS)
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    let heartbeat_slow: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'timing_heartbeat_tick'
           AND json_extract(payload, '$.duration_ms') > ?"
    )
    .bind(PERF_WARN_HEARTBEAT_MS)
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    let dream_slow: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'timing_dream_cycle'
           AND json_extract(payload, '$.duration_ms') > ?"
    )
    .bind(PERF_WARN_DREAM_MS)
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    let tick_total = monologue_slow + heartbeat_slow + dream_slow;
    results.push(CognitiveCheckResult {
        name: "performance_regression_internal_ticks".to_string(),
        passed: tick_total == 0,
        detail: format!(
            "Slow ticks: monologue={}, heartbeat={}, dream={}",
            monologue_slow, heartbeat_slow, dream_slow
        ),
    });

    let timing_rows = sqlx::query(
        "SELECT json_extract(payload, '$.t_total_ms') as t_total_ms,
                json_extract(payload, '$.t_llm_prefill_ms') as t_ttfb_ms
         FROM system_logs
         WHERE json_extract(payload, '$.event') = 'timing_turn'
         ORDER BY datetime(timestamp) DESC
         LIMIT 200",
    )
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();
    let mut totals: Vec<i64> = Vec::new();
    let mut ttfb: Vec<i64> = Vec::new();
    for row in timing_rows {
        if let Ok(value) = row.try_get::<f64, _>("t_total_ms") {
            totals.push(value as i64);
        }
        if let Ok(value) = row.try_get::<f64, _>("t_ttfb_ms") {
            ttfb.push(value as i64);
        }
    }
    if totals.is_empty() {
        results.push(CognitiveCheckResult {
            name: "latency_targets".to_string(),
            passed: false,
            detail: "No timing_turn samples available.".to_string(),
        });
    } else {
        let p50_total = percentile(&mut totals, 0.50).unwrap_or(0);
        let p95_total = percentile(&mut totals, 0.95).unwrap_or(0);
        let p50_ttfb = percentile(&mut ttfb, 0.50).unwrap_or(0);
        let passed = p50_total <= LATENCY_TARGET_P50_MS
            && p95_total <= LATENCY_TARGET_P95_MS
            && p50_ttfb <= LATENCY_TARGET_TTFB_MS;
        results.push(CognitiveCheckResult {
            name: "latency_targets".to_string(),
            passed,
            detail: format!(
                "p50_total={}ms (target {}), p95_total={}ms (target {}), p50_ttfb={}ms (target {})",
                p50_total,
                LATENCY_TARGET_P50_MS,
                p95_total,
                LATENCY_TARGET_P95_MS,
                p50_ttfb,
                LATENCY_TARGET_TTFB_MS
            ),
        });
    }

    let voice_slow: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'voice_service_start'
           AND json_extract(payload, '$.duration_ms') > 5000",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "voice_pipeline_nonblocking".to_string(),
        passed: voice_slow == 0,
        detail: if voice_slow == 0 {
            "No slow voice service starts detected.".to_string()
        } else {
            format!("Slow voice service starts detected: {}", voice_slow)
        },
    });

    // Scaffold leakage check
    let scaffold_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM messages
         WHERE role = 'assistant'
           AND (LOWER(content) LIKE '%<<<begin_section%' OR LOWER(content) LIKE '%<<<end_section%'
                OR LOWER(content) LIKE '%next steps%' OR LOWER(content) LIKE '%proposed response%')",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "output_scaffold_leakage".to_string(),
        passed: scaffold_count == 0,
        detail: if scaffold_count == 0 {
            "No scaffold markers detected in assistant output.".to_string()
        } else {
            format!("Scaffold markers detected in {} assistant messages.", scaffold_count)
        },
    });

    // FTS visibility check
    let fts_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM inner_monologue_entries WHERE conversation_id = ? AND stream_type = 'FTS'",
    )
    .bind(conversation_id)
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "fts_visibility".to_string(),
        passed: fts_count > 0,
        detail: if fts_count > 0 {
            format!("FTS entries recorded: {}", fts_count)
        } else {
            "No FTS entries recorded.".to_string()
        },
    });

    // DS candidate surface check (speculative candidates allowed before verified anchor)
    let ds_surface_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM inner_monologue_candidates c
         JOIN inner_monologue_entries e ON e.id = c.entry_id
         WHERE e.conversation_id = ?
           AND e.stream_type = 'DS'
           AND c.outcome IN ('accepted', 'executed')
           AND (
                json_extract(c.candidate_json, '$.speculative') = 1
                OR json_extract(c.candidate_json, '$.speculative') = 'true'
               )",
    )
    .bind(conversation_id)
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "ds_surface_speculative".to_string(),
        passed: ds_surface_count > 0,
        detail: if ds_surface_count > 0 {
            format!("DS speculative candidates surfaced: {}", ds_surface_count)
        } else {
            "No surfaced speculative DS candidates recorded.".to_string()
        },
    });

    // Monologue boilerplate check
    let boilerplate_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM inner_monologue_entries
         WHERE LOWER(thought) LIKE '%processing your message%'
            OR LOWER(thought) LIKE '%currently processing%'
            OR LOWER(thought) LIKE '%preparing to respond%'",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "monologue_boilerplate".to_string(),
        passed: boilerplate_count == 0,
        detail: if boilerplate_count == 0 {
            "No monologue boilerplate detected.".to_string()
        } else {
            format!("Monologue boilerplate entries detected: {}", boilerplate_count)
        },
    });

    // Monologue numeric grounding check
    let telemetry_rows = sqlx::query(
        "SELECT key, value FROM kv_store WHERE key LIKE 'telemetry.%'",
    )
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();
    let mut telemetry_values: Vec<f32> = Vec::new();
    for row in telemetry_rows {
        let key: String = row.get("key");
        let value: String = row.get("value");
        if key.starts_with("telemetry.") {
            if let Ok(parsed) = value.trim().parse::<f32>() {
                telemetry_values.push(parsed);
            }
        }
    }
    let last_user_input: Option<String> = sqlx::query_scalar(
        "SELECT content FROM messages WHERE conversation_id = ? AND role = 'user' ORDER BY datetime(created_at) DESC LIMIT 1",
    )
    .bind(conversation_id)
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten();
    let last_user_input = last_user_input.unwrap_or_default();
    let monologue_rows = sqlx::query(
        "SELECT thought FROM inner_monologue_entries WHERE conversation_id = ? ORDER BY datetime(created_at) DESC LIMIT 50",
    )
    .bind(conversation_id)
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();
    let mut numeric_invalid = 0usize;
    for row in monologue_rows {
        let thought: String = row.get("thought");
        let tokens = extract_numeric_tokens(&thought);
        if tokens.iter().any(|t| !numeric_token_allowed(t, &telemetry_values, &last_user_input)) {
            numeric_invalid += 1;
        }
    }
    results.push(CognitiveCheckResult {
        name: "monologue_numeric_grounding".to_string(),
        passed: numeric_invalid == 0,
        detail: if numeric_invalid == 0 {
            "Monologue numeric claims align with telemetry/user input.".to_string()
        } else {
            format!("Monologue numeric claims failed grounding: {}", numeric_invalid)
        },
    });

    let orphaned_evidence: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ics_evidence_events e
         LEFT JOIN ics_beliefs b ON e.belief_id = b.id
         WHERE b.id IS NULL",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "evidence_integrity".to_string(),
        passed: orphaned_evidence == 0,
        detail: if orphaned_evidence == 0 {
            "No evidence events reference missing beliefs.".to_string()
        } else {
            format!("Evidence events with missing beliefs: {}", orphaned_evidence)
        },
    });

    let ds_ask_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'intent_delivered'
           AND json_extract(payload, '$.candidate_kind') = 'AskUserQuestion'
           AND json_extract(payload, '$.source') = 'monologue'",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "ds_ask_user_question_surface".to_string(),
        passed: ds_ask_count > 0,
        detail: if ds_ask_count > 0 {
            format!("DS ask_user_question surfaced: {}", ds_ask_count)
        } else {
            "No surfaced DS ask_user_question intents detected.".to_string()
        },
    });

    let monologue_leak: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM messages m
         JOIN inner_monologue_entries e ON TRIM(m.content) = TRIM(e.thought)
         WHERE m.role = 'assistant'",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "no_verbatim_monologue_in_assistant".to_string(),
        passed: monologue_leak == 0,
        detail: if monologue_leak == 0 {
            "No assistant messages match inner monologue verbatim.".to_string()
        } else {
            format!("Assistant messages matching monologue text: {}", monologue_leak)
        },
    });

    let monologue_prompt_injected: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM system_logs
         WHERE json_extract(payload, '$.event') = 'prompt_metrics'
           AND EXISTS (
             SELECT 1 FROM json_each(payload, '$.section_sizes')
              WHERE json_extract(value, '$.title') = 'Monologue Intent'
                AND json_extract(value, '$.tokens') > 0
           )",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "monologue_intent_prompt_injected".to_string(),
        passed: monologue_prompt_injected > 0,
        detail: if monologue_prompt_injected > 0 {
            format!("Monologue Intent injected in prompts: {}", monologue_prompt_injected)
        } else {
            "No prompt metrics with Monologue Intent section detected.".to_string()
        },
    });

    let summary_echo: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM messages m
         JOIN conversation_summaries s ON m.conversation_id = s.conversation_id
         WHERE m.role = 'assistant'
           AND m.status = 'complete'
           AND TRIM(m.content) = TRIM(s.summary)",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "summary_echo_guard".to_string(),
        passed: summary_echo == 0,
        detail: if summary_echo == 0 {
            "No assistant messages equal rolling summary text.".to_string()
        } else {
            format!("Assistant messages matching rolling summary: {}", summary_echo)
        },
    });

    let self_awareness_dump: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM messages
         WHERE role = 'assistant'
           AND (
             LOWER(content) LIKE '%self aware%'
             OR LOWER(content) LIKE '%self-awareness%'
             OR LOWER(content) LIKE '%self awareness%'
             OR LOWER(content) LIKE '%conscious%'
             OR LOWER(content) LIKE '%sentient%'
           )
           AND (
             (CASE WHEN LOWER(content) LIKE '%controller gate%' THEN 1 ELSE 0 END)
           + (CASE WHEN LOWER(content) LIKE '%telemetry%' THEN 1 ELSE 0 END)
           + (CASE WHEN LOWER(content) LIKE '%tool manifest%' THEN 1 ELSE 0 END)
           + (CASE WHEN LOWER(content) LIKE '%kv memory%' THEN 1 ELSE 0 END)
           + (CASE WHEN LOWER(content) LIKE '%capability manifest%' THEN 1 ELSE 0 END)
           ) >= 2
           AND (
             LOWER(content) NOT LIKE '%self-report%'
             AND LOWER(content) NOT LIKE '%provisional self-report%'
             AND LOWER(content) NOT LIKE '%confidence:%'
             AND LOWER(content) NOT LIKE '%uncertainty:%'
           )",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "self_awareness_no_system_dump".to_string(),
        passed: self_awareness_dump == 0,
        detail: if self_awareness_dump == 0 {
            "Self-awareness replies avoid system dump phrasing.".to_string()
        } else {
            format!("Self-awareness replies with system dump phrasing: {}", self_awareness_dump)
        },
    });

    let monologue_surface_leak: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM messages
         WHERE role = 'assistant'
           AND json_extract(metadata, '$.origin') = 'monologue'
           AND (json_extract(metadata, '$.surface') IS NULL OR json_extract(metadata, '$.surface') = 0)",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap_or(0);
    results.push(CognitiveCheckResult {
        name: "monologue_surface_gate".to_string(),
        passed: monologue_surface_leak == 0,
        detail: if monologue_surface_leak == 0 {
            "No monologue-origin assistant messages without surface flag.".to_string()
        } else {
            format!("Monologue-origin assistant messages without surface flag: {}", monologue_surface_leak)
        },
    });

    let stream_rows = sqlx::query(
        "SELECT run_id, timestamp FROM system_logs
         WHERE json_extract(payload, '$.event') = 'stream_end'
           AND run_id IS NOT NULL",
    )
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();
    let complete_rows = sqlx::query(
        "SELECT run_id, timestamp FROM system_logs
         WHERE json_extract(payload, '$.event') = 'run_completed'
           AND run_id IS NOT NULL",
    )
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();
    let mut stream_map: HashMap<String, DateTime<Utc>> = HashMap::new();
    for row in stream_rows {
        let run_id: Option<String> = row.try_get("run_id").ok();
        let ts: Option<String> = row.try_get("timestamp").ok();
        if let (Some(id), Some(raw_ts)) = (run_id, ts) {
            if let Some(parsed) = parse_db_ts(&raw_ts) {
                stream_map.insert(id, parsed);
            }
        }
    }
    let mut max_gap_ms: i64 = 0;
    let mut gap_samples = 0usize;
    for row in complete_rows {
        let run_id: Option<String> = row.try_get("run_id").ok();
        let ts: Option<String> = row.try_get("timestamp").ok();
        let Some(id) = run_id else { continue };
        let Some(end_raw) = ts else { continue };
        let Some(end_ts) = parse_db_ts(&end_raw) else { continue };
        if let Some(start_ts) = stream_map.get(&id) {
            let diff = end_ts.signed_duration_since(*start_ts).num_milliseconds();
            if diff > max_gap_ms {
                max_gap_ms = diff;
            }
            gap_samples += 1;
        }
    }
    let stream_gap_pass = gap_samples == 0 || max_gap_ms <= 3000;
    results.push(CognitiveCheckResult {
        name: "stream_end_to_run_complete_ms".to_string(),
        passed: stream_gap_pass,
        detail: if gap_samples == 0 {
            "No streamed runs with completion timing recorded.".to_string()
        } else {
            format!("Max stream_end to run_completed gap: {} ms (samples: {})", max_gap_ms, gap_samples)
        },
    });

    let recent_monologue_rows = sqlx::query(
        "SELECT thought FROM inner_monologue_entries
         WHERE conversation_id = ?
         ORDER BY datetime(created_at) DESC
         LIMIT 12",
    )
    .bind(conversation_id)
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();
    let mut recent_thoughts: Vec<String> = recent_monologue_rows
        .iter()
        .filter_map(|row| row.try_get::<String, _>("thought").ok())
        .collect();
    recent_thoughts.reverse();
    let mut max_consecutive = 0usize;
    let mut current = 0usize;
    let mut last: Option<String> = None;
    for thought in recent_thoughts {
        let trimmed = thought.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        if last.as_deref() == Some(trimmed.as_str()) {
            current += 1;
        } else {
            current = 1;
            last = Some(trimmed);
        }
        if current > max_consecutive {
            max_consecutive = current;
        }
    }
    results.push(CognitiveCheckResult {
        name: "monologue_loop_breaker".to_string(),
        passed: max_consecutive <= 2,
        detail: if max_consecutive <= 2 {
            "No monologue loops detected in recent turns.".to_string()
        } else {
            format!("Monologue repeats detected (max consecutive repeats: {})", max_consecutive)
        },
    });

    results
}

#[cfg(test)]
mod tests {
    use super::numeric_token_allowed;

    #[test]
    fn numeric_token_requires_allowlist_or_user_input() {
        assert!(!numeric_token_allowed("5", &[], ""));
        assert!(numeric_token_allowed("5", &[5.0], ""));
        assert!(numeric_token_allowed("5", &[], "the number 5"));
        assert!(!numeric_token_allowed("9", &[1.0], ""));
        assert!(numeric_token_allowed("3.14", &[3.14], ""));
    }
}
