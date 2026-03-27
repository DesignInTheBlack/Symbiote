use super::super::*;
use sqlx::Row;
use chrono::{TimeZone, Utc};
use crate::core::system_log;

const SELF_REPORT_RELIABILITY_WARN: f32 = 0.45;

fn content_has_self_report(content: &str) -> bool {
    let lower = content.to_lowercase();
    lower.contains("self-report summary")
        || lower.contains("self report summary")
        || lower.contains("operational status:")
        || lower.contains("self-report")
}

impl Kernel {
    pub(crate) fn finalize_decision_report(
        &self,
        decision: &mut KernelDecision,
        state: &KernelState,
        prompt_build: Option<&CorePromptBuild>,
        anchor_hits: Option<usize>,
        response: Option<&str>,
        commit_result: &CommitResult,
        tool_result: Option<&ToolExecutionResult>,
        best_effort_dropped: bool,
    ) -> Option<String> {
        decision.report.anchor_hits = anchor_hits;
        decision.report.prompt_tokens_used = prompt_build.map(|p| p.total_tokens);
        decision.report.tier_trim_summary = prompt_build.and_then(|p| p.tier_trim_summary.clone());
        decision.report.monologue_tick_outcome = state.last_monologue_tick_outcome.clone();
        decision.report.monologue_status_emitted = state.last_monologue_status_emitted;
        decision.report.monologue_visible = state.last_monologue_visible;
        decision.report.background_jobs_dropped = Some(best_effort_dropped);
        decision.report.stop_scope = state.stop_scope.clone();
        decision.report.normalized_stop_reasons = normalize_stop_reasons(&decision.report.stop_reasons);

        if let Some(first) = commit_result.tool_dispatches.first() {
            decision.report.tool_attempted = Some(first.tool_name.clone());
        }

        let selected_action = SelectedAction::from_str(&decision.report.selected_action);
        if let Some(result) = tool_result {
            decision.report.tool_outcome = Some(tool_outcome_from_result(result).as_str().to_string());
        } else if matches!(selected_action, SelectedAction::ToolAttempt) && commit_result.tool_dispatches.is_empty() {
            decision.report.tool_outcome = Some(ToolOutcome::Blocked.as_str().to_string());
        }

        let mut response_override: Option<String> = None;
        let response_text = response.unwrap_or("");
        let mut delivered = match selected_action {
            SelectedAction::DirectAnswer => !response_text.trim().is_empty(),
            SelectedAction::BoundedAnswer => !response_text.trim().is_empty() && response_has_assumptions(response_text),
            SelectedAction::ClarifyingQuestion => {
                response_is_question(response_text) || commit_result.ask_question.is_some()
            }
            SelectedAction::ToolAttempt => !commit_result.tool_dispatches.is_empty(),
            SelectedAction::ExplainBlockers => !response_text.trim().is_empty(),
        };
        let mut artifact = match selected_action {
            SelectedAction::ClarifyingQuestion => DeliveryArtifact::QuestionMark,
            SelectedAction::ToolAttempt => DeliveryArtifact::ToolCall,
            SelectedAction::BoundedAnswer => DeliveryArtifact::BoundedAnswerBlock,
            SelectedAction::ExplainBlockers => DeliveryArtifact::ExplainBlockers,
            SelectedAction::DirectAnswer => DeliveryArtifact::None,
        };

        if selected_action != SelectedAction::ExplainBlockers
            && decision.report.allowed_capabilities.emit
            && !delivered
        {
            let unblock = decision
                .report
                .unblock_instructions
                .clone()
                .unwrap_or_else(|| unblock_instructions_for_reasons(&decision.report.stop_reasons));
            let fallback = explain_blockers_message(&decision.report.stop_reasons, &unblock);
            response_override = Some(fallback);
            decision.report.selected_action = SelectedAction::ExplainBlockers.as_str().to_string();
            decision.report.fallback_used = true;
            decision.report.fallback_type = Some(SelectedAction::ExplainBlockers.as_str().to_string());
            decision.report.unblock_instructions = Some(unblock);
            delivered = true;
            artifact = DeliveryArtifact::ExplainBlockers;
        }

        let final_response = response_override
            .as_deref()
            .or_else(|| response)
            .unwrap_or("");
        let user_visible = !final_response.trim().is_empty();
        let policy_boilerplate = response_is_policy_boilerplate(final_response);
        let has_question = response_is_question(final_response) || commit_result.ask_question.is_some();
        let has_next_step = response_has_next_step(final_response);
        let tool_attempted = decision.report.tool_attempted.is_some();
        let bounded_answer = response_has_assumptions(final_response);
        let explicit_blockers = decision.report.unblock_instructions.is_some()
            && matches!(
                SelectedAction::from_str(&decision.report.selected_action),
                SelectedAction::ExplainBlockers
            );

        decision.report.noop =
            (!user_visible || policy_boilerplate) && !has_question && !has_next_step && !tool_attempted;
        decision.report.minimally_helpful = bounded_answer || has_question || has_next_step || explicit_blockers;
        decision.report.selected_action_delivered = delivered;
        decision.report.delivery_artifact = artifact.as_str().to_string();
        if decision.report.cannot_respond && decision.report.unblock_instructions.is_none() {
            decision.report.unblock_instructions =
                Some(unblock_instructions_for_reasons(&decision.report.stop_reasons));
        }
        if decision.report.rationale.is_none() {
            if let Some(source) = decision.report.selected_action_source.clone() {
                decision.report.rationale = Some(source);
            } else if !decision.report.normalized_stop_reasons.is_empty() {
                decision.report.rationale =
                    Some(format!("blocked_by:{}", decision.report.normalized_stop_reasons.join(",")));
            }
        }

        response_override
    }

    pub(crate) async fn finalize_assistant_message(
        &self,
        run_id: &str,
        content: &str,
        response_origin: ResponseOrigin,
        gate_decision: Option<String>,
        gate_notice: Option<String>,
        gate_reasons: Option<Vec<String>>,
        extra_notice: Option<String>,
        self_report_source: Option<&str>,
    ) -> Option<String> {
        let message_row = sqlx::query(
            "SELECT message_id, conversation_id FROM messages
             WHERE run_id = ? AND role = 'assistant'
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .bind(run_id)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten();

        let Some(message_row) = message_row else {
            return None;
        };
        let message_id: String = message_row.try_get("message_id").ok()?;
        let conversation_id: String = message_row.try_get("conversation_id").ok()?;

        let existing_meta: Option<String> = sqlx::query_scalar(
            "SELECT metadata FROM messages WHERE message_id = ?",
        )
        .bind(&message_id)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten();

        let mut meta_value = existing_meta
            .as_deref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
            .unwrap_or_else(|| json!({}));
        if !meta_value.is_object() {
            meta_value = json!({});
        }
        let obj = match meta_value.as_object_mut() {
            Some(obj) => obj,
            None => return None,
        };
        obj.entry("source".to_string()).or_insert(json!("assistant"));
        obj.entry("origin".to_string()).or_insert(json!("assistant"));
        obj.insert(
            "response_origin".to_string(),
            json!(response_origin.as_str()),
        );
        obj.entry("surface".to_string()).or_insert(json!(true));
        obj.entry("candidate_kind".to_string())
            .or_insert(json!("EmitMessage"));
        obj.entry("candidate_id".to_string())
            .or_insert(json!(&message_id));
        obj.entry("bridge_id".to_string()).or_insert(json!(null));
        if !obj.contains_key("evidence_event_ids") {
            if let Ok(Some(raw_ids)) = sqlx::query_scalar::<_, String>(
                "SELECT evidence_event_ids
                 FROM decision_reports
                 WHERE run_id = ?
                 ORDER BY datetime(created_at) DESC, rowid DESC
                 LIMIT 1",
            )
            .bind(run_id)
            .fetch_optional(&self.db.pool)
            .await
            {
                if let Ok(mut ids) = serde_json::from_str::<Vec<i64>>(&raw_ids) {
                    ids.sort();
                    ids.dedup();
                    obj.insert("evidence_event_ids".to_string(), json!(ids));
                }
            }
        }
        if let Some(decision) = gate_decision {
            obj.insert("gate_decision".to_string(), json!(decision));
        }
        let mut final_content = content.to_string();
        if let Some(notice) = gate_notice.clone() {
            obj.insert("gate_notice".to_string(), json!(notice.clone()));
        }
        if let Some(notice) = extra_notice {
            obj.insert("extra_notice".to_string(), json!(notice));
        }
        if let Some(reasons) = gate_reasons {
            obj.insert("gate_reasons".to_string(), json!(reasons));
        }
        if content_has_self_report(&final_content) {
            let mut self_report_evidence_ids: Vec<i64> = obj
                .get("evidence_event_ids")
                .and_then(|value| value.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|v| v.as_i64())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if let Ok(mut ids) =
                crate::core::self_model_controller::collect_unified_self_evidence_ids(&self.db).await
            {
                self_report_evidence_ids.append(&mut ids);
            }
            self_report_evidence_ids.sort();
            self_report_evidence_ids.dedup();

            let controller_state = self.db.get_controller_state().await.ok().flatten().unwrap_or_default();
            let mut confidence = controller_state.confidence.clamp(0.0, 1.0);
            let mut uncertainty = controller_state.uncertainty.clamp(0.0, 1.0);
            let mut reliability = 0.5_f32;
            if let Ok(model) = self.db.get_self_model().await {
                if let Some(value) = model
                    .unified_state
                    .get("self_model_reliability")
                    .and_then(|v| v.as_f64())
                {
                    reliability = value as f32;
                }
            }
            if reliability < SELF_REPORT_RELIABILITY_WARN {
                confidence = (confidence * 0.6).clamp(0.0, 1.0);
                uncertainty = (uncertainty + 0.2).clamp(0.0, 1.0);
            }
            let mut constraints: Vec<String> = Vec::new();
            if controller_state.verification_needed {
                constraints.push("verification_needed".to_string());
            }
            if controller_state.reanchor_needed {
                constraints.push("reanchor_needed".to_string());
            }
            if !controller_state.missing_fields.is_empty() {
                constraints.push(format!(
                    "missing_fields: {}",
                    controller_state.missing_fields.join(", ")
                ));
            }
            if reliability < SELF_REPORT_RELIABILITY_WARN {
                constraints.push("self_model_reliability_low".to_string());
            }
            let status = if reliability < SELF_REPORT_RELIABILITY_WARN || controller_state.drift_score > 0.6 {
                "degraded"
            } else {
                "operational"
            };
            let speculative = self_report_evidence_ids.is_empty() || reliability < SELF_REPORT_RELIABILITY_WARN;
            let source = self_report_source.unwrap_or("implicit");
            obj.insert(
                "self_report".to_string(),
                json!({
                    "status": status,
                    "confidence": confidence,
                    "uncertainty": uncertainty,
                    "constraints": constraints,
                    "evidence_event_ids": self_report_evidence_ids,
                    "speculative": speculative,
                    "self_model_reliability": reliability,
                    "source": source,
                }),
            );

            if self_report_evidence_ids.is_empty() {
                let marker = "provisional self-report";
                if !final_content.to_lowercase().contains(marker) {
                    final_content = format!(
                        "{}\n\nNote: provisional self-report (no evidence).",
                        final_content.trim_end()
                    );
                }
            }
        }

        let metadata = serde_json::to_string(&meta_value).unwrap_or_else(|_| "{}".to_string());
        let _ = sqlx::query("UPDATE messages SET status = 'complete', content = ?, metadata = ? WHERE message_id = ?")
            .bind(&final_content)
            .bind(metadata)
            .bind(&message_id)
            .execute(&self.db.pool)
            .await;
        let _ = self.app_handle.emit("message_updated", ());
        if let Ok(Some(started_at)) = sqlx::query_scalar::<_, String>(
            "SELECT started_at FROM runs WHERE run_id = ?",
        )
        .bind(run_id)
        .fetch_optional(&self.db.pool)
        .await
        {
            let parsed = chrono::DateTime::parse_from_rfc3339(&started_at)
                .map(|dt| dt.with_timezone(&Utc))
                .or_else(|_| {
                    chrono::NaiveDateTime::parse_from_str(&started_at, "%Y-%m-%d %H:%M:%S")
                        .map(|dt| Utc.from_utc_datetime(&dt))
                        .or_else(|_| {
                            chrono::NaiveDateTime::parse_from_str(&started_at, "%Y-%m-%d %H:%M:%S%.f")
                                .map(|dt| Utc.from_utc_datetime(&dt))
                        })
                });
            if let Ok(started_at) = parsed {
                let latency_ms = Utc::now()
                    .signed_duration_since(started_at)
                    .num_milliseconds()
                    .max(0);
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(run_id),
                    None,
                    json!({
                        "event": "message_visible",
                        "conversation_id": conversation_id,
                        "message_id": message_id,
                        "latency_ms": latency_ms,
                    }),
                )
                .await;
            }
        }

        let subject_snapshot_present: Option<i64> = sqlx::query_scalar(
            "SELECT 1 FROM subject_snapshots WHERE conversation_id = ? LIMIT 1",
        )
        .bind(&conversation_id)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten();
        if subject_snapshot_present.is_none() {
            let mut snapshot_state = self.load_state(&conversation_id).await;
            let _ = self
                .build_and_persist_subject_snapshot(&mut snapshot_state, Some(run_id), Some(run_id), "post_finalize")
                .await;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                None,
                json!({
                    "event": "subject_snapshot_backfill",
                    "reason": "post_finalize",
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                }),
            )
            .await;
        }

        let _ = qualia::maybe_auto_label_for_message(
            &self.db,
            Some(&self.app_handle),
            &conversation_id,
            &message_id,
            Some(run_id),
            true,
        )
        .await;

        let tool_dispatches: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM tool_dispatches WHERE run_id = ?",
        )
        .bind(run_id)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0);
        if tool_dispatches == 0 {
            let control_map = crate::core::system_controls::load_control_map(&self.db).await;
            let tool_mode = crate::core::system_controls::mode_for("tool_execution", &control_map);
            if !crate::core::system_controls::mode_is_off(&tool_mode) {
                let _ = self.run_tool_heartbeat_tick().await;
            }
        }

        // Persist internal state snapshot per run
        let controller_state = self.db.get_controller_state().await.ok().flatten();
        let confidence = controller_state
            .as_ref()
            .map(|c| c.confidence as f64)
            .unwrap_or(0.5);
        let uncertainty = controller_state
            .as_ref()
            .map(|c| c.uncertainty as f64)
            .unwrap_or(0.5);
        let mut qualia_tag: Option<String> = None;
        let mut qualia_intensity: Option<f64> = None;
        if let Ok(Some(row)) = sqlx::query(
            "SELECT tag, intensity FROM qualia_labels
             WHERE event_id = ?
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .bind(&message_id)
        .fetch_optional(&self.db.pool)
        .await
        {
            qualia_tag = row.try_get("tag").ok();
            qualia_intensity = row.try_get("intensity").ok();
        }
        let mut self_model = match self.db.get_self_model().await {
            Ok(model) => Some(model),
            Err(err) => {
                let _ = self.db.ensure_self_model_row().await;
                match self.db.get_self_model().await {
                    Ok(model) => Some(model),
                    Err(_) => {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "kernel",
                            Some(run_id),
                            None,
                            json!({
                                "event": "internal_state_snapshot_skipped",
                                "reason": "self_model_missing",
                                "message_id": message_id,
                                "conversation_id": conversation_id,
                                "error": err.to_string(),
                            }),
                        )
                        .await;
                        None
                    }
                }
            }
        };
        if let Some(self_model) = self_model.take() {
            match self
                .db
                .insert_internal_state_snapshot(
                    Some(run_id),
                    Some(&message_id),
                    Some(&conversation_id),
                    confidence,
                    uncertainty,
                    qualia_tag.as_deref(),
                    qualia_intensity,
                    &self_model.internal_state_summary,
                )
                .await
            {
                Ok(()) => {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        Some(run_id),
                        None,
                        json!({
                            "event": "internal_state_snapshot_written",
                            "message_id": message_id,
                            "conversation_id": conversation_id,
                            "confidence": confidence,
                            "uncertainty": uncertainty,
                            "qualia_tag": qualia_tag,
                            "qualia_intensity": qualia_intensity,
                        }),
                    )
                    .await;
                }
                Err(err) => {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        Some(run_id),
                        None,
                        json!({
                            "event": "internal_state_snapshot_failed",
                            "message_id": message_id,
                            "conversation_id": conversation_id,
                            "error": err.to_string(),
                        }),
                    )
                    .await;
                }
            }
        }

        if let Ok(Some(ts)) = sqlx::query_scalar::<_, String>(
            "SELECT timestamp FROM system_logs WHERE run_id = ? AND json_extract(payload, '$.event') = 'stream_end' ORDER BY datetime(timestamp) DESC LIMIT 1",
        )
        .bind(run_id)
        .fetch_optional(&self.db.pool)
        .await
        {
            if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(&ts) {
                let delta_ms = Utc::now()
                    .signed_duration_since(parsed.with_timezone(&Utc))
                    .num_milliseconds()
                    .max(0);
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(run_id),
                    None,
                    json!({
                        "event": "stream_end_to_idle",
                        "stream_end_to_idle_ms": delta_ms,
                    }),
                )
                .await;
            }
        }

        Some(final_content)
    }
}
