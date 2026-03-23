use super::super::*;
use crate::core::kernel::run::build_qualia_modulation_context;

pub(crate) struct ArbitrationPhaseResult {
    pub decision: KernelDecision,
    pub anchor_hits: usize,
    pub gates_ms: i64,
}

impl Kernel {
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn run_arbitration_phase(
        &self,
        candidates: &mut Vec<Candidate>,
        state: &mut KernelState,
        all_outcomes: &[Outcome],
        conversation_id: &str,
        run_id: &str,
        trace_id: &str,
        settings: &crate::models::Settings,
        response_content_no_tags: &str,
        self_awareness_query: bool,
        mut gates_started: Option<Instant>,
        created_at: &mut i64,
    ) -> Result<ArbitrationPhaseResult, String> {
        let prev_ask_loop = state.ask_loop_breaker_triggered;
        let prev_tool_loop = state.tool_loop_breaker_triggered;
        self.apply_loop_detection(candidates, state, settings);
        self.log_loop_breakers(Some(run_id), Some(trace_id), state, prev_ask_loop, prev_tool_loop)
            .await;

        let mut decision_state = state.clone();
        self.apply_outcomes(&mut decision_state, all_outcomes).await;
        if let Some(promo) = self
            .maybe_semantic_promotion_candidate(conversation_id, &decision_state, settings, created_at)
            .await
        {
            candidates.push(promo);
        }

        let mut subject_snapshot_hash: Option<String> = None;
        let mut subject_state_snapshot: Option<subject_state::SubjectState> = None;
        if let Some((subject_state, snapshot)) = self
            .build_and_persist_subject_snapshot(&mut decision_state, Some(run_id), Some(run_id), "turn_arbitration")
            .await
        {
            subject_snapshot_hash = Some(snapshot.snapshot_hash.clone());
            subject_state_snapshot = Some(subject_state);
            state.last_subject_snapshot_hash = Some(snapshot.snapshot_hash);
            state.last_subject_snapshot_at = Some(snapshot.timestamp);
        }

        let mut gates_ms: i64 = 0;
        if let Some(started) = gates_started.take() {
            gates_ms = started.elapsed().as_millis() as i64;
        }
        if gates_ms > 0 {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                Some(trace_id),
                json!({
                    "event": "timing_response_parse",
                    "duration_ms": gates_ms,
                }),
            )
            .await;
            self.update_latency_avg("response_parse", gates_ms).await;
        }

        let _ = advance_run_phase(
            &self.db.pool,
            Some(&self.app_handle),
            run_id,
            RunPhase::Arbitration,
            Some("arbitration"),
        )
        .await;

        let anchor_hits = {
            let tool_names = self.tools.allowed_tool_names(settings);
            let anchor_vocab = build_anchor_vocab(&decision_state, &tool_names);
            count_anchor_hits(response_content_no_tags, &anchor_vocab)
        };
        let wave_context = self
            .wave_arbitration_context(Some(run_id), Some(trace_id))
            .await;
        let qualia_context = subject_state_snapshot
            .as_ref()
            .and_then(|state| build_qualia_modulation_context(&state.qualia));
        let residual_context = self
            .residual_influence_context(&decision_state, subject_state_snapshot.as_ref())
            .await;
        let mut decision = self.arbitrate(
            candidates,
            settings,
            &decision_state,
            false,
            Some(anchor_hits),
            wave_context,
            qualia_context,
            residual_context,
            Some(run_id),
        );
        if decision.report.snapshot_hash.is_none() {
            decision.report.snapshot_hash = subject_snapshot_hash
                .clone()
                .or_else(|| state.last_subject_snapshot_hash.clone());
        }

        if let (Some(snapshot_hash), Some(subject_state), Some(candidate)) = (
            subject_snapshot_hash.clone(),
            subject_state_snapshot.clone(),
            decision.accepted.first().cloned(),
        ) {
            let mut proposal = subject_controller::build_action_proposal(&candidate);
            let plan_hash = proposal.plan_hash.clone();
            let proposal_id = proposal.proposal_id.clone();
            if !plan_hash.trim().is_empty() {
                decision_state.last_plan_hash = Some(plan_hash.clone());
                state.last_plan_hash = Some(plan_hash.clone());
            }

            let mut snapshot_hash = snapshot_hash;
            let mut subject_state = subject_state;
            if state.last_plan_hash.is_some() {
                if let Some((plan_state, plan_snapshot)) = self
                    .build_and_persist_subject_snapshot(&mut decision_state, Some(run_id), Some(run_id), "plan_verification")
                    .await
                {
                    snapshot_hash = plan_snapshot.snapshot_hash.clone();
                    subject_state = plan_state;
                    subject_state_snapshot = Some(subject_state.clone());
                    state.last_subject_snapshot_hash = Some(snapshot_hash.clone());
                    state.last_subject_snapshot_at = Some(plan_snapshot.timestamp.clone());
                }
            }

            let verification = subject_controller::verify_action_proposal(&proposal, &subject_state.world_model);
            let plan_state =
                super::super::arbitration::plan_state_for_verification(&verification)
                    .to_string();
            proposal.plan_state = plan_state.clone();
            decision.report.plan_hash = Some(plan_hash);
            decision.report.proposal_id = Some(proposal_id.clone());
            decision.report.plan_state = Some(plan_state.clone());
            super::super::arbitration::apply_plan_verification_report(&mut decision.report, &verification);
            if plan_state == "verified" {
                state.workspace_active_plan_id = Some(proposal_id.clone());
                if let Err(err) = self
                    .db
                    .set_workspace_active_plan(conversation_id, Some(&proposal_id))
                    .await
                {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "warn",
                        "kernel",
                        Some(run_id),
                        Some(trace_id),
                        json!( {
                            "event": "active_plan_set_failed",
                            "proposal_id": proposal_id,
                            "error": err.to_string(),
                        }),
                    )
                    .await;
                } else {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        Some(run_id),
                        Some(trace_id),
                        json!( {
                            "event": "active_plan_set",
                            "proposal_id": proposal_id,
                        }),
                    )
                    .await;
                }
            }

            let signals = self
                .compute_gate_signals(
                    &mut decision_state,
                    &subject_state,
                    Some(&decision),
                    &candidate,
                    anchor_hits,
                    settings,
                )
                .await;
            let soft_gate = subject_controller::build_gate_decision(
                &subject_state,
                &candidate,
                &decision_state.stop_state,
                &signals,
            );
            let legacy_gate =
                subject_controller::build_gate_decision_legacy(&subject_state, &candidate, &decision_state.stop_state);
            let soft_decision = soft_gate.decision.clone();
            let legacy_decision = legacy_gate.decision.clone();
            let rollout_percent = settings.gate_rollout_percent.unwrap_or(100).clamp(0, 100);
            let shadow_mode = settings.gate_shadow_mode.unwrap_or(false);
            let rollout_bucket = gate_rollout_bucket(conversation_id);
            let use_soft_gate = !shadow_mode && (rollout_percent >= 100 || rollout_bucket < rollout_percent);
            let (mut gate, _shadow_gate) = if use_soft_gate {
                (soft_gate, legacy_gate)
            } else {
                (legacy_gate, soft_gate)
            };
            super::super::arbitration::apply_plan_verification_to_gate_decision(&mut gate, &verification);
            let gate_reasons_log = serde_json::from_str::<Value>(&gate.evidence_refs_json)
                .ok()
                .and_then(|value| value.get("reasons").cloned())
                .unwrap_or_else(|| json!([]));
            state.gate_high_risk_streak = decision_state.gate_high_risk_streak;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                Some(run_id),
                Some(trace_id),
                json!({
                    "event": "gate_decision_inputs",
                    "candidate_id": candidate.id,
                    "candidate_kind": format!("{:?}", candidate.kind),
                    "anchor_hits": anchor_hits,
                    "signals": signals,
                    "soft_decision": soft_decision,
                    "legacy_decision": legacy_decision,
                    "enforced_decision": gate.decision,
                    "gate_reasons": gate_reasons_log,
                    "plan_verification": {
                        "outcome": verification.outcome,
                        "confidence": verification.confidence,
                        "assumptions_checked": verification.assumptions_checked,
                        "assumptions_failed": verification.assumptions_failed,
                        "reasons": verification.reasons,
                        "conflict_topics": verification.conflict_topics,
                    },
                    "shadow_mode": shadow_mode,
                    "rollout_percent": rollout_percent,
                    "rollout_bucket": rollout_bucket,
                    "organism": subject_state.organism,
                }),
            )
            .await;
            if let Err(err) =
                subject_controller::persist_action_proposal(&self.db, &snapshot_hash, &proposal).await
            {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    Some(run_id),
                    Some(trace_id),
                    json!({
                        "event": "action_proposal_failed",
                        "proposal_id": proposal_id,
                        "error": err,
                    }),
                )
                .await;
            } else {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(run_id),
                    Some(trace_id),
                    json!({
                        "event": "action_proposal_written",
                        "proposal_id": proposal_id,
                        "snapshot_hash": snapshot_hash,
                        "intent": proposal.intent,
                    }),
                )
                .await;
            }
            if let Err(err) =
                subject_controller::persist_gate_decision(&self.db, &snapshot_hash, &proposal_id, &gate).await
            {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    Some(run_id),
                    Some(trace_id),
                    json!({
                        "event": "gate_decision_failed",
                        "decision_id": gate.decision_id,
                        "error": err,
                    }),
                )
                .await;
            } else {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    Some(run_id),
                    Some(trace_id),
                    json!({
                        "event": "gate_decision_written",
                        "decision_id": gate.decision_id,
                        "proposal_id": proposal_id,
                        "snapshot_hash": snapshot_hash,
                        "decision": gate.decision,
                    }),
                )
                .await;
            }
            decision.report.snapshot_hash = Some(snapshot_hash);
            decision.report.gate_decision_id = Some(gate.decision_id.clone());
            decision.report.gate_decision = Some(gate.decision.clone());
            let reasons: Vec<String> = serde_json::from_str::<Value>(&gate.evidence_refs_json)
                .ok()
                .and_then(|v| v.get("reasons").and_then(|r| r.as_array()).cloned())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|item| item.as_str())
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if !reasons.is_empty() {
                decision.report.gate_reasons = Some(reasons.clone());
            }
            decision.report.gate_notice = gate_notice_for(&gate.decision, &reasons);
            if !gate_allows_response(&gate.decision) {
                let reason_text = if reasons.is_empty() {
                    String::new()
                } else {
                    format!(" Reasons: {}", reasons.join(", "))
                };
                let (kind, message) = match gate.decision.as_str() {
                    "VERIFY" => (
                        CandidateKind::AskUserQuestion,
                        format!(
                            "I need to verify before proceeding. Can you confirm or provide evidence?{}",
                            reason_text
                        ),
                    ),
                    "DEFER" => (
                        CandidateKind::EmitMessage,
                        format!(
                            "Deferring action due to current internal constraints.{}",
                            reason_text
                        ),
                    ),
                    "DENY" => (
                        CandidateKind::EmitMessage,
                        format!(
                            "I cannot proceed with that action right now.{}",
                            reason_text
                        ),
                    ),
                    _ => (CandidateKind::EmitMessage, "Unable to proceed.".to_string()),
                };
                let payload = json!({
                    "content": message,
                    "question": message,
                    "bounded": true,
                    "gate_decision": gate.decision,
                    "gate_reasons": reasons,
                });
                let fallback_kind = kind.clone();
                let mut fallback = Candidate {
                    id: Uuid::new_v4().to_string(),
                    kind: fallback_kind.clone(),
                    payload,
                    evidence_event_ids: Vec::new(),
                    belief_ids: Vec::new(),
                    target_scope: None,
                    rationale: Some("gate_decision".to_string()),
                    expected_outcome: None,
                    cost: Some(0),
                    urgency: Some(0),
                    source: "gate_decision".to_string(),
                    priority_class: priority_class_for(&fallback_kind),
                    priority_rank: 0,
                    created_at: state.monologue_count,
                };
                fallback.refresh_meta();
                decision.accepted.clear();
                decision.accepted.push(fallback);
                decision.report.selected_action_source = Some("gate_decision".to_string());
                decision.report.cannot_respond = matches!(gate.decision.as_str(), "DENY" | "DEFER");
                decision.report.stop_reasons.push(StopReason {
                    category: StopReasonCategory::EvidenceBlock,
                    subcode: format!("gate_{}", gate.decision.to_lowercase()),
                    contract: None,
                });
                decision.report.allowed_capabilities.emit = true;
                decision.report.allowed_capabilities.tools = false;
                decision.report.allowed_capabilities.memory_write = false;
                decision.report.allowed_capabilities.self_claims = false;
                decision.report.allowed_capabilities.monologue_run = false;
                decision.report.allowed_capabilities.monologue_emit = false;
                decision.report.allowed_capabilities.background_jobs = false;
            }
        }

        let introspection_summary = if settings.enable_introspection.unwrap_or(true) {
            self.build_introspection_summary(conversation_id, settings, state)
                .await
        } else {
            None
        };
        if let (Some(summary), Some(subject_state), Some(snapshot_hash)) = (
            introspection_summary.as_ref(),
            subject_state_snapshot.as_ref(),
            decision.report.snapshot_hash.as_deref(),
        ) {
            if subject_state.workspace.ignition.active {
                self.record_introspection_entry(&mut decision_state, &decision, subject_state, snapshot_hash, summary)
                    .await;
            }
        }
        if let Some(subject_state) = subject_state_snapshot.as_ref() {
            if !subject_state.workspace.ignition.active && !self_awareness_query {
                let mut scrubbed_any = false;
                for candidate in decision.accepted.iter_mut() {
                    if !is_monologue_source(&candidate.source) {
                        continue;
                    }
                    if let Some(obj) = candidate.payload.as_object_mut() {
                        let mut changed = false;
                        if let Some(content) = obj.get("content").and_then(|v| v.as_str()) {
                            let (updated, did_change) = scrub_non_ignition_language(content);
                            if did_change {
                                obj.insert("content".to_string(), Value::String(updated));
                                changed = true;
                            }
                        }
                        if let Some(question) = obj.get("question").and_then(|v| v.as_str()) {
                            let (updated, did_change) = scrub_non_ignition_language(question);
                            if did_change {
                                obj.insert("question".to_string(), Value::String(updated));
                                changed = true;
                            }
                        }
                        scrubbed_any |= changed;
                    }
                }
                if scrubbed_any {
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        Some(run_id),
                        Some(trace_id),
                        json!({
                            "event": "ignition_language_scrubbed",
                            "reason": "ignition_inactive",
                        }),
                    )
                    .await;
                }
            }
        }

        self.defer_throttled_tools(&mut decision, &decision_state).await;
        self.log_tool_rejections(&decision.rejected).await;
        self.log_tool_bypasses(&decision, &decision_state, settings).await;
        self.enforce_monologue_question_grounding(
            &mut decision,
            settings,
            Some(run_id),
            Some(trace_id),
        )
            .await;
        self.enforce_grounding_on_emits(&mut decision, settings, Some(run_id), Some(trace_id))
            .await;
        if decision
            .accepted
            .iter()
            .all(|candidate| !candidate_user_visible(candidate))
        {
            if let Some(fallback) = self.build_tool_refusal_fallback(&decision, state, settings) {
                decision.accepted.push(fallback);
            }
        }

        Ok(ArbitrationPhaseResult {
            decision,
            anchor_hits,
            gates_ms,
        })
    }
}
