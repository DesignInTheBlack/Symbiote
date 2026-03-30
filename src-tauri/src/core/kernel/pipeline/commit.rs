use super::super::*;
use crate::core::identity;
use crate::core::memory::attention::evidence::{
    evidence_quality_tier,
    quality_floor_for_memory,
    quality_floor_for_self_claim,
};
use crate::core::sensitivity::{phi_consent_allowed, redact_sensitive_json, redact_sensitive_text};
use crate::models::WorkspaceMeta;

fn workspace_before_after_for_payload(
    before: &WorkspaceSnapshot,
    state: &KernelState,
    payload: &Value,
) -> (Option<Value>, Option<Value>) {
    let Some(obj) = payload.as_object() else {
        return (None, None);
    };
    let mut before_map = serde_json::Map::new();
    let mut after_map = serde_json::Map::new();
    for key in obj.keys() {
        match key.as_str() {
            "goal_thread" => {
                before_map.insert("goal_thread".to_string(), json!(before.goal_thread));
                after_map.insert("goal_thread".to_string(), json!(state.workspace_goal_thread));
            }
            "goal_stack" => {
                before_map.insert("goal_stack".to_string(), json!(before.goal_stack));
                after_map.insert("goal_stack".to_string(), json!(state.workspace_goal_stack));
            }
            "open_questions" => {
                before_map.insert("open_questions".to_string(), json!(before.open_questions));
                after_map.insert("open_questions".to_string(), json!(state.workspace_open_questions));
            }
            "active_hypotheses" => {
                before_map.insert("active_hypotheses".to_string(), json!(before.active_hypotheses));
                after_map.insert(
                    "active_hypotheses".to_string(),
                    json!(state.workspace_active_hypotheses),
                );
            }
            "working_set_topics" => {
                before_map.insert(
                    "working_set_topics".to_string(),
                    json!(before.working_set_topics),
                );
                after_map.insert(
                    "working_set_topics".to_string(),
                    json!(state.workspace_working_set_topics),
                );
            }
            "current_focus" => {
                before_map.insert("current_focus".to_string(), json!(before.current_focus));
                after_map.insert("current_focus".to_string(), json!(state.workspace_current_focus));
            }
            "focus_rationale" => {
                before_map.insert(
                    "focus_rationale".to_string(),
                    json!(before.focus_rationale),
                );
                after_map.insert(
                    "focus_rationale".to_string(),
                    json!(state.workspace_focus_rationale),
                );
            }
            "workspace_meta" => {
                before_map.insert("workspace_meta".to_string(), json!(before.workspace_meta));
                after_map.insert("workspace_meta".to_string(), json!(state.workspace_meta));
            }
            _ => {}
        }
    }
    let before_val = if before_map.is_empty() {
        None
    } else {
        Some(Value::Object(before_map))
    };
    let after_val = if after_map.is_empty() {
        None
    } else {
        Some(Value::Object(after_map))
    };
    (before_val, after_val)
}

struct EvidenceAttachOutcome {
    payload: Value,
    has_evidence: bool,
}

async fn ensure_candidate_evidence(
    kernel: &Kernel,
    candidate: &Candidate,
    conversation_id: &str,
    run_id: Option<&str>,
    trace_id: Option<&str>,
    cached_recent_user_evidence: &mut Option<Vec<i64>>,
    allow_fallback: bool,
    category: &str,
) -> EvidenceAttachOutcome {
    let mut payload = candidate.payload.clone();
    let evidence_class = candidate_evidence_class(candidate);
    let evidence_event_ids = extract_id_list(&payload, "evidence_event_ids");
    let belief_ids = extract_id_list(&payload, "belief_ids");
    let mut has_evidence = !evidence_event_ids.is_empty()
        || !belief_ids.is_empty()
        || matches!(evidence_class, Some("internal"));
    if !has_evidence && allow_fallback && !matches!(evidence_class, Some("internal")) {
        if cached_recent_user_evidence.is_none() {
            *cached_recent_user_evidence = Some(
                kernel
                    .db
                    .get_recent_user_evidence_ids(conversation_id, 2)
                    .await,
            );
        }
        if let Some(ids) = cached_recent_user_evidence.as_ref() {
            if !ids.is_empty() {
                set_id_list(&mut payload, "evidence_event_ids", ids);
                has_evidence = true;
                if let Some(obj) = payload.as_object_mut() {
                    obj.insert(
                        "evidence_provenance".to_string(),
                        Value::String("recent_user_evidence".to_string()),
                    );
                }
                let _ = system_log::log_event(
                    &kernel.db.pool,
                    Some(&kernel.app_handle),
                    "info",
                    "memory",
                    run_id,
                    trace_id,
                    json!({
                        "event": "candidate_evidence_attached",
                        "candidate_id": candidate.id,
                        "candidate_kind": format!("{:?}", candidate.kind),
                        "category": category,
                        "reason": "recent_user_evidence",
                        "evidence_event_ids": ids,
                    }),
                )
                .await;
            }
        }
    }

    EvidenceAttachOutcome { payload, has_evidence }
}

async fn check_memory_evidence_quality(
    kernel: &Kernel,
    state: &mut KernelState,
    candidate: &Candidate,
    evidence_event_ids: &[i64],
    settings: &crate::models::Settings,
    run_id: Option<&str>,
    trace_id: Option<&str>,
    category: &str,
    conversation_id: &str,
) -> bool {
    if evidence_event_ids.is_empty() {
        return true;
    }
    let evidence_gate_enabled = settings.enable_memory_evidence_gating.unwrap_or(true);
    if !evidence_gate_enabled {
        return true;
    }
    let strictness = settings.weight_evidence_strictness.unwrap_or(0.5);
    let floor = quality_floor_for_memory(strictness);
    if let Some(stats) = kernel.db.evidence_quality_stats(evidence_event_ids).await {
        if stats.min < floor {
            let tier = evidence_quality_tier(stats.min);
            let _ = system_log::log_event(
                &kernel.db.pool,
                Some(&kernel.app_handle),
                "warn",
                "memory",
                run_id,
                trace_id,
                json!({
                    "event": "memory_write_blocked",
                    "reason": "evidence_quality_low",
                    "category": category,
                    "candidate_id": candidate.id,
                    "candidate_kind": format!("{:?}", candidate.kind),
                    "candidate_source": candidate.source,
                    "quality_min": stats.min,
                    "quality_avg": stats.avg,
                    "quality_tier": tier.as_str(),
                    "quality_floor": floor,
                    "evidence_count": stats.count,
                }),
            )
            .await;
            let payload = json!({
                "reason": "evidence_quality_low",
                "category": category,
                "quality_min": stats.min,
                "quality_avg": stats.avg,
                "quality_tier": tier.as_str(),
                "quality_floor": floor,
                "evidence_count": stats.count,
            });
            let snippet = "memory_write_blocked evidence_quality_low".to_string();
            let _ = kernel
                .emit_system_evidence(
                    state,
                    settings,
                    conversation_id,
                    "memory_write_blocked",
                    run_id,
                    &snippet,
                    Some(&payload),
                )
                .await;
            return false;
        }
    }
    true
}

async fn update_workspace_field_quality(db: &Db, field: &mut WorkspaceFieldMeta) {
    if field.evidence_event_ids.is_empty() {
        field.evidence_quality = None;
        return;
    }
    if let Some(stats) = db.evidence_quality_stats(&field.evidence_event_ids).await {
        field.evidence_quality = Some(stats.avg);
    } else {
        field.evidence_quality = None;
    }
}

async fn update_workspace_meta_quality(db: &Db, meta: &mut WorkspaceMeta) {
    if let Some(field) = meta.goal_thread.as_mut() {
        update_workspace_field_quality(db, field).await;
    }
    if let Some(field) = meta.current_focus.as_mut() {
        update_workspace_field_quality(db, field).await;
    }
    if let Some(field) = meta.focus_rationale.as_mut() {
        update_workspace_field_quality(db, field).await;
    }
    for item in meta.open_questions.iter_mut() {
        if let Some(stats) = db.evidence_quality_stats(&item.evidence_event_ids).await {
            item.evidence_quality = Some(stats.avg);
        } else {
            item.evidence_quality = None;
        }
    }
    for item in meta.working_set_topics.iter_mut() {
        if let Some(stats) = db.evidence_quality_stats(&item.evidence_event_ids).await {
            item.evidence_quality = Some(stats.avg);
        } else {
            item.evidence_quality = None;
        }
    }
    for hypothesis in meta.active_hypotheses.iter_mut() {
        if let Some(stats) = db.evidence_quality_stats(&hypothesis.evidence_event_ids).await {
            hypothesis.evidence_quality = Some(stats.avg);
        } else {
            hypothesis.evidence_quality = None;
        }
    }
}

impl Kernel {
    pub(crate) async fn commit_cycle(
        &self,
        state: &mut KernelState,
        decision: &KernelDecision,
        conversation_id: &str,
        run_id: Option<&str>,
        trace_id: Option<&str>,
        settings: &crate::models::Settings,
        is_internal_tick: bool,
        early_result_tx: Option<oneshot::Sender<CommitResult>>,
        background: bool,
    ) -> Result<CommitResult, String> {
        let commit_started = Instant::now();
        let commit_started_at = Utc::now();
        if run_id.is_some() && !is_internal_tick {
            self.emit_module_status(run_id, "commit_cycle", "Finalizing updates", commit_started_at, 0);
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                run_id,
                trace_id,
                json!({
                    "event": "post_processing_entered",
                    "started_at": commit_started_at.to_rfc3339(),
                }),
            )
            .await;
        }
        let mut emit_content: Option<String> = None;
        let mut emit_source: Option<String> = None;
        let mut tool_dispatches: Vec<ToolDispatchRequest> = Vec::new();
        let mut thread_run: Option<ThreadRunRequest> = None;
        let mut research_cost = 0;

        let mut outcomes_to_apply: Vec<Outcome> = Vec::new();
        let mut inner_summary_json: Option<String> = None;
        let mut episodic_writes: Vec<EpisodicWrite> = Vec::new();
        let mut semantic_promotions: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ToolDispatchRequest> = Vec::new();
        let mut spawn_goal: Option<String> = None;
        let mut explicit_mode: Option<KernelMode> = None;
        let mut ask_question: Option<String> = None;
        let mut ask_slots: Vec<String> = Vec::new();
        let mut terminate_requested = false;
        let mut self_claims_to_write: Vec<SelfClaimInput> = Vec::new();
        let mut cached_recent_user_evidence: Option<Vec<i64>> = None;
        let mut workspace_updated = false;
        let workspace_snapshot = WorkspaceSnapshot::from_state(state);
        let mut inner_summary_written = false;
        let mut episodic_written = 0usize;
        let mut semantic_written = false;
        let mut self_claims_written = 0usize;

        for candidate in &decision.accepted {
            if !self.plan_preconditions_met(state, candidate) {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    run_id,
                    trace_id,
                    json!({
                        "event": "plan_step_commit_blocked",
                        "candidate_id": candidate.id,
                        "candidate_kind": format!("{:?}", candidate.kind),
                        "plan_step_id": candidate.payload.get("plan_step_id").and_then(|v| v.as_str()),
                        "step_index": candidate.payload.get("step_index").and_then(|v| v.as_i64()),
                    }),
                )
                .await;
                continue;
            }
            if is_meta_cog_candidate(candidate) && !matches!(candidate.kind, CandidateKind::NoOp) {
                let anchor = state
                    .workspace_current_focus
                    .clone()
                    .or_else(|| state.workspace_goal_thread.clone())
                    .unwrap_or_else(|| "current topic".to_string());
                state.meta_cog_pending = Some(MetaCogPending {
                    kind: format!("{:?}", candidate.kind),
                    anchor,
                    accepted_at: Utc::now().to_rfc3339(),
                    turn_index: state.monologue_count,
                    source: candidate.source.clone(),
                });
            }
            if !is_internal_tick && is_monologue_intent_candidate(candidate) {
                let bridge_id = candidate.payload.get("bridge_id").and_then(|v| v.as_str());
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    run_id,
                    trace_id,
                    json!({
                        "event": "intent_delivered",
                        "candidate_id": candidate.id,
                        "candidate_kind": format!("{:?}", candidate.kind),
                        "source": candidate.source,
                        "bridge_id": bridge_id,
                    }),
                )
                .await;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    run_id,
                    trace_id,
                    json!({
                        "event": "meta_cog_intent_delivered",
                        "candidate_id": candidate.id,
                        "candidate_kind": format!("{:?}", candidate.kind),
                        "source": candidate.source,
                        "bridge_id": bridge_id,
                    }),
                )
                .await;
            }
            match candidate.kind {
                CandidateKind::UpdateGoalThread => {
                    let pending_actions_before = state.pending_actions.clone();
                    let pending_reframes_before = state.pending_reframes.clone();
                    let pending_angles_before = state.pending_angles.clone();
                    if let Some(outcomes) = candidate.payload.get("outcomes") {
                        if let Ok(parsed) = serde_json::from_value::<Vec<Outcome>>(outcomes.clone()) {
                            outcomes_to_apply.extend(parsed.clone());
                            for outcome in parsed.iter() {
                                if outcome.action_type.starts_with("tool_dispatch_") {
                                    episodic_writes.push(EpisodicWrite {
                                        event_type: "tool_outcome".to_string(),
                                        payload: json!({
                                            "status": if outcome.success { "success" } else { "failed" },
                                            "summary_snippet": summarize_snippet(&outcome.observations, 180),
                                            "tool_name": outcome.source,
                                        }),
                                        source_type: "tool".to_string(),
                                        source_ref: outcome.action_id.clone(),
                                    });
                                } else if outcome.action_type.starts_with("thread_") {
                                    episodic_writes.push(EpisodicWrite {
                                        event_type: "thread_outcome".to_string(),
                                        payload: json!({
                                            "status": if outcome.success { "completed" } else { "failed" },
                                            "summary_snippet": summarize_snippet(&outcome.observations, 180),
                                        }),
                                        source_type: "thread".to_string(),
                                        source_ref: outcome.action_id.clone(),
                                    });
                                }
                            }
                        }
                    } else if let Some(harvest_type) = candidate.payload.get("harvest_type").and_then(|v| v.as_str()) {
                        let payload = candidate
                            .payload
                            .get("payload")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        if !payload.is_empty() {
                            match harvest_type {
                                "new_angle" => push_capped(&mut state.pending_angles, payload, 5),
                                "reframe" => push_capped(&mut state.pending_reframes, payload, 5),
                                "candidate_next_action" => push_capped(&mut state.pending_actions, payload, 5),
                                _ => {}
                            }
                        }
                    }
                    if is_monologue_source(&candidate.source) {
                        let summary = candidate
                            .payload
                            .get("goal")
                            .and_then(|v| v.as_str())
                            .or_else(|| candidate.payload.get("payload").and_then(|v| v.as_str()))
                            .map(|s| summarize_snippet(s, 180));
                        let evidence_event_ids = extract_id_list(&candidate.payload, "evidence_event_ids");
                        let belief_ids = extract_id_list(&candidate.payload, "belief_ids");
                        let before_after_changed = pending_actions_before != state.pending_actions
                            || pending_reframes_before != state.pending_reframes
                            || pending_angles_before != state.pending_angles;
                        let (before, after) = if before_after_changed {
                            (
                                Some(json!({
                                    "pending_actions": pending_actions_before,
                                    "pending_reframes": pending_reframes_before,
                                    "pending_angles": pending_angles_before,
                                })),
                                Some(json!({
                                    "pending_actions": state.pending_actions,
                                    "pending_reframes": state.pending_reframes,
                                    "pending_angles": state.pending_angles,
                                })),
                            )
                        } else {
                            (None, None)
                        };
                        let _ = system_log::log_monologue_state_update(
                            &self.db.pool,
                            Some(&self.app_handle),
                            run_id,
                            trace_id,
                            conversation_id,
                            &candidate.id,
                            &format!("{:?}", candidate.kind),
                            candidate
                                .target_scope
                                .as_deref()
                                .or(Some("goal_thread")),
                            &evidence_event_ids,
                            &belief_ids,
                            summary.as_deref(),
                            before,
                            after,
                        )
                        .await;
                    }
                }
                CandidateKind::UpdateInnerSummary => {
                    let evidence = ensure_candidate_evidence(
                        self,
                        candidate,
                        conversation_id,
                        run_id,
                        trace_id,
                        &mut cached_recent_user_evidence,
                        true,
                        "inner_summary",
                    )
                    .await;
                    if !evidence.has_evidence {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "memory",
                            run_id,
                            trace_id,
                            json!({
                                "event": "memory_write_blocked",
                                "reason": "missing_evidence",
                                "category": "inner_summary",
                                "candidate_id": candidate.id,
                                "candidate_kind": format!("{:?}", candidate.kind),
                                "candidate_source": candidate.source,
                                "evidence_class": candidate_evidence_class(candidate),
                            }),
                        )
                        .await;
                        continue;
                    }
                    let evidence_event_ids = extract_id_list(&evidence.payload, "evidence_event_ids");
                    if !check_memory_evidence_quality(
                        self,
                        state,
                        candidate,
                        &evidence_event_ids,
                        settings,
                        run_id,
                        trace_id,
                        "inner_summary",
                        conversation_id,
                    )
                    .await
                    {
                        continue;
                    }
                    if let Some(summary_json) = evidence.payload.get("summary_json").and_then(|v| v.as_str()) {
                        inner_summary_json = Some(summary_json.to_string());
                    }
                }
                CandidateKind::WriteEpisodic => {
                    let evidence = ensure_candidate_evidence(
                        self,
                        candidate,
                        conversation_id,
                        run_id,
                        trace_id,
                        &mut cached_recent_user_evidence,
                        true,
                        "episodic",
                    )
                    .await;
                    if !evidence.has_evidence {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "memory",
                            run_id,
                            trace_id,
                            json!({
                                "event": "memory_write_blocked",
                                "reason": "missing_evidence",
                                "category": "episodic",
                                "candidate_id": candidate.id,
                                "candidate_kind": format!("{:?}", candidate.kind),
                                "candidate_source": candidate.source,
                                "evidence_class": candidate_evidence_class(candidate),
                            }),
                        )
                        .await;
                        continue;
                    }
                    let evidence_event_ids = extract_id_list(&evidence.payload, "evidence_event_ids");
                    if !check_memory_evidence_quality(
                        self,
                        state,
                        candidate,
                        &evidence_event_ids,
                        settings,
                        run_id,
                        trace_id,
                        "episodic",
                        conversation_id,
                    )
                    .await
                    {
                        continue;
                    }
                    if let Some(event_type) = evidence.payload.get("event_type").and_then(|v| v.as_str()) {
                        let payload = evidence.payload.get("payload").cloned().unwrap_or_else(|| json!({}));
                        let source_type = candidate
                            .payload
                            .get("source_type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("kernel")
                            .to_string();
                        let source_ref = candidate
                            .payload
                            .get("source_ref")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        episodic_writes.push(EpisodicWrite {
                            event_type: event_type.to_string(),
                            payload,
                            source_type,
                            source_ref,
                        });
                    }
                }
                CandidateKind::PromoteSemantic => {
                    let evidence = ensure_candidate_evidence(
                        self,
                        candidate,
                        conversation_id,
                        run_id,
                        trace_id,
                        &mut cached_recent_user_evidence,
                        false,
                        "semantic_core",
                    )
                    .await;
                    if !evidence.has_evidence {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "memory",
                            run_id,
                            trace_id,
                            json!({
                                "event": "memory_write_blocked",
                                "reason": "missing_evidence",
                                "category": "semantic_core",
                                "candidate_id": candidate.id,
                                "candidate_kind": format!("{:?}", candidate.kind),
                                "candidate_source": candidate.source,
                                "evidence_class": candidate_evidence_class(candidate),
                            }),
                        )
                        .await;
                        continue;
                    }
                    let evidence_event_ids = extract_id_list(&evidence.payload, "evidence_event_ids");
                    if !check_memory_evidence_quality(
                        self,
                        state,
                        candidate,
                        &evidence_event_ids,
                        settings,
                        run_id,
                        trace_id,
                        "semantic_core",
                        conversation_id,
                    )
                    .await
                    {
                        continue;
                    }
                    if let Some(summary) = evidence.payload.get("summary").and_then(|v| v.as_str()) {
                        let (cleaned, removed) = strip_internal_diagnostics_lines(summary);
                        if removed > 0 {
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "warn",
                                "memory",
                                run_id,
                                trace_id,
                                json!({
                                    "event": "semantic_promotion_sanitized",
                                    "removed_lines": removed,
                                }),
                            )
                            .await;
                        }
                        let cleaned = cleaned.trim();
                        if !cleaned.is_empty() {
                            semantic_promotions.push(cleaned.to_string());
                            if is_monologue_source(&candidate.source) {
                                let evidence_event_ids =
                                    extract_id_list(&evidence.payload, "evidence_event_ids");
                                let belief_ids = extract_id_list(&evidence.payload, "belief_ids");
                                let after = Some(json!({ "semantic": cleaned }));
                                let _ = system_log::log_monologue_state_update(
                                    &self.db.pool,
                                    Some(&self.app_handle),
                                    run_id,
                                    trace_id,
                                    conversation_id,
                                    &candidate.id,
                                    &format!("{:?}", candidate.kind),
                                    candidate
                                        .target_scope
                                        .as_deref()
                                        .or(Some("semantic")),
                                    &evidence_event_ids,
                                    &belief_ids,
                                    Some(cleaned),
                                    None,
                                    after,
                                )
                                .await;
                            }
                        } else if removed > 0 {
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "warn",
                                "memory",
                                run_id,
                                trace_id,
                                json!({
                                    "event": "semantic_promotion_dropped",
                                    "reason": "only_internal_diagnostics",
                                }),
                            )
                            .await;
                        }
                    }
                }
                CandidateKind::ToolCall => {
                    let action_id = candidate
                        .payload
                        .get("action_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let tool_name = candidate
                        .payload
                        .get("tool_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args = candidate
                        .payload
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}")
                        .to_string();
                    if !action_id.is_empty() && !tool_name.is_empty() {
                        let proposal_id = decision.report.proposal_id.as_deref().unwrap_or("").trim();
                        let plan_step_id = {
                            if let Some(step_index) = candidate.payload.get("step_index").and_then(|v| v.as_i64()) {
                                if !proposal_id.is_empty() {
                                    Some(format!("{}:{}", proposal_id, step_index))
                                } else {
                                    None
                                }
                            } else {
                                candidate
                                    .payload
                                    .get("plan_step_id")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                            }
                        }
                            .or_else(|| {
                                if crate::core::tool_registry::ToolRegistry::is_context_only_tool(
                                    &tool_name,
                                ) {
                                    return None;
                                }
                                if let Some(step_index) = candidate.payload.get("step_index").and_then(|v| v.as_i64()) {
                                    if !proposal_id.is_empty() {
                                        return Some(format!("{}:{}", proposal_id, step_index));
                                    }
                                }
                                if !proposal_id.is_empty() {
                                    return Some(format!("{}:{}", proposal_id, action_id));
                                }
                                let active_plan_id =
                                    state.workspace_active_plan_id.as_deref().unwrap_or("");
                                if !active_plan_id.trim().is_empty() {
                                    return Some(format!("{}:{}", active_plan_id, action_id));
                                }
                                let plan_hash = decision
                                    .report
                                    .plan_hash
                                    .as_deref()
                                    .unwrap_or("")
                                    .trim();
                                if plan_hash.is_empty() {
                                    None
                                } else {
                                    Some(format!("{}:{}", plan_hash, action_id))
                                }
                            });
                        let call = ToolDispatchRequest {
                            action_id,
                            tool_name,
                            args_json: args,
                            plan_step_id,
                        };
                        let fingerprint = tool_fingerprint(&call.tool_name, &call.args_json);
                        if !fingerprint.is_empty() {
                            state.tool_call_fingerprints.push(fingerprint);
                        }
                        tool_calls.push(call);
                    }
                }
                CandidateKind::EmitMessage | CandidateKind::FlagForHuman => {
                    if emit_content.is_none() {
                        emit_content = candidate
                            .payload
                            .get("content")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        if emit_content.is_some() {
                            emit_source = Some(candidate.source.clone());
                        }
                    }
                }
                CandidateKind::AskUserQuestion => {
                    if let Some(question) = candidate
                        .payload
                        .get("payload")
                        .and_then(|v| v.as_str())
                        .or_else(|| candidate.payload.get("question").and_then(|v| v.as_str()))
                        .or_else(|| candidate.payload.get("content").and_then(|v| v.as_str()))
                    {
                        ask_question = Some(question.to_string());
                        ask_slots = extract_requested_slots(&candidate.payload);
                    }
                }
                CandidateKind::UpdateWorkspace => {
                    let mut workspace_candidate = candidate.clone();
                    let evidence = ensure_candidate_evidence(
                        self,
                        candidate,
                        conversation_id,
                        run_id,
                        trace_id,
                        &mut cached_recent_user_evidence,
                        true,
                        "workspace",
                    )
                    .await;
                    if !evidence.has_evidence {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "memory",
                            run_id,
                            trace_id,
                            json!({
                                "event": "memory_write_blocked",
                                "reason": "missing_evidence",
                                "category": "workspace",
                                "candidate_id": workspace_candidate.id,
                                "candidate_kind": format!("{:?}", workspace_candidate.kind),
                                "candidate_source": workspace_candidate.source,
                                "evidence_class": candidate_evidence_class(&workspace_candidate),
                            }),
                        )
                        .await;
                        continue;
                    }
                    let evidence_event_ids = extract_id_list(&evidence.payload, "evidence_event_ids");
                    if !check_memory_evidence_quality(
                        self,
                        state,
                        candidate,
                        &evidence_event_ids,
                        settings,
                        run_id,
                        trace_id,
                        "workspace",
                        conversation_id,
                    )
                    .await
                    {
                        continue;
                    }
                    workspace_candidate.payload = evidence.payload;
                    let disable_working_hypothesis =
                        settings.stability_disable_working_hypothesis.unwrap_or(true);
                    let workspace_before = WorkspaceSnapshot::from_state(state);
                    if self
                        .apply_workspace_update_with_policy(state, &workspace_candidate, disable_working_hypothesis)
                        .await
                    {
                        workspace_updated = true;
                        if is_monologue_source(&workspace_candidate.source) {
                            let fields = workspace_candidate
                                .payload
                                .as_object()
                                .map(|obj| {
                                    obj.keys()
                                        .cloned()
                                        .collect::<Vec<_>>()
                                        .join(",")
                                })
                                .unwrap_or_default();
                            let summary = if fields.is_empty() {
                                None
                            } else {
                                Some(format!("fields: {}", fields))
                            };
                            let evidence_event_ids =
                                extract_id_list(&workspace_candidate.payload, "evidence_event_ids");
                            let belief_ids = extract_id_list(&workspace_candidate.payload, "belief_ids");
                            let (before, after) = workspace_before_after_for_payload(
                                &workspace_before,
                                state,
                                &workspace_candidate.payload,
                            );
                            let _ = system_log::log_monologue_state_update(
                                &self.db.pool,
                                Some(&self.app_handle),
                                run_id,
                                trace_id,
                                conversation_id,
                                &workspace_candidate.id,
                                &format!("{:?}", workspace_candidate.kind),
                                workspace_candidate
                                    .target_scope
                                    .as_deref()
                                    .or(Some("workspace")),
                                &evidence_event_ids,
                                &belief_ids,
                                summary.as_deref(),
                                before,
                                after,
                            )
                            .await;
                        }
                    }
                }
                CandidateKind::AnchorShift => {
                    let old_anchor = candidate
                        .payload
                        .get("old_anchor")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let new_anchor = candidate
                        .payload
                        .get("new_anchor")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let observations = if old_anchor.is_empty() && new_anchor.is_empty() {
                        "anchor_shift".to_string()
                    } else {
                        format!("{} -> {}", old_anchor, new_anchor).trim().to_string()
                    };
                    outcomes_to_apply.push(Outcome {
                        action_type: "anchor_shift".to_string(),
                        success: true,
                        observations: observations.clone(),
                        source: "kernel".to_string(),
                        failure_kind: None,
                        target_key: None,
                        tags: vec!["candidate_kind::AnchorShift".to_string()],
                        action_id: Some(candidate.id.clone()),
                        timestamp: Utc::now().to_rfc3339(),
                    });
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        run_id,
                        trace_id,
                        json!({
                            "event": "anchor_shift_committed",
                            "candidate_id": candidate.id,
                            "old_anchor": old_anchor,
                            "new_anchor": new_anchor,
                        }),
                    )
                    .await;
                }
                CandidateKind::Terminate => {
                    terminate_requested = true;
                }
                CandidateKind::RecordSelfClaim => {
                    let claim_text = candidate
                        .payload
                        .get("claim_text")
                        .and_then(|v| v.as_str())
                        .or_else(|| candidate.payload.get("claim").and_then(|v| v.as_str()))
                        .or_else(|| candidate.payload.get("text").and_then(|v| v.as_str()))
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if claim_text.is_empty() {
                        continue;
                    }
                    let claim_key = candidate
                        .payload
                        .get("claim_key")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let confidence = candidate
                        .payload
                        .get("confidence")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.6) as f32;
                    let polarity = candidate
                        .payload
                        .get("polarity")
                        .and_then(|v| v.as_str())
                        .unwrap_or("assert")
                        .to_string();
                    let mut evidence_event_ids = extract_id_list(&candidate.payload, "evidence_event_ids");
                    if let Some(single_id) = candidate
                        .payload
                        .get("evidence_event_id")
                        .and_then(|v| v.as_i64())
                    {
                        if !evidence_event_ids.contains(&single_id) {
                            evidence_event_ids.push(single_id);
                        }
                    }
                    let belief_ids = extract_id_list(&candidate.payload, "belief_ids");
                    let mut provisional = candidate
                        .payload
                        .get("provisional")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let source_type = candidate
                        .payload
                        .get("source_type")
                        .and_then(|v| v.as_str())
                        .map(|v| v.to_string());
                    let self_awareness_mode = settings
                        .self_awareness_expression_mode
                        .as_deref()
                        .unwrap_or("conservative");
                    let self_awareness_allowed = settings.self_report_channel.unwrap_or(true)
                        && !self_awareness_mode.eq_ignore_ascii_case("conservative");
                    let allow_provisional_self_awareness = self_awareness_allowed
                        && source_type
                            .as_deref()
                            .map(|v| v.eq_ignore_ascii_case("self_awareness_query"))
                            .unwrap_or(false)
                        && self_claims::is_self_awareness_claim(&claim_text);
                    let mut requires_validation = candidate
                        .payload
                        .get("requires_validation")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let mut ttl_seconds = candidate
                        .payload
                        .get("ttl_seconds")
                        .and_then(|v| v.as_i64());
                    let mut promotion_rule = candidate
                        .payload
                        .get("promotion_rule")
                        .and_then(|v| v.as_str())
                        .map(|v| v.to_string());
                    let mut eviction_rule = candidate
                        .payload
                        .get("eviction_rule")
                        .and_then(|v| v.as_str())
                        .map(|v| v.to_string());
                    if allow_provisional_self_awareness && promotion_rule.is_none() {
                        promotion_rule = Some("no_promotion".to_string());
                    }
                    if allow_provisional_self_awareness && eviction_rule.is_none() {
                        eviction_rule = Some("ttl_expire".to_string());
                    }
                    if evidence_event_ids.is_empty()
                        && belief_ids.is_empty()
                        && !matches!(candidate_evidence_class(candidate), Some("internal"))
                    {
                        if cached_recent_user_evidence.is_none() {
                            cached_recent_user_evidence = Some(
                                self.db
                                    .get_recent_user_evidence_ids(conversation_id, 2)
                                    .await,
                            );
                        }
                        if let Some(ids) = cached_recent_user_evidence.as_ref() {
                            if !ids.is_empty() {
                                evidence_event_ids = ids.clone();
                                let _ = system_log::log_event(
                                    &self.db.pool,
                                    Some(&self.app_handle),
                                    "info",
                                    "memory",
                                    run_id,
                                    trace_id,
                                    json!({
                                        "event": "self_claim_evidence_attached",
                                        "claim_key": claim_key,
                                        "evidence_event_ids": evidence_event_ids,
                                        "reason": "recent_user_evidence",
                                    }),
                                )
                                .await;
                            }
                        }
                        if evidence_event_ids.is_empty() {
                            let identity_ids = self
                                .db
                                .get_recent_evidence_ids_by_source_types(
                                    &["identity_statement", "capability_statement"],
                                    4,
                                )
                                .await;
                            if !identity_ids.is_empty() {
                                evidence_event_ids = identity_ids;
                                let _ = system_log::log_event(
                                    &self.db.pool,
                                    Some(&self.app_handle),
                                    "info",
                                    "memory",
                                    run_id,
                                    trace_id,
                                    json!({
                                        "event": "self_claim_evidence_attached",
                                        "claim_key": claim_key,
                                        "evidence_event_ids": evidence_event_ids,
                                        "reason": "identity_capability_evidence",
                                    }),
                                )
                                .await;
                            }
                        }
                        if evidence_event_ids.is_empty() {
                            provisional = true;
                            requires_validation = true;
                            if ttl_seconds.is_none() && !allow_provisional_self_awareness {
                                ttl_seconds = Some(self_claims::stale_evidence_ttl_seconds());
                            }
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "warn",
                                "memory",
                                run_id,
                                trace_id,
                                json!({
                                    "event": "self_claim_missing_evidence",
                                    "claim_key": claim_key,
                                    "candidate_id": candidate.id,
                                }),
                            )
                            .await;
                        }
                    }

                    if evidence_event_ids.is_empty() && belief_ids.is_empty() {
                        if allow_provisional_self_awareness {
                            provisional = true;
                            requires_validation = true;
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "memory",
                                run_id,
                                trace_id,
                                json!({
                                    "event": "self_claim_provisionalized",
                                    "reason": "self_awareness_query",
                                    "candidate_id": candidate.id,
                                }),
                            )
                            .await;
                        } else {
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "warn",
                                "memory",
                                run_id,
                                trace_id,
                                json!({
                                    "event": "memory_write_blocked",
                                    "reason": "missing_evidence",
                                    "category": "self_claim",
                                    "candidate_id": candidate.id,
                                    "candidate_kind": format!("{:?}", candidate.kind),
                                    "candidate_source": candidate.source,
                                    "evidence_class": candidate_evidence_class(candidate),
                                }),
                            )
                            .await;
                            continue;
                        }
                    }

                    if let Some(latest) = self_claims::evidence_is_stale(&self.db, &evidence_event_ids).await {
                        provisional = true;
                        requires_validation = true;
                        if ttl_seconds.is_none() {
                            ttl_seconds = Some(self_claims::stale_evidence_ttl_seconds());
                        }
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "memory",
                            run_id,
                            trace_id,
                            json!({
                                "event": "self_claim_stale_evidence",
                                "claim_key": claim_key,
                                "candidate_id": candidate.id,
                                "evidence_event_ids": evidence_event_ids,
                                "latest_evidence_at": latest.to_rfc3339(),
                            }),
                        )
                        .await;
                    }

                    if !evidence_event_ids.is_empty() {
                        let strictness = settings.weight_evidence_strictness.unwrap_or(0.5);
                        let floor = quality_floor_for_self_claim(strictness);
                        let evidence_gate_enabled = settings.enable_memory_evidence_gating.unwrap_or(true);
                        if evidence_gate_enabled {
                            if let Some(stats) = self.db.evidence_quality_stats(&evidence_event_ids).await {
                                if stats.min < floor {
                                    let tier = evidence_quality_tier(stats.min);
                                    let _ = system_log::log_event(
                                        &self.db.pool,
                                        Some(&self.app_handle),
                                        "warn",
                                        "memory",
                                        run_id,
                                        trace_id,
                                        json!({
                                            "event": "memory_write_blocked",
                                            "reason": "evidence_quality_low",
                                            "category": "self_claim",
                                            "candidate_id": candidate.id,
                                            "candidate_kind": format!("{:?}", candidate.kind),
                                            "candidate_source": candidate.source,
                                            "quality_min": stats.min,
                                            "quality_avg": stats.avg,
                                            "quality_tier": tier.as_str(),
                                            "quality_floor": floor,
                                            "evidence_count": stats.count,
                                        }),
                                    )
                                    .await;
                                    continue;
                                }
                            }
                        }
                    }

                    self_claims_to_write.push(SelfClaimInput {
                        claim_text,
                        claim_key,
                        evidence_event_ids,
                        belief_ids,
                        confidence,
                        polarity,
                        source_run_id: run_id.map(|id| id.to_string()),
                        conversation_id: Some(conversation_id.to_string()),
                        provisional,
                        source_type,
                        requires_validation,
                        ttl_seconds,
                        promotion_rule,
                        eviction_rule,
                    });
                }
                CandidateKind::SpawnThread => {
                    if let Some(goal) = candidate
                        .payload
                        .get("goal")
                        .and_then(|v| v.as_str())
                        .or_else(|| candidate.payload.get("payload").and_then(|v| v.as_str()))
                    {
                        spawn_goal = Some(goal.to_string());
                    }
                }
                CandidateKind::ChangeMode => {
                    if let Some(mode) = candidate.payload.get("mode").and_then(|v| v.as_str()) {
                        explicit_mode = if mode.eq_ignore_ascii_case("play") {
                            Some(KernelMode::Play)
                        } else {
                            Some(KernelMode::Work)
                        };
                    }
                }
                _ => {}
            }
        }

        if !outcomes_to_apply.is_empty() {
            self.apply_outcomes(state, &outcomes_to_apply).await;
            state.last_outcome_at = outcomes_to_apply
                .last()
                .map(|o| o.timestamp.clone())
                .or_else(|| Some(Utc::now().to_rfc3339()));
        }

        if state.failure_count >= SAFE_HALT_FAILURE_THRESHOLD {
            if !state.halted {
                state.halted = true;
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    run_id,
                    trace_id,
                    json!({
                        "event": "safe_halt",
                        "failure_count": state.failure_count,
                    }),
                )
                .await;
            }
        }

        if let Some(mode) = explicit_mode {
            state.mode = mode;
            state.last_mode_switch_at = Some(Utc::now().to_rfc3339());
        }

        if let Some(question) = ask_question.clone() {
            if is_internal_tick {
                if !state.workspace_open_questions.iter().any(|q| q == &question) {
                    state.workspace_open_questions.push(question.clone());
                    state
                        .workspace_meta
                        .open_questions
                        .push(make_list_meta(&question, true, &[], &[]));
                    if state.workspace_open_questions.len() > 8 {
                        let excess = state.workspace_open_questions.len() - 8;
                        state.workspace_open_questions.drain(0..excess);
                        if state.workspace_meta.open_questions.len() > excess {
                            state.workspace_meta.open_questions.drain(0..excess);
                        } else {
                            state.workspace_meta.open_questions.clear();
                        }
                    }
                    workspace_updated = true;
                }
            } else {
                if let Some(content) = emit_content.as_mut() {
                    if content.trim().is_empty() {
                        *content = question.clone();
                    } else {
                        content.push_str("\n\n");
                        content.push_str(&question);
                    }
                } else {
                    emit_content = Some(question.clone());
                }
                state.pending_questions = vec![question.clone()];
                state.uncertainty_count += 1;
                state.task_phase = TaskPhase::AwaitingUser;
                if state.ask_budget_remaining > 0 {
                    state.ask_budget_remaining -= 1;
                }
            }
            if !ask_slots.is_empty() {
                state.last_asked_slots = ask_slots.clone();
                state.asked_slot_sets.push(normalize_slot_set(&ask_slots));
                let fingerprint = question_fingerprint(&question, &ask_slots);
                if !fingerprint.is_empty() {
                    state.question_fingerprints.push(fingerprint);
                }
            } else {
                state.last_asked_slots.clear();
                let fingerprint = question_fingerprint(&question, &[]);
                if !fingerprint.is_empty() {
                    state.question_fingerprints.push(fingerprint);
                }
            }
            state.recent_questions.push(question);
            let max_k = settings.loop_recent_k.unwrap_or(6).max(1) as usize;
            if state.recent_questions.len() > max_k {
                let excess = state.recent_questions.len() - max_k;
                state.recent_questions.drain(0..excess);
            }
            if state.question_fingerprints.len() > max_k * 2 {
                let excess = state.question_fingerprints.len() - max_k * 2;
                state.question_fingerprints.drain(0..excess);
            }
        }

        if is_internal_tick {
            if let Some(content) = emit_content.as_deref() {
                if !content.trim().is_empty() {
                    let fingerprint = emit_fingerprint(content);
                    if !fingerprint.is_empty() {
                        state.recent_emit_fingerprints.push(fingerprint);
                    }
                    state.recent_emit_messages.push(content.to_string());
                    let max_k = settings.loop_recent_k.unwrap_or(6).max(1) as usize;
                    if state.recent_emit_messages.len() > max_k {
                        let excess = state.recent_emit_messages.len() - max_k;
                        state.recent_emit_messages.drain(0..excess);
                    }
                    if state.recent_emit_fingerprints.len() > max_k * 2 {
                        let excess = state.recent_emit_fingerprints.len() - max_k * 2;
                        state.recent_emit_fingerprints.drain(0..excess);
                    }
                }
            }
        }

        if terminate_requested {
            let mut scope = StopScope::default();
            scope.tools = true;
            scope.memory_write = true;
            scope.self_claims = true;
            scope.monologue_run = true;
            scope.monologue_emit = true;
            scope.background_jobs = true;
            apply_stop_state(
                state,
                StopReason {
                    category: StopReasonCategory::LatchBlock,
                    subcode: "terminate_candidate".to_string(),
                    contract: None,
                },
                scope,
            );
            state.task_phase = TaskPhase::Terminated;
        }

        if settings.goal_loop_enabled.unwrap_or(true) {
            state.goal_loop_turn_count = state.goal_loop_turn_count.saturating_add(1);
            let interval = settings.goal_loop_interval_turns.unwrap_or(3).max(1) as i64;
            let has_active_goal = goal_stack_active_label(&state.workspace_goal_stack).is_some();
            let due = has_active_goal || (state.goal_loop_turn_count % interval == 0);
            let decision_reason: &str;
            let mut decision_summary: Option<Value> = None;
            let mut decision_load_avg: Option<i64> = None;
            if due {
                let load_threshold = settings.goal_loop_load_threshold_ms.unwrap_or(650).max(1) as i64;
                let mut load_avg: i64 = 0;
                if let Ok(Some(raw)) = self.db.get_key("latency_prompt_build_avg_ms").await {
                    if let Ok(value) = raw.parse::<f64>() {
                        load_avg = load_avg.max(value.round() as i64);
                    }
                }
                if let Ok(Some(raw)) = self.db.get_key("latency_emit_avg_ms").await {
                    if let Ok(value) = raw.parse::<f64>() {
                        load_avg = load_avg.max(value.round() as i64);
                    }
                }
                decision_load_avg = Some(load_avg);
                let now = Utc::now().to_rfc3339();
                if let Some(summary) = apply_goal_loop_tick(&mut state.workspace_goal_stack, &now) {
                    workspace_updated = true;
                    decision_reason = "advance";
                    decision_summary = Some(summary.clone());
                    if let Ok(summary_json) = serde_json::to_string(&summary) {
                        if let Some(event_id) = self
                            .db
                            .create_system_evidence_event(
                                &state.conversation_id,
                                "goal_loop",
                                "advance",
                                None,
                                &summary_json,
                            )
                            .await
                        {
                            merge_goal_stack_evidence(
                                &mut state.workspace_goal_stack,
                                &[event_id],
                                &[],
                            );
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "kernel",
                                run_id,
                                trace_id,
                                json!({
                                    "event": "goal_loop_evidence_created",
                                    "evidence_event_id": event_id,
                                    "summary": summary,
                                }),
                            )
                            .await;
                        }
                    }
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        run_id,
                        trace_id,
                        json!({
                            "event": "goal_loop_advance",
                        "summary": summary,
                        "turn": state.goal_loop_turn_count,
                    }),
                )
                .await;
                } else {
                    decision_reason = "missing_evidence";
                    let _ = system_log::log_event(
                        &self.db.pool,
                        Some(&self.app_handle),
                        "info",
                        "kernel",
                        run_id,
                        trace_id,
                        json!({
                            "event": "goal_loop_skip",
                            "reason": "missing_evidence",
                            "turn": state.goal_loop_turn_count,
                            "avg_ms": load_avg,
                            "threshold_ms": load_threshold,
                        }),
                    )
                    .await;
                }
            } else {
                decision_reason = if has_active_goal { "interval" } else { "no_active_goal" };
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "info",
                    "kernel",
                    run_id,
                    trace_id,
                    json!({
                        "event": "goal_loop_skip",
                        "reason": if has_active_goal { "interval" } else { "no_active_goal" },
                        "turn": state.goal_loop_turn_count,
                        "interval": interval,
                    }),
                )
                .await;
            }
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                run_id,
                trace_id,
                json!({
                    "event": "goal_loop_decision",
                    "reason": decision_reason,
                    "turn": state.goal_loop_turn_count,
                    "interval": interval,
                    "has_active_goal": has_active_goal,
                    "load_avg_ms": decision_load_avg,
                    "summary": decision_summary,
                }),
            )
            .await;
        }

        let mut thread_context_json: Option<String> = None;
        let mut thread_depth = state.thread_depth;
        if let Some(goal) = spawn_goal.as_deref() {
            thread_depth = state.thread_depth + 1;
            let context = self.build_thread_context_snapshot(conversation_id, goal, settings).await;
            thread_context_json = Some(context);
        }
        let mut thread_insert: Option<(String, String, String, i64)> = None;
        if let (Some(goal), Some(context_json)) = (spawn_goal.as_deref(), thread_context_json.as_deref()) {
            let thread_id = Uuid::new_v4().to_string();
            thread_insert = Some((
                thread_id.clone(),
                goal.to_string(),
                context_json.to_string(),
                thread_depth,
            ));
            state.thread_depth = thread_depth;
            state.active_threads.push(ThreadHandle {
                thread_id: thread_id.clone(),
                goal: goal.to_string(),
                status: "running".to_string(),
                spawned_at: Utc::now().to_rfc3339(),
                depth: thread_depth,
            });
            thread_run = Some(ThreadRunRequest {
                thread_id,
                goal: goal.to_string(),
                depth: thread_depth,
            });
        }

        let delta_fields = workspace_delta_fields(&workspace_snapshot, state);
        state.last_workspace_delta_fields = delta_fields.clone();
        state.last_workspace_delta_count = delta_fields.len() as i32;
        if !delta_fields.is_empty() {
            state.last_workspace_update_at = Some(Utc::now().to_rfc3339());
            workspace_updated = true;
        }

        if !tool_calls.is_empty() {
            for call in tool_calls.iter() {
                if is_research_tool_candidate(&call.tool_name) {
                    let cost = settings.research_cost_per_call.unwrap_or(1).max(1);
                    research_cost += cost;
                    if state.research_window_start.is_none() {
                        state.research_window_start = Some(Utc::now().to_rfc3339());
                    }
                    state.research_used += cost;
                }
                tool_dispatches.push(call.clone());
            }
        }

        let commit_result = CommitResult {
            emit_content: emit_content.clone(),
            emit_source: emit_source.clone(),
            tool_dispatches: tool_dispatches.clone(),
            thread_run: thread_run.clone(),
            research_cost,
            ask_question: if is_internal_tick { ask_question.clone() } else { None },
            ask_slots: if is_internal_tick { ask_slots.clone() } else { Vec::new() },
        };

        let gate_allows_writes = decision
            .report
            .gate_decision
            .as_deref()
            .map(gate_allows_writes_decision)
            .unwrap_or(false);
        let mut allow_writes = gate_allows_writes;
        if !gate_allows_writes {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "memory",
                run_id,
                trace_id,
                json!( {
                    "event": "memory_write_blocked",
                    "reason": "gate_decision",
                    "gate_decision": decision.report.gate_decision,
                    "gate_decision_id": decision.report.gate_decision_id,
                    "snapshot_hash": decision.report.snapshot_hash,
                    "conversation_id": conversation_id,
                }),
            )
            .await;
            let payload = json!({
                "reason": "gate_decision",
                "gate_decision": decision.report.gate_decision,
                "gate_decision_id": decision.report.gate_decision_id,
                "snapshot_hash": decision.report.snapshot_hash,
            });
            let snippet = format!(
                "memory_write_blocked gate_decision={}",
                decision.report.gate_decision.clone().unwrap_or_else(|| "unknown".to_string())
            );
            let _ = self
                .emit_system_evidence(
                    state,
                    settings,
                    conversation_id,
                    "memory_write_blocked",
                    run_id,
                    &snippet,
                    Some(&payload),
                )
                .await;
            inner_summary_json = None;
            episodic_writes.clear();
            semantic_promotions.clear();
            self_claims_to_write.clear();
        }
        let stop_allows_writes = decision.report.allowed_capabilities.memory_write;
        if allow_writes && !stop_allows_writes {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "memory",
                run_id,
                trace_id,
                json!( {
                    "event": "memory_write_blocked",
                    "reason": "stop_state",
                    "stop_reasons": decision.report.stop_reasons,
                    "conversation_id": conversation_id,
                }),
            )
            .await;
            let payload = json!({
                "reason": "stop_state",
                "stop_reasons": decision.report.stop_reasons,
            });
            let snippet = "memory_write_blocked stop_state".to_string();
            let _ = self
                .emit_system_evidence(
                    state,
                    settings,
                    conversation_id,
                    "memory_write_blocked",
                    run_id,
                    &snippet,
                    Some(&payload),
                )
                .await;
            inner_summary_json = None;
            episodic_writes.clear();
            semantic_promotions.clear();
            self_claims_to_write.clear();
            allow_writes = false;
        }

        if allow_writes && !is_internal_tick {
            let last_input = state.last_user_input.as_deref().map(str::trim).unwrap_or("");
            if !last_input.is_empty() {
                let snippet = identity::identity_statement_snippets(last_input, 1)
                    .into_iter()
                    .next();
                if let Some(snippet) = snippet {
                    if let Some(message_id) = state.last_user_message_id.as_deref() {
                        if let Some(evidence_id) = self
                            .db
                            .create_user_identity_evidence_event(message_id, &snippet)
                            .await
                        {
                            let _ = system_log::log_event(
                                &self.db.pool,
                                Some(&self.app_handle),
                                "info",
                                "memory",
                                run_id,
                                trace_id,
                                json!({
                                    "event": "identity_evidence_attached",
                                    "message_id": message_id,
                                    "evidence_event_id": evidence_id,
                                    "snippet": summarize_snippet(&snippet, 180),
                                    "conversation_id": conversation_id,
                                }),
                            )
                            .await;
                        }
                    } else {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "warn",
                            "memory",
                            run_id,
                            trace_id,
                            json!({
                                "event": "identity_evidence_missing",
                                "reason": "missing_message_id",
                                "conversation_id": conversation_id,
                            }),
                        )
                        .await;
                    }
                }
            }
        }

        if let Some(tx) = early_result_tx {
            let _ = tx.send(commit_result.clone());
        }

        if !is_internal_tick && workspace_updated {
            if let Some(focus) = state.workspace_current_focus.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
                let meta = state
                    .workspace_meta
                    .current_focus
                    .get_or_insert_with(WorkspaceFieldMeta::default);
                if meta.evidence_event_ids.is_empty() {
                    if let Some(evidence_id) = self
                        .db
                        .create_system_evidence_event(conversation_id, "workspace_focus", focus, run_id, focus)
                        .await
                    {
                        meta.evidence_event_ids.push(evidence_id);
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "memory",
                            run_id,
                            trace_id,
                            json!({
                                "event": "workspace_evidence_event_created",
                                "field": "current_focus",
                                "evidence_id": evidence_id,
                                "conversation_id": conversation_id,
                            }),
                        )
                        .await;
                    }
                }
            }
            if let Some(rationale) = state.workspace_focus_rationale.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
                let meta = state
                    .workspace_meta
                    .focus_rationale
                    .get_or_insert_with(WorkspaceFieldMeta::default);
                if meta.evidence_event_ids.is_empty() {
                    if let Some(evidence_id) = self
                        .db
                        .create_system_evidence_event(
                            conversation_id,
                            "workspace_focus_rationale",
                            rationale,
                            run_id,
                            rationale,
                        )
                        .await
                    {
                        meta.evidence_event_ids.push(evidence_id);
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "memory",
                            run_id,
                            trace_id,
                            json!({
                                "event": "workspace_evidence_event_created",
                                "field": "focus_rationale",
                                "evidence_id": evidence_id,
                                "conversation_id": conversation_id,
                            }),
                        )
                        .await;
                    }
                }
            }
        }

        let mut tx = self.db.pool.begin().await.map_err(|e| e.to_string())?;
        let mut policy_violations: Vec<Value> = Vec::new();
        let summary_reason = if is_internal_tick { "internal_tick" } else { "user_visible_turn" };

        if allow_writes {
            if let Some(summary_json) = inner_summary_json.as_deref() {
                let allowed = MemoryPolicy::is_allowed(
                    MemoryWriteCategory::InnerSummary,
                    MemoryWriteSource::Kernel,
                    summary_reason,
                );
                if !allowed {
                    policy_violations.push(json!({
                        "category": "inner_summary",
                        "reason_code": summary_reason,
                    }));
                } else {
                    let mut summary_payload = summary_json.to_string();
                    if !phi_consent_allowed(&self.db.pool, Some(conversation_id)).await {
                        if let Ok(value) = serde_json::from_str::<Value>(summary_json) {
                            let (redacted, sensitivity) = redact_sensitive_json(&value);
                            if let Some(level) = sensitivity {
                                summary_payload = redacted.to_string();
                                let _ = system_log::log_event(
                                    &self.db.pool,
                                    Some(&self.app_handle),
                                    "warn",
                                    "memory",
                                    run_id,
                                    trace_id,
                                    json!({
                                        "event": "phi_redacted",
                                        "scope": "inner_summary",
                                        "sensitivity": level.as_str(),
                                        "conversation_id": conversation_id,
                                    }),
                                )
                                .await;
                            }
                        } else {
                            let (redacted, sensitivity) = redact_sensitive_text(summary_json);
                            if let Some(level) = sensitivity {
                                summary_payload = redacted;
                                let _ = system_log::log_event(
                                    &self.db.pool,
                                    Some(&self.app_handle),
                                    "warn",
                                    "memory",
                                    run_id,
                                    trace_id,
                                    json!({
                                        "event": "phi_redacted",
                                        "scope": "inner_summary",
                                        "sensitivity": level.as_str(),
                                        "conversation_id": conversation_id,
                                    }),
                                )
                                .await;
                            }
                        }
                    }
                    sqlx::query(
                        "INSERT INTO inner_summaries (conversation_id, summary_json, updated_at, version)                 VALUES (?, ?, CURRENT_TIMESTAMP, 1)                 ON CONFLICT(conversation_id) DO UPDATE SET summary_json = excluded.summary_json,                     updated_at = CURRENT_TIMESTAMP,                     version = version + 1"
                    )
                    .bind(conversation_id)
                    .bind(&summary_payload)
                    .execute(&mut *tx)
                    .await
                    .map_err(|e| e.to_string())?;
                    let hash = hash_payload(&summary_payload);
                    let _ = self
                        .db
                        .log_memory_write_tx(
                            &mut tx,
                            Some(conversation_id),
                            "inner_summary",
                            "kernel",
                            summary_reason,
                            run_id,
                            trace_id,
                            Some(&hash),
                            decision.report.snapshot_hash.as_deref(),
                            decision.report.gate_decision_id.as_deref(),
                        )
                        .await;
                    inner_summary_written = true;
                }
            }
        }

        if allow_writes && settings.episodic_enabled.unwrap_or(true) {
            for write in episodic_writes {
                let reason_code = if write.event_type.starts_with("thread_") {
                    "thread_outcome"
                } else if write.event_type.starts_with("tool_") {
                    "tool_outcome"
                } else {
                    "meaningful_run"
                };
                let allowed = MemoryPolicy::is_allowed(
                    MemoryWriteCategory::Episodic,
                    MemoryWriteSource::Kernel,
                    reason_code,
                );
                if !allowed {
                    policy_violations.push(json!({
                        "category": "episodic",
                        "reason_code": reason_code,
                    }));
                    continue;
                }
                self.insert_episodic_event_tx(
                    &mut tx,
                    &write,
                    conversation_id,
                    run_id,
                    trace_id,
                )
                .await?;
                let payload_json = serde_json::to_string(&write.payload).unwrap_or_else(|_| "{}".to_string());
                let payload_hash = hash_payload(&payload_json);
                let _ = self
                    .db
                    .log_memory_write_tx(
                        &mut tx,
                        Some(conversation_id),
                        "episodic",
                        "kernel",
                        reason_code,
                        run_id,
                        trace_id,
                        Some(&payload_hash),
                        decision.report.snapshot_hash.as_deref(),
                        decision.report.gate_decision_id.as_deref(),
                    )
                    .await;
                episodic_written = episodic_written.saturating_add(1);
            }
        }

        if allow_writes {
            if let Some(summary) = semantic_promotions.last() {
            let (cleaned, removed) = strip_internal_diagnostics_lines(summary);
            if removed > 0 {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "memory",
                    run_id,
                    trace_id,
                    json!({
                        "event": "semantic_core_sanitized",
                        "removed_lines": removed,
                    }),
                )
                .await;
            }
            if cleaned.trim().is_empty() {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "memory",
                    run_id,
                    trace_id,
                    json!({
                        "event": "semantic_core_update_skipped",
                        "reason": "only_internal_diagnostics",
                    }),
                )
                .await;
            } else {
                let allowed = MemoryPolicy::is_allowed(
                    MemoryWriteCategory::SemanticCore,
                    MemoryWriteSource::Kernel,
                    "kernel_slow_promotion",
                );
                if !allowed {
                    policy_violations.push(json!({
                        "category": "semantic_core",
                        "reason_code": "kernel_slow_promotion",
                    }));
                } else {
                    self.set_semantic_core_tx(&mut tx, cleaned.trim()).await?;
                    let hash = hash_payload(cleaned.trim());
                    let _ = self
                        .db
                        .log_memory_write_tx(
                            &mut tx,
                            Some(conversation_id),
                            "semantic_core",
                            "kernel",
                            "kernel_slow_promotion",
                            run_id,
                            trace_id,
                            Some(&hash),
                            decision.report.snapshot_hash.as_deref(),
                            decision.report.gate_decision_id.as_deref(),
                        )
                        .await;
                    semantic_written = true;
                    state.last_semantic_promotion_at = Some(Utc::now().to_rfc3339());
                }
            }
            }
        }

        if !tool_calls.is_empty() {
            for call in tool_calls.iter() {
                self.record_tool_dispatch_tx(&mut tx, call, run_id).await?;
            }
        }

        if let Some((thread_id, goal, context_json, depth)) = thread_insert.as_ref() {
            self.insert_thread_run_tx(
                &mut tx,
                thread_id,
                conversation_id,
                run_id,
                goal,
                context_json,
                *depth,
            )
            .await?;
        }

        if !policy_violations.is_empty() {
            state.halted = true;
        }

        self.update_self_state(state, inner_summary_json.as_deref(), None, settings);
        let disable_working_hypothesis = settings.stability_disable_working_hypothesis.unwrap_or(true);
        refresh_working_memory(state, Utc::now(), disable_working_hypothesis);
        update_workspace_meta_quality(&self.db, &mut state.workspace_meta).await;

        let json_state = serde_json::to_string(state).unwrap_or_else(|_| "{}".to_string());
        let state_write_owner = if is_internal_tick {
            "internal_tick"
        } else if background {
            "chat_background"
        } else {
            "chat"
        };
        let kernel_hash = hash_payload(&json_state);
        let kernel_dirty = state
            .last_persisted_state_hash
            .as_deref()
            .map(|h| h != kernel_hash)
            .unwrap_or(true);
        if kernel_dirty {
            sqlx::query(
                "INSERT INTO kernel_states (conversation_id, state_json, state_write_owner, updated_at)
                 VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                 ON CONFLICT(conversation_id) DO UPDATE SET state_json = excluded.state_json,
                     state_write_owner = excluded.state_write_owner,
                     monologue_write_version = monologue_write_version + 1,
                     updated_at = CURRENT_TIMESTAMP"
            )
            .bind(conversation_id)
            .bind(&json_state)
            .bind(state_write_owner)
            .execute(&mut *tx)
            .await
            .map_err(|e| e.to_string())?;
        }

        let workspace_state_json = serde_json::to_string(&json!({
            "goal_thread": state.workspace_goal_thread.clone(),
            "active_plan_id": state.workspace_active_plan_id.clone(),
            "goal_stack": state.workspace_goal_stack.clone(),
            "open_questions": state.workspace_open_questions.clone(),
            "active_hypotheses": state.workspace_active_hypotheses.clone(),
            "working_set_topics": state.workspace_working_set_topics.clone(),
            "current_focus": state.workspace_current_focus.clone(),
            "focus_rationale": state.workspace_focus_rationale.clone(),
            "workspace_meta": state.workspace_meta.clone(),
        }))
        .unwrap_or_else(|_| "{}".to_string());
        let workspace_hash = hash_payload(&workspace_state_json);
        let workspace_dirty = state
            .last_persisted_workspace_hash
            .as_deref()
            .map(|h| h != workspace_hash)
            .unwrap_or(true);
        let mut workspace_persist_error: Option<String> = None;
        if workspace_dirty {
            let goal_stack_json =
                serde_json::to_string(&state.workspace_goal_stack).unwrap_or_else(|_| "[]".to_string());
            let open_questions_json =
                serde_json::to_string(&state.workspace_open_questions).unwrap_or_else(|_| "[]".to_string());
            let active_hypotheses_json =
                serde_json::to_string(&state.workspace_active_hypotheses).unwrap_or_else(|_| "[]".to_string());
            let working_set_topics_json =
                serde_json::to_string(&state.workspace_working_set_topics).unwrap_or_else(|_| "[]".to_string());
            let workspace_meta_json =
                serde_json::to_string(&state.workspace_meta).unwrap_or_else(|_| "{}".to_string());
            if let Err(err) = sqlx::query(
                "INSERT INTO workspace_state (conversation_id, goal_thread, active_plan_id, goal_stack_json, open_questions_json, active_hypotheses_json,
                    working_set_topics_json, current_focus, focus_rationale, workspace_meta_json, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                 ON CONFLICT(conversation_id) DO UPDATE SET
                    goal_thread = excluded.goal_thread,
                    active_plan_id = COALESCE(excluded.active_plan_id, workspace_state.active_plan_id),
                    goal_stack_json = excluded.goal_stack_json,
                    open_questions_json = excluded.open_questions_json,
                    active_hypotheses_json = excluded.active_hypotheses_json,
                    working_set_topics_json = excluded.working_set_topics_json,
                    current_focus = excluded.current_focus,
                    focus_rationale = excluded.focus_rationale,
                    workspace_meta_json = excluded.workspace_meta_json,
                    updated_at = CURRENT_TIMESTAMP",
            )
            .bind(conversation_id)
            .bind(state.workspace_goal_thread.as_deref())
            .bind(state.workspace_active_plan_id.as_deref())
            .bind(&goal_stack_json)
            .bind(&open_questions_json)
            .bind(&active_hypotheses_json)
            .bind(&working_set_topics_json)
            .bind(state.workspace_current_focus.as_deref())
            .bind(state.workspace_focus_rationale.as_deref())
            .bind(&workspace_meta_json)
            .execute(&mut *tx)
            .await
            {
                workspace_persist_error = Some(err.to_string());
            }
        }

        tx.commit().await.map_err(|e| e.to_string())?;
        if kernel_dirty {
            state.last_persisted_state_hash = Some(kernel_hash);
        }
        if workspace_dirty && workspace_persist_error.is_none() {
            state.last_persisted_workspace_hash = Some(workspace_hash.clone());
        }

        if let Some(err) = workspace_persist_error {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "kernel",
                run_id,
                trace_id,
                json!({
                    "event": "workspace_persist_failed",
                    "error": err,
                }),
            )
            .await;
        } else if workspace_dirty && workspace_updated {
            let (verified_count, speculative_count) = workspace_meta_counts(state);
            let workspace_meta_hash = hash_payload(
                &serde_json::to_string(&state.workspace_meta).unwrap_or_else(|_| "{}".to_string()),
            );
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                run_id,
                trace_id,
                json!({
                    "event": "workspace_update",
                    "conversation_id": conversation_id,
                    "workspace_hash": workspace_hash,
                    "workspace_meta_hash": workspace_meta_hash,
                    "verified_count": verified_count,
                    "speculative_count": speculative_count,
                }),
            )
            .await;
        }

        self.log_decision(decision, run_id, trace_id).await;

        if !self_claims_to_write.is_empty() {
            for claim in self_claims_to_write {
                let claim_text = claim.claim_text.clone();
                let claim_key = claim.claim_key.clone();
                if let Ok(Some(claim_id)) = self_claims::record_self_claim(&self.db, claim).await {
                    self_claims_written = self_claims_written.saturating_add(1);
                    if state.self_identity_claim_id.is_none()
                        && is_identity_self_claim(&claim_text, &claim_key)
                    {
                        state.self_identity_claim_id = Some(claim_id.clone());
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "memory",
                            run_id,
                            trace_id,
                            json!({
                                "event": "self_identity_anchor_set",
                                "claim_id": claim_id,
                                "claim_key": claim_key,
                                "claim_text": claim_text,
                            }),
                        )
                        .await;
                    }
                }
            }
        }

        if !policy_violations.is_empty() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "memory_policy",
                run_id,
                trace_id,
                json!({
                    "event": "memory_policy_violation",
                    "violations": policy_violations,
                    "conversation_id": conversation_id,
                }),
            )
            .await;
            let _ = system_log::log_contract_violation(
                &self.db.pool,
                Some(&self.app_handle),
                run_id,
                trace_id,
                "memory_policy",
                "memory_write_blocked",
                Some(json!({
                    "violations": policy_violations,
                    "conversation_id": conversation_id,
                })),
            )
            .await;
            let payload = json!({
                "reason": "memory_policy_violation",
                "violations": policy_violations,
            });
            let snippet = "memory_write_blocked policy_violation".to_string();
            let _ = self
                .emit_system_evidence(
                    state,
                    settings,
                    conversation_id,
                    "memory_write_blocked",
                    run_id,
                    &snippet,
                    Some(&payload),
                )
                .await;
        }

        if inner_summary_written {
            let payload = json!({
                "category": "inner_summary",
                "count": 1,
            });
            let _ = self
                .emit_system_evidence(
                    state,
                    settings,
                    conversation_id,
                    "memory_write",
                    run_id,
                    "memory_write inner_summary",
                    Some(&payload),
                )
                .await;
        }
        if episodic_written > 0 {
            let payload = json!({
                "category": "episodic",
                "count": episodic_written,
            });
            let _ = self
                .emit_system_evidence(
                    state,
                    settings,
                    conversation_id,
                    "memory_write",
                    run_id,
                    "memory_write episodic",
                    Some(&payload),
                )
                .await;
        }
        if semantic_written {
            let payload = json!({
                "category": "semantic_core",
                "count": 1,
            });
            let _ = self
                .emit_system_evidence(
                    state,
                    settings,
                    conversation_id,
                    "memory_write",
                    run_id,
                    "memory_write semantic_core",
                    Some(&payload),
                )
                .await;
        }
        if self_claims_written > 0 {
            let payload = json!({
                "category": "self_claim",
                "count": self_claims_written,
            });
            let _ = self
                .emit_system_evidence(
                    state,
                    settings,
                    conversation_id,
                    "memory_write",
                    run_id,
                    "memory_write self_claim",
                    Some(&payload),
                )
                .await;
        }

        if !is_internal_tick {
            if let Some(candidate) = decision.accepted.iter().find(|c| {
                matches!(
                    c.kind,
                    CandidateKind::EmitMessage
                        | CandidateKind::AskUserQuestion
                        | CandidateKind::ToolCall
                        | CandidateKind::UpdateWorkspace
                )
            }) {
                self.spawn_counterfactual_simulation(
                    conversation_id,
                    run_id,
                    candidate,
                    state.last_user_input.as_deref(),
                    settings,
                );
            }
        }

        if !is_internal_tick {
            let context = build_prediction_context(state);
            let db = self.db.clone();
            let model_client = self.model_client.clone();
            let app_handle = self.app_handle.clone();
            let settings_snapshot = settings.clone();
            let conversation_id = conversation_id.to_string();
            let run_id = run_id.map(|s| s.to_string());
            let trace_id = trace_id.map(|s| s.to_string());
            tokio::spawn(async move {
                Kernel::run_prediction_generation(
                    db,
                    model_client,
                    app_handle,
                    context,
                    conversation_id,
                    run_id,
                    trace_id,
                    settings_snapshot,
                )
                .await;
            });
        }

        if background {
            let duration_ms = commit_started.elapsed().as_millis() as i64;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                run_id,
                trace_id,
                json!({
                    "event": "timing_commit_cycle",
                    "duration_ms": duration_ms,
                    "path": "background_commit",
                }),
            )
            .await;
            if duration_ms > PERF_WARN_COMMIT_MS {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "perf",
                    run_id,
                    trace_id,
                    json!({
                        "event": "performance_regression",
                        "stage": "commit_cycle",
                        "duration_ms": duration_ms,
                    }),
                )
                .await;
            }
            if run_id.is_some() && !is_internal_tick {
                self.emit_module_status(run_id, "idle", "Idle", commit_started_at, duration_ms);
            }
        }

        if run_id.is_some() && !is_internal_tick {
            let duration_ms = commit_started.elapsed().as_millis() as i64;
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "kernel",
                run_id,
                trace_id,
                json!({
                    "event": "post_processing_exited",
                    "duration_ms": duration_ms,
                }),
            )
            .await;
        }

        Ok(commit_result)
    }
}
