use super::super::*;

impl Kernel {
    pub(crate) async fn capture_prompt_snapshot(
        &self,
        run_id: &str,
        trace_id: &str,
        build: &CorePromptBuild,
        attempt: i64,
    ) {
        let content = build.system_message.trim();
        if content.is_empty() {
            return;
        }
        let artifact_id = Uuid::new_v4().to_string();
        let payload = json!({
            "content": content,
            "prompt_hash": build.primary_prompt_hash,
            "canonical_prompt_hash": build.canonical_primary_hash,
            "memory_prompt_hash": build.memory_prompt_hash,
            "prompt_source": build.prompt_source,
            "prompt_layout": match build.prompt_layout {
                PromptLayout::Compact => "compact",
                PromptLayout::Full => "full",
            },
            "total_tokens": build.total_tokens,
            "total_chars": build.total_chars,
            "attempt": attempt,
            "captured_at": Utc::now().to_rfc3339(),
        });
        let _ = sqlx::query(
            "INSERT INTO artifacts (artifact_id, run_id, trace_id, type, schema_version, payload, produced_by, parent_artifact_ids, created_at)
             VALUES (?, ?, ?, 'prompt_capture', 1, ?, 'kernel', NULL, CURRENT_TIMESTAMP)",
        )
        .bind(&artifact_id)
        .bind(run_id)
        .bind(trace_id)
        .bind(payload.to_string())
        .execute(&self.db.pool)
        .await;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            Some(run_id),
            Some(trace_id),
            json!({
                "event": "prompt_captured",
                "artifact_id": artifact_id,
                "attempt": attempt,
            }),
        )
        .await;
    }

    pub(crate) async fn evaluate_compact_prompt(
        &self,
        input: &str,
        input_kind: CoreInputKind,
        self_audit_mode: bool,
        calculator_mode: bool,
        conversation_id: &str,
        state: &KernelState,
        settings: &crate::models::Settings,
    ) -> CompactPromptDecision {
        let mut disqualifiers = Vec::new();
        let enabled = settings.compact_prompt_enabled.unwrap_or(true);
        if !enabled {
            disqualifiers.push("disabled".to_string());
        }
        if !matches!(input_kind, CoreInputKind::User) {
            disqualifiers.push("non_user_input".to_string());
        }
        if self_audit_mode {
            disqualifiers.push("self_audit".to_string());
        }
        if calculator_mode {
            disqualifiers.push("calculator_mode".to_string());
        }
        if input.chars().count() > 240 {
            disqualifiers.push("input_len".to_string());
        }

        let self_awareness_allowed = settings.self_report_channel.unwrap_or(true)
            && !settings
                .self_awareness_expression_mode
                .as_deref()
                .unwrap_or("conservative")
                .eq_ignore_ascii_case("conservative");
        let self_awareness =
            matches!(input_kind, CoreInputKind::User) && self_awareness_allowed && is_self_awareness_query(input);
        let mut anchor_hits = 0usize;
        if matches!(input_kind, CoreInputKind::User) {
            let tool_names = self.tools.allowed_tool_names(settings);
            let anchor_vocab = build_anchor_vocab(state, &tool_names);
            anchor_hits = count_anchor_hits(input, &anchor_vocab);
        }

        let pending_prompts = self
            .db
            .count_pending_prompts(conversation_id)
            .await
            .unwrap_or(0);
        if pending_prompts > 0 {
            disqualifiers.push("pending_prompt".to_string());
        }

        let workspace_delta = state.last_workspace_delta_count;
        if workspace_delta > 1 {
            disqualifiers.push("workspace_delta".to_string());
        }

        if state.uncertainty_count > 0
            || state.stance.stance == "clarify"
            || !state.missing_slots.is_empty()
            || matches!(state.task_phase, TaskPhase::AwaitingUser)
        {
            disqualifiers.push("uncertainty_high".to_string());
        }

        let recent_memory_write = self.db.has_recent_memory_write(120).await.unwrap_or(false);
        if recent_memory_write {
            disqualifiers.push("recent_memory_write".to_string());
        }

        let mut force_reasons = Vec::new();
        if self_awareness {
            force_reasons.push("self_awareness".to_string());
        }
        if matches!(input_kind, CoreInputKind::User) && anchor_hits == 0 {
            force_reasons.push("low_anchor_hits".to_string());
        }

        let use_compact = !force_reasons.is_empty() || disqualifiers.is_empty();

        CompactPromptDecision {
            use_compact,
            disqualifiers,
            pending_prompts: pending_prompts as i64,
            recent_memory_write,
            workspace_delta,
            force_reasons,
            anchor_hits,
        }
    }
}
