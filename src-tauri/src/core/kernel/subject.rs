use super::*;

impl Kernel {

    pub(super) async fn build_and_persist_subject_snapshot(
        &self,
        state: &mut KernelState,
        run_id: Option<&str>,
        tick_id: Option<&str>,
        reason: &str,
    ) -> Option<(subject_state::SubjectState, subject_state::SubjectSnapshotRecord)> {
        let previous_state = subject_state::load_latest_subject_state(&self.db, &state.conversation_id).await;
        let subject_state = match subject_state::build_subject_state(&self.db, state, previous_state.as_ref()).await {
            Ok(state) => state,
            Err(err) => {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    run_id,
                    None,
                    json!({
                        "event": "subject_snapshot_failed",
                        "reason": "build_subject_state",
                        "error": err,
                    }),
                )
                .await;
                return None;
            }
        };
        let tick = tick_id.map(|s| s.to_string()).unwrap_or_else(|| Uuid::new_v4().to_string());
        let snapshot = match subject_state::snapshot_subject_state(
            &subject_state,
            &tick,
            &state.conversation_id,
            run_id,
        ) {
            Ok(record) => record,
            Err(err) => {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    run_id,
                    None,
                    json!({
                        "event": "subject_snapshot_failed",
                        "reason": "snapshot_subject_state",
                        "error": err,
                    }),
                )
                .await;
                return None;
            }
        };
        if let Err(err) = subject_state::persist_subject_snapshot(&self.db, &snapshot).await {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "kernel",
                run_id,
                None,
                json!({
                    "event": "subject_snapshot_failed",
                    "reason": "persist_subject_snapshot",
                    "error": err,
                }),
            )
            .await;
            return None;
        }
        let contributors = core_workspace::build_workspace_contributors(
            &self.db,
            state,
            &subject_state,
            &snapshot.tick_id,
        )
        .await;
        let contributors_summary = core_workspace::summarize_contributors(&contributors);
        let attention_schema_summary =
            crate::core::attention_schema::summarize_for_prompt(&subject_state.attention_schema);
        update_workspace_runtime_meta(
            state,
            &subject_state.workspace,
            &contributors,
            &contributors_summary,
            &subject_state.attention_schema,
            &attention_schema_summary,
        );
        state.last_subject_snapshot_hash = Some(snapshot.snapshot_hash.clone());
        state.last_subject_snapshot_at = Some(snapshot.timestamp.clone());

        let missing_contributors = contributors.missing.clone();
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            run_id,
            None,
            json!({
                "event": "workspace_snapshot",
                "conversation_id": state.conversation_id,
                "snapshot_hash": snapshot.snapshot_hash,
                "tick_id": snapshot.tick_id,
                "contributors": contributors,
                "missing": missing_contributors,
            }),
        )
        .await;
        if !missing_contributors.is_empty() {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "warn",
                "kernel",
                run_id,
                None,
                json!({
                    "event": "workspace_missing_contributors",
                    "conversation_id": state.conversation_id,
                    "snapshot_hash": snapshot.snapshot_hash,
                    "tick_id": snapshot.tick_id,
                    "missing": missing_contributors,
                }),
            )
            .await;
        }
        let existing_gate: Option<i64> = sqlx::query_scalar(
            "SELECT 1 FROM gate_decisions WHERE snapshot_hash = ? LIMIT 1",
        )
        .bind(&snapshot.snapshot_hash)
        .fetch_optional(&self.db.pool)
        .await
        .ok()
        .flatten();
        if existing_gate.is_none() {
            let mut created_at = 0;
            let candidate = self.make_candidate(
                CandidateKind::EmitMessage,
                json!({ "content": "", "user_visible": false }),
                "snapshot_backfill",
                &mut created_at,
            );
            let proposal = subject_controller::build_action_proposal(&candidate);
            if let Err(err) = subject_controller::persist_action_proposal(&self.db, &snapshot.snapshot_hash, &proposal).await {
                let _ = system_log::log_event(
                    &self.db.pool,
                    Some(&self.app_handle),
                    "warn",
                    "kernel",
                    run_id,
                    None,
                    json!({
                        "event": "action_proposal_failed",
                        "reason": "snapshot_backfill",
                        "snapshot_hash": snapshot.snapshot_hash,
                        "error": err,
                    }),
                )
                .await;
            } else {
                let gate_signals = subject_controller::GateSignals::baseline();
                let gate = subject_controller::build_gate_decision(
                    &subject_state,
                    &candidate,
                    &StopState::default(),
                    &gate_signals,
                );
                match subject_controller::persist_gate_decision(
                    &self.db,
                    &snapshot.snapshot_hash,
                    &proposal.proposal_id,
                    &gate,
                )
                .await
                {
                    Ok(()) => {
                        let _ = system_log::log_event(
                            &self.db.pool,
                            Some(&self.app_handle),
                            "info",
                            "kernel",
                            run_id,
                            None,
                            json!({
                                "event": "gate_decision_written",
                                "decision_id": gate.decision_id,
                                "proposal_id": proposal.proposal_id,
                                "snapshot_hash": snapshot.snapshot_hash,
                                "decision": gate.decision,
                                "reason": "snapshot_backfill",
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
                            run_id,
                            None,
                            json!({
                                "event": "gate_decision_failed",
                                "reason": "snapshot_backfill",
                                "snapshot_hash": snapshot.snapshot_hash,
                                "error": err,
                            }),
                        )
                        .await;
                    }
                }
            }
        }
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            run_id,
            None,
            json!({
                "event": "subject_snapshot_written",
                "conversation_id": state.conversation_id,
                "snapshot_hash": snapshot.snapshot_hash,
                "tick_id": snapshot.tick_id,
                "reason": reason,
            }),
        )
        .await;
        let valence = organism::compute_valence_signal(&subject_state.organism);
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            run_id,
            None,
            json!({
                "event": "valence_signal",
                "conversation_id": state.conversation_id,
                "snapshot_hash": snapshot.snapshot_hash,
                "tick_id": snapshot.tick_id,
                "reason": reason,
                "goal_progress": valence.goal_progress,
                "uncertainty_change": valence.uncertainty_change,
                "integrity_change": valence.integrity_change,
                "alignment_change": valence.alignment_change,
                "net_valence": valence.net_valence,
            }),
        )
        .await;
        let _ = system_log::log_event(
            &self.db.pool,
            Some(&self.app_handle),
            "info",
            "kernel",
            run_id,
            None,
            json!({
                "event": "intrinsic_reward_event",
                "conversation_id": state.conversation_id,
                "snapshot_hash": snapshot.snapshot_hash,
                "tick_id": snapshot.tick_id,
                "reason": reason,
                "magnitude": valence.net_valence,
            }),
        )
        .await;
        if let Some(wave_state) = self.wave_state_for_validation(run_id, None).await {
            let _ = system_log::log_event(
                &self.db.pool,
                Some(&self.app_handle),
                "info",
                "cognitive_wave",
                run_id,
                None,
                json!({
                    "event": "wave_subject_comparison",
                    "conversation_id": state.conversation_id,
                    "snapshot_hash": snapshot.snapshot_hash,
                    "tick_id": snapshot.tick_id,
                    "reason": reason,
                    "wave": {
                        "coherence": wave_state.coherence,
                        "turbulence": wave_state.turbulence,
                        "drift": wave_state.drift,
                        "dominance": wave_state.dominance,
                        "fragmentation": wave_state.fragmentation,
                        "total_energy": wave_state.total_energy,
                        "band_energy": wave_state.band_energy,
                    },
                    "subject": {
                        "organism": {
                            "arousal": subject_state.organism.arousal,
                            "stress": subject_state.organism.stress,
                            "fatigue": subject_state.organism.fatigue,
                            "uncertainty_pressure": subject_state.organism.uncertainty_pressure,
                            "social_alignment": subject_state.organism.social_alignment,
                            "integrity_risk": subject_state.organism.integrity_risk,
                        },
                        "attention": {
                            "meta_confidence": subject_state.attention.meta_confidence,
                        },
                        "qualia": {
                            "dominant_tag": subject_state.qualia.dominant_tag,
                            "dominant_intensity": subject_state.qualia.dominant_intensity,
                            "last_reward": subject_state.qualia.last_reward,
                            "prediction_confidence": subject_state.qualia.prediction_confidence,
                        },
                        "self_model": {
                            "confidence": subject_state.self_model.controller_state.confidence,
                            "uncertainty": subject_state.self_model.controller_state.uncertainty,
                            "drift_score": subject_state.self_model.controller_state.drift_score,
                            "autonomy_level": subject_state.self_model.controller_state.autonomy_level,
                        }
                    }
                }),
            )
            .await;
        }
        Some((subject_state, snapshot))
    }
}
