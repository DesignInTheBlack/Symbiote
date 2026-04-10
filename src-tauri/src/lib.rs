#![recursion_limit = "256"]

pub mod db;
pub mod core;
pub mod models;
pub mod commands;
mod commands_memory;
mod commands_memory_graph;

use std::sync::Arc;
use tauri::{Manager, WindowEvent};
use crate::db::Db;
use crate::core::ChatManager;
use crate::core::voice_manager_v2::VoiceManager; // ImportAdded
use crate::core::scheduler::Scheduler; // Import Added

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                let pool = window
                    .app_handle()
                    .state::<sqlx::SqlitePool>()
                    .inner()
                    .clone();
                tauri::async_runtime::spawn(async move {
                    let _ = crate::core::system_log::flush_logs(&pool).await;
                });
            }
        })
        .setup(|app| {
            let app_handle = app.handle().clone();
            let app_handle_for_db = app_handle.clone(); // Clone for async move

            let env_diag = std::env::var("SYMBIOTE_DIAG_STDERR").ok();
            let env_skip_prompts = std::env::var("SYMBIOTE_SKIP_PROMPT_LOGS").ok();
            let env_disable_scheduler = std::env::var("SYMBIOTE_DISABLE_SCHEDULER").ok();
            let env_disable_voice = std::env::var("SYMBIOTE_DISABLE_VOICE").ok();
            eprintln!(
                "[Env] SYMBIOTE_DIAG_STDERR={:?} SYMBIOTE_SKIP_PROMPT_LOGS={:?} SYMBIOTE_DISABLE_SCHEDULER={:?} SYMBIOTE_DISABLE_VOICE={:?}",
                env_diag, env_skip_prompts, env_disable_scheduler, env_disable_voice
            );

            let init_result = tauri::async_runtime::block_on(async move {
                let db = Db::new(&app_handle_for_db)
                    .await
                    .map_err(|e| format!("Failed to initialize database: {e}"))?;

                // Initialize Settings (Required by commands)
                let settings = db
                    .get_settings()
                    .await
                    .map(Arc::new)
                    .map_err(|e| format!("Failed to load settings: {e}"))?;

                let db_arc = Arc::new(db);

                // Initialize ModelClient globally
                let model_client = Arc::new(crate::core::model_client::ModelClient::new(
                    db_arc.pool.clone(),
                    app_handle_for_db.clone(),
                ));

                let chat_manager = Arc::new(
                    ChatManager::new(db_arc.clone(), model_client.clone(), app_handle_for_db.clone()).await,
                );
                let chat_manager_for_scheduler = chat_manager.clone();

                app_handle_for_db.manage(db_arc.clone());
                app_handle_for_db.manage(db_arc.pool.clone()); // Explicitly manage SqlitePool for commands
                app_handle_for_db.manage(settings.clone());
                app_handle_for_db.manage(model_client); // Manage ModelClient globally
                app_handle_for_db.manage(chat_manager);

                // Start Scheduler (allow disabling for diagnostics)
                if std::env::var("SYMBIOTE_DISABLE_SCHEDULER")
                    .ok()
                    .as_deref()
                    != Some("1")
                {
                    let scheduler = Scheduler::new(
                        db_arc.clone(),
                        app_handle_for_db.clone(),
                        chat_manager_for_scheduler.kernel.clone(),
                    );
                    scheduler.start();
                } else {
                    eprintln!("[Scheduler] Disabled via SYMBIOTE_DISABLE_SCHEDULER=1");
                }

                Ok::<(), String>(())
            });

            if let Err(err) = init_result {
                eprintln!("[FATAL] {err}");
                return Err(err.into());
            }

            // Initialize VoiceManager
            let voice_manager = VoiceManager::new();
            app_handle.manage(voice_manager); // Changed from app.manage to app_handle.manage
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_messages,
            commands::get_settings,
            commands::get_prompt_status,
            commands::get_rolling_summary,
            commands::get_rolling_summary_status,
            commands::get_live_summary,
            commands::get_live_summary_status,
            commands::get_inner_monologue_entries,
            commands::get_cognitive_readiness_report,
            commands::run_cognitive_checks,
            commands::get_system_logs,
            commands::get_system_capabilities,
            commands::get_evidence_lineage,
            commands::get_system_controls,
            commands::set_system_control,
            commands::get_system_control_events,
            commands::get_system_health_snapshot,
            commands::get_system_health_history,
            commands::capture_baseline_metrics,
            commands::get_baseline_metrics,
            commands::apply_recommendation,
            commands::dismiss_recommendation,
            commands::get_diagnostics_snapshot,
            commands::get_wave_status,
            commands::get_self_model,
            commands::get_parameter_registry,
            commands::update_parameter_registry,
            commands::update_settings,
            commands::get_phi_consent_scope,
            commands::set_phi_consent_scope,
            commands::list_themes,
            commands::read_theme_file,
            commands::send_message,
            commands::list_pending_prompts,
            commands::get_pending_prompt_count,
            commands::dismiss_pending_prompt,
            commands::rephrase_pending_prompt,
            commands::send_pending_prompt,
            commands::abort_generation,
            commands::test_connection,
            commands::clear_history,
            commands::get_episodic_events,
            commands::normalize_url,
            commands::submit_clarification,
            commands::reset_conversation_data,
            commands::reset_all_data,
            commands::start_voice_service,
            commands::restart_voice_service,
            commands::log_ui_timing,
            commands::log_tts_event,
            commands::trigger_reminder_response,
            commands::create_reminder,
            commands::search_episodic_events,
            commands::record_qualia_label,
            commands::record_qualia_reward,
            commands::record_outcome,
            commands::list_outcomes,
            commands::get_subject_snapshots,
            commands::get_gate_decisions,
            commands::get_context_tags,
            commands::get_user_intent_summary,
            commands::update_user_intent_summary,
            commands::get_introspection_entries,
            commands::get_audit_log,
            commands::get_error_events,
            commands::get_qualia_labels,
            commands_memory::memory_write,
            commands_memory::memory_retrieve,
            commands_memory::memory_retrieval_debug,
            commands_memory::memory_get_last_debug,
            commands_memory::memory_get_scopes,
            commands_memory::memory_consolidate,
            commands_memory::memory_health_check,
            commands_memory::memory_resolve_conflict,
            commands_memory::memory_resolve_clarify,
            commands_memory::memory_list_conflicts,
            commands_memory::self_memory_list_changes,
            commands_memory::self_memory_rollback,
            commands_memory::self_inspect,
            commands_memory::set_reflection_frozen,
            commands_memory::list_reflection_staging,
            commands_memory::approve_reflection_staging,
            commands_memory::reject_reflection_staging,
            commands_memory::self_model_rollback,
            commands_memory::identity_rollback,
            commands_memory::memory_get_provenance,
            commands_memory::memory_get_entity_provenance,
            commands_memory::memory_list_relation_shape_missing,
            commands_memory::record_strategy_trace,
            commands_memory::list_strategy_traces,
            commands_memory::create_policy_version,
            commands_memory::list_policy_versions,
            commands_memory::create_memory_claim,
            commands_memory::list_memory_claims,
            commands_memory::memory_get_claim_outcomes,
            commands_memory::memory_evaluate_claims,
            commands_memory::memory_backfill_relation_shape_claims,
            commands_memory::memory_backfill_rel_type_claims,
            commands_memory::update_memory_claim_status,
            commands_memory_graph::memory_get_graph,
            commands_memory_graph::memory_update_entity,
            commands_memory_graph::memory_delete_entity,
            commands_memory_graph::memory_delete_belief
        ])
        .build(tauri::generate_context!())
        .unwrap_or_else(|err| {
            eprintln!("[FATAL] error while building tauri application: {err}");
            std::process::exit(1);
        })
        .run(|app_handle, event| { // Modified signature to capture app_handle
            match event {
                tauri::RunEvent::ExitRequested { .. } => {
                    let voice = app_handle.state::<VoiceManager>();
                    voice.stop(Some(&app_handle));
                    let db = app_handle.state::<Arc<Db>>();
                    let _ = tauri::async_runtime::block_on(async {
                        let conversation_ids = db
                            .inner()
                            .list_conversation_ids(None)
                            .await
                            .unwrap_or_else(|_| vec!["default".to_string()]);
                        for conversation_id in conversation_ids {
                            let _ = crate::core::rolling_summary::archive_rolling_summary(
                                db.inner().clone(),
                                &conversation_id,
                                "scheduler",
                                "summary_archive",
                            )
                            .await;
                        }
                    });
                }
                _ => {}
            }
        });
}
