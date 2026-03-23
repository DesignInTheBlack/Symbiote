use sqlx::SqlitePool;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use once_cell::sync::Lazy;
use crate::core::memory::types::{Scope, SourceType, QueryIntent, MemoryPacket};
use crate::core::system_controls;
use crate::core::memory::compiler::{self, CompileContext, CompileResult};
use crate::core::memory::claims;
use crate::core::memory::retrieval::{self};
use crate::core::memory::attention::working_set;
use crate::core::memory::consolidation::{sketches, aliases};
use crate::core::world_model_reconcile;
use crate::core::memory::clarify::{self, ClarifyResult};
use crate::core::memory::dsl::{validate_memory_block, repair_memory_block, RepairContext};
use crate::core::model_client::ModelClient;
use crate::db::Db;
use crate::core::system_log;
use crate::core::memory::writer::EmbeddingConfig;
use sqlx::Row;
use uuid::Uuid;

const MEMORY_RETRIEVAL_CACHE_TTL_SECS: u64 = 30;
const MEMORY_RETRIEVAL_CACHE_LIMIT: usize = 64;
const MEMORY_GATE_FALLBACK_HOURS: i64 = 6;

struct GateDecisionResolution {
    decision: Option<String>,
    decision_id: Option<String>,
    snapshot_hash: Option<String>,
    fallback_reason: Option<String>,
}

#[derive(Clone)]
struct MemoryRetrievalCacheEntry {
    created_at: Instant,
    packet: MemoryPacket,
}

static MEMORY_RETRIEVAL_CACHE: Lazy<Mutex<HashMap<String, MemoryRetrievalCacheEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn gate_allows_memory(decision: Option<&str>) -> bool {
    matches!(
        decision,
        Some("ALLOW" | "ALLOW_WITH_NOTICE" | "ALLOW_WITH_AUDIT")
    )
}

pub struct MemoryApi {
    pool: SqlitePool,
    model_client: Option<Arc<ModelClient>>,
    session_id: String,
    embedding_config: Option<EmbeddingConfig>,
}

impl MemoryApi {
    pub async fn new(pool: SqlitePool, model_client: Option<Arc<ModelClient>>, session_id: String) -> Self {
        // Load embedding config from settings (only enable when a model is configured)
        let embedding_config = sqlx::query("SELECT api_base_url, embedding_model FROM settings WHERE id = 1")
            .fetch_optional(&pool)
            .await
            .ok()
            .flatten()
            .and_then(|row: sqlx::sqlite::SqliteRow| {
                use sqlx::Row;
                let base_url: String = row.get("api_base_url");
                let model: Option<String> = row.try_get("embedding_model").ok().flatten();
                let model = model
                    .map(|m| m.trim().to_string())
                    .filter(|m| !m.is_empty());
                model.map(|model| EmbeddingConfig {
                    base_url: normalize_base_url(&base_url),
                    model,
                    enabled: true,
                })
            });
        
        Self { pool, model_client, session_id, embedding_config }
    }

    async fn resolve_gate_decision(&self, snapshot_hash: Option<&str>) -> GateDecisionResolution {
        if let Some(hash) = snapshot_hash {
            if let Ok(row_opt) = sqlx::query(
                "SELECT decision_id, decision FROM gate_decisions
                 WHERE snapshot_hash = ?
                 ORDER BY datetime(created_at) DESC
                 LIMIT 1",
            )
            .bind(hash)
            .fetch_optional(&self.pool)
            .await
            {
                if let Some(row) = row_opt {
                    let decision_id: String = row.get("decision_id");
                    let decision: String = row.get("decision");
                    return GateDecisionResolution {
                        decision: Some(decision),
                        decision_id: Some(decision_id),
                        snapshot_hash: Some(hash.to_string()),
                        fallback_reason: None,
                    };
                }
            }
        }

        let window = format!("-{} hours", MEMORY_GATE_FALLBACK_HOURS);
        if let Ok(row_opt) = sqlx::query(
            "SELECT g.decision_id, g.decision, g.snapshot_hash
             FROM gate_decisions g
             JOIN subject_snapshots s ON s.snapshot_hash = g.snapshot_hash
             WHERE s.conversation_id = ?
               AND datetime(g.created_at) >= datetime('now', ?)
             ORDER BY datetime(g.created_at) DESC, g.decision_id DESC
             LIMIT 1",
        )
        .bind(&self.session_id)
        .bind(&window)
        .fetch_optional(&self.pool)
        .await
        {
            if let Some(row) = row_opt {
                let decision_id: String = row.get("decision_id");
                let decision: String = row.get("decision");
                let snapshot_hash: String = row.get("snapshot_hash");
                return GateDecisionResolution {
                    decision: Some(decision),
                    decision_id: Some(decision_id),
                    snapshot_hash: Some(snapshot_hash),
                    fallback_reason: Some("conversation_fallback".to_string()),
                };
            }
        }

        GateDecisionResolution {
            decision: Some("ALLOW_WITH_NOTICE".to_string()),
            decision_id: None,
            snapshot_hash: snapshot_hash.map(|v| v.to_string()),
            fallback_reason: Some("default_allow_notice".to_string()),
        }
    }
    
    /// Get embedding config reference for passing to compiler
    pub fn embedding_config(&self) -> Option<&EmbeddingConfig> {
        self.embedding_config.as_ref()
    }
    
    // §15.1 Parse & Compile
    pub async fn parse_and_compile(&self, dsl_lines: &str, scope: Scope, source: SourceType, source_ref: Option<String>, now: chrono::DateTime<chrono::Utc>) -> CompileResult {
        let snapshot_hash: Option<String> = sqlx::query_scalar(
            "SELECT snapshot_hash FROM subject_snapshots
             WHERE conversation_id = ?
             ORDER BY datetime(timestamp) DESC
             LIMIT 1",
        )
        .bind(&self.session_id)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();
        let gate_resolution = self.resolve_gate_decision(snapshot_hash.as_deref()).await;
        if let Some(reason) = gate_resolution.fallback_reason.as_deref() {
            let _ = system_log::log_event(
                &self.pool,
                None,
                "info",
                "memory",
                None,
                None,
                serde_json::json!({
                    "event": "memory_gate_fallback",
                    "reason": reason,
                    "conversation_id": self.session_id,
                    "snapshot_hash": gate_resolution.snapshot_hash,
                    "gate_decision": gate_resolution.decision,
                    "gate_decision_id": gate_resolution.decision_id,
                }),
            )
            .await;
        }
        let gate_decision = gate_resolution.decision;
        if !gate_allows_memory(gate_decision.as_deref()) {
            let _ = system_log::log_event(
                &self.pool,
                None,
                "warn",
                "memory",
                None,
                None,
                serde_json::json!({
                    "event": "memory_write_blocked",
                    "reason": "gate_decision",
                    "gate_decision": gate_decision,
                    "gate_decision_id": gate_resolution.decision_id,
                    "snapshot_hash": snapshot_hash,
                    "conversation_id": self.session_id,
                }),
            )
            .await;
            return CompileResult {
                written_ids: Vec::new(),
                conflict_ids: Vec::new(),
                pending_writes: Vec::new(),
                pending_clarify: None,
                claim_ids: Vec::new(),
                errors: vec!["memory_write_blocked: gate_decision".to_string()],
            };
        }
        let memory_mode = self.memory_write_mode().await;
        if !system_controls::allow_memory_write(&memory_mode, "memory_api_write") {
            let _ = system_log::log_event(
                &self.pool,
                None,
                "warn",
                "memory",
                None,
                None,
                serde_json::json!({
                    "event": "memory_write_blocked",
                    "reason": "system_control",
                    "mode": memory_mode,
                    "conversation_id": self.session_id,
                }),
            )
            .await;
            return CompileResult {
                written_ids: Vec::new(),
                conflict_ids: Vec::new(),
                pending_writes: Vec::new(),
                pending_clarify: None,
                claim_ids: Vec::new(),
                errors: vec!["memory_write_blocked: system_control".to_string()],
            };
        }
        let mut dsl_block = dsl_lines.trim().to_string();
        if dsl_block.is_empty() {
            return CompileResult {
                written_ids: Vec::new(),
                conflict_ids: Vec::new(),
                pending_writes: Vec::new(),
                pending_clarify: None,
                claim_ids: Vec::new(),
                errors: vec!["memory_dsl_empty".to_string()],
            };
        }

        let validation = validate_memory_block(&dsl_block);
        if !validation.valid {
            let repair_ctx = RepairContext { now, assistant_name: None };
            let outcome = repair_memory_block(&dsl_block, &repair_ctx);
            let _ = system_log::log_event(
                &self.pool,
                None,
                "info",
                "memory",
                None,
                None,
                serde_json::json!({
                    "event": "memory_dsl_repair_attempt",
                    "conversation_id": self.session_id,
                    "errors": validation.errors,
                    "statement_count": validation.statement_count,
                    "repaired": outcome.repaired,
                    "dropped_lines": outcome.dropped_lines,
                }),
            )
            .await;

            if let Some(repaired) = outcome.repaired_block {
                let repaired_validation = validate_memory_block(&repaired);
                if repaired_validation.valid {
                    dsl_block = repaired;
                    let _ = system_log::log_event(
                        &self.pool,
                        None,
                        "info",
                        "memory",
                        None,
                        None,
                        serde_json::json!({
                            "event": "memory_dsl_repaired",
                            "conversation_id": self.session_id,
                            "statement_count": repaired_validation.statement_count,
                        }),
                    )
                    .await;
                } else {
                    let _ = system_log::log_event(
                        &self.pool,
                        None,
                        "warn",
                        "memory",
                        None,
                        None,
                        serde_json::json!({
                            "event": "memory_dsl_invalid",
                            "conversation_id": self.session_id,
                            "errors": repaired_validation.errors,
                            "statement_count": repaired_validation.statement_count,
                        }),
                    )
                    .await;
                    return CompileResult {
                        written_ids: Vec::new(),
                        conflict_ids: Vec::new(),
                        pending_writes: Vec::new(),
                        pending_clarify: None,
                        claim_ids: Vec::new(),
                        errors: repaired_validation.errors,
                    };
                }
            } else {
                let _ = system_log::log_event(
                    &self.pool,
                    None,
                    "warn",
                    "memory",
                    None,
                    None,
                    serde_json::json!({
                        "event": "memory_dsl_invalid",
                        "conversation_id": self.session_id,
                        "errors": outcome.errors,
                        "statement_count": validation.statement_count,
                    }),
                )
                .await;
                return CompileResult {
                    written_ids: Vec::new(),
                    conflict_ids: Vec::new(),
                    pending_writes: Vec::new(),
                    pending_clarify: None,
                    claim_ids: Vec::new(),
                    errors: outcome.errors,
                };
            }
        }
        let ctx = CompileContext {
            pool: self.pool.clone(),
            model_client: self.model_client.clone(),
            session_id: self.session_id.clone(),
            scope,
            source,
            source_ref,
            now,
            embedding_config: self.embedding_config.clone(),
            skip_claims: false,
            allow_ambiguous_user_refs: matches!(source, SourceType::User),
        };
        
        let result = compiler::compile(&dsl_block, ctx).await;
        
        // Update Working Set (Written)
        if !result.written_ids.is_empty() {
             let entity_ids = self.collect_entity_ids_for_beliefs(&result.written_ids).await;
             let _ = working_set::update_working_set(&self.pool, &entity_ids, &result.written_ids).await;
        }

        if compiler::memory_claims_enabled(&self.pool).await {
            let _ = claims::evaluate_pending_claims(&self.pool, self.model_client.clone(), 10).await;
        }

        if result.errors.is_empty() {
            let existing_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memory_write_ledger")
                .fetch_optional(&self.pool)
                .await
                .ok()
                .flatten()
                .unwrap_or(0);
            if existing_count == 0 {
                let bootstrap_id = Uuid::new_v4().to_string();
                let _ = sqlx::query(
                    "INSERT INTO memory_write_ledger
                     (id, conversation_id, category, source, reason_code, gate_decision_id, created_at)
                     VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                )
                .bind(&bootstrap_id)
                .bind(&self.session_id)
                .bind("memory_pass")
                .bind("memory_writer")
                .bind("bootstrap")
                .bind(gate_resolution.decision_id.clone())
                .execute(&self.pool)
                .await;
                let _ = system_log::log_event(
                    &self.pool,
                    None,
                    "info",
                    "memory",
                    None,
                    None,
                    serde_json::json!({
                        "event": "memory_bootstrap_written",
                        "conversation_id": self.session_id,
                        "bootstrap_id": bootstrap_id,
                    }),
                )
                .await;
            }
        }
        
        result
    }
    
    // §15.2 Resolve Clarify
    pub async fn resolve_clarify(&self, pending_id: i64, user_reply: &str, scope: Scope, source: SourceType) -> ClarifyResult {
        let snapshot_hash: Option<String> = sqlx::query_scalar(
            "SELECT snapshot_hash FROM subject_snapshots
             WHERE conversation_id = ?
             ORDER BY datetime(timestamp) DESC
             LIMIT 1",
        )
        .bind(&self.session_id)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();
        let gate_resolution = self.resolve_gate_decision(snapshot_hash.as_deref()).await;
        if let Some(reason) = gate_resolution.fallback_reason.as_deref() {
            let _ = system_log::log_event(
                &self.pool,
                None,
                "info",
                "memory",
                None,
                None,
                serde_json::json!({
                    "event": "memory_gate_fallback",
                    "reason": reason,
                    "conversation_id": self.session_id,
                    "snapshot_hash": gate_resolution.snapshot_hash,
                    "gate_decision": gate_resolution.decision,
                    "gate_decision_id": gate_resolution.decision_id,
                }),
            )
            .await;
        }
        let gate_decision = gate_resolution.decision;
        if !gate_allows_memory(gate_decision.as_deref()) {
            let _ = system_log::log_event(
                &self.pool,
                None,
                "warn",
                "memory",
                None,
                None,
                serde_json::json!({
                    "event": "memory_write_blocked",
                    "reason": "gate_decision",
                    "gate_decision": gate_decision,
                    "gate_decision_id": gate_resolution.decision_id,
                    "snapshot_hash": snapshot_hash,
                    "conversation_id": self.session_id,
                }),
            )
            .await;
            return ClarifyResult {
                success: false,
                selected_entity_id: None,
                selected_label: None,
                compile_result: None,
                error: Some("memory_write_blocked: gate_decision".to_string()),
            };
        }
        let memory_mode = self.memory_write_mode().await;
        if !system_controls::allow_memory_write(&memory_mode, "memory_api_write") {
            let _ = system_log::log_event(
                &self.pool,
                None,
                "warn",
                "memory",
                None,
                None,
                serde_json::json!({
                    "event": "memory_write_blocked",
                    "reason": "system_control",
                    "mode": memory_mode,
                    "conversation_id": self.session_id,
                }),
            )
            .await;
            return ClarifyResult {
                success: false,
                selected_entity_id: None,
                selected_label: None,
                compile_result: None,
                error: Some("memory_write_blocked: system_control".to_string()),
            };
        }
        let result = clarify::resolve_clarify(
            pending_id,
            user_reply,
            &self.pool,
            self.model_client.clone(),
            scope,
            source,
            self.embedding_config.clone(),
        ).await;
        
        // Update Working Set with any new written IDs
        if let Some(ref compile_result) = result.compile_result {
            if !compile_result.written_ids.is_empty() {
                let entity_ids = self.collect_entity_ids_for_beliefs(&compile_result.written_ids).await;
                let _ = working_set::update_working_set(&self.pool, &entity_ids, &compile_result.written_ids).await;
            }
        }
        
        result
    }

    fn retrieval_cache_key(&self, query: &str, scopes: &[Scope], intent: &QueryIntent) -> String {
        let scopes_key = scopes
            .iter()
            .map(|scope| format!("{:?}", scope))
            .collect::<Vec<_>>()
            .join("|");
        format!(
            "{}|{}|{}|{:?}",
            self.session_id,
            query.trim().to_lowercase(),
            scopes_key,
            intent
        )
    }

    async fn retrieval_cache_get(
        &self,
        query: &str,
        scopes: &[Scope],
        intent: &QueryIntent,
    ) -> Option<MemoryPacket> {
        let query = query.trim();
        if query.is_empty() {
            return None;
        }
        let db = Db { pool: self.pool.clone() };
        let recent_write = db
            .has_recent_memory_write(MEMORY_RETRIEVAL_CACHE_TTL_SECS as i64)
            .await
            .unwrap_or(false);
        if recent_write {
            return None;
        }

        let key = self.retrieval_cache_key(query, scopes, intent);
        let mut cache = match MEMORY_RETRIEVAL_CACHE.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(entry) = cache.get(&key) {
            if entry.created_at.elapsed() <= Duration::from_secs(MEMORY_RETRIEVAL_CACHE_TTL_SECS) {
                return Some(entry.packet.clone());
            }
        }
        cache.remove(&key);
        None
    }

    fn retrieval_cache_store(
        &self,
        query: &str,
        scopes: &[Scope],
        intent: &QueryIntent,
        packet: &MemoryPacket,
    ) {
        let query = query.trim();
        if query.is_empty() {
            return;
        }
        let key = self.retrieval_cache_key(query, scopes, intent);
        let mut cache = match MEMORY_RETRIEVAL_CACHE.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if cache.len() >= MEMORY_RETRIEVAL_CACHE_LIMIT {
            if let Some(evict) = cache.keys().next().cloned() {
                cache.remove(&evict);
            }
        }
        cache.insert(
            key,
            MemoryRetrievalCacheEntry {
                created_at: Instant::now(),
                packet: packet.clone(),
            },
        );
    }
    
    // §15.3 Retrieve
    pub async fn retrieve(&self, query: &str, scopes: &[Scope], intent: QueryIntent) -> Result<MemoryPacket, String> {
        let mode = self.memory_retrieval_mode().await;
        if system_controls::mode_is_off(&mode) {
            return Ok(self.empty_packet());
        }
        self.retrieve_internal(query, scopes, intent, false).await
    }

    /// Fast-path retrieval without side effects (no salience or working-set writes).
    /// Intended for idle cognition paths where speed matters more than updates.
    pub async fn retrieve_fast(&self, query: &str, scopes: &[Scope], intent: QueryIntent) -> Result<MemoryPacket, String> {
        let mode = self.memory_retrieval_mode().await;
        if system_controls::mode_is_off(&mode) {
            return Ok(self.empty_packet());
        }
        let mut packet = retrieval::retrieve_with_options(
            query,
            scopes,
            intent,
            &self.pool,
            self.model_client.clone(),
            self.embedding_config.as_ref(),
            false,
        )
        .await?;
        if system_controls::mode_is_degraded(&mode) {
            packet = self.trim_packet_for_degraded(packet);
        }
        Ok(packet)
    }
    
    // §15.3 + §13 Retrieve with debug info
    pub async fn retrieve_with_debug(&self, query: &str, scopes: &[Scope], intent: QueryIntent) -> Result<MemoryPacket, String> {
        let mode = self.memory_retrieval_mode().await;
        if system_controls::mode_is_off(&mode) {
            return Ok(self.empty_packet());
        }
        self.retrieve_internal(query, scopes, intent, true).await
    }
    
    async fn retrieve_internal(&self, query: &str, scopes: &[Scope], intent: QueryIntent, debug: bool) -> Result<MemoryPacket, String> {
        let mode = self.memory_retrieval_mode().await;
        if system_controls::mode_is_off(&mode) {
            return Ok(self.empty_packet());
        }
        let mut cache_hit = false;
        let mut packet = if !debug {
            if !system_controls::mode_is_degraded(&mode) {
                if let Some(packet) = self.retrieval_cache_get(query, scopes, &intent).await {
                    cache_hit = true;
                    packet
                } else {
                    retrieval::retrieve_with_options(
                        query,
                        scopes,
                        intent.clone(),
                        &self.pool,
                        self.model_client.clone(),
                        self.embedding_config.as_ref(),
                        debug,
                    )
                    .await?
                }
            } else {
                retrieval::retrieve_with_options(
                    query,
                    scopes,
                    intent.clone(),
                    &self.pool,
                    self.model_client.clone(),
                    self.embedding_config.as_ref(),
                    debug,
                )
                .await?
            }
        } else {
            retrieval::retrieve_with_options(
                query,
                scopes,
                intent.clone(),
                &self.pool,
                self.model_client.clone(),
                self.embedding_config.as_ref(),
                debug,
            )
            .await?
        };
        if system_controls::mode_is_degraded(&mode) {
            packet = self.trim_packet_for_degraded(packet);
        }
        if !debug && !cache_hit {
            self.retrieval_cache_store(query, scopes, &intent, &packet);
        }

        // Collect belief IDs to boost salience on access.
        let mut accessed_belief_ids: Vec<i64> = Vec::new();
        for fact in &packet.facts {
            if !accessed_belief_ids.contains(&fact.id) {
                accessed_belief_ids.push(fact.id);
            }
        }
        for rel in &packet.relations {
            if !accessed_belief_ids.contains(&rel.id) {
                accessed_belief_ids.push(rel.id);
            }
        }
        if !accessed_belief_ids.is_empty() {
            let _ = crate::core::memory::attention::salience::boost_salience_for_beliefs(
                &self.pool,
                &accessed_belief_ids,
                0.2,
            )
            .await;
        }

        // FIX: Collect ENTITY IDs (not belief IDs) from accessed facts and relations.
        let mut accessed_entity_ids: Vec<i64> = vec![];
        
        // Collect entity IDs from facts
        for f in &packet.facts { 
            if !accessed_entity_ids.contains(&f.entity_id) {
                accessed_entity_ids.push(f.entity_id); 
            }
        }
        
        // Collect entity IDs from relation participants
        for rel in &packet.relations {
            for participant in &rel.participants {
                if !accessed_entity_ids.contains(&participant.entity_id) {
                    accessed_entity_ids.push(participant.entity_id);
                }
            }
        }
        
        if !accessed_entity_ids.is_empty() {
            // Update working set with entity IDs (not belief IDs!)
            let _ = working_set::update_working_set(&self.pool, &accessed_entity_ids, &[]).await;
            
            // Increment access_count on accessed entities
            for id in &accessed_entity_ids {
                let _ = sqlx::query("UPDATE ics_entities SET access_count = access_count + 1, last_accessed_at = CURRENT_TIMESTAMP WHERE id = ?")
                    .bind(id)
                    .execute(&self.pool)
                    .await;
            }
        }
        
        Ok(packet)
    }
    
    // §15.5 Consolidate (§12 maintenance tasks)
    pub async fn consolidate(&self) -> Result<ConsolidationResult, String> {
        use crate::core::memory::consolidation::{stale, archive};
        let consolidation_mode = self.memory_consolidation_mode().await;
        let memory_mode = self.memory_write_mode().await;
        if system_controls::mode_is_off(&consolidation_mode)
            || system_controls::mode_is_degraded(&consolidation_mode)
            || system_controls::mode_is_off(&memory_mode)
            || system_controls::mode_is_read_only(&memory_mode)
        {
            return Ok(ConsolidationResult {
                aliases_promoted: 0,
                sketches_updated: 0,
                stale_deactivated: 0,
                conflicts_archived: 0,
            });
        }
        // Run all maintenance tasks
        let aliases_count = aliases::promote_eligible_aliases(&self.pool).await?;
        let sketches_count = sketches::update_entity_sketches(&self.pool).await?;
        let stale_count = stale::deactivate_stale_inferences(
            &self.pool,
            &stale::StaleInferenceConfig::default()
        ).await?;
        let archived_count = archive::archive_old_conflicts(
            &self.pool,
            &archive::ArchiveConfig::default()
        ).await?;

        let reconcile_mode_raw: Option<String> = sqlx::query_scalar(
            "SELECT world_model_reconcile_mode FROM settings WHERE id = 1",
        )
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();
        let reconcile_mode =
            world_model_reconcile::WorldModelReconcileMode::from_str(reconcile_mode_raw.as_deref());
        let reconcile_report =
            world_model_reconcile::reconcile_conflict_sets(&self.pool, &self.session_id, reconcile_mode)
                .await
                .unwrap_or(world_model_reconcile::WorldModelReconcileReport {
                    scanned: 0,
                    resolved: 0,
                    unresolved: 0,
                    outcomes: Vec::new(),
                    mode: reconcile_mode.as_str().to_string(),
                });
        let _ = system_log::log_event(
            &self.pool,
            None,
            "info",
            "memory",
            None,
            None,
            serde_json::json!({
                "event": "world_model_reconcile_summary",
                "conversation_id": self.session_id.as_str(),
                "scanned": reconcile_report.scanned,
                "resolved": reconcile_report.resolved,
                "unresolved": reconcile_report.unresolved,
                "mode": reconcile_report.mode,
            }),
        )
        .await;
        
        Ok(ConsolidationResult {
            aliases_promoted: aliases_count,
            sketches_updated: sketches_count,
            stale_deactivated: stale_count,
            conflicts_archived: archived_count,
        })
    }

    async fn collect_entity_ids_for_beliefs(&self, belief_ids: &[i64]) -> Vec<i64> {
        if belief_ids.is_empty() {
            return Vec::new();
        }

        let placeholders = belief_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let mut entity_ids = Vec::new();

        let fact_query = format!(
            "SELECT DISTINCT subject_entity_id AS entity_id FROM ics_fact_beliefs WHERE belief_id IN ({})",
            placeholders
        );
        let mut fact_stmt = sqlx::query(&fact_query);
        for id in belief_ids {
            fact_stmt = fact_stmt.bind(id);
        }
        if let Ok(rows) = fact_stmt.fetch_all(&self.pool).await {
            for row in rows {
                if let Ok(id) = row.try_get::<i64, _>("entity_id") {
                    if !entity_ids.contains(&id) {
                        entity_ids.push(id);
                    }
                }
            }
        }

        let rel_query = format!(
            "SELECT DISTINCT entity_id FROM ics_rel_participants WHERE belief_id IN ({})",
            placeholders
        );
        let mut rel_stmt = sqlx::query(&rel_query);
        for id in belief_ids {
            rel_stmt = rel_stmt.bind(id);
        }
        if let Ok(rows) = rel_stmt.fetch_all(&self.pool).await {
            for row in rows {
                if let Ok(id) = row.try_get::<i64, _>("entity_id") {
                    if !entity_ids.contains(&id) {
                        entity_ids.push(id);
                    }
                }
            }
        }

        entity_ids
    }

    async fn memory_retrieval_mode(&self) -> String {
        let mode: Option<String> = sqlx::query_scalar::<_, String>(
            "SELECT mode FROM system_controls WHERE subsystem_id = ?",
        )
        .bind("memory_retrieval")
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();
        mode.unwrap_or_else(|| {
            system_controls::default_mode_for("memory_retrieval")
                .unwrap_or("normal")
                .to_string()
        })
    }

    async fn memory_write_mode(&self) -> String {
        let mode: Option<String> = sqlx::query_scalar::<_, String>(
            "SELECT mode FROM system_controls WHERE subsystem_id = ?",
        )
        .bind("memory_write")
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();
        mode.unwrap_or_else(|| {
            system_controls::default_mode_for("memory_write")
                .unwrap_or("normal")
                .to_string()
        })
    }

    async fn memory_consolidation_mode(&self) -> String {
        let mode: Option<String> = sqlx::query_scalar::<_, String>(
            "SELECT mode FROM system_controls WHERE subsystem_id = ?",
        )
        .bind("memory_consolidation")
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();
        mode.unwrap_or_else(|| {
            system_controls::default_mode_for("memory_consolidation")
                .unwrap_or("normal")
                .to_string()
        })
    }

    fn trim_packet_for_degraded(&self, mut packet: MemoryPacket) -> MemoryPacket {
        packet
            .facts
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        packet
            .relations
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        packet.conflicts.truncate(2);
        packet.facts.truncate(3);
        packet.relations.truncate(3);
        if let Some(events) = packet.episodic_events.as_mut() {
            events.truncate(3);
        }
        packet
    }

    fn empty_packet(&self) -> MemoryPacket {
        MemoryPacket {
            facts: Vec::new(),
            relations: Vec::new(),
            conflicts: Vec::new(),
            bound_handles: HashMap::new(),
            shadowed_by_scope_count: None,
            dropped_by_scope_count: None,
            episodic_events: None,
            debug_log: None,
        }
    }
}

pub fn infer_query_intent(query: &str) -> QueryIntent {
    let q = query.to_lowercase();
    if q.contains("why")
        || q.contains("explain")
        || q.contains("how did")
        || q.contains("evidence")
        || q.contains("source")
    {
        return QueryIntent::AskExplain;
    }
    if q.contains("history")
        || q.contains("previous")
        || q.contains("earlier")
        || q.contains("before")
        || q.contains("past")
        || q.contains("timeline")
    {
        return QueryIntent::AskHistory;
    }
    if q.starts_with("list")
        || q.contains(" list ")
        || q.contains("show all")
        || q.contains("everything")
        || q.contains("enumerate")
    {
        return QueryIntent::AskList;
    }
    QueryIntent::AskCurrent
}

fn normalize_base_url(raw: &str) -> String {
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        trimmed.to_string()
    } else {
        format!("{}/v1", trimmed)
    }
}

/// Result of consolidation operation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConsolidationResult {
    pub aliases_promoted: usize,
    pub sketches_updated: usize,
    pub stale_deactivated: usize,
    pub conflicts_archived: usize,
}
