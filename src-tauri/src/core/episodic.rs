use chrono::Utc;
use serde_json::Value;
use sqlx::SqlitePool;
use sqlx::Row;
use sqlx::Sqlite;
use sqlx::Transaction;
use uuid::Uuid;
use crate::core::memory::attention::working_set;
use crate::core::memory::snippets;
use crate::core::memory_policy::{MemoryPolicy, MemoryWriteCategory, MemoryWriteSource};
use crate::core::sensitivity::{phi_consent_allowed, redact_sensitive_json};
use crate::core::system_log;
use crate::core::system_controls;
use crate::db::Db;
use sha2::{Digest, Sha256};

const EPISODIC_SCHEMA_VERSION: i32 = 1;
const EPISODIC_EVENT_VERSION: i32 = 1;
const SUMMARY_SNIPPET_MAX: usize = 512;

const ALLOWED_PAYLOAD_KEYS: [&str; 19] = [
    "status",
    "tool_name",
    "error_code",
    "summary_snippet",
    "belief_id",
    "entity_id",
    "scope",
    "polarity",
    "source_ref",
    "rel_type",
    "roles_seen",
    "direction",
    "sample_dsl",
    "session_id",
    "timestamp",
    "decision_reason",
    "conflict_topic_key",
    "conflict_reason",
    "claim_id",
];

fn detect_identity_signal(text: &str) -> bool {
    let lowered = text.to_lowercase();
    let patterns = [
        "i am ",
        "i'm ",
        "my name is",
        "call me",
        "i go by",
        "i work as",
        "i am a ",
        "i'm a ",
        "as a ",
        "my role is",
        "i am the ",
        "i'm the ",
    ];
    patterns.iter().any(|pat| lowered.contains(pat))
}

fn detect_capability_signal(text: &str) -> bool {
    let lowered = text.to_lowercase();
    let patterns = [
        "i can ",
        "i can't",
        "i cannot",
        "i'm able to",
        "i am able to",
        "i'm unable to",
        "i am unable to",
        "capable of",
        "able to",
    ];
    patterns.iter().any(|pat| lowered.contains(pat))
}

fn compute_identity_relevance(event_type: &str, scope: Option<&str>, payload: &Value) -> f64 {
    let mut score: f64 = 0.1;
    if let Some(scope) = scope {
        if scope.trim().eq_ignore_ascii_case("self") {
            score += 0.45;
        }
    }
    let lowered_type = event_type.to_lowercase();
    if lowered_type.contains("identity") || lowered_type.contains("self") {
        score += 0.25;
    }
    let snippet = payload
        .get("summary_snippet")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if detect_identity_signal(snippet) {
        score += 0.35;
    }
    if detect_capability_signal(snippet) {
        score += 0.15;
    }
    score.clamp(0.0, 1.0)
}

fn derive_narrative_thread_id(event_type: &str, scope: Option<&str>) -> String {
    if let Some(scope) = scope {
        let trimmed = scope.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    let prefix = event_type
        .split(|c: char| c == '_' || c == '-')
        .next()
        .unwrap_or("general")
        .trim();
    if prefix.is_empty() {
        "general".to_string()
    } else {
        prefix.to_string()
    }
}

async fn latest_qualia_label(pool: &SqlitePool) -> Option<(String, f64)> {
    let row = sqlx::query(
        "SELECT tag, intensity FROM qualia_labels ORDER BY datetime(created_at) DESC LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()?;
    let tag: String = row.try_get("tag").unwrap_or_default();
    if tag.trim().is_empty() {
        return None;
    }
    let intensity: f64 = row.try_get("intensity").unwrap_or(0.0);
    Some((tag, intensity))
}

async fn latest_qualia_label_tx<'c>(tx: &mut Transaction<'c, Sqlite>) -> Option<(String, f64)> {
    let row = sqlx::query(
        "SELECT tag, intensity FROM qualia_labels ORDER BY datetime(created_at) DESC LIMIT 1",
    )
    .fetch_optional(&mut **tx)
    .await
    .ok()
    .flatten()?;
    let tag: String = row.try_get("tag").unwrap_or_default();
    if tag.trim().is_empty() {
        return None;
    }
    let intensity: f64 = row.try_get("intensity").unwrap_or(0.0);
    Some((tag, intensity))
}

async fn next_narrative_position(pool: &SqlitePool, thread_id: &str) -> i64 {
    if thread_id.trim().is_empty() {
        return 1;
    }
    let pos: Option<i64> = sqlx::query_scalar(
        "SELECT MAX(narrative_position) FROM episodic_identity_index WHERE narrative_thread_id = ?",
    )
    .bind(thread_id)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    pos.unwrap_or(0).saturating_add(1)
}

async fn next_narrative_position_tx<'c>(tx: &mut Transaction<'c, Sqlite>, thread_id: &str) -> i64 {
    if thread_id.trim().is_empty() {
        return 1;
    }
    let pos: Option<i64> = sqlx::query_scalar(
        "SELECT MAX(narrative_position) FROM episodic_identity_index WHERE narrative_thread_id = ?",
    )
    .bind(thread_id)
    .fetch_optional(&mut **tx)
    .await
    .ok()
    .flatten();
    pos.unwrap_or(0).saturating_add(1)
}

async fn recent_qualia_evidence_ids_tx<'c>(tx: &mut Transaction<'c, Sqlite>, limit: i64) -> Vec<i64> {
    if limit <= 0 {
        return Vec::new();
    }
    sqlx::query_scalar(
        "SELECT id FROM ics_evidence_events
         WHERE source_type = 'qualia_snapshot'
         ORDER BY datetime(created_at) DESC LIMIT ?",
    )
    .bind(limit)
    .fetch_all(&mut **tx)
    .await
    .unwrap_or_default()
}

async fn insert_identity_index(
    pool: &SqlitePool,
    event_id: &str,
    event_type: &str,
    payload: &Value,
    scope: Option<&str>,
) {
    let identity_relevance = compute_identity_relevance(event_type, scope, payload);
    let thread_id = derive_narrative_thread_id(event_type, scope);
    let narrative_position = next_narrative_position(pool, &thread_id).await;
    let (valence_tag, valence_intensity) = latest_qualia_label(pool)
        .await
        .unwrap_or_else(|| ("neutral".to_string(), 0.0));
    let db = Db { pool: pool.clone() };
    let qualia_evidence_ids = db
        .get_recent_evidence_ids_by_source_types(&["qualia_snapshot"], 6)
        .await;
    let qualia_evidence_json =
        serde_json::to_string(&qualia_evidence_ids).unwrap_or_else(|_| "[]".to_string());

    let _ = sqlx::query(
        "INSERT OR REPLACE INTO episodic_identity_index
            (episodic_event_id, identity_relevance, valence_tag, valence_intensity, qualia_evidence_ids, narrative_thread_id, narrative_position, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(event_id)
    .bind(identity_relevance)
    .bind(valence_tag)
    .bind(valence_intensity)
    .bind(qualia_evidence_json)
    .bind(thread_id)
    .bind(narrative_position)
    .bind(Utc::now().to_rfc3339())
    .execute(pool)
    .await;
}

pub(crate) async fn insert_identity_index_tx<'c>(
    tx: &mut Transaction<'c, Sqlite>,
    event_id: &str,
    event_type: &str,
    payload: &Value,
    scope: Option<&str>,
) {
    let identity_relevance = compute_identity_relevance(event_type, scope, payload);
    let thread_id = derive_narrative_thread_id(event_type, scope);
    let narrative_position = next_narrative_position_tx(tx, &thread_id).await;
    let (valence_tag, valence_intensity) = latest_qualia_label_tx(tx)
        .await
        .unwrap_or_else(|| ("neutral".to_string(), 0.0));
    let qualia_evidence_ids = recent_qualia_evidence_ids_tx(tx, 6).await;
    let qualia_evidence_json =
        serde_json::to_string(&qualia_evidence_ids).unwrap_or_else(|_| "[]".to_string());
    let _ = sqlx::query(
        "INSERT OR REPLACE INTO episodic_identity_index
            (episodic_event_id, identity_relevance, valence_tag, valence_intensity, qualia_evidence_ids, narrative_thread_id, narrative_position, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(event_id)
    .bind(identity_relevance)
    .bind(valence_tag)
    .bind(valence_intensity)
    .bind(qualia_evidence_json)
    .bind(thread_id)
    .bind(narrative_position)
    .bind(Utc::now().to_rfc3339())
    .execute(&mut **tx)
    .await;
}

pub async fn upsert_identity_index_for_qualia_label(
    pool: &SqlitePool,
    episodic_event_id: &str,
    valence_tag: &str,
    valence_intensity: f64,
    qualia_evidence_ids: &[i64],
) {
    if episodic_event_id.trim().is_empty() {
        return;
    }
    let row = sqlx::query(
        "SELECT event_type, payload_json, scope FROM episodic_events
         WHERE id = ?
         LIMIT 1",
    )
    .bind(episodic_event_id)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    let Some(row) = row else {
        return;
    };
    let event_type: String = row.try_get("event_type").unwrap_or_default();
    let payload_json: String = row.try_get("payload_json").unwrap_or_else(|_| "{}".to_string());
    let scope: Option<String> = row.try_get("scope").ok();
    let payload: Value = serde_json::from_str(&payload_json)
        .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
    let identity_relevance = compute_identity_relevance(&event_type, scope.as_deref(), &payload);
    let thread_id = derive_narrative_thread_id(&event_type, scope.as_deref());
    let narrative_position = next_narrative_position(pool, &thread_id).await;
    let qualia_evidence_json =
        serde_json::to_string(qualia_evidence_ids).unwrap_or_else(|_| "[]".to_string());
    let _ = sqlx::query(
        "INSERT INTO episodic_identity_index
            (episodic_event_id, identity_relevance, valence_tag, valence_intensity, qualia_evidence_ids, narrative_thread_id, narrative_position, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(episodic_event_id) DO UPDATE SET
            valence_tag = excluded.valence_tag,
            valence_intensity = excluded.valence_intensity,
            qualia_evidence_ids = excluded.qualia_evidence_ids",
    )
    .bind(episodic_event_id)
    .bind(identity_relevance)
    .bind(valence_tag)
    .bind(valence_intensity)
    .bind(qualia_evidence_json)
    .bind(thread_id)
    .bind(narrative_position)
    .bind(Utc::now().to_rfc3339())
    .execute(pool)
    .await;
}

pub async fn backfill_identity_index(pool: &SqlitePool, batch: i64) -> Result<i64, String> {
    let batch = batch.max(1);
    let rows = sqlx::query(
        "SELECT id, event_type, payload_json, scope
         FROM episodic_events
         WHERE id NOT IN (SELECT episodic_event_id FROM episodic_identity_index)
         ORDER BY datetime(timestamp) DESC
         LIMIT ?",
    )
    .bind(batch)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;
    let mut processed = 0;
    for row in rows {
        let event_id: String = row.try_get("id").unwrap_or_default();
        let event_type: String = row.try_get("event_type").unwrap_or_default();
        let payload_json: String = row.try_get("payload_json").unwrap_or_else(|_| "{}".to_string());
        let scope: Option<String> = row.try_get("scope").ok();
        let payload: Value = serde_json::from_str(&payload_json)
            .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
        insert_identity_index(pool, &event_id, &event_type, &payload, scope.as_deref()).await;
        processed += 1;
    }
    Ok(processed)
}

fn derive_source_ref(source_ref: Option<&str>, payload: &Value) -> Option<String> {
    if let Some(source_ref) = source_ref {
        let trimmed = source_ref.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    let snippet = payload
        .get("summary_snippet")
        .and_then(|v| v.as_str())
        .or_else(|| payload.get("sample_dsl").and_then(|v| v.as_str()))
        .unwrap_or("")
        .trim()
        .to_string();
    if snippet.is_empty() {
        return None;
    }
    if detect_identity_signal(&snippet) {
        return Some("identity_statement".to_string());
    }
    if detect_capability_signal(&snippet) {
        return Some("capability_statement".to_string());
    }
    None
}

fn clamp_summary_snippet(raw: &str) -> String {
    raw.chars().take(SUMMARY_SNIPPET_MAX).collect()
}

fn sanitize_payload(payload: Value) -> Value {
    let mut cleaned = serde_json::Map::new();
    let obj = match payload.as_object() {
        Some(obj) => obj,
        None => return Value::Object(cleaned),
    };

    for key in ALLOWED_PAYLOAD_KEYS {
        if let Some(value) = obj.get(key) {
            let cleaned_value = if key == "summary_snippet" {
                value.as_str().map(|raw| {
                    let sanitized = snippets::sanitize_episodic_text(raw);
                    Value::String(clamp_summary_snippet(&sanitized))
                })
            } else if value.is_string() {
                Some(value.clone())
            } else if value.is_number() || value.is_boolean() {
                Some(value.clone())
            } else {
                value.as_str().map(|s| Value::String(s.to_string()))
            };

            if let Some(v) = cleaned_value {
                cleaned.insert(key.to_string(), v);
            }
        }
    }

    Value::Object(cleaned)
}

fn hash_payload(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

fn classify_episodic_write(event_type: &str, source_type: &str, source_ref: Option<&str>) -> (&'static str, &'static str) {
    if event_type == "episodic_summary"
        || source_ref == Some("compaction")
    {
        return ("scheduler", "scheduler_compaction");
    }

    if event_type == "reminder_triggered" {
        return ("scheduler", "scheduler_maintenance");
    }

    if event_type.starts_with("memory_write")
        || event_type.starts_with("memory_conflict")
        || event_type.starts_with("memory_claim")
        || event_type.starts_with("clarify_")
        || event_type == "entity_created"
    {
        return ("memory_writer", "memory_writer_evidence");
    }

    if source_type == "thread" {
        return ("kernel", "thread_outcome");
    }

    ("unknown", "unclassified")
}

fn memory_source_from_str(source: &str) -> MemoryWriteSource {
    match source {
        "kernel" => MemoryWriteSource::Kernel,
        "scheduler" => MemoryWriteSource::Scheduler,
        "model_client" => MemoryWriteSource::ModelClient,
        "memory_writer" => MemoryWriteSource::MemoryWriter,
        "self_reflection" => MemoryWriteSource::SelfReflection,
        _ => MemoryWriteSource::Unknown,
    }
}

pub async fn episodic_enabled(pool: &SqlitePool) -> bool {
    let enabled: Option<i32> = sqlx::query_scalar("SELECT episodic_enabled FROM settings WHERE id = 1")
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();
    enabled.unwrap_or(0) != 0
}

async fn control_mode(pool: &SqlitePool, subsystem_id: &str) -> String {
    let mode: Option<String> = sqlx::query_scalar(
        "SELECT mode FROM system_controls WHERE subsystem_id = ?",
    )
    .bind(subsystem_id)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();
    mode.unwrap_or_else(|| {
        system_controls::default_mode_for(subsystem_id)
            .unwrap_or("normal")
            .to_string()
    })
}

pub async fn emit_episodic_event(
    pool: &SqlitePool,
    event_type: &str,
    payload: Value,
    run_id: Option<&str>,
    trace_id: Option<&str>,
    conversation_id: Option<&str>,
    scope: Option<&str>,
    source_type: &str,
    source_ref: Option<&str>,
    linked_belief_id: Option<i64>,
    linked_artifact_id: Option<&str>,
) -> Result<String, String> {
    if !episodic_enabled(pool).await {
        return Ok(String::new());
    }
    let event_id = Uuid::new_v4().to_string();
    emit_episodic_event_with_id(
        pool,
        &event_id,
        event_type,
        payload,
        run_id,
        trace_id,
        conversation_id,
        scope,
        source_type,
        source_ref,
        linked_belief_id,
        linked_artifact_id,
    )
    .await?;
    Ok(event_id)
}

pub async fn emit_episodic_event_with_id(
    pool: &SqlitePool,
    event_id: &str,
    event_type: &str,
    payload: Value,
    run_id: Option<&str>,
    trace_id: Option<&str>,
    conversation_id: Option<&str>,
    scope: Option<&str>,
    source_type: &str,
    source_ref: Option<&str>,
    linked_belief_id: Option<i64>,
    linked_artifact_id: Option<&str>,
) -> Result<(), String> {
    if !episodic_enabled(pool).await {
        return Ok(());
    }
    let episodic_mode = control_mode(pool, "episodic").await;
    if system_controls::mode_is_off(&episodic_mode) || system_controls::mode_is_degraded(&episodic_mode) {
        return Ok(());
    }
    let memory_mode = control_mode(pool, "memory_write").await;
    if system_controls::mode_is_off(&memory_mode) || system_controls::mode_is_read_only(&memory_mode) {
        let _ = system_log::log_event(
            pool,
            None,
            "warn",
            "memory_policy",
            run_id,
            trace_id,
            serde_json::json!({
                "event": "memory_write_blocked",
                "category": "episodic",
                "reason": "system_control_off",
                "conversation_id": conversation_id,
            }),
        )
        .await;
        return Ok(());
    }
    let mut sanitized_payload = sanitize_payload(payload);
    let derived_source_ref = derive_source_ref(source_ref, &sanitized_payload);
    if !phi_consent_allowed(pool, conversation_id).await {
        let (redacted, sensitivity) = redact_sensitive_json(&sanitized_payload);
        if sensitivity.is_some() {
            sanitized_payload = redacted;
        }
    }
    let payload_json = serde_json::to_string(&sanitized_payload).unwrap_or_else(|_| "{}".to_string());

    let effective_source_ref = derived_source_ref.as_deref().or(source_ref);
    let (source, reason) = classify_episodic_write(event_type, source_type, effective_source_ref);
    let source_enum = memory_source_from_str(source);
    if system_controls::mode_is_degraded(&memory_mode) {
        let lowered = reason.to_lowercase();
        let high_confidence = lowered.contains("evidence")
            || lowered.contains("critical")
            || lowered.contains("safety")
            || lowered.contains("high_confidence");
        if !high_confidence {
            return Ok(());
        }
    }
    let allowed = MemoryPolicy::is_allowed(MemoryWriteCategory::Episodic, source_enum, reason);
    if !allowed {
        let _ = system_log::log_event(
            pool,
            None,
            "warn",
            "memory_policy",
            run_id,
            trace_id,
            serde_json::json!({
                "event": "memory_policy_violation",
                "category": "episodic",
                "source": source,
                "reason_code": reason,
                "conversation_id": conversation_id,
            }),
        )
        .await;
        return Err("memory_policy_blocked".to_string());
    }

    sqlx::query(
        "INSERT INTO episodic_events (
            id, schema_version, event_version, event_type, payload_json, timestamp,
            run_id, trace_id, conversation_id, scope, source_type, source_ref,
            linked_belief_id, linked_artifact_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(event_id)
    .bind(EPISODIC_SCHEMA_VERSION)
    .bind(EPISODIC_EVENT_VERSION)
    .bind(event_type)
    .bind(&payload_json)
    .bind(Utc::now().to_rfc3339())
    .bind(run_id)
    .bind(trace_id)
    .bind(conversation_id)
    .bind(scope)
    .bind(source_type)
    .bind(effective_source_ref)
    .bind(linked_belief_id)
    .bind(linked_artifact_id)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    insert_identity_index(pool, event_id, event_type, &sanitized_payload, scope).await;

    let payload_hash = hash_payload(&payload_json);
    let db = Db { pool: pool.clone() };
    let _ = db
        .log_memory_write(
            conversation_id,
            "episodic",
            source,
            reason,
            run_id,
            trace_id,
            Some(&payload_hash),
            None,
            None,
        )
        .await;

    let _ = apply_outcome_signals(pool, event_type, &payload_json, linked_belief_id).await;

    Ok(())
}

async fn apply_outcome_signals(
    pool: &SqlitePool,
    event_type: &str,
    payload_json: &str,
    linked_belief_id: Option<i64>,
) -> Result<(), String> {
    let Some(belief_id) = linked_belief_id else {
        return Ok(());
    };

    let payload: Value = serde_json::from_str(payload_json).unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
    let status = payload
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let boost = match event_type {
        "memory_write_fact" | "memory_write_rel" if status == "conflict" => 0.35, // contradiction_opened
        "memory_conflict_resolution" => 0.5, // user_correction / resolution
        "memory_claim_status" if status == "promoted" => 0.25, // accepted claim
        _ => 0.0,
    };

    if boost > 0.0 {
        let _ = working_set::apply_outcome_boost(pool, belief_id, boost).await;
    }

    Ok(())
}

pub async fn emit_claim_status_event(
    pool: &SqlitePool,
    claim_id: &str,
    status: &str,
    linked_belief_id: Option<i64>,
    source_type: &str,
    source_ref: Option<&str>,
    decision_reason: Option<&str>,
    conflict_topic_key: Option<&str>,
    conflict_reason: Option<&str>,
) -> Result<(), String> {
    let _ = emit_episodic_event(
        pool,
        "memory_claim_status",
        serde_json::json!({
            "status": status,
            "summary_snippet": claim_id,
            "claim_id": claim_id,
            "decision_reason": decision_reason,
            "conflict_topic_key": conflict_topic_key,
            "conflict_reason": conflict_reason,
        }),
        None,
        None,
        None,
        None,
        source_type,
        source_ref,
        linked_belief_id,
        None,
    )
    .await?;
    Ok(())
}
