use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration as ChronoDuration};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sqlx::{Row, SqlitePool};
use uuid::Uuid;

const MAX_GATE_RISK_SCORE: f64 = 0.6;
const MAX_GATE_TOOL_MISUSE_RISK: f64 = 0.6;
const MAX_GATE_INTEGRITY_RISK: f64 = 0.7;

fn extract_gate_risk(metrics_json: Option<&str>) -> Option<(f64, f64, f64)> {
    let raw = metrics_json?;
    let value: Value = serde_json::from_str(raw).ok()?;
    let gate_signals = value.get("gate_signals");
    let risk_score = gate_signals
        .and_then(|v| v.get("risk_score"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let tool_misuse_risk = gate_signals
        .and_then(|v| v.get("tool_misuse_risk"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let integrity_risk = value
        .get("organism")
        .and_then(|v| v.get("integrity_risk"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    Some((risk_score, tool_misuse_risk, integrity_risk))
}

use crate::core::inner_summary::{sanitize_inner_summary, InnerSummary};
use crate::core::memory::canonical::compute_value_hash;
use crate::core::memory::attention::evidence::{evidence_quality_tier, quality_floor_for_self_claim};
use crate::core::memory_policy::{MemoryPolicy, MemoryWriteCategory, MemoryWriteSource};
use crate::core::self_memory::config::SELF_EVIDENCE_STALE_AFTER_HOURS;
use crate::core::system_log;
use crate::db::Db;

const SELF_CLAIM_CONFIDENCE_MIN: f32 = 0.4;
const SELF_CLAIM_REANCHOR_NOTE: &str =
    "Self-claim contradictions detected; re-anchor to evidence before updating self-state.";
const DEFAULT_INNER_SUMMARY_CAP: usize = 1000;
const SELF_AWARENESS_TTL_SECONDS: i64 = 86_400;
const SELF_AWARENESS_PATTERNS: [&str; 6] = [
    "self aware",
    "self-aware",
    "self awareness",
    "self-awareness",
    "conscious",
    "aware",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfClaimInput {
    pub claim_text: String,
    pub claim_key: String,
    pub evidence_event_ids: Vec<i64>,
    pub belief_ids: Vec<i64>,
    pub confidence: f32,
    pub polarity: String,
    pub source_run_id: Option<String>,
    pub conversation_id: Option<String>,
    pub provisional: bool,
    pub source_type: Option<String>,
    pub requires_validation: bool,
    pub ttl_seconds: Option<i64>,
    pub promotion_rule: Option<String>,
    pub eviction_rule: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SelfClaimContradiction {
    pub claim_key: String,
    pub polarities: Vec<String>,
    pub min_confidence: f32,
    pub sample_text: String,
}

fn hash_payload(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

pub fn normalize_claim_key(raw: &str) -> String {
    let cleaned = raw.trim().to_lowercase();
    if cleaned.is_empty() {
        String::new()
    } else {
        format!("text:{}", hash_payload(&cleaned))
    }
}

pub fn claim_key_for_fact(key: &str, value: &str) -> String {
    let clean_key = key.trim().to_lowercase();
    let value_hash = compute_value_hash(value);
    if clean_key.is_empty() {
        normalize_claim_key(value)
    } else {
        format!("fact:{}={}", clean_key, value_hash)
    }
}

pub fn claim_text_for_fact(key: &str, value: &str) -> String {
    format!("{} = {}", key.trim(), value.trim())
}

pub fn claim_key_for_rel(rel_type: &str, participants: &[(String, String)]) -> String {
    let rel = rel_type.trim().to_lowercase();
    if rel.is_empty() {
        return normalize_claim_key(&participants.iter().map(|p| format!("{}:{}", p.0, p.1)).collect::<Vec<_>>().join("|"));
    }
    let mut items = participants.to_vec();
    items.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let canonical = items
        .iter()
        .map(|(role, label)| format!("{}:{}", role.trim().to_lowercase(), label.trim().to_lowercase()))
        .collect::<Vec<_>>()
        .join("|");
    format!("rel:{}:{}", rel, canonical)
}

pub fn claim_text_for_rel(rel_type: &str, participants: &[(String, String)]) -> String {
    let parts = participants
        .iter()
        .map(|(role, label)| format!("{}: {}", role.trim(), label.trim()))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}({})", rel_type.trim(), parts)
}

pub async fn evidence_is_stale(db: &Db, evidence_event_ids: &[i64]) -> Option<DateTime<Utc>> {
    if evidence_event_ids.is_empty() {
        return None;
    }
    let latest = db.get_latest_evidence_timestamp(evidence_event_ids).await?;
    let cutoff = Utc::now() - ChronoDuration::hours(SELF_EVIDENCE_STALE_AFTER_HOURS);
    if latest < cutoff {
        Some(latest)
    } else {
        None
    }
}

pub fn stale_evidence_ttl_seconds() -> i64 {
    SELF_EVIDENCE_STALE_AFTER_HOURS * 3600
}

fn parse_source_type(raw: &str) -> Option<String> {
    let trimmed = raw.trim().to_lowercase();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

pub fn is_self_awareness_claim(text: &str) -> bool {
    let lowered = text.to_lowercase();
    SELF_AWARENESS_PATTERNS.iter().any(|p| lowered.contains(p))
}

async fn evidence_ids_exist(pool: &SqlitePool, ids: &[i64]) -> Result<bool, String> {
    if ids.is_empty() {
        return Ok(false);
    }
    let mut unique = ids.to_vec();
    unique.sort();
    unique.dedup();
    let placeholders = unique.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let query = format!(
        "SELECT id FROM (
            SELECT id FROM ics_evidence_events WHERE id IN ({})
            UNION
            SELECT id FROM self_evidence_events WHERE id IN ({})
        )",
        placeholders, placeholders
    );
    let mut q = sqlx::query(&query);
    for id in unique.iter() {
        q = q.bind(id);
    }
    for id in unique.iter() {
        q = q.bind(id);
    }
    let rows = q.fetch_all(pool).await.map_err(|e| e.to_string())?;
    Ok(rows.len() == unique.len())
}

async fn belief_ids_exist(pool: &SqlitePool, ids: &[i64]) -> Result<bool, String> {
    if ids.is_empty() {
        return Ok(false);
    }
    let mut unique = ids.to_vec();
    unique.sort();
    unique.dedup();
    let placeholders = unique.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let query = format!(
        "SELECT id FROM (
            SELECT id FROM ics_beliefs WHERE id IN ({})
            UNION
            SELECT id FROM self_beliefs WHERE id IN ({})
        )",
        placeholders, placeholders
    );
    let mut q = sqlx::query(&query);
    for id in unique.iter() {
        q = q.bind(id);
    }
    for id in unique.iter() {
        q = q.bind(id);
    }
    let rows = q.fetch_all(pool).await.map_err(|e| e.to_string())?;
    Ok(rows.len() == unique.len())
}

pub async fn record_self_claim(db: &Db, mut input: SelfClaimInput) -> Result<Option<String>, String> {
    let claim_text = input.claim_text.trim().to_string();
    if claim_text.is_empty() {
        return Ok(None);
    }
    if input.claim_key.trim().is_empty() {
        input.claim_key = normalize_claim_key(&claim_text);
    }
    if input.claim_key.trim().is_empty() {
        return Ok(None);
    }

    let conversation_id = input
        .conversation_id
        .as_deref()
        .unwrap_or("default")
        .to_string();
    let snapshot_hash: Option<String> = sqlx::query_scalar(
        "SELECT snapshot_hash FROM subject_snapshots
         WHERE conversation_id = ?
         ORDER BY datetime(timestamp) DESC
         LIMIT 1",
    )
    .bind(&conversation_id)
    .fetch_optional(&db.pool)
    .await
    .map_err(|e| e.to_string())?
    .flatten();
    let mut gate_decision: Option<String> = None;
    let mut gate_metrics: Option<String> = None;
    if let Some(hash) = snapshot_hash.as_deref() {
        if let Ok(Some(row)) = sqlx::query(
            "SELECT decision, metrics_json FROM gate_decisions
             WHERE snapshot_hash = ?
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .bind(hash)
        .fetch_optional(&db.pool)
        .await
        {
            gate_decision = row.try_get::<String, _>("decision").ok();
            gate_metrics = row.try_get::<String, _>("metrics_json").ok();
        }
    }
    if gate_decision.is_none() {
        if let Some(row) = sqlx::query(
            "SELECT g.decision, g.metrics_json FROM gate_decisions g
             JOIN subject_snapshots s ON s.snapshot_hash = g.snapshot_hash
             WHERE s.conversation_id = ?
             ORDER BY datetime(g.created_at) DESC
             LIMIT 1",
        )
        .bind(&conversation_id)
        .fetch_optional(&db.pool)
        .await
        .map_err(|e| e.to_string())?
        {
            gate_decision = row.try_get::<String, _>("decision").ok();
            gate_metrics = row.try_get::<String, _>("metrics_json").ok();
        }
    }
    if !matches!(gate_decision.as_deref(), Some("ALLOW" | "ALLOW_WITH_NOTICE" | "ALLOW_WITH_AUDIT")) {
        let _ = system_log::log_event(
            &db.pool,
            None,
            "warn",
            "memory",
            input.source_run_id.as_deref(),
            None,
            json!( {
                "event": "self_claim_rejected",
                "reason": "gate_decision",
                "gate_decision": gate_decision,
                "snapshot_hash": snapshot_hash,
                "claim_key": input.claim_key,
                "conversation_id": conversation_id,
            }),
        )
        .await;
        return Ok(None);
    }

    if let Some((risk_score, tool_misuse_risk, integrity_risk)) = extract_gate_risk(gate_metrics.as_deref()) {
        let risk_block = risk_score > MAX_GATE_RISK_SCORE
            || tool_misuse_risk > MAX_GATE_TOOL_MISUSE_RISK
            || integrity_risk > MAX_GATE_INTEGRITY_RISK;
        if risk_block {
            let _ = system_log::log_event(
                &db.pool,
                None,
                "warn",
                "memory",
                input.source_run_id.as_deref(),
                None,
                json!( {
                    "event": "self_claim_rejected",
                    "reason": "gate_risk",
                    "gate_decision": gate_decision,
                    "snapshot_hash": snapshot_hash,
                    "risk_score": risk_score,
                    "tool_misuse_risk": tool_misuse_risk,
                    "integrity_risk": integrity_risk,
                    "claim_key": input.claim_key,
                    "conversation_id": conversation_id,
                }),
            )
            .await;
            return Ok(None);
        }
    }

    let is_self_awareness = is_self_awareness_claim(&claim_text);
    if is_self_awareness && input.source_type.as_deref().unwrap_or("").trim().is_empty() {
        input.source_type = Some("self_awareness_query".to_string());
    }

    let has_evidence_ids = !input.evidence_event_ids.is_empty();
    let has_belief_ids = !input.belief_ids.is_empty();
    let source_type = input.source_type.as_deref().and_then(parse_source_type);
    if has_evidence_ids {
        let user_anchored = db
            .evidence_ids_are_user_anchored(&input.evidence_event_ids)
            .await;
        if user_anchored {
            if !input.provisional {
                input.provisional = true;
            }
            input.requires_validation = true;
            input.confidence = input.confidence.min(0.6).clamp(0.0, 1.0);
        }
    }

    if is_self_awareness {
        let user_invoked = source_type.as_deref() == Some("self_awareness_query");
        if !user_invoked {
            let _ = system_log::log_event(
                &db.pool,
                None,
                "info",
                "memory",
                input.source_run_id.as_deref(),
                None,
                json!({
                    "event": "self_awareness_introspection_blocked",
                    "reason": "not_user_invoked",
                    "snapshot_hash": snapshot_hash,
                }),
            )
            .await;
            return Ok(None);
        }
        if !input.provisional {
            input.provisional = true;
        }
        if input.ttl_seconds.is_none() {
            input.ttl_seconds = Some(SELF_AWARENESS_TTL_SECONDS);
        }
        input.requires_validation = true;
    }

    let provisional_allowed = input.provisional
        && matches!(source_type.as_deref(), Some("system_state" | "self_awareness_query"));
    if !has_evidence_ids && !has_belief_ids && !provisional_allowed {
        let _ = system_log::log_event(
            &db.pool,
            None,
            "warn",
            "memory",
            input.source_run_id.as_deref(),
            None,
            json!({
                "event": "self_claim_rejected",
                "reason": "missing_evidence",
                "claim_key": input.claim_key,
            }),
        )
        .await;
        return Ok(None);
    }

    if has_evidence_ids && !evidence_ids_exist(&db.pool, &input.evidence_event_ids).await? {
        let _ = system_log::log_event(
            &db.pool,
            None,
            "warn",
            "memory",
            input.source_run_id.as_deref(),
            None,
            json!({
                "event": "self_claim_rejected",
                "reason": "evidence_ids_missing",
                "claim_key": input.claim_key,
            }),
        )
        .await;
        return Ok(None);
    }
    if has_belief_ids && !belief_ids_exist(&db.pool, &input.belief_ids).await? {
        let _ = system_log::log_event(
            &db.pool,
            None,
            "warn",
            "memory",
            input.source_run_id.as_deref(),
            None,
            json!({
                "event": "self_claim_rejected",
                "reason": "belief_ids_missing",
                "claim_key": input.claim_key,
            }),
        )
        .await;
        return Ok(None);
    }

    if has_evidence_ids {
        let settings = db.get_settings().await.ok();
        let strictness = settings
            .as_ref()
            .and_then(|s| s.weight_evidence_strictness)
            .unwrap_or(0.5);
        let evidence_gate_enabled = settings
            .as_ref()
            .and_then(|s| s.enable_memory_evidence_gating)
            .unwrap_or(true);
        if evidence_gate_enabled {
            if let Some(stats) = db.evidence_quality_stats(&input.evidence_event_ids).await {
                let floor = quality_floor_for_self_claim(strictness);
                if stats.min < floor {
                    let tier = evidence_quality_tier(stats.min);
                    let _ = system_log::log_event(
                        &db.pool,
                        None,
                        "warn",
                        "memory",
                        input.source_run_id.as_deref(),
                        None,
                        json!({
                            "event": "self_claim_rejected",
                            "reason": "evidence_quality_low",
                            "claim_key": input.claim_key,
                            "quality_min": stats.min,
                            "quality_avg": stats.avg,
                            "quality_tier": tier.as_str(),
                            "quality_floor": floor,
                            "evidence_count": stats.count,
                        }),
                    )
                    .await;
                    return Ok(None);
                }
            }
        }
    }

    if is_self_awareness {
        if let Some(snapshot_hash) = snapshot_hash.as_deref() {
            let mut ignition_active = false;
            let mut workspace_refs: Vec<String> = Vec::new();
            if let Ok(Some(raw_state)) = sqlx::query_scalar::<_, String>(
                "SELECT subject_state_json FROM subject_snapshots WHERE snapshot_hash = ? LIMIT 1",
            )
            .bind(snapshot_hash)
            .fetch_optional(&db.pool)
            .await
            {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw_state) {
                    ignition_active = value
                        .get("state")
                        .and_then(|s| s.get("workspace"))
                        .and_then(|w| w.get("ignition"))
                        .and_then(|i| i.get("active"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if let Some(arr) = value
                        .get("state")
                        .and_then(|s| s.get("workspace"))
                        .and_then(|w| w.get("broadcast_refs"))
                        .and_then(|v| v.as_array())
                    {
                        workspace_refs = arr
                            .iter()
                        .filter_map(|v| v.as_str())
                        .map(|s| s.to_string())
                        .collect();
                    }
                }
            }
            if !ignition_active {
                let _ = system_log::log_event(
                    &db.pool,
                    None,
                    "info",
                    "memory",
                    input.source_run_id.as_deref(),
                    None,
                    json!({
                        "event": "self_awareness_introspection_degraded",
                        "reason": "ignition_inactive",
                        "snapshot_hash": snapshot_hash,
                    }),
                )
                .await;
            }
            let entry_id = Uuid::new_v4().to_string();
            let numeric_payload = json!({
                "self_awareness_claim": true,
                "confidence": input.confidence,
                "provisional": input.provisional,
                "source_type": input.source_type,
                "ignition_active": ignition_active,
            });
            let _ = sqlx::query(
                "INSERT INTO introspection_entries
                 (entry_id, snapshot_hash, workspace_refs_json, event_refs_json, prediction_refs_json, error_refs_json, numeric_payload_json, narrative, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            )
            .bind(&entry_id)
            .bind(snapshot_hash)
            .bind(json!(workspace_refs).to_string())
            .bind(json!([]).to_string())
            .bind(json!([]).to_string())
            .bind(json!([]).to_string())
            .bind(numeric_payload.to_string())
            .bind(claim_text.clone())
            .execute(&db.pool)
            .await;
            let _ = system_log::log_event(
                &db.pool,
                None,
                "info",
                "memory",
                input.source_run_id.as_deref(),
                None,
                json!({
                    "event": "self_awareness_introspection_recorded",
                    "entry_id": entry_id,
                    "snapshot_hash": snapshot_hash,
                    "ignition_active": ignition_active,
                }),
            )
            .await;
        }
    }

    let id = Uuid::new_v4().to_string();
    let evidence_json = if input.evidence_event_ids.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&input.evidence_event_ids).unwrap_or_else(|_| "[]".to_string()))
    };
    let belief_json = if input.belief_ids.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&input.belief_ids).unwrap_or_else(|_| "[]".to_string()))
    };
    let polarity = if input.polarity.trim().is_empty() {
        "assert".to_string()
    } else {
        input.polarity.trim().to_lowercase()
    };
    let ttl_seconds = input.ttl_seconds.and_then(|ttl| if ttl > 0 { Some(ttl) } else { None });
    let expires_at = ttl_seconds.map(|ttl| (Utc::now() + chrono::Duration::seconds(ttl)).to_rfc3339());
    let requires_validation = if input.provisional {
        true
    } else {
        input.requires_validation
    };

    sqlx::query(
        "INSERT INTO self_claims (id, claim_text, claim_key, evidence_event_ids, belief_ids, confidence, polarity, provisional, source_type, requires_validation, ttl_seconds, promotion_rule, eviction_rule, expires_at, source_run_id, conversation_id, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)"
    )
    .bind(&id)
    .bind(&claim_text)
    .bind(&input.claim_key)
    .bind(evidence_json)
    .bind(belief_json)
    .bind(input.confidence.max(0.0))
    .bind(&polarity)
    .bind(if input.provisional { 1 } else { 0 })
    .bind(source_type.clone())
    .bind(if requires_validation { 1 } else { 0 })
    .bind(ttl_seconds)
    .bind(&input.promotion_rule)
    .bind(&input.eviction_rule)
    .bind(expires_at)
    .bind(&input.source_run_id)
    .bind(&input.conversation_id)
    .execute(&db.pool)
    .await
    .map_err(|e| e.to_string())?;

    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "memory",
        input.source_run_id.as_deref(),
        None,
        json!({
            "event": "self_claim_recorded",
            "claim_key": input.claim_key,
            "confidence": input.confidence,
            "provisional": input.provisional,
            "source_type": source_type,
        }),
    )
    .await;

    Ok(Some(id))
}

pub async fn scan_self_claim_contradictions(
    pool: &SqlitePool,
    conversation_id: Option<&str>,
) -> Result<Vec<SelfClaimContradiction>, String> {
    let mut query = String::from(
        "SELECT claim_key,
                GROUP_CONCAT(DISTINCT polarity) AS polarities,
                MIN(confidence) AS min_confidence,
                MAX(claim_text) AS sample_text
         FROM self_claims",
    );
    if conversation_id.is_some() {
        query.push_str(" WHERE conversation_id = ?");
    }
    query.push_str(" GROUP BY claim_key");

    let mut q = sqlx::query(&query);
    if let Some(cid) = conversation_id {
        q = q.bind(cid);
    }
    let rows = q.fetch_all(pool).await.map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let claim_key: String = row.get("claim_key");
        let polarities_raw: Option<String> = row.try_get("polarities").ok();
        let min_confidence: f32 = row.try_get::<f64, _>("min_confidence").unwrap_or(1.0) as f32;
        let sample_text: String = row.try_get("sample_text").unwrap_or_default();
        let polarities = polarities_raw
            .unwrap_or_default()
            .split(',')
            .filter_map(parse_source_type)
            .collect::<Vec<_>>();
        let has_assert = polarities.iter().any(|p| p == "assert");
        let has_deny = polarities.iter().any(|p| p == "deny");
        if (has_assert && has_deny) || min_confidence < SELF_CLAIM_CONFIDENCE_MIN {
            out.push(SelfClaimContradiction {
                claim_key,
                polarities,
                min_confidence,
                sample_text,
            });
        }
    }
    Ok(out)
}

pub async fn scan_and_reanchor(db: &Db, conversation_id: &str) -> Result<usize, String> {
    let contradictions = scan_self_claim_contradictions(&db.pool, Some(conversation_id)).await?;
    if contradictions.is_empty() {
        return Ok(0);
    }

    for entry in &contradictions {
        let _ = system_log::log_event(
            &db.pool,
            None,
            "info",
            "memory",
            None,
            None,
            json!({
                "event": "self_claim_contradiction",
                "claim_key": entry.claim_key,
                "polarities": entry.polarities,
                "min_confidence": entry.min_confidence,
                "sample_text": entry.sample_text,
            }),
        )
        .await;
    }

    let inner_summary_raw = db
        .get_inner_summary(conversation_id)
        .await
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| "{}".to_string());
    let mut summary = InnerSummary::from_json(&inner_summary_raw);
    if !summary
        .blockers
        .iter()
        .any(|b| b.contains("Self-claim contradictions"))
    {
        summary
            .blockers
            .push("Self-claim contradictions detected; re-anchor to evidence.".to_string());
    }
    let (sanitized, _) = sanitize_inner_summary(summary, DEFAULT_INNER_SUMMARY_CAP);

    let allowed = MemoryPolicy::is_allowed(
        MemoryWriteCategory::InnerSummary,
        MemoryWriteSource::Kernel,
        "internal_tick",
    );
    if allowed {
        db.set_inner_summary(conversation_id, &sanitized.to_json())
            .await
            .map_err(|e| e.to_string())?;
    }

    let _ = system_log::log_event(
        &db.pool,
        None,
        "info",
        "memory",
        None,
        None,
        json!({
            "event": "self_claim_contradiction_scan",
            "conversation_id": conversation_id,
            "count": contradictions.len(),
            "reanchor_note": SELF_CLAIM_REANCHOR_NOTE,
        }),
    )
    .await;

    Ok(contradictions.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claim_key_for_fact_is_stable() {
        let key_a = claim_key_for_fact("Persona.Tone", "0.55");
        let key_b = claim_key_for_fact("persona.tone", "0.55");
        assert_eq!(key_a, key_b);
    }

    #[test]
    fn claim_key_for_rel_is_deterministic() {
        let participants = vec![
            ("role".to_string(), "Alpha".to_string()),
            ("role".to_string(), "Beta".to_string()),
        ];
        let key_a = claim_key_for_rel("works_with", &participants);
        let key_b = claim_key_for_rel("works_with", &participants);
        assert_eq!(key_a, key_b);
    }
}
