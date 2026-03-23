use sqlx::{SqlitePool, Row, QueryBuilder};
use crate::core::memory::types::{Scope, QueryIntent, MemoryPacket, ScoredFact, ScoredRel, ScoredRelParticipant};
use crate::models::EpisodicEvent;
use crate::core::memory::anchors;
use crate::core::memory::activation::{self, ActivationConfig};
use crate::core::cognitive_wave;
use crate::core::memory::traversal::{self, TraversalConfig};
use crate::core::memory::selection;
use crate::core::sensitivity::phi_consent_allowed;
use crate::core::memory::config::{
    RELATION_SCORE_THRESHOLD,
    WORKING_SET_BELIEF_SCORE_BOOST,
    WORKING_SET_ENTITY_SCORE_BOOST,
    MAX_RECALLED_FACTS,
    MAX_RECALLED_RELS,
};
use std::collections::{HashMap, HashSet};
use chrono::{DateTime, Utc};
use serde_json::Value;

/// I3: Time bucket priority for recency comparison
/// Lower value = higher priority
fn time_bucket_priority(kind: &str, _value: Option<&str>, _now: DateTime<Utc>) -> i32 {
    match kind {
        "range" => 0,  // Highest priority (if contains now, else handle separately)
        "exact" => 1,
        "day" => 2,
        "atemporal" => 3,
        _ => 4,
    }
}

/// §3.1 Scope specificity for shadowing
/// Higher value = more specific = higher priority
fn scope_specificity(scope_raw: &str) -> i32 {
    crate::core::memory::scope::scope_specificity_from_raw(scope_raw)
}

fn session_id_from_scopes(scopes: &[Scope]) -> Option<&str> {
    for scope in scopes {
        if let Scope::Context(id) = scope {
            let trimmed = id.trim();
            if !trimmed.is_empty() {
                return Some(trimmed);
            }
        }
    }
    None
}

const RELATION_ANCHOR_BOOST: f32 = 0.35;
const ANCHOR_WEAK_THRESHOLD: f32 = 50.0;
const FALLBACK_ENTITY_LIMIT: i64 = 5;
const FALLBACK_FACT_LIMIT: i64 = 5;
const FALLBACK_REL_LIMIT: i64 = 3;
const ACTIVATION_SEED_LIMIT: usize = 8;
const RERANK_TOP_N: usize = 30;

fn anchor_weight_from_score(score: f32, reason: &str) -> f32 {
    let effective_score = if score < 0.0 { 0.0 } else { score };
    let base = 1.0 / (1.0 + effective_score.max(0.01));
    let mut weight = 0.75 + base;
    if reason.starts_with("rel_match:") {
        weight += RELATION_ANCHOR_BOOST;
    }
    weight.max(0.25).min(2.5)
}

fn looks_relation_query(query: &str) -> bool {
    let q = query.to_lowercase();
    let keywords = [
        "relationship",
        "relation",
        "related",
        "connect",
        "connection",
        "between",
        "family",
        "friend",
        "friends",
        "spouse",
        "husband",
        "wife",
        "parent",
        "child",
        "sibling",
        "brother",
        "sister",
        "works with",
        "works_with",
        "owner",
        "owns",
        "creator",
        "created",
        "employer",
        "employee",
    ];
    keywords.iter().any(|k| q.contains(k))
}

fn sort_participants_by_role_id(participants: &mut Vec<ScoredRelParticipant>) {
    participants.sort_by(|a, b| a.role.cmp(&b.role).then(a.entity_id.cmp(&b.entity_id)));
}

fn build_order_map(
    raw: &str,
    participants: &[ScoredRelParticipant],
) -> Option<HashMap<(String, i64), usize>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut order_map: HashMap<(String, i64), usize> = HashMap::new();
    let mut parsed_len = 0usize;
    for (idx, entry) in trimmed.split('|').enumerate() {
        let mut parts = entry.splitn(2, ':');
        let role = parts.next().unwrap_or("").trim();
        let id_str = parts.next().unwrap_or("").trim();
        if role.is_empty() || id_str.is_empty() {
            return None;
        }
        let id = id_str.parse::<i64>().ok()?;
        let key = (role.to_string(), id);
        if order_map.contains_key(&key) {
            return None;
        }
        order_map.insert(key, idx);
        parsed_len += 1;
    }

    if parsed_len == 0 || parsed_len != participants.len() {
        return None;
    }

    for p in participants {
        let key = (p.role.clone(), p.entity_id);
        if !order_map.contains_key(&key) {
            return None;
        }
    }

    Some(order_map)
}

fn looks_like_stack_trace(lower: &str) -> bool {
    if lower.contains("stack trace") || lower.contains("traceback") {
        return true;
    }
    if lower.contains("exception") && lower.contains('\n') {
        return true;
    }
    if lower.contains("error:") && lower.contains('\n') {
        return true;
    }
    if lower.contains("panicked at") {
        return true;
    }
    false
}

fn looks_like_large_quote(input: &str) -> bool {
    if input.len() < 200 {
        return false;
    }
    let mut quote_lines = 0usize;
    let mut line_count = 0usize;
    for line in input.lines() {
        line_count += 1;
        if line.trim_start().starts_with('>') {
            quote_lines += 1;
        }
    }
    if line_count >= 3 && quote_lines >= 2 {
        return true;
    }
    let quote_marks = input.matches('"').count();
    line_count >= 3 && quote_marks >= 6
}

fn is_personal_query(query: &str) -> bool {
    let q = query.trim();
    if q.is_empty() {
        return false;
    }

    let lower = q.to_lowercase();
    if lower.contains("```") {
        return false;
    }
    if looks_like_stack_trace(&lower) || looks_like_large_quote(q) {
        return false;
    }
    if lower.contains("$user") {
        return true;
    }

    let tokens: Vec<String> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    if tokens.is_empty() {
        return false;
    }

    let pronouns = ["my", "me", "mine", "i", "myself", "im"];
    let relationship_tokens = [
        "parent",
        "child",
        "daughter",
        "son",
        "spouse",
        "mother",
        "father",
        "sibling",
        "brother",
        "sister",
        "husband",
        "wife",
        "partner",
        "friend",
        "friends",
        "mom",
        "dad",
        "kids",
        "children",
        "family",
    ];
    let attribute_tokens = [
        "name",
        "full_name",
        "preferred_name",
        "display_name",
        "birthday",
        "birthdate",
        "born",
        "age",
        "location",
        "address",
        "phone",
        "email",
        "job",
        "title",
        "employer",
        "company",
        "work",
        "live",
        "lives",
        "living",
        "from",
        "city",
        "country",
    ];

    let has_pronoun = tokens.iter().any(|t| pronouns.contains(&t.as_str()));
    let has_relationship = tokens
        .iter()
        .any(|t| relationship_tokens.contains(&t.as_str()));
    let has_attribute = tokens.iter().any(|t| attribute_tokens.contains(&t.as_str()));

    let personal_signal = has_pronoun || has_relationship;
    let target_signal = has_relationship || has_attribute;
    personal_signal && target_signal
}

pub async fn render_autobiographical_context(
    pool: &SqlitePool,
    conversation_id: Option<&str>,
    limit: i64,
) -> String {
    let limit = limit.clamp(1, 12);
    let rows = if let Some(conversation_id) = conversation_id {
        sqlx::query(
            "SELECT e.event_type, e.payload_json, e.timestamp, e.scope,
                    i.identity_relevance, i.valence_tag, i.valence_intensity,
                    i.narrative_thread_id, i.narrative_position
             FROM episodic_identity_index i
             JOIN episodic_events e ON e.id = i.episodic_event_id
             WHERE e.conversation_id = ?
             ORDER BY i.identity_relevance DESC, datetime(e.timestamp) DESC, i.narrative_position DESC
             LIMIT ?",
        )
        .bind(conversation_id)
        .bind(limit)
        .fetch_all(pool)
        .await
        .unwrap_or_default()
    } else {
        sqlx::query(
            "SELECT e.event_type, e.payload_json, e.timestamp, e.scope,
                    i.identity_relevance, i.valence_tag, i.valence_intensity,
                    i.narrative_thread_id, i.narrative_position
             FROM episodic_identity_index i
             JOIN episodic_events e ON e.id = i.episodic_event_id
             ORDER BY i.identity_relevance DESC, datetime(e.timestamp) DESC, i.narrative_position DESC
             LIMIT ?",
        )
        .bind(limit)
        .fetch_all(pool)
        .await
        .unwrap_or_default()
    };

    if rows.is_empty() {
        return "None".to_string();
    }

    let mut lines = Vec::new();
    for row in rows {
        let event_type: String = row.try_get("event_type").unwrap_or_default();
        let payload_json: String = row.try_get("payload_json").unwrap_or_else(|_| "{}".to_string());
        let timestamp: String = row.try_get("timestamp").unwrap_or_default();
        let scope: Option<String> = row.try_get("scope").ok();
        let relevance: f64 = row.try_get("identity_relevance").unwrap_or(0.0);
        let valence_tag: Option<String> = row.try_get("valence_tag").ok();
        let valence_intensity: f64 = row.try_get("valence_intensity").unwrap_or(0.0);
        let thread_id: Option<String> = row.try_get("narrative_thread_id").ok();
        let position: Option<i64> = row.try_get("narrative_position").ok();

        let summary = serde_json::from_str::<Value>(&payload_json)
            .ok()
            .and_then(|value| {
                value
                    .get("summary_snippet")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| value.get("sample_dsl").and_then(|v| v.as_str()).map(|s| s.to_string()))
                    .or_else(|| value.get("status").and_then(|v| v.as_str()).map(|s| s.to_string()))
            })
            .unwrap_or_else(|| event_type.clone());
        let valence = valence_tag.unwrap_or_else(|| "neutral".to_string());
        let thread_label = thread_id
            .or_else(|| scope.clone())
            .unwrap_or_else(|| "general".to_string());
        let position_label = position.unwrap_or(0);
        let line = format!(
            "- [{}:{}] {} (relevance {:.2}; valence {} {:.2}; at {})",
            thread_label,
            position_label,
            summary,
            relevance,
            valence,
            valence_intensity,
            timestamp
        );
        lines.push(line);
    }

    format!("Autobiographical Context:\n{}", lines.join("\n"))
}

use std::sync::Arc;
use crate::core::model_client::ModelClient;
use crate::core::memory::debug::{RetrievalDebugLog, AnchorDebug, TraversalDebug, SelectionDebug, ShadowingDebug, FactRankingDebug, RankingComponents, ShadowedBelief};

use crate::core::memory::writer::EmbeddingConfig;

/// Retrieve relevant memory for a query (Spec §10.10)
pub async fn retrieve(query: &str, scopes: &[Scope], intent: QueryIntent, pool: &SqlitePool, model_client: Option<Arc<ModelClient>>, embedding_config: Option<&EmbeddingConfig>) -> Result<MemoryPacket, String> {
    retrieve_with_options(query, scopes, intent, pool, model_client, embedding_config, false).await
}

/// Retrieve with optional debug logging (§10.10 + §13)
pub async fn retrieve_with_options(query: &str, scopes: &[Scope], intent: QueryIntent, pool: &SqlitePool, model_client: Option<Arc<ModelClient>>, embedding_config: Option<&EmbeddingConfig>, debug: bool) -> Result<MemoryPacket, String> {
    let start_time = std::time::Instant::now();
    let session_id = session_id_from_scopes(scopes).unwrap_or("default");
    let (seed_personal_user, lexical_fallback_enabled, half_life_hours) = match sqlx::query(
        "SELECT seed_personal_user, lexical_fallback_enabled, memory_half_life_hours FROM settings WHERE id = 1"
    )
    .fetch_optional(pool)
    .await
    {
        Ok(Some(row)) => {
            let seed: i32 = row.try_get("seed_personal_user").unwrap_or(1);
            let lexical: i32 = row.try_get("lexical_fallback_enabled").unwrap_or(0);
            let half_life: f64 = row.try_get("memory_half_life_hours").unwrap_or(168.0);
            (seed != 0, lexical != 0, half_life)
        }
        _ => (true, true, 168.0),
    };
    // 1. Anchors
    let mut anchors_list = anchors::find_anchors(query, pool, model_client, embedding_config).await?;
    let session_user_id = get_session_user_entity_id(pool, session_id).await?;

    let mut best_anchor_score = anchors_list
        .iter()
        .map(|a| a.score)
        .fold(f32::INFINITY, f32::min);
    let mut anchors_weak = anchors_list.is_empty() || best_anchor_score > ANCHOR_WEAK_THRESHOLD;

    if lexical_fallback_enabled && anchors_weak {
        if let Ok(fallback_anchors) = anchors::find_anchors_lexical_fallback(
            query,
            pool,
            FALLBACK_ENTITY_LIMIT,
            FALLBACK_FACT_LIMIT,
            FALLBACK_REL_LIMIT,
        ).await {
            for anchor in fallback_anchors {
                match anchors_list.iter_mut().find(|a| a.entity_id == anchor.entity_id) {
                    Some(existing) => {
                        if anchor.score < existing.score {
                            *existing = anchor;
                        }
                    }
                    None => anchors_list.push(anchor),
                }
            }
        }
    }
    best_anchor_score = anchors_list
        .iter()
        .map(|a| a.score)
        .fold(f32::INFINITY, f32::min);
    anchors_weak = anchors_list.is_empty() || best_anchor_score > ANCHOR_WEAK_THRESHOLD;

    let mut anchor_entries: Vec<(i64, f32)> = anchors_list
        .iter()
        .map(|a| (a.entity_id, anchor_weight_from_score(a.score, &a.reason)))
        .collect();
    let mut anchor_ids: Vec<i64> = anchor_entries.iter().map(|(id, _)| *id).collect();

    let mut seeded_user_anchor: Option<i64> = None;
    let mut seeded_user_reason: Option<&'static str> = None;
    let mut fallback_anchor: Option<i64> = None;

    let personal_query = is_personal_query(query);
    let should_seed_user = seed_personal_user && (personal_query || anchors_weak);

    if should_seed_user {
        if let Some(user_id) = session_user_id {
            if !anchor_ids.contains(&user_id) {
                anchor_entries.push((user_id, 1.0));
                anchor_ids.push(user_id);
                seeded_user_anchor = Some(user_id);
                seeded_user_reason = Some(if personal_query {
                    "seed:personal_user"
                } else {
                    "seed:weak_anchors"
                });
            }
        }
    }

    if anchor_ids.is_empty() {
        if let Some(user_id) = session_user_id {
            anchor_entries.push((user_id, 1.0));
            anchor_ids.push(user_id);
            fallback_anchor = Some(user_id);
        }
    }
    
    // Debug: capture anchor info
    let mut anchor_debug: Vec<AnchorDebug> = if debug {
        let mut debug_items: Vec<AnchorDebug> = anchors_list.iter().map(|a| AnchorDebug {
            entity_id: a.entity_id,
            label: format!("entity_{}", a.entity_id), // Label not available in anchor, use ID
            source: a.reason.clone(), // Use 'reason' as source
            score: a.score,
        }).collect();
        if let Some(user_id) = seeded_user_anchor {
            debug_items.push(AnchorDebug {
                entity_id: user_id,
                label: format!("entity_{}", user_id),
                source: seeded_user_reason.unwrap_or("seed:personal_user").to_string(),
                score: 0.0,
            });
        }
        if let Some(user_id) = fallback_anchor {
            debug_items.push(AnchorDebug {
                entity_id: user_id,
                label: format!("entity_{}", user_id),
                source: "fallback:session_user".to_string(),
                score: 0.0,
            });
        }
        debug_items
    } else {
        vec![]
    };

    let should_activate = !anchor_entries.is_empty() && (anchors_weak || anchor_entries.len() > 1);
    if should_activate {
        let activation_config = ActivationConfig::default();
        let mut activation_seeds = anchor_entries.clone();
        activation_seeds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if activation_seeds.len() > ACTIVATION_SEED_LIMIT {
            activation_seeds.truncate(ACTIVATION_SEED_LIMIT);
        }
        if let Ok(activation_scores) = activation::activate_with_ppr(pool, &activation_seeds, &activation_config).await {
            if let Some(contribution) = activation::wave_contribution_from_activation(&activation_scores) {
                let _ = cognitive_wave::try_contribute(pool, None, &contribution, None, None).await;
            }
            for (entity_id, score) in activation_scores {
                let weight = score * activation_config.weight_scale;
                if weight <= 0.0 {
                    continue;
                }
                if let Some(entry) = anchor_entries.iter_mut().find(|(id, _)| *id == entity_id) {
                    entry.1 += weight;
                } else {
                    anchor_entries.push((entity_id, weight));
                    anchor_ids.push(entity_id);
                    if debug {
                        anchor_debug.push(AnchorDebug {
                            entity_id,
                            label: format!("entity_{}", entity_id),
                            source: "activation:ppr".to_string(),
                            score,
                        });
                    }
                }
            }
        }
    }
    
    // 2. Traversal
    let config = TraversalConfig::default();
    let traversal_res = traversal::bounded_traversal(&anchor_entries, &config, pool).await?;
    
    // 3. Fetch Facts for Entities
    let mut facts = vec![];
    let scope_strs: Vec<String> = scopes.iter().map(|s| serde_json::to_string(s).unwrap_or_default()).collect();
    let scope_placeholders: String = scope_strs.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let mut dropped_by_scope_count = 0usize;

    if !scope_strs.is_empty() {
        let entity_ids: Vec<i64> = traversal_res.entities.keys().copied().collect();
        if !entity_ids.is_empty() {
            let entity_placeholders: String = entity_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let query = format!(
                "SELECT COUNT(*) as count
                 FROM ics_beliefs b
                 JOIN ics_fact_beliefs fb ON fb.belief_id = b.id
                 WHERE fb.subject_entity_id IN ({})
                   AND b.status = 'active'
                   AND b.scope NOT IN ({})",
                entity_placeholders,
                scope_placeholders
            );
            let mut q = sqlx::query(&query);
            for id in &entity_ids {
                q = q.bind(id);
            }
            for s in &scope_strs {
                q = q.bind(s);
            }
            if let Ok(row) = q.fetch_one(pool).await {
                let count: i64 = row.try_get("count").unwrap_or(0);
                dropped_by_scope_count = count.max(0) as usize;
            }
        }
    }
    
    // Time decay config (half-life hours)
    let base_half_life = half_life_hours.max(1.0);
    let now = chrono::Utc::now();
    
    for (entity_id, assoc_weight) in traversal_res.entities.iter() {
        // Fetch facts with timestamp and new fields - WITH SCOPE FILTER (§3)
        let query_str = if scope_strs.is_empty() {
            // No scope filter if empty - return all scopes
            "SELECT b.id, b.evidence_weight_total, b.confidence, b.salience, b.layer, b.last_evidence_at, b.topic_key,
                    b.time_bucket_kind, b.time_bucket_value, b.signature_hash, b.polarity, b.scope,
                    b.observed_at,
                    fb.key, fb.value_literal, e.label, fb.subject_entity_id
             FROM ics_beliefs b
             JOIN ics_fact_beliefs fb ON fb.belief_id = b.id
             JOIN ics_entities e ON e.id = fb.subject_entity_id
             WHERE fb.subject_entity_id = ? AND b.status = 'active'".to_string()
        } else {
            format!(
                "SELECT b.id, b.evidence_weight_total, b.confidence, b.salience, b.layer, b.last_evidence_at, b.topic_key,
                        b.time_bucket_kind, b.time_bucket_value, b.signature_hash, b.polarity, b.scope,
                        b.observed_at,
                        fb.key, fb.value_literal, e.label, fb.subject_entity_id
                 FROM ics_beliefs b
                 JOIN ics_fact_beliefs fb ON fb.belief_id = b.id
                 JOIN ics_entities e ON e.id = fb.subject_entity_id
                 WHERE fb.subject_entity_id = ? AND b.status = 'active' AND b.scope IN ({})",
                scope_placeholders
            )
        };
        
        let mut q = sqlx::query(&query_str).bind(*entity_id);
        for s in &scope_strs {
            q = q.bind(s);
        }
        
        let rows = q.fetch_all(pool).await.map_err(|e| e.to_string())?;
        
        for row in rows {
            let last_evidence: String = row.get("last_evidence_at");
            let evidence_weight: f64 = row.get("evidence_weight_total");
            let confidence: f64 = row.get("confidence");
            let salience: f64 = row.try_get("salience").unwrap_or(1.0);
            let layer: String = row.try_get("layer").unwrap_or_else(|_| "episodic".to_string());
            let w_layer = layer_weight(&layer);
            
            // W_time calculation
            let w_time = if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(&last_evidence) {
                let age_hours = (now - ts.with_timezone(&chrono::Utc)).num_hours() as f64;
                decay_weight(age_hours, base_half_life, evidence_weight)
            } else {
                1.0
            };
            
            // W_assoc = assoc_weight (from traversal)
            let w_assoc = (*assoc_weight) as f64;
            
            // I4: Use compute_support formula: S(b) = ln(1 + W) * (0.5 + 0.5 * C)
            let i4_support = selection::compute_support(evidence_weight as f32, confidence as f32);
            
            // Combine I4 support with time, association, and salience
            let score = (i4_support as f64 * w_time * w_assoc * salience * w_layer) as f32;
            
             let observed_at: Option<String> = row.try_get("observed_at").ok();
             let observed_at_formatted = observed_at.as_ref().map(|ts| {
                 crate::core::memory::time_format::format_when_told(ts, now)
             });

             facts.push(ScoredFact {
                 id: row.get("id"),
                 entity_id: row.get("subject_entity_id"),
                 entity_label: row.get("label"),
                 key: row.get("key"),
                 topic_key: row.get("topic_key"),
                 value: row.get("value_literal"),
                 confidence: confidence as f32,
                 score,
                 time_bucket_kind: row.try_get("time_bucket_kind").unwrap_or("atemporal".to_string()),
                 time_bucket_value: row.try_get("time_bucket_value").ok(),
                 signature_hash: row.try_get("signature_hash").unwrap_or_default(),
                 polarity: row.try_get("polarity").unwrap_or("assert".to_string()),
                 scope: row.try_get("scope").unwrap_or("\"global\"".to_string()),
                 observed_at,
                 observed_at_formatted,
             });
        }
    }
    
    // 4. Fetch Rels
    let mut relations = vec![];
    let mut relation_topic_keys: HashMap<i64, String> = HashMap::new();
    for belief_id in traversal_res.beliefs {
        // Fetch rel details with new fields
        let query_str = if scope_strs.is_empty() {
            "SELECT b.id, b.evidence_weight_total, b.confidence, b.salience, b.layer, b.last_evidence_at,
                    b.time_bucket_kind, b.time_bucket_value, b.signature_hash, b.polarity, b.scope,
                    b.observed_at,
                    b.topic_key,
                    rb.rel_type, rb.direction, rb.participants_canonical, rb.participants_ordered
             FROM ics_beliefs b
             JOIN ics_rel_beliefs rb ON rb.belief_id = b.id
             WHERE b.id = ?"
                .to_string()
        } else {
            format!(
                "SELECT b.id, b.evidence_weight_total, b.confidence, b.salience, b.layer, b.last_evidence_at,
                        b.time_bucket_kind, b.time_bucket_value, b.signature_hash, b.polarity, b.scope,
                        b.observed_at,
                        b.topic_key,
                        rb.rel_type, rb.direction, rb.participants_canonical, rb.participants_ordered
                 FROM ics_beliefs b
                 JOIN ics_rel_beliefs rb ON rb.belief_id = b.id
                 WHERE b.id = ? AND b.scope IN ({})",
                scope_placeholders
            )
        };

        let mut q = sqlx::query(&query_str).bind(belief_id);
        for s in &scope_strs {
            q = q.bind(s);
        }

        let row = q
            .fetch_optional(pool)
            .await
            .map_err(|e| e.to_string())?;
        
        if let Some(r) = row {
            let p_rows = sqlx::query(
                "SELECT rp.role, rp.entity_id, e.label 
                 FROM ics_rel_participants rp
                 JOIN ics_entities e ON e.id = rp.entity_id
                 WHERE rp.belief_id = ?
                 ORDER BY rp.role ASC, rp.entity_id ASC"
            )
            .bind(belief_id)
            .fetch_all(pool)
            .await
            .map_err(|e| e.to_string())?;
            
            let mut participants: Vec<ScoredRelParticipant> = p_rows
                .into_iter()
                .map(|pr| ScoredRelParticipant {
                    role: pr.get("role"),
                    entity_id: pr.get("entity_id"),
                    entity_label: pr.get("label"),
                })
                .collect();

            let direction: Option<String> = r.try_get("direction").ok();
            let participants_ordered: Option<String> = r.try_get("participants_ordered").ok();
            let participants_canonical: Option<String> = r.try_get("participants_canonical").ok();

            let mut order_is_trusted = false;
            if direction.is_some() {
                if let Some(raw) = participants_ordered.as_deref() {
                    if let Some(order_map) = build_order_map(raw, &participants) {
                        order_is_trusted = true;
                        participants.sort_by(|a, b| {
                            let a_key = order_map
                                .get(&(a.role.clone(), a.entity_id))
                                .cloned()
                                .unwrap_or(usize::MAX);
                            let b_key = order_map
                                .get(&(b.role.clone(), b.entity_id))
                                .cloned()
                                .unwrap_or(usize::MAX);
                            a_key
                                .cmp(&b_key)
                                .then_with(|| a.role.cmp(&b.role))
                                .then_with(|| a.entity_id.cmp(&b.entity_id))
                        });
                    } else {
                        sort_participants_by_role_id(&mut participants);
                    }
                } else {
                    sort_participants_by_role_id(&mut participants);
                }
            } else {
                let preferred_order = if participants_ordered.as_deref().unwrap_or("").is_empty() {
                    None
                } else {
                    match (participants_ordered.as_deref(), participants_canonical.as_deref()) {
                        (Some(ordered), Some(canonical)) if ordered != canonical => Some(ordered),
                        (Some(ordered), None) => Some(ordered),
                        _ => participants_canonical.as_deref(),
                    }
                };

                if let Some(raw) = preferred_order {
                    if let Some(order_map) = build_order_map(raw, &participants) {
                        participants.sort_by(|a, b| {
                            let a_key = order_map
                                .get(&(a.role.clone(), a.entity_id))
                                .cloned()
                                .unwrap_or(usize::MAX);
                            let b_key = order_map
                                .get(&(b.role.clone(), b.entity_id))
                                .cloned()
                                .unwrap_or(usize::MAX);
                            a_key
                                .cmp(&b_key)
                                .then_with(|| a.role.cmp(&b.role))
                                .then_with(|| a.entity_id.cmp(&b.entity_id))
                        });
                    } else {
                        sort_participants_by_role_id(&mut participants);
                    }
                } else {
                    sort_participants_by_role_id(&mut participants);
                }
            }
            
            let last_evidence: String = r.get("last_evidence_at");
            let evidence_weight: f64 = r.get("evidence_weight_total");
            let confidence: f64 = r.get("confidence");
            let salience: f64 = r.try_get("salience").unwrap_or(1.0);
            let layer: String = r.try_get("layer").unwrap_or_else(|_| "episodic".to_string());
            let w_layer = layer_weight(&layer);
            
            let w_time = if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(&last_evidence) {
                let age_hours = (now - ts.with_timezone(&chrono::Utc)).num_hours() as f64;
                decay_weight(age_hours, base_half_life, evidence_weight)
            } else {
                1.0
            };

            let mut w_assoc = 1.0;
            if !participants.is_empty() {
                let mut max_assoc = 0.0f64;
                for participant in &participants {
                    if let Some(weight) = traversal_res.entities.get(&participant.entity_id) {
                        let weight = *weight as f64;
                        if weight > max_assoc {
                            max_assoc = weight;
                        }
                    }
                }
                if max_assoc > 0.0 {
                    w_assoc = max_assoc;
                }
            }
            
            let base_score = selection::compute_support(evidence_weight as f32, confidence as f32);
            let score = (base_score as f64 * w_time * salience * w_assoc * w_layer) as f32;
            
            let observed_at: Option<String> = r.try_get("observed_at").ok();
            let observed_at_formatted = observed_at.as_ref().map(|ts| {
                crate::core::memory::time_format::format_when_told(ts, now)
            });

            let topic_key: String = r.try_get("topic_key").unwrap_or_default();
            relations.push(ScoredRel {
                id: r.get("id"),
                rel_type: r.get("rel_type"),
                participants,
                direction,
                order_is_trusted,
                confidence: r.get("confidence"),
                score,
                time_bucket_kind: r.try_get("time_bucket_kind").unwrap_or("atemporal".to_string()),
                time_bucket_value: r.try_get("time_bucket_value").ok(),
                signature_hash: r.try_get("signature_hash").unwrap_or_default(),
                polarity: r.try_get("polarity").unwrap_or("assert".to_string()),
                scope: r.try_get("scope").unwrap_or("\"global\"".to_string()),
                observed_at,
                observed_at_formatted,
            });
            relation_topic_keys.insert(r.get("id"), topic_key);
        }
    }

    let _ = apply_working_set_boost(pool, &mut facts, &mut relations).await;

    if fallback_anchor.is_some() && !looks_relation_query(query) {
        relations.retain(|rel| rel.score >= RELATION_SCORE_THRESHOLD);
    }

    if matches!(intent, QueryIntent::AskCurrent) {
        let deny_topics: HashSet<String> = relations
            .iter()
            .filter(|rel| rel.polarity == "deny")
            .filter_map(|rel| relation_topic_keys.get(&rel.id).cloned())
            .collect();
        if !deny_topics.is_empty() {
            relations.retain(|rel| {
                if rel.polarity == "deny" {
                    return true;
                }
                match relation_topic_keys.get(&rel.id) {
                    Some(topic_key) => !deny_topics.contains(topic_key),
                    None => true,
                }
            });
        }
    }

    // 5. Fetch Conflicts (linked to retrieved beliefs)
    let mut conflicts = vec![];
    let mut belief_ids: Vec<i64> = facts.iter().map(|f| f.id).collect();
    belief_ids.extend(relations.iter().map(|r| r.id));

    if !belief_ids.is_empty() {
        let placeholders: String = belief_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let query = format!(
            "SELECT DISTINCT conflict_set_id FROM ics_conflict_set_members WHERE belief_id IN ({})",
            placeholders
        );
        let mut q = sqlx::query(&query);
        for id in &belief_ids {
            q = q.bind(id);
        }
        let conflict_set_rows = q
            .fetch_all(pool)
            .await
            .map_err(|e| e.to_string())?;

        let conflict_set_ids: Vec<i64> = conflict_set_rows
            .into_iter()
            .filter_map(|row| row.try_get("conflict_set_id").ok())
            .collect();

        if !conflict_set_ids.is_empty() {
            let placeholders: String = conflict_set_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let query = format!(
                "SELECT id, topic_key, status, priority, resolution_note, created_at, updated_at\n                 FROM ics_conflict_sets WHERE id IN ({})",
                placeholders
            );
            let mut q = sqlx::query(&query);
            for id in &conflict_set_ids {
                q = q.bind(id);
            }
            let conflict_rows = q
                .fetch_all(pool)
                .await
                .map_err(|e| e.to_string())?;

            for row in conflict_rows {
                conflicts.push(crate::core::memory::types::ConflictSet {
                    id: row.get("id"),
                    topic_key: row.get("topic_key"),
                    status: match row.get::<String, _>("status").as_str() {
                        "open" => crate::core::memory::types::ConflictStatus::Open,
                        "archived" => crate::core::memory::types::ConflictStatus::Archived,
                        _ => crate::core::memory::types::ConflictStatus::Resolved,
                    },
                    priority: row.get("priority"),
                    resolution_note: row.get("resolution_note"),
                    created_at: row.get("created_at"),
                    updated_at: row.get("updated_at"),
                });
            }
        }
    }

    // ===== Negation Semantics (§11) =====
    // For AskCurrent: deny beliefs suppress matching assert beliefs
    // For MANY_SET/TIME_SERIES: suppress asserts whose signature matches a deny
    let facts_before_filter = facts.len();
    {
        // Collect deny signatures
        let deny_sigs: HashSet<String> = facts.iter()
            .filter(|f| f.polarity == "deny")
            .map(|f| {
                // Signature for matching: topic_key (signature_hash differs by polarity)
                f.topic_key.clone()
            })
            .collect();
        
        // Suppress asserts that match deny topic_keys
        if !deny_sigs.is_empty() {
            facts.retain(|f| {
                // Keep denies, and keep asserts not suppressed
                f.polarity == "deny" || !deny_sigs.contains(&f.topic_key)
            });
        }
    }

    let facts_after_negation = facts.len();
    // ===== §3.1 Scope Shadowing =====
    let mut shadowed_details: Vec<ShadowedBelief> = Vec::new();
    // Group facts by topic_key, prefer most-specific scope
    let (topic_groups_count, shadowed_by_scope_count) = {
        // Group by topic_key
        let mut topic_groups: HashMap<String, Vec<&ScoredFact>> = HashMap::new();
        for fact in &facts {
            topic_groups.entry(fact.topic_key.clone()).or_default().push(fact);
        }
        let count = topic_groups.len();
        
        // For each group, find the most specific scope
        let mut shadowed_ids: HashSet<i64> = HashSet::new();
        for (_topic, group) in &topic_groups {
            if group.len() <= 1 {
                continue; // No shadowing needed
            }
            
            // Find max scope specificity in this group
            let max_specificity = group.iter()
                .map(|f| scope_specificity(&f.scope))
                .max()
                .unwrap_or(1);
            let shadowing_scope = group.iter()
                .find(|f| scope_specificity(&f.scope) == max_specificity)
                .map(|f| f.scope.clone())
                .unwrap_or_default();

            // Shadow (exclude) beliefs with lower specificity
            for fact in group {
                let spec = scope_specificity(&fact.scope);
                if spec < max_specificity {
                    shadowed_ids.insert(fact.id);
                    if debug {
                        shadowed_details.push(ShadowedBelief {
                            id: fact.id,
                            topic_key: fact.topic_key.clone(),
                            scope: fact.scope.clone(),
                            shadowed_by_scope: shadowing_scope.clone(),
                        });
                    }
                }
            }
        }
        
        // Remove shadowed facts
        if !shadowed_ids.is_empty() {
            facts.retain(|f| !shadowed_ids.contains(&f.id));
        }

        let shadowed_by_scope_count = shadowed_ids.len();
        (count, shadowed_by_scope_count)
    };

    let facts_after_shadowing = facts.len();
    // ===== Apply QueryIntent (§10.1) =====
    let now = chrono::Utc::now();
    match intent {
        QueryIntent::AskCurrent => {
            // I3: Sort by time bucket priority (lower = more recent)
            facts.sort_by(|a, b| {
                let a_prio = time_bucket_priority(&a.time_bucket_kind, a.time_bucket_value.as_deref(), now);
                let b_prio = time_bucket_priority(&b.time_bucket_kind, b.time_bucket_value.as_deref(), now);
                a_prio.cmp(&b_prio)
                    // I7: Tie-break by signature_hash ascending
                    .then_with(|| a.signature_hash.cmp(&b.signature_hash))
            });
            
            // For AskCurrent: keep only best per topic_key
            let mut seen_topics: HashSet<String> = HashSet::new();
            facts.retain(|f| {
                if seen_topics.contains(&f.topic_key) {
                    false
                } else {
                    seen_topics.insert(f.topic_key.clone());
                    true
                }
            });
        },
        QueryIntent::AskList => {
            // I7: Sort by score desc, then signature_hash for tie-break
            facts.sort_by(|a, b| {
                b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.signature_hash.cmp(&b.signature_hash))
            });
        },
        QueryIntent::AskHistory => {
            // AskHistory: Include inactive (superseded) beliefs AND follow supersession chains
            
            // 1. Collect entity IDs we already have facts for
            let _entity_ids: Vec<i64> = facts.iter()
                .map(|f| f.entity_id)
                .collect();
            
            // 2. Query inactive beliefs for same topics
            let topic_keys: Vec<String> = facts.iter().map(|f| f.topic_key.clone()).collect();
            if !topic_keys.is_empty() {
                let placeholders: String = topic_keys.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                let query = format!(
                    "SELECT b.id, b.evidence_weight_total, b.confidence, b.salience, b.layer, b.last_evidence_at, b.topic_key,
                            b.time_bucket_kind, b.time_bucket_value, b.signature_hash, b.polarity, b.scope,
                            b.observed_at,
                            fb.key, fb.value_literal, e.label, fb.subject_entity_id
                     FROM ics_beliefs b
                     JOIN ics_fact_beliefs fb ON fb.belief_id = b.id
                     JOIN ics_entities e ON e.id = fb.subject_entity_id
                     WHERE b.status = 'inactive' AND b.topic_key IN ({})",
                    placeholders
                );
                
                let mut q = sqlx::query(&query);
                for tk in &topic_keys {
                    q = q.bind(tk);
                }
                
                if let Ok(inactive_rows) = q.fetch_all(pool).await {
                    for row in inactive_rows {
                        let id: i64 = row.get("id");
                        // Skip if already in facts
                        if facts.iter().any(|f| f.id == id) {
                            continue;
                        }
                        
                        let evidence_weight: f64 = row.get("evidence_weight_total");
                        let confidence: f64 = row.get("confidence");
                        let salience: f64 = row.try_get("salience").unwrap_or(1.0);
                        let layer: String = row.try_get("layer").unwrap_or_else(|_| "episodic".to_string());
                        let w_layer = layer_weight(&layer);
                        
                        let observed_at: Option<String> = row.try_get("observed_at").ok();
                        let observed_at_formatted = observed_at.as_ref().map(|ts| {
                            crate::core::memory::time_format::format_when_told(ts, now)
                        });
                        
                        // Score with penalty for inactive
                        let base_score = selection::compute_support(evidence_weight as f32, confidence as f32);
                        let score = base_score * 0.5 * salience as f32 * w_layer as f32; // 50% penalty for inactive
                        
                        facts.push(ScoredFact {
                            id,
                            entity_id: row.get("subject_entity_id"),
                            entity_label: row.get("label"),
                            key: row.get("key"),
                            topic_key: row.get("topic_key"),
                            value: row.get("value_literal"),
                            confidence: confidence as f32,
                            score,
                            time_bucket_kind: row.try_get("time_bucket_kind").unwrap_or("atemporal".to_string()),
                            time_bucket_value: row.try_get("time_bucket_value").ok(),
                            signature_hash: row.try_get("signature_hash").unwrap_or_default(),
                            polarity: row.try_get("polarity").unwrap_or("assert".to_string()),
                            scope: row.try_get("scope").unwrap_or("\"global\"".to_string()),
                            observed_at,
                            observed_at_formatted,
                        });
                    }
                }
            }
            
            // 3. Follow supersedes links for existing facts
            let existing_ids: Vec<i64> = facts.iter().map(|f| f.id).collect();
            for id in &existing_ids {
                if let Ok(superseded_rows) = sqlx::query(
                    "SELECT b.id, b.evidence_weight_total, b.confidence, b.salience, b.layer, b.last_evidence_at, b.topic_key,
                            b.time_bucket_kind, b.time_bucket_value, b.signature_hash, b.polarity, b.scope,
                            b.observed_at,
                            fb.key, fb.value_literal, e.label, fb.subject_entity_id
                     FROM ics_belief_links bl
                     JOIN ics_beliefs b ON b.id = bl.to_id
                     JOIN ics_fact_beliefs fb ON fb.belief_id = b.id
                     JOIN ics_entities e ON e.id = fb.subject_entity_id
                     WHERE bl.from_id = ? AND bl.link_type = 'supersedes'"
                )
                .bind(id)
                .fetch_all(pool)
                .await {
                    for row in superseded_rows {
                        let superseded_id: i64 = row.get("id");
                        if facts.iter().any(|f| f.id == superseded_id) {
                            continue;
                        }
                        
                        let evidence_weight: f64 = row.get("evidence_weight_total");
                        let confidence: f64 = row.get("confidence");
                        
                        let observed_at: Option<String> = row.try_get("observed_at").ok();
                        let observed_at_formatted = observed_at.as_ref().map(|ts| {
                            crate::core::memory::time_format::format_when_told(ts, now)
                        });
                        
                        let layer: String = row.try_get("layer").unwrap_or_else(|_| "episodic".to_string());
                        let w_layer = layer_weight(&layer);
                        // Score with heavier penalty for superseded
                        let base_score = selection::compute_support(evidence_weight as f32, confidence as f32);
                        let score = base_score * 0.3 * w_layer as f32; // 70% penalty for superseded
                        
                        facts.push(ScoredFact {
                            id: superseded_id,
                            entity_id: row.get("subject_entity_id"),
                            entity_label: row.get("label"),
                            key: row.get("key"),
                            topic_key: row.get("topic_key"),
                            value: row.get("value_literal"),
                            confidence: confidence as f32,
                            score,
                            time_bucket_kind: row.try_get("time_bucket_kind").unwrap_or("atemporal".to_string()),
                            time_bucket_value: row.try_get("time_bucket_value").ok(),
                            signature_hash: row.try_get("signature_hash").unwrap_or_default(),
                            polarity: row.try_get("polarity").unwrap_or("assert".to_string()),
                            scope: row.try_get("scope").unwrap_or("\"global\"".to_string()),
                            observed_at,
                            observed_at_formatted,
                        });
                    }
                }
            }
            
            // 4. Sort chronologically (oldest first for history view)
            facts.sort_by(|a, b| {
                // Primary: time_bucket_value ascending (oldest first)
                let a_time = a.time_bucket_value.as_deref().unwrap_or("");
                let b_time = b.time_bucket_value.as_deref().unwrap_or("");
                a_time.cmp(b_time)
                    // Tie-break: observed_at ascending
                    .then_with(|| {
                        let a_obs = a.observed_at.as_deref().unwrap_or("");
                        let b_obs = b.observed_at.as_deref().unwrap_or("");
                        a_obs.cmp(b_obs)
                    })
                    // Final tie-break: signature_hash
                    .then_with(|| a.signature_hash.cmp(&b.signature_hash))
            });
        },
        QueryIntent::AskExplain => {
            // Include all with full ranking
            facts.sort_by(|a, b| {
                b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.signature_hash.cmp(&b.signature_hash))
            });
        },
    }

    if matches!(intent, QueryIntent::AskList | QueryIntent::AskExplain) {
        apply_rerank(query, &mut facts, &mut relations);
        facts.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.signature_hash.cmp(&b.signature_hash))
        });
    }
    
    // I7: Also sort relations with tie-break
    relations.sort_by(|a, b| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.signature_hash.cmp(&b.signature_hash))
    });

    if facts.len() > MAX_RECALLED_FACTS {
        let mut ranked: Vec<(usize, f32)> = facts
            .iter()
            .enumerate()
            .map(|(idx, fact)| (idx, cap_priority_fact(fact)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep: std::collections::HashSet<usize> = ranked
            .into_iter()
            .take(MAX_RECALLED_FACTS)
            .map(|(idx, _)| idx)
            .collect();
        facts = facts
            .into_iter()
            .enumerate()
            .filter_map(|(idx, fact)| if keep.contains(&idx) { Some(fact) } else { None })
            .collect();
    }
    if relations.len() > MAX_RECALLED_RELS {
        let mut ranked: Vec<(usize, f32)> = relations
            .iter()
            .enumerate()
            .map(|(idx, rel)| (idx, cap_priority_rel(rel)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep: std::collections::HashSet<usize> = ranked
            .into_iter()
            .take(MAX_RECALLED_RELS)
            .map(|(idx, _)| idx)
            .collect();
        relations = relations
            .into_iter()
            .enumerate()
            .filter_map(|(idx, rel)| if keep.contains(&idx) { Some(rel) } else { None })
            .collect();
    }

    if facts.len() > MAX_RECALLED_FACTS {
        facts.truncate(MAX_RECALLED_FACTS);
    }
    if relations.len() > MAX_RECALLED_RELS {
        relations.truncate(MAX_RECALLED_RELS);
    }

    // Build debug log if requested
    let debug_log = if debug {
        Some(RetrievalDebugLog {
            query: query.to_string(),
            anchors: anchor_debug,
            traversal: TraversalDebug {
                starting_anchors: anchor_ids.len(),
                hops_taken: traversal_res.hops_taken,
                entities_visited: traversal_res.entities_visited,
                beliefs_collected: traversal_res.beliefs_collected,
                frontier_max_size: traversal_res.frontier_max_size,
                was_bounded: traversal_res.was_bounded,
            },
            selection: SelectionDebug {
                facts_before_filter,
                facts_after_negation,
                facts_after_shadowing,
                facts_final: facts.len(),
                top_facts: facts.iter().take(5).map(|f| FactRankingDebug {
                    id: f.id,
                    topic_key: f.topic_key.clone(),
                    value_preview: f.value.chars().take(50).collect(),
                    score: f.score,
                    components: RankingComponents::default(),
                }).collect(),
            },
            shadowing: ShadowingDebug {
                topic_groups_count,
                beliefs_shadowed: shadowed_details.len(),
                shadowed_details,
            },
            duration_ms: start_time.elapsed().as_millis() as u64,
        })
    } else {
        None
    };

    let conversation_id = session_id_from_scopes(scopes);
    if !phi_consent_allowed(pool, conversation_id).await {
        filter_sensitive_beliefs(pool, &mut facts, &mut relations).await;
    }

    let episodic_events = if matches!(intent, QueryIntent::AskExplain) {
        let mut belief_ids: Vec<i64> = Vec::new();
        for fact in facts.iter().take(10) {
            if !belief_ids.contains(&fact.id) {
                belief_ids.push(fact.id);
            }
        }
        for rel in relations.iter().take(10) {
            if !belief_ids.contains(&rel.id) {
                belief_ids.push(rel.id);
            }
        }

        if belief_ids.is_empty() {
            None
        } else {
            let events = fetch_episodic_events_for_beliefs(pool, &belief_ids, 50).await.unwrap_or_default();
            if events.is_empty() { None } else { Some(events) }
        }
    } else {
        None
    };

    let bound_handles = fetch_bound_handles(pool, session_id).await.unwrap_or_default();

    Ok(MemoryPacket {
        facts,
        relations,
        conflicts,
        bound_handles,
        shadowed_by_scope_count: Some(shadowed_by_scope_count),
        dropped_by_scope_count: Some(dropped_by_scope_count),
        episodic_events,
        debug_log,
    })
}

async fn filter_sensitive_beliefs(
    pool: &SqlitePool,
    facts: &mut Vec<ScoredFact>,
    relations: &mut Vec<ScoredRel>,
) {
    let mut ids: Vec<i64> = facts.iter().map(|f| f.id).collect();
    ids.extend(relations.iter().map(|r| r.id));
    if ids.is_empty() {
        return;
    }
    let mut builder: QueryBuilder<sqlx::Sqlite> = QueryBuilder::new(
        "SELECT belief_id FROM memory_sensitivity WHERE sensitivity IN ('pii', 'phi') AND belief_id IN (",
    );
    let mut separated = builder.separated(", ");
    for id in ids.iter() {
        separated.push_bind(id);
    }
    separated.push_unseparated(")");
    let rows: Vec<i64> = builder
        .build_query_scalar()
        .fetch_all(pool)
        .await
        .unwrap_or_default();
    if rows.is_empty() {
        return;
    }
    let sensitive: HashSet<i64> = rows.into_iter().collect();
    facts.retain(|fact| !sensitive.contains(&fact.id));
    relations.retain(|rel| !sensitive.contains(&rel.id));
}

fn cap_priority_fact(fact: &ScoredFact) -> f32 {
    let scope = scope_specificity(&fact.scope) as f32;
    let confidence = fact.confidence;
    (fact.score * 1.0) + (confidence * 0.4) + (scope * 0.2)
}

fn cap_priority_rel(rel: &ScoredRel) -> f32 {
    let scope = scope_specificity(&rel.scope) as f32;
    let confidence = rel.confidence;
    (rel.score * 1.0) + (confidence * 0.4) + (scope * 0.2)
}

fn apply_rerank(query: &str, facts: &mut [ScoredFact], relations: &mut [ScoredRel]) {
    let tokens = tokenize_query(query);
    if tokens.is_empty() {
        return;
    }

    let max_facts = RERANK_TOP_N.min(facts.len());
    if max_facts > 0 {
        let mut indices: Vec<usize> = (0..facts.len()).collect();
        indices.sort_by(|a, b| {
            facts[*b]
                .score
                .partial_cmp(&facts[*a].score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for idx in indices.into_iter().take(max_facts) {
            let text = format!(
                "{} {} {}",
                facts[idx].entity_label,
                facts[idx].key,
                facts[idx].value
            );
            let boost = 1.0 + (token_overlap_ratio(&tokens, &text) * 0.2);
            facts[idx].score *= boost;
        }
    }

    let max_rels = RERANK_TOP_N.min(relations.len());
    if max_rels > 0 {
        let mut indices: Vec<usize> = (0..relations.len()).collect();
        indices.sort_by(|a, b| {
            relations[*b]
                .score
                .partial_cmp(&relations[*a].score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for idx in indices.into_iter().take(max_rels) {
            let mut text = relations[idx].rel_type.clone();
            for participant in &relations[idx].participants {
                text.push(' ');
                text.push_str(&participant.role);
                text.push(' ');
                text.push_str(&participant.entity_label);
            }
            let boost = 1.0 + (token_overlap_ratio(&tokens, &text) * 0.2);
            relations[idx].score *= boost;
        }
    }
}

fn tokenize_query(query: &str) -> Vec<String> {
    query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn token_overlap_ratio(tokens: &[String], text: &str) -> f32 {
    if tokens.is_empty() {
        return 0.0;
    }
    let lower = text.to_lowercase();
    let mut hit = 0usize;
    for token in tokens {
        if lower.contains(token) {
            hit += 1;
        }
    }
    (hit as f32) / (tokens.len() as f32)
}

fn decay_weight(age_hours: f64, base_half_life: f64, evidence_weight: f64) -> f64 {
    let reinforcement = (evidence_weight / 5.0).max(0.0);
    let effective_half_life = base_half_life * (1.0 + reinforcement);
    if effective_half_life <= 0.0 {
        return 1.0;
    }
    0.5_f64.powf(age_hours / effective_half_life)
}

fn layer_weight(layer: &str) -> f64 {
    match layer.trim().to_lowercase().as_str() {
        "world" => 1.30,
        "semantic" => 1.15,
        "episodic" => 1.0,
        "working" => 0.75,
        _ => 1.0,
    }
}

async fn apply_working_set_boost(
    pool: &SqlitePool,
    facts: &mut Vec<ScoredFact>,
    relations: &mut Vec<ScoredRel>,
) -> Result<(), String> {
    let mut entity_ids = HashSet::new();
    let mut belief_ids = HashSet::new();

    for fact in facts.iter() {
        entity_ids.insert(fact.entity_id);
        belief_ids.insert(fact.id);
    }
    for rel in relations.iter() {
        belief_ids.insert(rel.id);
        for participant in &rel.participants {
            entity_ids.insert(participant.entity_id);
        }
    }

    let entity_ids: Vec<i64> = entity_ids.into_iter().collect();
    let belief_ids: Vec<i64> = belief_ids.into_iter().collect();

    let entity_activation = fetch_working_set_activation(pool, "entity", &entity_ids).await?;
    let belief_activation = fetch_working_set_activation(pool, "belief", &belief_ids).await?;

    let clamp = |value: f32| value.max(0.0).min(1.0);

    for fact in facts.iter_mut() {
        let entity_act = clamp(*entity_activation.get(&fact.entity_id).unwrap_or(&0.0));
        let belief_act = clamp(*belief_activation.get(&fact.id).unwrap_or(&0.0));
        let boost = 1.0
            + (entity_act * WORKING_SET_ENTITY_SCORE_BOOST)
            + (belief_act * WORKING_SET_BELIEF_SCORE_BOOST);
        fact.score *= boost;
    }

    for rel in relations.iter_mut() {
        let belief_act = clamp(*belief_activation.get(&rel.id).unwrap_or(&0.0));
        let mut entity_act = 0.0;
        for participant in &rel.participants {
            let value = clamp(*entity_activation.get(&participant.entity_id).unwrap_or(&0.0));
            if value > entity_act {
                entity_act = value;
            }
        }
        let boost = 1.0
            + (entity_act * WORKING_SET_ENTITY_SCORE_BOOST)
            + (belief_act * WORKING_SET_BELIEF_SCORE_BOOST);
        rel.score *= boost;
    }

    Ok(())
}

async fn fetch_working_set_activation(
    pool: &SqlitePool,
    item_type: &str,
    ids: &[i64],
) -> Result<HashMap<i64, f32>, String> {
    if ids.is_empty() {
        return Ok(HashMap::new());
    }

    let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let query = format!(
        "SELECT item_id, activation FROM ics_working_set WHERE item_type = ? AND item_id IN ({})",
        placeholders
    );
    let mut q = sqlx::query(&query).bind(item_type);
    for id in ids {
        q = q.bind(id);
    }

    let rows = q.fetch_all(pool).await.map_err(|e| e.to_string())?;
    let mut map = HashMap::new();
    for row in rows {
        let item_id: i64 = row.get("item_id");
        let activation: f64 = row.get("activation");
        map.insert(item_id, activation as f32);
    }
    Ok(map)
}

pub async fn fetch_episodic_events_for_beliefs(
    pool: &SqlitePool,
    belief_ids: &[i64],
    limit: i64,
) -> Result<Vec<EpisodicEvent>, String> {
    if belief_ids.is_empty() {
        return Ok(Vec::new());
    };

    let enabled: Option<i32> = sqlx::query_scalar("SELECT episodic_enabled FROM settings WHERE id = 1")
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();
    if enabled.unwrap_or(0) == 0 {
        return Ok(Vec::new());
    }

    let placeholders = belief_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let query = format!(
        "SELECT e.id, e.event_type, e.payload_json, e.timestamp, e.run_id, e.trace_id, e.conversation_id, e.scope, e.source_type, e.source_ref, e.linked_belief_id, e.linked_artifact_id
         FROM ics_evidence_events ev
         JOIN episodic_events e ON e.id = ev.episodic_event_id
         WHERE ev.belief_id IN ({})
         ORDER BY e.timestamp DESC, e.rowid DESC
         LIMIT ?",
        placeholders
    );

    let mut q = sqlx::query(&query);
    for id in belief_ids {
        q = q.bind(id);
    }
    q = q.bind(limit.max(1).min(200));

    let rows = q.fetch_all(pool).await.map_err(|e| e.to_string())?;
    let mut events = Vec::new();
    for row in rows {
        let payload_raw: String = row.get("payload_json");
        events.push(EpisodicEvent {
            id: row.get("id"),
            event_type: row.get("event_type"),
            payload: serde_json::from_str(&payload_raw).unwrap_or_else(|_| serde_json::json!({})),
            timestamp: row.get("timestamp"),
            run_id: row.try_get("run_id").ok(),
            trace_id: row.try_get("trace_id").ok(),
            conversation_id: row.try_get("conversation_id").ok(),
            scope: row.try_get("scope").ok(),
            source_type: row.get("source_type"),
            source_ref: row.try_get("source_ref").ok(),
            linked_belief_id: row.try_get("linked_belief_id").ok(),
            linked_artifact_id: row.try_get("linked_artifact_id").ok(),
        });
    }
    Ok(events)
}

async fn get_session_user_entity_id(pool: &SqlitePool, session_id: &str) -> Result<Option<i64>, String> {
    let session_id = session_id.trim();
    let session_id = if session_id.is_empty() { "default" } else { session_id };
    let row = sqlx::query(
        "SELECT entity_id FROM ics_session_bindings
         WHERE ref_text = 'user' AND session_id = ?
         LIMIT 1"
    )
    .bind(session_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    if let Some(r) = row {
        return Ok(Some(r.get::<i64, _>("entity_id")));
    }

    if session_id != "default" {
        let row = sqlx::query(
            "SELECT entity_id FROM ics_session_bindings
             WHERE ref_text = 'user' AND session_id = 'default'
             LIMIT 1"
        )
        .fetch_optional(pool)
        .await
        .map_err(|e| e.to_string())?;
        if let Some(r) = row {
            return Ok(Some(r.get::<i64, _>("entity_id")));
        }
    }

    let row = sqlx::query(
        "SELECT id FROM ics_entities
         WHERE keys LIKE ?
         ORDER BY access_count DESC
         LIMIT 1"
    )
    .bind("%\"sys:user\"%")
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    Ok(row.map(|r| r.get::<i64, _>("id")))
}

#[cfg(test)]
mod tests {
    use super::{is_personal_query, retrieve};
    use crate::core::memory::canonical::compute_value_hash;
    use crate::core::memory::types::{QueryIntent, Scope};
    use crate::db::Db;
    use sqlx::Row;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn personal_query_detects_relationships() {
        assert!(is_personal_query("Who is my father?"));
    }

    #[test]
    fn personal_query_detects_location() {
        assert!(is_personal_query("Where do I live?"));
    }

    #[test]
    fn personal_query_rejects_generic() {
        assert!(!is_personal_query("What is Rust?"));
    }

    async fn setup_db() -> Db {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("pool");
        let schema_path = PathBuf::from("src/db/schema.sql");
        let schema_sql = fs::read_to_string(&schema_path).expect("schema");
        sqlx::query(&schema_sql).execute(&pool).await.expect("apply schema");
        sqlx::query("INSERT INTO settings (id, schema_version, api_base_url) VALUES (1, 1, 'http://localhost')")
            .execute(&pool)
            .await
            .expect("seed settings");
        Db { pool }
    }

    #[tokio::test]
    async fn retrieve_returns_fact_from_db() {
        let db = setup_db().await;
        let entity_id: i64 = sqlx::query(
            "INSERT INTO ics_entities (label, label_canonical, aliases, aliases_canonical, keys, resolution_state)
             VALUES ('User', 'user', '[]', '[]', '[]', 'normal')
             RETURNING id",
        )
        .fetch_one(&db.pool)
        .await
        .map(|row| row.get::<i64, _>("id"))
        .expect("entity");

        let _ = sqlx::query(
            "INSERT INTO ics_session_bindings (session_id, ref_text, entity_id, created_at)
             VALUES ('default', 'user', ?, CURRENT_TIMESTAMP)",
        )
        .bind(entity_id)
        .execute(&db.pool)
        .await;

        let scope = serde_json::to_string(&Scope::Session).unwrap_or_else(|_| "\"session\"".to_string());
        let belief_id: i64 = sqlx::query(
            "INSERT INTO ics_beliefs
             (kind, scope, status, layer, polarity, confidence, salience, topic_key, signature_hash, evidence_weight_total, last_evidence_at, created_at)
             VALUES ('fact', ?, 'active', 'episodic', 'assert', 1.0, 1.0, 'fact:work', 'sig:work', 1.0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
             RETURNING id",
        )
        .bind(&scope)
        .fetch_one(&db.pool)
        .await
        .map(|row| row.get::<i64, _>("id"))
        .expect("belief");

        let value_hash = compute_value_hash("Acme");
        let _ = sqlx::query(
            "INSERT INTO ics_fact_beliefs (belief_id, subject_entity_id, key, value_literal, value_hash)
             VALUES (?, ?, 'workplace', 'Acme', ?)",
        )
        .bind(belief_id)
        .bind(entity_id)
        .bind(&value_hash)
        .execute(&db.pool)
        .await;

        let _ = sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id)
             VALUES (?, 'user', 'user', 'user said they work at Acme', 0.9, NULL)",
        )
        .bind(belief_id)
        .execute(&db.pool)
        .await;

        let packet = retrieve(
            "Acme",
            &[Scope::Session],
            QueryIntent::AskCurrent,
            &db.pool,
            None,
            None,
        )
        .await
        .expect("retrieve");

        assert!(packet.facts.iter().any(|fact| fact.value == "Acme"));
    }
}

async fn fetch_bound_handles(pool: &SqlitePool, session_id: &str) -> Result<HashMap<String, String>, String> {
    let session_id = session_id.trim();
    let session_id = if session_id.is_empty() { "default" } else { session_id };
    let rows = sqlx::query(
        "SELECT sb.ref_text, e.label, sb.session_id
         FROM ics_session_bindings sb
         JOIN ics_entities e ON e.id = sb.entity_id
         WHERE sb.ref_text IN ('user', 'assistant')
           AND sb.session_id = ?
         ORDER BY sb.created_at DESC"
    )
    .bind(session_id)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    let mut handles = HashMap::new();
    for row in rows {
        let ref_text: String = row.get("ref_text");
        let label: String = row.get("label");
        let handle = format!("${}", ref_text);
        if !handles.contains_key(&handle) {
            handles.insert(handle, label);
        }
    }

    if handles.is_empty() && session_id != "default" {
        let fallback = sqlx::query(
            "SELECT sb.ref_text, e.label, sb.session_id
             FROM ics_session_bindings sb
             JOIN ics_entities e ON e.id = sb.entity_id
             WHERE sb.ref_text IN ('user', 'assistant')
               AND sb.session_id = 'default'
             ORDER BY sb.created_at DESC"
        )
        .fetch_all(pool)
        .await
        .map_err(|e| e.to_string())?;

        for row in fallback {
            let ref_text: String = row.get("ref_text");
            let label: String = row.get("label");
            let handle = format!("${}", ref_text);
            if !handles.contains_key(&handle) {
                handles.insert(handle, label);
            }
        }
    }

    Ok(handles)
}


