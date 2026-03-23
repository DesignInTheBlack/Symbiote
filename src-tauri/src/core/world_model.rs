use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::Row;

use crate::db::Db;

const ENTITY_LIMIT: usize = 64;
const FACT_LIMIT: usize = 96;
const RELATION_LIMIT: usize = 96;
const CONFLICT_LIMIT: usize = 24;

const PROMPT_ENTITY_LIMIT: usize = 24;
const PROMPT_FACT_LIMIT: usize = 32;
const PROMPT_REL_LIMIT: usize = 32;
const PROMPT_CONFLICT_LIMIT: usize = 12;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorldModelSnapshot {
    pub generated_at: String,
    pub conversation_id: Option<String>,
    pub entity_count: usize,
    pub belief_count: usize,
    pub conflict_count: usize,
    pub entities: Vec<WorldModelEntity>,
    pub facts: Vec<WorldModelFact>,
    pub relations: Vec<WorldModelRelation>,
    pub conflicts: Vec<WorldModelConflict>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelEntity {
    pub id: i64,
    pub label: String,
    pub entity_type: Option<String>,
    pub aliases: Vec<String>,
    pub keys: Vec<String>,
    pub access_count: i64,
    pub last_accessed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelFact {
    pub belief_id: i64,
    pub entity_id: i64,
    pub entity_label: String,
    pub key: String,
    pub value: String,
    pub confidence: f64,
    pub evidence_weight: f64,
    pub polarity: String,
    pub scope: String,
    pub layer: String,
    pub topic_key: String,
    pub reconcile_state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelRelation {
    pub belief_id: i64,
    pub rel_type: String,
    pub direction: Option<String>,
    pub participants: Vec<WorldModelRelationParticipant>,
    pub confidence: f64,
    pub evidence_weight: f64,
    pub polarity: String,
    pub scope: String,
    pub layer: String,
    pub topic_key: String,
    pub reconcile_state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelRelationParticipant {
    pub role: String,
    pub entity_id: i64,
    pub entity_label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelConflict {
    pub id: i64,
    pub topic_key: String,
    pub status: String,
    pub priority: String,
    pub resolution_note: Option<String>,
    pub member_belief_ids: Vec<i64>,
    pub updated_at: String,
}

pub async fn build_world_model_snapshot(
    db: &Db,
    conversation_id: &str,
) -> Result<WorldModelSnapshot, String> {
    let entity_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ics_entities")
        .fetch_optional(&db.pool)
        .await
        .ok()
        .flatten()
        .unwrap_or(0);
    let belief_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ics_beliefs WHERE status = 'active' AND reconcile_state != 'retired'",
    )
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);
    let conflict_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ics_conflict_sets WHERE status != 'archived'",
    )
    .fetch_optional(&db.pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0);

    let entities = load_entities(db).await.unwrap_or_default();
    let facts = load_facts(db).await.unwrap_or_default();
    let relations = load_relations(db).await.unwrap_or_default();
    let conflicts = load_conflicts(db).await.unwrap_or_default();

    Ok(WorldModelSnapshot {
        generated_at: Utc::now().to_rfc3339(),
        conversation_id: Some(conversation_id.to_string()),
        entity_count: entity_count as usize,
        belief_count: belief_count as usize,
        conflict_count: conflict_count as usize,
        entities,
        facts,
        relations,
        conflicts,
    })
}

pub fn snapshot_from_subject_state_json(raw: &str) -> Option<WorldModelSnapshot> {
    let value: Value = serde_json::from_str(raw).ok()?;
    let world_model = value.get("state")?.get("world_model")?.clone();
    serde_json::from_value::<WorldModelSnapshot>(world_model).ok()
}

pub fn render_world_model_prompt(snapshot: &WorldModelSnapshot) -> String {
    if snapshot.entities.is_empty()
        && snapshot.facts.is_empty()
        && snapshot.relations.is_empty()
        && snapshot.conflicts.is_empty()
    {
        return "None".to_string();
    }

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("generated_at: {}", snapshot.generated_at));
    lines.push(format!(
        "counts: entities={} beliefs={} conflicts={}",
        snapshot.entity_count, snapshot.belief_count, snapshot.conflict_count
    ));

    if !snapshot.entities.is_empty() {
        lines.push("Entities:".to_string());
        for entity in snapshot.entities.iter().take(PROMPT_ENTITY_LIMIT) {
            let mut label = entity.label.clone();
            if let Some(entity_type) = entity.entity_type.as_deref().filter(|s| !s.trim().is_empty()) {
                label = format!("{} ({})", label, entity_type);
            }
            lines.push(format!("- {}", label));
        }
        let omitted = snapshot.entities.len().saturating_sub(PROMPT_ENTITY_LIMIT);
        if omitted > 0 {
            lines.push(format!("... {} more entities", omitted));
        }
    }

    if !snapshot.facts.is_empty() {
        lines.push("Facts:".to_string());
        for fact in snapshot.facts.iter().take(PROMPT_FACT_LIMIT) {
            let key = compact_text(&fact.key, 64);
            let value = compact_text(&fact.value, 120);
            let reconcile_note = if fact.reconcile_state.trim().eq_ignore_ascii_case("active") {
                "".to_string()
            } else {
                format!(", state: {}", fact.reconcile_state)
            };
            lines.push(format!(
                "- {} {} = {} [conf: {:.2}, evidence: {:.2}, scope: {}, layer: {}{}]",
                fact.entity_label,
                key,
                value,
                fact.confidence,
                fact.evidence_weight,
                fact.scope,
                fact.layer,
                reconcile_note,
            ));
        }
        let omitted = snapshot.facts.len().saturating_sub(PROMPT_FACT_LIMIT);
        if omitted > 0 {
            lines.push(format!("... {} more facts", omitted));
        }
    }

    if !snapshot.relations.is_empty() {
        lines.push("Relations:".to_string());
        for rel in snapshot.relations.iter().take(PROMPT_REL_LIMIT) {
            let participants = rel
                .participants
                .iter()
                .map(|p| format!("{}:#{}", p.role, p.entity_label))
                .collect::<Vec<_>>()
                .join(", ");
            let rel_type = compact_text(&rel.rel_type, 64);
            let direction = rel
                .direction
                .as_deref()
                .filter(|d| !d.trim().is_empty())
                .map(|d| format!(" dir={}", d))
                .unwrap_or_default();
            let reconcile_note = if rel.reconcile_state.trim().eq_ignore_ascii_case("active") {
                "".to_string()
            } else {
                format!(", state: {}", rel.reconcile_state)
            };
            lines.push(format!(
                "- {}({}) [conf: {:.2}, evidence: {:.2}, scope: {}, layer:{}{}{}]",
                rel_type,
                compact_text(&participants, 180),
                rel.confidence,
                rel.evidence_weight,
                rel.scope,
                rel.layer,
                direction,
                reconcile_note,
            ));
        }
        let omitted = snapshot.relations.len().saturating_sub(PROMPT_REL_LIMIT);
        if omitted > 0 {
            lines.push(format!("... {} more relations", omitted));
        }
    }

    if !snapshot.conflicts.is_empty() {
        lines.push("Conflicts:".to_string());
        for conflict in snapshot.conflicts.iter().take(PROMPT_CONFLICT_LIMIT) {
            let members = if conflict.member_belief_ids.is_empty() {
                "none".to_string()
            } else {
                conflict
                    .member_belief_ids
                    .iter()
                    .take(6)
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            lines.push(format!(
                "- {} [status: {}, priority: {}, members: {}]",
                conflict.topic_key, conflict.status, conflict.priority, members
            ));
        }
        let omitted = snapshot.conflicts.len().saturating_sub(PROMPT_CONFLICT_LIMIT);
        if omitted > 0 {
            lines.push(format!("... {} more conflicts", omitted));
        }
    }

    lines.join("\n")
}

async fn load_entities(db: &Db) -> Result<Vec<WorldModelEntity>, String> {
    let rows = sqlx::query(
        "SELECT id, label, entity_type, aliases, keys, access_count, last_accessed_at
         FROM ics_entities
         ORDER BY access_count DESC, datetime(last_accessed_at) DESC, id ASC
         LIMIT ?",
    )
    .bind(ENTITY_LIMIT as i64)
    .fetch_all(&db.pool)
    .await
    .map_err(|e| e.to_string())?;
    let mut entities = Vec::new();
    for row in rows {
        let aliases_raw: Option<String> = row.try_get("aliases").ok();
        let keys_raw: Option<String> = row.try_get("keys").ok();
        let aliases = parse_json_array(aliases_raw);
        let keys = parse_json_array(keys_raw);
        entities.push(WorldModelEntity {
            id: row.get::<i64, _>("id"),
            label: row.get::<String, _>("label"),
            entity_type: row.try_get::<String, _>("entity_type").ok(),
            aliases,
            keys,
            access_count: row.try_get::<i64, _>("access_count").unwrap_or(0),
            last_accessed_at: row.try_get::<String, _>("last_accessed_at").ok(),
        });
    }
    Ok(entities)
}

async fn load_facts(db: &Db) -> Result<Vec<WorldModelFact>, String> {
    let rows = sqlx::query(
        "SELECT b.id AS belief_id, b.scope, b.polarity, b.confidence, b.evidence_weight_total,
                b.layer, b.topic_key, b.reconcile_state, f.subject_entity_id, f.key, f.value_literal, e.label AS entity_label
         FROM ics_beliefs b
         JOIN ics_fact_beliefs f ON f.belief_id = b.id
         JOIN ics_entities e ON e.id = f.subject_entity_id
         WHERE b.status = 'active' AND b.reconcile_state != 'retired' AND b.kind = 'fact'
         ORDER BY
            CASE b.layer WHEN 'world' THEN 3 WHEN 'semantic' THEN 2 WHEN 'episodic' THEN 1 ELSE 0 END DESC,
            b.evidence_weight_total DESC,
            b.confidence DESC,
            datetime(b.last_evidence_at) DESC,
            b.id ASC
         LIMIT ?",
    )
    .bind(FACT_LIMIT as i64)
    .fetch_all(&db.pool)
    .await
    .map_err(|e| e.to_string())?;
    let mut facts = Vec::new();
    for row in rows {
        facts.push(WorldModelFact {
            belief_id: row.get::<i64, _>("belief_id"),
            entity_id: row.get::<i64, _>("subject_entity_id"),
            entity_label: row.get::<String, _>("entity_label"),
            key: row.get::<String, _>("key"),
            value: row.get::<String, _>("value_literal"),
            confidence: row.try_get::<f64, _>("confidence").unwrap_or(0.0),
            evidence_weight: row.try_get::<f64, _>("evidence_weight_total").unwrap_or(0.0),
            polarity: row.get::<String, _>("polarity"),
            scope: row.get::<String, _>("scope"),
            layer: row.get::<String, _>("layer"),
            topic_key: row.get::<String, _>("topic_key"),
            reconcile_state: row.try_get::<String, _>("reconcile_state").unwrap_or_else(|_| "active".to_string()),
        });
    }
    Ok(facts)
}

async fn load_relations(db: &Db) -> Result<Vec<WorldModelRelation>, String> {
    let rows = sqlx::query(
        "SELECT b.id AS belief_id, b.scope, b.polarity, b.confidence, b.evidence_weight_total,
                b.layer, b.topic_key, b.reconcile_state, r.rel_type, r.direction
         FROM ics_beliefs b
         JOIN ics_rel_beliefs r ON r.belief_id = b.id
         WHERE b.status = 'active' AND b.reconcile_state != 'retired' AND b.kind = 'rel'
         ORDER BY
            CASE b.layer WHEN 'world' THEN 3 WHEN 'semantic' THEN 2 WHEN 'episodic' THEN 1 ELSE 0 END DESC,
            b.evidence_weight_total DESC,
            b.confidence DESC,
            datetime(b.last_evidence_at) DESC,
            b.id ASC
         LIMIT ?",
    )
    .bind(RELATION_LIMIT as i64)
    .fetch_all(&db.pool)
    .await
    .map_err(|e| e.to_string())?;
    let mut relations = Vec::new();
    for row in rows {
        let belief_id: i64 = row.get("belief_id");
        let participant_rows = sqlx::query(
            "SELECT p.role, p.entity_id, e.label
             FROM ics_rel_participants p
             JOIN ics_entities e ON e.id = p.entity_id
             WHERE p.belief_id = ?
             ORDER BY p.role ASC, p.entity_id ASC",
        )
        .bind(belief_id)
        .fetch_all(&db.pool)
        .await
        .map_err(|e| e.to_string())?;
        let mut participants = Vec::new();
        for p in participant_rows {
            participants.push(WorldModelRelationParticipant {
                role: p.get::<String, _>("role"),
                entity_id: p.get::<i64, _>("entity_id"),
                entity_label: p.get::<String, _>("label"),
            });
        }
        relations.push(WorldModelRelation {
            belief_id,
            rel_type: row.get::<String, _>("rel_type"),
            direction: row.try_get::<String, _>("direction").ok(),
            participants,
            confidence: row.try_get::<f64, _>("confidence").unwrap_or(0.0),
            evidence_weight: row.try_get::<f64, _>("evidence_weight_total").unwrap_or(0.0),
            polarity: row.get::<String, _>("polarity"),
            scope: row.get::<String, _>("scope"),
            layer: row.get::<String, _>("layer"),
            topic_key: row.get::<String, _>("topic_key"),
            reconcile_state: row.try_get::<String, _>("reconcile_state").unwrap_or_else(|_| "active".to_string()),
        });
    }
    Ok(relations)
}

async fn load_conflicts(db: &Db) -> Result<Vec<WorldModelConflict>, String> {
    let rows = sqlx::query(
        "SELECT id, topic_key, status, priority, resolution_note, updated_at
         FROM ics_conflict_sets
         WHERE status != 'archived'
         ORDER BY
            CASE status WHEN 'open' THEN 2 WHEN 'resolved' THEN 1 ELSE 0 END DESC,
            datetime(updated_at) DESC,
            id ASC
         LIMIT ?",
    )
    .bind(CONFLICT_LIMIT as i64)
    .fetch_all(&db.pool)
    .await
    .map_err(|e| e.to_string())?;
    let mut conflicts = Vec::new();
    for row in rows {
        let conflict_id: i64 = row.get("id");
        let member_rows = sqlx::query(
            "SELECT belief_id
             FROM ics_conflict_set_members
             WHERE conflict_set_id = ?
             ORDER BY belief_id ASC",
        )
        .bind(conflict_id)
        .fetch_all(&db.pool)
        .await
        .map_err(|e| e.to_string())?;
        let member_belief_ids = member_rows
            .iter()
            .filter_map(|r| r.try_get::<i64, _>("belief_id").ok())
            .collect::<Vec<_>>();
        conflicts.push(WorldModelConflict {
            id: conflict_id,
            topic_key: row.get::<String, _>("topic_key"),
            status: row.get::<String, _>("status"),
            priority: row.get::<String, _>("priority"),
            resolution_note: row.try_get::<String, _>("resolution_note").ok(),
            member_belief_ids,
            updated_at: row.get::<String, _>("updated_at"),
        });
    }
    Ok(conflicts)
}

fn parse_json_array(raw: Option<String>) -> Vec<String> {
    let Some(raw) = raw else {
        return Vec::new();
    };
    serde_json::from_str::<Vec<String>>(&raw).unwrap_or_default()
}

fn compact_text(input: &str, max_len: usize) -> String {
    let trimmed = input.replace('\n', " ").replace('\r', " ");
    let collapsed = trimmed
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if max_len == 0 {
        return String::new();
    }
    if collapsed.chars().count() <= max_len {
        collapsed
    } else {
        let mut out: String = collapsed.chars().take(max_len).collect();
        out.push_str("...");
        out
    }
}
