use sqlx::Row;
use crate::core::memory::dsl::{FactStmt, Ref, RelDirection};
use crate::core::memory::canonical::normalize_role_token;
use crate::core::memory::resolver::{self, ResolveContext, ResolutionResult};
use crate::core::memory::writer::WriteContext;
use crate::core::system_log;
use serde_json::json;

#[derive(Clone, Debug)]
pub struct PromotionResult {
    pub rel_type: String,
    pub participants: Vec<(String, i64)>,
    pub direction: Option<RelDirection>,
    pub low_confidence: bool,
}

#[derive(Clone, Copy, Debug)]
struct HeuristicRel {
    rel_type: &'static str,
    subject_role: &'static str,
    value_role: &'static str,
    direction: RelDirection,
}

/// FACT->REL Promotion (Sec 6.3)
/// Checks if a fact should be promoted to a relation based on ics_promotion_maps,
/// then falls back to heuristic relation promotion for role-like facts.
pub async fn try_promote_fact(
    fact: &FactStmt,
    subject_id: i64,
    resolve_ctx: &ResolveContext,
    write_ctx: &WriteContext,
) -> Option<PromotionResult> {
    let log_promotion = |_event: &str, payload: serde_json::Value| async move {
        let _ = system_log::log_event(
            &write_ctx.pool,
            None,
            "info",
            "memory",
            None,
            None,
            payload,
        )
        .await;
    };

    if let Some(row) = sqlx::query(
        "SELECT to_rel_type, subject_role, value_role FROM ics_promotion_maps WHERE from_fact_key = ? AND status = 'active'"
    )
    .bind(&fact.key)
    .fetch_optional(&write_ctx.pool)
    .await
    .ok()
    .flatten()
    {
        let to_rel_type: String = row.get("to_rel_type");
        let subject_role_raw: String = row.get("subject_role");
        let value_role_raw: String = row.get("value_role");
        let subject_role = normalize_role_token(&subject_role_raw);
        let value_role = normalize_role_token(&value_role_raw);
        let value = fact.value.trim();
        let allow_new = allow_new_entity(value, false);
        let Some((value_id, created_new)) =
            resolve_value_entity(value, &value_role, resolve_ctx, write_ctx, allow_new).await
        else {
            log_promotion(
                "relation_promotion_skipped",
                json!({
                    "event": "relation_promotion_skipped",
                    "reason": "promotion_map_resolution_failed",
                    "fact_key": fact.key,
                    "value": value,
                    "rel_type": to_rel_type,
                    "conversation_id": write_ctx.conversation_id,
                }),
            )
            .await;
            return None;
        };
        log_promotion(
            "relation_promotion_created",
            json!({
                "event": "relation_promotion_created",
                "source": "promotion_map",
                "fact_key": fact.key,
                "value": value,
                "rel_type": to_rel_type,
                "created_new_entity": created_new,
                "low_confidence": false,
                "conversation_id": write_ctx.conversation_id,
            }),
        )
        .await;
        let participants = vec![
            (subject_role, subject_id),
            (value_role, value_id),
        ];
        return Some(PromotionResult {
            rel_type: to_rel_type,
            participants,
            direction: None,
            low_confidence: false,
        });
    }

    let heuristic = heuristic_relation(&fact.key)?;
    let value = fact.value.trim();
    if !looks_like_entity_label(value) {
        log_promotion(
            "relation_promotion_skipped",
            json!({
                "event": "relation_promotion_skipped",
                "reason": "value_not_entity_label",
                "fact_key": fact.key,
                "value": value,
                "rel_type": heuristic.rel_type,
                "conversation_id": write_ctx.conversation_id,
            }),
        )
        .await;
        return None;
    }
    let explicit_ref = value.starts_with('#') || value.starts_with('$');
    if !explicit_ref && !strong_entity_label(value) {
        log_promotion(
            "relation_promotion_skipped",
            json!({
                "event": "relation_promotion_skipped",
                "reason": "low_confidence_weak_label",
                "fact_key": fact.key,
                "value": value,
                "rel_type": heuristic.rel_type,
                "conversation_id": write_ctx.conversation_id,
            }),
        )
        .await;
        return None;
    }
    let subject_role = normalize_role_token(heuristic.subject_role);
    let value_role = normalize_role_token(heuristic.value_role);
    let allow_new = if explicit_ref { true } else { false };
    let Some((value_id, created_new)) =
        resolve_value_entity(value, &value_role, resolve_ctx, write_ctx, allow_new).await
    else {
        log_promotion(
            "relation_promotion_skipped",
            json!({
                "event": "relation_promotion_skipped",
                "reason": "low_confidence_no_coref",
                "fact_key": fact.key,
                "value": value,
                "rel_type": heuristic.rel_type,
                "conversation_id": write_ctx.conversation_id,
            }),
        )
        .await;
        return None;
    };
    {
        let _ = system_log::log_event(
            &write_ctx.pool,
            None,
            "warn",
            "memory",
            None,
            None,
            json!({
                "event": "memory_promotion_low_confidence",
                "fact_key": fact.key,
                "value": value,
                "rel_type": heuristic.rel_type,
                "direction": format!("{:?}", heuristic.direction),
                "created_new_entity": created_new,
                "conversation_id": write_ctx.conversation_id,
            }),
        )
        .await;
    }
    let participants = vec![
        (subject_role, subject_id),
        (value_role, value_id),
    ];
    log_promotion(
        "relation_promotion_created",
        json!({
            "event": "relation_promotion_created",
            "source": "heuristic",
            "fact_key": fact.key,
            "value": value,
            "rel_type": heuristic.rel_type,
            "created_new_entity": created_new,
            "low_confidence": true,
            "conversation_id": write_ctx.conversation_id,
        }),
    )
    .await;
    Some(PromotionResult {
        rel_type: heuristic.rel_type.to_string(),
        participants,
        direction: Some(heuristic.direction),
        low_confidence: true,
    })
}

fn allow_new_entity(value: &str, low_confidence: bool) -> bool {
    let trimmed = value.trim();
    if trimmed.starts_with('#') || trimmed.starts_with('$') {
        return true;
    }
    if low_confidence {
        return strong_entity_label(trimmed);
    }
    strong_entity_label(trimmed) || looks_like_entity_label(trimmed)
}

fn strong_entity_label(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return false;
    }
    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    if tokens.is_empty() {
        return false;
    }
    let capitalized = tokens
        .iter()
        .filter(|token| token.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
        .count();
    if capitalized >= 2 {
        return true;
    }
    if tokens.len() == 1 && capitalized == 1 && trimmed.len() >= 4 {
        return true;
    }
    false
}

fn heuristic_relation(key: &str) -> Option<HeuristicRel> {
    let key = key.trim().to_lowercase();
    match key.as_str() {
        "works_with" | "collaborates_with" => Some(HeuristicRel {
            rel_type: "works_with",
            subject_role: "person",
            value_role: "person",
            direction: RelDirection::Bidirectional,
        }),
        "friends_with" | "friend" | "friends" => Some(HeuristicRel {
            rel_type: "friends",
            subject_role: "person",
            value_role: "person",
            direction: RelDirection::Bidirectional,
        }),
        "spouse" | "spouse_of" | "partner" => Some(HeuristicRel {
            rel_type: "spouse_of",
            subject_role: "spouse",
            value_role: "spouse",
            direction: RelDirection::Bidirectional,
        }),
        "sibling" | "sibling_of" | "brother" | "sister" => Some(HeuristicRel {
            rel_type: "sibling_of",
            subject_role: "sibling",
            value_role: "sibling",
            direction: RelDirection::Bidirectional,
        }),
        "parent" | "parent_of" | "father" | "mother" => Some(HeuristicRel {
            rel_type: "parent_of",
            subject_role: "parent",
            value_role: "child",
            direction: RelDirection::Directed,
        }),
        "child" | "child_of" | "son" | "daughter" => Some(HeuristicRel {
            rel_type: "child_of",
            subject_role: "child",
            value_role: "parent",
            direction: RelDirection::Directed,
        }),
        "member_of" | "member" => Some(HeuristicRel {
            rel_type: "member_of",
            subject_role: "member",
            value_role: "group",
            direction: RelDirection::Directed,
        }),
        "works_at" | "employed_by" => Some(HeuristicRel {
            rel_type: "works_at",
            subject_role: "person",
            value_role: "work",
            direction: RelDirection::Directed,
        }),
        "reports_to" | "manager" => Some(HeuristicRel {
            rel_type: "reports_to",
            subject_role: "person",
            value_role: "person",
            direction: RelDirection::Directed,
        }),
        "lives_in" => Some(HeuristicRel {
            rel_type: "lives_in",
            subject_role: "person",
            value_role: "place",
            direction: RelDirection::Directed,
        }),
        "owns" | "owner_of" => Some(HeuristicRel {
            rel_type: "owns",
            subject_role: "owner",
            value_role: "object",
            direction: RelDirection::Directed,
        }),
        _ => None,
    }
}

fn looks_like_entity_label(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.len() > 64 {
        return false;
    }
    if trimmed.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let word_count = trimmed.split_whitespace().count();
    if word_count > 6 {
        return false;
    }
    let lowered = trimmed.to_lowercase();
    let role_hint = ["friend", "sister", "brother", "parent", "child", "partner", "boss", "manager", "coworker"]
        .iter()
        .any(|token| lowered.contains(token));
    let has_upper = trimmed.chars().any(|c| c.is_uppercase());
    has_upper || role_hint || word_count >= 2
}

fn infer_entity_type(value_role: &str) -> Option<&'static str> {
    match value_role.to_lowercase().as_str() {
        "person" | "user" | "owner" | "author" | "creator" | "subject" | "actor"
        | "parent" | "child" | "mother" | "father" | "daughter" | "son" | "spouse"
        | "sibling" | "brother" | "sister" | "partner" | "husband" | "wife" => Some("person"),
        "place" | "location" | "city" | "country" | "venue" => Some("place"),
        "work" | "project" | "product" | "book" | "movie" | "song" | "company" | "group" => Some("work"),
        "event" | "meeting" | "appointment" => Some("event"),
        "concept" | "idea" | "topic" | "category" | "object" | "thing" | "item" => Some("concept"),
        _ => None,
    }
}

async fn resolve_value_entity(
    value: &str,
    value_role: &str,
    resolve_ctx: &ResolveContext,
    write_ctx: &WriteContext,
    allow_new_entity: bool,
) -> Option<(i64, bool)> {
    let value_ref = parse_value_ref(value);
    match resolver::resolve_ref(&value_ref, resolve_ctx).await {
        Ok(ResolutionResult::Resolved(id)) => Some((id, false)),
        Ok(ResolutionResult::NewEntity(label)) => {
            if !allow_new_entity {
                return None;
            }
            let inferred_type = infer_entity_type(value_role);
            crate::core::memory::writer::create_entity(&label, inferred_type, write_ctx)
                .await
                .ok()
                .map(|id| (id, true))
        }
        _ => None,
    }
}

fn parse_value_ref(value: &str) -> Ref {
    if value.starts_with('$') {
        Ref::Handle(value[1..].to_string())
    } else if value.starts_with('#') {
        Ref::Label(value[1..].to_string())
    } else {
        Ref::Name(value.to_string())
    }
}
