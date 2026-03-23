use std::collections::{HashMap, HashSet};

use chrono::Utc;
use sqlx::Row;

use crate::core::attention_model::AttentionModel;
use crate::core::qualia::QualiaState;
use crate::core::workspace::WorkspaceState;
use crate::db::Db;
use crate::models::{AttentionSchemaState, AttentionSourceAttribution};

const SUPPRESSED_ITEMS_LIMIT: usize = 6;
const SOURCE_ATTRIBUTION_LIMIT: usize = 6;

fn normalize_source_label(source: &str) -> String {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return "unknown".to_string();
    }
    if let Some(prefix) = trimmed.strip_prefix("qualia:") {
        let label = prefix.trim();
        if label.is_empty() {
            return "qualia".to_string();
        }
        return format!("qualia:{}", label);
    }
    trimmed.to_string()
}

fn compute_stability(previous_focus: Option<&[String]>, current_focus: &[String]) -> f64 {
    let Some(previous_focus) = previous_focus else {
        return 0.5;
    };
    if previous_focus.is_empty() || current_focus.is_empty() {
        return 0.5;
    }
    let prev: HashSet<&str> = previous_focus.iter().map(|s| s.as_str()).collect();
    let curr: HashSet<&str> = current_focus.iter().map(|s| s.as_str()).collect();
    let union = prev.union(&curr).count();
    if union == 0 {
        return 1.0;
    }
    let intersection = prev.intersection(&curr).count();
    (intersection as f64 / union as f64).clamp(0.0, 1.0)
}

async fn compute_suppressed_items(
    db: &Db,
    focus_refs: &[String],
) -> Vec<String> {
    let rows = sqlx::query(
        "SELECT item_id, item_type
         FROM ics_working_set
         ORDER BY activation DESC
         LIMIT ?",
    )
    .bind(SUPPRESSED_ITEMS_LIMIT as i64)
    .fetch_all(&db.pool)
    .await
    .unwrap_or_default();

    let focus: HashSet<&str> = focus_refs.iter().map(|s| s.as_str()).collect();
    let mut suppressed = Vec::new();
    for row in rows {
        let item_id: i64 = row.try_get("item_id").unwrap_or(0);
        let item_type: String = row.try_get("item_type").unwrap_or_else(|_| "entity".to_string());
        if item_id <= 0 {
            continue;
        }
        let label = format!("{}:{}", item_type, item_id);
        if focus.contains(label.as_str()) {
            continue;
        }
        suppressed.push(label);
        if suppressed.len() >= SUPPRESSED_ITEMS_LIMIT {
            break;
        }
    }
    suppressed
}

pub async fn compute_attention_schema(
    db: &Db,
    workspace: &WorkspaceState,
    attention: &AttentionModel,
    _qualia: &QualiaState,
    previous_focus_refs: Option<&[String]>,
) -> Result<AttentionSchemaState, String> {
    let slots = workspace.slots.max(1) as f64;
    let focus_len = attention.current_focus_refs.len() as f64;
    let capacity_usage = (focus_len / slots).clamp(0.0, 1.0);

    let mut source_weights: HashMap<String, f64> = HashMap::new();
    for reason in attention.why_focused.iter() {
        let label = normalize_source_label(&reason.source);
        let weight = reason.weight.max(0.0);
        *source_weights.entry(label).or_insert(0.0) += weight;
    }
    let mut sources: Vec<(String, f64)> = source_weights.into_iter().collect();
    sources.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let selection_policy = sources
        .first()
        .map(|(label, _)| label.clone())
        .unwrap_or_else(|| "none".to_string());

    let total_weight: f64 = sources.iter().map(|(_, w)| *w).sum();
    let mut source_attribution = Vec::new();
    for (label, weight) in sources.into_iter().take(SOURCE_ATTRIBUTION_LIMIT) {
        let share = if total_weight > 0.0 {
            (weight / total_weight).clamp(0.0, 1.0)
        } else {
            0.0
        };
        source_attribution.push(AttentionSourceAttribution {
            source: label,
            weight,
            share,
        });
    }

    let suppressed_items = compute_suppressed_items(db, &attention.current_focus_refs).await;
    let stability = compute_stability(previous_focus_refs, &attention.current_focus_refs);

    Ok(AttentionSchemaState {
        capacity_usage,
        selection_policy,
        suppressed_items,
        stability,
        source_attribution,
        last_updated_at: Utc::now().to_rfc3339(),
    })
}

pub fn summarize_for_prompt(schema: &AttentionSchemaState) -> String {
    let mut parts = Vec::new();
    parts.push(format!("capacity_usage={:.2}", schema.capacity_usage));
    if !schema.selection_policy.trim().is_empty() {
        parts.push(format!("selection_policy={}", schema.selection_policy));
    }
    parts.push(format!("stability={:.2}", schema.stability));

    if !schema.suppressed_items.is_empty() {
        let items = schema
            .suppressed_items
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        parts.push(format!("suppressed=[{}]", items));
    }
    if !schema.source_attribution.is_empty() {
        let sources = schema
            .source_attribution
            .iter()
            .take(3)
            .map(|s| format!("{}:{:.2}", s.source, s.share))
            .collect::<Vec<_>>()
            .join(", ");
        parts.push(format!("sources={}", sources));
    }

    parts.join(" | ")
}
