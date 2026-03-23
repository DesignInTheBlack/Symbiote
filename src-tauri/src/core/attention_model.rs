use serde::{Deserialize, Serialize};
use sqlx::Row;

use crate::db::Db;
use crate::core::qualia::QualiaState;
use crate::core::workspace::WorkspaceState;
use crate::core::cognitive_wave;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionReason {
    pub source: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionModel {
    pub timestamp: String,
    pub current_focus_refs: Vec<String>,
    pub why_focused: Vec<AttentionReason>,
    pub meta_confidence: f64,
    pub next_focus_prediction: Option<String>,
}

pub async fn compute_attention_model(
    db: &Db,
    conversation_id: &str,
    workspace: &WorkspaceState,
    qualia: &QualiaState,
) -> Result<AttentionModel, String> {
    let mut focus_refs = workspace.broadcast_refs.clone();
    let mut reasons = Vec::new();
    if focus_refs.is_empty() {
        let rows = sqlx::query(
            "SELECT item_id, item_type, activation
             FROM ics_working_set
             ORDER BY activation DESC
             LIMIT 3",
        )
        .fetch_all(&db.pool)
        .await
        .unwrap_or_default();
        for row in rows {
            let item_id: i64 = row.try_get("item_id").unwrap_or(0);
            let item_type: String = row.try_get("item_type").unwrap_or_else(|_| "entity".to_string());
            let activation: f64 = row.try_get("activation").unwrap_or(0.0);
            focus_refs.push(format!("{}:{}", item_type, item_id));
            reasons.push(AttentionReason {
                source: "working_set".to_string(),
                weight: activation,
            });
        }
    } else {
        for _ in focus_refs.iter() {
            reasons.push(AttentionReason {
                source: "workspace_broadcast".to_string(),
                weight: 0.8,
            });
        }
    }

    if let Some(tag) = qualia.predicted_tag.as_ref().or(qualia.dominant_tag.as_ref()) {
        let weight = (0.2 + qualia.dominant_intensity * 0.5 + qualia.prediction_confidence * 0.3)
            .clamp(0.0, 1.0);
        reasons.push(AttentionReason {
            source: format!("qualia:{}", tag),
            weight,
        });
        if focus_refs.is_empty() {
            focus_refs.push(format!("qualia:{}", tag));
        }
    }

    if focus_refs.is_empty() {
        let message_id: Option<String> = sqlx::query_scalar(
            "SELECT message_id FROM messages
             WHERE conversation_id = ?
               AND role IN ('user','assistant')
               AND status = 'complete'
               AND (metadata IS NULL OR json_extract(metadata, '$.source') != 'monologue')
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&db.pool)
        .await
        .ok()
        .flatten();
        if let Some(message_id) = message_id {
            focus_refs.push(format!("message:{}", message_id));
            reasons.push(AttentionReason {
                source: "message_fallback".to_string(),
                weight: 0.2,
            });
        }
    }

    let meta_confidence = if reasons.is_empty() {
        0.4
    } else {
        let sum: f64 = reasons.iter().map(|r| r.weight).sum();
        (sum / reasons.len() as f64).clamp(0.0, 1.0)
    };
    let next_focus_prediction = focus_refs.first().cloned();

    cognitive_wave::update_gain_from_attention(
        &db.pool,
        None,
        meta_confidence,
        &["attention_model"],
    )
    .await;

    Ok(AttentionModel {
        timestamp: chrono::Utc::now().to_rfc3339(),
        current_focus_refs: focus_refs,
        why_focused: reasons,
        meta_confidence,
        next_focus_prediction,
    })
}
