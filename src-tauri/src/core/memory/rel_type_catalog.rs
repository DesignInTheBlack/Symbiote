use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use crate::core::memory::canonical::{compute_anchor_signature, compute_signature_hash, canonicalize_participants, normalize_rel_type, serialize_participant_ids};
use crate::core::memory::config::{REL_TYPE_ALIAS_MIN_CONFIDENCE, REL_TYPE_SIMILARITY_THRESHOLD, REL_TYPE_PROMOTE_MIN_EDGES};
use crate::core::memory::rel_vocab::is_canonical_relation;
use strsim::jaro_winkler;

#[derive(Debug, Clone)]
pub struct RelTypeResolution {
    pub rel_type_id: String,
    pub rel_type_norm: String,
    pub rel_type_raw: String,
}

fn alias_confident(status: &str, confidence: f32) -> bool {
    match status {
        "confirmed" => true,
        "provisional" => confidence >= REL_TYPE_ALIAS_MIN_CONFIDENCE,
        _ => false,
    }
}

async fn lookup_legacy_alias(pool: &SqlitePool, alias: &str) -> Option<String> {
    let row = sqlx::query(
        "SELECT rel_type, confidence, status FROM ics_rel_type_aliases WHERE alias = ? LIMIT 1"
    )
    .bind(alias)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()?;

    let rel_type: String = row.try_get("rel_type").unwrap_or_else(|_| alias.to_string());
    let status: String = row.try_get("status").unwrap_or_else(|_| "confirmed".to_string());
    let confidence: f32 = row.try_get::<f64, _>("confidence").unwrap_or(1.0) as f32;
    if status == "confirmed" || (status == "provisional" && confidence >= REL_TYPE_ALIAS_MIN_CONFIDENCE) {
        return Some(normalize_rel_type(&rel_type));
    }
    None
}

async fn resolve_merged_rel_type(pool: &SqlitePool, rel_type_id: &str) -> String {
    let mut current = rel_type_id.to_string();
    let mut hops = 0;
    loop {
        if hops > 5 {
            break;
        }
        let row = sqlx::query("SELECT merged_into FROM rel_type WHERE rel_type_id = ?")
            .bind(&current)
            .fetch_optional(pool)
            .await
            .ok()
            .flatten();
        if let Some(r) = row {
            let merged: Option<String> = r.try_get("merged_into").ok();
            if let Some(next) = merged {
                if next == current {
                    break;
                }
                current = next;
                hops += 1;
                continue;
            }
        }
        break;
    }
    current
}

async fn lookup_alias(pool: &SqlitePool, alias: &str) -> Option<(String, f32, String)> {
    let row = sqlx::query(
        "SELECT rel_type_id, confidence, status FROM rel_type_alias WHERE alias = ? LIMIT 1"
    )
    .bind(alias)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()?;

    let rel_type_id: String = row.try_get("rel_type_id").ok()?;
    if rel_type_id.trim().is_empty() {
        return None;
    }
    let confidence: f32 = row.try_get::<f64, _>("confidence").unwrap_or(1.0) as f32;
    let status: String = row.try_get("status").unwrap_or_else(|_| "confirmed".to_string());
    Some((rel_type_id, confidence, status))
}

async fn lookup_canonical(pool: &SqlitePool, canonical_name: &str) -> Option<String> {
    let row = sqlx::query("SELECT rel_type_id, merged_into FROM rel_type WHERE canonical_name = ? LIMIT 1")
        .bind(canonical_name)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()?;
    let rel_type_id: String = row.try_get("rel_type_id").ok()?;
    if rel_type_id.trim().is_empty() {
        return None;
    }
    let merged_into: Option<String> = row.try_get("merged_into").ok();
    if let Some(merged) = merged_into {
        return Some(merged);
    }
    Some(rel_type_id)
}

async fn lookup_canonical_name(pool: &SqlitePool, rel_type_id: &str) -> Option<String> {
    let row = sqlx::query("SELECT canonical_name FROM rel_type WHERE rel_type_id = ? LIMIT 1")
        .bind(rel_type_id)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()?;
    let canonical_name: String = row.try_get("canonical_name").ok()?;
    Some(canonical_name)
}

async fn load_shape_roles(pool: &SqlitePool, rel_type_id: &str) -> Option<(Vec<String>, Option<i64>)> {
    let row = sqlx::query("SELECT roles, expected_arity FROM rel_shape WHERE rel_type_id = ?")
        .bind(rel_type_id)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()?;
    let roles_raw: String = row.try_get("roles").ok().unwrap_or_else(|| "[]".to_string());
    let roles = serde_json::from_str::<Vec<String>>(&roles_raw).unwrap_or_default();
    let expected_arity: Option<i64> = row.try_get("expected_arity").ok();
    Some((roles, expected_arity))
}

fn roles_equivalent(roles_seen: &[String], candidate_roles: &[String], expected_arity: Option<i64>) -> bool {
    if let Some(expected) = expected_arity {
        if expected as usize != roles_seen.len() {
            return false;
        }
    }
    if roles_seen.is_empty() || candidate_roles.is_empty() {
        return true;
    }
    let mut a = roles_seen.to_vec();
    let mut b = candidate_roles.to_vec();
    a.sort();
    b.sort();
    a == b
}

fn tokenize_rel_type(name: &str) -> std::collections::HashSet<String> {
    let mut set = std::collections::HashSet::new();
    for token in name.split('_') {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed {
            "of" | "to" | "from" | "for" | "in" | "on" | "at" | "is" | "has" => continue,
            _ => {}
        }
        set.insert(trimmed.to_string());
    }
    set
}

fn has_token_overlap(a: &str, b: &str) -> bool {
    let a_tokens = tokenize_rel_type(a);
    let b_tokens = tokenize_rel_type(b);
    if a_tokens.is_empty() || b_tokens.is_empty() {
        return false;
    }
    a_tokens.intersection(&b_tokens).next().is_some()
}

async fn find_similar_candidate(
    pool: &SqlitePool,
    rel_type_norm: &str,
    roles_seen: &[String],
) -> Option<(String, f32)> {
    let rows = sqlx::query("SELECT rel_type_id, canonical_name FROM rel_type WHERE status IN ('canonical', 'provisional')")
        .fetch_all(pool)
        .await
        .ok()?;

    let mut best_id: Option<String> = None;
    let mut best_score = 0.0f32;
    for row in rows {
        let rel_type_id: String = row.try_get("rel_type_id").ok()?;
        let canonical_name: String = row.try_get("canonical_name").unwrap_or_default();
        let score = jaro_winkler(rel_type_norm, &canonical_name) as f32;
        if score > best_score {
            best_score = score;
            best_id = Some(rel_type_id);
        }
    }

    if best_score < REL_TYPE_SIMILARITY_THRESHOLD {
        return None;
    }

    if let Some(candidate_id) = best_id {
        if let Some((roles, expected_arity)) = load_shape_roles(pool, &candidate_id).await {
            if !roles_equivalent(roles_seen, &roles, expected_arity) {
                return None;
            }
        }
        let candidate_name = sqlx::query("SELECT canonical_name FROM rel_type WHERE rel_type_id = ? LIMIT 1")
            .bind(&candidate_id)
            .fetch_optional(pool)
            .await
            .ok()
            .flatten()
            .and_then(|r| r.try_get::<String, _>("canonical_name").ok())
            .unwrap_or_default();
        if !candidate_name.is_empty() && !has_token_overlap(rel_type_norm, &candidate_name) {
            return None;
        }
        return Some((candidate_id, best_score));
    }
    None
}

async fn insert_rel_type(
    pool: &SqlitePool,
    canonical_name: &str,
    status: &str,
) -> Result<String, String> {
    let rel_type_id = Uuid::new_v4().to_string();
    let insert_res = sqlx::query(
        "INSERT INTO rel_type (rel_type_id, canonical_name, status, created_at)
         VALUES (?, ?, ?, CURRENT_TIMESTAMP)"
    )
    .bind(&rel_type_id)
    .bind(canonical_name)
    .bind(status)
    .execute(pool)
    .await;
    if insert_res.is_ok() {
        return Ok(rel_type_id);
    }
    if let Some(existing) = lookup_canonical(pool, canonical_name).await {
        let resolved = resolve_merged_rel_type(pool, &existing).await;
        if !resolved.trim().is_empty() {
            return Ok(resolved);
        }
    }
    let _ = sqlx::query(
        "INSERT OR IGNORE INTO rel_type (rel_type_id, canonical_name, status, created_at)
         VALUES (?, ?, ?, CURRENT_TIMESTAMP)"
    )
    .bind(&rel_type_id)
    .bind(canonical_name)
    .bind(status)
    .execute(pool)
    .await;
    Ok(rel_type_id)
}

async fn ensure_alias(
    pool: &SqlitePool,
    alias: &str,
    rel_type_id: &str,
    confidence: f32,
    status: &str,
) {
    let _ = sqlx::query(
        "INSERT OR IGNORE INTO rel_type_alias (alias, rel_type_id, confidence, status, created_at)
         VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)"
    )
    .bind(alias)
    .bind(rel_type_id)
    .bind(confidence)
    .bind(status)
    .execute(pool)
    .await;
}

async fn ensure_raw_alias(
    pool: &SqlitePool,
    raw_norm: &str,
    rel_type_norm: &str,
    rel_type_id: &str,
    canonicalization_on: bool,
) {
    if !canonicalization_on {
        return;
    }
    if raw_norm == rel_type_norm {
        return;
    }
    ensure_alias(pool, raw_norm, rel_type_id, 1.0, "confirmed").await;
}

pub async fn resolve_rel_type(
    pool: &SqlitePool,
    rel_type_raw: &str,
    roles_seen: &[String],
    canonicalization_on: bool,
) -> Result<RelTypeResolution, String> {
    let raw_norm = normalize_rel_type(rel_type_raw);
    let mut rel_type_norm = raw_norm.clone();

    if canonicalization_on {
        if let Some((rel_type_id, confidence, status)) = lookup_alias(pool, &raw_norm).await {
            if alias_confident(&status, confidence) {
                let resolved_id = resolve_merged_rel_type(pool, &rel_type_id).await;
                let canonical_name = lookup_canonical_name(pool, &resolved_id).await.unwrap_or_else(|| rel_type_norm.clone());
                ensure_raw_alias(pool, &raw_norm, &rel_type_norm, &resolved_id, canonicalization_on).await;
                return Ok(RelTypeResolution {
                    rel_type_id: resolved_id,
                    rel_type_norm: canonical_name,
                    rel_type_raw: rel_type_raw.to_string(),
                });
            }
        }

        if let Some(legacy) = lookup_legacy_alias(pool, &raw_norm).await {
            rel_type_norm = legacy;
        }

        if let Some((rel_type_id, confidence, status)) = lookup_alias(pool, &rel_type_norm).await {
            if alias_confident(&status, confidence) {
                let resolved_id = resolve_merged_rel_type(pool, &rel_type_id).await;
                let canonical_name = lookup_canonical_name(pool, &resolved_id).await.unwrap_or_else(|| rel_type_norm.clone());
                ensure_raw_alias(pool, &raw_norm, &rel_type_norm, &resolved_id, canonicalization_on).await;
                return Ok(RelTypeResolution {
                    rel_type_id: resolved_id,
                    rel_type_norm: canonical_name,
                    rel_type_raw: rel_type_raw.to_string(),
                });
            }
        }

        if let Some(rel_type_id) = lookup_canonical(pool, &rel_type_norm).await {
            let resolved_id = resolve_merged_rel_type(pool, &rel_type_id).await;
            ensure_raw_alias(pool, &raw_norm, &rel_type_norm, &resolved_id, canonicalization_on).await;
            return Ok(RelTypeResolution {
                rel_type_id: resolved_id,
                rel_type_norm: rel_type_norm.clone(),
                rel_type_raw: rel_type_raw.to_string(),
            });
        }

    if let Some((candidate_id, score)) = find_similar_candidate(pool, &rel_type_norm, roles_seen).await {
        let resolved_id = resolve_merged_rel_type(pool, &candidate_id).await;
        ensure_alias(pool, &rel_type_norm, &resolved_id, score, "provisional").await;
        ensure_raw_alias(pool, &raw_norm, &rel_type_norm, &resolved_id, canonicalization_on).await;
        let canonical_name = lookup_canonical_name(pool, &resolved_id).await.unwrap_or_else(|| rel_type_norm.clone());
        return Ok(RelTypeResolution {
            rel_type_id: resolved_id,
            rel_type_norm: canonical_name,
            rel_type_raw: rel_type_raw.to_string(),
        });
    }
    }

    if canonicalization_on && !is_canonical_relation(&rel_type_norm) {
        return Err(format!("RelTypeNotAllowed: {}", rel_type_norm));
    }

    // Create new rel_type
    let mut rel_type_id = insert_rel_type(pool, &rel_type_norm, "provisional")
        .await
        .unwrap_or_else(|_| Uuid::new_v4().to_string());
    if rel_type_id.trim().is_empty() {
        rel_type_id = Uuid::new_v4().to_string();
    }
    ensure_alias(pool, &rel_type_norm, &rel_type_id, 1.0, "confirmed").await;
    ensure_raw_alias(pool, &raw_norm, &rel_type_norm, &rel_type_id, canonicalization_on).await;
    Ok(RelTypeResolution {
        rel_type_id,
        rel_type_norm,
        rel_type_raw: rel_type_raw.to_string(),
    })
}

pub async fn resolve_rel_type_from_id(
    pool: &SqlitePool,
    rel_type_id: &str,
    rel_type_raw: &str,
    canonicalization_on: bool,
) -> Result<RelTypeResolution, String> {
    let rel_type_id = rel_type_id.trim();
    if rel_type_id.is_empty() {
        return resolve_rel_type(pool, rel_type_raw, &[], canonicalization_on).await;
    }

    let resolved_id = resolve_merged_rel_type(pool, rel_type_id).await;
    let rel_type_norm = lookup_canonical_name(pool, &resolved_id)
        .await
        .unwrap_or_else(|| normalize_rel_type(rel_type_raw));
    let raw_norm = normalize_rel_type(rel_type_raw);
    ensure_raw_alias(pool, &raw_norm, &rel_type_norm, &resolved_id, canonicalization_on).await;

    Ok(RelTypeResolution {
        rel_type_id: resolved_id,
        rel_type_norm,
        rel_type_raw: rel_type_raw.to_string(),
    })
}

pub async fn merge_rel_type(pool: &SqlitePool, from_id: &str, into_id: &str) -> Result<(), String> {
    if from_id == into_id {
        return Ok(());
    }
    let mut tx = pool.begin().await.map_err(|e| e.to_string())?;
    let _ = sqlx::query("UPDATE rel_type SET merged_into = ?, status = 'deprecated' WHERE rel_type_id = ?")
        .bind(into_id)
        .bind(from_id)
        .execute(&mut *tx)
        .await;

    let _ = sqlx::query("UPDATE rel_type_alias SET rel_type_id = ? WHERE rel_type_id = ?")
        .bind(into_id)
        .bind(from_id)
        .execute(&mut *tx)
        .await;

    let _ = sqlx::query("UPDATE ics_rel_beliefs SET rel_type_id = ? WHERE rel_type_id = ?")
        .bind(into_id)
        .bind(from_id)
        .execute(&mut *tx)
        .await;

    let _ = tx.commit().await;

    // Recompute signatures for affected rels and dedup.
    let _ = recompute_rel_signatures(pool, into_id).await;
    let _ = dedup_rel_beliefs(pool, into_id).await;
    Ok(())
}

async fn recompute_rel_signatures(pool: &SqlitePool, rel_type_id: &str) -> Result<(), String> {
    let rows = sqlx::query(
        "SELECT b.id, b.scope, b.polarity, b.time_bucket_kind, b.time_bucket_value, rb.direction
         FROM ics_beliefs b
         JOIN ics_rel_beliefs rb ON rb.belief_id = b.id
         WHERE rb.rel_type_id = ? AND b.kind = 'rel'"
    )
    .bind(rel_type_id)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    if rows.is_empty() {
        return Ok(());
    }

    let roles_row = sqlx::query("SELECT anchor_roles FROM rel_shape WHERE rel_type_id = ?")
        .bind(rel_type_id)
        .fetch_optional(pool)
        .await
        .map_err(|e| e.to_string())?;
    let anchor_roles: Vec<String> = roles_row
        .and_then(|r| r.try_get::<String, _>("anchor_roles").ok())
        .and_then(|raw| serde_json::from_str::<Vec<String>>(&raw).ok())
        .unwrap_or_default();

    let mut tx = pool.begin().await.map_err(|e| e.to_string())?;

    for row in rows {
        let id: i64 = row.get("id");
        let scope: String = row.get("scope");
        let polarity: String = row.get("polarity");
        let time_bucket_kind: String = row.try_get("time_bucket_kind").unwrap_or_else(|_| "atemporal".to_string());
        let time_bucket_value: Option<String> = row.try_get("time_bucket_value").ok();
        let time_bucket_value_sig = time_bucket_value.clone().unwrap_or_default();
        let direction: Option<String> = row.try_get("direction").ok();

        let p_rows = sqlx::query("SELECT role, entity_id FROM ics_rel_participants WHERE belief_id = ?")
            .bind(id)
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| e.to_string())?;
        let participants = p_rows
            .into_iter()
            .map(|r| (r.get::<String, _>("role"), r.get::<i64, _>("entity_id")))
            .collect::<Vec<_>>();

        let participants_canonical = canonicalize_participants(&participants);
        let participants_id_sig = serialize_participant_ids(&participants);
        let anchor_signature = compute_anchor_signature(&anchor_roles, &participants, false);
        let topic_key = format!("rel:{}:{}", rel_type_id, anchor_signature);

        let mut sig_inputs = vec![
            ("rel_type_id".to_string(), rel_type_id.to_string()),
            ("participants".to_string(), participants_id_sig),
            ("arity".to_string(), participants.len().to_string()),
            ("scope".to_string(), scope.clone()),
            ("time_bucket_kind".to_string(), time_bucket_kind.clone()),
            ("time_bucket_value".to_string(), time_bucket_value_sig),
            ("polarity".to_string(), polarity.clone()),
        ];
        if let Some(direction) = direction.as_deref().filter(|v| !v.is_empty()) {
            sig_inputs.push(("direction".to_string(), direction.to_string()));
        }
        let sig_refs: Vec<(&str, &str)> = sig_inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let signature_hash = compute_signature_hash(&sig_refs);

        let _ = sqlx::query("UPDATE ics_beliefs SET topic_key = ?, signature_hash = ? WHERE id = ?")
            .bind(&topic_key)
            .bind(&signature_hash)
            .bind(id)
            .execute(&mut *tx)
            .await;

        let _ = sqlx::query("UPDATE ics_rel_beliefs SET participants_canonical = ?, anchor_signature = ? WHERE belief_id = ?")
            .bind(&participants_canonical)
            .bind(&anchor_signature)
            .bind(id)
            .execute(&mut *tx)
            .await;
    }

    let _ = tx.commit().await;
    Ok(())
}

async fn dedup_rel_beliefs(pool: &SqlitePool, rel_type_id: &str) -> Result<(), String> {
    let rows = sqlx::query(
        "SELECT b.signature_hash, GROUP_CONCAT(b.id) as ids
         FROM ics_beliefs b
         JOIN ics_rel_beliefs rb ON rb.belief_id = b.id
         WHERE rb.rel_type_id = ? AND b.kind = 'rel' AND b.status = 'active'
         GROUP BY b.signature_hash
         HAVING COUNT(*) > 1"
    )
    .bind(rel_type_id)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    for row in rows {
        let ids_raw: String = row.try_get("ids").unwrap_or_default();
        let ids: Vec<i64> = ids_raw
            .split(',')
            .filter_map(|v| v.parse::<i64>().ok())
            .collect();
        if ids.len() < 2 {
            continue;
        }

        let mut best_id: Option<i64> = None;
        let mut best_weight: f64 = -1.0;
        for id in &ids {
            if let Ok(row) = sqlx::query("SELECT evidence_weight_total FROM ics_beliefs WHERE id = ?")
                .bind(id)
                .fetch_one(pool)
                .await
            {
                let weight: f64 = row.try_get("evidence_weight_total").unwrap_or(0.0);
                if weight > best_weight {
                    best_weight = weight;
                    best_id = Some(*id);
                }
            }
        }

        for id in ids {
            if Some(id) == best_id {
                continue;
            }
            let _ = sqlx::query("UPDATE ics_beliefs SET status = 'inactive' WHERE id = ?")
                .bind(id)
                .execute(pool)
                .await;
        }
    }

    Ok(())
}

pub async fn curate_rel_types(pool: &SqlitePool) -> Result<(), String> {
    let rows = sqlx::query(
        "SELECT rt.rel_type_id, rt.canonical_name,
                COALESCE(COUNT(rb.belief_id), 0) as edge_count
         FROM rel_type rt
         LEFT JOIN ics_rel_beliefs rb ON rb.rel_type_id = rt.rel_type_id
         WHERE rt.status = 'provisional'
         GROUP BY rt.rel_type_id"
    )
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    for row in rows {
        let rel_type_id: String = row.try_get("rel_type_id").unwrap_or_default();
        let canonical_name: String = row.try_get("canonical_name").unwrap_or_default();
        let edge_count: i64 = row.try_get::<i64, _>("edge_count").unwrap_or(0);

        if let Some((candidate_id, _score)) = find_similar_candidate(pool, &canonical_name, &[]).await {
            // Merge into closest canonical candidate if equivalent.
            let _ = merge_rel_type(pool, &rel_type_id, &candidate_id).await;
            continue;
        }

        if edge_count >= REL_TYPE_PROMOTE_MIN_EDGES as i64 {
            let _ = sqlx::query("UPDATE rel_type SET status = 'canonical' WHERE rel_type_id = ?")
                .bind(&rel_type_id)
                .execute(pool)
                .await;
        }
    }
    Ok(())
}
