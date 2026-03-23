use sqlx::{SqlitePool, Row};
use tauri::State;
use serde::Serialize;
use crate::core::memory::canonical::canonicalize_string;
use crate::core::memory::dsl::{self, DslStatement, Ref};
use crate::core::memory::cache;
use std::collections::HashMap;

fn ref_to_label(r: &Ref) -> String {
    match r {
        Ref::Handle(handle) => format!("${}", handle),
        Ref::Label(label) => label.clone(),
        Ref::Filter(label, _key) => label.clone(),
        Ref::Name(name) => name.clone(),
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphNode {
    pub id: String,
    pub node_type: String,
    pub label: String,
    pub entity_type: Option<String>,
    pub confidence: Option<f32>,
    pub access_count: i64,
    pub activation: Option<f64>,  // Working set activation (0.0-1.0)
    pub last_accessed: Option<String>,
    pub key: Option<String>,
    pub value: Option<String>,
    // Phase 4: Temporal and scope fields
    pub scope: Option<String>,
    pub polarity: Option<String>,
    pub time_bucket_kind: Option<String>,
    pub time_bucket_value: Option<String>,
    // Relation participants (role, entity_name)
    pub participants: Option<Vec<(String, String)>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphLink {
    pub source: String,
    pub target: String,
    pub link_type: String,
    pub label: Option<String>,
    pub strength: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryGraph {
    pub nodes: Vec<GraphNode>,
    pub links: Vec<GraphLink>,
}

#[tauri::command]
pub async fn memory_get_graph(
    pool: State<'_, SqlitePool>,
    limit: Option<i64>,
) -> Result<MemoryGraph, String> {
    let limit = limit.unwrap_or(0);
    let mut nodes = Vec::new();
    let mut links = Vec::new();
    
    // 1. Get entities with access counts (from working set activation)
    // We order by activation desc, then recent access
    let entities = if limit > 0 {
        sqlx::query(
            "SELECT e.id, e.label, e.entity_type, e.last_accessed_at, e.access_count,
                    COALESCE(w.activation, 0) as activation
             FROM ics_entities e
             LEFT JOIN ics_working_set w ON w.item_id = e.id AND w.item_type = 'entity'
             WHERE e.resolution_state NOT IN ('merged', 'deleted')
             ORDER BY activation DESC, e.last_accessed_at DESC
             LIMIT ?"
        )
        .bind(limit)
        .fetch_all(&*pool)
        .await
        .map_err(|e| e.to_string())?
    } else {
        sqlx::query(
            "SELECT e.id, e.label, e.entity_type, e.last_accessed_at, e.access_count,
                    COALESCE(w.activation, 0) as activation
             FROM ics_entities e
             LEFT JOIN ics_working_set w ON w.item_id = e.id AND w.item_type = 'entity'
             WHERE e.resolution_state NOT IN ('merged', 'deleted')
             ORDER BY activation DESC, e.last_accessed_at DESC"
        )
        .fetch_all(&*pool)
        .await
        .map_err(|e| e.to_string())?
    };
    
    let entity_ids: Vec<i64> = entities.iter().map(|r| r.get("id")).collect();
    
    // Build a lookup map: entity_id -> label for resolving participant names
    let entity_labels: HashMap<i64, String> = entities
        .iter()
        .map(|r| (r.get::<i64, _>("id"), r.get::<String, _>("label")))
        .collect();
    
    for row in &entities {
        let id: i64 = row.get("id");
        nodes.push(GraphNode {
            id: format!("entity_{}", id),
            node_type: "entity".to_string(),
            label: row.get("label"),
            entity_type: row.try_get("entity_type").ok(),
            confidence: None,
            access_count: row.try_get::<i64, _>("access_count").unwrap_or(0),
            activation: Some(row.try_get::<f64, _>("activation").unwrap_or(0.0)),
            last_accessed: row.try_get("last_accessed_at").ok(),
            key: None,
            value: None,
            // Entities don't have temporal/scope info
            scope: None,
            polarity: None,
            time_bucket_kind: None,
            time_bucket_value: None,
            participants: None,
        });
    }
    
    if !entity_ids.is_empty() {
        // 2. Get facts for these entities (with temporal/scope data)
        let placeholders: String = entity_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let query_facts = format!(
            "SELECT b.id, b.confidence, b.scope, b.polarity, b.time_bucket_kind, b.time_bucket_value,
                    fb.subject_entity_id, fb.key, fb.value_literal
             FROM ics_beliefs b
             JOIN ics_fact_beliefs fb ON fb.belief_id = b.id
             WHERE fb.subject_entity_id IN ({})
               AND b.status = 'active'
               AND EXISTS (
                   SELECT 1 FROM ics_evidence_events e
                   WHERE e.belief_id = b.id
                     AND e.source_type IN ('user', 'tool', 'inference', 'user_identity')
               )",
            placeholders
        );
        
        let mut q_facts = sqlx::query(&query_facts);
        for id in &entity_ids {
            q_facts = q_facts.bind(id);
        }
        
        let facts = q_facts.fetch_all(&*pool).await.map_err(|e| e.to_string())?;
        
        for row in facts {
            let belief_id: i64 = row.get("id");
            let entity_id: i64 = row.get("subject_entity_id");
            
            nodes.push(GraphNode {
                id: format!("fact_{}", belief_id),
                node_type: "fact".to_string(),
                label: format!("{}: {}", row.get::<String, _>("key"), 
                    row.get::<String, _>("value_literal").chars().take(30).collect::<String>()),
                entity_type: None,
                confidence: row.try_get("confidence").ok(),
                access_count: 0,
                activation: None,
                last_accessed: None,
                key: Some(row.get("key")),
                value: Some(row.get("value_literal")),
                // Temporal and scope data from beliefs table
                scope: row.try_get("scope").ok(),
                polarity: row.try_get("polarity").ok(),
                time_bucket_kind: row.try_get("time_bucket_kind").ok(),
                time_bucket_value: row.try_get("time_bucket_value").ok(),
                participants: None,
            });
            
            links.push(GraphLink {
                source: format!("entity_{}", entity_id),
                target: format!("fact_{}", belief_id),
                link_type: "has_fact".to_string(),
                label: None,
                strength: 0.3,
            });
        }

        // 3. Get relationships BETWEEN these entities
        // We only fetch relations where BOTH participants are in our fetched set to avoid dangling links
        // 3. Get relationships as nodes (hypergraph representation)
        // This creates relationship nodes connected to all participants with role labels
        let query_rels = format!(
            "SELECT rb.belief_id, rb.rel_type, b.confidence, b.scope, b.polarity,
                    GROUP_CONCAT(rp.role || ':' || CAST(rp.entity_id AS TEXT), '|') as participants
             FROM ics_rel_beliefs rb
             JOIN ics_beliefs b ON b.id = rb.belief_id
             JOIN ics_rel_participants rp ON rp.belief_id = rb.belief_id
             WHERE b.status = 'active'
               AND EXISTS (
                   SELECT 1 FROM ics_rel_participants rp2 
                   WHERE rp2.belief_id = rb.belief_id 
                   AND rp2.entity_id IN ({})
               )
               AND EXISTS (
                   SELECT 1 FROM ics_evidence_events e
                   WHERE e.belief_id = rb.belief_id
                     AND e.source_type IN ('user', 'tool', 'inference', 'user_identity')
               )
             GROUP BY rb.belief_id",
            placeholders
        );
        
        let mut q_rels = sqlx::query(&query_rels);
        for id in &entity_ids {
            q_rels = q_rels.bind(id);
        }
        
        if let Ok(rels) = q_rels.fetch_all(&*pool).await {
            for row in rels {
                let belief_id: i64 = row.get("belief_id");
                let rel_type: String = row.get("rel_type");
                let participants_str: String = row.try_get("participants").unwrap_or_default();
                let confidence: Option<f32> = row.try_get("confidence").ok();
                
                // Parse participants: "role1:id1|role2:id2|role3:id3"
                let participants: Vec<(String, i64)> = participants_str
                    .split('|')
                    .filter_map(|p| {
                        let parts: Vec<&str> = p.split(':').collect();
                        if parts.len() == 2 {
                            if let Ok(id) = parts[1].parse::<i64>() {
                                return Some((parts[0].to_string(), id));
                            }
                        }
                        None
                    })
                    .collect();
                
                // Build participant list with names (resolve entity_id -> label)
                let participant_names: Vec<(String, String)> = participants
                    .iter()
                    .map(|(role, eid)| {
                        let name = entity_labels.get(eid).cloned().unwrap_or_else(|| format!("Entity #{}", eid));
                        (role.clone(), name)
                    })
                    .collect();
                
                // Add relationship as a node
                nodes.push(GraphNode {
                    id: format!("rel_{}", belief_id),
                    node_type: "relation".to_string(),
                    label: rel_type.clone(),
                    entity_type: None,
                    confidence,
                    access_count: 0,
                    activation: None,
                    last_accessed: None,
                    key: Some(format!("{} participants", participants.len())),
                    value: None,
                    // Relations can have scope from the belief
                    scope: row.try_get("scope").ok(),
                    polarity: row.try_get("polarity").ok(),
                    time_bucket_kind: None,
                    time_bucket_value: None,
                    participants: Some(participant_names),
                });
                
                // Create links from relationship node to each participant
                for (role, entity_id) in participants {
                    if entity_ids.contains(&entity_id) {
                        links.push(GraphLink {
                            source: format!("rel_{}", belief_id),
                            target: format!("entity_{}", entity_id),
                            link_type: "has_participant".to_string(),
                            label: Some(role),
                            strength: 0.8,
                        });
                    }
                }
            }
        }
        
        // 4. Get belief links for visualizing graph semantics
        let query_links = format!(
            "SELECT bl.from_id, bl.to_id, bl.link_type,
                    b_from.kind as from_kind, b_to.kind as to_kind
             FROM ics_belief_links bl
             JOIN ics_beliefs b_from ON b_from.id = bl.from_id
             JOIN ics_beliefs b_to ON b_to.id = bl.to_id
             WHERE bl.from_id IN (
                 SELECT id FROM ics_beliefs WHERE id IN (
                     SELECT belief_id FROM ics_fact_beliefs WHERE subject_entity_id IN ({0})
                     UNION
                     SELECT belief_id FROM ics_rel_participants WHERE entity_id IN ({0})
                 )
             )
             OR bl.to_id IN (
                 SELECT id FROM ics_beliefs WHERE id IN (
                     SELECT belief_id FROM ics_fact_beliefs WHERE subject_entity_id IN ({0})
                     UNION
                     SELECT belief_id FROM ics_rel_participants WHERE entity_id IN ({0})
                 )
             )
             AND b_from.status = 'active'
             AND b_to.status = 'active'",
            placeholders
        );

        let mut q_links = sqlx::query(&query_links);
        // Bind entity_ids 4 times (appears 4 times in query)
        for _ in 0..4 {
            for id in &entity_ids {
                q_links = q_links.bind(id);
            }
        }

        if let Ok(belief_links) = q_links.fetch_all(&*pool).await {
            for row in belief_links {
                let from_id: i64 = row.get("from_id");
                let to_id: i64 = row.get("to_id");
                let link_type: String = row.get("link_type");
                let from_kind: String = row.try_get("from_kind").unwrap_or("fact".to_string());
                let to_kind: String = row.try_get("to_kind").unwrap_or("fact".to_string());
                
                // Format node IDs based on belief kind
                let source_prefix = if from_kind == "rel" { "rel" } else { "fact" };
                let target_prefix = if to_kind == "rel" { "rel" } else { "fact" };
                
                links.push(GraphLink {
                    source: format!("{}_{}", source_prefix, from_id),
                    target: format!("{}_{}", target_prefix, to_id),
                    link_type: format!("belief_{}", link_type),
                    label: Some(link_type.clone()),
                    strength: match link_type.as_str() {
                        "supersedes" => 0.9,
                        "contradicts" => 0.8,
                        "supports" => 0.6,
                        "derived_from" => 0.5,
                        _ => 0.5,
                    },
                });
            }
        }
        
        // 5. Get conflict sets
        let query_conflicts = format!(
            "SELECT DISTINCT cs.id, cs.topic_key, cs.status, cs.priority
             FROM ics_conflict_sets cs
             JOIN ics_conflict_set_members cm ON cm.conflict_set_id = cs.id
             WHERE cm.belief_id IN (
                 SELECT id FROM ics_beliefs WHERE id IN (
                     SELECT belief_id FROM ics_fact_beliefs WHERE subject_entity_id IN ({0})
                     UNION
                     SELECT belief_id FROM ics_rel_participants WHERE entity_id IN ({0})
                 )
             )
             AND cs.status = 'open'",
            placeholders
        );

        let mut q_conflicts = sqlx::query(&query_conflicts);
        // Bind entity_ids twice (appears 2 times in query)
        for _ in 0..2 {
            for id in &entity_ids {
                q_conflicts = q_conflicts.bind(id);
            }
        }

        if let Ok(conflicts) = q_conflicts.fetch_all(&*pool).await {
            for row in conflicts {
                let conflict_id: i64 = row.get("id");
                let topic_key: String = row.try_get("topic_key").unwrap_or_default();
                let status: String = row.try_get("status").unwrap_or_default();
                let priority: String = row.try_get("priority").unwrap_or_default();
                
                // Add conflict set as a special node
                nodes.push(GraphNode {
                    id: format!("conflict_{}", conflict_id),
                    node_type: "conflict".to_string(),
                    label: format!("⚠️ Conflict: {}", topic_key.chars().take(20).collect::<String>()),
                    entity_type: Some(priority),
                    confidence: None,
                    access_count: 0,
                    activation: None,
                    last_accessed: None,
                    key: Some(topic_key.clone()),
                    value: Some(status),
                    // Conflicts don't have temporal/scope info
                    scope: None,
                    polarity: None,
                    time_bucket_kind: None,
                    time_bucket_value: None,
                    participants: None,
                });
                
                // Get members with belief kind to format node IDs correctly
                if let Ok(members) = sqlx::query(
                    "SELECT cm.belief_id, b.kind
                     FROM ics_conflict_set_members cm
                     JOIN ics_beliefs b ON b.id = cm.belief_id
                     WHERE cm.conflict_set_id = ?"
                )
                .bind(conflict_id)
                .fetch_all(&*pool)
                .await {
                    for member_row in members {
                        let belief_id: i64 = member_row.get("belief_id");
                        let belief_kind: String = member_row.try_get("kind").unwrap_or("fact".to_string());
                        let node_prefix = if belief_kind == "rel" { "rel" } else { "fact" };
                        
                        links.push(GraphLink {
                            source: format!("conflict_{}", conflict_id),
                            target: format!("{}_{}", node_prefix, belief_id),
                            link_type: "in_conflict".to_string(),
                            label: None,
                            strength: 0.7,
                        });
                    }
                }
            }
        }
    }

    // 6. Pending clarify items (surface unresolved writes in the graph)
    let pending_rows = if limit > 0 {
        sqlx::query(
            "SELECT id, original_dsl
             FROM ics_pending_clarify
             WHERE status = 'pending'
             ORDER BY created_at DESC
             LIMIT ?"
        )
        .bind(limit)
        .fetch_all(&*pool)
        .await
        .unwrap_or_default()
    } else {
        sqlx::query(
            "SELECT id, original_dsl
             FROM ics_pending_clarify
             WHERE status = 'pending'
             ORDER BY created_at DESC"
        )
        .fetch_all(&*pool)
        .await
        .unwrap_or_default()
    };

    for row in pending_rows {
        let pending_id: i64 = row.get("id");
        let original_dsl: String = row.get("original_dsl");
        let parsed = dsl::parse_memory_block(&original_dsl);
        let mut stmt_opt: Option<DslStatement> = None;
        for res in parsed {
            if let Ok(stmt) = res {
                stmt_opt = Some(stmt);
                break;
            }
        }
        let Some(stmt) = stmt_opt else { continue; };

        match stmt {
            DslStatement::Fact(fact) => {
                let fact_id = format!("pending_fact_{}", pending_id);
                let subject_label = ref_to_label(&fact.subject);
                let subject_id = format!("pending_ref_{}_0", pending_id);

                nodes.push(GraphNode {
                    id: fact_id.clone(),
                    node_type: "pending_fact".to_string(),
                    label: format!("{}: {}", fact.key, fact.value),
                    entity_type: None,
                    confidence: None,
                    access_count: 0,
                    activation: None,
                    last_accessed: None,
                    key: Some(fact.key),
                    value: Some(fact.value),
                    scope: fact.scope_expr,
                    polarity: Some(fact.polarity),
                    time_bucket_kind: None,
                    time_bucket_value: None,
                    participants: Some(vec![("subject".to_string(), subject_label.clone())]),
                });

                nodes.push(GraphNode {
                    id: subject_id.clone(),
                    node_type: "pending_ref".to_string(),
                    label: subject_label,
                    entity_type: None,
                    confidence: None,
                    access_count: 0,
                    activation: None,
                    last_accessed: None,
                    key: None,
                    value: None,
                    scope: None,
                    polarity: None,
                    time_bucket_kind: None,
                    time_bucket_value: None,
                    participants: None,
                });

                links.push(GraphLink {
                    source: subject_id,
                    target: fact_id,
                    link_type: "pending_subject".to_string(),
                    label: None,
                    strength: 0.4,
                });
            }
            DslStatement::Rel(rel) => {
                let rel_id = format!("pending_rel_{}", pending_id);
                let participant_names = rel
                    .participants
                    .iter()
                    .map(|(role, r)| (role.clone(), ref_to_label(r)))
                    .collect::<Vec<_>>();

                nodes.push(GraphNode {
                    id: rel_id.clone(),
                    node_type: "pending_relation".to_string(),
                    label: rel.rel_type.clone(),
                    entity_type: None,
                    confidence: None,
                    access_count: 0,
                    activation: None,
                    last_accessed: None,
                    key: Some(format!("{} participants", rel.participants.len())),
                    value: None,
                    scope: rel.scope_expr,
                    polarity: Some(rel.polarity),
                    time_bucket_kind: None,
                    time_bucket_value: None,
                    participants: Some(participant_names.clone()),
                });

                for (idx, (role, name)) in participant_names.into_iter().enumerate() {
                    let ref_id = format!("pending_ref_{}_{}", pending_id, idx);
                    nodes.push(GraphNode {
                        id: ref_id.clone(),
                        node_type: "pending_ref".to_string(),
                        label: name,
                        entity_type: None,
                        confidence: None,
                        access_count: 0,
                        activation: None,
                        last_accessed: None,
                        key: None,
                        value: None,
                        scope: None,
                        polarity: None,
                        time_bucket_kind: None,
                        time_bucket_value: None,
                        participants: None,
                    });

                    links.push(GraphLink {
                        source: rel_id.clone(),
                        target: ref_id,
                        link_type: "pending_participant".to_string(),
                        label: Some(role),
                        strength: 0.4,
                    });
                }
            }
        }
    }
    
    Ok(MemoryGraph { nodes, links })
}

#[tauri::command]
pub async fn memory_update_entity(
    pool: State<'_, SqlitePool>,
    entity_id: i64,
    new_label: String,
) -> Result<bool, String> {
    let label_canonical = canonicalize_string(&new_label);
    
    // Start transaction
    let mut tx = pool.begin().await.map_err(|e| e.to_string())?;
    
    // Update entity with canonical form
    sqlx::query(
        "UPDATE ics_entities 
         SET label = ?, label_canonical = ?, updated_at = CURRENT_TIMESTAMP 
         WHERE id = ?"
    )
    .bind(&new_label)
    .bind(&label_canonical)
    .bind(entity_id)
    .execute(&mut *tx)
    .await
    .map_err(|e| e.to_string())?;
    
    tx.commit().await.map_err(|e| e.to_string())?;
    cache::bump_cache_version();
    Ok(true)
}

#[tauri::command]
pub async fn memory_delete_entity(
    pool: State<'_, SqlitePool>,
    entity_id: i64,
) -> Result<bool, String> {
    // Start transaction
    let mut tx = pool.begin().await.map_err(|e| e.to_string())?;
    
    // Soft delete entity
    sqlx::query("UPDATE ics_entities SET resolution_state = 'deleted', last_accessed_at = CURRENT_TIMESTAMP WHERE id = ?")
        .bind(entity_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| e.to_string())?;
    
    // Mark associated fact beliefs as INACTIVE (spec uses 'inactive', not 'deleted')
    sqlx::query(
        "UPDATE ics_beliefs SET status = 'inactive', last_accessed_at = CURRENT_TIMESTAMP
         WHERE id IN (SELECT belief_id FROM ics_fact_beliefs WHERE subject_entity_id = ?)"
    )
    .bind(entity_id)
    .execute(&mut *tx)
    .await
    .map_err(|e| e.to_string())?;
    
    // Mark associated relation beliefs as INACTIVE
    sqlx::query(
        "UPDATE ics_beliefs SET status = 'inactive', last_accessed_at = CURRENT_TIMESTAMP
         WHERE id IN (SELECT belief_id FROM ics_rel_participants WHERE entity_id = ?)"
    )
    .bind(entity_id)
    .execute(&mut *tx)
    .await
    .map_err(|e| e.to_string())?;
    
    // Remove from working set
    sqlx::query(
        "DELETE FROM ics_working_set WHERE item_id = ? AND item_type = 'entity'"
    )
    .bind(entity_id)
    .execute(&mut *tx)
    .await
    .map_err(|e| e.to_string())?;
    
    tx.commit().await.map_err(|e| e.to_string())?;
    cache::bump_cache_version();
    Ok(true)
}

#[tauri::command]
pub async fn memory_delete_belief(
    pool: State<'_, SqlitePool>,
    belief_id: i64,
) -> Result<bool, String> {
    // Start transaction
    let mut tx = pool.begin().await.map_err(|e| e.to_string())?;
    
    // Mark belief as INACTIVE (spec: beliefs are never deleted, they become inactive)
    sqlx::query(
        "UPDATE ics_beliefs 
         SET status = 'inactive', last_accessed_at = CURRENT_TIMESTAMP 
         WHERE id = ?"
    )
    .bind(belief_id)
    .execute(&mut *tx)
    .await
    .map_err(|e| e.to_string())?;
    
    // Remove from working set
    sqlx::query(
        "DELETE FROM ics_working_set WHERE item_id = ? AND item_type = 'belief'"
    )
    .bind(belief_id)
    .execute(&mut *tx)
    .await
    .map_err(|e| e.to_string())?;
    
    tx.commit().await.map_err(|e| e.to_string())?;
    cache::bump_cache_version();
    Ok(true)
}
