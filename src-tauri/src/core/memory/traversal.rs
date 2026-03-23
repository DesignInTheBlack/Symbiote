use sqlx::SqlitePool;
use sqlx::Row;
use crate::core::memory::config::{MAX_HOPS, MAX_NODES, MAX_EXPANSIONS_PER_NODE, MAX_EXPANSIONS_FOR_ANCHOR};
use std::collections::{HashSet, VecDeque};

pub struct TraversalConfig {
    pub max_hops: usize,
    pub max_nodes: usize,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            max_hops: MAX_HOPS,
            max_nodes: MAX_NODES,
        }
    }
}

use std::collections::HashMap;

pub struct TraversalResult {
    pub entities: HashMap<i64, f32>, // EntityId -> Association Weight (1.0 = Anchor, lower = distant)
    pub beliefs: Vec<i64>, // Belief IDs encountered
    pub hops_taken: usize,
    pub entities_visited: usize,
    pub beliefs_collected: usize,
    pub frontier_max_size: usize,
    pub was_bounded: bool,
}

/// BFS Traversal from anchors
pub async fn bounded_traversal(anchors: &[(i64, f32)], config: &TraversalConfig, pool: &SqlitePool) -> Result<TraversalResult, String> {
    let mut visited_entities: HashMap<i64, f32> = HashMap::new();
    let mut visited_beliefs = HashSet::new(); // Rel beliefs or fact beliefs
    let mut queue = VecDeque::new();
    let mut hops_taken: usize = 0;
    let mut frontier_max_size: usize = 0;
    let mut was_bounded = false;
    
    // Init queue
    for (id, weight) in anchors {
        let anchor_weight = weight.max(0.1).min(2.5);
        if visited_entities.insert(*id, anchor_weight).is_none() {
            queue.push_back((*id, 0));
        }
    }
    
    while let Some((entity_id, depth)) = queue.pop_front() {
        if queue.len() > frontier_max_size {
            frontier_max_size = queue.len();
        }
        if depth >= config.max_hops {
            was_bounded = true;
            continue;
        }
        
        let current_weight = *visited_entities.get(&entity_id).unwrap_or(&0.0);
        let next_weight = current_weight * 0.5; // Decay factor per hop
        
        if visited_entities.len() >= config.max_nodes {
            was_bounded = true;
            break;
        }
        
        // Expand: Find relations involving this entity
        let expansion_limit = if depth == 0 {
            MAX_EXPANSIONS_FOR_ANCHOR
        } else {
            MAX_EXPANSIONS_PER_NODE
        };

        let rows = sqlx::query(
            r#"
            SELECT b.id as belief_id, rp.role, rp2.entity_id as neighbor_id
            FROM ics_rel_participants rp
            JOIN ics_beliefs b ON b.id = rp.belief_id
            JOIN ics_rel_participants rp2 ON rp2.belief_id = b.id
            WHERE rp.entity_id = ? AND b.status = 'active'
            AND rp2.entity_id != ? -- don't point back to self as neighbor
            ORDER BY
              CASE
                WHEN b.kind = 'relation' THEN 0
                ELSE 1
              END,
              CASE
                WHEN b.topic_key LIKE '%created_by%' THEN 2
                ELSE 1
              END,
              b.salience DESC,
              b.evidence_weight_total DESC,
              b.last_evidence_at DESC
            LIMIT ?
            "#
        )
        .bind(entity_id)
        .bind(entity_id)
        .bind(expansion_limit as i64)
        .fetch_all(pool)
        .await
        .map_err(|e| e.to_string())?;
        
        for row in rows {
            let belief_id: i64 = row.get("belief_id");
            let neighbor_id: i64 = row.get("neighbor_id");
            
            visited_beliefs.insert(belief_id);
            
            if !visited_entities.contains_key(&neighbor_id) {
                if visited_entities.len() < config.max_nodes {
                    visited_entities.insert(neighbor_id, next_weight);
                    queue.push_back((neighbor_id, depth + 1));
                    if depth + 1 > hops_taken {
                        hops_taken = depth + 1;
                    }
                }
            }
        }
    }
    
    let entities_visited = visited_entities.len();
    let beliefs_collected = visited_beliefs.len();

    Ok(TraversalResult {
        entities: visited_entities,
        beliefs: visited_beliefs.into_iter().collect(),
        hops_taken,
        entities_visited,
        beliefs_collected,
        frontier_max_size,
        was_bounded,
    })
}
