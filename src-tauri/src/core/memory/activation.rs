use sqlx::SqlitePool;
use sqlx::Row;
use std::collections::{HashMap, HashSet, VecDeque};
use once_cell::sync::Lazy;
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;
use num_complex::Complex32;
use crate::core::cognitive_wave::{AmplitudeBounds, DecayProfile, WaveBand, WaveContributionInput};

const MAX_CACHE_ENTRIES: usize = 256;

static ACTIVATION_CACHE: Lazy<Mutex<HashMap<String, Vec<(i64, f32)>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Debug, Clone)]
pub struct ActivationConfig {
    pub top_k_anchors: usize,
    pub max_depth: usize,
    pub max_nodes: usize,
    pub max_neighbors_per_node: usize,
    pub damping: f32,
    pub iterations: usize,
    pub min_score: f32,
    pub max_results: usize,
    pub weight_scale: f32,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            top_k_anchors: 5,
            max_depth: 2,
            max_nodes: 40,
            max_neighbors_per_node: 8,
            damping: 0.85,
            iterations: 8,
            min_score: 0.01,
            max_results: 8,
            weight_scale: 1.0,
        }
    }
}

pub fn wave_contribution_from_activation(scores: &[(i64, f32)]) -> Option<WaveContributionInput> {
    if scores.is_empty() {
        return None;
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max_score = sorted.first().map(|(_, score)| *score).unwrap_or(0.0).max(0.0);
    if max_score <= 0.0 {
        return None;
    }
    let mut coeffs = Vec::new();
    let take = sorted.len().min(12);
    for (idx, (_entity_id, score)) in sorted.into_iter().take(take).enumerate() {
        let norm = (score / max_score).clamp(0.0, 1.0);
        let phase = (idx as f32 * 0.3).sin().clamp(-1.0, 1.0);
        coeffs.push(Complex32::new(norm, phase * norm));
    }
    let amplitude = (0.1 + max_score * 0.2).clamp(0.05, 0.6);
    Some(WaveContributionInput {
        source: "memory_activation",
        band: WaveBand::Memory,
        coeffs,
        amplitude,
        amplitude_bounds: AmplitudeBounds::new(0.05, 0.6),
        decay_profile: DecayProfile::for_band(WaveBand::Memory),
    })
}

pub async fn activate_with_ppr(
    pool: &SqlitePool,
    anchors: &[(i64, f32)],
    config: &ActivationConfig,
) -> Result<Vec<(i64, f32)>, String> {
    if anchors.is_empty() || config.max_nodes == 0 {
        return Ok(Vec::new());
    }

    let mut seed_pairs = anchors.to_vec();
    seed_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    seed_pairs.truncate(config.top_k_anchors.max(1));

    let cache_key = make_cache_key(&seed_pairs, config);
    if let Some(cached) = get_cached(&cache_key).await {
        return Ok(cached);
    }

    let (nodes, adjacency) = build_induced_subgraph(pool, &seed_pairs, config).await?;
    if nodes.len() <= seed_pairs.len() {
        return Ok(Vec::new());
    }

    let mut node_ids: Vec<i64> = nodes.into_iter().collect();
    node_ids.sort();
    let mut index_map: HashMap<i64, usize> = HashMap::new();
    for (idx, id) in node_ids.iter().enumerate() {
        index_map.insert(*id, idx);
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); node_ids.len()];
    for (id, neighbors) in adjacency {
        let Some(&i) = index_map.get(&id) else { continue; };
        let mut mapped: Vec<usize> = neighbors
            .into_iter()
            .filter_map(|n| index_map.get(&n).cloned())
            .collect();
        mapped.sort_unstable();
        mapped.dedup();
        adj[i] = mapped;
    }

    let mut personalization = vec![0.0f32; node_ids.len()];
    let mut seed_set: HashSet<i64> = HashSet::new();
    let mut weight_sum = 0.0f32;
    for (id, weight) in &seed_pairs {
        if let Some(&idx) = index_map.get(id) {
            personalization[idx] += weight.max(0.0);
            weight_sum += weight.max(0.0);
            seed_set.insert(*id);
        }
    }

    if weight_sum > 0.0 {
        for value in personalization.iter_mut() {
            *value /= weight_sum;
        }
    } else {
        let denom = seed_pairs.len().max(1) as f32;
        for (id, _) in &seed_pairs {
            if let Some(&idx) = index_map.get(id) {
                personalization[idx] = 1.0 / denom;
                seed_set.insert(*id);
            }
        }
    }

    let scores = run_ppr(&adj, &personalization, config.damping, config.iterations);

    let mut results: Vec<(i64, f32)> = node_ids
        .iter()
        .zip(scores.iter())
        .filter(|(id, _)| !seed_set.contains(id))
        .map(|(id, score)| (*id, *score))
        .filter(|(_, score)| *score >= config.min_score)
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(config.max_results);

    store_cached(cache_key, results.clone()).await;
    Ok(results)
}

async fn build_induced_subgraph(
    pool: &SqlitePool,
    seeds: &[(i64, f32)],
    config: &ActivationConfig,
) -> Result<(HashSet<i64>, HashMap<i64, HashSet<i64>>), String> {
    let mut nodes: HashSet<i64> = HashSet::new();
    let mut adjacency: HashMap<i64, HashSet<i64>> = HashMap::new();
    let mut queue: VecDeque<(i64, usize)> = VecDeque::new();

    for (id, _) in seeds {
        if nodes.insert(*id) {
            queue.push_back((*id, 0));
        }
    }

    while let Some((entity_id, depth)) = queue.pop_front() {
        if depth >= config.max_depth {
            continue;
        }
        if nodes.len() >= config.max_nodes {
            break;
        }

        let neighbors = fetch_relation_neighbors(pool, entity_id, config.max_neighbors_per_node).await?;
        if neighbors.is_empty() {
            continue;
        }

        for neighbor_id in neighbors {
            adjacency.entry(entity_id).or_default().insert(neighbor_id);
            adjacency.entry(neighbor_id).or_default().insert(entity_id);

            if nodes.len() < config.max_nodes && nodes.insert(neighbor_id) {
                queue.push_back((neighbor_id, depth + 1));
            }
        }
    }

    Ok((nodes, adjacency))
}

async fn fetch_relation_neighbors(
    pool: &SqlitePool,
    entity_id: i64,
    limit: usize,
) -> Result<Vec<i64>, String> {
    let rows = sqlx::query(
        r#"
        SELECT DISTINCT rp2.entity_id as neighbor_id
        FROM ics_rel_participants rp1
        JOIN ics_beliefs b ON b.id = rp1.belief_id
        JOIN ics_rel_participants rp2 ON rp2.belief_id = b.id
        WHERE rp1.entity_id = ?
          AND rp2.entity_id != ?
          AND b.status = 'active'
        LIMIT ?
        "#,
    )
    .bind(entity_id)
    .bind(entity_id)
    .bind(limit as i64)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    Ok(rows.into_iter().map(|row| row.get::<i64, _>("neighbor_id")).collect())
}

fn run_ppr(adj: &[Vec<usize>], personalization: &[f32], damping: f32, iterations: usize) -> Vec<f32> {
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }

    let mut scores = personalization.to_vec();

    for _ in 0..iterations.max(1) {
        let mut next = vec![0.0f32; n];
        let mut dangling_mass = 0.0f32;

        for (i, neighbors) in adj.iter().enumerate() {
            if neighbors.is_empty() {
                dangling_mass += scores[i];
                continue;
            }

            let share = scores[i] / neighbors.len() as f32;
            for &j in neighbors {
                next[j] += damping * share;
            }
        }

        if dangling_mass > 0.0 {
            for (i, value) in personalization.iter().enumerate() {
                next[i] += damping * dangling_mass * value;
            }
        }

        let teleport = 1.0 - damping;
        for (i, value) in personalization.iter().enumerate() {
            next[i] += teleport * value;
        }

        scores = next;
    }

    scores
}

fn make_cache_key(anchors: &[(i64, f32)], config: &ActivationConfig) -> String {
    let mut parts = Vec::with_capacity(anchors.len() + 1);
    parts.push(format!(
        "cfg:{}:{}:{}:{}:{:.3}:{}:{:.4}:{}:{:.3}",
        config.top_k_anchors,
        config.max_depth,
        config.max_nodes,
        config.max_neighbors_per_node,
        config.damping,
        config.iterations,
        config.min_score,
        config.max_results,
        config.weight_scale
    ));
    for (id, weight) in anchors {
        parts.push(format!("a:{}:{:.3}", id, weight));
    }
    let mut hasher = Sha256::new();
    hasher.update(parts.join("|").as_bytes());
    hex::encode(hasher.finalize())
}

async fn get_cached(key: &str) -> Option<Vec<(i64, f32)>> {
    let guard = ACTIVATION_CACHE.lock().await;
    guard.get(key).cloned()
}

async fn store_cached(key: String, value: Vec<(i64, f32)>) {
    let mut guard = ACTIVATION_CACHE.lock().await;
    if guard.len() >= MAX_CACHE_ENTRIES {
        guard.clear();
    }
    guard.insert(key, value);
}
