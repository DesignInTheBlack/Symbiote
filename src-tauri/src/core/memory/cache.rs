use once_cell::sync::Lazy;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;

use crate::core::memory::types::{MemoryPacket, QueryIntent, Scope};

const MAX_CACHE_ENTRIES: usize = 256;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct CacheKey {
    conversation_id: String,
    context_hash: String,
    query_hash: String,
    scopes_hash: String,
    intent: String,
}

#[derive(Clone, Debug)]
struct CacheEntry {
    version: u64,
    packet: MemoryPacket,
    episodic_context: String,
}

static CACHE_VERSION: AtomicU64 = AtomicU64::new(1);
static CACHE: Lazy<Mutex<HashMap<CacheKey, CacheEntry>>> = Lazy::new(|| Mutex::new(HashMap::new()));

pub fn bump_cache_version() {
    CACHE_VERSION.fetch_add(1, Ordering::SeqCst);
}

fn current_version() -> u64 {
    CACHE_VERSION.load(Ordering::SeqCst)
}

fn hash_str(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

fn scopes_key(scopes: &[Scope]) -> String {
    scopes
        .iter()
        .map(|s| serde_json::to_string(s).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("|")
}

fn intent_key(intent: &QueryIntent) -> &'static str {
    match intent {
        QueryIntent::AskCurrent => "ask_current",
        QueryIntent::AskList => "ask_list",
        QueryIntent::AskHistory => "ask_history",
        QueryIntent::AskExplain => "ask_explain",
    }
}

fn make_key(
    conversation_id: &str,
    context_hash: &str,
    query: &str,
    scopes: &[Scope],
    intent: &QueryIntent,
) -> CacheKey {
    CacheKey {
        conversation_id: conversation_id.to_string(),
        context_hash: context_hash.to_string(),
        query_hash: hash_str(query),
        scopes_hash: hash_str(&scopes_key(scopes)),
        intent: intent_key(intent).to_string(),
    }
}

pub async fn get_cached(
    conversation_id: &str,
    context_hash: &str,
    query: &str,
    scopes: &[Scope],
    intent: &QueryIntent,
) -> Option<(MemoryPacket, String)> {
    let query = query.trim();
    if conversation_id.trim().is_empty() || context_hash.trim().is_empty() || query.is_empty() {
        return None;
    }

    let key = make_key(conversation_id, context_hash, query, scopes, intent);
    let mut guard = CACHE.lock().await;
    let current = current_version();

    match guard.get(&key) {
        Some(entry) if entry.version == current => {
            Some((entry.packet.clone(), entry.episodic_context.clone()))
        }
        Some(_) => {
            guard.remove(&key);
            None
        }
        None => None,
    }
}

pub async fn store_cached(
    conversation_id: &str,
    context_hash: &str,
    query: &str,
    scopes: &[Scope],
    intent: &QueryIntent,
    packet: MemoryPacket,
    episodic_context: String,
) {
    let query = query.trim();
    if conversation_id.trim().is_empty() || context_hash.trim().is_empty() || query.is_empty() {
        return;
    }

    let mut guard = CACHE.lock().await;
    if guard.len() >= MAX_CACHE_ENTRIES {
        guard.clear();
    }

    let key = make_key(conversation_id, context_hash, query, scopes, intent);
    guard.insert(
        key,
        CacheEntry {
            version: current_version(),
            packet,
            episodic_context,
        },
    );
}
