use std::collections::HashSet;

use crate::core::kernel::constants::NON_IGNITION_LANGUAGE_PATTERNS;
use sha2::{Digest, Sha256};

pub(crate) fn summarize_snippet(text: &str, max_len: usize) -> String {
    let cleaned = text.replace('\n', " ");
    if max_len == 0 {
        return String::new();
    }
    cleaned.chars().take(max_len).collect()
}

pub(crate) fn scrub_non_ignition_language(text: &str) -> (String, bool) {
    let mut output = text.to_string();
    let mut changed = false;
    for (pattern, replacement) in NON_IGNITION_LANGUAGE_PATTERNS.iter() {
        let next = pattern.replace_all(&output, *replacement).to_string();
        if next != output {
            output = next;
            changed = true;
        }
    }
    (output, changed)
}

pub(crate) fn clamp_char_boundary(input: &str, idx: usize) -> usize {
    if idx >= input.len() {
        return input.len();
    }
    if input.is_char_boundary(idx) {
        return idx;
    }
    let mut i = idx;
    while i > 0 && !input.is_char_boundary(i) {
        i -= 1;
    }
    i
}

pub(crate) fn token_set(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 2)
        .map(|s| s.to_string())
        .collect()
}

pub(crate) fn overlap_ratio(tokens: &HashSet<String>, anchor_tokens: &HashSet<String>) -> f32 {
    if tokens.is_empty() || anchor_tokens.is_empty() {
        return 0.0;
    }
    let hits = anchor_tokens.intersection(tokens).count();
    hits as f32 / anchor_tokens.len() as f32
}

pub(crate) fn token_similarity(a: &str, b: &str) -> f32 {
    let tokens_a: HashSet<String> = a
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|t| !t.is_empty())
        .collect();
    let tokens_b: HashSet<String> = b
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|t| !t.is_empty())
        .collect();
    if tokens_a.is_empty() || tokens_b.is_empty() {
        return 0.0;
    }
    let intersection = tokens_a.intersection(&tokens_b).count() as f32;
    let union = tokens_a.union(&tokens_b).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

pub(crate) fn lexical_similarity(a: &str, b: &str) -> f32 {
    let tokenize = |s: &str| {
        s.to_lowercase()
            .split_whitespace()
            .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|t| !t.is_empty())
            .map(|t| t.to_string())
            .collect::<HashSet<_>>()
    };
    let set_a = tokenize(a);
    let set_b = tokenize(b);
    if set_a.is_empty() && set_b.is_empty() {
        return 1.0;
    }
    if set_a.is_empty() || set_b.is_empty() {
        return 0.0;
    }
    let intersection = set_a.intersection(&set_b).count() as f32;
    let union = set_a.union(&set_b).count() as f32;
    if union == 0.0 { 0.0 } else { intersection / union }
}

pub(crate) fn count_tokens(text: &str) -> usize {
    text.split_whitespace().filter(|s| !s.is_empty()).count()
}

pub(crate) fn hash_payload(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}
