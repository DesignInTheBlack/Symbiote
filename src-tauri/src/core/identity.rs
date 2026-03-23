use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashSet;

static IDENTITY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(i am|i'm|im|my name is|call me|i go by|my role is|i work as|i am a|i'm a|i am an|i'm an|you are|you're|your name is|your role is|you are a|you are an)\b")
        .unwrap()
});

static CAPABILITY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(i can|i can't|i cannot|i am able to|i am unable to|i am not able to|i have the ability to|i do not have the ability to|i don't have the ability to)\b")
        .unwrap()
});

pub fn contains_identity_statement(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    IDENTITY_RE.is_match(trimmed)
}

pub fn contains_capability_statement(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    CAPABILITY_RE.is_match(trimmed)
}

pub fn identity_statement_snippets(text: &str, limit: usize) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() || limit == 0 {
        return Vec::new();
    }
    let cleaned = trimmed.replace('\n', " ").replace('\r', " ");
    let mut snippets = Vec::new();
    let mut seen = HashSet::new();
    for sentence in cleaned.split(|c| matches!(c, '.' | '!' | '?')) {
        let candidate = sentence.trim();
        if candidate.is_empty() {
            continue;
        }
        if !IDENTITY_RE.is_match(candidate) {
            continue;
        }
        if seen.insert(candidate.to_string()) {
            snippets.push(candidate.to_string());
        }
        if snippets.len() >= limit {
            break;
        }
    }
    if snippets.is_empty() && IDENTITY_RE.is_match(cleaned.as_str()) {
        snippets.push(cleaned);
    }
    snippets
}

pub fn capability_statement_snippets(text: &str, limit: usize) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() || limit == 0 {
        return Vec::new();
    }
    let cleaned = trimmed.replace('\n', " ").replace('\r', " ");
    let mut snippets = Vec::new();
    let mut seen = HashSet::new();
    for sentence in cleaned.split(|c| matches!(c, '.' | '!' | '?')) {
        let candidate = sentence.trim();
        if candidate.is_empty() {
            continue;
        }
        if !CAPABILITY_RE.is_match(candidate) {
            continue;
        }
        if seen.insert(candidate.to_string()) {
            snippets.push(candidate.to_string());
        }
        if snippets.len() >= limit {
            break;
        }
    }
    if snippets.is_empty() && CAPABILITY_RE.is_match(cleaned.as_str()) {
        snippets.push(cleaned);
    }
    snippets
}

pub fn is_identity_fact_key(key: &str) -> bool {
    let lowered = key.trim().to_lowercase();
    if lowered.is_empty() {
        return false;
    }
    let tokens = [
        "identity",
        "persona",
        "role",
        "name",
        "self",
        "assistant",
        "user",
    ];
    tokens.iter().any(|token| lowered.contains(token))
}

pub fn is_capability_fact_key(key: &str) -> bool {
    let lowered = key.trim().to_lowercase();
    if lowered.is_empty() {
        return false;
    }
    let tokens = [
        "capability",
        "capabilities",
        "ability",
        "abilities",
        "can",
        "cannot",
        "able",
        "unable",
    ];
    tokens.iter().any(|token| lowered.contains(token))
}
