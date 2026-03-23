use sha2::{Sha256, Digest};
use unicode_normalization::UnicodeNormalization;
use std::collections::HashMap;
use once_cell::sync::Lazy;

static REL_TYPE_ALIAS_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert("depend_on", "depends_on");
    map.insert("depends", "depends_on");
    map.insert("dependent_on", "depends_on");
    map.insert("dependency", "depends_on");
    map.insert("caused_by", "causes");
    map.insert("causing", "causes");
    map.insert("cause", "causes");
    map.insert("improved_by", "improves");
    map.insert("improves_on", "improves");
    map.insert("improvement", "improves");
    map.insert("blocked_by", "blocks");
    map.insert("blocking", "blocks");
    map.insert("related", "related_to");
    map.insert("relates_to", "related_to");
    map.insert("relation", "related_to");
    map.insert("implemented_by", "implemented_in");
    map.insert("implements", "implemented_in");
    map.insert("implementation", "implemented_in");
    map
});

/// Normalize string: Trim, collapse whitespace, NFC normalization.
/// Does NOT lowercase by default unless specified for case-insensitive fields.
pub fn canonicalize_string(s: &str) -> String {
    let collapsed: String = s.split_whitespace().collect::<Vec<&str>>().join(" ");
    collapsed.chars().nfc().collect::<String>()
}

/// Normalize entity labels: case-insensitive and whitespace-insensitive.
pub fn canonicalize_label(label: &str) -> String {
    let collapsed = canonicalize_string(label);
    let lower = collapsed.to_lowercase();
    lower.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Normalize relation type tokens: trim, collapse whitespace, lowercase, snake_case-ish.
pub fn normalize_rel_type(raw: &str) -> String {
    let collapsed = canonicalize_string(raw);
    let lower = collapsed.to_lowercase();
    let mut out = String::new();
    let mut prev_underscore = false;
    for ch in lower.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    let normalized = out.trim_matches('_').to_string();
    if let Some(mapped) = REL_TYPE_ALIAS_MAP.get(normalized.as_str()) {
        mapped.to_string()
    } else {
        normalized
    }
}

/// Compute value hash for fact values (Spec Appendix A)
pub fn compute_value_hash(value_literal: &str) -> String {
    let canonical = canonicalize_string(value_literal);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    hex::encode(hasher.finalize())
}

/// Compute topic key for facts: subject_id is NOT available yet here usually, 
/// but if we have the struct `FactBelief` which has IDs, we can use it.
/// Actually, topic_key needs to be computed *before* DB insertion often, but after resolution.
/// Spec §7.3: topic_key for Fact = "{subject_id}:{key}"
pub fn compute_topic_key_fact(subject_id: i64, key: &str) -> String {
    format!("{}:{}", subject_id, key)
}

/// Spec §7.3: topic_key for Rel = "{rel_type}" (Broad topic)
pub fn compute_topic_key_rel(rel_type: &str) -> String {
    rel_type.to_string()
}

/// Compute Signature Hash (Spec §7.2)
/// Used for ManySet deduplication.
/// Input: Key-Value pairs that uniquely identify the belief content.
/// For Fact: (key, value)
/// For Rel: (rel_type, sorted_participants)
pub fn compute_signature_hash(inputs: &[(&str, &str)]) -> String {
    let mut sorted: Vec<_> = inputs.to_vec();
    sorted.sort_by_key(|a| a.0); // Sort by key
    
    let mut hasher = Sha256::new();
    for (k, v) in sorted {
        hasher.update(k.as_bytes());
        hasher.update(b":");
        hasher.update(v.as_bytes());
        hasher.update(b"|");
    }
    hex::encode(hasher.finalize())
}

/// Canonicalize participants string for RelBelief (sorted roles)
/// Returns a string suitable for hashing or storage if we want a canonical representation.
/// Spec says Rel signature includes participants.
pub fn canonicalize_participants(participants: &[(String, i64)]) -> String {
    // Sort by role then entity_id
    let mut p = participants.to_vec();
    p.sort_by(|a, b| {
        a.0.cmp(&b.0).then(a.1.cmp(&b.1))
    });
    
    // Format: role:id|role:id
    serialize_participants(&p)
}

/// Serialize participant ids in their current order (no roles).
pub fn serialize_participant_ids(participants: &[(String, i64)]) -> String {
    participants
        .iter()
        .map(|(_, id)| id.to_string())
        .collect::<Vec<_>>()
        .join("|")
}

/// Compute anchor signature (Spec §7.0)
/// Inputs: anchor roles (list of role tokens), participants (all canonical participants)
/// Logic: Filter participants to those matching anchor roles.
/// If anchor_roles is empty or filter is empty, fallback to ALL participants.
pub fn compute_anchor_signature(anchor_roles: &[String], participants: &[(String, i64)], sort: bool) -> String {
    let mut relevant: Vec<(String, i64)> = if anchor_roles.is_empty() {
        participants.to_vec()
    } else {
        let set: std::collections::HashSet<String> = anchor_roles.iter().cloned().collect();
        let filtered: Vec<_> = participants.iter()
            .filter(|(r, _)| set.contains(r))
            .cloned()
            .collect();
            
        if filtered.is_empty() {
            participants.to_vec()
        } else {
            filtered
        }
    };
    
    if sort {
        // Sort logic (same as canonicalize_participants but for hashing)
        relevant.sort_by(|a, b| {
            a.0.cmp(&b.0).then(a.1.cmp(&b.1))
        });
    }
    
    // Serialize using participant ids only to avoid role-label drift.
    let raw = serialize_participant_ids(&relevant);
        
    let mut hasher = Sha256::new();
    hasher.update(raw.as_bytes());
    hex::encode(hasher.finalize())
}

/// Normalize a role token: trim, collapse whitespace, lowercase.
pub fn normalize_role_token(role: &str) -> String {
    let collapsed = canonicalize_string(role);
    collapsed.to_lowercase()
}

/// Resolve a role token using confirmed aliases (normalized).
pub fn canonicalize_role_token(role: &str, alias_map: &HashMap<String, String>) -> String {
    let normalized = normalize_role_token(role);
    alias_map
        .get(&normalized)
        .cloned()
        .unwrap_or(normalized)
}

/// Serialize participants in their current order.
pub fn serialize_participants(participants: &[(String, i64)]) -> String {
    participants
        .iter()
        .map(|(role, id)| format!("{}:{}", role, id))
        .collect::<Vec<_>>()
        .join("|")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_canonicalize_label_spacing_and_case() {
        assert_eq!(canonicalize_label("Mister Black"), "misterblack");
        assert_eq!(canonicalize_label("  Mister  Black "), "misterblack");
        assert_eq!(canonicalize_label("MisterBlack"), "misterblack");
    }

    #[test]
    fn test_normalize_role_token() {
        assert_eq!(normalize_role_token(" Father "), "father");
        assert_eq!(normalize_role_token(" lead   artist "), "lead artist");
    }

    #[test]
    fn test_canonicalize_role_token_with_alias() {
        let mut aliases = HashMap::new();
        aliases.insert("dad".to_string(), "father".to_string());
        assert_eq!(canonicalize_role_token("Dad", &aliases), "father");
        assert_eq!(canonicalize_role_token("mother", &aliases), "mother");
    }
}
