use crate::core::memory::types::Scope;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseError(pub String);

/// Parse scope string:
/// - global
/// - session OR session:valid_uuid (but spec says session is usually current)
/// - project:id
/// - context:id
pub fn parse_scope(s: &str) -> Result<Scope, ParseError> {
    let lower = s.trim().to_lowercase();
    match lower.as_str() {
        "global" => Ok(Scope::Global),
        "session" => Ok(Scope::Session),
        "self" => Ok(Scope::SelfScope),
        _ => {
            if let Some((kind, id)) = lower.split_once(':') {
                let id = id.trim().to_string();
                if id.is_empty() {
                    return Err(ParseError("Scope ID cannot be empty".to_string()));
                }
                match kind {
                    "project" => Ok(Scope::Project(id)),
                    "context" => Ok(Scope::Context(id)),
                    "session" => Ok(Scope::Session), // Allow session:custom? Types say Session is unit variant though.
                    "self" => Ok(Scope::SelfScope),
                    _ => Err(ParseError(format!("Unknown scope kind: {}", kind))),
                }
            } else {
                Err(ParseError(format!("Invalid scope format: {}", s)))
            }
        }
    }
}

/// Parse stored scope strings from DB or user input.
/// Accepts JSON-encoded Scope (serde) or shorthand "context:foo".
pub fn parse_scope_str(raw: &str) -> Option<Scope> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(scope) = serde_json::from_str::<Scope>(trimmed) {
        return Some(scope);
    }
    let dequoted = strip_quotes(trimmed);
    parse_scope(dequoted).ok()
}

fn strip_quotes(input: &str) -> &str {
    let bytes = input.as_bytes();
    if bytes.len() >= 2 {
        let first = bytes[0] as char;
        let last = bytes[bytes.len() - 1] as char;
        if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
            return &input[1..input.len() - 1];
        }
    }
    input
}

/// Priority: Context > Project > Session > Global (Spec A3.1)
/// Higher number = more specific/higher priority.
pub fn scope_priority(scope: &Scope) -> u8 {
    match scope {
        Scope::SelfScope => 5,
        Scope::Context(_) => 4,
        Scope::Project(_) => 3,
        Scope::Session => 2,
        Scope::Global => 1,
    }
}

/// Scope specificity for shadowing (Spec A3.1)
/// Higher number = more specific.
pub fn scope_specificity(scope: &Scope) -> i32 {
    match scope {
        Scope::SelfScope => 5,
        Scope::Context(_) => 4,
        Scope::Project(_) => 3,
        Scope::Session => 2,
        Scope::Global => 1,
    }
}

/// Convenience for raw scope strings from storage.
pub fn scope_specificity_from_raw(raw: &str) -> i32 {
    parse_scope_str(raw)
        .map(|scope| scope_specificity(&scope))
        .unwrap_or(1)
}

/// Check if `parent` is an ancestor of `child` (Spec A3.1)
/// Hierarchy: Global > Session > Project > Context (most specific)
/// For now, only Global is treated as a universal ancestor.
pub fn is_ancestor(parent: &Scope, child: &Scope) -> bool {
    if parent == child {
        return false;
    }
    match parent {
        Scope::Global => true, // Global is ancestor of any other scope
        _ => false,            // No other nesting defined in v4.1 simple spec
    }
}

/// Get all ancestors (e.g. for shadowing checks)
pub fn get_scope_ancestors(scope: &Scope) -> Vec<Scope> {
    match scope {
        Scope::SelfScope => vec![],
        Scope::Global => vec![],
        _ => vec![Scope::Global],
    }
}


/// Build default scope list for a conversation.
/// Context scopes shadow Global for recall.
pub fn scopes_for_conversation(conversation_id: Option<&str>) -> Vec<Scope> {
    match conversation_id.map(str::trim).filter(|id| !id.is_empty()) {
        Some(id) => vec![Scope::Context(id.to_string()), Scope::Global],
        None => vec![Scope::Global],
    }
}
