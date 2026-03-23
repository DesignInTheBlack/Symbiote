use crate::core::memory::dsl;
use once_cell::sync::Lazy;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Mutex;

const SNIPPET_CACHE_LIMIT: usize = 512;

static SNIPPET_CACHE: Lazy<Mutex<HashMap<String, String>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn snippet_cache_key(parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.as_bytes());
        hasher.update(b"|");
    }
    hex::encode(hasher.finalize())
}

fn cache_snippet(key: String, value: String) -> String {
    let mut guard = SNIPPET_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if guard.len() >= SNIPPET_CACHE_LIMIT {
        guard.clear();
    }
    guard.insert(key, value.clone());
    value
}

fn cached_snippet(parts: &[&str], build: impl FnOnce() -> String) -> String {
    let key = snippet_cache_key(parts);
    if let Some(existing) = SNIPPET_CACHE.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return existing.clone();
    }
    cache_snippet(key, build())
}

pub fn humanize_token(token: &str) -> String {
    token.replace('_', " ")
}

pub fn render_fact_snippet(label: &str, key: &str, value: &str, polarity: &str) -> String {
    cached_snippet(
        &[label, key, value, polarity],
        || {
            let label = label.trim();
            let key = humanize_token(key.trim());
            let value = value.trim();
            let prefix = if polarity == "deny" {
                "Fact (denied)"
            } else {
                "Fact"
            };

            if label.is_empty() {
                if key.is_empty() {
                    return format!("{}: {}", prefix, value);
                }
                return format!("{}: {} is {}", prefix, key, value);
            }

            format!("{}: {}'s {} is {}", prefix, label, key, value)
        },
    )
}

pub fn render_rel_snippet(rel_type: &str, participants: &[(String, String)], polarity: &str) -> String {
    render_rel_snippet_with_direction(rel_type, participants, polarity, None)
}

pub fn render_rel_snippet_with_direction(
    rel_type: &str,
    participants: &[(String, String)],
    polarity: &str,
    direction: Option<&str>,
) -> String {
    let mut parts_key = String::new();
    for (role, label) in participants {
        parts_key.push_str(role);
        parts_key.push(':');
        parts_key.push_str(label);
        parts_key.push('|');
    }
    let direction_key = direction.unwrap_or("");
    cached_snippet(
        &[rel_type, polarity, direction_key, &parts_key],
        || {
            let rel_name = humanize_token(rel_type.trim());
            let prefix = if polarity == "deny" {
                "Relation (denied)"
            } else {
                "Relation"
            };

            let arrow = match direction {
                Some("directed") => Some("->"),
                Some("bidirectional") => Some("<->"),
                _ => None,
            };

            if let (Some(arrow), 2) = (arrow, participants.len()) {
                let left = &participants[0];
                let right = &participants[1];
                return format!(
                    "{}: {} - {}: {} {} {}: {}",
                    prefix,
                    rel_name,
                    left.0.trim(),
                    left.1.trim(),
                    arrow,
                    right.0.trim(),
                    right.1.trim()
                );
            }

            let parts = participants
                .iter()
                .map(|(role, label)| format!("{}: {}", role.trim(), label.trim()))
                .collect::<Vec<_>>()
                .join(", ");

            if parts.is_empty() {
                return format!("{}: {}", prefix, rel_name);
            }
            format!("{}: {} - {}", prefix, rel_name, parts)
        },
    )
}

pub fn sanitize_episodic_text(input: &str) -> String {
    if input.trim().is_empty() {
        return String::new();
    }

    cached_snippet(&["sanitize", input], || {
        let mut cleaned_lines = Vec::new();
        let mut in_memory_block = false;

        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if in_memory_block {
                if trimmed.starts_with("```") || trimmed.eq_ignore_ascii_case("</memory>") {
                    in_memory_block = false;
                }
                continue;
            }

            let lower = trimmed.to_lowercase();
            if lower.starts_with("```memory") {
                in_memory_block = true;
                continue;
            }

            if trimmed.eq_ignore_ascii_case("<memory>") {
                in_memory_block = true;
                continue;
            }

            if trimmed.eq_ignore_ascii_case("</memory>") {
                continue;
            }

            let mut cleaned = strip_markers(trimmed);
            cleaned = cleaned.trim().to_string();
            if cleaned.is_empty() {
                continue;
            }
            let lower = cleaned.to_lowercase();
            if lower.contains("reason: focus") && lower.contains("supported by memory") {
                if let Some(idx) = lower.find("reason: focus") {
                    cleaned = cleaned[..idx].trim_end().to_string();
                }
                if cleaned.is_empty() {
                    continue;
                }
            }
            if is_modifier_only(&cleaned) {
                continue;
            }
            if dsl::is_dsl_line(&cleaned) {
                continue;
            }
            cleaned_lines.push(cleaned);
        }

        cleaned_lines.join("\n").trim().to_string()
    })
}

fn strip_markers(line: &str) -> String {
    let mut cleaned = line.to_string();
    for marker in [
        "<memory>",
        "</memory>",
        "<MEMORY_CONTEXT>",
        "</MEMORY_CONTEXT>",
        "<EPISODIC_CONTEXT>",
        "</EPISODIC_CONTEXT>",
        "<<MEMORY>>",
    ] {
        cleaned = strip_marker_case_insensitive(&cleaned, marker);
    }
    cleaned = cleaned.replace("<>", "").replace("<<>>", "");
    cleaned
}

fn strip_marker_case_insensitive(line: &str, marker: &str) -> String {
    if marker.is_empty() || line.is_empty() {
        return line.to_string();
    }
    let line_chars: Vec<char> = line.chars().collect();
    let marker_chars: Vec<char> = marker.chars().collect();
    if marker_chars.is_empty() {
        return line.to_string();
    }

    let mut out = String::with_capacity(line.len());
    let mut i = 0usize;
    while i < line_chars.len() {
        let mut matched = false;
        if i + marker_chars.len() <= line_chars.len() {
            matched = true;
            for j in 0..marker_chars.len() {
                if line_chars[i + j].to_ascii_lowercase()
                    != marker_chars[j].to_ascii_lowercase()
                {
                    matched = false;
                    break;
                }
            }
        }
        if matched {
            i += marker_chars.len();
            continue;
        }
        out.push(line_chars[i]);
        i += 1;
    }
    out
}

fn is_modifier_only(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed == "!" || trimmed.eq_ignore_ascii_case("!deny") {
        return true;
    }
    if let Some(raw) = trimmed.strip_prefix('~') {
        let raw = raw.trim_end_matches('%');
        if !raw.is_empty() && raw.parse::<f32>().is_ok() {
            return true;
        }
    }
    if trimmed.starts_with('@') && trimmed.len() > 1 {
        return true;
    }
    if trimmed.starts_with('^') && trimmed.len() > 1 {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::sanitize_episodic_text;

    #[test]
    fn sanitize_strips_memory_blocks_and_dsl_lines() {
        let input = "<memory>\n#Alice:age = \"30\"\n</memory>\nHello there\n";
        let output = sanitize_episodic_text(input);
        assert_eq!(output, "Hello there");
    }

    #[test]
    fn sanitize_preserves_hashtags_and_emails() {
        let input = "Email me at bob@example.com #urgent";
        let output = sanitize_episodic_text(input);
        assert_eq!(output, input);
    }

    #[test]
    fn sanitize_preserves_non_dsl_sentences() {
        let input = "Note: a = b\nWhen I say (hello), I mean hi.";
        let output = sanitize_episodic_text(input);
        assert_eq!(output, input);
    }

    #[test]
    fn sanitize_drops_modifier_only_lines() {
        let input = "We met yesterday\n~0.7\n@global\n";
        let output = sanitize_episodic_text(input);
        assert_eq!(output, "We met yesterday");
    }

    #[test]
    fn sanitize_strips_case_insensitive_markers_inline() {
        let input = "<memory_context>Keep</MEMORY_CONTEXT> <<memory>>";
        let output = sanitize_episodic_text(input);
        assert_eq!(output, "Keep");
    }
}
