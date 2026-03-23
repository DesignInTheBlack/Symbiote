use std::path::PathBuf;
use std::sync::{OnceLock, RwLock};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug)]
pub struct PromptSet {
    pub primary_prompt: String,
    pub memory_control_prompt: String,
    pub reflection_prompt: String,
    pub primary_hash: String,
    pub memory_hash: String,
    pub reflection_hash: String,
    pub source: String,
}

const PRIMARY_MARKER: &str = "//PRIMARY SYSTEM PROMPT";
const MEMORY_MARKER: &str = "//MEMORY CONTROL SYSTEM";
const REFLECTION_MARKER: &str = "//INTROSPECTION REFLECTION PROMPT";
const PROMPT_FILE: &str = "memory_syntax.md";

static PROMPT_CACHE: OnceLock<RwLock<Option<PromptSet>>> = OnceLock::new();

pub fn get_prompts() -> Result<PromptSet, String> {
    let cache = PROMPT_CACHE.get_or_init(|| RwLock::new(None));
    if let Some(prompts) = cache.read().unwrap_or_else(|e| e.into_inner()).clone() {
        return Ok(prompts);
    }

    let prompts = load_prompts_from_disk()?;
    *cache.write().unwrap_or_else(|e| e.into_inner()) = Some(prompts.clone());
    Ok(prompts)
}

pub fn reload_prompts() -> Result<(), String> {
    let prompts = load_prompts_from_disk()?;
    let cache = PROMPT_CACHE.get_or_init(|| RwLock::new(None));
    *cache.write().unwrap_or_else(|e| e.into_inner()) = Some(prompts);
    Ok(())
}

fn load_prompts_from_disk() -> Result<PromptSet, String> {
    let (content, source) = read_prompt_file()?;
    parse_prompt_sections(&content, &source)
}

fn read_prompt_file() -> Result<(String, String), String> {
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(PROMPT_FILE));
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(PROMPT_FILE),
    );

    for path in candidates {
        if path.is_file() {
            let content = std::fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
            return Ok((content, path.display().to_string()));
        }
    }

    let fallback = include_str!("../../../memory_syntax.md");
    if fallback.trim().is_empty() {
        return Err(format!("{} missing and embedded fallback is empty", PROMPT_FILE));
    }
    Ok((fallback.to_string(), "embedded".to_string()))
}

fn parse_prompt_sections(content: &str, source: &str) -> Result<PromptSet, String> {
    let primary_prompt = extract_section(content, PRIMARY_MARKER, Some(MEMORY_MARKER))?;
    let memory_control_prompt = extract_section(content, MEMORY_MARKER, Some(REFLECTION_MARKER))?;
    let reflection_prompt = extract_section(content, REFLECTION_MARKER, None)?;
    let primary_hash = hash_prompt(&primary_prompt);
    let memory_hash = hash_prompt(&memory_control_prompt);
    let reflection_hash = hash_prompt(&reflection_prompt);
    Ok(PromptSet {
        primary_prompt,
        memory_control_prompt,
        reflection_prompt,
        primary_hash,
        memory_hash,
        reflection_hash,
        source: source.to_string(),
    })
}

fn extract_section(content: &str, start_marker: &str, end_marker: Option<&str>) -> Result<String, String> {
    let start_idx = content
        .find(start_marker)
        .ok_or_else(|| format!("Missing prompt marker: {}", start_marker))?;
    let after_start = start_idx + start_marker.len();
    let remainder = &content[after_start..];

    let section = if let Some(end_marker) = end_marker {
        let end_idx = remainder
            .find(end_marker)
            .ok_or_else(|| format!("Missing prompt marker: {}", end_marker))?;
        &remainder[..end_idx]
    } else {
        remainder
    };

    let trimmed = section.trim();
    if trimmed.is_empty() {
        return Err(format!("Prompt section after {} is empty", start_marker));
    }

    Ok(trimmed.to_string())
}

fn hash_prompt(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}
