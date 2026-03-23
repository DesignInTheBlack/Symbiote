use crate::core::memory::types::MemoryPacket;
use crate::core::memory::config::MAX_MEMORY_CONTEXT_CHARS;
use crate::models::SelfModel;
use serde::{Deserialize, Serialize};

/// Format MemoryPacket for LLM System Prompt (Spec A15)
pub fn format_for_prompt(packet: &MemoryPacket) -> String {
    let mut parts = vec![];

    // Section 1: Recalled Facts
    if !packet.facts.is_empty() {
        parts.push("Facts:".to_string());
        for fact in &packet.facts {
            let time_str = fact.observed_at_formatted.as_deref().unwrap_or("unknown time");
            parts.push(format!(
                "- {}. Confidence {:.2}. Observed {}.",
                format_fact_sentence(fact),
                fact.confidence,
                time_str
            ));
        }
    }

    // Section 2: Recalled Relations
    if !packet.relations.is_empty() {
        parts.push("\nRelationships:".to_string());
        for rel in &packet.relations {
            let time_str = rel.observed_at_formatted.as_deref().unwrap_or("unknown time");
            parts.push(format!(
                "- {}. Confidence {:.2}. Observed {}.",
                format_rel_sentence(rel),
                rel.confidence,
                time_str
            ));
        }
    }

    // Section 3: Open Conflicts
    if !packet.conflicts.is_empty() {
        parts.push("\nConflicts:".to_string());
        for conflict in &packet.conflicts {
            parts.push(format!(
                "- Conflicting beliefs about {}. Status {:?}.",
                conflict.topic_key, conflict.status
            ));
        }
    }

    // Section 4: Entity handles (for disambiguation reference)
    if !packet.bound_handles.is_empty() {
        parts.push("\nKnown Handles:".to_string());
        for (handle, label) in &packet.bound_handles {
            parts.push(format!("- Handle {} = {}.", handle, label));
        }
    }

    if parts.is_empty() {
        return "".to_string();
    }

    let mut full = vec!["Note: role labels are directional; do not invert.".to_string()];
    full.extend(parts);
    let mut body = String::new();
    for (idx, line) in full.iter().enumerate() {
        let sep = if idx == 0 { "" } else { "\n" };
        if body.len() + sep.len() + line.len() > MAX_MEMORY_CONTEXT_CHARS {
            break;
        }
        if idx > 0 {
            body.push('\n');
        }
        body.push_str(line);
    }

    if body.is_empty() {
        return "".to_string();
    }
    let body = strip_internal_markers(&body);
    if body.trim().is_empty() {
        return "".to_string();
    }

    format!("\n\n<MEMORY_CONTEXT>\n{}\n</MEMORY_CONTEXT>\n", body)
}

fn strip_internal_markers(input: &str) -> String {
    let mut out = Vec::new();
    for line in input.lines() {
        if line.contains("<state_ref>") || line.contains("</state_ref>") {
            continue;
        }
        out.push(line.to_string());
    }
    out.join("\n")
}


fn format_fact_sentence(fact: &crate::core::memory::types::ScoredFact) -> String {
    let key = humanize_token(&fact.key);
    format!("{}'s {} is {}", fact.entity_label, key, fact.value)
}

fn format_rel_sentence(rel: &crate::core::memory::types::ScoredRel) -> String {
    let rel_name = humanize_token(&rel.rel_type);
    let negation = if rel.polarity == "deny" { "NOT " } else { "" };
    if let Some(direction) = rel.direction.as_deref() {
        if rel.participants.len() == 2 && rel.order_is_trusted {
            let left = &rel.participants[0];
            let right = &rel.participants[1];
            let arrow = if direction == "bidirectional" { "<->" } else { "->" };
            return format!(
                "Relationship {}{}: {} is {} {} {} is {}",
                negation,
                rel_name,
                left.role,
                left.entity_label,
                arrow,
                right.role,
                right.entity_label
            );
        }
        let participants = rel
            .participants
            .iter()
            .map(|p| format!("{}={}", p.role, p.entity_label))
            .collect::<Vec<_>>()
            .join("; ");
        return format!("Relationship {}{}: {}", negation, rel_name, participants);
    }
    if rel.participants.len() == 2 {
        let left = &rel.participants[0];
        let right = &rel.participants[1];
        if left.role == right.role {
            return format!(
                "Relationship {}{}: {} and {} (role: {})",
                negation,
                rel_name,
                left.entity_label,
                right.entity_label,
                left.role
            );
        }
        return format!(
            "Relationship {}{}: {} is {}, {} is {}",
            negation,
            rel_name,
            left.role,
            left.entity_label,
            right.role,
            right.entity_label
        );
    }

    let participants = rel
        .participants
        .iter()
        .map(|p| format!("{} is {}", p.role, p.entity_label))
        .collect::<Vec<_>>()
        .join(", ");
    format!("Relationship {}{}: {}", negation, rel_name, participants)
}

fn humanize_token(input: &str) -> String {
    input.replace('_', " ")
}

pub fn format_self_model_for_prompt(model: &SelfModel) -> String {
    let mut lines = Vec::new();
    lines.push("SELF_MODEL".to_string());
    lines.push(format!("capabilities: {}", model.capabilities.to_string()));
    lines.push(format!("limitations: {}", model.limitations.to_string()));
    lines.push(format!("active_tools: {}", model.active_tools.to_string()));
    lines.push(format!("memory_health: {}", model.memory_health.to_string()));
    lines.push(format!("persona: {}", model.persona.to_string()));
    lines.push(format!("goals: {}", model.goals.to_string()));
    if let Some(ts) = &model.last_reflection_at {
        lines.push(format!("last_reflection_at: {}", ts));
    }
    lines.push(format!("updated_at: {}", model.updated_at));
    format!("\n{}\n", lines.join("\n"))
}

pub fn format_persona_policy(model: &SelfModel) -> String {
    let persona = &model.persona;
    let tone = persona_value(persona, "tone", 0.55);
    let verbosity = persona_value(persona, "verbosity", 0.45);
    let directness = persona_value(persona, "directness", 0.7);
    let formality = persona_value(persona, "formality", 0.5);
    let initiative = persona_value(persona, "initiative", 0.5);

    let mut lines = Vec::new();
    lines.push("PERSONA POLICY".to_string());
    lines.push(format!("tone: {}", axis_descriptor("tone", tone)));
    lines.push(format!("verbosity: {}", axis_descriptor("verbosity", verbosity)));
    lines.push(format!("directness: {}", axis_descriptor("directness", directness)));
    lines.push(format!("formality: {}", axis_descriptor("formality", formality)));
    lines.push(format!("initiative: {}", axis_descriptor("initiative", initiative)));
    lines.push("Use these as soft style constraints. Do not override correctness, safety, or user constraints.".to_string());
    format!("\n{}\n", lines.join("\n"))
}

fn persona_value(persona: &serde_json::Value, key: &str, fallback: f32) -> f32 {
    persona
        .get(key)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(fallback)
}

fn axis_descriptor(axis: &str, value: f32) -> String {
    let level = if value < 0.35 {
        "low"
    } else if value > 0.65 {
        "high"
    } else {
        "balanced"
    };

    let guidance = match axis {
        "tone" => {
            if level == "low" {
                "reserved and neutral"
            } else if level == "high" {
                "warm and expressive"
            } else {
                "friendly but controlled"
            }
        }
        "verbosity" => {
            if level == "low" {
                "concise; avoid extra detail"
            } else if level == "high" {
                "expansive; include context"
            } else {
                "moderate detail"
            }
        }
        "directness" => {
            if level == "low" {
                "gentle and hedged"
            } else if level == "high" {
                "clear and direct"
            } else {
                "balanced directness"
            }
        }
        "formality" => {
            if level == "low" {
                "casual tone"
            } else if level == "high" {
                "formal tone"
            } else {
                "neutral formality"
            }
        }
        "initiative" => {
            if level == "low" {
                "reactive; ask before expanding"
            } else if level == "high" {
                "proactive; suggest next steps"
            } else {
                "situational initiative"
            }
        }
        _ => "balanced",
    };

    format!("{:.2} ({}, {})", value, level, guidance)
}

use regex::Regex;
use once_cell::sync::Lazy;

// Regex for markdown blocks
static MEMORY_BLOCK_MD: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)```memory\s*(.*?)\s*```").unwrap());
// Regex for XML blocks
static MEMORY_BLOCK_XML: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)<memory>\s*(.*?)\s*</memory>").unwrap());
static MEMORY_BLOCK_XML_STRICT: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)^\s*<memory>\s*(.*?)\s*</memory>\s*$").unwrap());
static MEMORY_BLOCK_ANGLE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)<<<BEGIN_MEMORY>>>\s*(.*?)\s*<<<END_MEMORY>>>").unwrap());
static EPISODIC_CONTEXT_XML: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)<EPISODIC_CONTEXT>\s*(.*?)\s*</EPISODIC_CONTEXT>").unwrap());
static INTERNAL_BLOCK_XML: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)<INTERNAL>\s*(.*?)\s*</INTERNAL>").unwrap());

#[derive(Default, Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SystemTagSet {
    pub memory: bool,
    pub clarify: bool,
    pub resolve: bool,
}

/// Format a minimal MemoryPacket for constrained contexts (clarify/repair).
pub fn format_for_prompt_limited(packet: &MemoryPacket, max_facts: usize, max_rels: usize) -> String {
    let mut parts = vec![];

    if !packet.facts.is_empty() && max_facts > 0 {
        parts.push("Facts:".to_string());
        for fact in packet.facts.iter().take(max_facts) {
            let time_str = fact.observed_at_formatted.as_deref().unwrap_or("unknown time");
            parts.push(format!(
                "- {}. Confidence {:.2}. Observed {}.",
                format_fact_sentence(fact),
                fact.confidence,
                time_str
            ));
        }
    }

    if !packet.relations.is_empty() && max_rels > 0 {
        parts.push("\nRelationships:".to_string());
        for rel in packet.relations.iter().take(max_rels) {
            let time_str = rel.observed_at_formatted.as_deref().unwrap_or("unknown time");
            parts.push(format!(
                "- {}. Confidence {:.2}. Observed {}.",
                format_rel_sentence(rel),
                rel.confidence,
                time_str
            ));
        }
    }

    if parts.is_empty() {
        return "".to_string();
    }

    let mut full = vec!["Note: role labels are directional; do not invert.".to_string()];
    full.extend(parts);
    let mut body = String::new();
    for (idx, line) in full.iter().enumerate() {
        let sep = if idx == 0 { "" } else { "\n" };
        if body.len() + sep.len() + line.len() > MAX_MEMORY_CONTEXT_CHARS {
            break;
        }
        if idx > 0 {
            body.push('\n');
        }
        body.push_str(line);
    }

    if body.is_empty() {
        return "".to_string();
    }

    format!("\n\n<MEMORY_CONTEXT>\n{}\n</MEMORY_CONTEXT>\n", body)
}

impl SystemTagSet {
    pub fn any(&self) -> bool {
        self.memory || self.clarify || self.resolve
    }
}

pub fn extract_memory_blocks(output: &str) -> Vec<String> {
    let mut blocks = vec![];
    
    for cap in MEMORY_BLOCK_MD.captures_iter(output) {
        if let Some(m) = cap.get(1) {
            blocks.push(m.as_str().to_string());
        }
    }
    
    for cap in MEMORY_BLOCK_XML.captures_iter(output) {
        if let Some(m) = cap.get(1) {
            blocks.push(m.as_str().to_string());
        }
    }

    for cap in MEMORY_BLOCK_ANGLE.captures_iter(output) {
        if let Some(m) = cap.get(1) {
            blocks.push(m.as_str().to_string());
        }
    }
    
    blocks
}

pub fn extract_single_memory_block_strict(output: &str) -> Option<String> {
    let caps = MEMORY_BLOCK_XML_STRICT.captures(output)?;
    let inner = caps.get(1)?.as_str().trim();
    if inner.is_empty() {
        return None;
    }
    Some(inner.to_string())
}

pub fn strip_memory_blocks(output: &str) -> String {
    let s0 = MEMORY_BLOCK_ANGLE.replace_all(output, "");
    let s = MEMORY_BLOCK_MD.replace_all(&s0, "");
    let s2 = MEMORY_BLOCK_XML.replace_all(&s, "");
    let s3 = EPISODIC_CONTEXT_XML.replace_all(&s2, "");
    s3.trim().to_string()
}

pub fn strip_internal_blocks(output: &str) -> String {
    if output.is_empty() {
        return String::new();
    }
    let s = INTERNAL_BLOCK_XML.replace_all(output, "");
    let mut lines = Vec::new();
    for line in s.lines() {
        let trimmed = line.trim();
        if trimmed.eq_ignore_ascii_case("internal state - disclose only with evidence") {
            continue;
        }
        lines.push(line);
    }
    lines.join("\n").trim().to_string()
}

const MEMORY_BLOCK_MD_START: &str = "```memory";
const MEMORY_BLOCK_MD_END: &str = "```";
const MEMORY_BLOCK_XML_START: &str = "<memory>";
const MEMORY_BLOCK_XML_END: &str = "</memory>";
const MEMORY_BLOCK_ANGLE_START: &str = "<<<BEGIN_MEMORY>>>";
const MEMORY_BLOCK_ANGLE_END: &str = "<<<END_MEMORY>>>";
const MEMORY_CONTEXT_START: &str = "<MEMORY_CONTEXT>";
const MEMORY_CONTEXT_END: &str = "</MEMORY_CONTEXT>";
const EPISODIC_CONTEXT_START: &str = "<EPISODIC_CONTEXT>";
const EPISODIC_CONTEXT_END: &str = "</EPISODIC_CONTEXT>";
const MEMORY_STREAM_MAX_START_LEN: usize = 18;

const SYSTEM_TAG_MEMORY: &str = "<<MEMORY>>";
const SYSTEM_TAG_CLARIFY: &str = "<<CLARIFY>>";
const SYSTEM_TAG_RESOLVE: &str = "<<RESOLVE>>";
const SYSTEM_TAG_MEMORY_LOWER: &str = "<<memory>>";
const SYSTEM_TAG_CLARIFY_LOWER: &str = "<<clarify>>";
const SYSTEM_TAG_RESOLVE_LOWER: &str = "<<resolve>>";
const SYSTEM_TAG_MAX_LEN: usize = 11;

#[derive(Clone, Copy)]
enum SystemTag {
    Memory,
    Clarify,
    Resolve,
}

impl SystemTag {
    fn token_lower(&self) -> &'static str {
        match self {
            SystemTag::Memory => SYSTEM_TAG_MEMORY_LOWER,
            SystemTag::Clarify => SYSTEM_TAG_CLARIFY_LOWER,
            SystemTag::Resolve => SYSTEM_TAG_RESOLVE_LOWER,
        }
    }

    fn token_len(&self) -> usize {
        match self {
            SystemTag::Memory => SYSTEM_TAG_MEMORY.len(),
            SystemTag::Clarify => SYSTEM_TAG_CLARIFY.len(),
            SystemTag::Resolve => SYSTEM_TAG_RESOLVE.len(),
        }
    }

    fn mark(&self, tags: &mut SystemTagSet) {
        match self {
            SystemTag::Memory => tags.memory = true,
            SystemTag::Clarify => tags.clarify = true,
            SystemTag::Resolve => tags.resolve = true,
        }
    }
}

fn match_system_tag(line: &str) -> Option<SystemTag> {
    let lower = line.trim().to_ascii_lowercase();
    match lower.as_str() {
        SYSTEM_TAG_MEMORY_LOWER => Some(SystemTag::Memory),
        SYSTEM_TAG_CLARIFY_LOWER => Some(SystemTag::Clarify),
        SYSTEM_TAG_RESOLVE_LOWER => Some(SystemTag::Resolve),
        _ => None,
    }
}

pub fn strip_system_tags(output: &str) -> (String, SystemTagSet) {
    if output.is_empty() {
        return (String::new(), SystemTagSet::default());
    }

    let mut tags = SystemTagSet::default();
    let mut cleaned_lines: Vec<&str> = Vec::new();

    for line in output.lines() {
        if let Some(tag) = match_system_tag(line) {
            tag.mark(&mut tags);
            continue;
        }
        cleaned_lines.push(line);
    }

    while matches!(cleaned_lines.last(), Some(line) if line.trim().is_empty()) {
        cleaned_lines.pop();
    }

    let mut cleaned = cleaned_lines.join("\n");
    if cleaned.len() < output.len() {
        cleaned = cleaned.trim_end().to_string();
    }
    (cleaned, tags)
}

pub fn enforce_trailing_system_tags(output: &str) -> (String, String, SystemTagSet) {
    let (cleaned, tags) = strip_system_tags(output);
    if !tags.any() {
        return (cleaned.clone(), cleaned, tags);
    }
    let mut with_tags = cleaned.trim_end().to_string();
    let mut tag_lines: Vec<&'static str> = Vec::new();
    if tags.memory {
        tag_lines.push(SYSTEM_TAG_MEMORY);
    }
    if tags.clarify {
        tag_lines.push(SYSTEM_TAG_CLARIFY);
    }
    if tags.resolve {
        tag_lines.push(SYSTEM_TAG_RESOLVE);
    }
    if !tag_lines.is_empty() {
        if !with_tags.is_empty() {
            with_tags.push('\n');
        }
        with_tags.push_str(&tag_lines.join("\n"));
    }
    (with_tags, cleaned, tags)
}

fn find_next_system_tag_case_insensitive(input: &str) -> Option<(usize, SystemTag)> {
    let lower = input.to_ascii_lowercase();
    let mut best: Option<(usize, SystemTag)> = None;
    for tag in [SystemTag::Memory, SystemTag::Clarify, SystemTag::Resolve] {
        if let Some(idx) = lower.find(tag.token_lower()) {
            let is_better = best.map_or(true, |(best_idx, _)| idx < best_idx);
            if is_better {
                best = Some((idx, tag));
            }
        }
    }
    best
}

pub struct MemoryStreamFilter {
    buffer: String,
    in_block: bool,
    active_end: Option<&'static str>,
}

impl MemoryStreamFilter {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            in_block: false,
            active_end: None,
        }
    }

    pub fn filter_chunk(&mut self, chunk: &str) -> String {
        self.buffer.push_str(chunk);
        let mut output = String::new();

        loop {
            if self.in_block {
                let end_marker = self.active_end.unwrap_or("");
                if let Some(end_idx) = self.buffer.find(end_marker) {
                    let end_len = end_marker.len();
                    self.buffer = self.buffer[end_idx + end_len..].to_string();
                    self.in_block = false;
                    self.active_end = None;
                    continue;
                }

                let keep = end_marker.len().saturating_sub(1);
                if self.buffer.len() > keep {
                    let start_idx = clamp_char_boundary(&self.buffer, self.buffer.len().saturating_sub(keep));
                    self.buffer = self.buffer[start_idx..].to_string();
                }
                break;
            }

            if let Some((idx, start_marker, end_marker)) = find_next_memory_start(&self.buffer) {
                output.push_str(&self.buffer[..idx]);
                self.buffer = self.buffer[idx + start_marker.len()..].to_string();
                self.in_block = true;
                self.active_end = Some(end_marker);
                continue;
            }

            let keep = MEMORY_STREAM_MAX_START_LEN.saturating_sub(1);
            if self.buffer.len() > keep {
                let emit_len = self.buffer.len() - keep;
                let emit_idx = clamp_char_boundary(&self.buffer, emit_len);
                if emit_idx > 0 {
                    output.push_str(&self.buffer[..emit_idx]);
                    self.buffer = self.buffer[emit_idx..].to_string();
                }
            }
            break;
        }

        output
    }

    pub fn finalize(&mut self) -> String {
        if self.in_block {
            self.buffer.clear();
            return String::new();
        }

        let remaining = self.buffer.clone();
        self.buffer.clear();
        remaining
    }
}

pub struct MemoryTriggerStreamFilter {
    buffer: String,
}

impl MemoryTriggerStreamFilter {
    pub fn new() -> Self {
        Self { buffer: String::new() }
    }

    pub fn filter_chunk(&mut self, chunk: &str) -> String {
        self.buffer.push_str(chunk);
        let mut output = String::new();

        loop {
            if let Some((idx, tag)) = find_next_system_tag_case_insensitive(&self.buffer) {
                output.push_str(&self.buffer[..idx]);
                self.buffer = self.buffer[idx + tag.token_len()..].to_string();
                continue;
            }

            let keep = SYSTEM_TAG_MAX_LEN.saturating_sub(1);
            if self.buffer.len() > keep {
                let emit_len = self.buffer.len() - keep;
                let emit_idx = clamp_char_boundary(&self.buffer, emit_len);
                if emit_idx > 0 {
                    output.push_str(&self.buffer[..emit_idx]);
                    self.buffer = self.buffer[emit_idx..].to_string();
                }
            }
            break;
        }

        output
    }

    pub fn finalize(&mut self) -> String {
        let mut output = String::new();
        loop {
            if let Some((idx, tag)) = find_next_system_tag_case_insensitive(&self.buffer) {
                output.push_str(&self.buffer[..idx]);
                self.buffer = self.buffer[idx + tag.token_len()..].to_string();
                continue;
            }
            output.push_str(&self.buffer);
            self.buffer.clear();
            break;
        }
        output
    }
}

fn find_next_memory_start(input: &str) -> Option<(usize, &'static str, &'static str)> {
    let mut best: Option<(usize, &'static str, &'static str)> = None;
    let markers = [
        (MEMORY_BLOCK_MD_START, MEMORY_BLOCK_MD_END),
        (MEMORY_BLOCK_XML_START, MEMORY_BLOCK_XML_END),
        (MEMORY_BLOCK_ANGLE_START, MEMORY_BLOCK_ANGLE_END),
        (MEMORY_CONTEXT_START, MEMORY_CONTEXT_END),
        (EPISODIC_CONTEXT_START, EPISODIC_CONTEXT_END),
    ];

    for (start, end) in markers {
        if let Some(idx) = input.find(start) {
            let is_better = best.map_or(true, |(best_idx, _, _)| idx < best_idx);
            if is_better {
                best = Some((idx, start, end));
            }
        }
    }

    best
}

fn clamp_char_boundary(input: &str, idx: usize) -> usize {
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

#[cfg(test)]
mod tests {
    use super::{clamp_char_boundary, extract_single_memory_block_strict, strip_system_tags, MemoryTriggerStreamFilter};

    #[test]
    fn strict_memory_block_requires_only_block() {
        let ok = "<memory>\n#A:foo = \"bar\"\n</memory>";
        assert_eq!(
            extract_single_memory_block_strict(ok),
            Some("#A:foo = \"bar\"".to_string())
        );
        let ok_angle = "<<<BEGIN_MEMORY>>>\n#A:foo = \"bar\"\n<<<END_MEMORY>>>";
        assert_eq!(extract_single_memory_block_strict(ok_angle), None);
        let extra = "hi\n<memory>\n#A:foo = \"bar\"\n</memory>";
        assert_eq!(extract_single_memory_block_strict(extra), None);
        let empty = "<memory>\n\n</memory>";
        assert_eq!(extract_single_memory_block_strict(empty), None);
    }

    #[test]
    fn trigger_stream_filter_removes_token_across_chunks() {
        let mut filter = MemoryTriggerStreamFilter::new();
        let a = filter.filter_chunk("Hello <<res");
        let b = filter.filter_chunk("OLVE>> world");
        let c = filter.finalize();
        assert_eq!(format!("{}{}{}", a, b, c), "Hello  world");
    }

    #[test]
    fn strip_system_tags_only_trailing_lines() {
        let (cleaned, tags) = strip_system_tags("Hi\n<<CLARIFY>>\n");
        assert_eq!(cleaned, "Hi");
        assert!(tags.clarify);
    }

    #[test]
    fn strip_system_tags_removes_non_trailing_tag_lines() {
        let (cleaned, tags) = strip_system_tags("Hi\n<<MEMORY>>\nCurrent focus: Alpha.");
        assert_eq!(cleaned, "Hi\nCurrent focus: Alpha.");
        assert!(tags.memory);
    }

    #[test]
    fn strip_system_tags_ignores_inline() {
        let (cleaned, tags) = strip_system_tags("Use <<CLARIFY>> literally.");
        assert_eq!(cleaned, "Use <<CLARIFY>> literally.");
        assert!(!tags.any());
    }

    #[test]
    fn clamp_char_boundary_handles_utf8() {
        let s = "A🌍B";
        let idx = 3; // inside emoji bytes
        let clamped = clamp_char_boundary(s, idx);
        assert!(s.is_char_boundary(clamped));
        assert_eq!(&s[..clamped], "A");
    }
}
