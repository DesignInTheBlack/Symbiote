use serde::{Deserialize, Serialize};
use serde_json::Value;
use regex::Regex;
use once_cell::sync::Lazy;
use chrono::{DateTime, Duration, Utc};
use crate::core::memory::rel_vocab::is_canonical_relation;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DslStatement {
    Fact(FactStmt),
    Rel(RelStmt),
}

/// Time expression parsed from DSL (§4)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeExpr {
    pub kind: String,        // "instant", "range", "relative"
    pub value: String,       // "2026-01-09" or "2026-01-01..2026-01-07" or "today"
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactStmt {
    pub subject: Ref,
    pub key: String,
    pub value: String,
    #[serde(default)]
    pub value_quoted: bool,
    pub certainty: Option<f32>,
    pub time_expr: Option<TimeExpr>,   // ^2026-01-09 or ^[start..end]
    pub scope_expr: Option<String>,    // @project:123
    pub source_ref: Option<String>,    // <http://example.com>
    pub polarity: String,              // "assert" or "deny"
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelStmt {
    pub rel_type: String,
    #[serde(default)]
    pub rel_type_id: Option<String>,
    pub participants: Vec<(String, Ref)>,
    pub direction: Option<RelDirection>,
    pub certainty: Option<f32>,
    pub time_expr: Option<TimeExpr>,
    pub scope_expr: Option<String>,
    pub source_ref: Option<String>,
    pub polarity: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelDirection {
    Directed,
    Bidirectional,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Ref {
    Handle(String),      // $user
    Label(String),       // #Alice
    Filter(String, String), // #Alice:email
    Name(String),        // "some name"
}

// $handle = #Label:key = value ~0.9
// parts: 
// 1. Subject Ref ($h or #L or "Name")
// 2. Key (:key)
// 3. Value (= value)
// 4. Certainty (~0.9) - optional

// rel_type(role: $ref, role: #Ref) ~0.9
static REL_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(?P<rel_type>[\w\.]+)\((?P<args>.*)\)\s*(?:\~(?P<certainty>[\d\.]+))?$").unwrap());

pub fn parse_memory_block(input: &str) -> Vec<Result<DslStatement, String>> {
    input.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty() && !line.starts_with("//"))
        .map(parse_line)
        .collect()
}

pub fn is_dsl_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with("//") {
        return false;
    }
    if !starts_with_dsl_token(trimmed) {
        return false;
    }
    parse_line(trimmed).is_ok()
}

pub(crate) fn parse_line(line: &str) -> Result<DslStatement, String> {
    let line = normalize_relation_prefix(line);
    // 1. Check for Relation: explicit "name(" pattern
    if let Some(caps) = REL_PATTERN.captures(line.as_str()) {
        let _rel_type = caps.name("rel_type").unwrap().as_str().to_string();
        let _args_str = caps.name("args").unwrap().as_str();
        
        // Parse modifiers from the rest of the line?
        // REL_PATTERN regex is `^(?P<rel_type>[\w\.]+)\((?P<args>.*)\)\s*(?:\~(?P<certainty>[\d\.]+))?$`.
        // This regex is too simple now. It consumes the suffix in the regex.
        // We should relax the regex or split manually.
        // Let's manually parse: find first `(` and matching `)`.
        
        // Manual split for modifiers after relation closing could be tricky if we use the old regex.
        // Better: Use parse_modifiers on the whole line? No, structure differs.
        
        // Let's use the regex to identify REL, but then re-parse the suffix if needed.
        // Actually, let's fix the logic to be consistent:
        // Rel: `type(args) ...modifiers`
        // Fact: `S:k = value ...modifiers`
        
        // Let's try parsing Rel manually first from the `(` index.
    }
    
    // Unified parsing approach
    let (base_stmt, modifiers) = parse_modifiers(&line);
    let clean_line = base_stmt.trim();

    // Check Rel
    if let Some(idx_open) = clean_line.find('(') {
        if let Some(idx_close) = clean_line.rfind(')') {
            if idx_close > idx_open && idx_close == clean_line.len() - 1 {
                let rel_type = clean_line[..idx_open].trim().to_string();
                let args_str = clean_line[idx_open+1..idx_close].trim();
                
                let (participants, direction) = parse_rel_args(args_str)?;

                return Ok(DslStatement::Rel(RelStmt {
                    rel_type,
                    rel_type_id: None,
                    participants,
                    direction,
                    certainty: modifiers.certainty,
                    time_expr: modifiers.time_expr,
                    scope_expr: modifiers.scope_expr,
                    source_ref: modifiers.source_ref,
                    polarity: modifiers.polarity,
                }));
            }
        }
    }

    // Shorthand name assignment: #Label: 'Name' or $handle: "Name"
    if let Some((lhs, rhs)) = clean_line.split_once(':') {
        let lhs = lhs.trim();
        let rhs = rhs.trim();
        if !lhs.is_empty() && !rhs.is_empty() && !rhs.contains('=') {
            let (value, value_quoted) = strip_quotes_with_flag(rhs);
            if value_quoted {
                return Ok(DslStatement::Fact(FactStmt {
                    subject: parse_ref(lhs),
                    key: "name".to_string(),
                    value,
                    value_quoted,
                    certainty: modifiers.certainty,
                    time_expr: modifiers.time_expr,
                    scope_expr: modifiers.scope_expr,
                    source_ref: modifiers.source_ref,
                    polarity: modifiers.polarity,
                }));
            }
        }
    }

    // Fact: S:k = value
    if let Some((lhs, rhs)) = clean_line.split_once('=') {
        let lhs = lhs.trim();
        let (value, value_quoted) = strip_quotes_with_flag(rhs.trim());
        
         let (subject_ref, key) = if let Some(idx) = lhs.find(':') {
             let (s, k) = lhs.split_at(idx);
             (parse_ref(s.trim()), k[1..].trim().to_string())
        } else if let Some(idx) = lhs.find('.') {
             let (s, k) = lhs.split_at(idx);
             (parse_ref(s.trim()), k[1..].trim().to_string())
        } else {
             return Err(format!("Fact must have a key: '{}'", lhs));
        };
        
        return Ok(DslStatement::Fact(FactStmt {
            subject: subject_ref,
            key,
            value,
            value_quoted,
            certainty: modifiers.certainty,
            time_expr: modifiers.time_expr,
            scope_expr: modifiers.scope_expr,
            source_ref: modifiers.source_ref,
            polarity: modifiers.polarity,
        }));
    }

    Err(format!("Unknown statement format: {}", line))
}

fn normalize_relation_prefix(line: &str) -> String {
    let trimmed = line.trim();
    if !trimmed.starts_with('#') {
        return trimmed.to_string();
    }
    let rest = &trimmed[1..];
    let Some(idx_open) = rest.find('(') else {
        return trimmed.to_string();
    };
    let head = rest[..idx_open].trim();
    if head.is_empty() {
        return trimmed.to_string();
    }
    if head
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
    {
        return rest.to_string();
    }
    trimmed.to_string()
}

fn starts_with_dsl_token(line: &str) -> bool {
    let first = line.chars().next().unwrap_or('\0');
    if matches!(first, '#' | '$' | '"' | '\'') {
        return true;
    }

    if let Some(idx) = line.find('(') {
        let head = line[..idx].trim();
        if !head.is_empty()
            && head
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
        {
            return true;
        }
    }

    false
}

pub struct RepairContext {
    pub now: DateTime<Utc>,
    pub assistant_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RepairOutcome {
    pub repaired_block: Option<String>,
    pub confidence: f32,
    pub repaired: bool,
    pub dropped_lines: usize,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MemoryBlockValidation {
    pub valid: bool,
    pub statement_count: usize,
    pub errors: Vec<String>,
}

pub fn repair_memory_block(raw_block: &str, ctx: &RepairContext) -> RepairOutcome {
    let mut lines_out: Vec<String> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    let mut repaired = false;
    let mut dropped_lines = 0usize;
    let mut confidence = 1.0f32;

    let mut last_stmt_idx: Option<usize> = None;
    let mut recognized_relation = false;
    let mut inferred_participant = false;
    let mut converted_prose = false;
    let mut dropped_semantic = false;
    let mut direction_change = false;
    let mut explicit_participants = false;

    let mut seen_created_by = false;

    for raw_line in raw_block.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }

        if trimmed.to_lowercase().contains("created_by") {
            seen_created_by = true;
        }

        if let Some(conf) = parse_confidence_line(trimmed) {
            if let Some(idx) = last_stmt_idx {
                lines_out[idx].push_str(&format!(" ~{:.3}", conf));
                repaired = true;
            } else {
                dropped_lines += 1;
                dropped_semantic = true;
            }
            continue;
        }

        if let Some(ts) = parse_observed_line(trimmed, ctx.now) {
            if let Some(idx) = last_stmt_idx {
                lines_out[idx].push_str(&format!(" ^{}", ts));
                repaired = true;
            } else {
                dropped_lines += 1;
                dropped_semantic = true;
            }
            continue;
        }

        let normalized = normalize_relation_prefix(trimmed);
        if normalized != trimmed {
            recognized_relation = true;
            repaired = true;
        }

        if parse_line(&normalized).is_ok() {
            if let Ok(DslStatement::Rel(rel)) = parse_line(&normalized) {
                if rel
                    .participants
                    .iter()
                    .all(|(_, r)| matches!(r, Ref::Handle(_) | Ref::Label(_)))
                {
                    explicit_participants = true;
                }
            }
            lines_out.push(normalized);
            last_stmt_idx = Some(lines_out.len().saturating_sub(1));
            continue;
        }

        if let Some(repaired_line) = repair_created_assignment(trimmed, ctx, seen_created_by) {
            converted_prose = true;
            inferred_participant = true;
            direction_change = true;
            repaired = true;
            lines_out.push(repaired_line);
            last_stmt_idx = Some(lines_out.len().saturating_sub(1));
            continue;
        }

        dropped_lines += 1;
        dropped_semantic = true;
        errors.push(format!("Unrepairable line: {}", trimmed));
    }

    if lines_out.is_empty() {
        return RepairOutcome {
            repaired_block: None,
            confidence: 0.0,
            repaired,
            dropped_lines,
            errors,
        };
    }

    if !is_finite(confidence) {
        confidence = 0.0;
    }

    if recognized_relation {
        confidence += 0.10;
    }
    if explicit_participants {
        confidence += 0.10;
    }
    if inferred_participant {
        confidence -= 0.25;
    }
    if converted_prose {
        confidence -= 0.20;
    }
    if dropped_semantic {
        confidence -= 0.15;
    }
    if direction_change {
        confidence -= 0.10;
    }

    confidence = confidence.clamp(0.0, 1.0);

    RepairOutcome {
        repaired_block: Some(lines_out.join("\n")),
        confidence,
        repaired,
        dropped_lines,
        errors,
    }
}

pub fn validate_memory_block(raw_block: &str) -> MemoryBlockValidation {
    let mut errors: Vec<String> = Vec::new();
    let mut statement_count = 0usize;

    for (idx, raw_line) in raw_block.lines().enumerate() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if !starts_with_dsl_token(trimmed) {
            errors.push(format!("line {}: non_dsl_token", idx + 1));
            continue;
        }
        if let Err(err) = parse_line(trimmed) {
            errors.push(format!("line {}: {}", idx + 1, err));
            continue;
        }
        statement_count = statement_count.saturating_add(1);
    }

    MemoryBlockValidation {
        valid: statement_count > 0 && errors.is_empty(),
        statement_count,
        errors,
    }
}

fn parse_confidence_line(line: &str) -> Option<f32> {
    let lower = line.to_lowercase();
    if !lower.starts_with("confidence") {
        return None;
    }
    let parts: Vec<&str> = line.split(|c| c == ':' || c == '=').collect();
    if parts.len() < 2 {
        return None;
    }
    let val = parts[1].trim().trim_end_matches('%');
    val.parse::<f32>().ok().map(|v| if v > 1.0 { v / 100.0 } else { v })
}

fn parse_observed_line(line: &str, now: DateTime<Utc>) -> Option<String> {
    let lower = line.to_lowercase();
    if !lower.starts_with("observed") {
        return None;
    }
    let parts: Vec<&str> = line.split(|c| c == ':' || c == '=').collect();
    if parts.len() < 2 {
        return None;
    }
    let raw = parts[1].trim().to_lowercase();
    if let Some(ts) = parse_relative_time(&raw, now) {
        return Some(ts);
    }
    if raw.len() >= 10 && raw.chars().all(|c| c.is_ascii_digit() || c == '-' || c == ':' || c == 't' || c == 'z' || c == '+') {
        return Some(raw);
    }
    None
}

fn parse_relative_time(raw: &str, now: DateTime<Utc>) -> Option<String> {
    let tokens: Vec<&str> = raw.split_whitespace().collect();
    if tokens.len() < 2 {
        return None;
    }
    let qty: i64 = tokens[0].parse().ok()?;
    let unit = tokens[1];
    let delta = match unit {
        "minute" | "minutes" => Duration::minutes(qty),
        "hour" | "hours" => Duration::hours(qty),
        "day" | "days" => Duration::days(qty),
        _ => return None,
    };
    let ts = now - delta;
    Some(ts.to_rfc3339())
}

fn repair_created_assignment(line: &str, ctx: &RepairContext, seen_created_by: bool) -> Option<String> {
    if !seen_created_by {
        return None;
    }
    let lower = line.to_lowercase();
    if !lower.starts_with("created") {
        return None;
    }
    let parts: Vec<&str> = line.split(|c| c == ':' || c == '=').collect();
    if parts.len() < 2 {
        return None;
    }
    let value = parts[1].trim().trim_matches('"').trim_matches('\'').trim();
    if value.is_empty() {
        return None;
    }
    let created_ref = if let Some(name) = ctx.assistant_name.as_deref() {
        if name.eq_ignore_ascii_case(value) {
            "$assistant".to_string()
        } else {
            format!("#{}", value)
        }
    } else {
        format!("#{}", value)
    };
    let rel_type = "created_by";
    if !is_canonical_relation(rel_type) {
        return None;
    }
    Some(format!(
        "{}(creator: $user -> created: {})",
        rel_type, created_ref
    ))
}

fn is_finite(v: f32) -> bool {
    v.is_finite()
}

fn parse_rel_args(args_str: &str) -> Result<(Vec<(String, Ref)>, Option<RelDirection>), String> {
    let args_str = args_str.trim();
    if args_str.is_empty() {
        return Ok((vec![], None));
    }

    let has_bidirectional = args_str.contains("<->");
    let has_directed = if has_bidirectional {
        args_str.replace("<->", "").contains("->")
    } else {
        args_str.contains("->")
    };
    if has_bidirectional && has_directed {
        return Err("Invalid relation: mixed arrow directions".to_string());
    }

    if has_bidirectional || has_directed {
        if args_str.contains(',') {
            return Err("Arrow relations must use exactly two participants without commas".to_string());
        }
        let token = if has_bidirectional { "<->" } else { "->" };
        let mut parts = args_str.splitn(2, token);
        let left = parts.next().unwrap_or("").trim();
        let right = parts.next().unwrap_or("").trim();
        if left.is_empty() || right.is_empty() || right.contains(token) {
            return Err("Invalid relation: malformed arrow syntax".to_string());
        }
        let left_part = parse_rel_participant(left)?;
        let right_part = parse_rel_participant(right)?;
        let direction = if has_bidirectional {
            Some(RelDirection::Bidirectional)
        } else {
            Some(RelDirection::Directed)
        };
        return Ok((vec![left_part, right_part], direction));
    }

    let mut participants = vec![];
    for arg in args_str.split(',') {
        let arg = arg.trim();
        if arg.is_empty() {
            continue;
        }
        participants.push(parse_rel_participant(arg)?);
    }
    Ok((participants, None))
}

fn parse_rel_participant(arg: &str) -> Result<(String, Ref), String> {
    let parts: Vec<&str> = arg.splitn(2, ':').collect();
    if parts.len() != 2 {
        return Err(format!("Invalid relation argument: '{}'", arg));
    }
    Ok((parts[0].trim().to_string(), parse_ref(parts[1].trim())))
}

struct ParsedModifiers {
    certainty: Option<f32>,
    time_expr: Option<TimeExpr>,
    scope_expr: Option<String>,
    source_ref: Option<String>,
    polarity: String,
}

/// Extract modifiers from the end of the line, returning (clean_content, modifiers)
fn parse_modifiers(line: &str) -> (String, ParsedModifiers) {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let mut certainty = None;
    let mut time_expr = None;
    let mut scope_expr = None;
    let mut source_ref = None;
    let mut polarity = "assert".to_string();
    
    // We iterate backwards, consuming tokens if they match a modifier pattern
    // Stop when a token doesn't match or we hit content
    
    // BUT: "value string" can be multiple tokens.
    // If we have quoted string at end, it's value. 
    // Modifiers are typically outside the semantic "content".
    // "value" ^time
    // "value with spaces" ^time
    // unquoted value with spaces ^time -> ambiguous if value contains ^time
    // We assume strict prefix for modifiers: ~ ^ @ !
    
    let mut consumed_count = 0;
    
    for token in parts.iter().rev() {
        if token.starts_with('~') {
            let raw = &token[1..];
            let normalized = if let Some(stripped) = raw.strip_suffix('%') {
                if let Ok(pct) = stripped.parse::<f32>() {
                    Some(pct / 100.0)
                } else {
                    None
                }
            } else {
                raw.parse::<f32>().ok()
            };
            if let Some(v) = normalized {
                certainty = Some(v);
                consumed_count += 1;
                continue;
            }
        }
        
        if token.starts_with('^') {
            // explicit parsing of time expr kind?
            // Spec: ^date or ^[start..end]
            let val = token[1..].to_string(); 
            let kind = if val.starts_with('[') && val.contains("..") {
                "range".to_string()
            } else if matches!(val.as_str(), "today" | "yesterday" | "this_week") {
                "relative".to_string()
            } else {
                "instant".to_string()
            };
            time_expr = Some(TimeExpr { kind, value: val });
            consumed_count += 1;
            continue;
        }
        
        if token.starts_with('@') {
            scope_expr = Some(token[1..].to_string());
            consumed_count += 1;
            continue;
        }

        if token.starts_with('<') && token.ends_with('>') && token.len() > 1 {
            source_ref = Some(token[1..token.len()-1].to_string());
            consumed_count += 1;
            continue;
        }
        
        if *token == "!deny" || *token == "!" { // ! suffix allowed
             polarity = "deny".to_string();
             consumed_count += 1;
             continue;
        }
        
        // If unrecognized, stop (it's part of the content)
        break;
    }
    
    // Reassemble the non-modifier parts
    let clean_len = parts.len() - consumed_count;
    let clean_line = parts[..clean_len].join(" ");
    
    (clean_line, ParsedModifiers {
        certainty,
        time_expr,
        scope_expr,
        source_ref,
        polarity
    })
}

fn parse_ref(s: &str) -> Ref {
    if s.starts_with('$') {
        Ref::Handle(s[1..].to_string())
    } else if s.starts_with('#') {
        Ref::Label(s[1..].to_string())
    } else {
        Ref::Name(strip_quotes(s))
    }
}

fn strip_quotes(s: &str) -> String {
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        s[1..s.len()-1].to_string()
    } else {
        s.to_string()
    }
}

fn strip_quotes_with_flag(s: &str) -> (String, bool) {
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        (s[1..s.len()-1].to_string(), true)
    } else {
        (s.to_string(), false)
    }
}

fn strip_code_fence(raw: &str) -> String {
    let trimmed = raw.trim();
    if !trimmed.starts_with("```") {
        return trimmed.to_string();
    }
    let mut lines = trimmed.lines();
    let _first = lines.next();
    let rest = lines.collect::<Vec<_>>().join("\n");
    if let Some(end_idx) = rest.rfind("```") {
        return rest[..end_idx].trim().to_string();
    }
    rest.trim().to_string()
}

fn extract_first_json_value(raw: &str) -> Option<String> {
    let mut stack: Vec<char> = Vec::new();
    let mut start: Option<usize> = None;
    let mut in_string = false;
    let mut escape = false;
    for (idx, ch) in raw.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' | '[' => {
                if stack.is_empty() {
                    start = Some(idx);
                }
                stack.push(ch);
            }
            '}' | ']' => {
                if let Some(open) = stack.pop() {
                    if (open == '{' && ch != '}') || (open == '[' && ch != ']') {
                        continue;
                    }
                    if stack.is_empty() {
                        let s = start?;
                        return Some(raw[s..=idx].to_string());
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_json_value(raw: &str) -> Option<Value> {
    let cleaned = strip_code_fence(raw);
    if cleaned.is_empty() {
        return None;
    }
    if let Ok(value) = serde_json::from_str::<Value>(&cleaned) {
        return Some(value);
    }
    let snippet = extract_first_json_value(&cleaned)?;
    serde_json::from_str::<Value>(&snippet).ok()
}

fn escape_quotes(raw: &str) -> String {
    raw.replace('"', "\\\"")
}

fn needs_quotes(raw: &str) -> bool {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return true;
    }
    trimmed.chars().any(|ch| {
        ch.is_whitespace()
            || matches!(ch, '~' | '^' | '@' | '<' | '!' | ',' | '(' | ')' | ':')
    })
}

fn format_ref(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.starts_with('$') || trimmed.starts_with('#') {
        return trimmed.to_string();
    }
    if needs_quotes(trimmed) {
        format!("\"{}\"", escape_quotes(trimmed))
    } else {
        trimmed.to_string()
    }
}

fn format_fact_value(raw: &str) -> String {
    let trimmed = raw.trim();
    if needs_quotes(trimmed) {
        format!("\"{}\"", escape_quotes(trimmed))
    } else {
        trimmed.to_string()
    }
}

fn format_confidence(val: f64) -> String {
    let mut normalized = val;
    if normalized > 1.0 && normalized <= 100.0 {
        normalized /= 100.0;
    }
    let clamped = normalized.clamp(0.0, 1.0);
    let mut out = format!("{:.2}", clamped);
    while out.contains('.') && out.ends_with('0') {
        out.pop();
    }
    if out.ends_with('.') {
        out.pop();
    }
    out
}

fn format_modifier_tokens(
    certainty: Option<f64>,
    time_expr: Option<String>,
    scope_expr: Option<String>,
    source_ref: Option<String>,
    polarity: Option<String>,
    polarity_bool: Option<bool>,
) -> String {
    let mut tokens: Vec<String> = Vec::new();
    if let Some(certainty) = certainty {
        if certainty.is_finite() {
            tokens.push(format!("~{}", format_confidence(certainty)));
        }
    }
    if let Some(time) = time_expr {
        let t = time.trim();
        if !t.is_empty() {
            if t.starts_with('^') {
                tokens.push(t.to_string());
            } else {
                tokens.push(format!("^{}", t));
            }
        }
    }
    if let Some(scope) = scope_expr {
        let s = scope.trim();
        if !s.is_empty() {
            if s.starts_with('@') {
                tokens.push(s.to_string());
            } else {
                tokens.push(format!("@{}", s));
            }
        }
    }
    if let Some(source) = source_ref {
        let s = source.trim();
        if !s.is_empty() {
            if s.starts_with('<') && s.ends_with('>') {
                tokens.push(s.to_string());
            } else {
                tokens.push(format!("<{}>", s));
            }
        }
    }
    let mut deny = false;
    if let Some(polarity) = polarity {
        let lower = polarity.trim().to_lowercase();
        if lower == "deny" || lower == "false" || lower == "negate" || lower == "negative" {
            deny = true;
        }
    }
    if let Some(polarity_bool) = polarity_bool {
        if !polarity_bool {
            deny = true;
        }
    }
    if deny {
        tokens.push("!".to_string());
    }
    if tokens.is_empty() {
        String::new()
    } else {
        format!(" {}", tokens.join(" "))
    }
}

fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_string()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn participants_from_value(value: &Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    match value {
        Value::Array(items) => {
            for item in items {
                if let Value::Object(obj) = item {
                    let role = obj
                        .get("role")
                        .or_else(|| obj.get("slot"))
                        .and_then(value_as_string)
                        .unwrap_or_default();
                    let reference = obj
                        .get("ref")
                        .or_else(|| obj.get("value"))
                        .or_else(|| obj.get("entity"))
                        .and_then(value_as_string)
                        .unwrap_or_default();
                    if !role.trim().is_empty() && !reference.trim().is_empty() {
                        out.push((role.trim().to_string(), reference.trim().to_string()));
                    }
                }
            }
        }
        Value::Object(obj) => {
            for (role, value) in obj {
                if let Some(reference) = value_as_string(value) {
                    let role_trimmed = role.trim();
                    let reference_trimmed = reference.trim();
                    if !role_trimmed.is_empty() && !reference_trimmed.is_empty() {
                        out.push((role_trimmed.to_string(), reference_trimmed.to_string()));
                    }
                }
            }
        }
        _ => {}
    }
    out
}

fn build_fact_line(obj: &serde_json::Map<String, Value>) -> Option<String> {
    let subject = obj
        .get("subject")
        .or_else(|| obj.get("entity"))
        .or_else(|| obj.get("ref"))
        .and_then(value_as_string)?;
    let key = obj.get("key").and_then(value_as_string)?;
    let value = obj.get("value").and_then(value_as_string)?;
    if subject.trim().is_empty() || key.trim().is_empty() || value.trim().is_empty() {
        return None;
    }
    let certainty = obj
        .get("certainty")
        .or_else(|| obj.get("confidence"))
        .and_then(|v| v.as_f64());
    let time_expr = obj.get("time").and_then(value_as_string);
    let scope_expr = obj.get("scope").and_then(value_as_string);
    let source_ref = obj.get("source").and_then(value_as_string);
    let polarity = obj.get("polarity").and_then(value_as_string);
    let polarity_bool = obj.get("assert").and_then(|v| v.as_bool());
    let modifiers = format_modifier_tokens(certainty, time_expr, scope_expr, source_ref, polarity, polarity_bool);
    Some(format!(
        "{}:{} = {}{}",
        format_ref(&subject),
        key.trim(),
        format_fact_value(&value),
        modifiers
    ))
}

fn build_rel_line(obj: &serde_json::Map<String, Value>) -> Option<String> {
    let rel_type = obj
        .get("rel_type")
        .or_else(|| obj.get("relation"))
        .or_else(|| obj.get("type"))
        .or_else(|| obj.get("kind"))
        .and_then(value_as_string)?;
    let participants_val = obj
        .get("participants")
        .or_else(|| obj.get("args"))
        .or_else(|| obj.get("roles"))?;
    let participants = participants_from_value(participants_val);
    if participants.len() < 2 {
        return None;
    }
    let certainty = obj
        .get("certainty")
        .or_else(|| obj.get("confidence"))
        .and_then(|v| v.as_f64());
    let time_expr = obj.get("time").and_then(value_as_string);
    let scope_expr = obj.get("scope").and_then(value_as_string);
    let source_ref = obj.get("source").and_then(value_as_string);
    let polarity = obj.get("polarity").and_then(value_as_string);
    let polarity_bool = obj.get("assert").and_then(|v| v.as_bool());
    let direction = obj
        .get("direction")
        .and_then(value_as_string)
        .map(|s| s.to_lowercase());
    let modifiers = format_modifier_tokens(certainty, time_expr, scope_expr, source_ref, polarity, polarity_bool);
    let rendered = if participants.len() == 2 && direction.as_deref().is_some() {
        let token = match direction.as_deref().unwrap_or("") {
            "bidirectional" | "<->" | "both" => "<->",
            "directed" | "->" | "forward" => "->",
            _ => "",
        };
        if !token.is_empty() {
            format!(
                "{}({}: {} {} {}: {}){}",
                rel_type.trim(),
                participants[0].0,
                format_ref(&participants[0].1),
                token,
                participants[1].0,
                format_ref(&participants[1].1),
                modifiers
            )
        } else {
            String::new()
        }
    } else {
        let args = participants
            .iter()
            .map(|(role, reference)| format!("{}: {}", role, format_ref(reference)))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}({}){}", rel_type.trim(), args, modifiers)
    };
    if rendered.is_empty() {
        None
    } else {
        Some(rendered)
    }
}

fn statements_from_value(value: &Value) -> Vec<String> {
    let mut lines = Vec::new();
    let handle_statement_obj = |obj: &serde_json::Map<String, Value>, lines: &mut Vec<String>| {
        let kind = obj
            .get("type")
            .or_else(|| obj.get("kind"))
            .and_then(value_as_string)
            .unwrap_or_default()
            .to_lowercase();
        if kind == "relation" || kind == "rel" {
            if let Some(line) = build_rel_line(obj) {
                lines.push(line);
                return;
            }
        }
        if kind == "fact" {
            if let Some(line) = build_fact_line(obj) {
                lines.push(line);
                return;
            }
        }
        if obj.get("rel_type").is_some() || obj.get("participants").is_some() {
            if let Some(line) = build_rel_line(obj) {
                lines.push(line);
                return;
            }
        }
        if obj.get("key").is_some() && obj.get("value").is_some() {
            if let Some(line) = build_fact_line(obj) {
                lines.push(line);
            }
        }
    };
    match value {
        Value::Array(items) => {
            for item in items {
                if let Value::Object(obj) = item {
                    handle_statement_obj(obj, &mut lines);
                }
            }
        }
        Value::Object(obj) => {
            if let Some(statements) = obj.get("statements").and_then(|v| v.as_array()) {
                for item in statements {
                    if let Value::Object(obj) = item {
                        handle_statement_obj(obj, &mut lines);
                    }
                }
            }
            if let Some(facts) = obj.get("facts").and_then(|v| v.as_array()) {
                for item in facts {
                    if let Value::Object(obj) = item {
                        if let Some(line) = build_fact_line(obj) {
                            lines.push(line);
                        }
                    }
                }
            }
            if let Some(rels) = obj.get("relations").and_then(|v| v.as_array()) {
                for item in rels {
                    if let Value::Object(obj) = item {
                        if let Some(line) = build_rel_line(obj) {
                            lines.push(line);
                        }
                    }
                }
            }
        }
        _ => {}
    }
    lines
}

pub(crate) fn memory_json_to_dsl(raw: &str) -> Option<String> {
    let value = parse_json_value(raw)?;
    let lines = statements_from_value(&value);
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rel() {
        let input = "likes(subject: $user, object: #Pizza) ~0.8";
        let res = parse_line(input).unwrap();
        if let DslStatement::Rel(r) = res {
            assert_eq!(r.rel_type, "likes");
            assert_eq!(r.certainty, Some(0.8));
            assert_eq!(r.participants.len(), 2);
            assert_eq!(r.participants[0].0, "subject");
            matches!(r.participants[0].1, Ref::Handle(_));
            assert_eq!(r.direction, None);
        } else {
            panic!("Expected Rel");
        }
    }

    #[test]
    fn test_parse_fact_colon() {
        let input = "$user:current_goal = 'Fix memory'";
        let res = parse_line(input).unwrap();
        if let DslStatement::Fact(f) = res {
            matches!(f.subject, Ref::Handle(_));
            assert_eq!(f.key, "current_goal");
            assert_eq!(f.value, "Fix memory");
        } else {
            panic!("Expected Fact");
        }
    }
    
    #[test]
    fn test_parse_fact_dot() {
        let input = "#Alice.email = bob@example.com";
        let res = parse_line(input).unwrap();
        if let DslStatement::Fact(f) = res {
            matches!(f.subject, Ref::Label(_));
            assert_eq!(f.key, "email");
            assert_eq!(f.value, "bob@example.com");
        } else {
            panic!("Expected Fact");
        }
    }

    #[test]
    fn test_parse_fact_time() {
        let input = "$user:location = 'Home' ^2025-01-01";
        let res = parse_line(input).unwrap();
        if let DslStatement::Fact(f) = res {
            assert!(f.time_expr.is_some());
            assert_eq!(f.time_expr.unwrap().value, "2025-01-01");
        } else {
            panic!("Expected Fact");
        }
    }

    #[test]
    fn test_parse_fact_scope() {
        let input = "$user:secret = 'hidden' @session";
        let res = parse_line(input).unwrap();
        if let DslStatement::Fact(f) = res {
            assert_eq!(f.scope_expr, Some("session".to_string()));
        } else {
            panic!("Expected Fact");
        }
    }

    #[test]
    fn test_parse_rel_combined() {
        let input = "met(subject: $user, object: #Alice) ~0.8 ^yesterday @global <http://example.com>";
        let res = parse_line(input).unwrap();
        if let DslStatement::Rel(r) = res {
            assert_eq!(r.certainty, Some(0.8));
            assert!(r.time_expr.is_some());
            assert_eq!(r.scope_expr, Some("global".to_string()));
            assert_eq!(r.source_ref, Some("http://example.com".to_string()));
            assert_eq!(r.direction, None);
        } else {
            panic!("Expected Rel");
        }
    }

    #[test]
    fn test_parse_percent_confidence() {
        let input = "married_to(person: $user, person: #Moxie) ~100%";
        let res = parse_line(input).unwrap();
        if let DslStatement::Rel(r) = res {
            assert_eq!(r.certainty, Some(1.0));
            assert_eq!(r.direction, None);
        } else {
            panic!("Expected Rel");
        }
    }

    #[test]
    fn test_parse_rel_directed_arrow() {
        let input = "parent_of(parent: #MisterBlack -> child: #Harlow)";
        let res = parse_line(input).unwrap();
        if let DslStatement::Rel(r) = res {
            assert_eq!(r.rel_type, "parent_of");
            assert_eq!(r.participants.len(), 2);
            assert_eq!(r.participants[0].0, "parent");
            assert_eq!(r.participants[1].0, "child");
            assert_eq!(r.direction, Some(RelDirection::Directed));
        } else {
            panic!("Expected Rel");
        }
    }

    #[test]
    fn test_parse_rel_bidirectional_arrow() {
        let input = "friends(person: #A <-> person: #B)";
        let res = parse_line(input).unwrap();
        if let DslStatement::Rel(r) = res {
            assert_eq!(r.rel_type, "friends");
            assert_eq!(r.participants.len(), 2);
            assert_eq!(r.direction, Some(RelDirection::Bidirectional));
        } else {
            panic!("Expected Rel");
        }
    }

    #[test]
    fn test_parse_rel_mixed_arrows_rejected() {
        let input = "knows(person: #A -> person: #B <-> person: #C)";
        let res = parse_line(input);
        assert!(res.is_err());
    }

    #[test]
    fn test_parse_relative_time() {
        let input = "$user:status = 'busy' ^yesterday";
        let res = parse_line(input).unwrap();
        if let DslStatement::Fact(f) = res {
            let time_expr = f.time_expr.expect("time expr");
            assert_eq!(time_expr.kind, "relative");
            assert_eq!(time_expr.value, "yesterday");
        } else {
            panic!("Expected Fact");
        }
    }

    #[test]
    fn test_parse_fact_source_and_deny() {
        let input = "$user:status = \"busy\" <http://example.com> !deny";
        let res = parse_line(input).unwrap();
        if let DslStatement::Fact(f) = res {
            assert_eq!(f.source_ref, Some("http://example.com".to_string()));
            assert_eq!(f.polarity, "deny");
        } else {
            panic!("Expected Fact");
        }
    }

    #[test]
    fn test_parse_rel_deny_shorthand() {
        let input = "knows(person: $user, person: #Alice) !";
        let res = parse_line(input).unwrap();
        if let DslStatement::Rel(r) = res {
            assert_eq!(r.polarity, "deny");
        } else {
            panic!("Expected Rel");
        }
    }

    #[test]
    fn test_parse_rel_directed_with_modifiers() {
        let input = "owns(owner: #Jane Doe -> item: #Acme Rocket) ~0.9 ^2025-01-01 @global <http://example.com> !deny";
        let res = parse_line(input).unwrap();
        if let DslStatement::Rel(r) = res {
            assert_eq!(r.rel_type, "owns");
            assert_eq!(r.direction, Some(RelDirection::Directed));
            assert_eq!(r.certainty, Some(0.9));
            assert_eq!(r.scope_expr, Some("global".to_string()));
            assert_eq!(r.source_ref, Some("http://example.com".to_string()));
            assert_eq!(r.polarity, "deny");
            let time_expr = r.time_expr.expect("time expr");
            assert_eq!(time_expr.value, "2025-01-01");
        } else {
            panic!("Expected Rel");
        }
    }

    #[test]
    fn test_memory_json_to_dsl_basic() {
        let raw = r##"{"facts":[{"subject":"$user","key":"father_name","value":"David","certainty":0.9}],"relations":[{"rel_type":"parent_of","participants":[{"role":"parent","ref":"#David"},{"role":"child","ref":"$user"}],"direction":"directed","certainty":0.8}]}"##;
        let dsl = memory_json_to_dsl(raw).expect("dsl");
        assert!(dsl.contains("$user:father_name"));
        assert!(dsl.contains("parent_of"));
        assert!(dsl.contains("parent: #David"));
        assert!(dsl.contains("child: $user"));
    }

    #[test]
    fn test_is_dsl_line_detects_valid_statements() {
        assert!(is_dsl_line("#Alice:age = \"30\""));
        assert!(is_dsl_line("friends(person: #A <-> person: #B)"));
    }

    #[test]
    fn test_is_dsl_line_avoids_false_positives() {
        assert!(!is_dsl_line("Note: a = b"));
        assert!(!is_dsl_line("Email me at bob@example.com #urgent"));
        assert!(!is_dsl_line("When I say (hello), I mean hi."));
    }
}
