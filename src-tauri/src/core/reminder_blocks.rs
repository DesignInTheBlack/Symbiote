use once_cell::sync::Lazy;
use regex::Regex;
use sqlx::{Pool, Sqlite};

const REMINDER_BLOCK_START: &str = "```reminder";
const REMINDER_BLOCK_END: &str = "```";

static REMINDER_BLOCK_MD: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)```reminder\s*(.*?)\s*```").unwrap());

#[derive(Debug, Clone)]
pub struct ReminderSpec {
    pub content: String,
    pub due_in: String,
    pub reminder_type: String,
}

pub fn extract_reminder_blocks(output: &str) -> Vec<String> {
    let mut blocks = vec![];

    for cap in REMINDER_BLOCK_MD.captures_iter(output) {
        if let Some(m) = cap.get(1) {
            blocks.push(m.as_str().to_string());
        }
    }

    blocks
}

pub fn strip_reminder_blocks(output: &str) -> String {
    let s = REMINDER_BLOCK_MD.replace_all(output, "");
    let mut cleaned = s.to_string();
    while cleaned.contains("\n\n") {
        cleaned = cleaned.replace("\n\n", "\n");
    }
    cleaned.trim().to_string()
}

pub fn append_reminder_markers(output: &str, specs: &[ReminderSpec]) -> String {
    if specs.is_empty() {
        return output.to_string();
    }

    let mut out = output.trim().to_string();
    let mut markers = Vec::new();

    for spec in specs {
        markers.push(format!(
            "[[REMINDER:CREATED content=\"{}\" due_in=\"{}\" type=\"{}\"]]",
            escape_marker_value(&spec.content),
            escape_marker_value(&spec.due_in),
            escape_marker_value(&spec.reminder_type)
        ));
    }

    if !out.is_empty() {
        out.push('\n');
    }
    out.push_str(&markers.join("\n"));
    out
}

fn escape_marker_value(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

pub fn parse_reminder_blocks(output: &str) -> Vec<ReminderSpec> {
    let mut specs = Vec::new();

    for block in extract_reminder_blocks(output) {
        if let Some(spec) = parse_reminder_block(&block) {
            specs.push(spec);
        }
    }

    specs
}

fn parse_reminder_block(block: &str) -> Option<ReminderSpec> {
    let mut content: Option<String> = None;
    let mut due_in: Option<String> = None;
    let mut reminder_type: Option<String> = None;

    for line in block.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("#") || line.starts_with("//") {
            continue;
        }

        let (key, value) = match split_kv(line) {
            Some(kv) => kv,
            None => continue,
        };

        let key = key.to_lowercase();
        let value = strip_quotes(value);

        match key.as_str() {
            "remind" | "content" => content = Some(value),
            "due_in" | "due" => due_in = Some(value),
            "type" => reminder_type = Some(value.to_uppercase()),
            _ => {}
        }
    }

    let content = content?;
    let due_in = due_in?;
    let reminder_type = reminder_type.unwrap_or_else(|| "REMINDER".to_string());

    Some(ReminderSpec {
        content,
        due_in,
        reminder_type,
    })
}

fn split_kv(line: &str) -> Option<(&str, &str)> {
    for (idx, ch) in line.char_indices() {
        if ch == ':' || ch == '=' {
            let key = line[..idx].trim();
            let value = line[idx + 1..].trim();
            if key.is_empty() || value.is_empty() {
                return None;
            }
            return Some((key, value));
        }
    }
    None
}

fn strip_quotes(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.len() >= 2 {
        let first = trimmed.chars().next().unwrap();
        let last = trimmed.chars().last().unwrap();
        if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
            return trimmed[1..trimmed.len() - 1].to_string();
        }
    }
    trimmed.to_string()
}

pub async fn create_reminder(
    pool: &Pool<Sqlite>,
    content: &str,
    due_in: &str,
    reminder_type: &str,
) -> Result<String, String> {
    let duration = parse_duration_string(due_in)
        .map_err(|e| format!("Invalid duration '{}': {}", due_in, e))?;

    let now = chrono::Utc::now();
    let due_at = now + duration;
    let reminder_id = uuid::Uuid::new_v4().to_string();

    let due_at_ts = due_at.timestamp();
    let created_at_ts = now.timestamp();

    sqlx::query("INSERT INTO reminders (id, content, due_at, type, status, created_at) VALUES (?, ?, ?, ?, ?, ?)")
        .bind(&reminder_id)
        .bind(content)
        .bind(due_at_ts)
        .bind(reminder_type)
        .bind("PENDING")
        .bind(created_at_ts)
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;

    println!("[REMINDER] Created '{}' due at {} (ts: {})", content, due_at, due_at_ts);
    Ok(reminder_id)
}

fn parse_duration_string(s: &str) -> Result<chrono::Duration, String> {
    let s = s.trim().to_lowercase();

    let digits_end = s.find(|c: char| !c.is_numeric()).unwrap_or(s.len());
    let (num_str, suffix_str) = s.split_at(digits_end);
    let num = num_str.trim().parse::<i64>().map_err(|_| "Invalid number")?;

    let suffix = suffix_str.trim();

    match suffix {
        "m" | "min" | "mins" | "minute" | "minutes" => Ok(chrono::Duration::minutes(num)),
        "h" | "hr" | "hrs" | "hour" | "hours" => Ok(chrono::Duration::hours(num)),
        "s" | "sec" | "secs" | "second" | "seconds" => Ok(chrono::Duration::seconds(num)),
        "" => Ok(chrono::Duration::minutes(num)),
        _ => Err(format!("Unknown duration unit: {}", suffix)),
    }
}

pub struct ReminderStreamFilter {
    buffer: String,
    in_block: bool,
}

impl ReminderStreamFilter {
    pub fn new() -> Self {
        Self { buffer: String::new(), in_block: false }
    }

    pub fn filter_chunk(&mut self, chunk: &str) -> String {
        self.buffer.push_str(chunk);
        let mut output = String::new();

        loop {
            if self.in_block {
                if let Some(end_idx) = self.buffer.find(REMINDER_BLOCK_END) {
                    self.buffer = self.buffer[end_idx + REMINDER_BLOCK_END.len()..].to_string();
                    self.in_block = false;
                    continue;
                }

                let keep = REMINDER_BLOCK_END.len() - 1;
                if self.buffer.len() > keep {
                    let start_idx = clamp_char_boundary(&self.buffer, self.buffer.len().saturating_sub(keep));
                    self.buffer = self.buffer[start_idx..].to_string();
                }
                break;
            } else if let Some(start_idx) = self.buffer.find(REMINDER_BLOCK_START) {
                output.push_str(&self.buffer[..start_idx]);
                self.buffer = self.buffer[start_idx + REMINDER_BLOCK_START.len()..].to_string();
                self.in_block = true;
                continue;
            } else {
                let keep = REMINDER_BLOCK_START.len() - 1;
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
    use super::*;

    #[test]
    fn parses_reminder_block() {
        let text = "Hello\n```reminder\nremind: \"take a break\"\ndue_in: \"5m\"\ntype: \"REMINDER\"\n```\nBye";
        let specs = parse_reminder_blocks(text);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].content, "take a break");
        assert_eq!(specs[0].due_in, "5m");
        assert_eq!(specs[0].reminder_type, "REMINDER");
    }

    #[test]
    fn strips_reminder_block() {
        let text = "Hello\n```reminder\nremind: x\ndue_in: 1m\n```\nBye";
        let stripped = strip_reminder_blocks(text);
        assert_eq!(stripped, "Hello\nBye");
    }

    #[test]
    fn stream_filter_hides_reminder_block() {
        let mut filter = ReminderStreamFilter::new();
        let a = filter.filter_chunk("Hello ```rem");
        let b = filter.filter_chunk("inder\nremind: x\ndue_in: 1m\n``` World");
        let c = filter.finalize();
        assert_eq!(format!("{}{}{}", a, b, c), "Hello  World");
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
