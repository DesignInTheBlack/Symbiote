use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;
use sqlx::SqlitePool;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SensitivityLevel {
    Pii,
    Phi,
}

impl SensitivityLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            SensitivityLevel::Pii => "pii",
            SensitivityLevel::Phi => "phi",
        }
    }
}

static EMAIL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        .unwrap_or_else(|_| Regex::new(r"$^").unwrap())
});
static PHONE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
        .unwrap_or_else(|_| Regex::new(r"$^").unwrap())
});
static SSN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap_or_else(|_| Regex::new(r"$^").unwrap()));

const HEALTH_KEYWORDS: &[&str] = &[
    "diagnosed",
    "diagnosis",
    "medical",
    "medication",
    "prescribed",
    "hospital",
    "clinic",
    "therapy",
    "symptom",
    "condition",
    "treatment",
    "patient",
    "doctor",
];

pub fn detect_sensitivity(text: &str) -> Option<SensitivityLevel> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if EMAIL_RE.is_match(trimmed) || PHONE_RE.is_match(trimmed) || SSN_RE.is_match(trimmed) {
        return Some(SensitivityLevel::Pii);
    }
    let lowered = trimmed.to_lowercase();
    if HEALTH_KEYWORDS.iter().any(|kw| lowered.contains(kw)) {
        return Some(SensitivityLevel::Phi);
    }
    None
}

fn merge_level(current: Option<SensitivityLevel>, next: SensitivityLevel) -> SensitivityLevel {
    match (current, next) {
        (Some(SensitivityLevel::Phi), _) => SensitivityLevel::Phi,
        (_, SensitivityLevel::Phi) => SensitivityLevel::Phi,
        _ => SensitivityLevel::Pii,
    }
}

pub fn redact_sensitive_text(text: &str) -> (String, Option<SensitivityLevel>) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return (text.to_string(), None);
    }

    let mut sensitivity: Option<SensitivityLevel> = None;
    let mut redacted = text.to_string();

    if EMAIL_RE.is_match(&redacted) {
        redacted = EMAIL_RE.replace_all(&redacted, "[REDACTED]").into_owned();
        sensitivity = Some(merge_level(sensitivity, SensitivityLevel::Pii));
    }
    if PHONE_RE.is_match(&redacted) {
        redacted = PHONE_RE.replace_all(&redacted, "[REDACTED]").into_owned();
        sensitivity = Some(merge_level(sensitivity, SensitivityLevel::Pii));
    }
    if SSN_RE.is_match(&redacted) {
        redacted = SSN_RE.replace_all(&redacted, "[REDACTED]").into_owned();
        sensitivity = Some(merge_level(sensitivity, SensitivityLevel::Pii));
    }

    let lowered = trimmed.to_lowercase();
    if HEALTH_KEYWORDS.iter().any(|kw| lowered.contains(kw)) {
        sensitivity = Some(merge_level(sensitivity, SensitivityLevel::Phi));
        redacted = "[REDACTED]".to_string();
    }

    (redacted, sensitivity)
}

pub fn redact_sensitive_json(value: &Value) -> (Value, Option<SensitivityLevel>) {
    fn redact_value(value: &Value, sensitivity: &mut Option<SensitivityLevel>) -> Value {
        match value {
            Value::String(s) => {
                let (redacted, level) = redact_sensitive_text(s);
                if let Some(level) = level {
                    *sensitivity = Some(merge_level(*sensitivity, level));
                }
                Value::String(redacted)
            }
            Value::Array(items) => {
                Value::Array(items.iter().map(|item| redact_value(item, sensitivity)).collect())
            }
            Value::Object(map) => {
                let mut out = serde_json::Map::new();
                for (key, val) in map.iter() {
                    out.insert(key.clone(), redact_value(val, sensitivity));
                }
                Value::Object(out)
            }
            _ => value.clone(),
        }
    }

    let mut sensitivity: Option<SensitivityLevel> = None;
    let redacted = redact_value(value, &mut sensitivity);
    (redacted, sensitivity)
}

pub async fn phi_consent_allowed(pool: &SqlitePool, conversation_id: Option<&str>) -> bool {
    if let Some(conversation_id) = conversation_id {
        if let Ok(Some(enabled)) = sqlx::query_scalar::<_, i64>(
            "SELECT enabled FROM phi_consent_scopes WHERE conversation_id = ?",
        )
        .bind(conversation_id)
        .fetch_optional(pool)
        .await
        {
            return enabled != 0;
        }
    }
    sqlx::query_scalar("SELECT phi_consent FROM settings WHERE id = 1")
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
        .map(|val: i64| val != 0)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_pii_email() {
        let text = "Contact me at jane.doe@example.com";
        assert_eq!(detect_sensitivity(text), Some(SensitivityLevel::Pii));
        let (redacted, level) = redact_sensitive_text(text);
        assert!(redacted.contains("[REDACTED]"));
        assert_eq!(level, Some(SensitivityLevel::Pii));
    }

    #[test]
    fn detects_phi_keywords() {
        let text = "The patient was diagnosed and prescribed medication.";
        assert_eq!(detect_sensitivity(text), Some(SensitivityLevel::Phi));
        let (redacted, level) = redact_sensitive_text(text);
        assert_eq!(redacted, "[REDACTED]");
        assert_eq!(level, Some(SensitivityLevel::Phi));
    }
}
