use once_cell::sync::Lazy;
use regex::Regex;

static FEEDBACK_PREFIX_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^\s*(?:\[feedback\]|\(feedback\)|feedback)\s*[:>\-]?\s*").unwrap()
});

pub fn extract_explicit_feedback(input: &str) -> (String, bool) {
    if let Some(mat) = FEEDBACK_PREFIX_RE.find(input) {
        let trimmed = input[mat.end()..].trim_start();
        if trimmed.is_empty() {
            return (input.trim().to_string(), true);
        }
        return (trimmed.to_string(), true);
    }
    (input.to_string(), false)
}

pub fn is_explicit_feedback(input: &str) -> bool {
    FEEDBACK_PREFIX_RE.is_match(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_feedback_prefix() {
        let (cleaned, explicit) = extract_explicit_feedback("feedback: this was helpful");
        assert!(explicit);
        assert_eq!(cleaned, "this was helpful");
    }

    #[test]
    fn extract_bracket_feedback_prefix() {
        let (cleaned, explicit) = extract_explicit_feedback("[feedback] needs more detail");
        assert!(explicit);
        assert_eq!(cleaned, "needs more detail");
    }

    #[test]
    fn extract_non_feedback() {
        let (cleaned, explicit) = extract_explicit_feedback("regular input");
        assert!(!explicit);
        assert_eq!(cleaned, "regular input");
    }

    #[test]
    fn detects_explicit_feedback_prefix() {
        assert!(is_explicit_feedback("feedback: great job"));
        assert!(is_explicit_feedback("[feedback] needs more detail"));
        assert!(is_explicit_feedback("(feedback) not quite there"));
        assert!(!is_explicit_feedback("this is not feedback"));
    }
}
