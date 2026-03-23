use crate::models::Settings;
use once_cell::sync::Lazy;
use tiktoken_rs::CoreBPE;

pub const DEFAULT_CHARS_PER_TOKEN: f32 = 4.0;
const DEFAULT_CONTEXT_LIMIT_TOKENS: i32 = 16_384;

static BPE: Lazy<Option<CoreBPE>> = Lazy::new(|| tiktoken_rs::cl100k_base().ok());

pub fn context_limit_tokens(settings: &Settings) -> usize {
    settings
        .model_context_limit
        .unwrap_or(DEFAULT_CONTEXT_LIMIT_TOKENS)
        .max(4096) as usize
}

pub fn safety_margin_tokens(limit: usize) -> usize {
    ((limit as f32) * 0.10).round() as usize
}

pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    if let Some(bpe) = BPE.as_ref() {
        return bpe.encode_with_special_tokens(text).len().max(1);
    }
    let chars = text.chars().count() as f32;
    let estimate = (chars / DEFAULT_CHARS_PER_TOKEN).ceil() as usize;
    estimate.max(1)
}

pub fn estimate_tokens_for_strings<'a, I>(strings: I) -> usize
where
    I: IntoIterator<Item = &'a str>,
{
    strings
        .into_iter()
        .map(estimate_tokens)
        .sum()
}

pub fn tokens_from_char_budget(char_budget: usize) -> usize {
    if char_budget == 0 {
        return 0;
    }
    if let Some(bpe) = BPE.as_ref() {
        let sample: String = "a".repeat(char_budget.min(2048));
        let tokens = bpe.encode_with_special_tokens(&sample).len();
        if char_budget <= 2048 {
            return tokens.max(1);
        }
        let scale = char_budget as f32 / 2048.0;
        return ((tokens as f32) * scale).ceil().max(1.0) as usize;
    }
    let estimate = (char_budget as f32 / DEFAULT_CHARS_PER_TOKEN).ceil() as usize;
    estimate.max(1)
}

pub fn truncate_to_token_budget(text: &str, token_budget: usize) -> (String, bool) {
    if token_budget == 0 {
        return (String::new(), !text.is_empty());
    }
    if let Some(bpe) = BPE.as_ref() {
        let tokens = bpe.encode_with_special_tokens(text);
        if tokens.len() <= token_budget {
            return (text.to_string(), false);
        }
        let truncated = bpe
            .decode(tokens[..token_budget].to_vec())
            .unwrap_or_default();
        return (truncated, true);
    }
    let estimated = estimate_tokens(text);
    if estimated <= token_budget {
        return (text.to_string(), false);
    }
    let original_chars = text.chars().count() as f32;
    let ratio = (token_budget as f32 / estimated as f32).min(1.0).max(0.0);
    let target_chars = (original_chars * ratio).floor() as usize;
    let target_chars = target_chars.max(1);
    let truncated: String = text.chars().take(target_chars).collect();
    (truncated, true)
}
