pub struct PersonaBounds {
    pub min: f32,
    pub max: f32,
}

pub const PERSONA_TONE: PersonaBounds = PersonaBounds { min: 0.2, max: 0.8 };
pub const PERSONA_VERBOSITY: PersonaBounds = PersonaBounds { min: 0.2, max: 0.7 };
pub const PERSONA_DIRECTNESS: PersonaBounds = PersonaBounds { min: 0.3, max: 0.9 };
pub const PERSONA_FORMALITY: PersonaBounds = PersonaBounds { min: 0.2, max: 0.8 };
pub const PERSONA_INITIATIVE: PersonaBounds = PersonaBounds { min: 0.2, max: 0.8 };

pub const MAX_DELTA_PER_REFLECTION: f32 = 0.05;
pub const MAX_LARGE_DELTA_PER_REFLECTION: f32 = 0.1;
pub const MAX_TOTAL_DAILY_DELTA: f32 = 0.2;
pub const SELF_EVIDENCE_STALE_AFTER_HOURS: i64 = 72;

pub fn clamp_value(value: f32, bounds: &PersonaBounds) -> f32 {
    value.max(bounds.min).min(bounds.max)
}
