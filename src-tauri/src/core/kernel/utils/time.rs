use chrono::{DateTime, Duration, NaiveDateTime, Utc};

pub(crate) fn timestamp_from_str(raw: &str) -> Option<DateTime<Utc>> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(raw) {
        return Some(dt.with_timezone(&Utc));
    }
    if let Ok(naive) = NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S") {
        return Some(DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc));
    }
    None
}

pub(crate) fn timestamp_expired(raw: Option<&str>, now: DateTime<Utc>) -> bool {
    raw.and_then(timestamp_from_str)
        .map(|ts| ts <= now)
        .unwrap_or(false)
}

pub(crate) fn compute_expires_at(now: DateTime<Utc>, secs: i64) -> String {
    (now + Duration::seconds(secs)).to_rfc3339()
}
