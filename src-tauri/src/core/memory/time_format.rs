use chrono::{DateTime, Utc, Duration, Datelike, Weekday};

/// Format a timestamp as human-readable relative time
/// Examples: "just now", "2 hours ago", "last Saturday", "January 5th"
pub fn format_relative_time(timestamp: DateTime<Utc>, now: DateTime<Utc>) -> String {
    let diff = now.signed_duration_since(timestamp);
    
    // Future times? Just show exact date
    if diff < Duration::zero() {
         return timestamp.format("%B %e, %Y").to_string();
    }
    
    if diff < Duration::seconds(60) {
        return "just now".to_string();
    }
    
    if diff < Duration::hours(1) {
        let mins = diff.num_minutes();
        return format!("{} minute{} ago", mins, if mins == 1 { "" } else { "s" });
    }
    
    if diff < Duration::hours(24) {
        let hours = diff.num_hours();
        return format!("{} hour{} ago", hours, if hours == 1 { "" } else { "s" });
    }
    
    if diff < Duration::days(2) {
        return format!("yesterday at {}", timestamp.format("%l:%M %p").to_string().trim());
    }
    
    if diff < Duration::days(7) {
        let weekday = match timestamp.weekday() {
            Weekday::Mon => "Monday",
            Weekday::Tue => "Tuesday",
            Weekday::Wed => "Wednesday",
            Weekday::Thu => "Thursday",
            Weekday::Fri => "Friday",
            Weekday::Sat => "Saturday",
            Weekday::Sun => "Sunday",
        };
        return format!("last {} at {}", weekday, timestamp.format("%l:%M %p").to_string().trim());
    }
    
    if diff < Duration::days(30) {
        let days = diff.num_days();
        return format!("{} days ago", days);
    }
    
    if diff < Duration::days(365) {
        return timestamp.format("%B %e").to_string(); // "January 5"
    }
    
    timestamp.format("%B %e, %Y").to_string() // "January 5, 2025"
}

/// Format for "when did you tell me X?" queries
pub fn format_when_told(observed_at: &str, now: DateTime<Utc>) -> String {
    match DateTime::parse_from_rfc3339(observed_at) {
        Ok(ts) => format_relative_time(ts.with_timezone(&Utc), now),
        Err(_) => "unknown time".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_just_now() {
        let now = Utc::now();
        let ts = now - Duration::seconds(30);
        assert_eq!(format_relative_time(ts, now), "just now");
    }
    
    #[test]
    fn test_minutes_ago() {
        let now = Utc::now();
        let ts = now - Duration::minutes(15);
        assert_eq!(format_relative_time(ts, now), "15 minutes ago");
    }
    
    #[test]
    fn test_hours_ago() {
        let now = Utc::now();
        let ts = now - Duration::hours(3);
        assert_eq!(format_relative_time(ts, now), "3 hours ago");
    }
    
    #[test]
    fn test_yesterday() {
        let now = Utc::now();
        let ts = now - Duration::hours(30);
        assert!(format_relative_time(ts, now).starts_with("yesterday"));
    }
    
    #[test]
    fn test_last_weekday() {
        let now = Utc::now();
        let ts = now - Duration::days(4);
        assert!(format_relative_time(ts, now).starts_with("last"));
    }
}
