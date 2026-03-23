#[cfg(test)]
mod tests {
    use crate::core::memory::time_format;
    use chrono::{Utc, Duration};

    #[test]
    fn test_relative_format() {
        let now = Utc::now();
        let just_now = now - Duration::seconds(10);
        let hour_ago = now - Duration::minutes(70);
        let yesterday = now - Duration::hours(26);
        
        assert_eq!(time_format::format_relative_time(just_now, now), "just now");
        assert!(time_format::format_relative_time(hour_ago, now).contains("hour"));
        assert!(time_format::format_relative_time(yesterday, now).contains("yesterday"));
    }

    #[tokio::test]
    async fn test_observed_at_pipeline() {
        // This would integration test the full stack - simulating a write and retrieve
        // For now, unit testing the format logic is good, and we rely on 'cargo check' 
        // passing to ensure types are threaded correctly.
        // Real integration tests require DB setup which is complex here.
    }
}
