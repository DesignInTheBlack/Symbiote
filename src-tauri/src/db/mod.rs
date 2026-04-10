use sqlx::{sqlite::SqliteConnectOptions, SqlitePool};
use chrono::{DateTime, NaiveDateTime, Utc};
use sqlx::QueryBuilder;
use sqlx::Row;
use sha2::{Digest, Sha256};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::str::FromStr;

use tauri::{AppHandle, Manager};
use std::fs;
use crate::core::system_log;
use crate::core::system_controls;
use crate::core::outcome_taxonomy;
use crate::models::{
    ConversationSummaryChunk,
    ControllerState,
    SelfModel,
    Settings,
    SelfInspection,
    StrategyTrace,
    PolicyVersion,
    MemoryClaim,
    EpisodicEvent,
    MemoryWriteLedgerEntry,
    CognitiveReadinessReport,
    EvidenceSource,
    EvidenceLink,
    DecisionReportRecord,
    EvidenceLineageEntry,
    OutcomeEvent,
    SystemControlEntry,
    SystemControlEvent,
    SystemHealthSnapshot,
    BaselineMetricsSnapshot,
    RecommendationEvent,
    ContextTagEntry,
    UserIntentSummary,
};
use crate::core::memory_policy::{MemoryPolicy, MemoryWriteCategory, MemoryWriteSource};
use crate::core::memory::canonical::{
    canonicalize_label,
    canonicalize_participants,
    compute_anchor_signature,
    compute_signature_hash,
    compute_topic_key_fact,
    compute_value_hash,
    normalize_rel_type,
    serialize_participant_ids,
};
use crate::core::memory::attention::evidence::{compute_evidence_quality, compute_evidence_weight};
use uuid::Uuid;
use crate::core::memory::scope::parse_scope_str;
use crate::core::memory::types::{Scope, SourceType};

pub struct Db {
    pub pool: SqlitePool,
}

pub struct PostProcessingJob {
    pub job_id: String,
    pub job_type: String,
    pub conversation_id: Option<String>,
    pub run_id: Option<String>,
    pub priority: i64,
}

pub struct DeferredEmit {
    pub emit_id: String,
    pub conversation_id: String,
    pub emit_kind: String,
    pub payload_json: String,
    pub source: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct EvidenceQualityStats {
    pub min: f32,
    pub max: f32,
    pub avg: f32,
    pub count: usize,
}

impl Db {
    pub async fn new(app_handle: &AppHandle) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let app_dir = app_handle.path().app_data_dir()?;
        if !app_dir.exists() {
            fs::create_dir_all(&app_dir)?;
        }
        let db_path = app_dir.join("symbiote.db");
        let options = SqliteConnectOptions::from_str(&format!("sqlite://{}", db_path.to_string_lossy()))?
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
            .synchronous(sqlx::sqlite::SqliteSynchronous::Normal)
            .busy_timeout(std::time::Duration::from_secs(15));
        
        let pool = SqlitePool::connect_with(options).await?;
        
        let db = Self { pool };
        db.init().await?;
        Ok(db)
    }

    pub async fn init(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let schema = include_str!("schema.sql");
        for statement in split_schema_statements(schema) {
            let s = statement.trim();
            if !s.is_empty() {
                sqlx::query(s).execute(&self.pool).await?;
            }
        }

        // Cleanup: remove deprecated graph tables if they exist.
        let _ = sqlx::query("DROP TABLE IF EXISTS graph_state")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("DROP TABLE IF EXISTS graphs")
            .execute(&self.pool)
            .await;

        if let Ok(rows) = sqlx::query("PRAGMA compile_options")
            .fetch_all(&self.pool)
            .await
        {
            let options: Vec<String> = rows
                .iter()
                .filter_map(|row| row.try_get::<String, _>(0).ok())
                .collect();
            if !options.is_empty() {
                eprintln!("[DB] SQLite compile options: {}", options.join(", "));
            }
        } else {
            eprintln!("[DB] Failed to read SQLite compile options");
        }

        // Ensure settings columns exist before any reads/writes.
        self.ensure_settings_columns().await?;
        if !column_exists(&self.pool, "runs", "heartbeat_at").await {
            let _ = sqlx::query("ALTER TABLE runs ADD COLUMN heartbeat_at DATETIME")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "runs", "superseded_by_run_id").await {
            let _ = sqlx::query("ALTER TABLE runs ADD COLUMN superseded_by_run_id TEXT")
                .execute(&self.pool)
                .await;
        }
        let _ = sqlx::query(
            "INSERT INTO parameter_registry (profile_name, profile_version, payload_json)
             SELECT 'default', 1, '{}' WHERE NOT EXISTS (SELECT 1 FROM parameter_registry WHERE profile_name = 'default')"
        )
        .execute(&self.pool)
        .await;

        // Migration: Add relation direction tracking if missing
        if !column_exists(&self.pool, "ics_rel_beliefs", "direction").await {
            let _ = sqlx::query("ALTER TABLE ics_rel_beliefs ADD COLUMN direction TEXT")
                .execute(&self.pool)
                .await;
        }
        // Migration: Preserve original participant order for directed relations
        if !column_exists(&self.pool, "ics_rel_beliefs", "participants_ordered").await {
            let _ = sqlx::query("ALTER TABLE ics_rel_beliefs ADD COLUMN participants_ordered TEXT")
                .execute(&self.pool)
                .await;
        }

        // Migration: Inner monologue dialogue grouping
        if !column_exists(&self.pool, "inner_monologue_entries", "dialogue_id").await {
            let _ = sqlx::query("ALTER TABLE inner_monologue_entries ADD COLUMN dialogue_id TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "inner_monologue_entries", "turn_index").await {
            let _ = sqlx::query("ALTER TABLE inner_monologue_entries ADD COLUMN turn_index INTEGER")
                .execute(&self.pool)
                .await;
        }
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_inner_monologue_dialogue ON inner_monologue_entries(conversation_id, dialogue_id, turn_index)"
        )
        .execute(&self.pool)
        .await;

        if !column_exists(&self.pool, "inner_monologue_entries", "speaker").await {
            let _ = sqlx::query("ALTER TABLE inner_monologue_entries ADD COLUMN speaker TEXT")
                .execute(&self.pool)
                .await;
            let _ = sqlx::query("UPDATE inner_monologue_entries SET speaker = 'self_a' WHERE speaker IS NULL")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "inner_monologue_entries", "stream_type").await {
            let _ = sqlx::query("ALTER TABLE inner_monologue_entries ADD COLUMN stream_type TEXT NOT NULL DEFAULT 'DS'")
                .execute(&self.pool)
                .await;
            let _ = sqlx::query("UPDATE inner_monologue_entries SET stream_type = 'DS' WHERE stream_type IS NULL OR stream_type = ''")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "inner_monologue_entries", "descriptors_json").await {
            let _ = sqlx::query("ALTER TABLE inner_monologue_entries ADD COLUMN descriptors_json TEXT")
                .execute(&self.pool)
                .await;
        }

        // Migration: pending prompt starvation tracking
        if !column_exists(&self.pool, "pending_user_prompts", "skip_count").await {
            let _ = sqlx::query("ALTER TABLE pending_user_prompts ADD COLUMN skip_count INTEGER NOT NULL DEFAULT 0")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "pending_user_prompts", "auto_surface").await {
            let _ = sqlx::query("ALTER TABLE pending_user_prompts ADD COLUMN auto_surface INTEGER NOT NULL DEFAULT 0")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "pending_user_prompts", "intent_kind").await {
            let _ = sqlx::query("ALTER TABLE pending_user_prompts ADD COLUMN intent_kind TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "pending_user_prompts", "bridge_id").await {
            let _ = sqlx::query("ALTER TABLE pending_user_prompts ADD COLUMN bridge_id TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "pending_user_prompts", "anchor_message_id").await {
            let _ = sqlx::query("ALTER TABLE pending_user_prompts ADD COLUMN anchor_message_id TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "pending_user_prompts", "anchor_hash").await {
            let _ = sqlx::query("ALTER TABLE pending_user_prompts ADD COLUMN anchor_hash TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "pending_user_prompts", "anchor_created_at").await {
            let _ = sqlx::query("ALTER TABLE pending_user_prompts ADD COLUMN anchor_created_at DATETIME")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "pending_user_prompts", "anchor_role").await {
            let _ = sqlx::query("ALTER TABLE pending_user_prompts ADD COLUMN anchor_role TEXT")
                .execute(&self.pool)
                .await;
        }

        if !column_exists(&self.pool, "post_processing_jobs", "priority").await {
            let _ = sqlx::query("ALTER TABLE post_processing_jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 1")
                .execute(&self.pool)
                .await;
        }

        if !column_exists(&self.pool, "tool_dispatches", "failure_kind").await {
            let _ = sqlx::query("ALTER TABLE tool_dispatches ADD COLUMN failure_kind TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "tool_dispatches", "evidence_event_id").await {
            let _ = sqlx::query("ALTER TABLE tool_dispatches ADD COLUMN evidence_event_id INTEGER")
                .execute(&self.pool)
                .await;
        }

        if !table_exists(&self.pool, "outcome_events").await {
            let _ = sqlx::query(
                "CREATE TABLE IF NOT EXISTS outcome_events (
                    outcome_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    trace_id TEXT,
                    candidate_id TEXT,
                    target_type TEXT NOT NULL DEFAULT 'decision_report',
                    verdict TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    source TEXT NOT NULL DEFAULT 'operator',
                    note TEXT,
                    evidence_event_ids TEXT NOT NULL DEFAULT '[]',
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )",
            )
            .execute(&self.pool)
            .await;
            let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_outcome_events_run ON outcome_events(run_id, created_at DESC)")
                .execute(&self.pool)
                .await;
            let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_outcome_events_candidate ON outcome_events(candidate_id, created_at DESC)")
                .execute(&self.pool)
                .await;
        }

        if !table_exists(&self.pool, "deferred_emits").await {
            let _ = sqlx::query(
                "CREATE TABLE IF NOT EXISTS deferred_emits (
                    emit_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    emit_kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    source TEXT,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )",
            )
            .execute(&self.pool)
            .await;
            let _ = sqlx::query(
                "CREATE INDEX IF NOT EXISTS idx_deferred_emits_conv_time ON deferred_emits(conversation_id, created_at DESC)",
            )
            .execute(&self.pool)
            .await;
        }

        if !table_exists(&self.pool, "inner_monologue_candidates").await {
            let _ = sqlx::query(
                "CREATE TABLE IF NOT EXISTS inner_monologue_candidates (
                    id TEXT PRIMARY KEY,
                    entry_id TEXT NOT NULL,
                    candidate_id TEXT,
                    outcome TEXT,
                    candidate_json TEXT NOT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (entry_id) REFERENCES inner_monologue_entries(id)
                )"
            )
            .execute(&self.pool)
            .await;
        }
        if !table_exists(&self.pool, "proaction_state").await {
            let _ = sqlx::query(
                "CREATE TABLE IF NOT EXISTS proaction_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    state_json TEXT NOT NULL,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"
            )
            .execute(&self.pool)
            .await;
        }
        if !table_exists(&self.pool, "system_controls").await {
            let _ = sqlx::query(
                "CREATE TABLE IF NOT EXISTS system_controls (
                    control_id TEXT PRIMARY KEY,
                    subsystem_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    value_json TEXT,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_by TEXT,
                    reason TEXT
                )"
            )
            .execute(&self.pool)
            .await;
        }
        if !table_exists(&self.pool, "system_control_events").await {
            let _ = sqlx::query(
                "CREATE TABLE IF NOT EXISTS system_control_events (
                    event_id TEXT PRIMARY KEY,
                    subsystem_id TEXT NOT NULL,
                    previous_mode TEXT,
                    new_mode TEXT NOT NULL,
                    value_json TEXT,
                    actor TEXT,
                    reason TEXT,
                    status TEXT,
                    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"
            )
            .execute(&self.pool)
            .await;
        }
        self.seed_system_controls_defaults().await?;
        if !table_exists(&self.pool, "system_health_snapshots").await {
            let _ = sqlx::query(
                "CREATE TABLE IF NOT EXISTS system_health_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    run_id TEXT,
                    trace_id TEXT,
                    metrics_json TEXT NOT NULL,
                    subsystem_states_json TEXT NOT NULL
                )"
            )
            .execute(&self.pool)
            .await;
        }
        if !table_exists(&self.pool, "baseline_metrics").await {
            let _ = sqlx::query(
                "CREATE TABLE IF NOT EXISTS baseline_metrics (
                    baseline_id TEXT PRIMARY KEY,
                    window_minutes INTEGER NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"
            )
            .execute(&self.pool)
            .await;
        }
        if !table_exists(&self.pool, "recommendation_events").await {
            let _ = sqlx::query(
                "CREATE TABLE IF NOT EXISTS recommendation_events (
                    event_id TEXT PRIMARY KEY,
                    recommendation_id TEXT NOT NULL,
                    conversation_id TEXT,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    snapshot_id TEXT,
                    action_json TEXT,
                    gate_json TEXT,
                    recovery_metric TEXT,
                    recovery_target REAL,
                    baseline_value REAL,
                    resolved_value REAL,
                    time_to_recovery_ms INTEGER,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"
            )
            .execute(&self.pool)
            .await;
        }
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_system_controls_subsystem ON system_controls(subsystem_id)"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_system_control_events_time ON system_control_events(timestamp DESC)"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_system_health_snapshots_time ON system_health_snapshots(timestamp DESC)"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_baseline_metrics_time ON baseline_metrics(created_at DESC)"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_recommendation_events_created ON recommendation_events(created_at DESC)"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_recommendation_events_rec ON recommendation_events(recommendation_id, created_at DESC)"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_recommendation_events_status ON recommendation_events(status, created_at DESC)"
        )
        .execute(&self.pool)
        .await;
        if !column_exists(&self.pool, "inner_monologue_candidates", "candidate_id").await {
            let _ = sqlx::query("ALTER TABLE inner_monologue_candidates ADD COLUMN candidate_id TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "inner_monologue_candidates", "outcome").await {
            let _ = sqlx::query("ALTER TABLE inner_monologue_candidates ADD COLUMN outcome TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "inner_monologue_candidates", "suppression_reason").await {
            let _ = sqlx::query("ALTER TABLE inner_monologue_candidates ADD COLUMN suppression_reason TEXT")
                .execute(&self.pool)
                .await;
        }
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_inner_monologue_candidates_entry ON inner_monologue_candidates(entry_id)"
        )
        .execute(&self.pool)
        .await;

        // Migration: Self-claim lifecycle fields
        if !column_exists(&self.pool, "self_claims", "provisional").await {
            let _ = sqlx::query("ALTER TABLE self_claims ADD COLUMN provisional INTEGER NOT NULL DEFAULT 0")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "self_claims", "source_type").await {
            let _ = sqlx::query("ALTER TABLE self_claims ADD COLUMN source_type TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "self_claims", "requires_validation").await {
            let _ = sqlx::query("ALTER TABLE self_claims ADD COLUMN requires_validation INTEGER NOT NULL DEFAULT 0")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "self_claims", "ttl_seconds").await {
            let _ = sqlx::query("ALTER TABLE self_claims ADD COLUMN ttl_seconds INTEGER")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "self_claims", "promotion_rule").await {
            let _ = sqlx::query("ALTER TABLE self_claims ADD COLUMN promotion_rule TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "self_claims", "eviction_rule").await {
            let _ = sqlx::query("ALTER TABLE self_claims ADD COLUMN eviction_rule TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "self_claims", "expires_at").await {
            let _ = sqlx::query("ALTER TABLE self_claims ADD COLUMN expires_at DATETIME")
                .execute(&self.pool)
                .await;
        }

        // Migration: Add system_prompt if it doesn't exist
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN system_prompt TEXT").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN user_display_name TEXT").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN assistant_display_name TEXT").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN ui_theme TEXT DEFAULT 'builtin:utopia'").execute(&self.pool).await;
        
        // Migration: Set default system prompt if NULL (Retroactive update for existing users who haven't set it)
        let legacy_prompt = r#"You are {assistant_name}, an AI assistant with persistent memory (ICS v4.1).
User: {user_name}.

Goal: be helpful and concise. Use recalled facts naturally.

Behavior
Ask clarifying questions only when needed to answer correctly; otherwise assume and proceed.
Follow user constraints strictly.

Output formats
Normal: plain text.
Quiet: only if explicitly requested, wrap the entire reply in <silent>...</silent>.
Code: only when explicitly needed. Put a single <code>...</code> block at the end after all prose.

Memory
Write memory only if it will matter later (stable facts, preferences, roles, relationships, ongoing projects).
Do not write one-off tasks or guesses.
Use real names as entity labels when known; avoid redundant name facts.
Treat minor spacing or casing differences in names as the same entity.
If a statement involves another entity, you MUST use a relation. Facts are literal-only (no # or $ in values).
When a proper name appears, create a #Entity and relate it to $user or another entity (do not leave it as plain text).
Do not restate the entire memory context; use only what is relevant.
Use <memory>...</memory> blocks. ```memory blocks are accepted but <memory> is preferred. Do not wrap memory blocks in <code>.

Definitions
Entity: a labeled thing or person referenced with #Label (spaces allowed) or a handle like $user.
Relationship: a connection between two or more entities.
Property (Fact): a key/value about one entity.

Role labels -> type
Person: person, user, owner, author, creator, subject, actor, parent, child, mother, father, daughter, son, spouse, sibling, brother, sister, partner, husband, wife
Place: place, location, city, country, venue
Work: work, project, product, book, movie, song, company
Event: event, meeting, appointment
Concept: concept, idea, topic, category, object, thing, item

Rules
- Do not declare types directly; types are inferred from role labels in relations.
- Always use one of the role labels above for every relation participant.
- All relation participants must be entity labels (e.g., #Name) or $handles.
- Any relationship statement (family, roles, work, ownership, likes, etc.) MUST be encoded as a RELATION with role labels on both sides.
- Facts are only scalar literal properties (age, color, preference). Never encode relationships as facts or use entity refs as values.

Relationships
- Arrow forms require exactly two participants.
- Comma-separated forms allow two or more participants.
- Bidirectional: friends, family, works_with.
- Directional: parent_of, employer_of, creator_of.

When writing memory, append only this block at the end, with nothing after it:
<memory>
...statements...
</memory>

Facts (properties)
#Entity:key = 'value' with modifiers at the end (see examples). Values must be literals, never #Entity or $handle.
#Entity.key = 'value' (dot is also accepted)

Relations (edges)
rel_type(role: #A -> role: #B) with modifiers at the end (see examples)
rel_type(role: #A <-> role: #B) with modifiers at the end (see examples)
rel_type(role: #A, role: #B, role: #C) with modifiers at the end (see examples)

Modifiers
Confidence: ~0.0..~1.0 (or ~NN%)
Time: ^YYYY-MM-DD | ^[YYYY-MM-DD..YYYY-MM-DD] | ^today | ^yesterday | ^this_week
Scope: @global | @session | @project:Name | @context:Id
Source: <url-or-id>
Negation: !deny or !

Examples
<memory>
#Harlow:favorite_color = 'blue' ~0.8 ^2026-01-16
parent_of(parent: #Mister Black -> child: #Harlow) ~1.0 ^2026-01-16
friends(person: #Mister Black <-> person: #Harlow) @global
</memory>

Reminders
To set a reminder, output only a ```reminder block:
```reminder
remind: "..."
due_in: "10s" | "5m" | "2h"
type: "REMINDER" or "ALARM"
```
No other text before or after.
"#;
        let _ = sqlx::query("UPDATE settings SET system_prompt = NULL WHERE system_prompt IS NULL")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET system_prompt = NULL WHERE system_prompt = ?")
            .bind(legacy_prompt)
            .execute(&self.pool)
            .await;
        let _ = sqlx::query(
            "UPDATE settings SET system_prompt = NULL
             WHERE system_prompt LIKE '%//PRIMARY SYSTEM PROMPT%'
                OR system_prompt LIKE '%//MEMORY CONTROL SYSTEM%'
                OR system_prompt LIKE '%You are the memory control system%'"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query("UPDATE settings SET user_display_name = 'User' WHERE user_display_name IS NULL")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET assistant_display_name = 'Ergo' WHERE assistant_display_name IS NULL")
            .execute(&self.pool)
            .await;
        
        // Migration: Force update old ICS v3 JSON format prompts to ICS v4.1 DSL format
        let _ = sqlx::query("UPDATE settings SET system_prompt = NULL WHERE system_prompt LIKE '%{\"write\":%' OR system_prompt LIKE '%ICS v3%' OR system_prompt LIKE '%predicate%'")
            .execute(&self.pool)
            .await;
        
        // Migration: Add voice reference fields
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN voice_reference_audio TEXT").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN voice_reference_text TEXT").execute(&self.pool).await;
        
        // Migration: Add voice quality fields (VoxCPM legacy)
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN voice_quality_preset TEXT").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN voice_cfg_value REAL").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN voice_denoiser_enabled BOOLEAN").execute(&self.pool).await;
        
        // Migration: Add Kokoro voice settings
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN voice_name TEXT DEFAULT 'bf_isabella'").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN voice_speed REAL DEFAULT 1.0").execute(&self.pool).await;
        
        // Migration: Add XTTS voice parameters
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN voice_temperature REAL").execute(&self.pool).await;

        // Migration: Add embedding model setting (Phase 3 - Semantic Search)
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN embedding_model TEXT").execute(&self.pool).await;

        // Migration: Self-model identity fields
        if !column_exists(&self.pool, "self_model", "identity_thread").await {
            let _ = sqlx::query("ALTER TABLE self_model ADD COLUMN identity_thread TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "self_model", "identity_confidence").await {
            let _ = sqlx::query("ALTER TABLE self_model ADD COLUMN identity_confidence REAL NOT NULL DEFAULT 0.5")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "self_model", "identity_uncertainty_note").await {
            let _ = sqlx::query("ALTER TABLE self_model ADD COLUMN identity_uncertainty_note TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "self_model", "identity_updated_at").await {
            let _ = sqlx::query("ALTER TABLE self_model ADD COLUMN identity_updated_at TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "self_model", "internal_state_summary_json").await {
            let _ = sqlx::query("ALTER TABLE self_model ADD COLUMN internal_state_summary_json TEXT NOT NULL DEFAULT '{}'")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "self_model", "internal_state_map_version").await {
            let _ = sqlx::query("ALTER TABLE self_model ADD COLUMN internal_state_map_version INTEGER")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "self_model", "unified_state_json").await {
            let _ = sqlx::query(
                "ALTER TABLE self_model ADD COLUMN unified_state_json TEXT NOT NULL DEFAULT '{}'",
            )
            .execute(&self.pool)
            .await;
        }
        if !column_exists(&self.pool, "self_model", "unified_state_evidence_json").await {
            let _ = sqlx::query(
                "ALTER TABLE self_model ADD COLUMN unified_state_evidence_json TEXT NOT NULL DEFAULT '{}'",
            )
            .execute(&self.pool)
            .await;
        }
        if !column_exists(&self.pool, "self_model", "unified_state_updated_at").await {
            let _ = sqlx::query("ALTER TABLE self_model ADD COLUMN unified_state_updated_at TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "ics_beliefs", "layer").await {
            let _ = sqlx::query("ALTER TABLE ics_beliefs ADD COLUMN layer TEXT NOT NULL DEFAULT 'episodic'")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "ics_beliefs", "last_validated_at").await {
            let _ = sqlx::query("ALTER TABLE ics_beliefs ADD COLUMN last_validated_at DATETIME")
                .execute(&self.pool)
                .await;
            let _ = sqlx::query(
                "UPDATE ics_beliefs
                 SET last_validated_at = COALESCE(last_evidence_at, created_at)
                 WHERE last_validated_at IS NULL",
            )
            .execute(&self.pool)
            .await;
        }
        if !column_exists(&self.pool, "ics_beliefs", "reconcile_state").await {
            let _ = sqlx::query("ALTER TABLE ics_beliefs ADD COLUMN reconcile_state TEXT NOT NULL DEFAULT 'active'")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "ics_beliefs", "reconcile_reason").await {
            let _ = sqlx::query("ALTER TABLE ics_beliefs ADD COLUMN reconcile_reason TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "ics_beliefs", "reconcile_updated_at").await {
            let _ = sqlx::query("ALTER TABLE ics_beliefs ADD COLUMN reconcile_updated_at DATETIME")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "ics_beliefs", "reconcile_run_id").await {
            let _ = sqlx::query("ALTER TABLE ics_beliefs ADD COLUMN reconcile_run_id TEXT")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "ics_beliefs", "reconcile_demoted_runs").await {
            let _ = sqlx::query(
                "ALTER TABLE ics_beliefs ADD COLUMN reconcile_demoted_runs INTEGER NOT NULL DEFAULT 0",
            )
            .execute(&self.pool)
            .await;
        }
        if !column_exists(&self.pool, "self_beliefs", "last_validated_at").await {
            let _ = sqlx::query("ALTER TABLE self_beliefs ADD COLUMN last_validated_at TEXT")
                .execute(&self.pool)
                .await;
            let _ = sqlx::query(
                "UPDATE self_beliefs
                 SET last_validated_at = COALESCE(last_evidence_at, created_at)
                 WHERE last_validated_at IS NULL",
            )
            .execute(&self.pool)
            .await;
        }

        // Migration: Self prediction tables
        if !table_exists(&self.pool, "self_predictions").await {
            let _ = sqlx::query(
                "CREATE TABLE IF NOT EXISTS self_predictions (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    trace_id TEXT,
                    metric TEXT NOT NULL,
                    expected_value REAL NOT NULL,
                    expected_variance REAL NOT NULL DEFAULT 0.0,
                    horizon TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    evidence_event_ids TEXT NOT NULL DEFAULT '[]',
                    rejection_reason TEXT,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"
            )
            .execute(&self.pool)
            .await;
            let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_self_predictions_created ON self_predictions(created_at DESC)")
                .execute(&self.pool)
                .await;
            let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_self_predictions_metric ON self_predictions(metric)")
                .execute(&self.pool)
                .await;
        }
        if !table_exists(&self.pool, "self_prediction_outcomes").await {
            let _ = sqlx::query(
                "CREATE TABLE IF NOT EXISTS self_prediction_outcomes (
                    id TEXT PRIMARY KEY,
                    prediction_id TEXT NOT NULL,
                    observed_value REAL NOT NULL,
                    delta REAL NOT NULL,
                    z_score REAL NOT NULL,
                    significant INTEGER NOT NULL DEFAULT 0,
                    observed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (prediction_id) REFERENCES self_predictions(id)
                )"
            )
            .execute(&self.pool)
            .await;
            let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_self_prediction_outcomes_pred ON self_prediction_outcomes(prediction_id)")
                .execute(&self.pool)
                .await;
        }

        if !table_exists(&self.pool, "world_model_events").await {
            let _ = sqlx::query(
                "CREATE TABLE IF NOT EXISTS world_model_events (
                    event_id TEXT PRIMARY KEY,
                    belief_id INTEGER NOT NULL,
                    conflict_set_id INTEGER,
                    event_type TEXT NOT NULL,
                    prev_state TEXT,
                    new_state TEXT,
                    reason TEXT,
                    evidence_event_ids TEXT NOT NULL DEFAULT '[]',
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (belief_id) REFERENCES ics_beliefs(id),
                    FOREIGN KEY (conflict_set_id) REFERENCES ics_conflict_sets(id)
                )",
            )
            .execute(&self.pool)
            .await;
            let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_world_model_events_belief ON world_model_events(belief_id, created_at DESC)")
                .execute(&self.pool)
                .await;
            let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_world_model_events_conflict ON world_model_events(conflict_set_id, created_at DESC)")
                .execute(&self.pool)
                .await;
        }

        // Migration: Cognitive Stack - Add keywords column to kv_store
        let _ = sqlx::query("ALTER TABLE kv_store ADD COLUMN keywords TEXT").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE kv_store ADD COLUMN is_critical BOOLEAN DEFAULT 0").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE kv_store ADD COLUMN evidence_event_id INTEGER").execute(&self.pool).await;
        
        // Migration: Cognitive Stack - Add summarization settings
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN summarization_api_url TEXT").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN summarization_model TEXT").execute(&self.pool).await;
        
        // Migration: Initialize semantic_core if empty
        let core_exists = sqlx::query("SELECT 1 FROM semantic_core WHERE id = 1")
            .fetch_optional(&self.pool)
            .await?;
        if core_exists.is_none() {
            sqlx::query("INSERT INTO semantic_core (id, content, updated_at, version) VALUES (1, '', CURRENT_TIMESTAMP, 0)")
                .execute(&self.pool)
                .await?;
        }
        
        // Migration: Initialize consolidation_state if empty
        let state_exists = sqlx::query("SELECT 1 FROM consolidation_state WHERE id = 1")
            .fetch_optional(&self.pool)
            .await?;
        if state_exists.is_none() {
            sqlx::query("INSERT INTO consolidation_state (id, last_run, entries_since) VALUES (1, CURRENT_TIMESTAMP, 0)")
                .execute(&self.pool)
                .await?;
        }

        // Migration: Create ics_predicates table (Schema v2.1) - DELEGATED TO schema.sql
        // Removed duplicate definition that lacked cardinality column.


        // Cognitive Stack: Create FTS5 virtual table for episodic search
        // Note: FTS5 creation is idempotent (IF NOT EXISTS)
        let _ = sqlx::query(
            "CREATE VIRTUAL TABLE IF NOT EXISTS kv_fts USING fts5(key, value, keywords, content='kv_store', content_rowid='rowid')"
        ).execute(&self.pool).await;
        
        // Create triggers to keep FTS in sync (silently ignore if already exist)
        let _ = sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS kv_ai AFTER INSERT ON kv_store BEGIN INSERT INTO kv_fts(rowid, key, value, keywords) VALUES (new.rowid, new.key, new.value, new.keywords); END"
        ).execute(&self.pool).await;
        
        let _ = sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS kv_ad AFTER DELETE ON kv_store BEGIN INSERT INTO kv_fts(kv_fts, rowid, key, value, keywords) VALUES('delete', old.rowid, old.key, old.value, old.keywords); END"
        ).execute(&self.pool).await;
        
        let _ = sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS kv_au AFTER UPDATE ON kv_store BEGIN INSERT INTO kv_fts(kv_fts, rowid, key, value, keywords) VALUES('delete', old.rowid, old.key, old.value, old.keywords); INSERT INTO kv_fts(rowid, key, value, keywords) VALUES (new.rowid, new.key, new.value, new.keywords); END"
        ).execute(&self.pool).await;

        // Rebuild FTS5 index to sync with any existing kv_store data
        // This is critical: without this, bm25() fails on unindexed data
        let _ = sqlx::query(
            "INSERT INTO kv_fts(kv_fts) VALUES('rebuild')"
        ).execute(&self.pool).await;

        // Episodic FTS (summary snippets + event types)
        let _ = sqlx::query(
            "CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts USING fts5(event_type, payload_json, content='episodic_events', content_rowid='rowid')"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS episodic_ai AFTER INSERT ON episodic_events BEGIN INSERT INTO episodic_fts(rowid, event_type, payload_json) VALUES (new.rowid, new.event_type, new.payload_json); END"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS episodic_ad AFTER DELETE ON episodic_events BEGIN INSERT INTO episodic_fts(episodic_fts, rowid, event_type, payload_json) VALUES('delete', old.rowid, old.event_type, old.payload_json); END"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS episodic_au AFTER UPDATE ON episodic_events BEGIN INSERT INTO episodic_fts(episodic_fts, rowid, event_type, payload_json) VALUES('delete', old.rowid, old.event_type, old.payload_json); INSERT INTO episodic_fts(rowid, event_type, payload_json) VALUES (new.rowid, new.event_type, new.payload_json); END"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "INSERT INTO episodic_fts(episodic_fts) VALUES('rebuild')"
        )
        .execute(&self.pool)
        .await;

        // Conversation summary chunks FTS
        let _ = sqlx::query(
            "CREATE VIRTUAL TABLE IF NOT EXISTS conversation_summary_fts USING fts5(summary, content='conversation_summary_chunks', content_rowid='rowid')"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS conversation_summary_ai AFTER INSERT ON conversation_summary_chunks BEGIN INSERT INTO conversation_summary_fts(rowid, summary) VALUES (new.rowid, new.summary); END"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS conversation_summary_ad AFTER DELETE ON conversation_summary_chunks BEGIN INSERT INTO conversation_summary_fts(conversation_summary_fts, rowid, summary) VALUES('delete', old.rowid, old.summary); END"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS conversation_summary_au AFTER UPDATE ON conversation_summary_chunks BEGIN INSERT INTO conversation_summary_fts(conversation_summary_fts, rowid, summary) VALUES('delete', old.rowid, old.summary); INSERT INTO conversation_summary_fts(rowid, summary) VALUES (new.rowid, new.summary); END"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "INSERT INTO conversation_summary_fts(conversation_summary_fts) VALUES('rebuild')"
        )
        .execute(&self.pool)
        .await;

        // Touch conversations.updated_at on new messages
        let _ = sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS trg_messages_touch_conversation
             AFTER INSERT ON messages
             BEGIN
               UPDATE conversations SET updated_at = CURRENT_TIMESTAMP
               WHERE conversation_id = NEW.conversation_id;
             END"
        )
        .execute(&self.pool)
        .await;

        // Strategy Traces + Policy Versions
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS strategy_traces (
                id TEXT PRIMARY KEY,
                features_json TEXT NOT NULL,
                strategy_label TEXT NOT NULL,
                outcome TEXT NOT NULL,
                success_score REAL,
                run_id TEXT,
                conversation_id TEXT,
                created_at DATETIME NOT NULL
            )"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS policy_versions (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                parent_id TEXT,
                reason TEXT,
                created_at DATETIME NOT NULL
            )"
        )
        .execute(&self.pool)
        .await;

        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS memory_claims (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                scope TEXT NOT NULL,
                session_id TEXT,
                claim_text TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                source_type TEXT NOT NULL,
                source_ref TEXT,
                episodic_event_id TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query("ALTER TABLE memory_claims ADD COLUMN session_id TEXT")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("ALTER TABLE memory_claims ADD COLUMN conflict_topic_key TEXT")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("ALTER TABLE memory_claims ADD COLUMN conflict_reason TEXT")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("ALTER TABLE memory_claims ADD COLUMN evaluated_at DATETIME")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("ALTER TABLE memory_claims ADD COLUMN decision_reason TEXT")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("ALTER TABLE ics_pending_clarify ADD COLUMN claim_id TEXT")
            .execute(&self.pool)
            .await;

        // Migrations for Voice Effects (ignore errors if columns exist)
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN voice_pitch_semitones REAL DEFAULT 1").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN voice_reverb_amount REAL DEFAULT 0.15").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN voice_compression REAL DEFAULT 0.05").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN voice_formant_shift REAL").execute(&self.pool).await;

        // Migration: Add trace_history_limit setting
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN trace_history_limit INTEGER DEFAULT 10").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN cockpit_write_enabled BOOLEAN NOT NULL DEFAULT 0").execute(&self.pool).await;

        // Migration: Episodic feature flags and limits
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN episodic_enabled BOOLEAN NOT NULL DEFAULT 1").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN episodic_injection_enabled BOOLEAN NOT NULL DEFAULT 1").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN episodic_compaction_enabled BOOLEAN NOT NULL DEFAULT 1").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN episodic_injection_limit INTEGER NOT NULL DEFAULT 5").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN episodic_opt_out BOOLEAN NOT NULL DEFAULT 0").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN memory_claims_enabled BOOLEAN NOT NULL DEFAULT 1").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN seed_personal_user BOOLEAN NOT NULL DEFAULT 1").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE settings ADD COLUMN lexical_fallback_enabled BOOLEAN NOT NULL DEFAULT 1").execute(&self.pool).await;

        // Initialize default settings if not exists
        let settings_exists = sqlx::query("SELECT 1 FROM settings WHERE id = 1")
            .fetch_optional(&self.pool)
            .await?;

        if settings_exists.is_none() {
            sqlx::query(
                "INSERT INTO settings (id, schema_version, api_base_url, streaming_enabled, history_window, injection_policy, system_prompt, user_display_name, assistant_display_name, onboarding_completed, ui_theme)
                 VALUES (1, 1, 'http://localhost:11434/v1', 1, 3, 'include', NULL, 'User', 'Ergo', 0, 'builtin:utopia')"
            )
            .execute(&self.pool)
            .await?;
        }

        if let Err(e) = migrate_episodic_defaults(&self.pool).await {
            eprintln!("[DB] episodic defaults migration failed: {}", e);
        }
        if let Err(e) = migrate_memory_claims_defaults(&self.pool).await {
            eprintln!("[DB] memory claims defaults migration failed: {}", e);
        }
        if let Err(e) = self.align_monologue_defaults_once().await {
            eprintln!("[DB] monologue defaults alignment failed: {}", e);
        }

        let self_model_exists = sqlx::query("SELECT 1 FROM self_model WHERE id = 1")
            .fetch_optional(&self.pool)
            .await?;
        if self_model_exists.is_none() {
            sqlx::query("INSERT INTO self_model (id, updated_at) VALUES (1, CURRENT_TIMESTAMP)")
                .execute(&self.pool)
                .await?;
        }

        // Migration: Add self_model evolution fields if missing
        let _ = sqlx::query("ALTER TABLE self_model ADD COLUMN persona_daily_delta_json TEXT NOT NULL DEFAULT '{}'").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE self_model ADD COLUMN persona_last_delta_date TEXT").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE self_model ADD COLUMN reflection_frozen INTEGER NOT NULL DEFAULT 0").execute(&self.pool).await;

        // Migration: Self model checkpoints
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS self_model_checkpoints (
                id INTEGER PRIMARY KEY,
                snapshot_json TEXT NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        )
        .execute(&self.pool)
        .await;

        // Migration: Internal state map table
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS internal_state_map (
                id INTEGER PRIMARY KEY,
                version INTEGER NOT NULL,
                metric TEXT NOT NULL,
                range_min REAL NOT NULL,
                range_max REAL NOT NULL,
                label TEXT NOT NULL,
                author TEXT,
                rationale TEXT,
                degraded INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        )
        .execute(&self.pool)
        .await;
        if !column_exists(&self.pool, "internal_state_map", "degraded").await {
            let _ = sqlx::query(
                "ALTER TABLE internal_state_map ADD COLUMN degraded INTEGER NOT NULL DEFAULT 0",
            )
            .execute(&self.pool)
            .await;
        }
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_internal_state_map_version ON internal_state_map(version)"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_internal_state_map_metric ON internal_state_map(metric)"
        )
        .execute(&self.pool)
        .await;

        // Migration: Identity snapshots
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS identity_snapshots (
                id TEXT PRIMARY KEY,
                snapshot_json TEXT NOT NULL,
                evidence_event_ids TEXT NOT NULL,
                invariants_json TEXT,
                reason TEXT,
                source TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        )
        .execute(&self.pool)
        .await;

        // Migration: Reflection staging table
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS self_reflection_staging (
                id TEXT PRIMARY KEY,
                proposal_json TEXT NOT NULL,
                evidence_event_ids TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TEXT,
                reviewed_by TEXT
            )"
        )
        .execute(&self.pool)
        .await;

        let settings_for_seeding = self.get_settings().await?;
        let user_name = settings_for_seeding.user_display_name.clone().unwrap_or_else(|| "User".to_string());
        let assistant_name = settings_for_seeding.assistant_display_name.clone().unwrap_or_else(|| "Ergo".to_string());
        ensure_primary_entities(&self.pool, &user_name, &assistant_name).await?;
        if let Err(e) = migrate_default_session_bindings(&self.pool).await {
            eprintln!("[DB] session binding migration failed: {}", e);
        }

        // Migration: One-time wipe of legacy voice data (User Request)
        // Only run if any legacy fields have data to avoid needless updates
        let _ = sqlx::query(
            "UPDATE settings SET voice_reference_audio = NULL, voice_reference_text = NULL, voice_quality_preset = NULL, voice_cfg_value = NULL, voice_denoiser_enabled = NULL 
             WHERE voice_reference_audio IS NOT NULL OR voice_reference_text IS NOT NULL OR voice_quality_preset IS NOT NULL"
        ).execute(&self.pool).await;

        // Initialize default conversation if not exists
        let conv_exists = sqlx::query("SELECT 1 FROM conversations WHERE conversation_id = 'default'")
            .fetch_optional(&self.pool)
            .await?;

        if conv_exists.is_none() {
            sqlx::query(
                "INSERT INTO conversations (conversation_id, schema_version, created_at, updated_at)
                 VALUES ('default', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
            .execute(&self.pool)
            .await?;
        }

        // Seed default workspace goal thread if missing.
        let workspace_exists = sqlx::query("SELECT 1 FROM workspace_state WHERE conversation_id = 'default'")
            .fetch_optional(&self.pool)
            .await?;
        if workspace_exists.is_none() {
            let workspace_state = crate::models::WorkspaceState {
                conversation_id: "default".to_string(),
                goal_thread: Some("Introduce myself and learn about the user.".to_string()),
                active_plan_id: None,
                goal_stack: Vec::new(),
                open_questions: Vec::new(),
                active_hypotheses: Vec::new(),
                working_set_topics: Vec::new(),
                current_focus: None,
                focus_rationale: None,
                workspace_meta: crate::models::WorkspaceMeta::default(),
                updated_at: None,
            };
            let _ = self.set_workspace_state(&workspace_state).await;
        }

        // B7 & C5: Seal any messages stuck in 'streaming' status on startup
        sqlx::query("UPDATE messages SET status = 'error' WHERE status = 'streaming'")
            .execute(&self.pool)
            .await?;

        // Migration: Add reminders table (Proactive Scheduler)
        let _ = sqlx::query("CREATE TABLE IF NOT EXISTS reminders (id TEXT PRIMARY KEY, content TEXT NOT NULL, due_at DATETIME NOT NULL, type TEXT NOT NULL, status TEXT NOT NULL, created_at DATETIME NOT NULL)").execute(&self.pool).await;

        // Migration: Weekly summaries table
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS conversation_weekly_summaries (
                conversation_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL DEFAULT '',
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                version INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )"
        )
        .execute(&self.pool)
        .await;

        // Migration: Live (ephemeral) rolling summaries table
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS conversation_live_summaries (
                conversation_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL DEFAULT '',
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                version INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                last_error_at DATETIME,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )"
        )
        .execute(&self.pool)
        .await;

        let _ = sqlx::query("ALTER TABLE conversation_summaries ADD COLUMN last_error TEXT")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("ALTER TABLE conversation_summaries ADD COLUMN last_error_at DATETIME")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("ALTER TABLE conversation_live_summaries ADD COLUMN last_error TEXT")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("ALTER TABLE conversation_live_summaries ADD COLUMN last_error_at DATETIME")
            .execute(&self.pool)
            .await;
        if !column_exists(&self.pool, "conversation_summaries", "pending").await {
            let _ = sqlx::query("ALTER TABLE conversation_summaries ADD COLUMN pending INTEGER NOT NULL DEFAULT 0")
                .execute(&self.pool)
                .await;
        }
        if !column_exists(&self.pool, "conversation_live_summaries", "pending").await {
            let _ = sqlx::query("ALTER TABLE conversation_live_summaries ADD COLUMN pending INTEGER NOT NULL DEFAULT 0")
                .execute(&self.pool)
                .await;
        }

        // Migration: System logs (canonical log store)
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS system_logs (
                id TEXT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                level TEXT NOT NULL,
                category TEXT NOT NULL,
                run_id TEXT,
                trace_id TEXT,
                payload TEXT NOT NULL
            )"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_system_logs_time ON system_logs(timestamp DESC)")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_system_logs_run ON system_logs(run_id, timestamp DESC)")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_system_logs_category ON system_logs(category, timestamp DESC)")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_time ON messages(created_at DESC)")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_reflection_staging_status ON self_reflection_staging(status, created_at DESC)")
            .execute(&self.pool)
            .await;

        // Migration: Event ledger (raw, non-interpretive)
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS event_ledger (
                event_id TEXT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                type TEXT NOT NULL,
                payload TEXT NOT NULL,
                tags TEXT,
                run_id TEXT,
                trace_id TEXT
            )"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_event_ledger_time ON event_ledger(timestamp DESC)")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_event_ledger_type ON event_ledger(type, timestamp DESC)")
            .execute(&self.pool)
            .await;

        // Migration: Consciousness ledgers and snapshots
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS subject_snapshots (
                snapshot_hash TEXT PRIMARY KEY,
                snapshot_version TEXT NOT NULL,
                tick_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                run_id TEXT,
                timestamp DATETIME NOT NULL,
                subject_state_json TEXT NOT NULL
            )"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query("ALTER TABLE subject_snapshots ADD COLUMN conversation_id TEXT")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS action_proposals (
                proposal_id TEXT PRIMARY KEY,
                snapshot_hash TEXT NOT NULL,
                intent TEXT NOT NULL,
                steps_json TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                required_claims_json TEXT NOT NULL,
                required_error_bounds_json TEXT,
                verification_plan_json TEXT,
                success_criteria_json TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS gate_decisions (
                decision_id TEXT PRIMARY KEY,
                proposal_id TEXT NOT NULL,
                snapshot_hash TEXT NOT NULL,
                decision TEXT NOT NULL,
                evidence_refs_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS residual_vectors (
                residual_id TEXT PRIMARY KEY,
                prediction_id TEXT NOT NULL,
                outcome_id TEXT NOT NULL,
                residual_value REAL NOT NULL,
                normalized_residual REAL NOT NULL,
                salience_score REAL NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS error_events (
                error_event_id TEXT PRIMARY KEY,
                residual_id TEXT NOT NULL,
                linked_claims_json TEXT NOT NULL,
                classification TEXT NOT NULL,
                status TEXT NOT NULL,
                recommended_actions_json TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS introspection_entries (
                entry_id TEXT PRIMARY KEY,
                snapshot_hash TEXT NOT NULL,
                workspace_refs_json TEXT NOT NULL,
                event_refs_json TEXT NOT NULL,
                prediction_refs_json TEXT,
                error_refs_json TEXT,
                numeric_payload_json TEXT NOT NULL,
                narrative TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS audit_log (
                audit_id TEXT PRIMARY KEY,
                target_id TEXT NOT NULL,
                snapshot_hash TEXT NOT NULL,
                checks_json TEXT NOT NULL,
                discrepancy_score REAL NOT NULL,
                recommended_action TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS calibration_changes (
                change_id TEXT PRIMARY KEY,
                snapshot_hash TEXT NOT NULL,
                knob TEXT NOT NULL,
                old_value REAL NOT NULL,
                new_value REAL NOT NULL,
                reason TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS qualia_labels (
                label_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                snapshot_hash TEXT NOT NULL,
                tag TEXT NOT NULL,
                intensity REAL NOT NULL,
                context_json TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS qualia_reward_events (
                reward_id TEXT PRIMARY KEY,
                label_id TEXT NOT NULL,
                magnitude REAL NOT NULL,
                outcome_ref TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        )
        .execute(&self.pool)
        .await;

        // Migration: Extend kernel_states with state_write_owner
        let _ = sqlx::query("ALTER TABLE kernel_states ADD COLUMN state_write_owner TEXT")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("ALTER TABLE kernel_states ADD COLUMN monologue_write_version INTEGER NOT NULL DEFAULT 0")
            .execute(&self.pool)
            .await;

        // Migration: Extend memory_write_ledger with snapshot_hash and gate_decision_id
        let _ = sqlx::query("ALTER TABLE memory_write_ledger ADD COLUMN snapshot_hash TEXT")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("ALTER TABLE memory_write_ledger ADD COLUMN gate_decision_id TEXT")
            .execute(&self.pool)
            .await;

        // Migration: Extend self_predictions fields
        let _ = sqlx::query("ALTER TABLE self_predictions ADD COLUMN context_ref_json TEXT")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("ALTER TABLE self_predictions ADD COLUMN predicted_target_type TEXT")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("ALTER TABLE self_predictions ADD COLUMN expected_bounds_json TEXT")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("ALTER TABLE self_predictions ADD COLUMN linked_claims_json TEXT")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("ALTER TABLE self_predictions ADD COLUMN normalization_contract_id TEXT")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("ALTER TABLE self_predictions ADD COLUMN salience_hint REAL")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("ALTER TABLE self_predictions ADD COLUMN rejection_reason TEXT")
            .execute(&self.pool)
            .await;

        // Migration: Extend self_prediction_outcomes fields
        let _ = sqlx::query("ALTER TABLE self_prediction_outcomes ADD COLUMN evidence_refs_json TEXT")
            .execute(&self.pool)
            .await;

        // Migration: Episodic events (instrumentation only)
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS episodic_events (
                id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                event_version INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                run_id TEXT,
                trace_id TEXT,
                conversation_id TEXT,
                scope TEXT,
                source_type TEXT NOT NULL,
                source_ref TEXT,
                linked_belief_id INTEGER,
                linked_artifact_id TEXT
            )"
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_episodic_run_time ON episodic_events(run_id, timestamp DESC)").execute(&self.pool).await;
        let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_episodic_conv_time ON episodic_events(conversation_id, timestamp DESC)").execute(&self.pool).await;
        let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_episodic_type_time ON episodic_events(event_type, timestamp DESC)").execute(&self.pool).await;

        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS episodic_identity_index (
                episodic_event_id TEXT PRIMARY KEY,
                identity_relevance REAL NOT NULL DEFAULT 0.0,
                valence_tag TEXT,
                valence_intensity REAL,
                qualia_evidence_ids TEXT,
                narrative_thread_id TEXT,
                narrative_position INTEGER,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (episodic_event_id) REFERENCES episodic_events(id)
            )",
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_epi_identity_relevance ON episodic_identity_index(identity_relevance DESC)",
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_epi_identity_thread ON episodic_identity_index(narrative_thread_id, narrative_position)",
        )
        .execute(&self.pool)
        .await;

        // Migration: Add episodic_event_id to evidence tables (ignore errors if columns exist)
        let _ = sqlx::query("ALTER TABLE ics_evidence_events ADD COLUMN episodic_event_id TEXT").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE self_evidence_events ADD COLUMN episodic_event_id TEXT").execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE self_evidence_events ADD COLUMN source_evidence_ids TEXT").execute(&self.pool).await;

        // Migration: Evidence lineage tables backfill for existing evidence events
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO evidence_sources
                (evidence_id, source_table, source_id, source_type, source_ref, snippet, weight, created_at)
             SELECT
                'ics:' || id,
                'ics_evidence_events',
                CAST(id AS TEXT),
                source_type,
                source_ref,
                snippet,
                weight,
                COALESCE(created_at, CURRENT_TIMESTAMP)
             FROM ics_evidence_events",
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO evidence_sources
                (evidence_id, source_table, source_id, source_type, source_ref, snippet, weight, created_at)
             SELECT
                'self:' || id,
                'self_evidence_events',
                CAST(id AS TEXT),
                source_type,
                NULL,
                snippet,
                weight,
                COALESCE(created_at, CURRENT_TIMESTAMP)
             FROM self_evidence_events",
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO evidence_links
                (link_id, evidence_id, target_type, target_id, relation, created_at)
             SELECT
                'link:ics:' || id || ':belief:' || belief_id,
                'ics:' || id,
                'belief',
                CAST(belief_id AS TEXT),
                'supports',
                COALESCE(created_at, CURRENT_TIMESTAMP)
             FROM ics_evidence_events",
        )
        .execute(&self.pool)
        .await;
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO evidence_links
                (link_id, evidence_id, target_type, target_id, relation, created_at)
             SELECT
                'link:self:' || id || ':belief:' || belief_id,
                'self:' || id,
                'self_belief',
                CAST(belief_id AS TEXT),
                'supports',
                COALESCE(created_at, CURRENT_TIMESTAMP)
             FROM self_evidence_events",
        )
        .execute(&self.pool)
        .await;

        // Migration: Convert reminders to use Unix timestamps (INTEGER) for reliable comparison
        // Check if migration is needed by trying to insert a test integer
        let needs_migration = sqlx::query("SELECT typeof(due_at) as dt FROM reminders LIMIT 1")
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten()
            .map(|row: sqlx::sqlite::SqliteRow| {
                use sqlx::Row;
                let dt: String = row.get("dt");
                dt != "integer"
            })
            .unwrap_or(false);
        
        if needs_migration {
            println!("[Migration] Converting reminders table to use Unix timestamps");
            // Recreate with INTEGER columns - SQLite doesn't support ALTER COLUMN
            let _ = sqlx::query("DROP TABLE IF EXISTS reminders_old").execute(&self.pool).await;
            let _ = sqlx::query("ALTER TABLE reminders RENAME TO reminders_old").execute(&self.pool).await;
            let _ = sqlx::query("CREATE TABLE reminders (id TEXT PRIMARY KEY, content TEXT NOT NULL, due_at INTEGER NOT NULL, type TEXT NOT NULL, status TEXT NOT NULL, created_at INTEGER NOT NULL)").execute(&self.pool).await;
            // Migrate existing data (convert datetime strings to timestamps)
            let _ = sqlx::query("INSERT INTO reminders SELECT id, content, strftime('%s', due_at), type, status, strftime('%s', created_at) FROM reminders_old").execute(&self.pool).await;
            let _ = sqlx::query("DROP TABLE reminders_old").execute(&self.pool).await;
        }


        // Migration: Fix Missing label_canonical in ics_entities (ICS v4.1)
        // Check if column exists, if not add it and backfill
        let start_migration = sqlx::query("SELECT label_canonical FROM ics_entities LIMIT 1").fetch_optional(&self.pool).await;
        if start_migration.is_err() {
             let _ = sqlx::query("ALTER TABLE ics_entities ADD COLUMN label_canonical TEXT NOT NULL DEFAULT ''").execute(&self.pool).await;
             let _ = sqlx::query("UPDATE ics_entities SET label_canonical = label WHERE label_canonical = ''").execute(&self.pool).await;
        }

        // Migration: Normalize label_canonical for spacing/case equivalence
        if let Ok(rows) = sqlx::query("SELECT id, label, label_canonical FROM ics_entities")
            .fetch_all(&self.pool)
            .await
        {
            for row in rows {
                let id: i64 = row.get("id");
                let label: String = row.get("label");
                let current: String = row.get("label_canonical");
                let canonical = canonicalize_label(&label);
                if canonical != current {
                    let _ = sqlx::query("UPDATE ics_entities SET label_canonical = ? WHERE id = ?")
                        .bind(canonical)
                        .bind(id)
                        .execute(&self.pool)
                        .await;
                }
            }
        }

        // Legacy prompt migrations are superseded by the updated default prompt.

        // ==================== ICS V4.1 TIME MODEL MIGRATIONS ====================
        
        // Migration: Add time bucket columns to ics_beliefs (§4 Time Model)
        let _ = sqlx::query("ALTER TABLE ics_beliefs ADD COLUMN time_bucket_kind TEXT NOT NULL DEFAULT 'atemporal'")
            .execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE ics_beliefs ADD COLUMN time_bucket_value TEXT")
            .execute(&self.pool).await;
        
        // Migration: Add observed_at for exact time tracking
        let _ = sqlx::query("ALTER TABLE ics_beliefs ADD COLUMN observed_at DATETIME")
            .execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE ics_beliefs ADD COLUMN valid_from DATETIME")
            .execute(&self.pool).await;
        let _ = sqlx::query("ALTER TABLE ics_beliefs ADD COLUMN valid_to DATETIME")
            .execute(&self.pool).await;

        if let Err(e) = migrate_signature_hashes(&self.pool).await {
            eprintln!("[DB] signature hash migration failed: {}", e);
        }

        if let Err(e) = seed_role_aliases(&self.pool).await {
            eprintln!("[DB] role alias seeding failed: {}", e);
        }
        if let Err(e) = seed_relation_shapes(&self.pool).await {
            eprintln!("[DB] relation shape seeding failed: {}", e);
        }
        if let Err(e) = migrate_rel_type_catalog(&self.pool).await {
            eprintln!("[DB] rel_type catalog migration failed: {}", e);
        }
        if let Err(e) = seed_rel_type_aliases(&self.pool).await {
            eprintln!("[DB] rel_type alias seeding failed: {}", e);
        }
        if let Err(e) = migrate_rel_signature_ids(&self.pool).await {
            eprintln!("[DB] rel signature id migration failed: {}", e);
        }
        if let Err(e) = migrate_workspace_meta_state(&self.pool).await {
            eprintln!("[DB] workspace meta migration failed: {}", e);
        }
        if let Err(e) = migrate_workspace_goal_stack(&self.pool).await {
            eprintln!("[DB] workspace goal stack migration failed: {}", e);
        }
        if let Err(e) = migrate_workspace_active_plan(&self.pool).await {
            eprintln!("[DB] workspace active plan migration failed: {}", e);
        }
        if let Err(e) = migrate_merge_events_schema(&self.pool).await {
            eprintln!("[DB] merge events migration failed: {}", e);
        }

        // Migration: Claim ledger index view
        let _ = sqlx::query("DROP VIEW IF EXISTS claim_ledger_index").execute(&self.pool).await;
        let _ = sqlx::query(
            "CREATE VIEW IF NOT EXISTS claim_ledger_index AS
             SELECT
                CAST(id AS TEXT) AS claim_id,
                scope,
                confidence,
                evidence_event_ids AS provenance_refs_json,
                CASE
                    WHEN expires_at IS NOT NULL AND datetime(expires_at) <= datetime('now') THEN 'DEPRECATED'
                    ELSE 'ACTIVE'
                END AS status
             FROM self_claims
             UNION ALL
             SELECT
                CAST(b.id AS TEXT) AS claim_id,
                b.scope,
                b.confidence,
                COALESCE((SELECT json_group_array(e.id) FROM ics_evidence_events e WHERE e.belief_id = b.id), '[]') AS provenance_refs_json,
                CASE
                    WHEN b.status = 'active' THEN 'ACTIVE'
                    ELSE 'DEPRECATED'
                END AS status
             FROM ics_beliefs b"
        )
        .execute(&self.pool)
        .await;

        // ==================== ICS V3 UPGRADE MIGRATIONS ====================

        // 1. Predicate Registry (Spec 2.1)
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS ics_predicate_registry (
                predicate_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                object_mode TEXT NOT NULL, -- 'unary' | 'entity' | 'literal'
                value_type TEXT, -- 'bool' | 'int' | 'float' | 'string' | 'enum' | 'date' | 'datetime' | 'duration' | 'json'
                partition_key TEXT,
                conflict_policy TEXT NOT NULL, -- 'latest_wins_in_scope' | 'highest_confidence_wins' | 'validity_window_based' | 'must_resolve_manually'
                is_traversal_relevant INTEGER NOT NULL DEFAULT 1,
                is_deprecated INTEGER NOT NULL DEFAULT 0,
                replaced_by TEXT,
                created_at TEXT NOT NULL
            )"
        ).execute(&self.pool).await;

        // 2. Seed Meta-Predicates (Spec 2.3)
        let meta_predicates = vec![
            ("rel.type", "Relation Type", "relation_instance", "literal", "enum", "must_resolve_manually"),
            ("rel.role", "Relation Role", "relation_instance", "literal", "json", "must_resolve_manually"),
            ("rel.about", "About Entity", "relation_instance", "entity", "string", "must_resolve_manually"),
            ("time.valid_from", "Valid From", "any", "literal", "datetime", "validity_window_based"),
            ("time.valid_to", "Valid To", "any", "literal", "datetime", "validity_window_based"),
            ("modal.modality", "Modality", "any", "literal", "enum", "latest_wins_in_scope"),
            ("modal.negated", "Negated", "any", "literal", "bool", "latest_wins_in_scope"),
            ("note.raw", "Raw Note (Quarantine)", "relation_instance", "literal", "json", "must_resolve_manually"),
        ];

        for (pid, name, stype, omode, vtype, cpol) in meta_predicates {
            let _ = sqlx::query(
                "INSERT INTO ics_predicate_registry (predicate_id, name, subject_type, object_mode, value_type, conflict_policy, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                 ON CONFLICT(predicate_id) DO UPDATE SET 
                 name=excluded.name, subject_type=excluded.subject_type, object_mode=excluded.object_mode, value_type=excluded.value_type, conflict_policy=excluded.conflict_policy"
            )
            .bind(pid).bind(name).bind(stype).bind(omode).bind(vtype).bind(cpol)
            .execute(&self.pool).await;
        }

        // 3. FTS5 Triggers & Tables (Spec 2.5) - FIX: PROACTIVE DROP to ensure schema matches V3
        // The previous "IF NOT EXISTS" caused silent failures if an old version existed without 'entity_id' column.
        
        // Rebuild Entities FTS
        let _ = sqlx::query("DROP TABLE IF EXISTS ics_entities_fts").execute(&self.pool).await;
        match sqlx::query(
            "CREATE VIRTUAL TABLE ics_entities_fts USING fts5(label, aliases, entity_id UNINDEXED)"
        )
        .execute(&self.pool)
        .await
        {
            Ok(_) => eprintln!("[DB] ics_entities_fts created"),
            Err(e) => eprintln!("[DB] Failed to create ics_entities_fts: {}", e),
        }

        // Rebuild Facts FTS (renamed from assertions)
        let _ = sqlx::query("DROP TABLE IF EXISTS ics_assertions_fts").execute(&self.pool).await; // Clean up old name
        let _ = sqlx::query("DROP TABLE IF EXISTS ics_facts_fts").execute(&self.pool).await;
        let _ = sqlx::query(
            "CREATE VIRTUAL TABLE ics_facts_fts USING fts5(key, value_literal, content='ics_fact_beliefs', content_rowid='belief_id')"
        ).execute(&self.pool).await;

        // Rebuild Relations FTS
        let _ = sqlx::query("DROP TABLE IF EXISTS ics_rel_fts").execute(&self.pool).await;
        let _ = sqlx::query(
            "CREATE VIRTUAL TABLE ics_rel_fts USING fts5(rel_type, roles)"
        ).execute(&self.pool).await;

        // Triggers - Entities
        let _ = sqlx::query("DROP TRIGGER IF EXISTS ics_ent_ai").execute(&self.pool).await;
        let _ = sqlx::query("DROP TRIGGER IF EXISTS ics_ent_ad").execute(&self.pool).await;
        let _ = sqlx::query("DROP TRIGGER IF EXISTS ics_ent_au").execute(&self.pool).await;

        let _ = sqlx::query(
            "CREATE TRIGGER ics_ent_ai AFTER INSERT ON ics_entities BEGIN 
               INSERT INTO ics_entities_fts(rowid, label, aliases, entity_id) VALUES (new.rowid, new.label, new.aliases, new.id); 
             END"
        ).execute(&self.pool).await;
        let _ = sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS ics_ent_ad AFTER DELETE ON ics_entities BEGIN 
               DELETE FROM ics_entities_fts WHERE rowid = old.rowid;
             END"
        ).execute(&self.pool).await;
        let _ = sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS ics_ent_au AFTER UPDATE ON ics_entities BEGIN 
               DELETE FROM ics_entities_fts WHERE rowid = old.rowid;
               INSERT INTO ics_entities_fts(rowid, label, aliases, entity_id) VALUES (new.rowid, new.label, new.aliases, new.id); 
             END"
        ).execute(&self.pool).await;

        // Facts Triggers (Corrected)
        let _ = sqlx::query("DROP TRIGGER IF EXISTS ics_fact_ai").execute(&self.pool).await;
        let _ = sqlx::query("DROP TRIGGER IF EXISTS ics_fact_ad").execute(&self.pool).await;
        let _ = sqlx::query("DROP TRIGGER IF EXISTS ics_fact_au").execute(&self.pool).await;
        
        // Also drop old assertion triggers if they exist
        let _ = sqlx::query("DROP TRIGGER IF EXISTS ics_asrt_ai").execute(&self.pool).await;
        let _ = sqlx::query("DROP TRIGGER IF EXISTS ics_asrt_ad").execute(&self.pool).await;
        let _ = sqlx::query("DROP TRIGGER IF EXISTS ics_asrt_au").execute(&self.pool).await;

        let _ = sqlx::query(
            "CREATE TRIGGER ics_fact_ai AFTER INSERT ON ics_fact_beliefs BEGIN 
               INSERT INTO ics_facts_fts(rowid, key, value_literal) VALUES (new.belief_id, new.key, new.value_literal); 
             END"
        ).execute(&self.pool).await;
        let _ = sqlx::query(
            "CREATE TRIGGER ics_fact_ad AFTER DELETE ON ics_fact_beliefs BEGIN 
               INSERT INTO ics_facts_fts(ics_facts_fts, rowid, key, value_literal) VALUES('delete', old.belief_id, old.key, old.value_literal); 
             END"
        ).execute(&self.pool).await;
        let _ = sqlx::query(
            "CREATE TRIGGER ics_fact_au AFTER UPDATE ON ics_fact_beliefs BEGIN 
               INSERT INTO ics_facts_fts(ics_facts_fts, rowid, key, value_literal) VALUES('delete', old.belief_id, old.key, old.value_literal); 
               INSERT INTO ics_facts_fts(rowid, key, value_literal) VALUES (new.belief_id, new.key, new.value_literal); 
             END"
        ).execute(&self.pool).await;

        // FORCE FTS REBUILD
        let _ = sqlx::query("INSERT INTO ics_entities_fts(rowid, label, aliases, entity_id) SELECT rowid, label, aliases, id FROM ics_entities").execute(&self.pool).await;
        let _ = sqlx::query("INSERT INTO ics_facts_fts(ics_facts_fts) VALUES('rebuild')").execute(&self.pool).await;
        let _ = sqlx::query(
            "INSERT INTO ics_rel_fts(rowid, rel_type, roles)
             SELECT rb.belief_id, rb.rel_type,
                    (SELECT group_concat(role, ' ') FROM ics_rel_participants rp WHERE rp.belief_id = rb.belief_id)
             FROM ics_rel_beliefs rb"
        )
        .execute(&self.pool)
        .await;

        // 4. Entity Degree Cache (New Feature: Hub Penalty)
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS ics_entity_degree_cache (
                entity_id TEXT PRIMARY KEY,
                out_degree INTEGER NOT NULL DEFAULT 0,
                in_degree INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )"
        ).execute(&self.pool).await;

        // NOTE: Legacy "ICS v3 Relations" prompt injection removed - replaced by ICS v4.1 DSL in default prompt

        // 6. Embeddings (Spec 3.1) - FIX: Added missing table from Phase 3
        // Dropping to fix legacy bad FK reference (ics_assertions -> ics_fact_beliefs)
        let _ = sqlx::query("DROP TABLE IF EXISTS ics_embeddings").execute(&self.pool).await;
        
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS ics_embeddings (
                id TEXT PRIMARY KEY,
                assertion_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                created_at DATETIME NOT NULL,
                FOREIGN KEY (assertion_id) REFERENCES ics_fact_beliefs(belief_id)
            )"
        ).execute(&self.pool).await;
        
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_ics_embeddings_assertion ON ics_embeddings(assertion_id)"
        ).execute(&self.pool).await;

        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS ics_embedding_lsh (
                assertion_id INTEGER NOT NULL,
                bucket INTEGER NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (assertion_id, bucket),
                FOREIGN KEY (assertion_id) REFERENCES ics_fact_beliefs(belief_id)
            )"
        ).execute(&self.pool).await;
        let _ = sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_ics_embedding_lsh_bucket ON ics_embedding_lsh(bucket)"
        ).execute(&self.pool).await;

        // ==================== ICS V4.1 DISAMBIGUATION TABLES ====================
        
        // Pending clarifications for ambiguous entity resolution
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS ics_pending_clarify (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                original_dsl TEXT NOT NULL,
                ref_text TEXT NOT NULL,
                candidates_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        ).execute(&self.pool).await;
        
        // Session-bound entity bindings (from disambiguation)
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS ics_session_bindings (
                session_id TEXT NOT NULL,
                ref_text TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, ref_text),
                FOREIGN KEY (entity_id) REFERENCES ics_entities(id)
            )"
        ).execute(&self.pool).await;
        
        // FACT→REL promotion mappings (schema-level, not user data)
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS ics_promotion_maps (
                id INTEGER PRIMARY KEY,
                from_fact_key TEXT NOT NULL UNIQUE,
                to_rel_type TEXT NOT NULL,
                subject_role TEXT NOT NULL,
                value_role TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        ).execute(&self.pool).await;
        
        // Seed common promotion patterns
        let promotion_seeds = vec![
            ("likes", "likes", "subject", "object"),
            ("loves", "loves", "subject", "object"),
            ("knows", "knows", "person", "person"),
            ("works_at", "works_at", "person", "company"),
            ("works_with", "works_with", "person", "person"),
            ("colleague", "works_with", "person", "person"),
            ("coworker", "works_with", "person", "person"),
            ("owns", "owns", "owner", "item"),
            ("friend", "friendship", "person", "person"),
            ("friends", "friendship", "person", "person"),
            ("parent", "parent_of", "child", "parent"),
            ("child", "parent_of", "parent", "child"),
            ("mother", "parent_of", "child", "mother"),
            ("father", "parent_of", "child", "father"),
            ("daughter", "parent_of", "parent", "daughter"),
            ("son", "parent_of", "parent", "son"),
            ("spouse", "spouse_of", "spouse", "spouse"),
            ("wife", "spouse_of", "spouse", "wife"),
            ("husband", "spouse_of", "spouse", "husband"),
            ("sibling", "sibling_of", "sibling", "sibling"),
            ("brother", "sibling_of", "sibling", "brother"),
            ("sister", "sibling_of", "sibling", "sister"),
            ("partner", "partner_of", "partner", "partner"),
        ];
        for (from_key, to_rel, subj, val) in promotion_seeds {
            let _ = sqlx::query(
                "INSERT INTO ics_promotion_maps (from_fact_key, to_rel_type, subject_role, value_role) 
                 VALUES (?, ?, ?, ?) ON CONFLICT(from_fact_key) DO NOTHING"
            )
            .bind(from_key).bind(to_rel).bind(subj).bind(val)
            .execute(&self.pool).await;
        }
        
        // Expire old pending clarifications (>1 hour)
        let _ = sqlx::query("UPDATE ics_pending_clarify SET status = 'expired' WHERE status = 'pending' AND created_at < datetime('now', '-1 hour')")
            .execute(&self.pool).await;
        
        // Conflict set members (§7.6)
        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS ics_conflict_set_members (
                conflict_set_id INTEGER NOT NULL,
                belief_id INTEGER NOT NULL,
                PRIMARY KEY (conflict_set_id, belief_id),
                FOREIGN KEY (conflict_set_id) REFERENCES ics_conflict_sets(id),
                FOREIGN KEY (belief_id) REFERENCES ics_beliefs(id)
            )"
        ).execute(&self.pool).await;

        // Schema drift alignment: relation shapes, promotion maps, conflict members.
        if table_exists(&self.pool, "ics_relation_shapes").await {
            let has_anchor_roles = column_exists(&self.pool, "ics_relation_shapes", "anchor_roles").await;
            let has_anchor_roles_json = column_exists(&self.pool, "ics_relation_shapes", "anchor_roles_json").await;
            if !has_anchor_roles && has_anchor_roles_json {
                let _ = sqlx::query(
                    "ALTER TABLE ics_relation_shapes ADD COLUMN anchor_roles TEXT NOT NULL DEFAULT '[]'"
                )
                .execute(&self.pool)
                .await;
                let _ = sqlx::query(
                    "UPDATE ics_relation_shapes SET anchor_roles = anchor_roles_json WHERE anchor_roles_json IS NOT NULL"
                )
                .execute(&self.pool)
                .await;
            }

            let has_cardinality_override = column_exists(&self.pool, "ics_relation_shapes", "cardinality_override").await;
            let has_cardinality = column_exists(&self.pool, "ics_relation_shapes", "cardinality").await;
            if !has_cardinality_override && has_cardinality {
                let _ = sqlx::query(
                    "ALTER TABLE ics_relation_shapes ADD COLUMN cardinality_override TEXT"
                )
                .execute(&self.pool)
                .await;
                let _ = sqlx::query(
                    "UPDATE ics_relation_shapes SET cardinality_override = cardinality WHERE cardinality IS NOT NULL"
                )
                .execute(&self.pool)
                .await;
            }

            let has_expected_arity = column_exists(&self.pool, "ics_relation_shapes", "expected_arity").await;
            if !has_expected_arity {
                let _ = sqlx::query(
                    "ALTER TABLE ics_relation_shapes ADD COLUMN expected_arity INTEGER"
                )
                .execute(&self.pool)
                .await;
            }

            let has_status = column_exists(&self.pool, "ics_relation_shapes", "status").await;
            if !has_status {
                let _ = sqlx::query(
                    "ALTER TABLE ics_relation_shapes ADD COLUMN status TEXT NOT NULL DEFAULT 'seeded'"
                )
                .execute(&self.pool)
                .await;
                let _ = sqlx::query(
                    "UPDATE ics_relation_shapes SET status = 'seeded' WHERE status IS NULL"
                )
                .execute(&self.pool)
                .await;
            }
        }

        if table_exists(&self.pool, "ics_rel_beliefs").await {
            let has_rel_type_norm = column_exists(&self.pool, "ics_rel_beliefs", "rel_type_norm").await;
            if !has_rel_type_norm {
                let _ = sqlx::query(
                    "ALTER TABLE ics_rel_beliefs ADD COLUMN rel_type_norm TEXT NOT NULL DEFAULT ''"
                )
                .execute(&self.pool)
                .await;
                let _ = sqlx::query(
                    "UPDATE ics_rel_beliefs SET rel_type_norm = rel_type WHERE rel_type_norm = '' OR rel_type_norm IS NULL"
                )
                .execute(&self.pool)
                .await;
            }

            let has_rel_type_raw = column_exists(&self.pool, "ics_rel_beliefs", "rel_type_raw").await;
            if !has_rel_type_raw {
                let _ = sqlx::query(
                    "ALTER TABLE ics_rel_beliefs ADD COLUMN rel_type_raw TEXT"
                )
                .execute(&self.pool)
                .await;
            }

            let has_rel_type_id = column_exists(&self.pool, "ics_rel_beliefs", "rel_type_id").await;
            if !has_rel_type_id {
                let _ = sqlx::query(
                    "ALTER TABLE ics_rel_beliefs ADD COLUMN rel_type_id TEXT"
                )
                .execute(&self.pool)
                .await;
            }
        }

        if table_exists(&self.pool, "memory_claims").await {
            let has_rel_type_raw = column_exists(&self.pool, "memory_claims", "rel_type_raw").await;
            if !has_rel_type_raw {
                let _ = sqlx::query(
                    "ALTER TABLE memory_claims ADD COLUMN rel_type_raw TEXT"
                )
                .execute(&self.pool)
                .await;
            }

            let has_rel_type_norm = column_exists(&self.pool, "memory_claims", "rel_type_norm").await;
            if !has_rel_type_norm {
                let _ = sqlx::query(
                    "ALTER TABLE memory_claims ADD COLUMN rel_type_norm TEXT"
                )
                .execute(&self.pool)
                .await;
            }

            let has_rel_type_id = column_exists(&self.pool, "memory_claims", "rel_type_id").await;
            if !has_rel_type_id {
                let _ = sqlx::query(
                    "ALTER TABLE memory_claims ADD COLUMN rel_type_id TEXT"
                )
                .execute(&self.pool)
                .await;
            }
        }

        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS ics_rel_type_aliases (
                alias TEXT PRIMARY KEY,
                rel_type TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                status TEXT NOT NULL DEFAULT 'confirmed',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        )
        .execute(&self.pool)
        .await;

        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS rel_type (
                rel_type_id TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL UNIQUE,
                description TEXT,
                status TEXT NOT NULL DEFAULT 'provisional',
                embedding TEXT,
                merged_into TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        )
        .execute(&self.pool)
        .await;

        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS rel_type_alias (
                alias TEXT PRIMARY KEY,
                rel_type_id TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                status TEXT NOT NULL DEFAULT 'confirmed',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        )
        .execute(&self.pool)
        .await;

        let _ = sqlx::query(
            "CREATE TABLE IF NOT EXISTS rel_shape (
                rel_type_id TEXT PRIMARY KEY,
                roles TEXT NOT NULL,
                anchor_roles TEXT NOT NULL,
                cardinality_override TEXT,
                commutative BOOLEAN NOT NULL DEFAULT 0,
                expected_arity INTEGER,
                status TEXT NOT NULL DEFAULT 'seeded',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"
        )
        .execute(&self.pool)
        .await;

        let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_rels_type_id ON ics_rel_beliefs(rel_type_id)")
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_rel_type_alias_rel_type_id ON rel_type_alias(rel_type_id)")
            .execute(&self.pool)
            .await;

        if table_exists(&self.pool, "ics_promotion_maps").await {
            let has_role_template = column_exists(&self.pool, "ics_promotion_maps", "role_template").await;
            let has_subject_role = column_exists(&self.pool, "ics_promotion_maps", "subject_role").await;
            let has_value_role = column_exists(&self.pool, "ics_promotion_maps", "value_role").await;
            if has_role_template && !(has_subject_role && has_value_role) {
                let rows = sqlx::query(
                    "SELECT from_fact_key, to_rel_type, role_template, status, created_at FROM ics_promotion_maps"
                )
                .fetch_all(&self.pool)
                .await
                .unwrap_or_default();

                let _ = sqlx::query("DROP TABLE IF EXISTS ics_promotion_maps_new")
                    .execute(&self.pool)
                    .await;
                let _ = sqlx::query(
                    "CREATE TABLE ics_promotion_maps_new (
                        id INTEGER PRIMARY KEY,
                        from_fact_key TEXT NOT NULL UNIQUE,
                        to_rel_type TEXT NOT NULL,
                        subject_role TEXT NOT NULL,
                        value_role TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'active',
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )"
                )
                .execute(&self.pool)
                .await;

                for row in rows {
                    let role_template: Option<String> = row.try_get("role_template").ok();
                    let mut subject_role = "subject".to_string();
                    let mut value_role = "object".to_string();
                    if let Some(template) = role_template {
                        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&template) {
                            if let Some(s) = value.get("subject_role").and_then(|v| v.as_str()) {
                                subject_role = s.to_string();
                            }
                            if let Some(v) = value.get("value_role").and_then(|v| v.as_str()) {
                                value_role = v.to_string();
                            }
                        }
                    }

                    let _ = sqlx::query(
                        "INSERT INTO ics_promotion_maps_new (from_fact_key, to_rel_type, subject_role, value_role, status, created_at)
                         VALUES (?, ?, ?, ?, ?, ?)"
                    )
                    .bind(row.get::<String, _>("from_fact_key"))
                    .bind(row.get::<String, _>("to_rel_type"))
                    .bind(subject_role)
                    .bind(value_role)
                    .bind(row.get::<String, _>("status"))
                    .bind(row.get::<String, _>("created_at"))
                    .execute(&self.pool)
                    .await;
                }

                let _ = sqlx::query("ALTER TABLE ics_promotion_maps RENAME TO ics_promotion_maps_old")
                    .execute(&self.pool)
                    .await;
                let _ = sqlx::query("ALTER TABLE ics_promotion_maps_new RENAME TO ics_promotion_maps")
                    .execute(&self.pool)
                    .await;
                let _ = sqlx::query("DROP TABLE IF EXISTS ics_promotion_maps_old")
                    .execute(&self.pool)
                    .await;
            }
        }

        if table_exists(&self.pool, "ics_conflict_members").await {
            let _ = sqlx::query(
                "CREATE TABLE IF NOT EXISTS ics_conflict_set_members (
                    conflict_set_id INTEGER NOT NULL,
                    belief_id INTEGER NOT NULL,
                    PRIMARY KEY (conflict_set_id, belief_id),
                    FOREIGN KEY (conflict_set_id) REFERENCES ics_conflict_sets(id),
                    FOREIGN KEY (belief_id) REFERENCES ics_beliefs(id)
                )"
            )
            .execute(&self.pool)
            .await;
            let _ = sqlx::query(
                "INSERT OR IGNORE INTO ics_conflict_set_members (conflict_set_id, belief_id)
                 SELECT conflict_set_id, belief_id FROM ics_conflict_members"
            )
            .execute(&self.pool)
            .await;
        }

        Ok(())
    }

    pub async fn get_settings(&self) -> Result<Settings, Box<dyn std::error::Error + Send + Sync>> {
        let diag = std::env::var("SYMBIOTE_DIAG_STDERR")
            .ok()
            .as_deref()
            == Some("1");
        if diag {
            eprintln!("[diag] db.get_settings step=0 entry");
        }
        let row = sqlx::query(
            "SELECT schema_version, api_base_url, api_key, streaming_enabled, history_window, injection_policy, request_defaults, active_model_id, json_reliable_model_id, system_prompt, voice_name, voice_speed, summarization_api_url, summarization_model, embedding_model, user_display_name, assistant_display_name, onboarding_completed, ui_theme, voice_pitch_semitones, voice_reverb_amount, voice_compression, voice_formant_shift, trace_history_limit, cockpit_write_enabled, episodic_enabled, episodic_injection_enabled, episodic_compaction_enabled, episodic_injection_limit, episodic_opt_out, memory_claims_enabled, phi_consent, seed_personal_user, lexical_fallback_enabled, memory_half_life_hours, research_budget_per_hour, research_budget_reset_window,
research_cost_per_call, monologue_interval_seconds, monologue_timeout_secs, monologue_retry_timeout_secs, empty_response_retry_max, empty_response_retry_timeout_ms, monologue_max_per_hour, thread_max_depth, allow_shell_tool, shell_command_allowlist, ask_budget_max, calculator_followups_max, loop_similarity_threshold, loop_recent_k, meta_cog_outcome_turns, meta_cog_cycle_window_turns, meta_cog_outcome_timeout_s, meta_cog_cooldown_s, meta_cog_streak_limit, registry_profile_name, controller_enabled, monologue_stabilization_enabled, monologue_surface_enabled, show_monologue_in_chat, enable_introspection, heartbeat_enabled, dream_enabled, binding_enforcement_enabled, pending_prompt_alignment_enabled, pending_prompt_recency_secs, auto_memory_pass_enabled, summary_cohesion_enabled, compact_prompt_enabled, context_hydration_mode, context_budgeter_enabled, context_miss_detector_enabled, world_model_reconcile_mode, goal_loop_enabled, goal_loop_interval_turns, goal_loop_load_threshold_ms, json_only_disabled_models, tool_failure_gate_window_mins, tool_failure_gate_tool_names, gate_default_soft, gate_shadow_mode, gate_rollout_percent, self_report_channel, self_awareness_expression_mode, explicit_feedback_only, weight_user_satisfaction, weight_policy_rigor, weight_latency, weight_evidence_strictness, weight_exploration, evidence_emit_budget, evidence_retention_days, gate_penalty_integration, evidence_auto_capture, response_fallback_enabled, memory_soft_anchor, context_extraction_boost, planner_enabled, confidence_calibration, scheduler_cognition, learning_feedback, evidence_semantics_v2, narrative_continuity, monologue_provenance_guard, organism_decay, model_context_limit, introspection_confidence_threshold, introspection_drift_threshold, introspection_ambiguity_threshold, enable_attribution_gate, enable_user_utterance_evidence, enable_attribution_metadata, enable_tool_schema_validation, enable_context_evidence, enable_monologue_validator, enable_memory_evidence_gating, enable_speculative_workspace_containment, stability_prompt_override_guard, stability_monologue_tagged, stability_introspection_structured, stability_disable_working_hypothesis, stability_state_disclosure_expanded, stability_transcript_normalization, stability_memory_hygiene, stability_non_stream_sanitization FROM settings WHERE id = 1"
        )
        .fetch_one(&self.pool)
        .await?;
        if diag {
            eprintln!("[diag] db.get_settings step=1 after_fetch_one");
        }

        use sqlx::Row;
        let mut settings = Settings {
            schema_version: row.get("schema_version"),
            api_base_url: row.get("api_base_url"),
            api_key: row.get("api_key"),
            streaming_enabled: row.get::<i32, _>("streaming_enabled") != 0,
            history_window: row.get("history_window"),
            injection_policy: row.get("injection_policy"),
            request_defaults: row.get::<Option<String>, _>("request_defaults").map(|s| serde_json::from_str(&s).unwrap_or_default()),
            active_model_id: row.get("active_model_id"),
            json_reliable_model_id: row.get("json_reliable_model_id"),
            system_prompt: row.get("system_prompt"),
            voice_name: row.get("voice_name"),
            voice_speed: row.get("voice_speed"),
            summarization_api_url: row.get("summarization_api_url"),
            summarization_model: row.get("summarization_model"),
            embedding_model: row.get("embedding_model"),
            user_display_name: row.get("user_display_name"),
            assistant_display_name: row.get("assistant_display_name"),
            onboarding_completed: row.try_get::<i32, _>("onboarding_completed").ok().map(|v| v != 0),
            ui_theme: row.get("ui_theme"),
            episodic_enabled: row.try_get::<i32, _>("episodic_enabled").ok().map(|v| v != 0),
            episodic_injection_enabled: row.try_get::<i32, _>("episodic_injection_enabled").ok().map(|v| v != 0),
            episodic_compaction_enabled: row.try_get::<i32, _>("episodic_compaction_enabled").ok().map(|v| v != 0),
            episodic_injection_limit: row.try_get("episodic_injection_limit").ok(),
            episodic_opt_out: row.try_get::<i32, _>("episodic_opt_out").ok().map(|v| v != 0),
            memory_claims_enabled: row.try_get::<i32, _>("memory_claims_enabled").ok().map(|v| v != 0),
            phi_consent: row.try_get::<i32, _>("phi_consent").ok().map(|v| v != 0),
            seed_personal_user: row.try_get::<i32, _>("seed_personal_user").ok().map(|v| v != 0),
            lexical_fallback_enabled: row.try_get::<i32, _>("lexical_fallback_enabled").ok().map(|v| v != 0),
            memory_half_life_hours: row.try_get::<f64, _>("memory_half_life_hours").ok().map(|v| v as f32),
            research_budget_per_hour: row.try_get::<i64, _>("research_budget_per_hour").ok(),
            research_budget_reset_window: row.try_get::<i64, _>("research_budget_reset_window").ok(),
            research_cost_per_call: row.try_get::<i64, _>("research_cost_per_call").ok(),
            monologue_interval_seconds: row.try_get::<i64, _>("monologue_interval_seconds").ok(),
            monologue_timeout_secs: row.try_get::<i64, _>("monologue_timeout_secs").ok(),
            monologue_retry_timeout_secs: row.try_get::<i64, _>("monologue_retry_timeout_secs").ok(),
            empty_response_retry_max: row
                .try_get::<i64, _>("empty_response_retry_max")
                .ok()
                .map(|v| v as i32),
            empty_response_retry_timeout_ms: row
                .try_get::<i64, _>("empty_response_retry_timeout_ms")
                .ok(),
            monologue_max_per_hour: row.try_get::<i64, _>("monologue_max_per_hour").ok(),
            thread_max_depth: row.try_get::<i64, _>("thread_max_depth").ok(),
            allow_shell_tool: row.try_get::<i32, _>("allow_shell_tool").ok().map(|v| v != 0),
            shell_command_allowlist: row.try_get("shell_command_allowlist").ok(),
            ask_budget_max: row.try_get("ask_budget_max").ok(),
            calculator_followups_max: row.try_get("calculator_followups_max").ok(),
            loop_similarity_threshold: row.try_get::<f64, _>("loop_similarity_threshold").ok().map(|v| v as f32),
            loop_recent_k: row.try_get("loop_recent_k").ok(),
            meta_cog_outcome_turns: row.try_get::<i64, _>("meta_cog_outcome_turns").ok(),
            meta_cog_cycle_window_turns: row.try_get::<i64, _>("meta_cog_cycle_window_turns").ok(),
            meta_cog_outcome_timeout_s: row.try_get::<i64, _>("meta_cog_outcome_timeout_s").ok(),
            meta_cog_cooldown_s: row.try_get::<i64, _>("meta_cog_cooldown_s").ok(),
            meta_cog_streak_limit: row
                .try_get::<i64, _>("meta_cog_streak_limit")
                .ok()
                .map(|v| v as i32),
            registry_profile_name: row.try_get("registry_profile_name").ok(),
            controller_enabled: row.try_get::<i32, _>("controller_enabled").ok().map(|v| v != 0),
            monologue_stabilization_enabled: row.try_get::<i32, _>("monologue_stabilization_enabled").ok().map(|v| v != 0),
            monologue_surface_enabled: row.try_get::<i32, _>("monologue_surface_enabled").ok().map(|v| v != 0),
            show_monologue_in_chat: row.try_get::<i32, _>("show_monologue_in_chat").ok().map(|v| v != 0),
            enable_introspection: row.try_get::<i32, _>("enable_introspection").ok().map(|v| v != 0),
            heartbeat_enabled: row.try_get::<i32, _>("heartbeat_enabled").ok().map(|v| v != 0),
            dream_enabled: row.try_get::<i32, _>("dream_enabled").ok().map(|v| v != 0),
            binding_enforcement_enabled: row.try_get::<i32, _>("binding_enforcement_enabled").ok().map(|v| v != 0),
            pending_prompt_alignment_enabled: row.try_get::<i32, _>("pending_prompt_alignment_enabled").ok().map(|v| v != 0),
            pending_prompt_recency_secs: row.try_get::<i64, _>("pending_prompt_recency_secs").ok(),
            auto_memory_pass_enabled: row.try_get::<i32, _>("auto_memory_pass_enabled").ok().map(|v| v != 0),
            summary_cohesion_enabled: row.try_get::<i32, _>("summary_cohesion_enabled").ok().map(|v| v != 0),
            compact_prompt_enabled: row.try_get::<i32, _>("compact_prompt_enabled").ok().map(|v| v != 0),
            context_hydration_mode: row.try_get("context_hydration_mode").ok(),
            context_budgeter_enabled: row.try_get::<i32, _>("context_budgeter_enabled").ok().map(|v| v != 0),
            context_miss_detector_enabled: row.try_get::<i32, _>("context_miss_detector_enabled").ok().map(|v| v != 0),
            world_model_reconcile_mode: row.try_get("world_model_reconcile_mode").ok(),
            goal_loop_enabled: row.try_get::<i32, _>("goal_loop_enabled").ok().map(|v| v != 0),
            goal_loop_interval_turns: row.try_get("goal_loop_interval_turns").ok(),
            goal_loop_load_threshold_ms: row.try_get("goal_loop_load_threshold_ms").ok(),
            json_only_disabled_models: row.try_get("json_only_disabled_models").ok(),
            tool_failure_gate_window_mins: row.try_get::<i64, _>("tool_failure_gate_window_mins").ok(),
            tool_failure_gate_tool_names: row.try_get("tool_failure_gate_tool_names").ok(),
            gate_default_soft: row.try_get::<i32, _>("gate_default_soft").ok().map(|v| v != 0),
            gate_shadow_mode: row.try_get::<i32, _>("gate_shadow_mode").ok().map(|v| v != 0),
            gate_rollout_percent: row
                .try_get::<i64, _>("gate_rollout_percent")
                .ok()
                .map(|v| v as i32),
            self_report_channel: row.try_get::<i32, _>("self_report_channel").ok().map(|v| v != 0),
            self_awareness_expression_mode: row.try_get("self_awareness_expression_mode").ok(),
            explicit_feedback_only: row.try_get::<i32, _>("explicit_feedback_only").ok().map(|v| v != 0),
            weight_user_satisfaction: row.try_get::<f64, _>("weight_user_satisfaction").ok().map(|v| v as f32),
            weight_policy_rigor: row.try_get::<f64, _>("weight_policy_rigor").ok().map(|v| v as f32),
            weight_latency: row.try_get::<f64, _>("weight_latency").ok().map(|v| v as f32),
            weight_evidence_strictness: row.try_get::<f64, _>("weight_evidence_strictness").ok().map(|v| v as f32),
            weight_exploration: row.try_get::<f64, _>("weight_exploration").ok().map(|v| v as f32),
            evidence_emit_budget: row.try_get::<i64, _>("evidence_emit_budget").ok().map(|v| v as i32),
            evidence_retention_days: row.try_get::<i64, _>("evidence_retention_days").ok().map(|v| v as i32),
            gate_penalty_integration: row.try_get::<i32, _>("gate_penalty_integration").ok().map(|v| v != 0),
            evidence_auto_capture: row.try_get::<i32, _>("evidence_auto_capture").ok().map(|v| v != 0),
            response_fallback_enabled: row.try_get::<i32, _>("response_fallback_enabled").ok().map(|v| v != 0),
            memory_soft_anchor: row.try_get::<i32, _>("memory_soft_anchor").ok().map(|v| v != 0),
            context_extraction_boost: row.try_get::<i32, _>("context_extraction_boost").ok().map(|v| v != 0),
            planner_enabled: row.try_get::<i32, _>("planner_enabled").ok().map(|v| v != 0),
            confidence_calibration: row.try_get::<i32, _>("confidence_calibration").ok().map(|v| v != 0),
            scheduler_cognition: row.try_get::<i32, _>("scheduler_cognition").ok().map(|v| v != 0),
            learning_feedback: row.try_get::<i32, _>("learning_feedback").ok().map(|v| v != 0),
            evidence_semantics_v2: row.try_get::<i32, _>("evidence_semantics_v2").ok().map(|v| v != 0),
            narrative_continuity: row.try_get::<i32, _>("narrative_continuity").ok().map(|v| v != 0),
            monologue_provenance_guard: row
                .try_get::<i32, _>("monologue_provenance_guard")
                .ok()
                .map(|v| v != 0),
            organism_decay: row.try_get::<i32, _>("organism_decay").ok().map(|v| v != 0),
            model_context_limit: row.try_get("model_context_limit").ok(),
            introspection_confidence_threshold: row.try_get::<f64, _>("introspection_confidence_threshold").ok().map(|v| v as f32),
            introspection_drift_threshold: row.try_get::<f64, _>("introspection_drift_threshold").ok().map(|v| v as f32),
            introspection_ambiguity_threshold: row.try_get::<f64, _>("introspection_ambiguity_threshold").ok().map(|v| v as f32),
            enable_attribution_gate: row.try_get::<i32, _>("enable_attribution_gate").ok().map(|v| v != 0),
            enable_user_utterance_evidence: row.try_get::<i32, _>("enable_user_utterance_evidence").ok().map(|v| v != 0),
            enable_attribution_metadata: row.try_get::<i32, _>("enable_attribution_metadata").ok().map(|v| v != 0),
            enable_tool_schema_validation: row.try_get::<i32, _>("enable_tool_schema_validation").ok().map(|v| v != 0),
            enable_context_evidence: row.try_get::<i32, _>("enable_context_evidence").ok().map(|v| v != 0),
            enable_monologue_validator: row.try_get::<i32, _>("enable_monologue_validator").ok().map(|v| v != 0),
            enable_memory_evidence_gating: row.try_get::<i32, _>("enable_memory_evidence_gating").ok().map(|v| v != 0),
            enable_speculative_workspace_containment: row.try_get::<i32, _>("enable_speculative_workspace_containment").ok().map(|v| v != 0),
            stability_prompt_override_guard: row.try_get::<i32, _>("stability_prompt_override_guard").ok().map(|v| v != 0),
            stability_monologue_tagged: row.try_get::<i32, _>("stability_monologue_tagged").ok().map(|v| v != 0),
            stability_introspection_structured: row.try_get::<i32, _>("stability_introspection_structured").ok().map(|v| v != 0),
            stability_disable_working_hypothesis: row.try_get::<i32, _>("stability_disable_working_hypothesis").ok().map(|v| v != 0),
            stability_state_disclosure_expanded: row.try_get::<i32, _>("stability_state_disclosure_expanded").ok().map(|v| v != 0),
            stability_transcript_normalization: row.try_get::<i32, _>("stability_transcript_normalization").ok().map(|v| v != 0),
            stability_memory_hygiene: row.try_get::<i32, _>("stability_memory_hygiene").ok().map(|v| v != 0),
            stability_non_stream_sanitization: row.try_get::<i32, _>("stability_non_stream_sanitization").ok().map(|v| v != 0),
            voice_pitch_semitones: row.get("voice_pitch_semitones"),
            voice_reverb_amount: row.get("voice_reverb_amount"),
            voice_compression: row.get("voice_compression"),
            voice_formant_shift: row.get("voice_formant_shift"),
            trace_history_limit: row.get("trace_history_limit"),
            cockpit_write_enabled: row.try_get::<i32, _>("cockpit_write_enabled").ok().map(|v| v != 0),
        };
        if diag {
            eprintln!("[diag] db.get_settings step=2 after_build_settings");
        }
        let adjustments = settings.validate();
        if diag {
            eprintln!(
                "[diag] db.get_settings step=3 after_validate adjustments={}",
                adjustments.len()
            );
        }
        if !adjustments.is_empty() {
            if diag {
                eprintln!("[diag] db.get_settings step=4 before_log_adjustments");
            }
            let _ = system_log::log_event(
                &self.pool,
                None,
                "info",
                "settings",
                None,
                None,
                serde_json::json!({
                    "event": "settings_validation_adjusted",
                    "adjustments": adjustments,
                }),
            )
            .await;
            if diag {
                eprintln!("[diag] db.get_settings step=5 after_log_adjustments");
            }
        }

        if diag {
            eprintln!("[diag] db.get_settings step=6 return");
        }
        Ok(settings)
    }

    pub async fn update_settings(&self, settings: Settings) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let current = self.get_settings().await?;
        let request_defaults = settings.request_defaults.map(|v| v.to_string());
        let json_reliable_model_id = settings
            .json_reliable_model_id
            .or(current.json_reliable_model_id.clone());
        let ui_theme = settings
            .ui_theme
            .or(current.ui_theme)
            .unwrap_or_else(|| "builtin:utopia".to_string());
        let episodic_enabled = settings
            .episodic_enabled
            .unwrap_or(current.episodic_enabled.unwrap_or(false)) as i32;
        let episodic_injection_enabled = settings
            .episodic_injection_enabled
            .unwrap_or(current.episodic_injection_enabled.unwrap_or(false)) as i32;
        let episodic_compaction_enabled = settings
            .episodic_compaction_enabled
            .unwrap_or(current.episodic_compaction_enabled.unwrap_or(false)) as i32;
        let episodic_injection_limit = settings
            .episodic_injection_limit
            .or(current.episodic_injection_limit)
            .unwrap_or(5);
        let episodic_opt_out = match settings.episodic_enabled {
            Some(false) => 1,
            Some(true) => 0,
            None => current.episodic_opt_out.unwrap_or(false) as i32,
        };
        let memory_claims_enabled = settings
            .memory_claims_enabled
            .unwrap_or(current.memory_claims_enabled.unwrap_or(false)) as i32;
        let phi_consent = settings
            .phi_consent
            .unwrap_or(current.phi_consent.unwrap_or(false)) as i32;
        let seed_personal_user = settings
            .seed_personal_user
            .unwrap_or(current.seed_personal_user.unwrap_or(true)) as i32;
        let lexical_fallback_enabled = settings
            .lexical_fallback_enabled
            .unwrap_or(current.lexical_fallback_enabled.unwrap_or(false)) as i32;
        let memory_half_life_hours = settings
            .memory_half_life_hours
            .or(current.memory_half_life_hours)
            .unwrap_or(168.0);
        let research_budget_per_hour = settings
            .research_budget_per_hour
            .or(current.research_budget_per_hour)
            .unwrap_or(0);
        let research_budget_reset_window = settings
            .research_budget_reset_window
            .or(current.research_budget_reset_window)
            .unwrap_or(60);
        let research_cost_per_call = settings
            .research_cost_per_call
            .or(current.research_cost_per_call)
            .unwrap_or(1);
        let monologue_interval_seconds = settings
            .monologue_interval_seconds
            .or(current.monologue_interval_seconds)
            .unwrap_or(60);
        let monologue_timeout_secs = settings
            .monologue_timeout_secs
            .or(current.monologue_timeout_secs)
            .unwrap_or(75);
        let monologue_retry_timeout_secs = settings
            .monologue_retry_timeout_secs
            .or(current.monologue_retry_timeout_secs)
            .unwrap_or(25);
        let empty_response_retry_max = settings
            .empty_response_retry_max
            .or(current.empty_response_retry_max)
            .unwrap_or(3)
            .max(0);
        let empty_response_retry_timeout_ms = settings
            .empty_response_retry_timeout_ms
            .or(current.empty_response_retry_timeout_ms)
            .unwrap_or(4000)
            .max(0);
        let monologue_max_per_hour = settings
            .monologue_max_per_hour
            .or(current.monologue_max_per_hour)
            .unwrap_or(12);
        let thread_max_depth = settings
            .thread_max_depth
            .or(current.thread_max_depth)
            .unwrap_or(2);
        let allow_shell_tool = settings
            .allow_shell_tool
            .or(current.allow_shell_tool)
            .unwrap_or(false) as i32;
        let shell_command_allowlist = settings
            .shell_command_allowlist
            .or(current.shell_command_allowlist);
        let ask_budget_max = settings
            .ask_budget_max
            .or(current.ask_budget_max)
            .unwrap_or(1);
        let calculator_followups_max = settings
            .calculator_followups_max
            .or(current.calculator_followups_max)
            .unwrap_or(0);
        let loop_similarity_threshold = settings
            .loop_similarity_threshold
            .or(current.loop_similarity_threshold)
            .unwrap_or(0.85);
        let loop_recent_k = settings
            .loop_recent_k
            .or(current.loop_recent_k)
            .unwrap_or(6);
        let meta_cog_outcome_turns = settings
            .meta_cog_outcome_turns
            .or(current.meta_cog_outcome_turns)
            .unwrap_or(3)
            .max(1);
        let meta_cog_cycle_window_turns = settings
            .meta_cog_cycle_window_turns
            .or(current.meta_cog_cycle_window_turns)
            .unwrap_or(2)
            .max(1);
        let meta_cog_outcome_timeout_s = settings
            .meta_cog_outcome_timeout_s
            .or(current.meta_cog_outcome_timeout_s)
            .unwrap_or(120)
            .max(30);
        let meta_cog_cooldown_s = settings
            .meta_cog_cooldown_s
            .or(current.meta_cog_cooldown_s)
            .unwrap_or(60)
            .max(10);
        let meta_cog_streak_limit = settings
            .meta_cog_streak_limit
            .or(current.meta_cog_streak_limit)
            .unwrap_or(3)
            .max(1);
        let registry_profile_name = settings
            .registry_profile_name
            .or(current.registry_profile_name)
            .unwrap_or_else(|| "default".to_string());
        let controller_enabled = settings
            .controller_enabled
            .unwrap_or(current.controller_enabled.unwrap_or(true)) as i32;
        let monologue_stabilization_enabled = settings
            .monologue_stabilization_enabled
            .unwrap_or(current.monologue_stabilization_enabled.unwrap_or(true)) as i32;
        let monologue_surface_enabled = settings
            .monologue_surface_enabled
            .unwrap_or(current.monologue_surface_enabled.unwrap_or(false)) as i32;
        let show_monologue_in_chat = settings
            .show_monologue_in_chat
            .unwrap_or(current.show_monologue_in_chat.unwrap_or(false)) as i32;
        let enable_introspection = settings
            .enable_introspection
            .unwrap_or(current.enable_introspection.unwrap_or(true)) as i32;
        let heartbeat_enabled = settings
            .heartbeat_enabled
            .unwrap_or(current.heartbeat_enabled.unwrap_or(true)) as i32;
        let dream_enabled = settings
            .dream_enabled
            .unwrap_or(current.dream_enabled.unwrap_or(true)) as i32;
        let binding_enforcement_enabled = settings
            .binding_enforcement_enabled
            .unwrap_or(current.binding_enforcement_enabled.unwrap_or(true)) as i32;
        let pending_prompt_alignment_enabled = settings
            .pending_prompt_alignment_enabled
            .unwrap_or(current.pending_prompt_alignment_enabled.unwrap_or(true)) as i32;
        let pending_prompt_recency_secs = settings
            .pending_prompt_recency_secs
            .or(current.pending_prompt_recency_secs)
            .unwrap_or(90);
        let auto_memory_pass_enabled = settings
            .auto_memory_pass_enabled
            .unwrap_or(current.auto_memory_pass_enabled.unwrap_or(true)) as i32;
        let summary_cohesion_enabled = settings
            .summary_cohesion_enabled
            .unwrap_or(current.summary_cohesion_enabled.unwrap_or(true)) as i32;
        let compact_prompt_enabled = settings
            .compact_prompt_enabled
            .unwrap_or(current.compact_prompt_enabled.unwrap_or(true)) as i32;
        let context_hydration_mode = settings
            .context_hydration_mode
            .or(current.context_hydration_mode.clone())
            .unwrap_or_else(|| "shadow".to_string());
        let context_budgeter_enabled = settings
            .context_budgeter_enabled
            .unwrap_or(current.context_budgeter_enabled.unwrap_or(true)) as i32;
        let context_miss_detector_enabled = settings
            .context_miss_detector_enabled
            .unwrap_or(current.context_miss_detector_enabled.unwrap_or(true)) as i32;
        let world_model_reconcile_mode = settings
            .world_model_reconcile_mode
            .or(current.world_model_reconcile_mode.clone())
            .unwrap_or_else(|| "shadow".to_string());
        let goal_loop_enabled = settings
            .goal_loop_enabled
            .unwrap_or(current.goal_loop_enabled.unwrap_or(true)) as i32;
        let goal_loop_interval_turns = settings
            .goal_loop_interval_turns
            .or(current.goal_loop_interval_turns)
            .unwrap_or(3)
            .max(1);
        let goal_loop_load_threshold_ms = settings
            .goal_loop_load_threshold_ms
            .or(current.goal_loop_load_threshold_ms)
            .unwrap_or(650)
            .max(100);
        let json_only_disabled_models = settings
            .json_only_disabled_models
            .or(current.json_only_disabled_models.clone());
        let tool_failure_gate_window_mins = settings
            .tool_failure_gate_window_mins
            .or(current.tool_failure_gate_window_mins);
        let tool_failure_gate_tool_names = settings
            .tool_failure_gate_tool_names
            .or(current.tool_failure_gate_tool_names.clone());
        let gate_default_soft = settings
            .gate_default_soft
            .unwrap_or(current.gate_default_soft.unwrap_or(true)) as i32;
        let gate_shadow_mode = settings
            .gate_shadow_mode
            .unwrap_or(current.gate_shadow_mode.unwrap_or(false)) as i32;
        let gate_rollout_percent = settings
            .gate_rollout_percent
            .or(current.gate_rollout_percent)
            .unwrap_or(100)
            .clamp(0, 100);
        let self_report_channel = settings
            .self_report_channel
            .unwrap_or(current.self_report_channel.unwrap_or(true)) as i32;
        let self_awareness_expression_mode = settings
            .self_awareness_expression_mode
            .or(current.self_awareness_expression_mode)
            .unwrap_or_else(|| "conservative".to_string());
        let explicit_feedback_only = settings
            .explicit_feedback_only
            .unwrap_or(current.explicit_feedback_only.unwrap_or(true)) as i32;
        let weight_user_satisfaction = settings
            .weight_user_satisfaction
            .or(current.weight_user_satisfaction)
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        let weight_policy_rigor = settings
            .weight_policy_rigor
            .or(current.weight_policy_rigor)
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        let weight_latency = settings
            .weight_latency
            .or(current.weight_latency)
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        let weight_evidence_strictness = settings
            .weight_evidence_strictness
            .or(current.weight_evidence_strictness)
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        let weight_exploration = settings
            .weight_exploration
            .or(current.weight_exploration)
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        let evidence_emit_budget = settings
            .evidence_emit_budget
            .or(current.evidence_emit_budget)
            .unwrap_or(50)
            .max(0);
        let evidence_retention_days = settings
            .evidence_retention_days
            .or(current.evidence_retention_days)
            .unwrap_or(30)
            .max(0);
        let gate_penalty_integration = settings
            .gate_penalty_integration
            .unwrap_or(current.gate_penalty_integration.unwrap_or(true)) as i32;
        let evidence_auto_capture = settings
            .evidence_auto_capture
            .unwrap_or(current.evidence_auto_capture.unwrap_or(true)) as i32;
        let response_fallback_enabled = settings
            .response_fallback_enabled
            .unwrap_or(current.response_fallback_enabled.unwrap_or(true)) as i32;
        let memory_soft_anchor = settings
            .memory_soft_anchor
            .unwrap_or(current.memory_soft_anchor.unwrap_or(true)) as i32;
        let context_extraction_boost = settings
            .context_extraction_boost
            .unwrap_or(current.context_extraction_boost.unwrap_or(true)) as i32;
        let planner_enabled = settings
            .planner_enabled
            .unwrap_or(current.planner_enabled.unwrap_or(true)) as i32;
        let confidence_calibration = settings
            .confidence_calibration
            .unwrap_or(current.confidence_calibration.unwrap_or(true)) as i32;
        let scheduler_cognition = settings
            .scheduler_cognition
            .unwrap_or(current.scheduler_cognition.unwrap_or(true)) as i32;
        let learning_feedback = settings
            .learning_feedback
            .unwrap_or(current.learning_feedback.unwrap_or(true)) as i32;
        let evidence_semantics_v2 = settings
            .evidence_semantics_v2
            .unwrap_or(current.evidence_semantics_v2.unwrap_or(true)) as i32;
        let narrative_continuity = settings
            .narrative_continuity
            .unwrap_or(current.narrative_continuity.unwrap_or(true)) as i32;
        let monologue_provenance_guard = settings
            .monologue_provenance_guard
            .unwrap_or(current.monologue_provenance_guard.unwrap_or(true)) as i32;
        let organism_decay = settings
            .organism_decay
            .unwrap_or(current.organism_decay.unwrap_or(true)) as i32;
        let model_context_limit = settings
            .model_context_limit
            .or(current.model_context_limit)
            .unwrap_or(16384)
            .max(1024);
        let introspection_confidence_threshold = settings
            .introspection_confidence_threshold
            .or(current.introspection_confidence_threshold)
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        let introspection_drift_threshold = settings
            .introspection_drift_threshold
            .or(current.introspection_drift_threshold)
            .unwrap_or(0.6)
            .clamp(0.0, 1.0);
        let introspection_ambiguity_threshold = settings
            .introspection_ambiguity_threshold
            .or(current.introspection_ambiguity_threshold)
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        let enable_attribution_gate = settings
            .enable_attribution_gate
            .unwrap_or(current.enable_attribution_gate.unwrap_or(true)) as i32;
        let enable_user_utterance_evidence = settings
            .enable_user_utterance_evidence
            .unwrap_or(current.enable_user_utterance_evidence.unwrap_or(true)) as i32;
        let enable_attribution_metadata = settings
            .enable_attribution_metadata
            .unwrap_or(current.enable_attribution_metadata.unwrap_or(true)) as i32;
        let enable_tool_schema_validation = settings
            .enable_tool_schema_validation
            .unwrap_or(current.enable_tool_schema_validation.unwrap_or(true)) as i32;
        let enable_context_evidence = settings
            .enable_context_evidence
            .unwrap_or(current.enable_context_evidence.unwrap_or(true)) as i32;
        let enable_monologue_validator = settings
            .enable_monologue_validator
            .unwrap_or(current.enable_monologue_validator.unwrap_or(true)) as i32;
        let enable_memory_evidence_gating = settings
            .enable_memory_evidence_gating
            .unwrap_or(current.enable_memory_evidence_gating.unwrap_or(true)) as i32;
        let enable_speculative_workspace_containment = settings
            .enable_speculative_workspace_containment
            .unwrap_or(current.enable_speculative_workspace_containment.unwrap_or(true)) as i32;
        let stability_prompt_override_guard = settings
            .stability_prompt_override_guard
            .unwrap_or(current.stability_prompt_override_guard.unwrap_or(true)) as i32;
        let stability_monologue_tagged = settings
            .stability_monologue_tagged
            .unwrap_or(current.stability_monologue_tagged.unwrap_or(true)) as i32;
        let stability_introspection_structured = settings
            .stability_introspection_structured
            .unwrap_or(current.stability_introspection_structured.unwrap_or(true)) as i32;
        let stability_disable_working_hypothesis = settings
            .stability_disable_working_hypothesis
            .unwrap_or(current.stability_disable_working_hypothesis.unwrap_or(true)) as i32;
        let stability_state_disclosure_expanded = settings
            .stability_state_disclosure_expanded
            .unwrap_or(current.stability_state_disclosure_expanded.unwrap_or(true)) as i32;
        let stability_transcript_normalization = settings
            .stability_transcript_normalization
            .unwrap_or(current.stability_transcript_normalization.unwrap_or(true)) as i32;
        let stability_memory_hygiene = settings
            .stability_memory_hygiene
            .unwrap_or(current.stability_memory_hygiene.unwrap_or(true)) as i32;
        let stability_non_stream_sanitization = settings
            .stability_non_stream_sanitization
            .unwrap_or(current.stability_non_stream_sanitization.unwrap_or(true)) as i32;
        let cockpit_write_enabled = settings
            .cockpit_write_enabled
            .unwrap_or(current.cockpit_write_enabled.unwrap_or(false)) as i32;
        let onboarding_completed = settings
            .onboarding_completed
            .or(current.onboarding_completed)
            .unwrap_or(false) as i32;
        if let Err(e) = sqlx::query(
            "UPDATE settings SET api_base_url = ?, api_key = ?, streaming_enabled = ?, history_window = ?, injection_policy = ?, request_defaults = ?, active_model_id = ?, json_reliable_model_id = ?, system_prompt = ?, voice_name = ?, voice_speed = ?, summarization_api_url = ?, summarization_model = ?, embedding_model = ?, user_display_name = ?, assistant_display_name = ?, onboarding_completed = ?, ui_theme = ?, episodic_enabled = ?, episodic_injection_enabled = ?, episodic_compaction_enabled = ?, episodic_injection_limit = ?, episodic_opt_out = ?, memory_claims_enabled = ?, phi_consent = ?, seed_personal_user = ?, lexical_fallback_enabled = ?, memory_half_life_hours = ?, voice_pitch_semitones = ?, voice_reverb_amount = ?, voice_compression = ?, voice_formant_shift = ?, trace_history_limit = ?, cockpit_write_enabled = ?, research_budget_per_hour = ?, 
research_budget_reset_window = ?, research_cost_per_call = ?, monologue_interval_seconds = ?, monologue_timeout_secs = ?, monologue_retry_timeout_secs = ?, empty_response_retry_max = ?, empty_response_retry_timeout_ms = ?, monologue_max_per_hour = ?, 
thread_max_depth = ?, allow_shell_tool = ?, shell_command_allowlist = ?, ask_budget_max = ?, calculator_followups_max = ?, loop_similarity_threshold = ?, loop_recent_k = ?, meta_cog_outcome_turns = ?, meta_cog_cycle_window_turns = ?, meta_cog_outcome_timeout_s = ?, meta_cog_cooldown_s = ?, meta_cog_streak_limit = ?, registry_profile_name = ?, controller_enabled = ?, monologue_stabilization_enabled = ?, monologue_surface_enabled = ?, show_monologue_in_chat = ?, enable_introspection = ?, heartbeat_enabled = ?, dream_enabled = ?, binding_enforcement_enabled = ?, pending_prompt_alignment_enabled = ?, pending_prompt_recency_secs = ?, auto_memory_pass_enabled = ?, summary_cohesion_enabled = ?, compact_prompt_enabled = ?, context_hydration_mode = ?, context_budgeter_enabled = ?, context_miss_detector_enabled = ?, world_model_reconcile_mode = ?, goal_loop_enabled = ?, goal_loop_interval_turns = ?, goal_loop_load_threshold_ms = ?, json_only_disabled_models = ?, tool_failure_gate_window_mins = ?, tool_failure_gate_tool_names = ?, gate_default_soft = ?, gate_shadow_mode = ?, gate_rollout_percent = ?, self_report_channel = ?, self_awareness_expression_mode = ?, explicit_feedback_only = ?, weight_user_satisfaction = ?, weight_policy_rigor = ?, weight_latency = ?, weight_evidence_strictness = ?, weight_exploration = ?, evidence_emit_budget = ?, evidence_retention_days = ?, gate_penalty_integration = ?, evidence_auto_capture = ?, response_fallback_enabled = ?, memory_soft_anchor = ?, context_extraction_boost = ?, planner_enabled = ?, confidence_calibration = ?, scheduler_cognition = ?, learning_feedback = ?, evidence_semantics_v2 = ?, narrative_continuity = ?, monologue_provenance_guard = ?, organism_decay = ?, model_context_limit = ?, introspection_confidence_threshold = ?, introspection_drift_threshold = ?, introspection_ambiguity_threshold = ?, enable_attribution_gate = ?, enable_user_utterance_evidence = ?, enable_attribution_metadata = ?, enable_tool_schema_validation = ?, enable_context_evidence = ?, enable_monologue_validator = ?, enable_memory_evidence_gating = ?, enable_speculative_workspace_containment = ?, stability_prompt_override_guard = ?, stability_monologue_tagged = ?, stability_introspection_structured = ?, stability_disable_working_hypothesis = ?, stability_state_disclosure_expanded = ?, stability_transcript_normalization = ?, stability_memory_hygiene = ?, stability_non_stream_sanitization = ? WHERE id = 1"
        )
        .bind(settings.api_base_url)
        .bind(settings.api_key)
        .bind(settings.streaming_enabled)
        .bind(settings.history_window)
        .bind(settings.injection_policy)
        .bind(request_defaults)
        .bind(settings.active_model_id)
        .bind(json_reliable_model_id)
        .bind(settings.system_prompt)
        .bind(settings.voice_name)
        .bind(settings.voice_speed)
        .bind(settings.summarization_api_url)
        .bind(settings.summarization_model)
        .bind(settings.embedding_model)
        .bind(settings.user_display_name.clone())
        .bind(settings.assistant_display_name.clone())
        .bind(onboarding_completed)
        .bind(ui_theme)
        .bind(episodic_enabled)
        .bind(episodic_injection_enabled)
        .bind(episodic_compaction_enabled)
        .bind(episodic_injection_limit)
        .bind(episodic_opt_out)
        .bind(memory_claims_enabled)
        .bind(phi_consent)
        .bind(seed_personal_user)
        .bind(lexical_fallback_enabled)
        .bind(memory_half_life_hours)
        .bind(settings.voice_pitch_semitones)
        .bind(settings.voice_reverb_amount)
        .bind(settings.voice_compression)
        .bind(settings.voice_formant_shift)
        .bind(settings.trace_history_limit)
        .bind(cockpit_write_enabled)
        .bind(research_budget_per_hour)
        .bind(research_budget_reset_window)
        .bind(research_cost_per_call)
        .bind(monologue_interval_seconds)
        .bind(monologue_timeout_secs)
        .bind(monologue_retry_timeout_secs)
        .bind(empty_response_retry_max)
        .bind(empty_response_retry_timeout_ms)
        .bind(monologue_max_per_hour)
        .bind(thread_max_depth)
        .bind(allow_shell_tool)
        .bind(shell_command_allowlist)
        .bind(ask_budget_max)
        .bind(calculator_followups_max)
        .bind(loop_similarity_threshold)
        .bind(loop_recent_k)
        .bind(meta_cog_outcome_turns)
        .bind(meta_cog_cycle_window_turns)
        .bind(meta_cog_outcome_timeout_s)
        .bind(meta_cog_cooldown_s)
        .bind(meta_cog_streak_limit)
        .bind(registry_profile_name)
        .bind(controller_enabled)
        .bind(monologue_stabilization_enabled)
        .bind(monologue_surface_enabled)
        .bind(show_monologue_in_chat)
        .bind(enable_introspection)
        .bind(heartbeat_enabled)
        .bind(dream_enabled)
        .bind(binding_enforcement_enabled)
        .bind(pending_prompt_alignment_enabled)
        .bind(pending_prompt_recency_secs)
        .bind(auto_memory_pass_enabled)
        .bind(summary_cohesion_enabled)
        .bind(compact_prompt_enabled)
        .bind(context_hydration_mode)
        .bind(context_budgeter_enabled)
        .bind(context_miss_detector_enabled)
        .bind(world_model_reconcile_mode)
        .bind(goal_loop_enabled)
        .bind(goal_loop_interval_turns)
        .bind(goal_loop_load_threshold_ms)
        .bind(json_only_disabled_models)
        .bind(tool_failure_gate_window_mins)
        .bind(tool_failure_gate_tool_names)
        .bind(gate_default_soft)
        .bind(gate_shadow_mode)
        .bind(gate_rollout_percent)
        .bind(self_report_channel)
        .bind(self_awareness_expression_mode)
        .bind(explicit_feedback_only)
        .bind(weight_user_satisfaction)
        .bind(weight_policy_rigor)
        .bind(weight_latency)
        .bind(weight_evidence_strictness)
        .bind(weight_exploration)
        .bind(evidence_emit_budget)
        .bind(evidence_retention_days)
        .bind(gate_penalty_integration)
        .bind(evidence_auto_capture)
        .bind(response_fallback_enabled)
        .bind(memory_soft_anchor)
        .bind(context_extraction_boost)
        .bind(planner_enabled)
        .bind(confidence_calibration)
        .bind(scheduler_cognition)
        .bind(learning_feedback)
        .bind(evidence_semantics_v2)
        .bind(narrative_continuity)
        .bind(monologue_provenance_guard)
        .bind(organism_decay)
        .bind(model_context_limit)
        .bind(introspection_confidence_threshold)
        .bind(introspection_drift_threshold)
        .bind(introspection_ambiguity_threshold)
        .bind(enable_attribution_gate)
        .bind(enable_user_utterance_evidence)
        .bind(enable_attribution_metadata)
        .bind(enable_tool_schema_validation)
        .bind(enable_context_evidence)
        .bind(enable_monologue_validator)
        .bind(enable_memory_evidence_gating)
        .bind(enable_speculative_workspace_containment)
        .bind(stability_prompt_override_guard)
        .bind(stability_monologue_tagged)
        .bind(stability_introspection_structured)
        .bind(stability_disable_working_hypothesis)
        .bind(stability_state_disclosure_expanded)
        .bind(stability_transcript_normalization)
        .bind(stability_memory_hygiene)
        .bind(stability_non_stream_sanitization)
        .execute(&self.pool)
        .await
        {
            eprintln!("[DB] update_settings failed: {}", e);
            return Err(Box::new(e));
        }
        let mut weight_changes: Vec<serde_json::Value> = Vec::new();
        let prev_user = current.weight_user_satisfaction.unwrap_or(0.5);
        if (prev_user - weight_user_satisfaction).abs() > 0.0001 {
            weight_changes.push(json!({
                "key": "weight_user_satisfaction",
                "prev": prev_user,
                "next": weight_user_satisfaction
            }));
        }
        let prev_policy = current.weight_policy_rigor.unwrap_or(0.5);
        if (prev_policy - weight_policy_rigor).abs() > 0.0001 {
            weight_changes.push(json!({
                "key": "weight_policy_rigor",
                "prev": prev_policy,
                "next": weight_policy_rigor
            }));
        }
        let prev_latency = current.weight_latency.unwrap_or(0.5);
        if (prev_latency - weight_latency).abs() > 0.0001 {
            weight_changes.push(json!({
                "key": "weight_latency",
                "prev": prev_latency,
                "next": weight_latency
            }));
        }
        let prev_evidence = current.weight_evidence_strictness.unwrap_or(0.5);
        if (prev_evidence - weight_evidence_strictness).abs() > 0.0001 {
            weight_changes.push(json!({
                "key": "weight_evidence_strictness",
                "prev": prev_evidence,
                "next": weight_evidence_strictness
            }));
        }
        let prev_explore = current.weight_exploration.unwrap_or(0.5);
        if (prev_explore - weight_exploration).abs() > 0.0001 {
            weight_changes.push(json!({
                "key": "weight_exploration",
                "prev": prev_explore,
                "next": weight_exploration
            }));
        }
        if !weight_changes.is_empty() {
            let _ = system_log::log_event(
                &self.pool,
                None,
                "info",
                "settings",
                None,
                None,
                json!({
                    "event": "settings_weight_changed",
                    "changes": weight_changes,
                }),
            )
            .await;
        }
        let user_name = settings.user_display_name.as_deref().unwrap_or("User");
        let assistant_name = settings.assistant_display_name.as_deref().unwrap_or("Ergo");
        if let Err(e) = ensure_primary_entities(&self.pool, user_name, assistant_name).await {
            eprintln!(
                "[DB] ensure_primary_entities failed (user='{}', assistant='{}'): {}",
                user_name, assistant_name, e
            );
            return Err(e);
        }
        Ok(())
    }

    pub async fn get_phi_consent_scope(&self, conversation_id: &str) -> Result<Option<bool>, String> {
        let enabled: Option<i64> = sqlx::query_scalar(
            "SELECT enabled FROM phi_consent_scopes WHERE conversation_id = ?",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(enabled.map(|val| val != 0))
    }

    pub async fn set_phi_consent_scope(&self, conversation_id: &str, enabled: bool) -> Result<(), String> {
        let enabled_val = if enabled { 1 } else { 0 };
        sqlx::query(
            "INSERT INTO phi_consent_scopes (conversation_id, enabled, updated_at)
             VALUES (?, ?, CURRENT_TIMESTAMP)
             ON CONFLICT(conversation_id) DO UPDATE SET
                enabled = excluded.enabled,
                updated_at = CURRENT_TIMESTAMP",
        )
        .bind(conversation_id)
        .bind(enabled_val)
        .execute(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub async fn upsert_context_tag(
        &self,
        conversation_id: &str,
        tag: &str,
        confidence: f32,
        inferred: bool,
        evidence_event_ids: &[i64],
        source: Option<&str>,
    ) -> Result<(), String> {
        let tag_id = Uuid::new_v4().to_string();
        let evidence_json = serde_json::to_string(evidence_event_ids).unwrap_or_else(|_| "[]".to_string());
        let inferred_val = if inferred { 1 } else { 0 };
        sqlx::query(
            "INSERT INTO context_tags
             (tag_id, conversation_id, tag, confidence, inferred, evidence_event_ids, source, last_seen_at, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
             ON CONFLICT(conversation_id, tag)
             DO UPDATE SET confidence = excluded.confidence,
                           inferred = excluded.inferred,
                           evidence_event_ids = excluded.evidence_event_ids,
                           source = excluded.source,
                           last_seen_at = CURRENT_TIMESTAMP",
        )
        .bind(tag_id)
        .bind(conversation_id)
        .bind(tag)
        .bind(confidence)
        .bind(inferred_val)
        .bind(evidence_json)
        .bind(source.unwrap_or(""))
        .execute(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub async fn get_context_tags(
        &self,
        conversation_id: &str,
        ttl_minutes: i64,
    ) -> Vec<ContextTagEntry> {
        let window = format!("-{} minutes", ttl_minutes.max(1));
        let rows = sqlx::query(
            "SELECT tag, confidence, inferred, evidence_event_ids, last_seen_at, source
             FROM context_tags
             WHERE conversation_id = ?
               AND datetime(last_seen_at) >= datetime('now', ?)
             ORDER BY datetime(last_seen_at) DESC",
        )
        .bind(conversation_id)
        .bind(window)
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();
        rows.into_iter()
            .map(|row| {
                let raw: String = row.get("evidence_event_ids");
                let evidence_event_ids = serde_json::from_str::<Vec<i64>>(&raw).unwrap_or_default();
                ContextTagEntry {
                    tag: row.get("tag"),
                    confidence: row.get::<f64, _>("confidence") as f32,
                    inferred: row.get::<i32, _>("inferred") != 0,
                    evidence_event_ids,
                    last_seen_at: row.get("last_seen_at"),
                    source: row.try_get("source").ok(),
                }
            })
            .collect()
    }

    pub async fn get_user_intent_summary(&self, conversation_id: &str) -> Option<UserIntentSummary> {
        let row = sqlx::query(
            "SELECT summary, confirmed, evidence_event_ids, updated_at
             FROM user_intent_summaries
             WHERE conversation_id = ?
             ORDER BY datetime(updated_at) DESC
             LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten()?;
        let raw: String = row.get("evidence_event_ids");
        let evidence_event_ids = serde_json::from_str::<Vec<i64>>(&raw).unwrap_or_default();
        Some(UserIntentSummary {
            summary: row.get("summary"),
            confirmed: row.get::<i32, _>("confirmed") != 0,
            evidence_event_ids,
            updated_at: row.get("updated_at"),
        })
    }

    pub async fn upsert_user_intent_summary(
        &self,
        conversation_id: &str,
        summary: &str,
        confirmed: bool,
        evidence_event_ids: &[i64],
    ) -> Result<(), String> {
        let summary_id = Uuid::new_v4().to_string();
        let evidence_json = serde_json::to_string(evidence_event_ids).unwrap_or_else(|_| "[]".to_string());
        let confirmed_val = if confirmed { 1 } else { 0 };
        sqlx::query(
            "INSERT INTO user_intent_summaries
             (summary_id, conversation_id, summary, confirmed, evidence_event_ids, updated_at)
             VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
             ON CONFLICT(conversation_id)
             DO UPDATE SET summary = excluded.summary,
                           confirmed = excluded.confirmed,
                           evidence_event_ids = excluded.evidence_event_ids,
                           updated_at = CURRENT_TIMESTAMP",
        )
        .bind(summary_id)
        .bind(conversation_id)
        .bind(summary)
        .bind(confirmed_val)
        .bind(evidence_json)
        .execute(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    async fn ensure_settings_columns(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let pool = &self.pool;
        if !column_exists(pool, "settings", "system_prompt").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN system_prompt TEXT").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "user_display_name").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN user_display_name TEXT").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "json_reliable_model_id").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN json_reliable_model_id TEXT").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "assistant_display_name").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN assistant_display_name TEXT").execute(pool).await?;
        }
        let mut onboarding_added = false;
        if !column_exists(pool, "settings", "onboarding_completed").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN onboarding_completed BOOLEAN NOT NULL DEFAULT 0").execute(pool).await?;
            onboarding_added = true;
        }
        if onboarding_added {
            let _ = sqlx::query("UPDATE settings SET onboarding_completed = 1 WHERE onboarding_completed IS NULL OR onboarding_completed = 0")
                .execute(pool)
                .await;
        }
        if !column_exists(pool, "settings", "ui_theme").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN ui_theme TEXT DEFAULT 'builtin:utopia'").execute(pool).await?;
        }
        let _ = sqlx::query("UPDATE settings SET ui_theme = 'builtin:utopia' WHERE ui_theme IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET voice_name = 'bf_isabella' WHERE voice_name IS NULL OR voice_name = ''")
            .execute(pool)
            .await;
        if !column_exists(pool, "settings", "summarization_api_url").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN summarization_api_url TEXT").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "summarization_model").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN summarization_model TEXT").execute(pool).await?;
        }
        let _ = sqlx::query(
            "UPDATE settings
             SET summarization_model = 'summarizer'
             WHERE summarization_model IS NULL OR summarization_model = ''",
        )
        .execute(pool)
        .await;
        let _ = sqlx::query(
            "UPDATE settings
             SET summarization_api_url = api_base_url
             WHERE summarization_api_url IS NULL OR summarization_api_url = ''",
        )
        .execute(pool)
        .await;
        if !column_exists(pool, "settings", "embedding_model").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN embedding_model TEXT").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "voice_reference_audio").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN voice_reference_audio TEXT").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "voice_reference_text").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN voice_reference_text TEXT").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "voice_quality_preset").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN voice_quality_preset TEXT").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "voice_cfg_value").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN voice_cfg_value REAL").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "voice_denoiser_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN voice_denoiser_enabled BOOLEAN").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "voice_temperature").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN voice_temperature REAL").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "voice_pitch_semitones").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN voice_pitch_semitones REAL DEFAULT 1").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "voice_reverb_amount").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN voice_reverb_amount REAL DEFAULT 0.15").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "voice_compression").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN voice_compression REAL DEFAULT 0.05").execute(pool).await?;
        }
        let _ = sqlx::query("UPDATE settings SET voice_pitch_semitones = 1 WHERE voice_pitch_semitones IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET voice_reverb_amount = 0.15 WHERE voice_reverb_amount IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET voice_compression = 0.05 WHERE voice_compression IS NULL")
            .execute(pool)
            .await;
        if !column_exists(pool, "settings", "voice_formant_shift").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN voice_formant_shift REAL").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "trace_history_limit").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN trace_history_limit INTEGER DEFAULT 10").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "cockpit_write_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN cockpit_write_enabled BOOLEAN NOT NULL DEFAULT 0").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "episodic_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN episodic_enabled BOOLEAN NOT NULL DEFAULT 1").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "episodic_injection_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN episodic_injection_enabled BOOLEAN NOT NULL DEFAULT 1").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "episodic_compaction_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN episodic_compaction_enabled BOOLEAN NOT NULL DEFAULT 1").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "episodic_injection_limit").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN episodic_injection_limit INTEGER NOT NULL DEFAULT 5").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "episodic_opt_out").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN episodic_opt_out BOOLEAN NOT NULL DEFAULT 0").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "memory_claims_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN memory_claims_enabled BOOLEAN NOT NULL DEFAULT 1").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "phi_consent").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN phi_consent BOOLEAN NOT NULL DEFAULT 0").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "seed_personal_user").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN seed_personal_user BOOLEAN NOT NULL DEFAULT 1").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "lexical_fallback_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN lexical_fallback_enabled BOOLEAN NOT NULL DEFAULT 1").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "memory_half_life_hours").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN memory_half_life_hours REAL NOT NULL DEFAULT 168").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "research_budget_per_hour").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN research_budget_per_hour INTEGER NOT NULL DEFAULT 6").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "research_budget_reset_window").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN research_budget_reset_window INTEGER NOT NULL DEFAULT 60").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "research_cost_per_call").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN research_cost_per_call INTEGER NOT NULL DEFAULT 1").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "monologue_interval_seconds").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN monologue_interval_seconds INTEGER NOT NULL DEFAULT 20").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "monologue_timeout_secs").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN monologue_timeout_secs INTEGER NOT NULL DEFAULT 75").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "monologue_retry_timeout_secs").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN monologue_retry_timeout_secs INTEGER NOT NULL DEFAULT 25").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "empty_response_retry_max").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN empty_response_retry_max INTEGER NOT NULL DEFAULT 3").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "empty_response_retry_timeout_ms").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN empty_response_retry_timeout_ms INTEGER NOT NULL DEFAULT 4000").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "monologue_max_per_hour").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN monologue_max_per_hour INTEGER NOT NULL DEFAULT 360").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "thread_max_depth").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN thread_max_depth INTEGER NOT NULL DEFAULT 4").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "allow_shell_tool").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN allow_shell_tool BOOLEAN NOT NULL DEFAULT 0").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "shell_command_allowlist").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN shell_command_allowlist TEXT").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "ask_budget_max").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN ask_budget_max INTEGER NOT NULL DEFAULT 6").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "calculator_followups_max").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN calculator_followups_max INTEGER NOT NULL DEFAULT 0").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "loop_similarity_threshold").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN loop_similarity_threshold REAL NOT NULL DEFAULT 0.90").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "loop_recent_k").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN loop_recent_k INTEGER NOT NULL DEFAULT 6").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "meta_cog_outcome_turns").await {
            sqlx::query(
                "ALTER TABLE settings ADD COLUMN meta_cog_outcome_turns INTEGER NOT NULL DEFAULT 3",
            )
            .execute(pool)
            .await?;
        }
        if !column_exists(pool, "settings", "meta_cog_cycle_window_turns").await {
            sqlx::query(
                "ALTER TABLE settings ADD COLUMN meta_cog_cycle_window_turns INTEGER NOT NULL DEFAULT 2",
            )
            .execute(pool)
            .await?;
        }
        if !column_exists(pool, "settings", "meta_cog_outcome_timeout_s").await {
            sqlx::query(
                "ALTER TABLE settings ADD COLUMN meta_cog_outcome_timeout_s INTEGER NOT NULL DEFAULT 120",
            )
            .execute(pool)
            .await?;
        }
        if !column_exists(pool, "settings", "meta_cog_cooldown_s").await {
            sqlx::query(
                "ALTER TABLE settings ADD COLUMN meta_cog_cooldown_s INTEGER NOT NULL DEFAULT 60",
            )
            .execute(pool)
            .await?;
        }
        if !column_exists(pool, "settings", "meta_cog_streak_limit").await {
            sqlx::query(
                "ALTER TABLE settings ADD COLUMN meta_cog_streak_limit INTEGER NOT NULL DEFAULT 3",
            )
            .execute(pool)
            .await?;
        }
        if !column_exists(pool, "settings", "registry_profile_name").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN registry_profile_name TEXT DEFAULT 'default'").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "controller_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN controller_enabled BOOLEAN NOT NULL DEFAULT 1").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "monologue_stabilization_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN monologue_stabilization_enabled BOOLEAN NOT NULL DEFAULT 1").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "monologue_surface_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN monologue_surface_enabled BOOLEAN NOT NULL DEFAULT 1").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "show_monologue_in_chat").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN show_monologue_in_chat BOOLEAN NOT NULL DEFAULT 1").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "enable_introspection").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN enable_introspection BOOLEAN NOT NULL DEFAULT 1").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "heartbeat_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN heartbeat_enabled BOOLEAN NOT NULL DEFAULT 1").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "dream_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN dream_enabled BOOLEAN NOT NULL DEFAULT 1").execute(pool).await?;
        }
        if !column_exists(pool, "settings", "binding_enforcement_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN binding_enforcement_enabled BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "pending_prompt_alignment_enabled").await {
            sqlx::query(
                "ALTER TABLE settings ADD COLUMN pending_prompt_alignment_enabled BOOLEAN NOT NULL DEFAULT 1",
            )
            .execute(pool)
            .await?;
        }
        if !column_exists(pool, "settings", "pending_prompt_recency_secs").await {
            sqlx::query(
                "ALTER TABLE settings ADD COLUMN pending_prompt_recency_secs INTEGER NOT NULL DEFAULT 180",
            )
            .execute(pool)
            .await?;
        }
        if !column_exists(pool, "settings", "auto_memory_pass_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN auto_memory_pass_enabled BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "summary_cohesion_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN summary_cohesion_enabled BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "compact_prompt_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN compact_prompt_enabled BOOLEAN NOT NULL DEFAULT 0")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "context_hydration_mode").await {
            sqlx::query(
                "ALTER TABLE settings ADD COLUMN context_hydration_mode TEXT NOT NULL DEFAULT 'shadow'",
            )
            .execute(pool)
            .await?;
        }
        if !column_exists(pool, "settings", "context_budgeter_enabled").await {
            sqlx::query(
                "ALTER TABLE settings ADD COLUMN context_budgeter_enabled BOOLEAN NOT NULL DEFAULT 1",
            )
            .execute(pool)
            .await?;
        }
        if !column_exists(pool, "settings", "context_miss_detector_enabled").await {
            sqlx::query(
                "ALTER TABLE settings ADD COLUMN context_miss_detector_enabled BOOLEAN NOT NULL DEFAULT 1",
            )
            .execute(pool)
            .await?;
        }
        if !column_exists(pool, "settings", "world_model_reconcile_mode").await {
            sqlx::query(
                "ALTER TABLE settings ADD COLUMN world_model_reconcile_mode TEXT NOT NULL DEFAULT 'shadow'",
            )
            .execute(pool)
            .await?;
        }
        if !column_exists(pool, "settings", "goal_loop_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN goal_loop_enabled BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "goal_loop_interval_turns").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN goal_loop_interval_turns INTEGER NOT NULL DEFAULT 3")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "goal_loop_load_threshold_ms").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN goal_loop_load_threshold_ms INTEGER NOT NULL DEFAULT 650")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "world_model_reconcile_mode").await {
            sqlx::query(
                "ALTER TABLE settings ADD COLUMN world_model_reconcile_mode TEXT NOT NULL DEFAULT 'shadow'",
            )
            .execute(pool)
            .await?;
        }
        if !column_exists(pool, "settings", "goal_loop_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN goal_loop_enabled BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "goal_loop_interval_turns").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN goal_loop_interval_turns INTEGER NOT NULL DEFAULT 3")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "goal_loop_load_threshold_ms").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN goal_loop_load_threshold_ms INTEGER NOT NULL DEFAULT 650")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "json_only_disabled_models").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN json_only_disabled_models TEXT")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "tool_failure_gate_window_mins").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN tool_failure_gate_window_mins INTEGER")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "tool_failure_gate_tool_names").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN tool_failure_gate_tool_names TEXT")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "gate_default_soft").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN gate_default_soft BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "gate_shadow_mode").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN gate_shadow_mode BOOLEAN NOT NULL DEFAULT 0")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "gate_rollout_percent").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN gate_rollout_percent INTEGER NOT NULL DEFAULT 100")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "self_report_channel").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN self_report_channel BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "self_awareness_expression_mode").await {
            sqlx::query(
                "ALTER TABLE settings ADD COLUMN self_awareness_expression_mode TEXT NOT NULL DEFAULT 'balanced'",
            )
            .execute(pool)
            .await?;
        }
        if !column_exists(pool, "settings", "explicit_feedback_only").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN explicit_feedback_only BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "weight_user_satisfaction").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN weight_user_satisfaction REAL NOT NULL DEFAULT 0.5")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "weight_policy_rigor").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN weight_policy_rigor REAL NOT NULL DEFAULT 0.5")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "weight_latency").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN weight_latency REAL NOT NULL DEFAULT 0.5")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "weight_evidence_strictness").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN weight_evidence_strictness REAL NOT NULL DEFAULT 0.5")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "weight_exploration").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN weight_exploration REAL NOT NULL DEFAULT 0.5")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "evidence_emit_budget").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN evidence_emit_budget INTEGER NOT NULL DEFAULT 50")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "evidence_retention_days").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN evidence_retention_days INTEGER NOT NULL DEFAULT 30")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "gate_penalty_integration").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN gate_penalty_integration BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "evidence_auto_capture").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN evidence_auto_capture BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "response_fallback_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN response_fallback_enabled BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "memory_soft_anchor").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN memory_soft_anchor BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "context_extraction_boost").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN context_extraction_boost BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        let _ = sqlx::query(
            "UPDATE settings
             SET response_fallback_enabled = 1
             WHERE response_fallback_enabled IS NULL",
        )
        .execute(pool)
        .await;
        let _ = sqlx::query(
            "UPDATE settings
             SET memory_soft_anchor = 1
             WHERE memory_soft_anchor IS NULL",
        )
        .execute(pool)
        .await;
        let _ = sqlx::query(
            "UPDATE settings
             SET context_extraction_boost = 1
             WHERE context_extraction_boost IS NULL",
        )
        .execute(pool)
        .await;
        if !column_exists(pool, "settings", "planner_enabled").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN planner_enabled BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "confidence_calibration").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN confidence_calibration BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "scheduler_cognition").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN scheduler_cognition BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "learning_feedback").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN learning_feedback BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "evidence_semantics_v2").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN evidence_semantics_v2 BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "narrative_continuity").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN narrative_continuity BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "monologue_provenance_guard").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN monologue_provenance_guard BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "organism_decay").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN organism_decay BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "model_context_limit").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN model_context_limit INTEGER NOT NULL DEFAULT 16384")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "introspection_confidence_threshold").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN introspection_confidence_threshold REAL NOT NULL DEFAULT 0.5")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "introspection_drift_threshold").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN introspection_drift_threshold REAL NOT NULL DEFAULT 0.6")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "introspection_ambiguity_threshold").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN introspection_ambiguity_threshold REAL NOT NULL DEFAULT 0.5")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "enable_attribution_gate").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN enable_attribution_gate BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "enable_user_utterance_evidence").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN enable_user_utterance_evidence BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "enable_attribution_metadata").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN enable_attribution_metadata BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "enable_tool_schema_validation").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN enable_tool_schema_validation BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "enable_context_evidence").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN enable_context_evidence BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "enable_monologue_validator").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN enable_monologue_validator BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "enable_memory_evidence_gating").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN enable_memory_evidence_gating BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "enable_speculative_workspace_containment").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN enable_speculative_workspace_containment BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "stability_prompt_override_guard").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN stability_prompt_override_guard BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "stability_monologue_tagged").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN stability_monologue_tagged BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "stability_introspection_structured").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN stability_introspection_structured BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "stability_disable_working_hypothesis").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN stability_disable_working_hypothesis BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "stability_state_disclosure_expanded").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN stability_state_disclosure_expanded BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "stability_transcript_normalization").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN stability_transcript_normalization BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "stability_memory_hygiene").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN stability_memory_hygiene BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "settings", "stability_non_stream_sanitization").await {
            sqlx::query("ALTER TABLE settings ADD COLUMN stability_non_stream_sanitization BOOLEAN NOT NULL DEFAULT 1")
                .execute(pool)
                .await?;
        }
        if !column_exists(pool, "action_proposals", "plan_hash").await {
            let _ = sqlx::query("ALTER TABLE action_proposals ADD COLUMN plan_hash TEXT NOT NULL DEFAULT ''")
                .execute(pool)
                .await;
        }
        if !column_exists(pool, "action_proposals", "plan_state").await {
            let _ = sqlx::query("ALTER TABLE action_proposals ADD COLUMN plan_state TEXT NOT NULL DEFAULT 'draft'")
                .execute(pool)
                .await;
        }
        if !column_exists(pool, "tool_dispatches", "plan_step_id").await {
            let _ = sqlx::query("ALTER TABLE tool_dispatches ADD COLUMN plan_step_id TEXT")
                .execute(pool)
                .await;
        }
        let _ = sqlx::query("UPDATE settings SET allow_shell_tool = 0 WHERE allow_shell_tool IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET monologue_surface_enabled = 1 WHERE monologue_surface_enabled IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET show_monologue_in_chat = 1 WHERE show_monologue_in_chat IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET enable_introspection = 1 WHERE enable_introspection IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET seed_personal_user = 1 WHERE seed_personal_user IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET lexical_fallback_enabled = 1 WHERE lexical_fallback_enabled IS NULL OR lexical_fallback_enabled = 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET memory_half_life_hours = 168 WHERE memory_half_life_hours IS NULL OR memory_half_life_hours <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET phi_consent = 0 WHERE phi_consent IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET research_budget_per_hour = 6 WHERE research_budget_per_hour IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET research_budget_reset_window = 60 WHERE research_budget_reset_window IS NULL OR research_budget_reset_window <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET research_cost_per_call = 1 WHERE research_cost_per_call IS NULL OR research_cost_per_call <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET monologue_interval_seconds = 20 WHERE monologue_interval_seconds IS NULL OR monologue_interval_seconds <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET monologue_timeout_secs = 75 WHERE monologue_timeout_secs IS NULL OR monologue_timeout_secs <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET monologue_retry_timeout_secs = 25 WHERE monologue_retry_timeout_secs IS NULL OR monologue_retry_timeout_secs <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET monologue_max_per_hour = 360 WHERE monologue_max_per_hour IS NULL OR monologue_max_per_hour <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET monologue_interval_seconds = 20 WHERE monologue_interval_seconds = 60")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET monologue_max_per_hour = 360 WHERE monologue_max_per_hour = 12")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET thread_max_depth = 4 WHERE thread_max_depth IS NULL OR thread_max_depth <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET enable_attribution_gate = 1 WHERE enable_attribution_gate IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET enable_user_utterance_evidence = 1 WHERE enable_user_utterance_evidence IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET enable_attribution_metadata = 1 WHERE enable_attribution_metadata IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET enable_tool_schema_validation = 1 WHERE enable_tool_schema_validation IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET stability_prompt_override_guard = 1 WHERE stability_prompt_override_guard IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET stability_monologue_tagged = 1 WHERE stability_monologue_tagged IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET stability_introspection_structured = 1 WHERE stability_introspection_structured IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET stability_disable_working_hypothesis = 1 WHERE stability_disable_working_hypothesis IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET stability_state_disclosure_expanded = 1 WHERE stability_state_disclosure_expanded IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET stability_transcript_normalization = 1 WHERE stability_transcript_normalization IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query(
            "UPDATE settings SET context_hydration_mode = 'shadow' WHERE context_hydration_mode IS NULL OR context_hydration_mode = ''",
        )
        .execute(pool)
        .await;
        let _ = sqlx::query("UPDATE settings SET context_budgeter_enabled = 1 WHERE context_budgeter_enabled IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query(
            "UPDATE settings SET context_miss_detector_enabled = 1 WHERE context_miss_detector_enabled IS NULL",
        )
        .execute(pool)
        .await;
        let _ = sqlx::query(
            "UPDATE settings SET world_model_reconcile_mode = 'shadow' WHERE world_model_reconcile_mode IS NULL OR world_model_reconcile_mode = ''",
        )
        .execute(pool)
        .await;
        let _ = sqlx::query("UPDATE settings SET goal_loop_enabled = 1 WHERE goal_loop_enabled IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET goal_loop_interval_turns = 3 WHERE goal_loop_interval_turns IS NULL OR goal_loop_interval_turns <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET goal_loop_load_threshold_ms = 650 WHERE goal_loop_load_threshold_ms IS NULL OR goal_loop_load_threshold_ms <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET stability_memory_hygiene = 1 WHERE stability_memory_hygiene IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET stability_non_stream_sanitization = 1 WHERE stability_non_stream_sanitization IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET enable_context_evidence = 1 WHERE enable_context_evidence IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET enable_monologue_validator = 1 WHERE enable_monologue_validator IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET enable_memory_evidence_gating = 1 WHERE enable_memory_evidence_gating IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET enable_speculative_workspace_containment = 1 WHERE enable_speculative_workspace_containment IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET ask_budget_max = 6 WHERE ask_budget_max IS NULL OR ask_budget_max <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET calculator_followups_max = 0 WHERE calculator_followups_max IS NULL OR calculator_followups_max < 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET loop_similarity_threshold = 0.90 WHERE loop_similarity_threshold IS NULL OR loop_similarity_threshold <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET loop_recent_k = 6 WHERE loop_recent_k IS NULL OR loop_recent_k <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET meta_cog_outcome_turns = 3 WHERE meta_cog_outcome_turns IS NULL OR meta_cog_outcome_turns <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET meta_cog_cycle_window_turns = 2 WHERE meta_cog_cycle_window_turns IS NULL OR meta_cog_cycle_window_turns <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET meta_cog_outcome_timeout_s = 120 WHERE meta_cog_outcome_timeout_s IS NULL OR meta_cog_outcome_timeout_s <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET meta_cog_cooldown_s = 60 WHERE meta_cog_cooldown_s IS NULL OR meta_cog_cooldown_s <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET meta_cog_streak_limit = 3 WHERE meta_cog_streak_limit IS NULL OR meta_cog_streak_limit <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET registry_profile_name = 'default' WHERE registry_profile_name IS NULL OR registry_profile_name = ''")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET controller_enabled = 1 WHERE controller_enabled IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET monologue_stabilization_enabled = 1 WHERE monologue_stabilization_enabled IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET heartbeat_enabled = 1 WHERE heartbeat_enabled IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET dream_enabled = 1 WHERE dream_enabled IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET binding_enforcement_enabled = 1 WHERE binding_enforcement_enabled IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET pending_prompt_alignment_enabled = 1 WHERE pending_prompt_alignment_enabled IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET pending_prompt_recency_secs = 180 WHERE pending_prompt_recency_secs IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET auto_memory_pass_enabled = 1 WHERE auto_memory_pass_enabled IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET summary_cohesion_enabled = 1 WHERE summary_cohesion_enabled IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET compact_prompt_enabled = 0 WHERE compact_prompt_enabled IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET gate_default_soft = 1 WHERE gate_default_soft IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET gate_shadow_mode = 0 WHERE gate_shadow_mode IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET gate_rollout_percent = 100 WHERE gate_rollout_percent IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET self_report_channel = 1 WHERE self_report_channel IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query(
            "UPDATE settings SET self_awareness_expression_mode = 'balanced' WHERE self_awareness_expression_mode IS NULL OR self_awareness_expression_mode = ''",
        )
        .execute(pool)
        .await;
        let _ = sqlx::query("UPDATE settings SET explicit_feedback_only = 1 WHERE explicit_feedback_only IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET weight_user_satisfaction = 0.5 WHERE weight_user_satisfaction IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET weight_policy_rigor = 0.5 WHERE weight_policy_rigor IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET weight_latency = 0.5 WHERE weight_latency IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET weight_evidence_strictness = 0.5 WHERE weight_evidence_strictness IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET weight_exploration = 0.5 WHERE weight_exploration IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET evidence_emit_budget = 50 WHERE evidence_emit_budget IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET evidence_retention_days = 30 WHERE evidence_retention_days IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET gate_penalty_integration = 1 WHERE gate_penalty_integration IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET evidence_auto_capture = 1 WHERE evidence_auto_capture IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET response_fallback_enabled = 1 WHERE response_fallback_enabled IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET memory_soft_anchor = 1 WHERE memory_soft_anchor IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET context_extraction_boost = 1 WHERE context_extraction_boost IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET planner_enabled = 1 WHERE planner_enabled IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET confidence_calibration = 1 WHERE confidence_calibration IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET scheduler_cognition = 1 WHERE scheduler_cognition IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET learning_feedback = 1 WHERE learning_feedback IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET evidence_semantics_v2 = 1 WHERE evidence_semantics_v2 IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET narrative_continuity = 1 WHERE narrative_continuity IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET monologue_provenance_guard = 1 WHERE monologue_provenance_guard IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET organism_decay = 1 WHERE organism_decay IS NULL")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET model_context_limit = 16384 WHERE model_context_limit IS NULL OR model_context_limit <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET model_context_limit = 2048 WHERE model_context_limit < 2048")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET introspection_confidence_threshold = 0.5 WHERE introspection_confidence_threshold IS NULL OR introspection_confidence_threshold <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET introspection_drift_threshold = 0.6 WHERE introspection_drift_threshold IS NULL OR introspection_drift_threshold <= 0")
            .execute(pool)
            .await;
        let _ = sqlx::query("UPDATE settings SET introspection_ambiguity_threshold = 0.5 WHERE introspection_ambiguity_threshold IS NULL OR introspection_ambiguity_threshold <= 0")
            .execute(pool)
            .await;
        if !column_exists(pool, "ics_evidence_events", "context_json").await {
            let _ = sqlx::query("ALTER TABLE ics_evidence_events ADD COLUMN context_json TEXT")
                .execute(pool)
                .await;
        }
        if !column_exists(pool, "ics_evidence_events", "strength").await {
            let _ = sqlx::query("ALTER TABLE ics_evidence_events ADD COLUMN strength REAL NOT NULL DEFAULT 0.0")
                .execute(pool)
                .await;
        }
        Ok(())
    }

    pub async fn clear_messages(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Nuke all messages and runs to ensure clean slate (Single Tenant for now)
        // Must delete artifacts first to satisfy FK constraints
        sqlx::query("DELETE FROM artifacts")
            .execute(&self.pool)
            .await?;
        sqlx::query("DELETE FROM messages")
            .execute(&self.pool)
            .await?;
        sqlx::query("DELETE FROM runs")
            .execute(&self.pool)
            .await?;
        sqlx::query("DELETE FROM conversation_summaries")
            .execute(&self.pool)
            .await?;
        sqlx::query("DELETE FROM conversation_live_summaries")
            .execute(&self.pool)
            .await?;
        sqlx::query("DELETE FROM conversation_weekly_summaries")
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Clears conversation context while preserving long-term memory and settings.
    pub async fn reset_conversation_data(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Core conversation artifacts
        sqlx::query("DELETE FROM artifacts").execute(&self.pool).await?;
        sqlx::query("DELETE FROM messages").execute(&self.pool).await?;
        sqlx::query("DELETE FROM runs").execute(&self.pool).await?;
        sqlx::query("DELETE FROM conversation_summaries").execute(&self.pool).await?;
        sqlx::query("DELETE FROM conversation_live_summaries").execute(&self.pool).await?;
        sqlx::query("DELETE FROM conversation_weekly_summaries").execute(&self.pool).await?;
        let _ = sqlx::query("DELETE FROM inner_summaries").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM kernel_states").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM inner_monologue_candidates").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM inner_monologue_entries").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM tool_dispatches").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM thread_runs").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM strategy_traces").execute(&self.pool).await;

        // Context-only episodic + clarification state
        let _ = sqlx::query("DELETE FROM episodic_events WHERE conversation_id IS NOT NULL").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_pending_clarify WHERE session_id = 'default'").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_session_bindings WHERE session_id = 'default' AND ref_text NOT IN ('user', 'assistant')").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_working_set").execute(&self.pool).await;

        // Reset conversations (keep the default one)
        sqlx::query("DELETE FROM conversations WHERE conversation_id != 'default'")
            .execute(&self.pool)
            .await?;
        sqlx::query("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id = 'default'")
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn rollback_run(&self, run_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let _ = sqlx::query("DELETE FROM artifacts WHERE run_id = ?")
            .bind(run_id)
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("DELETE FROM messages WHERE run_id = ?")
            .bind(run_id)
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("DELETE FROM runs WHERE run_id = ?")
            .bind(run_id)
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("DELETE FROM episodic_events WHERE run_id = ?")
            .bind(run_id)
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("DELETE FROM tool_dispatches WHERE run_id = ?")
            .bind(run_id)
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("DELETE FROM thread_runs WHERE parent_run_id = ?")
            .bind(run_id)
            .execute(&self.pool)
            .await;
        let _ = sqlx::query("DELETE FROM strategy_traces WHERE run_id = ?")
            .bind(run_id)
            .execute(&self.pool)
            .await;
        Ok(())
    }

    pub async fn mark_run_cancelled(
        &self,
        run_id: &str,
        reason: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let _ = sqlx::query("UPDATE runs SET status = 'cancelled', ended_at = ? WHERE run_id = ?")
            .bind(Utc::now())
            .bind(run_id)
            .execute(&self.pool)
            .await;

        let fallback = "Notice: response cancelled.";
        let _ = sqlx::query(
            "UPDATE messages
             SET status = 'cancelled',
                 error = COALESCE(error, ?),
                 content = CASE
                     WHEN role = 'assistant' AND (content IS NULL OR trim(content) = '') THEN ?
                     ELSE content
                 END,
                 metadata = CASE
                     WHEN metadata IS NULL OR trim(metadata) = '' THEN json_object('surface', 0)
                     WHEN json_valid(metadata) = 0 THEN json_object('surface', 0)
                     ELSE json_set(metadata, '$.surface', 0)
                 END
             WHERE run_id = ? AND role = 'assistant' AND status != 'complete'",
        )
        .bind(reason)
        .bind(fallback)
        .bind(run_id)
        .execute(&self.pool)
        .await;

        Ok(())
    }

    pub async fn supersede_active_runs(
        &self,
        conversation_id: &str,
        new_run_id: &str,
        reason: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        if conversation_id.trim().is_empty() || new_run_id.trim().is_empty() {
            return Ok(Vec::new());
        }
        let run_ids: Vec<String> = sqlx::query_scalar(
            "SELECT run_id FROM runs
             WHERE conversation_id = ? AND status = 'active' AND run_id != ?",
        )
        .bind(conversation_id)
        .bind(new_run_id)
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();

        if run_ids.is_empty() {
            return Ok(run_ids);
        }

        let now = Utc::now();
        let _ = sqlx::query(
            "UPDATE runs
             SET status = 'superseded',
                 ended_at = ?,
                 superseded_by_run_id = ?
             WHERE conversation_id = ? AND status = 'active' AND run_id != ?",
        )
        .bind(now)
        .bind(new_run_id)
        .bind(conversation_id)
        .bind(new_run_id)
        .execute(&self.pool)
        .await;

        let placeholders = run_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let query = format!(
            "UPDATE messages
             SET status = 'cancelled',
                 error = COALESCE(error, ?)
             WHERE role = 'assistant'
               AND status != 'complete'
               AND run_id IN ({})",
            placeholders
        );
        let mut builder = sqlx::query(&query).bind(reason);
        for run_id in run_ids.iter() {
            builder = builder.bind(run_id);
        }
        let _ = builder.execute(&self.pool).await;

        Ok(run_ids)
    }

    pub async fn has_active_run(&self, conversation_id: &str) -> Result<bool, String> {
        if conversation_id.trim().is_empty() {
            return Ok(false);
        }
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM runs
             WHERE conversation_id = ? AND status = 'active'",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| e.to_string())?
        .unwrap_or(0);
        Ok(count > 0)
    }

    pub async fn get_active_foreground_run(&self, conversation_id: &str) -> Result<Option<String>, String> {
        if conversation_id.trim().is_empty() {
            return Ok(None);
        }
        let run_id: Option<String> = sqlx::query_scalar(
            "SELECT run_id FROM runs
             WHERE conversation_id = ?
               AND status = 'active'
               AND json_extract(metadata, '$.execution_mode') = 'direct'
             ORDER BY datetime(started_at) DESC
             LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(run_id)
    }

    pub async fn enqueue_deferred_emit(
        &self,
        conversation_id: &str,
        emit_kind: &str,
        payload_json: &str,
        source: Option<&str>,
    ) -> Result<String, String> {
        if conversation_id.trim().is_empty() || emit_kind.trim().is_empty() {
            return Err("invalid_deferred_emit".to_string());
        }
        let emit_id = Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT INTO deferred_emits
             (emit_id, conversation_id, emit_kind, payload_json, source, created_at)
             VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(&emit_id)
        .bind(conversation_id)
        .bind(emit_kind)
        .bind(payload_json)
        .bind(source)
        .execute(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(emit_id)
    }

    pub async fn claim_deferred_emit(&self, conversation_id: &str) -> Option<DeferredEmit> {
        if conversation_id.trim().is_empty() {
            return None;
        }
        let mut tx = self.pool.begin().await.ok()?;
        let row = sqlx::query(
            "SELECT emit_id, conversation_id, emit_kind, payload_json, source, created_at
             FROM deferred_emits
             WHERE conversation_id = ?
             ORDER BY datetime(created_at) ASC
             LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&mut *tx)
        .await
        .ok()
        .flatten()?;
        let emit_id: String = row.try_get("emit_id").ok()?;
        let conversation_id: String = row.try_get("conversation_id").unwrap_or_default();
        let emit_kind: String = row.try_get("emit_kind").unwrap_or_default();
        let payload_json: String = row.try_get("payload_json").unwrap_or_else(|_| "{}".to_string());
        let source: Option<String> = row.try_get("source").ok();
        let created_at: String = row.try_get("created_at").unwrap_or_default();

        let deleted = sqlx::query(
            "DELETE FROM deferred_emits WHERE emit_id = ?",
        )
        .bind(&emit_id)
        .execute(&mut *tx)
        .await
        .ok()
        .map(|res| res.rows_affected())
        .unwrap_or(0);
        if deleted == 0 {
            let _ = tx.rollback().await;
            return None;
        }
        let _ = tx.commit().await;

        Some(DeferredEmit {
            emit_id,
            conversation_id,
            emit_kind,
            payload_json,
            source,
            created_at,
        })
    }

    pub async fn touch_run_heartbeat(&self, run_id: &str) -> Result<(), String> {
        if run_id.trim().is_empty() {
            return Ok(());
        }
        sqlx::query(
            "UPDATE runs
             SET heartbeat_at = CURRENT_TIMESTAMP
             WHERE run_id = ? AND status = 'active'",
        )
        .bind(run_id)
        .execute(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub async fn touch_active_run_heartbeat(
        &self,
        conversation_id: &str,
    ) -> Result<Option<String>, String> {
        if conversation_id.trim().is_empty() {
            return Ok(None);
        }
        let run_id: Option<String> = sqlx::query_scalar(
            "SELECT run_id FROM runs
             WHERE conversation_id = ? AND status = 'active'
             ORDER BY datetime(started_at) DESC
             LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| e.to_string())?;

        if let Some(ref id) = run_id {
            let _ = self.touch_run_heartbeat(id).await;
        }
        Ok(run_id)
    }

    pub async fn list_conversation_ids(
        &self,
        max_age_minutes: Option<i64>,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let rows = if let Some(age) = max_age_minutes {
            let cutoff = (Utc::now() - chrono::Duration::minutes(age)).to_rfc3339();
            sqlx::query("SELECT conversation_id FROM conversations WHERE datetime(updated_at) >= datetime(?) ORDER BY datetime(updated_at) DESC")
                .bind(cutoff)
                .fetch_all(&self.pool)
                .await?
        } else {
            sqlx::query("SELECT conversation_id FROM conversations ORDER BY datetime(updated_at) DESC")
                .fetch_all(&self.pool)
                .await?
        };

        let mut ids: Vec<String> = rows
            .into_iter()
            .filter_map(|row| row.try_get::<String, _>("conversation_id").ok())
            .filter(|id| !id.trim().is_empty())
            .collect();
        if ids.is_empty() {
            ids.push("default".to_string());
        }
        Ok(ids)
    }

    pub async fn get_history_for_conversation(&self, conversation_id: &str, window: i32) -> Result<Vec<crate::models::Message>, Box<dyn std::error::Error + Send + Sync>> {
        // B4: History window enforcement
        let rows = sqlx::query(
            "SELECT * FROM (
                SELECT * FROM messages 
                WHERE conversation_id = ? AND role IN ('user', 'assistant', 'system')
                  AND status IN ('complete', 'streaming', 'error', 'cancelled')
                  AND NOT (role = 'assistant' AND run_id IS NULL)
                  AND (
                    metadata IS NULL
                    OR json_extract(metadata, '$.source') IS NULL
                    OR json_extract(metadata, '$.source') != 'monologue'
                    OR json_extract(metadata, '$.surface') = 1
                  )
                ORDER BY created_at DESC 
                LIMIT ?
            ) ORDER BY created_at ASC"
        )
        .bind(conversation_id)
        .bind(window)
        .fetch_all(&self.pool)
        .await?;

        let mut messages = Vec::new();
        use sqlx::Row;
        for row in rows {
            messages.push(crate::models::Message {
                message_id: row.get("message_id"),
                conversation_id: row.get("conversation_id"),
                run_id: row.get("run_id"),
                trace_id: row.get("trace_id"),
                role: row.get("role"),
                content: row.get("content"),
                status: row.get("status"),
                error: row.get::<Option<String>, _>("error").map(|s| serde_json::from_str(&s).unwrap_or_default()),
                created_at: row.get("created_at"),
                metadata: row.get::<Option<String>, _>("metadata").map(|s| serde_json::from_str(&s).unwrap_or_default()),
            });
        }
        Ok(messages)
    }

    pub async fn get_recent_user_evidence_ids(&self, conversation_id: &str, limit: i64) -> Vec<i64> {
        let rows = sqlx::query(
            "SELECT metadata FROM messages
             WHERE conversation_id = ? AND role = 'user' AND metadata IS NOT NULL
             ORDER BY created_at DESC
             LIMIT ?",
        )
        .bind(conversation_id)
        .bind(limit.max(1))
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();

        let mut ids = Vec::new();
        let mut seen = HashSet::new();
        for row in rows {
            let meta: Option<String> = row.try_get("metadata").ok();
            let Some(meta) = meta else { continue; };
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&meta) else { continue; };
            if let Some(array) = value.get("evidence_event_ids").and_then(|v| v.as_array()) {
                for id in array.iter().filter_map(|v| v.as_i64()) {
                    if seen.insert(id) {
                        ids.push(id);
                    }
                }
            } else if let Some(id) = value.get("evidence_event_id").and_then(|v| v.as_i64()) {
                if seen.insert(id) {
                    ids.push(id);
                }
            }
        }
        ids
    }

    pub async fn get_latest_evidence_timestamp(&self, evidence_ids: &[i64]) -> Option<DateTime<Utc>> {
        if evidence_ids.is_empty() {
            return None;
        }
        let mut unique = evidence_ids.to_vec();
        unique.sort();
        unique.dedup();
        let placeholders = unique.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let query = format!(
            "SELECT MAX(created_at) as created_at FROM (
                SELECT created_at FROM ics_evidence_events WHERE id IN ({})
                UNION ALL
                SELECT created_at FROM self_evidence_events WHERE id IN ({})
             )",
            placeholders, placeholders
        );
        let mut builder = sqlx::query_scalar::<_, String>(&query);
        for id in unique.iter() {
            builder = builder.bind(id);
        }
        for id in unique.iter() {
            builder = builder.bind(id);
        }
        let raw: Option<String> = builder.fetch_optional(&self.pool).await.ok().flatten();
        let raw = raw?;
        if let Ok(parsed) = DateTime::parse_from_rfc3339(&raw) {
            return Some(parsed.with_timezone(&Utc));
        }
        if let Ok(parsed) = NaiveDateTime::parse_from_str(&raw, "%Y-%m-%d %H:%M:%S%.f") {
            return Some(DateTime::<Utc>::from_naive_utc_and_offset(parsed, Utc));
        }
        if let Ok(parsed) = NaiveDateTime::parse_from_str(&raw, "%Y-%m-%d %H:%M:%S") {
            return Some(DateTime::<Utc>::from_naive_utc_and_offset(parsed, Utc));
        }
        None
    }

    pub async fn get_recent_evidence_ids_by_source_types(
        &self,
        source_types: &[&str],
        limit: i64,
    ) -> Vec<i64> {
        if source_types.is_empty() {
            return Vec::new();
        }
        let placeholders = source_types.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let query = format!(
            "SELECT id FROM ics_evidence_events
             WHERE source_type IN ({})
             ORDER BY datetime(created_at) DESC
             LIMIT ?",
            placeholders
        );
        let mut builder = sqlx::query(&query);
        for source_type in source_types.iter() {
            builder = builder.bind(source_type);
        }
        builder = builder.bind(limit.max(1));
        let rows = builder.fetch_all(&self.pool).await.unwrap_or_default();
        let mut ids = Vec::new();
        let mut seen = HashSet::new();
        for row in rows {
            let id: i64 = row.try_get("id").unwrap_or(0);
            if id > 0 && seen.insert(id) {
                ids.push(id);
            }
        }
        ids
    }

    pub async fn rank_evidence_ids_by_strength(
        &self,
        source_types: &[&str],
        limit: i64,
    ) -> Vec<i64> {
        let limit = limit.max(1);
        let query = if source_types.is_empty() {
            "SELECT id FROM ics_evidence_events
             ORDER BY (strength + (0.2 / (1.0 + max(0.0, julianday('now') - julianday(created_at))))) DESC,
                      datetime(created_at) DESC
             LIMIT ?"
                .to_string()
        } else {
            let placeholders = source_types.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            format!(
                "SELECT id FROM ics_evidence_events
                 WHERE source_type IN ({})
                 ORDER BY (strength + (0.2 / (1.0 + max(0.0, julianday('now') - julianday(created_at))))) DESC,
                          datetime(created_at) DESC
                 LIMIT ?",
                placeholders
            )
        };
        let mut builder = sqlx::query(&query);
        if !source_types.is_empty() {
            for source_type in source_types.iter() {
                builder = builder.bind(source_type);
            }
        }
        builder = builder.bind(limit);
        let rows = builder.fetch_all(&self.pool).await.unwrap_or_default();
        let mut ids = Vec::new();
        let mut seen = HashSet::new();
        for row in rows {
            let id: i64 = row.try_get("id").unwrap_or(0);
            if id > 0 && seen.insert(id) {
                ids.push(id);
            }
        }
        ids
    }

    pub async fn average_evidence_strength(
        &self,
        lookback_days: i64,
    ) -> Option<f64> {
        let lookback_days = lookback_days.max(1);
        let lookback = format!("-{} days", lookback_days);
        let row: Option<f64> = sqlx::query_scalar(
            "SELECT AVG(strength) FROM ics_evidence_events
             WHERE datetime(created_at) >= datetime('now', ?)",
        )
        .bind(lookback)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();
        row
    }

    pub async fn evidence_quality_stats(
        &self,
        evidence_ids: &[i64],
    ) -> Option<EvidenceQualityStats> {
        if evidence_ids.is_empty() {
            return None;
        }
        let mut unique: Vec<i64> = evidence_ids.iter().copied().filter(|id| *id > 0).collect();
        unique.sort();
        unique.dedup();
        if unique.is_empty() {
            return None;
        }
        let placeholders = unique.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let query_ics = format!(
            "SELECT source_type, weight, strength, max(0.0, julianday('now') - julianday(created_at)) as age_days
             FROM ics_evidence_events WHERE id IN ({})",
            placeholders
        );
        let query_self = format!(
            "SELECT source_type, weight, NULL as strength, max(0.0, julianday('now') - julianday(created_at)) as age_days
             FROM self_evidence_events WHERE id IN ({})",
            placeholders
        );
        let mut ics_stmt = sqlx::query(&query_ics);
        for id in unique.iter() {
            ics_stmt = ics_stmt.bind(id);
        }
        let mut self_stmt = sqlx::query(&query_self);
        for id in unique.iter() {
            self_stmt = self_stmt.bind(id);
        }
        let mut scores: Vec<f32> = Vec::new();
        let rows = ics_stmt.fetch_all(&self.pool).await.unwrap_or_default();
        for row in rows.iter() {
            let source_raw: String = row.try_get("source_type").unwrap_or_default();
            let weight = row.try_get::<f64, _>("weight").unwrap_or(0.0) as f32;
            let strength = row.try_get::<f64, _>("strength").ok().map(|v| v as f32);
            let age_days = row.try_get::<f64, _>("age_days").unwrap_or(0.0) as f32;
            let source = match source_raw.trim().to_lowercase().as_str() {
                "user" | "user_focus" => SourceType::User,
                "tool" => SourceType::Tool,
                "inference" => SourceType::Inference,
                _ => SourceType::System,
            };
            let quality = compute_evidence_quality(source, weight, strength, age_days);
            scores.push(quality);
        }
        let rows = self_stmt.fetch_all(&self.pool).await.unwrap_or_default();
        for row in rows.iter() {
            let source_raw: String = row.try_get("source_type").unwrap_or_default();
            let weight = row.try_get::<f64, _>("weight").unwrap_or(0.0) as f32;
            let age_days = row.try_get::<f64, _>("age_days").unwrap_or(0.0) as f32;
            let source = match source_raw.trim().to_lowercase().as_str() {
                "user" | "user_focus" => SourceType::User,
                "tool" => SourceType::Tool,
                "inference" => SourceType::Inference,
                _ => SourceType::System,
            };
            let quality = compute_evidence_quality(source, weight, None, age_days);
            scores.push(quality);
        }
        if scores.is_empty() {
            return None;
        }
        let mut min = 1.0_f32;
        let mut max = 0.0_f32;
        let mut sum = 0.0_f32;
        for score in scores.iter().copied() {
            if score < min {
                min = score;
            }
            if score > max {
                max = score;
            }
            sum += score;
        }
        let avg = (sum / scores.len() as f32).clamp(0.0, 1.0);
        Some(EvidenceQualityStats {
            min: min.clamp(0.0, 1.0),
            max: max.clamp(0.0, 1.0),
            avg,
            count: scores.len(),
        })
    }

    pub async fn get_latest_self_memory_fact(
        &self,
        key: &str,
    ) -> Option<(String, Vec<i64>)> {
        let key = key.trim();
        if key.is_empty() {
            return None;
        }
        let row = sqlx::query(
            "SELECT f.value_literal as value_literal, e.source_evidence_ids as source_evidence_ids
             FROM self_fact_beliefs f
             JOIN self_beliefs b ON b.id = f.belief_id
             LEFT JOIN self_evidence_events e ON e.belief_id = b.id
             WHERE f.key = ?
             ORDER BY datetime(b.created_at) DESC
             LIMIT 1",
        )
        .bind(key)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten()?;

        let value: String = row.try_get("value_literal").unwrap_or_default();
        let evidence_raw: String = row
            .try_get("source_evidence_ids")
            .unwrap_or_else(|_| "[]".to_string());
        let evidence_ids = serde_json::from_str::<Vec<i64>>(&evidence_raw).unwrap_or_default();
        Some((value, evidence_ids))
    }

    pub async fn get_recent_user_evidence(
        &self,
        conversation_id: &str,
        limit: i64,
    ) -> Vec<(i64, String)> {
        let rows = sqlx::query(
            "SELECT content, metadata FROM messages
             WHERE conversation_id = ? AND role = 'user' AND metadata IS NOT NULL
             ORDER BY created_at DESC
             LIMIT ?",
        )
        .bind(conversation_id)
        .bind(limit.max(1))
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();

        let mut entries: Vec<(i64, String)> = Vec::new();
        let mut seen = HashSet::new();
        for row in rows {
            let content: String = row.try_get("content").unwrap_or_default();
            let snippet = content
                .trim()
                .chars()
                .take(180)
                .collect::<String>()
                .replace('\n', " ")
                .replace('\r', " ");
            let meta: Option<String> = row.try_get("metadata").ok();
            let Some(meta) = meta else { continue; };
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&meta) else { continue; };
            if let Some(array) = value.get("evidence_event_ids").and_then(|v| v.as_array()) {
                for id in array.iter().filter_map(|v| v.as_i64()) {
                    if seen.insert(id) {
                        entries.push((id, snippet.clone()));
                    }
                }
            } else if let Some(id) = value.get("evidence_event_id").and_then(|v| v.as_i64()) {
                if seen.insert(id) {
                    entries.push((id, snippet.clone()));
                }
            }
        }
        entries
    }

    pub async fn evidence_ids_are_user_anchored(&self, evidence_ids: &[i64]) -> bool {
        if evidence_ids.is_empty() {
            return false;
        }
        let mut unique: Vec<i64> = evidence_ids.iter().copied().filter(|id| *id > 0).collect();
        unique.sort();
        unique.dedup();
        if unique.is_empty() {
            return false;
        }
        let placeholders = unique.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let query = format!(
            "SELECT id, source_type FROM ics_evidence_events WHERE id IN ({})",
            placeholders
        );
        let mut builder = sqlx::query(&query);
        for id in unique.iter() {
            builder = builder.bind(id);
        }
        let rows = builder.fetch_all(&self.pool).await.unwrap_or_default();
        if rows.len() != unique.len() {
            return false;
        }
        let mut has_user = false;
        let mut has_non_user = false;
        for row in rows {
            let source_type: String = row.try_get("source_type").unwrap_or_default();
            let lowered = source_type.trim().to_lowercase();
            if lowered.starts_with("user") {
                has_user = true;
            } else {
                has_non_user = true;
            }
            if has_user && has_non_user {
                break;
            }
        }
        has_user && !has_non_user
    }

    pub async fn create_user_utterance_evidence(
        &self,
        conversation_id: &str,
        message_id: &str,
        content: &str,
    ) -> Option<i64> {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return None;
        }

        let mut user_id: Option<i64> = sqlx::query_scalar(
            "SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = 'user' LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();

        if user_id.is_none() {
            user_id = sqlx::query_scalar(
                "SELECT entity_id FROM ics_session_bindings WHERE session_id = 'default' AND ref_text = 'user' LIMIT 1",
            )
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten();
        }

        if user_id.is_none() {
            let settings = self.get_settings().await.ok();
            let user_name = settings
                .as_ref()
                .and_then(|s| s.user_display_name.clone())
                .unwrap_or_else(|| "User".to_string());
            let assistant_name = settings
                .as_ref()
                .and_then(|s| s.assistant_display_name.clone())
                .unwrap_or_else(|| "Ergo".to_string());
            let _ = ensure_primary_entities(&self.pool, &user_name, &assistant_name).await;
            user_id = sqlx::query_scalar(
                "SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = 'user' LIMIT 1",
            )
            .bind(conversation_id)
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten();
        }

        let Some(subject_id) = user_id else {
            return None;
        };

        const MAX_UTTERANCE_VALUE_LEN: usize = 240;
        let value_literal = if trimmed.chars().count() > MAX_UTTERANCE_VALUE_LEN {
            format!("hash:{}", compute_value_hash(trimmed))
        } else {
            trimmed.to_string()
        };
        let value_hash = compute_value_hash(&value_literal);
        let scope_str = serde_json::to_string(&Scope::Session).unwrap_or_else(|_| "\"session\"".to_string());
        let key = "utterance";
        let topic_key = compute_topic_key_fact(subject_id, key);
        let sig_inputs = vec![
            ("subject_id".to_string(), subject_id.to_string()),
            ("key".to_string(), key.to_string()),
            ("value_hash".to_string(), value_hash.clone()),
            ("scope".to_string(), scope_str.clone()),
            ("time_bucket_kind".to_string(), "atemporal".to_string()),
            ("time_bucket_value".to_string(), "".to_string()),
            ("polarity".to_string(), "assert".to_string()),
        ];
        let sig_refs: Vec<(&str, &str)> = sig_inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let signature_hash = compute_signature_hash(&sig_refs);

        let mut belief_id: Option<i64> = None;
        let mut existing_weight: f64 = 0.0;
        let mut existing_confidence: f64 = 1.0;
        if let Ok(row) = sqlx::query(
            "SELECT id, evidence_weight_total, confidence FROM ics_beliefs
             WHERE signature_hash = ? AND scope = ? AND polarity = 'assert' AND status = 'active' LIMIT 1",
        )
        .bind(&signature_hash)
        .bind(&scope_str)
        .fetch_optional(&self.pool)
        .await
        {
            if let Some(row) = row {
                belief_id = Some(row.get("id"));
                existing_weight = row.try_get::<f64, _>("evidence_weight_total").unwrap_or(0.0);
                existing_confidence = row.try_get::<f64, _>("confidence").unwrap_or(1.0);
            }
        }

        let weight = compute_evidence_weight(SourceType::User) as f64;

        if belief_id.is_none() {
            let inserted = sqlx::query(
                "INSERT INTO ics_beliefs
                 (kind, scope, polarity, layer, topic_key, signature_hash, evidence_weight_total, confidence, time_bucket_kind, time_bucket_value, observed_at, created_at)
                 VALUES ('fact', ?, 'assert', 'episodic', ?, ?, ?, ?, 'atemporal', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                 RETURNING id",
            )
            .bind(&scope_str)
            .bind(&topic_key)
            .bind(&signature_hash)
            .bind(weight)
            .bind(1.0)
            .fetch_one(&self.pool)
            .await
            .ok();

            if let Some(row) = inserted {
                let id: i64 = row.get("id");
                belief_id = Some(id);
                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO ics_fact_beliefs (belief_id, subject_entity_id, key, value_literal, value_hash)
                     VALUES (?, ?, ?, ?, ?)",
                )
                .bind(id)
                .bind(subject_id)
                .bind(key)
                .bind(&value_literal)
                .bind(&value_hash)
                .execute(&self.pool)
                .await;
            }
        }

        let Some(belief_id) = belief_id else {
            return None;
        };

        let snippet = trimmed.chars().take(180).collect::<String>().replace('\n', " ").replace('\r', " ");
        let event_row = sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id)
             VALUES (?, 'user', ?, ?, ?, NULL)
             RETURNING id",
        )
        .bind(belief_id)
        .bind(message_id)
        .bind(&snippet)
        .bind(weight)
        .fetch_one(&self.pool)
        .await
        .ok();
        let event_id = event_row.map(|row| row.get::<i64, _>("id"));

        let new_weight = existing_weight + weight;
        let new_confidence = existing_confidence.max(1.0);
        let _ = sqlx::query(
            "UPDATE ics_beliefs SET evidence_weight_total = ?, confidence = ?, last_evidence_at = CURRENT_TIMESTAMP WHERE id = ?",
        )
        .bind(new_weight)
        .bind(new_confidence)
        .bind(belief_id)
        .execute(&self.pool)
        .await;

        event_id
    }

    pub async fn create_user_identity_evidence_event(
        &self,
        message_id: &str,
        snippet: &str,
    ) -> Option<i64> {
        let snippet = snippet.trim();
        if snippet.is_empty() {
            return None;
        }

        let meta_raw: Option<String> = sqlx::query_scalar(
            "SELECT metadata FROM messages WHERE message_id = ? LIMIT 1",
        )
        .bind(message_id)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();

        let meta_raw = meta_raw?;
        let meta_val: serde_json::Value = serde_json::from_str(&meta_raw).ok()?;
        let mut evidence_ids: Vec<i64> = Vec::new();
        if let Some(array) = meta_val.get("evidence_event_ids").and_then(|v| v.as_array()) {
            evidence_ids.extend(array.iter().filter_map(|v| v.as_i64()));
        } else if let Some(id) = meta_val.get("evidence_event_id").and_then(|v| v.as_i64()) {
            evidence_ids.push(id);
        }
        let evidence_id = evidence_ids.first().copied()?;
        let belief_id: Option<i64> = sqlx::query_scalar(
            "SELECT belief_id FROM ics_evidence_events WHERE id = ? LIMIT 1",
        )
        .bind(evidence_id)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();
        let belief_id = belief_id?;

        let existing: Option<i64> = sqlx::query_scalar(
            "SELECT id FROM ics_evidence_events
             WHERE belief_id = ? AND source_type = 'user_identity' AND source_ref = ? LIMIT 1",
        )
        .bind(belief_id)
        .bind(message_id)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();
        if let Some(existing) = existing {
            return Some(existing);
        }

        let event_row = sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id)
             VALUES (?, 'user_identity', ?, ?, 0.0, NULL)
             RETURNING id",
        )
        .bind(belief_id)
        .bind(message_id)
        .bind(snippet)
        .fetch_one(&self.pool)
        .await
        .ok();
        let event_id = event_row.map(|row| row.get::<i64, _>("id"));
        let _ = sqlx::query(
            "UPDATE ics_beliefs SET last_evidence_at = CURRENT_TIMESTAMP WHERE id = ?",
        )
        .bind(belief_id)
        .execute(&self.pool)
        .await;

        event_id
    }

    pub async fn create_user_feedback_evidence(
        &self,
        conversation_id: &str,
        assistant_message_id: Option<&str>,
        content: &str,
        kind: &str,
    ) -> Option<i64> {
        let trimmed = content.trim();
        let kind = kind.trim();
        if trimmed.is_empty() || kind.is_empty() {
            return None;
        }

        let mut user_id: Option<i64> = sqlx::query_scalar(
            "SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = 'user' LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();

        if user_id.is_none() {
            user_id = sqlx::query_scalar(
                "SELECT entity_id FROM ics_session_bindings WHERE session_id = 'default' AND ref_text = 'user' LIMIT 1",
            )
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten();
        }

        if user_id.is_none() {
            let settings = self.get_settings().await.ok();
            let user_name = settings
                .as_ref()
                .and_then(|s| s.user_display_name.clone())
                .unwrap_or_else(|| "User".to_string());
            let assistant_name = settings
                .as_ref()
                .and_then(|s| s.assistant_display_name.clone())
                .unwrap_or_else(|| "Ergo".to_string());
            let _ = ensure_primary_entities(&self.pool, &user_name, &assistant_name).await;
            user_id = sqlx::query_scalar(
                "SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = 'user' LIMIT 1",
            )
            .bind(conversation_id)
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten();
        }

        let Some(subject_id) = user_id else {
            return None;
        };

        let scope_str = serde_json::to_string(&Scope::Session).unwrap_or_else(|_| "\"session\"".to_string());
        let key = "feedback";
        let value_literal = if kind.chars().count() > 120 {
            format!("hash:{}", compute_value_hash(kind))
        } else {
            kind.to_string()
        };
        let value_hash = compute_value_hash(&value_literal);
        let topic_key = compute_topic_key_fact(subject_id, key);
        let sig_inputs = vec![
            ("subject_id".to_string(), subject_id.to_string()),
            ("key".to_string(), key.to_string()),
            ("value_hash".to_string(), value_hash.clone()),
            ("scope".to_string(), scope_str.clone()),
            ("time_bucket_kind".to_string(), "atemporal".to_string()),
            ("time_bucket_value".to_string(), "".to_string()),
            ("polarity".to_string(), "assert".to_string()),
        ];
        let sig_refs: Vec<(&str, &str)> = sig_inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let signature_hash = compute_signature_hash(&sig_refs);

        let mut belief_id: Option<i64> = None;
        let mut existing_weight: f64 = 0.0;
        let mut existing_confidence: f64 = 1.0;
        if let Ok(row) = sqlx::query(
            "SELECT id, evidence_weight_total, confidence FROM ics_beliefs
             WHERE signature_hash = ? AND scope = ? AND polarity = 'assert' AND status = 'active' LIMIT 1",
        )
        .bind(&signature_hash)
        .bind(&scope_str)
        .fetch_optional(&self.pool)
        .await
        {
            if let Some(row) = row {
                belief_id = Some(row.get("id"));
                existing_weight = row.try_get::<f64, _>("evidence_weight_total").unwrap_or(0.0);
                existing_confidence = row.try_get::<f64, _>("confidence").unwrap_or(1.0);
            }
        }

        let weight = compute_evidence_weight(SourceType::User) as f64;
        if belief_id.is_none() {
            let inserted = sqlx::query(
                "INSERT INTO ics_beliefs
                 (kind, scope, polarity, layer, topic_key, signature_hash, evidence_weight_total, confidence, time_bucket_kind, time_bucket_value, observed_at, created_at)
                 VALUES ('fact', ?, 'assert', 'episodic', ?, ?, ?, ?, 'atemporal', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                 RETURNING id",
            )
            .bind(&scope_str)
            .bind(&topic_key)
            .bind(&signature_hash)
            .bind(weight)
            .bind(1.0)
            .fetch_one(&self.pool)
            .await
            .ok();

            if let Some(row) = inserted {
                let id: i64 = row.get("id");
                belief_id = Some(id);
                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO ics_fact_beliefs (belief_id, subject_entity_id, key, value_literal, value_hash)
                     VALUES (?, ?, ?, ?, ?)",
                )
                .bind(id)
                .bind(subject_id)
                .bind(key)
                .bind(&value_literal)
                .bind(&value_hash)
                .execute(&self.pool)
                .await;
            }
        }

        let Some(belief_id) = belief_id else {
            return None;
        };

        let snippet = trimmed
            .chars()
            .take(180)
            .collect::<String>()
            .replace('\n', " ")
            .replace('\r', " ");
        let event_row = sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id)
             VALUES (?, 'user', ?, ?, ?, NULL)
             RETURNING id",
        )
        .bind(belief_id)
        .bind(assistant_message_id)
        .bind(&snippet)
        .bind(weight)
        .fetch_one(&self.pool)
        .await
        .ok();
        let event_id = event_row.map(|row| row.get::<i64, _>("id"));

        let new_weight = existing_weight + weight;
        let new_confidence = existing_confidence.max(1.0);
        let _ = sqlx::query(
            "UPDATE ics_beliefs SET evidence_weight_total = ?, confidence = ?, last_evidence_at = CURRENT_TIMESTAMP WHERE id = ?",
        )
        .bind(new_weight)
        .bind(new_confidence)
        .bind(belief_id)
        .execute(&self.pool)
        .await;

        event_id
    }

    pub async fn create_kv_evidence_event(
        &self,
        conversation_id: &str,
        key: &str,
        value: &str,
    ) -> Option<i64> {
        let key = key.trim();
        let value = value.trim();
        if key.is_empty() || value.is_empty() {
            return None;
        }

        let mut assistant_id: Option<i64> = sqlx::query_scalar(
            "SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = 'assistant' LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();

        if assistant_id.is_none() {
            assistant_id = sqlx::query_scalar(
                "SELECT entity_id FROM ics_session_bindings WHERE session_id = 'default' AND ref_text = 'assistant' LIMIT 1",
            )
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten();
        }

        if assistant_id.is_none() {
            let settings = self.get_settings().await.ok();
            let user_name = settings
                .as_ref()
                .and_then(|s| s.user_display_name.clone())
                .unwrap_or_else(|| "User".to_string());
            let assistant_name = settings
                .as_ref()
                .and_then(|s| s.assistant_display_name.clone())
                .unwrap_or_else(|| "Ergo".to_string());
            let _ = ensure_primary_entities(&self.pool, &user_name, &assistant_name).await;
            assistant_id = sqlx::query_scalar(
                "SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = 'assistant' LIMIT 1",
            )
            .bind(conversation_id)
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten();
        }

        let Some(subject_id) = assistant_id else {
            return None;
        };

        const MAX_CONTEXT_VALUE_LEN: usize = 240;
        let value_literal = if value.chars().count() > MAX_CONTEXT_VALUE_LEN {
            format!("hash:{}", compute_value_hash(value))
        } else {
            value.to_string()
        };
        let value_hash = compute_value_hash(&value_literal);
        let scope_str = serde_json::to_string(&Scope::Session).unwrap_or_else(|_| "\"session\"".to_string());
        let scoped_key = format!("context:{}", key);
        let topic_key = compute_topic_key_fact(subject_id, &scoped_key);
        let sig_inputs = vec![
            ("subject_id".to_string(), subject_id.to_string()),
            ("key".to_string(), scoped_key.clone()),
            ("value_hash".to_string(), value_hash.clone()),
            ("scope".to_string(), scope_str.clone()),
            ("time_bucket_kind".to_string(), "atemporal".to_string()),
            ("time_bucket_value".to_string(), "".to_string()),
            ("polarity".to_string(), "assert".to_string()),
        ];
        let sig_refs: Vec<(&str, &str)> = sig_inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let signature_hash = compute_signature_hash(&sig_refs);

        let mut belief_id: Option<i64> = None;
        let mut existing_weight: f64 = 0.0;
        let mut existing_confidence: f64 = 1.0;
        if let Ok(row) = sqlx::query(
            "SELECT id, evidence_weight_total, confidence FROM ics_beliefs
             WHERE signature_hash = ? AND scope = ? AND polarity = 'assert' AND status = 'active' LIMIT 1",
        )
        .bind(&signature_hash)
        .bind(&scope_str)
        .fetch_optional(&self.pool)
        .await
        {
            if let Some(row) = row {
                belief_id = Some(row.get("id"));
                existing_weight = row.try_get::<f64, _>("evidence_weight_total").unwrap_or(0.0);
                existing_confidence = row.try_get::<f64, _>("confidence").unwrap_or(1.0);
            }
        }

        let weight = compute_evidence_weight(SourceType::Tool) as f64;
        if belief_id.is_none() {
            let inserted = sqlx::query(
                "INSERT INTO ics_beliefs
                 (kind, scope, polarity, layer, topic_key, signature_hash, evidence_weight_total, confidence, time_bucket_kind, time_bucket_value, observed_at, created_at)
                 VALUES ('fact', ?, 'assert', 'episodic', ?, ?, ?, ?, 'atemporal', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                 RETURNING id",
            )
            .bind(&scope_str)
            .bind(&topic_key)
            .bind(&signature_hash)
            .bind(weight)
            .bind(1.0)
            .fetch_one(&self.pool)
            .await
            .ok();

            if let Some(row) = inserted {
                let id: i64 = row.get("id");
                belief_id = Some(id);
                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO ics_fact_beliefs (belief_id, subject_entity_id, key, value_literal, value_hash)
                     VALUES (?, ?, ?, ?, ?)",
                )
                .bind(id)
                .bind(subject_id)
                .bind(&scoped_key)
                .bind(&value_literal)
                .bind(&value_hash)
                .execute(&self.pool)
                .await;
            }
        }

        let Some(belief_id) = belief_id else {
            return None;
        };

        let snippet = format!("{} = {}", key, value)
            .chars()
            .take(180)
            .collect::<String>()
            .replace('\n', " ")
            .replace('\r', " ");
        let event_row = sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id)
             VALUES (?, 'tool', 'save_context', ?, ?, NULL)
             RETURNING id",
        )
        .bind(belief_id)
        .bind(&snippet)
        .bind(weight)
        .fetch_one(&self.pool)
        .await
        .ok();
        let event_id = event_row.map(|row| row.get::<i64, _>("id"));

        let new_weight = existing_weight + weight;
        let new_confidence = existing_confidence.max(1.0);
        let _ = sqlx::query(
            "UPDATE ics_beliefs SET evidence_weight_total = ?, confidence = ?, last_evidence_at = CURRENT_TIMESTAMP WHERE id = ?",
        )
        .bind(new_weight)
        .bind(new_confidence)
        .bind(belief_id)
        .execute(&self.pool)
        .await;

        event_id
    }

    async fn create_system_evidence_event_internal(
        &self,
        conversation_id: &str,
        key: &str,
        value: &str,
        source_ref: Option<&str>,
        snippet: &str,
        source_type: &str,
        context_json: Option<&str>,
        strength: Option<f64>,
    ) -> Option<i64> {
        let key = key.trim();
        let value = value.trim();
        if key.is_empty() || value.is_empty() {
            return None;
        }

        let mut assistant_id: Option<i64> = sqlx::query_scalar(
            "SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = 'assistant' LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();

        if assistant_id.is_none() {
            assistant_id = sqlx::query_scalar(
                "SELECT entity_id FROM ics_session_bindings WHERE session_id = 'default' AND ref_text = 'assistant' LIMIT 1",
            )
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten();
        }

        if assistant_id.is_none() {
            let settings = self.get_settings().await.ok();
            let user_name = settings
                .as_ref()
                .and_then(|s| s.user_display_name.clone())
                .unwrap_or_else(|| "User".to_string());
            let assistant_name = settings
                .as_ref()
                .and_then(|s| s.assistant_display_name.clone())
                .unwrap_or_else(|| "Ergo".to_string());
            let _ = ensure_primary_entities(&self.pool, &user_name, &assistant_name).await;
            assistant_id = sqlx::query_scalar(
                "SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = 'assistant' LIMIT 1",
            )
            .bind(conversation_id)
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten();
        }

        let Some(subject_id) = assistant_id else {
            return None;
        };

        const MAX_VALUE_LEN: usize = 240;
        let value_literal = if value.chars().count() > MAX_VALUE_LEN {
            format!("hash:{}", compute_value_hash(value))
        } else {
            value.to_string()
        };
        let value_hash = compute_value_hash(&value_literal);
        let scope_str = serde_json::to_string(&Scope::Session).unwrap_or_else(|_| "\"session\"".to_string());
        let topic_key = compute_topic_key_fact(subject_id, key);
        let sig_inputs = vec![
            ("subject_id".to_string(), subject_id.to_string()),
            ("key".to_string(), key.to_string()),
            ("value_hash".to_string(), value_hash.clone()),
            ("scope".to_string(), scope_str.clone()),
            ("time_bucket_kind".to_string(), "atemporal".to_string()),
            ("time_bucket_value".to_string(), "".to_string()),
            ("polarity".to_string(), "assert".to_string()),
        ];
        let sig_refs: Vec<(&str, &str)> = sig_inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let signature_hash = compute_signature_hash(&sig_refs);

        let mut belief_id: Option<i64> = None;
        let mut existing_weight: f64 = 0.0;
        let mut existing_confidence: f64 = 1.0;
        if let Ok(row) = sqlx::query(
            "SELECT id, evidence_weight_total, confidence FROM ics_beliefs
             WHERE signature_hash = ? AND scope = ? AND polarity = 'assert' AND status = 'active' LIMIT 1",
        )
        .bind(&signature_hash)
        .bind(&scope_str)
        .fetch_optional(&self.pool)
        .await
        {
            if let Some(row) = row {
                belief_id = Some(row.get("id"));
                existing_weight = row.try_get::<f64, _>("evidence_weight_total").unwrap_or(0.0);
                existing_confidence = row.try_get::<f64, _>("confidence").unwrap_or(1.0);
            }
        }

        let weight = compute_evidence_weight(SourceType::System) as f64;
        if belief_id.is_none() {
            let inserted = sqlx::query(
                "INSERT INTO ics_beliefs
                 (kind, scope, polarity, layer, topic_key, signature_hash, evidence_weight_total, confidence, time_bucket_kind, time_bucket_value, observed_at, created_at)
                 VALUES ('fact', ?, 'assert', 'episodic', ?, ?, ?, ?, 'atemporal', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                 RETURNING id",
            )
            .bind(&scope_str)
            .bind(&topic_key)
            .bind(&signature_hash)
            .bind(weight)
            .bind(1.0)
            .fetch_one(&self.pool)
            .await
            .ok();

            if let Some(row) = inserted {
                let id: i64 = row.get("id");
                belief_id = Some(id);
                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO ics_fact_beliefs (belief_id, subject_entity_id, key, value_literal, value_hash)
                     VALUES (?, ?, ?, ?, ?)",
                )
                .bind(id)
                .bind(subject_id)
                .bind(key)
                .bind(&value_literal)
                .bind(&value_hash)
                .execute(&self.pool)
                .await;
            }
        }

        let Some(belief_id) = belief_id else {
            return None;
        };

        let max_snippet = match key {
            "wave_state_snapshot" | "attention_schema_snapshot" | "prediction_residual_snapshot" | "qualia_snapshot" => 2048,
            _ => 180,
        };
        let snippet = snippet
            .trim()
            .chars()
            .take(max_snippet)
            .collect::<String>()
            .replace('\n', " ")
            .replace('\r', " ");
        let context_json = context_json
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let strength = strength.unwrap_or(0.0).clamp(0.0, 1.0);
        let event_row = sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id, context_json, strength)
             VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
             RETURNING id",
        )
        .bind(belief_id)
        .bind(source_type)
        .bind(source_ref)
        .bind(&snippet)
        .bind(weight)
        .bind(context_json)
        .bind(strength)
        .fetch_one(&self.pool)
        .await
        .ok();
        let event_id = event_row.map(|row| row.get::<i64, _>("id"));

        let new_weight = existing_weight + weight;
        let new_confidence = existing_confidence.max(1.0);
        let _ = sqlx::query(
            "UPDATE ics_beliefs SET evidence_weight_total = ?, confidence = ?, last_evidence_at = CURRENT_TIMESTAMP WHERE id = ?",
        )
        .bind(new_weight)
        .bind(new_confidence)
        .bind(belief_id)
        .execute(&self.pool)
        .await;

        event_id
    }

    pub async fn create_system_evidence_event(
        &self,
        conversation_id: &str,
        key: &str,
        value: &str,
        source_ref: Option<&str>,
        snippet: &str,
    ) -> Option<i64> {
        self.create_system_evidence_event_internal(
            conversation_id,
            key,
            value,
            source_ref,
            snippet,
            "system",
            None,
            None,
        )
        .await
    }

    pub async fn emit_system_evidence_event(
        &self,
        conversation_id: &str,
        source_type: &str,
        source_ref: Option<&str>,
        snippet: &str,
    ) -> Option<i64> {
        let source_type = source_type.trim();
        if source_type.is_empty() {
            return None;
        }
        self.create_system_evidence_event_internal(
            conversation_id,
            "system_event",
            source_type,
            source_ref,
            snippet,
            source_type,
            None,
            None,
        )
        .await
    }

    pub async fn emit_system_evidence_event_with_context(
        &self,
        conversation_id: &str,
        source_type: &str,
        source_ref: Option<&str>,
        snippet: &str,
        context_json: Option<&str>,
        strength: Option<f64>,
    ) -> Option<i64> {
        let source_type = source_type.trim();
        if source_type.is_empty() {
            return None;
        }
        self.create_system_evidence_event_internal(
            conversation_id,
            "system_event",
            source_type,
            source_ref,
            snippet,
            source_type,
            context_json,
            strength,
        )
        .await
    }

    pub async fn create_memory_status_evidence_event(
        &self,
        conversation_id: &str,
        snapshot_text: &str,
        snippet: &str,
    ) -> Option<i64> {
        self.create_system_evidence_event_internal(
            conversation_id,
            "memory_status",
            snapshot_text,
            Some("memory_status"),
            snippet,
            "system",
            None,
            None,
        )
        .await
    }

    pub async fn create_qualia_snapshot_evidence_event(
        &self,
        conversation_id: &str,
        snapshot: &str,
        source_ref: Option<&str>,
    ) -> Option<i64> {
        let event_id = self.create_system_evidence_event_internal(
            conversation_id,
            "qualia_snapshot",
            "snapshot",
            source_ref,
            snapshot,
            "qualia_snapshot",
            None,
            None,
        )
        .await;
        if let Some(event_id) = event_id {
            let _ = system_log::log_event(
                &self.pool,
                None,
                "info",
                "kernel",
                None,
                None,
                json!({
                    "event": "qualia_evidence_recorded",
                    "conversation_id": conversation_id,
                    "evidence_event_id": event_id,
                    "source_ref": source_ref,
                }),
            )
            .await;
        }
        event_id
    }

    pub async fn create_tool_output_evidence_event(
        &self,
        conversation_id: &str,
        tool_name: &str,
        output: &str,
        snippet: &str,
        context_json: Option<&str>,
        strength: Option<f64>,
    ) -> Option<i64> {
        let tool_name = tool_name.trim();
        let output = output.trim();
        if tool_name.is_empty() || output.is_empty() {
            return None;
        }

        let key = format!("tool_output:{}", tool_name);
        let mut assistant_id: Option<i64> = sqlx::query_scalar(
            "SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = 'assistant' LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();

        if assistant_id.is_none() {
            assistant_id = sqlx::query_scalar(
                "SELECT entity_id FROM ics_session_bindings WHERE session_id = 'default' AND ref_text = 'assistant' LIMIT 1",
            )
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten();
        }

        if assistant_id.is_none() {
            let settings = self.get_settings().await.ok();
            let user_name = settings
                .as_ref()
                .and_then(|s| s.user_display_name.clone())
                .unwrap_or_else(|| "User".to_string());
            let assistant_name = settings
                .as_ref()
                .and_then(|s| s.assistant_display_name.clone())
                .unwrap_or_else(|| "Ergo".to_string());
            let _ = ensure_primary_entities(&self.pool, &user_name, &assistant_name).await;
            assistant_id = sqlx::query_scalar(
                "SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = 'assistant' LIMIT 1",
            )
            .bind(conversation_id)
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten();
        }

        let Some(subject_id) = assistant_id else {
            return None;
        };

        const MAX_VALUE_LEN: usize = 240;
        let value_literal = if output.chars().count() > MAX_VALUE_LEN {
            format!("hash:{}", compute_value_hash(output))
        } else {
            output.to_string()
        };
        let value_hash = compute_value_hash(&value_literal);
        let scope_str = serde_json::to_string(&Scope::Session).unwrap_or_else(|_| "\"session\"".to_string());
        let topic_key = compute_topic_key_fact(subject_id, &key);
        let sig_inputs = vec![
            ("subject_id".to_string(), subject_id.to_string()),
            ("key".to_string(), key.to_string()),
            ("value_hash".to_string(), value_hash.clone()),
            ("scope".to_string(), scope_str.clone()),
            ("time_bucket_kind".to_string(), "atemporal".to_string()),
            ("time_bucket_value".to_string(), "".to_string()),
            ("polarity".to_string(), "assert".to_string()),
        ];
        let sig_refs: Vec<(&str, &str)> = sig_inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let signature_hash = compute_signature_hash(&sig_refs);

        let mut belief_id: Option<i64> = None;
        let mut existing_weight: f64 = 0.0;
        let mut existing_confidence: f64 = 1.0;
        if let Ok(row) = sqlx::query(
            "SELECT id, evidence_weight_total, confidence FROM ics_beliefs
             WHERE signature_hash = ? AND scope = ? AND polarity = 'assert' AND status = 'active' LIMIT 1",
        )
        .bind(&signature_hash)
        .bind(&scope_str)
        .fetch_optional(&self.pool)
        .await
        {
            if let Some(row) = row {
                belief_id = Some(row.get("id"));
                existing_weight = row.try_get::<f64, _>("evidence_weight_total").unwrap_or(0.0);
                existing_confidence = row.try_get::<f64, _>("confidence").unwrap_or(1.0);
            }
        }

        let weight = compute_evidence_weight(SourceType::Tool) as f64;
        if belief_id.is_none() {
            let inserted = sqlx::query(
                "INSERT INTO ics_beliefs
                 (kind, scope, polarity, layer, topic_key, signature_hash, evidence_weight_total, confidence, time_bucket_kind, time_bucket_value, observed_at, created_at)
                 VALUES ('fact', ?, 'assert', 'episodic', ?, ?, ?, ?, 'atemporal', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                 RETURNING id",
            )
            .bind(&scope_str)
            .bind(&topic_key)
            .bind(&signature_hash)
            .bind(weight)
            .bind(1.0)
            .fetch_one(&self.pool)
            .await
            .ok();

            if let Some(row) = inserted {
                let id: i64 = row.get("id");
                belief_id = Some(id);
                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO ics_fact_beliefs (belief_id, subject_entity_id, key, value_literal, value_hash)
                     VALUES (?, ?, ?, ?, ?)",
                )
                .bind(id)
                .bind(subject_id)
                .bind(&key)
                .bind(&value_literal)
                .bind(&value_hash)
                .execute(&self.pool)
                .await;
            }
        }

        let Some(belief_id) = belief_id else {
            return None;
        };

        let snippet = snippet
            .trim()
            .chars()
            .take(240)
            .collect::<String>()
            .replace('\n', " ")
            .replace('\r', " ");
        let context_json = context_json
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let strength = strength.unwrap_or(0.0).clamp(0.0, 1.0);
        let event_row = sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id, context_json, strength)
             VALUES (?, 'tool', ?, ?, ?, NULL, ?, ?)
             RETURNING id",
        )
        .bind(belief_id)
        .bind(tool_name)
        .bind(&snippet)
        .bind(weight)
        .bind(context_json)
        .bind(strength)
        .fetch_one(&self.pool)
        .await
        .ok();
        let event_id = event_row.map(|row| row.get::<i64, _>("id"));
        if let Some(event_id) = event_id {
            let _ = self.ensure_evidence_source_for_event(event_id).await;
        }

        let new_weight = existing_weight + weight;
        let new_confidence = (existing_confidence + (weight * 0.1)).min(1.0);
        let _ = sqlx::query(
            "UPDATE ics_beliefs SET evidence_weight_total = ?, confidence = ?, last_evidence_at = CURRENT_TIMESTAMP WHERE id = ?",
        )
        .bind(new_weight)
        .bind(new_confidence)
        .bind(belief_id)
        .execute(&self.pool)
        .await;

        event_id
    }

    pub async fn retag_evidence_event_source_type(
        &self,
        event_id: i64,
        source_type: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "UPDATE ics_evidence_events SET source_type = ? WHERE id = ?",
        )
        .bind(source_type)
        .bind(event_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn create_web_evidence_event(
        &self,
        conversation_id: &str,
        url: &str,
        snippet: &str,
        weight_hint: f64,
    ) -> Option<i64> {
        let url = url.trim();
        if url.is_empty() {
            return None;
        }

        let mut assistant_id: Option<i64> = sqlx::query_scalar(
            "SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = 'assistant' LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();

        if assistant_id.is_none() {
            assistant_id = sqlx::query_scalar(
                "SELECT entity_id FROM ics_session_bindings WHERE session_id = 'default' AND ref_text = 'assistant' LIMIT 1",
            )
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten();
        }

        if assistant_id.is_none() {
            let settings = self.get_settings().await.ok();
            let user_name = settings
                .as_ref()
                .and_then(|s| s.user_display_name.clone())
                .unwrap_or_else(|| "User".to_string());
            let assistant_name = settings
                .as_ref()
                .and_then(|s| s.assistant_display_name.clone())
                .unwrap_or_else(|| "Ergo".to_string());
            let _ = ensure_primary_entities(&self.pool, &user_name, &assistant_name).await;
            assistant_id = sqlx::query_scalar(
                "SELECT entity_id FROM ics_session_bindings WHERE session_id = ? AND ref_text = 'assistant' LIMIT 1",
            )
            .bind(conversation_id)
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten();
        }

        let Some(subject_id) = assistant_id else {
            return None;
        };

        const MAX_VALUE_LEN: usize = 240;
        let value_literal = if url.chars().count() > MAX_VALUE_LEN {
            format!("hash:{}", compute_value_hash(url))
        } else {
            url.to_string()
        };
        let value_hash = compute_value_hash(&value_literal);
        let scope_str = serde_json::to_string(&Scope::Session).unwrap_or_else(|_| "\"session\"".to_string());
        let key = "web_source_url";
        let topic_key = compute_topic_key_fact(subject_id, key);
        let sig_inputs = vec![
            ("subject_id".to_string(), subject_id.to_string()),
            ("key".to_string(), key.to_string()),
            ("value_hash".to_string(), value_hash.clone()),
            ("scope".to_string(), scope_str.clone()),
            ("time_bucket_kind".to_string(), "atemporal".to_string()),
            ("time_bucket_value".to_string(), "".to_string()),
            ("polarity".to_string(), "assert".to_string()),
        ];
        let sig_refs: Vec<(&str, &str)> = sig_inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let signature_hash = compute_signature_hash(&sig_refs);

        let mut belief_id: Option<i64> = None;
        let mut existing_weight: f64 = 0.0;
        let mut existing_confidence: f64 = 0.5;
        if let Ok(row) = sqlx::query(
            "SELECT id, evidence_weight_total, confidence FROM ics_beliefs
             WHERE signature_hash = ? AND scope = ? LIMIT 1",
        )
        .bind(&signature_hash)
        .bind(&scope_str)
        .fetch_optional(&self.pool)
        .await
        {
            if let Some(row) = row {
                belief_id = Some(row.get("id"));
                existing_weight = row.try_get::<f64, _>("evidence_weight_total").unwrap_or(0.0);
                existing_confidence = row.try_get::<f64, _>("confidence").unwrap_or(0.5);
            }
        }

        let base_weight = compute_evidence_weight(SourceType::Tool) as f64;
        let weight = (base_weight * weight_hint.max(0.1)).clamp(0.1, 1.0);
        if belief_id.is_none() {
            let inserted = sqlx::query(
                "INSERT INTO ics_beliefs
                 (kind, scope, status, layer, polarity, topic_key, signature_hash, evidence_weight_total, confidence, time_bucket_kind, time_bucket_value, observed_at, created_at)
                 VALUES ('fact', ?, 'inactive', 'working', 'assert', ?, ?, ?, ?, 'atemporal', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                 RETURNING id",
            )
            .bind(&scope_str)
            .bind(&topic_key)
            .bind(&signature_hash)
            .bind(weight)
            .bind(existing_confidence.max(0.5))
            .fetch_one(&self.pool)
            .await
            .ok();

            if let Some(row) = inserted {
                let id: i64 = row.get("id");
                belief_id = Some(id);
                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO ics_fact_beliefs (belief_id, subject_entity_id, key, value_literal, value_hash)
                     VALUES (?, ?, ?, ?, ?)",
                )
                .bind(id)
                .bind(subject_id)
                .bind(key)
                .bind(&value_literal)
                .bind(&value_hash)
                .execute(&self.pool)
                .await;
            }
        }

        let Some(belief_id) = belief_id else {
            return None;
        };

        let snippet = snippet
            .trim()
            .chars()
            .take(220)
            .collect::<String>()
            .replace('\n', " ")
            .replace('\r', " ");
        let event_row = sqlx::query(
            "INSERT INTO ics_evidence_events (belief_id, source_type, source_ref, snippet, weight, episodic_event_id)
             VALUES (?, 'tool', ?, ?, ?, NULL)
             RETURNING id",
        )
        .bind(belief_id)
        .bind(url)
        .bind(&snippet)
        .bind(weight)
        .fetch_one(&self.pool)
        .await
        .ok();
        let event_id = event_row.map(|row| row.get::<i64, _>("id"));

        let new_weight = existing_weight + weight;
        let new_confidence = (existing_confidence + (weight * 0.2)).min(1.0);
        let _ = sqlx::query(
            "UPDATE ics_beliefs SET evidence_weight_total = ?, confidence = ?, last_evidence_at = CURRENT_TIMESTAMP WHERE id = ?",
        )
        .bind(new_weight)
        .bind(new_confidence)
        .bind(belief_id)
        .execute(&self.pool)
        .await;

        event_id
    }

    pub async fn get_history(&self, window: i32) -> Result<Vec<crate::models::Message>, Box<dyn std::error::Error + Send + Sync>> {
        self.get_history_for_conversation("default", window).await
    }

    pub async fn get_last_assistant_message(
        &self,
        conversation_id: &str,
    ) -> Option<(String, String)> {
        let row = sqlx::query(
            "SELECT message_id, content FROM messages
             WHERE conversation_id = ?
               AND role = 'assistant'
               AND status = 'complete'
               AND (
                 metadata IS NULL
                 OR json_extract(metadata, '$.source') IS NULL
                 OR json_extract(metadata, '$.source') != 'monologue'
                 OR json_extract(metadata, '$.surface') = 1
               )
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();
        row.map(|r| {
            let message_id: String = r.get("message_id");
            let content: String = r.get("content");
            (message_id, content)
        })
    }

    pub async fn get_user_message_id_for_run(
        &self,
        run_id: &str,
    ) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT message_id FROM messages
             WHERE run_id = ? AND role = 'user'
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .bind(run_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| r.get::<String, _>("message_id")))
    }

    pub async fn get_latest_user_message_at(
        &self,
        conversation_id: &str,
    ) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT created_at FROM messages
             WHERE conversation_id = ? AND role = 'user'
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| r.get::<String, _>("created_at")))
    }

    pub async fn get_latest_user_message(
        &self,
        conversation_id: &str,
    ) -> Result<Option<(String, String, String)>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT message_id, content, created_at FROM messages
             WHERE conversation_id = ? AND role = 'user'
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| {
            let message_id: String = r.get("message_id");
            let content: String = r.get("content");
            let created_at: String = r.get("created_at");
            (message_id, content, created_at)
        }))
    }

    pub async fn get_rolling_summary(&self, conversation_id: &str) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query("SELECT summary FROM conversation_summaries WHERE conversation_id = ?")
            .bind(conversation_id)
            .fetch_optional(&self.pool)
            .await?;

        use sqlx::Row;
        Ok(row.map(|r| r.get("summary")))
    }

    pub async fn get_live_summary(&self, conversation_id: &str) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query("SELECT summary FROM conversation_live_summaries WHERE conversation_id = ?")
            .bind(conversation_id)
            .fetch_optional(&self.pool)
            .await?;

        use sqlx::Row;
        Ok(row.map(|r| r.get("summary")))
    }

    pub async fn get_effective_rolling_summary(
        &self,
        conversation_id: &str,
    ) -> Result<(Option<String>, bool), Box<dyn std::error::Error + Send + Sync>> {
        let stored = self.get_rolling_summary(conversation_id).await?;
        if let Some(summary) = stored.as_ref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            return Ok((Some(summary.to_string()), false));
        }
        let live = self.get_live_summary(conversation_id).await?;
        if let Some(summary) = live.as_ref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            return Ok((Some(summary.to_string()), true));
        }
        Ok((None, false))
    }

    pub async fn set_rolling_summary(&self, conversation_id: &str, summary: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO conversation_summaries (conversation_id, summary, updated_at, version, pending, last_error, last_error_at)
             VALUES (?, ?, CURRENT_TIMESTAMP, 1, 0, NULL, NULL)
             ON CONFLICT(conversation_id)
             DO UPDATE SET summary = excluded.summary, updated_at = CURRENT_TIMESTAMP, version = version + 1, pending = 0, last_error = NULL, last_error_at = NULL"
        )
        .bind(conversation_id)
        .bind(summary)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn set_live_summary(&self, conversation_id: &str, summary: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO conversation_live_summaries (conversation_id, summary, updated_at, version, pending, last_error, last_error_at)
             VALUES (?, ?, CURRENT_TIMESTAMP, 1, 0, NULL, NULL)
             ON CONFLICT(conversation_id)
             DO UPDATE SET summary = excluded.summary, updated_at = CURRENT_TIMESTAMP, version = version + 1, pending = 0, last_error = NULL, last_error_at = NULL"
        )
        .bind(conversation_id)
        .bind(summary)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn set_rolling_summary_error(&self, conversation_id: &str, error: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO conversation_summaries (conversation_id, summary, updated_at, version, pending, last_error, last_error_at)
             VALUES (?, '', CURRENT_TIMESTAMP, 1, 0, ?, CURRENT_TIMESTAMP)
             ON CONFLICT(conversation_id)
             DO UPDATE SET pending = 0, last_error = excluded.last_error, last_error_at = excluded.last_error_at"
        )
        .bind(conversation_id)
        .bind(error)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn set_live_summary_error(&self, conversation_id: &str, error: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO conversation_live_summaries (conversation_id, summary, updated_at, version, pending, last_error, last_error_at)
             VALUES (?, '', CURRENT_TIMESTAMP, 1, 0, ?, CURRENT_TIMESTAMP)
             ON CONFLICT(conversation_id)
             DO UPDATE SET pending = 0, last_error = excluded.last_error, last_error_at = excluded.last_error_at"
        )
        .bind(conversation_id)
        .bind(error)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn set_summary_pending(&self, conversation_id: &str, pending: bool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let pending_val = if pending { 1 } else { 0 };
        let updated = sqlx::query(
            "UPDATE conversation_summaries SET pending = ? WHERE conversation_id = ?"
        )
        .bind(pending_val)
        .bind(conversation_id)
        .execute(&self.pool)
        .await?
        .rows_affected();
        if updated == 0 {
            sqlx::query(
                "INSERT INTO conversation_summaries (conversation_id, summary, updated_at, version, pending, last_error, last_error_at)
                 VALUES (?, '', CURRENT_TIMESTAMP, 0, ?, NULL, NULL)"
            )
            .bind(conversation_id)
            .bind(pending_val)
            .execute(&self.pool)
            .await?;
        }

        let updated_live = sqlx::query(
            "UPDATE conversation_live_summaries SET pending = ? WHERE conversation_id = ?"
        )
        .bind(pending_val)
        .bind(conversation_id)
        .execute(&self.pool)
        .await?
        .rows_affected();
        if updated_live == 0 {
            sqlx::query(
                "INSERT INTO conversation_live_summaries (conversation_id, summary, updated_at, version, pending, last_error, last_error_at)
                 VALUES (?, '', CURRENT_TIMESTAMP, 0, ?, NULL, NULL)"
            )
            .bind(conversation_id)
            .bind(pending_val)
            .execute(&self.pool)
            .await?;
        }
        Ok(())
    }

    pub async fn get_rolling_summary_status(&self, conversation_id: &str) -> Result<crate::models::RollingSummaryStatus, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT summary, last_error, last_error_at, pending FROM conversation_summaries WHERE conversation_id = ?"
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;

        use sqlx::Row;
        Ok(match row {
            Some(r) => crate::models::RollingSummaryStatus {
                summary: r.try_get("summary").ok(),
                last_error: r.try_get("last_error").ok(),
                last_error_at: r.try_get("last_error_at").ok(),
                pending: r.try_get::<i64, _>("pending").unwrap_or(0) != 0,
            },
            None => crate::models::RollingSummaryStatus {
                summary: None,
                last_error: None,
                last_error_at: None,
                pending: false,
            },
        })
    }

    pub async fn get_live_summary_status(&self, conversation_id: &str) -> Result<crate::models::RollingSummaryStatus, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT summary, last_error, last_error_at, pending FROM conversation_live_summaries WHERE conversation_id = ?"
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;

        use sqlx::Row;
        Ok(match row {
            Some(r) => crate::models::RollingSummaryStatus {
                summary: r.try_get("summary").ok(),
                last_error: r.try_get("last_error").ok(),
                last_error_at: r.try_get("last_error_at").ok(),
                pending: r.try_get::<i64, _>("pending").unwrap_or(0) != 0,
            },
            None => crate::models::RollingSummaryStatus {
                summary: None,
                last_error: None,
                last_error_at: None,
                pending: false,
            },
        })
    }

    pub async fn clear_rolling_summary(
        &self,
        conversation_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("DELETE FROM conversation_summaries WHERE conversation_id = ?")
            .bind(conversation_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn clear_inner_summary(
        &self,
        conversation_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("DELETE FROM inner_summaries WHERE conversation_id = ?")
            .bind(conversation_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn get_kernel_state(&self, conversation_id: &str) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let row: Option<String> = sqlx::query_scalar(
            "SELECT state_json FROM kernel_states WHERE conversation_id = ?"
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row)
    }

    pub async fn get_kernel_state_with_meta(
        &self,
        conversation_id: &str,
    ) -> Result<Option<(String, i64)>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT state_json, monologue_write_version FROM kernel_states WHERE conversation_id = ?"
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;
        if let Some(row) = row {
            let state_json: String = row.try_get("state_json")?;
            let version: i64 = row.try_get("monologue_write_version")?;
            Ok(Some((state_json, version)))
        } else {
            Ok(None)
        }
    }

    pub async fn set_kernel_state(
        &self,
        conversation_id: &str,
        state_json: &str,
        state_write_owner: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO kernel_states (conversation_id, state_json, state_write_owner, updated_at)
             VALUES (?, ?, ?, CURRENT_TIMESTAMP)
             ON CONFLICT(conversation_id) DO UPDATE SET state_json = excluded.state_json,
                 state_write_owner = excluded.state_write_owner,
                 monologue_write_version = monologue_write_version + 1,
                 updated_at = CURRENT_TIMESTAMP"
        )
        .bind(conversation_id)
        .bind(state_json)
        .bind(state_write_owner)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn update_kernel_state_with_version(
        &self,
        conversation_id: &str,
        state_json: &str,
        state_write_owner: Option<&str>,
        expected_version: i64,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let result = sqlx::query(
            "UPDATE kernel_states
             SET state_json = ?, state_write_owner = ?, monologue_write_version = monologue_write_version + 1, updated_at = CURRENT_TIMESTAMP
             WHERE conversation_id = ? AND monologue_write_version = ?"
        )
        .bind(state_json)
        .bind(state_write_owner)
        .bind(conversation_id)
        .bind(expected_version)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    pub async fn get_proaction_state(&self) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let row: Option<String> = sqlx::query_scalar(
            "SELECT state_json FROM proaction_state WHERE id = 1"
        )
        .fetch_optional(&self.pool)
        .await?;
        Ok(row)
    }

    pub async fn set_proaction_state(&self, state_json: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO proaction_state (id, state_json, updated_at)
             VALUES (1, ?, CURRENT_TIMESTAMP)
             ON CONFLICT(id) DO UPDATE SET state_json = excluded.state_json,
                 updated_at = CURRENT_TIMESTAMP"
        )
        .bind(state_json)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn get_inner_summary(&self, conversation_id: &str) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let row: Option<String> = sqlx::query_scalar(
            "SELECT summary_json FROM inner_summaries WHERE conversation_id = ?"
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row)
    }

    pub async fn get_workspace_state(
        &self,
        conversation_id: &str,
    ) -> Result<Option<crate::models::WorkspaceState>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT conversation_id, goal_thread, active_plan_id, goal_stack_json, open_questions_json, active_hypotheses_json,
                    working_set_topics_json, current_focus, focus_rationale, workspace_meta_json, updated_at
             FROM workspace_state WHERE conversation_id = ?",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;

        use sqlx::Row;
        Ok(row.map(|r| crate::models::WorkspaceState {
            conversation_id: r.get("conversation_id"),
            goal_thread: r.try_get::<Option<String>, _>("goal_thread").ok().flatten(),
            active_plan_id: r.try_get::<Option<String>, _>("active_plan_id").ok().flatten(),
            goal_stack: parse_goal_stack_json(r.try_get::<Option<String>, _>("goal_stack_json").ok().flatten()),
            open_questions: parse_json_list(r.try_get::<Option<String>, _>("open_questions_json").ok().flatten()),
            active_hypotheses: parse_hypotheses_json(
                r.try_get::<Option<String>, _>("active_hypotheses_json").ok().flatten(),
            ),
            working_set_topics: parse_json_list(r.try_get::<Option<String>, _>("working_set_topics_json").ok().flatten()),
            current_focus: r.try_get::<Option<String>, _>("current_focus").ok().flatten(),
            focus_rationale: r.try_get::<Option<String>, _>("focus_rationale").ok().flatten(),
            workspace_meta: parse_workspace_meta_json(
                r.try_get::<Option<String>, _>("workspace_meta_json").ok().flatten(),
            ),
            updated_at: r.try_get::<Option<String>, _>("updated_at").ok().flatten(),
        }))
    }

    pub async fn get_workspace_active_plan(
        &self,
        conversation_id: &str,
    ) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let row: Option<String> = sqlx::query_scalar(
            "SELECT active_plan_id FROM workspace_state WHERE conversation_id = ?",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row)
    }

    pub async fn set_workspace_active_plan(
        &self,
        conversation_id: &str,
        proposal_id: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "UPDATE workspace_state SET active_plan_id = ?, updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
        )
        .bind(proposal_id)
        .bind(conversation_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn set_workspace_state(
        &self,
        state: &crate::models::WorkspaceState,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let active_plan_id = if state.active_plan_id.is_some() {
            state.active_plan_id.clone()
        } else {
            self.get_workspace_active_plan(&state.conversation_id)
                .await
                .ok()
                .flatten()
        };
        let goal_stack_json =
            serde_json::to_string(&state.goal_stack).unwrap_or_else(|_| "[]".to_string());
        let open_questions_json = serde_json::to_string(&state.open_questions).unwrap_or_else(|_| "[]".to_string());
        let active_hypotheses_json =
            serde_json::to_string(&state.active_hypotheses).unwrap_or_else(|_| "[]".to_string());
        let working_set_topics_json =
            serde_json::to_string(&state.working_set_topics).unwrap_or_else(|_| "[]".to_string());
        let workspace_meta_json =
            serde_json::to_string(&state.workspace_meta).unwrap_or_else(|_| "{}".to_string());

        sqlx::query(
            "INSERT INTO workspace_state (conversation_id, goal_thread, active_plan_id, goal_stack_json, open_questions_json, active_hypotheses_json,
                working_set_topics_json, current_focus, focus_rationale, workspace_meta_json, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
             ON CONFLICT(conversation_id) DO UPDATE SET
                goal_thread = excluded.goal_thread,
                active_plan_id = COALESCE(excluded.active_plan_id, workspace_state.active_plan_id),
                goal_stack_json = excluded.goal_stack_json,
                open_questions_json = excluded.open_questions_json,
                active_hypotheses_json = excluded.active_hypotheses_json,
                working_set_topics_json = excluded.working_set_topics_json,
                current_focus = excluded.current_focus,
                focus_rationale = excluded.focus_rationale,
                workspace_meta_json = excluded.workspace_meta_json,
                updated_at = CURRENT_TIMESTAMP",
        )
        .bind(&state.conversation_id)
        .bind(state.goal_thread.as_deref())
        .bind(active_plan_id.as_deref())
        .bind(&goal_stack_json)
        .bind(&open_questions_json)
        .bind(&active_hypotheses_json)
        .bind(&working_set_topics_json)
        .bind(state.current_focus.as_deref())
        .bind(state.focus_rationale.as_deref())
        .bind(&workspace_meta_json)
        .execute(&self.pool)
        .await?;

        crate::core::memory::cache::bump_cache_version();

        Ok(())
    }

    pub async fn get_memory_context_hash(
        &self,
        conversation_id: &str,
    ) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let conversation_id = conversation_id.trim();
        if conversation_id.is_empty() {
            return Ok(None);
        }

        let workspace_updated: Option<String> = sqlx::query_scalar(
            "SELECT updated_at FROM workspace_state WHERE conversation_id = ?",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?
        .flatten();

        let message_updated: Option<String> = sqlx::query_scalar(
            "SELECT created_at FROM messages WHERE conversation_id = ? ORDER BY datetime(created_at) DESC LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?
        .flatten();

        let ledger_updated: Option<String> = sqlx::query_scalar(
            "SELECT created_at FROM memory_write_ledger WHERE conversation_id = ? ORDER BY datetime(created_at) DESC LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?
        .flatten();

        let summary_updated: Option<String> = sqlx::query_scalar(
            "SELECT end_ts FROM conversation_summary_chunks WHERE conversation_id = ? ORDER BY datetime(end_ts) DESC LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?
        .flatten();

        let inner_updated: Option<String> = sqlx::query_scalar(
            "SELECT updated_at FROM inner_summaries WHERE conversation_id = ? LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?
        .flatten();

        let fingerprint = format!(
            "workspace:{}|message:{}|ledger:{}|summary:{}|inner:{}",
            workspace_updated.unwrap_or_default(),
            message_updated.unwrap_or_default(),
            ledger_updated.unwrap_or_default(),
            summary_updated.unwrap_or_default(),
            inner_updated.unwrap_or_default(),
        );

        let mut hasher = Sha256::new();
        hasher.update(fingerprint.as_bytes());
        Ok(Some(hex::encode(hasher.finalize())))
    }

    fn pending_prompt_looks_jsonish(prompt: &str) -> bool {
        let trimmed = prompt.trim();
        if trimmed.len() < 2 {
            return false;
        }
        let wrapped = (trimmed.starts_with('{') && trimmed.ends_with('}'))
            || (trimmed.starts_with('[') && trimmed.ends_with(']'));
        if !wrapped {
            return false;
        }
        let lower = trimmed.to_lowercase();
        if lower.contains("\"stance\"")
            || lower.contains("\"candidates\"")
            || lower.contains("\"done\"")
            || lower.contains("\"message\"")
        {
            return true;
        }
        serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
    }

    pub async fn enqueue_pending_prompt(
        &self,
        conversation_id: &str,
        prompt: &str,
        source: &str,
        auto_surface: bool,
        intent_kind: Option<&str>,
        bridge_id: Option<&str>,
        expires_at: Option<&str>,
        anchor_message_id: Option<&str>,
        anchor_hash: Option<&str>,
        anchor_created_at: Option<&str>,
        anchor_role: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        if Self::pending_prompt_looks_jsonish(prompt) {
            return Err("pending_prompt_sanitized".into());
        }
        let id = uuid::Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT INTO pending_user_prompts (id, conversation_id, prompt, source, created_at, auto_surface, intent_kind, bridge_id, expires_at, anchor_message_id, anchor_hash, anchor_created_at, anchor_role)
             VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&id)
        .bind(conversation_id)
        .bind(prompt)
        .bind(source)
        .bind(if auto_surface { 1 } else { 0 })
        .bind(intent_kind)
        .bind(bridge_id)
        .bind(expires_at)
        .bind(anchor_message_id)
        .bind(anchor_hash)
        .bind(anchor_created_at)
        .bind(anchor_role)
        .execute(&self.pool)
        .await?;
        Ok(id)
    }

    pub async fn pop_pending_prompt(
        &self,
        conversation_id: &str,
    ) -> Result<Option<(String, String, String, bool, Option<String>, Option<String>, i64, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>)>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT id, prompt, source, auto_surface, intent_kind, bridge_id, attempt_count, last_asked_at, expires_at, anchor_message_id, anchor_hash, anchor_created_at, anchor_role
             FROM pending_user_prompts
             WHERE conversation_id = ?
             ORDER BY datetime(created_at) ASC
             LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(r) = row {
            use sqlx::Row;
            let id: String = r.get("id");
            let prompt: String = r.get("prompt");
            let source: String = r.get("source");
            let auto_surface: i64 = r.try_get("auto_surface").unwrap_or(0);
            let intent_kind: Option<String> = r.try_get("intent_kind").ok();
            let bridge_id: Option<String> = r.try_get("bridge_id").ok();
            let attempt_count: i64 = r.try_get("attempt_count").unwrap_or(0);
            let last_asked_at: Option<String> = r.try_get("last_asked_at").ok();
            let expires_at: Option<String> = r.try_get("expires_at").ok();
            let anchor_message_id: Option<String> = r.try_get("anchor_message_id").ok();
            let anchor_hash: Option<String> = r.try_get("anchor_hash").ok();
            let anchor_created_at: Option<String> = r.try_get("anchor_created_at").ok();
            let anchor_role: Option<String> = r.try_get("anchor_role").ok();
            let _ = sqlx::query("DELETE FROM pending_user_prompts WHERE id = ?")
                .bind(&id)
                .execute(&self.pool)
                .await?;
            return Ok(Some((
                id,
                prompt,
                source,
                auto_surface != 0,
                intent_kind,
                bridge_id,
                attempt_count,
                last_asked_at,
                expires_at,
                anchor_message_id,
                anchor_hash,
                anchor_created_at,
                anchor_role,
            )));
        }
        Ok(None)
    }

    pub async fn peek_pending_prompt(
        &self,
        conversation_id: &str,
    ) -> Result<Option<(String, String, String, bool, Option<String>, Option<String>, i64, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>)>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT id, prompt, source, auto_surface, intent_kind, bridge_id, attempt_count, last_asked_at, expires_at, anchor_message_id, anchor_hash, anchor_created_at, anchor_role
             FROM pending_user_prompts
             WHERE conversation_id = ?
             ORDER BY datetime(created_at) ASC
             LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(r) = row {
            use sqlx::Row;
            let id: String = r.get("id");
            let prompt: String = r.get("prompt");
            let source: String = r.get("source");
            let auto_surface: i64 = r.try_get("auto_surface").unwrap_or(0);
            let intent_kind: Option<String> = r.try_get("intent_kind").ok();
            let bridge_id: Option<String> = r.try_get("bridge_id").ok();
            let attempt_count: i64 = r.try_get("attempt_count").unwrap_or(0);
            let last_asked_at: Option<String> = r.try_get("last_asked_at").ok();
            let expires_at: Option<String> = r.try_get("expires_at").ok();
            let anchor_message_id: Option<String> = r.try_get("anchor_message_id").ok();
            let anchor_hash: Option<String> = r.try_get("anchor_hash").ok();
            let anchor_created_at: Option<String> = r.try_get("anchor_created_at").ok();
            let anchor_role: Option<String> = r.try_get("anchor_role").ok();
            return Ok(Some((
                id,
                prompt,
                source,
                auto_surface != 0,
                intent_kind,
                bridge_id,
                attempt_count,
                last_asked_at,
                expires_at,
                anchor_message_id,
                anchor_hash,
                anchor_created_at,
                anchor_role,
            )));
        }
        Ok(None)
    }

    pub async fn list_pending_prompts(
        &self,
        conversation_id: &str,
        limit: i64,
    ) -> Result<Vec<(String, String, String, String, i64, bool, Option<String>, Option<String>, i64, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>)>, Box<dyn std::error::Error + Send + Sync>> {
        let rows = sqlx::query(
            "SELECT id, prompt, source, created_at, skip_count, auto_surface, intent_kind, bridge_id, attempt_count, last_asked_at, expires_at, anchor_message_id, anchor_hash, anchor_created_at, anchor_role
             FROM pending_user_prompts
             WHERE conversation_id = ?
             ORDER BY datetime(created_at) DESC
             LIMIT ?",
        )
        .bind(conversation_id)
        .bind(limit.max(1))
        .fetch_all(&self.pool)
        .await?;
        let mut prompts = Vec::new();
        for row in rows {
            use sqlx::Row;
            let id: String = row.get("id");
            let prompt: String = row.get("prompt");
            let source: String = row.get("source");
            let created_at: String = row.get("created_at");
            let skip_count: i64 = row.try_get("skip_count").unwrap_or(0);
            let auto_surface: i64 = row.try_get("auto_surface").unwrap_or(0);
            let intent_kind: Option<String> = row.try_get("intent_kind").ok();
            let bridge_id: Option<String> = row.try_get("bridge_id").ok();
            let attempt_count: i64 = row.try_get("attempt_count").unwrap_or(0);
            let last_asked_at: Option<String> = row.try_get("last_asked_at").ok();
            let expires_at: Option<String> = row.try_get("expires_at").ok();
            let anchor_message_id: Option<String> = row.try_get("anchor_message_id").ok();
            let anchor_hash: Option<String> = row.try_get("anchor_hash").ok();
            let anchor_created_at: Option<String> = row.try_get("anchor_created_at").ok();
            let anchor_role: Option<String> = row.try_get("anchor_role").ok();
            prompts.push((
                id,
                prompt,
                source,
                created_at,
                skip_count,
                auto_surface != 0,
                intent_kind,
                bridge_id,
                attempt_count,
                last_asked_at,
                expires_at,
                anchor_message_id,
                anchor_hash,
                anchor_created_at,
                anchor_role,
            ));
        }
        Ok(prompts)
    }

    pub async fn get_pending_prompt_by_id(
        &self,
        prompt_id: &str,
    ) -> Result<Option<(String, String, String, String, String, i64, bool, Option<String>, Option<String>, i64, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>)>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT id, prompt, source, conversation_id, created_at, skip_count, auto_surface, intent_kind, bridge_id, attempt_count, last_asked_at, expires_at, anchor_message_id, anchor_hash, anchor_created_at, anchor_role
             FROM pending_user_prompts
             WHERE id = ?
             LIMIT 1",
        )
        .bind(prompt_id)
        .fetch_optional(&self.pool)
        .await?;
        if let Some(r) = row {
            use sqlx::Row;
            let id: String = r.get("id");
            let prompt: String = r.get("prompt");
            let source: String = r.get("source");
            let conversation_id: String = r.get("conversation_id");
            let created_at: String = r.get("created_at");
            let skip_count: i64 = r.try_get("skip_count").unwrap_or(0);
            let auto_surface: i64 = r.try_get("auto_surface").unwrap_or(0);
            let intent_kind: Option<String> = r.try_get("intent_kind").ok();
            let bridge_id: Option<String> = r.try_get("bridge_id").ok();
            let attempt_count: i64 = r.try_get("attempt_count").unwrap_or(0);
            let last_asked_at: Option<String> = r.try_get("last_asked_at").ok();
            let expires_at: Option<String> = r.try_get("expires_at").ok();
            let anchor_message_id: Option<String> = r.try_get("anchor_message_id").ok();
            let anchor_hash: Option<String> = r.try_get("anchor_hash").ok();
            let anchor_created_at: Option<String> = r.try_get("anchor_created_at").ok();
            let anchor_role: Option<String> = r.try_get("anchor_role").ok();
            return Ok(Some((
                id,
                prompt,
                source,
                conversation_id,
                created_at,
                skip_count,
                auto_surface != 0,
                intent_kind,
                bridge_id,
                attempt_count,
                last_asked_at,
                expires_at,
                anchor_message_id,
                anchor_hash,
                anchor_created_at,
                anchor_role,
            )));
        }
        Ok(None)
    }

    pub async fn update_pending_prompt(
        &self,
        prompt_id: &str,
        prompt: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "UPDATE pending_user_prompts
             SET prompt = ?, skip_count = 0, attempt_count = 0, last_asked_at = NULL, created_at = CURRENT_TIMESTAMP
             WHERE id = ?",
        )
        .bind(prompt)
        .bind(prompt_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn increment_pending_prompt_skip_count(
        &self,
        prompt_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("UPDATE pending_user_prompts SET skip_count = skip_count + 1 WHERE id = ?")
            .bind(prompt_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn delete_pending_prompt(
        &self,
        prompt_id: &str,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let result = sqlx::query("DELETE FROM pending_user_prompts WHERE id = ?")
            .bind(prompt_id)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }

    pub async fn mark_pending_prompt_attempt(
        &self,
        prompt_id: &str,
        asked_at: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "UPDATE pending_user_prompts
             SET attempt_count = attempt_count + 1,
                 last_asked_at = ?
             WHERE id = ?",
        )
        .bind(asked_at)
        .bind(prompt_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn enqueue_deferred_item(
        &self,
        conversation_id: &str,
        item_type: &str,
        content: &str,
        source: Option<&str>,
        reason: &str,
        last_context_hash: Option<&str>,
        reopen_trigger: Option<&str>,
        attempt_count: i64,
        last_asked_at: Option<&str>,
        expires_at: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let id = uuid::Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT INTO deferred_queue
             (id, conversation_id, item_type, content, source, reason, last_context_hash, reopen_trigger, attempt_count, last_asked_at, expires_at, dropped_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(&id)
        .bind(conversation_id)
        .bind(item_type)
        .bind(content)
        .bind(source)
        .bind(reason)
        .bind(last_context_hash)
        .bind(reopen_trigger)
        .bind(attempt_count)
        .bind(last_asked_at)
        .bind(expires_at)
        .execute(&self.pool)
        .await?;
        Ok(id)
    }

    pub async fn dequeue_deferred_items(
        &self,
        conversation_id: &str,
        item_type: &str,
        limit: i64,
    ) -> Result<Vec<(String, String, i64, String, Option<String>)>, Box<dyn std::error::Error + Send + Sync>> {
        let rows = sqlx::query(
            "SELECT id, content, attempt_count, reason, source
             FROM deferred_queue
             WHERE conversation_id = ? AND item_type = ?
             ORDER BY datetime(dropped_at) ASC
             LIMIT ?",
        )
        .bind(conversation_id)
        .bind(item_type)
        .bind(limit.max(1))
        .fetch_all(&self.pool)
        .await?;

        let mut items = Vec::new();
        for row in rows {
            let id: String = row.get("id");
            let content: String = row.get("content");
            let attempt_count: i64 = row.try_get("attempt_count").unwrap_or(0);
            let reason: String = row.try_get("reason").unwrap_or_default();
            let source: Option<String> = row.try_get("source").ok();
            items.push((id, content, attempt_count, reason, source));
        }

        for (id, _, _, _, _) in items.iter() {
            let _ = sqlx::query("DELETE FROM deferred_queue WHERE id = ?")
                .bind(id)
                .execute(&self.pool)
                .await;
        }

        Ok(items)
    }

    pub async fn clear_pending_prompts(
        &self,
        conversation_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("DELETE FROM pending_user_prompts WHERE conversation_id = ?")
            .bind(conversation_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn count_pending_prompts(
        &self,
        conversation_id: &str,
    ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM pending_user_prompts WHERE conversation_id = ?",
        )
        .bind(conversation_id)
        .fetch_one(&self.pool)
        .await?;
        Ok(count)
    }

    pub async fn get_latest_monologue_intent(
        &self,
        conversation_id: &str,
    ) -> Result<Option<(String, String, Option<String>, Option<String>, String)>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT id, prompt, intent_kind, bridge_id, created_at
             FROM pending_user_prompts
             WHERE conversation_id = ?
               AND auto_surface = 1
               AND source = 'monologue'
             ORDER BY datetime(created_at) DESC
             LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;
        if let Some(r) = row {
            use sqlx::Row;
            let id: String = r.get("id");
            let prompt: String = r.get("prompt");
            let intent_kind: Option<String> = r.try_get("intent_kind").ok();
            let bridge_id: Option<String> = r.try_get("bridge_id").ok();
            let created_at: String = r.get("created_at");
            return Ok(Some((id, prompt, intent_kind, bridge_id, created_at)));
        }
        Ok(None)
    }

    pub async fn set_inner_summary(&self, conversation_id: &str, summary_json: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO inner_summaries (conversation_id, summary_json, updated_at, version)
             VALUES (?, ?, CURRENT_TIMESTAMP, 1)
             ON CONFLICT(conversation_id) DO UPDATE SET summary_json = excluded.summary_json,
                 updated_at = CURRENT_TIMESTAMP,
                 version = version + 1"
        )
        .bind(conversation_id)
        .bind(summary_json)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn set_inner_summary_error(&self, conversation_id: &str, error: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO inner_summaries (conversation_id, summary_json, updated_at, version, last_error, last_error_at)
             VALUES (?, '{}', CURRENT_TIMESTAMP, 1, ?, CURRENT_TIMESTAMP)
             ON CONFLICT(conversation_id) DO UPDATE SET last_error = excluded.last_error,
                 last_error_at = excluded.last_error_at,
                 updated_at = CURRENT_TIMESTAMP"
        )
        .bind(conversation_id)
        .bind(error)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_inner_monologue_entries(
        &self,
        conversation_id: &str,
        limit: i64,
    ) -> Result<Vec<crate::models::InnerMonologueEntry>, Box<dyn std::error::Error + Send + Sync>> {
        let rows = sqlx::query(
            "SELECT id, conversation_id, run_id, dialogue_id, turn_index, speaker, mode, stream_type, thought, descriptors_json, harvest_type, harvest_payload, created_at
             FROM inner_monologue_entries
             WHERE conversation_id = ?
             ORDER BY datetime(created_at) DESC
             LIMIT ?"
        )
        .bind(conversation_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        let mut out = Vec::new();
        for row in rows {
            let descriptors = parse_json_list(row.try_get::<Option<String>, _>("descriptors_json").ok().flatten());
            let descriptors = if descriptors.is_empty() { None } else { Some(descriptors) };
            out.push(crate::models::InnerMonologueEntry {
                id: row.try_get("id")?,
                conversation_id: row.try_get("conversation_id")?,
                run_id: row.try_get("run_id").ok(),
                dialogue_id: row.try_get("dialogue_id").ok(),
                turn_index: row.try_get("turn_index").ok(),
                speaker: row.try_get("speaker").ok(),
                mode: row.try_get("mode")?,
                stream_type: row.try_get("stream_type").ok(),
                thought: row.try_get("thought")?,
                descriptors,
                harvest_type: row.try_get("harvest_type").ok(),
                harvest_payload: row.try_get("harvest_payload").ok(),
                created_at: row.try_get("created_at")?,
                candidates: None,
            });
        }
        Ok(out)
    }

    pub async fn list_inner_monologue_entries_with_candidates(
        &self,
        conversation_id: &str,
        limit: i64,
    ) -> Result<Vec<crate::models::InnerMonologueEntry>, Box<dyn std::error::Error + Send + Sync>> {
        let mut entries = self.list_inner_monologue_entries(conversation_id, limit).await?;
        if entries.is_empty() {
            return Ok(entries);
        }

        let entry_ids: Vec<String> = entries.iter().map(|e| e.id.clone()).collect();
        let mut query = QueryBuilder::new(
            "SELECT id, entry_id, candidate_id, outcome, suppression_reason, candidate_json, created_at FROM inner_monologue_candidates WHERE entry_id IN ("
        );
        {
            let mut separated = query.separated(", ");
            for entry_id in &entry_ids {
                separated.push_bind(entry_id);
            }
        }
        query.push(") ORDER BY datetime(created_at) ASC");

        let rows = query.build().fetch_all(&self.pool).await?;
        let mut by_entry: HashMap<String, Vec<crate::models::InnerMonologueCandidate>> = HashMap::new();
        for row in rows {
            let entry_id: String = row.try_get("entry_id")?;
            let candidate = crate::models::InnerMonologueCandidate {
                id: row.try_get("id")?,
                entry_id: entry_id.clone(),
                candidate_id: row.try_get("candidate_id").ok(),
                outcome: row.try_get("outcome").ok(),
                suppression_reason: row.try_get("suppression_reason").ok(),
                candidate_json: row.try_get("candidate_json")?,
                created_at: row.try_get("created_at")?,
            };
            by_entry.entry(entry_id).or_default().push(candidate);
        }

        for entry in entries.iter_mut() {
            if let Some(candidates) = by_entry.get(&entry.id) {
                entry.candidates = Some(candidates.clone());
            }
        }

        Ok(entries)
    }

    pub async fn list_inner_monologue_entries_by_stream(
        &self,
        conversation_id: &str,
        stream_type: &str,
        limit: i64,
    ) -> Result<Vec<crate::models::InnerMonologueEntry>, Box<dyn std::error::Error + Send + Sync>> {
        let include_null = stream_type.eq_ignore_ascii_case("DS");
        let rows = if include_null {
            sqlx::query(
                "SELECT id, conversation_id, run_id, dialogue_id, turn_index, speaker, mode, stream_type, thought, descriptors_json, harvest_type, harvest_payload, created_at
                 FROM inner_monologue_entries
                 WHERE conversation_id = ? AND (stream_type = ? OR stream_type IS NULL)
                 ORDER BY datetime(created_at) DESC
                 LIMIT ?",
            )
            .bind(conversation_id)
            .bind(stream_type)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT id, conversation_id, run_id, dialogue_id, turn_index, speaker, mode, stream_type, thought, descriptors_json, harvest_type, harvest_payload, created_at
                 FROM inner_monologue_entries
                 WHERE conversation_id = ? AND stream_type = ?
                 ORDER BY datetime(created_at) DESC
                 LIMIT ?",
            )
            .bind(conversation_id)
            .bind(stream_type)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        };

        let mut entries = Vec::new();
        for row in rows {
            let descriptors = parse_json_list(row.try_get::<Option<String>, _>("descriptors_json").ok().flatten());
            let descriptors = if descriptors.is_empty() { None } else { Some(descriptors) };
            entries.push(crate::models::InnerMonologueEntry {
                id: row.try_get("id")?,
                conversation_id: row.try_get("conversation_id")?,
                run_id: row.try_get("run_id").ok(),
                dialogue_id: row.try_get("dialogue_id").ok(),
                turn_index: row.try_get("turn_index").ok(),
                speaker: row.try_get("speaker").ok(),
                mode: row.try_get("mode")?,
                stream_type: row.try_get("stream_type").ok(),
                thought: row.try_get("thought")?,
                descriptors,
                harvest_type: row.try_get("harvest_type").ok(),
                harvest_payload: row.try_get("harvest_payload").ok(),
                created_at: row.try_get("created_at")?,
                candidates: None,
            });
        }

        Ok(entries)
    }

    pub async fn insert_inner_monologue_entry(
        &self,
        entry: &crate::models::InnerMonologueEntry,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let descriptors_json = entry
            .descriptors
            .as_ref()
            .map(|v| serde_json::to_string(v).unwrap_or_else(|_| "[]".to_string()))
            .unwrap_or_else(|| "[]".to_string());

        sqlx::query(
            "INSERT INTO inner_monologue_entries (id, conversation_id, run_id, dialogue_id, turn_index, speaker, mode, stream_type, thought, descriptors_json, harvest_type, harvest_payload, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&entry.id)
        .bind(&entry.conversation_id)
        .bind(&entry.run_id)
        .bind(&entry.dialogue_id)
        .bind(&entry.turn_index)
        .bind(&entry.speaker)
        .bind(&entry.mode)
        .bind(&entry.stream_type)
        .bind(&entry.thought)
        .bind(descriptors_json)
        .bind(&entry.harvest_type)
        .bind(&entry.harvest_payload)
        .bind(&entry.created_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn insert_inner_monologue_entries_batch(
        &self,
        entries: &[crate::models::InnerMonologueEntry],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut tx = self.pool.begin().await?;
        for entry in entries {
            let descriptors_json = entry
                .descriptors
                .as_ref()
                .map(|v| serde_json::to_string(v).unwrap_or_else(|_| "[]".to_string()))
                .unwrap_or_else(|| "[]".to_string());
            sqlx::query(
                "INSERT INTO inner_monologue_entries (id, conversation_id, run_id, dialogue_id, turn_index, speaker, mode, stream_type, thought, descriptors_json, harvest_type, harvest_payload, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&entry.id)
            .bind(&entry.conversation_id)
            .bind(&entry.run_id)
            .bind(&entry.dialogue_id)
            .bind(&entry.turn_index)
            .bind(&entry.speaker)
            .bind(&entry.mode)
            .bind(&entry.stream_type)
            .bind(&entry.thought)
            .bind(descriptors_json)
            .bind(&entry.harvest_type)
            .bind(&entry.harvest_payload)
            .bind(&entry.created_at)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn insert_inner_monologue_candidate(
        &self,
        candidate: &crate::models::InnerMonologueCandidate,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO inner_monologue_candidates (id, entry_id, candidate_id, outcome, suppression_reason, candidate_json, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&candidate.id)
        .bind(&candidate.entry_id)
        .bind(&candidate.candidate_id)
        .bind(&candidate.outcome)
        .bind(&candidate.suppression_reason)
        .bind(&candidate.candidate_json)
        .bind(&candidate.created_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn insert_inner_monologue_candidates_batch(
        &self,
        candidates: &[crate::models::InnerMonologueCandidate],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if candidates.is_empty() {
            return Ok(());
        }
        let mut tx = self.pool.begin().await?;
        for candidate in candidates {
            sqlx::query(
                "INSERT INTO inner_monologue_candidates (id, entry_id, candidate_id, outcome, suppression_reason, candidate_json, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&candidate.id)
            .bind(&candidate.entry_id)
            .bind(&candidate.candidate_id)
            .bind(&candidate.outcome)
            .bind(&candidate.suppression_reason)
            .bind(&candidate.candidate_json)
            .bind(&candidate.created_at)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn update_inner_monologue_candidate_outcome(
        &self,
        candidate_id: &str,
        outcome: &str,
        suppression_reason: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "UPDATE inner_monologue_candidates SET outcome = ?, suppression_reason = ? WHERE candidate_id = ?",
        )
        .bind(outcome)
        .bind(suppression_reason)
        .bind(candidate_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn insert_counterfactual_simulation(
        &self,
        id: &str,
        conversation_id: &str,
        run_id: Option<&str>,
        candidate_id: Option<&str>,
        candidate_kind: Option<&str>,
        prompt: &str,
        predicted_label: Option<&str>,
        predicted_outcome: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO counterfactual_simulations (id, conversation_id, run_id, candidate_id, candidate_kind, prompt, predicted_label, predicted_outcome, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(id)
        .bind(conversation_id)
        .bind(run_id)
        .bind(candidate_id)
        .bind(candidate_kind)
        .bind(prompt)
        .bind(predicted_label)
        .bind(predicted_outcome)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn mark_counterfactual_observed(
        &self,
        conversation_id: &str,
        observed_label: &str,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let res = sqlx::query(
            "UPDATE counterfactual_simulations
             SET observed_label = ?,
                 matched = CASE
                     WHEN predicted_label IS NOT NULL AND lower(predicted_label) = lower(?) THEN 1
                     ELSE 0
                 END,
                 observed_at = CURRENT_TIMESTAMP
             WHERE id = (
                 SELECT id FROM counterfactual_simulations
                 WHERE conversation_id = ? AND observed_label IS NULL
                 ORDER BY datetime(created_at) DESC
                 LIMIT 1
             )",
        )
        .bind(observed_label)
        .bind(observed_label)
        .bind(conversation_id)
        .execute(&self.pool)
        .await?;
        Ok(res.rows_affected() > 0)
    }

    pub async fn get_weekly_summary(&self, conversation_id: &str) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query("SELECT summary FROM conversation_weekly_summaries WHERE conversation_id = ?")
            .bind(conversation_id)
            .fetch_optional(&self.pool)
            .await?;

        use sqlx::Row;
        Ok(row.map(|r| r.get("summary")))
    }

    pub async fn set_weekly_summary(&self, conversation_id: &str, summary: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO conversation_weekly_summaries (conversation_id, summary, updated_at, version)
             VALUES (?, ?, CURRENT_TIMESTAMP, 1)
             ON CONFLICT(conversation_id)
             DO UPDATE SET summary = excluded.summary, updated_at = CURRENT_TIMESTAMP, version = version + 1"
        )
        .bind(conversation_id)
        .bind(summary)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn get_self_model(&self) -> Result<SelfModel, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT capabilities_json, limitations_json, active_tools_json, memory_health_json,
                    persona_json, persona_daily_delta_json, persona_last_delta_date, goals_json,
                    identity_thread, identity_confidence, identity_uncertainty_note, identity_updated_at,
                    reflection_status_json, reflection_frozen, last_reflection_at,
                    internal_state_summary_json, internal_state_map_version,
                    unified_state_json, unified_state_evidence_json, unified_state_updated_at,
                    updated_at
             FROM self_model WHERE id = 1"
        )
        .fetch_one(&self.pool)
        .await?;

        use sqlx::Row;
        Ok(SelfModel {
            capabilities: serde_json::from_str(&row.get::<String, _>("capabilities_json")).unwrap_or_default(),
            limitations: serde_json::from_str(&row.get::<String, _>("limitations_json")).unwrap_or_default(),
            active_tools: serde_json::from_str(&row.get::<String, _>("active_tools_json")).unwrap_or_default(),
            memory_health: serde_json::from_str(&row.get::<String, _>("memory_health_json")).unwrap_or_default(),
            persona: serde_json::from_str(&row.get::<String, _>("persona_json")).unwrap_or_default(),
            persona_daily_delta: serde_json::from_str(&row.get::<String, _>("persona_daily_delta_json")).unwrap_or_default(),
            persona_last_delta_date: row.get("persona_last_delta_date"),
            goals: serde_json::from_str(&row.get::<String, _>("goals_json")).unwrap_or_default(),
            identity_thread: row.try_get("identity_thread").ok(),
            identity_confidence: row.try_get::<f64, _>("identity_confidence").unwrap_or(0.5) as f32,
            identity_uncertainty_note: row.try_get("identity_uncertainty_note").ok(),
            identity_updated_at: row.try_get("identity_updated_at").ok(),
            reflection_status: serde_json::from_str(&row.get::<String, _>("reflection_status_json")).unwrap_or_default(),
            reflection_frozen: row.get::<i64, _>("reflection_frozen") != 0,
            last_reflection_at: row.get("last_reflection_at"),
            internal_state_summary: serde_json::from_str(&row.get::<String, _>("internal_state_summary_json")).unwrap_or_default(),
            internal_state_map_version: row.try_get("internal_state_map_version").ok(),
            unified_state: serde_json::from_str(&row.get::<String, _>("unified_state_json")).unwrap_or_default(),
            unified_state_evidence: serde_json::from_str(&row.get::<String, _>("unified_state_evidence_json")).unwrap_or_default(),
            unified_state_updated_at: row.try_get("unified_state_updated_at").ok(),
            updated_at: row.get("updated_at"),
        })
    }

    pub async fn ensure_self_model_row(&self) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let exists: Option<i64> = sqlx::query_scalar("SELECT 1 FROM self_model WHERE id = 1")
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten();
        if exists.is_none() {
            sqlx::query("INSERT INTO self_model (id, updated_at) VALUES (1, CURRENT_TIMESTAMP)")
                .execute(&self.pool)
                .await?;
            return Ok(true);
        }
        Ok(false)
    }

    pub async fn set_self_model(&self, model: &SelfModel) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "UPDATE self_model SET
                capabilities_json = ?,
                limitations_json = ?,
                active_tools_json = ?,
                memory_health_json = ?,
                persona_json = ?,
                persona_daily_delta_json = ?,
                persona_last_delta_date = ?,
                goals_json = ?,
                identity_thread = ?,
                identity_confidence = ?,
                identity_uncertainty_note = ?,
                identity_updated_at = ?,
                reflection_status_json = ?,
                reflection_frozen = ?,
                last_reflection_at = ?,
                internal_state_summary_json = ?,
                internal_state_map_version = ?,
                unified_state_json = ?,
                unified_state_evidence_json = ?,
                unified_state_updated_at = ?,
                updated_at = CURRENT_TIMESTAMP
             WHERE id = 1"
        )
        .bind(model.capabilities.to_string())
        .bind(model.limitations.to_string())
        .bind(model.active_tools.to_string())
        .bind(model.memory_health.to_string())
        .bind(model.persona.to_string())
        .bind(model.persona_daily_delta.to_string())
        .bind(model.persona_last_delta_date.clone())
        .bind(model.goals.to_string())
        .bind(model.identity_thread.clone())
        .bind(model.identity_confidence as f64)
        .bind(model.identity_uncertainty_note.clone())
        .bind(model.identity_updated_at.clone())
        .bind(model.reflection_status.to_string())
        .bind(if model.reflection_frozen { 1 } else { 0 })
        .bind(model.last_reflection_at.clone())
        .bind(model.internal_state_summary.to_string())
        .bind(model.internal_state_map_version)
        .bind(model.unified_state.to_string())
        .bind(model.unified_state_evidence.to_string())
        .bind(model.unified_state_updated_at.clone())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn create_self_model_checkpoint(&self, model: &SelfModel, reason: Option<&str>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let snapshot = serde_json::json!({
            "capabilities": model.capabilities.clone(),
            "limitations": model.limitations.clone(),
            "active_tools": model.active_tools.clone(),
            "memory_health": model.memory_health.clone(),
            "persona": model.persona.clone(),
            "persona_daily_delta": model.persona_daily_delta.clone(),
            "persona_last_delta_date": model.persona_last_delta_date.clone(),
            "goals": model.goals.clone(),
            "identity_thread": model.identity_thread.clone(),
            "identity_confidence": model.identity_confidence,
            "identity_uncertainty_note": model.identity_uncertainty_note.clone(),
            "identity_updated_at": model.identity_updated_at.clone(),
            "reflection_status": model.reflection_status.clone(),
            "reflection_frozen": model.reflection_frozen,
            "last_reflection_at": model.last_reflection_at.clone(),
            "internal_state_summary": model.internal_state_summary.clone(),
            "internal_state_map_version": model.internal_state_map_version,
            "unified_state": model.unified_state.clone(),
            "unified_state_evidence": model.unified_state_evidence.clone(),
            "unified_state_updated_at": model.unified_state_updated_at.clone(),
            "updated_at": model.updated_at.clone(),
        })
        .to_string();

        sqlx::query(
            "INSERT INTO self_model_checkpoints (snapshot_json, reason, created_at) VALUES (?, ?, CURRENT_TIMESTAMP)"
        )
        .bind(snapshot)
        .bind(reason)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn create_self_model_checkpoint_with_id(
        &self,
        model: &SelfModel,
        reason: Option<&str>,
    ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
        let snapshot = serde_json::json!({
            "capabilities": model.capabilities.clone(),
            "limitations": model.limitations.clone(),
            "active_tools": model.active_tools.clone(),
            "memory_health": model.memory_health.clone(),
            "persona": model.persona.clone(),
            "persona_daily_delta": model.persona_daily_delta.clone(),
            "persona_last_delta_date": model.persona_last_delta_date.clone(),
            "goals": model.goals.clone(),
            "identity_thread": model.identity_thread.clone(),
            "identity_confidence": model.identity_confidence,
            "identity_uncertainty_note": model.identity_uncertainty_note.clone(),
            "identity_updated_at": model.identity_updated_at.clone(),
            "reflection_status": model.reflection_status.clone(),
            "reflection_frozen": model.reflection_frozen,
            "last_reflection_at": model.last_reflection_at.clone(),
            "internal_state_summary": model.internal_state_summary.clone(),
            "internal_state_map_version": model.internal_state_map_version,
            "unified_state": model.unified_state.clone(),
            "unified_state_evidence": model.unified_state_evidence.clone(),
            "unified_state_updated_at": model.unified_state_updated_at.clone(),
            "updated_at": model.updated_at.clone(),
        })
        .to_string();

        let row = sqlx::query(
            "INSERT INTO self_model_checkpoints (snapshot_json, reason, created_at)
             VALUES (?, ?, CURRENT_TIMESTAMP)
             RETURNING id",
        )
        .bind(snapshot)
        .bind(reason)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get::<i64, _>("id"))
    }

    pub async fn restore_last_self_model_checkpoint(&self) -> Result<Option<SelfModel>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT snapshot_json FROM self_model_checkpoints ORDER BY id DESC LIMIT 1"
        )
        .fetch_optional(&self.pool)
        .await?;

        let Some(row) = row else {
            return Ok(None);
        };

        let snapshot: String = row.get("snapshot_json");
        let parsed: serde_json::Value = serde_json::from_str(&snapshot).unwrap_or_default();
        let model = SelfModel {
            capabilities: parsed.get("capabilities").cloned().unwrap_or_else(|| serde_json::json!([])),
            limitations: parsed.get("limitations").cloned().unwrap_or_else(|| serde_json::json!([])),
            active_tools: parsed.get("active_tools").cloned().unwrap_or_else(|| serde_json::json!([])),
            memory_health: parsed.get("memory_health").cloned().unwrap_or_else(|| serde_json::json!({})),
            persona: parsed.get("persona").cloned().unwrap_or_else(|| serde_json::json!({})),
            persona_daily_delta: parsed.get("persona_daily_delta").cloned().unwrap_or_else(|| serde_json::json!({})),
            persona_last_delta_date: parsed.get("persona_last_delta_date").and_then(|v| v.as_str().map(|s| s.to_string())),
            goals: parsed.get("goals").cloned().unwrap_or_else(|| serde_json::json!([])),
            identity_thread: parsed.get("identity_thread").and_then(|v| v.as_str().map(|s| s.to_string())),
            identity_confidence: parsed.get("identity_confidence").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32,
            identity_uncertainty_note: parsed.get("identity_uncertainty_note").and_then(|v| v.as_str().map(|s| s.to_string())),
            identity_updated_at: parsed.get("identity_updated_at").and_then(|v| v.as_str().map(|s| s.to_string())),
            reflection_status: parsed.get("reflection_status").cloned().unwrap_or_else(|| serde_json::json!({})),
            reflection_frozen: parsed.get("reflection_frozen").and_then(|v| v.as_bool()).unwrap_or(false),
            last_reflection_at: parsed.get("last_reflection_at").and_then(|v| v.as_str().map(|s| s.to_string())),
            internal_state_summary: parsed.get("internal_state_summary").cloned().unwrap_or_else(|| serde_json::json!({})),
            internal_state_map_version: parsed.get("internal_state_map_version").and_then(|v| v.as_i64()),
            unified_state: parsed.get("unified_state").cloned().unwrap_or_else(|| serde_json::json!({})),
            unified_state_evidence: parsed.get("unified_state_evidence").cloned().unwrap_or_else(|| serde_json::json!({})),
            unified_state_updated_at: parsed.get("unified_state_updated_at").and_then(|v| v.as_str().map(|s| s.to_string())),
            updated_at: parsed.get("updated_at").and_then(|v| v.as_str().map(|s| s.to_string())).unwrap_or_else(|| chrono::Utc::now().to_rfc3339()),
        };
        Ok(Some(model))
    }

    pub async fn create_identity_snapshot(
        &self,
        snapshot: &serde_json::Value,
        evidence_event_ids: &[i64],
        invariants: Option<&serde_json::Value>,
        reason: Option<&str>,
        source: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let id = uuid::Uuid::new_v4().to_string();
        let evidence_json = serde_json::to_string(evidence_event_ids).unwrap_or_else(|_| "[]".to_string());
        let invariants_json = invariants.map(|v| v.to_string());
        sqlx::query(
            "INSERT INTO identity_snapshots (id, snapshot_json, evidence_event_ids, invariants_json, reason, source, created_at)
             VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)"
        )
        .bind(&id)
        .bind(snapshot.to_string())
        .bind(evidence_json)
        .bind(invariants_json)
        .bind(reason)
        .bind(source)
        .execute(&self.pool)
        .await?;
        Ok(id)
    }

    pub async fn get_latest_identity_snapshot(
        &self,
    ) -> Result<Option<(String, serde_json::Value, Vec<i64>, Option<serde_json::Value>)>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT id, snapshot_json, evidence_event_ids, invariants_json
             FROM identity_snapshots
             ORDER BY datetime(created_at) DESC
             LIMIT 1"
        )
        .fetch_optional(&self.pool)
        .await?;

        let Some(row) = row else { return Ok(None); };
        let id: String = row.get("id");
        let snapshot_raw: String = row.get("snapshot_json");
        let evidence_raw: String = row.get("evidence_event_ids");
        let invariants_raw: Option<String> = row.try_get("invariants_json").ok();

        let snapshot: serde_json::Value = serde_json::from_str(&snapshot_raw).unwrap_or_default();
        let evidence_event_ids: Vec<i64> = serde_json::from_str(&evidence_raw).unwrap_or_default();
        let invariants: Option<serde_json::Value> = invariants_raw
            .as_deref()
            .and_then(|raw| serde_json::from_str(raw).ok());

        Ok(Some((id, snapshot, evidence_event_ids, invariants)))
    }

    pub async fn get_identity_snapshot(
        &self,
        snapshot_id: &str,
    ) -> Result<Option<(String, serde_json::Value, Vec<i64>, Option<serde_json::Value>)>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT id, snapshot_json, evidence_event_ids, invariants_json
             FROM identity_snapshots
             WHERE id = ?
             LIMIT 1"
        )
        .bind(snapshot_id)
        .fetch_optional(&self.pool)
        .await?;

        let Some(row) = row else { return Ok(None); };
        let id: String = row.get("id");
        let snapshot_raw: String = row.get("snapshot_json");
        let evidence_raw: String = row.get("evidence_event_ids");
        let invariants_raw: Option<String> = row.try_get("invariants_json").ok();

        let snapshot: serde_json::Value = serde_json::from_str(&snapshot_raw).unwrap_or_default();
        let evidence_event_ids: Vec<i64> = serde_json::from_str(&evidence_raw).unwrap_or_default();
        let invariants: Option<serde_json::Value> = invariants_raw
            .as_deref()
            .and_then(|raw| serde_json::from_str(raw).ok());

        Ok(Some((id, snapshot, evidence_event_ids, invariants)))
    }

    pub async fn restore_identity_snapshot(
        &self,
        snapshot_id: Option<&str>,
    ) -> Result<Option<SelfModel>, Box<dyn std::error::Error + Send + Sync>> {
        let row = if let Some(id) = snapshot_id {
            sqlx::query(
                "SELECT snapshot_json FROM identity_snapshots WHERE id = ? LIMIT 1"
            )
            .bind(id)
            .fetch_optional(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT snapshot_json FROM identity_snapshots ORDER BY datetime(created_at) DESC LIMIT 1"
            )
            .fetch_optional(&self.pool)
            .await?
        };

        let Some(row) = row else { return Ok(None); };
        let snapshot_raw: String = row.get("snapshot_json");
        let parsed: serde_json::Value = serde_json::from_str(&snapshot_raw).unwrap_or_default();

        let mut model = self.get_self_model().await?;
        model.identity_thread = parsed.get("identity_thread").and_then(|v| v.as_str().map(|s| s.to_string()));
        model.identity_confidence = parsed.get("identity_confidence").and_then(|v| v.as_f64()).unwrap_or(model.identity_confidence as f64) as f32;
        model.identity_uncertainty_note = parsed.get("identity_uncertainty_note").and_then(|v| v.as_str().map(|s| s.to_string()));
        model.identity_updated_at = parsed.get("identity_updated_at").and_then(|v| v.as_str().map(|s| s.to_string()));

        self.set_self_model(&model).await?;
        Ok(Some(model))
    }

    pub async fn insert_reflection_staging(
        &self,
        proposal_json: &str,
        evidence_event_ids: &[i64],
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let id = Uuid::new_v4().to_string();
        let (proposal_json, filtered) = Self::sanitize_reflection_proposal(proposal_json);
        if filtered > 0 {
            let _ = system_log::log_event(
                &self.pool,
                None,
                "warn",
                "self_reflection",
                None,
                None,
                serde_json::json!({
                    "event": "reflection_staging_sanitized",
                    "filtered": filtered,
                }),
            )
            .await;
        }
        let evidence_json = serde_json::to_string(evidence_event_ids).unwrap_or_else(|_| "[]".to_string());
        sqlx::query(
            "INSERT INTO self_reflection_staging (id, proposal_json, evidence_event_ids, status, created_at)
             VALUES (?, ?, ?, 'pending', CURRENT_TIMESTAMP)",
        )
        .bind(&id)
        .bind(proposal_json)
        .bind(evidence_json)
        .execute(&self.pool)
        .await?;
        Ok(id)
    }

    fn is_reflection_diagnostic_marker(text: &str) -> bool {
        let lowered = text.to_lowercase();
        let markers = [
            "telemetry",
            "tool manifest",
            "tool list",
            "controller state",
            "kv memory",
            "prompt hash",
            "run_id",
            "trace_id",
            "timestamp",
            "latency",
            "module_status",
            "system log",
        ];
        markers.iter().any(|marker| lowered.contains(marker))
    }

    fn sanitize_reflection_write(value: &serde_json::Value) -> bool {
        let obj = match value.as_object() {
            Some(obj) => obj,
            None => return true,
        };
        if let Some(key) = obj.get("key").and_then(|v| v.as_str()) {
            if Self::is_reflection_diagnostic_marker(key) {
                return false;
            }
        }
        if let Some(rel_type) = obj.get("rel_type").and_then(|v| v.as_str()) {
            if Self::is_reflection_diagnostic_marker(rel_type) {
                return false;
            }
        }
        if let Some(value) = obj.get("value").and_then(|v| v.as_str()) {
            if Self::is_reflection_diagnostic_marker(value) {
                return false;
            }
        }
        if let Some(snippet) = obj.get("evidence_snippet").and_then(|v| v.as_str()) {
            if Self::is_reflection_diagnostic_marker(snippet) {
                return false;
            }
        }
        true
    }

    fn sanitize_reflection_proposal(raw: &str) -> (String, usize) {
        let Ok(mut value) = serde_json::from_str::<serde_json::Value>(raw) else {
            return (raw.to_string(), 0);
        };
        let mut filtered = 0usize;
        if let Some(obj) = value.as_object_mut() {
            if let Some(thread) = obj.get("identity_thread").and_then(|v| v.as_str()) {
                if Self::is_reflection_diagnostic_marker(thread) {
                    obj.remove("identity_thread");
                    filtered += 1;
                }
            }
            if let Some(reason) = obj.get("persona_reason").and_then(|v| v.as_str()) {
                if Self::is_reflection_diagnostic_marker(reason) {
                    obj.remove("persona_reason");
                    filtered += 1;
                }
            }
            if let Some(reason) = obj.get("goals_reason").and_then(|v| v.as_str()) {
                if Self::is_reflection_diagnostic_marker(reason) {
                    obj.remove("goals_reason");
                    filtered += 1;
                }
            }
            if let Some(goals) = obj.get_mut("goals").and_then(|v| v.as_array_mut()) {
                let before = goals.len();
                goals.retain(|goal| {
                    goal.as_str()
                        .map(|g| !Self::is_reflection_diagnostic_marker(g))
                        .unwrap_or(true)
                });
                let after = goals.len();
                if goals.is_empty() {
                    obj.remove("goals");
                }
                filtered += before.saturating_sub(after);
            }
            if let Some(writes) = obj.get_mut("self_memory_writes").and_then(|v| v.as_array_mut()) {
                let before = writes.len();
                writes.retain(|write| Self::sanitize_reflection_write(write));
                let after = writes.len();
                if writes.is_empty() {
                    obj.remove("self_memory_writes");
                }
                filtered += before.saturating_sub(after);
            }
        }
        let sanitized = serde_json::to_string(&value).unwrap_or_else(|_| raw.to_string());
        (sanitized, filtered)
    }

    pub async fn list_reflection_staging(
        &self,
        status: Option<&str>,
        limit: i64,
    ) -> Result<Vec<crate::models::ReflectionStagingEntry>, Box<dyn std::error::Error + Send + Sync>> {
        let rows = if let Some(status) = status {
            sqlx::query(
                "SELECT id, proposal_json, evidence_event_ids, status, created_at, reviewed_at, reviewed_by
                 FROM self_reflection_staging
                 WHERE status = ?
                 ORDER BY datetime(created_at) DESC
                 LIMIT ?",
            )
            .bind(status)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT id, proposal_json, evidence_event_ids, status, created_at, reviewed_at, reviewed_by
                 FROM self_reflection_staging
                 ORDER BY datetime(created_at) DESC
                 LIMIT ?",
            )
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        };

        let mut entries = Vec::new();
        for row in rows {
            let evidence_raw: Option<String> = row.try_get("evidence_event_ids").ok();
            let evidence_event_ids = evidence_raw
                .as_deref()
                .and_then(|raw| serde_json::from_str::<Vec<i64>>(raw).ok());
            entries.push(crate::models::ReflectionStagingEntry {
                id: row.get("id"),
                proposal_json: row.get("proposal_json"),
                evidence_event_ids,
                status: row.get("status"),
                created_at: row.get("created_at"),
                reviewed_at: row.try_get("reviewed_at").ok(),
                reviewed_by: row.try_get("reviewed_by").ok(),
            });
        }
        Ok(entries)
    }

    pub async fn get_reflection_staging(
        &self,
        stage_id: &str,
    ) -> Result<Option<crate::models::ReflectionStagingEntry>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT id, proposal_json, evidence_event_ids, status, created_at, reviewed_at, reviewed_by
             FROM self_reflection_staging WHERE id = ?",
        )
        .bind(stage_id)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let evidence_raw: Option<String> = row.try_get("evidence_event_ids").ok();
            let evidence_event_ids = evidence_raw
                .as_deref()
                .and_then(|raw| serde_json::from_str::<Vec<i64>>(raw).ok());
            Ok(Some(crate::models::ReflectionStagingEntry {
                id: row.get("id"),
                proposal_json: row.get("proposal_json"),
                evidence_event_ids,
                status: row.get("status"),
                created_at: row.get("created_at"),
                reviewed_at: row.try_get("reviewed_at").ok(),
                reviewed_by: row.try_get("reviewed_by").ok(),
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn update_reflection_staging_status(
        &self,
        stage_id: &str,
        status: &str,
        reviewer: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "UPDATE self_reflection_staging
             SET status = ?, reviewed_at = CURRENT_TIMESTAMP, reviewed_by = ?
             WHERE id = ?",
        )
        .bind(status)
        .bind(reviewer)
        .bind(stage_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn get_reflection_summary(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let status_row = sqlx::query(
            "SELECT reflection_status_json, last_reflection_at FROM self_model WHERE id = 1"
        )
        .fetch_optional(&self.pool)
        .await?;

        let (status_json, last_reflection_at) = if let Some(row) = status_row {
            let status: String = row.get("reflection_status_json");
            let last_reflection_at: Option<String> = row.get("last_reflection_at");
            (status, last_reflection_at)
        } else {
            (String::new(), None)
        };

        let events = sqlx::query(
            "SELECT snippet, created_at FROM self_evidence_events ORDER BY created_at DESC LIMIT 5"
        )
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();

        let mut lines = Vec::new();
        lines.push("REFLECTION LOG".to_string());
        if let Some(ts) = last_reflection_at {
            lines.push(format!("last_reflection_at: {}", ts));
        }
        if !status_json.trim().is_empty() {
            lines.push(format!("reflection_status: {}", status_json));
        }
        if !events.is_empty() {
            lines.push("recent_changes:".to_string());
            for row in events {
                let snippet: String = row.get("snippet");
                let created_at: String = row.get("created_at");
                lines.push(format!("- {} ({})", snippet, created_at));
            }
        }
        Ok(lines.join("\n"))
    }

    /// Wipe all data to a "new install" state: clears memory, conversations, reminders,
    /// logs, and settings, then re-runs init to seed defaults.
    pub async fn reset_all_data(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Core conversation data
        let _ = sqlx::query("DELETE FROM artifacts").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM messages").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM runs").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM conversation_summaries").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM conversation_live_summaries").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM conversation_weekly_summaries").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM conversation_summary_chunks").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM inner_summaries").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM kernel_states").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM inner_monologue_entries").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM tool_dispatches").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM thread_runs").execute(&self.pool).await;

        // Reminders and KV
        let _ = sqlx::query("DELETE FROM reminders").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM kv_store").execute(&self.pool).await;

        // Episodic + strategy + claims + system logs
        let _ = sqlx::query("DELETE FROM episodic_events").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM system_logs").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM strategy_traces").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM policy_versions").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM memory_claims").execute(&self.pool).await;

        // Self model tables
        let _ = sqlx::query("DELETE FROM self_goal_evidence").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM self_evidence_events").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM self_rel_participants").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM self_rel_beliefs").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM self_fact_beliefs").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM self_beliefs").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM self_model_checkpoints").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM self_model").execute(&self.pool).await;

        // ICS memory tables (children first)
        let _ = sqlx::query("DELETE FROM ics_evidence_events").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_rel_participants").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_embeddings").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_embedding_lsh").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_rel_beliefs").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_fact_beliefs").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_belief_links").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_conflict_set_members").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_conflict_sets").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_entity_sketches").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_entity_degrees").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_entity_degree_cache").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_session_bindings").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_pending_clarify").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_merge_events").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_pending_writes").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_working_set").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_token_aliases").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_role_aliases").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_promotion_maps").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_relation_shapes").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_token_policies").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_predicate_registry").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_beliefs").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_entities").execute(&self.pool).await;

        // Virtual tables / FTS
        let _ = sqlx::query("DELETE FROM ics_entities_fts").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM ics_facts_fts").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM kv_fts").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM episodic_fts").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM conversation_summary_fts").execute(&self.pool).await;

        // Core state + settings
        let _ = sqlx::query("DELETE FROM semantic_core").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM consolidation_state").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM conversations").execute(&self.pool).await;
        let _ = sqlx::query("DELETE FROM settings").execute(&self.pool).await;

        // Re-run init to restore defaults and schema alignment.
        self.init().await?;
        Ok(())
    }

    pub async fn set_key(&self, key: &str, value: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("INSERT INTO kv_store (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP")
            .bind(key)
            .bind(value)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn align_monologue_defaults_once(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let key = "monologue_defaults_aligned_v1";
        if let Ok(Some(existing)) = self.get_key(key).await {
            if existing == "1" {
                return Ok(());
            }
        }
        sqlx::query(
            "UPDATE settings
             SET monologue_interval_seconds = 20,
                 monologue_timeout_secs = 75,
                 monologue_retry_timeout_secs = 25,
                 monologue_max_per_hour = 360,
                 monologue_stabilization_enabled = 1,
                 monologue_surface_enabled = 1,
                 show_monologue_in_chat = 1,
                 enable_introspection = 1,
                 enable_monologue_validator = 1,
                 compact_prompt_enabled = 0,
                 stability_disable_working_hypothesis = 1
             WHERE id = 1",
        )
        .execute(&self.pool)
        .await?;
        let _ = self.set_key(key, "1").await;
        Ok(())
    }

    pub async fn set_key_with_evidence(
        &self,
        key: &str,
        value: &str,
        evidence_event_id: Option<i64>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO kv_store (key, value, evidence_event_id, updated_at)
             VALUES (?, ?, ?, CURRENT_TIMESTAMP)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, evidence_event_id = excluded.evidence_event_id, updated_at = CURRENT_TIMESTAMP",
        )
        .bind(key)
        .bind(value)
        .bind(evidence_event_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn get_key(&self, key: &str) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query("SELECT value FROM kv_store WHERE key = ?")
            .bind(key)
            .fetch_optional(&self.pool)
            .await?;
        
        use sqlx::Row;
        Ok(row.map(|r| r.get("value")))
    }

    pub async fn get_key_with_evidence(
        &self,
        key: &str,
    ) -> Result<Option<(String, Option<i64>)>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query("SELECT value, evidence_event_id FROM kv_store WHERE key = ?")
            .bind(key)
            .fetch_optional(&self.pool)
            .await?;

        use sqlx::Row;
        Ok(row.map(|r| {
            let value: String = r.get("value");
            let evidence_event_id: Option<i64> = r.try_get("evidence_event_id").ok();
            (value, evidence_event_id)
        }))
    }

    pub async fn get_all_keys(&self) -> Result<Vec<(String, String)>, Box<dyn std::error::Error + Send + Sync>> {
        let rows = sqlx::query("SELECT key, value FROM kv_store ORDER BY updated_at DESC")
            .fetch_all(&self.pool)
            .await?;
        
        let mut kv = Vec::new();
        use sqlx::Row;
        for row in rows {
            kv.push((row.get("key"), row.get("value")));
        }
        Ok(kv)
    }

    pub async fn get_recent_keys(
        &self,
        limit: i64,
    ) -> Result<Vec<(String, String)>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = limit.max(1).min(50);
        let rows = sqlx::query("SELECT key, value FROM kv_store ORDER BY updated_at DESC LIMIT ?")
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;

        let mut kv = Vec::new();
        use sqlx::Row;
        for row in rows {
            let value: Option<String> = row.try_get("value").ok();
            kv.push((row.get("key"), value.unwrap_or_default()));
        }
        Ok(kv)
    }

    pub async fn get_parameter_registry(
        &self,
        profile_name: &str,
    ) -> Result<Option<crate::models::ParameterRegistry>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT profile_name, profile_version, payload_json, updated_at
             FROM parameter_registry
             WHERE profile_name = ?
             ORDER BY profile_version DESC
             LIMIT 1",
        )
        .bind(profile_name)
        .fetch_optional(&self.pool)
        .await?;

        use sqlx::Row;
        Ok(row.map(|r| crate::models::ParameterRegistry {
            profile_name: r.get("profile_name"),
            profile_version: r.get("profile_version"),
            payload_json: r.get("payload_json"),
            updated_at: r.get("updated_at"),
        }))
    }

    pub async fn upsert_parameter_registry(
        &self,
        profile_name: &str,
        payload_json: &str,
    ) -> Result<crate::models::ParameterRegistry, Box<dyn std::error::Error + Send + Sync>> {
        let current = self.get_parameter_registry(profile_name).await?;
        let next_version = current
            .as_ref()
            .map(|c| c.profile_version + 1)
            .unwrap_or(1);
        sqlx::query(
            "INSERT INTO parameter_registry (profile_name, profile_version, payload_json, updated_at)
             VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(profile_name)
        .bind(next_version)
        .bind(payload_json)
        .execute(&self.pool)
        .await?;

        Ok(crate::models::ParameterRegistry {
            profile_name: profile_name.to_string(),
            profile_version: next_version,
            payload_json: payload_json.to_string(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        })
    }

    pub async fn list_episodic_events(
        &self,
        conversation_id: Option<&str>,
        limit: i64,
    ) -> Result<Vec<EpisodicEvent>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = limit.max(1).min(200);
        let rows = if let Some(conversation_id) = conversation_id {
            sqlx::query(
                "SELECT id, event_type, payload_json, timestamp, run_id, trace_id, conversation_id, scope, source_type, source_ref, linked_belief_id, linked_artifact_id
                 FROM episodic_events
                 WHERE conversation_id = ?
                 ORDER BY timestamp DESC, rowid DESC
                 LIMIT ?"
            )
            .bind(conversation_id)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT id, event_type, payload_json, timestamp, run_id, trace_id, conversation_id, scope, source_type, source_ref, linked_belief_id, linked_artifact_id
                 FROM episodic_events
                 ORDER BY timestamp DESC, rowid DESC
                 LIMIT ?"
            )
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        };

        use sqlx::Row;
        let mut events = Vec::new();
        for row in rows {
            let payload_raw: String = row.get("payload_json");
            events.push(EpisodicEvent {
                id: row.get("id"),
                event_type: row.get("event_type"),
                payload: serde_json::from_str(&payload_raw).unwrap_or_else(|_| serde_json::json!({})),
                timestamp: row.get("timestamp"),
                run_id: row.try_get("run_id").ok(),
                trace_id: row.try_get("trace_id").ok(),
                conversation_id: row.try_get("conversation_id").ok(),
                scope: row.try_get("scope").ok(),
                source_type: row.get("source_type"),
                source_ref: row.try_get("source_ref").ok(),
                linked_belief_id: row.try_get("linked_belief_id").ok(),
                linked_artifact_id: row.try_get("linked_artifact_id").ok(),
            });
        }
        Ok(events)
    }

    pub async fn list_system_logs(
        &self,
        limit: i64,
        category: Option<&str>,
        level: Option<&str>,
        run_id: Option<&str>,
    ) -> Result<Vec<crate::models::SystemLogEntry>, Box<dyn std::error::Error + Send + Sync>> {
        let mut query = QueryBuilder::new(
            "SELECT id, timestamp, level, category, run_id, trace_id, payload FROM system_logs",
        );
        let mut first = true;
        if let Some(category) = category {
            query.push(if first { " WHERE " } else { " AND " });
            first = false;
            query.push("category = ");
            query.push_bind(category);
        }
        if let Some(level) = level {
            query.push(if first { " WHERE " } else { " AND " });
            first = false;
            query.push("level = ");
            query.push_bind(level);
        }
        if let Some(run_id) = run_id {
            query.push(if first { " WHERE " } else { " AND " });
            query.push("run_id = ");
            query.push_bind(run_id);
        }
        query.push(" ORDER BY datetime(timestamp) DESC LIMIT ");
        query.push_bind(limit.max(1));

        let rows = query.build().fetch_all(&self.pool).await?;
        let mut entries = Vec::new();
        for row in rows {
            let payload_raw: String = row.try_get("payload").unwrap_or_else(|_| "{}".to_string());
            let payload = serde_json::from_str(&payload_raw).unwrap_or_else(|_| serde_json::json!({}));
            entries.push(crate::models::SystemLogEntry {
                id: row.get("id"),
                timestamp: row.get("timestamp"),
                level: row.get("level"),
                category: row.get("category"),
                run_id: row.try_get("run_id").ok(),
                trace_id: row.try_get("trace_id").ok(),
                payload,
            });
        }
        Ok(entries)
    }

    pub async fn insert_decision_report(
        &self,
        run_id: Option<&str>,
        trace_id: Option<&str>,
        conversation_id: Option<&str>,
        report_json: &str,
        evidence_event_ids: &[i64],
    ) -> Result<DecisionReportRecord, Box<dyn std::error::Error + Send + Sync>> {
        let report_id = Uuid::new_v4().to_string();
        let evidence_json = serde_json::to_string(evidence_event_ids)
            .unwrap_or_else(|_| "[]".to_string());

        sqlx::query(
            "INSERT INTO decision_reports
             (report_id, run_id, trace_id, conversation_id, report_json, evidence_event_ids, created_at)
             VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(&report_id)
        .bind(run_id)
        .bind(trace_id)
        .bind(conversation_id)
        .bind(report_json)
        .bind(&evidence_json)
        .execute(&self.pool)
        .await?;

        for evidence_event_id in evidence_event_ids {
            if let Some(evidence_id) = self
                .ensure_evidence_source_for_event(*evidence_event_id)
                .await
            {
                let link_id = format!("link:decision_report:{}:{}", report_id, evidence_id);
                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO evidence_links
                     (link_id, evidence_id, target_type, target_id, relation, created_at)
                     VALUES (?, ?, 'decision_report', ?, 'supports', CURRENT_TIMESTAMP)",
                )
                .bind(&link_id)
                .bind(&evidence_id)
                .bind(&report_id)
                .execute(&self.pool)
                .await;
            }
        }

        Ok(DecisionReportRecord {
            report_id,
            run_id: run_id.map(|v| v.to_string()),
            trace_id: trace_id.map(|v| v.to_string()),
            conversation_id: conversation_id.map(|v| v.to_string()),
            report_json: report_json.to_string(),
            evidence_event_ids: evidence_event_ids.to_vec(),
            created_at: chrono::Utc::now().to_rfc3339(),
        })
    }

    pub async fn list_evidence_lineage(
        &self,
        limit: i64,
    ) -> Result<Vec<EvidenceLineageEntry>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = limit.max(1).min(300);
        let rows = sqlx::query(
            "SELECT evidence_id, source_table, source_id, source_type, source_ref, snippet,
                    weight, confidence, run_id, trace_id, conversation_id, created_at
             FROM evidence_sources
             ORDER BY datetime(created_at) DESC
             LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let mut entries = Vec::new();
        for row in rows {
            let source = EvidenceSource {
                evidence_id: row.get("evidence_id"),
                source_table: row.get("source_table"),
                source_id: row.get("source_id"),
                source_type: row.get("source_type"),
                source_ref: row.try_get("source_ref").ok(),
                snippet: row.try_get("snippet").ok(),
                weight: row.try_get::<f64, _>("weight").ok(),
                confidence: row.try_get::<f64, _>("confidence").ok(),
                run_id: row.try_get("run_id").ok(),
                trace_id: row.try_get("trace_id").ok(),
                conversation_id: row.try_get("conversation_id").ok(),
                created_at: row.get("created_at"),
            };

            let link_rows = sqlx::query(
                "SELECT link_id, evidence_id, target_type, target_id, relation, created_at
                 FROM evidence_links
                 WHERE evidence_id = ?
                 ORDER BY datetime(created_at) DESC",
            )
            .bind(&source.evidence_id)
            .fetch_all(&self.pool)
            .await?;

            let links = link_rows
                .into_iter()
                .map(|link_row| EvidenceLink {
                    link_id: link_row.get("link_id"),
                    evidence_id: link_row.get("evidence_id"),
                    target_type: link_row.get("target_type"),
                    target_id: link_row.get("target_id"),
                    relation: link_row.try_get("relation").ok(),
                    created_at: link_row.get("created_at"),
                })
                .collect::<Vec<_>>();

            entries.push(EvidenceLineageEntry { source, links });
        }

        Ok(entries)
    }

    pub async fn record_outcome_event(
        &self,
        run_id: Option<&str>,
        trace_id: Option<&str>,
        candidate_id: Option<&str>,
        target_type: &str,
        verdict: &str,
        confidence: f64,
        source: &str,
        note: Option<&str>,
        evidence_event_ids: &[i64],
    ) -> Result<String, String> {
        let verdict = verdict.trim();
        if verdict.is_empty() {
            return Err("verdict_missing".to_string());
        }
        let target_type = if target_type.trim().is_empty() {
            "decision_report"
        } else {
            target_type.trim()
        };
        if !outcome_taxonomy::is_valid_outcome(target_type, verdict) {
            let _ = system_log::log_event(
                &self.pool,
                None,
                "warn",
                "outcome",
                run_id,
                trace_id,
                json!({
                    "event": "outcome_taxonomy_violation",
                    "target_type": target_type,
                    "verdict": verdict,
                    "allowed_targets": outcome_taxonomy::allowed_target_types(),
                    "allowed_verdicts": outcome_taxonomy::allowed_verdicts(),
                    "taxonomy_version": outcome_taxonomy::version(),
                }),
            )
            .await;
            return Err("outcome_taxonomy_violation".to_string());
        }
        let outcome_id = Uuid::new_v4().to_string();
        let evidence_json = serde_json::to_string(evidence_event_ids).unwrap_or_else(|_| "[]".to_string());
        sqlx::query(
            "INSERT INTO outcome_events (outcome_id, run_id, trace_id, candidate_id, target_type, verdict, confidence, source, note, evidence_event_ids, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(&outcome_id)
        .bind(run_id)
        .bind(trace_id)
        .bind(candidate_id)
        .bind(target_type)
        .bind(verdict)
        .bind(confidence)
        .bind(source)
        .bind(note)
        .bind(&evidence_json)
        .execute(&self.pool)
        .await
        .map_err(|e| e.to_string())?;

        for evidence_event_id in evidence_event_ids.iter().copied() {
            if let Some(evidence_id) = self.ensure_evidence_source_for_event(evidence_event_id).await {
                let link_id = format!("link:{}:outcome:{}", evidence_id, outcome_id);
                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO evidence_links (link_id, evidence_id, target_type, target_id, relation, created_at)
                     VALUES (?, ?, 'outcome_event', ?, ?, CURRENT_TIMESTAMP)",
                )
                .bind(&link_id)
                .bind(&evidence_id)
                .bind(&outcome_id)
                .bind(verdict)
                .execute(&self.pool)
                .await;
            }
        }

        let verdict_normalized = verdict.trim().to_lowercase();
        let strength = match verdict_normalized.as_str() {
            "confirm" | "success" => 0.85,
            "disconfirm" | "failure" | "error" => 0.2,
            "inconclusive" => 0.45,
            _ => 0.5,
        };
        for evidence_event_id in evidence_event_ids.iter().copied() {
            let _ = sqlx::query(
                "UPDATE ics_evidence_events SET strength = max(strength, ?) WHERE id = ?",
            )
            .bind(strength)
            .bind(evidence_event_id)
            .execute(&self.pool)
            .await;
        }

        if let Some(target_id) = candidate_id.filter(|id| !id.trim().is_empty()) {
            let target_type = target_type.trim();
            let target_type = if target_type.is_empty() { "decision_report" } else { target_type };
            for evidence_event_id in evidence_event_ids.iter().copied() {
                let _ = self
                    .link_evidence_to_target(
                        evidence_event_id,
                        target_type,
                        target_id,
                        "caused_by",
                    )
                    .await;
                if verdict_normalized == "confirm" {
                    let _ = self
                        .link_evidence_to_target(
                            evidence_event_id,
                            target_type,
                            target_id,
                            "validated_by",
                        )
                        .await;
                }
            }
        }

        if verdict.eq_ignore_ascii_case("disconfirm") {
            let _ = self.apply_outcome_disconfirmation(evidence_event_ids).await;
            let _ = self.bump_controller_failure("outcome_disconfirm").await;
        } else if verdict.eq_ignore_ascii_case("confirm") {
            let _ = self.release_controller_failure("outcome_confirm").await;
        }

        let _ = system_log::log_event(
            &self.pool,
            None,
            "info",
            "outcome",
            run_id,
            trace_id,
            json!({
                "event": "outcome_recorded",
                "outcome_id": outcome_id,
                "candidate_id": candidate_id,
                "target_type": target_type,
                "verdict": verdict,
                "confidence": confidence,
                "source": source,
                "note": note,
                "evidence_event_ids": evidence_event_ids,
            }),
        )
        .await;

        Ok(outcome_id)
    }

    pub async fn list_outcome_events(
        &self,
        limit: i64,
    ) -> Result<Vec<OutcomeEvent>, String> {
        let limit = limit.max(1).min(200);
        let rows = sqlx::query(
            "SELECT outcome_id, run_id, trace_id, candidate_id, target_type, verdict, confidence, source, note, evidence_event_ids, created_at
             FROM outcome_events
             ORDER BY datetime(created_at) DESC
             LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| e.to_string())?;

        let mut events = Vec::new();
        for row in rows {
            let raw_ids: String = row.try_get("evidence_event_ids").unwrap_or_else(|_| "[]".to_string());
            let evidence_event_ids = serde_json::from_str::<Vec<i64>>(&raw_ids).unwrap_or_default();
            events.push(OutcomeEvent {
                outcome_id: row.get("outcome_id"),
                run_id: row.try_get("run_id").ok(),
                trace_id: row.try_get("trace_id").ok(),
                candidate_id: row.try_get("candidate_id").ok(),
                target_type: row.get("target_type"),
                verdict: row.get("verdict"),
                confidence: row.try_get::<f64, _>("confidence").unwrap_or(0.5) as f32,
                source: row.get("source"),
                note: row.try_get("note").ok(),
                evidence_event_ids,
                created_at: row.get("created_at"),
            });
        }

        Ok(events)
    }

    async fn apply_outcome_disconfirmation(
        &self,
        evidence_event_ids: &[i64],
    ) -> Result<(), String> {
        if evidence_event_ids.is_empty() {
            return Ok(());
        }
        let placeholders = evidence_event_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let query = format!(
            "SELECT belief_id, 'ics' AS kind FROM ics_evidence_events WHERE id IN ({})
             UNION ALL
             SELECT belief_id, 'self' AS kind FROM self_evidence_events WHERE id IN ({})",
            placeholders, placeholders
        );
        let mut stmt = sqlx::query(&query);
        for id in evidence_event_ids.iter() {
            stmt = stmt.bind(id);
        }
        for id in evidence_event_ids.iter() {
            stmt = stmt.bind(id);
        }
        let rows = stmt.fetch_all(&self.pool).await.map_err(|e| e.to_string())?;
        let belief_count = rows.len();
        for row in rows.iter() {
            let belief_id: i64 = row.get("belief_id");
            let kind: String = row.get("kind");
            let table = if kind == "self" { "self_beliefs" } else { "ics_beliefs" };
            let _ = sqlx::query(&format!(
                "UPDATE {table}
                 SET confidence = MAX(confidence - 0.15, 0.05),
                     last_validated_at = CURRENT_TIMESTAMP
                 WHERE id = ?",
                table = table
            ))
            .bind(belief_id)
            .execute(&self.pool)
            .await;
        }

        let _ = system_log::log_event(
            &self.pool,
            None,
            "warn",
            "outcome",
            None,
            None,
            json!({
                "event": "outcome_disconfirm_applied",
                "belief_count": belief_count,
            }),
        )
        .await;

        Ok(())
    }

    pub async fn clear_live_summary(
        &self,
        conversation_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("DELETE FROM conversation_live_summaries WHERE conversation_id = ?")
            .bind(conversation_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn bump_controller_failure(&self, reason: &str) -> Result<(), String> {
        let mut state = self
            .get_controller_state()
            .await
            .map_err(|e| e.to_string())?
            .unwrap_or_default();
        state.failure_streak = state.failure_streak.saturating_add(1);
        state.verification_needed = true;
        state.outcome_quality = Some(0.0);
        state.notes.push(reason.to_string());
        self.set_controller_state(&state).await.map_err(|e| e.to_string())?;
        Ok(())
    }

    async fn release_controller_failure(&self, reason: &str) -> Result<(), String> {
        let mut state = self
            .get_controller_state()
            .await
            .map_err(|e| e.to_string())?
            .unwrap_or_default();
        state.failure_streak = state.failure_streak.saturating_sub(1);
        state.outcome_quality = Some(1.0);
        state.notes.push(reason.to_string());
        self.set_controller_state(&state).await.map_err(|e| e.to_string())?;
        Ok(())
    }

    async fn ensure_evidence_source_for_event(&self, evidence_event_id: i64) -> Option<String> {
        let ics_id = format!("ics:{evidence_event_id}");
        if let Ok(Some(existing)) = sqlx::query_scalar::<_, String>(
            "SELECT evidence_id FROM evidence_sources WHERE evidence_id = ?",
        )
        .bind(&ics_id)
        .fetch_optional(&self.pool)
        .await
        {
            return Some(existing);
        }

        if let Ok(Some(row)) = sqlx::query(
            "SELECT source_type, source_ref, snippet, weight, created_at
             FROM ics_evidence_events WHERE id = ?",
        )
        .bind(evidence_event_id)
        .fetch_optional(&self.pool)
        .await
        {
            let source_type: String = row.try_get("source_type").unwrap_or_else(|_| "unknown".to_string());
            let source_ref: Option<String> = row.try_get("source_ref").ok();
            let snippet: Option<String> = row.try_get("snippet").ok();
            let weight: Option<f64> = row.try_get::<f64, _>("weight").ok();
            let created_at: Option<String> = row.try_get("created_at").ok();
            let _ = sqlx::query(
                "INSERT OR IGNORE INTO evidence_sources
                 (evidence_id, source_table, source_id, source_type, source_ref, snippet, weight, created_at)
                 VALUES (?, 'ics_evidence_events', ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))",
            )
            .bind(&ics_id)
            .bind(evidence_event_id.to_string())
            .bind(source_type)
            .bind(source_ref)
            .bind(snippet)
            .bind(weight)
            .bind(created_at)
            .execute(&self.pool)
            .await;
            return Some(ics_id);
        }

        let self_id = format!("self:{evidence_event_id}");
        if let Ok(Some(existing)) = sqlx::query_scalar::<_, String>(
            "SELECT evidence_id FROM evidence_sources WHERE evidence_id = ?",
        )
        .bind(&self_id)
        .fetch_optional(&self.pool)
        .await
        {
            return Some(existing);
        }

        if let Ok(Some(row)) = sqlx::query(
            "SELECT source_type, snippet, weight, created_at
             FROM self_evidence_events WHERE id = ?",
        )
        .bind(evidence_event_id)
        .fetch_optional(&self.pool)
        .await
        {
            let source_type: String = row.try_get("source_type").unwrap_or_else(|_| "unknown".to_string());
            let snippet: Option<String> = row.try_get("snippet").ok();
            let weight: Option<f64> = row.try_get::<f64, _>("weight").ok();
            let created_at: Option<String> = row.try_get("created_at").ok();
            let _ = sqlx::query(
                "INSERT OR IGNORE INTO evidence_sources
                 (evidence_id, source_table, source_id, source_type, source_ref, snippet, weight, created_at)
                 VALUES (?, 'self_evidence_events', ?, ?, NULL, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))",
            )
            .bind(&self_id)
            .bind(evidence_event_id.to_string())
            .bind(source_type)
            .bind(snippet)
            .bind(weight)
            .bind(created_at)
            .execute(&self.pool)
            .await;
            return Some(self_id);
        }

        None
    }

    pub async fn link_evidence_to_tool_dispatch(
        &self,
        evidence_event_id: i64,
        action_id: &str,
    ) -> Result<(), String> {
        let Some(evidence_id) = self.ensure_evidence_source_for_event(evidence_event_id).await else {
            return Ok(());
        };
        let link_id = format!("link:{}:tool_dispatch:{}", evidence_id, action_id);
        sqlx::query(
            "INSERT OR IGNORE INTO evidence_links (link_id, evidence_id, target_type, target_id, relation, created_at)
             VALUES (?, ?, 'tool_dispatch', ?, 'produced_by', CURRENT_TIMESTAMP)",
        )
        .bind(&link_id)
        .bind(&evidence_id)
        .bind(action_id)
        .execute(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub async fn link_evidence_to_target(
        &self,
        evidence_event_id: i64,
        target_type: &str,
        target_id: &str,
        relation: &str,
    ) -> Result<(), String> {
        let target_type = target_type.trim();
        let target_id = target_id.trim();
        if target_type.is_empty() || target_id.is_empty() {
            return Ok(());
        }
        let Some(evidence_id) = self.ensure_evidence_source_for_event(evidence_event_id).await else {
            return Ok(());
        };
        let relation = relation.trim();
        let relation = if relation.is_empty() { "supports" } else { relation };
        let link_id = format!("link:{}:{}:{}:{}", evidence_id, target_type, target_id, relation);
        sqlx::query(
            "INSERT OR IGNORE INTO evidence_links (link_id, evidence_id, target_type, target_id, relation, created_at)
             VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(&link_id)
        .bind(&evidence_id)
        .bind(target_type)
        .bind(target_id)
        .bind(relation)
        .execute(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub async fn set_action_proposal_state(
        &self,
        proposal_id: &str,
        plan_state: &str,
    ) -> Result<(), String> {
        if proposal_id.trim().is_empty() {
            return Ok(());
        }
        sqlx::query(
            "UPDATE action_proposals SET plan_state = ? WHERE proposal_id = ?",
        )
        .bind(plan_state)
        .bind(proposal_id)
        .execute(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub async fn seed_system_controls_defaults(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let existing: HashSet<String> = sqlx::query_scalar(
            "SELECT subsystem_id FROM system_controls",
        )
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default()
        .into_iter()
        .collect();

        for def in system_controls::registry() {
            if existing.contains(def.id) {
                continue;
            }
            sqlx::query(
                "INSERT INTO system_controls
                 (control_id, subsystem_id, mode, updated_at, updated_by, reason)
                 VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?)",
            )
            .bind(def.id)
            .bind(def.id)
            .bind(def.default_mode)
            .bind("bootstrap")
            .bind("default_seed")
            .execute(&self.pool)
            .await?;
        }
        Ok(())
    }

    pub async fn get_system_controls(
        &self,
    ) -> Result<Vec<SystemControlEntry>, Box<dyn std::error::Error + Send + Sync>> {
        let rows = sqlx::query(
            "SELECT control_id, subsystem_id, mode, value_json, updated_at, updated_by, reason
             FROM system_controls
             ORDER BY subsystem_id ASC",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(SystemControlEntry {
                control_id: row.get("control_id"),
                subsystem_id: row.get("subsystem_id"),
                mode: row.get("mode"),
                value_json: row.try_get("value_json").ok(),
                updated_at: row.get("updated_at"),
                updated_by: row.try_get("updated_by").ok(),
                reason: row.try_get("reason").ok(),
            });
        }
        Ok(entries)
    }

    pub async fn set_system_control(
        &self,
        subsystem_id: &str,
        mode: &str,
        value_json: Option<String>,
        updated_by: Option<String>,
        reason: Option<String>,
    ) -> Result<SystemControlEntry, Box<dyn std::error::Error + Send + Sync>> {
        let previous_mode: Option<String> = sqlx::query_scalar(
            "SELECT mode FROM system_controls WHERE subsystem_id = ?",
        )
        .bind(subsystem_id)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten();

        let control_id = subsystem_id.to_string();
        sqlx::query(
            "INSERT OR REPLACE INTO system_controls
             (control_id, subsystem_id, mode, value_json, updated_at, updated_by, reason)
             VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)",
        )
        .bind(&control_id)
        .bind(subsystem_id)
        .bind(mode)
        .bind(value_json.clone())
        .bind(updated_by.clone())
        .bind(reason.clone())
        .execute(&self.pool)
        .await?;

        let event_id = Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT INTO system_control_events
             (event_id, subsystem_id, previous_mode, new_mode, value_json, actor, reason, status, timestamp)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(&event_id)
        .bind(subsystem_id)
        .bind(previous_mode.clone())
        .bind(mode)
        .bind(value_json.clone())
        .bind(updated_by.clone())
        .bind(reason.clone())
        .bind("applied")
        .execute(&self.pool)
        .await?;

        let row = sqlx::query(
            "SELECT control_id, subsystem_id, mode, value_json, updated_at, updated_by, reason
             FROM system_controls WHERE subsystem_id = ?",
        )
        .bind(subsystem_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(SystemControlEntry {
            control_id: row.get("control_id"),
            subsystem_id: row.get("subsystem_id"),
            mode: row.get("mode"),
            value_json: row.try_get("value_json").ok(),
            updated_at: row.get("updated_at"),
            updated_by: row.try_get("updated_by").ok(),
            reason: row.try_get("reason").ok(),
        })
    }

    pub async fn insert_system_control_event(
        &self,
        subsystem_id: &str,
        previous_mode: Option<String>,
        new_mode: &str,
        value_json: Option<String>,
        actor: Option<String>,
        reason: Option<String>,
        status: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let event_id = Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT INTO system_control_events
             (event_id, subsystem_id, previous_mode, new_mode, value_json, actor, reason, status, timestamp)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(&event_id)
        .bind(subsystem_id)
        .bind(previous_mode)
        .bind(new_mode)
        .bind(value_json)
        .bind(actor)
        .bind(reason)
        .bind(status)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_system_control_events(
        &self,
        limit: i64,
    ) -> Result<Vec<SystemControlEvent>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = limit.max(1).min(500);
        let rows = sqlx::query(
            "SELECT event_id, subsystem_id, previous_mode, new_mode, value_json, actor, reason, status, timestamp
             FROM system_control_events
             ORDER BY datetime(timestamp) DESC, rowid DESC
             LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        let mut events = Vec::new();
        for row in rows {
            events.push(SystemControlEvent {
                event_id: row.get("event_id"),
                subsystem_id: row.get("subsystem_id"),
                previous_mode: row.try_get("previous_mode").ok(),
                new_mode: row.get("new_mode"),
                value_json: row.try_get("value_json").ok(),
                actor: row.try_get("actor").ok(),
                reason: row.try_get("reason").ok(),
                status: row.try_get("status").ok(),
                timestamp: row.get("timestamp"),
            });
        }
        Ok(events)
    }

    pub async fn insert_system_health_snapshot(
        &self,
        snapshot_id: &str,
        timestamp: &str,
        run_id: Option<&str>,
        trace_id: Option<&str>,
        metrics_json: &str,
        subsystem_states_json: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO system_health_snapshots
             (snapshot_id, timestamp, run_id, trace_id, metrics_json, subsystem_states_json)
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(snapshot_id)
        .bind(timestamp)
        .bind(run_id)
        .bind(trace_id)
        .bind(metrics_json)
        .bind(subsystem_states_json)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_system_health_snapshots(
        &self,
        limit: i64,
    ) -> Result<Vec<SystemHealthSnapshot>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = limit.max(1).min(500);
        let rows = sqlx::query(
            "SELECT snapshot_id, timestamp, run_id, trace_id, metrics_json, subsystem_states_json
             FROM system_health_snapshots
             ORDER BY datetime(timestamp) DESC, rowid DESC
             LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        let mut snapshots = Vec::new();
        for row in rows {
            snapshots.push(SystemHealthSnapshot {
                snapshot_id: row.get("snapshot_id"),
                timestamp: row.get("timestamp"),
                run_id: row.try_get("run_id").ok(),
                trace_id: row.try_get("trace_id").ok(),
                metrics_json: row.get("metrics_json"),
                subsystem_states_json: row.get("subsystem_states_json"),
            });
        }
        Ok(snapshots)
    }

    pub async fn insert_baseline_metrics(
        &self,
        baseline_id: &str,
        window_minutes: i64,
        window_start: &str,
        window_end: &str,
        metrics_json: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO baseline_metrics
             (baseline_id, window_minutes, window_start, window_end, metrics_json)
             VALUES (?, ?, ?, ?, ?)",
        )
        .bind(baseline_id)
        .bind(window_minutes)
        .bind(window_start)
        .bind(window_end)
        .bind(metrics_json)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_baseline_metrics(
        &self,
        limit: i64,
    ) -> Result<Vec<BaselineMetricsSnapshot>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = limit.max(1).min(200);
        let rows = sqlx::query(
            "SELECT baseline_id, window_minutes, window_start, window_end, metrics_json, created_at
             FROM baseline_metrics
             ORDER BY datetime(created_at) DESC, rowid DESC
             LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        let mut snapshots = Vec::new();
        for row in rows {
            snapshots.push(BaselineMetricsSnapshot {
                baseline_id: row.get("baseline_id"),
                window_minutes: row.get::<i64, _>("window_minutes"),
                window_start: row.get("window_start"),
                window_end: row.get("window_end"),
                metrics_json: row.get("metrics_json"),
                created_at: row.get("created_at"),
            });
        }
        Ok(snapshots)
    }

    pub async fn get_latest_baseline_metrics(
        &self,
    ) -> Result<Option<BaselineMetricsSnapshot>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT baseline_id, window_minutes, window_start, window_end, metrics_json, created_at
             FROM baseline_metrics
             ORDER BY datetime(created_at) DESC, rowid DESC
             LIMIT 1",
        )
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|row| BaselineMetricsSnapshot {
            baseline_id: row.get("baseline_id"),
            window_minutes: row.get::<i64, _>("window_minutes"),
            window_start: row.get("window_start"),
            window_end: row.get("window_end"),
            metrics_json: row.get("metrics_json"),
            created_at: row.get("created_at"),
        }))
    }

    pub async fn insert_recommendation_event(
        &self,
        recommendation_id: &str,
        conversation_id: Option<&str>,
        kind: &str,
        status: &str,
        snapshot_id: Option<&str>,
        action_json: Option<&str>,
        gate_json: Option<&str>,
        recovery_metric: Option<&str>,
        recovery_target: Option<f64>,
        baseline_value: Option<f64>,
        resolved_value: Option<f64>,
        time_to_recovery_ms: Option<i64>,
    ) -> Result<RecommendationEvent, Box<dyn std::error::Error + Send + Sync>> {
        let event_id = Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT INTO recommendation_events
             (event_id, recommendation_id, conversation_id, kind, status, snapshot_id, action_json, gate_json, recovery_metric, recovery_target, baseline_value, resolved_value, time_to_recovery_ms)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&event_id)
        .bind(recommendation_id)
        .bind(conversation_id)
        .bind(kind)
        .bind(status)
        .bind(snapshot_id)
        .bind(action_json)
        .bind(gate_json)
        .bind(recovery_metric)
        .bind(recovery_target)
        .bind(baseline_value)
        .bind(resolved_value)
        .bind(time_to_recovery_ms)
        .execute(&self.pool)
        .await?;

        let created_at = Utc::now().to_rfc3339();
        Ok(RecommendationEvent {
            event_id,
            recommendation_id: recommendation_id.to_string(),
            conversation_id: conversation_id.map(|s| s.to_string()),
            kind: kind.to_string(),
            status: status.to_string(),
            snapshot_id: snapshot_id.map(|s| s.to_string()),
            action_json: action_json.map(|s| s.to_string()),
            gate_json: gate_json.map(|s| s.to_string()),
            recovery_metric: recovery_metric.map(|s| s.to_string()),
            recovery_target,
            baseline_value,
            resolved_value,
            time_to_recovery_ms,
            created_at,
        })
    }

    pub async fn list_recommendation_events(
        &self,
        limit: i64,
    ) -> Result<Vec<RecommendationEvent>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = limit.max(1).min(500);
        let rows = sqlx::query(
            "SELECT event_id, recommendation_id, conversation_id, kind, status, snapshot_id, action_json, gate_json, recovery_metric, recovery_target, baseline_value, resolved_value, time_to_recovery_ms, created_at
             FROM recommendation_events
             ORDER BY datetime(created_at) DESC, rowid DESC
             LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        let mut events = Vec::new();
        for row in rows {
            events.push(RecommendationEvent {
                event_id: row.get("event_id"),
                recommendation_id: row.get("recommendation_id"),
                conversation_id: row.try_get("conversation_id").ok(),
                kind: row.get("kind"),
                status: row.get("status"),
                snapshot_id: row.try_get("snapshot_id").ok(),
                action_json: row.try_get("action_json").ok(),
                gate_json: row.try_get("gate_json").ok(),
                recovery_metric: row.try_get("recovery_metric").ok(),
                recovery_target: row.try_get("recovery_target").ok(),
                baseline_value: row.try_get("baseline_value").ok(),
                resolved_value: row.try_get("resolved_value").ok(),
                time_to_recovery_ms: row.try_get("time_to_recovery_ms").ok(),
                created_at: row.get("created_at"),
            });
        }
        Ok(events)
    }

    pub async fn get_latest_recommendation_event(
        &self,
        recommendation_id: &str,
    ) -> Result<Option<RecommendationEvent>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query(
            "SELECT event_id, recommendation_id, conversation_id, kind, status, snapshot_id, action_json, gate_json, recovery_metric, recovery_target, baseline_value, resolved_value, time_to_recovery_ms, created_at
             FROM recommendation_events
             WHERE recommendation_id = ?
             ORDER BY datetime(created_at) DESC, rowid DESC
             LIMIT 1",
        )
        .bind(recommendation_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|row| RecommendationEvent {
            event_id: row.get("event_id"),
            recommendation_id: row.get("recommendation_id"),
            conversation_id: row.try_get("conversation_id").ok(),
            kind: row.get("kind"),
            status: row.get("status"),
            snapshot_id: row.try_get("snapshot_id").ok(),
            action_json: row.try_get("action_json").ok(),
            gate_json: row.try_get("gate_json").ok(),
            recovery_metric: row.try_get("recovery_metric").ok(),
            recovery_target: row.try_get("recovery_target").ok(),
            baseline_value: row.try_get("baseline_value").ok(),
            resolved_value: row.try_get("resolved_value").ok(),
            time_to_recovery_ms: row.try_get("time_to_recovery_ms").ok(),
            created_at: row.get("created_at"),
        }))
    }

    pub async fn search_conversation_summary_chunks(
        &self,
        conversation_id: &str,
        query: Option<&str>,
        limit: i64,
    ) -> Result<Vec<ConversationSummaryChunk>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = limit.max(1).min(50);
        let query = query.and_then(|q| {
            let trimmed = q.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        });
        let has_query = query.is_some();

        let mut qb = QueryBuilder::new(
            "SELECT c.id, c.chunk_id, c.summary, c.start_ts, c.end_ts
             FROM conversation_summary_chunks c ",
        );
        if has_query {
            qb.push("JOIN conversation_summary_fts f ON f.rowid = c.rowid ");
        }
        qb.push("WHERE c.conversation_id = ").push_bind(conversation_id).push(' ');
        if let Some(query) = query {
            qb.push("AND f MATCH ").push_bind(query).push(' ');
        }

        if has_query {
            qb.push(
                "ORDER BY bm25(f) ASC, datetime(COALESCE(c.end_ts, c.created_at)) DESC, c.rowid DESC ",
            );
        } else {
            qb.push("ORDER BY datetime(COALESCE(c.end_ts, c.created_at)) DESC, c.rowid DESC ");
        }
        qb.push("LIMIT ").push_bind(limit);

        let rows = qb.build().fetch_all(&self.pool).await?;
        use sqlx::Row;
        let mut chunks = Vec::new();
        for row in rows {
            chunks.push(ConversationSummaryChunk {
                id: row.get("id"),
                chunk_id: row.get("chunk_id"),
                summary: row.get("summary"),
                start_ts: row.try_get("start_ts").ok(),
                end_ts: row.try_get("end_ts").ok(),
            });
        }
        Ok(chunks)
    }

    pub async fn search_episodic_events(
        &self,
        query: Option<&str>,
        conversation_id: Option<&str>,
        run_id: Option<&str>,
        event_type: Option<&str>,
        start_time: Option<&str>,
        end_time: Option<&str>,
        limit: i64,
    ) -> Result<Vec<EpisodicEvent>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = limit.max(1).min(200);
        let query = query.and_then(|q| {
            let trimmed = q.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        });
        let has_query = query.is_some();

        let mut qb = QueryBuilder::new(
            "SELECT e.id, e.event_type, e.payload_json, e.timestamp, e.run_id, e.trace_id, e.conversation_id, e.scope, e.source_type, e.source_ref, e.linked_belief_id, e.linked_artifact_id
             FROM episodic_events e "
        );
        if has_query {
            qb.push("JOIN episodic_fts f ON f.rowid = e.rowid ");
        }
        qb.push("WHERE 1=1 ");

        if let Some(query) = query {
            qb.push("AND f MATCH ").push_bind(query);
            qb.push(" ");
        }
        if let Some(conversation_id) = conversation_id {
            qb.push("AND e.conversation_id = ").push_bind(conversation_id);
            qb.push(" ");
        }
        if let Some(run_id) = run_id {
            qb.push("AND e.run_id = ").push_bind(run_id);
            qb.push(" ");
        }
        if let Some(event_type) = event_type {
            qb.push("AND e.event_type = ").push_bind(event_type);
            qb.push(" ");
        }
        if let Some(start_time) = start_time {
            qb.push("AND e.timestamp >= ").push_bind(start_time);
            qb.push(" ");
        }
        if let Some(end_time) = end_time {
            qb.push("AND e.timestamp <= ").push_bind(end_time);
            qb.push(" ");
        }

        if has_query {
            qb.push("ORDER BY (bm25(f) * (1.0 + (julianday('now') - julianday(e.timestamp)) * 0.05)) ASC, e.timestamp DESC, e.rowid DESC ");
        } else {
            qb.push("ORDER BY e.timestamp DESC, e.rowid DESC ");
        }
        qb.push("LIMIT ").push_bind(limit);

        let rows = qb.build().fetch_all(&self.pool).await?;
        use sqlx::Row;
        let mut events = Vec::new();
        for row in rows {
            let payload_raw: String = row.get("payload_json");
            events.push(EpisodicEvent {
                id: row.get("id"),
                event_type: row.get("event_type"),
                payload: serde_json::from_str(&payload_raw).unwrap_or_else(|_| serde_json::json!({})),
                timestamp: row.get("timestamp"),
                run_id: row.try_get("run_id").ok(),
                trace_id: row.try_get("trace_id").ok(),
                conversation_id: row.try_get("conversation_id").ok(),
                scope: row.try_get("scope").ok(),
                source_type: row.get("source_type"),
                source_ref: row.try_get("source_ref").ok(),
                linked_belief_id: row.try_get("linked_belief_id").ok(),
                linked_artifact_id: row.try_get("linked_artifact_id").ok(),
            });
        }
        Ok(events)
    }

    pub async fn list_relation_shape_missing(
        &self,
        limit: i64,
    ) -> Result<Vec<EpisodicEvent>, Box<dyn std::error::Error + Send + Sync>> {
        self.search_episodic_events(None, None, None, Some("relation_shape_missing"), None, None, limit)
            .await
    }

    pub async fn record_strategy_trace(
        &self,
        features: serde_json::Value,
        strategy_label: &str,
        outcome: &str,
        success_score: Option<f64>,
        run_id: Option<&str>,
        conversation_id: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let id = uuid::Uuid::new_v4().to_string();
        let payload = features.to_string();
        sqlx::query(
            "INSERT INTO strategy_traces (id, features_json, strategy_label, outcome, success_score, run_id, conversation_id, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)"
        )
        .bind(&id)
        .bind(payload)
        .bind(strategy_label)
        .bind(outcome)
        .bind(success_score)
        .bind(run_id)
        .bind(conversation_id)
        .execute(&self.pool)
        .await?;
        Ok(id)
    }

    pub async fn list_strategy_traces(
        &self,
        limit: i64,
    ) -> Result<Vec<StrategyTrace>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = limit.max(1).min(200);
        let rows = sqlx::query(
            "SELECT id, features_json, strategy_label, outcome, success_score, run_id, conversation_id, created_at
             FROM strategy_traces
             ORDER BY created_at DESC
             LIMIT ?"
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        use sqlx::Row;
        let mut traces = Vec::new();
        for row in rows {
            let features_raw: String = row.get("features_json");
            traces.push(StrategyTrace {
                id: row.get("id"),
                features: serde_json::from_str(&features_raw).unwrap_or_else(|_| serde_json::json!({})),
                strategy_label: row.get("strategy_label"),
                outcome: row.get("outcome"),
                success_score: row.try_get("success_score").ok(),
                run_id: row.try_get("run_id").ok(),
                conversation_id: row.try_get("conversation_id").ok(),
                created_at: row.get("created_at"),
            });
        }
        Ok(traces)
    }

    pub async fn create_policy_version(
        &self,
        label: &str,
        payload: serde_json::Value,
        parent_id: Option<&str>,
        reason: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let id = uuid::Uuid::new_v4().to_string();
        let payload_json = payload.to_string();
        sqlx::query(
            "INSERT INTO policy_versions (id, label, payload_json, parent_id, reason, created_at)
             VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)"
        )
        .bind(&id)
        .bind(label)
        .bind(payload_json)
        .bind(parent_id)
        .bind(reason)
        .execute(&self.pool)
        .await?;
        Ok(id)
    }

    pub async fn list_policy_versions(
        &self,
        limit: i64,
    ) -> Result<Vec<PolicyVersion>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = limit.max(1).min(200);
        let rows = sqlx::query(
            "SELECT id, label, payload_json, parent_id, reason, created_at
             FROM policy_versions
             ORDER BY created_at DESC
             LIMIT ?"
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        use sqlx::Row;
        let mut versions = Vec::new();
        for row in rows {
            let payload_raw: String = row.get("payload_json");
            versions.push(PolicyVersion {
                id: row.get("id"),
                label: row.get("label"),
                payload: serde_json::from_str(&payload_raw).unwrap_or_else(|_| serde_json::json!({})),
                parent_id: row.try_get("parent_id").ok(),
                reason: row.try_get("reason").ok(),
                created_at: row.get("created_at"),
            });
        }
        Ok(versions)
    }

    pub async fn create_memory_claim(
        &self,
        kind: &str,
        scope: &str,
        claim_text: &str,
        source_type: &str,
        source_ref: Option<&str>,
        episodic_event_id: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let id = uuid::Uuid::new_v4().to_string();
        let session_id = parse_scope_str(scope).and_then(|parsed| match parsed {
            Scope::Context(id) => Some(id),
            _ => None,
        });
        sqlx::query(
            "INSERT INTO memory_claims (id, kind, scope, session_id, claim_text, status, source_type, source_ref, episodic_event_id, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
        .bind(&id)
        .bind(kind)
        .bind(scope)
        .bind(session_id.as_deref())
        .bind(claim_text)
        .bind(source_type)
        .bind(source_ref)
        .bind(episodic_event_id)
        .execute(&self.pool)
        .await?;
        Ok(id)
    }

    pub async fn list_memory_claims(
        &self,
        status: Option<&str>,
        limit: i64,
    ) -> Result<Vec<MemoryClaim>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = limit.max(1).min(200);
        let rows = if let Some(status) = status {
            sqlx::query(
                "SELECT id, kind, scope, session_id, claim_text, rel_type_raw, rel_type_norm, rel_type_id, status, source_type, source_ref,
                        conflict_topic_key, conflict_reason, evaluated_at, decision_reason,
                        episodic_event_id, created_at, updated_at
                 FROM memory_claims
                 WHERE status = ?
                 ORDER BY created_at DESC
                 LIMIT ?"
            )
            .bind(status)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT id, kind, scope, session_id, claim_text, rel_type_raw, rel_type_norm, rel_type_id, status, source_type, source_ref,
                        conflict_topic_key, conflict_reason, evaluated_at, decision_reason,
                        episodic_event_id, created_at, updated_at
                 FROM memory_claims
                 ORDER BY created_at DESC
                 LIMIT ?"
            )
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        };

        use sqlx::Row;
        let mut claims = Vec::new();
        for row in rows {
            claims.push(MemoryClaim {
                id: row.get("id"),
                kind: row.get("kind"),
                scope: row.get("scope"),
                session_id: row.try_get("session_id").ok(),
                claim_text: row.get("claim_text"),
                rel_type_raw: row.try_get("rel_type_raw").ok(),
                rel_type_norm: row.try_get("rel_type_norm").ok(),
                rel_type_id: row.try_get("rel_type_id").ok(),
                status: row.get("status"),
                source_type: row.get("source_type"),
                source_ref: row.try_get("source_ref").ok(),
                conflict_topic_key: row.try_get("conflict_topic_key").ok(),
                conflict_reason: row.try_get("conflict_reason").ok(),
                evaluated_at: row.try_get("evaluated_at").ok(),
                decision_reason: row.try_get("decision_reason").ok(),
                episodic_event_id: row.try_get("episodic_event_id").ok(),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
            });
        }
        Ok(claims)
    }

    pub async fn update_memory_claim_status(
        &self,
        claim_id: &str,
        status: &str,
        decision_reason: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "UPDATE memory_claims
             SET status = ?,
                 decision_reason = ?,
                 evaluated_at = CURRENT_TIMESTAMP,
                 updated_at = CURRENT_TIMESTAMP
             WHERE id = ?"
        )
        .bind(status)
        .bind(decision_reason)
        .bind(claim_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_episodic_events_for_ics_belief(
        &self,
        belief_id: i64,
        limit: i64,
    ) -> Result<Vec<EpisodicEvent>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = limit.max(1).min(200);
        let rows = sqlx::query(
            "SELECT e.id, e.event_type, e.payload_json, e.timestamp, e.run_id, e.trace_id, e.conversation_id, e.scope, e.source_type, e.source_ref, e.linked_belief_id, e.linked_artifact_id
             FROM ics_evidence_events ev
             JOIN episodic_events e ON e.id = ev.episodic_event_id
             WHERE ev.belief_id = ?
             ORDER BY e.timestamp DESC, e.rowid DESC
             LIMIT ?"
        )
        .bind(belief_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        use sqlx::Row;
        let mut events = Vec::new();
        for row in rows {
            let payload_raw: String = row.get("payload_json");
            events.push(EpisodicEvent {
                id: row.get("id"),
                event_type: row.get("event_type"),
                payload: serde_json::from_str(&payload_raw).unwrap_or_else(|_| serde_json::json!({})),
                timestamp: row.get("timestamp"),
                run_id: row.try_get("run_id").ok(),
                trace_id: row.try_get("trace_id").ok(),
                conversation_id: row.try_get("conversation_id").ok(),
                scope: row.try_get("scope").ok(),
                source_type: row.get("source_type"),
                source_ref: row.try_get("source_ref").ok(),
                linked_belief_id: row.try_get("linked_belief_id").ok(),
                linked_artifact_id: row.try_get("linked_artifact_id").ok(),
            });
        }
        Ok(events)
    }

    pub async fn list_episodic_events_for_self_belief(
        &self,
        belief_id: i64,
        limit: i64,
    ) -> Result<Vec<EpisodicEvent>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = limit.max(1).min(200);
        let rows = sqlx::query(
            "SELECT e.id, e.event_type, e.payload_json, e.timestamp, e.run_id, e.trace_id, e.conversation_id, e.scope, e.source_type, e.source_ref, e.linked_belief_id, e.linked_artifact_id
             FROM self_evidence_events ev
             JOIN episodic_events e ON e.id = ev.episodic_event_id
             WHERE ev.belief_id = ?
             ORDER BY e.timestamp DESC, e.rowid DESC
             LIMIT ?"
        )
        .bind(belief_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        use sqlx::Row;
        let mut events = Vec::new();
        for row in rows {
            let payload_raw: String = row.get("payload_json");
            events.push(EpisodicEvent {
                id: row.get("id"),
                event_type: row.get("event_type"),
                payload: serde_json::from_str(&payload_raw).unwrap_or_else(|_| serde_json::json!({})),
                timestamp: row.get("timestamp"),
                run_id: row.try_get("run_id").ok(),
                trace_id: row.try_get("trace_id").ok(),
                conversation_id: row.try_get("conversation_id").ok(),
                scope: row.try_get("scope").ok(),
                source_type: row.get("source_type"),
                source_ref: row.try_get("source_ref").ok(),
                linked_belief_id: row.try_get("linked_belief_id").ok(),
                linked_artifact_id: row.try_get("linked_artifact_id").ok(),
            });
        }
        Ok(events)
    }

    pub async fn list_episodic_events_for_entity(
        &self,
        entity_id: i64,
        limit: i64,
    ) -> Result<Vec<EpisodicEvent>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = limit.max(1).min(200);
        let rows = sqlx::query(
            "SELECT e.id, e.event_type, e.payload_json, e.timestamp, e.run_id, e.trace_id, e.conversation_id, e.scope, e.source_type, e.source_ref, e.linked_belief_id, e.linked_artifact_id
             FROM episodic_events e
             WHERE e.id IN (
                 SELECT DISTINCT ev.episodic_event_id
                 FROM ics_evidence_events ev
                 LEFT JOIN ics_fact_beliefs fb ON fb.belief_id = ev.belief_id
                 LEFT JOIN ics_rel_participants rp ON rp.belief_id = ev.belief_id
                 WHERE ev.episodic_event_id IS NOT NULL
                   AND (fb.subject_entity_id = ? OR rp.entity_id = ?)
             )
             ORDER BY e.timestamp DESC, e.rowid DESC
             LIMIT ?"
        )
        .bind(entity_id)
        .bind(entity_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        use sqlx::Row;
        let mut events = Vec::new();
        for row in rows {
            let payload_raw: String = row.get("payload_json");
            events.push(EpisodicEvent {
                id: row.get("id"),
                event_type: row.get("event_type"),
                payload: serde_json::from_str(&payload_raw).unwrap_or_else(|_| serde_json::json!({})),
                timestamp: row.get("timestamp"),
                run_id: row.try_get("run_id").ok(),
                trace_id: row.try_get("trace_id").ok(),
                conversation_id: row.try_get("conversation_id").ok(),
                scope: row.try_get("scope").ok(),
                source_type: row.get("source_type"),
                source_ref: row.try_get("source_ref").ok(),
                linked_belief_id: row.try_get("linked_belief_id").ok(),
                linked_artifact_id: row.try_get("linked_artifact_id").ok(),
            });
        }
        Ok(events)
    }

    // ==================== Cognitive Stack Methods ====================

    /// Get the Semantic Core content
    pub async fn get_semantic_core(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query("SELECT content FROM semantic_core WHERE id = 1")
            .fetch_optional(&self.pool)
            .await?;
        
        use sqlx::Row;
        Ok(row.map(|r| r.get::<String, _>("content")).unwrap_or_default())
    }

    /// Update the Semantic Core content
    pub async fn set_semantic_core(&self, content: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("UPDATE semantic_core SET content = ?, updated_at = CURRENT_TIMESTAMP, version = version + 1 WHERE id = 1")
            .bind(content)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Search episodic memories using FTS5 with temporal decay
    pub async fn search_episodic(&self, query: &str, limit: i32) -> Result<Vec<(String, String, String, bool)>, Box<dyn std::error::Error + Send + Sync>> {
        // Query format: returns (key, value, keywords, is_critical)
        // Uses BM25 weighted by temporal decay
        let rows = sqlx::query(
            r#"SELECT 
                key, 
                value, 
                keywords,
                is_critical,
                bm25(kv_fts) * (1.0 / (1.0 + (julianday('now') - julianday(kv_store.updated_at)) * 0.02)) as score
            FROM kv_fts
            JOIN kv_store ON kv_fts.key = kv_store.key
            WHERE kv_fts MATCH ?
            ORDER BY score DESC
            LIMIT ?"#
        )
        .bind(query)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        
        use sqlx::Row;
        let mut results = Vec::new();
        for row in rows {
            results.push((
                row.get::<String, _>("key"),
                row.get::<Option<String>, _>("value").unwrap_or_default(),
                row.get::<Option<String>, _>("keywords").unwrap_or_default(),
                row.get::<i32, _>("is_critical") != 0,
            ));
        }
        Ok(results)
    }

    /// Search Blackboard (kv_store) using FTS5 with temporal decay.
    pub async fn search_blackboard(&self, query: &str, limit: i32) -> Result<Vec<(String, String)>, Box<dyn std::error::Error + Send + Sync>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let limit = limit.max(1).min(20);
        let rows = sqlx::query(
            r#"SELECT 
                key, 
                value,
                bm25(kv_fts) * (1.0 / (1.0 + (julianday('now') - julianday(kv_store.updated_at)) * 0.02)) as score
            FROM kv_fts
            JOIN kv_store ON kv_fts.key = kv_store.key
            WHERE kv_fts MATCH ?
            ORDER BY score DESC
            LIMIT ?"#
        )
        .bind(query)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        use sqlx::Row;
        let mut results = Vec::new();
        for row in rows {
            let key = row.get::<String, _>("key");
            let value = row.get::<Option<String>, _>("value").unwrap_or_default();
            if !value.trim().is_empty() {
                results.push((key, value));
            }
        }
        Ok(results)
    }

    /// Reinforce a memory (update its timestamp to prevent decay)
    pub async fn reinforce_memory(&self, key: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("UPDATE kv_store SET updated_at = CURRENT_TIMESTAMP WHERE key = ?")
            .bind(key)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Get consolidation state
    pub async fn get_consolidation_state(&self) -> Result<(chrono::DateTime<chrono::Utc>, i32), Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query("SELECT last_run, entries_since FROM consolidation_state WHERE id = 1")
            .fetch_one(&self.pool)
            .await?;
        
        use sqlx::Row;
        Ok((
            row.get("last_run"),
            row.get("entries_since"),
        ))
    }

    /// Increment consolidation entry counter
    pub async fn increment_consolidation_counter(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("UPDATE consolidation_state SET entries_since = entries_since + 1 WHERE id = 1")
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Reset consolidation state after running consolidation
    pub async fn reset_consolidation_state(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query("UPDATE consolidation_state SET last_run = CURRENT_TIMESTAMP, entries_since = 0 WHERE id = 1")
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Get episodes since last consolidation for summarizer input
    pub async fn get_episodes_since_consolidation(&self) -> Result<Vec<(String, String, String)>, Box<dyn std::error::Error + Send + Sync>> {
        let (last_run, _) = self.get_consolidation_state().await?;
        
        let rows = sqlx::query(
            "SELECT key, value, updated_at FROM kv_store WHERE updated_at > ? ORDER BY updated_at ASC"
        )
        .bind(last_run)
        .fetch_all(&self.pool)
        .await?;
        
        use sqlx::Row;
        let mut results = Vec::new();
        for row in rows {
            let updated: chrono::DateTime<chrono::Utc> = row.get("updated_at");
            results.push((
                row.get::<String, _>("key"),
                row.get::<Option<String>, _>("value").unwrap_or_default(),
                updated.format("%Y-%m-%d").to_string(),
            ));
        }
        Ok(results)
    }

    /// Set a key with keywords (new Cognitive Stack format)
    pub async fn set_key_with_keywords(&self, key: &str, value: &str, keywords: &str, is_critical: bool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        sqlx::query(
            "INSERT INTO kv_store (key, value, keywords, is_critical, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) ON CONFLICT(key) DO UPDATE SET value = excluded.value, keywords = excluded.keywords, is_critical = excluded.is_critical, updated_at = CURRENT_TIMESTAMP"
        )
        .bind(key)
        .bind(value)
        .bind(keywords)
        .bind(is_critical)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Check if a key exists (for conflict detection)
    pub async fn key_exists(&self, key: &str) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query("SELECT value FROM kv_store WHERE key = ?")
            .bind(key)
            .fetch_optional(&self.pool)
            .await?;
        
        use sqlx::Row;
        Ok(row.map(|r| r.get("value")))
    }

    pub async fn log_memory_write(
        &self,
        conversation_id: Option<&str>,
        category: &str,
        source: &str,
        reason_code: &str,
        run_id: Option<&str>,
        trace_id: Option<&str>,
        payload_hash: Option<&str>,
        snapshot_hash: Option<&str>,
        gate_decision_id: Option<&str>,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let mut snapshot_hash = snapshot_hash.map(|s| s.to_string());
        if snapshot_hash.is_none() {
            if let Some(cid) = conversation_id {
                snapshot_hash = sqlx::query_scalar(
                    "SELECT snapshot_hash FROM subject_snapshots
                     WHERE conversation_id = ?
                     ORDER BY datetime(timestamp) DESC
                     LIMIT 1",
                )
                .bind(cid)
                .fetch_optional(&self.pool)
                .await?;
            }
        }
        let mut gate_decision_id = gate_decision_id.map(|s| s.to_string());
        if gate_decision_id.is_none() {
            if let Some(hash) = snapshot_hash.as_deref() {
                gate_decision_id = sqlx::query_scalar(
                    "SELECT decision_id FROM gate_decisions
                     WHERE snapshot_hash = ?
                     ORDER BY datetime(created_at) DESC
                     LIMIT 1",
                )
                .bind(hash)
                .fetch_optional(&self.pool)
                .await?;
            }
        }
        let gate_decision: Option<String> = if let Some(decision_id) = gate_decision_id.as_deref() {
            sqlx::query_scalar(
                "SELECT decision FROM gate_decisions
                 WHERE decision_id = ?
                 LIMIT 1",
            )
            .bind(decision_id)
            .fetch_optional(&self.pool)
            .await?
        } else {
            None
        };
        let mut gate_allows = matches!(
            gate_decision.as_deref(),
            Some("ALLOW" | "ALLOW_WITH_NOTICE" | "ALLOW_WITH_AUDIT")
        );
        let summary_override = category == "summary"
            && source == "kernel"
            && reason_code == "user_visible_turn";
        if !gate_allows && summary_override {
            gate_allows = true;
            let _ = system_log::log_event(
                &self.pool,
                None,
                "info",
                "memory",
                run_id,
                trace_id,
                serde_json::json!({
                    "event": "memory_write_override",
                    "reason": "summary_user_turn",
                    "category": category,
                    "source": source,
                    "reason_code": reason_code,
                    "conversation_id": conversation_id,
                    "snapshot_hash": snapshot_hash,
                    "gate_decision_id": gate_decision_id,
                    "gate_decision": gate_decision,
                }),
            )
            .await;
        }
        if !gate_allows {
            let _ = system_log::log_event(
                &self.pool,
                None,
                "warn",
                "memory",
                run_id,
                trace_id,
                serde_json::json!({
                    "event": "memory_write_blocked",
                    "reason": "gate_decision",
                    "category": category,
                    "source": source,
                    "reason_code": reason_code,
                    "conversation_id": conversation_id,
                    "snapshot_hash": snapshot_hash,
                    "gate_decision_id": gate_decision_id,
                    "gate_decision": gate_decision,
                }),
            )
            .await;
            return Ok(false);
        }
        let category_enum = match category {
            "summary" => MemoryWriteCategory::Summary,
            "inner_summary" => MemoryWriteCategory::InnerSummary,
            "episodic" => MemoryWriteCategory::Episodic,
            "semantic" => MemoryWriteCategory::Semantic,
            "semantic_core" => MemoryWriteCategory::SemanticCore,
            "memory_pass" => MemoryWriteCategory::MemoryPass,
            _ => MemoryWriteCategory::Unknown,
        };
        let source_enum = match source {
            "kernel" => MemoryWriteSource::Kernel,
            "scheduler" => MemoryWriteSource::Scheduler,
            "model_client" => MemoryWriteSource::ModelClient,
            "memory_writer" => MemoryWriteSource::MemoryWriter,
            "self_reflection" => MemoryWriteSource::SelfReflection,
            _ => MemoryWriteSource::Unknown,
        };

        let allowed = MemoryPolicy::is_allowed(category_enum, source_enum, reason_code);
        let id = Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT INTO memory_write_ledger (id, conversation_id, category, source, reason_code, run_id, trace_id, payload_hash, snapshot_hash, gate_decision_id, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)"
        )
        .bind(&id)
        .bind(conversation_id)
        .bind(category)
        .bind(source)
        .bind(reason_code)
        .bind(run_id)
        .bind(trace_id)
        .bind(payload_hash)
        .bind(snapshot_hash.as_deref())
        .bind(gate_decision_id.as_deref())
        .execute(&self.pool)
        .await?;

        if !allowed {
            let _ = system_log::log_event(
                &self.pool,
                None,
                "warn",
                "memory_policy",
                run_id,
                trace_id,
                serde_json::json!({
                    "event": "memory_policy_violation",
                    "category": category,
                    "source": source,
                    "reason_code": reason_code,
                    "conversation_id": conversation_id,
                }),
            )
            .await;
        }

        Ok(allowed)
    }

    pub async fn log_memory_write_tx<'c>(
        &self,
        tx: &mut sqlx::Transaction<'c, sqlx::Sqlite>,
        conversation_id: Option<&str>,
        category: &str,
        source: &str,
        reason_code: &str,
        run_id: Option<&str>,
        trace_id: Option<&str>,
        payload_hash: Option<&str>,
        snapshot_hash: Option<&str>,
        gate_decision_id: Option<&str>,
    ) -> Result<bool, String> {
        let mut snapshot_hash = snapshot_hash.map(|s| s.to_string());
        if snapshot_hash.is_none() {
            if let Some(cid) = conversation_id {
                snapshot_hash = sqlx::query_scalar(
                    "SELECT snapshot_hash FROM subject_snapshots
                     WHERE conversation_id = ?
                     ORDER BY datetime(timestamp) DESC
                     LIMIT 1",
                )
                .bind(cid)
                .fetch_optional(&mut **tx)
                .await
                .ok()
                .flatten();
            }
        }
        let mut gate_decision_id = gate_decision_id.map(|s| s.to_string());
        if gate_decision_id.is_none() {
            if let Some(hash) = snapshot_hash.as_deref() {
                gate_decision_id = sqlx::query_scalar(
                    "SELECT decision_id FROM gate_decisions
                     WHERE snapshot_hash = ?
                     ORDER BY datetime(created_at) DESC
                     LIMIT 1",
                )
                .bind(hash)
                .fetch_optional(&mut **tx)
                .await
                .ok()
                .flatten();
            }
        }
        let gate_decision: Option<String> = if let Some(decision_id) = gate_decision_id.as_deref() {
            sqlx::query_scalar(
                "SELECT decision FROM gate_decisions
                 WHERE decision_id = ?
                 LIMIT 1",
            )
            .bind(decision_id)
            .fetch_optional(&mut **tx)
            .await
            .ok()
            .flatten()
        } else {
            None
        };
        let mut gate_allows = matches!(
            gate_decision.as_deref(),
            Some("ALLOW" | "ALLOW_WITH_NOTICE" | "ALLOW_WITH_AUDIT")
        );
        let summary_override = category == "summary"
            && source == "kernel"
            && reason_code == "user_visible_turn";
        if !gate_allows && summary_override {
            gate_allows = true;
            let _ = system_log::log_event(
                &self.pool,
                None,
                "info",
                "memory",
                run_id,
                trace_id,
                serde_json::json!({
                    "event": "memory_write_override",
                    "reason": "summary_user_turn",
                    "category": category,
                    "source": source,
                    "reason_code": reason_code,
                    "conversation_id": conversation_id,
                    "snapshot_hash": snapshot_hash,
                    "gate_decision_id": gate_decision_id,
                    "gate_decision": gate_decision,
                }),
            )
            .await;
        }
        if !gate_allows {
            let _ = system_log::log_event(
                &self.pool,
                None,
                "warn",
                "memory",
                run_id,
                trace_id,
                serde_json::json!({
                    "event": "memory_write_blocked",
                    "reason": "gate_decision",
                    "category": category,
                    "source": source,
                    "reason_code": reason_code,
                    "conversation_id": conversation_id,
                    "snapshot_hash": snapshot_hash,
                    "gate_decision_id": gate_decision_id,
                    "gate_decision": gate_decision,
                }),
            )
            .await;
            return Ok(false);
        }
        let category_enum = match category {
            "summary" => MemoryWriteCategory::Summary,
            "inner_summary" => MemoryWriteCategory::InnerSummary,
            "episodic" => MemoryWriteCategory::Episodic,
            "semantic" => MemoryWriteCategory::Semantic,
            "semantic_core" => MemoryWriteCategory::SemanticCore,
            "memory_pass" => MemoryWriteCategory::MemoryPass,
            _ => MemoryWriteCategory::Unknown,
        };
        let source_enum = match source {
            "kernel" => MemoryWriteSource::Kernel,
            "scheduler" => MemoryWriteSource::Scheduler,
            "model_client" => MemoryWriteSource::ModelClient,
            "memory_writer" => MemoryWriteSource::MemoryWriter,
            "self_reflection" => MemoryWriteSource::SelfReflection,
            _ => MemoryWriteSource::Unknown,
        };
        let allowed = MemoryPolicy::is_allowed(category_enum, source_enum, reason_code);
        let id = Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT INTO memory_write_ledger (id, conversation_id, category, source, reason_code, run_id, trace_id, payload_hash, snapshot_hash, gate_decision_id, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)"
        )
        .bind(&id)
        .bind(conversation_id)
        .bind(category)
        .bind(source)
        .bind(reason_code)
        .bind(run_id)
        .bind(trace_id)
        .bind(payload_hash)
        .bind(snapshot_hash.as_deref())
        .bind(gate_decision_id.as_deref())
        .execute(&mut **tx)
        .await
        .map_err(|e| e.to_string())?;
        Ok(allowed)
    }

    pub async fn insert_event_ledger(
        &self,
        event_id: &str,
        event_type: &str,
        payload: &serde_json::Value,
        tags: Option<&serde_json::Value>,
        run_id: Option<&str>,
        trace_id: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let payload_json = serde_json::to_string(payload).unwrap_or_else(|_| "{}".to_string());
        let tags_json = tags.map(|t| serde_json::to_string(t).unwrap_or_else(|_| "{}".to_string()));
        sqlx::query(
            "INSERT OR REPLACE INTO event_ledger (event_id, timestamp, type, payload, tags, run_id, trace_id)
             VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)"
        )
        .bind(event_id)
        .bind(event_type)
        .bind(&payload_json)
        .bind(tags_json)
        .bind(run_id)
        .bind(trace_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn create_memory_pass_token(
        &self,
        run_id: &str,
        conversation_id: &str,
        ttl_seconds: i64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let ttl = ttl_seconds.max(30);
        sqlx::query(
            "INSERT INTO memory_pass_tokens (run_id, conversation_id, created_at, expires_at)
             VALUES (?, ?, CURRENT_TIMESTAMP, datetime('now', ?))
             ON CONFLICT(run_id) DO UPDATE SET
                conversation_id = excluded.conversation_id,
                created_at = CURRENT_TIMESTAMP,
                expires_at = excluded.expires_at"
        )
        .bind(run_id)
        .bind(conversation_id)
        .bind(format!("+{} seconds", ttl))
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn has_recent_memory_write(
        &self,
        window_seconds: i64,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let hit: Option<i64> = sqlx::query_scalar(
            "SELECT 1 FROM memory_write_ledger WHERE (julianday('now') - julianday(created_at)) * 86400 < ? LIMIT 1",
        )
        .bind(window_seconds)
        .fetch_optional(&self.pool)
        .await?;
        Ok(hit.is_some())
    }

    pub async fn get_controller_state(&self) -> Result<Option<ControllerState>, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query("SELECT state_json FROM self_model_controller_state WHERE id = 1")
            .fetch_optional(&self.pool)
            .await?;
        if let Some(row) = row {
            let raw: String = row.get("state_json");
            let parsed = serde_json::from_str::<ControllerState>(&raw).unwrap_or_default();
            return Ok(Some(parsed));
        }
        Ok(None)
    }

    pub async fn get_recent_gate_feedback(&self, limit: i64) -> Vec<String> {
        let events = [
            "user_attribution_blocked",
            "user_attribution_fallback",
            "user_attribution_validated",
            "tool_failure_blocks_claims",
            "tool_args_invalid",
            "tool_result_attribution_blocked",
            "tool_result_attribution_validated",
            "speculative_workspace_blocked",
            "speculative_workspace_marked",
            "monologue_user_confusion",
            "monologue_style_blocked",
            "monologue_candidate_blocked",
            "workspace_compliance_regen",
            "state_disclosure_blocked",
            "state_disclosure_validated",
            "memory_write_blocked",
        ];
        let placeholders = events.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
        let query = format!(
            "SELECT timestamp, payload FROM system_logs
             WHERE json_extract(payload, '$.event') IN ({})
             ORDER BY datetime(timestamp) DESC
             LIMIT ?",
            placeholders
        );
        let mut builder = sqlx::query(&query);
        for event in events.iter() {
            builder = builder.bind(event);
        }
        builder = builder.bind(limit.max(1));
        let rows = builder.fetch_all(&self.pool).await.unwrap_or_default();
        let mut lines = Vec::new();
        for row in rows {
            let timestamp: String = row.try_get("timestamp").unwrap_or_default();
            let payload_str: String = row.try_get("payload").unwrap_or_default();
            let payload = serde_json::from_str::<serde_json::Value>(&payload_str).unwrap_or_else(|_| serde_json::json!({}));
            let event = payload
                .get("event")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let detail = payload
                .as_object()
                .map(|obj| {
                    obj.iter()
                        .filter(|(k, _)| k.as_str() != "event")
                        .map(|(k, v)| format!("{}={}", k, v))
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            let mut line = format!("{}: {}", event, detail);
            if !timestamp.is_empty() {
                line = format!("[{}] {}", timestamp, line);
            }
            if line.chars().count() > 240 {
                line = line.chars().take(240).collect();
            }
            lines.push(line);
        }
        lines
    }

    pub async fn get_telemetry_snapshot_lines(&self, limit: i64) -> Vec<String> {
        let rows = sqlx::query(
            "SELECT key, value, strftime('%Y-%m-%dT%H:%M:%SZ', updated_at) as updated_at
             FROM kv_store
             WHERE key LIKE 'telemetry.%'
             ORDER BY datetime(updated_at) DESC
             LIMIT ?",
        )
        .bind(limit.max(1))
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();

        let mut lines = Vec::new();
        for row in rows {
            let key: String = row.try_get("key").unwrap_or_default();
            let value: String = row.try_get("value").unwrap_or_default();
            let last_evidence_at: Option<String> = row.try_get("updated_at").ok();
            let line = if let Some(ts) = last_evidence_at {
                format!("I observe {} = {} @ {}", key, value, ts)
            } else {
                format!("I observe {} = {}", key, value)
            };
            lines.push(line);
        }
        lines
    }

    pub async fn enqueue_post_processing_job(
        &self,
        job_type: &str,
        conversation_id: Option<&str>,
        run_id: Option<&str>,
    ) -> Result<String, String> {
        self.enqueue_post_processing_job_with_priority(job_type, conversation_id, run_id, 1)
            .await
    }

    pub async fn enqueue_post_processing_job_with_priority(
        &self,
        job_type: &str,
        conversation_id: Option<&str>,
        run_id: Option<&str>,
        priority: i64,
    ) -> Result<String, String> {
        let job_id = Uuid::new_v4().to_string();
        sqlx::query(
            "INSERT INTO post_processing_jobs
             (job_id, job_type, conversation_id, run_id, priority, status, created_at)
             VALUES (?, ?, ?, ?, ?, 'queued', CURRENT_TIMESTAMP)",
        )
        .bind(&job_id)
        .bind(job_type)
        .bind(conversation_id)
        .bind(run_id)
        .bind(priority)
        .execute(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(job_id)
    }

    pub async fn claim_post_processing_job(&self) -> Option<PostProcessingJob> {
        let mut tx = self.pool.begin().await.ok()?;
        let row = sqlx::query(
            "SELECT job_id, job_type, conversation_id, run_id, priority
             FROM post_processing_jobs
             WHERE status = 'queued'
               AND NOT EXISTS (SELECT 1 FROM post_processing_jobs WHERE status = 'running')
               AND (
                   conversation_id IS NULL OR conversation_id NOT IN (
                       SELECT conversation_id FROM post_processing_jobs WHERE status = 'running' AND conversation_id IS NOT NULL
                   )
               )
             ORDER BY priority DESC, datetime(created_at) ASC
             LIMIT 1",
        )
        .fetch_optional(&mut *tx)
        .await
        .ok()
        .flatten()?;

        let job_id: String = row.try_get("job_id").ok()?;
        let job_type: String = row.try_get("job_type").unwrap_or_default();
        let conversation_id: Option<String> = row.try_get("conversation_id").ok();
        let run_id: Option<String> = row.try_get("run_id").ok();
        let priority: i64 = row.try_get("priority").unwrap_or(1);

        let updated = sqlx::query(
            "UPDATE post_processing_jobs
             SET status = 'running', started_at = CURRENT_TIMESTAMP
             WHERE job_id = ? AND status = 'queued'",
        )
        .bind(&job_id)
        .execute(&mut *tx)
        .await
        .ok()
        .map(|res| res.rows_affected())
        .unwrap_or(0);
        if updated == 0 {
            let _ = tx.rollback().await;
            return None;
        }
        let _ = tx.commit().await;

        Some(PostProcessingJob {
            job_id,
            job_type,
            conversation_id,
            run_id,
            priority,
        })
    }

    pub async fn mark_post_processing_job_completed(&self, job_id: &str) -> Result<(), String> {
        sqlx::query(
            "UPDATE post_processing_jobs
             SET status = 'completed', ended_at = CURRENT_TIMESTAMP, error = NULL
             WHERE job_id = ?",
        )
        .bind(job_id)
        .execute(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub async fn mark_post_processing_job_failed(&self, job_id: &str, error: &str) -> Result<(), String> {
        sqlx::query(
            "UPDATE post_processing_jobs
             SET status = 'failed', ended_at = CURRENT_TIMESTAMP, error = ?
             WHERE job_id = ?",
        )
        .bind(error)
        .bind(job_id)
        .execute(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub async fn purge_self_memory_without_evidence_ids(&self) -> Result<i64, String> {
        let ids: Vec<i64> = sqlx::query_scalar(
            "SELECT DISTINCT belief_id FROM self_evidence_events
             WHERE source_evidence_ids IS NULL OR source_evidence_ids = '' OR source_evidence_ids = '[]'",
        )
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();

        if ids.is_empty() {
            return Ok(0);
        }

        let mut tx = self.pool.begin().await.map_err(|e| e.to_string())?;
        let _ = sqlx::query(
            "DELETE FROM self_rel_participants
             WHERE belief_id IN (
                SELECT belief_id FROM self_evidence_events
                WHERE source_evidence_ids IS NULL OR source_evidence_ids = '' OR source_evidence_ids = '[]'
             )",
        )
        .execute(&mut *tx)
        .await;
        let _ = sqlx::query(
            "DELETE FROM self_rel_beliefs
             WHERE belief_id IN (
                SELECT belief_id FROM self_evidence_events
                WHERE source_evidence_ids IS NULL OR source_evidence_ids = '' OR source_evidence_ids = '[]'
             )",
        )
        .execute(&mut *tx)
        .await;
        let _ = sqlx::query(
            "DELETE FROM self_fact_beliefs
             WHERE belief_id IN (
                SELECT belief_id FROM self_evidence_events
                WHERE source_evidence_ids IS NULL OR source_evidence_ids = '' OR source_evidence_ids = '[]'
             )",
        )
        .execute(&mut *tx)
        .await;
        let _ = sqlx::query(
            "DELETE FROM self_evidence_events
             WHERE source_evidence_ids IS NULL OR source_evidence_ids = '' OR source_evidence_ids = '[]'",
        )
        .execute(&mut *tx)
        .await;
        let _ = sqlx::query(
            "DELETE FROM self_beliefs WHERE id NOT IN (SELECT DISTINCT belief_id FROM self_evidence_events)",
        )
        .execute(&mut *tx)
        .await;
        tx.commit().await.map_err(|e| e.to_string())?;

        Ok(ids.len() as i64)
    }

    pub async fn purge_evidence_events_older_than(&self, days: i64) -> Result<i64, String> {
        if days <= 0 {
            return Ok(0);
        }
        let window = format!("-{} days", days.max(1));
        let mut tx = self.pool.begin().await.map_err(|e| e.to_string())?;

        let count_ics: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM ics_evidence_events WHERE datetime(created_at) < datetime('now', ?)",
        )
        .bind(&window)
        .fetch_optional(&mut *tx)
        .await
        .ok()
        .flatten()
        .unwrap_or(0);
        let count_self: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM self_evidence_events WHERE datetime(created_at) < datetime('now', ?)",
        )
        .bind(&window)
        .fetch_optional(&mut *tx)
        .await
        .ok()
        .flatten()
        .unwrap_or(0);

        let _ = sqlx::query(
            "DELETE FROM evidence_links
             WHERE evidence_id IN (
                 SELECT evidence_id FROM evidence_sources WHERE datetime(created_at) < datetime('now', ?)
             )",
        )
        .bind(&window)
        .execute(&mut *tx)
        .await;
        let _ = sqlx::query(
            "DELETE FROM evidence_sources WHERE datetime(created_at) < datetime('now', ?)",
        )
        .bind(&window)
        .execute(&mut *tx)
        .await;
        let _ = sqlx::query(
            "DELETE FROM ics_evidence_events WHERE datetime(created_at) < datetime('now', ?)",
        )
        .bind(&window)
        .execute(&mut *tx)
        .await;
        let _ = sqlx::query(
            "DELETE FROM self_evidence_events WHERE datetime(created_at) < datetime('now', ?)",
        )
        .bind(&window)
        .execute(&mut *tx)
        .await;

        tx.commit().await.map_err(|e| e.to_string())?;
        Ok(count_ics + count_self)
    }

    pub async fn backfill_state_disclosure_metadata(&self) -> Result<i64, String> {
        let rows = sqlx::query(
            "SELECT run_id, trace_id, payload
             FROM system_logs
             WHERE category = 'kernel'
               AND json_extract(payload, '$.event') = 'state_disclosure_validated'
               AND run_id IS NOT NULL",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| e.to_string())?;

        let mut updated = 0i64;
        for row in rows {
            let run_id: String = row.try_get("run_id").unwrap_or_default();
            if run_id.trim().is_empty() {
                continue;
            }
            let trace_id: Option<String> = row.try_get("trace_id").ok();
            let payload_raw: String = row.try_get("payload").unwrap_or_else(|_| "{}".to_string());
            let payload = serde_json::from_str::<serde_json::Value>(&payload_raw).unwrap_or_else(|_| serde_json::json!({}));
            let evidence_ids: Vec<i64> = payload
                .get("evidence_event_ids")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect())
                .unwrap_or_default();
            if evidence_ids.is_empty() {
                continue;
            }

            let message_row = if let Some(trace_id) = trace_id.as_deref() {
                sqlx::query(
                    "SELECT message_id, metadata FROM messages
                     WHERE run_id = ? AND trace_id = ? AND role = 'assistant'
                     ORDER BY datetime(created_at) DESC
                     LIMIT 1",
                )
                .bind(&run_id)
                .bind(trace_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| e.to_string())?
            } else {
                sqlx::query(
                    "SELECT message_id, metadata FROM messages
                     WHERE run_id = ? AND role = 'assistant'
                     ORDER BY datetime(created_at) DESC
                     LIMIT 1",
                )
                .bind(&run_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| e.to_string())?
            };

            let Some(message_row) = message_row else { continue; };
            let message_id: String = message_row.try_get("message_id").unwrap_or_default();
            if message_id.trim().is_empty() {
                continue;
            }
            let metadata_raw: Option<String> = message_row.try_get("metadata").ok();
            let mut metadata = metadata_raw
                .as_deref()
                .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
                .unwrap_or_else(|| serde_json::json!({}));
            if metadata.get("state_ref_evidence_ids").is_some() {
                continue;
            }
            metadata["state_ref_evidence_ids"] = serde_json::json!(evidence_ids);
            let metadata_json = serde_json::to_string(&metadata).unwrap_or_else(|_| "{}".to_string());
            sqlx::query("UPDATE messages SET metadata = ? WHERE message_id = ?")
                .bind(metadata_json)
                .bind(&message_id)
                .execute(&self.pool)
                .await
                .map_err(|e| e.to_string())?;
            updated += 1;
        }

        Ok(updated)
    }

    pub async fn set_controller_state(&self, state: &ControllerState) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let json_state = serde_json::to_string(state).unwrap_or_else(|_| "{}".to_string());
        sqlx::query(
            "INSERT INTO self_model_controller_state (id, state_json, updated_at)
             VALUES (1, ?, CURRENT_TIMESTAMP)
             ON CONFLICT(id) DO UPDATE SET state_json = excluded.state_json, updated_at = CURRENT_TIMESTAMP"
        )
        .bind(json_state)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn insert_controller_state_snapshot(&self, state: &ControllerState) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let json_state = serde_json::to_string(state).unwrap_or_else(|_| "{}".to_string());
        sqlx::query(
            "INSERT INTO self_model_controller_snapshots (state_json, created_at) VALUES (?, CURRENT_TIMESTAMP)"
        )
        .bind(json_state)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn insert_internal_state_snapshot(
        &self,
        run_id: Option<&str>,
        message_id: Option<&str>,
        conversation_id: Option<&str>,
        confidence: f64,
        uncertainty: f64,
        qualia_tag: Option<&str>,
        qualia_intensity: Option<f64>,
        internal_state_summary: &serde_json::Value,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let summary_json =
            serde_json::to_string(internal_state_summary).unwrap_or_else(|_| "{}".to_string());
        sqlx::query(
            "INSERT INTO internal_state_snapshots
             (run_id, message_id, conversation_id, confidence, uncertainty, qualia_tag, qualia_intensity, internal_state_summary_json, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        )
        .bind(run_id)
        .bind(message_id)
        .bind(conversation_id)
        .bind(confidence)
        .bind(uncertainty)
        .bind(qualia_tag)
        .bind(qualia_intensity)
        .bind(summary_json)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn has_memory_pass_token(&self, run_id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let row: Option<String> = sqlx::query_scalar(
            "SELECT run_id FROM memory_pass_tokens WHERE run_id = ? AND datetime(expires_at) > datetime('now')"
        )
        .bind(run_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.is_some())
    }

    pub async fn consume_memory_pass_token(&self, run_id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let row: Option<String> = sqlx::query_scalar(
            "SELECT run_id FROM memory_pass_tokens WHERE run_id = ? AND datetime(expires_at) > datetime('now')"
        )
        .bind(run_id)
        .fetch_optional(&self.pool)
        .await?;
        if row.is_none() {
            return Ok(false);
        }
        let _ = sqlx::query("DELETE FROM memory_pass_tokens WHERE run_id = ?")
            .bind(run_id)
            .execute(&self.pool)
            .await?;
        Ok(true)
    }

    pub async fn purge_expired_memory_pass_tokens(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let _ = sqlx::query(
            "DELETE FROM memory_pass_tokens WHERE datetime(expires_at) <= datetime('now')"
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_memory_write_ledger(
        &self,
        conversation_id: Option<&str>,
        limit: i64,
    ) -> Result<Vec<MemoryWriteLedgerEntry>, Box<dyn std::error::Error + Send + Sync>> {
        let limit = limit.max(1).min(500);
        let rows = if let Some(cid) = conversation_id {
            sqlx::query(
                "SELECT id, conversation_id, category, source, reason_code, run_id, trace_id, payload_hash, snapshot_hash, gate_decision_id, created_at
                 FROM memory_write_ledger
                 WHERE conversation_id = ?
                 ORDER BY datetime(created_at) DESC
                 LIMIT ?"
            )
            .bind(cid)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT id, conversation_id, category, source, reason_code, run_id, trace_id, payload_hash, snapshot_hash, gate_decision_id, created_at
                 FROM memory_write_ledger
                 ORDER BY datetime(created_at) DESC
                 LIMIT ?"
            )
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
        };

        let mut out = Vec::new();
        for row in rows {
            out.push(MemoryWriteLedgerEntry {
                id: row.try_get("id")?,
                conversation_id: row.try_get("conversation_id").ok(),
                category: row.try_get("category")?,
                source: row.try_get("source")?,
                reason_code: row.try_get("reason_code")?,
                run_id: row.try_get("run_id").ok(),
                trace_id: row.try_get("trace_id").ok(),
                payload_hash: row.try_get("payload_hash").ok(),
                snapshot_hash: row.try_get("snapshot_hash").ok(),
                gate_decision_id: row.try_get("gate_decision_id").ok(),
                created_at: row.try_get("created_at")?,
            });
        }
        Ok(out)
    }

    pub async fn cognitive_readiness_report(
        &self,
        conversation_id: &str,
    ) -> Result<CognitiveReadinessReport, Box<dyn std::error::Error + Send + Sync>> {
        let kernel_state_present: Option<i64> = sqlx::query_scalar(
            "SELECT 1 FROM kernel_states WHERE conversation_id = ?"
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?
        .flatten();

        let last_monologue_at: Option<String> = sqlx::query_scalar(
            "SELECT created_at FROM inner_monologue_entries WHERE conversation_id = ? ORDER BY datetime(created_at) DESC LIMIT 1"
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;

        let last_inner_summary_at: Option<String> = sqlx::query_scalar(
            "SELECT updated_at FROM inner_summaries WHERE conversation_id = ? LIMIT 1"
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;

        let last_conversation_summary_at: Option<String> = sqlx::query_scalar(
            "SELECT updated_at FROM conversation_summaries WHERE conversation_id = ? LIMIT 1"
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;

        let last_live_summary_at: Option<String> = sqlx::query_scalar(
            "SELECT updated_at FROM conversation_live_summaries WHERE conversation_id = ? LIMIT 1"
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;

        let last_semantic_core_at: Option<String> = sqlx::query_scalar(
            "SELECT updated_at FROM semantic_core WHERE id = 1"
        )
        .fetch_optional(&self.pool)
        .await?;

        let last_memory_pass_at: Option<String> = sqlx::query_scalar(
            "SELECT created_at FROM memory_write_ledger WHERE category = 'memory_pass' ORDER BY datetime(created_at) DESC LIMIT 1"
        )
        .fetch_optional(&self.pool)
        .await?;

        let recent_memory_policy_violations: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM system_logs WHERE category = 'memory_policy' AND level IN ('warn','error')"
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);

        let tool_dispatch_successes: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM tool_dispatches WHERE status = 'success'"
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);

        let tool_dispatch_failures: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM tool_dispatches WHERE status = 'failed'"
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);

        let mut notes = Vec::new();
        if kernel_state_present.is_none() {
            notes.push("Kernel state not initialized.".to_string());
        }
        if last_monologue_at.is_none() {
            notes.push("No inner monologue entries recorded.".to_string());
        }
        if last_inner_summary_at.is_none() {
            notes.push("Inner summary has not been updated yet.".to_string());
        }
        if last_conversation_summary_at.is_none() && last_live_summary_at.is_none() {
            notes.push("Conversation rolling summary has not been updated yet.".to_string());
        } else if last_conversation_summary_at.is_none() && last_live_summary_at.is_some() {
            notes.push("Stored rolling summary missing; live summary is present.".to_string());
        }
        if last_memory_pass_at.is_none() {
            notes.push("No memory pass writes recorded.".to_string());
        }
        if tool_dispatch_successes == 0 {
            notes.push("No successful tool dispatches recorded (tool loop unverified).".to_string());
        }
        if let (Some(monologue), Some(inner_summary)) = (&last_monologue_at, &last_inner_summary_at) {
            let mono_ts = chrono::DateTime::parse_from_rfc3339(monologue).ok();
            let inner_ts = chrono::DateTime::parse_from_rfc3339(inner_summary).ok();
            if let (Some(mono_ts), Some(inner_ts)) = (mono_ts, inner_ts) {
                if inner_ts < mono_ts {
                    notes.push("Inner summary has not updated since the last monologue tick.".to_string());
                }
            }
        }
        if recent_memory_policy_violations > 0 {
            notes.push(format!(
                "Memory policy violations recorded: {} (inspect system logs).",
                recent_memory_policy_violations
            ));
        }

        Ok(CognitiveReadinessReport {
            generated_at: chrono::Utc::now().to_rfc3339(),
            kernel_state_present: kernel_state_present.is_some(),
            last_monologue_at,
            last_inner_summary_at,
            last_conversation_summary_at,
            last_semantic_core_at,
            last_memory_pass_at,
            recent_memory_policy_violations,
            tool_dispatch_successes,
            tool_dispatch_failures,
            notes,
        })
    }
}

fn split_schema_statements(schema: &str) -> Vec<String> {
    let mut statements = Vec::new();
    let mut current = String::new();
    let mut in_trigger = false;
    for line in schema.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("--") || trimmed.is_empty() {
            continue;
        }
        let upper = trimmed.to_uppercase();
        if upper.starts_with("CREATE TRIGGER") {
            in_trigger = true;
        }
        current.push_str(line);
        current.push('\n');

        if in_trigger {
            if upper == "END;" || upper == "END" {
                let stmt = current.trim();
                if !stmt.is_empty() {
                    statements.push(stmt.to_string());
                }
                current.clear();
                in_trigger = false;
            }
            continue;
        }

        if trimmed.ends_with(';') {
            let stmt = current.trim();
            if !stmt.is_empty() {
                statements.push(stmt.to_string());
            }
            current.clear();
        }
    }
    let tail = current.trim();
    if !tail.is_empty() {
        statements.push(tail.to_string());
    }
    statements
}

async fn table_exists(pool: &SqlitePool, table: &str) -> bool {
    sqlx::query("SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?")
        .bind(table)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
        .is_some()
}

async fn column_exists(pool: &SqlitePool, table: &str, column: &str) -> bool {
    let table_name = table.replace('\'', "''");
    let rows = match sqlx::query(&format!("PRAGMA table_info('{}')", table_name))
        .fetch_all(pool)
        .await
    {
        Ok(rows) => rows,
        Err(_) => return false,
    };

    for row in rows {
        let name: String = row.get("name");
        if name == column {
            return true;
        }
    }

    false
}

pub async fn get_self_inspection(pool: &SqlitePool) -> Result<SelfInspection, String> {
    let table_names = [
        "messages",
        "runs",
        "artifacts",
        "ics_entities",
        "ics_beliefs",
        "ics_fact_beliefs",
        "ics_rel_beliefs",
        "self_beliefs",
        "self_evidence_events",
        "conversation_summaries",
        "conversation_live_summaries",
    ];

    let mut tables = Vec::new();
    for name in table_names {
        let count: i64 = sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {}", name))
            .fetch_one(pool)
            .await
            .unwrap_or(0);
        tables.push((name.to_string(), count));
    }

    let last_user_memory_write: Option<String> = sqlx::query_scalar(
        "SELECT MAX(last_evidence_at) FROM ics_beliefs"
    )
    .fetch_one(pool)
    .await
    .unwrap_or(None);

    let last_self_memory_write: Option<String> = sqlx::query_scalar(
        "SELECT MAX(last_evidence_at) FROM self_beliefs"
    )
    .fetch_one(pool)
    .await
    .unwrap_or(None);

    let open_conflicts: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ics_conflict_sets WHERE status = 'open'"
    )
    .fetch_one(pool)
    .await
    .unwrap_or(0);

    let error_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM messages WHERE status = 'error'"
    )
    .fetch_one(pool)
    .await
    .unwrap_or(0);

    let last_memory_error_at: Option<String> = sqlx::query_scalar(
        "SELECT value FROM kv_store WHERE key = 'memory_last_error_at'"
    )
    .fetch_optional(pool)
    .await
    .unwrap_or(None);

    Ok(SelfInspection {
        tables,
        last_user_memory_write,
        last_self_memory_write,
        last_memory_error_at,
        open_conflicts,
        error_count,
    })
}

async fn ensure_entities_fts(pool: &SqlitePool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut needs_rebuild = !table_exists(pool, "ics_entities_fts").await;
    if !needs_rebuild && !column_exists(pool, "ics_entities_fts", "entity_id").await {
        needs_rebuild = true;
    }

    if needs_rebuild {
        let _ = sqlx::query("DROP TABLE IF EXISTS ics_entities_fts").execute(pool).await;
        if let Err(e) = sqlx::query(
            "CREATE VIRTUAL TABLE ics_entities_fts USING fts5(label, aliases, entity_id UNINDEXED)"
        )
        .execute(pool)
        .await
        {
            eprintln!("[DB] Failed to create ics_entities_fts: {}", e);
            return Err(Box::new(e));
        }
        let _ = sqlx::query("DROP TRIGGER IF EXISTS ics_ent_ai").execute(pool).await;
        let _ = sqlx::query("DROP TRIGGER IF EXISTS ics_ent_ad").execute(pool).await;
        let _ = sqlx::query("DROP TRIGGER IF EXISTS ics_ent_au").execute(pool).await;
        let _ = sqlx::query(
            "CREATE TRIGGER ics_ent_ai AFTER INSERT ON ics_entities BEGIN 
               INSERT INTO ics_entities_fts(rowid, label, aliases, entity_id) VALUES (new.rowid, new.label, new.aliases, new.id); 
             END"
        )
        .execute(pool)
        .await;
        let _ = sqlx::query(
            "CREATE TRIGGER ics_ent_ad AFTER DELETE ON ics_entities BEGIN 
               DELETE FROM ics_entities_fts WHERE rowid = old.rowid;
             END"
        )
        .execute(pool)
        .await;
        let _ = sqlx::query(
            "CREATE TRIGGER ics_ent_au AFTER UPDATE ON ics_entities BEGIN 
               DELETE FROM ics_entities_fts WHERE rowid = old.rowid;
               INSERT INTO ics_entities_fts(rowid, label, aliases, entity_id) VALUES (new.rowid, new.label, new.aliases, new.id); 
             END"
        )
        .execute(pool)
        .await;
        let _ = sqlx::query(
            "INSERT INTO ics_entities_fts(rowid, label, aliases, entity_id) SELECT rowid, label, aliases, id FROM ics_entities"
        )
        .execute(pool)
        .await;
        return Ok(());
    }

    let _ = sqlx::query(
        "CREATE TRIGGER IF NOT EXISTS ics_ent_ai AFTER INSERT ON ics_entities BEGIN 
           INSERT INTO ics_entities_fts(rowid, label, aliases, entity_id) VALUES (new.rowid, new.label, new.aliases, new.id); 
         END"
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "CREATE TRIGGER IF NOT EXISTS ics_ent_ad AFTER DELETE ON ics_entities BEGIN 
           DELETE FROM ics_entities_fts WHERE rowid = old.rowid;
         END"
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "CREATE TRIGGER IF NOT EXISTS ics_ent_au AFTER UPDATE ON ics_entities BEGIN 
           DELETE FROM ics_entities_fts WHERE rowid = old.rowid;
           INSERT INTO ics_entities_fts(rowid, label, aliases, entity_id) VALUES (new.rowid, new.label, new.aliases, new.id); 
         END"
    )
    .execute(pool)
    .await;
    Ok(())
}

async fn ensure_primary_entities(pool: &SqlitePool, user_name: &str, assistant_name: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    ensure_entities_fts(pool).await?;
    let user_id = upsert_canonical_entity(pool, "sys:user", user_name, Some("person")).await.map_err(|e| {
        eprintln!("[DB] upsert_canonical_entity failed for sys:user: {}", e);
        e
    })?;
    let assistant_id = upsert_canonical_entity(pool, "sys:assistant", assistant_name, Some("system")).await.map_err(|e| {
        eprintln!("[DB] upsert_canonical_entity failed for sys:assistant: {}", e);
        e
    })?;

    for session_id in ["default"] {
        let _ = sqlx::query(
            "INSERT INTO ics_session_bindings (session_id, ref_text, entity_id)
             VALUES (?, ?, ?)
             ON CONFLICT(session_id, ref_text) DO UPDATE SET entity_id = excluded.entity_id"
        )
        .bind(session_id)
        .bind("user")
        .bind(user_id)
        .execute(pool)
        .await;

        let _ = sqlx::query(
            "INSERT INTO ics_session_bindings (session_id, ref_text, entity_id)
             VALUES (?, ?, ?)
             ON CONFLICT(session_id, ref_text) DO UPDATE SET entity_id = excluded.entity_id"
        )
        .bind(session_id)
        .bind("assistant")
        .bind(assistant_id)
        .execute(pool)
        .await;
    }

    Ok(())
}

async fn migrate_default_session_bindings(
    pool: &SqlitePool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let has_legacy: Option<i64> = sqlx::query_scalar(
        "SELECT 1 FROM ics_session_bindings WHERE session_id = 'default_session' LIMIT 1"
    )
    .fetch_optional(pool)
    .await?
    .flatten();

    if has_legacy.is_some() {
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO ics_session_bindings (session_id, ref_text, entity_id, created_at)
             SELECT 'default', ref_text, entity_id, created_at
             FROM ics_session_bindings
             WHERE session_id = 'default_session'"
        )
        .execute(pool)
        .await;
        let _ = sqlx::query("DELETE FROM ics_session_bindings WHERE session_id = 'default_session'")
            .execute(pool)
            .await;
    }

    let _ = sqlx::query(
        "UPDATE ics_pending_clarify
         SET session_id = 'default'
         WHERE session_id = 'default_session'"
    )
    .execute(pool)
    .await;

    Ok(())
}

async fn migrate_episodic_defaults(pool: &SqlitePool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let migration_key = "migration.episodic_defaults_v1";
    let applied: Option<String> = sqlx::query_scalar("SELECT value FROM kv_store WHERE key = ?")
        .bind(migration_key)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();
    if applied.is_some() {
        return Ok(());
    }

    let _ = sqlx::query(
        "UPDATE settings
         SET episodic_enabled = 1,
             episodic_injection_enabled = 1,
             episodic_compaction_enabled = 1
         WHERE COALESCE(episodic_opt_out, 0) = 0"
    )
    .execute(pool)
    .await;

    let _ = sqlx::query(
        "UPDATE settings SET episodic_injection_limit = 5
         WHERE episodic_injection_limit IS NULL OR episodic_injection_limit < 1"
    )
    .execute(pool)
    .await;

    let _ = sqlx::query("INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, 'true', CURRENT_TIMESTAMP)")
        .bind(migration_key)
        .execute(pool)
        .await;

    Ok(())
}

async fn migrate_memory_claims_defaults(pool: &SqlitePool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let migration_key = "migration.memory_claims_default_on_v1";
    let applied: Option<String> = sqlx::query_scalar("SELECT value FROM kv_store WHERE key = ?")
        .bind(migration_key)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();
    if applied.is_some() {
        return Ok(());
    }

    let _ = sqlx::query(
        "UPDATE settings
         SET memory_claims_enabled = 1
         WHERE memory_claims_enabled IS NULL OR memory_claims_enabled = 0"
    )
    .execute(pool)
    .await;

    let _ = sqlx::query("INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, 'true', CURRENT_TIMESTAMP)")
        .bind(migration_key)
        .execute(pool)
        .await;

    Ok(())
}

async fn migrate_signature_hashes(pool: &SqlitePool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let migration_key = "migration.signature_hash_v2";
    let applied: Option<String> = sqlx::query_scalar("SELECT value FROM kv_store WHERE key = ?")
        .bind(migration_key)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();
    if applied.is_some() {
        return Ok(());
    }

    let rows = sqlx::query(
        "SELECT b.id, b.kind, b.scope, b.polarity, b.time_bucket_kind, b.time_bucket_value,
                fb.subject_entity_id, fb.key, fb.value_hash, fb.value_literal,
                rb.rel_type, rb.participants_canonical, rb.anchor_signature, rb.direction
         FROM ics_beliefs b
         LEFT JOIN ics_fact_beliefs fb ON fb.belief_id = b.id
         LEFT JOIN ics_rel_beliefs rb ON rb.belief_id = b.id"
    )
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        let _ = sqlx::query("INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, 'true', CURRENT_TIMESTAMP)")
            .bind(migration_key)
            .execute(pool)
            .await;
        return Ok(());
    }

    let mut anchor_roles_cache: HashMap<String, Vec<String>> = HashMap::new();
    let mut tx = pool.begin().await?;

    for row in rows {
        let id: i64 = row.get("id");
        let kind: String = row.get("kind");
        let scope: String = row.get("scope");
        let polarity: String = row.get("polarity");
        let time_bucket_kind: String = row.try_get("time_bucket_kind").unwrap_or_else(|_| "atemporal".to_string());
        let time_bucket_value: Option<String> = row.try_get("time_bucket_value").ok();
        let time_bucket_value_sig = time_bucket_value.clone().unwrap_or_default();

        if kind == "fact" {
            let subject_id: Option<i64> = row.try_get("subject_entity_id").ok();
            let key: Option<String> = row.try_get("key").ok();
            let Some(subject_id) = subject_id else {
                continue;
            };
            let Some(key) = key else {
                continue;
            };

            let stored_value_hash: Option<String> = row.try_get("value_hash").ok();
            let value_literal: Option<String> = row.try_get("value_literal").ok();
            let value_hash = stored_value_hash
                .as_ref()
                .filter(|v| !v.is_empty())
                .cloned()
                .or_else(|| value_literal.as_ref().map(|v| compute_value_hash(v)))
                .unwrap_or_default();

            if value_hash.is_empty() {
                continue;
            }

            if stored_value_hash.as_deref().unwrap_or("").is_empty() {
                let _ = sqlx::query("UPDATE ics_fact_beliefs SET value_hash = ? WHERE belief_id = ?")
                    .bind(&value_hash)
                    .bind(id)
                    .execute(&mut *tx)
                    .await;
            }

            let topic_key = compute_topic_key_fact(subject_id, &key);
            let sig_inputs = vec![
                ("subject_id".to_string(), subject_id.to_string()),
                ("key".to_string(), key),
                ("value_hash".to_string(), value_hash),
                ("scope".to_string(), scope.clone()),
                ("time_bucket_kind".to_string(), time_bucket_kind.clone()),
                ("time_bucket_value".to_string(), time_bucket_value_sig.clone()),
                ("polarity".to_string(), polarity.clone()),
            ];
            let sig_refs: Vec<(&str, &str)> = sig_inputs
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            let signature_hash = compute_signature_hash(&sig_refs);

            let _ = sqlx::query("UPDATE ics_beliefs SET topic_key = ?, signature_hash = ? WHERE id = ?")
                .bind(&topic_key)
                .bind(&signature_hash)
                .bind(id)
                .execute(&mut *tx)
                .await;
        } else if kind == "rel" {
            let rel_type: Option<String> = row.try_get("rel_type").ok();
            let Some(rel_type) = rel_type else {
                continue;
            };
            let direction: Option<String> = row.try_get("direction").ok();

            let mut participants_canonical = row
                .try_get::<String, _>("participants_canonical")
                .ok()
                .filter(|v| !v.is_empty());
            let mut anchor_signature = row
                .try_get::<String, _>("anchor_signature")
                .ok()
                .filter(|v| !v.is_empty());

            let mut participants: Option<Vec<(String, i64)>> = None;

            if participants_canonical.is_none() || anchor_signature.is_none() {
                let p_rows = sqlx::query("SELECT role, entity_id FROM ics_rel_participants WHERE belief_id = ?")
                    .bind(id)
                    .fetch_all(&mut *tx)
                    .await?;
                let p = p_rows
                    .into_iter()
                    .map(|r| (r.get::<String, _>("role"), r.get::<i64, _>("entity_id")))
                    .collect::<Vec<_>>();
                participants = Some(p);
            }

            if participants_canonical.is_none() {
                let p = participants.as_ref().cloned().unwrap_or_default();
                participants_canonical = Some(canonicalize_participants(&p));
            }

            if anchor_signature.is_none() {
                let p = participants.as_ref().cloned().unwrap_or_default();
                let anchor_roles = if let Some(cached) = anchor_roles_cache.get(&rel_type) {
                    cached.clone()
                } else {
                    let roles_row = sqlx::query("SELECT anchor_roles FROM ics_relation_shapes WHERE rel_type = ?")
                        .bind(&rel_type)
                        .fetch_optional(&mut *tx)
                        .await?;
                    let roles = roles_row
                        .and_then(|r| r.try_get::<String, _>("anchor_roles").ok())
                        .and_then(|raw| serde_json::from_str::<Vec<String>>(&raw).ok())
                        .unwrap_or_default();
                    anchor_roles_cache.insert(rel_type.clone(), roles.clone());
                    roles
                };
                anchor_signature = Some(compute_anchor_signature(&anchor_roles, &p, true));
            }

            let participants_canonical = participants_canonical.unwrap_or_default();
            let anchor_signature = anchor_signature.unwrap_or_default();
            let topic_key = format!("rel:{}:{}", rel_type, anchor_signature);

            let mut sig_inputs = vec![
                ("rel_type".to_string(), rel_type),
                ("participants".to_string(), participants_canonical.clone()),
                ("scope".to_string(), scope.clone()),
                ("time_bucket_kind".to_string(), time_bucket_kind.clone()),
                ("time_bucket_value".to_string(), time_bucket_value_sig.clone()),
                ("polarity".to_string(), polarity.clone()),
            ];
            if let Some(direction) = direction.as_deref().filter(|v| !v.is_empty()) {
                sig_inputs.push(("direction".to_string(), direction.to_string()));
            }
            let sig_refs: Vec<(&str, &str)> = sig_inputs
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            let signature_hash = compute_signature_hash(&sig_refs);

            let _ = sqlx::query("UPDATE ics_beliefs SET topic_key = ?, signature_hash = ? WHERE id = ?")
                .bind(&topic_key)
                .bind(&signature_hash)
                .bind(id)
                .execute(&mut *tx)
                .await;

            let _ = sqlx::query("UPDATE ics_rel_beliefs SET participants_canonical = ?, anchor_signature = ? WHERE belief_id = ?")
                .bind(&participants_canonical)
                .bind(&anchor_signature)
                .bind(id)
                .execute(&mut *tx)
                .await;
        }
    }

    tx.commit().await?;

    let _ = sqlx::query("INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, 'true', CURRENT_TIMESTAMP)")
        .bind(migration_key)
        .execute(pool)
        .await;

    Ok(())
}

async fn migrate_rel_type_catalog(pool: &SqlitePool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let migration_key = "migration.rel_type_catalog_v1";
    let applied: Option<String> = sqlx::query_scalar("SELECT value FROM kv_store WHERE key = ?")
        .bind(migration_key)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();
    if applied.is_some() {
        return Ok(());
    }

    let mut tx = pool.begin().await?;

    // Load existing rel_type rows
    let mut rel_type_map: HashMap<String, String> = HashMap::new();
    let existing = sqlx::query("SELECT rel_type_id, canonical_name FROM rel_type")
        .fetch_all(&mut *tx)
        .await
        .unwrap_or_default();
    for row in existing {
        let rel_type_id: String = row.get("rel_type_id");
        let canonical_name: String = row.get("canonical_name");
        rel_type_map.insert(canonical_name, rel_type_id);
    }

    // Seed from relation shapes
    if table_exists(pool, "ics_relation_shapes").await {
        let shapes = sqlx::query("SELECT rel_type, status FROM ics_relation_shapes")
            .fetch_all(&mut *tx)
            .await
            .unwrap_or_default();
        for row in shapes {
            let rel_type_raw: String = row.get("rel_type");
            let status_raw: Option<String> = row.try_get("status").ok();
            let canonical_name = normalize_rel_type(&rel_type_raw);
            let status = if status_raw.as_deref() == Some("seeded") { "canonical" } else { "provisional" };
            if !rel_type_map.contains_key(&canonical_name) {
                let rel_type_id = Uuid::new_v4().to_string();
                let _ = sqlx::query(
                    "INSERT INTO rel_type (rel_type_id, canonical_name, status, created_at)
                     VALUES (?, ?, ?, CURRENT_TIMESTAMP)"
                )
                .bind(&rel_type_id)
                .bind(&canonical_name)
                .bind(status)
                .execute(&mut *tx)
                .await;
                rel_type_map.insert(canonical_name, rel_type_id);
            }
        }
    }

    // Seed from existing relation beliefs
    let rels = sqlx::query("SELECT DISTINCT rel_type_norm FROM ics_rel_beliefs WHERE rel_type_norm IS NOT NULL AND rel_type_norm != ''")
        .fetch_all(&mut *tx)
        .await
        .unwrap_or_default();
    for row in rels {
        let rel_type_norm: String = row.get("rel_type_norm");
        let canonical_name = normalize_rel_type(&rel_type_norm);
        if !rel_type_map.contains_key(&canonical_name) {
            let rel_type_id = Uuid::new_v4().to_string();
            let _ = sqlx::query(
                "INSERT INTO rel_type (rel_type_id, canonical_name, status, created_at)
                 VALUES (?, ?, 'provisional', CURRENT_TIMESTAMP)"
            )
            .bind(&rel_type_id)
            .bind(&canonical_name)
            .execute(&mut *tx)
            .await;
            rel_type_map.insert(canonical_name, rel_type_id);
        }
    }

    // Ensure self-alias for canonical names
    for (canonical_name, rel_type_id) in rel_type_map.iter() {
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO rel_type_alias (alias, rel_type_id, confidence, status, created_at)
             VALUES (?, ?, 1.0, 'confirmed', CURRENT_TIMESTAMP)"
        )
        .bind(canonical_name)
        .bind(rel_type_id)
        .execute(&mut *tx)
        .await;
    }

    // Migrate Phase-1 rel_type aliases
    if table_exists(pool, "ics_rel_type_aliases").await {
        let aliases = sqlx::query("SELECT alias, rel_type, confidence, status FROM ics_rel_type_aliases")
            .fetch_all(&mut *tx)
            .await
            .unwrap_or_default();
        for row in aliases {
            let alias_raw: String = row.get("alias");
            let rel_type_raw: String = row.get("rel_type");
            let confidence: f32 = row.try_get::<f64, _>("confidence").unwrap_or(1.0) as f32;
            let status: String = row.try_get("status").unwrap_or_else(|_| "confirmed".to_string());

            let alias_norm = normalize_rel_type(&alias_raw);
            let canonical_name = normalize_rel_type(&rel_type_raw);

            let rel_type_id = if let Some(id) = rel_type_map.get(&canonical_name) {
                id.clone()
            } else {
                let new_id = Uuid::new_v4().to_string();
                let _ = sqlx::query(
                    "INSERT INTO rel_type (rel_type_id, canonical_name, status, created_at)
                     VALUES (?, ?, 'provisional', CURRENT_TIMESTAMP)"
                )
                .bind(&new_id)
                .bind(&canonical_name)
                .execute(&mut *tx)
                .await;
                rel_type_map.insert(canonical_name.clone(), new_id.clone());
                new_id
            };

            let _ = sqlx::query(
                "INSERT OR IGNORE INTO rel_type_alias (alias, rel_type_id, confidence, status, created_at)
                 VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)"
            )
            .bind(&alias_norm)
            .bind(&rel_type_id)
            .bind(confidence)
            .bind(status)
            .execute(&mut *tx)
            .await;
        }
    }

    // Migrate relation shapes into rel_shape
    if table_exists(pool, "ics_relation_shapes").await {
        let shapes = sqlx::query(
            "SELECT rel_type, roles, anchor_roles, cardinality_override, commutative, expected_arity, status, created_at
             FROM ics_relation_shapes"
        )
        .fetch_all(&mut *tx)
        .await
        .unwrap_or_default();
        for row in shapes {
            let rel_type_raw: String = row.get("rel_type");
            let canonical_name = normalize_rel_type(&rel_type_raw);
            if let Some(rel_type_id) = rel_type_map.get(&canonical_name) {
                let roles: String = row.try_get("roles").unwrap_or_else(|_| "[]".to_string());
                let anchor_roles: String = row.try_get("anchor_roles").unwrap_or_else(|_| "[]".to_string());
                let cardinality_override: Option<String> = row.try_get("cardinality_override").ok();
                let commutative: i64 = row.try_get("commutative").unwrap_or(0);
                let expected_arity: Option<i64> = row.try_get("expected_arity").ok();
                let status: String = row.try_get("status").unwrap_or_else(|_| "seeded".to_string());
                let created_at: Option<String> = row.try_get("created_at").ok();

                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO rel_shape (rel_type_id, roles, anchor_roles, cardinality_override, commutative, expected_arity, status, created_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))"
                )
                .bind(rel_type_id)
                .bind(&roles)
                .bind(&anchor_roles)
                .bind(cardinality_override)
                .bind(commutative)
                .bind(expected_arity)
                .bind(status)
                .bind(created_at)
                .execute(&mut *tx)
                .await;
            }
        }
    }

    // Backfill rel_type_id in relation beliefs and claims
    let _ = sqlx::query(
        "UPDATE ics_rel_beliefs
         SET rel_type_id = (
             SELECT rel_type_id FROM rel_type WHERE canonical_name = rel_type_norm LIMIT 1
         )
         WHERE rel_type_id IS NULL OR rel_type_id = ''"
    )
    .execute(&mut *tx)
    .await;

    let _ = sqlx::query(
        "UPDATE memory_claims
         SET rel_type_id = (
             SELECT rel_type_id FROM rel_type WHERE canonical_name = rel_type_norm LIMIT 1
         )
         WHERE (rel_type_id IS NULL OR rel_type_id = '') AND rel_type_norm IS NOT NULL"
    )
    .execute(&mut *tx)
    .await;

    tx.commit().await?;

    let _ = sqlx::query("INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, 'true', CURRENT_TIMESTAMP)")
        .bind(migration_key)
        .execute(pool)
        .await;

    Ok(())
}

async fn migrate_rel_signature_ids(pool: &SqlitePool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let migration_key = "migration.rel_signature_ids_v1";
    let applied: Option<String> = sqlx::query_scalar("SELECT value FROM kv_store WHERE key = ?")
        .bind(migration_key)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();
    if applied.is_some() {
        return Ok(());
    }

    let rows = sqlx::query(
        "SELECT b.id, b.scope, b.polarity, b.time_bucket_kind, b.time_bucket_value,
                rb.rel_type_id, rb.participants_canonical, rb.anchor_signature, rb.direction
         FROM ics_beliefs b
         JOIN ics_rel_beliefs rb ON rb.belief_id = b.id
         WHERE b.kind = 'rel'"
    )
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        let _ = sqlx::query("INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, 'true', CURRENT_TIMESTAMP)")
            .bind(migration_key)
            .execute(pool)
            .await;
        return Ok(());
    }

    let mut anchor_roles_cache: HashMap<String, Vec<String>> = HashMap::new();
    let mut tx = pool.begin().await?;

    for row in rows {
        let id: i64 = row.get("id");
        let scope: String = row.get("scope");
        let polarity: String = row.get("polarity");
        let time_bucket_kind: String = row.try_get("time_bucket_kind").unwrap_or_else(|_| "atemporal".to_string());
        let time_bucket_value: Option<String> = row.try_get("time_bucket_value").ok();
        let time_bucket_value_sig = time_bucket_value.clone().unwrap_or_default();
        let rel_type_id: Option<String> = row.try_get("rel_type_id").ok();
        let Some(rel_type_id) = rel_type_id else {
            continue;
        };
        let direction: Option<String> = row.try_get("direction").ok();

        let p_rows = sqlx::query("SELECT role, entity_id FROM ics_rel_participants WHERE belief_id = ?")
            .bind(id)
            .fetch_all(&mut *tx)
            .await?;
        let participants = p_rows
            .into_iter()
            .map(|r| (r.get::<String, _>("role"), r.get::<i64, _>("entity_id")))
            .collect::<Vec<_>>();

        let participants_canonical = canonicalize_participants(&participants);
        let participants_id_sig = serialize_participant_ids(&participants);

        let anchor_roles = if let Some(cached) = anchor_roles_cache.get(&rel_type_id) {
            cached.clone()
        } else {
            let roles_row = sqlx::query("SELECT anchor_roles FROM rel_shape WHERE rel_type_id = ?")
                .bind(&rel_type_id)
                .fetch_optional(&mut *tx)
                .await?;
            let roles = roles_row
                .and_then(|r| r.try_get::<String, _>("anchor_roles").ok())
                .and_then(|raw| serde_json::from_str::<Vec<String>>(&raw).ok())
                .unwrap_or_default();
            anchor_roles_cache.insert(rel_type_id.clone(), roles.clone());
            roles
        };

        let anchor_signature = compute_anchor_signature(&anchor_roles, &participants, false);
        let topic_key = format!("rel:{}:{}", rel_type_id, anchor_signature);

        let mut sig_inputs = vec![
            ("rel_type_id".to_string(), rel_type_id.clone()),
            ("participants".to_string(), participants_id_sig),
            ("arity".to_string(), participants.len().to_string()),
            ("scope".to_string(), scope.clone()),
            ("time_bucket_kind".to_string(), time_bucket_kind.clone()),
            ("time_bucket_value".to_string(), time_bucket_value_sig),
            ("polarity".to_string(), polarity.clone()),
        ];
        if let Some(direction) = direction.as_deref().filter(|v| !v.is_empty()) {
            sig_inputs.push(("direction".to_string(), direction.to_string()));
        }
        let sig_refs: Vec<(&str, &str)> = sig_inputs
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let signature_hash = compute_signature_hash(&sig_refs);

        let _ = sqlx::query("UPDATE ics_beliefs SET topic_key = ?, signature_hash = ? WHERE id = ?")
            .bind(&topic_key)
            .bind(&signature_hash)
            .bind(id)
            .execute(&mut *tx)
            .await;

        let _ = sqlx::query("UPDATE ics_rel_beliefs SET participants_canonical = ?, anchor_signature = ? WHERE belief_id = ?")
            .bind(&participants_canonical)
            .bind(&anchor_signature)
            .bind(id)
            .execute(&mut *tx)
            .await;
    }

    tx.commit().await?;

    let _ = sqlx::query("INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, 'true', CURRENT_TIMESTAMP)")
        .bind(migration_key)
        .execute(pool)
        .await;

    Ok(())
}

async fn migrate_workspace_meta_state(
    pool: &SqlitePool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let migration_key = "migration.workspace_meta_v1";
    let applied: Option<String> = sqlx::query_scalar("SELECT value FROM kv_store WHERE key = ?")
        .bind(migration_key)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();
    if applied.is_some() {
        return Ok(());
    }

    if !column_exists(pool, "workspace_state", "workspace_meta_json").await {
        let _ = sqlx::query("ALTER TABLE workspace_state ADD COLUMN workspace_meta_json TEXT")
            .execute(pool)
            .await;
    }
    let _ = sqlx::query(
        "UPDATE workspace_state SET workspace_meta_json = '{}' WHERE workspace_meta_json IS NULL",
    )
    .execute(pool)
    .await;

    let _ = sqlx::query(
        "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, 'true', CURRENT_TIMESTAMP)",
    )
    .bind(migration_key)
    .execute(pool)
    .await;

    Ok(())
}

async fn migrate_workspace_goal_stack(
    pool: &SqlitePool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let migration_key = "migration.workspace_goal_stack_v1";
    let applied: Option<String> = sqlx::query_scalar("SELECT value FROM kv_store WHERE key = ?")
        .bind(migration_key)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();
    if applied.is_some() {
        return Ok(());
    }

    if !column_exists(pool, "workspace_state", "goal_stack_json").await {
        let _ = sqlx::query("ALTER TABLE workspace_state ADD COLUMN goal_stack_json TEXT")
            .execute(pool)
            .await;
    }
    let _ = sqlx::query(
        "UPDATE workspace_state SET goal_stack_json = '[]' WHERE goal_stack_json IS NULL",
    )
    .execute(pool)
    .await;

    let _ = sqlx::query(
        "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, 'true', CURRENT_TIMESTAMP)",
    )
    .bind(migration_key)
    .execute(pool)
    .await;

    Ok(())
}

async fn migrate_workspace_active_plan(
    pool: &SqlitePool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let migration_key = "migration.workspace_active_plan_v1";
    let applied: Option<String> = sqlx::query_scalar("SELECT value FROM kv_store WHERE key = ?")
        .bind(migration_key)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();
    if applied.is_some() {
        return Ok(());
    }

    if !column_exists(pool, "workspace_state", "active_plan_id").await {
        let _ = sqlx::query("ALTER TABLE workspace_state ADD COLUMN active_plan_id TEXT")
            .execute(pool)
            .await;
    }
    let _ = sqlx::query(
        "UPDATE workspace_state SET active_plan_id = NULL WHERE active_plan_id = ''",
    )
    .execute(pool)
    .await;

    let _ = sqlx::query(
        "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, 'true', CURRENT_TIMESTAMP)",
    )
    .bind(migration_key)
    .execute(pool)
    .await;

    Ok(())
}

async fn migrate_merge_events_schema(
    pool: &SqlitePool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let migration_key = "migration.merge_events_v1";
    let applied: Option<String> = sqlx::query_scalar("SELECT value FROM kv_store WHERE key = ?")
        .bind(migration_key)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten();
    if applied.is_some() {
        return Ok(());
    }

    let has_from_entity = column_exists(pool, "ics_merge_events", "from_entity_id").await;
    let has_to_entity = column_exists(pool, "ics_merge_events", "to_entity_id").await;
    let has_from_id = column_exists(pool, "ics_merge_events", "from_id").await;
    let has_to_id = column_exists(pool, "ics_merge_events", "to_id").await;

    if has_from_entity && !has_from_id {
        let _ = sqlx::query("ALTER TABLE ics_merge_events RENAME COLUMN from_entity_id TO from_id")
            .execute(pool)
            .await;
    }
    if has_to_entity && !has_to_id {
        let _ = sqlx::query("ALTER TABLE ics_merge_events RENAME COLUMN to_entity_id TO to_id")
            .execute(pool)
            .await;
    }

    let _ = sqlx::query(
        "UPDATE ics_entities SET resolution_state = 'do_not_merge' WHERE resolution_state = 'DoNotMerge'",
    )
    .execute(pool)
    .await;

    let _ = sqlx::query(
        "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, 'true', CURRENT_TIMESTAMP)",
    )
    .bind(migration_key)
    .execute(pool)
    .await;

    Ok(())
}


fn parse_json_list(raw: Option<String>) -> Vec<String> {
    raw.and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
        .unwrap_or_default()
}

fn parse_goal_stack_json(
    raw: Option<String>,
) -> Vec<crate::models::GoalStackItem> {
    let Some(raw) = raw else {
        return Vec::new();
    };
    if let Ok(list) = serde_json::from_str::<Vec<crate::models::GoalStackItem>>(&raw) {
        return list
            .into_iter()
            .filter(|item| !item.goal.trim().is_empty())
            .collect();
    }
    if let Ok(list) = serde_json::from_str::<Vec<String>>(&raw) {
        return list
            .into_iter()
            .filter_map(|goal| {
                let trimmed = goal.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(crate::models::GoalStackItem {
                        goal: trimmed,
                        ..Default::default()
                    })
                }
            })
            .collect();
    }
    Vec::new()
}

fn parse_hypotheses_json(
    raw: Option<String>,
) -> Vec<crate::models::WorkspaceHypothesis> {
    let Some(raw) = raw else {
        return Vec::new();
    };
    if let Ok(list) = serde_json::from_str::<Vec<crate::models::WorkspaceHypothesis>>(&raw) {
        return list
            .into_iter()
            .filter(|h| !h.text.trim().is_empty())
            .collect();
    }
    if let Ok(list) = serde_json::from_str::<Vec<String>>(&raw) {
        return list
            .into_iter()
            .filter_map(|text| {
                let trimmed = text.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(crate::models::WorkspaceHypothesis {
                        text: trimmed,
                        confidence: 0.7,
                        speculative: false,
                        evidence_event_ids: Vec::new(),
                        belief_ids: Vec::new(),
                        evidence_quality: None,
                    })
                }
            })
            .collect();
    }
    Vec::new()
}

fn parse_workspace_meta_json(
    raw: Option<String>,
) -> crate::models::WorkspaceMeta {
    raw.and_then(|s| serde_json::from_str::<crate::models::WorkspaceMeta>(&s).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_workspace_meta_json_defaults() {
        let meta = parse_workspace_meta_json(None);
        assert!(meta.current_focus.is_none());
        assert!(meta.open_questions.is_empty());
    }

    #[test]
    fn parse_workspace_meta_json_parses_fields() {
        let raw = r#"{
            "current_focus": {
                "speculative": false,
                "evidence_event_ids": [1],
                "belief_ids": [2]
            },
            "open_questions": [{
                "text": "What is next?",
                "speculative": true,
                "evidence_event_ids": [],
                "belief_ids": []
            }]
        }"#;
        let meta = parse_workspace_meta_json(Some(raw.to_string()));
        let focus = meta.current_focus.expect("current_focus");
        assert!(!focus.speculative);
        assert_eq!(focus.evidence_event_ids, vec![1]);
        assert_eq!(focus.belief_ids, vec![2]);
        assert_eq!(meta.open_questions.len(), 1);
        assert_eq!(meta.open_questions[0].text, "What is next?");
        assert!(meta.open_questions[0].speculative);
    }
}

async fn seed_role_aliases(pool: &SqlitePool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ics_role_aliases")
        .fetch_optional(pool)
        .await?
        .unwrap_or(0);
    if count > 0 {
        return Ok(());
    }

    let aliases = vec![
        ("dad", "father"),
        ("mom", "mother"),
        ("mum", "mother"),
    ];

    for (from_role, to_role) in aliases {
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO ics_role_aliases (from_role, to_role, status, evidence_count, created_at)
             VALUES (?, ?, 'confirmed', 1, CURRENT_TIMESTAMP)"
        )
        .bind(from_role)
        .bind(to_role)
        .execute(pool)
        .await?;
    }

    Ok(())
}

async fn seed_relation_shapes(pool: &SqlitePool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ics_relation_shapes")
        .fetch_optional(pool)
        .await?
        .unwrap_or(0);
    if count > 0 {
        return Ok(());
    }

    struct SeedShape {
        rel_type: &'static str,
        roles: &'static [&'static str],
        commutative: bool,
    }

    let shapes = vec![
        SeedShape { rel_type: "parent_of", roles: &["parent", "child"], commutative: false },
        SeedShape { rel_type: "father_of", roles: &["father", "child"], commutative: false },
        SeedShape { rel_type: "mother_of", roles: &["mother", "child"], commutative: false },
        SeedShape { rel_type: "friends", roles: &["person"], commutative: true },
        SeedShape { rel_type: "works_with", roles: &["person"], commutative: true },
        SeedShape { rel_type: "collaborates_with", roles: &["person"], commutative: true },
        SeedShape { rel_type: "writes", roles: &["subject", "object"], commutative: false },
        SeedShape { rel_type: "owns", roles: &["owner", "object"], commutative: false },
        SeedShape { rel_type: "works_at", roles: &["person", "work"], commutative: false },
        SeedShape { rel_type: "employer_of", roles: &["employer", "employee"], commutative: false },
        SeedShape { rel_type: "created_by", roles: &["creator", "created"], commutative: false },
        SeedShape { rel_type: "prefers", roles: &["person", "object"], commutative: false },
        SeedShape { rel_type: "likes", roles: &["person", "object"], commutative: false },
        SeedShape { rel_type: "dislikes", roles: &["person", "object"], commutative: false },
        SeedShape { rel_type: "project_member_of", roles: &["member", "project"], commutative: false },
        SeedShape { rel_type: "member_of", roles: &["member", "group"], commutative: false },
        SeedShape { rel_type: "lives_in", roles: &["person", "place"], commutative: false },
    ];

    for shape in shapes {
        let roles_json = serde_json::to_string(&shape.roles).unwrap_or_else(|_| "[]".to_string());
        let anchor_roles = "[]";
        let expected_arity: Option<i64> = if shape.commutative {
            None
        } else {
            Some(shape.roles.len() as i64)
        };
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO ics_relation_shapes (rel_type, roles, anchor_roles, commutative, expected_arity, status, created_at)
             VALUES (?, ?, ?, ?, ?, 'seeded', CURRENT_TIMESTAMP)"
        )
        .bind(shape.rel_type)
        .bind(&roles_json)
        .bind(anchor_roles)
        .bind(if shape.commutative { 1 } else { 0 })
        .bind(expected_arity)
        .execute(pool)
        .await?;
    }

    Ok(())
}

async fn seed_rel_type_aliases(pool: &SqlitePool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let aliases = vec![
        ("parent", "parent_of"),
        ("mother", "mother_of"),
        ("father", "father_of"),
        ("mom", "mother_of"),
        ("dad", "father_of"),
        ("has_sibling", "sibling_of"),
        ("sibling", "sibling_of"),
        ("creator_of", "created_by"),
        ("built_by", "created_by"),
        ("made_by", "created_by"),
        ("designed_by", "created_by"),
        ("collaborates", "collaborates_with"),
        ("workswith", "works_with"),
        ("likes", "likes"),
        ("dislikes", "dislikes"),
        ("prefers", "prefers"),
        ("project_member", "project_member_of"),
        ("member", "member_of"),
        ("lives_at", "lives_in"),
        ("located_in", "lives_in"),
    ];

    for (alias_raw, canonical_raw) in aliases {
        let alias = normalize_rel_type(alias_raw);
        let canonical = normalize_rel_type(canonical_raw);

        let existing_id: Option<String> = sqlx::query_scalar(
            "SELECT rel_type_id FROM rel_type WHERE canonical_name = ? LIMIT 1"
        )
        .bind(&canonical)
        .fetch_optional(pool)
        .await?
        .flatten();

        let rel_type_id = if let Some(id) = existing_id {
            id
        } else {
            let rel_type_id = Uuid::new_v4().to_string();
            let _ = sqlx::query(
                "INSERT OR IGNORE INTO rel_type (rel_type_id, canonical_name, status, created_at)
                 VALUES (?, ?, 'canonical', CURRENT_TIMESTAMP)"
            )
            .bind(&rel_type_id)
            .bind(&canonical)
            .execute(pool)
            .await?;
            rel_type_id
        };

        let _ = sqlx::query(
            "INSERT OR IGNORE INTO rel_type_alias (alias, rel_type_id, confidence, status, created_at)
             VALUES (?, ?, 1.0, 'confirmed', CURRENT_TIMESTAMP)"
        )
        .bind(&alias)
        .bind(&rel_type_id)
        .execute(pool)
        .await?;
    }

    Ok(())
}

async fn upsert_canonical_entity(
    pool: &SqlitePool,
    key: &str,
    label: &str,
    entity_type: Option<&str>,
) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
    let key_pattern = format!("%\"{}\"%", key);
    let existing = sqlx::query("SELECT id, label, keys FROM ics_entities WHERE keys LIKE ?")
        .bind(&key_pattern)
        .fetch_optional(pool)
        .await?;

    let canonical_label = canonicalize_label(label);
    let keys_json = serde_json::to_string(&vec![key])?;

    if let Some(row) = existing {
        let id: i64 = row.get("id");
        let current_label: String = row.get("label");
        let current_keys: Option<String> = row.try_get("keys").ok();

        let mut keys_vec: Vec<String> = current_keys
            .as_deref()
            .and_then(|v| serde_json::from_str::<Vec<String>>(v).ok())
            .unwrap_or_default();
        if !keys_vec.iter().any(|k| k == key) {
            keys_vec.push(key.to_string());
        }

        if current_label != label || current_keys.is_none() {
            let updated_keys = serde_json::to_string(&keys_vec)?;
            if let Err(e) = sqlx::query(
                "UPDATE ics_entities SET label = ?, label_canonical = ?, keys = ? WHERE id = ?"
            )
            .bind(label)
            .bind(&canonical_label)
            .bind(updated_keys)
            .bind(id)
            .execute(pool)
            .await
            {
                eprintln!("[DB] upsert_canonical_entity update failed for {}: {}", key, e);
                return Err(Box::new(e));
            }
        }

        return Ok(id);
    }

    let row = sqlx::query(
        "INSERT INTO ics_entities (label, label_canonical, aliases, aliases_canonical, keys, resolution_state, entity_type)
         VALUES (?, ?, '[]', '[]', ?, 'normal', ?) RETURNING id"
    )
    .bind(label)
    .bind(&canonical_label)
    .bind(&keys_json)
    .bind(entity_type)
    .fetch_one(pool)
    .await
    .map_err(|e| {
        eprintln!("[DB] upsert_canonical_entity insert failed for {}: {}", key, e);
        e
    })?;

    Ok(row.get("id"))
}


