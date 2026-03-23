import type { SystemLogPayload } from "./systemLogEvents";

export type View = "chat" | "settings" | "trace";

export interface Message {
  message_id: string;
  run_id?: string | null;
  trace_id?: string | null;
  role: "user" | "assistant" | "system" | "internal";
  content: string;
  status: "complete" | "streaming" | "error" | "cancelled";
  created_at?: string;
  metadata?: any;
}

export interface ModuleStatus {
  stage: string;
  detail?: string | null;
  started_at?: string | null;
  duration_ms?: number | null;
}

export interface SystemLogEntry {
  id: string;
  timestamp: string;
  level: string;
  category: string;
  run_id?: string | null;
  trace_id?: string | null;
  payload: SystemLogPayload;
}

export interface SystemControlEntry {
  control_id: string;
  subsystem_id: string;
  mode: string;
  value_json?: string | null;
  updated_at: string;
  updated_by?: string | null;
  reason?: string | null;
}

export interface SystemControlEvent {
  event_id: string;
  subsystem_id: string;
  previous_mode?: string | null;
  new_mode: string;
  value_json?: string | null;
  actor?: string | null;
  reason?: string | null;
  status?: string | null;
  timestamp: string;
}

export interface SystemHealthSnapshot {
  snapshot_id: string;
  timestamp: string;
  run_id?: string | null;
  trace_id?: string | null;
  metrics: Record<string, any>;
  subsystem_states: any;
}

export interface WaveStatus {
  coherence?: number | null;
  dominance?: number | null;
  turbulence?: number | null;
  drift?: number | null;
  fragmentation?: number | null;
  total_energy?: number | null;
  band_energy?: Record<string, number> | null;
  last_projection_at?: string | null;
  last_contribution_at?: string | null;
  projection_age_seconds?: number | null;
  contribution_age_seconds?: number | null;
}

export interface SubjectSnapshotEntry {
  snapshot_hash: string;
  tick_id: string;
  timestamp: string;
  run_id?: string | null;
  subject_state_json: string;
}

export interface GateDecisionEntry {
  decision_id: string;
  proposal_id: string;
  snapshot_hash: string;
  decision: string;
  evidence_refs_json: string;
  metrics_json: string;
  created_at: string;
}

export interface ContextTagEntry {
  tag: string;
  confidence: number;
  inferred: boolean;
  evidence_event_ids: number[];
  last_seen_at: string;
  source?: string | null;
}

export interface UserIntentSummary {
  summary: string;
  confirmed: boolean;
  evidence_event_ids: number[];
  updated_at: string;
}

export interface EvidenceSource {
  evidence_id: string;
  source_table: string;
  source_id: string;
  source_type: string;
  source_ref?: string | null;
  snippet?: string | null;
  weight?: number | null;
  confidence?: number | null;
  run_id?: string | null;
  trace_id?: string | null;
  conversation_id?: string | null;
  created_at: string;
}

export interface EvidenceLink {
  link_id: string;
  evidence_id: string;
  target_type: string;
  target_id: string;
  relation?: string | null;
  created_at: string;
}

export interface EvidenceLineageEntry {
  source: EvidenceSource;
  links: EvidenceLink[];
}

export interface OutcomeEvent {
  outcome_id: string;
  run_id?: string | null;
  trace_id?: string | null;
  candidate_id?: string | null;
  target_type: string;
  verdict: "confirm" | "disconfirm" | "inconclusive" | string;
  confidence: number;
  source: string;
  note?: string | null;
  evidence_event_ids: number[];
  created_at: string;
}

export interface IntrospectionEntry {
  entry_id: string;
  snapshot_hash: string;
  numeric_payload_json: string;
  narrative: string;
  created_at: string;
}

export interface AuditLogEntry {
  audit_id: string;
  target_id: string;
  snapshot_hash: string;
  discrepancy_score: number;
  recommended_action: string;
  created_at: string;
}

export interface ErrorEventEntry {
  error_event_id: string;
  residual_id: string;
  classification: string;
  status: string;
  created_at: string;
}

export interface QualiaLabelEntry {
  label_id: string;
  event_id: string;
  snapshot_hash: string;
  tag: string;
  intensity: number;
  created_at: string;
}

export interface InnerMonologueCandidate {
  id: string;
  entry_id: string;
  candidate_id?: string | null;
  outcome?: string | null;
  candidate_json: string;
  created_at: string;
}

export interface InnerMonologueEntry {
  id: string;
  conversation_id: string;
  run_id?: string | null;
  dialogue_id?: string | null;
  turn_index?: number | null;
  speaker?: string | null;
  mode: string;
  stream_type?: string | null;
  thought: string;
  descriptors?: string[] | null;
  harvest_type?: string | null;
  harvest_payload?: string | null;
  created_at: string;
  candidates?: InnerMonologueCandidate[] | null;
}

export interface Settings {
  api_base_url: string;
  api_key: string | null;
  streaming_enabled: boolean;
  history_window: number;
  injection_policy: "include" | "exclude";
  active_model_id: string | null;
  system_prompt: string | null;
  voice_name: string | null;
  voice_speed: number | null;
  voice_pitch_semitones: number | null;
  voice_reverb_amount: number | null;
  voice_compression: number | null;
  voice_formant_shift: number | null;
  summarization_api_url: string | null;
  summarization_model: string | null;
  embedding_model: string | null;
  user_display_name: string | null;
  assistant_display_name: string | null;
  onboarding_completed?: boolean | null;
  ui_theme: string | null;
  trace_history_limit: number | null;
  cockpit_write_enabled?: boolean | null;
  episodic_enabled: boolean | null;
  episodic_injection_enabled: boolean | null;
  episodic_compaction_enabled: boolean | null;
  episodic_injection_limit: number | null;
  episodic_opt_out: boolean | null;
  memory_claims_enabled: boolean | null;
  phi_consent?: boolean | null;
  seed_personal_user?: boolean | null;
  lexical_fallback_enabled?: boolean | null;
  research_budget_per_hour?: number | null;
  research_budget_reset_window?: number | null;
  research_cost_per_call?: number | null;
  monologue_interval_seconds?: number | null;
  monologue_timeout_secs?: number | null;
  monologue_retry_timeout_secs?: number | null;
  empty_response_retry_max?: number | null;
  empty_response_retry_timeout_ms?: number | null;
  monologue_max_per_hour?: number | null;
  thread_max_depth?: number | null;
  allow_shell_tool?: boolean | null;
  shell_command_allowlist?: string | null;
  ask_budget_max?: number | null;
  calculator_followups_max?: number | null;
  loop_similarity_threshold?: number | null;
  loop_recent_k?: number | null;
  meta_cog_outcome_turns?: number | null;
  meta_cog_cycle_window_turns?: number | null;
  meta_cog_outcome_timeout_s?: number | null;
  meta_cog_cooldown_s?: number | null;
  meta_cog_streak_limit?: number | null;
  registry_profile_name?: string | null;
  controller_enabled?: boolean | null;
  monologue_stabilization_enabled?: boolean | null;
  monologue_surface_enabled?: boolean | null;
  show_monologue_in_chat?: boolean | null;
  enable_introspection?: boolean | null;
  heartbeat_enabled?: boolean | null;
  dream_enabled?: boolean | null;
  binding_enforcement_enabled?: boolean | null;
  pending_prompt_alignment_enabled?: boolean | null;
  auto_memory_pass_enabled?: boolean | null;
  summary_cohesion_enabled?: boolean | null;
  compact_prompt_enabled?: boolean | null;
  gate_default_soft?: boolean | null;
  gate_shadow_mode?: boolean | null;
  gate_rollout_percent?: number | null;
  self_report_channel?: boolean | null;
  explicit_feedback_only?: boolean | null;
  weight_user_satisfaction?: number | null;
  weight_policy_rigor?: number | null;
  weight_latency?: number | null;
  weight_evidence_strictness?: number | null;
  weight_exploration?: number | null;
  monologue_provenance_guard?: boolean | null;
  organism_decay?: boolean | null;
  model_context_limit?: number | null;
  introspection_confidence_threshold?: number | null;
  introspection_drift_threshold?: number | null;
  introspection_ambiguity_threshold?: number | null;
  enable_attribution_gate?: boolean | null;
  enable_user_utterance_evidence?: boolean | null;
  enable_attribution_metadata?: boolean | null;
  enable_tool_schema_validation?: boolean | null;
  enable_context_evidence?: boolean | null;
  enable_monologue_validator?: boolean | null;
  enable_memory_evidence_gating?: boolean | null;
  enable_speculative_workspace_containment?: boolean | null;
  stability_prompt_override_guard?: boolean | null;
  stability_monologue_tagged?: boolean | null;
  stability_introspection_structured?: boolean | null;
  stability_disable_working_hypothesis?: boolean | null;
  stability_state_disclosure_expanded?: boolean | null;
  stability_transcript_normalization?: boolean | null;
  stability_memory_hygiene?: boolean | null;
  stability_non_stream_sanitization?: boolean | null;
}

export interface PromptStatus {
  prompt_source: string;
  primary_prompt_hash: string;
  memory_prompt_hash: string;
  canonical_primary_hash: string;
  override_hash?: string | null;
  override_active: boolean;
  override_mismatch: boolean;
}


export interface TestResult {
  success: boolean;
  message: string;
}

export interface PendingClarification {
  question: string;
  runId: string;
  originalInput: string;
}

export interface PendingPrompt {
  id: string;
  prompt: string;
  source: string;
  created_at: string;
  skip_count: number;
  auto_surface: boolean;
  intent_kind?: string | null;
  bridge_id?: string | null;
}

export interface MemoryClarificationCandidate {
  entity_id: number;
  label: string;
  context: string;
}

export interface MemoryClarificationRequest {
  pending_id: number;
  ref_text: string;
  question: string;
  candidates: MemoryClarificationCandidate[];
}

export interface MemoryErrorPayload {
  errors: string[];
  timestamp?: string;
}

export interface MemoryClarifyResult {
  success: boolean;
  selected_entity_id: number | null;
  selected_label: string | null;
  error: string | null;
}

export interface AttentionSourceAttribution {
  source: string;
  weight: number;
  share: number;
}

export interface AttentionSchemaState {
  capacity_usage: number;
  selection_policy: string;
  suppressed_items: string[];
  stability: number;
  source_attribution: AttentionSourceAttribution[];
  last_updated_at: string;
}

export interface WorkspaceKernelContributor {
  cycle_id: string;
  broadcast_refs: string[];
  ignition_active: boolean;
  timestamp: string;
}

export interface WorkspaceMemoryContributor {
  recent_writes: number;
  top_topics: string[];
  last_write_at?: string | null;
}

export interface WorkspacePredictionContributor {
  last_prediction_at?: string | null;
  divergence_count: number;
  residual_count: number;
}

export interface WorkspaceAttentionContributor {
  focus_refs: string[];
  meta_confidence: number;
}

export interface WorkspaceSelfModelContributor {
  identity_confidence: number;
  last_reflection_at?: string | null;
}

export interface WorkspaceQualiaContributor {
  dominant_tag?: string | null;
  intensity: number;
}

export interface WorkspaceToolsContributor {
  success_rate: number;
  failure_rate: number;
  last_failure_kind?: string | null;
}

export interface WorkspaceOrganismContributor {
  integrity_risk: number;
  drift_score: number;
}

export interface WorkspaceErrorContributor {
  open_error_count: number;
  pattern_flags: string[];
}

export interface WorkspaceContributors {
  kernel?: WorkspaceKernelContributor | null;
  memory?: WorkspaceMemoryContributor | null;
  prediction?: WorkspacePredictionContributor | null;
  attention?: WorkspaceAttentionContributor | null;
  self_model?: WorkspaceSelfModelContributor | null;
  qualia?: WorkspaceQualiaContributor | null;
  tools?: WorkspaceToolsContributor | null;
  organism?: WorkspaceOrganismContributor | null;
  error_state?: WorkspaceErrorContributor | null;
  missing: string[];
  updated_at?: string | null;
}

export interface SelfModel {
  capabilities: any;
  limitations: any;
  active_tools: any;
  memory_health: any;
  persona: any;
  persona_daily_delta: any;
  persona_last_delta_date: string | null;
  goals: any;
  reflection_status: any;
  reflection_frozen: boolean;
  last_reflection_at: string | null;
  internal_state_summary?: any;
  internal_state_map_version?: number | null;
  unified_state?: any;
  unified_state_evidence?: any;
  unified_state_updated_at?: string | null;
  updated_at: string;
}

export interface ParameterRegistry {
  profile_name: string;
  profile_version: number;
  payload_json: string;
  updated_at: string;
}

export interface SelfInspection {
  tables: [string, number][];
  last_user_memory_write: string | null;
  last_self_memory_write: string | null;
  last_memory_error_at: string | null;
  open_conflicts: number;
  error_count: number;
}

export interface RollingSummaryStatus {
  summary: string | null;
  last_error: string | null;
  last_error_at: string | null;
  pending?: boolean;
}
