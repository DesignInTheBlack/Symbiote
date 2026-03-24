
import { useEffect, useMemo, useState } from "react";
import type {
  InnerMonologueEntry,
  Message,
  SystemLogEntry,
  SubjectSnapshotEntry,
  GateDecisionEntry,
  ContextTagEntry,
  IntrospectionEntry,
  AuditLogEntry,
  ErrorEventEntry,
  QualiaLabelEntry,
  EvidenceLineageEntry,
  SystemControlEntry,
  SystemControlEvent,
  SystemHealthSnapshot,
  ModuleStatus,
  UserIntentSummary,
  Settings,
} from "../types/app";
import { invokeWithTimeout } from "../utils/tauri";
import { SystemControlPanel, SubsystemState } from "../components/SystemControlPanel";
import { SystemHealthPanel } from "../components/SystemHealthPanel";
import { SystemControlHistory } from "../components/SystemControlHistory";
import { SystemHealthTimeline } from "../components/SystemHealthTimeline";
import { OutcomePanel } from "../components/OutcomePanel";
import { formatStatusDetail, formatStatusLabel } from "../utils/statusStages";

interface TraceViewProps {
  systemLogs: SystemLogEntry[];
  monologueEntries?: InnerMonologueEntry[];
  messages?: Message[];
  subjectSnapshots?: SubjectSnapshotEntry[];
  gateDecisions?: GateDecisionEntry[];
  contextTags?: ContextTagEntry[];
  intentSummary?: UserIntentSummary | null;
  introspectionEntries?: IntrospectionEntry[];
  auditLog?: AuditLogEntry[];
  errorEvents?: ErrorEventEntry[];
  qualiaLabels?: QualiaLabelEntry[];
  evidenceLineage?: EvidenceLineageEntry[];
  evidenceLineageError?: string | null;
  systemControls: SystemControlEntry[];
  systemControlEvents: SystemControlEvent[];
  systemHealthSnapshot: SystemHealthSnapshot | null;
  systemHealthHistory: SystemHealthSnapshot[];
  gateInputEvents: SystemLogEntry[];
  cockpitLastUpdated?: string | null;
  controlError?: string | null;
  healthError?: string | null;
  error?: string | null;
  allowControlWrites: boolean;
  cockpitWriteEnabled: boolean;
  settings?: Settings | null;
  onUpdateSettings?: (settings: Settings) => void;
  onToggleCockpitWrite: (enabled: boolean) => void;
  onRefresh: () => void;
  onClear: () => void;
}

type TraceEntry = {
  id: string;
  timestamp: string;
  level: string;
  category: string;
  event: string;
  run_id?: string | null;
  trace_id?: string | null;
  source: "system" | "monologue" | "message";
  payload: unknown;
  alert_ids?: string[];
  severity?: "info" | "warn" | "error";
  evidence_ids?: number[];
};

type CockpitTabId = "overview" | "controls" | "signals" | "diagnostics" | "logs";


type AlertItem = {
  id: string;
  label: string;
  value: string;
  tone: "warn" | "alert";
  reason: string;
  relatedEvents: string[];
};

type LensPreset = {
  id: string;
  label: string;
  runId: string;
  category: string;
  level: string;
  search: string;
  alertId?: string | null;
};

type RunStoryEvent = {
  id: string;
  label: string;
  timestamp: string;
  event: string;
  detail?: string;
  run_id?: string | null;
  source: "system" | "monologue" | "message";
};

const EVENT_LABELS: Record<string, string> = {
  decision_report: "Decision report",
  gate_decision: "Gate decision",
  gate_decision_inputs: "Gate inputs",
  memory_pass_start: "Memory pass started",
  memory_pass_result: "Memory pass result",
  memory_pass_invalid_output: "Memory pass invalid output",
  memory_pass_repair: "Memory pass repaired",
  memory_write_blocked: "Memory write blocked",
  contract_violation: "Contract violation",
  primary_response_json_unwrapped: "Primary response unwrapped",
  response_reasoning_json_invalid: "JSON invalid",
  response_reasoning_json_retry: "JSON retry",
  pending_prompt_sanitized: "Pending prompt sanitized",
  pending_prompt_surfaced: "Pending prompt surfaced",
  monologue_parse_failed: "Monologue parse failed",
  monologue_json_disabled: "Monologue JSON disabled",
  tool_dispatch: "Tool dispatch",
  tool_dispatch_failed: "Tool dispatch failed",
  tool_output_evidence_recorded: "Tool output recorded",
  outcome_summary_evidence: "Outcome evidence",
  monologue_tick: "Monologue tick",
  monologue_tick_result: "Monologue tick result",
  summary_archive_updated: "Summary archive updated",
  workspace_snapshot: "Workspace snapshot",
  run_stage_update: "Run stage update",
};

const EVENT_SEVERITY: Record<string, "info" | "warn" | "error"> = {
  memory_pass_invalid_output: "error",
  memory_pass_error: "error",
  memory_pass_repair: "warn",
  memory_write_blocked: "warn",
  contract_violation: "warn",
  response_reasoning_json_invalid: "error",
  response_reasoning_json_retry: "warn",
  pending_prompt_sanitized: "warn",
  monologue_parse_failed: "warn",
  monologue_json_disabled: "warn",
  tool_dispatch_failed: "error",
};

const EVENT_DETAIL_KEYS: Record<string, string[]> = {
  decision_report: ["decision", "selected_action", "selected_kind", "rationale", "anchor_hits"],
  gate_decision: ["decision", "enforced_decision", "soft_decision", "verify_rate"],
  gate_decision_inputs: ["enforced_decision", "soft_decision", "legacy_decision", "gate_reasons", "signals"],
  memory_pass_result: ["write_count", "facts", "relations", "scope"],
  memory_pass_invalid_output: ["reason", "model", "raw_snippet", "request_label"],
  memory_write_blocked: ["reason", "candidate_kind", "candidate_id"],
  contract_violation: ["policy", "reason", "candidate_kind", "snippet"],
  tool_dispatch: ["tool_name", "action_id", "status"],
  tool_dispatch_failed: ["tool_name", "action_id", "error"],
  pending_prompt_sanitized: ["reason", "candidate_kind", "source"],
  monologue_parse_failed: ["reason", "cooldown_until"],
  response_reasoning_json_invalid: ["request_label", "reason"],
};

const DEFAULT_DETAIL_KEYS = [
  "decision",
  "reason",
  "status",
  "candidate_kind",
  "candidate_id",
  "tool_name",
  "error",
  "duration_ms",
  "evidence_event_ids",
  "run_id",
  "trace_id",
];

const RUN_STORY_EVENTS = new Set([
  "decision_report",
  "gate_decision",
  "gate_decision_inputs",
  "memory_pass_start",
  "memory_pass_result",
  "memory_pass_invalid_output",
  "memory_write_blocked",
  "tool_dispatch",
  "tool_dispatch_failed",
  "pending_prompt_sanitized",
  "pending_prompt_surfaced",
  "monologue_tick",
  "monologue_tick_result",
  "summary_archive_updated",
  "workspace_snapshot",
  "outcome_summary_evidence",
  "run_stage_update",
]);

const LENS_STORAGE_KEY = "symbiote_cockpit_lenses";

const loadLenses = (): LensPreset[] => {
  try {
    const raw = localStorage.getItem(LENS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item) => item && typeof item.label === "string");
  } catch {
    return [];
  }
};

const persistLenses = (lenses: LensPreset[]) => {
  try {
    localStorage.setItem(LENS_STORAGE_KEY, JSON.stringify(lenses));
  } catch {
    // ignore storage errors
  }
};

const formatEventLabel = (value: string) => {
  const labeled = EVENT_LABELS[value];
  if (labeled) return labeled;
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (m) => m.toUpperCase());
};

const isDecisionTarget = (targetType: string) =>
  targetType.toLowerCase().includes("decision");

const truncate = (text: string, max = 120) => (
  text.length > max ? `${text.slice(0, max)}...` : text
);

const shortHash = (value: string) => (value.length > 10 ? `${value.slice(0, 6)}...${value.slice(-4)}` : value);

const buildSystemEntries = (logs: SystemLogEntry[]): TraceEntry[] =>
  logs.map((entry) => ({
    id: entry.id,
    timestamp: entry.timestamp,
    level: entry.level,
    category: entry.category,
    event: typeof entry.payload === "object" && entry.payload !== null && "event" in entry.payload
      ? String((entry.payload as any).event)
      : entry.category,
    run_id: entry.run_id,
    trace_id: entry.trace_id,
    source: "system",
    payload: entry.payload,
  }));

const buildMonologueEntries = (entries: InnerMonologueEntry[]): TraceEntry[] =>
  entries.map((entry) => ({
    id: `monologue:${entry.id}`,
    timestamp: entry.created_at,
    level: "info",
    category: "monologue",
    event: "monologue_turn",
    run_id: entry.run_id,
    trace_id: null,
    source: "monologue",
    payload: {
      mode: entry.mode,
      stream_type: entry.stream_type,
      speaker: entry.speaker,
      turn_index: entry.turn_index,
      thought: entry.thought,
      descriptors: entry.descriptors,
      harvest_type: entry.harvest_type,
      harvest_payload: entry.harvest_payload,
      candidates: entry.candidates,
    },
  }));

  const buildMessageEntries = (messages: Message[]): TraceEntry[] =>
  messages
    .filter((msg) => msg.role !== "internal")
    .map((msg) => ({
      id: `message:${msg.message_id}`,
      timestamp: msg.created_at || new Date().toISOString(),
      level: msg.status === "error" ? "error" : "info",
      category: "message",
      event: `message_${msg.role}`,
      run_id: msg.run_id ?? null,
      trace_id: msg.trace_id ?? null,
      source: "message",
      payload: {
        role: msg.role,
        status: msg.status,
        content: msg.content,
        metadata: msg.metadata,
      },
    }));

  const runIdKey = (value?: string | null) => (value && value.trim().length > 0 ? value : "no-run");
  const formatBadge = (value: number) => (value > 99 ? "99+" : `${value}`);

const INTENT_SAVE_TIMEOUT_MS = 6000;
const WEIGHT_DEFS = [
  { key: "weight_user_satisfaction", label: "User Satisfaction" },
  { key: "weight_policy_rigor", label: "Policy Rigor" },
  { key: "weight_latency", label: "Latency" },
  { key: "weight_evidence_strictness", label: "Evidence Strictness" },
  { key: "weight_exploration", label: "Exploration" },
] as const;

export const TraceView = ({
  systemLogs,
  monologueEntries = [],
  messages = [],
  subjectSnapshots = [],
  gateDecisions = [],
  contextTags = [],
  intentSummary = null,
  introspectionEntries = [],
  auditLog = [],
  errorEvents = [],
  qualiaLabels = [],
  evidenceLineage = [],
  evidenceLineageError,
  systemControlEvents,
  systemHealthSnapshot,
  systemHealthHistory,
  gateInputEvents,
  cockpitLastUpdated,
  controlError,
  healthError,
  error,
  allowControlWrites,
  cockpitWriteEnabled,
  settings,
  onUpdateSettings,
  onToggleCockpitWrite,
  onRefresh,
  onClear,
}: TraceViewProps) => {
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [activeLevel, setActiveLevel] = useState<string>("all");
  const [search, setSearch] = useState<string>("");
  const [activeRunId, setActiveRunId] = useState<string>("all");
  const [runSelectionLocked, setRunSelectionLocked] = useState<boolean>(false);
  const [activeAlertId, setActiveAlertId] = useState<string | null>(null);
  const [lensDraft, setLensDraft] = useState<string>("");
  const [savedLenses, setSavedLenses] = useState<LensPreset[]>(() => loadLenses());
  const [playbackIndex, setPlaybackIndex] = useState<number>(0);
  const [decisionFilter, setDecisionFilter] = useState<string>("all");
  const [collapsedRuns, setCollapsedRuns] = useState<Set<string>>(new Set());
  const [intentDraft, setIntentDraft] = useState<string>(intentSummary?.summary ?? "");
  const [intentConfirmed, setIntentConfirmed] = useState<boolean>(intentSummary?.confirmed ?? false);
  const [intentSaving, setIntentSaving] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<CockpitTabId>("overview");
  const [overviewExpanded, setOverviewExpanded] = useState<boolean>(false);
  const [diagnosticsExpanded, setDiagnosticsExpanded] = useState<boolean>(true);
  const [healthBannerCollapsed, setHealthBannerCollapsed] = useState(false);
  const [healthBannerDismissed, setHealthBannerDismissed] = useState(false);
  const allowEdits = cockpitWriteEnabled && Boolean(onUpdateSettings);

  useEffect(() => {
    setIntentDraft(intentSummary?.summary ?? "");
    setIntentConfirmed(intentSummary?.confirmed ?? false);
  }, [intentSummary]);

  useEffect(() => {
    persistLenses(savedLenses);
  }, [savedLenses]);


  const healthBanner = useMemo(() => {
    const snapshots: SystemHealthSnapshot[] = [];
    if (systemHealthSnapshot) {
      snapshots.push(systemHealthSnapshot);
    }
    if (systemHealthHistory.length > 0) {
      snapshots.push(...systemHealthHistory);
    }
    if (snapshots.length === 0) {
      return null;
    }

    const dayMap = new Map<string, { failed: boolean; reasons: Set<string> }>();

    const evaluateSnapshot = (snapshot: SystemHealthSnapshot) => {
      const metrics = snapshot.metrics || {};
      const failures: string[] = [];
      const monologueRate = metrics?.monologue?.success_rate;
      if (typeof monologueRate === "number" && monologueRate < 0.6) {
        failures.push("Monologue success rate < 0.6");
      }
      const promptTrim = metrics?.prompt_trim || {};
      const trimCount =
        (promptTrim["Rolling Summary"] ?? 0) +
        (promptTrim["Inner Summary"] ?? 0) +
        (promptTrim["Memory Context"] ?? 0);
      if (trimCount > 1) {
        failures.push("Prompt trims on anchors > 1");
      }
      const summaryChunks = metrics?.summaries?.summary_chunk_count ?? 0;
      if (summaryChunks < 1) {
        failures.push("Summary chunk count < 1");
      }
      const reflectionFrozen = metrics?.self_reflection?.reflection_frozen ?? false;
      const lastReflectionAt = metrics?.self_reflection?.last_reflection_at;
      if (!reflectionFrozen) {
        if (!lastReflectionAt) {
          failures.push("Self-reflection missing");
        } else {
          const last = new Date(lastReflectionAt);
          const snapTime = new Date(snapshot.timestamp);
          const diffHours = (snapTime.getTime() - last.getTime()) / (1000 * 60 * 60);
          if (!Number.isNaN(diffHours) && diffHours > 24) {
            failures.push("Self-reflection stale");
          }
        }
      }
      return failures;
    };

    for (const snapshot of snapshots) {
      const dayKey = snapshot.timestamp?.slice(0, 10) ?? "unknown";
      if (!dayMap.has(dayKey)) {
        dayMap.set(dayKey, { failed: false, reasons: new Set<string>() });
      }
      const failures = evaluateSnapshot(snapshot);
      if (failures.length > 0) {
        const entry = dayMap.get(dayKey)!;
        entry.failed = true;
        failures.forEach((reason) => entry.reasons.add(reason));
      }
    }

    const days = Array.from(dayMap.keys()).sort().reverse();
    if (days.length < 2) {
      return null;
    }
    const latestDay = days[0];
    const previousDay = days[1];
    const latest = dayMap.get(latestDay);
    const previous = dayMap.get(previousDay);
    if (!latest || !previous) {
      return null;
    }

    const latestDate = new Date(latestDay);
    const previousDate = new Date(previousDay);
    const dayDiff = Math.round((latestDate.getTime() - previousDate.getTime()) / (1000 * 60 * 60 * 24));
    if (dayDiff > 1) {
      return null;
    }
    if (!latest.failed || !previous.failed) {
      return null;
    }

    return {
      latestDay,
      previousDay,
      reasons: Array.from(latest.reasons.values()),
    };
  }, [systemHealthSnapshot, systemHealthHistory]);

  useEffect(() => {
    if (!healthBanner) {
      setHealthBannerDismissed(false);
      setHealthBannerCollapsed(false);
      return;
    }
    setHealthBannerDismissed(false);
  }, [healthBanner]);

  const allEntries = useMemo(() => {
    const entries = [
      ...buildSystemEntries(systemLogs),
      ...buildMonologueEntries(monologueEntries),
      ...buildMessageEntries(messages),
    ];
    return entries.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }, [systemLogs, monologueEntries, messages]);

  const runOptions = useMemo(() => {
    const counts = new Map<string, number>();
    for (const entry of allEntries) {
      if (!entry.run_id) continue;
      counts.set(entry.run_id, (counts.get(entry.run_id) ?? 0) + 1);
    }
    return Array.from(counts.entries())
      .map(([id, count]) => ({
        id,
        label: `${id.slice(0, 8)}...`,
        count,
      }))
      .sort((a, b) => b.count - a.count);
  }, [allEntries]);

  useEffect(() => {
    if (runSelectionLocked) return;
    const activeRun = (systemHealthSnapshot?.metrics as any)?.run?.active_run_id;
    if (activeRun && activeRun !== activeRunId) {
      setActiveRunId(activeRun);
      return;
    }
    if (activeRunId === "all" && runOptions.length > 0) {
      setActiveRunId(runOptions[0].id);
    }
  }, [systemHealthSnapshot, runOptions, runSelectionLocked, activeRunId]);

  const runFilter = activeRunId !== "all" ? activeRunId : null;

  const filteredSystemLogs = useMemo(() => {
    if (!runFilter) return systemLogs;
    return systemLogs.filter((entry) => entry.run_id === runFilter || entry.trace_id === runFilter);
  }, [systemLogs, runFilter]);

  const filteredMonologueEntries = useMemo(() => {
    if (!runFilter) return monologueEntries;
    return monologueEntries.filter((entry) => entry.run_id === runFilter);
  }, [monologueEntries, runFilter]);

  const filteredMessages = useMemo(() => {
    if (!runFilter) return messages;
    return messages.filter((msg) => msg.run_id === runFilter || msg.trace_id === runFilter);
  }, [messages, runFilter]);

  const filteredSubjectSnapshots = useMemo(() => {
    if (!runFilter) return subjectSnapshots;
    return subjectSnapshots.filter((snap) => snap.run_id === runFilter);
  }, [subjectSnapshots, runFilter]);

  const runScopedSnapshotHashes = useMemo(() => {
    return new Set(filteredSubjectSnapshots.map((snap) => snap.snapshot_hash));
  }, [filteredSubjectSnapshots]);

  const filteredGateDecisions = useMemo(() => {
    if (!runFilter) return gateDecisions;
    if (runScopedSnapshotHashes.size === 0) return [];
    return gateDecisions.filter((decision) => runScopedSnapshotHashes.has(decision.snapshot_hash));
  }, [gateDecisions, runFilter, runScopedSnapshotHashes]);

  const filteredGateInputEvents = useMemo(() => {
    if (!runFilter) return gateInputEvents;
    return gateInputEvents.filter((entry) => entry.run_id === runFilter || entry.trace_id === runFilter);
  }, [gateInputEvents, runFilter]);

  const combinedEntries = useMemo(() => {
    const entries = [
      ...buildSystemEntries(filteredSystemLogs),
      ...buildMonologueEntries(filteredMonologueEntries),
      ...buildMessageEntries(filteredMessages),
    ];
    return entries.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }, [filteredSystemLogs, filteredMonologueEntries, filteredMessages]);

  const decisionReports = useMemo(
    () => filteredSystemLogs.filter((entry) => (entry.payload as any)?.event === "decision_report").slice(0, 6),
    [filteredSystemLogs]
  );

  const feedbackBundlePayload = useMemo(() => {
    for (const entry of filteredSystemLogs) {
      const payload = entry.payload as any;
      if (payload && typeof payload === "object" && payload.event === "feedback_bundle_built") {
        return payload.bundle ?? payload.feedback_bundle ?? null;
      }
    }
    return null;
  }, [filteredSystemLogs]);

  const feedbackBundleLines = useMemo(() => {
    if (!feedbackBundlePayload || typeof feedbackBundlePayload !== "object") return [];
    const entries = Object.entries(feedbackBundlePayload as Record<string, any>);
    return entries.map(([key, value]) => {
      if (typeof value === "string") return `${key}: ${value}`;
      return `${key}: ${JSON.stringify(value)}`;
    });
  }, [feedbackBundlePayload]);

  const skipSummary = useMemo(() => {
    const tracked = new Set([
      "summary_archive_skipped",
      "self_claim_rejected",
      "self_reflection_skipped",
      "self_reflection_missing_evidence",
      "self_claim_missing_evidence",
      "self_claim_stale_evidence",
    ]);
    const counts: Record<string, number> = {};
    for (const entry of filteredSystemLogs) {
      const payload = entry.payload as any;
      const event = payload?.event;
      if (event && tracked.has(event)) {
        counts[event] = (counts[event] || 0) + 1;
      }
    }
    return counts;
  }, [filteredSystemLogs]);

  const latestAttentionSchema = useMemo(() => {
    const latest = filteredSubjectSnapshots[0];
    if (!latest) return null;
    try {
      const parsed = JSON.parse(latest.subject_state_json) as any;
      return parsed?.state?.attention_schema ?? null;
    } catch {
      return null;
    }
  }, [filteredSubjectSnapshots]);

  const latestWorkspaceContributors = useMemo(() => {
    for (const entry of filteredSystemLogs) {
      const payload = entry.payload as any;
      if (payload && typeof payload === "object" && payload.event === "workspace_snapshot") {
        return payload;
      }
    }
    return null;
  }, [filteredSystemLogs]);

  const selfReportMessages = useMemo(
    () => filteredMessages.filter((msg) => msg.role === "assistant" && (msg.metadata as any)?.self_report),
    [filteredMessages]
  );

  const decisionTargets = useMemo(() => {
    const targets = new Map<string, number>();
    for (const entry of evidenceLineage) {
      for (const link of entry.links) {
        if (!isDecisionTarget(link.target_type)) continue;
        const key = `${link.target_type}:${link.target_id}`;
        targets.set(key, (targets.get(key) ?? 0) + 1);
      }
    }
    return Array.from(targets.entries())
      .map(([id, count]) => ({ id, count }))
      .sort((a, b) => b.count - a.count);
  }, [evidenceLineage]);

  const filteredEvidenceLineage = useMemo(() => {
    if (decisionFilter === "all") return evidenceLineage;
    return evidenceLineage.filter((entry) =>
      entry.links.some((link) => `${link.target_type}:${link.target_id}` === decisionFilter)
    );
  }, [evidenceLineage, decisionFilter]);

  const subsystemStates = useMemo(() => {
    const raw = systemHealthSnapshot?.subsystem_states;
    if (Array.isArray(raw)) {
      return raw as SubsystemState[];
    }
    return [] as SubsystemState[];
  }, [systemHealthSnapshot]);
  const suppressionSummary = useMemo(() => {
    const counts = new Map<string, number>();
    for (const entry of filteredSystemLogs) {
      const payload = entry.payload as any;
      if (!payload || typeof payload !== "object") continue;
      if (payload.event === "monologue_suppression_summary" && payload.suppression_counts) {
        for (const [reason, count] of Object.entries(payload.suppression_counts)) {
          counts.set(reason, (counts.get(reason) ?? 0) + Number(count ?? 0));
        }
      } else if (payload.event === "monologue_candidate_blocked" && payload.reason) {
        const reason = String(payload.reason);
        counts.set(reason, (counts.get(reason) ?? 0) + 1);
      }
    }
    return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 6);
  }, [filteredSystemLogs]);

  const keyEvents = useMemo(() => {
    return [...filteredSystemLogs]
      .filter((entry) => entry.level === "warn" || entry.level === "error")
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, 8);
  }, [filteredSystemLogs]);

  const recentEvents = useMemo(() => {
    return [...filteredSystemLogs]
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, 5);
  }, [filteredSystemLogs]);

  const deltaItems = useMemo(() => {
    if (!systemHealthSnapshot || systemHealthHistory.length < 2) return [];
    const current = (systemHealthSnapshot.metrics ?? {}) as Record<string, any>;
    const previous = (systemHealthHistory[1]?.metrics ?? {}) as Record<string, any>;
    const items: { label: string; current: string; delta: string; tone: "up" | "down" | "flat" }[] = [];
    const pushDelta = (label: string, currentVal: number, prevVal: number, precision = 0) => {
      if (!Number.isFinite(currentVal) || !Number.isFinite(prevVal)) return;
      const delta = currentVal - prevVal;
      if (Math.abs(delta) < (precision > 0 ? 0.01 : 1)) return;
      const sign = delta > 0 ? "+" : "";
      items.push({
        label,
        current: precision > 0 ? currentVal.toFixed(precision) : `${Math.round(currentVal)}`,
        delta: `${sign}${precision > 0 ? delta.toFixed(precision) : Math.round(delta)}`,
        tone: delta > 0 ? "up" : "down",
      });
    };
    pushDelta("Errors Open", Number(current.errors?.open ?? 0), Number(previous.errors?.open ?? 0));
    pushDelta("Pending Prompts", Number(current.pending_prompts?.count ?? 0), Number(previous.pending_prompts?.count ?? 0));
    pushDelta("Gate Verify Rate", Number(current.gate?.verify_rate ?? 0) * 100, Number(previous.gate?.verify_rate ?? 0) * 100, 1);
    pushDelta("Memory Writes", Number(current.memory?.write_count ?? 0), Number(previous.memory?.write_count ?? 0));
    pushDelta("JSON Compliance", Number(current.json_compliance?.compliance_rate ?? 0) * 100, Number(previous.json_compliance?.compliance_rate ?? 0) * 100, 1);
    pushDelta("Monologue No-op", Number(current.monologue?.loop_noop_rate ?? 0) * 100, Number(previous.monologue?.loop_noop_rate ?? 0) * 100, 1);
    return items.slice(0, 8);
  }, [systemHealthSnapshot, systemHealthHistory]);

  const runStoryEvents = useMemo(() => {
    const items: RunStoryEvent[] = [];
    const runMeta = (systemHealthSnapshot?.metrics as any)?.run ?? {};
    const runMetaId = runMeta.active_run_id ? String(runMeta.active_run_id) : null;
    if (runMeta && (runMeta.active_run_id || runMeta.module_stage)) {
      if (!runFilter || runMetaId === runFilter) {
        const timestamp = runMeta.module_updated_at
          || systemHealthSnapshot?.timestamp
          || new Date().toISOString();
        const detailParts: string[] = [];
        if (runMeta.module_stage) {
          detailParts.push(formatEventLabel(String(runMeta.module_stage)));
        }
        if (runMeta.module_detail) {
          detailParts.push(String(runMeta.module_detail));
        }
        if (runMeta.active_run_id) {
          detailParts.push(`Run ${shortHash(String(runMeta.active_run_id))}`);
        }
        items.push({
          id: `run-stage:${timestamp}`,
          label: "Run stage update",
          timestamp,
          event: "run_stage_update",
          detail: detailParts.length > 0 ? detailParts.join(" | ") : undefined,
          run_id: runMetaId,
          source: "system",
        });
      }
    }
    for (const entry of filteredSystemLogs) {
      const payload = entry.payload as any;
      const eventName = payload && typeof payload === "object" && "event" in payload
        ? String(payload.event)
        : entry.category;
      if (!RUN_STORY_EVENTS.has(eventName)) continue;
      const detailKeys = ["decision", "reason", "status", "outcome", "tool_name", "candidate_kind", "error"];
      const detail = detailKeys
        .map((key) => payload?.[key])
        .find((value) => typeof value === "string" && value.trim().length > 0) as string | undefined;
      items.push({
        id: entry.id,
        label: formatEventLabel(eventName),
        timestamp: entry.timestamp,
        event: eventName,
        detail: detail ? truncate(detail, 140) : undefined,
        run_id: entry.run_id,
        source: "system",
      });
    }
    for (const msg of filteredMessages) {
      if (msg.role === "internal") continue;
      items.push({
        id: `message:${msg.message_id}`,
        label: msg.role === "user" ? "User message" : "Assistant response",
        timestamp: msg.created_at || new Date().toISOString(),
        event: `message_${msg.role}`,
        detail: truncate(String(msg.content ?? ""), 140),
        run_id: msg.run_id ?? null,
        source: "message",
      });
    }
    return items.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  }, [filteredSystemLogs, filteredMessages, systemHealthSnapshot, runFilter]);

  useEffect(() => {
    if (runStoryEvents.length === 0) {
      setPlaybackIndex(0);
      return;
    }
    setPlaybackIndex((prev) => Math.min(prev, runStoryEvents.length - 1));
  }, [runStoryEvents]);

  const rawHealth = useMemo(() => {
    if (!systemHealthSnapshot) return "{}";
    return JSON.stringify(systemHealthSnapshot.metrics ?? {}, null, 2);
  }, [systemHealthSnapshot]);

  const rawSubsystems = useMemo(() => {
    if (!subsystemStates.length) return "[]";
    return JSON.stringify(subsystemStates, null, 2);
  }, [subsystemStates]);

  const flowStats = useMemo(() => {
    const metrics = (systemHealthSnapshot?.metrics ?? {}) as Record<string, any>;
    const run = metrics.run ?? {};
    const gate = metrics.gate ?? {};
    const tools = metrics.tools ?? {};
    const memory = metrics.memory ?? {};
    const summaries = metrics.summaries ?? {};
    const errors = metrics.errors ?? {};
    const pending = metrics.pending_prompts ?? {};

    const summaryTotal = Number(summaries.rolling_updates ?? 0) + Number(summaries.inner_updates ?? 0);
    const summaryFailures = Number(summaries.rolling_failures ?? 0) + Number(summaries.inner_failures ?? 0);
    const pendingAgeMin = pending.oldest_age_seconds ? Math.round(Number(pending.oldest_age_seconds) / 60) : null;

    return [
      {
        label: "Active Run",
        value: run.active_run_id ? shortHash(String(run.active_run_id)) : "Idle",
        meta: run.module_stage ? formatEventLabel(String(run.module_stage)) : "No active stage",
        tooltip: "Single end-to-end execution thread for the current request.",
      },
      {
        label: "Gate Decisions",
        value: `${gate.total ?? 0}`,
        meta: `Verify ${(Number(gate.verify_rate ?? 0) * 100).toFixed(0)}%`,
        tooltip: "Policy gate decisions and verification rate for the run.",
      },
      {
        label: "Tool Dispatch",
        value: `${tools.dispatches ?? 0}`,
        meta: `${tools.failures ?? 0} failures`,
        tooltip: "Tool calls issued by the system and their failure count.",
      },
      {
        label: "Memory",
        value: `${memory.memory_pass_count ?? 0} passes`,
        meta: `${memory.write_count ?? 0} writes`,
        tooltip: "Memory passes and writes in the current window.",
      },
      {
        label: "Summaries",
        value: `${summaryTotal} updates`,
        meta: summaryFailures ? `${summaryFailures} failures` : "No failures",
        tooltip: "Rolling and inner summary updates.",
      },
      {
        label: "Errors",
        value: `${errors.total ?? 0}`,
        meta: `${errors.open ?? 0} open`,
        tooltip: "Total and open errors recorded recently.",
      },
      {
        label: "Pending Prompts",
        value: `${pending.count ?? 0}`,
        meta: pendingAgeMin !== null ? `Oldest ${pendingAgeMin}m` : "--",
        tooltip: "Queued prompts waiting to be processed.",
      },
    ];
  }, [systemHealthSnapshot]);

  const activityEntries = useMemo(() => combinedEntries.slice(0, 12), [combinedEntries]);

  const moduleStatus: ModuleStatus | null = useMemo(() => {
    const run = (systemHealthSnapshot?.metrics ?? {}).run ?? {};
    if (!run.module_stage) return null;
    return {
      stage: run.module_stage,
      detail: run.module_detail ?? null,
      started_at: run.module_updated_at ?? null,
    };
  }, [systemHealthSnapshot]);

  const statusLabel = moduleStatus?.stage
    ? formatStatusLabel(moduleStatus.stage)
    : "Idle";
  const statusDetail = moduleStatus?.stage
    ? formatStatusDetail(moduleStatus.stage, moduleStatus.detail)
    : "Standing by";

  const statusMetrics = useMemo(() => {
    const metrics = (systemHealthSnapshot?.metrics ?? {}) as Record<string, any>;
    const gate = metrics.gate ?? {};
    const organism = metrics.organism ?? {};
    const controller = metrics.controller ?? {};
    const errors = metrics.errors ?? {};
    const pending = metrics.pending_prompts ?? {};
    return {
      gate,
      organism,
      controller,
      errors,
      pending,
    };
  }, [systemHealthSnapshot]);

  const latestGateInput = filteredGateInputEvents[0];
  const latestGatePayload = (latestGateInput?.payload ?? {}) as any;
  const gateDecision =
    latestGatePayload.enforced_decision
    || latestGatePayload.soft_decision
    || latestGatePayload.legacy_decision
    || statusMetrics.gate?.last_decision
    || "--";
  const gateVerifyRate = Number(statusMetrics.gate?.verify_rate ?? 0);
  const organismStress = Number(statusMetrics.organism?.stress ?? 0);
  const controllerConfidence = Number(statusMetrics.controller?.confidence ?? 0.5);
  const errorOpen = Number(statusMetrics.errors?.open ?? 0);
  const pendingCount = Number(statusMetrics.pending?.count ?? 0);

  const eventCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const entry of filteredSystemLogs) {
      const payload = entry.payload as any;
      const eventName = payload && typeof payload === "object" && "event" in payload
        ? String(payload.event)
        : entry.category;
      counts.set(eventName, (counts.get(eventName) ?? 0) + 1);
    }
    return counts;
  }, [filteredSystemLogs]);

  const jsonCompliance = (systemHealthSnapshot?.metrics ?? {}) as Record<string, any>;
  const jsonMetrics = jsonCompliance.json_compliance ?? {};
  const memoryMetrics = jsonCompliance.memory ?? {};
  const monologueMetrics = jsonCompliance.monologue ?? {};
  const contractViolationCount = eventCounts.get("contract_violation") ?? 0;
  const memoryWriteBlockedCount = eventCounts.get("memory_write_blocked") ?? 0;
  const jsonComplianceRate = Number(jsonMetrics.compliance_rate ?? 1);
  const jsonInvalid = Number(jsonMetrics.invalid ?? 0);
  const loopNoopStreak = Number(monologueMetrics.loop_noop_streak ?? 0);

  const activeAlerts = useMemo(() => {
    const items: AlertItem[] = [];
    if (errorOpen > 0) {
      items.push({
        id: "errors_open",
        label: "Errors Open",
        value: `${errorOpen}`,
        tone: "alert",
        reason: "Open errors in the last health snapshot.",
        relatedEvents: ["tool_dispatch_failed", "memory_pass_error", "contract_violation"],
      });
    }
    if (pendingCount > 0) {
      items.push({
        id: "pending_prompts",
        label: "Pending Prompts",
        value: `${pendingCount}`,
        tone: "warn",
        reason: "Pending prompts are waiting to surface.",
        relatedEvents: ["pending_prompt_surfaced", "pending_prompt_held", "pending_prompt_starvation"],
      });
    }
    if (gateVerifyRate > 0.4) {
      items.push({
        id: "gate_verify",
        label: "Gate Verify",
        value: `${Math.round(gateVerifyRate * 100)}%`,
        tone: "warn",
        reason: "Verification rate is elevated.",
        relatedEvents: ["gate_decision"],
      });
    }
    if (organismStress > 0.7) {
      items.push({
        id: "organism_stress",
        label: "Organism Stress",
        value: organismStress.toFixed(2),
        tone: "warn",
        reason: "Stress signal is above the comfort band.",
        relatedEvents: ["organism_state", "controller_state"],
      });
    }
    if (controllerConfidence < 0.4) {
      items.push({
        id: "controller_confidence",
        label: "Low Confidence",
        value: `${Math.round(controllerConfidence * 100)}%`,
        tone: "warn",
        reason: "Controller confidence is low.",
        relatedEvents: ["decision_report", "gate_decision"],
      });
    }
    if (jsonInvalid > 0 || jsonComplianceRate < 0.85) {
      items.push({
        id: "json_compliance",
        label: "JSON Compliance",
        value: `${Math.round(jsonComplianceRate * 100)}%`,
        tone: "warn",
        reason: "Strict JSON tasks are failing or retrying.",
        relatedEvents: ["response_reasoning_json_invalid", "prediction_generation_retry", "qualia_auto_label_retry"],
      });
    }
    if (memoryWriteBlockedCount > 0 || Number(memoryMetrics.phi_write_blocked ?? 0) > 0) {
      items.push({
        id: "memory_blocked",
        label: "Memory Writes Blocked",
        value: `${memoryWriteBlockedCount || memoryMetrics.phi_write_blocked || 0}`,
        tone: "alert",
        reason: "Evidence gating blocked memory writes.",
        relatedEvents: ["memory_write_blocked", "memory_pass_invalid_output"],
      });
    }
    if (contractViolationCount > 0) {
      items.push({
        id: "contract_violation",
        label: "Contract Violations",
        value: `${contractViolationCount}`,
        tone: "warn",
        reason: "Ungrounded assertions were detected.",
        relatedEvents: ["contract_violation"],
      });
    }
    if (loopNoopStreak >= 3) {
      items.push({
        id: "monologue_loop",
        label: "Monologue Loop",
        value: `${loopNoopStreak}`,
        tone: "warn",
        reason: "Monologue loop/no-op streak detected.",
        relatedEvents: ["monologue_loop_detected", "monologue_loop_circuit_breaker"],
      });
    }
    return items
      .sort((a, b) => (a.tone === b.tone ? 0 : a.tone === "alert" ? -1 : 1))
      .slice(0, 5);
  }, [
    errorOpen,
    pendingCount,
    gateVerifyRate,
    organismStress,
    controllerConfidence,
    jsonComplianceRate,
    jsonInvalid,
    memoryMetrics,
    memoryWriteBlockedCount,
    contractViolationCount,
    loopNoopStreak,
  ]);

  useEffect(() => {
    if (activeAlertId && !activeAlerts.some((alert) => alert.id === activeAlertId)) {
      setActiveAlertId(null);
    }
  }, [activeAlertId, activeAlerts]);

  const alertEventMap = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const alert of activeAlerts) {
      for (const event of alert.relatedEvents) {
        const list = map.get(event) ?? [];
        list.push(alert.id);
        map.set(event, list);
      }
    }
    return map;
  }, [activeAlerts]);

  const entriesWithMeta = useMemo(() => {
    return combinedEntries.map((entry) => {
      const severity = EVENT_SEVERITY[entry.event] ?? (entry.level as "info" | "warn" | "error");
      const alertSet = new Set(alertEventMap.get(entry.event) ?? []);
      const evidence_ids: number[] = [];
      if (entry.payload && typeof entry.payload === "object") {
        const payload = entry.payload as any;
        if (typeof payload.alert_id === "string") {
          alertSet.add(payload.alert_id);
        }
        if (Array.isArray(payload.alert_ids)) {
          for (const id of payload.alert_ids) {
            if (typeof id === "string") alertSet.add(id);
          }
        }
        const list = payload.evidence_event_ids;
        if (Array.isArray(list)) {
          for (const id of list) {
            if (typeof id === "number") evidence_ids.push(id);
          }
        }
        if (typeof payload.evidence_event_id === "number") {
          evidence_ids.push(payload.evidence_event_id);
        }
        if (typeof payload.evidence_id === "number") {
          evidence_ids.push(payload.evidence_id);
        }
      }
      const deduped = Array.from(new Set(evidence_ids));
      return { ...entry, severity, alert_ids: Array.from(alertSet), evidence_ids: deduped };
    });
  }, [combinedEntries, alertEventMap]);

  const categories = useMemo(() => {
    const unique = new Set<string>(entriesWithMeta.map((entry) => entry.category));
    return ["all", ...Array.from(unique).sort()];
  }, [entriesWithMeta]);

  const levels = ["all", "info", "warn", "error"];

  const filteredEntries = useMemo(() => {
    const query = search.trim().toLowerCase();
    return entriesWithMeta.filter((entry) => {
      const level = entry.severity ?? entry.level;
      if (activeCategory !== "all" && entry.category !== activeCategory) return false;
      if (activeLevel !== "all" && level !== activeLevel) return false;
      if (activeAlertId && !(entry.alert_ids ?? []).includes(activeAlertId)) return false;
      if (!query) return true;
      const payloadText = typeof entry.payload === "string"
        ? entry.payload
        : JSON.stringify(entry.payload);
      return (
        entry.event.toLowerCase().includes(query)
        || entry.category.toLowerCase().includes(query)
        || payloadText.toLowerCase().includes(query)
      );
    });
  }, [entriesWithMeta, activeCategory, activeLevel, search, activeAlertId]);

  const groupedEntries = useMemo(() => {
    const map = new Map<string, TraceEntry[]>();
    for (const entry of filteredEntries) {
      const key = runIdKey(entry.run_id);
      const list = map.get(key) || [];
      list.push(entry);
      map.set(key, list);
    }
    return Array.from(map.entries());
  }, [filteredEntries]);

  const tabs = useMemo(() => ([
    { id: "overview" as const, label: "Overview", badge: activeAlerts.length },
    { id: "controls" as const, label: "Controls", badge: 0 },
    { id: "signals" as const, label: "Signals", badge: pendingCount },
    { id: "diagnostics" as const, label: "Diagnostics", badge: keyEvents.length },
    { id: "logs" as const, label: "Logs", badge: errorOpen },
  ]), [activeAlerts.length, pendingCount, keyEvents.length, errorOpen]);

  const toggleRun = (key: string) => {
    setCollapsedRuns((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const handleSaveIntent = async () => {
    if (!allowEdits || intentSaving) return;
    const summary = intentDraft.trim();
    if (!summary) return;
    setIntentSaving(true);
    try {
      await invokeWithTimeout("update_user_intent_summary", {
        summary,
        confirmed: intentConfirmed,
        evidence_event_ids: [],
      }, INTENT_SAVE_TIMEOUT_MS);
      onRefresh();
    } finally {
      setIntentSaving(false);
    }
  };

  const handleWeightChange = (key: typeof WEIGHT_DEFS[number]["key"], value: number) => {
    if (!settings || !onUpdateSettings || !allowEdits) return;
    const next = { ...settings, [key]: value };
    onUpdateSettings(next);
  };

  const handleResetWeights = () => {
    if (!settings || !onUpdateSettings || !allowEdits) return;
    const next = {
      ...settings,
      weight_user_satisfaction: 0.5,
      weight_policy_rigor: 0.5,
      weight_latency: 0.5,
      weight_evidence_strictness: 0.5,
      weight_exploration: 0.5,
    };
    onUpdateSettings(next);
  };

  const handleRunChange = (nextRunId: string) => {
    setActiveRunId(nextRunId);
    setRunSelectionLocked(true);
    setActiveAlertId(null);
  };

  const releaseRunLock = () => {
    setRunSelectionLocked(false);
    setActiveRunId("all");
  };

  const handleSaveLens = () => {
    const label = lensDraft.trim();
    if (!label) return;
    const id = typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `lens_${Date.now()}`;
    const lens: LensPreset = {
      id,
      label,
      runId: activeRunId,
      category: activeCategory,
      level: activeLevel,
      search,
      alertId: activeAlertId,
    };
    setSavedLenses((prev) => [lens, ...prev].slice(0, 12));
    setLensDraft("");
  };

  const handleApplyLens = (lens: LensPreset) => {
    setActiveRunId(lens.runId || "all");
    setRunSelectionLocked(true);
    setActiveCategory(lens.category || "all");
    setActiveLevel(lens.level || "all");
    setSearch(lens.search || "");
    setActiveAlertId(lens.alertId ?? null);
    setActiveTab("logs");
  };

  const handleDeleteLens = (lensId: string) => {
    setSavedLenses((prev) => prev.filter((lens) => lens.id !== lensId));
  };

  const clearFilters = () => {
    setActiveCategory("all");
    setActiveLevel("all");
    setSearch("");
    setActiveAlertId(null);
  };

  const handleAlertTrace = (alert: AlertItem) => {
    setActiveAlertId(alert.id);
    setActiveCategory("all");
    setActiveLevel("all");
    setSearch(alert.relatedEvents[0] ?? "");
    setActiveTab("logs");
  };

  const handleJumpToLogEvent = (eventName: string) => {
    setActiveCategory("all");
    setActiveLevel("all");
    setSearch(eventName);
    setActiveTab("logs");
  };

  const renderActivityFlow = () => (
    <section className="cockpit-panel flow-panel">
      <div className="panel-header">
        <div>
          <h2>Activity Flow</h2>
          <p>Live pipeline signals and most recent system events.</p>
        </div>
      </div>
      <div className="flow-metrics">
        {flowStats.map((stat) => (
          <div key={stat.label} className="flow-card">
            <div className="flow-card-label" title={stat.tooltip}>{stat.label}</div>
            <div className="flow-card-value">{stat.value}</div>
            <div className="flow-card-meta">{stat.meta}</div>
          </div>
        ))}
      </div>
      <div className="activity-feed">
        {activityEntries.length === 0 ? (
          <div className="panel-empty">No recent activity.</div>
        ) : (
          activityEntries.map((entry) => (
            <div key={entry.id} className={`activity-row activity-${entry.level}`}>
              <div>
                <div className="activity-title">{formatEventLabel(entry.event)}</div>
                <div className="activity-meta">
                  {entry.category} - {new Date(entry.timestamp).toLocaleTimeString()}
                </div>
              </div>
              <span className="activity-level">{entry.level}</span>
            </div>
          ))
        )}
      </div>
    </section>
  );

  const renderSystemStatus = () => (
    <section className="cockpit-panel status-panel">
      <div className="panel-header">
        <div>
          <h2>System Status</h2>
          <p>Critical posture signals and run-level posture.</p>
        </div>
      </div>
      <div className="status-grid">
        {flowStats.map((stat) => (
          <div key={stat.label} className="status-card">
            <div className="status-label" title={stat.tooltip}>{stat.label}</div>
            <div className="status-value">{stat.value}</div>
            <div className="status-meta">{stat.meta}</div>
          </div>
        ))}
      </div>
    </section>
  );

  const renderActiveAlerts = () => (
    <section className="cockpit-panel alert-panel">
      <div className="panel-header">
        <div>
          <h2>Active Alerts</h2>
          <p>Top issues that need attention right now.</p>
        </div>
      </div>
      {activeAlerts.length === 0 ? (
        <div className="panel-empty">No active alerts.</div>
      ) : (
        <div className="alert-list">
          {activeAlerts.map((alert) => (
            <div key={alert.id} className={`alert-row alert-${alert.tone}`}>
              <div>
                <strong>{alert.label}</strong>
                <div className="alert-reason">{alert.reason}</div>
              </div>
              <div className="alert-actions">
                <span className="alert-value">{alert.value}</span>
                <button className="btn btn-secondary btn-compact" onClick={() => handleAlertTrace(alert)}>
                  Why
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );

  const renderDeltaPanel = () => (
    <section className="cockpit-panel delta-panel">
      <div className="panel-header">
        <div>
          <h2>What Changed</h2>
          <p>Delta from the previous health snapshot.</p>
        </div>
      </div>
      {deltaItems.length === 0 ? (
        <div className="panel-empty">No material changes since the last snapshot.</div>
      ) : (
        <div className="delta-list">
          {deltaItems.map((item) => (
            <div key={item.label} className={`delta-row delta-${item.tone}`}>
              <span className="delta-label">{item.label}</span>
              <span className="delta-current">{item.current}</span>
              <span className="delta-change">{item.delta}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );

  const renderRunStory = () => (
    <section className="cockpit-panel run-story-panel">
      <div className="panel-header">
        <div>
          <h2>Run Story</h2>
          <p>Chronological narrative for the focused run.</p>
        </div>
      </div>
      {runStoryEvents.length === 0 ? (
        <div className="panel-empty">No run events available yet.</div>
      ) : (
        <div className="run-story-list">
          {runStoryEvents.slice(-14).map((event) => (
            <div key={event.id} className="run-story-row">
              <div className="run-story-time">
                {new Date(event.timestamp).toLocaleTimeString()}
              </div>
              <div className="run-story-body">
                <div className="run-story-title">{event.label}</div>
                {event.detail && (
                  <div className="run-story-detail">{event.detail}</div>
                )}
              </div>
              <button
                className="btn btn-secondary btn-compact"
                onClick={() => handleJumpToLogEvent(event.event)}
              >
                Trace
              </button>
            </div>
          ))}
        </div>
      )}
    </section>
  );

  const renderPlayback = () => {
    if (runStoryEvents.length === 0) {
      return (
        <section className="cockpit-panel playback-panel">
          <div className="panel-header">
            <div>
              <h2>Playback</h2>
              <p>Step through the run timeline.</p>
            </div>
          </div>
          <div className="panel-empty">No run events to play yet.</div>
        </section>
      );
    }
    const current = runStoryEvents[Math.min(playbackIndex, runStoryEvents.length - 1)];
    return (
      <section className="cockpit-panel playback-panel">
        <div className="panel-header">
          <div>
            <h2>Playback</h2>
            <p>Step through the focused run.</p>
          </div>
        </div>
        <div className="playback-controls">
          <button
            className="btn btn-secondary btn-compact"
            onClick={() => setPlaybackIndex((prev) => Math.max(0, prev - 1))}
          >
            Prev
          </button>
          <input
            type="range"
            min={0}
            max={Math.max(0, runStoryEvents.length - 1)}
            value={Math.min(playbackIndex, runStoryEvents.length - 1)}
            onChange={(e) => setPlaybackIndex(Number(e.target.value))}
          />
          <button
            className="btn btn-secondary btn-compact"
            onClick={() => setPlaybackIndex((prev) => Math.min(runStoryEvents.length - 1, prev + 1))}
          >
            Next
          </button>
        </div>
        <div className="playback-card">
          <div className="playback-title">{current.label}</div>
          <div className="playback-time">{new Date(current.timestamp).toLocaleTimeString()}</div>
          <div className="playback-detail">{current.detail || "No detail captured."}</div>
          <button
            className="btn btn-secondary btn-compact"
            onClick={() => handleJumpToLogEvent(current.event)}
          >
            Open Logs
          </button>
        </div>
      </section>
    );
  };

  const renderSkipSummary = () => {
    const items = Object.entries(skipSummary);
    if (items.length === 0) return null;
    return (
      <section className="cockpit-panel">
        <div className="panel-header">
          <div>
            <h2>Skip and Reject Summary</h2>
            <p>Recent skip and reject counts from system logs.</p>
          </div>
        </div>
        <div className="panel-body">
          {items.map(([event, count]) => (
            <div key={event} className="panel-line">
              <span>{formatEventLabel(event)}</span>
              <strong>{count}</strong>
            </div>
          ))}
        </div>
      </section>
    );
  };

  const renderFeedbackLoop = () => (
    <section className="cockpit-panel feedback-loop-panel">
      <div className="panel-header">
        <div>
          <h2>Feedback Loop</h2>
          <p>What the model perceives each turn: bundle, context tags, and intent summary.</p>
        </div>
      </div>
      <div className="feedback-loop-grid">
        <div className="feedback-block">
          <h3>Feedback Bundle</h3>
          {feedbackBundleLines.length === 0 ? (
            <div className="panel-empty">No feedback bundle logged yet.</div>
          ) : (
            <pre className="feedback-pre">{feedbackBundleLines.join("\n")}</pre>
          )}
        </div>
        <div className="feedback-block">
          <h3>Context Tags</h3>
          {contextTags.length === 0 ? (
            <div className="panel-empty">No context tags yet.</div>
          ) : (
            <div className="tag-list">
              {contextTags.map((tag) => (
                <div key={`${tag.tag}-${tag.last_seen_at}`} className="tag-row">
                  <span className="tag-name">{tag.tag}</span>
                  <span className="tag-meta">
                    {tag.confidence.toFixed(2)} {tag.inferred ? "inferred" : "evidence"} -
                    {" "}
                    {tag.evidence_event_ids.length > 0 ? tag.evidence_event_ids.join(", ") : "no evidence"}
                  </span>
                  <span className="tag-time">{new Date(tag.last_seen_at).toLocaleTimeString()}</span>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="feedback-block">
          <h3>User Intent Summary</h3>
          <textarea
            className="input intent-textarea"
            value={intentDraft}
            onChange={(e) => setIntentDraft(e.target.value)}
            placeholder="Summarize the user's intent (confirmation required)."
            disabled={!allowEdits}
          />
          <div className="intent-actions">
            <label className="intent-confirm">
              <input
                type="checkbox"
                checked={intentConfirmed}
                onChange={(e) => setIntentConfirmed(e.target.checked)}
                disabled={!allowEdits}
              />
              Confirmed by user
            </label>
            <button
              className="btn btn-secondary"
              onClick={handleSaveIntent}
              disabled={!allowEdits || intentSaving || !intentDraft.trim()}
            >
              {intentSaving ? "Saving..." : "Save"}
            </button>
          </div>
          {intentSummary && intentSummary.evidence_event_ids.length > 0 && (
            <div className="intent-evidence">
              Evidence: {intentSummary.evidence_event_ids.join(", ")}
            </div>
          )}
        </div>
      </div>
    </section>
  );

  const renderSelfReport = () => (
    <section className="cockpit-panel self-report-panel">
      <div className="panel-header">
        <div>
          <h2>Self-Report Lane</h2>
          <p>Unverified self-report messages flagged in output.</p>
        </div>
      </div>
      {selfReportMessages.length === 0 ? (
        <div className="panel-empty">No self-report messages detected.</div>
      ) : (
        <div className="self-report-list">
          {selfReportMessages.slice(0, 6).map((msg) => (
            <div key={msg.message_id} className="self-report-row">
              <div className="self-report-content">{truncate(msg.content, 140)}</div>
              <div className="self-report-time">{msg.created_at || "--"}</div>
            </div>
          ))}
        </div>
      )}
    </section>
  );

  const renderDecisionReports = () => (
    <section className="cockpit-panel decision-report-panel">
      <div className="panel-header">
        <div>
          <h2>Decision Reports</h2>
          <p>Latest kernel decisions and rationale markers.</p>
        </div>
      </div>
      {decisionReports.length === 0 ? (
        <div className="panel-empty">No decision reports yet.</div>
      ) : (
        <div className="decision-report-list">
          {decisionReports.map((entry) => {
            const payload = entry.payload as any;
            const selectedAction = payload?.selected_action ?? "--";
            const gateDecision = payload?.gate_decision ?? "--";
            const rationale = payload?.rationale ?? "--";
            return (
              <div key={entry.id} className="decision-report-row">
                <div className="decision-report-title">
                  <strong>{selectedAction}</strong> | gate {gateDecision}
                </div>
                <div className="history-meta">{new Date(entry.timestamp).toLocaleTimeString()}</div>
                <div className="decision-report-body">{truncate(String(rationale), 140)}</div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );

  const renderEventTimeline = () => (
    <section className="cockpit-panel event-panel">
      <div className="panel-header">
        <div>
          <h2>Event Timeline</h2>
          <p>Latest warning or error events across the system.</p>
        </div>
      </div>
      {keyEvents.length === 0 ? (
        <div className="panel-empty">No warning or error events.</div>
      ) : (
        <div className="event-list">
          {keyEvents.map((entry) => {
            const payload = entry.payload as any;
            const eventName = payload && typeof payload === "object" && "event" in payload
              ? String(payload.event)
              : entry.category;
            return (
              <div key={entry.id} className={`event-row event-${entry.level}`}>
                <div>
                  <strong>{formatEventLabel(eventName)}</strong>
                  <div className="history-meta">{entry.category} - {new Date(entry.timestamp).toLocaleTimeString()}</div>
                </div>
                <div className="history-meta">{entry.level}</div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );

  const renderEvidenceLineage = () => (
    <section className="cockpit-panel evidence-lineage-panel">
      <div className="panel-header">
        <div>
          <h2>Evidence Lineage</h2>
          <p>Recent evidence sources and where they are linked.</p>
        </div>
      </div>
      {decisionTargets.length > 0 && (
        <div className="evidence-lineage-filters">
          <label>
            Decision filter
            <select
              className="input evidence-filter-select"
              value={decisionFilter}
              onChange={(e) => setDecisionFilter(e.target.value)}
            >
              <option value="all">All decisions</option>
              {decisionTargets.map((target) => (
                <option key={target.id} value={target.id}>
                  {target.id} ({target.count})
                </option>
              ))}
            </select>
          </label>
          {decisionFilter !== "all" && (
            <button className="btn btn-secondary btn-compact" onClick={() => setDecisionFilter("all")}>
              Clear
            </button>
          )}
        </div>
      )}
      {evidenceLineageError ? (
        <div className="panel-empty">Failed to load evidence lineage.</div>
      ) : filteredEvidenceLineage.length === 0 ? (
        <div className="panel-empty">No evidence lineage entries yet.</div>
      ) : (
        <div className="evidence-lineage-list">
          {filteredEvidenceLineage.slice(0, 12).map((entry) => (
            <div key={entry.source.evidence_id} className="evidence-lineage-row">
              <div className="evidence-lineage-meta">
                <div className="evidence-lineage-title">
                  {entry.source.source_type} | {entry.source.source_table}
                </div>
                <div className="history-meta">
                  {entry.source.evidence_id} | {entry.source.created_at}
                </div>
              </div>
              <div className="evidence-lineage-snippet">
                {entry.source.snippet ? truncate(entry.source.snippet, 140) : "No snippet available."}
              </div>
              <div className="evidence-lineage-links">
                {entry.links.length === 0 ? (
                  <span className="history-meta">No links recorded.</span>
                ) : (
                  entry.links.slice(0, 4).map((link) => (
                    <div key={link.link_id} className="evidence-link-row">
                      <span className="evidence-link-target">
                        {link.target_type}:{link.target_id}
                      </span>
                      <span className="history-meta">{link.relation ?? "supports"}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );

  const renderSnapshots = () => (
    <section className="cockpit-panel snapshot-panel">
      <div className="panel-header">
        <div>
          <h2>Snapshots</h2>
          <p>Recent subject, gate, introspection, audit, and qualia signals.</p>
        </div>
      </div>
      <div className="snapshot-grid">
        <div className="snapshot-block">
          <h3>Subject Snapshots</h3>
          {filteredSubjectSnapshots.slice(0, 3).map((snap) => {
            let ignition = "unknown";
            try {
              const parsed = JSON.parse(snap.subject_state_json) as any;
              ignition = parsed?.state?.workspace?.ignition?.active ? "ignited" : "idle";
            } catch {
              ignition = "parse_error";
            }
            return (
              <div key={snap.snapshot_hash} className="snapshot-row">
                <span>{shortHash(snap.snapshot_hash)}</span>
                <span>{ignition}</span>
                <span>{snap.timestamp}</span>
              </div>
            );
          })}
          {filteredSubjectSnapshots.length === 0 && <div className="snapshot-empty">No snapshots</div>}
        </div>
        <div className="snapshot-block">
          <h3>Attention Schema</h3>
          {latestAttentionSchema ? (
            <>
              <div className="snapshot-row">
                <span>Capacity</span>
                <span>{Number(latestAttentionSchema.capacity_usage ?? 0).toFixed(2)}</span>
              </div>
              <div className="snapshot-row">
                <span>Stability</span>
                <span>{Number(latestAttentionSchema.stability ?? 0).toFixed(2)}</span>
              </div>
              <div className="snapshot-row">
                <span>Policy</span>
                <span>{latestAttentionSchema.selection_policy ?? "none"}</span>
              </div>
              <div className="snapshot-row">
                <span>Suppressed</span>
                <span>{(latestAttentionSchema.suppressed_items ?? []).length}</span>
              </div>
            </>
          ) : (
            <div className="snapshot-empty">No attention schema</div>
          )}
        </div>
        <div className="snapshot-block">
          <h3>Workspace Contributors</h3>
          {latestWorkspaceContributors ? (
            <>
              <div className="snapshot-row">
                <span>Missing</span>
                <span>{(latestWorkspaceContributors.missing ?? []).length
                  ? (latestWorkspaceContributors.missing ?? []).join(", ")
                  : "None"}</span>
              </div>
              <div className="snapshot-row">
                <span>Memory Writes</span>
                <span>{latestWorkspaceContributors.contributors?.memory?.recent_writes ?? "--"}</span>
              </div>
              <div className="snapshot-row">
                <span>Residuals</span>
                <span>{latestWorkspaceContributors.contributors?.prediction?.residual_count ?? "--"}</span>
              </div>
              <div className="snapshot-row">
                <span>Updated</span>
                <span>{latestWorkspaceContributors.contributors?.updated_at ?? "--"}</span>
              </div>
            </>
          ) : (
            <div className="snapshot-empty">No workspace snapshot</div>
          )}
        </div>
        <div className="snapshot-block">
          <h3>Gate Decisions</h3>
          {filteredGateDecisions.slice(0, 3).map((gate) => (
            <div key={gate.decision_id} className="snapshot-row">
              <span>{shortHash(gate.decision_id)}</span>
              <span>{gate.decision}</span>
              <span>{gate.created_at}</span>
            </div>
          ))}
          {filteredGateDecisions.length === 0 && <div className="snapshot-empty">No gate decisions</div>}
        </div>
        <div className="snapshot-block">
          <h3>Introspection</h3>
          {introspectionEntries.slice(0, 3).map((entry) => (
            <div key={entry.entry_id} className="snapshot-row">
              <span>{shortHash(entry.entry_id)}</span>
              <span>{truncate(entry.narrative, 48)}</span>
              <span>{entry.created_at}</span>
            </div>
          ))}
          {introspectionEntries.length === 0 && <div className="snapshot-empty">No introspection entries</div>}
        </div>
        <div className="snapshot-block">
          <h3>Audit Log</h3>
          {auditLog.slice(0, 3).map((audit) => (
            <div key={audit.audit_id} className="snapshot-row">
              <span>{shortHash(audit.audit_id)}</span>
              <span>{audit.recommended_action}</span>
              <span>{audit.discrepancy_score.toFixed(2)}</span>
            </div>
          ))}
          {auditLog.length === 0 && <div className="snapshot-empty">No audit entries</div>}
        </div>
        <div className="snapshot-block">
          <h3>Error Events</h3>
          {errorEvents.slice(0, 3).map((err) => (
            <div key={err.error_event_id} className="snapshot-row">
              <span>{shortHash(err.error_event_id)}</span>
              <span>{err.classification}</span>
              <span>{err.status}</span>
            </div>
          ))}
          {errorEvents.length === 0 && <div className="snapshot-empty">No error events</div>}
        </div>
        <div className="snapshot-block">
          <h3>Qualia Labels</h3>
          {qualiaLabels.slice(0, 3).map((label) => (
            <div key={label.label_id} className="snapshot-row">
              <span>{shortHash(label.label_id)}</span>
              <span>{label.tag}</span>
              <span>{label.intensity.toFixed(2)}</span>
            </div>
          ))}
          {qualiaLabels.length === 0 && <div className="snapshot-empty">No qualia labels</div>}
        </div>
      </div>
    </section>
  );

  const renderSuppressionSummary = () => (
    <section className="cockpit-panel suppression-panel">
      <div className="panel-header">
        <div>
          <h2>Suppression Summary</h2>
          <p>Top suppression reasons across recent monologue activity.</p>
        </div>
      </div>
      {suppressionSummary.length === 0 ? (
        <div className="panel-empty">No suppression summary yet.</div>
      ) : (
        <div className="suppression-list">
          {suppressionSummary.map(([reason, count]) => (
            <div key={reason} className="suppression-row">
              <span>{reason}</span>
              <span>{count}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );

  const formatPayloadValue = (value: unknown) => {
    if (value === null || value === undefined) return "--";
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      return String(value);
    }
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  };

  const formatPreviewValue = (value: unknown) => {
    if (typeof value === "string") return truncate(value, 180);
    if (Array.isArray(value)) {
      return value.length === 0 ? "0 items" : `${value.length} items`;
    }
    return formatPayloadValue(value);
  };

  const buildRows = (rows: Array<[string, unknown]>) =>
    rows
      .filter(([, value]) => value !== undefined && value !== null && value !== "")
      .map(([key, value]) => ({ key, value }));

  const renderRows = (rows: { key: string; value: unknown }[]) => (
    <div className="trace-payload-grid trace-payload-typed">
      {rows.map((row) => (
        <div key={row.key} className="trace-payload-row">
          <span className="trace-payload-key">{row.key}</span>
          <span className="trace-payload-value">{formatPreviewValue(row.value)}</span>
        </div>
      ))}
    </div>
  );

  const renderTypedPayload = (entry: TraceEntry) => {
    if (typeof entry.payload !== "object" || entry.payload === null) return null;
    const payload = entry.payload as Record<string, any>;
    switch (entry.event) {
      case "decision_report": {
        const anchorHits = Array.isArray(payload.anchor_hits) ? payload.anchor_hits.length : payload.anchor_hits;
        const rows = buildRows([
          ["Decision", payload.decision],
          ["Selected action", payload.selected_action],
          ["Selected kind", payload.selected_kind],
          ["Gate decision", payload.gate_decision],
          ["Rationale", typeof payload.rationale === "string" ? truncate(payload.rationale, 180) : payload.rationale],
          ["Anchor hits", anchorHits],
        ]);
        return rows.length > 0 ? renderRows(rows) : null;
      }
      case "gate_decision": {
        const reasonList = Array.isArray(payload.gate_reasons)
          ? payload.gate_reasons.slice(0, 3).join(", ")
          : payload.gate_reasons;
        const rows = buildRows([
          ["Enforced decision", payload.enforced_decision ?? payload.decision],
          ["Soft decision", payload.soft_decision],
          ["Verify rate", typeof payload.verify_rate === "number" ? `${Math.round(payload.verify_rate * 100)}%` : payload.verify_rate],
          ["Gate reasons", reasonList],
        ]);
        return rows.length > 0 ? renderRows(rows) : null;
      }
      case "gate_decision_inputs": {
        const reasonList = Array.isArray(payload.gate_reasons)
          ? payload.gate_reasons.slice(0, 3).join(", ")
          : payload.gate_reasons;
        const signalList = payload.signals && typeof payload.signals === "object" && !Array.isArray(payload.signals)
          ? Object.keys(payload.signals).slice(0, 4).join(", ")
          : Array.isArray(payload.signals)
            ? payload.signals.slice(0, 4).join(", ")
            : payload.signals;
        const rows = buildRows([
          ["Enforced decision", payload.enforced_decision],
          ["Soft decision", payload.soft_decision],
          ["Legacy decision", payload.legacy_decision],
          ["Gate reasons", reasonList],
          ["Signals", signalList],
        ]);
        return rows.length > 0 ? renderRows(rows) : null;
      }
      case "tool_dispatch": {
        const rows = buildRows([
          ["Tool", payload.tool_name],
          ["Action ID", payload.action_id],
          ["Status", payload.status],
          ["Duration ms", payload.duration_ms],
        ]);
        return rows.length > 0 ? renderRows(rows) : null;
      }
      case "tool_dispatch_failed": {
        const rows = buildRows([
          ["Tool", payload.tool_name],
          ["Action ID", payload.action_id],
          ["Error", payload.error],
        ]);
        return rows.length > 0 ? renderRows(rows) : null;
      }
      case "memory_pass_result": {
        const factsCount = Array.isArray(payload.facts) ? payload.facts.length : payload.facts;
        const relationsCount = Array.isArray(payload.relations) ? payload.relations.length : payload.relations;
        const rows = buildRows([
          ["Write count", payload.write_count],
          ["Facts", factsCount],
          ["Relations", relationsCount],
          ["Scope", payload.scope],
        ]);
        return rows.length > 0 ? renderRows(rows) : null;
      }
      case "memory_pass_invalid_output": {
        const rows = buildRows([
          ["Reason", payload.reason],
          ["Model", payload.model],
          ["Request label", payload.request_label],
          ["Raw snippet", payload.raw_snippet ? truncate(String(payload.raw_snippet), 160) : payload.raw_snippet],
        ]);
        return rows.length > 0 ? renderRows(rows) : null;
      }
      case "memory_write_blocked": {
        const rows = buildRows([
          ["Reason", payload.reason],
          ["Candidate kind", payload.candidate_kind],
          ["Candidate id", payload.candidate_id],
        ]);
        return rows.length > 0 ? renderRows(rows) : null;
      }
      case "response_reasoning_json_invalid": {
        const rows = buildRows([
          ["Request label", payload.request_label],
          ["Reason", payload.reason],
        ]);
        return rows.length > 0 ? renderRows(rows) : null;
      }
      case "monologue_parse_failed": {
        const rows = buildRows([
          ["Reason", payload.reason],
          ["Cooldown until", payload.cooldown_until],
        ]);
        return rows.length > 0 ? renderRows(rows) : null;
      }
      default:
        return null;
    }
  };

  const renderPayloadPreview = (entry: TraceEntry) => {
    const typed = renderTypedPayload(entry);
    if (typed) return typed;
    if (typeof entry.payload !== "object" || entry.payload === null) {
      return <pre className="trace-payload-raw">{formatPayloadValue(entry.payload)}</pre>;
    }
    const payload = entry.payload as Record<string, unknown>;
    const keys = EVENT_DETAIL_KEYS[entry.event] ?? DEFAULT_DETAIL_KEYS;
    const rows = keys
      .filter((key) => payload[key] !== undefined)
      .map((key) => ({ key, value: payload[key] }));
    if (rows.length === 0) {
      return <pre className="trace-payload-raw">{JSON.stringify(payload, null, 2)}</pre>;
    }
    return (
      <div className="trace-payload-grid">
        {rows.map((row) => (
          <div key={row.key} className="trace-payload-row">
            <span className="trace-payload-key">{row.key}</span>
            <span className="trace-payload-value">{formatPayloadValue(row.value)}</span>
          </div>
        ))}
      </div>
    );
  };

  const renderLogsPanel = () => (
    <section className="cockpit-panel log-panel">
      <div className="panel-header">
        <div>
          <h2>System Logs</h2>
          <p>Live operational trace across kernel, memory, monologue, tools, and messages.</p>
        </div>
      </div>

      {error && <div className="trace-error-banner">{error}</div>}
      {healthError && <div className="trace-error-banner">{healthError}</div>}

      <div className="trace-meta-bar">
        <div className="trace-meta-item">
          Focus run:
          <strong>{runFilter ? ` ${activeRunId.slice(0, 8)}...` : " All"}</strong>
        </div>
        <div className="trace-meta-item">Entries: {filteredEntries.length}</div>
        {activeAlertId && (
          <div className="trace-meta-item trace-alert-filter">
            Alert filter: {activeAlertId}
            <button className="btn btn-secondary btn-compact" onClick={() => setActiveAlertId(null)}>
              Clear
            </button>
          </div>
        )}
      </div>

      <div className="trace-controls">
        <div className="trace-filters">
          {categories.map((category) => (
            <button
              key={category}
              className={`trace-chip${activeCategory === category ? " active" : ""}`}
              onClick={() => setActiveCategory(category)}
            >
              {category}
            </button>
          ))}
        </div>
        <div className="trace-filters">
          {levels.map((level) => (
            <button
              key={level}
              className={`trace-chip${activeLevel === level ? " active" : ""}`}
              onClick={() => setActiveLevel(level)}
            >
              {level}
            </button>
          ))}
        </div>
        <div className="trace-search">
          <input
            className="input trace-search-input"
            placeholder="Search events, categories, payloads..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <button className="btn btn-secondary btn-compact" onClick={clearFilters}>
          Clear Filters
        </button>
      </div>

      <div className="trace-lenses">
        <div className="lens-save">
          <input
            className="input lens-input"
            placeholder="Save lens name..."
            value={lensDraft}
            onChange={(e) => setLensDraft(e.target.value)}
          />
          <button className="btn btn-secondary btn-compact" onClick={handleSaveLens}>
            Save Lens
          </button>
        </div>
        {savedLenses.length > 0 && (
          <div className="lens-list">
            {savedLenses.map((lens) => (
              <div key={lens.id} className="lens-chip">
                <button
                  className="lens-apply"
                  onClick={() => handleApplyLens(lens)}
                  title={`Run ${lens.runId || "all"} | ${lens.category}/${lens.level}`}
                >
                  {lens.label}
                </button>
                <button
                  className="lens-remove"
                  onClick={() => handleDeleteLens(lens.id)}
                  aria-label={`Remove ${lens.label}`}
                >
                  x
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="trace-list">
        {groupedEntries.length === 0 ? (
          <div className="trace-empty">No system logs recorded yet.</div>
        ) : (
          groupedEntries.map(([runKey, entries]) => {
            const collapsed = collapsedRuns.has(runKey);
            const label = runKey === "no-run" ? "Run: none" : `Run: ${runKey.slice(0, 8)}...`;
            return (
              <div key={runKey} className="trace-group">
                <button className="trace-group-header" onClick={() => toggleRun(runKey)}>
                  <span>{label}</span>
                  <span className="trace-group-count">{entries.length} events</span>
                </button>
                {!collapsed && (
                  <div className="trace-group-entries">
                    {entries.map((entry) => (
                      <div key={entry.id} className={`trace-entry trace-level-${entry.severity ?? entry.level}`}>
                        <div className="trace-entry-meta">
                          <span className="trace-entry-level">{entry.severity ?? entry.level}</span>
                          <span className="trace-entry-category">{entry.category}</span>
                          <span className="trace-entry-source">{entry.source}</span>
                          <span className="trace-entry-event">{formatEventLabel(entry.event)}</span>
                          <span className="trace-entry-time">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                        </div>
                        <div className="trace-entry-payload">
                          {renderPayloadPreview(entry)}
                          <details className="trace-entry-raw">
                            <summary>Raw JSON</summary>
                            <pre>{typeof entry.payload === "string" ? entry.payload : JSON.stringify(entry.payload, null, 2)}</pre>
                          </details>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </section>
  );

  const renderRawPanel = () => (
    <section className="cockpit-panel raw-panel">
      <div className="panel-header">
        <div>
          <h2>Raw JSON</h2>
          <p>Health and subsystem state payloads for deep inspection.</p>
        </div>
      </div>
      <details className="raw-drawer">
        <summary>Health Snapshot</summary>
        <pre>{rawHealth}</pre>
      </details>
      <details className="raw-drawer">
        <summary>Subsystem States</summary>
        <pre>{rawSubsystems}</pre>
      </details>
    </section>
  );

  return (
    <div className="cockpit-view">
      <div className="cockpit-header">
        <div className="cockpit-title-block">
          <div>
            <h1>System Cockpit</h1>
            <p className="trace-subtitle">
              Control plane + observability dashboard. Last refresh: {cockpitLastUpdated || "--"}
            </p>
          </div>
        </div>
        <div className="cockpit-actions">
          <div className="cockpit-write-toggle">
            <span className="cockpit-write-label">Cockpit Write</span>
            <button
              className={`cockpit-write-switch${cockpitWriteEnabled ? " active" : ""}`}
              onClick={() => onToggleCockpitWrite(!cockpitWriteEnabled)}
              type="button"
              aria-pressed={cockpitWriteEnabled}
              title={cockpitWriteEnabled ? "Write mode enabled" : "Read-only mode"}
            >
              <span className="cockpit-write-track" />
              <span className="cockpit-write-thumb" />
            </button>
            <span className="cockpit-write-status">
              {cockpitWriteEnabled ? "Enabled" : "Read-only"}
            </span>
          </div>
          <div className="trace-actions">
            <button className="btn btn-secondary" onClick={onRefresh}>Refresh</button>
            <button className="btn btn-secondary" onClick={onClear}>Clear Logs</button>
          </div>
        </div>
      </div>

      <div className="cockpit-focus">
        <div className="focus-group">
          <span className="focus-label">Focus Run</span>
          <select
            className="input focus-select"
            value={activeRunId}
            onChange={(e) => handleRunChange(e.target.value)}
          >
            <option value="all">All runs</option>
            {runOptions.map((opt) => (
              <option key={opt.id} value={opt.id}>
                {opt.label} ({opt.count})
              </option>
            ))}
          </select>
          {runSelectionLocked && (
            <button className="btn btn-secondary btn-compact" onClick={releaseRunLock}>
              Auto
            </button>
          )}
        </div>
        <div className="focus-group">
          <span className="focus-label">Run Events</span>
          <span className="focus-value">{runStoryEvents.length}</span>
        </div>
        <div className="focus-group">
          <span className="focus-label">Alerts</span>
          <span className="focus-value">{activeAlerts.length}</span>
        </div>
      </div>

      {healthBanner && !healthBannerDismissed && (
        <div className={`health-banner cockpit-health-banner ${healthBannerCollapsed ? "collapsed" : ""}`}>
          <div className="health-banner-title">
            <span>
              Health degraded for 2 consecutive days ({healthBanner.previousDay}, {healthBanner.latestDay}).
            </span>
            <div className="health-banner-actions">
              <button
                className="health-banner-toggle"
                onClick={() => setHealthBannerCollapsed((prev) => !prev)}
                aria-label={healthBannerCollapsed ? "Expand health banner" : "Collapse health banner"}
              >
                {healthBannerCollapsed ? "Show" : "Hide"}
              </button>
              <button
                className="health-banner-dismiss"
                onClick={() => setHealthBannerDismissed(true)}
                aria-label="Dismiss health banner"
              >
                Dismiss
              </button>
            </div>
          </div>
          {!healthBannerCollapsed && (
            <div className="health-banner-body">
              Failures: {healthBanner.reasons.length > 0 ? healthBanner.reasons.join(", ") : "Multiple criteria failed."}
            </div>
          )}
        </div>
      )}

      <div className="cockpit-toolbar">
        <div className="cockpit-tabs" role="tablist" aria-label="Cockpit tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              role="tab"
              className={`cockpit-tab${activeTab === tab.id ? " active" : ""}`}
              aria-selected={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
            >
              <span>{tab.label}</span>
              {tab.badge > 0 && (
                <span className="cockpit-tab-badge">{formatBadge(tab.badge)}</span>
              )}
            </button>
          ))}
        </div>
        <div className="cockpit-tab-status">
          <span className="cockpit-tab-status-label">Phase</span>
          <span className="cockpit-tab-status-value">{statusLabel}</span>
        </div>
      </div>

      <div className="cockpit-rail">
        <div className="cockpit-status-rail">
          <div className="cockpit-rail-card">
            <div className="rail-label" title="Current module stage in the active run pipeline.">System Phase</div>
            <div className="rail-value">{statusLabel}</div>
            <div className="rail-meta">{statusDetail}</div>
          </div>
          <div className="cockpit-rail-card">
            <div className="rail-label" title="Latest gate enforcement decision plus verify rate.">Gate Posture</div>
            <div className="rail-value">{gateDecision}</div>
            <div className="rail-meta">Verify {Math.round(gateVerifyRate * 100)}%</div>
          </div>
          <div className="cockpit-rail-card">
            <div className="rail-label" title="Highest priority issues flagged by health metrics.">Active Alerts</div>
            {activeAlerts.length === 0 ? (
              <div className="rail-value">None</div>
            ) : (
              <div className="rail-alerts">
                {activeAlerts.map((item) => (
                  <span key={item.id} className={`rail-pill ${item.tone}`}>
                    {item.label} {item.value}
                  </span>
                ))}
              </div>
            )}
            <div className="rail-meta">Pending {pendingCount} - Errors {errorOpen}</div>
          </div>
        </div>

        <div className="cockpit-recent-rail">
          <div className="rail-label" title="Most recent system log events for this run.">Recent Events</div>
          <div className="recent-events">
            {recentEvents.length === 0 ? (
              <div className="rail-meta">No recent events.</div>
            ) : (
              recentEvents.map((entry) => {
                const payload = entry.payload as any;
                const eventName = payload && typeof payload === "object" && "event" in payload
                  ? String(payload.event)
                  : entry.category;
                return (
                  <div key={entry.id} className={`recent-event-row recent-${entry.level}`}>
                    <span className="recent-event-title">{formatEventLabel(eventName)}</span>
                    <span className="recent-event-time">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>

      <div className="cockpit-body">
        {activeTab === "overview" && (
          <section className="cockpit-overview">
            <div className="cockpit-overview-toolbar">
              <div className="cockpit-section-title">Overview</div>
              <div className="cockpit-overview-actions">
                <button
                  className={`btn btn-secondary cockpit-expand${overviewExpanded ? " active" : ""}`}
                  onClick={() => setOverviewExpanded((prev) => !prev)}
                >
                  {overviewExpanded ? "Collapse" : "Expand All"}
                </button>
              </div>
            </div>

            <div className="cockpit-overview-stack">
              {renderSystemStatus()}
              {renderActiveAlerts()}
              {renderDeltaPanel()}
            </div>

            <div className="cockpit-overview-grid">
              {renderRunStory()}
              {renderPlayback()}
            </div>

            {overviewExpanded && (
              <div className="cockpit-overview-expanded">
                <div className="cockpit-overview-column">
                  {renderActivityFlow()}
                  {renderSuppressionSummary()}
                  {renderSkipSummary()}
                  {renderFeedbackLoop()}
                </div>
                <div className="cockpit-overview-column">
                  <SystemControlPanel
                    subsystemStates={subsystemStates}
                    allowWrites={allowControlWrites}
                    onRefresh={onRefresh}
                    error={controlError}
                  />
                  <SystemControlHistory events={systemControlEvents} />
                  <section className="cockpit-panel weight-panel">
                    <div className="panel-header">
                      <div>
                        <h2>Constraint Weights</h2>
                        <p>Adjust candidate ranking priorities. Changes are logged.</p>
                      </div>
                      <button
                        className="btn btn-secondary"
                        onClick={handleResetWeights}
                        disabled={!allowEdits}
                      >
                        Reset
                      </button>
                    </div>
                    {!settings && <div className="panel-empty">Settings not loaded.</div>}
                    {settings && (
                      <div className="weight-grid">
                        {WEIGHT_DEFS.map((item) => {
                          const value = Number((settings as any)[item.key] ?? 0.5);
                          return (
                            <div key={item.key} className="weight-row">
                              <div className="weight-label">{item.label}</div>
                              <input
                                type="range"
                                min={0}
                                max={1}
                                step={0.05}
                                value={value}
                                onChange={(e) => handleWeightChange(item.key, Number(e.target.value))}
                                disabled={!allowEdits}
                              />
                              <div className="weight-value">{value.toFixed(2)}</div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </section>
                  {renderSelfReport()}
                  <SystemHealthPanel snapshot={systemHealthSnapshot} history={systemHealthHistory} />
                  <SystemHealthTimeline history={systemHealthHistory} />
                  {renderEventTimeline()}
                  {renderSnapshots()}
                </div>
              </div>
            )}
          </section>
        )}

        {activeTab === "controls" && (
          <section className="cockpit-tab-content">
            <div className="cockpit-section-title">Control Plane</div>
            <div className="cockpit-tab-grid">
              <div className="cockpit-stack">
                <SystemControlPanel
                  subsystemStates={subsystemStates}
                  allowWrites={allowControlWrites}
                  onRefresh={onRefresh}
                  error={controlError}
                />
                <SystemControlHistory events={systemControlEvents} />
              </div>
              <div className="cockpit-stack">
                <section className="cockpit-panel weight-panel">
                  <div className="panel-header">
                    <div>
                      <h2>Constraint Weights</h2>
                      <p>Adjust candidate ranking priorities. Changes are logged.</p>
                    </div>
                    <button
                      className="btn btn-secondary"
                      onClick={handleResetWeights}
                      disabled={!allowEdits}
                    >
                      Reset
                    </button>
                  </div>
                  {!settings && <div className="panel-empty">Settings not loaded.</div>}
                  {settings && (
                    <div className="weight-grid">
                      {WEIGHT_DEFS.map((item) => {
                        const value = Number((settings as any)[item.key] ?? 0.5);
                        return (
                          <div key={item.key} className="weight-row">
                            <div className="weight-label">{item.label}</div>
                            <input
                              type="range"
                              min={0}
                              max={1}
                              step={0.05}
                              value={value}
                              onChange={(e) => handleWeightChange(item.key, Number(e.target.value))}
                              disabled={!allowEdits}
                            />
                            <div className="weight-value">{value.toFixed(2)}</div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </section>
              </div>
            </div>
          </section>
        )}

        {activeTab === "signals" && (
          <section className="cockpit-tab-content">
            <div className="cockpit-section-title">Signals</div>
            <div className="cockpit-tab-grid">
              <div className="cockpit-stack">
                {renderActivityFlow()}
                {renderSuppressionSummary()}
              </div>
              <div className="cockpit-stack">
                {renderFeedbackLoop()}
              </div>
            </div>
          </section>
        )}

        {activeTab === "diagnostics" && (
          <section className="cockpit-tab-content">
            <div className="cockpit-overview-toolbar">
              <div className="cockpit-section-title">Diagnostics</div>
              <div className="cockpit-overview-actions">
                <button
                  className={`btn btn-secondary cockpit-expand${diagnosticsExpanded ? " active" : ""}`}
                  onClick={() => setDiagnosticsExpanded((prev) => !prev)}
                >
                  {diagnosticsExpanded ? "Collapse" : "Expand"}
                </button>
              </div>
            </div>
            <div className={`cockpit-diagnostics ${diagnosticsExpanded ? "open" : ""}`}>
              <div className="cockpit-stack">
                <SystemHealthPanel snapshot={systemHealthSnapshot} history={systemHealthHistory} />
                <SystemHealthTimeline history={systemHealthHistory} />
              </div>
              <div className="cockpit-stack">
                {renderSelfReport()}
                {renderDecisionReports()}
                <OutcomePanel
                  messages={filteredMessages}
                  systemLogs={filteredSystemLogs}
                  allowWrites={allowControlWrites}
                />
                {renderEvidenceLineage()}
                {renderEventTimeline()}
                {renderSnapshots()}
              </div>
            </div>
          </section>
        )}

        {activeTab === "logs" && (
          <section className="cockpit-tab-content">
            <div className="cockpit-section-title">Logs</div>
            <div className="cockpit-stack">
              {renderLogsPanel()}
              <details className="deep-inspect">
                <summary>Deep Inspect</summary>
                {renderRawPanel()}
              </details>
            </div>
          </section>
        )}
      </div>
    </div>
  );
};
