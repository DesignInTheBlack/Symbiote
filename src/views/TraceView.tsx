
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
import { SystemStatePanel } from "../components/SystemStatePanel";
import { HealthGatePanel } from "../components/HealthGatePanel";
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
};

type CockpitTabId = "overview" | "controls" | "signals" | "diagnostics" | "logs";

type PinnedPanels = {
  gate: boolean;
  organism: boolean;
};

const formatEventLabel = (value: string) =>
  value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (m) => m.toUpperCase());

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
  const [collapsedRuns, setCollapsedRuns] = useState<Set<string>>(new Set());
  const [intentDraft, setIntentDraft] = useState<string>(intentSummary?.summary ?? "");
  const [intentConfirmed, setIntentConfirmed] = useState<boolean>(intentSummary?.confirmed ?? false);
  const [intentSaving, setIntentSaving] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<CockpitTabId>("overview");
  const [overviewExpanded, setOverviewExpanded] = useState<boolean>(false);
  const [diagnosticsExpanded, setDiagnosticsExpanded] = useState<boolean>(true);
  const [pinnedPanels, setPinnedPanels] = useState<PinnedPanels>({ gate: false, organism: false });
  const [healthBannerCollapsed, setHealthBannerCollapsed] = useState(false);
  const [healthBannerDismissed, setHealthBannerDismissed] = useState(false);
  const allowEdits = cockpitWriteEnabled && Boolean(onUpdateSettings);
  const decisionReports = useMemo(
    () => systemLogs.filter((entry) => (entry.payload as any)?.event === "decision_report").slice(0, 6),
    [systemLogs]
  );

  useEffect(() => {
    setIntentDraft(intentSummary?.summary ?? "");
    setIntentConfirmed(intentSummary?.confirmed ?? false);
  }, [intentSummary]);

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

  const combinedEntries = useMemo(() => {
    const entries = [
      ...buildSystemEntries(systemLogs),
      ...buildMonologueEntries(monologueEntries),
      ...buildMessageEntries(messages),
    ];
    return entries.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }, [systemLogs, monologueEntries, messages]);

  const feedbackBundlePayload = useMemo(() => {
    for (const entry of systemLogs) {
      const payload = entry.payload as any;
      if (payload && typeof payload === "object" && payload.event === "feedback_bundle_built") {
        return payload.bundle ?? payload.feedback_bundle ?? null;
      }
    }
    return null;
  }, [systemLogs]);

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
    for (const entry of systemLogs) {
      const payload = entry.payload as any;
      const event = payload?.event;
      if (event && tracked.has(event)) {
        counts[event] = (counts[event] || 0) + 1;
      }
    }
    return counts;
  }, [systemLogs]);

  const latestAttentionSchema = useMemo(() => {
    const latest = subjectSnapshots[0];
    if (!latest) return null;
    try {
      const parsed = JSON.parse(latest.subject_state_json) as any;
      return parsed?.state?.attention_schema ?? null;
    } catch {
      return null;
    }
  }, [subjectSnapshots]);

  const latestWorkspaceContributors = useMemo(() => {
    for (const entry of systemLogs) {
      const payload = entry.payload as any;
      if (payload && typeof payload === "object" && payload.event === "workspace_snapshot") {
        return payload;
      }
    }
    return null;
  }, [systemLogs]);

  const selfReportMessages = useMemo(
    () => messages.filter((msg) => msg.role === "assistant" && (msg.metadata as any)?.self_report),
    [messages]
  );

  const categories = useMemo(() => {
    const unique = new Set<string>(combinedEntries.map((entry) => entry.category));
    return ["all", ...Array.from(unique).sort()];
  }, [combinedEntries]);

  const levels = ["all", "info", "warn", "error"];

  const filteredEntries = useMemo(() => {
    const query = search.trim().toLowerCase();
    return combinedEntries.filter((entry) => {
      if (activeCategory !== "all" && entry.category !== activeCategory) return false;
      if (activeLevel !== "all" && entry.level !== activeLevel) return false;
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
  }, [combinedEntries, activeCategory, activeLevel, search]);

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

  const subsystemStates = useMemo(() => {
    const raw = systemHealthSnapshot?.subsystem_states;
    if (Array.isArray(raw)) {
      return raw as SubsystemState[];
    }
    return [] as SubsystemState[];
  }, [systemHealthSnapshot]);
  const suppressionSummary = useMemo(() => {
    const counts = new Map<string, number>();
    for (const entry of systemLogs) {
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
  }, [systemLogs]);

  const keyEvents = useMemo(() => {
    return [...systemLogs]
      .filter((entry) => entry.level === "warn" || entry.level === "error")
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, 8);
  }, [systemLogs]);

  const recentEvents = useMemo(() => {
    return [...systemLogs]
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, 5);
  }, [systemLogs]);

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
      },
      {
        label: "Gate Decisions",
        value: `${gate.total ?? 0}`,
        meta: `Verify ${(Number(gate.verify_rate ?? 0) * 100).toFixed(0)}%`,
      },
      {
        label: "Tool Dispatch",
        value: `${tools.dispatches ?? 0}`,
        meta: `${tools.failures ?? 0} failures`,
      },
      {
        label: "Memory",
        value: `${memory.memory_pass_count ?? 0} passes`,
        meta: `${memory.write_count ?? 0} writes`,
      },
      {
        label: "Summaries",
        value: `${summaryTotal} updates`,
        meta: summaryFailures ? `${summaryFailures} failures` : "No failures",
      },
      {
        label: "Errors",
        value: `${errors.total ?? 0}`,
        meta: `${errors.open ?? 0} open`,
      },
      {
        label: "Pending Prompts",
        value: `${pending.count ?? 0}`,
        meta: pendingAgeMin !== null ? `Oldest ${pendingAgeMin}m` : "--",
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

  const latestGateInput = gateInputEvents[0];
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

  const alertItems = useMemo(() => {
    const items: { label: string; value: string; tone: "warn" | "alert" }[] = [];
    if (errorOpen > 0) items.push({ label: "Errors", value: `${errorOpen}`, tone: "alert" });
    if (pendingCount > 0) items.push({ label: "Pending", value: `${pendingCount}`, tone: "warn" });
    if (gateVerifyRate > 0.4) items.push({ label: "Gate Verify", value: `${Math.round(gateVerifyRate * 100)}%`, tone: "warn" });
    if (organismStress > 0.7) items.push({ label: "Stress", value: organismStress.toFixed(2), tone: "warn" });
    if (controllerConfidence < 0.4) items.push({ label: "Confidence", value: `${Math.round(controllerConfidence * 100)}%`, tone: "warn" });
    return items.slice(0, 4);
  }, [errorOpen, pendingCount, gateVerifyRate, organismStress, controllerConfidence]);

  const tabs = useMemo(() => ([
    { id: "overview" as const, label: "Overview", badge: alertItems.length },
    { id: "controls" as const, label: "Controls", badge: 0 },
    { id: "signals" as const, label: "Signals", badge: pendingCount },
    { id: "diagnostics" as const, label: "Diagnostics", badge: keyEvents.length },
    { id: "logs" as const, label: "Logs", badge: errorOpen },
  ]), [alertItems.length, pendingCount, keyEvents.length, errorOpen]);

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

  const togglePin = (key: keyof PinnedPanels) => {
    setPinnedPanels((prev) => ({ ...prev, [key]: !prev[key] }));
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
            <div className="flow-card-label">{stat.label}</div>
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
      {evidenceLineageError ? (
        <div className="panel-empty">Failed to load evidence lineage.</div>
      ) : evidenceLineage.length === 0 ? (
        <div className="panel-empty">No evidence lineage entries yet.</div>
      ) : (
        <div className="evidence-lineage-list">
          {evidenceLineage.slice(0, 12).map((entry) => (
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
          {subjectSnapshots.slice(0, 3).map((snap) => {
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
          {subjectSnapshots.length === 0 && <div className="snapshot-empty">No snapshots</div>}
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
          {gateDecisions.slice(0, 3).map((gate) => (
            <div key={gate.decision_id} className="snapshot-row">
              <span>{shortHash(gate.decision_id)}</span>
              <span>{gate.decision}</span>
              <span>{gate.created_at}</span>
            </div>
          ))}
          {gateDecisions.length === 0 && <div className="snapshot-empty">No gate decisions</div>}
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
                      <div key={entry.id} className={`trace-entry trace-level-${entry.level}`}>
                        <div className="trace-entry-meta">
                          <span className="trace-entry-level">{entry.level}</span>
                          <span className="trace-entry-category">{entry.category}</span>
                          <span className="trace-entry-source">{entry.source}</span>
                          <span className="trace-entry-event">{formatEventLabel(entry.event)}</span>
                          <span className="trace-entry-time">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                        </div>
                        <details className="trace-entry-payload" open>
                          <summary>Payload</summary>
                          <pre>{typeof entry.payload === "string" ? entry.payload : JSON.stringify(entry.payload, null, 2)}</pre>
                        </details>
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
            <div className="rail-label">System Phase</div>
            <div className="rail-value">{statusLabel}</div>
            <div className="rail-meta">{statusDetail}</div>
          </div>
          <div className="cockpit-rail-card">
            <div className="rail-label">Gate Posture</div>
            <div className="rail-value">{gateDecision}</div>
            <div className="rail-meta">Verify {Math.round(gateVerifyRate * 100)}%</div>
          </div>
          <div className="cockpit-rail-card">
            <div className="rail-label">Active Alerts</div>
            {alertItems.length === 0 ? (
              <div className="rail-value">None</div>
            ) : (
              <div className="rail-alerts">
                {alertItems.map((item) => (
                  <span key={item.label} className={`rail-pill ${item.tone}`}>
                    {item.label} {item.value}
                  </span>
                ))}
              </div>
            )}
            <div className="rail-meta">Pending {pendingCount} - Errors {errorOpen}</div>
          </div>
        </div>

        <div className="cockpit-recent-rail">
          <div className="rail-label">Recent Events</div>
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
                <button
                  className={`cockpit-pin-toggle${pinnedPanels.gate ? " active" : ""}`}
                  onClick={() => togglePin("gate")}
                  type="button"
                >
                  Pin Gate
                </button>
                <button
                  className={`cockpit-pin-toggle${pinnedPanels.organism ? " active" : ""}`}
                  onClick={() => togglePin("organism")}
                  type="button"
                >
                  Pin Organism
                </button>
              </div>
            </div>

            <div className="cockpit-hero-grid">
              <div className="cockpit-panel cockpit-hero-panel">
                <SystemStatePanel
                  chatState="idle"
                  moduleStatus={moduleStatus}
                  healthSnapshot={systemHealthSnapshot}
                  healthHistory={systemHealthHistory}
                />
              </div>
              <HealthGatePanel
                snapshot={systemHealthSnapshot}
                history={systemHealthHistory}
                gateInputs={gateInputEvents}
              />
            </div>

            {(pinnedPanels.gate || pinnedPanels.organism) && (
              <div className="cockpit-pin-grid">
                {pinnedPanels.gate && (
                  <section className="cockpit-panel cockpit-pin-card">
                    <div className="panel-header">
                      <div>
                        <h2>Gate Snapshot</h2>
                        <p>Decision posture and verification trend.</p>
                      </div>
                    </div>
                    <div className="pin-metric">
                      <span>Decision</span>
                      <strong>{gateDecision}</strong>
                    </div>
                    <div className="pin-metric">
                      <span>Verify rate</span>
                      <strong>{Math.round(gateVerifyRate * 100)}%</strong>
                    </div>
                    <div className="pin-metric">
                      <span>Last gate</span>
                      <strong>{latestGateInput?.timestamp ? new Date(latestGateInput.timestamp).toLocaleTimeString() : "--"}</strong>
                    </div>
                  </section>
                )}
                {pinnedPanels.organism && (
                  <section className="cockpit-panel cockpit-pin-card">
                    <div className="panel-header">
                      <div>
                        <h2>Organism Snapshot</h2>
                        <p>Stress, fatigue, and alignment in focus.</p>
                      </div>
                    </div>
                    <div className="pin-metric">
                      <span>Stress</span>
                      <strong>{organismStress.toFixed(2)}</strong>
                    </div>
                    <div className="pin-metric">
                      <span>Confidence</span>
                      <strong>{Math.round(controllerConfidence * 100)}%</strong>
                    </div>
                    <div className="pin-metric">
                      <span>Pending</span>
                      <strong>{pendingCount}</strong>
                    </div>
                  </section>
                )}
              </div>
            )}

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
                  messages={messages}
                  systemLogs={systemLogs}
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
            <div className="cockpit-tab-grid">
              <div className="cockpit-stack">
                {renderLogsPanel()}
              </div>
              <div className="cockpit-stack">
                {renderRawPanel()}
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
};
