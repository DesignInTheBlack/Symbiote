import { useMemo } from "react";
import { ModuleStatus, SystemHealthSnapshot, SelfModel } from "../types/app";
import { SystemStateBlob, SystemPhase, AvatarMetrics } from "./SystemStateBlob";
import { formatStatusDetail, formatStatusLabel } from "../utils/statusStages";

interface SystemStatePanelProps {
  chatState: string;
  moduleStatus: ModuleStatus | null;
  memoryError?: { message: string; timestamp?: string } | null;
  rollingSummaryError?: string | null;
  pendingPromptCount?: number;
  healthSnapshot?: SystemHealthSnapshot | null;
  healthHistory?: SystemHealthSnapshot[] | null;
  selfModel?: SelfModel | null;
}

const clamp = (value: number, min = 0, max = 1) => Math.min(max, Math.max(min, value));
const truncate = (value: string, max = 40) => (value.length > max ? `${value.slice(0, max - 3)}...` : value);

const phaseFromStage = (stage?: string | null): SystemPhase | null => {
  if (!stage) return null;
  switch (stage) {
    case "memory_pass":
    case "rolling_summary":
    case "inner_summary":
    case "commit_cycle":
    case "thread_run":
    case "post_processing":
    case "finalize":
      return "consolidating";
    case "memory_retrieval":
    case "tool_call":
      return "retrieving";
    case "llm_wait":
    case "prompt_build":
    case "tool_select":
    case "ingest_input":
    case "arbitration":
    case "grounding":
    case "queued":
      return "thinking";
    case "streaming":
      return "responding";
    case "cancelled":
      return "stopping";
    case "error":
      return "error";
    case "idle":
      return "idle";
    default:
      return null;
  }
};

const baseHueForPhase: Record<SystemPhase, number> = {
  idle: 210,
  thinking: 260,
  responding: 160,
  consolidating: 45,
  retrieving: 120,
  awaiting: 30,
  error: 350,
  stopping: 0,
};

export const SystemStatePanel = ({
  chatState,
  moduleStatus,
  memoryError,
  rollingSummaryError,
  pendingPromptCount = 0,
  healthSnapshot,
  healthHistory = null,
  selfModel = null,
}: SystemStatePanelProps) => {
  const internalStateLine = useMemo(() => {
    const summary = selfModel?.internal_state_summary ?? null;
    if (!summary || typeof summary !== "object") return "";
    const labels = (summary as any).labels ?? {};
    const mappingVersion = (summary as any).mapping_version ?? selfModel?.internal_state_map_version ?? null;
    const mappingDegraded = Boolean((summary as any).mapping_degraded ?? false);
    const entries = Object.entries(labels) as Array<[string, string]>;
    if (entries.length === 0) return "";
    const parts = entries.slice(0, 4).map(([key, value]) => `${key}=${value}`);
    const prefix = mappingVersion
      ? `v${mappingVersion}${mappingDegraded ? " degraded" : ""}`
      : "state";
    return `${prefix} ${parts.join(" | ")}`;
  }, [selfModel]);
  const unifiedLine = useMemo(() => {
    const unified = selfModel?.unified_state ?? null;
    if (!unified || typeof unified !== "object") return "";
    const workspace = (unified as any).workspace ?? {};
    const focus = workspace.current_focus ?? workspace.currentFocus ?? "";
    const memory = (unified as any).memory ?? {};
    const memConf = Number(memory.avg_confidence ?? memory.avg_conf ?? 0);
    const controller = (unified as any).controller_state ?? {};
    const ctrlConf = Number(controller.confidence ?? 0);
    const parts: string[] = [];
    if (focus) {
      parts.push(`focus:${focus}`);
    }
    if (!Number.isNaN(memConf)) {
      parts.push(`mem ${(memConf * 100).toFixed(0)}%`);
    }
    if (!Number.isNaN(ctrlConf)) {
      parts.push(`ctrl ${(ctrlConf * 100).toFixed(0)}%`);
    }
    return parts.join(" | ");
  }, [selfModel]);
  const metrics = useMemo(() => {
    const snapshotMetrics = (healthSnapshot?.metrics ?? {}) as Record<string, any>;
    const avatar = snapshotMetrics.avatar ?? {};
    const controller = snapshotMetrics.controller ?? {};
    const organism = avatar.organism ?? snapshotMetrics.organism ?? {};
    const gate = snapshotMetrics.gate ?? {};
    const pending = snapshotMetrics.pending_prompts ?? {};
    const errors = snapshotMetrics.errors ?? {};
    const attentionSchema = snapshotMetrics.attention_schema ?? {};
    const workspaceContrib = snapshotMetrics.workspace_contributors ?? {};
    const selfClaims = snapshotMetrics.self_claims ?? {};
    const selfReflection = snapshotMetrics.self_reflection ?? {};
    const recommendations = snapshotMetrics.recommendations ?? {};
    const recommendationItems = Array.isArray(recommendations.items) ? recommendations.items : [];
    const recommendationEligible = recommendationItems.filter((item) => item?.status === "eligible").length;
    const recommendationTotal = recommendationItems.length;

    const phase = phaseFromStage(moduleStatus?.stage)
      ?? (avatar.processing_phase as SystemPhase)
      ?? "idle";

    const certainty = clamp(Number(avatar.certainty ?? controller.confidence ?? 0.5));
    const health = clamp(Number(avatar.health ?? 0.5));
    const memoryActivity = clamp(Number(avatar.memory_activity ?? 0));
    const stress = clamp(Number(organism.stress ?? 0));
    const fatigue = clamp(Number(organism.fatigue ?? 0));
    const alignment = clamp(Number(organism.social_alignment ?? 0.5));
    const gateActivity = clamp(Number(avatar.gate_activity ?? gate.verify_rate ?? 0));
    const pendingPrompts = Number(avatar.pending_prompts ?? pending.count ?? pendingPromptCount ?? 0);
    const uncertainty = clamp(Number(controller.uncertainty ?? 0.5));
    const errorOpen = Number(errors.open ?? 0);

    const tempShift = ((stress + fatigue) / 2 - alignment) * 30;
    const baseHue = baseHueForPhase[phase] ?? 210;
    const phaseHue = (baseHue + tempShift + 360) % 360;

    const history = (healthHistory && healthHistory.length > 0)
      ? healthHistory
      : (healthSnapshot ? [healthSnapshot] : []);
    const seriesSnapshots = [...history].slice(0, 24).reverse();

    const seriesOrFallback = seriesSnapshots.length > 0
      ? seriesSnapshots
      : (healthSnapshot ? [healthSnapshot] : []);

    const healthSeries = seriesOrFallback.map((snapshot) => {
      const metrics = (snapshot.metrics ?? {}) as Record<string, any>;
      const avatar = metrics.avatar ?? {};
      return clamp(Number(avatar.health ?? 0));
    });
    const memorySeries = seriesOrFallback.map((snapshot) => {
      const metrics = (snapshot.metrics ?? {}) as Record<string, any>;
      const avatar = metrics.avatar ?? {};
      return clamp(Number(avatar.memory_activity ?? 0));
    });
    const gateSeries = seriesOrFallback.map((snapshot) => {
      const metrics = (snapshot.metrics ?? {}) as Record<string, any>;
      const gate = metrics.gate ?? {};
      const avatar = metrics.avatar ?? {};
      return clamp(Number(avatar.gate_activity ?? gate.verify_rate ?? 0));
    });
    const stressSeries = seriesOrFallback.map((snapshot) => {
      const metrics = (snapshot.metrics ?? {}) as Record<string, any>;
      const organism = metrics.organism ?? {};
      return clamp(Number(organism.stress ?? 0));
    });
    const confidenceSeries = seriesOrFallback.map((snapshot) => {
      const metrics = (snapshot.metrics ?? {}) as Record<string, any>;
      const controller = metrics.controller ?? {};
      return clamp(Number(controller.confidence ?? 0.5));
    });
    const pendingRaw = seriesOrFallback.map((snapshot) => {
      const metrics = (snapshot.metrics ?? {}) as Record<string, any>;
      const pending = metrics.pending_prompts ?? {};
      return Number(pending.count ?? 0);
    });
    const errorsRaw = seriesOrFallback.map((snapshot) => {
      const metrics = (snapshot.metrics ?? {}) as Record<string, any>;
      const errors = metrics.errors ?? {};
      return Number(errors.open ?? 0);
    });
    const normalizeCount = (values: number[]) => {
      const max = Math.max(1, ...values);
      return values.map((value) => clamp(value / max));
    };

    const avatarMetrics: AvatarMetrics = {
      phase,
      certainty,
      health,
      memoryActivity,
      stress,
      fatigue,
      alignment,
      gateActivity,
      pendingPrompts,
      phaseHue,
      uncertainty,
      errorOpen,
      series: {
        health: healthSeries.length > 1 ? healthSeries : [health, health],
        memory: memorySeries.length > 1 ? memorySeries : [memoryActivity, memoryActivity],
        gate: gateSeries.length > 1 ? gateSeries : [gateActivity, gateActivity],
        stress: stressSeries.length > 1 ? stressSeries : [stress, stress],
        confidence: confidenceSeries.length > 1 ? confidenceSeries : [certainty, certainty],
        pending: normalizeCount(pendingRaw.length > 1 ? pendingRaw : [pendingPrompts, pendingPrompts]),
        errors: normalizeCount(errorsRaw.length > 1 ? errorsRaw : [errorOpen, errorOpen]),
      },
    };

    return {
      avatar: avatarMetrics,
      controller,
      organism,
      gate,
      recommendations: {
        total: recommendationTotal,
        eligible: recommendationEligible,
      },
      pending: {
        count: pendingPrompts,
        oldest: pending.oldest_at ?? avatar.pending_prompts ?? null,
      },
      attentionSchema,
      workspaceContrib,
      selfClaims,
      selfReflection,
    };
  }, [healthSnapshot, healthHistory, moduleStatus?.stage, pendingPromptCount]);

  const statusLabel = moduleStatus?.stage
    ? formatStatusLabel(moduleStatus.stage)
    : chatState.replace(/_/g, " ");
  const statusDetail = moduleStatus?.stage
    ? formatStatusDetail(moduleStatus.stage, moduleStatus.detail)
    : "Standing by";

  const workspaceSummary = metrics.workspaceContrib?.summary ?? "";
  const focusLine = unifiedLine || workspaceSummary || internalStateLine;
  const focusLabel = unifiedLine
    ? "Focus"
    : workspaceSummary
      ? "Workspace"
      : internalStateLine
        ? "State"
        : "";

  const essentialMetrics = [
    { label: "Health", value: `${(metrics.avatar.health * 100).toFixed(0)}%` },
    { label: "Confidence", value: `${(metrics.avatar.certainty * 100).toFixed(0)}%` },
    { label: "Gate", value: `${(metrics.avatar.gateActivity * 100).toFixed(0)}%` },
    { label: "Pending", value: `${metrics.avatar.pendingPrompts}` },
    { label: "Errors", value: `${metrics.avatar.errorOpen}` },
  ];

  return (
    <aside className="system-state-panel" data-phase={metrics.avatar.phase}>
      <div className="system-state-header">
        <div>
          <div className="system-state-title">System State</div>
          <div className="system-state-detail">{statusDetail}</div>
        </div>
        <div className="system-state-pill">{statusLabel}</div>
      </div>

      <div className="system-state-avatar" data-phase={metrics.avatar.phase}>
        <SystemStateBlob metrics={metrics.avatar} />
      </div>

      <div className="system-state-kpis">
        {essentialMetrics.map((metric) => (
          <div key={metric.label} className="system-state-kpi">
            <span className="system-state-kpi-label">{metric.label}</span>
            <span className="system-state-kpi-value">{metric.value}</span>
          </div>
        ))}
      </div>

      {focusLine && (
        <div className="system-state-focus">
          <span className="system-state-focus-label">{focusLabel}</span>
          <span className="system-state-focus-value">{truncate(focusLine, 90)}</span>
        </div>
      )}

      <div className="system-state-footer">
        {metrics.recommendations?.total > 0 && (
          <div className="system-state-signal">
            <span>Recommendations</span>
            <span>
              {metrics.recommendations.eligible}/{metrics.recommendations.total} eligible
            </span>
          </div>
        )}
        {memoryError && (
          <div className="system-state-signal system-state-alert">
            <span>Memory</span>
            <span>Issue detected</span>
          </div>
        )}
        {rollingSummaryError && !memoryError && (
          <div className="system-state-signal system-state-alert">
            <span>Summary</span>
            <span>Last run failed</span>
          </div>
        )}
      </div>
    </aside>
  );
};
