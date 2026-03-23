import { SystemHealthSnapshot, SystemLogEntry } from "../types/app";

interface HealthGatePanelProps {
  snapshot: SystemHealthSnapshot | null;
  history?: SystemHealthSnapshot[];
  gateInputs: SystemLogEntry[];
}

const formatTime = (value?: string | null) => {
  if (!value) return "--";
  try {
    return new Date(value).toLocaleTimeString();
  } catch {
    return value;
  }
};

const formatPercent = (value: number) => `${Math.round(value * 100)}%`;
const trendLabel = (current: number, previous?: number | null) => {
  if (previous === null || previous === undefined) return "trend: --";
  const delta = current - previous;
  if (Math.abs(delta) < 0.01) return "trend: flat";
  return delta > 0 ? "trend: up" : "trend: down";
};

export const HealthGatePanel = ({ snapshot, history = [], gateInputs }: HealthGatePanelProps) => {
  const metrics = (snapshot?.metrics ?? {}) as Record<string, any>;
  const prevMetrics = (history[1]?.metrics ?? {}) as Record<string, any>;
  const gate = metrics.gate ?? {};
  const prevGate = prevMetrics.gate ?? {};
  const organism = metrics.organism ?? {};
  const prevOrganism = prevMetrics.organism ?? {};
  const controller = metrics.controller ?? {};
  const prevController = prevMetrics.controller ?? {};
  const errors = metrics.errors ?? {};
  const memory = metrics.memory ?? {};
  const summaries = metrics.summaries ?? {};
  const pending = metrics.pending_prompts ?? {};
  const avatar = metrics.avatar ?? {};

  const latestGate = gateInputs[0];
  const latestPayload = (latestGate?.payload ?? {}) as any;
  const gateDecision =
    latestPayload.enforced_decision
    || latestPayload.soft_decision
    || latestPayload.legacy_decision
    || gate.last_decision
    || "--";
  const gateVerifyRate = Number(gate.verify_rate ?? 0);
  const gateAlert = gateVerifyRate > 0.4 || Number(gate.counts?.DENY ?? 0) > 0;

  const stress = Number(organism.stress ?? 0);
  const fatigue = Number(organism.fatigue ?? 0);
  const alignment = Number(organism.social_alignment ?? 0.5);
  const organismAlert = stress > 0.7 || fatigue > 0.7 || alignment < 0.35;

  const confidence = Number(controller.confidence ?? 0.5);
  const uncertainty = Number(controller.uncertainty ?? 0.5);
  const failureStreak = Number(controller.failure_streak ?? 0);
  const controllerAlert = confidence < 0.4 || uncertainty > 0.6 || failureStreak >= 3;

  const avatarHealth = Number(avatar.health ?? 0);
  const avatarAlert = avatarHealth < 0.5;

  const errorOpen = Number(errors.open ?? 0);
  const errorTotal = Number(errors.total ?? 0);
  const errorsAlert = errorOpen > 0 || errorTotal > 5;

  const memoryPasses = Number(memory.memory_pass_count ?? 0);
  const memoryWrites = Number(memory.write_count ?? 0);
  const memoryAlert = Boolean(memory.last_error_at);

  const rollingFailures = Number(summaries.rolling_failures ?? 0);
  const innerFailures = Number(summaries.inner_failures ?? 0);
  const summariesAlert = rollingFailures > 0 || innerFailures > 0;

  const pendingCount = Number(pending.count ?? 0);
  const pendingOldest = Number(pending.oldest_age_seconds ?? 0);
  const pendingAlert = pendingCount > 0 && pendingOldest > 300;

  return (
    <section className="cockpit-panel health-gate-panel">
      <div className="panel-header">
        <div>
          <h2>Health + Gates</h2>
          <p>Gate posture, organism load, and controller confidence in one view.</p>
        </div>
        <div className="health-gate-updated">Updated {formatTime(snapshot?.timestamp)}</div>
      </div>

      {!snapshot && <div className="panel-empty">No health snapshot yet.</div>}

      {snapshot && (
        <>
          <div className="health-gate-narrative">
            <span className={`rail-pill${gateAlert ? " alert" : ""}`}>{gateDecision}</span>
            <span className={`rail-pill${organismAlert ? " warn" : ""}`}>
              Stress {stress.toFixed(2)} / Fatigue {fatigue.toFixed(2)}
            </span>
            <span className={`rail-pill${controllerAlert ? " warn" : ""}`}>
              Confidence {formatPercent(confidence)}
            </span>
            <span className="rail-pill muted">Last gate {formatTime(latestGate?.timestamp)}</span>
          </div>

          <div className="health-tier">
            <div className="health-tier-label">Core Signals</div>
            <div className="health-tier-grid">
              <div className={`health-card primary${avatarAlert || errorsAlert ? " alert" : ""}`}>
                <h3>System Health</h3>
                <div className="health-value">{formatPercent(avatarHealth)}</div>
                <div className="health-meta">Errors open {errorOpen}</div>
                <div className="health-trend">{trendLabel(avatarHealth, Number((prevMetrics.avatar ?? {}).health ?? 0))}</div>
              </div>
              <div className={`health-card primary${gateAlert ? " alert" : ""}`}>
                <h3>Gate Posture</h3>
                <div className="health-value">Verify {formatPercent(gateVerifyRate)}</div>
                <div className="health-meta">Deny {gate.counts?.DENY ?? 0} · Notice {gate.counts?.ALLOW_WITH_NOTICE ?? 0}</div>
                <div className="health-trend">{trendLabel(gateVerifyRate, Number(prevGate.verify_rate ?? 0))}</div>
              </div>
              <div className={`health-card primary${organismAlert ? " alert" : ""}`}>
                <h3>Organism Load</h3>
                <div className="health-value">Stress {stress.toFixed(2)}</div>
                <div className="health-meta">Fatigue {fatigue.toFixed(2)} · Align {alignment.toFixed(2)}</div>
                <div className="health-trend">{trendLabel(stress, Number(prevOrganism.stress ?? 0))}</div>
              </div>
              <div className={`health-card primary${controllerAlert ? " alert" : ""}`}>
                <h3>Controller</h3>
                <div className="health-value">Confidence {formatPercent(confidence)}</div>
                <div className="health-meta">Uncertainty {formatPercent(uncertainty)} · Fail {failureStreak}</div>
                <div className="health-trend">{trendLabel(confidence, Number(prevController.confidence ?? 0.5))}</div>
              </div>
            </div>
          </div>

          <div className="health-tier">
            <div className="health-tier-label">Secondary Signals</div>
            <div className="health-tier-grid">
              <div className={`health-card${errorsAlert ? " alert" : ""}`}>
                <h3>Errors</h3>
                <div>Total {errorTotal}</div>
                <div>Open {errorOpen}</div>
                <div className="health-trend">{trendLabel(errorTotal, Number(prevMetrics.errors?.total ?? 0))}</div>
              </div>
              <div className={`health-card${memoryAlert ? " alert" : ""}`}>
                <h3>Memory</h3>
                <div>Passes {memoryPasses}</div>
                <div>Writes {memoryWrites}</div>
                <div className="health-trend">{trendLabel(memoryPasses, Number(prevMetrics.memory?.memory_pass_count ?? 0))}</div>
              </div>
              <div className={`health-card${summariesAlert ? " alert" : ""}`}>
                <h3>Summaries</h3>
                <div>Rolling {summaries.rolling_updates ?? 0}</div>
                <div>Inner {summaries.inner_updates ?? 0}</div>
                <div className="health-trend">Failures {rollingFailures + innerFailures}</div>
              </div>
              <div className={`health-card${pendingAlert ? " alert" : ""}`}>
                <h3>Pending</h3>
                <div>Queue {pendingCount}</div>
                <div>Oldest {pending.oldest_age_seconds ?? 0}s</div>
                <div className="health-trend">{trendLabel(pendingCount, Number(prevMetrics.pending_prompts?.count ?? 0))}</div>
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  );
};
