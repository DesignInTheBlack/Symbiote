import { SystemHealthSnapshot } from "../types/app";

interface SystemHealthPanelProps {
  snapshot: SystemHealthSnapshot | null;
  history?: SystemHealthSnapshot[];
}

const formatPercent = (value: number) => `${Math.round(value * 100)}%`;
const formatPercentMaybe = (value: number | null) => (value === null ? "n/a" : formatPercent(value));
const trendLabel = (current: number, previous?: number | null) => {
  if (previous === null || previous === undefined) return "trend: --";
  const delta = current - previous;
  if (Math.abs(delta) < 0.01) return "trend: flat";
  return delta > 0 ? "trend: up" : "trend: down";
};

export const SystemHealthPanel = ({ snapshot, history = [] }: SystemHealthPanelProps) => {
  const metrics = (snapshot?.metrics ?? {}) as Record<string, any>;
  const prevMetrics = (history[1]?.metrics ?? {}) as Record<string, any>;
  const gate = metrics.gate ?? {};
  const prevGate = prevMetrics.gate ?? {};
  const organism = metrics.organism ?? {};
  const prevOrganism = prevMetrics.organism ?? {};
  const controller = metrics.controller ?? {};
  const prevController = prevMetrics.controller ?? {};
  const qualia = metrics.qualia ?? {};
  const errors = metrics.errors ?? {};
  const prevErrors = prevMetrics.errors ?? {};
  const memory = metrics.memory ?? {};
  const prevMemory = prevMetrics.memory ?? {};
  const summaries = metrics.summaries ?? {};
  const pending = metrics.pending_prompts ?? {};
  const prevPending = prevMetrics.pending_prompts ?? {};
  const avatar = metrics.avatar ?? {};
  const prevAvatar = prevMetrics.avatar ?? {};
  const feedback = metrics.feedback ?? {};
  const monologue = metrics.monologue ?? {};
  const prediction = metrics.prediction ?? {};
  const attentionSchema = metrics.attention_schema ?? {};
  const workspaceContrib = metrics.workspace_contributors ?? {};
  const telemetry = metrics.telemetry ?? {};
  const outcomes = metrics.outcomes ?? {};
  const scorecard = metrics.scorecard ?? {};

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

  const errorTotal = Number(errors.total ?? 0);
  const errorOpen = Number(errors.open ?? 0);
  const errorsAlert = errorOpen > 0 || errorTotal > 5;

  const memoryPasses = Number(memory.memory_pass_count ?? 0);
  const memoryWrites = Number(memory.write_count ?? 0);
  const memoryValidationRuns = Number(memory.validation_runs ?? 0);
  const memoryDrift = Number(memory.drift_events ?? 0);
  const memoryValidationLast = memory.last_validation_at ?? "--";
  const memoryAlert = Boolean(memory.last_error_at) || memoryDrift > 0;

  const rollingFailures = Number(summaries.rolling_failures ?? 0);
  const innerFailures = Number(summaries.inner_failures ?? 0);
  const summariesAlert = rollingFailures > 0 || innerFailures > 0;

  const pendingCount = Number(pending.count ?? 0);
  const pendingOldest = Number(pending.oldest_age_seconds ?? 0);
  const pendingAlert = pendingCount > 0 && pendingOldest > 300;

  const avatarHealth = Number(avatar.health ?? 0);
  const avatarAlert = avatarHealth < 0.5;
  const feedbackTotal = Number(feedback.total ?? 0);
  const feedbackNegative = Number(feedback.negative ?? 0);
  const feedbackPositive = Number(feedback.positive ?? 0);
  const feedbackAlert = feedbackNegative > feedbackPositive && feedbackTotal > 0;

  const loopStateChangeRate = Number(monologue.loop_state_change_rate ?? 0);
  const loopNoopRate = Number(monologue.loop_noop_rate ?? 0);
  const loopNoopStreak = Number(monologue.loop_noop_streak ?? 0);
  const loopAlert = loopStateChangeRate < 0.2 || loopNoopStreak >= 3;

  const predictionComplete = Number(prediction.generation_complete ?? 0);
  const predictionRejected = Number(prediction.rejected ?? 0);
  const predictionFailed = Number(prediction.failed ?? 0);
  const predictionRepair = Number(prediction.json_repair ?? 0);
  const predictionRepairRate = Number(prediction.json_repair_rate ?? (predictionComplete > 0 ? predictionRepair / predictionComplete : 0));
  const residualShadowImpact = Number(prediction.residual_shadow_impact_pct ?? 0);
  const predictionAlert = predictionFailed > 0 || predictionRepairRate > 0.25 || residualShadowImpact > 5;

  const attentionCapacity = Number(attentionSchema.capacity_usage ?? 0);
  const attentionStability = Number(attentionSchema.avg_stability ?? attentionSchema.stability ?? 0);
  const attentionPolicy = attentionSchema.selection_policy ?? "none";
  const attentionAlert = attentionCapacity > 0.9 || attentionStability < 0.2;

  const workspaceSnapshots = Number(workspaceContrib.snapshots ?? 0);
  const workspaceMissing = Number(workspaceContrib.missing ?? 0);
  const workspaceMissingRate = Number(workspaceContrib.missing_rate ?? 0);
  const workspaceSummary = workspaceContrib.summary ?? "";
  const workspaceAlert = workspaceMissingRate > 0.3 && workspaceSnapshots > 0;

  const telemetryRuns = Number(telemetry.calibration_runs ?? 0);
  const telemetryDrift = Number(telemetry.drift_events ?? 0);
  const telemetryLast = telemetry.last_calibration_at ?? "--";
  const telemetryAlert = telemetryDrift > 0;
  const outcomeTotal = Number(outcomes.total ?? 0);
  const outcomeConfirm = Number(outcomes.confirm ?? 0);
  const outcomeDisconfirm = Number(outcomes.disconfirm ?? 0);
  const outcomeAccuracy = typeof outcomes.accuracy === "number" ? outcomes.accuracy : null;
  const outcomeAlert = outcomeTotal > 0 && outcomeAccuracy !== null && outcomeAccuracy < 0.6;
  const combinedScore = typeof scorecard.combined_score === "number" ? scorecard.combined_score : null;
  const driftPenalty = Number(scorecard.drift_penalty ?? 0);
  const scoreAlert = outcomeTotal > 0 && combinedScore !== null && combinedScore < 0.6;

  return (
    <section className="cockpit-panel health-panel">
      <div className="panel-header">
        <div>
          <h2>System Health</h2>
          <p>Core and secondary signals from the latest health snapshots.</p>
        </div>
      </div>

      {!snapshot && (
        <div className="panel-empty">No health snapshot yet.</div>
      )}

      {snapshot && (
        <>
          {outcomeTotal === 0 && (
            <div className="health-callout">
              <strong>Outcomes missing.</strong> Record confirm or disconfirm outcomes so the
              system can calibrate its predictions.
            </div>
          )}

          <div className="health-tier">
            <div className="health-tier-label">Core Signals</div>
            <div className="health-tier-grid">
              <div className={`health-card primary${avatarAlert || errorsAlert ? " alert" : ""}`}>
                <h3>System Health</h3>
                <div className="health-value">{formatPercent(avatarHealth)}</div>
                <div className="health-meta">Errors open {errorOpen}</div>
                <div className="health-trend">{trendLabel(avatarHealth, Number(prevAvatar.health ?? 0))}</div>
              </div>
              <div className={`health-card primary${gateAlert ? " alert" : ""}`}>
                <h3>Verify Rate</h3>
                <div className="health-value">{formatPercent(gateVerifyRate)}</div>
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
                <div className="health-value">{formatPercent(confidence)}</div>
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
                <div className="health-trend">{trendLabel(errorTotal, Number(prevErrors.total ?? 0))}</div>
              </div>
              <div className={`health-card${memoryAlert ? " alert" : ""}`}>
                <h3>Memory</h3>
                <div>Passes {memoryPasses}</div>
                <div>Writes {memoryWrites}</div>
                <div>Validation {memoryValidationRuns} / Drift {memoryDrift}</div>
                <div className="health-meta">Last validation {memoryValidationLast}</div>
                <div className="health-trend">{trendLabel(memoryPasses, Number(prevMemory.memory_pass_count ?? 0))}</div>
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
                <div className="health-trend">{trendLabel(pendingCount, Number(prevPending.count ?? 0))}</div>
              </div>
              <div className={`health-card${feedbackAlert ? " alert" : ""}`}>
                <h3>Feedback</h3>
                <div>Total {feedbackTotal}</div>
                <div>Positive {feedbackPositive}</div>
                <div>Negative {feedbackNegative}</div>
              </div>
              <div className="health-card">
                <h3>Qualia</h3>
                <div>Labels {qualia.labels ?? 0}</div>
                <div>Rewards {qualia.rewards ?? 0}</div>
                <div>Mean intensity {Number(qualia.mean_intensity ?? 0).toFixed(2)}</div>
              </div>
              <div className={`health-card${loopAlert ? " alert" : ""}`}>
                <h3>Loop Closure</h3>
                <div>State change {formatPercent(loopStateChangeRate)}</div>
                <div>No-op {formatPercent(loopNoopRate)}</div>
                <div>Streak {loopNoopStreak}</div>
              </div>
              <div className={`health-card${predictionAlert ? " alert" : ""}`}>
                <h3>Predictions</h3>
                <div>Complete {predictionComplete}</div>
                <div>Rejected {predictionRejected} / Failed {predictionFailed}</div>
                <div>Repair {formatPercent(predictionRepairRate)} / Shadow {residualShadowImpact.toFixed(1)}%</div>
              </div>
              <div className={`health-card${telemetryAlert ? " alert" : ""}`}>
                <h3>Telemetry</h3>
                <div>Calibrations {telemetryRuns}</div>
                <div>Drift {telemetryDrift}</div>
                <div className="health-trend">Last {telemetryLast}</div>
              </div>
              <div className={`health-card${outcomeAlert ? " alert" : ""}`}>
                <h3>Outcomes</h3>
                <div>Total {outcomeTotal}</div>
                <div>Confirm {outcomeConfirm} / Disconfirm {outcomeDisconfirm}</div>
                <div className="health-trend">Accuracy {formatPercentMaybe(outcomeAccuracy)}</div>
              </div>
              <div className={`health-card${scoreAlert ? " alert" : ""}`}>
                <h3>Scorecard</h3>
                <div>Combined {formatPercentMaybe(combinedScore)}</div>
                <div>Penalty {driftPenalty.toFixed(2)}</div>
                <div className="health-trend">Outcome {formatPercentMaybe(outcomeAccuracy)}</div>
              </div>
              <div className={`health-card${attentionAlert ? " alert" : ""}`}>
                <h3>Attention Schema</h3>
                <div>Capacity {formatPercent(attentionCapacity)}</div>
                <div>Stability {attentionStability.toFixed(2)}</div>
                <div>Policy {attentionPolicy}</div>
              </div>
              <div className={`health-card${workspaceAlert ? " alert" : ""}`}>
                <h3>Workspace Contributors</h3>
                <div>Snapshots {workspaceSnapshots}</div>
                <div>Missing {workspaceMissing} / Rate {formatPercent(workspaceMissingRate)}</div>
                <div>{workspaceSummary || "summary unavailable"}</div>
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  );
};
