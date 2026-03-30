import { SystemHealthSnapshot, RecommendationItem } from "../types/app";
import { invokeWithTimeout } from "../utils/tauri";

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
const formatDelta = (current: number, previous?: number | null, precision = 2) => {
  if (previous === null || previous === undefined || Number.isNaN(previous)) return "delta: --";
  const delta = current - previous;
  if (Math.abs(delta) < 0.0001) return "delta: 0";
  const sign = delta > 0 ? "+" : "";
  return `delta: ${sign}${delta.toFixed(precision)}`;
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
  const prevQualia = prevMetrics.qualia ?? {};
  const errors = metrics.errors ?? {};
  const prevErrors = prevMetrics.errors ?? {};
  const memory = metrics.memory ?? {};
  const prevMemory = prevMetrics.memory ?? {};
  const summaries = metrics.summaries ?? {};
  const prevSummaries = prevMetrics.summaries ?? {};
  const pending = metrics.pending_prompts ?? {};
  const prevPending = prevMetrics.pending_prompts ?? {};
  const avatar = metrics.avatar ?? {};
  const prevAvatar = prevMetrics.avatar ?? {};
  const feedback = metrics.feedback ?? {};
  const prevFeedback = prevMetrics.feedback ?? {};
  const monologue = metrics.monologue ?? {};
  const prevMonologue = prevMetrics.monologue ?? {};
  const prediction = metrics.prediction ?? {};
  const prevPrediction = prevMetrics.prediction ?? {};
  const attentionSchema = metrics.attention_schema ?? {};
  const prevAttentionSchema = prevMetrics.attention_schema ?? {};
  const workspaceContrib = metrics.workspace_contributors ?? {};
  const prevWorkspaceContrib = prevMetrics.workspace_contributors ?? {};
  const telemetry = metrics.telemetry ?? {};
  const prevTelemetry = prevMetrics.telemetry ?? {};
  const outcomes = metrics.outcomes ?? {};
  const prevOutcomes = prevMetrics.outcomes ?? {};
  const scorecard = metrics.scorecard ?? {};
  const prevScorecard = prevMetrics.scorecard ?? {};
  const recommendationsBlock = metrics.recommendations ?? {};
  const recommendations = (recommendationsBlock.items ?? []) as RecommendationItem[];
  const recommendationTelemetry = recommendationsBlock.telemetry ?? {};

  const applyRecommendation = async (rec: RecommendationItem) => {
    if (!rec.action) return;
    await invokeWithTimeout(
      "apply_recommendation",
      {
        recommendation_id: rec.recommendation_id,
        kind: rec.kind,
        snapshot_id: snapshot?.snapshot_id ?? null,
        action: rec.action,
        gate: rec.gate ?? null,
        recovery_metric: rec.recovery_metric ?? null,
        recovery_target: rec.recovery_target ?? null,
        baseline_value: rec.baseline_value ?? null,
      },
      15000,
    );
  };

  const dismissRecommendation = async (rec: RecommendationItem) => {
    await invokeWithTimeout(
      "dismiss_recommendation",
      {
        recommendation_id: rec.recommendation_id,
        kind: rec.kind,
        snapshot_id: snapshot?.snapshot_id ?? null,
        gate: rec.gate ?? null,
      },
      10000,
    );
  };

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
  const summaryUpdates = Number(summaries.rolling_updates ?? 0) + Number(summaries.inner_updates ?? 0);
  const prevSummaryUpdates = Number(prevSummaries.rolling_updates ?? 0) + Number(prevSummaries.inner_updates ?? 0);
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
  const recommendationAcceptance = Number(recommendationTelemetry.acceptance_rate ?? 0);
  const recommendationSuccess = Number(recommendationTelemetry.success_rate ?? 0);
  const recommendationMedian = recommendationTelemetry.median_time_to_recovery_ms ?? null;

  const diffItems: { label: string; current: string; delta: string }[] = [];
  const pushDiff = (label: string, current: number, prev?: number | null, precision = 2) => {
    if (prev === null || prev === undefined || Number.isNaN(prev)) return;
    const delta = current - prev;
    if (Math.abs(delta) < 0.0001) return;
    const sign = delta > 0 ? "+" : "";
    diffItems.push({
      label,
      current: precision > 0 ? current.toFixed(precision) : `${Math.round(current)}`,
      delta: `${sign}${delta.toFixed(precision)}`,
    });
  };
  pushDiff("System Health", avatarHealth, Number(prevAvatar.health ?? 0));
  pushDiff("Verify Rate", gateVerifyRate, Number(prevGate.verify_rate ?? 0));
  pushDiff("Stress", stress, Number(prevOrganism.stress ?? 0));
  pushDiff("Confidence", confidence, Number(prevController.confidence ?? 0.5));
  pushDiff("Errors Open", errorOpen, Number(prevErrors.open ?? 0), 0);
  pushDiff("Memory Passes", memoryPasses, Number(prevMemory.memory_pass_count ?? 0), 0);
  pushDiff("Pending Count", pendingCount, Number(prevPending.count ?? 0), 0);
  pushDiff("Predictions Failed", predictionFailed, Number(prevPrediction.failed ?? 0), 0);

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
          {diffItems.length > 0 && (
            <details className="health-diff">
              <summary>Show changes since previous snapshot</summary>
              <div className="health-diff-list">
                {diffItems.map((item) => (
                  <div key={item.label} className="health-diff-row">
                    <span>{item.label}</span>
                    <span>{item.current}</span>
                    <span>{item.delta}</span>
                  </div>
                ))}
              </div>
            </details>
          )}
          {recommendations.length > 0 && (
            <div className="health-recommendations">
              <div className="health-recommendations-header">
                <div>
                  <strong>Recommendations</strong>
                  <span className="health-recommendations-count">{recommendations.length}</span>
                </div>
                <div className="health-recommendations-metrics">
                  <span>Accept {formatPercent(recommendationAcceptance)}</span>
                  <span>Success {formatPercent(recommendationSuccess)}</span>
                  {typeof recommendationMedian === "number" && (
                    <span>Median {(recommendationMedian / 1000).toFixed(0)}s</span>
                  )}
                </div>
              </div>
              <div className="health-recommendations-list">
                {recommendations.map((rec) => {
                  const gateReasons = Array.isArray(rec.gate?.reasons) ? rec.gate.reasons.join(", ") : null;
                  return (
                    <div key={rec.recommendation_id} className={`health-recommendation-card ${rec.status}`}>
                      <div className="health-recommendation-body">
                        <div className="health-recommendation-title">{rec.title}</div>
                        <div className="health-recommendation-detail">{rec.detail}</div>
                        {gateReasons && (
                          <div className="health-recommendation-gate">Gate: {gateReasons}</div>
                        )}
                      </div>
                      <div className="health-recommendation-actions">
                        {rec.status === "eligible" && rec.action && (
                          <button className="btn btn-secondary" onClick={() => applyRecommendation(rec)}>
                            Apply
                          </button>
                        )}
                        <button className="btn btn-tertiary" onClick={() => dismissRecommendation(rec)}>
                          Dismiss
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div className="health-tier">
            <div className="health-tier-label">Core Signals</div>
            <div className="health-tier-grid">
              <div className={`health-card primary${avatarAlert || errorsAlert ? " alert" : ""}`}>
                <h3 title="Overall system posture derived from avatar health and error load.">System Health</h3>
                <div className="health-value">{formatPercent(avatarHealth)}</div>
                <div className="health-meta">Errors open {errorOpen}</div>
                <div className="health-trend">{trendLabel(avatarHealth, Number(prevAvatar.health ?? 0))}</div>
                <div className="health-delta">{formatDelta(avatarHealth, Number(prevAvatar.health ?? 0))}</div>
              </div>
              <div className={`health-card primary${gateAlert ? " alert" : ""}`}>
                <h3 title="Share of responses requiring verification.">Verify Rate</h3>
                <div className="health-value">{formatPercent(gateVerifyRate)}</div>
                <div className="health-meta">Deny {gate.counts?.DENY ?? 0} | Notice {gate.counts?.ALLOW_WITH_NOTICE ?? 0}</div>
                <div className="health-trend">{trendLabel(gateVerifyRate, Number(prevGate.verify_rate ?? 0))}</div>
                <div className="health-delta">{formatDelta(gateVerifyRate, Number(prevGate.verify_rate ?? 0))}</div>
              </div>
              <div className={`health-card primary${organismAlert ? " alert" : ""}`}>
                <h3 title="Stress, fatigue, and alignment signals.">Organism Load</h3>
                <div className="health-value">Stress {stress.toFixed(2)}</div>
                <div className="health-meta">Fatigue {fatigue.toFixed(2)} | Align {alignment.toFixed(2)}</div>
                <div className="health-trend">{trendLabel(stress, Number(prevOrganism.stress ?? 0))}</div>
                <div className="health-delta">{formatDelta(stress, Number(prevOrganism.stress ?? 0))}</div>
              </div>
              <div className={`health-card primary${controllerAlert ? " alert" : ""}`}>
                <h3 title="Controller confidence, uncertainty, and failure streak.">Controller</h3>
                <div className="health-value">{formatPercent(confidence)}</div>
                <div className="health-meta">Uncertainty {formatPercent(uncertainty)} | Fail {failureStreak}</div>
                <div className="health-trend">{trendLabel(confidence, Number(prevController.confidence ?? 0.5))}</div>
                <div className="health-delta">{formatDelta(confidence, Number(prevController.confidence ?? 0.5))}</div>
              </div>
            </div>
          </div>

          <div className="health-tier">
            <div className="health-tier-label">Secondary Signals</div>
            <div className="health-tier-grid">
              <div className={`health-card${errorsAlert ? " alert" : ""}`}>
                <h3 title="Error totals and open errors.">Errors</h3>
                <div>Total {errorTotal}</div>
                <div>Open {errorOpen}</div>
                <div className="health-trend">{trendLabel(errorTotal, Number(prevErrors.total ?? 0))}</div>
                <div className="health-delta">{formatDelta(errorTotal, Number(prevErrors.total ?? 0), 0)}</div>
              </div>
              <div className={`health-card${memoryAlert ? " alert" : ""}`}>
                <h3 title="Memory pass volume, writes, and validation drift.">Memory</h3>
                <div>Passes {memoryPasses}</div>
                <div>Writes {memoryWrites}</div>
                <div>Validation {memoryValidationRuns} / Drift {memoryDrift}</div>
                <div className="health-meta">Last validation {memoryValidationLast}</div>
                <div className="health-trend">{trendLabel(memoryPasses, Number(prevMemory.memory_pass_count ?? 0))}</div>
                <div className="health-delta">{formatDelta(memoryPasses, Number(prevMemory.memory_pass_count ?? 0), 0)}</div>
              </div>
              <div className={`health-card${summariesAlert ? " alert" : ""}`}>
                <h3 title="Rolling and inner summary updates and failures.">Summaries</h3>
                <div>Rolling {summaries.rolling_updates ?? 0}</div>
                <div>Inner {summaries.inner_updates ?? 0}</div>
                <div className="health-trend">Failures {rollingFailures + innerFailures}</div>
                <div className="health-delta">{formatDelta(summaryUpdates, prevSummaryUpdates, 0)}</div>
              </div>
              <div className={`health-card${pendingAlert ? " alert" : ""}`}>
                <h3 title="Pending prompts waiting to surface.">Pending</h3>
                <div>Queue {pendingCount}</div>
                <div>Oldest {pending.oldest_age_seconds ?? 0}s</div>
                <div className="health-trend">{trendLabel(pendingCount, Number(prevPending.count ?? 0))}</div>
                <div className="health-delta">{formatDelta(pendingCount, Number(prevPending.count ?? 0), 0)}</div>
              </div>
              <div className={`health-card${feedbackAlert ? " alert" : ""}`}>
                <h3 title="Feedback totals and positive/negative balance.">Feedback</h3>
                <div>Total {feedbackTotal}</div>
                <div>Positive {feedbackPositive}</div>
                <div>Negative {feedbackNegative}</div>
                <div className="health-delta">{formatDelta(feedbackTotal, Number(prevFeedback.total ?? 0), 0)}</div>
              </div>
              <div className="health-card">
                <h3 title="Qualia label volume and intensity.">Qualia</h3>
                <div>Labels {qualia.labels ?? 0}</div>
                <div>Rewards {qualia.rewards ?? 0}</div>
                <div>Mean intensity {Number(qualia.mean_intensity ?? 0).toFixed(2)}</div>
                <div className="health-delta">{formatDelta(Number(qualia.labels ?? 0), Number(prevQualia.labels ?? 0), 0)}</div>
              </div>
              <div className={`health-card${loopAlert ? " alert" : ""}`}>
                <h3 title="Loop closure and no-op rates for monologue ticks.">Loop Closure</h3>
                <div>State change {formatPercent(loopStateChangeRate)}</div>
                <div>No-op {formatPercent(loopNoopRate)}</div>
                <div>Streak {loopNoopStreak}</div>
                <div className="health-delta">{formatDelta(loopNoopRate, Number(prevMonologue.loop_noop_rate ?? 0))}</div>
              </div>
              <div className={`health-card${predictionAlert ? " alert" : ""}`}>
                <h3 title="Prediction generation health and JSON repair rates.">Predictions</h3>
                <div>Complete {predictionComplete}</div>
                <div>Rejected {predictionRejected} / Failed {predictionFailed}</div>
                <div>Repair {formatPercent(predictionRepairRate)} / Shadow {residualShadowImpact.toFixed(1)}%</div>
                <div className="health-delta">{formatDelta(predictionFailed, Number(prevPrediction.failed ?? 0), 0)}</div>
              </div>
              <div className={`health-card${telemetryAlert ? " alert" : ""}`}>
                <h3 title="Telemetry calibrations and drift events.">Telemetry</h3>
                <div>Calibrations {telemetryRuns}</div>
                <div>Drift {telemetryDrift}</div>
                <div className="health-trend">Last {telemetryLast}</div>
                <div className="health-delta">{formatDelta(telemetryDrift, Number(prevTelemetry.drift_events ?? 0), 0)}</div>
              </div>
              <div className={`health-card${outcomeAlert ? " alert" : ""}`}>
                <h3 title="Outcome confirmations, disconfirmations, and accuracy.">Outcomes</h3>
                <div>Total {outcomeTotal}</div>
                <div>Confirm {outcomeConfirm} / Disconfirm {outcomeDisconfirm}</div>
                <div className="health-trend">Accuracy {formatPercentMaybe(outcomeAccuracy)}</div>
                <div className="health-delta">{formatDelta(outcomeTotal, Number(prevOutcomes.total ?? 0), 0)}</div>
              </div>
              <div className={`health-card${scoreAlert ? " alert" : ""}`}>
                <h3 title="Combined score and drift penalty.">Scorecard</h3>
                <div>Combined {formatPercentMaybe(combinedScore)}</div>
                <div>Penalty {driftPenalty.toFixed(2)}</div>
                <div className="health-trend">Outcome {formatPercentMaybe(outcomeAccuracy)}</div>
                <div className="health-delta">{formatDelta(Number(combinedScore ?? 0), Number(prevScorecard.combined_score ?? 0))}</div>
              </div>
              <div className={`health-card${attentionAlert ? " alert" : ""}`}>
                <h3 title="Attention capacity usage and stability.">Attention Schema</h3>
                <div>Capacity {formatPercent(attentionCapacity)}</div>
                <div>Stability {attentionStability.toFixed(2)}</div>
                <div>Policy {attentionPolicy}</div>
                <div className="health-delta">{formatDelta(attentionCapacity, Number(prevAttentionSchema.capacity_usage ?? 0))}</div>
              </div>
              <div className={`health-card${workspaceAlert ? " alert" : ""}`}>
                <h3 title="Workspace contributor coverage and missing rate.">Workspace Contributors</h3>
                <div>Snapshots {workspaceSnapshots}</div>
                <div>Missing {workspaceMissing} / Rate {formatPercent(workspaceMissingRate)}</div>
                <div>{workspaceSummary || "summary unavailable"}</div>
                <div className="health-delta">{formatDelta(workspaceMissingRate, Number(prevWorkspaceContrib.missing_rate ?? 0))}</div>
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  );
};

