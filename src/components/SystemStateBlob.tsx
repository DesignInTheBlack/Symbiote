import React from "react";

export type SystemPhase =
  | "idle"
  | "thinking"
  | "responding"
  | "consolidating"
  | "retrieving"
  | "awaiting"
  | "error"
  | "stopping";

export interface AvatarMetrics {
  phase: SystemPhase;
  certainty: number;
  health: number;
  memoryActivity: number;
  stress: number;
  fatigue: number;
  alignment: number;
  gateActivity: number;
  pendingPrompts: number;
  phaseHue: number;
  uncertainty: number;
  errorOpen: number;
  series: {
    health: number[];
    memory: number[];
    gate: number[];
    stress: number[];
    confidence: number[];
    pending: number[];
    errors: number[];
  };
}

interface SystemStateBlobProps {
  metrics: AvatarMetrics;
}

const clamp = (value: number, min = 0, max = 1) => Math.min(max, Math.max(min, value));

const buildPath = (values: number[]) => {
  if (!values || values.length === 0) return "";
  const width = 100;
  const height = 100;
  const pad = 8;
  const span = Math.max(1, values.length - 1);
  const step = (width - pad * 2) / span;
  return values.map((value, index) => {
    const x = pad + step * index;
    const y = pad + (1 - clamp(value)) * (height - pad * 2);
    return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(" ");
};

export const SystemStateBlob: React.FC<SystemStateBlobProps> = ({ metrics }) => {
  const drive = clamp((metrics.memoryActivity + metrics.gateActivity + metrics.stress + (1 - metrics.health)) / 4);
  const pending = clamp(metrics.pendingPrompts / 6);
  const patternAlpha = clamp(metrics.uncertainty);
  const patternSize = `${Math.round(32 - metrics.memoryActivity * 14)}px`;
  const patternDrift = `${Math.round(24 - metrics.gateActivity * 10)}s`;
  const orbitCount = Math.min(6, Math.max(0, Math.round(metrics.pendingPrompts)));
  const orbiters = Array.from({ length: orbitCount });
  const errorIntensity = clamp(metrics.errorOpen / 5);

  const healthPath = buildPath(metrics.series.health);
  const memoryPath = buildPath(metrics.series.memory);
  const gatePath = buildPath(metrics.series.gate);
  const stressPath = buildPath(metrics.series.stress);
  const confidencePath = buildPath(metrics.series.confidence);
  const pendingPath = buildPath(metrics.series.pending);
  const errorsPath = buildPath(metrics.series.errors);

  const style: React.CSSProperties = {
    ["--avatar-certainty" as any]: clamp(metrics.certainty),
    ["--avatar-health" as any]: clamp(metrics.health),
    ["--avatar-memory" as any]: clamp(metrics.memoryActivity),
    ["--avatar-stress" as any]: clamp(metrics.stress),
    ["--avatar-fatigue" as any]: clamp(metrics.fatigue),
    ["--avatar-alignment" as any]: clamp(metrics.alignment),
    ["--avatar-gate" as any]: clamp(metrics.gateActivity),
    ["--avatar-phase-hue" as any]: metrics.phaseHue,
    ["--avatar-drive" as any]: drive,
    ["--avatar-pending" as any]: pending,
    ["--avatar-response" as any]: metrics.phase === "responding" ? 1 : 0,
    ["--avatar-error" as any]: metrics.phase === "error" ? 1 : 0,
    ["--avatar-error-open" as any]: errorIntensity,
    ["--avatar-pattern-alpha" as any]: patternAlpha,
    ["--avatar-pattern-size" as any]: patternSize,
    ["--avatar-pattern-drift" as any]: patternDrift,
  };

  return (
    <div
      className="system-state-blob"
      data-phase={metrics.phase}
      style={style}
      aria-label={`System avatar phase ${metrics.phase}`}
    >
      <div className="avatar-pattern" aria-hidden="true" />
      <svg className="avatar-telemetry" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <path className="telemetry-line line-health" d={healthPath} />
        <path className="telemetry-line line-memory" d={memoryPath} />
        <path className="telemetry-line line-gate" d={gatePath} />
        <path className="telemetry-line line-stress" d={stressPath} />
        <path className="telemetry-line line-confidence" d={confidencePath} />
        <path className="telemetry-line line-pending" d={pendingPath} />
        <path className="telemetry-line line-errors" d={errorsPath} />
      </svg>
      <div className="avatar-frame" aria-hidden="true">
        <svg className="avatar-rings" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
          <circle className="ring ring-health" cx="50" cy="50" r="22" />
          <circle className="ring ring-memory" cx="50" cy="50" r="34" />
          <circle className="ring ring-gate" cx="50" cy="50" r="45" />
          {metrics.errorOpen > 0 && <circle className="ring ring-fault" cx="50" cy="50" r="56" />}
        </svg>
        <div className="avatar-breath" />
        <div className="avatar-core">
          <span className="avatar-core-glow" />
          <span className="avatar-core-heart" />
        </div>
        {orbiters.length > 0 && (
          <div className="avatar-orbits">
            {orbiters.map((_, index) => (
              <span
                key={`orb-${index}`}
                className="avatar-orbit"
                style={{
                  ["--orbit-radius" as any]: `${48 + index * 10}px`,
                  ["--orbit-delay" as any]: `${index * -0.8}s`,
                }}
              />
            ))}
          </div>
        )}
      </div>
      {metrics.pendingPrompts > 0 && (
        <div className="avatar-pending" aria-label={`Pending prompts: ${metrics.pendingPrompts}`}>
          {metrics.pendingPrompts}
        </div>
      )}
    </div>
  );
};
