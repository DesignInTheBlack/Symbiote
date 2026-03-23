import { SystemHealthSnapshot, SystemLogEntry } from "../types/app";

interface GateSignalStripProps {
  gateInputs: SystemLogEntry[];
  snapshot: SystemHealthSnapshot | null;
}

const formatTime = (value?: string | null) => {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleTimeString();
  } catch {
    return value;
  }
};

export const GateSignalStrip = ({ gateInputs, snapshot }: GateSignalStripProps) => {
  const latest = gateInputs[0];
  const payload = (latest?.payload ?? {}) as any;
  const gate = (snapshot?.metrics ?? {}).gate ?? {};
  const organism = (snapshot?.metrics ?? {}).organism ?? {};
  const signals = payload.signals ? Object.keys(payload.signals) : [];

  return (
    <section className="cockpit-panel gate-strip">
      <div className="panel-header">
        <div>
          <h2>Gate + Organism</h2>
          <p>Latest gate inputs with organism state.</p>
        </div>
      </div>
      <div className="gate-strip-content">
        <div className="gate-strip-block">
          <div className="gate-strip-label">Latest Gate</div>
          <div className="gate-strip-value">{payload.enforced_decision || "—"}</div>
          <div className="gate-strip-meta">{formatTime(latest?.timestamp)}</div>
          {(payload.soft_decision || payload.legacy_decision) && (
            <div className="gate-strip-meta">
              Soft {payload.soft_decision || "—"} · Legacy {payload.legacy_decision || "—"}
            </div>
          )}
          {payload.gate_reasons && Array.isArray(payload.gate_reasons) && payload.gate_reasons.length > 0 && (
            <div className="gate-strip-meta">Reasons: {payload.gate_reasons.slice(0, 3).join(", ")}</div>
          )}
          {signals.length > 0 && (
            <div className="gate-strip-meta">Signals: {signals.slice(0, 4).join(", ")}</div>
          )}
        </div>
        <div className="gate-strip-block">
          <div className="gate-strip-label">Gate Counts</div>
          <div className="gate-strip-value">{gate.total ?? 0} total</div>
          <div className="gate-strip-meta">Verify rate {Math.round(Number(gate.verify_rate ?? 0) * 100)}%</div>
        </div>
        <div className="gate-strip-block">
          <div className="gate-strip-label">Organism</div>
          <div className="gate-strip-value">Stress {Number(organism.stress ?? 0).toFixed(2)}</div>
          <div className="gate-strip-meta">Fatigue {Number(organism.fatigue ?? 0).toFixed(2)}</div>
          <div className="gate-strip-meta">Alignment {Number(organism.social_alignment ?? 0.5).toFixed(2)}</div>
        </div>
      </div>
    </section>
  );
};
