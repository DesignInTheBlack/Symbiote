import { SystemHealthSnapshot } from "../types/app";

interface SystemHealthTimelineProps {
  history: SystemHealthSnapshot[];
}

const formatTime = (value: string) => {
  try {
    return new Date(value).toLocaleTimeString();
  } catch {
    return value;
  }
};

export const SystemHealthTimeline = ({ history }: SystemHealthTimelineProps) => {
  const recent = history.slice(0, 12);
  return (
    <section className="cockpit-panel timeline-panel">
      <div className="panel-header">
        <div>
          <h2>Health Timeline</h2>
          <p>Recent health snapshots for trend context.</p>
        </div>
      </div>
      {recent.length === 0 ? (
        <div className="panel-empty">No snapshots yet.</div>
      ) : (
        <div className="timeline-list">
          {recent.map((snapshot) => {
            const metrics = snapshot.metrics ?? {};
            const avatar = metrics.avatar ?? {};
            const gate = metrics.gate ?? {};
            const errors = metrics.errors ?? {};
            const monologue = metrics.monologue ?? {};
            const loopRate = Number(monologue.loop_state_change_rate ?? 0);
            return (
              <div key={snapshot.snapshot_id} className="timeline-row">
                <div>
                  <strong>{formatTime(snapshot.timestamp)}</strong>
                  <div className="timeline-meta">
                    Health {Math.round(Number(avatar.health ?? 0) * 100)}% / Loop {Math.round(loopRate * 100)}% / Gate {gate.total ?? 0} / Errors {errors.total ?? 0}
                  </div>
                </div>
                <div className="timeline-phase">{avatar.processing_phase || "—"}</div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
};
