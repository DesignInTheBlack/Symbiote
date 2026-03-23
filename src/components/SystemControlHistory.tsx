import { SystemControlEvent } from "../types/app";

interface SystemControlHistoryProps {
  events: SystemControlEvent[];
}

const formatTime = (value: string) => {
  try {
    return new Date(value).toLocaleTimeString();
  } catch {
    return value;
  }
};

export const SystemControlHistory = ({ events }: SystemControlHistoryProps) => {
  const recent = events.slice(0, 12);
  return (
    <section className="cockpit-panel history-panel">
      <div className="panel-header">
        <div>
          <h2>Control History</h2>
          <p>Recent control changes and outcomes.</p>
        </div>
      </div>
      {recent.length === 0 ? (
        <div className="panel-empty">No control events yet.</div>
      ) : (
        <div className="history-list">
          {recent.map((event) => (
            <div key={event.event_id} className="history-row">
              <div>
                <strong>{event.subsystem_id}</strong>
                <div className="history-meta">
                  {event.previous_mode ? `${event.previous_mode} ? ` : ""}{event.new_mode}
                </div>
                {event.reason && <div className="history-reason">{event.reason}</div>}
              </div>
              <div className="history-time">{formatTime(event.timestamp)}</div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
};
