import { useCallback, useEffect, useMemo, useState } from "react";
import type { Message, OutcomeEvent, SystemLogEntry } from "../types/app";
import { invokeWithTimeout } from "../utils/tauri";

const TIMEOUTS = {
  short: 5000,
  medium: 15000,
};

type OutcomePanelProps = {
  messages: Message[];
  systemLogs: SystemLogEntry[];
  allowWrites: boolean;
};

const truncate = (text: string, max = 120) =>
  text.length > max ? `${text.slice(0, max)}...` : text;

const extractEvidenceIds = (message: Message, logs: SystemLogEntry[]): number[] => {
  const meta = message.metadata ?? {};
  const direct = meta.evidence_event_ids as number[] | undefined;
  if (Array.isArray(direct) && direct.length > 0) {
    return direct.filter((id) => typeof id === "number");
  }
  if (typeof meta.evidence_event_id === "number") {
    return [meta.evidence_event_id];
  }
  if (message.run_id) {
    const report = logs.find(
      (entry) => (entry.payload as any)?.event === "decision_report" && entry.run_id === message.run_id
    );
    const ids = (report?.payload as any)?.evidence_event_ids;
    if (Array.isArray(ids)) {
      return ids.filter((id: any) => typeof id === "number");
    }
  }
  return [];
};

export const OutcomePanel = ({ messages, systemLogs, allowWrites }: OutcomePanelProps) => {
  const [outcomes, setOutcomes] = useState<OutcomeEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const recentMessages = useMemo(
    () =>
      messages
        .filter((msg) => msg.role === "assistant" && msg.status === "complete")
        .slice(-8)
        .reverse(),
    [messages]
  );

  const loadOutcomes = useCallback(async () => {
    try {
      setError(null);
      const list = await invokeWithTimeout<OutcomeEvent[]>(
        "list_outcomes",
        { limit: 40 },
        TIMEOUTS.medium
      );
      setOutcomes(list);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    void loadOutcomes();
  }, [loadOutcomes]);

  const recordOutcome = async (message: Message, verdict: "confirm" | "disconfirm" | "inconclusive") => {
    if (!allowWrites) return;
    const evidenceIds = extractEvidenceIds(message, systemLogs);
    const confidence = verdict === "inconclusive" ? 0.5 : 0.8;
    try {
      setLoading(true);
      await invokeWithTimeout(
        "record_outcome",
        {
          runId: message.run_id ?? undefined,
          traceId: message.trace_id ?? undefined,
          candidateId: message.message_id,
          targetType: "message",
          verdict,
          confidence,
          source: "operator",
          note: null,
          evidenceEventIds: evidenceIds,
        },
        TIMEOUTS.medium
      );
      await loadOutcomes();
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="cockpit-panel outcome-panel">
      <div className="panel-header">
        <div>
          <h2>Outcome Adjudication</h2>
          <p>Confirm or disconfirm recent outputs to close the loop with real-world results.</p>
        </div>
      </div>

      {error && <div className="panel-empty">Failed to load outcomes: {error}</div>}

      <div className="outcome-actions">
        {recentMessages.length === 0 ? (
          <div className="panel-empty">No recent assistant messages.</div>
        ) : (
          recentMessages.map((msg) => (
            <div key={msg.message_id} className="outcome-row">
              <div className="outcome-snippet">{truncate(msg.content)}</div>
              <div className="outcome-meta">
                <span className="history-meta">{msg.run_id ?? "no-run"}</span>
                <span className="history-meta">{msg.created_at ?? ""}</span>
              </div>
              <div className="outcome-buttons">
                <button
                  className="btn btn-secondary"
                  disabled={!allowWrites || loading}
                  onClick={() => recordOutcome(msg, "confirm")}
                >
                  Confirm
                </button>
                <button
                  className="btn btn-secondary"
                  disabled={!allowWrites || loading}
                  onClick={() => recordOutcome(msg, "disconfirm")}
                >
                  Disconfirm
                </button>
                <button
                  className="btn btn-secondary"
                  disabled={!allowWrites || loading}
                  onClick={() => recordOutcome(msg, "inconclusive")}
                >
                  Inconclusive
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      <div className="outcome-history">
        <h3>Recent Outcomes</h3>
        {outcomes.length === 0 ? (
          <div className="panel-empty">No outcomes recorded.</div>
        ) : (
          outcomes.slice(0, 12).map((event) => (
            <div key={event.outcome_id} className="outcome-history-row">
              <span className={`outcome-pill outcome-${event.verdict}`}>{event.verdict}</span>
              <span className="history-meta">{event.candidate_id ?? "--"}</span>
              <span className="history-meta">{event.created_at}</span>
            </div>
          ))
        )}
      </div>
    </section>
  );
};

