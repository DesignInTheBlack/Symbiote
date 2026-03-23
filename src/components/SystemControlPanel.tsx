import { useMemo, useState } from "react";
import { invokeWithTimeout } from "../utils/tauri";

export interface SubsystemState {
  id: string;
  label: string;
  class: string;
  default_mode: string;
  supported_modes: string[];
  depends_on: string[];
  enforcement_notes?: string;
  mode: string;
  updated_at?: string;
  updated_by?: string;
  reason?: string;
  value_json?: string | null;
}

interface SystemControlPanelProps {
  subsystemStates: SubsystemState[];
  allowWrites: boolean;
  onRefresh: () => void;
  error?: string | null;
}

const normalizeMode = (mode: string) => mode?.toLowerCase() || "normal";

export const SystemControlPanel = ({
  subsystemStates,
  allowWrites,
  onRefresh,
  error,
}: SystemControlPanelProps) => {
  const [pendingModes, setPendingModes] = useState<Record<string, string>>({});
  const [pendingReasons, setPendingReasons] = useState<Record<string, string>>({});
  const [pendingOverride, setPendingOverride] = useState<Record<string, boolean>>({});
  const [pendingError, setPendingError] = useState<string | null>(null);

  const sortedSubsystems = useMemo(() => {
    return [...subsystemStates].sort((a, b) => a.label.localeCompare(b.label));
  }, [subsystemStates]);

  const modeMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const subsystem of sortedSubsystems) {
      map.set(subsystem.id, normalizeMode(subsystem.mode));
    }
    return map;
  }, [sortedSubsystems]);

  const criticalWarning = sortedSubsystems.some((subsystem) => {
    if (subsystem.class !== "critical") return false;
    const mode = normalizeMode(subsystem.mode);
    return mode === "off" || mode === "degraded";
  });

  const getMode = (id: string, fallback: string) => {
    return pendingModes[id] ?? fallback;
  };

  const getReason = (id: string) => {
    return pendingReasons[id] ?? "";
  };

  const setMode = (id: string, mode: string) => {
    setPendingModes((prev) => ({ ...prev, [id]: mode }));
  };

  const setReason = (id: string, reason: string) => {
    setPendingReasons((prev) => ({ ...prev, [id]: reason }));
  };

  const setOverride = (id: string, value: boolean) => {
    setPendingOverride((prev) => ({ ...prev, [id]: value }));
  };

  const applyChange = async (subsystem: SubsystemState) => {
    const targetMode = normalizeMode(getMode(subsystem.id, subsystem.mode));
    const reason = getReason(subsystem.id).trim();

    if (!reason) {
      setPendingError("Reason is required for control changes.");
      return;
    }

    if (subsystem.class === "critical" && targetMode === "off") {
      const confirmed = window.confirm(
        `Disable critical subsystem '${subsystem.label}'? This can halt responses.`
      );
      if (!confirmed) return;
    }

    try {
      setPendingError(null);
      await invokeWithTimeout(
        "set_system_control",
        {
          subsystemId: subsystem.id,
          mode: targetMode,
          valueJson: subsystem.value_json ?? null,
          reason,
          overrideCritical: pendingOverride[subsystem.id] ?? false,
          actor: "ui",
        },
        15000
      );
      setPendingModes((prev) => {
        const next = { ...prev };
        delete next[subsystem.id];
        return next;
      });
      setPendingReasons((prev) => {
        const next = { ...prev };
        delete next[subsystem.id];
        return next;
      });
      setPendingOverride((prev) => {
        const next = { ...prev };
        delete next[subsystem.id];
        return next;
      });
      onRefresh();
    } catch (e: any) {
      setPendingError(String(e));
    }
  };

  const applySafeMode = async () => {
    const critical = sortedSubsystems.filter((sub) => sub.class === "critical");
    for (const subsystem of critical) {
      const reason = "Safe mode reset";
      try {
        await invokeWithTimeout(
          "set_system_control",
          {
            subsystemId: subsystem.id,
            mode: "normal",
            valueJson: subsystem.value_json ?? null,
            reason,
            overrideCritical: true,
            actor: "ui",
          },
          15000
        );
      } catch (e: any) {
        setPendingError(String(e));
      }
    }
    onRefresh();
  };

  return (
    <section className="cockpit-panel control-panel">
      <div className="panel-header">
        <div>
          <h2>System Controls</h2>
          <p>Dependency-aware control plane for runtime subsystems.</p>
        </div>
        <div className="panel-actions">
          <button className="btn btn-secondary" onClick={onRefresh}>Refresh</button>
          <button className="btn btn-secondary" onClick={applySafeMode} disabled={!allowWrites}>Safe Mode</button>
        </div>
      </div>

      {!allowWrites && (
        <div className="panel-banner warning">
          Cockpit write mode is disabled. Controls are read-only.
        </div>
      )}

      {criticalWarning && (
        <div className="panel-banner warning">
          One or more critical subsystems are degraded or off. Expect reduced capability.
        </div>
      )}

      {(pendingError || error) && (
        <div className="panel-banner error">{pendingError || error}</div>
      )}

      {sortedSubsystems.length === 0 ? (
        <div className="panel-empty">No subsystem registry loaded yet.</div>
      ) : (
        <div className="control-table">
        {sortedSubsystems.map((subsystem) => {
          const mode = getMode(subsystem.id, subsystem.mode);
          const reason = getReason(subsystem.id);
          const supportsReadOnly = subsystem.supported_modes?.includes("read_only");
          const dependencyIssues = (subsystem.depends_on || []).filter(
            (dep) => modeMap.get(dep) === "off"
          );
          return (
            <div key={subsystem.id} className="control-row">
              <div className="control-main">
                <div className="control-title">
                  <span className={`control-class ${subsystem.class}`}>{subsystem.class}</span>
                  <span>{subsystem.label}</span>
                </div>
                <div className="control-meta">
                  <span>ID: {subsystem.id}</span>
                  {subsystem.depends_on?.length > 0 && (
                    <span>Depends on: {subsystem.depends_on.join(", ")}</span>
                  )}
                </div>
                {dependencyIssues.length > 0 && (
                  <div className="control-warning">
                    Blocked by dependency: {dependencyIssues.join(", ")}
                  </div>
                )}
                {subsystem.enforcement_notes && (
                  <div className="control-notes">{subsystem.enforcement_notes}</div>
                )}
              </div>
              <div className="control-actions">
                <select
                  className="input"
                  value={mode}
                  onChange={(e) => setMode(subsystem.id, e.target.value)}
                  disabled={!allowWrites}
                >
                  {subsystem.supported_modes.map((option) => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
                <input
                  className="input"
                  placeholder="Reason"
                  value={reason}
                  onChange={(e) => setReason(subsystem.id, e.target.value)}
                  disabled={!allowWrites}
                />
                {subsystem.class === "critical" && (
                  <label className="control-override">
                    <input
                      type="checkbox"
                      checked={pendingOverride[subsystem.id] ?? false}
                      onChange={(e) => setOverride(subsystem.id, e.target.checked)}
                      disabled={!allowWrites}
                    />
                    Override
                  </label>
                )}
                <button
                  className="btn btn-primary"
                  onClick={() => applyChange(subsystem)}
                  disabled={!allowWrites || (!supportsReadOnly && mode === "read_only")}
                >
                  Apply
                </button>
              </div>
              <div className="control-status">
                <span>Current: {normalizeMode(subsystem.mode)}</span>
                {subsystem.updated_at && <span>Updated: {subsystem.updated_at}</span>}
                {subsystem.reason && <span>Reason: {subsystem.reason}</span>}
              </div>
            </div>
          );
        })}
      </div>
      )}
    </section>
  );
};
