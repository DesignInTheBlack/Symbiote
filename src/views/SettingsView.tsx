import { useEffect, useState, type ReactNode } from "react";
import { getUiDebugFlags, setUiDebugFlags, UiDebugFlags } from "../utils/async";
import { invokeWithTimeout } from "../utils/tauri";
import { ParameterRegistry, PromptStatus, Settings, SelfInspection, SelfModel, SystemControlEntry, SystemLogEntry, TestResult, WaveStatus } from "../types/app";
import { DEFAULT_THEME_ID, listThemeOptions, ThemeOption } from "../utils/theme";

interface SettingsViewProps {
  settings: Settings | null;
  settingsError: string | null;
  selfModel: SelfModel | null;
  selfInspection: SelfInspection | null;
  systemControls?: SystemControlEntry[];
  systemControlError?: string | null;
  onRefreshSystemControls?: () => void;
  showRaw: boolean;
  testResult: TestResult | null;
  onUpdateSettings: (settings: Settings) => void;
  onToggleShowRaw: (value: boolean) => void;
  onTestConnection: () => void;
  onSaveSettings: () => void;
  onWipeMemory: () => void;
  onResetConversationData: () => void;
  onSetReflectionFrozen: (frozen: boolean) => void;
  onRefreshSelfData: () => void;
  onRetrySettings: () => void;
}

const formatJson = (value: unknown) => {
  if (value === null || value === undefined) return "Not available.";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return "Not available.";
  }
};

const DEFAULT_VOICE_NAME = "bf_isabella";
const DEFAULT_VOICE_PITCH_SEMITONES = 1;
const DEFAULT_VOICE_REVERB_AMOUNT = 0.15;
const DEFAULT_VOICE_COMPRESSION = 0.05;

type SettingsSectionProps = {
  title: string;
  defaultOpen?: boolean;
  variant?: "danger";
  children: ReactNode;
};

const SettingsSection = ({ title, defaultOpen = false, variant, children }: SettingsSectionProps) => (
  <details
    className={`settings-section settings-section-collapsible${variant === "danger" ? " settings-section-danger" : ""}`}
    open={defaultOpen}
  >
    <summary className={`settings-section-summary${variant === "danger" ? " settings-danger-title" : ""}`}>
      <span>{title}</span>
      <span className="settings-section-chevron" aria-hidden="true">▾</span>
    </summary>
    <div className="settings-section-body">
      {children}
    </div>
  </details>
);

export const SettingsView = ({
  settings,
  settingsError,
  selfModel,
  selfInspection,
  systemControls = [],
  systemControlError,
  onRefreshSystemControls,
  showRaw,
  testResult,
  onUpdateSettings,
  onToggleShowRaw,
  onTestConnection,
  onSaveSettings,
  onWipeMemory,
  onResetConversationData,
  onSetReflectionFrozen,
  onRefreshSelfData,
  onRetrySettings,
}: SettingsViewProps) => {
  const isDev = import.meta.env.DEV;
  const [debugFlags, setDebugFlagsState] = useState<UiDebugFlags>(getUiDebugFlags());
  const [themeOptions, setThemeOptions] = useState<ThemeOption[]>([
    { id: DEFAULT_THEME_ID, label: "Default", source: "builtin", name: "default" },
  ]);
  const [themeDir, setThemeDir] = useState<string | null>(null);
  const [registryPayload, setRegistryPayload] = useState<string>("");
  const [registryMeta, setRegistryMeta] = useState<ParameterRegistry | null>(null);
  const [registryError, setRegistryError] = useState<string | null>(null);
  const [registryDirty, setRegistryDirty] = useState(false);
  const [registryLoading, setRegistryLoading] = useState(false);
  const [registryStatus, setRegistryStatus] = useState<string | null>(null);
  const [promptStatus, setPromptStatus] = useState<PromptStatus | null>(null);
  const [promptStatusError, setPromptStatusError] = useState<string | null>(null);
  const [promptStatusLoading, setPromptStatusLoading] = useState(false);
  const [promptTrimStats, setPromptTrimStats] = useState<Record<string, number> | null>(null);
  const [promptTrimCriticalCount, setPromptTrimCriticalCount] = useState(0);
  const [promptOverflowCount, setPromptOverflowCount] = useState(0);
  const [promptTrimError, setPromptTrimError] = useState<string | null>(null);
  const [promptTrimLoading, setPromptTrimLoading] = useState(false);
  const [waveControlReason, setWaveControlReason] = useState("");
  const [waveControlError, setWaveControlError] = useState<string | null>(null);
  const [waveStatus, setWaveStatus] = useState<WaveStatus | null>(null);
  const [waveStatusError, setWaveStatusError] = useState<string | null>(null);
  const [waveStatusLoading, setWaveStatusLoading] = useState(false);
  const [phiScopeEnabled, setPhiScopeEnabled] = useState<boolean | null>(null);
  const [phiScopeError, setPhiScopeError] = useState<string | null>(null);
  const [phiScopeLoading, setPhiScopeLoading] = useState(false);
  const phiScopeConversationId = "default";

  useEffect(() => {
    setDebugFlagsState(getUiDebugFlags());
  }, []);

  useEffect(() => {
    let active = true;
    if (!settings) return;
    setPhiScopeLoading(true);
    setPhiScopeError(null);
    invokeWithTimeout<boolean | null>("get_phi_consent_scope", { conversationId: phiScopeConversationId })
      .then((enabled) => {
        if (!active) return;
        setPhiScopeEnabled(enabled ?? null);
      })
      .catch((e) => {
        if (!active) return;
        setPhiScopeError(String(e));
      })
      .finally(() => {
        if (!active) return;
        setPhiScopeLoading(false);
      });
    return () => {
      active = false;
    };
  }, [settings]);

  useEffect(() => {
    let active = true;
    listThemeOptions()
      .then(({ options, dir }) => {
        if (!active) return;
        setThemeOptions(options);
        setThemeDir(dir);
      })
      .catch(() => {
        if (!active) return;
        setThemeOptions([{ id: DEFAULT_THEME_ID, label: "Default", source: "builtin", name: "default" }]);
      });
    return () => {
      active = false;
    };
  }, []);

  const formatRegistryPayload = (payload: string) => {
    if (!payload.trim()) return "";
    try {
      return JSON.stringify(JSON.parse(payload), null, 2);
    } catch {
      return payload;
    }
  };

  const loadRegistry = async (profile: string) => {
    setRegistryLoading(true);
    setRegistryError(null);
    setRegistryStatus(null);
    try {
      const registry = await invokeWithTimeout<ParameterRegistry>("get_parameter_registry", {
        profileName: profile,
      });
      setRegistryMeta(registry);
      setRegistryPayload(formatRegistryPayload(registry.payload_json));
      setRegistryDirty(false);
    } catch (e) {
      setRegistryError(String(e));
      setRegistryMeta(null);
    } finally {
      setRegistryLoading(false);
    }
  };

  const updatePhiScopeConsent = async (enabled: boolean) => {
    setPhiScopeLoading(true);
    setPhiScopeError(null);
    try {
      await invokeWithTimeout("set_phi_consent_scope", {
        conversationId: phiScopeConversationId,
        enabled,
      });
      setPhiScopeEnabled(enabled);
    } catch (e) {
      setPhiScopeError(String(e));
    } finally {
      setPhiScopeLoading(false);
    }
  };

  const loadPromptStatus = async () => {
    setPromptStatusLoading(true);
    setPromptStatusError(null);
    try {
      const status = await invokeWithTimeout<PromptStatus>("get_prompt_status");
      setPromptStatus(status);
    } catch (e) {
      setPromptStatusError(String(e));
      setPromptStatus(null);
    } finally {
      setPromptStatusLoading(false);
    }
  };

  const loadPromptTrimStats = async () => {
    setPromptTrimLoading(true);
    setPromptTrimError(null);
    try {
      const logs = await invokeWithTimeout<SystemLogEntry[]>("get_system_logs", {
        limit: 200,
        category: "kernel",
      });
      const counts: Record<string, number> = {};
      let criticalCount = 0;
      let overflowCount = 0;
      for (const log of logs) {
        const payload = log.payload as any;
        const event = payload?.event;
        if (event === "prompt_trim") {
          const title = String(payload?.title || "unknown");
          counts[title] = (counts[title] || 0) + 1;
        } else if (event === "prompt_trim_critical") {
          criticalCount += 1;
        } else if (event === "prompt_overflow") {
          overflowCount += 1;
        }
      }
      setPromptTrimStats(counts);
      setPromptTrimCriticalCount(criticalCount);
      setPromptOverflowCount(overflowCount);
    } catch (e) {
      setPromptTrimError(String(e));
      setPromptTrimStats(null);
      setPromptTrimCriticalCount(0);
      setPromptOverflowCount(0);
    } finally {
      setPromptTrimLoading(false);
    }
  };

  const loadWaveStatus = async () => {
    setWaveStatusLoading(true);
    setWaveStatusError(null);
    try {
      const status = await invokeWithTimeout<WaveStatus>("get_wave_status");
      setWaveStatus(status);
    } catch (e) {
      setWaveStatusError(String(e));
      setWaveStatus(null);
    } finally {
      setWaveStatusLoading(false);
    }
  };

  const handleResetPrompt = async () => {
    if (!settings) return;
    const updated = { ...settings, system_prompt: null };
    onUpdateSettings(updated);
    try {
      await invokeWithTimeout("update_settings", { settings: updated });
      await loadPromptStatus();
    } catch (e) {
      setPromptStatusError(String(e));
    }
  };

  const saveRegistry = async (profile: string) => {
    setRegistryLoading(true);
    setRegistryError(null);
    setRegistryStatus(null);
    try {
      const parsed = registryPayload.trim()
        ? JSON.parse(registryPayload)
        : {};
      const payloadJson = JSON.stringify(parsed, null, 2);
      const registry = await invokeWithTimeout<ParameterRegistry>("update_parameter_registry", {
        profileName: profile,
        payloadJson,
      });
      setRegistryMeta(registry);
      setRegistryPayload(formatRegistryPayload(registry.payload_json));
      setRegistryDirty(false);
      setRegistryStatus("Registry saved.");
    } catch (e) {
      setRegistryError(String(e));
    } finally {
      setRegistryLoading(false);
    }
  };

  const updateDebugFlags = (patch: Partial<UiDebugFlags>) => {
    const next = { ...debugFlags, ...patch };
    setDebugFlagsState(next);
    setUiDebugFlags(next);
  };

  useEffect(() => {
    if (!settings) return;
    const profile = settings.registry_profile_name || "default";
    void loadRegistry(profile);
    void loadPromptStatus();
    void loadPromptTrimStats();
  }, [settings?.registry_profile_name]);

  useEffect(() => {
    if (!settings) return;
    void loadPromptStatus();
    void loadPromptTrimStats();
  }, [settings?.system_prompt, settings?.user_display_name, settings?.assistant_display_name]);

  const renderError = settingsError ? (
    <div
      className="glass settings-error-banner"
    >
      <span>{settingsError}</span>
      <button className="btn btn-secondary" onClick={onRetrySettings}>
        Retry
      </button>
    </div>
  ) : null;

  if (!settings) {
    return (
      <div className="settings-container">
        <h1>Settings</h1>
        {renderError || <div className="settings-hint">Loading settings...</div>}
      </div>
    );
  }

  const episodicEnabled = Boolean(settings.episodic_enabled);
  const episodicInjectionEnabled = Boolean(settings.episodic_injection_enabled);
  const episodicCompactionEnabled = Boolean(settings.episodic_compaction_enabled);
  const episodicInjectionLimit = settings.episodic_injection_limit ?? 5;
  const memoryClaimsEnabled = Boolean(settings.memory_claims_enabled);
  const selectedTheme = settings.ui_theme ?? DEFAULT_THEME_ID;
  const voiceName = settings.voice_name ?? DEFAULT_VOICE_NAME;
  const voicePitch = settings.voice_pitch_semitones ?? DEFAULT_VOICE_PITCH_SEMITONES;
  const voiceReverb = settings.voice_reverb_amount ?? DEFAULT_VOICE_REVERB_AMOUNT;
  const voiceCompression = settings.voice_compression ?? DEFAULT_VOICE_COMPRESSION;
  const resolvedThemeOptions = themeOptions.some((theme) => theme.id === selectedTheme)
    ? themeOptions
    : [
        { id: selectedTheme, label: `${selectedTheme} (missing)`, source: "user", name: selectedTheme },
        ...themeOptions,
      ];
  const lastUserMemoryWrite = selfInspection?.last_user_memory_write || "Not available.";
  const lastSelfMemoryWrite = selfInspection?.last_self_memory_write || "Not available.";
  const lastMemoryErrorAt = selfInspection?.last_memory_error_at || "None recorded.";
  const reflectionStatus = (selfModel?.reflection_status as Record<string, unknown> | null) || null;
  const reflectionStatusLabel = (reflectionStatus?.status as string) || "unknown";
  const reflectionAllowlistCount =
    typeof reflectionStatus?.allowlist_count === "number" ? reflectionStatus?.allowlist_count : "n/a";
  const lastReflectionAt = selfModel?.last_reflection_at || "None";
  const registryProfile = settings.registry_profile_name || "default";
  const cockpitWriteEnabled = settings.cockpit_write_enabled ?? false;
  const waveFieldMode = systemControls.find((entry) => entry.subsystem_id === "cognitive_wave")?.mode ?? "off";
  const waveProjectionMode = systemControls.find((entry) => entry.subsystem_id === "cognitive_wave_projection")?.mode ?? "off";
  const qualiaAutoMode = systemControls.find((entry) => entry.subsystem_id === "qualia_auto")?.mode ?? "off";
  const waveFieldEnabled = waveFieldMode.toLowerCase() !== "off";
  const waveProjectionEnabled = waveProjectionMode.toLowerCase() !== "off";
  const qualiaAutoEnabled = qualiaAutoMode.toLowerCase() !== "off";
  const waveContributionAge = waveStatus?.contribution_age_seconds ?? null;
  const waveProjectionAge = waveStatus?.projection_age_seconds ?? null;
  const waveContributionStale = waveProjectionEnabled
    && (waveContributionAge === null || waveContributionAge > 600);

  useEffect(() => {
    void loadWaveStatus();
  }, [waveFieldMode, waveProjectionMode]);

  const applyWaveControl = async (subsystemId: string, nextEnabled: boolean) => {
    if (!cockpitWriteEnabled) {
      setWaveControlError("Cockpit write mode is disabled.");
      return;
    }
    try {
      setWaveControlError(null);
      const reason = waveControlReason.trim() || "settings_wave_toggle";
      await invokeWithTimeout(
        "set_system_control",
        {
          subsystemId,
          mode: nextEnabled ? "normal" : "off",
          valueJson: null,
          reason,
          overrideCritical: false,
          actor: "settings",
        },
        15000,
      );
      onRefreshSystemControls?.();
      void loadWaveStatus();
    } catch (e) {
      setWaveControlError(String(e));
    }
  };

  return (
    <div className="settings-container">
      <h1>Settings</h1>
      {renderError}
      <SettingsSection title="Core & Identity" defaultOpen>
        <div className="settings-group">
          <label className="settings-label">API Base URL</label>
          <input
            className="input"
            value={settings.api_base_url}
            onChange={(e) => onUpdateSettings({ ...settings, api_base_url: e.target.value })}
            placeholder="http://localhost:11434/v1"
          />
          <div className="settings-hint">OpenAI-compatible endpoint (e.g. llama.cpp, Ollama)</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">API Key (Optional)</label>
          <input
            className="input"
            type="password"
            value={settings.api_key || ""}
            onChange={(e) => onUpdateSettings({ ...settings, api_key: e.target.value || null })}
          />
        </div>

        <div className="settings-group">
          <label className="settings-label">System Prompt (Priming)</label>
          <textarea
            className="input"
            rows={4}
            style={{ resize: "vertical", minHeight: "100px" }}
            value={settings.system_prompt || ""}
            onChange={(e) => onUpdateSettings({ ...settings, system_prompt: e.target.value || null })}
            placeholder="e.g. You are a helpful AI assistant that specializes in Rust programming."
          />
          <div className="settings-hint">This is sent as the first 'system' message in every request.</div>
          <div className="settings-hint" style={{ marginTop: "6px" }}>
            {promptStatusLoading && "Checking active prompt..."}
            {!promptStatusLoading && promptStatus && (
              <>
                <div>Prompt source: {promptStatus.prompt_source}</div>
                <div>Primary hash: {promptStatus.primary_prompt_hash}</div>
                <div>Memory hash: {promptStatus.memory_prompt_hash}</div>
                {promptStatus.override_active && (
                  <div>Override hash: {promptStatus.override_hash || "unknown"}</div>
                )}
                {promptStatus.override_mismatch && (
                  <div className="text-error">
                    Override does not match canonical prompt. Guarding internal sections.
                  </div>
                )}
              </>
            )}
            {!promptStatusLoading && promptStatusError && (
              <div className="text-error">{promptStatusError}</div>
            )}
          </div>
          <div style={{ marginTop: "8px" }}>
            <button className="btn btn-secondary" onClick={handleResetPrompt}>
              Reset To Canonical
            </button>
          </div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Prompt Trim Dashboard</label>
          <div className="settings-hint">Shows recent prompt trim pressure (last 200 kernel logs).</div>
          {promptTrimLoading && <div className="settings-hint">Loading trim stats...</div>}
          {!promptTrimLoading && promptTrimError && (
            <div className="text-error">{promptTrimError}</div>
          )}
          {!promptTrimLoading && !promptTrimError && (
            <div className="settings-hint" style={{ marginTop: "6px" }}>
              <div>Critical trims: {promptTrimCriticalCount}</div>
              <div>Prompt overflows: {promptOverflowCount}</div>
              {promptTrimStats && Object.keys(promptTrimStats).length > 0 ? (
                Object.entries(promptTrimStats).map(([title, count]) => (
                  <div key={title}>{title}: {count}</div>
                ))
              ) : (
                <div>No prompt trim events in recent logs.</div>
              )}
            </div>
          )}
        </div>

        <div className="settings-group">
          <label className="settings-label">User Display Name</label>
          <input
            className="input"
            value={settings.user_display_name || ""}
            onChange={(e) => onUpdateSettings({ ...settings, user_display_name: e.target.value || null })}
            placeholder="User"
          />
          <div className="settings-hint">Sets the canonical user entity label and prompt injection.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Assistant Display Name</label>
          <input
            className="input"
            value={settings.assistant_display_name || ""}
            onChange={(e) => onUpdateSettings({ ...settings, assistant_display_name: e.target.value || null })}
            placeholder="Ergo"
          />
          <div className="settings-hint">Sets the canonical assistant entity label and prompt injection.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Theme</label>
          <select
            className="input"
            value={selectedTheme}
            onChange={(e) => onUpdateSettings({ ...settings, ui_theme: e.target.value })}
          >
            {resolvedThemeOptions.map((theme) => (
              <option key={theme.id} value={theme.id}>
                {theme.label}
              </option>
            ))}
          </select>
          <div className="settings-hint">
            Add custom themes as single .css files in {themeDir || "the app themes folder"}. Save settings to persist.
          </div>
        </div>

        <div className="settings-group" style={{ flexDirection: "row", alignItems: "center", gap: "12px" }}>
          <input
            type="checkbox"
            checked={settings.streaming_enabled}
            onChange={(e) => onUpdateSettings({ ...settings, streaming_enabled: e.target.checked })}
            id="streaming-toggle"
          />
          <label htmlFor="streaming-toggle" className="settings-label">Enable Streaming</label>
        </div>

        <div className="settings-group">
          <label className="settings-label">History Window (Prior Messages)</label>
          <input
            className="input"
            type="number"
            min="0"
            max="50"
            value={settings.history_window}
            onChange={(e) => onUpdateSettings({ ...settings, history_window: parseInt(e.target.value) || 0 })}
          />
        </div>

        <div className="settings-group">
          <label className="settings-label">Injection Policy</label>
          <select
            className="input"
            value={settings.injection_policy}
            onChange={(e) => onUpdateSettings({ ...settings, injection_policy: e.target.value as any })}
          >
            <option value="include">Include system/developer messages</option>
            <option value="exclude">Exclude injected messages</option>
          </select>
        </div>
      </SettingsSection>

      <SettingsSection title="Diagnostics & Cockpit">
        <div className="settings-group settings-group-border">
          <input
            type="checkbox"
            checked={showRaw}
            onChange={(e) => onToggleShowRaw(e.target.checked)}
            id="raw-mode-toggle"
          />
          <label
            htmlFor="raw-mode-toggle"
            className="settings-label settings-toggle"
            data-active={showRaw ? "true" : "false"}
          >
            Debug: Show Raw Output (No Filtering)
          </label>
        </div>

        <div className="settings-group">
          <label className="settings-label">System Log Limit: {settings.trace_history_limit || 10}</label>
          <input
            type="range"
            min="5"
            max="50"
            step="5"
            value={settings.trace_history_limit || 10}
            onChange={(e) => onUpdateSettings({ ...settings, trace_history_limit: parseInt(e.target.value) })}
            style={{ width: "100%" }}
          />
          <div className="settings-hint">Number of history events to preserve in SymbioteEnvelope (5-50).</div>
        </div>

        <div className="settings-group settings-group-border">
          <input
            type="checkbox"
            checked={settings.cockpit_write_enabled ?? false}
            onChange={(e) => onUpdateSettings({ ...settings, cockpit_write_enabled: e.target.checked })}
            id="cockpit-write-toggle"
          />
          <label
            htmlFor="cockpit-write-toggle"
            className="settings-label settings-toggle"
            data-active={settings.cockpit_write_enabled ? "true" : "false"}
          >
            Cockpit Write Mode (Allow control changes)
          </label>
          <div className="settings-hint">
            When disabled, the cockpit runs read-only and control changes are blocked.
          </div>
        </div>
      </SettingsSection>

      <SettingsSection title="Cognitive Wave">

        <div className="settings-group">
          <label className="settings-label">Control Change Reason (Optional)</label>
          <input
            className="input"
            value={waveControlReason}
            onChange={(e) => setWaveControlReason(e.target.value)}
            placeholder="settings_wave_toggle"
          />
          <div className="settings-hint">
            Used for system control audit logs. Default: <code>settings_wave_toggle</code>.
          </div>
        </div>

        <div className="settings-group settings-group-border">
          <label className="settings-label">Cognitive Wave Field</label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={waveFieldEnabled}
              onChange={(e) => applyWaveControl("cognitive_wave", e.target.checked)}
              disabled={!cockpitWriteEnabled}
            />
            <span>{waveFieldEnabled ? "Enabled" : "Disabled"}</span>
          </label>
          <div className="settings-hint">Enables Fourier field contributions and decay.</div>
        </div>

        <div className="settings-group settings-group-border">
          <label className="settings-label">Cognitive Wave Projection</label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={waveProjectionEnabled}
              onChange={(e) => applyWaveControl("cognitive_wave_projection", e.target.checked)}
              disabled={!cockpitWriteEnabled || !waveFieldEnabled}
            />
            <span>{waveProjectionEnabled ? "Enabled" : "Disabled"}</span>
          </label>
          <div className="settings-hint">
            Enables projection metrics for prompts and arbitration. Requires the wave field.
          </div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Wave Status</label>
          {waveStatusLoading && <div className="settings-hint">Loading wave status...</div>}
          {!waveStatusLoading && waveStatus && (
            <div className="settings-hint">
              <div>Coherence: {waveStatus.coherence?.toFixed(3) ?? "n/a"}</div>
              <div>Dominance: {waveStatus.dominance?.toFixed(3) ?? "n/a"}</div>
              <div>Turbulence: {waveStatus.turbulence?.toFixed(3) ?? "n/a"}</div>
              <div>Drift: {waveStatus.drift?.toFixed(3) ?? "n/a"}</div>
              <div>Total Energy: {waveStatus.total_energy?.toFixed(3) ?? "n/a"}</div>
              <div>Last Projection: {waveStatus.last_projection_at ?? "None"}</div>
              <div>Last Contribution: {waveStatus.last_contribution_at ?? "None"}</div>
              {waveProjectionAge !== null && (
                <div>Projection age: {waveProjectionAge}s</div>
              )}
              {waveContributionAge !== null && (
                <div>Contribution age: {waveContributionAge}s</div>
              )}
            </div>
          )}
          {!waveStatusLoading && !waveStatus && (
            <div className="settings-hint">Wave status unavailable.</div>
          )}
          {waveContributionStale && (
            <div className="settings-hint" style={{ color: "var(--danger, #d9534f)" }}>
              Projection enabled, but no recent contributions. Wave metrics may be stale.
            </div>
          )}
          {waveStatusError && (
            <div className="settings-hint" style={{ color: "var(--danger, #d9534f)" }}>
              {waveStatusError}
            </div>
          )}
          <button className="btn btn-secondary" onClick={loadWaveStatus}>
            Refresh Wave Status
          </button>
        </div>

        {(waveControlError || systemControlError) && (
          <div className="settings-hint" style={{ color: "var(--danger, #d9534f)" }}>
            {waveControlError || systemControlError}
          </div>
        )}
      </SettingsSection>

      <SettingsSection title="Qualia">

        <div className="settings-group settings-group-border">
          <label className="settings-label">Qualia Auto-label</label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={qualiaAutoEnabled}
              onChange={(e) => applyWaveControl("qualia_auto", e.target.checked)}
              disabled={!cockpitWriteEnabled}
            />
            <span>{qualiaAutoEnabled ? "Enabled" : "Disabled"}</span>
          </label>
          <div className="settings-hint">
            Auto-labels neutral qualia when no recent user labels exist.
          </div>
        </div>
      </SettingsSection>

      <SettingsSection title="Kernel & Stability">

        <div className="settings-group">
          <label className="settings-label">Follow-up Questions (Max)</label>
          <input
            className="input"
            type="number"
            min="0"
            max="5"
            value={settings.ask_budget_max ?? 1}
            onChange={(e) => onUpdateSettings({ ...settings, ask_budget_max: parseInt(e.target.value, 10) || 0 })}
          />
          <div className="settings-hint">How many clarifying questions the system can ask before proceeding.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Calculator Follow-ups (Max)</label>
          <input
            className="input"
            type="number"
            min="0"
            max="2"
            value={settings.calculator_followups_max ?? 0}
            onChange={(e) => onUpdateSettings({ ...settings, calculator_followups_max: parseInt(e.target.value, 10) || 0 })}
          />
          <div className="settings-hint">Clarifying questions allowed while in calculator mode.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Repeat Question Sensitivity</label>
          <input
            className="input"
            type="number"
            min="0.5"
            max="0.99"
            step="0.01"
            value={settings.loop_similarity_threshold ?? 0.85}
            onChange={(e) => onUpdateSettings({ ...settings, loop_similarity_threshold: parseFloat(e.target.value) })}
          />
          <div className="settings-hint">Higher = more likely to block repeating similar questions.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Repeat History Window</label>
          <input
            className="input"
            type="number"
            min="1"
            max="12"
            value={settings.loop_recent_k ?? 6}
            onChange={(e) => onUpdateSettings({ ...settings, loop_recent_k: parseInt(e.target.value, 10) || 1 })}
          />
          <div className="settings-hint">How many recent asks are checked for repeats.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Meta-cog Outcome Turns</label>
          <input
            className="input"
            type="number"
            min="1"
            max="10"
            value={settings.meta_cog_outcome_turns ?? 3}
            onChange={(e) => onUpdateSettings({ ...settings, meta_cog_outcome_turns: parseInt(e.target.value, 10) || 1 })}
          />
          <div className="settings-hint">Turns before evaluating a meta-cog action outcome.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Meta-cog Cycle Window</label>
          <input
            className="input"
            type="number"
            min="1"
            max="6"
            value={settings.meta_cog_cycle_window_turns ?? 2}
            onChange={(e) => onUpdateSettings({ ...settings, meta_cog_cycle_window_turns: parseInt(e.target.value, 10) || 1 })}
          />
          <div className="settings-hint">Turns used to detect repeated meta-cog cycles.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Meta-cog Outcome Timeout (s)</label>
          <input
            className="input"
            type="number"
            min="30"
            max="600"
            step="10"
            value={settings.meta_cog_outcome_timeout_s ?? 120}
            onChange={(e) => onUpdateSettings({ ...settings, meta_cog_outcome_timeout_s: parseInt(e.target.value, 10) || 120 })}
          />
          <div className="settings-hint">Maximum seconds to wait before scoring an outcome.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Meta-cog Cooldown (s)</label>
          <input
            className="input"
            type="number"
            min="10"
            max="600"
            step="10"
            value={settings.meta_cog_cooldown_s ?? 60}
            onChange={(e) => onUpdateSettings({ ...settings, meta_cog_cooldown_s: parseInt(e.target.value, 10) || 60 })}
          />
          <div className="settings-hint">Cooldown before new meta-cog actions after cycling.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Meta-cog Streak Limit</label>
          <input
            className="input"
            type="number"
            min="1"
            max="10"
            value={settings.meta_cog_streak_limit ?? 3}
            onChange={(e) => onUpdateSettings({ ...settings, meta_cog_streak_limit: parseInt(e.target.value, 10) || 1 })}
          />
          <div className="settings-hint">How many repeated outcomes trigger a cooldown.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Adaptive Controller</label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.controller_enabled ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, controller_enabled: e.target.checked })}
            />
            <span>Use telemetry-based gating and throttles</span>
          </label>
        </div>

        <div className="settings-group">
          <label className="settings-label">Monologue Loop Guard</label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.monologue_stabilization_enabled ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, monologue_stabilization_enabled: e.target.checked })}
            />
            <span>Reduce runaway internal loops</span>
          </label>
        </div>

        <div className="settings-group">
          <label className="settings-label">Internal Introspection</label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.enable_introspection ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, enable_introspection: e.target.checked })}
            />
            <span>Allow internal reflection and relaxed monologue gating</span>
          </label>
          <div className="settings-hint">Lets monologue turns persist even when anchors are weak.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Allow System-Originated Messages</label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.monologue_surface_enabled ?? false}
              onChange={(e) => onUpdateSettings({ ...settings, monologue_surface_enabled: e.target.checked })}
            />
            <span>Allow internal monologue to surface when you ask</span>
          </label>
          <div className="settings-hint">Does not auto-message; requires an explicit request to surface.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Show Monologue In Chat</label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.show_monologue_in_chat ?? false}
              onChange={(e) => onUpdateSettings({ ...settings, show_monologue_in_chat: e.target.checked })}
            />
            <span>Display internal monologue messages in the main chat</span>
          </label>
          <div className="settings-hint">Shows internal entries inline; keep off for a cleaner user-facing view.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Monologue Safety Filter</label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.enable_monologue_validator ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, enable_monologue_validator: e.target.checked })}
            />
            <span>Block confusing or user-addressed monologue</span>
          </label>
          <div className="settings-hint">Prevents greetings, direct user address, and self-disclaimer boilerplate in monologue.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Heartbeat</label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.heartbeat_enabled ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, heartbeat_enabled: e.target.checked })}
            />
            <span>Run periodic controller refresh</span>
          </label>
        </div>

        <div className="settings-group">
          <label className="settings-label">Dream Consolidation</label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.dream_enabled ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, dream_enabled: e.target.checked })}
            />
            <span>Run background memory consolidation</span>
          </label>
        </div>

        <div className="settings-group">
          <label className="settings-label">Require Workspace Anchors</label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.binding_enforcement_enabled ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, binding_enforcement_enabled: e.target.checked })}
            />
            <span>Require workspace-bound responses</span>
          </label>
          <div className="settings-hint">Forces workspace updates on monologue ticks and references in user-visible replies.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Surface Queued Prompts Only When Relevant</label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.pending_prompt_alignment_enabled ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, pending_prompt_alignment_enabled: e.target.checked })}
            />
            <span>Hold or discard queued prompts that do not match current focus</span>
          </label>
          <div className="settings-hint">If off, queued prompts can surface even when focus has drifted.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Auto Memory Writes</label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.auto_memory_pass_enabled ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, auto_memory_pass_enabled: e.target.checked })}
            />
            <span>Trigger memory writes without explicit tags</span>
          </label>
          <div className="settings-hint">Runs memory pass automatically when high-confidence candidates are detected.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Align Summaries To Workspace</label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.summary_cohesion_enabled ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, summary_cohesion_enabled: e.target.checked })}
            />
            <span>Anchor summaries to current workspace focus</span>
          </label>
          <div className="settings-hint">Keeps inner, monologue, and rolling summaries aligned with workspace state.</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Compact Prompt Mode</label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.compact_prompt_enabled ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, compact_prompt_enabled: e.target.checked })}
            />
            <span>Use smaller prompts on low-stakes turns</span>
          </label>
          <div className="settings-hint">Reduces prompt size on low-stakes turns without skipping cognition.</div>
        </div>

        <div className="settings-group" style={{ marginTop: "16px", borderTop: "1px solid var(--border-color)", paddingTop: "12px" }}>
          <label className="settings-label">Stability Flags</label>
          <div className="settings-hint">Feature flags for Stability Restored. Default on.</div>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.stability_prompt_override_guard ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, stability_prompt_override_guard: e.target.checked })}
            />
            <span>Guard internal sections when prompt override mismatches canonical</span>
          </label>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.stability_monologue_tagged ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, stability_monologue_tagged: e.target.checked })}
            />
            <span>Dialectic monologue with tagged stances (no named speakers)</span>
          </label>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.stability_introspection_structured ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, stability_introspection_structured: e.target.checked })}
            />
            <span>Structured, gated introspection blocks</span>
          </label>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.stability_disable_working_hypothesis ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, stability_disable_working_hypothesis: e.target.checked })}
            />
            <span>Disable "Working hypothesis" phrasing in prompts and output</span>
          </label>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.stability_state_disclosure_expanded ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, stability_state_disclosure_expanded: e.target.checked })}
            />
            <span>Expanded state disclosure detection</span>
          </label>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.stability_transcript_normalization ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, stability_transcript_normalization: e.target.checked })}
            />
            <span>Normalize transcript-style user input</span>
          </label>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.stability_memory_hygiene ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, stability_memory_hygiene: e.target.checked })}
            />
            <span>Block internal telemetry from entering general memory</span>
          </label>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.stability_non_stream_sanitization ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, stability_non_stream_sanitization: e.target.checked })}
            />
            <span>Sanitize non-streaming output</span>
          </label>
        </div>

        <div className="settings-group">
          <label className="settings-label">Registry Profile Name</label>
          <input
            className="input"
            value={settings.registry_profile_name || "default"}
            onChange={(e) => onUpdateSettings({ ...settings, registry_profile_name: e.target.value || "default" })}
          />
          <div className="settings-hint">Select the parameter registry profile used for slot defaults.</div>
        </div>
      </SettingsSection>

      <SettingsSection title="Gating & Feedback">

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.gate_default_soft ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, gate_default_soft: e.target.checked })}
            />
            <span>Default to graded gating (ALLOW_WITH_NOTICE)</span>
          </label>
          <div className="settings-hint">
            When enabled, novel or uncertain inputs respond with a notice instead of hard blocking.
          </div>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.gate_shadow_mode ?? false}
              onChange={(e) => onUpdateSettings({ ...settings, gate_shadow_mode: e.target.checked })}
            />
            <span>Shadow mode (compute soft gate but do not enforce)</span>
          </label>
          <div className="settings-hint">
            Logs soft-gate outcomes for comparison without changing live behavior.
          </div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Gate Rollout Percent: {settings.gate_rollout_percent ?? 100}%</label>
          <input
            type="range"
            min="0"
            max="100"
            step="5"
            value={settings.gate_rollout_percent ?? 100}
            onChange={(e) => onUpdateSettings({ ...settings, gate_rollout_percent: parseInt(e.target.value, 10) })}
            style={{ width: "100%" }}
          />
          <div className="settings-hint">
            Percent of sessions that enforce the soft gate when shadow mode is off.
          </div>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.self_report_channel ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, self_report_channel: e.target.checked })}
            />
            <span>Allow provisional self-report channel</span>
          </label>
          <div className="settings-hint">
            Allows self-report outputs without persisting memory when evidence is missing.
          </div>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.explicit_feedback_only ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, explicit_feedback_only: e.target.checked })}
            />
            <span>Require explicit feedback markers</span>
          </label>
          <div className="settings-hint">
            Only messages marked as feedback will influence alignment signals.
          </div>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.monologue_provenance_guard ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, monologue_provenance_guard: e.target.checked })}
            />
            <span>Enforce monologue provenance guard</span>
          </label>
          <div className="settings-hint">
            Strips user-attribution from monologue entries when unsupported by last user input.
          </div>
        </div>

        <div className="settings-group">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.organism_decay ?? true}
              onChange={(e) => onUpdateSettings({ ...settings, organism_decay: e.target.checked })}
            />
            <span>Smooth organism signals (EMA decay)</span>
          </label>
          <div className="settings-hint">
            Applies decay to fatigue and social alignment to prevent thrashing.
          </div>
        </div>
      </SettingsSection>

      <SettingsSection title="Parameter Registry">
        <div className="settings-group">
          <label className="settings-label">Active Profile</label>
          <div className="settings-hint">
            {registryMeta
              ? `${registryMeta.profile_name} (v${registryMeta.profile_version}) updated ${registryMeta.updated_at}`
              : `Using profile "${registryProfile}".`}
          </div>
        </div>
        <div className="settings-group">
          <label className="settings-label">Registry JSON</label>
          <textarea
            className="input settings-textarea-mono"
            rows={8}
            value={registryPayload}
            onChange={(e) => {
              setRegistryPayload(e.target.value);
              setRegistryDirty(true);
              setRegistryStatus(null);
            }}
            placeholder='{"meta_states":{"CALM":{"w":0.82,"pe":0.68,"var":0.35}}}'
          />
          <div className="settings-hint">Structured defaults for slot resolution. Must be valid JSON.</div>
        </div>
        {registryError && (
          <div className="settings-hint text-error">{registryError}</div>
        )}
        {registryStatus && (
          <div className="settings-hint text-success">{registryStatus}</div>
        )}
        <div className="settings-group settings-group-inline">
          <button
            className="btn btn-secondary"
            onClick={() => loadRegistry(registryProfile)}
            disabled={registryLoading}
          >
            {registryLoading ? "Loading..." : "Reload Registry"}
          </button>
          <button
            className="btn btn-primary"
            onClick={() => saveRegistry(registryProfile)}
            disabled={registryLoading || !registryDirty}
          >
            Save Registry
          </button>
        </div>
      </SettingsSection>

      <SettingsSection title="Voice">

        <div className="settings-group">
          <label className="settings-label">TTS Voice</label>
          <select
            className="input"
            value={voiceName}
            onChange={(e) => onUpdateSettings({ ...settings, voice_name: e.target.value })}
          >
            <optgroup label="American English (Female)">
              <option value="af_bella">Bella</option>
              <option value="af_nicole">Nicole</option>
              <option value="af_sarah">Sarah</option>
              <option value="af_sky">Sky</option>
              <option value="af_alloy">Alloy</option>
              <option value="af_heart">Heart</option>
            </optgroup>
            <optgroup label="American English (Male)">
              <option value="am_adam">Adam</option>
              <option value="am_michael">Michael</option>
            </optgroup>
            <optgroup label="British English (Female)">
              <option value="bf_emma">Emma</option>
              <option value="bf_isabella">Isabella</option>
            </optgroup>
            <optgroup label="British English (Male)">
              <option value="bm_george">George</option>
              <option value="bm_lewis">Lewis</option>
            </optgroup>
          </select>
        </div>

        <div className="settings-group">
          <label className="settings-label">Speech Speed: {(settings.voice_speed || 1.0).toFixed(2)}x</label>
          <input
            type="range"
            min="0.5"
            max="2.0"
            step="0.1"
            value={settings.voice_speed || 1.0}
            onChange={(e) => onUpdateSettings({ ...settings, voice_speed: parseFloat(e.target.value) })}
            style={{ width: "100%" }}
          />
          <div className="settings-hint">Adjust speech rate (0.5x = slower, 2.0x = faster)</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Pitch Shift: {voicePitch} semitones</label>
          <input
            type="range"
            min="-12"
            max="12"
            step="1"
            value={voicePitch}
            onChange={(e) => onUpdateSettings({ ...settings, voice_pitch_semitones: parseFloat(e.target.value) })}
            style={{ width: "100%" }}
          />
        </div>

        <div className="settings-group">
          <label className="settings-label">Reverb Amount: {(voiceReverb * 100).toFixed(0)}%</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={voiceReverb}
            onChange={(e) => onUpdateSettings({ ...settings, voice_reverb_amount: parseFloat(e.target.value) })}
            style={{ width: "100%" }}
          />
        </div>

        <div className="settings-group">
          <label className="settings-label">Compression: {(voiceCompression * 100).toFixed(0)}%</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={voiceCompression}
            onChange={(e) => onUpdateSettings({ ...settings, voice_compression: parseFloat(e.target.value) })}
            style={{ width: "100%" }}
          />
        </div>
      </SettingsSection>

      <SettingsSection title="Memory Consolidation">

        <div className="settings-group">
          <label className="settings-label">Summarization API URL</label>
          <input
            className="input"
            value={settings.summarization_api_url || ""}
            onChange={(e) => onUpdateSettings({ ...settings, summarization_api_url: e.target.value || null })}
            placeholder="http://localhost:11434/v1"
          />
          <div className="settings-hint">OpenAI-compatible endpoint for the summarization model (can be same as main LLM or a smaller model)</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Summarization Model</label>
          <input
            className="input"
            value={settings.summarization_model || ""}
            onChange={(e) => onUpdateSettings({ ...settings, summarization_model: e.target.value || null })}
            placeholder="phi-3 or qwen2:0.5b"
          />
          <div className="settings-hint">Small, fast model for consolidating memories (e.g., phi-3, qwen2:0.5b, gemma-2b)</div>
        </div>

        <div className="settings-group">
          <label className="settings-label">Embedding Model (Optional)</label>
          <input
            className="input"
            value={settings.embedding_model || ""}
            onChange={(e) => onUpdateSettings({ ...settings, embedding_model: e.target.value || null })}
            placeholder="text-embedding-3-small"
          />
          <div className="settings-hint">Enables semantic search for memory retrieval. Leave empty to disable (uses FTS only). For OpenAI: text-embedding-3-small. For Ollama: nomic-embed-text.</div>
        </div>
      </SettingsSection>

      <SettingsSection title="Episodic Memory">
        <div className="settings-group" style={{ flexDirection: "row", alignItems: "center", gap: "12px" }}>
          <input
            type="checkbox"
            checked={episodicEnabled}
            onChange={(e) => onUpdateSettings({ ...settings, episodic_enabled: e.target.checked })}
            id="episodic-enabled-toggle"
          />
          <label htmlFor="episodic-enabled-toggle" className="settings-label">Enable Episodic Memory</label>
        </div>
        <div className="settings-hint">Captures episodic events for retrieval, summaries, and provenance.</div>

        <div className="settings-group" style={{ flexDirection: "row", alignItems: "center", gap: "12px" }}>
          <input
            type="checkbox"
            checked={episodicInjectionEnabled}
            disabled={!episodicEnabled}
            onChange={(e) => onUpdateSettings({ ...settings, episodic_injection_enabled: e.target.checked })}
            id="episodic-injection-toggle"
          />
          <label htmlFor="episodic-injection-toggle" className="settings-label">Inject Episodic Context</label>
        </div>
        <div className="settings-hint">Includes episodic memories in model context for retrieval-augmented responses.</div>

        <div className="settings-group">
          <label className="settings-label">Episodic Injection Limit</label>
          <input
            className="input"
            type="number"
            min="1"
            max="50"
            disabled={!episodicEnabled || !episodicInjectionEnabled}
            value={episodicInjectionLimit}
            onChange={(e) => {
              const next = Math.max(1, Math.min(50, parseInt(e.target.value, 10) || 1));
              onUpdateSettings({ ...settings, episodic_injection_limit: next });
            }}
          />
          <div className="settings-hint">Maximum episodic events to inject into a request (1-50).</div>
        </div>

        <div className="settings-group" style={{ flexDirection: "row", alignItems: "center", gap: "12px" }}>
          <input
            type="checkbox"
            checked={episodicCompactionEnabled}
            disabled={!episodicEnabled}
            onChange={(e) => onUpdateSettings({ ...settings, episodic_compaction_enabled: e.target.checked })}
            id="episodic-compaction-toggle"
          />
          <label htmlFor="episodic-compaction-toggle" className="settings-label">Enable Episodic Compaction</label>
        </div>
        <div className="settings-hint">Allows the scheduler to compact episodic events into summaries.</div>

        <div className="settings-group" style={{ flexDirection: "row", alignItems: "center", gap: "12px" }}>
          <input
            type="checkbox"
            checked={memoryClaimsEnabled}
            disabled={!episodicEnabled}
            onChange={(e) => onUpdateSettings({ ...settings, memory_claims_enabled: e.target.checked })}
            id="memory-claims-toggle"
          />
          <label htmlFor="memory-claims-toggle" className="settings-label">Enable Memory Claims</label>
        </div>
        <div className="settings-hint">Allows memory claims to be compiled into structured facts.</div>

        <div className="settings-group settings-group-border" style={{ flexDirection: "row", alignItems: "center", gap: "12px" }}>
          <input
            type="checkbox"
            checked={settings.phi_consent ?? false}
            onChange={(e) => onUpdateSettings({ ...settings, phi_consent: e.target.checked })}
            id="phi-consent-toggle"
          />
          <label htmlFor="phi-consent-toggle" className="settings-label">
            Allow sensitive memory storage (PHI/PII)
          </label>
        </div>
        <div className="settings-hint">
          When disabled, detected PHI/PII is blocked from memory writes and retrieval.
        </div>

        <div className="settings-group" style={{ flexDirection: "row", alignItems: "center", gap: "12px" }}>
          <input
            type="checkbox"
            checked={(phiScopeEnabled ?? (settings.phi_consent ?? false)) && !phiScopeLoading}
            onChange={(e) => updatePhiScopeConsent(e.target.checked)}
            id="phi-scope-consent-toggle"
            disabled={phiScopeLoading}
          />
          <label htmlFor="phi-scope-consent-toggle" className="settings-label">
            Conversation PHI consent (default)
          </label>
        </div>
        <div className="settings-hint">
          Overrides the global PHI/PII setting for conversation <code>default</code>.{" "}
          {phiScopeEnabled === null ? "Using global setting." : "Override active."}
          {phiScopeError ? ` Error: ${phiScopeError}` : ""}
        </div>
      </SettingsSection>

      <SettingsSection title="Memory Health">
        <div className="settings-group">
          <label className="settings-label">Last User Memory Write</label>
          <div className="settings-hint">{lastUserMemoryWrite}</div>
        </div>
        <div className="settings-group">
          <label className="settings-label">Last Self Memory Write</label>
          <div className="settings-hint">{lastSelfMemoryWrite}</div>
        </div>
        <div className="settings-group">
          <label className="settings-label">Last Memory Error</label>
          <div className="settings-hint">{lastMemoryErrorAt}</div>
        </div>
      </SettingsSection>

      <SettingsSection title="Self Awareness">
        <div className="settings-group" style={{ flexDirection: "row", alignItems: "center", gap: "12px" }}>
          <input
            type="checkbox"
            checked={selfModel?.reflection_frozen || false}
            onChange={(e) => onSetReflectionFrozen(e.target.checked)}
            id="reflection-freeze-toggle"
          />
          <label htmlFor="reflection-freeze-toggle" className="settings-label">Freeze Reflection Loop</label>
        </div>
        <div className="settings-group" style={{ flexDirection: "row", alignItems: "center", gap: "12px" }}>
          <button className="btn btn-secondary" onClick={onRefreshSelfData}>
            Refresh Self Data
          </button>
        </div>
        <div className="settings-group">
          <label className="settings-label">Reflection Status</label>
          <div className="settings-hint">Status: {reflectionStatusLabel}</div>
          <div className="settings-hint">Last Reflection At: {lastReflectionAt}</div>
          <div className="settings-hint">Allowlist Count: {reflectionAllowlistCount}</div>
        </div>
        <div className="settings-group">
          <label className="settings-label">Self Model Snapshot</label>
          <pre className="settings-hint" style={{ whiteSpace: "pre-wrap", marginTop: "8px" }}>
            {formatJson(selfModel)}
          </pre>
        </div>
        <div className="settings-group">
          <label className="settings-label">Self Inspection</label>
          <pre className="settings-hint" style={{ whiteSpace: "pre-wrap", marginTop: "8px" }}>
            {formatJson(selfInspection)}
          </pre>
        </div>
      </SettingsSection>

      <SettingsSection title="Danger Zone" variant="danger">
        <div className="settings-group" style={{ gap: "10px" }}>
          <button
            className="btn btn-danger"
            onClick={onWipeMemory}
          >
            Wipe All Data (New Install)
          </button>
          <div className="settings-hint">
            Resets Symbiote to a new install state (settings, conversations, memory, and rolling summary).
          </div>
          <button className="btn btn-secondary" onClick={onResetConversationData}>
            Reset Conversation Data
          </button>
          <div className="settings-hint">
            Clears conversation history and rolling summary for the current conversation. Memory and settings stay.
          </div>
        </div>
      </SettingsSection>

      <SettingsSection title="Actions" defaultOpen>
        <div className="settings-group settings-group-inline">
          <button className="btn btn-secondary" onClick={onTestConnection}>Test Connection</button>
          <button className="btn btn-primary" onClick={onSaveSettings}>Save Settings</button>
        </div>
      </SettingsSection>

      {testResult && (
        <div className={`glass settings-test-result ${testResult.success ? "success" : "error"}`}>
          {testResult.message}
        </div>
      )}

      {isDev && (
        <SettingsSection title="Developer Overrides">
          <div className="settings-group" style={{ flexDirection: "row", alignItems: "center", gap: "12px" }}>
            <input
              type="checkbox"
              checked={debugFlags.simulateTimeouts}
              onChange={(e) => updateDebugFlags({ simulateTimeouts: e.target.checked })}
              id="debug-timeouts-toggle"
            />
            <label htmlFor="debug-timeouts-toggle" className="settings-label">Simulate Timeouts</label>
          </div>
          <div className="settings-hint">Forces async calls to time out quickly.</div>

          <div className="settings-group" style={{ flexDirection: "row", alignItems: "center", gap: "12px" }}>
            <input
              type="checkbox"
              checked={debugFlags.simulateFailures}
              onChange={(e) => updateDebugFlags({ simulateFailures: e.target.checked })}
              id="debug-failures-toggle"
            />
            <label htmlFor="debug-failures-toggle" className="settings-label">Simulate Failures</label>
          </div>
          <div className="settings-hint">Forces async calls to fail immediately for testing recovery paths.</div>
        </SettingsSection>
      )}
    </div>
  );
};
