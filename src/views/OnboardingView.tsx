import { useEffect, useRef, useState } from "react";
import { Settings } from "../types/app";

export type OnboardingPayload = {
  userName: string;
  assistantName: string;
  apiBaseUrl: string;
  summarizationApiUrl: string;
};

interface OnboardingViewProps {
  settings: Settings | null;
  settingsError: string | null;
  onComplete: (payload: OnboardingPayload) => Promise<void>;
  onRetry: () => void;
}

const trimValue = (value: string) => value.trim();

export const OnboardingView = ({
  settings,
  settingsError,
  onComplete,
  onRetry,
}: OnboardingViewProps) => {
  const [userName, setUserName] = useState("");
  const [assistantName, setAssistantName] = useState("");
  const [apiBaseUrl, setApiBaseUrl] = useState("");
  const [summarizationApiUrl, setSummarizationApiUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const initializedRef = useRef(false);

  useEffect(() => {
    if (!settings || initializedRef.current) return;
    setUserName(settings.user_display_name ?? "");
    setAssistantName(settings.assistant_display_name ?? "");
    setApiBaseUrl(settings.api_base_url ?? "");
    setSummarizationApiUrl(settings.summarization_api_url ?? settings.api_base_url ?? "");
    initializedRef.current = true;
  }, [settings]);

  const canSubmit = [userName, assistantName, apiBaseUrl, summarizationApiUrl]
    .every((value) => trimValue(value).length > 0);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSubmit || saving) return;
    setSaving(true);
    setError(null);
    try {
      await onComplete({
        userName: trimValue(userName),
        assistantName: trimValue(assistantName),
        apiBaseUrl: trimValue(apiBaseUrl),
        summarizationApiUrl: trimValue(summarizationApiUrl),
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  const summaryText = settings
    ? `History window will default to 3 turns for this new install.`
    : "Loading settings...";

  return (
    <div className="onboarding-shell">
      <div className="onboarding-aurora" aria-hidden="true"></div>
      <div className="onboarding-card">
        <section className="onboarding-intro">
          <p className="onboarding-kicker">Welcome to Symbiote</p>
          <h1>Calibrate your operator profile.</h1>
          <p className="onboarding-subtitle">
            Set your names and connect the primary and operations model endpoints. These defaults power
            every run and can be refined later in Settings.
          </p>
          <div className="onboarding-summary">
            <div className="onboarding-summary-row">
              <span className="onboarding-summary-label">Default state</span>
              <span className="onboarding-summary-value">{summaryText}</span>
            </div>
            <div className="onboarding-summary-row">
              <span className="onboarding-summary-label">Privacy</span>
              <span className="onboarding-summary-value">You will be prompted before saving non-local URLs.</span>
            </div>
          </div>
          {settingsError && (
            <div className="onboarding-error">
              <p>Settings failed to load: {settingsError}</p>
            </div>
          )}
        </section>

        <form className="onboarding-form" onSubmit={handleSubmit}>
          <div className="onboarding-field">
            <label className="onboarding-label" htmlFor="onboarding-user-name">User name</label>
            <input
              id="onboarding-user-name"
              className="input onboarding-input"
              value={userName}
              onChange={(event) => setUserName(event.target.value)}
              placeholder="Operator name"
              autoComplete="name"
            />
          </div>

          <div className="onboarding-field">
            <label className="onboarding-label" htmlFor="onboarding-assistant-name">Preferred assistant name</label>
            <input
              id="onboarding-assistant-name"
              className="input onboarding-input"
              value={assistantName}
              onChange={(event) => setAssistantName(event.target.value)}
              placeholder="Assistant name"
            />
          </div>

          <div className="onboarding-field">
            <label className="onboarding-label" htmlFor="onboarding-api-base-url">Primary Model URL</label>
            <input
              id="onboarding-api-base-url"
              className="input onboarding-input"
              value={apiBaseUrl}
              onChange={(event) => setApiBaseUrl(event.target.value)}
              placeholder="http://localhost:11434"
              inputMode="url"
            />
            <p className="onboarding-help">Primary model endpoint used for chat responses.</p>
          </div>

          <div className="onboarding-field">
            <label className="onboarding-label" htmlFor="onboarding-summary-url">System Operations Model URL</label>
            <input
              id="onboarding-summary-url"
              className="input onboarding-input"
              value={summarizationApiUrl}
              onChange={(event) => setSummarizationApiUrl(event.target.value)}
              placeholder="http://localhost:11434"
              inputMode="url"
            />
            <p className="onboarding-help">System operations endpoint used for summaries, memory, and background tasks.</p>
          </div>

          {error && <div className="onboarding-error">{error}</div>}

          <div className="onboarding-actions">
            <button className="btn btn-secondary" type="button" onClick={onRetry} disabled={saving}>
              Reload settings
            </button>
            <button className="btn btn-primary" type="submit" disabled={!canSubmit || saving || !settings}>
              {saving ? "Saving..." : "Complete setup"}
            </button>
          </div>
          <p className="onboarding-footnote">You can adjust these values later in Settings.</p>
        </form>
      </div>
    </div>
  );
};
