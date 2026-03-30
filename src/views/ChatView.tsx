import { useEffect, useRef, useState, type RefObject } from "react";
import { VoiceController, VoiceControllerHandle, VoiceStatus } from "../components/VoiceController";
import { InnerMonologueEntry, Message, PendingPrompt, Settings, ModuleStatus, SelfModel, SystemHealthSnapshot, RecommendationItem } from "../types/app";
import { MessageContent } from "../components/MessageContent";
import { StatusStrip } from "../components/StatusStrip";
import { invokeWithTimeout } from "../utils/tauri";
import { formatStatusDetail, formatStatusLabel } from "../utils/statusStages";

interface ChatViewProps {
  messages: Message[];
  streamingTokenBuffer: string | null;
  streamingMessageId: string | null;
  moduleStatus: ModuleStatus | null;
  input: string;
  feedbackMode: boolean;
  chatState: string;
  chatError: string | null;
  memoryError?: { message: string; timestamp?: string } | null;
  isBusy: boolean;
  settings: Settings | null;
  selfModel?: SelfModel | null;
  healthSnapshot?: SystemHealthSnapshot | null;
  showRaw: boolean;
  rollingSummary: string | null;
  rollingSummaryError: string | null;
  rollingSummaryErrorAt?: string | null;
  rollingSummaryPending?: boolean;
  liveSummary: string | null;
  liveSummaryError: string | null;
  liveSummaryErrorAt?: string | null;
  liveSummaryPending?: boolean;
  onSummaryOpen?: () => void;
  innerMonologueEntries?: InnerMonologueEntry[];
  innerMonologueError?: string | null;
  onMonologueOpen?: () => void;
  pendingPrompts: PendingPrompt[];
  pendingPromptCount: number;
  pendingPromptError?: string | null;
  onPendingPromptSend: (promptId: string) => void;
  onPendingPromptDismiss: (promptId: string) => void;
  onPendingPromptRephrase: (promptId: string, prompt: string) => void;
  onInputChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  onRecover: () => void;
  onFeedbackModeChange: (value: boolean) => void;
  onDismissMemoryError?: () => void;
  onAppendTranscript: (text: string) => void;
  voiceStatus: string;
  voiceEnabled: boolean;
  memoryGraphOpen: boolean;
  memoryErrorAt?: string | null;
  voiceRef: RefObject<VoiceControllerHandle>;
  onVoiceStatusChange: (status: VoiceStatus) => void;
  inputRef: RefObject<HTMLTextAreaElement>;
}

const sanitizeStreamingContent = (text: string | null) => {
  if (!text) return null;
  let cleaned = text
    .replace(/<<<BEGIN_SECTION:[^>]*>>>/g, "")
    .replace(/<<<END_SECTION:[^>]*>>>/g, "");
  cleaned = cleaned.replace(/(^|\n)\s*(Next Steps|Proposed Response)\s*(?=\n|$)/gi, "$1");
  return cleaned;
};

export const ChatView = ({
  messages,
  streamingTokenBuffer,
  streamingMessageId,
  moduleStatus,
  input,
  feedbackMode,
  chatState,
  chatError,
  memoryError,
  isBusy,
  settings,
  selfModel,
  healthSnapshot,
  showRaw,
  rollingSummary,
  rollingSummaryError,
  rollingSummaryErrorAt,
  rollingSummaryPending = false,
  liveSummary,
  liveSummaryError,
  liveSummaryErrorAt,
  liveSummaryPending = false,
  onSummaryOpen,
  innerMonologueEntries = [],
  innerMonologueError,
  onMonologueOpen,
  pendingPrompts,
  pendingPromptCount,
  pendingPromptError,
  onPendingPromptSend,
  onPendingPromptDismiss,
  onPendingPromptRephrase,
  onInputChange,
  onSend,
  onStop,
  onRecover,
  onFeedbackModeChange,
  onDismissMemoryError,
  onAppendTranscript,
  voiceStatus,
  voiceEnabled,
  memoryGraphOpen,
  memoryErrorAt,
  voiceRef,
  onVoiceStatusChange,
  inputRef,
}: ChatViewProps) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const renderTimingStartRef = useRef<number | null>(null);
  const appStartRef = useRef<number>(Date.now());
  const [isSummaryOpen, setIsSummaryOpen] = useState(false);
  const [isMonologueOpen, setIsMonologueOpen] = useState(false);
  const [isPromptQueueOpen, setIsPromptQueueOpen] = useState(false);
  const [editingPromptId, setEditingPromptId] = useState<string | null>(null);
  const [editingPromptText, setEditingPromptText] = useState("");
  const [renderAllMessages, setRenderAllMessages] = useState(false);
  const [qualiaDrafts, setQualiaDrafts] = useState<Record<string, { tag: string; intensity: number; labelId?: string; savedAt?: string }>>({});
  const [openQualiaId, setOpenQualiaId] = useState<string | null>(null);
  const [chatErrorDismissed, setChatErrorDismissed] = useState(false);
  const [errorStartAssistantId, setErrorStartAssistantId] = useState<string | null>(null);
  const maxRenderMessages = 120;
  const safeStreamingBuffer = sanitizeStreamingContent(streamingTokenBuffer);
  const qualiaTags = ["pleasant", "harsh", "calm", "neutral", "anxious", "focused"];
  const defaultQualia = { tag: "neutral", intensity: 0.5 };
  const storedSummaryMissing = (!rollingSummary || rollingSummary.trim().length === 0)
    && Boolean(liveSummary && liveSummary.trim().length > 0)
    && !rollingSummaryPending;
  const liveSummaryBody = liveSummary && liveSummary.trim().length > 0
    ? liveSummary
    : liveSummaryPending
      ? "Generating summary..."
      : "No live summary yet.";
  const storedSummaryBody = rollingSummary && rollingSummary.trim().length > 0
    ? rollingSummary
    : rollingSummaryPending
      ? "Generating summary..."
      : "No stored summary yet.";
  const formatSelfReportNotice = (selfReport: any) => {
    if (!selfReport || typeof selfReport !== "object") return null;
    const notice = typeof selfReport.notice === "string" ? selfReport.notice.trim() : "";
    if (notice) return notice;
    const parts: string[] = [];
    const status = typeof selfReport.status === "string" ? selfReport.status : "unknown";
    parts.push(`status=${status}`);
    const evidenceIds = Array.isArray(selfReport.evidence_event_ids) ? selfReport.evidence_event_ids : [];
    const speculative = Boolean(selfReport.speculative) || evidenceIds.length === 0;
    if (speculative) {
      parts.push("speculative");
    }
    if (typeof selfReport.confidence === "number") {
      parts.push(`conf=${selfReport.confidence.toFixed(2)}`);
    }
    if (typeof selfReport.uncertainty === "number") {
      parts.push(`uncert=${selfReport.uncertainty.toFixed(2)}`);
    }
    if (typeof selfReport.self_model_reliability === "number") {
      parts.push(`reliability=${selfReport.self_model_reliability.toFixed(2)}`);
    }
    const constraints = Array.isArray(selfReport.constraints)
      ? selfReport.constraints.filter((item: unknown) => typeof item === "string" && item.trim().length > 0)
      : [];
    if (constraints.length > 0) {
      const trimmed = constraints.slice(0, 2).join(", ");
      const suffix = constraints.length > 2 ? ` +${constraints.length - 2} more` : "";
      parts.push(`constraints=${trimmed}${suffix}`);
    }
    return parts.join(" | ");
  };

  const trustBadgeFor = (message: Message) => {
    if (message.role !== "assistant") return null;
    const meta = (message.metadata ?? {}) as any;
    const selfReport = meta?.self_report ?? null;
    const evidenceIds: number[] = [];
    if (Array.isArray(meta.evidence_event_ids)) {
      evidenceIds.push(...meta.evidence_event_ids);
    }
    if (Array.isArray(meta.top_evidence_event_ids)) {
      evidenceIds.push(...meta.top_evidence_event_ids);
    }
    if (typeof meta.evidence_event_id === "number") {
      evidenceIds.push(meta.evidence_event_id);
    }
    if (Array.isArray(selfReport?.evidence_event_ids)) {
      evidenceIds.push(...selfReport.evidence_event_ids);
    }
    const beliefIds = Array.isArray(meta.belief_ids) ? meta.belief_ids : [];
    const hasEvidence = evidenceIds.length > 0 || beliefIds.length > 0;
    const isSpeculative = Boolean(meta.speculative || meta.speculative_reason || selfReport?.speculative);
    const gateDecision = meta.gate_decision as string | undefined;
    if (gateDecision === "ALLOW_WITH_AUDIT") {
      return { label: "Audit", tone: "audit", detail: "Audited response. Review evidence chain." };
    }
    if (gateDecision === "ALLOW_WITH_NOTICE") {
      return { label: "Notice", tone: "notice", detail: "Notice issued. Evidence may be partial." };
    }
    if (hasEvidence && !isSpeculative) {
      return { label: "Grounded", tone: "high", detail: "Evidence-linked response." };
    }
    if (hasEvidence && isSpeculative) {
      return { label: "Partial", tone: "medium", detail: "Evidence present but response marked speculative." };
    }
    if (isSpeculative) {
      return { label: "Speculative", tone: "low", detail: "Speculative response without evidence." };
    }
    return { label: "Unverified", tone: "low", detail: "No evidence attached." };
  };

  const getQualiaDraft = (id: string) => qualiaDrafts[id] ?? defaultQualia;
  const updateQualiaDraft = (id: string, patch: Partial<{ tag: string; intensity: number; labelId?: string; savedAt?: string }>) => {
    setQualiaDrafts((prev) => ({
      ...prev,
      [id]: { ...defaultQualia, ...prev[id], ...patch },
    }));
  };
  const recordQualiaLabel = async (message: Message, tag: string, intensity: number) => {
    const labelId = await invokeWithTimeout<string>(
      "record_qualia_label",
      {
        eventId: message.message_id,
        tag,
        intensity,
        context: {
          role: message.role,
          createdAt: message.created_at,
          runId: message.run_id,
          traceId: message.trace_id,
        },
      },
      15000,
    );
    updateQualiaDraft(message.message_id, { labelId, savedAt: new Date().toISOString(), tag, intensity });
    return labelId;
  };
  const submitQualiaLabel = async (message: Message) => {
    const draft = getQualiaDraft(message.message_id);
    await recordQualiaLabel(message, draft.tag, draft.intensity);
  };
  const submitQualiaReward = async (message: Message, magnitude: number) => {
    const draft = getQualiaDraft(message.message_id);
    if (!draft.labelId) return;
    await invokeWithTimeout<string>(
      "record_qualia_reward",
      {
        labelId: draft.labelId,
        magnitude,
        outcomeRef: message.message_id,
      },
      15000,
    );
  };
  const deriveStreamStatus = () => {
    if (moduleStatus?.stage) {
      return {
        label: formatStatusLabel(moduleStatus.stage),
        detail: formatStatusDetail(moduleStatus.stage, moduleStatus.detail),
      };
    }
    if (chatState === "streaming") {
      return { label: "Streaming reply", detail: "Streaming response" };
    }
    if (chatState === "post_processing") {
      return { label: "Post-processing", detail: "Finalizing updates" };
    }
    if (chatState === "sending") {
      return { label: "Waiting on LLM", detail: "Generating response" };
    }
    if (chatState.startsWith("awaiting_")) {
      return { label: "Waiting", detail: "Awaiting your input" };
    }
    if (chatState === "stopping") {
      return { label: "Stopping", detail: "Halting generation" };
    }
    if (chatState === "error") {
      return { label: "Error", detail: "Response stalled" };
    }
    return { label: "Idle", detail: "Standing by" };
  };

  const recommendationItems = (healthSnapshot?.metrics?.recommendations?.items ?? []) as RecommendationItem[];
  const eligibleRecommendations = recommendationItems.filter((rec) => rec.status === "eligible");
  const applyRecommendation = async (rec: RecommendationItem) => {
    if (!rec.action) return;
    await invokeWithTimeout(
      "apply_recommendation",
      {
        recommendation_id: rec.recommendation_id,
        kind: rec.kind,
        snapshot_id: healthSnapshot?.snapshot_id ?? null,
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
        snapshot_id: healthSnapshot?.snapshot_id ?? null,
        gate: rec.gate ?? null,
      },
      10000,
    );
  };

  const streamStatus = deriveStreamStatus();

  const shouldRenderMessage = (message: Message) => {
    if (showRaw) return true;
    if (message.role === "assistant" && (message.status === "cancelled" || message.status === "error")) {
      return false;
    }
    const meta = message.metadata as any;
    const showMonologue = settings?.show_monologue_in_chat ?? false;
    if (meta?.origin === "monologue" && !showMonologue) {
      return false;
    }
    if (message.role === "internal" || message.role === "system") {
      if (!(showMonologue && meta?.origin === "monologue")) {
        return false;
      }
    }
    if (meta?.surface === false) return false;
    return true;
  };
  const filteredMessages = messages.filter(shouldRenderMessage);
  const assistantMessages = filteredMessages.filter(
    (message) => message.role === "assistant" && message.status === "complete",
  );
  const latestAssistantMessage =
    assistantMessages.length > 0 ? assistantMessages[assistantMessages.length - 1] : null;
  const lastMessageId = filteredMessages.length > 0 ? filteredMessages[filteredMessages.length - 1].message_id : null;
  const visibleMessages = renderAllMessages
    ? filteredMessages
    : filteredMessages.slice(Math.max(0, filteredMessages.length - maxRenderMessages));
  const promptCount = pendingPromptCount > 0 ? pendingPromptCount : pendingPrompts.length;
  const canStop = chatState === "sending" || chatState === "streaming" || chatState === "stopping";
  const latestAssistantMessageId = latestAssistantMessage?.message_id ?? null;

  const reflectionBadge = (() => {
    if (!selfModel) return null;
    if (selfModel.reflection_frozen) {
      return { label: "Reflection frozen", tone: "warn" };
    }
    const reflectionStatus =
      selfModel.reflection_status && typeof selfModel.reflection_status === "object"
        ? (selfModel.reflection_status as Record<string, any>)
        : {};
    const lastAttemptRaw = typeof reflectionStatus.last_run === "string" ? reflectionStatus.last_run : null;
    const lastAttempt = lastAttemptRaw ? Date.parse(lastAttemptRaw) : Number.NaN;
    const hasAttempt = !Number.isNaN(lastAttempt);
    const graceMs = 20 * 60 * 1000;
    const graceExpired = Date.now() - appStartRef.current >= graceMs;
    if (!selfModel.last_reflection_at) {
      if (!hasAttempt && !graceExpired) {
        return null;
      }
      return { label: "Reflection missing", tone: "warn" };
    }
    const last = Date.parse(selfModel.last_reflection_at);
    if (!Number.isNaN(last)) {
      const ageHours = (Date.now() - last) / 36e5;
      if (ageHours > 24) {
        return { label: `Reflection stale (${Math.round(ageHours)}h)`, tone: "warn" };
      }
    }
    return null;
  })();

  useEffect(() => {
    if (chatState === "error") {
      setChatErrorDismissed(false);
      setErrorStartAssistantId(latestAssistantMessageId);
      return;
    }
    setChatErrorDismissed(false);
    setErrorStartAssistantId(null);
  }, [chatState]);

  useEffect(() => {
    if (chatState !== "error") return;
    if (!latestAssistantMessageId) return;
    if (errorStartAssistantId && latestAssistantMessageId === errorStartAssistantId) return;
    setChatErrorDismissed(true);
  }, [chatState, latestAssistantMessageId, errorStartAssistantId]);

  const safeParseCandidate = (raw: string) => {
    try {
      return JSON.parse(raw) as { kind?: string; payload?: unknown };
    } catch {
      return null;
    }
  };

  const safeParseHarvest = (raw?: string | null) => {
    if (!raw) return null;
    try {
      return JSON.parse(raw) as { speculative?: boolean; anchor_source?: string };
    } catch {
      return null;
    }
  };

  const truncateText = (text: string, max = 240) => {
    if (text.length <= max) return text;
    return `${text.slice(0, max)}...`;
  };

  const collapseStatusEntries = (entries: InnerMonologueEntry[]) => {
    const output: (InnerMonologueEntry & { repeatCount?: number; isStatus?: boolean })[] = [];
    for (const entry of entries) {
      const isStatus = (entry.stream_type ?? "").toUpperCase() === "STATUS" || entry.mode === "status";
      if (isStatus && output.length > 0) {
        const last = output[output.length - 1];
        if (last.isStatus && last.thought === entry.thought) {
          last.repeatCount = (last.repeatCount ?? 1) + 1;
          last.created_at = entry.created_at;
          continue;
        }
      }
      output.push({ ...entry, repeatCount: 1, isStatus });
    }
    return output;
  };

  const collapsedMonologueEntries = collapseStatusEntries(innerMonologueEntries);

  const groupedMonologue = (() => {
    const groups: {
      key: string;
      created_at: string;
      mode: string;
      stream_type?: string | null;
      entries: (InnerMonologueEntry & { repeatCount?: number; isStatus?: boolean })[];
    }[] = [];
    const byKey = new Map<string, typeof groups[0]>();

    for (const entry of collapsedMonologueEntries) {
      const stream = entry.stream_type ?? "DS";
      const key = `${entry.dialogue_id ?? entry.id}:${stream}`;
      let group = byKey.get(key);
      if (!group) {
        group = {
          key,
          created_at: entry.created_at,
          mode: entry.mode,
          stream_type: stream,
          entries: [],
        };
        byKey.set(key, group);
        groups.push(group);
      }
      group.entries!.push(entry);
    }

    for (const group of groups) {
      group.entries = [...(group.entries || [])].sort((a, b) => {
        const aIndex = a.turn_index ?? 0;
        const bIndex = b.turn_index ?? 0;
        if (aIndex !== bIndex) {
          return aIndex - bIndex;
        }
        return a.created_at.localeCompare(b.created_at);
      });
    }

    return groups;
  })();

  const handleSummaryToggle = () => {
    setIsSummaryOpen((prev) => {
      const next = !prev;
      if (!prev && next) {
        onSummaryOpen?.();
      }
      return next;
    });
  };

  const handleMonologueToggle = () => {
    setIsMonologueOpen((prev) => {
      const next = !prev;
      if (!prev && next) {
        onMonologueOpen?.();
      }
      return next;
    });
  };

  const handlePromptQueueToggle = () => {
    setIsPromptQueueOpen((prev) => {
      const next = !prev;
      if (!next) {
        setEditingPromptId(null);
        setEditingPromptText("");
      }
      return next;
    });
  };

  const startPromptEdit = (prompt: PendingPrompt) => {
    setEditingPromptId(prompt.id);
    setEditingPromptText(prompt.prompt);
  };

  const cancelPromptEdit = () => {
    setEditingPromptId(null);
    setEditingPromptText("");
  };

  const submitPromptEdit = () => {
    if (!editingPromptId) return;
    const trimmed = editingPromptText.trim();
    if (!trimmed) return;
    onPendingPromptRephrase(editingPromptId, trimmed);
    setEditingPromptId(null);
    setEditingPromptText("");
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingTokenBuffer]);

  useEffect(() => {
    renderTimingStartRef.current = performance.now();
    const raf = requestAnimationFrame(() => {
      const start = renderTimingStartRef.current;
      if (start === null) return;
      const duration = performance.now() - start;
      renderTimingStartRef.current = null;
      void invokeWithTimeout("log_ui_timing", {
        event: "chat_render",
        duration_ms: Math.round(duration),
        message_id: lastMessageId ?? undefined,
      }).catch(() => {});
    });
    return () => cancelAnimationFrame(raf);
  }, [messages, streamingTokenBuffer]);

  useEffect(() => {
    if (promptCount === 0) {
      setIsPromptQueueOpen(false);
      setEditingPromptId(null);
      setEditingPromptText("");
      return;
    }
    if (editingPromptId && !pendingPrompts.some((prompt) => prompt.id === editingPromptId)) {
      setEditingPromptId(null);
      setEditingPromptText("");
    }
  }, [promptCount, pendingPrompts, editingPromptId]);

  return (
    <>
      <div className="chat-view">
        <div className="summary-eye-container">
          <button
            className="summary-eye-button"
            onClick={handleSummaryToggle}
            aria-label="Toggle rolling summary"
            title="Rolling summary"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12Z" />
              <circle cx="12" cy="12" r="3.5" />
            </svg>
          </button>
          <button
            className="summary-eye-button summary-brain-button"
            onClick={handleMonologueToggle}
            aria-label="Toggle inner monologue"
            title="Inner monologue"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M8.5 3a3.5 3.5 0 0 0-3.5 3.5v.5a3 3 0 0 0 0 6v1a3 3 0 0 0 3 3h.5a3 3 0 0 0 6 1.5A3 3 0 0 0 20 17a3 3 0 0 0 0-6v-1A3.5 3.5 0 0 0 16.5 6H16A3.5 3.5 0 0 0 8.5 3Z" />
              <path d="M9 7.5c0-1.5 1-2.5 2.5-2.5S14 6 14 7.5" />
              <path d="M10 11c0-1 1-2 2-2s2 1 2 2" />
              <path d="M11 14c0-.8.7-1.5 1.5-1.5S14 13.2 14 14" />
            </svg>
          </button>
          <button
            className="summary-eye-button summary-queue-button"
            onClick={handlePromptQueueToggle}
            aria-label="Toggle system prompt queue"
            title="System prompt queue"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M4 6h16" />
              <path d="M4 12h10" />
              <path d="M4 18h12" />
              <circle cx="18" cy="12" r="2.5" />
            </svg>
            {promptCount > 0 && (
              <span className="pending-prompt-badge">{promptCount}</span>
            )}
          </button>
          <button
            className={`summary-eye-button feedback-icon-button${feedbackMode ? " active" : ""}`}
            onClick={() => onFeedbackModeChange(!feedbackMode)}
            aria-label={`Feedback ${feedbackMode ? "On" : "Off"}`}
            aria-pressed={feedbackMode}
            title={`Feedback ${feedbackMode ? "On" : "Off"} — marks your next message as feedback`}
            disabled={isBusy}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h8" />
              <path d="M16 3h5v5" />
              <path d="M21 3l-7 7" />
            </svg>
            <span className="sr-only">Feedback {feedbackMode ? "On" : "Off"}</span>
          </button>
            {isSummaryOpen && (
            <div className="summary-panel glass">
              <div className="summary-panel-title">Live Summary</div>
              {liveSummaryError && (
                <div className="summary-panel-error">
                  {liveSummaryErrorAt
                    ? `Last error (${liveSummaryErrorAt}): ${liveSummaryError}`
                    : `Last error: ${liveSummaryError}`}
                </div>
              )}
              <div className="summary-panel-body">
                {liveSummaryBody}
              </div>

              <div className="summary-panel-divider" />

              <div className="summary-panel-title">Stored Summary (Governed)</div>
              {rollingSummaryError && (
                <div className="summary-panel-error">
                  {rollingSummaryErrorAt
                    ? `Last error (${rollingSummaryErrorAt}): ${rollingSummaryError}`
                    : `Last error: ${rollingSummaryError}`}
                </div>
              )}
              {storedSummaryMissing && (
                <div className="summary-panel-note">
                  Stored summary is gated; live summary is keeping context.
                </div>
              )}
              <div className="summary-panel-body">
                {storedSummaryBody}
              </div>
            </div>
          )}
          {isPromptQueueOpen && (
            <div className="summary-panel glass pending-prompt-panel">
              <div className="summary-panel-title">System Prompt Queue</div>
              {pendingPromptError && (
                <div className="summary-panel-error">{pendingPromptError}</div>
              )}
              <div className="pending-prompt-list">
                {pendingPrompts.length === 0 && !pendingPromptError && "No pending prompts."}
                {pendingPrompts.map((prompt) => (
                  <div key={prompt.id} className="pending-prompt-item">
                    <div className="pending-prompt-meta">
                      <span className="pending-prompt-source">{prompt.source}</span>
                      {prompt.auto_surface && (
                        <span className="pending-prompt-auto">auto-surface</span>
                      )}
                      {prompt.skip_count > 0 && (
                        <span className="pending-prompt-skip">skips {prompt.skip_count}</span>
                      )}
                      <span className="pending-prompt-time">{prompt.created_at}</span>
                    </div>
                    {editingPromptId === prompt.id ? (
                      <div className="pending-prompt-edit">
                        <textarea
                          className="pending-prompt-textarea"
                          value={editingPromptText}
                          onChange={(e) => setEditingPromptText(e.target.value)}
                          rows={3}
                        />
                        <div className="pending-prompt-actions">
                          <button className="btn btn-primary" onClick={submitPromptEdit}>
                            Save
                          </button>
                          <button className="btn btn-secondary" onClick={cancelPromptEdit}>
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <div className="pending-prompt-text">{prompt.prompt}</div>
                        <div className="pending-prompt-actions">
                          <button className="btn btn-primary" onClick={() => onPendingPromptSend(prompt.id)}>
                            Send
                          </button>
                          <button className="btn btn-secondary" onClick={() => startPromptEdit(prompt)}>
                            Rephrase
                          </button>
                          <button className="btn btn-secondary" onClick={() => onPendingPromptDismiss(prompt.id)}>
                            Dismiss
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
          {isMonologueOpen && (
            <div className="summary-panel glass summary-panel-monologue">
              <div className="summary-panel-title">System Thinking</div>
              {innerMonologueError && (
                <div className="summary-panel-error">{innerMonologueError}</div>
              )}
              <div className="summary-panel-body summary-panel-body-monologue">
                {collapsedMonologueEntries.length === 0 && !innerMonologueError && "No inner monologue yet."}
                {groupedMonologue.map((group) => (
                  <div key={group.key} className="monologue-entry monologue-group">
                    <div className="monologue-meta">
                      <span className="monologue-mode">{group.mode}</span>
                      <span className="monologue-mode">{group.stream_type ?? "DS"}</span>
                      <span className="monologue-time">{group.created_at}</span>
                    </div>
                    <div className="monologue-dialogue">
                      {group.entries?.map((entry, index) => (
                        <div key={entry.id} className={`monologue-turn ${entry.isStatus ? "monologue-status" : ""}`}>
                          <div className="monologue-turn-header">
                            <span className="monologue-speaker">
                              {entry.isStatus
                                ? "Status"
                                : `Stance: ${(entry.speaker || "self").replace("_", " ").toUpperCase()}`}
                            </span>
                            <span className="monologue-turn-index">
                              {entry.isStatus
                                ? entry.repeatCount && entry.repeatCount > 1
                                  ? `x${entry.repeatCount}`
                                  : "status"
                                : `Turn ${entry.turn_index ?? index + 1}`}
                            </span>
                          </div>
                          <div className="monologue-thought">{entry.thought}</div>
                          {!entry.isStatus && entry.descriptors && entry.descriptors.length > 0 && (
                            <div className="monologue-descriptors">
                              <strong>Descriptors (instrumental):</strong>{" "}
                              {entry.descriptors.join(", ")}
                            </div>
                          )}
                          {!entry.isStatus && entry.harvest_type && (
                            <div className="monologue-harvest">
                              {(() => {
                                const parsed = safeParseHarvest(entry.harvest_payload);
                                if (parsed?.speculative) {
                                  return (
                                    <>
                                      <strong>Speculative:</strong>{" "}
                                      {parsed.anchor_source ? `anchor=${parsed.anchor_source}` : ""}
                                    </>
                                  );
                                }
                                return (
                                  <>
                                    <strong>{entry.harvest_type}:</strong>{" "}
                                    {entry.harvest_payload || ""}
                                  </>
                                );
                              })()}
                            </div>
                          )}
                          {!entry.isStatus && entry.candidates && entry.candidates.length > 0 && (
                            <div className="monologue-candidates">
                              {entry.candidates.map((candidate) => {
                                const parsed = safeParseCandidate(candidate.candidate_json);
                                const kind = parsed?.kind ?? "candidate";
                                const payload = parsed?.payload ?? candidate.candidate_json;
                                const blockedReason = (parsed as any)?.blocked_reason as string | undefined;
                                const speculative = typeof payload === "object" && payload !== null
                                  ? (payload as any).speculative
                                  : false;
                                const speculativeReason = typeof payload === "object" && payload !== null
                                  ? (payload as any).speculative_reason
                                  : undefined;
                                const payloadText = typeof payload === "string"
                                  ? payload
                                  : JSON.stringify(payload);
                                return (
                                  <div key={candidate.id} className="monologue-candidate">
                                    <div className="monologue-candidate-kind">{kind}</div>
                                    {candidate.outcome && (
                                      <div className="monologue-candidate-outcome">{candidate.outcome}</div>
                                    )}
                                    {blockedReason && (
                                      <div className="monologue-candidate-outcome">
                                        blocked: {blockedReason}
                                      </div>
                                    )}
                                    {speculative && (
                                      <div className="monologue-candidate-outcome">
                                        speculative{speculativeReason ? `: ${speculativeReason}` : ""}
                                      </div>
                                    )}
                                    {payloadText && (
                                      <div className="monologue-candidate-payload">
                                        {truncateText(payloadText)}
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {memoryError && (
          <div className="chat-recovery-banner glass">
            <div>
              <strong>Memory write issue.</strong>{" "}
              {memoryError.timestamp ? `(${memoryError.timestamp}) ` : ""}
              {memoryError.message}
            </div>
            {onDismissMemoryError && (
              <button className="btn btn-secondary" onClick={onDismissMemoryError}>
                Dismiss
              </button>
            )}
          </div>
        )}

        <div className="message-list">
        {!renderAllMessages && messages.length > maxRenderMessages && (
          <button
            className="btn btn-secondary"
            style={{ alignSelf: "center" }}
            onClick={() => setRenderAllMessages(true)}
          >
            Load earlier messages ({messages.length - maxRenderMessages} hidden)
          </button>
        )}
        {visibleMessages.map((m) => {
          if (m.role === "internal") {
            return null;
          }
          const hasStreamingBuffer = Boolean(safeStreamingBuffer && safeStreamingBuffer.trim().length > 0);
          const isActiveStreamMessage = streamingMessageId
            ? m.message_id === streamingMessageId
            : m.status === "streaming";
          const isEmptyStreaming = m.status === "streaming" && !m.content.trim() && !hasStreamingBuffer;
          if (isEmptyStreaming && !isActiveStreamMessage) {
            return null;
          }
          const isDraft = m.role === "assistant" && (m.status !== "complete" || isActiveStreamMessage);

          const content = (isActiveStreamMessage && m.status === "streaming" && hasStreamingBuffer)
            ? safeStreamingBuffer!
            : m.content;
          const meta = m.metadata as any;
          const gateNotice: string | null = meta?.gate_notice ?? null;
          const extraNotice: string | null = meta?.extra_notice ?? null;
          const gateDecision: string | null = meta?.gate_decision ?? null;
          const selfReport = meta?.self_report ?? null;
          const selfReportNotice = formatSelfReportNotice(selfReport);
          const trustBadge = trustBadgeFor(m);

          return (
            <div key={m.message_id} className={`message ${m.role}${isDraft ? " draft" : ""}`}>
              <div className="message-bubble glass">
                {trustBadge && (
                  <div className={`trust-badge trust-${trustBadge.tone}`} title={trustBadge.detail}>
                    {trustBadge.label}
                  </div>
                )}
                {m.role === "assistant" && (
                  <button
                    className={`qualia-icon${openQualiaId === m.message_id ? " active" : ""}`}
                    aria-label="Qualia controls"
                    aria-expanded={openQualiaId === m.message_id}
                    onClick={() => setOpenQualiaId((prev) => (prev === m.message_id ? null : m.message_id))}
                    type="button"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
                      <circle cx="12" cy="12" r="9" />
                      <path d="M8 12h8" />
                      <path d="M12 8v8" />
                    </svg>
                  </button>
                )}
                {isDraft && (!content || !content.trim()) ? (
                  <div className="streaming-placeholder">
                    <div className="streaming-dots">
                      <span className="streaming-dot" />
                      <span className="streaming-dot" />
                      <span className="streaming-dot" />
                    </div>
                    <div className="streaming-status">
                      <div className="streaming-status-label">{streamStatus.label}</div>
                      <div className="streaming-status-detail">{streamStatus.detail}</div>
                    </div>
                  </div>
                ) : (
                  <MessageContent
                    content={content}
                    showRaw={showRaw}
                  />
                )}
                {gateNotice && (
                  <div className={`gate-notice ${gateDecision === "ALLOW_WITH_AUDIT" ? "audit" : "notice"}`}>
                    <span className="gate-notice-title">
                      {gateDecision === "ALLOW_WITH_AUDIT" ? "Audit" : "Notice"}
                    </span>
                    <span className="gate-notice-body">{gateNotice}</span>
                  </div>
                )}
                {extraNotice && (
                  <div className="gate-notice notice">
                    <span className="gate-notice-title">Notice</span>
                    <span className="gate-notice-body">{extraNotice}</span>
                  </div>
                )}
                {selfReport && (
                  <div className="self-report-banner">
                    <span className="self-report-title">Self-Report</span>
                    <span className="self-report-body">
                      {selfReportNotice || "Self-report available."}
                    </span>
                  </div>
                )}
                {m.status === "error" && (
                  <div className="message-error-text">Error generating response</div>
                )}
                {m.role === "assistant" && openQualiaId === m.message_id && (
                  <div className="qualia-menu">
                    <div className="qualia-menu-row">
                      <span className="qualia-menu-label">Tag</span>
                      <select
                        className="qualia-select"
                        value={getQualiaDraft(m.message_id).tag}
                        onChange={(e) => updateQualiaDraft(m.message_id, { tag: e.target.value })}
                      >
                        {qualiaTags.map((tag) => (
                          <option key={tag} value={tag}>{tag}</option>
                        ))}
                      </select>
                    </div>
                    <div className="qualia-menu-row">
                      <span className="qualia-menu-label">Intensity</span>
                      <input
                        className="qualia-range"
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={getQualiaDraft(m.message_id).intensity}
                        onChange={(e) => updateQualiaDraft(m.message_id, { intensity: Number(e.target.value) })}
                      />
                      <span className="qualia-value">{getQualiaDraft(m.message_id).intensity.toFixed(2)}</span>
                    </div>
                    <div className="qualia-menu-actions">
                      <button
                        className="btn btn-secondary qualia-save"
                        onClick={() => submitQualiaLabel(m)}
                      >
                        Tag
                      </button>
                      <button
                        className="btn btn-secondary qualia-reward"
                        onClick={() => submitQualiaReward(m, 0.5)}
                        disabled={!getQualiaDraft(m.message_id).labelId}
                      >
                        +Reward
                      </button>
                      <button
                        className="btn btn-secondary qualia-reward"
                        onClick={() => submitQualiaReward(m, -0.5)}
                        disabled={!getQualiaDraft(m.message_id).labelId}
                      >
                        -Reward
                      </button>
                      {getQualiaDraft(m.message_id).savedAt && (
                        <span className="qualia-saved">saved</span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
        <div ref={messagesEndRef} />

        </div>
      </div>

      {eligibleRecommendations.length > 0 && (
        <div className="chat-recommendations glass">
          <div className="chat-recommendations-header">
            <span>Recommended Actions</span>
            <span className="chat-recommendations-count">{eligibleRecommendations.length}</span>
          </div>
          <div className="chat-recommendations-list">
            {eligibleRecommendations.slice(0, 3).map((rec) => (
              <div key={rec.recommendation_id} className="chat-recommendation-item">
                <div className="chat-recommendation-body">
                  <div className="chat-recommendation-title">{rec.title}</div>
                  <div className="chat-recommendation-detail">{rec.detail}</div>
                </div>
                <div className="chat-recommendation-actions">
                  <button className="btn btn-secondary" onClick={() => applyRecommendation(rec)}>
                    Apply
                  </button>
                  <button className="btn btn-tertiary" onClick={() => dismissRecommendation(rec)}>
                    Dismiss
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="input-area">
        <div className="input-meta-row">
          <StatusStrip
            chatState={chatState}
            chatError={chatError}
            voiceStatus={voiceStatus}
            memoryGraphOpen={memoryGraphOpen}
            memoryErrorAt={memoryErrorAt || null}
            onRecoverChat={onRecover}
          />
          {reflectionBadge && (
            <div className={`reflection-badge ${reflectionBadge.tone}`}>
              {reflectionBadge.label}
            </div>
          )}
        </div>
        {chatState === "error" && !chatErrorDismissed && (
          <div className="chat-recovery-banner inline glass">
            <div>
              <strong>Chat stalled.</strong> {chatError || "You can recover and continue."}
            </div>
            <div className="chat-recovery-actions">
              <button className="btn btn-secondary" onClick={onRecover}>
                Recover
              </button>
              <button className="btn btn-secondary" onClick={() => setChatErrorDismissed(true)}>
                Dismiss
              </button>
            </div>
          </div>
        )}
        <div className="input-container">
          <textarea
            className="input chat-input glass"
            placeholder="Message Symbiote..."
            value={input}
            onChange={(e) => onInputChange(e.target.value)}
            ref={inputRef}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                onSend();
              }
            }}
            disabled={isBusy}
          />

          {voiceEnabled ? (
            <VoiceController
              ref={voiceRef}
              onTranscript={(text) => onAppendTranscript(text)}
              onStatusChange={onVoiceStatusChange}
              voiceName={settings?.voice_name || undefined}
              voiceSpeed={settings?.voice_speed || undefined}
              voicePitch={settings?.voice_pitch_semitones || undefined}
              voiceReverb={settings?.voice_reverb_amount || undefined}
              voiceCompression={settings?.voice_compression || undefined}
              voiceFormant={settings?.voice_formant_shift || undefined}
              className="mic-btn"
            />
          ) : (
            <button className="btn btn-secondary mic-btn mic-disabled" disabled type="button">
              Voice Off
            </button>
          )}

          {canStop ? (
            <button className="btn btn-secondary send-btn" onClick={onStop}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" /></svg>
            </button>
          ) : (
            <button className="btn btn-primary send-btn" onClick={onSend} disabled={!input.trim() || isBusy}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m22 2-7 20-4-9-9-4Z" /><path d="M22 2 11 13" /></svg>
            </button>
          )}
        </div>
      </div>
    </>
  );
};
