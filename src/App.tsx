import { useState, useEffect, useRef, useCallback, useMemo, lazy, Suspense } from "react";
import { listen } from "@tauri-apps/api/event";
import { VoiceControllerHandle, VoiceStatus } from "./components/VoiceController";
import { SidebarNav } from "./components/SidebarNav";
import { SystemStatePanel } from "./components/SystemStatePanel";
import { Toast } from "./components/Toast";
import { TitleBar } from "./components/TitleBar";
import { ChatView } from "./views/ChatView";
import { TraceView } from "./views/TraceView";
import { SettingsView } from "./views/SettingsView";
import { OnboardingView, type OnboardingPayload } from "./views/OnboardingView";
import { cleanForTTS } from "./utils/tts";
import { invokeWithTimeout } from "./utils/tauri";
import { applyTheme, DEFAULT_THEME_ID } from "./utils/theme";
import { ConflictView } from "./types/memoryTypes";
import {
  Message,
  PendingClarification,
  PendingPrompt,
  MemoryClarificationRequest,
  MemoryClarifyResult,
  MemoryErrorPayload,
  InnerMonologueEntry,
  SubjectSnapshotEntry,
  GateDecisionEntry,
  ContextTagEntry,
  IntrospectionEntry,
  AuditLogEntry,
  ErrorEventEntry,
  QualiaLabelEntry,
  EvidenceLineageEntry,
  ModuleStatus,
  RollingSummaryStatus,
  Settings,
  SelfInspection,
  SelfModel,
  SystemLogEntry,
  SystemControlEntry,
  SystemControlEvent,
  SystemHealthSnapshot,
  TestResult,
  UserIntentSummary,
  View,
} from "./types/app";
import { SYSTEM_LOG_EVENTS } from "./types/systemLogEvents";

const MemoryGraph3D = lazy(() => import("./components/MemoryGraph3D"));

const sanitizeStreamingToken = (token: string): string => {
  if (!token) return token;
  let cleaned = token
    .replace(/<<<BEGIN_SECTION:[^>]*>>>/g, "")
    .replace(/<<<END_SECTION:[^>]*>>>/g, "");
  cleaned = cleaned.replace(/(^|\n)\s*(Next Steps|Proposed Response)\s*(?=\n|$)/gi, "$1");
  return cleaned;
};

const FEEDBACK_PREFIX_RE = /^\s*(?:\[feedback\]|\(feedback\)|feedback)\s*[:>\-]?\s*/i;
const ensureFeedbackPrefix = (text: string) => {
  if (FEEDBACK_PREFIX_RE.test(text)) return text;
  return `feedback: ${text}`;
};

type ChatState =
  | "idle"
  | "sending"
  | "streaming"
  | "post_processing"
  | "awaiting_conflict"
  | "awaiting_clarification"
  | "awaiting_memory_clarification"
  | "stopping"
  | "error";

const TIMEOUTS = {
  short: 5000,
  medium: 15000,
  long: 30000,
};

const STREAM_SILENCE_MS = 5000;
const STAGE_WATCHDOG_MS: Record<string, number> = {
  ingest_input: 15000,
  prompt_build: 15000,
  memory_retrieval: 25000,
  tool_select: 15000,
  tool_call: 45000,
  llm_wait: 30000,
  streaming: 20000,
  arbitration: 15000,
  grounding: 20000,
  memory_pass: 15000,
  rolling_summary: 15000,
  inner_summary: 15000,
  commit_cycle: 15000,
  thread_run: 45000,
  post_processing: 8000,
  finalize: 10000,
};
const DEFAULT_WATCHDOG_MS = 20000;

const getWatchdogMs = (stage?: string | null) => {
  if (!stage) return DEFAULT_WATCHDOG_MS;
  return STAGE_WATCHDOG_MS[stage] ?? DEFAULT_WATCHDOG_MS;
};

function App() {
  const [view, setView] = useState<View>("chat");
  const [messages, setMessages] = useState<Message[]>([]);
  const voiceRef = useRef<VoiceControllerHandle>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [systemLogs, setSystemLogs] = useState<SystemLogEntry[]>([]);
  const [systemLogError, setSystemLogError] = useState<string | null>(null);
  const [evidenceLineage, setEvidenceLineage] = useState<EvidenceLineageEntry[]>([]);
  const [evidenceLineageError, setEvidenceLineageError] = useState<string | null>(null);
  const [systemControls, setSystemControls] = useState<SystemControlEntry[]>([]);
  const [systemControlEvents, setSystemControlEvents] = useState<SystemControlEvent[]>([]);
  const [systemControlError, setSystemControlError] = useState<string | null>(null);
  const [systemHealthSnapshot, setSystemHealthSnapshot] = useState<SystemHealthSnapshot | null>(null);
  const [systemHealthHistory, setSystemHealthHistory] = useState<SystemHealthSnapshot[]>([]);
  const [systemHealthError, setSystemHealthError] = useState<string | null>(null);
  const [cockpitLastUpdated, setCockpitLastUpdated] = useState<string | null>(null);
  const [gateInputEvents, setGateInputEvents] = useState<SystemLogEntry[]>([]);
  const [subjectSnapshots, setSubjectSnapshots] = useState<SubjectSnapshotEntry[]>([]);
  const [gateDecisions, setGateDecisions] = useState<GateDecisionEntry[]>([]);
  const [contextTags, setContextTags] = useState<ContextTagEntry[]>([]);
  const [intentSummary, setIntentSummary] = useState<UserIntentSummary | null>(null);
  const [introspectionEntries, setIntrospectionEntries] = useState<IntrospectionEntry[]>([]);
  const [auditLog, setAuditLog] = useState<AuditLogEntry[]>([]);
  const [errorEvents, setErrorEvents] = useState<ErrorEventEntry[]>([]);
  const [qualiaLabels, setQualiaLabels] = useState<QualiaLabelEntry[]>([]);
  const [input, setInput] = useState("");
  const [chatState, setChatState] = useState<ChatState>("idle");
  const [chatError, setChatError] = useState<string | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [rollingSummary, setRollingSummary] = useState<string | null>(null);
  const [rollingSummaryError, setRollingSummaryError] = useState<string | null>(null);
  const [rollingSummaryErrorAt, setRollingSummaryErrorAt] = useState<string | null>(null);
  const [rollingSummaryPending, setRollingSummaryPending] = useState(false);
  const [liveSummary, setLiveSummary] = useState<string | null>(null);
  const [liveSummaryError, setLiveSummaryError] = useState<string | null>(null);
  const [liveSummaryErrorAt, setLiveSummaryErrorAt] = useState<string | null>(null);
  const [liveSummaryPending, setLiveSummaryPending] = useState(false);
  const [innerMonologueEntries, setInnerMonologueEntries] = useState<InnerMonologueEntry[]>([]);
  const [innerMonologueError, setInnerMonologueError] = useState<string | null>(null);
  const [selfModel, setSelfModel] = useState<SelfModel | null>(null);
  const [selfInspection, setSelfInspection] = useState<SelfInspection | null>(null);
  const [memoryError, setMemoryError] = useState<{ message: string; timestamp?: string } | null>(null);
  const [pendingClarification, setPendingClarification] = useState<PendingClarification | null>(null);
  const [pendingMemoryClarify, setPendingMemoryClarify] = useState<MemoryClarificationRequest | null>(null);
  const [pendingConflict, setPendingConflict] = useState<{
    conflictId: number;
    options: { label: string; beliefId: number; preview: string }[];
  } | null>(null);
  const pendingClarificationRef = useRef<PendingClarification | null>(null);
  const pendingMemoryClarifyRef = useRef<MemoryClarificationRequest | null>(null);
  const pendingConflictRef = useRef<{
    conflictId: number;
    options: { label: string; beliefId: number; preview: string }[];
  } | null>(null);
  const [isGraphOpen, setIsGraphOpen] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState<VoiceStatus>("loading");
  const [toast, setToast] = useState<{
    message: string;
    type: "success" | "error";
    actionLabel?: string;
    onAction?: () => void;
  } | null>(null);

  // Debug State: Show raw, unfiltered LLM output
  const [showRaw, setShowRaw] = useState(false);
  const [feedbackMode, setFeedbackMode] = useState(false);

  const toastTimerRef = useRef<number | null>(null);

  const showToast = (
    message: string,
    type: "success" | "error" = "success",
    actionLabel?: string,
    onAction?: () => void,
  ) => {
    setToast({ message, type, actionLabel, onAction });
    if (toastTimerRef.current !== null) {
      window.clearTimeout(toastTimerRef.current);
      toastTimerRef.current = null;
    }
    const duration = actionLabel ? 8000 : 3000;
    toastTimerRef.current = window.setTimeout(() => setToast(null), duration);
  };

  const reportError = (
    message: string,
    error: unknown,
    actionLabel?: string,
    onAction?: () => void,
  ) => {
    console.error(message, error);
    showToast(message, "error", actionLabel, onAction);
  };

  useEffect(() => {
    return () => {
      if (toastTimerRef.current !== null) {
        window.clearTimeout(toastTimerRef.current);
        toastTimerRef.current = null;
      }
    };
  }, []);


  const lastMessageIdRef = useRef<string | null>(null);
  const lastSpokenMessageIdRef = useRef<string | null>(null);
  const ttsSpokenCharsRef = useRef<Record<string, number>>({});
  const streamingMessageIdRef = useRef<string | null>(null);
  const streamActiveRef = useRef(false);
  const clearedAtRef = useRef<string | null>(null);
  const isStreamingRef = useRef(false);
  const chatStateRef = useRef<ChatState>("idle");
  const chatErrorAtRef = useRef<number | null>(null);
  const suppressTokensRef = useRef(false);
  const pendingTokenBufferRef = useRef("");
  const tokenFlushTimerRef = useRef<number | null>(null);
  const lastActivityRef = useRef<number>(Date.now());
  const watchdogTimerRef = useRef<number | null>(null);
  const watchdogTriggeredRef = useRef(false);
  const postProcessingStartedRef = useRef<number | null>(null);
  const moduleStatusRef = useRef<ModuleStatus | null>(null);
  const lastPostProcessingRunIdRef = useRef<string | null>(null);
  const lastRunIdRef = useRef<string | null>(null);
  const streamStartAtRef = useRef<number | null>(null);
  const streamFirstTokenLoggedRef = useRef(false);

  const [streamingTokenBuffer, setStreamingTokenBuffer] = useState<string | null>(null);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const [moduleStatus, setModuleStatus] = useState<ModuleStatus | null>(null);
  const [pendingPromptCount, setPendingPromptCount] = useState(0);
  const [pendingPrompts, setPendingPrompts] = useState<PendingPrompt[]>([]);
  const [pendingPromptError, setPendingPromptError] = useState<string | null>(null);
  const systemLogLimitRef = useRef<number>(10);

  const controlModeFor = useCallback((subsystemId: string) => {
    const entry = systemControls.find((item) => item.subsystem_id === subsystemId);
    return (entry?.mode ?? "normal").toLowerCase();
  }, [systemControls]);

  const voiceOutputEnabled = controlModeFor("voice_output") !== "off";

  const recordTtsSpoken = (text: string, messageId?: string | null) => {
    if (!messageId) {
      return;
    }
    const trimmed = text.trim();
    if (!trimmed) {
      return;
    }
    const existing = ttsSpokenCharsRef.current[messageId] ?? 0;
    ttsSpokenCharsRef.current[messageId] = existing + trimmed.length;
  };

  const loadRollingSummary = async () => {
    try {
      const status = await invokeWithTimeout<RollingSummaryStatus>("get_rolling_summary_status", undefined, TIMEOUTS.medium);
      setRollingSummary(status.summary ?? null);
      setRollingSummaryError(status.last_error ?? null);
      setRollingSummaryErrorAt(status.last_error_at ?? null);
      setRollingSummaryPending(status.pending ?? false);
    } catch (e) {
      reportError("Failed to load rolling summary", e);
    }
  };

  const loadLiveSummary = async () => {
    try {
      const status = await invokeWithTimeout<RollingSummaryStatus>("get_live_summary_status", undefined, TIMEOUTS.medium);
      setLiveSummary(status.summary ?? null);
      setLiveSummaryError(status.last_error ?? null);
      setLiveSummaryErrorAt(status.last_error_at ?? null);
      setLiveSummaryPending(status.pending ?? false);
    } catch (e) {
      reportError("Failed to load live summary", e);
    }
  };

  const loadInnerMonologue = async () => {
    try {
      setInnerMonologueError(null);
      const entries = await invokeWithTimeout<InnerMonologueEntry[]>(
        "get_inner_monologue_entries",
        { limit: 50 },
        TIMEOUTS.medium
      );
      setInnerMonologueEntries(entries);
    } catch (e) {
      setInnerMonologueError(String(e));
      reportError("Failed to load inner monologue", e);
    }
  };

  const loadPendingPrompts = useCallback(async () => {
    try {
      setPendingPromptError(null);
      const prompts = await invokeWithTimeout<PendingPrompt[]>(
        "list_pending_prompts",
        { limit: 8 },
        TIMEOUTS.medium
      );
      setPendingPrompts(prompts);
    } catch (e) {
      setPendingPromptError(String(e));
      reportError("Failed to load pending prompts", e);
    }
  }, []);

  const loadPendingPromptCount = useCallback(async () => {
    try {
      const count = await invokeWithTimeout<number>(
        "get_pending_prompt_count",
        undefined,
        TIMEOUTS.short
      );
      if (!Number.isNaN(count)) {
        setPendingPromptCount(count);
        if (count === 0) {
          setPendingPrompts([]);
        }
      }
    } catch (e) {
      reportError("Failed to load pending prompt count", e);
    }
  }, []);

  useEffect(() => {
    if (pendingPromptCount > 0) {
      void loadPendingPrompts();
    }
  }, [pendingPromptCount, loadPendingPrompts]);

  useEffect(() => {
    void loadPendingPromptCount();
    void loadPendingPrompts();
  }, [loadPendingPromptCount, loadPendingPrompts]);


  const handlePendingPromptSend = async (promptId: string) => {
    try {
      await invokeWithTimeout("send_pending_prompt", { promptId }, TIMEOUTS.medium);
      await loadPendingPrompts();
    } catch (e) {
      reportError("Failed to send pending prompt", e);
    }
  };

  const handlePendingPromptDismiss = async (promptId: string) => {
    try {
      await invokeWithTimeout("dismiss_pending_prompt", { promptId }, TIMEOUTS.medium);
      await loadPendingPrompts();
    } catch (e) {
      reportError("Failed to dismiss pending prompt", e);
    }
  };

  const handlePendingPromptRephrase = async (promptId: string, prompt: string) => {
    try {
      await invokeWithTimeout("rephrase_pending_prompt", { promptId, prompt }, TIMEOUTS.medium);
      await loadPendingPrompts();
    } catch (e) {
      reportError("Failed to rephrase pending prompt", e);
    }
  };

  const loadSelfModel = async () => {
    try {
      const model = await invokeWithTimeout<SelfModel>("get_self_model", undefined, TIMEOUTS.medium);
      setSelfModel(model);
    } catch (e) {
      reportError("Failed to load self model", e);
    }
  };

  const loadSelfInspection = async () => {
    try {
      const inspection = await invokeWithTimeout<SelfInspection>("self_inspect", undefined, TIMEOUTS.medium);
      setSelfInspection(inspection);
    } catch (e) {
      reportError("Failed to load self inspection", e);
    }
  };

  const clearTokenFlushTimer = () => {
    if (tokenFlushTimerRef.current !== null) {
      window.clearTimeout(tokenFlushTimerRef.current);
      tokenFlushTimerRef.current = null;
    }
  };

  const flushStreamingTokens = () => {
    if (!streamActiveRef.current) {
      pendingTokenBufferRef.current = "";
      return;
    }
    const pending = pendingTokenBufferRef.current;
    if (!pending) {
      return;
    }
    pendingTokenBufferRef.current = "";

    setStreamingTokenBuffer((prev) => (prev || "") + pending);
    setMessages((prev) => {
      let streamId = streamingMessageIdRef.current;
      if (!streamId) {
        const existing = [...prev].reverse().find((msg) => msg.role === "assistant" && msg.status === "streaming");
        if (existing) {
          streamId = existing.message_id;
          streamingMessageIdRef.current = streamId;
          lastMessageIdRef.current = streamId;
          setStreamingMessageId(streamId);
        }
      }

      if (!streamId) {
        const placeholderId = `temp_stream_${Date.now()}`;
        streamingMessageIdRef.current = placeholderId;
        lastMessageIdRef.current = placeholderId;
        setStreamingMessageId(placeholderId);
        const placeholderMessage: Message = {
          message_id: placeholderId,
          role: "assistant",
          content: pending,
          status: "streaming",
        };
        return [
          ...prev,
          placeholderMessage,
        ];
      }

      let found = false;
      const next = prev.map((msg) => {
        if (msg.message_id === streamId) {
          found = true;
          return { ...msg, content: msg.content + pending, status: "streaming" as Message["status"] };
        }
        return msg;
      });
      if (!found) {
        const streamMessage: Message = {
          message_id: streamId,
          role: "assistant",
          content: pending,
          status: "streaming",
        };
        return [
          ...next,
          streamMessage,
        ];
      }
      return next;
    });
  };

  const scheduleTokenFlush = () => {
    if (tokenFlushTimerRef.current !== null) {
      return;
    }
    tokenFlushTimerRef.current = window.setTimeout(() => {
      tokenFlushTimerRef.current = null;
      flushStreamingTokens();
    }, 50);
  };


  useEffect(() => {
    pendingConflictRef.current = pendingConflict;
  }, [pendingConflict]);

  useEffect(() => {
    pendingClarificationRef.current = pendingClarification;
  }, [pendingClarification]);

  useEffect(() => {
    pendingMemoryClarifyRef.current = pendingMemoryClarify;
  }, [pendingMemoryClarify]);

  useEffect(() => {
    chatStateRef.current = chatState;
    isStreamingRef.current = chatState === "sending" || chatState === "streaming";
    if (import.meta.env.DEV) {
      console.log("[UI] Chat state:", chatState);
    }
  }, [chatState]);

  useEffect(() => {
    if (import.meta.env.DEV) {
      console.log("[UI] Voice status:", voiceStatus);
    }
  }, [voiceStatus]);

  useEffect(() => {
    if (!voiceOutputEnabled) {
      voiceRef.current?.stopTts("voice_output_disabled");
    }
  }, [voiceOutputEnabled]);

  useEffect(() => {
    if (import.meta.env.DEV) {
      console.log("[UI] Memory graph open:", isGraphOpen);
    }
  }, [isGraphOpen]);

  useEffect(() => {
    if (!settings) return;
    void applyTheme(settings.ui_theme || DEFAULT_THEME_ID);
  }, [settings?.ui_theme]);

  useEffect(() => {
    systemLogLimitRef.current = settings?.trace_history_limit ?? 10;
  }, [settings?.trace_history_limit]);

  useEffect(() => {
    moduleStatusRef.current = moduleStatus;
  }, [moduleStatus]);

  useEffect(() => {
    if (chatState === "post_processing") {
      if (postProcessingStartedRef.current === null) {
        postProcessingStartedRef.current = Date.now();
        void invokeWithTimeout("log_ui_timing", {
          event: "post_processing_entered",
          duration_ms: 0,
          detail: lastPostProcessingRunIdRef.current ?? undefined,
          run_id: lastPostProcessingRunIdRef.current ?? lastRunIdRef.current ?? undefined,
          message_id: lastMessageIdRef.current ?? streamingMessageIdRef.current ?? undefined,
        }, TIMEOUTS.short);
      }
    } else if (postProcessingStartedRef.current !== null) {
      const durationMs = Date.now() - postProcessingStartedRef.current;
      postProcessingStartedRef.current = null;
      void invokeWithTimeout("log_ui_timing", {
        event: "post_processing_exited",
        duration_ms: durationMs,
        detail: lastPostProcessingRunIdRef.current ?? undefined,
        run_id: lastPostProcessingRunIdRef.current ?? lastRunIdRef.current ?? undefined,
        message_id: lastMessageIdRef.current ?? streamingMessageIdRef.current ?? undefined,
      }, TIMEOUTS.short);
    }
  }, [chatState]);

  useEffect(() => {
    const isActive = chatState === "sending" || chatState === "streaming" || chatState === "post_processing";
    if (!isActive) {
      if (watchdogTimerRef.current !== null) {
        window.clearInterval(watchdogTimerRef.current);
        watchdogTimerRef.current = null;
      }
      watchdogTriggeredRef.current = false;
      return;
    }

    if (watchdogTimerRef.current !== null) {
      window.clearInterval(watchdogTimerRef.current);
    }
    watchdogTimerRef.current = window.setInterval(() => {
      const idleMs = Date.now() - lastActivityRef.current;
      if (chatStateRef.current === "streaming" && idleMs > STREAM_SILENCE_MS) {
        if (import.meta.env.DEV) {
          console.log("[UI] Stream silence detected. Switching to post-processing.");
        }
        void invokeWithTimeout("log_ui_timing", {
          event: "stream_silence",
          duration_ms: idleMs,
          detail: lastPostProcessingRunIdRef.current ?? undefined,
          run_id: lastPostProcessingRunIdRef.current ?? lastRunIdRef.current ?? undefined,
          message_id: lastMessageIdRef.current ?? streamingMessageIdRef.current ?? undefined,
        }, TIMEOUTS.short);
        showToast("Streaming stalled. Finalizing response.", "error", "Recover", handleRecover);
        setChatState("post_processing");
        return;
      }
      if (
        chatStateRef.current === "post_processing"
        && idleMs > getWatchdogMs("post_processing")
        && !moduleStatusRef.current
      ) {
        void invokeWithTimeout("log_ui_timing", {
          event: "post_processing_timeout",
          duration_ms: idleMs,
          detail: lastPostProcessingRunIdRef.current ?? undefined,
          run_id: lastPostProcessingRunIdRef.current ?? lastRunIdRef.current ?? undefined,
          message_id: lastMessageIdRef.current ?? streamingMessageIdRef.current ?? undefined,
        }, TIMEOUTS.short);
        setChatState("idle");
        setChatError(null);
        setModuleStatus(null);
        console.warn("[UI] Post-processing stalled. Input unlocked.");
        showToast("Post-processing stalled. Input unlocked.", "error", "Recover", handleRecover);
        return;
      }
      const stage = moduleStatusRef.current?.stage
        || (chatStateRef.current === "post_processing" ? "post_processing" : null);
      const watchdogMs = getWatchdogMs(stage);
      if (idleMs > watchdogMs && !watchdogTriggeredRef.current) {
        watchdogTriggeredRef.current = true;
        suppressTokensRef.current = true;
        chatErrorAtRef.current = Date.now();
        setChatError("No response detected. You can recover and continue.");
        setChatState("error");
        showToast("No response from the model.", "error", "Recover", handleRecover);
      }
    }, 1000);

    return () => {
      if (watchdogTimerRef.current !== null) {
        window.clearInterval(watchdogTimerRef.current);
        watchdogTimerRef.current = null;
      }
    };
  }, [chatState]);

  useEffect(() => {
    loadSettings();
    loadMessages();

    const unlistenUpdate = listen("message_updated", () => {
      lastActivityRef.current = Date.now();
      watchdogTriggeredRef.current = false;
      flushStreamingTokens();
      clearTokenFlushTimer();
      loadMessages();
    });
    const unlistenEmpty = listen("response_empty_error", () => {
      setChatError("Model returned an empty response. Please retry.");
      setChatState("error");
      showToast("Model returned an empty response.", "error", "Recover", handleRecover);
    });
    const unlistenToken = listen<string>("token", (event) => {
      if (suppressTokensRef.current) {
        return;
      }
      if (!streamActiveRef.current && streamingMessageIdRef.current === null) {
        streamActiveRef.current = true;
      }
      lastActivityRef.current = Date.now();
      watchdogTriggeredRef.current = false;
      const sanitizedToken = sanitizeStreamingToken(event.payload);
      if (!sanitizedToken) {
        return;
      }
      if (!streamFirstTokenLoggedRef.current) {
        streamFirstTokenLoggedRef.current = true;
        const startedAt = streamStartAtRef.current;
        if (startedAt) {
          void invokeWithTimeout("log_ui_timing", {
            event: "stream_first_token",
            duration_ms: Date.now() - startedAt,
            detail: streamingMessageIdRef.current ?? undefined,
            run_id: lastPostProcessingRunIdRef.current ?? lastRunIdRef.current ?? undefined,
            message_id: streamingMessageIdRef.current ?? undefined,
          }, TIMEOUTS.short);
        }
      }
      if (chatStateRef.current === "sending") {
        setChatState("streaming");
      }

      if (streamingMessageIdRef.current === null) {
        console.log("[Token listener] Starting new stream sequence");
        pendingTokenBufferRef.current = "";
        clearTokenFlushTimer();
        setStreamingTokenBuffer("");
        const placeholderId = `temp_stream_${Date.now()}`;
        streamingMessageIdRef.current = placeholderId;
        lastMessageIdRef.current = placeholderId;
        setStreamingMessageId(placeholderId);
      }

      pendingTokenBufferRef.current += sanitizedToken;
      scheduleTokenFlush();
    });

    const unlistenStreamEnd = listen<{ run_id?: string; content_len?: number }>("stream_end", (event) => {
      lastActivityRef.current = Date.now();
      watchdogTriggeredRef.current = false;
      flushStreamingTokens();
      clearTokenFlushTimer();
      if (event.payload?.run_id) {
        lastPostProcessingRunIdRef.current = event.payload.run_id;
        lastRunIdRef.current = event.payload.run_id;
      }
      streamActiveRef.current = false;
      if (chatStateRef.current === "sending" || chatStateRef.current === "streaming") {
        setChatState("post_processing");
      }
    });

    const unlistenActivation = listen<string>("assistant_message", (event) => {
      setMessages((prev) => [...prev, {
        message_id: `activation_${Date.now()}`,
        role: "assistant",
        content: event.payload,
        status: "complete"
      }]);
    });

    const unlistenSystemLog = listen<SystemLogEntry>("system_log", (event) => {
      const eventName = (event.payload as any)?.payload?.event;
      if (eventName && !SYSTEM_LOG_EVENTS.includes(eventName)) {
        if (import.meta.env.DEV) {
          console.warn("[UI] Dropping unknown system_log event:", eventName);
        }
        return;
      }
      setSystemLogs(prev => {
        const limit = systemLogLimitRef.current;
        const next = [event.payload, ...prev.filter(item => item.id !== event.payload.id)];
        return next.slice(0, limit);
      });
    });

    const unlistenInnerMonologue = listen<InnerMonologueEntry>("inner_monologue", (event) => {
      setInnerMonologueEntries(prev => [event.payload, ...prev.filter(item => item.id !== event.payload.id)]);
    });

    const unlistenModuleStatus = listen<{ stage?: string; detail?: string; started_at?: string; duration_ms?: number }>("module_status", (event) => {
      const stage = event.payload?.stage || "";
      if (!stage || stage === "idle") {
        setModuleStatus(null);
        return;
      }
      setModuleStatus({
        stage,
        detail: event.payload?.detail ?? null,
        started_at: event.payload?.started_at ?? null,
        duration_ms: typeof event.payload?.duration_ms === "number" ? event.payload?.duration_ms : null,
      });
    });

    const unlistenPromptCount = listen<number>("pending_prompt_count", (event) => {
      const count = typeof event.payload === "number" ? event.payload : 0;
      setPendingPromptCount(count);
      if (count === 0) {
        setPendingPrompts([]);
      }
    });

    const unlistenRollingSummary = listen<string>("rolling_summary_updated", () => {
      loadRollingSummary();
    });

    const unlistenRollingSummaryError = listen<string>("rolling_summary_error", () => {
      loadRollingSummary();
    });

    const unlistenLiveSummary = listen<string>("live_summary_updated", () => {
      loadLiveSummary();
    });

    const unlistenLiveSummaryError = listen<string>("live_summary_error", () => {
      loadLiveSummary();
    });

    const unlistenClarification = listen<{ question: string; run_id: string; original_input: string }>("clarification_request", (event) => {
      console.log("[Clarification] Question:", event.payload.question);
      setPendingClarification({
        question: event.payload.question,
        runId: event.payload.run_id,
        originalInput: event.payload.original_input
      });
      suppressTokensRef.current = true;
      setStreamingTokenBuffer(null);
      setChatError(null);
      setChatState("awaiting_clarification");
    });

    const unlistenConflict = listen<number[]>("memory_conflict", async (event) => {
      if (pendingConflictRef.current) {
        return;
      }
      try {
        const conflicts = await invokeWithTimeout<ConflictView[]>("memory_list_conflicts", undefined, TIMEOUTS.medium);
        const target = conflicts.find(conflict => event.payload.includes(conflict.id));
        if (!target || target.members.length === 0) {
          return;
        }

        const options = target.members.map((member, index) => ({
          label: String.fromCharCode(65 + index),
          beliefId: member.belief_id,
          preview: `${member.preview} (${member.polarity})`,
          evidenceSnippet: member.evidence_snippet || null,
          observedAt: member.observed_at || null,
        }));
        const optionList = options.map(option => {
          const evidence = option.evidenceSnippet ? `\n   evidence: ${option.evidenceSnippet}` : "";
          const observed = option.observedAt ? `\n   observed: ${option.observedAt}` : "";
          return `${option.label}) ${option.preview}${evidence}${observed}`;
        }).join("\n");
        const prompt = `I found conflicting memories. Which is correct?\n${optionList}\nReply with ${options.map(option => option.label).join(", ")}, or \"keep both\".`;

        setPendingConflict({ conflictId: target.id, options });
        suppressTokensRef.current = true;
        setStreamingTokenBuffer(null);
        setChatError(null);
        setChatState("awaiting_conflict");
        setMessages(prev => [
          ...prev,
          {
            message_id: `conflict_${Date.now()}`,
            role: "assistant",
            content: prompt,
            status: "complete"
          }
        ]);
      } catch (e) {
        reportError("Failed to load conflicts", e);
      }
    });

    const unlistenMemoryError = listen<MemoryErrorPayload>("memory_error", (event) => {
      const errors = event.payload.errors || [];
      const message = errors.length > 0 ? errors.join(" | ") : "Memory write failed.";
      setMemoryError({ message, timestamp: event.payload.timestamp });
      loadSelfInspection();
    });

    const unlistenMemoryClarify = listen<MemoryClarificationRequest>("memory_clarify", (event) => {
      if (pendingConflictRef.current || pendingMemoryClarifyRef.current || pendingClarificationRef.current) {
        return;
      }

      const payload = event.payload;
      if (!payload.candidates || payload.candidates.length === 0) {
        return;
      }

      const options = payload.candidates.map((candidate, index) => ({
        label: String.fromCharCode(65 + index),
        entityId: candidate.entity_id,
        labelText: candidate.label,
        context: candidate.context
      }));
      const optionList = options
        .map(option => {
          const context = option.context ? ` (${option.context})` : "";
          return `${option.label}) ${option.labelText}${context}`;
        })
        .join("\n");
      const prompt = `${payload.question}\n${optionList}\nReply with ${options.map(option => option.label).join(", ")}, or "new: <name>" to create, or "cancel".`;

      setPendingMemoryClarify(payload);
      suppressTokensRef.current = true;
      setStreamingTokenBuffer(null);
      setChatError(null);
      setChatState("awaiting_memory_clarification");
      setMessages(prev => [
        ...prev,
        {
          message_id: `memory_clarify_${Date.now()}`,
          role: "assistant",
          content: prompt,
          status: "complete"
        }
      ]);
    });

    const unlistenReminder = listen<any>("reminder_triggered", async (event) => {
      console.log("[App] Reminder Triggered:", event.payload);
      // Automatically trigger an agent response contextually
      try {
        await invokeWithTimeout("trigger_reminder_response", {
          reminderId: event.payload.id,
          content: event.payload.content
        }, TIMEOUTS.medium);
      } catch (e) {
        reportError("Failed to trigger reminder response", e);
      }
    });

    return () => {
      unlistenUpdate.then((u) => u());
      unlistenEmpty.then((u) => u());
      unlistenToken.then((u) => u());
      unlistenStreamEnd.then((u) => u());
      unlistenActivation.then((u) => u());

      unlistenClarification.then((u) => u());
      unlistenConflict.then((u) => u());
      unlistenMemoryError.then((u) => u());
      unlistenMemoryClarify.then((u) => u());
      unlistenSystemLog.then((u) => u());
      unlistenInnerMonologue.then((u) => u());
      unlistenModuleStatus.then((u) => u());
      unlistenPromptCount.then((u) => u());
      unlistenRollingSummary.then((u) => u());
      unlistenRollingSummaryError.then((u) => u());
      unlistenLiveSummary.then((u) => u());
      unlistenLiveSummaryError.then((u) => u());
      unlistenReminder.then((u) => u());
    };
  }, []);

  useEffect(() => {
    const lastMessage = messages[messages.length - 1];
    const streamId = streamingMessageIdRef.current;
    const streamMessage = streamId ? messages.find((msg) => msg.message_id === streamId) : null;
    const isAwaiting = Boolean(pendingConflict || pendingClarification || pendingMemoryClarify);
    if (streamMessage && streamMessage.role === "assistant" && streamMessage.status === "complete") {
      streamingMessageIdRef.current = null;
      setStreamingMessageId(null);
      streamActiveRef.current = false;
      setStreamingTokenBuffer(null);
    }
    if (lastMessage && lastMessage.role === "assistant" && lastMessage.status === "error") {
      streamActiveRef.current = false;
      streamingMessageIdRef.current = null;
      setStreamingMessageId(null);
    }
    if (lastMessage && lastMessage.role === "assistant" && lastMessage.status === "complete") {
      if (streamingMessageIdRef.current === lastMessage.message_id) {
        streamingMessageIdRef.current = null;
        setStreamingMessageId(null);
      }
      streamActiveRef.current = false;
      setStreamingTokenBuffer(null);
      if (!isAwaiting && (chatState === "sending" || chatState === "streaming" || chatState === "stopping" || chatState === "post_processing")) {
        setChatState("idle");
        setChatError(null);
        setModuleStatus(null);
      }

      let trackedId = lastMessageIdRef.current;
      if (trackedId && trackedId.startsWith("temp_stream_") && lastMessage.role === "assistant") {
        const spokenChars = ttsSpokenCharsRef.current[trackedId];
        if (typeof spokenChars === "number") {
          const existing = ttsSpokenCharsRef.current[lastMessage.message_id] ?? 0;
          ttsSpokenCharsRef.current[lastMessage.message_id] = existing + spokenChars;
          delete ttsSpokenCharsRef.current[trackedId];
        }
        trackedId = lastMessage.message_id;
      }

      const surfaceFlag = lastMessage.metadata?.surface;
      const originFlag = lastMessage.metadata?.origin;
      const responseOrigin = lastMessage.metadata?.response_origin;
      const isFinalMessage = lastMessage.status === "complete"
        && streamingMessageIdRef.current !== lastMessage.message_id
        && !streamActiveRef.current;
      const isSpeakableOrigin = responseOrigin === "primary"
        || responseOrigin === "fallback"
        || responseOrigin === null
        || responseOrigin === undefined;
      const shouldReflect = lastMessage.role === "assistant"
        && surfaceFlag !== false
        && originFlag !== "monologue"
        && originFlag !== "pending_prompt"
        && isSpeakableOrigin
        && isFinalMessage;

      if (lastMessage.message_id === trackedId) {
        if (shouldReflect) {
          // Avatar uses system health snapshots; no assistant-emotion classification.
        }
      } else {
        if (lastMessage.content.trim().length > 0) {
          lastMessageIdRef.current = lastMessage.message_id;
          if (shouldReflect) {
            // Avatar uses system health snapshots; no assistant-emotion classification.
          }

          // LEGACY: [[TASK:CREATE...]] tag format (backward compatibility)
          const tagMatch = lastMessage.content.match(/\[\[TASK:CREATE\s+([^\]]+)\]\]/);
          if (tagMatch) {
            const inner = tagMatch[1];
            const extract = (key: string) => {
              const m = inner.match(new RegExp(`${key}=['"](.*?)['"]`));
              return m ? m[1] : null;
            };

            const content = extract("content");
            const dueIn = extract("due_in");
            const type = extract("type");

            if (content && dueIn && type) {
              console.log("[App] Found Task Tag (Legacy). Invoking Backend...", { content, dueIn, type });
              invokeWithTimeout("create_reminder", {
                content,
                dueIn,
                reminderType: type
              }, TIMEOUTS.long)
                .then((id) => {
                  console.log("[App] Reminder Created Successfully! ID:", id);
                  showToast(`Reminder set: "${content}" (in ${dueIn})`);
                })
                .catch(e => {
                  console.error("[App] FAILED to create reminder:", e);
                  showToast(`Failed to set reminder: ${e}`, "error");
                });
            } else {
              console.warn("[App] Found Task Tag but failed to extract all fields:", { inner, content, dueIn, type });
            }
          }
        }
      }

      if (trackedId) {
        const alreadySpoken = lastSpokenMessageIdRef.current === lastMessage.message_id
          || (ttsSpokenCharsRef.current[lastMessage.message_id] ?? 0) > 0;
        if (shouldReflect && !alreadySpoken && voiceOutputEnabled) {
          const raw = lastMessage.content.trim();
          if (raw.length > 0) {
            const cleaned = cleanForTTS(raw);
            const speakText = cleaned.trim().length > 0 ? cleaned : raw;
            voiceRef.current?.speak(speakText, {
              messageId: lastMessage.message_id,
              runId: lastMessage.run_id ?? lastPostProcessingRunIdRef.current ?? lastRunIdRef.current ?? undefined,
            });
            recordTtsSpoken(speakText, lastMessage.message_id);
            lastSpokenMessageIdRef.current = lastMessage.message_id;
          }
        }
        const spokenChars = ttsSpokenCharsRef.current[trackedId];
        if (typeof spokenChars === "number") {
          const totalChars = lastMessage.content.trim().length;
          const ratio = totalChars > 0 ? Math.min(spokenChars / totalChars, 1) : 0;
          void invokeWithTimeout("log_ui_timing", {
            event: "tts_completion",
            duration_ms: 0,
            detail: `ratio=${ratio.toFixed(3)} spoken=${spokenChars} total=${totalChars}`,
            run_id: lastMessage.run_id ?? lastPostProcessingRunIdRef.current ?? lastRunIdRef.current ?? undefined,
            message_id: lastMessage.message_id,
          }, TIMEOUTS.short).catch((err) => {
            const errorDetail = err instanceof Error ? err.message : String(err);
            void invokeWithTimeout("log_tts_event", {
              event: "ui_log_failed",
              duration_ms: 0,
              detail: errorDetail,
              run_id: lastMessage.run_id ?? lastPostProcessingRunIdRef.current ?? lastRunIdRef.current ?? undefined,
              message_id: lastMessage.message_id,
            }, TIMEOUTS.short).catch(() => {});
            void invokeWithTimeout("log_tts_event", {
              event: "tts_completion",
              duration_ms: 0,
              detail: `ratio=${ratio.toFixed(3)} spoken=${spokenChars} total=${totalChars} | ui_log_failed=${errorDetail}`,
              run_id: lastMessage.run_id ?? lastPostProcessingRunIdRef.current ?? lastRunIdRef.current ?? undefined,
              message_id: lastMessage.message_id,
            }, TIMEOUTS.short).catch(() => {});
          });
          delete ttsSpokenCharsRef.current[trackedId];
        }
      }
    }
  }, [messages, pendingConflict, pendingClarification, pendingMemoryClarify, chatState, voiceOutputEnabled]);

  useEffect(() => {
    if (view === "settings") {
      loadSettings();
      loadSelfModel();
      loadSelfInspection();
    }
  }, [view]);

  const loadSettings = async () => {
    try {
      setSettingsError(null);
      const s = await invokeWithTimeout<Settings>("get_settings", undefined, TIMEOUTS.medium);
      setSettings(s);
    } catch (e) {
      setSettingsError(String(e));
      reportError("Failed to load settings", e, "Retry", () => loadSettings());
    }
  };

  const loadMessages = async () => {
    try {
      const msgs = await invokeWithTimeout<Message[]>("get_messages", undefined, TIMEOUTS.long);
      const clearedAt = clearedAtRef.current;
      const filtered = clearedAt
        ? msgs.filter((msg) => {
          if (!msg.created_at) return true;
          const created = Date.parse(msg.created_at);
          if (Number.isNaN(created)) return true;
          return created > Date.parse(clearedAt);
        })
        : msgs;
      const serverStreaming = [...filtered].reverse().find((msg) => msg.role === "assistant" && msg.status === "streaming");
      setMessages((prev) => {
        const streamId = serverStreaming?.message_id ?? streamingMessageIdRef.current;
        if (!streamActiveRef.current && !streamId) {
          return filtered;
        }
        const prevById = new Map(prev.map((msg) => [msg.message_id, msg]));
        const next = filtered.map((msg) => {
          if (streamId && msg.message_id === streamId && msg.status === "streaming") {
            const prevMsg = prevById.get(streamId);
            if (prevMsg && prevMsg.content.length >= msg.content.length) {
              return { ...msg, content: prevMsg.content, status: "streaming" as Message["status"] };
            }
          }
          return msg;
        });

        const existingIds = new Set(next.map((msg) => msg.message_id));
        if (streamId && !existingIds.has(streamId)) {
          const prevMsg = prevById.get(streamId);
          if (prevMsg) {
            next.push(prevMsg);
          }
        } else if (!streamId) {
          const prevStreaming = prev.find((msg) => msg.role === "assistant" && msg.status === "streaming");
          if (prevStreaming && !existingIds.has(prevStreaming.message_id)) {
            next.push(prevStreaming);
          }
        }
        return next;
      });
      const lastWithRunId = [...filtered].reverse().find((msg) => msg.run_id);
      if (lastWithRunId?.run_id) {
        lastRunIdRef.current = lastWithRunId.run_id;
      }
      if (chatStateRef.current === "error") {
        const lastAssistant = [...filtered]
          .reverse()
          .find((msg) => msg.role === "assistant" && msg.status === "complete");
        const lastAssistantAt = lastAssistant?.created_at ? Date.parse(lastAssistant.created_at) : NaN;
        if (lastAssistant && !Number.isNaN(lastAssistantAt)) {
          const errorAt = chatErrorAtRef.current ?? 0;
          if (lastAssistantAt > errorAt) {
            setChatState("idle");
            setChatError(null);
            setModuleStatus(null);
            suppressTokensRef.current = false;
            chatErrorAtRef.current = null;
          }
        }
      }
      if (serverStreaming) {
        streamingMessageIdRef.current = serverStreaming.message_id;
        setStreamingMessageId(serverStreaming.message_id);
        streamActiveRef.current = true;
        if (chatStateRef.current !== "post_processing") {
          setChatState("streaming");
        }
      } else if (
        chatStateRef.current === "sending"
        || chatStateRef.current === "streaming"
        || chatStateRef.current === "stopping"
        || chatStateRef.current === "post_processing"
      ) {
        if (!streamActiveRef.current) {
          setChatState("idle");
          setChatError(null);
          setModuleStatus(null);
        }
      }
    } catch (e) {
      reportError("Failed to load messages", e);
    }
  };

  const loadSystemLogs = async (limitOverride?: number) => {
    try {
      setSystemLogError(null);
      const limit = limitOverride ?? settings?.trace_history_limit ?? 10;
      const [
        logs,
        snapshots,
        gates,
        introspection,
        audits,
        errors,
        qualia,
        tags,
        intent,
      ] = await Promise.all([
        invokeWithTimeout<SystemLogEntry[]>("get_system_logs", { limit }, TIMEOUTS.medium),
        invokeWithTimeout<SubjectSnapshotEntry[]>("get_subject_snapshots", { limit }, TIMEOUTS.medium),
        invokeWithTimeout<GateDecisionEntry[]>("get_gate_decisions", { limit }, TIMEOUTS.medium),
        invokeWithTimeout<IntrospectionEntry[]>("get_introspection_entries", { limit }, TIMEOUTS.medium),
        invokeWithTimeout<AuditLogEntry[]>("get_audit_log", { limit }, TIMEOUTS.medium),
        invokeWithTimeout<ErrorEventEntry[]>("get_error_events", { limit }, TIMEOUTS.medium),
        invokeWithTimeout<QualiaLabelEntry[]>("get_qualia_labels", { limit }, TIMEOUTS.medium),
        invokeWithTimeout<ContextTagEntry[]>("get_context_tags", { ttl_minutes: 120 }, TIMEOUTS.medium),
        invokeWithTimeout<UserIntentSummary | null>("get_user_intent_summary", {}, TIMEOUTS.medium),
      ]);
      setSystemLogs(logs);
      setGateInputEvents(
        logs.filter((entry) => (entry.payload as any)?.event === "gate_decision_inputs").slice(0, 6)
      );
      setSubjectSnapshots(snapshots);
      setGateDecisions(gates);
      setIntrospectionEntries(introspection);
      setAuditLog(audits);
      setErrorEvents(errors);
      setQualiaLabels(qualia);
      setContextTags(tags);
      setIntentSummary(intent ?? null);
    } catch (e) {
      setSystemLogError(String(e));
      reportError("Failed to load system logs", e);
    }
  };

  const loadEvidenceLineage = async (limitOverride?: number) => {
    try {
      setEvidenceLineageError(null);
      const limit = limitOverride ?? settings?.trace_history_limit ?? 50;
      const lineage = await invokeWithTimeout<EvidenceLineageEntry[]>(
        "get_evidence_lineage",
        { limit },
        TIMEOUTS.medium
      );
      setEvidenceLineage(lineage);
    } catch (e) {
      setEvidenceLineageError(String(e));
      reportError("Failed to load evidence lineage", e);
    }
  };

  const loadSystemControls = useCallback(async () => {
    try {
      setSystemControlError(null);
      const controls = await invokeWithTimeout<SystemControlEntry[]>(
        "get_system_controls",
        undefined,
        TIMEOUTS.short
      );
      setSystemControls(controls);
    } catch (e) {
      setSystemControlError(String(e));
      reportError("Failed to load system controls", e);
    }
  }, []);

  const loadSystemControlEvents = useCallback(async () => {
    try {
      const events = await invokeWithTimeout<SystemControlEvent[]>(
        "get_system_control_events",
        { limit: 200 },
        TIMEOUTS.short
      );
      setSystemControlEvents(events);
    } catch (e) {
      reportError("Failed to load control events", e);
    }
  }, []);

  const loadSystemHealthSnapshot = useCallback(async () => {
    try {
      setSystemHealthError(null);
      const snapshot = await invokeWithTimeout<SystemHealthSnapshot>(
        "get_system_health_snapshot",
        undefined,
        TIMEOUTS.medium
      );
      setSystemHealthSnapshot(snapshot);
      setCockpitLastUpdated(new Date().toISOString());
    } catch (e) {
      setSystemHealthError(String(e));
      reportError("Failed to load system health snapshot", e);
    }
  }, []);

  const loadSystemHealthHistory = useCallback(async () => {
    try {
      const history = await invokeWithTimeout<SystemHealthSnapshot[]>(
        "get_system_health_history",
        { limit: 30 },
        TIMEOUTS.medium
      );
      setSystemHealthHistory(history);
    } catch (e) {
      reportError("Failed to load system health history", e);
    }
  }, []);

  useEffect(() => {
    void loadSystemControls();
    void loadSystemHealthSnapshot();
    void loadSystemHealthHistory();
  }, [loadSystemControls, loadSystemHealthSnapshot, loadSystemHealthHistory]);

  const loadCockpitData = useCallback(async () => {
    await Promise.all([
      loadSystemLogs(),
      loadEvidenceLineage(),
      loadSystemControls(),
      loadSystemControlEvents(),
      loadSystemHealthSnapshot(),
      loadSystemHealthHistory(),
    ]);
  }, [loadSystemControls, loadSystemControlEvents, loadSystemHealthSnapshot, loadSystemHealthHistory, settings?.trace_history_limit]);

  useEffect(() => {
    const mode = controlModeFor("ui_live_refresh");
    if (mode === "off") {
      return;
    }
    const intervalMs = mode === "degraded" ? 20000 : 8000;
    const timer = window.setInterval(() => {
      if (view === "trace") {
        void loadCockpitData();
      } else {
        void loadSystemHealthSnapshot();
      }
    }, intervalMs);
    return () => {
      window.clearInterval(timer);
    };
  }, [controlModeFor, loadSystemHealthSnapshot, loadCockpitData, view]);

  const handleRefreshSelfData = async () => {
    await loadSelfModel();
    await loadSelfInspection();
  };

  const handleConflictResolution = async (content: string): Promise<boolean> => {
    if (!pendingConflict) return true;

    const normalized = content.trim().toLowerCase();
    const keepBoth = normalized === "keep both" || normalized === "keep" || normalized === "both";
    if (keepBoth) {
      await invokeWithTimeout("memory_resolve_conflict", {
        conflictId: pendingConflict.conflictId,
        action: "keep_both",
        resolutionNote: "User chose to keep both.",
        userReply: content
      }, TIMEOUTS.long);
      setPendingConflict(null);
      setMessages(prev => [
        ...prev,
        {
          message_id: `conflict_resolved_${Date.now()}`,
          role: "assistant",
          content: "Got it. I will keep both memories.",
          status: "complete"
        }
      ]);
      return true;
    }

    const label = normalized.slice(0, 1).toUpperCase();
    const selected = pendingConflict.options.find(option => option.label === label);
    if (!selected) {
      setMessages(prev => [
        ...prev,
        {
          message_id: `conflict_retry_${Date.now()}`,
          role: "assistant",
          content: `Please reply with ${pendingConflict.options.map(option => option.label).join(", ")}, or \"keep both\".`,
          status: "complete"
        }
      ]);
      return false;
    }

    await invokeWithTimeout("memory_resolve_conflict", {
      conflictId: pendingConflict.conflictId,
      action: "pick_winner",
      winnerBeliefId: selected.beliefId,
      resolutionNote: `User selected ${label}.`,
      userReply: content
    }, TIMEOUTS.long);
    setPendingConflict(null);
    setMessages(prev => [
      ...prev,
      {
        message_id: `conflict_resolved_${Date.now()}`,
        role: "assistant",
        content: `Thanks. I will treat ${label} as the correct memory.`,
        status: "complete"
      }
    ]);
    return true;
  };

  const handleSend = async () => {
    const isBusy =
      chatState === "sending"
      || chatState === "streaming"
      || chatState === "stopping"
      || chatState === "post_processing";
    if (!input.trim() || isBusy) return;
    voiceRef.current?.stopTts("user_send");
    streamingMessageIdRef.current = null;
    setStreamingMessageId(null);
    streamActiveRef.current = false;
    pendingTokenBufferRef.current = "";
    clearTokenFlushTimer();
    setStreamingTokenBuffer("");

    const rawContent = input;
    const explicitFeedback = feedbackMode;
    if (explicitFeedback) {
      setFeedbackMode(false);
    }
    setInput("");
    try {
      if (pendingConflict) {
        suppressTokensRef.current = false;
        setChatError(null);
        setChatState("sending");
        lastActivityRef.current = Date.now();
        watchdogTriggeredRef.current = false;
        const resolved = await handleConflictResolution(rawContent);
        setChatState(resolved ? "idle" : "awaiting_conflict");
        return;
      }
      if (pendingMemoryClarify) {
        suppressTokensRef.current = false;
        setChatError(null);
        setChatState("sending");
        lastActivityRef.current = Date.now();
        watchdogTriggeredRef.current = false;
        try {
          const result = await invokeWithTimeout<MemoryClarifyResult>("memory_resolve_clarify", {
            pendingId: pendingMemoryClarify.pending_id,
            reply: rawContent,
            source: "user"
          }, TIMEOUTS.long);
          if (!result.success) {
            setMessages(prev => [
              ...prev,
              {
                message_id: `memory_clarify_failed_${Date.now()}`,
                role: "assistant",
                content: result.error || "I couldn't resolve that clarification yet. Please try again.",
                status: "complete"
              }
            ]);
            setChatState("awaiting_memory_clarification");
            return;
          }
          setPendingMemoryClarify(null);
          setMessages(prev => [
            ...prev,
            {
              message_id: `memory_clarify_resolved_${Date.now()}`,
              role: "assistant",
              content: result.selected_label
                ? `Got it. I'll use ${result.selected_label}.`
                : "Got it. I'll update memory with that clarification.",
              status: "complete"
            }
          ]);
          setChatState("idle");
        } catch (e: any) {
          setMessages(prev => [
            ...prev,
            {
              message_id: `memory_clarify_error_${Date.now()}`,
              role: "assistant",
              content: e.toString(),
              status: "complete"
            }
          ]);
          setChatState("awaiting_memory_clarification");
        }
        return;
      }
      suppressTokensRef.current = false;
      setChatError(null);
      setChatState("sending");
      chatStateRef.current = "sending";
      isStreamingRef.current = true;
      streamActiveRef.current = true;
      streamStartAtRef.current = Date.now();
      streamFirstTokenLoggedRef.current = false;
      lastActivityRef.current = Date.now();
      watchdogTriggeredRef.current = false;
      if (pendingClarification) {
        console.log("[Clarification] Submitting answer:", rawContent);
        await invokeWithTimeout("submit_clarification", {
          answer: rawContent,
          originalInput: pendingClarification.originalInput,
          originalRunId: pendingClarification.runId
        }, TIMEOUTS.long);
        setPendingClarification(null);
      } else {
        const content = explicitFeedback ? ensureFeedbackPrefix(rawContent) : rawContent;
        await invokeWithTimeout("send_message", { content }, TIMEOUTS.long);
      }
    } catch (e) {
      reportError("Failed to send message", e);
      setChatError("Failed to send message. You can retry.");
      setChatState("error");
      chatErrorAtRef.current = Date.now();
      streamActiveRef.current = false;
      setStreamingMessageId(null);
    }
  };

  const handleStop = async () => {
    setChatState("stopping");
    setChatError(null);
    // 1. Frontend Hard Stop (Audio & Sockets)
    if (voiceRef.current) {
      voiceRef.current.halt();
    }
    // 2. Clear Buffers
    streamingMessageIdRef.current = null;
    setStreamingMessageId(null);
    streamActiveRef.current = false;
    pendingTokenBufferRef.current = "";
    clearTokenFlushTimer();
    setStreamingTokenBuffer(null);
    suppressTokensRef.current = true;
    setModuleStatus(null);
    setPendingClarification(null);
    setPendingMemoryClarify(null);
    setPendingConflict(null);

    // 3. Backend Hard Abort
    try {
      await invokeWithTimeout("abort_generation", {
        run_id: lastRunIdRef.current ?? undefined,
      }, TIMEOUTS.short);
    } catch (e) {
      reportError("Failed to abort generation", e);
    }
    await loadMessages();
    setChatState("idle");
  };

  const handleRecover = async () => {
    suppressTokensRef.current = true;
    streamingMessageIdRef.current = null;
    setStreamingMessageId(null);
    streamActiveRef.current = false;
    pendingTokenBufferRef.current = "";
    clearTokenFlushTimer();
    setStreamingTokenBuffer(null);
    setChatError(null);
    setChatState("idle");
    setModuleStatus(null);
    try {
      await invokeWithTimeout("abort_generation", {
        run_id: lastRunIdRef.current ?? undefined,
      }, TIMEOUTS.short);
    } catch (e) {
      reportError("Failed to recover generation", e);
    }
    await loadMessages();
  };

  const testConnection = async () => {
    if (!settings) return;
    setTestResult(null);
    try {
      const modelId = await invokeWithTimeout<string>("test_connection", {
        url: settings.api_base_url,
        apiKey: settings.api_key,
      }, TIMEOUTS.long);
      setTestResult({ success: true, message: `Connected! Active model: ${modelId}` });
      setSettings({ ...settings, active_model_id: modelId });
    } catch (e) {
      setTestResult({ success: false, message: String(e) });
    }
  };

  useEffect(() => {
    if (view === "trace") {
      loadCockpitData();
    }
  }, [view, settings?.trace_history_limit, loadCockpitData]);

  const saveSettings = async () => {
    if (!settings) return;
    try {
      const [normalizedUrl, isLoopback] = await invokeWithTimeout<[string, boolean]>("normalize_url", {
        url: settings.api_base_url,
      }, TIMEOUTS.short);

      if (!isLoopback) {
        if (!confirm("Warning: This API URL is not local. Prompts may be sent off-device. Continue?")) {
          return;
        }
      }

      const updated = { ...settings, api_base_url: normalizedUrl };
      await invokeWithTimeout("update_settings", { settings: updated }, TIMEOUTS.long);
      setSettings(updated);
      showToast("Settings saved!");
    } catch (e) {
      reportError("Failed to save settings", e);
    }
  };

  const handleOnboardingComplete = async (payload: OnboardingPayload) => {
    if (!settings) {
      throw new Error("Settings are not available yet.");
    }
    const [primaryUrl, primaryLoopback] = await invokeWithTimeout<[string, boolean]>("normalize_url", {
      url: payload.apiBaseUrl,
    }, TIMEOUTS.short);
    const [opsUrl, opsLoopback] = await invokeWithTimeout<[string, boolean]>("normalize_url", {
      url: payload.summarizationApiUrl,
    }, TIMEOUTS.short);

    if (!primaryLoopback || !opsLoopback) {
      const proceed = confirm(
        "Warning: One or more API URLs are not local. Prompts may be sent off-device. Continue?"
      );
      if (!proceed) {
        return;
      }
    }

    const updated = {
      ...settings,
      api_base_url: primaryUrl,
      summarization_api_url: opsUrl,
      user_display_name: payload.userName,
      assistant_display_name: payload.assistantName,
      history_window: 3,
      onboarding_completed: true,
    };

    await invokeWithTimeout("update_settings", { settings: updated }, TIMEOUTS.long);
    setSettings(updated);
    setView("chat");
    showToast("Onboarding complete. Welcome to Symbiote!");
  };

  const handleWipeMemory = async () => {
    if (!confirm("?? Wipe all data? This resets Symbiote to a new install state (settings, conversations, and memory). This cannot be undone.")) {
      return;
    }
    try {
      await handleStop();
      await invokeWithTimeout("reset_all_data", undefined, TIMEOUTS.long);
      setRollingSummary(null);
      setRollingSummaryPending(false);
      setLiveSummaryPending(false);
      setSelfModel(null);
      setSelfInspection(null);
      showToast("All data wiped successfully!");
    } catch (e) {
      reportError("Failed to wipe data", e);
    }
  };

  const handleResetConversationData = async () => {
    if (!confirm("?? This will permanently delete conversation history and rolling summaries for the current conversation.\n\nYour memory and settings will be preserved.\n\nThis cannot be undone. Continue?")) return;
    try {
      await handleStop();
      setMessages([]);
      setModuleStatus(null);
      setPendingClarification(null);
      setPendingMemoryClarify(null);
      setPendingConflict(null);
      setStreamingTokenBuffer(null);
      setRollingSummary(null);
      setRollingSummaryPending(false);
      setLiveSummaryPending(false);
      suppressTokensRef.current = true;
      setChatState("idle");
      await invokeWithTimeout("reset_conversation_data", undefined, TIMEOUTS.long);
      await handleRefreshSelfData();
      showToast("Conversation data has been reset.");
    } catch (e) {
      reportError("Failed to reset conversation data", e);
      loadMessages();
    }
  };

  const handleSetReflectionFrozen = async (frozen: boolean) => {
    try {
      await invokeWithTimeout("set_reflection_frozen", { frozen }, TIMEOUTS.medium);
      await loadSelfModel();
    } catch (e) {
      reportError("Failed to update reflection freeze state", e);
    }
  };

  const handleClear = async () => {
    const isBusy = chatState === "sending" || chatState === "streaming" || chatState === "stopping";
    if (isBusy || !confirm("Clear messages from view? This does not delete history.")) return;
    clearedAtRef.current = new Date().toISOString();
    setMessages([]);
  };

  const handleSettingsChange = (nextSettings: Settings) => {
    setSettings(nextSettings);
  };

  const handleTraceSettingsUpdate = async (nextSettings: Settings) => {
    setSettings(nextSettings);
    try {
      await invokeWithTimeout("update_settings", { settings: nextSettings }, TIMEOUTS.long);
    } catch (e) {
      reportError("Failed to update settings", e);
    }
  };

  const handleCockpitWriteToggle = async (enabled: boolean) => {
    if (!settings) return;
    const updated = { ...settings, cockpit_write_enabled: enabled };
    setSettings(updated);
    try {
      await invokeWithTimeout("update_settings", { settings: updated }, TIMEOUTS.long);
      showToast(`Cockpit write ${enabled ? "enabled" : "set to read-only"}.`);
    } catch (e) {
      reportError("Failed to update cockpit write mode", e);
      setSettings(settings);
    }
  };

  const handleViewChange = useCallback((nextView: View) => {
    setView(nextView);
  }, [view]);

  const handleToggleGraph = useCallback(() => {
    setIsGraphOpen((prev) => {
      const next = !prev;
      if (!next && view === "chat") {
        window.setTimeout(() => inputRef.current?.focus(), 0);
      }
      return next;
    });
  }, [view]);

  const handleCloseGraph = useCallback(() => {
    setIsGraphOpen(false);
    if (view === "chat") {
      window.setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [view]);

  const allowControlWrites = settings?.cockpit_write_enabled ?? false;
  const onboardingRequired = settings?.onboarding_completed === false;

  if (onboardingRequired) {
    return (
      <div className="onboarding-container">
        <TitleBar />
        {toast && (
          <Toast
            message={toast.message}
            type={toast.type}
            actionLabel={toast.actionLabel}
            onAction={toast.onAction}
            onClose={() => setToast(null)}
          />
        )}
        <main className="onboarding-main">
          <OnboardingView
            settings={settings}
            settingsError={settingsError}
            onComplete={handleOnboardingComplete}
            onRetry={loadSettings}
          />
        </main>
      </div>
    );
  }

  return (
    <div className="app-container">
      <TitleBar />
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          actionLabel={toast.actionLabel}
          onAction={toast.onAction}
          onClose={() => setToast(null)}
        />
      )}
      <SidebarNav
        view={view}
        onViewChange={handleViewChange}
        onToggleGraph={handleToggleGraph}
        onStop={handleStop}
        onClear={handleClear}
      />

      <main className="content">
        {view === "chat" ? (
          <div className="chat-shell">
            <div className="chat-main">
                <ChatView
                  messages={messages}
                  streamingTokenBuffer={streamingTokenBuffer}
                  streamingMessageId={streamingMessageId}
                  moduleStatus={moduleStatus}
                  input={input}
                  feedbackMode={feedbackMode}
                  chatState={chatState}
                  chatError={chatError}
                  memoryError={memoryError}
                  isBusy={
                    chatState === "sending"
                    || chatState === "streaming"
                    || chatState === "stopping"
                    || chatState === "post_processing"
                  }
                  settings={settings}
                  selfModel={selfModel}
                  showRaw={showRaw}
                  rollingSummary={rollingSummary}
                  rollingSummaryError={rollingSummaryError}
                  rollingSummaryErrorAt={rollingSummaryErrorAt}
                  rollingSummaryPending={rollingSummaryPending}
                  liveSummary={liveSummary}
                  liveSummaryError={liveSummaryError}
                  liveSummaryErrorAt={liveSummaryErrorAt}
                  liveSummaryPending={liveSummaryPending}
                  onSummaryOpen={() => {
                    loadRollingSummary();
                    loadLiveSummary();
                  }}
                  innerMonologueEntries={innerMonologueEntries}
                  innerMonologueError={innerMonologueError}
                  onMonologueOpen={loadInnerMonologue}
                  pendingPrompts={pendingPrompts}
                  pendingPromptCount={pendingPromptCount}
                  pendingPromptError={pendingPromptError}
                  onPendingPromptSend={handlePendingPromptSend}
                  onPendingPromptDismiss={handlePendingPromptDismiss}
                  onPendingPromptRephrase={handlePendingPromptRephrase}
                  onInputChange={setInput}
                  onSend={handleSend}
                  onStop={handleStop}
                  onRecover={handleRecover}
                  onFeedbackModeChange={setFeedbackMode}
                  onDismissMemoryError={() => setMemoryError(null)}
                  onAppendTranscript={(text) => setInput(prev => prev + " " + text)}
                  voiceStatus={voiceOutputEnabled ? voiceStatus : "disabled"}
                  voiceEnabled={voiceOutputEnabled}
                  memoryGraphOpen={isGraphOpen}
                  memoryErrorAt={selfInspection?.last_memory_error_at || null}
                  voiceRef={voiceRef}
                  onVoiceStatusChange={setVoiceStatus}
                  inputRef={inputRef}
                />
            </div>
            <SystemStatePanel
              chatState={chatState}
              moduleStatus={moduleStatus}
              memoryError={memoryError}
              rollingSummaryError={rollingSummaryError}
              pendingPromptCount={pendingPromptCount}
              healthSnapshot={systemHealthSnapshot}
              healthHistory={systemHealthHistory}
              selfModel={selfModel}
            />
          </div>
        ) : view === "trace" ? (
          <TraceView
            systemLogs={systemLogs}
            monologueEntries={innerMonologueEntries}
            messages={messages}
            subjectSnapshots={subjectSnapshots}
            gateDecisions={gateDecisions}
            contextTags={contextTags}
            intentSummary={intentSummary}
            introspectionEntries={introspectionEntries}
            auditLog={auditLog}
            errorEvents={errorEvents}
            qualiaLabels={qualiaLabels}
            evidenceLineage={evidenceLineage}
            evidenceLineageError={evidenceLineageError}
            systemControls={systemControls}
            systemControlEvents={systemControlEvents}
            systemHealthSnapshot={systemHealthSnapshot}
            systemHealthHistory={systemHealthHistory}
            gateInputEvents={gateInputEvents}
            cockpitLastUpdated={cockpitLastUpdated}
            controlError={systemControlError}
            healthError={systemHealthError}
            error={systemLogError}
            allowControlWrites={allowControlWrites}
            cockpitWriteEnabled={allowControlWrites}
            settings={settings}
            onUpdateSettings={handleTraceSettingsUpdate}
            onToggleCockpitWrite={handleCockpitWriteToggle}
            onRefresh={() => loadCockpitData()}
            onClear={() => {
              setSystemLogs([]);
              setGateInputEvents([]);
              setSubjectSnapshots([]);
              setGateDecisions([]);
              setIntrospectionEntries([]);
              setAuditLog([]);
              setErrorEvents([]);
              setQualiaLabels([]);
              setEvidenceLineage([]);
            }}
          />
        ) : (
          <SettingsView
            settings={settings}
            settingsError={settingsError}
            selfModel={selfModel}
            selfInspection={selfInspection}
            systemControls={systemControls}
            systemControlError={systemControlError}
            onRefreshSystemControls={loadSystemControls}
            showRaw={showRaw}
            testResult={testResult}
            onUpdateSettings={handleSettingsChange}
            onToggleShowRaw={setShowRaw}
            onTestConnection={testConnection}
            onSaveSettings={saveSettings}
            onWipeMemory={handleWipeMemory}
            onResetConversationData={handleResetConversationData}
            onSetReflectionFrozen={handleSetReflectionFrozen}
            onRefreshSelfData={handleRefreshSelfData}
            onRetrySettings={loadSettings}
          />
        )}
      </main>

      {isGraphOpen && (
        <Suspense fallback={null}>
          <MemoryGraph3D isOpen={isGraphOpen} onClose={handleCloseGraph} />
        </Suspense>
      )}

    </div>
  );
}

export default App;
