import { useEffect, useState, useRef, forwardRef, useImperativeHandle } from 'react';
import { withTimeout } from '../utils/async';
import { invokeWithTimeout } from '../utils/tauri';
import { resumeAudioContext } from '../utils/audio';

// --- Inline AudioWorklet Processor ---
const WORKLET_CODE = `
class PCMProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const float32Data = input[0];
      // Convert Float32 to Int16 PCM
      const int16Data = new Int16Array(float32Data.length);
      for (let i = 0; i < float32Data.length; i++) {
        let s = Math.max(-1, Math.min(1, float32Data[i]));
        int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      this.port.postMessage(int16Data.buffer, [int16Data.buffer]);
    }
    return true;
  }
}
registerProcessor('pcm-processor', PCMProcessor);
`;

const WS_URL = "ws://127.0.0.1:11435/ws/audio";
const HTTP_URL = "http://127.0.0.1:11435";
const TTS_ACTIVITY_TIMEOUT_MS = 15000;
const VOICE_RESTART_MIN_INTERVAL_MS = 10000;
const VOICE_RESTART_MIN_UPTIME_MS = 5000;

export type VoiceStatus = "loading" | "ready" | "error" | "recording" | "speaking" | "reconnecting";

const MAX_HEALTH_RETRIES = 15;
const MAX_WS_RETRIES = 6;
const HEALTH_TIMEOUT_MS = 3000;

interface VoiceControllerProps {
    onTranscript: (text: string) => void;
    onStatusChange?: (status: VoiceStatus) => void;
    voiceName?: string;
    voiceSpeed?: number;
    voicePitch?: number;
    voiceReverb?: number;
    voiceCompression?: number;
    voiceFormant?: number;
    className?: string;
}

type TtsRequest = { text: string; meta?: { messageId?: string; runId?: string } };

export interface VoiceControllerHandle {
    speak: (text: string, meta?: { messageId?: string; runId?: string }) => void;
    stopTts: (reason?: string) => void;
    halt: () => void;
}

export const VoiceController = forwardRef<VoiceControllerHandle, VoiceControllerProps>(({
    onTranscript,
    onStatusChange,
    voiceName,
    voiceSpeed,
    voicePitch,
    voiceReverb,
    voiceCompression,
    voiceFormant,
    className
}, ref) => {
    const [status, setStatus] = useState<VoiceStatus>("loading");
    const wsRef = useRef<WebSocket | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const workletNodeRef = useRef<AudioWorkletNode | null>(null);
    const audioQueueRef = useRef<ArrayBuffer[]>([]);
    const isPlayingRef = useRef(false);
    const playbackCtxRef = useRef<AudioContext | null>(null); // Persistent Playback Context
    const statusRef = useRef<VoiceStatus>("loading");
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const healthPollRef = useRef<number | null>(null);
    const healthRetriesRef = useRef(0);
    const wsReconnectAttemptsRef = useRef(0);
    const wsReconnectTimerRef = useRef<number | null>(null);
    const lastEnergyRef = useRef<number>(0);
    const ttsActiveRef = useRef(false);
    const ttsQueueRef = useRef<TtsRequest[]>([]);
    const lastTtsActivityRef = useRef<number>(0);
    const ttsWatchdogRef = useRef<number | null>(null);
    const ttsMessageIdRef = useRef<string | null>(null);
    const ttsRunIdRef = useRef<string | null>(null);
    const ttsStartedAtRef = useRef<number | null>(null);
    const ttsFirstChunkLoggedRef = useRef(false);
    const lastVoiceRestartAtRef = useRef<number>(0);
    const pendingRestartTimerRef = useRef<number | null>(null);
    const voiceReadyAtRef = useRef<number | null>(null);
    const pendingRestartAfterSpeechRef = useRef(false);
    const pendingRestartReasonRef = useRef<string | undefined>(undefined);

    const emitVoiceEnergy = (value: number) => {
        const clamped = Math.max(0, Math.min(1, value));
        if (Math.abs(clamped - lastEnergyRef.current) < 0.01) {
            return;
        }
        lastEnergyRef.current = clamped;
        window.dispatchEvent(new CustomEvent("voice_energy", { detail: clamped }));
    };

    const setStatusSafe = (next: VoiceStatus | ((prev: VoiceStatus) => VoiceStatus)) => {
        setStatus((prev) => {
            const resolved = typeof next === "function" ? next(prev) : next;
            statusRef.current = resolved;
            onStatusChange?.(resolved);
            return resolved;
        });
    };

    // --- 1. Startup & Health Check ---
    useEffect(() => {
        // Init playback context once
        const ctx = new AudioContext({ sampleRate: 24000 }); // Match Kokoro
        playbackCtxRef.current = ctx;
        return () => {
            ctx.close();
        };
    }, []);

    const clearHealthPoll = () => {
        if (healthPollRef.current !== null) {
            clearInterval(healthPollRef.current);
            healthPollRef.current = null;
        }
    };

    const clearReconnectTimer = () => {
        if (wsReconnectTimerRef.current !== null) {
            clearTimeout(wsReconnectTimerRef.current);
            wsReconnectTimerRef.current = null;
        }
    };

    const scheduleReconnect = () => {
        wsReconnectAttemptsRef.current += 1;
        if (wsReconnectAttemptsRef.current > MAX_WS_RETRIES) {
            setErrorMessage("Voice WebSocket failed to reconnect.");
            setStatusSafe("error");
            return;
        }
        const delay = Math.min(30000, 1000 * Math.pow(2, wsReconnectAttemptsRef.current - 1));
        setStatusSafe("reconnecting");
        clearReconnectTimer();
        wsReconnectTimerRef.current = window.setTimeout(() => {
            initWebSocket();
        }, delay);
    };

    const clearTtsWatchdog = () => {
        if (ttsWatchdogRef.current !== null) {
            window.clearInterval(ttsWatchdogRef.current);
            ttsWatchdogRef.current = null;
        }
    };

    const startTtsWatchdog = () => {
        if (ttsWatchdogRef.current !== null) return;
        ttsWatchdogRef.current = window.setInterval(() => {
            if (!ttsActiveRef.current) {
                clearTtsWatchdog();
                return;
            }
            const idleMs = Date.now() - lastTtsActivityRef.current;
            if (idleMs > TTS_ACTIVITY_TIMEOUT_MS) {
                console.warn("[VoiceController] TTS watchdog timeout");
                void logUiTiming({
                    event: "tts_watchdog_trigger",
                    duration_ms: idleMs,
                    detail: ttsMessageIdRef.current ?? undefined,
                    run_id: ttsRunIdRef.current ?? undefined,
                    message_id: ttsMessageIdRef.current ?? undefined,
                });
                ttsActiveRef.current = false;
                setErrorMessage("TTS stalled. Restarting voice service.");
                setStatusSafe("error");
                restartVoiceService("tts_watchdog");
            }
        }, 1000);
    };

    const markTtsActivity = () => {
        lastTtsActivityRef.current = Date.now();
    };

    const logUiTiming = (payload: {
        event: string;
        duration_ms: number;
        detail?: string;
        run_id?: string;
        message_id?: string;
    }) => {
        return invokeWithTimeout("log_ui_timing", payload, 5000).catch((err) => {
            console.error("[VoiceController] log_ui_timing failed", err);
            const errorDetail = err instanceof Error ? err.message : String(err);
            const fallbackPayload = {
                ...payload,
                detail: payload.detail ? `${payload.detail} | ui_log_failed=${errorDetail}` : `ui_log_failed=${errorDetail}`,
            };
            void invokeWithTimeout("log_tts_event", {
                event: "ui_log_failed",
                duration_ms: 0,
                detail: errorDetail,
                run_id: payload.run_id,
                message_id: payload.message_id,
            }, 5000).catch(() => {});
            void invokeWithTimeout("log_tts_event", fallbackPayload, 5000).catch(() => {});
        });
    };

    const logTtsStop = (reason: string) => {
        const startedAt = ttsStartedAtRef.current;
        const durationMs = startedAt ? Date.now() - startedAt : 0;
        void logUiTiming({
            event: "tts_stop_reason",
            duration_ms: durationMs,
            detail: reason,
            run_id: ttsRunIdRef.current ?? undefined,
            message_id: ttsMessageIdRef.current ?? undefined,
        });
    };

    const resumePlaybackContext = async () => {
        const ctx = playbackCtxRef.current;
        if (!ctx) return;
        await resumeAudioContext(ctx);
    };

    const enqueueTts = (text: string, meta?: { messageId?: string; runId?: string }) => {
        ttsQueueRef.current.push({ text, meta });
    };

    const sendTtsRequest = (text: string, meta?: { messageId?: string; runId?: string }) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            enqueueTts(text, meta);
            if (statusRef.current === "error") {
                restartVoiceService("voice_not_ready");
            } else {
                startHealthPolling();
            }
            return;
        }
        // FLUSH: Clear any accumulated audio/echo in backend before we start speaking
        wsRef.current.send(JSON.stringify({ type: "stop_listening" }));
        console.log("[VoiceController] Sending TTS request:", text);
        void resumePlaybackContext();
        ttsMessageIdRef.current = meta?.messageId ?? null;
        ttsRunIdRef.current = meta?.runId ?? null;
        ttsStartedAtRef.current = Date.now();
        ttsFirstChunkLoggedRef.current = false;
        void logUiTiming({
            event: "tts_speak_start",
            duration_ms: 0,
            detail: ttsMessageIdRef.current ?? undefined,
            run_id: ttsRunIdRef.current ?? undefined,
            message_id: ttsMessageIdRef.current ?? undefined,
        });
        ttsActiveRef.current = true;
        markTtsActivity();
        startTtsWatchdog();
        wsRef.current.send(JSON.stringify({
            type: "tts",
            content: text,
            voice: voiceName || "bf_isabella",
            speed: voiceSpeed || 1.0,
            pitch_semitones: voicePitch ?? 1.0,
            reverb_amount: voiceReverb ?? 0.15,
            compression: voiceCompression ?? 0.05,
            formant_shift: voiceFormant || 0.0
        }));
        setStatusSafe("speaking");
    };

    const startNextTts = () => {
        if (statusRef.current === "recording") {
            return;
        }
        if (ttsActiveRef.current || isPlayingRef.current) {
            return;
        }
        const next = ttsQueueRef.current.shift();
        if (!next) {
            return;
        }
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            ttsQueueRef.current.unshift(next);
            if (statusRef.current === "error") {
                restartVoiceService("voice_not_ready");
            } else {
                startHealthPolling();
            }
            return;
        }
        sendTtsRequest(next.text, next.meta);
    };

    const startHealthPolling = () => {
        clearHealthPoll();
        healthRetriesRef.current = 0;
        setErrorMessage(null);
        setStatusSafe("loading");

        healthPollRef.current = window.setInterval(async () => {
            healthRetriesRef.current += 1;
            try {
                const res = await withTimeout(fetch(`${HTTP_URL}/health`), HEALTH_TIMEOUT_MS, "voice_health");
                const data = await res.json();
                if (data.status === "ready") {
                    clearHealthPoll();
                    wsReconnectAttemptsRef.current = 0;
                    setStatusSafe("ready");
                    initWebSocket();
                }
            } catch (_e) {
                if (healthRetriesRef.current >= MAX_HEALTH_RETRIES) {
                    clearHealthPoll();
                    setErrorMessage("Voice service did not respond.");
                    setStatusSafe("error");
                }
            }
        }, 1000);
    };

    const scheduleRestart = (delayMs: number, reason?: string) => {
        if (pendingRestartTimerRef.current !== null) {
            return;
        }
        pendingRestartTimerRef.current = window.setTimeout(() => {
            pendingRestartTimerRef.current = null;
            restartVoiceService(reason);
        }, delayMs);
    };

    const triggerPendingRestart = () => {
        if (!pendingRestartAfterSpeechRef.current) {
            return;
        }
        if (statusRef.current === "speaking") {
            return;
        }
        const reason = pendingRestartReasonRef.current || "deferred_restart";
        pendingRestartAfterSpeechRef.current = false;
        pendingRestartReasonRef.current = undefined;
        restartVoiceService(reason);
    };

    const restartVoiceService = (reason?: string) => {
        const now = Date.now();
        if (statusRef.current === "speaking" && reason !== "tts_watchdog") {
            pendingRestartAfterSpeechRef.current = true;
            pendingRestartReasonRef.current = reason || "deferred_restart";
            return;
        }
        const sinceLast = now - lastVoiceRestartAtRef.current;
        if (sinceLast < VOICE_RESTART_MIN_INTERVAL_MS) {
            scheduleRestart(VOICE_RESTART_MIN_INTERVAL_MS - sinceLast, reason || "restart_cooldown");
            return;
        }
        if (voiceReadyAtRef.current && now - voiceReadyAtRef.current < VOICE_RESTART_MIN_UPTIME_MS) {
            const remaining = VOICE_RESTART_MIN_UPTIME_MS - (now - voiceReadyAtRef.current);
            scheduleRestart(Math.max(remaining, 250), reason || "uptime_guard");
            return;
        }
        lastVoiceRestartAtRef.current = now;
        if (pendingRestartTimerRef.current !== null) {
            clearTimeout(pendingRestartTimerRef.current);
            pendingRestartTimerRef.current = null;
        }
        clearReconnectTimer();
        clearHealthPoll();
        clearTtsWatchdog();
        ttsActiveRef.current = false;
        stopAudioPlayback("restart_voice_service");
        stopRecording();
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        audioQueueRef.current = [];
        void logUiTiming({
            event: "voice_service_restart",
            duration_ms: 0,
            detail: reason,
        });
        invokeWithTimeout("restart_voice_service", undefined, 5000).catch((e) => {
            setErrorMessage(String(e));
            setStatusSafe("error");
        });
        startHealthPolling();
    };

    useEffect(() => {
        // Start the Python Service via Rust
        restartVoiceService("mount");
        return () => {
            cleanup();
        };
    }, []);

    // --- 2. WebSocket Init ---
    const initWebSocket = () => {
        if (wsRef.current) return;
        console.log("[VoiceController] Connecting to WebSocket...");
        const ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log("[VoiceController] WebSocket Connected");
            wsReconnectAttemptsRef.current = 0;
            setErrorMessage(null);
            setStatusSafe("ready");
            voiceReadyAtRef.current = Date.now();
            startNextTts();
        };

        ws.onclose = (event) => {
            console.log("[VoiceController] WebSocket Closed");
            void logUiTiming({
                event: "voice_ws_closed",
                duration_ms: 0,
                detail: `code=${event.code} reason=${event.reason || "none"} clean=${event.wasClean}`,
                run_id: ttsRunIdRef.current ?? undefined,
                message_id: ttsMessageIdRef.current ?? undefined,
            });
            // If we were recording, stop it properly
            if (statusRef.current === "recording") {
                stopRecording();
            }
            if (statusRef.current === "speaking") {
                stopAudioPlayback("ws_closed");
            }
            ttsActiveRef.current = false;
            clearTtsWatchdog();
            wsRef.current = null;

            // Auto-reconnect after delay
            scheduleReconnect();
        };

        ws.onerror = (e) => {
            console.error("[VoiceController] WS Error", e);
            // Close triggers onclose, so logic handles there
        };

        ws.onmessage = async (event) => {
            if (typeof event.data === "string") {
                const msg = JSON.parse(event.data);
                handleWsMessage(msg);
            } else {
                // Binary Audio (PCM from TTS) - comes as Blob, convert to ArrayBuffer
                const blob = event.data as Blob;
                const arrayBuffer = await blob.arrayBuffer();
                queueAudio(arrayBuffer);
            }
        };
        wsRef.current = ws;
    };

    const handleWsMessage = (msg: any) => {
        if (msg.type === "text") {
            onTranscript(msg.content);
        } else if (msg.type === "interrupt") {
            if (statusRef.current === "recording") {
                stopAudioPlayback("tts_interrupt");
                ttsActiveRef.current = false;
                clearTtsWatchdog();
            }
        } else if (msg.type === "tts_end") {
            // Marker for end of stream
            ttsActiveRef.current = false;
            clearTtsWatchdog();
        if (!isPlayingRef.current && audioQueueRef.current.length === 0) {
            if (ttsStartedAtRef.current !== null) {
                const durationMs = Date.now() - ttsStartedAtRef.current;
                void logUiTiming({
                    event: "tts_speak_end",
                        duration_ms: durationMs,
                        detail: ttsMessageIdRef.current ?? undefined,
                        run_id: ttsRunIdRef.current ?? undefined,
                        message_id: ttsMessageIdRef.current ?? undefined,
                    });
                    ttsStartedAtRef.current = null;
                    ttsMessageIdRef.current = null;
                    ttsRunIdRef.current = null;
            }
            setStatusSafe((prev) => prev === "speaking" ? "ready" : prev);
            emitVoiceEnergy(0);
            triggerPendingRestart();
            startNextTts();
        }
        } else if (msg.type === "tts_error") {
            stopAudioPlayback("tts_error");
            ttsActiveRef.current = false;
            clearTtsWatchdog();
            setErrorMessage(msg.detail || "TTS error");
            setStatusSafe("error");
        }
    };

    // --- 3. Audio Capture (Mic) ---
    const startRecording = async () => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || statusRef.current !== "ready") {
            setErrorMessage("Voice connection is not ready.");
            setStatusSafe("error");
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true } });
            const ctx = new AudioContext({ sampleRate: 16000 });
            audioContextRef.current = ctx;

            console.log("[VoiceController] AudioContext Sample Rate:", ctx.sampleRate);
            if (ctx.sampleRate !== 16000) {
                console.warn("[VoiceController] WARNING: Browser ignored sampleRate 16000 request. Got:", ctx.sampleRate);
            }

            const blob = new Blob([WORKLET_CODE], { type: "application/javascript" });
            const url = URL.createObjectURL(blob);
            await ctx.audioWorklet.addModule(url);

            const source = ctx.createMediaStreamSource(stream);
            const worklet = new AudioWorkletNode(ctx, "pcm-processor");

            worklet.port.onmessage = (e) => {
                // GATE: Prevent Self-Interruption / Echo
                // If the AI is currently speaking, invalid microphone input (echo) is inevitable on speakers.
                // We drop these input chunks to prevent the AI from hearing itself and triggering an interruption.
                if (isPlayingRef.current || ttsActiveRef.current) {
                    return;
                }

                if (wsRef.current?.readyState === WebSocket.OPEN) {
                    // console.log("[VoiceController] Sending chunk:", e.data.byteLength);
                    wsRef.current.send(e.data);
                }
            };

            source.connect(worklet);
            worklet.connect(ctx.destination); // Connect to speakers? User might hear themselves.
            // If we don't connect to destination, some browsers might not run the graph.
            // Let's keep it disconnected but ensure 'start' is called if needed?
            // Actually, trying to connect to destination to see if that forces it to run. 
            // BUT mute it?
            // createGain(0) -> destination.
            const mute = ctx.createGain();
            mute.gain.value = 0;
            worklet.connect(mute);
            mute.connect(ctx.destination);

            workletNodeRef.current = worklet;
            setStatusSafe("recording");

        } catch (e) {
            console.error("Mic Error", e);
            setErrorMessage("Microphone access failed.");
            setStatusSafe("error");
        }
    };

    const stopRecording = () => {
        console.log("[VoiceController] Stopping recording");

        // 1. Tell backend to stop listening/reset VAD state
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: "stop_listening" }));
        }

        // 2. Kill Audio Source Flow
        if (workletNodeRef.current) {
            workletNodeRef.current.port.onmessage = null; // Stop event flow
            workletNodeRef.current.disconnect();
            workletNodeRef.current = null;
        }

        // 3. Close Audio Context (Hard Stop)
        if (audioContextRef.current) {
            audioContextRef.current.close().catch(e => console.error("Error closing ctx", e));
            audioContextRef.current = null;
        }

        setStatusSafe("ready");
    };

    const toggleMic = () => {
        console.log("[VoiceController] Toggle mic - current status:", statusRef.current);
        void resumePlaybackContext();
        if (statusRef.current === "loading" || statusRef.current === "error" || statusRef.current === "reconnecting") {
            restartVoiceService("toggle_mic");
        } else if (statusRef.current === "recording") {
            stopRecording();
        } else if (statusRef.current === "speaking") {
            // Barge-in: Stop TTS immediately
            stopAudioPlayback("barge_in");
            // We set status to 'ready' in stopAudioPlayback, so user can click again to record
        } else {
            startRecording();
        }
    };

    const activeSourceNodeRef = useRef<AudioBufferSourceNode | null>(null);

    // --- 4. Audio Playback (TTS) ---
    const queueAudio = (buffer: ArrayBuffer) => {
        // console.log("[VoiceController] Received audio chunk, size:", buffer.byteLength);
        if (buffer.byteLength === 0) {
            return;
        }
        ttsActiveRef.current = true;
        markTtsActivity();
        if (!ttsFirstChunkLoggedRef.current && ttsStartedAtRef.current !== null) {
            ttsFirstChunkLoggedRef.current = true;
            const durationMs = Date.now() - ttsStartedAtRef.current;
            void logUiTiming({
                event: "tts_first_audio",
                duration_ms: durationMs,
                detail: ttsMessageIdRef.current ?? undefined,
                run_id: ttsRunIdRef.current ?? undefined,
                message_id: ttsMessageIdRef.current ?? undefined,
            });
        }
        startTtsWatchdog();
        audioQueueRef.current.push(buffer);
        if (!isPlayingRef.current) {
            playNextChunk();
        }
    };

    const playNextChunk = async () => {
        // If queue empty, we are done
        if (audioQueueRef.current.length === 0) {
            isPlayingRef.current = false;
            if (ttsStartedAtRef.current !== null) {
                const durationMs = Date.now() - ttsStartedAtRef.current;
                void logUiTiming({
                    event: "tts_speak_end",
                    duration_ms: durationMs,
                    detail: ttsMessageIdRef.current ?? undefined,
                    run_id: ttsRunIdRef.current ?? undefined,
                    message_id: ttsMessageIdRef.current ?? undefined,
                });
                ttsStartedAtRef.current = null;
                ttsMessageIdRef.current = null;
                ttsRunIdRef.current = null;
            }
            // Only reset status if we resolved to 'speaking' and are now done. 
            // Don't override 'recording' if user started talking.
            setStatusSafe((prev) => prev === "speaking" ? "ready" : prev);
            emitVoiceEnergy(0);
            triggerPendingRestart();
            if (!ttsActiveRef.current) {
                startNextTts();
            }
            return;
        }

        const ctx = playbackCtxRef.current;
        if (!ctx) return;
        await resumePlaybackContext();

        isPlayingRef.current = true;
        // If not recording, we can show speaking status
        setStatusSafe((prev) => prev === "recording" ? "recording" : "speaking");

        const chunk = audioQueueRef.current.shift()!;

        // Decode / Convert
        const int16 = new Int16Array(chunk);
        const float32 = new Float32Array(int16.length);
        let sumSquares = 0;
        for (let i = 0; i < int16.length; i++) {
            const sample = int16[i] / 32768.0;
            float32[i] = sample;
            sumSquares += sample * sample;
        }
        if (int16.length > 0) {
            const rms = Math.sqrt(sumSquares / int16.length);
            const energy = Math.min(1, Math.pow(rms * 3.2, 0.6));
            emitVoiceEnergy(energy);
        }

        if (float32.length === 0) {
            playNextChunk();
            return;
        }

        const audioBuf = ctx.createBuffer(1, float32.length, 24000);
        audioBuf.getChannelData(0).set(float32);

        const source = ctx.createBufferSource();
        source.buffer = audioBuf;
        source.connect(ctx.destination);

        // Track active source
        activeSourceNodeRef.current = source;

        source.onended = () => {
            activeSourceNodeRef.current = null;
            playNextChunk();
        };
        source.start();
    };

    const stopAudioPlayback = (reason?: string) => {
        console.log("[VoiceController] Stopping Audio Playback (Interrupt)");
        const shouldClearQueue = Boolean(reason && ["user_stop", "barge_in", "halt", "tts_interrupt"].includes(reason));
        if (shouldClearQueue) {
            ttsQueueRef.current = [];
        }

        // 1. Clear Queue
        audioQueueRef.current = [];
        isPlayingRef.current = false;
        ttsActiveRef.current = false;
        clearTtsWatchdog();
        if (ttsStartedAtRef.current !== null && reason) {
            logTtsStop(reason);
        }
        ttsStartedAtRef.current = null;
        ttsMessageIdRef.current = null;
        ttsRunIdRef.current = null;

        // 2. Stop Active Source IMMEDIATELY
        if (activeSourceNodeRef.current) {
            try {
                activeSourceNodeRef.current.stop();
            } catch (e) {
                // Ignore errors if already stopped
            }
            activeSourceNodeRef.current = null;
        }

        setStatusSafe((prev) => prev === "speaking" ? "ready" : prev);
        emitVoiceEnergy(0);
        triggerPendingRestart();
    };

    const cleanup = () => {
        clearReconnectTimer();
        clearHealthPoll();
        clearTtsWatchdog();
        if (pendingRestartTimerRef.current !== null) {
            window.clearTimeout(pendingRestartTimerRef.current);
            pendingRestartTimerRef.current = null;
        }
        stopAudioPlayback();
        stopRecording();
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
    };

    // --- Expose speak method via ref ---
    useImperativeHandle(ref, () => ({
        speak: (text: string, meta?: { messageId?: string; runId?: string }) => {
            enqueueTts(text, meta);
            startNextTts();
        },
        stopTts: (reason?: string) => {
            stopAudioPlayback(reason || "user_stop");
        },
        halt: () => {
            console.log("[VoiceController] HALT COMMAND RECEIVED");
            stopAudioPlayback("halt");
            stopRecording();
            // Force Close BS
            if (wsRef.current) {
                // Close immediately (code 1000)
                wsRef.current.close(1000, "Killswitch");
                // The onclose handler will restart it, effectively resetting state.
            }
        }
    }));

    // --- UI ---
    const isLoading = status === "loading" || status === "reconnecting";
    const isError = status === "error";
    const title = isError
        ? `${errorMessage || "Voice unavailable"}. Click to restart.`
        : isLoading
            ? "Connecting voice service..."
            : "Toggle Voice";

    return (
        <>
            <button
                className={`btn btn-secondary ${className || ''} ${status}`}
                onClick={toggleMic}
                title={title}
            >
                {isLoading && <span className="spinner"></span>}
                {isError && <span>!</span>}
                {status === "ready" && (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                        <line x1="12" y1="19" x2="12" y2="23" />
                        <line x1="8" y1="23" x2="16" y2="23" />
                    </svg>
                )}
                {status === "recording" && (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                        <rect x="6" y="6" width="12" height="12" rx="2" />
                    </svg>
                )}
                {status === "speaking" && (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                        <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07" />
                    </svg>
                )}
            </button>
        </>
    );
});
