export type UiDebugFlags = {
  simulateTimeouts: boolean;
  simulateFailures: boolean;
};

const TIMEOUT_KEY = "ui.simulateTimeouts";
const FAILURE_KEY = "ui.simulateFailures";

export class TimeoutError extends Error {
  timeoutMs: number;
  label: string;

  constructor(label: string, timeoutMs: number) {
    super(`Timed out after ${timeoutMs}ms${label ? `: ${label}` : ""}`);
    this.name = "TimeoutError";
    this.timeoutMs = timeoutMs;
    this.label = label;
  }
}

export const getUiDebugFlags = (): UiDebugFlags => {
  if (typeof window === "undefined") {
    return { simulateTimeouts: false, simulateFailures: false };
  }

  return {
    simulateTimeouts: window.localStorage.getItem(TIMEOUT_KEY) === "1",
    simulateFailures: window.localStorage.getItem(FAILURE_KEY) === "1",
  };
};

export const setUiDebugFlags = (flags: UiDebugFlags) => {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(TIMEOUT_KEY, flags.simulateTimeouts ? "1" : "0");
  window.localStorage.setItem(FAILURE_KEY, flags.simulateFailures ? "1" : "0");
};

export const withTimeout = async <T>(promise: Promise<T>, ms: number, label = "operation"): Promise<T> => {
  const flags = getUiDebugFlags();
  if (flags.simulateFailures) {
    throw new Error(`Simulated failure: ${label}`);
  }

  const timeoutMs = flags.simulateTimeouts ? Math.min(ms, 50) : ms;
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  const timeoutPromise = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(() => reject(new TimeoutError(label, timeoutMs)), timeoutMs);
  });

  try {
    return await Promise.race([promise, timeoutPromise]);
  } finally {
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
    }
  }
};
