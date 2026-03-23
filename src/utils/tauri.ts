import { invoke } from "@tauri-apps/api/core";
import { withTimeout } from "./async";

export const invokeWithTimeout = <T>(
  command: string,
  args?: Record<string, unknown>,
  ms = 15000,
) => withTimeout(invoke<T>(command, args), ms, command);
