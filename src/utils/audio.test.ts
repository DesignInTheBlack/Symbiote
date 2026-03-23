import { describe, expect, it, vi } from "vitest";
import { resumeAudioContext } from "./audio";

describe("resumeAudioContext", () => {
  it("resumes a suspended AudioContext", async () => {
    const resume = vi.fn().mockResolvedValue(undefined);
    const ctx = {
      state: "suspended",
      resume,
    } as unknown as AudioContext;

    await resumeAudioContext(ctx);
    expect(resume).toHaveBeenCalledTimes(1);
  });

  it("does not resume when running", async () => {
    const resume = vi.fn().mockResolvedValue(undefined);
    const ctx = {
      state: "running",
      resume,
    } as unknown as AudioContext;

    await resumeAudioContext(ctx);
    expect(resume).not.toHaveBeenCalled();
  });
});
