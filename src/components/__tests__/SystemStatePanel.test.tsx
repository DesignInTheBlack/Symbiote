import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SystemStatePanel } from "../SystemStatePanel";
import type { SystemHealthSnapshot } from "../../types/app";

const buildSnapshot = (): SystemHealthSnapshot => ({
  snapshot_id: "snap_state",
  timestamp: new Date().toISOString(),
  run_id: null,
  trace_id: null,
  metrics: {
    avatar: {
      health: 0.82,
      certainty: 0.65,
      memory_activity: 0.4,
      gate_activity: 0.2,
      pending_prompts: 2,
      organism: { stress: 0.1, fatigue: 0.2, social_alignment: 0.7 },
    },
    controller: { confidence: 0.65 },
    organism: { stress: 0.1, fatigue: 0.2, social_alignment: 0.7 },
  },
  subsystem_states: [],
});

describe("SystemStatePanel", () => {
  it("maps health snapshot metrics into avatar styles", () => {
    const snapshot = buildSnapshot();
    const { container } = render(
      <SystemStatePanel
        chatState="idle"
        moduleStatus={{ stage: "prompt_build", detail: null, started_at: null }}
        memoryError={null}
        rollingSummaryError={null}
        pendingPromptCount={2}
        healthSnapshot={snapshot}
      />
    );

    const blob = container.querySelector(".system-state-blob") as HTMLElement | null;
    expect(blob).toBeTruthy();
    const health = blob?.style.getPropertyValue("--avatar-health") || "0";
    expect(Number(health)).toBeGreaterThan(0.8);
  });
});
