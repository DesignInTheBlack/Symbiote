import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SystemHealthPanel } from "../SystemHealthPanel";
import type { SystemHealthSnapshot } from "../../types/app";

const buildSnapshot = (): SystemHealthSnapshot => ({
  snapshot_id: "snap_1",
  timestamp: new Date().toISOString(),
  run_id: null,
  trace_id: null,
  metrics: {
    gate: { total: 4, verify_rate: 0.25, counts: { ALLOW_WITH_NOTICE: 1, ALLOW_WITH_AUDIT: 1 } },
    organism: { stress: 0.2, fatigue: 0.3, social_alignment: 0.6 },
    controller: { confidence: 0.6, uncertainty: 0.4, failure_streak: 1 },
    qualia: { labels: 2, rewards: 1, mean_intensity: 0.4 },
    errors: { total: 1, open: 0 },
    memory: { memory_pass_count: 3, write_count: 2, last_error_at: "" },
    summaries: { rolling_updates: 1, inner_updates: 1, rolling_failures: 0, inner_failures: 0 },
    pending_prompts: { count: 1, oldest_age_seconds: 10, oldest_at: "" },
    avatar: { health: 0.8, processing_phase: "idle", memory_activity: 0.3 },
    logs: { warn: 0 },
  },
  subsystem_states: [],
});

describe("SystemHealthPanel", () => {
  it("renders core metrics", () => {
    const snapshot = buildSnapshot();
    render(<SystemHealthPanel snapshot={snapshot} history={[snapshot]} />);
    expect(screen.getByText(/system health/i)).toBeTruthy();
    expect(screen.getByText(/verify rate/i)).toBeTruthy();
    expect(screen.getByText(/organism/i)).toBeTruthy();
    expect(screen.getByText(/controller/i)).toBeTruthy();
  });
});
