import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import { TraceView } from "../TraceView";
import type { SystemLogEntry, SystemHealthSnapshot, SystemControlEntry, SystemControlEvent } from "../../types/app";

const buildSnapshot = (): SystemHealthSnapshot => ({
  snapshot_id: "snap_1",
  timestamp: new Date().toISOString(),
  run_id: null,
  trace_id: null,
  metrics: {
    avatar: { health: 0.5, processing_phase: "idle" },
    gate: { total: 0, verify_rate: 0 },
    organism: { stress: 0, fatigue: 0, social_alignment: 0.5 },
  },
  subsystem_states: [
    {
      id: "kernel_loop",
      label: "Kernel loop",
      class: "critical",
      default_mode: "normal",
      supported_modes: ["normal", "off"],
      depends_on: [],
      enforcement_notes: "Primary execution loop.",
      mode: "normal",
      updated_at: "",
      updated_by: "",
      reason: "",
      value_json: null,
    },
  ],
});

describe("TraceView", () => {
  test("renders cockpit panels and system log payload JSON", () => {
    const entry: SystemLogEntry = {
      id: "log_1",
      timestamp: new Date().toISOString(),
      level: "info",
      category: "kernel",
      payload: {
        event: "memory_pass_start",
        detail: "ok",
      },
    };
    const snapshot = buildSnapshot();
    const controlEvents: SystemControlEvent[] = [
      {
        event_id: "evt_1",
        subsystem_id: "kernel_loop",
        previous_mode: "normal",
        new_mode: "degraded",
        value_json: null,
        actor: "ui",
        reason: "test",
        status: "applied",
        timestamp: new Date().toISOString(),
      },
    ];

    render(
      <TraceView
        systemLogs={[entry]}
        systemControls={[] as SystemControlEntry[]}
        systemControlEvents={controlEvents}
        systemHealthSnapshot={snapshot}
        systemHealthHistory={[snapshot]}
        gateInputEvents={[]}
        allowControlWrites={false}
        cockpitWriteEnabled={false}
        onToggleCockpitWrite={() => {}}
        onRefresh={() => {}}
        onClear={() => {}}
      />,
    );

    expect(screen.getByText(/system cockpit/i)).toBeTruthy();
    expect(screen.getByText(/memory_pass_start/i)).toBeTruthy();
    expect(screen.getByText(/kernel loop/i)).toBeTruthy();
    expect(screen.getByText(/control history/i)).toBeTruthy();
    expect(screen.getByText(/health timeline/i)).toBeTruthy();
  });
});
