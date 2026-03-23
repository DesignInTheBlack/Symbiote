import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

import { SystemControlPanel, SubsystemState } from "../SystemControlPanel";
import { invokeWithTimeout } from "../../utils/tauri";

vi.mock("../../utils/tauri", () => ({
  invokeWithTimeout: vi.fn(),
}));

const mockedInvoke = invokeWithTimeout as unknown as ReturnType<typeof vi.fn>;

const baseSubsystem = (): SubsystemState => ({
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
});

describe("SystemControlPanel", () => {
  beforeEach(() => {
    mockedInvoke.mockReset();
  });

  it("renders subsystem list", () => {
    render(
      <SystemControlPanel
        subsystemStates={[baseSubsystem()]}
        allowWrites={false}
        onRefresh={() => {}}
      />
    );
    expect(screen.getByText(/kernel loop/i)).toBeTruthy();
  });

  it("requires confirmation for critical disables", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(
      <SystemControlPanel
        subsystemStates={[baseSubsystem()]}
        allowWrites={true}
        onRefresh={() => {}}
      />
    );

    const select = screen.getByRole("combobox");
    fireEvent.change(select, { target: { value: "off" } });
    fireEvent.change(screen.getByPlaceholderText("Reason"), { target: { value: "test" } });
    fireEvent.click(screen.getByRole("button", { name: /apply/i }));

    expect(mockedInvoke).not.toHaveBeenCalled();

    confirmSpy.mockReturnValue(true);
    fireEvent.click(screen.getByRole("button", { name: /apply/i }));

    expect(mockedInvoke).toHaveBeenCalled();
    confirmSpy.mockRestore();
  });
});
