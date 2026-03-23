import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Toast } from "../Toast";

describe("Toast", () => {
  it("renders error banner with action and triggers callbacks", () => {
    const onClose = vi.fn();
    const onAction = vi.fn();
    render(
      <Toast
        message="Model returned an empty response."
        type="error"
        onClose={onClose}
        actionLabel="Recover"
        onAction={onAction}
      />
    );

    expect(screen.getByText("Model returned an empty response.")).toBeTruthy();
    const action = screen.getByRole("button", { name: "Recover" });
    fireEvent.click(action);
    expect(onAction).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
