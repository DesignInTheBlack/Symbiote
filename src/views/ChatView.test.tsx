import { createRef, forwardRef } from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import type { Message, Settings } from "../types/app";
import type { VoiceControllerHandle, VoiceStatus } from "../components/VoiceController";
import { invokeWithTimeout } from "../utils/tauri";

vi.mock("../components/VoiceController", () => ({
  VoiceController: forwardRef(() => null),
}));

vi.mock("../components/StatusStrip", () => ({
  StatusStrip: () => null,
}));

vi.mock("../components/MessageContent", () => ({
  MessageContent: ({ content }: { content: string }) => <span>{content}</span>,
}));

vi.mock("../utils/tauri", () => ({
  invokeWithTimeout: vi.fn(),
}));

import { ChatView } from "./ChatView";

const mockedInvoke = invokeWithTimeout as unknown as ReturnType<typeof vi.fn>;

const buildProps = (messages: Message[]) => ({
  messages,
  streamingTokenBuffer: null,
  streamingMessageId: null,
  moduleStatus: null,
  input: "",
  feedbackMode: false,
  chatState: "idle",
  chatError: null,
  memoryError: null,
  isBusy: false,
  settings: null as Settings | null,
  showRaw: false,
  rollingSummary: null,
  rollingSummaryError: null,
  rollingSummaryErrorAt: null,
  rollingSummaryPending: false,
  liveSummary: null,
  liveSummaryError: null,
  liveSummaryErrorAt: null,
  liveSummaryPending: false,
  onSummaryOpen: undefined,
  innerMonologueEntries: [],
  innerMonologueError: null,
  onMonologueOpen: undefined,
  pendingPrompts: [],
  pendingPromptCount: 0,
  pendingPromptError: null,
  onPendingPromptSend: () => {},
  onPendingPromptDismiss: () => {},
  onPendingPromptRephrase: () => {},
  onInputChange: () => {},
  onSend: () => {},
  onStop: () => {},
  onRecover: () => {},
  onFeedbackModeChange: () => {},
  onDismissMemoryError: undefined,
  onAppendTranscript: () => {},
  voiceStatus: "ready" as VoiceStatus,
  voiceEnabled: true,
  memoryGraphOpen: false,
  memoryErrorAt: null,
  voiceRef: createRef<VoiceControllerHandle>(),
  onVoiceStatusChange: () => {},
  inputRef: createRef<HTMLTextAreaElement>(),
});

describe("ChatView", () => {
  beforeEach(() => {
    mockedInvoke.mockReset();
    mockedInvoke.mockImplementation((command: string) => {
      if (command === "record_qualia_label") {
        return Promise.resolve("label_1");
      }
      return Promise.resolve("ok");
    });
    if (!Element.prototype.scrollIntoView) {
      Element.prototype.scrollIntoView = () => {};
    }
  });

  it("does not render internal-role messages in the main chat list", () => {
    const messages: Message[] = [
      { message_id: "1", role: "internal", content: "internal note", status: "complete" },
      { message_id: "2", role: "assistant", content: "visible reply", status: "complete" },
    ];
    render(<ChatView {...buildProps(messages)} />);
    expect(screen.queryByText("internal note")).toBeNull();
    expect(screen.getByText("visible reply")).not.toBeNull();
  });

  it("hides monologue-origin assistant messages by default", () => {
    const messages: Message[] = [
      {
        message_id: "1",
        role: "assistant",
        content: "monologue surfaced",
        status: "complete",
        metadata: { origin: "monologue" },
      },
      { message_id: "2", role: "assistant", content: "normal reply", status: "complete" },
    ];
    render(<ChatView {...buildProps(messages)} />);
    expect(screen.queryByText("monologue surfaced")).toBeNull();
    expect(screen.getByText("normal reply")).not.toBeNull();
  });

  it("hides cancelled and error assistant placeholders by default", () => {
    const messages: Message[] = [
      { message_id: "1", role: "assistant", content: "cancelled", status: "cancelled" },
      { message_id: "2", role: "assistant", content: "errored", status: "error" },
      { message_id: "3", role: "assistant", content: "ok", status: "complete" },
    ];
    render(<ChatView {...buildProps(messages)} />);
    expect(screen.queryByText("cancelled")).toBeNull();
    expect(screen.queryByText("errored")).toBeNull();
    expect(screen.getByText("ok")).not.toBeNull();
  });

  it("shows qualia icon for assistant messages", () => {
    const messages: Message[] = [
      { message_id: "2", role: "assistant", content: "visible reply", status: "complete" },
    ];
    render(<ChatView {...buildProps(messages)} />);
    expect(screen.getByLabelText(/qualia controls/i)).toBeTruthy();
  });

  it("opens qualia menu and records reward", async () => {
    const messages: Message[] = [
      { message_id: "2", role: "assistant", content: "visible reply", status: "complete" },
    ];
    render(<ChatView {...buildProps(messages)} />);
    fireEvent.click(screen.getByLabelText(/qualia controls/i));
    expect(screen.getByText(/intensity/i)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Tag" }));
    await waitFor(() => {
      expect(mockedInvoke).toHaveBeenCalledWith(
        "record_qualia_label",
        expect.any(Object),
        15000
      );
    });
    await waitFor(() => {
      expect(screen.getByText(/saved/i)).toBeTruthy();
    });

    const rewardButton = screen.getByRole("button", { name: "+Reward" });
    fireEvent.click(rewardButton);
    await waitFor(() => {
      expect(mockedInvoke).toHaveBeenCalledWith(
        "record_qualia_reward",
        expect.any(Object),
        15000
      );
    });
  });

  it("renders feedback toggle in the summary icon row", () => {
    const messages: Message[] = [];
    render(<ChatView {...buildProps(messages)} />);
    const feedbackButton = screen.getByLabelText(/feedback off/i);
    expect(feedbackButton).toBeTruthy();
    expect(feedbackButton.closest(".summary-eye-container")).toBeTruthy();
  });
});
