import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MessageContent } from "../MessageContent";

describe("MessageContent", () => {
  it("strips protocol tags and attribution blocks from rendered output", () => {
    const content = [
      "Hello",
      "<<MEMORY>>",
      "<attribution>internal attribution</attribution>",
      "<state_ref>internal state</state_ref>",
      "World",
    ].join("\n");

    const { container } = render(<MessageContent content={content} showRaw={false} />);
    const text = container.textContent ?? "";
    expect(text).toContain("Hello");
    expect(text).toContain("World");
    expect(text).not.toContain("MEMORY");
    expect(text).not.toContain("internal attribution");
    expect(text).not.toContain("state_ref");
  });
});
