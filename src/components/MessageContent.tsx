interface MessageContentProps {
  content: string;
  showRaw: boolean;
}

export const MessageContent = ({
  content,
  showRaw,
}: MessageContentProps) => {
  if (showRaw) {
    return (
      <div className="message-raw">
        {content}
      </div>
    );
  }

  const parseReminderBlock = (block: string) => {
    let inner = block.replace(/^```reminder/i, "");
    if (inner.endsWith("```")) {
      inner = inner.slice(0, -3);
    }

    const result = { content: "", dueIn: "", type: "REMINDER" };

    for (const line of inner.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#") || trimmed.startsWith("//")) continue;
      const match = trimmed.match(/^([a-zA-Z_]+)\s*[:=]\s*(.+)$/);
      if (!match) continue;

      const key = match[1].toLowerCase();
      let value = match[2].trim();
      if ((value.startsWith("\"") && value.endsWith("\"")) || (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }

      if (key === "remind" || key === "content") result.content = value;
      else if (key === "due_in" || key === "due") result.dueIn = value;
      else if (key === "type") result.type = value.toUpperCase();
    }

    return result;
  };

  const sanitizeProtocol = (text: string) =>
    text
      .replace(/<<\s*(MEMORY|CLARIFY|RESOLVE)\s*>>/gi, "")
      .replace(/<attribution>[\s\S]*?<\/attribution>/gi, "")
      .replace(/<state_ref>[\s\S]*?<\/state_ref>/gi, "");

  const sanitized = sanitizeProtocol(content);

  // Regex to split: Group 1: ```reminder...```, Group 2: [[REMINDER:CREATED...]], Group 3: <code>...</code>, Group 4: ```...```, Group 5: [[TASK:CREATE...]]
  const parts = sanitized.split(/((?:```reminder[\s\S]*?(?:```|$))|(?:\[\[REMINDER:CREATED[\s\S]*?\]\])|(?:<code>[\s\S]*?(?:<\/code>|$))|(?:```[\s\S]*?(?:```|$))|(?:\[\[TASK:CREATE[\s\S]*?\]\]))/g);

  const renderReminderCard = (key: number, title: string, reminderContent: string, dueIn: string) => (
    <div key={key} className="reminder-card">
      <div className="reminder-card__title">{title}</div>
      <div className="reminder-card__content">{reminderContent}</div>
      <div className="reminder-card__meta">
        <svg className="reminder-card__icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></svg>
        Due in {dueIn}
      </div>
    </div>
  );

  return (
    <>
      {parts.map((part, index) => {
        if (part.startsWith("<code>")) {
          let code = part.slice(6);
          if (code.endsWith("</code>")) code = code.slice(0, -7);
          return <div key={index} className="code-block">{code}</div>;
        }
        else if (part.startsWith("```reminder")) {
          const reminder = parseReminderBlock(part);
          if (!reminder.content || !reminder.dueIn) {
            return null;
          }

          return renderReminderCard(index, `${reminder.type} SET`, reminder.content, reminder.dueIn);
        }
        else if (part.startsWith("```")) {
          let code = part.slice(3);
          if (code.endsWith("```")) {
            code = code.slice(0, -3);
          }

          // Heuristic: Remove language identifier
          const firstLineBreak = code.indexOf("\n");
          if (firstLineBreak > -1 && firstLineBreak < 20) {
            code = code.slice(firstLineBreak + 1);
          } else if (code.trim().startsWith("python ")) {
            code = code.slice(7);
          }

          return <div key={index} className="code-block">{code}</div>;
        } else if (part.startsWith("[[REMINDER:CREATED")) {
          const extract = (key: string) => {
            const match = part.match(new RegExp(`${key}=['"](.*?)['"]`));
            return match ? match[1] : "?";
          };
          const content = extract("content");
          const dueIn = extract("due_in");
          const type = extract("type");

          return renderReminderCard(index, `${type} SET`, content, dueIn);
        } else if (part.startsWith("[[TASK:CREATE")) {
          // LEGACY: Old tag-based syntax (kept for backward compatibility)
          const extract = (key: string) => {
            const match = part.match(new RegExp(`${key}=['"](.*?)['"]`));
            return match ? match[1] : "?";
          };
          const content = extract("content");
          const dueIn = extract("due_in");
          const type = extract("type");

          return renderReminderCard(index, `${type} SET`, content, dueIn);
        }
        else {
          // For normal text, we want to HIDE <silent> tags but SHOW the content
          let text = part.replace(/<\/?silent>/g, "");
          return <span key={index}>{text}</span>;
        }
      })}
    </>
  );
};
