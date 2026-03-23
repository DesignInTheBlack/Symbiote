interface StatusStripProps {
  chatState: string;
  voiceStatus: string;
  memoryGraphOpen: boolean;
  chatError?: string | null;
  memoryErrorAt?: string | null;
  onRecoverChat?: () => void;
}

const formatLabel = (value: string) => value.replace(/_/g, " ");

export const StatusStrip = ({
  chatState,
  voiceStatus,
  memoryGraphOpen,
  chatError,
  memoryErrorAt,
  onRecoverChat,
}: StatusStripProps) => {
  const chatActive = chatState === "sending" || chatState === "streaming" || chatState === "post_processing";
  const voiceActive = voiceStatus === "recording" || voiceStatus === "speaking";
  const hasMemoryError = Boolean(memoryErrorAt);

  return (
    <div className="status-strip">
      <div className="status-item" title={chatError || undefined}>
        <span
          className={`status-dot ${chatActive ? "active" : ""} ${chatState === "error" ? "error" : ""}`}
        />
        <span>Chat: {formatLabel(chatState)}</span>
        {chatState === "error" && onRecoverChat && (
          <button className="status-action" onClick={onRecoverChat}>
            Recover
          </button>
        )}
      </div>
      <div className="status-item">
        <span className={`status-dot ${voiceActive ? "active" : ""} ${voiceStatus === "error" ? "error" : ""}`} />
        <span>Voice: {formatLabel(voiceStatus)}</span>
      </div>
      <div className="status-item">
        <span className={`status-dot ${memoryGraphOpen ? "active" : ""}`} />
        <span>Memory Graph: {memoryGraphOpen ? "open" : "closed"}</span>
      </div>
      <div className="status-item" title={memoryErrorAt || undefined}>
        <span className={`status-dot ${hasMemoryError ? "error" : ""}`} />
        <span>Memory: {hasMemoryError ? "error" : "ok"}</span>
      </div>
    </div>
  );
};
