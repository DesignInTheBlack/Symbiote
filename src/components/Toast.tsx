interface ToastProps {
  message: string;
  type: "success" | "error";
  onClose: () => void;
  actionLabel?: string;
  onAction?: () => void;
}

export const Toast = ({ message, type, onClose, actionLabel, onAction }: ToastProps) => (
  <div className={`toast toast-${type}`}>
    <div className="toast-content">
      {type === "success" && <span className="toast-icon">OK</span>}
      {type === "error" && <span className="toast-icon">!</span>}
      <span>{message}</span>
    </div>
    {actionLabel && onAction && (
      <button
        className="toast-action"
        onClick={() => {
          onAction();
          onClose();
        }}
      >
        {actionLabel}
      </button>
    )}
    <button onClick={onClose} className="toast-close" aria-label="Close toast">x</button>
  </div>
);
