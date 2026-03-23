import { getCurrentWindow } from "@tauri-apps/api/window";

export const TitleBar = () => {
  const appWindow = getCurrentWindow();

  const handleMinimize = async () => {
    await appWindow.minimize();
  };

  const handleMaximize = async () => {
    const isMax = await appWindow.isMaximized();
    if (isMax) {
      await appWindow.unmaximize();
    } else {
      await appWindow.maximize();
    }
  };

  const handleClose = async () => {
    await appWindow.close();
  };

  return (
    <div className="titlebar" onDoubleClick={handleMaximize}>
      <div className="titlebar-left" data-tauri-drag-region>
        <span className="titlebar-app">Symbiote</span>
      </div>
      <div className="titlebar-drag" data-tauri-drag-region></div>
      <div className="titlebar-controls">
        <button className="titlebar-btn" aria-label="Minimize" onClick={handleMinimize}>
          <svg
            className="titlebar-icon"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
          >
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
        </button>
        <button className="titlebar-btn" aria-label="Maximize" onClick={handleMaximize}>
          <svg
            className="titlebar-icon"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <rect x="6" y="6" width="12" height="12" rx="1" ry="1" />
          </svg>
        </button>
        <button className="titlebar-btn close" aria-label="Close" onClick={handleClose}>
          <svg
            className="titlebar-icon"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>
    </div>
  );
};
