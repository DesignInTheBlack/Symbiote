# Themes

This project uses single-file CSS themes that override shared CSS variables.

## Where themes live
- Built-ins: `public/themes/*.css`
- User themes: the app data `themes` folder shown in Settings

A theme file is just CSS with a `:root { ... }` block. You can also add
extra selectors if you want to override component styles directly.

## How themes are applied
- The theme loader injects the theme CSS into a `<style id="symbiote-theme">` tag.
- `document.documentElement.dataset.theme` is set to the active theme ID.
- A `symbiote-theme-change` event is dispatched after applying a theme.

## Theme variables (contract)
### Core colors
- `--bg-color`
- `--surface-color`
- `--surface-gradient`
- `--accent-color`
- `--accent-color-hover`
- `--accent-primary`
- `--accent-secondary`
- `--text-primary`
- `--text-secondary`
- `--text-on-accent`
- `--border-color`
- `--error-color`
- `--success-color`

### Typography
- `--font-family`
- `--font-header`
- `--font-mono`

### Depth / neumorphism
- `--neu-flat`
- `--neu-pressed`
- `--neu-convex`
- `--neu-border`
- `--chamfer-top`
- `--chamfer-bottom`

### Title bar
- `--titlebar-height`
- `--titlebar-bg`
- `--titlebar-text`
- `--titlebar-border`
- `--titlebar-shadow`
- `--titlebar-control-hover`
- `--titlebar-control-close-hover`
- `--titlebar-control-close-text`

### Surfaces & overlays
- `--surface-overlay`
- `--surface-overlay-border`
- `--surface-muted`
- `--sidebar-shadow`
- `--scrollbar-thumb-hover`

### Code / raw text / terminal
- `--code-bg`
- `--code-border`
- `--code-text`
- `--raw-bg`
- `--raw-border`
- `--raw-text`
- `--terminal-bg`
- `--terminal-text`
- `--terminal-shadow`
- `--terminal-select-bg`

### Reminders
- `--reminder-card-bg`
- `--reminder-card-border`
- `--reminder-card-shadow`
- `--reminder-card-title`
- `--reminder-card-text`
- `--reminder-card-meta`
- `--reminder-card-icon`

### Inputs
- `--input-inset-shadow`
- `--input-border`
- `--input-placeholder`
- `--input-focus-shadow`

### Status + badges
- `--status-success`
- `--status-success-soft`
- `--status-success-border`
- `--status-warning`
- `--status-caution`
- `--status-caution-soft`
- `--status-caution-border`
- `--status-danger`
- `--status-danger-soft`
- `--status-danger-border`
- `--status-neutral`
- `--danger-color`
- `--danger-solid`
- `--danger-solid-text`
- `--badge-text`
- `--badge-shadow`
- `--accent-glow`
- `--time-badge-bg`
- `--time-badge-text`
- `--time-badge-border`
- `--status-action-border`
- `--status-action-hover-text`

### System state + trace UI (derived)
The system state avatar and TraceView rely on the status tokens above plus:
- `--surface-muted`
- `--surface-overlay`
- `--surface-overlay-border`
- `--code-bg`
- `--code-border`
- `--accent-color`
- `--success-color`
- `--error-color`

### System avatar (themeable)
The system avatar visuals read these explicit tokens:
- `--cockpit-avatar-bg`
- `--avatar-saturation`
- `--avatar-lightness`
- `--avatar-field-alpha`
- `--avatar-scan-alpha`
- `--avatar-noise-alpha`
- `--avatar-core-alpha`
- `--avatar-glow-alpha`

### Toasts
- `--toast-bg`
- `--toast-border`
- `--toast-shadow`
- `--toast-text`
- `--toast-muted`
- `--toast-action-border`

### Mic states
- `--mic-recording-bg`
- `--mic-speaking-bg`
- `--mic-error-bg`
- `--mic-reconnecting-bg`

### Graph + memory visuals
- `--memory-graph-bg`
- `--graph-grid`
- `--graph-minimap-stroke`
- `--graph-minimap-mask`
- `--graph-activate`
- `--graph-activate-shadow`
- `--graph-warning`
- `--graph-warning-glow`

### Graph palette
- `--graph-node-person`
- `--graph-node-place`
- `--graph-node-work`
- `--graph-node-concept`
- `--graph-node-event`
- `--graph-node-project`
- `--graph-node-system`
- `--graph-node-conflict`
- `--graph-node-default`
- `--graph-node-muted`
- `--graph-link-fact`
- `--graph-link-supersedes`
- `--graph-link-contradicts`
- `--graph-link-supports`
- `--graph-link-derived`
- `--graph-link-relation`
- `--graph-link-other`
- `--graph-label-base`
- `--graph-label-bg`
- `--graph-label-selected`
- `--graph-label-hover`
- `--graph-texture-line`
- `--graph-texture-dot`

### Vortex / core effects
- `--vortex-ring-gradient`
- `--vortex-ring-shadow`
- `--vortex-core-bg`
- `--vortex-core-shadow`
- `--core-gradient`
- `--core-glow`
- `--core-hole-bg`
- `--core-hole-shadow`
- `--core-hole-text`
- `--core-white-gradient`
- `--core-white-bg`
- `--core-white-glow`
- `--core-white-shadow`
- `--core-white-text`
- `--energy-stream-color`
- `--energy-stream-glow`

### Sliders
- `--slider-track-bg`
- `--slider-track-shadow`
- `--slider-track-border`
- `--slider-thumb-bg`
- `--slider-thumb-hover-bg`
- `--slider-thumb-active-bg`
- `--slider-thumb-shadow`
- `--slider-thumb-hover-shadow`
- `--slider-thumb-active-shadow`
- `--slider-thumb-border`
- `--slider-thumb-border-top`

### Confidence bar
- `--confidence-track-shadow`
- `--confidence-fill-gradient`
- `--confidence-glow`

## Notes
- If you need to override a component directly, add selectors in your theme
  file (it is applied after the base stylesheet).
- Canvas-driven elements (memory graph) read CSS variables at runtime. Changing
  a theme will refresh them on the next `symbiote-theme-change` event.
