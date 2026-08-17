const STYLE_ID = "fl-h3-temporal-reshot-modal-styles";

const STYLES = `
  .flh3r-compact,
  .flh3r-modal-shell {
    --flh3r-accent: #a78bfa;
    --flh3r-accent-strong: #7c3aed;
    --flh3r-border: #34343d;
    --flh3r-control: #252529;
    --flh3r-muted: #9696a1;
    color: #e4e4e7;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    box-sizing: border-box;
  }
  .flh3r-compact *, .flh3r-modal-shell * { box-sizing: border-box; }
  .flh3r-button {
    min-height: 25px;
    padding: 4px 8px;
    color: #d4d4d8;
    background: var(--flh3r-control);
    border: 1px solid var(--flh3r-border);
    border-radius: 5px;
    font: inherit;
    cursor: pointer;
  }
  .flh3r-button:hover { color: #fff; border-color: var(--flh3r-accent); }
  .flh3r-button.primary { color: #fff; background: #6d28d9; border-color: #8b5cf6; }
  .flh3r-button:disabled { opacity: .38; cursor: default; }
  .flh3r-spacer { flex: 1 1 auto; }

  .flh3r-compact {
    width: 100%;
    height: 100%;
    min-height: 0;
    display: grid;
    grid-template-rows: auto 28px;
    gap: 5px;
    padding: 5px;
    overflow: hidden;
    background: #17171b;
    border: 1px solid var(--flh3r-border);
    border-radius: 8px;
  }
  .flh3r-compact-preview {
    aspect-ratio: 16 / 9;
    min-height: 0;
    position: relative;
    overflow: hidden;
    background: #050507;
    border: 1px solid #303038;
    border-radius: 6px;
  }
  .flh3r-compact-preview video { width: 100%; height: 100%; display: block; object-fit: contain; }
  .flh3r-compact-empty {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    padding: 12px;
    color: #777783;
    background: radial-gradient(circle, rgba(124, 58, 237, .13), transparent 60%);
    font-size: 10px;
    text-align: center;
  }
  .flh3r-compact-empty[hidden] { display: none; }
  .flh3r-compact-play {
    width: 31px;
    height: 27px;
    position: absolute;
    left: 6px;
    bottom: 6px;
    padding: 0;
    color: #fff;
    background: rgba(24, 24, 29, .9);
    border: 1px solid rgba(255, 255, 255, .16);
    border-radius: 5px;
    cursor: pointer;
  }
  .flh3r-compact-status {
    min-width: 0;
    overflow: hidden;
    padding: 6px 8px;
    color: #a1a1aa;
    background: #222228;
    border-radius: 5px;
    font-size: 9px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .flh3r-compact-status[data-tone="ready"] { color: #bbf7d0; background: #14532d; }
  .flh3r-compact-status[data-tone="busy"] { color: #ddd6fe; background: #312e81; }
  .flh3r-compact-status[data-tone="error"] { color: #fecaca; background: #7f1d1d; }

  .flh3r-modal-overlay {
    position: fixed;
    inset: 0;
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2.5vh 2.5vw;
    background: rgba(0, 0, 0, .84);
    backdrop-filter: blur(4px);
    animation: flh3r-fade-in .15s ease-out;
  }
  .flh3r-modal-shell {
    width: 95vw;
    height: 94vh;
    max-width: 1900px;
    max-height: 1400px;
    min-width: 780px;
    min-height: 620px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: #111114;
    border: 1px solid #3f3f46;
    border-radius: 12px;
    box-shadow: 0 24px 80px rgba(0, 0, 0, .72);
    animation: flh3r-modal-in .18s ease-out;
  }
  .flh3r-modal-header {
    min-height: 54px;
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px 9px 16px;
    background: #1b1b20;
    border-bottom: 1px solid #303036;
  }
  .flh3r-modal-heading { min-width: 0; display: flex; flex-direction: column; gap: 2px; }
  .flh3r-modal-title { color: #fafafa; font-size: 14px; font-weight: 700; }
  .flh3r-modal-subtitle {
    max-width: 65vw;
    overflow: hidden;
    color: #a1a1aa;
    font-size: 10px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .flh3r-modal-main { flex: 1 1 auto; min-height: 0; display: flex; }

  .flh3r-library {
    width: 300px;
    flex: 0 0 300px;
    min-height: 0;
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 9px;
    padding: 11px;
    background: #17171b;
    border-right: 1px solid #303036;
    transition: width .16s ease, flex-basis .16s ease, padding .16s ease;
  }
  .flh3r-modal-shell.library-collapsed .flh3r-library { width: 14px; flex-basis: 14px; padding: 0; }
  .flh3r-modal-shell.library-collapsed .flh3r-library > :not(.flh3r-sidebar-toggle) {
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
  }
  .flh3r-library-label { color: #8b8b95; font-size: 8px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
  .flh3r-drop-zone {
    min-height: 72px;
    display: grid;
    place-items: center;
    padding: 10px;
    color: #a1a1aa;
    background: #202027;
    border: 1px dashed #52525b;
    border-radius: 7px;
    font-size: 10px;
    line-height: 1.4;
    text-align: center;
    cursor: pointer;
  }
  .flh3r-drop-zone.dragging { color: #ede9fe; background: #312e81; border-color: #a78bfa; }
  .flh3r-library-actions { display: flex; gap: 6px; }
  .flh3r-library-search, .flh3r-library-folder,
  .flh3r-inspector input, .flh3r-inspector textarea, .flh3r-inspector select {
    width: 100%;
    color: #f4f4f5;
    background: var(--flh3r-control);
    border: 1px solid #3f3f46;
    border-radius: 5px;
    outline: none;
    font: inherit;
  }
  .flh3r-library-search, .flh3r-library-folder { height: 29px; padding: 4px 7px; font-size: 9px; }
  .flh3r-library-results { flex: 1 1 auto; min-height: 0; overflow-y: auto; }
  .flh3r-library-row {
    width: 100%;
    display: grid;
    gap: 3px;
    margin: 0 0 5px;
    padding: 8px;
    color: #c4c4cc;
    background: #202026;
    border: 1px solid #303037;
    border-radius: 7px;
    text-align: left;
    cursor: pointer;
  }
  .flh3r-library-row:hover { border-color: #5b5b67; }
  .flh3r-library-row.active { background: #2e2347; border-color: #8b5cf6; }
  .flh3r-library-row strong { overflow: hidden; font-size: 9px; text-overflow: ellipsis; white-space: nowrap; }
  .flh3r-library-row small { overflow: hidden; color: #858590; font-size: 7.5px; text-overflow: ellipsis; white-space: nowrap; }
  .flh3r-library-message { min-height: 16px; color: #8b8b95; font-size: 8px; }
  .flh3r-library-message.error { color: #fda4af; }
  .flh3r-source-card { padding: 8px; color: #a1a1aa; background: #202027; border: 1px solid #303037; border-radius: 7px; font-size: 8px; line-height: 1.45; }
  .flh3r-sidebar-toggle {
    width: 28px;
    height: 52px;
    position: absolute;
    z-index: 4;
    top: 50%;
    right: -14px;
    padding: 0;
    color: #a1a1aa;
    background: #202027;
    border: 1px solid #3f3f46;
    border-radius: 0 8px 8px 0;
    font-size: 20px;
    transform: translateY(-50%);
    cursor: pointer;
  }

  .flh3r-editor-host { flex: 1 1 auto; min-width: 0; min-height: 0; }
  .flh3r-editor {
    width: 100%;
    height: 100%;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: #121216;
  }
  .flh3r-transport, .flh3r-toolbar, .flh3r-footer {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 7px 9px;
    background: #1b1b20;
    border-bottom: 1px solid #2b2b31;
  }
  .flh3r-toolbar { padding-block: 6px; background: #17191e; }
  .flh3r-footer { color: #777783; border-top: 1px solid #2b2b31; border-bottom: 0; font-size: 8px; }
  .flh3r-time { min-width: 145px; color: #fbbf24; font: 10px "Cascadia Mono", Consolas, monospace; }
  .flh3r-source-label { min-width: 0; overflow: hidden; color: #a1a1aa; font-size: 9px; text-overflow: ellipsis; white-space: nowrap; }
  .flh3r-volume { display: flex; align-items: center; gap: 5px; color: #a1a1aa; font-size: 8px; }
  .flh3r-volume input { width: 75px; accent-color: var(--flh3r-accent); }
  .flh3r-status { max-width: 360px; overflow: hidden; padding: 4px 8px; color: #a1a1aa; background: #27272a; border-radius: 10px; font-size: 9px; text-overflow: ellipsis; white-space: nowrap; }
  .flh3r-status[data-tone="ready"] { color: #d1fae5; background: #065f46; }
  .flh3r-status[data-tone="busy"] { color: #ddd6fe; background: #312e81; }
  .flh3r-status[data-tone="error"] { color: #fee2e2; background: #7f1d1d; }
  .flh3r-legend { display: flex; align-items: center; gap: 10px; color: #92929d; font-size: 8px; white-space: nowrap; }
  .flh3r-legend i { width: 10px; height: 7px; display: inline-block; margin-right: 3px; border-radius: 2px; }
  .flh3r-legend .source { background: #3f424c; }
  .flh3r-legend .context { background: #0e7490; }
  .flh3r-legend .selection { background: #7c3aed; }
  .flh3r-legend .tokens { background: #b45309; }
  .flh3r-legend .padding { background: #ca8a04; }
  .flh3r-video-stage {
    height: clamp(190px, 34vh, 360px);
    flex: 0 1 360px;
    min-height: 170px;
    position: relative;
    overflow: hidden;
    background: #050507;
    border-bottom: 1px solid #2b2b31;
  }
  .flh3r-video-stage video { width: 100%; height: 100%; display: block; object-fit: contain; }
  .flh3r-video-overlay {
    max-width: calc(100% - 16px);
    position: absolute;
    left: 8px;
    bottom: 8px;
    overflow: hidden;
    padding: 5px 8px;
    color: #d4d4d8;
    background: rgba(17, 17, 20, .84);
    border-radius: 5px;
    font-size: 8px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .flh3r-canvas-wrap { flex: 1 1 auto; min-height: 330px; position: relative; overflow: hidden; background: #0b0b0e; }
  .flh3r-canvas { width: 100%; height: 100%; display: block; outline: none; touch-action: none; }
  .flh3r-empty { position: absolute; inset: 0; display: grid; place-items: center; color: #71717a; pointer-events: none; }
  .flh3r-empty[hidden] { display: none; }

  .flh3r-inspector {
    width: 330px;
    flex: 0 0 330px;
    min-height: 0;
    overflow-y: auto;
    padding: 11px;
    background: #17171b;
    border-left: 1px solid #303036;
  }
  .flh3r-inspector-section { margin-bottom: 10px; padding: 10px; background: #202027; border: 1px solid #303037; border-radius: 8px; }
  .flh3r-inspector-title { margin-bottom: 8px; color: #ddd6fe; font-size: 9px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
  .flh3r-inspector-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
  .flh3r-field { min-width: 0; display: grid; gap: 3px; color: #898994; font-size: 8px; }
  .flh3r-field input, .flh3r-field select { height: 29px; padding: 4px 6px; }
  .flh3r-field.full { grid-column: 1 / -1; }
  .flh3r-inspector textarea { min-height: 145px; padding: 8px; resize: vertical; line-height: 1.45; }
  .flh3r-inspector input:focus, .flh3r-inspector textarea:focus, .flh3r-inspector select:focus,
  .flh3r-library-search:focus, .flh3r-library-folder:focus { border-color: var(--flh3r-accent); }
  .flh3r-diagnostics { display: grid; gap: 5px; margin: 0; }
  .flh3r-diagnostics div { display: flex; justify-content: space-between; gap: 8px; color: #8b8b95; font-size: 8px; }
  .flh3r-diagnostics dt, .flh3r-diagnostics dd { margin: 0; }
  .flh3r-diagnostics dd { color: #d4d4d8; font-family: "Cascadia Mono", Consolas, monospace; text-align: right; }
  .flh3r-notice { margin-top: 7px; padding: 7px; color: #a1a1aa; background: #18181d; border-left: 2px solid #52525b; border-radius: 4px; font-size: 8px; line-height: 1.45; }
  .flh3r-notice.warning { color: #fde68a; border-left-color: #f59e0b; }
  .flh3r-notice.error { color: #fecaca; border-left-color: #ef4444; }
  .flh3r-reference-status { color: #c4b5fd; }

  @keyframes flh3r-fade-in { from { opacity: 0; } to { opacity: 1; } }
  @keyframes flh3r-modal-in { from { opacity: 0; transform: scale(.975) translateY(-8px); } to { opacity: 1; transform: scale(1) translateY(0); } }
  @media (max-width: 1100px) {
    .flh3r-modal-overlay { padding: 0; }
    .flh3r-modal-shell { width: 100vw; height: 100vh; min-width: 0; min-height: 0; border-radius: 0; }
    .flh3r-library { width: 245px; flex-basis: 245px; }
    .flh3r-inspector { width: 280px; flex-basis: 280px; }
    .flh3r-legend { display: none; }
  }
`;

export function injectTemporalReshotStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = STYLES;
  document.head.appendChild(style);
}
