import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { addCanvasNavigation } from "./canvas_navigation.js";

const EVENT_NAME = "fl_minimax_h3_live_preview";
const MIN_PANEL_HEIGHT = 310;
const VHS_PREVIEW_EVENT = "VHS_latentpreview";
const VHS_PREVIEW_MARKER = "__vhs_latent_preview__";
const EXTERNAL_PREVIEW_WIDGETS = new Set(["$$canvas-image-preview", "vhslatentpreview"]);
const NODE_CONFIG = {
  FL_MiniMaxH3BeatKSampler: {
    stage: "sample",
    title: "Render previews",
    minHeight: 650,
  },
  FL_MiniMaxH3BeatUpscaleKSampler: {
    stage: "upscale",
    title: "Upscale previews",
    minHeight: 780,
  },
};
const instances = new Map();

const STYLES = `
  .flh3-preview-panel {
    --accent: #8b5cf6;
    --border: var(--border-color, #353842);
    --muted: var(--descrip-text, #979cab);
    background: var(--comfy-menu-bg, #18191e);
    border: 1px solid var(--border);
    border-radius: 9px;
    box-sizing: border-box;
    color: var(--input-text, #f4f4f5);
    display: grid;
    font: 11px/1.3 Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    gap: 7px;
    grid-template-rows: auto auto minmax(150px, 1fr) auto auto;
    height: 100%;
    min-height: 0;
    overflow: hidden;
    padding: 8px;
    width: 100%;
  }
  .flh3-preview-panel * { box-sizing: border-box; }
  .flh3-preview-header,
  .flh3-preview-nav {
    align-items: center;
    display: flex;
    gap: 7px;
    min-width: 0;
  }
  .flh3-preview-title {
    font-size: 12px;
    font-weight: 700;
    margin-right: auto;
  }
  .flh3-preview-toggle,
  .flh3-preview-follow {
    align-items: center;
    color: var(--muted);
    cursor: pointer;
    display: flex;
    gap: 5px;
    white-space: nowrap;
  }
  .flh3-preview-toggle input,
  .flh3-preview-follow input { accent-color: var(--accent); margin: 0; }
  .flh3-preview-status {
    background: #30323a;
    border-radius: 999px;
    color: #d4d4d8;
    font-size: 9px;
    font-weight: 700;
    max-width: 120px;
    overflow: hidden;
    padding: 4px 7px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .flh3-preview-status[data-tone="working"] { background: #312e81; color: #c7d2fe; }
  .flh3-preview-status[data-tone="ready"] { background: #14532d; color: #bbf7d0; }
  .flh3-preview-status[data-tone="warning"] { background: #713f12; color: #fde68a; }
  .flh3-preview-status[data-tone="error"] { background: #7f1d1d; color: #fecaca; }
  .flh3-preview-progress {
    background: #2b2d34;
    border-radius: 999px;
    height: 7px;
    overflow: hidden;
  }
  .flh3-preview-progress-fill {
    background: linear-gradient(90deg, #7c3aed, #22c55e);
    height: 100%;
    transition: width 160ms ease;
    width: 0;
  }
  .flh3-preview-player {
    background: #050507;
    border: 1px solid var(--border);
    border-radius: 7px;
    min-height: 0;
    overflow: hidden;
    position: relative;
  }
  .flh3-preview-player video,
  .flh3-preview-sampling {
    display: block;
    height: 100%;
    object-fit: contain;
    width: 100%;
  }
  .flh3-preview-sampling {
    background: #050507;
    inset: 0;
    position: absolute;
  }
  .flh3-preview-player video[hidden],
  .flh3-preview-sampling[hidden] { display: none; }
  .flh3-preview-placeholder {
    align-items: center;
    color: #747986;
    display: flex;
    inset: 0;
    justify-content: center;
    padding: 24px;
    position: absolute;
    text-align: center;
  }
  .flh3-preview-meta {
    background: rgba(24, 25, 30, .88);
    border-radius: 5px;
    bottom: 7px;
    color: #d4d4d8;
    left: 7px;
    max-width: calc(100% - 14px);
    overflow: hidden;
    padding: 4px 6px;
    position: absolute;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .flh3-preview-error {
    background: rgba(127, 29, 29, .92);
    border-radius: 5px;
    color: #fecaca;
    left: 7px;
    padding: 5px 7px;
    position: absolute;
    right: 7px;
    top: 7px;
  }
  .flh3-preview-error[hidden] { display: none; }
  .flh3-preview-nav button,
  .flh3-preview-chunk {
    background: var(--comfy-input-bg, #24262d);
    border: 1px solid var(--border);
    border-radius: 5px;
    color: inherit;
    cursor: pointer;
    font: inherit;
  }
  .flh3-preview-nav button { height: 25px; min-width: 30px; }
  .flh3-preview-nav button:disabled { cursor: default; opacity: .35; }
  .flh3-preview-position {
    color: #d4d4d8;
    font-variant-numeric: tabular-nums;
    min-width: 72px;
    text-align: center;
  }
  .flh3-preview-follow { margin-left: auto; }
  .flh3-preview-chunks {
    display: flex;
    gap: 4px;
    min-height: 27px;
    overflow-x: auto;
    padding-bottom: 2px;
  }
  .flh3-preview-chunk {
    color: #737987;
    flex: 0 0 27px;
    height: 25px;
    padding: 0;
  }
  .flh3-preview-chunk[data-ready="true"] { color: #ddd6fe; border-color: #6d28d9; }
  .flh3-preview-chunk[data-error="true"] { color: #fecaca; border-color: #991b1b; }
  .flh3-preview-chunk[data-active="true"] { background: #6d28d9; color: white; }
`;

function nodeKey(value) {
  return value === null || value === undefined ? "" : String(value);
}

function consumePreviewEvent(event) {
  event.preventDefault?.();
  event.stopImmediatePropagation?.();
  event.stopPropagation?.();
}

function removeExternalPreviewWidgets(node) {
  let removed = false;
  for (let index = (node.widgets?.length || 0) - 1; index >= 0; index -= 1) {
    const widget = node.widgets[index];
    if (!EXTERNAL_PREVIEW_WIDGETS.has(widget.name)) continue;
    widget.onRemove?.();
    node.widgets.splice(index, 1);
    removed = true;
  }
  if (!removed) return;
  node.imgs = undefined;
  node.imageIndex = null;
  const computedHeight = node.computeSize?.([node.size[0], node.size[1]])?.[1];
  if (Number.isFinite(computedHeight)) {
    node.setSize([node.size[0], Math.max(node.min_size?.[1] || 0, computedHeight)]);
  }
  node.graph?.setDirtyCanvas(true, true);
}

function injectStyles() {
  if (document.getElementById("flh3-preview-styles")) return;
  const style = document.createElement("style");
  style.id = "flh3-preview-styles";
  style.textContent = STYLES;
  document.head.appendChild(style);
}

function hideWidget(widget) {
  widget.hidden = true;
  widget.computeSize = () => [0, 0];
  widget.computedHeight = 0;
  widget.type = "converted-widget";
  if (widget.element) widget.element.style.display = "none";
}

function migrateUpscaleStepWindow(node, info) {
  if (node.comfyClass !== "FL_MiniMaxH3BeatUpscaleKSampler") return;
  if (!info?.inputs?.some((input) => input.name === "denoise")) return;
  if (!Array.isArray(info.widgets_values) || info.widgets_values.length < 10) return;

  const values = info.widgets_values;
  const oldSteps = Number(values[5]);
  const oldDenoise = Number(values[9]);
  if (!Number.isFinite(oldSteps) || !Number.isFinite(oldDenoise)) return;

  const totalSteps = oldDenoise > 0 ? Math.trunc(oldSteps / oldDenoise) : oldSteps;
  const startAtStep = oldDenoise > 0 ? totalSteps - oldSteps : totalSteps;
  const migrated = [
    ...values.slice(0, 5),
    totalSteps,
    startAtStep,
    totalSteps,
    ...values.slice(6, 9),
    ...values.slice(10),
  ];
  info.widgets_values = migrated;
  for (let index = 0; index < migrated.length && index < node.widgets.length; index += 1) {
    node.widgets[index].value = migrated[index];
  }
}

function eventNode(detail) {
  if (detail && typeof detail === "object") {
    return detail.node ?? detail.node_id;
  }
  return detail;
}

function previewUrl(preview) {
  const params = new URLSearchParams({
    filename: preview.filename,
    subfolder: preview.subfolder || "",
    type: preview.type || "temp",
    timestamp: Date.now(),
  });
  return api.apiURL(`/view?${params.toString()}`);
}

class BeatSamplerPreview {
  constructor(node, inputWidget, container, config) {
    this.node = node;
    this.inputWidget = inputWidget;
    this.container = container;
    this.config = config;
    this.chunks = new Map();
    this.errors = new Set();
    this.selected = null;
    this.total = 0;
    this.samplingIndex = null;
    this.samplingFrames = [];
    this.samplingFrameIndex = 0;
    this.samplingTimer = null;
    this.samplingGeneration = 0;
    this.hasAnimatedSampling = false;
    this.awaitingVhsStart = false;

    injectStyles();
    this.element = document.createElement("div");
    this.element.className = "flh3-preview-panel";
    this.element.innerHTML = `
      <div class="flh3-preview-header">
        <span class="flh3-preview-title">${config.title}</span>
        <label class="flh3-preview-toggle" title="Decode and encode one silent preview per completed render">
          <input data-role="enabled" type="checkbox"> live
        </label>
        <span class="flh3-preview-status" data-role="status">preview off</span>
      </div>
      <div class="flh3-preview-progress"><div class="flh3-preview-progress-fill" data-role="progress"></div></div>
      <div class="flh3-preview-player">
        <video data-role="video" controls muted playsinline preload="metadata"></video>
        <canvas class="flh3-preview-sampling" data-role="sampling" aria-label="Live sampling preview" hidden></canvas>
        <div class="flh3-preview-placeholder" data-role="placeholder">Enable live preview, then queue the workflow.</div>
        <div class="flh3-preview-error" data-role="error" hidden></div>
        <div class="flh3-preview-meta" data-role="meta">No completed render yet.</div>
      </div>
      <div class="flh3-preview-nav">
        <button data-role="previous" type="button" title="Previous completed render">&#9664;</button>
        <span class="flh3-preview-position" data-role="position">chunk - / -</span>
        <button data-role="next" type="button" title="Next completed render">&#9654;</button>
        <label class="flh3-preview-follow"><input data-role="follow" type="checkbox" checked> follow latest</label>
      </div>
      <div class="flh3-preview-chunks" data-role="chunks"></div>
    `;
    container.appendChild(this.element);

    const role = (name) => this.element.querySelector(`[data-role="${name}"]`);
    this.enabled = role("enabled");
    this.status = role("status");
    this.progress = role("progress");
    this.video = role("video");
    this.samplingImage = role("sampling");
    this.placeholder = role("placeholder");
    this.error = role("error");
    this.meta = role("meta");
    this.previous = role("previous");
    this.next = role("next");
    this.position = role("position");
    this.follow = role("follow");
    this.chunkStrip = role("chunks");

    this.enabled.checked = Boolean(inputWidget.value);
    this.enabled.addEventListener("change", () => {
      inputWidget.value = this.enabled.checked;
      inputWidget.callback?.(inputWidget.value);
      app.graph?.setDirtyCanvas(true, true);
      if (!this.enabled.checked) this.setStatus("idle", "preview off");
      else if (!this.chunks.size) this.setStatus("idle", "ready to queue");
      this.updatePlaceholder();
    });
    this.follow.addEventListener("change", () => {
      if (this.follow.checked && this.chunks.size) {
        this.showChunk(Math.max(...this.chunks.keys()), true);
      }
    });
    this.previous.addEventListener("click", () => this.move(-1));
    this.next.addEventListener("click", () => this.move(1));
    this.video.addEventListener("loadeddata", () => {
      this.placeholder.style.display = "none";
    });
    this.video.addEventListener("error", () => {
      if (this.video.src) this.showError("The temporary preview file could not be loaded.");
    });
    this.updateNavigation();
  }

  setStatus(tone, label) {
    this.status.dataset.tone = tone;
    this.status.textContent = label;
  }

  setProgress(completed, total = this.total) {
    const percent = total > 0 ? Math.max(0, Math.min(100, completed / total * 100)) : 0;
    this.progress.style.width = `${percent}%`;
  }

  showError(message = "") {
    this.error.textContent = message;
    this.error.hidden = !message;
  }

  updatePlaceholder() {
    if (this.video.src || !this.samplingImage.hidden) return;
    this.placeholder.style.display = "flex";
    this.placeholder.textContent = this.enabled.checked
      ? "Waiting for the first completed render..."
      : "Queue to see sampling progress. Enable live for playable completed chunks.";
  }

  clearSamplingPreview() {
    this.samplingGeneration += 1;
    if (this.samplingTimer !== null) clearInterval(this.samplingTimer);
    this.samplingTimer = null;
    for (const frame of this.samplingFrames) frame?.close?.();
    this.samplingFrames = [];
    this.samplingFrameIndex = 0;
    const context = this.samplingImage.getContext("2d");
    context?.clearRect(0, 0, this.samplingImage.width, this.samplingImage.height);
    this.samplingImage.hidden = true;
  }

  beginSampling(index, awaitingVhsStart = false) {
    this.samplingIndex = index;
    this.hasAnimatedSampling = false;
    this.awaitingVhsStart = awaitingVhsStart;
    this.clearSamplingPreview();
    if (!this.follow.checked && this.selected !== null) return;
    this.video.pause();
    this.video.hidden = true;
    this.placeholder.textContent = "Waiting for the live sampling preview...";
    this.placeholder.style.display = "flex";
    this.meta.textContent = `Sampling chunk ${index + 1} / ${this.total || "-"}`;
  }

  beginVhsSampling(detail) {
    if (this.awaitingVhsStart) {
      this.awaitingVhsStart = false;
    } else {
      this.beginSampling((this.samplingIndex ?? -1) + 1);
    }

    const length = Math.max(0, Number(detail?.length) || 0);
    const rate = Math.max(1, Number(detail?.rate) || 1);
    this.samplingFrames = new Array(length);
    this.samplingFrameIndex = 0;
    this.samplingTimer = setInterval(() => {
      const frame = this.samplingFrames[this.samplingFrameIndex];
      if (!frame) return;
      this.drawSamplingFrame(frame);
      this.samplingFrameIndex = (this.samplingFrameIndex + 1) % this.samplingFrames.length;
    }, 1000 / rate);
  }

  drawSamplingFrame(frame) {
    if (this.samplingImage.width !== frame.width || this.samplingImage.height !== frame.height) {
      this.samplingImage.width = frame.width;
      this.samplingImage.height = frame.height;
    }
    this.samplingImage.getContext("2d")?.drawImage(frame, 0, 0);
    this.samplingImage.hidden = false;
    this.video.pause();
    this.video.hidden = true;
    this.placeholder.style.display = "none";
  }

  async showSamplingPreview(blob, animated, frameIndex = 0) {
    if (animated) this.hasAnimatedSampling = true;
    else if (this.hasAnimatedSampling) return;
    if (!this.follow.checked && this.selected !== null) return;

    const generation = this.samplingGeneration;
    let frame;
    try {
      frame = await window.createImageBitmap(blob);
    } catch {
      return;
    }
    if (generation !== this.samplingGeneration) {
      frame.close?.();
      return;
    }
    if (animated) {
      if (!Number.isSafeInteger(frameIndex) || frameIndex < 0 || frameIndex >= this.samplingFrames.length) {
        frame.close?.();
        return;
      }
      this.samplingFrames[frameIndex]?.close?.();
      this.samplingFrames[frameIndex] = frame;
      if (this.samplingImage.hidden) this.drawSamplingFrame(frame);
    } else {
      this.drawSamplingFrame(frame);
      frame.close?.();
    }
    this.meta.textContent = this.total > 0 && this.samplingIndex !== null
      ? `Live sampling preview | chunk ${this.samplingIndex + 1} / ${this.total}`
      : "Live sampling preview";
  }

  reset() {
    this.chunks.clear();
    this.errors.clear();
    this.selected = null;
    this.total = 0;
    this.samplingIndex = null;
    this.hasAnimatedSampling = false;
    this.awaitingVhsStart = false;
    this.clearSamplingPreview();
    this.video.pause();
    this.video.removeAttribute("src");
    this.video.hidden = false;
    this.video.load();
    this.meta.textContent = "No completed render yet.";
    this.position.textContent = "chunk - / -";
    this.showError();
    this.setProgress(0, 0);
    this.setStatus("idle", this.enabled.checked ? "starting" : "preview off");
    this.renderChunks();
    this.updatePlaceholder();
  }

  update(detail) {
    if (detail.stage !== this.config.stage) return;
    const index = Number(detail.index);
    this.total = Math.max(this.total, Number(detail.total) || 0);

    if (detail.status === "start") {
      this.reset();
      this.total = Number(detail.total) || 0;
      this.setStatus("working", "starting");
    } else if (detail.status === "sampling") {
      this.beginSampling(index, true);
      this.setStatus("working", `sampling ${index + 1}/${this.total}`);
      this.setProgress(index, this.total);
    } else if (detail.status === "previewing") {
      this.setStatus("working", `encoding ${index + 1}/${this.total}`);
      this.setProgress(index, this.total);
    } else if (detail.status === "chunk_ready" && detail.preview) {
      this.chunks.set(index, detail);
      this.errors.delete(index);
      this.showError();
      this.setProgress(index + 1, this.total);
      this.setStatus("ready", `${this.chunks.size}/${this.total} ready`);
      if (this.follow.checked || this.selected === null) this.showChunk(index, true);
    } else if (detail.status === "preview_error") {
      this.errors.add(index);
      this.showError(`Chunk ${index + 1}: ${detail.error || "preview failed"}`);
      this.setStatus("error", "preview error");
    } else if (detail.status === "unavailable") {
      this.showError(detail.error || "Live preview is unavailable.");
      this.setStatus("warning", "unavailable");
    } else if (detail.status === "done") {
      this.setProgress(this.total, this.total);
      this.setStatus("ready", "complete");
    }
    this.renderChunks();
    this.updateNavigation();
  }

  showChunk(index, autoplay = false) {
    const detail = this.chunks.get(index);
    if (!detail?.preview) return;
    this.selected = index;
    const preview = detail.preview;
    this.clearSamplingPreview();
    this.hasAnimatedSampling = false;
    this.video.hidden = false;
    this.video.src = previewUrl(preview);
    this.video.load();
    this.placeholder.textContent = "Loading preview...";
    this.placeholder.style.display = "flex";

    const start = Number(detail.start_frame);
    const end = Number(detail.end_frame);
    const frames = Number.isFinite(start) && Number.isFinite(end)
      ? `frames ${start}-${Math.max(start, end - 1)}`
      : `${preview.frame_count} frames`;
    const seed = detail.seed === null || detail.seed === undefined ? "" : ` | seed ${detail.seed}`;
    this.meta.textContent = `${frames} | ${Number(preview.duration).toFixed(2)}s | ${preview.width}x${preview.height}${seed}`;
    this.position.textContent = `chunk ${index + 1} / ${this.total || "-"}`;
    this.renderChunks();
    this.updateNavigation();
    if (autoplay) this.video.play().catch(() => {});
  }

  move(direction) {
    const ready = [...this.chunks.keys()].sort((a, b) => a - b);
    if (!ready.length) return;
    const current = ready.indexOf(this.selected);
    const target = ready[Math.max(0, Math.min(ready.length - 1, current + direction))];
    if (target === undefined) return;
    this.follow.checked = false;
    this.showChunk(target, true);
  }

  updateNavigation() {
    const ready = [...this.chunks.keys()].sort((a, b) => a - b);
    const current = ready.indexOf(this.selected);
    this.previous.disabled = current <= 0;
    this.next.disabled = current < 0 || current >= ready.length - 1;
  }

  renderChunks() {
    this.chunkStrip.replaceChildren();
    for (let index = 0; index < this.total; index += 1) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "flh3-preview-chunk";
      button.textContent = String(index + 1);
      button.dataset.ready = String(this.chunks.has(index));
      button.dataset.error = String(this.errors.has(index));
      button.dataset.active = String(index === this.selected);
      button.disabled = !this.chunks.has(index);
      button.title = this.chunks.has(index) ? `Play render ${index + 1}` : `Render ${index + 1} is not ready`;
      button.addEventListener("click", () => {
        this.follow.checked = false;
        this.showChunk(index, true);
      });
      this.chunkStrip.appendChild(button);
    }
  }

  markExecutionError(message) {
    this.showError(message || "Sampling stopped with an error.");
    this.setStatus("error", "execution error");
  }

  markCached() {
    if (this.enabled.checked && !this.chunks.size) {
      this.setStatus("warning", "cached");
      this.showError("This node was cached, so no new live previews were generated.");
    }
  }

  configure() {
    hideWidget(this.inputWidget);
    this.enabled.checked = Boolean(this.inputWidget.value);
    this.updatePlaceholder();
  }

  dispose() {
    this.clearSamplingPreview();
    this.video.pause();
    this.video.removeAttribute("src");
    this.video.load();
    this.element.remove();
  }
}

function registerInstance(node, panel) {
  for (const [key, value] of instances) {
    if (value === panel && key !== nodeKey(node.id)) instances.delete(key);
  }
  instances.set(nodeKey(node.id), panel);
}

function previewPanel(detail) {
  const key = nodeKey(detail?.displayNodeId ?? detail?.nodeId ?? detail?.id);
  return instances.get(key) ?? instances.get(activeSamplerKey);
}

let activeSamplerKey = "";
const claimedPreviewBlobs = new WeakSet();

app.registerExtension({
  name: "ComfyUI.FL_MiniMaxH3.BeatSamplerPreview",
  nodeCreated(node) {
    const config = NODE_CONFIG[node.comfyClass || node.constructor?.comfyClass];
    if (!config) return;
    const inputWidget = node.widgets?.find((widget) => widget.name === "live_preview");
    if (!inputWidget) return;
    hideWidget(inputWidget);

    const container = document.createElement("div");
    container.style.height = "100%";
    container.style.minHeight = `${MIN_PANEL_HEIGHT}px`;
    container.style.overflow = "hidden";
    container.style.width = "100%";
    addCanvasNavigation(container, app.canvas);
    const domWidget = node.addDOMWidget(
      "fl_h3_live_preview",
      "fl-h3-live-preview",
      container,
      {
        getMinHeight: () => MIN_PANEL_HEIGHT,
        hideOnZoom: false,
        serialize: false,
      },
    );

    node.min_size = [
      Math.max(node.min_size?.[0] || 0, 420),
      Math.max(node.min_size?.[1] || 0, config.minHeight),
    ];
    node.setSize([
      Math.max(node.size[0], 420),
      Math.max(node.size[1], config.minHeight),
    ]);

    const panel = new BeatSamplerPreview(node, inputWidget, container, config);
    registerInstance(node, panel);
    removeExternalPreviewWidgets(node);

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      migrateUpscaleStepWindow(this, args[0]);
      const result = originalOnConfigure?.apply(this, args);
      panel.configure();
      registerInstance(this, panel);
      removeExternalPreviewWidgets(this);
      return result;
    };
    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
      instances.delete(nodeKey(this.id));
      panel.dispose();
      return originalOnRemoved?.apply(this, arguments);
    };
    domWidget.onRemove = () => {
      instances.delete(nodeKey(node.id));
      panel.dispose();
    };
  },
});

api.addEventListener("executing", (event) => {
  const key = nodeKey(eventNode(event.detail));
  const panel = instances.get(key);
  activeSamplerKey = panel ? key : "";
  if (!panel) return;
  removeExternalPreviewWidgets(panel.node);
  panel.reset();
});

api.addEventListener(VHS_PREVIEW_EVENT, (event) => {
  const panel = previewPanel(event.detail);
  if (!panel) return;
  panel.beginVhsSampling(event.detail);
  removeExternalPreviewWidgets(panel.node);
  consumePreviewEvent(event);
}, true);

api.addEventListener("b_preview_with_metadata", (event) => {
  const detail = event.detail;
  const panel = previewPanel(detail);
  if (!panel || !detail?.blob || typeof detail.blob !== "object") return;
  panel.showSamplingPreview(
    detail.blob,
    detail.parentNodeId === VHS_PREVIEW_MARKER,
    Number(detail.realNodeId),
  );
  removeExternalPreviewWidgets(panel.node);
  claimedPreviewBlobs.add(detail.blob);
  consumePreviewEvent(event);
}, true);

api.addEventListener("b_preview", (event) => {
  if (event.detail && claimedPreviewBlobs.delete(event.detail)) {
    consumePreviewEvent(event);
    return;
  }
  const panel = instances.get(activeSamplerKey);
  if (!panel || !event.detail || typeof event.detail !== "object") return;
  panel.showSamplingPreview(event.detail, false);
  removeExternalPreviewWidgets(panel.node);
  consumePreviewEvent(event);
}, true);

api.addEventListener(EVENT_NAME, (event) => {
  const detail = event.detail;
  if (!detail) return;
  instances.get(nodeKey(detail.node))?.update(detail);
});

api.addEventListener("execution_error", (event) => {
  const detail = event.detail || {};
  const panel = instances.get(nodeKey(eventNode(detail)));
  panel?.markExecutionError(detail.exception_message || detail.exception_type);
});

api.addEventListener("execution_cached", (event) => {
  const detail = event.detail;
  const nodes = Array.isArray(detail?.nodes) ? detail.nodes : Array.isArray(detail) ? detail : [];
  for (const nodeId of nodes) instances.get(nodeKey(nodeId))?.markCached();
});
