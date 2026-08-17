import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import {
  closeTemporalReshotForNode,
  openTemporalReshotEditor,
} from "./FL_MiniMaxH3TemporalReshotModal.js";
import {
  compactReshotStatus,
  migrateReshotView,
  previewReference,
  readReshotSettings,
} from "./FL_MiniMaxH3TemporalReshotState.js";
import { injectTemporalReshotStyles } from "./FL_MiniMaxH3TemporalReshotStyles.js";


const NODE_ID = "FL_MiniMaxH3TemporalReshotPlanner";
const COMPACT_WIDTH = 410;
const COMPACT_PANEL_HEIGHT = 270;

function findWidget(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name) || null;
}

function hideWidget(widget) {
  if (!widget) return;
  if (!widget.origType) widget.origType = widget.type;
  if (!widget.origComputeSize) widget.origComputeSize = widget.computeSize;
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
  widget.computedHeight = 0;
  widget.type = "converted-widget";
  if (widget.element) widget.element.style.display = "none";
}

function compactNode(node, force = false) {
  node.min_size = [360, 270];
  requestAnimationFrame(() => {
    const computed = node.computeSize?.() || node.size;
    const width = force ? COMPACT_WIDTH : Math.max(360, Math.min(node.size[0], 500));
    const height = Math.max(270, computed[1]);
    if (force || node.size[0] > 500 || node.size[1] > height + 35) node.setSize([width, height]);
  });
}

function setWidgetValue(widget, value) {
  if (!widget) return;
  widget.value = value;
  widget.callback?.call(widget, value);
}

class TemporalReshotController {
  constructor(node, widgets, container) {
    this.node = node;
    this.widgets = widgets;
    this.container = container;
    this.listeners = new Set();
    this.probeId = 0;
    this.disposed = false;
    this.error = "";
    this.path = String(widgets.video.value || "");
    this.prompt = String(widgets.prompt.value || "");
    this.refImageSize = String(widgets.refSize.value || "match");
    this.sourceInfo = null;
    this.settings = readReshotSettings(widgets.settings.value);
    this.node.properties ||= {};
    const cached = this.node.properties.flH3TemporalReshotSource;
    if (cached?.filename === this.path) {
      this.sourceInfo = { ...cached };
      delete this.sourceInfo.filename;
    }
    this.view = migrateReshotView(
      this.node.properties.flH3TemporalReshot,
      this.sourceInfo?.frame_count,
    );
    this.saveView(false);
    this.buildCompact();
    this.bindCompact();
    this.syncCompact();
    if (this.path) this.probeSource(this.path, false, false);
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notify() {
    this.syncCompact();
    for (const listener of this.listeners) listener(this);
    this.node.setDirtyCanvas?.(true, true);
  }

  graphChanged() {
    this.node.graph?.change?.();
    this.node.setDirtyCanvas?.(true, true);
  }

  buildCompact() {
    injectTemporalReshotStyles();
    this.container.innerHTML = `
      <div class="flh3r-compact">
        <div class="flh3r-compact-preview">
          <video data-role="video" muted playsinline preload="metadata"></video>
          <div class="flh3r-compact-empty" data-role="empty">Open the editor to choose a source and define the replacement interval.</div>
          <button class="flh3r-compact-play" data-action="play" type="button" title="Preview the selected range">▶</button>
        </div>
        <div class="flh3r-compact-status" data-role="status">No source selected</div>
      </div>
    `;
    this.compactVideo = this.container.querySelector('[data-role="video"]');
    this.compactEmpty = this.container.querySelector('[data-role="empty"]');
    this.compactPlay = this.container.querySelector('[data-action="play"]');
    this.compactStatus = this.container.querySelector('[data-role="status"]');
  }

  bindCompact() {
    this.compactPlay.addEventListener("click", () => {
      if (!this.compactVideo.src) return;
      if (this.compactVideo.paused) {
        this.compactVideo.currentTime = this.settings.start_frame / 24;
        this.compactVideo.play().catch(() => {});
      } else {
        this.compactVideo.pause();
      }
    });
    this.compactVideo.addEventListener("play", () => { this.compactPlay.textContent = "❚❚"; });
    this.compactVideo.addEventListener("pause", () => { this.compactPlay.textContent = "▶"; });
    this.compactVideo.addEventListener("timeupdate", () => {
      const start = this.settings.start_frame / 24;
      const end = (this.settings.start_frame + this.settings.frame_count) / 24;
      if (this.compactVideo.currentTime >= end || this.compactVideo.currentTime < start) {
        this.compactVideo.currentTime = start;
      }
    });
  }

  previewUrl() {
    if (!this.path) return "";
    const reference = previewReference(this.path);
    const params = new URLSearchParams({
      filename: reference.filename,
      subfolder: reference.subfolder,
      type: "input",
    });
    return api.apiURL(`/view?${params.toString()}`);
  }

  syncCompact() {
    const url = this.previewUrl();
    const current = this.compactVideo.dataset.source || "";
    if (url !== current) {
      this.compactVideo.pause();
      this.compactVideo.dataset.source = url;
      if (url) {
        this.compactVideo.src = url;
        this.compactVideo.load();
      } else {
        this.compactVideo.removeAttribute("src");
        this.compactVideo.load();
      }
    }
    this.compactEmpty.hidden = Boolean(this.path);
    this.compactPlay.disabled = !this.path;
    const status = compactReshotStatus(this.path, this.sourceInfo, this.settings, this.error);
    this.compactStatus.dataset.tone = status.tone;
    this.compactStatus.textContent = status.text;
    this.compactStatus.title = status.text;
  }

  async setSource(path, forceInitialize = false) {
    path = String(path || "");
    const changed = path !== this.path;
    this.path = path;
    this.error = "";
    const values = this.widgets.video.options?.values;
    if (path && Array.isArray(values) && !values.includes(path)) values.push(path);
    setWidgetValue(this.widgets.video, path);
    this.graphChanged();
    if (!path) {
      this.probeId += 1;
      this.sourceInfo = null;
      delete this.node.properties.flH3TemporalReshotSource;
      this.notify();
      return;
    }
    this.notify();
    await this.probeSource(path, forceInitialize || changed, true);
  }

  async probeSource(path, initialize, markGraph) {
    const probeId = ++this.probeId;
    this.error = "";
    try {
      const params = new URLSearchParams({ filename: path });
      const response = await api.fetchApi(`/fl/minimax-h3/temporal-reshot/info?${params.toString()}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `Video probe failed (${response.status}).`);
      if (this.disposed || probeId !== this.probeId || path !== this.path) return;
      this.sourceInfo = payload;
      this.node.properties.flH3TemporalReshotSource = { ...payload, filename: path };
      const initialized = this.node.properties.flH3TemporalReshotInitializedSource;
      if (initialize || initialized !== path) {
        const count = Math.min(72, payload.frame_count);
        this.settings = readReshotSettings(JSON.stringify({
          ...this.settings,
          start_frame: Math.max(0, Math.floor((payload.frame_count - count) / 2)),
          frame_count: count,
        }), payload.frame_count);
        this.node.properties.flH3TemporalReshotInitializedSource = path;
        setWidgetValue(this.widgets.settings, JSON.stringify(this.settings));
        this.view = migrateReshotView({ ...this.view, viewStart: 0, viewFrames: payload.frame_count }, payload.frame_count);
        this.saveView(false);
      } else {
        this.settings = readReshotSettings(this.widgets.settings.value, payload.frame_count);
        setWidgetValue(this.widgets.settings, JSON.stringify(this.settings));
      }
      this.notify();
      if (markGraph) this.graphChanged();
    } catch (error) {
      if (this.disposed || probeId !== this.probeId) return;
      this.sourceInfo = null;
      this.error = error.message || "Could not inspect the source video.";
      this.notify();
      if (markGraph) this.graphChanged();
    }
  }

  setSettings(settings) {
    this.settings = readReshotSettings(
      JSON.stringify(settings),
      this.sourceInfo?.frame_count,
    );
    setWidgetValue(this.widgets.settings, JSON.stringify(this.settings));
    this.notify();
    this.graphChanged();
  }

  setPrompt(prompt) {
    this.prompt = String(prompt || "");
    setWidgetValue(this.widgets.prompt, this.prompt);
    this.notify();
    this.graphChanged();
  }

  setRefImageSize(value) {
    this.refImageSize = value === "max" ? "max" : "match";
    setWidgetValue(this.widgets.refSize, this.refImageSize);
    this.notify();
    this.graphChanged();
  }

  updateView(values, markGraph = true) {
    this.view = migrateReshotView(
      { ...this.view, ...values },
      this.sourceInfo?.frame_count,
    );
    this.saveView(markGraph);
    this.notify();
  }

  saveView(markGraph = true) {
    this.node.properties.flH3TemporalReshot = { ...this.view };
    if (markGraph) this.graphChanged();
  }

  audioVaeConnected() {
    return this.node.inputs?.some((input) => input.name === "audio_vae" && input.link != null) || false;
  }

  referenceCount() {
    return this.node.inputs?.filter((input) =>
      (input.name === "ref_images" || input.name.startsWith("ref_images.")) && input.link != null
    ).length || 0;
  }

  configure() {
    hideWidget(this.widgets.video);
    hideWidget(this.widgets.prompt);
    hideWidget(this.widgets.settings);
    hideWidget(this.widgets.refSize);
    this.path = String(this.widgets.video.value || "");
    this.prompt = String(this.widgets.prompt.value || "");
    this.refImageSize = String(this.widgets.refSize.value || "match");
    const cached = this.node.properties.flH3TemporalReshotSource;
    this.sourceInfo = cached?.filename === this.path ? { ...cached } : null;
    if (this.sourceInfo) delete this.sourceInfo.filename;
    this.settings = readReshotSettings(this.widgets.settings.value, this.sourceInfo?.frame_count);
    this.view = migrateReshotView(this.node.properties.flH3TemporalReshot, this.sourceInfo?.frame_count);
    this.error = "";
    this.notify();
    if (this.path) this.probeSource(this.path, false, false);
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.probeId += 1;
    this.listeners.clear();
    this.compactVideo.pause();
    this.compactVideo.removeAttribute("src");
    this.compactVideo.load();
    this.container.replaceChildren();
  }
}

app.registerExtension({
  name: "ComfyUI.FL_MiniMaxH3.TemporalReshot",
  nodeCreated(node) {
    if ((node.comfyClass || node.constructor?.comfyClass) !== NODE_ID) return;
    const widgets = {
      video: findWidget(node, "video"),
      prompt: findWidget(node, "prompt"),
      settings: findWidget(node, "reshot_settings"),
      refSize: findWidget(node, "ref_image_size"),
    };
    if (!widgets.video || !widgets.prompt || !widgets.settings || !widgets.refSize) return;
    for (const widget of Object.values(widgets)) hideWidget(widget);

    const openWidget = node.addWidget("button", "Open Temporal Reshot Editor", null, () => {
      openTemporalReshotEditor(controller);
    }, { serialize: false });
    openWidget.serialize = false;

    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.height = "100%";
    container.style.minHeight = `${COMPACT_PANEL_HEIGHT}px`;
    container.style.overflow = "hidden";
    const domWidget = node.addDOMWidget("fl_h3_temporal_reshot_compact", "fl-h3-temporal-reshot-compact", container, {
      getMinHeight: () => COMPACT_PANEL_HEIGHT,
      hideOnZoom: false,
      serialize: false,
    });
    const controller = new TemporalReshotController(node, widgets, container);
    compactNode(node, true);

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const result = originalOnConfigure?.apply(this, args);
      controller.configure();
      requestAnimationFrame(() => compactNode(this, false));
      return result;
    };
    const originalOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function () {
      const result = originalOnConnectionsChange?.apply(this, arguments);
      controller.notify();
      return result;
    };
    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
      closeTemporalReshotForNode(this);
      controller.dispose();
      return originalOnRemoved?.apply(this, arguments);
    };
    domWidget.onRemove = () => controller.dispose();
  },
});
