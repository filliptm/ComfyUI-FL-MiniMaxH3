import { app } from "../../../scripts/app.js";
import { addCanvasNavigation } from "./canvas_navigation.js";
import { normalizeTransitionSettings } from "./FL_MiniMaxH3TransitionMath.js";


const NODE_ID = "FL_MiniMaxH3TransitionPrep";
const PANEL_HEIGHT = 570;
const MAX_PIXELS = 768 * 1344;

function findWidget(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name) || null;
}

function hideWidget(widget) {
  if (!widget) return;
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
  widget.computedHeight = 0;
  widget.type = "converted-widget";
  if (widget.element) widget.element.style.display = "none";
}

function setWidgetValue(widget, value) {
  if (!widget || widget.value === value) return;
  widget.value = value;
  widget.callback?.call(widget, value);
}

function injectStyles() {
  if (document.getElementById("flh3-transition-styles")) return;
  const style = document.createElement("style");
  style.id = "flh3-transition-styles";
  style.textContent = `
    .flh3-transition {
      --accent-a: #2563eb;
      --accent-generate: #7c3aed;
      --accent-b: #059669;
      --border: var(--border-color, #383b45);
      background: var(--comfy-menu-bg, #18191e);
      border: 1px solid var(--border);
      border-radius: 9px;
      box-sizing: border-box;
      color: var(--input-text, #f4f4f5);
      display: flex;
      flex-direction: column;
      font: 11px/1.35 Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      gap: 9px;
      height: 100%;
      overflow: hidden;
      padding: 10px;
      width: 100%;
    }
    .flh3-transition * { box-sizing: border-box; }
    .flh3-transition-header { align-items: center; display: flex; gap: 8px; }
    .flh3-transition-title { font-size: 13px; font-weight: 750; }
    .flh3-transition-badge {
      background: #312e81;
      border-radius: 999px;
      color: #ddd6fe;
      font-size: 9px;
      font-weight: 700;
      margin-left: auto;
      padding: 4px 7px;
    }
    .flh3-transition-timeline {
      border: 1px solid var(--border);
      border-radius: 7px;
      display: flex;
      height: 42px;
      overflow: hidden;
    }
    .flh3-transition-segment {
      align-items: center;
      display: flex;
      font-size: 10px;
      font-weight: 700;
      justify-content: center;
      min-width: 42px;
      overflow: hidden;
      text-align: center;
      white-space: nowrap;
    }
    .flh3-transition-segment[data-kind="a"] { background: var(--accent-a); }
    .flh3-transition-segment[data-kind="generate"] { background: var(--accent-generate); }
    .flh3-transition-segment[data-kind="b"] { background: var(--accent-b); }
    .flh3-transition-summary { color: var(--descrip-text, #a1a1aa); text-align: center; }
    .flh3-transition-summary strong { color: #f4f4f5; }
    .flh3-transition-grid { display: grid; gap: 7px; grid-template-columns: repeat(3, 1fr); }
    .flh3-transition label { color: var(--descrip-text, #a1a1aa); display: grid; gap: 3px; }
    .flh3-transition input,
    .flh3-transition select,
    .flh3-transition textarea {
      background: var(--comfy-input-bg, #24262d);
      border: 1px solid var(--border);
      border-radius: 5px;
      color: inherit;
      font: inherit;
      min-width: 0;
      padding: 6px 7px;
      width: 100%;
    }
    .flh3-transition textarea { min-height: 112px; resize: none; }
    .flh3-transition-row { display: grid; gap: 7px; grid-template-columns: 1fr 150px; }
    .flh3-transition-warning { color: #fbbf24; min-height: 15px; }
    .flh3-transition-help { color: var(--descrip-text, #8e94a3); font-size: 10px; }
  `;
  document.head.appendChild(style);
}

class TransitionPanel {
  constructor(node, widgets, container) {
    this.node = node;
    this.widgets = widgets;
    this.container = container;
    this.build();
    this.bind();
    this.configure();
  }

  build() {
    this.container.innerHTML = `
      <div class="flh3-transition">
        <div class="flh3-transition-header">
          <span class="flh3-transition-title">Automatic masked transition</span>
          <span class="flh3-transition-badge">no frame indices</span>
        </div>
        <div class="flh3-transition-timeline">
          <div class="flh3-transition-segment" data-role="segment-a" data-kind="a"></div>
          <div class="flh3-transition-segment" data-role="segment-generate" data-kind="generate"></div>
          <div class="flh3-transition-segment" data-role="segment-b" data-kind="b"></div>
        </div>
        <div class="flh3-transition-summary" data-role="summary"></div>
        <div class="flh3-transition-grid">
          <label>Bridge frames<input data-role="length" type="number" min="22" step="17"></label>
          <label>References<select data-role="reference"></select></label>
          <label>Control mode<select data-role="mode"><option value="source seam repair">source seam repair</option><option value="empty bridge">empty bridge</option></select></label>
          <label>Width<input data-role="width" type="number" min="32" step="32"></label>
          <label>Height<input data-role="height" type="number" min="32" step="32"></label>
          <label>Mask feather<input data-role="feather" type="number" min="0" max="64" step="1"></label>
        </div>
        <div class="flh3-transition-warning" data-role="warning"></div>
        <label>Continuous action and camera direction
          <textarea data-role="description"></textarea>
        </label>
        <div class="flh3-transition-row">
          <label>Overall soundscape<input data-role="soundscape" type="text"></label>
          <label>Canvas fit<select data-role="crop"><option value="center">center crop</option><option value="stretch">stretch</option></select></label>
        </div>
        <label>Non-diegetic music<input data-role="music" type="text"></label>
        <div class="flh3-transition-help" data-role="help"></div>
      </div>
    `;
    const role = (name) => this.container.querySelector(`[data-role="${name}"]`);
    this.length = role("length");
    this.reference = role("reference");
    this.width = role("width");
    this.height = role("height");
    this.mode = role("mode");
    this.feather = role("feather");
    this.description = role("description");
    this.soundscape = role("soundscape");
    this.music = role("music");
    this.crop = role("crop");
    this.summary = role("summary");
    this.warning = role("warning");
    this.help = role("help");
    this.segmentA = role("segment-a");
    this.segmentGenerate = role("segment-generate");
    this.segmentB = role("segment-b");
  }

  bind() {
    const updateText = (input, widget) => input.addEventListener("input", () => {
      setWidgetValue(widget, input.value);
      this.changed();
    });
    updateText(this.description, this.widgets.description);
    updateText(this.soundscape, this.widgets.soundscape);
    updateText(this.music, this.widgets.music);
    this.length.addEventListener("change", () => {
      const settings = normalizeTransitionSettings(this.length.value, this.reference.value);
      setWidgetValue(this.widgets.length, settings.length);
      setWidgetValue(this.widgets.reference, settings.referenceFrames);
      this.syncTimeline(settings);
      this.changed();
    });
    this.reference.addEventListener("change", () => {
      setWidgetValue(this.widgets.reference, Number(this.reference.value));
      this.syncTimeline();
      this.changed();
    });
    for (const [input, widget] of [[this.width, this.widgets.width], [this.height, this.widgets.height]]) {
      input.addEventListener("change", () => {
        const value = Math.max(32, Math.round(Number(input.value) / 32) * 32);
        setWidgetValue(widget, value);
        input.value = String(value);
        this.syncTimeline();
        this.changed();
      });
    }
    this.crop.addEventListener("change", () => {
      setWidgetValue(this.widgets.crop, this.crop.value);
      this.changed();
    });
    this.mode.addEventListener("change", () => {
      setWidgetValue(this.widgets.mode, this.mode.value);
      this.syncTimeline();
      this.changed();
    });
    this.feather.addEventListener("change", () => {
      const value = Math.max(0, Math.round(Number(this.feather.value) || 0));
      setWidgetValue(this.widgets.feather, value);
      this.feather.value = String(value);
      this.syncTimeline();
      this.changed();
    });
  }

  changed() {
    this.node.graph?.change?.();
    this.node.setDirtyCanvas?.(true, true);
  }

  syncTimeline(settings = null) {
    settings ||= normalizeTransitionSettings(this.widgets.length.value, this.widgets.reference.value);
    this.length.value = String(settings.length);
    const previous = Number(this.reference.value);
    this.reference.replaceChildren(...settings.options.map((value) => {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = `${value} frames`;
      return option;
    }));
    this.reference.value = String(settings.options.includes(previous) ? previous : settings.referenceFrames);
    const reference = Number(this.reference.value);
    if (reference !== settings.referenceFrames) {
      settings = normalizeTransitionSettings(settings.length, reference);
    }
    setWidgetValue(this.widgets.length, settings.length);
    setWidgetValue(this.widgets.reference, settings.referenceFrames);
    const feather = Math.min(
      settings.maxMaskFeatherTokens,
      Math.max(0, Math.round(Number(this.feather.value) || 0)),
    );
    this.feather.max = String(settings.maxMaskFeatherTokens);
    this.feather.value = String(feather);
    setWidgetValue(this.widgets.feather, feather);
    const referencePercent = settings.referenceFrames / settings.length * 100;
    const generatedPercent = settings.generatedFrames / settings.length * 100;
    this.segmentA.style.width = `${referencePercent}%`;
    this.segmentGenerate.style.width = `${generatedPercent}%`;
    this.segmentB.style.width = `${referencePercent}%`;
    const seamRepair = this.mode.value === "source seam repair";
    this.segmentA.textContent = `A context ${settings.referenceFrames}f`;
    this.segmentGenerate.textContent = seamRepair
      ? `Repair ${settings.sourceAEditFrames}A + ${settings.sourceBEditFrames}B`
      : `Generate ${settings.generatedFrames}f`;
    this.segmentB.textContent = `B context ${settings.referenceFrames}f`;
    this.summary.innerHTML = seamRepair
      ? `<strong>${settings.length}-frame repair window</strong> at 24 fps · edits ${settings.generatedFrames} existing seam frames · output duration unchanged`
      : `<strong>${settings.length} frames</strong> at 24 fps · ${settings.duration.toFixed(2)} seconds · output adds <strong>${settings.generatedFrames} frames</strong>`;
    this.feather.disabled = !seamRepair;
    this.help.textContent = seamRepair
      ? `Real tail/head frames fill the repair region. The mask ramps over ${this.feather.value} H3 tokens at each edge; use reduced sampler denoise.`
      : "The blue and green ranges are encoded source references and stay un-noised. Only the purple middle is generated.";
    const pixels = Number(this.width.value) * Number(this.height.value);
    this.warning.textContent = pixels > MAX_PIXELS
      ? `Canvas exceeds H3's ${MAX_PIXELS.toLocaleString()}-pixel limit.`
      : "";
  }

  configure() {
    for (const widget of Object.values(this.widgets)) hideWidget(widget);
    this.width.value = String(this.widgets.width.value);
    this.height.value = String(this.widgets.height.value);
    this.description.value = String(this.widgets.description.value || "");
    this.soundscape.value = String(this.widgets.soundscape.value || "");
    this.music.value = String(this.widgets.music.value || "N/A");
    this.crop.value = String(this.widgets.crop.value || "center");
    this.mode.value = String(this.widgets.mode.value || "empty bridge");
    this.feather.value = String(this.widgets.feather.value ?? 2);
    this.syncTimeline();
  }

  dispose() {
    this.container.replaceChildren();
  }
}

app.registerExtension({
  name: "ComfyUI.FL_MiniMaxH3.Transition",
  nodeCreated(node) {
    if ((node.comfyClass || node.constructor?.comfyClass) !== NODE_ID) return;
    const widgets = {
      description: findWidget(node, "transition_description"),
      soundscape: findWidget(node, "overall_soundscape"),
      music: findWidget(node, "non_diegetic_music"),
      width: findWidget(node, "width"),
      height: findWidget(node, "height"),
      length: findWidget(node, "length"),
      reference: findWidget(node, "reference_frames"),
      crop: findWidget(node, "crop_mode"),
      empty: findWidget(node, "empty_frame_level"),
      mode: findWidget(node, "control_mode"),
      feather: findWidget(node, "mask_feather_tokens"),
    };
    if (Object.values(widgets).some((widget) => !widget)) return;
    injectStyles();
    for (const widget of Object.values(widgets)) hideWidget(widget);

    const container = document.createElement("div");
    container.style.height = "100%";
    container.style.minHeight = `${PANEL_HEIGHT}px`;
    container.style.overflow = "hidden";
    container.style.width = "100%";
    addCanvasNavigation(container, app.canvas);
    const domWidget = node.addDOMWidget("fl_h3_transition", "fl-h3-transition", container, {
      getMinHeight: () => PANEL_HEIGHT,
      hideOnZoom: false,
      serialize: false,
    });
    const panel = new TransitionPanel(node, widgets, container);
    node.min_size = [520, 650];
    node.setSize([Math.max(node.size[0], 520), Math.max(node.size[1], 650)]);

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const result = originalOnConfigure?.apply(this, args);
      panel.configure();
      return result;
    };
    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
      panel.dispose();
      return originalOnRemoved?.apply(this, arguments);
    };
    domWidget.onRemove = () => panel.dispose();
  },
});
