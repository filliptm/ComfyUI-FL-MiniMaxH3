import {
  buildReshotWindow,
  clamp,
  frameAtX,
  moveSelection,
  normalizeReshotSettings,
  resizeSelection,
  xForFrame,
} from "./FL_MiniMaxH3TemporalReshotMath.js";
import {
  compactReshotStatus,
  filenameFromPath,
  formatTimecode,
  reshotWarnings,
} from "./FL_MiniMaxH3TemporalReshotState.js";


function isTypingTarget(target) {
  return target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    target instanceof HTMLButtonElement ||
    target?.isContentEditable;
}

export class TemporalReshotEditor {
  constructor({ controller, container, inspector }) {
    this.controller = controller;
    this.container = container;
    this.inspectorHost = inspector;
    this.drag = null;
    this.loopSelection = false;
    this.sourceUrl = "";
    this.drawRequest = null;
    this.build();
    this.bind();
    this.unsubscribe = controller.subscribe(() => this.sync());
    this.sync();
  }

  build() {
    this.root = document.createElement("div");
    this.root.className = "flh3r-editor";
    this.root.tabIndex = 0;
    this.root.innerHTML = `
      <div class="flh3r-transport">
        <button class="flh3r-button" data-action="play" type="button">Play</button>
        <button class="flh3r-button" data-action="stop" type="button">Stop</button>
        <button class="flh3r-button" data-action="selection" type="button">Loop selection</button>
        <button class="flh3r-button" data-action="mute" type="button">Unmute</button>
        <label class="flh3r-volume">Volume <input data-role="volume" type="range" min="0" max="100" step="1"><span data-role="volume-value">80%</span></label>
        <span class="flh3r-time" data-role="time">00:00.000 / 00:00.000</span>
        <span class="flh3r-source-label" data-role="source">No source selected</span>
        <span class="flh3r-spacer"></span>
        <span class="flh3r-status" data-role="status">Choose a source video</span>
      </div>
      <div class="flh3r-video-stage">
        <video data-role="video" playsinline preload="metadata"></video>
        <div class="flh3r-empty" data-role="video-empty">Choose a source video from the library.</div>
        <div class="flh3r-video-overlay" data-role="video-overlay">The exact purple range will be replaced.</div>
      </div>
      <div class="flh3r-toolbar">
        <span class="flh3r-legend">
          <span><i class="source"></i>source</span>
          <span><i class="context"></i>context</span>
          <span><i class="selection"></i>replacement</span>
          <span><i class="tokens"></i>expanded H3 tokens</span>
          <span><i class="padding"></i>padding</span>
        </span>
        <span class="flh3r-spacer"></span>
        <button class="flh3r-button" data-action="zoom-out" type="button">−</button>
        <button class="flh3r-button" data-action="zoom-in" type="button">+</button>
        <button class="flh3r-button" data-action="fit" type="button">Fit source</button>
      </div>
      <div class="flh3r-canvas-wrap">
        <canvas class="flh3r-canvas" tabindex="0" aria-label="Temporal reshot timeline"></canvas>
        <div class="flh3r-empty" data-role="timeline-empty">Load a source to edit its temporal replacement range.</div>
      </div>
      <div class="flh3r-footer">Drag the purple body to move · drag either edge to resize · wheel zoom · Alt/middle drag pan · Shift+arrows move · Alt+arrows resize</div>
    `;
    this.container.appendChild(this.root);

    this.inspectorHost.className = "flh3r-inspector";
    this.inspectorHost.innerHTML = `
      <section class="flh3r-inspector-section">
        <div class="flh3r-inspector-title">Replacement prompt</div>
        <textarea data-field="prompt" placeholder="Describe the new action, performance, camera motion, or scene direction."></textarea>
      </section>
      <section class="flh3r-inspector-section">
        <div class="flh3r-inspector-title">Exact interval</div>
        <div class="flh3r-inspector-grid">
          <label class="flh3r-field">Start frame<input data-field="start" type="number" min="0" step="1"></label>
          <label class="flh3r-field">End frame exclusive<input data-field="end" type="number" min="1" step="1"></label>
          <label class="flh3r-field">Frame count<input data-field="count" type="number" min="1" step="1"></label>
          <label class="flh3r-field">Duration<input data-field="duration" type="text" readonly></label>
        </div>
      </section>
      <section class="flh3r-inspector-section">
        <div class="flh3r-inspector-title">Context and join</div>
        <div class="flh3r-inspector-grid">
          <label class="flh3r-field">Context before<input data-field="before" type="number" min="0" step="1"></label>
          <label class="flh3r-field">Context after<input data-field="after" type="number" min="0" step="1"></label>
          <label class="flh3r-field full">Edge blend frames<input data-field="blend" type="number" min="0" step="1"></label>
        </div>
      </section>
      <section class="flh3r-inspector-section">
        <div class="flh3r-inspector-title">H3 render diagnostics</div>
        <dl class="flh3r-diagnostics" data-role="diagnostics"></dl>
        <div data-role="warnings"></div>
      </section>
      <section class="flh3r-inspector-section">
        <div class="flh3r-inspector-title">Conditioning</div>
        <label class="flh3r-field">Reference image sizing<select data-field="ref-size"><option value="match">Match canvas</option><option value="max">Maximum reference size</option></select></label>
        <div class="flh3r-notice" data-role="audio-status"></div>
        <div class="flh3r-notice flh3r-reference-status" data-role="reference-status"></div>
        <div class="flh3r-notice">The assembler preserves the original soundtrack. Source audio conditions generation only when audio_vae is connected.</div>
      </section>
    `;

    const find = (selector) => this.root.querySelector(selector);
    this.video = find('[data-role="video"]');
    this.videoEmpty = find('[data-role="video-empty"]');
    this.videoOverlay = find('[data-role="video-overlay"]');
    this.timelineEmpty = find('[data-role="timeline-empty"]');
    this.canvas = find(".flh3r-canvas");
    this.playButton = find('[data-action="play"]');
    this.loopButton = find('[data-action="selection"]');
    this.muteButton = find('[data-action="mute"]');
    this.time = find('[data-role="time"]');
    this.sourceLabel = find('[data-role="source"]');
    this.status = find('[data-role="status"]');
    this.volume = find('[data-role="volume"]');
    this.volumeValue = find('[data-role="volume-value"]');
    this.fields = {
      prompt: this.inspectorHost.querySelector('[data-field="prompt"]'),
      start: this.inspectorHost.querySelector('[data-field="start"]'),
      end: this.inspectorHost.querySelector('[data-field="end"]'),
      count: this.inspectorHost.querySelector('[data-field="count"]'),
      duration: this.inspectorHost.querySelector('[data-field="duration"]'),
      before: this.inspectorHost.querySelector('[data-field="before"]'),
      after: this.inspectorHost.querySelector('[data-field="after"]'),
      blend: this.inspectorHost.querySelector('[data-field="blend"]'),
      refSize: this.inspectorHost.querySelector('[data-field="ref-size"]'),
    };
    this.diagnostics = this.inspectorHost.querySelector('[data-role="diagnostics"]');
    this.warnings = this.inspectorHost.querySelector('[data-role="warnings"]');
    this.audioStatus = this.inspectorHost.querySelector('[data-role="audio-status"]');
    this.referenceStatus = this.inspectorHost.querySelector('[data-role="reference-status"]');
  }

  bind() {
    this.playButton.addEventListener("click", () => this.togglePlayback());
    this.root.querySelector('[data-action="stop"]').addEventListener("click", () => this.stopPlayback());
    this.loopButton.addEventListener("click", () => this.playSelection());
    this.muteButton.addEventListener("click", () => this.controller.updateView({ muted: !this.video.muted }));
    this.root.querySelector('[data-action="fit"]').addEventListener("click", () => this.fitView());
    this.root.querySelector('[data-action="zoom-in"]').addEventListener("click", () => this.zoom(0.7));
    this.root.querySelector('[data-action="zoom-out"]').addEventListener("click", () => this.zoom(1.4));
    this.volume.addEventListener("input", () => {
      const volume = Number(this.volume.value) / 100;
      this.video.volume = volume;
      this.controller.updateView({ volume });
      this.volumeValue.textContent = `${Math.round(volume * 100)}%`;
    });
    this.video.addEventListener("play", () => { this.playButton.textContent = "Pause"; });
    this.video.addEventListener("pause", () => { this.playButton.textContent = "Play"; });
    this.video.addEventListener("timeupdate", () => {
      const settings = this.controller.settings;
      if (this.loopSelection) {
        const start = settings.start_frame / 24;
        const end = (settings.start_frame + settings.frame_count) / 24;
        if (this.video.currentTime >= end || this.video.currentTime < start) this.video.currentTime = start;
      }
      this.updateTransport();
      this.scheduleDraw();
    });
    this.video.addEventListener("loadedmetadata", () => this.updateTransport());

    this.fields.prompt.addEventListener("input", () => this.controller.setPrompt(this.fields.prompt.value));
    this.fields.refSize.addEventListener("change", () => this.controller.setRefImageSize(this.fields.refSize.value));
    const numericFields = ["start", "end", "count", "before", "after", "blend"];
    for (const name of numericFields) {
      this.fields[name].addEventListener("change", () => this.commitField(name));
    }

    this.canvas.addEventListener("pointerdown", (event) => this.pointerDown(event));
    this.canvas.addEventListener("pointermove", (event) => this.pointerMove(event));
    this.canvas.addEventListener("pointerup", (event) => this.pointerUp(event));
    this.canvas.addEventListener("pointercancel", (event) => this.pointerUp(event));
    this.canvas.addEventListener("wheel", (event) => this.wheel(event), { passive: false });
    this.root.addEventListener("keydown", (event) => this.keyDown(event));
    this.resizeObserver = new ResizeObserver(() => this.scheduleDraw());
    this.resizeObserver.observe(this.canvas);
  }

  sync() {
    const { path, sourceInfo, settings, prompt, view, error } = this.controller;
    const url = path ? this.controller.previewUrl() : "";
    if (url !== this.sourceUrl) {
      this.sourceUrl = url;
      this.video.pause();
      this.loopSelection = false;
      if (url) {
        this.video.src = url;
        this.video.load();
      } else {
        this.video.removeAttribute("src");
        this.video.load();
      }
    }
    this.video.muted = view.muted;
    this.video.volume = view.volume;
    this.muteButton.textContent = view.muted ? "Unmute" : "Mute";
    this.volume.value = String(Math.round(view.volume * 100));
    this.volumeValue.textContent = `${Math.round(view.volume * 100)}%`;
    this.videoEmpty.hidden = Boolean(path);
    this.timelineEmpty.hidden = Boolean(sourceInfo?.frame_count);
    this.sourceLabel.textContent = path || "No source selected";
    const status = compactReshotStatus(path, sourceInfo, settings, error);
    this.status.dataset.tone = status.tone;
    this.status.textContent = status.text;
    if (document.activeElement !== this.fields.prompt) this.fields.prompt.value = prompt;
    this.fields.refSize.value = this.controller.refImageSize;
    const end = settings.start_frame + settings.frame_count;
    const values = {
      start: settings.start_frame,
      end,
      count: settings.frame_count,
      duration: `${(settings.frame_count / 24).toFixed(3)} seconds`,
      before: settings.context_before,
      after: settings.context_after,
      blend: settings.edge_blend_frames,
    };
    for (const [name, value] of Object.entries(values)) {
      if (document.activeElement !== this.fields[name]) this.fields[name].value = String(value);
    }
    this.videoOverlay.textContent = sourceInfo
      ? `Frames ${settings.start_frame}–${end - 1} replace ${(settings.frame_count / 24).toFixed(2)}s of ${sourceInfo.frame_count} source frames.`
      : "The exact purple range will be replaced.";
    this.renderDiagnostics();
    this.updateTransport();
    this.scheduleDraw();
  }

  commitField(name) {
    const total = this.controller.sourceInfo?.frame_count || 0;
    const value = Math.trunc(Number(this.fields[name].value));
    if (!Number.isFinite(value)) return this.sync();
    const settings = { ...this.controller.settings };
    if (name === "start") settings.start_frame = value;
    else if (name === "end") settings.frame_count = value - settings.start_frame;
    else if (name === "count") settings.frame_count = value;
    else if (name === "before") settings.context_before = value;
    else if (name === "after") settings.context_after = value;
    else if (name === "blend") settings.edge_blend_frames = value;
    this.controller.setSettings(normalizeReshotSettings(settings, total));
  }

  renderDiagnostics() {
    const info = this.controller.sourceInfo;
    if (!info?.frame_count) {
      this.diagnostics.innerHTML = "<div><dt>Source</dt><dd>Not loaded</dd></div>";
      this.warnings.replaceChildren();
      this.audioStatus.textContent = "Source audio status is unavailable.";
      this.referenceStatus.textContent = `${this.controller.referenceCount()} reference image input${this.controller.referenceCount() === 1 ? " is" : "s are"} connected on the graph node.`;
      return;
    }
    const settings = this.controller.settings;
    const window = buildReshotWindow(info.frame_count, settings);
    const rows = [
      ["Source FPS", `${Number(info.source_frame_rate ?? info.frame_rate).toFixed(3)}${info.converted_to_24_fps ? " → 24" : ""}`],
      ["Working source", `${window.workStart}–${window.workStart + window.workSourceFrames - 1}`],
      ["Aligned render", `${window.renderFrames} frames`],
      ["Selection offset", `${window.selectionOffset} frames`],
      ["Expanded edit", `${window.editStart}–${Math.max(window.editStart, window.editEnd - 1)}`],
      ["Video tokens", `${window.tokenStart}–${window.tokenEnd - 1}`],
      ["End padding", `${window.paddingFrames} frames`],
    ];
    this.diagnostics.replaceChildren(...rows.map(([label, value]) => {
      const row = document.createElement("div");
      const term = document.createElement("dt");
      const detail = document.createElement("dd");
      term.textContent = label;
      detail.textContent = value;
      row.append(term, detail);
      return row;
    }));
    const warnings = reshotWarnings(info, settings, this.controller.audioVaeConnected());
    this.warnings.replaceChildren(...warnings.map((message) => {
      const notice = document.createElement("div");
      notice.className = "flh3r-notice warning";
      notice.textContent = message;
      return notice;
    }));
    if (!warnings.length) {
      const notice = document.createElement("div");
      notice.className = "flh3r-notice";
      notice.textContent = "The configured working window is ready to queue.";
      this.warnings.appendChild(notice);
    }
    this.audioStatus.textContent = info.has_audio
      ? this.controller.audioVaeConnected()
        ? "Source audio is present and will condition the generated interval."
        : "Source audio is present, but audio_vae is not connected."
      : "The source has no audio stream.";
    const references = this.controller.referenceCount();
    this.referenceStatus.textContent = references
      ? `${references} reference image${references === 1 ? " is" : "s are"} connected through the graph node.`
      : "No reference images are connected. Add them through the graph node's autogrow sockets.";
  }

  updateTransport() {
    const current = Math.round((Number(this.video.currentTime) || 0) * 24);
    const total = this.controller.sourceInfo?.frame_count || 0;
    this.time.textContent = `${formatTimecode(current)} / ${formatTimecode(total)}`;
    this.loopButton.classList.toggle("primary", this.loopSelection);
  }

  togglePlayback() {
    if (!this.video.src) return;
    this.loopSelection = false;
    if (this.video.paused) this.video.play().catch(() => {});
    else this.video.pause();
  }

  stopPlayback() {
    this.video.pause();
    this.loopSelection = false;
    this.video.currentTime = this.controller.settings.start_frame / 24;
    this.updateTransport();
  }

  playSelection() {
    if (!this.video.src) return;
    this.loopSelection = true;
    this.video.currentTime = this.controller.settings.start_frame / 24;
    this.video.play().catch(() => {});
    this.updateTransport();
  }

  fitView() {
    const total = this.controller.sourceInfo?.frame_count || 0;
    this.controller.updateView({ viewStart: 0, viewFrames: total });
  }

  zoom(factor, anchor = null) {
    const total = this.controller.sourceInfo?.frame_count || 0;
    if (!total) return;
    const view = this.controller.view;
    const center = anchor ?? view.viewStart + view.viewFrames / 2;
    const ratio = (center - view.viewStart) / view.viewFrames;
    const frames = clamp(view.viewFrames * factor, Math.min(12, total), total);
    const start = clamp(center - ratio * frames, 0, Math.max(0, total - frames));
    this.controller.updateView({ viewStart: start, viewFrames: frames });
  }

  geometry() {
    const rect = this.canvas.getBoundingClientRect();
    return { rect, left: 116, width: Math.max(1, rect.width - 132), height: rect.height };
  }

  pointerFrame(event) {
    const geometry = this.geometry();
    const view = this.controller.view;
    return frameAtX(event.clientX - geometry.rect.left, geometry.left, geometry.width, view.viewStart, view.viewFrames);
  }

  pointerMode(event) {
    if (!this.controller.sourceInfo) return null;
    if (event.button === 1 || event.altKey) return "pan";
    const geometry = this.geometry();
    const view = this.controller.view;
    const x = event.clientX - geometry.rect.left;
    const settings = this.controller.settings;
    const startX = xForFrame(settings.start_frame, geometry.left, geometry.width, view.viewStart, view.viewFrames);
    const endX = xForFrame(settings.start_frame + settings.frame_count, geometry.left, geometry.width, view.viewStart, view.viewFrames);
    if (Math.abs(x - startX) <= 9) return "start";
    if (Math.abs(x - endX) <= 9) return "end";
    if (x > startX && x < endX) return "move";
    return "seek";
  }

  pointerDown(event) {
    const mode = this.pointerMode(event);
    if (!mode) return;
    event.preventDefault();
    this.canvas.focus();
    this.drag = {
      mode,
      pointerFrame: this.pointerFrame(event),
      clientX: event.clientX,
      settings: { ...this.controller.settings },
      viewStart: this.controller.view.viewStart,
    };
    this.canvas.setPointerCapture(event.pointerId);
    this.pointerMove(event);
  }

  pointerMove(event) {
    if (!this.drag) {
      const mode = this.pointerMode(event);
      this.canvas.style.cursor = mode === "start" || mode === "end" ? "ew-resize" : mode === "move" ? "grab" : "pointer";
      return;
    }
    event.preventDefault();
    const total = this.controller.sourceInfo.frame_count;
    if (this.drag.mode === "pan") {
      const geometry = this.geometry();
      const frames = this.controller.view.viewFrames;
      const start = clamp(this.drag.viewStart - (event.clientX - this.drag.clientX) / geometry.width * frames, 0, Math.max(0, total - frames));
      this.controller.updateView({ viewStart: start });
      return;
    }
    const frame = Math.round(this.pointerFrame(event));
    if (this.drag.mode === "seek") return this.seekFrame(frame);
    const settings = this.drag.mode === "move"
      ? moveSelection(this.drag.settings, frame - Math.round(this.drag.pointerFrame), total)
      : resizeSelection(this.drag.settings, this.drag.mode, frame, total);
    this.controller.setSettings(settings);
  }

  pointerUp(event) {
    if (!this.drag) return;
    if (this.canvas.hasPointerCapture(event.pointerId)) this.canvas.releasePointerCapture(event.pointerId);
    this.drag = null;
  }

  wheel(event) {
    if (!this.controller.sourceInfo) return;
    event.preventDefault();
    this.zoom(Math.exp(event.deltaY * 0.002), this.pointerFrame(event));
  }

  keyDown(event) {
    if (!this.controller.sourceInfo || isTypingTarget(event.target)) return;
    if (event.code === "Space" || event.key === " ") {
      event.preventDefault();
      if (!event.repeat) this.togglePlayback();
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      this.playSelection();
      return;
    }
    if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      this.seekFrame(event.key === "Home" ? 0 : this.controller.sourceInfo.frame_count - 1);
      return;
    }
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const delta = event.key === "ArrowLeft" ? -1 : 1;
    if (event.shiftKey) {
      this.controller.setSettings(moveSelection(this.controller.settings, delta, this.controller.sourceInfo.frame_count));
    } else if (event.altKey) {
      const end = this.controller.settings.start_frame + this.controller.settings.frame_count + delta;
      this.controller.setSettings(resizeSelection(this.controller.settings, "end", end, this.controller.sourceInfo.frame_count));
    } else {
      this.seekFrame(Math.round(this.video.currentTime * 24) + delta);
    }
  }

  seekFrame(frame) {
    if (!this.video.src) return;
    this.loopSelection = false;
    this.video.currentTime = clamp(Math.trunc(frame), 0, this.controller.sourceInfo.frame_count - 1) / 24;
    this.updateTransport();
    this.scheduleDraw();
  }

  scheduleDraw() {
    if (this.drawRequest !== null) return;
    this.drawRequest = requestAnimationFrame(() => {
      this.drawRequest = null;
      this.drawTimeline();
    });
  }

  drawTimeline() {
    const geometry = this.geometry();
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const pixelWidth = Math.max(1, Math.round(geometry.rect.width * ratio));
    const pixelHeight = Math.max(1, Math.round(geometry.rect.height * ratio));
    if (this.canvas.width !== pixelWidth || this.canvas.height !== pixelHeight) {
      this.canvas.width = pixelWidth;
      this.canvas.height = pixelHeight;
    }
    const context = this.canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, geometry.rect.width, geometry.rect.height);
    const info = this.controller.sourceInfo;
    if (!info?.frame_count) return;
    const settings = this.controller.settings;
    const view = this.controller.view;
    const plan = buildReshotWindow(info.frame_count, settings);
    const x = (frame) => xForFrame(frame, geometry.left, geometry.width, view.viewStart, view.viewFrames);
    const rows = [
      ["Source", 42, 38],
      ["Requested context", 98, 38],
      ["Expanded H3 edit", 154, 38],
      ["Aligned render", 210, 38],
    ];
    context.font = "9px Inter, sans-serif";
    context.textBaseline = "middle";
    for (const [label, y, height] of rows) {
      context.fillStyle = "#8b8b95";
      context.fillText(label, 10, y + height / 2);
      context.fillStyle = "#272a32";
      context.fillRect(geometry.left, y, geometry.width, height);
    }
    const drawRange = (from, to, y, height, color) => {
      const left = clamp(x(from), geometry.left, geometry.left + geometry.width);
      const right = clamp(x(to), geometry.left, geometry.left + geometry.width);
      if (right <= left) return [left, right];
      context.fillStyle = color;
      context.fillRect(left, y, right - left, height);
      return [left, right];
    };
    drawRange(0, info.frame_count, rows[0][1], rows[0][2], "#41444e");
    const selectionEnd = settings.start_frame + settings.frame_count;
    drawRange(plan.requestedStart, settings.start_frame, rows[1][1], rows[1][2], "#0e7490");
    drawRange(settings.start_frame, selectionEnd, rows[1][1], rows[1][2], "#7c3aed");
    drawRange(selectionEnd, plan.requestedEnd, rows[1][1], rows[1][2], "#0e7490");
    const [tokenLeft, tokenRight] = drawRange(plan.editStart, plan.editEnd, rows[2][1], rows[2][2], "rgba(180, 83, 9, .72)");
    context.save();
    context.beginPath();
    context.rect(tokenLeft, rows[2][1], Math.max(0, tokenRight - tokenLeft), rows[2][2]);
    context.clip();
    context.strokeStyle = "rgba(253, 186, 116, .55)";
    for (let hatch = tokenLeft - 40; hatch < tokenRight + 40; hatch += 9) {
      context.beginPath();
      context.moveTo(hatch, rows[2][1] + rows[2][2]);
      context.lineTo(hatch + rows[2][2], rows[2][1]);
      context.stroke();
    }
    context.restore();
    drawRange(settings.start_frame, selectionEnd, rows[2][1] + 7, rows[2][2] - 14, "#7c3aed");
    drawRange(plan.workStart, plan.workStart + plan.workSourceFrames, rows[3][1], rows[3][2], "#334155");
    if (plan.paddingFrames) {
      const right = geometry.left + geometry.width;
      context.fillStyle = "#ca8a04";
      context.fillRect(right - 9, rows[3][1], 9, rows[3][2]);
      context.fillStyle = "#fef08a";
      context.textAlign = "right";
      context.fillText(`+${plan.paddingFrames} padded`, right - 13, rows[3][1] + rows[3][2] / 2);
      context.textAlign = "left";
    }
    const startX = x(settings.start_frame);
    const endX = x(selectionEnd);
    context.fillStyle = "#ddd6fe";
    context.fillRect(startX - 2, rows[1][1] - 5, 4, rows[1][2] + 10);
    context.fillRect(endX - 2, rows[1][1] - 5, 4, rows[1][2] + 10);
    const playhead = x(clamp((Number(this.video.currentTime) || 0) * 24, 0, info.frame_count));
    context.fillStyle = "#fff";
    context.fillRect(playhead, 27, 1, 228);
    context.beginPath();
    context.moveTo(playhead - 4, 27);
    context.lineTo(playhead + 4, 27);
    context.lineTo(playhead, 34);
    context.fill();

    context.fillStyle = "#8c929f";
    context.textBaseline = "top";
    const divisions = Math.max(4, Math.min(10, Math.round(geometry.width / 130)));
    for (let index = 0; index <= divisions; index += 1) {
      const frame = Math.round(view.viewStart + view.viewFrames * index / divisions);
      const tickX = x(frame);
      context.fillRect(tickX, 18, 1, 8);
      const label = `${frame} · ${formatTimecode(frame)}`;
      const width = context.measureText(label).width;
      context.fillText(label, clamp(tickX - width / 2, geometry.left, geometry.left + geometry.width - width), 3);
    }
  }

  saveViewState() {
    this.controller.updateView({
      muted: this.video.muted,
      volume: this.video.volume,
    }, false);
  }

  dispose() {
    this.saveViewState();
    this.unsubscribe?.();
    this.resizeObserver?.disconnect();
    if (this.drawRequest !== null) cancelAnimationFrame(this.drawRequest);
    this.video.pause();
    this.video.removeAttribute("src");
    this.video.load();
    this.root.remove();
    this.inspectorHost.replaceChildren();
  }
}
