import { api } from "../../../scripts/api.js";
import { TemporalReshotEditor } from "./FL_MiniMaxH3TemporalReshotEditor.js";
import {
  filenameFromPath,
  formatBytes,
} from "./FL_MiniMaxH3TemporalReshotState.js";
import { injectTemporalReshotStyles } from "./FL_MiniMaxH3TemporalReshotStyles.js";


const VIDEO_FILE_RE = /\.(?:avi|gif|m4v|mkv|mov|mp4|webm)$/i;
const EDITORS = new Map();
let activeModal = null;

function supportedVideo(file) {
  return Boolean(file && ((file.type || "").startsWith("video/") || VIDEO_FILE_RE.test(file.name || "")));
}

class TemporalReshotModal {
  constructor(controller) {
    this.controller = controller;
    this.node = controller.node;
    this.entries = [];
    this.closed = false;
    this.previousBodyOverflow = "";
    this.build();
  }

  build() {
    injectTemporalReshotStyles();
    this.overlay = document.createElement("div");
    this.overlay.className = "flh3r-modal-overlay";
    this.overlay.setAttribute("role", "dialog");
    this.overlay.setAttribute("aria-modal", "true");
    this.overlay.setAttribute("aria-label", "FL MiniMax H3 Temporal Reshot Editor");
    this.overlay.innerHTML = `
      <div class="flh3r-modal-shell">
        <header class="flh3r-modal-header">
          <div class="flh3r-modal-heading">
            <div class="flh3r-modal-title">FL MiniMax H3 Temporal Reshot Editor</div>
            <div class="flh3r-modal-subtitle" data-role="subtitle">Choose a source video to begin.</div>
          </div>
          <span class="flh3r-spacer"></span>
          <button class="flh3r-button primary" data-action="done" type="button">Done</button>
        </header>
        <div class="flh3r-modal-main">
          <aside class="flh3r-library">
            <div class="flh3r-library-label">Source video</div>
            <div class="flh3r-drop-zone" data-role="drop-zone" tabindex="0">Drop a video here<br>or click to upload one</div>
            <div class="flh3r-library-actions">
              <button class="flh3r-button" data-action="choose" type="button">Choose file</button>
              <button class="flh3r-button" data-action="refresh" type="button">Refresh</button>
            </div>
            <input class="flh3r-library-search" data-role="search" type="search" placeholder="Search input videos">
            <select class="flh3r-library-folder" data-role="folder" aria-label="Filter input folder"></select>
            <div class="flh3r-library-results" data-role="results"></div>
            <div class="flh3r-source-card" data-role="source-card">No source selected.</div>
            <div class="flh3r-library-message" data-role="library-message"></div>
            <button class="flh3r-sidebar-toggle" data-action="toggle-library" type="button" aria-expanded="true">‹</button>
          </aside>
          <main class="flh3r-editor-host" data-role="editor"></main>
          <aside data-role="inspector"></aside>
        </div>
      </div>
    `;
    this.shell = this.overlay.querySelector(".flh3r-modal-shell");
    this.subtitle = this.overlay.querySelector('[data-role="subtitle"]');
    this.library = this.overlay.querySelector(".flh3r-library");
    this.dropZone = this.overlay.querySelector('[data-role="drop-zone"]');
    this.search = this.overlay.querySelector('[data-role="search"]');
    this.folder = this.overlay.querySelector('[data-role="folder"]');
    this.results = this.overlay.querySelector('[data-role="results"]');
    this.sourceCard = this.overlay.querySelector('[data-role="source-card"]');
    this.libraryMessage = this.overlay.querySelector('[data-role="library-message"]');
    this.libraryToggle = this.overlay.querySelector('[data-action="toggle-library"]');
    this.editorHost = this.overlay.querySelector('[data-role="editor"]');
    this.inspectorHost = this.overlay.querySelector('[data-role="inspector"]');
    this.fileInput = document.createElement("input");
    this.fileInput.type = "file";
    this.fileInput.accept = "video/*,.avi,.gif,.m4v,.mkv,.mov,.mp4,.webm";
    this.fileInput.hidden = true;
    this.library.appendChild(this.fileInput);
    this.syncLibraryVisibility();

    this.overlay.querySelector('[data-action="done"]').addEventListener("click", () => this.close());
    this.overlay.querySelector('[data-action="choose"]').addEventListener("click", () => this.chooseFile());
    this.overlay.querySelector('[data-action="refresh"]').addEventListener("click", () => this.refreshLibrary());
    this.libraryToggle.addEventListener("click", () => this.toggleLibrary());
    this.dropZone.addEventListener("click", () => this.chooseFile());
    this.dropZone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        this.chooseFile();
      }
    });
    this.fileInput.addEventListener("change", () => {
      const file = this.fileInput.files?.[0];
      if (file) this.uploadFile(file);
      this.fileInput.value = "";
    });
    this.search.addEventListener("input", () => this.renderFiles());
    this.folder.addEventListener("change", () => this.renderFiles(false));
    this.library.addEventListener("dragover", (event) => {
      if (!event.dataTransfer?.types?.includes("Files")) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
      this.dropZone.classList.add("dragging");
    });
    this.library.addEventListener("dragleave", (event) => {
      if (!this.library.contains(event.relatedTarget)) this.dropZone.classList.remove("dragging");
    });
    this.library.addEventListener("drop", (event) => {
      event.preventDefault();
      this.dropZone.classList.remove("dragging");
      const file = [...(event.dataTransfer?.files || [])].find(supportedVideo);
      if (file) this.uploadFile(file);
      else this.setLibraryMessage("Drop a supported video file.", true);
    });
    this.overlay.addEventListener("pointerdown", (event) => {
      if (event.target === this.overlay) this.close();
    });
    for (const type of ["pointerdown", "pointermove", "pointerup", "wheel"]) {
      this.shell.addEventListener(type, (event) => event.stopPropagation(), { passive: type === "wheel" });
    }
    this.keyHandler = (event) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      this.close();
    };
    this.overlay.addEventListener("keydown", this.keyHandler);
  }

  show() {
    if (activeModal && activeModal !== this) activeModal.close();
    activeModal = this;
    this.previousBodyOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    document.body.appendChild(this.overlay);
    this.editor = new TemporalReshotEditor({
      controller: this.controller,
      container: this.editorHost,
      inspector: this.inspectorHost,
    });
    EDITORS.set(this.node.id, this.editor);
    this.unsubscribe = this.controller.subscribe(() => this.sync());
    this.sync();
    this.refreshLibrary();
    requestAnimationFrame(() => {
      this.shell.tabIndex = -1;
      this.shell.focus({ preventScroll: true });
      this.editor.scheduleDraw();
    });
  }

  chooseFile() {
    this.fileInput.click();
  }

  async uploadFile(file) {
    if (!supportedVideo(file)) {
      this.setLibraryMessage("Choose a supported video file.", true);
      return;
    }
    this.setLibraryMessage(`Uploading ${file.name}…`);
    const body = new FormData();
    body.append("image", file);
    body.append("type", "input");
    try {
      const response = await api.fetchApi("/upload/image", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `Upload failed (${response.status}).`);
      const path = [payload.subfolder, payload.name].filter(Boolean).join("/").replace(/\\/g, "/");
      await this.refreshLibrary();
      await this.selectSource(path, true);
    } catch (error) {
      this.setLibraryMessage(error.message || "Video upload failed.", true);
    }
  }

  async refreshLibrary() {
    this.setLibraryMessage("Refreshing ComfyUI input videos…");
    try {
      const response = await api.fetchApi("/fl/minimax-h3/temporal-reshot/files");
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `Library request failed (${response.status}).`);
      this.entries = Array.isArray(payload.files) ? payload.files : [];
      this.renderFolders();
      this.renderFiles();
      this.setLibraryMessage(`${this.entries.length} input video${this.entries.length === 1 ? "" : "s"}.`);
    } catch (error) {
      this.entries = [];
      this.renderFolders();
      this.renderFiles();
      this.setLibraryMessage(error.message || "Could not refresh the input video library.", true);
    }
  }

  renderFolders() {
    const selected = this.folder.value;
    const folders = [...new Set(this.entries.map((entry) => entry.folder || ""))].sort((a, b) => a.localeCompare(b));
    this.folder.replaceChildren();
    const all = document.createElement("option");
    all.value = "";
    all.textContent = "All input folders";
    this.folder.appendChild(all);
    for (const value of folders.filter(Boolean)) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      this.folder.appendChild(option);
    }
    this.folder.value = folders.includes(selected) ? selected : "";
  }

  renderFiles(resetScroll = true) {
    const query = this.search.value.trim().toLocaleLowerCase();
    const folder = this.folder.value;
    const current = this.controller.path;
    const entries = this.entries.filter((entry) => {
      if (folder && entry.folder !== folder) return false;
      return !query || entry.path.toLocaleLowerCase().includes(query);
    });
    this.results.replaceChildren(...entries.map((entry) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "flh3r-library-row";
      button.classList.toggle("active", entry.path === current);
      const title = document.createElement("strong");
      title.textContent = entry.filename;
      const folderLine = document.createElement("small");
      folderLine.textContent = entry.folder || "ComfyUI input";
      const details = document.createElement("small");
      details.textContent = entry.error
        ? entry.error
        : `${entry.width}×${entry.height} · ${Number(entry.frame_rate).toFixed(3)} fps · ` +
          `${entry.frame_count || "?"}f · ${Number(entry.duration).toFixed(2)}s · ` +
          `${entry.has_audio ? "audio" : "silent"} · ${formatBytes(entry.size)}`;
      button.append(title, folderLine, details);
      button.addEventListener("click", () => this.selectSource(entry.path));
      return button;
    }));
    if (!entries.length) {
      const empty = document.createElement("div");
      empty.className = "flh3r-library-message";
      empty.textContent = this.entries.length ? "No videos match this filter." : "No videos found in ComfyUI input.";
      this.results.appendChild(empty);
    }
    if (resetScroll) this.results.scrollTop = 0;
  }

  async selectSource(path, forceInitialize = false) {
    this.setLibraryMessage(`Inspecting ${filenameFromPath(path)}…`);
    await this.controller.setSource(path, forceInitialize);
    if (this.controller.error) this.setLibraryMessage(this.controller.error, true);
    else this.setLibraryMessage("Source selected. Edits save directly to the node.");
    this.renderFiles(false);
  }

  setLibraryMessage(message, error = false) {
    this.libraryMessage.textContent = message || "";
    this.libraryMessage.classList.toggle("error", error);
  }

  sync() {
    const { path, sourceInfo, settings } = this.controller;
    const end = settings.start_frame + settings.frame_count;
    this.subtitle.textContent = path
      ? `${path} · frames ${settings.start_frame}–${end - 1} · edits save directly to the node`
      : "Choose a source from ComfyUI input or upload one video";
    this.sourceCard.textContent = sourceInfo
      ? `${sourceInfo.width}×${sourceInfo.height} · ${sourceInfo.frame_count} frames at 24 fps · ` +
        `${Number(sourceInfo.source_frame_rate ?? sourceInfo.frame_rate).toFixed(3)} source fps` +
        `${sourceInfo.converted_to_24_fps ? " normalized to 24 fps" : ""} · ` +
        `${sourceInfo.has_audio ? "source audio" : "silent"}`
      : "No source selected.";
    this.renderFiles(false);
  }

  syncLibraryVisibility() {
    const collapsed = this.controller.view.libraryCollapsed;
    this.shell.classList.toggle("library-collapsed", collapsed);
    this.libraryToggle.textContent = collapsed ? "›" : "‹";
    this.libraryToggle.setAttribute("aria-expanded", String(!collapsed));
    this.libraryToggle.title = collapsed ? "Show source library" : "Hide source library";
  }

  toggleLibrary() {
    this.controller.updateView({ libraryCollapsed: !this.controller.view.libraryCollapsed });
    this.syncLibraryVisibility();
    setTimeout(() => this.editor?.scheduleDraw(), 180);
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    this.unsubscribe?.();
    this.editor?.dispose();
    this.editor = null;
    EDITORS.delete(this.node.id);
    this.overlay.removeEventListener("keydown", this.keyHandler);
    this.overlay.remove();
    document.body.style.overflow = this.previousBodyOverflow;
    if (activeModal === this) activeModal = null;
  }
}

export function openTemporalReshotEditor(controller) {
  const modal = new TemporalReshotModal(controller);
  modal.show();
  return modal;
}

export function getTemporalReshotEditor(nodeId) {
  return EDITORS.get(nodeId);
}

export function closeTemporalReshotForNode(node) {
  if (activeModal?.node === node) activeModal.close();
}
