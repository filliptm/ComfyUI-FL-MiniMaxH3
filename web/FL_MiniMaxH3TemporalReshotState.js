import {
  DEFAULT_RESHOT_SETTINGS,
  buildReshotWindow,
  clamp,
  normalizeReshotSettings,
} from "./FL_MiniMaxH3TemporalReshotMath.js";


export const RESHOT_VIEW_VERSION = 2;

export function filenameFromPath(path) {
  return String(path || "")
    .replace(/ \[(input|output|temp)\]$/, "")
    .replace(/\\/g, "/")
    .split("/")
    .pop() || "";
}

export function previewReference(path) {
  const normalized = String(path || "")
    .replace(/ \[(input|output|temp)\]$/, "")
    .replace(/\\/g, "/");
  const parts = normalized.split("/");
  return { filename: parts.pop() || "", subfolder: parts.join("/") };
}

export function formatTimecode(frame, fps = 24) {
  const seconds = Math.max(0, Number(frame) || 0) / fps;
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, "0")}:${(seconds % 60).toFixed(3).padStart(6, "0")}`;
}

export function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 MB";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  return `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`;
}

export function migrateReshotView(value, totalFrames = 0) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const total = Math.max(0, Math.trunc(Number(totalFrames) || 0));
  const volume = Number(source.volume);
  const view = {
    version: RESHOT_VIEW_VERSION,
    viewStart: Number.isFinite(Number(source.viewStart)) ? Number(source.viewStart) : 0,
    viewFrames: Number.isFinite(Number(source.viewFrames)) ? Number(source.viewFrames) : total,
    libraryCollapsed: Boolean(source.libraryCollapsed),
    muted: source.muted === undefined ? true : Boolean(source.muted),
    volume: Number.isFinite(volume) ? clamp(volume, 0, 1) : 0.8,
  };
  if (total > 0) {
    view.viewFrames = clamp(view.viewFrames || total, Math.min(12, total), total);
    view.viewStart = clamp(view.viewStart, 0, Math.max(0, total - view.viewFrames));
  } else {
    view.viewStart = 0;
    view.viewFrames = 0;
  }
  return view;
}

export function readReshotSettings(value, totalFrames = 0) {
  try {
    return normalizeReshotSettings(JSON.parse(value), totalFrames);
  } catch {
    return normalizeReshotSettings(DEFAULT_RESHOT_SETTINGS, totalFrames);
  }
}

export function reshotWarnings(sourceInfo, settings, audioVaeConnected = false) {
  if (!sourceInfo?.frame_count) return [];
  const window = buildReshotWindow(sourceInfo.frame_count, settings);
  const warnings = [];
  if (window.renderFrames < 124) warnings.push("The render window is below H3's approximate 5-second trained range.");
  if (window.renderFrames > 362) warnings.push("The render window exceeds H3's approximate 15-second trained range.");
  if (window.paddingFrames) warnings.push(`${window.paddingFrames} end frame${window.paddingFrames === 1 ? "" : "s"} will be repeated for H3 alignment.`);
  const end = settings.start_frame + settings.frame_count;
  if (settings.start_frame === 0 || end === sourceInfo.frame_count) warnings.push("One side of the reshot has no source context.");
  if (settings.frame_count === sourceInfo.frame_count) warnings.push("The full source will be regenerated.");
  if (sourceInfo.has_audio && !audioVaeConnected) warnings.push("Connect audio_vae to use the source soundtrack as generation context.");
  return warnings;
}

export function compactReshotStatus(path, sourceInfo, settings, error = "") {
  if (error) return { tone: "error", text: error };
  if (!path) return { tone: "idle", text: "No source selected" };
  if (!sourceInfo?.frame_count) return { tone: "busy", text: `${filenameFromPath(path)} · inspecting source` };
  const normalized = normalizeReshotSettings(settings, sourceInfo.frame_count);
  const end = normalized.start_frame + normalized.frame_count;
  const render = buildReshotWindow(sourceInfo.frame_count, normalized).renderFrames;
  const configuredSourceRate = Number(sourceInfo.source_frame_rate ?? sourceInfo.frame_rate);
  const sourceRate = Number.isFinite(configuredSourceRate) ? configuredSourceRate : 24;
  const conversion = sourceInfo.converted_to_24_fps || sourceRate !== 24 || sourceInfo.source_constant_frame_rate === false
    ? ` · ${sourceRate.toFixed(3)}→24 fps`
    : "";
  return {
    tone: "ready",
    text: `${filenameFromPath(path)}${conversion} · frames ${normalized.start_frame}–${end - 1} · ` +
      `${(normalized.frame_count / 24).toFixed(2)}s · H3 ${render}f`,
  };
}
