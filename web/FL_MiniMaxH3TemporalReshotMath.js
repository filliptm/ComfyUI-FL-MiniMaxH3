export const DEFAULT_RESHOT_SETTINGS = Object.freeze({
  version: 1,
  start_frame: 0,
  frame_count: 72,
  context_before: 39,
  context_after: 39,
  edge_blend_frames: 0,
});

export function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

export function alignH3Frames(value) {
  let frames = Math.max(5, Math.trunc(value));
  while (frames % 17 !== 5) frames += 1;
  return frames;
}

export function videoTokenFrameEdges(videoTokens) {
  const pattern = [1, 4, 4, 4, 4];
  const edges = [0];
  for (let index = 0; index < videoTokens; index += 1) {
    edges.push(edges.at(-1) + pattern[index % pattern.length]);
  }
  return edges;
}

export function videoTokenCount(frameCount) {
  return frameCount <= 5 ? 2 : Math.trunc((frameCount - 5) / 17) * 5 + 2;
}

function lowerBound(values, target) {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = Math.trunc((low + high) / 2);
    if (values[middle] < target) low = middle + 1;
    else high = middle;
  }
  return low;
}

function upperBound(values, target) {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = Math.trunc((low + high) / 2);
    if (values[middle] <= target) low = middle + 1;
    else high = middle;
  }
  return low;
}

export function selectionTokenRange(renderFrames, selectionOffset, selectionFrames) {
  const edges = videoTokenFrameEdges(videoTokenCount(renderFrames));
  if (edges.at(-1) !== renderFrames) throw new Error("Render length is not on the H3 temporal grid.");
  const tokenStart = clamp(upperBound(edges, selectionOffset) - 1, 0, edges.length - 2);
  const tokenEnd = clamp(lowerBound(edges, selectionOffset + selectionFrames), tokenStart + 1, edges.length - 1);
  return {
    tokenStart,
    tokenEnd,
    edges,
    expandedStart: edges[tokenStart],
    expandedEnd: edges[tokenEnd],
  };
}

export function normalizeReshotSettings(value, totalFrames = 0) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const settings = { ...DEFAULT_RESHOT_SETTINGS };
  for (const name of ["start_frame", "frame_count", "context_before", "context_after", "edge_blend_frames"]) {
    const number = Number(source[name]);
    if (Number.isFinite(number)) settings[name] = Math.trunc(number);
  }
  settings.start_frame = Math.max(0, settings.start_frame);
  settings.frame_count = Math.max(1, settings.frame_count);
  settings.context_before = Math.max(0, settings.context_before);
  settings.context_after = Math.max(0, settings.context_after);
  settings.edge_blend_frames = Math.max(0, settings.edge_blend_frames);
  if (totalFrames > 0) {
    settings.frame_count = Math.min(settings.frame_count, totalFrames);
    settings.start_frame = clamp(settings.start_frame, 0, totalFrames - settings.frame_count);
    settings.edge_blend_frames = Math.min(settings.edge_blend_frames, settings.frame_count);
  }
  return settings;
}

export function moveSelection(settings, delta, totalFrames) {
  const next = normalizeReshotSettings(settings, totalFrames);
  next.start_frame = clamp(next.start_frame + Math.trunc(delta), 0, Math.max(0, totalFrames - next.frame_count));
  return next;
}

export function resizeSelection(settings, edge, frame, totalFrames) {
  const next = normalizeReshotSettings(settings, totalFrames);
  const end = next.start_frame + next.frame_count;
  if (edge === "start") {
    const start = clamp(Math.trunc(frame), 0, end - 1);
    next.start_frame = start;
    next.frame_count = end - start;
  } else {
    const resizedEnd = clamp(Math.trunc(frame), next.start_frame + 1, totalFrames);
    next.frame_count = resizedEnd - next.start_frame;
  }
  next.edge_blend_frames = Math.min(next.edge_blend_frames, next.frame_count);
  return next;
}

export function buildReshotWindow(totalFrames, settings) {
  const value = normalizeReshotSettings(settings, totalFrames);
  const selectionEnd = value.start_frame + value.frame_count;
  const requestedStart = Math.max(0, value.start_frame - value.context_before);
  const requestedEnd = Math.min(totalFrames, selectionEnd + value.context_after);
  const renderFrames = alignH3Frames(requestedEnd - requestedStart);
  let workStart = requestedStart;
  if (workStart + renderFrames > totalFrames) workStart = Math.max(0, totalFrames - renderFrames);
  const workSourceFrames = Math.min(renderFrames, totalFrames - workStart);
  const selectionOffset = value.start_frame - workStart;
  const tokens = selectionTokenRange(renderFrames, selectionOffset, value.frame_count);
  return {
    requestedStart,
    requestedEnd,
    renderFrames,
    workStart,
    workSourceFrames,
    paddingFrames: renderFrames - workSourceFrames,
    selectionOffset,
    editStart: workStart + tokens.expandedStart,
    editEnd: Math.min(totalFrames, workStart + tokens.expandedEnd),
    ...tokens,
  };
}

export function frameAtX(x, left, width, viewStart, viewFrames) {
  if (width <= 0 || viewFrames <= 0) return viewStart;
  return viewStart + clamp((x - left) / width, 0, 1) * viewFrames;
}

export function xForFrame(frame, left, width, viewStart, viewFrames) {
  if (width <= 0 || viewFrames <= 0) return left;
  return left + ((frame - viewStart) / viewFrames) * width;
}
