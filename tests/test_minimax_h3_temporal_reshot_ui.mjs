import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";


const source = await readFile(
  new URL("../web/FL_MiniMaxH3TemporalReshotMath.js", import.meta.url),
  "utf8",
);
const math = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
const state = await import(new URL("../web/FL_MiniMaxH3TemporalReshotState.js", import.meta.url));
const modalSource = await readFile(
  new URL("../web/FL_MiniMaxH3TemporalReshotModal.js", import.meta.url),
  "utf8",
);
const nodeSource = await readFile(
  new URL("../web/FL_MiniMaxH3TemporalReshot.js", import.meta.url),
  "utf8",
);
const stylesSource = await readFile(
  new URL("../web/FL_MiniMaxH3TemporalReshotStyles.js", import.meta.url),
  "utf8",
);


test("H3 frames align to 17k+5", () => {
  assert.equal(math.alignH3Frames(1), 5);
  assert.equal(math.alignH3Frames(10), 22);
  assert.equal(math.alignH3Frames(102), 107);
});

test("selection move and resize remain inside the source", () => {
  const base = math.normalizeReshotSettings({ start_frame: 10, frame_count: 20 }, 100);
  assert.deepEqual(
    { ...math.moveSelection(base, 1000, 100) },
    { ...base, start_frame: 80 },
  );
  const resized = math.resizeSelection(base, "start", 25, 100);
  assert.equal(resized.start_frame, 25);
  assert.equal(resized.frame_count, 5);
});

test("window shifts earlier at the source end", () => {
  const window = math.buildReshotWindow(360, {
    start_frame: 350,
    frame_count: 10,
    context_before: 0,
    context_after: 0,
  });
  assert.equal(window.renderFrames, 22);
  assert.equal(window.workStart, 338);
  assert.equal(window.selectionOffset, 12);
});

test("selection expands across complete H3 temporal tokens", () => {
  const range = math.selectionTokenRange(22, 3, 3);
  assert.deepEqual(range.edges, [0, 1, 5, 9, 13, 17, 18, 22]);
  assert.equal(range.tokenStart, 1);
  assert.equal(range.tokenEnd, 3);
  assert.equal(range.expandedStart, 1);
  assert.equal(range.expandedEnd, 9);
});

test("timeline coordinates round trip within a frame", () => {
  const x = math.xForFrame(125, 10, 500, 100, 100);
  assert.equal(x, 135);
  assert.equal(math.frameAtX(x, 10, 500, 100, 100), 125);
});

test("legacy embedded-panel view state migrates without losing viewport or audio preferences", () => {
  const view = state.migrateReshotView({
    viewStart: 30,
    viewFrames: 120,
    muted: false,
  }, 360);
  assert.equal(view.version, 2);
  assert.equal(view.viewStart, 30);
  assert.equal(view.viewFrames, 120);
  assert.equal(view.muted, false);
  assert.equal(view.volume, 0.8);
  assert.equal(view.libraryCollapsed, false);
});

test("invalid saved viewport is clamped to the current source", () => {
  const view = state.migrateReshotView({ viewStart: 999, viewFrames: -10, volume: 4 }, 100);
  assert.equal(view.viewFrames, 12);
  assert.equal(view.viewStart, 88);
  assert.equal(view.volume, 1);
});

test("compact status reports exact interval and aligned H3 length", () => {
  const status = state.compactReshotStatus(
    "folder/source.mp4",
    { frame_count: 360, frame_rate: 24, constant_frame_rate: true },
    { start_frame: 100, frame_count: 24, context_before: 39, context_after: 39 },
  );
  assert.equal(status.tone, "ready");
  assert.match(status.text, /source\.mp4/);
  assert.match(status.text, /frames 100–123/);
  assert.match(status.text, /H3 107f/);
});

test("non-24 fps sources report automatic timeline normalization", () => {
  const status = state.compactReshotStatus(
    "source.mp4",
    { frame_count: 48, frame_rate: 24, source_frame_rate: 30, converted_to_24_fps: true },
    { start_frame: 0, frame_count: 24, context_before: 39, context_after: 39 },
  );
  assert.equal(status.tone, "ready");
  assert.match(status.text, /30\.000→24 fps/);
  assert.ok(!state.reshotWarnings(
    { frame_count: 48, frame_rate: 24, source_frame_rate: 30, converted_to_24_fps: true },
    { start_frame: 0, frame_count: 24, context_before: 39, context_after: 39 },
  ).some((message) => message.includes("Convert")));
});

test("warnings distinguish missing audio conditioning from soundtrack preservation", () => {
  const warnings = state.reshotWarnings(
    { frame_count: 360, frame_rate: 24, constant_frame_rate: true, has_audio: true },
    { start_frame: 120, frame_count: 72, context_before: 39, context_after: 39 },
    false,
  );
  assert.ok(warnings.some((message) => message.includes("audio_vae")));
  assert.ok(!state.reshotWarnings(
    { frame_count: 360, frame_rate: 24, constant_frame_rate: true, has_audio: true },
    { start_frame: 120, frame_count: 72, context_before: 39, context_after: 39 },
    true,
  ).some((message) => message.includes("audio_vae")));
});

test("modal lifecycle restores the canvas and closes when its node is removed", () => {
  assert.match(modalSource, /document\.body\.style\.overflow = "hidden"/);
  assert.match(modalSource, /document\.body\.style\.overflow = this\.previousBodyOverflow/);
  assert.match(modalSource, /this\.editor\?\.dispose\(\)/);
  assert.match(modalSource, /activeModal\?\.node === node/);
  assert.match(nodeSource, /closeTemporalReshotForNode\(this\)/);
});

test("compact node keeps a sixteen by nine preview and opens the modal editor", () => {
  assert.match(stylesSource, /aspect-ratio:\s*16 \/ 9/);
  assert.match(nodeSource, /Open Temporal Reshot Editor/);
  assert.match(nodeSource, /hideWidget\(this\.widgets\.prompt\)/);
});
