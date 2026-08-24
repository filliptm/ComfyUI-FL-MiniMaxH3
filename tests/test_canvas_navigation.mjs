import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const WEB_URL = new URL("../web/", import.meta.url);

class FakeElement {
  constructor() {
    this.listeners = new Map();
    this.events = [];
  }

  addEventListener(name, handler) {
    const handlers = this.listeners.get(name) || [];
    handlers.push(handler);
    this.listeners.set(name, handlers);
  }

  dispatchEvent(event) {
    this.events.push(event);
  }

  emit(name, event) {
    for (const handler of this.listeners.get(name) || []) handler(event);
  }
}

class FakeEvent {
  constructor(type, init = {}) {
    Object.assign(this, init);
    this.type = type;
  }

  preventDefault() {
    this.defaultPrevented = true;
  }

  stopPropagation() {
    this.propagationStopped = true;
  }
}

globalThis.PointerEvent = FakeEvent;
globalThis.WheelEvent = FakeEvent;

const source = await readFile(new URL("canvas_navigation.js", WEB_URL), "utf8");
const moduleUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
const { addCanvasNavigation } = await import(moduleUrl);

function pointerEvent(type, overrides = {}) {
  return new FakeEvent(type, {
    pointerId: 4,
    button: -1,
    buttons: 0,
    altKey: false,
    ctrlKey: false,
    shiftKey: false,
    ...overrides,
  });
}

test("MiniMax H3 panels pass wheel and canvas drag gestures to LiteGraph", () => {
  const element = new FakeElement();
  const graphCanvas = new FakeElement();
  const canvas = { canvas: graphCanvas, dragZoomEnabled: true, read_only: false };
  addCanvasNavigation(element, canvas);

  const wheel = new FakeEvent("wheel", { deltaY: -120 });
  element.emit("wheel", wheel);
  assert.deepEqual(graphCanvas.events.map((event) => event.type), ["wheel"]);
  assert.equal(wheel.defaultPrevented, true);
  assert.equal(wheel.propagationStopped, true);

  element.emit("pointerdown", pointerEvent("pointerdown", { button: 0, buttons: 1 }));
  assert.equal(graphCanvas.events.length, 1);

  element.emit("pointerdown", pointerEvent("pointerdown", { button: 1, buttons: 4 }));
  element.emit("pointermove", pointerEvent("pointermove", { buttons: 4 }));
  element.emit("pointerup", pointerEvent("pointerup", { button: 1 }));
  element.emit("pointermove", pointerEvent("pointermove", { buttons: 4 }));
  assert.deepEqual(
    graphCanvas.events.map((event) => event.type),
    ["wheel", "pointerdown", "pointermove", "pointerup"],
  );
});

test("both MiniMax H3 DOM panels enable canvas navigation", async () => {
  for (const filename of ["FL_MiniMaxH3BeatSamplerPreview.js", "FL_MiniMaxH3TemporalReshot.js"]) {
    const nodeSource = await readFile(new URL(filename, WEB_URL), "utf8");
    assert.match(nodeSource, /import \{ addCanvasNavigation \} from "\.\/canvas_navigation\.js";/);
    assert.match(nodeSource, /addCanvasNavigation\(container, app\.canvas\);/);
  }
});
