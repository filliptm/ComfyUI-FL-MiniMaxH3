import assert from "node:assert/strict";
import fs from "node:fs/promises";
import test from "node:test";

const source = await fs.readFile(new URL("../web/FL_MiniMaxH3MotionRefine.js", import.meta.url), "utf8");
const code = source.replace('import { app } from "../../../scripts/app.js";', "const app = { registerExtension(extension) { globalThis.motionRefineExtension = extension; } };");
const { advancedControls } = await import(`data:text/javascript;base64,${Buffer.from(code).toString("base64")}`);

class Node {
  constructor() {
    this.properties = {};
    this.inputs = [];
    this.size = [400, 500];
    this.widgets = Object.entries({ target_long_side: 0, strength: 0.5, motion_coverage: "balanced", steps: 25,
      context_budget: 0, seed: 1, context_overlap: 17, sampler_name: "res_multistep", scheduler: "simple",
      max_hold: 4, audio_strength: 0.5, expand_to_end: true, upscale_method: "lanczos" })
      .map(([name, value]) => ({ name, value, type: "number", computeSize: () => [100, 20] }));
    this.graph = { setDirtyCanvas() {} };
    this.onNodeCreated();
  }
  addWidget(type, name, value, callback, options) {
    const widget = { type, name, value, callback, options };
    this.widgets.push(widget);
    return widget;
  }
  computeSize() { return [400, 100 + 20 * this.widgets.filter((w) => !w.hidden).length]; }
  setSize(size) { this.size = size; }
}
await globalThis.motionRefineExtension.beforeRegisterNodeDef(Node, { name: "FL_MiniMaxH3MotionRefine" });

test("advanced controls collapse without changing serialized values", () => {
  const node = new Node();
  const audio = node.widgets.find((w) => w.name === "audio_strength");
  const button = node.widgets.at(-1);
  assert.equal(audio.hidden, true);
  assert.equal(audio.value, 0.5);
  assert.equal(button.options.serialize, false);
  button.callback();
  assert.equal(audio.type, "number");
  assert.equal(audio.hidden, undefined);
  audio.value = 0.7;
  audio.callback();
  button.callback();
  assert.equal(audio.value, 0.7);
  assert.equal(button.name, "Show advanced (1 changed)");
  node.properties.motion_refine_advanced = true;
  node.onConfigure();
  assert.equal(audio.type, "number");
});

test("only relevant options are exposed, including linked controls", () => {
  const node = new Node();
  assert(!advancedControls(node).includes("max_hold"));
  assert(!advancedControls(node).includes("context_overlap"));
  assert(!advancedControls(node).includes("upscale_method"));
  node.widgets.find((w) => w.name === "motion_coverage").value = "uniform";
  node.widgets.find((w) => w.name === "target_long_side").value = 1024;
  node.inputs.push({ name: "context_budget", link: 12 });
  assert(advancedControls(node).includes("max_hold"));
  assert(advancedControls(node).includes("context_overlap"));
  assert(advancedControls(node).includes("upscale_method"));
  node.widgets.find((w) => w.name === "motion_coverage").value = "off";
  assert(!advancedControls(node).includes("expand_to_end"));
});
