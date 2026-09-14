import { app } from "../../../scripts/app.js";

const DEFAULTS = {
  context_overlap: 17,
  sampler_name: "res_multistep",
  scheduler: "simple",
  max_hold: 4,
  audio_strength: 0.5,
  expand_to_end: true,
  upscale_method: "lanczos",
};
const hiddenWidgets = new WeakMap();

export function advancedControls(node) {
  const value = (name) => node.widgets.find((widget) => widget.name === name)?.value;
  const linked = (name) => node.inputs?.some((input) => input.name === name && input.link != null);
  return Object.keys(DEFAULTS).filter((name) => {
    if (name === "max_hold") return linked("motion_coverage") || value("motion_coverage") === "uniform";
    if (name === "context_overlap") return linked("context_budget") || value("context_budget") > 0;
    if (name === "upscale_method") return linked("target_long_side") || value("target_long_side") > 0;
    if (name === "expand_to_end") return linked("motion_coverage") || value("motion_coverage") !== "off";
    return true;
  });
}

function refresh(node, button) {
  const expanded = Boolean(node.properties.motion_refine_advanced);
  const relevant = advancedControls(node);
  for (const widget of node.widgets) {
    if (!(widget.name in DEFAULTS)) continue;
    const visible = expanded && relevant.includes(widget.name);
    if (!visible && !hiddenWidgets.has(widget)) {
      hiddenWidgets.set(widget, { type: widget.type, computeSize: widget.computeSize, hidden: widget.hidden });
      widget.type = "converted-widget";
      widget.computeSize = () => [0, -4];
      widget.hidden = true;
    } else if (visible && hiddenWidgets.has(widget)) {
      Object.assign(widget, hiddenWidgets.get(widget));
      hiddenWidgets.delete(widget);
    }
  }
  const changed = node.widgets.filter((widget) => relevant.includes(widget.name) && widget.value !== DEFAULTS[widget.name]).length;
  button.name = `${expanded ? "Hide" : "Show"} advanced${changed ? ` (${changed} changed)` : ""}`;
  node.setSize([node.size[0], node.computeSize()[1]]);
  node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "FL.MiniMaxH3.MotionRefine",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "FL_MiniMaxH3MotionRefine") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      const button = this.addWidget("button", "Show advanced", null, () => {
        this.properties.motion_refine_advanced = !this.properties.motion_refine_advanced;
        refresh(this, button);
      }, { serialize: false });
      for (const widget of this.widgets) {
        if (widget === button) continue;
        const callback = widget.callback;
        widget.callback = (...args) => {
          const value = callback?.apply(widget, args);
          refresh(this, button);
          return value;
        };
      }
      const configured = this.onConfigure;
      this.onConfigure = function () {
        const value = configured?.apply(this, arguments);
        refresh(this, button);
        return value;
      };
      refresh(this, button);
      return result;
    };
  },
});
