"""FL-owned VDN-H3 model patch using the official non-turbo checkpoint weights."""
import logging

from comfy_api.latest import io

from ._vdn.adapter_apply import apply_adapter
from ._vdn.adapters import convert_adapter
from ._vdn.branch import LinearBranch
from ._vdn.checkpoint import ensure_checkpoint, load_checkpoint
from ._vdn.hybrid import VDNState, apply_vdn


_log = logging.getLogger("comfy.fl_minimax_h3_vdn")


def apply_fl_vdn(model, strength):
    path = ensure_checkpoint()
    config, branch_weights, adapter_state, adapter_config = load_checkpoint(path)

    diffusion_model = model.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "blocks", None)
    if not blocks or not hasattr(getattr(blocks[0], "attn", None), "qkv_proj"):
        raise RuntimeError(
            "FL MiniMax H3 VDN requires a native ComfyUI MiniMax H3 diffusion model."
        )
    if len(blocks) != len(branch_weights):
        raise RuntimeError(
            f"FL MiniMax H3 VDN has {len(branch_weights)} checkpoint blocks, but the "
            f"loaded model has {len(blocks)}."
        )

    for key, patch in model.object_patches.items():
        if not key.endswith(".attn.forward"):
            continue
        if getattr(patch, "_vdn_forward", False):
            raise RuntimeError("The connected model already has a VDN attention patch.")
        _log.warning(
            "FL MiniMax H3 VDN is replacing the existing attention patch %s.",
            key,
        )

    attention = blocks[0].attn
    heads = attention.heads
    head_dim = attention.head_dim
    hidden = diffusion_model.hidden_size
    linear_dim = config["linear_head_dim"]
    expected = {
        "to_out_linear.weight": (hidden, heads * linear_dim),
        "beta_proj.weight": (heads, hidden),
        "alpha.A_log": (heads,),
        "alpha.dt_bias": (heads * linear_dim,),
        "alpha.down.weight": (linear_dim, hidden),
        "alpha.up.weight": (heads * linear_dim, linear_dim),
        "output_gate.down.weight": (linear_dim, hidden),
        "output_gate.up.weight": (heads * linear_dim, linear_dim),
        "output_gate.up.bias": (heads * linear_dim,),
        "softmax_gate.up.weight": (heads, hidden),
        "softmax_gate.up.bias": (heads,),
        "norm.weight": (linear_dim,),
    }
    for key, shape in expected.items():
        tensor = branch_weights[0].get(key)
        if tensor is None or tuple(tensor.shape) != shape:
            actual = None if tensor is None else tuple(tensor.shape)
            raise RuntimeError(
                f"FL MiniMax H3 VDN checkpoint tensor {key} has shape {actual}; "
                f"expected {shape}."
            )

    branches = [
        LinearBranch(
            weights,
            heads,
            head_dim,
            delta_rule=config["delta_rule"],
            bridge=config["bridge"],
            a_fp32=config["a_fp32"],
            short_conv=config["short_conv"],
            enable_text_state=config["enable_text_state"],
        )
        for weights in branch_weights
    ]
    state = VDNState(path.name, config, branches, heads, head_dim)
    patched = model.clone()
    apply_vdn(patched, state)
    applied = apply_adapter(
        patched,
        convert_adapter(adapter_state, adapter_config),
        strength,
    )
    _log.info(
        "FL MiniMax H3 VDN applied %s on %d blocks with %d Stage-B adapters; "
        "turbo adapter disabled by design.",
        path.name,
        len(branches),
        applied,
    )
    return patched


class FL_MiniMaxH3VDN(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3VDN",
            display_name="FL MiniMax H3 VDN",
            category="FL/MiniMax H3/Model",
            description=(
                "Applies the official VDN-H3 linear-attention branch and required Stage-B "
                "adapter. Missing Stage-DMD-250 files download automatically on first use; "
                "the turbo adapter is never downloaded or applied."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Strength of the required Stage-B VDN adapter.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="Cloned MiniMax H3 model with FL-owned VDN attention applied.",
                )
            ],
        )

    @classmethod
    def execute(cls, model, strength):
        return io.NodeOutput(apply_fl_vdn(model, strength))
