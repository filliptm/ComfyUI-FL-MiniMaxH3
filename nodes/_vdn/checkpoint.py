"""Download and load the official non-turbo VDN-H3 checkpoint artifacts."""
import json
import logging
from pathlib import Path

import comfy.utils
import folder_paths
from huggingface_hub import snapshot_download


_log = logging.getLogger("comfy.fl_minimax_h3_vdn")

REPO_ID = "OpenVDN/vdn-minimax-h3"
REVISION = "18be6bcc4ee72585eee322ba28b5ccac2cf85ef0"
STAGE = "stage-dmd-step-250"
REQUIRED_FILES = (
    f"{STAGE}/metadata.json",
    f"{STAGE}/model_spec.json",
    f"{STAGE}/linear_branch/config.json",
    f"{STAGE}/linear_branch/model.safetensors",
    f"{STAGE}/adapters/default/adapter_config.json",
    f"{STAGE}/adapters/default/adapter_model.safetensors",
)


def model_root():
    return Path(folder_paths.models_dir) / "vdn"


def checkpoint_path():
    return model_root() / STAGE


def ensure_checkpoint():
    path = checkpoint_path()
    root = model_root()
    if all((root / relative).is_file() for relative in REQUIRED_FILES):
        return path

    _log.info(
        "FL MiniMax H3 VDN: downloading the Stage-DMD-250 branch and default adapter "
        "to %s. The turbo adapter is excluded.",
        path,
    )
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=root,
        allow_patterns=list(REQUIRED_FILES),
    )
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "FL MiniMax H3 VDN download completed without required files: "
            + ", ".join(missing)
        )
    return path


def _read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _transform_config(model_spec):
    transforms = [
        transform
        for transform in model_spec.get("transforms", ())
        if transform.get("type") == "hybrid_attention"
    ]
    if len(transforms) != 1 or transforms[0].get("version") != 2:
        raise ValueError(
            "FL MiniMax H3 VDN requires one version-2 hybrid_attention transform."
        )
    config = transforms[0]["config"]
    linear = config["linear_attention"]
    softmax = config["softmax_attention"]
    short_conv = tuple(linear.get("short_conv", {}).get("targets", ()))
    if linear.get("delta_rule") != "vdn_solve" or linear.get("bridge") != "alpha":
        raise ValueError(
            "FL MiniMax H3 VDN only supports the released Stage-DMD-250 vdn_solve configuration."
        )
    if any(target not in ("q", "k", "v") for target in short_conv):
        raise ValueError(f"Unsupported VDN short-convolution targets: {short_conv}")
    if config.get("anchor_frames") not in ("none", "columns", "rows", "both"):
        raise ValueError(f"Unsupported VDN anchor mode: {config.get('anchor_frames')}")
    return {
        "enable_softmax_gate": bool(config.get("enable_softmax_gate", True)),
        "anchor_frames": config["anchor_frames"],
        "radius": int(softmax["radius"]),
        "chunk": int(softmax.get("chunk", 0)),
        "delta_rule": "vdn_solve",
        "bridge": "alpha",
        "a_fp32": bool(linear.get("a_fp32", True)),
        "linear_head_dim": int(linear["linear_head_dim"]),
        "short_conv": short_conv,
        "enable_text_state": bool(linear.get("enable_text_state", False)),
    }


def _load_branches(path):
    state = comfy.utils.load_torch_file(
        str(path / "linear_branch" / "model.safetensors"),
        safe_load=True,
    )
    block_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state
            if ".attn.to_out_linear.weight" in key
        }
    )
    if block_indices != list(range(len(block_indices))):
        raise ValueError("FL MiniMax H3 VDN checkpoint has non-contiguous transformer blocks.")

    branches = []
    required = {
        "to_out_linear.weight",
        "beta_proj.weight",
        "norm.weight",
        "alpha.A_log",
        "alpha.dt_bias",
        "alpha.down.weight",
        "alpha.up.weight",
        "output_gate.down.weight",
        "output_gate.up.weight",
        "output_gate.up.bias",
    }
    for index in block_indices:
        prefix = f"transformer_blocks.{index}.attn."
        weights = {}
        for key, tensor in state.items():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            if name.startswith("linear_attention."):
                name = name[len("linear_attention."):]
            weights[name] = tensor
        missing = required - set(weights)
        if missing:
            raise ValueError(
                f"FL MiniMax H3 VDN block {index} is missing tensors: {sorted(missing)}"
            )
        branches.append(weights)
    return branches


def load_checkpoint(path):
    model_spec = _read_json(path / "model_spec.json")
    if model_spec.get("base", {}).get("class_name") != "MiniMaxH3Transformer3DModel":
        raise ValueError("FL MiniMax H3 VDN checkpoint does not target MiniMax H3.")
    config = _transform_config(model_spec)
    branches = _load_branches(path)
    adapter_path = path / "adapters" / "default"
    adapter = comfy.utils.load_torch_file(
        str(adapter_path / "adapter_model.safetensors"),
        safe_load=True,
    )
    adapter_config = _read_json(adapter_path / "adapter_config.json")
    return config, branches, adapter, adapter_config
