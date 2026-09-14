import math
import re

import comfy.lora
import comfy.lora_convert
import comfy.utils
import folder_paths
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy_api.latest import io


def parse_block_overrides(text, block_count, refiner_count):
    overrides = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"(blocks|refiner)\.(\d+)(?:-(\d+))?\s*=\s*(\S+)", line)
        if match is None:
            raise ValueError(f"Block override line {line_number}: use blocks.0-9=0.5 or refiner.0=1.")
        group, first, last, value = match.groups()
        first, last = int(first), int(last or first)
        count = block_count if group == "blocks" else refiner_count
        if not 0 <= first <= last < count:
            raise ValueError(f"Block override line {line_number}: {group} indices must be between 0 and {count - 1}.")
        try:
            strength = float(value)
        except ValueError:
            raise ValueError(f"Block override line {line_number}: strength must be a number.") from None
        if not math.isfinite(strength) or not -10 <= strength <= 10:
            raise ValueError(f"Block override line {line_number}: strength must be finite and between -10 and 10.")
        for index in range(first, last + 1):
            overrides[(group, index)] = strength
    return overrides


def patch_group(key):
    name = key if isinstance(key, str) else key[0]
    match = re.match(r"diffusion_model\.(blocks|token_refiner\.blocks)\.(\d+)\.", name)
    if match:
        return ("blocks" if match[1] == "blocks" else "refiner", int(match[2]))
    return ("other", None)


class FL_MiniMaxH3LoraBlockLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3LoraBlockLoader",
            display_name="FL MiniMax H3 LoRA Block Loader",
            category="FL/MiniMax H3/Model",
            description="Load one H3 LoRA with per-block strengths. Only this adapter is scaled; existing model patches are preserved. All multipliers at 1 match normal model-only LoRA loading.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength_model", default=1.0, min=-10.0, max=10.0, step=0.05),
                io.Float.Input("blocks_strength", default=1.0, min=-10.0, max=10.0, step=0.05, tooltip="Multiplier for main transformer blocks."),
                io.Float.Input("refiner_strength", default=1.0, min=-10.0, max=10.0, step=0.05, tooltip="Multiplier for token-refiner blocks; not the external Qwen encoder."),
                io.Float.Input("other_strength", default=1.0, min=-10.0, max=10.0, step=0.05, tooltip="Multiplier for adapter targets outside both stacks: projections, time embedding and final layer."),
                io.String.Input("block_overrides", default="", multiline=True, tooltip="One rule per line: blocks.0-9=0.5 or refiner.1=0. Zero-based, inclusive ranges. Replaces the group multiplier; overall strength still applies. Later rules win. # comments allowed."),
            ],
            outputs=[io.Model.Output(display_name="model"), io.String.Output(display_name="block_report")],
        )

    @classmethod
    def execute(cls, model, lora_name, strength_model, blocks_strength, refiner_strength, other_strength, block_overrides):
        diffusion = model.get_model_object("diffusion_model")
        if not isinstance(diffusion, MiniMaxH3Model):
            raise ValueError("FL H3 LoRA Block Loader requires a native MiniMax H3 model.")
        for value in (strength_model, blocks_strength, refiner_strength, other_strength):
            if not math.isfinite(value) or not -10 <= value <= 10:
                raise ValueError("LoRA strengths must be finite and between -10 and 10.")
        counts = {"blocks": len(diffusion.blocks), "refiner": len(diffusion.token_refiner.blocks)}
        overrides = parse_block_overrides(block_overrides, counts["blocks"], counts["refiner"])
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        result = model.clone()
        if strength_model == 0:
            return io.NodeOutput(result, "Bypassed: overall strength is 0. No adapter weights loaded.")

        weights, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        weights = comfy.lora_convert.convert_lora(weights)
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        patches = comfy.lora.load_lora(weights, key_map)
        if not patches:
            raise ValueError("No compatible H3 adapter weights found. Choose an H3 LoRA supported by ComfyUI's standard loader.")
        if metadata:
            result.set_attachments("lora_metadata", metadata)

        defaults = {"blocks": blocks_strength, "refiner": refiner_strength, "other": other_strength}
        batches = {}
        matched = {}
        for key, patch in patches.items():
            group = patch_group(key)
            multiplier = overrides.get(group, defaults[group[0]])
            strength = strength_model * multiplier
            matched[group] = matched.get(group, 0) + 1
            if strength != 0:
                batches.setdefault(strength, {})[key] = patch
        applied = 0
        for strength, batch in batches.items():
            accepted = result.add_patches(batch, strength)
            if len(accepted) != len(batch):
                raise ValueError("Some H3 LoRA targets could not be patched. Check that the adapter matches the loaded model.")
            applied += len(accepted)

        report = [f"{lora_name}: {applied}/{len(patches)} matched targets enabled; overall strength {strength_model:g}.",
                  "Effective strength = overall × block multiplier. Unmatched checkpoint keys are reported by ComfyUI."]
        groups = [(group, index) for group, count in counts.items() for index in range(count)] + [("other", None)]
        for group in groups:
            multiplier = overrides.get(group, defaults[group[0]])
            label = group[0] if group[1] is None else f"{group[0]}.{group[1]}"
            report.append(f"{label}: {strength_model * multiplier:g}; {matched.get(group, 0)} adapter targets")
        return io.NodeOutput(result, "\n".join(report))
