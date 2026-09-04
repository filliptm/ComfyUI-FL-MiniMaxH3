"""Apply the required Stage-B VDN adapter without the optional turbo adapter."""
import torch

import comfy.lora
import comfy.patcher_extension
import comfy.weight_adapter


class _FrugalLoRA(comfy.weight_adapter.LoRAAdapter):
    def bypass_forward(self, org_forward, x, *args, **kwargs):
        base_out = org_forward(x, *args, **kwargs)
        if getattr(self, "is_conv", False):
            return super().bypass_forward(org_forward, x, *args, **kwargs)
        up, down, alpha = self.weights[0], self.weights[1], self.weights[2]
        rank = down.shape[0]
        scale = (alpha / rank if alpha is not None else 1.0) * self.multiplier
        down = down.to(dtype=x.dtype)
        up = up.to(dtype=x.dtype)
        return base_out.add_(
            torch.nn.functional.linear(torch.nn.functional.linear(x, down), up),
            alpha=scale,
        )


def apply_adapter(model, converted, strength):
    modules = sorted(converted)
    adapter_state = {}
    for path, (down, up, scale) in converted.items():
        adapter_state[path + ".lora_A.weight"] = down.contiguous()
        adapter_state[path + ".lora_B.weight"] = up.contiguous()
        adapter_state[path + ".alpha"] = torch.tensor(scale * down.shape[0])

    key_map = {module: f"diffusion_model.{module}.weight" for module in modules}
    loaded = comfy.lora.load_lora(adapter_state, key_map, log_missing=False)
    state_keys = set(model.model.state_dict())
    manager = comfy.weight_adapter.BypassInjectionManager()
    applied = 0
    for module in modules:
        key = key_map[module]
        adapter = loaded.get(key)
        if adapter is None or key not in state_keys:
            continue
        if isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
            adapter = _FrugalLoRA(adapter.loaded_keys, adapter.weights)
        if not isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):
            continue
        manager.add_adapter(key, adapter, strength=strength)
        applied += 1

    if not applied:
        raise RuntimeError(
            "FL MiniMax H3 VDN could not map the Stage-B adapter onto the loaded model."
        )
    manager.create_injections(model.model)
    hooks = list(manager.hooks)

    def inject(model_patcher):
        for hook in hooks:
            hook.inject()

    def eject(model_patcher):
        for hook in reversed(hooks):
            hook.eject()

    model.set_injections(
        "fl_minimax_h3_vdn_lora",
        [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)],
    )
    return applied
