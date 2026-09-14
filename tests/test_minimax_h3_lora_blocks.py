import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import comfy.lora
import comfy.model_patcher
import comfy.ops
import comfy.sd


spec = importlib.util.spec_from_file_location("fl_h3_lora_blocks_test", Path(__file__).parents[1] / "nodes" / "FL_MiniMaxH3LoraBlockLoader.py")
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


class LoraBlockTests(unittest.TestCase):
    def setUp(self):
        base = torch.nn.Module()
        base.diffusion_model = loader.MiniMaxH3Model(hidden_size=8, num_layers=3, token_refiner_num_layers=2,
            num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=8, text_dim=8,
            timestep_input_dim=8, time_embed_hidden_size=8, time_embed_dim=8,
            dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
        base.model_config = SimpleNamespace(unet_config={})
        self.model = comfy.model_patcher.ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))
        self.names = ["blocks.0.mlp.fc1", "blocks.1.mlp.fc1", "blocks.2.mlp.fc1",
                      "token_refiner.blocks.0.mlp.fc1", "token_refiner.blocks.1.mlp.fc1", "condition_proj"]
        self.weights = {}
        for name in self.names:
            key = "diffusion_model." + name
            shape = base.state_dict()[key + ".weight"].shape
            self.weights[key + ".lora_up.weight"] = torch.ones(shape[0], 2)
            self.weights[key + ".lora_down.weight"] = torch.ones(2, shape[1])
        self.args = dict(model=self.model, lora_name="fixture.safetensors", strength_model=1.0,
                         blocks_strength=1.0, refiner_strength=1.0, other_strength=1.0, block_overrides="")
        mock = patch.object(loader.folder_paths, "get_full_path_or_raise", return_value="fixture.safetensors")
        mock.start()
        self.addCleanup(mock.stop)
        mock = patch.object(loader.comfy.utils, "load_torch_file", return_value=(self.weights, {"fixture": "metadata"}))
        self.read = mock.start()
        self.addCleanup(mock.stop)

    def execute(self, **changes):
        return loader.FL_MiniMaxH3LoraBlockLoader.execute(**{**self.args, **changes}).result

    def test_defaults_numerically_match_standard_lora_loader(self):
        result, report = self.execute()
        standard, _ = comfy.sd.load_lora_for_models(self.model, None, self.weights, 1.0, 0.0)
        self.assertEqual(set(result.patches), set(standard.patches))
        for key in standard.patches:
            weight = torch.zeros_like(self.model.model.state_dict()[key])
            actual = comfy.lora.calculate_weight(result.patches[key], weight.clone(), key)
            expected = comfy.lora.calculate_weight(standard.patches[key], weight.clone(), key)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIn("6/6 matched targets enabled", report)
        self.assertEqual(self.model.patches, {})
        self.assertEqual(result.get_attachment("lora_metadata"), {"fixture": "metadata"})

    def test_group_scaling_overrides_zero_and_negative(self):
        result, report = self.execute(strength_model=0.5, blocks_strength=0.8, refiner_strength=0.4,
            other_strength=-1.0, block_overrides="blocks.0-1=0\nblocks.1=-0.5\nrefiner.1=2 # replace group multiplier")
        expected = {self.names[1]: -0.25, self.names[2]: 0.4, self.names[3]: 0.2, self.names[4]: 1.0, self.names[5]: -0.5}
        self.assertEqual(len(result.patches), 5)
        for name, strength in expected.items():
            key = f"diffusion_model.{name}.weight"
            self.assertEqual(result.patches[key][0][0], strength)
            weight = torch.zeros_like(self.model.model.state_dict()[key])
            actual = comfy.lora.calculate_weight(result.patches[key], weight, key)
            torch.testing.assert_close(actual, torch.full_like(actual, strength * 2))
        self.assertIn("blocks.0: 0; 1 adapter targets", report)

    def test_upstream_adapter_is_not_rescaled(self):
        key = "diffusion_model.blocks.0.mlp.fc1.weight"
        upstream = ("diff", (torch.ones_like(self.model.model.state_dict()[key]),))
        self.model.add_patches({key: upstream}, 0.75)
        result, _ = self.execute(block_overrides="blocks.0=0")
        self.assertEqual(result.patches[key][0][0], 0.75)
        self.assertEqual(len(self.model.patches), 1)
        self.assertEqual(len(self.model.patches[key]), 1)

    def test_zero_overall_does_not_read_weights(self):
        result, report = self.execute(strength_model=0)
        self.read.assert_not_called()
        self.assertIsNot(result, self.model)
        self.assertIn("Bypassed", report)

    def test_invalid_rules_fail_before_loading(self):
        for rule in ("blocks.3=1", "refiner.2=1", "blocks.2-1=1", "blocks.0=nan", "blocks.0=inf", "blocks.0=11", "blocks.0=no", "0:1"):
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                self.execute(block_overrides=rule)
        self.read.assert_not_called()

    def test_empty_or_incompatible_adapter_fails(self):
        self.read.return_value = ({}, None)
        with self.assertRaisesRegex(ValueError, "No compatible H3"):
            self.execute()

    def test_wrong_model_and_nonfinite_strength_fail(self):
        with self.assertRaisesRegex(ValueError, "native MiniMax H3"):
            self.execute(model=SimpleNamespace(get_model_object=lambda _: torch.nn.Linear(2, 2)))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.execute(blocks_strength=float("nan"))

    def test_offset_patch_keys_keep_their_target_group(self):
        self.assertEqual(loader.patch_group(("diffusion_model.blocks.2.attn.qkv_proj.weight", (0, 1, 2))), ("blocks", 2))


if __name__ == "__main__":
    unittest.main()
