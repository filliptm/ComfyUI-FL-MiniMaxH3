import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F


ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT.parents[1]))


def load_module(relative_path):
    path = pathlib.Path(relative_path)
    module_name = "fl_minimax_h3_vdn_tests." + ".".join(path.with_suffix("").parts)
    parts = module_name.split(".")
    for index in range(1, len(parts)):
        package_name = ".".join(parts[:index])
        if package_name in sys.modules:
            continue
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT.joinpath(*path.parts[:max(0, index - 1)]))]
        sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


checkpoint = load_module("nodes/_vdn/checkpoint.py")
window = load_module("nodes/_vdn/window.py")
hybrid = load_module("nodes/_vdn/hybrid.py")
node = load_module("nodes/FL_MiniMaxH3VDN.py")


class FakeState:
    def __init__(self, heads, hidden):
        self.layout = SimpleNamespace(full_cover=True)
        self.branches = [object()]
        self.cfg = {"linear_enabled": True, "enable_softmax_gate": True}
        self.weights = {
            "softmax_gate.up.weight": torch.zeros(heads, hidden),
            "softmax_gate.up.bias": torch.zeros(heads),
        }

    def weights_on(self, index, device, dtype):
        return self.weights


class VDNDownloadTests(unittest.TestCase):
    def test_download_manifest_excludes_turbo_adapter(self):
        self.assertTrue(any("adapters/default/" in path for path in checkpoint.REQUIRED_FILES))
        self.assertFalse(any("turbo" in path.lower() for path in checkpoint.REQUIRED_FILES))

    def test_missing_checkpoint_downloads_only_the_allowlisted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            def create_files(**kwargs):
                for relative in kwargs["allow_patterns"]:
                    path = pathlib.Path(directory) / "vdn" / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()

            with mock.patch.object(checkpoint.folder_paths, "models_dir", directory), mock.patch.object(
                checkpoint, "snapshot_download", side_effect=create_files
            ) as download:
                path = checkpoint.ensure_checkpoint()

            self.assertEqual(path, pathlib.Path(directory) / "vdn" / checkpoint.STAGE)
            kwargs = download.call_args.kwargs
            self.assertEqual(kwargs["repo_id"], checkpoint.REPO_ID)
            self.assertEqual(kwargs["revision"], checkpoint.REVISION)
            self.assertEqual(tuple(kwargs["allow_patterns"]), checkpoint.REQUIRED_FILES)


class VDNMathTests(unittest.TestCase):
    def test_grouped_window_matches_direct_attention(self):
        torch.manual_seed(0)
        frames, tokens, heads, head_dim = 4, 3, 2, 4
        video_start = 2
        video_end = video_start + frames * tokens
        sequence = video_end + 1
        query = torch.randn(sequence, heads, head_dim)
        key = torch.randn(sequence, heads, head_dim)
        value = torch.randn(sequence, heads, head_dim)
        bounds = window.window_bounds(frames, 1)

        output = window.window_softmax_grouped(
            query,
            key,
            value,
            video_start,
            video_end,
            frames,
            tokens,
            bounds,
            head_dim ** -0.5,
        )
        self.assertEqual(output.shape, query.shape)
        global_rows = torch.tensor([0, 1, video_end])
        expected = F.scaled_dot_product_attention(
            query[global_rows].permute(1, 0, 2).unsqueeze(0),
            key.permute(1, 0, 2).unsqueeze(0),
            value.permute(1, 0, 2).unsqueeze(0),
            scale=head_dim ** -0.5,
        ).squeeze(0).permute(1, 0, 2)
        self.assertTrue(torch.allclose(output[global_rows], expected))

    def test_full_cover_dense_output_is_reshaped_before_head_gate(self):
        tokens, heads, head_dim = 7, 2, 3
        hidden = heads * head_dim
        attention = SimpleNamespace(
            heads=heads,
            head_dim=head_dim,
            qkv_proj=torch.nn.Linear(hidden, hidden * 3, bias=False),
            q_norm=torch.nn.Identity(),
            k_norm=torch.nn.Identity(),
            out_proj=torch.nn.Identity(),
        )
        forward = hybrid.make_vdn_forward(attention, FakeState(heads, hidden), 0)

        with mock.patch.object(
            hybrid,
            "optimized_attention",
            return_value=torch.ones(1, tokens, hidden),
        ):
            output = forward(torch.randn(tokens, hidden))

        self.assertEqual(output.shape, (tokens, hidden))
        self.assertTrue(torch.equal(output, torch.full_like(output, 0.5)))


class VDNNodeTests(unittest.TestCase):
    def test_schema_has_no_turbo_controls(self):
        schema = node.FL_MiniMaxH3VDN.define_schema()
        self.assertEqual(schema.node_id, "FL_MiniMaxH3VDN")
        self.assertEqual([value.id for value in schema.inputs], ["model", "strength"])


if __name__ == "__main__":
    unittest.main()
