import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

import comfy.nested_tensor


ROOT = pathlib.Path(__file__).parents[1]


def load_module(relative_path):
    path = pathlib.Path(relative_path)
    module_name = "fl_minimax_h3_neural_upscale_tests." + ".".join(path.with_suffix("").parts)
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


upscale = load_module("nodes/FL_MiniMaxH3NeuralLatentUpscale.py")


def latent():
    video = torch.randn((1, 24, 3, 4, 6))
    audio = torch.randn((1, 32, 2, 20))
    video_mask = torch.ones_like(video)
    audio_mask = torch.zeros_like(audio)
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
        "metadata": {"preserved": True},
    }


def zero_parameters(model):
    for parameter in model.parameters():
        parameter.data.zero_()


class NeuralLatentUpscaleArchitectureTests(unittest.TestCase):
    def test_2d_backbone_preserves_time_and_changes_only_space(self):
        model = upscale._VideoLatentResizer2D(32, 8, ["residual"], ["residual"], [])
        zero_parameters(model)

        output = model(torch.randn((1, 24, 3, 4, 6)), 2.0, (8, 12))

        self.assertEqual(output.shape, (1, 24, 3, 8, 12))

    def test_3d_backbone_chunking_preserves_time(self):
        model = upscale._LatentResizer3D(
            32,
            8,
            ["residual", "temporal"],
            ["residual", "temporal"],
            [3],
            [3],
        )
        zero_parameters(model)

        output = model(torch.randn((1, 24, 5, 3, 4)), 2.0, (5, 6, 8), 2)

        self.assertEqual(output.shape, (1, 24, 5, 6, 8))

    def test_checkpoint_loader_detects_2d_and_uses_strict_layout(self):
        source = upscale._VideoLatentResizer2D(32, 8, ["residual"], ["residual"], [])
        zero_parameters(source)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = pathlib.Path(directory) / "test.pth"
            torch.save({"model": source.state_dict()}, checkpoint)
            with (
                mock.patch.object(upscale.folder_paths, "get_full_path_or_raise", return_value=str(checkpoint)),
                mock.patch.object(upscale.comfy.model_management, "get_torch_device", return_value=torch.device("cpu")),
                mock.patch.object(upscale.comfy.model_management, "vae_offload_device", return_value=torch.device("cpu")),
            ):
                loaded = upscale.load_h3_latent_upscale_model("test.pth", "fp32")

        self.assertEqual(loaded.architecture, "2d")
        self.assertEqual(set(loaded.model.state_dict()), set(source.state_dict()))

    def test_rejects_non_h3_checkpoint_before_shape_access(self):
        with self.assertRaisesRegex(ValueError, "not a supported MiniMax H3"):
            upscale._build_model({"other.weight": torch.empty((2, 2))})


class NeuralLatentUpscaleNodeTests(unittest.TestCase):
    def test_2d_node_preserves_audio_metadata_and_resizes_nested_mask(self):
        source = latent()
        source_video, source_audio = source["samples"].unbind()
        model = upscale._H3LatentUpscaleModel.__new__(upscale._H3LatentUpscaleModel)
        model.architecture = "2d"
        model.upscale = mock.Mock(return_value=torch.zeros((1, 24, 3, 8, 12)))

        output = upscale.FL_MiniMaxH3NeuralLatentUpscale2D.execute(model, source, 2.0).result[0]

        video, audio = output["samples"].unbind()
        video_mask, audio_mask = output["noise_mask"].unbind()
        self.assertEqual(video.shape, (1, 24, 3, 8, 12))
        self.assertIs(audio, source_audio)
        self.assertEqual(video_mask.shape, video.shape)
        self.assertIs(audio_mask, source["noise_mask"].unbind()[1])
        self.assertIs(output["metadata"], source["metadata"])
        self.assertEqual(source_video.shape, (1, 24, 3, 4, 6))

    def test_3d_dimension_mode_aligns_pixels_and_preserves_audio(self):
        source = latent()
        source_audio = source["samples"].unbind()[1]
        model = upscale._H3LatentUpscaleModel.__new__(upscale._H3LatentUpscaleModel)
        model.architecture = "3d"
        model.upscale = mock.Mock(return_value=torch.zeros((1, 24, 3, 8, 12)))
        mode = {"mode": "target dimensions", "width": 192, "height": 128}

        output = upscale.FL_MiniMaxH3NeuralLatentUpscale3D.execute(model, source, mode, 32, 16).result[0]

        video, audio = output["samples"].unbind()
        self.assertEqual(video.shape, (1, 24, 3, 8, 12))
        self.assertIs(audio, source_audio)
        model.upscale.assert_called_once_with(source["samples"].unbind()[0], 2.0, (3, 8, 12), 16)

    def test_nodes_reject_the_wrong_checkpoint_architecture(self):
        model = upscale._H3LatentUpscaleModel.__new__(upscale._H3LatentUpscaleModel)
        model.architecture = "3d"
        with self.assertRaisesRegex(ValueError, "requires a 2D"):
            upscale.FL_MiniMaxH3NeuralLatentUpscale2D.execute(model, latent(), 2.0)

    def test_schemas_use_native_latent_and_upscale_model_types(self):
        loader = upscale.FL_MiniMaxH3LatentUpscaleModelLoader.define_schema()
        node_2d = upscale.FL_MiniMaxH3NeuralLatentUpscale2D.define_schema()
        node_3d = upscale.FL_MiniMaxH3NeuralLatentUpscale3D.define_schema()

        self.assertEqual(loader.outputs[0].io_type, "LATENT_UPSCALE_MODEL")
        self.assertEqual([value.io_type for value in node_2d.inputs], ["LATENT_UPSCALE_MODEL", "LATENT", "FLOAT"])
        self.assertEqual(node_3d.inputs[0].io_type, "LATENT_UPSCALE_MODEL")
        self.assertEqual(node_3d.inputs[1].io_type, "LATENT")


if __name__ == "__main__":
    unittest.main()
