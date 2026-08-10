import importlib.util
import math
import pathlib
import sys
import types
import unittest
from unittest import mock

import torch

import comfy.nested_tensor


ROOT = pathlib.Path(__file__).parents[1]


def load_module(relative_path):
    path = pathlib.Path(relative_path)
    module_name = "fl_minimax_h3_latent_upscale_tests." + ".".join(path.with_suffix("").parts)
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


upscale = load_module("nodes/FL_MiniMaxH3LatentUpscale.py")


def h3_latent(video_shape=(1, 24, 3, 32, 18), audio_shape=(1, 32, 2, 40)):
    video = torch.arange(math.prod(video_shape), dtype=torch.float32).reshape(video_shape)
    audio = torch.arange(math.prod(audio_shape), dtype=torch.float32).reshape(audio_shape)
    noise_mask = comfy.nested_tensor.NestedTensor((torch.ones_like(video), torch.zeros_like(audio)))
    metadata = {"version": 1, "index": 2, "motion_context": {"video_steps": 2}}
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": noise_mask,
        "fl_h3_shot": metadata,
        "custom": {"preserved": True},
    }


class MiniMaxH3LatentUpscaleTests(unittest.TestCase):
    def test_schema_exposes_one_native_latent_transform(self):
        schema = upscale.FL_MiniMaxH3LatentUpscale.define_schema()

        self.assertEqual(schema.node_id, "FL_MiniMaxH3LatentUpscale")
        self.assertEqual(schema.category, "FL/MiniMax H3/Latent")
        self.assertEqual([value.io_type for value in schema.inputs], ["LATENT", "INT", "COMBO"])
        self.assertEqual(len(schema.outputs), 1)
        self.assertEqual(schema.outputs[0].io_type, "LATENT")

    def test_upscales_video_spatially_and_preserves_audio_time_and_metadata(self):
        latent = h3_latent()
        source_video, source_audio = latent["samples"].unbind()

        output = upscale.upscale_h3_latent(latent, 896, "bilinear")
        video, audio = output["samples"].unbind()

        self.assertEqual(video.shape, (1, 24, 3, 56, 32))
        self.assertEqual(video.dtype, source_video.dtype)
        self.assertEqual(video.device, source_video.device)
        self.assertEqual(source_video.shape, (1, 24, 3, 32, 18))
        self.assertEqual(audio.data_ptr(), source_audio.data_ptr())
        self.assertIs(output["noise_mask"], latent["noise_mask"])
        self.assertIs(output["fl_h3_shot"], latent["fl_h3_shot"])
        self.assertIs(output["custom"], latent["custom"])
        self.assertIsNot(output, latent)

    def test_forwards_latent_dimensions_and_method_to_common_upscale(self):
        latent = h3_latent(video_shape=(1, 24, 2, 32, 18))
        video, audio = latent["samples"].unbind()
        resized = torch.zeros((1, 24, 2, 64, 36))

        with mock.patch.object(upscale.comfy.utils, "common_upscale", return_value=resized) as common_upscale:
            output = upscale.upscale_h3_latent(latent, 1024, "bislerp")

        common_upscale.assert_called_once_with(video, 36, 64, "bislerp", "disabled")
        output_video, output_audio = output["samples"].unbind()
        self.assertIs(output_video, resized)
        self.assertIs(output_audio, audio)

    def test_matching_target_is_a_no_op_without_returning_the_input_dictionary(self):
        latent = h3_latent()
        source_video, source_audio = latent["samples"].unbind()

        with mock.patch.object(upscale.comfy.utils, "common_upscale") as common_upscale:
            output = upscale.FL_MiniMaxH3LatentUpscale.execute(
                latent,
                512,
                "bislerp",
            ).result[0]

        common_upscale.assert_not_called()
        video, audio = output["samples"].unbind()
        self.assertIs(video, source_video)
        self.assertIs(audio, source_audio)
        self.assertIsNot(output, latent)

    def test_target_must_be_aligned_and_must_not_downscale(self):
        latent = h3_latent()

        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            upscale.upscale_h3_latent(latent, 900, "bislerp")
        with self.assertRaisesRegex(ValueError, "smaller than"):
            upscale.upscale_h3_latent(latent, 256, "bislerp")

    def test_rejects_non_h3_latent_structures(self):
        with self.assertRaisesRegex(TypeError, "latent dictionary"):
            upscale.upscale_h3_latent(torch.zeros((1, 4, 8, 8)), 1024, "bislerp")
        with self.assertRaisesRegex(TypeError, "nested H3"):
            upscale.upscale_h3_latent({"samples": torch.zeros((1, 24, 2, 4, 4))}, 1024, "bislerp")
        with self.assertRaisesRegex(ValueError, "exactly one video and one audio"):
            upscale.upscale_h3_latent({
                "samples": comfy.nested_tensor.NestedTensor((torch.zeros((1, 24, 2, 4, 4)),)),
            }, 1024, "bislerp")
        with self.assertRaisesRegex(ValueError, "video latent shape"):
            upscale.upscale_h3_latent(h3_latent(video_shape=(1, 16, 2, 4, 4)), 1024, "bislerp")
        with self.assertRaisesRegex(ValueError, "audio latent shape"):
            upscale.upscale_h3_latent(h3_latent(audio_shape=(1, 16, 2, 40)), 1024, "bislerp")


if __name__ == "__main__":
    unittest.main()
