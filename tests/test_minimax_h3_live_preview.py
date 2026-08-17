import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

import comfy.nested_tensor
from comfy_api.latest import io


ROOT = pathlib.Path(__file__).parents[1]
PACKAGE = "fl_minimax_h3_live_preview_tests"


def load_module(relative_path):
    path = pathlib.Path(relative_path)
    module_name = PACKAGE + "." + ".".join(path.with_suffix("").parts)
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


preview = load_module("nodes/_live_preview.py")
sampler = load_module("nodes/FL_MiniMaxH3BeatKSampler.py")


def nested_latent(index=0, total=1):
    video = torch.zeros((1, 24, 2, 4, 5))
    audio = torch.zeros((1, 32, 2, 8))
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "fl_h3_shot": {
            "version": 1,
            "index": index,
            "start_frame": index * 4,
            "end_frame": (index + 1) * 4,
            "authored_frames": 4,
            "render_frames": 6,
            "total_frames": total * 4,
            "fps": 24.0,
            "seed": index + 10,
            "motion_context": {
                "version": 1,
                "source_index": index - 1 if index else None,
                "video_frames": 0,
                "video_steps": 0,
                "audio_frames": 0,
                "trim_frames": 2,
            },
        },
    }


class PreviewVAE:
    def decode(self, video):
        frames = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1, 1)
        return frames.expand(1, 8, 40, 80, 3).clone()


class SavedVideo:
    def __init__(self):
        self.path = None
        self.options = None

    def save_to(self, path, **options):
        self.path = path
        self.options = options
        pathlib.Path(path).touch()


class LivePreviewHelperTests(unittest.TestCase):
    def test_preview_uses_only_authored_frames_and_downscales(self):
        saved_video = SavedVideo()
        with (
            tempfile.TemporaryDirectory() as temp_directory,
            mock.patch.object(preview.folder_paths, "get_temp_directory", return_value=temp_directory),
            mock.patch.object(
                preview.InputImpl,
                "VideoFromComponents",
                return_value=saved_video,
            ) as create_video,
        ):
            result = preview.create_preview(
                nested_latent(),
                PreviewVAE(),
                nested_latent()["fl_h3_shot"],
                "sample",
                0,
                40,
            )

        components = create_video.call_args.args[0]
        self.assertEqual(components.images.shape, (4, 20, 40, 3))
        self.assertTrue(torch.all(components.images[0] == 2))
        self.assertTrue(torch.all(components.images[-1] == 5))
        self.assertEqual(result["frame_count"], 4)
        self.assertEqual((result["width"], result["height"]), (40, 20))
        self.assertEqual(result["type"], "temp")
        self.assertEqual(result["subfolder"], preview.PREVIEW_SUBFOLDER)
        self.assertEqual(saved_video.options["crf"], 30)

    def test_reshot_preview_starts_at_the_selected_source_offset(self):
        latent = nested_latent()
        metadata = latent["fl_h3_shot"]
        metadata["authored_frames"] = 2
        metadata["reshot"] = {"selection_offset": 3}
        saved_video = SavedVideo()
        with (
            tempfile.TemporaryDirectory() as temp_directory,
            mock.patch.object(preview.folder_paths, "get_temp_directory", return_value=temp_directory),
            mock.patch.object(preview.InputImpl, "VideoFromComponents", return_value=saved_video) as create_video,
        ):
            preview.create_preview(latent, PreviewVAE(), metadata, "sample", 0, 80)

        images = create_video.call_args.args[0].images
        self.assertEqual(images.shape[0], 2)
        self.assertTrue(torch.all(images[0] == 3))
        self.assertTrue(torch.all(images[1] == 4))

    def test_preview_failure_is_reported_without_raising(self):
        with (
            mock.patch.object(preview, "create_preview", side_effect=RuntimeError("encoder failed")),
            mock.patch.object(preview, "send_preview_event") as emit,
            self.assertLogs(level="WARNING"),
        ):
            result = preview.publish_preview(
                nested_latent(),
                PreviewVAE(),
                nested_latent()["fl_h3_shot"],
                "12",
                "prompt",
                "sample",
                0,
                1,
                512,
            )

        self.assertIsNone(result)
        self.assertEqual(emit.call_args_list[0].args[3], "previewing")
        self.assertEqual(emit.call_args_list[1].args[3], "preview_error")
        self.assertEqual(emit.call_args_list[1].kwargs["error"], "encoder failed")


def beat_plan(count=2):
    shots = []
    for index in range(count):
        shots.append({
            "index": index,
            "start_frame": index * 4,
            "end_frame": (index + 1) * 4,
            "authored_frames": 4,
            "render_frames": 4,
            "conditioning": f"conditioning-{index}",
            "latent": {"samples": f"latent-{index}"},
        })
    return {
        "type": "minimax_h3_beat_shot_plan",
        "version": 1,
        "fps": 24.0,
        "total_frames": count * 4,
        "shots": shots,
    }


class SamplerPreviewIntegrationTests(unittest.TestCase):
    def test_schemas_expose_an_optional_disabled_preview_toggle(self):
        for node_class in (
            sampler.FL_MiniMaxH3BeatKSampler,
            sampler.FL_MiniMaxH3BeatUpscaleKSampler,
        ):
            schema = node_class.define_schema()
            live_preview = next(value for value in schema.inputs if value.id == "live_preview")
            self.assertEqual(live_preview.io_type, "BOOLEAN")
            self.assertTrue(live_preview.optional)
            self.assertFalse(live_preview.default)
            self.assertEqual(schema.hidden, [io.Hidden.unique_id])

    def test_beat_sampler_publishes_each_completed_render(self):
        plan = beat_plan()
        completed = [nested_latent(0, 2), nested_latent(1, 2)]
        with (
            mock.patch.object(sampler.nodes, "common_ksampler", side_effect=[(completed[0],), (completed[1],)]),
            mock.patch.object(sampler, "apply_previous_shot_context", side_effect=lambda conditioning, *_: conditioning),
            mock.patch.object(sampler, "send_preview_event") as emit,
            mock.patch.object(sampler, "publish_preview") as publish,
        ):
            output = sampler.FL_MiniMaxH3BeatKSampler.execute(
                model=object(),
                shot_plan=plan,
                seed=20,
                seed_mode="increment",
                steps=4,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                vae=object(),
                live_preview=True,
            ).result[0]

        self.assertEqual(len(output), 2)
        self.assertEqual(publish.call_count, 2)
        self.assertEqual([call.args[6] for call in publish.call_args_list], [0, 1])
        self.assertEqual([call.args[2]["seed"] for call in publish.call_args_list], [20, 21])
        self.assertEqual(emit.call_args_list[0].args[3], "start")
        self.assertEqual(emit.call_args_list[-1].args[3], "done")

    def test_upscale_sampler_publishes_after_removing_the_noise_mask(self):
        latent = nested_latent()
        metadata = latent["fl_h3_shot"]
        shot = {
            "index": 0,
            "start_frame": 0,
            "end_frame": 4,
            "authored_frames": 4,
            "render_frames": 6,
            "motion_context": metadata["motion_context"],
            "conditioning": "conditioning",
            "timeline": {"type": "minimax_h3_prompt_timeline"},
            "latent": {"samples": "source"},
        }
        plan = {
            "type": "minimax_h3_beat_shot_plan",
            "version": 1,
            "fps": 24.0,
            "total_frames": 4,
            "shots": [shot],
        }

        def sample(*args, **kwargs):
            return (args[8].copy(),)

        with (
            mock.patch.object(sampler, "_pixel_upscale_latent", return_value=(latent, 80, 40, 160, 80)),
            mock.patch.object(sampler, "_apply_timeline", return_value="conditioning"),
            mock.patch.object(sampler, "primary_only_noise_mask", return_value="mask"),
            mock.patch.object(sampler.nodes, "common_ksampler", side_effect=sample),
            mock.patch.object(sampler, "send_preview_event"),
            mock.patch.object(sampler, "publish_preview") as publish,
        ):
            output = sampler.FL_MiniMaxH3BeatUpscaleKSampler.execute(
                model=object(),
                shot_plan=plan,
                latent=latent,
                vae=object(),
                target_long_side=160,
                upscale_method="bicubic",
                seed=30,
                seed_mode="fixed",
                steps=4,
                start_at_step=1,
                end_at_step=4,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
                live_preview=True,
            ).result[0]

        self.assertNotIn("noise_mask", output)
        self.assertIs(publish.call_args.args[0], output)
        self.assertEqual(publish.call_args.args[2]["seed"], 30)
        self.assertEqual(publish.call_args.args[5:8], ("upscale", 0, 1))

    def test_disabled_preview_has_no_side_effects(self):
        plan = beat_plan(1)
        with (
            mock.patch.object(sampler.nodes, "common_ksampler", return_value=(nested_latent(),)),
            mock.patch.object(sampler, "send_preview_event") as emit,
            mock.patch.object(sampler, "publish_preview") as publish,
        ):
            sampler.FL_MiniMaxH3BeatKSampler.execute(
                object(),
                plan,
                0,
                "fixed",
                1,
                1.0,
                "euler",
                "normal",
                1.0,
            )

        emit.assert_not_called()
        publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
