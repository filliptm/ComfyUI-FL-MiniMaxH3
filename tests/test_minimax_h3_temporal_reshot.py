import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from fractions import Fraction
from unittest import mock

import torch

import comfy.nested_tensor


ROOT = pathlib.Path(__file__).parents[1]


def load_module(relative_path):
    path = pathlib.Path(relative_path)
    module_name = "fl_minimax_h3_temporal_reshot_tests." + ".".join(path.with_suffix("").parts)
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


reshot = load_module("nodes/FL_MiniMaxH3TemporalReshot.py")
routes = load_module("routes/temporal_reshot.py")


def settings(**values):
    return {**reshot.DEFAULT_RESHOT_SETTINGS, **values}


class FakeClip:
    def __init__(self):
        self.tokens = None

    def tokenize(self, prompt, **kwargs):
        self.tokens = {"prompt": prompt, **kwargs}
        return self.tokens

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros((1, 1, 1)), {"prompt": tokens["prompt"]}]]


class FakeVideoVAE:
    def encode(self, images):
        frames, height, width = images.shape[:3]
        latent_t = reshot.minimax_h3.video_latent_t(frames)
        return torch.ones((1, 24, latent_t, height // 16, width // 16))


class TemporalWindowTests(unittest.TestCase):
    def test_frame_resampling_drops_and_duplicates_evenly(self):
        frames = torch.arange(5)
        self.assertEqual(reshot._resample_frames(frames, 3).tolist(), [0, 2, 4])
        self.assertEqual(reshot._resample_frames(frames[:3], 5).tolist(), [0, 0, 1, 2, 2])

    def test_settings_are_integer_and_bounded(self):
        parsed = reshot.parse_reshot_settings(
            '{"version":1,"start_frame":7,"frame_count":12,"context_before":5,'
            '"context_after":9,"edge_blend_frames":2}'
        )
        self.assertEqual(parsed["start_frame"], 7)
        self.assertEqual(parsed["edge_blend_frames"], 2)
        with self.assertRaisesRegex(ValueError, "frame_count"):
            reshot.parse_reshot_settings('{"version":1,"frame_count":0}')
        with self.assertRaisesRegex(ValueError, "integer"):
            reshot.parse_reshot_settings('{"version":1,"start_frame":1.5}')

    def test_center_window_aligns_to_h3_grid(self):
        window = reshot.build_reshot_window(
            360,
            settings(start_frame=144, frame_count=24, context_before=39, context_after=39),
        )
        self.assertEqual(window["work_start_frame"], 105)
        self.assertEqual(window["render_frames"], 107)
        self.assertEqual(window["selection_offset"], 39)
        self.assertEqual(window["padding_frames"], 0)

    def test_end_window_shifts_earlier_before_padding(self):
        window = reshot.build_reshot_window(
            360,
            settings(start_frame=350, frame_count=10, context_before=0, context_after=0),
        )
        self.assertEqual(window["render_frames"], 22)
        self.assertEqual(window["work_start_frame"], 338)
        self.assertEqual(window["selection_offset"], 12)

    def test_short_source_repeats_only_after_real_source(self):
        window = reshot.build_reshot_window(
            4,
            settings(start_frame=1, frame_count=2, context_before=9, context_after=9),
        )
        self.assertEqual(window["render_frames"], 5)
        self.assertEqual(window["work_source_frames"], 4)
        self.assertEqual(window["padding_frames"], 1)

    def test_selection_expands_to_complete_video_tokens(self):
        token_start, token_end, edges = reshot.selection_token_range(7, 22, 3, 3)
        self.assertEqual(edges, [0, 1, 5, 9, 13, 17, 18, 22])
        self.assertEqual((token_start, token_end), (1, 3))
        self.assertEqual((edges[token_start], edges[token_end]), (1, 9))

    def test_temporal_noise_mask_edits_full_video_frame_and_never_audio(self):
        video = torch.zeros((1, 24, 7, 3, 4))
        audio = torch.zeros((1, 32, 2, 37))
        mask = reshot._reshot_noise_mask(video, audio, 2, 5).unbind()
        self.assertTrue(torch.all(mask[0][:, :, 2:5] == 1))
        self.assertTrue(torch.all(mask[0][:, :, :2] == 0))
        self.assertTrue(torch.all(mask[0][:, :, 5:] == 0))
        self.assertEqual(mask[1].count_nonzero().item(), 0)

    def test_source_audio_is_padded_to_the_aligned_render(self):
        class AudioVAE:
            audio_sample_rate = 24

            def __init__(self):
                self.input = None

            def encode(self, waveform):
                self.input = waveform
                return torch.ones((1, 32, 2, 8))

        audio_vae = AudioVAE()
        expected = torch.zeros((1, 32, 2, 8))
        encoded = reshot._working_audio_latent(
            audio_vae,
            {"waveform": torch.ones((1, 1, 2)), "sample_rate": 24},
            5,
            expected,
        )
        self.assertEqual(audio_vae.input.shape, (1, 5, 1))
        self.assertTrue(torch.all(encoded == 1))


class VideoLibraryTests(unittest.TestCase):
    def test_probe_normalizes_a_30_fps_source_to_a_24_fps_timeline(self):
        stream = types.SimpleNamespace(
            average_rate=Fraction(30, 1),
            duration=60,
            time_base=Fraction(1, 30),
            frames=60,
            width=1280,
            height=720,
            codec_context=types.SimpleNamespace(name="h264"),
        )
        container = mock.MagicMock()
        container.__enter__.return_value = container
        container.__exit__.return_value = False
        container.duration = 2 * reshot.av.time_base
        container.format.name = "mov,mp4"
        container.streams.video = [stream]
        container.streams.audio = []
        container.demux.return_value = [types.SimpleNamespace(pts=index) for index in range(60)]
        video_input = mock.Mock()
        video_input.get_bit_depth.return_value = 8
        with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
            with (
                mock.patch.object(reshot.av, "open", return_value=container),
                mock.patch.object(reshot.InputImpl, "VideoFromFile", return_value=video_input),
            ):
                info = reshot.probe_source_video(source.name)
        self.assertEqual(info["source_frame_rate"], 30.0)
        self.assertEqual(info["source_frame_count"], 60)
        self.assertTrue(info["source_constant_frame_rate"])
        self.assertTrue(info["converted_to_24_fps"])
        self.assertEqual(info["frame_rate"], 24.0)
        self.assertEqual(info["frame_count"], 48)

    def test_available_files_are_recursive_sorted_and_video_only(self):
        with tempfile.TemporaryDirectory() as directory:
            input_dir = pathlib.Path(directory)
            (input_dir / "nested").mkdir()
            (input_dir / "z.mp4").touch()
            (input_dir / "nested" / "A.MOV").touch()
            (input_dir / "notes.txt").touch()
            with (
                mock.patch.object(reshot.folder_paths, "get_input_directory", return_value=directory),
                mock.patch.object(
                    reshot.folder_paths,
                    "recursive_search",
                    return_value=(["z.mp4", "notes.txt", "nested/A.MOV"], None),
                ),
            ):
                files = reshot.available_video_files()
        self.assertEqual(files, ["nested/A.MOV", "z.mp4"])

    def test_resolver_rejects_paths_outside_comfyui_input(self):
        with tempfile.TemporaryDirectory() as directory:
            input_dir = pathlib.Path(directory) / "input"
            input_dir.mkdir()
            outside = pathlib.Path(directory) / "outside.mp4"
            outside.touch()
            with (
                mock.patch.object(reshot.folder_paths, "get_input_directory", return_value=str(input_dir)),
                mock.patch.object(reshot.folder_paths, "annotated_filepath", return_value=("../outside.mp4", None)),
                mock.patch.object(reshot.folder_paths, "get_annotated_filepath", return_value=str(outside)),
            ):
                with self.assertRaisesRegex(ValueError, "inside the ComfyUI input directory"):
                    reshot.resolve_video_path("../outside.mp4")

    def test_library_entries_include_recursive_paths_and_header_metadata(self):
        stream = types.SimpleNamespace(
            average_rate=Fraction(24, 1),
            duration=None,
            time_base=None,
            frames=48,
            width=1280,
            height=720,
        )
        container = mock.MagicMock()
        container.__enter__.return_value = container
        container.__exit__.return_value = False
        container.duration = 2 * reshot.av.time_base
        container.streams.video = [stream]
        container.streams.audio = [object()]
        with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
            path = pathlib.Path(source.name)
            with (
                mock.patch.object(reshot, "available_video_files", return_value=["folder/clip.mp4"]),
                mock.patch.object(reshot, "resolve_video_path", return_value=path),
                mock.patch.object(reshot.av, "open", return_value=container),
            ):
                entries = reshot.video_library_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["path"], "folder/clip.mp4")
        self.assertEqual(entries[0]["folder"], "folder")
        self.assertEqual(entries[0]["frame_count"], 48)
        self.assertEqual(entries[0]["frame_rate"], 24.0)
        self.assertTrue(entries[0]["declared_24_fps"])
        self.assertTrue(entries[0]["has_audio"])

    def test_library_keeps_unreadable_file_with_actionable_error(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
            path = pathlib.Path(source.name)
            with (
                mock.patch.object(reshot, "available_video_files", return_value=["broken.mp4"]),
                mock.patch.object(reshot, "resolve_video_path", return_value=path),
                mock.patch.object(reshot.av, "open", side_effect=reshot.av.error.FFmpegError(1, "bad video")),
            ):
                entries = reshot.video_library_entries()
        self.assertEqual(entries[0]["path"], "broken.mp4")
        self.assertIn("error", entries[0])


class VideoLibraryRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_files_route_returns_library_payload(self):
        expected = [{"path": "clips/a.mp4", "filename": "a.mp4"}]
        with mock.patch.object(routes, "video_library_entries", return_value=expected):
            response = await routes.temporal_reshot_files(object())
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.text), {"files": expected})

    async def test_files_route_reports_probe_failure(self):
        with mock.patch.object(routes, "video_library_entries", side_effect=ValueError("bad library")):
            response = await routes.temporal_reshot_files(object())
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(response.text)["error"], "bad library")


class PlannerTests(unittest.TestCase):
    def test_planner_builds_single_masked_reshot_with_references(self):
        clip = FakeClip()
        source_images = torch.rand((22, 32, 32, 3))
        components = types.SimpleNamespace(images=source_images, audio=None)
        probe = {
            "width": 32,
            "height": 32,
            "duration": 100 / 24,
            "frame_rate": 24.0,
            "frame_rate_numerator": 24,
            "frame_rate_denominator": 1,
            "frame_count": 100,
            "frame_count_estimated": False,
            "bit_depth": 8,
            "codec": "h264",
            "container": "mov,mp4",
            "has_audio": False,
            "size": 123,
            "mtime_ns": 456,
        }
        encoded_settings = (
            '{"version":1,"start_frame":10,"frame_count":4,"context_before":5,'
            '"context_after":5,"edge_blend_frames":0}'
        )
        with (
            mock.patch.object(reshot, "resolve_video_path", return_value=pathlib.Path("source.mp4")),
            mock.patch.object(reshot, "probe_source_video", return_value=probe),
            mock.patch.object(reshot, "source_fingerprint", return_value={"size": 123, "mtime_ns": 456}),
            mock.patch.object(reshot, "_decode_working_components", return_value=(components, source_images)),
            mock.patch.object(reshot.minimax_h3, "adapt_canvas", return_value=(32, 32)),
            mock.patch.object(reshot.minimax_h3, "_resize", side_effect=lambda images, *_: images),
        ):
            output = reshot.FL_MiniMaxH3TemporalReshotPlanner.execute(
                clip,
                FakeVideoVAE(),
                "source.mp4",
                "A new camera move.",
                encoded_settings,
                "match",
                ref_images={"ref_image_1": torch.rand((1, 32, 32, 3))},
            )
        plan = output[0]
        shot = plan["shots"][0]
        self.assertEqual(plan["mode"], "temporal_reshot")
        self.assertEqual((shot["start_frame"], shot["end_frame"]), (10, 14))
        self.assertEqual(shot["render_frames"], 22)
        self.assertEqual(clip.tokens["prompt"], "A new camera move.")
        self.assertEqual(len(clip.tokens["minimax_ref_items"]), 1)
        masks = shot["latent"]["noise_mask"].unbind()
        self.assertGreater(masks[0].count_nonzero().item(), 0)
        self.assertEqual(masks[1].count_nonzero().item(), 0)
        self.assertEqual(shot["reshot"]["selection_frames"], 4)

def assembler_fixture(blend_frames=0):
    reshot_metadata = {
        "version": 1,
        "selection_offset": 3,
        "selection_frames": 4,
        "edge_blend_frames": blend_frames,
    }
    shot = {
        "index": 0,
        "start_frame": 8,
        "end_frame": 12,
        "authored_frames": 4,
        "render_frames": 22,
        "reshot": reshot_metadata,
    }
    plan = {
        "type": "minimax_h3_beat_shot_plan",
        "version": 1,
        "mode": "temporal_reshot",
        "fps": 24,
        "total_frames": 20,
        "source": {
            "filename": "source.mp4",
            "fingerprint": {"size": 123, "mtime_ns": 456},
            "bit_depth": 10,
        },
        "shots": [shot],
    }
    video = torch.zeros((1, 24, 7, 2, 2))
    audio = torch.zeros((1, 32, 2, 37))
    latent = {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "fl_h3_shot": {
            "version": 1,
            "index": 0,
            "start_frame": 8,
            "end_frame": 12,
            "render_frames": 22,
            "reshot": dict(reshot_metadata),
        },
    }
    return plan, latent


class FakeDecodeVAE:
    def decode(self, video):
        return torch.full((22, 4, 5, 3), 0.75)


class AssemblerTests(unittest.TestCase):
    def test_assembler_replaces_exact_range_and_preserves_video_components(self):
        plan, latent = assembler_fixture()
        original = torch.linspace(0, 1, 20 * 4 * 5 * 3).reshape(20, 4, 5, 3)
        audio = {"waveform": torch.ones((1, 2, 100)), "sample_rate": 48000}
        alpha = torch.rand((20, 4, 5, 1))
        components = types.SimpleNamespace(
            images=original.clone(),
            audio=audio,
            frame_rate=Fraction(24, 1),
            metadata={"title": "source"},
            alpha=alpha,
        )
        source_input = mock.Mock()
        source_input.get_components.return_value = components
        with (
            mock.patch.object(reshot, "resolve_video_path", return_value=pathlib.Path("source.mp4")),
            mock.patch.object(reshot, "source_fingerprint", return_value={"size": 123, "mtime_ns": 456}),
            mock.patch.object(reshot.InputImpl, "VideoFromFile", return_value=source_input),
        ):
            output = reshot.FL_MiniMaxH3TemporalReshotAssembler.execute(
                plan, latent, FakeDecodeVAE()
            )[0]
        result = output.get_components()
        self.assertTrue(torch.equal(result.images[:8], original[:8]))
        self.assertTrue(torch.equal(result.images[12:], original[12:]))
        self.assertTrue(torch.all(result.images[8:12] == 0.75))
        self.assertIs(result.audio, audio)
        self.assertIs(result.alpha, alpha)
        self.assertEqual(result.metadata, {"title": "source"})
        self.assertEqual(result.frame_rate, Fraction(24, 1))
        self.assertEqual(output.get_bit_depth(), 10)

    def test_assembler_normalizes_source_frames_and_output_rate(self):
        plan, latent = assembler_fixture()
        original = torch.arange(30, dtype=torch.float32).view(30, 1, 1, 1).expand(30, 4, 5, 3).clone()
        alpha = torch.arange(30, dtype=torch.float32).view(30, 1, 1, 1).expand(30, 4, 5, 1).clone()
        components = types.SimpleNamespace(
            images=original,
            audio=None,
            frame_rate=Fraction(30, 1),
            metadata={},
            alpha=alpha,
        )
        source_input = mock.Mock()
        source_input.get_components.return_value = components
        expected_images = reshot._resample_frames(original, 20)
        expected_alpha = reshot._resample_frames(alpha, 20)
        with (
            mock.patch.object(reshot, "resolve_video_path", return_value=pathlib.Path("source.mp4")),
            mock.patch.object(reshot, "source_fingerprint", return_value={"size": 123, "mtime_ns": 456}),
            mock.patch.object(reshot.InputImpl, "VideoFromFile", return_value=source_input),
        ):
            output = reshot.FL_MiniMaxH3TemporalReshotAssembler.execute(plan, latent, FakeDecodeVAE())[0]
        result = output.get_components()
        self.assertEqual(result.images.shape[0], 20)
        self.assertTrue(torch.equal(result.images[:8], expected_images[:8]))
        self.assertTrue(torch.equal(result.images[12:], expected_images[12:]))
        self.assertTrue(torch.equal(result.alpha, expected_alpha))
        self.assertEqual(result.frame_rate, Fraction(24, 1))

    def test_edge_blend_changes_only_frames_inside_selection(self):
        source = torch.zeros((6, 2, 2, 3))
        generated = torch.ones_like(source)
        reshot._blend_selection(source, generated, 2)
        self.assertGreater(source[0].mean().item(), 0)
        self.assertLess(source[0].mean().item(), source[2].mean().item())
        self.assertEqual(source[2].mean().item(), 1)
        self.assertEqual(source[3].mean().item(), 1)
        self.assertLess(source[-1].mean().item(), 1)

    def test_assembler_rejects_changed_source(self):
        plan, latent = assembler_fixture()
        with (
            mock.patch.object(reshot, "resolve_video_path", return_value=pathlib.Path("source.mp4")),
            mock.patch.object(reshot, "source_fingerprint", return_value={"size": 999, "mtime_ns": 456}),
        ):
            with self.assertRaisesRegex(ValueError, "source video changed"):
                reshot.FL_MiniMaxH3TemporalReshotAssembler.execute(
                    plan, latent, FakeDecodeVAE()
                )


if __name__ == "__main__":
    unittest.main()
