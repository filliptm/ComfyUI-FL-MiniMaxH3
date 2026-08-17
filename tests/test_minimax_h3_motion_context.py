import importlib.util
import pathlib
import sys
import types
import unittest

import torch

import comfy.ldm.minimax.model as minimax_model
import comfy.nested_tensor


ROOT = pathlib.Path(__file__).parents[1]


def load_module(relative_path):
    path = pathlib.Path(relative_path)
    module_name = "fl_minimax_h3_motion_tests." + ".".join(path.with_suffix("").parts)
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


timeline = load_module("nodes/FL_MiniMaxH3PromptTimeline.py")
motion = load_module("nodes/FL_MiniMaxH3ShotMotionContext.py")
runtime = sys.modules["fl_minimax_h3_motion_tests.nodes._motion_context"]
sampler = load_module("nodes/FL_MiniMaxH3BeatKSampler.py")
assembler = load_module("nodes/FL_MiniMaxH3ShotAssembler.py")


class FakeClip:
    def tokenize(self, prompt, **kwargs):
        return {"prompt": prompt, **kwargs}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros((1, 1, 1)), {"prompt": tokens["prompt"]}]]


class FakeVideoVAE:
    def __init__(self):
        self.encoded = None
        self.decode_shapes = []

    def decode(self, video):
        self.decode_shapes.append(tuple(video.shape))
        latent_t = video.shape[2]
        frame_count = 5 if latent_t == 2 else ((latent_t - 2) // 5) * 17 + 5
        token_start = video.storage_offset() // video.stride(2)
        frame_start = token_start // 5 * 17
        return torch.arange(frame_start, frame_start + frame_count, dtype=torch.float32).view(1, frame_count, 1, 1, 1).expand(
            1, frame_count, 4, 4, 3
        )

    def encode(self, images):
        self.encoded = images.clone()
        frames = images.shape[0]
        latent_t = {1: 1, 5: 2, 22: 7, 39: 12}.get(frames, 2)
        return torch.zeros((1, 24, latent_t, images.shape[1] // 2, images.shape[2] // 2))


class FakeAudioVAE:
    audio_sample_rate = 24000

    def encode(self, audio):
        return torch.zeros((1, 32, 2, round(audio.shape[1] / 600)))


def prompt_schedule():
    sections = []
    for index, (start, end) in enumerate(((0, 54), (54, 109), (109, 163)), 1):
        sections.append({
            "line": index,
            "start": start / 24,
            "end": end / 24,
            "fade_in_end": start / 24,
            "fade_out_start": end / 24,
            "crossfade_start": start / 24,
            "crossfade_end": start / 24,
            "start_frame": start,
            "end_frame": end,
            "prompt": f"Shot {index}.",
            "curve": "cosine",
        })
    return {
        "type": "fl_prompt_schedule",
        "version": 2,
        "duration": 163 / 24,
        "audio_duration": 163 / 24,
        "source_unit": "frames",
        "fps": 24.0,
        "sections": sections,
    }


def planned_shots(ref_images=None):
    return timeline.FL_MiniMaxH3BeatShotPlanner.execute(
        clip=FakeClip(),
        vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(),
        prompt_schedule=prompt_schedule(),
        timeline_audio={
            "waveform": torch.zeros((1, 2, 163000)),
            "sample_rate": 24000,
        },
        global_prompt="Same character.",
        width=64,
        height=64,
        affect_audio="video only",
        ref_image_size="match",
        ref_images=ref_images,
    ).result[0]


def nested_latent(video_t=17, audio_t=94, metadata=None):
    value = {
        "samples": comfy.nested_tensor.NestedTensor((
            torch.zeros((1, 24, video_t, 4, 4)),
            torch.arange(audio_t, dtype=torch.float32).view(1, 1, 1, audio_t).expand(
                1, 32, 2, audio_t
            ),
        )),
    }
    if metadata is not None:
        value["fl_h3_shot"] = metadata
    return value


class ShotMotionContextPlanTests(unittest.TestCase):
    def test_default_context_extends_only_destination_renders(self):
        source = planned_shots()

        plan = motion.FL_MiniMaxH3ShotMotionContext.execute(
            source,
            "5",
            22,
        ).result[0]

        self.assertEqual([shot["render_frames"] for shot in plan["shots"]], [56, 73, 73])
        self.assertEqual(plan["total_render_frames"], 202)
        self.assertEqual([shot["authored_frames"] for shot in plan["shots"]], [54, 55, 54])
        self.assertEqual([shot["motion_context"]["trim_frames"] for shot in plan["shots"]], [0, 5, 5])
        self.assertEqual(plan["shots"][1]["timeline"]["authored_frame_count"], 60)
        self.assertEqual(plan["shots"][1]["timeline"]["sections"][0]["motion_context"], True)
        self.assertEqual(plan["shots"][1]["timeline"]["sections"][1]["start"], 5 / 24)
        self.assertIsNot(plan, source)
        self.assertNotIn("motion_context", source["shots"][1])

    def test_zero_context_is_an_exact_plan_no_op(self):
        source = planned_shots()

        output = motion.FL_MiniMaxH3ShotMotionContext.execute(source, "0", 0).result[0]

        self.assertIs(output, source)

    def test_overrides_are_per_destination_render(self):
        plan = motion.FL_MiniMaxH3ShotMotionContext.execute(
            planned_shots(),
            "1",
            1,
            "2: 22, 10\n3: 0, 12",
        ).result[0]

        self.assertEqual(plan["shots"][1]["motion_context"]["video_frames"], 22)
        self.assertEqual(plan["shots"][1]["render_frames"], 90)
        self.assertEqual(plan["shots"][2]["motion_context"]["video_frames"], 0)
        self.assertEqual(plan["shots"][2]["motion_context"]["audio_frames"], 12)
        self.assertEqual(plan["shots"][2]["render_frames"], 56)

    def test_context_cannot_reach_before_the_source_authored_shot(self):
        source = planned_shots()
        source["shots"][0]["authored_frames"] = 4

        with self.assertRaisesRegex(ValueError, "use 1 or less"):
            motion.FL_MiniMaxH3ShotMotionContext.execute(source, "5", 0)

    def test_context_extension_rebuilds_both_visual_reference_conditionings(self):
        source = planned_shots({"ref_image_1": torch.zeros((1, 64, 64, 3))})

        plan = motion.FL_MiniMaxH3ShotMotionContext.execute(source, "5", 22).result[0]

        shot = plan["shots"][1]
        self.assertIn("reference_free_conditioning", shot)
        self.assertEqual(
            shot["timeline"]["conditioning_groups"][0]["section_indices"],
            shot["timeline"]["reference_free_conditioning_groups"][0]["section_indices"],
        )


class RuntimePatchTests(unittest.TestCase):
    def test_unmarked_payload_passes_through_the_model_wrapper(self):
        payload = {"seed": 1}
        calls = []

        def execute(*args, **kwargs):
            calls.append((args, kwargs))
            return "sampled"

        output = runtime._model_wrapper(
            execute,
            "x",
            "t",
            c_crossattn=torch.zeros((1, 7, 1)),
            minimax_payload=payload,
        )

        self.assertEqual(output, "sampled")
        self.assertEqual(payload, {"seed": 1})
        self.assertEqual(len(calls), 1)

    def test_marked_layout_anchors_context_inside_the_target_timeline(self):
        keyframes = [
            {
                "resolved_frame_index": 0,
                runtime.VIDEO_FRAME_MARKER: frame,
                "latent": torch.zeros((1, 24, 1, 4, 4)),
            }
            for frame in (0, 1, 5)
        ]
        refs = [{
            "kind": "audio",
            "ref_audio_t": 4,
            runtime.AUDIO_END_FRAME_MARKER: 5.0,
        }]

        payload = {
            "keyframes": keyframes,
            "refs": refs,
            "frame_count": 22,
            "layout": minimax_model.PackedLayout(
                7, 7, 4, 4, 16, keyframes=keyframes, refs=refs, frame_count=22
            ),
        }

        runtime._prepare_payload(
            payload,
            torch.zeros((1, 7, 1)),
            [(1, 24, 7, 4, 4), (1, 32, 2, 16)],
        )

        layout = payload["layout"]
        origin = 11.0
        cond = [segment for segment in layout.segments if segment[2] == "cond"]
        self.assertEqual(
            [layout.position_ids[start, 0].item() for start, _, _ in cond],
            [origin, origin + 5 / 3, origin + 25 / 3],
        )
        ref_audio = next(segment for segment in layout.segments if segment[2] == "ref_audio")
        times = layout.position_ids[ref_audio[0]:ref_audio[1], 0]
        self.assertAlmostEqual(times[3].item(), origin + 25 / 3 - 1)
        self.assertEqual(len(payload["cond_video_latents"]), 3)
        target_audio = next(segment for segment in layout.segments if segment[2] == "audio")
        target_video = next(segment for segment in layout.segments if segment[2] == "video")
        self.assertEqual(target_audio[1], target_video[0])
        self.assertEqual(target_video[1], layout.seq_len)

    def test_runtime_wrapper_is_scoped_to_a_cloned_model_patcher(self):
        class FakeModel:
            def __init__(self, model_options=None, wrappers=None):
                self.model_options = dict(model_options or {})
                self.wrappers = {
                    kind: {key: list(values) for key, values in groups.items()}
                    for kind, groups in (wrappers or {}).items()
                }

            def clone(self):
                return FakeModel(self.model_options, self.wrappers)

            def get_wrappers(self, wrapper_type, key):
                return self.wrappers.get(wrapper_type, {}).get(key, [])

            def add_wrapper_with_key(self, wrapper_type, key, wrapper):
                self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

        spectrum_binding = object()
        source = FakeModel({"spectrum_h3_binding": spectrum_binding})

        patched = runtime.motion_context_model(source)

        wrapper_type = runtime.comfy.patcher_extension.WrappersMP.APPLY_MODEL
        self.assertIsNot(patched, source)
        self.assertIs(patched.model_options["spectrum_h3_binding"], spectrum_binding)
        self.assertFalse(source.get_wrappers(wrapper_type, runtime._WRAPPER_KEY))
        self.assertEqual(len(patched.get_wrappers(wrapper_type, runtime._WRAPPER_KEY)), 1)


class ContextConditioningTests(unittest.TestCase):
    def test_previous_authored_tail_is_added_without_losing_existing_metadata(self):
        previous = nested_latent()
        existing_ref = {"kind": "audio", "ref_audio_t": 1, "audio_latent": torch.zeros((1, 32, 2, 1))}
        conditioning = [[torch.zeros((1, 1, 1)), {"mask": "keep", "minimax_refs": [existing_ref]}]]
        source = {"authored_frames": 54}
        target = {
            "render_frames": 73,
            "motion_context": {
                "video_frames": 5,
                "audio_frames": 22,
            },
        }
        vae = FakeVideoVAE()

        resolved = runtime.apply_previous_shot_context(
            conditioning, previous, source, target, vae
        )

        metadata = resolved[0][1]
        self.assertEqual(metadata["mask"], "keep")
        self.assertEqual(len(metadata["minimax_keyframes"]), 2)
        self.assertEqual(
            [value[runtime.VIDEO_FRAME_MARKER] for value in metadata["minimax_keyframes"]],
            [0, 1],
        )
        self.assertEqual(metadata["minimax_refs"][0], existing_ref)
        audio_ref = metadata["minimax_refs"][1]
        self.assertEqual(audio_ref["ref_audio_t"], 37)
        self.assertEqual(audio_ref["audio_latent"].shape[-1], 37)
        self.assertTrue(torch.equal(vae.encoded[:, 0, 0, 0], torch.arange(49, 54)))
        self.assertEqual(vae.decode_shapes, [(1, 24, 12, 4, 4)])
        self.assertNotIn("minimax_keyframes", conditioning[0][1])

    def test_shared_motion_context_is_prepared_once_for_both_reference_paths(self):
        previous = nested_latent()
        conditionings = [
            [[torch.zeros((1, 1, 1)), {"path": "selected"}]],
            [[torch.zeros((1, 1, 1)), {"path": "visual-off"}]],
        ]
        source = {"authored_frames": 54}
        target = {
            "render_frames": 73,
            "motion_context": {
                "video_frames": 5,
                "audio_frames": 22,
            },
        }
        vae = FakeVideoVAE()

        resolved = runtime.apply_previous_shot_contexts(
            conditionings,
            previous,
            source,
            target,
            vae,
        )

        self.assertEqual(vae.decode_shapes, [(1, 24, 12, 4, 4)])
        self.assertEqual([value[0][1]["path"] for value in resolved], ["selected", "visual-off"])
        self.assertTrue(all("minimax_keyframes" in value[0][1] for value in resolved))
        self.assertTrue(all("minimax_refs" in value[0][1] for value in resolved))


class MotionContextConsumerTests(unittest.TestCase):
    def test_upscale_mask_protects_context_video_steps(self):
        samples = nested_latent(video_t=7)["samples"]

        video, audio = sampler.primary_only_noise_mask(samples, 2).unbind()

        self.assertTrue(torch.all(video[:, :, :2] == 0))
        self.assertTrue(torch.all(video[:, :, 2:] == 1))
        self.assertTrue(torch.all(audio == 0))

    def test_assembler_removes_the_hidden_context_prefix(self):
        metadata = {
            "version": 1,
            "index": 0,
            "start_frame": 0,
            "end_frame": 5,
            "authored_frames": 5,
            "render_frames": 22,
            "total_frames": 5,
            "fps": 24.0,
            "motion_context": {"trim_frames": 5},
        }
        value = nested_latent(video_t=7, metadata=metadata)

        vae = FakeVideoVAE()
        images = assembler.FL_MiniMaxH3ShotAssembler.execute([value], [vae]).result[0]

        self.assertTrue(torch.equal(images[:, 0, 0, 0], torch.arange(5, 10, dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
