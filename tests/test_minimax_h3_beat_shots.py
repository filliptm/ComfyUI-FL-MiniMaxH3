import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import torch

import comfy.nested_tensor


ROOT = pathlib.Path(__file__).parents[1]


def load_module(name, relative_path):
    path = pathlib.Path(relative_path)
    module_name = "fl_minimax_h3_tests." + ".".join(path.with_suffix("").parts)
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


timeline = load_module(
    "fl_minimax_h3_beat_shot_planner",
    "nodes/FL_MiniMaxH3PromptTimeline.py",
)
sampler = load_module(
    "fl_minimax_h3_beat_ksampler",
    "nodes/FL_MiniMaxH3BeatKSampler.py",
)
assembler = load_module(
    "fl_minimax_h3_shot_assembler",
    "nodes/FL_MiniMaxH3ShotAssembler.py",
)


def schedule(boundaries=(0, 54, 109, 163, 217), render_groups=None):
    sections = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
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
        if render_groups is not None and render_groups[index - 1] is not None:
            sections[-1]["render_group"] = render_groups[index - 1]
    return {
        "type": "fl_prompt_schedule",
        "version": 2,
        "duration": boundaries[-1] / 24,
        "audio_duration": boundaries[-1] / 24,
        "source_unit": "frames",
        "fps": 24.0,
        "sections": sections,
    }


class FakeClip:
    def __init__(self):
        self.encoded_prompts = []
        self.tokenized_ref_types = []

    def tokenize(self, prompt, **kwargs):
        self.tokenized_ref_types.append([
            item["type"] for item in kwargs.get("minimax_ref_items", [])
        ])
        return {"prompt": prompt, **kwargs}

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded_prompts.append(tokens["prompt"])
        return [[torch.zeros((1, 1, 1)), {"prompt": tokens["prompt"]}]]


class FakeVideoVAE:
    def __init__(self):
        self.encoded_images = []

    def encode(self, images):
        self.encoded_images.append(images)
        return torch.zeros((1, 24, 2, 2, 2))


class FakeAudioVAE:
    audio_sample_rate = 24000

    def __init__(self):
        self.encoded_samples = []

    def encode(self, audio):
        self.encoded_samples.append(audio.shape[1])
        latent_t = max(1, round(audio.shape[1] / self.audio_sample_rate * 40))
        return torch.zeros((1, 32, 2, latent_t))


class ShotPlannerTests(unittest.TestCase):
    def test_schema_uses_timeline_authored_groups_without_planner_mode_controls(self):
        inputs = {
            value.id: value
            for value in timeline.FL_MiniMaxH3BeatShotPlanner.define_schema().inputs
        }

        self.assertNotIn("render_mode", inputs)
        self.assertNotIn("sections_per_chunk", inputs)
        self.assertNotIn("individual_start_section", inputs)
        self.assertIn("prompt_envelopes", inputs)
        fidelity = inputs["visual_condition_fidelity"]
        self.assertEqual((fidelity.default, fidelity.min, fidelity.max, fidelity.step), (1.0, 0.0, 1.0, 0.01))
        self.assertEqual(inputs["visual_reference_mode"].default, "full")
        self.assertEqual(
            inputs["visual_reference_mode"].options,
            ["full", "qwen only", "latent only", "off"],
        )
        strength = inputs["reference_strength"]
        self.assertEqual((strength.default, strength.min, strength.max, strength.step), (1.0, 0.0, 1.0, 0.01))

    def test_visual_reference_modes_control_qwen_and_latent_paths_independently(self):
        image = torch.zeros((1, 64, 64, 3))
        video = torch.zeros((5, 64, 64, 3))
        audio = {
            "waveform": torch.zeros((1, 2, 24000)),
            "sample_rate": 24000,
        }
        expected = {
            "full": (
                ["image", "audio", "video", "audio"],
                ["image", "video_audio", "audio"],
                2,
            ),
            "qwen only": (
                ["image", "audio", "video", "audio"],
                ["audio", "audio"],
                0,
            ),
            "latent only": (
                ["audio", "audio"],
                ["image", "video_audio", "audio"],
                2,
            ),
            "off": (
                ["audio", "audio"],
                ["audio", "audio"],
                0,
            ),
        }

        with mock.patch.object(timeline.minimax_h3, "adapt_canvas", return_value=(64, 64)):
            for mode, (item_types, block_kinds, encode_count) in expected.items():
                with self.subTest(mode=mode):
                    vae = FakeVideoVAE()
                    items, blocks = timeline._prepare_references(
                        vae,
                        FakeAudioVAE(),
                        64,
                        64,
                        5,
                        "match",
                        {"ref_image_1": image},
                        {"ref_video_1": video},
                        {"ref_video_audio_1": audio},
                        {"ref_audio_1": audio},
                        visual_reference_mode=mode,
                    )

                    self.assertEqual([item["type"] for item in items], item_types)
                    self.assertEqual([block["kind"] for block in blocks], block_kinds)
                    self.assertEqual(len(vae.encoded_images), encode_count)

    def test_prompt_envelope_set_is_sliced_for_each_shot(self):
        audio = {
            "waveform": torch.zeros((1, 2, 217000)),
            "sample_rate": 24000,
        }
        prompt_envelopes = {
            "type": "fl_prompt_envelope_set",
            "version": 1,
            "fps": 24.0,
            "duration": 217 / 24,
            "envelopes": [{
                "slot": 1,
                "source": "beat_grid",
                "prompt": "Pulse outward.",
                "weights": [float(index % 2) for index in range(217)],
            }],
        }

        plan = timeline.FL_MiniMaxH3BeatShotPlanner.execute(
            clip=FakeClip(),
            vae=FakeVideoVAE(),
            audio_vae=FakeAudioVAE(),
            prompt_schedule=schedule(),
            timeline_audio=audio,
            global_prompt="Same character.",
            width=64,
            height=64,
            affect_audio="video only",
            ref_image_size="match",
            prompt_envelopes=prompt_envelopes,
        ).result[0]

        self.assertEqual(len(plan["shots"]), 4)
        for shot in plan["shots"]:
            envelopes = shot["timeline"]["prompt_envelopes"]
            self.assertEqual(len(envelopes), 1)
            self.assertEqual(envelopes[0]["prompt"], "Pulse outward.")
            self.assertEqual(len(envelopes[0]["weights"]), shot["render_frames"])

    def test_current_timeline_becomes_four_independent_h3_shots(self):
        audio_vae = FakeAudioVAE()
        audio = {
            "waveform": torch.zeros((1, 2, 217000)),
            "sample_rate": 24000,
        }

        plan = timeline.FL_MiniMaxH3BeatShotPlanner.execute(
            clip=FakeClip(),
            vae=FakeVideoVAE(),
            audio_vae=audio_vae,
            prompt_schedule=schedule(),
            timeline_audio=audio,
            global_prompt="Same character.",
            width=64,
            height=64,
            affect_audio="video only",
            ref_image_size="match",
        ).result[0]

        self.assertEqual(plan["total_frames"], 217)
        self.assertEqual(plan["total_render_frames"], 224)
        self.assertEqual(
            [shot["authored_frames"] for shot in plan["shots"]],
            [54, 55, 54, 54],
        )
        self.assertEqual(
            [shot["render_frames"] for shot in plan["shots"]],
            [56, 56, 56, 56],
        )
        self.assertEqual(audio_vae.encoded_samples, [54000, 55000, 54000, 54000])
        for shot in plan["shots"]:
            self.assertIsInstance(
                shot["latent"]["samples"],
                comfy.nested_tensor.NestedTensor,
            )
            self.assertEqual(len(shot["latent"]["samples"].unbind()), 2)
            self.assertEqual(shot["timeline"]["type"], "minimax_h3_prompt_timeline")
            self.assertEqual(
                shot["timeline"]["global_conditioning"][0][1]["minimax_visual_cond_noise_aug"],
                timeline.VISUAL_COND_TIMESTEP,
            )

    def test_visual_condition_fidelity_reaches_every_conditioning_path(self):
        frame_count = 5
        prompt_envelopes = {
            "type": "fl_prompt_envelope_set",
            "version": 1,
            "fps": 24.0,
            "duration": frame_count / 24,
            "envelopes": [{
                "slot": 1,
                "source": "beat_grid",
                "prompt": "Pulse outward.",
                "weights": [1.0] * frame_count,
            }],
        }
        plan = timeline.FL_MiniMaxH3BeatShotPlanner.execute(
            clip=FakeClip(),
            vae=FakeVideoVAE(),
            audio_vae=FakeAudioVAE(),
            prompt_schedule=schedule((0, frame_count)),
            timeline_audio={
                "waveform": torch.zeros((1, 2, 5000)),
                "sample_rate": 24000,
            },
            global_prompt="Same character.",
            width=64,
            height=64,
            affect_audio="video only",
            ref_image_size="match",
            visual_condition_fidelity=0.8,
            prompt_envelopes=prompt_envelopes,
            ref_images={"ref_image_1": torch.zeros((1, 64, 64, 3))},
        ).result[0]

        shot = plan["shots"][0]
        timeline_object = shot["timeline"]
        conditionings = [
            timeline_object["global_conditioning"],
            *(group["conditioning"] for group in timeline_object["conditioning_groups"]),
            *(group["conditioning"] for group in timeline_object["prompt_envelope_groups"]),
            shot["conditioning"],
        ]
        resized_latent, _ = timeline.minimax_h3._empty_av_latent(96, 64, frame_count)
        conditionings.append(timeline._apply_timeline(timeline_object, resized_latent))

        expected = timeline.VISUAL_COND_TIMESTEP * 0.8
        for conditioning in conditionings:
            self.assertTrue(conditioning)
            for _, metadata in conditioning:
                self.assertEqual(metadata["minimax_visual_cond_noise_aug"], expected)
        self.assertIn("minimax_refs", timeline_object["global_conditioning"][0][1])

    def test_planner_builds_an_audio_preserving_visual_reference_free_baseline(self):
        clip = FakeClip()
        reference_audio = {
            "waveform": torch.zeros((1, 2, 24000)),
            "sample_rate": 24000,
        }
        plan = timeline.FL_MiniMaxH3BeatShotPlanner.execute(
            clip=clip,
            vae=FakeVideoVAE(),
            audio_vae=FakeAudioVAE(),
            prompt_schedule=schedule((0, 5)),
            timeline_audio={
                "waveform": torch.zeros((1, 2, 5000)),
                "sample_rate": 24000,
            },
            global_prompt="Same character.",
            width=64,
            height=64,
            affect_audio="video only",
            ref_image_size="match",
            visual_reference_mode="full",
            reference_strength=0.35,
            ref_images={"ref_image_1": torch.zeros((1, 64, 64, 3))},
            ref_audios={"ref_audio_1": reference_audio},
        ).result[0]

        shot = plan["shots"][0]
        selected = shot["timeline"]["global_conditioning"][0][1]["minimax_refs"]
        baseline = shot["timeline"]["reference_free_global_conditioning"][0][1]["minimax_refs"]
        self.assertEqual([block["kind"] for block in selected], ["image", "audio", "audio"])
        self.assertEqual([block["kind"] for block in baseline], ["audio", "audio"])
        self.assertTrue(shot["has_visual_references"])
        self.assertEqual(shot["reference_strength"], 0.35)
        self.assertEqual(plan["visual_reference_mode"], "full")
        self.assertIn("reference_free_conditioning", shot)
        self.assertIn(["image", "audio", "audio"], clip.tokenized_ref_types)
        self.assertIn(["audio", "audio"], clip.tokenized_ref_types)

    def test_reuses_static_references_and_repeated_conditioning(self):
        value = schedule()
        for section in value["sections"]:
            section["prompt"] = "Repeated action."
        clip = FakeClip()
        vae = FakeVideoVAE()
        audio_vae = FakeAudioVAE()
        timeline_audio = {
            "waveform": torch.zeros((1, 2, 217000)),
            "sample_rate": 24000,
        }
        reference_audio = {
            "waveform": torch.zeros((1, 2, 24000)),
            "sample_rate": 24000,
        }

        plan = timeline.FL_MiniMaxH3BeatShotPlanner.execute(
            clip=clip,
            vae=vae,
            audio_vae=audio_vae,
            prompt_schedule=value,
            timeline_audio=timeline_audio,
            global_prompt="Same character.",
            width=64,
            height=64,
            affect_audio="video only",
            ref_image_size="match",
            ref_images={"ref_image_1": torch.zeros((1, 64, 64, 3))},
            ref_audios={"ref_audio_1": reference_audio},
        ).result[0]

        self.assertEqual(len(vae.encoded_images), 1)
        self.assertEqual(audio_vae.encoded_samples, [54000, 24000, 55000, 54000, 54000])
        self.assertEqual(
            clip.encoded_prompts,
            [
                "Same character.",
                "Same character.\n\nRepeated action.",
                "Same character.",
                "Same character.\n\nRepeated action.",
            ],
        )
        refs = [shot["timeline"]["global_conditioning"][0][1]["minimax_refs"] for shot in plan["shots"]]
        self.assertTrue(all(items[0] is refs[0][0] for items in refs))
        self.assertTrue(all(items[2] is refs[0][2] for items in refs))
        self.assertEqual(len({id(items[1]) for items in refs}), 4)

    def test_timeline_groups_build_arbitrary_shared_context_render_units(self):
        audio_vae = FakeAudioVAE()
        audio = {
            "waveform": torch.zeros((1, 2, 313000)),
            "sample_rate": 24000,
        }

        plan = timeline.FL_MiniMaxH3BeatShotPlanner.execute(
            clip=FakeClip(),
            vae=FakeVideoVAE(),
            audio_vae=audio_vae,
            prompt_schedule=schedule(
                (0, 33, 51, 117, 183, 251, 313),
                (1, 1, 2, 2, None, None),
            ),
            timeline_audio=audio,
            global_prompt="Same character.",
            width=64,
            height=64,
            affect_audio="video only",
            ref_image_size="match",
        ).result[0]

        self.assertNotIn("render_mode", plan)
        self.assertEqual(
            [(shot["start_frame"], shot["end_frame"]) for shot in plan["shots"]],
            [(0, 51), (51, 183), (183, 251), (251, 313)],
        )
        self.assertEqual(
            [len(shot["timeline"]["authored_sections"]) for shot in plan["shots"]],
            [2, 2, 1, 1],
        )
        self.assertEqual(
            [shot["authored_frames"] for shot in plan["shots"]],
            [51, 132, 68, 62],
        )
        self.assertEqual(audio_vae.encoded_samples, [51000, 132000, 68000, 62000])
        self.assertEqual(
            [
                round(section["start"] * 24)
                for section in plan["shots"][1]["timeline"]["authored_sections"]
            ],
            [0, 66],
        )
        self.assertEqual(
            [
                round(section["end"] * 24)
                for section in plan["shots"][1]["timeline"]["authored_sections"]
            ],
            [66, 132],
        )
        self.assertTrue(plan["shots"][0]["timeline"]["boundary_adjustments"])

    def test_render_groups_can_appear_anywhere_in_the_timeline(self):
        sections, _ = timeline._beat_shot_sections(
            schedule(
                (0, 34, 68, 102, 136, 170, 204),
                (None, 1, 1, None, 2, 2),
            )
        )

        groups = timeline._render_section_groups(sections)

        self.assertEqual([len(group) for group in groups], [1, 2, 1, 2])
        self.assertEqual(
            [(group[0]["start_frame"], group[-1]["end_frame"]) for group in groups],
            [(0, 34), (34, 102), (102, 136), (136, 204)],
        )

    def test_missing_group_metadata_preserves_one_render_per_prompt_section(self):
        sections, _ = timeline._beat_shot_sections(schedule())

        groups = timeline._render_section_groups(sections)

        self.assertEqual(len(groups), len(sections))
        self.assertTrue(all(len(group) == 1 for group in groups))

    def test_nonconsecutive_or_invalid_render_groups_are_rejected(self):
        sections, _ = timeline._beat_shot_sections(
            schedule(render_groups=(1, None, 1, None))
        )
        with self.assertRaisesRegex(ValueError, "consecutive"):
            timeline._render_section_groups(sections)

        value = schedule()
        value["sections"][1]["render_group"] = "first"
        with self.assertRaisesRegex(ValueError, "invalid render group"):
            timeline._beat_shot_sections(value)

    def test_crossfade_is_rejected_for_independent_shots(self):
        value = schedule((0, 54, 109))
        value["sections"][1]["crossfade_start"] = 50 / 24
        value["sections"][1]["crossfade_end"] = 58 / 24

        with self.assertRaisesRegex(ValueError, "Remove the crossfade"):
            timeline._beat_shot_sections(value)

    def test_gap_is_rejected_instead_of_silently_changing_duration(self):
        value = schedule((0, 54, 109))
        value["sections"][1]["start"] = 55 / 24
        value["sections"][1]["fade_in_end"] = 55 / 24
        value["sections"][1]["start_frame"] = 55
        value["sections"][1]["crossfade_start"] = 55 / 24
        value["sections"][1]["crossfade_end"] = 55 / 24

        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            timeline._beat_shot_sections(value)

    def test_audio_slice_is_a_copy_with_exact_frame_boundaries(self):
        waveform = torch.arange(12000, dtype=torch.float32).reshape(1, 1, -1)
        shot = timeline._shot_audio(
            {"waveform": waveform, "sample_rate": 24000},
            5,
            10,
        )

        self.assertEqual(shot["waveform"].shape[-1], 5000)
        self.assertEqual(shot["waveform"][0, 0, 0], waveform[0, 0, 5000])
        shot["waveform"].zero_()
        self.assertNotEqual(waveform[0, 0, 5000], 0)

    def test_schema_marks_schedule_audio_optional_and_exposes_render_outputs(self):
        schema = timeline.FL_MiniMaxH3BeatShotPlanner.define_schema()
        inputs = {value.id: value for value in schema.inputs}
        self.assertTrue(inputs["prompt_schedule"].optional)
        self.assertTrue(inputs["timeline_audio"].optional)
        self.assertIn("length", inputs)
        self.assertEqual(inputs["length"].default, 124)
        self.assertEqual(
            [value.display_name for value in schema.outputs],
            ["shot_plan", "scheduled", "latent", "semantic"],
        )
        outputs = {value.display_name: value for value in schema.outputs}
        self.assertEqual(outputs["scheduled"].io_type, "CONDITIONING")
        self.assertEqual(outputs["latent"].io_type, "LATENT")
        self.assertEqual(outputs["semantic"].io_type, "CONDITIONING")

    def test_manual_single_render_plans_without_schedule_or_audio(self):
        clip = FakeClip()
        plan, scheduled, latent, semantic = timeline.FL_MiniMaxH3BeatShotPlanner.execute(
            clip=clip,
            vae=FakeVideoVAE(),
            audio_vae=FakeAudioVAE(),
            prompt_schedule=None,
            timeline_audio=None,
            length=124,
            global_prompt="A single shot.",
            width=64,
            height=64,
            affect_audio="video only",
            ref_image_size="match",
        ).result

        self.assertEqual(plan["total_frames"], 124)
        self.assertEqual(len(plan["shots"]), 1)
        shot = plan["shots"][0]
        self.assertEqual(shot["start_frame"], 0)
        self.assertEqual(shot["end_frame"], 124)
        self.assertEqual(shot["prompt"], "A single shot.")
        self.assertIsInstance(latent["samples"], comfy.nested_tensor.NestedTensor)
        self.assertIs(scheduled, shot["conditioning"])
        self.assertIs(latent, shot["latent"])
        self.assertTrue(semantic)
        self.assertEqual(clip.encoded_prompts[0], "A single shot.")

    def test_schedule_mode_outputs_first_render_values(self):
        plan, scheduled, latent, semantic = timeline.FL_MiniMaxH3BeatShotPlanner.execute(
            clip=FakeClip(),
            vae=FakeVideoVAE(),
            audio_vae=FakeAudioVAE(),
            prompt_schedule=schedule((0, 54, 109)),
            timeline_audio={
                "waveform": torch.zeros((1, 2, 109000)),
                "sample_rate": 24000,
            },
            global_prompt="Same character.",
            width=64,
            height=64,
            affect_audio="video only",
            ref_image_size="match",
        ).result

        self.assertEqual(len(plan["shots"]), 2)
        first = plan["shots"][0]
        self.assertIs(scheduled, first["conditioning"])
        self.assertIs(latent, first["latent"])
        self.assertIs(semantic, first["conditioning"])


class BeatKSamplerTests(unittest.TestCase):
    def test_outputs_a_native_latent_list(self):
        output = sampler.FL_MiniMaxH3BeatKSampler.define_schema().outputs[0]

        self.assertEqual(output.io_type, "LATENT")
        self.assertTrue(output.is_output_list)

    def test_samples_every_shot_with_incremented_seeds(self):
        calls = []
        shots = [
            {
                "index": index,
                "start_frame": index * 5,
                "end_frame": (index + 1) * 5,
                "authored_frames": 5,
                "render_frames": 5,
                "conditioning": f"conditioning-{index}",
                "latent": {"samples": f"latent-{index}"},
            }
            for index in range(3)
        ]
        shots[0]["reshot"] = {"version": 1, "selection_offset": 3}
        plan = {
            "type": "minimax_h3_beat_shot_plan",
            "version": 1,
            "fps": 24,
            "total_frames": 15,
            "shots": shots,
        }

        def sample(*args, **kwargs):
            calls.append((args, kwargs))
            return ({"samples": f"sampled-{len(calls)}"},)

        with mock.patch.object(sampler.nodes, "common_ksampler", side_effect=sample):
            output = sampler.FL_MiniMaxH3BeatKSampler.execute(
                model=object(),
                shot_plan=plan,
                seed=100,
                seed_mode="increment",
                steps=20,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
            ).result[0]

        self.assertEqual(
            [latent["fl_h3_shot"]["seed"] for latent in output],
            [100, 101, 102],
        )
        self.assertEqual(
            [latent["samples"] for latent in output],
            ["sampled-1", "sampled-2", "sampled-3"],
        )
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            output[0]["fl_h3_shot"]["reshot"],
            {"version": 1, "selection_offset": 3},
        )
        for index, (args, kwargs) in enumerate(calls):
            self.assertEqual(args[6], f"conditioning-{index}")
            self.assertEqual(args[7], f"conditioning-{index}")
            self.assertEqual(kwargs["denoise"], 1.0)

    def test_visual_reference_strength_guides_against_the_reference_free_conditioning(self):
        shot = {
            "index": 0,
            "start_frame": 0,
            "end_frame": 5,
            "authored_frames": 5,
            "render_frames": 5,
            "conditioning": "selected",
            "reference_free_conditioning": "visual-off",
            "has_visual_references": True,
            "reference_strength": 0.25,
            "latent": {"samples": "latent"},
        }
        plan = {
            "type": "minimax_h3_beat_shot_plan",
            "version": 1,
            "fps": 24,
            "total_frames": 5,
            "shots": [shot],
        }

        with mock.patch.object(
            sampler.nodes,
            "common_ksampler",
            return_value=({"samples": "sampled"},),
        ) as sample:
            sampler.FL_MiniMaxH3BeatKSampler.execute(
                model=object(),
                shot_plan=plan,
                seed=1,
                seed_mode="fixed",
                steps=20,
                cfg=2.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
            )

        args = sample.call_args.args
        self.assertEqual(args[3], 0.5)
        self.assertEqual(args[6], "selected")
        self.assertEqual(args[7], "visual-off")

    def test_zero_visual_reference_strength_samples_only_the_reference_free_path(self):
        positive, negative, cfg = sampler._reference_guidance(
            "selected",
            "visual-off",
            2.0,
            0.0,
            True,
        )

        self.assertEqual((positive, negative, cfg), ("visual-off", "visual-off", 1.0))

    def test_sampling_error_names_the_shot_and_frame_range(self):
        plan = {
            "type": "minimax_h3_beat_shot_plan",
            "version": 1,
            "shots": [{
                "start_frame": 10,
                "end_frame": 20,
                "conditioning": [],
                "latent": {"samples": "latent"},
            }],
        }
        with mock.patch.object(
            sampler.nodes,
            "common_ksampler",
            side_effect=RuntimeError("model failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "frames 10-19"):
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


def upscale_latent(width=288, height=512):
    video = torch.zeros((1, 24, 2, height // 16, width // 16))
    audio = torch.randn((1, 32, 2, 10))
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "fl_h3_shot": {
            "version": 1,
            "index": 0,
            "start_frame": 0,
            "end_frame": 5,
            "authored_frames": 5,
            "render_frames": 5,
            "total_frames": 5,
            "fps": 24.0,
            "seed": 1,
        },
    }


def upscale_plan():
    return {
        "type": "minimax_h3_beat_shot_plan",
        "version": 1,
        "fps": 24,
        "total_frames": 5,
        "shots": [{
            "index": 0,
            "start_frame": 0,
            "end_frame": 5,
            "authored_frames": 5,
            "render_frames": 5,
            "conditioning": "source-conditioning",
            "timeline": {"type": "minimax_h3_prompt_timeline"},
            "latent": {"samples": "source-latent"},
        }],
    }


class FakePixelVAE:
    def __init__(self, temporal_offset=0):
        self.temporal_offset = temporal_offset
        self.decode_input = None
        self.encode_input = None

    def decode(self, video):
        self.decode_input = video
        latent_t = video.shape[2]
        frame_count = 5 if latent_t == 2 else ((latent_t - 2) // 5) * 17 + 5
        return torch.zeros((
            video.shape[0],
            frame_count,
            video.shape[-2] * 16,
            video.shape[-1] * 16,
            3,
        ))

    def encode(self, images):
        self.encode_input = images
        return torch.zeros((
            1,
            24,
            self.decode_input.shape[2] + self.temporal_offset,
            images.shape[-3] // 16,
            images.shape[-2] // 16,
        ))


class BeatUpscaleKSamplerTests(unittest.TestCase):
    def test_schema_uses_vae_long_side_and_native_latents(self):
        schema = sampler.FL_MiniMaxH3BeatUpscaleKSampler.define_schema()

        self.assertEqual(schema.inputs[2].io_type, "LATENT")
        self.assertEqual(schema.inputs[3].io_type, "VAE")
        self.assertEqual(schema.inputs[4].io_type, "INT")
        self.assertEqual(schema.inputs[8].id, "steps")
        self.assertEqual(schema.inputs[9].id, "start_at_step")
        self.assertEqual(schema.inputs[10].id, "end_at_step")
        self.assertNotIn("denoise", [value.id for value in schema.inputs])
        self.assertEqual(schema.outputs[0].io_type, "LATENT")
        self.assertFalse(schema.outputs[0].is_output_list)

    def test_target_long_side_preserves_orientation_and_proportions(self):
        self.assertEqual(sampler._target_canvas(288, 512, 512), (288, 512))
        self.assertEqual(sampler._target_canvas(288, 512, 896), (512, 896))
        self.assertEqual(sampler._target_canvas(288, 512, 1024), (576, 1024))
        self.assertEqual(sampler._target_canvas(288, 512, 1344), (768, 1344))
        self.assertEqual(sampler._target_canvas(512, 288, 1024), (1024, 576))
        self.assertEqual(sampler._target_canvas(512, 512, 896), (896, 896))

    def test_target_long_side_requires_h3_pixel_alignment(self):
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            sampler._target_canvas(288, 512, 900)

    def test_rejects_the_pre_context_plan_before_decoding(self):
        value = upscale_latent()
        value["fl_h3_shot"]["motion_context"] = {
            "version": 1,
            "source_index": None,
            "video_frames": 0,
            "video_steps": 0,
            "audio_frames": 0,
            "trim_frames": 0,
        }
        vae = FakePixelVAE()

        with self.assertRaisesRegex(ValueError, "connect its output to both samplers"):
            sampler.FL_MiniMaxH3BeatUpscaleKSampler.execute(
                model=object(),
                shot_plan=upscale_plan(),
                latent=value,
                vae=vae,
                target_long_side=896,
                upscale_method="bicubic",
                seed=100,
                seed_mode="increment",
                steps=16,
                start_at_step=12,
                end_at_step=16,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
            )

        self.assertIsNone(vae.decode_input)

    def test_pixel_round_trip_changes_only_video_spatial_dimensions(self):
        value = upscale_latent()
        source_video, source_audio = value["samples"].unbind()
        vae = FakePixelVAE()

        output, source_width, source_height, width, height = sampler._pixel_upscale_latent(
            value,
            vae,
            896,
            "bicubic",
        )
        video, audio = output["samples"].unbind()

        self.assertEqual((source_width, source_height), (288, 512))
        self.assertEqual((width, height), (512, 896))
        self.assertEqual(video.shape, (1, 24, 2, 56, 32))
        self.assertEqual(source_video.shape, (1, 24, 2, 32, 18))
        self.assertEqual(vae.decode_input.data_ptr(), source_video.data_ptr())
        self.assertEqual(vae.encode_input.shape, (5, 896, 512, 3))
        self.assertEqual(audio.data_ptr(), source_audio.data_ptr())
        self.assertEqual(output["fl_h3_shot"], value["fl_h3_shot"])
        self.assertIsNot(output, value)

    def test_refines_video_only_with_rebuilt_conditioning(self):
        value = upscale_latent()
        vae = FakePixelVAE()
        calls = []

        def sample(*args, **kwargs):
            calls.append((args, kwargs))
            return (args[8].copy(),)

        with (
            mock.patch.object(sampler, "_apply_timeline", return_value="target-conditioning") as apply,
            mock.patch.object(sampler.nodes, "common_ksampler", side_effect=sample),
        ):
            output = sampler.FL_MiniMaxH3BeatUpscaleKSampler.execute(
                model=object(),
                shot_plan=upscale_plan(),
                latent=value,
                vae=vae,
                target_long_side=896,
                upscale_method="bicubic",
                seed=100,
                seed_mode="increment",
                steps=16,
                start_at_step=12,
                end_at_step=16,
                cfg=1.0,
                sampler_name="euler",
                scheduler="normal",
            ).result[0]

        resized = apply.call_args.args[1]
        video, audio = resized["samples"].unbind()
        self.assertEqual(video.shape[-2:], (56, 32))
        self.assertEqual(audio.data_ptr(), value["samples"].unbind()[1].data_ptr())
        args, kwargs = calls[0]
        self.assertEqual(args[1], 100)
        self.assertEqual(args[6], "target-conditioning")
        self.assertEqual(args[7], "target-conditioning")
        self.assertEqual(kwargs["denoise"], 1.0)
        self.assertEqual(kwargs["start_step"], 12)
        self.assertEqual(kwargs["last_step"], 16)
        self.assertTrue(kwargs["force_full_denoise"])
        video_mask, audio_mask = args[8]["noise_mask"].unbind()
        self.assertTrue(torch.all(video_mask == 1))
        self.assertTrue(torch.all(audio_mask == 0))
        self.assertNotIn("noise_mask", output)
        self.assertEqual(output["fl_h3_shot"], value["fl_h3_shot"])

    def test_pixel_refinement_uses_the_same_visual_reference_guidance(self):
        value = upscale_latent()
        plan = upscale_plan()
        plan["shots"][0].update({
            "has_visual_references": True,
            "reference_strength": 0.25,
        })

        def conditioning(_timeline, _latent, reference_free=False):
            return "visual-off" if reference_free else "selected"

        with (
            mock.patch.object(sampler, "_apply_timeline", side_effect=conditioning),
            mock.patch.object(
                sampler.nodes,
                "common_ksampler",
                side_effect=lambda *args, **kwargs: (args[8].copy(),),
            ) as sample,
        ):
            sampler.FL_MiniMaxH3BeatUpscaleKSampler.execute(
                model=object(),
                shot_plan=plan,
                latent=value,
                vae=FakePixelVAE(),
                target_long_side=896,
                upscale_method="bicubic",
                seed=100,
                seed_mode="fixed",
                steps=16,
                start_at_step=12,
                end_at_step=16,
                cfg=2.0,
                sampler_name="euler",
                scheduler="normal",
            )

        args = sample.call_args.args
        self.assertEqual(args[3], 0.5)
        self.assertEqual(args[6], "selected")
        self.assertEqual(args[7], "visual-off")

    def test_rejects_a_target_smaller_than_the_decoded_video(self):
        with self.assertRaisesRegex(ValueError, "smaller than"):
            sampler._pixel_upscale_latent(
                upscale_latent(512, 896),
                FakePixelVAE(),
                512,
                "bicubic",
            )

    def test_rejects_a_vae_that_changes_temporal_latent_length(self):
        with self.assertRaisesRegex(ValueError, "temporal latent dimensions"):
            sampler._pixel_upscale_latent(
                upscale_latent(),
                FakePixelVAE(temporal_offset=1),
                896,
                "bicubic",
            )


class FakeDecodeVAE:
    def __init__(self):
        self.calls = 0

    def decode(self, video):
        self.calls += 1
        latent_t = video.shape[2]
        frame_count = 5 if latent_t == 2 else ((latent_t - 2) // 5) * 17 + 5
        return torch.full((1, frame_count, 4, 6, 3), float(self.calls))


def sampled_shot(index, start_frame, end_frame, latent_t, total_frames):
    video = torch.zeros((1, 24, latent_t, 4, 6))
    audio = torch.zeros((1, 32, 2, 10))
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "fl_h3_shot": {
            "version": 1,
            "index": index,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "authored_frames": end_frame - start_frame,
            "render_frames": 56,
            "total_frames": total_frames,
            "fps": 24.0,
            "seed": index,
        },
    }


class ShotAssemblerTests(unittest.TestCase):
    def test_consumes_the_complete_latent_list(self):
        schema = assembler.FL_MiniMaxH3ShotAssembler.define_schema()

        self.assertEqual(schema.inputs[0].io_type, "LATENT")
        self.assertTrue(schema.is_input_list)

    def test_decodes_independently_trims_padding_and_assembles_pixels(self):
        latents = [
            sampled_shot(0, 0, 54, 17, 109),
            sampled_shot(1, 54, 109, 17, 109),
        ]
        vae = FakeDecodeVAE()

        images = assembler.FL_MiniMaxH3ShotAssembler.execute(latents, [vae]).result[0]

        self.assertEqual(vae.calls, 2)
        self.assertEqual(images.shape, (109, 4, 6, 3))
        self.assertTrue(torch.all(images[:54] == 1))
        self.assertTrue(torch.all(images[54:] == 2))

    def test_rejects_non_nested_latent(self):
        value = sampled_shot(0, 0, 5, 2, 5)
        value["samples"] = torch.zeros((1, 4, 2, 2))

        with self.assertRaisesRegex(TypeError, "not a nested H3 latent"):
            assembler.FL_MiniMaxH3ShotAssembler.execute([value], [FakeDecodeVAE()])


if __name__ == "__main__":
    unittest.main()
