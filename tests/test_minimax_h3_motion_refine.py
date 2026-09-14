import copy
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import torch

import comfy.latent_formats
import comfy.ldm.minimax.model as h3_model
import comfy.model_patcher
import comfy.nested_tensor
import execution


ROOT = pathlib.Path(__file__).parents[1]
PACKAGE = "fl_h3_motion_refine_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "nodes")]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.FL_MiniMaxH3MotionRefine", ROOT / "nodes/FL_MiniMaxH3MotionRefine.py")
refine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = refine
spec.loader.exec_module(refine)

motion = None
motion_path = ROOT.parent / "ComfyUI-MAINodes/motion.py"
if motion_path.is_file():
    motion_spec = importlib.util.spec_from_file_location("fl_motion_refine_mainodes", motion_path)
    motion = importlib.util.module_from_spec(motion_spec)
    motion_spec.loader.exec_module(motion)


def latent(frames=22, size=2):
    video_t = refine.h3.video_latent_t(frames)
    video = torch.arange(video_t, dtype=torch.float32).view(1, 1, video_t, 1, 1).expand(1, 24, video_t, size, size).clone()
    audio = torch.ones(1, 32, 2, round(frames / 24 * 40))
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def model():
    inner = torch.nn.Module()
    inner.latent_format = comfy.latent_formats.MiniMaxH3AV()
    inner.model_sampling = object()
    return comfy.model_patcher.ModelPatcher(inner, torch.device("cpu"), torch.device("cpu"))


class VideoVAE:
    def encode_h3_frame_sequence(self, images, indices):
        return self.encode(images[indices])

    def decode_h3_selected(self, video, indices):
        return self.decode(video)[:, indices]

    def decode(self, video):
        indices = [i for i in range(video.shape[2]) for _ in range(refine.h3.FRAME_PER_TOKEN[i % 5])]
        return video[0, 0, indices, 0, 0].view(1, -1, 1, 1, 1).expand(1, len(indices), video.shape[-2] * 16, video.shape[-1] * 16, 3).clone()

    def encode(self, images):
        t = 1 if len(images) == 1 else refine.h3.video_latent_t(len(images))
        edges = refine.frame_edges([refine.h3.FRAME_PER_TOKEN[i % 5] for i in range(t)])
        return images[edges[:-1], 0, 0, 0].view(1, 1, t, 1, 1).expand(1, 24, t, images.shape[1] // 16, images.shape[2] // 16).clone()


class AudioVAE:
    audio_sample_rate = 32000

    def decode(self, audio):
        return torch.ones(1, round(audio.shape[-1] / 40 * self.audio_sample_rate), 2)

    def encode(self, waveform):
        return torch.ones(1, 32, 2, round(waveform.shape[1] / self.audio_sample_rate * 40))


def run_pipeline(*args, **kwargs):
    expanded = refine.FL_MiniMaxH3MotionRefine.execute(*args, **kwargs).expand
    prep, sample = list(expanded.values())
    prepared = refine.FL_MiniMaxH3MotionPrepare.execute(**prep["inputs"])[0]
    return refine.FL_MiniMaxH3MotionSample.execute(**dict(sample["inputs"], prepared=prepared))


shots_module = sys.modules[f"{PACKAGE}._motion_refine_shots"]


def shot_fixture(lengths=(22, 20), context_frames=22):
    shots, latents, cursor = [], [], 0
    for index, authored in enumerate(lengths):
        trim = context_frames if index else 0
        frames = refine.h3.align_frame_count(trim + authored)
        source = latent(frames)
        context = {"version": 1, "source_index": index - 1 if index else None,
            "video_frames": trim, "video_steps": refine.VIDEO_CONTEXT_STEPS[trim],
            "audio_frames": 0, "trim_frames": trim}
        shot = dict(index=index, start_frame=cursor, end_frame=cursor + authored,
            authored_frames=authored, render_frames=frames, motion_context=context,
            conditioning=[[torch.full((1, 2, 3), float(index + 1)), {}]], latent=source)
        source["fl_h3_shot"] = {key: shot[key] for key in ("index", "start_frame", "end_frame", "authored_frames", "render_frames", "motion_context")}
        source["fl_h3_shot"].update(version=1, total_frames=sum(lengths), fps=24)
        shots.append(shot)
        latents.append(source)
        cursor += authored
    return dict(type="minimax_h3_beat_shot_plan", version=1, fps=24, total_frames=cursor, shots=shots), latents


class MotionShotTests(unittest.TestCase):
    def test_real_layout_reproduces_reported_mismatch_and_accepts_resized_context(self):
        runtime = sys.modules[f"{PACKAGE}._motion_context"]
        reference = {"kind": "image", "latent_h": 24, "latent_w": 24,
            "latent": torch.zeros(1, 24, 1, 24, 24)}
        anchors = [{"resolved_frame_index": 0, refine.VIDEO_FRAME_MARKER: i,
            "latent": torch.zeros(1, 24, 1, 32, 48)} for i in range(7)]
        payload = {"keyframes": anchors, "refs": [reference], "visual_cond_noise_aug": 1.0}
        shapes = [(1, 24, 7, 64, 96), (1, 32, 2, 37)]
        runtime._prepare_payload(payload, torch.zeros(1, 2, 3), shapes)
        rows = h3_model.MiniMaxH3Model._cond_video_rows(types.SimpleNamespace(patch_size=(1, 2, 2)), payload, "cpu")
        count = int((~payload["layout"].img_update).sum())
        self.assertEqual(tuple(rows.shape), (2832, 96))
        self.assertEqual(count, 10896)
        with self.assertRaises(RuntimeError):
            torch.empty(count, 96)[:] = rows
        for anchor in anchors:
            anchor["latent"] = torch.zeros(1, 24, 1, 64, 96)
        payload.pop("layout")
        runtime._prepare_payload(payload, torch.zeros(1, 2, 3), shapes)
        rows = h3_model.MiniMaxH3Model._cond_video_rows(types.SimpleNamespace(patch_size=(1, 2, 2)), payload, "cpu")
        packed = torch.empty(payload["layout"].img_update.shape[0], 96)
        packed[~payload["layout"].img_update] = rows
        self.assertEqual(tuple(rows.shape), (10896, 96))

    def test_image_anchors_resize_once_preserve_timing_and_leave_refs_unchanged(self):
        guide = torch.ones(1, 24, 1, 2, 2)
        reference = {"kind": "image", "latent": guide}
        values = {"minimax_keyframes": [{"resolved_frame_index": 5, "latent": guide}], "minimax_refs": [reference]}
        conditioning = [[torch.ones(1, 2, 3), values], [torch.ones(1, 2, 3), values]]
        vae = mock.Mock(wraps=VideoVAE())
        result = refine.resize_image_anchors(conditioning, vae, 64, 64, "bilinear")
        self.assertEqual(vae.encode.call_count, 1)
        self.assertEqual(result[0][1]["minimax_keyframes"][0]["latent"].shape, (1, 24, 1, 4, 4))
        self.assertEqual(result[0][1]["minimax_keyframes"][0]["resolved_frame_index"], 5)
        self.assertIs(result[0][1]["minimax_refs"][0], reference)
        self.assertIs(values["minimax_keyframes"][0]["latent"], guide)
        vae.reset_mock()
        refine.resize_image_anchors(conditioning, vae, 32, 32, "bilinear")
        vae.decode.assert_not_called()
        vae.encode.assert_not_called()

    def test_eight_shots_recover_461_frames_in_numeric_order(self):
        lengths = [58] * 7 + [55]
        plan, latents = shot_fixture(lengths)
        shots_module.validate_latents(plan, latents)
        images = {f"shot_{i}": torch.full((count, 2, 2, 3), i) for i, count in reversed(list(enumerate(lengths)))}
        reports = {key: key for key in images}
        output, report = shots_module.FL_MiniMaxH3MotionCollect.execute(461, images, reports).result
        self.assertEqual(output.shape[0], 461)
        self.assertTrue(torch.all(output[:58] == 0))
        self.assertTrue(torch.all(output[-55:] == 7))
        self.assertIn("8 shots assembled", report)

    @unittest.skipUnless(motion is not None, "Install ComfyUI-MAINodes")
    def test_zero_context_metadata_is_valid_without_plan(self):
        _, latents = shot_fixture()
        real_ops = {name: getattr(motion, name)() for name in refine._MAINODES}
        with mock.patch.object(refine, "motion_nodes", return_value=real_ops):
            result = refine.FL_MiniMaxH3MotionPrepare.execute(latents[0], VideoVAE(), motion_coverage="off")[0]
        self.assertEqual(result["frames"], 22)

    def test_metadata_must_match_exact_context_plan_and_order(self):
        plan, latents = shot_fixture()
        self.assertEqual(len(shots_module.validate_latents(plan, latents)), 2)
        with self.assertRaisesRegex(ValueError, "does not match"):
            shots_module.validate_latents(plan, latents[::-1])
        wrong = copy.deepcopy(plan)
        wrong["shots"][1]["motion_context"]["audio_frames"] = 5
        with self.assertRaisesRegex(ValueError, "exact plan"):
            shots_module.validate_latents(wrong, latents)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            shots_module.validate_latents(plan, latents[:1])
        with self.assertRaisesRegex(ValueError, "reshot"):
            shots_module.validate_plan(dict(plan, mode="temporal_reshot"))

    def test_shared_assembled_pixels_restore_only_hidden_regions(self):
        plan, latents = shot_fixture()
        baseline = torch.full((42, 32, 32, 3), 99.)
        vae = mock.Mock(wraps=VideoVAE())
        source = latents[1]["samples"].unbind()[0]
        images = shots_module.shot_images(vae, source, plan["shots"][1], baseline, 42)
        torch.testing.assert_close(images[22:42], baseline[22:42])
        self.assertEqual(vae.decode_h3_selected.call_args.args[1], [*range(22), *range(42, 56)])
        vae.decode.assert_not_called()
        self.assertTrue(torch.all(baseline == 99))
        with self.assertRaisesRegex(ValueError, "full assembled"):
            shots_module.shot_images(vae, source, plan["shots"][1], baseline[:22], 42)

    def test_audio_uses_global_shot_slice_and_keeps_native_prefix(self):
        plan, latents = shot_fixture()
        sr = 32000
        baseline = {"sample_rate": sr, "waveform": torch.arange(56000.).view(1, 1, -1).expand(1, 2, -1)}
        audio = shots_module.shot_audio(AudioVAE(), latents[1], plan["shots"][1], baseline, 42)
        offset = round(22 / 24 * sr)
        self.assertEqual(audio["waveform"].shape[-1], round(56 / 24 * sr))
        torch.testing.assert_close(audio["waveform"][..., :offset], torch.ones(1, 2, offset))
        torch.testing.assert_close(audio["waveform"][..., offset:offset + 56000 - offset], baseline["waveform"][..., offset:])
        with self.assertRaisesRegex(ValueError, "audio_vae"):
            shots_module.shot_audio(None, latents[1], plan["shots"][1], baseline, 42)

    def test_context_anchors_retime_without_mutating_source(self):
        source = latent(56)
        holds = [1] * 22 + [2] * 34
        holds[-1] += refine.h3.align_frame_count(sum(holds)) - sum(holds)
        target = latent(sum(holds))
        metadata = {"minimax_keyframes": [{"resolved_frame_index": 0,
            refine.VIDEO_FRAME_MARKER: 18, "latent": torch.ones(1, 24, 1, 2, 2)}],
            "minimax_refs": [{"kind": "audio", refine.AUDIO_END_FRAME_MARKER: 22.2}]}
        cond = [[torch.ones(1, 2, 3), metadata]]
        result = refine.retime_conditioning(cond, source, target, holds, shot_mode=True, context_prefix=22)
        self.assertEqual(result[0][1]["minimax_keyframes"][0][refine.VIDEO_FRAME_MARKER], 18)
        self.assertEqual(result[0][1]["minimax_refs"][0][refine.AUDIO_END_FRAME_MARKER], 22.2)
        with self.assertRaisesRegex(ValueError, "shot_plan"):
            refine.retime_conditioning(cond, source, target, holds)

    @unittest.skipUnless(motion is not None, "Install ComfyUI-MAINodes")
    def test_stretch_holds_preserve_prefix_and_recovery_count(self):
        plan, latents = shot_fixture()
        data = shots_module.FL_MiniMaxH3MotionShot.execute(latents, [plan], [1], [VideoVAE()])[0]
        real_ops = {name: getattr(motion, name)() for name in refine._MAINODES}
        with mock.patch.object(refine, "motion_nodes", return_value=real_ops):
            prepared = refine.FL_MiniMaxH3MotionPrepare.execute(vae=VideoVAE(), audio_vae=AudioVAE(),
                shot_data=data, motion_coverage="uniform")[0]
        self.assertEqual(prepared["holds"][:22], [1] * 22)
        self.assertEqual(prepared["holds"][42:-1], [1] * 13)
        self.assertEqual(sum(prepared["holds"]) % 17, 5)
        self.assertEqual(len(refine.frame_edges(prepared["holds"])[22:42]), 20)

    @unittest.skipUnless(motion is not None, "Install ComfyUI-MAINodes")
    def test_executor_shot_list_prompts_masks_assembly_and_seed_cache(self):
        plan, latents = shot_fixture()
        vae = mock.Mock(wraps=VideoVAE())
        outputs = []
        class Source(refine.io.ComfyNode):
            @classmethod
            def define_schema(cls):
                return refine.io.Schema(node_id="ShotTestSource", inputs=[], outputs=[
                    refine.io.Model.Output(), shots_module.H3ShotPlan.Output(),
                    refine.io.Latent.Output(is_output_list=True), refine.io.Vae.Output()])
            @classmethod
            def execute(cls):
                return refine.io.NodeOutput(model(), plan, latents, vae)
        class Sink(refine.io.ComfyNode):
            @classmethod
            def define_schema(cls):
                return refine.io.Schema(node_id="ShotTestSink", inputs=[refine.io.Image.Input("images"),
                    refine.io.String.Input("report")], outputs=[], is_output_node=True)
            @classmethod
            def execute(cls, images, report):
                outputs.append((images, report))
                return refine.io.NodeOutput()
        registry = {"ShotTestSource": Source, "ShotTestSink": Sink}
        for cls in (refine.FL_MiniMaxH3MotionRefine, refine.FL_MiniMaxH3MotionPrepare,
                refine.FL_MiniMaxH3MotionSample, shots_module.FL_MiniMaxH3MotionShot, shots_module.FL_MiniMaxH3MotionCollect):
            registry[cls.__name__] = cls
        prompt = {"source": {"class_type": "ShotTestSource", "inputs": {}},
            "refine": {"class_type": "FL_MiniMaxH3MotionRefine", "inputs": {
                "model": ["source", 0], "shot_plan": ["source", 1], "latent": ["source", 2],
                "vae": ["source", 3], "motion_coverage": "off", "target_long_side": 64, "seed": 1}},
            "sink": {"class_type": "ShotTestSink", "inputs": {"images": ["refine", 0], "report": ["refine", 1]}}}
        guider = mock.Mock()
        guider.sample.side_effect = lambda noise, samples, *args, **kwargs: samples
        real_ops = {name: getattr(motion, name)() for name in refine._MAINODES}
        with mock.patch.dict(refine.nodes.NODE_CLASS_MAPPINGS, registry), \
             mock.patch.object(refine, "motion_nodes", return_value=real_ops), \
             mock.patch.object(refine, "Guider_Basic", return_value=guider), \
             mock.patch.object(refine.latent_preview, "prepare_callback", return_value=None), \
             mock.patch.object(refine.comfy.samplers, "calculate_sigmas", side_effect=lambda m, s, n: torch.linspace(1, 0, n + 1)):
            executor = execution.PromptExecutor(mock.Mock(), cache_type=execution.CacheType.CLASSIC,
                cache_args={"ram": 0, "ram_inactive": 0})
            for seed in (1, 2):
                prompt["refine"]["inputs"]["seed"] = seed
                executor.execute(copy.deepcopy(prompt), str(seed), execute_outputs=["sink"])
                self.assertTrue(executor.success, executor.status_messages)
        self.assertEqual(vae.encode_h3_frame_sequence.call_count, 2)
        self.assertEqual(guider.sample.call_count, 4)
        self.assertEqual(outputs[-1][0].shape, (42, 64, 64, 3))
        self.assertIn("2 shots assembled into 42 frames", outputs[-1][1])
        prompts = [call.args[0][0][0].mean().item() for call in guider.set_conds.call_args_list]
        self.assertEqual(sorted(prompts), [1., 1., 2., 2.])
        for call, condition in zip(guider.sample.call_args_list, guider.set_conds.call_args_list):
            if condition.args[0][0][0].mean() == 2:
                values = condition.args[0][0][1]
                anchors = values["minimax_keyframes"]
                self.assertEqual(len(anchors), 7)
                layout = h3_model.PackedLayout(2, 17, 4, 4, 93, keyframes=anchors)
                rows = h3_model.MiniMaxH3Model._cond_video_rows(types.SimpleNamespace(patch_size=(1, 2, 2)),
                    {"cond_video_latents": [a["latent"] for a in anchors], "visual_cond_noise_aug": 1.0}, "cpu")
                packed = torch.empty(layout.img_update.shape[0], 96)
                packed[~layout.img_update] = rows
                self.assertTrue(all(a["latent"].shape[-2:] == (4, 4) for a in anchors))
                vm, am = call.kwargs["denoise_mask"].unbind()
                self.assertTrue(torch.all(vm[:, :, :7] == 0))
                self.assertTrue(torch.all(vm[:, :, 7:] == 1))
                self.assertTrue(torch.all(am[..., :37] == 0))
                self.assertTrue(torch.all(am[..., 37:] == .5))


class MotionRefineTests(unittest.TestCase):
    @unittest.skipUnless(motion is not None, "Install ComfyUI-MAINodes for integration tests")
    def test_executor_reuses_preparation_across_seed_changes(self):
        vae = mock.Mock(wraps=VideoVAE())
        class Source(refine.io.ComfyNode):
            @classmethod
            def define_schema(cls):
                return refine.io.Schema(node_id="MotionTestSource", inputs=[], outputs=[
                    refine.io.Model.Output(), refine.io.Conditioning.Output(),
                    refine.io.Latent.Output(), refine.io.Vae.Output()])
            @classmethod
            def execute(cls):
                return refine.io.NodeOutput(model(), [[torch.ones(1, 2, 3), {}]], latent(), vae)
        class Sink(refine.io.ComfyNode):
            @classmethod
            def define_schema(cls):
                return refine.io.Schema(node_id="MotionTestSink", inputs=[refine.io.Image.Input("images")], outputs=[], is_output_node=True)
            @classmethod
            def execute(cls, images):
                return refine.io.NodeOutput()
        registry = {"MotionTestSource": Source, "MotionTestSink": Sink}
        for cls in (refine.FL_MiniMaxH3MotionRefine, refine.FL_MiniMaxH3MotionPrepare, refine.FL_MiniMaxH3MotionSample):
            registry[cls.__name__] = cls
        prompt = {"source": {"class_type": "MotionTestSource", "inputs": {}},
            "refine": {"class_type": "FL_MiniMaxH3MotionRefine", "inputs": {
                "model": ["source", 0], "positive": ["source", 1], "latent": ["source", 2],
                "vae": ["source", 3], "motion_coverage": "off", "seed": 1}},
            "sink": {"class_type": "MotionTestSink", "inputs": {"images": ["refine", 0]}}}
        guider = mock.Mock()
        guider.sample.side_effect = lambda noise, samples, *args, **kwargs: samples
        real_ops = {name: getattr(motion, name)() for name in refine._MAINODES}
        with mock.patch.dict(refine.nodes.NODE_CLASS_MAPPINGS, registry), \
             mock.patch.object(refine, "motion_nodes", return_value=real_ops), \
             mock.patch.object(refine, "Guider_Basic", return_value=guider), \
             mock.patch.object(refine.latent_preview, "prepare_callback", return_value=None), \
             mock.patch.object(refine.comfy.samplers, "calculate_sigmas", side_effect=lambda m, s, n: torch.linspace(1, 0, n + 1)):
            executor = execution.PromptExecutor(mock.Mock(), cache_type=execution.CacheType.CLASSIC,
                cache_args={"ram": 0, "ram_inactive": 0})
            for seed in (1, 2):
                prompt["refine"]["inputs"]["seed"] = seed
                executor.execute(copy.deepcopy(prompt), str(seed), execute_outputs=["sink"])
                self.assertTrue(executor.success, executor.status_messages)
            self.assertEqual(vae.encode_h3_frame_sequence.call_count, 1)
            self.assertEqual(guider.sample.call_count, 2)
            prompt["refine"]["inputs"]["target_long_side"] = 64
            executor.execute(copy.deepcopy(prompt), "resize", execute_outputs=["sink"])
            self.assertTrue(executor.success, executor.status_messages)
            self.assertEqual(vae.encode_h3_frame_sequence.call_count, 2)

    def test_sampling_changes_do_not_change_preparation_inputs(self):
        base = dict(model=["model", 0], positive=["cond", 0], latent=["source", 0], vae=["vae", 0])
        first = refine.FL_MiniMaxH3MotionRefine.execute(**base).expand
        second = refine.FL_MiniMaxH3MotionRefine.execute(**base, seed=23, steps=10,
            strength=0.8, sampler_name="sa_solver", scheduler="beta57", audio_strength=0.25).expand
        self.assertEqual(list(first.values())[0], list(second.values())[0])
        changed = refine.FL_MiniMaxH3MotionRefine.execute(**base, target_long_side=1024).expand
        self.assertNotEqual(list(first.values())[0], list(changed.values())[0])

    @unittest.skipUnless(motion is not None, "Install ComfyUI-MAINodes for integration tests")
    def test_shared_decode_is_reused_and_preparation_is_not_mutated(self):
        source = latent()
        vae = mock.Mock(wraps=VideoVAE())
        images = VideoVAE().decode(source["samples"].unbind()[0])[0]
        originals = images.clone()
        real_ops = {name: getattr(motion, name)() for name in refine._MAINODES}
        with mock.patch.object(refine, "motion_nodes", return_value=real_ops):
            prepared = refine.FL_MiniMaxH3MotionPrepare.execute(source, vae,
                motion_coverage="off", baseline_images=images)[0]
        vae.decode.assert_not_called()
        torch.testing.assert_close(images, originals, rtol=0, atol=0)
        before = prepared["video"]["samples"].clone()
        real_ops["H3V2VInit"].build(prepared["video"], audio_latent=prepared["audio"],
            audio_strength=0.5, audio_mode="custom (use audio_strength)")
        torch.testing.assert_close(prepared["video"]["samples"], before, rtol=0, atol=0)

    def test_schema_has_six_main_widgets_and_no_passthrough_outputs(self):
        schema = refine.FL_MiniMaxH3MotionRefine.define_schema()
        main = [value.id for value in schema.inputs if value.io_type in ("INT", "FLOAT", "COMBO", "BOOLEAN") and not value.advanced]
        self.assertEqual(main, ["target_long_side", "strength", "motion_coverage", "steps", "context_budget", "seed"])
        self.assertEqual([value.io_type for value in schema.outputs], ["IMAGE", "STRING"])

    def test_missing_mainodes_is_actionable_without_import_failure(self):
        with mock.patch.dict(refine.nodes.NODE_CLASS_MAPPINGS, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ComfyUI-MAINodes"):
                refine.motion_nodes()

    @unittest.skipUnless(motion is not None, "Install ComfyUI-MAINodes for integration tests")
    def test_exact_recovery_matches_original_frame_selection(self):
        frames = torch.arange(22).view(22, 1, 1, 1).float()
        holds = [1, 2, 3, 4] * 5 + [1, 1]
        stretched, used, count, _ = motion.H3TimeSmear().smear(frames, 1, refine.json.dumps({"holds": holds}), expand_to_end=True)
        self.assertEqual(count % 17, 5)
        recovered = motion.H3ExactRecover().recover(stretched, used)[0]
        torch.testing.assert_close(recovered, frames)

    def test_anchor_mapping_and_source_conditioning_are_preserved(self):
        source = latent()
        target = latent(39)
        anchor = {"resolved_frame_index": 5, "latent": torch.ones(1, 24, 1, 2, 2)}
        reference = {"kind": "image", "latent": torch.ones(1, 24, 1, 2, 2)}
        cond = [[torch.ones(1, 2, 3), {"minimax_keyframes": [anchor], "minimax_refs": [reference], "minimax_frame_count": 22}]]
        holds = [2] * 17 + [1] * 5
        mapped = refine.retime_conditioning(cond, source, target, holds)
        self.assertEqual(mapped[0][1]["minimax_keyframes"][0]["resolved_frame_index"], 10)
        self.assertEqual(mapped[0][1]["minimax_frame_count"], 39)
        self.assertEqual(anchor["resolved_frame_index"], 5)
        self.assertIs(mapped[0][1]["minimax_refs"][0], reference)
        self.assertIs(mapped[0][0], cond[0][0])

    def test_temporal_mask_identity_resize_and_retime(self):
        source = latent()
        sv, sa = source["samples"].unbind()
        target = latent(39, 4)
        tv, ta = target["samples"].unbind()
        weights = [0, 0, 1, 1, 1, 0, 0]
        mask = refine._flatten_mask(sv.shape, sa.shape, weights, [1] * sa.shape[-1])
        identity = refine.temporal_mask(mask, sv, sa, sv, sa, [1] * 22)
        torch.testing.assert_close(identity, mask)
        expanded = refine.temporal_mask(mask, sv, sa, tv, ta, [2] * 17 + [1] * 5)
        self.assertEqual(expanded.shape, (1, tv.numel() + ta.numel()))
        self.assertTrue(torch.all(expanded[:, tv.numel():] == 1))
        self.assertEqual(expanded.max().item(), 1)
        self.assertEqual(expanded.min().item(), 0)
        # All channels and spatial cells receive the same temporal weights.
        expanded_video = expanded[:, :tv.numel()].reshape(tv.shape)
        torch.testing.assert_close(expanded_video[:, 0], expanded_video[:, 23])

    def test_unsupported_guides_masks_and_areas_fail(self):
        source = latent()
        for values in [
            {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": torch.ones(1, 24, 2, 2, 2)}]},
            {"minimax_keyframes": [{"resolved_frame_index": 0, "audio_latent": torch.ones(1)}]},
            {"mask": torch.ones(1, 22, 2, 2)},
            {"area": (1, 1, 0, 0)},
            {"minimax_refs": [{"fl_motion_context_audio_end_frame": 10}]},
        ]:
            with self.subTest(values=list(values)), self.assertRaises(ValueError):
                refine.retime_conditioning([[torch.ones(1), values]], source, source, [1] * 22)

    def test_context_windows_use_clone_static_pyramid_and_no_freenoise(self):
        original = model()
        options = copy.deepcopy(original.model_options)
        cloned, count, length, overlap = refine.context_model(original, 37, 24, 5)
        self.assertIsNot(cloned, original)
        self.assertGreater(count, 1)
        self.assertEqual((length, overlap), (7, 1))
        handler = cloned.model_options["context_handler"]
        self.assertEqual(handler.context_schedule.name, "standard_static")
        self.assertEqual(handler.fuse_method.name, "pyramid")
        self.assertFalse(handler.freenoise)
        self.assertEqual(original.model_options, options)
        with self.assertRaisesRegex(ValueError, "overlap"):
            refine.context_model(original, 37, 22, 25)
        whole, count, length, overlap = refine.context_model(original, 37, 0, 17)
        self.assertEqual((count, length, overlap), (1, 37, 0))
        self.assertNotIn("context_handler", whole.model_options)

    @unittest.skipUnless(motion is not None, "Install ComfyUI-MAINodes for integration tests")
    def test_pipeline_runs_real_mainodes_and_preserves_input(self):
        source = latent()
        video, audio = source["samples"].unbind()
        originals = [video.clone(), audio.clone()]
        real_ops = {name: getattr(motion, name)() for name in refine._MAINODES}
        for coverage, size in [("off", 0), ("uniform", 64), ("balanced", 0)]:
            with self.subTest(coverage=coverage):
                guider = mock.Mock()
                guider.sample.side_effect = lambda noise, samples, *args, **kwargs: samples
                audio_vae = mock.Mock(wraps=AudioVAE())
                audio_vae.audio_sample_rate = 32000
                audio_vae.audio_sample_rate_output = 32000
                with mock.patch.object(refine, "motion_nodes", return_value=real_ops), \
                     mock.patch.object(refine, "Guider_Basic", return_value=guider), \
                     mock.patch.object(refine.latent_preview, "prepare_callback", return_value=None), \
                     mock.patch.object(refine.comfy.samplers, "calculate_sigmas", side_effect=lambda m, s, n: torch.linspace(1, 0, n + 1)):
                    result = run_pipeline(model(), [[torch.ones(1, 2, 3), {}]], source,
                        VideoVAE(), audio_vae, motion_coverage=coverage, target_long_side=size)
                self.assertEqual(tuple(result[0].shape), (22, size or 32, size or 32, 3))
                self.assertIn("12/25 refinement steps", result[1])
                self.assertEqual(len(guider.sample.call_args.args[3]), 13)
                torch.testing.assert_close(guider.sample.call_args.args[3], torch.linspace(1, 0, 26)[13:])
                self.assertEqual(guider.sample.call_args.kwargs["seed"], 20260902)
                video_mask, audio_mask = guider.sample.call_args.kwargs["denoise_mask"].unbind()
                self.assertTrue(torch.all(video_mask == 1))
                self.assertTrue(torch.all(audio_mask == 0.5))
                if coverage == "off":
                    audio_vae.decode.assert_not_called()
                    audio_vae.encode.assert_not_called()
                torch.testing.assert_close(video, originals[0])
                torch.testing.assert_close(audio, originals[1])

    def test_invalid_h3_lengths_and_reshots_fail_before_vae(self):
        source = latent()
        source["fl_h3_shot"] = {"reshot": {"start": 0}}
        vae = mock.Mock()
        with self.assertRaisesRegex(ValueError, "Temporal reshots"):
            run_pipeline(model(), [], source, vae, AudioVAE())
        vae.decode.assert_not_called()
        source = latent()
        v, a = source["samples"].unbind()
        source["samples"] = comfy.nested_tensor.NestedTensor((v, a[..., :-1]))
        with self.assertRaisesRegex(ValueError, "durations differ"):
            run_pipeline(model(), [], source, vae, AudioVAE())

    def test_cancellation_propagates_before_decode(self):
        vae = mock.Mock()
        with mock.patch.object(refine, "motion_nodes", return_value={}), \
             mock.patch.object(refine.comfy.model_management, "throw_exception_if_processing_interrupted", side_effect=InterruptedError("cancelled")):
            with self.assertRaises(InterruptedError):
                run_pipeline(model(), [], latent(), vae)
        vae.decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
