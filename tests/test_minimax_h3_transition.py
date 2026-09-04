import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import torch

import comfy.nested_tensor
from comfy_api.latest import io


ROOT = pathlib.Path(__file__).parents[1]


def load_module(relative_path):
    path = pathlib.Path(relative_path)
    module_name = "fl_minimax_h3_transition_tests." + ".".join(path.with_suffix("").parts)
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


transition = load_module("nodes/FL_MiniMaxH3Transition.py")


def empty_latent(frame_count=22, height=32, width=32):
    video_t = transition.minimax_h3.video_latent_t(frame_count)
    audio_t = round(frame_count / transition.minimax_h3.FPS * transition.minimax_h3.AUDIO_LATENT_FPS)
    video = torch.zeros((1, 24, video_t, height // 16, width // 16))
    audio = torch.zeros((1, 32, 2, audio_t))
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


class FakeVAE:
    def __init__(self):
        self.encoded_images = []

    def encode(self, images):
        self.encoded_images.append(images.clone())
        frames, height, width = images.shape[:3]
        video_t = transition.minimax_h3.video_latent_t(frames)
        return torch.ones((1, 24, video_t, height // 16, width // 16))


class TransitionMathTests(unittest.TestCase):
    def test_prompt_uses_the_resolved_picture_two_time(self):
        prompt = transition.build_transition_prompt(90, "Move continuously.", "Room tone.", "N/A")
        self.assertIn("3.75-second mark", prompt)
        self.assertIn("integrated_multimodal_description: Move continuously.", prompt)

    def test_reference_boundaries_align_to_h3_tokens(self):
        self.assertEqual(transition._transition_token_range(27, 90, 22), (7, 20))
        with self.assertRaisesRegex(ValueError, r"17k\+5"):
            transition._transition_token_range(27, 90, 21)

    def test_mask_protects_both_reference_windows_and_generates_audio(self):
        latent = empty_latent(90)
        video, _ = transition.h3_tensors(latent, "test")
        output = transition._masked_transition_latent(latent, torch.ones_like(video), 90, 22)
        video_mask, audio_mask = output["noise_mask"].unbind()
        self.assertEqual(video_mask[:, :, :7].count_nonzero().item(), 0)
        self.assertTrue(torch.all(video_mask[:, :, 7:20] == 1))
        self.assertEqual(video_mask[:, :, 20:].count_nonzero().item(), 0)
        self.assertTrue(torch.all(audio_mask == 1))

    def test_mask_feathers_both_edges_inside_the_edit(self):
        latent = empty_latent(90)
        video, _ = transition.h3_tensors(latent, "test")
        output = transition._masked_transition_latent(
            latent,
            torch.ones_like(video),
            90,
            22,
            feather_tokens=2,
        )
        video_mask, _ = output["noise_mask"].unbind()
        self.assertEqual(video_mask[:, :, :7].count_nonzero().item(), 0)
        self.assertTrue(torch.allclose(video_mask[:, :, 7], torch.full_like(video_mask[:, :, 7], 1 / 3)))
        self.assertTrue(torch.allclose(video_mask[:, :, 8], torch.full_like(video_mask[:, :, 8], 2 / 3)))
        self.assertTrue(torch.all(video_mask[:, :, 9:18] == 1))
        self.assertTrue(torch.allclose(video_mask[:, :, 18], torch.full_like(video_mask[:, :, 18], 2 / 3)))
        self.assertTrue(torch.allclose(video_mask[:, :, 19], torch.full_like(video_mask[:, :, 19], 1 / 3)))
        self.assertEqual(video_mask[:, :, 20:].count_nonzero().item(), 0)


class TransitionNodeTests(unittest.TestCase):
    def test_prep_selects_source_ends_and_adds_both_guides(self):
        video_a = torch.linspace(0, 1, 7).view(7, 1, 1, 1).expand(7, 32, 32, 3).clone()
        video_b = torch.linspace(1, 0, 8).view(8, 1, 1, 1).expand(8, 32, 32, 3).clone()
        positive = [[torch.zeros((1, 1, 1)), {}]]
        latent = empty_latent()
        with (
            mock.patch.object(
                transition.minimax_h3.MiniMaxH3ImageToVideo,
                "execute",
                return_value=io.NodeOutput(positive, latent),
            ),
            mock.patch.object(
                transition.minimax_h3,
                "_resize",
                side_effect=lambda images, *_: images[..., :3],
            ),
        ):
            output = transition.FL_MiniMaxH3TransitionPrep.execute(
                object(),
                FakeVAE(),
                video_a,
                video_b,
                "Move continuously.",
                "Room tone.",
                "N/A",
                32,
                32,
                22,
                5,
                "center",
                0.5,
            )
        conditioning, masked, plan, normalized_a, normalized_b = output.args
        keyframes = conditioning[0][1]["minimax_keyframes"]
        self.assertEqual([item["resolved_frame_index"] for item in keyframes], [0, 17])
        self.assertEqual(plan["generated_frames"], 12)
        self.assertTrue(torch.equal(normalized_a, video_a))
        self.assertTrue(torch.equal(normalized_b, video_b))
        video_mask, _ = masked["noise_mask"].unbind()
        self.assertGreater(video_mask.count_nonzero().item(), 0)

    def test_assembler_restores_exact_references_and_inserts_only_the_middle(self):
        plan = transition._transition_plan(22, 5, 32, 32, "center", "prompt")
        video_a = torch.full((8, 32, 32, 3), 0.1)
        video_b = torch.full((9, 32, 32, 3), 0.9)
        bridge = torch.full((22, 32, 32, 3), 0.5)
        images, restored = transition.assemble_transition(plan, video_a, bridge, video_b)
        self.assertEqual(images.shape[0], 29)
        self.assertTrue(torch.equal(restored[:5], video_a[-5:]))
        self.assertTrue(torch.equal(restored[-5:], video_b[:5]))
        self.assertTrue(torch.all(images[8:20] == 0.5))
        self.assertTrue(torch.equal(images[:8], video_a))
        self.assertTrue(torch.equal(images[-9:], video_b))

    def test_source_seam_repair_uses_real_frames_and_preserves_total_duration(self):
        video_a = torch.arange(14).view(14, 1, 1, 1).expand(14, 32, 32, 3).float()
        video_b = torch.arange(100, 115).view(15, 1, 1, 1).expand(15, 32, 32, 3).float()
        positive = [[torch.zeros((1, 1, 1)), {}]]
        latent = empty_latent()
        vae = FakeVAE()
        with (
            mock.patch.object(
                transition.minimax_h3.MiniMaxH3ImageToVideo,
                "execute",
                return_value=io.NodeOutput(positive, latent),
            ),
            mock.patch.object(
                transition.minimax_h3,
                "_resize",
                side_effect=lambda images, *_: images[..., :3],
            ),
        ):
            output = transition.FL_MiniMaxH3TransitionPrep.execute(
                object(),
                vae,
                video_a,
                video_b,
                "Repair the cut continuously.",
                "Room tone.",
                "N/A",
                32,
                32,
                22,
                5,
                "center",
                0.5,
                "source seam repair",
                2,
            )
        _, masked, plan, normalized_a, normalized_b = output.args
        self.assertEqual(plan["source_a_edit_frames"], 6)
        self.assertEqual(plan["source_b_edit_frames"], 6)
        self.assertEqual(plan["mask_feather_tokens"], 1)
        self.assertTrue(torch.equal(vae.encoded_images[0][:11], video_a[-11:]))
        self.assertTrue(torch.equal(vae.encoded_images[0][11:], video_b[:11]))
        video_mask, _ = masked["noise_mask"].unbind()
        self.assertTrue(torch.all(video_mask[:, :, 2] == 0.5))
        self.assertTrue(torch.all(video_mask[:, :, 3] == 1))
        self.assertTrue(torch.all(video_mask[:, :, 4] == 0.5))

        bridge = torch.full((22, 32, 32, 3), 0.5)
        images, restored = transition.assemble_transition(
            plan,
            normalized_a,
            bridge,
            normalized_b,
        )
        self.assertEqual(images.shape[0], video_a.shape[0] + video_b.shape[0])
        self.assertTrue(torch.equal(images[:8], video_a[:-6]))
        self.assertTrue(torch.all(images[8:20] == 0.5))
        self.assertTrue(torch.equal(images[20:], video_b[6:]))
        self.assertTrue(torch.equal(restored[:5], video_a[-11:-6]))
        self.assertTrue(torch.equal(restored[-5:], video_b[6:11]))


if __name__ == "__main__":
    unittest.main()
