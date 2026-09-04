import torch

import comfy.nested_tensor
import nodes
from comfy.ldm.minimax.model import FRAME_PER_TOKEN
from comfy_api.latest import io
from comfy_extras import nodes_minimax_h3 as minimax_h3

from ._latent_helpers import h3_tensors


H3TransitionPlan = io.Custom("FL_H3_TRANSITION_PLAN")
_OWNER = "FL MiniMax H3 Transition Prep"
DEFAULT_TRANSITION_DESCRIPTION = (
    "[Shot 1] A single continuous shot begins exactly from the framing, lighting, color, "
    "subject appearance, environment, and ongoing motion established by Picture 1. The camera "
    "and subjects move continuously without a cut, reset, duplicated features, or sudden identity "
    "change. Across the middle, every visible change progresses naturally toward Picture 2 while "
    "preserving anatomy, wardrobe, object identity, scale, screen direction, texture, and color "
    "continuity. The final moment arrives exactly at Picture 2's composition, state, and continuing "
    "motion."
)
DEFAULT_SOUNDSCAPE = (
    "Natural production sound remains continuous across the transition, with room tone and movement "
    "sounds matching the visible action."
)


def build_transition_prompt(frame_count, transition_description, overall_soundscape, non_diegetic_music):
    duration = frame_count / minimax_h3.FPS
    return (
        "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns "
        f"with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the {duration:.2f}-second mark of the target video.\n\n"
        f"integrated_multimodal_description: {transition_description.strip()}\n\n"
        f"overall_soundscape: {overall_soundscape.strip()}\n\n"
        f"non_diegetic_music: {non_diegetic_music.strip()}"
    )


def _validate_canvas(width, height):
    if width % minimax_h3.CANVAS_MULTIPLE or height % minimax_h3.CANVAS_MULTIPLE:
        raise ValueError(f"{_OWNER}: width and height must be divisible by {minimax_h3.CANVAS_MULTIPLE}.")
    if width * height > minimax_h3.MAX_PIXELS:
        raise ValueError(
            f"{_OWNER}: {width}x{height} exceeds H3's {minimax_h3.MAX_PIXELS}-pixel canvas limit."
        )


def _validate_images(images, name, reference_frames):
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError(f"{_OWNER}: {name} must be an IMAGE batch.")
    if images.shape[0] < reference_frames:
        raise ValueError(
            f"{_OWNER}: {name} has {images.shape[0]} frames but needs at least {reference_frames}."
        )


def _transition_token_range(video_t, frame_count, reference_frames):
    edges = [0]
    for token in range(video_t):
        edges.append(edges[-1] + FRAME_PER_TOKEN[token % len(FRAME_PER_TOKEN)])
    if edges[-1] != frame_count:
        raise ValueError(
            f"{_OWNER}: {video_t} video tokens resolve to {edges[-1]} frames, expected {frame_count}."
        )
    try:
        token_start = edges.index(reference_frames)
        token_end = edges.index(frame_count - reference_frames)
    except ValueError as error:
        raise ValueError(
            f"{_OWNER}: reference frames must align to H3's 17k+5 temporal grid."
        ) from error
    if token_end <= token_start:
        raise ValueError(f"{_OWNER}: reference frames leave no generated middle frames.")
    return token_start, token_end


def _feather_video_mask(video_latent, token_start, token_end, feather_tokens):
    video_mask = torch.zeros_like(video_latent)
    token_count = token_end - token_start
    feather_tokens = _effective_feather_tokens(token_count, feather_tokens)
    video_mask[:, :, token_start:token_end].fill_(1)
    for offset in range(feather_tokens):
        amount = (offset + 1) / (feather_tokens + 1)
        video_mask[:, :, token_start + offset].fill_(amount)
        video_mask[:, :, token_end - offset - 1].fill_(amount)
    return video_mask


def _effective_feather_tokens(token_count, feather_tokens):
    return min(max(0, int(feather_tokens)), max(0, (token_count - 1) // 2))


def _masked_transition_latent(
    latent,
    video_latent,
    frame_count,
    reference_frames,
    feather_tokens=0,
):
    empty_video, audio = h3_tensors(latent, _OWNER)
    if video_latent.shape != empty_video.shape:
        raise ValueError(
            f"{_OWNER}: video VAE returned {tuple(video_latent.shape)}, expected {tuple(empty_video.shape)}."
        )
    token_start, token_end = _transition_token_range(
        video_latent.shape[2], frame_count, reference_frames
    )
    video_mask = _feather_video_mask(
        video_latent,
        token_start,
        token_end,
        feather_tokens,
    )
    output = latent.copy()
    output["samples"] = comfy.nested_tensor.NestedTensor((video_latent, audio))
    output["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (video_mask, torch.ones_like(audio))
    )
    return output


def _transition_plan(
    frame_count,
    reference_frames,
    width,
    height,
    crop_mode,
    prompt,
    control_mode="empty bridge",
    source_a_edit_frames=0,
    source_b_edit_frames=0,
    mask_feather_tokens=0,
):
    return {
        "type": "fl_minimax_h3_transition_plan",
        "version": 1,
        "fps": minimax_h3.FPS,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "reference_frames": reference_frames,
        "generated_frames": frame_count - 2 * reference_frames,
        "control_mode": control_mode,
        "source_a_edit_frames": source_a_edit_frames,
        "source_b_edit_frames": source_b_edit_frames,
        "mask_feather_tokens": mask_feather_tokens,
        "crop_mode": crop_mode,
        "prompt": prompt,
    }


def _validate_plan(plan):
    if (
        not isinstance(plan, dict)
        or plan.get("type") != "fl_minimax_h3_transition_plan"
        or plan.get("version") != 1
    ):
        raise ValueError("FL MiniMax H3 Transition Assembler requires a Transition Prep plan.")
    return plan


def assemble_transition(plan, video_a, bridge, video_b):
    plan = _validate_plan(plan)
    for images, name in ((video_a, "video_a"), (bridge, "bridge"), (video_b, "video_b")):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"FL MiniMax H3 Transition Assembler: {name} must be an RGB IMAGE batch.")
    frame_count = plan["frame_count"]
    reference_frames = plan["reference_frames"]
    expected_size = (plan["height"], plan["width"])
    if video_a.shape[1:3] != expected_size or video_b.shape[1:3] != expected_size:
        raise ValueError("FL MiniMax H3 Transition Assembler received source frames from a different plan canvas.")
    if bridge.shape[0] < frame_count or bridge.shape[1:3] != expected_size:
        raise ValueError(
            "FL MiniMax H3 Transition Assembler received a decoded bridge with the wrong length or canvas."
        )
    if bridge.device != video_a.device or bridge.dtype != video_a.dtype:
        bridge = bridge.to(device=video_a.device, dtype=video_a.dtype)
    if video_b.device != video_a.device or video_b.dtype != video_a.dtype:
        video_b = video_b.to(device=video_a.device, dtype=video_a.dtype)
    restored = bridge[:frame_count].clone()
    control_mode = plan.get("control_mode", "empty bridge")
    if control_mode == "source seam repair":
        edit_a = plan.get("source_a_edit_frames", 0)
        edit_b = plan.get("source_b_edit_frames", 0)
        if video_a.shape[0] < reference_frames + edit_a or video_b.shape[0] < reference_frames + edit_b:
            raise ValueError(
                "FL MiniMax H3 Transition Assembler received source videos that are shorter than the repair plan."
            )
        restored[:reference_frames].copy_(
            video_a[-(reference_frames + edit_a):-edit_a]
        )
        restored[-reference_frames:].copy_(
            video_b[edit_b:edit_b + reference_frames]
        )
        middle = restored[reference_frames:frame_count - reference_frames]
        return torch.cat((video_a[:-edit_a], middle, video_b[edit_b:]), dim=0), restored
    if control_mode != "empty bridge":
        raise ValueError(f"FL MiniMax H3 Transition Assembler received unknown control mode: {control_mode}")
    restored[:reference_frames].copy_(video_a[-reference_frames:])
    restored[-reference_frames:].copy_(video_b[:reference_frames])
    middle = restored[reference_frames:frame_count - reference_frames]
    return torch.cat((video_a, middle, video_b), dim=0), restored


class FL_MiniMaxH3TransitionPrep(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3TransitionPrep",
            display_name="FL MiniMax H3 Transition Prep",
            category="FL/MiniMax H3/Transition",
            description=(
                "Selects the tail of video A and head of video B, builds the official FL2VA prompt, "
                "and prepares either an empty bridge or a feather-masked repair of the real source seam."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Image.Input("video_a", tooltip="Video A frames at 24 fps. Its tail becomes the opening guide."),
                io.Image.Input("video_b", tooltip="Video B frames at 24 fps. Its head becomes the closing guide."),
                io.String.Input(
                    "transition_description",
                    default=DEFAULT_TRANSITION_DESCRIPTION,
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="Describe the continuous action and camera motion from Picture 1 to Picture 2.",
                ),
                io.String.Input(
                    "overall_soundscape",
                    default=DEFAULT_SOUNDSCAPE,
                    multiline=True,
                ),
                io.String.Input("non_diegetic_music", default="N/A", multiline=False),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input(
                    "length",
                    default=90,
                    min=22,
                    max=3600,
                    step=17,
                    tooltip="Total H3 bridge length. The node snaps it to the 17k+5 frame grid.",
                ),
                io.Int.Input(
                    "reference_frames",
                    default=22,
                    min=5,
                    max=362,
                    step=17,
                    tooltip="Protected frames from each source. Valid values follow H3's 17k+5 grid.",
                ),
                io.Combo.Input(
                    "crop_mode",
                    options=["center", "stretch"],
                    default="center",
                    tooltip="Center crops both videos to the H3 canvas or stretches them to fit.",
                ),
                io.Float.Input(
                    "empty_frame_level",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Combo.Input(
                    "control_mode",
                    options=["empty bridge", "source seam repair"],
                    default="empty bridge",
                    tooltip=(
                        "Source seam repair fills the edit with real tail/head frames and keeps output duration unchanged."
                    ),
                ),
                io.Int.Input(
                    "mask_feather_tokens",
                    default=2,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="Softens each temporal mask edge over this many H3 latent tokens.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="masked_latent"),
                H3TransitionPlan.Output(display_name="transition_plan"),
                io.Image.Output(display_name="video_a"),
                io.Image.Output(display_name="video_b"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        vae,
        video_a,
        video_b,
        transition_description,
        overall_soundscape,
        non_diegetic_music,
        width,
        height,
        length,
        reference_frames,
        crop_mode,
        empty_frame_level,
        control_mode="empty bridge",
        mask_feather_tokens=2,
    ):
        _validate_canvas(width, height)
        if not isinstance(transition_description, str) or not transition_description.strip():
            raise ValueError(f"{_OWNER}: enter a transition description.")
        if not isinstance(overall_soundscape, str) or not overall_soundscape.strip():
            raise ValueError(f"{_OWNER}: enter an overall soundscape or use N/A.")
        if not isinstance(non_diegetic_music, str) or not non_diegetic_music.strip():
            raise ValueError(f"{_OWNER}: enter non-diegetic music or use N/A.")
        frame_count, _, _ = minimax_h3.temporal_shape(length)
        reference_frames = int(reference_frames)
        if reference_frames < 5 or reference_frames % 17 != 5:
            raise ValueError(f"{_OWNER}: reference frames must be 5, 22, 39, and so on.")
        if 2 * reference_frames >= frame_count:
            raise ValueError(f"{_OWNER}: reference frames leave no generated middle frames.")
        if control_mode not in ("empty bridge", "source seam repair"):
            raise ValueError(f"{_OWNER}: unknown control mode: {control_mode}")
        generated_frames = frame_count - 2 * reference_frames
        edit_a = generated_frames // 2 if control_mode == "source seam repair" else 0
        edit_b = generated_frames - edit_a if control_mode == "source seam repair" else 0
        _validate_images(video_a, "video A", reference_frames + edit_a)
        _validate_images(video_b, "video B", reference_frames + edit_b)

        prompt = build_transition_prompt(
            frame_count,
            transition_description,
            overall_soundscape,
            non_diegetic_music,
        )
        positive, latent = minimax_h3.MiniMaxH3ImageToVideo.execute(
            clip, vae, prompt, width, height, frame_count
        ).args
        crop = "center" if crop_mode == "center" else "disabled"
        normalized_a = minimax_h3._resize(video_a, width, height, crop)
        normalized_b = minimax_h3._resize(video_b, width, height, crop)
        if control_mode == "source seam repair":
            control_a = normalized_a[-(reference_frames + edit_a):].clone()
            control_b = normalized_b[:reference_frames + edit_b].clone()
            reference_a = control_a[:reference_frames]
            reference_b = control_b[-reference_frames:]
            control = torch.cat((control_a, control_b), dim=0)
        else:
            reference_a = normalized_a[-reference_frames:].clone()
            reference_b = normalized_b[:reference_frames].clone()
            empty = torch.full(
                (generated_frames, height, width, 3),
                empty_frame_level,
                dtype=reference_a.dtype,
                device=reference_a.device,
            )
            control = torch.cat((reference_a, empty, reference_b), dim=0)
        video_latent = vae.encode(control)
        token_start, token_end = _transition_token_range(
            video_latent.shape[2], frame_count, reference_frames
        )
        feather_tokens = (
            _effective_feather_tokens(token_end - token_start, mask_feather_tokens)
            if control_mode == "source seam repair"
            else 0
        )
        masked_latent = _masked_transition_latent(
            latent,
            video_latent,
            frame_count,
            reference_frames,
            feather_tokens,
        )
        positive = minimax_h3.MiniMaxH3AddGuide.execute(
            positive,
            latent,
            0,
            vae=vae,
            image=reference_a,
        )[0]
        positive = minimax_h3.MiniMaxH3AddGuide.execute(
            positive,
            latent,
            -reference_frames,
            vae=vae,
            image=reference_b,
        )[0]
        plan = _transition_plan(
            frame_count,
            reference_frames,
            width,
            height,
            crop_mode,
            prompt,
            control_mode,
            edit_a,
            edit_b,
            feather_tokens,
        )
        return io.NodeOutput(positive, masked_latent, plan, normalized_a, normalized_b)


class FL_MiniMaxH3TransitionAssembler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3TransitionAssembler",
            display_name="FL MiniMax H3 Transition Assembler",
            category="FL/MiniMax H3/Transition",
            description=(
                "Restores protected reference pixels and inserts the generated bridge or source-seam repair "
                "without manual overlap indices."
            ),
            inputs=[
                H3TransitionPlan.Input("transition_plan"),
                io.Image.Input("video_a"),
                io.Image.Input("bridge", tooltip="Decoded H3 bridge from the Transition Prep latent."),
                io.Image.Input("video_b"),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Image.Output(display_name="restored_bridge"),
            ],
        )

    @classmethod
    def execute(cls, transition_plan, video_a, bridge, video_b):
        images, restored = assemble_transition(transition_plan, video_a, bridge, video_b)
        return io.NodeOutput(images, restored)
