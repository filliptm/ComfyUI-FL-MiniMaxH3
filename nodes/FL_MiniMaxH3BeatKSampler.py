import logging
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
import nodes
from comfy_api.latest import io
from comfy_execution.utils import get_executing_context

from ._latent_helpers import H3_PIXEL_MULTIPLE, primary_only_noise_mask, target_canvas
from ._live_preview import publish_preview, send_preview_event
from ._motion_context import apply_previous_shot_context, apply_previous_shot_contexts, motion_context_model
from .FL_MiniMaxH3PromptTimeline import _apply_timeline, _h3_tensors


H3ShotPlan = io.Custom("FL_H3_SHOT_PLAN")
_MAX_SEED = 0xffffffffffffffff
_SAMPLE_PREVIEW_LONG_SIDE = 512
_UPSCALE_PREVIEW_LONG_SIDE = 768


def _execution_ids(node_class):
    context = get_executing_context()
    node_id = node_class.hidden.unique_id if node_class.hidden is not None else None
    if context is not None:
        node_id = node_id or context.node_id
        return node_id, context.prompt_id
    return node_id, None


def _validate_shot_plan(plan):
    if not isinstance(plan, dict) or plan.get("type") != "minimax_h3_beat_shot_plan":
        raise TypeError("FL MiniMax H3 Beat KSampler received an invalid shot plan.")
    if plan.get("version") != 1:
        raise ValueError("FL MiniMax H3 Beat KSampler supports shot plan version 1.")
    shots = plan.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("FL MiniMax H3 Beat KSampler requires at least one planned render.")
    for index, shot in enumerate(shots, 1):
        if not isinstance(shot, dict) or "latent" not in shot or "conditioning" not in shot:
            raise ValueError(f"FL MiniMax H3 Beat KSampler render {index} is incomplete.")
    return shots


def _reference_guidance(
    conditioning,
    reference_free_conditioning,
    cfg,
    reference_strength,
    has_visual_references,
):
    if not has_visual_references or reference_free_conditioning is None:
        return conditioning, conditioning, cfg
    scale = cfg * reference_strength
    if scale == 0:
        return reference_free_conditioning, reference_free_conditioning, 1.0
    return conditioning, reference_free_conditioning, scale


def _target_canvas(source_width, source_height, target_long_side):
    return target_canvas(
        source_width,
        source_height,
        target_long_side,
        "FL MiniMax H3 Beat Pixel Upscale KSampler",
    )


def _shot_for_latent(shot_plan, latent):
    if isinstance(shot_plan, dict) and shot_plan.get("mode") == "temporal_reshot":
        raise ValueError(
            "FL MiniMax H3 Beat Pixel Upscale KSampler does not support temporal reshots. "
            "Connect the first Beat KSampler directly to Temporal Reshot Assembler."
        )
    shots = _validate_shot_plan(shot_plan)
    metadata = latent.get("fl_h3_shot") if isinstance(latent, dict) else None
    if not isinstance(metadata, dict) or metadata.get("version") != 1:
        raise ValueError(
            "FL MiniMax H3 Beat Pixel Upscale KSampler requires a latent from "
            "FL MiniMax H3 Beat KSampler."
        )
    index = metadata.get("index")
    if not isinstance(index, int) or index < 0 or index >= len(shots):
        raise ValueError("FL MiniMax H3 Beat Pixel Upscale KSampler received invalid shot metadata.")
    shot = shots[index]
    if (
        shot.get("start_frame") != metadata.get("start_frame")
        or shot.get("end_frame") != metadata.get("end_frame")
        or shot.get("authored_frames") != metadata.get("authored_frames")
        or shot.get("render_frames") != metadata.get("render_frames")
        or shot.get("motion_context") != metadata.get("motion_context")
    ):
        raise ValueError(
            "FL MiniMax H3 Beat Pixel Upscale KSampler latent does not match the connected "
            "shot plan. Connect the exact plan used by FL MiniMax H3 Beat KSampler; when using "
            "FL MiniMax H3 Shot Motion Context, connect its output to both samplers."
        )
    return shot, metadata


def _pixel_upscale_latent(latent, vae, target_long_side, upscale_method):
    video, audio = _h3_tensors(latent)
    images = vae.decode(video)
    if images.ndim != 5 or images.shape[0] != 1 or images.shape[-1] != 3:
        raise ValueError(
            "FL MiniMax H3 Beat Pixel Upscale KSampler requires an H3 video VAE "
            "that decodes one planned render to [1, frames, height, width, RGB]."
        )

    images = images[0]
    source_width = images.shape[-2]
    source_height = images.shape[-3]
    target_width, target_height = _target_canvas(
        source_width,
        source_height,
        target_long_side,
    )
    if (target_width, target_height) != (source_width, source_height):
        images = comfy.utils.common_upscale(
            images.movedim(-1, 1),
            target_width,
            target_height,
            upscale_method,
            "disabled",
        ).movedim(1, -1)

    resized_video = vae.encode(images)
    if resized_video.ndim != 5 or resized_video.shape[:3] != video.shape[:3]:
        raise ValueError(
            "FL MiniMax H3 Beat Pixel Upscale KSampler VAE re-encode changed the "
            "H3 batch, channel, or temporal latent dimensions "
            f"from {tuple(video.shape[:3])} to {tuple(resized_video.shape[:3])}."
        )

    output = latent.copy()
    output["samples"] = comfy.nested_tensor.NestedTensor((resized_video, audio))
    return output, source_width, source_height, target_width, target_height


class FL_MiniMaxH3BeatKSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3BeatKSampler",
            display_name="FL MiniMax H3 Beat KSampler",
            category="FL/MiniMax H3/Sampling",
            description=(
                "Samples every MiniMax H3 beat-planned render independently. A planned render "
                "can contain one prompt section, a shared-context chunk, or a hidden prior-shot prefix."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="MiniMax H3 model used for every planned render.",
                ),
                H3ShotPlan.Input(
                    "shot_plan",
                    tooltip="Render plan from the Beat Shot Planner or Shot Motion Context node.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=_MAX_SEED,
                    control_after_generate=True,
                    tooltip="Base noise seed for the first planned render.",
                ),
                io.Combo.Input(
                    "seed_mode",
                    options=["increment", "fixed"],
                    default="increment",
                    tooltip=(
                        "increment adds the render index to the base seed; fixed reuses the same seed "
                        "for every planned render."
                    ),
                ),
                io.Int.Input(
                    "steps",
                    default=20,
                    min=1,
                    max=10000,
                    tooltip="Sampling steps performed independently for every planned render.",
                ),
                io.Float.Input(
                    "cfg",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    tooltip=(
                        "Multiplier for the planned visual reference strength. Leave at 1 for the "
                        "planner's direct 0-1 blend; higher values can over-guide references."
                    ),
                ),
                io.Combo.Input(
                    "sampler_name",
                    options=comfy.samplers.KSampler.SAMPLERS,
                    tooltip="ComfyUI sampler used for every planned render.",
                ),
                io.Combo.Input(
                    "scheduler",
                    options=comfy.samplers.KSampler.SCHEDULERS,
                    tooltip="ComfyUI scheduler used for every planned render.",
                ),
                io.Float.Input(
                    "denoise",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Denoise strength applied independently to every planned latent.",
                ),
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip=(
                        "MiniMax H3 video VAE. Required only when the connected Shot Motion "
                        "Context plan uses visual context, or when live preview is enabled."
                    ),
                ),
                io.Boolean.Input(
                    "live_preview",
                    default=False,
                    optional=True,
                    tooltip="Decode a lightweight silent preview after each completed render.",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latents",
                    is_output_list=True,
                    tooltip=(
                        "One editable nested H3 latent per planned render. Connected latent nodes "
                        "run independently across the list and preserve render boundaries."
                    ),
                )
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        model,
        shot_plan,
        seed,
        seed_mode,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        vae=None,
        live_preview=False,
    ):
        shots = _validate_shot_plan(shot_plan)
        node_id, prompt_id = _execution_ids(cls)
        if live_preview:
            send_preview_event(
                node_id,
                prompt_id,
                "sample",
                "start",
                total=len(shots),
            )
            if vae is None:
                send_preview_event(
                    node_id,
                    prompt_id,
                    "sample",
                    "unavailable",
                    total=len(shots),
                    error="Connect the MiniMax H3 video VAE to enable live preview.",
                )
        uses_motion_context = any(
            isinstance(shot.get("motion_context"), dict)
            and (
                shot["motion_context"].get("video_frames", 0)
                or shot["motion_context"].get("audio_frames", 0)
            )
            for shot in shots
        )
        if uses_motion_context:
            model = motion_context_model(model)
        sampled = []
        progress = comfy.utils.ProgressBar(len(shots))

        for position, shot in enumerate(shots):
            shot_seed = seed if seed_mode == "fixed" else (seed + position) & _MAX_SEED
            if live_preview and vae is not None:
                send_preview_event(
                    node_id,
                    prompt_id,
                    "sample",
                    "sampling",
                    index=position,
                    total=len(shots),
                )
            conditioning = shot["conditioning"]
            reference_free_conditioning = shot.get("reference_free_conditioning")
            has_visual_references = bool(shot.get("has_visual_references"))
            if position:
                if has_visual_references and reference_free_conditioning is not None:
                    conditioning, reference_free_conditioning = apply_previous_shot_contexts(
                        [conditioning, reference_free_conditioning],
                        sampled[-1],
                        shots[position - 1],
                        shot,
                        vae,
                    )
                else:
                    conditioning = apply_previous_shot_context(
                        conditioning,
                        sampled[-1],
                        shots[position - 1],
                        shot,
                        vae,
                    )
            positive, negative, guidance = _reference_guidance(
                conditioning,
                reference_free_conditioning,
                cfg,
                shot.get("reference_strength", shot_plan.get("reference_strength", 1.0)),
                has_visual_references,
            )
            logging.info(
                "FL MiniMax H3 beat sampler: render %d/%d, frames %d-%d, seed %d.",
                position + 1,
                len(shots),
                shot["start_frame"],
                shot["end_frame"] - 1,
                shot_seed,
            )
            try:
                latent = nodes.common_ksampler(
                    model,
                    shot_seed,
                    steps,
                    guidance,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    shot["latent"],
                    denoise=denoise,
                )[0]
            except Exception as error:
                raise RuntimeError(
                    "FL MiniMax H3 Beat KSampler failed on "
                    f"render {position + 1}/{len(shots)} "
                    f"(frames {shot['start_frame']}-{shot['end_frame'] - 1})."
                ) from error
            latent["fl_h3_shot"] = {
                "version": 1,
                "index": position,
                "start_frame": shot["start_frame"],
                "end_frame": shot["end_frame"],
                "authored_frames": shot["authored_frames"],
                "render_frames": shot["render_frames"],
                "total_frames": shot_plan["total_frames"],
                "fps": shot_plan["fps"],
                "seed": shot_seed,
            }
            if isinstance(shot.get("motion_context"), dict):
                latent["fl_h3_shot"]["motion_context"] = dict(shot["motion_context"])
            if isinstance(shot.get("reshot"), dict):
                latent["fl_h3_shot"]["reshot"] = dict(shot["reshot"])
            sampled.append(latent)
            if live_preview and vae is not None:
                publish_preview(
                    latent,
                    vae,
                    latent["fl_h3_shot"],
                    node_id,
                    prompt_id,
                    "sample",
                    position,
                    len(shots),
                    _SAMPLE_PREVIEW_LONG_SIDE,
                )
            progress.update(1)

        if live_preview and vae is not None:
            send_preview_event(
                node_id,
                prompt_id,
                "sample",
                "done",
                total=len(shots),
            )
        return io.NodeOutput(sampled)


class FL_MiniMaxH3BeatUpscaleKSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3BeatUpscaleKSampler",
            display_name="FL MiniMax H3 Beat Pixel Upscale KSampler",
            category="FL/MiniMax H3/Sampling",
            description=(
                "Decodes each independently sampled MiniMax H3 render, resizes its pixels, "
                "re-encodes it, and refines only its video latent."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="MiniMax H3 model used for the refinement pass.",
                ),
                H3ShotPlan.Input(
                    "shot_plan",
                    tooltip=(
                        "The exact shot plan connected to the first Beat KSampler, including "
                        "the output of Shot Motion Context when it is used."
                    ),
                ),
                io.Latent.Input(
                    "latent",
                    tooltip="Editable latent list from FL MiniMax H3 Beat KSampler.",
                ),
                io.Vae.Input(
                    "vae",
                    tooltip="MiniMax H3 video VAE used for the pixel round trip.",
                ),
                io.Int.Input(
                    "target_long_side",
                    default=896,
                    min=H3_PIXEL_MULTIPLE,
                    max=nodes.MAX_RESOLUTION,
                    step=H3_PIXEL_MULTIPLE,
                    tooltip=(
                        "Target pixel size for the longest side. The other side stays "
                        "proportional and both dimensions remain aligned to 32 pixels."
                    ),
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=["bicubic", "bilinear", "nearest-exact", "lanczos"],
                    default="bicubic",
                    tooltip="Pixel interpolation used between H3 VAE decode and re-encode.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=_MAX_SEED,
                    control_after_generate=True,
                    tooltip="Base refinement seed for the first planned render.",
                ),
                io.Combo.Input(
                    "seed_mode",
                    options=["increment", "fixed"],
                    default="increment",
                    tooltip="increment adds the render index to the base refinement seed.",
                ),
                io.Int.Input(
                    "steps",
                    default=80,
                    min=1,
                    max=10000,
                    tooltip="Total steps in the H3 refinement schedule.",
                ),
                io.Int.Input(
                    "start_at_step",
                    default=60,
                    min=0,
                    max=10000,
                    tooltip="Start the refinement pass at this step in the total schedule.",
                ),
                io.Int.Input(
                    "end_at_step",
                    default=80,
                    min=0,
                    max=10000,
                    tooltip="End the refinement pass at this step in the total schedule.",
                ),
                io.Float.Input(
                    "cfg",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    tooltip=(
                        "Multiplier for the planned visual reference strength. Leave at 1 for the "
                        "planner's direct 0-1 blend; higher values can over-guide references."
                    ),
                ),
                io.Combo.Input(
                    "sampler_name",
                    options=comfy.samplers.KSampler.SAMPLERS,
                    tooltip="Sampler used to restore detail after H3 VAE re-encoding.",
                ),
                io.Combo.Input(
                    "scheduler",
                    options=comfy.samplers.KSampler.SCHEDULERS,
                    tooltip="Scheduler used for the refinement pass.",
                ),
                io.Boolean.Input(
                    "live_preview",
                    default=False,
                    optional=True,
                    tooltip="Decode a lightweight silent preview after each refined render.",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip=(
                        "Refined native H3 latent. Render timing metadata and the original "
                        "audio latent are preserved."
                    ),
                )
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        model,
        shot_plan,
        latent,
        vae,
        target_long_side,
        upscale_method,
        seed,
        seed_mode,
        steps,
        start_at_step,
        end_at_step,
        cfg,
        sampler_name,
        scheduler,
        live_preview=False,
    ):
        shot, metadata = _shot_for_latent(shot_plan, latent)
        node_id, prompt_id = _execution_ids(cls)
        total = len(shot_plan["shots"])
        if live_preview and metadata["index"] == 0:
            send_preview_event(
                node_id,
                prompt_id,
                "upscale",
                "start",
                total=total,
            )
        if live_preview:
            send_preview_event(
                node_id,
                prompt_id,
                "upscale",
                "sampling",
                index=metadata["index"],
                total=total,
            )
        timeline = shot.get("timeline")
        if not isinstance(timeline, dict):
            raise ValueError(
                "FL MiniMax H3 Beat Pixel Upscale KSampler requires a shot plan created "
                "by the current Beat Shot Planner."
            )
        resized, source_width, source_height, target_width, target_height = _pixel_upscale_latent(
            latent,
            vae,
            target_long_side,
            upscale_method,
        )
        conditioning = _apply_timeline(timeline, resized)
        reference_free_conditioning = None
        has_visual_references = bool(shot.get("has_visual_references"))
        if has_visual_references:
            reference_free_conditioning = _apply_timeline(
                timeline,
                resized,
                reference_free=True,
            )
        positive, negative, guidance = _reference_guidance(
            conditioning,
            reference_free_conditioning,
            cfg,
            shot.get("reference_strength", shot_plan.get("reference_strength", 1.0)),
            has_visual_references,
        )
        sample_input = resized.copy()
        motion_context = metadata.get("motion_context") or {}
        sample_input["noise_mask"] = primary_only_noise_mask(
            resized["samples"],
            motion_context.get("video_steps", 0),
        )
        shot_seed = (
            seed
            if seed_mode == "fixed"
            else (seed + metadata["index"]) & _MAX_SEED
        )
        logging.info(
            "FL MiniMax H3 beat pixel upscale sampler: render %d, %dx%d to %dx%d, seed %d.",
            metadata["index"] + 1,
            source_width,
            source_height,
            target_width,
            target_height,
            shot_seed,
        )
        try:
            output = nodes.common_ksampler(
                model,
                shot_seed,
                steps,
                guidance,
                sampler_name,
                scheduler,
                positive,
                negative,
                sample_input,
                denoise=1.0,
                start_step=start_at_step,
                last_step=end_at_step,
                force_full_denoise=True,
            )[0]
        except Exception as error:
            raise RuntimeError(
                "FL MiniMax H3 Beat Pixel Upscale KSampler failed on "
                f"render {metadata['index'] + 1} "
                f"({source_width}x{source_height} to {target_width}x{target_height})."
            ) from error
        output.pop("noise_mask", None)
        if live_preview:
            preview_metadata = dict(metadata)
            preview_metadata["seed"] = shot_seed
            publish_preview(
                output,
                vae,
                preview_metadata,
                node_id,
                prompt_id,
                "upscale",
                metadata["index"],
                total,
                _UPSCALE_PREVIEW_LONG_SIDE,
            )
            if metadata["index"] == total - 1:
                send_preview_event(
                    node_id,
                    prompt_id,
                    "upscale",
                    "done",
                    total=total,
                )
        return io.NodeOutput(output)
