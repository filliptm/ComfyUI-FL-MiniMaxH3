import logging
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
import nodes
from comfy_api.latest import io

from ._latent_helpers import primary_only_noise_mask
from .FL_MiniMaxH3PromptTimeline import _apply_timeline, _h3_tensors


H3ShotPlan = io.Custom("FL_H3_SHOT_PLAN")
_MAX_SEED = 0xffffffffffffffff
_H3_PIXEL_MULTIPLE = 32


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


def _target_canvas(source_width, source_height, target_long_side):
    if target_long_side % _H3_PIXEL_MULTIPLE:
        raise ValueError(
            "FL MiniMax H3 Beat Pixel Upscale KSampler target long side must be "
            f"divisible by {_H3_PIXEL_MULTIPLE}."
        )
    source_long_side = max(source_width, source_height)
    if target_long_side < source_long_side:
        raise ValueError(
            "FL MiniMax H3 Beat Pixel Upscale KSampler target long side is smaller than "
            f"the {source_width}x{source_height} decoded video."
        )

    scale = target_long_side / source_long_side
    if source_width >= source_height:
        target_width = target_long_side
        target_height = round(source_height * scale / _H3_PIXEL_MULTIPLE) * _H3_PIXEL_MULTIPLE
    else:
        target_width = round(source_width * scale / _H3_PIXEL_MULTIPLE) * _H3_PIXEL_MULTIPLE
        target_height = target_long_side
    return (
        max(_H3_PIXEL_MULTIPLE, target_width),
        max(_H3_PIXEL_MULTIPLE, target_height),
    )


def _shot_for_latent(shot_plan, latent):
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
    ):
        raise ValueError(
            "FL MiniMax H3 Beat Pixel Upscale KSampler latent does not match the connected shot plan."
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
    if (
        resized_video.ndim != 5
        or resized_video.shape[:3] != video.shape[:3]
    ):
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
                "can contain one prompt section or a shared-context chunk."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="MiniMax H3 model used for every planned render.",
                ),
                H3ShotPlan.Input(
                    "shot_plan",
                    tooltip="Render plan from FL MiniMax H3 Beat Shot Planner.",
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
                    tooltip="MiniMax H3 guidance scale. The normal H3 workflow uses CFG 1.",
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
    ):
        shots = _validate_shot_plan(shot_plan)
        sampled = []
        progress = comfy.utils.ProgressBar(len(shots))

        for position, shot in enumerate(shots):
            shot_seed = seed if seed_mode == "fixed" else (seed + position) & _MAX_SEED
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
                    cfg,
                    sampler_name,
                    scheduler,
                    shot["conditioning"],
                    shot["conditioning"],
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
            sampled.append(latent)
            progress.update(1)

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
                    tooltip="The same shot plan connected to the first beat KSampler.",
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
                    min=_H3_PIXEL_MULTIPLE,
                    max=nodes.MAX_RESOLUTION,
                    step=_H3_PIXEL_MULTIPLE,
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
                    default=20,
                    min=1,
                    max=10000,
                    tooltip="Sampling steps for the H3 refinement pass.",
                ),
                io.Float.Input(
                    "cfg",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    tooltip="MiniMax H3 guidance scale. The normal H3 workflow uses CFG 1.",
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
                io.Float.Input(
                    "denoise",
                    default=0.25,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Refinement strength. Lower values preserve motion and identity; "
                        "higher values invent more detail."
                    ),
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
        cfg,
        sampler_name,
        scheduler,
        denoise,
    ):
        shot, metadata = _shot_for_latent(shot_plan, latent)
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
        sample_input = resized.copy()
        sample_input["noise_mask"] = primary_only_noise_mask(resized["samples"])
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
                cfg,
                sampler_name,
                scheduler,
                conditioning,
                conditioning,
                sample_input,
                denoise=denoise,
            )[0]
        except Exception as error:
            raise RuntimeError(
                "FL MiniMax H3 Beat Pixel Upscale KSampler failed on "
                f"render {metadata['index'] + 1} "
                f"({source_width}x{source_height} to {target_width}x{target_height})."
            ) from error
        output.pop("noise_mask", None)
        return io.NodeOutput(output)
