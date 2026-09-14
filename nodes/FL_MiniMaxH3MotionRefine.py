import bisect
import itertools
import json
import logging
import time

import torch

import comfy.context_windows as context_windows
import comfy.latent_formats
import comfy.model_management
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
import latent_preview
import nodes
from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder
from comfy_extras import nodes_minimax_h3 as h3
from comfy_extras.nodes_audio import VAEEncodeAudio, vae_decode_audio
from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise

from ._latent_helpers import h3_tensors, target_canvas
from .FL_MiniMaxH3PromptTimeline import _flatten_mask
from ._motion_context import AUDIO_END_FRAME_MARKER, VIDEO_FRAME_MARKER, VIDEO_CONTEXT_STEPS, motion_context_model, apply_previous_shot_context
from ._motion_refine_shots import ShotData, H3ShotPlan, validate_plan, shot_images, shot_audio


_OWNER = "FL MiniMax H3 Motion Refine"
_COVERAGE = {
    "balanced": "balanced (default)",
    "economical": "economy (tight spans)",
    "wide": "max quality (wide plateau)",
}
_MAINODES = ("H3JerkOracle", "H3TimeSmear", "H3AudioSmear", "H3V2VInit")


def motion_nodes():
    missing = [name for name in _MAINODES if name not in nodes.NODE_CLASS_MAPPINGS]
    if missing:
        raise RuntimeError(f"{_OWNER}: install/update ComfyUI-MAINodes and restart ComfyUI. Missing: {', '.join(missing)}.")
    operations = {name: nodes.NODE_CLASS_MAPPINGS[name]() for name in _MAINODES}
    if not hasattr(operations["H3TimeSmear"], "plan"):
        raise RuntimeError(f"{_OWNER}: update ComfyUI-MAINodes with the Time Smear planning API and restart.")
    return operations


def frame_edges(holds):
    if not holds or any(type(value) is not int or value < 1 for value in holds):
        raise ValueError(f"{_OWNER}: hold counts must be positive integers.")
    return [0, *itertools.accumulate(holds)]


def frame_to_expanded(frame, edges):
    if not 0 <= frame <= len(edges) - 1:
        raise ValueError(f"{_OWNER}: conditioning anchor is outside the source clip.")
    index = int(frame)
    if index == len(edges) - 1:
        return edges[-1]
    return edges[index] + (frame - index) * (edges[index + 1] - edges[index])


def temporal_mask(mask, source_video, source_audio, target_video, target_audio, holds):
    # FL timeline masks are packed video/audio weights, constant across spatial cells and channels.
    video_size, audio_size = source_video.numel(), source_audio.numel()
    if tuple(mask.shape) != (1, video_size + audio_size):
        raise ValueError(f"{_OWNER}: use unmasked semantic conditioning or an FL H3 timeline mask; spatial masks are not supported.")
    video = mask[:, :video_size].reshape(source_video.shape)
    audio = mask[:, video_size:].reshape(source_audio.shape)
    vw, aw = video[0, 0, :, 0, 0], audio[0, 0, 0]
    if not torch.equal(video, vw[None, None, :, None, None].expand_as(video)) or not torch.equal(audio, aw[None, None, None, :].expand_as(audio)):
        raise ValueError(f"{_OWNER}: conditioning masks must contain temporal weights only.")
    vw, aw = vw.detach().float().cpu(), aw.detach().float().cpu()
    source_tokens = [token for token in range(source_video.shape[2]) for _ in range(h3.FRAME_PER_TOKEN[token % 5])]
    expanded_tokens = [token for token, count in zip(source_tokens, holds) for _ in range(count)]
    cursor = 0
    video_weights = []
    for token in range(target_video.shape[2]):
        span = h3.FRAME_PER_TOKEN[token % 5]
        video_weights.append(vw[expanded_tokens[cursor:cursor + span]].float().mean().item())
        cursor += span
    edges = frame_edges(holds)
    audio_indices = []
    for token in range(target_audio.shape[-1]):
        frame = min((token + 0.5) * h3.FPS / h3.AUDIO_LATENT_FPS, edges[-1] - 1e-6)
        source_frame = bisect.bisect_right(edges, frame) - 1
        source_time = source_frame + (frame - edges[source_frame]) / holds[source_frame]
        audio_indices.append(min(int(source_time * h3.AUDIO_LATENT_FPS / h3.FPS), len(aw) - 1))
    return _flatten_mask(target_video.shape, target_audio.shape, video_weights, aw[audio_indices].tolist())


def retime_conditioning(conditioning, source, target, holds, shot_mode=False, context_prefix=0):
    source_video, source_audio = h3_tensors(source, _OWNER)
    target_video, target_audio = h3_tensors(target, _OWNER)
    edges = frame_edges(holds)
    result = []
    for embedding, values in conditioning:
        if values.get("control") is not None or "area" in values:
            raise ValueError(f"{_OWNER}: spatial areas and ControlNet conditioning are not supported.")
        updated = values.copy()
        if values.get("mask") is not None:
            updated["mask"] = temporal_mask(values["mask"], source_video, source_audio, target_video, target_audio, holds)
        anchors = []
        for anchor in values.get("minimax_keyframes", []):
            # A clip/audio guide has its own clock; moving only its first frame would silently desync it.
            guide = anchor.get("latent")
            if anchor.get("audio_latent") is not None or (guide is not None and guide.shape[2] != 1):
                raise ValueError(f"{_OWNER}: use image anchors or untimed references; timed video/audio guides are not supported.")
            mapped = anchor.copy()
            if VIDEO_FRAME_MARKER in anchor:
                if not shot_mode:
                    raise ValueError(f"{_OWNER}: connect shot_plan for hidden prior-shot motion context.")
                mapped[VIDEO_FRAME_MARKER] = frame_to_expanded(anchor[VIDEO_FRAME_MARKER], edges)
            mapped["resolved_frame_index"] = frame_to_expanded(anchor["resolved_frame_index"], edges)
            anchors.append(mapped)
        if "minimax_keyframes" in values:
            updated["minimax_keyframes"] = anchors
        if "minimax_frame_count" in values:
            updated["minimax_frame_count"] = edges[-1]
        refs = []
        for ref in values.get("minimax_refs", []):
            if (AUDIO_END_FRAME_MARKER in ref or VIDEO_FRAME_MARKER in ref) and not shot_mode:
                raise ValueError(f"{_OWNER}: hidden prior-shot motion context is not supported; use an independent shot.")
            mapped = ref
            if AUDIO_END_FRAME_MARKER in ref:
                mapped = ref.copy()
                end = ref[AUDIO_END_FRAME_MARKER]
                # The fractional offset aligns the fixed 40 Hz reference, not the stretched target.
                mapped[AUDIO_END_FRAME_MARKER] = frame_to_expanded(context_prefix, edges) + (end - context_prefix)
            refs.append(mapped)
        if "minimax_refs" in values:
            updated["minimax_refs"] = refs
        result.append([embedding, updated])
    return result


def context_model(model, video_t, budget, overlap):
    if budget < 0 or (budget and budget < 5) or overlap < 0:
        raise ValueError(f"{_OWNER}: context budget must be 0 (whole clip) or at least 5 frames; overlap cannot be negative.")
    cloned = model.clone()
    if "context_handler" in cloned.model_options:
        raise ValueError(f"{_OWNER}: connect a model without an existing context handler; this node owns its context budget.")
    if budget == 0:
        return cloned, 1, video_t, 0
    frames = h3.align_frame_count(budget)
    if frames > budget:
        frames -= 17
    length = h3.video_latent_t(frames)
    cycles, remainder = divmod(overlap, 17)
    shared = cycles * 5 + remainder // 4
    if shared >= length:
        raise ValueError(f"{_OWNER}: overlap must be smaller than the resolved context window ({frames} frames).")
    if video_t <= length:
        return cloned, 1, video_t, 0
    handler = context_windows.IndexListContextHandler(
        context_schedule=context_windows.get_matching_context_schedule("standard_static"),
        fuse_method=context_windows.get_matching_fuse_method("pyramid"),
        context_length=length, context_overlap=shared, dim=2, causal_window_fix=True,
    )
    cloned.model_options["context_handler"] = handler
    context_windows.create_prepare_sampling_wrapper(cloned)
    windows = handler.context_schedule.func(video_t, handler, {})
    return cloned, len(windows), length, shared


def decode_video(vae, video, expected):
    images = vae.decode(video)
    if images.ndim != 5 or images.shape[0] != 1 or images.shape[1] != expected or images.shape[-1] != 3:
        raise ValueError(f"{_OWNER}: expected the H3 video VAE to decode {expected} RGB frames, got {tuple(images.shape)}.")
    return images[0]


def resize_image_anchors(conditioning, vae, width, height, method):
    if conditioning is None:
        return None
    resized = {}
    result = []
    for embedding, values in conditioning:
        anchors = []
        for anchor in values.get("minimax_keyframes", []):
            guide = anchor.get("latent")
            if guide is None or guide.shape[2] != 1 or anchor.get("audio_latent") is not None:
                raise ValueError(f"{_OWNER}: use image anchors; timed video/audio guides are not supported.")
            mapped = anchor
            if guide.shape[-2:] != (height // 16, width // 16):
                key = id(guide)
                if key not in resized:
                    images = decode_video(vae, guide, 1)
                    images = comfy.utils.common_upscale(images.movedim(-1, 1), width, height, method, "disabled").movedim(1, -1)
                    encoded = vae.encode(images)
                    if tuple(encoded.shape) != (1, 24, 1, height // 16, width // 16):
                        raise ValueError(f"{_OWNER}: image anchor VAE encoding must match the target canvas.")
                    resized[key] = encoded
                mapped = {**anchor, "latent": resized[key]}
            anchors.append(mapped)
        updated = values.copy()
        if "minimax_keyframes" in values:
            updated["minimax_keyframes"] = anchors
        result.append([embedding, updated])
    return result


class FL_MiniMaxH3MotionRefine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3MotionRefine",
            display_name=_OWNER,
            category="FL/MiniMax H3/Sampling",
            enable_expand=True,
            hidden=[io.Hidden.unique_id, io.Hidden.dynprompt],
            description="Stretch fast motion, optionally upscale, refine, and recover original timing. Connect shot_plan for a planned latent list with per-shot prompts, hidden context and hard cuts. Keep original audio wired to export. Requires ComfyUI-MAINodes.",
            inputs=[
                io.Model.Input("model", raw_link=True),
                io.Conditioning.Input("positive", optional=True, raw_link=True, tooltip="Required without shot_plan. Shot-plan mode uses each shot's own conditioning instead."),
                io.Latent.Input("latent", raw_link=True, tooltip="Completed native H3 video/audio latent, not an empty latent or temporal reshot."),
                io.Vae.Input("vae", raw_link=True),
                io.Vae.Input("audio_vae", optional=True, raw_link=True, tooltip="Required when the timeline expands; not needed with coverage off."),
                io.Image.Input("baseline_images", optional=True, raw_link=True, tooltip="Optional shared decode of this exact source latent. Connect the same images used by the before panel."),
                io.Audio.Input("baseline_audio", optional=True, raw_link=True, tooltip="Optional shared decode of this exact source latent's soundtrack."),
                H3ShotPlan.Input("shot_plan", optional=True, tooltip="Connect the exact plan used by Beat KSampler, including Shot Motion Context when present. Baseline images/audio then refer to the assembled timeline."),
                io.Int.Input("target_long_side", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32, tooltip="0 keeps source size. Otherwise upscale the long edge; dimensions align to 32 pixels."),
                io.Float.Input("strength", default=0.5, min=0.05, max=1.0, step=0.05, tooltip="Fraction of the full schedule to run. 25 steps x 0.5 = 12 refinement steps, not 25."),
                io.Combo.Input("motion_coverage", options=["balanced", "economical", "wide", "uniform", "off"], default="balanced", tooltip="Off benchmarks spatial refinement without temporal stretching. Uniform uses max_hold everywhere."),
                io.Int.Input("steps", default=25, min=4, max=100),
                io.Int.Input("context_budget", default=0, min=0, max=10000, tooltip="0 = whole expanded clip (matches the reference workflow). Otherwise a frame-equivalent window budget, snapped down. A causal anchor may add one latent position. Not a total VRAM cap."),
                io.Int.Input("seed", default=20260902, min=0, max=0xffffffffffffffff, control_after_generate=io.ControlAfterGenerate.fixed),
                io.Int.Input("context_overlap", default=17, min=0, max=1000, advanced=True),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="res_multistep", advanced=True),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple", advanced=True),
                io.Int.Input("max_hold", default=4, min=2, max=8, advanced=True, tooltip="Used only by uniform coverage. Adaptive coverage presets own their hold limits."),
                io.Float.Input("audio_strength", default=0.5, min=0, max=1, step=0.05, advanced=True, tooltip="Refinement freedom for the stretched baseline audio conditioning. Export keeps the original soundtrack separately."),
                io.Boolean.Input("expand_to_end", default=True, advanced=True, tooltip="Extend a short unheld tail through the last motion burst, matching H3 Time Smear."),
                io.Combo.Input("upscale_method", options=["lanczos", "bicubic", "bilinear"], default="lanczos", advanced=True),
            ],
            outputs=[io.Image.Output(display_name="images"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, positive=None, latent=None, vae=None, audio_vae=None, target_long_side=0,
                strength=0.5, motion_coverage="balanced", steps=25, context_budget=0,
                seed=20260902, context_overlap=17, sampler_name="res_multistep", scheduler="simple",
                max_hold=4, audio_strength=0.5, expand_to_end=True, upscale_method="lanczos",
                baseline_images=None, baseline_audio=None, shot_plan=None):
        graph = GraphBuilder()
        if shot_plan is not None:
            shots = validate_plan(shot_plan)
            plan_link = cls.hidden.dynprompt.get_node(cls.hidden.unique_id)["inputs"]["shot_plan"]
            collected = {}
            for index in range(len(shots)):
                source = graph.node("FL_MiniMaxH3MotionShot", latents=latent, shot_plan=plan_link,
                    index=index, vae=vae, audio_vae=audio_vae,
                    baseline_images=baseline_images, baseline_audio=baseline_audio)
                prepared = graph.node("FL_MiniMaxH3MotionPrepare", shot_data=source.out(0), vae=vae,
                    audio_vae=audio_vae, target_long_side=target_long_side, motion_coverage=motion_coverage,
                    max_hold=max_hold, expand_to_end=expand_to_end, upscale_method=upscale_method)
                sampled = graph.node("FL_MiniMaxH3MotionSample", prepared=prepared.out(0), shot_data=source.out(0),
                    model=model, vae=vae, strength=strength, steps=steps, context_budget=context_budget,
                    seed=(seed + index) & 0xffffffffffffffff, context_overlap=context_overlap,
                    sampler_name=sampler_name, scheduler=scheduler, audio_strength=audio_strength)
                collected[f"images.shot_{index}"] = sampled.out(0)
                collected[f"reports.shot_{index}"] = sampled.out(1)
            assembled = graph.node("FL_MiniMaxH3MotionCollect", total_frames=shot_plan["total_frames"], **collected)
            return io.NodeOutput(assembled.out(0), assembled.out(1), expand=graph.finalize())
        if positive is None:
            raise ValueError(f"{_OWNER}: connect positive for a single shot, or shot_plan for planned renders.")
        prepared = graph.node("FL_MiniMaxH3MotionPrepare", latent=latent, vae=vae, positive=positive,
            audio_vae=audio_vae, target_long_side=target_long_side, motion_coverage=motion_coverage,
            max_hold=max_hold, expand_to_end=expand_to_end, upscale_method=upscale_method,
            baseline_images=baseline_images, baseline_audio=baseline_audio)
        sampled = graph.node("FL_MiniMaxH3MotionSample", prepared=prepared.out(0),
            model=model, positive=positive, latent=latent, vae=vae, strength=strength,
            steps=steps, context_budget=context_budget, seed=seed, context_overlap=context_overlap,
            sampler_name=sampler_name, scheduler=scheduler, audio_strength=audio_strength)
        return io.NodeOutput(sampled.out(0), sampled.out(1), expand=graph.finalize())


Prepared = io.Custom("FL_H3_MOTION_PREP")


def stage_inputs(names):
    inputs = [value for value in FL_MiniMaxH3MotionRefine.define_schema().inputs if value.id in names]
    for value in inputs:
        value.rawLink = False
        if value.id == "latent":
            value.optional = True
    return inputs


class FL_MiniMaxH3MotionPrepare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FL_MiniMaxH3MotionPrepare", category="FL/MiniMax H3/Internal",
            is_dev_only=True,
            inputs=stage_inputs({"latent", "positive", "vae", "audio_vae", "target_long_side", "motion_coverage",
                "max_hold", "expand_to_end", "upscale_method", "baseline_images", "baseline_audio"}) + [ShotData.Input("shot_data", optional=True)],
            outputs=[Prepared.Output()])

    @classmethod
    def execute(cls, latent=None, vae=None, audio_vae=None, target_long_side=0, motion_coverage="balanced",
                max_hold=4, expand_to_end=True, upscale_method="lanczos",
                baseline_images=None, baseline_audio=None, shot_data=None, positive=None):
        started = time.perf_counter()
        if shot_data is not None:
            latent = shot_data["latent"]
            positive = shot_data["positive"]
        video, audio = h3_tensors(latent, _OWNER)
        if video.shape[2] < 2 or (video.shape[2] - 2) % 5:
            raise ValueError(f"{_OWNER}: source video must use H3's 5n+2 latent positions.")
        frames = (video.shape[2] - 2) // 5 * 17 + 5
        if audio.shape[-1] != round(frames / h3.FPS * h3.AUDIO_LATENT_FPS):
            raise ValueError(f"{_OWNER}: source audio and video durations differ.")
        metadata = latent.get("fl_h3_shot", {})
        context = metadata.get("motion_context") or {}
        if metadata.get("reshot") or (shot_data is None and any(context.get(key, 0) for key in ("video_frames", "audio_frames", "trim_frames"))):
            raise ValueError(f"{_OWNER}: connect the matching shot_plan for hidden motion context. Temporal reshots are not supported.")
        if motion_coverage not in (*_COVERAGE, "off", "uniform") or not 2 <= max_hold <= 8:
            raise ValueError(f"{_OWNER}: invalid motion coverage or hold limit.")
        width, height = video.shape[-1] * 16, video.shape[-2] * 16
        out_width, out_height = target_canvas(width, height, target_long_side or max(width, height), _OWNER)
        operations = motion_nodes()
        comfy.model_management.throw_exception_if_processing_interrupted()
        if motion_coverage in _COVERAGE:
            if video.shape[2] < 4:
                raise ValueError(f"{_OWNER}: adaptive coverage needs at least 22 source frames; use uniform or off for a five-frame clip.")
            hold_map = operations["H3JerkOracle"].read(latent, frames, 0.75, 4, True,
                preset=_COVERAGE[motion_coverage])[0]
        else:
            hold_map = json.dumps({"holds": [max_hold if motion_coverage == "uniform" else 1] * frames, "world_len": frames})
        if motion_coverage == "off":
            holds, used_map, expanded = [1] * frames, hold_map, frames
        else:
            holds, used_map, expanded, _, _ = operations["H3TimeSmear"].plan(
                frames, 1, hold_map, expand_to_end)
        if shot_data is not None and motion_coverage != "off":
            trim, authored = shot_data["trim"], shot_data["authored"]
            holds[:trim] = [1] * trim
            holds[trim + authored:] = [1] * (frames - trim - authored)
            expanded = max(39, h3.align_frame_count(sum(holds)))
            holds[-1] += expanded - sum(holds)
            used_map = json.dumps({"holds": holds, "world_len": frames})
        edges = frame_edges(holds)
        if len(holds) != frames or edges[-1] != expanded or expanded % 17 != 5:
            raise ValueError(f"{_OWNER}: invalid expanded H3 frame map.")
        if expanded != frames and audio_vae is None:
            raise ValueError(f"{_OWNER}: connect the H3 audio VAE to stretch the baseline audio with the video.")
        planned = time.perf_counter()
        positive = resize_image_anchors(positive, vae, out_width, out_height, upscale_method)
        if shot_data is not None and shot_data["previous"] is not None:
            positive = apply_previous_shot_context(positive, shot_data["previous"], shot_data["source_shot"],
                shot_data["shot"], vae, target_size=(out_width, out_height), upscale_method=upscale_method)
        if shot_data is not None:
            baseline_images = shot_images(vae, video, shot_data["shot"], shot_data["baseline_images"], shot_data["total_frames"])
        images = baseline_images if baseline_images is not None else decode_video(vae, video, frames)
        if tuple(images.shape) != (frames, height, width, 3):
            raise ValueError(f"{_OWNER}: shared baseline images must match the source latent's frame count and dimensions.")
        if (width, height) != (out_width, out_height):
            images = comfy.utils.common_upscale(images.movedim(-1, 1), out_width, out_height, upscale_method, "disabled").movedim(1, -1)
        decoded = time.perf_counter()
        indices = [i for i, count in enumerate(holds) for _ in range(count)]
        encoded = {"samples": vae.encode_h3_frame_sequence(images, indices)}
        del images
        if encoded["samples"].shape[2] != h3.video_latent_t(expanded):
            raise ValueError(f"{_OWNER}: video VAE encoded the wrong temporal length.")
        encoded_at = time.perf_counter()
        comfy.model_management.throw_exception_if_processing_interrupted()
        if shot_data is not None:
            baseline_audio = shot_audio(audio_vae, latent, shot_data["shot"], shot_data["baseline_audio"], shot_data["total_frames"])
        if expanded == frames and not (shot_data is not None and baseline_audio is not None):
            encoded_audio = {"samples": audio}
        else:
            baseline_audio = baseline_audio if baseline_audio is not None else vae_decode_audio(audio_vae, latent)
            duration = baseline_audio["waveform"].shape[-1] / baseline_audio["sample_rate"]
            if abs(duration - frames / h3.FPS) > 1 / h3.AUDIO_LATENT_FPS:
                raise ValueError(f"{_OWNER}: shared baseline audio must match the source duration.")
            stretched_audio = operations["H3AudioSmear"].smear(baseline_audio, used_map, fps=h3.FPS)[0]
            encoded_audio = VAEEncodeAudio.execute(audio_vae, stretched_audio)[0]
        finished = time.perf_counter()
        timings = (f"plan {planned-started:.2f}s, decode/resize {decoded-planned:.2f}s, "
                   f"video encode {encoded_at-decoded:.2f}s, audio {finished-encoded_at:.2f}s")
        logging.info("%s preparation: %s", _OWNER, timings)
        return io.NodeOutput({"video": encoded, "audio": encoded_audio, "holds": holds,
            "positive": positive,
            "frames": frames, "expanded": expanded, "width": width, "height": height,
            "out_width": out_width, "out_height": out_height, "coverage": motion_coverage,
            "timings": timings})


class FL_MiniMaxH3MotionSample(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FL_MiniMaxH3MotionSample", category="FL/MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[Prepared.Input("prepared"), *stage_inputs({"model", "positive", "latent", "vae",
                "strength", "steps", "context_budget", "seed", "context_overlap", "sampler_name",
                "scheduler", "audio_strength"}), ShotData.Input("shot_data", optional=True)],
            outputs=[io.Image.Output(display_name="images"), io.String.Output(display_name="report")])

    @classmethod
    def execute(cls, prepared, model, positive=None, latent=None, vae=None, strength=0.5, steps=25,
                context_budget=0, seed=20260902, context_overlap=17,
                sampler_name="res_multistep", scheduler="simple", audio_strength=0.5, shot_data=None):
        started = time.perf_counter()
        trim = 0
        if shot_data is not None:
            latent, positive = shot_data["latent"], shot_data["positive"]
            trim = shot_data["trim"]
            model = motion_context_model(model)
        if prepared["positive"] is not None:
            positive = prepared["positive"]
        if not isinstance(model.get_model_object("latent_format"), comfy.latent_formats.MiniMaxH3Video):
            raise ValueError(f"{_OWNER}: connect a MiniMax H3 model.")
        if not 0.05 <= strength <= 1 or not 4 <= steps <= 100 or not 0 <= audio_strength <= 1:
            raise ValueError(f"{_OWNER}: invalid strength, step budget, or audio strength.")
        frames, expanded = prepared["frames"], prepared["expanded"]
        out_width, out_height = prepared["out_width"], prepared["out_height"]
        holds = prepared["holds"]
        run_steps = max(1, round(steps * strength))
        sample_model, windows, window_t, overlap_t = context_model(model, h3.video_latent_t(expanded), context_budget, context_overlap)
        report = (f"{frames} -> {expanded} -> {frames} frames at 24 fps | {prepared['width']}x{prepared['height']} -> {out_width}x{out_height}\n"
                  f"{run_steps}/{steps} refinement steps | {prepared['coverage']} coverage | seed {seed}\n"
                  f"{windows} context window(s), {window_t} latent positions, {overlap_t} shared"
                  + (" (+1 causal anchor where available)" if windows > 1 else "")
                  + "\nOriginal audio stays on the export path. Frame recovery restores timing, not original pixels.")
        logging.info("%s: %s", _OWNER, report)
        comfy.model_management.throw_exception_if_processing_interrupted()
        init = motion_nodes()["H3V2VInit"].build(prepared["video"], audio_latent=prepared["audio"],
            audio_strength=audio_strength, audio_mode="custom (use audio_strength)")[0]
        conditioning = retime_conditioning(positive, latent, init, holds, shot_mode=shot_data is not None, context_prefix=trim)
        if trim:
            video, audio = h3_tensors(init, _OWNER)
            masks = init.get("noise_mask")
            vm, am = (part.clone() for part in masks.unbind()) if masks is not None else (torch.ones_like(video), torch.ones_like(audio))
            vm[:, :, :VIDEO_CONTEXT_STEPS[trim]] = 0
            am[..., :round(trim / h3.FPS * h3.AUDIO_LATENT_FPS)] = 0
            init["noise_mask"] = comfy.nested_tensor.NestedTensor((vm, am))
        guider = Guider_Basic(sample_model)
        guider.set_conds(conditioning)
        sigmas = comfy.samplers.calculate_sigmas(sample_model.get_model_object("model_sampling"), scheduler, steps)[steps - run_steps:]
        noise = Noise_RandomNoise(seed)
        sampler = comfy.samplers.sampler_object(sampler_name)
        callback = latent_preview.prepare_callback(sample_model, run_steps)
        sampled = guider.sample(noise.generate_noise(init), init["samples"], sampler, sigmas,
            denoise_mask=init.get("noise_mask"), callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
        sampled = sampled.to(comfy.model_management.intermediate_device())
        del init, guider, conditioning
        sampled_at = time.perf_counter()
        comfy.model_management.throw_exception_if_processing_interrupted()
        refined_video, _ = h3_tensors({"samples": sampled}, _OWNER)
        authored = shot_data["authored"] if shot_data is not None else frames
        recovered = vae.decode_h3_selected(refined_video, frame_edges(holds)[trim:trim + authored])[0]
        if tuple(recovered.shape) != (authored, out_height, out_width, 3):
            raise ValueError(f"{_OWNER}: recovered output has unexpected dimensions {tuple(recovered.shape)}.")
        timing = f"sample/setup {sampled_at-started:.2f}s, selected decode {time.perf_counter()-sampled_at:.2f}s"
        logging.info("%s render: %s", _OWNER, timing)
        report += "\nPreparation (cached when unchanged): " + prepared["timings"] + "\nThis render: " + timing
        if shot_data is not None:
            report = f"Shot {shot_data['index'] + 1}: {authored} authored frames, {trim} hidden prefix frames\n" + report
        return io.NodeOutput(recovered, report)
