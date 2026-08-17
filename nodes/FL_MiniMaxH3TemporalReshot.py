import bisect
import json
import math
import os
from fractions import Fraction
from pathlib import Path

import av
import psutil
import torch

import comfy.nested_tensor
import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl, Types, io
from comfy_extras import nodes_minimax_h3 as minimax_h3

from .FL_MiniMaxH3PromptTimeline import (
    H3ShotPlan,
    _encode_prompt,
    _prepare_references,
    _video_token_edges,
)


_OWNER = "FL MiniMax H3 Temporal Reshot"
VIDEO_EXTENSIONS = {".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
DEFAULT_RESHOT_SETTINGS = {
    "version": 1,
    "start_frame": 0,
    "frame_count": 72,
    "context_before": 39,
    "context_after": 39,
    "edge_blend_frames": 0,
}
DEFAULT_RESHOT_SETTINGS_JSON = json.dumps(DEFAULT_RESHOT_SETTINGS, separators=(",", ":"))


def available_video_files():
    input_dir = Path(folder_paths.get_input_directory()).resolve()
    os.makedirs(input_dir, exist_ok=True)
    files, _ = folder_paths.recursive_search(str(input_dir))
    result = []
    for filename in files:
        if Path(filename).suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        path = (input_dir / filename).resolve()
        try:
            path.relative_to(input_dir)
        except ValueError:
            continue
        if path.is_file():
            result.append(Path(filename).as_posix())
    return sorted(result, key=str.casefold)


def video_library_entries():
    entries = []
    for filename in available_video_files():
        entry = {
            "path": filename,
            "filename": Path(filename).name,
            "folder": Path(filename).parent.as_posix()
            if Path(filename).parent != Path(".")
            else "",
        }
        try:
            path = resolve_video_path(filename)
            entry["size"] = path.stat().st_size
            with av.open(str(path), mode="r") as container:
                if not container.streams.video:
                    raise ValueError("No video stream")
                stream = container.streams.video[0]
                rate = Fraction(stream.average_rate) if stream.average_rate else Fraction(1)
                if container.duration is not None:
                    duration = float(container.duration / av.time_base)
                elif stream.duration is not None and stream.time_base is not None:
                    duration = float(stream.duration * stream.time_base)
                elif stream.frames:
                    duration = float(Fraction(stream.frames, 1) / rate)
                else:
                    duration = 0.0
                entry.update({
                    "width": int(stream.width),
                    "height": int(stream.height),
                    "duration": duration,
                    "frame_rate": float(rate),
                    "frame_count": int(stream.frames) if stream.frames else int(round(duration * float(rate))),
                    "has_audio": bool(container.streams.audio),
                    "declared_24_fps": rate == minimax_h3.FPS,
                })
        except (OSError, ValueError, av.error.FFmpegError) as error:
            entry["error"] = str(error)
        entries.append(entry)
    return entries


def resolve_video_path(filename):
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(f"{_OWNER}: choose a source video.")

    filename = filename.strip()
    name, annotated_dir = folder_paths.annotated_filepath(filename)
    input_dir = Path(folder_paths.get_input_directory()).resolve()
    if annotated_dir is not None and Path(annotated_dir).resolve() != input_dir:
        raise ValueError(f"{_OWNER}: source videos must be inside the ComfyUI input directory.")
    try:
        path = Path(folder_paths.get_annotated_filepath(name, str(input_dir))).resolve()
        path.relative_to(input_dir)
    except ValueError as error:
        raise ValueError(
            f"{_OWNER}: source videos must be inside the ComfyUI input directory."
        ) from error
    if not path.is_file():
        raise ValueError(f"{_OWNER}: source video does not exist: {filename}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"{_OWNER}: unsupported video format: {path.suffix or filename}")
    return path


def source_fingerprint(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def probe_source_video(path):
    path = Path(path)
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"{_OWNER}: no video stream found in {path.name}.")
        stream = container.streams.video[0]
        rate = Fraction(stream.average_rate) if stream.average_rate else Fraction(1)
        if stream.duration is not None and stream.time_base is not None:
            source_duration = float(stream.duration * stream.time_base)
        elif stream.frames:
            source_duration = float(Fraction(stream.frames, 1) / rate)
        elif container.duration is not None:
            source_duration = float(container.duration / av.time_base)
        else:
            source_duration = 0.0
        source_frame_count_estimated = not bool(stream.frames)
        source_frame_count = (
            int(stream.frames)
            if stream.frames
            else int(round(source_duration * float(rate)))
        )
        width = int(stream.width)
        height = int(stream.height)
        codec = stream.codec_context.name if stream.codec_context is not None else ""
        container_format = container.format.name or ""
        has_audio = bool(container.streams.audio)
        timestamps = sorted(
            float(packet.pts * stream.time_base)
            for packet in container.demux(stream)
            if packet.pts is not None
        )
        source_frame_interval = 1 / float(rate)
        source_constant_frame_rate = all(
            abs(second - first - source_frame_interval) <= max(0.0015, source_frame_interval * 0.05)
            for first, second in zip(timestamps, timestamps[1:])
        )
        if source_frame_count_estimated and timestamps:
            source_frame_count = len(timestamps)
            source_frame_count_estimated = False
    if source_frame_count <= 0:
        source_frame_count = InputImpl.VideoFromFile(str(path)).get_frame_count()
        source_frame_count_estimated = True
    if source_duration <= 0:
        source_duration = source_frame_count / float(rate)
    converted_to_24_fps = rate != minimax_h3.FPS or not source_constant_frame_rate
    frame_count = (
        max(1, round(source_duration * minimax_h3.FPS))
        if converted_to_24_fps
        else source_frame_count
    )
    bit_depth = int(InputImpl.VideoFromFile(str(path)).get_bit_depth())
    return {
        "width": width,
        "height": height,
        "duration": frame_count / minimax_h3.FPS,
        "frame_rate": float(minimax_h3.FPS),
        "frame_rate_numerator": minimax_h3.FPS,
        "frame_rate_denominator": 1,
        "frame_count": frame_count,
        "frame_count_estimated": source_frame_count_estimated,
        "constant_frame_rate": True,
        "source_duration": source_duration,
        "source_frame_rate": float(rate),
        "source_frame_rate_numerator": rate.numerator,
        "source_frame_rate_denominator": rate.denominator,
        "source_frame_count": source_frame_count,
        "source_constant_frame_rate": source_constant_frame_rate,
        "converted_to_24_fps": converted_to_24_fps,
        "bit_depth": bit_depth,
        "codec": codec,
        "container": container_format,
        "has_audio": has_audio,
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    }


def parse_reshot_settings(value):
    try:
        configured = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{_OWNER}: reshot settings are not valid JSON.") from error
    if not isinstance(configured, dict):
        raise ValueError(f"{_OWNER}: reshot settings must be a JSON object.")
    settings = DEFAULT_RESHOT_SETTINGS.copy()
    settings.update(configured)
    if settings["version"] != 1 or isinstance(settings["version"], bool):
        raise ValueError(f"{_OWNER}: reshot settings version 1 is required.")
    for name in (
        "start_frame",
        "frame_count",
        "context_before",
        "context_after",
        "edge_blend_frames",
    ):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{_OWNER}: {name} must be an integer.")
    if settings["start_frame"] < 0:
        raise ValueError(f"{_OWNER}: start_frame cannot be negative.")
    if settings["frame_count"] < 1:
        raise ValueError(f"{_OWNER}: frame_count must be at least 1.")
    if settings["context_before"] < 0 or settings["context_after"] < 0:
        raise ValueError(f"{_OWNER}: context frame counts cannot be negative.")
    if settings["edge_blend_frames"] < 0:
        raise ValueError(f"{_OWNER}: edge_blend_frames cannot be negative.")
    if settings["edge_blend_frames"] > settings["frame_count"]:
        raise ValueError(f"{_OWNER}: edge_blend_frames cannot exceed frame_count.")
    return settings


def build_reshot_window(source_frames, settings):
    start = settings["start_frame"]
    end = start + settings["frame_count"]
    if start >= source_frames or end > source_frames:
        raise ValueError(
            f"{_OWNER}: selected frames {start}-{end - 1} exceed the {source_frames}-frame source."
        )

    requested_start = max(0, start - settings["context_before"])
    requested_end = min(source_frames, end + settings["context_after"])
    render_frames = minimax_h3.align_frame_count(max(5, requested_end - requested_start))
    work_start = requested_start
    if work_start + render_frames > source_frames:
        work_start = max(0, source_frames - render_frames)
    work_source_frames = min(render_frames, source_frames - work_start)
    selection_offset = start - work_start
    if selection_offset < 0 or selection_offset + settings["frame_count"] > work_source_frames:
        raise ValueError(f"{_OWNER}: could not place the selected range inside its render window.")
    return {
        "work_start_frame": work_start,
        "work_source_frames": work_source_frames,
        "render_frames": render_frames,
        "padding_frames": render_frames - work_source_frames,
        "selection_offset": selection_offset,
        "selection_frames": settings["frame_count"],
        "requested_start_frame": requested_start,
        "requested_end_frame": requested_end,
    }


def selection_token_range(video_t, render_frames, selection_offset, selection_frames):
    edges = _video_token_edges(video_t)
    if edges[-1] != render_frames:
        raise ValueError(
            f"{_OWNER}: {video_t} video tokens resolve to {edges[-1]} frames, expected {render_frames}."
        )
    selection_end = selection_offset + selection_frames
    token_start = max(0, bisect.bisect_right(edges, selection_offset) - 1)
    token_end = min(video_t, bisect.bisect_left(edges, selection_end))
    if token_end <= token_start:
        token_end = min(video_t, token_start + 1)
    return token_start, token_end, edges


def _resample_frames(frames, frame_count):
    if not isinstance(frames, torch.Tensor) or frames.ndim < 1 or frames.shape[0] < 1:
        raise ValueError(f"{_OWNER}: decoded video contains no frames.")
    if frames.shape[0] == frame_count:
        return frames
    source_count = frames.shape[0]
    indices = [
        min(source_count - 1, ((2 * index + 1) * source_count) // (2 * frame_count))
        for index in range(frame_count)
    ]
    return frames[indices]


def _decode_working_components(path, window):
    source = InputImpl.VideoFromFile(
        str(path),
        start_time=window["work_start_frame"] / minimax_h3.FPS,
        duration=window["work_source_frames"] / minimax_h3.FPS,
    )
    components = source.get_components()
    images = components.images
    expected = window["work_source_frames"]
    if images.ndim != 4 or images.shape[-1] < 3 or images.shape[0] < 1:
        raise ValueError(
            f"{_OWNER}: decoded no usable frames from the selected source interval."
        )
    images = _resample_frames(images[..., :3], expected)
    padding = window["padding_frames"]
    if padding:
        images = torch.cat((images, images[-1:].expand(padding, -1, -1, -1)), dim=0)
    return components, images


def _working_audio_latent(audio_vae, audio, render_frames, expected):
    if audio_vae is None or audio is None:
        return expected
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError(f"{_OWNER}: the source audio waveform is invalid.")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"{_OWNER}: the source audio sample rate is invalid.")
    target_samples = round(render_frames * sample_rate / minimax_h3.FPS)
    if waveform.shape[-1] < target_samples:
        padded = torch.zeros(
            (*waveform.shape[:-1], target_samples),
            dtype=waveform.dtype,
            device=waveform.device,
        )
        padded[..., :waveform.shape[-1]].copy_(waveform)
        waveform = padded
    else:
        waveform = waveform[..., :target_samples]
    encoded, _ = minimax_h3._encode_ref_audio(
        audio_vae,
        {"waveform": waveform, "sample_rate": sample_rate},
    )
    if encoded.shape != expected.shape:
        raise ValueError(
            f"{_OWNER}: source audio encoded to {tuple(encoded.shape)}, expected {tuple(expected.shape)}."
        )
    return encoded


def _source_video_latent(vae, images, canvas_width, canvas_height, render_frames):
    resized = minimax_h3._resize(images, canvas_width, canvas_height, "disabled")
    video = vae.encode(resized)
    expected_t = minimax_h3.video_latent_t(render_frames)
    if (
        not isinstance(video, torch.Tensor)
        or video.ndim != 5
        or video.shape[0] != 1
        or video.shape[1] != 24
        or video.shape[2] != expected_t
        or video.shape[-2] != canvas_height // 16
        or video.shape[-1] != canvas_width // 16
    ):
        shape = tuple(video.shape) if isinstance(video, torch.Tensor) else type(video).__name__
        raise ValueError(f"{_OWNER}: video VAE returned invalid H3 latent shape {shape}.")
    return video


def _reshot_noise_mask(video, audio, token_start, token_end):
    video_mask = torch.zeros_like(video)
    video_mask[:, :, token_start:token_end].fill_(1)
    audio_mask = torch.zeros_like(audio)
    return comfy.nested_tensor.NestedTensor((video_mask, audio_mask))


def _validate_plan_and_latent(shot_plan, latent):
    if (
        not isinstance(shot_plan, dict)
        or shot_plan.get("type") != "minimax_h3_beat_shot_plan"
        or shot_plan.get("version") != 1
        or shot_plan.get("mode") != "temporal_reshot"
    ):
        raise ValueError(
            "FL MiniMax H3 Temporal Reshot Assembler requires a Temporal Reshot Planner plan."
        )
    shots = shot_plan.get("shots")
    if not isinstance(shots, list) or len(shots) != 1:
        raise ValueError("FL MiniMax H3 Temporal Reshot Assembler requires exactly one reshot render.")
    metadata = latent.get("fl_h3_shot") if isinstance(latent, dict) else None
    if not isinstance(metadata, dict) or metadata.get("version") != 1:
        raise ValueError(
            "FL MiniMax H3 Temporal Reshot Assembler requires a latent from the Beat KSampler."
        )
    shot = shots[0]
    if (
        metadata.get("index") != 0
        or metadata.get("start_frame") != shot.get("start_frame")
        or metadata.get("end_frame") != shot.get("end_frame")
        or metadata.get("render_frames") != shot.get("render_frames")
        or metadata.get("reshot") != shot.get("reshot")
    ):
        raise ValueError(
            "FL MiniMax H3 Temporal Reshot Assembler latent does not match the connected plan."
        )
    return shot, metadata


def _video_latent(latent):
    samples = latent.get("samples")
    if not isinstance(samples, comfy.nested_tensor.NestedTensor):
        raise TypeError("FL MiniMax H3 Temporal Reshot Assembler requires a nested H3 latent.")
    tensors = samples.unbind()
    if len(tensors) != 2:
        raise ValueError("FL MiniMax H3 Temporal Reshot Assembler requires video and audio latents.")
    video = tensors[0]
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
        raise ValueError("FL MiniMax H3 Temporal Reshot Assembler received an invalid video latent.")
    return video


def _flatten_decoded(images):
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            "FL MiniMax H3 Temporal Reshot Assembler expected decoded RGB video frames."
        )
    return images


def _resize_to_source(images, width, height):
    if (images.shape[2], images.shape[1]) == (width, height):
        return images
    return comfy.utils.common_upscale(
        images.movedim(-1, 1), width, height, "lanczos", "disabled"
    ).movedim(1, -1)


def _check_source_decode_memory(source):
    frame_count = source.get("frame_count")
    source_frame_count = source.get("source_frame_count", frame_count)
    width = source.get("width")
    height = source.get("height")
    if not all(
        isinstance(value, int) and value > 0
        for value in (frame_count, source_frame_count, width, height)
    ):
        return
    frame_bytes = width * height * 3 * 4
    estimated_peak = (source_frame_count + frame_count) * frame_bytes
    available = psutil.virtual_memory().available
    if estimated_peak > available:
        raise MemoryError(
            "FL MiniMax H3 Temporal Reshot Assembler needs about "
            f"{estimated_peak / 1024 ** 3:.1f} GB of free system memory to decode this source; "
            f"{available / 1024 ** 3:.1f} GB is available. Use a shorter or smaller source."
        )


def _blend_selection(source, generated, blend_frames):
    if blend_frames == 0:
        source.copy_(generated)
        return
    count = source.shape[0]
    positions = torch.arange(count, dtype=generated.dtype, device=generated.device)
    left = ((positions + 1) / (blend_frames + 1)).clamp(max=1)
    right = ((count - positions) / (blend_frames + 1)).clamp(max=1)
    amount = torch.minimum(left, right)
    amount = 0.5 - 0.5 * torch.cos(math.pi * amount)
    source.mul_(1 - amount[:, None, None, None]).add_(generated * amount[:, None, None, None])


class _TemporalReshotVideo(InputImpl.VideoFromComponents):
    def __init__(self, components, bit_depth):
        super().__init__(components, bit_depth=bit_depth)
        self._reshot_components = components

    def get_components(self):
        return Types.VideoComponents(
            images=self._reshot_components.images,
            audio=self._reshot_components.audio,
            frame_rate=self._reshot_components.frame_rate,
            metadata=self._reshot_components.metadata,
            alpha=self._reshot_components.alpha,
        )


class FL_MiniMaxH3TemporalReshotPlanner(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3TemporalReshotPlanner",
            display_name="FL MiniMax H3 Temporal Reshot Planner",
            category="FL/MiniMax H3/Prompting",
            description=(
                "Normalizes a source video to 24 fps, selects a full-frame interval, and prepares "
                "a native H3 temporal inpaint latent with surrounding motion context."
            ),
            inputs=[
                io.Clip.Input("clip", tooltip="MiniMax H3 text encoder for the replacement prompt."),
                io.Vae.Input("vae", tooltip="MiniMax H3 video VAE used to encode the source window."),
                io.Combo.Input(
                    "video",
                    options=["", *available_video_files()],
                    upload=io.UploadType.video,
                    tooltip="Source video. It is normalized to 24 fps before the selected range is replaced.",
                ),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="New action, performance, camera motion, or scene direction for the interval.",
                ),
                io.String.Input(
                    "reshot_settings",
                    default=DEFAULT_RESHOT_SETTINGS_JSON,
                    multiline=False,
                    tooltip="Timeline selection state managed by the temporal reshot interface.",
                ),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                    tooltip="Reference image sizing, matching the standard H3 reference node.",
                ),
                io.Vae.Input(
                    "audio_vae",
                    optional=True,
                    tooltip="Optional H3 audio VAE. When connected, source audio conditions the reshot.",
                ),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    tooltip="Optional character, wardrobe, prop, or scene references for the new interval.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"),
                        prefix="ref_image_",
                        min=0,
                        max=9,
                    ),
                ),
            ],
            outputs=[H3ShotPlan.Output(display_name="shot_plan")],
        )

    @classmethod
    def execute(
        cls,
        clip,
        vae,
        video,
        prompt,
        reshot_settings=DEFAULT_RESHOT_SETTINGS_JSON,
        ref_image_size="match",
        audio_vae=None,
        ref_images=None,
    ):
        path = resolve_video_path(video)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{_OWNER}: enter a prompt for the replacement interval.")
        probe = probe_source_video(path)
        settings = parse_reshot_settings(reshot_settings)
        window = build_reshot_window(probe["frame_count"], settings)
        components, images = _decode_working_components(path, window)
        canvas_width, canvas_height = minimax_h3.adapt_canvas(probe["width"], probe["height"])
        video_latent = _source_video_latent(
            vae,
            images,
            canvas_width,
            canvas_height,
            window["render_frames"],
        )
        empty, _ = minimax_h3._empty_av_latent(
            canvas_width,
            canvas_height,
            window["render_frames"],
        )
        empty_audio = empty["samples"].unbind()[1]
        audio_latent = _working_audio_latent(
            audio_vae,
            components.audio,
            window["render_frames"],
            empty_audio,
        )
        token_start, token_end, token_edges = selection_token_range(
            video_latent.shape[2],
            window["render_frames"],
            window["selection_offset"],
            window["selection_frames"],
        )
        latent = {
            "samples": comfy.nested_tensor.NestedTensor((video_latent, audio_latent)),
            "noise_mask": _reshot_noise_mask(video_latent, audio_latent, token_start, token_end),
        }
        ref_items, ref_blocks = _prepare_references(
            vae,
            audio_vae,
            canvas_width,
            canvas_height,
            window["render_frames"],
            ref_image_size,
            ref_images,
            None,
            None,
            None,
        )
        conditioning = _encode_prompt(clip, prompt.strip(), ref_items, ref_blocks)
        reshot = {
            "version": 1,
            **window,
            "edit_token_start": token_start,
            "edit_token_end": token_end,
            "token_frame_edges": token_edges,
            "edit_start_frame": window["work_start_frame"] + token_edges[token_start],
            "edit_end_frame": min(
                probe["frame_count"],
                window["work_start_frame"] + token_edges[token_end],
            ),
            "edge_blend_frames": settings["edge_blend_frames"],
            "source_audio_conditioned": audio_vae is not None and components.audio is not None,
        }
        shot = {
            "index": 0,
            "start_frame": settings["start_frame"],
            "end_frame": settings["start_frame"] + settings["frame_count"],
            "authored_frames": settings["frame_count"],
            "render_frames": window["render_frames"],
            "padding_frames": window["padding_frames"],
            "prompt": prompt.strip(),
            "conditioning": conditioning,
            "latent": latent,
            "reshot": reshot,
        }
        source = {
            "filename": video,
            "fingerprint": source_fingerprint(path),
            **probe,
        }
        plan = {
            "type": "minimax_h3_beat_shot_plan",
            "version": 1,
            "mode": "temporal_reshot",
            "fps": minimax_h3.FPS,
            "width": canvas_width,
            "height": canvas_height,
            "total_frames": probe["frame_count"],
            "total_render_frames": window["render_frames"],
            "source": source,
            "shots": [shot],
        }
        return io.NodeOutput(plan)

    @classmethod
    def fingerprint_inputs(cls, video, **kwargs):
        try:
            path = resolve_video_path(video)
            fingerprint = source_fingerprint(path)
            return f"{fingerprint['size']}:{fingerprint['mtime_ns']}"
        except (OSError, ValueError):
            return float("nan")

    @classmethod
    def validate_inputs(
        cls,
        video,
        prompt="",
        reshot_settings=DEFAULT_RESHOT_SETTINGS_JSON,
        **kwargs,
    ):
        try:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{_OWNER}: enter a prompt for the replacement interval.")
            path = resolve_video_path(video)
            settings = parse_reshot_settings(reshot_settings)
            probe = probe_source_video(path)
            build_reshot_window(probe["frame_count"], settings)
        except (OSError, ValueError, av.error.FFmpegError) as error:
            return str(error)
        return True


class FL_MiniMaxH3TemporalReshotAssembler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3TemporalReshotAssembler",
            display_name="FL MiniMax H3 Temporal Reshot Assembler",
            category="FL/MiniMax H3/Output",
            description=(
                "Replaces only the selected source frames with the sampled H3 interval while "
                "preserving every frame outside it and retaining the source soundtrack."
            ),
            inputs=[
                H3ShotPlan.Input("shot_plan", tooltip="The exact Temporal Reshot Planner output."),
                io.Latent.Input("latent", tooltip="Sampled latent from FL MiniMax H3 Beat KSampler."),
                io.Vae.Input("vae", tooltip="MiniMax H3 video VAE used to decode the reshot."),
            ],
            outputs=[io.Video.Output(display_name="video")],
        )

    @classmethod
    def execute(cls, shot_plan, latent, vae):
        shot, _ = _validate_plan_and_latent(shot_plan, latent)
        source = shot_plan.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("filename"), str):
            raise ValueError("FL MiniMax H3 Temporal Reshot Assembler received no source video.")
        path = resolve_video_path(source["filename"])
        if source_fingerprint(path) != source.get("fingerprint"):
            raise ValueError(
                "FL MiniMax H3 Temporal Reshot Assembler source video changed after planning. "
                "Run the planner and sampler again."
            )
        reshot = shot["reshot"]
        decoded = _flatten_decoded(vae.decode(_video_latent(latent)))
        offset = reshot["selection_offset"]
        count = reshot["selection_frames"]
        if decoded.shape[0] < offset + count:
            raise ValueError(
                "FL MiniMax H3 Temporal Reshot Assembler decoded too few frames for the selection."
            )
        generated = decoded[offset:offset + count]

        _check_source_decode_memory(source)
        source_input = InputImpl.VideoFromFile(str(path))
        components = source_input.get_components()
        images = _resample_frames(components.images, shot_plan["total_frames"])
        if images.ndim != 4:
            raise ValueError(
                "FL MiniMax H3 Temporal Reshot Assembler could not decode source video frames."
            )
        alpha = components.alpha
        if isinstance(alpha, torch.Tensor):
            alpha = _resample_frames(alpha, shot_plan["total_frames"])
        generated = _resize_to_source(generated, images.shape[2], images.shape[1]).to(
            device=images.device,
            dtype=images.dtype,
        )
        start = shot["start_frame"]
        end = shot["end_frame"]
        blend_frames = reshot["edge_blend_frames"]
        if blend_frames > count:
            raise ValueError(
                "FL MiniMax H3 Temporal Reshot Assembler edge blend cannot exceed the selection length."
            )
        _blend_selection(images[start:end], generated, blend_frames)
        output = _TemporalReshotVideo(
            Types.VideoComponents(
                images=images,
                audio=components.audio,
                frame_rate=Fraction(minimax_h3.FPS, 1),
                metadata=components.metadata,
                alpha=alpha,
            ),
            bit_depth=source["bit_depth"],
        )
        return io.NodeOutput(output)
