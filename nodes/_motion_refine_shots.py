import torch
import torchaudio

import comfy.model_management
from comfy_api.latest import io
from comfy_extras import nodes_minimax_h3 as h3
from comfy_extras.nodes_audio import vae_decode_audio

from ._latent_helpers import h3_tensors
from ._motion_context import VIDEO_CONTEXT_STEPS
from .FL_MiniMaxH3PromptTimeline import H3ShotPlan


ShotData = io.Custom("FL_H3_MOTION_SHOT")
_OWNER = "FL MiniMax H3 Motion Refine"


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get("type") != "minimax_h3_beat_shot_plan" or plan.get("version") != 1:
        raise ValueError(f"{_OWNER}: connect a version 1 H3 beat shot plan.")
    if plan.get("mode") == "temporal_reshot":
        raise ValueError(f"{_OWNER}: temporal reshot plans are not supported.")
    shots = plan.get("shots")
    if not isinstance(shots, list) or not shots or plan.get("fps") != h3.FPS:
        raise ValueError(f"{_OWNER}: the shot plan must contain renders at 24 fps.")
    cursor = 0
    for index, shot in enumerate(shots):
        context = shot.get("motion_context") or {}
        trim = context.get("trim_frames", 0)
        authored = shot.get("authored_frames")
        render = shot.get("render_frames")
        if (shot.get("index") != index or shot.get("start_frame") != cursor
                or type(authored) is not int or authored <= 0
                or shot.get("end_frame") != cursor + authored
                or type(render) is not int or render < 5 or render % 17 != 5
                or trim not in VIDEO_CONTEXT_STEPS or trim != context.get("video_frames", 0)
                or trim + authored > render or not shot.get("conditioning") or shot.get("reshot")):
            raise ValueError(f"{_OWNER}: shot {index + 1} has invalid timing, context, or conditioning.")
        if context.get("video_frames", 0) or context.get("audio_frames", 0):
            if index == 0 or context.get("source_index") != index - 1:
                raise ValueError(f"{_OWNER}: shot {index + 1} must reference the previous baseline shot.")
        cursor += authored
    if cursor != plan.get("total_frames"):
        raise ValueError(f"{_OWNER}: shot ranges do not match the total frame count.")
    return shots


def validate_latents(plan, latents):
    shots = validate_plan(plan)
    if len(latents) != len(shots):
        raise ValueError(f"{_OWNER}: expected {len(shots)} rendered latents, received {len(latents)}.")
    size = None
    for index, (shot, latent) in enumerate(zip(shots, latents)):
        metadata = latent.get("fl_h3_shot", {})
        if (metadata.get("version") != 1 or metadata.get("total_frames") != plan["total_frames"]
                or metadata.get("fps") != h3.FPS or metadata.get("reshot")
                or any(metadata.get(key) != shot.get(key) for key in
                    ("index", "start_frame", "end_frame", "authored_frames", "render_frames", "motion_context"))):
            raise ValueError(f"{_OWNER}: latent {index + 1} does not match shot_plan. Connect the exact plan used by Beat KSampler, including Shot Motion Context.")
        video, audio = h3_tensors(latent, _OWNER)
        if (video.shape[2] != h3.video_latent_t(shot["render_frames"])
                or audio.shape[-1] != round(shot["render_frames"] / h3.FPS * h3.AUDIO_LATENT_FPS)):
            raise ValueError(f"{_OWNER}: latent {index + 1} has a different duration from its shot plan.")
        if size is not None and size != video.shape[-2:]:
            raise ValueError(f"{_OWNER}: all shots must have the same source resolution.")
        size = video.shape[-2:]
    return shots


def shot_images(vae, video, shot, baseline, total_frames):
    frames = shot["render_frames"]
    trim = (shot.get("motion_context") or {}).get("trim_frames", 0)
    authored = shot["authored_frames"]
    shape = (frames, video.shape[-2] * 16, video.shape[-1] * 16, 3)
    if baseline is None:
        images = vae.decode(video)[0]
        if tuple(images.shape) != shape:
            raise ValueError(f"{_OWNER}: video VAE decoded an unexpected shot shape.")
        return images
    if tuple(baseline.shape) != (total_frames, *shape[1:]):
        raise ValueError(f"{_OWNER}: baseline_images must be the full assembled timeline at source resolution.")
    images = baseline.new_empty(shape)
    images[trim:trim + authored].copy_(baseline[shot["start_frame"]:shot["end_frame"]])
    hidden = [*range(trim), *range(trim + authored, frames)]
    if hidden:
        decoded = vae.decode_h3_selected(video, hidden)[0]
        images[hidden] = decoded.to(device=images.device, dtype=images.dtype)
    return images


def shot_audio(audio_vae, latent, shot, baseline, total_frames):
    if baseline is None:
        return None
    if audio_vae is None:
        raise ValueError(f"{_OWNER}: connect audio_vae when supplying baseline_audio in shot-plan mode.")
    sr = baseline["sample_rate"]
    waveform = baseline["waveform"]
    if waveform.ndim != 3 or waveform.shape[0] != 1 or abs(waveform.shape[-1] / sr - total_frames / h3.FPS) > 1 / h3.AUDIO_LATENT_FPS:
        raise ValueError(f"{_OWNER}: baseline_audio must match the full assembled timeline duration.")
    decoded = vae_decode_audio(audio_vae, latent)
    audio = decoded["waveform"]
    if decoded["sample_rate"] != sr:
        audio = torchaudio.functional.resample(audio, decoded["sample_rate"], sr)
    length = round(shot["render_frames"] / h3.FPS * sr)
    if audio.shape[-1] < length:
        audio = torch.nn.functional.pad(audio, (0, length - audio.shape[-1]), mode="replicate")
    audio = audio[..., :length].clone()
    start, end = (round(shot[key] / h3.FPS * sr) for key in ("start_frame", "end_frame"))
    trim = (shot.get("motion_context") or {}).get("trim_frames", 0)
    offset = round(trim / h3.FPS * sr)
    count = min(end - start, length - offset)
    source = waveform[..., start:start + count]
    if source.shape[-1] != count or source.shape[1] not in (1, audio.shape[1]):
        raise ValueError(f"{_OWNER}: baseline_audio has too few samples or incompatible channels.")
    audio[..., offset:offset + count].copy_(source)
    return {"waveform": audio, "sample_rate": sr}


class FL_MiniMaxH3MotionShot(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FL_MiniMaxH3MotionShot", category="FL/MiniMax H3/Internal",
            is_dev_only=True, is_input_list=True,
            inputs=[io.Latent.Input("latents"), H3ShotPlan.Input("shot_plan"),
                io.Int.Input("index", default=0, min=0), io.Vae.Input("vae"),
                io.Vae.Input("audio_vae", optional=True), io.Image.Input("baseline_images", optional=True),
                io.Audio.Input("baseline_audio", optional=True)], outputs=[ShotData.Output()])

    @classmethod
    def execute(cls, latents, shot_plan, index, vae, audio_vae=None, baseline_images=None, baseline_audio=None):
        inputs = {"shot_plan": shot_plan, "index": index, "vae": vae,
            "audio_vae": audio_vae, "baseline_images": baseline_images, "baseline_audio": baseline_audio}
        for name, values in inputs.items():
            if values is not None and len(values) != 1:
                raise ValueError(f"{_OWNER}: {name} must have exactly one value; only latents accepts a list.")
        plan, index, vae = shot_plan[0], index[0], vae[0]
        shots = validate_latents(plan, latents)
        shot, latent = shots[index], latents[index]
        comfy.model_management.throw_exception_if_processing_interrupted()
        positive = shot["conditioning"]
        if baseline_audio and baseline_audio[0] is not None and (not audio_vae or audio_vae[0] is None):
            raise ValueError(f"{_OWNER}: connect audio_vae when supplying baseline_audio in shot-plan mode.")
        return io.NodeOutput({"latent": latent, "positive": positive, "shot": shot, "total_frames": plan["total_frames"],
            "previous": latents[index - 1] if index else None,
            "source_shot": shots[index - 1] if index else None,
            "baseline_images": baseline_images[0] if baseline_images else None,
            "baseline_audio": baseline_audio[0] if baseline_audio else None,
            "trim": (shot.get("motion_context") or {}).get("trim_frames", 0),
            "authored": shot["authored_frames"], "index": index})


class FL_MiniMaxH3MotionCollect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FL_MiniMaxH3MotionCollect", category="FL/MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[io.Int.Input("total_frames", default=1, min=1),
                io.Autogrow.Input("images", template=io.Autogrow.TemplatePrefix(
                    io.Image.Input("image"), prefix="shot_", max=io.Autogrow._MaxNames)),
                io.Autogrow.Input("reports", template=io.Autogrow.TemplatePrefix(
                    io.String.Input("report"), prefix="shot_", max=io.Autogrow._MaxNames))],
            outputs=[io.Image.Output(), io.String.Output()])

    @classmethod
    def execute(cls, total_frames, images, reports):
        keys = [f"shot_{index}" for index in range(len(images))]
        if set(images) != set(keys) or set(reports) != set(keys):
            raise ValueError(f"{_OWNER}: assembled shots are incomplete or out of order.")
        if sum(images[key].shape[0] for key in keys) != total_frames:
            raise ValueError(f"{_OWNER}: recovered shots do not match the authored duration.")
        output = torch.cat([images[key] for key in keys], dim=0)
        report = f"{len(keys)} shots assembled into {total_frames} frames at 24 fps; original cuts preserved.\n\n"
        return io.NodeOutput(output, report + "\n\n".join(reports[key] for key in keys))
