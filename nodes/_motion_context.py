import torch

import comfy.ldm.minimax.model as minimax_model
import comfy.nested_tensor
import comfy.patcher_extension
from comfy_extras import nodes_minimax_h3 as minimax_h3


VIDEO_CONTEXT_FRAMES = (0, 1, 5, 22, 39)
VIDEO_CONTEXT_STEPS = {0: 0, 1: 1, 5: 2, 22: 7, 39: 12}
VIDEO_FRAME_MARKER = "fl_motion_context_frame_index"
AUDIO_END_FRAME_MARKER = "fl_motion_context_audio_end_frame"
_WRAPPER_KEY = "fl_minimax_h3_motion_context"
_VIDEO_CHUNK_STEPS = len(minimax_model.FRAME_PER_TOKEN)
_VIDEO_CHUNK_FRAMES = sum(minimax_model.FRAME_PER_TOKEN)
_VIDEO_CHUNK_OVERLAP_STEPS = 2


def _has_motion_context(keyframes, refs):
    return any(VIDEO_FRAME_MARKER in keyframe for keyframe in (keyframes or ())) or any(
        AUDIO_END_FRAME_MARKER in ref for ref in (refs or ())
    )


def _target_origin(text_len, refs):
    cursor = float(text_len)
    for ref in refs or ():
        kind = ref["kind"]
        if kind == "image":
            cursor += 1.0
        elif kind == "audio":
            cursor += float(ref["ref_audio_t"])
        elif kind in ("video", "video_audio"):
            cursor += max(
                float(ref["ref_audio_t"]),
                sum(minimax_model._video_t_spans(ref["latent_t"])),
            )
    return cursor


def _reference_segments(layout, keyframes, refs):
    segments = layout.segments[1 + len(keyframes or ()):]
    position = 0
    resolved = []
    for ref in refs or ():
        spans = []
        kind = ref["kind"]
        expected = []
        if kind == "image":
            expected.append("ref_img")
        elif kind == "audio":
            if ref["ref_audio_t"] > 0:
                expected.append("ref_audio")
        elif kind in ("video", "video_audio"):
            if ref["ref_audio_t"] > 0:
                expected.append("ref_audio")
            expected.append("ref_img")
        for expected_kind in expected:
            if position >= len(segments) or segments[position][2] != expected_kind:
                raise RuntimeError(
                    "FL MiniMax H3 motion context could not map the current H3 reference layout."
                )
            start, end, segment_kind = segments[position]
            position += 1
            spans.append((start, end, segment_kind))
        resolved.append((ref, spans))
    return resolved


def _apply_layout_context(layout, text_len, keyframes, refs):
    keyframes = keyframes or ()
    refs = refs or ()
    if not _has_motion_context(keyframes, refs):
        return

    marked_keyframes = [keyframe for keyframe in keyframes if VIDEO_FRAME_MARKER in keyframe]
    if refs and marked_keyframes and len(marked_keyframes) != len(keyframes):
        raise ValueError(
            "FL MiniMax H3 motion context cannot mix its anchored frames with stock H3 "
            "keyframes when references are also connected."
        )

    origin = _target_origin(text_len, refs)
    cond_segments = [segment for segment in layout.segments if segment[2] == "cond"]
    for keyframe, (start, end, _) in zip(keyframes, cond_segments):
        if VIDEO_FRAME_MARKER in keyframe:
            frame = keyframe[VIDEO_FRAME_MARKER]
            layout.position_ids[start:end, 0] = origin + minimax_model.FRAME_RESCALE * frame

    for ref, spans in _reference_segments(layout, keyframes, refs):
        if AUDIO_END_FRAME_MARKER not in ref:
            continue
        audio_span = next((span for span in spans if span[2] == "ref_audio"), None)
        if audio_span is None:
            raise ValueError("FL MiniMax H3 motion context received an empty audio context.")
        start, end, _ = audio_span
        audio_t = ref["ref_audio_t"]
        end_time = origin + minimax_model.FRAME_RESCALE * ref[AUDIO_END_FRAME_MARKER]
        times = end_time - audio_t + torch.arange(audio_t, dtype=torch.float64)
        layout.position_ids[start:end, 0] = times.repeat(2)


def _prepare_payload(payload, cross_attn, latent_shapes):
    keyframes = payload.get("keyframes")
    refs = payload.get("refs")
    if not _has_motion_context(keyframes, refs):
        return
    if cross_attn is None or latent_shapes is None or len(latent_shapes) < 2:
        raise RuntimeError("FL MiniMax H3 motion context could not resolve the H3 layout inputs.")

    video_shape = latent_shapes[0]
    signature = (
        cross_attn.shape[1],
        video_shape[2],
        (video_shape[3] + 1) // 2 * 2,
        (video_shape[4] + 1) // 2 * 2,
        latent_shapes[1][-1],
    )
    layout = payload.get("layout")
    if layout is None or layout.signature != signature:
        resolved_keyframes = [
            {**keyframe, "resolved_frame_index": 0}
            if VIDEO_FRAME_MARKER in keyframe else keyframe
            for keyframe in (keyframes or ())
        ]
        layout = minimax_model.PackedLayout(
            *signature,
            keyframes=resolved_keyframes,
            refs=refs,
            frame_count=payload.get("frame_count"),
        )
        payload["layout"] = layout

    _apply_layout_context(layout, signature[0], keyframes, refs)
    payload["cond_video_latents"] = [
        keyframe["latent"] for keyframe in (keyframes or ())
    ] + [ref["latent"] for ref in (refs or ()) if "latent" in ref]
    payload["cond_audio_latents"] = [
        ref["audio_latent"]
        for ref in (refs or ())
        if ref.get("audio_latent") is not None
    ]


def _model_wrapper(executor, x, timestep, c_concat=None, c_crossattn=None, control=None,
                   transformer_options={}, **kwargs):
    payload = kwargs.get("minimax_payload")
    if isinstance(payload, dict):
        _prepare_payload(payload, c_crossattn, kwargs.get("latent_shapes"))
    return executor(
        x,
        timestep,
        c_concat,
        c_crossattn,
        control,
        transformer_options,
        **kwargs,
    )


def motion_context_model(model):
    patched = model.clone()
    wrapper_type = comfy.patcher_extension.WrappersMP.APPLY_MODEL
    if not patched.get_wrappers(wrapper_type, _WRAPPER_KEY):
        patched.add_wrapper_with_key(wrapper_type, _WRAPPER_KEY, _model_wrapper)
    return patched


def _h3_tensors(latent):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not isinstance(samples, comfy.nested_tensor.NestedTensor):
        raise TypeError("FL MiniMax H3 motion context expects a nested H3 latent.")
    tensors = samples.unbind()
    if len(tensors) != 2:
        raise ValueError("FL MiniMax H3 motion context expects video and audio latent streams.")
    return tensors


def _decode_video_window(video, start_frame, end_frame, vae):
    if video.shape[2] < _VIDEO_CHUNK_STEPS + _VIDEO_CHUNK_OVERLAP_STEPS:
        return vae.decode(video), start_frame, end_frame

    chunk_count = (video.shape[2] - _VIDEO_CHUNK_OVERLAP_STEPS) // _VIDEO_CHUNK_STEPS
    first_chunk = min(start_frame // _VIDEO_CHUNK_FRAMES, chunk_count - 1)
    last_chunk = min((end_frame - 1) // _VIDEO_CHUNK_FRAMES, chunk_count - 1)
    decode_chunk = max(0, first_chunk - 1)
    latent_start = decode_chunk * _VIDEO_CHUNK_STEPS
    latent_end = (
        last_chunk * _VIDEO_CHUNK_STEPS
        + _VIDEO_CHUNK_STEPS
        + _VIDEO_CHUNK_OVERLAP_STEPS
    )
    images = vae.decode(video[:, :, latent_start:latent_end])
    frame_offset = decode_chunk * _VIDEO_CHUNK_FRAMES
    return images, start_frame - frame_offset, end_frame - frame_offset


def _video_keyframes(previous, authored_frames, trim_frames, context_frames, vae):
    video, _ = _h3_tensors(previous)
    authored_end = trim_frames + authored_frames
    images, context_start, context_end = _decode_video_window(
        video,
        authored_end - context_frames,
        authored_end,
        vae,
    )
    if images.ndim != 5 or images.shape[0] != 1 or images.shape[-1] != 3:
        raise ValueError(
            "FL MiniMax H3 motion context requires a video VAE that decodes to "
            "[1, frames, height, width, RGB]."
        )
    if context_start < 0 or images.shape[1] < context_end:
        raise ValueError("FL MiniMax H3 motion context decoded too few source frames.")
    encoded = vae.encode(images[0, context_start:context_end])
    expected_steps = VIDEO_CONTEXT_STEPS[context_frames]
    if encoded.ndim != 5 or encoded.shape[0] != 1 or encoded.shape[2] != expected_steps:
        raise ValueError(
            "FL MiniMax H3 motion context VAE encoded "
            f"{context_frames} frames to an unexpected temporal shape {tuple(encoded.shape)}."
        )

    offsets = []
    frame = 0
    for step in range(expected_steps):
        offsets.append(frame)
        frame += minimax_model.FRAME_PER_TOKEN[step % len(minimax_model.FRAME_PER_TOKEN)]
    return [
        {
            "resolved_frame_index": 0,
            VIDEO_FRAME_MARKER: offset,
            "latent": encoded[:, :, step:step + 1].clone(),
        }
        for step, offset in enumerate(offsets)
    ]


def _audio_reference(previous, authored_frames, trim_frames, audio_frames, video_frames):
    _, audio = _h3_tensors(previous)
    authored_end = trim_frames + authored_frames
    end_step = round(authored_end * minimax_h3.AUDIO_LATENT_FPS / minimax_h3.FPS)
    context_steps = round(audio_frames * minimax_h3.AUDIO_LATENT_FPS / minimax_h3.FPS)
    start_step = end_step - context_steps
    if start_step < 0 or end_step > audio.shape[-1]:
        raise ValueError("FL MiniMax H3 motion context could not slice the source audio latent.")
    audio_end_offset = (
        end_step - authored_end * minimax_h3.AUDIO_LATENT_FPS / minimax_h3.FPS
    ) / minimax_model.FRAME_RESCALE
    return {
        "kind": "audio",
        "ref_audio_t": context_steps,
        "audio_latent": audio[..., start_step:end_step].clone(),
        AUDIO_END_FRAME_MARKER: video_frames + audio_end_offset,
    }


def apply_previous_shot_context(conditioning, previous, source_shot, target_shot, vae):
    context = target_shot.get("motion_context")
    if not isinstance(context, dict):
        return conditioning

    video_frames = context.get("video_frames", 0)
    audio_frames = context.get("audio_frames", 0)
    if not video_frames and not audio_frames:
        return conditioning

    source_context = source_shot.get("motion_context") or {}
    trim_frames = source_context.get("trim_frames", 0)
    keyframes = []
    if video_frames:
        if vae is None:
            raise ValueError(
                "FL MiniMax H3 Beat KSampler needs its optional vae input when visual motion "
                "context is enabled."
            )
        keyframes = _video_keyframes(
            previous,
            source_shot["authored_frames"],
            trim_frames,
            video_frames,
            vae,
        )
    audio_ref = None
    if audio_frames:
        audio_ref = _audio_reference(
            previous,
            source_shot["authored_frames"],
            trim_frames,
            audio_frames,
            video_frames,
        )

    resolved = []
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise TypeError("FL MiniMax H3 motion context received invalid conditioning.")
        metadata = entry[1].copy()
        if keyframes:
            metadata["minimax_keyframes"] = list(metadata.get("minimax_keyframes") or ()) + keyframes
            metadata["minimax_frame_count"] = target_shot["render_frames"]
        if audio_ref is not None:
            metadata["minimax_refs"] = list(metadata.get("minimax_refs") or ()) + [audio_ref]
        resolved.append([entry[0], metadata])
    return resolved
