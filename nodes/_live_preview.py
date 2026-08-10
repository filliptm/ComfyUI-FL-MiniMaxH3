import logging
import os
import threading
import uuid
from fractions import Fraction

import comfy.nested_tensor
import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl, Types
from server import PromptServer


EVENT_NAME = "fl_minimax_h3_live_preview"
PREVIEW_SUBFOLDER = "fl_minimax_h3_previews"
_MAX_PREVIEW_FILES = 48
_preview_files_lock = threading.Lock()


def send_preview_event(node_id, prompt_id, stage, status, **details):
    server = PromptServer.instance
    if server is None:
        return
    payload = {
        "node": node_id,
        "prompt_id": prompt_id,
        "stage": stage,
        "status": status,
        **details,
    }
    try:
        server.send_sync(EVENT_NAME, payload, server.client_id)
    except Exception:
        logging.debug("FL MiniMax H3 live preview event could not be sent.", exc_info=True)


def _video_latent(latent):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not isinstance(samples, comfy.nested_tensor.NestedTensor):
        raise TypeError("the completed render is not a nested H3 latent")
    tensors = samples.unbind()
    if len(tensors) != 2:
        raise ValueError("the completed render does not contain video and audio latents")
    video = tensors[0]
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
        raise ValueError("the completed render has an invalid H3 video latent")
    return video


def _visible_frames(latent, vae, metadata):
    images = vae.decode(_video_latent(latent))
    if images.ndim == 5:
        images = images.reshape(
            -1,
            images.shape[-3],
            images.shape[-2],
            images.shape[-1],
        )

    authored_frames = metadata.get("authored_frames")
    motion_context = metadata.get("motion_context") or {}
    trim_frames = motion_context.get("trim_frames", 0)
    if not isinstance(authored_frames, int) or authored_frames <= 0:
        raise ValueError("the completed render has an invalid authored frame count")
    if not isinstance(trim_frames, int) or trim_frames < 0:
        raise ValueError("the completed render has an invalid motion-context trim")
    if (
        images.ndim != 4
        or images.shape[-1] != 3
        or images.shape[0] < trim_frames + authored_frames
    ):
        raise ValueError("the video VAE decoded too few visible RGB frames")
    return images[trim_frames:trim_frames + authored_frames].clone()


def _preview_size(width, height, max_long_side):
    scale = min(1.0, max_long_side / max(width, height))
    target_width = max(2, round(width * scale / 2) * 2)
    target_height = max(2, round(height * scale / 2) * 2)
    return target_width, target_height


def _resize_frames(images, max_long_side):
    height = int(images.shape[-3])
    width = int(images.shape[-2])
    target_width, target_height = _preview_size(width, height, max_long_side)
    if (target_width, target_height) == (width, height):
        return images
    return comfy.utils.common_upscale(
        images.movedim(-1, 1),
        target_width,
        target_height,
        "bilinear",
        "disabled",
    ).movedim(1, -1)


def _preview_path(stage, index):
    if stage not in ("sample", "upscale"):
        raise ValueError("live preview received an invalid sampler stage")
    output_dir = os.path.join(folder_paths.get_temp_directory(), PREVIEW_SUBFOLDER)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{stage}_{index + 1:04}_{uuid.uuid4().hex[:12]}.mp4"
    return output_dir, filename, os.path.join(output_dir, filename)


def _trim_preview_files(output_dir):
    with _preview_files_lock:
        files = []
        for entry in os.scandir(output_dir):
            if entry.is_file() and entry.name.endswith(".mp4"):
                files.append((entry.stat().st_mtime_ns, entry.path))
        files.sort()
        for _, file_path in files[:-_MAX_PREVIEW_FILES]:
            try:
                os.remove(file_path)
            except OSError:
                logging.debug("FL MiniMax H3 could not remove an old live preview.", exc_info=True)


def create_preview(latent, vae, metadata, stage, index, max_long_side):
    images = _resize_frames(
        _visible_frames(latent, vae, metadata),
        max_long_side,
    )
    fps = metadata.get("fps")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
        raise ValueError("the completed render has an invalid frame rate")

    output_dir, filename, output_path = _preview_path(stage, index)
    video = InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=images,
            audio=None,
            frame_rate=Fraction(round(fps * 1000), 1000),
        ),
        bit_depth=8,
    )
    try:
        video.save_to(
            output_path,
            format=Types.VideoContainer.MP4,
            codec=Types.VideoCodec.H264,
            crf=30,
        )
    except Exception:
        if os.path.isfile(output_path):
            try:
                os.remove(output_path)
            except OSError:
                logging.debug("FL MiniMax H3 could not remove a failed live preview.", exc_info=True)
        raise
    _trim_preview_files(output_dir)

    frame_count = int(images.shape[0])
    return {
        "filename": filename,
        "subfolder": PREVIEW_SUBFOLDER,
        "type": "temp",
        "frame_count": frame_count,
        "frame_rate": float(fps),
        "duration": frame_count / fps,
        "width": int(images.shape[-2]),
        "height": int(images.shape[-3]),
    }


def publish_preview(
    latent,
    vae,
    metadata,
    node_id,
    prompt_id,
    stage,
    index,
    total,
    max_long_side,
):
    send_preview_event(
        node_id,
        prompt_id,
        stage,
        "previewing",
        index=index,
        total=total,
    )
    try:
        preview = create_preview(
            latent,
            vae,
            metadata,
            stage,
            index,
            max_long_side,
        )
    except Exception as error:
        message = str(error) or error.__class__.__name__
        logging.warning("FL MiniMax H3 live preview skipped: %s", message)
        send_preview_event(
            node_id,
            prompt_id,
            stage,
            "preview_error",
            index=index,
            total=total,
            error=message,
        )
        return None

    send_preview_event(
        node_id,
        prompt_id,
        stage,
        "chunk_ready",
        index=index,
        total=total,
        start_frame=metadata.get("start_frame"),
        end_frame=metadata.get("end_frame"),
        seed=metadata.get("seed"),
        preview=preview,
    )
    return preview
