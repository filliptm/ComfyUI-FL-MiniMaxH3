import logging

import torch

import comfy.nested_tensor
import comfy.utils
from comfy_api.latest import io


def _shot_metadata(latent, position):
    if not isinstance(latent, dict):
        raise TypeError(f"FL MiniMax H3 Shot Assembler input {position} is not a latent.")
    metadata = latent.get("fl_h3_shot")
    if not isinstance(metadata, dict) or metadata.get("version") != 1:
        raise ValueError(
            f"FL MiniMax H3 Shot Assembler latent {position} is missing H3 shot metadata. "
            "Use latent nodes that preserve the latent dictionary."
        )
    return metadata


def _video_latent(latent, position):
    samples = latent.get("samples")
    if not isinstance(samples, comfy.nested_tensor.NestedTensor):
        raise TypeError(
            f"FL MiniMax H3 Shot Assembler render {position} is not a nested H3 latent."
        )
    tensors = samples.unbind()
    if len(tensors) != 2:
        raise ValueError(
            f"FL MiniMax H3 Shot Assembler render {position} must contain video and audio latents."
        )
    video = tensors[0]
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
        raise ValueError(
            f"FL MiniMax H3 Shot Assembler render {position} has an invalid video latent."
        )
    return video


class FL_MiniMaxH3ShotAssembler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3ShotAssembler",
            display_name="FL MiniMax H3 Shot Assembler",
            category="FL/MiniMax H3/Output",
            description=(
                "Decodes planned MiniMax H3 renders separately, removes hidden motion context and "
                "H3 padding, and assembles the authored frames in pixel space."
            ),
            inputs=[
                io.Latent.Input(
                    "latents",
                    tooltip=(
                        "Editable H3 latent list from FL MiniMax H3 Beat KSampler, optionally "
                        "processed by metadata-preserving latent nodes."
                    ),
                ),
                io.Vae.Input(
                    "vae",
                    tooltip="MiniMax H3 video VAE used to decode every planned render.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="Final frame batch with hard cuts at the authored beat boundaries.",
                )
            ],
            is_input_list=True,
        )

    @classmethod
    def execute(cls, latents, vae):
        if not isinstance(latents, list) or not latents:
            raise ValueError("FL MiniMax H3 Shot Assembler requires at least one H3 latent.")
        if not isinstance(vae, list) or len(vae) != 1:
            raise ValueError("FL MiniMax H3 Shot Assembler requires exactly one video VAE.")
        vae = vae[0]
        metadata = [_shot_metadata(latent, position) for position, latent in enumerate(latents, 1)]
        total_frames = metadata[0].get("total_frames")
        if not isinstance(total_frames, int) or total_frames <= 0:
            raise ValueError("FL MiniMax H3 Shot Assembler received an invalid total frame count.")

        output = None
        cursor = 0
        progress = comfy.utils.ProgressBar(len(latents))
        for position, (latent, shot) in enumerate(zip(latents, metadata), 1):
            authored_frames = shot.get("authored_frames")
            if not isinstance(authored_frames, int) or authored_frames <= 0:
                raise ValueError(
                    f"FL MiniMax H3 Shot Assembler render {position} has an invalid authored length."
                )
            if (
                shot.get("index") != position - 1
                or shot.get("start_frame") != cursor
                or shot.get("end_frame") != cursor + authored_frames
                or shot.get("total_frames") != total_frames
            ):
                raise ValueError(
                    f"FL MiniMax H3 Shot Assembler latent {position} is out of order "
                    "or has inconsistent shot metadata."
                )
            images = vae.decode(_video_latent(latent, position))
            if images.ndim == 5:
                images = images.reshape(
                    -1,
                    images.shape[-3],
                    images.shape[-2],
                    images.shape[-1],
                )
            motion_context = shot.get("motion_context") or {}
            trim_frames = motion_context.get("trim_frames", 0)
            if not isinstance(trim_frames, int) or trim_frames < 0:
                raise ValueError(
                    f"FL MiniMax H3 Shot Assembler render {position} has an invalid context trim."
                )
            if images.ndim != 4 or images.shape[0] < trim_frames + authored_frames:
                raise ValueError(
                    f"FL MiniMax H3 Shot Assembler decoded too few frames for render {position}."
                )
            images = images[trim_frames:trim_frames + authored_frames]
            if output is None:
                output = torch.empty(
                    (total_frames, *images.shape[1:]),
                    dtype=images.dtype,
                    device=images.device,
                )
            elif images.shape[1:] != output.shape[1:]:
                raise ValueError(
                    f"FL MiniMax H3 Shot Assembler render {position} decoded at a different resolution."
                )
            if cursor + authored_frames > total_frames:
                raise ValueError("FL MiniMax H3 Shot Assembler decoded more than the planned duration.")
            output[cursor:cursor + authored_frames].copy_(images)
            cursor += authored_frames
            progress.update(1)

        if cursor != total_frames:
            raise ValueError(
                f"FL MiniMax H3 Shot Assembler produced {cursor} frames; expected {total_frames}."
            )
        logging.info(
            "FL MiniMax H3 shot assembler: decoded %d planned renders into %d frames.",
            len(latents),
            total_frames,
        )
        return io.NodeOutput(output)
