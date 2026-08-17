import comfy.nested_tensor
import comfy.utils
import nodes
from comfy_api.latest import io

from ._latent_helpers import (
    H3_PIXEL_MULTIPLE,
    H3_SPATIAL_DOWNSCALE,
    h3_tensors,
    target_canvas,
)


_OWNER = "FL MiniMax H3 Latent Upscale"
_UPSCALE_METHODS = ["bislerp", "bicubic", "bilinear", "nearest-exact"]


def upscale_h3_latent(latent, target_long_side, upscale_method):
    metadata = latent.get("fl_h3_shot") if isinstance(latent, dict) else None
    if isinstance(metadata, dict) and isinstance(metadata.get("reshot"), dict):
        raise ValueError(
            "FL MiniMax H3 Latent Upscale does not support temporal reshots because their "
            "source-preservation mask is tied to the planner canvas."
        )
    video, audio = h3_tensors(latent, _OWNER)
    source_width = video.shape[-1] * H3_SPATIAL_DOWNSCALE
    source_height = video.shape[-2] * H3_SPATIAL_DOWNSCALE
    target_width, target_height = target_canvas(
        source_width,
        source_height,
        target_long_side,
        _OWNER,
    )

    resized_video = video
    if (target_width, target_height) != (source_width, source_height):
        resized_video = comfy.utils.common_upscale(
            video,
            target_width // H3_SPATIAL_DOWNSCALE,
            target_height // H3_SPATIAL_DOWNSCALE,
            upscale_method,
            "disabled",
        )

    output = latent.copy()
    output["samples"] = comfy.nested_tensor.NestedTensor((resized_video, audio))
    return output


class FL_MiniMaxH3LatentUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3LatentUpscale",
            display_name="FL MiniMax H3 Latent Upscale",
            category="FL/MiniMax H3/Latent",
            description=(
                "Upscales only the spatial dimensions of a native MiniMax H3 video latent "
                "while preserving its temporal length, audio latent, and shot metadata."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip="Native nested MiniMax H3 video/audio latent.",
                ),
                io.Int.Input(
                    "target_long_side",
                    default=1024,
                    min=H3_PIXEL_MULTIPLE,
                    max=nodes.MAX_RESOLUTION,
                    step=H3_PIXEL_MULTIPLE,
                    tooltip=(
                        "Target pixel size for the longest side. Aspect ratio is preserved and "
                        "both output dimensions stay aligned to 32 pixels."
                    ),
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=_UPSCALE_METHODS,
                    default="bislerp",
                    tooltip="Interpolation applied directly to each spatial H3 latent slice.",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip=(
                        "Spatially upscaled H3 latent with unchanged video duration, audio, "
                        "and metadata."
                    ),
                )
            ],
        )

    @classmethod
    def execute(cls, latent, target_long_side, upscale_method):
        return io.NodeOutput(upscale_h3_latent(latent, target_long_side, upscale_method))
