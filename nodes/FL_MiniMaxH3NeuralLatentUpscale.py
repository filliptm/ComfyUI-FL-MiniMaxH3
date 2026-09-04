import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.model_patcher
import comfy.nested_tensor
import comfy.ops
import comfy.utils
import folder_paths
import nodes
from comfy.ldm.minimax.vae import LATENTS_MEAN, LATENTS_STD
from comfy.ldm.modules.attention import optimized_attention
from comfy_api.latest import io

from ._latent_helpers import H3_PIXEL_MULTIPLE, H3_SPATIAL_DOWNSCALE, h3_tensors


_MODEL_FOLDER = "latent_upscale_models"
_OWNER_2D = "FL MiniMax H3 Neural Latent Upscale 2D"
_OWNER_3D = "FL MiniMax H3 Neural Latent Upscale 3D"
_PRECISIONS = ["auto", "fp16", "bf16", "fp32"]
_RESIZE_SCALE = "scale by multiplier"
_RESIZE_DIMENSIONS = "target dimensions"
_RESIZE_MEGAPIXELS = "megapixels"
ops = comfy.ops.disable_weight_init


def _normalization(channels):
    return ops.GroupNorm(32, channels)


class _Attention2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = _normalization(channels)
        self.q = ops.Conv2d(channels, channels, 1)
        self.k = ops.Conv2d(channels, channels, 1)
        self.v = ops.Conv2d(channels, channels, 1)
        self.proj_out = ops.Conv2d(channels, channels, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        h = self.norm(x)
        query = self.q(h).flatten(2).transpose(1, 2).unsqueeze(1)
        key = self.k(h).flatten(2).transpose(1, 2).unsqueeze(1)
        value = self.v(h).flatten(2).transpose(1, 2).unsqueeze(1)
        h = optimized_attention(query, key, value, 1, skip_reshape=True)
        h = h.transpose(1, 2).reshape(batch, channels, height, width)
        return x + self.proj_out(h)


class _Attention3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = _normalization(channels)
        self.q = ops.Conv3d(channels, channels, 1)
        self.k = ops.Conv3d(channels, channels, 1)
        self.v = ops.Conv3d(channels, channels, 1)
        self.proj_out = ops.Conv3d(channels, channels, 1)

    def forward(self, x):
        batch, channels, frames, height, width = x.shape
        h = self.norm(x)
        query = self.q(h).flatten(2).transpose(1, 2).unsqueeze(1)
        key = self.k(h).flatten(2).transpose(1, 2).unsqueeze(1)
        value = self.v(h).flatten(2).transpose(1, 2).unsqueeze(1)
        h = optimized_attention(query, key, value, 1, skip_reshape=True)
        h = h.transpose(1, 2).reshape(batch, channels, frames, height, width)
        return x + self.proj_out(h)


class _Residual2D(nn.Module):
    def __init__(self, channels, embedding_channels):
        super().__init__()
        self.in_layers = nn.Sequential(
            _normalization(channels),
            nn.SiLU(),
            ops.Conv2d(channels, channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            ops.Linear(embedding_channels, channels * 2),
        )
        self.out_norm = _normalization(channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Identity(),
            ops.Conv2d(channels, channels, 3, padding=1),
        )
        self.skip = nn.Identity()

    def forward(self, x, embedding):
        h = self.in_layers(x)
        scale, shift = self.emb_layers(embedding).to(h).chunk(2, dim=1)
        h = self.out_norm(h).mul(1 + scale[:, :, None, None]).add_(shift[:, :, None, None])
        return self.skip(x) + self.out_layers(h)


class _Residual3D(nn.Module):
    def __init__(self, channels, embedding_channels):
        super().__init__()
        self.in_layers = nn.Sequential(
            _normalization(channels),
            nn.SiLU(),
            ops.Conv3d(channels, channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            ops.Linear(embedding_channels, channels * 2),
        )
        self.out_norm = _normalization(channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Identity(),
            ops.Conv3d(channels, channels, 3, padding=1),
        )
        self.skip = nn.Identity()

    def forward(self, x, embedding):
        h = self.in_layers(x)
        scale, shift = self.emb_layers(embedding).to(h).chunk(2, dim=1)
        h = self.out_norm(h).mul(1 + scale[:, :, None, None, None]).add_(shift[:, :, None, None, None])
        return self.skip(x) + self.out_layers(h)


class _TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.norm = _normalization(channels)
        self.dwconv = ops.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0),
            groups=channels,
        )
        self.pwconv = ops.Conv3d(channels, channels, 1)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


def _make_blocks(kinds, channels, embedding_channels, temporal_kernels, dimensions):
    blocks = []
    temporal_index = 0
    for kind in kinds:
        if kind == "residual":
            block = _Residual2D(channels, embedding_channels) if dimensions == 2 else _Residual3D(channels, embedding_channels)
        elif kind == "attention":
            block = _Attention2D(channels) if dimensions == 2 else _Attention3D(channels)
        elif kind == "temporal" and dimensions == 3:
            block = _TemporalConv(channels, temporal_kernels[temporal_index])
            temporal_index += 1
        else:
            raise ValueError(f"Unsupported latent upscaler block type: {kind}")
        blocks.append(block)
    return nn.ModuleList(blocks)


class _LatentResizer2D(nn.Module):
    def __init__(self, channels, embedding_channels, in_kinds, out_kinds):
        super().__init__()
        self.conv_in = ops.Conv2d(24, channels, 3, padding=1)
        self.embed = nn.Sequential(
            ops.Linear(1, embedding_channels),
            nn.SiLU(),
            ops.Linear(embedding_channels, embedding_channels),
        )
        self.in_blocks = _make_blocks(in_kinds, channels, embedding_channels, [], 2)
        self.out_blocks = _make_blocks(out_kinds, channels, embedding_channels, [], 2)
        self.norm_out = _normalization(channels)
        self.conv_out = ops.Conv2d(channels, 24, 3, padding=1)


class _VideoLatentResizer2D(nn.Module):
    def __init__(self, channels, embedding_channels, in_kinds, out_kinds, temporal_kernels):
        super().__init__()
        self.resizer = _LatentResizer2D(channels, embedding_channels, in_kinds, out_kinds)
        self.temporal_blocks = nn.ModuleList([_TemporalConv(channels, kernel) for kernel in temporal_kernels])
        self.temporal_every = 2

    @staticmethod
    def _to_video(x, batch, frames):
        _, channels, height, width = x.shape
        return x.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)

    @staticmethod
    def _to_frames(x):
        batch, channels, frames, height, width = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)

    def forward(self, x, scale, target_size):
        batch, _, frames, _, _ = x.shape
        embedding = self.resizer.embed(x.new_full((1, 1), scale - 1)).expand(batch * frames, -1)
        out = self.resizer.conv_in(self._to_frames(x))

        for index, block in enumerate(self.resizer.in_blocks):
            out = block(out, embedding) if isinstance(block, _Residual2D) else block(out)
            if self.temporal_blocks and index % self.temporal_every == 0:
                out = self._to_frames(self.temporal_blocks[0](self._to_video(out, batch, frames)))

        out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        for index, block in enumerate(self.resizer.out_blocks):
            out = block(out, embedding) if isinstance(block, _Residual2D) else block(out)
            if self.temporal_blocks and index % self.temporal_every == 0:
                out = self._to_frames(self.temporal_blocks[1](self._to_video(out, batch, frames)))

        out = self.resizer.conv_out(F.silu(self.resizer.norm_out(out)))
        return self._to_video(out, batch, frames)


class _LatentResizer3D(nn.Module):
    def __init__(self, channels, embedding_channels, in_kinds, out_kinds, in_temporal_kernels, out_temporal_kernels):
        super().__init__()
        self.conv_in = ops.Conv3d(24, channels, 3, padding=1)
        self.embed = nn.Sequential(
            ops.Linear(1, embedding_channels),
            nn.SiLU(),
            ops.Linear(embedding_channels, embedding_channels),
        )
        self.in_blocks = _make_blocks(in_kinds, channels, embedding_channels, in_temporal_kernels, 3)
        self.out_blocks = _make_blocks(out_kinds, channels, embedding_channels, out_temporal_kernels, 3)
        self.norm_out = _normalization(channels)
        self.conv_out = ops.Conv3d(channels, 24, 3, padding=1)
        self.temporal_overlap = max(in_temporal_kernels + out_temporal_kernels, default=0)

    def _forward_segment(self, x, scale, target_size):
        embedding = self.embed(x.new_full((1, 1), scale - 1)).expand(x.shape[0], -1)
        x = self.conv_in(x)
        for block in self.in_blocks:
            x = block(x, embedding) if isinstance(block, _Residual3D) else block(x)
        x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            x = block(x, embedding) if isinstance(block, _Residual3D) else block(x)
        return self.conv_out(F.silu(self.norm_out(x)))

    def forward(self, x, scale, target_size, chunk_size):
        frames = x.shape[2]
        if chunk_size <= 0 or frames <= chunk_size:
            return self._forward_segment(x, scale, target_size)

        overlap = self.temporal_overlap
        padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode="replicate") if overlap else x
        output = x.new_zeros((x.shape[0], x.shape[1], frames, target_size[1], target_size[2]))
        weights = x.new_zeros((1, 1, frames, 1, 1))

        for segment_start in range(0, frames, chunk_size):
            segment_end = min(frames, segment_start + chunk_size)
            output_start = max(0, segment_start - overlap)
            output_end = min(frames, segment_end + overlap)
            input_start = max(0, output_start - overlap)
            input_end = min(frames + 2 * overlap, output_end + overlap)
            segment = padded[:, :, input_start:input_end].contiguous()
            segment_output = self._forward_segment(
                segment,
                scale,
                (input_end - input_start, target_size[1], target_size[2]),
            )
            valid_start = output_start + overlap - input_start
            valid_length = output_end - output_start
            valid = segment_output[:, :, valid_start:valid_start + valid_length]

            weight = x.new_ones((valid_length,))
            if segment_start > output_start:
                blend = segment_start - output_start
                weight[:blend] = torch.arange(1, blend + 1, device=x.device, dtype=x.dtype) / (blend + 1)
            if output_end > segment_end:
                blend = output_end - segment_end
                weight[-blend:] = torch.arange(blend, 0, -1, device=x.device, dtype=x.dtype) / (blend + 1)
            weight = weight.view(1, 1, valid_length, 1, 1)
            output[:, :, output_start:output_end].add_(valid * weight)
            weights[:, :, output_start:output_end].add_(weight)

        return output.div_(weights.clamp_min_(1e-8))


def _block_layout(state_dict, prefix):
    indices = sorted({
        int(match.group(1))
        for key in state_dict
        if (match := re.match(rf"{re.escape(prefix)}\.(\d+)\.", key))
    })
    if indices != list(range(len(indices))):
        raise ValueError(f"MiniMax H3 latent upscaler has a non-contiguous {prefix} layout.")

    kinds = []
    temporal_kernels = []
    for index in indices:
        block_prefix = f"{prefix}.{index}."
        keys = [key[len(block_prefix):] for key in state_dict if key.startswith(block_prefix)]
        if any(key.startswith("in_layers.") for key in keys):
            kinds.append("residual")
        elif "dwconv.weight" in keys:
            kinds.append("temporal")
            temporal_kernels.append(int(state_dict[f"{block_prefix}dwconv.weight"].shape[2]))
        elif "q.weight" in keys:
            kinds.append("attention")
        else:
            raise ValueError(f"MiniMax H3 latent upscaler has an unsupported block at {block_prefix[:-1]}.")
    return kinds, temporal_kernels


def _load_state_dict(model_name):
    path = folder_paths.get_full_path_or_raise(_MODEL_FOLDER, model_name)
    state_dict = comfy.utils.load_torch_file(path, safe_load=True)
    if not isinstance(state_dict, dict):
        raise ValueError("MiniMax H3 latent upscaler checkpoint must contain a state dictionary.")
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    if any(key.startswith("upscaler.") for key in state_dict):
        state_dict = {
            key.removeprefix("upscaler."): value
            for key, value in state_dict.items()
            if key.startswith("upscaler.")
        }
    if not state_dict or any(not isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValueError("MiniMax H3 latent upscaler checkpoint contains unsupported non-tensor values.")
    return state_dict


def _model_dtype(precision, device):
    if precision == "auto":
        return comfy.model_management.vae_dtype(device, allowed_dtypes=[torch.float16, torch.bfloat16, torch.float32])
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    if not comfy.model_management.supports_dtype(device, dtype):
        raise ValueError(f"{precision} is not supported on the selected ComfyUI device.")
    return dtype


def _build_model(state_dict):
    if "resizer.conv_in.weight" in state_dict:
        if "resizer.embed.0.weight" not in state_dict:
            raise ValueError("MiniMax H3 2D latent upscaler checkpoint is missing resizer.embed.0.weight.")
        conv = state_dict["resizer.conv_in.weight"]
        if conv.ndim != 4 or conv.shape[1] != 24:
            raise ValueError("The selected checkpoint is not a 24-channel MiniMax H3 2D latent upscaler.")
        channels = int(conv.shape[0])
        if channels % 32:
            raise ValueError("MiniMax H3 latent upscaler channels must be divisible by 32.")
        embedding_channels = int(state_dict["resizer.embed.0.weight"].shape[0])
        in_kinds, _ = _block_layout(state_dict, "resizer.in_blocks")
        out_kinds, _ = _block_layout(state_dict, "resizer.out_blocks")
        temporal_indices = sorted({
            int(match.group(1))
            for key in state_dict
            if (match := re.match(r"temporal_blocks\.(\d+)\.dwconv\.weight", key))
        })
        temporal_kernels = [int(state_dict[f"temporal_blocks.{index}.dwconv.weight"].shape[2]) for index in temporal_indices]
        if temporal_indices and temporal_indices != [0, 1]:
            raise ValueError("MiniMax H3 2D temporal checkpoint must contain temporal blocks 0 and 1.")
        model = _VideoLatentResizer2D(channels, embedding_channels, in_kinds, out_kinds, temporal_kernels)
        return "2d", model

    if "conv_in.weight" in state_dict:
        if "embed.0.weight" not in state_dict:
            raise ValueError("MiniMax H3 3D latent upscaler checkpoint is missing embed.0.weight.")
        conv = state_dict["conv_in.weight"]
        if conv.ndim != 5 or conv.shape[1] != 24:
            raise ValueError("The selected checkpoint is not a 24-channel MiniMax H3 3D latent upscaler.")
        channels = int(conv.shape[0])
        if channels % 32:
            raise ValueError("MiniMax H3 latent upscaler channels must be divisible by 32.")
        embedding_channels = int(state_dict["embed.0.weight"].shape[0])
        in_kinds, in_temporal_kernels = _block_layout(state_dict, "in_blocks")
        out_kinds, out_temporal_kernels = _block_layout(state_dict, "out_blocks")
        model = _LatentResizer3D(
            channels,
            embedding_channels,
            in_kinds,
            out_kinds,
            in_temporal_kernels,
            out_temporal_kernels,
        )
        return "3d", model

    raise ValueError("The selected checkpoint is not a supported MiniMax H3 2D or 3D latent upscaler.")


class _H3LatentUpscaleModel:
    def __init__(self, architecture, model, dtype):
        self.architecture = architecture
        self.model = model.eval()
        self.dtype = dtype
        self.load_device = comfy.model_management.get_torch_device()
        self.offload_device = comfy.model_management.vae_offload_device()
        self.model.manual_cast_dtype = dtype
        self.patcher = comfy.model_patcher.CoreModelPatcher(
            self.model,
            load_device=self.load_device,
            offload_device=self.offload_device,
        )

    def upscale(self, video, scale, target_size, chunk_size=0):
        comfy.model_management.load_model_gpu(self.patcher)
        source_device = video.device
        source_dtype = video.dtype
        normalized = video.to(device=self.load_device, dtype=self.dtype)
        mean = normalized.new_tensor(LATENTS_MEAN).view(1, 24, 1, 1, 1)
        std = normalized.new_tensor(LATENTS_STD).view(1, 24, 1, 1, 1)
        normalized = normalized.sub(mean).div_(std)
        if self.architecture == "2d":
            output = self.model(normalized, scale, target_size[-2:])
        else:
            output = self.model(normalized, scale, target_size, chunk_size)
        return output.mul(std).add_(mean).to(device=source_device, dtype=source_dtype)


def load_h3_latent_upscale_model(model_name, precision):
    device = comfy.model_management.get_torch_device()
    dtype = _model_dtype(precision, device)
    state_dict = _load_state_dict(model_name)
    with torch.device("meta"):
        architecture, model = _build_model(state_dict)
    state_dict = {
        key: value.to(dtype=dtype) if value.is_floating_point() else value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True, assign=True)
    comfy.model_management.archive_model_dtypes(model)
    return _H3LatentUpscaleModel(architecture, model, dtype)


def _validate_latent(latent, owner):
    metadata = latent.get("fl_h3_shot") if isinstance(latent, dict) else None
    if isinstance(metadata, dict) and isinstance(metadata.get("reshot"), dict):
        raise ValueError(f"{owner} does not support temporal reshots because their source mask is tied to the planner canvas.")
    return h3_tensors(latent, owner)


def _resize_noise_mask(latent, target_size):
    noise_mask = latent.get("noise_mask")
    if not isinstance(noise_mask, comfy.nested_tensor.NestedTensor):
        return noise_mask
    tensors = noise_mask.unbind()
    if not tensors or tensors[0].ndim != 5:
        return noise_mask
    video_mask = F.interpolate(tensors[0], size=target_size, mode="nearest")
    return comfy.nested_tensor.NestedTensor((video_mask, *tensors[1:]))


def _upscaled_latent(latent, video, audio, target_size):
    output = latent.copy()
    output["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
    resized_mask = _resize_noise_mask(latent, target_size)
    if resized_mask is not None:
        output["noise_mask"] = resized_mask
    return output


def _scale_target(video, scale):
    return (
        int(video.shape[2]),
        max(1, round(video.shape[3] * scale)),
        max(1, round(video.shape[4] * scale)),
    )


def _target_from_mode(video, mode, align):
    selected = mode["mode"]
    source_height = int(video.shape[3])
    source_width = int(video.shape[4])
    if selected == _RESIZE_SCALE:
        requested_width = source_width * H3_SPATIAL_DOWNSCALE * mode["scale"]
        requested_height = source_height * H3_SPATIAL_DOWNSCALE * mode["scale"]
    elif selected == _RESIZE_DIMENSIONS:
        requested_width = mode["width"]
        requested_height = mode["height"]
    elif selected == _RESIZE_MEGAPIXELS:
        pixels = mode["megapixels"] * 1024 * 1024
        requested_height = math.sqrt(pixels / (source_width / source_height))
        requested_width = requested_height * source_width / source_height
    else:
        raise ValueError(f"Unsupported MiniMax H3 latent upscale mode: {selected}")

    if requested_width <= 0 or requested_height <= 0:
        raise ValueError("FL MiniMax H3 neural latent upscale dimensions must be positive.")
    alignment = math.lcm(max(1, int(align)), H3_SPATIAL_DOWNSCALE)
    pixel_width = max(H3_SPATIAL_DOWNSCALE, round(requested_width / alignment) * alignment)
    pixel_height = max(H3_SPATIAL_DOWNSCALE, round(requested_height / alignment) * alignment)
    target_width = pixel_width // H3_SPATIAL_DOWNSCALE
    target_height = pixel_height // H3_SPATIAL_DOWNSCALE
    width_scale = target_width / source_width
    height_scale = target_height / source_height
    effective_scale = (width_scale + height_scale) / 2
    if width_scale < 1 or height_scale < 1:
        raise ValueError("FL MiniMax H3 neural latent upscalers only support spatial upscaling.")
    if effective_scale > 4:
        raise ValueError("FL MiniMax H3 neural latent upscalers support an effective scale up to 4x.")
    return (int(video.shape[2]), target_height, target_width), effective_scale


class FL_MiniMaxH3LatentUpscaleModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3LatentUpscaleModelLoader",
            display_name="FL MiniMax H3 Load Latent Upscaler",
            category="FL/MiniMax H3/Loaders",
            description="Loads a checkpoint-compatible MiniMax H3 2D or 3D neural latent upscaler with ComfyUI model offloading.",
            inputs=[
                io.Combo.Input("model_name", options=folder_paths.get_filename_list(_MODEL_FOLDER)),
                io.Combo.Input("precision", options=_PRECISIONS, default="auto"),
            ],
            outputs=[io.LatentUpscaleModel.Output(display_name="upscale_model")],
        )

    @classmethod
    def execute(cls, model_name, precision):
        return io.NodeOutput(load_h3_latent_upscale_model(model_name, precision))


class FL_MiniMaxH3NeuralLatentUpscale2D(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3NeuralLatentUpscale2D",
            display_name="FL MiniMax H3 Neural Latent Upscale 2D",
            category="FL/MiniMax H3/Latent",
            description="Fast learned spatial upscale for a native H3 AV latent. Video time and the original audio latent are preserved.",
            inputs=[
                io.LatentUpscaleModel.Input("upscale_model"),
                io.Latent.Input("latent"),
                io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.05),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, upscale_model, latent, scale):
        if not isinstance(upscale_model, _H3LatentUpscaleModel) or upscale_model.architecture != "2d":
            raise ValueError(f"{_OWNER_2D} requires a 2D MiniMax H3 latent upscaler checkpoint.")
        if not 1 <= scale <= 4:
            raise ValueError(f"{_OWNER_2D} scale must be between 1x and 4x.")
        video, audio = _validate_latent(latent, _OWNER_2D)
        target_size = _scale_target(video, scale)
        if target_size == tuple(video.shape[-3:]):
            return io.NodeOutput(latent.copy())
        output_video = upscale_model.upscale(video, scale, target_size)
        return io.NodeOutput(_upscaled_latent(latent, output_video, audio, target_size))


class FL_MiniMaxH3NeuralLatentUpscale3D(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3NeuralLatentUpscale3D",
            display_name="FL MiniMax H3 Neural Latent Upscale 3D",
            category="FL/MiniMax H3/Latent",
            description="Temporally coherent learned H3 latent upscale with optional time chunking. The audio latent is passed through unchanged.",
            inputs=[
                io.LatentUpscaleModel.Input("upscale_model"),
                io.Latent.Input("latent"),
                io.DynamicCombo.Input(
                    "mode",
                    options=[
                        io.DynamicCombo.Option(_RESIZE_SCALE, [
                            io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.05),
                        ]),
                        io.DynamicCombo.Option(_RESIZE_DIMENSIONS, [
                            io.Int.Input("width", default=1280, min=H3_PIXEL_MULTIPLE, max=nodes.MAX_RESOLUTION, step=H3_PIXEL_MULTIPLE),
                            io.Int.Input("height", default=704, min=H3_PIXEL_MULTIPLE, max=nodes.MAX_RESOLUTION, step=H3_PIXEL_MULTIPLE),
                        ]),
                        io.DynamicCombo.Option(_RESIZE_MEGAPIXELS, [
                            io.Float.Input("megapixels", default=1.0, min=0.1, max=16.0, step=0.1),
                        ]),
                    ],
                ),
                io.Int.Input("align", default=H3_PIXEL_MULTIPLE, min=1, max=512, step=1, advanced=True),
                io.Int.Input(
                    "temporal_chunk_size",
                    default=32,
                    min=0,
                    max=512,
                    step=1,
                    advanced=True,
                    tooltip="Latent time steps per chunk. Set to 0 for full temporal context.",
                ),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, upscale_model, latent, mode, align, temporal_chunk_size):
        if not isinstance(upscale_model, _H3LatentUpscaleModel) or upscale_model.architecture != "3d":
            raise ValueError(f"{_OWNER_3D} requires a 3D MiniMax H3 latent upscaler checkpoint.")
        video, audio = _validate_latent(latent, _OWNER_3D)
        target_size, scale = _target_from_mode(video, mode, align)
        if target_size == tuple(video.shape[-3:]):
            return io.NodeOutput(latent.copy())
        output_video = upscale_model.upscale(video, scale, target_size, temporal_chunk_size)
        return io.NodeOutput(_upscaled_latent(latent, output_video, audio, target_size))
