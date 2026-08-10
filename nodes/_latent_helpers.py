import torch

import comfy.nested_tensor


H3_PIXEL_MULTIPLE = 32
H3_SPATIAL_DOWNSCALE = 16


def h3_tensors(latent, owner):
    if not isinstance(latent, dict):
        raise TypeError(f"{owner} expects a latent dictionary.")
    samples = latent.get("samples")
    if not isinstance(samples, comfy.nested_tensor.NestedTensor):
        raise TypeError(f"{owner} expects a nested H3 video/audio latent.")
    tensors = samples.unbind()
    if len(tensors) != 2:
        raise ValueError(f"{owner} expects exactly one video and one audio latent.")
    video, audio = tensors
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
        raise ValueError(f"{owner} expects video latent shape [1, 24, T, H, W].")
    if audio.ndim != 4 or audio.shape[0] != 1 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError(f"{owner} expects audio latent shape [1, 32, 2, T40].")
    return video, audio


def target_canvas(source_width, source_height, target_long_side, owner):
    if target_long_side % H3_PIXEL_MULTIPLE:
        raise ValueError(
            f"{owner} target long side must be divisible by {H3_PIXEL_MULTIPLE}."
        )
    if target_long_side < max(source_width, source_height):
        raise ValueError(
            f"{owner} target long side is smaller than the {source_width}x{source_height} source."
        )

    scale = target_long_side / max(source_width, source_height)
    if source_width >= source_height:
        target_width = target_long_side
        target_height = round(source_height * scale / H3_PIXEL_MULTIPLE) * H3_PIXEL_MULTIPLE
    else:
        target_width = round(source_width * scale / H3_PIXEL_MULTIPLE) * H3_PIXEL_MULTIPLE
        target_height = target_long_side
    return (
        max(H3_PIXEL_MULTIPLE, target_width),
        max(H3_PIXEL_MULTIPLE, target_height),
    )


def primary_only_noise_mask(samples, protected_video_steps=0):
    if not isinstance(samples, comfy.nested_tensor.NestedTensor):
        return None

    tensors = samples.unbind()
    video = torch.ones_like(tensors[0])
    protected = int(protected_video_steps)
    if protected:
        video[:, :, :protected] = 0
    return comfy.nested_tensor.NestedTensor(
        (video, *(torch.zeros_like(tensor) for tensor in tensors[1:]))
    )
