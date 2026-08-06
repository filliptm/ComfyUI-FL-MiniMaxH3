import torch

import comfy.nested_tensor


def primary_only_noise_mask(samples):
    if not isinstance(samples, comfy.nested_tensor.NestedTensor):
        return None

    tensors = samples.unbind()
    return comfy.nested_tensor.NestedTensor(
        (torch.ones_like(tensors[0]), *(torch.zeros_like(tensor) for tensor in tensors[1:]))
    )
