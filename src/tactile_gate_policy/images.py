"""Identical pixel conversion and resizing for dataset and live observations."""

import numpy as np
import torch
import torch.nn.functional as F


def prepare_rgb(image, image_size):
    """Return CPU CHW float RGB in [0,1] without mutating the input."""
    value = image.detach().cpu().clone() if torch.is_tensor(image) else torch.from_numpy(np.asarray(image).copy())
    if value.ndim != 3:
        raise ValueError("RGB input must be CHW or HWC")
    if value.shape[0] != 3 and value.shape[-1] == 3:
        value = value.permute(2, 0, 1)
    if value.shape[0] != 3:
        raise ValueError("RGB input requires three channels")
    value = value.float() / 255.0 if value.dtype == torch.uint8 else value.float()
    if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError("RGB must be uint8 or float in [0,1]")
    if tuple(value.shape[-2:]) != tuple(image_size):
        value = F.interpolate(value.unsqueeze(0), size=image_size, mode="bilinear", align_corners=False, antialias=True).squeeze(0)
    return value.contiguous()
