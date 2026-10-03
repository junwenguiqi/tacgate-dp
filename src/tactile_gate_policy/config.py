"""Portable configuration for the gated multimodal diffusion policy."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional, Tuple

import yaml


@dataclass(frozen=True)
class PolicyConfig:
    obs_horizon: int = 4
    num_modalities: int = 5
    token_dim: int = 256
    action_horizon: int = 8
    action_dim: int = 8
    n_action_steps: int = 4
    control_dt: float = 0.3
    condition_dim: int = 256
    num_diffusion_steps: int = 100
    num_inference_steps: int = 100
    router_layers: int = 2
    router_heads: int = 4
    condition_layers: int = 2
    condition_heads: int = 4
    dropout: float = 0.1
    dp_unet_diffusion_step_embed_dim: int = 128
    dp_unet_down_dims: Tuple[int, ...] = (64, 128)
    dp_unet_kernel_size: int = 3
    dp_unet_n_groups: int = 8
    dp_unet_cond_predict_scale: bool = True
    image_size: Tuple[int, int] = (96, 96)
    crop_size: Optional[Tuple[int, int]] = (84, 84)
    vision_weights: Optional[str] = None
    vision_group_norm: bool = True
    share_vision_backbone: bool = False
    share_tactile_backbone: bool = False
    imagenet_norm: bool = True
    proprio_dim: int = 22
    tactile_dim: int = 231
    tactile_channels: int = 3
    tactile_hidden_dims: Tuple[int, ...] = (32, 64, 128)

    def __post_init__(self):
        positive = ("obs_horizon", "token_dim", "action_horizon", "action_dim", "n_action_steps", "condition_dim", "num_diffusion_steps", "num_inference_steps", "router_layers", "router_heads", "condition_layers", "condition_heads", "proprio_dim", "tactile_dim", "tactile_channels", "dp_unet_n_groups")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_modalities != 5:
            raise ValueError("The policy requires exactly five modalities")
        if self.action_dim != 8:
            raise ValueError("FR3 actions require xyz + quaternion(x,y,z,w) + gripper (8D)")
        if self.n_action_steps > self.action_horizon:
            raise ValueError("n_action_steps cannot exceed action_horizon")
        if self.num_inference_steps > self.num_diffusion_steps:
            raise ValueError("num_inference_steps cannot exceed num_diffusion_steps")
        if self.token_dim % self.router_heads or self.token_dim % self.condition_heads:
            raise ValueError("token_dim must be divisible by both attention head counts")
        if self.control_dt <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("control_dt must be positive and dropout must be in [0,1)")
        if self.tactile_dim % self.tactile_channels:
            raise ValueError("tactile_dim must be divisible by tactile_channels")
        if len(self.dp_unet_down_dims) < 2 or any(d <= 0 or d % self.dp_unet_n_groups for d in self.dp_unet_down_dims):
            raise ValueError("At least two positive U-Net widths divisible by n_groups are required")
        if self.action_horizon % (2 ** (len(self.dp_unet_down_dims) - 1)):
            raise ValueError("action_horizon must be divisible by the U-Net downsampling factor")
        if self.dp_unet_diffusion_step_embed_dim < 4 or self.dp_unet_diffusion_step_embed_dim % 2:
            raise ValueError("diffusion step embedding width must be even and at least four")
        if self.dp_unet_kernel_size <= 0 or self.dp_unet_kernel_size % 2 == 0:
            raise ValueError("U-Net kernel size must be a positive odd number")
        if len(self.image_size) != 2 or min(self.image_size) < 32:
            raise ValueError("image_size must contain two dimensions of at least 32 pixels")
        if self.crop_size is not None and (len(self.crop_size) != 2 or min(self.crop_size) < 32 or any(c > i for c, i in zip(self.crop_size, self.image_size))):
            raise ValueError("crop_size must be >=32 and fit image_size")
        if not self.tactile_hidden_dims or any(d <= 0 for d in self.tactile_hidden_dims):
            raise ValueError("tactile_hidden_dims must contain positive widths")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown policy configuration keys: {sorted(unknown)}")
        data = dict(data)
        for name in ("image_size", "crop_size", "dp_unet_down_dims", "tactile_hidden_dims"):
            if data.get(name) is not None:
                data[name] = tuple(data[name])
        return cls(**data)


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return value
