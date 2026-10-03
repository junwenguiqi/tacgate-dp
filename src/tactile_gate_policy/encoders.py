"""Real RGB, taxel CNN, and proprioception encoders for the five-modal policy.

The visual path follows Diffusion Policy's MIT-licensed ``get_resnet``,
``MultiImageObsEncoder`` and ``CropRandomizer`` implementation:
https://github.com/real-stanford/diffusion_policy
ResNet18 has its classification head removed, optionally uses GroupNorm,
and sees independent random training crops or deterministic evaluation crops.
The local implementation keeps the release independent of the research tree.

Touch is a one-dimensional index sequence, not an invented 2D sensor map.
The default 231 values are interpreted as 77 consecutive 3-channel taxels.
Set tactile_channels=1 if the source does not use that interleaved layout.
"""

from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .config import PolicyConfig


MODALITY_KEYS = (
    "global_rgb", "wrist_rgb", "left_tactile", "right_tactile", "proprio"
)


def _group_count(channels: int, preferred: int = 8) -> int:
    """Choose an exact divisor, including small configurable CNN widths."""
    return next(n for n in range(min(preferred, channels), 0, -1) if channels % n == 0)


def _replace_batch_norm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            # DP uses one group per 16 channels for the ResNet stages.
            replacement = nn.GroupNorm(
                num_groups=max(1, child.num_features // 16),
                num_channels=child.num_features,
                eps=child.eps,
                affine=child.affine,
            )
            if child.affine:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            _replace_batch_norm(child)


def make_dp_resnet18(cfg: PolicyConfig) -> nn.Module:
    """DP's torchvision ResNet18 backbone, with no classifier or auto-download."""
    weights = None
    if cfg.vision_weights is not None:
        if cfg.vision_weights == "DEFAULT":
            weights = ResNet18_Weights.DEFAULT
        else:
            try:
                weights = ResNet18_Weights[cfg.vision_weights]
            except KeyError as exc:
                raise ValueError(
                    "vision_weights must be None, DEFAULT, or a ResNet18_Weights name"
                ) from exc
    backbone = resnet18(weights=weights)
    backbone.fc = nn.Identity()
    if cfg.vision_group_norm:
        _replace_batch_norm(backbone)
    return backbone


class DPImageEncoder(nn.Module):
    """DP ResNet18 visual features projected to one policy observation token.

    Input is floating-point RGB ``(N,3,H,W)`` in [0,1]. Images are resized to
    image_size, cropped, and optionally ImageNet-normalized. Specifying explicit
    pretrained weights may download them; the default vision_weights=None does
    not access the network.
    """

    def __init__(self, cfg: PolicyConfig, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.image_size = tuple(cfg.image_size)
        self.crop_size = None if cfg.crop_size is None else tuple(cfg.crop_size)
        self.imagenet_norm = cfg.imagenet_norm
        self.backbone = make_dp_resnet18(cfg) if backbone is None else backbone
        self.projection = nn.Sequential(nn.Linear(512, cfg.token_dim), nn.LayerNorm(cfg.token_dim))
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) <= 0:
            raise ValueError("RGB encoder expects (N,3,H,W) with nonempty images")
        if not images.is_floating_point():
            raise TypeError("RGB inputs must be floating-point tensors in [0,1]")
        if tuple(images.shape[-2:]) != self.image_size:
            images = F.interpolate(images, size=self.image_size, mode="bilinear", align_corners=False, antialias=True)
        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            height, width = images.shape[-2:]
            if self.training:
                # One independent crop for each frame, as in DP CropRandomizer.
                top = torch.randint(height - crop_h + 1, (images.shape[0],), device=images.device)
                left = torch.randint(width - crop_w + 1, (images.shape[0],), device=images.device)
            else:
                top = torch.full((images.shape[0],), (height - crop_h) // 2, device=images.device, dtype=torch.long)
                left = torch.full((images.shape[0],), (width - crop_w) // 2, device=images.device, dtype=torch.long)
            rows = top[:, None, None] + torch.arange(crop_h, device=images.device)[None, :, None]
            cols = left[:, None, None] + torch.arange(crop_w, device=images.device)[None, None, :]
            pixels = rows * width + cols
            pixels = pixels.reshape(images.shape[0], 1, -1).expand(-1, 3, -1)
            images = images.flatten(2).gather(2, pixels).reshape(-1, 3, crop_h, crop_w)
        if self.imagenet_norm:
            images = (images - self.rgb_mean.to(dtype=images.dtype)) / self.rgb_std.to(dtype=images.dtype)
        return images

    def forward(self, images: Tensor) -> Tensor:
        return self.projection(self.backbone(self.preprocess(images)))


class TactileCNNEncoder(nn.Module):
    """Conv1d over taxel indices with configurable interleaved channels.

    Accepts ``(N,tactile_dim)`` or ``(N,tactile_dim/channels,channels)``.
    Values must already be scaled by the dataset/online preprocessing layer.
    This does not claim that neighbouring indices are neighbouring physical
    taxels; a calibrated physical map can replace this encoder later.
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.input_dim = cfg.tactile_dim
        self.channels = cfg.tactile_channels
        self.taxels = cfg.tactile_dim // cfg.tactile_channels
        layers = []
        input_channels = self.channels
        for output_channels in cfg.tactile_hidden_dims:
            layers.extend([
                nn.Conv1d(input_channels, output_channels, kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(_group_count(output_channels), output_channels),
                nn.GELU(),
            ])
            input_channels = output_channels
        layers.extend([nn.AdaptiveAvgPool1d(1), nn.Flatten(start_dim=1)])
        self.cnn = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Dropout(cfg.dropout), nn.Linear(input_channels, cfg.token_dim), nn.LayerNorm(cfg.token_dim)
        )

    def forward(self, tactile: Tensor) -> Tensor:
        if tactile.ndim == 2 and tactile.shape[-1] == self.input_dim:
            tactile = tactile.reshape(tactile.shape[0], self.taxels, self.channels)
        elif tactile.ndim != 3 or tuple(tactile.shape[1:]) != (self.taxels, self.channels):
            raise ValueError(f"Touch must be (N,{self.input_dim}) or (N,{self.taxels},{self.channels})")
        if not tactile.is_floating_point():
            raise TypeError("Touch inputs must be floating point and already scaled")
        return self.projection(self.cnn(tactile.transpose(1, 2)))


class MultimodalObsEncoder(nn.Module):
    """Encode synchronized observations into ``(B,T,5,token_dim)``.

    Fixed modality order is global_rgb, wrist_rgb, left_tactile, right_tactile,
    proprio. RGB shape is (B,T,3,H,W), touch is (B,T,231) or (B,T,77,3), and
    proprioception is (B,T,22), subject to configuration. Two cameras use
    independent projection heads even when their ResNet backbone is shared.
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.share_vision_backbone:
            backbone = make_dp_resnet18(cfg)
            self.vision_encoders = nn.ModuleDict({key: DPImageEncoder(cfg, backbone) for key in MODALITY_KEYS[:2]})
        else:
            self.vision_encoders = nn.ModuleDict({key: DPImageEncoder(cfg) for key in MODALITY_KEYS[:2]})
        left = TactileCNNEncoder(cfg)
        right = left if cfg.share_tactile_backbone else TactileCNNEncoder(cfg)
        self.tactile_encoders = nn.ModuleDict({"left_tactile": left, "right_tactile": right})
        self.proprio_encoder = nn.Sequential(
            nn.Linear(cfg.proprio_dim, cfg.token_dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.token_dim, cfg.token_dim), nn.LayerNorm(cfg.token_dim),
        )

    def forward(self, obs: Dict[str, Tensor]) -> Tensor:
        missing = set(MODALITY_KEYS) - set(obs)
        if missing:
            raise KeyError(f"Missing observation modalities: {sorted(missing)}")
        proprio = obs["proprio"]
        if proprio.ndim != 3 or proprio.shape[-1] != self.cfg.proprio_dim:
            raise ValueError(f"proprio must have shape (B,T,{self.cfg.proprio_dim})")
        batch, horizon = proprio.shape[:2]
        if batch <= 0 or horizon != self.cfg.obs_horizon:
            raise ValueError(f"Observations require a nonempty batch and T={self.cfg.obs_horizon}")
        tokens = []
        for key in MODALITY_KEYS:
            value = obs[key]
            if tuple(value.shape[:2]) != (batch, horizon):
                raise ValueError(f"{key} batch and time dimensions must match proprio")
            if not value.is_floating_point():
                raise TypeError(f"{key} must be a floating-point tensor")
            if value.device != proprio.device or value.dtype != proprio.dtype:
                raise ValueError("All observation modalities must have matching device and dtype")
            if key in self.vision_encoders:
                if value.ndim != 5 or value.shape[2] != 3:
                    raise ValueError(f"{key} must have shape (B,T,3,H,W)")
                feature = self.vision_encoders[key](value.flatten(0, 1))
            elif key in self.tactile_encoders:
                if value.ndim not in (3, 4):
                    raise ValueError(f"{key} must be flat or taxel/channel touch")
                feature = self.tactile_encoders[key](value.flatten(0, 1))
            else:
                feature = self.proprio_encoder(value.flatten(0, 1))
            tokens.append(feature.reshape(batch, horizon, self.cfg.token_dim))
        return torch.stack(tokens, dim=2)
