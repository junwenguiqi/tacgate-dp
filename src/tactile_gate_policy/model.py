"""Observation routing, feature gating, and DP action noise prediction."""

from typing import Optional

import torch
from torch import nn

from .config import PolicyConfig
from .vendor.diffusion_policy.conditional_unet1d import ConditionalUnet1D


DEFAULT_MODALITIES = ("global_rgb", "wrist_rgb", "left_tactile", "right_tactile", "proprio")


def _check_tokens(z_obs, cfg):
    expected = (cfg.obs_horizon, cfg.num_modalities, cfg.token_dim)
    if z_obs.ndim != 4 or tuple(z_obs.shape[1:]) != expected:
        raise ValueError(f"Observation tokens must have shape (B, {expected}), got {tuple(z_obs.shape)}")


class TimeModalityRouter(nn.Module):
    """Predict an independent sigmoid gate for each time/modality token."""

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.time_embedding = nn.Parameter(torch.empty(1, cfg.obs_horizon, 1, cfg.token_dim))
        self.modality_embedding = nn.Parameter(torch.empty(1, 1, cfg.num_modalities, cfg.token_dim))
        layer = nn.TransformerEncoderLayer(
            cfg.token_dim, cfg.router_heads, cfg.token_dim * 4, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.router_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(cfg.token_dim), nn.Linear(cfg.token_dim, 1))
        nn.init.normal_(self.time_embedding, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)

    def forward(self, z_obs):
        _check_tokens(z_obs, self.cfg)
        tokens = z_obs + self.time_embedding + self.modality_embedding
        tokens = self.encoder(tokens.flatten(1, 2))
        return torch.sigmoid(self.head(tokens)).reshape(z_obs.shape[:3])


class FeatureGateApplier(nn.Module):
    """Blend observations with learned per-modality missing-data tokens."""

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.num_modalities, cfg.token_dim))

    def forward(self, z_obs, gate):
        if gate.shape != z_obs.shape[:3]:
            raise ValueError(f"Gate shape must be {tuple(z_obs.shape[:3])}, got {tuple(gate.shape)}")
        weight = gate.unsqueeze(-1)
        return weight * z_obs + (1 - weight) * self.mask_token


class CrossModalConditionBackbone(nn.Module):
    """Fuse gated tokens into one condition for the entire action horizon."""

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.time_embedding = nn.Parameter(torch.empty(1, cfg.obs_horizon, 1, cfg.token_dim))
        self.modality_embedding = nn.Parameter(torch.empty(1, 1, cfg.num_modalities, cfg.token_dim))
        layer = nn.TransformerEncoderLayer(
            cfg.token_dim, cfg.condition_heads, cfg.token_dim * 4, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.condition_layers, enable_nested_tensor=False)
        self.proj = nn.Sequential(
            nn.LayerNorm(cfg.token_dim), nn.Linear(cfg.token_dim, cfg.condition_dim),
            nn.GELU(), nn.LayerNorm(cfg.condition_dim),
        )
        nn.init.normal_(self.time_embedding, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)

    def forward(self, tokens):
        _check_tokens(tokens, self.cfg)
        tokens = tokens + self.time_embedding + self.modality_embedding
        return self.proj(self.encoder(tokens.flatten(1, 2)).mean(dim=1))


class DPActionDiffusionHead(nn.Module):
    """ConditionalUnet1D from the original Diffusion Policy implementation."""

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.model = ConditionalUnet1D(
            input_dim=cfg.action_dim, local_cond_dim=None, global_cond_dim=cfg.condition_dim,
            diffusion_step_embed_dim=cfg.dp_unet_diffusion_step_embed_dim,
            down_dims=cfg.dp_unet_down_dims, kernel_size=cfg.dp_unet_kernel_size,
            n_groups=cfg.dp_unet_n_groups, cond_predict_scale=cfg.dp_unet_cond_predict_scale,
        )

    def forward(self, noisy_action, timestep, condition):
        if noisy_action.ndim != 3 or tuple(noisy_action.shape[1:]) != (self.cfg.action_horizon, self.cfg.action_dim):
            raise ValueError(f"Noisy action must be (B,{self.cfg.action_horizon},{self.cfg.action_dim})")
        if tuple(condition.shape) != (noisy_action.shape[0], self.cfg.condition_dim):
            raise ValueError("Condition batch or width does not match noisy action")
        return self.model(sample=noisy_action, timestep=timestep, global_cond=condition)


class GatedMultimodalPolicy(nn.Module):
    """Token-level model shared by stage A/B and reverse diffusion inference."""

    def __init__(self, cfg: Optional[PolicyConfig] = None):
        super().__init__()
        self.cfg = cfg or PolicyConfig()
        self.router = TimeModalityRouter(self.cfg)
        self.gate_applier = FeatureGateApplier(self.cfg)
        self.condition_backbone = CrossModalConditionBackbone(self.cfg)
        self.action_head = DPActionDiffusionHead(self.cfg)

    def encode_condition(self, z_obs, gate=None):
        _check_tokens(z_obs, self.cfg)
        gate = self.router(z_obs) if gate is None else gate
        condition = self.condition_backbone(self.gate_applier(z_obs, gate))
        return condition, gate

    def forward(self, z_obs, noisy_action, timestep, gate=None):
        condition, gate = self.encode_condition(z_obs, gate)
        return {
            "gate": gate, "condition": condition,
            "pred_action_noise": self.action_head(noisy_action, timestep, condition),
        }


def gate_budget_loss(gate, target_mean=0.65):
    return (gate.mean() - target_mean).square()


def gate_smooth_loss(gate):
    return (gate[:, 1:] - gate[:, :-1]).abs().mean() if gate.shape[1] > 1 else gate.new_zeros(())
