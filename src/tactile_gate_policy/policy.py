"""End-to-end multimodal DDPM training and receding-horizon inference."""

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import DDPMScheduler
from torch import nn
from torch.nn import functional as F

from .config import PolicyConfig
from .encoders import MultimodalObsEncoder
from .model import GatedMultimodalPolicy, gate_budget_loss, gate_smooth_loss
from .normalizer import NumericNormalizer


@dataclass(frozen=True)
class LossConfig:
    min_keep_prob: float = 0.35
    max_keep_prob: float = 0.95
    target_gate_mean: float = 0.65
    lambda_budget: float = 0.1
    lambda_smooth: float = 0.02
    lambda_tactile_symmetry: float = 0.0

    def __post_init__(self):
        if not 0 <= self.min_keep_prob <= self.max_keep_prob <= 1:
            raise ValueError("Keep probabilities must satisfy 0 <= min <= max <= 1")
        if not 0 <= self.target_gate_mean <= 1:
            raise ValueError("target_gate_mean must be in [0,1]")
        if min(self.lambda_budget, self.lambda_smooth, self.lambda_tactile_symmetry) < 0:
            raise ValueError("Gate regularization weights cannot be negative")


def sample_random_gate(batch_size, obs_horizon, num_modalities, *, device=None,
                       dtype=torch.float32, min_keep_prob=0.35, max_keep_prob=0.95,
                       generator=None):
    """Sample Bernoulli gates with one uniform keep probability per item.

    An all-missing time row gets one uniformly selected modality restored.
    This fallback changes the marginal keep rate slightly and prevents fully
    masked observations. Random draws do not touch any input observations.
    """
    if not 0 <= min_keep_prob <= max_keep_prob <= 1:
        raise ValueError("Keep probabilities must satisfy 0 <= min <= max <= 1")
    if min(batch_size, obs_horizon, num_modalities) < 1:
        raise ValueError("Random-gate dimensions must be positive")
    probability = torch.rand((batch_size, 1, 1), device=device, generator=generator)
    probability = min_keep_prob + (max_keep_prob - min_keep_prob) * probability
    gate = (torch.rand((batch_size, obs_horizon, num_modalities), device=device,
                       generator=generator) < probability).to(dtype=dtype)
    missing = gate.sum(dim=-1) == 0
    chosen = torch.randint(num_modalities, (batch_size, obs_horizon), device=device,
                           generator=generator)
    fallback = F.one_hot(chosen, num_classes=num_modalities).to(dtype=dtype)
    return torch.where(missing.unsqueeze(-1), fallback, gate)


def _make_scheduler(num_diffusion_steps):
    # Match DP's DDPM epsilon objective. Standardized actions are unbounded;
    # clipping them to [-1,1] would distort learned action statistics.
    return DDPMScheduler(
        num_train_timesteps=num_diffusion_steps, beta_start=0.0001, beta_end=0.02,
        beta_schedule="squaredcos_cap_v2", variance_type="fixed_small",
        clip_sample=False, prediction_type="epsilon",
    )


class TactileGatePolicy(nn.Module):
    """RGB ResNet18 + tactile CNN + time/modality gate + DP action U-Net.

    Public observations contain RGB tensors in [0,1] (or uint8), raw 0..255
    tactile distribution values, and unnormalized robot state. Action training
    labels are raw xyz, quaternion(x,y,z,w), gripper. Numeric statistics are
    stored as registered buffers and therefore travel with checkpoints.
    """

    def __init__(self, cfg: Optional[PolicyConfig] = None):
        super().__init__()
        self.cfg = cfg or PolicyConfig()
        self.encoder = MultimodalObsEncoder(self.cfg)
        self.model = GatedMultimodalPolicy(self.cfg)
        self.normalizer = NumericNormalizer(action_dim=self.cfg.action_dim,
                                            proprio_dim=self.cfg.proprio_dim)
        self.noise_scheduler = _make_scheduler(self.cfg.num_diffusion_steps)

    def _normalize_obs(self, obs):
        if not isinstance(obs, dict):
            raise TypeError("Observations must be a modality-to-tensor dictionary")
        values = dict(obs)
        # Keep the mapping and caller tensors unchanged. Do not infer whether
        # floating RGB values are 0..255: the public floating contract is [0,1].
        for key in ("global_rgb", "wrist_rgb"):
            value = values[key]
            if value.dtype == torch.uint8:
                value = value.float() / 255.0
            elif not value.is_floating_point():
                raise TypeError(f"{key} must be uint8 or floating RGB")
            if not torch.isfinite(value).all() or value.amin() < 0 or value.amax() > 1:
                raise ValueError(f"{key} floating RGB must be finite and in [0,1]")
            values[key] = value
        return self.normalizer.normalize_obs(values)

    def compute_loss(self, batch, stage="a", loss_config=None):
        """Training noise MSE; stage B additionally learns router penalties.

        Stage A bypasses the Router using random binary gates. The training
        entry point controls encoder/Router requires_grad for stage transfer.
        Returns one differentiable 'loss'; reported metrics are detached.
        """
        if stage not in ("a", "b"):
            raise ValueError("stage must be 'a' or 'b'")
        options = loss_config or LossConfig()
        if isinstance(options, dict):
            options = LossConfig(**options)
        action = batch["action"]
        expected = (self.cfg.action_horizon, self.cfg.action_dim)
        if action.ndim != 3 or tuple(action.shape[1:]) != expected or action.shape[0] < 1:
            raise ValueError(f"Training actions must be nonempty (B,{expected[0]},{expected[1]})")
        if not action.is_floating_point() or not torch.isfinite(action).all():
            raise ValueError("Training actions must be finite floating-point values")
        action = self.normalizer.normalize_action(action)
        tokens = self.encoder(self._normalize_obs(batch["obs"]))
        if tokens.shape[0] != action.shape[0]:
            raise ValueError("Action and observation batch sizes do not match")
        noise = torch.randn_like(action)
        timestep = torch.randint(self.cfg.num_diffusion_steps, (action.shape[0],),
                                 device=action.device, dtype=torch.long)
        noisy_action = self.noise_scheduler.add_noise(action, noise, timestep)
        gate = None
        if stage == "a":
            gate = sample_random_gate(
                action.shape[0], self.cfg.obs_horizon, self.cfg.num_modalities,
                device=tokens.device, dtype=tokens.dtype,
                min_keep_prob=options.min_keep_prob, max_keep_prob=options.max_keep_prob,
            )
        output = self.model(tokens, noisy_action, timestep, gate)
        gate = output["gate"]
        action_loss = F.mse_loss(output["pred_action_noise"], noise)
        budget = gate_budget_loss(gate, options.target_gate_mean)
        smooth = gate_smooth_loss(gate)
        symmetry = (gate[:, :, 2] - gate[:, :, 3]).abs().mean()
        loss = action_loss
        if stage == "b":
            loss = loss + options.lambda_budget * budget + options.lambda_smooth * smooth
            loss = loss + options.lambda_tactile_symmetry * symmetry
        return {
            "loss": loss, "action_loss": action_loss.detach(), "gate": gate.detach(),
            "gate_mean": gate.mean().detach(), "budget_loss": budget.detach(),
            "smooth_loss": smooth.detach(), "tactile_symmetry_loss": symmetry.detach(),
        }

    @torch.no_grad()
    def predict_action(self, obs, generator=None, num_inference_steps=None):
        """Reverse DDPM once and execute a prefix of the predicted horizon.

        Gates and the global condition are computed once from historical
        observations. No future ground-truth actions enter inference. A caller
        supplied torch.Generator controls all action diffusion randomness.
        """
        steps = self.cfg.num_inference_steps if num_inference_steps is None else num_inference_steps
        if not isinstance(steps, int) or not 1 <= steps <= self.cfg.num_diffusion_steps:
            raise ValueError("Inference steps must be an integer in [1,num_diffusion_steps]")
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            tokens = self.encoder(self._normalize_obs(obs))
            condition, gate = self.model.encode_condition(tokens)
            action = torch.randn(
                (tokens.shape[0], self.cfg.action_horizon, self.cfg.action_dim),
                device=tokens.device, dtype=tokens.dtype, generator=generator,
            )
            scheduler = _make_scheduler(self.cfg.num_diffusion_steps)
            scheduler.set_timesteps(steps, device=tokens.device)
            for timestep in scheduler.timesteps:
                noise = self.model.action_head(action, timestep, condition)
                action = scheduler.step(noise, timestep, action, generator=generator).prev_sample
            action = self.normalizer.denormalize_action(action)
            if not torch.isfinite(action).all():
                raise RuntimeError("Reverse diffusion produced nonfinite actions")
            # Executable poses require unit quaternions. Near-zero predictions
            # use xyzw identity rather than an undefined orientation.
            quaternion = action[..., 3:7]
            norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
            identity = torch.zeros_like(quaternion)
            identity[..., 3] = 1
            unit = torch.where(norm > 1e-8, quaternion / norm.clamp_min(1e-8), identity)
            action = torch.cat((action[..., :3], unit, action[..., 7:]), dim=-1)
            return {
                "action": action[:, :self.cfg.n_action_steps], "action_pred": action,
                "gate": gate, "condition": condition,
            }
        finally:
            # Preserve mixed modes, e.g. policy.train() with encoder.eval().
            # Calling self.train(old_mode) would erase frozen encoder settings.
            for module, training in modes:
                module.training = training
