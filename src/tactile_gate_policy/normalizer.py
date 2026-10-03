"""Training-only streaming numeric normalization with checkpointed buffers."""

import torch
from torch import nn


class NumericNormalizer(nn.Module):
    def __init__(self, action_dim=8, proprio_dim=22):
        super().__init__()
        if action_dim <= 0 or proprio_dim <= 0:
            raise ValueError("normalizer dimensions must be positive")
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))
        self.register_buffer("proprio_mean", torch.zeros(proprio_dim))
        self.register_buffer("proprio_std", torch.ones(proprio_dim))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def fit(self, dataset, max_windows=None):
        """Fit population statistics on numeric windows from the training split.

        Each sampled timestep contributes once per training window, matching the
        actual window sampling distribution. RGB and tactile data are not read.
        Memory use is independent of the number of windows.
        """
        if max_windows is not None and (isinstance(max_windows, bool) or int(max_windows) != max_windows or max_windows <= 0):
            raise ValueError("max_windows must be a positive integer or None")
        count = len(dataset) if max_windows is None else min(len(dataset), int(max_windows))
        if count == 0:
            raise ValueError("Cannot fit a normalizer on an empty training dataset")
        stats = {"action": [0, torch.zeros_like(self.action_mean, dtype=torch.float64, device="cpu"), torch.zeros_like(self.action_mean, dtype=torch.float64, device="cpu")],
                 "proprio": [0, torch.zeros_like(self.proprio_mean, dtype=torch.float64, device="cpu"), torch.zeros_like(self.proprio_mean, dtype=torch.float64, device="cpu")]}
        for index in range(count):
            numeric = dataset.get_numeric(index) if hasattr(dataset, "get_numeric") else dataset[index]
            for name, values in (("action", numeric["action"]), ("proprio", numeric["obs"]["proprio"])):
                n, mean, m2 = stats[name]
                values = torch.as_tensor(values, dtype=torch.float64, device="cpu")
                if values.ndim < 2 or values.shape[-1] != mean.numel():
                    raise ValueError(f"Unexpected {name} shape {tuple(values.shape)}")
                values = values.reshape(-1, mean.numel())
                if not values.numel() or not torch.isfinite(values).all():
                    raise ValueError(f"{name} statistics require nonempty finite values")
                batch_n = len(values)
                batch_mean = values.mean(0)
                batch_m2 = ((values - batch_mean) ** 2).sum(0)
                delta = batch_mean - mean
                total = n + batch_n
                stats[name] = [total, mean + delta * (batch_n / total),
                               m2 + batch_m2 + delta.square() * (n * batch_n / total)]
        for name, (n, mean, m2) in stats.items():
            target_mean = getattr(self, f"{name}_mean")
            target_std = getattr(self, f"{name}_std")
            target_mean.copy_(mean.to(target_mean))
            target_std.copy_((m2 / n).sqrt().clamp_min(1e-6).to(target_std))
        self.fitted.fill_(True)
        return self

    def _require_fitted(self):
        if not bool(self.fitted):
            raise RuntimeError("Fit NumericNormalizer on the training dataset before use")

    def normalize_obs(self, obs):
        self._require_fitted()
        result = dict(obs)
        result["proprio"] = (obs["proprio"] - self.proprio_mean) / self.proprio_std
        for name in ("left_tactile", "right_tactile"):
            result[name] = obs[name] / 255.0
        return result

    def normalize_action(self, action):
        self._require_fitted()
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action):
        self._require_fitted()
        return action * self.action_std + self.action_mean
