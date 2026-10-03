"""Checkpoint policy, optimizer, configuration and reproducible RNG state."""

import os
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import PolicyConfig

CHECKPOINT_VERSION = 1


def capture_rng():
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (name, keys.tolist(), position, has_gauss, cached_gaussian),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gaussian))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(path, policy, *, optimizer=None, scheduler=None, progress=None, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "policy_config": policy.cfg.to_dict(),
        "policy_state": policy.state_dict(),
        "progress": progress or {},
        "metadata": metadata or {},
        "rng_state": capture_rng(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path, device="cpu"):
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint format")
    for key in ("policy_config", "policy_state", "progress", "metadata", "rng_state"):
        if key not in payload:
            raise ValueError(f"Checkpoint is missing {key}")
    return payload


def load_policy(path, device="cpu"):
    from .policy import TactileGatePolicy

    payload = load_checkpoint(path, device)
    config = PolicyConfig.from_dict(payload["policy_config"])
    # The saved state already includes backbone weights; restore without a download.
    policy = TactileGatePolicy(replace(config, vision_weights=None)).to(device)
    policy.cfg = config
    policy.load_state_dict(payload["policy_state"], strict=True)
    policy.eval()
    return policy, payload
