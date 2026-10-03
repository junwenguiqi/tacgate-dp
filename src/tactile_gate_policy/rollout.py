"""Receding-horizon runner and an explicit adapter boundary for robot systems."""

import argparse
import importlib
import json
import time
from collections import deque
from typing import Mapping, Protocol

import numpy as np
import torch

from .checkpoint import load_policy
from .images import prepare_rgb

MODALITIES = ("global_rgb", "wrist_rgb", "left_tactile", "right_tactile", "proprio")


class RobotAdapter(Protocol):
    def observe(self) -> Mapping:
        """Return the next synchronized control-tick observation, blocking if needed."""

    def execute(self, action: np.ndarray, control_dt: float) -> bool:
        """Execute an absolute 8D action for one control interval; return terminal."""


class ObservationBuffer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.frames = deque(maxlen=cfg.obs_horizon)

    def append(self, observation):
        missing = set(MODALITIES) - set(observation)
        if missing:
            raise ValueError(f"Missing observation modalities: {sorted(missing)}")
        frame = {}
        for key in ("global_rgb", "wrist_rgb"):
            frame[key] = prepare_rgb(observation[key], self.cfg.image_size)
        for key in ("left_tactile", "right_tactile"):
            tactile = torch.as_tensor(observation[key], dtype=torch.float32).detach().cpu().flatten().clone()
            if tactile.numel() == self.cfg.tactile_dim + 3:
                tactile = tactile[3:]
            if tactile.numel() != self.cfg.tactile_dim or not torch.isfinite(tactile).all() or tactile.min() < 0 or tactile.max() > 255:
                raise ValueError("Tactile input must contain 231 distributed values in [0,255], or 234 values with three leading force channels")
            frame[key] = tactile
        proprio = torch.as_tensor(observation["proprio"], dtype=torch.float32).detach().cpu().flatten().clone()
        if proprio.numel() != self.cfg.proprio_dim or not torch.isfinite(proprio).all():
            raise ValueError(f"proprio requires {self.cfg.proprio_dim} finite raw values")
        frame["proprio"] = proprio
        self.frames.append(frame)

    def batch(self, device="cpu"):
        if len(self.frames) != self.cfg.obs_horizon:
            raise ValueError("Collect a complete observation history before prediction")
        return {key: torch.stack([frame[key] for frame in self.frames]).unsqueeze(0).to(device) for key in MODALITIES}


def run_rollout(policy, adapter: RobotAdapter, *, max_steps=40, seed=42):
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    device = next(policy.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    history = ObservationBuffer(policy.cfg)
    for _ in range(policy.cfg.obs_horizon):
        history.append(adapter.observe())
    steps = chunks = 0
    terminal = False
    gate_sum = torch.zeros(5)
    inference_seconds = 0.0
    while steps < max_steps and not terminal:
        start = time.perf_counter()
        prediction = policy.predict_action(history.batch(device), generator=generator)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - start
        chunks += 1
        gate_sum += prediction["gate"].detach().cpu().mean(dim=(0, 1))
        for action in prediction["action"][0].detach().cpu().numpy():
            if steps >= max_steps:
                break
            if not np.isfinite(action).all():
                raise RuntimeError("Policy produced a nonfinite command")
            terminal = bool(adapter.execute(action.copy(), policy.cfg.control_dt))
            steps += 1
            if terminal:
                break
            history.append(adapter.observe())
    return {"executed_steps": steps, "predicted_chunks": chunks, "terminal": terminal, "gate_per_modality": (gate_sum / chunks).tolist(), "mean_inference_seconds": inference_seconds / chunks}


class ToyRobotAdapter:
    """In-memory adapter for integration checks; no real hardware connection."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.pose = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.gripper = 0.04
        self.tick = 0

    def observe(self):
        self.tick += 1
        h, w = self.cfg.image_size
        x = np.linspace(0, 1, w, dtype=np.float32)[None, :].repeat(h, axis=0)
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None].repeat(w, axis=1)
        image = np.stack([x, y, np.full_like(x, (self.tick % 10) / 10)], axis=-1)
        proprio = np.concatenate([self.pose, np.zeros(14, dtype=np.float32), [self.gripper]]).astype(np.float32)
        tactile = (np.sin(np.arange(self.cfg.tactile_dim) / 8 + self.tick / 10) + 1) * 60
        return {"global_rgb": image, "wrist_rgb": image[:, ::-1].copy(), "left_tactile": tactile.astype(np.float32), "right_tactile": tactile[::-1].astype(np.float32).copy(), "proprio": proprio}

    def execute(self, action, control_dt):
        self.pose = action[:7].copy()
        self.gripper = float(action[7])
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", default="toy", help="toy or importable_module:factory; factory receives PolicyConfig")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    policy, payload = load_policy(args.checkpoint, args.device)
    if payload["metadata"].get("stage") != "b":
        raise ValueError("Rollout requires a Stage B checkpoint with a trained router")
    if args.adapter == "toy":
        adapter = ToyRobotAdapter(policy.cfg)
    else:
        module_name, separator, factory_name = args.adapter.partition(":")
        if not separator or not module_name or not factory_name:
            raise ValueError("Adapter must be 'toy' or 'module:factory'")
        adapter = getattr(importlib.import_module(module_name), factory_name)(policy.cfg)
    print(json.dumps(run_rollout(policy, adapter, max_steps=args.max_steps, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
