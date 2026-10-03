"""Generate a small FR3-shaped fixture for integration checks, not task training."""

from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from PIL import Image


def create_demo_dataset(root, *, frames=48, image_size=(32, 32), dt=0.1, episode_index=0):
    root = Path(root)
    if frames <= 0 or dt <= 0:
        raise ValueError("frames and dt must be positive")
    episode_dir = root / "data" / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    parquet = episode_dir / f"episode_{episode_index:06d}.parquet"
    if parquet.exists():
        raise FileExistsError(parquet)
    left_rel = f"tactile/episode_{episode_index:06d}/left.zarr"
    right_rel = f"tactile/episode_{episode_index:06d}/right.zarr"
    (root / left_rel).parent.mkdir(parents=True, exist_ok=True)
    arrays = []
    for relative in (left_rel, right_rel):
        array = zarr.open(str(root / relative), mode="w", shape=(frames, 234), chunks=(min(frames, 64), 234), dtype="u1")
        values = np.zeros((frames, 234), dtype=np.uint8)
        values[:, 3:] = np.clip(120 + 70 * np.sin(np.arange(frames)[:, None] * 0.1 + np.arange(231)[None, :] * 0.04), 0, 255).astype(np.uint8)
        array[:] = values
        arrays.append(array)
    h, w = image_size
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :].repeat(h, axis=0)
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None].repeat(w, axis=1)
    rows = []
    for frame in range(frames):
        rgb = np.stack([x, y, np.full_like(x, (frame * 7) % 256)], axis=-1)
        front_rel = f"images/episode_{episode_index:06d}/front_{frame:06d}.png"
        wrist_rel = f"images/episode_{episode_index:06d}/wrist_{frame:06d}.png"
        (root / front_rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(root / front_rel)
        Image.fromarray(rgb[:, ::-1].copy()).save(root / wrist_rel)
        pose = [0.4 + frame * 0.002, 0.02 * np.sin(frame * 0.1), 0.3, 0.0, 0.0, 0.0, 1.0]
        gripper = float(0.04 + 0.005 * np.sin(frame * 0.1))
        row = {
            "frame_index": frame, "timestamp": frame * dt,
            "image.path": front_rel, "wrist_image.path": wrist_rel,
            "image.width": w, "image.height": h,
            "wrist_image.width": w, "wrist_image.height": h,
            "robot.ee_pose": pose, "robot.q": [float(np.sin(frame * 0.05 + j)) for j in range(7)],
            "robot.dq": [float(0.05 * np.cos(frame * 0.05 + j)) for j in range(7)],
            "robot.gripper_pos": gripper, "action.sent_action8": pose + [gripper], "sync.valid": True,
        }
        for side, relative in (("left", left_rel), ("right", right_rel)):
            row.update({f"tactile.{side}.zarr_path": relative, f"tactile.{side}.zarr_index": frame, f"tactile.{side}.force_raw_len": 3, f"tactile.{side}.distribution_raw_start": 3, f"tactile.{side}.distribution_raw_len": 231})
        rows.append(row)
    pd.DataFrame(rows).to_parquet(parquet, index=False)
    return root


def tiny_configuration():
    from .config import PolicyConfig

    cfg = PolicyConfig(
        obs_horizon=2, action_horizon=4, n_action_steps=2, control_dt=0.1,
        token_dim=32, condition_dim=32, num_diffusion_steps=4, num_inference_steps=2,
        router_layers=1, router_heads=4, condition_layers=1, condition_heads=4,
        dropout=0.1, dp_unet_diffusion_step_embed_dim=32, dp_unet_down_dims=(16, 32),
        image_size=(32, 32), crop_size=(32, 32), share_vision_backbone=True,
        tactile_hidden_dims=(16, 32),
    )
    return {
        "policy": cfg.to_dict(),
        "data": {"frame_stride": 1, "val_fraction": 0.25, "require_sync_valid": True},
        "training": {"stage": "a", "epochs": 1, "batch_size": 2, "num_workers": 0, "seed": 42, "device": "cpu", "cpu_threads": 1, "freeze_encoders": True, "warmup_steps": 0, "max_steps": 2},
        "loss": {"target_gate_mean": 0.65, "lambda_budget": 0.1, "lambda_smooth": 0.02},
    }
