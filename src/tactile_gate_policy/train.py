"""Stage A/B training, stage transfer and exact same-stage continuation."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .checkpoint import capture_rng, load_checkpoint, restore_rng, save_checkpoint
from .config import PolicyConfig, load_config
from .dataset import split_dataset
from .policy import LossConfig, TactileGatePolicy


@dataclass(frozen=True)
class TrainingConfig:
    stage: str = "a"
    epochs: int = 100
    batch_size: int = 16
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    warmup_steps: int = 100
    gradient_clip_norm: float = 1.0
    freeze_encoders: bool = False
    seed: int = 42
    device: str = "cpu"
    cpu_threads: int = 4
    max_steps: int | None = None
    max_normalizer_windows: int | None = None

    def __post_init__(self):
        if self.stage not in ("a", "b"):
            raise ValueError("stage must be 'a' or 'b'")
        if min(self.epochs, self.batch_size, self.cpu_threads) < 1 or self.num_workers < 0:
            raise ValueError("Invalid epochs / batch_size / num_workers / cpu_threads")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.warmup_steps < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("Invalid optimizer configuration")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive")


def move_batch(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_batch(item, device) for key, item in value.items()}
    return value


def make_datasets(dataset_root, policy_config, data_config, seed):
    allowed = {"frame_stride", "val_fraction", "require_sync_valid", "episode_indices"}
    unknown = set(data_config) - allowed
    if unknown:
        raise ValueError(f"Unknown data keys: {sorted(unknown)}")
    options = dict(data_config)
    fraction = options.pop("val_fraction", 0.2)
    return split_dataset(
        dataset_root, val_fraction=fraction, seed=seed,
        obs_horizon=policy_config.obs_horizon,
        action_horizon=policy_config.action_horizon,
        image_size=policy_config.image_size,
        **options,
    )


def _optimizer(policy, cfg):
    decay, no_decay = [], []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        excluded = parameter.ndim <= 1 or any(part in name.lower() for part in ("norm", "embedding", "mask_token", "pos_emb"))
        (no_decay if excluded else decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg.learning_rate)


def _lr_scheduler(optimizer, total_steps, warmup):
    def multiplier(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        fraction = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
        return 0.1 + 0.9 * (1.0 + math.cos(math.pi * fraction)) / 2.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


@torch.no_grad()
def validation_loss(policy, loader, stage, loss_config, seed):
    rng = capture_rng()
    was_training = policy.training
    try:
        torch.manual_seed(seed)
        policy.eval()
        total, count = 0.0, 0
        device = next(policy.parameters()).device
        for batch in loader:
            batch = move_batch(batch, device)
            info = policy.compute_loss(batch, stage=stage, loss_config=loss_config)
            size = batch["action"].shape[0]
            total += float(info["loss"]) * size
            count += size
        if not count:
            raise ValueError("Validation dataset has no windows")
        result = total / count
        if not math.isfinite(result):
            raise RuntimeError("Nonfinite validation loss")
        return result
    finally:
        policy.train(was_training)
        restore_rng(rng)


def train(configuration, dataset_root, output_dir, *, init_checkpoint=None, resume=None):
    """Train within one stage; init_checkpoint transfers weights, resume restores state."""
    if init_checkpoint and resume:
        raise ValueError("Choose either init_checkpoint or resume")
    if set(configuration) - {"policy", "training", "data", "loss"}:
        raise ValueError("Only policy/training/data/loss configuration sections are supported")
    cfg = TrainingConfig(**configuration.get("training", {}))
    requested_policy = PolicyConfig.from_dict(configuration.get("policy", {}))
    loss_config = LossConfig(**configuration.get("loss", {}))
    torch.set_num_threads(cfg.cpu_threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    device = torch.device(cfg.device)
    data_config = configuration.get("data", {})
    train_data, val_data = make_datasets(dataset_root, requested_policy, data_config, cfg.seed)
    for subset in (train_data, val_data):
        inferred = subset.control_dt
        if not math.isclose(inferred, requested_policy.control_dt, rel_tol=0.05, abs_tol=1e-4):
            raise ValueError(f"Data interval {inferred:.6f}s does not match policy.control_dt={requested_policy.control_dt}; adjust frame_stride and control_dt together")

    payload = load_checkpoint(resume or init_checkpoint, device) if (resume or init_checkpoint) else None
    if cfg.stage == "b" and payload is None:
        raise ValueError("Stage B requires --init-checkpoint from Stage A, or --resume")
    if payload and PolicyConfig.from_dict(payload["policy_config"]) != requested_policy:
        raise ValueError("Checkpoint architecture/configuration differs from requested policy")
    policy = TactileGatePolicy(replace(requested_policy, vision_weights=None) if payload else requested_policy).to(device)
    policy.cfg = requested_policy
    if payload:
        policy.load_state_dict(payload["policy_state"], strict=True)
    else:
        policy.normalizer.fit(train_data, max_windows=cfg.max_normalizer_windows)
    if not bool(policy.normalizer.fitted):
        raise ValueError("Checkpoint normalizer has not been fitted")

    for parameter in policy.model.router.parameters():
        parameter.requires_grad_(cfg.stage == "b")
    for parameter in policy.encoder.parameters():
        parameter.requires_grad_(not cfg.freeze_encoders)
    optimizer = _optimizer(policy, cfg)
    steps_per_epoch = math.ceil(len(train_data) / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    scheduler = _lr_scheduler(optimizer, total_steps, min(cfg.warmup_steps, max(0, total_steps - 1)))
    progress = {"epoch": 0, "batch_in_epoch": 0, "global_step": 0, "best_val_loss": None}
    metadata = {
        "stage": cfg.stage,
        "configuration": {"policy": requested_policy.to_dict(), "training": asdict(cfg), "data": data_config, "loss": configuration.get("loss", {})},
        "dataset_root": str(Path(dataset_root).resolve()),
        "train_split": train_data.split_manifest,
        "val_split": val_data.split_manifest,
    }
    if resume:
        old_metadata = payload["metadata"]
        if old_metadata.get("stage") != cfg.stage:
            raise ValueError("--resume must stay in the same stage; use --init-checkpoint to change stage")
        for key in ("dataset_root", "train_split", "val_split"):
            if old_metadata.get(key) != metadata[key]:
                raise ValueError(f"Cannot resume with changed {key}")
        old_configuration = old_metadata.get("configuration", {})
        old_training = dict(old_configuration.get("training", {}))
        current_training = asdict(cfg)
        for optional in ("max_steps", "device", "cpu_threads"):
            old_training.pop(optional, None)
            current_training.pop(optional, None)
        if old_training != current_training or old_configuration.get("loss", {}) != configuration.get("loss", {}):
            raise ValueError("Resume requires matching training/loss configuration, except device/cpu_threads/max_steps")
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        progress.update(payload["progress"])
        restore_rng(payload["rng_state"])
    elif init_checkpoint:
        for key in ("dataset_root", "train_split", "val_split"):
            if payload["metadata"].get(key) != metadata[key]:
                raise ValueError(f"Stage transfer requires the same {key} to preserve normalization and validation isolation")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    if last_path.exists() and not resume:
        raise FileExistsError(f"{last_path} exists; resume it or choose a new output directory")
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata["configuration"], handle, sort_keys=False)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    stopped = False
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as log:
        for epoch in range(progress["epoch"], cfg.epochs):
            generator = torch.Generator().manual_seed(cfg.seed + epoch)
            loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, generator=generator)
            skip = progress["batch_in_epoch"] if epoch == progress["epoch"] else 0
            for batch_index, batch in enumerate(loader):
                if batch_index < skip:
                    continue
                if cfg.max_steps is not None and progress["global_step"] >= cfg.max_steps:
                    stopped = True
                    break
                policy.train()
                if cfg.freeze_encoders:
                    policy.encoder.eval()
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                info = policy.compute_loss(batch, stage=cfg.stage, loss_config=loss_config)
                if not torch.isfinite(info["loss"]):
                    raise RuntimeError("Nonfinite training loss")
                info["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.gradient_clip_norm, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                progress.update(epoch=epoch, batch_in_epoch=batch_index + 1, global_step=progress["global_step"] + 1)
                record = {"stage": cfg.stage, "epoch": epoch, "step": progress["global_step"], "train_loss": float(info["loss"].detach()), "action_loss": float(info["action_loss"]), "gate_mean": float(info["gate"].mean()), "grad_norm": float(grad_norm), "lr": optimizer.param_groups[0]["lr"]}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
            completed_epoch = progress["batch_in_epoch"] >= steps_per_epoch
            if completed_epoch:
                progress.update(epoch=epoch + 1, batch_in_epoch=0)
            val_loss = validation_loss(policy, val_loader, cfg.stage, loss_config, cfg.seed + 10000)
            record = {"stage": cfg.stage, "step": progress["global_step"], "val_loss": val_loss}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if progress["best_val_loss"] is None or val_loss < progress["best_val_loss"]:
                progress["best_val_loss"] = val_loss
                save_checkpoint(output_dir / "best.pt", policy, optimizer=optimizer, scheduler=scheduler, progress=progress, metadata=metadata)
            save_checkpoint(last_path, policy, optimizer=optimizer, scheduler=scheduler, progress=progress, metadata=metadata)
            if stopped or (cfg.max_steps is not None and progress["global_step"] >= cfg.max_steps):
                break
    return {"checkpoint": str(last_path), "global_step": progress["global_step"], "best_val_loss": progress["best_val_loss"], "train_windows": len(train_data), "val_windows": len(val_data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--device")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    training = config.setdefault("training", {})
    for key in ("device", "max_steps"):
        if getattr(args, key) is not None:
            training[key] = getattr(args, key)
    print(json.dumps(train(config, args.dataset_root, args.output_dir, init_checkpoint=args.init_checkpoint, resume=args.resume)), flush=True)


if __name__ == "__main__":
    main()
