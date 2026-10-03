"""Offline holdout evaluation; action errors are not robot task success rates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import load_policy
from .train import make_datasets, move_batch


@torch.no_grad()
def evaluate(checkpoint, dataset_root=None, *, device="cpu", batch_size=4, max_batches=None, seed=42, num_inference_steps=None, output=None):
    if batch_size <= 0 or (max_batches is not None and max_batches <= 0):
        raise ValueError("batch_size and max_batches must be positive")
    policy, payload = load_policy(checkpoint, device)
    metadata = payload["metadata"]
    configuration = metadata.get("configuration")
    if not configuration:
        raise ValueError("Evaluation requires a training checkpoint with dataset split metadata")
    dataset_root = dataset_root or metadata["dataset_root"]
    _, validation = make_datasets(dataset_root, policy.cfg, configuration.get("data", {}), configuration["training"]["seed"])
    if validation.split_manifest != metadata.get("val_split"):
        raise ValueError("Validation split differs from the checkpoint's held-out split")
    generator = torch.Generator(device=device).manual_seed(seed)
    loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    total_elements = position_elements = gripper_elements = quaternion_count = 0
    square_error = position_error = gripper_error = quaternion_error = 0.0
    gate_sum = torch.zeros(policy.cfg.obs_horizon, 5, dtype=torch.float64)
    samples = 0
    predicted_chunks, target_chunks, gates = [], [], []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = move_batch(batch, device)
        result = policy.predict_action(batch["obs"], generator=generator, num_inference_steps=num_inference_steps)
        prediction, target = result["action_pred"], batch["action"]
        square_error += float((prediction - target).square().sum())
        total_elements += target.numel()
        position_error += float((prediction[..., :3] - target[..., :3]).abs().sum())
        position_elements += target[..., :3].numel()
        gripper_error += float((prediction[..., 7] - target[..., 7]).abs().sum())
        gripper_elements += target[..., 7].numel()
        q_pred, q_target = prediction[..., 3:7], target[..., 3:7]
        target_norm = q_target.norm(dim=-1, keepdim=True)
        if (target_norm < 1e-6).any():
            raise ValueError("Ground-truth actions contain zero quaternions")
        q_target = q_target / target_norm
        dot = (q_pred * q_target).sum(dim=-1).abs().clamp(0, 1)
        quaternion_error += float((2 * torch.acos(dot)).sum())
        quaternion_count += dot.numel()
        gate_sum += result["gate"].detach().cpu().double().sum(dim=0)
        samples += target.shape[0]
        if output:
            predicted_chunks.append(prediction.cpu().numpy())
            target_chunks.append(target.cpu().numpy())
            gates.append(result["gate"].cpu().numpy())
    if not samples:
        raise ValueError("No validation samples were evaluated")
    mean_gate = gate_sum / samples
    metrics = {
        "split": "validation",
        "checkpoint_stage": metadata.get("stage"),
        "samples": samples,
        "action_mse": square_error / total_elements,
        "position_mae": position_error / position_elements,
        "gripper_mae": gripper_error / gripper_elements,
        "quaternion_angle_error_rad": quaternion_error / quaternion_count,
        "gate_per_modality": mean_gate.mean(dim=0).tolist(),
        "gate_heatmap": mean_gate.tolist(),
        "seed": seed,
        "num_inference_steps": num_inference_steps or policy.cfg.num_inference_steps,
    }
    if not all(np.isfinite(metrics[key]) for key in ("action_mse", "position_mae", "gripper_mae", "quaternion_angle_error_rad")):
        raise RuntimeError("Nonfinite evaluation metrics")
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        np.savez_compressed(output / "predictions.npz", action_pred=np.concatenate(predicted_chunks), action_target=np.concatenate(target_chunks), gate=np.concatenate(gates))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    result = evaluate(args.checkpoint, args.dataset_root, device=args.device, batch_size=args.batch_size, max_batches=args.max_batches, seed=args.seed, num_inference_steps=args.num_inference_steps, output=args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
