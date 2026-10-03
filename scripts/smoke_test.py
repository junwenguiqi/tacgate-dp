"""Exercise dataset -> Stage A -> resume -> Stage B -> evaluation -> closed loop."""

import copy
import json
import sys
import tempfile
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from tactile_gate_policy.checkpoint import load_policy
from tactile_gate_policy.demo_data import create_demo_dataset, tiny_configuration
from tactile_gate_policy.evaluate import evaluate
from tactile_gate_policy.rollout import ToyRobotAdapter, run_rollout
from tactile_gate_policy.train import train


def main():
    scratch = RELEASE_ROOT / ".smoke"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="integration_", dir=scratch) as temporary:
        root = Path(temporary)
        dataset = create_demo_dataset(root / "data")
        config = tiny_configuration()
        first = copy.deepcopy(config)
        first["training"]["max_steps"] = 1
        stage_a = train(first, dataset, root / "a")
        stage_a = train(config, dataset, root / "a", resume=stage_a["checkpoint"])
        stage_b_config = copy.deepcopy(config)
        stage_b_config["training"]["stage"] = "b"
        stage_b = train(stage_b_config, dataset, root / "b", init_checkpoint=stage_a["checkpoint"])
        metrics = evaluate(stage_b["checkpoint"], dataset, batch_size=2, max_batches=1, num_inference_steps=2, output=root / "evaluation")
        policy, _ = load_policy(stage_b["checkpoint"])
        rollout = run_rollout(policy, ToyRobotAdapter(policy.cfg), max_steps=3)
        if stage_a["global_step"] != 2 or stage_b["global_step"] != 2 or rollout["executed_steps"] != 3:
            raise RuntimeError("End-to-end integration did not complete")
        print(json.dumps({"status": "OK", "stage_a_steps": stage_a["global_step"], "stage_b_steps": stage_b["global_step"], "evaluation": metrics, "toy_rollout": rollout}, indent=2))


if __name__ == "__main__":
    main()
