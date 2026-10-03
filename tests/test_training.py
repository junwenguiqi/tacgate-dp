import copy
import tempfile
import unittest
from pathlib import Path

import torch

from tactile_gate_policy.checkpoint import load_checkpoint, load_policy
from tactile_gate_policy.demo_data import create_demo_dataset, tiny_configuration
from tactile_gate_policy.evaluate import evaluate
from tactile_gate_policy.rollout import ObservationBuffer, ToyRobotAdapter, run_rollout
from tactile_gate_policy.train import train


class TrainingIntegrationTests(unittest.TestCase):
    def test_resume_transfer_evaluate_and_rollout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_demo_dataset(root / "data", frames=32)
            configuration = tiny_configuration()
            initial = copy.deepcopy(configuration)
            initial["training"]["max_steps"] = 1
            result = train(initial, dataset, root / "resumed")
            self.assertEqual(result["global_step"], 1)
            result = train(configuration, dataset, root / "resumed", resume=result["checkpoint"])
            self.assertEqual(result["global_step"], 2)
            continuous = train(configuration, dataset, root / "continuous")
            resumed_state = load_checkpoint(result["checkpoint"])["policy_state"]
            continuous_state = load_checkpoint(continuous["checkpoint"])["policy_state"]
            for key in resumed_state:
                self.assertTrue(torch.equal(resumed_state[key], continuous_state[key]), key)
            del resumed_state, continuous_state
            stage_b = copy.deepcopy(configuration)
            stage_b["training"]["stage"] = "b"
            transferred = train(stage_b, dataset, root / "b", init_checkpoint=result["checkpoint"])
            metrics = evaluate(transferred["checkpoint"], dataset, max_batches=1, batch_size=1, num_inference_steps=2)
            self.assertGreater(metrics["samples"], 0)
            self.assertTrue(torch.isfinite(torch.tensor(metrics["action_mse"])))
            policy, _ = load_policy(transferred["checkpoint"])
            observed = run_rollout(policy, ToyRobotAdapter(policy.cfg), max_steps=3)
            self.assertEqual(observed["executed_steps"], 3)
            self.assertEqual(observed["predicted_chunks"], 2)
            buffer = ObservationBuffer(policy.cfg)
            with self.assertRaises(ValueError):
                buffer.batch()
            with self.assertRaises(ValueError):
                buffer.append({})
            mismatched = copy.deepcopy(configuration)
            mismatched["policy"]["control_dt"] = 0.3
            with self.assertRaisesRegex(ValueError, "interval"):
                train(mismatched, dataset, root / "wrong_dt")
            with self.assertRaisesRegex(ValueError, "Stage B requires"):
                train(stage_b, dataset, root / "uninitialized_b")


if __name__ == "__main__":
    unittest.main()
