"""Behavioral checks for end-to-end routing, training, and action sampling."""

import unittest
from unittest import mock

import torch

from tactile_gate_policy.config import PolicyConfig
from tactile_gate_policy.policy import LossConfig, TactileGatePolicy, sample_random_gate


def tiny_policy_config():
    return PolicyConfig(
        obs_horizon=2, action_horizon=4, n_action_steps=2,
        token_dim=32, condition_dim=32, num_diffusion_steps=4, num_inference_steps=2,
        router_layers=1, condition_layers=1, router_heads=4, condition_heads=4,
        dropout=0.1, dp_unet_diffusion_step_embed_dim=32, dp_unet_down_dims=(16, 32),
        image_size=(32, 32), crop_size=(32, 32), share_vision_backbone=True,
        tactile_hidden_dims=(16, 32),
    )


def synthetic_batch(cfg, batch_size=2):
    shape = (batch_size, cfg.obs_horizon)
    obs = {
        "global_rgb": torch.rand(*shape, 3, *cfg.image_size),
        "wrist_rgb": torch.rand(*shape, 3, *cfg.image_size),
        "left_tactile": torch.randint(0, 256, (*shape, cfg.tactile_dim)).float(),
        "right_tactile": torch.randint(0, 256, (*shape, cfg.tactile_dim)).float(),
        "proprio": torch.randn(*shape, cfg.proprio_dim),
    }
    action = torch.randn(batch_size, cfg.action_horizon, cfg.action_dim)
    action[..., 3:7] = torch.nn.functional.normalize(action[..., 3:7], dim=-1)
    return {"obs": obs, "action": action}


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.cfg = tiny_policy_config()
        self.policy = TactileGatePolicy(self.cfg)
        self.batch = synthetic_batch(self.cfg)
        samples = [
            {"obs": {key: value[i] for key, value in self.batch["obs"].items()},
             "action": self.batch["action"][i]}
            for i in range(self.batch["action"].shape[0])
        ]
        self.policy.normalizer.fit(samples)

    def test_stage_a_skips_router_but_trains_encoders_and_action_model(self):
        self.policy.train()
        result = self.policy.compute_loss(self.batch, stage="a")
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertTrue(result["loss"].requires_grad)
        self.assertFalse(result["gate"].requires_grad)
        self.assertTrue(torch.all((result["gate"] == 0) | (result["gate"] == 1)))
        result["loss"].backward()
        self.assertTrue(all(parameter.grad is None for parameter in self.policy.model.router.parameters()))
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in self.policy.encoder.parameters()))
        for module in (self.policy.model.condition_backbone, self.policy.model.action_head):
            self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                for parameter in module.parameters()))

    def test_stage_b_trains_router_through_action_and_gate_losses(self):
        self.policy.train()
        result = self.policy.compute_loss(self.batch, stage="b")
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertTrue(torch.all((result["gate"] > 0) & (result["gate"] < 1)))
        result["loss"].backward()
        gradients = [parameter.grad for parameter in self.policy.model.router.parameters()]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)

    def test_prediction_is_deterministic_normalizes_pose_and_preserves_inputs(self):
        original = {key: value.clone() for key, value in self.batch["obs"].items()}
        self.policy.train()
        self.policy.encoder.eval()
        modes = {name: module.training for name, module in self.policy.named_modules()}
        first = self.policy.predict_action(self.batch["obs"], generator=torch.Generator().manual_seed(123))
        second = self.policy.predict_action(self.batch["obs"], generator=torch.Generator().manual_seed(123))
        self.assertEqual(first["action"].shape, (2, self.cfg.n_action_steps, self.cfg.action_dim))
        self.assertEqual(first["action_pred"].shape, (2, self.cfg.action_horizon, self.cfg.action_dim))
        self.assertTrue(torch.equal(first["action_pred"], second["action_pred"]))
        self.assertTrue(torch.equal(first["gate"], second["gate"]))
        self.assertTrue(torch.isfinite(first["action_pred"]).all())
        norm = torch.linalg.vector_norm(first["action_pred"][..., 3:7], dim=-1)
        self.assertTrue(torch.allclose(norm, torch.ones_like(norm), atol=1e-6))
        self.assertEqual(modes, {name: module.training for name, module in self.policy.named_modules()})
        for key, value in self.batch["obs"].items():
            self.assertTrue(torch.equal(value, original[key]))

    def test_predict_uint8_rgb_matches_float_rgb(self):
        obs = dict(self.batch["obs"])
        for key in ("global_rgb", "wrist_rgb"):
            obs[key] = (obs[key] * 255).to(torch.uint8)
        floats = {key: value.float() / 255 if value.dtype == torch.uint8 else value
                  for key, value in obs.items()}
        first = self.policy.predict_action(obs, generator=torch.Generator().manual_seed(5))
        second = self.policy.predict_action(floats, generator=torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(first["action_pred"], second["action_pred"]))

    def test_near_zero_quaternion_uses_identity(self):
        with mock.patch.object(self.policy.normalizer, "denormalize_action", side_effect=torch.zeros_like):
            result = self.policy.predict_action(self.batch["obs"], generator=torch.Generator().manual_seed(9))
        identity = torch.zeros_like(result["action_pred"][..., 3:7])
        identity[..., 3] = 1
        self.assertTrue(torch.equal(result["action_pred"][..., 3:7], identity))

    def test_invalid_inputs_restore_mixed_modes(self):
        self.policy.train()
        self.policy.encoder.eval()
        modes = {name: module.training for name, module in self.policy.named_modules()}
        obs = dict(self.batch["obs"])
        obs["global_rgb"] = obs["global_rgb"] * 256
        with self.assertRaisesRegex(ValueError, "in \\[0,1\\]"):
            self.policy.predict_action(obs)
        self.assertEqual(modes, {name: module.training for name, module in self.policy.named_modules()})
        with self.assertRaises(ValueError):
            self.policy.compute_loss(self.batch, stage="unknown")
        with self.assertRaises(ValueError):
            self.policy.predict_action(self.batch["obs"], num_inference_steps=5)

    def test_random_gate_respects_actual_probability_and_fallback(self):
        for probability in (0.2, 0.8):
            gate = sample_random_gate(
                5000, 8, 5, min_keep_prob=probability, max_keep_prob=probability,
                generator=torch.Generator().manual_seed(19),
            )
            # Restoring exactly one element in all-zero rows adds (1-p)^M/M.
            expected = probability + (1 - probability) ** 5 / 5
            self.assertAlmostEqual(float(gate.mean()), expected, delta=0.01)
            self.assertTrue(torch.all(gate.sum(dim=-1) >= 1))
        with self.assertRaises(ValueError):
            LossConfig(min_keep_prob=0.9, max_keep_prob=0.1)


if __name__ == "__main__":
    unittest.main()
