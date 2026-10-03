"""Behavioral checks for actual pixel and CNN touch encoding (no downloads)."""

import unittest

import torch
from torch import nn

from tactile_gate_policy.config import PolicyConfig
from tactile_gate_policy.encoders import DPImageEncoder, MODALITY_KEYS, MultimodalObsEncoder, TactileCNNEncoder


def small_config(**changes):
    values = dict(obs_horizon=2, token_dim=32, condition_dim=32, image_size=(40, 40), crop_size=(32, 32), dropout=0.0, vision_weights=None, tactile_hidden_dims=(16, 32))
    values.update(changes)
    return PolicyConfig(**values)


def observations(cfg):
    return {
        "global_rgb": torch.rand(1, cfg.obs_horizon, 3, *cfg.image_size),
        "wrist_rgb": torch.rand(1, cfg.obs_horizon, 3, *cfg.image_size),
        "left_tactile": torch.rand(1, cfg.obs_horizon, cfg.tactile_dim),
        "right_tactile": torch.rand(1, cfg.obs_horizon, cfg.tactile_dim),
        "proprio": torch.randn(1, cfg.obs_horizon, cfg.proprio_dim),
    }


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)

    def test_real_pixels_change_only_their_modality_and_eval_repeats(self):
        cfg = small_config()
        model = MultimodalObsEncoder(cfg).eval()
        obs = observations(cfg)
        with torch.no_grad():
            first = model(obs)
            repeated = model(obs)
            changed_obs = dict(obs)
            changed_obs["global_rgb"] = torch.zeros_like(obs["global_rgb"])
            changed = model(changed_obs)
        self.assertEqual(tuple(first.shape), (1, 2, 5, 32))
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        self.assertGreater((first[:, :, 0] - changed[:, :, 0]).abs().max().item(), 1e-4)
        torch.testing.assert_close(first[:, :, 1:], changed[:, :, 1:], rtol=0, atol=0)
        self.assertIsInstance(model.vision_encoders["global_rgb"].backbone.fc, nn.Identity)
        self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in model.modules()))

    def test_crop_is_centered_at_eval_and_random_per_training_frame(self):
        cfg = small_config(imagenet_norm=False)
        encoder = DPImageEncoder(cfg).eval()
        image = torch.arange(40 * 40, dtype=torch.float32).reshape(1, 1, 40, 40).expand(16, 3, -1, -1)
        center = encoder.preprocess(image)
        torch.testing.assert_close(center, image[:, :, 4:36, 4:36])
        encoder.train()
        crops = encoder.preprocess(image)
        self.assertEqual(tuple(crops.shape), (16, 3, 32, 32))
        self.assertGreater(torch.unique(crops[:, 0, 0, 0]).numel(), 1)
        self.assertGreater((crops - center).abs().max().item(), 0)

    def test_both_touch_cnns_receive_gradients_and_flat_layout_matches(self):
        cfg = small_config()
        model = MultimodalObsEncoder(cfg).eval()
        obs = observations(cfg)
        obs["left_tactile"].requires_grad_()
        obs["right_tactile"].requires_grad_()
        output = model(obs)
        # A weighted objective avoids the zero mean enforced by LayerNorm.
        weights = torch.arange(cfg.token_dim, dtype=output.dtype)
        (output[:, :, 2:4] * weights).sum().backward()
        for key in ("left_tactile", "right_tactile"):
            self.assertGreater(obs[key].grad.abs().sum().item(), 0)
            conv = next(m for m in model.tactile_encoders[key].modules() if isinstance(m, nn.Conv1d))
            self.assertGreater(conv.weight.grad.abs().sum().item(), 0)
        encoder = model.tactile_encoders["left_tactile"]
        flat = obs["left_tactile"].detach().flatten(0, 1)
        torch.testing.assert_close(encoder(flat), encoder(flat.reshape(-1, 77, 3)))
        self.assertIsNot(model.tactile_encoders["left_tactile"], model.tactile_encoders["right_tactile"])

    def test_sharing_and_single_channel_touch_are_supported(self):
        cfg = small_config(share_vision_backbone=True, share_tactile_backbone=True, tactile_channels=1)
        model = MultimodalObsEncoder(cfg).eval()
        self.assertIs(model.vision_encoders["global_rgb"].backbone, model.vision_encoders["wrist_rgb"].backbone)
        self.assertIsNot(model.vision_encoders["global_rgb"].projection, model.vision_encoders["wrist_rgb"].projection)
        self.assertIs(model.tactile_encoders["left_tactile"], model.tactile_encoders["right_tactile"])
        self.assertEqual(tuple(model(observations(cfg)).shape), (1, 2, len(MODALITY_KEYS), cfg.token_dim))
        touch = TactileCNNEncoder(cfg).eval()
        self.assertEqual(tuple(touch(torch.randn(2, 231, 1)).shape), (2, cfg.token_dim))

    def test_missing_keys_and_wrong_layouts_fail_explicitly(self):
        cfg = small_config(share_vision_backbone=True)
        model = MultimodalObsEncoder(cfg).eval()
        obs = observations(cfg)
        with self.assertRaises(KeyError):
            model({key: value for key, value in obs.items() if key != "wrist_rgb"})
        malformed = dict(obs)
        malformed["left_tactile"] = torch.zeros(1, 2, 230)
        with self.assertRaises(ValueError):
            model(malformed)
        malformed = dict(obs)
        malformed["global_rgb"] = (obs["global_rgb"] * 255).to(torch.uint8)
        with self.assertRaises(TypeError):
            model(malformed)


if __name__ == "__main__":
    unittest.main()
