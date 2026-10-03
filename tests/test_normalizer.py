import io
import unittest

import torch

from tactile_gate_policy.normalizer import NumericNormalizer


class NumericSamples:
    def __init__(self, samples):
        self.samples = samples
        self.calls = 0

    def __len__(self):
        return len(self.samples)

    def get_numeric(self, index):
        self.calls += 1
        return self.samples[index]

    def __getitem__(self, index):
        raise AssertionError("Normalizer must not request images")


class NormalizerTests(unittest.TestCase):
    def samples(self):
        return [{"obs": {"proprio": torch.arange(44, dtype=torch.float32).reshape(2, 22) + i},
                 "action": torch.arange(24, dtype=torch.float32).reshape(3, 8) + i}
                for i in (0, 5, 10)]

    def test_streaming_stats_match_training_windows_without_images(self):
        samples = self.samples()
        dataset = NumericSamples(samples)
        norm = NumericNormalizer().fit(dataset)
        self.assertTrue(bool(norm.fitted))
        self.assertEqual(dataset.calls, 3)
        actions = torch.cat([x["action"] for x in samples])
        proprio = torch.cat([x["obs"]["proprio"] for x in samples])
        torch.testing.assert_close(norm.action_mean, actions.mean(0))
        torch.testing.assert_close(norm.action_std, actions.std(0, correction=0))
        torch.testing.assert_close(norm.proprio_mean, proprio.mean(0))
        torch.testing.assert_close(norm.proprio_std, proprio.std(0, correction=0))
        limited = NumericNormalizer().fit(dataset, max_windows=1)
        torch.testing.assert_close(limited.action_mean, samples[0]["action"].mean(0))

    def test_normalization_is_fresh_and_roundtrips(self):
        norm = NumericNormalizer().fit(self.samples())
        action = self.samples()[1]["action"]
        original = action.clone()
        torch.testing.assert_close(norm.denormalize_action(norm.normalize_action(action)), action)
        torch.testing.assert_close(action, original)
        obs = {"proprio": self.samples()[1]["obs"]["proprio"],
               "left_tactile": torch.full((2, 231), 255.), "right_tactile": torch.full((2, 231), 127.5),
               "global_rgb": torch.ones(2, 3, 8, 8)}
        originals = {k: x.clone() for k, x in obs.items()}
        normalized = norm.normalize_obs(obs)
        self.assertIsNot(normalized, obs)
        self.assertTrue(torch.all(normalized["left_tactile"] == 1))
        self.assertTrue(torch.all(normalized["right_tactile"] == 0.5))
        for k in obs:
            torch.testing.assert_close(obs[k], originals[k])

    def test_constant_stats_checkpoint_and_unfitted_errors(self):
        norm = NumericNormalizer()
        with self.assertRaisesRegex(RuntimeError, "training dataset"):
            norm.normalize_action(torch.zeros(3, 8))
        samples = [{"obs": {"proprio": torch.ones(2, 22)}, "action": torch.ones(3, 8)}]
        norm.fit(samples)
        self.assertTrue(torch.all(norm.action_std >= 1e-6))
        self.assertTrue(torch.all(norm.proprio_std >= 1e-6))
        stream = io.BytesIO()
        torch.save(norm.state_dict(), stream)
        stream.seek(0)
        restored = NumericNormalizer()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        self.assertTrue(bool(restored.fitted))
        torch.testing.assert_close(restored.normalize_action(samples[0]["action"]), torch.zeros(3, 8))
        with self.assertRaises(ValueError):
            restored.fit([])
        with self.assertRaises(ValueError):
            restored.fit(samples, max_windows=0)
        samples[0]["action"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            restored.fit(samples)


if __name__ == "__main__":
    unittest.main()
