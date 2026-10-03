import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from PIL import Image
from torch.utils.data import DataLoader

from tactile_gate_policy.dataset import Fr3PaxiniDataset, split_dataset
from tactile_gate_policy.demo_data import create_demo_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = create_demo_dataset(Path(self.temp.name), frames=12, image_size=(8, 10))
        self.parquet = self.root / "data/episodes/episode_000000.parquet"

    def dataset(self, **kwargs):
        return Fr3PaxiniDataset(self.root, obs_horizon=2, action_horizon=3, image_size=(8, 10), **kwargs)

    def edit(self, fn):
        df = pd.read_parquet(self.parquet)
        fn(df)
        df.to_parquet(self.parquet, index=False)

    def test_window_future_actions_and_stride(self):
        data = self.dataset()
        self.assertEqual(len(data), 8)
        item = data[0]
        torch.testing.assert_close(item["frame_index"], torch.tensor([0, 1]))
        torch.testing.assert_close(item["action_frame_index"], torch.tensor([2, 3, 4]))
        torch.testing.assert_close(data[-1]["action_frame_index"], torch.tensor([9, 10, 11]))
        self.assertEqual(tuple(item["obs"]["proprio"].shape), (2, 22))
        self.assertEqual(tuple(item["action"].shape), (3, 8))
        stride = self.dataset(frame_stride=2)
        self.assertEqual(len(stride), 4)
        torch.testing.assert_close(stride[0]["frame_index"], torch.tensor([0, 2]))
        torch.testing.assert_close(stride[0]["action_frame_index"], torch.tensor([4, 6, 8]))
        self.assertAlmostEqual(stride.control_dt, 0.2)
        with self.assertRaises(IndexError):
            data[8]

    def test_real_rgb_pixels_tactile_drop_and_collation(self):
        Image.new("RGB", (10, 8), (255, 64, 0)).save(self.root / "images/episode_000000/front_000000.png")
        array = zarr.open(str(self.root / "tactile/episode_000000/left.zarr"), mode="r+")
        array[0, :3] = 255
        array[0, 3:] = 17
        data = self.dataset()
        item = data[0]
        torch.testing.assert_close(item["obs"]["global_rgb"][0, :, 0, 0], torch.tensor([1., 64/255., 0.]))
        self.assertEqual(tuple(item["obs"]["left_tactile"].shape), (2, 231))
        self.assertTrue(torch.all(item["obs"]["left_tactile"][0] == 17))
        self.assertEqual(item["obs"]["global_rgb"].dtype, torch.float32)
        batch = next(iter(DataLoader(data, batch_size=2)))
        self.assertEqual(tuple(batch["obs"]["global_rgb"].shape), (2, 2, 3, 8, 10))
        self.assertEqual(tuple(batch["action"].shape), (2, 3, 8))

    def test_sync_filters_action_frames_as_well_as_obs(self):
        self.edit(lambda df: df.__setitem__("sync.valid", [True] * 4 + [False] + [True] * 7))
        data = self.dataset()
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0]["window_index"], 5)
        self.assertEqual(len(self.dataset(require_sync_valid=False)), 8)

    def test_unsafe_paths_and_nonfinite_numeric_fail(self):
        self.edit(lambda df: df.loc.__setitem__((0, "image.path"), "../../../outside.png"))
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.dataset()
        create_demo_dataset(self.root, frames=12, image_size=(8, 10), episode_index=1)
        other = self.root / "data/episodes/episode_000001.parquet"
        df = pd.read_parquet(other)
        df.at[0, "action.sent_action8"] = np.full(8, np.nan)
        df.to_parquet(other, index=False)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.dataset(episode_indices=[1])

    def test_timestamps_and_metadata_validation(self):
        self.edit(lambda df: df.loc.__setitem__((2, "timestamp"), df.loc[1, "timestamp"]))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.dataset()
        self.edit(lambda df: df.__setitem__("timestamp", np.arange(12) * 0.1))
        self.edit(lambda df: df.loc.__setitem__((0, "tactile.left.distribution_raw_start"), 2))
        with self.assertRaisesRegex(ValueError, "must equal 3"):
            self.dataset()

    def test_numeric_reads_no_images_and_worker_pickle_drops_zarr(self):
        data = self.dataset()
        numeric = data.get_numeric(0)
        self.assertEqual(set(numeric["obs"]), {"proprio"})
        self.assertFalse(data._arrays)
        data[0]
        self.assertTrue(data._arrays)
        restored = pickle.loads(pickle.dumps(data))
        self.assertFalse(restored._arrays)
        torch.testing.assert_close(restored[0]["action"], data[0]["action"])

    def test_split_by_episode_and_disjoint_time_blocks(self):
        create_demo_dataset(self.root, frames=12, image_size=(8, 10), episode_index=1)
        train, val = split_dataset(self.root, val_fraction=0.5, obs_horizon=2, action_horizon=3, image_size=(8, 10))
        self.assertFalse(set(train.episode_indices) & set(val.episode_indices))
        train, val = split_dataset(self.root, val_fraction=0.5, episode_indices=[0], obs_horizon=2, action_horizon=3)
        train_rows, val_rows = set(), set()
        for data, rows in ((train, train_rows), (val, val_rows)):
            for i in range(len(data)):
                _, _, obs, actions = data._indices(i)
                rows.update(obs.tolist() + actions.tolist())
        self.assertFalse(train_rows & val_rows)
        self.assertEqual(train.split_manifest["episodes"]["0"]["segment"], [0, 6])
        self.assertEqual(val.split_manifest["episodes"]["0"]["segment"], [6, 12])
        with self.assertRaisesRegex(ValueError, "No valid windows"):
            split_dataset(self.root, val_fraction=0.2, episode_indices=[0], obs_horizon=2, action_horizon=3)


if __name__ == "__main__":
    unittest.main()
