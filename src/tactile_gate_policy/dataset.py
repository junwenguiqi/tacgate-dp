"""Leakage-safe windows of synchronized FR3 RGB, Paxini, and robot data."""

from pathlib import Path, PureWindowsPath
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .images import prepare_rgb


PROPRIO_COLUMNS = ("robot.ee_pose", "robot.q", "robot.dq", "robot.gripper_pos")
PROPRIO_DIMS = (7, 7, 7, 1)


class Fr3PaxiniDataset(Dataset):
    """Read real images and future actions from FR3 Parquet + Zarr episodes.

    ``image_size`` is (height, width). ``segments`` maps episode IDs to half-open
    row intervals. A window's observations and future actions must fit entirely
    in its interval. ``frame_stride`` spaces both observation and action frames;
    window starts advance one raw frame, allowing all eligible examples.
    """

    def __init__(self, dataset_root, episode_indices=None, obs_horizon=4,
                 action_horizon=8, image_size=(96, 96), frame_stride=1,
                 require_sync_valid=True, segments=None):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.obs_horizon = self._positive_int(obs_horizon, "obs_horizon")
        self.action_horizon = self._positive_int(action_horizon, "action_horizon")
        self.frame_stride = self._positive_int(frame_stride, "frame_stride")
        if len(image_size) != 2:
            raise ValueError("image_size must be (height, width)")
        self.image_size = tuple(self._positive_int(x, "image_size") for x in image_size)
        self.require_sync_valid = bool(require_sync_valid)
        self._arrays = {}
        self._episodes = {}
        self._numeric = {}
        self._windows = []
        self._segments = {}
        if episode_indices is None:
            episode_indices = self.discover_episodes(self.dataset_root)
        episode_indices = [self._episode_id(i) for i in episode_indices]
        if not episode_indices or len(set(episode_indices)) != len(episode_indices):
            raise ValueError("episode_indices must contain distinct available episode IDs")
        self.episode_indices = tuple(sorted(episode_indices))
        segment_map = {self._episode_id(k): v for k, v in (segments or {}).items()}
        if set(segment_map) - set(self.episode_indices):
            raise ValueError("segments contains an episode outside episode_indices")
        dts = []
        offsets = np.arange(self.obs_horizon + self.action_horizon) * self.frame_stride
        for episode in self.episode_indices:
            parquet = self._safe_path(f"data/episodes/episode_{episode:06d}.parquet")
            if not parquet.is_file():
                raise FileNotFoundError(f"episode Parquet not found: {parquet}")
            df = pd.read_parquet(parquet).reset_index(drop=True)
            self._validate_columns(df, episode)
            times = df["timestamp"].to_numpy(dtype=np.float64)
            if len(times) < 2 or not np.isfinite(times).all():
                raise ValueError(f"episode {episode}: at least two finite timestamps are required")
            delta = np.diff(times)
            if np.any(delta <= 0):
                raise ValueError(f"episode {episode}: timestamps must be strictly increasing")
            dt = float(np.median(delta))
            if not np.allclose(delta, dt, rtol=0.1, atol=0.001):
                raise ValueError(f"episode {episode}: inconsistent timestamp intervals; median={dt:.6g}s")
            dts.append(dt)
            frame_indices = df["frame_index"].to_numpy(dtype=np.int64)
            if np.any(np.diff(frame_indices) <= 0):
                raise ValueError(f"episode {episode}: frame_index must increase strictly")
            action = self._stack(df["action.sent_action8"], 8, "action.sent_action8")
            proprio = np.concatenate([
                self._stack(df[column], dim, column)
                for column, dim in zip(PROPRIO_COLUMNS, PROPRIO_DIMS)
            ], axis=-1)
            for column in ("image.path", "wrist_image.path", "tactile.left.zarr_path", "tactile.right.zarr_path"):
                for rel in df[column].unique():
                    self._safe_path(rel)
            for side in ("left", "right"):
                for suffix, expected in (("force_raw_len", 3), ("distribution_raw_start", 3), ("distribution_raw_len", 231)):
                    if not (df[f"tactile.{side}.{suffix}"] == expected).all():
                        raise ValueError(f"episode {episode}: tactile.{side}.{suffix} must equal {expected}")
                zi = df[f"tactile.{side}.zarr_index"].to_numpy()
                if not np.isfinite(zi).all() or np.any(zi < 0) or np.any(zi != np.floor(zi)):
                    raise ValueError(f"episode {episode}: invalid tactile.{side}.zarr_index")
            start, end = segment_map.get(episode, (0, len(df)))
            if (isinstance(start, bool) or isinstance(end, bool) or int(start) != start
                    or int(end) != end or not 0 <= start < end <= len(df)):
                raise ValueError(f"episode {episode}: segment must be 0 <= start < end <= {len(df)}")
            start, end = int(start), int(end)
            self._segments[episode] = (start, end)
            self._episodes[episode] = df
            self._numeric[episode] = (proprio, action)
            valid = df["sync.valid"].fillna(False).to_numpy()
            if not all(isinstance(x, (bool, np.bool_)) for x in valid):
                raise ValueError(f"episode {episode}: sync.valid must contain booleans")
            span = int(offsets[-1]) + 1
            for window in range(start, end - span + 1):
                if not self.require_sync_valid or valid[window + offsets].all():
                    self._windows.append((episode, window))
        if not np.allclose(dts, np.median(dts), rtol=0.02, atol=0.001):
            raise ValueError("Selected episodes have incompatible timestamp intervals")
        self.observed_dt = float(np.median(dts))
        self.control_dt = self.observed_dt * self.frame_stride
        self.inferred_control_dt = self.control_dt
        if not self._windows:
            raise ValueError("No valid windows: segments must be long enough for both horizons and frame_stride, and sync.valid must permit a complete window")

    @staticmethod
    def _positive_int(value, name):
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @staticmethod
    def _episode_id(value):
        if isinstance(value, str):
            if not value.isdecimal():
                raise ValueError("episode IDs must be nonnegative integers")
            return int(value)
        if isinstance(value, bool) or int(value) != value or value < 0:
            raise ValueError("episode IDs must be nonnegative integers")
        return int(value)

    @classmethod
    def discover_episodes(cls, dataset_root):
        root = Path(dataset_root).expanduser().resolve()
        directory = root / "data" / "episodes"
        if not directory.is_dir():
            raise FileNotFoundError(f"episode directory not found: {directory}")
        ids = sorted(int(match.group(1)) for path in directory.glob("episode_*.parquet")
                     if (match := re.fullmatch(r"episode_(\d+)\.parquet", path.name))
                     and path.name == f"episode_{int(match.group(1)):06d}.parquet")
        if not ids:
            raise ValueError(f"No episode Parquet files in {directory}")
        return ids

    def _safe_path(self, relative):
        if not isinstance(relative, (str, Path)) or not str(relative):
            raise ValueError("Dataset paths must be nonempty relative paths")
        windows_path = PureWindowsPath(str(relative))
        relative = Path(str(relative).replace("\\", "/"))
        if relative.is_absolute() or relative.drive or windows_path.drive:
            raise ValueError(f"Dataset path must be relative: {relative}")
        path = (self.dataset_root / relative).resolve()
        try:
            path.relative_to(self.dataset_root)
        except ValueError as exc:
            raise ValueError(f"Dataset path escapes dataset_root: {relative}") from exc
        return path

    @staticmethod
    def _validate_columns(df, episode):
        required = ["frame_index", "timestamp", "image.path", "wrist_image.path",
                    "action.sent_action8", "sync.valid", *PROPRIO_COLUMNS]
        for side in ("left", "right"):
            required.extend(f"tactile.{side}.{suffix}" for suffix in
                            ("zarr_path", "zarr_index", "force_raw_len", "distribution_raw_start", "distribution_raw_len"))
        missing = [name for name in required if name not in df]
        if missing:
            raise KeyError(f"episode {episode} missing columns: {missing}")

    @staticmethod
    def _stack(values, expected_dim, name):
        try:
            result = np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in values])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric vectors of length {expected_dim}") from exc
        if result.shape != (len(values), expected_dim) or not np.isfinite(result).all():
            raise ValueError(f"{name} must contain finite vectors of length {expected_dim}, got {result.shape}")
        return result

    @property
    def split_manifest(self):
        return {"episodes": {str(ep): {"segment": list(self._segments[ep]),
                                     "num_windows": sum(e == ep for e, _ in self._windows)}
                             for ep in self.episode_indices},
                "obs_horizon": self.obs_horizon, "action_horizon": self.action_horizon,
                "frame_stride": self.frame_stride, "num_windows": len(self),
                "require_sync_valid": self.require_sync_valid, "control_dt": self.control_dt}

    def __len__(self):
        return len(self._windows)

    def _indices(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode, start = self._windows[index]
        obs_indices = start + np.arange(self.obs_horizon) * self.frame_stride
        action_indices = start + (self.obs_horizon + np.arange(self.action_horizon)) * self.frame_stride
        return episode, start, obs_indices, action_indices

    def get_numeric(self, index):
        """Return only proprioception and actions; never open images or Zarr."""
        episode, _, obs_indices, action_indices = self._indices(index)
        proprio, action = self._numeric[episode]
        return {"obs": {"proprio": torch.from_numpy(proprio[obs_indices].copy())},
                "action": torch.from_numpy(action[action_indices].copy())}

    def _images(self, paths):
        images = []
        for relative in paths:
            with Image.open(self._safe_path(relative)) as image:
                tensor = prepare_rgb(image.convert("RGB"), self.image_size)
            images.append(tensor)
        return torch.stack(images)

    def _tactile(self, df, side):
        import zarr

        result = np.empty((len(df), 231), dtype=np.float32)
        paths = df[f"tactile.{side}.zarr_path"].to_numpy()
        indices = df[f"tactile.{side}.zarr_index"].to_numpy(dtype=np.int64)
        for relative in dict.fromkeys(paths.tolist()):
            path = str(self._safe_path(relative))
            if path not in self._arrays:
                self._arrays[path] = zarr.open(path, mode="r")
            array = self._arrays[path]
            if len(array.shape) != 2 or array.shape[1] != 234:
                raise ValueError(f"Expected tactile array (N,234): {path}, got {array.shape}")
            selected = np.flatnonzero(paths == relative)
            selected_indices = indices[selected]
            if np.any(selected_indices >= array.shape[0]):
                raise ValueError(f"Tactile Zarr index out of bounds: {path}")
            result[selected] = np.asarray(array.oindex[selected_indices, slice(3, None)], dtype=np.float32)
        if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 255):
            raise ValueError("Tactile distribution must contain finite raw values in [0,255]")
        return torch.from_numpy(result)

    def __getitem__(self, index):
        episode, start, obs_indices, action_indices = self._indices(index)
        df = self._episodes[episode]
        obs_df = df.iloc[obs_indices]
        result = self.get_numeric(index)
        result["obs"].update({
            "global_rgb": self._images(obs_df["image.path"]),
            "wrist_rgb": self._images(obs_df["wrist_image.path"]),
            "left_tactile": self._tactile(obs_df, "left"),
            "right_tactile": self._tactile(obs_df, "right"),
        })
        result.update({"episode_index": episode, "window_index": start,
                       "frame_index": torch.from_numpy(df.iloc[obs_indices]["frame_index"].to_numpy(dtype=np.int64)),
                       "action_frame_index": torch.from_numpy(df.iloc[action_indices]["frame_index"].to_numpy(dtype=np.int64))})
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state


# Keep the descriptive name available to callers inspecting the original format.
Fr3PaxiniRawWindowDataset = Fr3PaxiniDataset


def split_dataset(dataset_root, val_fraction=0.2, seed=0, **dataset_kwargs):
    """Split whole episodes, or disjoint time blocks for a single episode.

    A split too short for a complete valid window raises ValueError. No random
    window split is used, since overlapping observations/actions would leak.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    kwargs = dict(dataset_kwargs)
    if kwargs.get("segments") is not None:
        raise ValueError("split_dataset creates segments; do not supply segments")
    kwargs.pop("segments", None)
    ids = kwargs.pop("episode_indices", None)
    if ids is None:
        ids = Fr3PaxiniDataset.discover_episodes(dataset_root)
    ids = [Fr3PaxiniDataset._episode_id(i) for i in ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("episode_indices must contain distinct available IDs")
    if len(ids) > 1:
        shuffled = np.random.default_rng(seed).permutation(sorted(ids)).tolist()
        n_val = min(len(ids) - 1, max(1, int(round(len(ids) * val_fraction))))
        train_ids, val_ids = shuffled[n_val:], shuffled[:n_val]
        train = Fr3PaxiniDataset(dataset_root, episode_indices=train_ids, **kwargs)
        val = Fr3PaxiniDataset(dataset_root, episode_indices=val_ids, **kwargs)
    else:
        episode = ids[0]
        root = Path(dataset_root).expanduser().resolve()
        parquet = root / "data" / "episodes" / f"episode_{episode:06d}.parquet"
        try:
            parquet.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError("Episode Parquet escapes dataset_root") from exc
        n_frames = len(pd.read_parquet(parquet, columns=["frame_index"]))
        boundary = int(n_frames * (1 - val_fraction))
        train = Fr3PaxiniDataset(dataset_root, episode_indices=ids,
                                segments={episode: (0, boundary)}, **kwargs)
        val = Fr3PaxiniDataset(dataset_root, episode_indices=ids,
                              segments={episode: (boundary, n_frames)}, **kwargs)
    if not np.isclose(train.control_dt, val.control_dt, rtol=0.02, atol=0.001):
        raise ValueError("Train and validation episodes have incompatible control intervals")
    return train, val
