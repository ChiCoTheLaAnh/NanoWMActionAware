"""VizDoom Basic data source for action-aligned HDF5 episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch

from ..base import DataSource, TrajectoryData


SCHEMA_VERSION = 1
ACTION_NAMES = ("left", "right", "shoot")
REQUIRED_DATASETS = {
    "frames",
    "actions",
    "action_onehot",
    "rewards",
    "dones",
    "seed",
    "success",
}


class VizDoomDataSource(DataSource):
    """Read deterministic VizDoom Basic episodes collected by the project.

    Each stored frame is the observation immediately before the action at the
    same index. NanoWM shifts action embeddings internally, so ``action[t]``
    conditions the prediction of ``frame[t + 1]``.
    """

    def __init__(self, data_path: str, n_rollout: Optional[int] = None):
        self.data_path = Path(data_path)
        if not self.data_path.is_dir():
            raise FileNotFoundError(f"VizDoom data directory not found: {self.data_path}")

        files = sorted(self.data_path.glob("episode_*.hdf5"), key=self._episode_number)
        if n_rollout is not None:
            if n_rollout <= 0:
                raise ValueError("n_rollout must be positive when provided")
            files = files[:n_rollout]
        if not files:
            raise FileNotFoundError(
                f"No episode_*.hdf5 files found in {self.data_path}"
            )

        self._file_paths = files
        self._lengths = []
        self._metadata: Dict[int, Dict[str, object]] = {}
        self._action_cache: Dict[int, torch.Tensor] = {}

        seen_seeds = set()
        for index, path in enumerate(self._file_paths):
            length, seed, success = self._inspect_file(path)
            if seed in seen_seeds:
                raise ValueError(f"Duplicate VizDoom seed {seed} in {path}")
            seen_seeds.add(seed)
            self._lengths.append(length)
            self._metadata[index] = {
                "episode_id": path.name,
                "seed": seed,
                "success": success,
            }

    @staticmethod
    def _episode_number(path: Path) -> int:
        try:
            return int(path.stem.rsplit("_", maxsplit=1)[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid VizDoom episode filename: {path.name}") from exc

    @staticmethod
    def _inspect_file(path: Path) -> Tuple[int, int, bool]:
        with h5py.File(path, "r") as handle:
            missing = REQUIRED_DATASETS.difference(handle.keys())
            if missing:
                raise ValueError(f"{path} is missing datasets: {sorted(missing)}")
            if int(handle.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
                raise ValueError(f"{path} has an unsupported schema version")
            try:
                action_names = tuple(json.loads(str(handle.attrs["action_names"])))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path} has invalid action_names metadata") from exc
            if action_names != ACTION_NAMES:
                raise ValueError(
                    f"{path} action_names {action_names} do not match {ACTION_NAMES}"
                )

            frames = handle["frames"]
            actions = handle["actions"]
            onehot = handle["action_onehot"]
            rewards = handle["rewards"]
            dones = handle["dones"]
            length = int(actions.shape[0])
            if length <= 0:
                raise ValueError(f"{path} contains no transitions")
            if frames.ndim != 4 or frames.shape[0] != length or frames.shape[-1] != 3:
                raise ValueError(f"{path} has invalid frames shape {frames.shape}")
            if onehot.shape != (length, len(ACTION_NAMES)):
                raise ValueError(f"{path} has invalid action_onehot shape {onehot.shape}")
            if rewards.shape != (length,) or dones.shape != (length,):
                raise ValueError(f"{path} has inconsistent reward/done lengths")
            if frames.dtype != np.dtype(np.uint8):
                raise ValueError(f"{path} frames must be uint8, got {frames.dtype}")
            if actions.dtype != np.dtype(np.int64):
                raise ValueError(f"{path} actions must be int64, got {actions.dtype}")
            if onehot.dtype != np.dtype(np.float32):
                raise ValueError(f"{path} action_onehot must be float32, got {onehot.dtype}")
            if rewards.dtype != np.dtype(np.float32):
                raise ValueError(f"{path} rewards must be float32, got {rewards.dtype}")
            if dones.dtype.kind != "b":
                raise ValueError(f"{path} dones must be bool, got {dones.dtype}")

            return length, int(handle["seed"][()]), bool(handle["success"][()])

    def _check_index(self, index: int) -> None:
        if index < 0 or index >= len(self._file_paths):
            raise IndexError(f"Index {index} out of range [0, {len(self._file_paths)})")

    def load_trajectory(self, index: int) -> TrajectoryData:
        self._check_index(index)
        if index not in self._action_cache:
            path = self._file_paths[index]
            with h5py.File(path, "r") as handle:
                actions = handle["actions"][:]
                onehot = handle["action_onehot"][:]
                dones = handle["dones"][:].astype(np.bool_, copy=False)

            if np.any((actions < 0) | (actions >= len(ACTION_NAMES))):
                raise ValueError(f"{path} contains an out-of-range action ID")
            expected = np.eye(len(ACTION_NAMES), dtype=np.float32)[actions]
            if not np.array_equal(onehot, expected):
                raise ValueError(f"{path} action_onehot does not match actions")
            if not bool(dones[-1]) or np.any(dones[:-1]):
                raise ValueError(f"{path} has invalid terminal flags")
            self._action_cache[index] = torch.from_numpy(onehot).float()

        length = self._lengths[index]
        return TrajectoryData(
            states=torch.zeros(length, 0, dtype=torch.float32),
            actions=self._action_cache[index],
            seq_length=length,
            meta=dict(self._metadata[index]),
        )

    def load_visual_frames(
        self, index: int, start: int, end: int, step: int = 1
    ) -> torch.Tensor:
        self._check_index(index)
        length = self._lengths[index]
        if step <= 0:
            raise ValueError("step must be positive")
        if start < 0 or end > length or start >= end:
            raise ValueError(
                f"Invalid frame range [{start}, {end}) for episode length {length}"
            )

        with h5py.File(self._file_paths[index], "r") as handle:
            frames = handle["frames"][start:end:step]
        return torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0

    def get_num_trajectories(self) -> int:
        return len(self._file_paths)

    def get_seq_length(self, index: int) -> int:
        self._check_index(index)
        return self._lengths[index]

    @property
    def action_dim(self) -> int:
        return len(ACTION_NAMES)

    @property
    def state_dim(self) -> int:
        return 0
