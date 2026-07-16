"""Tests for the VizDoom DataSource and four-frame loader contract."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

COLLECTOR_PATH = SRC / "scripts" / "collect_vizdoom.py"
SPEC = importlib.util.spec_from_file_location("test_vizdoom_collector", COLLECTOR_PATH)
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collector
SPEC.loader.exec_module(collector)

from wm_datasets.data_source.factory import create_data_source  # noqa: E402
from wm_datasets.data_source.game import VizDoomDataSource  # noqa: E402
from wm_datasets.world_model_dataset import WorldModelDataset  # noqa: E402


def make_episode(seed: int, length: int) -> collector.Episode:
    actions = np.arange(length, dtype=np.int64) % 3
    frames = np.zeros((length, 12, 16, 3), dtype=np.uint8)
    for index in range(length):
        frames[index, ..., 0] = index * 10
        frames[index, ..., 1] = seed
        frames[index, ..., 2] = 200
    dones = np.zeros(length, dtype=np.bool_)
    dones[-1] = True
    return collector.Episode(
        frames=frames,
        actions=actions,
        action_onehot=np.eye(3, dtype=np.float32)[actions],
        rewards=np.arange(length, dtype=np.float32),
        dones=dones,
        seed=seed,
        success=True,
    )


def write_episode(directory: Path, index: int, seed: int, length: int) -> Path:
    path = directory / f"episode_{index:05d}.hdf5"
    collector.write_episode(path, make_episode(seed, length), (16, 12), 4, "test")
    return path


class VizDoomDataSourceTests(unittest.TestCase):
    def test_loads_rgb_actions_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            write_episode(directory, 1, 43, 5)
            write_episode(directory, 0, 42, 6)

            source = VizDoomDataSource(str(directory))
            self.assertEqual(source.get_num_trajectories(), 2)
            self.assertEqual(source.get_seq_length(0), 6)
            self.assertEqual(source.action_dim, 3)
            self.assertEqual(source.state_dim, 0)

            trajectory = source.load_trajectory(0)
            self.assertEqual(tuple(trajectory.actions.shape), (6, 3))
            self.assertEqual(tuple(trajectory.states.shape), (6, 0))
            self.assertEqual(trajectory.meta["episode_id"], "episode_00000.hdf5")
            self.assertEqual(trajectory.meta["seed"], 42)

            frames = source.load_visual_frames(0, 0, 2)
            self.assertEqual(tuple(frames.shape), (2, 3, 12, 16))
            self.assertAlmostEqual(float(frames[0, 0, 0, 0]), 0.0)
            self.assertAlmostEqual(float(frames[0, 1, 0, 0]), 42 / 255.0)
            self.assertAlmostEqual(float(frames[0, 2, 0, 0]), 200 / 255.0)

    def test_factory_and_n_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            write_episode(directory, 0, 42, 5)
            write_episode(directory, 1, 43, 5)
            source = create_data_source(
                dataset_name="vizdoom_basic", data_path=str(directory), n_rollout=1
            )
            self.assertIsInstance(source, VizDoomDataSource)
            self.assertEqual(source.get_num_trajectories(), 1)

    def test_four_frame_dataset_skips_short_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            write_episode(directory, 0, 42, 5)
            write_episode(directory, 1, 43, 3)
            source = VizDoomDataSource(str(directory))
            dataset = WorldModelDataset(
                data_source=source,
                num_frames=4,
                image_size=(32, 32),
                split="train",
                split_ratio=1.0,
                normalize_action=False,
                slice_mode="exhaustive",
                resize_mode="pad",
            )
            self.assertEqual(len(dataset), 2)
            item = dataset[0]
            self.assertEqual(tuple(item["video"].shape), (4, 3, 32, 32))
            self.assertEqual(tuple(item["action"].shape), (4, 3))
            self.assertGreaterEqual(float(item["video"].min()), -1.0)
            self.assertLessEqual(float(item["video"].max()), 1.0)

    def test_missing_dataset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_episode(Path(temp_dir), 0, 42, 5)
            with h5py.File(path, "a") as handle:
                del handle["rewards"]
            with self.assertRaisesRegex(ValueError, "missing datasets"):
                VizDoomDataSource(temp_dir)

    def test_wrong_frame_dtype_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_episode(Path(temp_dir), 0, 42, 5)
            with h5py.File(path, "a") as handle:
                frames = handle["frames"][:].astype(np.float32)
                del handle["frames"]
                handle.create_dataset("frames", data=frames)
            with self.assertRaisesRegex(ValueError, "frames must be uint8"):
                VizDoomDataSource(temp_dir)

    def test_corrupt_onehot_and_terminal_flags_are_rejected_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_episode(Path(temp_dir), 0, 42, 5)
            source = VizDoomDataSource(temp_dir)
            with h5py.File(path, "a") as handle:
                handle["action_onehot"][0] = np.asarray([0.0, 1.0, 0.0])
            with self.assertRaisesRegex(ValueError, "action_onehot"):
                source.load_trajectory(0)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_episode(Path(temp_dir), 0, 42, 5)
            source = VizDoomDataSource(temp_dir)
            with h5py.File(path, "a") as handle:
                handle["dones"][1] = True
            with self.assertRaisesRegex(ValueError, "terminal flags"):
                source.load_trajectory(0)

    def test_unknown_factory_dataset_still_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            create_data_source(dataset_name="not_a_dataset", data_path="unused")

    def test_hydra_vizdoom_config_composes(self) -> None:
        config_dir = str((SRC / "configs").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(
                config_name="config",
                overrides=["dataset=game/vizdoom_basic", "model=nanowm_s2"],
            )
        dataset = OmegaConf.to_container(config.dataset, resolve=True)
        self.assertEqual(dataset["name"], "vizdoom_basic")
        self.assertEqual(dataset["spec"]["action_dim"], 3)
        self.assertEqual(dataset["frame_interval"], 1)
        self.assertEqual(dataset["loader"]["resize_mode"], "pad")


if __name__ == "__main__":
    unittest.main()
