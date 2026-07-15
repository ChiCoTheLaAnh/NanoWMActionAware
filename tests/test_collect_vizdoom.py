"""Tests for the Day 1 VizDoom collector and HDF5 schema."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "scripts" / "collect_vizdoom.py"
SPEC = importlib.util.spec_from_file_location("collect_vizdoom", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collector
SPEC.loader.exec_module(collector)


def synthetic_episode(seed: int = 42) -> collector.Episode:
    actions = np.asarray([0, 1, 2, 0], dtype=np.int64)
    return collector.Episode(
        frames=np.zeros((4, 120, 160, 3), dtype=np.uint8),
        actions=actions,
        action_onehot=np.eye(3, dtype=np.float32)[actions],
        rewards=np.asarray([-4.0, -4.0, 97.0, -4.0], dtype=np.float32),
        dones=np.asarray([False, False, False, True], dtype=np.bool_),
        seed=seed,
        success=True,
    )


class CollectorUnitTests(unittest.TestCase):
    def test_parse_resolution(self) -> None:
        self.assertEqual(collector.parse_resolution("160x120"), (160, 120))
        with self.assertRaises(Exception):
            collector.parse_resolution("160")
        with self.assertRaises(Exception):
            collector.parse_resolution("0x120")

    def test_uniform_actions_are_deterministic(self) -> None:
        first = collector.sample_uniform_actions(seed=42, length=100)
        second = collector.sample_uniform_actions(seed=42, length=100)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all((first >= 0) & (first <= 2)))

    def test_hdf5_round_trip_and_checksum(self) -> None:
        episode = synthetic_episode()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "episode_00000.hdf5"
            collector.write_episode(path, episode, (160, 120), 4, "test")
            loaded = collector.read_and_validate_episode(path)
            self.assertEqual(loaded.seed, 42)
            self.assertTrue(loaded.success)
            self.assertTrue(np.array_equal(loaded.actions, episode.actions))
            self.assertEqual(len(collector.sha256_file(path)), 64)

    def test_mismatched_onehot_is_rejected(self) -> None:
        episode = synthetic_episode()
        episode.action_onehot[0] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "action_onehot"):
            collector.validate_episode(episode, (160, 120))

    def test_nonterminal_final_transition_is_rejected(self) -> None:
        episode = synthetic_episode()
        episode.dones[-1] = False
        with self.assertRaisesRegex(ValueError, "final transition"):
            collector.validate_episode(episode, (160, 120))


@unittest.skipUnless(
    os.environ.get("NANOWM_RUN_VIZDOOM_INTEGRATION") == "1",
    "set NANOWM_RUN_VIZDOOM_INTEGRATION=1 to run VizDoom integration",
)
class VizDoomIntegrationTests(unittest.TestCase):
    def test_same_seed_replays_frames_and_actions(self) -> None:
        first, _ = collector.collect_episode(42, 4, (160, 120))
        second, _ = collector.collect_episode(42, 4, (160, 120))
        self.assertTrue(np.array_equal(first.actions, second.actions))
        self.assertTrue(np.array_equal(first.frames, second.frames))


if __name__ == "__main__":
    unittest.main()
