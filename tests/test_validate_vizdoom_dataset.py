"""Tests for the Day 2 VizDoom validation and evidence helpers."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = SRC / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

VALIDATOR_PATH = SCRIPTS / "validate_vizdoom_dataset.py"
SPEC = importlib.util.spec_from_file_location("vizdoom_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def make_episode(seed: int, actions: np.ndarray) -> validator.collector.Episode:
    length = len(actions)
    frames = np.zeros((length, 12, 16, 3), dtype=np.uint8)
    frames[..., 0] = seed
    dones = np.zeros(length, dtype=np.bool_)
    dones[-1] = True
    return validator.collector.Episode(
        frames=frames,
        actions=actions.astype(np.int64),
        action_onehot=np.eye(3, dtype=np.float32)[actions],
        rewards=np.zeros(length, dtype=np.float32),
        dones=dones,
        seed=seed,
        success=True,
    )


def build_dataset(directory: Path) -> None:
    records = []
    for index, seed in enumerate((42, 43)):
        actions = np.asarray([0, 1, 2, index, 1], dtype=np.int64)
        episode = make_episode(seed, actions)
        path = directory / f"episode_{index:05d}.hdf5"
        validator.collector.write_episode(path, episode, (16, 12), 4, "test")
        records.append(
            {
                "file": path.name,
                "seed": seed,
                "length": len(actions),
                "return": 0.0,
                "success": True,
                "sha256": validator.collector.sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "episodes": 2,
        "frame_skip": 4,
        "resolution": "16x12",
        "action_names": ["left", "right", "shoot"],
        "files": records,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


class ValidatorUnitTests(unittest.TestCase):
    def test_validates_manifest_schema_and_action_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            build_dataset(directory)
            report, episodes = validator.validate_dataset(directory)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["episodes"], 2)
            self.assertEqual(report["total_transitions"], 10)
            self.assertEqual(len(episodes), 2)
            self.assertTrue(report["manifest_checksums_valid"])
            self.assertTrue(all(report["action_counts"].values()))

    def test_checksum_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            build_dataset(directory)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"][0]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                validator.validate_dataset(directory)

    def test_plots_and_loader_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            evidence = directory / "evidence"
            build_dataset(directory)
            report, episodes = validator.validate_dataset(directory)
            validator.plot_action_distribution(report, evidence / "actions.png")
            validator.plot_episode_visualization(episodes, evidence / "episodes.png")
            batch = validator.validate_loader(directory, evidence / "batch.png")
            self.assertTrue((evidence / "actions.png").is_file())
            self.assertTrue((evidence / "episodes.png").is_file())
            self.assertTrue((evidence / "batch.png").is_file())
            self.assertEqual(batch["video_shape"], [2, 4, 3, 256, 256])
            self.assertEqual(batch["action_shape"], [2, 4, 3])


@unittest.skipUnless(
    os.environ.get("NANOWM_RUN_VIZDOOM_INTEGRATION") == "1",
    "set NANOWM_RUN_VIZDOOM_INTEGRATION=1 to run VizDoom replay validation",
)
class ValidatorIntegrationTests(unittest.TestCase):
    def test_local_pilot_replay_alignment(self) -> None:
        dataset_dir = ROOT / "data" / "vizdoom_basic" / "pilot"
        _, episodes = validator.validate_dataset(dataset_dir)
        report = validator.validate_alignment(dataset_dir, episodes)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["episodes_checked"], len(episodes))


if __name__ == "__main__":
    unittest.main()
