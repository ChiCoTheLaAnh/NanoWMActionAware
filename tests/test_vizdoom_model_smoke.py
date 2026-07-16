"""Tests for Day 3 fixed-subset and evidence helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = SRC / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


smoke = load_module("vizdoom_model_smoke", SCRIPTS / "vizdoom_model_smoke.py")
baselines = load_module(
    "vizdoom_baselines", SRC / "evaluation" / "vizdoom_baselines.py"
)


class FakeDataset:
    def __init__(self, count: int = 64):
        self.available = [
            {"traj_idx": index // 8, "start_frame": index, "end_frame": index + 4}
            for index in range(count)
        ]
        self.selected = []

    def sample_fixed_slice_specs(self, size: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(self.available), generator=generator)[:size]
        return [self.available[int(index)] for index in indices]

    def set_fixed_slices(self, slices):
        self.selected = list(slices)


class VizDoomModelSmokeTests(unittest.TestCase):
    def test_smoke_and_dev_profiles_compose(self) -> None:
        with initialize_config_dir(config_dir=str(SRC / "configs"), version_base=None):
            smoke_cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=vizdoom_smoke",
                    "dataset=game/vizdoom_basic",
                    "model=nanowm_s2",
                ],
            )
            dev_cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=vizdoom_dev",
                    "dataset=game/vizdoom_basic",
                    "model=nanowm_s2",
                ],
            )
        self.assertEqual(smoke_cfg.experiment.training.max_steps, 1000)
        self.assertEqual(smoke_cfg.experiment.training.batch_size, 1)
        self.assertEqual(smoke_cfg.experiment.training.gradient_accumulation, 8)
        self.assertEqual(smoke_cfg.experiment.training.fixed_subset_size, 32)
        self.assertEqual(smoke_cfg.experiment.training.overfit_batches, 1.0)
        self.assertFalse(smoke_cfg.experiment.evaluation.metrics.evaluate)
        self.assertEqual(dev_cfg.experiment.training.max_steps, 10000)
        self.assertEqual(dev_cfg.experiment.training.overfit_batches, 0.0)

        from experiments.train_experiment import TrainExperiment

        experiment = TrainExperiment.__new__(TrainExperiment)
        experiment.cfg = smoke_cfg
        experiment._validate_project_profile_contract()
        smoke_cfg.experiment.training.max_steps = 1001
        with self.assertRaisesRegex(ValueError, "safety contract"):
            experiment._validate_project_profile_contract()

    def test_fixed_subset_is_stable_and_hash_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fixed.json"
            first_dataset = FakeDataset()
            first = smoke.ensure_fixed_subset(first_dataset, path, size=32, seed=42)
            second_dataset = FakeDataset()
            second = smoke.ensure_fixed_subset(second_dataset, path, size=32, seed=42)
            self.assertEqual(len(first_dataset.selected), 32)
            self.assertEqual(first_dataset.selected, second_dataset.selected)
            self.assertEqual(first["slices_sha256"], second["slices_sha256"])

            payload = json.loads(path.read_text())
            payload["slices"][0]["start_frame"] += 1
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                smoke.ensure_fixed_subset(FakeDataset(), path, size=32, seed=42)

    def test_action_variants_are_deterministic(self) -> None:
        actions = torch.stack(
            [torch.full((4, 3), float(index)) for index in range(4)]
        )
        first = baselines.build_action_variants(actions, seed=7)
        second = baselines.build_action_variants(actions, seed=7)
        self.assertTrue(torch.equal(first["correct"], actions))
        self.assertEqual(torch.count_nonzero(first["zero"]).item(), 0)
        self.assertTrue(torch.equal(first["shuffled"], second["shuffled"]))
        self.assertFalse(torch.any(torch.all(first["shuffled"] == actions, dim=(1, 2))))

    def test_checkpoint_and_loss_reports_extract_required_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "checkpoint.ckpt"
            torch.save(
                {
                    "global_step": 500,
                    "epoch": 3,
                    "optimizer_states": [{"state": {}}],
                    "lr_schedulers": [{"last_epoch": 500}],
                    "state_dict": {"model.weight": torch.ones(1)},
                },
                checkpoint,
            )
            summary = smoke.checkpoint_summary(checkpoint)
            self.assertEqual(summary["global_step"], 500)
            self.assertEqual(summary["optimizer_state_count"], 1)
            self.assertEqual(summary["scheduler_state_count"], 1)

            log = root / "rank_0.log"
            log.write_text(
                "\n".join(
                    f"(step={step:07d}/epoch=0000) Train Loss: {1.0 / step:.4f}"
                    for step in range(10, 70, 10)
                )
            )
            losses = smoke.parse_training_losses([log])
            self.assertEqual([item["step"] for item in losses], [10, 20, 30, 40, 50, 60])
            self.assertLess(losses[-1]["loss"], losses[0]["loss"])


if __name__ == "__main__":
    unittest.main()
