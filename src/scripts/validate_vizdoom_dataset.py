#!/usr/bin/env python3
"""Validate VizDoom episodes and produce the Day 2 evidence bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import collect_vizdoom as collector  # noqa: E402
from wm_datasets import create_train_val_datasets  # noqa: E402


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_manifest(dataset_dir: Path) -> Dict[str, Any]:
    path = dataset_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"VizDoom manifest not found: {path}")
    return json.loads(path.read_text())


def validate_dataset(dataset_dir: Path) -> Tuple[Dict[str, Any], List[collector.Episode]]:
    """Validate all episode files and their agreement with the manifest."""
    manifest = _load_manifest(dataset_dir)
    files = sorted(dataset_dir.glob("episode_*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {dataset_dir}")
    if int(manifest.get("episodes", -1)) != len(files):
        raise ValueError(
            f"Manifest declares {manifest.get('episodes')} episodes, found {len(files)}"
        )

    manifest_records = {record["file"]: record for record in manifest.get("files", [])}
    if set(manifest_records) != {path.name for path in files}:
        raise ValueError("Manifest file list does not exactly match the episode files")

    episodes: List[collector.Episode] = []
    seeds = set()
    counts: Counter[int] = Counter()
    file_reports = []
    for path in files:
        episode = collector.read_and_validate_episode(path)
        if episode.seed in seeds:
            raise ValueError(f"Duplicate seed {episode.seed} in {path}")
        seeds.add(episode.seed)

        record = manifest_records[path.name]
        checksum = collector.sha256_file(path)
        if checksum != record.get("sha256"):
            raise ValueError(f"Checksum mismatch for {path}")
        if int(record.get("seed", -1)) != episode.seed:
            raise ValueError(f"Seed mismatch between {path} and manifest")
        if int(record.get("length", -1)) != len(episode.actions):
            raise ValueError(f"Length mismatch between {path} and manifest")
        if bool(record.get("success")) != episode.success:
            raise ValueError(f"Success mismatch between {path} and manifest")

        with h5py.File(path, "r") as handle:
            required_attrs = {
                "schema_version",
                "action_names",
                "frame_skip",
                "resolution",
                "vizdoom_version",
            }
            missing_attrs = required_attrs.difference(handle.attrs.keys())
            if missing_attrs:
                raise ValueError(f"{path} is missing attributes: {sorted(missing_attrs)}")
            if int(handle.attrs["frame_skip"]) != int(manifest["frame_skip"]):
                raise ValueError(f"Frame-skip mismatch for {path}")
            if str(handle.attrs["resolution"]) != str(manifest["resolution"]):
                raise ValueError(f"Resolution mismatch for {path}")

        counts.update(int(action) for action in episode.actions)
        episodes.append(episode)
        file_reports.append(
            {
                "file": path.name,
                "seed": episode.seed,
                "length": len(episode.actions),
                "success": episode.success,
                "sha256": checksum,
            }
        )

    missing_actions = [
        collector.ACTION_NAMES[index]
        for index in range(len(collector.ACTION_NAMES))
        if counts[index] == 0
    ]
    if missing_actions:
        raise ValueError(f"Pilot dataset has no samples for actions: {missing_actions}")

    total = sum(counts.values())
    report = {
        "status": "passed",
        "schema_version": collector.SCHEMA_VERSION,
        "dataset_dir": os.path.relpath(dataset_dir.resolve(), Path.cwd()),
        "episodes": len(episodes),
        "total_transitions": total,
        "seed_min": min(seeds),
        "seed_max": max(seeds),
        "unique_seeds": len(seeds),
        "successes": sum(int(episode.success) for episode in episodes),
        "terminal_flags_valid": True,
        "manifest_checksums_valid": True,
        "action_counts": {
            name: counts[index] for index, name in enumerate(collector.ACTION_NAMES)
        },
        "action_fractions": {
            name: counts[index] / total
            for index, name in enumerate(collector.ACTION_NAMES)
        },
        "files": file_reports,
    }
    return report, episodes


def _setup_replay_game(seed: int, resolution: Tuple[int, int]):
    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise RuntimeError("vizdoom is required for semantic replay validation") from exc

    width, height = resolution
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    game.set_seed(seed)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(collector._screen_resolution(vzd, width, height))
    game.init()
    buttons = tuple(collector._button_name(button) for button in game.get_available_buttons())
    if buttons != collector.EXPECTED_BUTTONS:
        game.close()
        raise ValueError(f"Unexpected VizDoom action map {buttons}")
    return game


def validate_alignment(
    dataset_dir: Path, episodes: Sequence[collector.Episode]
) -> Dict[str, Any]:
    """Replay stored actions and prove action[t] maps frame[t] to frame[t+1]."""
    files = sorted(dataset_dir.glob("episode_*.hdf5"))
    episode_reports = []
    checked_transitions = 0

    for path, episode in zip(files, episodes):
        with h5py.File(path, "r") as handle:
            resolution = collector.parse_resolution(str(handle.attrs["resolution"]))
            frame_skip = int(handle.attrs["frame_skip"])

        game = _setup_replay_game(episode.seed, resolution)
        try:
            width, height = resolution
            for index, action_id in enumerate(episode.actions):
                state = game.get_state()
                if state is None or state.screen_buffer is None:
                    raise ValueError(f"Replay ended before stored frame {index} in {path}")
                replay_frame = collector._normalise_frame(
                    state.screen_buffer, width, height
                )
                if not np.array_equal(replay_frame, episode.frames[index]):
                    raise ValueError(f"Stored frame {index} does not match replay in {path}")

                reward = float(
                    game.make_action(episode.action_onehot[index].astype(int).tolist(), frame_skip)
                )
                done = bool(game.is_episode_finished())
                if not np.isclose(reward, float(episode.rewards[index])):
                    raise ValueError(f"Reward mismatch at transition {index} in {path}")
                if done != bool(episode.dones[index]):
                    raise ValueError(f"Terminal mismatch at transition {index} in {path}")

                if index + 1 < len(episode.frames):
                    if done:
                        raise ValueError(f"Replay terminated before frame {index + 1} in {path}")
                    next_state = game.get_state()
                    if next_state is None or next_state.screen_buffer is None:
                        raise ValueError(f"Replay has no next frame after action {index} in {path}")
                    next_frame = collector._normalise_frame(
                        next_state.screen_buffer, width, height
                    )
                    if not np.array_equal(next_frame, episode.frames[index + 1]):
                        raise ValueError(
                            f"action[{index}] does not map frame[{index}] to "
                            f"frame[{index + 1}] in {path}"
                        )
                    checked_transitions += 1

            if not game.is_episode_finished():
                raise ValueError(f"Replay did not terminate after stored actions in {path}")
        finally:
            game.close()

        episode_reports.append(
            {
                "file": path.name,
                "seed": episode.seed,
                "frames_checked": len(episode.frames),
                "aligned_transitions_checked": max(0, len(episode.frames) - 1),
            }
        )

    return {
        "status": "passed",
        "contract": "action[t] maps observation[t] to observation[t+1]",
        "episodes_checked": len(episodes),
        "frames_checked": sum(len(episode.frames) for episode in episodes),
        "aligned_transitions_checked": checked_transitions,
        "terminal_action_note": (
            "The final stored action terminates the episode and has no stored successor; "
            "NanoWM ignores the final clip action after its internal one-step shift."
        ),
        "episodes": episode_reports,
    }


def plot_action_distribution(report: Dict[str, Any], path: Path) -> None:
    names = list(collector.ACTION_NAMES)
    counts = [report["action_counts"][name] for name in names]
    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(names, counts, color=("#4c78a8", "#f58518", "#54a24b"))
    axis.set_title("VizDoom Pilot Action Distribution")
    axis.set_ylabel("Transitions")
    axis.bar_label(bars)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_episode_visualization(episodes: Sequence[collector.Episode], path: Path) -> None:
    eligible = [episode for episode in episodes if len(episode.frames) >= 4]
    if not eligible:
        eligible = list(episodes)
    selected = np.linspace(0, len(eligible) - 1, min(3, len(eligible)), dtype=int)
    figure, axes = plt.subplots(len(selected), 3, figsize=(11, 3.4 * len(selected)), squeeze=False)
    for row, episode_index in enumerate(selected):
        episode = eligible[int(episode_index)]
        frame_indices = [0, len(episode.frames) // 2, len(episode.frames) - 1]
        for column, frame_index in enumerate(frame_indices):
            axes[row, column].imshow(episode.frames[frame_index])
            action = collector.ACTION_NAMES[int(episode.actions[frame_index])]
            axes[row, column].set_title(
                f"seed={episode.seed} frame={frame_index} action={action}"
            )
            axes[row, column].axis("off")
    figure.suptitle("VizDoom Pilot Episodes: First, Middle, Final Frames")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def validate_loader(dataset_dir: Path, preview_path: Path) -> Dict[str, Any]:
    train_dataset, val_dataset = create_train_val_datasets(
        dataset_name="vizdoom_basic",
        data_path=str(dataset_dir),
        num_frames=4,
        frame_interval=1,
        image_size=(256, 256),
        split_ratio=0.9,
        normalize_action=False,
        normalize_state=False,
        normalize_pixel=True,
        random_seed=42,
        train_slice_mode="exhaustive",
        val_slice_mode="exhaustive",
        stride=1,
        resize_mode="pad",
    )
    loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    video = batch["video"]
    actions = batch["action"]
    expected_video_shape = (2, 4, 3, 256, 256)
    expected_action_shape = (2, 4, 3)
    if tuple(video.shape) != expected_video_shape:
        raise ValueError(f"Unexpected video batch shape {tuple(video.shape)}")
    if tuple(actions.shape) != expected_action_shape:
        raise ValueError(f"Unexpected action batch shape {tuple(actions.shape)}")
    if float(video.min()) < -1.00001 or float(video.max()) > 1.00001:
        raise ValueError("Video batch is outside the expected [-1, 1] range")
    if not torch.allclose(actions.sum(dim=-1), torch.ones_like(actions[..., 0])):
        raise ValueError("DataLoader actions are not one-hot")

    figure, axes = plt.subplots(1, 4, figsize=(12, 3))
    for frame_index in range(4):
        image = ((video[0, frame_index] + 1.0) / 2.0).permute(1, 2, 0).numpy()
        action_id = int(actions[0, frame_index].argmax())
        axes[frame_index].imshow(np.clip(image, 0.0, 1.0))
        role = "context" if frame_index == 0 else f"future {frame_index}"
        axes[frame_index].set_title(
            f"{role}\na={collector.ACTION_NAMES[action_id]}"
        )
        axes[frame_index].axis("off")
    figure.suptitle("Verified 1-context + 3-future DataLoader Clip")
    figure.tight_layout()
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(preview_path, dpi=150)
    plt.close(figure)

    short_episodes = sum(
        int(train_dataset.data_source.get_seq_length(index) < 4)
        for index in range(train_dataset.data_source.get_num_trajectories())
    )
    return {
        "status": "passed",
        "train_slices": len(train_dataset),
        "validation_slices": len(val_dataset),
        "short_episodes_skipped_by_slice_indexer": short_episodes,
        "batch_size": int(video.shape[0]),
        "video_shape": list(video.shape),
        "action_shape": list(actions.shape),
        "video_dtype": str(video.dtype),
        "action_dtype": str(actions.dtype),
        "video_min": float(video.min()),
        "video_max": float(video.max()),
        "context_frames": 1,
        "future_frames": 3,
        "action_alignment": (
            "NanoWM shifts actions internally: action[t] conditions frame[t+1]; "
            "the final clip action is unused."
        ),
        "first_batch_video_names": list(batch["video_name"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--evidence-dir", type=Path, default=Path("reports/evidence/day2")
    )
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--skip-loader", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_dir = args.dataset_dir.resolve()
    evidence_dir = args.evidence_dir.resolve()
    try:
        dataset_report, episodes = validate_dataset(dataset_dir)
        plot_action_distribution(
            dataset_report, evidence_dir / "action-distribution.png"
        )
        plot_episode_visualization(
            episodes, evidence_dir / "episode-visualization.png"
        )
        _write_json(evidence_dir / "dataset-validation.json", dataset_report)

        if args.skip_replay:
            alignment_report = {"status": "skipped"}
        else:
            alignment_report = validate_alignment(dataset_dir, episodes)
        _write_json(evidence_dir / "alignment-report.json", alignment_report)

        if args.skip_loader:
            batch_report = {"status": "skipped"}
        else:
            batch_report = validate_loader(
                dataset_dir, evidence_dir / "batch-preview.png"
            )
        _write_json(evidence_dir / "batch-summary.json", batch_report)
    except Exception as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "status": "passed",
        "episodes": dataset_report["episodes"],
        "transitions": dataset_report["total_transitions"],
        "alignment": alignment_report["status"],
        "loader": batch_report["status"],
        "evidence_dir": str(evidence_dir),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
