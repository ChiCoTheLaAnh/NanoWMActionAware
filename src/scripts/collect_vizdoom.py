#!/usr/bin/env python3
"""Collect deterministic VizDoom Basic pilot episodes in HDF5 format."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import h5py
import numpy as np


SCHEMA_VERSION = 1
ACTION_NAMES = ("left", "right", "shoot")
EXPECTED_BUTTONS = ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK")
MAX_EVIDENCE_VIDEO_BYTES = 5_000_000


@dataclass(frozen=True)
class Episode:
    frames: np.ndarray
    actions: np.ndarray
    action_onehot: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    seed: int
    success: bool


def parse_resolution(value: str) -> Tuple[int, int]:
    """Parse WIDTHxHEIGHT into a positive integer pair."""
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"resolution must look like WIDTHxHEIGHT, got {value!r}"
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return width, height


def sample_uniform_actions(seed: int, length: int) -> np.ndarray:
    """Return a reproducible sequence of action IDs in [0, 2]."""
    if length < 0:
        raise ValueError("length must be non-negative")
    return np.random.default_rng(seed).integers(
        low=0, high=len(ACTION_NAMES), size=length, dtype=np.int64
    )


def _normalise_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.shape == (3, height, width):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape != (height, width, 3):
        raise ValueError(
            f"unexpected frame shape {frame.shape}; expected {(height, width, 3)}"
        )
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8, copy=False)
    return np.ascontiguousarray(frame)


def _button_name(button: Any) -> str:
    return str(button).rsplit(".", maxsplit=1)[-1]


def _screen_resolution(vizdoom: Any, width: int, height: int) -> Any:
    name = f"RES_{width}X{height}"
    try:
        return getattr(vizdoom.ScreenResolution, name)
    except AttributeError as exc:
        raise ValueError(
            f"VizDoom does not provide {name}; choose a supported native resolution"
        ) from exc


def collect_episode(
    seed: int,
    frame_skip: int,
    resolution: Tuple[int, int],
    policy: str = "uniform",
) -> Tuple[Episode, str]:
    """Collect one independently seeded episode using the original VizDoom API."""
    if frame_skip <= 0:
        raise ValueError("frame_skip must be positive")
    if policy != "uniform":
        raise ValueError(f"unsupported policy {policy!r}; expected 'uniform'")

    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise RuntimeError(
            "vizdoom is required for collection; install requirements-colab.txt"
        ) from exc

    width, height = resolution
    game = vzd.DoomGame()
    try:
        game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
        game.set_seed(int(seed))
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(_screen_resolution(vzd, width, height))
        game.init()

        buttons = tuple(_button_name(button) for button in game.get_available_buttons())
        if buttons != EXPECTED_BUTTONS:
            raise ValueError(
                f"unexpected VizDoom Basic action map {buttons}; expected {EXPECTED_BUTTONS}"
            )

        rng = np.random.default_rng(seed)
        frames = []
        actions = []
        rewards = []
        dones = []

        while not game.is_episode_finished():
            state = game.get_state()
            if state is None or state.screen_buffer is None:
                raise RuntimeError("VizDoom returned no frame before episode termination")

            action_id = int(rng.integers(0, len(ACTION_NAMES)))
            action_vector = np.eye(len(ACTION_NAMES), dtype=np.int32)[action_id]
            frames.append(_normalise_frame(state.screen_buffer, width, height))
            actions.append(action_id)

            reward = float(game.make_action(action_vector.tolist(), frame_skip))
            rewards.append(reward)
            dones.append(bool(game.is_episode_finished()))

        episode = Episode(
            frames=np.stack(frames, axis=0),
            actions=np.asarray(actions, dtype=np.int64),
            action_onehot=np.eye(len(ACTION_NAMES), dtype=np.float32)[actions],
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.bool_),
            seed=int(seed),
            success=not bool(game.is_episode_timeout_reached()),
        )
        validate_episode(episode, resolution)
        return episode, str(getattr(vzd, "__version__", "unknown"))
    finally:
        game.close()


def validate_episode(episode: Episode, resolution: Tuple[int, int]) -> None:
    """Validate the in-memory schema and frame/action alignment contract."""
    width, height = resolution
    length = len(episode.actions)
    if length == 0:
        raise ValueError("episode must contain at least one transition")

    expected_shapes = {
        "frames": (length, height, width, 3),
        "action_onehot": (length, len(ACTION_NAMES)),
        "rewards": (length,),
        "dones": (length,),
    }
    for field, expected in expected_shapes.items():
        actual = getattr(episode, field).shape
        if actual != expected:
            raise ValueError(f"{field} shape {actual} does not match {expected}")

    if episode.frames.dtype != np.uint8:
        raise ValueError(f"frames must be uint8, got {episode.frames.dtype}")
    if episode.actions.dtype != np.int64:
        raise ValueError(f"actions must be int64, got {episode.actions.dtype}")
    if episode.action_onehot.dtype != np.float32:
        raise ValueError(
            f"action_onehot must be float32, got {episode.action_onehot.dtype}"
        )
    if episode.rewards.dtype != np.float32:
        raise ValueError(f"rewards must be float32, got {episode.rewards.dtype}")
    if episode.dones.dtype != np.bool_:
        raise ValueError(f"dones must be bool, got {episode.dones.dtype}")
    if np.any((episode.actions < 0) | (episode.actions >= len(ACTION_NAMES))):
        raise ValueError("actions must contain only IDs 0, 1, or 2")
    expected_onehot = np.eye(len(ACTION_NAMES), dtype=np.float32)[episode.actions]
    if not np.array_equal(episode.action_onehot, expected_onehot):
        raise ValueError("action_onehot does not match actions")
    if not bool(episode.dones[-1]):
        raise ValueError("the final transition must be terminal")
    if np.any(episode.dones[:-1]):
        raise ValueError("only the final transition may be terminal")


def write_episode(
    path: Path,
    episode: Episode,
    resolution: Tuple[int, int],
    frame_skip: int,
    vizdoom_version: str,
) -> None:
    """Atomically write one validated episode."""
    validate_episode(episode, resolution)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    width, height = resolution
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("frames", data=episode.frames, compression="gzip")
        handle.create_dataset("actions", data=episode.actions)
        handle.create_dataset("action_onehot", data=episode.action_onehot)
        handle.create_dataset("rewards", data=episode.rewards)
        handle.create_dataset("dones", data=episode.dones)
        handle.create_dataset("seed", data=np.int64(episode.seed))
        handle.create_dataset("success", data=np.bool_(episode.success))
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["action_names"] = json.dumps(ACTION_NAMES)
        handle.attrs["frame_skip"] = int(frame_skip)
        handle.attrs["resolution"] = f"{width}x{height}"
        handle.attrs["vizdoom_version"] = vizdoom_version
    temporary.replace(path)


def read_and_validate_episode(path: Path) -> Episode:
    """Read an HDF5 episode and validate all required fields and attributes."""
    with h5py.File(path, "r") as handle:
        required = {
            "frames", "actions", "action_onehot", "rewards", "dones",
            "seed", "success",
        }
        missing = required.difference(handle.keys())
        if missing:
            raise ValueError(f"{path} is missing datasets: {sorted(missing)}")
        if int(handle.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"{path} has an unsupported schema version")
        resolution = parse_resolution(str(handle.attrs["resolution"]))
        episode = Episode(
            frames=handle["frames"][:],
            actions=handle["actions"][:],
            action_onehot=handle["action_onehot"][:],
            rewards=handle["rewards"][:],
            dones=handle["dones"][:].astype(np.bool_, copy=False),
            seed=int(handle["seed"][()]),
            success=bool(handle["success"][()]),
        )
    validate_episode(episode, resolution)
    return episode


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_video(path: Path, frames: np.ndarray, fps: int = 12) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio and imageio-ffmpeg are required for MP4 output") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        path,
        frames,
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=8,
    )
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"video writer did not produce {path}")
    if path.stat().st_size > MAX_EVIDENCE_VIDEO_BYTES:
        raise RuntimeError(
            f"evidence video is {path.stat().st_size} bytes; limit is "
            f"{MAX_EVIDENCE_VIDEO_BYTES}"
        )


def git_revision(ref: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", ref], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def collect_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seen_seeds = set()
    first_frames: Optional[np.ndarray] = None
    vizdoom_version = "unknown"

    for episode_index in range(args.episodes):
        seed = args.base_seed + episode_index
        if seed in seen_seeds:
            raise RuntimeError(f"duplicate seed {seed}")
        seen_seeds.add(seed)
        episode, vizdoom_version = collect_episode(
            seed=seed,
            frame_skip=args.frame_skip,
            resolution=args.resolution,
            policy=args.policy,
        )
        episode_path = output_dir / f"episode_{episode_index:05d}.hdf5"
        write_episode(
            episode_path,
            episode,
            resolution=args.resolution,
            frame_skip=args.frame_skip,
            vizdoom_version=vizdoom_version,
        )
        round_trip = read_and_validate_episode(episode_path)
        if round_trip.seed != seed:
            raise RuntimeError(f"seed changed during HDF5 round trip for {episode_path}")
        if first_frames is None:
            first_frames = episode.frames
        records.append(
            {
                "file": episode_path.name,
                "seed": seed,
                "length": len(episode.actions),
                "return": float(episode.rewards.sum()),
                "success": bool(episode.success),
                "sha256": sha256_file(episode_path),
            }
        )

    if args.video_out is not None:
        if first_frames is None:
            raise RuntimeError("no frames available for evidence video")
        render_video(args.video_out.resolve(), first_frames)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "collector": "src/scripts/collect_vizdoom.py",
        "policy": args.policy,
        "base_seed": args.base_seed,
        "episodes": args.episodes,
        "frame_skip": args.frame_skip,
        "resolution": f"{args.resolution[0]}x{args.resolution[1]}",
        "action_names": list(ACTION_NAMES),
        "vizdoom_version": vizdoom_version,
        "nanowm_revision": git_revision("HEAD"),
        "upstream_revision": git_revision("upstream/main"),
        "files": records,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "status": "ok",
        "episodes": args.episodes,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "video": str(args.video_out.resolve()) if args.video_out else None,
        "successes": sum(int(record["success"]) for record in records),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--policy", choices=("uniform",), default="uniform")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--resolution", type=parse_resolution, default=(160, 120))
    parser.add_argument("--video-out", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.frame_skip <= 0:
        parser.error("--frame-skip must be positive")
    try:
        summary = collect_dataset(args)
    except Exception as exc:
        print(f"collection failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
