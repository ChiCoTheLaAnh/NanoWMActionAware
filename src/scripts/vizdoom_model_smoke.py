#!/usr/bin/env python3
"""Day 3 VizDoom model smoke-test and evidence utilities.

The script deliberately keeps evidence generation separate from the training
callbacks so a tiny overfit run does not need I3D/FID assets. It operates on a
fully resolved Hydra config saved by a training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def canonical_slices_sha256(slices: list[dict[str, int]]) -> str:
    payload = json.dumps(slices, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: Path):
    cfg = OmegaConf.load(path)
    required = ("model", "dataset", "experiment", "latent_codec")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Resolved config is missing required sections: {missing}")
    if cfg.dataset.name != "vizdoom_basic":
        raise ValueError(f"Expected dataset.name=vizdoom_basic, got {cfg.dataset.name!r}")
    return cfg


def create_dataset(cfg, split: str):
    from wm_datasets import create_world_model_dataset

    loader = OmegaConf.to_container(cfg.dataset.loader, resolve=True)
    loader.pop("training_fixed_subset_path", None)
    loader.pop("training_fixed_subset_size", None)
    loader.pop("training_fixed_subset_seed", None)
    loader.pop("validation_fixed_subset_path", None)
    loader.pop("validation_fixed_subset_size", None)
    loader.pop("validation_fixed_subset_seed", None)
    loader.pop("validation_size", None)
    loader.pop("train_slice_mode", None)
    loader.pop("val_slice_mode", None)
    return create_world_model_dataset(
        dataset_name=cfg.dataset.name,
        num_frames=cfg.model.num_frames,
        frame_interval=cfg.dataset.frame_interval,
        image_size=(cfg.model.image_size, cfg.model.image_size),
        split=split,
        slice_mode="exhaustive",
        **loader,
    )


def ensure_fixed_subset(
    dataset,
    manifest_path: Path,
    *,
    size: int,
    seed: int,
    dataset_name: str = "vizdoom_basic",
) -> dict[str, Any]:
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        slices = payload.get("slices", [])
        if len(slices) != size:
            raise ValueError(
                f"Fixed subset has {len(slices)} slices, expected {size}: {manifest_path}"
            )
        actual_hash = canonical_slices_sha256(slices)
        expected_hash = payload.get("slices_sha256")
        if expected_hash is None:
            raise ValueError(f"Fixed subset is missing slices_sha256: {manifest_path}")
        if actual_hash != expected_hash:
            raise ValueError(
                f"Fixed subset hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
    else:
        slices = dataset.sample_fixed_slice_specs(size, seed=seed)
        if len(slices) != size:
            raise ValueError(f"Dataset only provides {len(slices)} slices; {size} are required")
        actual_hash = canonical_slices_sha256(slices)
        payload = {
            "dataset": dataset_name,
            "selection_seed": seed,
            "slices_sha256": actual_hash,
            "slices": slices,
        }
        write_json(manifest_path, payload)

    dataset.set_fixed_slices(slices)
    payload["slices_sha256"] = actual_hash
    return payload


def load_fixed_batch(cfg, manifest_path: Path, *, size: int, seed: int, batch_size: int):
    dataset = create_dataset(cfg, split="train")
    manifest = ensure_fixed_subset(dataset, manifest_path, size=size, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return next(iter(loader)), manifest


def run_vae(args: argparse.Namespace) -> None:
    from latent_codecs import build_latent_codec
    from torchvision.utils import save_image

    cfg = load_config(args.config)
    device = torch.device(args.device)
    batch, manifest = load_fixed_batch(
        cfg,
        args.subset_manifest,
        size=args.subset_size,
        seed=args.subset_seed,
        batch_size=args.batch_size,
    )
    video = batch["video"].to(device)
    b, frames = video.shape[:2]

    codec = build_latent_codec(cfg).to(device).eval()
    codec.requires_grad_(False)
    with torch.no_grad():
        flat = video.reshape(b * frames, *video.shape[2:])
        latents = codec.encode(flat)
        reconstruction = codec.decode(latents)

    expected_latent_shape = (
        b * frames,
        int(cfg.model.latent_channels),
        int(cfg.model.latent_size),
        int(cfg.model.latent_size),
    )
    if tuple(latents.shape) != expected_latent_shape:
        raise RuntimeError(
            f"VAE latent shape mismatch: got {tuple(latents.shape)}, expected {expected_latent_shape}"
        )
    if tuple(reconstruction.shape) != tuple(flat.shape):
        raise RuntimeError(
            f"VAE reconstruction shape mismatch: got {tuple(reconstruction.shape)}, "
            f"expected {tuple(flat.shape)}"
        )
    if not torch.isfinite(latents).all() or not torch.isfinite(reconstruction).all():
        raise RuntimeError("VAE encode/decode produced NaN or Inf")

    mse = torch.mean((reconstruction.float() - flat.float()) ** 2).item()
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(4.0 / mse)
    interleaved = torch.stack((flat, reconstruction.clamp(-1, 1)), dim=1).flatten(0, 1)
    image_path = args.output_dir / "vae-reconstruction.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    save_image((interleaved + 1.0) / 2.0, image_path, nrow=2)

    report = {
        "status": "passed",
        "device": str(device),
        "subset_manifest": str(args.subset_manifest),
        "subset_slices_sha256": manifest["slices_sha256"],
        "video_names": list(batch["video_name"]),
        "input_shape": list(video.shape),
        "flattened_input_shape": list(flat.shape),
        "latent_shape": list(latents.shape),
        "reconstruction_shape": list(reconstruction.shape),
        "finite": True,
        "reconstruction_mse_minus1_to1": mse,
        "reconstruction_psnr_db": psnr,
        "reconstruction_image": str(image_path),
    }
    write_json(args.output_dir / "vae-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def load_model_and_codec(cfg, checkpoint: Path, device: torch.device):
    from latent_codecs import build_latent_codec
    from models import get_models
    from utils.nanowm_utils import find_model

    model = get_models(cfg).to(device)
    model.load_state_dict(find_model(str(checkpoint)), strict=True)
    model.eval()
    codec = build_latent_codec(cfg).to(device).eval()
    codec.requires_grad_(False)
    return model, codec


def sample_predictions(cfg, model, codec, video, actions, *, device, sampling_steps, seed):
    from diffusion import create_diffusion
    from diffusion.df_sample import dfot_sample

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    b, frames = video.shape[:2]
    with torch.no_grad():
        flat = video.to(device).reshape(b * frames, *video.shape[2:])
        latents = codec.encode(flat).reshape(
            b,
            frames,
            int(cfg.model.latent_channels),
            int(cfg.model.latent_size),
            int(cfg.model.latent_size),
        )
        diffusion = create_diffusion(
            timestep_respacing=str(sampling_steps),
            noise_schedule=cfg.experiment.diffusion.noise_schedule,
            pred_name=cfg.experiment.diffusion.pred_name,
            diffusion_steps=cfg.experiment.diffusion.diffusion_steps,
            snr_gamma=cfg.experiment.diffusion.snr_gamma,
            zero_terminal_snr=cfg.experiment.diffusion.zero_terminal_snr,
        )
        predicted = dfot_sample(
            diffusion=diffusion,
            model=model.forward,
            shape=latents.shape,
            context=latents[:, : int(cfg.model.n_context_frames)],
            n_context_frames=int(cfg.model.n_context_frames),
            scheduling_mode=str(cfg.experiment.evaluation.scheduling_mode or cfg.model.scheduling_mode),
            num_sampling_steps=sampling_steps,
            model_kwargs={"y": None, "action": actions.to(device)},
            device=device,
            progress=False,
            eta=0.0,
            clip_denoised=False,
            history_stabilization_level=cfg.experiment.diffusion.history_stabilization_level,
        )
        pred = ((codec.decode(predicted.flatten(0, 1)) + 1.0) / 2.0).clamp(0, 1)
    gt = ((video.to(device).float() + 1.0) / 2.0).clamp(0, 1)
    pred = pred.reshape(b, frames, *pred.shape[1:])
    return gt, pred


def run_render(args: argparse.Namespace) -> None:
    from sample.sampling_utils import save_comparison_video, save_video

    cfg = load_config(args.config)
    device = torch.device(args.device)
    batch, manifest = load_fixed_batch(
        cfg,
        args.subset_manifest,
        size=args.subset_size,
        seed=args.subset_seed,
        batch_size=args.num_samples,
    )
    model, codec = load_model_and_codec(cfg, args.checkpoint, device)
    gt, pred = sample_predictions(
        cfg,
        model,
        codec,
        batch["video"],
        batch["action"],
        device=device,
        sampling_steps=args.sampling_steps,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for index in range(gt.shape[0]):
        stem = f"sample_{index:04d}"
        gt_path = args.output_dir / f"{stem}_gt.mp4"
        pred_path = args.output_dir / f"{stem}_pred.mp4"
        compare_path = args.output_dir / f"{stem}_compare.mp4"
        save_video(gt[index], str(gt_path), fps=args.fps)
        save_video(pred[index], str(pred_path), fps=args.fps)
        save_comparison_video(gt[index], pred[index], str(compare_path), fps=args.fps)
        artifacts.append(
            {"video_name": batch["video_name"][index], "gt": str(gt_path), "pred": str(pred_path), "compare": str(compare_path)}
        )

    n_context = int(cfg.model.n_context_frames)
    future = pred[:, n_context:]
    temporal_l1 = (
        torch.mean(torch.abs(future[:, 1:] - future[:, :-1])).item()
        if future.shape[1] > 1
        else 0.0
    )
    gt_future = gt[:, n_context:]
    gt_temporal_l1 = (
        torch.mean(torch.abs(gt_future[:, 1:] - gt_future[:, :-1])).item()
        if gt_future.shape[1] > 1
        else 0.0
    )
    report = {
        "status": "passed" if temporal_l1 > args.motion_threshold else "failed_static_prediction",
        "checkpoint": str(args.checkpoint),
        "subset_slices_sha256": manifest["slices_sha256"],
        "sampling_steps": args.sampling_steps,
        "predicted_future_temporal_l1": temporal_l1,
        "ground_truth_future_temporal_l1": gt_temporal_l1,
        "motion_threshold": args.motion_threshold,
        "artifacts": artifacts,
    }
    write_json(args.output_dir / "prediction-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit("Generated prediction is effectively static")


def checkpoint_summary(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path),
        "global_step": int(checkpoint.get("global_step", -1)),
        "epoch": int(checkpoint.get("epoch", -1)),
        "optimizer_state_count": len(checkpoint.get("optimizer_states", [])),
        "scheduler_state_count": len(checkpoint.get("lr_schedulers", [])),
        "state_dict_tensor_count": len(checkpoint.get("state_dict", {})),
    }


def run_resume_report(args: argparse.Namespace) -> None:
    before = checkpoint_summary(args.before)
    after = checkpoint_summary(args.after)
    advanced_steps = after["global_step"] - before["global_step"]
    full_state = all(
        summary["optimizer_state_count"] > 0
        and summary["scheduler_state_count"] > 0
        and summary["state_dict_tensor_count"] > 0
        for summary in (before, after)
    )
    passed = before["global_step"] >= 0 and advanced_steps >= args.min_advanced_steps and full_state
    report = {
        "status": "passed" if passed else "failed",
        "before": before,
        "after": after,
        "advanced_optimizer_steps": advanced_steps,
        "minimum_required_advanced_steps": args.min_advanced_steps,
        "full_trainer_state_present": full_state,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("Checkpoint resume evidence did not pass")


LOSS_PATTERN = re.compile(r"step=(\d+).*?Train Loss:\s*([0-9.eE+-]+)")


def parse_training_losses(paths: list[Path]) -> list[dict[str, float]]:
    by_step: dict[int, float] = {}
    for path in paths:
        for match in LOSS_PATTERN.finditer(path.read_text(encoding="utf-8", errors="replace")):
            by_step[int(match.group(1))] = float(match.group(2))
    return [{"step": step, "loss": by_step[step]} for step in sorted(by_step)]


def run_loss_report(args: argparse.Namespace) -> None:
    losses = parse_training_losses(args.logs)
    if len(losses) < args.window * 2:
        raise ValueError(
            f"Need at least {args.window * 2} logged losses, found {len(losses)}"
        )
    initial = sum(item["loss"] for item in losses[: args.window]) / args.window
    final = sum(item["loss"] for item in losses[-args.window :]) / args.window
    passed = math.isfinite(initial) and math.isfinite(final) and final < initial
    report = {
        "status": "passed" if passed else "failed",
        "logged_points": len(losses),
        "window": args.window,
        "initial_mean_loss": initial,
        "final_mean_loss": final,
        "relative_change": (final - initial) / initial if initial else None,
        "first_step": losses[0]["step"],
        "last_step": losses[-1]["step"],
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("Tiny-set loss did not improve")


def add_subset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="Resolved Hydra config.yaml")
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=32)
    parser.add_argument("--subset-seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    vae = subparsers.add_parser("vae", help="Encode/decode a verified fixed VizDoom batch")
    add_subset_args(vae)
    vae.add_argument("--output-dir", type=Path, required=True)
    vae.add_argument("--batch-size", type=int, default=2)
    vae.add_argument("--device", default="cuda")
    vae.set_defaults(func=run_vae)

    render = subparsers.add_parser("render", help="Render GT versus checkpoint prediction")
    add_subset_args(render)
    render.add_argument("--checkpoint", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--num-samples", type=int, default=4)
    render.add_argument("--sampling-steps", type=int, default=10)
    render.add_argument("--motion-threshold", type=float, default=1e-4)
    render.add_argument("--seed", type=int, default=3407)
    render.add_argument("--fps", type=int, default=4)
    render.add_argument("--device", default="cuda")
    render.set_defaults(func=run_render)

    resume = subparsers.add_parser("resume-report", help="Verify two full-state checkpoints")
    resume.add_argument("--before", type=Path, required=True)
    resume.add_argument("--after", type=Path, required=True)
    resume.add_argument("--output", type=Path, required=True)
    resume.add_argument("--min-advanced-steps", type=int, default=10)
    resume.set_defaults(func=run_resume_report)

    loss = subparsers.add_parser("loss-report", help="Summarize tiny-set loss improvement")
    loss.add_argument("--logs", type=Path, nargs="+", required=True)
    loss.add_argument("--output", type=Path, required=True)
    loss.add_argument("--window", type=int, default=5)
    loss.set_defaults(func=run_loss_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
