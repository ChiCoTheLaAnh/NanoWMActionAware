#!/usr/bin/env python3
"""Evaluate copy-last, correct, zero, and shuffled actions on VizDoom."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


SRC = Path(__file__).resolve().parents[1]
SCRIPTS = SRC / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vizdoom_model_smoke import (  # noqa: E402
    create_dataset,
    ensure_fixed_subset,
    load_config,
    load_model_and_codec,
    sample_predictions,
    write_json,
)


def build_action_variants(actions: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    if actions.ndim != 3:
        raise ValueError(f"Expected actions [N,T,A], got {tuple(actions.shape)}")
    if actions.shape[0] < 2:
        raise ValueError("Shuffled-action evaluation requires at least two samples")
    generator = torch.Generator().manual_seed(seed)
    shift = int(torch.randint(1, actions.shape[0], (1,), generator=generator).item())
    permutation = torch.roll(torch.arange(actions.shape[0]), shifts=shift)
    return {
        "correct": actions.clone(),
        "zero": torch.zeros_like(actions),
        "shuffled": actions[permutation].clone(),
    }


def basic_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = pred.float() - target.float()
    mse = torch.mean(error.square()).item()
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
    return {"mse": mse, "psnr_db": psnr}


class PerceptualMetrics:
    def __init__(self, device: torch.device, max_batch_size: int = 8):
        import lpips
        import piqa

        self.ssim = piqa.SSIM(
            window_size=11,
            sigma=1.5,
            n_channels=3,
            reduction="mean",
        ).to(device).eval()
        self.lpips = lpips.LPIPS(net="alex").to(device).eval()
        self.max_batch_size = max_batch_size

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        ssim_values = []
        lpips_values = []
        for start in range(0, pred.shape[0], self.max_batch_size):
            pred_chunk = pred[start : start + self.max_batch_size]
            target_chunk = target[start : start + self.max_batch_size]
            ssim_values.append(self.ssim(pred_chunk, target_chunk))
            lpips_values.append(
                self.lpips(pred_chunk * 2.0 - 1.0, target_chunk * 2.0 - 1.0).mean()
            )
        return {
            "ssim": torch.stack(ssim_values).mean().item(),
            "lpips": torch.stack(lpips_values).mean().item(),
        }


def evaluate_horizons(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    n_context: int,
    perceptual: PerceptualMetrics | None,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    available = target.shape[1] - n_context
    for horizon in sorted({1, available}):
        pred_h = pred[:, n_context : n_context + horizon].flatten(0, 1)
        target_h = target[:, n_context : n_context + horizon].flatten(0, 1)
        metrics = basic_metrics(pred_h, target_h)
        if perceptual is not None:
            metrics.update(perceptual(pred_h, target_h))
        results[f"future_{horizon}_step"] = metrics
    return results


def collate_entire_dataset(dataset) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    return next(iter(loader))


def run(args: argparse.Namespace) -> None:
    from sample.sampling_utils import save_comparison_video

    cfg = load_config(args.config)
    device = torch.device(args.device)
    dataset = create_dataset(cfg, split="val")
    manifest = ensure_fixed_subset(
        dataset,
        args.subset_manifest,
        size=args.subset_size,
        seed=args.subset_seed,
    )
    entire = collate_entire_dataset(dataset)
    action_variants = build_action_variants(entire["action"], args.action_shuffle_seed)
    model, codec = load_model_and_codec(cfg, args.checkpoint, device)
    perceptual = None if args.skip_perceptual else PerceptualMetrics(device)
    n_context = int(cfg.model.n_context_frames)

    all_predictions: dict[str, list[torch.Tensor]] = {key: [] for key in action_variants}
    all_targets: list[torch.Tensor] = []
    for start in range(0, len(dataset), args.batch_size):
        end = min(start + args.batch_size, len(dataset))
        video = entire["video"][start:end]
        target_for_batch = None
        for name, variant_actions in action_variants.items():
            target, prediction = sample_predictions(
                cfg,
                model,
                codec,
                video,
                variant_actions[start:end],
                device=device,
                sampling_steps=args.sampling_steps,
                seed=args.diffusion_seed + start,
            )
            all_predictions[name].append(prediction.cpu())
            target_for_batch = target.cpu()
        assert target_for_batch is not None
        all_targets.append(target_for_batch)

    target = torch.cat(all_targets).to(device)
    predictions = {name: torch.cat(chunks).to(device) for name, chunks in all_predictions.items()}
    copy_last = target[:, n_context - 1 : n_context].expand(
        -1, target.shape[1], -1, -1, -1
    ).clone()
    predictions = {"copy_last": copy_last, **predictions}

    metrics = {
        name: evaluate_horizons(
            prediction,
            target,
            n_context=n_context,
            perceptual=perceptual,
        )
        for name, prediction in predictions.items()
    }
    one_step_key = "future_1_step"
    correct_psnr = metrics["correct"][one_step_key]["psnr_db"]
    copy_psnr = metrics["copy_last"][one_step_key]["psnr_db"]
    action_use_gap_mse = (
        metrics["shuffled"][one_step_key]["mse"]
        - metrics["correct"][one_step_key]["mse"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_index = 0
    videos = {}
    for name, prediction in predictions.items():
        path = args.output_dir / f"sample_{sample_index:04d}_{name}_compare.mp4"
        save_comparison_video(
            target[sample_index].cpu(),
            prediction[sample_index].cpu(),
            str(path),
            fps=args.fps,
        )
        videos[name] = str(path)

    report = {
        "status": "passed" if correct_psnr > copy_psnr else "baseline_does_not_beat_copy_last",
        "checkpoint": str(args.checkpoint),
        "subset_manifest": str(args.subset_manifest),
        "subset_slices_sha256": manifest["slices_sha256"],
        "samples": len(dataset),
        "sampling_steps": args.sampling_steps,
        "metrics": metrics,
        "one_step_correct_minus_copy_last_psnr_db": correct_psnr - copy_psnr,
        "one_step_action_use_gap_mse": action_use_gap_mse,
        "videos": videos,
    }
    write_json(args.output_dir / "vizdoom-baselines.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_copy_last_win and report["status"] != "passed":
        raise SystemExit("Development model did not beat copy-last at one step")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Resolved Hydra config.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=32)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--action-shuffle-seed", type=int, default=31415)
    parser.add_argument("--diffusion-seed", type=int, default=3407)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-perceptual", action="store_true")
    parser.add_argument("--require-copy-last-win", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
