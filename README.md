# Action-Aware NanoWM

Preventing action-conditioning collapse in low-compute game world models.

## Overview

Action-Aware NanoWM is a research project investigating whether a compact diffusion world model can be made genuinely responsive to its action input. A video predictor can generate plausible futures while effectively ignoring the action that is supposed to condition them; this project calls that failure mode **action-conditioning collapse**.

The project adapts NanoWM-S/2 to VizDoom Basic and adds an inverse-dynamics regularizer. The auxiliary head must infer the action from the predicted transition—without seeing the action embedding directly—so the generated future is encouraged to preserve action-specific information.

The core research question is:

> Can an auxiliary inverse-dynamics objective make a small diffusion world model genuinely respond to actions while preserving video quality and improving planning?

This repository builds on the [official Nano World Model project](https://github.com/simchowitzlabpublic/nano-world-model). It is a low-compute reproducibility study, not a claim of state-of-the-art game-playing performance.

## Project Status

**Status:** Week 1 data gate complete; Day 3 model smoke implementation is ready for a Colab GPU run

The official NanoWM source is integrated at upstream revision `2ee3c35`. The
VizDoom collector, HDF5 schema, DataSource, Hydra dataset config, and four-frame
DataLoader path are implemented. The 100-episode pilot set (seeds 42–141) has
4,459 stored transitions; replay validation confirmed that every nonterminal
stored action maps its observation to the following observation. A fresh Colab T4
runtime passed the Day 1 dependency, import, and configuration smoke tests at
project revision `23fb917`. No model, checkpoint, or training result has been
produced.

The Day 3 path now includes a deterministic 32-clip manifest, a dedicated
NanoWM-S/2 smoke profile, VAE reconstruction and prediction evidence tools,
and full-state checkpoint-resume verification. These are implemented but have
not yet passed the GPU exit gate; open
[`notebooks/02_colab_day3.ipynb`](notebooks/02_colab_day3.ipynb) in a fresh
Colab GPU runtime to execute them.

- Detailed design: [Project Plan](docs/PROJECT_PLAN.md)
- Current work and evidence: [Progress Tracker](PROGRESS.md)

## Fixed Scope

| Area | Decision |
| --- | --- |
| Primary environment | VizDoom Basic |
| Stretch environment | VizDoom Take Cover |
| World model | NanoWM-S/2, approximately 40M parameters |
| Latent representation | Frozen Stable Diffusion VAE |
| Input | 1 context frame + 3 future frames at 256 × 256 |
| Actions | Move left, move right, shoot |
| Baseline conditioning | Additive action injection |
| Proposed objective | Diffusion loss + inverse-dynamics loss |
| Compute target | One Colab GPU or one personal GPU |
| Planned duration | 8 weeks |

The proposed training objective is:

$$
\mathcal{L}_{\text{total}}
= \mathcal{L}_{\text{diffusion}}
+ \lambda_{\text{action}}\mathcal{L}_{\text{inverse}}.
$$

The inverse-dynamics head receives the context latent, predicted next latent, and their difference. It must never receive the action embedding directly.

## Planned Outputs

- An action-aligned VizDoom dataset with seed-based train, validation, and test splits.
- A standard additive-action NanoWM-S baseline.
- An action-aware NanoWM variant with inverse-dynamics regularization.
- A counterfactual benchmark branching the same state into left, right, and shoot futures.
- Action Use Gap, counterfactual consistency, visual-fidelity, and long-horizon evaluations.
- A discrete model-predictive-control planner and frozen reward scorer.
- A branching-future demo, Colab notebook, checkpoints, results, and technical report.

## Success Criteria

The minimum credible result requires a reproducible dataset and DataSource, trained standard and action-aware models, a larger Action Use Gap for the proposed model, no more than 10% LPIPS degradation, a counterfactual branch demo, and either a planning experiment or a rigorous planning failure analysis.

Target outcomes include:

- At least 60% inverse-action accuracy.
- A positive Action Use Gap whose confidence interval excludes zero.
- At least 20% relative improvement in counterfactual consistency.
- At least 10 percentage points of planning-success improvement over the standard NanoWM planner.

## Roadmap

| Week | Milestone |
| --- | --- |
| 1 | Environment, pilot data, and VizDoom DataSource |
| 2 | Pipeline smoke tests and development baseline |
| 3 | Full dataset and standard NanoWM baseline |
| 4 | Inverse-dynamics head and lambda sweep |
| 5 | Final multi-seed comparison |
| 6 | Counterfactual benchmark |
| 7 | Discrete model-predictive planning |
| 8 | Demo, reproducibility release, and report |

Milestone completion, exit gates, artifacts, experiment runs, and blockers are tracked only in [PROGRESS.md](PROGRESS.md).

## Installation

The supported training runtime will be Google Colab. Open
[`notebooks/01_colab_day1.ipynb`](notebooks/01_colab_day1.ipynb) in a GPU
runtime to install the Day 1 dependencies without replacing Colab's
CUDA-compatible PyTorch or binary scientific stack, compose a NanoWM Hydra
configuration, mount Google Drive, and collect the pilot data.

The notebook and [`requirements-colab.txt`](requirements-colab.txt) passed a
fresh Colab run with Python 3.12.13, a Tesla T4, PyTorch 2.11.0+cu128, NumPy
2.0.2, h5py 3.16.0, and VizDoom 1.3.0. See the tracked
[smoke report](reports/evidence/day1/smoke-report.json) and
[dependency freeze](reports/evidence/day1/pip-freeze.txt). This validates the
Day 1 configuration and collection path only, not model training.

The collector interface is:

```bash
python src/scripts/collect_vizdoom.py \
  --output-dir "$VIZDOOM_DATA_DIR/pilot" \
  --episodes 10 \
  --base-seed 42 \
  --policy uniform \
  --frame-skip 4 \
  --resolution 160x120 \
  --video-out reports/evidence/day1/pilot_episode_00000.mp4
```

Raw HDF5 episodes belong in Google Drive or another gitignored data directory.
The tracked [pilot manifest](reports/evidence/day1/pilot-manifest.json),
[Colab smoke report](reports/evidence/day1/smoke-report.json), and
[pilot video](reports/evidence/day1/pilot_episode_00000.mp4) provide the Day 1
evidence.

## Reproduction

Compose the Day 3 smoke profile without starting a run:

```bash
python src/main.py --cfg job \
  experiment=vizdoom_smoke dataset=game/vizdoom_basic model=nanowm_s2
```

The profile trains only a deterministic 32-clip subset, uses batch size 1 with
gradient accumulation 8, checkpoints every 100 steps, and caps the run at
1,000 optimizer steps. The Colab notebook performs the documented 500-step
initial run, fresh-runtime resume to step 600, VAE check, loss check, and
GT-versus-prediction render.

After the Day 3 gate passes, evaluate the development checkpoint against
copy-last, correct, zero, and shuffled actions with:

```bash
python src/evaluation/vizdoom_baselines.py \
  --config <run>/config.yaml \
  --checkpoint <run>/checkpoints/latest/<checkpoint>.ckpt \
  --subset-manifest reports/evidence/week2/fixed-validation-clips.json \
  --output-dir reports/evidence/week2/baselines \
  --require-copy-last-win
```

Development/final reproduction commands, checkpoint locations, resolved
training configurations, and expected metric outputs will be added only after
their corresponding pipeline stages pass the documented exit gates.

The data-alignment gate is complete. Do not collect the full dataset or begin
long training runs until the Day 3 VAE, tiny-set overfit, generated-motion, and
checkpoint-resume checks also pass.
