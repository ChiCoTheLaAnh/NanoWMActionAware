# Action-Aware NanoWM

Preventing action-conditioning collapse in low-compute game world models.

## Overview

Action-Aware NanoWM is a research project investigating whether a compact diffusion world model can be made genuinely responsive to its action input. A video predictor can generate plausible futures while effectively ignoring the action that is supposed to condition them; this project calls that failure mode **action-conditioning collapse**.

The project adapts NanoWM-S/2 to VizDoom Basic and adds an inverse-dynamics regularizer. The auxiliary head must infer the action from the predicted transition—without seeing the action embedding directly—so the generated future is encouraged to preserve action-specific information.

The core research question is:

> Can an auxiliary inverse-dynamics objective make a small diffusion world model genuinely respond to actions while preserving video quality and improving planning?

This repository builds on the [official Nano World Model project](https://github.com/simchowitzlabpublic/nano-world-model). It is a low-compute reproducibility study, not a claim of state-of-the-art game-playing performance.

## Project Status

**Status:** Planning and kickoff

The research plan and progress-tracking structure are in place. The upstream NanoWM code, runtime environment, VizDoom dataset, trained models, and evaluation results have not yet been integrated or produced.

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

Installation instructions are pending integration of the official NanoWM codebase and validation of a pinned environment. Until that work is complete, commands from upstream should not be treated as tested instructions for this repository.

## Reproduction

Reproduction commands, dataset manifests, checkpoint locations, resolved configurations, and expected metric outputs will be added only after their corresponding pipeline stages pass the documented exit gates.

Do not collect the full dataset or begin long training runs until frame/action alignment, storage, loading, and the four-frame DataLoader batch have been verified.

