# Action-Aware NanoWM Progress Tracker

This file is the only source of truth for execution status. The [project plan](docs/PROJECT_PLAN.md) defines intent and experimental design; status changes belong here and require an artifact, metric, log, or explicit decision as evidence.

## Status

| Field | Value |
| --- | --- |
| Last updated | 2026-07-14 |
| Overall status | Kickoff documentation complete; engineering not started |
| Current phase | Week 0 — repository bootstrap |
| Active milestone | Integrate upstream NanoWM and complete Day 1 |
| Blocked | No |

Legend: `[ ]` not started · `[x]` complete. Add an evidence link before checking an engineering or experimental task.

## Next Up

1. Integrate the official NanoWM codebase while keeping this repository as `origin` and adding NanoWM as `upstream`.
2. Reproduce the upstream environment and import smoke test.
3. Pin the verified dependency versions.
4. Install and launch VizDoom Basic.
5. Collect 10 pilot episodes and save one rendered episode.

Do not collect the full dataset until frame/action alignment and a four-frame DataLoader batch pass their exit gates.

## Documentation Foundation

- [x] Create a project-specific `README.md`.
- [x] Convert the complete source plan to `docs/PROJECT_PLAN.md`.
- [x] Create this detailed progress tracker.
- [x] Remove the duplicate plaintext plan after verifying the Markdown conversion.

Evidence: [README](README.md) · [Project plan](docs/PROJECT_PLAN.md) · [Progress tracker](PROGRESS.md)

## First 72 Hours

### Day 1 — Environment and Initial Collection

- [ ] Integrate/fork the official NanoWM codebase and configure the upstream remote.
- [ ] Create the Colab or personal-GPU environment from the upstream environment definition.
- [ ] Pin the verified dependencies.
- [ ] Pass configuration and import smoke tests.
- [ ] Install VizDoom.
- [ ] Launch VizDoom Basic.
- [ ] Collect 10 pilot episodes.
- [ ] Render one pilot episode video.

**Exit gate:** the environment starts reliably, VizDoom Basic runs, and 10 raw episodes plus one rendered video exist.

**Expected artifacts:** pinned environment definition, smoke-test log, raw pilot episodes, and rendered episode video.

### Day 2 — Alignment and Data Loading

- [ ] Save frames, actions, rewards, terminal flags, and seeds.
- [ ] Plot three random episodes.
- [ ] Check the action distribution.
- [ ] Verify that action $a_t$ maps observation $o_t$ to $o_{t+1}$.
- [ ] Implement `VizDoomDataSource`.
- [ ] Load one batch containing one context frame and three future frames.

**Exit gate:** a DataLoader batch returns the expected four frames and correctly aligned action sequence.

**Expected artifacts:** verified batch dump, dataset visualization, action-distribution plot, and alignment report.

### Day 3 — Model Smoke Test

- [ ] Run frozen VAE encode/decode.
- [ ] Create a fixed set of 32 clips.
- [ ] Overfit the 32 clips for 500–1,000 steps.
- [ ] Render ground truth versus prediction.
- [ ] Save a checkpoint.
- [ ] Restart the runtime and resume training from the checkpoint.

**Exit gate:** generated clips contain motion, the tiny set can be overfit, and checkpoint resume continues valid training.

**Expected artifacts:** VAE reconstruction, training log, comparison video, and resumable checkpoint.

## Eight-Week Checklist

### Week 1 — Environment and Data

- [ ] Complete all Day 1 tasks.
- [ ] Collect 100 pilot episodes only after the first 10 validate correctly.
- [ ] Implement the HDF5 episode format.
- [ ] Visualize frame sequences and action distributions.
- [ ] Verify frame/action alignment and terminal handling.
- [ ] Implement and register `VizDoomDataSource`.
- [ ] Add the VizDoom Basic Hydra dataset configuration.

**Exit gate:** a DataLoader batch returns one context frame, three future frames, and the correctly aligned action sequence.

**Expected artifacts:** collector, DataSource, pilot dataset, seed metadata, and visualization notebook.

### Week 2 — Baseline Pipeline

- [ ] Run VAE encode/decode checks.
- [ ] Overfit 32 fixed clips.
- [ ] Implement copy-last-frame evaluation.
- [ ] Implement zero-action and shuffled-action evaluation.
- [ ] Train a 5,000–10,000-step development NanoWM baseline.
- [ ] Render correct-action and shuffled-action futures.
- [ ] Validate save/resume behavior.

**Exit gate:** the development model beats copy-last-frame at one-step prediction.

**Expected artifacts:** development checkpoint, ground-truth-versus-prediction video, resolved configuration, and smoke-test metrics.

### Week 3 — Full Standard NanoWM

- [ ] Collect 75,000–150,000 transitions.
- [ ] Finalize train, validation, and test seed manifests.
- [ ] Freeze a shared evaluation set.
- [ ] Train the additive-action baseline for 20,000–30,000 steps.
- [ ] Measure PSNR, SSIM, LPIPS, and Action Use Gap.
- [ ] Store the complete resolved configuration with the run.

**Exit gate:** validation LPIPS is stable and qualitative futures show meaningful scene motion.

**Expected artifacts:** full dataset, manifests, fixed benchmark JSON, standard checkpoint, and training report.

### Week 4 — Action-Aware Method

- [ ] Recover the predicted clean latent from the v-prediction output.
- [ ] Implement the inverse-dynamics head.
- [ ] Verify structurally that the head cannot access the action embedding.
- [ ] Add action-loss weight, warm-up, ramp, and noisy-timestep controls.
- [ ] Overfit an action-balanced clip set.
- [ ] Run $\lambda_{\text{action}} = 0.05$.
- [ ] Run $\lambda_{\text{action}} = 0.10$.
- [ ] Run $\lambda_{\text{action}} = 0.25$ if compute permits.
- [ ] Produce per-class metrics and a confusion matrix.

**Exit gate:** inverse-action accuracy exceeds the 33.3% random baseline.

**Expected artifacts:** action-aware model code, lambda-ablation results, resolved configurations, checkpoints, and confusion matrix.

### Week 5 — Core Final Experiments

- [ ] Select the best lambda using development results.
- [ ] Train three final seeds for the standard additive baseline.
- [ ] Train three final seeds for the selected action-aware model.
- [ ] Evaluate correct, zero, and shuffled actions on identical samples.
- [ ] Evaluate visual quality at one and three steps.
- [ ] Compute bootstrap confidence intervals.
- [ ] Produce the final comparison table and statistical plots.

**Exit gate:** action sensitivity improves while LPIPS increases by no more than 10% relative to the standard baseline.

**Expected artifacts:** six final runs, best checkpoints, metrics JSON, comparison table, confidence intervals, and main plots.

### Week 6 — Counterfactual Benchmark

- [ ] Build 100–200 fixed branching states.
- [ ] Verify identical observations after replaying each shared action prefix.
- [ ] Record real left, right, and shoot branches.
- [ ] Generate corresponding standard and action-aware predictions.
- [ ] Compute counterfactual direction and magnitude metrics.
- [ ] Render qualitative grids and branching videos.
- [ ] Analyze success and failure cases, including ambiguous shots.

**Exit gate:** predicted action deltas correlate with real action deltas better than the standard baseline.

**Expected artifacts:** branch metadata, counterfactual dataset, metrics JSON, comparison grid, videos, and failure analysis.

### Week 7 — Planning

- [ ] Train and validate the reward/goal scorer on real VAE latents.
- [ ] Freeze one scorer for all planner comparisons.
- [ ] Implement exhaustive horizon-four MPC over 81 action sequences.
- [ ] Add candidate chunking and 10-step development DDIM sampling.
- [ ] Evaluate random and scripted policies.
- [ ] Evaluate standard and action-aware NanoWM planners.
- [ ] Measure return, success, shots, episode length, and latency.
- [ ] Attempt categorical CEM only if the MVP is complete and compute remains.

**Exit gate:** the proposed planner beats random; improvement over the standard NanoWM planner remains the target result.

**Expected artifacts:** frozen scorer, planner code, results JSON, benchmark table, latency measurements, and episode videos.

### Week 8 — Demo and Release

- [ ] Build the branching-future demo.
- [ ] Clean and verify the Colab notebook.
- [ ] Document exact reproduction commands.
- [ ] Release dataset split and counterfactual metadata.
- [ ] Publish final checkpoints and resolved configurations.
- [ ] Write the 4–6 page technical report.
- [ ] Record a 60–90 second demonstration video.
- [ ] Run a clean-clone reproduction check on one accessible GPU.

**Exit gate:** another researcher can reproduce the main prediction, action-sensitivity, counterfactual, and planning results.

**Expected artifacts:** public repository, Colab notebook, demo, checkpoints, evaluation results, technical report, and demonstration video.

## Experiment Matrix

Use identical model size, VAE, data splits, training steps, optimizer, learning rate, diffusion target, noise schedule, evaluation samples, and final random seeds unless a row explicitly changes one variable.

| ID | Model / action input | Auxiliary objective | Priority | Status | Run/config | Checkpoint | Results |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | Copy-last-frame / none | None | Required | Not started | — | N/A | — |
| E1 | NanoWM / zero or shuffled | None | Required | Not started | — | — | — |
| E2 | NanoWM additive / correct | None | Required | Not started | — | — | — |
| E3 | NanoWM additive / correct | Inverse dynamics, $\lambda=0.05$ | Required | Not started | — | — | — |
| E4 | NanoWM additive / correct | Inverse dynamics, $\lambda=0.10$ | Required | Not started | — | — | — |
| E5 | NanoWM additive / correct | Inverse dynamics, $\lambda=0.25$ | Compute permitting | Not started | — | — | — |
| E6 | NanoWM FiLM / correct | Best inverse-dynamics setting | Stretch | Not started | — | — | — |

## Success Metrics

| Metric | Minimum credible criterion | Target | Current | Evidence |
| --- | --- | --- | --- | --- |
| Action Use Gap | Proposed model exceeds standard baseline | Positive with confidence interval excluding zero | Not measured | — |
| LPIPS degradation | No more than 10% relative degradation | Preserve baseline quality | Not measured | — |
| Inverse-action accuracy | Above 33.3% random baseline | At least 60% | Not measured | — |
| Counterfactual consistency | Better than standard baseline | At least 20% relative improvement | Not measured | — |
| Planning success | Beat random or provide rigorous failure analysis | At least 10 percentage points above standard planner | Not measured | — |

## Blockers

No active blocker. Upstream integration and environment validation are the next unstarted tasks.

When adding a blocker, record its date, affected milestone, owner, attempted mitigations, and the condition required to clear it.

## Risks Under Watch

| Risk | Early signal | Planned response | State |
| --- | --- | --- | --- |
| GPU memory | Batch size 1 cannot finish a forward pass | Accumulate gradients, checkpoint activations, cache latents, then reduce model size | Not assessed |
| Current-frame copying | Acceptable PSNR but incorrect motion | Increase frame skip and rebalance action transitions | Not assessed |
| Ignored actions | Correct and shuffled actions produce similar futures | Add inverse loss; reserve FiLM for stretch ablation | Not assessed |
| Inverse-head shortcut | High action accuracy without changed video | Prohibit direct action-embedding access | Not assessed |
| Ambiguous shoot action | Low shoot recall | Report per-class results and increase successful-shot samples | Not assessed |
| Weak reward scorer | Poor validation performance on real latents | Validate and freeze the scorer before planner evaluation | Not assessed |
| Slow planning | One action requires several minutes | Use 10 DDIM steps, chunking, and horizon-four enumeration | Not assessed |
| Runtime interruption | Training progress is lost | Checkpoint every 1,000–2,000 steps and test resume early | Not assessed |

## Decisions

| Date | Decision | Rationale |
| --- | --- | --- |
| 2026-07-14 | Use English for repository documentation | Supports a public research release |
| 2026-07-14 | Keep task status only in `PROGRESS.md` | Avoid conflicting status across documents |
| 2026-07-14 | Use VizDoom Basic and three discrete actions | Keeps data, planning, and compute tractable |
| 2026-07-14 | Validate 100 pilot episodes before full collection | Prevents scaling an alignment or format error |
| 2026-07-14 | Keep the inverse head isolated from the action embedding | Prevents a conditioning-copy shortcut |

## Evidence Index

| Artifact | Location | Status |
| --- | --- | --- |
| Project README | [README.md](README.md) | Complete |
| Detailed research plan | [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | Complete |
| Progress tracker | [PROGRESS.md](PROGRESS.md) | Complete |
| Environment smoke-test log | — | Not produced |
| Pilot episode video | — | Not produced |
| Dataset validation report | — | Not produced |
| Baseline checkpoint | — | Not produced |
| Action-aware checkpoint | — | Not produced |
| Final metrics | — | Not produced |

