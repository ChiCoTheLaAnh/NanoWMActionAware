# Action-Aware NanoWM Progress Tracker

This file is the only source of truth for execution status. The [project plan](docs/PROJECT_PLAN.md) defines intent and experimental design; status changes belong here and require an artifact, metric, log, or explicit decision as evidence.

## Status

| Field | Value |
| --- | --- |
| Last updated | 2026-07-16 |
| Overall status | Week 1 data exit gate passed; Day 3 smoke implementation is ready for a Colab GPU run |
| Current phase | Week 2 — baseline pipeline |
| Active milestone | Day 3 — model smoke test |
| Blocked | No |

Legend: `[ ]` not started · `[x]` complete. Add an evidence link before checking an engineering or experimental task.

## Next Up

1. Run frozen VAE encode/decode on a verified four-frame VizDoom batch.
2. Freeze a deterministic set of 32 training clips.
3. Overfit the 32 clips for 500–1,000 steps and render predictions.
4. Save a checkpoint, restart the runtime, and resume the full trainer state.
5. Record the reconstruction, training log, comparison video, and resume evidence.

The Day 3 implementation is available through `experiment=vizdoom_smoke`,
`src/scripts/vizdoom_model_smoke.py`, and `notebooks/02_colab_day3.ipynb`.
The checklist remains open until the fresh-runtime GPU run produces passing
evidence under `reports/evidence/day3/`.

Do not begin a long development run until the VAE, tiny-set overfit, generated-motion, and checkpoint-resume checks pass.

## Documentation Foundation

- [x] Create a project-specific `README.md`.
- [x] Convert the complete source plan to `docs/PROJECT_PLAN.md`.
- [x] Create this detailed progress tracker.
- [x] Remove the duplicate plaintext plan after verifying the Markdown conversion.

Evidence: [README](README.md) · [Project plan](docs/PROJECT_PLAN.md) · [Progress tracker](PROGRESS.md)

## First 72 Hours

### Day 1 — Environment and Initial Collection

- [x] Integrate/fork the official NanoWM codebase and configure the upstream remote.
- [x] Create the Colab or personal-GPU environment from the upstream environment definition.
- [x] Pin the verified dependencies.
- [x] Pass configuration and import smoke tests.
- [x] Install VizDoom in an isolated local pilot environment.
- [x] Launch VizDoom Basic and verify deterministic seed replay.
- [x] Collect 10 pilot episodes with seeds 42–51.
- [x] Render one pilot episode video.

**Exit gate:** the environment starts reliably, VizDoom Basic runs, and 10 raw episodes plus one rendered video exist.

**Expected artifacts:** pinned environment definition, smoke-test log, raw pilot episodes, and rendered episode video.

**Current evidence:** upstream `2ee3c35` · [Colab smoke report](reports/evidence/day1/smoke-report.json) · [pip freeze](reports/evidence/day1/pip-freeze.txt) · [pilot manifest](reports/evidence/day1/pilot-manifest.json) · [pilot video](reports/evidence/day1/pilot_episode_00000.mp4)

**Gate state:** passed on a fresh Colab T4 runtime. The configuration smoke test passed at project revision `23fb917`; 10 episodes with seeds 42–51 and one 30-frame video were produced. The upstream SHA is recorded in the Colab smoke report because the Colab clone did not configure an `upstream` remote.

### Day 2 — Alignment and Data Loading

- [x] Save frames, actions, rewards, terminal flags, and seeds.
- [x] Plot three representative episodes.
- [x] Check the action distribution.
- [x] Verify that action $a_t$ maps observation $o_t$ to $o_{t+1}$.
- [x] Implement `VizDoomDataSource`.
- [x] Load one batch containing one context frame and three future frames.

**Exit gate:** a DataLoader batch returns the expected four frames and correctly aligned action sequence.

**Expected artifacts:** verified batch dump, dataset visualization, action-distribution plot, and alignment report.

**Current evidence:** [dataset validation](reports/evidence/day2/dataset-validation.json) · [alignment report](reports/evidence/day2/alignment-report.json) · [batch summary](reports/evidence/day2/batch-summary.json) · [episode visualization](reports/evidence/day2/episode-visualization.png) · [action distribution](reports/evidence/day2/action-distribution.png) · [batch preview](reports/evidence/day2/batch-preview.png)

**Gate state:** passed with 100 episodes (seeds 42–141), 4,459 stored transitions, 4,359 replay-verified observation-to-observation transitions, and a `[2, 4, 3, 256, 256]` video / `[2, 4, 3]` action batch. All three actions are balanced within 1.3 percentage points.

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

- [x] Complete all Day 1 tasks.
- [x] Collect 100 pilot episodes only after the first 10 validate correctly.
- [x] Implement the HDF5 episode format.
- [x] Visualize frame sequences and action distributions.
- [x] Verify frame/action alignment and terminal handling.
- [x] Implement and register `VizDoomDataSource`.
- [x] Add the VizDoom Basic Hydra dataset configuration.

**Exit gate:** a DataLoader batch returns one context frame, three future frames, and the correctly aligned action sequence.

**Expected artifacts:** collector, DataSource, pilot dataset, seed metadata, and visualization notebook.

**Gate state:** passed. The 100-episode pilot is stored in the gitignored data directory; the tracked Day 2 reports contain schema, checksum, seed, action-distribution, replay-alignment, Hydra/DataLoader, and visualization evidence.

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

No active blocker. Day 2 schema, alignment, visualization, and DataSource work
is complete. Day 3 is awaiting execution on a Colab GPU; this is pending work,
not a blocker.

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
| Colab package drift | Binary ABI or dependency-check failure after a runtime update | Preserve Colab-owned binary packages, pin only missing project packages, and capture `pip freeze` | Mitigated for Day 1; monitor |

## Decisions

| Date | Decision | Rationale |
| --- | --- | --- |
| 2026-07-14 | Use English for repository documentation | Supports a public research release |
| 2026-07-14 | Keep task status only in `PROGRESS.md` | Avoid conflicting status across documents |
| 2026-07-14 | Use VizDoom Basic and three discrete actions | Keeps data, planning, and compute tractable |
| 2026-07-14 | Validate 100 pilot episodes before full collection | Prevents scaling an alignment or format error |
| 2026-07-14 | Keep the inverse head isolated from the action embedding | Prevents a conditioning-copy shortcut |
| 2026-07-14 | Rebase project documentation onto NanoWM upstream `2ee3c35` | Preserves upstream lineage and future synchronization |
| 2026-07-14 | Put project scripts under upstream-native `src/scripts/` | Keeps additions consistent with the integrated codebase |
| 2026-07-14 | Keep Day 1 partially open until a fresh Colab smoke run passes | Local VizDoom evidence does not validate the GPU training runtime |
| 2026-07-15 | Preserve Colab's binary and platform package stack | Downgrading NumPy and h5py caused an ABI mismatch in the Python 3.12 runtime |
| 2026-07-15 | Accept the fresh T4 smoke run as the Day 1 environment baseline | Hydra composition, strict dependency checking, deterministic collection, and evidence generation passed without model download or training |
| 2026-07-15 | Treat stored action $a_t$ as the transition from observation $o_t$ to $o_{t+1}$ | Matches the collector and NanoWM's internal one-step action shift; replay verified all 100 pilot episodes |
| 2026-07-15 | Use letterbox resizing for VizDoom and keep collection frame skip outside the loader | Preserves the 4:3 image geometry while avoiding a second temporal subsampling step |

## Evidence Index

| Artifact | Location | Status |
| --- | --- | --- |
| Project README | [README.md](README.md) | Complete |
| Detailed research plan | [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | Complete |
| Progress tracker | [PROGRESS.md](PROGRESS.md) | Complete |
| Colab notebook | [notebooks/01_colab_day1.ipynb](notebooks/01_colab_day1.ipynb) | Validated on a fresh Colab T4 runtime |
| Day 3 Colab notebook | [notebooks/02_colab_day3.ipynb](notebooks/02_colab_day3.ipynb) | Ready to run; GPU evidence not yet produced |
| Local VizDoom smoke report | [reports/evidence/day1/local-vizdoom-smoke.json](reports/evidence/day1/local-vizdoom-smoke.json) | Complete |
| Pilot manifest | [reports/evidence/day1/pilot-manifest.json](reports/evidence/day1/pilot-manifest.json) | Complete |
| Pilot episode video | [reports/evidence/day1/pilot_episode_00000.mp4](reports/evidence/day1/pilot_episode_00000.mp4) | Complete |
| Colab environment smoke-test report | [reports/evidence/day1/smoke-report.json](reports/evidence/day1/smoke-report.json) | Complete |
| Colab dependency freeze | [reports/evidence/day1/pip-freeze.txt](reports/evidence/day1/pip-freeze.txt) | Complete |
| Day 2 dataset validation | [reports/evidence/day2/dataset-validation.json](reports/evidence/day2/dataset-validation.json) | Passed on 100 episodes |
| Day 2 alignment report | [reports/evidence/day2/alignment-report.json](reports/evidence/day2/alignment-report.json) | Passed on 4,359 aligned transitions |
| Day 2 DataLoader batch | [reports/evidence/day2/batch-summary.json](reports/evidence/day2/batch-summary.json) | Passed: 1 context + 3 future frames |
| Day 2 visual evidence | [reports/evidence/day2/episode-visualization.png](reports/evidence/day2/episode-visualization.png) · [action distribution](reports/evidence/day2/action-distribution.png) · [batch preview](reports/evidence/day2/batch-preview.png) | Complete |
| Baseline checkpoint | — | Not produced |
| Action-aware checkpoint | — | Not produced |
| Final metrics | — | Not produced |
