# Stage 3 From Stage 2 Step 5000, H50

## Confirmed Production Startup

At 2026-09-11 15:32:42 +08:00 the first 20 successful H50 updates passed
production startup verification; 32 updates were observed subsequently. Supervisor
PID 3306518 and initial launcher PID 3306742 continue in the background toward
150000 updates. First launch was 15:25:42 +08:00. All 696 inherited VLM/Query/
Slot/Flow tensors exactly matched Stage 2, and all 149 Expert tensors matched
base at final dtype. Model serialization verified source/runtime H50. Base
Expert SHA256: `38877236134df1e54c9493e3178decc8c1b69ba2d4b37359b7c826b719940d97`.

First-20 total-loss formula maximum absolute error: 4.4107437e-7. Mean raw
losses over those windows: AR 0.180672, Slot 0.483644, Optical Flow 0.0450170,
FM 0.604700 (inactive omitted losses contribute zero to this window mean).
Mean optimizer-window time for updates 2-20: 4.23068 seconds. Observed peak
allocated/reserved GPU memory through update 32: 39.3543/76.7813 GiB. No skipped
updates, nonfinite reported component gradients or W&B failures in the checked
startup windows. These are startup measurements, not full-run guarantees.

W&B: https://wandb.ai/jumbo3r-zhejiang-university/zr0-three-stage-formal/runs/stage3_from_stage2_step5000_h50_slot0p1_flow1_20260911-stage3_joint-retry000

Evidence: `stage3_joint/retries/attempt-000/startup_verified.json`,
`component_sources_verified.json`, `checkpoint_serialization_verified.json`,
`resolved_dataset_manifest.json`, `training_metrics.jsonl` and supervisor events.
No Stage 3 H50 checkpoint exists yet at startup; first full save is update 2000.
Same-stage H50 recovery, epoch-boundary behavior and full 150000-update completion
remain unobserved. Prior checkpoints are unchanged. Task files are staged; no commit.


Created 2026-09-11 Asia/Shanghai. Owner: lq. Purpose: restart formal Stage 3
with the original ZR-0 action prediction horizon and the requested loss weights.
Target: 150000 successful optimizer updates from zero. No subsequent experiment
is queued. This is cross-stage initialization; Stage 2 optimizer, scheduler,
RNG, sampler cursor and exposure state are not resumed.

## Sources and Execution

VLM (vision encoder, merger/projector, language model and tied embeddings),
32 Difference Queries, structured Slot Head and dense-regression-v1 Optical
Flow Head inherit from:
`/opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260909/commit_3eefb602_flow10/recovery_checkpoints/stage2_aux/step-005000-attempt-000/latest-model-optimizer-lr`.

Only Action Expert, including state/action encoders and decoder, loads from
`/opt/data/private/lq/models/ZR-0`. Source and runtime horizon are both 50.
All five components train, with no LoRA or frozen submodules. The first 16
Queries retain Slot roles, the last 16 retain Flow roles; all 32 condition FM.
Actual loaded tensors are compared exactly at final dtype before update 1.
No whole-base VLM reload, mismatch suppression, or random Expert layers.

Baseline runtime commit: `3eefb602417d3bd4b20bef2b47660b404aefb565`, reused from
the existing runtime snapshot. Current-worktree adapters are explicitly loaded
by `scripts/train_vla_loss_weight_resume.py`; hashes are sealed in
`code_identity.json`. Unrelated staged/unstaged changes are retained and are
not loaded into the pinned training loop. No commit is created.

Configuration: `configs/three_stage_formal_stage3_h50_20260911.json`.
Output: `/opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260911/stage3_from_stage2_step5000_h50_slot0p1_flow1`.

```bash
PYTHONNOUSERSITE=1 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python scripts/run_stage3_h50.py --mode prepare
PYTHONNOUSERSITE=1 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -u scripts/run_stage3_h50.py --mode run
```

The complete Accelerate commands and all parser defaults are materialized before
launch in `launch_plan.json` and each attempt's `expanded_options.json` and
`experiment.md`. Repeated prepare refuses existing output. An exclusive lock
and exclusive start record prevent duplicate launches.

## Data and Horizon

Reuse saved audit `outputs/three_stage_validation_20260908/flow_exclusion_108/audit_snapshot.json`,
SHA256 `19e7b8b5ddc213b57d41660524a89b935ce8870c7432498e90d1dbfcb9753b9f`.
Only saved identities are checked; original payloads are not re-audited.

| Dataset under /opt/data/private/lq/datasets | Frozen Stage 3 train samples | Statistics content hash |
| --- | ---: | --- |
| droid_1.0.1_stage05_full_95658_20260831 | 19823040 | 9fe87371384944aee47192b37d583a913e8989139aaac7556734ea21857d7966 |
| molmoact_dataset_household-v3_stage05 | 794199 | 3955199f69c0cd15b2ce2de22c3cff2c6ee0fc4e7f24933c8a312ed3591d8caf |
| molmoact_dataset_tabletop-v3_stage05 | 310743 | f41fa9d88d38bc762d5efb98dc3ef65c933eefdb04f1163bfb1e8049e7e1d125 |
| RH20T-v30_stage05 | 3131833 | 294db9e546545a754d4cc936de16fe7699197aa44faeb73a7a205ed6ab56ea27 |

Total 24059815 quality/dual-camera admitted training frames. Natural sample-equal
frame mixture uses ratios 1/1/1/1, weighted by dataset populations. No LIBERO,
validation or test set. Each dataset retains its independent train-only
state/action quantile statistics from its own `joint/stats.json`; full DROID
does not use partial DROID statistics. The saved normalization population and
values remain fixed when horizon changes; no statistics are recomputed.

Original sidecars remain labeled H10, with their hashes intact. The explicit
H50 adapter validates all original sidecar contracts while accepting that
recorded source horizon. Runtime action chunks use 50 steps from the original
canonical reader. Previously FM-ineligible frames with valid actions in steps
11-50 are checked lazily using current canonical state validity. Invalid states,
invalid actions and episode-tail padding remain masked. A read of saved
validity arrays found 1675 potential extra FM frames in 269 DROID episodes,
zero in the other three datasets; this is an upper bound before state validity.
Extending existing FM windows references zero additional valid statistics-action
rows in those arrays. The adaptation is included in the saved dataset contract.

AR reads each sampled row's `train_data` when available. AR, Slot, Flow and FM
use independent masks; missing auxiliary targets never remove a Stage 3 sample.
Retain the approved 108 RH20T Flow exclusions and DROID episode mapping. Slot
and Flow labels, internal losses, Flow delta=20 frames and units are unchanged.

Canonical state: absolute [x,y,z,roll,pitch,yaw,gripper_open]; action: relative
native-next-step end-effector delta plus binary gripper open. Seven meaningful
dimensions, padded to 64; q01/q99 normalization and existing clipping [-15,15].
Angles in radians, translation in metres. Window=1, chunk/prediction horizon=50.
DROID 15 FPS, other datasets 10 FPS. Execution horizon/control rollout: not
applicable to offline training. Flow never uses action statistics.

## Images and Optimization

First external plus wrist RGB: DROID/RH20T 320x180 source images;
Household/Tabletop 640x480. Each view resized directly to 224x224 bicubic,
without aspect preservation, crop, letterbox or geometric augmentation.
Processor mean/std=0.5 per channel. Ordered separate views, current images
only; no future labels/images in context. RH20T wrist offset <=100 ms.
Text max length=1024. Retain mRoPE, DeepStack, RMSNorm and teacher forcing.
No evaluation is scheduled; later evaluation must state its own image pipeline.

Four NVIDIA A800-SXM4-80GB GPUs, BF16, SDPA, ZeRO-2, VLM gradient checkpointing.
Global batch = 4 GPUs * 16 samples/GPU * 2 accumulation steps = 128.
Seed=42, epoch cap=16, 4 workers/rank, prefetch=2, PyAV threads=1,
OMP/MKL threads=4. Expected successful-window exposures=19200000 samples;
actual exposure and distributed padding are recorded by the existing tracker.

Every component parameter group uses AdamW, peak LR=1e-5, minimum LR=1e-6,
betas=(0.9,0.95), epsilon=1e-6, weight decay=0.01, clip=1.0, cosine schedule
over 150000 updates with 12000 warmup updates (8%). No LR multipliers.
Warmup 8%/epsilon 1e-6 follow reviewed scripts; paper values are 5%/1e-8.
Stage LR is the project setting, not a claim of exact pretraining replication.

```text
L = 1.0 * L_AR + 0.1 * L_slot + 1.0 * L_optical_flow + 5.0 * L_FM
```

AR is token CE. Slot is the existing structured multitask loss. Flow is the
existing dense regression loss. FM is masked velocity regression. All available
objectives backpropagate. Head inactivity protects its native optimizer state;
shared components update for valid remaining objectives. Zero-valued loss does
not imply absent supervision. Stop after 20 consecutive skipped windows.

W&B is required online: project `zr0-three-stage-formal`, group
`stage3_from_stage2_step5000_h50_slot0p1_flow1_20260911`; run names add
`-stage3_joint-retryNNN`. Actual run URLs are stored in `wandb_identity.json`.
Metrics every update include raw/weighted losses, per-component gradients,
LR, supervision/update flags, throughput and GPU memory. Keep batched metric
reductions, inactive-only expensive component diagnostics and fast data resume.

Full ZeRO-2 checkpoint every 2000 updates and at completion, with immutable
copies under `recovery_checkpoints/stage3_joint/step-NNNNNN-attempt-NNN`.
Automatic retries: at most three, delays 60/120/240 seconds, using the latest
complete same-stage checkpoint, unchanged H50 contract and exact restored
state verification. W&B failures cannot fall back to offline mode. UUID GPU
gate before every launch; no automatic batch reduction. Manual stop disables
retries. No loss-based early stopping, validation interval or best-model metric.

## Verification and Results

Before launch: CPU regression tests for explicit initialization weights, H50
FM masks, newly available supervision, invalid state, tail padding, strict
failure propagation and old-path compatibility; complete source checkpoint,
cached preparation and parsed command checks. At startup: exact component
source checks, serialization provenance, correct fresh counters and first
20 successful production updates. Record actual results below. H50 full-model
throughput/memory and same-stage H50 recovery are unverified at document creation.
No extra diagnostic optimizer update or fault injection is scheduled.

CPU verification completed before launch: 51 focused regression cases passed
(24.84 seconds); Python compilation and whitespace checks passed. The real
Stage 2 archive, native four-rank states and saved audit binding passed.
All four real datasets constructed successfully with H50 runtime contracts,
with lengths 19823040 / 794199 / 3131833 / 310743 and unchanged independent
statistics. No original payload audit, statistics regeneration or GPU update
was performed by these checks.
