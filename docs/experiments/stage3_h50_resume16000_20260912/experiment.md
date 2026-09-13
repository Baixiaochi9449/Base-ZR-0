# Stage 3 H50 — resume from update 16000

Created 2026-09-12 Asia/Shanghai. Owner: lq. User authorization: resume the original H50 Stage 3 from 16000 with all training parameters unchanged. Continue to cumulative update 150000, leaving 134000 updates. The earlier process was stopped by the user at last logged update 16208; the 208 unsaved updates are replayed from the saved training cursor, not retained.

## Resume sources and implementation

All five model components (complete VLM vision encoder, merger/projector, language model; Query32; Slot Head; Optical Flow Head; complete Action Expert) and four-rank AdamW/ZeRO-2 states, scheduler, rank RNG, sampler cursor and exposure state resume from:
`/opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260911/stage3_from_stage2_step5000_h50_slot0p1_flow1/recovery_checkpoints/stage3_joint/step-016000-attempt-000/latest-model-optimizer-lr`.

Original H50 experiment: `/opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260911/stage3_from_stage2_step5000_h50_slot0p1_flow1`. Initial ancestry: VLM/Query/Slot/Flow from Stage 2 step5000 and Expert from `/opt/data/private/lq/models/ZR-0`; this resume loads every component from the saved Stage 3 checkpoint. All components remain trainable, without LoRA or detached FM conditioning. Both source and runtime action horizon remain 50.

Training runtime remains `3eefb602417d3bd4b20bef2b47660b404aefb565`. Only `scripts/run_stage3_h50.py` gains explicit same-stage resume orchestration; it reuses the pinned `attempt_item`, `RecoverySupervisor`, checkpoint validation and production training loop. The old launcher's sealed bytes are checked against commit `d17267551a79af3361f34f8b51f51b6b031a1ded`; all other sealed training adapters, runtime and original prepared artifacts must match exactly. Existing unrelated worktree changes remain preserved. Actual Git state is saved with prelaunch evidence. No commit is created.

Configuration: `/opt/data/private/lq/ZR-0/configs/three_stage_formal_stage3_h50_resume16000_20260912.json`. New output under the original experiment: `/opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260911/stage3_from_stage2_step5000_h50_slot0p1_flow1/resume_from16000_20260912`. Previous logs/checkpoints are preserved. Complete commands and defaults are written before every launch to `experiment.md` and `expanded_options.json`. The resume plan rejects any training hyperparameter difference from the original plan; only source/recovery flags, verification flags and output/W&B run identities change.

W&B remains online required in project `zr0-three-stage-formal`, group `stage3_from_stage2_step5000_h50_slot0p1_flow1_20260911`; new run prefix `stage3_h50_resume16000_20260912-stage3_joint`, attempts append `-retryNNN`. The new run avoids overlapping the earlier run's logged steps 16001–16208. URL is recorded after initialization in `wandb_identity.json`.

```bash
cd /opt/data/private/lq/ZR-0
source /opt/data/private/lq/miniconda3/etc/profile.d/conda.sh
conda activate ZR-0
PYTHONNOUSERSITE=1 python scripts/run_stage3_h50.py --config /opt/data/private/lq/ZR-0/configs/three_stage_formal_stage3_h50_resume16000_20260912.json --mode prepare
PYTHONNOUSERSITE=1 python -u scripts/run_stage3_h50.py --config /opt/data/private/lq/ZR-0/configs/three_stage_formal_stage3_h50_resume16000_20260912.json --mode run
```

`prepare` is CPU-only and refuses existing output; `run` requires prepared identities and an exclusive lock. No fresh initialization is permitted on the resume branch. Retain bounded retries, UUID resource gate, PyAV thread handling, full archives every 2000 updates and stop at 150000.

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


## Resume verification and results

No full dataset audit or statistics recomputation is requested. Saved checkpoint source/configuration/file identities and the existing native four-rank recovery contract are checked before launch. At the first resumed production window, exact component and saved optimizer/master-parameter/scheduler/RNG/cursor/exposure comparisons precede any optimizer update. The first 20 successful updates 16001–16020 verify unchanged loss coefficients, global batch, active-group update behavior, scheduler positions and online W&B. No separate diagnostic optimizer updates are performed.

This is a real resume; training samples are reached by the existing fast sampler skip, not decoding all 16000 historical windows. The startup checks do not establish fully equivalent uninterrupted trajectories or independently replay the next window for a forward tolerance comparison. Epoch boundaries remain outside this startup verification. No rollout or validation success metric is configured.

Status: prepared documentation; launch and real resume verification pending. Actual times, checkpoint, losses, resource usage and runtime findings are appended after startup and completion.

Prelaunch result (2026-09-12T19:46:35.366668+08:00): six CPU regression cases passed; source inventory (39 files), native four-rank checkpoint structure and the original sealed runtime/adapters passed. Expanded options also match the actual original attempt-000 training settings; only the recorded resume/output/run-identity fields differ. Launcher dispatched in tmux session `zr0-stage3-h50-resume16000-20260912`; script `launch_resume.sh` and `prelaunch_evidence.json` are saved in the new output. Real model restoration and first20 updates remain pending.

## Observed resume result — 2026-09-12

Training launcher started at 2026-09-12T19:47:29.827568+08:00, supervisor PID 685219, launcher PID 686490, four ranks 687099–687102; tmux session `zr0-stage3-h50-resume16000-20260912`. At 2026-09-12T19:57:57.394179+08:00, update 16030/150000 was recorded and training remained active. Online W&B: [stage3_h50_resume16000_20260912-stage3_joint-retry000](https://wandb.ai/jumbo3r-zhejiang-university/zr0-three-stage-formal/runs/stage3_h50_resume16000_20260912-stage3_joint-retry000). Next complete scheduled checkpoint is 18000; no new checkpoint exists in this continuation at the observation time. The latest complete recovery source is the preserved original 16000 checkpoint above.

The supervisor's `stage3_joint/retries/attempt-000/startup_verified.json` reports **passed** for production updates 16001–16020. All 845 component tensors match the 16000 source exactly; all five components remain trainable. Four-rank FP32 master parameters, AdamW states, scheduler, rank RNG and sampler cursor match exactly, plus rank 0's global exposure state. Diagnostic optimizer updates = 0. Logged fast resume: epoch 0, batch index 32000, historical dataset reads = 0. LR restored to 9.98134647456885e-6 and advances at the original global schedule positions. Frozen data audit and independent statistics are reused without recomputation.

First 20 successful-update raw loss means: AR = 0.18289269 (10^-1), Slot = 0.58495879 (10^-1), Optical Flow = 0.04010181 (10^-2), FM = 0.18261042 (10^-1). Total mean = 1.19454246; minimum = 0.73285508. Outer weights remain 1/0.1/1/5; maximum formula difference = 7.4505806e-08. All component gradients are finite; first 20 component-gradient means are VLM 4.040095, Query 0.145771, Slot 0.478215, Flow 9.875682, Expert 5.112921. Detailed minima/maxima and weighted losses are saved in `resume_startup_report.json`.

Updates 2–20 average 3.7830 s/update (median 3.6528 s); this short sample excludes future save overhead. Peak allocated/reserved memory = 39.1356/74.8027 GiB. Through observed update 16030: no skipped updates or W&B failures; no fatal exception/OOM in the startup log. Initial NCCL barrier device-inference and torch_dtype deprecation warnings were logged; four-rank restoration and subsequent collectives/updates completed successfully. Six CPU command regressions passed, including the unchanged original fresh plan and rejection of unintended training parameter changes. Relevant launcher/config/test/documentation/reference changes are staged; no commit was created. Training continues in the background toward cumulative 150000, with original save 2000 and bounded retry policy. Completion, later checkpoints and any later failures require subsequent observation; rollout evaluation and next-window replay equivalence remain untested.
