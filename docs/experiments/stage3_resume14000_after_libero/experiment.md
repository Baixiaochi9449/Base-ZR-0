# Stage 3 Resume From 14000 After LIBERO

Created 2026-09-11, Asia/Shanghai. Owner: lq; execution: Codex. The user explicitly
authorized completing the current LIBERO run, then resuming the original Stage 3
with `L = AR + 0.1 * Slot + 1.0 * OpticalFlow + 5.0 * FM`.

## Initialization and Execution

The only scheduled stage is `stage3_joint`. Its cumulative target remains 150000
successful optimizer updates: 136000 additional updates after the restored 14000.
Stage 1 and Stage 2 are not scheduled. No further experiment follows completion.

Source checkpoint:
`/opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260910/stage3_resume8000_slot05_flow5/recovery_checkpoints/stage3_joint/step-014000-attempt-000/latest-model-optimizer-lr`.
Base lineage: `/opt/data/private/lq/models/ZR-0`. All five components are restored
from this full Stage 3 checkpoint, including the already trained Action Expert
and its state/action encoders and decoders. LIBERO weights are not used. Model,
optimizer, scheduler, all four rank RNG states, sampler cursor and exposure resume
together. Original checkpoint bytes are preserved. Exact loading checks execute
before the first optimizer update; no extra diagnostic updates or diagnostic
forward passes are introduced.

Runtime commit: `3eefb602417d3bd4b20bef2b47660b404aefb565`, pinned under
`outputs/runtime_snapshots/3eefb602417d3bd4b20bef2b47660b404aefb565`. Current unrelated
Flow V2 worktree/index changes are retained and excluded from runtime imports.
The existing `scripts/train_vla_loss_weight_resume.py` and
`utils/resume_loss_weights.py` adapt only the explicitly authorized Slot/Flow
outer coefficients; other artifact/configuration checks remain strict. The new
queue uses the pinned production recovery supervisor and training loop.

Configuration: `configs/three_stage_formal_stage3_resume14000_after_libero.json`.
Output:
`/opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260910/stage3_resume14000_slot0p1_flow1_after_libero`.
Attempt output: `stage3_joint/retries/attempt-000`, with a new directory per retry.
All parsed defaults and the actual command are saved before training in
`launch_plan.json`, `stage3_joint/expanded_options.json`, and each attempt.

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python scripts/watch_stage3_after_libero.py --config configs/three_stage_formal_stage3_resume14000_after_libero.json --mode prepare
env CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python scripts/watch_stage3_after_libero.py --config configs/three_stage_formal_stage3_resume14000_after_libero.json --mode launch
```

Default `--mode check` is CPU-only and starts no monitoring or training. `prepare`
creates exclusive output and documents; `launch` starts the durable monitor.
Creating `retry_disabled` in the new output stops its queue and owned training.
There is no commit. Only task files and the corresponding reference section are staged.

## Completion Dependency and Recovery

Predecessor:
`outputs/ckpts/ZR0-stage3-step14000-LIBERO-action-only-dq32-h10-gbs64-seed42`.
Every 15 seconds the monitor reads its supervisor status, following all retries.
Handoff requires 34184 updates, an explicit completion with exit code zero, a
sealed final checkpoint with matching inventory, complete four-rank ZeRO-2 Adam
state and the LIBERO scheduler counters, and exit of its supervisor/training
processes. Manual stop, exhausted retries, inconsistent archives or abnormal exit
do not trigger Stage 3. The existing four-UUID resource gate runs before each
Stage 3 attempt. Waiting uses CPU only. Existing external GPU allocations are
never terminated by this queue.

File locks and PID/start identities prevent duplicate dispatch. A restarted
monitor can resume waiting before handoff; after dispatch it refuses a second
launch and retains recorded process/state evidence for inspection. Stage 3 allows
at most three automatic retries after 60/120/240 seconds, restoring its newest
valid checkpoint, with the original 14000 archive as fallback before a new save.
Discarded logged updates after the chosen checkpoint are recorded. No budget,
batch or learning-rate change occurs on retry. W&B failure stops execution.

W&B is required online: project `zr0-three-stage-formal`, group
`stage3_resume14000_slot0p1_flow1_after_libero_20260911`, first run name and ID
`stage3_resume14000_slot0p1_flow1_after_libero_20260911-stage3_joint-retry000`.
Actual URL is pending launch and will be recorded in `wandb_identity.json`,
`startup_verified.json` and queue runtime events.

## Data and Inputs

Reuse the passed frozen preparation under
`outputs/three_stage_validation_20260908/flow_exclusion_108`; no data payload audit,
Flow regeneration or normalization recomputation. Audit snapshot SHA256:
`19e7b8b5ddc213b57d41660524a89b935ce8870c7432498e90d1dbfcb9753b9f`.
The source archive's `resolved_dataset_manifest.json` and frozen index files retain
the complete source/index/statistics identities. Each dataset uses its own
train-only state/action statistics; full DROID statistics were independently
audited and do not reuse partial DROID statistics.

| Dataset path under `/opt/data/private/lq/datasets/` | Frozen Stage 3 rows | Statistics SHA256 |
| --- | ---: | --- |
| `droid_1.0.1_stage05_full_95658_20260831` | 19823040 | `9fe87371384944aee47192b37d583a913e8989139aaac7556734ea21857d7966` |
| `molmoact_dataset_household-v3_stage05` | 794199 | `3955199f69c0cd15b2ce2de22c3cff2c6ee0fc4e7f24933c8a312ed3591d8caf` |
| `molmoact_dataset_tabletop-v3_stage05` | 310743 | `f41fa9d88d38bc762d5efb98dc3ef65c933eefdb04f1163bfb1e8049e7e1d125` |
| `RH20T-v30_stage05` | 3131833 | `294db9e546545a754d4cc936de16fe7699197aa44faeb73a7a205ed6ab56ea27` |

Total: 24059815 quality/dual-camera train rows. No LIBERO or evaluation subset is
mixed into pretraining. Dataset sample ratios 1:1:1:1 apply to frame populations,
retaining natural frame-proportional mixing, sampler order and distributed
padding. Actual exposures and padding are recorded by the existing seen tracker;
nominal additional exposure is 136000 * 128 = 17408000 samples, excluding retries.
The previously approved 108 RH20T Flow rows in 12 episodes remain excluded as Flow
targets; other valid objectives remain available. RH20T wrist skew <=100 ms.
Original full-DROID-to-Flow mapping and all source identities are retained.

AR, Slot, Flow and FM use independent availability masks. Current state, task text
and two current RGB views enter the context; future images and supervision do not.
DROID/RH20T first exterior left + wrist left originate at 320x180;
Household/Tabletop first_view + wrist_image at 640x480. Both ordered RGB views
resize directly to 224x224 using bicubic interpolation, without aspect preservation,
crop, pad, letterbox or augmentation. Base processor mean/std are 0.5 per channel.
Text limit 1024, window 1. No evaluation is scheduled; evaluation preprocessing and
rollout metrics are not claimed.

Canonical state: absolute end-effector xyz, roll/pitch/yaw, gripper openness.
Action: next-step relative translation/rotation and binary gripper openness
(1=open, 0=closed). Translation in metres, angles in radians with wrapped deltas.
State/action each have 7 meaningful dimensions, padded to 64 with validity masks.
Existing canonical normalization, dataset-specific statistics and clip [-15,15]
remain unchanged; Flow does not use action statistics. Action chunk and prediction
horizon are 10; DROID 15 FPS, other datasets 10 FPS. Execution horizon and executed
steps are not applicable to offline training.

## Model, Loss and Optimization

All parameters of the VLM visual encoder, merger/projector, DeepStack, language
model and tied embeddings train; all 32 Queries, Slot Head, Optical Flow Head and
the complete Action Expert train. No component is frozen, no LoRA and no detach.
First 16 Queries retain existing Slot groups; last 16 are Flow queries; all 32
condition the Expert. mRoPE, RMSNorm, teacher forcing and `[C,Q,T,P]` are unchanged.

| Objective | Meaning and supervision | Outer coefficient |
| --- | --- | ---: |
| AR | Teacher-forced reasoning-token cross entropy; VLM and Query | 1.0 |
| Slot | Existing structured state/spatial/contact targets; Slot Head and shared context | 0.1 |
| Optical Flow | Dense regression v1, delta 20 frames; Flow Head and shared context | 1.0 |
| FM | Action velocity flow-matching regression; Expert and shared context | 5.0 |

All four losses participate in backpropagation when their supervision masks are
valid. Flags remain `stage3_joint`, `vlm_and_action`, `structured_slots_v1` and
`dense_regression_v1`. Slot internal Q1-Q9 coefficients remain
0.2/0.2/0.04/0.04/0.04/0.2/0.12/0.12/0.04. Flow target/grid/head remain
56/14/256, two layers and eight attention heads. Only Slot 0.5 -> 0.1 and
Flow 5 -> 1 change from the source objective. Component inactivity is determined
by supervision counts/gradient paths, never numerical zero loss. Inactive Heads
retain parameters, Adam moments and group step; active shared components update.

Five existing component optimizer groups all retain AdamW beta=(0.9,0.95), epsilon
1e-6, weight decay 0.01, peak LR 1e-5, minimum LR 1e-6, no LR multipliers, clip 1.0.
Cosine scheduler and optimizer state resume at 14000, with original total 150000
and warmup 12000; no warmup reset. Global scheduler advances once per successful
update; inactive groups use the existing global-schedule policy. Warmup 8% and
epsilon 1e-6 follow the reviewed public scripts; paper values are 5% and 1e-8.
Project stage LRs do not claim exact reproduction of the paper's pretraining.

Four NVIDIA A800-SXM4-80GB GPUs, micro-batch 16, GAS 2:
`global_batch_size = 4 * 16 * 2 = 128`. BF16, ZeRO-2, SDPA, VLM gradient
checkpointing, seed 42, epoch upper bound 16 and explicit total-update limit.
Workers 4/rank, prefetch 2, PyAV threads 1, OMP/MKL threads 4. Fast sampler resume
and batched metric reductions remain enabled; component diagnostics `inactive`
retain Head protection without copying active optimizer states every window.

Log every update; full checkpoint every 2000 updates, at epoch intervals and at
completion, with immutable copies under `recovery_checkpoints/stage3_joint`.
First scheduled resumed save is 16000. No validation interval or loss-based early
stopping; 20 consecutive unsuccessful windows stop the trainer. AMP/optimizer
skips do not advance the successful update count or scheduler.

## Verification and Results

CPU checks cover weight override strictness, unchanged non-weight options,
successful/failed/retried/manual-stop completion states, incomplete JSONL/save,
duplicate launch, and first-window source/state/loss/gradient checks. No training
GPU is allocated during preparation. The 39-file original inventory and saved
four-rank checkpoint contract are checked without rescanning dataset payloads.

After launch, before update 14001 the existing runtime verifies 845 inherited
tensors exactly and optimizer/master parameters/scheduler/RNG/cursor with zero
tolerance; rank 0 verifies exposure. The queue records the first 20 successful
updates' raw/weighted losses, group gradients/activity, scheduler, memory and W&B.
Loss formula comparison uses rtol=1e-5, atol=1e-6 only for rounded metric sums;
loaded-state comparisons remain exact. No changed-objective trajectory equivalence
or rollout success claim is made. Epoch boundaries are not covered by startup.

At creation, LIBERO is running and the successor has not launched. Actual start,
end, completed steps, immutable checkpoint, minimum loss, resource peaks,
interruptions/retries, deviations and W&B URL are appended by the queue/native
supervisor. There is no validation best metric or downstream evaluation result.

Preparation passed on 2026-09-11: all 39 source files match the sealed inventory;
full four-rank model/optimizer/scheduler/RNG/cursor/exposure checks pass. The pinned
command parser confirms only authorized weight/source/output/W&B changes. The
cached audit SHA256 matches the value above. CPU regression: 41 tests pass;
report: `cpu_tests.xml` in the output root. The startup checker also passes the
prior real 8001-8020 log/state records, including an inactive Slot window whose
loss fields are absent (supervision count zero); formula error <=1.10e-7. Its final
tested identity was sealed before queue launch. No source data audit or diagnostic
optimizer update was performed. New-objective GPU verification remains pending.

Queue activation verified at 2026-09-11 00:20:54 +08:00: detached supervisor PID
2060916 (start_ticks 3899291609) is waiting for LIBERO completion, with no successor
training child or handoff receipt. LIBERO passed step 9240 with required online
W&B and no reported skipped update. The queue uses no GPU allocation. Receipts:
`detached_launch.json`, `queue_activation_verified.json`, `queue_events.jsonl`.
The new Stage 3 W&B run and production startup result remain pending.

## User-Requested Stop at 20000

This section supersedes the historical running/queued status and instruction to
continue to 150000. On 2026-09-11 the user requested stopping after step 20000 is
saved. Scheduler length, objective, data and original configuration are unchanged.

LIBERO completed successfully at 06:44:21 +08:00. Stage 3 launched at 06:44:46;
exact inheritance of 845 tensors and all four rank optimizer/scheduler/RNG/cursor
states passed before resumed updates. First 20 updates passed the automatic
startup check at 06:52:17, formula error <=7.01e-8. W&B run:
`https://wandb.ai/jumbo3r-zhejiang-university/zr0-three-stage-formal/runs/stage3_resume14000_slot0p1_flow1_after_libero_20260911-stage3_joint-retry000`.

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -u scripts/stop_three_stage_after_checkpoint.py --output outputs/three_stage_formal_20260910/stage3_resume14000_slot0p1_flow1_after_libero --target 20000 --execute
```

The observer checks saved process/start identities and four rank stdout owners.
After the committed step-20000 post-save metric it pauses ranks while the existing
supervisor archives the stationary checkpoint. Only after complete inventory and
four-rank checkpoint validation does it disable retries and terminate this run's
process groups. No foreign process is signaled. Syntax and live ownership checks
passed. At 13:06:22 +08:00 ranks paused with final logged step 20000. Completion
receipt and actual cleanup results follow; unlogged in-flight work cannot be
proven absent. No diagnostic updates or source-data audit occur.

Stop completed at **2026-09-11 13:08:30 +08:00**. Last logged global step 20000,
6000 successful updates after resume, zero logged skips/nonfinite total losses or
W&B failure windows. Final raw AR/Slot/Flow/FM: 0.167060/0.193355/0.071055/0.211407;
final weighted total 1.314484; minimum total 0.262466. Peak allocated/reserved GPU
memory 43.9695/77.6563 GiB. No validation or rollout metric was measured.

Final immutable full checkpoint (39 files, four-rank contract validated):
`/opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260910/stage3_resume14000_slot0p1_flow1_after_libero/recovery_checkpoints/stage3_joint/step-020000-attempt-000/latest-model-optimizer-lr`.

Supervisor 2060916, launcher 2226743, four ranks and owned DataLoader workers have
exited. `retry_disabled` prevents automatic restart; all four GPUs report 2 MiB
and 0% utilization. Evidence: `stop_20000_completed.json` and
`stop_20000_events.jsonl` in the output root. No further training is running for
this experiment. The original 150000-step scheduler is preserved in the saved
state; stopping at 20000 is the user's explicit deviation from that execution
budget. Checkpoints 14000, 16000, 18000 and the completed LIBERO result remain intact.
