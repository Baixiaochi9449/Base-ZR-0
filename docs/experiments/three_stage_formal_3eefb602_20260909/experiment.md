# Formal Continuation at 3eefb602

## Stopped and Preserved

The user stopped this weight-1 Stage 2 at 2026-09-09 22:05:23 +08:00. Last logged update 1393; latest complete full checkpoint is update 1000, with all 35 saved inventory entries unchanged. Later 393 logged updates are not in the saved checkpoint. All ranks/workers and the supervisor exited, GPUs were released, and `retry_disabled` prevents restart of this sequence. A subsequent explicit user request authorizes a fresh weight-10 Stage 2 in separate `commit_3eefb602_flow10` outputs; its VLM/Query come from completed Stage 1, not this Stage 2. Previous active-status sections are historical.

- Created: 2026-09-09, Asia/Shanghai. Owner: lq; execution: coding agent.
- Purpose: start Stage 2 from the completed formal Stage 1 using the exact user-requested commit `3eefb602417d3bd4b20bef2b47660b404aefb565`.
- Runtime export: `/opt/data/private/lq/ZR-0/outputs/runtime_snapshots/3eefb602417d3bd4b20bef2b47660b404aefb565`.
- The pinned launcher references an experiment Markdown template that was not tracked in the commit. The existing template was copied into the export as documentation; all executable runtime files remain byte-identical to the requested commit and prior audit binding. The first preparation passed the saved audit and checkpoint checks, then failed on that missing template before creating an output or launching training.
- Current working tree: staged Flow V2 modifications remain preserved and are excluded from this pinned runtime. No checkout, stash, new worktree, staging or commit is performed.
- Config: `configs/three_stage_formal_3eefb602_stage2_20260909.json`.
- Base: `/opt/data/private/lq/models/ZR-0`.
- Stage 1 source: `/opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260909/speed_resume_local_scalars_3000/recovery_checkpoints/stage1_ar/step-010000-stage1_ar/latest-model-optimizer-lr`.
- Output: `/opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260909/commit_3eefb602_stage2`.
- W&B: required online, project `zr0-three-stage-formal`, group `three_stage_formal_20260909`; per-attempt run names and URLs are recorded by the pinned recovery supervisor and `wandb_identity.json`.

## Execution

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python /opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260909/pinned_stage2_control.py prepare
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python /opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260909/pinned_stage2_control.py launch
```

The experiment wrapper reuses the requested commit's existing plan builder and recovery supervisor. It validates and imports the completed 10000-update Stage 1, which is never launched again. Stage 2 starts at zero logical updates with newly initialized Heads and a new optimizer/scheduler. Existing automatic retries preserve complete checkpoints and resume each stage within its original budget. Stage 3 remains scheduled under the user's earlier explicit formal-training authorization.

Stage 2 inherits the Stage 1 VLM and Query exactly. Vision encoder, merger/projector and language model weights are frozen; Query gradients continue through the VLM. Query, Slot Head and dense-regression-v1 Flow Head train. No Action Expert is constructed. Loss is `1 * Slot + 1 * Flow`, with existing internal coefficients and independent activity masks. Inactive independent Heads retain their parameters and Adam state while other active groups update.

Stage 2 uses 5000 successful updates, peak/minimum LR `2e-5 / 2e-6`, 400 warmup updates, cosine schedule, AdamW `(0.9, 0.95)`, epsilon `1e-6`, weight decay `0.01`, clip `1.0`, and no component LR multipliers. Checkpoints save every 1000 updates and at stage end. Stage 3 retains 150000 updates, warmup 12000, peak/minimum LR `1e-5 / 1e-6`, checkpoint interval 2000, and `AR + Slot + Flow + 5 * FM`. Its VLM, Query and Heads inherit Stage 2; only Action Expert and state/action codecs come from the base.

Four NVIDIA A800-SXM4-80GB GPUs; BF16, ZeRO-2, SDPA, seed 42. `global_batch_size = 4 * 16 * 2 = 128`. Stage 2 nominal exposure: 640000 samples; actual exposure and padding are recorded by the existing tracker. Frozen VLM gradient checkpointing is disabled in Stage 2; Stage 3 retains the existing enabled setting. Four loader workers per rank, prefetch 2, OMP/MKL 4 and PyAV 1. Batched metric logging and fast sampler resume remain enabled. Logging occurs every successful update. No rollout validation or loss-based early stopping; 20 consecutive ineffective windows stop training. The retry policy permits three retries with 60/120/240-second delays.

## Data and Saved Audit

The full DROID, Household-v3, Tabletop-v3 and RH20T-v30 identities, splits, episode/sample counts, camera sources, canonical conversions and independent per-dataset statistics remain those in the previously passed frozen audit. Full DROID does not use partial DROID statistics. The 108 excluded RH20T Flow rows remain excluded. Natural sample-equal frame sampling and the Stage 1/2 intersection remain unchanged; Flow absence does not remove a Stage 2 sample. No data payload audit is rerun.

Audit: `/opt/data/private/lq/ZR-0/outputs/three_stage_validation_20260908/flow_exclusion_108/audit_snapshot.json`; SHA256 `19e7b8b5ddc213b57d41660524a89b935ce8870c7432498e90d1dbfcb9753b9f`. Dataset/statistics paths and all counts are fully expanded in each output `experiment.md`, `resolved_dataset_manifest.json`, and the cached routes referenced by the configuration.

DROID/RH20T current first exterior and wrist images are 320x180; Household/Tabletop first-view and wrist images are 640x480. Each RGB image is directly resized to 224x224 with bicubic interpolation, without crop, padding, aspect-ratio preservation or new augmentation; base processor mean/std 0.5. Views remain separate and ordered. Future images and target labels do not enter context C. Text limit 1024. Query roles remain 32 total with trailing 16 Flow queries. State/action have seven canonical dimensions padded to 64 with validity masks; window 1, prediction/action horizon 10. DROID 15 FPS, other datasets 10 FPS. No policy execution horizon or rollout evaluation applies to this offline training.

## Verification and Results

All pinned runtime files match the existing passed audit binding exactly. The existing 118-case CPU regression evidence applies unchanged. Startup checks validate the saved environment/file identities and full four-rank Stage 1 checkpoint. Exact cross-stage component-source verification is enabled before the first optimizer update. No training implementation changes or extra diagnostic optimizer updates are introduced.

At preparation, Stage 1 has completed 10000 updates; Stage 2 and Stage 3 have not started. Actual timestamps, process identities, GPU gate, commands, W&B URLs, loss, gradients, memory, checkpoint and failures are recorded in the output directory. Final offline task success and unobserved real epoch-boundary behavior remain unverified.

### Startup Result

Observed 2026-09-09 21:29:32 +08:00: Stage 2 is running normally at 33/5000 successful updates. Supervisor 212890 launched Accelerate PID 212984 at 21:24:13 +08:00; ranks are 213072/213073/213074/213075. The 626 VLM source tensors and one Query tensor match the completed Stage 1 checkpoint exactly before updates; VLM trainable parameter count is zero and no Action Expert is constructed. Checkpoint serialization diagnostics complete without optimizer updates.

Loss at update 33 is 0.8824559450 = Slot 0.8029061556 + Flow 0.0795497745. Query/Slot/Flow gradient norms are 0.7114181 / 2.7338848 / 33.3011475; these are finite pre-clip diagnostics. No skipped windows, nonfinite losses or W&B failures occurred in the observed startup. Recent ten-window mean is 0.773578 seconds/update, excluding periodic checkpoint overhead. Peak tensor allocation/reservation is 11.4974/13.0098 GiB; observed NVML use is about 14.7-14.9 GiB per GPU.

Run: `stage2_aux/retries/attempt-000` under the output root. W&B: https://wandb.ai/jumbo3r-zhejiang-university/zr0-three-stage-formal/runs/three_stage_formal_20260909-stage2_aux-retry000 . Evidence: `startup_verified.json`, `component_sources_verified.json`, `checkpoint_serialization_verified.json`, `training_metrics.jsonl`. Initial attempt 000 is the first Stage 2 process, not a failed-training retry. No new same-stage resume or rollout has yet occurred. Startup observation is complete; the authorized background supervisor continues the formal schedule.
