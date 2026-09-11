# Three-stage Formal Training

## Local Scalar Transfer Continuation, 2026-09-09

The current batched-collective process continues from saved update 2000. A
further logging-only adaptation groups local metric tensor transfers and reuses
their CPU values for TensorBoard and JSON. It uses the existing
`--batch_metric_reductions` flag; all values, types, record frequency, W&B inputs,
training computations, data/statistics and update budgets are preserved.
CPU regressions and the complete update-3000 stop are required before activation.
All 115 regression cases pass (`local_scalars_cpu_r1.xml`); ten focused scalar
cases also pass after final test cleanup. Runtime hashes are sealed in
`local_scalars_tested_runtime.json`. Monitor 124600 waits for complete update
3000 and checks rank-group cleanup. Actual throughput remains pending; the
new runtime will be measured from 3001 to 4001 including the update-4000 save.
The final adaptation also combines TensorBoard scalar events using its standard
Summary format and existing scalar encoder. Tags, steps, values and text remain
unchanged; scalar tags within an event share its wall time. The separate
learning-rate timing event retains its original call. Final regression evidence
supersedes the earlier scalar-only report: 118 cases pass in
`local_logging_cpu_r1.xml`; use `local_logging_tested_runtime.json` for activation.
Actual CPU logging helper cost was 32.730 versus 3.193 ms/window for 103 current
metrics over 100 windows, with all EventAccumulator values exact. This is a CPU
logging benchmark, not a full-model throughput result.
Configuration: `configs/three_stage_formal_local_scalars_resume_20260909.json`;
retry policy: `configs/three_stage_formal_local_scalars_retry_20260909.json`;
output: `outputs/three_stage_formal_20260909/speed_resume_local_scalars_3000`;
source: `speed_resume_batched_2000/stage1_ar/latest-model-optimizer-lr` at 3000.
The original online W&B identity and 10000/5000/150000 targets remain unchanged.
The immutable source audit is reused. Earlier continuation sections describe
preserved predecessor processes; actual results are recorded after verification.

## Batched Metric Continuation, 2026-09-09

The first repaired four-GPU process resumes update 1000 correctly and runs near
1000 updates/hour before checkpoint overhead. To provide margin, the next
continuation explicitly enables `--batch_metric_reductions`. Only detached
token-statistic and W&B scalar communication is batched; all metrics and their
frequency, loss/gradient/optimizer/scheduler calculations, data order and batch
128 remain unchanged. CPU evidence: 105 passing cases, including two-rank Gloo
and exact production-window parameter/metric comparisons. A separate CPU
process-group shutdown test passes. Tested runtime hashes and report identities
are in `performance_repair_20260909/batched_metrics_tested_runtime.json`.

The first repaired four-GPU process completed updates 1001 through 2000 and
stopped at 11:43:05 +08:00 after the complete update-2000 checkpoint passed
verification. Monitor PID 105039 terminated the identified rank groups and
confirmed that no group members remained. All four GPUs were released. The
1001-to-2000 TensorBoard interval contains 999 update intervals, takes
3654.482 seconds and includes the save: 984.107 updates/hour. Its average
optimizer-window time is 3.552 seconds, with zero skipped windows or W&B
failures. Evidence: `performance_repair_20260909/initial_resumed_throughput.json`
and `metric_batching_stop_completed.json`. The original update-1000
checkpoint and the whole first resumed output are preserved. The planned new
configuration is `configs/three_stage_formal_batched_resume_20260909.json`, retry
policy `configs/three_stage_formal_batched_retry_20260909.json`, output
`outputs/three_stage_formal_20260909/speed_resume_batched_2000`, and source
`speed_resume_four_gpu_1000/stage1_ar/latest-model-optimizer-lr` at update 2000.
Final stage budgets remain 10000/5000/150000; online W&B retains the original
Stage 1 identity. This is a same-stage continuation, not a fresh Stage 1.

Activation requires the complete stop receipt, matching tested runtime hashes,
saved audit identity, archived prior binding, and the ordinary four-UUID gate.
The immutable data audit and independent dataset statistics are reused. No
source payload scan or additional model diagnostic update is performed. Actual
batched-runtime throughput remains to be verified after launch. Earlier startup
and output-directory records below are historical.

## Four-GPU Continuation After Orphan Cleanup, 2026-09-09

The user explicitly authorized termination of their residual Stage 1 and
continuation on four GPUs. Device-handle inspection identified three orphan
DataLoader workers (54224/54229/54276), owned by lq, in the old rank-0 process
group/session 51717, with stdout pointing to `speed_resume_1000/stage1_ar/train.log`.
Their identities were rechecked before SIGTERM. All three exited, the NVML
allocation for PID 1534740 disappeared, and all four GPUs reported 2 MiB used.
The earlier external-process attribution was therefore incorrect. Evidence:
`outputs/three_stage_formal_20260909/performance_repair_20260909/orphan_workers_released.json`.

Current formal configuration: `configs/three_stage_formal_four_gpu_resume_20260909.json`.
Retry policy: `configs/three_stage_formal_four_gpu_retry_20260909.json`.
Current output: `outputs/three_stage_formal_20260909/speed_resume_four_gpu_1000`.
The source is the original complete Stage 1 update-1000 checkpoint. The tested
runtime, frozen audit, independent dataset statistics, W&B identity and total
10000/5000/150000 budgets remain unchanged. Prior failed output directories are
retained. This section supersedes previous blocked/current-output statements.
Actual startup, restored-state checks and throughput will be recorded here.

## Fast Data-Cursor Recovery, 2026-09-09

The user additionally requested optimization of the slow resume itself. The first performance-repair startup restored the step-1000 VLM/Query and original W&B run, but the old loop decoded and discarded historical batches. It was deliberately stopped at skipped micro-batch 1486 before any new optimizer update. The complete original step-1000 checkpoint remains the sole resume source.

`--fast_resume_data_skip` now uses Accelerate 1.6.0 `skip_first_batches` at the prepared batch-sampler boundary, preserving epoch, distributed tail/remainder and absolute batch indices. Only frozen Stage05 datasets and the existing deterministic Slot wrapper are accepted. The default is false for compatibility; new three-stage commands enable it, including later-stage retry commands. The original full DataLoader length, RNG restoration, sampler cursor, exposure, optimizer and scheduler contracts remain unchanged. Historical images and labels are not read during the skip.

The current configuration is `configs/three_stage_formal_fast_resume_20260909.json`, with retry policy `configs/three_stage_formal_fast_retry_20260909.json` and output `outputs/three_stage_formal_20260909/speed_resume_direct_1000`. Stage 1 resumes the original step-1000 checkpoint and W&B run; Stage 2/3 inherit this sequence. Budgets remain 10000/5000/150000, all other training contracts below remain in force. Prior startup records are historical and do not represent new optimizer updates.

Verification: 89 CPU tests pass, including real two-rank Gloo and all four rank layouts. On the actual 24,059,815-row frozen Stage 3 sampler, index-only recovery to micro-batch 2000/200000 took 0.2508/12.3928 seconds and exactly matched the old sampler's subsequent indices; no historical image/label reads occurred. These timings exclude model loading and real image decoding. The previously saved three-stage full-model next-window hashes are reused for CPU input verification before the next GPU launch. Actual startup and sustained throughput results will be appended after measurement.

The real input verification subsequently passed all 12 stage/rank windows exactly (`fast_resume_real_windows.json`), with zero optimizer updates. The fast-resume supervisor launched at 09:57:58 +08:00 (PID 71465), followed by retry watcher 71766. The original GPU gate and all three configured retries were blocked by GPU 0 / NVML PID 1534740 using 15364 MiB, a process not visible in this environment. At 10:11:03 +08:00 the watcher recorded `retry_supervisor_stopped` and exited. Both supervisor and watcher are absent from the process table. No training child or new optimizer update started in this output sequence. Resource thresholds remain unchanged and the external process was not signaled. Stage 1 remains at the original complete step-1000 checkpoint; Stage 2/3 have not started. Actual full-model resume verification and sustained throughput remain pending.

## Performance Repair and Planned Step-1000 Resume, 2026-09-09

The user requested the existing Stage 1 run stop after its complete step-1,000 checkpoint, then resume with throughput repaired while retaining normal training semantics. The original retry watcher was deliberately disabled for this stop. The stop/repair evidence is under `outputs/three_stage_formal_20260909/performance_repair_20260909`. An initially created temporary worktree was not used for implementation or tests and was removed at the user's direction; all changes are in the current worktree.

The resumed sequence uses `configs/three_stage_formal_speed_resume_20260909.json`, writes to `outputs/three_stage_formal_20260909/speed_resume_1000`, and preserves the original checkpoint. Stage 1 resumes the same online W&B run with `resume=must`; its scheduler still has length 10,000 and warmup 800. Later stages retain the original 5,000/150,000 budgets and inherit the new formal outputs. Their initialization and all data/statistics, image, batch, loss and optimizer contracts below are unchanged.

Formal commands now explicitly set `--component_update_diagnostics inactive`. Native inactive-Head protection and exact inactive-state checks remain; active-component parameter/Adam snapshots and the duplicate native FP64 gradient norm are omitted. Existing loss/module-gradient/throughput/memory logs, supervision/update flags, group step/LR and global scheduler checks remain. `full` mode retains complete bounded-validation diagnostics; the CLI default is unset for backward compatibility. The old statement below that every active group's full delta is logged applies only to the original process before the repair.

The resumed Stage 1 also enables `--verify_resume_state`: once before update 1001, after restoring rank RNG and before adding new exposure, compare the actual loaded native optimizer, FP32 master partitions, scheduler, rank RNG and sampler/cursor to the saved files; main rank also checks exposure arrays/bitsets. Existing source verification checks model/Query tensors. No diagnostic forward or extra optimizer update is performed. A new runtime binding preserves the immutable audit snapshot and archives previous binding/certificate files; source payloads are not re-audited.

CPU evidence: 161 distinct tests passed across `cpu_tests_r3.xml` (151 passing cases) and `cpu_gloo_r4.xml` (11 passing cases, including the ten previously blocked by optional Triton imports in CPU subprocesses). Production speed and exact loaded-state checks remain pending until the planned stop/resume. The existing retry runner will use `configs/three_stage_formal_speed_retry_20260909.json` and retains the original checkpoint as a fallback before the first new save.

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH=.:lerobot /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python outputs/three_stage_formal_20260909/performance_repair_20260909/continue_after_stop.py --mode run-after-stop
```

This explicit continuation waits for the identified stop monitor, requires a verified step-1,000 checkpoint and the tested code identities, archives/rebinds only runtime metadata, materializes the full commands/options/documents, then invokes the existing formal runner and retry watcher. It cannot fall back to base initialization. Subsequent runtime records supersede this planned status.

## Automatic Retry Authorization, 2026-09-09

Activated at 2026-09-09 01:49:37 +08:00: retry watcher PID 4137985 attached to supervisor 4124301; original training PIDs are unchanged. Stage 1 reached 64 confirmed updates, loss 3.203702926635742, W&B online with no backlog. Tests passed: 13 retry cases in 6.51 seconds plus eight unchanged formal cases in the prior combined run. Actual three-stage validation checkpoint layouts and resume commands passed CPU-only inspection, with zero diagnostic optimizer updates; validation weights are not used for formal initialization/retry. No failure was injected into formal training. Activation evidence: `/opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260909/retry_activation_verified.json`.


The user explicitly authorized automatic retries after failures. This supersedes earlier no-retry statements in this experiment. `configs/three_stage_formal_retry_20260909.json` enables the independent `scripts/watch_three_stage_formal.py --mode watch` supervisor; without that explicit mode the script only checks readiness, and the original runner retains its prior default. The watcher attaches to the existing supervisor's PID and process start identity without interrupting healthy training. An exclusive lock prevents duplicate watchers. Operator interruption is not retried.

Each stage permits at most three retries with 60/120/240-second delays. Each retry passes the same four-UUID GPU gate and reuses unchanged runtime/data/statistics identities, global batch 128, schedules and required online W&B. Runtime contract changes are not automatically repaired or re-audited. Completed stages are retained. A retry restores the newest complete same-stage checkpoint's model, native optimizer, scheduler, rank RNG, sampler cursor and exposure through the existing production resume path; without one it uses that stage's original initialization. Fresh Stage 3 always loads its Expert only from base; same-stage Stage 3 recovery uses its own saved Expert. Final logical targets remain 10,000/5,000/150,000; work after a rollback checkpoint is recomputed and discarded logged updates are recorded separately, with any unlogged last update explicitly unknown.

The watcher preserves complete checkpoints under `recovery_checkpoints/<stage>/step-<update>-<attempt>/latest-model-optimizer-lr`. It copies after a committed metric, validates matching model/client/scheduler/four-rank state and exposure, checks source stability before publication, and keeps older complete copies. Incomplete live saves are rejected. ZeRO-2 master partitions omit `group_paddings` while Adam moments retain them; the validator checks that exact size relationship and permits inactive-head native steps below the global step. CPU checkpoint deserialization disables optional Triton import only in the watcher, without initializing CUDA or changing the fresh training child interpreter.

Retries write independent `<stage>/retries/attempt-NNN` documents, commands, logs and online W&B runs (`-retryNNN` suffix) in the original experiment group. This avoids overwriting failed evidence or W&B dropping replayed lower-numbered steps. Source checkpoint and rollback accounting are linked in `formal_events.jsonl`. The supervisor stops after the retry limit or Stage 3 completion; no budget, batch, loss or normalization changes are made automatically. Its own start identity/policy is saved in `retry_watch_started.json`, log in `retry_watch.log`, and a recovered full-sequence result in `formal_recovered_summary.json`. Creating `retry_disabled` under the experiment root disables the watcher; operator termination is not automatically reversed.

```bash
setsid --fork env CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH=.:lerobot OMP_NUM_THREADS=2 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -u scripts/watch_three_stage_formal.py --config configs/three_stage_formal_retry_20260909.json --mode watch >> outputs/three_stage_formal_20260909/retry_watch.log 2>&1 < /dev/null
```

Tests: command routing and fixed budgets, inactive-head/padded optimizer partitions, incomplete-save rejection and fallback to preserved copies, copy-time mutation, partial metric writes, healthy-process attachment, PID reuse, manual interruption, three-retry exhaustion and resumed-segment completion. Existing full-model recovery evidence is reused; no fault is injected into the active formal run and no extra optimizer diagnostic update is performed. No source audit, normalization regeneration, staging or commit is performed.


## Confirmed Formal Startup

At 2026-09-09 01:20:50 +08:00, supervisor PID 4124301 and Stage 1 Accelerate PID 4125082 were running independently of the terminal. Rank PIDs: 4125235/4125236/4125237/4125238. The UUID gate passed, all 626 VLM state tensors matched the sole base exactly, Query was freshly initialized, and actual H50-source/H10-runtime model serialization passed before update 1. Three successful updates were confirmed with global batch 128; latest total loss 9.114498138427734, no skipped updates or W&B backlog. Stage 2/3 remain queued; the first formal checkpoint is scheduled at Stage 1 update 1000. This records startup, not completion.

W&B: https://wandb.ai/jumbo3r-zhejiang-university/zr0-three-stage-formal/runs/three_stage_formal_20260909-stage1_ar . Full startup evidence: `/opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260909/launch_verification.json`. Training continues under the sequential supervisor with automatic failure stop. No staging or commit occurred.

Detached invocation:

```bash
setsid --fork env PYTHONNOUSERSITE=1 PYTHONPATH=.:lerobot /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -u scripts/run_three_stage_formal.py --config configs/three_stage_formal_20260909.json --mode run >> outputs/three_stage_formal_20260909/launcher.log 2>&1 < /dev/null
```


Created 2026-09-09 (Asia/Shanghai). Owner: lq; execution: Codex. User explicitly authorized Stage 1 10,000 / Stage 2 5,000 / Stage 3 150,000 successful optimizer updates after completed 100/100/100 validation. This is a fresh formal experiment, with each later stage initialized from its formal predecessor. Validation checkpoints remain separate.

Base: `/opt/data/private/lq/models/ZR-0`. Code baseline: `3bad66373877bcdfa9c1f86e134816fc8a426207`; current uncommitted three-stage integration, source verification, cached preparation and formal launcher changes are recorded in `code_identity.json`. Existing unrelated staged and unstaged user files are retained; this task does not stage or commit.

Configuration: `configs/three_stage_formal_20260909.json`, referencing immutable preparation options `configs/three_stage_validation_20260908.json`. Output: `/opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260909`. Full stage commands and all parser defaults are materialized in `launch_plan.json` and each stage's `expanded_options.json` before any GPU process starts.

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH=.:lerobot /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -u scripts/run_three_stage_formal.py --config configs/three_stage_formal_20260909.json --mode prepare
env PYTHONNOUSERSITE=1 PYTHONPATH=.:lerobot /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -u scripts/run_three_stage_formal.py --config configs/three_stage_formal_20260909.json --mode run
```

The detached supervisor records its PID in `formal_started.json`, all child PIDs and exit outcomes in `formal_events.jsonl`, and updates this document and stage documents. SIGTERM interrupts owned training groups. Failure stops the sequence without retry or budget expansion. Stage 3 completion ends the sequence. No evaluation or downstream training is scheduled.

## Saved Data and Weight Identities

Preparation root: `/opt/data/private/lq/ZR-0/outputs/three_stage_validation_20260908`; active revision: `flow_exclusion_108`. The passed `audit_snapshot.json` SHA256 is `19e7b8b5ddc213b57d41660524a89b935ce8870c7432498e90d1dbfcb9753b9f`. Reuse the saved 78,877 file identities, indices, source hashes and statistics. Startup checks file metadata and implementation binding; it does not re-audit payloads, tokenize the corpus, regenerate Flow, or recompute statistics. The pre-update model tensor equality check verifies this process's actual loading, using the original source files.

All 17 base file identities, original processor/configuration and weight shard hashes are in preparation `base_identity.json`. Base Expert weights SHA256: `38877236134df1e54c9493e3178decc8c1b69ba2d4b37359b7c826b719940d97`. Base VLM hidden size 2048, state/action padded dimension 64; original Expert horizon 50 has only the validated runtime length override to 10. Original H50 config bytes and runtime H10 provenance are preserved in each checkpoint. No shape mismatch suppression or random Expert layer insertion.

| Dataset | Release path under `/opt/data/private/lq/datasets/` | Source episodes / frames | Stage 1/2 intersection | Stage 3 quality and dual-camera |
| --- | --- | --- | --- | --- |
| full DROID | `droid_1.0.1_stage05_full_95658_20260831` | 95,658 / 27,630,375 | 1,815,749 | 19,823,040 |
| Household-v3 | `molmoact_dataset_household-v3_stage05` | 5,936 / 794,199 | 260,412 | 794,199 |
| Tabletop-v3 | `molmoact_dataset_tabletop-v3_stage05` | 1,881 / 310,743 | 73,081 | 310,743 |
| RH20T-v30 | `RH20T-v30_stage05` | 8,142 / 3,898,056 | 109,214 | 3,131,833 |
| Total | No LIBERO | 111,617 / 32,633,373 | 2,258,456 | 24,059,815 |

Train-only statistics and admitted train indices; no validation/test rows in statistics. Existing canonical state/action conversion and clipping to [-15,15] are shared algorithms; each dataset has independent values. Padding and invalid dimensions are masked; constant dimensions and inverse transforms were checked. Files: preparation `{droid,household,tabletop,rh20t}/joint/stats.json`, with identities in each `frozen_index.json`. Statistics content hashes:

- DROID: `9fe87371384944aee47192b37d583a913e8989139aaac7556734ea21857d7966` (full DROID independently prepared; no partial statistics reuse).
- Household: `3955199f69c0cd15b2ce2de22c3cff2c6ee0fc4e7f24933c8a312ed3591d8caf`.
- Tabletop: `f41fa9d88d38bc762d5efb98dc3ef65c933eefdb04f1163bfb1e8049e7e1d125`.
- RH20T: `294db9e546545a754d4cc936de16fe7699197aa44faeb73a7a205ed6ab56ea27`.

Natural valid-frame sampling reuses the sample-equal sampler, ratios 1:1:1:1 applied to dataset frame populations, with deterministic integer global-batch allocation and distributed padding recorded by the existing seen tracker. Stage 1/2 use the identical frozen intersection; Stage 3 uses all quality/dual-camera samples and independent AR/Slot/Flow/FM masks. Flow absence does not filter Stage 2. The approved 108 Flow rows across 12 RH20T episodes are excluded as Flow targets only; source HDF5 and valid other objectives remain. RH20T wrist skew remains <=100 ms. Full DROID uses the saved full-episode to original 9,465-episode Flow mapping; original IDs and hashes remain intact. Flow units/normalization are independent of action statistics. Lazy data reads and bounded worker handles are retained.

## Stages and Optimization

| Stage | Initialization and trained components | Frozen / absent | Successful updates | Peak / minimum LR | Warmup | Save interval |
| --- | --- | --- | --- | --- | --- | --- |
| stage1_ar | Base VLM; fresh Query | Slot, Flow and Expert absent | 10,000 | 1e-5 / 1e-6 | 800 | 1,000 |
| stage2_aux | Formal Stage 1 VLM and Query; fresh Slot/Flow heads; train Query and both heads | Entire VLM frozen; Expert absent | 5,000 | 2e-5 / 2e-6 | 400 | 1,000 |
| stage3_joint | Formal Stage 2 VLM, Query, Slot, Flow; Expert and state/action codecs only from base; train all | None | 150,000 | 1e-5 / 1e-6 | 12,000 | 2,000 |

VLM means visual encoder, visual merger/projector, language model and tied embeddings. These are all trainable in Stages 1/3, all frozen in Stage 2 while preserving Query gradients through the VLM. Each stage rebuilds optimizer/scheduler and starts its own counter at zero. No LoRA. Every loaded component is compared tensor-by-tensor at final BF16 dtype before update 1; actual model serialization and provenance validation also run with zero updates. Stage 3 must preserve all four inherited components exactly during initialization.

All trainable component groups use AdamW betas (0.9,0.95), epsilon 1e-6, weight decay 0.01, cosine with minimum ratio 0.1, clip 1.0; no group LR multipliers. Inactive independent heads retain parameters, Adam moments, native group step and LR. Shared Query/VLM and FM-supervised Expert update according to their actual active paths. Only successful global updates advance the scheduler. Consecutive 20 unsuccessful windows stop training. Numeric zero loss is not used to infer supervision.

Warmup 8% and epsilon 1e-6 follow the reviewed public scripts; the paper uses 5% and 1e-8. Stage LRs are the project plan, not a claim of exact original pretraining replication.

## Loss and Inputs

- Stage 1: `L = 1.0 * L_AR`. Teacher-forced action-reasoning token cross entropy supervises VLM and Query. Runtime reads frozen indices, current RGB/context and AR targets only; no Slot/Flow labels or heads.
- Stage 2: `L = 1.0 * L_slot + 1.0 * L_optical_flow`. Existing structured Slot internal losses and dense Flow regression retain audited internal coefficients; supervised heads and Query receive gradients through the frozen VLM.
- Stage 3: `L = 1.0 * L_AR + 1.0 * L_slot + 1.0 * L_optical_flow + 5.0 * L_FM`. FM velocity regression supervises Expert/codecs and conditioning VLM/Query. Missing targets use independent masks; inactive objectives do not enter supervision counts.

Explicit stage and `slot_aux_type` / `optical_flow_aux_type` / outer weights control objectives. Query count 32, last 16 for Flow, first 16 retain Slot role groups. `[C,Q,T,P]`, mRoPE, DeepStack, RMSNorm, teacher forcing and all 32 Expert conditioning queries remain unchanged; `.generate()` is rejected. Future images and supervision labels do not enter context C. Max text length 1024; saved base-processor audit maxima DROID/Household/Tabletop/RH20T: 941/724/693/789.

DROID/RH20T fixed first exterior left + wrist left views at source 320x180; Household/Tabletop first_view + wrist_image at 640x480. Each current RGB view is resized directly to 224x224 with bicubic interpolation, without aspect-ratio preservation, crop, letterbox or new geometric augmentation; base processor channel mean/std 0.5. Two images remain separate ordered views. No future frames in current inputs. No evaluation is run; any later evaluation must document matching processing separately.

Canonical state is absolute end-effector `[x,y,z,roll,pitch,yaw,gripper_open]`; action is native-next-step relative `[dx,dy,dz,droll,dpitch,dyaw,gripper_open]`. Both have 7 meaningful dimensions and are padded to 64 with dimension/validity masks. Translation uses metres in each source's base/world frame; lowercase xyz RPY angles use radians and wrapped component deltas. State gripper openness is continuous, action openness is binary, 1=open and 0=closed. Window 1; action chunk/prediction horizon 10; DROID 15 FPS, other datasets 10 FPS. Execution horizon and executed control steps are not applicable to offline training; no policy rollout is performed.

## Resources, Logging and Checkpoints

Four NVIDIA A800-SXM4-80GB cards, micro-batch 16 per rank, GAS 2: `global_batch_size = 4 * 16 * 2 = 128`. BF16, Accelerate/DeepSpeed ZeRO-2, SDPA, seed 42, epochs upper bound 16 with explicit successful-update limits. Existing VLM gradient checkpointing is enabled in Stages 1/3 and disabled for the frozen Stage 2 VLM. Four workers/rank, prefetch factor 2, PyAV decoder threads 1, OMP/MKL threads 4. GPU UUIDs are resolved and checked before each stage using the existing three-consecutive-idle gate (71,680 MiB free minimum, 1,024 MiB used maximum, utilization 0).

Every successful update logs loss, gradients, per-group activity/state changes, LR, throughput and memory. Full parameter/optimizer delta diagnostics are retained from validation and incur CPU copy/comparison overhead. Planned successful-window exposures: 1,280,000 / 640,000 / 19,200,000 samples; actual exposure, retries and distributed padding are recorded by the seen tracker. No fixed 50-step process split and no 100-step validation cap. Checkpoints save model snapshots at each configured interval and stage end, plus full current ZeRO-2 optimizer/scheduler, per-rank RNG, cursor and sampler/exposure state in `<stage>/latest-model-optimizer-lr`. Historical validation checkpoints are never overwritten.

W&B is mandatory online, project `zr0-three-stage-formal`, group `three_stage_formal_20260909`, run IDs/names `three_stage_formal_20260909-{stage1_ar,stage2_aux,stage3_joint}`. Actual URLs are written to each `wandb_identity.json` at initialization. W&B failure stops training; no offline fallback. Training logs: `<stage>/train.log`; metrics: `<stage>/training_metrics.jsonl`; supervisor: `launcher.log`. No validation interval, rollout metric, best-checkpoint criterion or loss-based early stopping is configured.

## Verification and Runtime Results

Prior evidence: full-model 100/100/100 validation and exact four-rank save/resume checks passed; natural inactive Slot windows passed on production ZeRO-2. Report: `/opt/data/private/lq/ZR-0/outputs/three_stage_validation_20260908/validation_results.md`. Formal entrypoint tests check full schedules, source routing, separate outputs, default-off source verification, original H50 config loading with runtime H10 and complete rank state at stage transitions. No source audit is repeated.

Formal launch regression: 29 relevant cases passed (27 in the combined run, one corrected horizon fixture in 10.14 seconds, one added materialization/gate-failure test in 8.82 seconds). A CPU-only preparation failure from passing a string to the Path-based SHA helper was fixed; its partial output is preserved at `outputs/three_stage_formal_20260909_prepare_failed`. No training update occurred during that failure. The new test exercises materialization, refuses overwrite/re-entry, and verifies that a gate failure starts no child and performs no retry.

At document creation no formal update or checkpoint exists. Actual start/end, completed updates, final/minimum loss, memory peaks, W&B and failures are appended automatically below. No formal resume has occurred. Production whole-window Flow absence, global no-supervision, overflow and real epoch-boundary cases were not observed in validation; the RH20T full-population clipping frequency remains unmeasured. Offline losses are not task success rates.
