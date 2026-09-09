# Four-Dataset DQ32 H10 GBS256 Seed42

## Current Status: Formal AR Stopped After Step 1256

Checked on2026-09-06 Asia/Shanghai. Formal AR launched12:54:36 and the sequential runner recorded exit1 at15:12:14, after1256 of27582 completed optimizer updates (321536 global samples). Final loss0.2987856567, minimum recorded training loss0.1740577966, LR9.109862219e-6, Query/VLM gradient norms0.05120210/1.50572026; recorded target truncation remains zero and W&B was connected. No validation or rollout was run. Actual micro32/GAS2/world4/GBS256 and all planned training parameters remained unchanged.

The first observed failure was rank0 DataLoader worker2 opening a PyAV decoder for DROID global_index2041670: `av.error.MemoryError: [Errno 12] Cannot allocate memory`. The sample maps to episode7035/frame52, main-camera video `videos/observation.images.exterior_1_left/chunk-000/file-022.mp4`, timestamp560.5333333651224. The existing dataset error wrapper propagated this error; Accelerate terminated the remaining ranks and the runner stopped all subsequent stages. This traceback is a CPU video-decoder allocation failure, not a reported CUDA OOM or token-length failure.

Read-only diagnosis: one CPU decode of that exact frame through the installed production PyAV path succeeded, producing shape[1,3,180,320]. No dataset scan, token audit, optimizer update or training restart was performed. W&B's last host-memory sample at15:11:46 showed1541038.98MiB available; that host-level measurement does not establish worker/container headroom. The container memory limit is480GiB, with historical peak480GiB/failcnt18/oom_kill11. These counters have no event timestamps; there is no matching kernel OOM-kill record in15:00-15:20, so they do not establish the cause of this failure. Exact worker memory/thread usage at the exception is unavailable, and resource exhaustion versus decoder resource accumulation remains unresolved.

No formal checkpoint exists: the run stopped before its first configured5000-step save, so the1256 formal updates cannot be resumed from disk. Existing probe step2/step3 checkpoints and all prior artifacts remain preserved and are not substitutes for formal state. The formal tmux/runner/ranks have exited; GPUs0-3 were idle at diagnosis. Joint probe/resume/formal never started, and no automatic continuation remains active.

Evidence: run-root `formal_ar.ar-formal.ar_formal_after_gate.log` (first error line3346), `formal/ar/training_metrics.jsonl`, `lifecycle.jsonl`, and sibling `.formal_continue.log`. This investigation updates documentation only; no code, configuration, checkpoint or dependency is changed, and no commit is created.

## Historical Status: Formal AR Running

As of2026-09-06 12:58:51 Asia/Shanghai, formal AR has recorded17 optimizer updates. First step loss3.8246400356; step17 loss3.3913247585, LR1.2509064539521395e-7, Query/VLM gradient norms426.51556/405.86154. All recorded loss/LR/gradients are finite and target truncation is zero. Each update uses256 global samples; current peak reserved72.609375GiB. W&B remote_available=1. This is formal training under the27582-step budget, not an extended probe.

Active runner PID2583187, tmux `four_dataset_h10_formal_20260906`; formal launcher/process-group2583879 started12:54:36. Actual prepared scale: world4, micro32, GAS2, GBS256. CLI confirms fresh base `/opt/data/private/lq/models/Qwen3-VL-2B-Instruct`, seed42, resume_training=False, fresh Query/optimizer/scheduler/global step; the base processor audit is selected. Budget27582/warmup1379/save5000 remains unchanged. No formal checkpoint is claimed before its configured save point.

Formal output: `outputs/four_dataset_dq32_h10_gbs256_seed42_20260906_004536/formal/ar`. Runner log: sibling `.formal_continue.log`; train log: run-root `formal_ar.ar-formal.ar_formal_after_gate.log`; metrics: `formal/ar/training_metrics.jsonl`. W&B: https://wandb.ai/jumbo3r-zhejiang-university/zr0-stage05-four-dataset/runs/four_dataset_dq32_h10_gbs256_seed42_20260906_004536-ar-formal . Source snapshot: `source_identity_ar_formal_after_gate.json`; environment/diff/start records use the same suffix.

The same background runner waits for successful AR completion and complete checkpoint/processor-audit/resource gates, then runs the original Joint batch probes, one-update recovery gate and fresh95423-step formal Joint stage with4771 warmup. Joint has not started. Failed report reuse records old/new fields and stops; it cannot launch another full audit. Formal failures stop subsequent stages. The only full audit in this task completed once; actual recovery completed exactly one update2->3 and its saved processor reused that audit. Targeted checks:7 selector,4 preflight and8 continuation cases passed, plus CLI/dry-run/compile/diff checks. All original products and user changes are preserved; no commit was made. Earlier stopped-stage entries below are historical.

## Authorized Saved Processor Audit

2026-09-06: the user authorized diagnosis, an independent existing-tool audit for the saved processor, and continuation of the original experiment. Base and step2 checkpoint loaded processor/tokenizer classes, full tokenizer backend, vocabulary/added tokens, special token maps/IDs, effective chat template, padding/truncation settings and image processor dictionaries are equal. Existing adapter comparison on the four longest two-view samples and RH20T wrist-fallback sample found exactly equal input_ids, labels, attention masks, image placeholders, grids and pixels. These five samples diagnose serialization changes only and do not establish full-dataset equivalence. The result is retained in `processor_comparison.json`.

Only processor audit artifacts and runner selection are adapted. `--processor-audits` selects a trusted report by existing processor file identities and runs the existing full validator, including at stage-end save gates. Per the latest user instruction, no further full audit is started automatically: failed reuse records old/new fields and whether existing data/runtime identities changed, then stops. Directory, global-step and model-weight changes alone do not trigger scanning. H10 sidecars, eligible indices, statistics, old reports and checkpoints remain unchanged. No new semantic hash or contract system is introduced.

The single authorized complete audit was not restarted. Audit tmux PID2489105 ran the existing tool from12:05:46 to12:29:28 Asia/Shanghai and completed validation. Counts: DROID9507523, Household794199, Tabletop310743, RH20T3509414; total14121879; required_max_length941. All dataset measurements and data/implementation identities equal the base audit; differing report fields are only processor identity and consequent content hash. Processor differences are recorded source path and model-file records. Applicable identities are fully recorded in `processor_audits/audit-001/report.json`, with independent `trusted_spec.json`: report content hash `a16346ec6747e8813bed735c05fcc972c93578f0c603e10bfde29891a6f9c673`, file SHA256 `1fb9ba433fd0ffb81dcbe4257515c4c3efc05620b9ff1acd63eb61309e49d121`, implementation SHA256 `ca23666ebe657b28d34602d388b2a98c679478664fb69588e85f66a6241ee004`. This report may be reused at any path that passes all existing identity checks. Base fresh training retains its original base-processor report. Audit completion is not recovery success or formal training startup.

Recovery output is independent: `probes/ar/mbs32_gas2_resume_processor_audit`. It loads the original `probes/ar/mbs32_gas2/latest-model-optimizer-lr` and saves only the resumed step3 through the production save path. Original step2 model/optimizer/processor files are preserved. Joint recovery similarly uses an independent suffix directory.

Actual recovery: runner2500441 / tmux `four_dataset_h10_processor_continue_20260906` started12:35:59; AR-resume launcher/process-group2501016 started12:37:02 and exited zero12:42:29 Asia/Shanghai. Four-rank DeepSpeed restored step2 and scheduler8/9 at LR1.3425421036992092e-6, skipped four prior micro-batches and executed exactly one update. Step3 loss1.9587852955, Query gradient67.7206497, VLM gradient65.6937408, LR1.1533337816991923e-6,256 samples,154843 supervised tokens, zero truncation. Saved scheduler12/13; peak allocated69.94445GiB/reserved77.08008GiB. Original step2 checkpoint remains intact. Existing checkpoint preflight and saved-processor audit reuse both passed for the independent step3 checkpoint at12:47:52, without another data scan.

At12:42:30, after training exited successfully, the sequential runner stopped at its resource gate: no compute processes, eachGPU81226MiB free, but one GPU0 utilization sample was100%. Subsequent observations showed four idle GPUs. Resource rules were retained; recovery was not retried. The completed-gate continuation below requires the recorded successful recovery and verified step3 checkpoint, revalidates it, and starts fresh formal AR from the Qwen base. No probe training is repeated. Planned tmux: `four_dataset_h10_formal_20260906`; log: sibling `four_dataset_dq32_h10_gbs256_seed42_20260906_004536.formal_continue.log`.

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:lerobot /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -u scripts/run_stage05_h10_experiment.py --config configs/four_dataset_dq32_h10_gbs256_seed42_20260906.json --continue-ar-formal --continuation-id ar_formal_after_gate --processor-audits
```

Independent audit directory: run-root `processor_audits/audit-001`; CPU audit tmux `four_dataset_h10_processor_audit_20260906`. The directory records the complete audit command, processor path and log. Continuation is planned in tmux `four_dataset_h10_processor_continue_20260906`, using the command below; runtime PID and results will be appended. AR uses the existing32/2 step2 checkpoint for only one recovery update, then formal AR starts fresh from the Qwen base for27582 steps/warmup1379. Successful formal AR automatically gates the original Joint probe/resume/formal95423/warmup4771 sequence. All original configuration below remains unchanged.

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:lerobot /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -u scripts/run_stage05_h10_experiment.py --config configs/four_dataset_dq32_h10_gbs256_seed42_20260906.json --continue-ar-resume-gate --continuation-id processor_audit --processor-audits
```

## Authorized CUDA Correction And Continuation

2026-09-06: the user authorized correction of the runner's CUDA visibility and continuation from the existing AR32/2 step2 checkpoint. Preflight now exposes GPUs0,1,2,3 while the unchanged validator loads tensors on CPU. At 11:35:39 Asia/Shanghai the actual checkpoint passed the complete existing AR resume preflight, including scheduler/client/optimizer state. GPUs0-3 were idle, with 81226MiB free each and about254TiB disk free. This preflight is not itself a distributed recovery test.

The explicit continuation skips AR64/1 and fresh AR32/2; it reuses all H10 preparation, audit, samples and configuration. It runs AR resume once to step3 through the existing four-rank DeepSpeed entry, then verifies the result. On success formal AR starts from the Qwen base with fresh seed42/Query/state, micro32/GAS2,27582 updates and1379 warmup. The original gated Joint probe/resume/fresh flow follows formal AR completion, with95423 formal updates and4771 warmup. All original parameters below remain in force. Any recovery failure stops the sequence without retry.

Continuation tmux: `four_dataset_dq32_h10_gbs256_seed42_20260906_continue`; log: `outputs/four_dataset_dq32_h10_gbs256_seed42_20260906_004536.continue.log`. The original source/environment/start records remain intact; new records use `_ar_gate_continuation` names. An exclusive runner lock and one-use marker prevent duplicate continuation. Actual outcome is recorded below.

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:lerobot /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -u scripts/run_stage05_h10_experiment.py --config configs/four_dataset_dq32_h10_gbs256_seed42_20260906.json --continue-ar-resume-gate
```

Actual continuation outcome (Asia/Shanghai): runner PID2479815 started11:42:53. Corrected checkpoint preflight passed11:43:44. AR-resume launcher PID/process-group2480202 started11:43:45 with micro32/GAS2/GBS256 and the original step2 checkpoint, then stopped11:44:52 with exit1. Its existing token-audit validation reported `Stage05 token audit is stale: processor/tokenizer files identity changed; rerun the complete token audit`. The launcher failed before creating Accelerate/DeepSpeed ranks, so there were zero additional optimizer updates and no step3 checkpoint. Formal AR and every Joint stage remain unstarted. The tmux session exited, no experiment GPU processes remain, and each GPU again has81226MiB free. No automatic continuation or retry remains active.

Read-only file-identity evidence: the report's `audited_processor_path` is `/opt/data/private/lq/models/Qwen3-VL-2B-Instruct`; the resume launcher validates `probes/ar/mbs32_gas2/latest-model-optimizer-lr`. The report's `tokenizer.json` is7032403 bytes/SHA256 `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7`, while the saved checkpoint's is11422654 bytes/SHA256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`. The base has `chat_template.json`; the checkpoint has `chat_template.jinja`. Other differing identity records include config/preprocessor/tokenizer config, merges and auxiliary token files. These file differences are established; semantic equivalence has not been evaluated. The audit and trusted hashes were not altered, and no data scan or audit rerun was performed. This is an additional blocker beyond CUDA visibility, left unchanged under the user's stop-on-failure instruction.

Failure traceback: run-root `probes_ar_mbs32_gas2.ar-resume.log`; sequence status: `lifecycle.jsonl` and sibling `.continue.log`. Validation for this correction:4 targeted tests passed/3 deselected; continuation dry-run, CLI help, Python compile, Bash syntax and diff checks passed. All original artifacts and staged/user edits remain preserved; no commit was created.

- Created: 2026-09-06 Asia/Shanghai. Owner: lq.
- Purpose: fresh AR for approximately 0.5 epoch, then fresh Joint for one epoch.
- Base: `/opt/data/private/lq/models/Qwen3-VL-2B-Instruct`.
- Code baseline: `1f88937a1375eb86b5e78c759815301b67524882`, current uncommitted worktree and staged repairs. No commit is created. The run records its actual diff and source hashes.
- Configuration: `configs/four_dataset_dq32_h10_gbs256_seed42_20260906.json`.
- Output: `outputs/four_dataset_dq32_h10_gbs256_seed42_20260906_004536`.
- Registry: independent selection derived from the four confirmed entries in `dataset2feature.yaml`; historical H32 registry and artifacts remain unchanged.
- Seed: 42, reset for each fresh probe and formal stage.
- W&B: project `zr0-stage05-four-dataset`, group equal to experiment name, stage-specific names and IDs, policy `best_effort` as explicitly requested. Run URLs are recorded when available. No raw samples are uploaded.
- Local records: `training_metrics.jsonl`, TensorBoard, initialization/resolved/seen manifests, `train.log`, checkpoint, and orchestrator stage records.

## Data And Eligibility

| Dataset | Source | AR frames | Generated H10 Joint frames | Main/wrist |
| --- | --- | ---: | ---: | --- |
| DROID | `/opt/data/private/lq/datasets/droid_1.0.1_stage05_full_95658_20260831` | 9507523 | 19821342 | exterior_1_left / wrist_left |
| Household | `/opt/data/private/lq/datasets/molmoact_dataset_household-v3_stage05` | 794199 | 794199 | first_view / wrist_image |
| Tabletop | `/opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05` | 310743 | 310743 | first_view / wrist_image |
| RH20T | `/opt/data/private/lq/datasets/RH20T-v30_stage05` | 3509414 | 3501934 | exterior_1_left / wrist_left |

No validation/test split or held-out evaluation is added. AR uses only existing eligible `train_data`, complete normalized target and assistant termination; `slot_data` remains metadata. AR natural probabilities are 67.3248%, 5.6239%, 2.2004%, 24.8509%. Joint includes only FM_count>0; text-bearing samples use AR+FM and textless samples only FM. Expected Joint probabilities are 81.1412%, 3.2512%, 1.2721%, 14.3356%, subject to generated H10 counts. Existing reproducible natural mixing and epoch shuffle remain unchanged for formal training; probes use explicitly recorded representative real samples only.

The completed H10 build confirms all counts above and matches the earlier compact-validity prediction. Total Joint frames are 24428218, giving 95423 optimizer updates and 4771 warmup steps. Existing rank padding adds two repeated samples; the final global batch has 188 samples. All four AR eligible arrays are identical to their historical counterparts. DROID state statistics lose 1550 eligible rows relative to H32 and were regenerated. All four action-statistics payloads and the other three state-statistics payloads remain identical. No normalization rule is changed. Detailed results are in the run's `data_summary.json` and `data_preparation_progress.json`.

Canonical state: absolute base/world EEF `[x_m,y_m,z_m,roll_rad,pitch_rad,yaw_rad,gripper_open]`. Canonical action: native-next-step relative EEF `[dx_m,dy_m,dz_m,droll_rad,dpitch_rad,dyaw_rad,gripper_open]`; SciPy lowercase xyz RPY with wrapped deltas, metres, gripper 1=open. Independent q01/q99 and original stats_key per source; clip remains [-15,15]. Seven real dimensions padded to 64, with dimension and episode-tail masks. H=10 is the action chunk/prediction horizon throughout config, future Expert contract, dataset, masks, and checkpoint; execution horizon is not applicable to pretraining. FPS: DROID 15, other sources 10.

## Images And Model

Only main then wrist; never second external view. RH20T wrist skew beyond 100 ms retains main only and is counted. Original Molmo images are 640x480; audited DROID/RH20T video frames are 320x180. Existing processor uses RGB, direct bicubic resize to 224x224 without preserving aspect ratio, no crop/pad/letterbox/augmentation, rescale 1/255 and mean/std [0.5,0.5,0.5]. Patch/temporal/merge sizes 16/2/2; 49 visual tokens per view. Max length 1024, complete targets with no truncation. A new token audit binds the selected H10 sidecars and the unchanged production tokenization functions. No evaluation or rollout is requested.

AR trains visual encoder, visual merger/projector, language model and Difference Query; Expert is absent and its H10 configuration is saved only as a future Joint reference. Joint loads only final formal AR VLM/Query, randomly initializes Expert under seed 42, and trains all previous modules plus state/action encoders, DiT/cross-attention, and action decoder. No historical Expert or probe state initializes formal training. Joint optimizer, scheduler, global step and sampler all start fresh.

Preserve [C,Q,T,P]: Q reads C and bidirectional Q, no T/P; T reads Q and causal text prefix, no direct C. Expert receives only final RMSNorm Query through VLM cross-attention, with native state/noisy-action/timestep inputs and no Query detach. Text uses teacher forcing; generation is unchanged.

## Optimizer And Scale

- AR: total_loss=1*AR_loss, normalized over global valid target tokens.
- Joint: total_loss=1*AR_loss+5*FM_loss, each branch normalized over its existing global valid elements.
- One AdamW group, peak LR 1e-5, minimum LR 1e-6, betas (0.9,0.95), epsilon 1e-8, weight decay 0.01, clip 1.0. No LR scaling or module multipliers.
- Cosine with 5% warmup. AR budget 27582 optimizer steps, warmup 1379. Expected Joint budget 95423 steps, warmup 4771; final sampler counts determine the exact budget.
- Four A800-SXM4-80GB GPUs 0,1,2,3; BF16, SDPA, existing gradient checkpointing.
- global_batch_size = 4 * micro_batch * GAS = 256. Probe candidates 64/1,32/2,16/4,8/8,4/16,2/32,1/64. Actual per-stage choice is pending.
- Each candidate: at most two optimizer updates. Successful candidate: save/resume with at most one additional update. Only explicit CUDA OOM allows a lower candidate.
- Formal saves: every 5000 optimizer steps and stage end via existing model and recoverable-state mechanism. Logging every step. No validation interval, early stopping, pilot, or extra training.
- Paper schedule 3e-5/3e-6 is not used. Repository Stage05 defaults H32/GBS128 are not this experiment's H10/GBS256 choice.

## Commands And Lifecycle

Preparation and sequential runner commands, exact stage argv, environment versions, source hashes, chosen micro/GAS, peak memory, PID, tmux session, timestamps and exit codes are recorded in the output directory before each launch. Every probe/formal directory receives this document before model initialization. Formal AR completion and checkpoint validation are required before any Joint probe; Joint probe state is discarded for fresh formal initialization.

Resource gate initially found all four GPUs idle, approximately 254 TiB filesystem free and 1.7 TiB host memory available. A conservative 1 TiB free-disk gate covers new checkpoints and probes; old completed Joint snapshots are about 5.1 GiB and recoverable state about 41 GiB. No old checkpoints are deleted.

Initial preparation command (CPU only; tmux session `four_dataset_h10_prepare_20260906`):

```bash
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.:lerobot /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -u scripts/prepare_stage05_h10_experiment.py --config configs/four_dataset_dq32_h10_gbs256_seed42_20260906.json
```

Sequential run command, to be started after the preparation marker and final dry-run pass:

```bash
env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:lerobot /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -u scripts/run_stage05_h10_experiment.py --config configs/four_dataset_dq32_h10_gbs256_seed42_20260906.json
```

Preparation log: `outputs/four_dataset_dq32_h10_gbs256_seed42_20260906_004536.prepare.log`. Scheduler CPU checks: 35 passed, 19 deselected. Launcher and representative batch sharding checks: 7 passed, 8 deselected. Production-entry help, Python compile, Bash syntax and diff checks passed. These CPU checks are not a real distributed recovery result.

Preparation passed: all eight H10 sidecars were generated and validated, and the independent token audit confirms required_max_length=941 <= 1024. Content hash `3774558a7bd223153376d383a119c4444548b42eccc90c02dd42f8f6374b1dbe` and file SHA256 `1105e9ad1172e3eefb9030cac6d48832bd3ae16b3a646873d09c00a2a59e0c8b` are pinned in the independent experiment configuration. Joint/downstream deep state preflight and token-audit helper defaults remain deferred.

## Actual GPU Gates And Stop

- Local times: 2026-09-06 Asia/Shanghai. AR64/1 started 01:39:36 and failed 01:42:05 with explicit CUDA OOM on all ranks before a completed optimizer update. Only that launch's process group was cleaned up.
- AR32/2 started 01:42:06, completed exactly two updates and exited normally at 01:45:13. Step losses: 3.1368899345 and 2.2158646584; both had zero target truncation and finite recorded gradients. Each update used 256 global samples and two micro-batches per rank.
- Peak PyTorch allocated/reserved memory: 69.9262/73.9629 GiB. An nvidia-smi observation during saving showed 77623 MiB used and 3605 MiB free on the most occupied GPUs; this observation is not a complete device-memory peak trace.
- AR checkpoint: `probes/ar/mbs32_gas2/latest-model-optimizer-lr`; model snapshot: `probes/ar/mbs32_gas2/step-2`. Saved scheduler last_epoch=8 and _step_count=9; saved seen manifest global_step=2 and total_seen=512. Scheduler/count validation passed before optimizer deserialization failed. Full recoverability has not been verified.
- At 01:46:02 the sequential runner stopped during AR resume preflight. The newly added `scripts/run_stage05_h10_experiment.py::verify_run` explicitly hides CUDA for the preflight subprocess. Loading real BF16 DeepSpeed optimizer state imports DeepSpeed/Triton, whose installed autotuner requires an active driver and raises `RuntimeError: 0 active drivers ([]). There should only be one.` The validator reports `Stage05 AR resume optimizer state cannot be loaded`. This is a launch-environment failure, not an OOM or evidence of scheduler corruption. No contract check was skipped and no automatic retry was made.
- Resume updates: 0. Formal AR updates: 0. Joint probe/resume/formal: not started. The runner and this experiment's GPU processes have exited; no automatic Joint continuation remains active. All H10 artifacts, logs and approximately 37 GiB of new experiment output are retained.
- Background identifiers: preparation tmux `four_dataset_h10_prepare_20260906`, PID2193526; runner tmux `four_dataset_dq32_h10_gbs256_seed42_20260906`, PID2210107; both sessions exited. Probe launch process groups: 2210197 (64/1), 2286942 (32/2).
- W&B successful two-step probe: https://wandb.ai/jumbo3r-zhejiang-university/zr0-stage05-four-dataset/runs/four_dataset_dq32_h10_gbs256_seed42_20260906_004536-ar-probe-mbs32 . W&B reported zero uploaded media/artifacts.
- Logs: `outputs/four_dataset_dq32_h10_gbs256_seed42_20260906_004536.runner.log`, and run-root `probes_ar_mbs64_gas1.ar-smoke.log`, `probes_ar_mbs32_gas2.ar-smoke.log`, `lifecycle.jsonl`.
- CPU checks: 35 scheduler/preflight tests and 7 launcher/sharding tests passed. The actual H10 formal launcher dry-run, CLI help, Python compile and Bash syntax passed. No full suite, long pilot, data regeneration after validation, LIBERO rollout, commit, reset, or historical-checkpoint modification occurred.
