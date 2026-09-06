# Tabletop v3 joint-init checkpoint LIBERO evaluation

- Created: 2026-09-05, Asia/Shanghai. Owner: lq.
- Purpose: evaluate the trained checkpoint on the same 2,000 official LIBERO episodes as prior evaluations, using GPUs 0,1,2,3 and the largest useful task batch with measured safe concurrency.
- Base model: Qwen3-VL-2B-Instruct.
- Checkpoint: /opt/data/private/lq/ZR-0/outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm-difference-query-nq32-tabletop-v3-joint-init/step-34184
- Training experiment: /opt/data/private/lq/ZR-0/docs/experiments/libero_wo_ecot_pt_dq32_tabletop_v3_joint_init/experiment.md; Tabletop v3 joint step-19424 initialization, then LIBERO action-only fine-tuning step-34184.
- Evaluation code: /opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration; commit 0f19a4b93d1341678a0102d3599cb1608ae5faae. Clean source, no inference code edits.
- Main checkout has pre-existing user changes. This evaluation adds only this tracked experiment document. No commit or checkpoint modification.
- Config: existing evaluation.libero_eval.launcher and scripts/run_libero_shard_supervisor.py, explicit supervisor_plan.json. Full per-shard commands, fingerprints and environments generated before model launch.
- Hardware: four NVIDIA A800-SXM4-80GB, initially idle, 81,226 MiB free/card. Container RAM limit 480 GiB, CPU quota about192 cores.
- Training stages, trainable modules, optimizer, learning rates, scheduler, losses, gradient accumulation, W&B project/group/run/URL: N/A; evaluation only, all weights inference-only.
- Training precision: BF16. Evaluation VLM BF16, checkpoint DQ FP32; runtime contract requires SDPA with 32 queries, requested backend auto. No legacy loading overrides.

## Dataset and protocol

- LIBERO checkout: /opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/LIBERO, clean commit 8f1084e3132a39270c3a13ebe37270a43ece2a01.
- Benchmark: libero_spatial, libero_object, libero_goal, libero_10; ten tasks/suite, fifty official initial states/task (indices0..49); 2,000 unique episodes.
- Split/sampling: official benchmark initial states, no train sampling or replacement. Seed7, episode_deterministic_v1 with per-episode/per-replan Flow seeds.
- Policy dataset metadata: libero_wo_ecot_pt at /opt/data/private/lq/datasets/HuggingFaceVLA/libero; checkpoint manifest must match action/state quantile statistics and observation contract.
- Inputs: current agent and wrist RGB, language task and state; no future observations or optical-flow labels.
- Evaluation images: 256x256 RGB, rotate180, aspect-preserving 448x448 zero-pad letterbox, uint8; Qwen processor rescale1/255, mean/std[0.5,0.5,0.5]; no augmentation/crop. Current frame first, agent then wrist; window1.
- TRAIN/EVAL DIFFERENCE: training used 224x224 bicubic with no letterbox. Evaluation deliberately retains previous two evaluations' 448x448 input protocol for comparison.
- State: end-effector position, axis-angle rotation and gripper qpos, effective8/padded64 dimensions.
- Action: LIBERO OSC_POSE delta position/rotation plus gripper, effective7/padded64 dimensions. Quantile q01/q99 normalization from the registered checkpoint-validated dataset.
- Action chunk, prediction horizon, execution horizon:10; execute10 steps/replan; wait10; denoise5; control20Hz; replay videos10fps.
- Success condition: official environment done=True. Rate = successful validated episode records / all2,000 records.
- Client: /opt/data/private/lq/ZR-0-eval-envs/libero-client-py3104/bin/python, CPython3.10.4, PYTHONNOUSERSITE=1.
- Server: /opt/data/private/lq/.conda/envs/zr0-eval/bin/python. Both pip check passed.
- Renderer: locked OSMesa/native libraries and fixed-scene RGB SHA256 ad6b208149c3b95295794671b638aeb9688c0583d3c8c2f2b83094ce0f788f7b.
- Thread environment for concurrent configuration: OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=LP_NUM_THREADS=1. Fixed-scene identity must still pass.

## Capacity admission

- Maximum useful batch under the existing per-task scheduler is50, since each task has50 official episodes. Larger configured batches cannot produce larger requests without changing scheduling code.
- Single service: /opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/result/eval/libero-tabletop-v3-joint-init-step34184-b50-capacity-20260905, GPU0, B50/workers50, port8440, spatialtask0, fifty isolated smoke episodes, forward audit enabled.
- Three concurrent services: /opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/result/eval/libero-tabletop-v3-joint-init-step34184-b50-3svc-capacity-20260905, GPU1, ports8441..8443, B50/workers50 per service, spatialtasks0,1,2; 150 isolated capacity episodes, forward audit enabled.
- Capacity records are not included in formal success rates.
- Admission requires finite actions, real full batches, one VLM and five DiT forwards/request, exact SDPA/DQ32 runtime identity, complete result integrity, GPU headroom >=5GiB, and safe container memory usage.
- Regression tests: runtime contract/supervisor13 tests and launcher/result store/server identity/tiny VLA43 tests passed. Tiny B1/2/4/8 batch-serial maximum error2.98e-8.
- Initial test invocation assumed pytest, which is absent. Tests were successfully run through their native unittest runner; no environment packages changed.

## Formal scheduling and execution

Pending capacity measurements. Final GPU/service/port/task mapping, launch command, start time, supervisor identity, memory measurements and output directory will be appended before formal launch.
The permanent supervisor owns each launcher and descendants, fails fast on shard failure, retains atomic episode JSON, supports same-fingerprint resume, and summarizes only complete coverage.

## Initial B50 configuration (superseded)

- Single-service smoke:50/50 complete,49 successes; formal-rate conclusion is not drawn from smoke. Peak GPU8537MiB, actualB50 seven times. All28 requests finite with VLM1/DiT5. Guard passed.
- Three-service capacity:150/150 complete,145 successes, supervisor655.508s; all133 requests finite with VLM1/DiT5. Peak combined GPU23793MiB; actual batches up to45 under concurrency. All three result stores complete.
- Both checks overlapped at200env workers. Observed container usage466,564,255,744bytes (434.522GiB) against480GiB limit; after cleanup returned toabout90GiB. No OOM, NaN or process failure.
- Formal setting: one service per GPU0,1,2,3; batch50/workers50 each, total200envs. Higher useful batching is impossible with50official episodes/task. Extra services would exceed practical host memory headroom, even though GPU VRAM is mostly free. No padding allocations are used to artificially occupy memory.
- GPU0 port8450 task ids0,2 in allfour suites:400episodes.
- GPU1 port8451 task ids1,4,6 in allfour suites:600episodes.
- GPU2 port8452 task ids3,8 in allfour suites:400episodes.
- GPU3 port8453 task ids5,7,9 in allfour suites:600episodes.
- Task groups selected using prior ActionOnly task duration estimates to reduce longest shard runtime. Suite order libero_10,libero_goal,libero_object,libero_spatial processes the long suite first. Only scheduling differs; official episode keys, seeds, inputs and success semantics remain identical.
- Formal directory: /opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/result/eval/libero-tabletop-v3-joint-init-step34184-det-v1-b50-4gpu-sdpa-20260905
- Full configuration and every launcher command: /opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/result/eval/libero-tabletop-v3-joint-init-step34184-det-v1-b50-4gpu-sdpa-20260905/supervisor_plan.json.
- Audit disabled for formal evaluation. All ports and candidates checked before launch, exact fingerprints checked again against runtime manifests.

```bash
cd /opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration
nohup setsid env PYTHONNOUSERSITE=1 PYTHONPATH=/opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration:/opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/lerobot OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 /opt/data/private/lq/.conda/envs/zr0-eval/bin/python scripts/run_libero_shard_supervisor.py /opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/result/eval/libero-tabletop-v3-joint-init-step34184-det-v1-b50-4gpu-sdpa-20260905/supervisor_plan.json > /opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/result/eval/libero-tabletop-v3-joint-init-step34184-det-v1-b50-4gpu-sdpa-20260905/supervisor.log 2>&1 < /dev/null &
```

## Initial B50 launch record (superseded)

- Start:2026-09-05T22:29:57+08:00.
- Detached supervisor PID1096984. Background session, owned launcher/descendant lifecycle through ManagedSubprocessRunner.
- Checkpoint content SHA256:109e9d6b8ea05acaa7cf19255df63ea5cee6f533f3067901aabab2224624e7fa.
- Source snapshot SHA256:9693a3e034924baafec6c062a4f902de49c688c7ec3e6453aca0096389c60b30.
- Prelaunch candidate generation independently rehashed source/checkpoint and validated all2,000 unique keys, four free ports, clean source, pinned cleanLIBERO and completed smoke result stores.
- Current status: launched, verifying runtime manifests, four client connections, first formal records, and container memory before handoff.
- B50 final status: supervisor failed with primary error "received signal15", elapsed870.1297871097922s after intentional stop. All owned processes reclaimed; GPU memory returned to2MiB/card and container usage toabout73GiB. Allfirst task groups had50/50 records; no published record was removed.
- B50 rejection: at2026-09-05T22:42:54+08:00 first task transition reachedcgroup480GiB; memory.failcnt increased0->18. Its single-task smoke capacity estimate was insufficient for task-transition peaks. Do not resumeB50 unattended.

## Current B45 configuration

- New output directory:/opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/result/eval/libero-tabletop-v3-joint-init-step34184-det-v1-b45-4gpu-sdpa-20260905
- Batch/workers45per service, four services onGPU0,1,2,3;180environmentworkers total. One service/GPU. Ports8460,8461,8462,8463 respectively.
- Taskids andsuiteorder identical toB50 plan;50official episodes/task,2,000episodes total. This newrun reruns all2,000;B50 partial records are excluded because batch/workers changefingerprints.
- Existing client/server,SDPA,DQ32,source/checkpoint hashes,images,seeds,actions,videos and success criterion unchanged.
- Complete launcher commands andkeys:/opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/result/eval/libero-tabletop-v3-joint-init-step34184-det-v1-b45-4gpu-sdpa-20260905/supervisor_plan.json.
- Preflight: recompute candidate fingerprints, confirm freeports8460..8463, clean evaluation source/pinnedLIBERO, immutablecheckpoint, and strict runtime identity.
- Initial additional gate was superseded by the user's instruction to proceed with the fixed configuration without further capacity trials; task-transition memory behavior has not yet been verified forB45 at handoff.
- No inference code edits; only this tracked report is staged. No commit.
- B45 launched:2026-09-05T22:50:05+08:00; supervisorPID1266841, detachedsession. Ports8460..8463 free at preflight; allcandidate fingerprints recomputed successfully. Waiting for runtime/startup and firsttask-transition memory admission.
- User steering at2026-09-05T22:51:50+08:00: keep an appropriate fixed configuration and proceed directly; no further capacity trials or retuning. B45/workers45 onGPUs0..3 is now fixed. SupervisorPID1266841 confirmed alive withPPID1 and independentSID1266841; no startup errors in shard logs. FullB45 task-transition validation is not yet complete at handoff. Permanent supervisor continues formal evaluation and automatic complete-only summary.

```bash
cd /opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration
nohup setsid env PYTHONNOUSERSITE=1 PYTHONPATH=/opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration:/opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/lerobot OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 /opt/data/private/lq/.conda/envs/zr0-eval/bin/python scripts/run_libero_shard_supervisor.py /opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/result/eval/libero-tabletop-v3-joint-init-step34184-det-v1-b45-4gpu-sdpa-20260905/supervisor_plan.json > /opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/result/eval/libero-tabletop-v3-joint-init-step34184-det-v1-b45-4gpu-sdpa-20260905/supervisor.log 2>&1 < /dev/null &
```
