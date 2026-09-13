# LIBERO evaluation: Stage3 action-only DQ32 checkpoint

- Experiment name: `libero_stage3_step14000_actiononly_dq32_det_v1_b45_4gpu_sdpa_20260911`
- Purpose: evaluate the Stage3 LIBERO action-only checkpoint on the official 2,000-episode benchmark using all four A800 GPUs.
- Created: 2026-09-11, Asia/Shanghai.
- Owner: lq.
- Base model: Qwen3-VL-2B-Instruct with Difference Query and 64D Action Expert.
- Checkpoint: `/opt/data/private/lq/ZR-0/outputs/ckpts/ZR0-stage3-step14000-LIBERO-action-only-dq32-h10-gbs64-seed42/attempt-000/recovery_checkpoints/step-034184/latest-model-optimizer-lr`.
- Training source: `/opt/data/private/lq/ZR-0/docs/experiments/libero_stage3_step14000/experiment.md` and the checkpoint metadata.
- Evaluation code: `/opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration`.
- Source commit: `0f19a4b93d1341678a0102d3599cb1608ae5faae`.
- LIBERO commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Source snapshot SHA-256: `9693a3e034924baafec6c062a4f902de49c688c7ec3e6453aca0096389c60b30`.
- Current root checkout contains unrelated user training changes; they are not used by this evaluation and are not reverted.
- Evaluation output: `/opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/result/eval/libero-stage3-step34184-actiononly-dq32-det-v1-b45-4gpu-sdpa-20260911`.
- W&B: not applicable; this is inference-only evaluation.
- Status before launch: preflight passed; candidate fingerprints and supervisor plan generated.

## Dataset and protocol

- Benchmark: official LIBERO `libero_10`, `libero_goal`, `libero_object`, and `libero_spatial` suites.
- Coverage: all 10 official tasks per suite, initial-state indices 0-49, 50 episodes/task, 2,000 unique episode keys.
- Seed: 7; RNG protocol `episode_deterministic_v1` with deterministic per-episode/replan seeds.
- LIBERO checkout: `/opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/LIBERO`.
- Client interpreter: `/opt/data/private/lq/ZR-0-eval-envs/libero-client-py3104/bin/python` (CPython 3.10.4 environment, user site disabled).
- Server interpreter: `/opt/data/private/lq/.conda/envs/zr0-eval/bin/python`.
- Renderer: OSMesa with the locked native renderer identity and fixed-scene checksum.
- Observation: current agent and wrist RGB views, task language, and robot state; no future observation or optical-flow label.
- Image processing: render 256x256 RGB, rotate 180 degrees, aspect-preserving zero-pad letterbox to 448x448, Qwen processor normalization, no crop or augmentation.
- State/action: effective state dimension 8 and OSC_POSE action dimension 7, both padded to 64D with registered quantile normalization.
- Action horizon/prediction horizon: 10; execute 10 steps per replan; wait 10 steps; five denoising steps; control 20 Hz; videos 10 fps.
- Success condition: official LIBERO environment `done=True`.

## Model and runtime contract

- Checkpoint kind: `action_only`; BF16 Qwen VLM and FP32 Difference Query weights.
- Checkpoint content SHA-256: `3b600e30627b1e0807c658965c907ac29fc9285930de6e10cfcba87d6ea7d0da`.
- Difference Query sidecar: version 1, enabled, 32 queries, hidden size 2048, required attention backend `sdpa`.
- Difference Query sidecar SHA-256: `9ee4bb2bce02ceeef95a029b1f6055644e7454d4e0dc214e303d245bc3e7553c`.
- Runtime contract: requested `auto`, required/resolved `sdpa`, contract version 1, canonical SHA-256 `72a705ce800b17086facdee0c906d21667dacc787dc0eff0542b638416bad9f8`.
- Model load must attest `actual=sdpa`, Difference Query enabled, and `num_difference_queries=32` before serving clients.
- No checkpoint file is modified or migrated.

## Scale and scheduling

- Hardware: four NVIDIA A800-SXM4-80GB GPUs, devices 0, 1, 2, and 3.
- Services: one SDPA server per GPU; ports 8470, 8471, 8472, and 8473.
- Fixed batch configuration: `batch_size=45`, `num_env_workers=45` per service, 180 concurrent LIBERO workers total.
- Rationale: B45 is the largest previously validated safe formal configuration. B50 was rejected after a task-transition peak reached the 480 GiB container memory limit; artificial VRAM allocation is not used.
- Shard mapping, all-four-suite task coverage, candidate fingerprints, and full commands are recorded in `supervisor_plan.json`.
- Permanent supervisor: `scripts/run_libero_shard_supervisor.py`; it owns launcher/client/server process trees, fails fast, preserves published episode JSON, and only summarizes complete coverage.

## Preflight validation

- Checkpoint files and required DQ/Action Expert artifacts present; metadata identifies `action_only`, horizon 10, and manifest format 4.
- Both server and client `pip check` passed.
- Strict client/environment identity passed, including dependency lock, Python user-site isolation, LIBERO origin/commit, OSMesa/native libraries, renderer lock, and fixed-scene checksum.
- Server identity passed on an A800 with CUDA 12.4 and reported the exact DQ32/SDPA contract above.
- Runtime-contract tests: 8 passed. Supervisor lifecycle/failure/resume tests: 5 passed.

## Launch command

The exact detached launch command and every shard launcher command are stored in the run directory `supervisor_plan.json`; the supervisor is started only after candidate fingerprints and port/GPU checks pass.

```bash
cd /opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration
nohup setsid env PYTHONNOUSERSITE=1 PYTHONPATH=/opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration:/opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/lerobot OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 /opt/data/private/lq/.conda/envs/zr0-eval/bin/python scripts/run_libero_shard_supervisor.py /opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/result/eval/libero-stage3-step34184-actiononly-dq32-det-v1-b45-4gpu-sdpa-20260911/supervisor_plan.json > /opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/result/eval/libero-stage3-step34184-actiononly-dq32-det-v1-b45-4gpu-sdpa-20260911/supervisor.log 2>&1 < /dev/null &
```

## Completion record

- Actual start: `2026-09-11 13:35:31 Asia/Shanghai` (supervisor attempt).
- Actual end: `2026-09-11 14:46:23 Asia/Shanghai` (combined summary and supervisor result publication).
- Supervisor elapsed: `4251.538 s` (`1 h 10 min 51.5 s`); measured throughput `747.58 episodes/hour`.
- Final status: `complete`; all four launchers exited successfully, all process trees were reaped, and post-run GPU memory returned to 2 MiB per card.
- Per-shard episode counts: GPU0 `400/400`, GPU1 `600/600`, GPU2 `400/400`, GPU3 `600/600`; combined `2000/2000`, with no missing episode keys.
- Overall result: `1963/2000` successful, success rate `98.15%`, mean episode duration `113.735 s`.
- Suite results: `libero_10` `478/500` (`95.60%`); `libero_goal` `494/500` (`98.80%`); `libero_object` `496/500` (`99.20%`); `libero_spatial` `495/500` (`99.00%`).
- Actual batch histogram is recorded in `summary/summary.json` and includes batch size 45 for 337 requests; model forward count was 1,974.
- Peak sampled GPU memory: GPU0 `8,757 MiB`, GPU1 `10,987 MiB`, GPU2 `10,161 MiB`, GPU3 `9,011 MiB` (each card has 81,920 MiB total); sampled utilization peaked at 97%, 98%, 96%, and 97% respectively.
- No NaN, CUDA OOM, launcher failure, interruption, or resume was observed; no configuration deviation was recorded.
- Authoritative artifacts: `supervisor_result.json`, `combined_manifest.json`, `summary/summary.json`, `summary/summary.csv`, and `summary/summary.md` in the evaluation output directory.
