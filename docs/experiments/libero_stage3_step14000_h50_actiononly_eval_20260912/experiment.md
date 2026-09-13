# LIBERO evaluation: Stage3 h50 action-only DQ32 checkpoint

- Experiment name: `libero_stage3_step14000_h50_actiononly_dq32_det_v1_b45_4gpu_sdpa_20260912`
- Purpose: evaluate the h50-trained Stage3 LIBERO action-only checkpoint with the established 2,000-episode LIBERO protocol.
- Created: 2026-09-12, Asia/Shanghai.
- Owner: lq.
- Base model: Qwen3-VL-2B-Instruct with Difference Query and 64D Action Expert.
- Checkpoint: `/opt/data/private/lq/ZR-0/outputs/ckpts/ZR0-stage3-step14000-h50-LIBERO-action-only-dq32-h10-gbs64-seed42/step-34184`.
- Training source: checkpoint metadata and the corresponding Stage3 training experiment documentation.
- Evaluation code: `/opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration`.
- Source commit: `0f19a4b93d1341678a0102d3599cb1608ae5faae`.
- LIBERO commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Source snapshot SHA-256: `9693a3e034924baafec6c062a4f902de49c688c7ec3e6453aca0096389c60b30`.
- Current root checkout contains unrelated user training changes; they are not used by this evaluation and are not reverted.
- Evaluation output: `/opt/data/private/lq/ZR-0-worktrees/libero-batch-eval-main-integration/result/eval/libero-stage3-step34184-h50-actiononly-dq32-det-v1-b45-4gpu-sdpa-20260912`.
- W&B: not applicable; this is inference-only evaluation.

## Dataset and protocol

- Benchmark: official LIBERO `libero_10`, `libero_goal`, `libero_object`, and `libero_spatial` suites.
- Coverage: all 10 official tasks per suite, initial-state indices 0-49, 50 episodes/task, 2,000 unique episode keys.
- Seed: 7; RNG protocol `episode_deterministic_v1`.
- Client interpreter: `/opt/data/private/lq/ZR-0-eval-envs/libero-client-py3104/bin/python` (CPython 3.10.4, user site disabled).
- Server interpreter: `/opt/data/private/lq/.conda/envs/zr0-eval/bin/python`.
- Renderer: OSMesa with the locked native renderer identity and fixed-scene checksum.
- Image processing: render 256x256 RGB, rotate 180 degrees, aspect-preserving zero-pad letterbox to 448x448, Qwen processor normalization, no crop or augmentation.
- Action/state: 7D OSC action and 8D state padded to 64D with the registered quantile normalization; action horizon and prediction horizon 10; execute 10 steps per replan; wait 10 steps; five denoising steps; control 20 Hz; videos 10 fps.
- Success condition: official LIBERO environment `done=True`.

## Model and runtime contract

- Checkpoint kind: `action_only`; BF16 Qwen VLM and FP32 Difference Query weights.
- Checkpoint content SHA-256: `0de199431afed6610f03b55c6d08aec9cdc5a177d1fd97441e5afa25a6b64111`.
- Difference Query sidecar: version 1, enabled, 32 queries, hidden size 2048, required attention backend `sdpa`.
- Difference Query sidecar SHA-256: `9ee4bb2bce02ceeef95a029b1f6055644e7454d4e0dc214e303d245bc3e7553c`.
- Difference Query weights SHA-256: `13f495173f62c796e9fb5cd579374381d9f9b2aa53ebfd23000027b6a9e6baa7`.
- Action Expert weights SHA-256: `9ad4ec008ea114d679d74f23ad4ed58685a230e2713b68ea859160e906d2f4bf`.
- Checkpoint metadata records source action horizon 50 and target/evaluation action horizon 10; no checkpoint file is modified or migrated.
- Runtime contract: requested `auto`, required/resolved `sdpa`, contract version 1, canonical SHA-256 `72a705ce800b17086facdee0c906d21667dacc787dc0eff0542b638416bad9f8`.

## Scale and scheduling

- Hardware: four NVIDIA A800-SXM4-80GB GPUs, devices 0, 1, 2, and 3.
- Services: one SDPA server per GPU; ports 8480, 8481, 8482, and 8483.
- Fixed batch configuration: `batch_size=45`, `num_env_workers=45` per service, 180 concurrent LIBERO workers total.
- Rationale: B45 is the largest previously validated safe formal configuration. B50 was rejected after a task-transition peak reached the 480 GiB container memory limit; artificial VRAM allocation is not used.
- Shard mapping and complete commands are recorded in `supervisor_plan.json`.
- Permanent supervisor: `scripts/run_libero_shard_supervisor.py`; it owns launcher/client/server process trees, fails fast, preserves published episode JSON, and only summarizes complete coverage.

## Preflight validation

- Checkpoint artifacts present and contract resolution passed for all required DQ/Action Expert files.
- Client `pip check` passed; strict client identity, LIBERO origin/commit, OSMesa/native renderer and renderer lock passed.
- Server identity passed on an A800 with CUDA 12.4 and reported the exact DQ32/SDPA contract above.
- The server environment's existing `pip check` reports lerobot metadata dependency mismatches (opencv-python-headless absent and version constraints for accelerate, av, huggingface-hub and setuptools); no environment mutation was performed. Functional server identity and the locked runtime dependency checks passed.

## Launch command

The exact detached launch command and every shard launcher command are stored in the run directory `supervisor_plan.json`.

## Completion record

- Actual start: `2026-09-12 18:18:04 Asia/Shanghai` (supervisor attempt).
- Actual end: `2026-09-12 19:20:39 Asia/Shanghai` (combined summary and supervisor result publication).
- Supervisor elapsed: `3754.732 s` (`1 h 2 min 34.7 s`); final status `complete`.
- Per-shard episode counts: GPU0 `400/400`, GPU1 `600/600`, GPU2 `400/400`, GPU3 `600/600`; combined `2000/2000`, with no missing episode keys.
- Overall result: `1968/2000` successful, success rate `98.40%`, mean episode duration `112.068 s`.
- Suite results: `libero_10` `482/500` (`96.40%`); `libero_goal` `493/500` (`98.60%`); `libero_object` `499/500` (`99.80%`); `libero_spatial` `494/500` (`98.80%`).
- Final fingerprint: `5c278668d8131fe1173b33a25e9b7becec2992377bf704c2854bf258ed12a73a`.
- Actual batch histogram, request timing, and forward counts are recorded in `summary/summary.json`; no episode key was missing.
- Peak sampled GPU memory: GPU0 `9487 MiB`, GPU1 `8999 MiB`, GPU2 `8387 MiB`, GPU3 `8299 MiB` (each card has 81920 MiB total); sampled utilization peaked at 98%, 97%, 97%, and 96% respectively.
- No NaN, CUDA OOM, launcher failure, interruption, or resume was observed; no configuration deviation was recorded.
- After cleanup, all four GPUs returned to 2 MiB used. Authoritative artifacts are `supervisor_result.json`, `combined_manifest.json`, `summary/summary.json`, `summary/summary.csv`, and `summary/summary.md` in the evaluation output directory.
