# Video decoding and GPU resource preflight

- Created: 2026-09-07, Asia/Shanghai. Owner: lq; execution: Codex.
- Purpose: verify and minimally repair CPU PyAV resource ownership and sequential GPU availability checks before three-stage validation.
- Code baseline: 29796d94059153e332449563657b1155acce7f56 plus existing user changes, including RoboTwin work. Those changes are preserved. No staging or commit is authorized.
- Output: a new isolated directory under `outputs/resource_preflight_20260907`; existing outputs are read-only.
- This task runs CPU decoding, tests and two bounded CUDA resource probes after the four-GPU gate passes. Each future training stage is limited to 100 successful optimizer updates. No formal AR/Joint branch or 20k/5k/150k training is authorized.
- No model/checkpoint/optimizer/loss is used by CPU decoder stress or read-only GPU polling; training seed, batch, optimizer, W&B run and rollout configuration are not applicable to these diagnostics. No validation training has started. Future training must have its own complete experiment configuration and W&B enabled.

## Initial evidence

- Production: `Stage05MixedPretrainingDataset._decode_video` -> local LeRobot `decode_video_frames` -> torchvision PyAV `VideoReader`. Stage05 uses 4 workers per rank, 4 ranks (up to 16 concurrent worker decoders); each worker reads main/wrist sequentially, prefetch_factor=3, persistent_workers=False.
- Installed PyAV 12.3.0, torchvision 0.21.0+cu124; loaded FFmpeg reports 6.1.1, with libavcodec 60.31.102 and libavformat 60.16.100; loaded libdav1d reports 1.4.1. Versions were read from the wheel libraries, not the system ffmpeg executable.
- Real partial-DROID AV1 `observation.images.exterior_1_left/chunk-000/file-149.mp4`, 320x180: three reads using the old VideoReader, even with `num_threads=1`, retained 224 additional threads per read (64 -> 288 -> 512 -> 736). RSS 421 -> 608 MiB, descriptors returned to 4 after each container.close. Codec remained open after container.close. One diagnostic-only gc.collect released the threads and RSS fell to 442 MiB. This establishes delayed decoder release in this environment, not the sole cause of the historical failure.
- Historical failure is at full-DROID global_index=2041670, main-camera codec open, av.error.MemoryError (Errno 12); it is not CUDA OOM. The original sample/video mapping is recorded below.
- Visible cgroup v1 memory limit 480 GiB, current use about 153 GiB; memory.failcnt=18 and oom_kill=11 are historical cumulative counters, without a causal timestamp. Host available memory about 1.59 TiB. Cgroup pids.max=max; process RLIMIT_NPROC/AS/RSS unlimited, NOFILE=1048576. These snapshots do not establish the historical limit state.
- Initially four A800 GPUs had active compute PIDs 1299899/1299901/1299903/1299905 and 45-78 GiB used. This was an actual occupancy blocker, not only residual utilization. No process was stopped; the later release is recorded below.
- The previous H10 gate already queried compute PIDs and its launcher already waited for the child. Its confirmed defects were single-snapshot utilization rejection and physical-index selection that was not bound to CUDA UUIDs. The standalone audit was process-only; the LIBERO sequencer used an unbounded memory-only wait. These distinctions were verified from the actual implementations.

## Planned checks

- Fresh-process before/after real AV1 repeated reads, video switching, early return, seek/decode/conversion errors, worker reads and worker exit. Record RSS, threads, descriptors, host/cgroup state and throughput. Compare frame tensors and timestamp ordering exactly; exercise the production adapter and preserve action/label alignment.
- GPU simulations: residual utilization, active other/self processes, unreleased memory, failed/malformed query, CUDA device identities, MIG/MPS ambiguity and timeout. Read-only real polling must block existing workloads.
- Re-run relevant Stage1 compatibility, sampler resume, empty-supervision and checkpoint tests without changing their expectations to bypass checks.

## Results

- Decoder ownership is confirmed to be defective in the old implementation. The eight-read isolated baseline peaked at 1,807 threads and 1,070 MiB RSS, with only 4 file descriptors after container close. Automatic GC later reduced the process to 15 threads; therefore this is proven delayed native-resource release and a concurrency peak, not proof of an indefinitely monotonic leak or of the sole historical OOM cause.
- Historical sample: full DROID episode 7035, frame_index 52, source timestamp 3.4666666984558105 s, main video `videos/observation.images.exterior_1_left/chunk-000/file-022.mp4`, video timestamp 560.5333333651224 s. It decoded successfully in both versions, along with three other AV1 videos. This does not certify every byte of the historical video or exclude historical system pressure.
- Fixed implementation: 256 same-video, 256 multi-video, 256 real timestamp-validation failures, and 256 injected failures after a real AV1 frame. Threads stayed 15, descriptors stayed 4. After excluding each phase's first quarter as warmup, RSS ranges were respectively 886.816-886.820, 886.820-886.820, 887.172-887.172 and 887.188-887.191 MiB. No forced GC, retries or sample replacement was used. Four recent exception tracebacks were deliberately retained. Cgroup memory failure/OOM-kill counters did not increase.
- Eight before/after frame tensors match exactly, including reordered and duplicate timestamp requests. Four complete production samples from DROID and RH20T also match exactly for pixels, tokens, frame/episode identity, state/actions, action masks, text and metadata labels.
- Production loader: existing natural mixture/grouped sampler, seed42, actual DROID and RH20T adapters, batch2, workers4, prefetch3, persistent_workers=False, 128 batches in each of epochs0/1 (512 consumed samples). No model is loaded. All eight worker instances exited after owned iterator teardown. Warm worker threads stayed15; RSS within each worker varied by at most5.3 MiB (approximately2.2-2.9 GiB total per worker, including bounded Arrow/source caches). Descriptors fluctuated54-179 with prefetch/tensor IPC; they did not grow monotonically. This tests two video-bearing sources on CPU, not four distributed GPU ranks.
- Short-read timing for the same eight requests: baseline total about0.62 s, fixed about0.24 s; the old automatic thread setup dominates these short seeks. Fixed production reading consumed512 samples in17.93 s, including worker startup/shutdown and diagnostics. This is not a full-training throughput claim or a sequential full-video benchmark.
- New decoder/gate plus existing H10 runner tests initially passed 59 cases. Broader regression initially passed 260 of 282 cases: 13 failed through installed DeepSpeed/Triton import with CUDA entirely hidden; 9 historical Stage05 launcher checks rejected a stale token-audit implementation identity. The follow-up `test_aux_review4.py` passed all 30 pytest cases, including those 13, with driver visibility restored and CPU test tensors. However, the additional assertion that CUDA remain uninitialized failed: DeepSpeed initialized CUDA despite `ACCELERATE_USE_CPU=true`. This diagnostic command exited 1 and is not claimed to be a successful CPU-only run. Its process exited; no production training was started.
- The registry is the only mismatching file among the six certificate dependencies. Its current SHA256 is `f078ae995d6f268018806a6cf72db280a736e42edf5474417d29e9e1181c9ce4`; HEAD and the certificate both record `57dd6d3e5f5595e436f30a9b01597fa3c56a1c53b988a0fb885dc4b3289c0cef`. The difference is the user's separately staged RoboTwin dataset entry. The other five dependency files match exactly. No certificate, trusted hash or source registry was rewritten to pass these nine checks.
- The first real default 60-second GPU gate returned NO-GO with occupied-device evidence. The later final gate waited 57.02 seconds and returned GO at 2026-09-07 23:00:00 Asia/Shanghai after three consecutive idle samples. Each GPU then had 2 MiB used, 81,226 MiB free, utilization 0 and no compute process. Full observations are in `gpu_gate_initial.samples.jsonl` and `gpu_gate_final.samples.jsonl`.
- Actual sequential resource probes ran 2026-09-07 23:04:25-23:04:53 Asia/Shanghai. Two fresh processes (3567253, 3567383) each performed 16 float32 64x64 matrix products on each of the four UUID-pinned A800s, then exited 0. The shared gate waited 11.58 and 11.32 seconds for exit/release. Fifteen samples include six that explicitly observed a live owned child; all six remained WAIT. Three gates passed (initial plus two releases), with three consecutive idle samples each. No model, data, optimizer or W&B training run was involved; effective training updates were zero. This validates real process exit sequencing; residual-utilization timing is separately controlled by simulation.
- No three-stage training, formal AR/Joint, downstream fine-tuning, rollout, environment installation, staging or commit was performed by this task. Historical products are unchanged. Diagnostics terminate after their bounded work and cannot enter a formal training branch.
- Final targeted regression: 82 passed in 65.01 seconds across decoder resources, GPU gates, H10 runner and LIBERO launcher tests (`final_targeted.xml`). Bash syntax checks for both LIBERO scripts and existing Stage05/structured-slot launchers passed; `git diff --check` passed. A real AV1 log confirms `codec=libdav1d requested_threads=1 actual_thread_count=1 thread_type=SLICE`, output shape `(1, 3, 180, 320)`. The nine unrelated stale-audit failures and the failed no-CUDA-initialization assertion remain explicitly recorded, not relabeled as passes.

## Three-stage validation status

- Stage 1, Stage 2 and Stage 3 effective training updates: 0 / 0 / 0. Authorized future limit: 100 / 100 / 100; no reduction in batch size has been made.
- GPU occupancy is no longer the recorded blocker after the final gate and probes, but every future launch must recheck availability.
- The old Stage05 launcher is blocked by the registry/certificate identity mismatch above. Stage 1 compatibility, sampler restore, missing-supervision behavior and checkpoint completeness checks remain enabled.
- A concrete three-stage validation configuration has not been supplied or selected. The historical H10 runner implements two-stage AR/Joint with automatic formal continuation; `run_structured_slot_stage.sh` implements horizon 32 Stage 2/3, defaults to one epoch and does not expose a 100-update cap. Neither can be launched unchanged for the requested validation. The initial checkpoint, batch/GAS, Slot/Flow selection and weights need to be bound in a new bounded validation configuration before launch. A clarification was requested; no historical formal schedule is inferred from an instruction to continue.

## Implementation and configuration

- `LEROBOT_PYAV_THREADS` defaults to1; explicit `decode_video_frames(..., num_threads=N)` takes precedence. Zero is an explicit opt-in to legacy FFmpeg automatic threading; it does not restore delayed cleanup. Settings affect PyAV only. Stream-discovery options are supplied at av.open, and codec.thread_count is set before the first decode opens the codec. INFO logs report PID, codec, requested/actual codec thread count and thread type.
- Public cleanup order: close the owned decode generator, release the current frame reference, close the owned open codec only when its installed API exposes close, close the container, then drop references. Nested finally blocks still close the container if generator/codec cleanup raises. Errors propagate. No private torchvision `_c` or PyAV free operation is used. Other backends retain their prior decode/selection behavior.
- Existing seek uses rounded stream-time-base offset, backward keyframe seek, then selects the nearest loaded timestamp with the original strict tolerance, ordering and float32 RGB conversion. No image size, augmentation, timestamp, action or label semantics change. No active decoder cache is added; existing bounded Parquet/episode/canonical caches and worker count stay unchanged.
- `GPUResourcePolicy` uses environment prefix `ZR0_GPU_GATE_`: `POLL_SECONDS=2`, `CONSECUTIVE_SAMPLES=3`, `TIMEOUT_SECONDS=60`, `MIN_FREE_MIB=71680`, `MAX_USED_MIB=1024`, `MAX_UTILIZATION=0`, `QUERY_TIMEOUT_SECONDS=5`. The70 GiB free threshold is the previous H10 gate's reservation. The1 GiB maximum-used allowance permits idle driver/display bookkeeping; it never excuses a compute PID. Thresholds and identities are written in every sample.
- The H10 runner's `ZR0_CUDA_VISIBLE_DEVICES` selection is resolved by CUDA driver enumeration in a fresh process and cross-checked against nvidia-smi UUID/PCI identity. Launches are pinned to those UUIDs. Driver enumeration allocates no model/tensors. Failed/malformed process queries, missing devices, unsupported MIG and explicit MPS environments fail closed. Own launchers and process groups must exit. GPU sampling failure/timeout retains the prior sample and reasons. No other process is terminated.
- H10 launch/probe/resume/preflight and the standalone audit share this gate; the separate LIBERO sequencer's memory-only unlimited GPU wait was also replaced, with a fresh gate before smoke/resume/formal entry. The LIBERO launcher now uses the gate's `ZR0_CUDA_VISIBLE_DEVICES` UUID selection for both preflight and training; its default remains 0,1,2,3. Historical formal branches remain present but are never invoked by these diagnostic scripts. This task has no auto-continuation into formal training. A gate remains a point-in-time check, not an exclusive GPU reservation; later launch errors retain existing logs.

## Reproduction

From the repository root, use `/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python` with `PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:lerobot OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=` for CPU video checks. All output directories must be new.

```text
scripts/audit_video_decode_resources.py --mode baseline --output outputs/resource_preflight_20260907/decode_baseline
scripts/audit_video_decode_resources.py --mode fixed --iterations 256 --reference outputs/resource_preflight_20260907/decode_baseline --output outputs/resource_preflight_20260907/decode_fixed_with_faults
scripts/audit_video_decode_resources.py --mode production-baseline --output outputs/resource_preflight_20260907/production_baseline
scripts/audit_video_decode_resources.py --mode production --iterations 128 --workers 4 --reference outputs/resource_preflight_20260907/production_baseline --output outputs/resource_preflight_20260907/production_fixed
```

Run GPU polling with visible devices explicitly selected, without launching training:

```text
scripts/audit_stage05_gpu_gate.py --visible-devices 0,1,2,3 --output outputs/resource_preflight_20260907/gpu_gate_final.json
scripts/audit_gpu_probe_exit.py --visible-devices 0,1,2,3 --output outputs/resource_preflight_20260907/gpu_probe_exit
```

Each video directory includes `experiment.md`, `results.json` and per-iteration `resources.jsonl`; golden tensors remain ignored runtime artifacts. GPU output has a separate append-only samples JSONL. The probe directory has its own command, experiment record, child logs and results. Unit/regression reports are `regression.xml`, `aux_cpu_driver_visible.xml` and `final_targeted.xml` in the same root.

Final targeted test command (CPU-hidden environment as above):

```text
/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -m pytest -q tests/test_video_decode_resources.py tests/test_gpu_resource_gate.py tests/test_stage05_experiment_launch.py tests/test_libero_wo_ecot_pt_launcher.py --junitxml=outputs/resource_preflight_20260907/final_targeted.xml
```

API reference: installed PyAV12.3.0 type stubs and torchvision0.21 source were inspected, together with [PyAV12.3 input container](https://github.com/PyAV-Org/PyAV/blob/v12.3.0/av/container/input.pyx) and [codec context](https://github.com/PyAV-Org/PyAV/blob/v12.3.0/av/codec/context.pyx). The former closes input I/O; the latter exposes public codec close in this installed version. No third-party algorithm was copied.
