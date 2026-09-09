# RoboTwin 50 tasks, two settings, 20 rollouts

- Created: 2026-09-07, Asia/Shanghai. Owner: lq; prepared by Codex.
- Purpose: evaluate the downloaded ZR-0 RoboTwin checkpoint over the full
  task/setting matrix with the user's requested reduced rollout count.
- Status: four-GPU relaunch authorized; see Four-GPU Execution below. Earlier
  single-GPU results are retained as interrupted diagnostics, excluded from this run.
- Baseline: `29796d94059153e332449563657b1155acce7f56`. The generated `plan.json`
  records the actual commit, dirty Git status and source configuration hashes.
- Changes: one suite config, preparation script, focused tests and documentation;
  the existing single-task template changes from 100 to 20 episodes and uses a
  descriptive result label. Earlier metadata/server changes remain in place.
- Checkpoint/base policy: `/opt/data/private/lq/models/ZR-0-RoboTwin2.0-Aloha-AgileX`.
  This is an externally trained checkpoint; the original training run ID, split,
  optimizer and W&B URL are unavailable. No local training is performed.
- Training modules, losses, optimizer, batch size, W&B project/group/run: N/A.

## Evaluation Definition

`configs/robotwin_eval_50x2x20.json` is the suite configuration. Tasks come from
`evaluation/RoboTwin/task_config/_eval_step_limit.yml`; all 50 task modules and
instruction files must exist. Each task is expanded over `demo_clean` and
`demo_randomized`, giving 100 configurations and 2,000 policy rollouts.
Each setting has 1,000 rollouts. The repository's original example used 100 per
task/setting, which would give 10,000 rollouts for the same complete matrix.

- `n_episodes=20`: number of policy trials, including policy failures.
  The client's existing expert solvability check remains enabled. Rejected
  candidate seeds and expert demonstrations do not count toward these 20.
- `seed=0`: each group starts its candidate seed sequence at 100000, then skips
  expert failures. Accepted seed sequences can differ by task/setting. Logs
  retain the client's seed messages. The server separately seeds its RNG at 42
  on startup; its RNG stream continues across groups in sequential execution.
- `instruction_type=unseen`; use the original instruction generation path.
- `max_episode_steps`: per-task values from the official table, 400 to 1700.
  Examples: `adjust_bottle=400`, `open_microwave=1500`,
  `put_bottles_dustbin=1700`. These count policy `take_action` calls, not physics
  substeps or inference requests. Success can terminate an episode early.
- `n_action_steps=16`; checkpoint prediction/action horizon and chunk are 16.
  Execute up to 16 queued actions before requesting a new chunk; episode
  termination can truncate the last chunk. Observation window is 1.
- Host/port: `127.0.0.1:8022`. `ckpt_setting=zr0_local_stats_e20` is only a result
  label; scene selection comes from `task_config`, model selection from server.
- Clean/Random profile contents are unchanged, including domain randomization
  and `eval_video_log=true`. Their `episode_num=50` is for data collection and
  does not control the client's policy evaluation count.
- Per-group success rate is `successful_policy_rollouts / 20`, in increments
  of 5 percentage points. Report Clean and Random separately as the mean of
  their 50 task rates, equivalently total setting successes / 1000. A combined
  mean is optional; do not silently combine settings or count incomplete runs
  as complete. No benchmark scores have been produced yet.

## Data, Images And Actions

The server uses `demo_data.robotwin2.0-aloha-agilex`, metadata under
`demo_data/robotwin2.0-aloha-agilex/meta`. Quantiles were computed over all
6,075,103 frames / 27,500 episodes of the local LeRobot v3 dataset at
`/opt/data/private/lq/datasets/lerobot/robotwin_unified`. This data supplies
normalization only; evaluation uses newly simulated episodes, not replayed
training trajectories. State, language and three RGB views are policy inputs;
the policy outputs actions. No flow labels are used.

Statistics SHA256:
`1b0edfb3297a9b9d23eb3344e19a8d3891a0a46fb8811ed3c2e9efa5191f4089`.
**Local statistics were accepted for diagnostic evaluation; equivalence to the
official checkpoint's training statistics remains unverified. These results
must not be presented as a verified reproduction of the official scores.**

Image dimensions below are width x height. Simulated D435 head, left wrist and
right wrist images are 320x240; the client sends RGB without resizing.
The server converts to a tensor and back to RGB PIL, and the existing
`prepare_qwen_vl_inputs_cpu` / `process_vision_info` path resizes each view to
224x224 using PIL's default RGB bicubic interpolation. This does not preserve
the original 4:3 aspect ratio. No crop, letterbox, spatial padding or evaluation
augmentation is applied. The processor is called with `do_resize=false`,
rescales by 1/255, and uses mean/std `[0.5, 0.5, 0.5]`. Each camera remains a
separate labeled image in head/left-wrist/right-wrist order, not a stitched
mosaic. Patch size is 16; the expected per-image grid is `[1, 14, 14]`.

**Resolution difference:** local source training images are 640x480; simulation
images are 320x240 because the original evaluation profiles select D435.
The current repository's v2 training and evaluation helper uses the same
224x224 model input size. The released checkpoint has no training manifest,
so its original training preprocessing/augmentation cannot be fully verified.
This preparation does not change image processing or camera configuration.

ALOHA state/action dimensions are 14: left six joint positions plus gripper,
then right six positions plus gripper. Actions are absolute qpos targets;
grippers use the existing continuous scalar encoding. Quantile normalization
maps q01/q99 to -1/+1 (epsilon 1e-8), with the existing input safety clip at
[-15,15]. State is padded to 64; action masks retain the first 14 of 64 model
dimensions. The server extracts those 14 and applies inverse quantile scaling.
Inference uses direct action, BF16, five denoising steps and the explicit legacy
missing-manifest option from `scripts/run_robotwin_legacy_server.sh`.

Local data FPS is 30. Simulator timestep is 1/250 second, but each qpos target
uses TOPP and a variable number of physics substeps: policy actions do not have
a fixed 30 Hz or 250 Hz control rate. Wall-clock throughput is not measured.

## Commands And Outputs

Preparation runs on CPU and only writes configs, a plan, this document and a
shell script. Existing output directories are rejected to preserve prior runs.

```bash
cd /opt/data/private/lq/ZR-0
PYTHONNOUSERSITE=1 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python \
  scripts/prepare_robotwin_eval_suite.py \
  --config configs/robotwin_eval_50x2x20.json \
  --output outputs/evaluations/robotwin_50x2x20_seed0
```

Launch later, using two terminals. The GPU index 1 below is an example and must
be chosen against resource availability at launch; no GPU is reserved here.
The prepared runner requires an active RoboTwin Conda environment so its
installed CUDA/Vulkan activation hooks are applied.

```bash
# Terminal 1: model server in the existing ZR-0 environment.
cd /opt/data/private/lq/ZR-0
CUDA_VISIBLE_DEVICES=1 bash scripts/run_robotwin_legacy_server.sh serve

# Terminal 2: after server readiness, simulator in the RoboTwin environment.
source /opt/data/private/lq/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
cd /opt/data/private/lq/ZR-0
CUDA_VISIBLE_DEVICES=1 bash outputs/evaluations/robotwin_50x2x20_seed0/run_all.sh
```

The runner invokes the original `script/eval_policy_client.py --config <yaml>`
from `evaluation/RoboTwin`, sequentially: all 50 Clean tasks in alphabetical
order, then all 50 Random tasks. It logs each group under the generated output's
`logs/`, stops on the first client failure, and appends runtime start/end/exit
status to that output's `experiment.md`. Rerunning in the same output directory
is rejected once `logs/` exists; there is no automatic retry or resume.
Use a fresh `--output` for a separate full run.

Client results and videos retain their original location:
`evaluation/RoboTwin/eval_result/<task>/ZR0/<task_config>/zr0_local_stats_e20/<timestamp>/`.
Each directory contains `_result.txt` and episode videos. Generated configs,
logs and videos are ignored by Git; the suite config/script/tests/document are
tracked. The checkpoint recorded in `plan.json` is the intended server model;
the existing client does not query/verify the connected server's checkpoint.

The original client has no full-suite scheduler. This generated runner is
serial; parallel evaluation is not enabled here. A future parallel scheduler
would need independently assigned simulator workers, model servers/ports,
resource limits, seed/RNG policy and result isolation. At preparation time,
model loading, client/server integration, rollout time and success rates were
untested; subsequent runtime checks are recorded under Actual Launch.

## Validation

All 10 focused tests passed (0.44 seconds) in the existing ZR-0 environment.
The actual local preparation generated all 100 YAMLs: 50 tasks, both settings,
20 trials per group, 1,000 per setting and 2,000 in total. Every YAML was checked
against the plan and the official task limit; all eight recorded source hashes
matched. Generated Bash syntax passed; no runtime `logs/` directory existed
at preparation time.

Focused matrix and command-routing tests cover task/profile expansion, per-task
step limits, invalid counts/horizons, task-count drift, duplicate settings,
generated YAML/Bash, output preservation, required environment/GPU selection,
quoted paths, sequential routing, first-failure exit status and runtime logging.
Unit fixtures do not require the downloaded checkpoint or simulation assets.

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -m pytest -q \
  tests/test_robotwin_eval_suite.py
bash -n outputs/evaluations/robotwin_50x2x20_seed0/run_all.sh
```

A CPU check using the actual checkpoint processor and production
`prepare_qwen_vl_inputs_cpu` accepted three synthetic 320x240 RGB observations:
`image_grid_thw=[[1,14,14],[1,14,14],[1,14,14]]`, corresponding to three 224x224
images. `pixel_values.shape=[588,1536]`; zero-valued RGB mapped to -1 with the
recorded 0.5 mean/std. `torch.cuda.is_initialized()` remained false.
No model weights, GPU inference, training, expert rollout or policy rollout
are part of this preparation task.

## Actual Launch

The user explicitly requested the full configured evaluation on 2026-09-07.
Preflight at 22:34 +08:00: all four A800-SXM4-80GB GPUs were idle; GPU 0 is
selected for both model inference and the single simulator worker. Port 8022
was free. All 100 generated configs match the plan (2,000 policy trials), and
all eight source hashes remain unchanged. Baseline commit is still
`29796d94059153e332449563657b1155acce7f56`; prior staged changes and unrelated
working-tree changes are preserved. Local unverified statistics remain selected.

Persistent tmux sessions: `zr0_rt_server_8022` and `zr0_rt_eval_e20`.
The model starts first; the batch starts only after `/healthz` responds OK.
The model process uses the existing ZR-0 Python environment; the batch activates
RoboTwin, including its existing CUDA/Vulkan environment variables and hooks.
Both use `CUDA_VISIBLE_DEVICES=0`. No task, seed, rollout count, image pipeline,
action setting, denoising count, attention backend or expert check is changed.

Commands inside the respective tmux sessions (working directory is repository
root) are:

```bash
# Model session
CUDA_VISIBLE_DEVICES=0 bash scripts/run_robotwin_legacy_server.sh serve \
  > outputs/evaluations/robotwin_50x2x20_seed0/server.log 2>&1

# Evaluation session, started after model readiness
source /opt/data/private/lq/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
CUDA_VISIBLE_DEVICES=0 bash outputs/evaluations/robotwin_50x2x20_seed0/run_all.sh \
  > outputs/evaluations/robotwin_50x2x20_seed0/batch.log 2>&1
```

The evaluation session stops the owned model tmux session when the batch exits.
The existing batch script appends actual start/end/exit status to the output
directory's `experiment.md`. Server logs, batch logs, per-group logs and videos
are runtime artifacts excluded from Git. Startup outcome and observed progress
will be recorded below after launch.

### Initial Startup Repair

The model started successfully with the original `flash_attention_2` backend
and served `/healthz`. The first batch ran from 22:38:06 to 22:38:36 +08:00,
then exited 1 before any episode because `script/test_render.py` was absent.
No policy trial or expert trial was counted. The owned model session stopped
when the batch failed. These logs are preserved under the output directory's
`launch_failures/attempt1/` before retrying the same complete matrix.

Restored the expected `test_render.py::Sapien_TEST` entry by directly calling
the existing `script/check_render.py::check_render` in a temporary directory.
This checks actual headless SAPIEN ray-traced images and physics before entry
to the original evaluator, with no task/model/config changes or skipped check.
The helper runs only on explicit invocation (including the original client
startup call); importing the compatibility module does not initialize SAPIEN.
Temporary check images are removed after the check. No benchmark image
preprocessing or evaluation seed is changed.

The restored entry passed a real GPU check: 320x240 ray-traced frames, 200
physics steps at 0.004 seconds, final object height 0.1800 m (initial 0.8000 m),
mean before/after image difference 0.020527. The model was restarted and
`/healthz` returned OK before relaunching the original batch.

Additional launch-source SHA256 values:

- `script/test_render.py`:
  `51a39a3ff27ef4b7a5c6c49126e6b84f942872926acaf60e6b68f8fdfb3bce3f`.
- `script/check_render.py`:
  `353551816b958fa9ea953e2481b89f98052f95c7618c98dcdd01ac065290142a`.

### Running Batch Verification

The full retry started at 2026-09-07 22:42:22 +08:00 (14:42:22 UTC).
At 22:45:04 +08:00, both tmux sessions were alive and the first accepted policy
trial was running in `adjust_bottle` / `demo_clean`, past step 100 of 400.
Candidate seed 100000 was rejected for an unstable object by the existing
expert screening; it did not count as a policy trial. Repeated model responses
contain 16 actions of 14 dimensions; the following observation reflects executed
actions, verifying real client/server/robot interaction. The first inference
took 1.9213 seconds and the second 0.1296 seconds. These are model forward
times, not episode durations or a whole-suite time estimate.

The initial video is being written to
`evaluation/RoboTwin/eval_result/adjust_bottle/ZR0/demo_clean/zr0_local_stats_e20/2026-09-07 22:42:31/episode0.mp4`.
No complete policy trial or task score was available at this observation time;
the remaining batch runs asynchronously. The batch writes its actual final
exit/time into the output `experiment.md` and stops at the first client error.
Inspect live progress with:

```bash
tail -f /opt/data/private/lq/ZR-0/outputs/evaluations/robotwin_50x2x20_seed0/batch.log
```

The expected missing-manifest warning and a read-only NumPy state conversion
warning were emitted; neither stopped inference. No OOM or model runtime error
was observed during startup verification. This is not a completed success-rate
report. The machine-readable startup snapshot is `launch_validation.json` in
the output directory; source changes are staged without creating a commit.

At 22:48:07 +08:00 the first policy trial (seed 100001) completed all 400 steps
without task success, and the original client recorded `0/1`. The second policy
trial was already executing (past step 38). The model health endpoint remained
OK. This confirms episode completion, score accounting and transition to the
next episode; one failure is not a completed task score or suite result.

## Four-GPU Execution

On 2026-09-07 the user authorized GPU 0, 1, 2 and 3 for the running evaluation.
The original single-GPU batch was interrupted at 22:59:53 +08:00 after four
completed failed trials and during trial five of Clean `adjust_bottle`.
The shell recorded exit 0 after interruption, but this is NOT a complete run:
no 20-trial task result exists. Original logs/videos remain in
`outputs/evaluations/robotwin_50x2x20_seed0` and its original result label.
Those partial trials are excluded from the new complete matrix, which restarts
from the beginning because the original client has no episode resume contract.

New output: `outputs/evaluations/robotwin_50x2x20_seed0_4gpu`.
Result label: `zr0_local_stats_e20_p4`, isolating results from the partial run.
Use the same checkpoint, stats, 50 tasks, two profiles, 20 trials, environment
seed 0, unseen language, per-task limits, 16-step chunk/execution, BF16 and five
denoising steps. Images, normalization and simulator physics remain as documented
above. Training/W&B remain N/A. The current commit is recorded in `plan.json`.

`prepare_robotwin_eval_suite.py --gpus 0 1 2 3` explicitly enables independent
workers; omitting `--gpus` preserves the serial preparation behavior. The 100
task/setting groups are assigned round-robin, 25 groups / 500 trials to each
GPU. This changes scheduling order. Each GPU has one model server and one
sequential simulator, with ports 8022/8023/8024/8025 respectively. Servers each
start RNG seed 42; their sampling streams differ from the original one-server
serial sequence, even though environment seeds remain the same.

The original evaluator, scene settings and source task list are reused. Each
worker waits up to approximately 10 minutes for its own server health endpoint,
then invokes the existing generated `run_all.sh`. Server/client logs, episode
debug images and experiment runtime records are isolated under `workers/gpuN/`.
`ZR0_OBSERVATION_DEBUG_ROOT` changes only observation JPEG output location;
unset retains the original `temp/` directory and identical buffer tensors.
Each worker releases its model process when finished or failed and records its
exit code. Other independent workers continue if one fails. No automatic retry
or episode-level resume is introduced. Existing output directories are refused.

SAPIEN's actual default renderer device was verified under
`CUDA_VISIBLE_DEVICES=1`: logical cuda:0 mapped to physical PCI 0000:1c:00.0.
The existing renderer check now reports its device/PCI in each client's log so
the actual mapping can be checked on all workers. No renderer-selection override
or monkey patch is needed with the installed SAPIEN version.

```bash
cd /opt/data/private/lq/ZR-0
PYTHONNOUSERSITE=1 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python \
  scripts/prepare_robotwin_eval_suite.py \
  --config configs/robotwin_eval_50x2x20.json \
  --output outputs/evaluations/robotwin_50x2x20_seed0_4gpu \
  --gpus 0 1 2 3
tmux new-session -d -s zr0_rt_eval_4gpu \
  -c /opt/data/private/lq/ZR-0 \
  'bash outputs/evaluations/robotwin_50x2x20_seed0_4gpu/run_parallel.sh'
```

The generated `workers/gpuN/launch.sh` files record complete individual launch
commands. All workers load the same official model via the existing explicit
legacy compatibility launcher. Results remain diagnostic with respect to the
unverified official normalization statistics. Runtime outcome will be appended.

### Authorized Launch On 2026-09-08

The user released GPUs 0-3 and explicitly requested launch. Preflight at
13:29:50 +08:00 confirmed each A800 had 2 MiB used, zero utilization and no
compute process. Ports 8022-8025 were bindable. All 100 configs matched the
prepared plan, all recorded source hashes matched, and each worker has 25
groups / 500 policy trials. No previous four-GPU logs existed.

Current code baseline: `3bad66373877bcdfa9c1f86e134816fc8a426207`, with the
existing uncommitted validation/training changes preserved. The exact working
tree patch, status and runtime source hashes are archived in the output directory
for launch provenance. The existing four-GPU command above is used unchanged.
The parent session is `zr0_rt_eval_4gpu`. Individual model readiness, render
device placement and initial policy execution are checked after launch.

Actual four-GPU parent launch: 2026-09-08 13:30:51 +08:00. All four model
services loaded the requested checkpoint with `flash_attention_2` and their
health endpoints returned HTTP 200 / OK. All four original render checks
passed and reported the expected physical PCI devices: GPU0 0000:18:00.0,
GPU1 0000:1c:00.0, GPU2 0000:c2:00.0, GPU3 0000:c5:00.0. Their first Clean
tasks are adjust_bottle, beat_block_hammer, blocks_ranking_rgb and
blocks_ranking_size. Every group retains 20 trials; each worker owns 25 groups.
No test rollout was inserted ahead of the scheduled policy trials.

Runtime validation at 2026-09-08 13:36:47 +08:00 confirmed real policy
inference and advancing simulator actions on all four workers. The observed
first-episode step counters were 290/400, 305/400, 321/1200 and 304/1200 for
GPU0-3 respectively. Each model log contains completed VLA forward calls.
GPU memory use was approximately 12.6 GiB per device. No worker exit-code
file or fatal startup traceback was present; the tmux session remains active.
This verifies startup and execution, not completion of the 2,000-trial run.

### Remote Restart On 2026-09-08

The local tmux session `zr0_rt_eval_4gpu` was stopped at the user's request.
Stop time was approximately 14:17:21 +08:00. GPU0-3 had completed 9/10/3/3
policy trials (25 total, zero successes); no full task/profile result completed.
The evaluator has no episode-level resume or completed-job skip mechanism, so
the remote run uses a fresh output directory and repeats the full matrix.

Remote host: `lq@10.82.1.223:25408`; repository commit `3bad663`. The remote
host contains the requested checkpoint, RoboTwin assets, dataset metadata and
the `ZR-0`/`RoboTwin` Conda environments. Its GPUs are A800 80 GB devices with
about 54,053-54,055 MiB already resident per card (approximately 52.8 GiB).

The first remote attempt used the normal compiled policy. All four servers
loaded successfully, but all four clients received `ConnectionClosedError` on
their first send and exited with code 1. The worker cleanup terminates its
server after client failure; this does not establish that the server died
first. No CUDA OOM was logged, and no policy inference completed. A speculative
second attempt disabling compilation failed identically. That diagnostic
code change was removed; normal compiled inference is restored. Both failed
output directories (`..._20260908` and `..._20260908_retry`) remain preserved.

The remote tmux server inherited HTTP/HTTPS proxy variables pointing to
`127.0.0.1:7897`. The installed WebSocket library's `get_proxy(parse_uri(...))`
resolved `ws://127.0.0.1:8022` through that proxy; setting `NO_PROXY` and
`no_proxy` to `127.0.0.1,localhost,::1` made it return `None` (direct connection).
The next launch applies that standard environment override only to its own
session and retains all original model settings.

New output:
`/opt/data/private/lq/ZR-0/outputs/evaluations/robotwin_50x2x20_seed0_remote_direct_20260908`

It uses GPUs 0-3, ports 8022-8025, 25 groups / 500 policy trials per worker,
20 trials per task/profile, seed 0, unseen instructions, 16-step action
chunks, task-specific execution limits, BF16 and five denoising steps. The
launch is hosted in remote tmux session `zr0_rt_eval_remote_direct_4gpu`.
The full commands on the remote host are:

```bash
cd /opt/data/private/lq/ZR-0
/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python \
  scripts/prepare_robotwin_eval_suite.py \
  --config configs/robotwin_eval_50x2x20.json \
  --output outputs/evaluations/robotwin_50x2x20_seed0_remote_direct_20260908 \
  --gpus 0 1 2 3
tmux new-session -d -s zr0_rt_eval_remote_direct_4gpu \
  -c /opt/data/private/lq/ZR-0 \
  'env NO_PROXY=127.0.0.1,localhost,::1 no_proxy=127.0.0.1,localhost,::1 bash outputs/evaluations/robotwin_50x2x20_seed0_remote_direct_20260908/run_parallel.sh > outputs/evaluations/robotwin_50x2x20_seed0_remote_direct_20260908/launcher.log 2>&1'
```

Runtime checks and any deviations are recorded below.

The direct-connection launch started on `interactive88133` at 2026-09-08
14:58:24 +08:00. It uses the original compiled policy. Around 14:59 all four
health endpoints returned OK, four clients existed, and each model process's
environment contained its intended CUDA index and both loopback proxy overrides.
At 15:03:05 all four services had completed multiple real VLA forward calls
(recent durations approximately 0.10-0.11 s), and GPU0-3 policy steps reached
49/400, 56/400, 62/1200 and 56/1200 respectively. Both model and client
environments were checked for CUDA index and loopback proxy overrides. All
four original render checks passed, matching PCI devices 42:00.0, 46:00.0,
98:00.0 and 9b:00.0. GPU totals were about 66,978-67,031 MiB, roughly
12.7 GiB added by evaluation with 13.9 GiB remaining per card. No worker
exit or CUDA OOM was present. Source hashes and launch patch/status are
archived in the output directory. This verifies remote startup and actual
inference, not completion or benchmark accuracy; the suite remains running.
