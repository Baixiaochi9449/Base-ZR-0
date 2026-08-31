# LIBERO Evaluation Runbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `simple_scripts/eval_libero.md` with a copy-paste runbook for running all four LIBERO suites with the existing model, source checkout, and Conda environments on the current server.

**Architecture:** Keep the workflow in one Markdown file and split it into preflight, launch, monitoring, summary, and cleanup phases. Start one policy server per GPU and one CPU simulator client per suite, keep suite/GPU/port mappings explicit, and write each run into a timestamped directory.

**Tech Stack:** Bash, Markdown, Python 3.10 server environment, Python 3.8 LIBERO client environment, WebSocket policy server/client, headless MuJoCo with OSMesa.

## Global Constraints

- Use only the existing repository at `/opt/data/private/lq/ZR-0`.
- Use only the existing model at `/opt/data/private/lq/models/ZR-0-libero`.
- Use the existing `zr0-eval` and `zr0-libero-eval` environments; do not document installation or environment creation.
- Preserve repository evaluation defaults; do not add overrides for resize size, replan steps, denoising steps, seeds, trial count, or suite horizons.
- Map `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10` to GPU/port pairs `0/8100`, `1/8001`, `2/8002`, and `3/8103`.
- Create a new timestamped result directory for every run.
- Never terminate unrelated processes; clean up only PIDs launched by the documented commands.

---

### Task 1: Replace the Command Record with an Operational Runbook

**Files:**
- Modify: `simple_scripts/eval_libero.md`
- Reference: `server.py`
- Reference: `evaluation/libero_eval/run_libero_eval.py`
- Reference: `result/eval/ZR-0-LIBERO_official_seed7_50trials_20260830_195202/run_manifest.yaml`

**Interfaces:**
- Consumes: existing server/client Python executables, model directory, LIBERO config, and verified summary parser.
- Produces: a standalone Markdown runbook whose Bash blocks can be executed in order in one SSH shell.

- [ ] **Step 1: Replace the file with the approved runbook structure**

Write these sections in order:

1. Title, scope, and the warning that all launch/wait/cleanup blocks must run in the same SSH shell.
2. Existing resource paths and core script descriptions.
3. Default evaluation parameter table sourced from the repository scripts and verified run manifest.
4. SSH connection and timestamped result-directory initialization.
5. Required-path, GPU, and port preflight checks.
6. Four policy-server launches with PID files and readiness checks.
7. Headless LIBERO exports and four client launches with PID files.
8. Monitoring and client completion checks.
9. Summary generation with the verified `summarize_eval.py` parser.
10. Server cleanup and result-directory layout.
11. Previous verified run results and short troubleshooting notes.

Use this exact resource setup in the runbook:

```bash
cd /opt/data/private/lq/ZR-0

WORKSPACE=/opt/data/private/lq/ZR-0
MODEL_DIR=/opt/data/private/lq/models/ZR-0-libero
SERVER_PY=/opt/data/private/lq/.conda/envs/zr0-eval/bin/python
CLIENT_PY=/opt/data/private/lq/.conda/envs/zr0-libero-eval/bin/python
VERIFIED_RUN=/opt/data/private/lq/ZR-0/result/eval/ZR-0-LIBERO_official_seed7_50trials_20260830_195202
RUN_ID="ZR-0-LIBERO_official_seed7_50trials_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$WORKSPACE/result/eval/$RUN_ID"

mkdir -p "$RUN_DIR"/{logs,videos,env,artifacts,summary,pids}
cp -a "$VERIFIED_RUN/env/libero-config" "$RUN_DIR/env/libero-config"
cp "$VERIFIED_RUN/artifacts/summarize_eval.py" "$RUN_DIR/artifacts/"
printf 'Result directory: %s\n' "$RUN_DIR"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
```

Use a loop containing `test -e` for required paths, `nvidia-smi` for GPU visibility,
and the following non-destructive port check:

```bash
for port in 8100 8001 8002 8103; do
  if ss -ltnH "( sport = :$port )" | read -r _; then
    echo "ERROR: port $port is already in use" >&2
    exit 1
  fi
done
```

Launch servers through this function and these mappings:

```bash
launch_server() {
  local suite=$1 gpu=$2 port=$3
  CUDA_VISIBLE_DEVICES="$gpu" "$SERVER_PY" -u server.py \
    --dataset_entry demo_data.libero_v21 \
    --ckpt_dir "$MODEL_DIR" \
    --inference_mode direct_action \
    --port "$port" \
    > "$RUN_DIR/logs/server_${suite}.log" 2>&1 &
  echo $! > "$RUN_DIR/pids/server_${suite}.pid"
}

launch_server libero_spatial 0 8100
launch_server libero_object  1 8001
launch_server libero_goal    2 8002
launch_server libero_10      3 8103
```

Define cleanup immediately after launch so only recorded server PIDs are stopped:

```bash
cleanup_servers() {
  local pid_file pid
  for pid_file in "$RUN_DIR"/pids/server_*.pid; do
    [[ -f "$pid_file" ]] || continue
    pid=$(<"$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
    fi
  done
}
trap cleanup_servers EXIT INT TERM
```

Use this bounded readiness loop, which checks both the saved PID and listening
port and prints the last 50 server-log lines on failure:

```bash
wait_for_server() {
  local suite=$1 port=$2 pid attempt
  pid=$(<"$RUN_DIR/pids/server_${suite}.pid")
  for ((attempt = 1; attempt <= 120; attempt++)); do
    if ss -ltnH "( sport = :$port )" | read -r _; then
      echo "$suite server is ready on port $port"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: $suite server exited before becoming ready" >&2
      tail -n 50 "$RUN_DIR/logs/server_${suite}.log" >&2
      return 1
    fi
    sleep 5
  done
  echo "ERROR: timed out waiting for $suite server on port $port" >&2
  tail -n 50 "$RUN_DIR/logs/server_${suite}.log" >&2
  return 1
}

wait_for_server libero_spatial 8100
wait_for_server libero_object  8001
wait_for_server libero_goal    8002
wait_for_server libero_10      8103
```

Set the client environment exactly as follows:

```bash
export LIBERO_CONFIG_PATH="$RUN_DIR/env/libero-config"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=/opt/data/private/lq/.conda/envs/zr0-libero-eval/usr/lib/x86_64-linux-gnu:/opt/data/private/lq/.conda/envs/zr0-libero-eval/lib
```

Launch clients through this function and the same suite/port mappings:

```bash
launch_client() {
  local suite=$1 port=$2
  "$CLIENT_PY" -u -m evaluation.libero_eval.run_libero_eval \
    --args.task-suite-name "$suite" \
    --args.port "$port" \
    --args.video-out-path "$RUN_DIR/videos/$suite" \
    > "$RUN_DIR/logs/client_${suite}.attempt1.log" 2>&1 &
  echo $! > "$RUN_DIR/pids/client_${suite}.pid"
}

launch_client libero_spatial 8100
launch_client libero_object  8001
launch_client libero_goal    8002
launch_client libero_10      8103
```

Document monitoring from a second SSH shell using the concrete `RUN_DIR` printed
at initialization:

```bash
tail -F "$RUN_DIR"/logs/client_*.attempt1.log
```

In the launch shell, wait only on the four recorded client PIDs and fail if any
client exits nonzero:

```bash
client_failed=0
for suite in libero_spatial libero_object libero_goal libero_10; do
  pid=$(<"$RUN_DIR/pids/client_${suite}.pid")
  if wait "$pid"; then
    echo "$suite completed successfully"
  else
    echo "ERROR: $suite client exited nonzero" >&2
    client_failed=1
  fi
done

if ((client_failed != 0)); then
  echo "At least one suite failed; inspect $RUN_DIR/logs" >&2
  exit 1
fi
```

After successful completion, generate summaries with:

```bash
"$CLIENT_PY" "$RUN_DIR/artifacts/summarize_eval.py"
cat "$RUN_DIR/summary/summary.md"
```

Finish with `cleanup_servers` followed by `trap - EXIT`, and explain that the
result directory contains `logs/`, `videos/`, `summary/`, `pids/`, `env/`, and
`artifacts/`.

- [ ] **Step 2: Check Markdown and shell syntax**

Run:

```bash
git diff --check -- simple_scripts/eval_libero.md
awk '/^```bash$/ {inside=1; next} /^```$/ && inside {inside=0; print ""; next} inside' \
  simple_scripts/eval_libero.md > /tmp/eval_libero_runbook_blocks.sh
bash -n /tmp/eval_libero_runbook_blocks.sh
```

Expected: both commands exit with status 0 and print no syntax errors.

- [ ] **Step 3: Verify paths and client option names**

Run:

```bash
test -f server.py
test -f evaluation/libero_eval/run_libero_eval.py
test -d /opt/data/private/lq/models/ZR-0-libero
test -x /opt/data/private/lq/.conda/envs/zr0-eval/bin/python
test -x /opt/data/private/lq/.conda/envs/zr0-libero-eval/bin/python
test -f result/eval/ZR-0-LIBERO_official_seed7_50trials_20260830_195202/env/libero-config/config.yaml
LIBERO_CONFIG_PATH=result/eval/ZR-0-LIBERO_official_seed7_50trials_20260830_195202/env/libero-config \
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
LD_LIBRARY_PATH=/opt/data/private/lq/.conda/envs/zr0-libero-eval/usr/lib/x86_64-linux-gnu:/opt/data/private/lq/.conda/envs/zr0-libero-eval/lib \
/opt/data/private/lq/.conda/envs/zr0-libero-eval/bin/python \
  -m evaluation.libero_eval.run_libero_eval --help > /tmp/eval_libero_help.txt
rg -- '--args.task-suite-name|--args.port|--args.video-out-path' /tmp/eval_libero_help.txt
```

Expected: all path checks pass and all three documented client flags appear.

- [ ] **Step 4: Verify mappings, defaults, and scope**

Run:

```bash
rg -n 'libero_spatial.*0.*8100|libero_object.*1.*8001|libero_goal.*2.*8002|libero_10.*3.*8103' \
  simple_scripts/eval_libero.md
rg -n '448|256|replan|denois|50|seed' simple_scripts/eval_libero.md
if rg -n 'conda create|pip install|git clone|modelscope download|huggingface-cli download' \
  simple_scripts/eval_libero.md; then
  echo 'ERROR: installation instructions are out of scope' >&2
  exit 1
fi
```

Expected: all four mappings and documented defaults are present, while no
installation or download command is present.

- [ ] **Step 5: Review the final diff and commit only the runbook**

Run:

```bash
git diff -- simple_scripts/eval_libero.md
git add simple_scripts/eval_libero.md
git commit --only simple_scripts/eval_libero.md -m "docs: add LIBERO evaluation runbook"
```

Expected: the diff contains only the approved runbook content and the commit
does not include unrelated working-tree changes.
