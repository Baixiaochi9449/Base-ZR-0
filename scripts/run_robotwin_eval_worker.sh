#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 4 ]]; then
    echo "Usage: bash $0 WORKER_DIR GPU PORT CHECKPOINT" >&2
    exit 2
fi
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKER_DIR=$(cd "$1" && pwd)
export CUDA_VISIBLE_DEVICES=$2
export ZR0_SERVER_PORT=$3
export ZR0_ROBOTWIN_CKPT=$4
export ZR0_OBSERVATION_DEBUG_ROOT="$WORKER_DIR/observations"
if [[ -e "$WORKER_DIR/server.log" || -e "$WORKER_DIR/logs" ]]; then
    echo "Worker already started; prepare a fresh output directory." >&2
    exit 2
fi
cd "$ROOT_DIR"
PYTHON_BIN=${ZR0_SERVER_PYTHON:-/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python}
"$PYTHON_BIN" -c 'import socket, sys; s=socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1]))); s.close()' "$ZR0_SERVER_PORT"
bash scripts/run_robotwin_legacy_server.sh serve > "$WORKER_DIR/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
    status=$?
    trap - EXIT
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    printf '%s\n' "$status" > "$WORKER_DIR/exit_code.txt"
    printf '\nWorker finished (UTC): %s; exit: %s\n' "$(date -u +%FT%TZ)" "$status" >> "$WORKER_DIR/experiment.md"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ready=0
for ((attempt=0; attempt<150; attempt++)); do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "Model server exited during startup." >&2; exit 1; }
    if curl -fsS --max-time 2 "http://127.0.0.1:$ZR0_SERVER_PORT/healthz" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 2
done
[[ "$ready" == 1 ]] || { echo "Model server readiness timed out." >&2; exit 1; }
source "${ROBOTWIN_CONDA_SH:-/opt/data/private/lq/miniconda3/etc/profile.d/conda.sh}"
conda activate RoboTwin
bash "$WORKER_DIR/run_all.sh" > "$WORKER_DIR/batch.log" 2>&1
