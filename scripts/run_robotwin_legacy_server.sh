#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${ZR0_SERVER_PYTHON:-/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python}
CHECKPOINT=${ZR0_ROBOTWIN_CKPT:-/opt/data/private/lq/models/ZR-0-RoboTwin2.0-Aloha-AgileX}
PORT=${ZR0_SERVER_PORT:-8022}
MODE=${1:-print}

if [[ $# -gt 1 || ( "$MODE" != print && "$MODE" != serve ) ]]; then
    echo "Usage: bash $0 [print|serve]" >&2
    exit 2
fi
if [[ ! "$PORT" =~ ^[1-9][0-9]{0,4}$ ]] || (( PORT > 65535 )); then
    echo "ZR0_SERVER_PORT must be an integer from 1 to 65535" >&2
    exit 2
fi

export PYTHONNOUSERSITE=1
cd "$ROOT_DIR"
command=(
    "$PYTHON_BIN" -u "$ROOT_DIR/server.py"
    --dataset_entry demo_data.robotwin2.0-aloha-agilex
    --ckpt_dir "$CHECKPOINT"
    --inference_mode direct_action
    --window_size 1
    --num_denoised_steps 5
    --max_pad_state_and_action_length 64
    --port "$PORT"
    --allow_legacy_checkpoint_without_manifest
)

echo "RoboTwin diagnostic server: local statistics are not verified against official checkpoint training." >&2
echo "Legacy missing-manifest compatibility is explicitly enabled; existing manifests remain validated." >&2
if [[ "$MODE" == print ]]; then
    printf 'cd %q && PYTHONNOUSERSITE=1 ' "$ROOT_DIR"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the GPU selected for this server}"
exec "${command[@]}"
