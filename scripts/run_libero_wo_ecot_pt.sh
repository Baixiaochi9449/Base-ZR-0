#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_PATH=/opt/data/private/lq/models/Qwen3-VL-2B-Instruct
FAST_PATH="$ROOT_DIR/fast"
DATASET_PATH=/opt/data/private/lq/datasets/HuggingFaceVLA/libero
DATASET_ENTRY=libero_wo_ecot_pt
OUTPUT_DIR=${ZR0_OUTPUT_DIR:-"$ROOT_DIR/outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm"}
LOG_BASE_DIR="$ROOT_DIR/outputs/train_logs/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm"
WANDB_LOCAL_DIR="$ROOT_DIR/outputs/wandb"
WANDB_PROJECT=ZR-0-LIBERO
WANDB_GROUP=libero-wo-ecot-pt
SAVE_STEP_INTERVAL=${ZR0_SAVE_STEP_INTERVAL:-2000}
MAX_TRAIN_STEPS=${ZR0_MAX_TRAIN_STEPS:-}
RESUME_CKPT="$OUTPUT_DIR/latest-model-optimizer-lr"
WANDB_RUN_ID_FILE="$OUTPUT_DIR/.wandb-run-id"
WANDB_RUN_NAME_FILE="$OUTPUT_DIR/.wandb-run-name"

usage() {
    echo "Usage: $0 {preflight|train|resume}" >&2
}

print_command() {
    printf '%q' "$1"
    shift
    printf ' %q' "$@"
    printf '\n'
}

run_preflight() {
    python "$ROOT_DIR/scripts/preflight_libero_wo_ecot_pt.py" \
        --model-path "$MODEL_PATH" \
        --fast-path "$FAST_PATH" \
        --dataset-path "$DATASET_PATH" \
        --min-cgroup-memory-headroom-gib 120 \
        --min-gpu-free-gib 70 \
        --require-wandb
}

mode=${1:-}
case "$mode" in
    preflight)
        run_preflight
        exit 0
        ;;
    train|resume)
        ;;
    *)
        usage
        exit 2
        ;;
esac

MODEL_INPUT_PATH="$MODEL_PATH"
if [[ "$mode" == "resume" ]]; then
    MODEL_INPUT_PATH="$RESUME_CKPT"
fi

if [[ "$mode" == "resume" ]]; then
    RUN_NAME=${ZR0_RUN_NAME:-$(cat "$WANDB_RUN_NAME_FILE" 2>/dev/null || true)}
    WANDB_RUN_ID=${ZR0_WANDB_RUN_ID:-$(cat "$WANDB_RUN_ID_FILE" 2>/dev/null || true)}
    WANDB_RESUME=must
else
    RUN_NAME=${ZR0_RUN_NAME:-qwen3vl2b-libero-wo-ecot-pt-seed42-$(date +%Y%m%d-%H%M%S)}
    WANDB_RUN_ID=${ZR0_WANDB_RUN_ID:-$(python -c 'import secrets, string; alphabet = string.ascii_lowercase + string.digits; print("".join(secrets.choice(alphabet) for _ in range(8)))')}
    WANDB_RESUME=never
fi

if [[ -z "$RUN_NAME" || -z "$WANDB_RUN_ID" ]]; then
    echo "Missing W&B run metadata for mode: $mode" >&2
    exit 1
fi

LOG_DIR="$LOG_BASE_DIR/$RUN_NAME"

train_args=(
    accelerate launch
    --num_processes 4
    --config_file "$ROOT_DIR/accelerate_configs/accelerate_config.yaml"
    "$ROOT_DIR/train_vla.py"
    --vlm_name_or_path "$MODEL_INPUT_PATH"
    --FAST_tokenizer_path "$FAST_PATH"
    --per_device_train_batch_size 16
    --seed 42
    --epochs 8
    --save_ckpt_interval 4
    --save_step_interval "$SAVE_STEP_INTERVAL"
    --peak_learning_rate 2e-5
    --min_lr_rate 0.1
    --tensorboard_log_dir "$LOG_DIR"
    --output_ckpt_dir "$OUTPUT_DIR"
    --tune_vlm
    --tune_action_expert
    --loss_type action
    --action_expert_loss_weight 1.0
    --lr_scheduler cosine
    --dataset_entries "$DATASET_ENTRY"
    --window_size 1
    --action_horizon 10
    --max_pad_state_and_action_length 64
    --save_optimizer_and_lr_states
    --wandb_project "$WANDB_PROJECT"
    --wandb_run_name "$RUN_NAME"
    --wandb_run_id "$WANDB_RUN_ID"
    --wandb_resume "$WANDB_RESUME"
    --wandb_dir "$WANDB_LOCAL_DIR"
    --wandb_group "$WANDB_GROUP"
    --wandb_tags ablation wo-ecot-pt libero-v21 qwen3-vl-2b
)

if [[ -n "$MAX_TRAIN_STEPS" ]]; then
    train_args+=(--max_train_steps "$MAX_TRAIN_STEPS")
fi

if [[ "$mode" == "resume" ]]; then
    train_args+=(
        --action_expert_name_or_path "$RESUME_CKPT"
        --resume_training
    )
fi

if [[ "${ZR0_DRY_RUN:-0}" == "1" ]]; then
    print_command env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 "${train_args[@]}"
    exit 0
fi

run_preflight
cd "$ROOT_DIR"

if [[ "$mode" == "train" && -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite existing output directory: $OUTPUT_DIR" >&2
    exit 1
fi

if [[ "$mode" == "resume" ]]; then
    if [[ ! -d "$RESUME_CKPT" || ! -f "$RESUME_CKPT/scheduler.pt" ]]; then
        echo "Resume checkpoint is incomplete or missing: $RESUME_CKPT" >&2
        exit 1
    fi
fi

if [[ "$mode" == "train" ]]; then
    mkdir -p "$OUTPUT_DIR"
    printf '%s\n' "$WANDB_RUN_ID" > "$WANDB_RUN_ID_FILE"
    printf '%s\n' "$RUN_NAME" > "$WANDB_RUN_NAME_FILE"
fi

mkdir -p "$LOG_DIR" "$WANDB_LOCAL_DIR"

print_command env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 "${train_args[@]}"
exec env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 "${train_args[@]}"
