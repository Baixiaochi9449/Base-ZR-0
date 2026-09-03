#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
    echo "Usage: $0 {ar|joint} {train|resume}" >&2
}

require_env() {
    local name=$1
    if [[ -z "${!name:-}" ]]; then
        echo "Missing required environment variable: $name" >&2
        exit 1
    fi
}

require_positive_integer() {
    local name=$1
    local value=${!name}
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer" >&2
        exit 1
    fi
}

print_command() {
    printf '%q' "$1"
    shift
    printf ' %q' "$@"
    printf '\n'
}

stage=${1:-}
case "$stage" in
    ar|joint)
        ;;
    *)
        usage
        exit 2
        ;;
esac
run_mode=${2:-train}
case "$run_mode" in
    train|resume)
        ;;
    *)
        usage
        exit 2
        ;;
esac

required_variables=(
    MODEL_PATH OUTPUT_DIR EXPERIMENT_DOC MAX_LENGTH
    EPOCHS MAX_TRAIN_STEPS SAVE_STEP_INTERVAL EXPECTED_GLOBAL_BATCH_SIZE
    NUM_GPUS PER_DEVICE_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
    DATASET_ENTRIES SAMPLE_RATIOS CUDA_VISIBLE_DEVICES
    WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_GROUP WANDB_RUN_NAME WANDB_RUN_ID
)
for name in "${required_variables[@]}"; do
    require_env "$name"
done
for name in EPOCHS MAX_TRAIN_STEPS SAVE_STEP_INTERVAL EXPECTED_GLOBAL_BATCH_SIZE NUM_GPUS PER_DEVICE_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS MAX_LENGTH; do
    require_positive_integer "$name"
done
if [[ -n "${WANDB_MODE:-}" && "$WANDB_MODE" != "online" ]]; then
    echo "WANDB_MODE must be unset or 'online' for formal training" >&2
    exit 1
fi
if [[ ! -f "$EXPERIMENT_DOC" ]]; then
    echo "EXPERIMENT_DOC does not exist: $EXPERIMENT_DOC" >&2
    exit 1
fi
if [[ "$run_mode" == "resume" ]]; then
    expected_resume_path="$OUTPUT_DIR/latest-model-optimizer-lr"
    if [[ "$(realpath -m "$MODEL_PATH")" != "$(realpath -m "$expected_resume_path")" ]]; then
        echo "resume MODEL_PATH must be $expected_resume_path" >&2
        exit 1
    fi
    if [[ ! -d "$MODEL_PATH" || ! -f "$MODEL_PATH/scheduler.pt" ]]; then
        echo "resume checkpoint is incomplete or missing: $MODEL_PATH" >&2
        exit 1
    fi
fi
if [[ "$stage" == "joint" || "$run_mode" == "resume" ]]; then
    query_config="$MODEL_PATH/difference_query_config.json"
    query_weights="$MODEL_PATH/difference_query.safetensors"
    checkpoint_metadata="$MODEL_PATH/zr0_checkpoint_metadata.json"
    if [[ ! -f "$query_config" || ! -f "$query_weights" || ! -f "$checkpoint_metadata" ]]; then
        echo "joint MODEL_PATH must be an AR checkpoint with Difference Query sidecars" >&2
        exit 1
    fi
    checkpoint_kind=$(
        python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("checkpoint_kind"))' "$checkpoint_metadata"
    )
    expected_checkpoint_kind=ar_only
    if [[ "$stage" == "joint" && "$run_mode" == "resume" ]]; then
        expected_checkpoint_kind=joint
    fi
    if [[ "$checkpoint_kind" != "$expected_checkpoint_kind" ]]; then
        echo "$stage $run_mode MODEL_PATH checkpoint kind must be $expected_checkpoint_kind, got: $checkpoint_kind" >&2
        exit 1
    fi
    query_shape=$(
        python -c 'import json,sys; c=json.load(open(sys.argv[1], encoding="utf-8")); print(str(c.get("enabled")) + ":" + str(c.get("num_difference_queries")))' "$query_config"
    )
    if [[ "$query_shape" != "True:32" ]]; then
        echo "joint MODEL_PATH Difference Query must be enabled with 32 queries" >&2
        exit 1
    fi
    if [[ "$stage" == "joint" && "$run_mode" == "resume" && \
          ( ! -f "$MODEL_PATH/action_expert.safetensors" || \
            ! -f "$MODEL_PATH/action_expert_config.json" ) ]]; then
        echo "joint resume checkpoint is missing Action Expert weights or config: $MODEL_PATH" >&2
        exit 1
    fi
fi

global_batch_size=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if (( global_batch_size != EXPECTED_GLOBAL_BATCH_SIZE )); then
    echo "global batch size must be $EXPECTED_GLOBAL_BATCH_SIZE, got $global_batch_size = $NUM_GPUS x $PER_DEVICE_BATCH_SIZE x $GRADIENT_ACCUMULATION_STEPS" >&2
    exit 1
fi

read -r -a dataset_entries <<< "$DATASET_ENTRIES"
read -r -a dataset_sample_ratios <<< "$SAMPLE_RATIOS"
if (( ${#dataset_entries[@]} != ${#dataset_sample_ratios[@]} )); then
    echo "DATASET_ENTRIES and SAMPLE_RATIOS must have equal length" >&2
    exit 1
fi
for ratio in "${dataset_sample_ratios[@]}"; do
    if ! python -c 'import math,sys; x=float(sys.argv[1]); sys.exit(0 if math.isfinite(x) and 0 < x <= 1 else 1)' "$ratio" 2>/dev/null; then
        echo "SAMPLE_RATIOS values must be finite, positive, and at most one: $ratio" >&2
        exit 1
    fi
done

IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#visible_devices[@]} != NUM_GPUS )); then
    echo "CUDA_VISIBLE_DEVICES must list exactly NUM_GPUS devices" >&2
    exit 1
fi
if [[ "$CUDA_VISIBLE_DEVICES" != "0,1,2,3" || "$NUM_GPUS" != "4" ]]; then
    echo "this experiment requires NUM_GPUS=4 and CUDA_VISIBLE_DEVICES=0,1,2,3" >&2
    exit 1
fi

loss_type=vlm
action_expert_loss_weight=0
if [[ "$stage" == "joint" ]]; then
    loss_type=vlm_and_action
    action_expert_loss_weight=5.0
fi
accelerate_bin=${ACCELERATE_BIN:-accelerate}
accelerate_config=${ACCELERATE_CONFIG:-"$ROOT_DIR/accelerate_configs/accelerate_config.yaml"}
python_bin=${TRAIN_PYTHON:-}
if [[ -z "$python_bin" ]]; then
    sibling_python="$(dirname "$accelerate_bin")/python"
    if [[ "$accelerate_bin" == */* && -x "$sibling_python" ]]; then
        python_bin="$sibling_python"
    else
        python_bin=python
    fi
fi
if [[ ! -f "$accelerate_config" ]]; then
    echo "Accelerate config does not exist: $accelerate_config" >&2
    exit 1
fi
read -r config_gas config_micro_batch config_global_batch < <(
    "$python_bin" -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["deepspeed_config"]; print(c["gradient_accumulation_steps"], c["train_micro_batch_size_per_gpu"], c["train_batch_size"])' "$accelerate_config"
)
if [[ ! "$config_gas" =~ ^[1-9][0-9]*$ || "$config_gas" != "$GRADIENT_ACCUMULATION_STEPS" ]]; then
    echo "Accelerate/DeepSpeed config GAS must be integer $GRADIENT_ACCUMULATION_STEPS, got: $config_gas" >&2
    exit 1
fi
if [[ ! "$config_micro_batch" =~ ^[1-9][0-9]*$ || "$config_micro_batch" != "$PER_DEVICE_BATCH_SIZE" ]]; then
    echo "Accelerate/DeepSpeed config micro-batch must be integer $PER_DEVICE_BATCH_SIZE, got: $config_micro_batch" >&2
    exit 1
fi
if [[ ! "$config_global_batch" =~ ^[1-9][0-9]*$ || "$config_global_batch" != "$EXPECTED_GLOBAL_BATCH_SIZE" ]]; then
    echo "Accelerate/DeepSpeed config global batch must be integer $EXPECTED_GLOBAL_BATCH_SIZE, got: $config_global_batch" >&2
    exit 1
fi
logging_steps=${LOGGING_STEPS:-10}
if [[ ! "$logging_steps" =~ ^[1-9][0-9]*$ ]]; then
    echo "LOGGING_STEPS must be a positive integer" >&2
    exit 1
fi
wandb_resume=never
if [[ "$run_mode" == "resume" ]]; then
    wandb_resume=must
fi

train_args=(
    "$accelerate_bin" launch
    --config_file "$accelerate_config"
    --num_processes "$NUM_GPUS"
    --mixed_precision bf16
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    "$ROOT_DIR/train_vla.py"
    --vlm_name_or_path "$MODEL_PATH"
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --expected_global_batch_size "$EXPECTED_GLOBAL_BATCH_SIZE"
    --seed 42
    --epochs "$EPOCHS"
    --max_train_steps "$MAX_TRAIN_STEPS"
    --save_step_interval "$SAVE_STEP_INTERVAL"
    --peak_learning_rate 1e-5
    --min_lr_rate 0.1
    --warmup_ratio 0.05
    --adam_beta1 0.9
    --adam_beta2 0.95
    --adam_epsilon 1e-8
    --tensorboard_log_dir "$OUTPUT_DIR/tensorboard"
    --output_ckpt_dir "$OUTPUT_DIR"
    --tune_vlm
    --loss_type "$loss_type"
    --vlm_loss_weight 1.0
    --action_expert_loss_weight "$action_expert_loss_weight"
    --lr_scheduler cosine
    --dataset_entries "${dataset_entries[@]}"
    --dataset_sample_ratios "${dataset_sample_ratios[@]}"
    --window_size 1
    --action_horizon 32
    --max_pad_state_and_action_length 64
    --max_length "$MAX_LENGTH"
    --dataloader_num_workers 4
    --logging_steps "$logging_steps"
    --log_training_diagnostics
    --use_difference_query
    --num_difference_queries 32
    --vlm_attention_backend sdpa
    --save_optimizer_and_lr_states
    --wandb_project "$WANDB_PROJECT"
    --wandb_group "$WANDB_GROUP"
    --wandb_run_name "$WANDB_RUN_NAME"
    --wandb_run_id "$WANDB_RUN_ID"
    --wandb_resume "$wandb_resume"
    --wandb_dir "$OUTPUT_DIR/wandb"
    --wandb_tags query-conditioned-ar future-difference "$stage"
)
if [[ "$stage" == "joint" ]]; then
    train_args+=(--tune_action_expert)
fi
if [[ "$run_mode" == "resume" ]]; then
    train_args+=(--resume_training)
    if [[ "$stage" == "joint" ]]; then
        train_args+=(--action_expert_name_or_path "$MODEL_PATH")
    fi
fi

if [[ "${ZR0_DRY_RUN:-0}" == "1" ]]; then
    print_command env PYTHONNOUSERSITE=1 WANDB_MODE=online CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${train_args[@]}"
    exit 0
fi
if [[ "$run_mode" == "train" ]]; then
    if [[ -e "$OUTPUT_DIR" ]]; then
        echo "Refusing to overwrite existing formal output directory: $OUTPUT_DIR" >&2
        exit 1
    fi
    mkdir -p "$OUTPUT_DIR"
    cp -- "$EXPERIMENT_DOC" "$OUTPUT_DIR/experiment.md"
elif [[ ! -f "$OUTPUT_DIR/experiment.md" ]]; then
    echo "resume output is missing experiment.md: $OUTPUT_DIR" >&2
    exit 1
fi
exec > >(tee -a "$OUTPUT_DIR/train.log") 2>&1
record_name=launch_manifest_fresh.json
if [[ "$run_mode" == "resume" ]]; then
    record_name=launch_manifest_resume.json
fi
"$python_bin" "$ROOT_DIR/scripts/record_query_pretrain_launch.py" \
    --stage "$stage" \
    --run-mode "$run_mode" \
    --world-size "$NUM_GPUS" \
    --launcher "$ROOT_DIR/scripts/run_query_ar_joint_formal.sh" \
    --output "$OUTPUT_DIR/$record_name" \
    -- "${train_args[@]}"
cd "$ROOT_DIR"
print_command env PYTHONNOUSERSITE=1 WANDB_MODE=online CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${train_args[@]}"
start_time=$(date --iso-8601=seconds)
{
    printf '\n### Formal launcher run: %s %s\n\n' "$stage" "$run_mode"
    printf -- '- started: %s\n' "$start_time"
    printf -- '- command:\n\n```bash\n'
    print_command env PYTHONNOUSERSITE=1 WANDB_MODE=online CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${train_args[@]}"
    printf '```\n'
} >> "$OUTPUT_DIR/experiment.md"
set +e
env PYTHONNOUSERSITE=1 WANDB_MODE=online CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${train_args[@]}"
exit_status=$?
set -e
{
    printf -- '- ended: %s\n' "$(date --iso-8601=seconds)"
    printf -- '- exit status: %s\n' "$exit_status"
    printf -- '- latest recovery checkpoint: %s\n' "$OUTPUT_DIR/latest-model-optimizer-lr"
} >> "$OUTPUT_DIR/experiment.md"
exit "$exit_status"
