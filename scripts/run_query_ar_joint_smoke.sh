#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
usage() {
    echo "Usage: $0 {ar-step1|ar-resume-step2|joint-step1|joint-resume-step2}" >&2
}

require_env() {
    local name=$1
    if [[ -z "${!name:-}" ]]; then
        echo "Missing required environment variable: $name" >&2
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
    ar-step1|ar-resume-step2|joint-step1|joint-resume-step2)
        ;;
    *)
        usage
        exit 2
        ;;
esac

for name in \
    ZR0_MODEL_PATH \
    ZR0_DATASET_ENTRIES \
    ZR0_DATASET_SAMPLE_RATIOS \
    ZR0_MAX_LENGTH \
    ZR0_SMOKE_OUTPUT_ROOT \
    ZR0_EXPERIMENT_DOC \
    ZR0_NUM_GPUS \
    ZR0_PER_DEVICE_BATCH_SIZE \
    ZR0_GRADIENT_ACCUMULATION_STEPS \
    ZR0_EXPECTED_GLOBAL_BATCH_SIZE \
    ZR0_WANDB_PROJECT \
    ZR0_WANDB_GROUP \
    ZR0_WANDB_AR_RUN_NAME \
    ZR0_WANDB_AR_RUN_ID \
    ZR0_WANDB_JOINT_RUN_NAME \
    ZR0_WANDB_JOINT_RUN_ID \
    WANDB_API_KEY \
    WANDB_ENTITY
do
    require_env "$name"
done
if [[ -n "${WANDB_MODE:-}" && "$WANDB_MODE" != "online" ]]; then
    echo "WANDB_MODE must be unset or 'online' for real smoke" >&2
    exit 1
fi

read -r -a dataset_entries <<< "$ZR0_DATASET_ENTRIES"
read -r -a dataset_sample_ratios <<< "$ZR0_DATASET_SAMPLE_RATIOS"
if (( ${#dataset_entries[@]} != ${#dataset_sample_ratios[@]} )); then
    echo "ZR0_DATASET_ENTRIES and ZR0_DATASET_SAMPLE_RATIOS must have equal length" >&2
    exit 1
fi
for value in ZR0_NUM_GPUS ZR0_PER_DEVICE_BATCH_SIZE ZR0_GRADIENT_ACCUMULATION_STEPS ZR0_EXPECTED_GLOBAL_BATCH_SIZE; do
    if [[ ! "${!value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value must be a positive integer" >&2
        exit 1
    fi
done
smoke_global_batch=$((ZR0_NUM_GPUS * ZR0_PER_DEVICE_BATCH_SIZE * ZR0_GRADIENT_ACCUMULATION_STEPS))
if (( smoke_global_batch != ZR0_EXPECTED_GLOBAL_BATCH_SIZE )); then
    echo "smoke global batch mismatch: $smoke_global_batch != $ZR0_EXPECTED_GLOBAL_BATCH_SIZE" >&2
    exit 1
fi
for ratio in "${dataset_sample_ratios[@]}"; do
    if ! python -c 'import math,sys; x=float(sys.argv[1]); sys.exit(0 if math.isfinite(x) and 0 < x <= 1 else 1)' "$ratio" 2>/dev/null; then
        echo "ZR0_DATASET_SAMPLE_RATIOS values must be finite and in (0, 1]: $ratio" >&2
        exit 1
    fi
done
if [[ ! "$ZR0_MAX_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
    echo "ZR0_MAX_LENGTH must be a positive integer" >&2
    exit 1
fi

accelerate_bin=${ZR0_ACCELERATE_BIN:-accelerate}
accelerate_config=${ZR0_ACCELERATE_CONFIG:-"$ROOT_DIR/accelerate_configs/accelerate_config.yaml"}
python_bin=${ZR0_TRAIN_PYTHON:-}
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
if [[ ! "$config_gas" =~ ^[1-9][0-9]*$ || "$config_gas" != "$ZR0_GRADIENT_ACCUMULATION_STEPS" ]]; then
    echo "Accelerate/DeepSpeed config GAS must be integer $ZR0_GRADIENT_ACCUMULATION_STEPS, got: $config_gas" >&2
    exit 1
fi
if [[ ! "$config_micro_batch" =~ ^[1-9][0-9]*$ || "$config_micro_batch" != "$ZR0_PER_DEVICE_BATCH_SIZE" ]]; then
    echo "Accelerate/DeepSpeed config micro-batch must be integer $ZR0_PER_DEVICE_BATCH_SIZE, got: $config_micro_batch" >&2
    exit 1
fi
if [[ ! "$config_global_batch" =~ ^[1-9][0-9]*$ || "$config_global_batch" != "$ZR0_EXPECTED_GLOBAL_BATCH_SIZE" ]]; then
    echo "Accelerate/DeepSpeed config global batch must be integer $ZR0_EXPECTED_GLOBAL_BATCH_SIZE, got: $config_global_batch" >&2
    exit 1
fi
visible_devices=${ZR0_CUDA_VISIBLE_DEVICES:-0,1,2,3}
if [[ "$visible_devices" != "0,1,2,3" || "$ZR0_NUM_GPUS" != "4" ]]; then
    echo "real smoke requires ZR0_NUM_GPUS=4 and ZR0_CUDA_VISIBLE_DEVICES=0,1,2,3" >&2
    exit 1
fi
ar_output="$ZR0_SMOKE_OUTPUT_ROOT/ar"
joint_output="$ZR0_SMOKE_OUTPUT_ROOT/joint"
ar_checkpoint="$ar_output/latest-model-optimizer-lr"
joint_checkpoint="$joint_output/latest-model-optimizer-lr"

case "$stage" in
    ar-step1)
        model_input=$ZR0_MODEL_PATH
        output_dir=$ar_output
        loss_type=vlm
        action_expert_loss_weight=0
        max_steps=1
        wandb_run_name=$ZR0_WANDB_AR_RUN_NAME
        wandb_run_id=$ZR0_WANDB_AR_RUN_ID
        wandb_resume=never
        ;;
    ar-resume-step2)
        model_input=$ar_checkpoint
        output_dir=$ar_output
        loss_type=vlm
        action_expert_loss_weight=0
        max_steps=2
        wandb_run_name=$ZR0_WANDB_AR_RUN_NAME
        wandb_run_id=$ZR0_WANDB_AR_RUN_ID
        wandb_resume=must
        ;;
    joint-step1)
        model_input=$ar_checkpoint
        output_dir=$joint_output
        loss_type=vlm_and_action
        action_expert_loss_weight=5.0
        max_steps=1
        wandb_run_name=$ZR0_WANDB_JOINT_RUN_NAME
        wandb_run_id=$ZR0_WANDB_JOINT_RUN_ID
        wandb_resume=never
        ;;
    joint-resume-step2)
        model_input=$joint_checkpoint
        output_dir=$joint_output
        loss_type=vlm_and_action
        action_expert_loss_weight=5.0
        max_steps=2
        wandb_run_name=$ZR0_WANDB_JOINT_RUN_NAME
        wandb_run_id=$ZR0_WANDB_JOINT_RUN_ID
        wandb_resume=must
        ;;
esac

train_args=(
    "$accelerate_bin" launch
    --config_file "$accelerate_config"
    --num_processes "$ZR0_NUM_GPUS"
    --mixed_precision bf16
    --gradient_accumulation_steps "$ZR0_GRADIENT_ACCUMULATION_STEPS"
    "$ROOT_DIR/train_vla.py"
    --vlm_name_or_path "$model_input"
    --per_device_train_batch_size "$ZR0_PER_DEVICE_BATCH_SIZE"
    --gradient_accumulation_steps "$ZR0_GRADIENT_ACCUMULATION_STEPS"
    --expected_global_batch_size "$ZR0_EXPECTED_GLOBAL_BATCH_SIZE"
    --seed 42
    --epochs 1
    --max_train_steps "$max_steps"
    --save_step_interval 1
    --peak_learning_rate 1e-5
    --min_lr_rate 0.1
    --warmup_ratio 0.05
    --adam_beta1 0.9
    --adam_beta2 0.95
    --adam_epsilon 1e-8
    --tensorboard_log_dir "$output_dir/tensorboard"
    --output_ckpt_dir "$output_dir"
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
    --max_length "$ZR0_MAX_LENGTH"
    --dataloader_num_workers 4
    --logging_steps 1
    --log_training_diagnostics
    --use_difference_query
    --num_difference_queries 32
    --vlm_attention_backend sdpa
    --save_optimizer_and_lr_states
    --wandb_project "$ZR0_WANDB_PROJECT"
    --wandb_group "$ZR0_WANDB_GROUP"
    --wandb_run_name "$wandb_run_name"
    --wandb_run_id "$wandb_run_id"
    --wandb_resume "$wandb_resume"
    --wandb_dir "$output_dir/wandb"
    --wandb_tags query-conditioned-ar future-difference smoke "$loss_type"
)

if [[ "$loss_type" == "vlm_and_action" ]]; then
    train_args+=(--tune_action_expert)
fi
if [[ "$stage" == "ar-resume-step2" ]]; then
    train_args+=(--resume_training)
elif [[ "$stage" == "joint-resume-step2" ]]; then
    train_args+=(--action_expert_name_or_path "$joint_checkpoint" --resume_training)
fi

if [[ "${ZR0_DRY_RUN:-0}" == "1" ]]; then
    print_command env PYTHONNOUSERSITE=1 WANDB_MODE=online CUDA_VISIBLE_DEVICES="$visible_devices" "${train_args[@]}"
    exit 0
fi

if [[ ! -f "$ZR0_EXPERIMENT_DOC" ]]; then
    echo "Missing smoke experiment document: $ZR0_EXPERIMENT_DOC" >&2
    exit 1
fi
case "$stage" in
    ar-step1)
        if [[ -e "$ar_output" ]]; then
            echo "Refusing to overwrite existing AR smoke output: $ar_output" >&2
            exit 1
        fi
        ;;
    ar-resume-step2|joint-step1)
        if [[ ! -d "$ar_checkpoint" || ! -f "$ar_checkpoint/scheduler.pt" ]]; then
            echo "AR smoke checkpoint is incomplete or missing: $ar_checkpoint" >&2
            exit 1
        fi
        if [[ "$stage" == "joint-step1" && \
              ( ! -f "$ar_checkpoint/difference_query_config.json" || \
                ! -f "$ar_checkpoint/difference_query.safetensors" || \
                ! -f "$ar_checkpoint/zr0_checkpoint_metadata.json" ) ]]; then
            echo "AR smoke checkpoint is missing checkpoint metadata or Difference Query sidecars: $ar_checkpoint" >&2
            exit 1
        fi
        if [[ "$stage" == "joint-step1" ]]; then
            checkpoint_kind=$(
                python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("checkpoint_kind"))' \
                    "$ar_checkpoint/zr0_checkpoint_metadata.json"
            )
            if [[ "$checkpoint_kind" != "ar_only" ]]; then
                echo "joint smoke warm start checkpoint kind must be ar_only, got: $checkpoint_kind" >&2
                exit 1
            fi
        fi
        if [[ "$stage" == "joint-step1" && -e "$joint_output" ]]; then
            echo "Refusing to overwrite existing joint smoke output: $joint_output" >&2
            exit 1
        fi
        ;;
    joint-resume-step2)
        if [[ ! -d "$joint_checkpoint" || ! -f "$joint_checkpoint/scheduler.pt" || \
              ! -f "$joint_checkpoint/action_expert.safetensors" || \
              ! -f "$joint_checkpoint/action_expert_config.json" ]]; then
            echo "Joint smoke checkpoint is incomplete or missing: $joint_checkpoint" >&2
            exit 1
        fi
        ;;
esac

mkdir -p "$output_dir"
experiment_record="$output_dir/experiment.md"
if [[ ! -f "$experiment_record" ]]; then
    cp -- "$ZR0_EXPERIMENT_DOC" "$experiment_record"
fi
exec > >(tee -a "$output_dir/train.log") 2>&1
cd "$ROOT_DIR"
print_command env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$visible_devices" "${train_args[@]}"
start_time=$(date --iso-8601=seconds)
{
    printf '\n### Launcher run: %s\n\n' "$stage"
    printf -- '- started: %s\n' "$start_time"
    printf -- '- requested terminal optimizer step: %s\n' "$max_steps"
    printf -- '- command:\n\n```bash\n'
    print_command env PYTHONNOUSERSITE=1 WANDB_MODE=online CUDA_VISIBLE_DEVICES="$visible_devices" "${train_args[@]}"
    printf '```\n'
} >> "$experiment_record"

set +e
env PYTHONNOUSERSITE=1 WANDB_MODE=online CUDA_VISIBLE_DEVICES="$visible_devices" "${train_args[@]}"
exit_status=$?
set -e
{
    printf -- '- ended: %s\n' "$(date --iso-8601=seconds)"
    printf -- '- exit status: %s\n' "$exit_status"
    printf -- '- checkpoint directory: %s\n' "$output_dir/latest-model-optimizer-lr"
} >> "$experiment_record"
exit "$exit_status"
