#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${ZR0_TRAIN_PYTHON:-/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python}
ACCELERATE_BIN=${ZR0_ACCELERATE_BIN:-/opt/data/private/lq/miniconda3/envs/ZR-0/bin/accelerate}
ACCELERATE_CONFIG=${ZR0_ACCELERATE_CONFIG:-"$ROOT_DIR/accelerate_configs/accelerate_config.yaml"}
OUTPUT_ROOT=${ZR0_OUTPUT_ROOT:-"$ROOT_DIR/outputs/stage05_four_dataset_pretraining_20260904"}
EXPERIMENT_DOC=${ZR0_EXPERIMENT_DOC:-"$ROOT_DIR/docs/experiments/stage05_four_dataset_pretraining_20260904/experiment.md"}
MODEL_PATH=${ZR0_MODEL_PATH:-/opt/data/private/lq/models/Qwen3-VL-2B-Instruct}
ACTION_EXPERT_CONFIG=${ZR0_ACTION_EXPERT_CONFIG_PATH:-"$ROOT_DIR/configs/stage05_four_dataset_action_expert.json"}
TOKEN_LENGTH_AUDIT=${ZR0_TOKEN_LENGTH_AUDIT:-"$OUTPUT_ROOT/audits/token_length_audit_v9_format2.json"}
TOKEN_AUDIT_SPEC=${ZR0_TOKEN_AUDIT_SPEC:-"$ROOT_DIR/configs/stage05_four_dataset_experiment.json"}
TOKEN_AUDIT_SCRIPT=${ZR0_TOKEN_AUDIT_SCRIPT:-"$ROOT_DIR/scripts/audit_stage05_token_lengths.py"}
TRAIN_ENTRYPOINT=${ZR0_TRAIN_ENTRYPOINT:-"$ROOT_DIR/train_vla.py"}
MAX_LENGTH=${ZR0_MAX_LENGTH:-}
ACTION_HORIZON=${ZR0_ACTION_HORIZON:-32}
VISIBLE_DEVICES=${ZR0_CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_GPUS=${ZR0_NUM_GPUS:-4}
MICRO_BATCH_EXPLICIT=${ZR0_PER_DEVICE_BATCH_SIZE+x}
GAS_EXPLICIT=${ZR0_GRADIENT_ACCUMULATION_STEPS+x}
MICRO_BATCH=${ZR0_PER_DEVICE_BATCH_SIZE:-16}
GAS=${ZR0_GRADIENT_ACCUMULATION_STEPS:-2}
GLOBAL_BATCH=${ZR0_EXPECTED_GLOBAL_BATCH_SIZE:-128}
WANDB_PENDING_CAPACITY=${ZR0_WANDB_PENDING_CAPACITY:-256}
WANDB_RETRY_BASE_STEPS=${ZR0_WANDB_RETRY_BASE_STEPS:-1}
WANDB_RETRY_MAX_STEPS=${ZR0_WANDB_RETRY_MAX_STEPS:-128}
WANDB_FINISH_MAX_ATTEMPTS=${ZR0_WANDB_FINISH_MAX_ATTEMPTS:-2}
WANDB_FINISH_TIMEOUT_SECONDS=${ZR0_WANDB_FINISH_TIMEOUT_SECONDS:-15}

usage() {
    echo "Usage: $0 {ar-smoke|ar-resume|ar-pilot|joint-smoke|joint-resume|joint-pilot|ar-formal|joint-formal}" >&2
}

stage=${1:-}
if [[ "$stage" == --help || "$stage" == -h ]]; then
    usage
    exit 0
fi
case "$stage" in
    ar-smoke|ar-resume|ar-pilot|joint-smoke|joint-resume|joint-pilot|ar-formal|joint-formal) ;;
    *) usage; exit 2 ;;
esac
if [[ -z "$MAX_LENGTH" || ! "$MAX_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
    echo "ZR0_MAX_LENGTH must be set to the audited positive integer" >&2
    exit 1
fi
if [[ ! "$ACTION_HORIZON" =~ ^[1-9][0-9]*$ ]]; then
    echo "ZR0_ACTION_HORIZON must be a positive integer" >&2
    exit 1
fi
for value in "$NUM_GPUS" "$MICRO_BATCH" "$GAS" "$GLOBAL_BATCH" \
    "$WANDB_PENDING_CAPACITY" "$WANDB_RETRY_BASE_STEPS" \
    "$WANDB_RETRY_MAX_STEPS" "$WANDB_FINISH_MAX_ATTEMPTS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "batch and W&B settings must be positive integers" >&2; exit 1; }
done
"$PYTHON_BIN" -c 'import math,sys; value=float(sys.argv[1]); assert math.isfinite(value) and value > 0' \
    "$WANDB_FINISH_TIMEOUT_SECONDS" || {
    echo "W&B finish timeout must be a finite positive number" >&2; exit 1;
}
(( WANDB_RETRY_MAX_STEPS >= WANDB_RETRY_BASE_STEPS )) || {
    echo "W&B retry max steps must be at least retry base steps" >&2
    exit 1
}
(( NUM_GPUS * MICRO_BATCH * GAS == GLOBAL_BATCH )) || {
    echo "global batch mismatch: $NUM_GPUS x $MICRO_BATCH x $GAS != $GLOBAL_BATCH" >&2
    exit 1
}
[[ "$NUM_GPUS" == 4 && "$VISIBLE_DEVICES" == 0,1,2,3 ]] || {
    echo "this experiment requires exactly CUDA devices 0,1,2,3" >&2
    exit 1
}
[[ -f "$EXPERIMENT_DOC" ]] || { echo "missing experiment document: $EXPERIMENT_DOC" >&2; exit 1; }
[[ -f "$ACTION_EXPERT_CONFIG" ]] || {
    echo "missing authoritative Action Expert config: $ACTION_EXPERT_CONFIG" >&2; exit 1;
}
[[ -f "$TOKEN_AUDIT_SPEC" ]] || {
    echo "missing trusted Stage05 token audit specification: $TOKEN_AUDIT_SPEC" >&2; exit 1;
}

read -r CONFIG_GAS CONFIG_MICRO CONFIG_GLOBAL < <(
    "$PYTHON_BIN" -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1]))["deepspeed_config"]; print(c["gradient_accumulation_steps"],c["train_micro_batch_size_per_gpu"],c["train_batch_size"])' "$ACCELERATE_CONFIG"
)
[[ "$CONFIG_GAS" == "$GAS" && "$CONFIG_MICRO" == "$MICRO_BATCH" && "$CONFIG_GLOBAL" == "$GLOBAL_BATCH" ]] || {
    echo "Accelerate/DeepSpeed batch integers differ from CLI: config=$CONFIG_MICRO/$CONFIG_GAS/$CONFIG_GLOBAL" >&2
    exit 1
}

entries=(stage05_droid_mixed stage05_household_mixed stage05_tabletop_mixed stage05_rh20t_mixed)
loss_type=vlm
fm_weight=0
max_steps=2
resume=0
checkpoint_load_purpose=
run_dir="$OUTPUT_ROOT/smoke/ar"
model_input="$MODEL_PATH"
case "$stage" in
    ar-resume)
        max_steps=3; resume=1; checkpoint_load_purpose=stage05_ar_resume
        model_input="$OUTPUT_ROOT/smoke/ar/latest-model-optimizer-lr"
        ;;
    ar-pilot)
        max_steps=100; run_dir="$OUTPUT_ROOT/pilot/ar"
        ;;
    joint-smoke)
        loss_type=vlm_and_action; fm_weight=5; checkpoint_load_purpose=stage05_ar_to_joint; run_dir="$OUTPUT_ROOT/smoke/joint"
        model_input="$OUTPUT_ROOT/pilot/ar/latest-model-optimizer-lr"
        ;;
    joint-resume)
        loss_type=vlm_and_action; fm_weight=5; checkpoint_load_purpose=stage05_joint_resume; max_steps=3; resume=1
        run_dir="$OUTPUT_ROOT/smoke/joint"; model_input="$run_dir/latest-model-optimizer-lr"
        ;;
    joint-pilot)
        loss_type=vlm_and_action; fm_weight=5; checkpoint_load_purpose=stage05_ar_to_joint; max_steps=100; run_dir="$OUTPUT_ROOT/pilot/joint"
        model_input="$OUTPUT_ROOT/pilot/ar/latest-model-optimizer-lr"
        ;;
    ar-formal)
        max_steps=${ZR0_FORMAL_MAX_STEPS:-}
        run_dir=${ZR0_FORMAL_AR_OUTPUT_DIR:-"$OUTPUT_ROOT/formal/ar"}
        ;;
    joint-formal)
        loss_type=vlm_and_action; fm_weight=5; checkpoint_load_purpose=stage05_ar_to_joint; max_steps=${ZR0_FORMAL_MAX_STEPS:-}
        run_dir=${ZR0_FORMAL_JOINT_OUTPUT_DIR:-"$OUTPUT_ROOT/formal/joint"}
        model_input=${ZR0_FORMAL_AR_CHECKPOINT:-"$OUTPUT_ROOT/formal/ar/latest-model-optimizer-lr"}
        ;;
esac

run_dir=${ZR0_RUN_OUTPUT_DIR:-"$run_dir"}
if (( resume )); then
    model_input="$run_dir/latest-model-optimizer-lr"
fi
model_input=${ZR0_INITIAL_CHECKPOINT:-"$model_input"}
save_step_interval=${ZR0_SAVE_STEP_INTERVAL:-"$max_steps"}

if [[ "$stage" == *-formal ]]; then
    [[ -n "$MICRO_BATCH_EXPLICIT" && -n "$GAS_EXPLICIT" ]] || {
        echo "formal template requires explicit probe-validated ZR0_PER_DEVICE_BATCH_SIZE and ZR0_GRADIENT_ACCUMULATION_STEPS" >&2; exit 1;
    }
    [[ "$max_steps" =~ ^[1-9][0-9]*$ ]] || {
        echo "formal template requires explicit positive ZR0_FORMAL_MAX_STEPS" >&2; exit 1;
    }
    [[ -n "${ZR0_WANDB_PROJECT:-}" ]] || {
        echo "formal template requires ZR0_WANDB_PROJECT; W&B remains best-effort" >&2; exit 1;
    }
    if [[ "${ZR0_DRY_RUN:-0}" != 1 && "${ZR0_ALLOW_FORMAL_TRAINING:-0}" != 1 ]]; then
        echo "formal training is locked; set ZR0_ALLOW_FORMAL_TRAINING=1 only after explicit approval" >&2
        exit 1
    fi
fi

[[ "$save_step_interval" =~ ^[1-9][0-9]*$ ]] || {
    echo "save step interval must be a positive integer" >&2; exit 1;
}

if (( resume )); then
    [[ -f "$model_input/scheduler.pt" ]] || { echo "incomplete resume checkpoint: $model_input" >&2; exit 1; }
else
    [[ ! -e "$run_dir" ]] || { echo "refusing to overwrite output: $run_dir" >&2; exit 1; }
fi
if [[ "$stage" == joint-smoke || "$stage" == joint-pilot || "$stage" == joint-formal ]]; then
    [[ -f "$model_input/zr0_checkpoint_metadata.json" && -f "$model_input/difference_query.safetensors" ]] || {
        echo "Joint warm start requires an AR-only VLM+Query checkpoint: $model_input" >&2; exit 1;
    }
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" \
        -m utils.stage05_checkpoint_contract \
        --checkpoint "$model_input" --purpose stage05_ar_to_joint \
        --external-config "$ACTION_EXPERT_CONFIG" \
        --action-dim 64 --state-dim 64 --action-horizon "$ACTION_HORIZON" \
        --num-difference-queries 32 >/dev/null
fi

if [[ "$stage" == ar-resume || "$stage" == joint-resume ]]; then
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" \
        -m utils.stage05_checkpoint_contract \
        --checkpoint "$model_input" --purpose "$checkpoint_load_purpose" \
        --resume-training --validate-resume-artifacts \
        --action-dim 64 --state-dim 64 --action-horizon "$ACTION_HORIZON" \
        --num-difference-queries 32 >/dev/null
fi

TOKEN_AUDIT_VALIDATION=$(
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" \
        "$TOKEN_AUDIT_SCRIPT" \
        --validate-only --report "$TOKEN_LENGTH_AUDIT" \
        --processor-path "$model_input" --max-length "$MAX_LENGTH" \
        --trusted-spec "$TOKEN_AUDIT_SPEC"
)
REQUIRED_MAX_LENGTH=$(
    "$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.argv[1])["required_max_length"])' \
        "$TOKEN_AUDIT_VALIDATION"
)

config_validation_args=(
    -m utils.action_expert_config
    --config "$ACTION_EXPERT_CONFIG"
    --vlm "$model_input"
    --action-dim 64 --state-dim 64 --action-horizon "$ACTION_HORIZON"
)
if [[ "$loss_type" == vlm_and_action ]]; then
    config_validation_args+=(
        --difference-query-config "$model_input/difference_query_config.json"
        --num-difference-queries 32
    )
fi
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "${config_validation_args[@]}" >/dev/null

args=(
    "$ACCELERATE_BIN" launch --config_file "$ACCELERATE_CONFIG"
    --num_processes "$NUM_GPUS" --mixed_precision bf16
    --gradient_accumulation_steps "$GAS" "$TRAIN_ENTRYPOINT"
    --vlm_name_or_path "$model_input"
    --action_expert_config_path "$ACTION_EXPERT_CONFIG"
    --per_device_train_batch_size "$MICRO_BATCH"
    --gradient_accumulation_steps "$GAS" --expected_global_batch_size "$GLOBAL_BATCH"
    --seed 42 --epochs 1 --max_train_steps "$max_steps" --save_step_interval "$save_step_interval"
    --peak_learning_rate 1e-5 --min_lr_rate 0.1 --warmup_ratio 0.05
    --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-8
    --tensorboard_log_dir "$run_dir/tensorboard" --output_ckpt_dir "$run_dir"
    --tune_vlm --loss_type "$loss_type" --vlm_loss_weight 1.0
    --action_expert_loss_weight "$fm_weight" --lr_scheduler cosine
    --dataset_entries "${entries[@]}" --dataset_sample_ratios 1 1 1 1
    --window_size 1 --action_horizon "$ACTION_HORIZON" --max_pad_state_and_action_length 64
    --max_length "$MAX_LENGTH" --dataloader_num_workers 4 --logging_steps 1
    --log_training_diagnostics --use_difference_query --num_difference_queries 32
    --vlm_attention_backend sdpa --save_optimizer_and_lr_states
    --wandb_failure_policy best_effort --wandb_dir "$run_dir/wandb"
    --wandb_pending_capacity "$WANDB_PENDING_CAPACITY"
    --wandb_retry_base_steps "$WANDB_RETRY_BASE_STEPS"
    --wandb_retry_max_steps "$WANDB_RETRY_MAX_STEPS"
    --wandb_finish_max_attempts "$WANDB_FINISH_MAX_ATTEMPTS"
    --wandb_finish_timeout_seconds "$WANDB_FINISH_TIMEOUT_SECONDS"
)
if [[ -n "$checkpoint_load_purpose" ]]; then
    args+=(--checkpoint_load_purpose "$checkpoint_load_purpose")
fi
if [[ -n "${ZR0_WANDB_PROJECT:-}" ]]; then
    args+=(--wandb_project "$ZR0_WANDB_PROJECT")
    [[ -n "${ZR0_WANDB_GROUP:-}" ]] && args+=(--wandb_group "$ZR0_WANDB_GROUP")
    [[ -n "${ZR0_WANDB_RUN_NAME:-}" ]] && args+=(--wandb_run_name "$ZR0_WANDB_RUN_NAME")
    [[ -n "${ZR0_WANDB_RUN_ID:-}" ]] && args+=(--wandb_run_id "$ZR0_WANDB_RUN_ID")
    args+=(--wandb_resume "$([[ $resume == 1 ]] && echo must || echo never)")
fi
[[ "$loss_type" == vlm_and_action ]] && args+=(--tune_action_expert)
if (( resume )); then
    args+=(--resume_training)
    [[ "$loss_type" == vlm_and_action ]] && args+=(--action_expert_name_or_path "$model_input")
fi

if [[ "${ZR0_DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" "${args[@]}"
    printf '\n'
    exit 0
fi
mkdir -p "$run_dir"
cp -n -- "$EXPERIMENT_DOC" "$run_dir/experiment.md"
exec > >(tee -a "$run_dir/train.log") 2>&1
cd "$ROOT_DIR"
printf '%q ' env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" "${args[@]}"
printf '\n'
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" "${args[@]}"
