#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${ZR0_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNTIME_ROOT=${ZR0_RUNTIME_ROOT:-$ROOT_DIR}
PYTHON_BIN=${ZR0_TRAIN_PYTHON:-/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python}
MODEL_PATH=/opt/data/private/lq/models/Qwen3-VL-2B-Instruct
PRETRAIN_JOINT_CKPT=${ZR0_PRETRAIN_JOINT_CKPT:-"$ROOT_DIR/outputs/pretrain/tabletop_v3_dq32_joint_gbs128_seed42_mbs16_gas2/step-19424"}
FAST_PATH="$ROOT_DIR/fast"
DATASET_PATH=/opt/data/private/lq/datasets/HuggingFaceVLA/libero
DATASET_ENTRY=libero_wo_ecot_pt
ACCELERATE_CONFIG="$ROOT_DIR/accelerate_configs/accelerate_config.yaml"
EXPERIMENT_TEMPLATE="$ROOT_DIR/experiment.md"
INITIAL_MODEL_PATH="$MODEL_PATH"
INITIAL_ACTION_EXPERT_PATH=
EXPECTED_CHECKPOINT_KIND=
EXPECTED_SOURCE_ACTION_HORIZON=
EXPECTED_NUM_DIFFERENCE_QUERIES=
REFERENCE_ACTION_EXPERT_CONFIG=
USE_EXPLICIT_LIBERO_BATCH_CONTRACT=0
USE_PRETRAINED_CHECKPOINT=0
LIBERO_ACTION_HORIZON=${ZR0_LIBERO_ACTION_HORIZON:-10}
EXPERIMENT_ARM=${2:-baseline_fa2}
case "$EXPERIMENT_ARM" in
    baseline_fa2)
        ARM_SUFFIX=FinalRMSNorm
        WANDB_GROUP=libero-wo-ecot-pt
        ;;
    baseline_sdpa)
        ARM_SUFFIX=FinalRMSNorm-baseline-sdpa
        WANDB_GROUP=libero-wo-ecot-pt-baseline-sdpa
        ;;
    difference_query)
        ARM_SUFFIX=FinalRMSNorm-difference-query-nq32
        WANDB_GROUP=libero-wo-ecot-pt-difference-query
        ;;
    difference_query_pretrained)
        USE_PRETRAINED_CHECKPOINT=1
        ARM_SUFFIX=FinalRMSNorm-difference-query-nq32-tabletop-v3-joint-init
        WANDB_GROUP=libero-wo-ecot-pt-difference-query-tabletop-v3-joint-init
        ACCELERATE_CONFIG="$ROOT_DIR/accelerate_configs/libero_zero2_bf16_mbs16_gas1.yaml"
        EXPERIMENT_TEMPLATE="$ROOT_DIR/docs/experiments/libero_wo_ecot_pt_dq32_tabletop_v3_joint_init/experiment.md"
        INITIAL_MODEL_PATH="$PRETRAIN_JOINT_CKPT"
        INITIAL_ACTION_EXPERT_PATH="$PRETRAIN_JOINT_CKPT"
        EXPECTED_CHECKPOINT_KIND=joint
        EXPECTED_NUM_DIFFERENCE_QUERIES=32
        # The legacy source is its own authority; do not substitute a future
        # LIBERO checkpoint that may not exist or may have a different contract.
        REFERENCE_ACTION_EXPERT_CONFIG="$PRETRAIN_JOINT_CKPT/action_expert_config.json"
        USE_EXPLICIT_LIBERO_BATCH_CONTRACT=1
        ;;
    difference_query_stage3)
        USE_PRETRAINED_CHECKPOINT=1
        USE_EXPLICIT_LIBERO_BATCH_CONTRACT=1
        ARM_SUFFIX=stage3-step14000-action-only-dq32
        WANDB_GROUP=libero-stage3-step14000-action-only-dq32-seed42
        RUNTIME_ROOT=${ZR0_RUNTIME_ROOT:-"$ROOT_DIR/outputs/runtime_snapshots/3eefb602417d3bd4b20bef2b47660b404aefb565"}
        PRETRAIN_JOINT_CKPT=${ZR0_PRETRAIN_JOINT_CKPT:-"$ROOT_DIR/outputs/three_stage_formal_20260910/stage3_resume8000_slot05_flow5/recovery_checkpoints/stage3_joint/step-014000-attempt-000/latest-model-optimizer-lr"}
        INITIAL_MODEL_PATH="$PRETRAIN_JOINT_CKPT"
        INITIAL_ACTION_EXPERT_PATH="$PRETRAIN_JOINT_CKPT"
        REFERENCE_ACTION_EXPERT_CONFIG="$PRETRAIN_JOINT_CKPT/action_expert_config.json"
        EXPECTED_CHECKPOINT_KIND=joint
        EXPECTED_NUM_DIFFERENCE_QUERIES=32
        ACCELERATE_CONFIG="$ROOT_DIR/accelerate_configs/libero_zero2_bf16_mbs16_gas1.yaml"
        EXPERIMENT_TEMPLATE="$ROOT_DIR/docs/experiments/libero_stage3_step14000/experiment.md"
        ZR0_OUTPUT_DIR=${ZR0_OUTPUT_DIR:-"$ROOT_DIR/outputs/ckpts/ZR0-stage3-step14000-LIBERO-action-only-dq32-h10-gbs64-seed42"}
        ;;
    *)
        echo "Unknown experiment arm: $EXPERIMENT_ARM" >&2
        exit 2
        ;;
esac
export ZR0_RUNTIME_ROOT="$RUNTIME_ROOT"
export PYTHONPATH="$RUNTIME_ROOT:$RUNTIME_ROOT/lerobot${PYTHONPATH:+:$PYTHONPATH}"
OUTPUT_DIR=${ZR0_OUTPUT_DIR:-"$ROOT_DIR/outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-$ARM_SUFFIX"}
EXPERIMENT_DOC="$OUTPUT_DIR/experiment.md"
LOG_BASE_DIR="$ROOT_DIR/outputs/train_logs/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-$ARM_SUFFIX"
WANDB_LOCAL_DIR="$ROOT_DIR/outputs/wandb"
WANDB_PROJECT=ZR-0-LIBERO
SAVE_STEP_INTERVAL=${ZR0_SAVE_STEP_INTERVAL:-2000}
MAX_TRAIN_STEPS=${ZR0_MAX_TRAIN_STEPS:-}
RESUME_CKPT=${ZR0_RESUME_CKPT:-"$OUTPUT_DIR/latest-model-optimizer-lr"}
WANDB_RUN_ID_FILE="$OUTPUT_DIR/.wandb-run-id"
WANDB_RUN_NAME_FILE="$OUTPUT_DIR/.wandb-run-name"

usage() {
    echo "Usage: $0 {preflight|train|resume} [baseline_fa2|baseline_sdpa|difference_query|difference_query_pretrained|difference_query_stage3]" >&2
}

print_command() {
    printf '%q' "$1"
    shift
    printf ' %q' "$@"
    printf '\n'
}

run_preflight() {
    local preflight_python=python
    if [[ "$EXPERIMENT_ARM" == "difference_query_stage3" ]]; then
        preflight_python="$PYTHON_BIN"
    fi
    local args=(
        "$preflight_python" "$RUNTIME_ROOT/scripts/preflight_libero_wo_ecot_pt.py"
        --model-path "$MODEL_INPUT_PATH"
        --fast-path "$FAST_PATH" \
        --dataset-path "$DATASET_PATH" \
        --min-cgroup-memory-headroom-gib 120 \
        --min-gpu-free-gib 70 \
        --require-wandb
    )
    if [[ "$USE_PRETRAINED_CHECKPOINT" == "1" ]]; then
        args+=(--output-path "$OUTPUT_DIR" --min-output-free-gib 200)
    fi
    if [[ -n "$EXPECTED_CHECKPOINT_KIND" ]]; then
        args+=(--expected-checkpoint-kind "$EXPECTED_CHECKPOINT_KIND")
    fi
    if [[ -n "$EXPECTED_NUM_DIFFERENCE_QUERIES" ]]; then
        args+=(--expected-num-difference-queries "$EXPECTED_NUM_DIFFERENCE_QUERIES")
    fi
    if [[ -n "$EXPECTED_SOURCE_ACTION_HORIZON" ]]; then
        args+=(--expected-source-action-horizon "$EXPECTED_SOURCE_ACTION_HORIZON")
    fi
    if [[ -n "$REFERENCE_ACTION_EXPERT_CONFIG" ]]; then
        args+=(--reference-action-expert-config "$REFERENCE_ACTION_EXPERT_CONFIG")
    fi
    "${args[@]}"
}

validate_checkpoint_for_training() {
    if [[ "$USE_PRETRAINED_CHECKPOINT" != "1" || "$EXPERIMENT_ARM" == "difference_query_stage3" ]]; then
        # The Stage 3 arm performs this check once in run_preflight. Dry-run
        # expands configuration without loading checkpoint tensor payloads.
        return
    fi
    if [[ ! -d "$MODEL_INPUT_PATH" ]]; then
        echo "missing LIBERO initialization/resume checkpoint: $MODEL_INPUT_PATH" >&2
        exit 1
    fi
    "$PYTHON_BIN" - "$MODEL_INPUT_PATH" "$EXPECTED_CHECKPOINT_KIND" "$EXPECTED_SOURCE_ACTION_HORIZON" "$EXPECTED_NUM_DIFFERENCE_QUERIES" "$REFERENCE_ACTION_EXPERT_CONFIG" "$ROOT_DIR/scripts/preflight_libero_wo_ecot_pt.py" <<'PY'
import importlib.util
import sys
from pathlib import Path

script = Path(sys.argv[6]).resolve()
spec = importlib.util.spec_from_file_location("zr0_libero_preflight", script)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
try:
    module.validate_checkpoint_artifacts(
        Path(sys.argv[1]),
        sys.argv[2],
        int(sys.argv[4]) if sys.argv[4] else None,
        int(sys.argv[3]) if sys.argv[3] else None,
        Path(sys.argv[5]) if sys.argv[5] else None,
    )
except module.PreflightError as error:
    print(f"Checkpoint preflight failed: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

validate_explicit_batch_config() {
    if [[ "$USE_EXPLICIT_LIBERO_BATCH_CONTRACT" != "1" ]]; then
        return
    fi
    config_gas=$(awk '$1 == "gradient_accumulation_steps:" {print $2}' "$ACCELERATE_CONFIG")
    config_micro=$(awk '$1 == "train_micro_batch_size_per_gpu:" {print $2}' "$ACCELERATE_CONFIG")
    config_global=$(awk '$1 == "train_batch_size:" {print $2}' "$ACCELERATE_CONFIG")
    config_world=$(awk '$1 == "num_processes:" {print $2}' "$ACCELERATE_CONFIG")
    if [[ "$config_gas:$config_micro:$config_global:$config_world" != "1:16:64:4" ]]; then
        echo "LIBERO config must be GAS=1, micro-batch=16, global-batch=64, world-size=4; got $config_gas:$config_micro:$config_global:$config_world" >&2
        exit 1
    fi
}

mode=${1:-}
case "$mode" in
    preflight|train|resume)
        ;;
    *)
        usage
        exit 2
        ;;
esac

if [[ ! "$LIBERO_ACTION_HORIZON" =~ ^[1-9][0-9]*$ ]]; then
    echo "ZR0_LIBERO_ACTION_HORIZON must be a positive integer" >&2
    exit 1
fi

MODEL_INPUT_PATH="$INITIAL_MODEL_PATH"
ACTION_EXPERT_INPUT_PATH="$INITIAL_ACTION_EXPERT_PATH"
if [[ "$mode" == "resume" ]]; then
    MODEL_INPUT_PATH="$RESUME_CKPT"
    ACTION_EXPERT_INPUT_PATH="$RESUME_CKPT"
    if [[ "$USE_PRETRAINED_CHECKPOINT" == "1" ]]; then
        EXPECTED_CHECKPOINT_KIND=action_only
        EXPECTED_SOURCE_ACTION_HORIZON="$LIBERO_ACTION_HORIZON"
    fi
fi

if [[ "$USE_PRETRAINED_CHECKPOINT" == "1" && "$mode" != "resume" ]]; then
    [[ -f "$MODEL_INPUT_PATH/action_expert_config.json" ]] || {
        echo "missing legacy Joint Action Expert config: $MODEL_INPUT_PATH/action_expert_config.json" >&2
        exit 1
    }
    EXPECTED_SOURCE_ACTION_HORIZON=$(
        "$PYTHON_BIN" -c 'import json,sys; value=json.load(open(sys.argv[1]))["action_horizon"]; assert isinstance(value,int) and not isinstance(value,bool) and value > 0; print(value)' \
            "$MODEL_INPUT_PATH/action_expert_config.json"
    )
fi

if [[ "$mode" == "preflight" ]]; then
    validate_explicit_batch_config
    run_preflight
    exit 0
fi

validate_checkpoint_for_training

if [[ "$mode" == "resume" ]]; then
    RUN_NAME=${ZR0_RUN_NAME:-$(cat "$WANDB_RUN_NAME_FILE" 2>/dev/null || true)}
    WANDB_RUN_ID=${ZR0_WANDB_RUN_ID:-$(cat "$WANDB_RUN_ID_FILE" 2>/dev/null || true)}
    WANDB_RESUME=must
else
    if [[ "$EXPERIMENT_ARM" == "difference_query_stage3" ]]; then
        default_run_name=zr0-stage3-step14000-libero-dq32-seed42-$(date +%Y%m%d-%H%M%S)
    elif [[ "$EXPERIMENT_ARM" == "difference_query_pretrained" ]]; then
        default_run_name=qwen3vl2b-libero-wo-ecot-dq32-tabletop-v3-joint-init-seed42-$(date +%Y%m%d-%H%M%S)
    else
        default_run_name=qwen3vl2b-libero-wo-ecot-pt-seed42-$(date +%Y%m%d-%H%M%S)
    fi
    RUN_NAME=${ZR0_RUN_NAME:-$default_run_name}
    WANDB_RUN_ID=${ZR0_WANDB_RUN_ID:-$("$PYTHON_BIN" -c 'import secrets, string; alphabet = string.ascii_lowercase + string.digits; print("".join(secrets.choice(alphabet) for _ in range(8)))')}
    WANDB_RESUME=never
fi

if [[ -z "$RUN_NAME" || -z "$WANDB_RUN_ID" ]]; then
    echo "Missing W&B run metadata for mode: $mode" >&2
    exit 1
fi

LOG_DIR="$LOG_BASE_DIR/$RUN_NAME"
TRAIN_ENTRY="$RUNTIME_ROOT/train_vla.py"
if [[ "$EXPERIMENT_ARM" == "difference_query_stage3" ]]; then
    TRAIN_ENTRY="$ROOT_DIR/scripts/train_libero_finetune.py"
fi

train_args=(
    accelerate launch
    --num_processes 4
    --config_file "$ACCELERATE_CONFIG"
    "$TRAIN_ENTRY"
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
    --action_horizon "$LIBERO_ACTION_HORIZON"
    --max_pad_state_and_action_length 64
    --save_optimizer_and_lr_states
    --wandb_project "$WANDB_PROJECT"
    --wandb_run_name "$RUN_NAME"
    --wandb_run_id "$WANDB_RUN_ID"
    --wandb_resume "$WANDB_RESUME"
    --wandb_dir "$WANDB_LOCAL_DIR"
    --wandb_group "$WANDB_GROUP"
    --wandb_tags ablation wo-ecot-pt libero-v21 qwen3-vl-2b "$EXPERIMENT_ARM"
)

if [[ "$USE_PRETRAINED_CHECKPOINT" == "1" ]]; then
    # This is a new downstream initialization from the completed Stage05 Joint
    # checkpoint; it is not a resume and must load the pretrained Expert.
    train_args+=(
        --checkpoint_load_purpose downstream_finetune
    )
    if [[ "$mode" != "resume" ]]; then
        train_args+=(
            --action_expert_config_path "$PRETRAIN_JOINT_CKPT/action_expert_config.json"
        )
    fi
fi

if [[ "$EXPERIMENT_ARM" == "difference_query_stage3" ]]; then
    train_args+=(--vlm_loss_weight 0.0 --slot_loss_weight 0.0 --optical_flow_loss_weight 0.0)
fi

if [[ "$USE_EXPLICIT_LIBERO_BATCH_CONTRACT" == "1" ]]; then
    train_args+=(
        --gradient_accumulation_steps 1
        --expected_global_batch_size 64
        --adam_beta1 0.9
        --adam_beta2 0.95
        --adam_epsilon 1e-6
        --warmup_ratio 0.08
        --max_length 1200
        --dataloader_num_workers 24
        --logging_steps 10
        --log_training_diagnostics
        --wandb_failure_policy required
    )
fi

case "$EXPERIMENT_ARM" in
    baseline_fa2)
        ;;
    baseline_sdpa)
        train_args+=(--vlm_attention_backend sdpa)
        ;;
    difference_query|difference_query_pretrained|difference_query_stage3)
        train_args+=(
            --use_difference_query
            --num_difference_queries 32
            --vlm_attention_backend sdpa
        )
        ;;
esac

if [[ -n "$MAX_TRAIN_STEPS" ]]; then
    train_args+=(--max_train_steps "$MAX_TRAIN_STEPS")
fi

if [[ -n "$ACTION_EXPERT_INPUT_PATH" ]]; then
    train_args+=(--action_expert_name_or_path "$ACTION_EXPERT_INPUT_PATH")
fi

if [[ "$mode" == "resume" ]]; then
    train_args+=(--resume_training)
fi

write_experiment_record() {
    if [[ ! -f "$EXPERIMENT_DOC" ]]; then
        if [[ ! -f "$EXPERIMENT_TEMPLATE" ]]; then
            echo "Missing experiment template: $EXPERIMENT_TEMPLATE" >&2
            return 1
        fi
        cp -- "$EXPERIMENT_TEMPLATE" "$EXPERIMENT_DOC"
    fi

    {
        printf '\n## Runtime launch record\n\n'
        printf -- '- 时间：%s\n' "$(date --iso-8601=seconds)"
        printf -- '- 模式：`%s`\n' "$mode"
        printf -- '- 实验臂：`%s`\n' "$EXPERIMENT_ARM"
        printf -- '- 输出目录：`%s`\n' "$OUTPUT_DIR"
        printf -- '- W&B project/group/run name/run ID：`%s` / `%s` / `%s` / `%s`\n' \
            "$WANDB_PROJECT" "$WANDB_GROUP" "$RUN_NAME" "$WANDB_RUN_ID"
        printf -- '- W&B run URL：待 `wandb.init` 成功后补充。\n'
        printf -- '- 完整启动命令：\n\n```bash\n'
        print_command env PYTHONNOUSERSITE=1 "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "${train_args[@]}"
        printf '```\n'
    } >> "$EXPERIMENT_DOC"
}

write_launch_manifest() {
    if [[ "$USE_PRETRAINED_CHECKPOINT" != "1" ]]; then
        return
    fi
    local manifest_args=()
    if [[ "$EXPERIMENT_ARM" == "difference_query_stage3" ]]; then
        manifest_args+=(--experiment "$EXPERIMENT_ARM")
    fi
    "$PYTHON_BIN" "$ROOT_DIR/scripts/record_libero_finetune_launch.py" \
        "${manifest_args[@]}" \
        --run-mode "$mode" \
        --world-size 4 \
        --launcher "$0" \
        --output "$OUTPUT_DIR/launch_manifest_${mode}.json" \
        -- "${train_args[@]}"
}

if [[ "${ZR0_DRY_RUN:-0}" == "1" ]]; then
    validate_explicit_batch_config
    print_command env PYTHONNOUSERSITE=1 "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "${train_args[@]}"
    exit 0
fi

validate_explicit_batch_config
run_preflight
cd "$RUNTIME_ROOT"

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

if [[ "$mode" == "train" || ( "$EXPERIMENT_ARM" == "difference_query_stage3" && ! -e "$OUTPUT_DIR" ) ]]; then
    mkdir -p "$OUTPUT_DIR"
    printf '%s\n' "$WANDB_RUN_ID" > "$WANDB_RUN_ID_FILE"
    printf '%s\n' "$RUN_NAME" > "$WANDB_RUN_NAME_FILE"
fi

write_experiment_record
write_launch_manifest

mkdir -p "$LOG_DIR" "$WANDB_LOCAL_DIR"

print_command env PYTHONNOUSERSITE=1 "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "${train_args[@]}"
exec env PYTHONNOUSERSITE=1 "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "${train_args[@]}"
