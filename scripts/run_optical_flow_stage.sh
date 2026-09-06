#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: bash scripts/run_optical_flow_stage.sh stage2_aux|stage3_joint [--print-command]}"
: "${INIT_CHECKPOINT:?Set the explicit stage initialization checkpoint}"
: "${OUTPUT_DIR:?Set the new experiment output directory}"
: "${NUM_FLOW_QUERIES:?Set the explicit number of trailing flow queries}"
: "${OPTICAL_FLOW_LOSS_WEIGHT:?Set a positive optical flow loss weight}"
: "${WANDB_PROJECT:?Set the W&B project}"
: "${WANDB_RUN_NAME:?Set the W&B run name}"
: "${ACCELERATE_CONFIG:?Set the existing Accelerate/DeepSpeed configuration}"

case "$stage" in
  stage2_aux) stage_flags=() ;;
  stage3_joint)
    : "${AR_DATASET_ENTRY:?Stage3 requires an additional dataset providing AR and FM labels}"
    stage_flags=(--tune_action_expert)
    ;;
  *) exit 2 ;;
esac
datasets=(stage06_libero_flow)
if [[ "$stage" == stage3_joint ]]; then
  datasets+=("$AR_DATASET_ENTRY")
fi
command=(accelerate launch --config_file "$ACCELERATE_CONFIG" train_vla.py
  --training_stage "$stage" --init_from_checkpoint "$INIT_CHECKPOINT"
  --optical_flow_aux_type dense_regression_v1 --num_flow_queries "$NUM_FLOW_QUERIES"
  --optical_flow_loss_weight "$OPTICAL_FLOW_LOSS_WEIGHT" --slot_aux_type none
  --optical_flow_data_root /opt/data/private/lq/datasets/lerobot/libero/stage06_flow/libero_delta10
  --optical_flow_manifest manifest.2849ed69240ad542.jsonl
  --dataset_entries "${datasets[@]}" --tune_vlm "${stage_flags[@]}"
  --action_horizon "${ACTION_HORIZON:-32}" --window_size 1 --seed 42
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE:-4}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --peak_learning_rate "${PEAK_LEARNING_RATE:-1e-5}"
  --epochs "${EPOCHS:-1}" --save_optimizer_and_lr_states
  --output_ckpt_dir "$OUTPUT_DIR" --tensorboard_log_dir "$OUTPUT_DIR/tensorboard"
  --wandb_project "$WANDB_PROJECT" --wandb_group "${WANDB_GROUP:-optical-flow-provisional}"
  --wandb_run_name "$WANDB_RUN_NAME" --wandb_failure_policy required)
if [[ "${2:-}" == --print-command ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
test -f "$OUTPUT_DIR/experiment.md" || { printf '%s\n' 'Create the complete experiment.md before launching.' >&2; exit 2; }
exec "${command[@]}"
