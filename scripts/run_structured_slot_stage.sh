#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: run_structured_slot_stage.sh stage2_aux|stage3_joint [--print-command]}"
: "${INIT_CHECKPOINT:?Set the source checkpoint}"
: "${OUTPUT_DIR:?Set the experiment output directory}"
: "${ACCELERATE_CONFIG:?Set an existing non-ZeRO or ZeRO-2 config}"
: "${WANDB_PROJECT:?Set the W&B project}"
: "${WANDB_RUN_NAME:?Set the W&B run name}"
datasets=(stage05_droid_partial_mixed stage05_household_mixed stage05_tabletop_mixed stage05_rh20t_mixed)
flags=()
if [[ "${WITH_SLOT:-1}" == 1 ]]; then
  : "${SLOT_SUPERVISION_DIR:?Set the completed Slot audit directory}"
  : "${SLOT_LOSS_WEIGHT:?Set a positive Slot loss weight}"
  flags+=(--slot_aux_type structured_slots_v1 --slot_loss_weight "$SLOT_LOSS_WEIGHT"
    --slot_supervision_dir "$SLOT_SUPERVISION_DIR")
else
  flags+=(--slot_aux_type none --slot_loss_weight 0)
fi
case "$stage" in
  stage2_aux) ;;
  stage3_joint) flags+=(--tune_vlm --tune_action_expert) ;;
  *) exit 2 ;;
esac
if [[ "${WITH_FLOW:-0}" == 1 ]]; then
  : "${OPTICAL_FLOW_LOSS_WEIGHT:?Set a positive Flow loss weight}"
  flags+=(--optical_flow_data_root /opt/data/private/lq/datasets)
  flags+=(--optical_flow_aux_type dense_regression_v1
    --optical_flow_loss_weight "$OPTICAL_FLOW_LOSS_WEIGHT"
    --flow_delta_frames 20
    --stage2_aux_sampling any_aux_valid)
else
  flags+=(--optical_flow_aux_type none --optical_flow_loss_weight 0)
fi
command=("${ACCELERATE_BIN:-accelerate}" launch --config_file "$ACCELERATE_CONFIG" train_vla.py
  --training_stage "$stage" --init_from_checkpoint "$INIT_CHECKPOINT"
  --aux_dataset_config "${AUX_DATASET_CONFIG:-configs/aux_four_dataset_v1.json}"
  --use_difference_query --num_difference_queries 32 --num_flow_queries 8
  --dataset_entries "${datasets[@]}" "${flags[@]}"
  --action_horizon 32 --window_size 1 --seed 42
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE:-1}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --peak_learning_rate "${PEAK_LEARNING_RATE:-1e-5}" --epochs "${EPOCHS:-1}"
  --save_optimizer_and_lr_states --output_ckpt_dir "$OUTPUT_DIR"
  --tensorboard_log_dir "$OUTPUT_DIR/tensorboard"
  --wandb_project "$WANDB_PROJECT" --wandb_group "${WANDB_GROUP:-structured-slots-v1}"
  --wandb_run_name "$WANDB_RUN_NAME" --wandb_failure_policy required)
if [[ "${2:-}" == --print-command ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
test -f "$OUTPUT_DIR/experiment.md" || { printf '%s\n' 'Create experiment.md before launching.' >&2; exit 2; }
exec "${command[@]}"
