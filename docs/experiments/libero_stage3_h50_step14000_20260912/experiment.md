# LIBERO fine-tuning from Stage 3 H50 step 14000 — completed

- Created: 2026-09-12, Asia/Shanghai; owner: lq. User authorized stopping the active H50 Stage 3 and fine-tuning its saved step-14000 model with the previous LIBERO configuration, horizon 10.
- Source: `/opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260911/stage3_from_stage2_step5000_h50_slot0p1_flow1/recovery_checkpoints/stage3_joint/step-014000-attempt-000/latest-model-optimizer-lr`. Base ancestry: `/opt/data/private/lq/models/ZR-0`.
- Runtime: existing `outputs/runtime_snapshots/3eefb602417d3bd4b20bef2b47660b404aefb565`; existing launcher/wrapper are reused. Working-tree unrelated Flow V2 modifications are not imported. Exact code status, staged/unstaged diff hashes, source weights and complete parsed options are in `/opt/data/private/lq/ZR-0/outputs/ckpts/ZR0-stage3-step14000-h50-LIBERO-action-only-dq32-h10-gbs64-seed42/launch_manifest_train.json`.
- Output: `/opt/data/private/lq/ZR-0/outputs/ckpts/ZR0-stage3-step14000-h50-LIBERO-action-only-dq32-h10-gbs64-seed42`. Configuration: `accelerate_configs/libero_zero2_bf16_mbs16_gas1.yaml`.
- W&B online required: project `ZR-0-LIBERO`, group `libero-stage3-step14000-action-only-dq32-seed42`, run `zr0-stage3-step14000-h50-libero-dq32-h10-20260912`, ID `aok9gry8`; URL: https://wandb.ai/jumbo3r-zhejiang-university/ZR-0-LIBERO/runs/aok9gry8.
- Initialization is downstream fine-tuning, not optimizer resume. Inherit complete VLM (vision encoder, merger/projector, DeepStack and language decoder), Query32 and Action Expert from this checkpoint; train them all, no frozen modules, LoRA or detached conditioning. Slot/Optical Flow Heads are absent. Optimizer/scheduler/global step reset.
- Source horizon 50 becomes runtime horizon 10 using the existing override; no weights are resized or randomly replaced. Before the first update, VLM 626 / Query 1 / Expert 149 tensors matched the source exactly after dtype conversion; 2,690,580,544 trainable parameters, fresh counters 0/0. Evidence: `libero_initialization_verified.json`.

## Data, images and action contract

- `libero_wo_ecot_pt`, LeRobot v2.1, `/opt/data/private/lq/datasets/HuggingFaceVLA/libero`: 1,693 episodes, 273,465 frames, 40 tasks, 10 FPS. All supplied frames train; no validation/test loader. Sample ratio 1, seed 42 plus epoch shuffle. Each epoch presents 273,468 samples including 3 distributed duplicates; 8 epochs present 2,187,744 including 24 duplicates.
- Input: current task text, current state, RGB `observation.images.image` then `observation.images.image2`; no ECoT, future images or auxiliary labels. Window 1, text limit 1200. Two 256×256 views directly resized to 224×224 bicubic; no crop, letterbox or augmentation; rescale 1/255, mean/std both [0.5,0.5,0.5].
- LIBERO state/action effective dimensions 8/7, padded to 64 with validity masks and episode-tail masking. Existing end-effector delta position, rotation and gripper representation. Prediction/action chunk horizon 10; evaluation execution horizon and rollout settings are not established by this training.
- LIBERO's own q01/q99 training statistics, existing quantile normalization, clipping [-15,15]. Dataset manifest SHA256 `a3c74b5acfdf414effc33c4f832dee534791bd11f51dba61ef950adce9f3cd54`; statistics SHA256 `aafe658c89aad69f2b6db5a7109d447ff72040e7ab13c1f0938a2e6237e5942d`. No pretraining data audit or statistics recomputation.

## Objective, optimizer and budget

- `L = 1.0 * L_FM`, existing masked action velocity regression. AR/Slot/Optical Flow coefficients all 0 and not computed. All 32 Queries condition the Expert; FM gradients reach inherited VLM/Query/Expert.
- 4 A800-SXM4-80GB, nominal global batch = 4 GPUs × micro-batch 16 × GAS 1 = 64. Last global batch in each epoch is 60. BF16, ZeRO-2, no offload, SDPA, VLM gradient checkpointing.
- One AdamW parameter group, all modules peak LR 2e-5 / minimum 2e-6, betas (0.9,0.95), epsilon 1e-6, weight decay 0.01, clip 1.0; no LR multipliers. Cosine, warmup 8% = 2,734 global updates. Existing Accelerate scheduler advances four internal ticks/global update.
- Seed 42, 8 epochs / 34,184 updates. Workers 24/rank, prefetch 3. Logging every 10 updates; save every 2,000 updates, epochs 4/8 and final. No validation, early stopping, automatic rollout or successor training. This run used the direct launcher in tmux, not the historical dedicated retry supervisor.

## Completion and verification

- Launch record: 2026-09-12 08:44:35 +08:00. Final complete archive: 17:25:36; W&B finish record: 17:25:44. Completed 34,184/34,184 updates and 8/8 epochs, approximately 8 h 41 min including startup and saving. No same-stage resume occurred.
- Final FM/total loss `0.008453371934592724`; minimum logged loss `0.007475606165826321`; last 1,000 updates' logged mean `0.024837008603654877`. These are training losses, not validation success rates.
- Final gradient norms: VLM 0.145889, Query 0.00148414, Expert 0.155675. Logged peak allocated/reserved GPU memory 18.1028/23.1406 GiB. No nonfinite loss/gradient or skipped update in recorded metrics, and no fatal exception/OOM in the launch log.
- W&B finished and synced; failures, dropped/pending payloads and finish timeouts are zero. At inspection 17:58:45 no matching trainer/rank process remained and all GPUs were at 2 MiB / 0% utilization. No new training was started.
- Final model for inference/evaluation: `/opt/data/private/lq/ZR-0/outputs/ckpts/ZR0-stage3-step14000-h50-LIBERO-action-only-dq32-h10-gbs64-seed42/step-34184`.
- Immutable full optimizer/scheduler checkpoint: `/opt/data/private/lq/ZR-0/outputs/ckpts/ZR0-stage3-step14000-h50-LIBERO-action-only-dq32-h10-gbs64-seed42/recovery_checkpoints/step-034184/latest-model-optimizer-lr`. All 26 files' sizes/mtimes match the completion receipt. This check did not reload optimizer tensors or execute a resume diagnostic.
- Original pretraining checkpoint and previous experiments are preserved. No LIBERO rollout evaluation or success rate is available for this run. No new training code was changed for this result record.

## Runtime launch record

- 时间：2026-09-12T08:44:35+08:00
- 模式：`train`
- 实验臂：`difference_query_stage3`
- 输出目录：`/opt/data/private/lq/ZR-0/outputs/ckpts/ZR0-stage3-step14000-h50-LIBERO-action-only-dq32-h10-gbs64-seed42`
- W&B project/group/run name/run ID：`ZR-0-LIBERO` / `libero-stage3-step14000-action-only-dq32-seed42` / `zr0-stage3-step14000-h50-libero-dq32-h10-20260912` / `aok9gry8`
- W&B run URL：https://wandb.ai/jumbo3r-zhejiang-university/ZR-0-LIBERO/runs/aok9gry8
- 完整启动命令：

```bash
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0\,1\,2\,3 accelerate launch --num_processes 4 --config_file /opt/data/private/lq/ZR-0/accelerate_configs/libero_zero2_bf16_mbs16_gas1.yaml /opt/data/private/lq/ZR-0/scripts/train_libero_finetune.py --vlm_name_or_path /opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260911/stage3_from_stage2_step5000_h50_slot0p1_flow1/recovery_checkpoints/stage3_joint/step-014000-attempt-000/latest-model-optimizer-lr --FAST_tokenizer_path /opt/data/private/lq/ZR-0/fast --per_device_train_batch_size 16 --seed 42 --epochs 8 --save_ckpt_interval 4 --save_step_interval 2000 --peak_learning_rate 2e-5 --min_lr_rate 0.1 --tensorboard_log_dir /opt/data/private/lq/ZR-0/outputs/train_logs/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-stage3-step14000-action-only-dq32/zr0-stage3-step14000-h50-libero-dq32-h10-20260912 --output_ckpt_dir /opt/data/private/lq/ZR-0/outputs/ckpts/ZR0-stage3-step14000-h50-LIBERO-action-only-dq32-h10-gbs64-seed42 --tune_vlm --tune_action_expert --loss_type action --action_expert_loss_weight 1.0 --lr_scheduler cosine --dataset_entries libero_wo_ecot_pt --window_size 1 --action_horizon 10 --max_pad_state_and_action_length 64 --save_optimizer_and_lr_states --wandb_project ZR-0-LIBERO --wandb_run_name zr0-stage3-step14000-h50-libero-dq32-h10-20260912 --wandb_run_id aok9gry8 --wandb_resume never --wandb_dir /opt/data/private/lq/ZR-0/outputs/wandb --wandb_group libero-stage3-step14000-action-only-dq32-seed42 --wandb_tags ablation wo-ecot-pt libero-v21 qwen3-vl-2b difference_query_stage3 --checkpoint_load_purpose downstream_finetune --action_expert_config_path /opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260911/stage3_from_stage2_step5000_h50_slot0p1_flow1/recovery_checkpoints/stage3_joint/step-014000-attempt-000/latest-model-optimizer-lr/action_expert_config.json --vlm_loss_weight 0.0 --slot_loss_weight 0.0 --optical_flow_loss_weight 0.0 --gradient_accumulation_steps 1 --expected_global_batch_size 64 --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-6 --warmup_ratio 0.08 --max_length 1200 --dataloader_num_workers 24 --logging_steps 10 --log_training_diagnostics --wandb_failure_policy required --use_difference_query --num_difference_queries 32 --vlm_attention_backend sdpa --action_expert_name_or_path /opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260911/stage3_from_stage2_step5000_h50_slot0p1_flow1/recovery_checkpoints/stage3_joint/step-014000-attempt-000/latest-model-optimizer-lr
```
