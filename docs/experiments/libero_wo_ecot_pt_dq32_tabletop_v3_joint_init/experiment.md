# LIBERO DQ32 微调：Tabletop v3 Joint 初始化

## 1. 基本信息

- 实验名称：`libero_wo_ecot_pt_dq32_tabletop_v3_joint_init`
- 实验目的：使用 Tabletop v3 Difference Query 两阶段预训练的最终 Joint 模型初始化 LIBERO action-only 微调，并与原始 Qwen 初始化的 DQ32 LIBERO 实验保持相同微调配置。
- 创建时间：2026-09-04（Asia/Shanghai）。
- 负责人：lq。
- 预训练来源：`outputs/pretrain/tabletop_v3_dq32_joint_gbs128_seed42_mbs16_gas2/step-19424`。
- 参考实验：`outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm-difference-query-nq32`。
- 代码基准、staged diff、launcher/config/model/data hash：正式启动前写入输出目录的 `launch_manifest_train.json`。
- 配置文件：`accelerate_configs/libero_zero2_bf16_mbs16_gas1.yaml`。
- launcher：`scripts/run_libero_wo_ecot_pt.sh` 的 `difference_query_pretrained` 实验臂。
- 自动串联：`scripts/run_libero_finetune_after_pretrain.sh run` 等待正式预训练和 GPU 门禁后执行 preflight、smoke/resume、正式训练；不终止其他 GPU 进程。
- 随机种子：42。
- 输出目录：`outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm-difference-query-nq32-tabletop-v3-joint-init`。
- W&B entity/project/group：`jumbo3r-zhejiang-university` / `ZR-0-LIBERO` / `libero-wo-ecot-pt-difference-query-tabletop-v3-joint-init`。
- W&B run name、run ID、URL：启动时生成并追加到本文件副本。

本实验不会覆盖参考 checkpoint。当前仓库已有的 staged/unstaged 修改属于先前 Difference Query 和预训练工作；正式 manifest 记录启动时的完整状态，不回退或提交这些修改。

## 2. 初始化与训练阶段

本实验只有一个 LIBERO action-only 微调阶段。初次启动时：

- VLM：从正式 Joint `step-19424` 完整加载并训练，包括视觉编码器、merger、DeepStack 和 language decoder。
- Difference Query：从同一 checkpoint 加载 `[32, 2048]` 权重并训练，attention backend 固定为 SDPA。
- Action Expert：从同一 checkpoint 加载并训练，包括 state encoder、action/timestep encoder、9 层 DiT、decoder 和 positional embedding。
- Action Expert 架构沿用预训练 checkpoint；运行配置把 `action_horizon` 从预训练的 32 解析为 LIBERO 的 10。该字段不改变权重形状。
- fresh checkpoint warm-start 会先严格验证源配置，再仅覆盖运行时 `action_horizon`；action/state/hidden/DiT 等其余结构字段仍须完全一致。LIBERO checkpoint 保存有效 horizon 10，后续 resume 不允许再次跨 horizon。
- 不设置 `--resume_training`，不加载预训练 optimizer、scheduler 或 global step；为微调重新创建 optimizer 和 scheduler。

若微调中断，`resume` 模式只从本实验的 `latest-model-optimizer-lr` 恢复完整微调模型、optimizer、scheduler、global step、数据位置和同一个 W&B run。正式训练不得从 smoke checkpoint 启动。

## 3. 数据集与采样

- 数据集：LeRobot v2.1 `libero_wo_ecot_pt`。
- 路径：`/opt/data/private/lq/datasets/HuggingFaceVLA/libero`。
- 规模：1,693 episodes、273,465 frames、40 tasks；全部作为 train，无 validation/test dataloader。
- 采样：sample ratio 1.0；每 epoch 以 `seed=42+epoch` 确定性 shuffle。
- 四 rank 对齐：每 epoch 呈现 273,468 次，确定性重复 3 帧；每 rank 68,367 个样本。
- 自然尾批：每 rank 最后 15 个样本，最后 optimizer step 的实际 global batch 为 60，不补至 64、不丢弃。
- 相机固定顺序：`observation.images.image`、`observation.images.image2`。
- 语言：使用 task prompt；action-only 不把 VLM CE 纳入总 loss。
- 状态：8 维；动作：7 维；均使用数据集 `stats.json` 的 q01/q99 quantile min-max normalization。

## 4. 图像处理

- 两路原始 RGB 图像均为 256 x 256。
- loader 转为 PIL RGB 后传入 224 x 224；不保持长宽比。
- processor 调用使用 `do_resize=False`，因为尺寸已经由消息中的 `resized_height/resized_width` 固定。
- 插值为 PIL 默认 bicubic；无 crop、pad、letterbox 或数据增强。
- rescale 为 `1/255`，mean/std 均为 `[0.5, 0.5, 0.5]`。
- Qwen3-VL patch size 16、spatial merge 2；每路 49 个视觉 token，总计 98 个。
- 多视角按当前时间、上述 camera 顺序依次进入上下文；`window_size=1`，不使用历史或未来图像。

## 5. 动作、Loss 与优化器

- action/state model dimension：64；有效 action/state dimension：7/8。
- action chunk、prediction horizon：10；episode 尾部无效时间步和 57 个 padding action 维度不参与 loss。
- Flow Matching：`u ~ Beta(1.5, 1.0)`，沿用 ZR-0 当前噪声、速度目标和 mask 实现。
- `loss_type=action`，`action_expert_loss_weight=1.0`；`total_loss = flow_matching_loss`，不计算或反传 AR loss。
- 每个 optimizer step 的 loss 以跨 rank 的真实有效 action element 总数归一化，包括 global batch 60 的尾步。
- AdamW：peak LR `2e-5`、minimum LR `2e-6`、betas `(0.9, 0.95)`、epsilon `1e-6`、weight decay `0.01`。
- scheduler：8% linear warmup 后 cosine decay；总计 34,184 optimizer steps，warmup 2,734 steps。
- 所有可训练参数共用相同 optimizer 超参数，不为梯度诊断拆分 optimizer group。

## 6. 训练规模与保存

```text
global_batch_size = 4 GPUs x micro_batch 16 x GAS 1 = 64
steps_per_epoch = ceil(273465 / 64) = 4273
total_optimizer_steps = 4273 x 8 = 34184
```

- GPU：4 x NVIDIA A800-SXM4-80GB，仅使用设备 0、1、2、3。
- precision：BF16；DeepSpeed ZeRO-2；无 optimizer/parameter offload。
- VLM gradient checkpointing：开启；Action Expert activation checkpointing 未实现。
- gradient clipping：1.0。
- max sequence length：1,200。
- dataloader workers：24。
- 日志：每 10 optimizer steps；stdout、本地 JSONL、TensorBoard 和 checkpoint manifest 保留完整记录，正式训练要求 W&B 在线初始化成功。
- checkpoint：每 2,000 optimizer steps、第 4/8 epoch 和最终 step 保存模型；`latest-model-optimizer-lr` 保存恢复状态。
- validation/early stopping：无。

## 7. 启动、Smoke 与恢复

正式预训练完成后先运行：

```bash
cd /opt/data/private/lq/ZR-0
bash scripts/run_libero_wo_ecot_pt.sh preflight difference_query_pretrained
```

真实 smoke 使用独立的 timestamp 输出目录和 W&B run：初次运行 2 optimizer steps并每步保存，然后从 smoke 的 `latest-model-optimizer-lr` 恢复到第 3 步。必须检查完整 Joint 初始化来源、world size 4、micro-batch 16、GAS 1、global batch 64、SDPA、loss/梯度/hidden/action 有限和 checkpoint 可恢复。smoke 权重不得作为正式训练来源。

全部门禁通过后运行：

```bash
cd /opt/data/private/lq/ZR-0
bash scripts/run_libero_wo_ecot_pt.sh train difference_query_pretrained
```

中断恢复：

```bash
cd /opt/data/private/lq/ZR-0
bash scripts/run_libero_wo_ecot_pt.sh resume difference_query_pretrained
```

正式进程放入独立持久会话。确认初始 optimizer step、W&B、TensorBoard、本地日志和显存正常后允许其自行持续训练，不进行持续监督。

## 8. 启动门禁与完成后记录

- `step-19424` 必须声明 `checkpoint_kind=joint`，且具备完整 VLM shards、DQ32 config/weights、Action Expert config/weights。
- Query 必须为 enabled、Nq=32、hidden size 2048、SDPA；Action Expert 除 horizon 外必须与参考 LIBERO `step-34184` 架构一致。
- 初始化 manifest 和 stdout 必须同时记录源 horizon 32、有效 horizon 10、override 状态、源配置文件 SHA256 和有效配置 canonical SHA256。
- 正式训练前检查 GPU 计算进程、每卡至少 70 GiB 空闲、cgroup headroom 至少 120 GiB、输出盘至少 200 GiB 空闲和 W&B 认证。
- 任一门禁或真实 smoke 失败时停止并报告，不改 LR、batch、loss、视角、步数或 checkpoint 来源绕过失败。
- 完成或中断后补充实际起止时间、完成步数、最终 checkpoint、最低 train loss、W&B URL、NaN/OOM/中断、resume 情况、配置偏差和日志位置。
- LIBERO rollout/evaluation 不属于本次微调启动范围，需另行授权和记录。

## 9. 自动串联排队记录

- 排队时间：2026-09-04T21:32:25+08:00。
- 排队时 Joint 预训练进度：`14580/19424` optimizer steps。
- tmux session：`zr0-libero-dq32-after-pretrain-20260904`；启动验收时 `pane_dead=0`。
- pipeline 日志：`outputs/full_logs/libero-dq32-tabletop-v3-joint-init/pipeline-20260904.log`。
- 当前状态：等待完整正式 `step-19424`；等待阶段不创建微调模型、不占用额外 GPU。
- 后续自动顺序：GPU 门禁 -> W&B/数据/checkpoint preflight -> 定向测试 -> 隔离 2-step smoke -> smoke resume 至 step 3 -> 有限性与恢复断言 -> 全新正式训练。
- 任一门禁失败时 pipeline 以非零状态停止并保留日志，不会降低配置、终止其他进程或启动正式训练。
