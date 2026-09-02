# ZR-0 Difference Query LIBERO 三臂实验说明

## 1. 基本信息与状态

- 实验名称：`baseline_fa2` / `baseline_sdpa` / `difference_query`
- 实验目的：检验动作专家只读取固定 32 个 learnable Difference Query 后的 LIBERO 成功率变化，并用无 Query+SDPA 控制组分离 attention backend 影响。
- 创建时间：2026-09-01（Asia/Shanghai）
- 负责人：lq
- 基座模型：`/opt/data/private/lq/models/Qwen3-VL-2B-Instruct`
- 基座模型：Qwen3-VL-2B-Instruct，VLM 从本地预训练 checkpoint 加载；Action Expert 每臂按 seed 42 随机初始化。
- 初始化 checkpoint：VLM 为上述目录；fresh train 不加载 `action_expert.safetensors`。
- 代码基准 commit：`166193d7d50c18080f954d8771b8a8a6b6927bb3`
- 未提交/暂存修改：Difference Query 实现、审核修复、测试、本文件、`reference.md` 与架构分析均属于当前任务；工作树另有用户已有的 `simple_scripts/eval_libero.md`、`result/eval/...` 评估产物、`代码修改1.md` 和 `代码修改2.md`，不属于本实验实现。
- 配置文件：`accelerate_configs/accelerate_config.yaml`、`dataset2feature.yaml`、`scripts/run_libero_wo_ecot_pt.sh`
- 随机种子：训练/server 42；LIBERO client 7。
- 当前状态：`difference_query` 臂已于 2026-09-02 13:44:11 +08:00 启动正式训练，W&B run `46caif1b` 正在记录；LIBERO rollout 未执行。

## 2. 三实验臂

| 实验臂 | Difference Query | Qwen backend | Nq | 输出目录 | W&B group | 结果 |
|---|---|---|---:|---|---|---|
| `baseline_fa2` | 关闭 | 自动 FA2；不可用时 eager | - | `outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm` | `libero-wo-ecot-pt` | 历史 NoECoT 基线已完成（run `jc8exz8p`） |
| `baseline_sdpa` | 关闭 | 显式 SDPA | - | `outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm-baseline-sdpa` | `libero-wo-ecot-pt-baseline-sdpa` | 未执行 |
| `difference_query` | 开启，动作专家只读 Query | 强制 SDPA | 32 | `outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm-difference-query-nq32` | `libero-wo-ecot-pt-difference-query` | 正在训练（run `46caif1b`） |

三臂 W&B project 均为 `ZR-0-LIBERO`；run name 默认由 launcher 生成，也可用 `ZR0_RUN_NAME` 固定；run ID 自动生成或由 `ZR0_WANDB_RUN_ID` 指定。当前 Difference Query run URL 为 `https://wandb.ai/jumbo3r-zhejiang-university/ZR-0-LIBERO/runs/46caif1b`。Query+SDPA 相对 FA2 baseline 同时改变压缩和 backend，结论必须同时对比 `baseline_sdpa`。

正式 `train`/`resume` 通过 preflight 后、启动进程前，launcher 会把本文件复制到对应实验臂的 `OUTPUT_DIR/experiment.md`（已有文件不覆盖），并追加实际启动时间、模式、实验臂、输出目录、W&B project/group/run name/run ID 和完整命令。W&B run URL 仍需在 `wandb.init` 成功后补充；dry-run 不创建实验目录或说明文件。

## 3. 完整启动命令

正式训练前先登录 W&B 并分别运行 preflight。不得在 W&B 不可连接时切换 offline/disabled 模式。

```bash
cd /opt/data/private/lq/ZR-0
bash scripts/run_libero_wo_ecot_pt.sh preflight baseline_fa2
bash scripts/run_libero_wo_ecot_pt.sh train baseline_fa2

bash scripts/run_libero_wo_ecot_pt.sh preflight baseline_sdpa
bash scripts/run_libero_wo_ecot_pt.sh train baseline_sdpa

bash scripts/run_libero_wo_ecot_pt.sh preflight difference_query
bash scripts/run_libero_wo_ecot_pt.sh train difference_query
```

恢复训练时使用相同实验臂；launcher 从该臂的 `latest-model-optimizer-lr` 同时加载 VLM、Action Expert、Difference Query（若有）、DeepSpeed optimizer 和 scheduler：

```bash
bash scripts/run_libero_wo_ecot_pt.sh resume baseline_fa2
bash scripts/run_libero_wo_ecot_pt.sh resume baseline_sdpa
bash scripts/run_libero_wo_ecot_pt.sh resume difference_query
```

## 4. 数据集与采样

- 数据集：`libero_wo_ecot_pt`，本地版本 `/opt/data/private/lq/datasets/HuggingFaceVLA/libero`。
- 规模：1,693 episodes，273,465 frames，40 tasks；sample ratio 1.0。
- 划分：launcher 只使用训练集合；没有独立 validation/test dataloader。正式 test 是四个 LIBERO suite rollout。
- 采样：每 epoch 使用 seed `42+epoch` 的随机 subset 顺序；8 epochs。边界 action index clamp 到 episode 最后帧。
- 相机：训练使用 `observation.images.image`、`observation.images.image2`；评估 payload 使用 agentview 与 wrist image，对应 server dataset metadata。
- 标签：使用 state、语言、连续 action 和为 VLM 构造的 FAST 离散动作文本；不使用光流。当前 action-only 总 loss 不反传 VLM CE。
- state：8 个有效维，经 q01/q99 quantile min-max normalization 后补零至 64。
- action：7 个有效维，经相同 q01/q99 normalization 后补零至 64；末端执行器没有独立离散/阈值代码，7 维整体直接交给 LIBERO。

## 5. 训练阶段和可训练模块

每个实验只有一个 action-only 阶段，三臂执行顺序互相独立，不串联 checkpoint。

| 模块 | `baseline_fa2` | `baseline_sdpa` | `difference_query` |
|---|---|---|---|
| Qwen3-VL 视觉编码器、merger、DeepStack | 全量训练 | 全量训练 | 全量训练 |
| Qwen3-VL language decoder（不含 LM head） | 由 action loss 全量训练 | 同左 | 同左 |
| Qwen conditional LM head | 参数未冻结；兼容 HEAD forward 会调用并生成 logits/CE，但 CE 不进总 loss，LM head 无梯度 | 同左 | action-only 不调用、无梯度 |
| state encoder | 随机初始化并训练 | 相同 seed 随机初始化并训练 | 相同 seed 随机初始化并训练 |
| action+timestep encoder | 随机初始化并训练 | 同左 | 同左 |
| 9 层 DiT Action Expert | 随机初始化并训练 | 同左 | 同左 |
| action decoder / action position embedding | 随机初始化并训练 | 同左 | 同左 |
| Difference Query | 不创建 | 不创建 | `[32,2048]`，隔离 RNG 初始化并训练 |

不使用 LoRA，不冻结视觉塔，不启用 `detach_vlm_outputs_for_action_expert`。Query 初始化不推进全局 RNG，因此三臂随后创建的随机 Action Expert 初始权重一致。

## 6. 优化器、scheduler 与训练规模

所有可训练参数共用一个参数组：AdamW，peak LR `2e-5`，minimum LR `2e-6`（min rate 0.1），betas `(0.9,0.95)`，eps `1e-6`，weight decay `0.01`。scheduler 为 8% warmup 后 cosine decay；warmup 为 2,734 global steps，计划 34,184 global optimizer steps。不同模块没有 LR multiplier。

- GPU：4 x NVIDIA A800-SXM4-80GB。
- per-device batch size：16。
- gradient accumulation：1。
- `global_batch_size = 4 x 16 x 1 = 64`。
- epochs：8；计划处理 `273,465 x 8 = 2,187,720` 条样本（最后不完整 batch 保留）。
- mixed precision：bf16。
- DeepSpeed：ZeRO stage 2，无 optimizer/parameter offload；gradient clipping 1.0。
- VLM gradient checkpointing：开启；Action Expert gradient checkpointing 实际未实现。
- validation：训练期间无 validation loop、无 early stopping。
- log：训练 scalar 每 10 global steps；W&B 必须在线。
- checkpoint：每 2,000 steps；每 4 epochs；同时保存 optimizer/scheduler 状态。

## 7. Loss 设计

三臂均为 `loss_type=action`：

```text
u ~ Beta(1.5, 1.0)
t = (0.999 - u) / 0.999
epsilon ~ N(0, I)
x_t = ((1-t) epsilon + t action) * action_mask
target_velocity = (action - epsilon) * action_mask

total_loss = flow_matching_loss
flow_matching_loss =
    sum(((pred_velocity * action_mask) - target_velocity)^2)
    / max(sum(action_mask), 1)
```

`action_expert_loss_weight=1.0` 在 action-only 分支不另做缩放。`baseline_fa2` 和 `baseline_sdpa` 为复现历史 HEAD 行为，均调用 `Qwen3VLForConditionalGeneration.forward`，保留二维 mask、labels、hidden-state 输出和训练/推理 cache 参数；因此 action-only 有 labels 时仍计算随后不进入总 loss 的 LM logits/CE。`baseline_sdpa` 只改变 attention backend。`difference_query` 臂保留 labels 供 C/T 重排，但 action-only 调用 Qwen 原生 multimodal base，不执行 conditional LM head、不生成全词表 logits，也不计算 CE；`vlm` / `vlm_and_action` 才走 teacher-forced LM head 和 CE。三臂的 final normalized hidden 都参与 action loss 反向，Difference Query 没有独立 loss，只在 `difference_query` 臂通过 action loss 训练。关闭 Query 时它不进入计算图且无参数。

三臂的成功率比较继续固定相同数据顺序、batch、训练步数、optimizer、scheduler、horizon 和评估参数。由于 Query-off 基线保留 HEAD logits/CE 计算，而 Query-on action-only 使用高效 base-model 路径，三臂运行时间和显存存在额外调用层级差异，不能把这些差异直接解释为 Difference Query 本身带来的加速。

## 8. 图像处理

训练：原始两视角 256x256 RGB；loader 转 PIL RGB，并指定 224x224；processor `do_resize=False`，最终 grid 为每图 14x14 patch、2x2 merger 后 49 个 VLM token。无 crop/pad/augmentation；rescale 1/255，mean/std 均 `[0.5,0.5,0.5]`。多视角按时间优先、相机次优先依次插入 token 序列，当前 `window_size=1`。

评估：LIBERO render 256x256，两视角先旋转 180 度，再以 PIL bilinear resize/pad 到 448x448；server loader 再处理到实际模型输入 224x224，无 crop，使用相同 mean/std。训练和评估的最终模型尺寸一致，但评估多一次 256->448 bilinear 中间 resize；该差异必须保留并在结果中注明。

## 9. 动作与推理配置

- 模型 action dimension/state dimension：64；LIBERO 有效 action/state 分别为 7/8。
- action chunk / action horizon / prediction horizon：10。
- execution horizon / replan steps：10；每次预测后执行 10 步再重规划。
- 数据 FPS / LIBERO 控制频率：10 Hz 数据时间偏移；评估环境 20 Hz（10 步约 0.5 秒模拟控制时间）。
- direct-action 去噪：5 步显式 Euler；flow time 为 0、0.2、0.4、0.6、0.8。
- action padding：有效前 7 维 mask=True，后 57 维 False；噪声、velocity、loss、Euler 每步均应用 mask。
- state padding：有效前 8 维，后 56 维补零并由 `state_mask` 标记。
- Query 模式不支持 subtask/autoregressive generation，正式评估必须使用 `--inference_mode direct_action`。

## 10. Server 与 LIBERO 四套件命令

训练完成后把三个 checkpoint 路径分别设为实际最终 step。下面展示每臂的 server 命令；每次只启动当前要评估的一个 checkpoint，端口可按资源调整但 client 必须一致。

```bash
cd /opt/data/private/lq/ZR-0
SERVER_PY=/opt/data/private/lq/.conda/envs/zr0-eval/bin/python

# baseline_fa2
CUDA_VISIBLE_DEVICES=0 "$SERVER_PY" -u server.py \
  --dataset_entry demo_data.libero_v21 \
  --ckpt_dir outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm/<final-step> \
  --inference_mode direct_action --port 8100

# baseline_sdpa
CUDA_VISIBLE_DEVICES=0 "$SERVER_PY" -u server.py \
  --dataset_entry demo_data.libero_v21 \
  --ckpt_dir outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm-baseline-sdpa/<final-step> \
  --inference_mode direct_action --vlm_attention_backend sdpa --port 8100

# difference_query
CUDA_VISIBLE_DEVICES=0 "$SERVER_PY" -u server.py \
  --dataset_entry demo_data.libero_v21 \
  --ckpt_dir outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm-difference-query-nq32/<final-step> \
  --inference_mode direct_action --use_difference_query \
  --num_difference_queries 32 --vlm_attention_backend sdpa --port 8100
```

对每个 checkpoint 分别运行相同四套件。client seed=7，每任务 50 rollouts，每 suite 500 episodes；成功由环境 `done=True` 判定，suite success rate 为成功 episode/500。

```bash
CLIENT_PY=/opt/data/private/lq/.conda/envs/zr0-libero-eval/bin/python
export LIBERO_CONFIG_PATH=/opt/data/private/lq/ZR-0/result/eval/ZR-0-LIBERO_official_seed7_50trials_20260830_195202/env/libero-config
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

"$CLIENT_PY" -u -m evaluation.libero_eval.run_libero_eval \
  --args.task-suite-name libero_spatial --args.port 8100 \
  --args.video-out-path <result-dir>/videos/libero_spatial
"$CLIENT_PY" -u -m evaluation.libero_eval.run_libero_eval \
  --args.task-suite-name libero_object --args.port 8100 \
  --args.video-out-path <result-dir>/videos/libero_object
"$CLIENT_PY" -u -m evaluation.libero_eval.run_libero_eval \
  --args.task-suite-name libero_goal --args.port 8100 \
  --args.video-out-path <result-dir>/videos/libero_goal
"$CLIENT_PY" -u -m evaluation.libero_eval.run_libero_eval \
  --args.task-suite-name libero_10 --args.port 8100 \
  --args.video-out-path <result-dir>/videos/libero_10
```

每个结果目录必须补充实际 checkpoint、训练实验臂、client/server 完整命令、seed、suite、rollout 数、图像链、H=10、execution horizon=10、20 Hz、成功率口径和日志/视频路径。

## 11. 完成后补充项

当前已有实际开始时间和 W&B run ID/URL；结束时间、完成 steps、最终 checkpoint、最低 train loss、最佳验证指标、中断/resume 记录、计划偏差、四 suite 与总成功率仍待训练或评估完成后填充。

官方 checkpoint 上一次 2,000 episodes 的 97.6% 只作为历史参考，不是上述任何实验臂的结果。三臂必须各自训练和评估后才能比较。

## 12. Difference Query 正式训练运行记录

- 实际启动时间：2026-09-02T13:44:11+08:00。
- 运行状态：正在训练；初期已观察至 optimizer step 150，无 OOM、NaN 或 Inf。
- 持久会话：`tmux` session `zr0-dq-nq32-20260902-134333`，pane PID `3438613`。
- W&B：project `ZR-0-LIBERO`，group `libero-wo-ecot-pt-difference-query`，run name `qwen3vl2b-libero-wo-ecot-difference-query-nq32-full-20260902-134333`，run ID `46caif1b`，URL `https://wandb.ai/jumbo3r-zhejiang-university/ZR-0-LIBERO/runs/46caif1b`。
- 输出目录：`/opt/data/private/lq/ZR-0/outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm-difference-query-nq32`。
- TensorBoard：`/opt/data/private/lq/ZR-0/outputs/train_logs/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm-difference-query-nq32/qwen3vl2b-libero-wo-ecot-difference-query-nq32-full-20260902-134333`。
- 完整 stdout/stderr：`/opt/data/private/lq/ZR-0/outputs/full_logs/qwen3vl2b-libero-wo-ecot-difference-query-nq32-full-20260902-134333.log`。
- 初期观察：GPU 0 的 step 10/20/30/40 loss 分别为 1.3359/1.2969/1.2109/1.3203；step 150 loss=1.2500，LR=1.0991e-6，global grad norm=3.3292。四个 rank 均有有限 loss 记录。
- 初期资源：GPU 0/1/2/3 占用约 27.0/25.5/25.1/24.8 GiB；无 OOM。
- 预计完成：step 10–150 观测节拍为 1.115 s/step，估算剩余约 10.5 h，以 W&B 实际进度为准。
- 完整启动命令：

```bash
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --config_file /opt/data/private/lq/ZR-0/accelerate_configs/accelerate_config.yaml /opt/data/private/lq/ZR-0/train_vla.py --vlm_name_or_path /opt/data/private/lq/models/Qwen3-VL-2B-Instruct --FAST_tokenizer_path /opt/data/private/lq/ZR-0/fast --per_device_train_batch_size 16 --seed 42 --epochs 8 --save_ckpt_interval 4 --save_step_interval 2000 --peak_learning_rate 2e-5 --min_lr_rate 0.1 --tensorboard_log_dir /opt/data/private/lq/ZR-0/outputs/train_logs/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm-difference-query-nq32/qwen3vl2b-libero-wo-ecot-difference-query-nq32-full-20260902-134333 --output_ckpt_dir /opt/data/private/lq/ZR-0/outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm-difference-query-nq32 --tune_vlm --tune_action_expert --loss_type action --action_expert_loss_weight 1.0 --lr_scheduler cosine --dataset_entries libero_wo_ecot_pt --window_size 1 --action_horizon 10 --max_pad_state_and_action_length 64 --save_optimizer_and_lr_states --wandb_project ZR-0-LIBERO --wandb_run_name qwen3vl2b-libero-wo-ecot-difference-query-nq32-full-20260902-134333 --wandb_run_id 46caif1b --wandb_resume never --wandb_dir /opt/data/private/lq/ZR-0/outputs/wandb --wandb_group libero-wo-ecot-pt-difference-query --wandb_tags ablation wo-ecot-pt libero-v21 qwen3-vl-2b difference_query --use_difference_query --num_difference_queries 32 --vlm_attention_backend sdpa
```
