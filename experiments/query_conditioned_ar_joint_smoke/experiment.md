# Query-Conditioned AR / Joint Resume Smoke

## 基本信息

- 实验名称：`query_conditioned_ar_joint_smoke`
- 实验目的：验证 Difference Query 的 AR-only 一步训练与恢复，以及从 AR checkpoint 仅 warm-start VLM/Query 后随机初始化 Action Expert 的 joint 一步训练与恢复。
- 创建时间：2026-09-02（Asia/Shanghai）。
- 负责人：lq；实现与记录协作：Codex。
- 基座模型：`/opt/data/private/lq/models/Qwen3-VL-2B-Instruct`，本地 revision 标识 `master`，`model.safetensors` SHA256 `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`。
- 初始化 checkpoint：AR step 1 使用基座模型；AR resume 使用 `ar/latest-model-optimizer-lr`；joint step 1 的 VLM/Query 使用 AR checkpoint，Action Expert 不从该 checkpoint 加载；joint resume 使用 `joint/latest-model-optimizer-lr` 的完整模型和训练状态。
- 代码基准 commit：`d3b8f943477a52a2836543219dab25ff6bd07941`。
- 当前未提交代码：本任务的 Query AR/joint 模型、训练、adapter、测试、启动器和文档修改尚未提交；用户原有 `代码修改2.md` 修改不属于本实验，也不得加入本任务暂存区。
- 配置入口：`scripts/run_query_ar_joint_smoke.sh`；Accelerate 配置默认 `accelerate_configs/accelerate_config.yaml`，可由 `ZR0_ACCELERATE_CONFIG` 覆盖。
- 随机种子：42。
- 输出目录：外部参数 `ZR0_SMOKE_OUTPUT_ROOT` 下的 `ar/` 与 `joint/`。
- W&B：真实 smoke 也以 online 模式记录到 entity `jumbo3r-zhejiang-university`、project `ZR-0-Pretraining`；最终 micro 16/GAS 2 的 group 为 `tabletop-v3-dq32-mbs16-gas2-smoke-seed42`，AR/Joint run 分别为 `5g4w76pt`/`jwndf741`。本地日志和 checkpoint 仍为权威记录。
- 运行状态：真实 2B/四卡 AR、Joint、save 与 resume smoke 已完成；没有终止或抢占既有 GPU 进程。
- smoke checkpoint 声明：`ZR0_SMOKE_OUTPUT_ROOT` 下的 checkpoint 仅用于接口和恢复验证，不是正式实验 checkpoint，不得作为正式训练结果发布。

## 数据集

- 名称与版本：`molmoact_dataset_tabletop-v3_stage05`，LeRobot v3 future-difference 发布包。
- 路径：计划为 `/opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05`，实际由 `ZR0_DATASET_ENTRIES` 对应 registry entry 决定。
- 范围：发布包全部 310,743 帧、1,881 episodes、14 tasks 均可读取；release metadata 中不存在 `training_eligible`，resolved manifest 记录 `exists=false, used=false, source=unavailable_in_release`。使用发布包全部可读取样本，未按 upstream `training_eligible` 过滤，也不扫描 Stage05 或推测 eligibility 分布。
- 划分：smoke 从 registry 配置的训练范围采样；validation/test 不运行。
- 采样：dataset entries 与 sample ratios 分别由 `ZR0_DATASET_ENTRIES` 和 `ZR0_DATASET_SAMPLE_RATIOS` 显式传入，数量必须一致，比例必须在 `(0,1]` 且不定义 oversampling。v3 以 seed 42 按 epoch 打乱 episode，episode 内按 frame 连续读取；相同 seed/epoch 可在 resume 时重建顺序。真实 smoke 使用 4 个 DataLoader worker 和每 worker 单 episode 文件缓存。
- 视角：当前帧 `first_view`、`second_view`、`wrist_image` 三路图像。
- 标签：AR 使用 task 和规范化 `train_data`；joint 额外使用当前 7 维 state 与当前起连续 32 步的 7 维真实 action。
- 禁止输入：Qwen user context 不含 `slot_data`、`train_data`、未来图像、未来 state 或 future timestamp；`train_data` 只作为 assistant target。v3 训练和 direct inference 共用 prompt builder，三路 camera 顺序与 task `strip()` 后的 `<TASK> ... </TASK>` 完全一致；真实样本两条路径 C 均为 177 token 且逐 token 相同。
- 归一化：raw state/action 使用全数据集 `meta/stats_gr00t.json` 中对应 `q01/q99` 做逐维 min-max normalization；不回退到 sample/batch/episode 统计。

## 训练阶段和完整命令

共同环境变量示例中的值必须在运行前按长度审计和目标机器确定；`ZR0_MAX_LENGTH` 不在代码中硬编码。只要 assistant target 会被截断，adapter 必须在 forward 前报告 episode/frame/sample ID、原始长度与 max length 并退出。

```bash
export ZR0_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct
export ZR0_DATASET_ENTRIES='molmoact_tabletop_v3_stage05'
export ZR0_DATASET_SAMPLE_RATIOS='1.0'
export ZR0_MAX_LENGTH='<由只读长度审计决定>'
export ZR0_SMOKE_OUTPUT_ROOT=/path/to/query-conditioned-ar-joint-smoke

bash scripts/run_query_ar_joint_smoke.sh ar-step1
bash scripts/run_query_ar_joint_smoke.sh ar-resume-step2
bash scripts/run_query_ar_joint_smoke.sh joint-step1
bash scripts/run_query_ar_joint_smoke.sh joint-resume-step2
```

阶段 1 `ar-step1`：训练 Qwen VLM 和 32 个 Difference Query；不构造或保存 Action Expert 实例/权重，checkpoint 不包含 `action_expert.safetensors`；`action_expert_config.json` 只是后续 Joint 随机构造所需的架构参数。`loss_type=vlm`，从基座模型开始并保存 VLM/Query、optimizer/scheduler/global step、checkpoint kind 和 resolved dataset manifest。

阶段 2 `ar-resume-step2`：从 `checkpoint_kind=ar_only` 的 step-1 checkpoint 恢复 VLM、Query 及完整训练状态，继续到同步 optimizer step 2；不查找或加载 Action Expert 权重。

阶段 3 `joint-step1`：从 `checkpoint_kind=ar_only` 的 checkpoint 加载 Qwen VLM 和 Difference Query；启动器要求 checkpoint metadata 与 Query config/weight sidecar 存在，避免误用 joint checkpoint 或静默随机初始化 Query；Action Expert 以固定 seed 全新随机初始化；训练 VLM、Query 和 Action Expert 到 step 1。

阶段 4 `joint-resume-step2`：从 joint step-1 checkpoint 恢复 VLM、Query、Action Expert、optimizer、scheduler 和 global step，继续到 step 2。

## 优化器、Loss 与规模

- 两个分支共用一个 AdamW 参数组，只包含 `requires_grad=True` 参数；peak LR `1e-5`，weight decay `0.01`，betas `(0.9, 0.95)`，epsilon `1e-8`。
- Scheduler：linear warmup 后 cosine decay，warmup ratio `0.05`，最低 LR/peak LR 比例 `0.1`；没有模块学习率倍率。
- Gradient clipping：由默认 Accelerate/DeepSpeed 配置执行 `1.0`。
- AR：`total_loss = 1.0 * autoregressive_loss`；AR loss 监督 canonical JSON、`<|im_end|>` 与 chat template 结构性换行，参与 VLM/Query 反向传播，Action Expert/FМ loss 不存在于计算图。
- Joint：`total_loss = 1.0 * autoregressive_loss + 5.0 * flow_matching_loss`；两个 loss 均参与反向传播。
- GPU：4 张 A800-SXM4-80GB；逐级探测 per-device micro-batch `1,2,4,8,16,32`，初始探测 GAS 固定 1；micro 32 的两步 smoke 通过，但在正式长程运行 step 11 OOM，最终稳定候选改为 micro 16/GAS 2。
- 正式候选 `effective_global_batch_size = 16 x 2 x 4 = 128`；Accelerate config、launch 参数、训练 CLI 和 DeepSpeed runtime engine 均断言同一个整数 GAS 2。
- 最终候选每阶段 step 1 处理 128 个样本，resume 到 step 2 后累计处理 256 个样本；epoch 参数为 1，但由 `max_train_steps` 先终止。
- 混合精度：BF16；gradient checkpointing：训练 VLM 时启用。
- validation、early stopping：不适用于本 smoke；checkpoint 每个同步 optimizer step 保存；TensorBoard 日志按训练循环默认间隔记录。

## 图像处理

- 原始三路图像均为 RGB `640x480`（宽 x 高）。
- adapter 为每路图像显式请求 `224x224`，不保持长宽比；`qwen_vl_utils.fetch_image` 使用 Pillow RGB `Image.resize` 的默认 bicubic 重采样。
- 不 crop、不 pad、不 letterbox、不做数据增强。
- processor 不再次 resize；按 Qwen3-VL processor 先缩放像素，再用 mean `[0.5, 0.5, 0.5]`、std `[0.5, 0.5, 0.5]` 归一化。
- 三视角按 `first_view`、`second_view`、`wrist_image` 顺序作为独立 image content 插入同一个 user message；训练与本 smoke 的 forward 预处理一致。未运行 rollout 评估。

## 动作配置

- 原始 action/state dimension：7；模型 padding dimension：64。
- action 表示：数据集原始 7 维连续控制量；末端执行器维度沿用数据集定义，不另行离散化。
- action chunk/horizon/prediction horizon：32/32/32；训练不执行环境动作，因此 execution horizon 不适用。
- 控制频率：数据集 10 FPS。
- episode 尾部不足 32 步时补零，不跨 episode、不重复最后动作。
- `action_mask = temporal_valid[:, None] & dimension_valid[None, :]`，形状 `[32,64]`；无效时间和 padding 维不参与 Flow Matching loss。
- 解码路径：normalized action decoder 输出裁至 7 维，再用相同 q01/q99 min-max denormalization。

## 实际结果

- 实际运行：2026-09-03，逐级完成 micro-batch `1,2,4,8,16,32` 的真实 AR/Joint step 1；随后在第一次正式 AR 中确认 micro 32 不能稳定覆盖后续长度分布，最终候选根目录为 `outputs/probe/tabletop_v3_dq32_mbs16_gas2_20260903_212659`。
- 实际完成步数：最终 AR 与 Joint 均 fresh 到 step 1，再从各自完整 checkpoint resume 到 step 2。
- 最终 checkpoint：上述根目录的 `ar/step-2`、`joint/step-2` 及对应 `latest-model-optimizer-lr`；每次 save 均包含四个 ZeRO-2 optimizer shard 与 scheduler。AR 不含 Action Expert 权重；Joint 含 Action Expert 权重。
- AR step 1：loss `4.411907`，有效 token `60,559`，VLM/Query grad norm `382.207/326.424`，peak allocated/reserved `29.138/37.203 GiB`，吞吐 `18.929 samples/s`。
- Joint step 1：AR/FM/total `2.155546/1.385566/9.083376`，有效 AR token/FM element `60,559/28,672`，VLM/Query/Action Expert grad norm `38.465/31.572/8.792`，peak allocated/reserved `31.908/39.543 GiB`，吞吐 `18.944 samples/s`。
- Resume step 2：AR loss `2.273553`，peak reserved `41.994 GiB`；Joint AR/FM/total `2.021886/1.359084/8.817307`，peak reserved `44.512 GiB`。均从完整四卡 ZeRO checkpoint 恢复 optimizer、scheduler 和 global step，并继续实际参数更新；每个 optimizer step 均为两个 micro-batch/rank、128 global samples。
- W&B run：AR `https://wandb.ai/jumbo3r-zhejiang-university/ZR-0-Pretraining/runs/5g4w76pt`；Joint `https://wandb.ai/jumbo3r-zhejiang-university/ZR-0-Pretraining/runs/jwndf741`。
- NaN/OOM：最终 micro 16/GAS 2 smoke 无 NaN/Inf 或 OOM。micro 32/GAS 1 的短 smoke 曾通过，但第一次正式 AR 在 step 11 OOM；失败目录和 run `45aukw18` 保留为证据，没有改变训练目标或有效 global batch。
- checkpoint 恢复：已用真实 Qwen3-VL-2B、真实 v3 数据、四卡 ZeRO-2 完成验证；正式 Joint 仍只允许正式 AR step-7284，禁止使用本 smoke。
- 与计划偏差：原模板中的单卡 tiny smoke 被四卡真实模型/真实数据 smoke 取代；长度使用全量审计确认的 1024。
- 后续评估：不在本 smoke 范围。

## 已完成的只读审计

长度审计命令（stdout 为 JSON；只读取 episode/frame/task/train_data，图像使用 metadata 尺寸的内存 dummy）：

```bash
PYTHONPATH=/opt/data/private/lq/ZR-0 \
/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python \
scripts/audit_future_difference_lengths.py \
  --dataset-root /opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05 \
  --processor-path /opt/data/private/lq/models/Qwen3-VL-2B-Instruct \
  --max-length 1200
```

- 范围：全部 310,743 帧；未按 upstream `training_eligible` 过滤。
- `max_length=1200`：sequence overflow 0，input-only truncation 0，target truncation 0，target truncation ratio `0.0`。
- C token：min 177，mean 177.280，p50 177，p90/p95 178，p99/p99.9/max 179。
- JSON 内容 token：min 141，mean 474.246，p50 471，p90 504，p95 511，p99 521，p99.9 541，max 566。
- assistant termination 固定 2 token（`[151645,198]`，即 `<|im_end|>` 与换行）；完整 T、实际保留 T 与有效监督 token 均为 JSON 内容长度加 2，且无截断。
- 投影总长：min 320，mean 653.526，p50 651，p90 683，p95 691，p99 700，p99.9 720，max 746。
- 决策：对当前 processor/chat template/三路 224x224 请求尺寸，正式命令采用 `MAX_LENGTH=1024`；最大投影总长 746，完整 target 和 termination 均可保留。更换任一条件后必须重跑审计。

Interval 审计命令（只读取 steps、episode mapping 和上游 annotation JSONL）：

```bash
PYTHONPATH=/opt/data/private/lq/ZR-0 \
/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python \
scripts/audit_future_difference_intervals.py \
  --dataset-root /opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05 \
  --action-horizon 32
```

- 范围：全部 310,743 帧；annotation available 310,743，unavailable 0；eligibility 未读取、未计算。
- nominal `[t,t+31]` 闭区间分类：exact 374，full 41,383，partial 268,986，none 0。
- available 中非 exact mismatch：310,369，比例 `0.9987964331`。
- annotation 闭区间长度：min 2，p50 19，p90 41，p95 45，p99 50，max 51。
- chunk 交集长度：min 1，p50 15，p90/p95/p99/max 32。
- audit 输出额外记录 `info.json::codebase_version=v3.0`，固定 metadata 文件集合的相对路径/大小/SHA-256 和聚合 identity；不 hash Parquet、图像或视频。本轮真实全量覆盖 3,766 个 metadata 文件，identity SHA-256 为 `cbcee48a48b3eb2d9bfb62cd846eccd1ec6af5d73065fe57cdeba7013c1b85f5`。

每次 dataset 构造还会把确定性 resolved manifest 打印到主进程，并保存为 `<output>/resolved_dataset_manifest.json`；相同完整内容与 content hash 进入每个普通/DeepSpeed checkpoint。每个 entry 固定 camera/字段/维度/H、stats 文件 SHA-256 及 state/action q01/q99；同一路径替换 stats 会改变 manifest hash。resume 将 adapter、camera 顺序、字段、维度、normalization、stats identity/q01/q99、H、sample ratio 等视为数据语义，不一致直接退出。direct policy 必须从 checkpoint manifest 选择唯一同名 entry 并校验后才归一化；新 checkpoint 缺 manifest 默认失败，legacy 仅能显式传 `--allow_legacy_checkpoint_without_manifest` 并产生强警告。
