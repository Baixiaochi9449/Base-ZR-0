# Tabletop v3 Difference Query 32 Two-Stage Pretraining

## 基本信息

- 实验名称：`tabletop_v3_dq32_gbs128_seed42`
- 实验目的：先用完整 `train_data` 监督训练 Qwen3-VL-2B 与 32 个 Difference Query，再从 AR 最终权重 warm-start VLM/Query，并以 seed 42 新建 Action Expert 进行 AR+Flow Matching 联合训练。
- 创建时间：2026-09-03（Asia/Shanghai）。
- 负责人：lq；执行与记录协作：Codex。
- 基座模型：`/opt/data/private/lq/models/Qwen3-VL-2B-Instruct`。
- 模型来源标识：本地 ModelScope checkout 仅能确认可变 revision `master`，无法证明不可变远端 commit；`model.safetensors` SHA256 为 `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`。
- 代码基准 commit：`d3b8f943477a52a2836543219dab25ff6bd07941`；正式启动前另记录 staged diff、launcher 和模型/数据哈希。该 HEAD 是用户在 preflight 期间恢复上次提交内容后形成的当前基准，执行方未回退或改写。
- 当前未提交代码：本任务的 GAS、日志诊断、视觉 manifest、launcher、测试和文档修改；用户已有 `代码修改2.md`、`代码修改3.md` 与既有未跟踪实验文档不属于本实验且不暂存。
- 随机种子：42。Query 和 Action Expert 的实际初始化统计/哈希由每阶段 `initialization_manifest_fresh.json` 记录。
- 正式输出：第一次 AR 尝试 `outputs/pretrain/tabletop_v3_dq32_ar_gbs128_seed42` 因 micro-batch 32 在 step 11 OOM 而封存；正式重试使用新目录 `outputs/pretrain/tabletop_v3_dq32_ar_gbs128_seed42_mbs16_gas2` 和 `outputs/pretrain/tabletop_v3_dq32_joint_gbs128_seed42_mbs16_gas2`，不覆盖失败目录。
- Probe/smoke 输出：`outputs/probe/tabletop_v3_dq32_mbs*_gas*_20260903_*`，与正式输出严格隔离；正式 Joint 不得使用其中任何权重。
- W&B：SDK 0.29.0，entity `jumbo3r-zhejiang-university`，project `ZR-0-Pretraining`，正式重试 group `tabletop-v3-dq32-gbs128-seed42-mbs16-gas2`。正式重试 run ID 固定为 AR `v3enlq43`、Joint `ptcmynsm`；失败的第一次 AR run 为 `45aukw18`，未启动的旧 Joint ID `b1juqv7i` 不再使用。启动认证是硬门禁；启动后短暂断连由 SDK/本地缓冲重试，不影响本地日志或 checkpoint。
- 当前状态：全量审计、最终 micro/GAS 的真实四卡 ZeRO-2 AR/Joint optimizer step、checkpoint save/resume 以及单元/CUDA 门禁均已通过；第一次正式 AR 在 step 11 OOM，micro 16/GAS 2 正式重试尚未启动。

## 配置来源区分

- ZR-0 原有设计：Qwen3-VL backbone、final RMSNorm 后的 VLM hidden、含 proprioception/noisy action/flow timestep 的 Action Expert，以及 `tau ~ Beta(1.5,1.0)` 的 Flow Matching。
- 本实验 Difference Query：显式 `use_difference_query=True`、`num_difference_queries=32`、SDPA；使用既有 `[C,Q,T,P]` mask，Action Expert 唯一可见的 VLM context 为 final RMSNorm 后 32 个 Query hidden。
- GBS 128 实验配置：4 GPU、per-device micro-batch 16、GAS 2、peak LR `1e-5`、AR 7,284 steps、Joint 19,424 steps；这些数值不是原论文默认训练规模。

## 数据集

- 名称/版本：`molmoact_dataset_tabletop-v3_stage05`，LeRobot v3.0。
- 路径：`/opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05`。
- 范围：1,881 episodes，310,743 unique frames，全部发布帧；`sample_ratio=1.0`。
- 划分：不做 train/validation/test 拆分，不运行验证。
- Eligibility：发布包没有 `training_eligible`，不据此过滤；manifest 固定记录 `exists=false, used=false, source=unavailable_in_release`。
- 采样：按 frame 训练，seed 42 进行 episode-grouped epoch shuffle，episode 内按 frame 顺序；不修改源数据。
- 文本监督：完整 canonical `train_data` JSON 和 assistant termination `[151645,198]`，不允许截断或跳过异常样本。
- 相机固定顺序：`first_view`、`second_view`、`wrist_image`，仅当前时刻三路图像进入 C；未来图像、target、slot、未来 state/action 不进入 C。
- State/action：连续 7 维，数据集 10 FPS；Joint 使用当前 state 与从当前帧开始的 action horizon 32。
- Interval 审计：268,986 个 frame 属于 annotation interval 与 `[t,t+31]` 的 partial overlap（约 86.562%）。这只是独立审计指标，不等于 action temporal mask 无效率，也不用于过滤样本。

## 序列长度审计

- 审计范围：全部 310,743 个 `train_data`，只读取 metadata/parquet 文本，不加载真实图像。
- `max_length=1024`；C 长度 min/mean/p50/p90/p95/p99/max 为 `177/177.280/177/178/178/179/179`。
- JSON target 长度 min/mean/p50/p90/p95/p99/max 为 `141/474.246/471/504/511/521/566`。
- termination 固定 2 token；完整监督区最大 568；投影总长 p50/p90/p95/p99/max 为 `651/683/691/700/746`。
- 非法 JSON 0，空监督 0，input truncation 0，target/termination truncation 0，overflow 0。

## 图像处理

- 原始三路图像：RGB，`640x480`（宽 x 高）。
- 每路显式 resize 到 `224x224`，不保持长宽比，Pillow bicubic；无 crop、pad、letterbox 或增强。
- rescale factor `1/255`，mean/std 均为 `[0.5,0.5,0.5]`。
- vision patch/temporal patch/spatial merge 为 `16/2/2`。
- 实测每路 `image_grid_thw=[1,14,14]`、49 个视觉 token，三路总计 147；单样本 `pixel_values` 形状 `[588,1536]`。
- 以上字段及相机顺序写入每个 checkpoint 的 `resolved_dataset_manifest.json`。

## 训练阶段

### AR-only

- 初始化：本地 Qwen3-VL-2B-Instruct + seed 42 随机 Query `[32,2048]`。
- `loss_type=vlm`；`total_loss = 1.0 * AR loss`，FM weight 0。
- 训练 Qwen VLM 与 Difference Query；不构造、不加载、不保存 Action Expert。
- 3 epochs，`max_train_steps=7284` optimizer steps；在 2428、4856、7284 保存模型 checkpoint，并持续保存 latest optimizer/scheduler/global step。

### Joint

- 初始化：仅从正式 AR `step-7284` 精确加载 VLM 与 Query；不得使用 smoke checkpoint。
- Action Expert：seed 42 全新随机初始化，不从 AR checkpoint 加载；optimizer 和 scheduler 全新创建。
- `loss_type=vlm_and_action`；`total_loss = 1.0 * AR loss + 5.0 * Flow Matching loss`。
- 训练 Qwen VLM、Difference Query、Action Expert；8 epochs，`max_train_steps=19424` optimizer steps。
- 在 4856、9712、14568、19424 保存模型 checkpoint，并持续保存 latest 完整恢复状态。
- Joint 中断恢复时，同一 Joint latest checkpoint 同时提供 VLM/Query、Action Expert、optimizer、scheduler 和 global step；W&B 使用原 run ID 与 `resume=must`。

## 优化器和训练规模

- AdamW：betas `(0.9,0.95)`，epsilon `1e-8`，weight decay `0.01`；所有可训练参数保持一个 optimizer group。
- LR：peak `1e-5`，minimum `1e-6`，linear warmup 后 cosine；AR warmup 364 steps，Joint warmup 971 steps；无模块 LR multiplier。
- BF16，gradient clip norm `1.0`。Qwen VLM 启用 gradient checkpointing；Action Expert 当前没有 gradient-checkpointing 实现，不修改模型补充。
- GPU：仅 `CUDA_VISIBLE_DEVICES=0,1,2,3`；4 张 A800-SXM4-80GB。
- `effective_global_batch_size = 16 x 2 x 4 = 128`（per-device micro-batch x GAS x world size，名义完整 optimizer window）。
- 自然尾批：每 epoch 310,743 unique frames；4-rank 对齐后 310,744 次呈现并确定性重复 1 帧。前 2,427 个 optimizer step 为 global batch 128，最后一步每 rank 两个自然 micro-batch（16+6，共 22 samples）、实际 global batch 88；不补到 128、不丢弃。
- 每 epoch 2,428 optimizer steps。AR 共呈现 932,232 个样本，Joint 共呈现 2,485,952 个样本；loss 始终按当前 window 的真实有效 AR token 和 FM action element 分母归一化。
- DataLoader workers 4，pin memory 开启；validation/early stopping 不适用。

## 日志与停止条件

- 权威本地记录：各阶段 `train.log` 中的 stdout/stderr、`training_metrics.jsonl`、TensorBoard、`launch_manifest_*.json` 和 checkpoint manifests；W&B 是镜像监控，不是 checkpoint 正确性的单点依赖。
- 每 10 optimizer steps及 step 1/最终 step记录 raw/weighted AR、raw/weighted FM、total、AR token count、FM action element count、LR、global及 VLM/Query/Action Expert grad norm、吞吐、GPU peak、读取重试、截断数、action invalid element/timestep ratio。
- 分模块 grad norm 由参数名归属和 ZeRO-2 本地梯度分片统计，只归约平方和标量，不改变 optimizer group。
- 任一真实 ZeRO-2 smoke、GAS=2 boundary、resume、SDPA、W&B 启动认证、manifest 或 finite 检查失败时，停在 preflight，不改变 LR、有效全局 batch、视角、步数或 loss 绕过。
- 正式训练中 NaN/Inf、确定性数据错误、checkpoint 错误或 OOM 时停止并保留证据；不终止其他 GPU 进程。

## 完整启动入口

- Smoke：`scripts/run_query_ar_joint_smoke.sh` 的 `ar-step1`、`ar-resume-step2`、`joint-step1`、`joint-resume-step2`；最终正式配置固定为 4 GPU x micro 16 x GAS 2。
- 正式：`scripts/run_query_ar_joint_formal.sh ar train`，AR 验证完成后运行 `scripts/run_query_ar_joint_formal.sh joint train`。
- 恢复：分别运行 `scripts/run_query_ar_joint_formal.sh ar resume` 或 `scripts/run_query_ar_joint_formal.sh joint resume`；MODEL_PATH 必须严格为对应输出的 `latest-model-optimizer-lr`。
- 每次正式 fresh/resume 启动前，launcher 写入完整展开命令、解析配置、Git HEAD/staged diff hash、launcher hash、模型 hash、dataset manifest/hash 和尾批合同。

## 实际结果

- Preflight：最新 `tests/` 全集 `261 passed, 3 skipped`；随后以 `ZR0_RUN_CUDA_TESTS=1 ZR0_RUN_ZERO2_TESTS=1` 运行 `5 passed`，覆盖 BF16 efficient SDPA、1-rank 与 2-rank ZeRO-2 production save/resume。launcher 子进程最初因测试未固定 `sys.executable` 而在无 PyYAML 的 base Python 中产生 7 个收集后失败；修正测试环境注入后 launcher `21/21`、全集均通过。全仓根目录收集会进入 vendored RoboTwin 与历史 `result/eval` 快照，因其独立依赖/重复模块名产生 collection error，不属于本实验门禁。
- 资源快照：4 张 A800-SXM4-80GB；preflight 时 GPU 0/1/2/3 各约 13,504 MiB 已用、67,724 MiB 可用、利用率 0%。既有进程未终止或抢占。输出文件系统约 258 TiB 可用；基座模型约 4.0 GiB，数据集约 198 GiB。
- W&B 认证：用户 `jumbo3r` 以目标 entity 访问 `ZR-0-Pretraining` 成功；真实 probe 均在线记录。首次只读探测因 SDK 0.29.0 移除了旧 `wandb.util.generate_id` 入口失败，一次本地 service 瞬时启动超时后官方 API 查询成功；两者未启动训练 run。
- 长度审计：310,743 条全量完成；非法 JSON 0、空监督 0、overflow/input/target/termination truncation 均为 0，最大投影总长 746，正式采用 1024。Interval 审计 identity SHA256 `cbcee48a48b3eb2d9bfb62cd846eccd1ec6af5d73065fe57cdeba7013c1b85f5`，partial 268,986，仅记录不采样/过滤。
- 初始真实 smoke `outputs/smoke/tabletop_v3_dq32_gbs128_seed42_20260903_190408` 在 optimizer step 0 暴露 Accelerate 1.6.0 将 DeepSpeed `gradient_accumulation_steps: auto` 解析为 `int("auto")` 的错误。随后把 config、outer launch、训练 CLI 和 runtime engine 全部固定为同一整数，并增加 prepare 前后断言；未通过改变 LR、视角、step 或 loss 绕过。
- 其他 step-0 probe 失败保留在 `outputs/probe/tabletop_v3_dq32_mbs1_gas1_20260903{,_194132,_194455}`：依次暴露旧 dataset entry、custom batch sampler 与 Accelerate prepare 的 `batch_size=None` 适配问题。`..._194715` 和 `..._195319` 在首个 backward 暴露 ZeRO-2 梯度诊断读取时机问题，均未完成 optimizer step/checkpoint。最终诊断在 DeepSpeed optimizer step 前按参数归属读取 partitioned/averaged gradients，不改变 optimizer groups。
- 真实显存探测按 micro-batch `1,2,4,8,16,32` 递增，均使用 4 GPU、GAS 1、真实 Qwen3-VL-2B、三视角、1024、BF16、SDPA、gradient checkpointing 和 ZeRO-2。AR/Joint step-1 峰值 allocated GiB 分别为 `13.895/17.534`、`13.900/17.538`、`13.908/17.547`、`17.270/20.111`、`27.826/30.689`、`48.769/51.674`；micro 32 的两步 smoke 可以运行，但第一次正式 AR 在 step 10 已达 reserved `69.824 GiB`，随后 step 11 rank 1/2 backward 各申请 `13.66 GiB` 时 OOM。因此 micro 32 不是长程稳定配置，按约束只降低 micro-batch 并等比例提高 GAS。
- 最终 smoke 根目录：`outputs/probe/tabletop_v3_dq32_mbs16_gas2_20260903_212659`。AR W&B `5g4w76pt`：step 1 AR loss `4.411907`，AR token `60,559`，VLM/Query grad norm `382.207/326.424`，peak allocated/reserved `29.138/37.203 GiB`，吞吐 `18.929 samples/s`；完整恢复后 step 2 AR loss `2.273553`，peak reserved `41.994 GiB`。Joint W&B `jwndf741`：只从 AR smoke warm-start VLM/Query并以 seed 42 新建 Action Expert；step 1 AR/FM/total=`2.155546/1.385566/9.083376`，AR/FM denominator=`60,559/28,672`，VLM/Query/Action Expert grad norm=`38.465/31.572/8.792`，peak allocated/reserved `31.908/39.543 GiB`；完整 joint resume 后 step 2 AR/FM/total=`2.021886/1.359084/8.817307`，peak reserved `44.512 GiB`。每步均处理两个 micro-batch/rank 和 128 global samples；两阶段反复保存四个 ZeRO shard、scheduler 和 global step，无 NaN/Inf、OOM、读取重试或截断。
- AR smoke Query 初始化为 `[32,2048]` FP32，seed 42，mean/std `-9.458381e-07/0.02011986`，L2 `5.150684`，SHA256 `003b1650a11271d934e768e615a752fd618530bbf7b26fc28e37b27f5faffb51`；Action Expert 未构造、参数数 0、无权重文件。Joint 的 Action Expert 参数数 `558,784,576`，seed 42 初始化 SHA256 `85e3dd446e90d55bc7ed8b98a585a273f4f36b48a8d9f954ab6d427ef70c34e7`。
- AR 正式训练：第一次 run `45aukw18` 完成 step 1 至 10（step 10 loss `4.004437`）后在 step 11 backward OOM，退出状态 1；保存间隔为 2,428，因此没有可恢复 checkpoint。失败目录原样保留，micro 16/GAS 2 重试尚未启动。
- Joint 正式训练：未启动，实际 step 0，无 checkpoint、W&B URL 或曲线。
- NaN/OOM：第一次正式 AR 发生上述 OOM；此前与最终 micro 16/GAS 2 smoke 均未发生 NaN/Inf。调整只把 `32 x 1 x 4` 改为 `16 x 2 x 4`，没有改变 LR、有效 global batch、视角、正式步数或 loss 权重。
- 后续 rollout/LIBERO：不在本实验范围。
