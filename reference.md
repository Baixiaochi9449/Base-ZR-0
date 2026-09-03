# Implementation Reference

## Tabletop v3 DQ32 两阶段预训练基础设施

- 日期：2026-09-03
- 修改目的：修复训练 CLI 缺失 GAS 导致正式入口必然在 Accelerator 初始化前失败的问题，并为指定的 4-GPU、GBS 128、AR-only 到 Joint 实验补齐可追溯的运行尺度、视觉输入 contract 和本地诊断。
- 涉及文件：`accelerate_configs/accelerate_config.yaml`、`utils/cli_options.py`、`train_vla.py`、`utils/load_training_dataset.py`、`utils/optimizer_step_loss.py`、`utils/training_tokenization.py`、`utils/dataset_adapters.py`、`utils/dataset_spec.py`、`utils/dataset_manifest.py`、`utils/wandb_training_logger.py`、两阶段 launcher、launch recorder、相关测试与实验文档。
- 配置开关：`--gradient_accumulation_steps` 默认 1；`--expected_global_batch_size` 默认不校验；`--logging_steps` 默认 10；`--log_training_diagnostics` 默认关闭。关闭诊断时不计算分模块 norm、吞吐、显存和数据质量扩展指标，不改变 optimizer、loss 或更新。
- 默认状态：保持单 optimizer group 和原 loss。正式配置将 per-device micro-batch/GAS/world/global 明确固定为 `16/2/4/128`；DeepSpeed config、外层 `accelerate launch`、训练 CLI 和 prepare 后 engine 必须是同一 GAS 整数，禁止 `auto`，任一不一致立即退出。普通非 launcher 调用仍保留 CLI 的 GAS=1 默认值。
- 实现来源：基于仓库现有 optimizer-step window、Accelerate/DeepSpeed ZeRO-2、v3 adapter、resolved dataset manifest 和 W&B logger 做最小适配；没有修改 Qwen forward、Difference Query mask、final RMSNorm 或 Action Expert/Flow Matching 模型结构。
- 具体设计：prepare 前后分别打印并硬校验 world size、micro-batch、GAS 和 nominal GBS。`EpochGroupedDistributedBatchSampler` 先构造全局 episode-grouped 顺序，只补到 world size 的倍数，再按 rank stride 切分；310,743 unique frames 每 epoch只确定性重复 1 帧，4 rank 各 77,686 个 sample，最后一个 batch 各 22 个 sample，形成 2,428 个 optimizer steps 和真实 global tail 88。`DataLoaderConfiguration(even_batches=False)` 禁止 Accelerate 补满尾批；AR/FM 继续使用该窗跨 rank 的真实有效元素分母。诊断按参数名把互斥且无遗漏的 trainable 参数归属为 VLM、Difference Query 或 Action Expert；ZeRO-2 在 engine 清空 gradient 之前从 partition/averaged gradient 累加 FP32 平方和，只 all-reduce 三个标量，不改变 optimizer group、step 顺序或超参数。
- 输入与输出：v3 adapter 为真实 Qwen processor 解析 `vision_input_contract`，记录三相机顺序/原始尺寸、224x224 bicubic、归一化、patch/merge、`image_grid_thw`、每路49/总147视觉 token和 pixel shape；resolved manifest version 4 将其写入 output 和每个普通/DeepSpeed checkpoint。训练另写 `initialization_manifest_{fresh,resume}.json`、逐 optimizer step 的 `training_metrics.jsonl`，launcher 将 stdout/stderr 同步追加到各阶段 `train.log`。
- 与原有流程的关系：现有 AR-only 不构造或保存 Action Expert 实例/权重；AR checkpoint 中的 `action_expert_config.json` 仅保存 Joint 后续随机构造所需的架构参数。AR warm-start Joint 只加载 VLM/Query，Joint resume 恢复完整状态；joint resume 在启动前要求 checkpoint 同时具备 Query、Action Expert、scheduler 和 kind metadata sidecar。数据读取 transient retry 仍只重试同一样本；adapter 额外返回 retry count 用于日志。268,986 partial-overlap frames 只写入 interval 审计，绝不参与 action mask 或采样。
- W&B 行为：online `wandb.init` 失败仍阻止正式启动；训练已开始后 `run.log` 短暂异常被本地缓冲并在后续 step/finish 重试，stdout、JSONL、TensorBoard 和 checkpoint 始终为权威记录。
- 关闭功能后的行为：不传新参数的旧调用使用 GAS 1、无 GBS assertion、10-step 日志和无扩展诊断；无真实 image processor 的测试/legacy adapter 不生成 vision contract。所有新增实验功能均由 launcher 显式启用。
- 验证方式：自然尾批测试覆盖全部六组 `micro/GAS=(1,32),(2,16),(4,8),(8,4),(16,2),(32,1)`，均得到 310,744 次呈现、1 个重复、2,428 steps 和 tail 88；诊断开关比较单 optimizer group、loss、grad 和参数更新。真实 processor 验证三路 grid 和147视觉 token。真实四卡 Qwen3-VL-2B AR/Joint 在 micro 1/2/4/8/16/32 均完成 optimizer step；最终 `micro/GAS=16/2` 完成两阶段 fresh、save、完整 optimizer/scheduler/global-step resume 和第二步更新。最新 `tests/` 为 261 passed/3 opt-in skipped，显式 CUDA/ZeRO-2 为 5 passed。launch recorder 已对单文件与 HF sharded safetensors 身份解析做测试，并在真实三 shard checkpoint 上验证。
- 已知限制：Action Expert 没有 activation gradient checkpointing；本地模型只可标识 ModelScope `master`，因此使用权重 SHA256 作为不可变身份。Linux kernel 5.4 低于 DeepSpeed 建议的 5.5，但全部真实 smoke 未发生 hang。micro 32 的短 smoke 峰值 reserved 为 AR 55.260 GiB、Joint 56.838 GiB，但第一次正式 AR 在 step 10 达到 69.824 GiB 并在 step 11 OOM；因此正式重试只将配置调整为 micro 16/GAS 2，最终 smoke peak reserved 为 AR 41.994 GiB、Joint 44.512 GiB，有效全局 batch 保持128。

## 第五轮定向修复：optimizer-step 全局目标与训练部署契约

- 日期：2026-09-03
- 修改目的：使 AR/FM 的实际反向目标与日志都严格等于一个 optimizer step 内所有 data-parallel rank、全部 micro-batch 的各自有效元素均值，并把 v2 observation history、VQA/VLA 监督能力和 v3 direct-only 限制纳入 checkpoint/policy 契约。
- 涉及文件：`utils/optimizer_step_loss.py::global_supervision_counts/scaled_microbatch_loss/OptimizerStepMetricAccumulator`；`train_vla.py::iter_optimizer_step_windows/run_optimizer_step_window/train`；`model/reasoning_vla_model.py::ZR0Model._loss_outputs/forward`；`model/flow_matching_action_head.py::masked_loss_sum_and_count`；`utils/dataset_spec.py::ObjectiveRequirements/ObservationContract/resolve_objective_requirements/resolve_dataset_spec`；`utils/dataset_manifest.py`；`utils/load_training_dataset.py::build_concat_streaming_dataset`；`policies/reasoning_vla_policy.py::ZR0Policy.__init__`；`server.py`、`utils/cli_options.py`、`accelerate_configs/accelerate_config.yaml` 及定向测试。
- 配置开关：复用 `loss_type`、两个 loss weight、`gradient_accumulation_steps` 和 `window_size`；新增保守兼容开关 `--allow_legacy_checkpoint_without_observation_contract`，默认关闭。该轮曾将 DeepSpeed GAS 设为 `auto`；当前正式入口已按本文件顶部 2026-09-03 条目改为 config/launcher/CLI/runtime 同一显式整数。
- 默认状态：每个 optimizer-step window 先只从 batch tensor 统计 AR shift 后有效 label 数与 action mask 数，不保留计算图；随后逐 micro-batch forward/backward。action-only 的全局 FM count 为零立即失败；joint 全 VQA window 的 FM 为图连接的有限零。每个 window 创建独立 detached 日志 accumulator，optimizer step 后即释放。
- 实现来源：基于仓库现有 Qwen causal loss、Flow Matching 逐元素 MSE、Accelerate 1.6.0 和 DeepSpeed 0.15.4 调用路径做最小适配；optimizer-step reducer 与 observation contract 为本仓库自定义实现。没有修改 Qwen decoder、Difference Query 或 Action Expert 结构。
- 原始实现位置或外部来源：AR numerator 复用 `Qwen3VLForConditionalGeneration` 返回的 local mean，并乘与 HF shift 完全一致的 `labels[...,1:] != -100` count；FM numerator 复用 `FlowmatchingActionHead.forward` 的逐元素 MSE。Accelerate 非 DeepSpeed `backward` 会除以 GAS；DeepSpeed wrapper 调用 `engine.backward/step`，由 engine 按 GAS 缩放并执行 ZeRO 边界、clip、optimizer 与 scheduler。
- 具体设计：设 data-parallel world size 为 W、配置 GAS 为 G，完整 optimizer window 的跨 rank AR/FM 分母分别为全局有效 token/action element count。每个 micro-batch 传给 `accelerator.backward` 的值为 `G*W*(lambda_AR*local_AR_sum/global_AR_count + lambda_FM*local_FM_sum/global_FM_count)`。Accelerate/DeepSpeed 恰好除一次 G，DDP/ZeRO 梯度归约恰好平均一次 W，最终梯度等于两个全局均值的加权和；AR 与 FM 不共享 denominator，权重各应用一次。末尾不足 G 的 window 仍用 G 抵消框架固定缩放，并显式把最后一个实际 micro-batch 设为同步/DeepSpeed accumulation boundary。代码不直接读取或重缩放 `.grad`。
- 输入与输出：模型额外返回可微 local `ar_loss_sum`/`flow_matching_loss_sum` 与 detached count；原有 mean、weighted alias 和 `total_loss` 保持兼容。训练 logger 跨 window 与 rank 先汇总 detached numerator/count，再计算 raw/weighted/total；token/padding/truncation 指标同样聚合全部 micro-batch，invalid placeholder 不计入 count，min/max 通过 gather 计算。日志 `total_loss` 与本 optimizer step 的实际目标一致。
- 与原有流程的关系：optimizer、scheduler、global step、save/resume 顺序保持每个 optimizer boundary 一次；三种 loss 的公开定义和权重不变。v2 manifest 记录 versioned observation contract：`window_size`、旧到新的 frame-major 排列，以及由上一 policy execution horizon 决定的历史 stride；camera 内容与顺序仍由既有 manifest 字段单独锁定。v3 只允许 window 1 和 `direct_action`。action-only 在 processor、采样或数据读取前拒绝 VQA；joint 的 VQA requirements 只有 images/task/target，VLA 按 AR/FM 目标声明实际依赖，因而不改变 v3 AR-only 的懒加载隔离。
- 关闭功能后的行为：这些是训练目标与契约正确性修复，不提供恢复 micro-batch mean 累积或 action-only VQA 的开关。旧 checkpoint 缺 observation contract 默认拒绝；只有显式兼容开关开启时，才警告并采用用户当前明确给出的 contract。manifest resume 仍严格比较完整内容，包括 sample ratio；policy 只比较集中定义的部署语义字段。
- 验证方式：严格 RED/GREEN；真实 Accelerate CPU loop 比较 GAS=1/2 的 AR-only、action-only、joint、VLA+VQA、全 VQA，以及不足 GAS 的末尾 window，核对参数更新和 step 日志；两进程 gloo 覆盖 rank 有效数不同、一个 rank FM count=0 和跨 rank 重分配不变量。契约测试覆盖 v2 window 1/3 保存、resume/policy mismatch、legacy 显式 override、v3 window/direct-only、action-only VQA preflight 和 joint VQA/VLA 独立 requirements。
- 已知限制：该轮当时只执行 CPU/tiny 与两进程 gloo；后续真实 CUDA、Qwen3-VL-2B、NCCL 四卡和 DeepSpeed ZeRO-2 证据见本文件顶部 2026-09-03 条目。正式训练和 LIBERO rollout 仍未启动。

## 第四轮定向修复：尾部动作、全局 FM、推理 checkpoint 与 VQA 混合

- 日期：2026-09-03
- 修改目的：关闭 v2 episode 尾部重复动作监督、多 data-parallel rank masked FM 的 local-mean 偏差、action-only checkpoint 无法 direct inference、VLA+VQA joint 被逐样本 action 校验拒绝，以及旧 v2 metadata、padding 日志和长度审计三个边界缺陷。
- 涉及文件：`utils/load_training_dataset.py::_v2_action_temporal_valid/prepare_action_expert_inputs_cpu/StreamingLeRobotSampleDataset/VQADataset/custom_collate_fn`；`model/flow_matching_action_head.py::distributed_masked_mean/FlowmatchingActionHead.forward`；`model/reasoning_vla_model.py::ZR0Model._validate_training_inputs/from_pretrained`；`policies/reasoning_vla_policy.py::ZR0Policy.__init__`；`utils/dataset_adapters.py::LeRobotV3FutureDifferenceDataset._action_inputs`；`utils/dataset_spec.py::_resolve_v2_spec`；`utils/future_difference_audit.py::FutureDifferenceTokenMeasurer`；对应定向、checkpoint、policy 和 server 测试。
- 配置开关：没有新增训练功能开关。`loss_type`、loss 权重、`action_horizon`、manifest 和现有 `allow_legacy_checkpoint_without_manifest` 均保持原接口；`ZR0Model.from_pretrained(for_action_inference=True)` 仅由正式 action policy 加载入口使用。
- 默认状态：LeRobot v2 action/joint 样本必须携带上游生成的 `action_is_pad [H] bool`；VLA 样本显式标记 `action_supervision_available=true`，VQA joint 样本标记为 false 并使用全零 state/action mask。未提供该标记的旧模型调用按全 action-supervised 兼容处理。
- 实现来源：基于仓库原 `LeRobotDataset._get_query_indices` 已生成的 `<key>_is_pad`、原 masked FM、checkpoint kind 和 VQA dummy contract 做最小适配；data-parallel 全局均值与 sample-level availability 校验为本仓库自定义修复，没有修改 Qwen、Difference Query 或 DiT 结构。
- 原始实现位置或外部来源：复用 `lerobot/lerobot/common/datasets/lerobot_dataset.py::_get_query_indices/getitem_with_delta_timestamps` 的 episode clamp 与 `action_is_pad`；复用 `model/flow_matching_action_head.py::FlowmatchingActionHead.forward` 的逐元素 MSE；复用 `model/reasoning_vla_model.py::_read_checkpoint_kind/from_pretrained` 与 resolved manifest 校验。
- 具体设计：v2 action mask 固定为 `(~action_is_pad)[:,None] & dimension_valid[None,:]`，无效尾部 action 清零且不进入 FM；legacy FAST 只接收相同 temporal-valid slice，clamp 后重复末帧不生成文本 action token。`distributed_masked_mean` 以 FP32 累加 local sum、精确汇总全局 valid count；所有 rank 均执行 collective，反向标量使用 `world_size * local_sum / global_count`，前向数值用 detach 校正为全局 mean。global count 为零时返回与图相连的有限零。
- 输入与输出：joint 允许 `action_supervision_available=false` 的 VQA 样本，其 action/state/mask 均为形状兼容的零张量；VLA 样本仍必须有有效 state/action，action-only 拒绝无 action 监督样本。全 VQA batch 的 FM raw/weighted 分支为有限零，混合 batch 的 VQA 元素不产生 FM 梯度。action inference 显式接受 `action_only` 和 `joint` checkpoint 并加载 Action Expert；`ar_only`/legacy AR 在构造前报错，不随机初始化 Action Expert。
- 与原有流程的关系：三种 loss 公式和权重、AR-only 无 Action Expert、AR→joint fresh Action Expert、joint resume、checkpoint kind 完整性、Difference Query `[C,Q,T,P]`/mRoPE/SDPA/final RMSNorm、v3 target/action/q01-q99 均未改变。旧 v2 缺 `grounding_camera_keys` 时规范为空 tuple，存在时保留原内容与顺序。collate 裁剪后按实际 `attention_mask` 宽度重新计算现有 `padding_token_count`，并将这个可直接推导的 post-collate 指标标记为有效；长度审计仅在 `context_tokens > max_length` 时标记 input truncation，target/termination 完整性判断不变。
- 关闭功能后的行为：这些是正确性修复，不提供恢复尾部重复监督、rank-local 均值或静默 AR action inference 的开关。非分布式 FM 退化为同一 valid-element mean；无 VQA 的纯 VLA batch 保留原 loss 语义。
- 验证方式：严格 RED/GREEN；v2 partial-horizon/FAST/end-to-end、两进程 CPU gloo 不同 valid count 与单 rank 零 count、纯 VLA/纯 VQA/混合 availability、action-only/joint/ar-only 的 model-policy-server 加载、旧 v2 metadata、post-collate padding、context `< / = / > max_length` 和 termination-only overflow 均有定向测试；另运行 Difference Query、Query-off、三 loss、checkpoint、v2/v3、manifest、policy 和完整测试集回归。
- 已知限制：两进程 gloo 验证了 DDP 梯度平均语义；真实多 GPU NCCL、DeepSpeed ZeRO、Qwen3-VL-2B、正式训练、完整数据扫描和 LIBERO rollout 未在本轮运行。

## 第三轮定向修复：v2 完整监督、数据错误、零长度 logits 与混合指标

- 日期：2026-09-03
- 修改目的：阻止 v2 assistant content/termination 被部分截断后继续计算 AR loss；消除坏样本随机替换和 `None` 静默过滤；让 Query-off action-only 不投影任何 vocabulary token；补齐 grounding camera 与 mixed token 指标一致性。
- 涉及文件：`utils/training_tokenization.py`；`utils/load_training_dataset.py::tokenize_vision_language_inputs/StreamingLeRobotSampleDataset/VQADataset/custom_collate_fn`；`utils/dataset_adapters.py::tokenize_future_difference_message/LeRobotV3FutureDifferenceDataset`；`utils/dataset_manifest.py`；`model/qwen_vl_backbone.py`；`train_vla.py::batch_token_metrics`；对应定向和回归测试。
- 配置开关：继续使用外部 `max_length`；dataset entry 可选 `max_transient_retries`，默认 2，表示首次读取之外最多重试两次。没有新增静默 skip 开关。
- 默认状态：teacher-forced v2/v3/VQA target 始终先以 `truncation=False,padding=False` 完成一次 processor 调用，再验证完整 assistant content 和 chat termination 均落在 `max_length` 内；不完整即抛 `DatasetIntegrityError`。Query-off action-only 在 Transformers 4.57.1 conditional wrapper 上传空 LongTensor `logits_to_keep`。
- 实现来源：基于原仓库 v2/VQA loader 和前一轮 v3 精确边界实现适配；零长度 logits 直接使用 Transformers 4.57.1 `Qwen3VLForConditionalGeneration.forward` 官方 `Union[int,Tensor]` 索引接口，不复制 Qwen forward。
- 原始实现位置或外部来源：复用 `utils/dataset_adapters.py` 原 v3 target/termination 探针、Qwen chat template、`Qwen3VLForConditionalGeneration` conditional wrapper 和现有 final RMSNorm hidden 提取；公共边界、错误类型、重试器和 metric schema 为本仓库自定义抽取。
- 具体设计：`utils/training_tokenization.py` 是 assistant content/termination 边界与 token metric schema 的唯一实现。任何 `C+T>max_length` 均在 padding/collate 前报 dataset entry、episode/frame/sample、C、原始 content/T、termination、总长度、max length 和实际可保留 T。永久 schema/target/action/state/stats/图像错误立即失败；仅 timeout、EINTR、EAGAIN、ESTALE、ETIMEDOUT 可对同一 index 有限重试，耗尽后保留原 cause。v2、v3 与 VQA 均不随机换样本；collate 对 `None` 或空 batch 立即失败。
- 输入与输出：target 存在时 labels 只覆盖完整 assistant content 加标准 termination，P 全为 `-100`。所有 token metric 由 `TOKENIZATION_METRIC_SCHEMA` 定义，并配套 `<metric>_valid`；mixed batch 缺失项用同 dtype 零占位但 validity=false，真实零仍 validity=true。logger 只聚合 valid 样本，count 输出 mean/min/max，bool 输出 ratio。
- 与原有流程的关系：Difference Query C/Q/T/P、HF shift、三种 loss、AR-only Action Expert 隔离、AR→joint、action mask、q01/q99 和 checkpoint kind 不变。v2 legacy ECoT/FAST target 内容仍按原逻辑构造，只把随后可能发生的部分截断改为 fail-fast。Query-off 仍保留二维 mask、conditional wrapper、FA2/eager、视觉 scatter、DeepStack、mRoPE、PEFT hook 和完整 hidden states；AR/joint 继续正常计算 LM logits/CE。
- 关闭功能后的行为：本轮是训练数据正确性修复，没有允许恢复旧静默截断/换样本的开关；不选择 Query-off action-only 时不会使用空 logits 索引。
- 验证方式：严格 RED/GREEN；覆盖 1152→180 复现、超长 C/T、单词/JSON/termination 中途裁剪、刚好适配、v2/v3 同边界、永久与瞬时 I/O、VQA/None/空 batch、grounding camera 成员与顺序、mixed validity/masked aggregation、official `[B,0,V]` logits、final hidden/action golden、PEFT 和真实 processor+tiny vision scatter/DeepStack/mRoPE。
- 已知限制：空索引仍会调用 `lm_head` 一次，但输入 shape 为 `[B,0,H]`，输出为 `[B,0,V]`，不执行任何 token 的 vocabulary projection。真实 2B、GPU FA2、多 GPU、完整数据扫描、正式训练和 LIBERO rollout 均未在本轮执行。

## LeRobot v3 Future Difference 数据 Adapter 与 Token 边界

- 日期：2026-09-02
- 修改目的：在不改变既有 LeRobot v2 默认行为的前提下，增加可与 v2 混合训练的 LeRobot v3 future-difference 数据入口，并对 action horizon、量化归一化、AR target schema 和 token 截断实施 fail-fast 边界。
- 涉及文件：`utils/dataset_adapters.py` 的 `LeRobotV3FutureDifferenceDataset`、target/message/token boundary 和 registry；`utils/dataset_spec.py` 的共享 schema/q01-q99 解析；`utils/dataset_manifest.py`；`utils/load_training_dataset.py` 的 dataset builder、`EpochGroupedSampler`、旧 v2 `max_length` 透传和 `custom_collate_fn`；`policies/reasoning_vla_policy.py`；`utils/cli_options.py`、`train_vla.py`、`dataset2feature.yaml` 及对应测试。
- 配置开关：VLA entry 的 `dataset_adapter`，合法值为 `lerobot_v2`、`lerobot_v3_future_difference`；CLI `--max_length`、可选 `--dataset_sample_ratios` 和 `--dataloader_num_workers`；entry 可选 `max_transient_retries` 默认 2。
- 默认状态：已有 `dataset_type: vla` entry 未写 `dataset_adapter` 时使用 `lerobot_v2`；公共 CLI 保留原有 `--max_length=1200` 兼容默认值，正式模板要求显式提供审计后确定的 `MAX_LENGTH`，adapter 内不另设新的固定长度；`--dataset_sample_ratios` 默认 `None` 并使用 YAML ratio，显式值和 YAML 值都必须位于 `(0,1]`。只有显式选择 v3 adapter 才启用新数据流。
- 实现来源：原仓库适配 + 外部索引规则参考 + 自定义新增。
- 原始实现位置或外部来源：复用 `utils/load_training_dataset.py::tokenize_vision_language_inputs` 的旧 v2 路径、`utils/normalization.py::min_max_norm/min_max_denorm`、原 `StreamingLeRobotSampleDataset` 和 concat/dataloader 构造；v3 的 `steps_data_index.pkl` 与 `source_data_uri` 定位规则仅参考 sibling 仓库 commit `3824d36cdf76bf0a9d537635de92a38f3920e9a3` 的 `Flow-image-generation/stage06_flow/index.py`，未复制其训练架构或数据处理实现。registry、严格 schema、列级 LRU、loss 分支、mask 和 token 边界为本仓库自定义实现。
- 具体设计：初始化时严格读取并验证 `meta/steps_data_index.pkl`、`meta/stage05_episode_mapping.jsonl` 和 `meta/tasks.parquet`，按 `(episode, frame)` 通过 mapping 的 `source_data_uri` 定位真实 Parquet 行；每个 worker 的文件/列 LRU 上限为 1，避免同时驻留多个约 160 MB 的三相机 episode 表。action chunk 另以 Arrow scalar/column 构造单 episode 轻量 frame-action 索引，禁止为每个样本把含图像的整表 `to_pylist()`。`EpochGroupedSampler` 以 `seed+epoch` 确定性打乱 episode/sample 单元，episode 内按 frame 顺序读取；相同 seed/epoch 在 checkpoint resume 时重建相同次序，同时提高单文件缓存命中率。v3 未显式设置 worker 数时使用 4，legacy 数据保持原 24；正式模板显式 4、smoke 显式 1。v2、v3 和 VQA 的永久读取/数据错误直接带样本标识抛出，不随机换样本；白名单瞬时 I/O 只重试相同 index。`sample_ratio` 使用固定 seed 42，ratio=1 发布全部 310743 个 step。vlm 只选取三路当前图像、task 和 `train_data` 列且不打开 stats；action 只选取图像、task、state、actions 并不读取/解析 `train_data`；joint 读取两组字段。`build_future_difference_message` 是 v3 训练和 direct inference 唯一 user prompt builder，固定 camera 顺序、图像/文本排列、task `strip()`、`<TASK> ... </TASK>` 与 assistant generation prefix；训练只额外附加 canonical assistant target。真实同一样本两条路径 C 均为 177 token 且逐 token 相同，原位置 169 的 `<\TASK>` 差异已消失。
- 输入与输出：`train_data` 必须是非空 JSON object，键集合严格为 `Task_temporal`、`Spatial_motion`、`Contact_interaction`、`Object_constraints`，按该顺序输出紧凑 UTF-8 JSON，值均为非空字符串。action/joint 使用当前 state 和同 episode 的真实连续 `[t,t+H-1]` actions，`H=action_horizon` 为外部正整数参数，且不得超过 Action Expert `max_seq_len`；episode 尾部补零，不重复末动作。q01/q99 归一化后输出 `observation.state [1,64]`、`state_mask [1,64]`、`action [H,64]`、`action_mask [H,64] bool`，无效位置为零；adapter、manifest、Action Expert config 和 batch H 不一致时 fail-fast。正式模板和本 smoke 仍显式使用 H=32。
- 与原有流程的关系：concat builder 逐 entry 解析 adapter，允许 v2/v3 混合；`ObjectiveRequirements` 是 target/action/state/stats/FAST 依赖的唯一事实来源。v3 从不需要 FAST；v2 action-only 不需要 target/FAST；v2 AR-only 必须配置动作无关的预计算 `target_text_field`，否则在 dataset 构造前失败；满足该条件时不初始化 FAST、不加载 stats、不投影 state/action 列。v2 joint 的原 ECoT+离散动作 target 仍依赖 FAST，语义不变。主 VLM processor 仍由 builder 统一加载。外部 `max_length` 从 CLI 经 `train_vla.py` 传到 builder/adapter；公共 tokenizer 签名和默认 1200 保持兼容，但 v2/v3 teacher-forced 路径都先无截断编码再验证完整 target。legacy 样本集合仍逐 epoch 确定性随机化，v3 使用随机 episode 顺序加 episode 内连续 frame；这是为可恢复顺序和 Parquet I/O 明确引入的采样次序变化，不改变每 epoch 样本集合。collate 对共享 token metric 全量补齐，并以独立 validity mask 区分真实零与缺失占位。
- 关闭功能后的行为：不选择 `lerobot_v3_future_difference` 时继续使用原 `LeRobotDataset` 和 `StreamingLeRobotSampleDataset`；v2 action/joint 的图像、ECoT/action token 和 loss 行为保持旧语义，只有显式 AR-only requirements 会关闭无关 state/action/stats/FAST 依赖。未提供 ratio override 时 YAML sampling 行为不变。
- Token 边界：v2/v3 teacher-forced processor 均固定 `truncation=False,padding=False` 得到真实序列；assistant content token 必须作为精确连续 span 存在，assistant termination 由标准 chat template 的显式探针确定。T 定义为 content（v3 为 canonical JSON，v2 可为显式文本或原 ECoT/FAST 文本）加完整 termination；当前真实 Qwen3-VL template 的 termination token ID 为 `[151645,198]`，分别是 `<|im_end|>` 和结构性换行，两者均按标准 SFT 语义监督。P 只包含右侧 padding，labels 全为 `-100`。输出分别记录 `context_token_count`、`json_content_token_count`（通用含义为 content）、`chat_termination_token_count`、`target_region_token_count`、`padding_token_count`、原始/保留 content 数、有效监督数和截断分类及各自 validity。任何 content 或 termination 被裁均作为 `target_truncated` 带稳定样本 ID 和完整长度事实报错；不得切掉尾部继续算 AR loss。action-only 不输出 labels，target 指标 validity=false。
- 验证方式：严格 TDD；微型两 episode Parquet 覆盖 registry/default/unknown、loss 列隔离、无 stats AR、canonical 错误 ID、tail zero pad、跨 episode、stats shape/finiteness/span、7 维 norm-denorm、H=1/10/16/32、termination/P 边界、全 mask 防护、确定性采样、collate、v2 默认 tokenizer、v2/v3 mixed builder、requirements/FAST 条件加载、v2 投影列与 stats 访问 spy、ratio CLI/override。真实发布包使用本地 Qwen3-VL processor 对两个样本验证训练/direct C=177 且逐 token 相同、termination `[151645,198]`、T/P labels、HF shift 与 Query 不可见 T；真实 v3 policy 从匹配 manifest 构造并解析三路 camera、7 维 state/action 和共享 q01/q99。
- 已知限制：adapter 固定该发布包的 7 维 state/action 与 64 维模型 padding，但 action horizon 不再固定；已用实际 Qwen processor 和 policy metadata 路径验证，未加载 2B 模型权重或启动 GPU 训练。

## 共享 Dataset Spec、Policy 归一化与 Resolved Manifest

- 日期：2026-09-02
- 修改目的：保证训练 adapter、direct-action policy 与 checkpoint 对 camera/字段/维度/q01-q99/action horizon 使用同一份解析事实，并让恢复训练可检查数据语义漂移。
- 涉及文件：`utils/dataset_spec.py`、`utils/dataset_manifest.py`、`utils/dataset_adapters.py`、`utils/load_training_dataset.py`、`policies/reasoning_vla_policy.py`、`model/reasoning_vla_model.py`、`train_vla.py`、`tests/test_dataset_spec_policy.py`、`tests/test_dataset_manifest.py`。
- 配置开关：复用 entry 的 `dataset_adapter`、camera/state/action/target/stats 字段和外部 `action_horizon`；没有新增猜测字段名的 fallback。
- 默认状态：v2 继续从 `LeRobotDatasetMetadata` 读取原 camera/stats/schema；只有显式 v3 adapter 读取 `meta/info.json` 与 `meta/stats_gr00t.json::statistics.state/actions.q01/q99`。
- 实现来源：原仓库 normalization 工具适配 + 自定义共享 schema/manifest。
- 具体设计：`ResolvedDatasetSpec` 是训练与 policy 的唯一解析接口；v3 stats 只由 `validate_v3_quantile_stats` 实现一次，严格拒绝缺字段、非 7 维、NaN/Inf 或 `q01>=q99`。v3 policy 只取配置顺序的当前三路图像和真实 7 维 state，使用共享 state q01/q99 normalization；模型 action 输出先裁至 7 维，再由 `denormalize_actions` 使用共享 action q01/q99。禁止 sample/batch/episode 临时统计。v2 policy 保留 metadata 行为。
- 输入与输出：dataset builder 给每个 entry 生成独立 manifest，记录 entry/path/type/resolved adapter、task/target/state/action 字段、loss requirements、`camera_keys` 与 `grounding_camera_keys` 的各自顺序、维度、H、stats 相对路径/键/文件 SHA-256、规范化后的 state/action q01/q99、normalization、ratio、eligibility exists/used/source 和 data version。q01/q99 转为确定性 JSON 数组；manifest 以排序 JSON 计算 SHA-256 content hash，不写凭据或环境秘密。同一路径 stats 内容变化会同时改变 `stats_sha256` 和 manifest hash。
- 与原有流程的关系：主进程启动时打印 manifest，并保存到 `<output>/resolved_dataset_manifest.json`；每个 `step-*` 和 `latest-model-optimizer-lr` checkpoint 同样保存完整内容与 hash。`--resume_training` 对完整解析结果比较，entry 顺序、adapter、普通 camera 顺序、grounding camera 内容及顺序、字段/维度、stats identity/q01/q99、normalization、H、sample ratio 等变化均 fail-fast；sample ratio 被明确视为训练数据选择语义。resume 与 direct policy 共用 `DATASET_SEMANTIC_FIELDS`；policy 从 checkpoint 选择唯一同名 entry，并在 state normalization/action denormalization 前逐项校验当前 spec，mismatch 显示 checkpoint/current 值和 entry。新 checkpoint 缺 manifest 默认失败。legacy 仅能通过 `--allow_legacy_checkpoint_without_manifest` 显式 opt-in，并输出强警告。AR→joint 是 warm start 而非 resume，不做跨 loss manifest 相等要求。
- 验证方式：真实 v3 metadata 加匹配 checkpoint manifest 的 policy 构造、固定 normalized action 反归一化、v2 metadata 回归、单 v2/单 v3/mixed manifest、同路径 stats 原地变更、确定性序列化/hash、save/resume、policy 唯一 entry/legacy opt-in，以及 target/camera/stats/H 冲突测试。
- 已知限制：self-consistency policy 文件仍沿用其原有独立实现；本次推理修复范围是生产 direct-action `policies/reasoning_vla_policy.py`。

## Query-Conditioned AR 与联合训练目标

- 日期：2026-09-02
- 修改目的：把 teacher-forced future-difference AR、Flow Matching action 和两者联合训练定义为三个互斥的训练模式，保证 loss 权重、梯度路径、冻结模块和同步 optimizer step 的语义可验证且可恢复。
- 涉及文件：`model/reasoning_vla_model.py::ZR0Model.forward`、`model/qwen_vl_backbone.py::QwenVLBackbone.forward`、`train_vla.py` 的 optimizer/scheduler/step/logging helpers 与训练循环、`utils/cli_options.py`、`tests/test_loss_training_interface.py`、`tests/test_qwen_vl_backbone.py`、`tests/test_train_resume.py`、`tests/test_difference_query_sequence.py`、`tests/test_zr0_difference_query.py`。
- 配置开关：复用单一 `--loss_type {vlm,action,vlm_and_action}` 和 `--vlm_loss_weight`、`--action_expert_loss_weight`；新增可选 `--adam_beta1`、`--adam_beta2`、`--adam_epsilon`、`--warmup_ratio`，复用已有 `--max_train_steps`。不增加重复的 loss 布尔开关。
- 默认状态：默认 loss 模式和两个权重仍为原来的 `vlm_and_action`、`1.0/1.0`；AdamW 默认 `(0.9,0.95)`, epsilon `1e-6`，未给 warmup ratio 时保留旧 8% 且单进程最多 20,000 step 的算法。Difference Query 本身仍默认关闭。
- 实现来源：基于原仓库训练循环、Qwen causal LM loss 和 Flow Matching Action Head 修改。
- 原始实现位置或外部来源：复用 `Qwen3VLForConditionalGeneration.forward(labels=...)` 的原生 shift causal LM loss、`FlowmatchingActionHead.forward` 的 masked FM loss、`utils/training_checkpoint.py` 的 DeepSpeed 保存/恢复；没有复制外部训练实现。
- 具体设计：`vlm` 严格计算 `total_loss = vlm_loss_weight * ar_loss`；模型构造前同时要求 `tune_vlm=True`、`tune_action_expert=False` 且无 Action Expert 权重路径，并完全不实例化 Action Expert。`action` 严格计算 `total_loss = action_expert_loss_weight * flow_matching_loss`；`vlm_and_action` 为两项加权和。输出保留兼容键并增加 AR/FM/total 和 weighted alias。活跃权重必须有限且大于零。AR/joint 每个样本 labels 至少一个有效 token；action/joint 在 backbone 前检查 state/action/mask 与 Action Expert H 一致。Difference Query 配合 detach 时拒绝 action/joint。forward 的可选 legacy mode 参数只允许与构造期 `self.loss_type` 完全一致，`None` 使用构造值；任何动态切换在计算前报出 constructed/requested mode。checkpoint kind、dataset requirements、active components 和训练循环均以构造期 mode 为准。
- 输入与输出：AR/joint 的 Qwen labels 监督完整 assistant T（canonical JSON + `<|im_end|>` + 换行）；HF causal shift 由最后一个 Query hidden 预测第一个 T。AR 模型没有 Action Expert 参数。action/joint 的 Action Expert 仍额外接收独立 state encoder 的 proprioception，因此 Query 是“VLM 上下文 Query bottleneck”，不是全部控制条件的瓶颈。
- 与原有流程的关系：Query C/Q/T/P 重排和 block mask 不变：Q 只看 C/Q，T 只看全部 Q 和过去 T，不看 C/P/未来 T。Query-on action-only 使用 Qwen multimodal base model，避免 LM head；Query-off action-only 保持官方 conditional wrapper、二维 mask、FA2/eager、视觉 scatter、DeepStack、mRoPE 与 PEFT hook，但移除 labels 并传空 LongTensor `logits_to_keep`，LM head 收到 `[B,0,H]` 并返回 `[B,0,V]`，不计算任何 token 的 vocabulary projection 或 CE。AR/joint 不使用该优化，继续计算原生 CE。optimizer 只接收 `requires_grad=True` 参数。batch token 边界和截断统计在同步 optimizer step 的日志间隔记录到 TensorBoard/W&B。`max_train_steps` 同时限定 scheduler horizon、progress、保存和恢复位置，只在同步 optimizer boundary 增加 durable global step。
- 关闭功能后的行为：不启用 Difference Query 时仍走原二维 attention mask 和完整 VLM prefix 动作条件；不设置新增 optimizer/warmup 参数时保持旧默认。未选择 AR-only 时 Action Expert 路径按所选 action/joint 模式运行。
- 验证方式：测试覆盖三个 loss 公式和 1/5 权重、各分支有限非零梯度、AR 无 Action Expert 分配/参数/optimizer 项、冲突 CLI/构造拒绝、forward mode 不一致拒绝、非法权重与缺失监督、HF shift、C/Q/T/P、H 冲突、Query detach、action-only 无 labels、Query-off PEFT wrapper 与空 Tensor `logits_to_keep`、零长度/完整 logits 的 final hidden 和固定种子 action golden、同步 max-step/保存/恢复、token metric validity 和 masked 日志。
- 已知限制：Difference Query `.generate()` 仍明确抛 `NotImplementedError`；本任务验证 teacher-forced AR，不声称验证自由文本生成质量。真实 2B GPU smoke 因共享 GPU 满载未启动。

## Future Difference 全量只读审计

- 日期：2026-09-02
- 修改目的：在正式训练前量化 assistant target 长度与 `[t,t+31]` action chunk 对上游语言动作区间的覆盖关系，避免凭经验选择 `max_length` 或猜测 eligibility。
- 涉及文件：`utils/future_difference_audit.py`、`scripts/audit_future_difference_lengths.py`、`scripts/audit_future_difference_intervals.py`、`tests/test_future_difference_audits.py`。
- 配置开关：两个脚本均为独立只读命令；长度审计要求外部 `--max-length` 和 `--processor-path`，interval 审计要求外部 `--action-horizon`。训练不会自动运行审计。
- 默认状态：不修改数据、不影响训练默认行为，也不写 checkpoint；审计结果仅输出 JSON 到 stdout，进度写 stderr。
- 实现来源：自定义新增；复用 v3 adapter 的 canonical target、message builder 和精确 token boundary API。
- 原始实现位置或外部来源：episode/file 范围来自发布包 `meta/info.json` 与 episode metadata；interval 根目录由 `meta/stage05_merge.json::stage05_dir` 定位，new/old episode 由 `meta/stage05_episode_mapping.jsonl` 映射，上游 `training_samples*.jsonl` 只读取 semantic anchor 与闭区间字段。
- 具体设计：长度审计只读取 `episode_index/frame_index/task_index/train_data`，以 metadata 尺寸创建 dummy 图像，不打开真实图像/state/action；分别统计 C、JSON、termination、完整 T、保留/监督、投影总长和截断分布。interval 审计按闭区间规则分类，并固定 hash dataset 侧 `meta/info.json`、`steps_data_index.pkl`、`stage05_episode_mapping.jsonl`、`stage05_merge.json`，以及实际解析的 annotation `status.json`/`training_samples*.jsonl`；每项记录相对路径、字节数和 SHA-256，聚合为 audit identity。明确不 hash Parquet、图像或视频。
- 输入与输出：输入只涉及上述元数据和文本；输出为机器可读 JSON 汇总。范围固定为发布包全部 310,743 帧、1,881 episodes、14 tasks。当前 release 的 `meta/info.json::features` 不含 `training_eligible`，manifest 固定记录 `exists=false, used=false, source=unavailable_in_release`；YAML 声明不能覆盖实际 metadata。使用发布包全部可读取样本，未按 upstream `training_eligible` 过滤，也不扫描 Stage05、读取或推断 eligibility。
- 与原有流程的关系：审计不参与 sample selection。正式训练模板仍要求外部 `MAX_LENGTH`；必须根据本审计结果显式填写。
- 关闭功能后的行为：不运行脚本即没有额外 I/O 或计算。
- 验证方式：synthetic 闭区间/off-by-one、四分类、unavailable、只读快照、percentile、termination 截断和 metadata identity 变更测试 6/6。真实全量长度审计：310,743 样本在 1200 下 overflow/input-only/target truncation 均为 0；JSON p50/p99/max=471/521/566，termination 恒为 2，完整 T 与监督 p50/p99/max=473/523/568，投影总长 max=746。真实 interval：exact 374、full 41,383、partial 268,986、none/unavailable 0；identity 覆盖 3,766 个 metadata 文件，SHA-256 `cbcee48a48b3eb2d9bfb62cd846eccd1ec6af5d73065fe57cdeba7013c1b85f5`。
- 已知限制：长度结论绑定本地 `Qwen3-VL-2B-Instruct` processor、三路 224x224 请求尺寸和当前 canonical chat template；更换 processor/template/图像尺寸后必须重跑。interval 分类使用 padding 前 nominal 32 步区间，episode 尾部实际 action mask 另由 adapter 控制。

## Query AR/Joint Checkpoint 与启动模板

- 日期：2026-09-02
- 修改目的：提供可复现的 AR 一步+恢复、joint warm-start+恢复 smoke，以及在启动正式训练前强制核对训练规模、数据混合和论文参数的模板。
- 涉及文件：`scripts/run_query_ar_joint_smoke.sh`、`scripts/run_query_ar_joint_formal.sh`、`experiments/query_conditioned_ar_joint_smoke/experiment.md`、`tests/test_query_ar_joint_checkpoint.py`、`tests/test_query_ar_joint_launchers.py`。
- 配置开关：smoke 通过 `ZR0_*` 环境变量和四个显式 stage 名称控制；正式模板通过 `MODEL_PATH/OUTPUT_DIR/EXPERIMENT_DOC/MAX_LENGTH/EPOCHS/NUM_GPUS/PER_DEVICE_BATCH_SIZE/GRADIENT_ACCUMULATION_STEPS/DATASET_ENTRIES/SAMPLE_RATIOS` 与 W&B 环境变量控制，缺失即退出。
- 默认状态：脚本不会被训练代码自动调用；正式模板不预填 epochs、数据 entries 或 sample ratios。`ZR0_DRY_RUN=1` 只打印脱敏命令。
- 实现来源：基于仓库现有 Accelerate/DeepSpeed launcher 和 `utils/training_checkpoint.py` 适配。
- 具体设计：AR step 1/恢复只保存和加载 VLM、Query、optimizer、scheduler、global step 与未来 joint 所需 Action Expert config，不分配或保存 `action_expert.safetensors`。checkpoint metadata 显式记录 `ar_only/joint/action_only`；新 checkpoint kind 与用途冲突时 fail-fast，legacy 无 kind 才进入带日志的保守兼容路径。joint step 1 只把 `ar_only` checkpoint 作为 VLM/Query 来源，按保存 config 和固定 seed 新建 Action Expert；smoke/formal launcher 均验证 kind，正式 launcher 还验证 Query enabled 和 Nq=32。joint 自身 resume 必须把同一 `joint` checkpoint 同时作为 VLM/Query 与 Action Expert 来源，再恢复完整状态。其余 formal 参数约束不变。
- 输入与输出：输入为外部模型、数据 entry/ratio、审计决定的 max length 和实验文档；输出分别写 AR/joint smoke 或正式 output directory。smoke checkpoint 明确不是正式实验 checkpoint。
- 与原有流程的关系：继续使用现有 `latest-model-optimizer-lr` sidecar 和普通 `save_pretrained`；新增 `zr0_checkpoint_metadata.json` 与 resolved manifest。legacy checkpoint 保持可读，但不会通过缺文件静默猜测为新格式。
- 关闭功能后的行为：不执行脚本则无行为变化。
- 验证方式：tiny CPU 验证 AR 无 Action Expert 文件、VLM/Query 精确恢复、joint 从 AR warm-start 的 fresh Action Expert、kind 用途冲突、joint model/optimizer/scheduler/global step 完整恢复。launcher 17 项测试覆盖命令、必填变量、1024、ratio、W&B online、AR kind/Query sidecar、记录保留和四阶段路径；两个脚本通过 `bash -n`。
- 已知限制：该条目记录时真实 2B smoke 尚未运行；后续四卡真实 smoke/save/resume 结果见本文件顶部 2026-09-03 条目。正式训练仍未启动。

## LIBERO 评估运行说明

- 日期：2026-09-01
- 修改目的：为当前服务器已有环境补充可直接执行、可追溯且不会覆盖历史结果的 ZR-0-LIBERO 评估说明。
- 涉及文件：`simple_scripts/eval_libero.md`、`reference.md`
- 配置开关：无；本次只修改文档。
- 默认状态：不改变任何代码、模型或评估默认行为。
- 实现来源：直接复用仓库原有评估实现。
- 原始实现位置或外部来源：`server.py` 的 `deploy`；`evaluation/libero_eval/run_libero_eval.py` 的 `Args`、`eval_libero` 和 `_get_libero_env`；`policies/reasoning_vla_policy.py` 的 `ZR0Policy.infer`；`utils/image_tools.py` 的 `resize_with_pad`；当前仓库内 LIBERO 的 `libero/libero/envs/env_wrapper.py`。
- 具体设计：文档将运行拆分为已有资源确认、配置说明、启动前检查、四 GPU 服务端启动、四 suite 客户端启动、进度监控、结果汇总和按 PID 清理。每次运行创建新的时间戳结果目录，并记录模型来源与 revision、代码 commit、未提交修改、LIBERO 配置与 GPU 快照。
- 输入与输出：输入为官方 ZR-0-LIBERO checkpoint、两个相机图像、8 维机器人状态和任务语言；输出为 7 维动作 chunk，以及日志、rollout 视频和 JSON/CSV/Markdown 成功率汇总。
- 与原有流程的关系：只对已有 server-client 评估流程进行说明，不新增包装器，不修改数据流或启动参数。
- 关闭功能后的行为：不适用；文档不会被运行时代码加载。
- 验证方式：检查 Markdown diff、提取 Bash 代码块执行 `bash -n`、核对所有绝对路径、运行客户端 `--help` 核对 Tyro 参数，并对照代码检查 GPU/端口映射、图像处理、动作处理、随机种子、rollout 数和成功率口径。
- 已知限制：命令针对当前服务器的固定绝对路径和四 GPU 并行评估；换机器或换端口时必须同步调整服务端、客户端和 LIBERO 配置。

## 可开关 Difference Query 动作条件

- 日期：2026-09-02
- 修改目的：把可变长 Qwen 图文上下文压缩为固定数量 learnable Difference Query，使动作专家在实验臂中只读取 Query hidden，并提供无 Query 的 FA2/eager 基线和 SDPA backend 控制组。
- 涉及文件：`model/difference_query.py` 的 `DifferenceQuery`、`build_difference_query_sequence`、`resolve_difference_query_config`、`save_difference_query_artifacts`；`model/qwen_vl_backbone.py` 的 `QwenVLBackbone`；`model/reasoning_vla_model.py` 的 `ZR0Model`；`utils/cli_options.py`、`utils/training_checkpoint.py`、`train_vla.py`、`server.py`、`policies/reasoning_vla_policy.py`、`policies/reasoning_vla_policy_sc.py`；`scripts/run_libero_wo_ecot_pt.sh`；相关测试与文档。
- 配置开关：`use_difference_query: Optional[bool]`、`num_difference_queries: Optional[int]`、`vlm_attention_backend: Optional[str]`；CLI 为 `--use_difference_query` / `--no-use_difference_query`、`--num_difference_queries`、`--vlm_attention_backend`。
- 默认状态：关闭。无 Query checkpoint 且 CLI 未显式开启时，继续使用原二维 attention mask、原 Qwen `input_ids` 调用和原 FA2/eager 自动 backend；不创建 Query 参数，不增加 loss，不改变动作专家输入输出接口。
- 实现来源：基于仓库原有模块修改 + 自定义新增。
- 原始实现位置或外部来源：复用 `model/qwen_vl_backbone.py::QwenVLBackbone.forward` 的 Qwen forward/final RMSNorm 和 `model/reasoning_vla_model.py::ZR0Model` 的动作专家调用、保存/加载路径；复用 Transformers 4.57.1 `Qwen3VLModel.get_rope_index`、`inputs_embeds` 视觉 scatter、DeepStack 和 SDPA 四维 bool mask 支持。Difference Query 参数、C/Q/T/P 重排、block mask、三态解析和 checkpoint 完整性逻辑为本仓库自定义实现，没有复制外部项目代码。
- 具体设计：Query 参数为精确 `[Nq,H]`，隔离 RNG 后以 FP32 初始化，forward 转成 token embedding dtype/device。训练按首个受监督 label 重排为 `[C,Q,T,P]`，direct inference 为 `[C,Q,P]`；ID 0 只作 mRoPE/视觉对齐占位符，真实 embedding 由 Query 覆盖。序列索引和四维 block mask 使用广播、gather 和张量 region 映射构造，不逐样本调用 `.item()`；mask 实现 C causal、Q 读取 C/Q 且 Q 双向、T 只读取 Q/T causal、P 行列全屏蔽。final RMSNorm 后只 gather Query hidden 给动作专家。保存和加载均拒绝 NaN/Inf Query 权重；两个不同的本地加载目录必须同时为 legacy，或同时声明一致的 Difference Query 架构，禁止将单侧 Query checkpoint 与无 sidecar 的 legacy checkpoint 混合；双侧启用时权重也必须一致。同一路径会先去重，未提供 action checkpoint 路径时只检查 VLM 目录。
- 输入与输出：输入仍为 Qwen `input_ids/attention_mask/pixel_values/image_grid_thw` 和可选 labels；Query VLM 内部序列长度为 `L+Nq`。动作专家只收到 `backbone_embeddings [B,Nq,H]` 与全 True `action_expert_cross_attn_mask [B,Nq] bool`。Flow Matching、DiT、state/action encoder 和 decoder 结构不变；loss 组合由本次新增的三个显式训练模式控制。
- 与原有流程的关系：训练、普通保存、DeepSpeed 导出/恢复、`from_pretrained`、policy/server、direct-action 共用一个 Query 配置解析器。Query checkpoint 自动启用；disabled config 是明确架构声明，不能被 CLI 开启，且 hidden size 仍须匹配实际 VLM；只有完全没有 Query config/weight 的老 checkpoint 才允许显式开启后随机初始化。两个加载目录同时含 Query 时要求 config 和权重完全一致。Query 模式固定 SDPA并打印实际 backend；`baseline_sdpa` 在无 Query 下控制 backend 影响。Query-on action-only 无 labels时把完整 user prompt 视为 C，并调用 Qwen multimodal base forward 获取 final normalized Query hidden，不调用 LM head 或计算 CE。Query-on 的 `vlm`/`vlm_and_action` 仍走 conditional teacher-forced loss。Query-off 的 AR/joint 路径复用 `Qwen3VLForConditionalGeneration.forward` 并计算原生 CE；action-only 移除 labels 并设置空 Tensor `logits_to_keep`，沿用 conditional model 与 PEFT wrapper/hook，LM head 输入/输出序列长度均为 0。两条 action 条件路径的 final normalized hidden 语义在测试容差内一致，但计算量不同。Query 模式的 wrapper、实际 `backbone.model.generate()` 和 PEFT base model generate 均使用具名方法拒绝生成。
- 关闭功能后的行为：仍不创建 Query，保持二维 mask、完整 VLM prefix 动作条件、原 conditional wrapper 调用层级和 checkpoint 加载兼容性；action-only 不计算 CE，也不对任何 hidden token执行 vocabulary projection，但 conditional wrapper 仍会以零长度输入调用 LM head。`baseline_sdpa` 只改变 attention backend。新保存的 baseline checkpoint 只额外包含声明 disabled 的 `difference_query_config.json`。
- 验证方式：unittest 覆盖 CLI 隔离解析 `None/True/False/Nq`、disabled/enabled/损坏 checkpoint、双目录 Query/legacy 混用的双向拒绝及双 legacy/双一致声明/同目录/单目录允许矩阵、Nq=8/32/64 的 mask/真实 tiny Qwen/direct-action、隔离 RNG、最大长度 CPU C/Q/T/P mask、关闭路径 HEAD 参数/PEFT hook/final hidden/mask/固定种子动作 golden、Query-on action-only LM-head 拒绝、真实 tiny Qwen 的 T/C 因果隔离和冻结 Qwen 后 Query 梯度、PIL RGB 实际 processor/vision/DeepStack/mRoPE 对齐、tiny Qwen+tiny Action Expert backward/direct denoise、真实 Qwen+processor+Action Expert 完整 checkpoint round-trip、生成入口拒绝、单进程未初始化 distributed 构造与 step 同步。CUDA BF16 tiny Query 已捕获 efficient SDPA kernel 并记录有限输出/峰值显存；DeepSpeed ZeRO-2 已通过独立 `torchrun` world-size=1 和 2-rank 的更新、保存、重构、恢复及继续一步测试，复用生产 helper 并核对 Query sidecar、optimizer、scheduler、global step、跨 rank Query 一致性和有限性。另运行 CLI help/实际解析、三臂 launcher dry-run、Python/Bash syntax、`git diff --check`。未启动正式训练或完整 LIBERO rollout。
- 已知限制：Query 模式不支持 subtask/autoregressive generation；teacher-forced target 前向仍支持。Query+SDPA 相对 FA2 baseline 有 backend 混杂，必须联合 `baseline_sdpa` 解读。Transformers 4.57.1 的空 Tensor 索引可安全保持 conditional wrapper/PEFT/多模态语义并生成 `[B,0,V]`；它没有完全绕过 `lm_head` 函数调用，只是该调用处理零个 token。Query-off 与 Query-on action-only 的调用层级和计算量不同，运行时间或显存不能直接解释为 Query 本身带来的加速。四卡真实 Qwen3-VL-2B 短步生产恢复已在后续完成；正式训练和成功率评估仍未执行。

## 跨 epoch 恢复与单进程 step 同步修复

- 日期：2026-09-02
- 修改目的：修复恢复训练在后续每个 epoch 重复跳过相同 batch 前缀的问题，并使单进程训练不依赖已初始化的 `torch.distributed` process group。
- 涉及文件：`train_vla.py` 的 `should_skip_resumed_batch`、`synchronize_global_step` 和训练循环；`utils/training_checkpoint.py` 的生产保存/恢复与 PyTorch 2.6 safe-globals 注册；`tests/test_train_resume.py`、`tests/test_difference_query_cuda_zero.py`、`tests/zero2_checkpoint_worker.py`。
- 配置开关：沿用原有 `--resume_training`，没有新增默认行为开关。
- 默认状态：fresh training 的数据遍历、optimizer/scheduler 次序和 global-step 语义不变。
- 实现来源：基于仓库原有恢复循环修改。
- 原始实现位置或外部来源：`train_vla.py::train` 原有 `resume_epoch/resume_batch_idx` 计算、DeepSpeed optimizer 恢复和 scheduler state 恢复逻辑；没有使用外部实现。
- 具体设计：完整跳过 `epoch < resume_epoch`；batch 前缀条件收紧为 `epoch == resume_epoch and batch_idx < resume_batch_idx`，后续 epoch 从 batch 0 开始。optimizer step 后仍只在同步梯度边界由主 rank 增加 `global_completed_steps`；单进程直接返回该值，多进程仅在 distributed 可用且已初始化时从 rank 0 broadcast，否则显式报错。DeepSpeed safe-globals、engine optimizer/model 恢复、scheduler 恢复和 global step 提取集中在一个生产 helper，训练和门控测试共同调用，避免 PyTorch 2.6 反序列化列表漂移。
- 输入与输出：输入为恢复出的 global step、dataloader 长度、gradient accumulation 和当前 epoch/batch；输出为正确的数据遍历位置及所有 rank 一致的 global step。
- 与原有流程的关系：不改变 checkpoint 格式、scheduler state 加载、optimizer state 加载、数据 shuffle seed 或三实验臂参数，只修复恢复位置判断和单进程同步边界。
- 关闭功能后的行为：未启用 `--resume_training` 时不执行任何恢复跳过；单进程 fresh training 不再调用 `dist.broadcast`。
- 验证方式：跨两个 epoch 的恢复回归证明只在恢复 epoch 跳过前缀，并核对 global step 从 6 增至 12；单进程未初始化、多进程未初始化报错及多进程已初始化 broadcast 均有单元测试。ZeRO-2 world-size=1 与 2-rank `torchrun` 已实际通过一次更新、生产保存、重构恢复和继续一步。
- 已知限制：本次没有启动正式 DeepSpeed 训练；四卡真实 2B 短步生产恢复已在后续完成，见顶部 2026-09-03 条目。
