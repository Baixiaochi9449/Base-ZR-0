# Implementation Reference

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
- 输入与输出：输入仍为 Qwen `input_ids/attention_mask/pixel_values/image_grid_thw` 和可选 labels；Query VLM 内部序列长度为 `L+Nq`。动作专家只收到 `backbone_embeddings [B,Nq,H]` 与全 True `action_expert_cross_attn_mask [B,Nq] bool`。Flow Matching、DiT、state/action encoder、decoder 和总 loss 公式不变。
- 与原有流程的关系：训练、普通保存、DeepSpeed 导出/恢复、`from_pretrained`、policy/server、direct-action 共用一个 Query 配置解析器。Query checkpoint 自动启用；disabled config 是明确架构声明，不能被 CLI 开启，且 hidden size 仍须匹配实际 VLM；只有完全没有 Query config/weight 的老 checkpoint 才允许显式开启后随机初始化。两个加载目录同时含 Query 时要求 config 和权重完全一致。Query 模式固定 SDPA并打印实际 backend；`baseline_sdpa` 在无 Query 下控制 backend 影响。Query-on 的 action-only/direct 保留 labels 供 C/T 重排，调用 Qwen 原生 multimodal base forward 获取 final normalized hidden，不调用 LM head、生成全词表 logits 或计算 CE；Query-on 的 `vlm`/`vlm_and_action` 仍走 conditional teacher-forced loss。Query-off 的所有 loss/direct 路径则严格复用 `Qwen3VLForConditionalGeneration.forward`，保留 `input_ids`、二维 mask、labels、`output_hidden_states=True`、训练/推理 `use_cache` 和 PEFT wrapper/hook。两条 action 条件路径的 final normalized hidden 语义在测试容差内一致，但计算量不同。Query 模式的 wrapper、实际 `backbone.model.generate()` 和 PEFT base model generate 均使用具名方法拒绝生成。正式 launcher 在 `exec` 前将根目录实验模板写入各臂 `OUTPUT_DIR/experiment.md`，追加实际 arm、时间、W&B 标识和完整命令；resume 保留已有记录并继续追加。
- 关闭功能后的行为：仍不创建 Query，保持二维 mask、完整 VLM prefix 动作条件、原 conditional HEAD 调用层级和参数、loss 数值语义及 checkpoint 加载兼容性。即使 action-only 最终丢弃 VLM CE，Query-off 也有意保留 LM head、全词表 logits 和有 labels 时的 CE 计算，以优先复现历史 HEAD 基线；`baseline_sdpa` 只改变 attention backend。新保存的 baseline checkpoint 只额外包含声明 disabled 的 `difference_query_config.json`。
- 验证方式：unittest 覆盖 CLI 隔离解析 `None/True/False/Nq`、disabled/enabled/损坏 checkpoint、双目录 Query/legacy 混用的双向拒绝及双 legacy/双一致声明/同目录/单目录允许矩阵、Nq=8/32/64 的 mask/真实 tiny Qwen/direct-action、隔离 RNG、最大长度 CPU C/Q/T/P mask、关闭路径 HEAD 参数/PEFT hook/final hidden/mask/固定种子动作 golden、Query-on action-only LM-head 拒绝、真实 tiny Qwen 的 T/C 因果隔离和冻结 Qwen 后 Query 梯度、PIL RGB 实际 processor/vision/DeepStack/mRoPE 对齐、tiny Qwen+tiny Action Expert backward/direct denoise、真实 Qwen+processor+Action Expert 完整 checkpoint round-trip、生成入口拒绝、单进程未初始化 distributed 构造与 step 同步。CUDA BF16 tiny Query 已捕获 efficient SDPA kernel 并记录有限输出/峰值显存；DeepSpeed ZeRO-2 已通过独立 `torchrun` world-size=1 和 2-rank 的更新、保存、重构、恢复及继续一步测试，复用生产 helper 并核对 Query sidecar、optimizer、scheduler、global step、跨 rank Query 一致性和有限性。另运行 CLI help/实际解析、三臂 launcher dry-run、Python/Bash syntax、`git diff --check`。未启动正式训练或完整 LIBERO rollout。
- 已知限制：Query 模式不支持 subtask/autoregressive generation；teacher-forced target 前向仍支持。Query+SDPA 相对 FA2 baseline 有 backend 混杂，必须联合 `baseline_sdpa` 解读。Query-off 与 Query-on action-only 的 Qwen 调用层级和无用 logits/CE 计算量不同，因此运行时间或显存不能直接解释为 Query 本身带来的加速。4-GPU、真实 Qwen3-VL-2B 短步生产恢复和正式三臂成功率仍未执行，不能由官方 97.6% 历史结果推断。

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
- 已知限制：本次没有启动正式 DeepSpeed 训练；4-GPU 真实 2B 短步生产恢复仍待执行。
