# Stage05 多数据集预训练数据审计

- 审计日期：2026-09-03
- 数据集：DROID、MolmoAct Household、MolmoAct Tabletop、RH20T
- 目标代码库：`/opt/data/private/lq/ZR-0`
- 审计方式：只读检查，未修改任何数据集文件

## 1. 结论摘要

**当前不建议直接启动四库联合预训练。** 四个发布目录的 Parquet、episode 元数据和 Stage05 文本标签总体完整，但训练接口、动作语义、异常值、相机同步、许可和数据划分仍有阻断项。

| 数据集 | 文件完整性 | Stage05 标签 | 当前 ZR-0 `lerobot_v3_future_difference` 可直接读取 | 预训练结论 |
|---|---:|---:|---:|---|
| DROID | 通过 | 仅部分覆盖 | 否 | 阻断 |
| MolmoAct Household | 通过 | 全覆盖 | 否 | 补索引和配置后再验收 |
| MolmoAct Tabletop | 通过 | 全覆盖 | 是 | 有条件可用 |
| RH20T | 结构通过，数值/同步有问题 | 全覆盖 | 否 | 阻断 |

最重要的发现如下：

1. 只有 MolmoAct Tabletop 已配置且能通过当前 adapter 初始化；Household 缺 `meta/steps_data_index.pkl`，DROID 和 RH20T 还同时存在视频字段、维度和统计文件不兼容。
2. 四库的动作空间不是天然同一种语义。MolmoAct/RH20T 主要是 7D 末端执行器增量，DROID 当前 `action` 是 8D 关节位置加夹爪；只做分位数归一化不能消除这种控制语义差异。
3. RH20T 有 4 帧极端状态异常，并产生 8 帧极端动作；异常来自上游 RH20T 源轨迹，而不是 Stage05 合并过程。现有 `stats.json` 已被污染并包含 `NaN`。
4. RH20T 的 wrist/exterior2 与主视角存在明显复用和时间偏移。未来差分文本使用三视角时，这会产生跨时刻的视觉条件。
5. DROID 只有 9,507,523 / 27,630,375 帧有 Stage05 标签；若未显式建立训练 eligibility，读取到空标签会直接失败。
6. 所有数据集目前只有 `train` split。必须按 episode、任务和来源建立独立 validation/test，不能随机按帧切分。
7. RH20T 同时含 CC BY-SA 4.0 与 CC BY-NC 4.0 子集。商业用途必须先按许可拆分并完成合规确认。

## 2. 总体规模与混合风险

| 数据集 | Episodes | Frames | FPS | 原始时长 | 表观大小 | 原始帧占比 |
|---|---:|---:|---:|---:|---:|---:|
| DROID | 95,658 | 27,630,375 | 15 | 511.67 h | 368.55 GiB | 84.669% |
| MolmoAct Household | 5,936 | 794,199 | 10 | 22.06 h | 622.20 GiB | 2.434% |
| MolmoAct Tabletop | 1,881 | 310,743 | 10 | 8.63 h | 197.61 GiB | 0.952% |
| RH20T | 8,142 | 3,898,056 | 10 | 108.28 h | 607.48 GiB | 11.945% |
| **合计** | **111,617** | **32,633,373** | - | **650.65 h** | **1.754 TiB** | **100%** |

总表观字节数为 1,928,277,036,035 B。审计时所在文件系统可用空间约 258 TB，容量不是近期阻断项；更现实的风险是大量小 Parquet、视频随机 seek、PNG 解码和 HDF5 并发读取造成的吞吐下降。

若只统计非空的 Stage05 future-difference 标签，共 14,510,521 帧：

| 数据集 | 有标签 Episodes | 有标签 Frames | 标签帧占本库 | 在全部标签帧中的占比 |
|---|---:|---:|---:|---:|
| DROID | 29,432 | 9,507,523 | 34.410% | 65.522% |
| MolmoAct Household | 5,936 | 794,199 | 100% | 5.473% |
| MolmoAct Tabletop | 1,881 | 310,743 | 100% | 2.142% |
| RH20T | 8,142 | 3,898,056 | 100% | 26.864% |

因此，即使剔除 DROID 的无标签部分，以 `sample_ratio=1.0` 直接拼接仍会让 DROID 占文本监督的 65.5%。当前 episode-grouped sampler 每个 epoch 会遍历所有选中样本，不会自动对数据集等权。

## 3. 完整性与可读性检查

### 3.1 Parquet 与 episode 元数据

| 数据集 | 数据 Parquet | 行数核对 | Footer/Schema | Episode 索引、长度、数据引用 |
|---|---:|---:|---:|---:|
| DROID | 86 | 27,630,375，一致 | 全部通过 | 95,658 个 episode 连续且一致 |
| Household | 5,489 | 794,199，一致 | 全部通过 | 5,936 个 episode 连续且一致 |
| Tabletop | 1,613 | 310,743，一致 | 全部通过 | 1,881 个 episode 连续且一致 |
| RH20T | 25 | 3,898,056，一致 | 全部通过 | 8,142 个 episode 连续且一致 |

未发现零字节数据 Parquet、无法读取的 footer、schema 漂移、episode 长度求和不一致、frame span 断裂或 metadata 指向不存在的数据文件。

### 3.2 视觉数据

| 数据集 | 存储形式 | 三视角 | 编解码信息 | 审计结果 |
|---|---|---|---|---|
| DROID | 784 个视频 | exterior1 / exterior2 / wrist | AV1, 320x180, 15 FPS | 所有容器可打开；每视角首/中/尾分层抽样解码通过 |
| Household | Parquet 内嵌 PNG | first_view / second_view / wrist_image | 640x480 | 跨文件首/中/尾共 45 张抽样解码通过，均非空白 |
| Tabletop | Parquet 内嵌 PNG | first_view / second_view / wrist_image | 640x480 | 跨文件首/中/尾共 45 张抽样解码通过，均非空白 |
| RH20T | 269 个视频 | exterior1 / exterior2 / wrist | H.264, 320x180, 10 FPS | 所有容器可打开；每视角首/中/尾分层抽样解码通过 |

当前 future-difference prompt 会把三个视角都指定为 **224x224**。这会把 4:3 和 16:9 源图直接变为正方形；在正式预训练前应通过可视化 batch 确认是否接受形变，或改为保持宽高比的 resize/pad 策略。Stage06 光流本身也是 224x224。

本次未逐帧解码全部视觉内容，也未对约 1.93 TB 数据做全量内容哈希。视觉结论是“容器全检查 + 分层抽样解码”，不是逐帧无损认证。

## 4. Stage05 future-difference 文本

四个库所有非空 `train_data` 均完成 JSON 全扫描：没有解析错误，且键严格为以下四项，值均为非空字符串：

```text
Task_temporal
Spatial_motion
Contact_interaction
Object_constraints
```

| 数据集 | 非空标签帧 | 空标签帧 | 字符数 min / mean / max | 结果 |
|---|---:|---:|---:|---|
| DROID | 9,507,523 | 18,122,852 | 683 / 1884.54 / 2838 | 非空部分通过；空值必须过滤 |
| Household | 794,199 | 0 | 686 / 1687.53 / 2268 | 通过 |
| Tabletop | 310,743 | 0 | 684 / 1763.82 / 2238 | 通过 |
| RH20T | 3,898,056 | 0 | 未见异常长尾 | 通过 |

DROID 的 29,432 个 episode 全量有标注，66,226 个 episode 全量无标注，没有发现 episode 内部分标注的情况。Stage05 合并清单的缺失策略为 `null`，与现状一致。

### 4.1 文本区间与 32 步 action chunk

对 Tabletop 的 310,743 个训练位置，使用当前默认 `action_horizon=32` 与上游 `language_action_interval` 做了全量闭区间比较：

| 分类 | 帧数 | 比例 | 含义 |
|---|---:|---:|---|
| exact | 374 | 0.120% | 文本区间与 action chunk 两端完全一致 |
| full | 41,383 | 13.317% | 文本区间完全覆盖 action chunk，但端点不同 |
| partial | 268,986 | 86.562% | 仅部分覆盖 32 步 action chunk |
| none | 0 | 0% | 完全不相交 |

标注区间长度 min/mean/median/p90/p99/max 为 2 / 21.12 / 19 / 41 / 50 / 51 帧；32 步 chunk 与标注区间的平均交集为 16.57 步，中位数为 15 步。

这不是 JSON 完整性问题，而是联合 VLM/action 监督的语义对齐问题：多数样本的文本只描述 action chunk 的一部分。Household、DROID、RH20T 在生成可用的 `steps_data_index.pkl` 后，也必须运行同一审计；不能由 Tabletop 的结果外推为已通过。

### 4.2 Token 长度

字符数不能代替 Qwen processor 的精确 token 数。当前 CLI 默认 `max_length=1200`，而每个样本还包含三个 224x224 图像 token、相机名、任务文本、chat 模板和终止 token。启动前必须用最终选定的 processor 和 `max_length` 对全部 eligible 行运行精确 token 审计，并要求：

- `input_truncated_count == 0`
- `target_truncated_count == 0`
- assistant 终止 token 未被截断

仓库已经提供 `scripts/audit_future_difference_lengths.py`，但本次没有在“尚未确定最终 processor/max_length”的情况下给出误导性的通用通过结论。

## 5. 动作、状态与统计量

### 5.1 跨库动作语义

| 数据集 | 当前 state | 当前 action | 关键问题 |
|---|---|---|---|
| DROID | 8D：关节位置 + 夹爪 | 8D：关节位置 + 夹爪 | 与 7D EEF delta 不是同一控制空间；另有 7D `action.original`，但它是绝对 Cartesian pose + gripper |
| Household | 7D | 7D EEF delta + gripper | 数值有限，兼容当前维度约束 |
| Tabletop | 7D | 7D EEF delta + gripper | 数值有限，兼容当前维度约束 |
| RH20T | 8D：xyz + quaternion(wxyz) + gripper | 7D：world-frame achieved EEF delta + next-frame normalized gripper | state 维度不兼容；动作有异常和有效位 |

不能仅靠每库 q01/q99 归一化把 DROID joint-position action 与其他库的 EEF delta action 混为一个动作专家目标。正式方案应在以下两条路线中明确选择一条：

1. 统一转换到相同坐标系、相同增量定义、相同旋转表示、相同 gripper 时序；或
2. 保留不同 embodiment/action schema，并使用明确的数据集/机器人条件与分头输出。

### 5.2 数值质量

Household、Tabletop 的 7D state/action 全量扫描未发现空值、错误长度、`NaN` 或 `Inf`，分位数范围也符合小幅 EEF delta 的预期。

RH20T 发现：

- `action.valid == false` 共 8,176 帧，其中 8,142 帧是 episode 末帧，另有 34 帧是非末帧。
- 4 帧 state 有极端值，导致相邻的 8 帧 action 出现极端 delta；这 8 帧均标记为 action invalid。
- 极端量级分别涉及约 `3.554e36`、`3.355e7`、`8.935e17`、`1.655e12`。
- 异常来自原始 `/opt/data/private/lq/datasets/RH20T-v30` 的 `tcp.npy` / `tcp_base.npy`，不是 Stage05 merge 新引入。
- 受影响的新 episode 为 403、1248、3176、4350；对应旧 episode 为 446、1423、3878、5294。
- `stats.json` 的 `action.std[0]`、`observation.state.std[0]`、`observation.tcp_pose.std[0]` 为 `NaN`，已有 q01/q99 也被极端值污染。

当前 v3 adapter 只根据 episode 尾部缺帧生成 action mask，不读取 RH20T 的 `action.valid`。若直接训练，invalid 动作仍会参与 loss。

必须先隔离或修复上述 4 个 episode，按 `action.valid` 屏蔽监督，然后基于最终 eligible 数据重算 q01/q99；不要复用当前 RH20T `stats.json`。

## 6. Episode 长度与 action 尾部 padding

| 数据集 | Episode 长度 min / p50 / p90 / p99 / max | `<32` 的 episodes | H=32 尾部 padding 槽位比例 |
|---|---|---:|---:|
| DROID | 1 / 221.5 / 547 / 1195 / 3191 | 926 | 5.350% |
| Household | 34 / 115 / 229 / 308 / 393 | 0 | 11.585% |
| Tabletop | 37 / 160 / 290 / 423 / 562 | 0 | 9.383% |
| RH20T | 18 / 388 / 836 / 1767.6 / 6247 | 2 | 3.238% |

尾部 action mask 的结构本身合理，但文本区间并不随这个 mask 自动对齐。另一个跨库问题是时间尺度：32 步在 DROID 15 FPS 下约为 2.13 s，在其他三个 10 FPS 库中约为 3.2 s。统一步数并不等于统一预测时长。

## 7. 任务文本与分布

### DROID

- 元数据报告 49,630 个 task index，但 `meta/tasks.parquet` 没有当前 adapter 所要求的 `task` 列，仅有 `task_index` 与 `__index_level_0__`。
- 数据行中的 `language_instruction` 可作为候选任务字段，但有 7,807,335 帧为空，占全库 28.256%。这些空指令全部位于无 Stage05 标签部分。
- 因而若只训练 Stage05 eligible 子集，空 instruction 问题可随 eligibility 一起排除；仍需生成符合 adapter 合约的非空 task 表。

### MolmoAct

- Household 有 82 个任务，头部任务 `pour me some water` 占约 9.18%，存在一定长尾但不极端。
- Tabletop 只有 14 个任务，`load the plate` 占 15.47%，`load bowl` 占 11.78%，`hang mug` 占 11.10%。应按任务报告验证指标，避免总体 loss 掩盖小任务退化。

### RH20T

- 141 个任务，包含 flexiv、ur5、franka、kuka 等机器人和多类夹爪。
- 8,062 个 episode 有成功评分，其中 7,483 成功、579 失败；另有 80 个无评分。
- 需要明确失败轨迹是否作为行为克隆监督保留。若保留，应给出失败/成功条件或权重，而不是无区分混合。

## 8. RH20T 多相机同步

RH20T 以 exterior1 为主时间轴；其自身对齐精确。其他相机存在以下偏移：

| 相机 | 复用帧 | 复用率 | `abs(offset)>100 ms` | `>500 ms` | 最大偏移 |
|---|---:|---:|---:|---:|---:|
| wrist | 767,455 | 19.688% | 421,983 | 60,309 | 52,201 ms |
| exterior2 | 210,268 | 5.394% | 6,065 | 1,646 | 2,205 ms |

wrist 有 5,305 / 8,142 个 episode 至少出现一帧偏移大于 100 ms，4,194 个 episode 至少出现一帧大于 500 ms。极端 wrist 偏移出现在新 episode 4912、frame 2613。

建议为三视角 future-difference 训练设置同步门槛，例如过滤 `abs(offset)>100 ms` 的辅助视角/样本，或给相机有效性 mask；至少应做“仅 exterior1”和“三视角带同步过滤”的消融。Stage06 光流使用 exterior1，因此这项同步问题不直接污染现有光流侧车，但会污染三视角 VLM 条件。

## 9. Stage06 光流侧车

| 数据集 | H5 episodes | 对应 frames | Delta | 相机 | 加权有效像素率 | 平均 flow magnitude | 状态 |
|---|---:|---:|---:|---|---:|---:|---|
| DROID | 0 | 0 | - | - | - | - | 缺失 |
| Household | 5,936 | 794,199 | 20 帧 | first_view | 0.97010 | 0.01716 | 完整 |
| Tabletop | 1,881 | 310,743 | 20 帧 | first_view | 0.98551 | 0.00870 | 完整 |
| RH20T | 8,142 | 3,898,056 | 20 帧 | exterior1 | 0.97933 | 0.00528 | 完整 |

三个现有 Stage06 目录的 manifest、H5 引用、quality 文件、episode/frame 数和文件大小全部对应；合计 15,959 个 H5、5,002,998 帧、620.36 GiB。H5 内容为 float16 gzip 的 `[T,2,224,224]` flow 和 `[T,1,224,224]` mask，episode 尾部使用 clamp，并各有一个 identity frame。

现有光流仅覆盖全部 Stage05 标签帧的 34.478%，因为 DROID 尚无 Stage06。更关键的是，当前 ZR-0 仓库未找到读取 `stage06_flow`、`flow_target` 或 `flow_valid_mask` 的训练路径；这些文件目前不会自动进入 loss。若预训练目标包含光流，必须先实现并验证 reader、mask、尺度定义和 checkpoint manifest。

## 10. 许可与数据治理

- DROID 本地 README 标注 Apache-2.0。
- MolmoAct Household/Tabletop 本地发布目录未发现可独立确认来源和许可的 README/license 文件；需要补充来源版本、下载地址、许可快照和生成链路。
- RH20T 选中集合由两类许可组成：
  - RH20T-C：4,059 episodes，1,982,363 frames，55.07 h，CC BY-SA 4.0。
  - RH20T-NC：4,083 episodes，1,915,693 frames，53.21 h，CC BY-NC 4.0。

商业或对外发布用途不能默认混用 RH20T-NC。应把许可类别写入 episode-level manifest，并在构建训练集时显式选择。以上是工程审计结论，不构成法律意见。

## 11. 当前 ZR-0 训练接口兼容性

当前 `lerobot_v3_future_difference` adapter 的硬约束包括：

- 必须有 `meta/steps_data_index.pkl`。
- 必须恰好配置三个 `dtype=image` 字段；不接受 LeRobot v3 `dtype=video`。
- joint loss 下 state/action 必须都是 7D。
- 必须有 `meta/stats_gr00t.json`，其中 state/actions 的 q01/q99 都是有限 7D 且逐维 `q99>q01`。
- `meta/tasks.parquet` 必须有 `task_index`、非空 `task`。
- 不读取数据集自带的 `action.valid`。

实测初始化结果：

| 数据集 | 首个失败点 | 后续已知失败点 |
|---|---|---|
| DROID | camera 字段是 `video` 而非 `image` | state/action 8D、缺 stats_gr00t、缺 steps、task 表不兼容、空标签需过滤 |
| Household | 缺 `steps_data_index.pkl` | 尚未加入 dataset2feature 配置 |
| Tabletop | 无；初始化成功，长度 310,743 | 仍需区间、token、split 验收 |
| RH20T | camera 字段是 `video` 而非 `image` | state 8D、缺 stats_gr00t、缺 steps、未使用 action.valid、异常统计量 |

当前 `dataset2feature.yaml` 只包含 `molmoact_tabletop_v3_stage05`，另外三库没有配置项。

## 12. 预训练前必须完成的准入清单

### P0：阻断项

- [ ] 决定统一动作表示，或设计多 embodiment/action-head 方案；记录坐标系、旋转表示、gripper 定义和动作时序。
- [ ] 为 DROID/RH20T 增加受测试的视频读取路径，或离线转为 adapter 接受的 image 表示。
- [ ] 为 DROID、Household、RH20T 生成并验证 `steps_data_index.pkl`；DROID 只纳入 9,507,523 个有标签帧。
- [ ] 给 DROID 建立非空 task 映射，给四库建立显式 `training_eligible` 或等价不可绕过的过滤。
- [ ] RH20T 隔离/修复 4 个异常 episode，在 action loss 中使用 `action.valid`，然后重算 stats_gr00t。
- [ ] 处理 RH20T 三视角时间偏移，至少使无效辅助视角不参与视觉条件。
- [ ] 按 episode/任务/来源建立无泄漏 validation/test split，并冻结 split manifest。
- [ ] 使用最终 processor 和 `max_length` 做全量 token 审计，确保 target 与 chat termination 均不截断。
- [ ] 对四库运行 H=32 文本区间审计，并决定 partial overlap 的过滤、截短、重标或 action horizon 策略。
- [ ] 完成 RH20T 许可拆分及 MolmoAct 来源/许可追溯。

### P1：正式长跑前

- [ ] 明确 15 FPS 与 10 FPS 的时间尺度：按秒统一 horizon，或显式接受不同预测时长。
- [ ] 生成过滤后、按最终动作定义计算的每库 q01/q99；训练 manifest 固定文件哈希。
- [ ] 设计数据混合权重并记录“按帧、按 episode、按任务、按小时”中的目标口径。
- [ ] 做 dataloader 吞吐压测，覆盖视频 seek、内嵌 PNG、HDF5、多 worker 和多机共享存储。
- [ ] 抽样人工审核 Stage05 文本的任务一致性、空间描述、接触状态和物体约束；本次仅验证结构，未验证语义真实性。
- [ ] 若使用 Stage06，补 DROID 光流，并先证明当前训练代码真实消费 flow/mask。

## 13. 建议的数据混合基线

不建议以四库 `sample_ratio=1.0` 作为默认基线。可先做两个可解释的配方：

1. **自然分布基线**：遍历全部 eligible 帧，保留真实数据量差异；报告 DROID 65.522%、Household 5.473%、Tabletop 2.142%、RH20T 26.864% 的实际监督占比。
2. **等库贡献诊断基线**：以 Tabletop 的 310,743 帧为上限下采样，近似 ratio 为 DROID `0.03268`、Household `0.39127`、Tabletop `1.0`、RH20T `0.07972`。该配方只用于判断大库支配效应，不应自动作为最终最优配方。

由于当前 `sample_ratio` 只允许 `(0,1]`，只能下采样，不能过采样小库。若目标是按任务或 embodiment 平衡，需要新的 sampler，而不是仅调四个全局 ratio。

## 14. 建议的分阶段放行顺序

1. 先用 Tabletop 跑小规模端到端 smoke，但处理或量化 32 步 chunk 与文本区间的 partial overlap。
2. 给 Household 补 steps/config 后加入，验证同源 MolmoAct 两库的联合训练和任务均衡。
3. 修复 RH20T 数值、valid mask、同步、许可和 state schema 后加入；先做单库稳定性测试。
4. 最后加入 DROID。先决定 7D/8D 动作语义、视频 reader、task 映射和 eligible subset；不要为了追求规模直接混入 1,812 万空标签帧。
5. 每次扩库都保留固定 validation、数据 manifest、样本占比、token 截断率、无效 action 比例、dataloader 吞吐和梯度统计。

## 15. 审计边界

本报告完成了结构化元数据、Parquet footer/schema/行数、episode 映射、低维数值、Stage05 JSON 和 Stage06 manifest/引用的一致性检查；视频做全容器检查和分层帧解码，内嵌图像做跨文件分层抽样。

本报告没有完成以下工作，因此不能把“结构通过”理解为“语义无噪声”：

- 未逐帧解码和人工观看全部视频/图像。
- 未对 1.93 TB 内容建立全量 checksum 基线。
- 未人工复核数百万条生成文本和光流的语义准确性。
- 未在最终 processor/max_length 下完成全量 token 长度审计。
- 除 Tabletop 外，尚未完成 action chunk 与文本标注区间的全量对齐统计。
- 未做真实多 GPU dataloader 吞吐与训练稳定性压测。

在 P0 清单全部关闭并产出冻结的 dataset manifest 前，四库联合预训练应保持为 **NO-GO**。
