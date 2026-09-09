# 四个 Stage05 数据集审计报告

- 日期：2026-09-08，Asia/Shanghai；负责人：lq，执行：Codex。
- 代码库：`/opt/data/private/lq/ZR-0`；HEAD：`3bad66373877bcdfa9c1f86e134816fc8a426207`。
- 工作树已有其他任务的未提交修改，本次按读取时的实际文件审计；没有修改数据集、训练代码、配置或历史实验。
- 机器可读证据：[audit_evidence_20260908.json](audit_evidence_20260908.json)，含元数据 SHA256、逐库计数、抽样文件、光流映射、sidecar 检查和独立的 RoboTwin 结果证据。
- 本次不启动训练、模型推理或仿真评估。数据检查使用 CPU。

## 1. 结论与优先事项

**四个发布目录的结构、Parquet 总行数和 episode 元数据一致；不应把这种完整性通过视为当前 Joint 训练可以直接启动。**

| 数据集 | 发布结构 | Stage05 标签覆盖 | 当前配置的 AR sidecar | 当前配置的 Joint sidecar | 主要限制 |
|---|---|---|---|---|---|
| DROID 完整库 | 通过 | 34.410% 帧非空 | 加载校验通过 | generator stale | 大量无标注帧；光流仅覆盖部分 episode 且需编号映射 |
| MolmoAct Household | 通过 | 100% 帧非空 | 加载校验通过 | generator stale | 主数据体积大；没有独立 validation/test |
| MolmoAct Tabletop | 通过 | 100% 帧非空 | 加载校验通过 | generator stale | 任务少、分布不均；没有独立 validation/test |
| RH20T | 结构通过 | 100% 帧非空 | 过滤后的索引校验通过 | generator stale | 极端状态/动作、原始统计污染、腕部相机不同步 |

需要优先处理的事项：

1. **当前四个 Joint sidecar 与代码身份不匹配。** `load_stage05_sidecar(..., verify_source=True)` 全部报 `Stage05 sidecar generator is stale`。需要用目标训练代码重新生成并验证，或显式选定已经验证的匹配产物；不应绕过身份检查。
2. **RH20T 原始 `meta/stats.json` 不能直接作为动作归一化统计。** 全量扫描证实极端值仍存在，三个统计字段为 NaN。必须使用正确过滤、动作转换后生成的独立统计。
3. **DROID 全量库不能默认全部提供 AR/Slot/Flow 监督。** `train_data`、`slot_data` 各有 18,122,852 帧为 null；光流只有 9,465 个 episode，且其 merged 编号不等于完整库编号。
4. **四库只有 train split。** 本次是完整性审计，不构成 held-out 验证集。训练质量评估仍需按 episode 及来源分组划分，避免按帧切分造成相邻帧泄漏。

以上是数据和配置检查结果，不包含训练吞吐、收敛、最终任务成功率或当前 GPU 可用性结论。

## 2. 路径与规模

| 简称 | 完整路径 |
|---|---|
| DROID | `/opt/data/private/lq/datasets/droid_1.0.1_stage05_full_95658_20260831` |
| Household | `/opt/data/private/lq/datasets/molmoact_dataset_household-v3_stage05` |
| Tabletop | `/opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05` |
| RH20T | `/opt/data/private/lq/datasets/RH20T-v30_stage05` |

四库 `codebase_version` 均为 `v3.0`。下表 episode、帧数和任务数取当前 `meta/info.json`，并与 episode Parquet、任务表和所有数据 Parquet footer 交叉核对。

| 数据集 | Episodes | Frames | Tasks | FPS | 帧时长合计 | 原始帧占比 | Split |
|---|---:|---:|---:|---:|---:|---:|---|
| DROID | 95,658 | 27,630,375 | 49,630 | 15 | 511.674 h | 84.669% | train=0:95658 |
| Household | 5,936 | 794,199 | 82 | 10 | 22.061 h | 2.434% | train=0:5936 |
| Tabletop | 1,881 | 310,743 | 14 | 10 | 8.632 h | 0.952% | train=0:1881 |
| RH20T | 8,142 | 3,898,056 | 141 | 10 | 108.279 h | 11.945% | train=0:8142 |
| 合计 | 111,617 | 32,633,373 | 不合并跨库任务编号 | 不同 | 650.646 h | 100% | 无独立 validation/test |

时长计算为 `frames / fps / 3600`，不是去重后的独立行为时长。任务编号仅在各库内有效，不能直接相加作为联合任务类别数。

| 数据集 | 数据 Parquet | Row groups | 主数据表观大小 | Stage06 H5 表观大小 |
|---|---:|---:|---:|---:|
| DROID | 86 | 433 | 368.048 GiB | 428.245 GiB |
| Household | 5,489 | 5,489 | 521.591 GiB | 100.499 GiB |
| Tabletop | 1,613 | 1,613 | 157.234 GiB | 40.274 GiB |
| RH20T | 25 | 93 | 127.257 GiB | 479.588 GiB |
| 合计 | 7,213 | 7,628 | 1,174.131 GiB | 1,048.606 GiB |

主数据大小只计算 `data/**/*.parquet` 与视频 `.mp4`；光流大小是 manifest 中 H5 大小求和并逐文件核实。未计 metadata、quality JSON、历史日志和其他附属文件，也没有按硬链接或底层存储去重。不能与旧报告的整个目录大小直接比较。

## 3. 检查范围与可信边界

| 检查项 | 本次覆盖范围 | 结果 |
|---|---|---|
| Parquet footer、同库 schema、总行数、零字节文件 | 全部 7,213 个数据文件 | 未发现失败；schema 比较不含 metadata 描述字段 |
| Episode 编号连续性、length、dataset_from/to_index 连续性 | 全部 111,617 个 episode 元数据 | 通过；length 总和与 info/footer 一致 |
| Episode 对数据文件、视频文件的引用 | 全部元数据引用 | 缺失引用为 0 |
| 新旧 episode mapping | 四份 JSONL 全量 | 行数正确，new 连续，old 无重复 |
| 任务表 | 四份 Parquet 全量 | task_index 无重复；DROID 有一条空任务文本 |
| `train_data`、`slot_data` null 数 | 全部 row group 的 footer statistics | 全部有可用 null_count，无未知行 |
| 标签 JSON 形状、抽样 state/action 有限性 | 20 个分层文件，合计 4,348 行 | 非空标签抽样均通过；数值样本未见 NaN/Inf |
| 视频容器头 | 全部 1,053 个视频 | 全部可读，分辨率/FPS/codec 与下文一致 |
| 图像解码 | 每库 3 个文件位置 × 3 相机，共 36 张 | 全部可解码，像素有变化 |
| RH20T state/action、action.valid、相机同步数值 | 全部 3,898,056 行 | 发现下文极端值和同步偏移 |
| 光流 manifest、H5/quality 引用和 H5 大小 | 全部 25,424 条记录 | 引用缺失和大小不符均为 0 |
| 光流 H5 结构 | 每库首条记录，共 4 个 H5 | 形状、属性可读取；未全量读取 flow tensor |
| 当前 registry 的 AR/Joint sidecar | 4×2 个配置路径，使用生产加载器 | AR 全通过，Joint 全部身份过期 |

标签分层方法：按数据文件路径排序，选取 `linspace(0, n-1, 5)` 五个位置，每个文件取中间 row group 的前至多 256 行。图像选取首、中、末三个文件；视频解码各选中文件的首帧，内嵌图取选中文件中间 row group 的首行。**这不是每个视频内部首/中/尾的逐段检查，也不是全部标签语义正确的认证。**

Footer 的 null_count 只能区分 null 与非 null，不证明非 null 文本全部合法。本次没有重新解析全部 1,451 万条非空标签，也未逐帧确认 frame_index/timestamp、视频损坏、Slot 掩码语义、视觉描述准确性或重复轨迹。

没有对数 TB 内容重算全量 SHA256，没有做所有 H5 的数值/遮挡质量检查，也没有执行训练 adapter 的完整模型前向。证据中的小型元数据哈希用于绑定此次读取对象，不代替整个数据集内容哈希。

## 4. Stage05 标签与任务文本

| 数据集 | `train_data` 非 null | `train_data` null | `slot_data` null | 非空帧比例 | 本次抽样行 / 非空标签行 |
|---|---:|---:|---:|---:|---:|
| DROID | 9,507,523 | 18,122,852 | 18,122,852 | 34.410% | 1,280 / 55 |
| Household | 794,199 | 0 | 0 | 100% | 721 / 721 |
| Tabletop | 310,743 | 0 | 0 | 100% | 1,067 / 1,067 |
| RH20T | 3,898,056 | 0 | 0 | 100% | 1,280 / 1,280 |
| 合计 | 14,510,521 | 18,122,852 | 18,122,852 | 44.465% | 4,348 / 3,123 |

同库两个标签列空值总数相同；本次没有全量比较两个列的空值位置逐行相同。

非空 `train_data` 抽样均能解析为 JSON 对象，恰有 `Task_temporal`、`Spatial_motion`、`Contact_interaction`、`Object_constraints` 四个非空字符串字段。非空 `slot_data` 抽样均包含 `query_1` 至 `query_9` 九个字段；未将“有九个字段”当作每个 Slot 都具有有效监督。

DROID 的缺失策略在 `meta/stage05_merge.json` 中为 `missing_stage05_policy=null`，其余三库为 `exclude`。因此 DROID 大量 null 是发布策略的一部分。不能把 null 转成空目标后参与 AR/Slot loss，也不能因为缺 AR 标签就自动认定动作监督不可用。

四份 merge 清单的 `status` 当前仍为 `configured`，本报告对完整性的判断来自实际文件检查，而非这个状态字符串。完整的 `source_dataset`、`stage05_dir` 和清单 hash 已记录到证据 JSON。

DROID 的任务文本保存在 `meta/tasks.parquet` 的 `__index_level_0__` 列，其他三库使用 `task`。DROID 空任务项为 task_index=0；episode 元数据将 7,807,335 帧关联到空任务文本。训练时应以生产任务恢复/过滤逻辑决定可用性，本次没有把这项 metadata 计数当作重扫原始 `language_instruction` 的结果。

Household 的头部任务 `pour me some water` 覆盖 72,873 帧（约 9.18%）；Tabletop 的 `load the plate`、`load the bowl`、`hang the mug` 分别为 48,064、36,598、34,504 帧（约 15.47%、11.78%、11.10%）。这些数值由 episode 任务成员关系按长度加权，适合观察分布，不是跨库去重后的语义分类。

## 5. 图像与多视角

所有分辨率以下均写作宽×高。

| 数据集 | 发布相机 | 存储与实际尺寸 | 容器数 / 解码抽样 |
|---|---|---|---|
| DROID | exterior_1_left、exterior_2_left、wrist_left | AV1，320×180，15 FPS | 302+299+183=784 个视频；9 帧抽样 |
| Household | first_view、second_view、wrist_image | Parquet 内嵌 PNG，640×480 | 9 张抽样 |
| Tabletop | first_view、second_view、wrist_image | Parquet 内嵌 PNG，640×480 | 9 张抽样 |
| RH20T | exterior_1_left、exterior_2_left、wrist_left | H.264，320×180，10 FPS | 100+97+72=269 个视频；9 帧抽样 |

当前 `dataset2feature.yaml` 的 `stage05_*_mixed` 路径使用主相机和腕部相机两个视角，未使用 second_view/exterior2。`utils/stage05_dataset.py::build_stage05_message`、`resolve_stage05_vision_contract` 明确将单张图设置为 224×224，bicubic，不保持原比例，无 crop/pad/letterbox/增强；具体 rescale、mean/std 从实际 processor 读取。本次未重新加载指定训练 checkpoint 的 processor，因此不额外声称某个 mean/std 或 token 总数是本次实测。

RH20T 的主相机为 exterior1。全量帧级同步结果：

| 相机 | 复用帧数 | 复用率 | 绝对偏移 >100 ms | >500 ms | 最大绝对偏移 |
|---|---:|---:|---:|---:|---:|
| exterior1 | 0 | 0% | 0 | 0 | 0 ms |
| wrist | 767,455 | 19.688% | 421,983 | 60,309 | 52,201 ms |
| exterior2 | 210,268 | 5.394% | 6,065 | 1,646 | 2,205 ms |

因此不能假设三张图天然严格同步。已验证的 AR sidecar 记录 wrist omitted=421,983，即偏移超过 100 ms 时不使用腕部图像。允许单视角的旧 AR 路径与要求双视角交集的新实验配置不是同一个样本集合。

本次没有获得 DROID 每相机独立采集时间的全量验证证据，不能从 packed-video 起始时间推导其物理相机同步质量。

## 6. 动作、状态与 RH20T 数值质量

| 数据集 | 发布 state | 发布 action | 当前仓库的适配 |
|---|---|---|---|
| DROID | `observation.state` 8D 关节+夹爪；另有 Cartesian pose/gripper | `action` 8D 关节+夹爪；另有 Cartesian target/gripper | `canonical_droid_arrays` 使用 Cartesian target 减当前 Cartesian state，并 wrap 旋转角；不直接把 8D joint action 当成 7D EEF delta |
| Household | `state` 7D | `actions` 7D | `canonical_molmo_arrays` 处理状态、原生动作和夹爪 |
| Tabletop | `state` 7D | `actions` 7D | 同 Household |
| RH20T | `observation.state` 8D：xyz、wxyz quaternion、gripper | `action` 7D：world-frame achieved EEF delta、rotvec、next-frame gripper | `canonical_rh20t_arrays` 将 quaternion/rotvec 转为 RPY 及 wrapped delta，结合 `action.valid` |

当前 canonical state/action 均为 7D，前六维为 xyz/RPY 或其增量，第七维为夹爪。维度一致仍不消除 commanded target 与 achieved delta 的控制差异；具体语义应沿用各转换函数和 source metadata，不能仅靠归一化混合。

当前代码的夹爪转换也不是原值直接拼接：DROID state 使用 `1-clip(g)`，action 使用 `g<0.2`；RH20T action 使用 `g>=0.5`。这些是本次读取到的现有实现，本次未修改阈值或重算 canonical 统计。

RH20T 全量 3,898,056 行的检查结果：

- 原始 state/action 没有 NaN/Inf，但有限值并不代表物理合理。
- 以 `max(abs(xyz))>1000` 定位到 4 帧极端状态、8 帧极端动作；这是定位异常的审计阈值，不是建议采用的训练阈值。
- 最大绝对 xyz 值约 `3.5543094e36`。8 个极端动作均为 `action.valid=false`。
- `action.valid=false` 共 8,176 帧，其中 8,142 个末帧、34 个非末帧。

| 新 episode | frame_index | 极端坐标 |
|---:|---:|---|
| 403 | 612 | x=-3.5543094244408723e36 |
| 1248 | 18 | x=33,554,684 |
| 3176 | 351 | y=8.934673406188585e17 |
| 4350 | 249 | z=1,654,714,138,624 |

`meta/stats.json` 的 `action.std[0]`、`observation.state.std[0]`、`observation.tcp_pose.std[0]` 均为 NaN。这个文件包含非标准 JSON 数值字面量，Python 默认解析虽可接受，严格 JSON 消费端可能拒绝。仅排除非有限原始动作不足以消除上述有限极端值。

现有 `utils/stage05_sidecar.py` 已列出异常 episode `{403,1248,3176,4350}`，且 canonical/adapter 路径会读取 `action.valid`。原始发布数据没有因此被修复；新训练仍应使用经过验证的过滤结果和匹配统计。当前 Joint sidecar 身份过期，不能直接把历史统计的 GO 当作当前加载器的 GO。

DROID、Household、Tabletop 的原始 `stats.json` 本次未发现非有限字段；其 state/action 数值本次仅做上述分层抽查，没有重新计算全量分位数或全面物理范围诊断。

## 7. 光流覆盖与 episode 身份

| 数据集 | Manifest 后缀 | H5 episodes | Manifest frames | 对原库帧覆盖 | 原编号映射后长度不符 | 直接 merged 编号不匹配 |
|---|---|---:|---:|---:|---:|---:|
| DROID | d76a6dd745045034 | 9,465 | 3,573,588 | 12.934% | 0 | 9,465 |
| Household | 7a445e019d8e566c | 5,936 | 794,199 | 100% | 0 | 0 |
| Tabletop | f61339e88e1b99c9 | 1,881 | 310,743 | 100% | 0 | 0 |
| RH20T | 81ff8628f688bea5 | 8,142 | 3,898,056 | 100% | 0 | 0 |
| 合计 | - | 25,424 | 8,576,586 | 26.282% | 0 | 9,465 |

完整清单路径为各库根目录的：

```text
DROID:    stage06_flow/droid/manifest.d76a6dd745045034.jsonl
Household: stage06_flow/molmoact_household/manifest.7a445e019d8e566c.jsonl
Tabletop:  stage06_flow/molmoact_tabletop/manifest.f61339e88e1b99c9.jsonl
RH20T:     stage06_flow/rh20t/manifest.81ff8628f688bea5.jsonl
```

所有 manifest 的 source episode 都能经 `meta/stage05_episode_mapping.jsonl` 的 `old_episode_index -> new_episode_index` 映射到发布库，映射后的 episode length 与 flow frame_count 全部一致。H5、quality JSON 引用均存在，H5 大小与清单全部一致；没有全量重算 H5 内容 hash。

**DROID 清单沿用部分库的编号。** 例如 flow `merged_episode_index=0` 的 `source_episode_index=21041`，在完整 DROID 中应匹配 episode 21041。9,465 条记录全部需要这个映射；若直接按 merged=0 去读完整库 episode 0，会把视觉与光流监督配错。

当前 registry 的 `stage05_droid_mixed` 未配置 optical_flow_data_root；有光流路径的是另一个 `stage05_droid_partial_mixed` 条目。目录存在不等于完整 DROID 的默认训练路径已启用光流。工作树中的三阶段准备脚本另有显式 `flow_episode_map` 路径，但本次不对其完整训练接入或尚在进行的准备任务出具验收结论。

每库首个 H5 的结构检查显示：flow 为 float16 `[T,2,224,224]`，valid_mask 为 uint8 `[T,1,224,224]`，nominal delta=20 帧，tail_policy=clamp，flow_units=`normalized_source_image_extent`。20 帧对应 DROID 约 1.333 秒，其他库 2 秒。mask 属性定义为有限且前向落点在界内，**不等同于遮挡真值**。这些 H5 的字段类型和属性是四个样本的检查结果，未全量外推为每个文件内容均无问题。

## 8. 当前训练接入与历史报告的区别

本次直接调用生产函数 `utils.stage05_sidecar.load_stage05_sidecar(path, verify_source=True)`，检查 `dataset2feature.yaml` 指向的：

```text
outputs/stage05_four_dataset_pretraining_20260904/data_sidecars_v7/ar/{droid,household,tabletop,rh20t}
outputs/stage05_four_dataset_pretraining_20260904/data_sidecars_v7/joint/{droid,household,tabletop,rh20t}
```

| 数据集 | 已验证的 AR eligible frames | Joint manifest 声明的 action eligible frames | 本次 Joint 加载 |
|---|---:|---:|---|
| DROID | 9,507,523 | 19,822,892 | FAIL：generator stale |
| Household | 794,199 | 794,199 | FAIL：generator stale |
| Tabletop | 310,743 | 310,743 | FAIL：generator stale |
| RH20T | 3,509,414 | 3,501,934 | FAIL：generator stale |
| 合计 | 14,121,879 | 24,429,768（历史声明，非当前通过值） | 4/4 失败 |

AR PASS 包括 manifest、自哈希、文件内容 hash、source inventory、索引形状和计数检查，不代表已经执行完整训练前向。Joint 在 generator 身份检查处被拒绝，未继续其后续全部检查；stale 是代码/产物身份不匹配，不应自动解释为原始 Parquet 损坏。

这些 sidecar 的生成 horizon 均为 32。工作树另有三阶段配置使用 H=10；不得把 H32 的 Joint 样本数和统计无条件用作 H10 新实验的验收值。相同步数也不代表相同预测时间：H32 在 DROID 为约 2.133 秒，在其余三库为 3.2 秒；H10 分别约 0.667 秒和 1 秒。

| 数据集 | Episode 长度 min / p50 / p90 / p99 / max | `<10` episodes | `<32` episodes |
|---|---|---:|---:|
| DROID | 1 / 221.5 / 547 / 1195 / 3191 | 280 | 926 |
| Household | 34 / 115 / 229 / 308 / 393 | 0 | 0 |
| Tabletop | 37 / 160 / 290 / 423 / 562 | 0 | 0 |
| RH20T | 18 / 388 / 836 / 1767.59 / 6247 | 0 | 2 |

短 episode 和 episode 尾部的动作块需使用 temporal mask；RH20T 还需叠加 action.valid。实际执行 horizon 和机器人控制频率是训练/评估配置问题，不能从这些发布目录中唯一确定。

历史文档 [2026-09-03 审计](../data_audit_stage05_pretraining_20260903.md) 中的“DROID 无光流”“仓库没有四库 canonical/光流读取路径”已不适用于当前工作树。本次确认光流文件和读取/转换代码已经存在，但不能据此省略覆盖和身份检查。

[2026-09-04 数据准入报告](../experiments/stage05_four_dataset_pretraining_20260904/data_admission_report.md) 的 GO、941 token 长度下限和区间覆盖统计对应当时的代码/processor/sidecar。当前标签长度审计、token 边界、Slot 来源语义和 H10 对齐没有在本任务重新全量执行；本报告不将旧数值重新认证为当前实验参数。

## 9. 建议的后续处理与留存

1. 固定目标训练代码、processor、H、相机策略与统计合同后，重新验证或生成匹配的 Joint sidecar；保留源数据只读。
2. 明确区分源帧数、AR eligible、FM eligible、Slot 有效 anchor、Flow 覆盖及多监督交集。不要用一个总样本数替代全部目标的监督规模。
3. DROID 全库光流接入必须使用 source episode 映射，并进一步验证 source fingerprint、时间戳、相机和文件内容身份；本次长度匹配只是其中一层验证。
4. RH20T 保留异常/失败/无评分过滤、action.valid 和腕部同步策略，基于最终定义重新生成可信 canonical 统计。
5. 另行建立按 episode/来源分组的 validation/test；按数据集与任务分别报告效果，防止 DROID 的原始 84.669% 帧占比掩盖小库表现。

这些是审计建议，本次未执行数据修复或新训练产物生成。

证据 JSON 保存了四库所有本次读取的 `meta/*.json` 的 SHA256、抽样文件和行数、视频 header 摘要、异常帧、光流路径/映射结果，以及读取时的 registry/adapter/canonical/sidecar 源码 SHA256。环境为 NumPy 2.2.6、PyArrow 25.0.1、PyAV 12.3.0、Pillow 12.3.0、h5py 3.16.0。

本次一次性检查程序与分项原始输出暂存于 `/tmp/zr0-data-audit-20260908-v7Mydd/`；长期追溯以本报告、证据 JSON 和其绑定的输入文件为准。没有修改这些数据目录，没有新增 checkpoint、模型权重或训练日志。
