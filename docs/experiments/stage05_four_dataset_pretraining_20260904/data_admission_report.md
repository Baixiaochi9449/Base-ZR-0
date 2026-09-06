# Stage05 四数据集混合预训练数据准入报告

- 审计日期：2026-09-04（Asia/Shanghai）
- 代码基准：`1f88937a1375eb86b5e78c759815301b67524882`
- 最终 sidecar：`outputs/stage05_four_dataset_pretraining_20260904/data_sidecars_v7`
- 审计范围：全部最终 eligible 帧；固定只读审计，不从训练集移除样本，也不称为 validation。
- 原始数据保持只读，没有改写 Parquet、视频或 metadata。

## 1. 独立准入结论

- AR 数据准入：**GO**。仅依赖有效主视角、可信任务和合法非空 `train_data`，不依赖 canonical action 或动作统计。
- Joint 数据与 canonical 准入：**GO**。四库均有权威字段定义/生成代码支持的确定性 canonical 转换；每个 Joint 样本严格满足 H=32 chunk 内 `FM_count > 0`。
- AR 运行门禁：**NO-GO（仅当前 GPU 资源）**。GPU 0--3 均有现存高负载进程，未抢占、未做 micro-batch 探测、smoke 或 pilot。
- Joint 运行门禁：**NO-GO（仅当前 GPU 资源）**。数据/canonical 已通过，但 AR pilot checkpoint 尚不能生成，因此 Joint runtime 阶段未开始。

canonical 或 DROID FM 映射即使失败也不会改变 AR 数据准入。本次映射通过，因此只剩独立的运行资源门禁。

## 2. 最终 eligible 数与自然比例

### AR-only

| 数据集 | AR eligible 帧 | 自然采样比例 |
|---|---:|---:|
| DROID | 9,507,523 | 0.6732477314 |
| Household | 794,199 | 0.0562389042 |
| Tabletop | 310,743 | 0.0220043664 |
| RH20T | 3,509,414 | 0.2485089980 |
| 合计 | 14,121,879 | 1.0000000000 |

### Joint

| 数据集 | Joint action eligible 帧 | 其中同时 AR eligible | 自然采样比例 |
|---|---:|---:|---:|
| DROID | 19,822,892 | 9,507,474 | 0.8114236697 |
| Household | 794,199 | 794,199 | 0.0325094778 |
| Tabletop | 310,743 | 310,743 | 0.0127198506 |
| RH20T | 3,501,934 | 3,501,934 | 0.1433470019 |
| 合计 | 24,429,768 | 14,114,350 | 1.0000000000 |

Joint 比例严格以 `joint_action_eligible_frames` 为分母。DROID 有 148 个 zero-FM chunk，其中 49 帧有文本；RH20T 有 7,480 个 zero-FM chunk。两者都不进入 Joint，带文本的 7,529 帧只留在 AR-only。

采样器每个 epoch 恰好消费每库全部 eligible index，并按剩余帧数在 128-sample block 内做最大余数分配；不会用伪权重改变自然比例。实际训练时 checkpoint manifest 另保存 seen、unique、duplicate、AR-eligible 和 FM-eligible 计数。

## 3. episode 与过滤

| 数据集 | 源 episode/帧 | 最终准入说明 |
|---|---:|---|
| DROID | 95,658 / 27,630,375 | 74,740 个 episode 有可信任务；18,122,852 帧无合法标注而不进 AR；Joint 另排除 148 个 zero-FM chunk |
| Household | 5,936 / 794,199 | 全部标注帧通过 AR 与 Joint |
| Tabletop | 1,881 / 310,743 | 全部标注帧通过 AR 与 Joint |
| RH20T | 8,142 / 3,898,056 | 只保留 7,480 个明确成功 episode；排除失败 578 个/348,501 帧、无评分 80 个/36,391 帧、异常 4 个/3,750 帧；逐步应用 `action.valid` |

DROID 无 Stage05 标注的帧不进 AR；任务先取可信 episode/task metadata 和原语言字段，无法恢复时排除，不生成伪任务。`slot_data` 仅透传，不决定 eligibility 或 loss。

## 4. canonical 7D 合同与证据

统一 state 为绝对 base/world-frame EEF：

`[x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad, gripper_open]`

统一 action 为同一原生时间步的相对 EEF 命令/实现量：

`[dx_m, dy_m, dz_m, droll_rad, dpitch_rad, dyaw_rad, gripper_open]`

旋转使用 SciPy lowercase `xyz` RPY，角度为 rad，分量差 wrap 到 `[-pi, pi)`；平移为 m；gripper 为 `1=open, 0=closed`。H=32 保持各库原生 FPS：DROID 15 Hz（约 2.13 s），其余 10 Hz（约 3.2 s），不重采样。

权威基线来自 `/opt/data/private/lq/FD-ID-FlowVLA` commit `3824d36cdf76bf0a9d537635de92a38f3920e9a3`：

- Molmo canonical 生成代码：`starVLA/dataloader/difference_query_eef.py`，SHA256 `e333b31d6418f6525ce491e9cd95bdc796c7060952d0f14e0c2cd06b1151733a`。
- RH20T 转换代码：`lerobot_v21_to_v30/convert_rh20t_raw_to_v30.py`，SHA256 `9fb57147e7d670b33002f192f630334bae78b9ad77e1765ecee69d5c6ddbcdf2`；转换 metadata SHA256 `f8dbe65ce8d6b61e9d4dcb47340659bf1e47a7fa07c16ee3b2fdd948e40e8c37`。
- DROID 官方语义由原审计固定到 commit `33ae6a67274f36d2e29525b86f23a56616ef43a7` 的 `droid/franka/robot.py` 与 `misc/transformations.py`：state/target 为绝对 xyz+Euler(rad)，原 gripper 低值为 open。

DROID 全量 27,630,375 行中 `action.original` 与 `action.cartesian_position + action.gripper_position` 全部一致，但它是绝对 target，不能直接当 canonical delta；实际转换为 target pose 减当前 state pose，并做旋转 wrap。与相邻 state delta 的 Pearson 为 `[0.8166, 0.8851, 0.8638, 0.7816, 0.9052, 0.8455]`，方向一致率为 `[0.8204, 0.8493, 0.8358, 0.8408, 0.8387, 0.8624]`。100 万对抽样 target-to-next 误差：平移 p50/p95/p99 为 0.0188/0.0622/0.0747 m，旋转为 0.0441/0.1467/0.1952 rad。这些是控制跟踪诊断，不是硬阈值。

RH20T state 的 wxyz quaternion 先转换为 RPY；action rotvec 按 `q_target = exp(rotvec) * q_current` 组合后再转换为 wrapped RPY delta。所有 `action.valid=false` 步均屏蔽。轨迹审计中 Household/Tabletop/RH20T 数学重建误差在约 `2e-7` 内；DROID 抽样命令与下一状态的 MAE 分别为 xyz `[0.01187,0.00592,0.00751]` m、RPY `[0.01745,0.01759,0.02004]` rad。纯数学 round-trip 使用严格容差，真实机器人跟踪误差没有使用 `1e-6` 门槛。

轨迹审计内容 hash：`32bba163d0194c964b9740e3b15096c098bbe3943bc3f35f6d0b33a3f2917695`；可视化 SHA256：`6d786f0dfd55a0de86473723144c6fa11d0d40b7ac04386cfce27aba6020ce43`。

## 5. 独立归一化统计

流程为过滤无效样本 -> canonical 转换 -> 只统计真实有效元素 -> 每库 q01/q99 -> 沿用仓库 `[-1,1]` quantile min-max 公式（训练路径保留现有 `[-15,15]` clip）。

| stats_key | state count | action count | stats 内容 hash |
|---|---:|---:|---|
| `stage05_droid` | 19,822,892 | 19,806,850 | `996b12d63e837640ef5949ce5fa8affcdcf90f6b69bd130812ec77acbdb7b9b2` |
| `stage05_household` | 794,199 | 794,199 | `3955199f69c0cd15b2ce2de22c3cff2c6ee0fc4e7f24933c8a312ed3591d8caf` |
| `stage05_tabletop` | 310,743 | 310,743 | `f41fa9d88d38bc762d5efb98dc3ef65c933eefdb04f1163bfb1e8049e7e1d125` |
| `stage05_rh20t` | 3,501,934 | 3,501,909 | `294db9e546545a754d4cc936de16fe7699197aa44faeb73a7a205ed6ab56ea27` |

完整 q01/q99 数组保存在每个 Joint sidecar 的 `stats.json`。policy/server 必须显式匹配 `stats_key`；缺失、未知或 manifest 不匹配立即报错。

round-trip 审计明确分成两种口径：

- 数学内部值：只在 q01、q01/q99 中点和 q99 上验证公式，四库 FP32 最大误差分别为 DROID `1.490116e-8`、其余三库 `3.725290e-9`。这不代表真实数据经过 clipping 后可逆。
- 真实生产数据：通过 `Stage05MixedPretrainingDataset._canonical_chunk()` 取得 canonical action，再调用训练实际使用的 `min_max_norm_unclipped()`、`min_max_norm()` 和 `min_max_denorm()`。DROID/Household/Tabletop 各分层 64 个 episode，覆盖 episode 首/中/尾、各维 q01/q99 邻近值和极值候选；RH20T 覆盖全部 7,480 个 Joint episode、3,501,909 个有效 action step，包括全量极值。

真实审计的每维结果如下。`count`、`pre-clip min/max`、`clip count/rate`、反归一化误差 max/mean/p99 和每维最坏样本均完整保存在 `audits/real_action_roundtrip_clipping_v1.json`；内容 hash 为 `e4d9af7f1787486be180cfd35ba4833a6e6f835e8afc64156af51d63ccd9bb3d`，文件 SHA256 为 `101871c2983bd5b2baf1868ddae8034d498cd167226ae5a3179056a3e4246474`。

| 数据集 | 每维有效 count | pre-clip min（维0..6） | pre-clip max（维0..6） | clip count（维0..6） | 最大反归一化误差 |
|---|---:|---|---|---|---:|
| DROID | 30,232 | `[-1.425,-1.723,-1.586,-3.009,-1.529,-2.430,-1]` | `[1.369,1.668,1.452,3.011,1.563,2.822,1]` | `[0,0,0,0,0,0,0]` | `4.470348e-8` |
| Household | 25,910 | `[-2.724,-1.747,-1.676,-1.704,-1.370,-1.862,-1]` | `[2.555,1.743,1.644,1.717,1.689,1.805,1]` | `[0,0,0,0,0,0,0]` | `1.490116e-8` |
| Tabletop | 24,545 | `[-1.767,-2.084,-1.804,-1.586,-1.568,-1.496,-1]` | `[3.137,2.009,1.798,1.513,1.944,2.197,1]` | `[0,0,0,0,0,0,0]` | `1.117587e-8` |
| RH20T（全量） | 3,501,909 | `[-28.321,-20.165,-13.618,-16.365,-25.006,-17.063,-1]` | `[27.539,15.005,16.721,15.166,27.127,15.985,1]` | `[8,3,3,4,4,5,0]` | `0.2932883` |

DROID/Household/Tabletop 的完整 stats min/max 归一化后均位于 `[-15,15]`，因此可以对全量有效 action 严格推出 clip 数为零。RH20T 的每维 clip 比例为 `[2.284468e-6,8.566756e-7,8.566756e-7,1.142234e-6,1.142234e-6,1.427793e-6,0]`；最大不可逆误差位于 episode 5580、frame 742、time 74.2 s、维度 4。保留既有 `[-15,15]` clipping，不改变训练语义。

按 canonical 维度顺序，实际 q01/q99 为：

```text
stage05_droid
state q01 = [0.26754051,-0.44499442,-0.04211944,-3.13752222,-1.20494103,-2.06214762,0.00881058]
state q99 = [0.78013611, 0.43706876, 0.77910650, 3.13758206, 0.87291354, 1.89761996,1.00000000]
action q01=[-0.06473137,-0.05479775,-0.05341356,-0.15121055,-0.13863276,-0.19189391,0.00000000]
action q99=[ 0.06524748, 0.05404621, 0.06717584, 0.15289021, 0.13105318, 0.18978953,1.00000000]

stage05_household
state q01 = [0.20011315,-0.38127145,-0.00333031,-3.13750577,-0.98753798,-1.72027803,0.00002462]
state q99 = [0.74848706, 0.57892656, 0.66217446, 3.13754034, 0.55540925, 0.60642564,1.00000000]
action q01=[-0.01762403,-0.02415747,-0.01967333,-0.05308160,-0.04469402,-0.05434052,0.00000000]
action q99=[ 0.01628786, 0.03164507, 0.02510980, 0.05209722, 0.03808610, 0.05904453,1.00000000]

stage05_tabletop
state q01 = [0.35903701,-0.43363771,0.31208092,-3.13096142,-0.90260100,-2.00892568,0.00000000]
state q99 = [0.75783664, 0.28440276,0.66986835, 3.12996483, 0.53275979, 0.09838159,1.00000000]
action q01=[-0.01173521,-0.01602891,-0.00916268,-0.03775463,-0.03453676,-0.03481518,0.00000000]
action q99=[ 0.01230122, 0.01644563, 0.01424611, 0.03498320, 0.03091864, 0.04115157,1.00000000]

stage05_rh20t
state q01 = [0.36651254,-0.27241617,0.00747447,-3.14134693,-0.55661148,-3.14153504,0.00000000]
state q99 = [0.74289030, 0.32406265,0.32845646, 3.14147830, 0.43801096, 3.14148426,1.00000000]
action q01=[-0.00737162,-0.01042636,-0.00785124,-0.03163397,-0.02483503,-0.04217191,0.00000000]
action q99=[ 0.00829953, 0.01048864, 0.00995204, 0.03244716, 0.02353358, 0.04484838,1.00000000]
```

## 6. 双视角与视频读取

- Molmo：固定 `first_view` 后 `wrist_image`；DROID/RH20T：固定 `exterior_1_left` 后 `wrist_left`。
- `second_view/exterior2` 不在 Parquet projection、源 inventory 或解码路径中；每个完整样本恰好两个 placeholder。
- RH20T 腕部偏差绝对值超过 100 ms 时仅保留主视角：421,983 帧，占有主视角帧的 10.825473%。
- DROID release 只有共享的 episode-local `timestamp`，没有每相机 capture timestamp，不能把 packed-video `from_timestamp` 当相机偏差；该限制显式写入 manifest。
- 真实解码样本原始分辨率：视频 320x180，Molmo 内嵌图 640x480；Qwen processor 输入固定 RGB 224x224、bicubic、无 crop/pad/augmentation，rescale 1/255，mean/std 均为 0.5。

在当前环境对 AV1（DROID）和 H264（RH20T）各相机连续 32 帧测试 PyAV/TorchCodec 均成功、值有限且落在 `[0,1]`。选择已有 LeRobot PyAV 路径，因为无新增依赖且四路合计实测延迟更低；具体后端仍封装在 adapter。可视化：`audits/visual_batch_v3/decoded_two_view_batch.png`。

## 7. token 与时间区间审计

当前 Qwen processor 实测每个 224x224 视角产生 49 个视觉 token，一视角合计 49、两视角合计 98；assistant termination 为 2 tokens。对全部 14,121,879 个 AR eligible 样本逐条计算 chat-template、任务、完整 target 与终止 token 后：

| 数据集 | 一/两视角 AR 帧 | full tokens min / p50 / p99 / max |
|---|---:|---:|
| DROID | 0 / 9,507,523 | 266 / 653 / 751 / 941 |
| Household | 0 / 794,199 | 266 / 614 / 674 / 724 |
| Tabletop | 0 / 310,743 | 267 / 598 / 647 / 693 |
| RH20T | 377,581 / 3,131,833 | 218 / 619 / 691 / 789 |

因此保证输入、完整 `train_data` 和 termination 零截断的最小 `max_length` 为 **941**；launcher 在本四库实验中从已验证报告读取该值，940 立即失败，941 和 1024 接受。2026-09-05 重新运行了全部 AR eligible 样本的 format-v2 审计，报告为 `audits/token_length_audit_v9_format2.json`；content hash 为 `852fa0fc20ab9bd6583eded888fb2d22aa3803274b36989f3150ad0f7b4a81cf`，文件 SHA256 为 `7b46995a1b2f1eec70202a972f1fe7266f0e523f6c94c8ab10c7db4a12e827a8`，implementation identity 为 `10180375e7a8e8980e0b43f67a5ac965badc6070531af33c66e639cce7c68660`，三者与版本化实验规格一致。旧格式或缺身份字段的报告不再接受，也没有常量 941 回退。训练 adapter 仍逐样本检查 target 与 termination 零截断。

format v2 将长度结论绑定到实际四个 AR sidecar，而不是路径名推断：

| dataset entry / stats key | manifest content hash | eligible index SHA256 | eligible count |
|---|---|---|---:|
| `stage05_droid_mixed` / `stage05_droid` | `77b52af750ee02229f370e4e4d51b77e5ae03a380373887ec979d333a375a45d` | `83a2997e976f89757fefa33757bc474a5d067bc2ea1d334db4305055f2940ffa` | 9,507,523 |
| `stage05_household_mixed` / `stage05_household` | `27229f06da9c6e95b637121b9807da5a4e70761609cf95b3e527631ba7f93989` | `022a8423e718736ab01d94f3b4263dfda365cedceea18040be761c404352f6cf` | 794,199 |
| `stage05_tabletop_mixed` / `stage05_tabletop` | `18037813d9970093a9b55ad9214bcb3dd004cbd9a9fb4dc64583b49b7b6e0c01` | `69fa759d9c0a8b3b184abb65fbdc648492365317ed0430927ab6afa7ffd76a72` | 310,743 |
| `stage05_rh20t_mixed` / `stage05_rh20t` | `0b5875bd4c73132c01978d364f4463039909cf314b5faeac96e6b2209ed2ddad` | `de4ca4948e130b67b3fc2357bb79450a2ec08f913176ec6a59ea8e9cbc9b0af0` | 3,509,414 |

报告同时保存各 sidecar 的 format version 2 和完整 generator identity；processor 绝对路径为 `/opt/data/private/lq/models/Qwen3-VL-2B-Instruct`，类为 `Qwen3VLProcessor` / `Qwen2TokenizerFast` / `Qwen2VLImageProcessorFast`，版本为 Transformers 4.57.1、qwen-vl-utils 0.0.14、tokenizers 0.22.2。身份覆盖 tokenizer、特殊 token、chat template、preprocessor/model/video config 文件及其 SHA256、视觉参数、processor/qwen-vl-utils 运行时源码，以及 target canonicalization、Stage05 message/双视角构造、chat-template 调用、训练 tokenization 和审计脚本的源码 SHA256。生产 `--validate-only` 会先验报告自哈希，再逐项重算外部身份和 main->wrist 一/两视角合同；任一变化均要求重新执行完整审计。

四库文本区间与 H=32 动作块的全量结果如下；14,121,879 帧均找到原始 annotation，unavailable 为 0：

| 数据集 | exact | full | partial | none | 交集长度 mean / p50 / p99 |
|---|---:|---:|---:|---:|---:|
| DROID | 8,744 | 2,919,716 | 6,579,063 | 0 | 19.54 / 19 / 32 |
| Household | 629 | 71,656 | 721,914 | 0 | 13.65 / 10 / 32 |
| Tabletop | 374 | 41,383 | 268,986 | 0 | 16.57 / 15 / 32 |
| RH20T | 2,317 | 1,585,413 | 1,921,684 | 0 | 23.77 / 29 / 32 |

审计内容 hash 为 `40a5f508d573a74265d89f11775772c51389ef8d43f825e5792837e7483348ce`，JSON 文件 SHA256 为 `d3c88b9f94f56a099058a9f32291041a5b88e2e81095b522a2ad8d5fa88141e5`。不完全对齐只记录为已知限制，不裁剪 FM，也不重新生成文本。

## 8. 可复现性与 sidecar hash

| 数据集 | AR manifest hash | Joint manifest hash | 源 inventory hash |
|---|---|---|---|
| DROID | `77b52af750ee02229f370e4e4d51b77e5ae03a380373887ec979d333a375a45d` | `183419f103245b60cdff98ba7de4a7e2aedc7e28556297cfff82b6e1abdde9e6` | `bb7acb39bf61e0fd44893aaac9c94f2d83a472135cf551b01bd6c8a8ea8c7de5` |
| Household | `27229f06da9c6e95b637121b9807da5a4e70761609cf95b3e527631ba7f93989` | `31e0ae1d85596fb215a283225e821b6937bdb1f8360bbeeac6faacfd150922dc` | `6cb10cd7796f5453a1f29fe38e1d77c5a0b38260511a6406c78333fc0401a0c1` |
| Tabletop | `18037813d9970093a9b55ad9214bcb3dd004cbd9a9fb4dc64583b49b7b6e0c01` | `4acd1869621dec0588fab021d2b2440b1fb38a1b7927f3fb8694b93b2310440f` | `c4285304f14df458347ffec4fa89f5976f252db805306cb5fa05fce8fb908423` |
| RH20T | `0b5875bd4c73132c01978d364f4463039909cf314b5faeac96e6b2209ed2ddad` | `02c4d52e6fd205aba160d3ac0cfe23464ad0ac320bd1641c3449f227b9789f5b` | `5ff01d6ea82fe3c040b16e2af2265c34c80f93787d6dd8472ae0ba3fd9ac9134` |

sidecar format version 为 2。AR generator identity 的源码依赖为 `utils/stage05_sidecar.py`、`utils/dataset_adapters.py`、`scripts/build_stage05_sidecars.py`；Joint 额外包含 `utils/stage05_canonical.py`。manifest 对每个相对路径记录 SHA256，并记录无时间戳的关键生成参数与 Python/Numpy/PyArrow（Joint 另含 SciPy）版本；任一依赖、format version 或参数变化都会拒绝旧 sidecar。AR/Joint generator identity 分别为 `55953e53c828a7ae31e3c9668755a18213db4d199a28bf8dcf17ad659be5d046` 和 `250488db14db790cf86abfb9fba4c324d9e895ea6b32dded4ad3faa1964b326e`。

构建只写同文件系统的 `.zr0-stage05-sidecar-incomplete-*` 唯一临时目录；在 `.npy`/Parquet/stats/manifest 全部完成后校验完整性、shape、count、文件 hash、源 inventory 和可加载性，再用目录 rename 原子发布。既有目标不覆盖；并发同目标只有一个 rename 成功；异常仅清理本次且同时匹配专用前缀和 incomplete marker 的目录，不触碰相似名称或有效 sidecar。v7 与 v6 的 eligible index、count 和 Joint stats 已逐项验证完全一致；本次同时显式记录了可复现的 `--horizon` 生成参数。索引继续使用 mmap `.npy`，episode 元数据使用小型 Parquet。

## 9. 测试和运行门禁

- 修改前基线：263 passed，3 skipped。
- 第一轮 P2 定向测试：81 passed；覆盖 sidecar identity/原子发布、真实 clipping、Action Expert 配置、max-length floor、W&B 有界失败及相关 checkpoint 回归。
- 二次 P2 定向测试新增覆盖 Stage05 AR->Joint 完整配置合同、token audit format v2 全部身份漂移、launcher 生产校验入口，以及阻塞 W&B finish 的墙钟 deadline；三组分别为 12、20、17 passed。完整回归为 350 passed、3 skipped、6 个既有第三方/legacy warning，耗时 359.61 s。
- `max_length=941` 的真实 adapter 审计通过：四库 AR/Joint 均产生有效 Qwen tensor，DROID 文本缺失的 Joint 样本为 `AR_count=0, FM_count=224`；内容 hash `75ef447875b33cd4094be64c6f3f5e9dd35c1bb93d71d9ef8f2356f334e2f5b7`。
- GPU 门禁采集时 GPU 0/1/2/3 空闲显存分别为 9,543/19,519/8,319/36,613 MiB，利用率为 100/100/74/92%，均存在现有计算进程。
- 因资源门禁 NO-GO：最大稳定 micro-batch 探测、单库 forward、四卡 AR smoke/resume/100-step pilot、Joint smoke/resume/100-step pilot 均未启动；训练步数、loss、吞吐、显存曲线、checkpoint 和 W&B run 均不存在。

本轮不混入通用 VQA/VL 数据，这是相对原始 ZR-0 预训练协议的已知差异，存在 VLM 灾难性遗忘风险；不因此扩大数据范围。

## 第三次审核补充：可信 token 绑定与 Horizon 合同

本轮 sidecar 生成身份发生变化，因此没有复用旧 v7 报告。已对四个 AR sidecar 的全部 eligible frame 重新生成 format-v2 报告：
`outputs/stage05_four_dataset_pretraining_20260904/audits/token_length_audit_v9_format2.json`。

报告的 `required_max_length` 仍为 **941**；报告 content hash `852fa0fc20ab9bd6583eded888fb2d22aa3803274b36989f3150ad0f7b4a81cf`、文件 SHA256 `7b46995a1b2f1eec70202a972f1fe7266f0e523f6c94c8ab10c7db4a12e827a8` 和 implementation identity SHA256 `10180375e7a8e8980e0b43f67a5ac965badc6070531af33c66e639cce7c68660` 已写入版本化可信规格 `configs/stage05_four_dataset_experiment.json`。Stage05 launcher 在任何模型/数据分配前先校验该规格、报告文件 hash、报告自哈希、四个 sidecar/eligible index、processor/tokenizer/chat-template、包版本和源码身份；报告移动到另一目录但内容不变仍可验证，身份或规格任一变化都会要求重新审计。环境变量不能覆盖规格中的期望 hash。LIBERO 下游不继承此 Stage05 941 门禁。

Stage05 checkpoint purpose 也已写入生产合同：AR->Joint 与同实验 Joint resume 必须保持 checkpoint 中的 H（当前实验配置为 H=32）；只有显式 `downstream_finetune` 新实验初始化允许通过白名单覆盖为任意合法 H_ft。architecture hash 覆盖除 runtime horizon 外的完整 Expert 结构字段，runtime contract hash 另覆盖 H、purpose、checkpoint kind 与 Stage05 manifest。H 改变不影响当前 Action Expert 的 state_dict key/shape，加载使用 strict=True；下游使用目标数据集自己的 stats_key/q01/q99，新建 optimizer/scheduler，不恢复预训练状态。

第四次审核进一步验证了 H 不是 32->10 的特例：生产配置的 32->10、16->8、8->16 和 H 不变均保持完整 Expert state-dict key/shape，fresh downstream 只覆盖经过容量检查的正整数 H_ft；同实验 resume 与推理不得覆盖 checkpoint H。真实 Stage05 Joint manifest 必须由生产 resolver 产生且为 `vlm_and_action`，其 checkpoint 保存完整 Stage05/通用合同与 Expert 权重。脚本实际引用的旧 Tabletop Joint step-19424 通过严格 legacy 配置与 safetensors key/shape 只读预检，可作为新的 LIBERO H=10 fresh 初始化来源；它不被伪装成 Stage05 checkpoint，也不恢复旧训练状态。

本轮六项 Finding 定向生产入口测试为 **88 passed**；完整 CPU 回归为 **383 passed, 3 skipped, 6 warnings**（393.87 s）。Stage05 AR 941/1024 dry-run、940 预期失败、无显式 trusted-spec 的 token audit validate-only、LIBERO 旧 checkpoint H=10 dry-run、train/server help 和 launcher shell syntax 均通过。Joint resume 对可解析 safetensors、完整 key/shape、scheduler、DeepSpeed/client/optimizer state 及一致 global step 完成 CPU 预检；未执行真实多卡 DeepSpeed 动态 resume、GPU training 或 rollout。
