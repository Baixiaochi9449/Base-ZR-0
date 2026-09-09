# RoboTwin 抖动诊断：官方 checkpoint 的特征处理不兼容

日期：2026-09-08。范围：当前 `/opt/data/private/lq/ZR-0` 及远程正在运行的
`robotwin_50x2x20_seed0_remote_direct_20260908` 评估。不使用其他工作树的结果推断本轮原因。

## 1. 首要问题与证据

当前 `model/qwen_vl_backbone.py::QwenVLBackbone.forward` 在原始
`hidden_states[-1]` 上额外应用 `language_model.norm`，随后交给 Action Expert。
该行为来自提交 `166193d7d50c18080f954d8771b8a8a6b6927bb3` 的本地训练稳定性修复。
它没有区分按新逻辑训练的模型与作者提供的旧 checkpoint。

原仓库提交 `b1440d4` 和作者当前公开实现直接使用 `hidden_states[-1]`：
<https://github.com/RUCKBReasoning/ZR-0/blob/main/model/qwen_vl_backbone.py>。
本轮服务器使用 Transformers 4.57.1。真实模型的 forward hook 确认该张量等于
Qwen 最终 RMSNorm 的输入；额外归一化使特征 RMS 从 7.01-7.54 变成 2.61-2.66。
因此传给已训练 Action Expert 的数值分布发生了实质变化。

在 `robotwin_unified` 的 4 条轨迹、12 个观测上，用相同图像、状态、归一化统计和
噪声种子 42/43 对照，共 48 次动作预测。覆盖 adjust_bottle、beat_block_hammer
和 blocks_ranking_rgb。每次预测 16 步，5 次去噪，BF16。

| 指标 | 当前额外 RMSNorm | 原始特征 | 相对改善 |
| --- | ---: | ---: | ---: |
| 12 个关节的平均绝对预测误差，rad | 0.0483793 | 0.0242544 | 49.87% |
| 平均绝对二阶差分，rad | 0.0246559 | 0.0131358 | 46.72% |
| 夹爪预测 MAE | 0.00570222 | 0.00355039 | 37.74% |

24 组配对预测中，23 组关节 MAE 改善。专家轨迹的平均绝对二阶差分为
0.00174395 rad。这里二阶差分定义为 `mean(abs(a[t+2]-2*a[t+1]+a[t]))`，
只用作离散动作抖动的代理指标，不能解释为固定物理频率下的加速度或 jerk。

这些数据直接支持：当前新增特征归一化显著损害此官方 checkpoint 的动作精度和
平滑性。尚未运行修复后的闭环仿真，不能据此认定它是全部失败的唯一原因，也不能
声称成功率已经恢复。

## 2. BF16 反归一化是次要因素

`utils/normalization.py::min_max_denorm` 将 q01/q99 转换到预测动作的 BF16
类型后进行缩放和平移；此行为同样存在于原仓库。
同一归一化动作仅改变反归一化运算类型，最大输出差异约 0.01983 rad（1.14 度）。

另用 6 个观测、2 个种子进行 24 次配对预测，单独改 FP32 反归一化后，当前特征的
二阶差分从 0.0249615 降至 0.0244463 rad，改善 2.06%；原始特征则改善 5.65%。
预测 MAE 基本不变。因此只改反归一化精度不足以解释或修复主要异常。

## 3. 已核对的其他路径

- 客户端 `policy/ZR0/deploy_policy.py` 的动作队列依次执行 16 步，没有发现乱序、
  重复动作或把绝对关节目标当增量累加的代码。
- 左臂 6 关节+夹爪、右臂 6 关节+夹爪，共 14 维；客户端、仿真和数据元信息一致。
- `envs/_base_task.py::take_action` 默认采用 qpos 和 TOPP；与仓库原始版本无差异。
  这不等于已排除接触动力学或实际关节跟踪误差，当前日志没有记录实际 qpos。
- Action Expert 权重使用 strict=True 加载；维度和键都受验证。模型输出均为有限值。
- 三相机标签、排列和 224x224 模型输入与当前原始评估路径一致，实际输入网格已记录。
  仿真原图 320x240、下载数据视频 640x480；不能仅凭原图分辨率差异认定根因。
- 归一化/反归一化的 q01/q99 公式未发现改动。统计值来自本地全量数据重算；作者原始
  统计文件仍缺失，因此统计值的精确一致性仍未被证明。
- 四份运行日志均存在动作内部的来回变化：相邻动作差的 99 分位数约 0.0625 rad，
  最大值 0.119-0.176 rad。视频记录按策略动作采样，10 FPS 播放，不能反推实际控制频率。

## 4. 修复边界与复现资料

应为官方旧 checkpoint 显式选择原始特征处理；保留本地新训练模型需要的 RMSNorm
行为，不能全局删除该层。完成后应在独立输出目录进行同种子、小规模闭环对照，
重新核对抖动、夹爪接触和成功率，再决定是否重跑全量评估。

本次只增加诊断脚本与说明，没有修改正式模型、服务器参数或正在进行的评估。

- 脚本：`scripts/diagnose_robotwin_checkpoint.py`。
- 完整命令和实验条件：`docs/experiments/robotwin_jitter_diagnosis_20260908/experiment.md`。
- 首次结果：`outputs/diagnostics/robotwin_jitter_20260908/diagnosis.json`、`actions.npz`。
- 动作曲线：同目录 `conditioning_comparison.png`；可对照同一观测、同一噪声的两种预测。
- 精度对照：`outputs/diagnostics/robotwin_jitter_20260908_fp32_denorm/diagnosis.json`。
- 两次独立 GPU 诊断均退出 0，峰值 allocated 显存约 5.08 GiB，没有 OOM 或 NaN。
- 图像、动作数组与诊断输出由 Git 忽略，脚本和说明文档纳入暂存区；没有创建 commit。
