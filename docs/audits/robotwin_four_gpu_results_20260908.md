# RoboTwin 远程四卡评估结果核查

- 核查日期：2026-09-08，Asia/Shanghai。
- 服务器：`ssh -p 25408 lq@10.82.1.223`，实际主机名 `interactive88133`。
- 范围：核查既有实验结果、manifest、推理记录、动作执行记录及视频文件；未重新启动评估。
- 机器可读证据：[audit_evidence_20260908.json](audit_evidence_20260908.json) 的 `robotwin` 字段，保留每卡完整 server/client 命令和实际配置。
- 本地报告所在代码基准：`3bad66373877bcdfa9c1f86e134816fc8a426207`，工作树已有其他未提交修改；这些修改不参与历史得分计算。

## 1. 结果

**该次四卡实验已正常完成，`adjust_bottle / demo_randomized` 合计成功 0/100，成功率 0%。**

这次只评估一个任务、一种场景设置。`demo_clean` 是 checkpoint setting 标签；实际环境设置是 `demo_randomized`。不得将这个结果写成 RoboTwin 全任务平均分，也不得与另一个 `robotwin_50x2x20_seed0_4gpu` 实验合并。

| GPU | 硬件 | 完成状态 | Client exit | 成功/次数 | 成功率 | 结束时间（北京时间） | 视频数 |
|---|---|---|---:|---:|---:|---|---:|
| 0 | NVIDIA A800-SXM4-80GB | COMPLETED | 0 | 0/25 | 0% | 2026-09-08 02:31:22 | 25 |
| 1 | NVIDIA A800-SXM4-80GB | COMPLETED | 0 | 0/25 | 0% | 2026-09-08 01:09:57 | 25 |
| 2 | NVIDIA A800-SXM4-80GB | COMPLETED | 0 | 0/25 | 0% | 2026-09-08 02:15:10 | 25 |
| 3 | NVIDIA A800-SXM4-80GB | COMPLETED | 0 | 0/25 | 0% | 2026-09-08 01:27:06 | 25 |
| 合计 | 4 卡 | 全部完成 | 全部 0 | 0/100 | 0% | 最晚 02:31:22 | 100 |

四个 launcher 的 `started_at` 均为北京时间 2026-09-05 22:38:58。最后一个分片结束于 2026-09-08 02:31:22，总历时约 51 小时 52 分钟。

## 2. 保存位置

以下路径均在上述远程服务器上。`ZR-0-experiments` 是实验输出根目录；本次真正运行代码的 Git 工作树是：

```text
/opt/data/private/lq/ZR-0-worktrees/robotwin-reconstructed-normalization-smoke
```

实验根目录：

```text
/opt/data/private/lq/ZR-0-experiments/robotwin-formal-eval-reconstructed-parallel-raster-v3-20260905
```

四份最终得分文件的完整路径：

```text
/opt/data/private/lq/ZR-0-experiments/robotwin-formal-eval-reconstructed-parallel-raster-v3-20260905/shards/gpu-0/results/adjust_bottle/evaluation.RoboTwin.policy.ZR0.deploy_policy/demo_randomized/demo_clean/2026-09-05 22:40:34/_result.txt
/opt/data/private/lq/ZR-0-experiments/robotwin-formal-eval-reconstructed-parallel-raster-v3-20260905/shards/gpu-1/results/adjust_bottle/evaluation.RoboTwin.policy.ZR0.deploy_policy/demo_randomized/demo_clean/2026-09-05 22:40:33/_result.txt
/opt/data/private/lq/ZR-0-experiments/robotwin-formal-eval-reconstructed-parallel-raster-v3-20260905/shards/gpu-2/results/adjust_bottle/evaluation.RoboTwin.policy.ZR0.deploy_policy/demo_randomized/demo_clean/2026-09-05 22:40:34/_result.txt
/opt/data/private/lq/ZR-0-experiments/robotwin-formal-eval-reconstructed-parallel-raster-v3-20260905/shards/gpu-3/results/adjust_bottle/evaluation.RoboTwin.policy.ZR0.deploy_policy/demo_randomized/demo_clean/2026-09-05 22:40:33/_result.txt
```

每份结果文件含以下原始内容，时间戳随分片相差一秒：

```text
Instruction Type: unseen

Success: 0/25
Success Rate: 0.000000
```

每个 `_result.txt` 所在的同一目录有 `episode0.mp4` 至 `episode24.mp4` 共 25 个视频。本次核对了文件数量，没有全量观看或解码这 100 个评估视频。

以实验根目录为基准，相关文件如下：

| 相对路径 | 内容 |
|---|---|
| `shards/gpu-i/formal_manifest.json` | 最终 COMPLETED 状态、退出码、开始/结束时间、模型/代码身份、结果路径及 SHA256 |
| `shards/gpu-i/runtime/client.stdout.log` | 每次 rollout 的动作进度、Fail、累计成功率和最终保存路径 |
| `shards/gpu-i/runtime/client.stderr.log` | 仿真、ffmpeg 等 stderr 信息 |
| `shards/gpu-i/runtime/server.stdout.log`、`server.stderr.log` | 模型服务日志 |
| `shards/gpu-i/runtime/forward_audit.jsonl` | 每次模型前向的输出形状、有限性及 processor tensor 信息 |
| `shards/gpu-i/runtime/client_progress.jsonl` | 每个 episode 的推理、动作执行前后记录 |
| `shards/gpu-i/runtime/formal_eval_config.json` | 实际任务、场景、seed、步数、端口等配置 |
| `shards/gpu-i/launch_formal.sh` | 含完整参数和日志重定向的原始启动脚本 |
| `commands.log` | 数据审计、归一化重建命令以及四个正式启动脚本路径 |
| `experiment.md` | 实验设计和早期启动记录；状态更新已落后于最终结果 |

其中 `i` 为 0、1、2、3。

## 3. 交叉验证证据

1. 四个 manifest 均为 `status=COMPLETED`、`client_exit_status=0`。
2. 四份 `_result.txt` 均为 `0/25`，其实际 SHA256 与各自 manifest 的 `result_sha256` 一致。
3. 每份 client stdout 最后一条累计成功率为 `0/25 => 0.0%`，并记录相同结果文件路径。
4. 每卡实际读取 625 条 `forward_audit.jsonl`，与 manifest 计数一致；全部 `actions_finite=true`，动作 tensor 为 `[1,16,14]`。
5. 每卡读取 21,250 条 `client_progress.jsonl`；25 个 episode 每个都存在 400 条 `after_take_action`，且所有执行动作均记录为有限值。四卡合计执行 40,000 个策略动作。
6. 四卡分别有 25 个 `.mp4` 文件，与 rollout 数量一致。

结果 SHA256：

| 分片 | `_result.txt` SHA256 |
|---|---|
| GPU 0、2 | `cae2c5d01b740d618659f1f336d6635b10b678bbd3d18eee3c6b0bcae39e4a4b` |
| GPU 1、3 | `5f066c151181a89224606ac2b06abc3dbcb057120440c9d201c820c799dd4f3d` |

相同 hash 是因为同组分片得分文件中的文字和秒级时间戳相同，不表示 rollout 重复。配置的候选 seed 流分别从 100000、100001、100002、100003 开始，步长均为 4；专家可行性过滤可以跳过候选 seed。没有在本次核查中额外重做所有接受 seed 的去重审计。

根目录 `experiment.md` 的旧记录仍是 `PREPARING` / in-progress，最后只记录每卡完成 17 或 18 个动作。本报告以最终 manifest、得分文件和动作日志为依据更新结论；未改写远程历史文件。

## 4. 模型与评估条件

| 项目 | 实际配置或已有 manifest 记录 |
|---|---|
| 代码工作树 | `/opt/data/private/lq/ZR-0-worktrees/robotwin-reconstructed-normalization-smoke` |
| Git 基准 | `bd561209f8499e33a704115b1c4147262d54fdca`，加当时暂存的实验改动 |
| source snapshot SHA256 | `d5c5687f67d9ea05ebe120bf5bfb17cc0475f48579a2b272b8c4ac44d6cb2e64` |
| Checkpoint | `/opt/data/private/lq/ZR-0-experiments/robotwin-reconstructed-normalization-smoke-20260905/inputs/ZR-0-robotwin` |
| Checkpoint 来源 | 历史 manifest 记录为 ModelScope `seeklhy/ZR-0-robotwin@master` |
| 权重聚合 SHA256 | `6b9d5a77ec2105f731bb8d5a738689a3721dd76e8424391217cccd043327f9f9`，本次未重新扫描全部权重 |
| Simulator | RoboTwin `stable_2.0`，commit `13c3c47ff4312dd62484bcd51be034af55c062d1` |
| Task / setting | `adjust_bottle / demo_randomized` |
| 指令 | `unseen` |
| Rollout | 4 分片，每片 25 个通过专家可行性检查的场景，共 100 个 |
| Seed | 候选流起点 100000+i，stride=4；模型启动 seed=42，历史协议 `legacy_serial_v1` |
| 每 episode 上限 | 400 次策略 `take_action` |
| Action | 14D 双臂绝对关节目标，夹爪在维度 6、13；预测/action/execution horizon 均为 16，5 个 denoise steps |
| 输入图像 | head、left wrist、right wrist RGB，仿真原始 320x240，client 不 resize/crop/pad/增强 |
| 模型实际视觉输入 | 每视角 224x224；前向记录 `image_grid_thw=[[1,14,14],[1,14,14],[1,14,14]]`，`pixel_values=[588,1536]` |
| 后端 | attention=eager；Mesa 25.1.9 Lavapipe 软件光栅渲染，shader=default，无 RT/OIDN |
| GPU / port | 0/9100、1/9101、2/9102、3/9103 |
| 总成功率 | 四卡成功数之和 / 四卡有效策略 rollout 数之和 = 0/100 |
| W&B | 非训练任务，原实验记为 N/A |

图像进一步处理的 resize 插值、rescale、mean/std、训练时预处理及控制频率，本次未从历史运行源码和 processor 配置逐项重新验证；不能把另一工作树的当前默认值当作本实验实测值。物理子步频率也不等同于策略动作频率。

原始四卡启动入口均已保留，例如 GPU 0 的完整原始命令位于 `shards/gpu-0/launch_formal.sh`；该脚本以 `python -m evaluation.robotwin_eval formal-eval` 启动，并传入 `--gpu-index 0 --port 9100 --shard-index 0 --shard-count 4 --n-episodes 25 --attention-backend eager --confirmed-reconstructed-normalization`。其余三卡使用各自编号和端口。原命令仅用于追溯，本次没有执行这些脚本。

## 5. 如何解释这个 0%

这是已经跑完的策略失败结果，并非“没有结果文件所以按 0 计分”。所有 episode 都执行到 400 步；有限动作和成功完成日志不能证明动作语义或归一化一定正确。

实验显式标记 `normalization_origin=reconstructed_public_data`、`publication_warning=RECONSTRUCTED_DO_NOT_PUBLISH`。作者原始 q01/q99 未取得，使用公开 RoboTwin v3 的 6,075,103 行重建统计。归一化 artifact ID 为 `e5bceb0df81182e9a48fafe71012a55b6245cf7c3da798d68bdf7a068679be3c`。

同时，软件光栅渲染与官方 NVIDIA RT+OIDN 视觉分布不同，仿真原图 320x240 与历史记录的训练原图 640x480 也不同。因此该结果只能表述为上述 checkpoint 在上述重建统计和渲染条件下的单任务诊断结果。**本次核查没有确定成功率为零的根因，也没有重新评估其他任务。**
