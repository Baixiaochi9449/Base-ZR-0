# 当前 LIBERO 评估文件、流程、并行与恢复说明

本文总结当前服务器上 ZR-0 的 LIBERO 评估实现。当前可复现基准是已经完成的
`step-34184` 评估：

```text
/opt/data/private/lq/ZR-0/result/eval/
Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm_step-34184_seed7_50trials_auto_20260901_233411
```

下文将该目录简称为 `RUN_DIR`。该次评估使用 GPU 0、1、2、3，将四个 LIBERO
suite 分配到四张 GPU 并行运行。最终完成 2,000 个 episode，成功 1,926 个，
总成功率为 96.3%。

## 1. 核心结论

| 能力 | 当前是否支持 | 实际行为 |
| --- | --- | --- |
| 四个 suite 多 GPU 并行 | 支持 | 四组独立的模型服务端和 LIBERO 客户端分别运行在 GPU 0、1、2、3 |
| 单个 suite 内 task 并行 | 不支持 | 10 个 task 按 `task_id` 串行执行 |
| 单个 task 内 episode 并行 | 不支持 | 50 个 episode 按 `episode_idx` 串行执行 |
| 模型推理 batch size | 不支持 | 每次 WebSocket 请求只包含一个环境的一份 observation |
| action chunk | 支持 | 每次预测并执行 10 个动作；这是时间维 action chunk，不是 batch 并行 |
| 失败 suite 自动重试 | 支持 | 同一次 `run_eval.sh` 中只重试失败 suite，最多一次，并从 episode 0 重跑 |
| episode 级断点续评 | 不支持 | 没有进度 checkpoint、`--resume` 参数或跳过已完成 episode 的逻辑 |
| 整个控制器中断后自动续评 | 不支持 | 重新执行 `run_eval.sh` 会再次把四个 suite 全部加入运行队列 |

因此，当前提高评估速度的主要方式是现有的“四个 suite 分别占用一张 GPU”并行。
不能通过增加 `batch_size` 提速，也不能把中断点之后的剩余 episode 直接续上。

## 2. 环境和固定路径

| 资源 | 当前路径 |
| --- | --- |
| ZR-0 仓库 | `/opt/data/private/lq/ZR-0` |
| 仓库内独立 LIBERO 客户端 | `/opt/data/private/lq/ZR-0/LIBERO` |
| 模型服务端环境 | `/opt/data/private/lq/.conda/envs/zr0-eval` |
| LIBERO 客户端环境 | `/opt/data/private/lq/.conda/envs/zr0-libero-eval` |
| 当前 checkpoint | `/opt/data/private/lq/ZR-0/outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm/step-34184` |
| 当前结果目录 | `RUN_DIR`，即本文开头给出的完整目录 |

服务端和客户端使用两个独立环境。服务端环境负责加载模型并执行 GPU 推理；客户端
环境负责创建 LIBERO/MuJoCo 环境、无头渲染和 rollout。

执行本文后续查询命令前，可在 shell 中设置当前结果目录：

```bash
RUN_DIR=/opt/data/private/lq/ZR-0/result/eval/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm_step-34184_seed7_50trials_auto_20260901_233411
```

## 3. 评估文件清单

### 3.1 仓库通用评估源码

| 文件或目录 | 作用 |
| --- | --- |
| `server.py` | 服务端入口；解析模型和推理配置，加载 `ZR0Policy`，启动 WebSocket 服务 |
| `evaluation/libero_eval/run_libero_eval.py` | LIBERO 客户端入口；创建任务环境、预处理 observation、请求 action、执行 rollout 并写日志和视频 |
| `policies/reasoning_vla_policy.py` | 加载 checkpoint、组织模型输入、执行 action 推理和反归一化 |
| `model/reasoning_vla_model.py` | ZR-0 模型推理主路径 |
| `model/flow_matching_action_head.py` | Action Expert 和 flow-matching 去噪推理 |
| `utils/websocket_server_policy.py` | WebSocket 推理服务端协议 |
| `utils/websocket_client_policy.py` | WebSocket 同步客户端；一次发送一份 observation 并等待一份 action 响应 |
| `utils/image_tools.py` | 客户端图像 `resize_with_pad` 和 `uint8` 转换 |
| `utils/normalization.py` | state/action 归一化与反归一化辅助逻辑 |
| `dataset2feature.yaml` | `demo_data.libero_v21` 数据条目和统计量选择配置 |
| `demo_data/libero_v21/meta/` | LIBERO v2.1 的数据元信息和 state/action 统计量 |
| `simple_scripts/eval_libero.md` | 使用当前服务器已有环境手工运行官方模型评估的操作手册 |
| `evaluation/libero_eval/README.md` | 仓库原始的通用 LIBERO 评估介绍；不包含当前运行的完整保护和汇总逻辑 |

仓库根目录中的文件可能继续发生修改。复现某次已经完成的正式评估时，应优先使用
该结果目录中的 `source/` 快照，而不是直接使用当前工作区源码。

### 3.2 当前 `step-34184` 运行固化的源码

`RUN_DIR/source/` 保存了评估实际执行时使用的源码快照，主要包括：

```text
source/
├── server.py
├── evaluation/libero_eval/run_libero_eval.py
├── policies/
├── model/
├── utils/
├── dataset2feature.yaml
└── demo_data/libero_v21/meta/
```

`RUN_DIR/artifacts/source-snapshot.sha256` 记录这些文件的 SHA-256。当前
`run_eval.sh` 会先进入 `RUN_DIR/source/`，所以本次结果不依赖评估期间仓库根目录
之后发生的代码修改。

### 3.3 当前运行的控制和校验脚本

| 文件 | 作用 | 是否适合直接复用于新 checkpoint |
| --- | --- | --- |
| `RUN_DIR/artifacts/run_eval.sh` | 四 suite 评估控制器；启动和回收服务端/客户端、检查日志、重试失败 suite、调用汇总 | 否；checkpoint、结果目录、GPU 映射和环境路径均已固化 |
| `RUN_DIR/artifacts/watch_and_run.sh` | 监控指定训练 tmux；确认训练正常结束和最终 checkpoint 稳定后启动 `run_eval.sh` | 否；训练 tmux、PID、命令哈希和最终 step 均为一次性配置 |
| `RUN_DIR/artifacts/eval_guard.py` | 检查 checkpoint 完整性、GPU 空闲显存、端口块、suite 日志结构和 2,000 episode 汇总条件 | 可作为新运行模板，但常量和目录约束必须核对 |
| `RUN_DIR/artifacts/summarize_eval.py` | 调用 guard 生成 Markdown/JSON/CSV 汇总，并更新 `experiment.md` 和 `run_manifest.yaml` | 仅适用于具有相同结果目录结构的新运行模板 |
| `RUN_DIR/artifacts/test_eval_guard.py` | `eval_guard.py` 的单元测试 | 可用于修改 guard 后的回归检查 |
| `RUN_DIR/artifacts/checkpoint-weights.sha256` | 实际 checkpoint 权重哈希 | 只对应 `step-34184` |

这些脚本是当前评估记录的一部分，不是参数化的仓库级通用 launcher。评估新
checkpoint 时应新建独立结果目录、重新固化源码和脚本，不能直接改写这个已完成的
`RUN_DIR`。

### 3.4 配置、日志和结果文件

| 路径 | 内容 |
| --- | --- |
| `RUN_DIR/run_manifest.yaml` | checkpoint、代码版本、数据、resolved 评估参数、GPU/端口映射和最终结果 |
| `RUN_DIR/experiment.md` | 本次训练上下文、完整评估协议、启动命令和实际结果 |
| `RUN_DIR/env/libero-config/config.yaml` | 本次客户端使用的 LIBERO 本地路径配置 |
| `RUN_DIR/env/status.txt` | watcher 的最终状态；当前为 `complete` |
| `RUN_DIR/env/status_detail.txt` | 状态的详细说明 |
| `RUN_DIR/env/evaluation_status.txt` | 评估控制器状态；当前为 `complete` |
| `RUN_DIR/env/*.exit_code` | watcher 和评估控制器退出码 |
| `RUN_DIR/env/client_*.successful_attempt` | 每个 suite 最终采用的成功 attempt 编号 |
| `RUN_DIR/logs/watcher.log` | 训练结束、checkpoint 和资源门禁日志 |
| `RUN_DIR/logs/evaluation-controller.log` | 四 suite 启动、等待、失败重试、回收和汇总日志 |
| `RUN_DIR/logs/server_*.attemptN.log` | 每个 suite 每次尝试的模型服务端日志 |
| `RUN_DIR/logs/client_*.attemptN.log` | rollout 原始日志；正式成功率由这些日志校验和汇总得到 |
| `RUN_DIR/pids/` | 当时的进程 PID 和退出码记录；评估结束后不能把 PID 文件等同于存活进程 |
| `RUN_DIR/videos/<suite>/attemptN/` | rollout 视频；不同 attempt 分目录保存 |
| `RUN_DIR/summary/summary.md` | 人类可读的 suite 和逐 task 结果 |
| `RUN_DIR/summary/summary.json` | 结构化完整结果 |
| `RUN_DIR/summary/summary.csv` | 便于表格分析的 suite 和逐 task 结果 |

## 4. 当前评估配置

当前配置来自 `RUN_DIR/run_manifest.yaml` 和本次源码快照。运行命令没有自行覆盖
论文/仓库默认的 action chunk、horizon、去噪步数或图像尺寸。

| 配置 | 值 |
| --- | --- |
| Suites | `libero_spatial`、`libero_object`、`libero_goal`、`libero_10` |
| Tasks / rollouts | 每 suite 10 tasks；每 task 50 rollouts；共 2,000 episodes |
| 服务端 / 客户端 seed | 42 / 7 |
| Inference mode | `direct_action` |
| Observation window | 1 |
| Action / prediction horizon | 10 |
| Execution horizon / replan steps | 10 |
| 去噪步数 | 5 |
| State/action padding 宽度 | 64 |
| 原始渲染尺寸 | `256 x 256` |
| 客户端 resize | 保持长宽比、zero pad 到 `448 x 448`，双线性插值，无 crop |
| 模型实际图像尺寸 | 每视角 `224 x 224` |
| 相机 | `agentview_image` 和 `robot0_eye_in_hand_image` |
| 控制频率 | 20 Hz |
| 成功条件 | LIBERO 环境返回 `done=True` |

四组 suite 的资源映射为：

| Suite | GPU | 当前运行端口 | 最大动作步数 |
| --- | ---: | ---: | ---: |
| `libero_spatial` | 0 | 8200 | 280 |
| `libero_object` | 1 | 8201 | 280 |
| `libero_goal` | 2 | 8202 | 300 |
| `libero_10` | 3 | 8203 | 520 |

每个 episode 在上述最大动作步数之外，先执行 10 个稳定等待步。每次模型推理返回
10 步 action chunk，客户端顺序执行完 10 步后再请求下一次推理。

## 5. 当前评估流程

```text
训练正常结束
  -> 校验训练退出码和最终保存标记
  -> 两次校验最终 checkpoint 的文件完整性和稳定性
  -> 等待 GPU 0-3 每张至少 12 GiB 空闲显存，并选择完整端口块
  -> run_eval.sh 为四个 suite 各启动一个模型服务端
  -> 等待四个服务端输出 Start serving.
  -> 为四个 suite 各启动一个 LIBERO 客户端
  -> 四个 suite 并行；每个 suite 内 task 和 episode 串行
  -> 校验每个 suite 恰好完成 500 episodes 且无运行时错误
  -> 失败 suite 从 episode 0 自动重试一次，成功 suite 不重跑
  -> 汇总四个 suite 的 2,000 episodes
  -> 写 summary、experiment、manifest，并回收本次记录的进程
```

### 5.1 可选的训练结束 watcher

本次 watcher 的历史启动命令是：

```bash
cd /opt/data/private/lq/ZR-0
tmux new-session -d -s zr0-final-eval-watch-20260901-233411 \
  "bash $RUN_DIR/artifacts/watch_and_run.sh"
```

它只属于这次 `step-34184` 实验。脚本会校验指定训练 tmux 的 pane PID、进程启动
时间和启动命令哈希，要求训练正常退出、训练日志包含最终保存标记，并确认
`step-34184` 是最高且稳定的 checkpoint。当前训练和评估均已结束，不应再次启动
这个 watcher。

### 5.2 四 suite 评估控制器

watcher 实际执行的命令是：

```bash
bash "$RUN_DIR/artifacts/run_eval.sh" 8200,8201,8202,8203
```

`run_eval.sh` 为每个 suite 启动以下两类进程。

模型服务端命令的等价形式为：

```bash
CUDA_VISIBLE_DEVICES=GPU_ID \
  /opt/data/private/lq/.conda/envs/zr0-eval/bin/python -u server.py \
  --dataset_entry demo_data.libero_v21 \
  --ckpt_dir /opt/data/private/lq/ZR-0/outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm/step-34184 \
  --inference_mode direct_action \
  --port PORT
```

LIBERO 客户端命令的等价形式为：

```bash
/opt/data/private/lq/.conda/envs/zr0-libero-eval/bin/python -u \
  -m evaluation.libero_eval.run_libero_eval \
  --args.task-suite-name SUITE \
  --args.port PORT \
  --args.video-out-path "$RUN_DIR/videos/SUITE/attemptN"
```

命令只显式设置 suite、端口和视频目录，其余参数继续使用源码快照中的默认值。

### 5.3 rollout 和结果汇总

客户端按以下顺序执行每个 suite：

1. 按 `task_id=0..9` 依次创建 LIBERO 环境。
2. 读取每个 task 的官方固定 initial states，并按 `episode_idx=0..49` 依次运行。
3. 每个 episode 先执行 10 步 dummy action，使物体稳定。
4. 旋转两个相机画面 180 度，resize-with-pad 后与 8 维机器人状态一起发给服务端。
5. 服务端返回 action chunk，客户端执行前 10 步，再重新请求模型。
6. LIBERO 返回 `done=True` 时计为成功；达到 suite 对应最大步数仍未成功则计为失败。
7. 客户端在日志中记录逐 episode、逐 task 和 suite 总结果。

客户端正常退出后，`eval_guard.py` 要求每个 suite 日志包含 10 个 task 结果、500 个
episode、唯一的 suite 总结果，并拒绝包含 traceback、OOM 等运行时错误的日志。
`summarize_eval.py` 只在四个 suite 总计恰好 2,000 个 episode 时生成最终汇总。

## 6. 并行能力说明

### 6.1 当前已支持：suite 级多进程、多 GPU 并行

`run_eval.sh` 同时维护四套独立进程：

```text
GPU 0: libero_spatial server <-> libero_spatial client
GPU 1: libero_object  server <-> libero_object  client
GPU 2: libero_goal    server <-> libero_goal    client
GPU 3: libero_10      server <-> libero_10      client
```

每个服务端各加载一份完整模型，每个客户端各自创建 LIBERO 环境。四个 suite 不
共享模型进程或环境进程，因此可以并行推进，某个 suite 先完成也不会阻塞其他 suite。

### 6.2 当前不支持：batch size 或 suite 内并行

`run_libero_eval.py` 中 task 循环和 episode 循环都是普通串行 `for` 循环。客户端
每次只构造一个 `request_data`，`WebsocketClientPolicy.infer()` 发送一次请求并同步
等待一次响应；没有 `batch_size`、环境数量或 worker 数量参数。

服务端虽然使用异步 WebSocket 接收连接，但 handler 中直接同步调用
`self._policy.infer(obs)`，没有请求队列、动态 batching 或并发模型 forward。因而：

- 不能通过命令行设置 `batch_size > 1`；
- action chunk 长度 10 不代表同时评估 10 个环境；
- 不应让多个客户端共享同一个服务端来假设获得批量推理加速；
- 若要增加 batch 评估，需要同时改造 vectorized LIBERO 环境、请求协议、模型批量
  预处理/推理/反归一化、日志和成功率统计，不能只新增一个 CLI 参数。

## 7. Resume 和失败重试说明

### 7.1 已支持的范围

在同一次 `run_eval.sh` 生命周期内，如果某个 suite 的服务端加载失败、服务端中途
退出、客户端非零退出、运行超时，或日志完整性校验失败，控制器会：

1. 只停止该 suite 的客户端和服务端；
2. 保留已经成功的其他 suite；
3. 为失败 suite 启动 `attempt2`；
4. 从该 suite 的 `task_id=0, episode_idx=0` 重新运行；
5. 若第二次仍失败，则整次评估标记为失败。

这属于“suite 级整组重试”，不是断点续评。

### 7.2 不支持的范围

客户端没有持久化当前 task、episode、环境状态或随机数状态，也没有读取已有日志来
跳过已完成 episode。因此：

- 单个 suite 在第 300 个 episode 中断后，不能从第 301 个继续；
- 重新执行客户端会从该 suite 第一个 task 的第一个 episode 开始；
- 重新执行整个 `run_eval.sh` 会重新运行四个 suite，而不会自动识别旧的
  `successful_attempt` 文件并跳过已完成 suite；
- 不应拼接两段不完整日志来计算正式结果，因为环境和随机状态无法证明连续一致；
- 不完整日志的中间成功率只能用于观察进度，不能作为最终评估结果。

技术上可以保留已完整通过校验的 suite，只手工重跑未完成 suite，然后重新汇总，
但当前没有封装成受支持的 resume 命令，操作时还需要正确维护 attempt 编号、日志、
视频目录和 `successful_attempt` 文件。正式实验更稳妥的做法是为失败 suite 从头重跑。

## 8. 当前结果和状态入口

当前 `step-34184` 评估已完成，四个 suite 均在 attempt 1 通过校验：

| Suite | Successes | Episodes | Success rate |
| --- | ---: | ---: | ---: |
| `libero_spatial` | 488 | 500 | 97.6% |
| `libero_object` | 500 | 500 | 100.0% |
| `libero_goal` | 486 | 500 | 97.2% |
| `libero_10` | 452 | 500 | 90.4% |
| **总计** | **1,926** | **2,000** | **96.3%** |

查看最终状态和汇总：

```bash
cat "$RUN_DIR/env/status.txt"
cat "$RUN_DIR/env/status_detail.txt"
cat "$RUN_DIR/summary/summary.md"
```

查看某个正在运行的新评估的 episode 进度时，应读取其对应客户端日志：

```bash
rg '# episodes completed so far:|Total success rate:' \
  "$RUN_DIR"/logs/client_*.attempt*.log
```

PID 文件只是历史记录。判断进程是否仍在运行时，需要再执行 `ps -p PID` 或
`kill -0 PID`，不能只根据 `RUN_DIR/pids/` 中存在文件作结论。

## 9. 使用边界

- `simple_scripts/eval_libero.md` 用于已有环境下手工启动一次官方模型评估。
- 本文用于理解当前实际评估实现和限制，不替代每次实验自己的 `experiment.md` 和
  `run_manifest.yaml`。
- `step-34184` 的 `watch_and_run.sh`、`run_eval.sh` 和 `RUN_DIR` 已经固化并完成，
  不应修改或作为新 checkpoint 的输出目录重复运行。
- 新 checkpoint 必须建立新的结果目录，记录新的 checkpoint、代码快照、完整命令、
  GPU/端口、日志和汇总，以保证结果可追溯。
