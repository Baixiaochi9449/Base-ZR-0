# 在当前服务器运行 ZR-0-LIBERO 评估

本文说明如何使用当前服务器上已经准备好的模型、代码、LIBERO 客户端和
Conda 环境，完成 `libero_spatial`、`libero_object`、`libero_goal` 和
`libero_10` 四个任务集的评估。

本文不包含模型下载、LIBERO 下载、环境创建或依赖安装。除任务集、端口、
checkpoint 和输出目录外，命令不覆盖仓库的默认评估参数。

> 以下“初始化、启动、等待、汇总和清理”命令必须在同一个 SSH shell 中按顺序
> 执行，因为 Bash 只能 `wait` 当前 shell 启动的子进程。完整评估耗时较长，建议
> 先进入 `tmux`，避免 SSH 断开后任务退出。

## 1. 已有资源

| 资源 | 路径 |
| --- | --- |
| ZR-0 仓库 | `/opt/data/private/lq/ZR-0` |
| 官方 ZR-0-LIBERO 模型 | `/opt/data/private/lq/models/ZR-0-libero` |
| 模型来源 | `modelscope://seeklhy/ZR-0-libero`，revision `fd9ea33c44688398990014da13308df608ace3b8` |
| 当前仓库内的 LIBERO | `/opt/data/private/lq/ZR-0/LIBERO` |
| 模型服务端环境 | `/opt/data/private/lq/.conda/envs/zr0-eval` |
| LIBERO 客户端环境 | `/opt/data/private/lq/.conda/envs/zr0-libero-eval` |
| 上一次完整验证结果 | `/opt/data/private/lq/ZR-0/result/eval/ZR-0-LIBERO_official_seed7_50trials_20260830_195202` |

评估使用两个核心脚本：

- `server.py`：加载 ZR-0-LIBERO checkpoint，并通过 WebSocket 提供动作推理。
- `evaluation/libero_eval/run_libero_eval.py`：创建 LIBERO 环境、请求动作并执行 rollout。

该 checkpoint 是官方发布模型，本地没有与它一一对应的训练实验目录；模型来源、
revision 和文件哈希记录在上一次验证结果的 `run_manifest.yaml` 与
`artifacts/model_weights.sha256` 中。评估使用 LIBERO 四个 suite 自带的固定
initial states，每个任务取前 50 个状态；rollout 不读取训练集帧作为测试样本。

四组评估并行运行时的固定映射如下。`8100` 和 `8103` 是上一次完整评估实际
使用且验证通过的端口；当时 `8000` 和 `8003` 已被其他进程占用。

| Task suite | GPU | Server port | 最大动作步数 | Rollout 数 |
| --- | ---: | ---: | ---: | ---: |
| `libero_spatial` | 0 | 8100 | 280 | 10 tasks x 50 = 500 |
| `libero_object` | 1 | 8001 | 280 | 10 tasks x 50 = 500 |
| `libero_goal` | 2 | 8002 | 300 | 10 tasks x 50 = 500 |
| `libero_10` | 3 | 8103 | 520 | 10 tasks x 50 = 500 |

## 2. 评估配置

以下配置来自当前仓库代码、模型配置和数据元信息。启动命令保持这些默认值，
不要为了复现结果自行修改。

| 配置 | 值 | 来源 |
| --- | --- | --- |
| Dataset entry | `demo_data.libero_v21` | 服务端命令和 `dataset2feature.yaml` |
| Inference mode | `direct_action` | 服务端命令 |
| 服务端随机种子 | 42 | `server.py` |
| 客户端随机种子 | 7 | `run_libero_eval.py` |
| Observation window | 1 | `server.py` 默认值 |
| Action/prediction horizon | 10 | `action_expert_config.json` |
| Execution horizon / replan steps | 10 | `run_libero_eval.py` 默认值 |
| 去噪步数 | 5 | `server.py` 默认值 |
| 状态和动作最大 padding 长度 | 64 | `server.py` 默认值 |
| 每任务 rollout 数 | 50 | `run_libero_eval.py` 默认值 |
| 每个 episode 的稳定等待步数 | 10 | `run_libero_eval.py` 默认值 |
| LIBERO 控制频率 | 20 Hz | `LIBERO/libero/libero/envs/env_wrapper.py` 默认值 |
| 训练数据 FPS | 10 | `demo_data/libero_v21/meta/info.json` |

### 图像处理

- 使用 `agentview_image` 和 `robot0_eye_in_hand_image` 两个相机视角。
- LIBERO 原始渲染分辨率为 `256 x 256`，客户端先将两个视角旋转 180 度。
- 客户端使用双线性插值和保持长宽比的 zero padding resize 到 `448 x 448`；
  当前输入为正方形，因此不会产生额外黑边，然后转换为 `uint8`。
- 服务端构造 Qwen-VL 输入时，为每个视角指定 `224 x 224`；当前模型的实际
  `image_grid_thw` 为 `[1, 14, 14]`，patch size 为 16，因此送入视觉编码器的
  最终空间尺寸为 `224 x 224`。
- 不进行 crop。模型 processor 将像素乘以 `1/255`，再使用
  `mean=[0.5, 0.5, 0.5]`、`std=[0.5, 0.5, 0.5]` 归一化。
- `window_size=1`，每次推理只使用两个相机的当前帧。

### 状态和动作

- 状态为 8 维：末端位置 3 维、末端四元数转换得到的 axis-angle 3 维、
  gripper qpos 2 维。
- 动作为 LIBERO `OSC_POSE` 的 7 维控制量：末端平移 3 维、旋转 3 维和
  gripper 1 维。
- 模型内部动作维度 padding 到 64；输出时只截取前 7 维和前 10 个动作。
- 状态使用相同的 `q01/q99` 分位数做逐维 min-max normalization，8 个有效维度
  之后以 0 padding 到 64，并用 `state_mask` 标记前 8 维。推理动作通过
  `infer_action_mask` 标记每个 horizon 的前 7 维，剩余 57 维为 padding。
- `dataset2feature.yaml` 中 `use_quantile=true`，预测动作使用训练数据的
  `q01/q99` 统计量从归一化空间反归一化，然后直接传给 LIBERO 环境。
- 每次推理预测并执行 10 步，执行完后重新规划。LIBERO 环境控制频率为
  20 Hz，因此一次执行 chunk 对应 0.5 秒模拟控制时间。
- episode 成功由 LIBERO 环境返回的 `done=True` 判定；suite 成功率为
  成功 episode 数除以已完成 episode 数。四个 suite 共计 2000 个 episode。

## 3. 连接服务器并初始化运行目录

在本地连接远端服务器：

```bash
ssh -p 25408 lq@10.82.1.223
```

建议在远端创建一个 `tmux` 会话，然后在其中执行本节之后的所有命令：

```bash
tmux new -s zr0-libero-eval
```

设置固定路径并为本次评估创建新的时间戳目录。不要把 `RUN_DIR` 指向上一次
验证目录，否则会覆盖已有日志和视频。

```bash
cd /opt/data/private/lq/ZR-0

WORKSPACE=/opt/data/private/lq/ZR-0
MODEL_DIR=/opt/data/private/lq/models/ZR-0-libero
SERVER_PY=/opt/data/private/lq/.conda/envs/zr0-eval/bin/python
CLIENT_PY=/opt/data/private/lq/.conda/envs/zr0-libero-eval/bin/python
VERIFIED_RUN=/opt/data/private/lq/ZR-0/result/eval/ZR-0-LIBERO_official_seed7_50trials_20260830_195202
LIBERO_CONFIG_SOURCE="$VERIFIED_RUN/env/libero-config"
SUMMARY_SOURCE="$VERIFIED_RUN/artifacts/summarize_eval.py"

RUN_ID="ZR-0-LIBERO_official_seed7_50trials_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$WORKSPACE/result/eval/$RUN_ID"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
```

## 4. 启动前检查

检查代码、模型、环境和已有 LIBERO 配置是否完整：

```bash
required_paths=(
  "$WORKSPACE/server.py"
  "$WORKSPACE/evaluation/libero_eval/run_libero_eval.py"
  "$WORKSPACE/LIBERO"
  "$MODEL_DIR/action_expert.safetensors"
  "$MODEL_DIR/action_expert_config.json"
  "$SERVER_PY"
  "$CLIENT_PY"
  "$LIBERO_CONFIG_SOURCE/config.yaml"
  "$SUMMARY_SOURCE"
)

for required_path in "${required_paths[@]}"; do
  if [[ ! -e "$required_path" ]]; then
    echo "ERROR: missing required path: $required_path" >&2
    exit 1
  fi
done
```

查看 GPU 0-3 的型号、剩余显存和当前进程。这里只检查，不会停止已有进程：

```bash
nvidia-smi --id=0,1,2,3 \
  --query-gpu=index,name,memory.total,memory.used,memory.free \
  --format=csv
nvidia-smi
```

检查四个端口是否可以绑定。任何端口占用都会直接终止当前 shell；不要结束
不属于自己的进程。如需换端口，必须同时修改对应 suite 的服务端和客户端端口。

```bash
check_port_free() {
  "$CLIENT_PY" - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("0.0.0.0", port))
except OSError as exc:
    print(f"ERROR: port {port} is unavailable: {exc}", file=sys.stderr)
    raise SystemExit(1)
finally:
    sock.close()
PY
}

for port in 8100 8001 8002 8103; do
  check_port_free "$port" || exit 1
done
```

所有检查通过后再创建目录，并保存本次运行开始时的代码状态：

```bash
mkdir -p "$RUN_DIR"/{logs,videos,env,artifacts,summary,pids}
cp -a "$LIBERO_CONFIG_SOURCE" "$RUN_DIR/env/libero-config"
cp "$SUMMARY_SOURCE" "$RUN_DIR/artifacts/summarize_eval.py"

git -C "$WORKSPACE" rev-parse HEAD > "$RUN_DIR/env/code-commit.txt"
git -C "$WORKSPACE" status --short > "$RUN_DIR/env/git-status-start.txt"
nvidia-smi > "$RUN_DIR/env/nvidia-smi-start.txt"

cat > "$RUN_DIR/experiment.md" <<EOF
# ZR-0-LIBERO Evaluation

- Created: $(date --iso-8601=seconds)
- Owner: lq
- Checkpoint: $MODEL_DIR
- Checkpoint source: modelscope://seeklhy/ZR-0-libero
- Checkpoint revision: fd9ea33c44688398990014da13308df608ace3b8
- Corresponding training experiment: official release; no local training run directory
- Code commit: $(git -C "$WORKSPACE" rev-parse HEAD)
- Uncommitted changes: see env/git-status-start.txt
- LIBERO source: $WORKSPACE/LIBERO
- Dataset entry: demo_data.libero_v21
- Suites: libero_spatial, libero_object, libero_goal, libero_10
- Server seed: 42
- Client seed: 7
- Trials: 50 per task, 500 per suite, 2000 total
- Result directory: $RUN_DIR
- Full commands: simple_scripts/eval_libero.md
EOF

printf 'Result directory: %s\n' "$RUN_DIR"
```

记住终端打印的 `RUN_DIR`。后续日志、视频和汇总都写入该目录。

## 5. 启动四个模型服务端

定义清理函数和服务端启动函数。清理函数只处理当前 shell 保存的服务端 PID，
不会按名称批量结束其他进程。

```bash
SERVER_PIDS=()

cleanup_servers() {
  local pid
  for pid in "${SERVER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup_servers EXIT INT TERM

launch_server() {
  local suite=$1 gpu=$2 port=$3 pid
  CUDA_VISIBLE_DEVICES="$gpu" "$SERVER_PY" -u server.py \
    --dataset_entry demo_data.libero_v21 \
    --ckpt_dir "$MODEL_DIR" \
    --inference_mode direct_action \
    --port "$port" \
    > "$RUN_DIR/logs/server_${suite}.log" 2>&1 &
  pid=$!
  SERVER_PIDS+=("$pid")
  echo "$pid" > "$RUN_DIR/pids/server_${suite}.pid"
}

launch_server libero_spatial 0 8100
launch_server libero_object  1 8001
launch_server libero_goal    2 8002
launch_server libero_10      3 8103
```

模型加载完成后，`server.py` 会输出 `Start serving.`。下面最多等待 10 分钟，
并同时检查服务进程是否仍然存活：

```bash
wait_for_server() {
  local suite=$1 pid attempt
  pid=$(<"$RUN_DIR/pids/server_${suite}.pid")

  for ((attempt = 1; attempt <= 120; attempt++)); do
    if rg -q '^Start serving\.$' "$RUN_DIR/logs/server_${suite}.log"; then
      echo "$suite server is ready"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: $suite server exited before becoming ready" >&2
      tail -n 50 "$RUN_DIR/logs/server_${suite}.log" >&2
      return 1
    fi
    sleep 5
  done

  echo "ERROR: timed out waiting for $suite server" >&2
  tail -n 50 "$RUN_DIR/logs/server_${suite}.log" >&2
  return 1
}

wait_for_server libero_spatial || exit 1
wait_for_server libero_object  || exit 1
wait_for_server libero_goal    || exit 1
wait_for_server libero_10      || exit 1
```

可以再次核对 PID 和显存：

```bash
for pid_file in "$RUN_DIR"/pids/server_*.pid; do
  pid=$(<"$pid_file")
  ps -p "$pid" -o pid,etime,cmd
done
nvidia-smi
```

## 6. 启动四个 LIBERO 客户端

配置当前仓库内的 LIBERO 和无头 OSMesa 渲染：

```bash
export LIBERO_CONFIG_PATH="$RUN_DIR/env/libero-config"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=/opt/data/private/lq/.conda/envs/zr0-libero-eval/usr/lib/x86_64-linux-gnu:/opt/data/private/lq/.conda/envs/zr0-libero-eval/lib
```

启动四个客户端。这里仅传入 suite、对应服务端端口和视频目录，其他评估配置
继续使用脚本默认值。

```bash
CLIENT_PIDS=()

launch_client() {
  local suite=$1 port=$2 pid
  "$CLIENT_PY" -u -m evaluation.libero_eval.run_libero_eval \
    --args.task-suite-name "$suite" \
    --args.port "$port" \
    --args.video-out-path "$RUN_DIR/videos/$suite" \
    > "$RUN_DIR/logs/client_${suite}.attempt1.log" 2>&1 &
  pid=$!
  CLIENT_PIDS+=("$pid")
  echo "$pid" > "$RUN_DIR/pids/client_${suite}.pid"
}

launch_client libero_spatial 8100
launch_client libero_object  8001
launch_client libero_goal    8002
launch_client libero_10      8103
```

## 7. 查看进度并等待完成

可以新开一个 SSH shell 查看日志。新 shell 中先定位最新运行目录，再执行
`tail`；不要在这个监控 shell 中执行清理命令。

```bash
RUN_DIR=$(find /opt/data/private/lq/ZR-0/result/eval \
  -mindepth 1 -maxdepth 1 -type d \
  -name 'ZR-0-LIBERO_official_seed7_50trials_*' \
  -printf '%p\n' | sort | tail -n 1)
printf 'Monitoring: %s\n' "$RUN_DIR"
tail -F "$RUN_DIR"/logs/client_*.attempt1.log
```

也可以查看每个 suite 最近一次进度记录：

```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  echo "===== $suite ====="
  rg '# episodes completed so far:|Total success rate:' \
    "$RUN_DIR/logs/client_${suite}.attempt1.log" | tail -n 2
done
```

回到启动客户端的原 shell，等待四个客户端结束并记录退出码：

```bash
client_failed=0
for suite in libero_spatial libero_object libero_goal libero_10; do
  pid=$(<"$RUN_DIR/pids/client_${suite}.pid")
  if wait "$pid"; then
    echo 0 > "$RUN_DIR/pids/client_${suite}.exit_code"
    echo "$suite completed successfully"
  else
    exit_code=$?
    echo "$exit_code" > "$RUN_DIR/pids/client_${suite}.exit_code"
    echo "ERROR: $suite client exited with code $exit_code" >&2
    client_failed=1
  fi
done

if ((client_failed != 0)); then
  echo "At least one suite failed; inspect $RUN_DIR/logs" >&2
  exit 1
fi
```

正常完整评估中，每个 client 日志结尾应包含：

```text
Total success rate: ...
Total episodes: 500
```

## 8. 汇总结果

上一步四个客户端都以 0 退出后，使用已验证的汇总脚本生成 JSON、CSV 和
Markdown 结果。该脚本还会检查每组是否恰好完成 500 个 episode，以及日志中
是否出现客户端运行时错误。

```bash
"$CLIENT_PY" "$RUN_DIR/artifacts/summarize_eval.py"
cat "$RUN_DIR/summary/summary.md"

git -C "$WORKSPACE" status --short > "$RUN_DIR/env/git-status-end.txt"
nvidia-smi > "$RUN_DIR/env/nvidia-smi-end.txt"
```

结果目录结构如下：

```text
RUN_DIR/
├── artifacts/    # 本次使用的汇总脚本
├── env/          # 代码版本、工作区状态、LIBERO 配置和 GPU 快照
├── logs/         # 四个服务端日志和四个客户端日志
├── pids/         # PID 与客户端退出码
├── summary/      # summary.json、summary.csv、summary.md
├── videos/       # 各 suite 的 rollout 视频
└── experiment.md # checkpoint、代码版本和完整运行信息
```

## 9. 停止本次模型服务

汇总完成后，在原启动 shell 中停止本次启动的四个服务端，并取消退出 trap：

```bash
cleanup_servers
trap - EXIT INT TERM
```

确认本次 PID 已退出：

```bash
for pid_file in "$RUN_DIR"/pids/server_*.pid; do
  pid=$(<"$pid_file")
  if kill -0 "$pid" 2>/dev/null; then
    echo "WARNING: server PID $pid is still running" >&2
  fi
done
```

不要使用不带 PID 限定的 `pkill python`、`killall python` 等命令。

## 10. 上一次验证结果

上一次完整运行使用相同 checkpoint、默认参数和 GPU/端口映射，结果保存在：

```text
/opt/data/private/lq/ZR-0/result/eval/ZR-0-LIBERO_official_seed7_50trials_20260830_195202
```

| Suite | Episodes | Successes | Success rate |
| --- | ---: | ---: | ---: |
| `libero_spatial` | 500 | 494 | 98.8% |
| `libero_object` | 500 | 497 | 99.4% |
| `libero_goal` | 500 | 493 | 98.6% |
| `libero_10` | 500 | 468 | 93.6% |
| **总计** | **2000** | **1952** | **97.6%** |

完整配置和校验信息分别见该目录下的 `run_manifest.yaml` 和
`summary/verification.log`。

## 11. 常见问题

### 端口已占用

启动前检查会打印被占用的端口并退出。不要结束来源不明的进程。选择其他空闲
端口后，需要同时修改该 suite 的 `launch_server` 和 `launch_client` 参数。

### 服务端未就绪或提前退出

查看对应日志，例如：

```bash
tail -n 100 "$RUN_DIR/logs/server_libero_spatial.log"
```

优先检查 checkpoint 路径、GPU 剩余显存和 CUDA 报错。不要通过减少 action
horizon、去噪步数或图像尺寸规避错误，否则评估配置不再与默认设置一致。

### OSMesa 或 OpenGL 初始化失败

确认客户端是在设置 `LIBERO_CONFIG_PATH`、`MUJOCO_GL`、
`PYOPENGL_PLATFORM` 和 `LD_LIBRARY_PATH` 的同一个 shell 中启动，并使用的是
`zr0-libero-eval` 环境中的 Python。

### 客户端中途失败

不要把不完整日志作为最终成功率。保留失败目录用于排查，在新的时间戳目录中
重新运行；只有四个 suite 都完成 500 个 episode 后再执行正式汇总。
