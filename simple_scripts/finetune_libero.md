# LIBERO 微调：手动启动、恢复与参数修改

本文对应 Stage 3 step-14000 初始化的 LIBERO action-only 微调。训练代码使用已有 `3eefb602417d3bd4b20bef2b47660b404aefb565` 快照。下面的命令不会因打开文档而执行；`dry-run` 仅展开配置。

## 1. 本次实际使用的命令

本次在仓库目录下，通过后台进程执行：

```bash
cd /opt/data/private/lq/ZR-0
PYTHONNOUSERSITE=1 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -u \
  scripts/watch_libero_finetune.py --execute
```

这个 supervisor 内部执行：

```bash
bash /opt/data/private/lq/ZR-0/scripts/run_libero_wo_ecot_pt.sh train difference_query_stage3
```

它把 source、输出目录、W&B run 和 GPU UUID 通过环境变量传给 launcher；launcher 再调用 Accelerate 和 `scripts/train_libero_finetune.py`。本次完整展开命令、全部默认参数和权重/统计身份见：

- [launch_manifest_train.json](../outputs/ckpts/ZR0-stage3-step14000-LIBERO-action-only-dq32-h10-gbs64-seed42/attempt-000/launch_manifest_train.json)
- [实验说明](../docs/experiments/libero_stage3_step14000/experiment.md)
- [W&B run](https://wandb.ai/jumbo3r-zhejiang-university/ZR-0-LIBERO/runs/cc7a748a)

`watch_libero_finetune.py` 是这次实验的专用 supervisor，内部固定了 source、34,184 步预算和四卡配置。它只提供 `--execute`、`--output-root`，不是可自由传入学习率/轮数的通用入口。默认输出目录已存在，直接重跑上面的历史命令会拒绝覆盖。下次手动微调使用第 2 节；修改超参数使用第 5 节。

## 2. 下次使用相同训练配置

建议先在终端进入一个新 tmux 会话，再执行本节。不同实验必须使用不同输出目录和 W&B run ID。需要四张空闲 GPU，并已通过 `wandb login` 或环境变量配置 W&B 凭据；不要把 API key 写入文档或启动日志。

```bash
tmux new -s libero-finetune
```

下面的配置和函数在同一个 Bash 终端中执行。通常只需要修改 `LIBERO_SOURCE` 和 `LIBERO_RUN_NAME`。GPU、输出路径、保存间隔也可以修改，但该入口仍要求四卡。

<!-- BEGIN MANUAL SETUP -->
```bash
cd /opt/data/private/lq/ZR-0
source /opt/data/private/lq/miniconda3/etc/profile.d/conda.sh
conda activate ZR-0

LIBERO_REPO=/opt/data/private/lq/ZR-0
LIBERO_PYTHON=/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python
LIBERO_RUNTIME="$LIBERO_REPO/outputs/runtime_snapshots/3eefb602417d3bd4b20bef2b47660b404aefb565"
LIBERO_SOURCE="$LIBERO_REPO/outputs/three_stage_formal_20260910/stage3_resume8000_slot05_flow5/recovery_checkpoints/stage3_joint/step-014000-attempt-000/latest-model-optimizer-lr"
LIBERO_RUN_NAME="libero-dq32-manual-$(date +%Y%m%d-%H%M%S)"
LIBERO_OUTPUT="$LIBERO_REPO/outputs/ckpts/$LIBERO_RUN_NAME"
LIBERO_RUN_ID=$("$LIBERO_PYTHON" -c 'import uuid; print(uuid.uuid4().hex[:8])')
LIBERO_GPUS=0,1,2,3
LIBERO_SAVE_EVERY=2000
LIBERO_RESUME=

libero_manual() (
  set -euo pipefail
  local requested_mode=${1:-dry-run}
  local launch_mode
  case "$requested_mode" in
    dry-run|train) launch_mode=train ;;
    dry-run-resume|resume) launch_mode=resume ;;
    *) printf 'Usage: libero_manual {dry-run|train|dry-run-resume|resume}\n' >&2; return 2 ;;
  esac
  if [[ "$launch_mode" == resume && -z "$LIBERO_RESUME" ]]; then
    printf 'Set LIBERO_RESUME to a complete LIBERO checkpoint first.\n' >&2
    return 2
  fi
  # Always use a new output, including same-stage resume: verification files are exclusive.
  if [[ -e "$LIBERO_OUTPUT" ]]; then
    printf 'Choose a new output directory: %s\n' "$LIBERO_OUTPUT" >&2
    return 2
  fi
  unset ZR0_DRY_RUN ZR0_MAX_TRAIN_STEPS
  local launch_env=(env
    PYTHONNOUSERSITE=1 CONDA_DEFAULT_ENV=ZR-0 WANDB_MODE=online
    OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
    "PATH=$(dirname "$LIBERO_PYTHON"):$PATH"
    "PYTHONPATH=$LIBERO_RUNTIME:$LIBERO_RUNTIME/lerobot"
    "ZR0_RUNTIME_ROOT=$LIBERO_RUNTIME" "ZR0_TRAIN_PYTHON=$LIBERO_PYTHON"
    "CUDA_VISIBLE_DEVICES=$LIBERO_GPUS" "ZR0_CUDA_VISIBLE_DEVICES=$LIBERO_GPUS"
    "ZR0_PRETRAIN_JOINT_CKPT=$LIBERO_SOURCE" "ZR0_RESUME_CKPT=$LIBERO_RESUME"
    "ZR0_OUTPUT_DIR=$LIBERO_OUTPUT" "ZR0_RUN_NAME=$LIBERO_RUN_NAME"
    "ZR0_WANDB_RUN_ID=$LIBERO_RUN_ID" "ZR0_SAVE_STEP_INTERVAL=$LIBERO_SAVE_EVERY"
    ZR0_LIBERO_ACTION_HORIZON=10)
  if [[ "$requested_mode" == dry-run* ]]; then
    "${launch_env[@]}" ZR0_DRY_RUN=1 bash "$LIBERO_REPO/scripts/run_libero_wo_ecot_pt.sh" \
      "$launch_mode" difference_query_stage3
    return
  fi
  mkdir -p "$(dirname "$LIBERO_OUTPUT")"
  local gate devices
  gate=$("${launch_env[@]}" "$LIBERO_PYTHON" -m utils.gpu_resource_gate \
    --visible-devices "$LIBERO_GPUS" --expected-count 4 --log "${LIBERO_OUTPUT}.gpu-gate.jsonl")
  devices=$("$LIBERO_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["cuda_visible_devices"])' <<< "$gate")
  launch_env+=("CUDA_VISIBLE_DEVICES=$devices" "ZR0_CUDA_VISIBLE_DEVICES=$devices")
  "${launch_env[@]}" bash "$LIBERO_REPO/scripts/run_libero_wo_ecot_pt.sh" \
    "$launch_mode" difference_query_stage3 2>&1 | tee -a "${LIBERO_OUTPUT}.log"
)
```
<!-- END MANUAL SETUP -->

先预览；预览不启动 GPU 训练、不创建输出目录、不重新审计数据：

```bash
libero_manual dry-run
```

确认终端打印的 source、output、batch、loss、W&B 身份后，手动启动：

```bash
libero_manual train
```

入口自动进行 GPU UUID 门禁及环境/W&B/checkpoint 检查，创建实验说明、启动 manifest，并在第一次更新前逐张量比较 VLM/Query/Expert 来源。每次保存完整状态时还会生成不可覆盖的恢复副本。这里只手动启动一个训练进程组，没有调用自动重试 supervisor。

启动命令在 tmux 中前台运行，`Ctrl-b` 后按 `d` 可退出 tmux 显示而保留训练；重新进入用 `tmux attach -t libero-finetune`。需要停止时在该会话中按 `Ctrl-c`，保留已有完整 checkpoint。中断不会自动保存未到保存点的更新。

## 3. 手动恢复同一个 LIBERO 实验

先确认原进程已退出，且 source 是本次 LIBERO 的完整 checkpoint。`step-2000` 通常只是模型目录，不能用作完整 optimizer resume；优先使用带 `checkpoint_complete.json` 的恢复副本，其实际模型目录名为 `latest-model-optimizer-lr`。

保留第 2 节的 `LIBERO_SOURCE` 为原预训练来源，用于 Expert 结构检查；设置恢复目录和原 W&B 身份，并为此次恢复选一个新的输出目录：

```bash
LIBERO_PREVIOUS_RUN=/opt/data/private/lq/ZR-0/outputs/ckpts/你的上一次微调目录
LIBERO_RESUME="$LIBERO_PREVIOUS_RUN/recovery_checkpoints/step-002000/latest-model-optimizer-lr"
LIBERO_RUN_ID=$(<"$LIBERO_PREVIOUS_RUN/.wandb-run-id")
LIBERO_RUN_NAME=$(<"$LIBERO_PREVIOUS_RUN/.wandb-run-name")
LIBERO_OUTPUT="${LIBERO_PREVIOUS_RUN}-resume-$(date +%Y%m%d-%H%M%S)"
libero_manual dry-run-resume
```

```bash
libero_manual resume
```

这里恢复完整模型、AdamW 状态、scheduler 和 global step，总目标仍为 34,184 步，**不是追加 34,184 步**。W&B 使用原 run ID 和 `resume=must`。旧 LIBERO 数据位置按已有逻辑恢复，不承诺各 rank RNG 或连续训练轨迹完全等价。

不要把 `LIBERO_OUTPUT` 设成原目录：专用入口的初始化核验文件采用独占创建，重复写入会报错。也不要为 resume 生成新的 W&B run ID；没有原在线 run 时 `resume=must` 会失败。

## 4. 哪些参数可以修改

### 4.1 第 2 节可以直接修改的参数

| 参数 | 用途 | 修改要求 |
| --- | --- | --- |
| `LIBERO_SOURCE` | 新微调的 VLM、Query、Expert 权重来源 | 选择已完成的 `joint` checkpoint；包含完整 VLM/processor、DQ32、Expert；本快捷入口的参考配置要求 horizon 10 |
| `LIBERO_RUN_NAME` | W&B 显示名称、默认日志路径 | 新实验使用新名称；resume 保留原名称 |
| `LIBERO_OUTPUT` | 新模型、optimizer、训练日志、核验记录目录 | 必须是尚不存在的新目录；初次启动和恢复均如此 |
| `LIBERO_RUN_ID` | W&B 唯一身份 | 新实验重新生成；resume 从原目录读取 |
| `LIBERO_GPUS` | 指定使用哪四张卡 | 必须有四张可用卡；支持四个 ordinal 或 UUID；不能只删成两张 |
| `LIBERO_SAVE_EVERY` | 每隔多少次有效更新保存 checkpoint | 正整数；越小保存越频繁、I/O 和空间开销越大；不是训练总步数 |
| `LIBERO_RESUME` | 完整 LIBERO 恢复目录 | 仅 `resume` 使用；不能填预训练模型来伪装同阶段恢复 |
| `LIBERO_RUNTIME` | Python 实际导入的训练代码 | 为复用本实验保留指定快照；改变版本属于新实验，需要重新验证代码行为 |

这条快捷入口沿用固定 W&B project/group：`ZR-0-LIBERO` / `libero-stage3-step14000-action-only-dq32-seed42`。换了预训练来源后，自己的 `LIBERO_RUN_NAME` 应写清来源；需要独立 project/group 时使用第 5 节。

### 4.2 训练参数的含义和修改位置

以下值是本次默认值。第 2 节的环境变量只覆盖上表中的项目；不存在通用的 `ZR0_LR` 或 `ZR0_EPOCHS` 覆盖机制。修改下表中的超参数，请使用第 5 节的原生命令。

| 参数 | 默认值 | 用途与修改影响 |
| --- | --- | --- |
| `--epochs` | `8` | 完整遍历数据的次数；当前每 epoch 4,273 次更新，8 epoch 共 34,184 次 |
| `--peak_learning_rate` | `2e-5` | 所有可训练参数的峰值 LR；改变会改变优化过程 |
| `--min_lr_rate` | `0.1` | 最低 LR / 峰值 LR；当前最低 LR 为 `2e-6` |
| `--lr_scheduler` | `cosine` | warmup 后余弦衰减；原生入口也支持 `constant` |
| `--warmup_ratio` | `0.08` | warmup 占总预算的比例；当前 2,734 次全局更新 |
| `--adam_beta1 / --adam_beta2` | `0.9 / 0.95` | AdamW 一阶、二阶动量的衰减系数 |
| `--adam_epsilon` | `1e-6` | AdamW 分母的数值稳定项 |
| weight decay | `0.01` | 参数衰减；写在 pinned `train_vla.py::build_adamw_optimizer` 中，没有 `--weight_decay` CLI |
| `--seed` | `42` | 采样、初始化和随机训练过程的种子；新实验可改，resume 保持一致 |
| `--per_device_train_batch_size` | `16` | 每张卡每次 forward 的样本数，影响显存与吞吐 |
| `--gradient_accumulation_steps` | `1` | 一次更新累计多少个 micro-batch |
| `--expected_global_batch_size` | `64` | 运行时断言期望全局 batch；不是自动调整 batch 的开关 |
| `--num_processes` | `4` | Accelerate 启动的训练 rank 数，需与 GPU 数和 YAML 一致 |
| `--dataloader_num_workers` | `24` | 每 rank 的数据加载 worker 数；调小降低 CPU/内存占用，可能影响供数速度 |
| `--prefetch_factor` | `3` | 每个 worker 预取的 batch 数，影响内存和加载等待 |
| `--logging_steps` | `10` | 日志记录间隔；不是 optimizer 更新频率 |
| `--save_step_interval` | `2000` | 按有效更新次数保存模型和完整训练状态 |
| `--save_ckpt_interval` | `4` | 按 epoch 保存的间隔；与按 step 保存同时生效 |
| `--max_length` | `1200` | 输入 token 序列上限；调小可能截断任务上下文 |
| `--window_size` | `1` | 输入历史帧窗口长度；改变属于模型输入合同变化，不作为普通调参项 |
| `--action_horizon` | `10` | 预测动作序列长度；不是环境实际执行步数；改变需核验数据、Expert 和评估一致性 |
| `--max_pad_state_and_action_length` | `64` | 模型 state/action padding 维度；须与 checkpoint 架构一致，不能改为有效维度 8/7 |
| `--num_difference_queries` | `32` | 已训练 Query 数量；须与 source 权重一致，不可仅改为 16 或 64 |
| `--vlm_attention_backend` | `sdpa` | VLM attention 实现；更换需验证环境及数值行为 |
| `--tune_vlm / --tune_action_expert` | 均启用 | 训练完整 VLM/Query 与 Expert；移除会改变训练模块范围 |
| `--loss_type` | `action` | 只使用 FM 动作目标，不计算 AR CE |
| `--action_expert_loss_weight` | `1.0` | FM 外层系数，当前 `total_loss = FM` |
| `--vlm_loss_weight` | `0.0` | AR 外层系数；仅增大它不会把 action-only 自动改成联合训练 |
| `--slot_loss_weight / --optical_flow_loss_weight` | `0.0 / 0.0` | 辅助目标外层系数；Heads 和数据标签均关闭，不能只改系数来开启 |
| `--detach_vlm_outputs_for_action_expert` | 未启用 | 默认 FM 梯度能回传到 VLM/Query；启用会改变梯度路径 |
| `--use_lora` | 未启用 | 当前全参数训练；LoRA 的 rank/alpha 等默认字段不生效 |
| `--save_optimizer_and_lr_states` | 启用 | 保存完整训练恢复状态；关闭后模型保存不等于可 resume |
| `--wandb_project / --wandb_group / --wandb_run_name` | 见第 1 节记录 | 在线实验分组与显示名称，新实验可改 |
| `--wandb_run_id / --wandb_resume` | 新 ID / `never` | 新实验身份；同阶段恢复用原 ID / `must` |
| `--wandb_failure_policy` | `required` | 正式训练要求 W&B；不通过切换 offline/disabled 绕过 |

BF16、ZeRO-2、clip `1.0`、offload 关闭及 batch 的 DeepSpeed 配置位于 `accelerate_configs/libero_zero2_bf16_mbs16_gas1.yaml`。若改变 GPU/micro-batch/GAS，需创建独立 YAML，同时修改 `num_processes`、`deepspeed_config.gradient_accumulation_steps`、`train_micro_batch_size_per_gpu`、`train_batch_size` 及命令行对应值，满足 `global_batch = GPUs * micro_batch * GAS`。不要直接改正在运行实验使用的原 YAML 或 pinned 源码。

专用 `train_libero_finetune.py` 会校验固定 epochs、seed、batch、horizon、loss、Adam 超参数；专用 manifest recorder 也校验 34,184 步。设置 `ZR0_MAX_TRAIN_STEPS` 不会解除这些限制。原生入口的 `--max_train_steps` 还会改变 scheduler 总长度；若只是保存后提前退出，原生入口提供独立的 `--save_and_exit_after_updates`，两者不是同一含义。

图像配置沿用双视角 `256x256 -> 224x224` bicubic、无增强、mean/std 均 `[0.5,0.5,0.5]`；有效 state/action 为 8/7 维，使用 LIBERO 自身 q01/q99 统计，归一化后截断 `[-15,15]`。这些不是普通 launcher 环境变量，修改需要验证数据/processor/模型/评估合同。数据是 1,693 episodes、273,465 frames、40 tasks、10 FPS，全量训练，无验证 loader 或自动 rollout。

## 5. 自定义学习率、轮数等：原生训练命令

此路径用于**新建**参数不同的实验。它直接复用 pinned `train_vla.py`，保留其原有 checkpoint 合同检查、初始化记录及 W&B；它没有第 2 节 wrapper 的首次更新前逐张量对比、完整 checkpoint 自动归档或自动重试。原生入口照常保存 `step-N` 模型和 `latest-model-optimizer-lr` 完整状态。不要把修改后的实验说成与当前实验完全相同。

先执行第 2 节配置块；下列数组集中列出可编辑参数。默认仍为本次配置，通常只需修改 `--epochs`、`--peak_learning_rate`、`--warmup_ratio`、`--seed` 等。输出目录和 run ID 每次重新生成。

<!-- BEGIN CUSTOM COMMAND -->
```bash
LIBERO_CUSTOM_NAME="libero-custom-$(date +%Y%m%d-%H%M%S)"
LIBERO_CUSTOM_OUTPUT="$LIBERO_REPO/outputs/ckpts/$LIBERO_CUSTOM_NAME"
LIBERO_CUSTOM_ID=$("$LIBERO_PYTHON" -c 'import uuid; print(uuid.uuid4().hex[:8])')
LIBERO_CUSTOM_COMMAND=(
  "$LIBERO_PYTHON" -m accelerate.commands.launch
  --num_processes 4
  --config_file "$LIBERO_REPO/accelerate_configs/libero_zero2_bf16_mbs16_gas1.yaml"
  "$LIBERO_RUNTIME/train_vla.py"
  --vlm_name_or_path "$LIBERO_SOURCE"
  --action_expert_name_or_path "$LIBERO_SOURCE"
  --action_expert_config_path "$LIBERO_SOURCE/action_expert_config.json"
  --checkpoint_load_purpose downstream_finetune
  --FAST_tokenizer_path "$LIBERO_REPO/fast"
  --dataset_entries libero_wo_ecot_pt
  --tune_vlm --tune_action_expert --use_difference_query
  --num_difference_queries 32 --vlm_attention_backend sdpa
  --loss_type action --action_expert_loss_weight 1.0
  --vlm_loss_weight 0.0 --slot_loss_weight 0.0 --optical_flow_loss_weight 0.0
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1
  --expected_global_batch_size 64
  --epochs 8 --seed 42
  --peak_learning_rate 2e-5 --min_lr_rate 0.1
  --lr_scheduler cosine --warmup_ratio 0.08
  --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-6
  --window_size 1 --action_horizon 10 --max_pad_state_and_action_length 64
  --max_length 1200 --dataloader_num_workers 24 --prefetch_factor 3
  --save_ckpt_interval 4 --save_step_interval 2000 --save_optimizer_and_lr_states
  --logging_steps 10 --log_training_diagnostics
  --output_ckpt_dir "$LIBERO_CUSTOM_OUTPUT"
  --tensorboard_log_dir "$LIBERO_CUSTOM_OUTPUT/tensorboard"
  --wandb_project ZR-0-LIBERO --wandb_group libero-manual-custom
  --wandb_run_name "$LIBERO_CUSTOM_NAME" --wandb_run_id "$LIBERO_CUSTOM_ID"
  --wandb_resume never --wandb_failure_policy required
  --wandb_dir "$LIBERO_CUSTOM_OUTPUT/wandb"
)
printf '%q ' "${LIBERO_CUSTOM_COMMAND[@]}"
printf '\n'
```
<!-- END CUSTOM COMMAND -->

上面的代码只构造、打印命令，不训练。确定参数后执行下面的启动块。该块复用 GPU/W&B/checkpoint preflight，记录实际参数和完整命令，创建独立 `experiment.md`，最后才训练。示例固定四卡和 horizon 10；改变这些合同还需同步本块 preflight 及 YAML。

```bash
(
  set -euo pipefail
  export PYTHONNOUSERSITE=1 CONDA_DEFAULT_ENV=ZR-0 WANDB_MODE=online
  export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
  export PYTHONPATH="$LIBERO_RUNTIME:$LIBERO_RUNTIME/lerobot"
  export CUDA_VISIBLE_DEVICES="$LIBERO_GPUS"
  cd "$LIBERO_RUNTIME"
  if [[ -e "$LIBERO_CUSTOM_OUTPUT" ]]; then
    printf 'Choose a new output: %s\n' "$LIBERO_CUSTOM_OUTPUT" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$LIBERO_CUSTOM_OUTPUT")"
  gate=$("$LIBERO_PYTHON" -m utils.gpu_resource_gate --visible-devices "$LIBERO_GPUS" \
    --expected-count 4 --log "${LIBERO_CUSTOM_OUTPUT}.gpu-gate.jsonl")
  CUDA_VISIBLE_DEVICES=$("$LIBERO_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["cuda_visible_devices"])' <<< "$gate")
  export CUDA_VISIBLE_DEVICES
  "$LIBERO_PYTHON" "$LIBERO_RUNTIME/scripts/preflight_libero_wo_ecot_pt.py" \
    --model-path "$LIBERO_SOURCE" --fast-path "$LIBERO_REPO/fast" \
    --dataset-path /opt/data/private/lq/datasets/HuggingFaceVLA/libero \
    --expected-checkpoint-kind joint --expected-num-difference-queries 32 \
    --expected-source-action-horizon 10 \
    --reference-action-expert-config "$LIBERO_SOURCE/action_expert_config.json" \
    --min-cgroup-memory-headroom-gib 120 --min-gpu-free-gib 70 \
    --output-path "$LIBERO_CUSTOM_OUTPUT" --min-output-free-gib 200 --require-wandb
  "$LIBERO_PYTHON" - "$LIBERO_CUSTOM_OUTPUT" "$LIBERO_REPO" "${LIBERO_CUSTOM_COMMAND[@]}" <<'PY'
import json
from pathlib import Path
import shlex
import subprocess
import sys
from utils.cli_options import parse_train_options

output, repository = map(Path, sys.argv[1:3])
command = sys.argv[3:]
entry = next(i for i, token in enumerate(command) if token.endswith('/train_vla.py'))
options = vars(parse_train_options(command[entry + 1:]))
output.mkdir(exist_ok=False)
record = dict(command=command, options=options,
              git_status=subprocess.check_output(['git', 'status', '--short'], cwd=repository, text=True))
(output / 'manual_launch.json').write_text(json.dumps(record, indent=2) + '\n')
(output / '.wandb-run-id').write_text(options['wandb_run_id'] + '\n')
(output / '.wandb-run-name').write_text(options['wandb_run_name'] + '\n')
guide = (repository / 'simple_scripts/finetune_libero.md').read_text()
(output / 'launch_guide.md').write_text(guide)
(output / 'experiment.md').write_text(
    '# Manual LIBERO Fine-Tuning\n\n'
    'Owner: lq. Fresh downstream action-only training, initialized from the source below.\n'
    'Runtime: pinned 3eefb602. VLM vision/merger/DeepStack/language, Query and Expert train; '
    'no auxiliary Heads, LoRA or detach. Optimizer/scheduler/global step reset.\n'
    'LIBERO: 1693 episodes, 273465 frames, 40 tasks, all train, 10 FPS. '
    'Own q01/q99 stats, clip [-15,15], state/action dimensions 8/7 padded to 64. '
    'Two current RGB views in image/image2 order, 256x256 to 224x224 bicubic, '
    'no crop/pad/augmentation; rescale 1/255, mean/std .5. '
    'No rollout or validation; execution horizon is unspecified.\n'
    'Four GPUs, BF16, ZeRO-2, no offload, clip 1; VLM gradient checkpointing. '
    'One AdamW group, weight decay .01. Exact selected hyperparameters, trainable switches, '
    'loss weights, seed, sources, outputs and W&B identity follow in the options. '
    'Batch = world size * micro-batch * GAS. Epoch budget and warmup derive from the options.\n'
    'No automatic retry/archive or wrapper tensor comparison. Native initialization and '
    'training/checkpoint diagnostics remain enabled.\n\n'
    '## Effective Options\n\n```json\n' + json.dumps(options, indent=2) + '\n```\n\n'
    '## Command\n\n```bash\n' + shlex.join(command) + '\n```\n\n'
    '## Results\n\nNot started. Fill start/end time, actual updates, checkpoint, lowest loss, '
    'W&B URL, failures/resumes and any configuration deviations from training.log. '
    'See launch_guide.md for the fixed data/image/action contracts and parameter meanings.\n')
PY
  "${LIBERO_CUSTOM_COMMAND[@]}" 2>&1 | tee -a "$LIBERO_CUSTOM_OUTPUT/training.log"
)
```

原生命令中的默认总 loss 仍为 `FM`，VLM/Query 通过 FM 接收梯度。与学习率、epoch 不同，改训练目标、Query 数、state/action 维度、图像尺寸、数据集或 LoRA 属于结构/数据合同变化，不能仅修改一个数值就视为已验证。

自定义实验结束或中断后，补充输出目录的 `experiment.md`：实际起止时间、完成步数、最终完整 checkpoint、最低训练 loss、W&B URL、NaN/OOM/中断及恢复情况。保留模型、日志、统计及启动配置；不要覆盖原实验。

## 6. 验证范围

本文使用现有 launcher、GPU 门禁、preflight、checkpoint 加载与生产训练循环，不修改正在运行实验的代码。文档中的 Bash 块进行语法检查；快捷入口仅执行 dry-run；自定义命令使用 pinned CLI parser 检查默认及修改后的超参数，不执行 GPU 训练。当前已运行实验的完整模型证据见第 1 节实验说明，自定义配置需要自己的实际训练验证。
