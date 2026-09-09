# 预训练模型语言能力检查

`inspect_vlm_text.py` 对本地 checkpoint 的 Qwen3-VL 做独立的纯文本问答，便于人工检查中文、英文、推理与指令遵循，以及使用同一组问题比较预训练前后的输出。借鉴 `FD-ID-FlowVLA/simple_script/vlm_text` 的 `question-chat` 模式，不计算自动能力分数。

## 单题检查

在当前仓库根目录执行：

```bash
/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python \
  simple_scripts/vlm_text/inspect_vlm_text.py \
  --checkpoint /opt/data/private/lq/models/ZR-0 \
  --question '请用中文解释机器人模仿学习，并给出一个具体例子。'
```

默认 CPU、FP32、4 个 CPU 线程、SDPA、固定 seed 42、最多生成 256 个 token；只读取本地文件，不联网下载。CPU 推理较慢。需要 GPU 时显式指定，例如使用一张空闲卡：

```bash
CUDA_VISIBLE_DEVICES=0 \
/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python \
  simple_scripts/vlm_text/inspect_vlm_text.py \
  --checkpoint /opt/data/private/lq/models/ZR-0 \
  --device cuda:0 \
  --max-new-tokens 512 \
  --question '只回答计算结果：17乘以23等于多少？' \
  --question 'Explain the difference between position and velocity in two sentences.' \
  --output-dir outputs/vlm_text/zr0_questions_001
```

`--device` 中的序号对应 `CUDA_VISIBLE_DEVICES` 设置后的逻辑设备编号。`--dtype auto` 在支持 BF16 的 CUDA 设备使用 BF16，其他情况使用 FP32；可以显式选择 `float32`、`bfloat16` 或 CUDA `float16`。不会自动更换设备、降低精度或重试 OOM。使用现有 ZR-0 环境，无须修改依赖；已验证的版本为 PyTorch 2.6.0、Transformers 4.57.1。

生成固定为 `do_sample=false`、`num_beams=1`，保留 checkpoint 的停止 token。显式使用 `use_model_defaults=false`，防止 Transformers 将 checkpoint 中的采样温度、top-p 等设置回填进这次检查；这些选项只作用于当前推理，不修改磁盘配置。

## 批量与对照检查

可重复传入 `--question`，也可传 `--questions-file /absolute/questions.jsonl`，两者互斥。UTF-8 JSONL 每行一个对象，例如：

```jsonl
{"id":"zh_explain","question":"用两句话解释什么是过拟合。"}
{"id":"arithmetic","question":"只输出数字：17乘以23等于多少？","reference":"391"}
{"id":"instruction","question":"List exactly three colors, one per line, with no numbering."}
{"id":"translation","question":"将这句话翻译成英文：请把红色杯子放到盘子左边。"}
```

`id` 可省略，默认按顺序从 `"1"` 编号；显式 ID 必须是唯一的非空字符串。`reference` 仅用于人工对照，不会加入模型输入。每题独立，没有跨题对话历史。可选 `--system-prompt` 会应用到每一题。

更换 `--checkpoint` 即可比较原始 `/opt/data/private/lq/models/Qwen3-VL-2B-Instruct`、`/opt/data/private/lq/models/ZR-0` 和本仓库 `save_pretrained()` 导出的完整预训练 checkpoint。对照时保持问题、system prompt、dtype、attention backend 和 token 上限一致，并使用不同输出目录。`--training-experiment` 可记录来源训练实验路径或名称。

## 输出

终端打印每题回答。未指定 `--output-dir` 时自动创建仓库下 `outputs/vlm_text/<时间戳>/`；显式指定时必须是尚不存在的新目录，且不能位于 checkpoint 内部。

- `result.json`：checkpoint 路径、配置/processor JSON 与模板 SHA256、权重文件大小和 mtime、代码 commit 与工作树状态、完整命令、实际 device/dtype、完整生成配置、问题、参考文本、实际输入/输出 token、回答、耗时及运行状态。
- `experiment.md`：运行说明、来源 checkpoint、启动命令、输入和运行状态。此工具仅做推理，没有训练、W&B、数据集 split、图像预处理或动作 rollout。

每题结束后保存结果；后续题目失败会保留已完成回答，并将运行标记为 `failed`。`complete` 只表示所有生成调用完成，不表示模型回答正确。`hit_max_new_tokens` 表示达到 token 上限；`stopped_on_eos` 表示最后一个 token 是停止 token；只有达到上限且没有停止 token 才标记 `truncated=true`。截断时应提高 `--max-new-tokens` 并使用新的输出目录重新检查。

## 范围与兼容性

脚本只在显式运行时生效，不被训练、模型、数据集或机器人评估入口导入。它复用本仓库 `QwenVLBackbone` 所用的 `Qwen3VLForConditionalGeneration` / `AutoProcessor` 接口，以及 `ZR0Model.save_pretrained()` 的 checkpoint 布局；不实例化完整 VLA，不加载 Action Expert、Difference Query、Slot 或 Flow。

这检查的是 **checkpoint 中 VLM 的通用文本问答能力**，不代表带图像/state/DQ 的 future-difference 生成能力或机器人任务成功率。当前仓库仍保持 DQ `generate()` 的原有禁止逻辑；本脚本没有移除或绕过运行中的模型保护。视觉权重作为完整 Qwen3-VL 的一部分被加载，但纯文本输入不运行视觉分支。

仅支持具有完整 Qwen3-VL safetensors 权重和 processor 的本地目录；不支持 FD-ID-FlowVLA 的 `.pt` 格式、仅包含 Action Expert 的目录或未合并的 LoRA adapter。遇到这些输入直接报错，不静默退回基座模型；缺失或不匹配的 VLM 参数也会报错。权重身份仅记录文件名/大小/mtime，不宣称做过权重全文 SHA256 校验。

参考源：`/opt/data/private/lq/FD-ID-FlowVLA/simple_script/vlm_text/inspect_vlm_text.py::generate_plain_chat`，commit `3824d36cdf76bf0a9d537635de92a38f3920e9a3`。借鉴聊天模板、只解码新增 token 和保存人工检查证据的方式；未引入该项目的 StarVLA、数据集合同或 DQ 模型依赖。

## 开发验证

```bash
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 \
/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python \
  -m unittest discover -s tests -p test_vlm_text_inspection.py -v
```

2026-09-08：6 项 CPU 测试通过，涵盖问题校验、adapter/缺失分片拒绝、输入前缀去除、EOS/截断、实际 Transformers 采样配置回填的禁用、独立问题、推理模式、上下文上限、防覆盖及失败记录。真实 ZR-0 的两题 CPU/FP32 检查保存在 `outputs/vlm_text/zr0_cpu_smoke_20260908_greedy/`。模型将 `17 × 23` 回答为 `1931`，翻译题输出机器人推理文本并达到 32-token 上限；推理流程通过不代表语言能力通过。未运行 GPU 检查或训练。
