# ZR-0 模型架构、训练与推理流程深度分析

> 分析基线：仓库 `082a6a5` 及当前工作树；当前实验 checkpoint 为 `outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT/step-2000`，官方推理基线为 `/opt/data/private/lq/models/ZR-0-libero`。行号均对应分析时的当前文件。  
> 证据标记：**[代码明确]** 表示执行代码直接实现；**[配置决定]** 表示取决于本次生效配置或 checkpoint；**[无法确认]** 表示仓库和 checkpoint 均没有足够语义信息。

## 1. 一句话总结模型

ZR-0 是一个以 Qwen3-VL-2B 因果多模态解码器提供图像/指令前缀特征、再由 9 层“3 个双向 self-attention block + 6 个 VLM cross-attention block”交错的 DiT 动作专家通过 flow matching 和显式 Euler 积分并行生成连续 action chunk 的双流 VLA；机器人状态不进入 VLM，而被编码为动作专家的单独 state token。

## 2. 重点文件和完整调用链

### 2.1 重点文件

| 领域 | 文件 | 关键类/函数 |
|---|---|---|
| 当前训练入口 | `train_vla.py:20-474` | `parse_option`, `train`, checkpoint/optimizer/scheduler |
| 当前实验启动 | `scripts/run_libero_wo_ecot_pt.sh:54-146` | 4 卡、action-only、H=10、全量微调 |
| 数据注册 | `dataset2feature.yaml:31-35` | `libero_wo_ecot_pt` |
| 数据与 batch | `utils/load_training_dataset.py:20-607` | `StreamingLeRobotSampleDataset`, prompt、processor、collator |
| LeRobot 取样 | `lerobot/lerobot/common/datasets/lerobot_dataset.py:754-954` | 时间索引、边界 padding、`add_ECoT` |
| 总模型 | `model/reasoning_vla_model.py:12-178` | `ZR0Model` |
| VLM 包装 | `model/qwen_vl_backbone.py:11-192` | `QwenVLBackbone`、条件 mask |
| 动作专家 | `model/flow_matching_action_head.py:139-550` | state/action encoder、flow loss、Euler 采样 |
| DiT | `model/cross_attention_dit.py:30-337` | timestep AdaLN、交错 self/cross attention |
| 归一化 | `utils/normalization.py:48-109` | quantile/min-max norm 与 denorm |
| 推理策略 | `policies/reasoning_vla_policy.py:19-138` | checkpoint、观测预处理、动作裁切/反归一化 |
| 历史帧 | `utils/obs_buffer.py:12-58` | `ObservationBuffer` |
| 服务端 | `server.py:9-57` | seed、policy、WebSocket server |
| LIBERO 客户端 | `evaluation/libero_eval/run_libero_eval.py:42-201` | 图像/状态、replan、环境执行 |
| Qwen 实际实现 | `.../site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:59-1366` | vision、DeepStack、mRoPE、causal decoder、CE |
| 生效 VLM 配置 | `outputs/.../step-2000/config.json:1-65` | 28 层文本塔、24 层视觉塔 |
| 生效动作配置 | `outputs/.../step-2000/action_expert_config.json:62-90` | H=10、D=2048、9 层 DiT |
| 数据元数据 | `/opt/data/private/lq/datasets/HuggingFaceVLA/libero/meta/info.json:1-97` | 2 相机、state=8、action=7、256x256 |

### 2.2 完整训练调用链

```text
scripts/run_libero_wo_ecot_pt.sh
  -> accelerate launch train_vla.py
  -> train_vla.train
  -> build_concat_streaming_dataset
     -> LeRobotDataset
     -> StreamingLeRobotSampleDataset.__getitem__
        -> deterministic_test_time_n_action_steps
        -> LeRobotDataset.getitem_with_delta_timestamps
           -> _get_query_indices / _query_hf_dataset / add_ECoT
        -> prepare_action_expert_inputs_cpu
        -> prepare_qwen_vl_inputs_cpu
           -> FAST tokenizer（仅生成 VLM 文本监督）
           -> tokenize_vision_language_inputs
              -> apply_chat_template
              -> qwen_vl_utils.process_vision_info
              -> Qwen3VLProcessor
  -> DataLoader(custom_collate_fn)
  -> ZR0Model.__init__
     -> QwenVLBackbone(Qwen3VLForConditionalGeneration)
     -> FlowmatchingActionHead(DiT + encoders + decoder)
  -> ZR0Model.forward
     -> QwenVLBackbone.forward
     -> FlowmatchingActionHead.forward
     -> flow-matching masked MSE
  -> Accelerator/DeepSpeed backward
  -> AdamW.step -> cosine scheduler.step
  -> ZR0Model.save_pretrained + DeepSpeed optimizer checkpoint
```

### 2.3 完整推理调用链

```text
server.py::deploy
  -> ZR0Policy.__init__
     -> ZR0Model.from_pretrained
        -> Qwen VLM + processor + action_expert.safetensors
     -> bf16 -> eval -> torch.compile
  -> WebsocketPolicyServer
  -> ZR0Policy.infer(observation)
     -> ObservationBuffer
     -> prepare_action_expert_inputs_cpu（state norm + infer mask）
     -> prepare_qwen_vl_inputs_cpu（images + task）
     -> custom_collate_fn -> CUDA
     -> direct: ZR0Model.get_action_direct
        -> VLM 一次 -> 动作专家 N 次 Euler 更新
     -> two-stage: ZR0Model.get_action_subtask
        -> VLM.generate 子任务
        -> prompt+subtask 完整 VLM forward
        -> 动作专家 N 次 Euler 更新
     -> [:n_action_steps, :action_dim]
     -> min_max_denorm
     -> WebSocket 返回 actions
  -> LIBERO/RoboCasa/RoboTwin client
     -> action queue -> env.step/take_action -> 新观测 -> 再规划
```

## 3. 整体模型架构图

```text
两视角/历史 RGB                               语言任务
       |                                        |
       v                                        v
Qwen3-VL image processor                  chat template/tokenizer
pixel patches [sum(P),1536]               input_ids [B,L]
       |                                        |
       v                                        |
Qwen3-VL Vision Transformer (24 x D=1024)       |
  final merger + DeepStack mergers              |
       |  每图 49 x D=2048                       |
       +--------------> image placeholder 替换 <-+
                              |
                              v
             Qwen3-VL causal text decoder (28 x D=2048)
               + mRoPE + causal/padding mask + DeepStack
                              |
                  last hidden [B,L,2048]
                              |
               prompt/subtask bool condition mask
                              |
                              v (K,V，cross-attention)
state [B,1,64] -> state MLP -> [B,1,2048]                 t bucket
                                                         |  |
noise/noisy action [B,H,64] -> action+t encoder -> [B,H,2048]
                              + learned action-position embedding
                              |
                              v
                  concat [state; H actions] [B,H+1,2048]
                              |
             9-layer DiT: SA,CA,CA,CA,SA,CA,CA,CA,SA
             SA = state/action 双向 self-attention
             CA = state/action query -> VLM prefix K/V
                              |
                   [B,H+1,1024] -> MLP head
                              |
                    velocity [B,H,64]
                              |
             train: masked flow-MSE
             infer: N 步显式 Euler -> normalized action chunk
                              |
             slice real dims -> quantile denorm -> env action
```

### 3.1 模块与参数量

下表以**当前 step-2000 checkpoint**为主；该 checkpoint 由原始 Qwen3-VL-2B 和随机动作专家开始训练。参数量由 safetensors shape 逐项求和，并与训练日志中的打印值一致。

| 模块 | 实现类 | 输入 | 输出 | 层数 | 隐藏维度 | 参数量 | 当前实验是否训练 |
|---|---|---|---|---:|---:|---:|---|
| VLM 文本/LM 部分 | `Qwen3VLForConditionalGeneration` / `Qwen3VLTextModel` | 文本 ID + 已替换视觉 embedding | final hidden / logits | 28 | 2048；FFN 6144 | 1,720,574,976 | 是 |
| 视觉编码器及 merger | `Qwen3VLVisionModel` | `[sum(P),1536]` patch rows | 每图 49 个 2048-d token | 24 | 1024；FFN 4096 | 406,957,056 | 是 |
| 状态编码器 | `CategorySpecificMLP` | `[B,1,64]` | `[B,1,2048]` | 2 Linear | 512 -> 2048 | 1,083,904 | 是 |
| 动作/时间编码器 | `MultiEmbodimentActionEncoder` | `[B,H,64]`, `[B]` | `[B,H,2048]` | 3 Linear | 2048 | 12,720,128 | 是 |
| 动作位置 embedding | `nn.Embedding` | action position `[H]` | `[H,2048]` | 1 table | 2048 | 524,288 | 是 |
| DiT | `DiT` | state/action token + VLM K/V + t | `[B,H+1,1024]` | 9 | 2048；FFN 8192 | 543,898,624 | 是 |
| 动作输出 head | `CategorySpecificMLP` | `[B,H+1,1024]` | `[B,H+1,64]` | 2 Linear | 512 | 557,632 | 是 |
| 动作专家合计 | `FlowmatchingActionHead` | 上述条件 | flow velocity/action | 9 个主 block | 2048 | 558,784,576 | 是 |
| 当前模型合计 | `ZR0Model` | 多模态观测 | action chunk | - | - | **2,686,316,608** | **2,686,316,608（100%）** |

证据：`train_vla.py:229-246` 统计参数；实际日志打印 VLM 2,127,532,032、DiT 543,898,624、动作专家 558,784,576、总计 2,686,316,608、可训练比例 100%。当前启动同时传入 `--tune_vlm` 和 `--tune_action_expert`（`scripts/run_libero_wo_ecot_pt.sh:92-94`）。

官方 `/opt/data/private/lq/models/ZR-0-libero` 的 VLM 因多 2050 个词表项而为 2,131,730,432，总模型为 **2,690,515,008**。`ZR0Policy` 用 `ZR0Model.from_pretrained` 的默认 `tune_vlm=False, tune_action_expert=False` 加载，所以推理进程中的可训练参数为 0（`model/reasoning_vla_model.py:166-177`）。

### 3.2 预训练、新增与连接模块

- **[代码明确]** Qwen3-VL 文本塔、视觉塔、内置 vision patch merger、DeepStack merger 均来自 `vlm_name_or_path` 的预训练权重（`model/qwen_vl_backbone.py:28-31`）。
- **[配置决定]** 当前 fresh run 未提供 `--action_expert_name_or_path`，state encoder、action encoder、DiT、action decoder、position embedding 全部随机初始化（`train_vla.py:188-204`）；step-2000 已训练这些参数。
- **[代码明确]** 从完整 ZR checkpoint 微调/推理时，动作专家从 `action_expert.safetensors` 全量加载（`model/reasoning_vla_model.py:34-38,166-174`）。
- **[代码明确]** 没有 VLM-to-DiT projector：代码要求两边都是 2048，projector 被注释（`model/flow_matching_action_head.py:343-348`）。
- **[代码明确]** 默认关闭 Difference Query 时没有 Resampler、Query Token 或独立聚合 token；可选 Difference Query 的当前实现与数据流见第 18 节。视觉塔有 Qwen 自带 2x2 patch merger；动作位置 embedding、state/action MLP 是新增连接模块。
- **[配置决定]** LoRA 是可选 adapter，当前 `use_lora=False`；启用时目标模块由 CLI 指定，默认 `gate_proj,up_proj,down_proj`（`train_vla.py:69-75`; `model/qwen_vl_backbone.py:41-51`）。

## 4. VLM 输入和 token 排列

### 4.1 VLM 接收与不接收的输入

| 输入项 | 原始格式 | 预处理方式 | 编码方式 | Token 数量 | 输入位置 | 训练/推理 |
|---|---|---|---|---:|---|---|
| 全局语言指令 | Python `str`，`data["task"]` | `.strip()`，包入 `<TASK> ... <\TASK>\n` | Qwen tokenizer | 随文本变化 | 所有图像之后、assistant marker 之前 | 都使用 |
| 当前图像 | 每相机 RGB，训练元数据 256x256x3 | PIL RGB -> 224x224 -> normalize/patchify | 24 层 vision tower + merger | **49/图** | 相机名称文本之后 | 都使用 |
| 历史图像 | `[window,C,H,W]` | 与当前图相同；时间优先、相机次优先 | 同上 | `49 * 相机数 * 历史帧数` | 当前帧之前 | `window_size>1` 时都使用；当前配置为 1 |
| 机器人状态 | 当前 LIBERO `[8] float32` | quantile norm、右补零到 64 | state MLP | **不进入 VLM**；动作专家 1 token | 动作专家序列首位 | 都使用 |
| Difference Query / query | 默认无；开启为 `[Nq,2048]` FP32 learnable 参数 | 插入 VLM embedding 序列 | `[B,Nq,2048]` | 默认 0；实验 32 | context 后、target 前 | 开启时训练/直接动作推理 |
| Chat/视觉特殊 token | `<|im_start|>`, role, `<|vision_start|>`, image placeholders, `<|vision_end|>`, `<|im_end|>` | chat template/processor 插入 | token embedding/视觉特征替换 | 随图片数 | 包围对应内容 | 都使用 |
| `<robot_action_N>` | FAST 量化后的动作码字符串 | 仅训练 assistant 输出中构造 | 官方 ZR tokenizer 中每码 1 special token；当前 step-2000 中会拆成普通子词 | 随 FAST 输出 | assistant marker 之后 | 仅训练文本目标；不作为连续动作专家 token |
| `<SUB_TASK>...</SUB_TASK>` | ECoT 的首个 To-do Action 或 `done` | 10% ECoT 样本选择；two-stage 推理生成 | 普通 VLM token/special token | 随文本变化 | assistant marker 之后 | 仅 ECoT two-stage 训练/推理 |
| ECoT JSON | `embodied_cot` JSON | direct 训练加入离散动作字段 | VLM assistant 文本 | 可变 | assistant marker 之后 | direct 训练有；direct 推理跳过 |
| flow timestep/noise | scalar / `[B,H,64]` | 动作专家内部采样 | sinusoid + timestep MLP | 不进入 VLM | 仅动作专家 | 训练/推理都用 |

`prepare_qwen_vl_inputs_cpu` 的真实顺序见 `utils/load_training_dataset.py:176-225`。当前 `window_size=1`、两相机时，展开为：

```text
<|im_start|> user \n
  "observation.images.image"
  <|vision_start|> 49 x <|image_pad|> <|vision_end|>
  "observation.images.image2"（官方评估 entry 则是 wrist_image）
  <|vision_start|> 49 x <|image_pad|> <|vision_end|>
  "<TASK> {instruction} <\TASK>\n"
<|im_end|> \n
<|im_start|> assistant \n
[训练 direct: ECoT JSON + Discrete Action Tokens]
[训练/two-stage: <SUB_TASK>...</SUB_TASK>]
```

没有显式 BOS 出现在实测 chat template 开头；开头是 `<|im_start|>`。`<\TASK>` 是实际代码的反斜杠拼写，不是标准 `</TASK>`（`utils/load_training_dataset.py:201`）。

### 4.2 实际张量 shape、dtype、device

以目标 LIBERO 数据的一条真实 raw observation、两张 224x224 模型输入做只读 processor 实测：

```text
input_ids:       [1, 143] int64, CPU -> CUDA
attention_mask:  [1, 143] int64, CPU -> CUDA, 全 1
pixel_values:    [392, 1536] float32, CPU -> CUDA；vision patch embed 内转 bf16
image_grid_thw:  [2, 3] int64 = [[1,14,14], [1,14,14]]
inputs_embeds:   [1, 143, 2048] bf16, CUDA（Qwen 内部构造，不由 wrapper 直接传入）
position_ids:    [3, 1, 143] int64, CUDA（Qwen 内部 mRoPE）
last_hidden:     [1, 143, 2048] bf16, CUDA
```

143 只是一条具体 task 的实测长度，不是常量。一般 `L = 文本/chat token 数 + 49 * 图像数`。训练先 pad/truncate 到 1200，再由 collator 截到 batch 最大有效长度加 1，因此训练 batch 为 `[B,L_batch]`，通常保留一列右 padding（`utils/load_training_dataset.py:118-160,587-593`）。

## 5. 相机输入和图像预处理

### 5.1 当前 LIBERO 数据

- **[配置决定]** 当前训练数据有两个相机：`observation.images.image`、`observation.images.image2`，均为 256x256 RGB（数据 `info.json:17-42`）。
- **[代码明确]** 顺序来自 `info["features"]` 的插入顺序，`LeRobotDatasetMetadata.camera_keys` 按 mapping 遍历返回（`lerobot_dataset.py:218-231`）。当前顺序是 image -> image2。
- **[代码明确]** `window_size>1` 时同时使用历史帧和当前帧；展开顺序是“时间外层、相机内层”，最后一个时间点命名为当前相机，其余命名 `historical-observation-{i}.{camera_key}`（`utils/load_training_dataset.py:178-189`）。当前配置 `window_size=1`，没有历史图。
- **[代码明确]** 多视角/多时刻不在 channel 或 tensor time 维拼接，而是在 VLM 序列的 token 维依次插入。视觉塔内部还按每张图分段做 attention，图片之间在视觉塔中不互看；进入文本 decoder 后才按因果序列交互。
- **[代码明确]** 缺少任一 metadata 相机没有替代值或 mask；policy 对每个 key 执行 `obs.get` 后转图，缺失会在 `ToTensor`/shape 路径报错（`policies/reasoning_vla_policy.py:76-85`）。

### 5.2 训练与 LIBERO 推理预处理对比

| 项目 | 训练（当前数据） | LIBERO 推理 | 是否一致 | 代码位置 |
|---|---|---|---|---|
| 原始环境/数据尺寸 | 256x256 | simulator render 256x256 | 是 | 数据 `info.json:17-42`; `run_libero_eval.py:31-32` |
| 客户端预 resize | 无 | 256 -> 448，保持比例并零 pad，PIL BILINEAR | **否** | `run_libero_eval.py:49,136-145`; `utils/image_tools.py:19-60` |
| 朝向 | 数据中已存的方向，loader 不旋转 | 两个视角均旋转 180 度 | 依赖数据采集；代码称为匹配训练 | `run_libero_eval.py:137-139` |
| server/Qwen resize | PIL RGB -> 224x224 | 448 PIL RGB -> 224x224 | 目标一致、来源不同 | `load_training_dataset.py:172-200` |
| 第二次插值 | `qwen_vl_utils` 的 `PIL.Image.resize` 未传 resample，PIL 默认 NEAREST | 同样 NEAREST | 算法一致 | `qwen_vl_utils/vision_process.py:93-140` |
| crop | 无 | 无 | 是 | 同上 |
| pad | 无 | 客户端可能零 pad；方形 LIBERO 实际不需 pad | 条件一致 | `utils/image_tools.py:52-60` |
| processor resize | `do_resize=False` | `do_resize=False` | 是 | `load_training_dataset.py:120-140` |
| 颜色 | PIL `convert("RGB")` | HWC numpy -> `ToTensor` -> PIL -> RGB | 是（RGB） | `vision_process.py:85-120`; policy `:76-80` |
| rescale | `/255` | `/255` | 是 | checkpoint `preprocessor_config.json:8-32` |
| normalize | mean `[.5,.5,.5]`, std `[.5,.5,.5]` | 相同 checkpoint processor | 是 | 同上 |
| augmentation | loader 未传 `image_transforms`，无增强 | 无增强 | 是 | `build_concat_streaming_dataset:530-545` |
| 模型实际尺寸 | **224x224** | **224x224**，不是 448 | 是 | prompt image fields `:173-200`; processor 实测 |

因此“推理图像 448”只是客户端中间传输尺寸；视觉塔实际收到的网格仍是 224/16=14。推理相较训练多了一次 256->448 BILINEAR，再做 448->224 NEAREST，像素采样链并不完全一致。

## 6. 视觉编码器结构

| 属性 | 实际值 | 证据 |
|---|---:|---|
| 模型 | Qwen3-VL 内置 `Qwen3VLVisionModel` | checkpoint `config.json:42-61` |
| block 数 | 24 | `vision_config.depth=24` |
| hidden / FFN | 1024 / 4096 | `config.json:51-54` |
| heads / head dim | 16 / 64 | `config.json:56`; 1024/16 |
| patch | Conv3d kernel=stride `(2,16,16)` | Transformers source `:59-75` |
| temporal handling | 静态图复制最后帧至 2，再形成 `grid_t=1` | image processor source `:232-260` |
| 单图 patch rows | 14x14=196，row width `3*2*16*16=1536` | processor 实测和 source |
| 视觉 attention | 每图/帧段内部双向；不同图分段隔离 | Transformers source `:168-248,727-744` |
| CLS | 无 | patch 序列直接送 blocks/merger |
| final 特征 | 第 24 层输出，经 2x2 merger -> 49x2048/图 | source `:93-106,751-753` |
| 多层特征 | vision block 5、11、17 的 DeepStack merger 特征 | config `deepstack_visual_indexes`; source `:590-599,745-749` |
| 注入方式 | final 特征替换 image placeholders；3 组 DeepStack 特征加到文本 decoder 第 0、1、2 层之后的视觉位置 | source `:861-867,1137-1175` |
| 参数量 | 406,957,056 | 当前/官方 checkpoint safetensors |
| 是否冻结 | 当前 step-2000 训练：否；policy 推理：冻结/eval；由 `tune_vlm` 决定 | `qwen_vl_backbone.py:53-65` |

这不是“取某个单一视觉层输出”：文本 decoder 的初始视觉 token 来自最后层 merger，同时前三个文本层分别接受视觉层 5/11/17 的 DeepStack 增量。

## 7. Attention Mask 与可见性矩阵

这里必须区分三种 attention：视觉塔 attention、VLM decoder attention、动作专家 attention。它们不是一张共享 mask。

### 7.1 VLM decoder mask

- **[代码明确]** Qwen 文本塔调用 `create_causal_mask`，是 decoder-only 因果注意力，不是 prefix 双向 attention（Transformers source `:834-841`）。
- **[代码明确]** 外部 `attention_mask` 是 `[B,L] int64`，1=有效、0=padding。逻辑上 token 只能看有效的自身及更早位置。
- **[配置决定]** 当前环境有 FlashAttention 2 时 wrapper 选 `flash_attention_2`，否则 eager（`qwen_vl_backbone.py:18-31`）。FA2 通常保留 2D padding 信息并使用 causal 标志；eager 等价为 `[B,1,L,L]` additive mask，允许值 0、阻断值为 dtype 最小值。逻辑可见性相同。
- **[代码明确]** mRoPE `position_ids` 由 Qwen 内部按 temporal/height/width 构造，shape `[3,B,L]`；padding 位置被排除（Transformers source `:916-1033,1177-1229`）。

下表描述 `use_difference_query=False` 的默认基线；矩阵中的“按序”表示只有 key 位于 query 左侧或自身时可见。机器人状态和连续动作 token 不在 VLM 序列；开启 Query 后的矩阵见第 18 节。

| VLM Query \ Key | 图像 | 本体状态 | 文本 | Difference Query | VLM 离散动作 token |
|---|---:|---:|---:|---:|---:|
| 图像位置 | 按序 | 不存在 | 仅更早文本 | 不存在 | 通常不可见（动作输出在后） |
| 本体状态 | 不存在 | 不存在 | 不存在 | 不存在 | 不存在 |
| 文本 | 仅更早图像 | 不存在 | 按序 | 不存在 | 仅当离散动作在其更早位置 |
| Difference Query | 不存在 | 不存在 | 不存在 | 不存在 | 不存在 |
| VLM 离散动作 token | 可见更早全部图像 | 不存在 | 可见更早文本 | 不存在 | 按序 |

视觉塔内部是另一张 block-diagonal 双向 mask：每张图自己的 196 个 patch 互相可见，不同图片 patch 互不可见；这不改变它们进入文本 decoder 后的因果关系。

### 7.2 动作专家读取 VLM 的 cross-attention mask

`QwenVLBackbone` 从 VLM final hidden 产生 `[B,L] bool` mask（`qwen_vl_backbone.py:92-153,172-190`）：

- direct：True 到并包含 `<|im_start|>assistant\n` 的最后一个 token；assistant 的 ECoT/离散动作输出全部 False。
- subtask：True 到并包含 `</SUB_TASK>`（ID 153718）；其后 False。
- 未找到硬编码 marker 时：该样本全部 True，包括可能的 padding/意外 assistant 输出。
- batch 内每个样本独立找截止位置，所以可见长度可不同。

该 bool mask 经 diffusers `AttnProcessor2_0` 变为有效 `[B,32,1,L]` 并在所有 `H+1` query 上广播；`True=可见`、`False=不可见`。

### 7.3 动作专家内部矩阵

实际 DiT 9 层的类型依次为：

```text
layer 0 SA -> 1 CA -> 2 CA -> 3 CA -> 4 SA -> 5 CA -> 6 CA -> 7 CA -> 8 SA
```

SA 层中 `causal_mask_in_self_attn=false`，state 与所有 action token 双向互见；CA 层中 state/action 都作为 query，读取被上述 mask 选中的 VLM K/V（`cross_attention_dit.py:294-328`）。

| Action-expert Query \ Key | 图像 | 本体状态 token | 文本 | Difference Query | 连续动作 token |
|---|---:|---:|---:|---:|---:|
| VLM 图像/文本 token | 不适用；VLM 已先算完 | 不可见 | 仅 VLM 自身因果关系 | 不存在 | 不可见 |
| 本体状态 token | CA 可见选中图像 | SA 可见 | CA 可见选中文本/子任务 | 不存在 | SA 全部可见 |
| 连续动作 token | CA 可见选中图像 | SA 可见 | CA 可见选中文本/子任务 | 不存在 | SA 全部双向可见 |
| Difference Query | 不存在 | 不存在 | 不存在 | 不存在 | 不存在 |

训练和推理使用同一 mask 函数。direct 训练虽包含 assistant 输出监督，动作专家仍只读 prompt；direct 推理恰好只构造到 assistant generation marker，因而条件前缀一致。two-stage 则显式把生成/标注的 subtask 纳入条件。

## 8. VLM 和动作专家的交互方式

### 8.1 关系结论

- 动作专家是独立 DiT，不是 VLM 的若干共享层，也不是读取一个聚合 token 的 MLP。
- 两者是**先串行、后在动作专家每个 CA 层交互**：VLM 完整 forward 一次，动作专家的第 1/2/3/5/6/7 层读取 VLM **final hidden state 的被选前缀**。
- 传递方式是 cross-attention；VLM hidden 作 K/V，state/action 作 Q。没有把 VLM token 与动作 token 拼成一套联合 self-attention。
- direct 读取全部 prompt（图像、相机标签、task、chat specials、assistant marker），不读取 assistant ECoT 或 VLM 离散动作 token。
- 默认 subtask 路径读取 prompt + 生成/监督 subtask；subtask 只是 VLM 普通 token，不是 Query Token。Difference Query 模式明确禁止 autoregressive `.generate()`。
- 两边 hidden 都是 2048，因此没有 projector。若不相等，代码直接 assert，不能自动投影。
- 动作信息不会写回 VLM；VLM token 永远看不到 state/连续动作 token。
- 不共享 attention 层、QKV、norm 或 KV cache。VLM 用 mRoPE；DiT action position 用 learned embedding，DiT block 自身配置无 positional embedding。
- 默认 `detach_vlm_outputs_for_action_expert=False`，flow loss 会经 cross-attention K/V 回传到 VLM；设为 True 时在 `ZR0Model.forward` 对 VLM hidden `.detach()`（`reasoning_vla_model.py:125-129`）。
- 默认关闭 Query 时仅有 direct prompt 与 prompt+subtask 两种截止 mask；开启后动作专家严格只读取 `[B,Nq,2048]` Query hidden。

### 8.2 逐层伪代码

```python
# VLM 一次
vl = qwen(input_ids, images, attention_mask,
          output_hidden_states=True, use_cache=not training)
K_V = vl.hidden_states[-1]                 # [B,L,2048]
cond_mask = prompt_mask or subtask_mask     # [B,L] bool

# 动作专家
s = state_mlp(norm_state_64)               # [B,1,2048]
a = action_time_encoder(x_t_64, t_bucket)  # [B,H,2048]
a = a + learned_action_pos[:H]
x = concat(s, a, dim=1)                    # [B,H+1,2048]

for i in range(9):
    t_cond = timestep_mlp(t_bucket)         # [B,2048]
    x = AdaLN(x, t_cond)
    if i in {0, 4, 8}:
        x = x + SelfAttention(x)            # 双向 state/action
    else:
        x = x + CrossAttention(Q=x, K=K_V, V=K_V,
                               key_mask=cond_mask)
    x = x + FFN(LayerNorm(x))

x = time_conditioned_output_norm(x)         # [B,H+1,2048]
x = output_proj(x)                          # [B,H+1,1024]
velocity64 = action_decoder(x)[:, -H:]       # [B,H,64]
```

## 9. 动作专家详细结构

| 属性 | 实际值 | 来源配置 | 代码位置 |
|---|---:|---|---|
| 类型 | Flow-matching DiT action head | checkpoint | `flow_matching_action_head.py:302-341` |
| 主 block 数 | 9 | `diffusion_transformer_cfg.num_layers` | config `:74-90` |
| self / cross block 数 | 3 / 6 | `interleave_self_attention=true` | `cross_attention_dit.py:296-327` |
| hidden size | 2048 | 32x64 | config `:63-76` |
| FFN intermediate | 8192 | diffusers 默认 `4*dim`，checkpoint weight `[8192,2048]` | `cross_attention_dit.py:143-150` |
| heads | 32 | checkpoint | config `:75` |
| head dim | 64 | checkpoint | config `:76` |
| attention Q/K/V | 各 2048 -> 2048，带 bias | `attention_bias=true` | block `:130-139` |
| activation | GELU approximate(tanh) | checkpoint | config `:81` |
| norm1 | timestep-conditioned AdaLayerNorm，内部非 affine LN | `ada_norm` | `cross_attention_dit.py:48-74,124-128` |
| norm2 | LayerNorm，非 affine，eps 1e-5 | checkpoint | block `:141-150` |
| output norm | 非 affine LN eps 1e-6 + timestep shift/scale | 代码固定 | DiT `:265-268,330-337` |
| dropout | 0.2；attention output/FFN 路径有 dropout，eval 关闭 | checkpoint | config `:79,88`; block `:130-153,191-192` |
| DiT position encoding | `None` | checkpoint | config `:86-87` |
| action chunk position | learned `[256,2048]`，仅加到 action token | `add_pos_embed=true` | action head `:350-352,441-445` |
| timestep bucket | 1000 | checkpoint | config `:73` |
| action encoder time | 2048-d sinusoid，与 action projection concat | 代码固定 | action encoder `:167-213` |
| DiT time encoder | Timesteps 256 -> MLP 2048 -> 2048 | 代码固定 | `cross_attention_dit.py:30-45` |
| state encoder | 64 -> 512 ReLU -> 2048 | checkpoint | action head `:325-330` |
| action encoder | 64 -> 2048；concat t 后 4096 -> 2048 Swish -> 2048 | checkpoint | action head `:331-335` |
| DiT output | 2048 -> 1024 | checkpoint | DiT `:267-268` |
| action head | 1024 -> 512 ReLU -> 64 | checkpoint | action head `:336-341` |
| cross-attention | 有，6 层，VLM final hidden 为 K/V | 代码 | DiT `:319-327` |
| causal self-attention | 否 | checkpoint | config `:89-90` |
| KV cache | 动作专家无 | 代码 | 每个 Euler step 重算 9 层 |
| 训练目标 | Flow Matching velocity | 代码 | action head `:422-479` |

每个 DiT block 58,742,784 参数；9 个共 528,685,056。整个 DiT（含 timestep encoder 和输出投影）543,898,624。

## 10. 动作专家输入输出

### 10.1 输入张量

当前 LIBERO 配置中 `B` 为 batch，`H=10`，原始 state dim=8、action dim=7，模型 pad dim=64。

| 张量 | 含义 | Shape | Dtype/device（模型内） | 来源 | 进入模块 |
|---|---|---|---|---|---|
| `backbone_embeddings` | VLM 所有 token final hidden | `[B,L,2048]` | bf16/CUDA | Qwen hidden `[-1]` | 6 个 CA block 的 K/V |
| `action_expert_cross_attn_mask` | VLM 条件可见位 | `[B,L]` | bool/CUDA | marker 截止函数 | CA key mask |
| `observation.state` | 归一化并 pad 的状态 | `[B,1,64]` | bf16/CUDA | dataset/policy | state encoder |
| `state_mask` | 前 8 维为 1 | `[B,1,64]` | int64/CUDA | pad 函数 | **被传入但从未使用** |
| `action` | 归一化 GT action | `[B,10,64]` | bf16/CUDA | 训练 dataset | flow path |
| `action_mask` | 前 7 维为 1 | `[B,10,64]` | int64 -> bf16 | pad 函数 | noise/trajectory/output/loss mask |
| `infer_action_mask` | 前 7 维为 1 | `[B,10,64]` | int64 -> bf16 | policy | 初始噪声/每步输出 mask |
| `noise` | 标准高斯 | `[B,10,64]` | bf16/CUDA | `torch.randn` | flow interpolation |
| `t` | 每样本一个连续 flow time | `[B]`，广播 `[B,1,1]` | bf16/CUDA | Beta 采样变换 | 轨迹构造 |
| `t_discretized` | timestep bucket | `[B]` | int64/CUDA | `long(t*1000)` | action encoder + DiT timestep encoder |
| `cat_ids` | embodiment 类别 | `[B]`，全 0 | int64 | 动作专家内部 | category-specific Linear |

尽管类名支持 multi-embodiment，`num_categories/num_embodiments` 当前硬编码为 1 且 `cat_ids` 全 0（`flow_matching_action_head.py:325-340,416-420`）。

### 10.2 输出及物理语义

```text
动作专家训练输入 shape：noisy action [B,10,64] + state [B,1,64] + VLM [B,L,2048]
动作专家原始输出 shape：predicted velocity [B,10,64]
推理积分后原始 action shape：[B,10,64]（normalized）
反归一化后输出 shape：[n_action_steps,7]
发送给 LIBERO 环境的单步 action shape：[7]
```

- 输出是连续值，不是离散 action token；所有 H 个动作并行表示一个 chunk。
- current checkpoint `action_horizon=10`。policy 按请求截取 `n_action_steps`，官方 LIBERO 配置每次执行 10 步后重规划（policy `:122-129`; LIBERO client `:152-179`）。
- 训练/推理均使用 q01/q99 quantile min-max；公式见第 12 节。反归一化后没有 clip（`normalization.py:99-109`）。
- action mask 只屏蔽 pad 到 64 的维度；没有为 gripper 单独离散化、阈值化或裁剪。
- **[无法确认]** 目标数据 `info.json` 只把 7 维命名为整体 `actions`，没有逐维名字；代码直接把它传给 `env.step`。因此不能仅凭代码断言 7 维一定是位置/速度/末端增量或 gripper 的具体编码。
- LIBERO 观测 state 的 8 维可以由评估代码确认：EEF position 3 + quaternion 转 axis-angle 3 + gripper qpos 2（`run_libero_eval.py:158-164`）。这不等于 action 的语义证明。

## 11. 训练数据流

当前实际实验配置由 `scripts/run_libero_wo_ecot_pt.sh:76-108` 决定：4 进程、每卡 16、global batch 64、8 epochs、H=10、window=1、action-only loss、VLM/动作专家都训练、无 LoRA、LR=2e-5。

1. **数据读取。** `LeRobotDataset` 按随机 subset index 读取当前帧；H=10 action 时间偏移为 `[0..9]/10 s`。每 epoch subset seed 为 `42+epoch`（`load_training_dataset.py:299-328`）。
2. **历史 stride。** 训练为每样本确定 1..H 的 `test_time_n_action_steps`；仅 window>1 时它控制历史帧间隔。当前 window=1，因此没有实际影响（`:229-235,312-329`）。
3. **边界处理。** 超出 episode 的未来 action index 被 clamp 到最后帧，同时产生 `action_is_pad`（`lerobot_dataset.py:754-770`）；下游没有使用该时间 padding 标记。
4. **状态/动作归一化。** q01/q99 -> 标称 `[-1,1]`，实际 clip 到 `[-15,15]`，然后维度补零到 64（`load_training_dataset.py:45-66`; `normalization.py:48-78`）。
5. **无 ECoT 数据。** `add_ECoT` 总会生成 JSON；当前 raw 数据无 cot/future_sub_tasks/bbox，因而 JSON 初始为空。`ecot_supported=False`，`sub_task_flag` 恒 0（`lerobot_dataset.py:921-954`; loader `:272-297,350-359`）。
6. **FAST 文本目标。** FAST 对未 pad 的 normalized action 量化，构造 `<robot_action_N>` 字符串，并加入 assistant JSON（loader `:204-223`）。这是 VLM 文本目标，不替代连续 flow 监督。
7. **图像/prompt。** 两图逐一 224 处理，task/chat template tokenization；训练 max length 1200、右 pad、prompt labels 置 -100（`:109-160,163-227`）。
8. **collate。** `input_ids/masks/actions/state` stack；所有样本的 `pixel_values` 和 `image_grid_thw` 沿第 0 维 concat；文本截到 batch 最大有效长度+1（`:568-595`）。
9. **VLM forward。** Qwen 计算 final hidden，也会因 labels 存在而计算 `vlm_loss`；当前 `loss_type=action` 最终不使用该 CE（`reasoning_vla_model.py:115-133`）。
10. **flow forward。** 每个样本采 t/noise，构造 `x_t` 和目标 velocity，经过 state/action encoders 与交错 DiT，计算 masked MSE。
11. **反向。** 当前没有 detach，action loss 经 6 个 cross-attention block 回传至 VLM 所有参与 prefix 表征的参数，包括视觉塔；assistant 输出 token 不在动作条件 mask 中。
12. **优化。** AdamW：lr 2e-5、betas `(0.9,0.95)`、eps `1e-6`、weight decay `0.01`（`train_vla.py:272-280`）。DeepSpeed ZeRO-2、bf16、grad accumulation 1、clip 1.0（`accelerate_config.yaml:3-17`）。
13. **scheduler/checkpoint。** cosine 至 0.1x LR；global 计划 34,184 steps，有效 warmup 2,734 global steps。当前 checkpoint 的 DeepSpeed `global_steps=2000`、`global_samples=128000`，scheduler 内部 step 8000；训练未达到 8 epochs 的计划终点。

## 12. Loss 设计与公式

### 12.1 全部实际 loss

| Loss | 监督目标 | 预测值 | Mask | 计算方式 | 权重 |
|---|---|---|---|---|---:|
| VLM next-token CE | assistant ECoT/subtask/离散动作文本及结束 token | Qwen logits `[B,L,V]` | prompt/pad label `-100` | HF causal LM shifted CE | combined 时 `vlm_loss_weight`，默认 1 |
| Flow matching | `a - eps` velocity | action expert `[B,H,64]` | action dim mask `M` | masked sum MSE / `M.sum()` | combined 时 `action_expert_loss_weight`；当前 action-only 为原始 1 |

没有 query/slot loss、auxiliary loss、classifier-free loss，也不是 diffusion noise-prediction loss。`training_progress` 和注释中的时间权重代码当前不参与 loss（action head `:464-479`）。

### 12.2 Flow matching 公式

对归一化 GT action `a`、有效维 mask `M`：

```text
u ~ Beta(alpha=1.5, beta=1.0)
t = (s-u)/s,  s=0.999
eps ~ N(0,I)
x_t = ((1-t) eps + t a) odot M
v*  = (a-eps) odot M
t_bucket = long(1000 t)
v_theta = ActionExpert(x_t, t_bucket, state, VLM_prefix)

L_action = sum ||(v_theta odot M) - v*||^2 / max(sum M, 1)
```

代码位置：`flow_matching_action_head.py:385-387,422-479`。loss 按整个 batch、H 和有效动作维的**标量元素平均**，不是先逐样本平均。64 维中的 pad 维不参与；episode 尾部重复的时间步仍参与，因为 `action_is_pad` 被忽略。

`s=0.999` 使极小概率样本的 t 略小于 0；代码没有 clamp。这是实际实现，不是标准地限制在严格 `[0,1]`。

### 12.3 VLM CE 与组合

对 labels 非 -100 的位置集合 `Y`：

```text
L_vlm = -(1/|Y|) sum_{i in Y} log p_theta(y_i | y_<i, images, prompt)

L_total = w_vlm L_vlm + w_action L_action
```

prompt 和 assistant marker 本身被 mask；assistant 输出及结尾 `<|im_end|>` 受监督（`load_training_dataset.py:86-100,150-160`）。HF 在模型内部完成 causal shift（Transformers Qwen source `:1315-1366`）。

`ZR0Model.forward` 支持：

- `vlm`：只返回 CE，梯度只训练 VLM。
- `action`：只返回 raw flow loss；传入的两个 loss weight **不会应用**。
- `vlm_and_action`：才使用两项权重（`reasoning_vla_model.py:122-147`）。

当前 step-2000 是 `action`，所以没有文本 CE 梯度；但因未 detach，flow loss 同时训练 VLM 和动作专家。VQA 样本的 action mask 全 0，flow loss为 0，只有启用 VLM loss 时才产生有效监督（loader `:433-447`）。

### 12.4 归一化公式

当前 `use_quantile=true`：

```text
m = q01, Mx = q99
a_norm = clip(2 * (a-m)/(Mx-m+1e-8) - 1, -15, 15)
a = ((a_norm+1)/2) * (Mx-m+1e-8) + m
```

注释称归一化到 `[-1,1]`，但实际 clip 边界是 `[-15,15]`；反归一化和送环境前均无范围 clip（`normalization.py:48-109`）。

## 13. 完整推理流程

### 13.1 direct-action

| 步骤 | Tensor/操作 | Shape（当前 LIBERO） |
|---:|---|---|
| 1 | client RGB 两视角、task、state、`n_action_steps=10` | images `[256,256,3]`；state `[8]` |
| 2 | client resize，server buffer，Qwen resize/normalize | 中间 `[448,448,3]`；模型每图 224x224 |
| 3 | processor | IDs `[1,L]`; pixels `[392,1536]`; grid `[2,3]` |
| 4 | state quantile norm + pad | `[1,1,64]`；infer mask `[1,10,64]` |
| 5 | VLM full forward 一次 | hidden `[1,L,2048]` bf16 |
| 6 | prompt condition mask | `[1,L]` bool |
| 7 | 初始 `x_0 ~ N(0,I) odot M` | `[1,10,64]` bf16 |
| 8 | 每个 Euler step 编 action+t、跑完整 9 层 DiT | velocity `[1,10,64]` |
| 9 | `x <- (x + dt*v) odot M` | `[1,10,64]` |
| 10 | 取前 10 步、前 7 维，q01/q99 denorm | `[10,7]` |
| 11 | client action queue 每步 `env.step` | 每次 `[7]` |
| 12 | queue 空后取新观测重规划 | 每 10 环境步一次 |

默认 `N=5`：`dt=0.2`，velocity 评估时间为 `t={0,.2,.4,.6,.8}`，bucket `{0,200,400,600,800}`；显式 Euler 五次后到达 t=1。代码没有 SDE、随机 Heun/Runge-Kutta、自适应 ODE solver 或 CFG（`flow_matching_action_head.py:502-550`）。

VLM 在一个 action chunk 内只算一次，供 5 个动作专家 step 复用；动作专家每步完整重算且无 KV cache。下一次 replan 会重新计算 VLM。wrapper 在 eval 给 Qwen `use_cache=True`，但 direct 路径不消费返回的 `past_key_values`，因此没有跨 replan cache（`qwen_vl_backbone.py:155-166`）。

### 13.2 two-stage

1. 使用 prompt 和图像调用 `Qwen.generate(do_sample=False, eos_token_id=153718)`，贪心生成 subtask。
2. 对 `prompt + generated subtask` 建全 1 attention mask，再做一次完整 VLM forward。
3. action expert mask 截到 `</SUB_TASK>`，然后执行与 direct 相同的 Euler flow。

生成阶段可使用 Qwen 自身 cache，但随后又完整 forward，且动作专家不复用生成 cache（`reasoning_vla_model.py:79-113`）。当前 step-2000 tokenizer 没有 `<SUB_TASK>` special token，见第 16 节风险；官方 ZR-0-libero 有 ID 153717/153718。

### 13.3 随机性

- 服务端固定 Python/NumPy/Torch/CUDA seed=42，并启用 cuDNN deterministic（`server.py:9-16,32-35`）。
- 每次 direct sampling 仍从高斯噪声开始；同一进程的连续请求会推进 RNG 状态，所以相同 observation 不保证每次返回相同 action。重启并保持完全相同调用序列时才可复现。
- subtask 文本 `do_sample=False`，其生成本身为贪心；action flow 仍随机。
- LIBERO client 独立 seed=7（`run_libero_eval.py:66-71`）。

## 14. 训练与推理差异

| 检查项 | 当前训练 | 官方 LIBERO 推理入口 | 结论/后果 |
|---|---|---|---|
| 相机数 | 2 | 2 | 数量一致 |
| 相机 key | `image`, `image2` | demo entry/client：`image`, `wrist_image` | **不一致**。若 server 用训练 entry，client 缺 `image2` 会报错；若用 demo entry，输入可跑但相机标签 token 分布改变 |
| 相机顺序 | metadata 插入顺序 | metadata 插入顺序 | 两者均主视角后腕视角，但 key 文本不同 |
| raw/中间尺寸 | 256 -> 224 | 256 -> 448 -> 224 | 模型尺寸一致，采样链不一致 |
| normalize | q01/q99；图像 mean/std .5 | 相同算法；取 server dataset entry stats | 代码匹配；两个 LIBERO stats 文件数值接近但非逐字节相同 |
| prompt | 两图标签 + `<TASK>...<\TASK>` | 同模板 | 模板一致，camera label 取决于 entry |
| state | 8 -> 64 | 8 -> 64 | 一致 |
| action dim | 7 -> 64 | 64 -> 前 7 -> denorm | 一致 |
| action horizon | 10 | checkpoint 10，replan 10 | 一致 |
| history | window 1 | window 1 | 一致 |
| expert mask | direct prompt 截止 | direct prompt 截止 | 一致 |
| VLM assistant 输出 | 训练构造离散动作 JSON，但 action-only 不用 CE | 不生成 | flow 条件一致，因为 expert mask 排除输出 |
| flow time | transformed Beta 随机连续 t | 固定均匀 Euler grid | 合理的 train/sample 差异 |
| VLM cache | train false | eval true但 direct 未复用 | 行为输出应一致，性能路径不同 |
| action denorm | GT norm 用 q01/q99 | pred 用同 entry q01/q99 逆变换 | 匹配；无 clip |
| checkpoint 新模块 | fresh run 中动作专家随机后训练 | `from_pretrained` 要求 action config+weights | step-2000 文件齐全，direct 可加载 |

当前训练 checkpoint 若要配现有 LIBERO client，不能同时做到“相机 key 文本完全匹配训练”和“client payload key 直接匹配”而不调整某一侧；这是明确的接口差异，不是视觉数量差异。

## 15. 关键配置汇总

### 15.1 当前 step-2000

| 配置 | 值 | 证据 |
|---|---:|---|
| VLM | `/opt/data/private/lq/models/Qwen3-VL-2B-Instruct` | launcher `:6,81` |
| VLM layers/hidden | 28 / 2048 | step config `:16-23` |
| VLM heads/KV heads/head dim | 16 / 8 / 128 | step config `:13,21-23` |
| VLM FFN/activation/norm | 6144 / SiLU / RMSNorm eps 1e-6 | step config `:14,18,24` |
| VLM vocab | 151936 weights；tokenizer length 151669 | step config `:37`; tokenizer 实测 |
| vision | 24 layers, D=1024, FFN=4096, 16 heads | step config `:42-61` |
| vision patch/merge | 16 / 2 | step config `:59-61` |
| action expert | D=2048, 9 layers, 32x64 heads | action config `:63-78` |
| state/action padded dim | 64 / 64 | action config `:67-68` |
| action horizon | 10 | action config `:69` |
| dataset | 273,465 frames, 1693 episodes, 40 tasks, 10 Hz | data info `:4-10` |
| train batch | 16/GPU x 4 = 64 global | launcher `:78,83` |
| loss | action flow only | launcher `:94` |
| trainable | VLM + action expert，100% | launcher `:92-93`; log |
| optimizer | AdamW 2e-5, wd .01, betas .9/.95 | launcher `:88`; train `:272-279` |
| distributed | ZeRO-2 bf16, grad accum 1, clip 1 | accelerate config `:3-17` |
| current progress | global step 2000 / 128,000 samples | DeepSpeed checkpoint state |

### 15.2 官方 ZR-0-libero 推理

| 配置 | 值 | 证据 |
|---|---:|---|
| 总参数 | 2,690,515,008 | safetensors shape 求和 |
| VLM vocab | 153986 weights；2050 个额外 action/subtask token | official config `:37`; `added_tokens.json` |
| action horizon/dim | 10 / 64（环境取前 7） | official action config `:68-70` |
| inference mode | direct_action | `simple_scripts/eval_libero.md:23-49` |
| denoise steps | 5 | server default/manifest |
| replan | 10 | client `Args.replan_steps` |
| model input | 每图 224，两个视角 | processor 与实测 |
| client intermediate | 448 | client `Args.resize_size` |

## 16. 代码中存在的疑点和运行风险

1. **当前 step-2000 不含动作/subtask special token。** 原始 Qwen checkpoint 没有运行 `add_qwen_special_tokens`；`<robot_action_N>` 被拆成普通子词。对当前 action-only/direct 条件不阻断，但不等价于官方的单 token 离散动作监督；two-stage 硬编码 eos 153718 不在该词表可生成范围，找不到截止 token时 mask 回退为全 True。
2. **当前训练与 LIBERO client 的第二相机 key 不同。** `image2` 与 `wrist_image` 的接口后果见第 14 节。
3. **server 默认 dataset entry 无效。** `server.py:20` 默认 `libero_v21`，`dataset2feature.yaml` 中实际是 `demo_data.libero_v21` 和 `libero_wo_ecot_pt`；不显式传参会 `KeyError`。
4. **任务结束 tag 拼写错误。** 实际是 `<\TASK>` 而不是 `</TASK>`（loader `:201`）。训练/推理都用同代码时一致，但与文档/预期格式不一致。
5. **mask marker 硬编码且失败时过度开放。** assistant 三元 ID 和 `</SUB_TASK>` ID 写死；找不到就让所有 token 可见，可能把 assistant GT、生成尾部或 padding 暴露给动作专家（`qwen_vl_backbone.py:102-122,137-151`）。
6. **`state_mask` 未使用。** 它被 prepare/传递，但 state encoder 直接吃完整 64 维；当前 pad 值为 0，所以不会读随机垃圾，但 mask 本身没有任何计算效果。
7. **episode 尾部 action padding 未用于 loss。** LeRobot clamp 到最后 action 并给 `action_is_pad`，VLA 预处理只创建维度 mask，重复尾帧仍被监督。
8. **归一化注释与代码不符。** 文档说 `[-1,1]`，实际允许到 `[-15,15]`；推理反归一化后无 clip，极端 flow 输出可超出数据范围。
9. **DiT 错误分支错误地 `raise` 字符串。** mask dtype 不为 bool 时 `raise("...")` 会触发 Python `TypeError`，而非期望的清晰异常（`cross_attention_dit.py:282-283`）。
10. **动作专家 gradient checkpointing 只是声明。** `DiT._supports_gradient_checkpointing=True`，但 forward 明写 `TODO: implement gradient checkpointing`（`:209,294`）；只有 VLM 的 checkpointing 被启用。
11. **历史 buffer 无服务端 reset 协议。** `ObservationBuffer.reset` 存在，但 WebSocket 只实现 infer，episode 切换不调用；`window_size>1` 会混入上一 episode。当前官方 window=1 不受影响。
12. **缺相机无 graceful fallback。** 没有零图、重复图或 camera mask；metadata 与 payload 必须完全匹配。
13. **`grounding_camera_keys` 无实际用途。** 参数从 metadata 一路传入，但 `prepare_qwen_vl_inputs_cpu` 不读取它。
14. **多 embodiment category 实际关闭。** encoder 名称虽为 category-specific，类别数硬编码 1，所有样本 `cat_ids=0`。
15. **文本长度和失败样本处理。** 训练超过 1200 token 会静默截断；dataset 单样本异常最多重试 30 次后返回 `None`，若整个 batch 都为空 collator 返回 `None`，训练 loop 没有 guard（loader `:331-374,568-575`）。
16. **环境版本敏感。** vendored LeRobot 声明 `datasets==3.6.0`；在本机 `zr0-eval` 的 datasets 4.8.5 上，只读构造 dataset 会在 `torch.stack(Column)` 失败。实际训练日志表明正确的训练环境已成功运行到 step 2000，因此这是环境复现风险，不是该 checkpoint 已失败的结论。
17. **当前 checkpoint 尚非计划完成态。** 启动计划 8 epochs/34,184 global steps，现有目录只有 step-2000/latest，且当前无训练进程；它是中间 checkpoint。
18. **README 是总体模型说明，不是当前实验配置。** README `:43` 称 joint CE+flow；当前实验明确 action-only。README “2.6B”是近似值，当前/官方精确值分别约 2.686B/2.691B。

## 17. 当前代码无法确认的信息

- LIBERO 7 个 action 维度逐维代表位置、速度、末端位姿、增量还是绝对量；元数据只给整体名称 `actions`。
- gripper action 的精确数值约定（连续、符号、开合方向）；代码没有单独处理或逐维名字。
- 当前训练数据生成时的 180 度旋转究竟在哪个离线步骤完成；评估注释说旋转是为匹配训练，但本仓库没有该数据的完整转换 provenance。
- 官方 ZR-0 预训练时每一阶段究竟冻结了哪些模块；checkpoint 不记录训练历史。只能确认当前脚本/当前日志的 requires-grad 状态。
- 当前 step-2000 后续为何停止，以及是否计划恢复；checkpoint 只确认停在 global step 2000。
- 不同硬件/backend 下 Qwen 内部物理 attention mask 的具体存储 shape；逻辑 causal/padding 语义可确认，FA2/eager 的物理表示不同。
- FAST 对每条 action chunk 最终产生的 token 数不是固定 HxD，取决于 FAST tokenizer 实现和数据；代码只把返回序列逐码映射。
- `torch.compile` 的实际图分段、吞吐与数值误差；代码启用且官方评估成功，但没有在模型定义中固定这些运行时细节。

## 总表

| 分析项 | 结论 | 代码或配置证据 |
|---|---|---|
| VLM | Qwen3-VL conditional generation，28 层 causal decoder，当前 step VLM 2,127,532,032 参数 | `qwen_vl_backbone.py:28-31`; step config `:8-37` |
| 视觉编码器 | Qwen3-VL 24 层 ViT，D=1024，final+3层 DeepStack，2x2 merger | step config `:42-61`; Transformers source `:564-753` |
| 相机视角数 | 当前 LIBERO 训练 2：image、image2；官方 eval 2：image、wrist_image | 两份 dataset `info.json` |
| 训练图像尺寸 | raw 256；模型实际 224；49 VLM视觉 token/图 | data info；loader `:172-200`; processor 实测 |
| 推理图像尺寸 | render 256 -> client 448 -> 模型实际 224 | LIBERO client `:49,136-145`; loader |
| VLM hidden size | 2048 | step config `:16` |
| 动作专家层数 | 9：3 SA + 6 CA | action config `:78-90`; DiT `:296-327` |
| 动作专家 hidden size | 2048，32 heads x 64；FFN 8192 | action config；checkpoint weights |
| 动作维度 | 模型 64，当前 LIBERO 有效 7 | action config `:67`; data info `:52-59` |
| Action horizon | 当前/官方 LIBERO checkpoint 均 10 | action config `:69` |
| 动作专家读取的 VLM 内容 | direct=完整 prompt 至 assistant marker；two-stage=prompt+subtask；均读 final hidden | `qwen_vl_backbone.py:92-190` |
| Attention mask 类型 | 视觉塔每图双向；VLM causal+padding；动作 SA 双向；动作 CA 使用 bool prefix mask | Qwen source；DiT source |
| 训练 loss | 当前仅 masked flow-velocity MSE；通用代码可加 VLM causal CE | launcher `:94`; `reasoning_vla_model.py:115-147` |
| 推理迭代步数 | 默认 5，显式 Euler，t=0,.2,.4,.6,.8 | `server.py:25`; action head `:511-548` |
| 参数量 | 当前 2,686,316,608；官方 ZR-0-libero 2,690,515,008 | 训练日志 + safetensors shape |
| 可训练参数 | 当前训练 100%；policy 推理 0 | 训练日志；`from_pretrained` 默认参数 |
| 当前 checkpoint | step 2000，128,000 global samples，中间态 | DeepSpeed state + checkpoint 目录 |

## 18. 可选 Difference Query（当前实现）

### 18.1 开关、参数和 backend

- `use_difference_query` 在 Python/CLI 加载接口中是真三态，默认 `None`；CLI 用 `--use_difference_query` / `--no-use_difference_query` 表达显式开/关。`num_difference_queries` 为 `Optional[int]`。没有 Query checkpoint 且未显式开启时保持原行为；只有显式开启才允许随机初始化，未指定 Nq 时取 32。
- Query checkpoint 会自动启用。显式关闭、显式 Nq 或 backend 与 checkpoint 冲突会直接报错。`enabled=false` config 是明确架构声明，不等于老 checkpoint：它阻止 CLI 开启、校验实际 VLM hidden size，旁边出现 Query 权重视为损坏；只有 config 和 weight 都不存在才允许随机初始化。Query 模式固定使用 SDPA；无 Query 且 backend 为 `None` 时仍按原逻辑自动选择 FlashAttention 2，否则回退 eager。`baseline_sdpa` 是无 Query 的 backend 控制组。
- learnable 参数 `difference_query.weight` 的精确 shape 为 `[Nq,2048]`。它先在隔离 CPU RNG context 中以 FP32 normal initialization 创建，不推进全局 RNG；forward 时按 token embedding 的 dtype/device 转换。Nq=8/32/64 分别新增 16,384/65,536/131,072 个参数。
- 当前 Nq=32 实验臂相对同一随机 Action Expert 基线只新增 65,536 参数。开关关闭时不创建 Query 参数、不扩 tokenizer、不增加 loss，也不改变原二维 Qwen mask 或动作专家输入。

### 18.2 序列与 mRoPE

设右 padding 前有效 prompt/context 为 C、learnable Query 为 Q、首个 `labels != -100` 起的 teacher-forced target 为 T、padding 为 P。逐样本重排为：

```text
训练:           [C, Q, T, P]
direct inference: [C, Q, P]
```

辅助 token ID 使用 Qwen 普通词表 ID 0（`!`），构造时断言它不是视觉、padding、EOS、BOS 或其他 control special token。辅助 ID 只用于 token embedding、视觉 placeholder 对齐和 `get_rope_index`；对应 embedding 随后被真实 learnable Query 覆盖。二维有效位 mask 排除 P，mRoPE 由扩展后的辅助 ID、实际 `image_grid_thw` 和该有效位 mask 计算，因此原视觉 placeholder、vision feature scatter、DeepStack 注入位置及视觉 position IDs 保持对齐。

### 18.3 VLM attention mask

传给 Qwen SDPA 的实际 mask 为 `[B,1,L+Nq,L+Nq] bool`，与 embedding 同 device，`True` 表示可见：

| Query \ Key | C | Q | T | P |
|---|---|---|---|---|
| C | causal | False | False | False |
| Q | True | 双向 True | False | False |
| T | False | True | causal | False |
| P | False | False | False | False |

因此改变 T 不会改变 Q，改变 C 会改变 Q；所有有效行都不能读 P，P 行也完全屏蔽。只要求 SDPA 在全屏蔽 P 行产生有限 attention 输出，不要求残差/MLP 后 P hidden 为零。

### 18.4 动作条件、loss 与限制

Qwen 最后一层 decoder hidden 仍经过现有 final RMSNorm。随后按逐样本 Query position gather：

```text
backbone_embeddings:                 [B,Nq,2048]
action_expert_cross_attn_mask:        [B,Nq] bool，全 True
```

`ZR0Model` 在训练和直接动作去噪边界检查 batch、Nq、hidden size、mask shape/dtype/device、全 True 和有限性。Flow Matching、DiT、state/action encoder 和 action decoder 未修改；没有独立 Query loss，action loss 通过动作专家 cross-attention 回传到 Query。Qwen 冻结时 Query 仍可训练。`detach_vlm_outputs_for_action_expert=True` 与 Query action-only 训练冲突并报错。

Query-on 的 `loss_type=action` 和 direct-action 保留 labels 供 Query 序列构建器识别 C/T，但调用 Qwen 原生 multimodal base forward，直接取得 final normalized hidden；不会执行 conditional LM head、分配 `[B,L,V]` 全词表 logits 或计算 CE。Query-on 的 `loss_type=vlm` / `vlm_and_action` 仍走 conditional teacher-forced LM loss。Query-off 则无论 action-only 还是 direct 都严格走 `Qwen3VLForConditionalGeneration.forward`，保留原 `input_ids`、二维 mask、labels、`output_hidden_states=True`、训练/推理 `use_cache` 和 PEFT/LoRA wrapper/hook；有 labels 时即使 CE 不进入 action-only 总 loss，也保留 LM head/logits/CE 计算。两种 action 条件路径的 final hidden 在固定输入下于测试容差内一致，但计算成本不同。Query 模式的 wrapper、公开 `backbone.model.generate()` 和 PEFT base model 的 autoregressive `.generate()` 都在真正生成前抛出 `NotImplementedError`；guard 使用模块级具名方法，加载时重新安装，不进入 checkpoint state。

### 18.5 checkpoint 与入口

每次 `save_pretrained` 都写 `difference_query_config.json`；启用时另写单键 FP32 `difference_query.safetensors`，关闭时不保留 Query 权重。VLM/action 两个规范化加载目录会分别检查 config/weight 配对、enabled、Nq、hidden size、tensor shape 和数值有限性；两处都有声明时 config 必须完全相同，启用时权重也必须完全相同。普通保存、DeepSpeed 的 `latest-model-optimizer-lr` 导出、`from_pretrained`、train CLI、policy/server 和 direct-action 均使用同一解析器。真实 tiny Qwen+真实 processor+真实 Action Expert 的完整 round-trip 已验证 Query、backbone hidden 和固定种子动作一致。

### 18.6 公平实验和验证状态

`scripts/run_libero_wo_ecot_pt.sh` 提供 `baseline_fa2`、`baseline_sdpa`、`difference_query` 三臂。三者的 Qwen3-VL-2B、随机 Action Expert、seed 42、数据、global batch 64、8 epochs、optimizer/scheduler、H=10 和评估参数相同；仅 Query/backend、输出目录和 W&B 标识不同。`baseline_fa2` 是原 HEAD 调用兼容基线，`baseline_sdpa` 只改变 attention backend；`difference_query` 的 action-only 使用高效 base-model 路径。正式 train/resume 在启动前把根目录实验模板写入对应 `OUTPUT_DIR/experiment.md`，并追加 arm、时间、W&B 标识和完整命令。Query+SDPA 与 FA2 基线同时改变了压缩方式和 backend，必须用无 Query+SDPA 控制组分离 backend 影响；三臂成功率仍按相同 batch、数据和优化配置比较，但运行时间/显存因调用层级不同，不能直接作为 Query 加速结论。

自动验证覆盖关闭路径 HEAD 参数/PEFT hook/final hidden/mask/固定种子动作 golden、向量化 C/Q/T/P mask 与最大长度 CPU 结构、Nq=8/32/64 的 mask/真实 Qwen/direct-action、隔离 RNG、disabled/enabled/checkpoint 损坏分支与真实 round-trip、隔离 CLI `None/True/False/Nq`、底层 generate 拒绝、Query-on action-only LM-head 拒绝、tiny Qwen 的 T/C 隔离、真实 PIL processor+vision+DeepStack+mRoPE，以及 tiny Qwen+真实 tiny Action Expert 的 action-only backward/direct denoise。恢复测试跨两个 epoch，并覆盖单进程未初始化 distributed；batch 前缀只在恢复 epoch 跳过，后续 epoch 不重复跳过。CUDA BF16 tiny Query 已实际捕获 efficient SDPA kernel；ZeRO-2 的独立 `torchrun` world-size=1 与 2-rank 已实际通过生产 helper 的更新、Query sidecar/model/optimizer/scheduler/global-step 保存恢复及继续一步，并验证跨 rank Query 一致和有限性；4-GPU 真实 2B 生产 smoke 未执行。正式训练和 LIBERO rollout 尚未执行；97.6% 仅是官方 checkpoint 的历史评估参考，不是本三臂结果。

## 19. Stage05 purpose contract and configurable downstream horizon

Stage05 checkpoint consumers explicitly pass `checkpoint_load_purpose`: `stage05_ar_to_joint` (AR-only source, random Expert, source experiment horizon), `stage05_joint_resume` (same Joint experiment, Expert/optimizer/scheduler/step resume, unchanged horizon), or `downstream_finetune` (new experiment, pretrained VLM/DQ32/Expert weights, fresh optimizer/scheduler and an explicit target horizon). Direct inference uses the explicit `inference` artifact-load purpose. The legacy `allow_ar_warm_start` flag cannot bypass a Stage05 contract.

The Action Expert contract separates `architecture_hash` from `runtime_contract_hash`. The former hashes the complete parsed configuration except `action_horizon`; the latter includes source/target horizon, purpose, checkpoint kind and Stage05 manifest. Production-shape comparisons for H=32/10/16/8, including 32->10, 16->8, 8->16 and unchanged H, have identical complete `state_dict` keys/shapes because H only controls the runtime action-token slice. Every explicit target H is a positive integer checked against `max_seq_len=256` and `max_num_positional_embeddings=128`; weights therefore load with `strict=True`, with no `strict=False` or resize path. Downstream fresh initialization is the only mode allowed to override H and save a checkpoint declaring that target H; resume and inference restore the saved H. LIBERO action normalization and `stats_key` are resolved from LIBERO, never from the Stage05 four-dataset stats.

Stage05 AR and Joint are distinguished by their resolved production manifest (`vlm` versus `vlm_and_action`) and checkpoint kind. Stage05 Joint save persists both contracts and complete Expert weights. A checkpoint with any Stage05 identity cannot fall back to the legacy path when its contract is missing. The legacy Tabletop step-19424 used by the LIBERO launcher instead passes the strict old-generic path: its complete config and Expert safetensors keys/shapes are validated, all pretrained weights are loaded, and fresh downstream state is created without fabricating Stage05 history. Joint resume additionally parses scheduler, DeepSpeed/client and optimizer state and checks global-step consistency before allocation; this is CPU preflight evidence, not a completed multi-GPU resume.

Stage05 token length is additionally trusted through the versioned `configs/stage05_four_dataset_experiment.json`. It fixes audit format v2, `required_max_length=941`, the audit report content/file hashes and implementation identity. Launcher validation happens before model/data allocation and rejects report self-rehashes, sidecar/processor/source drift and environment-variable hash overrides. The regenerated v9 report is the current Stage05 input identity; LIBERO uses its existing tokenization contract independently of the Stage05 941 gate.
