# ZR-0 Gradient Explosion Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the Qwen-to-action-expert feature interface, stop on non-finite training state, and prove the LIBERO w/o ECoT PT run remains stable before restarting the full experiment.

**Architecture:** Keep the existing Qwen conditional-generation forward, but pass its recorded final decoder output through Qwen's own final text RMSNorm before cross-attention. Add a distributed scalar-finiteness guard around loss and DeepSpeed's global gradient norm, plus an optional maximum-step stop that leaves the full eight-epoch scheduler unchanged.

**Tech Stack:** Python 3.10, PyTorch 2.6, Transformers 4.57.1, PEFT 0.17.1, Accelerate 1.6.0, DeepSpeed 0.15.4, BF16, unittest/pytest, Bash, W&B.

## Global Constraints

- Preserve Qwen3-VL-2B-Instruct initialization, a random action expert, public LIBERO v2.1 only, global batch size 64, action horizon 10, and action loss weight 1.
- Preserve peak LR `2e-5`, cosine schedule, minimum LR rate `0.1`, BF16, ZeRO-2, gradient clipping `1.0`, seed `42`, and eight full epochs.
- Do not freeze or detach the VLM. Do not load an action-expert checkpoint for a fresh run.
- Do not resume from or overwrite the failed run.
- Stage all changes and do not create a git commit.

---

### Task 1: Return Qwen's Final-Normalized Features

**Files:**
- Create: `tests/test_qwen_vl_backbone.py`
- Modify: `model/qwen_vl_backbone.py:1-10,155-167`

**Interfaces:**
- Consumes: a plain Qwen conditional model or a PEFT wrapper exposing `get_base_model()`.
- Produces: `_get_qwen_final_text_norm(model: nn.Module) -> nn.Module` and normalized `backbone_embeddings`.

- [ ] **Step 1: Write the failing regression test**

Create a fake conditional model containing `model.language_model.norm`, returning a deliberately unnormalized tensor as `hidden_states[-1]`. Wrap it in a fake PEFT object for a second subtest. Construct `QwenVLBackbone` with `__new__`, call `forward()`, and require both variants to return the output of the fake model's `nn.RMSNorm`, not the raw tensor:

```python
class FakePeftModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model_for_test = base_model

    def get_base_model(self):
        return self.base_model_for_test

    def forward(self, **kwargs):
        return self.base_model_for_test(**kwargs)


def test_forward_returns_final_rmsnorm_output_for_plain_and_peft_models():
    raw = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])
    for wrap_in_peft in (False, True):
        base_model = FakeConditionalModel(raw)
        model = FakePeftModel(base_model) if wrap_in_peft else base_model
        backbone = make_backbone_without_loading_weights(model)
        outputs = backbone(BatchFeature({
            "input_ids": torch.tensor([[1, 2]]),
            "sub_task_flag": torch.tensor([0]),
        }))
        expected = base_model.model.language_model.norm(raw)
        torch.testing.assert_close(outputs.backbone_embeddings, expected)
        assert not torch.equal(outputs.backbone_embeddings, raw)
```

- [ ] **Step 2: Run the test and verify RED**

```bash
PYTHONNOUSERSITE=1 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -m pytest tests/test_qwen_vl_backbone.py -q
```

Expected: FAIL because the current backbone returns the raw decoder output.

- [ ] **Step 3: Implement the minimal correction**

Add to `model/qwen_vl_backbone.py`:

```python
def _get_qwen_final_text_norm(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    return model.model.language_model.norm
```

Replace the raw embedding assignment with:

```python
raw_last_hidden_state = model_outputs["hidden_states"][-1]
embeddings = _get_qwen_final_text_norm(self.model)(raw_last_hidden_state)
```

- [ ] **Step 4: Run the Task 1 test and verify `1 passed`**

- [ ] **Step 5: Stage Task 1 without committing**

```bash
git add model/qwen_vl_backbone.py tests/test_qwen_vl_backbone.py
git diff --cached --check
```

---

### Task 2: Stop All Ranks On Non-Finite Loss Or Gradient Norm

**Files:**
- Create: `utils/training_numerics.py`
- Create: `tests/test_training_numerics.py`
- Modify: `train_vla.py:1-16,373-449`

**Interfaces:**
- Produces: `assert_all_finite(accelerator, value, name: str, step: int) -> None`.
- Consumes: `accelerator.device`, `accelerator.num_processes`, and `accelerator.reduce(..., reduction="sum")`.

- [ ] **Step 1: Write failing tests**

Use a four-rank fake accelerator whose reduction returns a configured finite count. Require count 4 to pass, local NaN/count 0 to raise, and remote failure/count 3 to raise on every rank:

```python
def test_all_finite_values_pass_on_every_rank():
    assert_all_finite(FakeAccelerator(finite_count=4), torch.tensor(1.25), "loss", 7)


def test_local_nan_raises_with_step_and_metric():
    with pytest.raises(FloatingPointError, match="loss.*step 7"):
        assert_all_finite(FakeAccelerator(finite_count=0), torch.tensor(float("nan")), "loss", 7)


def test_remote_non_finite_gradient_raises_everywhere():
    with pytest.raises(FloatingPointError, match="global gradient norm.*step 11"):
        assert_all_finite(FakeAccelerator(finite_count=3), 2.0, "global gradient norm", 11)
```

- [ ] **Step 2: Run the test and verify collection fails because `utils.training_numerics` is absent**

```bash
PYTHONNOUSERSITE=1 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -m pytest tests/test_training_numerics.py -q
```

- [ ] **Step 3: Implement the helper**

```python
import math
import torch


def assert_all_finite(accelerator, value, name: str, step: int) -> None:
    if isinstance(value, torch.Tensor):
        local_finite = torch.isfinite(value.detach()).all()
    else:
        local_finite = torch.tensor(math.isfinite(float(value)), device=accelerator.device)
    finite_count = accelerator.reduce(local_finite.to(torch.long), reduction="sum")
    if int(finite_count.item()) != accelerator.num_processes:
        raise FloatingPointError(
            f"Non-finite {name} detected across training ranks at step {step}."
        )
```

- [ ] **Step 4: Verify the Task 2 unit tests pass**

- [ ] **Step 5: Integrate both guards in `train_vla.py`**

Before backward, call the helper for `loss` and `global_completed_steps + 1`. Immediately after `accelerator.backward(loss)`, optimizer/scheduler calls, and `zero_grad()`, read `model.get_global_grad_norm()` and call the helper when it is not `None`. This check must precede global-step updates and all checkpoint saves. Cache the finite norm for TensorBoard and W&B instead of querying it twice later.

- [ ] **Step 6: Run focused tests**

```bash
PYTHONNOUSERSITE=1 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -m pytest tests/test_training_numerics.py tests/test_wandb_training_logger.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Stage Task 2 without committing**

```bash
git add utils/training_numerics.py tests/test_training_numerics.py train_vla.py
git diff --cached --check
```

---

### Task 3: Add A Scheduler-Faithful Smoke Stop And Fresh Output Path

**Files:**
- Modify: `train_vla.py:20-70,282-319,326-469`
- Modify: `scripts/run_libero_wo_ecot_pt.sh:1-150`
- Modify: `tests/test_libero_wo_ecot_pt_launcher.py:15-110`

**Interfaces:**
- Produces: optional `--max_train_steps N`, leaving `num_total_batches` and scheduler horizon at eight epochs.
- Produces: launcher overrides `ZR0_OUTPUT_DIR`, `ZR0_MAX_TRAIN_STEPS`, and `ZR0_SAVE_STEP_INTERVAL`.

- [ ] **Step 1: Add failing launcher tests**

Extend `run_launcher()` to accept `extra_env`. Add a smoke test with a unique output path, max steps 200, and save interval 100; require all three values in the dry-run command. Add another test requiring the default output path to contain `FinalRMSNorm` and not equal the failed output directory.

```python
result = self.run_launcher("train", extra_env={
    "ZR0_OUTPUT_DIR": str(smoke_output),
    "ZR0_MAX_TRAIN_STEPS": "200",
    "ZR0_SAVE_STEP_INTERVAL": "100",
})
self.assertIn("--max_train_steps 200", result.stdout)
self.assertIn("--save_step_interval 100", result.stdout)
self.assertIn(f"--output_ckpt_dir {smoke_output}", result.stdout)
```

- [ ] **Step 2: Run launcher tests and verify the new assertions fail**

```bash
PYTHONNOUSERSITE=1 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -m pytest tests/test_libero_wo_ecot_pt_launcher.py -q
```

- [ ] **Step 3: Add the optional training stop**

Add `--max_train_steps` with default `None` and reject values below one. After each completed distributed step compute:

```python
reached_max_train_steps = (
    opt.max_train_steps is not None
    and global_completed_steps >= opt.max_train_steps
)
do_save = (
    global_completed_steps > 0
    and (
        global_completed_steps % opt.save_step_interval == 0
        or reached_max_train_steps
    )
)
```

Break the batch loop after logging when the maximum is reached, then break the epoch loop. Do not change `num_total_batches`, warmup steps, or scheduler construction.

- [ ] **Step 4: Add launcher overrides**

```bash
OUTPUT_DIR=${ZR0_OUTPUT_DIR:-"$ROOT_DIR/outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm"}
SAVE_STEP_INTERVAL=${ZR0_SAVE_STEP_INTERVAL:-2000}
MAX_TRAIN_STEPS=${ZR0_MAX_TRAIN_STEPS:-}
```

Use `$SAVE_STEP_INTERVAL` in the command and conditionally append `--max_train_steps "$MAX_TRAIN_STEPS"` when nonempty.

- [ ] **Step 5: Verify Tasks 1-3 together**

```bash
PYTHONNOUSERSITE=1 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -m pytest tests/test_libero_wo_ecot_pt_launcher.py tests/test_training_numerics.py tests/test_qwen_vl_backbone.py -q
```

- [ ] **Step 6: Stage Task 3 without committing**

```bash
git add train_vla.py scripts/run_libero_wo_ecot_pt.sh tests/test_libero_wo_ecot_pt_launcher.py
git diff --cached --check
```

---

### Task 4: Static And Real-Model Verification

**Files:** Verify only.

**Interfaces:** Consumes Tasks 1-3 and produces fresh regression and GPU evidence.

- [ ] **Step 1: Run all lightweight tests**

```bash
PYTHONNOUSERSITE=1 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -m pytest tests -q
```

- [ ] **Step 2: Compile changed Python files**

```bash
PYTHONNOUSERSITE=1 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -m py_compile model/qwen_vl_backbone.py utils/training_numerics.py train_vla.py
```

- [ ] **Step 3: Verify one real LIBERO sample through base Qwen**

Use `build_concat_streaming_dataset` and the corrected `QwenVLBackbone`. Hook the same forward's `language_model.norm` and assert exact equality with returned `backbone_embeddings`, all values finite, and RMS below 10.

- [ ] **Step 4: Repeat the step-20000 action-head comparison**

With RNG seed 12345, require final-normalized feature RMS below 10, finite loss/gradients, and a lower action-head gradient norm than the raw-feature path.

---

### Task 5: Four-GPU Smoke Training And Checkpoint Audit

**Files:** Runtime artifacts only in a new timestamped smoke directory.

**Interfaces:** Consumes the corrected launcher and the user's W&B credential; produces a 200-step BF16/ZeRO-2 run.

- [ ] **Step 1: Run preflight and verify GPUs 0-3 each have at least 70 GiB free**

```bash
PYTHONNOUSERSITE=1 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python scripts/preflight_libero_wo_ecot_pt.py --require-wandb
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
```

- [ ] **Step 2: Launch 200 isolated steps**

Set a timestamped `ZR0_OUTPUT_DIR`, a smoke-specific `ZR0_RUN_NAME`, `ZR0_MAX_TRAIN_STEPS=200`, and `ZR0_SAVE_STEP_INTERVAL=100`, then execute `bash scripts/run_libero_wo_ecot_pt.sh train`. Require four ranks, global batch 64, a random expert, online W&B, step-100/200 saves, and clean exit at 200.

- [ ] **Step 3: Verify smoke metrics**

Read every W&B `train/loss` and `train/grad_norm` value. Assert all values are finite and report min, max, and last values.

- [ ] **Step 4: Verify every smoke checkpoint tensor is finite**

Open every `.safetensors` file at steps 100 and 200 using `safetensors.safe_open`, load every tensor, and assert `torch.isfinite(tensor).all()` with zero failures.

- [ ] **Step 5: Stage and audit everything without committing**

```bash
git add model/qwen_vl_backbone.py train_vla.py utils/training_numerics.py scripts/run_libero_wo_ecot_pt.sh tests/test_qwen_vl_backbone.py tests/test_training_numerics.py tests/test_libero_wo_ecot_pt_launcher.py docs/superpowers/specs/2026-09-01-zr0-gradient-explosion-fix-design.md docs/superpowers/plans/2026-09-01-zr0-gradient-explosion-fix.md
git diff --cached --check
git status --short
```

---

### Task 6: Restart The Full Ablation From Scratch

**Files:** Runtime artifacts under the new `FinalRMSNorm` output and log directories only.

**Interfaces:** Consumes a passing smoke gate and produces a new full eight-epoch run.

- [ ] **Step 1: Re-run preflight, verify four free GPUs, and ensure the new full output path is absent**

- [ ] **Step 2: Launch without smoke overrides**

Unset `ZR0_OUTPUT_DIR`, `ZR0_MAX_TRAIN_STEPS`, and `ZR0_SAVE_STEP_INTERVAL`; use a fresh run name and execute `bash scripts/run_libero_wo_ecot_pt.sh train`. Confirm four GPUs, per-device batch 16, global batch 64, eight epochs, horizon 10, action-only loss, random expert, peak LR `2e-5`, BF16 ZeRO-2, and W&B online.

- [ ] **Step 3: Monitor through the first saved checkpoint**

Require all logged loss and global-gradient values through step 2000 to be finite and scan every saved tensor for finiteness. Keep the goal active while this evidence is incomplete; process startup alone does not prove the issue resolved.
