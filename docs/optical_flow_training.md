# Optional Optical Flow Training

`training_stage` is the objective authority. CLI tracks whether `loss_type` was
explicit: an omitted legacy default is not a conflict. Without a stage, all three
legacy loss modes are unchanged and OF defaults off. Slot is not implemented;
`slot_aux_type=none` creates no module and emits no numeric `slot_loss`.

| Stage | Computed objectives | Action Expert |
|---|---|---|
| `stage1_ar` | AR only | absent |
| `stage2_aux` | OF only, Slot not implemented | absent |
| `stage3_joint` | AR+FM, optional OF; provisional joint | fresh seeded initialization unless explicitly loaded |

Stage2 requires `dense_regression_v1`, a positive OF weight and an explicit query
count. Stage3 with OF is **AR+FM+OF provisional joint**, not four-loss training.
AR and FM weights must be positive in stage3. Training defaults do not silently
enable OF. The final `num_flow_queries` of the existing Difference Query states
feed the Head; the Action Expert continues to read all Query states.

The source LIBERO v3 dataset has no `train_data` field. Its Stage06 entry supplies
OF and, for joint training, FM. Include a dataset with actual AR labels for
stage3, such as `molmoact_tabletop_v3_stage05`. No AR targets are synthesized from
missing text. Dataset adapters keep their own image/action/statistics contracts.

## Commands

The launcher is a template, not an executed experiment. Before formal training,
create `$OUTPUT_DIR/experiment.md` with all fields required by AGENTS.md,
including exact source config, module learning rates, data mixture, preprocessing,
action horizons, W&B identity and the expanded command. W&B is required by the
launcher; connectivity failures must stop formal training.

```sh
export INIT_CHECKPOINT=outputs/pretrain/tabletop_v3_dq32_ar_gbs128_seed42_mbs16_gas2/step-7284
export OUTPUT_DIR=outputs/optical_flow_stage2
export NUM_FLOW_QUERIES=8 OPTICAL_FLOW_LOSS_WEIGHT=1.0
export WANDB_PROJECT=zr0-flow WANDB_RUN_NAME=stage2-of-only
export ACCELERATE_CONFIG=path/to/existing_deepspeed_config.yaml
bash scripts/run_optical_flow_stage.sh stage2_aux --print-command
```

Remove `--print-command` only when formal training is intended. Stage3 uses the
same template with these explicit changes:

```sh
export INIT_CHECKPOINT=outputs/optical_flow_stage2/latest-model-optimizer-lr
export OUTPUT_DIR=outputs/optical_flow_stage3
export WANDB_RUN_NAME=stage3-ar-fm-of-provisional
export AR_DATASET_ENTRY=molmoact_tabletop_v3_stage05
bash scripts/run_optical_flow_stage.sh stage3_joint --print-command
```

For same-stage resume, use the expanded command, replace `--init_from_checkpoint`
with `--resume_from_checkpoint PATH/latest-model-optimizer-lr`, and keep the stage,
OF/query architecture, seed, batch, accumulation and dataset contracts identical.
Set the original W&B ID and `--wandb_resume must`. Only the checkpoint with
optimizer/scheduler/runtime states supports resume; a `step-N` weights export
supports initialization, not complete training resume.

## Contracts

- Init loads VLM/Query and any saved OF Head. Optimizer, scheduler, sampler and RNG
  start fresh at step 0. Source Expert weights are ignored, even when present.
  Stage3 Expert construction uses the run seed in a CPU RNG isolation context.
  `--action_expert_name_or_path` must name an independent valid Expert checkpoint
  to override fresh Expert initialization. It does not override VLM Query weights.
- Resume loads all model weights and DeepSpeed optimizer/scaler state, external
  scheduler state and per-rank RNG/data cursor. Global step counts updates; cursor
  counts consumed batches, including skipped batches. Changed sampler/world size
  is rejected. Worker randomness is not serialized; Stage06 has no random
  augmentations, and exact continuation requires deterministic dataset adapters.
  `ZR0Model.from_pretrained` on a saved stage reconstructs that stage's model
  weights, including a trained stage3 Expert. An explicit `init_from_checkpoint`
  requests fresh Expert initialization instead. Optimizer restoration remains
  the training entrypoint's responsibility.
- Save includes `optical_flow_aux_config.json`, optional
  `optical_flow_aux.safetensors`, config/weight SHA256, stage, and VLM/Query
  before/after initialization checksums. Missing/corrupt/conflicting artifacts
  fail loading. Existing Stage05 source contracts are still validated.
- Enabled OF checkpoints require enabled Difference Query config and weights,
  consistent query counts, and the same Query/Flow input hidden size. An OF
  sidecar without a Query declaration is corrupt, never a legacy source for
  random Query initialization. Disabled OF stage metadata still permits an
  explicitly disabled Query declaration, preserving stage1 Query-off checkpoints.
- OF supports the existing ZeRO-2 export path and non-ZeRO parameters only.
  Training startup and both checkpoint save entrypoints reject ZeRO-3; direct
  Head export also rejects parameters bearing DeepSpeed partition metadata.
  No full-parameter ZeRO-3 gather is implemented. A file hash verifies bytes,
  not complete parameter shapes. This review tested non-ZeRO CPU save/load and
  simulated rejection, not OF on a real ZeRO-2/ZeRO-3 engine. The repository's
  earlier ZeRO-2 training validation predates this auxiliary head.
- Direct-action inference validates saved artifacts and skips OF prediction.
  No flow root, HDF5 or flow labels are required by inference.
- Stage06 training and policy use `resolve_dataset_spec` for label provenance.
  Training hashes the JSONL manifest; inference resolves the same camera, delta
  and digest from the checkpoint's hashed `resolved_dataset_manifest.json`.
  Source dataset image metadata and action statistics are still validated, but
  inference does not initialize `OpticalFlowReader` or scan HDF5 files. This
  verifies the recorded training provenance, not the current dense label bytes.
- Head input is cast once, differentiably, to the projection parameter dtype.
  A default FP32 Head accepts BF16/FP16 Qwen hidden states without autocast.
  Explicit `head.to(dtype)` controls saved parameter dtype; outer autocast may
  select operation dtypes, while OF loss arithmetic remains FP32. Native FP16
  training still needs normal loss-scaling care for small gradients.
- Source flow/mask is 224x224. Mask-weighted area pooling produces 56x56 targets;
  cells require at least .5 valid coverage. Components stay normalized to source
  image extent. Flow delta=10; shortened tail/identity labels are not supervised.
- `L_OF = mean_eligible(mean_valid(sqrt(EPE^2+epsilon^2)-epsilon)
  + motion_weight * mean_motion(sqrt(EPE^2+epsilon^2)-epsilon))`, epsilon=.001,
  motion threshold=.01, motion weight=1. Empty motion masks contribute no term.
  Stage2 total=`OF_weight*L_OF`; stage3 adds `AR_weight*AR + FM_weight*FM`.
- Numerators/counts span the entire accumulation window and all ranks. Backward
  scales local numerators by world size and accumulation to match a single global
  mean under Accelerate/DDP averaging. Missing samples enter neither numerator
  nor denominator and have no dummy zero-flow label. Empty local ranks retain a
  graph-connected zero. Globally empty stage2 windows skip forward/backward,
  AdamW, scheduler and global step. Stage3 retains available AR/FM objectives.
- Logs: OF/EPE/motion EPE/zero baseline, valid and motion fractions, eligible
  samples/pixels, coverage and cumulative skipped batches. The existing optimizer
  grouping, LR/scheduler and weight decay defaults are reused; no hidden LR group
  multipliers are introduced. Default production Head has 2,753,538 parameters at
  VLM width 2048, independent of the number of Flow Queries.
- JSONL and W&B preserve `training_stage`, `provisional`, and
  `slot_loss_computed`, `ar_loss_computed`, `flow_loss_computed`,
  `fm_loss_computed`. A computed flag means at least one microbatch on at least
  one rank had valid supervision for that loss. A supervised numerical zero is
  computed; a graph-connected zero without labels is not. Slot stays false and
  has no numeric loss. Skipped stage2 windows log false flags at the unchanged
  global step. All tensor collectives, including W&B metrics and loss weights,
  use `accelerator.device`; strings and globally resolved booleans are payload
  metadata. Real GPU/NCCL logging has not been validated by the CPU checks.

See `optical_flow_data_audit.md` for label provenance and
`experiments/optical_flow_cpu/experiment.md` for bounded verification results.
# Second review contracts (2026-09-06)

Stage06 persists the **model image input contract** in the resolved dataset
manifest: integer `image_width`/`image_height` resize hints (currently 224x224),
boolean `do_resize=false` for the processor, and boolean
`random_geometric_augmentation=false`. The hints are consumed by
`process_vision_info`; the processor does not perform a second resize. Policy
reads this saved contract and checks it against the shared runtime definition.
Missing fields, wrong types, or mismatched sizes/preprocessing fail even when
the manifest hash is valid. This validation uses JSON metadata, not HDF5 scans.
Source images (256x256), MegaFlow label generation output (224x224), and loss
targets (56x56) are separate contracts; matching numbers do not establish
identical interpolation between the external generator and model preprocessing.

Staged checkpoints share these structured fields between
`zr0_checkpoint_metadata.json` and `optical_flow_aux_config.json`:

| Field | Meaning |
| --- | --- |
| `training_stage`, `stage_description` | Canonical current stage and readable description |
| `provisional` | Always true for stage3 while Slot is unimplemented |
| `ar_loss_computed`, `flow_loss_computed`, `fm_loss_computed` | Any completed optimizer window in the current stage had actual corresponding supervision on any rank |
| `slot_loss_computed` | Always false; no numeric Slot loss is fabricated |
| `stage_metadata_version` | 2: cumulative state requires a confirmed optimizer update; v1 is legacy |
| `completed_optimizer_windows` | Confirmed optimizer updates; a lower bound when legacy history is unknown |
| `loss_computed_scope` | `current_stage_completed_optimizer_windows` |
| `legacy_history_unknown` | Prior history unavailable; unknown computed flags are JSON null |

Fresh saves before training have false flags even if objectives are configured.
Standalone forward/evaluation does not establish training history. A supervised
loss of zero counts; a skipped global empty stage2 window does not. Flags are
cumulative ORs of globally reduced window flags, not the latest window metrics.
JSONL/W&B continue to report actual per-window computed flags separately.
Explicit init (including same-stage init) resets this history; resume restores
it. Source evidence is available as `source_stage_training_state` during loading.
Old staged checkpoints without this schema (including v1 without AMP outcome
checks) warn and preserve unknown history; verified window count starts at zero;
future observed losses become true, while unobserved legacy history stays null.
Stage3 remains provisional independently of whether it has run any window.

Stage2 accepts explicit `loss_type="aux"` in forward and CLI. It still computes
only supervised OF, never AR/FM or Action Expert, and never Slot. Other stages
reject aux as a mode conflict. No existing loss default or Query-off path changes.

## Optimizer update outcomes (2026-09-06)

One backend outcome controls scheduler advancement, durable global step and
checkpoint cumulative state. Window logs include boolean
`optimizer_update_applied`/`optimizer_update_skipped` and `optimizer_skip_reason`:

| Reason | Backward | Update/scheduler/global step/completed count |
| --- | --- | --- |
| `none` | Executed | Confirmed update, advance once |
| `no_supervision` | Not executed | All skipped |
| `amp_overflow` | Executed | Actual optimizer skipped, no advancement |
| `optimizer_step_skipped` | Executed | Backend reports another no-update result, no advancement |

Per-window computed flags still describe actual forward supervision, so an
overflow window can have `flow_loss_computed=true` in JSONL/W&B while adding no
checkpoint evidence. Skip logs use the unchanged global step. Only successful
updates accumulate checkpoint flags and completed count. StepLR and the existing
constant/cosine schedules follow the same outcome. Engine-owned DeepSpeed
schedulers are guarded internally and are not stepped twice by the trainer;
skipped engine scheduler state must remain unchanged.

Supported outcome providers are AcceleratedOptimizer.step_was_skipped,
DeepSpeedEngine.was_step_applied, and native non-AMP PyTorch AdamW/Adam/SGD
(which have no implicit skip path). Unknown backends, multiple optimizers,
unsupported optimizer implementations and inconsistent rank outcomes fail
explicitly. No update is inferred from loss finiteness, backward calls, scale
changes or the DeepSpeed optimizer wrapper's no-overflow fallback.

CPU AMP and Gloo validation does not establish GPU/NCCL or DeepSpeed dynamic
save/resume correctness. ZeRO-3 remains unsupported; real ZeRO-2 and GPU/NCCL
smoke still require separate execution. No formal training is started here.
