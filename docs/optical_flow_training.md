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

## Wan VAE Latent V2

Select V2 with the existing `--optical_flow_aux_type wan_vae_latent_v2` switch.
It additionally requires an explicit `--flow_vae_model_path`, a locked positive
`--flow_color_scale`, delta/source/valid-fraction rules and enabled Difference
Query. `none`, Query-off and `dense_regression_v1` do not additionally import the
V2 target module, parse a Wan path, instantiate `AutoencoderKLWan` or access a V2
cache; the existing Action Expert has its own Diffusers dependency, and V1 retains
its existing HDF5 label reads. Action inference validates the
checkpoint sidecar but omits the Adapter and never resolves the VAE path.

The online builder loads only `AutoencoderKLWan`, freezes it in eval mode and
uses deterministic `latent_dist.mode()`. RGB `[B,3,1,224,224]` is transformed
from `[0,1]` to `[-1,1]`; the result is normalized by the VAE's 48 saved means
and standard deviations. The downloaded Wan2.2 VAE measures
`[B,48,1,14,14]`; any `Tz != 1` fails. Decoder, RGB loss, EPE, KL and diffusion
loss are absent from training. The Adapter predicts `[B,48,14,14]` from only
the final `num_flow_queries`; its default 2048-to-256, two-layer, four-head
configuration has 2,172,208 parameters.

V2 loss is the global mean of per-sample FP32 latent-element MSE. One selector
applies availability, per-source nominal/configured delta, label source and
whole-mask valid fraction to the loss, activity check, metrics and distributed
denominator. A globally empty OF-only accumulation window clears gradients and
does not update AdamW, scheduler, successful step or stage history. Joint AR/FM
supervision still updates when V2 has no eligible sample.

First create one immutable training-split calibration file. This is a deliberate
offline operation and was not run during the implementation review:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=. \
/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python \
  scripts/calibrate_flow_color_scale.py \
  --root /opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05/stage06_flow/molmoact_tabletop \
  --manifest /opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05/stage06_flow/molmoact_tabletop/manifest.f61339e88e1b99c9.jsonl \
  --dataset-root /opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05 \
  --output '<INDEPENDENT_CACHE_ROOT>/flow_color_calibration.train.json' \
  --split train --expected-actual-delta 20 --label-source 1 \
  --training-index '<MANIFEST_BOUND_TRAIN_FRAME_INDEX_JSON>' \
  --min-valid-fraction 0.95 --quantile 0.99 \
  --reservoir-capacity 1000000 --seed 42
```

The required training index is a version-1 JSON split contract with `split=train`,
`selection=explicit_episode_frame_allowlist`, the Flow manifest SHA256, exact
dataset/camera lists, and explicit `{episode, frame}` rows. `--dataset-root`
binds that allowlist to the dataset's hashed `meta/info.json` split ranges and
hashed `meta/stage05_episode_mapping.jsonl`; `meta/stage05_merge.json` binds the
manifest dataset ID to that source. Each Flow `source_episode_index` is
mapped through the authoritative old-to-new episode map before train membership
is accepted; Flow and dataset episode IDs are not assumed equal. The authority
must assign every episode exactly once, so an incomplete split sidecar is
rejected. The script
rejects validation/test rows, missing authority metadata, mapping conflicts and
out-of-range rows. For a dataset with no verifiable split metadata, the only
fallback is the explicit `--externally-trusted-training-index` flag; its output
states `training_membership=not_independently_verified` and must not be described
as independently leakage-checked.
In independently verified mode, `meta/info.json` must declare every manifest
camera as an image/video feature and `meta/modality.json` must include it in the
published video contract. The mapping's `source_data_uri` is resolved relative to
the dataset root (absolute URIs are accepted only when their resolved target is
still inside that root), must be an existing Parquet file, and is checked against
the compact `meta/episodes` sidecar. The actual Parquet `episode_index` metadata
must cover the mapped Stage05 target episode and the sidecar row count; one
Parquet file may therefore legitimately serve several mapped episodes. On first
use of a unique source file, the verifier reads only its `episode_index` column
in fixed-size batches and compares per-episode counts with the sidecar; it does
not decode image/state/action columns. Resolved
source paths, file sizes, source/target episode identities, camera feature shape
and the modality digest are recorded in the calibration authority contract.
Missing, external, symlink-escaped or identity-conflicting sources fail closed.
It uses a uniform fixed-capacity reservoir and reports that its quantile is
estimated. Output includes the split-contract/index hashes, included frame range,
units, sample and observed counts, seed, filter funnel, coverage and truncation
estimate. It refuses to overwrite an existing output. Validation/test jobs must
read the locked `flow_color_scale` from this train result.

Calibration validates each HDF5 block before filtering it. The raw `valid_mask`
must contain only 0/1 values, `valid_fraction` must be finite, in range and agree
with the mask mean, and all flow values in the block must be finite. These checks
cover tail and nonmatching `actual_delta_frames`/`label_source` rows. Only then
are train, full-delta, label-source and minimum-valid-fraction filters applied.

```json
{
  "version": 1,
  "split": "train",
  "selection": "explicit_episode_frame_allowlist",
  "flow_manifest_sha256": "<SHA256>",
  "dataset_ids": ["molmoact_tabletop"],
  "cameras": ["first_view"],
  "frames": [{"episode": 0, "frame": 0}]
}
```

For a bounded cache smoke, substitute that locked scale and encode only the
requested number of samples. The cache directory must be separate from labels:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=. \
/opt/data/private/lq/miniconda3/envs/ZR-0/bin/python \
  scripts/build_flow_latent_cache.py \
  --root /opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05/stage06_flow/molmoact_tabletop \
  --manifest /opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05/stage06_flow/molmoact_tabletop/manifest.f61339e88e1b99c9.jsonl \
  --vae-model-path /opt/data/private/lq/models/Wan2.2-TI2V-5B-Diffusers \
  --scale '<LOCKED_TRAIN_SCALE>' --expected-actual-delta 20 --label-source 1 \
  --min-valid-fraction 0.95 --cache-dtype float32 --limit 8 \
  --output '<INDEPENDENT_CACHE_ROOT>/wan_v2_latents'
```

The following stage-2 template is complete after replacing the uppercase path
placeholders and locked scale. It is documentation only and was not launched:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=. \
/opt/data/private/lq/miniconda3/envs/ZR-0/bin/accelerate launch \
  --config_file '<ZERO2_ACCELERATE_CONFIG>' --num_processes 4 \
  --mixed_precision bf16 --gradient_accumulation_steps 2 train_vla.py \
  --training_stage stage2_aux --loss_type aux \
  --vlm_name_or_path '<STAGE1_CHECKPOINT>' \
  --init_from_checkpoint '<STAGE1_CHECKPOINT>' \
  --action_expert_config_path '<STAGE1_CHECKPOINT>/action_expert_config.json' \
  --use_difference_query --num_difference_queries 32 --num_flow_queries 16 \
  --optical_flow_aux_type wan_vae_latent_v2 --optical_flow_loss_weight 1.0 \
  --flow_vae_model_path /opt/data/private/lq/models/Wan2.2-TI2V-5B-Diffusers \
  --flow_color_scale '<LOCKED_TRAIN_SCALE>' --flow_delta_frames 20 \
  --flow_label_source 1 --flow_sample_min_valid_fraction 0.95 \
  --flow_latent_cache_mode online --flow_v2_hidden_dim 256 \
  --flow_v2_num_heads 4 --flow_v2_num_layers 2 --flow_v2_mlp_ratio 4 \
  --aux_dataset_config '<MOLMOACT_AUX_DATASET_CONFIG>' \
  --dataset_entries stage05_tabletop_mixed --dataset_sample_ratios 1 \
  --window_size 1 --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 --peak_learning_rate 2e-5 \
  --lr_scheduler cosine --warmup_ratio 0.08 --min_lr_rate 0.1 \
  --epochs 16 --max_train_steps '<CONTROLLED_SMOKE_STEPS>' \
  --save_step_interval '<SMOKE_SAVE_INTERVAL>' --save_optimizer_and_lr_states \
  --output_ckpt_dir '<STAGE2_OUTPUT>' --tensorboard_log_dir '<STAGE2_OUTPUT>/tensorboard' \
  --wandb_project '<WANDB_PROJECT>' --wandb_group '<WANDB_GROUP>' \
  --wandb_run_name '<WANDB_RUN>' --wandb_failure_policy required \
  --tune_vlm
```

This controlled-smoke template uses online targets and probes the measured shape
on each rank device. Use strict mode only after every sample reachable by that
training index has a verified cache entry; then add the independent cache dir
and `--flow_latent_shape 48 14 14`, plus the builder report's explicit
`--flow_latent_cache_manifest_sha256 '<LOCKED_CACHE_MANIFEST_SHA256>'`. A bounded eight-entry cache cannot back a
full shuffled dataset and strict mode will correctly fail on the first missing
entry. Stage 3 replaces init with the complete Stage-2 checkpoint, uses
`--training_stage stage3_joint`, positive AR/FM weights, `--tune_action_expert`,
and the existing explicit Action Expert initialization source. Same-stage resume
uses `--resume_from_checkpoint`, restores the complete optimizer/scheduler/RNG
state and retains identical V2 protocol fields. A V1 checkpoint can initialize
V2 only across stages with an explicit V2 type; it cannot be a V2 resume.

The official cache writer publishes `cache_manifest.v1.json` only after all
requested entries have been atomically installed and revalidated. Each manifest
record locks SHA256 over canonical `(dtype, shape, contiguous latent bytes)`, the
source identity and target protocol. Strict mode requires the manifest file's
SHA256 in configuration. Each process parses and validates the manifest once,
then uses its entry index while the complete file stat identity is unchanged.
A stat change triggers a fresh locked-SHA check and full parse; stat is only a
change detector, not a content digest. The selected latent tensor is still loaded
and content-hashed on every read. A different manifest is rejected and never
adopted.

Before the Stage06 reader, cache writer or calibration consumes a Flow HDF5, its
actual content identity must match the manifest SHA256. A process-local record
binds that result to the expected digest and complete file stat identity, so
stable files are not rehashed per sample. A stat change closes any reader handle,
clears its row index and revalidates content before reopening. Flow HDF5 and cache
manifests are immutable during training; replacing either file is unsupported
even when a previous handle exists.

V2 cache and checkpoint metadata bind the full target protocol, VAE config hash,
VAE weight-content hash, normalization, color scale/directions, measured latent
shape, filter rules, cache version/manifest fingerprint, Adapter configuration and saved
weights. Strict read never falls back to online encoding. Old or damaged entries
must be rebuilt explicitly; existing cache files are never silently overwritten.
