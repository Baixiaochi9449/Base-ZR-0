# Stage05 Four-Dataset Difference Query Pretraining

## Basic Information

- Experiment: `stage05_four_dataset_pretraining_20260904`
- Purpose: two-stage AR-only then Joint pretraining over four Stage05 robot datasets.
- Created: 2026-09-04 14:51 Asia/Shanghai
- Owner: lq
- Base model: `/opt/data/private/lq/models/Qwen3-VL-2B-Instruct`
- Initialization: AR starts from the base model and random DQ32; Joint starts from the AR 100-step pilot VLM+Query and initializes Action Expert from seed 42.
- Code baseline: Git `1f88937a1375eb86b5e78c759815301b67524882`; the final handoff retains uncommitted/staged task changes and does not create a commit.
- Dataset config: `dataset2feature.yaml`, entries `stage05_{droid,household,tabletop,rh20t}_mixed`.
- Action Expert config: `configs/stage05_four_dataset_action_expert.json`, explicitly passed as `--action_expert_config_path`. Its authority is the completed Tabletop Joint checkpoint `outputs/pretrain/tabletop_v3_dq32_joint_gbs128_seed42_mbs16_gas2/step-2428/action_expert_config.json` (source SHA256 `0d0ea06180304f3d2a226c3f13ff1b43f12cb8c75ce2b971a5dfcf6286fea196`); the tracked minimal JSON has source SHA256 `aa70464c0f7008a2154ecd9deb117a4a11995dfcf89ca094f27bb8a30ac63a9b` and fully parsed canonical hash `e014799a51284ac731fae4018343e3f6480f65bab84bfd85205215271c06437f`.
- Launcher: `scripts/run_stage05_four_dataset_pretraining.sh`.
- Seed: 42.
- Output root: `outputs/stage05_four_dataset_pretraining_20260904`.
- W&B: planned project/group/run are supplied through `ZR0_WANDB_PROJECT`, `ZR0_WANDB_GROUP`, and `ZR0_WANDB_RUN_NAME`; no run or URL exists because GPU admission failed before training.
- Authority: local `training_metrics.jsonl`, TensorBoard, copied experiment document, resolved manifest, initialization manifest, and checkpoint. W&B uses `best_effort` and cannot stop smoke, pilot, or a later formal run when initialization/network logging fails.

The detailed, immutable data result is in `data_admission_report.md`. There is no pretraining validation split. Fixed read-only audit samples remain part of training eligibility and are not a generalization validation set.

## Datasets

| Source | Version/path | Eligible subset | Native FPS | Ordered views |
|---|---|---|---:|---|
| DROID | `/opt/data/private/lq/datasets/droid_1.0.1_stage05_full_95658_20260831` | AR: Stage05+trusted task+main; Joint: successful data gate and FM_count>0 | 15 | exterior1, wrist |
| Household | `/opt/data/private/lq/datasets/molmoact_dataset_household-v3_stage05` | all admitted annotation frames; Joint FM_count>0 | 10 | first_view, wrist_image |
| Tabletop | `/opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05` | all admitted annotation frames; Joint FM_count>0 | 10 | first_view, wrist_image |
| RH20T | `/opt/data/private/lq/datasets/RH20T-v30_stage05` | explicit success only; excludes unrated/failed/4 anomalies; Joint FM_count>0 with action.valid | 10 | exterior1, wrist if abs(skew)<=100 ms |

AR natural probabilities are DROID/Household/Tabletop/RH20T = 0.6732477314/0.0562389042/0.0220043664/0.2485089980. Joint probabilities are 0.8114236697/0.0325094778/0.0127198506/0.1433470019, strictly from final Joint action-eligible frames. No generic VQA/VL data is mixed in. This differs from the original ZR-0 pretraining protocol and introduces a known VLM catastrophic-forgetting risk.

`slot_data` is retained as audit metadata only. It does not determine eligibility and has no loss. No validation/test partition is used for this pretraining run.

## Training Stages

### AR-only

- Purpose: train complete `train_data` plus the assistant termination token with DQ32 conditioning semantics.
- Initialization: Qwen3-VL-2B base model; 32 Difference Queries randomly initialized under seed 42.
- Trainable: visual encoder, visual merger/projector, language model, and Difference Query parameters (`--tune_vlm`).
- Frozen/absent: Action Expert is not constructed, loaded, executed, or saved. There is no Action Expert optimizer state.
- Loss: AR only. `total_loss = 1.0 * AR_loss`; valid target-token global sum divided by valid target-token count over every micro-batch and all four ranks in one optimizer step.
- Checkpoint: VLM weights, Difference Query weights/config, optimizer/scheduler/global step, data and seen manifests, plus a versioned future-Joint contract marked `not_constructed_future_joint_config_reference`. Metadata records `checkpoint_kind=ar_only`, checkpoint metadata version 1 and contract/config schema version 1. The contract contains the complete parsed Action Expert config, exact original file bytes and SHA-256, canonical config SHA-256, VLM hidden size, DQ count, action/state dimensions, H=32, the associated resolved experiment-manifest hash, and its own canonical content hash. It contains no Action Expert tensor or optimizer state and is not described as a trained Expert.

### Joint

- Purpose: train `AR + 5*FM` on every action-eligible sample.
- Initialization: load only VLM+Query from the completed AR pilot checkpoint. Before model construction, the Stage05 launcher requires `checkpoint_kind=ar_only`, compatible metadata/contract versions, no Expert weights, intact config raw/canonical/contract hashes, matching VLM/DQ/action/state/H fields, and an intact associated resolved manifest. It then requires the explicit tracked external Action Expert config to have the identical full canonical hash before creating the Expert under a fresh seed-42 initialization. Missing, empty, damaged or incomplete config, any structural drift (including layers/heads/dropout/position fields), or missing hashes are fatal; `{}` and class-default fallback are forbidden. Legacy Tabletop AR checkpoints without this Stage05 contract are rejected by this launcher with an explicit compatibility error, while generic old-checkpoint loading remains unchanged.
- Trainable: visual encoder, visual merger/projector, language model, Difference Query, Action Expert state/action encoders, DiT/cross-attention, and action decoder (`--tune_vlm --tune_action_expert`).
- Frozen: no requested model module is frozen.
- Loss: `total_loss = 1.0 * AR_loss + 5.0 * FM_loss`; both weights remain CLI configurable. AR uses the optimizer-step global valid-token denominator; FM uses the optimizer-step global valid-action-element denominator. A textless Joint sample has labels all `-100` and contributes exactly zero AR numerator/count/gradient while retaining FM. Every Joint sample has `FM_count > 0`.
- Conditioning: Action Expert key/value receives only final RMSNorm Query hidden `[B,32,H]`; it may receive proprioception, noisy action, and flow timestep directly, but never C or teacher-forced T.
- Checkpoint/resume: Joint saves/restores the full VLM+Query+Action Expert, optimizer, scheduler, global step, resolved data/stats/filter hashes, and dataset-seen state.

Both stages preserve `[C,Q,T,P]`: Q reads all C and bidirectional Q but no T/P; T reads all Q and causal T but no C/P/future T. Query mode is SDPA. `.generate()` restrictions are unchanged.

## Optimizer And Schedule

Both stages use one AdamW parameter group with beta `(0.9,0.95)`, epsilon `1e-8`, weight decay `0.01`, peak LR `1e-5`, minimum LR `1e-6` (`min_lr_rate=0.1`), 5% linear warmup followed by cosine decay, and gradient clipping `1.0`. BF16 and gradient checkpointing are enabled. There are no module-specific LR multipliers.

The `1e-5/1e-6` schedule is this project's conservative pilot configuration. It must not be represented as the original ZR-0 paper's `3e-5/3e-6` configuration.

## Training Scale

- Hardware: requested GPU IDs 0,1,2,3; NVIDIA A800-SXM4-80GB, four ranks.
- Precision/backend: BF16, SDPA, gradient checkpointing.
- Horizon: action/prediction horizon 32; execution horizon is not applicable to pretraining and is not changed.
- Candidate micro-batch/GAS from the prior Tabletop-only stable run: 16/2. It is not yet accepted for the four-dataset mixture.
- Required equation: `global_batch_size = 4 * per_device_batch_size * gradient_accumulation_steps = 128`.
- Final micro-batch/GAS: pending a non-contending maximum-stable-micro-batch probe. Both must be explicit integers and match CLI, Accelerate and DeepSpeed; `auto` is forbidden.
- Smoke: 2 fresh optimizer steps, save, resume to step 3.
- Pilot: 100 optimizer steps for AR, then 100 for Joint.
- Formal: step count undecided; launcher stays locked unless the user supplies an explicit count and approval variable.
- Checkpoint interval: final step of each smoke/pilot invocation; formal interval must be decided with formal step count.
- Log interval: every optimizer step for smoke/pilot.
- Validation interval and early stopping: none, because this run intentionally has no pretraining validation split.

At the 2026-09-04 GPU gate, free memory on GPU 0/1/2/3 was 9,543/19,519/8,319/36,613 MiB and utilization was 100/100/74/92%, with pre-existing workloads on every device. The gate is NO-GO, so the probe and every training stage remain unstarted.

## Image Processing

Molmo reads only `first_view` and `wrist_image`; DROID/RH20T read only `observation.images.exterior_1_left` and `observation.images.wrist_left`. `second_view/exterior2` is never projected, inventoried, decoded, or sent to the processor. A sample contains main then wrist; RH20T skew over 100 ms degrades to main-only and never substitutes a second exterior view.

Raw images are RGB: Molmo embedded images are 640x480; audited videos are 320x180. The current Qwen processor resizes each image directly to 224x224 with bicubic interpolation, without aspect-ratio preservation, crop, pad, letterbox, or augmentation. It rescales by 1/255 and normalizes with mean/std `[0.5,0.5,0.5]`. Patch/temporal/merge sizes are 16/2/2. The pinned processor produces 49 visual tokens per view (49/98 total for one/two views); the all-eligible-frame audit found the exact minimum zero-truncation `max_length=941`. Token audit format v2 binds that result to each AR sidecar format/manifest/generator/eligible-index hash and count, the resolved dataset entry/stats key, all local tokenizer/chat-template/processor configs, processor classes and package versions, runtime processor source files, and the repository message/tokenization call chain. The complete rerun is `audits/token_length_audit_v9_format2.json`; its content/file/implementation hashes are fixed in `configs/stage05_four_dataset_experiment.json`. The launcher accepts no legacy report or constant fallback and invokes the audit script's `--validate-only` path before allocation. Training and later inference use the same adapter contract and still retain per-sample zero-truncation checks.

## Action And Normalization

Canonical state is absolute base/world EEF `[x_m,y_m,z_m,roll_rad,pitch_rad,yaw_rad,gripper_open]`. Canonical action is native-next-step relative EEF `[dx_m,dy_m,dz_m,droll_rad,dpitch_rad,dyaw_rad,gripper_open]`. Rotation is SciPy lowercase `xyz` RPY in radians with wrapped component deltas; translation uses metres; gripper uses `1=open`.

Each dataset owns a separate q01/q99 `stats_key`. Filtering and canonical conversion happen before statistics; padding, episode-tail steps, invalid dimensions and `action.valid=false` never enter q01/q99. State/action pad dimension is 64, real dimension is 7, and H=32 temporal/dimension masks select valid FM elements. Inference must explicitly select the matching `stats_key`; mismatch is fatal.

The existing production normalization clip remains `[-15,15]`. A production-path audit found no clipping anywhere in the full DROID/Household/Tabletop valid-action populations. RH20T was audited over all 3,501,909 valid action steps and clipped `[8,3,3,4,4,5,0]` elements by dimension; the worst irreversible denormalization error is `0.2932883` at episode 5580/frame 742/dimension 4. Mathematical in-range formula checks are reported separately from these real clipped-data results in `data_admission_report.md`.

The complete parsed Action Expert structural configuration is:

```json
{
  "add_pos_embed": true,
  "vlm_output_embedding_dim": 2048,
  "action_or_state_token_embedding_dim": 2048,
  "mlp_hidden_size": 512,
  "max_seq_len": 256,
  "action_dim": 64,
  "state_dim": 64,
  "action_horizon": 32,
  "noise_beta_alpha": 1.5,
  "noise_beta_beta": 1.0,
  "noise_s": 0.999,
  "num_timestep_buckets": 1000,
  "diffusion_transformer_cfg": {
    "num_attention_heads": 32,
    "attention_head_dim": 64,
    "output_dim": 1024,
    "num_layers": 9,
    "dropout": 0.2,
    "attention_bias": true,
    "activation_fn": "gelu-approximate",
    "upcast_attention": false,
    "norm_type": "ada_norm",
    "norm_elementwise_affine": false,
    "norm_eps": 1e-05,
    "max_num_positional_embeddings": 128,
    "positional_embeddings": null,
    "final_dropout": true,
    "interleave_self_attention": true,
    "causal_mask_in_self_attn": false
  }
}
```

## Logging And Commands

The launcher writes the complete command to `train.log`, copies this document into the run directory, and writes local JSONL/TensorBoard/manifests/checkpoints. W&B metadata is supplied explicitly when desired, but `--wandb_failure_policy best_effort` guarantees W&B is not a single point of failure. Remote payloads use a configurable bounded FIFO (default 256); overflow drops the oldest remote-only payload, while local JSONL/TensorBoard remain complete. Consecutive failures use step-based exponential backoff from 1 to at most 128 steps. `finish()` makes at most two final flush attempts, and pending flush plus final remote finish share a configurable `time.monotonic()` wall-clock deadline (`--wandb_finish_timeout_seconds`, default 15 seconds, finite and positive). A timeout abandons and permanently disables that run; `finish_timed_out`, timeout seconds, pending/drop/failure counters and abandoned state are appended to local JSONL/TensorBoard before local writers close. An isolated fake remote that blocks for 60 seconds returned from `finish()` in 0.150804 seconds with a configured 0.15-second deadline and exited normally. `required` keeps synchronous failure semantics and does not use the best-effort timeout.

The currently identity-bound audited floor is `max_length>=941`; 940 fails before model initialization, while 941 and 1024 pass. Any sidecar, eligible index, tokenizer/chat template/processor config, dependency code or relevant package identity change requires a new complete token audit rather than falling back to 941. These are templates; none were executed because the GPU gate failed:

```bash
export ZR0_MAX_LENGTH=941
export ZR0_PER_DEVICE_BATCH_SIZE=<PROBED_INTEGER>
export ZR0_GRADIENT_ACCUMULATION_STEPS=<128/(4*PROBED_INTEGER)>

scripts/run_stage05_four_dataset_pretraining.sh ar-smoke
scripts/run_stage05_four_dataset_pretraining.sh ar-resume
scripts/run_stage05_four_dataset_pretraining.sh ar-pilot
scripts/run_stage05_four_dataset_pretraining.sh joint-smoke
scripts/run_stage05_four_dataset_pretraining.sh joint-resume
scripts/run_stage05_four_dataset_pretraining.sh joint-pilot
```

Formal AR template, deliberately locked and without a default step count:

```bash
ZR0_MAX_LENGTH=941 \
ZR0_PER_DEVICE_BATCH_SIZE=<PROBED_INTEGER> \
ZR0_GRADIENT_ACCUMULATION_STEPS=<128/(4*PROBED_INTEGER)> \
ZR0_FORMAL_MAX_STEPS=<USER_DECISION> \
ZR0_FORMAL_AR_OUTPUT_DIR=<NEW_OUTPUT_DIR> \
ZR0_WANDB_PROJECT=<PROJECT> ZR0_WANDB_GROUP=<GROUP> ZR0_WANDB_RUN_NAME=<RUN> \
ZR0_ALLOW_FORMAL_TRAINING=1 \
scripts/run_stage05_four_dataset_pretraining.sh ar-formal
```

Formal Joint additionally supplies `ZR0_FORMAL_AR_CHECKPOINT` and uses `joint-formal`. W&B initialization failure is reported and local logging continues; it never silently disables the configured W&B attempt.

## Third-review checkpoint and token contracts

Stage05 checkpoint loading now has an explicit `--checkpoint_load_purpose`: `stage05_ar_to_joint` and `stage05_joint_resume` are the same pretraining experiment and require the complete Stage05 contract with unchanged checkpoint H (this experiment uses H=32); `downstream_finetune` is a new experiment initialization and is the only purpose allowed to override a verified source Expert to an explicitly chosen legal H_ft. `inference` is an artifact-only load purpose. The contract stores architecture and runtime/training hashes, full Expert config, VLM/DQ/dimension/H fields, and the Stage05 manifest hash. AR-only checkpoints remain Expert-free; Joint resume loads Expert weights and optimizer/scheduler state, while downstream creates fresh optimizer/scheduler state.

The production Action Expert was constructed at H=32, H=16, H=10 and H=8 and compared key-by-key and shape-by-shape: all keys and shapes are identical. The verified transitions include 32->10, 16->8, 8->16 and unchanged H, so this is a general explicit H_pre->H_ft fresh-initialization rule rather than a hard-coded 32->10 exception. Every supported H_ft uses `strict=True` with no resize or dropped parameter. H is checked against `max_seq_len=256` and `max_num_positional_embeddings=128`; non-horizon architecture drift still fails. LIBERO downstream uses its own q01/q99 and `stats_key`, emits its configured `[B,H_ft,64]` chunks (the current example is H_ft=10), and saves/restores the declared horizon; it does not inherit Stage05 token or action statistics.

Stage05 token length is bound by the external versioned spec `configs/stage05_four_dataset_experiment.json`, not by an environment override or a fallback constant. The spec fixes audit format v2, required length 941, the report content/file SHA256 and implementation identity SHA256. Because the sidecar identity changed, the complete audit was regenerated as `audits/token_length_audit_v9_format2.json`; the final hashes are recorded in the data admission report and spec. LIBERO is intentionally outside this Stage05 941 gate.

## Actual Result

- Data audit: AR GO; Joint canonical/data GO. Full counts, q01/q99 hashes, camera synchronization, video backend, visual examples, trajectory reconstruction, and source hashes are in `data_admission_report.md`.
- Fourth-review six-Finding production-path tests passed independently: 88 passed. They cover production `vlm_and_action` manifest/save/load, strict legacy Tabletop Joint initialization, direct Python/purpose/resume gates, parseable Expert and training-state resume artifacts, default trusted-spec token validation, inference contract/H tampering, and arbitrary compatible horizons. The complete CPU regression is 383 passed, 3 skipped, with 6 existing third-party/legacy warnings in 393.87 seconds.
- Training start/end: not started.
- Completed optimizer steps: AR 0; Joint 0.
- Checkpoint: none.
- Minimum loss/best validation metric: unavailable; no training and no validation split.
- W&B run/URL: none.
- NaN/OOM/resume: no GPU forward or training was started. CPU production preflight opened and shape-checked safetensors plus scheduler/DeepSpeed/client/optimizer state and global-step consistency; real multi-GPU DeepSpeed resume remains pending.
- Deviation: required GPU resource gate failed due to pre-existing workloads, so single-dataset forward, four-card smoke/resume, and both 100-step pilots stopped before launch as specified. This review additionally used CPU contract/model tests and dry-runs only; no GPU training was started.
