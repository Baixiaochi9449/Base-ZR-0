# LIBERO Fine-Tuning From Stage 3 Step 14000

- Created: 2026-09-10, Asia/Shanghai. Owner: lq. Explicit user authorization: fine-tune LIBERO from the saved Stage 3 step 14000 using the previous LIBERO experiment configuration.
- Experiment: `libero-stage3-step14000-action-only-dq32-seed42`.
- Reference: `outputs/ckpts/Qwen3-VL-2B-Instruct-LIBERO-wo-ECoT-PT-FinalRMSNorm-difference-query-nq32`.
- Source checkpoint: `/opt/data/private/lq/ZR-0/outputs/three_stage_formal_20260910/stage3_resume8000_slot05_flow5/recovery_checkpoints/stage3_joint/step-014000-attempt-000/latest-model-optimizer-lr`.
- Source stop completed at 2026-09-10 19:15:45 +08:00: 39-file full checkpoint preserved, last logged step 14000, old ranks/workers/supervisors stopped and retries disabled. An unlogged in-flight update after saving cannot be excluded; only the immutable step-14000 weights initialize this experiment.
- Base ancestry: `/opt/data/private/lq/models/ZR-0`; no component is reloaded from base during fine-tuning.
- Training/model/data runtime: existing `outputs/runtime_snapshots/3eefb602417d3bd4b20bef2b47660b404aefb565`. Current working-tree Flow V2 modifications are not imported. The current launcher and optional verification/archive wrapper are recorded separately.
- Config: `accelerate_configs/libero_zero2_bf16_mbs16_gas1.yaml`; all parsed defaults, source weight SHA256, dataset/statistics identity, code status and expanded command are saved in `launch_manifest_train.json`.
- Output root: `outputs/ckpts/ZR0-stage3-step14000-LIBERO-action-only-dq32-h10-gbs64-seed42`; each retry has a separate output directory and command record.
- W&B: online required, entity `jumbo3r-zhejiang-university`, project `ZR-0-LIBERO`, group `libero-stage3-step14000-action-only-dq32-seed42`. Run name/ID and URL are recorded at launch. Connection failures do not switch to offline mode.

## Initialization And Objective

One action-only downstream stage, `checkpoint_load_purpose=downstream_finetune`, no `training_stage`. VLM and Expert paths both point to step 14000. Fresh initialization does not load pretraining optimizer/scheduler/global step.

Train the complete inherited VLM vision encoder, merger/projector, DeepStack, language decoder, all 32 Difference Queries and the entire inherited Action Expert (state/action/timestep encoders, 9-layer DiT, decoder and positional embedding). No LoRA, frozen component or detached Query conditioning. The LM head is not invoked for action-only CE. Slot and Optical Flow Heads are absent. All 32 Queries condition the Expert without auxiliary role partitioning during fine-tuning.

`L = 1.0 * L_FM`; AR, Slot and Optical Flow outer coefficients are explicitly zero and those losses are absent from the graph. FM is the existing masked velocity regression along the noise/action interpolation, with existing Beta(1.5,1.0) time sampling and global valid-action-element reduction. There is no new loss implementation.

The entire inherited VLM/Query/Expert key sets and tensors are compared exactly after final dtype conversion and ZeRO-2 preparation, before the first optimizer update. The optimizer must start with one empty AdamW parameter group; fresh engine/scheduler counters must be zero. Retry verification compares against the selected LIBERO checkpoint and restored global scheduler position.

## Data And Images

- Dataset: LeRobot v2.1 `libero_wo_ecot_pt`, `/opt/data/private/lq/datasets/HuggingFaceVLA/libero`; 1,693 episodes, 273,465 frames, 40 tasks, 10 FPS. All supplied frames are train; no validation/test loader.
- Sample ratio 1.0, deterministic shuffle with seed 42 plus epoch. Each epoch has 273,468 distributed presentations (three deterministic duplicates), 68,367 samples/rank, 4,273 updates. The final global batch is 60, without dropping or padding it to 64. Eight epochs present 2,187,744 samples including 24 duplicates.
- Inputs: current task language, current state and two RGB views in order `observation.images.image`, `observation.images.image2`. No ECoT, future image, Slot or Optical Flow labels.
- State/action effective dimensions 8/7; model padding dimension 64. Use LIBERO's own q01/q99 training statistics and existing quantile min-max normalization; clipping is `[-15,15]`, not `[-1,1]`. Constant dimensions and invalid/padded elements retain the existing handling. Pretraining dataset statistics are not reused.
- Original images: 256 x 256 RGB per view. Existing Qwen image preparation resizes to 224 x 224 with bicubic interpolation, no aspect preservation, crop, letterbox or augmentation. Processor `do_resize=False` after explicit resize; rescale 1/255, mean/std each `[0.5,0.5,0.5]`.
- Patch size 16, spatial merge 2, 49 visual tokens per view; both current views form context, window 1. Maximum text sequence length 1200.
- No new pretraining payload audit. Existing source audit is retained; this launch records checkpoint and LIBERO metadata/statistics identities once. Per-launch GPU/resource checks and checkpoint loading checks remain required.

## Optimizer And Scale

- Four NVIDIA A800-SXM4-80GB GPUs, UUID resource gate before every attempt. `global_batch_size = 4 * 16 * 1 = 64` nominal; micro-batch 16, GAS 1.
- Budget: 8 epochs, 34,184 successful optimizer updates. Scheduler resets for this downstream experiment. Warmup 2,734 updates (8%), then cosine from peak 2e-5 to minimum 2e-6.
- One AdamW parameter group for all trainable modules, betas `(0.9,0.95)`, epsilon 1e-6, weight decay 0.01, no LR multipliers. Legacy Accelerate scheduler uses four internal scheduler ticks per global update; internal total/warmup are 136,736/10,936.
- BF16, ZeRO-2, no offload, SDPA, VLM gradient checkpointing enabled; Action Expert uses its existing activation behavior. Clip global gradient norm 1.0. Seed 42.
- DataLoader: 24 workers/rank, prefetch factor 3. Existing PyAV resource handling.
- Logging every 10 updates to stdout, JSONL, TensorBoard and online W&B. Startup verification uses updates 1 through 20 of the formal budget; no separate smoke optimizer updates.
- Save every 2,000 updates, at epochs 4/8, and final step. Model directories `step-N`; full ZeRO-2 state `latest-model-optimizer-lr`; immutable full copies `recovery_checkpoints/step-N/latest-model-optimizer-lr`.
- No validation interval, rollout or early stopping. Do not automatically start another training experiment after completion.

## Action And Evaluation Contract

Action/state padding dimensions 64, effective action/state dimensions 7/8. Existing LIBERO end-effector delta position, rotation and gripper encoding is unchanged. Action chunk and prediction horizon are 10; episode-tail time padding and 57 padded action dimensions are masked. Expert conditioning dimension 2048, output dimension 1024, 32 heads of dimension 64, 9 DiT layers, dropout 0.2, MLP hidden 512, positional embedding `[256,2048]`. Source and runtime horizon are both 10, so no horizon override is applied.

No rollout is part of this training launch. Execution horizon, action execution frequency, number of rollouts and success rate must be specified in a future evaluation record; they cannot be inferred from prediction horizon. Evaluation image preprocessing must be recorded and compared with the above training settings.

## Commands And Recovery

```bash
cd /opt/data/private/lq/ZR-0
source /opt/data/private/lq/miniconda3/etc/profile.d/conda.sh
conda activate ZR-0
python scripts/watch_libero_finetune.py --execute
```

The supervisor launches `bash scripts/run_libero_wo_ecot_pt.sh train difference_query_stage3`, records the exact environment-independent expanded command before execution, and may retry at most three times after delays 60/120/240 seconds. Retries retain model, dataset, batch, losses and total budget; only a complete checkpoint from this LIBERO experiment is eligible. A failure after training starts but before any own complete checkpoint stops recovery instead of resetting to pretraining weights. Every retry has separate logs and experiment documentation. W&B is required and a disabled-retry marker stops future launches. This preserves the legacy LIBERO model/optimizer/scheduler and inferred data-position resume; rank RNG and exact trajectory equivalence are not claimed for that legacy path.

## Results

- Attempt 000 launched at 2026-09-10 21:57:06 +08:00; four ranks 1919886-1919889 train under launcher 1918747 and supervisor 1918552. W&B initialized online: https://wandb.ai/jumbo3r-zhejiang-university/ZR-0-LIBERO/runs/cc7a748a . Run ID `cc7a748a`; run name `zr0-step14000-libero-dq32-seed42-20260910-215700`.
- Startup evidence passed at 22:04:25 +08:00: `startup_verified.json` and `attempt-000/libero_initialization_verified.json` under the experiment root. VLM 626, Query 1 and Expert 149 tensors match source exactly after BF16/ZeRO-2 preparation; all 2,690,580,544 included parameters are trainable. Fresh engine/scheduler counters were 0/0, scheduler step count 1, and the newly constructed AdamW state was empty. Existing warmup lambda uses `(current_step+1)/warmup`, so initial LR is 1.82882223847842e-9.
- Formal startup updates 1/10/20 logged raw FM/total losses 0.585717/0.599540/0.417004 (order 1e-1); no auxiliary or AR losses were computed. At step 20, VLM/Query/Expert gradient norms were 2.71446/0.282161/2.04206. No reported skips, nonfinite values, input truncation or W&B failures. Peak allocated/reserved memory 18.1028/23.1406 GiB; mean logged optimizer-window time at steps 10/20 was 0.85264 s, excluding input fetching and logging.
- Continued observation reached step 230 with FM loss 0.278669, finite gradients and online W&B. Training remains running toward 34184; no end time or final checkpoint yet. First own checkpoint is due at step 2000. Epoch boundaries, real checkpoint/resume and rollout evaluation have not been observed in this startup check.
- Source archive inventory and runtime/entry hashes remained unchanged. Dataset manifest SHA256 `a3c74b5acfdf414effc33c4f832dee534791bd11f51dba61ef950adce9f3cd54`; LIBERO `dataset_meta.stats` SHA256 `aafe658c89aad69f2b6db5a7109d447ff72040e7ab13c1f0938a2e6237e5942d`; exact q01/q99 values and model shard hashes are in `attempt-000/launch_manifest_train.json`.
- CPU verification: 31 distinct cases passed across the existing launcher suite and new Stage 3 tests. The initial launcher suite had 22 passes and one fake-interpreter regression; preserving the old arms' interpreter dispatch fixed it, and its targeted rerun passed. Eight Stage 3 tests and the pinned parser contract check passed; bash syntax, Python compilation and diff whitespace checks passed. No GPU smoke updates or pretraining payload re-audit were performed. Source/config/test/docs files from this task are staged; no commit was created.
