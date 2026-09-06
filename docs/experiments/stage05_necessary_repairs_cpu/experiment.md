# Stage05 Necessary Repairs CPU Smoke

- Created: 2026-09-05, Asia/Shanghai. Owner: lq.
- Purpose: exercise AR fresh -> production AR save -> AR-to-Joint fresh load ->
  Joint fresh, with exactly one CPU optimizer step per phase.
- Baseline: `1f88937a1375eb86b5e78c759815301b67524882`, plus the pre-existing
  staged worktree and this task's checkpoint/entry-point repairs. No commit.
- Entry: `tests/stage05_cpu_smoke.py`; result and checkpoints are written under
  the explicitly supplied, new `outputs/verification/` directory.
- Initial model: existing tiny Qwen3-VL image-test configuration, random base
  initialization seed 13, hidden size 32, two text layers, one visual layer.
  The local Qwen3-VL-2B processor/tokenizer is reused, without loading 2B weights.
- Action Expert: existing production config with hidden size 32, MLP 16,
  4 x 8 attention heads, two DiT layers, output 16, dropout zero for this CPU
  smoke only. State/action padding 64 and H=32 remain unchanged.
- Training seed: 42 at each fresh stage. AR has no Action Expert; Joint loads
  VLM and all 32 Queries exactly and initializes the real Action Expert fresh.
- Trainable: visual encoder, visual merger, text decoder, Difference Query;
  Joint additionally trains the state/action encoders, position embedding, DiT
  and action decoder. No LoRA, Slot or Optical Flow module.
- Data: the first admitted sample from each AR/Joint production Tabletop
  Stage05 adapter, using its configured source path and sidecars in
  `dataset2feature.yaml`. Only that sample's source episode is read. The other
  three dataset specifications supply the production four-source manifest;
  they do not supply training samples to this two-step check.
- Dataset: `molmoact_dataset_tabletop-v3_stage05`, metadata v3.0,
  `/opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05`;
  source contains 310,743 frames / 1,881 episodes. Sidecars are the existing
  `outputs/stage05_four_dataset_pretraining_20260904/data_sidecars_v7/{ar,joint}/tabletop`.
- Split/scale: existing train eligibility, no validation/test split; one sample
  per phase, no epoch sweep. No changes to sampling, canonical representation,
  q01/q99, temporal or dimension masking. Native data FPS 10.
- Images: real first_view and wrist_image, both source metadata 640 x 480
  (width x height), processed independently in that order to 224 x 224 RGB,
  bicubic, without preserving aspect ratio or applying crop/pad/augmentation.
  Rescale by 1/255, then mean/std both [0.5, 0.5, 0.5]. Both phases measured
  image_grid_thw [[1, 14, 14], [1, 14, 14]], patch size 16. No evaluation images.
- Actions: canonical 7D EEF deltas and gripper openness, padded to 64, H=32;
  AR does not request actions/stats; Joint uses production per-dataset q01/q99.
  No inference execution horizon or rollout applies.
- Device: CPU, GPU count zero, one process, batch 1, accumulation 1;
  effective batch = 1 CPU process x 1 sample x 1 accumulation = 1.
- Precision: FP32 CPU optimization, Qwen gradient checkpointing enabled;
  serialize the VLM in BF16 to match the production checkpoint loader's dtype.
- Optimizer: existing single-group AdamW, LR 1e-5, betas (0.9, 0.95), eps 1e-8,
  weight decay 0.01. Constant scheduler, minimum LR 1e-5, zero warmup, no clipping.
  Each stage starts with empty optimizer state; no scheduler/state transfer.
- Loss: AR = 1 * AR_loss; Joint = 1 * AR_loss + 5 * FM_loss. The production
  optimizer-window reduction/backward is used unchanged. Disabled FM in AR
  has no module, forward call or loss graph.
- Save/log: after each of the two steps. No validation, early stopping, W&B run,
  group, project or URL; this is a local CPU verification, not formal training.
- Scope limit: this checks production model/data/loss/save/load calls with a
  tiny model, not the full DeepSpeed train() loop or a GPU resume.
  The harness disables only Accelerate's optional DeepSpeed discovery during
  CPU model unwrapping: installed Triton otherwise fails without a GPU driver.

```bash
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:lerobot \
  /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python tests/stage05_cpu_smoke.py \
  --output-dir outputs/verification/stage05_necessary_repairs_cpu_20260905_run2
```

Actual result: passed, 2026-09-05 21:26:32 to 21:26:47 Asia/Shanghai
(model/result artifact timestamps). Exactly two CPU optimizer steps completed:

| Phase | Steps | AR loss | FM loss | Total loss |
| --- | --- | --- | --- | --- |
| AR fresh | 1 | 11.9112071599 | disabled | 11.9112071599 |
| Joint fresh | 1 | 11.9110835825 | 1.2347628730 | 18.0848979473 |

Both phases consumed sample_global_index=0, episode=0, frame=0 and updated
Difference Query parameters. The Joint model restored the saved VLM and Query
weights exactly, initialized its Expert fresh, and started with empty optimizer
state. Final artifacts are `ar-checkpoint`, `joint-checkpoint` and `result.json`
inside the output directory in the command above. The checkpoints are model
exports; this smoke does not generate DeepSpeed resume state. AR resume state
validation is covered separately by the focused synthetic ZeRO-2 tests.

No NaN, OOM, checkpoint resume, validation metric, W&B run or rollout occurred.
With one step per stage, each reported loss is also that stage's minimum observed
training loss. No GPU or complete dataset scan was run. The only execution
adaptation was the local optional-dependency guard described above.

Initial attempt (`stage05_necessary_repairs_cpu_20260905`, 21:24 Asia/Shanghai)
stopped before any optimizer step: optional DeepSpeed/Triton import during
Transformers save_pretrained raised `0 active drivers`. No AR checkpoint or
training update was produced. The second attempt uses the CPU-only unwrapping
guard above.
