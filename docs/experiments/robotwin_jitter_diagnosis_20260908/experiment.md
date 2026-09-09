# RoboTwin checkpoint jitter diagnosis

- Created: 2026-09-08, Asia/Shanghai; owner: lq; investigation: Codex.
- Purpose: diagnose severe motion jitter in the ongoing remote evaluation.
- Baseline commit: 3bad663, with existing staged/unstaged changes; production
  model code is unchanged by this diagnostic. The diagnostic script is new.
- Checkpoint: `/opt/data/private/lq/models/ZR-0-RoboTwin2.0-Aloha-AgileX`.
- Training experiment: externally trained official checkpoint; run ID unavailable.
- Source data: `/opt/data/private/lq/datasets/lerobot/robotwin_unified`, LeRobot v3,
  27,500 episodes, 6,075,103 frames; train split. No training is performed.
- Statistics: `demo_data/robotwin2.0-aloha-agilex/meta/stats.json`, locally
  reconstructed q01/q99; exact checkpoint equivalence remains unverified.
- Sample selection: episodes 0, 1, 550, 1100; three evenly spaced positions
  before the last 16 frames. These samples are diagnostic, not a validation set.
- Inputs: current head/left-wrist/right-wrist RGB frames, 14D state and instruction.
  Video timestamps use the v3 episode metadata offsets, decoded via PyAV.
- Image processing: source 640x480; current production Qwen helper requests
  224x224 per view, PIL bicubic resize through qwen-vl-utils, no crop/pad or data
  augmentation; actual image_grid_thw is recorded. AutoProcessor rescales by
  1/255 and uses checkpoint mean/std `[0.5, 0.5, 0.5]`. Model dimensions are
  checked in output.
- Actions: absolute joint targets, left 6 joints + gripper, right 6 + gripper;
  state/action quantile min-max normalization, zero padding to 64; horizon 16,
  five denoising steps, BF16. Dataset FPS: 30. There is no simulator control loop
  or success-rate measurement in this offline diagnostic.
- Comparison: current extra final RMSNorm versus original `hidden_states[-1]`.
  Both use exactly the same VLM pass, state, statistics and per-sample seed.
  Seeds: 42 and 43. Outputs include joint MAE, gripper MAE, first/second temporal
  differences, feature RMS and BF16-versus-FP32 denormalization rounding error.
- Runtime: one independent process on remote GPU 0, A800 80 GB, batch size 1;
  CUDA allocator capped at 12% of device memory; no torch.compile. The ongoing
  evaluation's model state, random generators and observation buffers are not used.
- Output: `outputs/diagnostics/robotwin_jitter_20260908/` (ignored artifacts).
- W&B, optimizer, gradients, loss/backpropagation and training stages: N/A.

Command, from `/opt/data/private/lq/ZR-0` on `lq@10.82.1.223:25408`:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -u scripts/diagnose_robotwin_checkpoint.py --checkpoint /opt/data/private/lq/models/ZR-0-RoboTwin2.0-Aloha-AgileX --dataset /opt/data/private/lq/datasets/lerobot/robotwin_unified --stats demo_data/robotwin2.0-aloha-agilex/meta/stats.json --evaluation outputs/evaluations/robotwin_50x2x20_seed0_remote_direct_20260908 --output outputs/diagnostics/robotwin_jitter_20260908 --device cuda:0
```

## Results

The first run completed successfully on remote GPU 0. All 12 samples produced
finite actions in both modes. Actual inputs were three RGB 640x480 views;
all processor grids were `[1, 14, 14]` (224x224 pixels per view).

| Metric, mean over 12 observations x 2 noise seeds | Current extra RMSNorm | Original raw features |
| --- | ---: | ---: |
| Joint MAE against next 16 expert actions, radians | 0.0483793 | 0.0242544 |
| Gripper MAE | 0.00570222 | 0.00355039 |
| Mean absolute second joint difference, radians | 0.0246559 | 0.0131358 |

Raw feature RMS was 7.01-7.54, reduced to 2.61-2.66 by the extra RMSNorm.
Runtime hooks confirmed that the original final hidden tensor is exactly the
input to Qwen's final text norm under Transformers 4.57.1. The current code
feeds its normalized value into the unchanged legacy Action Expert.

This change entered in commit `166193d7d50c18080f954d8771b8a8a6b6927bb3`, for
local training stability. Original repository commit `b1440d4` and upstream
`https://github.com/RUCKBReasoning/ZR-0/blob/main/model/qwen_vl_backbone.py`
use `hidden_states[-1]` directly. The current direct-action path applies the
extra norm unconditionally, including this external legacy checkpoint.

The paired results establish a detrimental conditioning change on these
demonstrations; they do not establish that it explains every failed rollout.
No corrected closed-loop simulation has been run in this diagnostic.

The first run also measured up to 0.01983 rad rounding error when comparing
current BF16 denormalization with the same normalized outputs denormalized in
FP32. This behavior is inherited from upstream, and is separate from the
new extra RMSNorm. A second bounded run records FP32 denormalization temporal
metrics for episodes 0 and 550, reusing all other settings:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -u scripts/diagnose_robotwin_checkpoint.py --checkpoint /opt/data/private/lq/models/ZR-0-RoboTwin2.0-Aloha-AgileX --dataset /opt/data/private/lq/datasets/lerobot/robotwin_unified --stats demo_data/robotwin2.0-aloha-agilex/meta/stats.json --evaluation outputs/evaluations/robotwin_50x2x20_seed0_remote_direct_20260908 --output outputs/diagnostics/robotwin_jitter_20260908_fp32_denorm --episodes 0 550 --device cuda:0
```

Official training-statistics equivalence remains unverified. Production
checkpoint, server configuration and model implementation were not changed.

The second run also completed with finite outputs throughout. For current
conditioning, FP32 denormalization changed the mean second difference from
0.0249615 to 0.0244463 rad (2.06% lower); with original raw conditioning, it
changed 0.0128290 to 0.0121040 rad (5.65% lower). Joint MAE was essentially
unchanged. Thus BF16 denormalization is a secondary contributor in these tests,
and switching only that operation to FP32 does not address the principal
conditioning mismatch. Model peak allocated VRAM was 5.08 GiB; both diagnostics
exited with status 0 and did not encounter OOM or nonfinite predictions.
Result files were finalized at 2026-09-08 16:03:38 +08:00 (first run) and
16:07:02 +08:00 (precision comparison). Saved arrays were independently checked
for shape `(16, 14)`, finite values and agreement with reported joint MAE;
all 72 predictions passed. Python AST parsing and `git diff --check` passed.

The first comparison has 48 predictions (12 observations x 2 seeds x 2 modes),
the second 24 predictions (6 observations x 2 seeds x 2 modes). No model weights
were trained, saved, overwritten or resumed. The first run's conditioning plot
is `outputs/diagnostics/robotwin_jitter_20260908/conditioning_comparison.png`.
