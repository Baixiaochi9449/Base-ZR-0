# Optical Flow Data Audit

Date: 2026-09-06. Source: local Stage06 artifacts and the supplied planning audit.

- Root: `/opt/data/private/lq/datasets/lerobot/libero/stage06_flow/libero_delta10`.
- Authoritative manifest: `manifest.2849ed69240ad542.jsonl`; 1693 episodes,
  273465 source frames. Source and merged episode IDs are identical, 0..1692.
  Join to LIBERO parquet by `episode_index, frame_index`, never HDF5 row offset.
- Camera: `observation.images.image`; original images 256x256, labels 224x224.
  Flow `[N,2,224,224]` float16, mask `[N,1,224,224]` uint8; explicit int64
  `frame_index` and `target_frame_index`, int16 delta, uint8 label source.
- Forward flow, nominal delta 10 at 10 FPS, tail clamp, units
  `normalized_source_image_extent`. Do not divide components by four at 56x56.
- `validity_semantics=finite_and_forward_destination_in_bounds_not_occlusion`.
  These are MegaFlow pseudo-labels, not ground truth or occlusion labels.
- Supervise only `actual_delta_frames == 10` and `label_source == 1`.
  Shortened tail pairs and final identity (`label_source == 2`) are excluded.
- The supplied audit found 1693 complete status/quality reports and batch summary
  1693/1693. Episode P1/P50/P99: valid fraction .973/.989/.993;
  maximum magnitude .140/.212/.404; mean magnitude .0027/.0074/.0196.
- `manifest.a9a4b17704739c1f.jsonl` is empty. The old final-validation log's
  1693 missing result refers to that empty manifest, not the complete manifest.
- Generator external source: MegaFlow revision
  `ee5b61813db0a76ac0db9034899aade72a0d230c`, recorded tree hash
  `2e60993bc8b8d221bc634f87606a943c6d7d5aa234a0f2baa5a92b15610f0820`;
  checkpoint SHA256 `6b8524bdce14f35abeaebdd11a315d28f7845a67d2f03225270243fe84423790`.
  Recorded external code path: `/home/lq/VLA/FD-ID-FlowVLA/Flow-image-generation/megaflow`.
  This task consumes labels and does not reimplement or execute the generator.
- Training uses explicit 224x224 resize hints, processor `do_resize=False`,
  no random crop/flip/rotation. Generator interpolation has not been verified
  identical to the training image helper. This is an external provenance limit.
  The inspected local `qwen_vl_utils.vision_process.fetch_image` uses RGB PIL
  `image.resize((224,224))` (PIL's default bicubic for RGB). No padding/crop;
  256x256 to 224x224 preserves aspect ratio. The actual checkpoint processor
  rescales by 1/255 and normalizes with mean/std `[.5,.5,.5]`/`[.5,.5,.5]`.
- Startup validates every declared HDF5's structure/index/metadata without loading
  dense arrays. Before a production reader, cache writer or calibration reads a
  file, it lazily computes the full HDF5 byte SHA256 once per process/worker and
  compares it with the manifest; stable stat identity reuses that verified result.
  Runtime validates fetched arrays, and a changed/replaced file is rehashed after
  old handles are evicted.

Verification results are recorded in `docs/experiments/optical_flow_cpu/experiment.md`.

Revalidation in this task: all 1693 HDF5 structures and 273465 explicit frame
identities passed startup validation. Manifest SHA256:
`0a39a44943bf22edc8501419d5cb8f39e31117bb23c86c1c4c48030f045d5c26`.
Actual Stage06 dataset, real processor and video decoder returned image grids
`[1,14,14]` per image (224x224 at patch size 16), for both cameras and both tested
samples. Frame 0 is supervised; frame 213 of episode 0 is not. The sampler covers
all 273465 frames without filtering uncovered/tail labels.
All 1693 quality and matching status files were re-read in this task; every
status is complete. Recomputed P1/P50/P99 agree with the supplied audit above.
The obsolete manifest is zero bytes. Joint adapter additionally returned
action `[2,32,64]`, state `[2,1,64]`, valid action-element counts `[224,7]`,
zero AR target tokens (expected for this original LIBERO source) and only one
flow target for the same first/tail sample pair.

## MolmoAct Tabletop V2 Audit (2026-09-10)

This section is a separate dataset contract for `wan_vae_latent_v2`; none of the
LIBERO delta-10 statistics above are reused.

- Root: `/opt/data/private/lq/datasets/molmoact_dataset_tabletop-v3_stage05/stage06_flow/molmoact_tabletop`.
  Manifest: `manifest.f61339e88e1b99c9.jsonl`, SHA256
  `0438bbe865b2df8da7515b9bfa995483ef0ceea8161756898da57ef6a8c6023c`.
- The manifest has 1881 episodes and 310743 frames for dataset
  `molmoact_tabletop`, camera `first_view`. HDF5 flow is float16
  `[T,2,224,224]`, mask is uint8 `[T,1,224,224]`, and saved valid fraction is
  float32. Each row carries source/target frame, actual frame/time delta and
  label source. Reader joins by mapped dataset episode plus source frame.
- `stage06_config.f61339e88e1b99c9.global.20260818T070952218100Z-8a934652.json`
  declares delta 20 and tail clamp. The actual deltas cover 0 through 20:
  273123 rows have the full delta, 35739 have positive short tail deltas, and
  1881 final identity rows have delta 0/source 2.
- The fixed V2 rule is expected actual delta 20, `flow_label_source=1`, tail
  excluded and whole-mask `flow_sample_min_valid_fraction=0.95`. A read-only
  scan of the saved scalar arrays found 265136 qualifying rows, 85.3232% of all
  rows and 97.0705% of full-delta rows. This is an exact scalar-field count;
  it is not a magnitude calibration or pixel distribution estimate.
- Generator evidence is FD-ID-FlowVLA commit
  `3824d36cdf76bf0a9d537635de92a38f3920e9a3`:
  `stage06_flow/generate.py` supplies source then target to MegaFlow;
  `flow_ops.py::normalize_and_resize_flow` validates `source + (u,v)` within
  bounds, divides `u` by source `W-1` and `v` by source `H-1`, bilinear-resizes
  normalized vectors to 224, and nearest-resizes masks. The stored direction is
  source-to-target, with positive `u` right and positive `v` down.
- The fixed color transform follows the vendored
  `megaflow/utils/flow_viz.py::flow_uv_to_colors`: RGB channel order and
  `atan2(-v,-u)`. V2 supplies flow divided by one locked training scale and
  retains float interpolation without the reference visualization's uint8
  floor. Zero flow and invalid pixels are white. It never calls the reference
  `flow_to_image` per-image maximum normalization.
- Each manifest/HDF5 supplies dataset, camera, generation, label and file-content
  identities. The V2 cache additionally binds these to mapped and label episode,
  source/target frame, FPS, source/target timestamps and actual delta. Missing
  labels remain unsupervised; malformed declared labels fail. Production readers
  verify actual HDF5 content against the declared SHA256 before first use. The
  verified result is reused only while expected SHA256 and full stat identity are
  unchanged; a changed/replaced file closes the prior handle and is revalidated.
- Current `stage06_flow_manifest_v2` / `stage06_flow_v2` files require float
  `[T]` `valid_fraction` values in `[0,1]`. The reader checks this scalar schema
  lazily when a file opens and compares each consumed value with the corresponding
  binary-mask mean using absolute tolerance `1e-6`. Missing version markers are
  an error. Only `stage06_flow_legacy_v1` explicitly declared by the manifest or
  reader contract permits an unversioned HDF5 and optional `valid_fraction`; if
  that scalar is present, the same shape/value/per-frame checks apply.

No full magnitude calibration, latent-cache build or source-data write was run.
