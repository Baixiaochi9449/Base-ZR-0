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
  dense arrays. Runtime validates fetched arrays. Full-file checksum scanning of
  all dense labels is not performed; manifest SHA256 is retained as provenance.

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
