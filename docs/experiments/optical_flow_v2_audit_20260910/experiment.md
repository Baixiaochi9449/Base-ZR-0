# Optical Flow Head V2 Independent-Review Repairs

- Created: 2026-09-10, Asia/Shanghai.
- Owner: repository user / Codex implementation.
- Purpose: repair V2 device, eligibility/normalization, cache/checkpoint identity,
  dtype, delta fallback and calibration-memory findings without formal training.
- Code baseline: `3eefb602417d3bd4b20bef2b47660b404aefb565`; the working tree contains
  uncommitted/staged V2 work plus unrelated user experiment files. No commit was
  created and no existing user change was reverted.
- VAE: `/opt/data/private/lq/models/Wan2.2-TI2V-5B-Diffusers/vae`,
  `AutoencoderKLWan`, z-dim 48. Config SHA256
  `d996c340fe9a7df5d7371f76a7d8d6956f6c98256080074d8434fa5eeac11360`;
  weight-content SHA256
  `539bd15b6ce8236827e884343c4d9245312767babed041043e181d859b5aff90`.
- VAE reference: RynnWorld-4D commit
  `2c748539849ada70d9148327e2195543ba3a921a`,
  `utils/pre-process.py::PreProcess.encode_video`.
- Dataset: MolmoAct Tabletop Stage05/Stage06, manifest
  `manifest.f61339e88e1b99c9.jsonl`, train scope, camera `first_view`, delta 20,
  source 1, tail excluded, sample valid fraction >=0.95. Exact scalar scan:
  265136/310743 eligible (85.3232%). No validation/test split was read or used.
- Flow representation: normalized source-image extent at 224x224, fixed float
  RGB Middlebury wheel, invalid/zero white, global locked scale required. No
  calibration value was produced in this review.
- Model smoke configuration: fake VAE z-shape `(4,2,2)` for unit tests; real VAE
  metadata reports `(48,14,14)` from the prior CPU probe. Default Adapter at
  VLM width 2048 has 2,172,208 parameters. Query count is externally configured.
- Objectives: V2 latent MSE only in stage2; AR+V2+FM in provisional stage3.
  Slot remains governed by its existing switch and was not fabricated. Tests use
  FP32 CPU and fake-VAE targets. No Action Expert runs in stage2 or deployment OF.
- Optimizer tests: tiny AdamW/SGD, one-step schedules, accumulation 1, CPU Gloo
  two ranks. They check update/no-update behavior and mathematical global means;
  they are not production LR or throughput measurements.
- Hardware: four NVIDIA A800-SXM4-80GB devices were visible but at 99-100%
  utilization with roughly 6.8/6.8/6.8/26.7 GiB free during final inspection.
  No GPU memory was allocated for this review.
- W&B: absent because no formal or bounded real training run was launched.
- Outputs: source/tests/docs in the working tree and temporary pytest/CLI files
  under `/tmp`; no model checkpoint, cache, calibration, label or rollout output.

Validation commands:

```sh
env PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=2 \
  /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -m pytest -q \
  tests/test_optical_flow_v2.py tests/test_optical_flow_aux.py \
  tests/test_optical_flow_review.py tests/test_optical_flow_review_round2.py \
  tests/test_aux_review4.py tests/test_optimizer_amp_skip.py \
  -k 'not real_wan_builder and not policy_loads_action_only_and_joint_and_rejects_ar_only'

env PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=2 \
  /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python -m pytest -q \
  tests/test_component_updates.py tests/test_preparation_audit_cache.py \
  tests/test_slot_data.py tests/test_three_stage_validation.py \
  tests/test_three_stage_formal.py tests/test_three_stage_formal_retry.py \
  tests/test_aux_review5.py

env PYTHONNOUSERSITE=1 PYTHONPATH=. \
  /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python train_vla.py --help
env PYTHONNOUSERSITE=1 PYTHONPATH=. \
  /opt/data/private/lq/miniconda3/envs/ZR-0/bin/python \
  scripts/run_three_stage_validation.py --mode commands
```

Results: the first overlapping group passed 151 tests with 2 deliberate
deselections; the focused final V2 file passed 19 with the real-Wan GPU test
deselected; the stage/data/checkpoint group passed 103. Python compilation, CLI
help, launcher command generation and `git diff --check` passed. The warnings
were DeepSpeed/Pydantic deprecations. One intermediate CPU run with CUDA hidden
failed only while Accelerate imported DeepSpeed/Triton and found no active CUDA
driver; the same regression passed with normal device visibility and made no GPU
allocation.

Not verified: post-fix real Wan device migration/loss/backward on CUDA,
decoder round-trip after these repairs, NCCL, a real DeepSpeed/ZeRO-2 engine,
FP16 cache error against real FP32 targets, a locked full-train calibration,
Adapter versus mean-latent baseline, shuffled-Query dependence, 2B training,
checkpoint recovery under the real backend, or rollout. Before controlled real
smoke, produce/review the locked training calibration, build a bounded matching
cache if strict mode is selected, and obtain an idle GPU. Formal training still
requires a complete experiment record and online W&B.

## Second review

- Cache integrity: entry protocol 3 hashes canonical dtype, shape and contiguous
  latent bytes. The official bounded builder uses no-replace atomic entry writes,
  then atomically publishes `cache_manifest.v1.json`; strict configuration and
  checkpoints lock the manifest SHA256. Reads revalidate the locked manifest and
  actual returned tensor. Legacy entries, altered tensors/manifests and conflicting
  existing writer inputs fail without overwrite.
- Calibration split: calibration protocol 4 requires a manifest-bound version-1
  JSON train frame allowlist and verifies all episode/frame rows against source
  identities and ranges. No directory-name or CLI split inference remains.
- Reader schema: current Stage06 v2 requires finite float `[T]` valid fractions in
  `[0,1]`; consumed rows are checked against mask means at absolute tolerance
  `1e-6`. An unversioned legacy input is accepted only when the manifest or
  reader explicitly declares `stage06_flow_legacy_v1`.
- GPU gate: real Wan CUDA pytest is disabled unless
  `ZR0_RUN_REAL_WAN_CUDA_TEST=1` and exactly one `CUDA_VISIBLE_DEVICES` entry are
  supplied. It reuses `utils.gpu_resource_gate.wait_for_gpus` before VAE loading;
  a pass is a point-in-time resource check, not a reservation.
- Final second-review validation used `PYTHONNOUSERSITE=1`, `PYTHONPATH=.`, the
  `/opt/data/private/lq/miniconda3/envs/ZR-0` interpreter and three disjoint test
  groups: V2 focused `37 passed, 1 skipped`; legacy Flow/Query/checkpoint/AMP
  `178 passed`; stage/data/component/resume `103 passed`. The only skip was the
  default real-Wan CUDA gate because `ZR0_RUN_REAL_WAN_CUDA_TEST=1` was not set.
  CLI help, both auxiliary-script help commands, six-command launcher dry-run,
  `py_compile` and `git diff --check` passed. A read-only real Tabletop row returned
  delta 20, source 1 and mask mean `0.9769810438` through the updated reader.
- No real Wan/GPU allocation, cache build, calibration, training or rollout ran.
  Online CUDA smoke still needs an explicitly selected idle device that passes the
  project gate plus an approved locked train scale. Strict smoke additionally
  needs an official cache whose published manifest SHA256 is supplied in config
  and whose entries cover the smoke sample index.

## Third review

- HDF5 identity: Stage06 first access now verifies actual Flow file content
  against the manifest SHA256. A process/worker caches the result only for the
  same expected digest and complete stat identity. A changed or replaced file
  evicts the HDF5 handle and row index before revalidation; mismatch is a
  `DatasetIntegrityError`, never missing supervision or cache regeneration.
  Existing preparation-audit records remain usable because they bind an actual
  digest to the same stat identity.
- Strict manifest lookup: strict builders retain the fully validated manifest
  entry index per process while the complete manifest stat is stable. A stat
  change rechecks the configured SHA256 and reparses only if that lock still
  matches. Selected latent bytes, dtype, shape, source and protocol continue to
  be checked on every lookup. The stat check is a change detector, not a hash or
  exclusivity guarantee.
- Calibration authority: protocol 4 requires dataset `meta/info.json` and
  `meta/stage05_episode_mapping.jsonl` by default; `meta/stage05_merge.json`
  binds the Flow dataset ID to that source. Flow source episodes are mapped to
  dataset episodes before checking authoritative train ranges. An explicit
  externally-trusted mode remains for sources without verifiable split metadata
  and is recorded as not independently verified.
- Schema: missing manifest and HDF5 version markers no longer select legacy.
  Current v2 requires both markers and `valid_fraction`; only explicit
  `stage06_flow_legacy_v1` accepts the supported unversioned layout.
- No full calibration/cache generation, formal training, rollout, GPU, NCCL or
  ZeRO-2 operation was started in this review. Final regression evidence is
  recorded after the complete test run.
- Before the fix, a local current-schema reproduction accepted an HDF5 whose
  Flow tensor was changed to 123 while its manifest SHA stayed fixed, and five
  strict target reads performed five full manifest validations. After the fix,
  three non-overlapping CPU groups passed: `53 passed, 1 skipped` for V2 plus
  preparation-audit cache, `75 passed` for existing Flow/Stage05/Stage06/Slot,
  and `293 passed` for Query, checkpoint, optimizer, AMP, AR/action/Slot and
  stage transitions. The only skip was the default real-Wan CUDA gate. A
  read-only real Tabletop episode-0 row verified HDF5 SHA256
  `6bf18a53d502628739c018ebb9b65ce4945d174ce886265c1ea03733c16867e9`
  before returning delta 20/source 1. Main CLI help, calibration/cache help,
  six-command launcher dry-run, `py_compile` and both diff checks passed.

## Fourth review follow-up

- The production reader, online cache builder and calibration now hash the
  current HDF5 bytes and compare them with the manifest SHA256. A preparation
  audit cache only supplies the expected immutable source/stat binding; it is
  not treated as the current dense-content digest. A changed or replaced file
  evicts handles and row indices before revalidation.
- Strict cache manifests use a process/worker-local validated index while the
  complete manifest stat identity is stable. The synthetic regression has
  10,001 entries, proves repeated target lookup does not revalidate the full
  manifest, and covers truncate/replacement and locked-SHA rejection. Latent
  content remains verified per selected entry.
- Calibration requires authoritative split metadata covering all episodes and
  the Stage05 old-to-new mapping; explicit external allowlists are marked
  `not_independently_verified`. Current v2 schema requires both version markers
  and `valid_fraction`; only explicit `stage06_flow_legacy_v1` enables the
  supported unversioned input.
- This follow-up passed `46 passed, 1 skipped` in the full V2 file, `78 passed`
  for the complete Flow/Stage05/Stage06/Slot reader group, `95 passed` for
  optimizer/AMP/Difference Query/checkpoint, `48 passed` for review/config
  regressions, and `24 passed` for the Stage05 mixed-sidecar group. The
  designated ZR-0 environment resolved the repository LeRobot package; an
  earlier system-Python invocation was rejected at import time and is not
  counted. No GPU, cache, calibration or training workload was started.
