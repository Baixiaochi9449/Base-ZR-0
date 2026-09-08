# Implementation Reference

## PyAV decoder ownership and bounded GPU resource gates

- Date: 2026-09-07. Purpose: prevent delayed AV1 decoder-thread release and replace single GPU-utilization snapshots with bounded, recorded availability checks before sequential launches.
- Files/interfaces: `lerobot/lerobot/common/datasets/video_utils.py::{resolve_pyav_threads,decode_video_frames_pyav,decode_video_frames,decode_video_frames_torchvision}`; `utils/gpu_resource_gate.py::{GPUResourcePolicy,wait_for_gpus,cuda_device_identities}`; H10 runner `resource_gate/launch/verify_run`, standalone `audit_stage05_gpu_gate.py`, LIBERO sequencer gates and `run_libero_wo_ecot_pt.sh` device selection. Diagnostics: `scripts/audit_video_decode_resources.py`, `scripts/audit_gpu_probe_exit.py`; tests: `test_video_decode_resources.py`, `test_gpu_resource_gate.py`, existing H10 and LIBERO launcher tests.
- Source: adapt the existing local LeRobot timestamp/nearest-frame/RGB pipeline and existing H10 resource thresholds. Custom public-PyAV ownership helper is needed because installed torchvision's PyAV VideoReader ignores num_threads and offers no public reader cleanup. Custom shared GPU polling replaces two one-shot checks and a memory-only unbounded wait; no external algorithm or framework is added. Installed PyAV12.3.0/torchvision0.21 APIs and upstream PyAV v12.3.0 container/codec source were inspected; exact links and evidence are in `experiments/resource_preflight_20260907/experiment.md`.
- Configuration/defaults: explicit `num_threads` or `LEROBOT_PYAV_THREADS` (default1, explicit0 requests automatic threads). This intentionally changes PyAV's unsafe automatic-thread default only; no unrelated thread pool is changed. Cleanup is always active, including with0 threads. `ZR0_GPU_GATE_{POLL_SECONDS,CONSECUTIVE_SAMPLES,TIMEOUT_SECONDS,MIN_FREE_MIB,MAX_USED_MIB,MAX_UTILIZATION,QUERY_TIMEOUT_SECONDS}` defaults to2/3/60/71680/1024/0/5. These are corrections to existing resource management, not new model/objective features; there is no switch bypassing ownership or safety checks.
- Input/output/dataflow: path+timestamps -> owned lazy PyAV container/codec/generator -> existing nearest-frame selection -> unchanged RGB float tensor. Generator closes before codec/container, with nested finally blocks and public API capability checks; errors propagate. No active decoder cache or cross-worker sharing is introduced; existing bounded data caches persist. GPU checks resolve actual CUDA-visible UUID/PCI identities, poll compute processes/memory/utilization plus own child/group exit, log every observation and pin H10 launches to the verified UUID order. Unknown queries/identity/MIG/MPS configurations reject; timeout retains diagnostics. Thresholds allow an explicit idle memory baseline but never ignore a compute PID.
- Compatibility: image resize, selected frames/timestamps, AR targets, actions/masks, losses, optimizer/scheduler, worker count, sampler algorithm/resume, checkpoint validation and Stage1 compatibility certificates are unchanged. H10 source identity now also records the decoder implementation. Source registry and historical artifacts are not rewritten. Historical formal runner branches are not invoked by this preflight; future training authorization remains three stages of100 successful updates only.
- Verification: real AV1 delayed release reproduced (224 additional native threads per retained old decoder in this environment); historical full-DROID global_index2041670 located and included. Eight frame-tensor and four complete production-sample comparisons are exact. Four256-iteration fixed decoder scenarios hold threads/FDs constant after warmup, including injected failure after a real frame. Production4-worker DROID/RH20T two-epoch-prefix reading checks bounded resources and worker exit.59 initial targeted tests pass; broader regression and environment/artifact blockers are documented in the experiment record. Real occupied-GPU polling rejects, then later passes after release and three confirmations. Two actual four-GPU resource probes exit successfully and the gate waits for each owned child; zero training updates. Final H10/LIBERO tests cover UUID handoff. No formal training, automatic staging or commit.
- Final verification: 82 targeted decoder/GPU/H10/LIBERO tests passed, along with Bash syntax and diff checks. The actual PyAV log reports requested_threads=1 and actual_thread_count=1. The existing full regression retains nine stale-audit failures caused by the separately edited registry; the driver-visible auxiliary tests pass their 30 cases but initialize CUDA, so their extra CPU-only assertion is recorded as failed.
- Limits: delayed native resource release is proven, but historical MemoryError cannot be assigned a unique cause without historical resource telemetry. CPU stress is not full distributed training. GPU gate checks are not exclusive reservations. Complete three-stage100-update validation requires a concrete checkpoint/batch/head configuration plus valid data/checkpoint gates; its update counts remain 0/0/0. The two resource probes are complete and exited.

## Fifth auxiliary review: explicit Head removal and static objectives

- Date: 2026-09-07. Purpose: permit Stage 2 to Stage 3 Head ablations and reject
  Stage 2 configurations that cannot produce any positive weighted objective.
- Source: adaptations of the existing `resolve_slot_checkpoint`,
  `resolve_flow_checkpoint`, `ZR0Model`, `parse_train_options`, launcher and
  `global_supervision_counts`; no external implementation. The new connected
  `utils/aux_objectives.py` shares the coefficient chain between the existing
  phase resolver and window activity calculation, which previously differed.
- Switches: the existing `WITH_SLOT=0` / `WITH_FLOW=0` launcher flags now emit
  explicit `none` and zero outer weight. An omitted CLI Head option still
  inherits the source configuration. Only explicit `init_from_checkpoint`
  across different declared stages permits removing saved Heads; ordinary
  loading and same-stage resume remain strict. No new missing-Head random
  initialization policy is introduced.
- Source validation: source config fields, hashes, statistics, complete Flow
  tensor shapes, Query tensors and common role metadata are checked before
  constructing target modules, including when both target Heads are absent.
  Validation reads checkpoint artifacts, not disabled dataset labels. Saved
  Query total, weights and role boundaries are retained. Expert initialization
  retains the existing independent seed42 policy and all-Query conditioning.
- Config-object input: `SlotConfig()` / `OpticalFlowConfig()` explicitly remove
  their respective saved Head during cross-stage init. Unspecified constructor
  defaults inherit the saved inactive settings; nondefault conflicting
  settings and an explicit changed Query count are rejected. CLI fields retain
  their exact explicitness, including explicit values equal to defaults.
- Outputs: the target checkpoint exports only active Head weights. Common
  metadata additionally persists the complete Slot runtime config plus hash,
  including disabled Slot settings. Legacy checkpoints without this optional
  field keep their previous defaults/artifact-based config resolution. Flow
  config was already always exported. Resume rejects enabling a saved disabled
  Head or changing its config. Source files are never rewritten.
- Static objective semantics: Stage 2 requires an enabled Head with a positive
  outer coefficient and a positive effective task. Q1-Q8 always contain their
  unit-coefficient base losses; Q9 requires at least one positive presence,
  bbox or risk coefficient. Positive Flow is an alternative. CLI and model
  validate the final inherited/overridden config before dataset/large-model
  construction. Error messages report the stage and complete weight chain.
  Stage 3 zero-weight Slot ablations remain legal through AR/FM.
- Runtime compatibility: static validity does not inspect sample masks or loss
  values. Raw coverage, loss reductions, positive-weight zero loss, GAS/rank
  aggregation and empty-window synchronization/skipping are unchanged. Stage1,
  explicit four-source dataset routes and sampler epoch restoration retain
  their existing behavior. Slot artifact identity inputs are unchanged, so
  existing slot_sources_v3 and Aux/Joint sidecars remain usable.
- Validation: `tests/test_aux_review5.py` records fail-first conversion,
  source-integrity and CLI/model startup tests. The existing production matrix
  now initializes heads-off Stage3 from real Stage2 Both; optional
  `--cross-stage-removals` covers the other removals with real four-source data.
  `scripts/verify_auxiliary_sampler_resume.py` verifies the converted checkpoint
  with a fresh process and the full production sampler. Commands, results and
  GPU/multi-rank limitations are recorded in `docs/structured_slot_review5.md`.

## Third Optical Flow review: verified optimizer updates

- Date: 2026-09-06. Purpose: do not count AMP overflow as completed supervision.
- Source: adaptations of `train_vla.py::run_optimizer_step_window`,
  `utils/optimizer_step_loss.py`, and
  `utils/optical_flow_checkpoint.py::record_stage_training_window`.
  New custom `OptimizerWindowStep` centralizes the backend outcome because the
  existing loss accumulator cannot establish whether parameters were updated.
  `advance_global_step` consumes the same validated result as checkpoint state.
- Inputs/outputs: a window's existing optimizer/engine, scheduler and Accelerator
  produce `optimizer_update_applied`, `optimizer_update_skipped` (booleans), and
  `optimizer_skip_reason` (`none`, `no_supervision`, `amp_overflow`, or
  `optimizer_step_skipped`). Device-local collective tensors verify all ranks
  agree on update status and normalize skip reason. Unknown results or rank
  disagreement fail instead of creating completed-window evidence.
- Backend contract: AcceleratedOptimizer uses its actual `step_was_skipped`,
  with synchronized gradient boundaries checked. Native non-AMP PyTorch
  AdamW/Adam/SGD have no implicit skip mechanism and retain their existing step
  behavior. Raw optimizers with an external scaler are rejected; AMP requires
  the AcceleratedOptimizer result API. Other optimizer implementations, backend
  types outside NO/MULTI_CPU/MULTI_GPU/DEEPSPEED, and multiple optimizers fail
  explicitly rather than risk partial or unknown updates.
- DeepSpeed: use public `engine.was_step_applied()` after Accelerate's backward
  has called engine.step. An optional engine optimizer overflow signal names
  the reason; the wrapper's fallback `step_was_skipped=false` is not evidence.
  Engine-owned schedulers are not stepped again by the trainer. Their state is
  checked against a pre-window snapshot on a skipped update. External schedulers
  run only on success. This is based on inspection of installed Accelerate
  `optimizer.py`, `utils/deepspeed.py`, and DeepSpeed `runtime/engine.py`; tests
  simulate the engine contract and do not validate a real DeepSpeed runtime.
- Empty supervision still skips backward/optimizer/scheduler. AMP overflow
  keeps per-window forward/loss metrics (computed flags may be true) but updates
  neither durable global step nor cumulative flags/count. JSONL/W&B report skip
  reason at the unchanged global step for every stage, including legacy loss
  modes. Only no-supervision increments the existing `flow_skipped_batches`.
- Checkpoint state schema is now `stage_metadata_version=2` and includes integer
  `completed_optimizer_windows`, incremented only on a verified update. Version
  1 did not check AMP results, so v1 or unversioned history warns and migrates to
  unknown AR/OF/FM flags, Slot false, `legacy_history_unknown=true`, and zero
  verified windows. The count is a lower bound when legacy history is unknown;
  subsequent verified updates increase it. Resume preserves this evidence;
  explicit init resets it. DeepSpeed's own internal attempt counters are backend
  state; durable training global step comes from the existing client checkpoint
  state and is a count of confirmed updates.
- Switches/defaults: no new experiment flags, objectives or optimizer settings.
  Existing successful updates, Query-off, masks, label normalization, dtype and
  module ownership are retained; skipped updates now have correct state/logging.
  ZeRO-3 remains rejected. Supersedes the v1 checkpoint-state semantics below.
- Tests: new `tests/test_optimizer_amp_skip.py` uses real CPU GradScaler and
  AcceleratedOptimizer with Inf gradient injection and finite recovery, across
  all three stages and StepLR/constant/cosine/AcceleratedScheduler. It checks
  parameters, scheduler, global step, counter, flags and JSONL/W&B. Engine
  ownership, unknown backends, multiple optimizers and rank disagreement use
  explicit simulations. Existing regression workers use the production metric
  serializer to retain new boolean/string results. Commands, results and limits
  are recorded in `docs/experiments/optical_flow_cpu/experiment.md`.

## Second Optical Flow review: image contract, checkpoint state and aux mode

- Date: 2026-09-06. Purpose: reject inconsistent Stage06 model image inputs,
  persist structured cumulative stage evidence, and accept explicit stage2 aux.
- Source: adaptations of `utils/dataset_spec.py::resolve_stage06_flow_contract`,
  `utils/dataset_manifest.py::dataset_spec_to_manifest` and semantic validation,
  `utils/load_training_dataset.py::prepare_qwen_vl_inputs_cpu`,
  `model/reasoning_vla_model.py::ZR0Model`,
  `utils/optical_flow_checkpoint.py` and `train_vla.py::run_optimizer_step_window`.
  `utils/cli_options.py` help now documents the already supported stage2 aux mode.
  No external implementation or new loss/head is introduced.
- Image contract: the shared custom `qwen_image_input_contract` supplies current
  224x224 model resize hints. `do_resize=false` means no second processor resize;
  `random_geometric_augmentation=false` means no unsynchronized random transform.
  These dimensions describe the model input, not MegaFlow label generation.
  Training creates this contract; inference reads the saved contract and compares
  it to the same runtime definition. Missing, malformed or conflicting contracts
  fail. `vision_input_contract` participates in manifest semantic comparisons.
  Policy/server validation reads checkpoint JSON plus source info/stats, not HDF5.
  v2 and Query-off preprocessing remains the same.
- Custom cumulative state helpers in `utils/optical_flow_checkpoint.py` extend
  existing save/load code because per-window metrics cannot establish training
  history. Inputs are globally reduced boolean window flags; output is versioned
  JSON metadata shared by the ordinary metadata file and OF sidecar. Fields:
  `training_stage`, `stage_description`, `provisional`, all four
  `*_loss_computed` flags, `stage_metadata_version=2` (v1 superseded above),
  `loss_computed_scope=current_stage_completed_optimizer_windows`, and
  `legacy_history_unknown`. Window flags are ORed only after the optimizer window;
  standalone forward/evaluation and globally skipped stage2 windows do not mark
  training evidence. A supervised numerical zero still counts as computed.
- Fresh construction and explicit initialization reset cumulative flags; resume
  restores them. Cross-stage source evidence is exposed separately as
  `source_stage_training_state`, including inference loading. Legacy staged
  checkpoints without the schema warn, keep unknown AR/OF/FM flags as JSON null,
  and retain `legacy_history_unknown=true` on resave. Observed subsequent losses
  become true. Slot is always false; stage3 is always provisional. Ordinary and
  OF copies must agree. Latest-window JSONL/W&B flags remain per-window metrics.
- `ZR0Model._resolve_training_loss_type` recognizes aux while retaining exact
  constructed-mode matching. Only stage2 can be constructed with aux; stage1,
  stage3 and legacy objective conflicts still fail before any module forward.
- Switches/defaults/compatibility: existing `training_stage`, Query and OF
  switches only; no default, optimizer, sampling, mask, Head, loss normalization
  or action-conditioning change. Legacy unstaged objectives acquire no stage
  logging or cumulative checkpoint fields. No inference Head/HDF5 requirement.
- Verification: `tests/test_optical_flow_review_round2.py` tests actual tiny
  backward/window flags and structured save/load, init/resume, legacy unknowns
  and inconsistent metadata copies. `tests/test_optical_flow_review.py` tests
  real Stage06 metadata/tiny policy initialization with valid, missing, malformed
  and rehashed 448x448 contracts. The existing two-rank Gloo regression now checks
  cumulative flags on both ranks. Full commands/results and environment limits
  are in `docs/experiments/optical_flow_cpu/experiment.md`.

## Optical Flow review corrections

- Date: 2026-09-06. Purpose: repair distributed logging devices, Stage06 policy
  provenance, unsafe ZeRO-3 exports, incomplete Query/OF checkpoints, mixed
  precision forward, and missing stage/loss-computed log metadata.
- Sources: adaptations of `utils/optimizer_step_loss.py::OptimizerStepMetricAccumulator`,
  `train_vla.py::run_optimizer_step_window`, `utils/wandb_training_logger.py::WandbTrainingLogger`,
  `utils/dataset_spec.py::resolve_dataset_spec`, `utils/stage06_dataset.py::Stage06LiberoDataset`,
  `policies/reasoning_vla_policy.py::ZR0Policy`,
  `model/difference_query.py::_read_checkpoint_query_data`,
  `model/reasoning_vla_model.py::ZR0Model`, and `utils/optical_flow_checkpoint.py`.
  No external architecture or loss implementation is introduced.
- Devices and metrics: supervision counts, detached sums, weights and collective
  inputs use `accelerator.device`, independently of AR/FM presence. W&B also
  places tensor inputs on that device before reduction. The existing JSONL/W&B
  path preserves stage strings and boolean computed/provisional fields; Slot
  remains false with no numeric Slot loss. Computed means valid supervision on
  at least one rank/microbatch, including a legitimate numerical zero. Empty
  global stage2 windows log false flags and coverage/skips without updating the
  optimizer, scheduler or global step. Stage3 without Slot is provisional.
- Dataset contract: custom `resolve_stage06_flow_contract` centralizes the
  previously training-only manifest digest, canonical camera/delta schema and
  224x224 vision metadata. Training reads JSONL and compares its hash to the
  reader's startup hash. Policy/server read that provenance from the checkpoint's
  hashed resolved manifest, while validating source image metadata and action
  statistics through the existing resolver. No HDF5 access is needed in policy.
- Checkpoints: OF training startup and both save entrypoints reuse a new
  rejection guard from `utils/optical_flow_checkpoint.py`; direct Head save
  rejects DeepSpeed partition metadata as well. Supported paths are non-ZeRO
  and the existing ZeRO-2 path; ZeRO-3 full-parameter gather is not implemented.
  Hash success alone is not proof of full parameters. OF sidecars require a
  Query declaration; enabled OF requires complete enabled Query artifacts,
  consistent counts and Query/Head input width. This supersedes the older
  statement that missing Query artifacts alone allow legacy random initialization:
  there must also be no OF artifacts. Explicit Query-off with disabled OF
  stage metadata remains valid. Init/resume ownership and RNG behavior are unchanged.
- Dtype: `model/optical_flow_aux_head.py::DenseRegressionFlowHead.forward` casts
  only its trailing Query input to the projection parameter dtype. This is a
  differentiable boundary cast, not a parameter conversion. Default parameters
  and exported weights remain FP32; explicit `.to(dtype)` and outer autocast
  retain their usual roles. OF loss stays FP32. Attention masks, future-target
  isolation, query slicing, Action Expert inputs and loss normalization are unchanged.
- Switches/defaults: the existing stage, OF and Query switches apply. No new
  CLI flags, objectives, sampling rules or optimizer settings. OF-off does not
  construct the Head or read Stage06; legacy objectives keep their output schema.
- Verification: `tests/test_optical_flow_review.py` covers real tiny Qwen
  FP32/BF16/FP16 backward with/without CPU autocast, all Head/input dtype pairs,
  Query/OF integrity, export rejection, accelerator-device simulation and actual
  accumulator/JSONL/W&B calls. A real Stage06 metadata test saves and loads a tiny
  checkpoint through `ZR0Model` and `ZR0Policy` with HDF5 access forbidden.
  `tests/test_optical_flow_aux.py` extends real two-rank CPU Gloo tests through
  W&B logging for mixed labels and synchronized empty windows. Existing legacy
  loss, Query, policy/server, optimizer, dataset and resume regressions are rerun;
  old logger test doubles now declare their CPU device.
- Limits: no real NCCL, OF DeepSpeed engine save/resume, full 2B continuous
  training or rollout is claimed. Earlier repository ZeRO-2 validation predates
  OF. CPU FP16 small gradients can underflow; use the normal precision/scaling
  setup for experiments. Commands and actual review results are recorded in
  `docs/experiments/optical_flow_cpu/experiment.md`; operational contracts are in
  `docs/optical_flow_training.md`.

## Saved processor audit selection for the H10 experiment

- Date: 2026-09-06. Purpose: resolve the explicitly authorized mismatch between the base processor's audit identity and normal saved processor file identities, then continue the original two-stage experiment.
- Sources: `scripts/compare_stage05_saved_processor.py` directly reuses `AutoProcessor` and `Stage05MixedPretrainingDataset` for bounded diagnosis. `scripts/run_stage05_h10_experiment.py::select_token_audit` reuses the existing token-audit file records, trusted-spec validation and validation API; no validator or new identity framework is added. A selected report/spec is passed through the shell launcher's existing environment interfaces. The one authorized complete audit ran through the unchanged existing audit command.
- Diagnostic scope: loaded tokenizer backend, vocabularies, special tokens, actual template and image processor configuration match. Four longest two-view source samples plus the RH20T wrist-fallback sample have exactly equal input_ids, labels, masks, image placeholders, grid and pixels. This sample result is not a whole-dataset equivalence proof; an independent full existing-tool audit is required for the saved processor identity.
- Switches: `--processor-audits` defaults off. When enabled, each actual initialization/resume processor and stage-end saved checkpoint is matched against existing report file identities, then validated against current data, implementation and trusted spec. Matching identities reuse reports regardless of processor directory, optimizer step or model weights. Per the latest user instruction, automatic audit generation has been removed: failed reuse records exact old/new existing identity fields, data eligibility identity changes and loaded processor runtime identity changes, then stops. These diagnostics are not a new identity or semantic-hash system. No old report, processor or checkpoint is overwritten. No H10 sidecars or statistics are regenerated.
- Continuation: `--continue-ar-resume-gate --continuation-id processor_audit --processor-audits` retains old attempt records/logs and allows only the recorded pre-training processor-audit failure to proceed with the existing AR32/GAS2 checkpoint. Original failure-stop, source pinning, exclusive lock, fresh formal initialization and Joint sequence remain. AR fresh selects the base audit; resume selects its checkpoint audit; Joint selects the formal AR processor audit and its own saved processor audit when resuming.
- Recovery outputs: with processor-audit selection enabled, the original two-update checkpoint remains the resume input, while step3 saves into a separate `mbs*_gas*_resume_<continuation-id>` probe directory through the unchanged production save path. Existing step2 optimizer/processor artifacts remain intact. Formal stages still initialize from the original base or final formal AR source, never the recovered probe.
- Validation: bounded processor comparison, targeted selector/continuation tests, CLI/dry-run, syntax/diff checks and the authorized one-update real recovery gate. Runtime results belong to the experiment document. Core model, scheduler, loss, normalization, audit implementation and checkpoint contracts remain unchanged; all deferred issues remain deferred.
- Actual full audit: the already-running job was not restarted when the user restricted further audits. It finished and validated at12:29:28 Asia/Shanghai, covering14121879 eligible frames with required_max_length941. Data and implementation identities and all four dataset measurements match the base audit exactly. Only processor model-file identity and recorded source path differ. Independent report content hash `a16346ec6747e8813bed735c05fcc972c93578f0c603e10bfde29891a6f9c673`, file SHA256 `1fb9ba433fd0ffb81dcbe4257515c4c3efc05620b9ff1acd63eb61309e49d121`, implementation SHA256 `ca23666ebe657b28d34602d388b2a98c679478664fb69588e85f66a6241ee004`. The report and trusted spec are under the experiment's `processor_audits/audit-001/`; applicability is decided by the original validator, not its recorded path alone.
- Actual recovery: the original four-rank DeepSpeed entry loaded step2 optimizer/scheduler state and completed exactly one update to step3, loss1.9587853, finite Query/VLM gradients67.72065/65.69374, zero truncation. Independent step3 state passed full checkpoint validation and reused audit-001 at12:47:52. Scheduler counters8/9 became12/13; original step2 files remain intact. After the successful training exit, the runner had stopped on a transient GPU utilization sample (GPU0=100% with no compute processes and81226MiB free); subsequent checks showed all GPUs idle. No resource rule was relaxed and no recovery update was repeated.
- Completed-gate selection: `--continue-ar-formal` defaults off and requires both an exited-zero AR resume record and a completed step3 checkpoint gate. It revalidates that checkpoint and continues directly to fresh formal AR and the existing Joint sequence, preserving prior attempt records with a unique continuation ID. It cannot resume a failed or unverified recovery or repeat probe updates. This is a stage-selection adaptation only; resource gates and all training behavior remain unchanged.
- Formal startup confirmed: runner2583187 / tmux `four_dataset_h10_formal_20260906`, AR launcher/process-group2583879, started12:54:36 Asia/Shanghai. Actual prepared scale is4*32*2=256, fresh Qwen base/seed42/Query/state,27582 updates/warmup1379/save5000. At12:58:51,17 formal optimizer updates were recorded (first loss3.8246400, step17 loss3.3913248), with finite loss/LR/gradients and zero truncation. W&B remote_available=1. The same live runner owns the gated Joint probe/resume/formal95423/warmup4771 sequence; Joint has not started. No further full audits are automatic. Final targeted coverage:7 selector cases,4 preflight environment/reuse cases and8 continuation cases passed; real five-sample comparison, the one full audit, one actual recovery update, CLI/dry-runs, compile and diff checks passed. No model, loss, scheduler, data, normalization or checkpoint-system implementation changed in this task.

## H10 experiment CUDA visibility and authorized AR gate continuation

- Date: 2026-09-06. Purpose: correct only the sequential runner's CUDA-hidden recovery preflight and continue the existing experiment from its successful AR 32/2 two-update checkpoint.
- Source: adapt `scripts/run_stage05_h10_experiment.py::verify_run/main/launch`; reuse the existing resource gate, checkpoint validator and production shell launcher. Preflight now exposes GPUs 0,1,2,3 after the resource check; the validator retains its existing CPU tensor loading. The launch environment also explicitly exposes these GPUs to the shell's own preflight. No dependency, checkpoint validator, training loop, optimizer or scheduler changes.
- Switch: `--continue-ar-resume-gate`, default off. It reuses existing H10 artifacts, representative AR samples and the saved 32/2 Accelerate configuration; skips all fresh AR probes; verifies step2, runs production AR resume to step3, verifies it, then enters the original fresh formal AR and sequential Joint flow. Without the flag the original fresh probe sequence remains. Failure still stops all following stages.
- Continuation records: an exclusive runner lock and one-use continuation marker prevent duplicate instances. The original runner record, source hashes, environment and diff remain intact. Separate continuation records pin the corrected runner while requiring all other previously pinned training files to remain unchanged. No historical checkpoint is migrated or edited.
- Tests: `tests/test_stage05_experiment_launch.py` checks GPU environment propagation for both preflight purposes and the continuation's skip, fresh initialization, failure-stop and preserved-record behavior using subprocess mocks. Actual step2 checkpoint preflight passed at 11:35:39 Asia/Shanghai with the corrected environment; this alone is not a real four-rank recovery result. Actual resume/formal status is maintained in the experiment document.
- Actual continuation: four targeted tests passed (three unrelated tests deselected); continuation dry-run, CLI help, Python compile, Bash syntax and diff checks passed. The single tmux runner PID2479815 started at11:42:53 Asia/Shanghai; corrected CPU-loading checkpoint preflight passed again at11:43:44. The AR-resume launcher (process group2480202) then failed its existing token-audit validation at11:44:52, before Accelerate/ranks initialized: `processor/tokenizer files identity changed`. The audit binds base-model processor files while the saved checkpoint has different file records, including tokenizer/config serialization and JSON-to-Jinja template files. No audit/validator/checkpoint changes or retry were made. Global step remains2; formal AR/Joint are unstarted and the tmux runner has exited. This later failure supersedes the earlier CUDA-only blocker; it is outside this correction's scope.
- Compatibility: H10/DQ32, AR=1*AR and Joint=AR+5*FM, data/normalization, all training parameters and original stage budgets remain unchanged. Joint/downstream deep state preflight and token-audit helper defaults remain deferred.

## Four-dataset DQ32 H10 GBS256 seed42 experiment and Joint scheduler repair

- Date: 2026-09-06. Purpose: execute the explicitly authorized two-stage experiment with isolated H10 data/configuration, short real four-GPU probes, checkpoint recovery gates and automatic sequential formal AR/Joint launches.
- Sources: adapt only `utils/stage05_checkpoint_contract.py::validate_stage05_resume_artifacts` so Joint uses the AR path's existing saved `dp_world_size` scheduler stepping evidence. The installed Accelerate 1.6.0 wrapper and DeepSpeed 0.15.4 save path are shared by both stages. No scheduler call, LR curve, warmup, optimizer, loss or save format changes. Joint/downstream deep optimizer-state preflight and token-audit helper defaults remain deferred. The earlier section's statement that Joint retained a multiplier of one describes the previous repair and is superseded by this explicitly authorized change.
- Tests: `tests/test_stage05_ar_scheduler_resume.py` now covers both purposes through the production scheduler save helper with a tiny model-state facade and real CPU AdamW/wrapper; `tests/test_stage05_checkpoint_contract.py` adds the already-saved DP field to its fixture. Missing saved evidence is rejected without migration or guessed multipliers. The initial targeted run passed 35 tests; no full suite is run.
- Experiment configuration: `configs/four_dataset_dq32_h10_gbs256_seed42_20260906.json`. `scripts/prepare_stage05_h10_experiment.py` invokes existing `build_stage05_sidecar` and the byte-identical existing token audit script in an isolated configuration view. That view owns its registry and explicitly selected trusted spec; source dependencies retain their actual hashes. Historical registry/config/audit files remain unchanged and usable. H32 artifacts are never relabeled as H10.
- Training entry: `scripts/stage05_experiment_train.py` selects the experiment registry in the launch process and invokes the existing `train_vla.train`. An explicitly selected representative-sample fixture replaces only probe batch selection; no second training loop, model, adapter, loss, normalization, or formal sampler is added. Without that probe environment variable the production sampler is used unchanged.
- Launcher: existing `scripts/run_stage05_four_dataset_pretraining.sh` accepts optional experiment-specific audit script/spec, training entry, input checkpoint, output directory and save interval. Defaults remain the prior experiment. `scripts/run_stage05_h10_experiment.py` sequentially executes probe/resume/formal for AR then Joint, stopping on failure. Only confirmed CUDA OOM in a fresh probe advances the predefined candidate list; cleanup is restricted to that launch's process group. No formal restart or historical checkpoint cleanup is implemented.
- Training constants: H10, DQ32, max_length1024, seed42, GBS256; AR=1*AR, Joint=1*AR+5*FM; AdamW betas(0.9,0.95), epsilon1e-8, weight_decay0.01, clip1.0, LR1e-5 to1e-6, warmup5%. Existing optimizer-update budget interface sets AR27582/warmup1379. New H10 sidecars must confirm Joint counts before its budget is fixed. Formal save interval5000 plus final step. Explicit user W&B policy is best_effort.
- Experiment documentation: `docs/experiments/four_dataset_dq32_h10_gbs256_seed42_20260906/experiment.md`; each runtime directory receives its own document before training. Data preparation and lifecycle records contain actual counts, commands, runtime choices, process IDs and outcomes. Raw samples are not uploaded.
- Compatibility and checks: generic Expert contracts, model/Query/mask, data adapters, sampling and normalization, token audit implementation, and historical H32 configuration remain unchanged. Directed scheduler/launcher/fixture tests, CLI help, syntax, diff checks and the authorized two-step/one-resumed-step GPU gates are the verification boundary. Runtime results are updated below after the gates.
- Completed preparation: all eight H10 sidecars were generated once and validated; AR arrays retain 14121879 frames exactly. Joint has 24428218 frames (DROID19821342, Household794199, Tabletop310743, RH20T3501934), 95423 optimizer updates and 4771 warmup steps, with two rank-padding repeats and a 188-sample final global batch. Only DROID state statistics change; all action statistics and the other state statistics are equal to H32. CPU scheduler tests: 35 passed; launcher/fixture tests: 7 passed. The independently bound token audit passed with required_max_length941, and its actual hashes are pinned in the experiment config. The H10/GBS256 formal launcher dry-run passed.
- Actual GPU outcome: 64/1 failed with confirmed CUDA OOM before one completed update; its ranks exited and the authorized 32/2 candidate completed exactly two optimizer updates, saved state and exited zero. Losses3.13689/2.21586, zero truncation, peak allocated69.9262GiB/reserved73.9629GiB. Saved scheduler counters8/9 match global_step2 and saved DP4. Recovery preflight then failed because the new runner hides CUDA, causing DeepSpeed/Triton optimizer-state deserialization to raise `0 active drivers`. This launch-environment issue is recorded, not bypassed or treated as OOM. The failure-stop policy halted the runner at 2026-09-06 01:46:02 Asia/Shanghai; resume, formal AR and all Joint GPU stages remain unstarted. All artifacts and the complete traceback are retained, and no GPU process or automatic Joint continuation remains active.

## 第六次审核 P1：Stage05 AR resume scheduler 计数

- 日期：2026-09-05。
- 修改目的：修复 AR resume 将 optimizer global step 与底层 scheduler step 错当成一一对应，导致合法四进程保存状态被拒绝的问题。
- 涉及文件：`utils/stage05_checkpoint_contract.py::validate_stage05_resume_artifacts`；`tests/test_stage05_ar_scheduler_resume.py`；`tests/test_stage05_necessary_repairs.py::_write_ar_training_state`；`tests/test_stage05_launcher.py::test_ar_resume_dry_run_uses_own_purpose_and_validates_state`。
- 配置开关和默认状态：仅现有 `stage05_ar_resume` purpose 使用修正后的 scheduler 预检；其他用途保持原行为。完整性校验不提供绕过开关，没有新增训练配置或保存元数据。
- 实现来源：基于仓库已有 AR resume 校验适配；复用 DeepSpeed 已保存的 `mp_rank_*_model_states.pt::dp_world_size` 和 optimizer 分区信息。未修改保存或恢复 helper。
- 核实依据：本机 Accelerate 1.6.0、Transformers 4.57.1、PyTorch 2.6.0+cu124、DeepSpeed 0.15.4。`train_vla.py::train` 用 `calculate_warmup_steps` 和生产 cosine scheduler 将 warmup/总步数乘进程数；`Accelerator.prepare` 包装为 `AcceleratedScheduler(step_with_optimizer=True, split_batches=False)`。`run_optimizer_step_window` 每个 optimizer boundary 调用一次外层 scheduler，其内部按进程数步进；`utils/training_checkpoint.py::checkpoint_model_optimizer_scheduler` 原样保存底层 scheduler state。安装版 `DeepSpeedEngine._save_checkpoint` 同时保存 `dp_world_size`。当前 Stage05 配置是纯数据并行 ZeRO-2，故该字段就是此步进规则所需的保存时进程数。
- 具体设计和输入输出：AR 分支读取各 model/client state 中的正整数 `dp_world_size`，要求它与已有 optimizer 分区文件数一致；原 optimizer `partition_count` 校验继续生效。预期 `last_epoch = last_global_step * saved_dp_world_size`、`_step_count = last_epoch + 1`，不一致即报错。倍率不来自 scheduler 待校验计数或恢复机器的 GPU 数；optimizer moments 仍按原 global step 校验。Joint 分支倍率仍为 1，没有增加其状态检查。
- 兼容性和旧 checkpoint：已有 DeepSpeed checkpoint 使用原保存字段即可通过，不迁移或修改历史文件。若缺少或损坏 `dp_world_size`，明确拒绝，说明无法验证 scheduler 步进依据，不猜测倍率、不重置 scheduler。AR fresh、AR->Joint fresh、Joint/downstream、generic 合同和实际学习率调度均不改变。
- 验证方式：修复前使用实际包装器和生产 cosine 构造进行 CPU 复现，单进程通过，四进程语义在 global_step=1、last_epoch=4、_step_count=5 被错误拒绝。修复后 scheduler/AR 定向组 42 passed、13 deselected；覆盖独立修改 last_epoch、_step_count、global_step、共同修改两项计数，保存进程信息缺失/冲突，以及恢复后一步的完整 scheduler state、LR 和参数与不中断对照精确一致。每条 CPU 用例最多三个 scalar AdamW optimizer updates。Joint 分支固定原语义的回归通过；Python compile、Bash syntax 和工作树 diff 检查通过。
- Launcher 验证：AR resume 定向 dry-run 2 passed、9 deselected，分别读取单进程和四进程 scheduler 保存状态，并确认损坏 scheduler 在启动前被拒绝；launcher 代码未修改。
- 验证范围：测试只替换实际包装器的进程数查询，单进程 CPU 执行；fixture 使用生产 batch/GAS 配置和 scheduler 保存/读取调用，没有真实模型、数据或 DeepSpeed 分布式进程。不能据此宣称真实 DeepSpeed 多卡恢复通过。本轮不运行 GPU、多卡任务、完整测试、pilot、审计或正式训练；Joint/downstream 深度预检和 token audit helper 继续 deferred。

## Stage05 AR resume purpose 与 generic Expert 合同必要修复

- 日期：2026-09-05
- 修改目的：让 Stage05 `ar-resume` 使用独立恢复用途，并封闭直接 `ZR0Model(...)` 绕过已有 generic `action_expert_contract` 的入口。
- 涉及文件：`utils/stage05_checkpoint_contract.py` 的 purpose 注册、`validate_stage05_checkpoint_for_purpose`、`validate_stage05_resume_artifacts`、`validate_generic_action_expert_contract`；`model/reasoning_vla_model.py::ZR0Model.__init__/from_pretrained`；`train_vla.py::resolve_action_expert_config`；`utils/cli_options.py`；`scripts/run_stage05_four_dataset_pretraining.sh`。测试和验证脚本为 `tests/test_stage05_necessary_repairs.py`、`tests/test_stage05_launcher.py`、`tests/test_query_ar_joint_checkpoint.py`、`tests/stage05_cpu_smoke.py`；实验说明为 `docs/experiments/stage05_necessary_repairs_cpu/experiment.md`。
- 配置开关：显式 `--checkpoint_load_purpose stage05_ar_resume --resume_training --loss_type vlm`，launcher 的 `ar-resume` 自动选择该用途。purpose 默认仍为空；不增加模型、loss 或训练行为开关。
- 默认状态：AR fresh 和 AR->Joint fresh 的现有语义不变；generic 合同一旦存在就必须验证，不能通过省略 purpose、损坏为空值或 `allow_ar_warm_start=True` 关闭完整性检查。
- 实现来源：基于原仓库模块适配，没有外部实现。
- 原始实现位置与适配：复用 `utils/stage05_checkpoint_contract.py::_validate_contract_file_and_config` 的配置/manifest 哈希、已有 resume 文件读取和 generic 哈希校验；复用 `model/difference_query.py::resolve_difference_query_config` 校验 Query；复用 `utils/action_expert_config.py::load_action_expert_config/read_vlm_hidden_size` 解析配置；复用原 meta-device Expert shape 推导及 safetensors header 校验。没有修改这些 Query/配置解析实现。
- AR resume 设计：要求 checkpoint kind=`ar_only`、manifest/runtime loss=`vlm`，校验原 H、完整 Stage05 合同、manifest、DQ 配置/权重及可选外部配置一致性；要求无 Expert 权重。沿用既有 AR 导出合同中的未来 Joint 用途标识，新增的加载 purpose 独立为 `stage05_ar_resume`，恢复时不走 AR->Joint 初始化分支。校验 scheduler、DeepSpeed model/client state、ZeRO-2 optimizer/base state、FP32 partitions、参数组和 Adam moments，并核对 scheduler/client/optimizer global step；缺失、不完整或冲突均提前失败。新增训练状态深检查仅在 AR resume 分支执行。
- Generic 设计：直接构造和加载入口统一检查 raw/canonical/content/architecture/runtime hash、源及目标 H、VLM hidden size、action/state/embedding 维度；用解析配置推导完整 Expert key/shape，同时对比合同和 safetensors，防止伪造 shape 清单。模型使用校验器返回的配置对象，拒绝调用方额外架构偏移；权重加载保持 `strict=True`。已有 fresh horizon override 仍先验证源合同，再沿用原配置解析器返回有效配置，未改变通用 horizon 语义。
- 输入与输出：输入为本地 checkpoint、明确用途及可选请求配置；输出为已验证的配置对象，或在真实模型/Accelerator 分配前抛出错误。AR resume 随后继续原生产恢复路径；generic 校验不创建训练参数，只用 meta-device 获取结构。
- 与原有流程及关闭行为：不选择 AR resume 时不执行新增 AR 状态预检；无 generic 合同且无 Stage05 身份时保留原 legacy 路径。Difference Query、attention mask、AR/FM loss、adapter、采样、归一化、Sidecar、W&B、token audit v2、双视角和通用 horizon 实现均未修改。
- 验证方式：新增定向组 38 passed；已有 Stage05 contract/launcher 定向组 26 passed、5 deselected；generic fresh horizon 兼容回归 2 passed、10 deselected。旧 mock checkpoint 补齐 VLM config 并统一 mock Expert 的构造/shape 校验来源。故障注入覆盖 AR 合同及训练状态、generic 各哈希/hidden/维度/权重 key/shape，并用哨兵确认模型/Accelerator 未初始化。train 与 contract CLI help、launcher fresh/AR resume dry-run、Python compile、Bash syntax、`git diff --check` 通过。
- CPU smoke：真实生产配置解析、Tabletop adapter/processor、ZR0Model、optimizer-window 和 save/load 路径，使用 tiny Qwen/Expert，共两步 CPU optimizer 更新。AR loss=11.9112071599；保存 Stage05 AR 后 Joint 精确恢复 VLM/Query，fresh Expert/optimizer，AR loss=11.9110835825、FM loss=1.2347628730、total=18.0848979473。具体数据、超参数、两次执行记录和 checkpoint 路径见实验说明。
- 已知限制与 deferred：本轮不处理 Stage05 Joint resume 的 DeepSpeed 训练状态深度预检、downstream resume 的训练状态预检、token audit helper 的默认 trusted spec。未运行完整测试套件、GPU/四卡 smoke、正式训练、完整数据扫描或 LIBERO rollout；CPU smoke 不等同于真实 DeepSpeed 动态恢复。首次 CPU 尝试因可选 DeepSpeed/Triton 导入在零步时失败，验证脚本内隔离该依赖探测后两步通过，生产依赖探测未变。

## Tabletop Joint 初始化的 LIBERO DQ32 微调入口

- 日期：2026-09-04
- 修改目的：在两阶段 Tabletop v3 Difference Query 预训练完成后，用最终 Joint checkpoint 的 VLM、32 个 Query 和 Action Expert 完整初始化 LIBERO action-only 微调，同时严格复现既有 DQ32 LIBERO 实验的微调超参数。
- 涉及文件：`scripts/run_libero_wo_ecot_pt.sh` 的 `difference_query_pretrained` 实验臂；`scripts/run_libero_finetune_after_pretrain.sh` 自动串联；`scripts/preflight_libero_wo_ecot_pt.py` 的 checkpoint/output 门禁；`scripts/record_libero_finetune_launch.py`；`accelerate_configs/libero_zero2_bf16_mbs16_gas1.yaml`；`tests/test_libero_wo_ecot_pt_launcher.py`；独立实验说明。
- 配置开关：只有显式选择 launcher 实验臂 `difference_query_pretrained` 才启用完整 Joint warm start、独立 GAS=1 配置、严格 W&B 和扩展门禁；`ZR0_PRETRAIN_JOINT_CKPT` 仅用于显式覆盖默认的正式 `step-19424` 路径。原三个实验臂默认行为不变。
- 默认状态：launcher 仍默认 `baseline_fa2`；不选择新实验臂时不会读取 Tabletop checkpoint、加载预训练 Action Expert或使用新增配置。
- 实现来源：直接复用仓库原有模块并扩展现有 launcher/preflight；没有修改模型结构、Difference Query mask、Flow Matching、数据 adapter 或训练循环。
- 原始实现位置或外部来源：复用 `model/reasoning_vla_model.py::ZR0Model` 已有的双 checkpoint 加载和 `model/difference_query.py::resolve_difference_query_config`；复用 `train_vla.py::resolve_action_expert_config` 将 LIBERO horizon 10 写入相同架构；复用 `scripts/run_libero_wo_ecot_pt.sh` 的训练/resume/W&B/文档流程以及 `scripts/record_query_pretrain_launch.py` 的文件 hash helper。
- 具体设计：fresh 微调把同一个 Joint checkpoint 同时传给 `--vlm_name_or_path` 和 `--action_expert_name_or_path`，但不传 `--resume_training`，因此加载完整模型权重后新建 AdamW/cosine scheduler。严格配置加载器先验证源 horizon 及全部容量，再仅在明确的 downstream fresh 路径用显式 H_ft 构造有效配置；action/state/hidden/DiT 等其他字段仍严格一致，Expert 权重仍用严格 state-dict 加载。微调 checkpoint 保存有效 H_ft，后续 `--resume_training` 必须严格匹配该 H_ft 并恢复完整 action-only 模型和训练状态。初始化 manifest/stdout 同时记录源/有效 horizon、override 状态和源/有效配置 hash。专用 Accelerate/DeepSpeed 文件固定 world size 4、micro-batch 16、GAS 1、global batch 64；GAS 不传给 `accelerate launch`，训练 CLI、配置和 prepare 后 engine 各自显式校验。preflight 要求 source/resume checkpoint kind、VLM shards、Query/Expert sidecar、DQ32/SDPA、source horizon 和参考 Expert 架构一致，并检查磁盘空间。launch manifest 固化 Git HEAD、staged diff、launcher/config、VLM/Query/Expert 权重、dataset manifest、完整解析参数、命令和自然尾批 contract。自动串联脚本只轮询完整 Joint checkpoint 与每卡 70 GiB 空闲，不杀进程；随后在隔离目录完成真实 2-step fresh、恢复到 step 3 和有限性/恢复状态断言，只有通过后才启动正式输出。
- 输入与输出：输入为正式 Joint `step-19424` 和 `libero_wo_ecot_pt` 的 273,465 frames；输出写入全新 `...difference-query-nq32-tabletop-v3-joint-init` 目录。每 epoch 四 rank 对齐呈现 273,468 次、重复 3 帧、4,273 optimizer steps，尾步每 rank 15/global 60；8 epochs 共 34,184 steps。两路 256x256 RGB 按 `image,image2` 进入 224x224 Qwen processor；state/action 8/7 维按 q01/q99 归一化并补至 64，H=10 的尾部 mask 沿用原实现。
- 与原有流程的关系：训练仍为 `loss_type=action`、`total_loss=flow_matching_loss`、Expert loss weight 1.0；VLM/Query/Expert 全量训练，SDPA、Nq32、BF16、ZeRO-2、LR `2e-5 -> 2e-6`、8% warmup、AdamW `(0.9,0.95,eps=1e-6,wd=0.01)`、gradient clip 1.0 与参考实验一致。`action_horizon` 是不改变权重 shape 的运行时序列长度；跨 horizon 只允许 fresh Expert 权重迁移，AR-only、无 Expert 权重的显式配置和 resume 仍保持原严格检查。唯一实验变量是完整 Tabletop Joint 权重初始化；新 optimizer/scheduler 不继承预训练衰减状态。
- 关闭功能后的行为：不选择 `difference_query_pretrained` 时现有基座初始化、随机 Action Expert、原输出目录和 W&B group 均不变；不改变 server、rollout 或原 checkpoint。
- 验证方式：launcher dry-run 锁定 fresh/resume 权重来源、非 resume fresh 语义、唯一 train CLI GAS、global batch、DQ32/SDPA、action-only 和新 W&B group；YAML 测试锁定 ZeRO-2 batch contract；preflight 单测覆盖完整 Joint 接受、Action Expert 缺件拒绝和输出盘检查；正式启动前还要求真实四卡 2-step smoke、save、resume 第 3 步以及有限 loss/梯度/hidden/action。smoke 的 JSONL 验收只统计含 `total_loss` 的 optimizer-step 记录，明确忽略同文件中的 `wandb_finish` 生命周期事件。
- 已知限制：正式微调必须等待 Joint `step-19424` 完成，不能使用中间或 smoke checkpoint。训练期间没有 validation；LIBERO rollout 不在本入口范围。参考实验基于旧代码基准，而本实验会在 launch manifest 中记录当前代码和未提交改动，因此“配置一致”不等于代码 commit 完全相同。

## Stage05 四数据集 Difference Query 混合预训练

- 日期：2026-09-04
- 修改目的：在已验证的 Tabletop Difference Query/Query-conditioned AR 基础上，以最小公共 registry + adapter + sidecar 接入 DROID、Household、Tabletop、RH20T，并严格拆分 AR 与 Joint 准入、统计、采样和 checkpoint 语义。
- 涉及文件：`dataset2feature.yaml`、`configs/stage05_four_dataset_action_expert.json`；`utils/stage05_{canonical,sidecar,dataset,roundtrip,checkpoint_contract}.py`；`utils/action_expert_config.py`、`utils/dataset_{spec,adapters,manifest,seen_tracker}.py`；`utils/load_training_dataset.py`、`utils/normalization.py`、`utils/optimizer_step_loss.py`、`utils/wandb_training_logger.py`、`utils/cli_options.py`；`model/qwen_vl_backbone.py`、`model/reasoning_vla_model.py`；`train_vla.py`、policy/server；`scripts/build_stage05_sidecars.py`、`scripts/audit_stage05_*.py`、`scripts/run_stage05_four_dataset_pretraining.sh`；现有两阶段 launcher 兼容修正；`tests/test_stage05_*.py`、`tests/test_action_expert_config.py` 与相关回归测试；本实验文档和数据准入报告。
- 配置开关：四库功能只在显式选择 `dataset_adapter: stage05_mixed_pretraining` 的四个新 dataset entry 时启用；AR/Joint 沿用 `loss_type=vlm/vlm_and_action`；DQ 沿用 `--use_difference_query --num_difference_queries`；loss 权重沿用 CLI；Joint 架构由显式 `--action_expert_config_path` 唯一提供；视频后端由 entry 的 `video_backend` 选择；W&B 由 `--wandb_failure_policy={best_effort,required}`、有界 queue/backoff 参数、`--wandb_finish_max_attempts` 和 `--wandb_finish_timeout_seconds` 控制。
- 默认状态：旧 Tabletop、LIBERO、Query-off、checkpoint 和普通 dataset entry 不进入新 adapter。Stage05 launcher 只接受身份校验通过的 token audit format v2，并从报告读取当前 `required_max_length=941`；公共入口默认不变。当前完整报告为 `audits/token_length_audit_v9_format2.json`，其 content/file/implementation hash 固定在 `configs/stage05_four_dataset_experiment.json`。W&B 默认 `best_effort`，远端 FIFO 上限 256、重试 1 到 128 step、结束最多 flush 2 次且共享 15 秒墙钟 deadline；本地 JSONL、TensorBoard、配置、manifest 和 checkpoint 是权威记录。显式 `required` 保留同步远端失败即停行为。
- 实现来源：原仓库适配 + 自定义新增；canonical 语义依据数据生成代码与官方字段定义，不从字段维数猜测。
- 原始实现位置或外部来源：复用 `utils/load_training_dataset.py` 的 ConcatDataset/episode-group sampler、`utils/training_tokenization.py` 的完整 target/终止 token 边界、`utils/normalization.py::min_max_norm`、现有 Difference Query/Qwen/Action Expert/checkpoint/optimizer-step logger。Molmo canonical 权威来源为 `/opt/data/private/lq/FD-ID-FlowVLA` commit `3824d36cdf76bf0a9d537635de92a38f3920e9a3` 的 `starVLA/dataloader/difference_query_eef.py`；RH20T 来源为同 commit 的 `lerobot_v21_to_v30/convert_rh20t_raw_to_v30.py`；DROID 字段语义固定到其官方代码 commit `33ae6a67274f36d2e29525b86f23a56616ef43a7` 的 `droid/franka/robot.py` 与 `misc/transformations.py`。
- 具体设计：AR sidecar 只按主视角、可信任务和合法非空 `train_data` 建索引，完全不读取 action/stats；Joint sidecar 在成功/异常过滤和 canonical 有效性之后，仅接纳 H=32 chunk 中 `FM_count>0` 的帧，文本只决定额外 AR。format-v2 manifest 对实际调用链逐文件记录相对路径和 SHA256：AR 覆盖 sidecar、target canonicalization/adapter 和 builder，Joint 再覆盖 canonical 转换；关键生成参数和运行依赖版本使用确定性 hash，拒绝 format/source/config/code stale。所有文件先写同文件系统的专用 unique incomplete 目录，完整校验 shape/count/hash/loadability 后以 rename 原子发布；目标存在不覆盖，并发只允许一个成功，异常只清理匹配专用前缀与 marker 的本次目录。四库继续用 mmap `.npy` eligible/validity/filter 索引和紧凑 episode Parquet。自然 mixer 每个 epoch 消费所有最终 eligible 帧，并按剩余帧数在 128-sample block 中分配。seen tracker 在每个 checkpoint 保存分库 seen/unique/duplicate/AR/FM 数量及 resume bitset。token audit format v2 另将长度结论绑定到四个实际 AR sidecar 的 manifest/generator/eligible-index/count、processor/tokenizer/chat-template/视觉配置与运行时源码，以及 target/message/双视角/chat-template/训练 tokenization 调用链；报告自哈希通过后仍须逐项重算外部身份，任何漂移都要求完整重审计。
- 输入与输出：统一 adapter 输出按 main->wrist 排列的一或两张 RGB 图、task、可选 `train_data/slot_data`、canonical state/action、temporal/dimension mask、dataset/stats key、episode/frame/global ID。Molmo 只投影 `first_view/wrist_image`，DROID/RH20T 只解码 `exterior_1_left/wrist_left`；从不读取 `second_view/exterior2`。RH20T wrist 偏差超过 100 ms 时退化成主视角。输出 canonical state 为绝对 `[x_m,y_m,z_m,roll,pitch,yaw,gripper_open]`，action 为原生下一步相对 `[dx,dy,dz,droll,dpitch,dyaw,gripper_open]`，每库独立 q01/q99 和严格 `stats_key`。真实 round-trip 脚本通过生产 `_canonical_chunk()` 和生产 normalize/clip/denormalize helper 统计有效元素；DROID/Household/Tabletop 全量零 clip，RH20T 全量 3,501,909 步在维 0..6 clip `[8,3,3,4,4,5,0]`，最坏不可逆误差 0.2932883，数学内部值检查与真实 clipping 结果分开报告。
- 与原有流程的关系：DQ 的 `[C,Q,T,P]` mask、SDPA、final RMSNorm Query hidden、Action Expert 只读 `[B,32,H]` Query 条件和 generate 限制未改变。AR-only 不构造/加载/执行/保存 Action Expert 参数；新的四库 Stage05 AR checkpoint 保存 `checkpoint_kind=ar_only`、metadata/contract/config schema version、完整未来 Joint 配置、原始文件与 canonical hash、VLM hidden、DQ 数、action/state/H、关联 resolved experiment manifest hash及合同自哈希。Joint launcher 在模型构造前重新计算 checkpoint 原始/canonical/contract/manifest hash，并要求显式外部配置的完整 canonical hash完全相等；只有通过后才按 seed 42 新建 Action Expert，Joint resume 才恢复完整状态。optimizer-step loss 继续对整个 GAS window 与全部 rank 的 AR token/FM element 分子分母分别归一化；无文本 Joint 样本用全 `-100` labels，AR 分子/分母/梯度均为零。W&B best-effort 远端积压有界且指数退避，结束阶段用 `time.monotonic()` deadline 约束 pending flush 与 `run.finish()`；超时将 run 标为 abandoned，并把 timeout/pending/drop/failure 终态追加到本地 JSONL/TensorBoard。阻塞 60 秒的 fake finish 在 0.15 秒配置下实测 0.150804 秒返回并由子进程正常退出；`required` 模式仍同步暴露远端异常。
- 关闭功能后的行为：不选新 entry 时数据列、视角数、stats、采样与模型输入保持原逻辑；关闭 DQ 时保持 Query-off；AR-only 不触碰 action 列或依赖；`slot_data` 只保留不参与 loss。通用旧 checkpoint 加载兼容规则保持不变；但旧 Tabletop AR checkpoint 缺少新合同，明确不能作为四库 Stage05 Joint warm-start。
- 验证方式：修改前 `263 passed, 3 skipped`；初版完整 `tests/` 为 `280 passed, 3 skipped`；第一轮 P2 定向测试为 `81 passed`。二次 P2 覆盖 Stage05 AR checkpoint 完整合同与任意 Expert 结构漂移、token audit format v2 的 sidecar/eligible/processor/chat-template/源码/自哈希身份漂移、launcher 940/941/1024 和旧 checkpoint 边界，以及 W&B 60 秒阻塞 finish 的隔离子进程 deadline；三组定向测试分别为 `12/20/17 passed`，最终完整回归为 `350 passed, 3 skipped, 6 warnings`，耗时 359.61 秒。v5 与 v4 的 eligible index、count、Joint stats 完全一致；真实 round-trip/clipping 报告内容 hash 为 `e4d9af7f1787486be180cfd35ba4833a6e6f835e8afc64156af51d63ccd9bb3d`。CLI help、AR/Joint dry-run、token audit validate-only、Bash syntax、staged Python compile 和 staged diff 检查均纳入本轮验收。
- 已知限制：本轮不混入通用 VQA/VL 数据，和原始 ZR-0 预训练协议不同，存在 VLM 灾难性遗忘风险。DROID release 没有逐相机 capture timestamp，不能得到主/腕真实 skew。DROID/Household/Tabletop 的真实误差 percentile 是确定性分层样本，但其全量零 clipping 结论由全量 min/max 单调性严格推出；RH20T clipping 为全量统计。按本轮要求没有启动 GPU smoke/pilot/formal，既有资源门禁结论未重新采集。

## Tabletop v3 DQ32 两阶段预训练基础设施

- 日期：2026-09-03
- 修改目的：修复训练 CLI 缺失 GAS 导致正式入口必然在 Accelerator 初始化前失败的问题，并为指定的 4-GPU、GBS 128、AR-only 到 Joint 实验补齐可追溯的运行尺度、视觉输入 contract 和本地诊断。
- 涉及文件：`accelerate_configs/accelerate_config.yaml`、`utils/cli_options.py`、`train_vla.py`、`utils/load_training_dataset.py`、`utils/optimizer_step_loss.py`、`utils/training_tokenization.py`、`utils/dataset_adapters.py`、`utils/dataset_spec.py`、`utils/dataset_manifest.py`、`utils/wandb_training_logger.py`、两阶段 launcher、launch recorder、相关测试与实验文档。
- 配置开关：`--gradient_accumulation_steps` 默认 1；`--expected_global_batch_size` 默认不校验；`--logging_steps` 默认 10；`--log_training_diagnostics` 默认关闭。关闭诊断时不计算分模块 norm、吞吐、显存和数据质量扩展指标，不改变 optimizer、loss 或更新。
- 默认状态：保持单 optimizer group 和原 loss。正式配置将 per-device micro-batch/GAS/world/global 明确固定为 `16/2/4/128`；DeepSpeed config、外层 `accelerate launch`、训练 CLI 和 prepare 后 engine 必须是同一 GAS 整数，禁止 `auto`，任一不一致立即退出。普通非 launcher 调用仍保留 CLI 的 GAS=1 默认值。
- 实现来源：基于仓库现有 optimizer-step window、Accelerate/DeepSpeed ZeRO-2、v3 adapter、resolved dataset manifest 和 W&B logger 做最小适配；没有修改 Qwen forward、Difference Query mask、final RMSNorm 或 Action Expert/Flow Matching 模型结构。
- 具体设计：prepare 前后分别打印并硬校验 world size、micro-batch、GAS 和 nominal GBS。`EpochGroupedDistributedBatchSampler` 先构造全局 episode-grouped 顺序，只补到 world size 的倍数，再按 rank stride 切分；310,743 unique frames 每 epoch只确定性重复 1 帧，4 rank 各 77,686 个 sample，最后一个 batch 各 22 个 sample，形成 2,428 个 optimizer steps 和真实 global tail 88。`DataLoaderConfiguration(even_batches=False)` 禁止 Accelerate 补满尾批；AR/FM 继续使用该窗跨 rank 的真实有效元素分母。诊断按参数名把互斥且无遗漏的 trainable 参数归属为 VLM、Difference Query 或 Action Expert；ZeRO-2 在 engine 清空 gradient 之前从 partition/averaged gradient 累加 FP32 平方和，只 all-reduce 三个标量，不改变 optimizer group、step 顺序或超参数。
- 输入与输出：v3 adapter 为真实 Qwen processor 解析 `vision_input_contract`，记录三相机顺序/原始尺寸、224x224 bicubic、归一化、patch/merge、`image_grid_thw`、每路49/总147视觉 token和 pixel shape；resolved manifest version 4 将其写入 output 和每个普通/DeepSpeed checkpoint。训练另写 `initialization_manifest_{fresh,resume}.json`、逐 optimizer step 的 `training_metrics.jsonl`，launcher 将 stdout/stderr 同步追加到各阶段 `train.log`。
- 与原有流程的关系：现有 AR-only 不构造或保存 Action Expert 实例/权重；AR checkpoint 中的 `action_expert_config.json` 仅保存 Joint 后续随机构造所需的架构参数。AR warm-start Joint 只加载 VLM/Query，Joint resume 恢复完整状态；joint resume 在启动前要求 checkpoint 同时具备 Query、Action Expert、scheduler 和 kind metadata sidecar。数据读取 transient retry 仍只重试同一样本；adapter 额外返回 retry count 用于日志。268,986 partial-overlap frames 只写入 interval 审计，绝不参与 action mask 或采样。
- W&B 行为：该条目最初要求 online `wandb.init` 失败阻止正式启动；2026-09-04 的四库实验已将全局默认修订为 `best_effort`，stdout、JSONL、TensorBoard、配置、manifest 和 checkpoint 为权威记录，初始化/网络失败会告警但不停止训练。需要旧行为时显式传 `--wandb_failure_policy required`。
- 关闭功能后的行为：不传新参数的旧调用使用 GAS 1、无 GBS assertion、10-step 日志和无扩展诊断；无真实 image processor 的测试/legacy adapter 不生成 vision contract。所有新增实验功能均由 launcher 显式启用。
- 验证方式：自然尾批测试覆盖全部六组 `micro/GAS=(1,32),(2,16),(4,8),(8,4),(16,2),(32,1)`，均得到 310,744 次呈现、1 个重复、2,428 steps 和 tail 88；诊断开关比较单 optimizer group、loss、grad 和参数更新。真实 processor 验证三路 grid 和147视觉 token。真实四卡 Qwen3-VL-2B AR/Joint 在 micro 1/2/4/8/16/32 均完成 optimizer step；最终 `micro/GAS=16/2` 完成两阶段 fresh、save、完整 optimizer/scheduler/global-step resume 和第二步更新。最新 `tests/` 为 261 passed/3 opt-in skipped，显式 CUDA/ZeRO-2 为 5 passed。launch recorder 已对单文件与 HF sharded safetensors 身份解析做测试，并在真实三 shard checkpoint 上验证。
- 已知限制：Action Expert 没有 activation gradient checkpointing；本地模型只可标识 ModelScope `master`，因此使用权重 SHA256 作为不可变身份。Linux kernel 5.4 低于 DeepSpeed 建议的 5.5，但全部真实 smoke 未发生 hang。micro 32 的短 smoke 峰值 reserved 为 AR 55.260 GiB、Joint 56.838 GiB，但第一次正式 AR 在 step 10 达到 69.824 GiB 并在 step 11 OOM；因此正式重试只将配置调整为 micro 16/GAS 2，最终 smoke peak reserved 为 AR 41.994 GiB、Joint 44.512 GiB，有效全局 batch 保持128。

## 第五轮定向修复：optimizer-step 全局目标与训练部署契约

- 日期：2026-09-03
- 修改目的：使 AR/FM 的实际反向目标与日志都严格等于一个 optimizer step 内所有 data-parallel rank、全部 micro-batch 的各自有效元素均值，并把 v2 observation history、VQA/VLA 监督能力和 v3 direct-only 限制纳入 checkpoint/policy 契约。
- 涉及文件：`utils/optimizer_step_loss.py::global_supervision_counts/scaled_microbatch_loss/OptimizerStepMetricAccumulator`；`train_vla.py::iter_optimizer_step_windows/run_optimizer_step_window/train`；`model/reasoning_vla_model.py::ZR0Model._loss_outputs/forward`；`model/flow_matching_action_head.py::masked_loss_sum_and_count`；`utils/dataset_spec.py::ObjectiveRequirements/ObservationContract/resolve_objective_requirements/resolve_dataset_spec`；`utils/dataset_manifest.py`；`utils/load_training_dataset.py::build_concat_streaming_dataset`；`policies/reasoning_vla_policy.py::ZR0Policy.__init__`；`server.py`、`utils/cli_options.py`、`accelerate_configs/accelerate_config.yaml` 及定向测试。
- 配置开关：复用 `loss_type`、两个 loss weight、`gradient_accumulation_steps` 和 `window_size`；新增保守兼容开关 `--allow_legacy_checkpoint_without_observation_contract`，默认关闭。该轮曾将 DeepSpeed GAS 设为 `auto`；当前正式入口已按本文件顶部 2026-09-03 条目改为 config/launcher/CLI/runtime 同一显式整数。
- 默认状态：每个 optimizer-step window 先只从 batch tensor 统计 AR shift 后有效 label 数与 action mask 数，不保留计算图；随后逐 micro-batch forward/backward。action-only 的全局 FM count 为零立即失败；joint 全 VQA window 的 FM 为图连接的有限零。每个 window 创建独立 detached 日志 accumulator，optimizer step 后即释放。
- 实现来源：基于仓库现有 Qwen causal loss、Flow Matching 逐元素 MSE、Accelerate 1.6.0 和 DeepSpeed 0.15.4 调用路径做最小适配；optimizer-step reducer 与 observation contract 为本仓库自定义实现。没有修改 Qwen decoder、Difference Query 或 Action Expert 结构。
- 原始实现位置或外部来源：AR numerator 复用 `Qwen3VLForConditionalGeneration` 返回的 local mean，并乘与 HF shift 完全一致的 `labels[...,1:] != -100` count；FM numerator 复用 `FlowmatchingActionHead.forward` 的逐元素 MSE。Accelerate 非 DeepSpeed `backward` 会除以 GAS；DeepSpeed wrapper 调用 `engine.backward/step`，由 engine 按 GAS 缩放并执行 ZeRO 边界、clip、optimizer 与 scheduler。
- 具体设计：设 data-parallel world size 为 W、配置 GAS 为 G，完整 optimizer window 的跨 rank AR/FM 分母分别为全局有效 token/action element count。每个 micro-batch 传给 `accelerator.backward` 的值为 `G*W*(lambda_AR*local_AR_sum/global_AR_count + lambda_FM*local_FM_sum/global_FM_count)`。Accelerate/DeepSpeed 恰好除一次 G，DDP/ZeRO 梯度归约恰好平均一次 W，最终梯度等于两个全局均值的加权和；AR 与 FM 不共享 denominator，权重各应用一次。末尾不足 G 的 window 仍用 G 抵消框架固定缩放，并显式把最后一个实际 micro-batch 设为同步/DeepSpeed accumulation boundary。代码不直接读取或重缩放 `.grad`。
- 输入与输出：模型额外返回可微 local `ar_loss_sum`/`flow_matching_loss_sum` 与 detached count；原有 mean、weighted alias 和 `total_loss` 保持兼容。训练 logger 跨 window 与 rank 先汇总 detached numerator/count，再计算 raw/weighted/total；token/padding/truncation 指标同样聚合全部 micro-batch，invalid placeholder 不计入 count，min/max 通过 gather 计算。日志 `total_loss` 与本 optimizer step 的实际目标一致。
- 与原有流程的关系：optimizer、scheduler、global step、save/resume 顺序保持每个 optimizer boundary 一次；三种 loss 的公开定义和权重不变。v2 manifest 记录 versioned observation contract：`window_size`、旧到新的 frame-major 排列，以及由上一 policy execution horizon 决定的历史 stride；camera 内容与顺序仍由既有 manifest 字段单独锁定。v3 只允许 window 1 和 `direct_action`。action-only 在 processor、采样或数据读取前拒绝 VQA；joint 的 VQA requirements 只有 images/task/target，VLA 按 AR/FM 目标声明实际依赖，因而不改变 v3 AR-only 的懒加载隔离。
- 关闭功能后的行为：这些是训练目标与契约正确性修复，不提供恢复 micro-batch mean 累积或 action-only VQA 的开关。旧 checkpoint 缺 observation contract 默认拒绝；只有显式兼容开关开启时，才警告并采用用户当前明确给出的 contract。manifest resume 仍严格比较完整内容，包括 sample ratio；policy 只比较集中定义的部署语义字段。
- 验证方式：严格 RED/GREEN；真实 Accelerate CPU loop 比较 GAS=1/2 的 AR-only、action-only、joint、VLA+VQA、全 VQA，以及不足 GAS 的末尾 window，核对参数更新和 step 日志；两进程 gloo 覆盖 rank 有效数不同、一个 rank FM count=0 和跨 rank 重分配不变量。契约测试覆盖 v2 window 1/3 保存、resume/policy mismatch、legacy 显式 override、v3 window/direct-only、action-only VQA preflight 和 joint VQA/VLA 独立 requirements。
- 已知限制：该轮当时只执行 CPU/tiny 与两进程 gloo；后续真实 CUDA、Qwen3-VL-2B、NCCL 四卡和 DeepSpeed ZeRO-2 证据见本文件顶部 2026-09-03 条目。正式训练和 LIBERO rollout 仍未启动。

## 第四轮定向修复：尾部动作、全局 FM、推理 checkpoint 与 VQA 混合

- 日期：2026-09-03
- 修改目的：关闭 v2 episode 尾部重复动作监督、多 data-parallel rank masked FM 的 local-mean 偏差、action-only checkpoint 无法 direct inference、VLA+VQA joint 被逐样本 action 校验拒绝，以及旧 v2 metadata、padding 日志和长度审计三个边界缺陷。
- 涉及文件：`utils/load_training_dataset.py::_v2_action_temporal_valid/prepare_action_expert_inputs_cpu/StreamingLeRobotSampleDataset/VQADataset/custom_collate_fn`；`model/flow_matching_action_head.py::distributed_masked_mean/FlowmatchingActionHead.forward`；`model/reasoning_vla_model.py::ZR0Model._validate_training_inputs/from_pretrained`；`policies/reasoning_vla_policy.py::ZR0Policy.__init__`；`utils/dataset_adapters.py::LeRobotV3FutureDifferenceDataset._action_inputs`；`utils/dataset_spec.py::_resolve_v2_spec`；`utils/future_difference_audit.py::FutureDifferenceTokenMeasurer`；对应定向、checkpoint、policy 和 server 测试。
- 配置开关：没有新增训练功能开关。`loss_type`、loss 权重、`action_horizon`、manifest 和现有 `allow_legacy_checkpoint_without_manifest` 均保持原接口；`ZR0Model.from_pretrained(for_action_inference=True)` 仅由正式 action policy 加载入口使用。
- 默认状态：LeRobot v2 action/joint 样本必须携带上游生成的 `action_is_pad [H] bool`；VLA 样本显式标记 `action_supervision_available=true`，VQA joint 样本标记为 false 并使用全零 state/action mask。未提供该标记的旧模型调用按全 action-supervised 兼容处理。
- 实现来源：基于仓库原 `LeRobotDataset._get_query_indices` 已生成的 `<key>_is_pad`、原 masked FM、checkpoint kind 和 VQA dummy contract 做最小适配；data-parallel 全局均值与 sample-level availability 校验为本仓库自定义修复，没有修改 Qwen、Difference Query 或 DiT 结构。
- 原始实现位置或外部来源：复用 `lerobot/lerobot/common/datasets/lerobot_dataset.py::_get_query_indices/getitem_with_delta_timestamps` 的 episode clamp 与 `action_is_pad`；复用 `model/flow_matching_action_head.py::FlowmatchingActionHead.forward` 的逐元素 MSE；复用 `model/reasoning_vla_model.py::_read_checkpoint_kind/from_pretrained` 与 resolved manifest 校验。
- 具体设计：v2 action mask 固定为 `(~action_is_pad)[:,None] & dimension_valid[None,:]`，无效尾部 action 清零且不进入 FM；legacy FAST 只接收相同 temporal-valid slice，clamp 后重复末帧不生成文本 action token。`distributed_masked_mean` 以 FP32 累加 local sum、精确汇总全局 valid count；所有 rank 均执行 collective，反向标量使用 `world_size * local_sum / global_count`，前向数值用 detach 校正为全局 mean。global count 为零时返回与图相连的有限零。
- 输入与输出：joint 允许 `action_supervision_available=false` 的 VQA 样本，其 action/state/mask 均为形状兼容的零张量；VLA 样本仍必须有有效 state/action，action-only 拒绝无 action 监督样本。全 VQA batch 的 FM raw/weighted 分支为有限零，混合 batch 的 VQA 元素不产生 FM 梯度。action inference 显式接受 `action_only` 和 `joint` checkpoint 并加载 Action Expert；`ar_only`/legacy AR 在构造前报错，不随机初始化 Action Expert。
- 与原有流程的关系：三种 loss 公式和权重、AR-only 无 Action Expert、AR→joint fresh Action Expert、joint resume、checkpoint kind 完整性、Difference Query `[C,Q,T,P]`/mRoPE/SDPA/final RMSNorm、v3 target/action/q01-q99 均未改变。旧 v2 缺 `grounding_camera_keys` 时规范为空 tuple，存在时保留原内容与顺序。collate 裁剪后按实际 `attention_mask` 宽度重新计算现有 `padding_token_count`，并将这个可直接推导的 post-collate 指标标记为有效；长度审计仅在 `context_tokens > max_length` 时标记 input truncation，target/termination 完整性判断不变。
- 关闭功能后的行为：这些是正确性修复，不提供恢复尾部重复监督、rank-local 均值或静默 AR action inference 的开关。非分布式 FM 退化为同一 valid-element mean；无 VQA 的纯 VLA batch 保留原 loss 语义。
- 验证方式：严格 RED/GREEN；v2 partial-horizon/FAST/end-to-end、两进程 CPU gloo 不同 valid count 与单 rank 零 count、纯 VLA/纯 VQA/混合 availability、action-only/joint/ar-only 的 model-policy-server 加载、旧 v2 metadata、post-collate padding、context `< / = / > max_length` 和 termination-only overflow 均有定向测试；另运行 Difference Query、Query-off、三 loss、checkpoint、v2/v3、manifest、policy 和完整测试集回归。
- 已知限制：两进程 gloo 验证了 DDP 梯度平均语义；真实多 GPU NCCL、DeepSpeed ZeRO、Qwen3-VL-2B、正式训练、完整数据扫描和 LIBERO rollout 未在本轮运行。

## 第三轮定向修复：v2 完整监督、数据错误、零长度 logits 与混合指标

- 日期：2026-09-03
- 修改目的：阻止 v2 assistant content/termination 被部分截断后继续计算 AR loss；消除坏样本随机替换和 `None` 静默过滤；让 Query-off action-only 不投影任何 vocabulary token；补齐 grounding camera 与 mixed token 指标一致性。
- 涉及文件：`utils/training_tokenization.py`；`utils/load_training_dataset.py::tokenize_vision_language_inputs/StreamingLeRobotSampleDataset/VQADataset/custom_collate_fn`；`utils/dataset_adapters.py::tokenize_future_difference_message/LeRobotV3FutureDifferenceDataset`；`utils/dataset_manifest.py`；`model/qwen_vl_backbone.py`；`train_vla.py::batch_token_metrics`；对应定向和回归测试。
- 配置开关：继续使用外部 `max_length`；dataset entry 可选 `max_transient_retries`，默认 2，表示首次读取之外最多重试两次。没有新增静默 skip 开关。
- 默认状态：teacher-forced v2/v3/VQA target 始终先以 `truncation=False,padding=False` 完成一次 processor 调用，再验证完整 assistant content 和 chat termination 均落在 `max_length` 内；不完整即抛 `DatasetIntegrityError`。Query-off action-only 在 Transformers 4.57.1 conditional wrapper 上传空 LongTensor `logits_to_keep`。
- 实现来源：基于原仓库 v2/VQA loader 和前一轮 v3 精确边界实现适配；零长度 logits 直接使用 Transformers 4.57.1 `Qwen3VLForConditionalGeneration.forward` 官方 `Union[int,Tensor]` 索引接口，不复制 Qwen forward。
- 原始实现位置或外部来源：复用 `utils/dataset_adapters.py` 原 v3 target/termination 探针、Qwen chat template、`Qwen3VLForConditionalGeneration` conditional wrapper 和现有 final RMSNorm hidden 提取；公共边界、错误类型、重试器和 metric schema 为本仓库自定义抽取。
- 具体设计：`utils/training_tokenization.py` 是 assistant content/termination 边界与 token metric schema 的唯一实现。任何 `C+T>max_length` 均在 padding/collate 前报 dataset entry、episode/frame/sample、C、原始 content/T、termination、总长度、max length 和实际可保留 T。永久 schema/target/action/state/stats/图像错误立即失败；仅 timeout、EINTR、EAGAIN、ESTALE、ETIMEDOUT 可对同一 index 有限重试，耗尽后保留原 cause。v2、v3 与 VQA 均不随机换样本；collate 对 `None` 或空 batch 立即失败。
- 输入与输出：target 存在时 labels 只覆盖完整 assistant content 加标准 termination，P 全为 `-100`。所有 token metric 由 `TOKENIZATION_METRIC_SCHEMA` 定义，并配套 `<metric>_valid`；mixed batch 缺失项用同 dtype 零占位但 validity=false，真实零仍 validity=true。logger 只聚合 valid 样本，count 输出 mean/min/max，bool 输出 ratio。
- 与原有流程的关系：Difference Query C/Q/T/P、HF shift、三种 loss、AR-only Action Expert 隔离、AR→joint、action mask、q01/q99 和 checkpoint kind 不变。v2 legacy ECoT/FAST target 内容仍按原逻辑构造，只把随后可能发生的部分截断改为 fail-fast。Query-off 仍保留二维 mask、conditional wrapper、FA2/eager、视觉 scatter、DeepStack、mRoPE、PEFT hook 和完整 hidden states；AR/joint 继续正常计算 LM logits/CE。
- 关闭功能后的行为：本轮是训练数据正确性修复，没有允许恢复旧静默截断/换样本的开关；不选择 Query-off action-only 时不会使用空 logits 索引。
- 验证方式：严格 RED/GREEN；覆盖 1152→180 复现、超长 C/T、单词/JSON/termination 中途裁剪、刚好适配、v2/v3 同边界、永久与瞬时 I/O、VQA/None/空 batch、grounding camera 成员与顺序、mixed validity/masked aggregation、official `[B,0,V]` logits、final hidden/action golden、PEFT 和真实 processor+tiny vision scatter/DeepStack/mRoPE。
- 已知限制：空索引仍会调用 `lm_head` 一次，但输入 shape 为 `[B,0,H]`，输出为 `[B,0,V]`，不执行任何 token 的 vocabulary projection。真实 2B、GPU FA2、多 GPU、完整数据扫描、正式训练和 LIBERO rollout 均未在本轮执行。

## LeRobot v3 Future Difference 数据 Adapter 与 Token 边界

- 日期：2026-09-02
- 修改目的：在不改变既有 LeRobot v2 默认行为的前提下，增加可与 v2 混合训练的 LeRobot v3 future-difference 数据入口，并对 action horizon、量化归一化、AR target schema 和 token 截断实施 fail-fast 边界。
- 涉及文件：`utils/dataset_adapters.py` 的 `LeRobotV3FutureDifferenceDataset`、target/message/token boundary 和 registry；`utils/dataset_spec.py` 的共享 schema/q01-q99 解析；`utils/dataset_manifest.py`；`utils/load_training_dataset.py` 的 dataset builder、`EpochGroupedSampler`、旧 v2 `max_length` 透传和 `custom_collate_fn`；`policies/reasoning_vla_policy.py`；`utils/cli_options.py`、`train_vla.py`、`dataset2feature.yaml` 及对应测试。
- 配置开关：VLA entry 的 `dataset_adapter`，合法值为 `lerobot_v2`、`lerobot_v3_future_difference`；CLI `--max_length`、可选 `--dataset_sample_ratios` 和 `--dataloader_num_workers`；entry 可选 `max_transient_retries` 默认 2。
- 默认状态：已有 `dataset_type: vla` entry 未写 `dataset_adapter` 时使用 `lerobot_v2`；公共 CLI 保留原有 `--max_length=1200` 兼容默认值，正式模板要求显式提供审计后确定的 `MAX_LENGTH`，adapter 内不另设新的固定长度；`--dataset_sample_ratios` 默认 `None` 并使用 YAML ratio，显式值和 YAML 值都必须位于 `(0,1]`。只有显式选择 v3 adapter 才启用新数据流。
- 实现来源：原仓库适配 + 外部索引规则参考 + 自定义新增。
- 原始实现位置或外部来源：复用 `utils/load_training_dataset.py::tokenize_vision_language_inputs` 的旧 v2 路径、`utils/normalization.py::min_max_norm/min_max_denorm`、原 `StreamingLeRobotSampleDataset` 和 concat/dataloader 构造；v3 的 `steps_data_index.pkl` 与 `source_data_uri` 定位规则仅参考 sibling 仓库 commit `3824d36cdf76bf0a9d537635de92a38f3920e9a3` 的 `Flow-image-generation/stage06_flow/index.py`，未复制其训练架构或数据处理实现。registry、严格 schema、列级 LRU、loss 分支、mask 和 token 边界为本仓库自定义实现。
- 具体设计：初始化时严格读取并验证 `meta/steps_data_index.pkl`、`meta/stage05_episode_mapping.jsonl` 和 `meta/tasks.parquet`，按 `(episode, frame)` 通过 mapping 的 `source_data_uri` 定位真实 Parquet 行；每个 worker 的文件/列 LRU 上限为 1，避免同时驻留多个约 160 MB 的三相机 episode 表。action chunk 另以 Arrow scalar/column 构造单 episode 轻量 frame-action 索引，禁止为每个样本把含图像的整表 `to_pylist()`。`EpochGroupedSampler` 以 `seed+epoch` 确定性打乱 episode/sample 单元，episode 内按 frame 顺序读取；相同 seed/epoch 在 checkpoint resume 时重建相同次序，同时提高单文件缓存命中率。v3 未显式设置 worker 数时使用 4，legacy 数据保持原 24；正式模板显式 4、smoke 显式 1。v2、v3 和 VQA 的永久读取/数据错误直接带样本标识抛出，不随机换样本；白名单瞬时 I/O 只重试相同 index。`sample_ratio` 使用固定 seed 42，ratio=1 发布全部 310743 个 step。vlm 只选取三路当前图像、task 和 `train_data` 列且不打开 stats；action 只选取图像、task、state、actions 并不读取/解析 `train_data`；joint 读取两组字段。`build_future_difference_message` 是 v3 训练和 direct inference 唯一 user prompt builder，固定 camera 顺序、图像/文本排列、task `strip()`、`<TASK> ... </TASK>` 与 assistant generation prefix；训练只额外附加 canonical assistant target。真实同一样本两条路径 C 均为 177 token 且逐 token 相同，原位置 169 的 `<\TASK>` 差异已消失。
- 输入与输出：`train_data` 必须是非空 JSON object，键集合严格为 `Task_temporal`、`Spatial_motion`、`Contact_interaction`、`Object_constraints`，按该顺序输出紧凑 UTF-8 JSON，值均为非空字符串。action/joint 使用当前 state 和同 episode 的真实连续 `[t,t+H-1]` actions，`H=action_horizon` 为外部正整数参数，且不得超过 Action Expert `max_seq_len`；episode 尾部补零，不重复末动作。q01/q99 归一化后输出 `observation.state [1,64]`、`state_mask [1,64]`、`action [H,64]`、`action_mask [H,64] bool`，无效位置为零；adapter、manifest、Action Expert config 和 batch H 不一致时 fail-fast。正式模板和本 smoke 仍显式使用 H=32。
- 与原有流程的关系：concat builder 逐 entry 解析 adapter，允许 v2/v3 混合；`ObjectiveRequirements` 是 target/action/state/stats/FAST 依赖的唯一事实来源。v3 从不需要 FAST；v2 action-only 不需要 target/FAST；v2 AR-only 必须配置动作无关的预计算 `target_text_field`，否则在 dataset 构造前失败；满足该条件时不初始化 FAST、不加载 stats、不投影 state/action 列。v2 joint 的原 ECoT+离散动作 target 仍依赖 FAST，语义不变。主 VLM processor 仍由 builder 统一加载。外部 `max_length` 从 CLI 经 `train_vla.py` 传到 builder/adapter；公共 tokenizer 签名和默认 1200 保持兼容，但 v2/v3 teacher-forced 路径都先无截断编码再验证完整 target。legacy 样本集合仍逐 epoch 确定性随机化，v3 使用随机 episode 顺序加 episode 内连续 frame；这是为可恢复顺序和 Parquet I/O 明确引入的采样次序变化，不改变每 epoch 样本集合。collate 对共享 token metric 全量补齐，并以独立 validity mask 区分真实零与缺失占位。
- 关闭功能后的行为：不选择 `lerobot_v3_future_difference` 时继续使用原 `LeRobotDataset` 和 `StreamingLeRobotSampleDataset`；v2 action/joint 的图像、ECoT/action token 和 loss 行为保持旧语义，只有显式 AR-only requirements 会关闭无关 state/action/stats/FAST 依赖。未提供 ratio override 时 YAML sampling 行为不变。
- Token 边界：v2/v3 teacher-forced processor 均固定 `truncation=False,padding=False` 得到真实序列；assistant content token 必须作为精确连续 span 存在，assistant termination 由标准 chat template 的显式探针确定。T 定义为 content（v3 为 canonical JSON，v2 可为显式文本或原 ECoT/FAST 文本）加完整 termination；当前真实 Qwen3-VL template 的 termination token ID 为 `[151645,198]`，分别是 `<|im_end|>` 和结构性换行，两者均按标准 SFT 语义监督。P 只包含右侧 padding，labels 全为 `-100`。输出分别记录 `context_token_count`、`json_content_token_count`（通用含义为 content）、`chat_termination_token_count`、`target_region_token_count`、`padding_token_count`、原始/保留 content 数、有效监督数和截断分类及各自 validity。任何 content 或 termination 被裁均作为 `target_truncated` 带稳定样本 ID 和完整长度事实报错；不得切掉尾部继续算 AR loss。action-only 不输出 labels，target 指标 validity=false。
- 验证方式：严格 TDD；微型两 episode Parquet 覆盖 registry/default/unknown、loss 列隔离、无 stats AR、canonical 错误 ID、tail zero pad、跨 episode、stats shape/finiteness/span、7 维 norm-denorm、H=1/10/16/32、termination/P 边界、全 mask 防护、确定性采样、collate、v2 默认 tokenizer、v2/v3 mixed builder、requirements/FAST 条件加载、v2 投影列与 stats 访问 spy、ratio CLI/override。真实发布包使用本地 Qwen3-VL processor 对两个样本验证训练/direct C=177 且逐 token 相同、termination `[151645,198]`、T/P labels、HF shift 与 Query 不可见 T；真实 v3 policy 从匹配 manifest 构造并解析三路 camera、7 维 state/action 和共享 q01/q99。
- 已知限制：adapter 固定该发布包的 7 维 state/action 与 64 维模型 padding，但 action horizon 不再固定；已用实际 Qwen processor 和 policy metadata 路径验证，未加载 2B 模型权重或启动 GPU 训练。

## 共享 Dataset Spec、Policy 归一化与 Resolved Manifest

- 日期：2026-09-02
- 修改目的：保证训练 adapter、direct-action policy 与 checkpoint 对 camera/字段/维度/q01-q99/action horizon 使用同一份解析事实，并让恢复训练可检查数据语义漂移。
- 涉及文件：`utils/dataset_spec.py`、`utils/dataset_manifest.py`、`utils/dataset_adapters.py`、`utils/load_training_dataset.py`、`policies/reasoning_vla_policy.py`、`model/reasoning_vla_model.py`、`train_vla.py`、`tests/test_dataset_spec_policy.py`、`tests/test_dataset_manifest.py`。
- 配置开关：复用 entry 的 `dataset_adapter`、camera/state/action/target/stats 字段和外部 `action_horizon`；没有新增猜测字段名的 fallback。
- 默认状态：v2 继续从 `LeRobotDatasetMetadata` 读取原 camera/stats/schema；只有显式 v3 adapter 读取 `meta/info.json` 与 `meta/stats_gr00t.json::statistics.state/actions.q01/q99`。
- 实现来源：原仓库 normalization 工具适配 + 自定义共享 schema/manifest。
- 具体设计：`ResolvedDatasetSpec` 是训练与 policy 的唯一解析接口；v3 stats 只由 `validate_v3_quantile_stats` 实现一次，严格拒绝缺字段、非 7 维、NaN/Inf 或 `q01>=q99`。v3 policy 只取配置顺序的当前三路图像和真实 7 维 state，使用共享 state q01/q99 normalization；模型 action 输出先裁至 7 维，再由 `denormalize_actions` 使用共享 action q01/q99。禁止 sample/batch/episode 临时统计。v2 policy 保留 metadata 行为。
- 输入与输出：dataset builder 给每个 entry 生成独立 manifest，记录 entry/path/type/resolved adapter、task/target/state/action 字段、loss requirements、`camera_keys` 与 `grounding_camera_keys` 的各自顺序、维度、H、stats 相对路径/键/文件 SHA-256、规范化后的 state/action q01/q99、normalization、ratio、eligibility exists/used/source 和 data version。q01/q99 转为确定性 JSON 数组；manifest 以排序 JSON 计算 SHA-256 content hash，不写凭据或环境秘密。同一路径 stats 内容变化会同时改变 `stats_sha256` 和 manifest hash。
- 与原有流程的关系：主进程启动时打印 manifest，并保存到 `<output>/resolved_dataset_manifest.json`；每个 `step-*` 和 `latest-model-optimizer-lr` checkpoint 同样保存完整内容与 hash。`--resume_training` 对完整解析结果比较，entry 顺序、adapter、普通 camera 顺序、grounding camera 内容及顺序、字段/维度、stats identity/q01/q99、normalization、H、sample ratio 等变化均 fail-fast；sample ratio 被明确视为训练数据选择语义。resume 与 direct policy 共用 `DATASET_SEMANTIC_FIELDS`；policy 从 checkpoint 选择唯一同名 entry，并在 state normalization/action denormalization 前逐项校验当前 spec，mismatch 显示 checkpoint/current 值和 entry。新 checkpoint 缺 manifest 默认失败。legacy 仅能通过 `--allow_legacy_checkpoint_without_manifest` 显式 opt-in，并输出强警告。AR→joint 是 warm start 而非 resume，不做跨 loss manifest 相等要求。
- 验证方式：真实 v3 metadata 加匹配 checkpoint manifest 的 policy 构造、固定 normalized action 反归一化、v2 metadata 回归、单 v2/单 v3/mixed manifest、同路径 stats 原地变更、确定性序列化/hash、save/resume、policy 唯一 entry/legacy opt-in，以及 target/camera/stats/H 冲突测试。
- 已知限制：self-consistency policy 文件仍沿用其原有独立实现；本次推理修复范围是生产 direct-action `policies/reasoning_vla_policy.py`。

## Query-Conditioned AR 与联合训练目标

- 日期：2026-09-02
- 修改目的：把 teacher-forced future-difference AR、Flow Matching action 和两者联合训练定义为三个互斥的训练模式，保证 loss 权重、梯度路径、冻结模块和同步 optimizer step 的语义可验证且可恢复。
- 涉及文件：`model/reasoning_vla_model.py::ZR0Model.forward`、`model/qwen_vl_backbone.py::QwenVLBackbone.forward`、`train_vla.py` 的 optimizer/scheduler/step/logging helpers 与训练循环、`utils/cli_options.py`、`tests/test_loss_training_interface.py`、`tests/test_qwen_vl_backbone.py`、`tests/test_train_resume.py`、`tests/test_difference_query_sequence.py`、`tests/test_zr0_difference_query.py`。
- 配置开关：复用单一 `--loss_type {vlm,action,vlm_and_action}` 和 `--vlm_loss_weight`、`--action_expert_loss_weight`；新增可选 `--adam_beta1`、`--adam_beta2`、`--adam_epsilon`、`--warmup_ratio`，复用已有 `--max_train_steps`。不增加重复的 loss 布尔开关。
- 默认状态：默认 loss 模式和两个权重仍为原来的 `vlm_and_action`、`1.0/1.0`；AdamW 默认 `(0.9,0.95)`, epsilon `1e-6`，未给 warmup ratio 时保留旧 8% 且单进程最多 20,000 step 的算法。Difference Query 本身仍默认关闭。
- 实现来源：基于原仓库训练循环、Qwen causal LM loss 和 Flow Matching Action Head 修改。
- 原始实现位置或外部来源：复用 `Qwen3VLForConditionalGeneration.forward(labels=...)` 的原生 shift causal LM loss、`FlowmatchingActionHead.forward` 的 masked FM loss、`utils/training_checkpoint.py` 的 DeepSpeed 保存/恢复；没有复制外部训练实现。
- 具体设计：`vlm` 严格计算 `total_loss = vlm_loss_weight * ar_loss`；模型构造前同时要求 `tune_vlm=True`、`tune_action_expert=False` 且无 Action Expert 权重路径，并完全不实例化 Action Expert。`action` 严格计算 `total_loss = action_expert_loss_weight * flow_matching_loss`；`vlm_and_action` 为两项加权和。输出保留兼容键并增加 AR/FM/total 和 weighted alias。活跃权重必须有限且大于零。AR/joint 每个样本 labels 至少一个有效 token；action/joint 在 backbone 前检查 state/action/mask 与 Action Expert H 一致。Difference Query 配合 detach 时拒绝 action/joint。forward 的可选 legacy mode 参数只允许与构造期 `self.loss_type` 完全一致，`None` 使用构造值；任何动态切换在计算前报出 constructed/requested mode。checkpoint kind、dataset requirements、active components 和训练循环均以构造期 mode 为准。
- 输入与输出：AR/joint 的 Qwen labels 监督完整 assistant T（canonical JSON + `<|im_end|>` + 换行）；HF causal shift 由最后一个 Query hidden 预测第一个 T。AR 模型没有 Action Expert 参数。action/joint 的 Action Expert 仍额外接收独立 state encoder 的 proprioception，因此 Query 是“VLM 上下文 Query bottleneck”，不是全部控制条件的瓶颈。
- 与原有流程的关系：Query C/Q/T/P 重排和 block mask 不变：Q 只看 C/Q，T 只看全部 Q 和过去 T，不看 C/P/未来 T。Query-on action-only 使用 Qwen multimodal base model，避免 LM head；Query-off action-only 保持官方 conditional wrapper、二维 mask、FA2/eager、视觉 scatter、DeepStack、mRoPE 与 PEFT hook，但移除 labels 并传空 LongTensor `logits_to_keep`，LM head 收到 `[B,0,H]` 并返回 `[B,0,V]`，不计算任何 token 的 vocabulary projection 或 CE。AR/joint 不使用该优化，继续计算原生 CE。optimizer 只接收 `requires_grad=True` 参数。batch token 边界和截断统计在同步 optimizer step 的日志间隔记录到 TensorBoard/W&B。`max_train_steps` 同时限定 scheduler horizon、progress、保存和恢复位置，只在同步 optimizer boundary 增加 durable global step。
- 关闭功能后的行为：不启用 Difference Query 时仍走原二维 attention mask 和完整 VLM prefix 动作条件；不设置新增 optimizer/warmup 参数时保持旧默认。未选择 AR-only 时 Action Expert 路径按所选 action/joint 模式运行。
- 验证方式：测试覆盖三个 loss 公式和 1/5 权重、各分支有限非零梯度、AR 无 Action Expert 分配/参数/optimizer 项、冲突 CLI/构造拒绝、forward mode 不一致拒绝、非法权重与缺失监督、HF shift、C/Q/T/P、H 冲突、Query detach、action-only 无 labels、Query-off PEFT wrapper 与空 Tensor `logits_to_keep`、零长度/完整 logits 的 final hidden 和固定种子 action golden、同步 max-step/保存/恢复、token metric validity 和 masked 日志。
- 已知限制：Difference Query `.generate()` 仍明确抛 `NotImplementedError`；本任务验证 teacher-forced AR，不声称验证自由文本生成质量。真实 2B GPU smoke 因共享 GPU 满载未启动。

## Future Difference 全量只读审计

- 日期：2026-09-02
- 修改目的：在正式训练前量化 assistant target 长度与 `[t,t+31]` action chunk 对上游语言动作区间的覆盖关系，避免凭经验选择 `max_length` 或猜测 eligibility。
- 涉及文件：`utils/future_difference_audit.py`、`scripts/audit_future_difference_lengths.py`、`scripts/audit_future_difference_intervals.py`、`tests/test_future_difference_audits.py`。
- 配置开关：两个脚本均为独立只读命令；长度审计要求外部 `--max-length` 和 `--processor-path`，interval 审计要求外部 `--action-horizon`。训练不会自动运行审计。
- 默认状态：不修改数据、不影响训练默认行为，也不写 checkpoint；审计结果仅输出 JSON 到 stdout，进度写 stderr。
- 实现来源：自定义新增；复用 v3 adapter 的 canonical target、message builder 和精确 token boundary API。
- 原始实现位置或外部来源：episode/file 范围来自发布包 `meta/info.json` 与 episode metadata；interval 根目录由 `meta/stage05_merge.json::stage05_dir` 定位，new/old episode 由 `meta/stage05_episode_mapping.jsonl` 映射，上游 `training_samples*.jsonl` 只读取 semantic anchor 与闭区间字段。
- 具体设计：长度审计只读取 `episode_index/frame_index/task_index/train_data`，以 metadata 尺寸创建 dummy 图像，不打开真实图像/state/action；分别统计 C、JSON、termination、完整 T、保留/监督、投影总长和截断分布。interval 审计按闭区间规则分类，并固定 hash dataset 侧 `meta/info.json`、`steps_data_index.pkl`、`stage05_episode_mapping.jsonl`、`stage05_merge.json`，以及实际解析的 annotation `status.json`/`training_samples*.jsonl`；每项记录相对路径、字节数和 SHA-256，聚合为 audit identity。明确不 hash Parquet、图像或视频。
- 输入与输出：输入只涉及上述元数据和文本；输出为机器可读 JSON 汇总。范围固定为发布包全部 310,743 帧、1,881 episodes、14 tasks。当前 release 的 `meta/info.json::features` 不含 `training_eligible`，manifest 固定记录 `exists=false, used=false, source=unavailable_in_release`；YAML 声明不能覆盖实际 metadata。使用发布包全部可读取样本，未按 upstream `training_eligible` 过滤，也不扫描 Stage05、读取或推断 eligibility。
- 与原有流程的关系：审计不参与 sample selection。正式训练模板仍要求外部 `MAX_LENGTH`；必须根据本审计结果显式填写。
- 关闭功能后的行为：不运行脚本即没有额外 I/O 或计算。
- 验证方式：synthetic 闭区间/off-by-one、四分类、unavailable、只读快照、percentile、termination 截断和 metadata identity 变更测试 6/6。真实全量长度审计：310,743 样本在 1200 下 overflow/input-only/target truncation 均为 0；JSON p50/p99/max=471/521/566，termination 恒为 2，完整 T 与监督 p50/p99/max=473/523/568，投影总长 max=746。真实 interval：exact 374、full 41,383、partial 268,986、none/unavailable 0；identity 覆盖 3,766 个 metadata 文件，SHA-256 `cbcee48a48b3eb2d9bfb62cd846eccd1ec6af5d73065fe57cdeba7013c1b85f5`。
- 已知限制：长度结论绑定本地 `Qwen3-VL-2B-Instruct` processor、三路 224x224 请求尺寸和当前 canonical chat template；更换 processor/template/图像尺寸后必须重跑。interval 分类使用 padding 前 nominal 32 步区间，episode 尾部实际 action mask 另由 adapter 控制。

## Query AR/Joint Checkpoint 与启动模板

- 日期：2026-09-02
- 修改目的：提供可复现的 AR 一步+恢复、joint warm-start+恢复 smoke，以及在启动正式训练前强制核对训练规模、数据混合和论文参数的模板。
- 涉及文件：`scripts/run_query_ar_joint_smoke.sh`、`scripts/run_query_ar_joint_formal.sh`、`experiments/query_conditioned_ar_joint_smoke/experiment.md`、`tests/test_query_ar_joint_checkpoint.py`、`tests/test_query_ar_joint_launchers.py`。
- 配置开关：smoke 通过 `ZR0_*` 环境变量和四个显式 stage 名称控制；正式模板通过 `MODEL_PATH/OUTPUT_DIR/EXPERIMENT_DOC/MAX_LENGTH/EPOCHS/NUM_GPUS/PER_DEVICE_BATCH_SIZE/GRADIENT_ACCUMULATION_STEPS/DATASET_ENTRIES/SAMPLE_RATIOS` 与 W&B 环境变量控制，缺失即退出。
- 默认状态：脚本不会被训练代码自动调用；正式模板不预填 epochs、数据 entries 或 sample ratios。`ZR0_DRY_RUN=1` 只打印脱敏命令。
- 实现来源：基于仓库现有 Accelerate/DeepSpeed launcher 和 `utils/training_checkpoint.py` 适配。
- 具体设计：AR step 1/恢复只保存和加载 VLM、Query、optimizer、scheduler、global step 与未来 joint 所需 Action Expert config，不分配或保存 `action_expert.safetensors`。checkpoint metadata 显式记录 `ar_only/joint/action_only`；新 checkpoint kind 与用途冲突时 fail-fast，legacy 无 kind 才进入带日志的保守兼容路径。joint step 1 只把 `ar_only` checkpoint 作为 VLM/Query 来源，按保存 config 和固定 seed 新建 Action Expert；smoke/formal launcher 均验证 kind，正式 launcher 还验证 Query enabled 和 Nq=32。joint 自身 resume 必须把同一 `joint` checkpoint 同时作为 VLM/Query 与 Action Expert 来源，再恢复完整状态。其余 formal 参数约束不变。
- 输入与输出：输入为外部模型、数据 entry/ratio、审计决定的 max length 和实验文档；输出分别写 AR/joint smoke 或正式 output directory。smoke checkpoint 明确不是正式实验 checkpoint。
- 与原有流程的关系：继续使用现有 `latest-model-optimizer-lr` sidecar 和普通 `save_pretrained`；新增 `zr0_checkpoint_metadata.json` 与 resolved manifest。legacy checkpoint 保持可读，但不会通过缺文件静默猜测为新格式。
- 关闭功能后的行为：不执行脚本则无行为变化。
- 验证方式：tiny CPU 验证 AR 无 Action Expert 文件、VLM/Query 精确恢复、joint 从 AR warm-start 的 fresh Action Expert、kind 用途冲突、joint model/optimizer/scheduler/global step 完整恢复。launcher 17 项测试覆盖命令、必填变量、1024、ratio、W&B online、AR kind/Query sidecar、记录保留和四阶段路径；两个脚本通过 `bash -n`。
- 已知限制：该条目记录时真实 2B smoke 尚未运行；后续四卡真实 smoke/save/resume 结果见本文件顶部 2026-09-03 条目。正式训练仍未启动。

## LIBERO 评估运行说明

- 日期：2026-09-01
- 修改目的：为当前服务器已有环境补充可直接执行、可追溯且不会覆盖历史结果的 ZR-0-LIBERO 评估说明。
- 涉及文件：`simple_scripts/eval_libero.md`、`reference.md`
- 配置开关：无；本次只修改文档。
- 默认状态：不改变任何代码、模型或评估默认行为。
- 实现来源：直接复用仓库原有评估实现。
- 原始实现位置或外部来源：`server.py` 的 `deploy`；`evaluation/libero_eval/run_libero_eval.py` 的 `Args`、`eval_libero` 和 `_get_libero_env`；`policies/reasoning_vla_policy.py` 的 `ZR0Policy.infer`；`utils/image_tools.py` 的 `resize_with_pad`；当前仓库内 LIBERO 的 `libero/libero/envs/env_wrapper.py`。
- 具体设计：文档将运行拆分为已有资源确认、配置说明、启动前检查、四 GPU 服务端启动、四 suite 客户端启动、进度监控、结果汇总和按 PID 清理。每次运行创建新的时间戳结果目录，并记录模型来源与 revision、代码 commit、未提交修改、LIBERO 配置与 GPU 快照。
- 输入与输出：输入为官方 ZR-0-LIBERO checkpoint、两个相机图像、8 维机器人状态和任务语言；输出为 7 维动作 chunk，以及日志、rollout 视频和 JSON/CSV/Markdown 成功率汇总。
- 与原有流程的关系：只对已有 server-client 评估流程进行说明，不新增包装器，不修改数据流或启动参数。
- 关闭功能后的行为：不适用；文档不会被运行时代码加载。
- 验证方式：检查 Markdown diff、提取 Bash 代码块执行 `bash -n`、核对所有绝对路径、运行客户端 `--help` 核对 Tyro 参数，并对照代码检查 GPU/端口映射、图像处理、动作处理、随机种子、rollout 数和成功率口径。
- 已知限制：命令针对当前服务器的固定绝对路径和四 GPU 并行评估；换机器或换端口时必须同步调整服务端、客户端和 LIBERO 配置。

## 可开关 Difference Query 动作条件

- 日期：2026-09-02
- 修改目的：把可变长 Qwen 图文上下文压缩为固定数量 learnable Difference Query，使动作专家在实验臂中只读取 Query hidden，并提供无 Query 的 FA2/eager 基线和 SDPA backend 控制组。
- 涉及文件：`model/difference_query.py` 的 `DifferenceQuery`、`build_difference_query_sequence`、`resolve_difference_query_config`、`save_difference_query_artifacts`；`model/qwen_vl_backbone.py` 的 `QwenVLBackbone`；`model/reasoning_vla_model.py` 的 `ZR0Model`；`utils/cli_options.py`、`utils/training_checkpoint.py`、`train_vla.py`、`server.py`、`policies/reasoning_vla_policy.py`、`policies/reasoning_vla_policy_sc.py`；`scripts/run_libero_wo_ecot_pt.sh`；相关测试与文档。
- 配置开关：`use_difference_query: Optional[bool]`、`num_difference_queries: Optional[int]`、`vlm_attention_backend: Optional[str]`；CLI 为 `--use_difference_query` / `--no-use_difference_query`、`--num_difference_queries`、`--vlm_attention_backend`。
- 默认状态：关闭。无 Query checkpoint 且 CLI 未显式开启时，继续使用原二维 attention mask、原 Qwen `input_ids` 调用和原 FA2/eager 自动 backend；不创建 Query 参数，不增加 loss，不改变动作专家输入输出接口。
- 实现来源：基于仓库原有模块修改 + 自定义新增。
- 原始实现位置或外部来源：复用 `model/qwen_vl_backbone.py::QwenVLBackbone.forward` 的 Qwen forward/final RMSNorm 和 `model/reasoning_vla_model.py::ZR0Model` 的动作专家调用、保存/加载路径；复用 Transformers 4.57.1 `Qwen3VLModel.get_rope_index`、`inputs_embeds` 视觉 scatter、DeepStack 和 SDPA 四维 bool mask 支持。Difference Query 参数、C/Q/T/P 重排、block mask、三态解析和 checkpoint 完整性逻辑为本仓库自定义实现，没有复制外部项目代码。
- 具体设计：Query 参数为精确 `[Nq,H]`，隔离 RNG 后以 FP32 初始化，forward 转成 token embedding dtype/device。训练按首个受监督 label 重排为 `[C,Q,T,P]`，direct inference 为 `[C,Q,P]`；ID 0 只作 mRoPE/视觉对齐占位符，真实 embedding 由 Query 覆盖。序列索引和四维 block mask 使用广播、gather 和张量 region 映射构造，不逐样本调用 `.item()`；mask 实现 C causal、Q 读取 C/Q 且 Q 双向、T 只读取 Q/T causal、P 行列全屏蔽。final RMSNorm 后只 gather Query hidden 给动作专家。保存和加载均拒绝 NaN/Inf Query 权重；两个不同的本地加载目录必须同时为 legacy，或同时声明一致的 Difference Query 架构，禁止将单侧 Query checkpoint 与无 sidecar 的 legacy checkpoint 混合；双侧启用时权重也必须一致。同一路径会先去重，未提供 action checkpoint 路径时只检查 VLM 目录。
- 输入与输出：输入仍为 Qwen `input_ids/attention_mask/pixel_values/image_grid_thw` 和可选 labels；Query VLM 内部序列长度为 `L+Nq`。动作专家只收到 `backbone_embeddings [B,Nq,H]` 与全 True `action_expert_cross_attn_mask [B,Nq] bool`。Flow Matching、DiT、state/action encoder 和 decoder 结构不变；loss 组合由本次新增的三个显式训练模式控制。
- 与原有流程的关系：训练、普通保存、DeepSpeed 导出/恢复、`from_pretrained`、policy/server、direct-action 共用一个 Query 配置解析器。Query checkpoint 自动启用；disabled config 是明确架构声明，不能被 CLI 开启，且 hidden size 仍须匹配实际 VLM；只有完全没有 Query config/weight 的老 checkpoint 才允许显式开启后随机初始化。两个加载目录同时含 Query 时要求 config 和权重完全一致。Query 模式固定 SDPA并打印实际 backend；`baseline_sdpa` 在无 Query 下控制 backend 影响。Query-on action-only 无 labels时把完整 user prompt 视为 C，并调用 Qwen multimodal base forward 获取 final normalized Query hidden，不调用 LM head 或计算 CE。Query-on 的 `vlm`/`vlm_and_action` 仍走 conditional teacher-forced loss。Query-off 的 AR/joint 路径复用 `Qwen3VLForConditionalGeneration.forward` 并计算原生 CE；action-only 移除 labels 并设置空 Tensor `logits_to_keep`，沿用 conditional model 与 PEFT wrapper/hook，LM head 输入/输出序列长度均为 0。两条 action 条件路径的 final normalized hidden 语义在测试容差内一致，但计算量不同。Query 模式的 wrapper、实际 `backbone.model.generate()` 和 PEFT base model generate 均使用具名方法拒绝生成。
- 关闭功能后的行为：仍不创建 Query，保持二维 mask、完整 VLM prefix 动作条件、原 conditional wrapper 调用层级和 checkpoint 加载兼容性；action-only 不计算 CE，也不对任何 hidden token执行 vocabulary projection，但 conditional wrapper 仍会以零长度输入调用 LM head。`baseline_sdpa` 只改变 attention backend。新保存的 baseline checkpoint 只额外包含声明 disabled 的 `difference_query_config.json`。
- 验证方式：unittest 覆盖 CLI 隔离解析 `None/True/False/Nq`、disabled/enabled/损坏 checkpoint、双目录 Query/legacy 混用的双向拒绝及双 legacy/双一致声明/同目录/单目录允许矩阵、Nq=8/32/64 的 mask/真实 tiny Qwen/direct-action、隔离 RNG、最大长度 CPU C/Q/T/P mask、关闭路径 HEAD 参数/PEFT hook/final hidden/mask/固定种子动作 golden、Query-on action-only LM-head 拒绝、真实 tiny Qwen 的 T/C 因果隔离和冻结 Qwen 后 Query 梯度、PIL RGB 实际 processor/vision/DeepStack/mRoPE 对齐、tiny Qwen+tiny Action Expert backward/direct denoise、真实 Qwen+processor+Action Expert 完整 checkpoint round-trip、生成入口拒绝、单进程未初始化 distributed 构造与 step 同步。CUDA BF16 tiny Query 已捕获 efficient SDPA kernel 并记录有限输出/峰值显存；DeepSpeed ZeRO-2 已通过独立 `torchrun` world-size=1 和 2-rank 的更新、保存、重构、恢复及继续一步测试，复用生产 helper 并核对 Query sidecar、optimizer、scheduler、global step、跨 rank Query 一致性和有限性。另运行 CLI help/实际解析、三臂 launcher dry-run、Python/Bash syntax、`git diff --check`。未启动正式训练或完整 LIBERO rollout。
- 已知限制：Query 模式不支持 subtask/autoregressive generation；teacher-forced target 前向仍支持。Query+SDPA 相对 FA2 baseline 有 backend 混杂，必须联合 `baseline_sdpa` 解读。Transformers 4.57.1 的空 Tensor 索引可安全保持 conditional wrapper/PEFT/多模态语义并生成 `[B,0,V]`；它没有完全绕过 `lm_head` 函数调用，只是该调用处理零个 token。Query-off 与 Query-on action-only 的调用层级和计算量不同，运行时间或显存不能直接解释为 Query 本身带来的加速。四卡真实 Qwen3-VL-2B 短步生产恢复已在后续完成；正式训练和成功率评估仍未执行。

## 跨 epoch 恢复与单进程 step 同步修复

- 日期：2026-09-02
- 修改目的：修复恢复训练在后续每个 epoch 重复跳过相同 batch 前缀的问题，并使单进程训练不依赖已初始化的 `torch.distributed` process group。
- 涉及文件：`train_vla.py` 的 `should_skip_resumed_batch`、`synchronize_global_step` 和训练循环；`utils/training_checkpoint.py` 的生产保存/恢复与 PyTorch 2.6 safe-globals 注册；`tests/test_train_resume.py`、`tests/test_difference_query_cuda_zero.py`、`tests/zero2_checkpoint_worker.py`。
- 配置开关：沿用原有 `--resume_training`，没有新增默认行为开关。
- 默认状态：fresh training 的数据遍历、optimizer/scheduler 次序和 global-step 语义不变。
- 实现来源：基于仓库原有恢复循环修改。
- 原始实现位置或外部来源：`train_vla.py::train` 原有 `resume_epoch/resume_batch_idx` 计算、DeepSpeed optimizer 恢复和 scheduler state 恢复逻辑；没有使用外部实现。
- 具体设计：完整跳过 `epoch < resume_epoch`；batch 前缀条件收紧为 `epoch == resume_epoch and batch_idx < resume_batch_idx`，后续 epoch 从 batch 0 开始。optimizer step 后仍只在同步梯度边界由主 rank 增加 `global_completed_steps`；单进程直接返回该值，多进程仅在 distributed 可用且已初始化时从 rank 0 broadcast，否则显式报错。DeepSpeed safe-globals、engine optimizer/model 恢复、scheduler 恢复和 global step 提取集中在一个生产 helper，训练和门控测试共同调用，避免 PyTorch 2.6 反序列化列表漂移。
- 输入与输出：输入为恢复出的 global step、dataloader 长度、gradient accumulation 和当前 epoch/batch；输出为正确的数据遍历位置及所有 rank 一致的 global step。
- 与原有流程的关系：不改变 checkpoint 格式、scheduler state 加载、optimizer state 加载、数据 shuffle seed 或三实验臂参数，只修复恢复位置判断和单进程同步边界。
- 关闭功能后的行为：未启用 `--resume_training` 时不执行任何恢复跳过；单进程 fresh training 不再调用 `dist.broadcast`。
- 验证方式：跨两个 epoch 的恢复回归证明只在恢复 epoch 跳过前缀，并核对 global step 从 6 增至 12；单进程未初始化、多进程未初始化报错及多进程已初始化 broadcast 均有单元测试。ZeRO-2 world-size=1 与 2-rank `torchrun` 已实际通过一次更新、生产保存、重构恢复和继续一步。
- 已知限制：本次没有启动正式 DeepSpeed 训练；四卡真实 2B 短步生产恢复已在后续完成，见顶部 2026-09-03 条目。

## Optical Flow Auxiliary Head 与显式三阶段训练

- 日期：2026-09-06。
- 修改目的：接入训练期连续光流回归；此条记录原 OF-only 实现。其“Slot 未实现”状态已由下方 Structured Slot Heads v1 条目取代；Slot 关闭时仍保持 OF-only/provisional 行为及历史 checkpoint 兼容。
- 涉及文件：`model/optical_flow_aux_head.py` 的 `DenseRegressionFlowHead`/factory；`utils/optical_flow_config.py` 的配置、stage/Slot registry；`utils/optical_flow_reader.py::OpticalFlowReader`；`utils/optical_flow_loss.py`；`utils/stage06_dataset.py::Stage06LiberoDataset`；`utils/optical_flow_checkpoint.py`；现有 `ZR0Model`、CLI、dataset spec/collator、optimizer loss accumulator、训练循环与 checkpoint helper 的最小接入；`dataset2feature.yaml`、启动模板、审计/实验文档和对应测试。
- 实现来源：Head、OF reader/loss、集中配置与 Flow artifact 合同是自定义新增。现有模型没有稠密光流输出或 HDF5 对齐读取能力，不能直接复用 FM Action Expert 作为光流监督 Head。数据侧复用并适配 `utils/stage05_dataset.py::Stage05MixedPretrainingDataset` 的 `_source_data_path`、`_episode_rows`、`_video_path`、`_decode_video`；复用 `utils/load_training_dataset.py::prepare_qwen_vl_inputs_cpu`、`prepare_action_expert_inputs_cpu`、collator、episode sampler；复用 `QwenVLBackbone`、Difference Query mask/sidecar、`ZR0Model._loss_outputs`、`utils/optimizer_step_loss.py` 的全局窗口归一化、`utils/training_checkpoint.py` 的 DeepSpeed 保存恢复。不复制或修改 Difference Query attention mask/Action Expert 数据流。
- 外部来源：仅消费外部 MegaFlow pseudo-label，未复制模型代码。生成器版本 `ee5b61813db0a76ac0db9034899aade72a0d230c`、外部路径、生成权重 SHA256 和完整 manifest 来源见 `docs/optical_flow_data_audit.md`；生成端 resize 插值未验证为训练端相同，不据此宣称完全一致。
- 配置开关与默认值：`training_stage=None`、`optical_flow_aux_type=none`、`slot_aux_type=none`；OF loss weight=0；首次启用必须显式正 `num_flow_queries`、正 OF weight。默认 delta=10、resolution=56、grid=14、hidden=256、heads=8、layers=2、cell valid fraction=.5、motion weight=1、motion threshold=.01、epsilon=.001、flow seed=42。Stage1 禁止 OF/Slot/FM，stage2 必须启用 dense Head；stage3 必须启用 AR/FM，OF 可选。CLI 单独保留 `loss_type_explicit`，未传旧默认不构成 stage 冲突。
- 自定义结构与输入输出：只将 `[B,Nq,H]` 的末尾 N 个 Query 传入 Head；LN -> input projection -> learned 14x14 grid -> 默认两层 8-head cross-attention/residual LN/4x MLP -> bilinear 56x56 -> 3x3 Conv/GroupNorm/GELU -> 3x3 Conv 输出 `[B,2,56,56]`。输出 bias=0、weight std=.001 非零；H=2048 时总计 2,753,538 参数，参数量不随 Flow Query 数变化。初始化隔离 CPU RNG，不改 VLM/Query 或 CUDA RNG。Action Expert 仍读取全部 Query。
- 数据流：Stage06 registry 读取原始 LIBERO v3 parquet/video，按真实 `(episode_index, frame_index)` 查 manifest/HDF5；启动逐文件验证结构/字段/dtype/index/delta/camera/units/episode 边界，worker 独立懒加载 LRU handle，pickle 不带 handle。数据损坏抛 `DatasetIntegrityError`；未覆盖和尾部 identity/短 delta 返回 unavailable/None。collator 用稀疏索引字典保留真实标签，绝不补全零光流。默认不筛除帧；无同步 flow 几何变换时拒绝随机几何增强。源图像 256x256、实际模型输入 224x224，两相机按 registry 顺序；监督主相机。Stage06 原始数据无 AR 文本，joint 需混入有真实 AR 标签的数据。
- Loss 与统计：mask-weighted area 到 56x56，不除以4、不重归一化分量；float32 robust EPE=`sqrt(dx^2+dy^2+epsilon^2)-epsilon`，每有效样本计算 valid/motion mean 后按有效样本数平均。跨 rank/整个累积窗口聚合 numerator/count，局部空监督 graph-connected zero；stage2 全局空监督在 forward/backward/AdamW/scheduler 前同步跳过，不增加 global step。Stage3 保留 AR/FM。日志记录 EPE、motion EPE、zero baseline、有效/运动比例、有效样本/像素、coverage/skipped batches。详细公式见 `docs/optical_flow_training.md`。
- Optimizer 与恢复：沿用原有 AdamW 单组/LR/scheduler/weight decay 默认语义；增加 OF/Slot owner 接口，实际没有的模块不进 optimizer。Stage1/2 不构造 Expert；stage3 按 seed 新建 Expert，只有显式独立来源才加载。`init_from_checkpoint` 只加载 VLM/Query/已有 OF，从 step0 新建训练状态；`resume_from_checkpoint` 同阶段恢复完整模型、DeepSpeed optimizer/scaler、scheduler、每 rank RNG、sampler seed/批量合同及独立数据游标。数据游标包含已跳过 batch，global step 只计更新。新增阶段 sidecar 保存 OF 配置/权重 hash、stage 和初始化前后 VLM/Query checksum，CLI 冲突/缺配置/缺权重/shape mismatch 报错。旧 Stage05 source 合同在新 stage 初始化时仍验证。旧默认路径的合同规则不变。
- 关闭后的行为：旧三种 loss_type、Query-off/on、已有 DataLoader/训练逻辑保持；不读取 Stage06/HDF5，不创建 Flow Head，不增加 loss 或参数。推理不要求 flow root，直接动作路径不调用 OF Head；旧 checkpoint 无 sidecar 默认关闭。
- 验证方式：`tests/test_optical_flow_aux.py` 覆盖三阶段/Slot、CLI 显式来源、输入隔离、参数量、梯度、mask pooling、mixed missing、HDF5 LRU/worker/pickle/损坏、跨阶段 stale Expert 隔离、RNG/cursor、CPU Gloo 空 rank/global skip/全局归一化；旧 Query/loss/checkpoint/dataset 回归。`scripts/verify_optical_flow_cpu.py` 验证全部 1693 HDF5 结构/273465 frame 映射；真实 AR step-7284 的 625 VLM 张量和 Query 精确恢复。真实视频/processor 输出 `[1,14,14]` grid，即224x224。16真实标签+固定 synthetic Query overfit，EPE .0250344 -> .00419236，zero baseline .0100784；完整配置/初次失败与后续结果见 `docs/experiments/optical_flow_cpu/experiment.md`。
- 已知限制：Slot/VQ tokens/推理闭环未实现；稠密文件没有全量重算 SHA256（读取结构与少量标签，保留 manifest hash）；worker 随机状态不序列化，精确 resume 要求无随机增强的确定性 adapter。没有启动正式训练、GPU smoke、完整 VLM overfit、CUDA/DeepSpeed 动态 resume 或 LIBERO rollout；CPU Head overfit 不等于策略性能验证。

## 第四次审核：Stage05 checkpoint purpose、可配置下游 Horizon 与 token audit 可信规格

- 日期：2026-09-05
- 修改目的：封闭 Stage05 AR->Joint/Joint resume 配置合同绕过，支持从任意已验证源 horizon 以显式白名单 H_ft 初始化下游实验，并把 Stage05 token audit 绑定到版本化实验规格之外的报告文件与实现身份。
- 涉及文件：`utils/stage05_checkpoint_contract.py`、`utils/action_expert_config.py`、`model/reasoning_vla_model.py`、`train_vla.py`、`utils/cli_options.py`、Stage05/LIBERO launcher、policy loader；`configs/stage05_four_dataset_experiment.json`；对应 checkpoint、token audit、H=10 测试。
- 配置开关：`--checkpoint_load_purpose={stage05_ar_to_joint,stage05_joint_resume,downstream_finetune,inference}`；Stage05 同实验阶段保持 checkpoint horizon，只有 fresh `downstream_finetune` 允许显式设置合法 H_ft，且下游 resume 必须匹配已保存 H。H_pre/H_ft 不硬编码为 32/10，也不限制只能缩短。旧通用 checkpoint 路径保留严格兼容行为，但不能宣称 Stage05 合同；任何带 Stage05 身份却缺合同的产物按损坏 checkpoint 拒绝。
- 实现来源：基于既有 `ZR0Model`/Action Expert/checkpoint loader 的适配；architecture hash 为完整解析配置去除 `action_horizon` 后的确定性 SHA256，runtime contract hash 另绑定 source/target H、checkpoint kind、purpose、训练阶段、loss type 和 Stage05 manifest。加载先校验合同、配置、manifest、Query 和 Expert 权重，再构造模型；生产配置在 H=32/10/16/8 的 Expert 参数 key/shape 已逐项验证完全相同，统一使用 strict=True。Stage05 Joint 只接受生产 manifest 的 `loss_type=vlm_and_action`，AR-only 只接受 `vlm`。
- 下游语义：Stage05 Joint resume 必须同一 Joint checkpoint、原 H、完整 Expert 权重/optimizer/scheduler/step；下游 fine-tune 是新实验初始化，加载 VLM、DQ32、Expert 权重，创建新 optimizer/scheduler，不继承预训练状态，使用目标数据集自己的 stats_key/q01/q99，输出并恢复声明的 H_ft checkpoint，direct-action 每次返回 H_ft 步。
- Token audit 可信绑定：`configs/stage05_four_dataset_experiment.json` 固定 format v2、required max 941、报告 content/file SHA256 和 implementation identity SHA256；launcher 与独立 `--validate-only` 在未传 `--trusted-spec` 时都自动使用这份版本控制规格，禁止通过省略参数或环境变量绕过。sidecar 身份更新后已重新生成 `audits/token_length_audit_v9_format2.json` 并更新规格；LIBERO 不继承 Stage05 941 门禁。
- 验证方式：六项 Finding 定向生产入口组为 88 passed；全量 CPU 回归为 383 passed、3 skipped。覆盖真实 `vlm_and_action` manifest 经 `ZR0Model.save_pretrained` 保存/验证、purpose/resume 冲突、损坏合同/Expert safetensors/训练状态、H 32->10、16->8、8->16 和不变 H、token 规格/报告移动/自重哈希篡改、launcher 940/941/1024，以及实际旧 Tabletop step-19424 的 LIBERO H=10 dry-run。train/server help、三个 shell launcher 语法和 validate-only 均通过。
- 已知限制：本轮未启动 GPU、Stage05 多卡训练、真实 DeepSpeed 多卡 resume、LIBERO 训练或 rollout。Joint resume 的合同、safetensors、scheduler、DeepSpeed/client/optimizer state 与 global step 只完成 CPU 生产预检；不能据此宣称动态多卡恢复完成。若实际 checkpoint 的 `save_pretrained` 改写 processor 文件，必须对该 checkpoint 重新执行 token audit，严格 validator 会拒绝沿用 base-model 报告。

## Structured Slot Heads v1

- 本节记录首轮实现和当时验证。单集路由、损坏标签屏蔽、启动扫描 Flow mask 及历史 digest 替换等描述已经废弃；当前有效合同以文末第三轮修复记录为准。
- 日期：2026-09-06。
- 最终验证记录：完整相关回归 235 passed（6 项依赖/测试 scheduler warning）；随后补齐整数负损失系数校验，Head/loss 定向回归 40 passed，包含新增22项配置用例，计数有重叠。CLI help、四种 launcher dry-run、Bash 语法和 diff whitespace 检查通过。实际29个任务文件清单与展开命令见 `docs/structured_slot_training.md`。
- 修改目的：仅在训练期向前部 Difference Query 添加 Q1-Q9 结构化监督，支持 Slot-only 与 Slot+Flow；解决合并数据沿用历史标签、缺失监督和跨 rank/GAS 归一化问题。
- 涉及文件：新增 `utils/slot_config.py`（`SlotConfig`、registry、`resolve_query_layout`）、`model/structured_slot_head.py`（`StructuredSlotHead`、`slot_groups`、factory）、`utils/slot_labels.py`（`normalize_slot_labels`、`task_validity`）、`utils/slot_loss.py`（`slot_loss`、`finalize_slot_metrics`）、`utils/slot_supervision.py`（审计、`SlotSupervisionReader`、`SlotSupervisedDataset`）、`utils/slot_checkpoint.py`（三文件保存/读取/完整性验证）；接入 `ZR0Model`、`utils/cli_options.py`、`utils/optimizer_step_loss.py`、`train_vla.py`、现有 v3/Stage05 adapter/collator、Flow 配置/阶段状态/reader。新增审计脚本、CPU smoke、Slot launcher、ZeRO-2 YAML、固定统计和审计/训练/实验文档。
- 配置开关与默认状态：`slot_aux_type=none`、`slot_loss_weight=0`；启用 `structured_slots_v1` 需要 DQ 开启、至少七个 Slot token 和审计目录。原“Head 启用必须正外层权重”要求已由下方第四轮修复更新为允许零权重消融；Stage2 仍必须至少有一个正权重优化目标。首次分区须显式 `num_flow_queries`，Flow 关闭允许零；DQ32/F8 分组为 `[3,3,3,5,3,3,4]`。已有 checkpoint 分区优先，显式冲突报错；Flow 关闭仍保留尾部分区。
- 实现来源：Head、结构化标签/损失/固定统计/锚点校验为自定义新增；现有 Flow Head 的稠密二维回归无法直接输出多种 Slot 结构与独立 mask。工程接入基于原仓库适配，未引用外部论文或复制外部代码。
- 原始实现与适配：复用 `utils/future_difference_audit.py::_load_episode_mapping/_resolve_annotation_root/_active_training_samples/_data_paths` 定位原始样本；复用 `LeRobotV3FutureDifferenceDataset`、`Stage05MixedPretrainingDataset` 图像/动作/文本路径，Stage 2 允许只构建图像/task、丢弃 AR target，之后由共享 wrapper 附加标签。复用 `QwenVLBackbone` 已有 final RMSNorm Query 输出与 attention mask，不改 Query 双向注意力或 Action Expert 输入。扩展 `GlobalSupervisionCounts/scaled_microbatch_loss` 和 `run_optimizer_step_window` 独立统计九个任务分母；复用 `OptimizerWindowStep` 成功更新确认与训练 runtime 保存恢复入口。
- Head 结构与输入输出：前部 Slot Query 的七个连续组分别无参数均值池化，整数最大余数法分组，固定顺序打破平局；LayerNorm+Linear 输出 Q1 `[B,2]`、Q2 `[B,2,7]`、Q3/Q4 `[B,2,4]`、Q5 `[B,3,3]`、Q6 `[B,2,2]`、Q7/Q8 `[B,4]`、Q9 `[B,3,6]`。Q7/Q8 共享 LN；其他 Heads 独立。Q9 单个 Linear(H,18)，输出 bbox/risk/presence logit。Q1/Q6/risk 使用 sigmoid，bbox 为指定合法 xyxy 参数化。H=2048 实测 178,247 参数，测试校验小于 0.2M 且不随 Query 数改变，不硬编码估计值断言。初始化使用隔离 CPU RNG。
- 监督与数据流：源 `base_data` 校验 v5 schema、first_view、10FPS、语义起止时间、米制独立相机坐标。73,081 个源锚点与合并 Parquet 标签逐条一致；237,662 个沿用标签全部关闭 Slot mask。Reader 再核验源文件摘要和锚点集合，防止将相同沿用标签伪装成新锚点；每次读取标签核验 hash。未知非空类别携带 dataset/episode/frame 报错；null/缺失/非法 mask/非有限数值关闭对应监督。Q9 显式 mask=0 是 presence 负例，缺失 mask 不制造负例，presence/bbox/risk 独立有效性。标签、源标注、类别/Q5 统计不进入 prompt、Query 输入或 AR target。
- 采样：Stage 2 默认 `stage2_aux_sampling=slot_valid`；显式 `any_aux_valid` 可混入有效 Flow-only 样本，Flow reader 按已有 delta/source 与 56x56 pooling-mask 规则构建可用集合。Stage 3 保留原采样。Slot-off 完全保留既有 Flow-only 采样。原图/resize/相机排列、动作表示/horizon、已有 optimizer/scheduler 默认值均不改变。
- Loss：任务权重 Q1-Q9 为 `.20,.20,.04,.04,.04,.20,.12,.12,.04`。每项先样本内 masked mean，再按全局有效样本数平均，缺失项不重新分配权重。Smooth L1 beta=1；CE smoothing=.02，固定 sqrt 逆频率、最大比10、有效训练标签平均权重1。Q5 target/gripper/relative 各轴固定 mean/std，std 下限 .001m；Head 输出归一化位移，恢复米制后计算相对一致性 Smooth L1。单调权重 .1、GIoU .1、Q5 一致性 .05、Q9 presence/bbox/risk=.5/.25/.25；全部由集中配置及 CLI 控制并保存。所有 loss 与 reduction 使用 FP32。
- 分布式与日志：空 rank 返回连接所有 Head 参数和 Slot Query 的零；DDP/world-size/GAS 缩放沿用原实现。全局 Stage 2 无有效 Slot/Flow 同步跳过 backward、AdamW、scheduler/global step。各任务计数、可用状态、raw/weighted loss；分类混淆矩阵、RMSE 平方和、presence TP/FP/FN 全局求和后算指标；缺失指标不输出伪造数值。只有成功 optimizer 更新才累计 checkpoint `slot_loss_computed`。
- 阶段与兼容：Stage 1 仍 AR-only；Stage 2 不实例化 Action Expert，VLM 仅冻结参数而保留反向图，默认只训练 Query/启用辅助 Heads。Stage 3 默认训练 VLM/Query/Heads/新初始化 Expert。Slot 阶段须显式 `init_from_checkpoint` 或同阶段 resume；1->2 新建 Heads，2->3 恢复辅助 Heads；init 重建 optimizer/scheduler/step/RNG，旧 Expert 不隐式复用。保存 `slot_aux_config.json`、`slot_head.safetensors`、`slot_supervision_stats.json`，绑定 schema、统计生成规则、词表、边界、配置、hidden size/shape 与摘要；不完整、shape/词表/分区/CLI 冲突报错。兼容旧“Slot未实现”状态；沿用 ZeRO-3 拒绝策略。推理不实例化 Slot Head、不读取样本标签、不执行 Slot 预测。
- 关闭功能后的行为：不构造 Slot、不添加参数或 loss、不加载监督索引，不产生新 Slot artifacts；保留原 Query-off、AR-only、Flow-only、action-only、server 路径。
- 验证方式：`tests/test_structured_slots.py`、`tests/test_slot_integration.py`、`tests/test_slot_data.py` 覆盖分组/参数/输出、Q1-Q9 手算损失、独立 mask、锚点伪造、真实 tiny Qwen 输入隔离、冻结梯度、GAS 与双 rank Gloo 空 rank/空窗口、AMP 跳步、checkpoint/推理、四种 launcher dry-run；现有 Flow/Query/adapter/阶段/恢复/server 回归。真实 DQ32/H32 step-7284 与 Tabletop episode0 frame0/9 的 CPU Stage2/3 各两步，模型/Slot exact roundtrip，全部有限且 Query/Head 更新；详见 `docs/experiments/structured_slots_v1_cpu/results.json`。
- 审计结果：完整统计见 `docs/structured_slot_audit.json`；类别频次与先前计划一致，Q5 最大米制残差 `8.88e-16`；未发现非法正 bbox/contact/risk、mask、非有限值或非零 Q9 padding。Q9 三个位置正例为 `[39530,174,0]`，显式负位置179539。Slot 语义 tK 间隔1-50帧/中位8/P90=27，Flow 保留固定t+10，不添加跨 Head 同时域假设。
- 已知限制：真实 GPU smoke/Slot ZeRO-2 动态恢复及完整 benchmark 未运行（四张 A800 均被既有任务占用）；正式训练未启动。真实 CPU 联合 smoke 仅两个锚点，不证明收敛或任务成功率。Q9 第三位置无正例、Q5 有多米级独立估计极值；不额外裁剪。v1 审计限定完整 train-only 数据，运行需保留源标注可读以复核时间对齐。v3 的 aux 允许列表使 `dataset_adapters.py` 全文件 SHA 改变，旧 Stage05 sidecar 会被原有 generator identity 门禁判为 stale；不得修改摘要冒充兼容或覆盖旧产物。Stage05 新实验须独立重建 sidecar；原 Stage05 Joint resume 保留原代码/sidecar 合同。本文两条正式示例使用无该 sidecar 依赖的 v3 adapter。
- 已废弃的第二轮状态（2026-09-06）：曾替换 AR adapter digest 并仅检查部分 token 实现身份，且四集生产数据流未贯通。第三轮已删除这种摘要替换和部分校验，不得将第二轮 launcher 通过视为真实四集训练通过。

## 四数据集 Auxiliary 第三轮修复

- 日期：2026-09-07。
- 修改目的：修复独立审核 F1-F11，接通 partial DROID、Household、Tabletop、RH20T 的 Stage 2/3 Slot-only、Flow-only、Slot+Flow 六种配置。
- 涉及文件与接口：`utils/stage05_dataset.py::Stage05MixedPretrainingDataset`、`utils/optical_flow_reader.py::OpticalFlowReader`、`utils/slot_supervision.py::audit_slots/SlotSupervisedDataset`、`utils/slot_labels.py::normalize_slot_labels`、`utils/slot_loss.py::slot_loss`、`utils/optical_flow_loss.py::prepare_flow_targets`、`utils/dataset_seen_tracker.py::DatasetSeenTracker`、`utils/dataset_manifest.py`、`utils/dataset_spec.py`、`utils/slot_checkpoint.py`、`utils/load_training_dataset.py`、`utils/cli_options.py`、`train_vla.py`；新增辅助合同、路由、索引和兼容模块及产物/诊断脚本。
- 配置开关：复用 `slot_aux_type`、`optical_flow_aux_type`、各自 loss weight 和 checkpoint 自动恢复优先级；默认均关闭。launcher 使用独立 `WITH_SLOT`（默认1）与 `WITH_FLOW`（默认0），新增可选 `--aux_dataset_config` 指向 `configs/aux_four_dataset_v1.json`。Flow 关闭时不读取 Flow manifest、不打开 HDF5、不建索引；旧 options 仅补确实缺失的 Slot 默认字段。
- 实现来源：原仓库适配。模型、loss、collator、采样、全局 GAS/rank 计数、checkpoint 保存恢复均复用前述原接口。没有引入外部训练框架或复制模型。自定义新增的 `utils/aux_data_contract.py` 将原映射、manifest 内容摘要、HDF5 metadata 与源时间戳组成可持久化合同；`utils/slot_routing.py` 将已有单源 reader 组成严格按 dataset path/identity/camera 路由的集合；`utils/aux_sampling.py` 读取经过内容/来源验证的离线候选索引。原 reader 不支持四源路由和源身份恢复检查，因此需要这些独立模块。
- 输入输出和数据流：registry + 新产物 overlay -> 生产 Stage05 样本的真实 dataset/episode/frame -> 各源 reader -> 现有 collator -> 生产 Head/loss。Flow 样本携带 target、mask、actual/nominal delta、FPS、label source、排除原因；首访 episode 才验证真实帧映射和 HDF5 内容，成功后原子加入有界 LRU，异常清除缓存，worker 不共享句柄。稀疏与乱序帧不再使用行号伪造 frame_index。
- 监督语义：四源 nominal delta20；DROID15 FPS=4/3秒，其余10 FPS=2秒。原 LIBERO 默认 delta10 保留。loss 使用每样本已验证的 nominal delta；尾部短间隔/identity、缺标签和空有效 mask 单独计数。Slot 正 mask 下损坏数值、几何、结构或必需 mask 缺失报带字段路径的错误；合法 null/缺失子项和显式 invalid 仍屏蔽，Q1按字段存在性，Q9缺 mask 不制造负例。
- 统计：四集分别验证语义锚点与来源；Q2/Q7/Q8词表顺序固定，类别权重合并训练锚点计数生成；Q5均值/标准差按 `slot_dataset_index` 路由。聚合统计以版本2保存到 Slot checkpoint，包含每集完整版本1统计。wrapper 保留 manifest、源身份和过滤前长度，seen/anchor/Q1-Q9/Flow样本及有效像素/排除原因跨rank汇总。
- 采样与关闭行为：Stage 3沿用 Joint索引，不在启动时扫描 Flow mask。Stage 2 Slot-only 使用锚点；Flow-only 使用元数据候选；联合使用并集。候选规则为 nominal delta、source1和正 valid_fraction，是保守候选集，精确 pooled mask 在当前样本验证。原正式训练/评估图像、动作horizon、optimizer和学习率默认值未修改。
- AR兼容：`generator_identity` 始终报告真实文件摘要。`utils/stage05_compatibility.py` 和 `configs/stage05_ar_compatibility_v1.json` 仅接受经过审阅的完整历史/当前依赖清单对；任一文件或整体清单改变仍拒绝。历史 token audit、processor/data/truncation/sampling 校验保留，Joint不使用AR兼容例外。依据及复现实验见 `docs/structured_slot_review3.md`。
- 新产物：`scripts/build_auxiliary_artifacts.py` 复用现有 sidecar、Slot audit和精确 token audit工具，在 `/opt/data/private/lq/ZR-0-artifacts/structured_slots_review3_v1` 独立生成；只扫描必要标签/元数据，不全量解码图像或 Flow mask。历史产物和运行中配置保留。
- 验证方式：第三轮先复现失败，再修复；真实生产构造、batch、tiny backward、CPU全量回归、有限GPU/多GPU结果分别记录在 `docs/structured_slot_review3.md` 及 `docs/experiments/structured_slots_review3/experiment.md`。本条不将尚未完成的验证视为通过；详细状态以闭环记录为准。
- 已知限制：离线 Flow 候选不能保证所有样本池化后都有有效像素，训练时按精确 mask 排除且不进入分母。Slot 固定统计不裁剪原始位移极值。诊断使用 tiny模型，不证明2B模型收敛或任务成功率。本次不启动正式训练、不自动暂存、不创建commit。
- 最终索引合同：新增独立 `slot_v2` 索引及 `slot_sources_v2/slot_routes.json`，保留先前离线 audit。索引版本2将锚点内容、Stage05映射、源标注根目录和索引/normalizer完整实现摘要绑定到固定统计，记录原 audit 的摘要作为派生依据。源标注首次访问严格验证，成功后缓存最多4个episode；标签Parquet缓存最多2个。Stage3直接复用原索引及采样分组，不再逐帧构建辅助有效性列表。旧索引版本1的启动校验与伪造锚点拒绝测试继续保留。
- 最终真实验证：六种配置均构造四集生产dataset/dataloader，真实batch中每项启用的Q任务有效数4，Flow合格样本4/有效池化像素12498；Stage2冻结VLM、更新Query和启用Heads、无Expert，Stage3活动模块均更新。真实非锚点/尾部batch的Slot分母为零。两卡A800 BF16/ZeRO2联合模式在Stage2/3均完成有限更新、新进程模型/optimizer/scheduler/RNG/数据计数精确恢复；Stage2空rank、尾部空microbatch及全局空窗口检查通过。GPU确切复算使用诊断显式确定性算子，不改变生产默认配置。完整原388项、补充111项及新增32项的最终结果、文件身份与命令见 `docs/experiments/structured_slots_review3/results.json` 和 `commands.md`。

## 四数据集 Auxiliary 第四轮修复

- 日期：2026-09-07。目的：关闭 R4-1 显式数据路由被双 Head 关闭状态忽略，以及 R4-2 仅零权重 Slot 标签触发 AdamW 更新；独立验证真实生产 sampler 的新进程恢复。
- 实现来源：原仓库适配，无外部算法或新训练框架。修改 `utils/load_training_dataset.py::build_concat_streaming_dataset`，使显式 `aux_dataset_config` 始终选择数据/sidecar 路由，并拒绝空、错误类型和缺失的数据集路由。复用既有 Stage05 reader 开关，Slot/Flow 关闭时仍不读取标签 manifest、候选索引或 HDF5。未提供 overlay 的旧 Stage1 路径不变。
- 活动监督：扩展既有 `utils/optimizer_step_loss.py::GlobalSupervisionCounts/global_supervision_counts`，保留原分母和原始标签覆盖计数；新增按有效正系数判断的活动计数。Slot 同时考虑外层系数、Q1-Q9 任务系数，Q9 还按 presence/bbox/risk 的独立 mask 和系数取并集。AR/FM/Flow 使用各自当前目标和正系数。`train_vla.py::run_optimizer_step_window` 在完整 GAS 窗口跨 rank 求和后统一跳步，不根据 loss 数值决定更新。没有活动监督时不调用 forward/backward/optimizer/scheduler，不清空先前窗口状态；同一窗口其他 microbatch/rank 的活动梯度照常保留及同步。
- 原有参数拒绝合同：窗口入口复用 `ZR0Model._validate_loss_weight`，在跳步判断前执行已有 AR/FM 有限/非负及当前目标正权重校验，避免非法 AR/FM 零系数被当成合法无监督窗口。该验证未放宽原 Stage1/Joint 模型的拒绝条件。
- 配置兼容：`utils/slot_config.py::SlotConfig.validate` 允许 Head 启用且 `slot_loss_weight=0` 的消融，仍要求内部存在正任务系数、所有系数有限非负、关闭 Head 时外层权重为0。默认 `slot_aux_type=none/slot_loss_weight=0` 不变。原 AR/FM 配置验证、Stage2 双 Head 关闭拒绝、loss reduction、label masks、采样规则、图像/动作处理和 optimizer 默认值不变。
- 输出与日志：保留 `slot_Q*_valid_count`、`slot_sample_coverage` 和标签可用状态；独立记录 `slot_Q*_active_count`、`slot_active_supervision_count`、`ar_active_token_count`、`fm_active_element_count`、`flow_active_sample_count`、`active_supervision_available`。跳步窗口不伪造 loss；`optimizer_update_applied=false` 且 scheduler/global step 与 AdamW 动量/步数不变。正常监督恰好数值 loss=0 仍按活动监督执行更新。
- 产物：配置校验文件属于 Slot 索引身份依赖。使用原 `scripts/build_auxiliary_artifacts.py --phase slot-index --index-version 3` 派生独立 `slot_v3` 和 `slot_sources_v3`，逐项核对与 v2 的锚点、映射、类别权重和 Q5 统计完全一致，只新增当前完整实现身份；历史 v2、AR/Joint/Flow 产物保留。launcher 继续要求显式传入 Slot 目录，本轮命令传 v3。
- 验证工具：扩展 `scripts/verify_auxiliary_production.py` 为第七种 Stage3 双 Head 关闭模式，对不存在的辅助产物和 reader/HDF5 访问设置拒绝探针。新增 `scripts/verify_auxiliary_sampler_resume.py` 复用生产 loader/sampler/collator、训练窗口、runtime/checkpoint orchestration 与现有 `_TinyEngine` CPU checkpoint 测试适配器；生产数据无需固定样本文件，按真实 cursor 跳过后恢复 RNG，再比较后续样本身份、所有 batch 张量、loss、参数、optimizer/scheduler 和 seen 状态。CPU checkpoint 传输适配器不代表 ZeRO-2 验证。
- 测试：`tests/test_aux_review4.py` 覆盖零任务/外层/Q9 子项系数、空标签、已有 AdamW 状态不变、数值零 loss 的有效更新、GAS 前/中/后无活动 microbatch、双 rank 单侧无活动及全局空窗口、其他 Flow/AR/FM 目标有效更新、错误显式路由拒绝。最终全量回归和实际 sampler 范围以 `docs/structured_slot_review4.md` 及本轮实验文档为准。
- 独立 sampler 复现与修复：初次对照从指定 epoch 同时新建两个 prepared loader，暴露了验证不足；随后加入完整遍历前一 epoch 的小规模生产 sampler 对照，以及真实四集的实际 sampler epoch 断言，均复现 epoch1 被 Accelerate `DataLoaderShard.__iter__` 按默认 `iteration=0` 重设的问题。新增 `utils/load_training_dataset.py::set_dataloader_epoch` 同时通知原始 sampler 和 prepared loader，接入实际训练循环及诊断脚本；保留未包装 loader 的兼容行为和原 shuffle 算法。最终 epoch1 证据只采用这一修复之后的独立输出目录。
- 已知限制：第三轮 `verify_auxiliary_zero2.py` 的固定真实样本恢复证据只证明该固定输入的恢复；第四轮生产 sampler 验证单独记录。未授权完整2B、四rank或正式训练，本轮无相关就绪声明。
- 最终验收：最后一次源码修改后 561 个唯一用例全部通过（原531加新增30），无失败或跳过；历史 launcher12/12、token audit12/12、AR scheduler/resume33/33、necessary repairs38/38 保持通过。七种四集真实数据配置均完成 CPU BF16 tiny 更新，双 Head 关闭使用不可用辅助路径仍完成 AR/FM 更新。生产 sampler 的 workers0/2/4、GAS2/3、epoch0/1 三组对照在新进程恢复后，后续数据与 mask、参数及完整 optimizer/scheduler/seen 状态 rtol=atol=0 一致。四张 GPU 均被已有任务占用，本轮未执行 GPU/ZeRO-2；多 rank 真实 sampler、完整2B及四rank更新仍未验证。逐项闭环、完整命令、模块来源和文件摘要见 `docs/structured_slot_review4.md` 与 `docs/experiments/structured_slots_review4/{commands.md,results.json}`。

## RoboTwin Conda 环境与显式 CUDA、渲染检查

- 日期：2026-09-07。
- 修改目的：按用户要求安装独立 Python 3.10 仿真环境，补齐原安装脚本缺失的依赖清单，并验证 CUDA 与 SAPIEN 离屏渲染。
- 涉及文件：`evaluation/RoboTwin/script/requirements.txt`、`constraints.txt`、`check_render.py::check_render`、`check_cuda.py::check_cuda`、`.gitignore`；安装命令、环境路径、版本清单和结果见 `docs/experiments/robotwin_environment_20260907/experiment.md` 及同目录产物。
- 实现来源：依赖清单直接恢复自 RoboTwin stable_2.0 commit `13c3c47ff4312dd62484bcd51be034af55c062d1` 的 [script/requirements.txt](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/script/requirements.txt)。版本约束是对本仓库 `script/_install.sh` 的显式安装适配。PyTorch3D stable/0.7.8 来源 commit `75ebeeaea0908c5527e7b1e305fbc7681382db47`；CuRobo v0.7.8 来源 commit `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`。
- 配置开关：仅显式使用 `PIP_CONSTRAINT` 或 `pip -c script/constraints.txt` 才启用兼容版本约束；原 `_install.sh` 内容和默认调用不变。用户通过 `conda activate RoboTwin` 启用该环境的 CUDA/Vulkan/EGL 路径；退出环境恢复原变量。
- 渲染检查设计：复用 `envs/_base_task.py::setup_scene` 的 RT shader、32 samples/pixel、path depth 8、OIDN 与 1/250 秒物理步长；复用 `envs/camera/camera.py` 的 `get_picture("Color")` API。自定义合成方块用于独立验证，避免依赖尚未下载的任务资源；检查 RGB 有限、非空、方块可见，以及 200 步模拟后落地和图像变化。
- CUDA 检查设计：基于 [CuRobo v0.7.8 的 motion_gen_api_example.py](https://github.com/NVlabs/curobo/blob/d64c4b005459db10c5dd867d8b30a87d5bda9bdb/examples/motion_gen_api_example.py) 适配小型可达目标，直接复用 `MotionGenConfig`、`MotionGen.plan_single` 和自带 Franka 配置；用合成 ground mesh 验证 Warp 碰撞路径。自定义诊断入口同时调用 PyTorch3D `knn_points` 并与 `torch.cdist` 对照，检查五个 CUDA 扩展及轨迹末端误差。原任务入口依赖尚未下载的资源，无法直接满足独立安装验收。
- 输入输出：渲染脚本必须显式执行并传入 `--output-dir`，输出两张 320x240 PNG 和 `result.json`；CUDA 脚本显式传入 `--output`，输入固定 seed 42 的小型点云及 Franka 合成目标，输出版本、数值误差、轨迹形状和显存统计 JSON。不读取策略模型或训练数据；没有 resize/crop/pad、模型归一化或策略动作输出。
- 关闭后的行为：未执行检查脚本、未使用约束文件或未激活环境时，不增加模型依赖加载、GPU 运算或评估步骤；训练/推理数据流、默认模型和动作配置均不变。
- 验证方式：NVIDIA Vulkan 四卡枚举、`pip check`、PyTorch3D CUDA KNN 对照、CuRobo 五个扩展与实际 mesh-world 轨迹规划、SAPIEN RT/OIDN 动态渲染、Conda 激活及路径恢复均通过；源代码与官方固定 commit 的 Git blob 摘要逐项一致。具体命令和数值见安装实验文档。
- 已知限制：本次环境安装未执行策略 rollout；后续的任务配置与资源恢复已完成，见下节；checkpoint 归一化统计及正式评估仍需另行核对。用户态 NVIDIA 图形库绑定宿主驱动 535.104.05，宿主驱动升级后需要同步更新。

## RoboTwin 官方配置与资源恢复

- 日期：2026-09-07。
- 修改目的：按用户要求补齐当前库的任务、相机、机器人映射配置及物体、机器人、背景纹理资源。
- 涉及文件：`evaluation/RoboTwin/task_config/` 七个官方文件、`script/check_assets.py::check_files/check_render`；完整来源、命令、版本及结果见 `docs/experiments/robotwin_assets_20260907/experiment.md` 和同目录 JSON。大型资源及临时下载文件被现有忽略规则排除，不放入 Git。
- 实现来源：配置直接恢复自 [RoboTwin stable_2.0 的 task_config](https://github.com/RoboTwin-Platform/RoboTwin/tree/13c3c47ff4312dd62484bcd51be034af55c062d1/task_config)，不改变内容；资源来自 [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/785feb15aa4a4f532395ad2b1d2be5f28cb561ad)。直接复用本仓库 `assets/_download.py` 和 `script/update_embodiment_config_path.py::main`，网络传输适配记录在实验文档。
- 自定义检查：`check_files` 用 YAML/XML 解析器检查引用、机器人网格、生成的规划器配置及纹理编号；`check_render` 复用 `envs/adjust_bottle.py::adjust_bottle.setup_demo` 和 `envs/_base_task.py::get_obs/close_env`，初始化真实任务验证资源和三路相机。原入口需要启动完整评估，无法直接满足仅安装验收且不加载策略的需求。
- 配置开关：只在手动执行检查脚本时检查文件；`--render` 默认关闭，关闭时不导入 SAPIEN/Torch/CuRobo 或占用 GPU。`--task-config` 默认 `demo_clean`，`--seed` 默认 0，只作用于诊断。官方配置由既有评估 `task_config` 字段选择，没有新增生产默认开关。
- 输入输出：输入为当前 RoboTwin 配置和资源；`--output` 必须显式提供，输出检查 JSON，渲染启用时另输出原始三路 320x240 RGB PNG。诊断固定任务 `adjust_bottle`，禁用数据和视频保存；不调用策略或专家 rollout，没有 resize/crop/pad/归一化及动作 chunk 改动。
- 兼容性：默认评估脚本、模型、动作执行和训练数据流不变；恢复缺失的官方依赖文件。机器人路径通过原模板替换机制适配当前绝对目录。
- 验证方式：七个配置的固定 Git blob 摘要、全部下载 LFS SHA256 与 ZIP 完整性均通过；20,841 个解压文件存在且大小匹配，五种机器人、六个规划器配置、10,000 seen/1,000 unseen 纹理引用有效。seed 0 的 Clean/Randomized 真实任务初始化、原 CuRobo 规划器预热、三路 RGB 与 14 维状态检查均通过，并查看了六张截图；未执行策略 rollout。精确命令和结果见本次实验说明。
- 已知限制：资源可用性检查不等于策略成功率；正式评估的 checkpoint、归一化统计及模型服务仍需单独核对。

## ZR-0 LeRobot environment import repair

- Date: 2026-09-07. Purpose: restore the model server's existing
  `lerobot.common` imports in the ZR-0 Conda environment.
- Source: direct reuse of this repository's already installed editable
  `lerobot/` distribution (0.1.0), including
  `lerobot/lerobot/common/datasets/lerobot_dataset.py::LeRobotDatasetMetadata`.
  No package source, model, dataset, dependency version or training code changes.
- Diagnosis: user-site LeRobot 0.4.4 took precedence over the correct editable
  installation. The environment-local installation itself was intact.
- Configuration: persist `PYTHONNOUSERSITE=1` using `conda env config vars set`
  only for `/opt/data/private/lq/miniconda3/envs/ZR-0`. Activating this environment
  disables user-site package loading and resolves LeRobot to the current repo.
  This changes the environment's import policy, not model feature defaults.
- Compatibility and disabling: installed packages, checkpoints and user-site
  files are preserved. Deactivation restores user-site loading in base.
  Explicitly unset the Conda variable to restore the previous import policy;
  the original shadowing issue can then recur. Direct Python invocations without
  activation need `PYTHONNOUSERSITE=1` or `-s`.
- Inputs/outputs: existing local LIBERO demo metadata is used only for CPU
  dependency diagnostics, through `resolve_dataset_spec` and
  `prepare_action_expert_inputs_cpu`. No image preprocessing, action settings,
  normalization statistics or production data flow is changed.
- Validation and exact commands: see
  `docs/experiments/zr0_lerobot_environment_20260907/experiment.md`. Checks cover
  import origins, environment activation/deactivation, pip requirements,
  server/policy imports, the existing metadata resolver and CPU input preparation.
- Limitations: no model weights loaded, CUDA context, training or policy rollout.
  RoboTwin normalization metadata and full evaluation remain separate work.

## Docs 与 Evaluation 目录忽略规则

- 日期：2026-09-07。
- 修改目的：按用户要求忽略仓库根目录下 `docs/` 和 `evaluation/` 中尚未被 Git 跟踪的内容，同时保留既有文件及其跟踪状态。
- 涉及文件：`.gitignore`、`reference.md`。
- 配置开关：不适用；根目录忽略规则 `/docs/` 和 `/evaluation/` 始终生效。
- 默认状态：两个目录中的未跟踪文件默认被 Git 忽略。
- 实现来源：Git 标准忽略规则，无外部实现。
- 具体设计：使用仓库根目录锚定的 `/docs/` 和 `/evaluation/` 目录规则；不执行索引取消跟踪操作。
- 输入与输出：两个目录下未来新增且未显式强制加入的文件不会出现在普通 `git status` 或 `git add` 结果中；磁盘内容不变。
- 与原有流程的关系：不改变训练、评估、模型、数据处理逻辑或既有 Git 跟踪状态，只改变未跟踪文件的默认可见性。
- 关闭功能后的行为：删除对应忽略规则后，未跟踪文件会重新出现在 Git 状态中。
- 验证方式：`git check-ignore` 命中两条根目录规则，暂存区不存在因本次操作产生的目录删除，并核对磁盘文件数量不变。
- 已知限制：`.gitignore` 不影响已经跟踪或已经暂存的文件；这些文件后续发生修改时仍会被 Git 报告。

## RoboTwin evaluation metadata provenance audit

- Date: 2026-09-07. Purpose: verify the supplied local `robotwin_unified`
  metadata before using its normalization with the official ZR-0 checkpoint.
- Source: read-only inspection of the existing v2 metadata loader,
  `utils/dataset_spec.py`, `utils/normalization.py`, and the local v3 metadata;
  official release sources and identities are recorded in
  `docs/experiments/robotwin_metadata_20260907/experiment.md`.
- Results: 27,500 episodes, 6,075,103 frames, three 480x640 RGB cameras and
  14-dimensional state/action validated. Episode ranges, task references and
  all eight data-file row counts are consistent. Source statistics contain
  min/max/mean/std/count but no q01/q99 required by the quantile normalizer.
- Official-source result: the checkpoint's matching metadata was not found in
  the checked official GitHub tree/history or ModelScope release files.
  Local recalculation cannot establish official-checkpoint equivalence.
- Subsequent decision: the user explicitly accepted local recomputation for
  diagnostic evaluation. The implementation below supersedes the original
  blocked preparation status; official-checkpoint equivalence remains unverified.
- Validation: structured JSON/Parquet reads, full metadata index/range checks,
  source hashes and official file-list queries; no GPU, model load or rollout.

## RoboTwin local diagnostic metadata export

- Date: 2026-09-07. Purpose: make the supplied v3 dataset's metadata usable by
  the existing v2 inference reader, after explicit acceptance of local statistics.
- Files: `scripts/prepare_robotwin_eval_metadata.py`,
  `tests/test_robotwin_eval_metadata.py`, `dataset2feature.yaml`,
  `demo_data/README.md`, `demo_data/robotwin2.0-aloha-agilex/`, and the preceding
  audit's experiment document and validation results.
- Source: directly reuse
  `lerobot/lerobot/common/datasets/utils.py::get_stats`; do not change the
  calculation or normalization functions. The exporter adapts input reading
  to selected Parquet columns and a separate metadata-only destination because
  `calculate_global_stats` would overwrite the original dataset's statistics
  and does not export the v3 episode/task tables to the v2 metadata interface.
- Structure/data flow: validate original episode/task tables; read all frames
  once into float32 state/action arrays; check dimensions, finite values and
  frame/episode indices; compute exact full-array q01/q99 and the other original
  statistics; compare source file hashes before/after; publish the four v2
  metadata files and provenance atomically from a temporary directory.
  No video decoding, trajectory conversion, dataset symlinks or source writes.
- Inputs/outputs: source is the local `robotwin_unified` v3 directory. Output
  is `demo_data/robotwin2.0-aloha-agilex`, with all 27,500 episode records,
  23,559 instruction IDs, 6,075,103 contributing frames, original three camera
  names and 14 motor dimensions. `metadata_only` and
  `local_recomputed_checkpoint_unverified` markers preserve the export's purpose.
  Source/output SHA256 values, helper identity and NumPy version are recorded.
- Switches: the export script requires explicit source/output and
  `--allow-unverified-stats` (default false); it refuses existing outputs or
  outputs within the original dataset. The server uses the new metadata only
  when `--dataset_entry demo_data.robotwin2.0-aloha-agilex` is selected.
  The registry retains `use_quantile: true`; all other entries and defaults
  remain unchanged. There is no new production model or loss feature.
- Compatibility: reuse `LeRobotDatasetMetadata`, `resolve_dataset_spec`,
  `prepare_action_expert_inputs_cpu`, `min_max_norm` and `denormalize_actions`.
  Source camera size, actual model resize, action horizon/execution, padding and
  normalization formulas are unchanged; only the explicitly selected entry's
  metadata/statistics source is supplied. This is not a v2 training dataset.
- Validation: seven focused tests cover uneven-file full-frame quantiles,
  existing-reader loading, source preservation, output protection, explicit
  choice, shape/NaN/index/count rejection. Full-data export succeeded and
  checked all 12 original metadata/Parquet file hashes before and after.
  Production metadata/input/denormalization checks are recorded in the experiment.
- Limitations: statistics are not author-provided and are not proven equivalent
  to the checkpoint's training statistics. No model weights, GPU context,
  training or policy rollout are started by this task.

## RoboTwin legacy checkpoint launch preparation

- Date: 2026-09-07. Purpose: explicitly select the existing legacy compatibility
  path for the downloaded official checkpoint and the accepted local metadata.
- Files: `scripts/run_robotwin_legacy_server.sh`,
  `tests/test_robotwin_legacy_server.py`, `tests/test_server_action_checkpoint.py`,
  `tests/test_policy_manifest.py`, and
  `docs/experiments/robotwin_legacy_checkpoint_20260907/{experiment.md,compatibility.json}`.
- Source: directly reuse `server.py::deploy`,
  `utils/cli_options.py::parse_server_options`, `ZR0Policy`, and
  `utils/dataset_manifest.py::validate_policy_dataset_manifest`. The new shell
  launcher only assembles the existing server command, sets package isolation
  and the repository working directory, and prints or executes the command.
  Existing launchers target training and do not provide this evaluation command.
- Switches/defaults: default mode `print` does not import the model. Explicit
  `serve` requires the caller's `CUDA_VISIBLE_DEVICES` and executes the original
  server with `--allow_legacy_checkpoint_without_manifest`. Optional environment
  overrides select Python/checkpoint/port; defaults are the existing ZR-0 Conda
  Python, downloaded RoboTwin checkpoint and port 8022. The shared parser and
  policy retain their strict false default. There is no training feature.
- Inputs/outputs: the fixed dataset entry is the local diagnostic RoboTwin
  metadata. The command fixes existing settings: direct action, window 1,
  five denoising steps and padding 64. The checkpoint retains horizon 16;
  Difference Query stays disabled through existing legacy resolution, with
  no new attention backend selection. Print mode emits the shell command;
  serve mode starts the original WebSocket server and preserves its behavior.
- Compatibility: the missing-manifest option does not bypass an existing
  manifest's integrity or semantic checks. The separate observation-contract
  override is not enabled. No checkpoint files or training manifests are
  fabricated, and no production Python/data/image/action logic is changed.
  The launcher explicitly reports local statistics' unverified equivalence.
- Verification: 20 focused tests, Bash syntax and command parsing passed.
  Actual metadata plus checkpoint configuration passes the explicit legacy
  path and fails the strict path. Existing header/shape validators confirm
  149 Action Expert tensors, two VLM shards/625 indexed tensors, and disabled
  Difference Query. Checkpoint file identities remain unchanged.
- Limitations: only checkpoint/config/manifest structure is checked; GPU weight
  loading, numeric forward execution, WebSocket integration and rollouts have
  not run. Exact commands, stats SHA256 and results are in the experiment doc.

## RoboTwin complete 50 x 2 x 20 evaluation preparation

- Date: 2026-09-07. Purpose: prepare all 50 tasks in both Clean and Random
  settings, with 20 policy rollouts per task/setting (2,000 total), as requested.
- Files: `configs/robotwin_eval_50x2x20.json`,
  `scripts/prepare_robotwin_eval_suite.py`, `tests/test_robotwin_eval_suite.py`,
  `evaluation/RoboTwin/policy/ZR0/deploy_policy.yml`, and
  `docs/experiments/robotwin_eval_50x2x20_20260907/experiment.md`.
- Source: direct reuse of the original
  `evaluation/RoboTwin/script/eval_policy_client.py::main/eval_policy`, the
  ZR0 deployment template and `task_config/_eval_step_limit.yml`. No client,
  simulator, policy, image or model code is changed. The original client accepts
  one YAML per invocation and has no matrix scheduler; a small custom generator
  supplies those YAMLs and a sequential command script using that same entry.
- Structure/data flow: `build_plan` validates task/profile/count/horizon settings
  and expands their Cartesian product; `prepare_suite` writes 100 generated
  YAMLs, source hashes and Git provenance in `plan.json`, the requested config,
  an experiment document and `run_all.sh` under an ignored output directory.
  `render_runner` routes configs to the original client with per-group logs,
  fail-fast execution and runtime status in the experiment document.
- Switches/defaults: suite preparation requires an explicit script invocation
  and output directory; it never launches evaluation. Running the generated
  shell script is a separate explicit action requiring the RoboTwin Conda
  environment and GPU selection. No automatic parallelism or resume is added.
  Without invocation, existing single-task execution remains available.
- Requested behavior changes: the single-task template now uses `n_episodes=20`
  instead of 100 and a descriptive result label. The suite overrides template
  task/scene/count from its central config and step limits from the official
  50-task table (400 to 1700), instead of forcing every task to 400 steps.
  A global step-limit override is rejected. Scene selection remains solely
  `task_config`; `ckpt_setting` still does not load a model or select a scene.
- Compatibility: keep seed 0, unseen instructions, expert solvability checking,
  execution horizon 16, original videos, camera profiles and result locations.
  Twenty means accepted policy trials, not twenty policy successes; expert
  attempts are additional. The original strict model-manifest defaults remain.
- Validation: focused matrix/count/step-limit and runner-routing tests, generated
  YAML and Bash syntax checks; detailed results recorded in the experiment doc.
- Limitations: local statistics are not proven equivalent to official training
  statistics; this label is preserved in the suite and plan. GPU loading,
  WebSocket integration, throughput and success rates were untested at the
  preparation stage; subsequent launch validation is recorded below.

## RoboTwin 50 x 2 x 20 evaluation launch

- Date: 2026-09-07. Purpose: execute the prepared full matrix after explicit user
  authorization. Reuse `scripts/run_robotwin_legacy_server.sh` and generated
  `outputs/evaluations/robotwin_50x2x20_seed0/run_all.sh` without parameter changes.
- Source/design: existing server and client execution only; tmux supplies
  persistent process sessions and file redirection supplies runtime logs.
  GPU 0 hosts the model and one serial simulation worker. The batch starts
  after the existing `/healthz` endpoint confirms server readiness.
- Switches/compatibility: explicit launch only; default preparation remains
  inactive. No training, dataset, image, horizon, model or loss changes.
  Local normalization statistics retain the existing unverified provenance.
- Files: launch details and actual results in
  `docs/experiments/robotwin_eval_50x2x20_20260907/experiment.md` and the output
  directory's `experiment.md`; logs/videos are ignored runtime artifacts.
- Validation: idle GPUs, available port, complete 100-group/2,000-trial matrix
  and eight source hashes checked before launch. Runtime checks are recorded
  in the experiment document as they complete.

## RoboTwin missing render-check entry restoration

- Date: 2026-09-07. Purpose: resolve the missing `test_render` import encountered
  on the first authorized evaluation launch, before any rollout.
- Files: `evaluation/RoboTwin/script/test_render.py` and the existing
  `evaluation/RoboTwin/script/check_render.py` (now explicitly tracked), plus
  the evaluation experiment document. The client itself is unchanged.
- Source: direct reuse of this repository's `check_render.py::check_render`,
  created during the earlier simulator installation and aligned with
  `envs/_base_task.py::setup_scene`. The small compatibility entry restores
  the `Sapien_TEST` function imported by the existing evaluation/collection
  launchers; it does not duplicate rendering or task logic.
- Inputs/outputs: no arguments; run the existing headless render/physics check
  in a temporary directory and remove its diagnostic images afterward. Existing
  validation failures propagate and still stop startup.
- Switch/default: importing the entry is inactive; only explicit execution or
  the original launcher's existing call runs the check. No evaluation/collection
  switches, task seeds, rollouts, camera settings or model behavior change.
- Validation: direct GPU render check and subsequent actual client startup;
  outcomes recorded in the evaluation experiment document. The failed initial
  launch produced no trials and its logs are retained before retry.
- Runtime result: the render check passed; the complete batch restarted at
  22:42:22 +08:00. At 22:45:04, repeated actual model forwards, 16x14 action
  responses, updated robot observations, over 100 policy steps and video writes
  were verified in the first Clean task. The batch remains asynchronous; this
  startup verification is not a completed success-rate result.
- By 22:48:07 the first policy trial completed 400 steps with failure (0/1,
  seed 100001) and the second trial had started, verifying episode transition
  and result counting. Full task/suite scores are still pending.

## RoboTwin independent GPU evaluation workers

- Date: 2026-09-07. Purpose: use all four GPUs for the authorized 50x2x20 matrix.
- Files: `scripts/prepare_robotwin_eval_suite.py::assign_workers/prepare_suite`,
  `scripts/run_robotwin_eval_worker.sh`, `utils/obs_buffer.py::ObservationBuffer`,
  `evaluation/RoboTwin/script/check_render.py`, focused suite/worker tests and
  `docs/experiments/robotwin_eval_50x2x20_20260907/experiment.md`.
- Source: adapt the existing matrix generator and directly reuse its
  `render_runner`, the legacy server launcher, health endpoint and original
  evaluator. A small custom worker shell manages one server's lifetime and
  starts one existing client batch after readiness. Existing launchers had
  neither disjoint assignments nor per-worker model lifetime management.
- Inputs/outputs: `--gpus 0 1 2 3` partitions 100 task/setting groups round-robin
  into 25 per GPU. Ports increment from 8022; generated configs and a complete
  plan connect each client to its assigned server. Each worker records logs,
  an exit code and experiment runtime information in its own directory.
- Switch/default: `--gpus` is absent by default, retaining serial behavior.
  `ZR0_OBSERVATION_DEBUG_ROOT` is set per worker to isolate visualization JPEGs;
  unset keeps the existing `temp` path. Observation values, tensors and model
  inputs are unchanged. Renderer diagnostics only add device/PCI to output.
- Explicit behavior changes: parallel scheduling uses independent seed-42
  model RNG streams and appends `_p4` to result labels. Environment seed 0,
  task/profile/count/limits, model, stats, image processing and action horizons
  remain unchanged. Partial single-GPU trials are preserved but excluded from
  the new run; it restarts the full matrix, with no episode resume assumption.
- Verification: partition uniqueness/coverage, matching ports, unchanged client
  semantics, serial fallback, worker startup/cleanup/failure routing, isolated
  visualization values, Bash syntax, plus actual four-GPU runtime checks.
- Limitations: no automatic retries/resume or guaranteed 4x wall-clock speedup.
  Other workers continue after one worker fails. Statistics equivalence remains
  unverified; actual startup/progress is recorded in the experiment document.
