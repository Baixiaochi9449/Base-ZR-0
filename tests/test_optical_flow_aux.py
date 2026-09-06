import json
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
from accelerate import Accelerator

from model.optical_flow_aux_head import build_optical_flow_head
from model.reasoning_vla_model import ZR0Model
from utils.cli_options import parse_train_options
from utils.optical_flow_config import OpticalFlowConfig, resolve_stage
from utils.optical_flow_loss import optical_flow_loss, prepare_flow_targets
from utils.optical_flow_reader import OpticalFlowReader
from utils.optical_flow_checkpoint import save_flow_artifacts, resolve_flow_checkpoint, load_flow_weights
from utils.load_training_dataset import custom_collate_fn
from utils.training_tokenization import DatasetIntegrityError
from train_vla import run_optimizer_step_window, trainable_parameter_owner
from test_difference_query_tiny_vla import TinyVlaDifferenceQueryTest as _TinyVla


def config(**kwargs):
    return replace(OpticalFlowConfig(optical_flow_aux_type="dense_regression_v1", num_flow_queries=2,
                   optical_flow_loss_weight=1.0, flow_head_hidden_dim=16, flow_head_num_layers=1), **kwargs)


def batch(available=(True, False)):
    size = len(available)
    return {"input_ids": torch.ones(size, 3, dtype=torch.long),
            "flow_supervision_available": torch.tensor(available),
            "flow_actual_delta_frames": torch.full((size,), 10), "flow_label_source": torch.ones(size, dtype=torch.long),
            "flow_target": {i: torch.full((2, 224, 224), .02) for i, yes in enumerate(available) if yes},
            "flow_valid_mask": {i: torch.ones(1, 224, 224, dtype=torch.bool) for i, yes in enumerate(available) if yes}}


def test_cli_explicit_loss_and_stages():
    assert parse_train_options([]).loss_type == "vlm_and_action"
    assert not parse_train_options([]).loss_type_explicit
    assert parse_train_options(["--training_stage", "stage1_ar", "--tune_vlm"]).loss_type == "vlm"
    with pytest.raises(SystemExit):
        parse_train_options(["--training_stage", "stage1_ar", "--tune_vlm", "--loss_type=vlm_and_action"])
    with pytest.raises(SystemExit):
        parse_train_options(["--training_stage", "stage2_aux"])
    options = parse_train_options(["--training_stage", "stage2_aux", "--optical_flow_aux_type", "dense_regression_v1",
                                  "--optical_flow_loss_weight", "1", "--num_flow_queries", "2", "--optical_flow_data_root", "/unused"])
    assert (options.loss_type, options.vlm_loss_weight, options.action_expert_loss_weight) == ("aux", 0, 0)
    with pytest.raises(ValueError):
        resolve_stage("stage1_ar", flow=config())
    with pytest.raises(NotImplementedError):
        resolve_stage("stage3_joint", flow=config(), slot_aux_type="fake")
    assert resolve_stage("stage3_joint", flow=OpticalFlowConfig()) == "vlm_and_action"


def test_head_isolation_parameters_and_rng():
    torch.manual_seed(3)
    state = torch.get_rng_state().clone()
    head = build_optical_flow_head(32, config())
    assert torch.equal(state, torch.get_rng_state())
    other = build_optical_flow_head(32, config(num_flow_queries=4))
    assert head.parameter_counts() == other.parameter_counts()
    queries = torch.randn(2, 4, 32, requires_grad=True)
    original = head(queries)
    changed = queries.detach().clone()
    changed[:, :2] += 100
    torch.testing.assert_close(original, head(changed), rtol=0, atol=0)
    loss = optical_flow_loss(original, batch(), config())["optical_flow_loss"]
    loss.backward()
    assert torch.count_nonzero(queries.grad[:, :2]) == 0
    assert queries.grad[0, -2:].abs().sum() > 0
    assert queries.grad[1].abs().sum() == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert head.output.weight.abs().sum() > 0


def test_mask_downsample_and_missing_labels():
    data = batch()
    indices, target, mask = prepare_flow_targets(data, config(), torch.device("cpu"))
    assert indices.tolist() == [0]
    torch.testing.assert_close(target, torch.full_like(target, .02))
    prediction = torch.zeros(2, 2, 56, 56, requires_grad=True)
    outputs = optical_flow_loss(prediction, data, config())
    assert outputs["optical_flow_loss_count"] == 1
    outputs["optical_flow_loss"].backward()
    assert prediction.grad[1].abs().sum() == 0
    data["flow_valid_mask"][0].zero_()
    assert len(prepare_flow_targets(data, config(), torch.device("cpu"))[0]) == 0
    empty = optical_flow_loss(prediction, data, config())
    assert empty["optical_flow_loss"].requires_grad
    empty["optical_flow_loss"].backward()


def fixture_manifest(root, frames=(3, 13), episode=0):
    path = root / f"{episode}.h5"
    n = len(frames)
    with h5py.File(path, "w") as f:
        f.attrs.update(camera_key="observation.images.image", merged_episode_index=episode, source_episode_index=episode,
                       nominal_delta_frames=10, flow_units="normalized_source_image_extent", flow_direction="forward_only",
                       tail_policy="clamp", fps=10., validity_semantics="finite_and_forward_destination_in_bounds_not_occlusion")
        f["frame_index"] = np.array(frames, dtype=np.int64)
        f["target_frame_index"] = np.array([frames[-1]] * n, dtype=np.int64)
        f["actual_delta_frames"] = np.array([frames[-1] - x for x in frames], dtype=np.int16)
        f["label_source"] = np.array([1] * (n - 1) + [2], dtype=np.uint8)
        f["flow"] = np.full((n, 2, 224, 224), .02, dtype=np.float16)
        f["valid_mask"] = np.ones((n, 1, 224, 224), dtype=np.uint8)
    return {"merged_episode_index": episode, "source_episode_index": episode,
            "camera_key": "observation.images.image", "frame_count": n, "hdf5_path": path.name}


def test_reader_mapping_lru_pickle_and_corruption(tmp_path):
    entries = [fixture_manifest(tmp_path, episode=i) for i in range(2)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(e) for e in entries))
    reader = OpticalFlowReader(tmp_path, manifest, max_handles=1)
    assert reader.read(0, 3)["flow_supervision_available"]
    assert reader.read(0, 0)["flow_target"] is None
    assert reader.read(0, 13)["flow_target"] is None
    first = next(iter(reader._handles.values()))
    reader.read(1, 3)
    assert not first.id.valid
    restored = pickle.loads(pickle.dumps(reader))
    assert not restored._handles
    assert restored.read(0, 3)["flow_supervision_available"]
    restored.close()
    reader.close()
    with h5py.File(tmp_path / "0.h5", "a") as f:
        del f["valid_mask"]
    with pytest.raises(DatasetIntegrityError):
        reader.read(0, 3)
    with pytest.raises(DatasetIntegrityError):
        OpticalFlowReader(tmp_path, manifest)
    manifest.write_text(json.dumps(entries[1]) + "\n" + json.dumps(entries[1]))
    with pytest.raises(DatasetIntegrityError, match="duplicate"):
        OpticalFlowReader(tmp_path, manifest)


def test_collator_does_not_fabricate_flow():
    first = {"input_ids": torch.tensor([1, 2]), **{
        k: v[0] for k, v in batch((True,)).items() if k != "input_ids"}}
    second = {"input_ids": torch.tensor([1, 2])}
    result = custom_collate_fn([first, second])
    assert result["flow_supervision_available"].tolist() == [True, False]
    assert set(result["flow_target"]) == {0}


def test_empty_window_skips_adamw_scheduler_and_forward():
    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with patch.object(model, "forward", side_effect=AssertionError("empty forward")):
        outputs = run_optimizer_step_window(model=model, batches=[batch((False, False))], accelerator=Accelerator(cpu=True),
            optimizer=optimizer, lr_scheduler=scheduler, training_progress=0, loss_type="aux",
            vlm_loss_weight=0, action_expert_loss_weight=0, next_global_step=1, optical_flow_config=config())
    assert outputs["optimizer_update_skipped"] == 1
    assert scheduler.last_epoch == 0 and not optimizer.state
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_flow_checkpoint_hash_conflict_and_roundtrip(tmp_path):
    head = build_optical_flow_head(32, config())
    model = SimpleNamespace(optical_flow_config=config(), optical_flow_aux=head, training_stage="stage2_aux", stage_description="OF-only")
    save_flow_artifacts(model, tmp_path)
    from model.difference_query import save_difference_query_artifacts
    save_difference_query_artifacts(tmp_path, enabled=True, hidden_size=32,
        difference_query=torch.randn(4, 32))
    saved, payload = resolve_flow_checkpoint(tmp_path, stage="stage3_joint")
    copy = build_optical_flow_head(32, saved)
    load_flow_weights(copy, tmp_path, payload)
    query = torch.randn(1, 4, 32)
    torch.testing.assert_close(head(query), copy(query), rtol=0, atol=0)
    with pytest.raises(ValueError, match="conflicts"):
        resolve_flow_checkpoint(tmp_path, config(num_flow_queries=3))
    with pytest.raises(ValueError, match="same training_stage"):
        resolve_flow_checkpoint(tmp_path, stage="stage3_joint", resume=True)
    (tmp_path / "optical_flow_aux.safetensors").unlink()
    with pytest.raises(ValueError, match="without OF weights"):
        resolve_flow_checkpoint(tmp_path)


def tiny_stage_model(stage):
    from types import MethodType
    from transformers.feature_extraction_utils import BatchFeature
    model = ZR0Model.__new__(ZR0Model)
    torch.nn.Module.__init__(model)
    model.backbone = _TinyVla.make_backbone()
    def prepare(_self, inputs):
        return BatchFeature({k: inputs[k] for k in ("input_ids", "attention_mask", "labels") if k in inputs})
    model.backbone.prepare_inputs = MethodType(prepare, model.backbone)
    model.training_stage = stage
    model.optical_flow_config = OpticalFlowConfig() if stage == "stage1_ar" else config()
    model.loss_type = resolve_stage(stage, flow=model.optical_flow_config)
    model.action_expert = _TinyVla.make_action_head() if stage == "stage3_joint" else None
    model.action_expert_config = _TinyVla.make_action_head().config
    model.optical_flow_aux = build_optical_flow_head(32, model.optical_flow_config)
    model.use_difference_query, model.num_difference_queries = True, 4
    model.detach_vlm_outputs_for_action_expert = False
    return model


def tiny_batch():
    return {**batch((True,)), "input_ids": torch.tensor([[5, 6, 7, 8, 127]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
            "labels": torch.tensor([[-100, -100, 7, 8, -100]]),
            "observation.state": torch.randn(1, 1, 4), "state_mask": torch.ones(1, 1, 4, dtype=torch.bool),
            "action": torch.randn(1, 3, 4), "action_mask": torch.ones(1, 3, 4, dtype=torch.bool)}


@pytest.mark.parametrize("stage,expected", [("stage1_ar", {"ar_loss"}), ("stage2_aux", {"optical_flow_loss"}),
                                            ("stage3_joint", {"ar_loss", "optical_flow_loss", "flow_matching_loss"})])
def test_three_stage_forward_and_ownership(stage, expected):
    model = tiny_stage_model(stage)
    data = tiny_batch()
    outputs = model(data, 0.)
    assert {key for key in ("ar_loss", "optical_flow_loss", "flow_matching_loss") if key in outputs} == expected
    assert outputs["slot_loss_computed"] is False and "slot_loss" not in outputs
    torch.testing.assert_close(outputs["loss"], sum(outputs[key] for key in expected))
    outputs["loss"].backward()
    owners = {trainable_parameter_owner(name) for name, p in model.named_parameters() if p.requires_grad}
    assert ("action_expert" in owners) == (stage == "stage3_joint")
    assert ("optical_flow_aux" in owners) == (stage != "stage1_ar")
    assert model.backbone.difference_query.weight.grad.abs().sum() > 0


def test_cqtp_flow_invariance_and_action_reads_all_queries():
    model = tiny_stage_model("stage3_joint").eval()
    data = tiny_batch()
    def prediction(inputs):
        return model.optical_flow_aux(model.backbone(model.backbone.prepare_inputs(inputs))["backbone_embeddings"])
    first = prediction(data)
    changed = dict(data, input_ids=data["input_ids"].clone(), labels=data["labels"].clone())
    changed["input_ids"][0, 2:4] = torch.tensor([9, 10])
    changed["labels"][0, 2:4] = torch.tensor([9, 10])
    torch.testing.assert_close(first, prediction(changed), rtol=0, atol=0)
    changed["input_ids"][0, 0] = 11
    assert not torch.equal(first, prediction(changed))
    with patch.object(model.action_expert, "forward", wraps=model.action_expert.forward) as action:
        outputs = model(data, 0.)
        assert action.call_args.args[0]["backbone_embeddings"].shape[1] == 4
    modified = dict(data, flow_target={0: data["flow_target"][0] * 2})
    torch.manual_seed(37)
    a = model(data, 0.)
    torch.manual_seed(37)
    b = model(modified, 0.)
    torch.testing.assert_close(a["ar_loss"], b["ar_loss"], rtol=0, atol=0)
    torch.testing.assert_close(a["flow_matching_loss"], b["flow_matching_loss"], rtol=0, atol=0)
    assert a["optical_flow_loss"] != b["optical_flow_loss"]


def test_stage_transition_ignores_stale_expert_and_preserves_queries(tmp_path):
    from test_query_ar_joint_checkpoint import _TinyBackbone, _TinyActionExpert, QueryArWarmStartCheckpointTest
    from safetensors.torch import save_file
    from utils.optical_flow_checkpoint import module_checksum
    with patch("model.reasoning_vla_model.QwenVLBackbone", _TinyBackbone), patch("model.reasoning_vla_model.FlowmatchingActionHead", _TinyActionExpert), patch("model.flow_matching_action_head.FlowmatchingActionHead", _TinyActionExpert):
        kwargs = dict(action_expert_name_or_path=None, action_expert_config=QueryArWarmStartCheckpointTest.action_config(), tune_vlm=True)
        stage1 = ZR0Model(str(tmp_path / "base"), **kwargs, training_stage="stage1_ar", use_difference_query=True, num_difference_queries=4)
        ar = tmp_path / "ar"
        stage1.save_pretrained(ar)
        # Deliberately stale valid Expert weights must not enter either new stage.
        stale = _TinyActionExpert(None, True)
        with torch.no_grad():
            for parameter in stale.parameters():
                parameter.fill_(99)
        save_file(stale.state_dict(), str(ar / "action_expert.safetensors"))
        stage2 = ZR0Model(str(ar), **kwargs, training_stage="stage2_aux", optical_flow_config=config(), init_from_checkpoint=str(ar))
        assert stage2.action_expert is None
        assert module_checksum(stage1.backbone) == module_checksum(stage2.backbone)
        assert stage2.aux_initialization["before"] == stage2.aux_initialization["after"]
        aux = tmp_path / "aux"
        stage2.save_pretrained(aux)
        save_file(stale.state_dict(), str(aux / "action_expert.safetensors"))
        torch.manual_seed(2)
        stage3 = ZR0Model(str(aux), **kwargs, training_stage="stage3_joint", init_from_checkpoint=str(aux))
        torch.manual_seed(98)
        again = ZR0Model(str(aux), **kwargs, training_stage="stage3_joint", init_from_checkpoint=str(aux))
        assert module_checksum(stage2.backbone) == module_checksum(stage3.backbone)
        assert module_checksum(stage2.optical_flow_aux) == module_checksum(stage3.optical_flow_aux)
        assert module_checksum(stage3.action_expert) == module_checksum(again.action_expert)
        assert module_checksum(stale) != module_checksum(stage3.action_expert)
        joint = tmp_path / "joint"
        stage3.save_pretrained(joint)
        restored = ZR0Model.from_pretrained(joint, tune_vlm=True, tune_action_expert=True)
        assert module_checksum(restored) == module_checksum(stage3)
        aux_restored = ZR0Model.from_pretrained(aux, tune_vlm=True, tune_action_expert=False)
        assert module_checksum(aux_restored) == module_checksum(stage2)
        inference = ZR0Model.from_pretrained(joint, for_action_inference=True)
        assert inference.optical_flow_aux is None
        assert module_checksum(inference.action_expert) == module_checksum(stage3.action_expert)


class ScalarFlowModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(.1))
        self.training_stage = "stage2_aux"
        self.optical_flow_config = config()

    def forward(self, data, *_args, **_kwargs):
        from utils.optical_flow_config import stage_loss_metadata
        prediction = self.value.expand(data["input_ids"].shape[0], 2, 56, 56)
        result = optical_flow_loss(prediction, data, config())
        return {**result, **stage_loss_metadata("stage2_aux", flow=result["optical_flow_loss_count"] > 0)}


def distributed_flow_worker(rank, rendezvous, output):
    import torch.distributed as dist
    from accelerate.utils import DistributedType
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    class CPUAccelerator:
        device = torch.device("cpu")
        is_main_process = rank == 0
        num_processes = 2
        gradient_accumulation_steps = 2
        distributed_type = DistributedType.MULTI_CPU
        def reduce(self, value, reduction):
            assert value.device == self.device
            value = value.clone()
            dist.all_reduce(value)
            return value / 2 if reduction == "mean" else value
        def gather(self, value):
            assert value.device == self.device
            gathered = [torch.zeros_like(value) for _ in range(2)]
            dist.all_gather(gathered, value)
            return torch.cat(gathered)
        def no_sync(self, model):
            return model.no_sync()
        def unwrap_model(self, model):
            return model.module
        def backward(self, value):
            (value / self.gradient_accumulation_steps).backward()
    accelerator = CPUAccelerator()
    model = torch.nn.parallel.DistributedDataParallel(ScalarFlowModel())
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    data = [batch((rank == 0, False)), batch((rank == 0, rank == 0))]
    metrics = run_optimizer_step_window(model=model, batches=data, accelerator=accelerator,
        optimizer=optimizer, lr_scheduler=scheduler, training_progress=0, loss_type="aux", vlm_loss_weight=0,
        action_expert_loss_weight=0, next_global_step=1, optical_flow_config=config())
    assert metrics["flow_eligible_samples"] == 3
    assert metrics["flow_coverage"] == 3 / 8
    assert model.module.stage_training_state["flow_loss_computed"] is True
    updated = model.module.value.detach().clone()
    skipped = run_optimizer_step_window(model=model, batches=[batch((False,)), batch((False,))], accelerator=accelerator,
        optimizer=optimizer, lr_scheduler=scheduler, training_progress=0, loss_type="aux", vlm_loss_weight=0,
        action_expert_loss_weight=0, next_global_step=2, optical_flow_config=config())
    assert skipped["optimizer_update_skipped"] == 1 and scheduler.last_epoch == 1
    assert torch.equal(updated, model.module.value)
    assert model.module.stage_training_state["flow_loss_computed"] is True
    assert model.module.stage_training_state["slot_loss_computed"] is False
    from utils.wandb_training_logger import WandbTrainingLogger
    logs = []
    remote = SimpleNamespace(log=lambda payload, step: logs.append(payload))
    with patch.dict("sys.modules", {"wandb": SimpleNamespace(init=lambda **_: remote)}):
        logger = WandbTrainingLogger(accelerator, project="test", run_name="test", run_id="test",
                                     resume="never", log_dir="/tmp")
    for record in (metrics, skipped):
        logger.log(step=1, mean_metrics=record, scalar_metrics={})
    if rank == 0:
        assert logs[0]["flow_loss_computed"] is True
        assert logs[1]["flow_loss_computed"] is False
        assert all(record["slot_loss_computed"] is False for record in logs)
    if rank == 0:
        torch.save(updated, output)
    dist.destroy_process_group()


def test_distributed_empty_rank_and_global_normalization(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(distributed_flow_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path / "result.pt")), nprocs=2, join=True)
    model = ScalarFlowModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    result = model(batch((True, True, True)))
    result["optical_flow_loss"].backward()
    optimizer.step()
    torch.testing.assert_close(model.value, torch.load(tmp_path / "result.pt", weights_only=True), rtol=1e-6, atol=1e-7)


def test_rng_and_resume_cursor_survive_skipped_batches(tmp_path):
    import random
    from utils.training_checkpoint import capture_rng_state, restore_rng_state, restore_stage_runtime
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng_state(state)
    assert random.random() == expected[0] and np.random.rand() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    contract = {"seed": 42, "batch_size": 2, "gradient_accumulation_steps": 2, "dataloader_length": 20}
    torch.save({"rng": state, "cursor": {"epoch": 2, "batch_idx": 13}, "global_step": 3,
                "world_size": 1, "sampler_contract": contract, "skipped_flow_batches": 8, "scaler": None},
               tmp_path / "training_runtime_rank0.pt")
    model = SimpleNamespace(training_sampler_contract=contract)
    accelerator = SimpleNamespace(process_index=0, num_processes=1, scaler=None)
    assert restore_stage_runtime(tmp_path, model, accelerator, 3) == (2, 13)
    assert model.skipped_flow_batches == 8


def test_stage3_empty_flow_preserves_ar_and_fm_update():
    model = tiny_stage_model("stage3_joint")
    data = tiny_batch()
    data.update(flow_supervision_available=torch.tensor([False]), flow_target={}, flow_valid_mask={})
    before = model.backbone.difference_query.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    torch.manual_seed(91)
    expected_fm = model(data, 0.)["flow_matching_loss"].detach()
    torch.manual_seed(91)
    metrics = run_optimizer_step_window(model=model, batches=[data], accelerator=Accelerator(cpu=True),
        optimizer=optimizer, lr_scheduler=scheduler, training_progress=0, loss_type="vlm_and_action",
        vlm_loss_weight=1, action_expert_loss_weight=1, next_global_step=1, optical_flow_config=config())
    assert metrics["optimizer_update_skipped"] == 0 and metrics["flow_eligible_samples"] == 0
    assert not torch.equal(before, model.backbone.difference_query.weight)
    assert metrics["ar_loss"] > 0 and metrics["flow_matching_loss"] > 0
    torch.testing.assert_close(metrics["flow_matching_loss"].float(), expected_fm.float())


class WorkerFlowDataset(torch.utils.data.Dataset):
    def __init__(self, reader):
        self.reader = reader
    def __len__(self):
        return 4
    def __getitem__(self, index):
        import os
        result = self.reader.read(index % 2, 3)
        return os.getpid(), result["flow_target"].mean(), len(self.reader._handles)


def test_reader_independent_spawn_workers(tmp_path):
    import os
    entries = [fixture_manifest(tmp_path, episode=i) for i in range(2)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(entry) for entry in entries))
    reader = OpticalFlowReader(tmp_path, manifest, max_handles=1)
    reader.read(0, 3)
    loader = torch.utils.data.DataLoader(WorkerFlowDataset(reader), batch_size=1, num_workers=2, multiprocessing_context="spawn")
    rows = list(loader)
    assert len({int(row[0]) for row in rows}) == 2
    assert all(int(row[0]) != os.getpid() and int(row[2]) == 1 for row in rows)
    assert len(reader._handles) == 1
    reader.close()


def test_stage06_spec_quantiles_match_existing_normalizer(tmp_path):
    from utils.dataset_spec import resolve_dataset_spec, resolve_objective_requirements, denormalize_actions
    from utils.load_training_dataset import prepare_action_expert_inputs_cpu
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"fps": 10, "codebase_version": "v3.0", "features": {
        "observation.images.image": {"shape": [256, 256, 3]}, "observation.state": {"shape": [8]}, "action": {"shape": [7]}}}))
    (meta / "stats.json").write_text(json.dumps({key: {"q01": [-1.] * dim, "q99": [1.] * dim}
        for key, dim in (("observation.state", 8), ("action", 7))}))
    entry = {"dataset_path": str(tmp_path), "dataset_adapter": "stage06_libero_flow", "sample_ratio": 1.,
             "camera_keys": ["observation.images.image"]}
    manifest = tmp_path / "flow.jsonl"
    manifest.write_text(json.dumps({"camera_key": "observation.images.image"}) + "\n")
    entry["optical_flow_manifest"] = str(manifest)
    requirements = resolve_objective_requirements("vlm_and_action", adapter="stage06_libero_flow", target_text_field=None)
    spec = resolve_dataset_spec("fixture", entry, action_horizon=3, requirements=requirements)
    data = {"observation.state": torch.zeros(1, 8), "action": torch.full((3, 7), .25), "action_is_pad": torch.tensor([False, True, True])}
    inputs = prepare_action_expert_inputs_cpu(data, spec.normalization_stats, 64, True)
    assert inputs["action_mask"].sum() == 7
    assert inputs["observation.state"].shape == (1, 64)
    torch.testing.assert_close(denormalize_actions(inputs["action"], spec)[0], data["action"][0])


@pytest.mark.parametrize("stage", ["stage2_aux", "stage3_joint"])
def test_launch_template_explicit_objectives(stage):
    import os
    import shlex
    import subprocess
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, INIT_CHECKPOINT="/unused-source", OUTPUT_DIR="/unused-output", NUM_FLOW_QUERIES="8",
               OPTICAL_FLOW_LOSS_WEIGHT="1", WANDB_PROJECT="fixture", WANDB_RUN_NAME="fixture", ACCELERATE_CONFIG="unused.yaml",
               AR_DATASET_ENTRY="molmoact_tabletop_v3_stage05")
    command = shlex.split(subprocess.check_output(["bash", str(root / "scripts/run_optical_flow_stage.sh"), stage, "--print-command"], env=env, text=True))
    arguments = command[command.index("train_vla.py") + 1:]
    options = parse_train_options(arguments)
    assert options.training_stage == stage and options.init_from_checkpoint == "/unused-source"
    assert options.optical_flow_aux_type == "dense_regression_v1" and options.optical_flow_loss_weight == 1
    assert options.num_flow_queries == 8 and options.wandb_failure_policy == "required"
