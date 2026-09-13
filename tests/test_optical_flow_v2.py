import copy
import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import pytest

from model.optical_flow_aux_head import build_optical_flow_head
from scripts.build_flow_latent_cache import _sample_batch, build_cache
from scripts.calibrate_flow_color_scale import ReservoirQuantile, _parse_episode_splits, calibrate
from train_vla import advance_global_step, run_optimizer_step_window
from utils.optical_flow_config import OpticalFlowConfig
from utils.optical_flow_checkpoint import (
    checkpoint_stage_metadata, read_flow_artifacts, resolve_flow_checkpoint,
    save_flow_artifacts, validate_v2_runtime_payload,
)
from utils.optical_flow_loss import optical_flow_loss
from utils.optical_flow_reader import OpticalFlowReader
from utils.optical_flow_v2 import (
    CACHE_MANIFEST_NAME, WanFlowTargetBuilder, flow_to_fixed_rgb,
    latent_content_identity, select_v2_flow_indices,
)


REAL_WAN_CUDA_TEST_ENV = "ZR0_RUN_REAL_WAN_CUDA_TEST"


def _require_real_wan_cuda_test(*, env=None, gate=None, cuda_available=None):
    env = dict(os.environ if env is None else env)
    if env.get(REAL_WAN_CUDA_TEST_ENV) != "1":
        pytest.skip(f"set {REAL_WAN_CUDA_TEST_ENV}=1 to opt in to the real Wan CUDA test")
    visible = env.get("CUDA_VISIBLE_DEVICES")
    devices = [] if visible is None else [value.strip() for value in visible.split(",")]
    if len(devices) != 1 or not devices[0]:
        pytest.skip("real Wan CUDA test requires one explicitly selected CUDA_VISIBLE_DEVICES entry")
    if gate is None:
        from utils.gpu_resource_gate import wait_for_gpus
        gate = wait_for_gpus
    try:
        result = gate(env=env, expected_count=1)
    except Exception as error:
        pytest.skip(f"project GPU resource gate did not pass: {error}")
    cuda_available = torch.cuda.is_available if cuda_available is None else cuda_available
    if not cuda_available():
        pytest.skip("selected device passed the resource query but PyTorch CUDA is unavailable")
    return result


def v2_config(**kwargs):
    values = dict(optical_flow_aux_type="wan_vae_latent_v2", num_flow_queries=2,
                  optical_flow_loss_weight=1.0, flow_vae_model_path="/tmp/vae",
                  flow_color_scale=1.0, flow_delta_frames=20, flow_latent_shape=(4, 2, 2))
    values.update(kwargs)
    return OpticalFlowConfig(**values)


def test_v2_requires_explicit_vae_path_and_scale():
    with pytest.raises(ValueError, match="flow_vae_model_path"):
        v2_config(flow_vae_model_path=None).validate()
    with pytest.raises(ValueError, match="flow_color_scale"):
        v2_config(flow_color_scale=None).validate()
    with pytest.raises(ValueError, match="flow_label_source"):
        v2_config(flow_label_source=0).validate()
    with pytest.raises(ValueError, match="flow_latent_cache_manifest_sha256"):
        v2_config(flow_latent_cache_mode="strict", flow_latent_cache_dir="/cache").validate()


def test_v2_fixed_color_zero_and_invalid_are_static_white():
    flow = torch.zeros(2, 2, 2)
    valid = torch.tensor([[[True, False], [True, True]]])
    rgb = flow_to_fixed_rgb(flow, valid, 1.0)
    assert torch.equal(rgb[:, 0, 1], torch.ones(3))
    assert torch.equal(rgb[:, 0, 0], torch.ones(3))


def test_v2_scale_is_fixed_across_samples():
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    a = flow_to_fixed_rgb(torch.tensor([[[0.25]], [[0.0]]]), valid, 1.0)
    b = flow_to_fixed_rgb(torch.tensor([[[0.5]], [[0.0]]]), valid, 1.0)
    assert not torch.equal(a, b)


def test_v2_color_direction_rgb_and_magnitude_follow_verified_wheel():
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    right = flow_to_fixed_rgb(torch.tensor([[[1.0]], [[0.0]]]), valid, 1.0)[:, 0, 0]
    half_right = flow_to_fixed_rgb(torch.tensor([[[0.5]], [[0.0]]]), valid, 1.0)[:, 0, 0]
    down = flow_to_fixed_rgb(torch.tensor([[[0.0]], [[1.0]]]), valid, 1.0)[:, 0, 0]
    assert right[0] == pytest.approx(1.0) and right[1:].abs().sum() == 0
    torch.testing.assert_close(half_right, torch.tensor([1.0, 0.5, 0.5]))
    assert down[0] > down[1] > down[2]


def test_v1_and_none_defaults_remain_valid():
    assert OpticalFlowConfig().validate().optical_flow_aux_type == "none"
    assert OpticalFlowConfig(optical_flow_aux_type="dense_regression_v1", num_flow_queries=2,
                             optical_flow_loss_weight=1.0, flow_head_hidden_dim=16,
                             flow_head_num_layers=1).validate()


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def v2_batch(valid_fractions=(1.0, 0.5), *, dataset="toy", label="label-v1",
             include_nominal=True):
    size = len(valid_fractions)
    targets, masks = {}, {}
    for index, fraction in enumerate(valid_fractions):
        targets[index] = torch.zeros(2, 224, 224)
        mask = torch.zeros(1, 224, 224, dtype=torch.bool)
        mask.flatten()[:round(mask.numel() * fraction)] = True
        masks[index] = mask
    batch = {
        "input_ids": torch.ones(size, 2, dtype=torch.long),
        "flow_supervision_available": torch.ones(size, dtype=torch.bool),
        "flow_actual_delta_frames": torch.full((size,), 20),
        "flow_label_source": torch.ones(size, dtype=torch.long),
        "flow_target": targets, "flow_valid_mask": masks,
        "flow_dataset_id": [dataset] * size, "flow_camera": ["first_view"] * size,
        "flow_manifest_sha256": [_digest("manifest")] * size,
        "flow_generation_identity": [_digest("generation")] * size,
        "flow_label_identity": [_digest(label)] * size,
        "flow_label_file_sha256": [_digest("file")] * size,
        "flow_episode_id": torch.arange(size), "flow_label_episode_id": torch.arange(size),
        "flow_frame_index": torch.zeros(size, dtype=torch.long),
        "flow_target_frame_index": torch.full((size,), 20),
        "flow_fps": torch.full((size,), 10.0, dtype=torch.float64),
        "flow_source_timestamp_s": torch.zeros(size, dtype=torch.float64),
        "flow_target_timestamp_s": torch.full((size,), 2.0, dtype=torch.float64),
        "flow_actual_delta_s": torch.full((size,), 2.0, dtype=torch.float64),
    }
    if include_nominal:
        batch["flow_nominal_delta_frames"] = torch.full((size,), 20)
    return batch


class _LatentDist:
    def __init__(self, value):
        self.value = value

    def mode(self):
        return self.value


class FakeWanVAE(torch.nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.channels = channels
        self.encode_calls = 0

    def encode(self, value):
        self.encode_calls += 1
        latent = torch.zeros(value.shape[0], self.channels, 1, 2, 2,
                             device=value.device, dtype=value.dtype) + self.anchor * 0
        return SimpleNamespace(latent_dist=_LatentDist(latent))


def fake_vae_directory(root, *, weight=b"weights"):
    vae = Path(root) / "vae"
    vae.mkdir(parents=True)
    (vae / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKLWan", "z_dim": 4,
        "latents_mean": [0.0] * 4, "latents_std": [1.0] * 4}))
    (vae / "diffusion_pytorch_model.safetensors").write_bytes(weight)
    return Path(root)


def fake_builder(tmp_path, **config_overrides):
    model_path = fake_vae_directory(tmp_path / "model")
    config = v2_config(flow_vae_model_path=str(model_path), flow_latent_shape=None, **config_overrides)
    with patch.object(WanFlowTargetBuilder, "_load_vae", return_value=FakeWanVAE()):
        return WanFlowTargetBuilder(config, device=torch.device("cpu"))


def test_v2_filter_count_and_mse_share_the_095_rule(tmp_path):
    builder = fake_builder(tmp_path)
    batch = v2_batch((1.0, 0.5), include_nominal=False)
    assert select_v2_flow_indices(batch, builder.config) == [0]
    prediction = torch.ones(2, 4, 2, 2, requires_grad=True)
    outputs = optical_flow_loss(prediction, batch, builder.config, builder)
    assert outputs["optical_flow_loss_count"] == 1
    torch.testing.assert_close(outputs["optical_flow_loss"], torch.tensor(1.0))
    outputs["optical_flow_loss"].backward()
    assert prediction.grad[0].abs().sum() > 0 and prediction.grad[1].abs().sum() == 0


def test_v2_nominal_delta_fallback_and_malformed_fields():
    config = v2_config()
    assert select_v2_flow_indices(v2_batch((1.0, 1.0), include_nominal=False), config) == [0, 1]
    source_two = v2_batch((1.0,))
    source_two["flow_label_source"] = torch.tensor([2])
    assert select_v2_flow_indices(source_two, replace(config, flow_label_source=2)) == [0]
    assert select_v2_flow_indices(source_two, config) == []
    scalar = v2_batch((1.0, 1.0))
    scalar["flow_nominal_delta_frames"] = torch.tensor(20)
    with pytest.raises(ValueError, match="one value per sample"):
        select_v2_flow_indices(scalar, config)
    missing = v2_batch((1.0,))
    missing["flow_target"] = {}
    with pytest.raises(ValueError, match="missing flow_target"):
        select_v2_flow_indices(missing, config)
    with pytest.raises(ValueError, match="flow_delta_frames"):
        replace(config, flow_delta_frames=0).validate()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_v2_adapter_accepts_mixed_query_dtype_without_detach(dtype):
    head = build_optical_flow_head(8, v2_config(flow_v2_hidden_dim=8, flow_v2_num_heads=2,
        flow_v2_num_layers=1), (4, 2, 2)).float()
    queries = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
    prediction = head(queries)
    assert prediction.dtype == torch.float32 and torch.isfinite(prediction).all()
    prediction.float().square().mean().backward()
    assert queries.grad is not None and queries.grad.dtype == dtype
    assert torch.isfinite(queries.grad).all() and queries.grad.abs().sum() > 0


class SelectionTargetBuilder:
    latent_shape = (1, 1, 1)

    def __init__(self, config):
        self.config = config

    def build_targets(self, batch, device):
        indices = select_v2_flow_indices(batch, self.config)
        return (torch.zeros(len(indices), *self.latent_shape, device=device),
                torch.tensor(indices, dtype=torch.long, device=device))


class ScalarV2Model(torch.nn.Module):
    def __init__(self, *, joint=False):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(1.0))
        self.optical_flow_config = v2_config(flow_latent_shape=(1, 1, 1))
        self.target_builder = SelectionTargetBuilder(self.optical_flow_config)
        self.training_stage = "stage3_joint" if joint else "stage2_aux"
        self.joint = joint

    def forward(self, batch, *_args, **_kwargs):
        from utils.optical_flow_config import stage_loss_metadata
        prediction = self.value.expand(batch["input_ids"].shape[0], 1, 1, 1)
        flow = optical_flow_loss(prediction, batch, self.optical_flow_config, self.target_builder)
        result = {**flow, **stage_loss_metadata(self.training_stage,
            flow=bool(flow["optical_flow_loss_count"] > 0),
            ar=self.joint, fm=self.joint)}
        if self.joint:
            ar_count = batch["labels"][..., 1:].ne(-100).sum().float()
            fm_count = batch["action_mask"].sum().float()
            result.update(ar_loss_sum=self.value.square() * ar_count, ar_loss_count=ar_count,
                          flow_matching_loss_sum=self.value.square() * fm_count,
                          flow_matching_loss_count=fm_count)
        return result


class LocalAccelerator:
    from accelerate.utils import DistributedType
    device = torch.device("cpu")
    is_main_process = True
    num_processes = 1
    gradient_accumulation_steps = 1
    distributed_type = DistributedType.NO
    scaler = None

    def reduce(self, value, reduction="sum"):
        return value

    def gather(self, value):
        return value

    def unwrap_model(self, model):
        return getattr(model, "module", model)

    def no_sync(self, model):
        return nullcontext()

    def backward(self, value):
        value.backward()


def _run_v2_window(model, batch, optimizer, scheduler, accelerator=None):
    accelerator = accelerator or LocalAccelerator()
    bare = getattr(model, "module", model)
    return run_optimizer_step_window(model=model, batches=[batch], accelerator=accelerator,
        optimizer=optimizer, lr_scheduler=scheduler, training_progress=0,
        loss_type="vlm_and_action" if bare.joint else "aux",
        vlm_loss_weight=1.0 if bare.joint else 0.0,
        action_expert_loss_weight=1.0 if bare.joint else 0.0,
        next_global_step=1, optical_flow_config=bare.optical_flow_config,
        training_stage=bare.training_stage)


def test_v2_fully_filtered_of_window_preserves_all_update_state():
    model = ScalarV2Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    model.value.grad = torch.ones_like(model.value)
    before_value = model.value.detach().clone()
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    before_stage = copy.deepcopy(checkpoint_stage_metadata(model))
    result = _run_v2_window(model, v2_batch((0.5,)), optimizer, scheduler)
    assert torch.equal(before_value, model.value) and model.value.grad is None and not optimizer.state
    assert scheduler.state_dict() == before_scheduler
    assert checkpoint_stage_metadata(model) == before_stage
    assert result["optimizer_skip_reason"] == "no_supervision"
    assert advance_global_step(0, result, LocalAccelerator()) == 0


def test_v2_filtered_flow_does_not_block_joint_ar_fm_update():
    model = ScalarV2Model(joint=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    batch = v2_batch((0.5,))
    batch.update(labels=torch.tensor([[0, 1]]), action_mask=torch.ones(1, 1, dtype=torch.bool))
    result = _run_v2_window(model, batch, optimizer, scheduler)
    assert result["optimizer_update_applied"] and result["flow_active_sample_count"] == 0
    assert scheduler.last_epoch == 1 and optimizer.state
    assert checkpoint_stage_metadata(model)["completed_optimizer_windows"] == 1


def _write_cache_fixture(root):
    h5_path = root / "episode.h5"
    generation, label = _digest("generation"), _digest("label")
    with h5py.File(h5_path, "w") as handle:
        handle.attrs.update(
            schema_version="stage06_flow_v2", nominal_delta_frames=20, fps=10.0,
            generation_identity=generation, label_identity=label, dataset_id="toy",
            camera_key="first_view", merged_episode_index=0, source_episode_index=10,
            flow_units="normalized_source_image_extent", flow_direction="forward_only",
            tail_policy="clamp",
            validity_semantics="finite_and_forward_destination_in_bounds_not_occlusion",
        )
        handle["frame_index"] = np.asarray([0, 20], dtype=np.int64)
        handle["target_frame_index"] = np.asarray([20, 20], dtype=np.int64)
        handle["actual_delta_frames"] = np.asarray([20, 0], dtype=np.int16)
        handle["actual_delta_s"] = np.asarray([2.0, 0.0], dtype=np.float32)
        handle["source_timestamp_s"] = np.asarray([0.0, 2.0], dtype=np.float64)
        handle["target_timestamp_s"] = np.asarray([2.0, 2.0], dtype=np.float64)
        handle["label_source"] = np.asarray([1, 2], dtype=np.uint8)
        handle["flow"] = np.zeros((2, 2, 224, 224), dtype=np.float16)
        handle["valid_mask"] = np.ones((2, 1, 224, 224), dtype=np.uint8)
        handle["valid_fraction"] = np.ones(2, dtype=np.float32)
    file_hash = hashlib.sha256(h5_path.read_bytes()).hexdigest()
    row = {"hdf5_path": h5_path.name, "frame_count": 2, "merged_episode_index": 0,
           "source_episode_index": 10, "schema_version": "stage06_flow_manifest_v2",
           "dataset_id": "toy", "camera_key": "first_view", "generation_identity": generation,
           "label_identity": label, "sha256": file_hash}
    manifest = root / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    return manifest, row


def _current_flow_reader_fixture(root):
    from test_optical_flow_aux import fixture_manifest
    entry = fixture_manifest(root)
    entry.pop("schema_contract")
    entry["schema_version"] = "stage06_flow_manifest_v2"
    with h5py.File(root / entry["hdf5_path"], "a") as handle:
        handle.attrs["schema_version"] = "stage06_flow_v2"
        handle["valid_fraction"] = np.ones(entry["frame_count"], dtype=np.float32)
    entry["sha256"] = hashlib.sha256((root / entry["hdf5_path"]).read_bytes()).hexdigest()
    manifest = root / "current-manifest.jsonl"
    manifest.write_text(json.dumps(entry) + "\n")
    return manifest, root / entry["hdf5_path"]


def _refresh_single_manifest_hash(manifest, h5_path):
    row = json.loads(Path(manifest).read_text())
    row["sha256"] = hashlib.sha256(Path(h5_path).read_bytes()).hexdigest()
    Path(manifest).write_text(json.dumps(row) + "\n")


@pytest.mark.parametrize("mutation", ["missing", "shape", "nan", "inf", "low", "high"])
def test_current_flow_schema_valid_fraction_structure_is_strict(tmp_path, mutation):
    from utils.training_tokenization import DatasetIntegrityError
    manifest, h5_path = _current_flow_reader_fixture(tmp_path)
    with h5py.File(h5_path, "a") as handle:
        del handle["valid_fraction"]
        if mutation != "missing":
            if mutation == "shape":
                values = np.ones((2, 1), dtype=np.float32)
            else:
                values = np.ones(2, dtype=np.float32)
                values[0] = {"nan": np.nan, "inf": np.inf, "low": -.01, "high": 1.01}[mutation]
            handle["valid_fraction"] = values
    _refresh_single_manifest_hash(manifest, h5_path)
    with pytest.raises(DatasetIntegrityError, match="valid_fraction"):
        OpticalFlowReader(tmp_path, manifest).read(0, 3)


def test_current_flow_schema_checks_each_read_mask_fraction_agreement(tmp_path):
    from utils.training_tokenization import DatasetIntegrityError
    manifest, h5_path = _current_flow_reader_fixture(tmp_path)
    with h5py.File(h5_path, "a") as handle:
        handle["valid_mask"][0, 0, 0, 0] = 0
    _refresh_single_manifest_hash(manifest, h5_path)
    reader = OpticalFlowReader(tmp_path, manifest)
    with pytest.raises(DatasetIntegrityError, match="valid_fraction does not match valid_mask"):
        reader.read(0, 3)


def test_current_flow_schema_checks_fraction_for_excluded_frame_reads(tmp_path):
    from utils.training_tokenization import DatasetIntegrityError
    manifest, h5_path = _current_flow_reader_fixture(tmp_path)
    with h5py.File(h5_path, "a") as handle:
        handle["valid_fraction"][1] = 0.5
    _refresh_single_manifest_hash(manifest, h5_path)
    with pytest.raises(DatasetIntegrityError, match="valid_fraction does not match valid_mask"):
        OpticalFlowReader(tmp_path, manifest).read(0, 13)


def test_reader_rejects_unverified_hdf5_content_and_revalidates_changed_file(tmp_path, monkeypatch):
    from utils.training_tokenization import DatasetIntegrityError
    manifest, h5_path = _current_flow_reader_fixture(tmp_path)
    with h5py.File(h5_path, "a") as handle:
        handle["flow"][:] = 123.0
    with pytest.raises(DatasetIntegrityError, match="content SHA256 differs.*dataset=unversioned"):
        OpticalFlowReader(tmp_path, manifest).read(0, 3)

    _refresh_single_manifest_hash(manifest, h5_path)
    locked_hash = json.loads(manifest.read_text())["sha256"]
    reader = OpticalFlowReader(tmp_path, manifest)
    calls = {"count": 0}
    from utils import optical_flow_reader as reader_module
    original_digest = reader_module.digest_file

    def counted_digest(path):
        calls["count"] += 1
        return original_digest(path)

    monkeypatch.setattr(reader_module, "digest_file", counted_digest)
    assert reader.read(0, 3)["flow_supervision_available"]
    assert reader.read(0, 3)["flow_supervision_available"]
    assert calls["count"] == 1
    reader.close()
    with h5py.File(h5_path, "a") as handle:
        handle["flow"][0, 0, 0, 0] = 77.0
    with pytest.raises(DatasetIntegrityError, match="content SHA256 differs"):
        reader.read(0, 3)
    assert calls["count"] == 2 and json.loads(manifest.read_text())["sha256"] == locked_hash


def test_reader_rejects_replaced_hdf5_instead_of_reusing_open_handle(tmp_path):
    import shutil
    from utils.training_tokenization import DatasetIntegrityError
    manifest, h5_path = _current_flow_reader_fixture(tmp_path)
    reader = OpticalFlowReader(tmp_path, manifest)
    assert reader.read(0, 3)["flow_supervision_available"]
    replacement = tmp_path / "replacement.h5"
    shutil.copy2(h5_path, replacement)
    with h5py.File(replacement, "a") as handle:
        handle["flow"][:] = 88.0
    replacement.replace(h5_path)
    with pytest.raises(DatasetIntegrityError, match="content SHA256 differs"):
        reader.read(0, 3)
    assert not reader._handles


def test_schema_requires_current_marker_or_explicit_legacy_contract(tmp_path):
    from test_optical_flow_aux import fixture_manifest
    from utils.training_tokenization import DatasetIntegrityError
    entry = fixture_manifest(tmp_path)
    entry.pop("schema_contract")
    h5_path = tmp_path / entry["hdf5_path"]
    entry["sha256"] = hashlib.sha256(h5_path.read_bytes()).hexdigest()
    manifest = tmp_path / "unknown-schema.jsonl"
    manifest.write_text(json.dumps(entry) + "\n")
    with pytest.raises(DatasetIntegrityError, match="schema is missing"):
        OpticalFlowReader(tmp_path, manifest)

    (tmp_path / "legacy").mkdir()
    legacy = fixture_manifest(tmp_path / "legacy")
    legacy_manifest = tmp_path / "legacy/manifest.jsonl"
    legacy_manifest.write_text(json.dumps(legacy) + "\n")
    assert OpticalFlowReader(tmp_path / "legacy", legacy_manifest).read(0, 3)[
        "flow_supervision_available"]
    (tmp_path / "current").mkdir()
    current_manifest, _ = _current_flow_reader_fixture(tmp_path / "current")
    assert OpticalFlowReader(tmp_path / "current", current_manifest).read(0, 3)[
        "flow_supervision_available"]


def test_current_schema_cannot_silently_degrade_when_both_version_markers_are_removed(tmp_path):
    from utils.training_tokenization import DatasetIntegrityError
    manifest, h5_path = _current_flow_reader_fixture(tmp_path)
    row = json.loads(manifest.read_text())
    row.pop("schema_version")
    with h5py.File(h5_path, "a") as handle:
        del handle.attrs["schema_version"]
        del handle["valid_fraction"]
    row["sha256"] = hashlib.sha256(h5_path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(DatasetIntegrityError, match="schema is missing"):
        OpticalFlowReader(tmp_path, manifest)


def _strict_cache_fixture(tmp_path):
    model_path = fake_vae_directory(tmp_path / "model")
    config = v2_config(flow_vae_model_path=str(model_path), flow_latent_shape=None)
    with patch.object(WanFlowTargetBuilder, "_load_vae", return_value=FakeWanVAE()):
        online = WanFlowTargetBuilder(config, device=torch.device("cpu"))
    manifest, row = _write_cache_fixture(tmp_path)
    report = build_cache(tmp_path, manifest, tmp_path / "cache", online, limit=1)
    strict_config = replace(config, flow_latent_cache_mode="strict",
                            flow_latent_cache_dir=str(tmp_path / "cache"),
                            flow_latent_cache_manifest_sha256=report["cache_manifest_sha256"],
                            flow_latent_shape=online.latent_shape)
    with patch.object(WanFlowTargetBuilder, "_load_vae", side_effect=AssertionError("strict loaded VAE")):
        strict = WanFlowTargetBuilder(strict_config, device=torch.device("cpu"))
    with h5py.File(tmp_path / row["hdf5_path"], "r") as handle:
        batch = _sample_batch(handle, row, 0, report["manifest_sha256"], 0)
    return online, strict, batch, report


def test_official_cache_writer_is_readable_by_strict_without_loading_vae(tmp_path):
    online, strict, batch, report = _strict_cache_fixture(tmp_path)
    target, indices = strict.build_targets(batch, torch.device("cpu"))
    assert indices.tolist() == [0] and tuple(target.shape) == (1, 4, 2, 2)
    assert strict.vae is None and strict.protocol_fingerprint == online.protocol_fingerprint
    assert report["cache_manifest_entry_count"] == 1
    assert hashlib.sha256((tmp_path / "cache" / CACHE_MANIFEST_NAME).read_bytes()).hexdigest() == report[
        "cache_manifest_sha256"]
    _, created = online.write_cache_entry(batch, 0, torch.zeros(online.latent_shape),
                                           root=tmp_path / "cache")
    assert not created

    other_dataset = copy.deepcopy(batch)
    other_dataset["flow_dataset_id"] = ["other"]
    changed_label = copy.deepcopy(batch)
    changed_label["flow_label_identity"] = [_digest("label-v2")]
    assert strict.cache_path(other_dataset, 0) != strict.cache_path(batch, 0)
    assert strict.cache_path(changed_label, 0) != strict.cache_path(batch, 0)
    forged = strict.cache_path(changed_label, 0)
    forged.parent.mkdir(parents=True, exist_ok=True)
    forged.write_bytes(strict.cache_path(batch, 0).read_bytes())
    with pytest.raises(ValueError, match="absent from the locked manifest"):
        strict.build_targets(changed_label, torch.device("cpu"))


def test_official_fp16_cache_keeps_precision_gate_and_returns_fp32_target(tmp_path):
    model_path = fake_vae_directory(tmp_path / "model")
    config = v2_config(flow_vae_model_path=str(model_path), flow_latent_shape=None)
    with patch.object(WanFlowTargetBuilder, "_load_vae", return_value=FakeWanVAE()):
        online = WanFlowTargetBuilder(config, device=torch.device("cpu"))
    manifest, row = _write_cache_fixture(tmp_path)
    report = build_cache(tmp_path, manifest, tmp_path / "cache16", online, limit=1,
                         cache_dtype=torch.float16, max_fp16_error=0)
    strict_config = replace(
        config, flow_latent_cache_mode="strict", flow_latent_cache_dir=str(tmp_path / "cache16"),
        flow_latent_cache_manifest_sha256=report["cache_manifest_sha256"],
        flow_latent_shape=online.latent_shape)
    strict = WanFlowTargetBuilder(strict_config, device=torch.device("cpu"))
    with h5py.File(tmp_path / row["hdf5_path"], "r") as handle:
        batch = _sample_batch(handle, row, 0, report["manifest_sha256"], 0)
    target, _ = strict.build_targets(batch, torch.device("cpu"))
    assert target.dtype == torch.float32 and report["max_fp16_abs_error"] == 0


@pytest.mark.parametrize("mutation", ["latent", "same_shape", "file_corruption"])
def test_strict_cache_rejects_entry_content_changes(tmp_path, mutation):
    _, strict, batch, _ = _strict_cache_fixture(tmp_path)
    path = strict.cache_path(batch, 0)
    if mutation == "file_corruption":
        path.write_bytes(b"not a torch cache entry")
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["latent"] = payload["latent"] + (100 if mutation == "latent" else 1)
        if mutation == "same_shape":
            payload["content"] = latent_content_identity(payload["latent"])
        torch.save(payload, path)
    with pytest.raises(ValueError, match="cache|digest|manifest"):
        strict.build_targets(batch, torch.device("cpu"))


def test_strict_cache_rejects_manifest_tampering_after_builder_init(tmp_path):
    _, strict, batch, _ = _strict_cache_fixture(tmp_path)
    manifest_path = tmp_path / "cache" / CACHE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    record = next(iter(manifest["entries"].values()))
    record["content"]["sha256"] = _digest("forged")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="locked SHA256"):
        strict.build_targets(batch, torch.device("cpu"))


def test_strict_cache_indexes_validated_manifest_until_file_identity_changes(tmp_path, monkeypatch):
    online, _, batch, report = _strict_cache_fixture(tmp_path)
    manifest_path = tmp_path / "cache" / CACHE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    original_relative, original_record = next(iter(manifest["entries"].items()))
    for index in range(10_001):
        manifest["entries"][f"unused/entry_{index:04d}.pt"] = copy.deepcopy(original_record)
    data = (json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
    manifest_path.write_bytes(data)
    locked_sha256 = hashlib.sha256(data).hexdigest()
    strict_config = replace(
        online.config, flow_latent_cache_mode="strict", flow_latent_cache_dir=str(tmp_path / "cache"),
        flow_latent_cache_manifest_sha256=locked_sha256, flow_latent_shape=online.latent_shape,
    )
    strict = WanFlowTargetBuilder(strict_config, device=torch.device("cpu"))
    calls = {"count": 0}
    original_validate = strict._validate_cache_manifest

    def counted_validate(*args, **kwargs):
        calls["count"] += 1
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(strict, "_validate_cache_manifest", counted_validate)
    for _ in range(5):
        target, _ = strict.build_targets(batch, torch.device("cpu"))
        assert tuple(target.shape) == (1, *online.latent_shape)
    assert calls["count"] == 0

    manifest_path.write_bytes(data)
    strict.build_targets(batch, torch.device("cpu"))
    assert calls["count"] == 1

    manifest_path.write_bytes(data[:-11])
    with pytest.raises(ValueError, match="locked SHA256|invalid V2 cache manifest"):
        strict.build_targets(batch, torch.device("cpu"))
    assert calls["count"] == 1

    manifest_path.write_bytes(data)
    strict.build_targets(batch, torch.device("cpu"))
    assert calls["count"] == 2

    with patch("utils.optical_flow_v2.os.getpid", return_value=os.getpid() + 1000):
        strict.build_targets(batch, torch.device("cpu"))
        strict.build_targets(batch, torch.device("cpu"))
    assert calls["count"] == 3

    changed = json.loads(data)
    changed["entries"][original_relative]["content"]["sha256"] = _digest("tampered")
    manifest_path.write_text(json.dumps(changed, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="locked SHA256"):
        strict.build_targets(batch, torch.device("cpu"))


def test_cache_source_change_is_rejected_before_strict_target_can_be_used(tmp_path):
    from utils.training_tokenization import DatasetIntegrityError
    online, strict, batch, report = _strict_cache_fixture(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    h5_path = tmp_path / "episode.h5"
    with h5py.File(h5_path, "a") as handle:
        handle["flow"][0] = 123.0
    reader = OpticalFlowReader(tmp_path, manifest, delta_frames=20, camera_key="first_view")
    with pytest.raises(DatasetIntegrityError, match="content SHA256 differs"):
        sample = reader.read(0, 0)
        strict.build_targets(sample, torch.device("cpu"))
    with pytest.raises(DatasetIntegrityError, match="content SHA256 differs"):
        build_cache(tmp_path, manifest, tmp_path / "new-cache", online, limit=1)
    assert report["cache_manifest_entry_count"] == 1 and batch["flow_label_file_sha256"]


def test_cache_writer_rejects_existing_conflicting_content(tmp_path):
    online, strict, batch, _ = _strict_cache_fixture(tmp_path)
    path = strict.cache_path(batch, 0)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["latent"] += 100
    torch.save(payload, path)
    correct = torch.zeros(online.latent_shape)
    with pytest.raises(ValueError, match="was not overwritten"):
        online.write_cache_entry(batch, 0, correct, root=tmp_path / "cache")


def test_strict_cache_rejects_legacy_entry_without_content_digest(tmp_path):
    _, strict, batch, _ = _strict_cache_fixture(tmp_path)
    path = strict.cache_path(batch, 0)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["version"] = 2
    del payload["content"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="unsupported V2 latent cache entry"):
        strict.build_targets(batch, torch.device("cpu"))


def test_strict_checkpoint_rejects_cache_manifest_fingerprint_change(tmp_path):
    _, strict, _, _ = _strict_cache_fixture(tmp_path)
    config = strict.config
    head = build_optical_flow_head(8, config, strict.latent_shape)
    model = SimpleNamespace(optical_flow_config=config, optical_flow_aux=head,
                            flow_target_builder=strict, training_stage="stage2_aux")
    checkpoint = tmp_path / "checkpoint"
    from model.difference_query import save_difference_query_artifacts
    save_difference_query_artifacts(checkpoint, enabled=True, hidden_size=8,
                                    difference_query=torch.randn(4, 8))
    save_flow_artifacts(model, checkpoint)
    payload = read_flow_artifacts(checkpoint)
    validate_v2_runtime_payload(payload, strict, head)
    resumed_config, resumed_payload = resolve_flow_checkpoint(
        checkpoint, stage="stage2_aux", resume=True)
    with patch.object(WanFlowTargetBuilder, "_load_vae", side_effect=AssertionError("strict loaded VAE")):
        resumed_builder = WanFlowTargetBuilder(resumed_config, device=torch.device("cpu"))
    validate_v2_runtime_payload(resumed_payload, resumed_builder, head)
    assert payload["v2_protocol"]["cache_manifest"]["sha256"] == config.flow_latent_cache_manifest_sha256
    sidecar = checkpoint / "optical_flow_aux_config.json"
    changed = json.loads(sidecar.read_text())
    changed["config"]["flow_latent_cache_manifest_sha256"] = _digest("different-manifest")
    from utils.optical_flow_checkpoint import json_hash
    changed["config_sha256"] = json_hash(changed["config"])
    sidecar.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="cache manifest identity"):
        read_flow_artifacts(checkpoint)


def test_v2_checkpoint_protocol_and_weight_identity_are_enforced(tmp_path):
    model_path = fake_vae_directory(tmp_path / "model")
    config = v2_config(flow_vae_model_path=str(model_path), flow_latent_shape=None)
    with patch.object(WanFlowTargetBuilder, "_load_vae", return_value=FakeWanVAE()):
        builder = WanFlowTargetBuilder(config, device=torch.device("cpu"))
    config = replace(config, flow_latent_shape=builder.latent_shape)
    head = build_optical_flow_head(8, config, builder.latent_shape)
    model = SimpleNamespace(optical_flow_config=config, optical_flow_aux=head,
        flow_target_builder=builder, training_stage="stage2_aux")
    from model.difference_query import save_difference_query_artifacts
    save_difference_query_artifacts(tmp_path / "checkpoint", enabled=True, hidden_size=8,
                                    difference_query=torch.randn(4, 8))
    save_flow_artifacts(model, tmp_path / "checkpoint")
    payload = read_flow_artifacts(tmp_path / "checkpoint")
    validate_v2_runtime_payload(payload, builder, head)

    sidecar = tmp_path / "checkpoint/optical_flow_aux_config.json"
    original = json.loads(sidecar.read_text())
    tampered = copy.deepcopy(original)
    del tampered["v2_protocol"]["target_protocol"]["vae"]["weights_sha256"]
    tampered["v2_metadata_sha256"] = __import__("utils.optical_flow_checkpoint", fromlist=["json_hash"]).json_hash(
        tampered["v2_protocol"])
    sidecar.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="VAE identity"):
        read_flow_artifacts(tmp_path / "checkpoint")
    sidecar.write_text(json.dumps(original))

    tampered = copy.deepcopy(original)
    tampered["v2_protocol"]["target_protocol"]["color"]["scale"] = 2.0
    tampered["v2_protocol"]["target_protocol_fingerprint"] = __import__(
        "utils.optical_flow_v2", fromlist=["canonical_json_hash"]
    ).canonical_json_hash(tampered["v2_protocol"]["target_protocol"])
    tampered["v2_protocol"]["cache_protocol_fingerprint"] = tampered[
        "v2_protocol"]["target_protocol_fingerprint"]
    tampered["v2_metadata_sha256"] = __import__(
        "utils.optical_flow_checkpoint", fromlist=["json_hash"]
    ).json_hash(tampered["v2_protocol"])
    sidecar.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="target protocol conflicts"):
        read_flow_artifacts(tmp_path / "checkpoint")
    sidecar.write_text(json.dumps(original))

    (model_path / "vae/diffusion_pytorch_model.safetensors").write_bytes(b"changed-weights")
    with patch.object(WanFlowTargetBuilder, "_load_vae", return_value=FakeWanVAE()):
        changed = WanFlowTargetBuilder(replace(config, flow_latent_shape=None), device=torch.device("cpu"))
    with pytest.raises(ValueError, match="VAE weights/config/protocol"):
        validate_v2_runtime_payload(payload, changed, head)


def test_v2_action_inference_omits_head_and_never_resolves_vae(tmp_path):
    from model.reasoning_vla_model import ZR0Model
    from test_query_ar_joint_checkpoint import (
        _TinyActionExpert, _TinyBackbone, QueryArWarmStartCheckpointTest,
    )
    model_path = fake_vae_directory(tmp_path / "model")
    config = v2_config(flow_vae_model_path=str(model_path), flow_latent_shape=None,
                       flow_v2_hidden_dim=8, flow_v2_num_heads=2, flow_v2_num_layers=1)
    common_patches = (
        patch("model.reasoning_vla_model.QwenVLBackbone", _TinyBackbone),
        patch("model.reasoning_vla_model.FlowmatchingActionHead", _TinyActionExpert),
        patch("model.flow_matching_action_head.FlowmatchingActionHead", _TinyActionExpert),
    )
    for active in common_patches:
        active.start()
    try:
        with patch.object(WanFlowTargetBuilder, "_load_vae", return_value=FakeWanVAE()):
            source = ZR0Model(
                str(tmp_path / "base"), action_expert_name_or_path=None,
                action_expert_config=QueryArWarmStartCheckpointTest.action_config(),
                training_stage="stage3_joint", optical_flow_config=config,
                use_difference_query=True, num_difference_queries=4,
                tune_vlm=True, tune_action_expert=True, flow_target_device=torch.device("cpu"),
            )
        checkpoint = tmp_path / "checkpoint"
        source.save_pretrained(checkpoint)
        with patch("utils.optical_flow_v2.read_vae_identity",
                   side_effect=AssertionError("action inference resolved VAE")):
            restored = ZR0Model.from_pretrained(checkpoint, for_action_inference=True)
        assert restored.flow_target_builder is None and restored.optical_flow_aux is None
        assert restored.action_expert is not None
    finally:
        for active in reversed(common_patches):
            active.stop()


def test_explicit_cross_stage_v1_to_v2_replaces_only_the_head(tmp_path):
    from model.difference_query import save_difference_query_artifacts
    from utils.optical_flow_checkpoint import save_flow_artifacts
    v1 = OpticalFlowConfig(optical_flow_aux_type="dense_regression_v1", num_flow_queries=2,
        optical_flow_loss_weight=1.0, flow_head_hidden_dim=16, flow_head_num_layers=1)
    model = SimpleNamespace(optical_flow_config=v1,
        optical_flow_aux=build_optical_flow_head(8, v1), training_stage="stage2_aux")
    save_difference_query_artifacts(tmp_path, enabled=True, hidden_size=8,
                                    difference_query=torch.randn(4, 8))
    save_flow_artifacts(model, tmp_path)
    requested = v2_config(flow_vae_model_path="/explicit/vae")
    resolved, payload = resolve_flow_checkpoint(tmp_path, requested,
        explicit_fields=["optical_flow_aux_type", "flow_vae_model_path", "flow_color_scale"],
        stage="stage3_joint", initialize=True)
    assert resolved.optical_flow_aux_type == "wan_vae_latent_v2" and payload["replace_flow_head"]


def test_calibration_reservoir_has_strict_memory_bound_and_seeded_result():
    first, second = ReservoirQuantile(101, 7), ReservoirQuantile(101, 7)
    for start in range(0, 20_000, 317):
        values = np.arange(start, min(start + 317, 20_000), dtype=np.float32)
        first.update(values)
        second.update(values)
    assert first.size == 101 and first.seen == 20_000 and first.resident_bytes == 101 * 4
    assert np.array_equal(first.values, second.values)
    assert 0 <= first.quantile(0.99) <= 19_999


def test_calibration_split_authority_must_cover_all_episodes():
    with pytest.raises(ValueError, match="cover every episode"):
        _parse_episode_splits({"total_episodes": 2, "splits": {"train": "0:1"}})


def _write_training_index(path, manifest, frames, *, dataset_ids=("toy",), cameras=("first_view",)):
    payload = {"version": 1, "split": "train",
               "selection": "explicit_episode_frame_allowlist",
               "flow_manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
               "dataset_ids": list(dataset_ids), "cameras": list(cameras),
               "frames": [{"episode": episode, "frame": frame} for episode, frame in frames]}
    Path(path).write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _write_split_authority(root, old_to_new, splits):
    meta = Path(root) / "meta"
    data = Path(root) / "data" / "chunk-000"
    episodes = meta / "episodes" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    episodes.mkdir(parents=True)
    total = max(old_to_new.values()) + 1
    (meta / "info.json").write_text(json.dumps({
        "total_episodes": total, "splits": splits,
        "features": {"first_view": {"dtype": "image", "shape": [480, 640, 3]}},
        "fps": 10,
    }) + "\n")
    (meta / "modality.json").write_text(json.dumps({"video": {
        "first_view": {"original_key": "first_view"},
    }}) + "\n")
    (meta / "stage05_merge.json").write_text(json.dumps({"expected_dataset_id": "toy"}) + "\n")
    rows = [
        {"old_episode_index": old, "new_episode_index": new,
         "source_data_uri": f"data/chunk-000/file-{new:03d}.parquet"}
        for old, new in sorted(old_to_new.items(), key=lambda item: item[1])
    ]
    (meta / "stage05_episode_mapping.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n")
    episode_rows = []
    for row in rows:
        path = data / f"file-{row['new_episode_index']:03d}.parquet"
        table = pa.table({"episode_index": [row["new_episode_index"]] * 4,
                          "frame_index": list(range(4))})
        pq.write_table(table, path)
        episode_rows.append({
            "episode_index": row["new_episode_index"],
            "data/chunk_index": 0,
            "data/file_index": row["new_episode_index"],
            "length": 4,
        })
    pq.write_table(pa.table({key: [row[key] for row in episode_rows]
                             for key in episode_rows[0]}),
                   episodes / "file-000.parquet")
    return Path(root)


def test_calibration_uses_only_train_full_delta_source_and_quality(tmp_path):
    h5_path = tmp_path / "calibration.h5"
    generation, label = _digest("generation"), _digest("label")
    with h5py.File(h5_path, "w") as handle:
        handle.attrs.update(schema_version="stage06_flow_v2", nominal_delta_frames=20,
                            dataset_id="toy", camera_key="first_view",
                            generation_identity=generation, label_identity=label)
        handle["frame_index"] = np.arange(4)
        handle["actual_delta_frames"] = np.asarray([20, 20, 10, 0])
        handle["label_source"] = np.asarray([1, 1, 1, 2])
        flow = np.zeros((4, 2, 224, 224), dtype=np.float32)
        flow[0, 0] = 0.25
        handle["flow"] = flow
        masks = np.ones((4, 1, 224, 224), dtype=np.uint8)
        masks[1, :, :112] = 0
        handle["valid_mask"] = masks
        handle["valid_fraction"] = masks.reshape(4, -1).mean(axis=1).astype(np.float32)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"hdf5_path": h5_path.name, "frame_count": 4,
        "merged_episode_index": 0, "source_episode_index": 10,
        "schema_version": "stage06_flow_manifest_v2",
        "dataset_id": "toy", "camera_key": "first_view",
        "generation_identity": generation, "label_identity": label,
        "sha256": hashlib.sha256(h5_path.read_bytes()).hexdigest()}) + "\n")
    dataset_root = _write_split_authority(tmp_path / "dataset", {10: 0}, {"train": "0:1"})
    training_index = _write_training_index(
        tmp_path / "train-index.json", manifest, [(0, frame) for frame in range(4)])
    report = calibrate(tmp_path, manifest, expected_actual_delta=20, label_source=1,
                       min_valid_fraction=0.95, reservoir_capacity=1000,
                       quantile=0.99, seed=11, split="train", training_index=training_index,
                       dataset_root=dataset_root)
    assert report["flow_color_scale"] == pytest.approx(0.25)
    assert report["frame_filter_counts"] == {
        "manifest_frames": 4, "train_selected_frames": 4, "full_delta_frames": 2,
        "label_source_frames": 2, "valid_fraction_frames": 1}
    assert report["sampled_values"] == 1000 < report["observed_valid_values"]
    assert report["dataset_ids"] == ["toy"] and report["label_identities"] == [label]
    assert report["training_index_entry_count"] == 4
    assert report["split_authority"]["training_membership"] == "independently_verified"
    assert report["camera_contract"]["first_view"]["feature_shape"] == [480, 640, 3]
    assert report["mapping_source_identity"]["0"]["target_episode_index"] == 0
    with pytest.raises(ValueError, match="train split"):
        calibrate(tmp_path, manifest, expected_actual_delta=20, split="validation")
    with h5py.File(h5_path, "a") as handle:
        handle["flow"][0, 0, 0, 0] = 99.0
    with pytest.raises(ValueError, match="content SHA256 differs"):
        calibrate(tmp_path, manifest, expected_actual_delta=20, label_source=1,
                  training_index=training_index, dataset_root=dataset_root,
                  reservoir_capacity=1000)


def test_calibration_excludes_validation_frames_from_mixed_manifest(tmp_path):
    generation, label = _digest("generation"), _digest("label")
    rows = []
    for flow_episode, source_episode, magnitude in ((0, 10, 0.1), (1, 20, 10.0)):
        h5_path = tmp_path / f"mixed-{flow_episode}.h5"
        with h5py.File(h5_path, "w") as handle:
            handle.attrs.update(schema_version="stage06_flow_v2", nominal_delta_frames=20,
                                dataset_id="toy", camera_key="first_view",
                                generation_identity=generation, label_identity=label)
            handle["frame_index"] = np.asarray([0])
            handle["actual_delta_frames"] = np.asarray([20])
            handle["label_source"] = np.asarray([1])
            flow = np.zeros((1, 2, 224, 224), dtype=np.float32)
            flow[0, 0] = magnitude
            handle["flow"] = flow
            handle["valid_mask"] = np.ones((1, 1, 224, 224), dtype=np.uint8)
            handle["valid_fraction"] = np.ones(1, dtype=np.float32)
        rows.append({"hdf5_path": h5_path.name, "frame_count": 1,
            "merged_episode_index": flow_episode, "source_episode_index": source_episode,
            "schema_version": "stage06_flow_manifest_v2", "dataset_id": "toy",
            "camera_key": "first_view", "generation_identity": generation,
            "label_identity": label, "sha256": hashlib.sha256(h5_path.read_bytes()).hexdigest()})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    dataset_root = _write_split_authority(
        tmp_path / "dataset", {10: 0, 20: 1}, {"train": "0:1", "validation": "1:2"})
    training_index = _write_training_index(tmp_path / "train-index.json", manifest, [(0, 0)])
    report = calibrate(tmp_path, manifest, expected_actual_delta=20, reservoir_capacity=1000,
                       quantile=.99, seed=3, training_index=training_index,
                       dataset_root=dataset_root)
    assert report["flow_color_scale"] == pytest.approx(.1)
    assert report["frame_filter_counts"]["train_selected_frames"] == 1
    assert report["training_selection"] == "authoritative_dataset_split_subset"
    assert report["included_scope"] == {
        "episode_count": 1, "episode_min": 0, "episode_max": 0,
        "frame_min": 0, "frame_max": 0,
    }
    forged = _write_training_index(tmp_path / "forged-train-index.json", manifest, [(1, 0)])
    with pytest.raises(ValueError, match="outside the authoritative train split"):
        calibrate(tmp_path, manifest, expected_actual_delta=20, training_index=forged,
                  dataset_root=dataset_root)


def test_calibration_rejects_missing_or_mismatched_training_contract(tmp_path):
    h5_path = tmp_path / "flow.h5"
    generation, label = _digest("generation"), _digest("label")
    with h5py.File(h5_path, "w") as handle:
        handle.attrs.update(schema_version="stage06_flow_v2", nominal_delta_frames=20,
                            dataset_id="toy", camera_key="first_view",
                            generation_identity=generation, label_identity=label)
        handle["frame_index"] = np.asarray([0])
        handle["actual_delta_frames"] = np.asarray([20])
        handle["label_source"] = np.asarray([1])
        handle["flow"] = np.ones((1, 2, 224, 224), dtype=np.float32)
        handle["valid_mask"] = np.ones((1, 1, 224, 224), dtype=np.uint8)
        handle["valid_fraction"] = np.ones(1, dtype=np.float32)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"hdf5_path": h5_path.name, "frame_count": 1,
        "merged_episode_index": 0, "source_episode_index": 10,
        "schema_version": "stage06_flow_manifest_v2",
        "dataset_id": "toy", "camera_key": "first_view",
        "generation_identity": generation, "label_identity": label,
        "sha256": hashlib.sha256(h5_path.read_bytes()).hexdigest()}) + "\n")
    dataset_root = _write_split_authority(tmp_path / "dataset", {10: 0}, {"train": "0:1"})
    with pytest.raises(ValueError, match="explicit manifest-bound"):
        calibrate(tmp_path, manifest, expected_actual_delta=20, dataset_root=dataset_root)
    with pytest.raises(ValueError, match="requires --dataset-root"):
        calibrate(tmp_path, manifest, expected_actual_delta=20)
    index = _write_training_index(tmp_path / "train-index.json", manifest, [(0, 9)])
    with pytest.raises(ValueError, match="outside the Flow manifest"):
        calibrate(tmp_path, manifest, expected_actual_delta=20, training_index=index,
                  dataset_root=dataset_root)
    payload = json.loads(Path(index).read_text())
    payload["flow_manifest_sha256"] = _digest("other")
    Path(index).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source/split contract"):
        calibrate(tmp_path, manifest, expected_actual_delta=20, training_index=index,
                  dataset_root=dataset_root)


def test_calibration_authority_rejects_source_mapping_errors_and_marks_external_trust(tmp_path):
    h5_path = tmp_path / "flow.h5"
    generation, label = _digest("generation"), _digest("label")
    with h5py.File(h5_path, "w") as handle:
        handle.attrs.update(schema_version="stage06_flow_v2", nominal_delta_frames=20,
                            dataset_id="toy", camera_key="first_view",
                            generation_identity=generation, label_identity=label)
        handle["frame_index"] = np.asarray([0])
        handle["actual_delta_frames"] = np.asarray([20])
        handle["label_source"] = np.asarray([1])
        handle["flow"] = np.full((1, 2, 224, 224), .2, dtype=np.float32)
        handle["valid_mask"] = np.ones((1, 1, 224, 224), dtype=np.uint8)
        handle["valid_fraction"] = np.ones(1, dtype=np.float32)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"hdf5_path": h5_path.name, "frame_count": 1,
        "merged_episode_index": 7, "source_episode_index": 99,
        "schema_version": "stage06_flow_manifest_v2", "dataset_id": "toy",
        "camera_key": "first_view", "generation_identity": generation,
        "label_identity": label, "sha256": hashlib.sha256(h5_path.read_bytes()).hexdigest()}) + "\n")
    index = _write_training_index(tmp_path / "train-index.json", manifest, [(7, 0)])
    wrong_root = _write_split_authority(tmp_path / "wrong", {10: 0}, {"train": "0:1"})
    with pytest.raises(ValueError, match="cannot be mapped"):
        calibrate(tmp_path, manifest, expected_actual_delta=20, training_index=index,
                  dataset_root=wrong_root)
    source_mismatch_root = _write_split_authority(
        tmp_path / "source-mismatch", {99: 0}, {"train": "0:1"})
    (source_mismatch_root / "meta/stage05_merge.json").write_text(
        json.dumps({"expected_dataset_id": "other"}) + "\n")
    with pytest.raises(ValueError, match="dataset identity differs"):
        calibrate(tmp_path, manifest, expected_actual_delta=20, training_index=index,
                  dataset_root=source_mismatch_root)
    missing_root = tmp_path / "missing-authority"
    missing_root.mkdir()
    with pytest.raises(ValueError, match="meta/info.json"):
        calibrate(tmp_path, manifest, expected_actual_delta=20, training_index=index,
                  dataset_root=missing_root)
    malformed_root = _write_split_authority(tmp_path / "malformed", {99: 0}, {"train": "0:1"})
    mapping_path = malformed_root / "meta/stage05_episode_mapping.jsonl"
    mapping_path.write_text(mapping_path.read_text() + json.dumps({
        "old_episode_index": 100, "new_episode_index": 0,
        "source_data_uri": "data/duplicate.parquet",
    }) + "\n")
    with pytest.raises(ValueError, match="not one-to-one"):
        calibrate(tmp_path, manifest, expected_actual_delta=20, training_index=index,
                  dataset_root=malformed_root)
    report = calibrate(tmp_path, manifest, expected_actual_delta=20, training_index=index,
                       externally_trusted=True, reservoir_capacity=128)
    assert report["split_authority"] == {
        "mode": "externally_trusted_allowlist",
        "training_membership": "not_independently_verified",
        "flow_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


def test_calibration_authority_binds_camera_and_source_uri_identity(tmp_path):
    h5_path = tmp_path / "flow.h5"
    generation, label = _digest("generation"), _digest("label")
    with h5py.File(h5_path, "w") as handle:
        handle.attrs.update(schema_version="stage06_flow_v2", nominal_delta_frames=20,
                            dataset_id="toy", camera_key="first_view",
                            generation_identity=generation, label_identity=label)
        handle["frame_index"] = np.asarray([0])
        handle["actual_delta_frames"] = np.asarray([20])
        handle["label_source"] = np.asarray([1])
        handle["flow"] = np.full((1, 2, 224, 224), .2, dtype=np.float32)
        handle["valid_mask"] = np.ones((1, 1, 224, 224), dtype=np.uint8)
        handle["valid_fraction"] = np.ones(1, dtype=np.float32)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"hdf5_path": h5_path.name, "frame_count": 1,
        "merged_episode_index": 0, "source_episode_index": 10,
        "schema_version": "stage06_flow_manifest_v2", "dataset_id": "toy",
        "camera_key": "first_view", "generation_identity": generation,
        "label_identity": label, "sha256": hashlib.sha256(h5_path.read_bytes()).hexdigest()}) + "\n")
    index = _write_training_index(tmp_path / "train-index.json", manifest, [(0, 0)])

    def run(root):
        return calibrate(tmp_path, manifest, expected_actual_delta=20,
                         training_index=index, dataset_root=root, reservoir_capacity=128)

    dataset_root = _write_split_authority(tmp_path / "dataset-authority", {10: 0}, {"train": "0:1"})
    assert run(dataset_root)["split_authority"]["training_membership"] == "independently_verified"

    info = json.loads((dataset_root / "meta/info.json").read_text())
    info["features"]["not_a_dataset_camera"] = info["features"].pop("first_view")
    (dataset_root / "meta/info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="camera"):
        run(dataset_root)

    info["features"]["first_view"] = info["features"].pop("not_a_dataset_camera")
    (dataset_root / "meta/info.json").write_text(json.dumps(info))
    mapping_path = dataset_root / "meta/stage05_episode_mapping.jsonl"
    mapping = json.loads(mapping_path.read_text())
    mapping["source_data_uri"] = "data/chunk-000/missing.parquet"
    mapping_path.write_text(json.dumps(mapping) + "\n")
    with pytest.raises(ValueError, match="not an in-root parquet|absent from dataset"):
        run(dataset_root)

    mapping["source_data_uri"] = str((tmp_path / "outside.parquet").resolve())
    mapping_path.write_text(json.dumps(mapping) + "\n")
    with pytest.raises(ValueError, match="escapes dataset root"):
        run(dataset_root)

    mapping["source_data_uri"] = "data/chunk-000/file-000.parquet"
    mapping_path.write_text(json.dumps(mapping) + "\n")
    source = dataset_root / mapping["source_data_uri"]
    pq.write_table(pa.table({"episode_index": [99], "frame_index": [0]}), source)
    with pytest.raises(ValueError, match="episode identity|statistics|row count"):
        run(dataset_root)


def _write_calibration_block_fixture(root, *, frame_count=17, invalid=None):
    generation, label = _digest("generation"), _digest("label")
    h5_path = Path(root) / "flow-block.h5"
    deltas = np.full(frame_count, 20, dtype=np.int16)
    sources = np.ones(frame_count, dtype=np.uint8)
    deltas[:16] = 0
    sources[:16] = 2
    flow = np.zeros((frame_count, 2, 224, 224), dtype=np.float32)
    flow[-1, 0] = 0.5
    masks = np.ones((frame_count, 1, 224, 224), dtype=np.uint8)
    fractions = masks.reshape(frame_count, -1).mean(axis=1).astype(np.float32)
    if invalid is not None:
        kind, frame = invalid
        if kind == "nan_fraction":
            fractions[frame] = np.nan
        elif kind == "non_binary_mask":
            masks[frame, 0, 0, 0] = 2
        elif kind == "fraction_out_of_range":
            fractions[frame] = 2
        elif kind == "fraction_mismatch":
            fractions[frame] = 0
        elif kind == "nan_flow":
            flow[frame, 0, 0, 0] = np.nan
        else:
            raise AssertionError(kind)
    with h5py.File(h5_path, "w") as handle:
        handle.attrs.update(schema_version="stage06_flow_v2", nominal_delta_frames=20,
                            dataset_id="toy", camera_key="first_view",
                            generation_identity=generation, label_identity=label)
        handle["frame_index"] = np.arange(frame_count)
        handle["actual_delta_frames"] = deltas
        handle["label_source"] = sources
        handle["flow"] = flow
        handle["valid_mask"] = masks
        handle["valid_fraction"] = fractions
    manifest = Path(root) / "flow-block-manifest.jsonl"
    manifest.write_text(json.dumps({"hdf5_path": h5_path.name, "frame_count": frame_count,
        "merged_episode_index": 0, "source_episode_index": 10,
        "schema_version": "stage06_flow_manifest_v2", "dataset_id": "toy",
        "camera_key": "first_view", "generation_identity": generation,
        "label_identity": label, "sha256": hashlib.sha256(h5_path.read_bytes()).hexdigest()}) + "\n")
    authority = _write_split_authority(Path(root) / "block-dataset", {10: 0}, {"train": "0:1"})
    index = _write_training_index(Path(root) / "flow-block-index.json", manifest,
                                  [(0, frame) for frame in range(frame_count)])
    return manifest, authority, index


@pytest.mark.parametrize("invalid", [("nan_fraction", 0), ("non_binary_mask", 0),
                                      ("fraction_out_of_range", 0), ("fraction_mismatch", 0),
                                      ("nan_flow", 0)])
def test_calibration_validates_excluded_blocks_before_filtering(tmp_path, invalid):
    manifest, authority, index = _write_calibration_block_fixture(tmp_path, invalid=invalid)
    with pytest.raises(ValueError, match="valid_fraction|valid_mask|flow"):
        calibrate(tmp_path, manifest, expected_actual_delta=20,
                  training_index=index, dataset_root=authority, reservoir_capacity=128)


def test_calibration_validates_multiple_blocks_then_keeps_last_qualified_frame(tmp_path):
    manifest, authority, index = _write_calibration_block_fixture(tmp_path)
    report = calibrate(tmp_path, manifest, expected_actual_delta=20,
                       training_index=index, dataset_root=authority, reservoir_capacity=128)
    assert report["flow_color_scale"] == pytest.approx(.5)
    assert report["frame_filter_counts"] == {
        "manifest_frames": 17, "train_selected_frames": 17, "full_delta_frames": 1,
        "label_source_frames": 1, "valid_fraction_frames": 1,
    }


def distributed_v2_worker(rank, rendezvous, output):
    import torch.distributed as dist
    from accelerate.utils import DistributedType
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)

    class CPUAccelerator:
        device = torch.device("cpu")
        is_main_process = rank == 0
        num_processes = 2
        gradient_accumulation_steps = 1
        distributed_type = DistributedType.MULTI_CPU
        scaler = None

        def reduce(self, value, reduction="sum"):
            value = value.clone()
            dist.all_reduce(value)
            return value / 2 if reduction == "mean" else value

        def gather(self, value):
            values = [torch.zeros_like(value) for _ in range(2)]
            dist.all_gather(values, value)
            return torch.cat(values)

        def unwrap_model(self, model):
            return model.module

        def backward(self, value):
            value.backward()

    model = torch.nn.parallel.DistributedDataParallel(ScalarV2Model())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    result = _run_v2_window(model, v2_batch((1.0 if rank == 0 else 0.5,)),
                            optimizer, scheduler, CPUAccelerator())
    assert result["flow_eligible_samples"] == 1 and result["optimizer_update_applied"]
    if rank == 0:
        torch.save(model.module.value.detach(), output)
    dist.destroy_process_group()


def test_two_rank_uneven_v2_supervision_matches_one_sample_global_objective(tmp_path):
    import torch.multiprocessing as mp
    output = tmp_path / "distributed.pt"
    mp.spawn(distributed_v2_worker,
             args=(str(tmp_path / "rendezvous"), str(output)), nprocs=2, join=True)
    control = ScalarV2Model()
    optimizer = torch.optim.SGD(control.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    _run_v2_window(control, v2_batch((1.0,)), optimizer, scheduler)
    torch.testing.assert_close(torch.load(output, weights_only=True), control.value)


def test_real_wan_cuda_gate_is_opt_in_and_checks_resources_without_allocating_gpu():
    with pytest.raises(pytest.skip.Exception, match=REAL_WAN_CUDA_TEST_ENV):
        _require_real_wan_cuda_test(env={})
    with pytest.raises(pytest.skip.Exception, match="explicitly selected"):
        _require_real_wan_cuda_test(env={REAL_WAN_CUDA_TEST_ENV: "1"})
    calls = []

    def pass_gate(**kwargs):
        calls.append(kwargs)
        return {"gate": "GO"}

    result = _require_real_wan_cuda_test(
        env={REAL_WAN_CUDA_TEST_ENV: "1", "CUDA_VISIBLE_DEVICES": "2"},
        gate=pass_gate, cuda_available=lambda: True)
    assert result == {"gate": "GO"}
    assert calls == [{"env": {REAL_WAN_CUDA_TEST_ENV: "1", "CUDA_VISIBLE_DEVICES": "2"},
                      "expected_count": 1}]
    with pytest.raises(pytest.skip.Exception, match="resource gate did not pass"):
        _require_real_wan_cuda_test(
            env={REAL_WAN_CUDA_TEST_ENV: "1", "CUDA_VISIBLE_DEVICES": "2"},
            gate=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("busy")),
            cuda_available=lambda: pytest.fail("CUDA probe ran before the resource gate"))


def test_real_wan_builder_moves_from_cpu_to_prediction_gpu_and_backpropagates():
    _require_real_wan_cuda_test()
    model_path = "/opt/data/private/lq/models/Wan2.2-TI2V-5B-Diffusers"
    config = v2_config(flow_vae_model_path=model_path, flow_latent_shape=None)
    builder = WanFlowTargetBuilder(config, device=torch.device("cpu"))
    batch = v2_batch((1.0,))
    target, indices = builder.build_targets(batch, torch.device("cuda:0"))
    prediction = torch.zeros_like(target, requires_grad=True)
    loss = optical_flow_loss(prediction, batch, builder.config, builder)["optical_flow_loss"]
    loss.backward()
    assert indices.tolist() == [0] and target.device.type == "cuda"
    assert prediction.grad is not None and all(parameter.grad is None for parameter in builder.vae.parameters())
