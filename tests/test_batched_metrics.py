"""Batched detached metric collectives must retain values and logging behavior."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import numpy as np
import torch

from utils.optimizer_step_loss import OptimizerStepMetricAccumulator
from utils.training_tokenization import TOKENIZATION_METRIC_SCHEMA
from utils.wandb_training_logger import WandbTrainingLogger


class LocalAccelerator:
    device = torch.device("cpu")
    num_processes = 1
    is_main_process = True

    def __init__(self):
        self.calls = []

    def reduce(self, value, reduction):
        assert not value.requires_grad
        self.calls.append(("reduce", reduction, tuple(value.shape)))
        return value.clone()

    def gather(self, value):
        assert not value.requires_grad
        self.calls.append(("gather", tuple(value.shape)))
        return value.clone()


def accumulated_metrics(accelerator, batched, mode, rank=0):
    accumulator = OptimizerStepMetricAccumulator(loss_type="vlm_and_action",
        vlm_loss_weight=1., action_expert_loss_weight=5., collect_diagnostics=True,
        device=accelerator.device, batch_metric_reductions=batched)
    for micro in range(2):
        outputs = {"ar_loss_sum": torch.tensor(float(3 + rank + micro), requires_grad=True),
                   "ar_loss_count": torch.tensor(2), "flow_matching_loss_sum": torch.tensor(float(rank)),
                   "flow_matching_loss_count": torch.tensor(rank), "training_stage": "stage3_joint",
                   "provisional": False, "ar_loss_computed": True, "fm_loss_computed": bool(rank)}
        batch = {"data_read_retry_count": torch.tensor([0, rank, 1]),
                 "action_mask": torch.tensor([[[rank > 0, False]], [[True, True]], [[False, False]]])}
        if mode != "missing":
            for key, spec in TOKENIZATION_METRIC_SCHEMA.items():
                values = torch.tensor([rank + micro, 99, 7], dtype=spec.dtype)
                valid = torch.tensor([True, False, mode == "full"])
                if key == "padding_token_count":
                    valid[:] = False
                batch[key], batch[key + "_valid"] = values, valid
        accumulator.update(outputs, batch)
        assert outputs["ar_loss_sum"].requires_grad
        assert outputs["ar_loss_sum"].grad is None
    return accumulator.finalize(accelerator)


def assert_metrics_exact(reference, candidate):
    assert reference.keys() == candidate.keys()
    for key, expected in reference.items():
        actual = candidate[key]
        if isinstance(expected, torch.Tensor):
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            assert not actual.requires_grad and torch.equal(actual, expected), key
        else:
            assert type(actual) is type(expected) and actual == expected, key


@pytest.mark.parametrize("mode", ("full", "partial", "missing"))
def test_token_metrics_are_exact_with_masks_ratios_and_absent_targets(mode):
    scalar, batched = LocalAccelerator(), LocalAccelerator()
    expected = accumulated_metrics(scalar, False, mode)
    actual = accumulated_metrics(batched, True, mode)
    assert_metrics_exact(expected, actual)
    assert len(scalar.calls) - len(batched.calls) == 4 * len(TOKENIZATION_METRIC_SCHEMA) - 2


class FakeRun:
    def __init__(self, failure=False):
        self.logs = []
        self.failure = failure

    def log(self, payload, step):
        if self.failure:
            raise ConnectionError("offline")
        self.logs.append((payload, step))


def logged_metrics(accelerator, batched, metrics, *, project="test", failure=False):
    run = FakeRun(failure)
    with patch.dict(sys.modules, {"wandb": SimpleNamespace(init=lambda **_: run)}):
        logger = WandbTrainingLogger(accelerator, project=project, run_name="test", run_id="test",
            resume="never", log_dir="/tmp/wandb-unit-test", failure_policy="required",
            batch_metric_reductions=batched)
    logger.log(step=1001, mean_metrics=metrics, scalar_metrics={"lr": 1e-5})
    return run.logs


@pytest.mark.parametrize("main", (True, False))
@pytest.mark.parametrize("numeric", (True, False))
def test_wandb_preserves_scalars_strings_booleans_and_rank_behavior(main, numeric):
    metrics = {"stage": "stage1_ar", "active": True}
    if numeric:
        metrics.update(loss=torch.tensor(0.123456789, dtype=torch.float64, requires_grad=True),
                       count=1001, norm=0.5, rate=torch.tensor(1e-5))
    scalar, batched = LocalAccelerator(), LocalAccelerator()
    scalar.is_main_process = batched.is_main_process = main
    expected = logged_metrics(scalar, False, metrics)
    actual = logged_metrics(batched, True, metrics)
    assert actual == expected
    assert len(scalar.calls) == (4 if numeric else 0)
    assert len(batched.calls) == int(numeric)
    if numeric:
        assert metrics["loss"].requires_grad and metrics["loss"].grad is None
    if not main:
        assert actual == []


def test_batched_logging_keeps_disabled_and_required_failure_behavior():
    accelerator = LocalAccelerator()
    assert logged_metrics(accelerator, True, {"loss": 1.}, project=None) == []
    assert accelerator.calls == []
    with pytest.raises(RuntimeError, match="required W&B logging failed"):
        logged_metrics(accelerator, True, {"loss": 1.}, failure=True)


def test_batching_defaults_off():
    from utils.cli_options import build_train_parser
    assert build_train_parser().get_default("batch_metric_reductions") is False


@pytest.mark.parametrize("dtype", (torch.float64, torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool))
def test_local_metric_batching_preserves_scalar_types_values_order_and_inputs(dtype):
    from train_vla import json_scalar_metrics, tensorboard_loss_value
    values = {"stage": "stage3_joint", "tensor": torch.tensor(.123456789, dtype=dtype),
              "active": False, "nested_scalar": torch.tensor([[16777217]], dtype=dtype),
              "python_double": .123456789123456789, "python_int": 2**54 + 1,
              "numpy_double": np.float64(.123456789123456789)}
    before = {name: value.clone() for name, value in values.items() if isinstance(value, torch.Tensor)}
    expected = json_scalar_metrics(values)
    actual = json_scalar_metrics(values, batch_tensors=True)
    assert list(actual) == list(expected)
    assert json.dumps(actual) == json.dumps(expected)
    for name in expected:
        assert type(actual[name]) is type(expected[name])
        if not isinstance(values[name], str):
            assert tensorboard_loss_value(actual[name]) == tensorboard_loss_value(values[name])
    for name, original in before.items():
        assert torch.equal(values[name], original)


def test_local_metric_batching_keeps_python_float_precision_and_detaches_tensors():
    from train_vla import json_scalar_metrics
    loss = torch.tensor(.125, dtype=torch.float64, requires_grad=True)
    values = {"loss": loss, "float": .123456789123456789, "nan": torch.tensor(float("nan")),
              "inf": torch.tensor(float("inf")), "negative_zero": torch.tensor(-0.)}
    expected = json_scalar_metrics(values)
    actual = json_scalar_metrics(values, batch_tensors=True)
    assert json.dumps(actual) == json.dumps(expected)
    assert actual["float"] == values["float"]
    assert loss.requires_grad and loss.grad is None


@pytest.mark.parametrize("values", ({}, {"active": True, "stage": "stage1_ar", "count": 128}))
def test_local_metric_batching_accepts_no_tensors(values):
    from train_vla import json_scalar_metrics
    assert json_scalar_metrics(values, batch_tensors=True) == json_scalar_metrics(values)


def test_local_metric_batching_rejects_non_scalars_like_original():
    from train_vla import json_scalar_metrics
    for batched in (False, True):
        with pytest.raises(ValueError, match="only one element"):
            json_scalar_metrics({"bad": torch.ones(2)}, batch_tensors=batched)


@pytest.mark.parametrize("mode", ("empty", "text", "mixed"))
def test_tensorboard_batching_keeps_all_tags_steps_scalar_and_text_values(tmp_path, mode):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter
    from train_vla import json_scalar_metrics, write_tensorboard_training_metrics
    metrics = {} if mode == "empty" else {"training_stage": "stage1_ar"}
    if mode == "mixed":
        metrics.update(vlm_loss=torch.tensor(.123456789, dtype=torch.float64, requires_grad=True),
                       count=16777217, active=True, rate=1e-5, finite=torch.tensor(float("inf")),
                       unknown=torch.tensor(float("nan")), **{"spaced name": .25})
    results, calls = {}, {}
    for batched in (False, True):
        directory = tmp_path / str(batched)
        writer = SummaryWriter(str(directory))
        local = json_scalar_metrics(metrics, batch_tensors=True) if batched else None
        with patch.object(writer.file_writer, "add_summary", wraps=writer.file_writer.add_summary) as add:
            for step in (1001, 1002, 1003):
                write_tensorboard_training_metrics(writer, metrics, step, local_scalars=local)
            calls[batched] = add.call_count
        writer.close()
        events = EventAccumulator(str(directory), size_guidance={"scalars": 0, "tensors": 0})
        events.Reload()
        results[batched] = {
            "scalars": {tag: [(event.step, event.value) for event in events.Scalars(tag)]
                        for tag in events.Tags()["scalars"]},
            "texts": {tag: [(event.step, event.tensor_proto.SerializeToString().hex())
                            for event in events.Tensors(tag)] for tag in events.Tags()["tensors"]}}
    assert json.dumps(results[True], sort_keys=True) == json.dumps(results[False], sort_keys=True)
    if mode == "mixed":
        assert calls[False] == 3 * len(metrics) and calls[True] == 6
        assert metrics["vlm_loss"].requires_grad and metrics["vlm_loss"].grad is None
    else:
        assert calls[False] == calls[True]


def gloo_worker():
    from accelerate import Accelerator
    accelerator = Accelerator(cpu=True)
    for mode in ("full", "partial", "missing"):
        expected = accumulated_metrics(accelerator, False, mode, accelerator.process_index)
        actual = accumulated_metrics(accelerator, True, mode, accelerator.process_index)
        assert_metrics_exact(expected, actual)
    metrics = {"loss": torch.tensor(accelerator.process_index + .25, dtype=torch.float64),
               "count": 1001 + accelerator.process_index, "active": True, "stage": "stage1_ar"}
    expected = logged_metrics(accelerator, False, metrics)
    actual = logged_metrics(accelerator, True, metrics)
    assert actual == expected
    if accelerator.is_main_process:
        assert actual[0][0]["loss"] == .75
        assert actual[0][0]["count"] == 1001.5
        print("RESULT " + json.dumps({"token_metrics_exact": True, "wandb_payload_exact": True}), flush=True)
    accelerator.wait_for_everyone()


def test_two_rank_gloo_metrics_match_with_distinct_rank_values():
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=2", str(Path(__file__).resolve()), "--gloo-worker"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
        text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line[7:]) for line in result.stdout.splitlines() if line.startswith("RESULT ")]
    assert records == [{"token_metrics_exact": True, "wandb_payload_exact": True}]


if __name__ == "__main__" and sys.argv[-1] == "--gloo-worker":
    gloo_worker()
