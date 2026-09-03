import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tests/optimizer_step_training_worker.py"


def test_deepspeed_config_uses_an_explicit_integer_gradient_accumulation():
    config = yaml.safe_load(
        (ROOT / "accelerate_configs/accelerate_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    gas = config["deepspeed_config"]["gradient_accumulation_steps"]
    assert isinstance(gas, int)
    assert gas > 0
    micro_batch = config["deepspeed_config"]["train_micro_batch_size_per_gpu"]
    global_batch = config["deepspeed_config"]["train_batch_size"]
    assert isinstance(micro_batch, int) and micro_batch > 0
    assert global_batch == 4 * micro_batch * gas


def _run_worker(*args, processes=1):
    command = [sys.executable]
    if processes > 1:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={processes}",
        ]
    command += [str(WORKER), *args]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "lerobot"), environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT ")]
    assert len(lines) == 1, result.stdout + result.stderr
    return json.loads(lines[0][len("RESULT ") :])


@pytest.fixture(scope="module")
def comparison_suite():
    return _run_worker("--comparison-suite")["comparisons"]


@pytest.mark.parametrize(
    "loss_type,scenario",
    [
        ("vlm", "standard"),
        ("action", "standard"),
        ("vlm_and_action", "standard"),
        ("vlm_and_action", "joint_vqa"),
        ("vlm_and_action", "joint_all_vqa"),
    ],
)
def test_real_accelerate_optimizer_update_is_invariant_to_microbatch_partition(
    comparison_suite, loss_type, scenario
):
    comparison = comparison_suite[f"{loss_type}:{scenario}"]
    gas_one = comparison["gas1"]
    gas_two = comparison["gas2"]
    assert gas_two["ar_parameter"] == pytest.approx(gas_one["ar_parameter"], abs=1e-7)
    assert gas_two["fm_parameter"] == pytest.approx(gas_one["fm_parameter"], abs=1e-7)
    assert gas_two["metrics"] == pytest.approx(gas_one["metrics"], abs=1e-7)
    assert gas_two["metrics"]["total_loss"] == pytest.approx(
        gas_two["metrics"].get("weighted_ar_loss", 0.0)
        + gas_two["metrics"].get("weighted_flow_matching_loss", 0.0),
        abs=1e-7,
    )


def test_real_accelerate_incomplete_accumulation_window_matches_single_batch(
    comparison_suite,
):
    combined = comparison_suite["partial"]["gas1"]
    partial = comparison_suite["partial"]["gas3"]
    assert partial["ar_parameter"] == pytest.approx(combined["ar_parameter"], abs=1e-7)
    assert partial["fm_parameter"] == pytest.approx(combined["fm_parameter"], abs=1e-7)
    assert partial["metrics"] == pytest.approx(combined["metrics"], abs=1e-7)


def test_two_rank_gloo_uses_global_ar_and_fm_denominators_with_zero_local_fm():
    distributed = _run_worker(
        "--loss-type",
        "vlm_and_action",
        "--gas",
        "2",
        "--distributed",
        processes=2,
    )
    # AR: x^2 sum=55/count=5; FM: x^2 sum=14/count=3 at weight=0.5.
    expected_objective = 2.0 * (55.0 * 0.25 / 5.0) + 5.0 * (14.0 * 0.25 / 3.0)
    expected_ar_gradient = 2.0 * (2.0 * 0.5 * 55.0 / 5.0)
    expected_fm_gradient = 5.0 * (2.0 * 0.5 * 14.0 / 3.0)
    assert distributed["ar_parameter"] == pytest.approx(
        0.5 - 0.01 * expected_ar_gradient, abs=1e-6
    )
    assert distributed["fm_parameter"] == pytest.approx(
        0.5 - 0.01 * expected_fm_gradient, abs=1e-6
    )
    assert distributed["metrics"]["total_loss"] == pytest.approx(expected_objective, abs=1e-6)
    assert distributed["metrics"]["flow_matching_loss"] == pytest.approx(14.0 * 0.25 / 3.0)
    assert distributed["metrics"]["context_token_count_min"] == 11.0
    assert distributed["metrics"]["context_token_count_max"] == 12.0


def test_two_rank_gloo_update_is_invariant_to_rank_redistribution():
    arguments = (
        "--loss-type",
        "vlm_and_action",
        "--gas",
        "2",
        "--distributed",
    )
    original = _run_worker(*arguments, processes=2)
    redistributed = _run_worker(
        *arguments,
        "--distributed-partition",
        "redistributed",
        processes=2,
    )
    assert redistributed["ar_parameter"] == pytest.approx(
        original["ar_parameter"], abs=1e-6
    )
    assert redistributed["fm_parameter"] == pytest.approx(
        original["fm_parameter"], abs=1e-6
    )
    for key in (
        "ar_loss",
        "ar_loss_count",
        "flow_matching_loss",
        "flow_matching_loss_count",
        "weighted_ar_loss",
        "weighted_flow_matching_loss",
        "total_loss",
    ):
        assert redistributed["metrics"][key] == pytest.approx(
            original["metrics"][key], abs=1e-6
        )


def test_optimizer_step_metrics_aggregate_all_microbatch_token_values(comparison_suite):
    result = comparison_suite["vlm:standard"]["gas2"]
    assert result["metrics"]["context_token_count"] == pytest.approx(12.0)
    assert result["metrics"]["context_token_count_min"] == 11.0
    assert result["metrics"]["context_token_count_max"] == 13.0
    assert result["metrics"]["padding_token_count"] == 2.0


def test_joint_all_vqa_window_has_zero_fm_and_no_fm_parameter_update(comparison_suite):
    result = comparison_suite["vlm_and_action:joint_all_vqa"]["gas2"]
    assert result["metrics"]["flow_matching_loss_count"] == 0.0
    assert result["metrics"]["flow_matching_loss"] == 0.0
    assert result["fm_parameter"] == 0.5
    assert result["ar_parameter"] != 0.5
