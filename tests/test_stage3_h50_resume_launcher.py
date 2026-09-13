"""Verify resume commands against the preserved H50 production contract on CPU."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("h50_resume_launcher", ROOT / "scripts/run_stage3_h50.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def experiment():
    result = MODULE.Experiment(ROOT / "configs/three_stage_formal_stage3_h50_resume16000_20260912.json")
    try:
        yield result
    finally:
        result.restore()


def test_resume_retains_training_contract_and_loads_all_components(experiment):
    plan = experiment.build_plan()
    item = plan["stages"][0]
    opts = item["options"]
    source = str(experiment.resume_source)
    assert item["resume_step"] == 16000
    assert item["updates"] == opts["max_train_steps"] == 150000
    assert opts["resume_training"] and opts["verify_resume_state"]
    assert opts["init_from_checkpoint"] is None
    for key in ("vlm_name_or_path", "action_expert_name_or_path", "resume_from_checkpoint"):
        assert opts[key] == source
    assert opts["action_expert_config_path"] == source + "/action_expert_config.json"
    assert opts["action_horizon"] == 50
    assert opts["expected_global_batch_size"] == 128
    assert opts["warmup_ratio"] == 0.08
    assert opts["fast_resume_data_skip"]
    assert opts["wandb_group"] == "stage3_from_stage2_step5000_h50_slot0p1_flow1_20260911"
    assert "resume16000" in opts["wandb_run_id"]


@pytest.mark.parametrize("field,value", [("action_horizon", 10), ("peak_learning_rate", 2e-5),
                                       ("slot_loss_weight", 0.5), ("warmup_ratio", 0.05)])
def test_unintended_training_changes_are_rejected(experiment, monkeypatch, capsys, field, value):
    original = experiment.recovery.attempt_item

    def changed(*args, **kwargs):
        item = original(*args, **kwargs)
        MODULE.replace_arg(item["command"], field, value)
        return item

    monkeypatch.setattr(experiment.recovery, "attempt_item", changed)
    if field == "slot_loss_weight":
        # The existing loss adapter rejects this earlier, inside argparse.
        with pytest.raises(SystemExit) as error:
            experiment.build_plan()
        assert error.value.code == 2
        assert "differs from the authorized override" in capsys.readouterr().err
    else:
        with pytest.raises(ValueError, match="training hyperparameters"):
            experiment.build_plan()


def test_fresh_configuration_still_builds_original_plan():
    experiment = MODULE.Experiment(ROOT / "configs/three_stage_formal_stage3_h50_20260911.json")
    try:
        assert experiment.resume_source is None
        assert experiment.build_plan() == json.loads((experiment.output / "launch_plan.json").read_text())
    finally:
        experiment.restore()
