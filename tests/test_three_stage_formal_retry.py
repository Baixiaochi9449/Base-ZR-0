"""Retry recovery never changes model budgets or restarts a healthy supervisor."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_three_stage_formal import ROOT, load_plan, verify_finished_stage
from scripts.watch_three_stage_formal import (
    TAG, RecoverySupervisor, attempt_item, candidate_checkpoint, last_metric,
    preserve_checkpoint, validate_checkpoint,
)


def plan():
    return load_plan(ROOT / "configs/three_stage_formal_20260909.json")


@pytest.mark.parametrize("index", (0, 1, 2))
def test_retry_resumes_own_stage_state_without_changing_schedule_or_data(tmp_path, index, monkeypatch):
    monkeypatch.setattr("utils.slot_checkpoint.resolve_slot_checkpoint", lambda directory, requested, **kwargs: (requested, None))
    monkeypatch.setattr("utils.optical_flow_checkpoint.resolve_flow_checkpoint", lambda directory, requested, **kwargs: (requested, None))
    item = plan()["stages"][index]
    checkpoint = tmp_path / "snapshot" / TAG
    changed = attempt_item(item, tmp_path / "retry", 1, checkpoint=checkpoint)
    options = changed["options"]
    assert options["resume_training"] and options["resume_from_checkpoint"] == str(checkpoint)
    assert options["vlm_name_or_path"] == str(checkpoint) and options["init_from_checkpoint"] is None
    assert options["action_expert_config_path"] == str(checkpoint / "action_expert_config.json")
    assert options["max_train_steps"] == (10000, 5000, 150000)[index]
    for key in ("aux_dataset_config", "preparation_audit_cache", "peak_learning_rate", "warmup_ratio",
                "per_device_train_batch_size", "gradient_accumulation_steps", "expected_global_batch_size",
                "vlm_loss_weight", "slot_loss_weight", "optical_flow_loss_weight", "action_expert_loss_weight",
                "save_step_interval", "bounded_three_stage_validation", "save_and_exit_after_updates"):
        assert options[key] == item["options"][key]
    assert options["wandb_run_id"] != item["options"]["wandb_run_id"]
    assert options["wandb_resume"] == "never" and options["wandb_failure_policy"] == "required"
    if index == 2:
        assert options["action_expert_name_or_path"] == str(checkpoint)


def test_fresh_stage3_after_recovered_stage2_keeps_expert_base_source(tmp_path):
    original = plan()["stages"][2]
    predecessor = tmp_path / "stage2_recovered" / TAG
    changed = attempt_item(original, tmp_path / "run", 0, predecessor=predecessor)["options"]
    assert not changed["resume_training"]
    assert changed["init_from_checkpoint"] == str(predecessor)
    assert changed["vlm_name_or_path"] == str(predecessor)
    assert changed["action_expert_name_or_path"] == "/opt/data/private/lq/models/ZR-0"


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.optical_flow_checkpoint.read_flow_artifacts", lambda path: {})
    monkeypatch.setattr("utils.stage05_checkpoint_contract.validate_action_expert_config_provenance", lambda *args: {})
    monkeypatch.setattr("utils.three_stage_sources.weight_map", lambda path: {})
    path = tmp_path / "source" / TAG
    path.mkdir(parents=True)
    (path / "zr0_checkpoint_metadata.json").write_text(json.dumps(dict(training_stage="stage3_joint", completed_optimizer_windows=100)))
    for name in ("difference_query.safetensors", "difference_query_config.json", "tokenizer.json",
                 "preprocessor_config.json", "resolved_dataset_manifest.json", "slot_head.safetensors", "action_expert.safetensors"):
        (path / name).write_text("fixture")
    torch.save(dict(last_epoch=100, _step_count=101), path / "scheduler.pt")
    torch.save(dict(last_global_step=100, dp_world_size=4, module={"weight": torch.ones(1)}), path / "mp_rank_00_model_states.pt")
    names = ("vlm", "difference_query", "action_expert", "optical_flow_aux", "slot_aux")
    for rank in range(4):
        runtime = dict(global_step=100, world_size=4, rng={"torch": torch.ones(1)}, cursor={"batch_idx": 200}, sampler_contract={"seed": 42})
        torch.save(runtime, path / f"training_runtime_rank{rank}.pt")
        native = dict(param_groups=[dict(component=name, params=[i]) for i, name in enumerate(names)],
            state={i: dict(step=98 if name == "slot_aux" else 100, exp_avg=torch.ones(2), exp_avg_sq=torch.ones(2)) for i, name in enumerate(names)})
        torch.save(dict(optimizer_state_dict=dict(zero_stage=2, base_optimizer_state=native,
            single_partition_of_fp32_groups=[torch.ones(1 if rank == 3 and name == "slot_aux" else 2) for name in names],
            group_paddings=[1 if rank == 3 and name == "slot_aux" else 0 for name in names], partition_count=[4] * len(names))),
            path / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")
    (path / "data_seen_manifest.json").write_text(json.dumps(dict(global_step=100, total_seen=12800)))
    np.savez_compressed(path / "data_seen_state.npz", seen=np.array([10000, 1000, 800, 1000]))
    return path


def test_checkpoint_accepts_inactive_head_step_and_rejects_partial_rank(checkpoint):
    assert validate_checkpoint(checkpoint, "stage3_joint", 150000) == 100
    torch.save(dict(global_step=99, world_size=4, rng={"torch": torch.ones(1)}), checkpoint / "training_runtime_rank3.pt")
    with pytest.raises(ValueError, match="rank RNG/step"):
        validate_checkpoint(checkpoint, "stage3_joint", 150000)


def test_checkpoint_rejects_missing_adam_moment(checkpoint):
    path = checkpoint / "bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt"
    value = torch.load(path, weights_only=False)
    del value["optimizer_state_dict"]["base_optimizer_state"]["state"][4]["exp_avg_sq"]
    torch.save(value, path)
    with pytest.raises(ValueError, match="Adam state"):
        validate_checkpoint(checkpoint, "stage3_joint", 150000)


def test_preserved_checkpoint_survives_corrupted_live_save(checkpoint, tmp_path):
    destination = tmp_path / "archive" / "step-000100"
    saved = preserve_checkpoint(checkpoint, destination, "stage3_joint", 150000)
    assert (destination / "checkpoint_complete.json").is_file()
    (checkpoint / "scheduler.pt").write_bytes(b"partial save")
    events = []
    chosen = candidate_checkpoint(dict(output_dir=str(checkpoint.parent), stage="stage3_joint", updates=150000),
        tmp_path / "archive", lambda **value: events.append(value))
    assert chosen == saved
    assert events[0]["status"] == "checkpoint_rejected"


def test_retry_before_first_new_save_retains_original_resume_checkpoint(checkpoint, tmp_path):
    item = dict(output_dir=str(tmp_path / "resumed"), stage="stage3_joint", updates=150000,
                options={"resume_from_checkpoint": str(checkpoint)})
    assert candidate_checkpoint(item, tmp_path / "archive", lambda **value: None) == checkpoint


def test_copy_detects_source_mutation_and_does_not_publish(checkpoint, tmp_path, monkeypatch):
    import shutil
    original = shutil.copytree
    def changed(source, destination):
        result = original(source, destination)
        (source / "tokenizer.json").write_text("concurrently changed source")
        return result
    monkeypatch.setattr("scripts.watch_three_stage_formal.shutil.copytree", changed)
    destination = tmp_path / "archive" / "step-000100"
    with pytest.raises(ValueError, match="changed while copying"):
        preserve_checkpoint(checkpoint, destination, "stage3_joint", 150000)
    assert not destination.exists()


def supervisor_fixture(tmp_path, monkeypatch):
    value = copy.deepcopy(plan())
    value["formal"]["output_root"] = str(tmp_path)
    value["stages"] = value["stages"][:1]
    (tmp_path / "code_identity.json").write_text('{"runtime":{}}')
    supervisor = RecoverySupervisor(dict(max_retries_per_stage=3, retry_delays_seconds=[60, 120, 240], poll_seconds=30), value)
    monkeypatch.setattr(supervisor, "check_runtime", lambda: None)
    monkeypatch.setattr("scripts.watch_three_stage_formal.time.sleep", lambda seconds: None)
    return supervisor


def test_healthy_process_is_not_restarted_and_pid_reuse_is_not_followed(tmp_path, monkeypatch):
    supervisor = supervisor_fixture(tmp_path, monkeypatch)
    original = dict(pid=123, start_ticks=42)
    identities = iter([original, original, {**original, "start_ticks": 99}])
    monkeypatch.setattr("scripts.watch_three_stage_formal.process_identity", lambda pid: next(identities))
    observed, recovered = [], []
    monkeypatch.setattr(supervisor, "emit", lambda **values: observed.append(values))
    monkeypatch.setattr(supervisor, "preserve", lambda *args: observed.append("preserve"))
    monkeypatch.setattr(supervisor, "recover", lambda events: recovered.append(events))
    supervisor.watch_original(original)
    assert observed.count("preserve") == 2 and len(recovered) == 1
    assert observed[0]["active_training_restarted"] is False


def test_manual_interruption_is_not_retried(tmp_path, monkeypatch):
    supervisor = supervisor_fixture(tmp_path, monkeypatch)
    (tmp_path / "formal_events.jsonl").write_text(json.dumps(dict(status="stopped", error="KeyboardInterrupt()")) + "\n")
    monkeypatch.setattr(supervisor, "emit", lambda **values: None)
    with pytest.raises(KeyboardInterrupt, match="deliberately interrupted"):
        supervisor.watch_original(None)


def test_persistent_failure_has_exactly_three_retries(tmp_path, monkeypatch):
    supervisor = supervisor_fixture(tmp_path, monkeypatch)
    events = []
    monkeypatch.setattr(supervisor, "emit", lambda **values: events.append(values))
    def fail(**kwargs):
        raise RuntimeError("GPU gate unavailable")
    monkeypatch.setattr("utils.gpu_resource_gate.wait_for_gpus", fail)
    with pytest.raises(RuntimeError, match="limit exhausted"):
        supervisor.recover([dict(status="running", stage="stage1_ar")])
    assert [event["retry"] for event in events if event["status"] == "retry_failed"] == [1, 2, 3]


def test_partial_metric_line_does_not_erase_previous_success(tmp_path):
    (tmp_path / "training_metrics.jsonl").write_text('{"step":42,"optimizer_update_applied":true}\n{"step":43')
    assert last_metric(tmp_path)["step"] == 42


def test_completion_accepts_only_updates_after_resume(tmp_path):
    run = tmp_path / "attempt"
    run.mkdir()
    checkpoint = tmp_path / TAG
    checkpoint.mkdir()
    (run / "training_metrics.jsonl").write_text(json.dumps(dict(step=100, total_loss=.5, optimizer_update_applied=True)) + "\n")
    (run / "wandb_identity.json").write_text('{}')
    (checkpoint / "zr0_checkpoint_metadata.json").write_text('{"training_stage":"stage1_ar"}')
    torch.save(dict(last_epoch=100), checkpoint / "scheduler.pt")
    for rank in range(4):
        torch.save(dict(global_step=100, world_size=4), checkpoint / f"training_runtime_rank{rank}.pt")
        (checkpoint / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").touch()
    item = dict(output_dir=str(run), stage="stage1_ar", updates=100)
    assert verify_finished_stage(item, start_step=99, checkpoint=checkpoint)["completed_updates"] == 100
    with pytest.raises(RuntimeError, match="update count"):
        verify_finished_stage(item, start_step=98, checkpoint=checkpoint)
