"""Formal launch budgets, independent sources and stage completion checks."""

import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.run_three_stage_formal import ROOT, execute, load_plan, prepare, verify_finished_stage
from scripts.run_three_stage_validation import training_command
from utils.bounded_validation import execution_limit
from utils.cli_options import parse_train_options


def test_formal_plan_uses_full_schedules_and_only_formal_predecessors():
    plan = load_plan(ROOT / "configs/three_stage_formal_20260909.json")
    for i, item in enumerate(plan["stages"]):
        opt = parse_train_options(item["command"][item["command"].index(str(ROOT / "train_vla.py")) + 1:])
        assert opt.max_train_steps == (10000, 5000, 150000)[i]
        assert execution_limit(opt, opt.max_train_steps) == opt.max_train_steps
        assert opt.save_step_interval == (1000, 1000, 2000)[i]
        assert opt.warmup_ratio * opt.max_train_steps == (800, 400, 12000)[i]
        assert opt.verify_three_stage_initialization
        assert not opt.bounded_three_stage_validation and opt.save_and_exit_after_updates is None
        assert not opt.resume_training
        assert opt.component_optimizer_groups and opt.log_training_diagnostics
        assert opt.component_update_diagnostics == "inactive"
        assert opt.fast_resume_data_skip
        assert opt.batch_metric_reductions
        assert opt.wandb_failure_policy == "required" and opt.wandb_resume == "never"
        assert opt.action_expert_config_path == plan["preparation"]["base_model"] + "/action_expert_config.json"
        expected_source = (plan["preparation"]["base_model"] if i == 0 else
            plan["stages"][i - 1]["output_dir"] + "/latest-model-optimizer-lr")
        assert opt.vlm_name_or_path == expected_source
        if i:
            assert opt.init_from_checkpoint == expected_source
        if i == 2:
            assert opt.action_expert_name_or_path == plan["preparation"]["base_model"]
            assert (opt.vlm_loss_weight, opt.slot_loss_weight, opt.optical_flow_loss_weight, opt.action_expert_loss_weight) == (1., 1., 1., 5.)
        assert opt.preparation_audit_cache.endswith("three_stage_validation_20260908/flow_exclusion_108/audit_snapshot.json")
    with pytest.raises(ValueError, match="cannot enter bounded"):
        training_command(plan["preparation"], "stage1_ar", 50, formal_run=plan["formal"])


@pytest.mark.parametrize("mutation", ["budget", "output", "warmup", "save"])
def test_formal_plan_rejects_contract_changes(tmp_path, mutation):
    plan = load_plan(ROOT / "configs/three_stage_formal_20260909.json")
    formal, prep = copy.deepcopy(plan["formal"]), copy.deepcopy(plan["preparation"])
    if mutation == "budget":
        formal["stage_updates"][0] = 20000
    elif mutation == "output":
        formal["output_root"] = prep["output_root"] + "/validation"
    else:
        key = "warmup_steps" if mutation == "warmup" else "save_step_interval"
        prep["formal_command_templates_only"]["stage1_ar"][key] += 1
    prep_path = tmp_path / "preparation.json"
    prep_path.write_text(json.dumps(prep))
    formal["preparation_config"] = str(prep_path)
    path = tmp_path / "formal.json"
    path.write_text(json.dumps(formal))
    with pytest.raises(ValueError):
        load_plan(path)


def test_formal_stage1_resume_preserves_source_schedule_and_wandb(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.slot_checkpoint.resolve_slot_checkpoint", lambda directory, requested, **kwargs: (requested, None))
    monkeypatch.setattr("utils.optical_flow_checkpoint.resolve_flow_checkpoint", lambda directory, requested, **kwargs: (requested, None))
    original = load_plan(ROOT / "configs/three_stage_formal_20260909.json")
    formal = copy.deepcopy(original["formal"])
    checkpoint = tmp_path / "original" / "latest-model-optimizer-lr"
    checkpoint.mkdir(parents=True)
    metadata = checkpoint / "zr0_checkpoint_metadata.json"
    metadata.write_text(json.dumps(dict(training_stage="stage1_ar", completed_optimizer_windows=1000)))
    formal.update(output_root=str(tmp_path / "resumed"), stage1_resume_from_checkpoint=str(checkpoint))
    path = tmp_path / "formal.json"
    path.write_text(json.dumps(formal))
    plan = load_plan(path)
    first = plan["stages"][0]
    assert first["resume_step"] == 1000
    options = first["options"]
    assert options["resume_training"] and options["resume_from_checkpoint"] == str(checkpoint)
    assert options["verify_resume_state"]
    assert options["vlm_name_or_path"] == str(checkpoint)
    assert options["action_expert_config_path"] == str(checkpoint / "action_expert_config.json")
    assert options["wandb_resume"] == "must"
    assert options["wandb_run_id"] == original["stages"][0]["options"]["wandb_run_id"]
    for key in ("max_train_steps", "save_step_interval", "warmup_ratio", "peak_learning_rate",
                "per_device_train_batch_size", "gradient_accumulation_steps", "preparation_audit_cache",
                "aux_dataset_config", "vlm_loss_weight", "action_expert_loss_weight"):
        assert options[key] == original["stages"][0]["options"][key]
    for previous, item in zip(plan["stages"], plan["stages"][1:]):
        assert not item["options"]["resume_training"]
        assert item["options"]["init_from_checkpoint"] == previous["output_dir"] + "/latest-model-optimizer-lr"
    assert plan["stages"][2]["options"]["action_expert_name_or_path"] == formal.get("base_model", original["preparation"]["base_model"])
    metadata.write_text(json.dumps(dict(training_stage="stage2_aux", completed_optimizer_windows=1000)))
    with pytest.raises(ValueError, match="stage/update count mismatch"):
        load_plan(path)


def test_formal_h50_runtime_override_is_opt_in(tmp_path):
    from train_vla import resolve_action_expert_config
    from test_query_ar_joint_checkpoint import QueryArWarmStartCheckpointTest
    config = QueryArWarmStartCheckpointTest.action_config().to_dict()
    config.update(action_horizon=50, action_dim=64, state_dim=64, max_seq_len=64)
    config["diffusion_transformer_cfg"]["max_num_positional_embeddings"] = 64
    (tmp_path / "action_expert_config.json").write_text(json.dumps(config))
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"hidden_size": 3}}))
    args = ["--training_stage", "stage1_ar", "--tune_vlm", "--vlm_name_or_path", str(tmp_path),
        "--action_expert_config_path", str(tmp_path / "action_expert_config.json"),
        "--action_horizon", "10", "--max_pad_state_and_action_length", "64"]
    opt = parse_train_options(args)
    assert not opt.verify_three_stage_initialization
    with pytest.raises(ValueError, match="horizon"):
        resolve_action_expert_config(opt)
    opt = parse_train_options(args + ["--verify_three_stage_initialization"])
    resolved = resolve_action_expert_config(opt, return_resolved=True)
    assert resolved.source_action_horizon == 50 and resolved.config.action_horizon == 10


def test_formal_completion_requires_full_rank_states_and_exact_budget(tmp_path):
    item = dict(output_dir=str(tmp_path), stage="stage1_ar", updates=2)
    metrics = [dict(step=i, total_loss=1. / i, optimizer_update_applied=True) for i in (1, 2)]
    (tmp_path / "training_metrics.jsonl").write_text("\n".join(map(json.dumps, metrics)))
    (tmp_path / "wandb_identity.json").write_text('{"url":"https://wandb.ai/test"}')
    checkpoint = tmp_path / "latest-model-optimizer-lr"
    checkpoint.mkdir()
    (checkpoint / "zr0_checkpoint_metadata.json").write_text('{"training_stage":"stage1_ar"}')
    torch.save(dict(last_epoch=2), checkpoint / "scheduler.pt")
    for rank in range(4):
        torch.save(dict(global_step=2, world_size=4), checkpoint / f"training_runtime_rank{rank}.pt")
        (checkpoint / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").touch()
    assert verify_finished_stage(item)["completed_updates"] == 2
    with pytest.raises(RuntimeError, match="stopped at 2/3"):
        verify_finished_stage({**item, "updates": 3})
    torch.save(dict(global_step=1, world_size=4), checkpoint / "training_runtime_rank3.pt")
    with pytest.raises(RuntimeError, match="rank runtime"):
        verify_finished_stage(item)


def test_formal_materialization_and_gate_failure_never_start_training(tmp_path, monkeypatch):
    formal = json.loads((ROOT / "configs/three_stage_formal_20260909.json").read_text())
    formal["output_root"] = str(tmp_path / "formal")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(formal))
    plan = load_plan(config)
    monkeypatch.setattr("utils.three_stage_preflight.implementation_identity", lambda: {})
    monkeypatch.setattr("utils.three_stage_preflight.validate_preparation", lambda config: None)
    def unavailable(**kwargs):
        raise RuntimeError("test resource gate failure")
    monkeypatch.setattr("utils.gpu_resource_gate.wait_for_gpus", unavailable)
    output = prepare(plan)
    for item in plan["stages"]:
        run = Path(item["output_dir"])
        assert (run / "experiment.md").is_file()
        assert json.loads((run / "expanded_options.json").read_text()) == item["options"]
    with pytest.raises(FileExistsError):
        prepare(plan)
    with pytest.raises(RuntimeError, match="test resource gate failure"):
        execute(plan)
    records = [json.loads(line) for line in (output / "formal_events.jsonl").read_text().splitlines()]
    assert records[-1]["status"] == "stopped" and not records[-1]["automatic_retry"]
    assert records[-1]["completed_stages"] == []
    assert not any(row.get("status") == "running" for row in records)
    with pytest.raises(FileExistsError):
        execute(plan)
