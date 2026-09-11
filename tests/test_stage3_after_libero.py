import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from stage3_queue_checks import predecessor_status, read_events, verify_startup


@pytest.mark.parametrize("status", ["running", "retry_wait", "launching", "process_exited"])
def test_incomplete_and_retry_do_not_launch(status):
    assert predecessor_status([dict(status=status, step=34184)], 34184) == "waiting_for_completion"


@pytest.mark.parametrize("events,disabled,alive", [
    ([dict(status="stopped")], False, True),
    ([dict(status="running")], True, True),
    ([dict(status="running")], False, False),
    ([dict(status="complete", step=34184, exit_code=1, checkpoint="cp")], False, False),
    ([dict(status="complete", step=34000, exit_code=0, checkpoint="cp")], False, False),
    ([dict(status="complete", step=34184, exit_code=0)], False, False),
])
def test_failure_interruption_and_incomplete_save_cannot_launch(events, disabled, alive):
    with pytest.raises(RuntimeError):
        predecessor_status(events, 34184, disabled=disabled, supervisor_alive=alive)


def test_complete_still_waits_for_supervisor_exit():
    events = [dict(status="complete", step=34184, exit_code=0, checkpoint="cp")]
    assert predecessor_status(events, 34184, supervisor_alive=True) == "waiting_for_exit"
    assert predecessor_status(events, 34184, supervisor_alive=False) == "ready"


def test_partial_jsonl_cannot_publish_completion(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"status":"running"}\n{"status":"complete"')
    assert read_events(path) == [dict(status="running")]
    path.write_bytes(b'{"broken"}\n')
    with pytest.raises(ValueError):
        read_events(path)


def startup_fixture(run):
    states = dict(source="original-stage3", global_step=14000, optimizer_updates=0,
        master_parameters_exact=True, optimizer_exact=True, rng_exact=True,
        sampler_cursor_exact=True, scheduler_exact=True, exposure_exact=True)
    for rank in range(4):
        (run / f"saved_state_verified_rank{rank}.json").write_text(json.dumps(dict(rank=rank, **states)))
    counts = dict(vlm=626, query=1, slot_aux=34, optical_flow_aux=35, action_expert=149)
    components = {key: dict(source="original-stage3", tensors_exact=value, parameters=10,
                           trainable_parameters=10) for key, value in counts.items()}
    (run / "component_sources_verified.json").write_text(json.dumps(components))
    (run / "wandb_identity.json").write_text(json.dumps(dict(url="https://wandb.ai/test")))
    row = dict(step=14001, scheduler_step_before=14000, scheduler_step_after=14001,
        optimizer_step_global_samples=128, optimizer_microbatches_per_rank=2,
        optimizer_update_applied=True, optimizer_update_skipped=False, wandb_remote_enabled=1,
        wandb_remote_available=1, ar_loss=0.0, slot_loss_raw=2.0, optical_flow_loss=0.0,
        flow_matching_loss=0.02, weighted_ar_loss=0.0, slot_loss_weighted=0.2,
        weighted_optical_flow_loss=0.0, weighted_flow_matching_loss=0.1, total_loss=0.3,
        gpu_peak_memory_reserved_gib=60.0)
    for group in ("vlm", "difference_query", "slot_aux", "optical_flow_aux", "action_expert"):
        active = group != "optical_flow_aux"
        row.update({group + "_grad_norm": 0.0, group + "_active_supervision": active,
                    group + "_update_applied": active})
    (run / "training_metrics.jsonl").write_text(json.dumps(row) + "\n")
    return row


def test_zero_valid_loss_and_inactive_head_are_accepted(tmp_path):
    startup_fixture(tmp_path)
    weights = dict(ar=1, slot=0.1, optical_flow=1, fm=5)
    assert verify_startup(tmp_path, weights, count=2) is None
    result = verify_startup(tmp_path, weights, count=1)
    assert result["restored_step"] == 14000 and result["diagnostic_optimizer_updates"] == 0


@pytest.mark.parametrize("change", [dict(slot_loss_weighted=1.0), dict(total_loss=9.0),
    dict(scheduler_step_after=56004), dict(wandb_remote_available=0),
    dict(optical_flow_aux_update_applied=True), dict(vlm_grad_norm=float("nan"))])
def test_startup_detects_bad_weights_state_or_logging(tmp_path, change):
    row = startup_fixture(tmp_path)
    (tmp_path / "training_metrics.jsonl").write_text(json.dumps({**row, **change}) + "\n")
    with pytest.raises(ValueError):
        verify_startup(tmp_path, dict(ar=1, slot=0.1, optical_flow=1, fm=5), count=1)


def test_expert_source_cannot_be_base_or_libero(tmp_path):
    startup_fixture(tmp_path)
    path = tmp_path / "component_sources_verified.json"
    components = json.loads(path.read_text())
    components["action_expert"]["source"] = "base"
    path.write_text(json.dumps(components))
    with pytest.raises(ValueError, match="inheritance"):
        verify_startup(tmp_path, dict(ar=1, slot=0.1, optical_flow=1, fm=5), count=1)


def test_cross_stage_startup_requires_split_component_sources(tmp_path):
    row = startup_fixture(tmp_path)
    row.update(step=1, scheduler_step_before=0, scheduler_step_after=1)
    (tmp_path / "training_metrics.jsonl").write_text(json.dumps(row) + "\n")
    path = tmp_path / "component_sources_verified.json"
    components = json.loads(path.read_text())
    sources = {name: "stage2" for name in components}
    sources["action_expert"] = "base"
    for name, component in components.items():
        component["source"] = sources[name]
    path.write_text(json.dumps(components))
    weights = dict(ar=1, slot=0.1, optical_flow=1, fm=5)
    result = verify_startup(tmp_path, weights, count=1, initial_sources=sources)
    assert result["restored_step"] == 0 and result["saved_state"] == []
    components["vlm"]["source"] = "base"
    path.write_text(json.dumps(components))
    with pytest.raises(ValueError, match="inheritance"):
        verify_startup(tmp_path, weights, count=1, initial_sources=sources)


@pytest.mark.parametrize("valid_count", [0, 1, None])
def test_omitted_inactive_head_metrics_require_zero_supervision(tmp_path, valid_count):
    row = startup_fixture(tmp_path)
    del row["optical_flow_loss"], row["weighted_optical_flow_loss"]
    if valid_count is not None:
        row["flow_active_sample_count"] = valid_count
    (tmp_path / "training_metrics.jsonl").write_text(json.dumps(row) + "\n")
    weights = dict(ar=1, slot=0.1, optical_flow=1, fm=5)
    if valid_count == 0:
        assert verify_startup(tmp_path, weights, count=1)["status"] == "passed"
    else:
        with pytest.raises(ValueError, match="absent supervision"):
            verify_startup(tmp_path, weights, count=1)


def load_queue_module():
    spec = importlib.util.spec_from_file_location("stage3_queue_test", ROOT / "scripts/watch_stage3_after_libero.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_duplicate_launch_is_idempotent(tmp_path, monkeypatch):
    module = load_queue_module()
    queue = object.__new__(module.Queue)
    queue.output = tmp_path
    queue.check_identity = lambda: None
    identity = dict(pid=123, start_ticks=42)
    class Recovery:
        @staticmethod
        def process_identity(pid):
            return identity
    queue.recovery = Recovery()
    (tmp_path / "detached_launch.json").write_text(json.dumps(dict(pid=123, identity=identity)))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("duplicate launch"))
    queue.launch()


def test_dispatched_queue_cannot_start_again(tmp_path, monkeypatch):
    module = load_queue_module()
    queue = object.__new__(module.Queue)
    queue.output = tmp_path
    queue.check_identity = lambda: None
    (tmp_path / "handoff_started.json").write_text("{}")
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("duplicate successor"))
    with pytest.raises(RuntimeError, match="already been dispatched"):
        queue.launch()


@pytest.mark.parametrize("valid_archive", [True, False])
def test_handoff_waits_for_retry_cleanup_and_validates_final_archive(tmp_path, monkeypatch, valid_archive):
    module = load_queue_module()
    queue = object.__new__(module.Queue)
    queue.output = tmp_path
    queue.predecessor = tmp_path / "libero"
    queue.predecessor.mkdir()
    queue.settings = dict(predecessor=dict(target_updates=34184, poll_seconds=15))
    queue.check_identity = lambda: None
    identity = dict(pid=123)
    (tmp_path / "predecessor_identity.json").write_text(json.dumps(dict(supervisor=identity)))
    cycles = [0]
    checkpoint = tmp_path / "final-checkpoint"
    def events(path):
        if cycles[0] == 0:
            return [dict(status="retry_wait")]
        return [dict(status="complete", step=34184, exit_code=0, checkpoint=str(checkpoint))]
    monkeypatch.setattr(module, "read_events", events)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: cycles.__setitem__(0, cycles[0] + 1))
    monkeypatch.setattr(module, "live_processes", lambda *a, **k: [identity] if cycles[0] == 1 else [])
    class Recovery:
        @staticmethod
        def process_identity(pid):
            return identity if cycles[0] == 0 else None
    class Libero:
        @staticmethod
        def latest_complete(output):
            return checkpoint
        @staticmethod
        def validate_resume(source):
            assert cycles[0] == 2 and source == checkpoint
            if not valid_archive:
                raise ValueError("incomplete optimizer")
            return 34184
        @staticmethod
        def inventory(source):
            return {"model": [123, 456]}
    queue.recovery, queue.libero = Recovery(), Libero()
    writes, emitted = [], []
    queue.write = lambda path, value: writes.append((path, value))
    queue.emit = lambda **event: emitted.append(event)
    if valid_archive:
        queue.wait_for_predecessor()
        assert len(writes) == 1 and writes[0][1]["all_training_processes_exited"]
    else:
        with pytest.raises(ValueError, match="incomplete optimizer"):
            queue.wait_for_predecessor()
        assert not writes
    assert [event["status"] for event in emitted] == ["waiting_for_completion", "waiting_for_process_cleanup", "ready"]
