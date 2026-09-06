import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import audit_stage05_token_lengths as audit_tool
from scripts import run_stage05_h10_experiment as runner


def files(identity):
    return [{"path": "tokenizer.json", "sha256": identity, "size": 1}]


@pytest.fixture
def audit_selection(tmp_path, monkeypatch):
    from transformers import AutoProcessor

    reports = {}
    for name, identity in (("base", "base-files"), ("saved", "saved-files")):
        directory = tmp_path if name == "base" else tmp_path / "processor_audits/audit-001"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ("token_audit_h10_format2.json" if name == "base" else "report.json")
        path.write_text(json.dumps({"processor_identity": {"model_files": files(identity), "processor_class": "same"},
                                    "data_identity": [],
                                    "required_max_length": 941, "content_hash": "content",
                                    "implementation_identity": {"sha256": "implementation"}}))
        (directory / "trusted_spec.json").write_text("{}")
        reports[name] = path
    calls = []
    monkeypatch.setattr(audit_tool, "_validate_trusted_spec", lambda *args: None)

    def validate(path, **kwargs):
        calls.append((path, kwargs))
        return 941

    monkeypatch.setattr(audit_tool, "validate_token_audit", validate)
    records = []
    monkeypatch.setattr(runner, "record", lambda *args, **kwargs: records.append(kwargs))
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(audit_tool, "_processor_runtime_identity", lambda *args: {"processor_class": "same"})
    monkeypatch.setattr(audit_tool, "_data_identities", lambda **kwargs: [])
    monkeypatch.setattr(audit_tool, "_repository_implementation_identity", lambda *args: {"sha256": "implementation"})
    env = {"ZR0_MODEL_PATH": "/base", "ZR0_MAX_LENGTH": "1024", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    return SimpleNamespace(output=tmp_path, reports=reports, calls=calls, records=records, env=env)


@pytest.mark.parametrize("processor,identity,expected", [
    ("/base", "base-files", "base"),
    ("/probe/latest-model-optimizer-lr", "saved-files", "saved"),
    ("/formal/ar/latest-model-optimizer-lr", "saved-files", "saved"),
    ("/joint/saved-at-another-path", "saved-files", "saved"),
])
def test_matching_processor_identity_reuses_report_at_any_path(audit_selection, monkeypatch, processor, identity, expected):
    fixture = audit_selection
    monkeypatch.setattr(audit_tool, "_processor_files", lambda path: files(identity))
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected full audit"))
    selected = runner.select_token_audit(fixture.output, processor, fixture.env)
    assert selected["ZR0_TOKEN_LENGTH_AUDIT"] == str(fixture.reports[expected])
    assert len(fixture.calls) == 1
    assert fixture.calls[0][1]["processor_path"] == Path(processor)
    assert fixture.calls[0][1]["repository_root"] == fixture.output / "config_view"
    assert fixture.calls[0][1]["trusted_spec_path"].name == "trusted_spec.json"
    assert "ZR0_TOKEN_LENGTH_AUDIT" not in fixture.env


def test_matching_report_validation_failure_is_not_reaudited(audit_selection, monkeypatch):
    fixture = audit_selection
    monkeypatch.setattr(audit_tool, "_processor_files", lambda path: files("saved-files"))

    def fail(*args, **kwargs):
        raise ValueError("data identity changed")

    monkeypatch.setattr(audit_tool, "validate_token_audit", fail)
    monkeypatch.setattr(audit_tool, "_data_identities", lambda **kwargs: [{"eligible_count": 2}])
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected full audit"))
    with pytest.raises(RuntimeError, match="No valid existing processor audit"):
        runner.select_token_audit(fixture.output, "/saved", fixture.env)
    assert "data identity changed" in fixture.records[-1]["validation_errors"][0]["error"]
    assert fixture.records[-1]["differences"][-1]["data_eligibility_identity_changed"] is True


def test_new_saved_identity_reports_old_new_fields_without_reauditing(audit_selection, monkeypatch):
    fixture = audit_selection
    original = {path: path.read_bytes() for path in fixture.reports.values()}
    monkeypatch.setattr(audit_tool, "_processor_files", lambda path: files("new-saved-files"))
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected full audit"))
    with pytest.raises(RuntimeError, match="No full audit started"):
        runner.select_token_audit(fixture.output, "/new-saved", fixture.env)
    result = fixture.records[-1]
    assert result["status"] == "reuse_rejected"
    saved_difference = result["differences"][-1]
    assert saved_difference["changed_fields"]["processor_identity.model_files.tokenizer.json"] == {
        "old": files("saved-files")[0], "new": files("new-saved-files")[0]}
    assert saved_difference["data_eligibility_identity_changed"] is False
    assert saved_difference["loaded_processor_runtime_identity_changed"] is False
    assert all(path.read_bytes() == content for path, content in original.items())
    assert not (fixture.output / "processor_audits/audit-002").exists()


def test_unexpected_processor_behavior_stops_before_full_audit(audit_selection, monkeypatch):
    fixture = audit_selection
    monkeypatch.setattr(audit_tool, "_processor_files", lambda path: files("new-saved-files"))
    monkeypatch.setattr(audit_tool, "_processor_runtime_identity", lambda *args: {"processor_class": "different"})
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected full audit"))
    with pytest.raises(RuntimeError, match="No full audit started"):
        runner.select_token_audit(fixture.output, "/different", fixture.env)
    result = fixture.records[-1]["differences"][-1]
    assert result["loaded_processor_runtime_identity_changed"] is True
    assert result["changed_fields"]["processor_identity.processor_class"] == {"old": "same", "new": "different"}
