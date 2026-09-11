from dataclasses import dataclass, replace

import pytest

from utils.resume_loss_weights import loss_weight_resolver


@dataclass(frozen=True)
class Config:
    slot_loss_weight: float = 1.0
    optical_flow_loss_weight: float = 10.0
    num_queries: int = 32
    enabled: bool = True

    def to_dict(self):
        return dict(vars(self))

    def validate(self):
        return self


SAVED = Config()
PAYLOAD = {"training_stage": "stage3_joint", "complete": True}


def strict_resolver(directory, requested=None, *, explicit_fields=None, **kwargs):
    if directory == "corrupt":
        raise ValueError("corrupt checkpoint")
    if requested is not None:
        fields = requested.to_dict() if explicit_fields is None else explicit_fields
        if any(getattr(requested, key) != getattr(SAVED, key) for key in fields):
            raise ValueError("checkpoint config conflict")
    return SAVED, PAYLOAD


@pytest.mark.parametrize("field,value", [("slot_loss_weight", 0.5), ("optical_flow_loss_weight", 5.0),
    ("slot_loss_weight", 0.1), ("optical_flow_loss_weight", 1.0)])
def test_only_authorized_outer_weight_changes(field, value):
    requested = replace(SAVED, **{field: value})
    with pytest.raises(ValueError, match="config conflict"):
        strict_resolver("source", requested)
    events = []
    resolver = loss_weight_resolver(strict_resolver, field, value, on_override=events.append)
    effective, payload = resolver("source", requested, resume=True, stage="stage3_joint")
    assert effective == requested
    assert payload is PAYLOAD
    assert SAVED == Config()
    assert events == [dict(source="source", field=field, saved_value=getattr(SAVED, field), effective_value=value)]
    with pytest.raises(ValueError, match="config conflict"):
        resolver("source", replace(requested, num_queries=16), resume=True, stage="stage3_joint")
    with pytest.raises(ValueError, match="corrupt"):
        resolver("corrupt", requested, resume=True, stage="stage3_joint")


@pytest.mark.parametrize("kwargs", [dict(resume=False, stage="stage3_joint"),
    dict(resume=True, stage="stage2_aux"), dict(resume=True, stage="stage3_joint", initialize=True)])
def test_other_loading_paths_remain_strict(kwargs):
    resolver = loss_weight_resolver(strict_resolver, "slot_loss_weight", 0.5)
    with pytest.raises(ValueError, match="config conflict"):
        resolver("source", replace(SAVED, slot_loss_weight=0.5), **kwargs)


def test_omitted_override_inherits_checkpoint_and_wrong_override_fails():
    resolver = loss_weight_resolver(strict_resolver, "slot_loss_weight", 0.5)
    assert resolver("source", SAVED, resume=True, stage="stage3_joint", explicit_fields={"num_queries"})[0] == SAVED
    assert resolver("source")[0] == SAVED
    with pytest.raises(ValueError, match="authorized override"):
        resolver("source", replace(SAVED, slot_loss_weight=0.2), resume=True, stage="stage3_joint")


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_weight_rejected(value):
    with pytest.raises(ValueError, match="finite and positive"):
        loss_weight_resolver(strict_resolver, "slot_loss_weight", value)


def test_architecture_field_cannot_be_authorized():
    with pytest.raises(ValueError, match="only auxiliary outer"):
        loss_weight_resolver(strict_resolver, "num_queries", 16)


@pytest.mark.parametrize("field,value", [("slot_loss_weight", 0.1), ("optical_flow_loss_weight", 1.0)])
def test_explicit_stage2_initialization_override(tmp_path, field, value):
    import json
    metadata = tmp_path / "zr0_checkpoint_metadata.json"
    metadata.write_text(json.dumps(dict(training_stage="stage2_aux")))
    requested = replace(SAVED, **{field: value})
    resolver = loss_weight_resolver(strict_resolver, field, value, allow_stage2_init=True)
    assert resolver(tmp_path, requested, stage="stage3_joint", initialize=True)[0] == requested
    with pytest.raises(ValueError, match="config conflict"):
        resolver(tmp_path, replace(requested, num_queries=16), stage="stage3_joint", initialize=True)
    metadata.write_text(json.dumps(dict(training_stage="stage1_ar")))
    with pytest.raises(ValueError, match="Stage 2 source"):
        resolver(tmp_path, requested, stage="stage3_joint", initialize=True)
