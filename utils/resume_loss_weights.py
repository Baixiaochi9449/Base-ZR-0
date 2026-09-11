"""Opt-in outer-loss overrides for same-stage checkpoint resumes."""

from dataclasses import replace
from functools import wraps
import math


def loss_weight_resolver(resolver, field, value, *, on_override=None, allow_stage2_init=False):
    if field not in {"slot_loss_weight", "optical_flow_loss_weight"}:
        raise ValueError("only auxiliary outer loss weights may be overridden")
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("resume loss weight must be finite and positive")

    @wraps(resolver)
    def resolve(directory, requested=None, *, explicit_fields=None,
                resume=False, stage=None, initialize=False):
        kwargs = dict(explicit_fields=explicit_fields, resume=resume,
                      stage=stage, initialize=initialize)
        cross_stage = allow_stage2_init and initialize and not resume and stage == "stage3_joint"
        if requested is None or stage != "stage3_joint" or not (cross_stage or (resume and not initialize)):
            return resolver(directory, requested, **kwargs)
        fields = set(requested.to_dict()) if explicit_fields is None else set(explicit_fields)
        if field not in fields:
            return resolver(directory, requested, **kwargs)
        if getattr(requested, field) != value:
            raise ValueError(f"requested {field} differs from the authorized override")
        # The original resolver still validates artifacts and every other field.
        kwargs["explicit_fields"] = fields - {field}
        saved, payload = resolver(directory, requested, **kwargs)
        if cross_stage:
            import json
            from pathlib import Path
            metadata = json.loads((Path(directory) / "zr0_checkpoint_metadata.json").read_text())
            if metadata.get("training_stage") != "stage2_aux":
                raise ValueError("outer-weight initialization requires a Stage 2 source")
        if payload is None or not saved.enabled:
            raise ValueError("loss override requires a complete enabled checkpoint Head")
        effective = replace(saved, **{field: value}).validate()
        if on_override is not None:
            on_override(dict(source=str(directory), field=field,
                             saved_value=getattr(saved, field), effective_value=value))
        return effective, payload

    return resolve


def install_loss_weight_overrides(*, slot_weight, optical_flow_weight, on_override=None,
                                 allow_stage2_init=False):
    """Install in an explicit entrypoint before the model/parser imports."""
    from utils import optical_flow_checkpoint, slot_checkpoint

    originals = []
    for module, name, field, value in (
        (slot_checkpoint, "resolve_slot_checkpoint", "slot_loss_weight", slot_weight),
        (optical_flow_checkpoint, "resolve_flow_checkpoint", "optical_flow_loss_weight", optical_flow_weight),
    ):
        original = getattr(module, name)
        adapted = loss_weight_resolver(original, field, value, on_override=on_override,
                                      allow_stage2_init=allow_stage2_init)
        originals.append((module, name, original, adapted))
    for module, name, original, adapted in originals:
        setattr(module, name, adapted)

    def restore():
        for module, name, original, adapted in originals:
            if getattr(module, name) is adapted:
                setattr(module, name, original)

    return restore
