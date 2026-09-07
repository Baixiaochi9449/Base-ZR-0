"""Exact release-pair compatibility, never a per-file verification bypass.

The certificate binds every byte of both dependency inventories. Its rationale
and AR equivalence checks are recorded in docs/structured_slot_review3.md.
Any subsequent implementation edit invalidates the certificate.
"""

import json
from pathlib import Path


def verified_legacy_identity(recorded, current, scope):
    path = Path(__file__).resolve().parents[1] / "configs/stage05_ar_compatibility_v1.json"
    if not path.is_file():
        return False
    certificate = json.loads(path.read_text())
    pair = certificate.get(scope, {})
    return (certificate.get("version") == 1 and recorded == pair.get("historical")
            and current == pair.get("current"))
