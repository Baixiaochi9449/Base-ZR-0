"""Produce supervision artifacts without modifying the source dataset."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.slot_config import SlotConfig
from utils.slot_supervision import audit_slots, audit_slot_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--slot-config", help="Optional JSON SlotConfig file controlling fixed statistical rules")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.geometry_only:
        report_path = output / "slot_audit.json"
        report = json.loads(report_path.read_text())
        report["raw_geometry_scan"] = audit_slot_geometry(args.dataset_root)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report["raw_geometry_scan"], indent=2))
        return
    config = SlotConfig(**json.loads(Path(args.slot_config).read_text())) if args.slot_config else SlotConfig()
    report, stats, index = audit_slots(args.dataset_root, config.validate(), progress=lambda n: print(f"audited {n} frames", flush=True) if n % 10000 < 200 else None)
    report["raw_geometry_scan"] = audit_slot_geometry(args.dataset_root)
    for name, value in (("slot_audit.json", report), ("slot_supervision_stats.json", stats), ("slot_anchor_index.json", index)):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k in {"frames", "anchors", "carried_labels_masked", "q9_risk_range", "interval_frames"}}, indent=2))


if __name__ == "__main__":
    main()
