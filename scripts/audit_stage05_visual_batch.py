#!/usr/bin/env python3
"""Decode a fixed Stage05 audit batch and save the actual 224px two-view inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.stage05_dataset import Stage05MixedPretrainingDataset


ENTRIES = (
    "stage05_droid_mixed",
    "stage05_household_mixed",
    "stage05_tabletop_mixed",
    "stage05_rh20t_mixed",
)


def _wrist_valid(dataset, global_index: int) -> bool:
    packed = np.load(
        dataset.sidecar_root / "validity_packed.npy", mmap_mode="r", allow_pickle=False
    )
    return bool((int(packed[global_index // 8, 1]) >> (7 - global_index % 8)) & 1)


def _sample_positions(dataset) -> list[int]:
    result = [0]
    if dataset.kind == "rh20t":
        first_state = _wrist_valid(dataset, int(dataset.indices[0]))
        for position, global_index in enumerate(dataset.indices):
            if _wrist_valid(dataset, int(global_index)) != first_state:
                result.append(position)
                break
    if len(result) == 1:
        result.append(len(dataset) // 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.processor_path)
    registry = yaml.safe_load((ROOT / "dataset2feature.yaml").read_text(encoding="utf-8"))
    rows = []
    records = []
    for dataset_id, entry_name in enumerate(ENTRIES):
        entry = dict(registry[entry_name])
        entry["dataset_entry"] = entry_name
        dataset = Stage05MixedPretrainingDataset(
            entry=entry,
            processor=processor,
            loss_type="vlm",
            max_length=4096,
            action_horizon=32,
            max_pad_state_and_action_length=64,
            dataset_id=dataset_id,
        )
        for position in _sample_positions(dataset):
            global_index = int(dataset.indices[position])
            episode_meta, base = dataset._episode_for_global(global_index)
            episode = int(episode_meta["episode_index"])
            source_row = dataset._episode_rows(episode)[base]
            images = dataset._images(
                episode,
                source_row,
                f"visual audit {entry_name} index={global_index}",
            )
            cells = []
            for label, image in images:
                resized = image.resize((224, 224), resample=Image.Resampling.BICUBIC)
                cell = Image.new("RGB", (224, 248), "white")
                cell.paste(resized, (0, 24))
                ImageDraw.Draw(cell).text((5, 5), label, fill="black")
                cells.append(cell)
            if len(cells) == 1:
                cells.append(Image.new("RGB", (224, 248), (235, 235, 235)))
            row = Image.new("RGB", (448, 272), "white")
            row.paste(cells[0], (0, 24))
            row.paste(cells[1], (224, 24))
            ImageDraw.Draw(row).text(
                (5, 5), f"{entry_name} ep={episode} frame={base}", fill="black"
            )
            rows.append(row)
            records.append(
                {
                    "dataset_entry": entry_name,
                    "episode_index": episode,
                    "frame_index": base,
                    "global_index": global_index,
                    "views": [label for label, _ in images],
                    "source_sizes": [list(image.size) for _, image in images],
                    "processor_resize_wh": [224, 224],
                }
            )
    canvas = Image.new("RGB", (448, 272 * len(rows)), "white")
    for row_number, row in enumerate(rows):
        canvas.paste(row, (0, row_number * 272))
    image_path = args.output_dir / "decoded_two_view_batch.png"
    canvas.save(image_path)
    report = {
        "image": str(image_path),
        "camera_order": ["main camera", "wrist camera"],
        "forbidden_views_read": False,
        "records": records,
    }
    (args.output_dir / "visual_batch.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
