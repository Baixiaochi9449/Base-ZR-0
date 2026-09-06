#!/usr/bin/env python3
"""Diagnose a saved processor using the existing Stage05 sample path."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lerobot"))

import numpy as np
import torch
import yaml
from transformers import AutoProcessor

from utils.stage05_dataset import Stage05MixedPretrainingDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    output = Path(config["output_root"])
    processors = [AutoProcessor.from_pretrained(path, local_files_only=True)
                  for path in (config["base_model"], str(args.checkpoint))]
    base, saved = processors
    properties = {
        "processor_class": type(base) is type(saved),
        "tokenizer_class": type(base.tokenizer) is type(saved.tokenizer),
        "tokenizer_backend": json.loads(base.tokenizer.backend_tokenizer.to_str()) ==
                             json.loads(saved.tokenizer.backend_tokenizer.to_str()),
        "vocabulary": base.tokenizer.get_vocab() == saved.tokenizer.get_vocab(),
        "added_vocabulary": base.tokenizer.get_added_vocab() == saved.tokenizer.get_added_vocab(),
        "special_tokens": base.tokenizer.special_tokens_map == saved.tokenizer.special_tokens_map,
        "special_token_ids": base.tokenizer.all_special_ids == saved.tokenizer.all_special_ids,
        "effective_chat_template": base.chat_template == saved.chat_template,
        "image_processor_config": base.image_processor.to_dict() == saved.image_processor.to_dict(),
    }
    for name in ("padding_side", "truncation_side", "model_max_length",
                 "clean_up_tokenization_spaces", "split_special_tokens"):
        properties[name] = getattr(base.tokenizer, name) == getattr(saved.tokenizer, name)
    if not all(properties.values()):
        raise RuntimeError(f"unexpected loaded processor differences: {properties}")
    registry = yaml.safe_load((output / "config_view/dataset2feature.yaml").read_text())
    fixture = json.loads((output / "ar_probe_samples.json").read_text())
    records = []
    for entry_name in dict.fromkeys(item["dataset_entry"] for item in fixture):
        entry = {**registry[entry_name], "dataset_entry": entry_name}
        dataset = Stage05MixedPretrainingDataset(
            entry=entry, processor=base, loss_type="vlm", max_length=config["max_length"],
            action_horizon=config["action_horizon"], max_pad_state_and_action_length=64,
        )
        for selected in (item for item in fixture if item["dataset_entry"] == entry_name):
            index = int(selected["global_index"])
            position = int(np.searchsorted(dataset.indices, index))
            if position >= len(dataset) or int(dataset.indices[position]) != index:
                raise RuntimeError(f"sample is not eligible: {selected}")
            dataset.processor = base
            before = dataset[position]
            dataset.processor = saved
            after = dataset[position]
            keys = ("input_ids", "labels", "attention_mask", "pixel_values", "image_grid_thw")
            equality = {key: torch.equal(before[key], after[key]) for key in keys}
            image_token_id = base.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            image_positions = before["input_ids"].eq(image_token_id)
            equality["image_placeholder_positions"] = torch.equal(
                image_positions, after["input_ids"].eq(image_token_id)
            )
            record = {**selected, "equal": equality,
                      "image_placeholders": int(image_positions.sum()),
                      "image_grid_thw": before["image_grid_thw"].tolist(),
                      "pixel_values_shape": list(before["pixel_values"].shape),
                      "supervised_tokens": int(before["labels"].ne(-100).sum()),
                      "unpadded_tokens": int(before["attention_mask"].sum())}
            print(json.dumps(record, sort_keys=True), flush=True)
            records.append(record)
            if not all(equality.values()):
                raise RuntimeError(f"unexpected actual processing difference: {record}")
    report = {"base_processor": config["base_model"], "saved_processor": str(args.checkpoint.resolve()),
              "loaded_properties_equal": properties, "samples": records,
              "status": "representative_samples_equal",
              "scope": "Diagnostic samples only; not a full-dataset equivalence proof or token audit."}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
