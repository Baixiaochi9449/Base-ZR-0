import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from utils.future_difference_audit import (
    FutureDifferenceTokenMeasurer,
    audit_future_difference_intervals,
    audit_future_difference_lengths,
    classify_interval_overlap,
    summarize_token_lengths,
)


class _CharacterTokenizer:
    pad_token_id = 0

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) + 1 for char in text]


class _CharacterProcessor:
    tokenizer = _CharacterTokenizer()

    @staticmethod
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        del tokenize, add_generation_prompt
        pieces = []
        for message in messages:
            pieces.append(f"<{message['role']}>")
            content = message["content"]
            if isinstance(content, str):
                pieces.append(content)
            else:
                pieces.extend(
                    "<image>" if item["type"] == "image" else item["text"]
                    for item in content
                )
        if messages[-1]["role"] == "assistant":
            pieces.append("</assistant>")
        return "".join(pieces)

    def __call__(self, *, text, images, videos, **kwargs):
        del images, videos, kwargs
        ids = torch.tensor([self.tokenizer.encode(text[0])], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class TargetLengthAuditTest(unittest.TestCase):
    def test_audit_scripts_are_directly_executable(self):
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        for script_name in (
            "audit_future_difference_lengths.py",
            "audit_future_difference_intervals.py",
        ):
            result = subprocess.run(
                [sys.executable, str(root / "scripts" / script_name), "--help"],
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_processor_measurer_uses_exact_target_boundary(self):
        measurer = FutureDifferenceTokenMeasurer(
            _CharacterProcessor(),
            camera_shapes={
                "first_view": (480, 640, 3),
                "second_view": (480, 640, 3),
                "wrist_image": (480, 640, 3),
            },
        )
        full = measurer("move the cup", "target-text", 10_000, "episode=0 frame=0 sample=0")
        termination_cut = measurer(
            "move the cup",
            "target-text",
            full["context_tokens"] + full["original_target_tokens"],
            "episode=0 frame=0 sample=0",
        )
        target_cut = measurer(
            "move the cup",
            "target-text",
            full["context_tokens"] + full["original_target_tokens"] - 1,
            "episode=0 frame=0 sample=0",
        )

        self.assertFalse(termination_cut["input_truncated"])
        self.assertTrue(termination_cut["target_truncated"])
        self.assertEqual(termination_cut["kept_target_tokens"], termination_cut["original_target_tokens"])
        self.assertGreater(termination_cut["chat_termination_tokens"], 0)
        self.assertTrue(target_cut["target_truncated"])
        self.assertEqual(
            target_cut["kept_target_tokens"], target_cut["original_target_tokens"] - 1
        )

    def test_summarizes_target_percentiles_and_separates_truncation_kinds(self):
        records = []
        for index, target_tokens in enumerate((10, 20, 30, 40, 50)):
            records.append(
                {
                    "context_tokens": 100 + index,
                    "original_target_tokens": target_tokens,
                    "kept_target_tokens": 45 if target_tokens == 50 else target_tokens,
                    "supervised_tokens": 45 if target_tokens == 50 else target_tokens,
                    "projected_sequence_tokens": 170 if index == 1 else 120 + target_tokens,
                    "input_truncated": index == 1,
                    "target_truncated": target_tokens == 50,
                }
            )

        result = summarize_token_lengths(records, max_length=160)

        self.assertEqual(result["sample_count"], 5)
        self.assertEqual(result["input_truncated_count"], 1)
        self.assertEqual(result["input_only_truncated_count"], 1)
        self.assertEqual(result["target_only_truncated_count"], 1)
        self.assertEqual(result["input_and_target_truncated_count"], 0)
        self.assertEqual(result["target_truncated_count"], 1)
        self.assertAlmostEqual(result["target_truncated_ratio"], 0.2)
        self.assertEqual(result["original_target_tokens"]["max"], 50)
        self.assertEqual(result["original_target_tokens"]["p50"], 30)
        self.assertAlmostEqual(result["original_target_tokens"]["p90"], 46)
        self.assertAlmostEqual(result["original_target_tokens"]["p95"], 48)
        self.assertAlmostEqual(result["original_target_tokens"]["p99"], 49.6)
        self.assertAlmostEqual(result["original_target_tokens"]["p99.9"], 49.96)
        self.assertEqual(result["context_tokens"]["max"], 104)
        self.assertEqual(result["projected_sequence_tokens"]["max"], 170)

    def test_reads_only_text_columns_and_does_not_modify_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
            (root / "data" / "chunk-000").mkdir(parents=True)
            (root / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "total_frames": 2,
                        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                    }
                ),
                encoding="utf-8",
            )
            pq.write_table(
                pa.table(
                    {
                        "episode_index": [0],
                        "data/chunk_index": [0],
                        "data/file_index": [0],
                    }
                ),
                root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
            )
            pq.write_table(
                pa.table({"task_index": [7], "task": ["move the cup"]}),
                root / "meta" / "tasks.parquet",
            )
            pq.write_table(
                pa.table(
                    {
                        "episode_index": [0, 0],
                        "frame_index": [0, 1],
                        "task_index": [7, 7],
                        "train_data": ["target-a", "target-b"],
                        "first_view": [b"must-not-read", b"must-not-read"],
                        "actions": [[1.0], [2.0]],
                    }
                ),
                root / "data" / "chunk-000" / "file-000.parquet",
            )
            before = _snapshot_tree(root)
            seen = []
            progress = []

            def measure(task, target, max_length, sample_id):
                seen.append((task, target, max_length, sample_id))
                target_tokens = 8 if target.endswith("a") else 18
                return {
                    "context_tokens": 20,
                    "original_target_tokens": target_tokens,
                    "kept_target_tokens": min(target_tokens, max_length - 20),
                    "supervised_tokens": min(target_tokens, max_length - 20),
                    "projected_sequence_tokens": 20 + target_tokens,
                    "input_truncated": False,
                    "target_truncated": 20 + target_tokens > max_length,
                }

            result = audit_future_difference_lengths(
                root,
                max_length=30,
                measure_sample=measure,
                canonicalize_target=lambda raw, sample_id: raw,
                progress_sample=progress.append,
            )

            self.assertEqual(result["sample_count"], 2)
            self.assertEqual(result["target_truncated_count"], 1)
            self.assertEqual(progress, [1, 2])
            self.assertEqual(seen[0], ("move the cup", "target-a", 30, "episode=0 frame=0 sample=0"))
            self.assertEqual(_snapshot_tree(root), before)


class IntervalAuditTest(unittest.TestCase):
    def test_closed_interval_classification_includes_both_endpoints(self):
        exact = classify_interval_overlap(0, 31, 0, 31)
        full = classify_interval_overlap(2, 33, 0, 40)
        partial = classify_interval_overlap(0, 31, 31, 40)
        none = classify_interval_overlap(0, 31, 32, 40)

        self.assertEqual((exact.classification, exact.intersection_length), ("exact", 32))
        self.assertEqual((full.classification, full.intersection_length), ("full", 32))
        self.assertEqual((partial.classification, partial.intersection_length), ("partial", 1))
        self.assertEqual((none.classification, none.intersection_length), ("none", 0))

    def test_uses_episode_mapping_reports_unavailable_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            annotation_root = root / "upstream" / "05_merge" / "dataset"
            (root / "meta").mkdir(parents=True)
            (root / "meta" / "info.json").write_text(
                json.dumps({"codebase_version": "v3.0-fixture"}),
                encoding="utf-8",
            )
            (annotation_root / "episode_000010" / "shard_00000" / "rank_00000").mkdir(
                parents=True
            )
            steps = [(0, 0), (0, 1), (0, 10), (0, 42), (1, 0)]
            with (root / "meta" / "steps_data_index.pkl").open("wb") as handle:
                pickle.dump({"steps": steps}, handle)
            (root / "meta" / "stage05_episode_mapping.jsonl").write_text(
                "\n".join(
                    (
                        json.dumps({"new_episode_index": 0, "old_episode_index": 10}),
                        json.dumps({"new_episode_index": 1, "old_episode_index": 11}),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "meta" / "stage05_merge.json").write_text(
                json.dumps({"stage05_dir": str(annotation_root)}),
                encoding="utf-8",
            )
            records = [
                {
                    "base_data": {
                        "semantic_anchor_frame": 0,
                        "sample_id": "old10-f0",
                        "language_action_interval": {
                            "start_frame_inclusive": 0,
                            "end_frame_inclusive": 31,
                        },
                    }
                },
                {
                    "base_data": {
                        "semantic_anchor_frame": 1,
                        "sample_id": "old10-f1",
                        "language_action_interval": {
                            "start_frame_inclusive": 1,
                            "end_frame_inclusive": 40,
                        },
                    }
                },
                {
                    "base_data": {
                        "semantic_anchor_frame": 10,
                        "sample_id": "old10-f10",
                        "language_action_interval": {
                            "start_frame_inclusive": 10,
                            "end_frame_inclusive": 20,
                        },
                    }
                },
            ]
            annotation_path = (
                annotation_root
                / "episode_000010"
                / "shard_00000"
                / "rank_00000"
                / "training_samples.test.jsonl"
            )
            annotation_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            before = _snapshot_tree(root)

            result = audit_future_difference_intervals(root, action_horizon=32)

            self.assertEqual(result["sample_count"], 5)
            self.assertEqual(result["available_count"], 4)
            self.assertEqual(result["unavailable_count"], 1)
            self.assertEqual(
                result["classification_counts"],
                {"exact": 1, "full": 1, "partial": 1, "none": 1},
            )
            self.assertEqual(result["mismatch_count"], 3)
            self.assertAlmostEqual(result["mismatch_ratio"], 0.75)
            self.assertEqual(result["unavailable_reasons"], {"annotation_episode_missing": 1})
            self.assertEqual(result["annotation_interval_length"]["min"], 11)
            self.assertEqual(result["annotation_interval_length"]["max"], 40)
            self.assertEqual(result["codebase_version"], "v3.0-fixture")
            identity_paths = {
                item["relative_path"] for item in result["metadata_identity_files"]
            }
            self.assertIn("dataset/meta/info.json", identity_paths)
            self.assertIn(
                "annotation/episode_000010/shard_00000/rank_00000/training_samples.test.jsonl",
                identity_paths,
            )
            first_identity = result["audit_identity_sha256"]
            self.assertEqual(_snapshot_tree(root), before)
            json.dumps(result)

            mapping_path = root / "meta" / "stage05_episode_mapping.jsonl"
            mapping_path.write_text(
                mapping_path.read_text(encoding="utf-8").replace("\n", " \n"),
                encoding="utf-8",
            )
            changed = audit_future_difference_intervals(root, action_horizon=32)
            self.assertNotEqual(changed["audit_identity_sha256"], first_identity)


if __name__ == "__main__":
    unittest.main()
