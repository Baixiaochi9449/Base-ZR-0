import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import GenerationConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.generation.utils import GenerationMixin


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_scripts.vlm_text import inspect_vlm_text as inspection


class TextInspectionTest(unittest.TestCase):
    def test_jsonl_preserves_reference_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            record = {"id": "math", "question": "17 * 23?", "reference": "391"}
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            args = SimpleNamespace(question=None, questions_file=path)
            self.assertEqual(inspection.read_questions(args), [record])
            path.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                inspection.read_questions(args)

    def test_invalid_questions_fail_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            for payload in ("", "[]", '{"question":" "}', '{"question":"q","reference":0}', "{"):
                with self.subTest(payload=payload):
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        inspection.read_questions(SimpleNamespace(question=None, questions_file=path))

    def test_checkpoint_rejects_adapters_and_missing_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"model_type":"qwen3_vl"}', encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                '{"weight_map":{"weight":"missing.safetensors"}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing.*shard"):
                inspection.inspect_checkpoint(root)
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "LoRA/PEFT"):
                inspection.inspect_checkpoint(root)

    def test_generation_removes_prompt_and_distinguishes_eos_at_limit(self):
        class TinyModel(torch.nn.Module):
            def __init__(self, tokens):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=64))
                self.tokens = tokens
                self.generation_config = GenerationConfig(
                    do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                    transformers_version="4.57.1",
                )

            def generate(self, **kwargs):
                self.grad_enabled = torch.is_grad_enabled()
                self.inference_enabled = torch.is_inference_mode_enabled()
                self.effective_config, _ = GenerationMixin._prepare_generation_config(
                    self, kwargs["generation_config"], use_model_defaults=kwargs.get("use_model_defaults"),
                )
                return torch.cat([kwargs["input_ids"], torch.tensor([self.tokens])], dim=1)

        class Processor:
            def __init__(self):
                self.tokenizer = self
                self.conversations = []

            def apply_chat_template(self, messages, **kwargs):
                self.conversations.append(messages)
                return BatchFeature({"input_ids": torch.tensor([[10, 11, 12]]),
                                     "attention_mask": torch.ones(1, 3, dtype=torch.long)})

            def decode(self, tokens, **kwargs):
                return repr(tokens)

        args = SimpleNamespace(system_prompt=None, max_new_tokens=2)
        for tokens, eos, truncated in (([4, 5], [99], True), ([4, 99], [98, 99], False), ([99], 99, False)):
            with self.subTest(tokens=tokens):
                model, processor = TinyModel(tokens), Processor()
                config = GenerationConfig(eos_token_id=eos, do_sample=False, max_new_tokens=2)
                answer = inspection.generate_answer(model, processor, "first", args, config)
                self.assertEqual(answer["generated_token_ids"], tokens)
                self.assertEqual(answer["prompt_token_ids"], [10, 11, 12])
                self.assertEqual(answer["truncated"], truncated)
                self.assertEqual(answer["hit_max_new_tokens"], len(tokens) == 2)
                self.assertFalse(model.grad_enabled)
                self.assertTrue(model.inference_enabled)
                self.assertFalse(model.effective_config.do_sample)
                self.assertEqual(model.effective_config.temperature, 1.0)
                self.assertEqual(model.effective_config.top_p, 1.0)
                self.assertTrue(model.generation_config.do_sample)
                inspection.generate_answer(model, processor, "second", args, config)
                self.assertEqual(len(processor.conversations[-1]), 1)
                self.assertEqual(processor.conversations[-1][0]["content"][0]["text"], "second")
        args.max_new_tokens = 64
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            inspection.generate_answer(model, processor, "long", args, SimpleNamespace(eos_token_id=99))

    def test_existing_output_and_checkpoint_subdirectory_are_never_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = inspection.build_parser().parse_args([
                "--checkpoint", str(root / "checkpoint"), "--question", "hello", "--output-dir", str(root),
            ])
            marker = root / "result.json"
            marker.write_text("original", encoding="utf-8")
            with patch.object(inspection, "inspect_checkpoint", return_value={}):
                with self.assertRaises(FileExistsError):
                    inspection.run(args)
                args.output_dir = args.checkpoint / "results"
                with self.assertRaisesRegex(ValueError, "outside the checkpoint"):
                    inspection.run(args)
            self.assertEqual(marker.read_text(encoding="utf-8"), "original")

    def test_failed_load_leaves_reviewable_failure_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = inspection.build_parser().parse_args([
                "--checkpoint", str(root / "checkpoint"), "--question", "hello",
                "--output-dir", str(root / "output"),
            ])
            with patch.object(inspection, "inspect_checkpoint", return_value={"path": str(args.checkpoint)}), \
                    patch.object(inspection, "load_model", side_effect=ValueError("bad weights")):
                with self.assertRaisesRegex(ValueError, "bad weights"):
                    inspection.run(args)
            report = json.loads((args.output_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertIn("bad weights", report["error"])
            self.assertIn("finished_at", report)
            self.assertTrue((args.output_dir / "experiment.md").is_file())


if __name__ == "__main__":
    unittest.main()
