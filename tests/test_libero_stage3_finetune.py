import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("libero_entry", ROOT / "scripts/train_libero_finetune.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
sys.path.insert(0, str(ROOT / "scripts"))
import watch_libero_finetune as supervisor


class LiberoStage3Tests(unittest.TestCase):
    def test_exact_comparison_uses_all_inherited_components_and_no_heads(self):
        model = SimpleNamespace(slot_aux=None, optical_flow_aux=None, training_stage=None,
            num_difference_queries=32, backbone=SimpleNamespace(
                model=torch.nn.Linear(2, 2), difference_query=torch.nn.Embedding(32, 2)),
            action_expert=torch.nn.Linear(2, 2))
        calls = []
        def compare(module, source, *, weights):
            calls.append((module, source, weights))
            return {"exact": True}
        evidence = entry.verify_model(model, "/source/14000", compare)
        self.assertEqual(set(evidence), {"vlm", "query", "action_expert"})
        self.assertEqual([call[2] for call in calls],
                         [None, "difference_query.safetensors", "action_expert.safetensors"])
        self.assertTrue(all(call[1] == "/source/14000" for call in calls))
        model.slot_aux = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "auxiliary Heads"):
            entry.verify_model(model, "/source/14000", compare)

    def test_frozen_component_and_failed_exact_check_are_rejected(self):
        model = SimpleNamespace(slot_aux=None, optical_flow_aux=None, training_stage=None,
            num_difference_queries=32, backbone=SimpleNamespace(
                model=torch.nn.Linear(2, 2), difference_query=torch.nn.Embedding(32, 2)),
            action_expert=torch.nn.Linear(2, 2))
        def mismatch(*args, **kwargs):
            raise RuntimeError("source tensor differs")
        with self.assertRaisesRegex(RuntimeError, "tensor differs"):
            entry.verify_model(model, "/source", mismatch)
        model.backbone.model.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "frozen"):
            entry.verify_model(model, "/source", mismatch)

    def test_optimizer_rejects_inherited_momentum_and_split_groups(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.AdamW([parameter], lr=2e-5, betas=(0.9, 0.95), eps=1e-6)
        entry.verify_optimizer(optimizer)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        with self.assertRaisesRegex(ValueError, "newly initialized"):
            entry.verify_optimizer(optimizer)
        other = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.AdamW([{"params": [parameter]}, {"params": [other]}])
        with self.assertRaises(ValueError):
            entry.verify_optimizer(optimizer)

    def test_global_and_legacy_scheduler_positions(self):
        for step in (0, 2000, 34184):
            scheduler = SimpleNamespace(state_dict=lambda: {
                "last_epoch": step * 4, "_step_count": step * 4 + 1}, get_last_lr=lambda: [1e-5])
            result = entry.verify_update_position(SimpleNamespace(global_steps=step), scheduler, step)
            self.assertEqual(result["global_step_before"], step)
            with self.assertRaises(ValueError):
                entry.verify_update_position(SimpleNamespace(global_steps=step + 1), scheduler, step)

    def test_sealed_checkpoint_is_immutable_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempt-000"
            source = attempt / entry.TAG
            source.mkdir(parents=True)
            (source / "model.safetensors").write_bytes(b"weights")
            archived = Path(entry.seal_checkpoint(attempt, 2000))
            self.assertEqual(supervisor.latest_complete(root), archived)
            (source / "model.safetensors").write_bytes(b"next checkpoint")
            self.assertEqual((archived / "model.safetensors").read_bytes(), b"weights")
            (archived / "model.safetensors").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                supervisor.latest_complete(root)

    def test_pretraining_checkpoint_is_never_a_retry_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "latest-model-optimizer-lr").mkdir()
            self.assertIsNone(supervisor.latest_complete(root))


if __name__ == "__main__":
    unittest.main()
