import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.qwen_vl_backbone import QwenVLBackbone


class FakeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(2)


class FakeQwenBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = FakeLanguageModel()


class FakeConditionalModel(nn.Module):
    def __init__(self, hidden_state: torch.Tensor):
        super().__init__()
        self.model = FakeQwenBody()
        self.hidden_state = hidden_state

    def forward(self, **_kwargs):
        return {"hidden_states": (self.hidden_state,)}


class FakePeftModel(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model_for_test = base_model

    def get_base_model(self):
        return self.base_model_for_test

    def forward(self, **kwargs):
        return self.base_model_for_test(**kwargs)


class QwenVLBackboneTest(unittest.TestCase):
    @staticmethod
    def make_backbone(model: nn.Module) -> QwenVLBackbone:
        backbone = QwenVLBackbone.__new__(QwenVLBackbone)
        nn.Module.__init__(backbone)
        backbone.tune_vlm = True
        backbone.model = model
        return backbone

    def test_forward_returns_final_rmsnorm_for_plain_and_peft_models(self):
        raw = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])

        for wrap_in_peft in (False, True):
            with self.subTest(wrap_in_peft=wrap_in_peft):
                base_model = FakeConditionalModel(raw)
                model = FakePeftModel(base_model) if wrap_in_peft else base_model
                backbone = self.make_backbone(model)
                inputs = BatchFeature(
                    {
                        "input_ids": torch.tensor([[1, 2]]),
                        "sub_task_flag": torch.tensor([0]),
                    }
                )

                outputs = backbone(inputs)

                expected = base_model.model.language_model.norm(raw)
                torch.testing.assert_close(outputs.backbone_embeddings, expected)
                self.assertFalse(torch.equal(outputs.backbone_embeddings, raw))


if __name__ == "__main__":
    unittest.main()
