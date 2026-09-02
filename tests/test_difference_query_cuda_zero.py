import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.difference_query import DifferenceQuery
from model.qwen_vl_backbone import QwenVLBackbone


RUN_CUDA = os.environ.get("ZR0_RUN_CUDA_TESTS") == "1"
RUN_ZERO2 = os.environ.get("ZR0_RUN_ZERO2_TESTS") == "1"


def make_tiny_backbone() -> QwenVLBackbone:
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "attention_dropout": 0.0,
            "rope_scaling": {
                "mrope_interleaved": True,
                "mrope_section": [2, 1, 1],
                "rope_type": "default",
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "deepstack_visual_indexes": [],
        },
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )
    config._attn_implementation = "sdpa"
    backbone = QwenVLBackbone.__new__(QwenVLBackbone)
    nn.Module.__init__(backbone)
    backbone.tune_vlm = False
    backbone.model = Qwen3VLForConditionalGeneration(config)
    backbone.model.requires_grad_(False)
    backbone.use_difference_query = True
    backbone.num_difference_queries = 8
    backbone.query_placeholder_token_id = 0
    backbone.difference_query = DifferenceQuery(8, 32, initializer_std=0.02)
    return backbone


def make_cuda_backbone() -> QwenVLBackbone:
    return make_tiny_backbone().cuda().to(dtype=torch.bfloat16).eval()


class DifferenceQueryConditionalHarnessTest(unittest.TestCase):
    def test_cuda_and_zero_harness_models_construct_on_cpu(self):
        backbone = make_tiny_backbone()
        zero_model = TinyZeroQueryModel()

        self.assertEqual(backbone.difference_query.weight.shape, (8, 32))
        self.assertEqual(zero_model.difference_query.weight.shape, (8, 16))

    def test_production_checkpoint_helper_registers_safe_globals_without_dist(self):
        from utils.training_checkpoint import register_deepspeed_checkpoint_safe_globals

        distributed_was_initialized = torch.distributed.is_initialized()
        registered = register_deepspeed_checkpoint_safe_globals()

        self.assertEqual(torch.distributed.is_initialized(), distributed_was_initialized)
        self.assertGreaterEqual(len(registered), 3)


@unittest.skipUnless(
    RUN_CUDA and torch.cuda.is_available(),
    "set ZR0_RUN_CUDA_TESTS=1 on an idle CUDA host",
)
class DifferenceQueryCudaTest(unittest.TestCase):
    def test_bf16_forward_uses_sdpa_and_records_peak_memory(self):
        from torch.profiler import ProfilerActivity, profile

        torch.cuda.reset_peak_memory_stats()
        backbone = make_cuda_backbone()
        inputs = BatchFeature(
            {
                "input_ids": torch.tensor(
                    [[5, 6, 7, 8, 127]], device="cuda"
                ),
                "attention_mask": torch.tensor(
                    [[1, 1, 1, 1, 0]], device="cuda"
                ),
            }
        )
        with torch.no_grad(), profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as trace:
            outputs = backbone(inputs, compute_vlm_loss=False)
        torch.cuda.synchronize()

        kernel_names = sorted(
            event.key
            for event in trace.key_averages()
            if "scaled_dot_product" in event.key
        )
        peak_bytes = torch.cuda.max_memory_allocated()
        self.assertTrue(kernel_names, "no scaled-dot-product attention kernel ran")
        self.assertEqual(outputs.backbone_embeddings.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(outputs.backbone_embeddings).all())
        print(
            "CUDA BF16 Difference Query peak bytes:",
            peak_bytes,
            "SDPA kernels:",
            kernel_names,
        )


class TinyZeroQueryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.difference_query = DifferenceQuery(8, 16, initializer_std=0.02)
        self.projection = nn.Linear(16, 1)

    def forward(self, inputs):
        query = self.difference_query.for_batch(inputs.shape[0], inputs)
        return self.projection(query + inputs.unsqueeze(1)).square().mean()


class DifferenceQueryZero2Test(unittest.TestCase):
    def run_torchrun_worker(self, world_size: int) -> None:
        env = os.environ.copy()
        for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(name, None)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = str(ROOT)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world_size}",
            str(ROOT / "tests" / "zero2_checkpoint_worker.py"),
        ]
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=360,
            )
        except subprocess.TimeoutExpired as error:
            self.fail(
                f"ZeRO-2 {world_size}-rank smoke timed out\n"
                f"stdout:\n{error.stdout or ''}\nstderr:\n{error.stderr or ''}"
            )
        if result.returncode != 0:
            self.fail(
                f"ZeRO-2 {world_size}-rank smoke exited {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        self.assertIn(f"ZERO2_SMOKE_OK world_size={world_size}", result.stdout)

    @unittest.skipUnless(
        RUN_ZERO2 and torch.cuda.is_available(),
        "set ZR0_RUN_ZERO2_TESTS=1 on an idle CUDA host",
    )
    def test_world_size_one_production_save_resume(self):
        self.run_torchrun_worker(world_size=1)

    @unittest.skipUnless(
        RUN_ZERO2 and torch.cuda.device_count() >= 2,
        "set ZR0_RUN_ZERO2_TESTS=1 on an idle host with at least two CUDA devices",
    )
    def test_two_rank_production_save_resume(self):
        self.run_torchrun_worker(world_size=2)


if __name__ == "__main__":
    unittest.main()
