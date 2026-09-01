import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.wandb_training_logger import WandbTrainingLogger


class FakeAccelerator:
    def __init__(self, is_main_process: bool):
        self.is_main_process = is_main_process
        self.reduce_calls = []

    def reduce(self, value, reduction):
        self.reduce_calls.append((value.item(), reduction))
        return value + 1


class FakeRun:
    def __init__(self):
        self.logs = []
        self.finished = []

    def log(self, metrics, step):
        self.logs.append((metrics, step))

    def finish(self, exit_code=0):
        self.finished.append(exit_code)


class FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.init_calls = []
        self.run = FakeRun()

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run


class WandbTrainingLoggerTest(unittest.TestCase):
    def make_logger(self, accelerator, fake_wandb, project="ZR-0-LIBERO"):
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            return WandbTrainingLogger(
                accelerator,
                project=project,
                run_name="test-run",
                run_id="test1234",
                resume="never",
                log_dir="/tmp/wandb",
                group="libero-wo-ecot-pt",
                tags=["ablation", "wo-ecot-pt"],
                config={"global_batch_size": 64},
            )

    def test_main_process_initializes_one_online_run_and_logs_reduced_metrics(self):
        accelerator = FakeAccelerator(is_main_process=True)
        fake_wandb = FakeWandb()
        logger = self.make_logger(accelerator, fake_wandb)

        logger.log(
            step=10,
            mean_metrics={"train/loss": torch.tensor(2.0)},
            scalar_metrics={"train/learning_rate": 2e-5},
        )
        logger.finish(exit_code=0)

        self.assertEqual(len(fake_wandb.init_calls), 1)
        init = fake_wandb.init_calls[0]
        self.assertEqual(init["project"], "ZR-0-LIBERO")
        self.assertEqual(init["id"], "test1234")
        self.assertEqual(init["resume"], "never")
        self.assertEqual(init["mode"], "online")
        self.assertEqual(accelerator.reduce_calls, [(2.0, "mean")])
        self.assertEqual(fake_wandb.run.logs[0][1], 10)
        self.assertEqual(fake_wandb.run.logs[0][0]["train/loss"], 3.0)
        self.assertEqual(
            fake_wandb.run.logs[0][0]["train/learning_rate"], 2e-5
        )
        self.assertEqual(fake_wandb.run.finished, [0])

    def test_non_main_process_reduces_metrics_without_creating_a_run(self):
        accelerator = FakeAccelerator(is_main_process=False)
        fake_wandb = FakeWandb()
        logger = self.make_logger(accelerator, fake_wandb)

        logger.log(
            step=10,
            mean_metrics={"train/loss": torch.tensor(4.0)},
            scalar_metrics={},
        )
        logger.finish(exit_code=0)

        self.assertEqual(fake_wandb.init_calls, [])
        self.assertEqual(accelerator.reduce_calls, [(4.0, "mean")])
        self.assertEqual(fake_wandb.run.logs, [])

    def test_empty_project_disables_wandb_and_collectives(self):
        accelerator = FakeAccelerator(is_main_process=True)
        fake_wandb = FakeWandb()
        logger = self.make_logger(accelerator, fake_wandb, project=None)

        logger.log(
            step=10,
            mean_metrics={"train/loss": torch.tensor(2.0)},
            scalar_metrics={},
        )

        self.assertEqual(fake_wandb.init_calls, [])
        self.assertEqual(accelerator.reduce_calls, [])


if __name__ == "__main__":
    unittest.main()
