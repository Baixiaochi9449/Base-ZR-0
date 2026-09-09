import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.wandb_training_logger import WandbTrainingLogger
from utils.cli_options import parse_train_options


class FakeAccelerator:
    device = torch.device("cpu")
    def __init__(self, is_main_process: bool):
        self.is_main_process = is_main_process
        self.reduce_calls = []

    def reduce(self, value, reduction):
        self.reduce_calls.append((value.item(), reduction))
        return value + 1


class FakeDistributedInitFailureAccelerator(FakeAccelerator):
    num_processes = 2
    device = torch.device("cpu")

    def reduce(self, value, reduction):
        self.reduce_calls.append((value.item(), reduction))
        if reduction == "min":
            return torch.zeros_like(value)
        return value


class FakeRun:
    def __init__(self):
        self.logs = []
        self.finished = []
        self.fail_log_calls = 0

    def log(self, metrics, step):
        if self.fail_log_calls:
            self.fail_log_calls -= 1
            raise ConnectionError("temporary W&B outage")
        self.logs.append((metrics, step))

    def finish(self, exit_code=0):
        self.finished.append(exit_code)


class FailingFinishRun(FakeRun):
    def finish(self, exit_code=0):
        raise ConnectionError("finish failed")


class BlockingLogRun(FakeRun):
    def log(self, metrics, step):
        time.sleep(60)


class FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.init_calls = []
        self.run = FakeRun()

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run


class FailingInitWandb(FakeWandb):
    def init(self, **kwargs):
        raise ConnectionError("offline")


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
        self.assertEqual(fake_wandb.run.finished, [])

    def test_component_python_scalars_share_tensor_reduction_on_every_rank(self):
        for is_main_process in (True, False):
            with self.subTest(is_main_process=is_main_process):
                accelerator = FakeAccelerator(is_main_process=is_main_process)
                fake_wandb = FakeWandb()
                logger = self.make_logger(accelerator, fake_wandb)
                loss = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
                metrics = {
                    "train/loss": loss,
                    "train/vlm_native_grad_norm": 0.25,
                    "train/scheduler_step_after": 1,
                    "train/optimizer_update_applied": True,
                    "train/optimizer_skip_reason": "none",
                }
                with patch.object(accelerator, "reduce", wraps=accelerator.reduce) as reduce:
                    logger.log(step=1, mean_metrics=metrics, scalar_metrics={})
                self.assertEqual(accelerator.reduce_calls,
                    [(2.0, "mean"), (0.25, "mean"), (1.0, "mean")])
                for call in reduce.call_args_list:
                    tensor = call.args[0]
                    self.assertEqual(tensor.dtype, torch.float32)
                    self.assertEqual(tensor.device, accelerator.device)
                    self.assertFalse(tensor.requires_grad)
                self.assertTrue(loss.requires_grad)
                self.assertEqual(loss.dtype, torch.float64)
                if is_main_process:
                    payload, step = fake_wandb.run.logs[0]
                    self.assertEqual(step, 1)
                    self.assertEqual(payload["train/vlm_native_grad_norm"], 1.25)
                    self.assertEqual(payload["train/scheduler_step_after"], 2.0)
                    self.assertIs(payload["train/optimizer_update_applied"], True)
                    self.assertEqual(payload["train/optimizer_skip_reason"], "none")
                else:
                    self.assertEqual(fake_wandb.run.logs, [])
                    self.assertEqual(fake_wandb.init_calls, [])

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

    def test_best_effort_initialization_failure_keeps_local_training_alive(self):
        accelerator = FakeAccelerator(is_main_process=True)
        with self.assertLogs("utils.wandb_training_logger", level="WARNING"):
            logger = self.make_logger(accelerator, FailingInitWandb())
        self.assertFalse(logger.enabled)
        self.assertIsNone(logger.run)

    def test_best_effort_initialization_failure_disables_every_rank(self):
        accelerator = FakeDistributedInitFailureAccelerator(is_main_process=False)
        logger = self.make_logger(accelerator, FakeWandb())
        self.assertFalse(logger.enabled)
        logger.log(
            step=10,
            mean_metrics={"train/loss": torch.tensor(2.0)},
            scalar_metrics={},
        )
        self.assertEqual(accelerator.reduce_calls, [(1, "min")])

    def test_post_init_logging_failure_is_buffered_and_retried(self):
        accelerator = FakeAccelerator(is_main_process=True)
        fake_wandb = FakeWandb()
        logger = self.make_logger(accelerator, fake_wandb)
        fake_wandb.run.fail_log_calls = 1

        with self.assertLogs("utils.wandb_training_logger", level="WARNING"):
            logger.log(
                step=10,
                mean_metrics={"train/loss": torch.tensor(2.0)},
                scalar_metrics={},
            )
        self.assertEqual(len(logger._pending), 1)

        logger.log(
            step=20,
            mean_metrics={"train/loss": torch.tensor(4.0)},
            scalar_metrics={},
        )
        self.assertEqual(len(logger._pending), 0)
        self.assertEqual([step for _, step in fake_wandb.run.logs], [10, 20])

    def test_permanent_failure_is_bounded_and_backed_off(self):
        accelerator = FakeAccelerator(is_main_process=True)
        fake_wandb = FakeWandb()
        logger = self.make_logger(accelerator, fake_wandb)
        logger.pending_capacity = 32
        fake_wandb.run.fail_log_calls = 100_000
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "training_metrics.jsonl"
            with local_path.open("w", encoding="utf-8") as local_log:
                for step in range(10_000):
                    diagnostics = logger.log(
                        step=step,
                        mean_metrics={"train/loss": torch.tensor(float(step))},
                        scalar_metrics={},
                    )
                    local_log.write(
                        json.dumps({"step": step, **diagnostics}, sort_keys=True) + "\n"
                    )
            local_records = local_path.read_text(encoding="utf-8").splitlines()
        self.assertLessEqual(len(logger._pending), 32)
        self.assertEqual(len(local_records), 10_000)
        self.assertEqual(json.loads(local_records[-1])["step"], 9_999)
        self.assertGreater(logger.dropped_payload_count, 0)
        remote_calls = 100_000 - fake_wandb.run.fail_log_calls
        self.assertLess(remote_calls, 100)

    def test_overflow_drops_oldest_and_recovery_sends_retained_payloads(self):
        accelerator = FakeAccelerator(is_main_process=True)
        fake_wandb = FakeWandb()
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            logger = WandbTrainingLogger(
                accelerator,
                project="test",
                run_name="bounded",
                run_id=None,
                resume="never",
                log_dir="/tmp/wandb",
                pending_capacity=2,
                retry_base_steps=4,
                retry_max_steps=4,
            )
        fake_wandb.run.fail_log_calls = 1
        logger.log(step=0, mean_metrics={}, scalar_metrics={"value": 0})
        logger.log(step=1, mean_metrics={}, scalar_metrics={"value": 1})
        logger.log(step=2, mean_metrics={}, scalar_metrics={"value": 2})
        assert logger.dropped_payload_count == 1
        logger.log(step=4, mean_metrics={}, scalar_metrics={"value": 4})
        assert [step for _, step in fake_wandb.run.logs] == [2, 4]
        assert logger.remote_recovery_count == 1

    def test_finish_has_bounded_remote_attempts(self):
        accelerator = FakeAccelerator(is_main_process=True)
        fake_wandb = FakeWandb()
        logger = self.make_logger(accelerator, fake_wandb)
        fake_wandb.run.fail_log_calls = 100
        logger.log(step=0, mean_metrics={}, scalar_metrics={})
        before = fake_wandb.run.fail_log_calls
        logger.finish()
        assert before - fake_wandb.run.fail_log_calls <= logger.finish_max_attempts
        assert fake_wandb.run.finished == [0]

    def test_best_effort_finish_exception_is_not_propagated(self):
        accelerator = FakeAccelerator(is_main_process=True)
        fake_wandb = FakeWandb()
        fake_wandb.run = FailingFinishRun()
        logger = self.make_logger(accelerator, fake_wandb)
        with self.assertLogs("utils.wandb_training_logger", level="WARNING"):
            diagnostics = logger.finish()
        self.assertEqual(diagnostics["wandb_remote_abandoned"], 1)
        self.assertEqual(diagnostics["wandb_finish_timed_out"], 0)

    def test_pending_flush_uses_the_same_finish_deadline(self):
        accelerator = FakeAccelerator(is_main_process=True)
        fake_wandb = FakeWandb()
        fake_wandb.run = BlockingLogRun()
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            logger = WandbTrainingLogger(
                accelerator,
                project="test",
                run_name="blocking-log",
                run_id=None,
                resume="never",
                log_dir="/tmp/wandb",
                finish_timeout_seconds=0.1,
            )
        logger._pending.append(({"value": 1}, 1))
        started = time.monotonic()
        diagnostics = logger.finish()
        self.assertLess(time.monotonic() - started, 0.75)
        self.assertEqual(diagnostics["wandb_finish_timed_out"], 1)
        self.assertEqual(diagnostics["wandb_pending_payloads"], 1)
        self.assertEqual(fake_wandb.run.finished, [])

    def test_blocking_best_effort_finish_returns_and_subprocess_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "training_metrics.jsonl"
            code = textwrap.dedent(
                f"""
                import json
                import sys
                import time
                import types
                import torch
                from pathlib import Path
                from utils.wandb_training_logger import WandbTrainingLogger

                class Accelerator:
                    is_main_process = True
                    num_processes = 1
                    device = torch.device('cpu')
                    def reduce(self, value, reduction):
                        return value

                class Run:
                    def log(self, metrics, step):
                        pass
                    def finish(self, exit_code=0):
                        time.sleep(60)

                module = types.ModuleType('wandb')
                module.init = lambda **kwargs: Run()
                sys.modules['wandb'] = module
                logger = WandbTrainingLogger(
                    Accelerator(), project='test', run_name='blocking', run_id=None,
                    resume='never', log_dir={str(Path(directory))!r},
                    failure_policy='best_effort', finish_timeout_seconds=0.15,
                )
                started = time.monotonic()
                diagnostics = logger.finish()
                elapsed = time.monotonic() - started
                record = {{'event': 'wandb_finish', 'elapsed': elapsed, **diagnostics}}
                Path({str(output)!r}).write_text(json.dumps(record) + '\\n')
                print(json.dumps(record))
                """
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env={**os.environ, "PYTHONNOUSERSITE": "1"},
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["wandb_finish_timed_out"], 1)
            self.assertEqual(record["wandb_remote_abandoned"], 1)
            self.assertLess(record["elapsed"], 0.75)

    def test_required_finish_exception_remains_synchronous_failure(self):
        accelerator = FakeAccelerator(is_main_process=True)
        fake_wandb = FakeWandb()
        fake_wandb.run = FailingFinishRun()
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            logger = WandbTrainingLogger(
                accelerator,
                project="required",
                run_name="required",
                run_id=None,
                resume="never",
                log_dir="/tmp/wandb",
                failure_policy="required",
                finish_timeout_seconds=0.1,
            )
        with self.assertRaisesRegex(RuntimeError, "required W&B finish failed"):
            logger.finish()

    def test_finish_timeout_must_be_finite_and_positive(self):
        accelerator = FakeAccelerator(is_main_process=True)
        fake_wandb = FakeWandb()
        for value in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(value=value), patch.dict(
                sys.modules, {"wandb": fake_wandb}
            ), self.assertRaisesRegex(ValueError, "finite positive"):
                WandbTrainingLogger(
                    accelerator,
                    project="test",
                    run_name="test",
                    run_id=None,
                    resume="never",
                    log_dir="/tmp/wandb",
                    finish_timeout_seconds=value,
                )

    def test_finish_timeout_cli_validation_and_default(self):
        self.assertEqual(parse_train_options([]).wandb_finish_timeout_seconds, 15.0)
        for value in ("0", "-1", "inf", "nan"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parse_train_options(["--wandb_finish_timeout_seconds", value])

    def test_required_runtime_failure_keeps_explicit_failure_semantics(self):
        accelerator = FakeAccelerator(is_main_process=True)
        fake_wandb = FakeWandb()
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            logger = WandbTrainingLogger(
                accelerator,
                project="required",
                run_name="required",
                run_id=None,
                resume="never",
                log_dir="/tmp/wandb",
                failure_policy="required",
            )
        fake_wandb.run.fail_log_calls = 1
        with self.assertRaisesRegex(RuntimeError, "required W&B logging failed"):
            logger.log(step=0, mean_metrics={}, scalar_metrics={})


if __name__ == "__main__":
    unittest.main()
