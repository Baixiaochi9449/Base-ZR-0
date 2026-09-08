import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from utils.cli_options import parse_server_options

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_robotwin_legacy_server.sh"


def run_launcher(*args, **overrides):
    env = {key: value for key, value in os.environ.items()
           if key not in {"CUDA_VISIBLE_DEVICES", "ZR0_SERVER_PYTHON",
                          "ZR0_ROBOTWIN_CKPT", "ZR0_SERVER_PORT"}}
    env.update(overrides)
    return subprocess.run(["bash", str(SCRIPT), *args], cwd="/tmp", env=env,
                          text=True, capture_output=True, timeout=20)


def test_default_prints_explicit_legacy_command_without_running_python():
    result = run_launcher(ZR0_SERVER_PYTHON="/nonexistent/python")
    assert result.returncode == 0, result.stderr
    words = shlex.split(result.stdout)
    options = parse_server_options(words[words.index(str(ROOT / "server.py")) + 1:])
    assert options.allow_legacy_checkpoint_without_manifest
    assert not options.allow_legacy_checkpoint_without_observation_contract
    assert options.dataset_entry == "demo_data.robotwin2.0-aloha-agilex"
    assert options.inference_mode == "direct_action"
    assert options.window_size == 1 and options.num_denoised_steps == 5
    assert options.max_pad_state_and_action_length == 64 and options.port == 8022
    assert options.use_difference_query is None and options.vlm_attention_backend is None
    assert "not verified" in result.stderr


def test_serve_routes_same_arguments_and_isolates_user_site(tmp_path):
    capture = tmp_path / "fake python"
    capture.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), "
        "'user_site': os.environ['PYTHONNOUSERSITE'], "
        "'gpu': os.environ['CUDA_VISIBLE_DEVICES']}))\n"
    )
    capture.chmod(0o755)
    result = run_launcher("serve", ZR0_SERVER_PYTHON=str(capture),
                          ZR0_ROBOTWIN_CKPT="/tmp/checkpoint with spaces",
                          ZR0_SERVER_PORT="9102", CUDA_VISIBLE_DEVICES="2")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["cwd"] == str(ROOT) and payload["user_site"] == "1"
    assert payload["gpu"] == "2"
    assert payload["argv"][:2] == ["-u", str(ROOT / "server.py")]
    options = parse_server_options(payload["argv"][2:])
    assert options.ckpt_dir == "/tmp/checkpoint with spaces" and options.port == 9102
    assert options.allow_legacy_checkpoint_without_manifest


def test_serve_requires_gpu_selection_before_executing_python():
    result = run_launcher("serve", ZR0_SERVER_PYTHON="/nonexistent/python")
    assert result.returncode != 0
    assert "CUDA_VISIBLE_DEVICES" in result.stderr


@pytest.mark.parametrize("port", ["0", "65536", "8022x", "-1", "08022"])
def test_invalid_ports_rejected(port):
    result = run_launcher(ZR0_SERVER_PORT=port)
    assert result.returncode == 2 and "ZR0_SERVER_PORT" in result.stderr


def test_unknown_mode_rejected():
    result = run_launcher("start")
    assert result.returncode == 2 and "Usage:" in result.stderr
