import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("exit_code", [0, 7])
def test_worker_waits_for_own_server_routes_environment_and_cleans_up(tmp_path, exit_code):
    worker = tmp_path / "worker with spaces"
    worker.mkdir()
    (worker / "experiment.md").write_text("Fixture\n")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    fake = tmp_path / "fake python"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import http.server, os, sys\n"
        "if sys.argv[1] == '-c': os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "server = http.server.HTTPServer(('127.0.0.1', int(os.environ['ZR0_SERVER_PORT'])), http.server.BaseHTTPRequestHandler)\n"
        "def get(self):\n"
        "    self.send_response(200); self.end_headers(); self.wfile.write(b'OK')\n"
        "http.server.BaseHTTPRequestHandler.do_GET = get\n"
        "server.serve_forever()\n"
    )
    fake.chmod(0o755)
    hook = tmp_path / "conda.sh"
    hook.write_text("conda() { export CONDA_DEFAULT_ENV=RoboTwin; }\n")
    (worker / "run_all.sh").write_text(
        'test "$CONDA_DEFAULT_ENV" = RoboTwin\n'
        'test "$CUDA_VISIBLE_DEVICES" = 2\n'
        'test "$ZR0_OBSERVATION_DEBUG_ROOT" = "' + str(worker / "observations") + '"\n'
        f'exit {exit_code}\n'
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_robotwin_eval_worker.sh"), str(worker), "2", str(port), "/fake/checkpoint"],
        env={**os.environ, "ZR0_SERVER_PYTHON": str(fake), "ROBOTWIN_CONDA_SH": str(hook)},
        text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == exit_code, result.stderr
    assert (worker / "exit_code.txt").read_text().strip() == str(exit_code)
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", port)) != 0


def test_observation_debug_root_preserves_buffer_values(tmp_path, monkeypatch):
    import torch
    from utils.obs_buffer import ObservationBuffer

    monkeypatch.setenv("ZR0_OBSERVATION_DEBUG_ROOT", str(tmp_path / "worker"))
    buffer = ObservationBuffer(1)
    pixels = torch.zeros(3, 4, 4)
    buffer.add_observation({"head": pixels})
    result = buffer.get_inference_time_observations(["head"], visualize=True)
    assert torch.equal(result["head"], pixels.unsqueeze(0))
    assert len(list((tmp_path / "worker").glob("*/head/0.jpg"))) == 1
    monkeypatch.delenv("ZR0_OBSERVATION_DEBUG_ROOT")
    monkeypatch.chdir(tmp_path)
    buffer.get_inference_time_observations(["head"], visualize=True)
    assert len(list((tmp_path / "temp").glob("*/head/0.jpg"))) == 1
