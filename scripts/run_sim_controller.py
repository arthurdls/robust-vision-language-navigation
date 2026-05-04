#!/usr/bin/env python3
"""
Simulator lifecycle controller: a persistent daemon on the simulator machine.

Accepts HTTP commands from a remote orchestrator to start and stop the
simulator for different maps. The orchestrator sends requests like:

    POST /start  {"scene": "Greek_Island", "port": 9000, "api_port": 9001, ...}
    POST /stop
    GET  /status

This script manages the run_simulator.py process locally. It should be left
running on the simulator machine for the duration of a multi-map experiment.

Usage:
  python scripts/run_sim_controller.py
  python scripts/run_sim_controller.py --port 9002
"""

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, request as flask_request

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.config import (
    DEFAULT_SIM_API_PORT,
    DEFAULT_SIM_CONTROLLER_PORT,
    DEFAULT_SIM_PORT,
    DEFAULT_SEED,
    DEFAULT_TIME_DILATION,
)
from rvln.paths import load_env_vars

logger = logging.getLogger(__name__)


def _get_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


app = Flask(__name__)

_SIMULATOR_SCRIPT = Path(__file__).resolve().parent / "run_simulator.py"

_sim_proc: subprocess.Popen | None = None
_current_scene: str | None = None
_sim_api_port: int | None = None


def _is_running() -> bool:
    return _sim_proc is not None and _sim_proc.poll() is None


def _stop_sim(timeout: float = 15.0) -> None:
    global _sim_proc, _current_scene, _sim_api_port

    if _sim_proc is None:
        return
    if _sim_proc.poll() is not None:
        _sim_proc = None
        _current_scene = None
        _sim_api_port = None
        return

    logger.info("Stopping simulator (PID %d, scene=%s)...", _sim_proc.pid, _current_scene)
    os.kill(_sim_proc.pid, signal.SIGTERM)
    try:
        _sim_proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Simulator did not exit in %ds, sending SIGKILL", timeout)
        os.kill(_sim_proc.pid, signal.SIGKILL)
        _sim_proc.wait(timeout=5)

    _sim_proc = None
    _current_scene = None
    _sim_api_port = None
    logger.info("Simulator stopped.")


def _wait_for_health(host: str, port: int, timeout: float = 120.0, interval: float = 3.0) -> bool:
    """Poll the sim API /health endpoint until the env is initialized."""
    import requests as req

    url = f"http://{host}:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = req.get(url, timeout=5)
            if resp.ok and resp.json().get("initialized"):
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


@app.route("/status", methods=["GET"])
def handle_status():
    return jsonify({
        "running": _is_running(),
        "scene": _current_scene,
        "api_port": _sim_api_port,
    })


@app.route("/start", methods=["POST"])
def handle_start():
    global _sim_proc, _current_scene, _sim_api_port

    data = flask_request.get_json(force=True)
    scene = data.get("scene")
    if not scene:
        return jsonify({"error": "missing 'scene' parameter"}), 400

    sim_port = int(data.get("port", DEFAULT_SIM_PORT))
    api_port = int(data.get("api_port", DEFAULT_SIM_API_PORT))
    time_dilation = int(data.get("time_dilation", DEFAULT_TIME_DILATION))
    seed = int(data.get("seed", DEFAULT_SEED))
    startup_timeout = float(data.get("startup_timeout", 120.0))

    if _is_running():
        if _current_scene == scene:
            return jsonify({
                "status": "already_running",
                "scene": _current_scene,
                "api_port": _sim_api_port,
            })
        _stop_sim()
        time.sleep(3)

    cmd = [
        sys.executable, str(_SIMULATOR_SCRIPT),
        "--scene", scene,
        "--port", str(sim_port),
        "--api-port", str(api_port),
        "--time_dilation", str(time_dilation),
        "--seed", str(seed),
    ]
    logger.info("Starting simulator: %s", " ".join(cmd))
    _sim_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    if not _wait_for_health("127.0.0.1", api_port, timeout=startup_timeout):
        _sim_proc.terminate()
        _sim_proc.wait(timeout=10)
        _sim_proc = None
        return jsonify({
            "error": f"Simulator for '{scene}' did not become ready within {startup_timeout}s",
        }), 504

    _current_scene = scene
    _sim_api_port = api_port
    logger.info("Simulator ready: scene=%s, api_port=%d", scene, api_port)

    return jsonify({
        "status": "started",
        "scene": _current_scene,
        "api_port": _sim_api_port,
        "pid": _sim_proc.pid,
    })


@app.route("/stop", methods=["POST"])
def handle_stop():
    if not _is_running():
        return jsonify({"status": "not_running"})
    _stop_sim()
    return jsonify({"status": "stopped"})


def main():
    load_env_vars()

    parser = argparse.ArgumentParser(
        description="Simulator lifecycle controller (run on the simulator machine)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_SIM_CONTROLLER_PORT,
                        help=f"Port for the controller HTTP API (default: {DEFAULT_SIM_CONTROLLER_PORT})")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    def shutdown(signum, frame):
        logger.info("Controller shutting down...")
        _stop_sim()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    local_ip = _get_local_ip()

    print(f"\nSim controller listening on port {args.port}.")
    print(f"Waiting for orchestrator commands (POST /start, POST /stop, GET /status)")
    print(f"\nOn the orchestrator machine, run:")
    print(f"  python scripts/run_all_conditions.py \\")
    print(f"    --sim-controller {local_ip}:{args.port} \\")
    print(f"    --sim_host {local_ip} \\")
    print(f"    --sim_api_port {DEFAULT_SIM_API_PORT}")
    print(f"\nTo run specific conditions (e.g., full system and open-loop):")
    print(f"  python scripts/run_all_conditions.py \\")
    print(f"    --sim-controller {local_ip}:{args.port} \\")
    print(f"    --sim_host {local_ip} \\")
    print(f"    --sim_api_port {DEFAULT_SIM_API_PORT} \\")
    print(f"    --conditions 0,3")
    print(f"\nTo run a single map:")
    print(f"  python scripts/run_all_conditions.py \\")
    print(f"    --sim-controller {local_ip}:{args.port} \\")
    print(f"    --sim_host {local_ip} \\")
    print(f"    --sim_api_port {DEFAULT_SIM_API_PORT} \\")
    print(f"    --map greek_island")
    print(f"\nPress Ctrl+C to stop the controller.\n")

    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
