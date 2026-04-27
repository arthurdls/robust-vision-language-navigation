#!/usr/bin/env python3
"""
Launch the Unreal Engine simulator for remote or local use.

Resolves the binary path from the scene JSON and UnrealEnv directory,
configures unrealcv.ini (port, resolution), launches the binary, waits
for the UnrealCV TCP port to accept connections, then prints connection
instructions for the control machine.

Usage (from repo root):
  python scripts/run_simulator.py
  python scripts/run_simulator.py --port 9000 --gpu-id 0
  python scripts/run_simulator.py --scene DowntownWest --no-offscreen
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.config import DEFAULT_SIM_PORT
from rvln.paths import DOWNTOWN_OVERLAY_JSON, UNREAL_ENV_ROOT, load_env_vars


def get_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def resolve_binary(scene: str, unreal_root: Path) -> tuple[Path, Path]:
    """Return (binary_path, unrealcv_ini_path) for the given scene."""
    scene_json = DOWNTOWN_OVERLAY_JSON
    if scene != "DowntownWest":
        from gym_unrealcv.envs.utils.misc import get_settingpath
        scene_json = Path(get_settingpath(f"Track/{scene}.json"))

    with open(scene_json) as f:
        setting = json.load(f)

    if "linux" in sys.platform:
        env_bin = setting["env_bin"]
    elif "darwin" in sys.platform:
        env_bin = setting.get("env_bin_mac", setting["env_bin"])
    elif "win" in sys.platform:
        env_bin = setting.get("env_bin_win", setting["env_bin"])
    else:
        print(f"Unsupported platform: {sys.platform}", file=sys.stderr)
        sys.exit(1)

    binary = unreal_root / env_bin
    if not binary.exists():
        print(f"Binary not found: {binary}", file=sys.stderr)
        print(f"Download it first: python tools/download_simulator.py", file=sys.stderr)
        sys.exit(1)

    ini_path = binary.parent / "unrealcv.ini"
    return binary, ini_path


def write_unrealcv_ini(ini_path: Path, port: int, resolution: tuple[int, int]) -> None:
    """Write or update unrealcv.ini with port and resolution."""
    lines = [
        "[UnrealCV.Core]",
        f"Port={port}",
        f"Width={resolution[0]}",
        f"Height={resolution[1]}",
    ]
    ini_path.write_text("\n".join(lines) + "\n")


def wait_for_port(host: str, port: int, timeout: float = 60.0, interval: float = 2.0) -> bool:
    """Block until the TCP port accepts connections or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(interval)
    return False


def main():
    load_env_vars()

    parser = argparse.ArgumentParser(description="Launch the Unreal simulator")
    parser.add_argument("--port", type=int, default=DEFAULT_SIM_PORT,
                        help=f"UnrealCV listen port (default: {DEFAULT_SIM_PORT})")
    parser.add_argument("--scene", type=str, default="DowntownWest",
                        help="Scene name from scene JSON (default: DowntownWest)")
    parser.add_argument("--offscreen", action=argparse.BooleanOptionalAction, default=True,
                        help="Headless rendering (default: on)")
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="GPU index for multi-GPU machines")
    parser.add_argument("--resolution", type=str, default="256x256",
                        help="Rendering resolution WxH (default: 256x256)")
    parser.add_argument("--unreal-root", type=Path, default=UNREAL_ENV_ROOT,
                        help=f"Unreal binaries root (default: {UNREAL_ENV_ROOT})")
    args = parser.parse_args()

    width, height = (int(x) for x in args.resolution.split("x"))

    binary, ini_path = resolve_binary(args.scene, args.unreal_root)

    binary.chmod(binary.stat().st_mode | 0o755)

    write_unrealcv_ini(ini_path, args.port, (width, height))
    print(f"Wrote {ini_path}: port={args.port}, resolution={width}x{height}")

    cmd = [str(binary.resolve())]
    if args.offscreen:
        cmd.append("-RenderOffscreen")
    if args.gpu_id is not None:
        cmd.extend([f"-graphicsadapter={args.gpu_id}"])

    print(f"Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)

    def shutdown(signum, frame):
        print("\nShutting down simulator...")
        proc.terminate()
        proc.wait(timeout=10)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"Waiting for UnrealCV on port {args.port}...")
    if wait_for_port("127.0.0.1", args.port):
        local_ip = get_local_ip()
        print(f"\nSimulator ready on port {args.port}.")
        print(f"\nOn the control machine, run:")
        print(f"  python scripts/start_server.py")
        print(f"  python scripts/run_integration.py --task <TASK.json> --sim_host {local_ip} --sim_port {args.port}")
        print(f"\nPress Ctrl+C to stop the simulator.")
    else:
        print(f"\nTimeout: UnrealCV did not start on port {args.port} within 60s.", file=sys.stderr)
        proc.terminate()
        sys.exit(1)

    proc.wait()


if __name__ == "__main__":
    main()
