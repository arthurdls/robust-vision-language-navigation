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
    """Block until the TCP port is in use (server is listening) or timeout.

    Uses bind() to detect occupancy instead of connect(), so we never open
    a TCP connection to the UnrealCV server (which would poison its
    single-client slot).
    """
    start = time.time()
    while time.time() - start < timeout:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("0.0.0.0", port))
            s.close()
            time.sleep(interval)
        except OSError:
            s.close()
            return True
    return False


def kill_existing_simulator(port: int, binary: Path, timeout: float = 10.0) -> None:
    """If an UnrealCV simulator is already listening on port, kill it and wait for the port to free."""
    try:
        out = subprocess.check_output(
            ["ss", "-tlnp", f"sport = :{port}"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return

    pids: list[int] = []
    for line in out.splitlines():
        if f":{port}" not in line:
            continue
        # Extract pid from ss output: users:(("name",pid=NNN,fd=N))
        for part in line.split(","):
            if part.startswith("pid="):
                try:
                    pids.append(int(part.split("=")[1]))
                except ValueError:
                    pass

    if not pids:
        return

    binary_name = binary.name
    for pid in pids:
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
            exe = cmdline[0].decode(errors="replace")
        except OSError:
            continue
        if binary_name not in exe:
            print(
                f"Port {port} is held by PID {pid} ({exe}), which is not the simulator binary. Aborting.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Killing existing simulator (PID {pid}) on port {port}...")
        os.kill(pid, signal.SIGTERM)

    start = time.time()
    while time.time() - start < timeout:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("0.0.0.0", port))
            s.close()
            print("Previous simulator stopped.")
            return
        except OSError:
            s.close()
            time.sleep(0.5)

    print(f"Previous simulator did not release port {port} in {timeout}s, sending SIGKILL...",
          file=sys.stderr)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    time.sleep(2)


def count_connections(port: int, state: str) -> int:
    """Count TCP connections on port in a given state (established, close-wait, etc.)."""
    try:
        out = subprocess.check_output(
            ["ss", "-tn", "state", state, f"sport = :{port}"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0
    return sum(1 for line in out.splitlines() if f":{port}" in line)


def launch_simulator(cmd: list[str]) -> subprocess.Popen:
    """Launch the simulator binary and return the Popen handle."""
    return subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def stop_process(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    """Terminate a process, escalating to SIGKILL if needed."""
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


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

    kill_existing_simulator(args.port, binary)

    binary.chmod(binary.stat().st_mode | 0o755)

    write_unrealcv_ini(ini_path, args.port, (width, height))
    print(f"Wrote {ini_path}: port={args.port}, resolution={width}x{height}")

    cmd = [str(binary.resolve())]
    if args.offscreen:
        cmd.append("-RenderOffscreen")
    if args.gpu_id is not None:
        cmd.extend([f"-graphicsadapter={args.gpu_id}"])

    print(f"Launching: {' '.join(cmd)}")
    proc = launch_simulator(cmd)

    _current_proc = proc

    def shutdown(signum, frame):
        print("\nShutting down simulator...")
        stop_process(_current_proc)
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

    # Monitor loop: watch for client disconnections and auto-restart.
    #
    # UnrealCV allows only one client. After any disconnection (clean or not),
    # the server often cannot accept new clients. The only reliable recovery
    # is a full restart of the simulator binary.
    #
    # Strategy: track ESTABLISHED connections. When one disappears, wait a
    # short grace period (the client might just be reconnecting). If no new
    # client appears, restart. Also restart immediately on CLOSE_WAIT (the
    # server received FIN but hasn't cleaned up its end).
    had_client = False
    client_gone_since: float | None = None
    grace_seconds = 10.0

    while True:
        ret = proc.poll()
        if ret is not None:
            print(f"\nSimulator exited (code {ret}).")
            break

        n_established = count_connections(args.port, "established")
        n_close_wait = count_connections(args.port, "close-wait")
        needs_restart = False

        if n_close_wait > 0 and n_established == 0:
            print(f"\nStale CLOSE_WAIT detected on port {args.port}, restarting simulator...")
            needs_restart = True
        elif n_established > 0:
            if not had_client:
                print(f"Client connected on port {args.port}.")
            had_client = True
            client_gone_since = None
        elif had_client and n_established == 0:
            if client_gone_since is None:
                client_gone_since = time.time()
                print(f"Client disconnected from port {args.port}, waiting {grace_seconds:.0f}s for reconnect...")
            elif time.time() - client_gone_since >= grace_seconds:
                print(f"No reconnect after {grace_seconds:.0f}s, restarting simulator...")
                needs_restart = True

        if needs_restart:
            stop_process(proc)
            time.sleep(2)
            proc = launch_simulator(cmd)
            _current_proc = proc
            had_client = False
            client_gone_since = None
            print(f"Waiting for UnrealCV on port {args.port}...")
            if wait_for_port("127.0.0.1", args.port):
                print(f"Simulator restarted and ready on port {args.port}.")
            else:
                print("Timeout: simulator did not restart.", file=sys.stderr)
                proc.terminate()
                sys.exit(1)

        time.sleep(3)


if __name__ == "__main__":
    main()
