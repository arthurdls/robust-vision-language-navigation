#!/usr/bin/env python3
"""
Run the batch UAV-Flow evaluation suite.

Executes evaluation tasks against the DowntownWest Unreal environment.
Task JSONs are read from tasks/uav_flow/. Results (trajectory logs and plots)
are written to results/uav_flow_results/.

The OpenVLA server must be running separately:
  python scripts/start_server.py

Usage (from repo root):
  python scripts/run_eval.py
  python scripts/run_eval.py -p 5007 -m 50

All unrecognised arguments are forwarded to the batch runner.
"""

import os
import sys
import runpy

from rvln.paths import REPO_ROOT, UAV_FLOW_EVAL, DOWNTOWN_ENV_ID, BATCH_SCRIPT
from rvln.sim.env_setup import setup_env_and_imports


def _build_argv(extra_args: list):
    """Build sys.argv for batch_run_act_all.py, filling in defaults for missing args."""
    specified = set()
    i = 0
    while i < len(extra_args):
        specified.add(extra_args[i])
        i += 1
        if i < len(extra_args) and not extra_args[i].startswith("-"):
            i += 1

    argv = ["batch_run_act_all.py"]
    defaults = {
        "-e": DOWNTOWN_ENV_ID,
        "-f": str(REPO_ROOT / "tasks" / "uav_flow"),
        "-o": str(REPO_ROOT / "results" / "uav_flow_results"),
        "-p": "5007",
        "-m": "100",
        "-t": "10",
    }
    long_to_short = {
        "--env_id": "-e", "--time_dilation": "-t", "--seed": "-s",
        "--json_folder": "-f", "--images_dir": "-o", "--server_port": "-p",
        "--max_steps": "-m",
    }

    for short, val in defaults.items():
        long = [k for k, v in long_to_short.items() if v == short]
        if short not in specified and not any(l in specified for l in long):
            argv.extend([short, val])

    argv.extend(extra_args)
    return argv


def main():
    extra_args = sys.argv[1:]

    if not BATCH_SCRIPT.exists():
        print(
            f"Error: batch_run_act_all.py not found at {BATCH_SCRIPT}",
            file=sys.stderr,
        )
        sys.exit(1)

    setup_env_and_imports()
    import gym_unrealcv  # noqa: F401

    os.chdir(str(UAV_FLOW_EVAL))

    sys.argv = _build_argv(extra_args)
    print(f"[scripts] Running batch_run_act_all.py with args: {sys.argv[1:]}")
    print(f"[scripts] CWD: {os.getcwd()}")
    print(f"[scripts] UnrealEnv: {os.environ.get('UnrealEnv')}")

    runpy.run_path(str(BATCH_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
