#!/usr/bin/env python3
"""
Run the UAV-Flow simulation evaluator using the original UAV-Flow-Eval code
(UAV-Flow/UAV-Flow-Eval/batch_run_act_all.py) with no modifications.

Uses the real DowntownWest environment (UnrealZoo-UE4 Collection). Config overlay
at config/uav_flow_envs/Track/DowntownWest.json points to envs/UnrealZoo-UE4/.

Sets up:
  - UnrealEnv pointing to repo envs/
  - gym_unrealcv from UAV-Flow-Eval; DowntownWest config from config/uav_flow_envs/

The OpenVLA server must be running separately:
  python scripts/start_openvla_server.py

Task JSONs are read from tasks/uav_flow_tasks/; results (trajectory logs and
plots) are written to results/uav_flow_results/.

Usage (from repo root):
  python scripts/start_openvla_sim.py
  python scripts/start_openvla_sim.py -p 5007 -m 50

All unrecognised arguments are forwarded to batch_run_act_all.py.
"""

import os
import sys
import time
import runpy
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_UAV_FLOW_EVAL = _REPO_ROOT / "UAV-Flow" / "UAV-Flow-Eval"
_BATCH_SCRIPT = _UAV_FLOW_EVAL / "batch_run_act_all.py"
_UAV_FLOW_ENVS_OVERLAY = _REPO_ROOT / "config" / "uav_flow_envs"
_DOWNTOWN_OVERLAY_JSON = _UAV_FLOW_ENVS_OVERLAY / "Track" / "DowntownWest.json"
_DOWNTOWN_ENV_ID = "UnrealTrack-DowntownWest-ContinuousColor-v0"


def _setup_env_and_imports():
    """Set UnrealEnv, inject UAV-Flow-Eval onto sys.path, import its gym_unrealcv."""
    os.environ.setdefault("UnrealEnv", str(_REPO_ROOT / "envs"))

    uav_eval_str = str(_UAV_FLOW_EVAL)
    if uav_eval_str in sys.path:
        sys.path.remove(uav_eval_str)
    sys.path.insert(0, uav_eval_str)

    # Flush any previously-imported gym_unrealcv so we get the UAV-Flow-Eval version
    for key in list(sys.modules):
        if key == "gym_unrealcv" or key.startswith("gym_unrealcv."):
            del sys.modules[key]

    import gym_unrealcv  # noqa: F401  -- triggers env registrations from UAV-Flow-Eval

    # Monkey-patch so DowntownWest uses our overlay config (Linux binary path)
    import gym_unrealcv.envs.utils.misc as _misc
    _original_get_settingpath = _misc.get_settingpath

    def _get_settingpath(filename):
        if filename == "Track/DowntownWest.json" and _DOWNTOWN_OVERLAY_JSON.exists():
            return str(_DOWNTOWN_OVERLAY_JSON)
        return _original_get_settingpath(filename)

    _misc.get_settingpath = _get_settingpath

    # Monkey-patch remove_agent to add a timeout so the simulator does not stall when
    # Unreal (e.g. Collection/UE4) never updates camera count after destroy_obj.
    import gym_unrealcv.envs.base_env as _base_env

    _REMOVE_AGENT_WAIT_TIMEOUT_S = 10.0
    _REMOVE_AGENT_WAIT_SLEEP_S = 0.2

    def _patched_remove_agent(self, name):
        agent_index = self.player_list.index(name)
        self.player_list.remove(name)
        last_cam_list = self.cam_list
        self.cam_list = self.remove_cam(name)
        self.action_space.pop(agent_index)
        self.observation_space.pop(agent_index)
        self.unrealcv.destroy_obj(name)
        self.agents.pop(name)
        st_time = time.time()
        time.sleep(1)
        print(f'waiting for remove agent {name}...')
        while self.unrealcv.get_camera_num() > len(last_cam_list) + 1:
            if time.time() - st_time > _REMOVE_AGENT_WAIT_TIMEOUT_S:
                print('Remove agent wait timed out; continuing.')
                break
            time.sleep(_REMOVE_AGENT_WAIT_SLEEP_S)
        print('Remove finished!')

    _base_env.UnrealCv_base.remove_agent = _patched_remove_agent

    # Monkey-patch Track.get_tracker_init_point so direction is a scalar when None:
    # np.random.sample(1) returns shape (1,), so distance * np.cos(direction) is an array
    # and float() raises "only 0-dimensional arrays can be converted to Python scalars".
    import gym_unrealcv.envs.track as _track_module

    def _patched_get_tracker_init_point(self, target_pos, distance, direction=None):
        if direction is None:
            direction = 2 * np.pi * np.random.sample(1)
        else:
            direction = direction % (2 * np.pi)
        direction = float(np.asarray(direction).flat[0])
        distance = float(np.asarray(distance).flat[0])
        dx = float(distance * np.cos(direction))
        dy = float(distance * np.sin(direction))
        x = dx + float(np.asarray(target_pos[0]).flat[0])
        y = dy + float(np.asarray(target_pos[1]).flat[0])
        z = float(np.asarray(target_pos[2]).flat[0])
        cam_pos_exp = [x, y, z]
        yaw = float(direction / np.pi * 180 - 180)
        return [cam_pos_exp, yaw]

    _track_module.Track.get_tracker_init_point = _patched_get_tracker_init_point


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
        "-e": _DOWNTOWN_ENV_ID,
        "-f": str(_REPO_ROOT / "tasks" / "uav_flow_tasks"),
        "-o": str(_REPO_ROOT / "results" / "uav_flow_results"),
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

    if not _BATCH_SCRIPT.exists():
        print(
            f"Error: batch_run_act_all.py not found at {_BATCH_SCRIPT}",
            file=sys.stderr,
        )
        sys.exit(1)

    _setup_env_and_imports()

    os.chdir(str(_UAV_FLOW_EVAL))

    sys.argv = _build_argv(extra_args)
    print(f"[scripts] Running batch_run_act_all.py with args: {sys.argv[1:]}")
    print(f"[scripts] CWD: {os.getcwd()}")
    print(f"[scripts] UnrealEnv: {os.environ.get('UnrealEnv')}")

    runpy.run_path(str(_BATCH_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
