#!/usr/bin/env python3
"""
Scout map locations: launch env, skip full reset, then print initial camera
position and orientation every 2 seconds.

Usage (from repo root):
  python scripts/scout_locations.py
  # Forward: w
  # Backward: s
  # Left: a
  # Right: d
  # Ascend: e
  # Descend: c
  # Yaw left: left arrow
  # Yaw right: right arrow
"""

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_VARS_FILE = _REPO_ROOT / "ai_framework" / ".env_vars"
_UAV_FLOW_EVAL = _REPO_ROOT / "UAV-Flow" / "UAV-Flow-Eval"
_UAV_FLOW_ENVS_OVERLAY = _REPO_ROOT / "config" / "uav_flow_envs"
_DOWNTOWN_OVERLAY_JSON = _UAV_FLOW_ENVS_OVERLAY / "Track" / "DowntownWest.json"
_DOWNTOWN_ENV_ID = "UnrealTrack-DowntownWest-ContinuousColor-v0"

DEFAULT_TIME_DILATION = 10
DEFAULT_SEED = 0


class _StopAfterRemoveAgents(Exception):
    """Raised after set_population (remove agents) so we skip the rest of reset and enter the while loop."""


def _load_env_vars() -> None:
    """Load ai_framework/.env_vars into os.environ."""
    if not _ENV_VARS_FILE.exists():
        return
    try:
        with open(_ENV_VARS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("export "):
                    continue
                rest = line[7:].strip()
                if "=" not in rest:
                    continue
                key, _, value = rest.partition("=")
                key = key.strip()
                value = value.strip()
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                if key:
                    os.environ.setdefault(key, value)
    except Exception:
        pass


def _setup_env_and_imports() -> None:
    """Same as run_ltl_planner: UnrealEnv, path, gym_unrealcv, monkey-patches."""
    os.environ.setdefault("UnrealEnv", str(_REPO_ROOT / "envs"))
    uav_eval_str = str(_UAV_FLOW_EVAL)
    if uav_eval_str in sys.path:
        sys.path.remove(uav_eval_str)
    sys.path.insert(0, uav_eval_str)

    for key in list(sys.modules):
        if key == "gym_unrealcv" or key.startswith("gym_unrealcv."):
            del sys.modules[key]

    import gym_unrealcv  # noqa: F401

    import gym_unrealcv.envs.utils.misc as _misc
    _original_get_settingpath = _misc.get_settingpath

    def _get_settingpath(filename):
        if filename == "Track/DowntownWest.json" and _DOWNTOWN_OVERLAY_JSON.exists():
            return str(_DOWNTOWN_OVERLAY_JSON)
        return _original_get_settingpath(filename)

    _misc.get_settingpath = _get_settingpath

    import gym_unrealcv.envs.base_env as _base_env

    def _patched_remove_agent(self, name):
        """Update in-memory state only; skip Unreal destroy_obj so we don't wait, then enter the main loop."""
        agent_index = self.player_list.index(name)
        self.player_list.remove(name)
        self.cam_list = self.remove_cam(name)
        self.action_space.pop(agent_index)
        self.observation_space.pop(agent_index)
        self.agents.pop(name)
        # Intentionally skip: unrealcv.destroy_obj(name) and the wait loop

    _base_env.UnrealCv_base.remove_agent = _patched_remove_agent


def _import_batch():
    """Add paths, chdir to UAV-Flow-Eval, import batch_run_act_all."""
    ai_src = str(_REPO_ROOT / "ai_framework" / "src")
    if ai_src not in sys.path:
        sys.path.insert(0, ai_src)

    os.chdir(str(_UAV_FLOW_EVAL))
    if str(_UAV_FLOW_EVAL) not in sys.path:
        sys.path.insert(0, str(_UAV_FLOW_EVAL))

    import batch_run_act_all as batch
    return batch


def main():
    _load_env_vars()
    parser = argparse.ArgumentParser(
        description="Scout map locations: teleport drone to x,y,z,yaw to observe the view"
    )
    parser.add_argument(
        "-e", "--env_id",
        default=_DOWNTOWN_ENV_ID,
        help="Environment ID",
    )
    parser.add_argument(
        "-t", "--time_dilation",
        type=int,
        default=DEFAULT_TIME_DILATION,
        help="Time dilation for simulator",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed",
    )
    args = parser.parse_args()

    _setup_env_and_imports()
    _import_batch()

    import gym
    from gym_unrealcv.envs.wrappers import time_dilation, configUE, augmentation

    env = gym.make(args.env_id)
    if int(args.time_dilation) > 0:
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    env.unwrapped.agents_category = ["drone"]
    env = configUE.ConfigUEWrapper(env, resolution=(256, 256))
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env.seed(int(args.seed))

    # Skip set_population (remove agents) and the rest of reset; go straight to the while loop.
    _original_set_population = env.unwrapped.set_population

    def _set_population_then_stop(num_agents):
        raise _StopAfterRemoveAgents()

    env.unwrapped.set_population = _set_population_then_stop
    try:
        env.reset()
    except _StopAfterRemoveAgents:
        pass
    env.unwrapped.set_population = _original_set_population

    # Keep the camera at initialization view; do not switch to the drone's camera.
    env.unwrapped.unrealcv.set_phy(env.unwrapped.player_list[0], 0)

    # Initial camera is the env's third/top-view camera (cam_id[0]).
    initial_cam_id = env.unwrapped.cam_id[0]

    while True:
        loc = env.unwrapped.unrealcv.get_cam_location(initial_cam_id)
        rot = env.unwrapped.unrealcv.get_cam_rotation(initial_cam_id)
        x, y, z = loc
        pitch, yaw, roll = rot
        print(
            f"Camera: position=({x:.2f}, {y:.2f}, {z:.2f}), "
            f"orientation pitch={pitch:.2f} yaw={yaw:.2f} roll={roll:.2f}"
        )
        time.sleep(2)
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
