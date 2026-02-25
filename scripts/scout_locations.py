#!/usr/bin/env python3
"""
Scout map locations: same environment as run_openvla_ltl.py, but only teleports
the drone to user-specified global positions so you can observe what different
positions look like.

Usage (from repo root):
  python scripts/scout_locations.py
  # Then at the prompt enter: 100, 100, 140, 61   (x, y, z, yaw)
  # Or: -600, -1270, 128, 61
  # Enter q or quit to exit.
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
    """Same as run_openvla_ltl: UnrealEnv, path, gym_unrealcv, monkey-patches."""
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
        print("waiting for remove agent {}...".format(name))
        while self.unrealcv.get_camera_num() > len(last_cam_list) + 1:
            if time.time() - st_time > _REMOVE_AGENT_WAIT_TIMEOUT_S:
                print("Remove agent wait timed out; continuing.")
                break
            time.sleep(_REMOVE_AGENT_WAIT_SLEEP_S)
        print("Remove finished!")

    _base_env.UnrealCv_base.remove_agent = _patched_remove_agent

    import gym_unrealcv.envs.track as _track_module
    import numpy as np

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


def _parse_position(s: str):
    """Parse 'x, y, z, yaw' into four floats. Raises ValueError if invalid."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("Enter exactly 4 numbers: x, y, z, yaw (e.g. 100, 100, 140, 61)")
    return [float(x) for x in parts]


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
    batch = _import_batch()

    import gym
    from gym_unrealcv.envs.wrappers import time_dilation, configUE, augmentation

    env = gym.make(args.env_id)
    if int(args.time_dilation) > 0:
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    env.unwrapped.agents_category = ["drone"]
    env = configUE.ConfigUEWrapper(env, resolution=(256, 256))
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env.seed(int(args.seed))
    env.reset()
    env.unwrapped.unrealcv.set_viewport(env.unwrapped.player_list[0])
    env.unwrapped.unrealcv.set_phy(env.unwrapped.player_list[0], 0)

    time.sleep(batch.SLEEP_SHORT_S)
    env.unwrapped.unrealcv.new_obj("bp_character_C", "BP_Character_21", [0, 0, 0])
    env.unwrapped.unrealcv.set_appearance("BP_Character_21", 0)
    env.unwrapped.unrealcv.set_obj_rotation("BP_Character_21", [0, 0, 0])
    time.sleep(batch.SLEEP_SHORT_S)
    env.unwrapped.unrealcv.new_obj("BP_BaseCar_C", "BP_Character_22", [1000, 0, 0])
    env.unwrapped.unrealcv.set_appearance("BP_Character_22", 2)
    env.unwrapped.unrealcv.set_obj_rotation("BP_Character_22", [0, 0, 0])
    env.unwrapped.unrealcv.set_phy("BP_Character_22", 0)
    time.sleep(batch.SLEEP_SHORT_S)

    # Initial teleport so view is defined (optional default)
    env.unwrapped.unrealcv.set_obj_location(
        env.unwrapped.player_list[0], [0.0, 0.0, 100.0]
    )
    env.unwrapped.unrealcv.set_rotation(env.unwrapped.player_list[0], -180)
    batch.set_cam(env)
    time.sleep(batch.SLEEP_AFTER_RESET_S)

    print("Scout locations (same env as run_openvla_ltl). Enter: x, y, z, yaw")
    print("Example: 100, 100, 140, 61   or   -600, -1270, 128, 61")
    print("Quit: q or quit")
    print()

    while True:
        try:
            line = input("x, y, z, yaw> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in ("q", "quit"):
            break
        try:
            x, y, z, yaw = _parse_position(line)
        except ValueError as e:
            print(e)
            continue

        env.unwrapped.unrealcv.set_obj_location(
            env.unwrapped.player_list[0], [x, y, z]
        )
        env.unwrapped.unrealcv.set_rotation(
            env.unwrapped.player_list[0], yaw - 180
        )
        batch.set_cam(env)
        time.sleep(0.2)
        print("Teleported to ({}, {}, {}), yaw={}".format(x, y, z, yaw))

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
