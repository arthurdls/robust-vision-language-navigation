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
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.paths import DOWNTOWN_ENV_ID, DEFAULT_TIME_DILATION, DEFAULT_SEED
from rvln.sim.env_setup import load_env_vars, setup_env_and_imports, import_batch_module

import gym_unrealcv.envs.base_env as _base_env


class _StopAfterRemoveAgents(Exception):
    """Raised after set_population (remove agents) so we skip the rest of reset and enter the while loop."""


def _patch_remove_agent_skip_destroy() -> None:
    """Override remove_agent to skip destroy_obj entirely (faster for scouting)."""

    def _patched_remove_agent(self, name):
        agent_index = self.player_list.index(name)
        self.player_list.remove(name)
        self.cam_list = self.remove_cam(name)
        self.action_space.pop(agent_index)
        self.observation_space.pop(agent_index)
        self.agents.pop(name)

    _base_env.UnrealCv_base.remove_agent = _patched_remove_agent


def main():
    load_env_vars()
    parser = argparse.ArgumentParser(
        description="Scout map locations: teleport drone to x,y,z,yaw to observe the view"
    )
    parser.add_argument(
        "-e", "--env_id",
        default=DOWNTOWN_ENV_ID,
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

    setup_env_and_imports()
    _patch_remove_agent_skip_destroy()
    import_batch_module()

    import gymnasium as gym
    import gym_unrealcv
    from gym_unrealcv.envs.wrappers import time_dilation, configUE, augmentation

    gym_unrealcv.register_env(args.env_id)
    env = gym.make(args.env_id)
    if int(args.time_dilation) > 0:
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    env.unwrapped.agents_category = ["drone"]
    env = configUE.ConfigUEWrapper(env, resolution=(256, 256))
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    # Skip set_population (remove agents) and the rest of reset; go straight to the while loop.
    _original_set_population = env.unwrapped.set_population

    def _set_population_then_stop(num_agents):
        raise _StopAfterRemoveAgents()

    env.unwrapped.set_population = _set_population_then_stop
    try:
        env.reset(seed=int(args.seed))
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
