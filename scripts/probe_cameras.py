#!/usr/bin/env python3
"""
Quick probe: set up env like run_openvla_ltl, run a basic command (e.g. "Go to the bush"),
and save a frame from every camera at each step. Outputs go to probe_results/<camera_number>/.

Usage (from repo root):
  python scripts/probe_cameras.py
  python scripts/probe_cameras.py --instruction "Go to the tree" --num_steps 3

OpenVLA server must be running: python scripts/start_openvla_server.py
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, List

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROBE_RESULTS_DIR = _REPO_ROOT / "probe_results"
_DEFAULT_INITIAL_POSITION = "-600, -1270, 128, 61"

logger = logging.getLogger(__name__)


def _parse_initial_position(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--initial-position must be 4 numbers: x,y,z,yaw")
    return [float(x) for x in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe all cameras while running a basic command.")
    parser.add_argument(
        "--instruction",
        default="Go to the bush",
        help="Instruction to send to OpenVLA (default: Go to the bush)",
    )
    parser.add_argument(
        "--initial-position",
        default=_DEFAULT_INITIAL_POSITION,
        help="Initial pose x,y,z,yaw (default: %s)" % _DEFAULT_INITIAL_POSITION,
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5,
        help="Number of inference steps (default: 5)",
    )
    parser.add_argument(
        "--env_id",
        default="UnrealTrack-DowntownWest-ContinuousColor-v0",
        help="Gym env id",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=5007,
        help="OpenVLA server port",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    # Paths and imports (same as run_openvla_ltl)
    _UAV_FLOW_EVAL = _REPO_ROOT / "UAV-Flow" / "UAV-Flow-Eval"
    _BATCH_SCRIPT = _UAV_FLOW_EVAL / "batch_run_act_all.py"
    _DOWNTOWN_OVERLAY_JSON = _REPO_ROOT / "config" / "uav_flow_envs" / "Track" / "DowntownWest.json"

    if not _BATCH_SCRIPT.exists():
        logger.error("batch_run_act_all.py not found at %s", _BATCH_SCRIPT)
        sys.exit(1)

    # Setup env and imports (from run_openvla_ltl)
    import run_openvla_ltl as rtl

    rtl._setup_env_and_imports()
    batch, _, _, _, _ = rtl._import_batch_and_helpers()

    os.chdir(str(_UAV_FLOW_EVAL))
    server_url = "http://127.0.0.1:{}".format(args.server_port) + "/predict"

    import gym
    from gym_unrealcv.envs.wrappers import time_dilation, configUE, augmentation

    env = gym.make(args.env_id)
    env.unwrapped.agents_category = ["drone"]
    env = configUE.ConfigUEWrapper(env, resolution=(256, 256))
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env.seed(args.seed)
    env.reset()
    env.unwrapped.unrealcv.set_viewport(env.unwrapped.player_list[0])
    env.unwrapped.unrealcv.set_phy(env.unwrapped.player_list[0], 0)
    logger.info(env.unwrapped.unrealcv.get_camera_config())

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

    initial_pos = rtl._normalize_initial_pos(_parse_initial_position(args.initial_position))
    initial_x, initial_y, initial_z = initial_pos[0:3]
    initial_yaw = initial_pos[4]

    env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[0], initial_pos[0:3])
    env.unwrapped.unrealcv.set_rotation(env.unwrapped.player_list[0], initial_pos[4] - 180)
    batch.set_cam(env)
    time.sleep(batch.SLEEP_AFTER_RESET_S)

    n_cams = env.unwrapped.unrealcv.get_camera_num()
    _PROBE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for cam_id in range(n_cams):
        (_PROBE_RESULTS_DIR / str(cam_id)).mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except ImportError:
        cv2 = None

    def save_all_cameras(step: int) -> None:
        for cam_id in range(n_cams):
            try:
                img = env.unwrapped.unrealcv.get_image(cam_id, "lit")
                if img is not None and cv2 is not None:
                    path = _PROBE_RESULTS_DIR / str(cam_id) / "frame_{:06d}.png".format(step)
                    cv2.imwrite(str(path), img)
                    logger.debug("Saved camera %d frame %d", cam_id, step)
            except Exception as e:
                logger.warning("Camera %d frame %d: %s", cam_id, step, e)

    # Initial frames (step 0)
    save_all_cameras(0)
    image = rtl._set_drone_cam_and_get_image(env)
    current_pose = [0.0, 0.0, 0.0, 0.0]
    trajectory_log: List[Any] = []

    for step in range(1, args.num_steps + 1):
        if image is None:
            logger.warning("No image, stopping probe.")
            break

        response = batch.send_prediction_request(
            image=Image.fromarray(image),
            proprio=np.array(current_pose),
            instr=(args.instruction or "").strip().lower(),
            server_url=server_url,
        )
        if response is None:
            logger.warning("No response, stopping probe.")
            break

        action_poses = response.get("action")
        if not isinstance(action_poses, list) or len(action_poses) == 0:
            logger.warning("Empty action, stopping probe.")
            break

        try:
            new_image, current_pose, _ = rtl._apply_action_poses(
                env,
                action_poses,
                initial_x,
                initial_y,
                initial_z,
                initial_yaw,
                batch.set_cam,
                trajectory_log,
                sleep_s=0.1,
            )
        except Exception as e:
            logger.error("apply_action_poses: %s", e)
            break

        if new_image is not None:
            image = new_image

        save_all_cameras(step)

    logger.info("Probe done. Frames saved under %s", _PROBE_RESULTS_DIR)


if __name__ == "__main__":
    main()
