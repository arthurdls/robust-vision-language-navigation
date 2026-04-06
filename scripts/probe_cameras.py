#!/usr/bin/env python3
"""
Quick probe: set up env like run_ltl_planner, run a basic command (e.g. "Go to the bush"),
and save a frame from every camera at each step. Outputs go to probe_results/<camera_number>/.

Usage (from repo root):
  python scripts/probe_cameras.py
  python scripts/probe_cameras.py --instruction "Go to the tree" --num_steps 3

OpenVLA server must be running: python scripts/start_openvla_server.py
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, List

import numpy as np
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from sim_common import (
    BATCH_SCRIPT,
    DEFAULT_INITIAL_POSITION,
    REPO_ROOT,
    apply_action_poses,
    import_batch_module,
    load_env_vars,
    normalize_initial_pos,
    parse_position,
    set_drone_cam_and_get_image,
    setup_env_and_imports,
    setup_sim_env,
)

_PROBE_RESULTS_DIR = REPO_ROOT / "probe_results"

logger = logging.getLogger(__name__)


def main() -> None:
    load_env_vars()
    parser = argparse.ArgumentParser(description="Probe all cameras while running a basic command.")
    parser.add_argument(
        "--instruction",
        default="Go to the bush",
        help="Instruction to send to OpenVLA (default: Go to the bush)",
    )
    parser.add_argument(
        "--initial-position",
        default=DEFAULT_INITIAL_POSITION,
        help="Initial pose x,y,z,yaw (default: %s)" % DEFAULT_INITIAL_POSITION,
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

    if not BATCH_SCRIPT.exists():
        logger.error("batch_run_act_all.py not found at %s", BATCH_SCRIPT)
        sys.exit(1)

    setup_env_and_imports()
    batch = import_batch_module()

    server_url = "http://127.0.0.1:{}".format(args.server_port) + "/predict"

    env = setup_sim_env(args.env_id, 0, args.seed, batch)

    initial_pos = normalize_initial_pos(parse_position(args.initial_position))
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

    save_all_cameras(0)
    image = set_drone_cam_and_get_image(env)
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
            new_image, current_pose, _ = apply_action_poses(
                env,
                action_poses,
                initial_x,
                initial_y,
                initial_z,
                initial_yaw,
                batch.set_cam,
                trajectory_log=trajectory_log,
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
