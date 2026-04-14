#!/usr/bin/env python3
"""
Interactive REPL for issuing natural-language commands directly to OpenVLA.

Uses the same simulation environment as the other runners. After the simulator
initialises, you are prompted for the initial drone position (x,y,z,yaw) and
then enter an interactive loop where you type commands that are sent straight
to the VLA — no LTL planning or diary monitoring.

Capabilities:
- An action runs until the model predicts small changes 10 steps in a row, or
  you press Ctrl+C to abort.
- Relative position fed to the model is reset at the start of each action
  (first step gets 0,0,0,0) and accumulates until the action ends.
- 'where' (or 'location', 'pos', 'pose') prints current position/orientation.
- 'pos x,y,z,yaw' or 'teleport x,y,z,yaw' teleports the drone.
- 'undo' (or 'u') restores the drone to the position before the last action.
- 'record' / 'record start' saves frames under results/repl_results/run_<ts>/.
- Up/Down arrow keys browse command history (~/.rvln_openvla_repl_history).
- The simulator is not closed when the script exits.

OpenVLA server must be running: python scripts/start_openvla_server.py

Usage (from repo root):
  python scripts/run_repl.py
  python scripts/run_repl.py -p 5007
  python scripts/run_repl.py --use-default-cam   # skip camera selection
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import readline  # noqa: F401
except ImportError:
    pass

from PIL import Image

from rvln.paths import (
    BATCH_SCRIPT,
    DEFAULT_INITIAL_POSITION,
    DEFAULT_SERVER_PORT,
    DEFAULT_SEED,
    DEFAULT_TIME_DILATION,
    DOWNTOWN_ENV_ID,
    DRONE_CAM_ID,
    REPO_ROOT,
)
from rvln.sim.env_setup import (
    apply_action_poses,
    import_batch_module,
    interactive_camera_select,
    load_env_vars,
    normalize_angle,
    parse_position,
    relative_pose_to_world,
    set_drone_cam_and_get_image,
    setup_env_and_imports,
    setup_sim_env,
    state_for_openvla,
)

_REPL_HISTORY_PATH = Path.home() / ".rvln_openvla_repl_history"
_REPL_RESULTS_DIR = REPO_ROOT / "results" / "repl_results"

_recording_state: Dict[str, Any] = {"dir": None, "frame_index": 0}

logger = logging.getLogger(__name__)


def _load_repl_history() -> None:
    """Load REPL history from file so up/down arrow works across sessions."""
    try:
        import readline
        if _REPL_HISTORY_PATH.exists():
            readline.read_history_file(str(_REPL_HISTORY_PATH))
        readline.set_history_length(500)
    except (ImportError, OSError):
        pass


def _save_repl_history() -> None:
    """Save REPL history to file."""
    try:
        import readline
        readline.write_history_file(str(_REPL_HISTORY_PATH))
    except (ImportError, OSError):
        pass


def _get_current_pose(env: Any) -> List[float]:
    """Read current drone position and yaw from the simulator. Returns [x, y, z, yaw_deg]."""
    x, y, z = env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0])
    roll, yaw, pitch = env.unwrapped.unrealcv.get_obj_rotation(env.unwrapped.player_list[0])
    yaw_deg = normalize_angle(float(yaw))
    return [float(x), float(y), float(z), yaw_deg]


def _valid_action_poses(action_poses: Any) -> bool:
    """Return True if action_poses is a list of pose tuples with 4 numeric elements each."""
    if not isinstance(action_poses, list) or len(action_poses) == 0:
        return False
    for ap in action_poses:
        if not (isinstance(ap, (list, tuple)) and len(ap) >= 4):
            return False
        try:
            for j in range(4):
                float(ap[j])
        except (TypeError, ValueError):
            return False
    return True


def _run_one_step(
    env: Any,
    instruction: str,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    origin_yaw: float,
    current_pose: List[float],
    server_url: str,
    batch: Any,
    drone_cam_id: int,
) -> tuple:
    """
    One VLA step: get image, send to server, apply poses.
    Returns (new_image, new_relative_pose) or (None, current_pose) on error.
    """
    batch.set_cam(env)
    image = set_drone_cam_and_get_image(env, drone_cam_id)
    if image is None:
        logger.warning("No image from drone camera.")
        return None, current_pose

    rec_dir = _recording_state.get("dir")
    if rec_dir is not None and image is not None:
        try:
            import cv2
            frame_path = rec_dir / "frame_{:06d}.png".format(_recording_state["frame_index"])
            cv2.imwrite(str(frame_path), image)
            _recording_state["frame_index"] += 1
        except Exception as e:
            logger.debug("Failed to save recorded frame: %s", e)

    proprio = state_for_openvla(current_pose)
    response = batch.send_prediction_request(
        image=Image.fromarray(image),
        proprio=proprio,
        instr=instruction.strip().lower(),
        server_url=server_url,
    )

    if response is None:
        logger.warning("No valid response from VLA.")
        return image, current_pose

    action_poses_data = response.get("action")
    if not _valid_action_poses(action_poses_data):
        logger.warning("Response 'action' empty or invalid (expected list of [x,y,z,yaw] poses).")
        return image, current_pose

    try:
        new_image, new_pose, _ = apply_action_poses(
            env,
            action_poses_data,
            origin_x,
            origin_y,
            origin_z,
            origin_yaw,
            batch.set_cam,
            sleep_s=0.1,
            drone_cam_id=drone_cam_id,
        )
    except Exception as e:
        logger.error("Error executing action: %s", e)
        return image, current_pose

    return (new_image if new_image is not None else image), new_pose


def _run_action_until_done(
    env: Any,
    instruction: str,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    origin_yaw: float,
    server_url: str,
    batch: Any,
    drone_cam_id: int,
) -> tuple:
    """
    Run VLA action until model predicts small changes 10 steps in a row, or Ctrl+C.
    Returns (final_relative_pose, aborted_by_user).
    """
    current_pose: List[float] = [0.0, 0.0, 0.0, 0.0]
    last_pose: Optional[List[float]] = None
    small_count = 0
    image = None
    aborted = False

    while True:
        try:
            image, current_pose = _run_one_step(
                env,
                instruction,
                origin_x,
                origin_y,
                origin_z,
                origin_yaw,
                current_pose,
                server_url,
                batch,
                drone_cam_id,
            )
        except KeyboardInterrupt:
            aborted = True
            break

        if image is None:
            break

        pose_now = current_pose
        if last_pose is not None:
            diffs = [abs(a - b) for a, b in zip(pose_now, last_pose)]
            if (
                all(d < batch.ACTION_SMALL_DELTA_POS for d in diffs[:3])
                and diffs[3] < batch.ACTION_SMALL_DELTA_YAW
            ):
                small_count += 1
            else:
                small_count = 0
            if small_count >= batch.ACTION_SMALL_STEPS:
                logger.debug(
                    "Detected %d steps with very small change, ending action.",
                    batch.ACTION_SMALL_STEPS,
                )
                break
        last_pose = pose_now

    return current_pose, aborted


def _drain_connections_after_abort(env: Any, batch: Any, drone_cam_id: int) -> None:
    """After Ctrl+C, drain UnrealCV socket so the next action gets clean reads."""
    for _ in range(3):
        try:
            batch.set_cam(env)
            set_drone_cam_and_get_image(env, drone_cam_id)
        except Exception:
            pass
        time.sleep(0.2)


def main() -> None:
    load_env_vars()
    parser = argparse.ArgumentParser(
        description="Interactive REPL for natural-language drone commands via OpenVLA"
    )
    parser.add_argument(
        "-p",
        "--server_port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="OpenVLA server port",
    )
    parser.add_argument(
        "-e",
        "--env_id",
        default=DOWNTOWN_ENV_ID,
        help="Environment ID",
    )
    parser.add_argument(
        "-t",
        "--time_dilation",
        type=int,
        default=DEFAULT_TIME_DILATION,
        help="Time dilation for simulator",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--use-default-cam",
        action="store_true",
        help="Use the default drone camera (ID %d) and skip interactive camera selection." % DRONE_CAM_ID,
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    if not BATCH_SCRIPT.exists():
        logger.error("batch_run_act_all.py not found at %s", BATCH_SCRIPT)
        sys.exit(1)

    setup_env_and_imports()
    batch = import_batch_module()

    import requests
    _original_requests_post = requests.post

    def _repl_requests_post(*a, **kw):
        h = dict(kw.get("headers") or {})
        h["Connection"] = "close"
        kw["headers"] = h
        return _original_requests_post(*a, **kw)

    requests.post = _repl_requests_post

    server_url = "http://127.0.0.1:{}".format(args.server_port) + "/predict"

    env = setup_sim_env(args.env_id, int(args.time_dilation), int(args.seed), batch)

    logger.info("Simulator ready.")
    _load_repl_history()
    try:
        line = input("\nEnter initial drone position (x,y,z,yaw) [default: {}]: ".format(DEFAULT_INITIAL_POSITION))
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)
    line = line.strip()
    if not line:
        line = DEFAULT_INITIAL_POSITION
    try:
        pos = parse_position(line)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)

    env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[0], pos[0:3])
    env.unwrapped.unrealcv.set_rotation(env.unwrapped.player_list[0], pos[3] - 180)
    batch.set_cam(env)
    time.sleep(batch.SLEEP_AFTER_RESET_S)

    if args.use_default_cam:
        drone_cam_id = DRONE_CAM_ID
    else:
        logger.info("Camera selection: use the window to pick the camera for OpenVLA.")
        drone_cam_id = interactive_camera_select(env, pos, batch)

    origin_x, origin_y, origin_z, origin_yaw = pos[0], pos[1], pos[2], pos[3]
    current_pose: List[float] = [0.0, 0.0, 0.0, 0.0]
    origin_history: List[tuple] = []

    print(
        "REPL: action (natural language) | 'pos x,y,z,yaw' or 'teleport x,y,z,yaw' | 'where' for current pose | "
        "'record' / 'record start' to start saving VLA frames, 'record stop' to stop | 'undo' to undo last action | 'quit' to exit. Simulator stays open. Up/Down arrows: history."
    )
    print()

    try:
        while True:
            try:
                line = input("Action: ")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting. Simulator not closed.")
                break
            line = line.strip()
            if not line:
                continue

            if line.lower() in ("quit", "exit", "q"):
                print("Exiting. Simulator not closed.")
                break

            if line.lower() in ("record", "record start", "start record"):
                _REPL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                run_name = "run_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                run_dir = _REPL_RESULTS_DIR / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                _recording_state["dir"] = run_dir
                _recording_state["frame_index"] = 0
                print("Recording started. Frames will be saved to: {}".format(run_dir))
                continue

            if line.lower() in ("record stop", "stop record"):
                if _recording_state["dir"] is None:
                    print("Not recording.")
                else:
                    n = _recording_state["frame_index"]
                    path = _recording_state["dir"]
                    _recording_state["dir"] = None
                    _recording_state["frame_index"] = 0
                    print("Recording stopped. Saved {} frames to: {}".format(n, path))
                continue

            if line.lower() in ("where", "location", "pos", "pose"):
                pose = _get_current_pose(env)
                print("x={}, y={}, z={}, yaw={}".format(pose[0], pose[1], pose[2], pose[3]))
                continue

            if line.lower() in ("undo", "u"):
                if not origin_history:
                    print("Nothing to undo.")
                    continue
                ox, oy, oz, oyaw = origin_history.pop()
                env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[0], [ox, oy, oz])
                env.unwrapped.unrealcv.set_rotation(env.unwrapped.player_list[0], oyaw - 180)
                batch.set_cam(env)
                time.sleep(0.2)
                origin_x, origin_y, origin_z, origin_yaw = ox, oy, oz, oyaw
                current_pose = [0.0, 0.0, 0.0, 0.0]
                print("Undone. Position: x={}, y={}, z={}, yaw={}".format(ox, oy, oz, oyaw))
                continue

            if line.lower().startswith("pos ") or line.lower().startswith("teleport "):
                prefix = "pos " if line.lower().startswith("pos ") else "teleport "
                rest = line[len(prefix):].strip()
                try:
                    pos = parse_position(rest)
                except ValueError as e:
                    print("Invalid position: {}".format(e))
                    continue
                env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[0], pos[0:3])
                env.unwrapped.unrealcv.set_rotation(env.unwrapped.player_list[0], pos[3] - 180)
                batch.set_cam(env)
                time.sleep(0.2)
                origin_x, origin_y, origin_z, origin_yaw = pos[0], pos[1], pos[2], pos[3]
                current_pose = [0.0, 0.0, 0.0, 0.0]
                print("Position set to x={}, y={}, z={}, yaw={}".format(*pos))
                continue

            origin_history.append((origin_x, origin_y, origin_z, origin_yaw))
            batch.reset_model(server_url)
            new_pose, aborted = _run_action_until_done(
                env,
                line,
                origin_x,
                origin_y,
                origin_z,
                origin_yaw,
                server_url,
                batch,
                drone_cam_id,
            )
            origin_x, origin_y, origin_z, origin_yaw = relative_pose_to_world(
                origin_x, origin_y, origin_z, origin_yaw, new_pose
            )
            current_pose = [0.0, 0.0, 0.0, 0.0]
            if aborted:
                _drain_connections_after_abort(env, batch, drone_cam_id)
                print("Action aborted.")
            else:
                print("ACTION COMPLETE")
            print()
    finally:
        _save_repl_history()


if __name__ == "__main__":
    main()
