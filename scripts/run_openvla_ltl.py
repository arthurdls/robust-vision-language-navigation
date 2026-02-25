#!/usr/bin/env python3
"""
Run OpenVLA simulation with LTL-based symbolic planning and goal adherence monitoring.

Uses the same env/setup as start_openvla_sim.py but runs a custom control loop that:
- Breaks the task instruction into LTL subgoals and feeds one subgoal at a time to OpenVLA.
- When OpenVLA reports done, verifies the subgoal via a goal monitor before advancing.
- Saves every camera frame sent to the model and run metadata under results/ltl_results/.

OpenVLA server must be running: python scripts/start_openvla_server.py

Usage (from repo root):
  # Ad-hoc command (require --initial-position)
  python scripts/run_openvla_ltl.py -c "Go to the red building..." --initial-position 100,100,100,61
  # Single task from tasks/ltl_tasks/
  python scripts/run_openvla_ltl.py --task first_task.json
  # All tasks in tasks/ltl_tasks/
  python scripts/run_openvla_ltl.py --run_all_tasks
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_VARS_FILE = _REPO_ROOT / "ai_framework" / ".env_vars"
_UAV_FLOW_EVAL = _REPO_ROOT / "UAV-Flow" / "UAV-Flow-Eval"
_BATCH_SCRIPT = _UAV_FLOW_EVAL / "batch_run_act_all.py"
_UAV_FLOW_ENVS_OVERLAY = _REPO_ROOT / "config" / "uav_flow_envs"
_DOWNTOWN_OVERLAY_JSON = _UAV_FLOW_ENVS_OVERLAY / "Track" / "DowntownWest.json"
_DOWNTOWN_ENV_ID = "UnrealTrack-DowntownWest-ContinuousColor-v0"

# LTL task and result paths
LTL_TASKS_DIR = _REPO_ROOT / "tasks" / "ltl_tasks"
LTL_RESULTS_DIR = _REPO_ROOT / "results" / "ltl_results"

DEFAULT_SERVER_PORT = 5007
DEFAULT_MAX_STEPS = 100
DEFAULT_TIME_DILATION = 10
DEFAULT_SEED = 0
# Default initial position (x,y,z,yaw) when -c is used without --initial-position; supports negative numbers
DEFAULT_INITIAL_POSITION = "-600, -1270, 128, 61"
IMAGE_HISTORY_LEN = 10
GOAL_MONITOR_PERIODIC_STEPS = 30
# Thresholds for "small movement" to end current subgoal (tighter = smaller deltas required)
SMALL_DELTA_POS = 3.0
SMALL_DELTA_YAW = 1.0
# Drone POV camera ID (determined from probe_cameras.py; used for frames sent to OpenVLA and saved)
DRONE_CAM_ID = 5

# State format sent to OpenVLA (must match UAV-Flow/UAV-Flow-Eval/batch_run_act_all.py and
# UAV-Flow/OpenVLA-UAV/vla-scripts/openvla_act.py): [x, y, z, yaw_deg] relative to initial pose.
# Server expects proprio[-1] in degrees (openvla_act uses np.deg2rad(proprio[-1])).
PROPRIO_LEN = 4

logger = logging.getLogger(__name__)


def _state_for_openvla(current_pose: List[float]) -> np.ndarray:
    """Format current state for OpenVLA /predict payload (same as UAV-Flow batch_run_act_all).

    Returns a 4-float array [x, y, z, yaw_degrees]. In the LTL loop, current_pose is always
    relative to the start of the current sub-task and is reset to [0,0,0,0] when advancing.
    """
    if len(current_pose) >= PROPRIO_LEN:
        out = [
            float(current_pose[0]),
            float(current_pose[1]),
            float(current_pose[2]),
            float(current_pose[3]),
        ]
    else:
        out = [0.0] * PROPRIO_LEN
        for i in range(min(len(current_pose), PROPRIO_LEN)):
            out[i] = float(current_pose[i])
    arr = np.array(out, dtype=np.float32)
    _verify_proprio_format(arr)
    return arr


def _verify_proprio_format(proprio: np.ndarray) -> None:
    """Verify proprio is in intended format for OpenVLA: [x, y, z, yaw_deg], 4 floats, yaw in degrees."""
    assert proprio.shape == (PROPRIO_LEN,), (
        f"proprio must have length {PROPRIO_LEN} (x, y, z, yaw_deg), got shape {proprio.shape}"
    )
    assert proprio.dtype == np.float32, (
        f"proprio must be float32 for server, got dtype {proprio.dtype}"
    )
    assert np.all(np.isfinite(proprio)), (
        f"proprio must be finite, got {proprio.tolist()}"
    )
    # yaw (index 3) is in degrees; server uses np.deg2rad(proprio[-1])
    logger.debug(
        "proprio format OK: [x=%.2f, y=%.2f, z=%.2f, yaw_deg=%.2f]",
        float(proprio[0]), float(proprio[1]), float(proprio[2]), float(proprio[3]),
    )


def _load_env_vars() -> None:
    """Load ai_framework/.env_vars (export KEY=value lines) into os.environ so API keys are set."""
    if not _ENV_VARS_FILE.exists():
        return
    try:
        with open(_ENV_VARS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("export "):
                    continue
                rest = line[7:].strip()  # after "export "
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
    except Exception as e:
        logger.warning("Could not load %s: %s", _ENV_VARS_FILE, e)


# ----- Inlined from openvla_loop_helpers -----
def _transform_to_global(
    x: float, y: float, initial_yaw: float
) -> Tuple[float, float]:
    """Transform relative x,y to global frame given initial yaw (degrees)."""
    theta = np.radians(initial_yaw)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    global_x = x * cos_theta - y * sin_theta
    global_y = x * sin_theta + y * cos_theta
    return global_x, global_y


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-180, 180) degrees."""
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def _relative_pose_to_world(
    origin_x: float,
    origin_y: float,
    origin_z: float,
    origin_yaw: float,
    relative_pose: List[float],
) -> Tuple[float, float, float, float]:
    """Convert relative pose [x, y, z, yaw_deg] to world (x, y, z, yaw_deg) given origin."""
    rx, ry, rz, yaw_deg = relative_pose[0], relative_pose[1], relative_pose[2], relative_pose[3]
    gx, gy = _transform_to_global(rx, ry, origin_yaw)
    world_x = origin_x + gx
    world_y = origin_y + gy
    world_z = origin_z + rz
    world_yaw = _normalize_angle(yaw_deg + origin_yaw)
    return (world_x, world_y, world_z, world_yaw)


def _set_cam_at_drone_and_get_image(env: Any, cam_id: int) -> np.ndarray:
    """Position the given camera at the drone and capture from it."""
    x, y, z = env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0])
    roll, yaw, pitch = env.unwrapped.unrealcv.get_obj_rotation(env.unwrapped.player_list[0])
    env.unwrapped.unrealcv.set_cam(cam_id, [x, y, z], [roll, pitch, yaw])
    return env.unwrapped.unrealcv.get_image(cam_id, "lit")


def _set_drone_cam_and_get_image(env: Any, cam_id: Optional[int] = None) -> np.ndarray:
    """Position the drone POV camera at the drone and capture from it (used for OpenVLA input and saved frames)."""
    return _set_cam_at_drone_and_get_image(env, cam_id if cam_id is not None else DRONE_CAM_ID)


def _interactive_camera_select(
    env: Any,
    initial_pos: List[float],
    batch: Any,
) -> int:
    """
    Stop before the main loop: cycle through each camera with the drone at initial_pos,
    show the view, let user choose which camera to use for OpenVLA and saving.
    Returns the selected camera ID.
    """
    try:
        import cv2
    except ImportError:
        logger.error("--camera-select requires opencv-python (cv2). Install with: pip install opencv-python")
        sys.exit(1)

    initial_pos = _normalize_initial_pos(initial_pos)
    env.unwrapped.unrealcv.set_obj_location(
        env.unwrapped.player_list[0], initial_pos[0:3]
    )
    env.unwrapped.unrealcv.set_rotation(
        env.unwrapped.player_list[0], initial_pos[4] - 180
    )
    batch.set_cam(env)
    time.sleep(batch.SLEEP_AFTER_RESET_S)

    n_cams = env.unwrapped.unrealcv.get_camera_num()
    if n_cams <= 0:
        logger.error("No cameras available.")
        sys.exit(1)

    current = 0
    window_name = "Camera select: n=next, p=prev, s or Enter=select, q or Esc=quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        img = _set_cam_at_drone_and_get_image(env, current)
        if img is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        # OpenCV uses BGR; image from Unreal may be RGB
        if img.ndim == 3 and img.shape[2] == 3:
            display = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            display = img
        label = "Camera %d / %d" % (current, n_cams)
        cv2.putText(
            display, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow(window_name, display)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("s") or k == 13 or k == 10:  # s or Enter
            cv2.destroyAllWindows()
            logger.info("Selected camera %d for OpenVLA and saving.", current)
            return current
        if k == ord("q") or k == 27:  # q or Esc
            cv2.destroyAllWindows()
            logger.info("Camera selection cancelled.")
            sys.exit(0)
        if k == ord("n"):
            current = (current + 1) % n_cams
        elif k == ord("p"):
            current = (current - 1) % n_cams


def _apply_action_poses(
    env: Any,
    action_poses: List[Any],
    initial_x: float,
    initial_y: float,
    initial_z: float,
    initial_yaw: float,
    set_cam: Callable[[Any], None],
    trajectory_log: List[Dict[str, Any]],
    sleep_s: float = 0.1,
    drone_cam_id: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], List[float], int]:
    """
    Apply action poses in env, update trajectory_log.
    Returns (image_after_last_pose, current_pose, steps_applied).
    current_pose is [x, y, z, yaw_deg] relative to initial (UAV-Flow format for OpenVLA).
    """
    current_pose: List[float] = [0.0, 0.0, 0.0, 0.0]
    image = None
    steps = 0

    for i, action_pose in enumerate(action_poses):
        if not (
            isinstance(action_pose, (list, tuple)) and len(action_pose) >= 4
        ):
            continue
        relative_x = float(action_pose[0])
        relative_y = float(action_pose[1])
        relative_z = float(action_pose[2])
        relative_yaw_rad = float(action_pose[3])
        relative_yaw = float(np.degrees(relative_yaw_rad))
        relative_yaw = (relative_yaw + 180) % 360 - 180

        global_x, global_y = _transform_to_global(
            relative_x, relative_y, initial_yaw
        )
        absolute_yaw = _normalize_angle(relative_yaw + initial_yaw)
        absolute_pos = [
            global_x + initial_x,
            global_y + initial_y,
            relative_z + initial_z,
            absolute_yaw,
        ]

        env.unwrapped.unrealcv.set_obj_location(
            env.unwrapped.player_list[0], absolute_pos[:3]
        )
        env.unwrapped.unrealcv.set_rotation(
            env.unwrapped.player_list[0], absolute_pos[3] - 180
        )
        set_cam(env)

        current_pose = [relative_x, relative_y, relative_z, relative_yaw]
        steps += 1
        trajectory_log.append({
            "state": [
                [relative_x, relative_y, relative_z],
                [0, relative_yaw, 0],
            ]
        })

        if i == len(action_poses) - 1:
            image = _set_drone_cam_and_get_image(env, drone_cam_id)

        time.sleep(sleep_s)

    return image, current_pose, steps


def _setup_env_and_imports():
    """Same as start_openvla_sim: UnrealEnv, path, gym_unrealcv, monkey-patches."""
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
        """No-op: do not remove agents (e.g. at start of run)."""
        pass

    _base_env.UnrealCv_base.remove_agent = _patched_remove_agent

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


def _import_batch_and_helpers():
    """Add paths, chdir to UAV-Flow-Eval, import batch_run_act_all and repo modules."""
    ai_src = str(_REPO_ROOT / "ai_framework" / "src")
    if ai_src not in sys.path:
        sys.path.insert(0, ai_src)

    os.chdir(str(_UAV_FLOW_EVAL))
    if str(_UAV_FLOW_EVAL) not in sys.path:
        sys.path.insert(0, str(_UAV_FLOW_EVAL))

    import batch_run_act_all as batch

    from modules.goal_monitor import GoalAdherenceMonitor, GoalMonitorResult
    from modules.llm_user_interface import LLM_User_Interface
    from modules.ltl_planner import LTL_Symbolic_Planner

    return batch, LTL_Symbolic_Planner, LLM_User_Interface, GoalAdherenceMonitor, GoalMonitorResult


def _normalize_initial_pos(initial_pos: List[float]) -> List[float]:
    """Ensure initial_pos has at least 5 elements for batch (x,y,z,?,yaw). 4 elements => [x,y,z,yaw] -> [x,y,z,0,yaw]."""
    if len(initial_pos) >= 5:
        return list(initial_pos)
    if len(initial_pos) == 4:
        return [float(initial_pos[0]), float(initial_pos[1]), float(initial_pos[2]), 0.0, float(initial_pos[3])]
    raise ValueError("initial_pos must have 4 or 5 elements (x,y,z,yaw or x,y,z,?,yaw)")


def run_ltl_control_loop(
    initial_pos: List[float],
    env: Any,
    full_instruction: str,
    max_steps: Optional[int],
    trajectory_log: List[Dict[str, Any]],
    server_url: str,
    batch: Any,
    planner: Any,
    goal_monitor: Any,
    reset_model_fn: Any,
    run_dir: Optional[Path] = None,
    subgoals_out: Optional[List[str]] = None,
    goal_adherence_on: bool = False,
    drone_cam_id: Optional[int] = None,
) -> None:
    """
    LTL-aware control loop: plan from full_instruction, run subgoals one at a time,
    verify with goal monitor when OpenVLA reports done, advance or retry.
    If run_dir is set, saves every frame sent to the model under run_dir/frames/.
    """
    full_instruction = full_instruction.strip().lower()
    initial_pos = _normalize_initial_pos(initial_pos)
    initial_x, initial_y, initial_z = initial_pos[0:3]
    initial_yaw = initial_pos[4]
    env.unwrapped.unrealcv.set_obj_location(
        env.unwrapped.player_list[0], initial_pos[0:3]
    )
    env.unwrapped.unrealcv.set_rotation(
        env.unwrapped.player_list[0], initial_pos[4] - 180
    )
    batch.set_cam(env)
    time.sleep(batch.SLEEP_AFTER_RESET_S)
    cam_id = drone_cam_id if drone_cam_id is not None else DRONE_CAM_ID
    image = _set_drone_cam_and_get_image(env, cam_id)

    # Origin for current subgoal: relative pose is (0,0,0,0) at subgoal start so model gets clean state
    subgoal_origin_x, subgoal_origin_y = initial_x, initial_y
    subgoal_origin_z, subgoal_origin_yaw = initial_z, initial_yaw
    current_pose: List[float] = [0.0, 0.0, 0.0, 0.0]
    last_pose = None
    small_count = 0
    step_count = 0
    image_history = deque(maxlen=IMAGE_HISTORY_LEN)
    frame_index = 0
    subgoals_used = subgoals_out if subgoals_out is not None else []

    planner.plan_from_natural_language(full_instruction)
    current_subgoal = planner.get_next_predicate()
    if current_subgoal is None:
        raise RuntimeError(
            "LTL planning produced no subgoals. Fix the instruction or planner (e.g. ensure predicate descriptions are unique)."
        )

    logger.info("Start LTL control loop. First subgoal: %s", current_subgoal)

    while current_subgoal is not None:
        batch.set_cam(env)
        if image is not None:
            image_history.append(image.copy())

        if image is None:
            logger.warning("No image, ending control loop.")
            break

        # Save every frame sent to the model
        if run_dir is not None and image is not None:
            frames_dir = run_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frames_dir / "frame_{:06d}.png".format(frame_index)
            try:
                import cv2
                cv2.imwrite(str(frame_path), image)
            except Exception as e:
                logger.debug("Failed to save frame %s: %s", frame_path, e)
            frame_index += 1

        # proprio is relative to current sub-task start (reset to [0,0,0,0] when subgoal advances)
        response = batch.send_prediction_request(
            image=Image.fromarray(image),
            proprio=_state_for_openvla(current_pose),
            instr=(current_subgoal or "").lower(),
            server_url=server_url,
        )

        if response is None:
            logger.warning("No valid response, ending control.")
            break

        action_poses = response.get("action")
        if not isinstance(action_poses, list) or len(action_poses) == 0:
            logger.warning("Response 'action' empty or invalid, stopping.")
            break

        try:
            new_image, current_pose, steps_added = _apply_action_poses(
                env,
                action_poses,
                subgoal_origin_x,
                subgoal_origin_y,
                subgoal_origin_z,
                subgoal_origin_yaw,
                batch.set_cam,
                trajectory_log,
                sleep_s=0.1,
                drone_cam_id=cam_id,
            )
        except Exception as e:
            logger.error("Error executing action: %s", e)
            break

        step_count += steps_added
        if new_image is not None:
            image = new_image

        pose_now = current_pose
        advanced_this_iteration = False
        if last_pose is not None:
            diffs = [abs(a - b) for a, b in zip(pose_now, last_pose)]
            if (
                all(d < SMALL_DELTA_POS for d in diffs[:3])
                and diffs[3] < SMALL_DELTA_YAW
            ):
                small_count += 1
            else:
                small_count = 0
            if small_count >= batch.ACTION_SMALL_STEPS:
                logger.info(
                    "Detected %d steps with very small change, ending current subgoal.",
                    batch.ACTION_SMALL_STEPS,
                )
                subgoals_used.append(current_subgoal)
                planner.advance_state(current_subgoal)
                current_subgoal = planner.get_next_predicate()
                reset_model_fn(server_url)
                # Reset position state so model gets clean start for next subgoal
                subgoal_origin_x, subgoal_origin_y, subgoal_origin_z, subgoal_origin_yaw = _relative_pose_to_world(
                    subgoal_origin_x, subgoal_origin_y, subgoal_origin_z, subgoal_origin_yaw, current_pose
                )
                current_pose = [0.0, 0.0, 0.0, 0.0]
                small_count = 0
                step_count = 0
                advanced_this_iteration = True
                if current_subgoal is None:
                    logger.info("No more subgoals. Task complete.")
                    break
                logger.info("Next subgoal: %s", current_subgoal)
        last_pose = pose_now

        if not advanced_this_iteration and max_steps is not None and step_count >= max_steps:
            logger.info("Reached max_steps %d for current subgoal, advancing.", max_steps)
            subgoals_used.append(current_subgoal)
            planner.advance_state(current_subgoal)
            current_subgoal = planner.get_next_predicate()
            reset_model_fn(server_url)
            subgoal_origin_x, subgoal_origin_y, subgoal_origin_z, subgoal_origin_yaw = _relative_pose_to_world(
                subgoal_origin_x, subgoal_origin_y, subgoal_origin_z, subgoal_origin_yaw, current_pose
            )
            current_pose = [0.0, 0.0, 0.0, 0.0]
            small_count = 0
            step_count = 0
            advanced_this_iteration = True
            if current_subgoal is None:
                logger.info("No more subgoals. Task complete.")
                break
            logger.info("Next subgoal: %s", current_subgoal)

        # Optional periodic goal check (only when goal adherence is on); skip if we already advanced
        if not advanced_this_iteration and goal_adherence_on and step_count % GOAL_MONITOR_PERIODIC_STEPS == 0 and step_count > 0:
            result = goal_monitor.check(
                list(image_history),
                current_subgoal,
                full_goal=full_instruction,
                model_claimed_done=False,
            )
            if result.goal_achieved:
                logger.info("Periodic check: full goal achieved.")
                break

        # When OpenVLA reports done: verify with goal monitor if on, else advance (skip if we already advanced this iteration)
        if not advanced_this_iteration and response.get("done") is True:
            if goal_adherence_on:
                result = goal_monitor.check(
                    list(image_history),
                    current_subgoal,
                    full_goal=full_instruction,
                    model_claimed_done=True,
                )
                if result.subgoal_achieved:
                    logger.info("Subgoal verified: %s. Advancing planner.", current_subgoal)
                    subgoals_used.append(current_subgoal)
                    planner.advance_state(current_subgoal)
                    current_subgoal = planner.get_next_predicate()
                    reset_model_fn(server_url)
                    subgoal_origin_x, subgoal_origin_y, subgoal_origin_z, subgoal_origin_yaw = _relative_pose_to_world(
                        subgoal_origin_x, subgoal_origin_y, subgoal_origin_z, subgoal_origin_yaw, current_pose
                    )
                    current_pose = [0.0, 0.0, 0.0, 0.0]
                    step_count = 0
                    if current_subgoal is None:
                        logger.info("No more subgoals. Task complete.")
                        break
                else:
                    if result.suggest_retry:
                        logger.info(
                            "Goal monitor: subgoal not achieved, suggest retry. Continuing with same subgoal."
                        )
                    # Continue loop with same current_subgoal (retry)
            else:
                # Goal adherence off: trust model and advance
                logger.info("Model reported done. Advancing planner (goal adherence off).")
                subgoals_used.append(current_subgoal)
                planner.advance_state(current_subgoal)
                current_subgoal = planner.get_next_predicate()
                reset_model_fn(server_url)
                subgoal_origin_x, subgoal_origin_y, subgoal_origin_z, subgoal_origin_yaw = _relative_pose_to_world(
                    subgoal_origin_x, subgoal_origin_y, subgoal_origin_z, subgoal_origin_yaw, current_pose
                )
                current_pose = [0.0, 0.0, 0.0, 0.0]
                step_count = 0
                if current_subgoal is None:
                    logger.info("No more subgoals. Task complete.")
                    break



def _parse_initial_position(s: str) -> List[float]:
    """Parse --initial-position comma-separated string into 4 floats (x,y,z,yaw). Supports negative numbers (e.g. '-600, -1270, 128, 61')."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(
            "--initial-position must be 4 comma-separated numbers: x,y,z,yaw (e.g. -600, -1270, 128, 61)"
        )
    return [float(x) for x in parts]


def _load_task_from_json(path: Path) -> Dict[str, Any]:
    """Load ltl task JSON: instruction and initial_pos (4 or 5 elements)."""
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Task JSON must be an object")
    instruction = data.get("instruction", "").strip().lower()
    initial_pos = data.get("initial_pos")
    if not instruction:
        raise ValueError("Task JSON must have 'instruction'")
    if not initial_pos or not isinstance(initial_pos, (list, tuple)) or len(initial_pos) < 4:
        raise ValueError("Task JSON must have 'initial_pos' with 4 or 5 numbers (x,y,z,yaw)")
    return {"instruction": instruction, "initial_pos": [float(x) for x in initial_pos]}


def _resolve_tasks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Build list of tasks from -c, --task, or --run_all_tasks (exactly one)."""
    cmd = getattr(args, "command", None)
    task_file = getattr(args, "task", None)
    run_all = getattr(args, "run_all_tasks", False)

    count = sum([1 if cmd else 0, 1 if task_file else 0, 1 if run_all else 0])
    if count != 1:
        raise SystemExit(
            "Exactly one of -c/--command, --task, or --run_all_tasks is required.\n"
            "  -c \"instruction\" [--initial-position x,y,z,yaw]  run one ad-hoc task (position optional, default: -600,-1270,128,61)\n"
            "  --task first_task.json                           run one task from tasks/ltl_tasks/\n"
            "  --run_all_tasks                                  run all JSONs in tasks/ltl_tasks/"
        )

    if cmd is not None:
        initial_pos_str = getattr(args, "initial_position", None) or DEFAULT_INITIAL_POSITION
        return [{"instruction": cmd.strip().lower(), "initial_pos": _parse_initial_position(initial_pos_str)}]

    if task_file is not None:
        path = Path(task_file)
        if not path.is_absolute():
            path = LTL_TASKS_DIR / path.name
        if not path.exists():
            raise SystemExit("Task file not found: {}".format(path))
        return [_load_task_from_json(path)]

    # run_all_tasks
    LTL_TASKS_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(glob.glob(str(LTL_TASKS_DIR / "*.json")))
    if not json_files:
        raise SystemExit("No JSON files found in {}".format(LTL_TASKS_DIR))
    tasks = []
    for jf in json_files:
        try:
            tasks.append(_load_task_from_json(Path(jf)))
        except Exception as e:
            logger.warning("Skipping %s: %s", jf, e)
    return tasks


def main():
    _load_env_vars()
    parser = argparse.ArgumentParser(
        description="Run OpenVLA with LTL planning and goal adherence monitoring"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "-c",
        "--command",
        type=str,
        default=None,
        help="Natural language instruction (optional --initial-position, default: -600,-1270,128,61)",
    )
    mode.add_argument(
        "--task",
        type=str,
        default=None,
        metavar="TASK.json",
        help="Run single task from tasks/ltl_tasks/ (e.g. first_task.json)",
    )
    mode.add_argument(
        "--run_all_tasks",
        action="store_true",
        help="Run all JSON tasks in tasks/ltl_tasks/",
    )
    parser.add_argument(
        "--initial-position",
        type=str,
        default=DEFAULT_INITIAL_POSITION,
        metavar="x,y,z,yaw",
        help="Initial position as comma-separated x,y,z,yaw (supports negatives, e.g. '-600, -1270, 128, 61'). Default: %(default)s",
    )
    parser.add_argument(
        "-e",
        "--env_id",
        default=_DOWNTOWN_ENV_ID,
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
        "-o",
        "--results_dir",
        default=str(LTL_RESULTS_DIR),
        help="Base directory for run outputs (default: results/ltl_results)",
    )
    parser.add_argument(
        "-p",
        "--server_port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="OpenVLA server port",
    )
    parser.add_argument(
        "-m",
        "--max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Max inference steps per task",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--llm_model",
        default="gpt-4o-mini",
        help="LLM for LTL decomposition (and goal monitor if not set)",
    )
    parser.add_argument(
        "--goal_monitor_model",
        default=None,
        help="VLM for goal monitor (default: same as --llm_model)",
    )
    parser.add_argument(
        "--goal-adherence-on",
        dest="goal_adherence_on",
        action="store_true",
        default=False,
        help="Enable goal adherence monitor (off by default). When on, verifies subgoals when model says done and runs periodic full-goal checks.",
    )
    parser.add_argument(
        "--camera-select",
        action="store_true",
        help="Stop before the main loop; cycle through each camera and select which one to use for OpenVLA and saving to disk (requires display and opencv-python).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    tasks = _resolve_tasks(args)

    if not _BATCH_SCRIPT.exists():
        logger.error("batch_run_act_all.py not found at %s", _BATCH_SCRIPT)
        sys.exit(1)

    _setup_env_and_imports()
    (
        batch,
        LTL_Symbolic_Planner,
        LLM_User_Interface,
        GoalAdherenceMonitor,
        _,
    ) = _import_batch_and_helpers()

    os.chdir(str(_UAV_FLOW_EVAL))
    server_url = "http://127.0.0.1:{}".format(args.server_port) + "/predict"
    results_base = Path(args.results_dir)
    results_base.mkdir(parents=True, exist_ok=True)

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

    llm_interface = LLM_User_Interface(model=args.llm_model)
    planner = LTL_Symbolic_Planner(llm_interface)
    goal_monitor_model = args.goal_monitor_model or args.llm_model
    goal_monitor = GoalAdherenceMonitor(model=goal_monitor_model)

    # Optional: stop before main loop and interactively select which camera to use
    drone_cam_id = DRONE_CAM_ID
    if args.camera_select:
        logger.info("Camera selection: stopping before main loop. Use the window to pick the camera for OpenVLA and saving.")
        drone_cam_id = _interactive_camera_select(env, tasks[0]["initial_pos"], batch)

    for idx, task in enumerate(tasks):
        instruction = task["instruction"]
        initial_pos = task["initial_pos"]
        start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_name = "run_{}".format(start_time) if len(tasks) == 1 else "run_{}_{:02d}".format(start_time, idx)
        run_dir = results_base / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "\n===== Task %d/%d =====",
            idx + 1,
            len(tasks),
        )
        batch.reset_model(server_url)
        logger.info("instruction: %s", instruction)

        trajectory_log = []
        subgoals_used = []

        run_ltl_control_loop(
            initial_pos,
            env,
            instruction,
            args.max_steps,
            trajectory_log,
            server_url,
            batch,
            planner,
            goal_monitor,
            batch.reset_model,
            run_dir=run_dir,
            subgoals_out=subgoals_used,
            goal_adherence_on=args.goal_adherence_on,
            drone_cam_id=drone_cam_id,
        )

        # Write trajectory_log.json and run_info.json
        with open(run_dir / "trajectory_log.json", "w") as f:
            json.dump(trajectory_log, f, indent=2)
        end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        run_info = {
            "instruction": instruction,
            "initial_pos": initial_pos,
            "start_time": start_time,
            "end_time": end_time,
            "subgoals": subgoals_used,
        }
        with open(run_dir / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)

        logger.info("Run saved to %s", run_dir)
        logger.info("===== Task %d finished =====\n", idx + 1)

    env.close()


if __name__ == "__main__":
    main()
