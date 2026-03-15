#!/usr/bin/env python3
"""
OpenVLA REPL: same simulation setup as run_openvla_ltl, then an interactive REPL.

- After the simulator initializes, you are prompted for the initial drone position (x,y,z,yaw).
- You can then type natural-language actions; each is sent directly to the VLA (no LTL planning).
- An action runs until the model predicts small changes 10 steps in a row, or you press Ctrl+C to abort.
- Relative position fed to the model is reset only at the start of each action (first step gets 0,0,0,0) and accumulates until the action ends.
- When the action finishes, "ACTION COMPLETE" is printed (or "Action aborted." if Ctrl+C), and the REPL continues.
- 'where' (or 'location', 'pos', 'pose') prints current position/orientation (x, y, z, yaw).
- 'pos x,y,z,yaw' or 'teleport x,y,z,yaw' teleports the drone to that proprio at any time.
- 'undo' (or 'u') restores the drone to the position before the last VLA action; multiple undos in a row are supported.
- 'record' or 'record start' starts saving every frame sent to the VLA under results/repl_results/run_YYYY_MM_DD_HH_MM_SS/; 'record stop' stops and reports how many frames were saved.
- Up/Down arrow keys browse through previous actions (history saved to ~/.rvln_openvla_repl_history).
- The simulator is not closed when the script exits (e.g. on quit).

OpenVLA server must be running: python scripts/start_openvla_server.py

Usage (from repo root):
  python scripts/run_openvla_repl.py
  python scripts/run_openvla_repl.py -p 5007
  python scripts/run_openvla_repl.py --use-default-cam   # skip camera selection
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Enable readline for up/down arrow history (Unix; no-op on Windows if readline not installed)
try:
    import readline  # noqa: F401
except ImportError:
    pass

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_VARS_FILE = _REPO_ROOT / "ai_framework" / ".env_vars"
_UAV_FLOW_EVAL = _REPO_ROOT / "UAV-Flow" / "UAV-Flow-Eval"
_BATCH_SCRIPT = _UAV_FLOW_EVAL / "batch_run_act_all.py"
_UAV_FLOW_ENVS_OVERLAY = _REPO_ROOT / "config" / "uav_flow_envs"
_DOWNTOWN_OVERLAY_JSON = _UAV_FLOW_ENVS_OVERLAY / "Track" / "DowntownWest.json"
_DOWNTOWN_ENV_ID = "UnrealTrack-DowntownWest-ContinuousColor-v0"

DEFAULT_SERVER_PORT = 5007
DEFAULT_TIME_DILATION = 10
DEFAULT_SEED = 0
DEFAULT_INITIAL_POSITION = "-600,-1270,128,61"
PROPRIO_LEN = 4
DRONE_CAM_ID = 5

# REPL history file (for up/down arrow browsing)
_REPL_HISTORY_PATH = Path.home() / ".rvln_openvla_repl_history"
# Directory for REPL recording outputs (frames sent to VLA)
_REPL_RESULTS_DIR = _REPO_ROOT / "results" / "repl_results"

# Recording state: when "dir" is not None, frames are saved there (frame_000000.png, ...)
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


def _state_for_openvla(current_pose: List[float]) -> np.ndarray:
    """Format current state for OpenVLA /predict payload: [x, y, z, yaw_degrees]."""
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
    return np.array(out, dtype=np.float32)


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
    except Exception as e:
        logger.warning("Could not load %s: %s", _ENV_VARS_FILE, e)


def _transform_to_global(
    x: float, y: float, initial_yaw: float
) -> tuple:
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


def _normalize_initial_pos(initial_pos: List[float]) -> List[float]:
    """Ensure initial_pos has at least 5 elements for batch (x,y,z,?,yaw). 4 elements => [x,y,z,yaw] -> [x,y,z,0,yaw]."""
    if len(initial_pos) >= 5:
        return list(initial_pos)
    if len(initial_pos) == 4:
        return [float(initial_pos[0]), float(initial_pos[1]), float(initial_pos[2]), 0.0, float(initial_pos[3])]
    raise ValueError("initial_pos must have 4 or 5 elements (x,y,z,yaw or x,y,z,?,yaw)")


def _interactive_camera_select(
    env: Any,
    initial_pos: List[float],
    batch: Any,
) -> int:
    """
    Cycle through each camera with the drone at initial_pos; let user choose which camera to use.
    Returns the selected camera ID.
    """
    try:
        import cv2
    except ImportError:
        logger.error("Camera select requires opencv-python (cv2). Install with: pip install opencv-python")
        sys.exit(1)

    import time
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
        img = _set_drone_cam_and_get_image(env, current)
        if img is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
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
        if k == ord("s") or k == 13 or k == 10:
            cv2.destroyAllWindows()
            logger.info("Selected camera %d for OpenVLA.", current)
            return current
        if k == ord("q") or k == 27:
            cv2.destroyAllWindows()
            logger.info("Camera selection cancelled.")
            sys.exit(0)
        if k == ord("n"):
            current = (current + 1) % n_cams
        elif k == ord("p"):
            current = (current - 1) % n_cams


def _set_drone_cam_and_get_image(env: Any, cam_id: Optional[int] = None) -> Optional[np.ndarray]:
    """Position the drone POV camera at the drone and capture from it. Returns None if capture fails (e.g. UnrealCV returned text instead of image bytes)."""
    import time
    cam = cam_id if cam_id is not None else DRONE_CAM_ID
    x, y, z = env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0])
    roll, yaw, pitch = env.unwrapped.unrealcv.get_obj_rotation(env.unwrapped.player_list[0])
    env.unwrapped.unrealcv.set_cam(cam, [x, y, z], [roll, pitch, yaw])
    for attempt in range(2):
        try:
            return env.unwrapped.unrealcv.get_image(cam, "lit")
        except TypeError:
            # UnrealCV client sometimes returns str instead of bytes (e.g. error message), causing decode_bmp to fail
            logger.warning(
                "get_image returned invalid data (attempt %d/2). Retrying after short delay.",
                attempt + 1,
            )
            time.sleep(0.5)
    logger.warning("get_image failed after retries (simulator may have sent non-binary response).")
    return None


def _apply_action_poses(
    env: Any,
    action_poses: List[Any],
    initial_x: float,
    initial_y: float,
    initial_z: float,
    initial_yaw: float,
    set_cam: Any,
    sleep_s: float = 0.1,
    drone_cam_id: Optional[int] = None,
) -> tuple:
    """
    Apply action poses in env. Returns (image_after_last_pose, current_pose).
    current_pose is [x, y, z, yaw_deg] relative to initial.
    """
    current_pose: List[float] = [0.0, 0.0, 0.0, 0.0]
    image = None

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

        if i == len(action_poses) - 1:
            image = _set_drone_cam_and_get_image(env, drone_cam_id)

        import time
        time.sleep(sleep_s)

    return image, current_pose


def _relative_pose_to_world(
    origin_x: float,
    origin_y: float,
    origin_z: float,
    origin_yaw: float,
    relative_pose: List[float],
) -> tuple:
    """Convert relative pose [x, y, z, yaw_deg] to world (x, y, z, yaw_deg) given origin."""
    rx, ry, rz, yaw_deg = relative_pose[0], relative_pose[1], relative_pose[2], relative_pose[3]
    gx, gy = _transform_to_global(rx, ry, origin_yaw)
    world_x = origin_x + gx
    world_y = origin_y + gy
    world_z = origin_z + rz
    world_yaw = _normalize_angle(yaw_deg + origin_yaw)
    return (world_x, world_y, world_z, world_yaw)


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

    def _patched_remove_agent(self, name):
        agent_index = self.player_list.index(name)
        self.player_list.remove(name)
        self.cam_list = self.remove_cam(name)
        self.action_space.pop(agent_index)
        self.observation_space.pop(agent_index)
        self.agents.pop(name)

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


def _import_batch() -> Any:
    """Add paths, chdir to UAV-Flow-Eval, import batch_run_act_all."""
    ai_src = str(_REPO_ROOT / "ai_framework" / "src")
    if ai_src not in sys.path:
        sys.path.insert(0, ai_src)

    os.chdir(str(_UAV_FLOW_EVAL))
    if str(_UAV_FLOW_EVAL) not in sys.path:
        sys.path.insert(0, str(_UAV_FLOW_EVAL))

    import batch_run_act_all as batch
    return batch


def _parse_position(s: str) -> List[float]:
    """Parse 'x,y,z,yaw' into 4 floats. Supports negative numbers."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("Position must be 4 comma-separated numbers: x,y,z,yaw")
    return [float(x) for x in parts]


def _get_current_pose(env: Any) -> List[float]:
    """Read current drone position and yaw from the simulator. Returns [x, y, z, yaw_deg]."""
    x, y, z = env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0])
    roll, yaw, pitch = env.unwrapped.unrealcv.get_obj_rotation(env.unwrapped.player_list[0])
    # Sim returns yaw in same convention as we display; normalize to [-180, 180)
    yaw_deg = _normalize_angle(float(yaw))
    return [float(x), float(y), float(z), yaw_deg]


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
    One VLA step: get image, send to server, apply poses. Origin is fixed for the whole action.
    Returns (new_image, new_relative_pose) or (None, current_pose) on error.
    """
    batch.set_cam(env)
    image = _set_drone_cam_and_get_image(env, drone_cam_id)
    if image is None:
        logger.warning("No image from drone camera.")
        return None, current_pose

    # Save frame to recording dir when recording is active
    rec_dir = _recording_state.get("dir")
    if rec_dir is not None and image is not None:
        try:
            import cv2
            frame_path = rec_dir / "frame_{:06d}.png".format(_recording_state["frame_index"])
            cv2.imwrite(str(frame_path), image)
            _recording_state["frame_index"] += 1
        except Exception as e:
            logger.debug("Failed to save recorded frame: %s", e)

    proprio = _state_for_openvla(current_pose)
    response = batch.send_prediction_request(
        image=Image.fromarray(image),
        proprio=proprio,
        instr=instruction.strip().lower(),
        server_url=server_url,
    )

    if response is None:
        logger.warning("No valid response from VLA.")
        return image, current_pose

    action_poses = response.get("action")
    if not _valid_action_poses(action_poses):
        logger.warning("Response 'action' empty or invalid (expected list of [x,y,z,yaw] poses).")
        return image, current_pose

    try:
        new_image, new_pose = _apply_action_poses(
            env,
            action_poses,
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
    Relative pose is reset only at the start (first step gets [0,0,0,0]) and accumulates until the action ends.
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

        # Check for small consecutive changes (same as LTL / batch_run_act_all)
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
    import time
    for _ in range(3):
        try:
            batch.set_cam(env)
            _set_drone_cam_and_get_image(env, drone_cam_id)
        except Exception:
            pass
        time.sleep(0.2)


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


def main() -> None:
    _load_env_vars()
    parser = argparse.ArgumentParser(
        description="OpenVLA REPL: same sim as run_openvla_ltl, then interactive actions"
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

    if not _BATCH_SCRIPT.exists():
        logger.error("batch_run_act_all.py not found at %s", _BATCH_SCRIPT)
        sys.exit(1)

    _setup_env_and_imports()
    batch = _import_batch()
    # Force a new TCP connection per request so Ctrl+C doesn't leave the HTTP stream out of sync
    import requests
    _original_requests_post = requests.post
    def _repl_requests_post(*args, **kwargs):
        h = dict(kwargs.get("headers") or {})
        h["Connection"] = "close"
        kwargs["headers"] = h
        return _original_requests_post(*args, **kwargs)
    requests.post = _repl_requests_post

    server_url = "http://127.0.0.1:{}".format(args.server_port) + "/predict"

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

    import time
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
        pos = _parse_position(line)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)

    # pos is [x, y, z, yaw]
    env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[0], pos[0:3])
    env.unwrapped.unrealcv.set_rotation(env.unwrapped.player_list[0], pos[3] - 180)
    batch.set_cam(env)
    time.sleep(batch.SLEEP_AFTER_RESET_S)

    # By default run interactive camera select; skip when --use-default-cam
    if args.use_default_cam:
        drone_cam_id = DRONE_CAM_ID
    else:
        logger.info("Camera selection: use the window to pick the camera for OpenVLA.")
        drone_cam_id = _interactive_camera_select(env, pos, batch)

    origin_x, origin_y, origin_z, origin_yaw = pos[0], pos[1], pos[2], pos[3]
    current_pose: List[float] = [0.0, 0.0, 0.0, 0.0]
    # Stack of (x, y, z, yaw) at the start of each VLA action; pop on undo
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

            # Start recording: save frames sent to VLA under results/repl_results/run_YYYY_MM_DD_HH_MM_SS/
            if line.lower() in ("record", "record start", "start record"):
                _REPL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                run_name = "run_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                run_dir = _REPL_RESULTS_DIR / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                _recording_state["dir"] = run_dir
                _recording_state["frame_index"] = 0
                print("Recording started. Frames will be saved to: {}".format(run_dir))
                continue

            # Stop recording
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

            # Current position/orientation
            if line.lower() in ("where", "location", "pos", "pose"):
                pose = _get_current_pose(env)
                print("x={}, y={}, z={}, yaw={}".format(pose[0], pose[1], pose[2], pose[3]))
                continue

            # Undo: restore position to before the last VLA action (can undo multiple times)
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

            # Teleport: pos x,y,z,yaw or teleport x,y,z,yaw
            if line.lower().startswith("pos ") or line.lower().startswith("teleport "):
                prefix = "pos " if line.lower().startswith("pos ") else "teleport "
                rest = line[len(prefix):].strip()
                try:
                    pos = _parse_position(rest)
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

            # Run VLA action until small changes 10 steps in a row, or Ctrl+C
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
            # Next action's origin = current world pose (so relative pose resets at start of next action)
            origin_x, origin_y, origin_z, origin_yaw = _relative_pose_to_world(
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
