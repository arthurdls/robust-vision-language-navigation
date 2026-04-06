"""
Shared simulation utilities for OpenVLA control loops (LTL, REPL, goal adherence).

Extracted from run_ltl_planner.py and run_repl.py to eliminate duplication.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ─── Path constants ───────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_VARS_FILE = REPO_ROOT / "ai_framework" / ".env_vars"
UAV_FLOW_EVAL = REPO_ROOT / "UAV-Flow" / "UAV-Flow-Eval"
BATCH_SCRIPT = UAV_FLOW_EVAL / "batch_run_act_all.py"
UAV_FLOW_ENVS_OVERLAY = REPO_ROOT / "config" / "uav_flow_envs"
DOWNTOWN_OVERLAY_JSON = UAV_FLOW_ENVS_OVERLAY / "Track" / "DowntownWest.json"
DOWNTOWN_ENV_ID = "UnrealTrack-DowntownWest-ContinuousColor-v0"

# ─── Sim / server defaults ───────────────────────────────────────────────────

DEFAULT_SERVER_PORT = 5007
DEFAULT_TIME_DILATION = 10
DEFAULT_SEED = 0
DEFAULT_INITIAL_POSITION = "-600,-1270,128,61"
DRONE_CAM_ID = 5
PROPRIO_LEN = 4

logger = logging.getLogger(__name__)


# ─── Environment variable loading ────────────────────────────────────────────

def load_env_vars() -> None:
    """Load ai_framework/.env_vars (export KEY=value lines) into os.environ so API keys are set."""
    if not ENV_VARS_FILE.exists():
        return
    try:
        with open(ENV_VARS_FILE, "r") as f:
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
        logger.warning("Could not load %s: %s", ENV_VARS_FILE, e)


# ─── Coordinate transforms ───────────────────────────────────────────────────

def transform_to_global(
    x: float, y: float, initial_yaw: float
) -> Tuple[float, float]:
    """Transform relative x,y to global frame given initial yaw (degrees)."""
    theta = np.radians(initial_yaw)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    global_x = x * cos_theta - y * sin_theta
    global_y = x * sin_theta + y * cos_theta
    return global_x, global_y


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-180, 180) degrees."""
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def normalize_initial_pos(initial_pos: List[float]) -> List[float]:
    """Ensure initial_pos has at least 5 elements for batch (x,y,z,?,yaw).

    4 elements [x,y,z,yaw] -> [x,y,z,0,yaw].
    """
    if len(initial_pos) >= 5:
        return list(initial_pos)
    if len(initial_pos) == 4:
        return [float(initial_pos[0]), float(initial_pos[1]), float(initial_pos[2]),
                0.0, float(initial_pos[3])]
    raise ValueError("initial_pos must have 4 or 5 elements (x,y,z,yaw or x,y,z,?,yaw)")


def relative_pose_to_world(
    origin_x: float,
    origin_y: float,
    origin_z: float,
    origin_yaw: float,
    relative_pose: List[float],
) -> Tuple[float, float, float, float]:
    """Convert relative pose [x, y, z, yaw_deg] to world (x, y, z, yaw_deg) given origin."""
    rx, ry, rz, yaw_deg = (relative_pose[0], relative_pose[1],
                            relative_pose[2], relative_pose[3])
    gx, gy = transform_to_global(rx, ry, origin_yaw)
    world_x = origin_x + gx
    world_y = origin_y + gy
    world_z = origin_z + rz
    world_yaw = normalize_angle(yaw_deg + origin_yaw)
    return (world_x, world_y, world_z, world_yaw)


# ─── OpenVLA state formatting ────────────────────────────────────────────────

def verify_proprio_format(proprio: np.ndarray) -> None:
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
    logger.debug(
        "proprio format OK: [x=%.2f, y=%.2f, z=%.2f, yaw_deg=%.2f]",
        float(proprio[0]), float(proprio[1]), float(proprio[2]), float(proprio[3]),
    )


def state_for_openvla(current_pose: List[float]) -> np.ndarray:
    """Format current state for OpenVLA /predict payload.

    Returns a 4-float array [x, y, z, yaw_degrees] with format verification.
    """
    if len(current_pose) >= PROPRIO_LEN:
        out = [float(current_pose[i]) for i in range(PROPRIO_LEN)]
    else:
        out = [0.0] * PROPRIO_LEN
        for i in range(min(len(current_pose), PROPRIO_LEN)):
            out[i] = float(current_pose[i])
    arr = np.array(out, dtype=np.float32)
    verify_proprio_format(arr)
    return arr


# ─── Sim environment bootstrapping ───────────────────────────────────────────

def setup_env_and_imports() -> None:
    """Set UnrealEnv, add paths, import gym_unrealcv, and apply monkey-patches."""
    os.environ.setdefault("UnrealEnv", str(REPO_ROOT / "envs"))
    uav_eval_str = str(UAV_FLOW_EVAL)
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
        if filename == "Track/DowntownWest.json" and DOWNTOWN_OVERLAY_JSON.exists():
            return str(DOWNTOWN_OVERLAY_JSON)
        return _original_get_settingpath(filename)

    _misc.get_settingpath = _get_settingpath

    import gym_unrealcv.envs.base_env as _base_env

    def _patched_remove_agent(self, name):
        """Update env state so set_population() loop terminates; skip Unreal destroy_obj."""
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


def import_batch_module() -> Any:
    """Add ai_framework/src to path, chdir to UAV-Flow-Eval, import and return batch_run_act_all."""
    ai_src = str(REPO_ROOT / "ai_framework" / "src")
    if ai_src not in sys.path:
        sys.path.insert(0, ai_src)

    os.chdir(str(UAV_FLOW_EVAL))
    if str(UAV_FLOW_EVAL) not in sys.path:
        sys.path.insert(0, str(UAV_FLOW_EVAL))

    import batch_run_act_all as batch
    return batch


def setup_sim_env(env_id: str, time_dilation_val: int, seed: int, batch: Any) -> Any:
    """Create gym environment with wrappers, reset, and spawn NPCs. Returns wrapped env."""
    import gym
    from gym_unrealcv.envs.wrappers import time_dilation, configUE, augmentation

    env = gym.make(env_id)
    if time_dilation_val > 0:
        env = time_dilation.TimeDilationWrapper(env, time_dilation_val)
    env.unwrapped.agents_category = ["drone"]
    env = configUE.ConfigUEWrapper(env, resolution=(256, 256))
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env.seed(seed)
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

    return env


# ─── Camera helpers ───────────────────────────────────────────────────────────

def set_drone_cam_and_get_image(env: Any, cam_id: Optional[int] = None) -> Optional[np.ndarray]:
    """Position the drone POV camera at the drone and capture an image.

    Retries once on TypeError (UnrealCV sometimes returns text instead of bytes).
    Returns None if capture fails after retries.
    """
    cam = cam_id if cam_id is not None else DRONE_CAM_ID
    x, y, z = env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0])
    roll, yaw, pitch = env.unwrapped.unrealcv.get_obj_rotation(env.unwrapped.player_list[0])
    env.unwrapped.unrealcv.set_cam(cam, [x, y, z], [roll, pitch, yaw])
    for attempt in range(2):
        try:
            return env.unwrapped.unrealcv.get_image(cam, "lit")
        except TypeError:
            logger.warning(
                "get_image returned invalid data (attempt %d/2). Retrying after short delay.",
                attempt + 1,
            )
            time.sleep(0.5)
    logger.warning("get_image failed after retries (simulator may have sent non-binary response).")
    return None


def interactive_camera_select(env: Any, initial_pos: List[float], batch: Any) -> int:
    """Cycle through cameras with the drone at initial_pos; let user choose.

    Returns the selected camera ID.
    """
    try:
        import cv2
    except ImportError:
        logger.error("Camera select requires opencv-python (cv2). Install with: pip install opencv-python")
        sys.exit(1)

    initial_pos = normalize_initial_pos(initial_pos)
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
        img = set_drone_cam_and_get_image(env, current)
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


# ─── Action execution ────────────────────────────────────────────────────────

def apply_action_poses(
    env: Any,
    action_poses: List[Any],
    initial_x: float,
    initial_y: float,
    initial_z: float,
    initial_yaw: float,
    set_cam: Callable[[Any], None],
    trajectory_log: Optional[List[Dict[str, Any]]] = None,
    sleep_s: float = 0.1,
    drone_cam_id: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], List[float], int]:
    """Apply action poses in env.

    Returns (image_after_last_pose, current_pose, steps_applied).
    current_pose is [x, y, z, yaw_deg] relative to initial (UAV-Flow format for OpenVLA).
    If trajectory_log is provided, appends one entry per valid pose.
    """
    current_pose: List[float] = [0.0, 0.0, 0.0, 0.0]
    image = None
    steps = 0

    for i, action_pose in enumerate(action_poses):
        if not (isinstance(action_pose, (list, tuple)) and len(action_pose) >= 4):
            continue
        relative_x = float(action_pose[0])
        relative_y = float(action_pose[1])
        relative_z = float(action_pose[2])
        relative_yaw_rad = float(action_pose[3])
        relative_yaw = float(np.degrees(relative_yaw_rad))
        relative_yaw = (relative_yaw + 180) % 360 - 180

        global_x, global_y = transform_to_global(
            relative_x, relative_y, initial_yaw
        )
        absolute_yaw = normalize_angle(relative_yaw + initial_yaw)
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
        if trajectory_log is not None:
            trajectory_log.append({
                "state": [
                    [relative_x, relative_y, relative_z],
                    [0, relative_yaw, 0],
                ]
            })

        if i == len(action_poses) - 1:
            image = set_drone_cam_and_get_image(env, drone_cam_id)

        time.sleep(sleep_s)

    return image, current_pose, steps


# ─── Argument helpers ─────────────────────────────────────────────────────────

def parse_position(s: str) -> List[float]:
    """Parse comma-separated 'x,y,z,yaw' string into 4 floats. Supports negative numbers."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(
            "Position must be 4 comma-separated numbers: x,y,z,yaw "
            "(e.g. -600,-1270,128,61)"
        )
    return [float(x) for x in parts]
