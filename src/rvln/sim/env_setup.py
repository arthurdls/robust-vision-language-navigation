"""
Shared simulation utilities for OpenVLA control loops (LTL, REPL, goal adherence).

Client scripts talk to the sim API server via SimClient. The server manages
the gym/UnrealCV environment locally. This module provides the SimClient-based
helpers that all control scripts use.

setup_env_and_imports() is only needed on the server side (and by scout_locations).
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rvln.paths import (
    REPO_ROOT,
    SCENES_DIR,
    ENVS_DIR,
    DEFAULT_SERVER_PORT,
    DEFAULT_TIME_DILATION,
    DEFAULT_SEED,
    PROPRIO_LEN,
    load_env_vars,
)
from rvln.sim.transforms import (  # noqa: F401 -- re-exported for existing callers
    transform_to_global,
    normalize_angle,
    normalize_initial_pos,
    relative_pose_to_world,
    parse_position,
)
from rvln.config import DEFAULT_SIM_API_PORT, DEFAULT_SIM_HOST, DEFAULT_SIM_PORT, DEFAULT_STEP_SLEEP_S

logger = logging.getLogger(__name__)


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


# ─── Server-side gym bootstrapping (used by sim_server and scout_locations) ──

def setup_env_and_imports() -> None:
    """Set UnrealEnv, import gym_unrealcv, and apply monkey-patches.

    Only needed on the simulator machine (sim_server.py) and by scout_locations.
    Client scripts do NOT call this; they use SimClient instead.
    """
    os.environ.setdefault("UnrealEnv", str(ENVS_DIR))

    import gym_unrealcv  # noqa: F401

    import gym_unrealcv.envs.utils.misc as _misc
    _original_get_settingpath = _misc.get_settingpath

    def _get_settingpath(filename):
        overlay = SCENES_DIR / filename
        if overlay.exists():
            return str(overlay)
        return _original_get_settingpath(filename)

    _misc.get_settingpath = _get_settingpath

    import gym_unrealcv.envs.base_env as _base_env

    def _patched_remove_agent(self, name):
        agent_index = self.player_list.index(name)
        self.player_list.remove(name)
        self.cam_list = self.remove_cam(name)
        self._action_spaces.pop(agent_index)
        self._observation_spaces.pop(agent_index)
        self._sync_spaces()
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

    from unrealcv.api import UnrealCv_API

    def _patched_init_map(self):
        self.cam = self.get_camera_config()
        self.obj_dict = {}

    UnrealCv_API.init_map = _patched_init_map


def import_batch_module() -> Any:
    """Import and return the batch runner module."""
    from rvln.eval import batch_runner as batch
    return batch


# ─── SimClient-based environment setup ──────────────────────────────────────

def setup_sim_env(
    time_dilation_val: int,
    seed: int,
    batch: Any,
    sim_host: str = DEFAULT_SIM_HOST,
    sim_port: int = DEFAULT_SIM_PORT,
    sim_api_port: int = DEFAULT_SIM_API_PORT,
    init_retries: int = 10,
    retry_delay: float = 15.0,
) -> Any:
    """Connect to the sim API server and initialize the environment.

    Returns a SimClient. The server (run_simulator.py) must already be running.
    The server determines which map to use; clients query it via
    client.get_map_info() after connecting.

    The server's /init endpoint is idempotent: if the first request times out
    while the server is still initializing, retries will either trigger the
    same initialization check or hit the fast "already_initialized" path once
    the server finishes.
    """
    from requests.exceptions import ConnectionError, ReadTimeout

    from rvln.sim.sim_client import SimClient

    client = SimClient(f"http://{sim_host}:{sim_api_port}")
    for attempt in range(1, init_retries + 1):
        try:
            client.init_env(time_dilation_val, seed)
            break
        except (ReadTimeout, ConnectionError) as exc:
            if attempt == init_retries:
                raise
            logger.warning(
                "init_env attempt %d/%d failed (%s), retrying in %.0fs...",
                attempt, init_retries, exc, retry_delay,
            )
            time.sleep(retry_delay)
    time.sleep(batch.SLEEP_AFTER_RESET_S)
    return client


# ─── Camera helpers ───────────────────────────────────────────────────────────

def _push_to_playback(env: Any, image: Optional[np.ndarray]) -> None:
    """If a playback buffer is attached to *env*, update it with the latest frame.

    The buffer is set up by ``rvln.eval.video_recorder.ensure_episode_playback``;
    this hook is a no-op when no recorder is active.
    """
    if image is None:
        return
    buf = getattr(env, "_playback_buffer", None)
    if buf is not None:
        try:
            buf.update(image)
        except Exception as exc:
            logger.warning("playback buffer update failed: %s", exc)


def set_drone_cam_and_get_image(env: Any, cam_id: Optional[int] = None) -> Optional[np.ndarray]:
    """Capture a frame from the drone's camera via the sim server.

    Returns the image as an ndarray, or None on failure.
    """
    from rvln.sim.sim_client import SimClient

    image: Optional[np.ndarray] = None
    if isinstance(env, SimClient):
        image, _pos, _rot = env.get_frame(cam_id)
    else:
        # Fallback for direct gym env (scout_locations)
        cam = cam_id if cam_id is not None else env.unwrapped.agents[env.unwrapped.player_list[0]]['cam_id']
        x, y, z = env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0])
        roll, yaw, pitch = env.unwrapped.unrealcv.get_obj_rotation(env.unwrapped.player_list[0])
        env.unwrapped.unrealcv.set_cam(cam, [x, y, z], [roll, pitch, yaw])
        for attempt in range(2):
            try:
                image = env.unwrapped.unrealcv.get_image(cam, "lit")
                break
            except TypeError:
                logger.warning(
                    "get_image returned invalid data (attempt %d/2). Retrying after short delay.",
                    attempt + 1,
                )
                time.sleep(0.5)
        if image is None:
            logger.warning("get_image failed after retries.")

    _push_to_playback(env, image)
    return image


def interactive_camera_select(env: Any, initial_pos: List[float], batch: Any) -> int:
    """Cycle through cameras with the drone at initial_pos; let user choose.

    When env is a SimClient, the camera picker GUI runs on the server
    (the simulator machine, which has the display). For a direct gym env
    (scout_locations), the picker runs locally.

    Returns the selected camera ID.
    """
    from rvln.sim.sim_client import SimClient

    initial_pos = normalize_initial_pos(initial_pos)

    if isinstance(env, SimClient):
        logger.info("Requesting camera selection on simulator (check the simulator display)...")
        return env.select_camera(initial_pos[0:3], initial_pos[3])

    try:
        import cv2
    except ImportError:
        logger.error("Camera select requires opencv-python (cv2).")
        sys.exit(1)

    env.unwrapped.unrealcv.set_obj_location(
        env.unwrapped.player_list[0], initial_pos[0:3]
    )
    env.unwrapped.unrealcv.set_rotation(
        env.unwrapped.player_list[0], initial_pos[3] - 180
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
    set_cam_fn: Any = None,
    trajectory_log: Optional[List[Dict[str, Any]]] = None,
    sleep_s: float = DEFAULT_STEP_SLEEP_S,
    drone_cam_id: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], List[float], int]:
    """Apply action poses. Works with SimClient or direct gym env.

    Computes world coordinates from relative poses, sends them to the server
    (or applies directly), and returns (image, current_pose, steps_applied).
    current_pose is [x, y, z, yaw_deg] relative to initial.
    """
    from rvln.sim.sim_client import SimClient

    current_pose: List[float] = [0.0, 0.0, 0.0, 0.0]
    image = None
    steps = 0

    world_positions: List[List[float]] = []
    valid_poses: List[Tuple[float, float, float, float]] = []

    for action_pose in action_poses:
        if not (isinstance(action_pose, (list, tuple)) and len(action_pose) >= 4):
            continue
        relative_x = float(action_pose[0])
        relative_y = float(action_pose[1])
        relative_z = float(action_pose[2])
        relative_yaw_rad = float(action_pose[3])
        relative_yaw = float(np.degrees(relative_yaw_rad))
        relative_yaw = (relative_yaw + 180) % 360 - 180

        global_x, global_y = transform_to_global(relative_x, relative_y, initial_yaw)
        absolute_yaw = normalize_angle(relative_yaw + initial_yaw)
        absolute_pos = [
            global_x + initial_x,
            global_y + initial_y,
            relative_z + initial_z,
            absolute_yaw,
        ]

        world_positions.append(absolute_pos)
        valid_poses.append((relative_x, relative_y, relative_z, relative_yaw))

    if not world_positions:
        return None, current_pose, 0

    if trajectory_log is not None:
        for rel_x, rel_y, rel_z, rel_yaw in valid_poses:
            trajectory_log.append({
                "state": [
                    [rel_x, rel_y, rel_z],
                    [0, rel_yaw, 0],
                ]
            })

    current_pose = list(valid_poses[-1])
    steps = len(world_positions)

    if isinstance(env, SimClient):
        image, _pos, _rot, _steps = env.step(world_positions, drone_cam_id, sleep_s)
    else:
        for i, abs_pos in enumerate(world_positions):
            env.unwrapped.unrealcv.set_obj_location(
                env.unwrapped.player_list[0], abs_pos[:3]
            )
            env.unwrapped.unrealcv.set_rotation(
                env.unwrapped.player_list[0], abs_pos[3] - 180
            )
            if set_cam_fn is not None:
                set_cam_fn(env)
            if i == len(world_positions) - 1:
                image = set_drone_cam_and_get_image(env, drone_cam_id)
            time.sleep(sleep_s)

    _push_to_playback(env, image)
    return image, current_pose, steps
