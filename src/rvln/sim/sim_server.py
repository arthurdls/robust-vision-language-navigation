"""
Sim API server: wraps the gym/UnrealCV environment behind a Flask HTTP API.

Started by run_simulator.py after the Unreal binary is ready. Client scripts
talk to this server instead of connecting to UnrealCV directly.

Endpoints:
    POST /init       - create gym env, reset, spawn NPCs
    POST /teleport   - move drone to absolute position
    POST /step       - apply a sequence of positions, return final frame
    POST /get_frame  - capture current frame
    POST /get_pose   - query current drone position/rotation
    POST /get_camera_frame - get frame from a specific camera (for camera selection)
    POST /select_camera    - interactive camera picker (runs GUI on server)
    POST /close      - shut down gym env
"""

import base64
import logging
import time
from io import BytesIO
from typing import Any, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request

from rvln.config import SLEEP_SHORT_S

logger = logging.getLogger(__name__)

app = Flask(__name__)

_env: Optional[Any] = None
_drone_name: Optional[str] = None
_drone_cam_id: int = 0
_initialized = False


def _sync_cam(cam_id: Optional[int] = None) -> None:
    if cam_id is None:
        cam_id = _drone_cam_id
    x, y, z = _env.unwrapped.unrealcv.get_obj_location(_drone_name)
    roll, yaw, pitch = _env.unwrapped.unrealcv.get_obj_rotation(_drone_name)
    _env.unwrapped.unrealcv.set_cam(cam_id, [x, y, z], [roll, pitch, yaw])


def _capture_image(cam_id: Optional[int] = None) -> Optional[np.ndarray]:
    if cam_id is None:
        cam_id = _drone_cam_id
    _sync_cam(cam_id)
    for attempt in range(2):
        try:
            return _env.unwrapped.unrealcv.get_image(cam_id, "lit")
        except TypeError:
            logger.warning("get_image returned invalid data (attempt %d/2)", attempt + 1)
            time.sleep(0.5)
    return None


def _image_to_base64_jpeg(image: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _get_pose() -> tuple:
    pos = list(_env.unwrapped.unrealcv.get_obj_location(_drone_name))
    rot = list(_env.unwrapped.unrealcv.get_obj_rotation(_drone_name))
    return [float(v) for v in pos], [float(v) for v in rot]


@app.route("/init", methods=["POST"])
def handle_init():
    global _env, _drone_name, _drone_cam_id, _initialized

    if _initialized:
        return jsonify({"status": "already_initialized", "drone_name": _drone_name,
                        "drone_cam_id": _drone_cam_id,
                        "cam_count": _env.unwrapped.unrealcv.get_camera_num() if _env else 0})

    data = request.get_json(force=True)
    env_id = data.get("env_id")
    if not env_id:
        return jsonify({"error": "env_id is required in /init payload"}), 400
    time_dilation_val = int(data.get("time_dilation", 10))
    seed = int(data.get("seed", 0))

    import os
    import gymnasium as gym
    import gym_unrealcv
    from gym_unrealcv.envs.wrappers import time_dilation as td_mod, configUE, augmentation
    from rvln.paths import ENVS_DIR

    os.environ.setdefault("UnrealEnv", str(ENVS_DIR))

    from rvln.sim.env_setup import setup_env_and_imports
    setup_env_and_imports()

    gym_unrealcv.register_env(env_id)
    env = gym.make(env_id)
    if time_dilation_val > 0:
        env = td_mod.TimeDilationWrapper(env, time_dilation_val)
    env.unwrapped.agents_category = ["drone"]
    env = configUE.ConfigUEWrapper(env, resolution=(256, 256))
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env.reset(seed=seed)

    drone_name = env.unwrapped.player_list[0]
    drone_cam_id = env.unwrapped.agents[drone_name]['cam_id']
    env.unwrapped.unrealcv.set_viewport(drone_name)
    env.unwrapped.unrealcv.set_phy(drone_name, 0)

    time.sleep(SLEEP_SHORT_S)
    env.unwrapped.unrealcv.new_obj("bp_character_C", "BP_Character_21", [0, 0, 0])
    env.unwrapped.unrealcv.set_appearance("BP_Character_21", 0)
    env.unwrapped.unrealcv.set_obj_rotation("BP_Character_21", [0, 0, 0])
    time.sleep(SLEEP_SHORT_S)
    env.unwrapped.unrealcv.new_obj("BP_BaseCar_C", "BP_Character_22", [1000, 0, 0])
    env.unwrapped.unrealcv.set_appearance("BP_Character_22", 2)
    env.unwrapped.unrealcv.set_obj_rotation("BP_Character_22", [0, 0, 0])
    env.unwrapped.unrealcv.set_phy("BP_Character_22", 0)
    time.sleep(SLEEP_SHORT_S)

    _env = env
    _drone_name = drone_name
    _drone_cam_id = drone_cam_id
    _initialized = True

    cam_count = _env.unwrapped.unrealcv.get_camera_num()
    logger.info("Sim env initialized: drone=%s, drone_cam=%d, cameras=%d",
                _drone_name, _drone_cam_id, cam_count)

    return jsonify({"status": "ready", "drone_name": _drone_name,
                    "drone_cam_id": _drone_cam_id, "cam_count": cam_count})


@app.route("/teleport", methods=["POST"])
def handle_teleport():
    if not _initialized:
        return jsonify({"error": "not initialized"}), 400

    data = request.get_json(force=True)
    position = data["position"]
    yaw = float(data["yaw"])

    _env.unwrapped.unrealcv.set_obj_location(_drone_name, position[:3])
    _env.unwrapped.unrealcv.set_rotation(_drone_name, yaw - 180)
    _sync_cam()

    return jsonify({"status": "ok"})


@app.route("/step", methods=["POST"])
def handle_step():
    if not _initialized:
        return jsonify({"error": "not initialized"}), 400

    data = request.get_json(force=True)
    positions = data["positions"]
    cam_id = int(data.get("cam_id", 0))
    sleep_s = float(data.get("sleep_s", 0.1))

    steps_applied = 0
    for i, pose in enumerate(positions):
        if not (isinstance(pose, (list, tuple)) and len(pose) >= 4):
            continue

        abs_x, abs_y, abs_z, abs_yaw = float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3])
        _env.unwrapped.unrealcv.set_obj_location(_drone_name, [abs_x, abs_y, abs_z])
        _env.unwrapped.unrealcv.set_rotation(_drone_name, abs_yaw - 180)
        _sync_cam(cam_id)
        steps_applied += 1
        time.sleep(sleep_s)

    image = _capture_image(cam_id)
    pos, rot = _get_pose()

    resp = {"position": pos, "rotation": rot, "steps_applied": steps_applied}
    if image is not None:
        resp["image"] = _image_to_base64_jpeg(image)
    else:
        resp["image"] = None

    return jsonify(resp)


@app.route("/get_frame", methods=["POST"])
def handle_get_frame():
    if not _initialized:
        return jsonify({"error": "not initialized"}), 400

    data = request.get_json(force=True)
    cam_id = int(data.get("cam_id", 0))

    image = _capture_image(cam_id)
    pos, rot = _get_pose()

    resp = {"position": pos, "rotation": rot}
    if image is not None:
        resp["image"] = _image_to_base64_jpeg(image)
    else:
        resp["image"] = None

    return jsonify(resp)


@app.route("/get_pose", methods=["POST"])
def handle_get_pose():
    if not _initialized:
        return jsonify({"error": "not initialized"}), 400

    pos, rot = _get_pose()
    return jsonify({"position": pos, "rotation": rot})


@app.route("/get_camera_frame", methods=["POST"])
def handle_get_camera_frame():
    if not _initialized:
        return jsonify({"error": "not initialized"}), 400

    data = request.get_json(force=True)
    cam_id = int(data.get("cam_id", 0))
    position = data.get("position")
    yaw = data.get("yaw")

    if position is not None and yaw is not None:
        _env.unwrapped.unrealcv.set_obj_location(_drone_name, position[:3])
        _env.unwrapped.unrealcv.set_rotation(_drone_name, float(yaw) - 180)

    _sync_cam(cam_id)
    image = _capture_image(cam_id)
    cam_count = _env.unwrapped.unrealcv.get_camera_num()

    resp = {"cam_count": cam_count}
    if image is not None:
        resp["image"] = _image_to_base64_jpeg(image)
    else:
        resp["image"] = None

    return jsonify(resp)


@app.route("/select_camera", methods=["POST"])
def handle_select_camera():
    if not _initialized:
        return jsonify({"error": "not initialized"}), 400

    import cv2

    data = request.get_json(force=True)
    position = data.get("position")
    yaw = data.get("yaw")

    if position is not None and yaw is not None:
        _env.unwrapped.unrealcv.set_obj_location(_drone_name, position[:3])
        _env.unwrapped.unrealcv.set_rotation(_drone_name, float(yaw) - 180)
    _sync_cam()
    time.sleep(1.0)

    n_cams = _env.unwrapped.unrealcv.get_camera_num()
    if n_cams <= 0:
        return jsonify({"error": "no cameras available"}), 500

    current = 0
    window_name = "Camera select: n=next, p=prev, s or Enter=select, q or Esc=quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    selected = None
    while True:
        _sync_cam(current)
        image = _capture_image(current)

        if image is None:
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        if image.ndim == 3 and image.shape[2] == 3:
            display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            display = image.copy()

        label = "Camera %d / %d" % (current, n_cams)
        cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(window_name, display)
        k = cv2.waitKey(0) & 0xFF

        if k == ord("s") or k == 13 or k == 10:
            selected = current
            break
        if k == ord("q") or k == 27:
            break
        if k == ord("n"):
            current = (current + 1) % n_cams
        elif k == ord("p"):
            current = (current - 1) % n_cams

    cv2.destroyAllWindows()

    if selected is None:
        return jsonify({"error": "camera selection cancelled"}), 400

    logger.info("Selected camera %d for OpenVLA.", selected)
    return jsonify({"cam_id": selected, "cam_count": n_cams})


@app.route("/close", methods=["POST"])
def handle_close():
    global _env, _initialized

    if _env is not None:
        try:
            _env.close()
        except Exception as e:
            logger.warning("Error closing env: %s", e)
        _env = None
        _initialized = False

    return jsonify({"status": "ok"})


def run_server(port: int = 9001) -> None:
    app.run(host="0.0.0.0", port=port, threaded=False)
