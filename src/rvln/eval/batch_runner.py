"""
Batch UAV evaluation runner.

Drives the OpenVLA control loop for batch task evaluation in the Unreal sim.
Vendored from UAV-Flow/UAV-Flow-Eval/batch_run_act_all.py (commit 0114801).
"""

import base64
import json
import logging
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
from PIL import Image

from rvln.sim.pose import calculate_new_pose  # noqa: F401

logger = logging.getLogger(__name__)

# ====== Constants ======
IMG_INPUT_SIZE: Tuple[int, int] = (224, 224)
SLEEP_SHORT_S: float = 1.0
SLEEP_AFTER_RESET_S: float = 2.0
ACTION_SMALL_DELTA_POS: float = 3.0
ACTION_SMALL_DELTA_YAW: float = 1.0
ACTION_SMALL_STEPS: int = 10


def send_prediction_request(
    image: Image.Image,
    proprio: np.ndarray,
    instr: str,
    server_url: str,
) -> Optional[Dict[str, Any]]:
    """Send a request to the inference service and return JSON response."""
    proprio_list = proprio.tolist()
    img_io = BytesIO()
    if image.size != IMG_INPUT_SIZE:
        image = image.resize(IMG_INPUT_SIZE)
    image.save(img_io, format="PNG")
    img_data = img_io.getvalue()
    img_base64 = base64.b64encode(img_data).decode("utf-8")
    payload: Dict[str, Any] = {
        "image": img_base64,
        "proprio": proprio_list,
        "instr": instr,
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(
            server_url,
            data=json.dumps(payload),
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return None


def set_cam(env: Any) -> None:
    """Compute and set camera pose based on current object pose."""
    x, y, z = env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0])
    roll, yaw, pitch = env.unwrapped.unrealcv.get_obj_rotation(env.unwrapped.player_list[0])
    env.unwrapped.unrealcv.set_cam(0, [x, y, z], [roll, pitch, yaw])


def reset_model(server_url: str) -> None:
    """Call server /reset to reset the model."""
    try:
        resp = requests.post(server_url.replace("/predict", "/reset"), timeout=10)
        logger.info(f"Model reset response: {resp.status_code}")
    except Exception as e:
        logger.error(f"Model reset failed: {e}")
