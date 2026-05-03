"""
Sim API client: lightweight HTTP client for the sim server.

Provides the interface that control scripts use to interact with the simulator.
Works identically whether the server is on localhost or a remote host.
"""

import base64
import logging
from typing import Optional

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


class SimClient:
    """HTTP client for the sim API server."""

    def __init__(self, server_url: str, timeout: float = 30.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.drone_name: Optional[str] = None
        self.drone_cam_id: int = 0
        self.cam_count: int = 0

    def _post(self, endpoint: str, data: Optional[dict] = None) -> dict:
        url = f"{self.server_url}{endpoint}"
        resp = requests.post(url, json=data or {}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _decode_image(b64: Optional[str]) -> Optional[np.ndarray]:
        if b64 is None:
            return None
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def init_env(self, env_id: str, time_dilation: int, seed: int) -> dict:
        resp = self._post("/init", {
            "env_id": env_id,
            "time_dilation": time_dilation,
            "seed": seed,
        })
        self.drone_name = resp.get("drone_name")
        self.drone_cam_id = resp.get("drone_cam_id", 0)
        self.cam_count = resp.get("cam_count", 0)
        logger.info("SimClient connected: drone=%s, drone_cam=%d, cameras=%d",
                    self.drone_name, self.drone_cam_id, self.cam_count)
        return resp

    def teleport(self, position: list, yaw: float) -> None:
        self._post("/teleport", {"position": list(position), "yaw": float(yaw)})

    def step(
        self,
        positions: list,
        cam_id: Optional[int] = None,
        sleep_s: float = 0.1,
    ) -> tuple:
        """Apply positions and return (image, world_position, world_rotation, steps_applied).

        positions: list of [x, y, z, yaw] in absolute world coordinates.
        Returns (ndarray_or_None, [x,y,z], [roll,yaw,pitch], steps_applied).
        """
        resp = self._post("/step", {
            "positions": [[float(v) for v in p] for p in positions],
            "cam_id": cam_id if cam_id is not None else self.drone_cam_id,
            "sleep_s": sleep_s,
        })
        image = self._decode_image(resp.get("image"))
        return (
            image,
            resp["position"],
            resp["rotation"],
            resp.get("steps_applied", len(positions)),
        )

    def get_frame(self, cam_id: Optional[int] = None) -> tuple:
        """Capture current frame. Returns (image, position, rotation)."""
        resp = self._post("/get_frame", {"cam_id": cam_id if cam_id is not None else self.drone_cam_id})
        image = self._decode_image(resp.get("image"))
        return image, resp["position"], resp["rotation"]

    def get_pose(self) -> tuple:
        """Query drone world pose. Returns (position, rotation)."""
        resp = self._post("/get_pose")
        return resp["position"], resp["rotation"]

    def get_camera_frame(self, cam_id: int, position: list, yaw: float) -> tuple:
        """Get frame from a specific camera for camera selection.

        Returns (image, cam_count).
        """
        resp = self._post("/get_camera_frame", {
            "cam_id": cam_id,
            "position": list(position),
            "yaw": float(yaw),
        })
        image = self._decode_image(resp.get("image"))
        return image, resp.get("cam_count", 0)

    def select_camera(self, position: list, yaw: float) -> int:
        """Run interactive camera selection on the server (blocks until user picks).

        Returns the selected camera ID.
        """
        url = f"{self.server_url}/select_camera"
        resp = requests.post(url, json={
            "position": list(position),
            "yaw": float(yaw),
        }, timeout=300.0)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        self.cam_count = data.get("cam_count", self.cam_count)
        return data["cam_id"]

    def close(self) -> None:
        try:
            self._post("/close")
        except Exception as e:
            logger.warning("Error closing sim client: %s", e)
