#!/usr/bin/env python3
"""
Single-file real-drone integration runner.

Pipeline:
1) Prompt user for an instruction (unless provided via CLI).
2) LTL decomposition with LLMUserInterface + LTLSymbolicPlanner.
3) Subgoal conversion + GoalAdherenceMonitor supervision.
4) OpenVLA /predict + /reset calls.
5) Real command streaming to drone server using boieng wire format:
   [frame_count, vx, vy, vz, yaw] as float32 over TCP.
6) Pose source:
   - Primary: external odometry feed (HTTP poll or UDP stream)
   - Fallback: dead-reckoning from sent commands.

Operator help: when the monitor detects a stall (completion plateau),
exhausts its correction budget, or hits the step/time limit, the drone
pauses and the operator is prompted with four options:
  [1] Provide a new low-level OpenVLA instruction for the current subgoal.
  [2] Replan from a new high-level natural-language instruction (re-runs
      the LTL planner from the drone's current position).
  [3] Skip (continue or end the current subgoal).
  [4] Abort the mission.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import shutil
import signal
import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image


from rvln.config import (
    ACTION_SMALL_DELTA_POS,
    ACTION_SMALL_DELTA_YAW,
    ACTION_SMALL_STEPS,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_INIT_TIMEOUT,
    DEFAULT_CAMERA_RETRIES,
    DEFAULT_COMMAND_DT_S,
    DEFAULT_CONTROL_PORT,
    DEFAULT_CONTROL_RETRIES,
    DEFAULT_CONTROL_RETRY_SLEEP,
    DEFAULT_DIARY_CHECK_INTERVAL,
    DEFAULT_DIARY_CHECK_INTERVAL_S,
    DEFAULT_DIARY_MODE,
    DEFAULT_LLM_FALLBACK_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_CORRECTIONS,
    DEFAULT_MAX_SECONDS_PER_SUBGOAL,
    DEFAULT_MAX_STEPS_PER_SUBGOAL,
    DEFAULT_ODOM_STALE_TIMEOUT_S,
    DEFAULT_ODOM_UDP_HOST,
    DEFAULT_ODOM_UDP_PORT,
    DEFAULT_OPENVLA_PREDICT_URL,
    DEFAULT_PREFERRED_SERVER_HOST,
    DEFAULT_STALL_COMPLETION_FLOOR,
    DEFAULT_STALL_THRESHOLD,
    DEFAULT_STALL_WINDOW,
    DEFAULT_VLM_FALLBACK_MODEL,
    DEFAULT_VLM_MODEL,
    IMG_INPUT_SIZE,
)
from rvln.paths import REPO_ROOT, load_env_vars
from rvln.sim.env_setup import state_for_openvla
from rvln.sim.transforms import normalize_angle, parse_position, relative_pose
from rvln.ai.utils.llm_providers import LLMFactory


logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "hardware"

stop_capture = False


def sanitize_name(text: str, max_len: int = 48) -> str:
    clean = text.lower().replace(" ", "_")
    safe = "".join(c for c in clean if c.isalnum() or c == "_")
    return safe[:max_len] or "subgoal"


def signal_handler(sig, frame):
    global stop_capture
    stop_capture = True
    logger.info("Signal received; shutting down.")
    cv2.destroyAllWindows()


class ThreadedCamera:
    def __init__(self, src: int, fps: int, max_reopen_attempts: int, init_timeout: float):
        self.src = src
        self.fps = fps
        self.max_reopen_attempts = max_reopen_attempts
        self.init_timeout = init_timeout
        self.capture = cv2.VideoCapture(src)
        self.frame: Optional[np.ndarray] = None
        self.read_once = False
        self.failed = False
        self.failure_reason = ""
        self._reopen_attempts = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self) -> None:
        global stop_capture
        start = time.time()
        while not stop_capture:
            if not self.capture.isOpened():
                self._reopen_attempts += 1
                if self._reopen_attempts > self.max_reopen_attempts:
                    self.failed = True
                    self.failure_reason = (
                        f"Camera {self.src} failed to open after "
                        f"{self.max_reopen_attempts} retries"
                    )
                    return
                logger.warning(
                    "Camera %s not open. Retry %d/%d",
                    self.src, self._reopen_attempts, self.max_reopen_attempts,
                )
                self.capture = cv2.VideoCapture(self.src)
                time.sleep(1.0)
                if time.time() - start > self.init_timeout and not self.read_once:
                    self.failed = True
                    self.failure_reason = (
                        f"Camera {self.src} did not produce frames in "
                        f"{self.init_timeout:.1f}s"
                    )
                    return
                continue

            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            center_h, center_w = h // 2, w // 2
            size = min(h, w) // 2
            cropped = frame[
                center_h - size:center_h + size,
                center_w - size:center_w + size,
            ]
            resized = cv2.resize(cropped, (640, 640))
            with self._lock:
                self.frame = resized
                self.read_once = True
            if self.fps > 0:
                time.sleep(1.0 / float(self.fps))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def wait_until_ready(self) -> bool:
        start = time.time()
        while not stop_capture:
            if self.failed:
                return False
            if self.read_once:
                return True
            if time.time() - start > self.init_timeout:
                self.failed = True
                self.failure_reason = (
                    f"Camera {self.src} did not produce frames in {self.init_timeout:.1f}s"
                )
                return False
            time.sleep(0.1)
        return False

    def release(self) -> None:
        try:
            self.capture.release()
        except Exception:
            pass


class HttpFrameCamera:
    """Camera shim that pulls JPEG/PNG frames from a remote HTTP endpoint.

    Mirrors the public surface of ThreadedCamera (read, wait_until_ready,
    release, failed, failure_reason) so callers don't care about the source.
    """

    def __init__(self, url: str, fps: int, init_timeout: float, poll_timeout: float = 0.5):
        self.url = url
        self.fps = max(1, fps)
        self.init_timeout = init_timeout
        self.poll_timeout = poll_timeout
        self.frame: Optional[np.ndarray] = None
        self.read_once = False
        self.failed = False
        self.failure_reason = ""
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _decode(self, payload: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(payload, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _update(self) -> None:
        global stop_capture
        period = 1.0 / float(self.fps)
        while not stop_capture:
            t0 = time.time()
            try:
                resp = requests.get(self.url, timeout=self.poll_timeout)
                resp.raise_for_status()
                frame = self._decode(resp.content)
            except Exception as exc:
                logger.warning("Frame poll failed: %s", exc)
                frame = None

            if frame is not None:
                with self._lock:
                    self.frame = frame
                    self.read_once = True

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def wait_until_ready(self) -> bool:
        start = time.time()
        while not stop_capture:
            if self.read_once:
                return True
            if time.time() - start > self.init_timeout:
                self.failed = True
                self.failure_reason = (
                    f"HTTP frame source {self.url} did not produce frames in "
                    f"{self.init_timeout:.1f}s"
                )
                return False
            time.sleep(0.1)
        return False

    def release(self) -> None:
        # Daemon thread exits with stop_capture; nothing to free.
        pass


class OpenVLAClient:
    def __init__(self, predict_url: str, timeout_s: float = 30.0):
        self.predict_url = predict_url
        self.timeout_s = timeout_s

    @property
    def reset_url(self) -> str:
        return self.predict_url.replace("/predict", "/reset")

    def reset_model(self) -> None:
        try:
            resp = requests.post(self.reset_url, timeout=10)
        except Exception as exc:
            raise RuntimeError(f"OpenVLA model reset failed: {exc}") from exc
        if resp.status_code == 404:
            # OpenVLA's bare /predict server doesn't expose /reset;
            # scripts/start_server.py patches it in. Treat absence as a no-op
            # rather than aborting every subgoal.
            logger.warning(
                "OpenVLA server has no /reset endpoint at %s; skipping reset.",
                self.reset_url,
            )
            return
        try:
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"OpenVLA model reset failed: {exc}") from exc
        logger.info("Model reset: %s", resp.status_code)

    def predict(self, image_bgr: np.ndarray, proprio: np.ndarray, instr: str) -> Optional[Dict[str, Any]]:
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        if pil.size != IMG_INPUT_SIZE:
            pil = pil.resize(IMG_INPUT_SIZE)
        buf = BytesIO()
        pil.save(buf, format="PNG")
        payload = {
            "image": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "proprio": proprio.tolist(),
            "instr": instr,
        }
        resp = requests.post(
            self.predict_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()


class FrameRecorder:
    """Optional per-frame metadata log for cost/replay analysis.

    Frame PNGs are written every step regardless (the goal-adherence monitor
    needs them on disk); this recorder writes one JSON line per *recorded*
    frame to ``recording_log.jsonl``, downsampled to ``record_fps``.

    Each line ties a frame to:
      - subgoal index, subgoal NL, low-level OpenVLA instruction
      - current LTL formula (NL form)
      - world pose and subgoal-relative pose
      - the OpenVLA action tensor returned for this frame
      - cumulative input/output token counts across planner, subgoal
        converter, and goal-adherence monitor (for cost tracking)

    ``ctx`` is a mutable dict the caller updates as the run progresses
    (so the recorder always sees the current llm_interface and ltl_plan,
    even after a replan). Required keys: ``llm_interface`` (LLMUserInterface
    or None) and ``ltl_plan`` (dict).
    """

    def __init__(self, log_path: Optional[Path], record_fps: float, ctx: Dict[str, Any]):
        self.log_path = log_path
        self.min_interval = 1.0 / record_fps if record_fps > 0 else 0.0
        self.ctx = ctx
        self._last_logged_t = 0.0
        self._fh = open(log_path, "a") if log_path is not None else None

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def maybe_log(
        self,
        *,
        step: int,
        subgoal_index: int,
        subgoal_nl: str,
        current_instruction: str,
        frame_path: Path,
        world_pose: List[float],
        subgoal_rel_pose: List[float],
        openvla_action: Any,
        converter,
        monitor,
    ) -> None:
        if not self.enabled:
            return
        now = time.time()
        if (now - self._last_logged_t) < self.min_interval:
            return
        self._last_logged_t = now

        in_tok = 0
        out_tok = 0
        llmi = self.ctx.get("llm_interface")
        sources = [
            getattr(llmi, "llm_call_records", []) or [],
            getattr(converter, "llm_call_records", []) or [],
            getattr(monitor, "vlm_rtts", []) or [],
        ]
        for src in sources:
            for r in src:
                in_tok += r.get("input_tokens", 0) if isinstance(r, dict) else 0
                out_tok += r.get("output_tokens", 0) if isinstance(r, dict) else 0

        ltl_plan = self.ctx.get("ltl_plan") or {}
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "frame_path": str(frame_path),
            "subgoal_index": subgoal_index,
            "subgoal_nl": subgoal_nl,
            "openvla_instruction": current_instruction,
            "ltl_nl_formula": ltl_plan.get("ltl_nl_formula", ""),
            "world_pose": list(world_pose),
            "subgoal_rel_pose": list(subgoal_rel_pose),
            "openvla_action": openvla_action,
            "cumulative_input_tokens": in_tok,
            "cumulative_output_tokens": out_tok,
        }
        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def get_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError as exc:
        logger.warning("Could not detect local IP (%s), defaulting to 127.0.0.1", exc)
        return "127.0.0.1"
    finally:
        sock.close()


def resolve_server_address(preferred_host: str, preferred_port: int, timeout_s: float = 0.75) -> Tuple[str, int]:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(timeout_s)
    try:
        probe.connect((preferred_host, preferred_port))
        logger.info("Using preferred control server %s:%d", preferred_host, preferred_port)
        return preferred_host, preferred_port
    except OSError:
        local_ip = get_local_ip()
        logger.warning(
            "Preferred server %s:%d unreachable. Falling back to local IP %s:%d",
            preferred_host, preferred_port, local_ip, preferred_port,
        )
        return local_ip, preferred_port
    finally:
        probe.close()


class DroneControlClient:
    def __init__(self, host: str, port: int, connect_retries: int, retry_sleep_s: float):
        self.host = host
        self.port = port
        self.connect_retries = connect_retries
        self.retry_sleep_s = retry_sleep_s
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        for attempt in range(1, self.connect_retries + 1):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                logger.info("Connected to control server %s:%d", self.host, self.port)
                return
            except Exception as exc:
                logger.warning(
                    "Control server connect attempt %d/%d failed: %s",
                    attempt, self.connect_retries, exc,
                )
                time.sleep(self.retry_sleep_s)
        raise RuntimeError(f"Could not connect to control server {self.host}:{self.port}")

    def send_command(self, frame_count: int, command: np.ndarray) -> None:
        if self.sock is None:
            raise RuntimeError("Control socket not connected")
        payload = np.array([frame_count, *command.tolist()], dtype=np.float32).tobytes()
        self.sock.sendall(payload)

    def close(self) -> None:
        if self.sock is None:
            return
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass
        self.sock = None


@dataclass
class PoseSample:
    x: float
    y: float
    z: float
    yaw_deg: float
    ts: float


class OdometryPoseProvider:
    """Pose source with a background polling thread.

    HTTP and UDP I/O happen on a daemon thread at a fixed cadence; ``get_pose``
    is a lock-protected read of the cached latest sample, so the hot control
    loop never blocks on a slow odometry server. A sample older than
    ``stale_timeout_s`` is treated as missing, which lets ``PoseManager``
    fall over to dead-reckoning the same way the synchronous version did.
    """

    def __init__(
        self,
        http_url: Optional[str],
        udp_host: str,
        udp_port: int,
        stale_timeout_s: float,
        poll_hz: float = 50.0,
    ):
        self.http_url = http_url
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.stale_timeout_s = stale_timeout_s
        self._poll_interval = 1.0 / poll_hz if poll_hz > 0 else 0.02
        self._lock = threading.Lock()
        self.last_sample: Optional[PoseSample] = None
        self._udp_sock: Optional[socket.socket] = None
        if udp_port > 0:
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_sock.bind((udp_host, udp_port))
            self._udp_sock.setblocking(False)
            logger.info("Listening for odometry UDP packets on %s:%d", udp_host, udp_port)

        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="odom-poller",
            daemon=True,
        )
        self._thread.start()

    def _parse_pose(self, payload: Dict[str, Any]) -> Optional[PoseSample]:
        try:
            x = float(payload["x"])
            y = float(payload["y"])
            z = float(payload["z"])
            yaw = float(payload["yaw"])
            return PoseSample(x=x, y=y, z=z, yaw_deg=normalize_angle(yaw), ts=time.time())
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse pose payload: %s (data: %s)", exc, payload)
            return None

    def _store(self, sample: PoseSample) -> None:
        with self._lock:
            self.last_sample = sample

    def _poll_http_once(self) -> None:
        if not self.http_url:
            return
        try:
            resp = requests.get(self.http_url, timeout=0.25)
            resp.raise_for_status()
            pose = self._parse_pose(resp.json())
        except Exception as exc:
            logger.warning("Odometry HTTP poll failed (%s): %s", self.http_url, exc)
            return
        if pose is not None:
            self._store(pose)

    def _poll_udp_drain(self) -> None:
        if self._udp_sock is None:
            return
        while True:
            try:
                data, _addr = self._udp_sock.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError:
                # Socket closed during shutdown.
                return
            except Exception as exc:
                logger.warning("Odometry UDP recv error: %s", exc)
                break
            try:
                payload = json.loads(data.decode("utf-8"))
            except Exception as exc:
                logger.warning("Odometry UDP packet decode failed: %s", exc)
                continue
            pose = self._parse_pose(payload)
            if pose is not None:
                self._store(pose)

    def _poll_loop(self) -> None:
        # All I/O lives here so get_pose() is non-blocking. Exceptions are
        # caught and logged so a transient odom outage never kills the thread.
        while not self._stop.is_set():
            try:
                self._poll_udp_drain()
                self._poll_http_once()
            except Exception as exc:  # defense in depth
                logger.warning("Odometry poll iteration failed: %s", exc)
            self._stop.wait(self._poll_interval)

    def get_pose(self) -> Optional[List[float]]:
        with self._lock:
            sample = self.last_sample
        if sample is None:
            return None
        if time.time() - sample.ts > self.stale_timeout_s:
            return None
        return [sample.x, sample.y, sample.z, sample.yaw_deg]

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("Odometry poll thread did not exit within 2s.")
        if self._udp_sock is not None:
            try:
                self._udp_sock.close()
            except Exception:
                pass


class DeadReckoningPoseProvider:
    """Integrate sent commands into a world-frame pose estimate.

    Wire commands have different meanings under different ``action_pose_mode``
    settings, so the integration formula depends on the mode:

    - ``delta_from_pose`` + ``command_is_velocity``: command is (vx, vy, vz,
      yaw_rate). Integrate ``pose += v * dt``. Used by the simulated mock.
    - ``delta_from_pose`` + not velocity: command is a per-step pose delta
      in subgoal-relative frame. Adding it to ``world_pose`` yields the
      target world pose (see math comment in ``update_from_command``).
    - ``direct``: command is the *absolute* target relative pose (the raw
      OpenVLA prediction). Without the subgoal origin the DR provider can't
      reconstruct world coords from it, so we leave ``world_pose`` alone.
      ``PoseManager`` continuously syncs DR via ``set_world_pose`` whenever
      odometry is healthy; on failover the DR pose stays frozen at the last
      good odom value (a safe-but-stale snapshot rather than divergent
      garbage).
    """

    def __init__(
        self,
        initial_world_pose: List[float],
        command_is_velocity: bool,
        action_pose_mode: str = "delta_from_pose",
    ):
        self.world_pose = list(initial_world_pose)
        self.command_is_velocity = command_is_velocity
        self.action_pose_mode = action_pose_mode
        self._direct_mode_warned = False

    def set_world_pose(self, pose: List[float]) -> None:
        self.world_pose = [float(p) for p in pose]

    def update_from_command(self, command: np.ndarray, dt_s: float) -> None:
        if self.action_pose_mode == "direct":
            # Wire command is an absolute target in subgoal-relative frame; we
            # don't have the subgoal origin here, so we cannot compute a world
            # pose from it. Skip integration; rely on odometry sync (or accept
            # a frozen last-known-pose fallback if odom dies).
            if not self._direct_mode_warned:
                logger.info(
                    "Dead-reckoning is a frozen-snapshot fallback in "
                    "action_pose_mode=direct; pose only advances while "
                    "odometry is healthy."
                )
                self._direct_mode_warned = True
            return

        vx, vy, vz, yaw = (
            float(command[0]),
            float(command[1]),
            float(command[2]),
            float(command[3]),
        )
        if self.command_is_velocity:
            self.world_pose[0] += vx * dt_s
            self.world_pose[1] += vy * dt_s
            self.world_pose[2] += vz * dt_s
            self.world_pose[3] = normalize_angle(self.world_pose[3] + math.degrees(yaw) * dt_s)
        else:
            self.world_pose[0] += vx
            self.world_pose[1] += vy
            self.world_pose[2] += vz
            self.world_pose[3] = normalize_angle(self.world_pose[3] + math.degrees(yaw))

    def get_pose(self) -> List[float]:
        return list(self.world_pose)


class PoseManager:
    def __init__(self, odom: Optional[OdometryPoseProvider], dr: DeadReckoningPoseProvider):
        self.odom = odom
        self.dr = dr
        self.mode = "odometry" if odom is not None else "dead_reckoning"
        self._failed_over = False

    def get_world_pose(self) -> List[float]:
        if self.mode == "odometry" and self.odom is not None:
            pose = self.odom.get_pose()
            if pose is not None:
                self.dr.set_world_pose(pose)
                return pose
            self.mode = "dead_reckoning"
            self._failed_over = True
            logger.warning("Odometry unavailable/stale. Falling back to dead-reckoning.")
        return self.dr.get_pose()

    def update_from_command(self, command: np.ndarray, dt_s: float) -> None:
        self.dr.update_from_command(command, dt_s)

    @property
    def failed_over(self) -> bool:
        return self._failed_over

    def close(self) -> None:
        if self.odom is not None:
            self.odom.close()


def resolve_model_with_fallback(primary_model: str, fallback_model: str) -> str:
    try:
        LLMFactory.create("openai", model=primary_model)
        return primary_model
    except Exception as exc:
        logger.warning(
            "Model '%s' unavailable (%s). Trying fallback '%s'.",
            primary_model, exc, fallback_model,
        )
        try:
            LLMFactory.create("openai", model=fallback_model)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Neither primary model '{primary_model}' nor fallback model "
                f"'{fallback_model}' is available: {fallback_exc}"
            ) from fallback_exc
        logger.info("Using fallback model '%s'.", fallback_model)
        return fallback_model


def to_command_from_action_pose(action_pose: List[float], current_relative_pose: List[float], mode: str) -> np.ndarray:
    x = float(action_pose[0])
    y = float(action_pose[1])
    z = float(action_pose[2])
    yaw = float(action_pose[3])
    if mode == "delta_from_pose":
        return np.array(
            [
                x - float(current_relative_pose[0]),
                y - float(current_relative_pose[1]),
                z - float(current_relative_pose[2]),
                yaw - math.radians(float(current_relative_pose[3])),
            ],
            dtype=np.float32,
        )
    return np.array([x, y, z, yaw], dtype=np.float32)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _convergence_loop(
    monitor,
    control: DroneControlClient,
    pose_manager: PoseManager,
    command_dt_s: float,
    frame_offset: int,
    step: int,
    frame_path,
    subgoal_rel_pose: List[float],
) -> Optional[Dict[str, Any]]:
    """Send zero-velocity commands while waiting for convergence VLM result."""
    monitor.request_convergence(frame_path, list(subgoal_rel_pose))
    zero_cmd = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    global_frame_idx = frame_offset + step

    while not stop_capture:
        result = monitor.poll_result()
        if result is not None:
            return {
                "action": result.action,
                "new_instruction": result.new_instruction,
                "reasoning": result.reasoning,
                "completion_pct": result.completion_pct,
            }
        control.send_command(global_frame_idx, zero_cmd)
        pose_manager.update_from_command(zero_cmd, command_dt_s)
        time.sleep(command_dt_s)

    return None


def _ask_operator_for_help(
    subgoal_nl: str,
    header: str,
    completion_pct: float,
    current_instruction: str,
    reasoning: str = "",
) -> Tuple[str, str]:
    """Prompt operator for help when the drone is stuck.

    Returns (choice, value) where choice is one of:
      "instruction"      - new low-level OpenVLA instruction (value = the instruction)
      "override_subgoal" - new subgoal text (value = subgoal)
      "replan"           - new high-level mission instruction to re-plan (value = the instruction)
      "skip"             - continue or end subgoal without changes
      "abort"            - stop the mission entirely
    """
    print(f"\n{'='*60}")
    print(f"{header}")
    print(f"  Subgoal: {subgoal_nl}")
    print(f"  Completion: {completion_pct:.0%}")
    print(f"  Current instruction: {current_instruction}")
    if reasoning:
        print(f"  Reasoning: {reasoning}")
    print(f"{'='*60}")
    print("[1] New low-level instruction (e.g. 'move forward 1m')")
    print("[2] Override current subgoal")
    print("[3] Replan from new high-level instruction")
    print("[4] Skip (continue/end subgoal)")
    print("[5] Abort mission")
    while True:
        choice = input("Choice [1/2/3/4/5]: ").strip()
        if choice == "1":
            instr = input("Instruction: ").strip()
            if not instr:
                print("Empty instruction, please try again.")
                continue
            return ("instruction", instr)
        elif choice == "2":
            subgoal = input("New subgoal: ").strip()
            if not subgoal:
                print("Empty subgoal, please try again.")
                continue
            return ("override_subgoal", subgoal)
        elif choice == "3":
            instr = input("New high-level instruction: ").strip()
            if not instr:
                print("Empty instruction, please try again.")
                continue
            return ("replan", instr)
        elif choice == "4":
            return ("skip", "")
        elif choice == "5":
            return ("abort", "")
        else:
            print("Invalid choice, please enter 1, 2, 3, 4, or 5.")


@dataclass
class OperatorHelpResult:
    """Result of prompting the operator for help."""
    stop_reason: Optional[str]  # Set if the loop should break ("operator_abort", "replan", "override_subgoal", "max_steps", "max_seconds")
    replan_instruction: str
    new_instruction: Optional[str]  # Set if choice was "instruction"
    new_subgoal: Optional[str]  # Set if choice was "override_subgoal"
    reasoning: str


def _handle_ask_help(
    control: DroneControlClient,
    frame_offset: int,
    step: int,
    subgoal_nl: str,
    header: str,
    completion_pct: float,
    current_instruction: str,
    reasoning: str = "",
) -> OperatorHelpResult:
    """Zero the drone, prompt operator, and return a structured result.

    The caller is responsible for applying instruction changes and resetting
    state (openvla origin, small_count, etc.) when new_instruction is set.
    """
    control.send_command(
        frame_offset + step,
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    logger.warning(
        "%s at step %d (completion: %.0f%%). Asking operator for help.",
        header, step, completion_pct * 100,
    )
    choice, value = _ask_operator_for_help(
        subgoal_nl, header, completion_pct, current_instruction, reasoning,
    )
    logger.info(
        "Operator chose '%s' at step %d (value: %s)",
        choice, step, repr(value) if value else "(none)",
    )
    if choice == "abort":
        logger.info("Operator aborted mission at step %d.", step)
        return OperatorHelpResult(
            stop_reason="operator_abort", replan_instruction="",
            new_instruction=None, new_subgoal=None, reasoning=reasoning,
        )
    elif choice == "replan":
        logger.info(
            "Operator requesting full replan with new instruction: '%s'", value,
        )
        return OperatorHelpResult(
            stop_reason="replan", replan_instruction=value,
            new_instruction=None, new_subgoal=None, reasoning=reasoning,
        )
    elif choice == "override_subgoal":
        logger.info(
            "Operator overriding subgoal: '%s' -> '%s'", subgoal_nl, value,
        )
        return OperatorHelpResult(
            stop_reason="override_subgoal", replan_instruction="",
            new_instruction=None, new_subgoal=value, reasoning=reasoning,
        )
    elif choice == "instruction":
        logger.info(
            "Operator correction: '%s' -> '%s' (subgoal unchanged: '%s')",
            current_instruction, value, subgoal_nl,
        )
        return OperatorHelpResult(
            stop_reason=None, replan_instruction="",
            new_instruction=value, new_subgoal=None, reasoning=reasoning,
        )
    else:
        logger.info("Operator skipped at step %d.", step)
        return OperatorHelpResult(
            stop_reason=header.lower().replace(" ", "_").strip("_"),
            replan_instruction="", new_instruction=None,
            new_subgoal=None, reasoning=reasoning,
        )


def run_subgoal(
    subgoal_nl: str,
    subgoal_index: int,
    run_dir: Path,
    frames_dir: Path,
    openvla: OpenVLAClient,
    camera: ThreadedCamera,
    control: DroneControlClient,
    pose_manager: PoseManager,
    monitor_model: str,
    llm_model: str,
    check_interval: int,
    max_steps: int,
    max_corrections: int,
    frame_offset: int,
    command_dt_s: float,
    action_pose_mode: str,
    trajectory_log: List[Dict[str, Any]],
    check_interval_s: Optional[float] = None,
    max_seconds: Optional[float] = None,
    stall_window: int = 10,
    stall_threshold: float = 0.05,
    stall_completion_floor: float = 0.8,
    constraints: Optional[List[Any]] = None,
    recorder: Optional["FrameRecorder"] = None,
) -> Dict[str, Any]:
    """Execute a single subgoal with OpenVLA, goal adherence monitoring, and operator help.

    Converts the subgoal to an OpenVLA instruction, then runs a frame loop
    that calls OpenVLA for actions and the goal adherence monitor for progress checks.
    The drone pauses and prompts the operator when:
      - Checkpoint stall is detected (completion plateau).
      - The convergence correction budget is exhausted.
      - The step budget (max_steps) or time budget (max_seconds) is reached.

    The operator can provide a new low-level instruction, request a full
    mission replan, skip, or abort. On replan the function returns with
    stop_reason="replan" and replan_instruction set.

    Returns a dict with: subgoal, converted_instruction, total_steps,
    stop_reason, corrections_used, last_completion_pct, peak_completion,
    vlm_calls, next_origin, replan_instruction.
    """
    from rvln.ai.goal_adherence_monitor import DiaryCheckResult, GoalAdherenceMonitor
    from rvln.ai.subgoal_converter import SubgoalConverter

    converter = SubgoalConverter(model=llm_model)
    conversion = converter.convert(subgoal_nl)
    converted_instruction = conversion.instruction
    current_instruction = converted_instruction

    subgoal_dir = run_dir / f"subgoal_{subgoal_index:02d}_{sanitize_name(subgoal_nl)}"
    diary_artifacts = subgoal_dir / "diary_artifacts"
    diary_artifacts.mkdir(parents=True, exist_ok=True)
    monitor = GoalAdherenceMonitor(
        subgoal=subgoal_nl,
        check_interval=check_interval,
        model=monitor_model,
        artifacts_dir=diary_artifacts,
        max_corrections=max_corrections,
        check_interval_s=check_interval_s,
        stall_window=stall_window,
        stall_threshold=stall_threshold,
        stall_completion_floor=stall_completion_floor,
        constraints=constraints,
    )

    openvla.reset_model()

    origin_world = pose_manager.get_world_pose()
    openvla_pose_origin = [0.0, 0.0, 0.0, 0.0]
    last_pose = None
    small_count = 0
    last_correction_step = -check_interval
    stop_reason = "max_steps"
    total_steps = 0
    replan_instruction = ""
    new_subgoal_override = ""
    override_history: List[Dict[str, Any]] = []

    use_async = check_interval_s is not None
    last_correction_time = time.time() if use_async else None
    subgoal_start_time = time.time()
    subgoal_rel_pose = [0.0, 0.0, 0.0, 0.0]
    frame_path = None

    step_base = 0
    try:
        while True:
            for step in range(step_base, step_base + max_steps):
                # 1. Check stop_capture
                if stop_capture:
                    stop_reason = "interrupted"
                    break

                # 1b. Check max_seconds (time-based budget for async mode)
                if max_seconds is not None and (time.time() - subgoal_start_time) >= max_seconds:
                    total_steps = step
                    elapsed = time.time() - subgoal_start_time
                    help_result = _handle_ask_help(
                        control, frame_offset, step, subgoal_nl,
                        f"MAX TIME REACHED ({elapsed:.1f}s)",
                        monitor.last_completion_pct, current_instruction,
                    )
                    if help_result.stop_reason:
                        stop_reason = help_result.stop_reason
                        replan_instruction = help_result.replan_instruction
                        new_subgoal_override = help_result.new_subgoal or ""
                        total_steps = step
                        break
                    override_history.append({
                        "step": step,
                        "type": "operator_help",
                        "old_instruction": current_instruction,
                        "new_instruction": help_result.new_instruction,
                        "reasoning": help_result.reasoning,
                    })
                    current_instruction = help_result.new_instruction
                    openvla_pose_origin = list(subgoal_rel_pose)
                    small_count = 0
                    last_pose = None
                    if use_async:
                        last_correction_time = time.time()
                    last_correction_step = step
                    subgoal_start_time = time.time()
                    openvla.reset_model()
                    continue

                # 2. Async mode: poll for pending monitor results
                if use_async:
                    async_result = monitor.poll_result()
                    if async_result is not None:
                        if async_result.action == "stop":
                            stop_reason = "monitor_complete"
                            total_steps = step
                            break
                        if async_result.action == "force_converge":
                            override_history.append({
                                "step": step,
                                "type": "force_converge",
                                "reasoning": async_result.reasoning,
                            })
                            conv_dict = _convergence_loop(
                                monitor, control, pose_manager, command_dt_s,
                                frame_offset, step, frame_path, subgoal_rel_pose,
                            )
                            if conv_dict is None:
                                stop_reason = "interrupted"
                                break
                            if conv_dict["action"] == "stop":
                                stop_reason = "monitor_complete"
                                break
                            if conv_dict["action"] == "ask_help":
                                help_result = _handle_ask_help(
                                    control, frame_offset, step, subgoal_nl,
                                    "MAX CORRECTIONS REACHED", conv_dict["completion_pct"],
                                    current_instruction, conv_dict["reasoning"],
                                )
                                if help_result.stop_reason:
                                    stop_reason = help_result.stop_reason
                                    replan_instruction = help_result.replan_instruction
                                    new_subgoal_override = help_result.new_subgoal or ""
                                    total_steps = step
                                    break
                                override_history.append({
                                    "step": step,
                                    "type": "operator_help",
                                    "old_instruction": current_instruction,
                                    "new_instruction": help_result.new_instruction,
                                    "reasoning": help_result.reasoning,
                                })
                                current_instruction = help_result.new_instruction
                                openvla_pose_origin = list(subgoal_rel_pose)
                                small_count = 0
                                last_pose = None
                                last_correction_time = time.time()
                                last_correction_step = step
                                openvla.reset_model()
                            elif conv_dict.get("new_instruction"):
                                override_history.append({
                                    "step": step,
                                    "type": f"convergence_{conv_dict['action']}",
                                    "old_instruction": current_instruction,
                                    "new_instruction": conv_dict["new_instruction"],
                                    "reasoning": conv_dict["reasoning"],
                                })
                                current_instruction = conv_dict["new_instruction"]
                                openvla_pose_origin = list(subgoal_rel_pose)
                                small_count = 0
                                last_pose = None
                                last_correction_time = time.time()
                                last_correction_step = step
                                openvla.reset_model()
                            else:
                                stop_reason = "convergence_no_command"
                                break

                        if async_result.action == "ask_help":
                            help_result = _handle_ask_help(
                                control, frame_offset, step, subgoal_nl,
                                "STALL DETECTED", async_result.completion_pct,
                                current_instruction, async_result.reasoning,
                            )
                            if help_result.stop_reason:
                                stop_reason = help_result.stop_reason
                                replan_instruction = help_result.replan_instruction
                                new_subgoal_override = help_result.new_subgoal or ""
                                total_steps = step
                                break
                            override_history.append({
                                "step": step,
                                "type": "operator_help",
                                "old_instruction": current_instruction,
                                "new_instruction": help_result.new_instruction,
                                "reasoning": help_result.reasoning,
                            })
                            current_instruction = help_result.new_instruction
                            openvla_pose_origin = list(subgoal_rel_pose)
                            small_count = 0
                            last_pose = None
                            last_correction_time = time.time()
                            last_correction_step = step
                            openvla.reset_model()

                # 3. Grab frame, compute pose
                ok, frame = camera.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                world_pose = pose_manager.get_world_pose()
                subgoal_rel_pose = relative_pose(world_pose, origin_world)

                global_frame_idx = frame_offset + step
                frame_path = frames_dir / f"frame_{global_frame_idx:06d}.png"
                cv2.imwrite(str(frame_path), frame)

                # 4. Call monitor.on_frame
                result = monitor.on_frame(frame_path, displacement=list(subgoal_rel_pose))

                if not use_async:
                    # Sync mode: on_frame may return "stop" or "force_converge"
                    if result.action == "stop":
                        stop_reason = "monitor_complete"
                        total_steps = step
                        break

                    if result.action == "force_converge":
                        override_history.append({
                            "step": step,
                            "type": "force_converge",
                            "reasoning": result.reasoning,
                        })

                    if result.action == "ask_help":
                        help_result = _handle_ask_help(
                            control, frame_offset, step, subgoal_nl,
                            "STALL DETECTED", result.completion_pct,
                            current_instruction, result.reasoning,
                        )
                        if help_result.stop_reason:
                            stop_reason = help_result.stop_reason
                            replan_instruction = help_result.replan_instruction
                            new_subgoal_override = help_result.new_subgoal or ""
                            total_steps = step
                            break
                        override_history.append({
                            "step": step,
                            "type": "operator_help",
                            "old_instruction": current_instruction,
                            "new_instruction": help_result.new_instruction,
                            "reasoning": help_result.reasoning,
                        })
                        current_instruction = help_result.new_instruction
                        openvla_pose_origin = list(subgoal_rel_pose)
                        small_count = 0
                        last_pose = None
                        last_correction_step = step
                        openvla.reset_model()

                # 5. openvla.predict() and send commands (identical for both modes)
                openvla_pose = [c - o for c, o in zip(subgoal_rel_pose, openvla_pose_origin)]
                response = openvla.predict(
                    image_bgr=frame,
                    proprio=state_for_openvla(openvla_pose),
                    instr=current_instruction.strip().lower(),
                )

                action_poses = response.get("action")
                if not isinstance(action_poses, list) or len(action_poses) == 0:
                    stop_reason = "empty_action"
                    total_steps = step
                    break

                if recorder is not None and recorder.enabled:
                    recorder.maybe_log(
                        step=step,
                        subgoal_index=subgoal_index,
                        subgoal_nl=subgoal_nl,
                        current_instruction=current_instruction,
                        frame_path=frame_path,
                        world_pose=list(world_pose),
                        subgoal_rel_pose=list(subgoal_rel_pose),
                        openvla_action=action_poses,
                        converter=converter,
                        monitor=monitor,
                    )

                for action_pose in action_poses:
                    if not (isinstance(action_pose, (list, tuple)) and len(action_pose) >= 4):
                        continue
                    current_world = pose_manager.get_world_pose()
                    current_rel = relative_pose(current_world, origin_world)
                    cmd = to_command_from_action_pose(action_pose, current_rel, action_pose_mode)
                    control.send_command(global_frame_idx, cmd)
                    pose_manager.update_from_command(cmd, command_dt_s)
                    updated_rel = relative_pose(pose_manager.get_world_pose(), origin_world)
                    trajectory_log.append({
                        "state": [
                            [updated_rel[0], updated_rel[1], updated_rel[2]],
                            [0, updated_rel[3], 0],
                        ]
                    })
                    time.sleep(command_dt_s)

                # 6. Convergence detection
                total_steps = step + 1
                world_pose = pose_manager.get_world_pose()
                subgoal_rel_pose = relative_pose(world_pose, origin_world)

                if use_async:
                    # Async mode: time-based convergence guard
                    converged = False
                    elapsed_since_correction = time.time() - last_correction_time
                    if last_pose is not None and elapsed_since_correction >= check_interval_s:
                        diffs = [abs(a - b) for a, b in zip(subgoal_rel_pose, last_pose)]
                        if all(d < ACTION_SMALL_DELTA_POS for d in diffs[:3]) and diffs[3] < ACTION_SMALL_DELTA_YAW:
                            small_count += 1
                        else:
                            small_count = 0
                        if small_count >= ACTION_SMALL_STEPS:
                            converged = True
                    last_pose = list(subgoal_rel_pose)

                    if converged:
                        conv_dict = _convergence_loop(
                            monitor, control, pose_manager, command_dt_s,
                            frame_offset, step, frame_path, subgoal_rel_pose,
                        )
                        if conv_dict is None:
                            stop_reason = "interrupted"
                            break
                        if conv_dict["action"] == "stop":
                            stop_reason = "monitor_complete"
                            break
                        if conv_dict["action"] == "ask_help":
                            help_result = _handle_ask_help(
                                control, frame_offset, step, subgoal_nl,
                                "MAX CORRECTIONS REACHED", conv_dict["completion_pct"],
                                current_instruction, conv_dict["reasoning"],
                            )
                            if help_result.stop_reason:
                                stop_reason = help_result.stop_reason
                                replan_instruction = help_result.replan_instruction
                                new_subgoal_override = help_result.new_subgoal or ""
                                total_steps = step
                                break
                            override_history.append({
                                "step": step,
                                "type": "operator_help",
                                "old_instruction": current_instruction,
                                "new_instruction": help_result.new_instruction,
                                "reasoning": help_result.reasoning,
                            })
                            current_instruction = help_result.new_instruction
                            openvla_pose_origin = list(subgoal_rel_pose)
                            small_count = 0
                            last_pose = None
                            last_correction_time = time.time()
                            last_correction_step = step
                            openvla.reset_model()
                        elif conv_dict.get("new_instruction"):
                            override_history.append({
                                "step": step,
                                "type": f"convergence_{conv_dict['action']}",
                                "old_instruction": current_instruction,
                                "new_instruction": conv_dict["new_instruction"],
                                "reasoning": conv_dict["reasoning"],
                            })
                            current_instruction = conv_dict["new_instruction"]
                            openvla_pose_origin = list(subgoal_rel_pose)
                            small_count = 0
                            last_pose = None
                            last_correction_time = time.time()
                            last_correction_step = step
                            openvla.reset_model()
                        else:
                            stop_reason = "convergence_no_command"
                            break
                else:
                    # Sync mode: step-based convergence guard (original behavior)
                    converged = result.action == "force_converge"
                    steps_since_correction = step - last_correction_step

                    if last_pose is not None and steps_since_correction >= check_interval:
                        diffs = [abs(a - b) for a, b in zip(subgoal_rel_pose, last_pose)]
                        if all(d < ACTION_SMALL_DELTA_POS for d in diffs[:3]) and diffs[3] < ACTION_SMALL_DELTA_YAW:
                            small_count += 1
                        else:
                            small_count = 0
                        if small_count >= ACTION_SMALL_STEPS:
                            converged = True
                    last_pose = list(subgoal_rel_pose)

                    if converged:
                        conv_result = monitor.on_convergence(
                            frame_path, displacement=list(subgoal_rel_pose)
                        )

                        if conv_result.action == "stop":
                            stop_reason = "monitor_complete"
                            break

                        if conv_result.action == "ask_help":
                            help_result = _handle_ask_help(
                                control, frame_offset, step, subgoal_nl,
                                "MAX CORRECTIONS REACHED", conv_result.completion_pct,
                                current_instruction, conv_result.reasoning,
                            )
                            if help_result.stop_reason:
                                stop_reason = help_result.stop_reason
                                replan_instruction = help_result.replan_instruction
                                new_subgoal_override = help_result.new_subgoal or ""
                                total_steps = step
                                break
                            override_history.append({
                                "step": step,
                                "type": "operator_help",
                                "old_instruction": current_instruction,
                                "new_instruction": help_result.new_instruction,
                                "reasoning": help_result.reasoning,
                            })
                            current_instruction = help_result.new_instruction
                            openvla_pose_origin = list(subgoal_rel_pose)
                            small_count = 0
                            last_pose = None
                            last_correction_step = step
                            openvla.reset_model()
                        elif conv_result.new_instruction:
                            override_history.append({
                                "step": step,
                                "type": f"convergence_{conv_result.action}",
                                "old_instruction": current_instruction,
                                "new_instruction": conv_result.new_instruction,
                                "reasoning": conv_result.reasoning,
                            })
                            current_instruction = conv_result.new_instruction
                            openvla_pose_origin = list(subgoal_rel_pose)
                            small_count = 0
                            last_pose = None
                            last_correction_step = step
                            openvla.reset_model()
                        else:
                            stop_reason = "convergence_no_command"
                            break
            else:
                total_steps = step + 1
                help_result = _handle_ask_help(
                    control, frame_offset, step, subgoal_nl,
                    "MAX STEPS REACHED", monitor.last_completion_pct,
                    current_instruction, f"Max steps ({total_steps}) reached.",
                )
                if help_result.stop_reason:
                    stop_reason = help_result.stop_reason
                    replan_instruction = help_result.replan_instruction
                    new_subgoal_override = help_result.new_subgoal or ""
                    total_steps = step + 1
                    break
                override_history.append({
                    "step": step,
                    "type": "operator_help",
                    "old_instruction": current_instruction,
                    "new_instruction": help_result.new_instruction,
                    "reasoning": help_result.reasoning,
                })
                current_instruction = help_result.new_instruction
                openvla_pose_origin = list(subgoal_rel_pose)
                small_count = 0
                last_pose = None
                if use_async:
                    last_correction_time = time.time()
                last_correction_step = step
                step_base = step + 1
                openvla.reset_model()
                continue
            break
    except Exception as exc:
        stop_reason = f"error: {exc}"
        logger.error("run_subgoal failed at step %d: %s", total_steps, exc)

    all_llm_records = converter.llm_call_records + monitor.vlm_rtts
    write_json(
        subgoal_dir / "diary_summary.json",
        {
            "subgoal": subgoal_nl,
            "converted_instruction": converted_instruction,
            "diary": monitor.diary,
            "override_history": override_history,
            "corrections_used": monitor.corrections_used,
            "last_completion_pct": monitor.last_completion_pct,
            "peak_completion": monitor.peak_completion,
            "parse_failures": monitor.parse_failures,
            "vlm_calls": monitor.vlm_calls,
            "vlm_rtts": monitor.vlm_rtts,
            "llm_call_records": all_llm_records,
            "stop_reason": stop_reason,
            "total_steps": total_steps,
        },
    )
    monitor.cleanup()

    next_world_origin = pose_manager.get_world_pose()
    return {
        "subgoal": subgoal_nl,
        "converted_instruction": converted_instruction,
        "total_steps": total_steps,
        "stop_reason": stop_reason,
        "corrections_used": monitor.corrections_used,
        "last_completion_pct": monitor.last_completion_pct,
        "peak_completion": monitor.peak_completion,
        "vlm_calls": monitor.vlm_calls,
        "vlm_rtts": monitor.vlm_rtts,
        "llm_call_records": all_llm_records,
        "next_origin": list(next_world_origin),
        "replan_instruction": replan_instruction,
        "new_subgoal_override": new_subgoal_override,
    }


def parse_args() -> argparse.Namespace:
    description = (
        "Drive a real (or mocked) MiniNav drone with the OpenVLA action "
        "server, an LTL planner, and a live goal-adherence monitor.\n\n"
        "A pose source is required: either external odometry "
        "(--odom_http_url or --odom_udp_port) or --dead-reckoning."
    )

    epilog = (
        "Examples:\n"
        "  # Live flight (USB camera 0, control server at 192.168.0.101:8080)\n"
        "  python scripts/run_hardware.py \\\n"
        "      --instruction \"take off and circle the red cone\" \\\n"
        "      --odom_http_url http://192.168.0.101:8090/pose\n"
        "\n"
        "  # Fully simulated (start_mock_hardware.py + start_server.py running)\n"
        "  python scripts/run_hardware.py \\\n"
        "      --preferred_server_host 127.0.0.1 \\\n"
        "      --control_port 8080 \\\n"
        "      --camera_url http://127.0.0.1:8081/frame \\\n"
        "      --odom_http_url http://127.0.0.1:8081/pose \\\n"
        "      --openvla_predict_url http://127.0.0.1:5007/predict \\\n"
        "      --initial_position 0,0,0,0 \\\n"
        "      --command_is_velocity \\\n"
        "      --action_pose_mode delta_from_pose \\\n"
        "      --instruction \"move forward 10m, then turn toward the red car\"\n"
        "\n"
        "  # Same, with a 5fps recording for cost analysis\n"
        "  python scripts/run_hardware.py ... --record --record_fps 5\n"
        "\n"
        "Outputs (under --results_dir/run_<YYYY_MM_DD_HH_MM_SS_us>/):\n"
        "  trajectory_log.json  per-step state vector\n"
        "  run_info.json        full run summary (LTL plan, subgoal results, token totals)\n"
        "  subgoal_<NN>_*/      per-subgoal artifacts (diary, monitor reasoning)\n"
        "  frames/              camera frames (only if --record)\n"
        "  recording_log.jsonl  one entry per recorded frame (only if --record)\n"
    )

    parser = argparse.ArgumentParser(
        prog="run_hardware.py",
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g_task = parser.add_argument_group("Task")
    g_task.add_argument(
        "--instruction", type=str, default=None,
        help=(
            "Natural-language mission instruction (e.g. 'take off and circle "
            "the red cone'). If omitted, you'll be prompted on stdin."
        ),
    )
    g_task.add_argument(
        "--initial_position", type=str, default="0,0,0,0",
        help=(
            "Starting world pose as 'x,y,z,yaw_deg'. The LTL planner and "
            "dead-reckoning provider use this as their origin. "
            "(default: %(default)s)"
        ),
    )
    g_task.add_argument(
        "--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR),
        help="Root directory for run artifacts. (default: %(default)s)",
    )

    g_camera = parser.add_argument_group("Camera")
    g_camera.add_argument(
        "--camera", type=int, default=0,
        help=(
            "Local cv2 device index (only used when --camera_url is unset). "
            "(default: %(default)s)"
        ),
    )
    g_camera.add_argument(
        "--camera_url", type=str, default=None,
        help=(
            "HTTP URL serving JPEG/PNG frames (e.g. http://127.0.0.1:8081/frame). "
            "When set, replaces the local cv2 capture with an HTTP-pull camera. "
            "Use this to test against the simulated MiniNav frame feed."
        ),
    )
    g_camera.add_argument(
        "--fps", type=int, default=DEFAULT_CAMERA_FPS,
        help="Target capture rate. (default: %(default)s)",
    )
    g_camera.add_argument(
        "--camera_retries", type=int, default=DEFAULT_CAMERA_RETRIES,
        help="Reopen attempts after the cv2 device drops. (default: %(default)s)",
    )
    g_camera.add_argument(
        "--camera_init_timeout", type=float, default=DEFAULT_CAMERA_INIT_TIMEOUT,
        help="Seconds to wait for the first frame before bailing out. (default: %(default)s)",
    )

    g_record = parser.add_argument_group("Recording")
    g_record.add_argument(
        "--record", action="store_true",
        help=(
            "Persist camera frames under run_dir/frames and write "
            "recording_log.jsonl with per-frame metadata (subgoal, LTL "
            "formula, low-level OpenVLA instruction, action tensor, "
            "cumulative input/output tokens). When off, frames are written "
            "to a temp dir and discarded at run end (the goal-adherence "
            "monitor still gets them)."
        ),
    )
    g_record.add_argument(
        "--record_fps", type=float, default=DEFAULT_CAMERA_FPS,
        help=(
            "When --record is set, throttle recording_log.jsonl entries "
            "to at most this many per second. Defaults to the camera FPS "
            "(one entry per step). Lower this to keep the log compact "
            "for long runs. (default: %(default)s)"
        ),
    )

    g_control = parser.add_argument_group("Control server (boieng wire)")
    g_control.add_argument(
        "--preferred_server_host", type=str, default=DEFAULT_PREFERRED_SERVER_HOST,
        help=(
            "Tried first; if unreachable in 0.75s, falls back to this "
            "machine's LAN IP. Set to 127.0.0.1 for local mock. "
            "(default: %(default)s)"
        ),
    )
    g_control.add_argument(
        "--control_port", type=int, default=DEFAULT_CONTROL_PORT,
        help="TCP port the drone control server listens on. (default: %(default)s)",
    )
    g_control.add_argument(
        "--control_retries", type=int, default=DEFAULT_CONTROL_RETRIES,
        help="Connect attempts before giving up. (default: %(default)s)",
    )
    g_control.add_argument(
        "--control_retry_sleep", type=float, default=DEFAULT_CONTROL_RETRY_SLEEP,
        help="Seconds between connect attempts. (default: %(default)s)",
    )
    g_control.add_argument(
        "--command_dt_s", type=float, default=DEFAULT_COMMAND_DT_S,
        help=(
            "Time interval between commands sent to the control server. "
            "Also the dt used for dead-reckoning integration. "
            "(default: %(default)s)"
        ),
    )
    g_control.add_argument(
        "--action_pose_mode", choices=["direct", "delta_from_pose"], default="direct",
        help=(
            "How to convert each OpenVLA action pose into a 4-float boieng "
            "command. 'direct': send the action pose verbatim (live drone, "
            "absolute pose). 'delta_from_pose': send (action - current "
            "world pose), used by the simulated mock_server. "
            "(default: %(default)s)"
        ),
    )
    g_control.add_argument(
        "--command_is_velocity", action="store_true",
        help=(
            "Treat each sent command as a velocity (m/s, rad/s) rather than "
            "an absolute pose. Required for the simulated mock_server (which "
            "integrates incoming commands as velocities) and for sane "
            "dead-reckoning."
        ),
    )

    g_openvla = parser.add_argument_group("OpenVLA")
    g_openvla.add_argument(
        "--openvla_predict_url", type=str, default=DEFAULT_OPENVLA_PREDICT_URL,
        help=(
            "Full URL of the OpenVLA /predict endpoint. /reset is derived "
            "from this URL by replacing the trailing path component; if "
            "the server has no /reset (e.g. raw OpenVLA without "
            "scripts/start_server.py's patch) it's silently skipped. "
            "(default: %(default)s)"
        ),
    )

    g_planner = parser.add_argument_group("LTL planner & subgoal converter")
    g_planner.add_argument(
        "--llm_model", type=str, default=DEFAULT_LLM_MODEL,
        help=(
            "LLM used for LTL natural-language planning and subgoal -> "
            "OpenVLA-instruction conversion. (default: %(default)s)"
        ),
    )
    g_planner.add_argument(
        "--llm_fallback_model", type=str, default=DEFAULT_LLM_FALLBACK_MODEL,
        help=(
            "Used if the primary LLM is unreachable. Set to '' to disable "
            "fallback. (default: %(default)s)"
        ),
    )

    g_monitor = parser.add_argument_group("Goal-adherence monitor (VLM)")
    g_monitor.add_argument(
        "--monitor_model", type=str, default=DEFAULT_VLM_MODEL,
        help="VLM that scores progress and detects stalls. (default: %(default)s)",
    )
    g_monitor.add_argument(
        "--monitor_fallback_model", type=str, default=DEFAULT_VLM_FALLBACK_MODEL,
        help="VLM fallback model. (default: %(default)s)",
    )
    g_monitor.add_argument(
        "--diary-mode", choices=("frame", "time"), default=DEFAULT_DIARY_MODE,
        help=(
            "Checkpoint cadence: 'frame' (sync, every N control steps) or "
            "'time' (async, every N seconds, monitor runs in a thread). "
            "(default: %(default)s)"
        ),
    )
    g_monitor.add_argument(
        "--diary_check_interval", type=int, default=DEFAULT_DIARY_CHECK_INTERVAL,
        help=(
            "Frame mode: invoke the monitor every N steps. "
            "(default: %(default)s)"
        ),
    )
    g_monitor.add_argument(
        "--diary_check_interval_s", type=float, default=DEFAULT_DIARY_CHECK_INTERVAL_S,
        help=(
            "Time mode: target seconds between monitor checkpoints. "
            "(default: %(default)s)"
        ),
    )
    g_monitor.add_argument(
        "--max_steps_per_subgoal", type=int, default=DEFAULT_MAX_STEPS_PER_SUBGOAL,
        help=(
            "Hard step budget per subgoal; on exhaustion the operator is "
            "prompted for help. (default: %(default)s)"
        ),
    )
    g_monitor.add_argument(
        "--max_seconds_per_subgoal", type=float, default=DEFAULT_MAX_SECONDS_PER_SUBGOAL,
        help=(
            "Time budget per subgoal in seconds (time mode only). "
            "(default: %(default)s)"
        ),
    )
    g_monitor.add_argument(
        "--max_corrections", type=int, default=DEFAULT_MAX_CORRECTIONS,
        help=(
            "Operator-help requests allowed per subgoal before the run "
            "aborts. (default: %(default)s)"
        ),
    )
    g_monitor.add_argument(
        "--stall_window", type=int, default=DEFAULT_STALL_WINDOW,
        help=(
            "Number of consecutive checkpoints with flat completion needed "
            "to trigger a stall help request. (default: %(default)s)"
        ),
    )
    g_monitor.add_argument(
        "--stall_threshold", type=float, default=DEFAULT_STALL_THRESHOLD,
        help=(
            "Max completion delta across stall_window checkpoints to count "
            "as stalled. (default: %(default)s)"
        ),
    )
    g_monitor.add_argument(
        "--stall_completion_floor", type=float, default=DEFAULT_STALL_COMPLETION_FLOOR,
        help=(
            "Don't trigger stall detection above this completion fraction. "
            "(default: %(default)s)"
        ),
    )

    g_pose = parser.add_argument_group(
        "Pose source (one of --odom_* or --dead-reckoning is required)",
    )
    g_pose.add_argument(
        "--odom_http_url", type=str, default=None,
        help=(
            "HTTP endpoint returning JSON {x,y,z,yaw}. Polled each control "
            "step. Mutually exclusive with --dead-reckoning."
        ),
    )
    g_pose.add_argument(
        "--odom_udp_host", type=str, default=DEFAULT_ODOM_UDP_HOST,
        help="UDP bind host for streamed odometry. (default: %(default)s)",
    )
    g_pose.add_argument(
        "--odom_udp_port", type=int, default=DEFAULT_ODOM_UDP_PORT,
        help=(
            "UDP port for streamed odometry. 0 disables UDP. "
            "(default: %(default)s)"
        ),
    )
    g_pose.add_argument(
        "--odom_stale_timeout_s", type=float, default=DEFAULT_ODOM_STALE_TIMEOUT_S,
        help=(
            "Seconds without a fresh odometry sample before falling back to "
            "dead-reckoning. (default: %(default)s)"
        ),
    )
    g_pose.add_argument(
        "--odom_poll_hz", type=float, default=50.0,
        help=(
            "Background polling rate for the odometry provider (Hz). The "
            "control loop reads a cached value, so this just bounds the "
            "freshness of that cache. (default: %(default)s)"
        ),
    )
    g_pose.add_argument(
        "--dead-reckoning", action="store_true", default=False,
        help=(
            "Estimate world pose by integrating sent commands instead of "
            "using external odometry. Drift accumulates; pair with "
            "--command_is_velocity for sane integration."
        ),
    )

    g_misc = parser.add_argument_group("Misc")
    g_misc.add_argument(
        "--extra-env-file", type=str, default=None,
        help=(
            "Optional env file loaded after .env and .env.local (overrides). "
            "Useful for per-flight API keys or hostnames."
        ),
    )
    g_misc.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging verbosity. (default: %(default)s)",
    )

    return parser.parse_args()


def main() -> None:
    global stop_capture
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    # Register interrupt handlers here (not at import time) so importing this
    # module from tests/notebooks doesn't hijack SIGINT/SIGTERM globally.
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    load_env_vars(args.extra_env_file)
    initial_world_pose = parse_position(args.initial_position)
    instruction = args.instruction or input("Enter initial instruction: ").strip()
    if not instruction:
        raise SystemExit("Instruction is required.")

    # Validate pose-source flags before opening camera/control sockets so a bad
    # CLI doesn't leave the cv2 capture or TCP socket dangling on SystemExit.
    has_odom = args.odom_http_url or args.odom_udp_port > 0
    has_dr = args.dead_reckoning
    if has_odom and has_dr:
        raise SystemExit("Cannot use both --dead-reckoning and external odometry (--odom_http_url / --odom_udp_port).")
    if not has_odom and not has_dr:
        raise SystemExit(
            "No pose source specified. Use --odom_http_url / --odom_udp_port for "
            "external odometry, or --dead-reckoning for estimated poses."
        )

    # Surface flag combinations that look inconsistent. We warn rather than
    # fail so live-hardware operators (the priority path) can override if they
    # know what they're doing, but the misconfiguration is loud.
    if has_dr and args.action_pose_mode == "direct" and not args.command_is_velocity:
        logger.warning(
            "Primary --dead-reckoning with --action_pose_mode=direct is not "
            "physically meaningful (the wire command is an absolute target, "
            "not a delta or velocity). DR will hold the initial pose and not "
            "advance until odometry is provided."
        )
    if args.action_pose_mode == "delta_from_pose" and not args.command_is_velocity and not has_odom:
        logger.warning(
            "Sim-style --action_pose_mode=delta_from_pose typically pairs "
            "with --command_is_velocity for a usable DR fallback."
        )
    if args.action_pose_mode == "direct" and args.command_is_velocity:
        logger.warning(
            "--action_pose_mode=direct with --command_is_velocity is unusual: "
            "direct mode sends absolute target poses, but the velocity flag "
            "implies the receiver integrates them as rates. Double-check "
            "your control server expects this."
        )

    llm_model = resolve_model_with_fallback(args.llm_model, args.llm_fallback_model)
    monitor_model = resolve_model_with_fallback(args.monitor_model, args.monitor_fallback_model)

    # Microseconds avoid run-dir collisions when two pipelines launch in the
    # same second.
    run_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    run_dir = Path(args.results_dir) / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Frames must always be on disk for the goal-adherence monitor's VLM
    # calls. With --record they live under run_dir/frames; without --record
    # they go to a private temp dir cleaned up at exit.
    if args.record:
        frames_dir = run_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        recording_log_path: Optional[Path] = run_dir / "recording_log.jsonl"
        frames_tempdir: Optional[str] = None
    else:
        frames_tempdir = tempfile.mkdtemp(prefix="rvln_frames_")
        frames_dir = Path(frames_tempdir)
        recording_log_path = None

    if args.camera_url:
        logger.info("Camera source: HTTP feed at %s", args.camera_url)
        camera = HttpFrameCamera(
            url=args.camera_url,
            fps=args.fps,
            init_timeout=args.camera_init_timeout,
        )
    else:
        camera = ThreadedCamera(
            src=args.camera,
            fps=args.fps,
            max_reopen_attempts=args.camera_retries,
            init_timeout=args.camera_init_timeout,
        )
    if not camera.wait_until_ready():
        reason = camera.failure_reason or "camera failed to initialize"
        raise RuntimeError(reason)

    control_host, control_port = resolve_server_address(
        args.preferred_server_host, args.control_port
    )
    control = DroneControlClient(
        host=control_host,
        port=control_port,
        connect_retries=args.control_retries,
        retry_sleep_s=args.control_retry_sleep,
    )
    control.connect()

    odom_provider = None
    if has_odom:
        odom_provider = OdometryPoseProvider(
            http_url=args.odom_http_url,
            udp_host=args.odom_udp_host,
            udp_port=args.odom_udp_port,
            stale_timeout_s=args.odom_stale_timeout_s,
            poll_hz=args.odom_poll_hz,
        )

    if has_dr:
        print("\033[91mWARNING: Dead-reckoning mode active. Pose will drift over time.\033[0m")

    dr_provider = DeadReckoningPoseProvider(
        initial_world_pose=initial_world_pose,
        command_is_velocity=args.command_is_velocity,
        action_pose_mode=args.action_pose_mode,
    )
    pose_manager = PoseManager(odom=odom_provider, dr=dr_provider)

    openvla = OpenVLAClient(predict_url=args.openvla_predict_url)

    start_ts = datetime.now().isoformat()
    trajectory_log: List[Dict[str, Any]] = []
    subgoal_summaries: List[Dict[str, Any]] = []
    frame_offset = 0
    ltl_plan: Dict[str, Any] = {}
    llm_interface = None
    recorder_ctx: Dict[str, Any] = {"llm_interface": None, "ltl_plan": ltl_plan}
    recorder = FrameRecorder(
        log_path=recording_log_path,
        record_fps=args.record_fps,
        ctx=recorder_ctx,
    )

    try:
        from rvln.ai.llm_interface import LLMUserInterface
        from rvln.ai.ltl_planner import LTLSymbolicPlanner

        llm_interface = LLMUserInterface(model=llm_model)
        recorder_ctx["llm_interface"] = llm_interface
        planner = LTLSymbolicPlanner(llm_interface)
        planner.plan_from_natural_language(instruction)

        ltl_plan = {
            "ltl_nl_formula": llm_interface.ltl_nl_formula.get("ltl_nl_formula", ""),
            "pi_predicates": dict(planner.pi_map),
        }
        recorder_ctx["ltl_plan"] = ltl_plan

        use_time_mode = args.diary_mode == "time"
        check_interval_s = args.diary_check_interval_s if use_time_mode else None
        max_seconds = args.max_seconds_per_subgoal if use_time_mode else None

        current_subgoal = planner.get_next_predicate()
        subgoal_index = 0
        while current_subgoal is not None and not stop_capture:
            subgoal_index += 1

            active_constraints = planner.get_active_constraints()
            if active_constraints:
                logger.info(
                    "Active constraints for subgoal %d: %s",
                    subgoal_index, active_constraints,
                )

            logger.info("Running subgoal %d: %s", subgoal_index, current_subgoal)
            result = run_subgoal(
                subgoal_nl=current_subgoal,
                subgoal_index=subgoal_index,
                run_dir=run_dir,
                frames_dir=frames_dir,
                openvla=openvla,
                camera=camera,
                control=control,
                pose_manager=pose_manager,
                monitor_model=monitor_model,
                llm_model=llm_model,
                check_interval=args.diary_check_interval,
                max_steps=args.max_steps_per_subgoal,
                max_corrections=args.max_corrections,
                frame_offset=frame_offset,
                command_dt_s=args.command_dt_s,
                action_pose_mode=args.action_pose_mode,
                trajectory_log=trajectory_log,
                check_interval_s=check_interval_s,
                max_seconds=max_seconds,
                stall_window=args.stall_window,
                stall_threshold=args.stall_threshold,
                stall_completion_floor=args.stall_completion_floor,
                constraints=active_constraints,
                recorder=recorder,
            )
            frame_offset += result["total_steps"]
            subgoal_summaries.append(result)

            if result["stop_reason"] == "operator_abort":
                logger.info("Mission aborted by operator.")
                break

            if result["stop_reason"] == "replan":
                new_instruction = result["replan_instruction"]
                logger.info(
                    "Full replan requested. Old instruction: '%s'. New instruction: '%s'.",
                    instruction, new_instruction,
                )
                instruction = new_instruction
                llm_interface = LLMUserInterface(model=llm_model)
                recorder_ctx["llm_interface"] = llm_interface
                planner = LTLSymbolicPlanner(llm_interface)
                planner.plan_from_natural_language(new_instruction)
                ltl_plan = {
                    "ltl_nl_formula": llm_interface.ltl_nl_formula.get("ltl_nl_formula", ""),
                    "pi_predicates": dict(planner.pi_map),
                }
                recorder_ctx["ltl_plan"] = ltl_plan
                logger.info("Replan LTL: %s", json.dumps(ltl_plan, indent=2))
                current_subgoal = planner.get_next_predicate()
                continue

            if result["stop_reason"] == "override_subgoal":
                new_sub = result.get("new_subgoal_override", "")
                logger.info(
                    "Subgoal overridden by operator: '%s' -> '%s'. "
                    "Re-running subgoal (planner state unchanged).",
                    current_subgoal, new_sub,
                )
                current_subgoal = new_sub
                continue

            planner.advance_state(current_subgoal)
            current_subgoal = planner.get_next_predicate()

    finally:
        end_ts = datetime.now().isoformat()
        # Guard against llm_interface being unbound (e.g. planner import or
        # plan_from_natural_language raised before assignment); otherwise the
        # original exception gets masked by NameError from this block.
        planner_records = (
            llm_interface.llm_call_records if llm_interface is not None else []
        )
        try:
            with open(run_dir / "trajectory_log.json", "w") as f:
                json.dump(trajectory_log, f, indent=2)
            run_info = {
                "task": {
                    "instruction": instruction,
                    "initial_pos": initial_world_pose,
                    "max_steps_per_subgoal": args.max_steps_per_subgoal,
                    "diary_check_interval": args.diary_check_interval,
                    "max_corrections": args.max_corrections,
                },
                "llm_model": llm_model,
                "monitor_model": monitor_model,
                "models": {
                    "ltl_nl_planning": llm_model,
                    "subgoal_converter": llm_model,
                    "goal_adherence_monitor": monitor_model,
                    "openvla_predict_url": args.openvla_predict_url,
                },
                "ltl_plan": ltl_plan,
                "subgoal_count": len(subgoal_summaries),
                "subgoal_summaries": subgoal_summaries,
                "total_steps": sum(s["total_steps"] for s in subgoal_summaries),
                "total_vlm_calls": sum(s.get("vlm_calls", 0) for s in subgoal_summaries),
                "all_llm_call_records": (
                    planner_records
                    + [r for s in subgoal_summaries for r in s.get("llm_call_records", [])]
                ),
                "total_input_tokens": (
                    sum(r.get("input_tokens", 0) for r in planner_records)
                    + sum(
                        sum(r.get("input_tokens", 0) for r in s.get("llm_call_records", []))
                        for s in subgoal_summaries
                    )
                ),
                "total_output_tokens": (
                    sum(r.get("output_tokens", 0) for r in planner_records)
                    + sum(
                        sum(r.get("output_tokens", 0) for r in s.get("llm_call_records", []))
                        for s in subgoal_summaries
                    )
                ),
                "total_corrections": sum(s.get("corrections_used", 0) for s in subgoal_summaries),
                "odometry_failed_over": pose_manager.failed_over,
                "start_time": start_ts,
                "end_time": end_ts,
            }
            write_json(run_dir / "run_info.json", run_info)
            logger.info(
                "Run complete. Saved to %s (%d subgoals, %d steps)",
                run_dir,
                run_info["subgoal_count"],
                run_info["total_steps"],
            )
        except Exception as exc:
            logger.error("Failed to write run summary: %s", exc)
            raise
        finally:
            stop_capture = True
            recorder.close()
            pose_manager.close()
            control.close()
            camera.release()
            cv2.destroyAllWindows()
            if frames_tempdir is not None:
                shutil.rmtree(frames_tempdir, ignore_errors=True)


if __name__ == "__main__":
    main()
