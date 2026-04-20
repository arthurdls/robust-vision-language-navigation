#!/usr/bin/env python3
"""
Single-file real-drone integration runner.

Pipeline:
1) Prompt user for an instruction (unless provided via CLI).
2) LTL decomposition with LLM_User_Interface + LTL_Symbolic_Planner.
3) Subgoal conversion + LiveDiaryMonitor supervision.
4) OpenVLA /predict + /reset calls.
5) Real command streaming to drone server using boieng wire format:
   [frame_count, vx, vy, vz, yaw] as float32 over TCP.
6) Pose source:
   - Primary: external odometry feed (HTTP poll or UDP stream)
   - Fallback: dead-reckoning from sent commands.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import signal
import socket
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


from rvln.paths import REPO_ROOT, load_env_vars
from rvln.sim.env_setup import state_for_openvla
from rvln.ai.utils.llm_providers import LLMFactory


logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "hardware"
IMG_INPUT_SIZE: Tuple[int, int] = (224, 224)
ACTION_SMALL_DELTA_POS = 3.0
ACTION_SMALL_DELTA_YAW = 1.0
ACTION_SMALL_STEPS = 10

stop_capture = False


def normalize_angle(angle_deg: float) -> float:
    angle_deg = angle_deg % 360.0
    if angle_deg > 180.0:
        angle_deg -= 360.0
    return angle_deg


def parse_position(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("Position must be x,y,z,yaw")
    return [float(p) for p in parts]


def relative_pose(current_world: List[float], origin_world: List[float]) -> List[float]:
    return [
        float(current_world[0] - origin_world[0]),
        float(current_world[1] - origin_world[1]),
        float(current_world[2] - origin_world[2]),
        float(normalize_angle(current_world[3] - origin_world[3])),
    ]


def sanitize_name(text: str, max_len: int = 48) -> str:
    clean = text.lower().replace(" ", "_")
    safe = "".join(c for c in clean if c.isalnum() or c == "_")
    return safe[:max_len] or "subgoal"


def signal_handler(sig, frame):
    global stop_capture
    stop_capture = True
    logger.info("Signal received; shutting down.")
    cv2.destroyAllWindows()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


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
                logger.debug("Frame poll failed: %s", exc)
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
            logger.info("Model reset: %s", resp.status_code)
        except Exception as exc:
            logger.warning("Model reset failed: %s", exc)

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
        try:
            resp = requests.post(
                self.predict_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("OpenVLA predict failed: %s", exc)
            return None


def get_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
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
        self.sock.send(payload)

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
    def __init__(
        self,
        http_url: Optional[str],
        udp_host: str,
        udp_port: int,
        stale_timeout_s: float,
    ):
        self.http_url = http_url
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.stale_timeout_s = stale_timeout_s
        self.last_sample: Optional[PoseSample] = None
        self._udp_sock: Optional[socket.socket] = None
        if udp_port > 0:
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_sock.bind((udp_host, udp_port))
            self._udp_sock.setblocking(False)
            logger.info("Listening for odometry UDP packets on %s:%d", udp_host, udp_port)

    def _parse_pose(self, payload: Dict[str, Any]) -> Optional[PoseSample]:
        try:
            x = float(payload["x"])
            y = float(payload["y"])
            z = float(payload["z"])
            yaw = float(payload["yaw"])
            return PoseSample(x=x, y=y, z=z, yaw_deg=normalize_angle(yaw), ts=time.time())
        except Exception:
            return None

    def _poll_http(self) -> None:
        if not self.http_url:
            return
        try:
            resp = requests.get(self.http_url, timeout=0.25)
            resp.raise_for_status()
            pose = self._parse_pose(resp.json())
            if pose is not None:
                self.last_sample = pose
        except Exception:
            return

    def _poll_udp(self) -> None:
        if self._udp_sock is None:
            return
        while True:
            try:
                data, _addr = self._udp_sock.recvfrom(65535)
            except BlockingIOError:
                break
            except Exception:
                break
            try:
                payload = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            pose = self._parse_pose(payload)
            if pose is not None:
                self.last_sample = pose

    def get_pose(self) -> Optional[List[float]]:
        self._poll_udp()
        self._poll_http()
        if self.last_sample is None:
            return None
        if time.time() - self.last_sample.ts > self.stale_timeout_s:
            return None
        return [
            self.last_sample.x,
            self.last_sample.y,
            self.last_sample.z,
            self.last_sample.yaw_deg,
        ]

    def close(self) -> None:
        if self._udp_sock is not None:
            try:
                self._udp_sock.close()
            except Exception:
                pass


class DeadReckoningPoseProvider:
    def __init__(self, initial_world_pose: List[float], command_is_velocity: bool):
        self.world_pose = list(initial_world_pose)
        self.command_is_velocity = command_is_velocity

    def set_world_pose(self, pose: List[float]) -> None:
        self.world_pose = [float(p) for p in pose]

    def update_from_command(self, command: np.ndarray, dt_s: float) -> None:
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
    provider = "gemini" if primary_model.startswith("gemini") else "openai"
    try:
        LLMFactory.create(provider, model=primary_model)
        return primary_model
    except Exception as exc:
        logger.warning(
            "Model '%s' unavailable (%s). Falling back to '%s'.",
            primary_model, exc, fallback_model,
        )
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
    check_interval: int,
    max_steps: int,
    max_corrections: int,
    frame_offset: int,
    command_dt_s: float,
    action_pose_mode: str,
    trajectory_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    from rvln.ai.diary_monitor import DiaryCheckResult, LiveDiaryMonitor
    from rvln.ai.subgoal_converter import SubgoalConverter

    converter = SubgoalConverter(model=monitor_model)
    converted_instruction = converter.convert(subgoal_nl)
    current_instruction = converted_instruction

    subgoal_dir = run_dir / f"subgoal_{subgoal_index:02d}_{sanitize_name(subgoal_nl)}"
    diary_artifacts = subgoal_dir / "diary_artifacts"
    diary_artifacts.mkdir(parents=True, exist_ok=True)
    monitor = LiveDiaryMonitor(
        subgoal=subgoal_nl,
        check_interval=check_interval,
        model=monitor_model,
        artifacts_dir=diary_artifacts,
        max_corrections=max_corrections,
    )

    openvla.reset_model()

    origin_world = pose_manager.get_world_pose()
    openvla_pose_origin = [0.0, 0.0, 0.0, 0.0]
    last_pose = None
    small_count = 0
    last_correction_step = -check_interval
    stop_reason = "max_steps"
    total_steps = 0
    override_history: List[Dict[str, Any]] = []

    for step in range(max_steps):
        if stop_capture:
            stop_reason = "interrupted"
            break

        ok, frame = camera.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        world_pose = pose_manager.get_world_pose()
        subgoal_rel_pose = relative_pose(world_pose, origin_world)

        global_frame_idx = frame_offset + step
        frame_path = frames_dir / f"frame_{global_frame_idx:06d}.png"
        cv2.imwrite(str(frame_path), frame)

        try:
            result = monitor.on_frame(frame_path, displacement=list(subgoal_rel_pose))
        except Exception as exc:
            logger.error("monitor.on_frame failed at step %d: %s", step, exc)
            result = DiaryCheckResult(
                action="continue",
                new_instruction="",
                reasoning="monitor_error",
                diary_entry="",
                completion_pct=monitor.last_completion_pct,
            )

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

        openvla_pose = [c - o for c, o in zip(subgoal_rel_pose, openvla_pose_origin)]
        response = openvla.predict(
            image_bgr=frame,
            proprio=state_for_openvla(openvla_pose),
            instr=current_instruction.strip().lower(),
        )
        if response is None:
            stop_reason = "no_response"
            total_steps = step
            break

        action_poses = response.get("action")
        if not isinstance(action_poses, list) or len(action_poses) == 0:
            stop_reason = "empty_action"
            total_steps = step
            break

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

        total_steps = step + 1
        world_pose = pose_manager.get_world_pose()
        subgoal_rel_pose = relative_pose(world_pose, origin_world)
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
            try:
                conv_result = monitor.on_convergence(
                    frame_path, displacement=list(subgoal_rel_pose)
                )
            except Exception as exc:
                logger.error("monitor.on_convergence failed at step %d: %s", step, exc)
                conv_result = DiaryCheckResult(
                    action="stop",
                    new_instruction="",
                    reasoning="convergence_monitor_error",
                    diary_entry="",
                    completion_pct=monitor.last_completion_pct,
                )

            if conv_result.action == "stop":
                stop_reason = "monitor_complete"
                break

            if conv_result.new_instruction:
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
        stop_reason = "max_steps"
        total_steps = max_steps

    write_json(
        subgoal_dir / "diary_summary.json",
        {
            "subgoal": subgoal_nl,
            "converted_instruction": converted_instruction,
            "diary": monitor.diary,
            "override_history": override_history,
            "corrections_used": monitor.corrections_used,
            "last_completion_pct": monitor.last_completion_pct,
            "high_water_mark": monitor.high_water_mark,
            "parse_failures": monitor.parse_failures,
            "vlm_calls": monitor.vlm_calls,
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
        "high_water_mark": monitor.high_water_mark,
        "vlm_calls": monitor.vlm_calls,
        "next_origin": list(next_world_origin),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-file real drone integration runner.",
    )
    parser.add_argument("--instruction", type=str, default=None, help="Initial task instruction. If omitted, prompt in CLI.")
    parser.add_argument("--initial_position", type=str, default="0,0,0,0", help="x,y,z,yaw (degrees).")
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--camera_url",
        type=str,
        default=None,
        help=(
            "HTTP URL serving JPEG/PNG frames (e.g. http://127.0.0.1:8081/frame). "
            "When set, replaces the local cv2 capture with an HTTP-pull camera. "
            "Use this to test against the simulated MiniNav frame feed."
        ),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera_retries", type=int, default=15)
    parser.add_argument("--camera_init_timeout", type=float, default=8.0)
    parser.add_argument("--record", action="store_true", help="Save camera frames.")
    parser.add_argument("--preferred_server_host", type=str, default="192.168.0.101")
    parser.add_argument("--control_port", type=int, default=8080)
    parser.add_argument("--control_retries", type=int, default=10)
    parser.add_argument("--control_retry_sleep", type=float, default=2.0)
    parser.add_argument("--openvla_predict_url", type=str, default="http://127.0.0.1:5007/predict")
    parser.add_argument("--llm_model", type=str, default="gpt-4o")
    parser.add_argument("--monitor_model", type=str, default="gpt-5.4")
    parser.add_argument("--llm_fallback_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--monitor_fallback_model", type=str, default="gpt-4o")
    parser.add_argument("--max_steps_per_subgoal", type=int, default=300)
    parser.add_argument("--diary_check_interval", type=int, default=10)
    parser.add_argument("--max_corrections", type=int, default=15)
    parser.add_argument("--command_dt_s", type=float, default=0.1)
    parser.add_argument(
        "--action_pose_mode",
        choices=["direct", "delta_from_pose"],
        default="direct",
        help="How to map OpenVLA action pose to boieng command stream.",
    )
    parser.add_argument(
        "--command_is_velocity",
        action="store_true",
        help="For dead-reckoning: treat sent commands as velocities (m/s, rad/s).",
    )
    parser.add_argument("--odom_http_url", type=str, default=None)
    parser.add_argument("--odom_udp_host", type=str, default="0.0.0.0")
    parser.add_argument("--odom_udp_port", type=int, default=0)
    parser.add_argument("--odom_stale_timeout_s", type=float, default=1.0)
    parser.add_argument(
        "--dead-reckoning",
        action="store_true",
        default=False,
        help="Use dead-reckoning for pose estimation instead of external odometry.",
    )
    parser.add_argument(
        "--extra-env-file",
        type=str,
        default=None,
        help="Optional env file loaded after .env and .env.local (overrides).",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    global stop_capture
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    load_env_vars(args.extra_env_file)
    initial_world_pose = parse_position(args.initial_position)
    instruction = args.instruction or input("Enter initial instruction: ").strip()
    if not instruction:
        raise SystemExit("Instruction is required.")

    llm_model = resolve_model_with_fallback(args.llm_model, args.llm_fallback_model)
    monitor_model = resolve_model_with_fallback(args.monitor_model, args.monitor_fallback_model)

    run_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = Path(args.results_dir) / f"run_{run_stamp}"
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

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

    has_odom = args.odom_http_url or args.odom_udp_port > 0
    has_dr = args.dead_reckoning
    if has_odom and has_dr:
        raise SystemExit("Cannot use both --dead-reckoning and external odometry (--odom_http_url / --odom_udp_port).")
    if not has_odom and not has_dr:
        raise SystemExit(
            "No pose source specified. Use --odom_http_url / --odom_udp_port for "
            "external odometry, or --dead-reckoning for estimated poses."
        )

    odom_provider = None
    if has_odom:
        odom_provider = OdometryPoseProvider(
            http_url=args.odom_http_url,
            udp_host=args.odom_udp_host,
            udp_port=args.odom_udp_port,
            stale_timeout_s=args.odom_stale_timeout_s,
        )

    if has_dr:
        print("\033[91mWARNING: Dead-reckoning mode active. Pose will drift over time.\033[0m")

    dr_provider = DeadReckoningPoseProvider(
        initial_world_pose=initial_world_pose,
        command_is_velocity=args.command_is_velocity,
    )
    pose_manager = PoseManager(odom=odom_provider, dr=dr_provider)

    openvla = OpenVLAClient(predict_url=args.openvla_predict_url)

    start_ts = datetime.now().isoformat()
    trajectory_log: List[Dict[str, Any]] = []
    subgoal_summaries: List[Dict[str, Any]] = []
    frame_offset = 0

    try:
        from rvln.ai.llm_interface import LLM_User_Interface
        from rvln.ai.ltl_planner import LTL_Symbolic_Planner

        llm_interface = LLM_User_Interface(model=llm_model)
        planner = LTL_Symbolic_Planner(llm_interface)
        planner.plan_from_natural_language(instruction)

        ltl_plan = {
            "ltl_nl_formula": llm_interface.ltl_nl_formula.get("ltl_nl_formula", ""),
            "pi_predicates": dict(planner.pi_map),
        }

        current_subgoal = planner.get_next_predicate()
        subgoal_index = 0
        while current_subgoal is not None and not stop_capture:
            subgoal_index += 1
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
                check_interval=args.diary_check_interval,
                max_steps=args.max_steps_per_subgoal,
                max_corrections=args.max_corrections,
                frame_offset=frame_offset,
                command_dt_s=args.command_dt_s,
                action_pose_mode=args.action_pose_mode,
                trajectory_log=trajectory_log,
            )
            frame_offset += result["total_steps"]
            subgoal_summaries.append(result)
            planner.advance_state(current_subgoal)
            current_subgoal = planner.get_next_predicate()

        with open(run_dir / "trajectory_log.json", "w") as f:
            json.dump(trajectory_log, f, indent=2)

        end_ts = datetime.now().isoformat()
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
                "subgoal_converter": monitor_model,
                "live_diary_monitor": monitor_model,
                "openvla_predict_url": args.openvla_predict_url,
            },
            "ltl_plan": ltl_plan,
            "subgoal_count": len(subgoal_summaries),
            "subgoal_summaries": subgoal_summaries,
            "total_steps": sum(s["total_steps"] for s in subgoal_summaries),
            "total_vlm_calls": sum(s.get("vlm_calls", 0) for s in subgoal_summaries),
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
    finally:
        stop_capture = True
        pose_manager.close()
        control.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
