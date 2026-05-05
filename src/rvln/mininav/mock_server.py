#!/usr/bin/env python3
"""
Full simulator for the MiniNav drone-side hardware.

Stands in for both halves of a real onboard companion:
  * TCP control sink: accepts [frame_count, vx, vy, vz, yaw] float32 packets
    in the boieng_mininav.py wire format and logs them to CSV.
  * HTTP frame feed: serves GET /frame as image/jpeg, sourced from a
    configurable directory (default: random PNGs auto-discovered under
    results/**/frames). Falls back to a generated white frame if no images
    are available, so the pipeline can be exercised end-to-end with no real
    camera attached.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import os
import random
import signal
import socket
import struct
import threading
import time
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Optional

from rvln.config import (
    DEFAULT_FRAME_GLOB,
    DEFAULT_FRAME_PORT,
    DEFAULT_FRAME_SAMPLE_CAP,
    DEFAULT_FRAME_SIZE,
)

# boieng_mininav.py sends 5 float32 values per packet:
# frame_count + vx + vy + vz + yaw
FLOATS_PER_PACKET = 5
PACKET_SIZE_BYTES = FLOATS_PER_PACKET * 4
PACKET_STRUCT = struct.Struct("<5f")


def _get_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


class SimulatedPose:
    """Thread-safe integrated pose updated from received velocity commands."""

    def __init__(self, initial: List[float]):
        self._lock = threading.Lock()
        self._pose = list(initial)

    def integrate(self, vx: float, vy: float, vz: float, yaw_rad_s: float, dt_s: float) -> None:
        with self._lock:
            self._pose[0] += vx * dt_s
            self._pose[1] += vy * dt_s
            self._pose[2] += vz * dt_s
            yaw = self._pose[3] + math.degrees(yaw_rad_s) * dt_s
            yaw = yaw % 360.0
            if yaw > 180.0:
                yaw -= 360.0
            self._pose[3] = yaw

    def get(self) -> List[float]:
        with self._lock:
            return list(self._pose)

    def as_json_bytes(self) -> bytes:
        p = self.get()
        return json.dumps({"x": p[0], "y": p[1], "z": p[2], "yaw": p[3]}).encode()


@dataclass
class DroneCommand:
    timestamp_iso: str
    frame_count: int
    vx: float
    vy: float
    vz: float
    yaw: float
    dt_s: float
    speed_xyz: float
    yaw_deg_s: float
    packet_index: int


def _discover_default_frames(sample_cap: int) -> List[str]:
    """Glob results/**/frames/*.png from REPO_ROOT, return up to sample_cap paths."""
    try:
        from rvln.paths import REPO_ROOT
    except ImportError:
        return []
    results_root = REPO_ROOT / "results"
    if not results_root.is_dir():
        return []
    paths = [str(p) for p in results_root.glob(DEFAULT_FRAME_GLOB)]
    if not paths:
        return []
    if len(paths) > sample_cap:
        paths = random.sample(paths, sample_cap)
    return paths


def _load_frames(frames_dir: Optional[str], sample_cap: int) -> List[str]:
    if frames_dir is None:
        return _discover_default_frames(sample_cap)
    base = os.path.abspath(frames_dir)
    if not os.path.isdir(base):
        return []
    paths: List[str] = []
    for root, _dirs, files in os.walk(base):
        for name in files:
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, name))
    if len(paths) > sample_cap:
        paths = random.sample(paths, sample_cap)
    return paths


def _generate_white_jpeg(size: int) -> bytes:
    from PIL import Image
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class FrameFeedServer:
    """HTTP server providing GET /frame (image) and GET /pose (JSON).

    Frame sources (in priority order):
      1. ``frames_dir`` - serve from a directory of images
      2. ``webcam`` device index - capture from local webcam (default)
      3. White fallback JPEG if nothing else is available
    """

    def __init__(
        self,
        host: str,
        port: int,
        frames_dir: Optional[str],
        frame_size: int,
        sample_cap: int,
        log_event,
        webcam: Optional[int] = 0,
        pose: Optional[SimulatedPose] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.frame_size = frame_size
        self._log = log_event
        self._fallback_jpeg: Optional[bytes] = None
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.pose = pose

        self._webcam_frame: Optional[bytes] = None
        self._webcam_lock = threading.Lock()
        self._webcam_cap = None
        self._webcam_thread: Optional[threading.Thread] = None
        self._webcam_stop = False

        if frames_dir is not None:
            self.frames: List[str] = _load_frames(frames_dir, sample_cap)
            if self.frames:
                self._log(
                    f"Frame feed pool: {len(self.frames)} images (source={frames_dir})"
                )
            else:
                self._log(f"No images found in {frames_dir}; falling back to white frame.")
                self._fallback_jpeg = _generate_white_jpeg(self.frame_size)
        elif webcam is not None:
            self.frames = []
            self._start_webcam(webcam)
        else:
            self.frames = []
            self._fallback_jpeg = _generate_white_jpeg(self.frame_size)
            self._log(
                f"Webcam disabled; serving generated {self.frame_size}x"
                f"{self.frame_size} white JPEG."
            )

    def _start_webcam(self, device: int) -> None:
        import cv2
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            self._log(f"Webcam {device} failed to open; falling back to white frame.")
            print(f"\033[93mWARNING: Webcam {device} could not be opened. "
                  f"Serving a generated white frame instead. "
                  f"Use --frames_dir to serve from a directory of images.\033[0m")
            self._fallback_jpeg = _generate_white_jpeg(self.frame_size)
            return
        self._webcam_cap = cap
        self._webcam_thread = threading.Thread(target=self._webcam_loop, daemon=True)
        self._webcam_thread.start()
        self._log(f"Webcam {device} opened for frame feed.")

    def _webcam_loop(self) -> None:
        import cv2
        cap = self._webcam_cap
        while not self._webcam_stop:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            h, w = frame.shape[:2]
            center_h, center_w = h // 2, w // 2
            half = min(h, w) // 2
            cropped = frame[center_h - half:center_h + half, center_w - half:center_w + half]
            resized = cv2.resize(cropped, (self.frame_size, self.frame_size))
            ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with self._webcam_lock:
                    self._webcam_frame = buf.tobytes()
            time.sleep(1.0 / 30.0)
        cap.release()

    def _next_frame_bytes(self) -> bytes:
        with self._webcam_lock:
            if self._webcam_frame is not None:
                return self._webcam_frame
        if self.frames:
            path = random.choice(self.frames)
            try:
                with open(path, "rb") as f:
                    return f.read()
            except OSError as exc:
                self._log(f"WARNING: Failed to read frame {path}: {exc}")
        if self._fallback_jpeg is None:
            self._fallback_jpeg = _generate_white_jpeg(self.frame_size)
        return self._fallback_jpeg

    def _build_handler(self):
        feed = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 (stdlib API)
                route = self.path.split("?", 1)[0]
                if route == "/frame":
                    payload = feed._next_frame_bytes()
                    ctype = "image/png" if payload[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(payload)
                elif route == "/pose" and feed.pose is not None:
                    body = feed.pose.as_json_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *_args, **_kwargs):
                return

        return _Handler

    def start(self) -> None:
        self._httpd = HTTPServer((self.host, self.port), self._build_handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        self._log(f"Frame feed listening on http://{self.host}:{self.port}/frame")
        if self.pose is not None:
            self._log(f"Pose endpoint listening on http://{self.host}:{self.port}/pose")

    def stop(self) -> None:
        self._webcam_stop = True
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None


class MiniNavDroneServer:
    def __init__(
        self,
        host: str,
        port: int,
        output_dir: str,
        print_every: int,
        recv_buf_size: int,
        timeout_s: float,
        frame_feed: Optional[FrameFeedServer] = None,
        pose: Optional[SimulatedPose] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.output_dir = output_dir
        self.print_every = max(1, print_every)
        self.recv_buf_size = recv_buf_size
        self.timeout_s = timeout_s
        self.frame_feed = frame_feed
        self.pose = pose

        self._stop_requested = False
        self._server_socket: Optional[socket.socket] = None
        self._client_socket: Optional[socket.socket] = None
        self._client_addr = None

        self._packet_index = 0
        self._last_command_time: Optional[float] = None
        self._last_frame_count: Optional[int] = None
        self._session_started_at = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(
            self.output_dir, f"simulated_drone_commands_{self._session_started_at}.csv"
        )
        self.log_path = os.path.join(
            self.output_dir, f"simulated_drone_events_{self._session_started_at}.log"
        )

        self._init_csv()

    def _init_csv(self) -> None:
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp_iso",
                    "frame_count",
                    "vx",
                    "vy",
                    "vz",
                    "yaw",
                    "dt_s",
                    "speed_xyz",
                    "yaw_deg_s",
                    "packet_index",
                ],
            )
            writer.writeheader()

    def _log_event(self, msg: str) -> None:
        stamp = dt.datetime.now().isoformat(timespec="milliseconds")
        line = f"{stamp} | {msg}"
        print(line)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def request_stop(self, *_args) -> None:
        # Signal-handler safe: only flip the flag and stop the (separate-thread)
        # HTTP frame feed. Sockets are torn down by run() once its accept/recv
        # loops time out and observe the flag, avoiding EBADF on the in-flight
        # accept() call.
        self._stop_requested = True
        self._log_event("Stop requested; shutting down server.")
        if self.frame_feed is not None:
            self.frame_feed.stop()

    def _close_client(self) -> None:
        if self._client_socket is not None:
            try:
                self._client_socket.close()
            except OSError:
                pass
            self._client_socket = None
            self._client_addr = None

    def _close_server(self) -> None:
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None

    def _listen(self) -> None:
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(1)
        self._server_socket.settimeout(1.0)
        self._log_event(f"Listening on {self.host}:{self.port}")

    def _accept_client(self) -> None:
        assert self._server_socket is not None
        while not self._stop_requested:
            try:
                client, addr = self._server_socket.accept()
                client.settimeout(self.timeout_s)
                self._client_socket = client
                self._client_addr = addr
                self._log_event(f"Client connected from {addr[0]}:{addr[1]}")
                return
            except socket.timeout:
                continue

    def _decode_command(self, payload: bytes) -> DroneCommand:
        frame_count_f, vx, vy, vz, yaw = PACKET_STRUCT.unpack(payload)
        now = time.time()
        dt_s = 0.0 if self._last_command_time is None else now - self._last_command_time
        self._last_command_time = now

        self._packet_index += 1
        frame_count = int(round(frame_count_f))
        speed_xyz = (vx * vx + vy * vy + vz * vz) ** 0.5
        yaw_deg_s = yaw * (180.0 / 3.141592653589793)

        if self.pose is not None and dt_s > 0:
            self.pose.integrate(vx, vy, vz, yaw, dt_s)

        return DroneCommand(
            timestamp_iso=dt.datetime.now().isoformat(timespec="milliseconds"),
            frame_count=frame_count,
            vx=vx,
            vy=vy,
            vz=vz,
            yaw=yaw,
            dt_s=dt_s,
            speed_xyz=speed_xyz,
            yaw_deg_s=yaw_deg_s,
            packet_index=self._packet_index,
        )

    def _write_command(self, cmd: DroneCommand) -> None:
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(cmd).keys()))
            writer.writerow(asdict(cmd))

    def _print_summary(self, cmd: DroneCommand) -> None:
        frame_jump = (
            None
            if self._last_frame_count is None
            else cmd.frame_count - self._last_frame_count
        )
        self._last_frame_count = cmd.frame_count

        if cmd.packet_index % self.print_every != 0:
            return

        self._log_event(
            (
                f"pkt={cmd.packet_index:06d} "
                f"frame={cmd.frame_count} "
                f"jump={frame_jump if frame_jump is not None else 'n/a'} "
                f"vel=({cmd.vx:+.3f}, {cmd.vy:+.3f}, {cmd.vz:+.3f}) "
                f"yaw={cmd.yaw:+.3f}rad/s ({cmd.yaw_deg_s:+.1f}deg/s) "
                f"speed={cmd.speed_xyz:.3f} "
                f"dt={cmd.dt_s:.3f}s"
            )
        )

    def _consume_client_stream(self) -> None:
        assert self._client_socket is not None

        buffer = bytearray()
        while not self._stop_requested:
            try:
                chunk = self._client_socket.recv(self.recv_buf_size)
                if not chunk:
                    self._log_event("Client disconnected.")
                    break
                buffer.extend(chunk)

                while len(buffer) >= PACKET_SIZE_BYTES:
                    payload = bytes(buffer[:PACKET_SIZE_BYTES])
                    del buffer[:PACKET_SIZE_BYTES]
                    cmd = self._decode_command(payload)
                    self._write_command(cmd)
                    self._print_summary(cmd)
            except socket.timeout:
                continue
            except ConnectionResetError:
                self._log_event("Client connection reset.")
                break
            except OSError as exc:
                self._log_event(f"Socket error: {exc}")
                break

    def run(self) -> None:
        self._log_event("Starting simulated MiniNav drone server.")
        self._log_event(f"Command packet format: {FLOATS_PER_PACKET} float32 ({PACKET_SIZE_BYTES} bytes)")
        self._log_event(f"CSV output: {self.csv_path}")

        if self.frame_feed is not None:
            self.frame_feed.start()

        self._listen()
        while not self._stop_requested:
            self._accept_client()
            if self._stop_requested:
                break
            if self._client_socket is None:
                continue
            self._consume_client_stream()
            self._close_client()
            self._last_command_time = None
            self._last_frame_count = None

        self._close_client()
        self._close_server()
        self._log_event("Server shutdown complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulated drone server for boieng_mininav.py command stream."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8080, help="Bind TCP port.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="simulated_mininav_server_logs",
        help="Directory for CSV and event logs.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="Print a summary every N packets.",
    )
    parser.add_argument(
        "--recv_buf_size",
        type=int,
        default=4096,
        help="Socket recv buffer size in bytes.",
    )
    parser.add_argument(
        "--timeout_s",
        type=float,
        default=1.0,
        help="Socket timeout for non-blocking shutdown loop.",
    )
    parser.add_argument(
        "--frame_port",
        type=int,
        default=DEFAULT_FRAME_PORT,
        help=(
            "HTTP port for the GET /frame image feed. Set to 0 to disable "
            f"the frame server. Default: {DEFAULT_FRAME_PORT}."
        ),
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help=(
            "Directory to source frames from (recursively). Default: "
            "auto-discover results/**/frames/*.png. Falls back to a "
            "generated white frame if no images are found."
        ),
    )
    parser.add_argument(
        "--frame_size",
        type=int,
        default=DEFAULT_FRAME_SIZE,
        help="Size of the white fallback frame in pixels (square).",
    )
    parser.add_argument(
        "--frame_sample_cap",
        type=int,
        default=DEFAULT_FRAME_SAMPLE_CAP,
        help="Cap on the number of frames sampled from the source pool.",
    )
    parser.add_argument(
        "--webcam",
        type=int,
        default=0,
        help="Webcam device index (default: 0). Ignored when --frames_dir is set.",
    )
    parser.add_argument(
        "--no-webcam",
        action="store_true",
        help="Disable webcam; serve a generated white frame instead.",
    )
    parser.add_argument(
        "--initial_position",
        type=str,
        default="0,0,0,0",
        help="Initial simulated pose x,y,z,yaw (default: 0,0,0,0).",
    )
    parser.add_argument(
        "--openvla_host",
        type=str,
        default=None,
        help="OpenVLA server host for the printed connection command (default: this machine's IP).",
    )
    return parser.parse_args()


def _parse_initial_position(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--initial_position must be x,y,z,yaw")
    return [float(p) for p in parts]


def _print_connection_info(
    local_ip: str,
    tcp_port: int,
    frame_port: int,
    openvla_host: str,
) -> None:
    print()
    print("=" * 60)
    print("Mock drone ready. On the client machine run:")
    print()
    print(f"  python scripts/run_hardware.py \\")
    print(f"      --control_host {local_ip} \\")
    print(f"      --control_port {tcp_port} \\")
    print(f"      --camera_url http://{local_ip}:{frame_port}/frame \\")
    print(f"      --odom_http_url http://{local_ip}:{frame_port}/pose \\")
    print(f"      --openvla_predict_url http://{openvla_host}:5007/predict \\")
    print(f"      --command_is_velocity \\")
    print(f"      --action_pose_mode delta_from_pose \\")
    print(f"      --instruction \"your instruction here\"")
    print()
    print("(--command_is_velocity / --action_pose_mode delta_from_pose are")
    print(" required: the mock integrates incoming commands as velocities.)")
    print()
    print("=" * 60)
    print()


def main() -> None:
    args = parse_args()

    initial_pose = _parse_initial_position(args.initial_position)
    pose = SimulatedPose(initial_pose)

    server = MiniNavDroneServer(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
        print_every=args.print_every,
        recv_buf_size=args.recv_buf_size,
        timeout_s=args.timeout_s,
        pose=pose,
    )

    if args.frame_port > 0:
        webcam_device = None if (args.no_webcam or args.frames_dir) else args.webcam
        server.frame_feed = FrameFeedServer(
            host=args.host,
            port=args.frame_port,
            frames_dir=args.frames_dir,
            frame_size=args.frame_size,
            sample_cap=args.frame_sample_cap,
            log_event=server._log_event,
            webcam=webcam_device,
            pose=pose,
        )

    signal.signal(signal.SIGINT, server.request_stop)
    signal.signal(signal.SIGTERM, server.request_stop)

    local_ip = _get_local_ip()
    display_ip = args.host if args.host not in ("0.0.0.0", "::") else local_ip
    openvla_host = args.openvla_host or display_ip
    if args.frame_port > 0:
        _print_connection_info(display_ip, args.port, args.frame_port, openvla_host)

    server.run()


if __name__ == "__main__":
    main()
