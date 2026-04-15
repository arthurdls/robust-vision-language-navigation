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


# boieng_mininav.py sends 5 float32 values per packet:
# frame_count + vx + vy + vz + yaw
FLOATS_PER_PACKET = 5
PACKET_SIZE_BYTES = FLOATS_PER_PACKET * 4
PACKET_STRUCT = struct.Struct("<5f")

DEFAULT_FRAME_PORT = 8081
DEFAULT_FRAME_SIZE = 640
DEFAULT_FRAME_GLOB = "**/frames/*.png"
DEFAULT_FRAME_SAMPLE_CAP = 200


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
    except Exception:
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
    """Serves GET /frame with image/jpeg, randomly selected from a frame pool."""

    def __init__(
        self,
        host: str,
        port: int,
        frames_dir: Optional[str],
        frame_size: int,
        sample_cap: int,
        log_event,
    ) -> None:
        self.host = host
        self.port = port
        self.frame_size = frame_size
        self.frames: List[str] = _load_frames(frames_dir, sample_cap)
        self._log = log_event
        self._fallback_jpeg: Optional[bytes] = None
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

        if self.frames:
            self._log(
                f"Frame feed pool: {len(self.frames)} images "
                f"(source={frames_dir or 'auto-discovered results/**/frames'})"
            )
        else:
            self._fallback_jpeg = _generate_white_jpeg(self.frame_size)
            self._log(
                f"Frame feed pool empty; serving generated {self.frame_size}x"
                f"{self.frame_size} white JPEG."
            )

    def _next_frame_bytes(self) -> bytes:
        if self.frames:
            path = random.choice(self.frames)
            try:
                with open(path, "rb") as f:
                    return f.read()
            except OSError:
                pass
        if self._fallback_jpeg is None:
            self._fallback_jpeg = _generate_white_jpeg(self.frame_size)
        return self._fallback_jpeg

    def _build_handler(self):
        feed = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 (stdlib API)
                if self.path.split("?", 1)[0] != "/frame":
                    self.send_response(404)
                    self.end_headers()
                    return
                payload = feed._next_frame_bytes()
                ctype = "image/png" if payload[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args, **_kwargs):
                return

        return _Handler

    def start(self) -> None:
        self._httpd = HTTPServer((self.host, self.port), self._build_handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        self._log(f"Frame feed listening on http://{self.host}:{self.port}/frame")

    def stop(self) -> None:
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
    ) -> None:
        self.host = host
        self.port = port
        self.output_dir = output_dir
        self.print_every = max(1, print_every)
        self.recv_buf_size = recv_buf_size
        self.timeout_s = timeout_s
        self.frame_feed = frame_feed

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
        self._stop_requested = True
        self._log_event("Stop requested; shutting down server.")
        self._close_client()
        self._close_server()
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    server = MiniNavDroneServer(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
        print_every=args.print_every,
        recv_buf_size=args.recv_buf_size,
        timeout_s=args.timeout_s,
    )

    if args.frame_port > 0:
        server.frame_feed = FrameFeedServer(
            host=args.host,
            port=args.frame_port,
            frames_dir=args.frames_dir,
            frame_size=args.frame_size,
            sample_cap=args.frame_sample_cap,
            log_event=server._log_event,
        )

    signal.signal(signal.SIGINT, server.request_stop)
    signal.signal(signal.SIGTERM, server.request_stop)

    server.run()


if __name__ == "__main__":
    main()
