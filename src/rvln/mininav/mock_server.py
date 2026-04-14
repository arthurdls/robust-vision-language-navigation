#!/usr/bin/env python3
"""
TCP simulator server for boieng_mininav.py.

Expected client payload (per command packet):
    [frame_count, vx, vy, vz, yaw] as float32
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import signal
import socket
import struct
import time
from dataclasses import dataclass, asdict
from typing import Optional


# boieng_mininav.py sends 5 float32 values per packet:
# frame_count + vx + vy + vz + yaw
FLOATS_PER_PACKET = 5
PACKET_SIZE_BYTES = FLOATS_PER_PACKET * 4
PACKET_STRUCT = struct.Struct("<5f")


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


class MiniNavDroneServer:
    def __init__(
        self,
        host: str,
        port: int,
        output_dir: str,
        print_every: int,
        recv_buf_size: int,
        timeout_s: float,
    ) -> None:
        self.host = host
        self.port = port
        self.output_dir = output_dir
        self.print_every = max(1, print_every)
        self.recv_buf_size = recv_buf_size
        self.timeout_s = timeout_s

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

    signal.signal(signal.SIGINT, server.request_stop)
    signal.signal(signal.SIGTERM, server.request_stop)

    server.run()


if __name__ == "__main__":
    main()
