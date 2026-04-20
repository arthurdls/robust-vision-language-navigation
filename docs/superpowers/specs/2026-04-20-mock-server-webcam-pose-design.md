# Mock Server: Webcam Feed, Integrated Pose, and Connection Info

**Date:** 2026-04-20
**Status:** Draft

## Goal

Make `start_simulate_hardware.py` a turnkey mock drone: webcam frames by
default, integrated pose tracking from received commands, and a printed
connection string so a second machine can run `run_hardware.py` against it
with zero manual URL assembly.

Also make `run_hardware.py` pose source selection explicit and mutually
exclusive: either external odometry OR dead-reckoning, never both, never
neither.

## Design

### 1. mock_server.py: Webcam frame feed (default)

- Default frame source: local webcam via `cv2.VideoCapture(0)`.
- A daemon thread grabs frames continuously, center-crops to square,
  resizes to `frame_size` (default 640), stores latest under a lock.
- `GET /frame` JPEG-encodes and returns the latest frame.
- Old file-based mode: `--frames_dir <path>` overrides webcam.
- `--no-webcam` flag disables webcam and falls back to white frame.
- If webcam fails to open, log a warning and fall back to white frame
  (don't crash the server).
- New `--webcam` flag accepts an optional device index (default 0).

### 2. mock_server.py: Integrated pose via /pose endpoint

- New `SimulatedPose` class: holds `[x, y, z, yaw]` with a
  `threading.Lock`.
- Initialized at `(0, 0, 0, 0)` (configurable via `--initial_position`).
- Each decoded TCP command integrates velocity into position:
  `x += vx * dt`, `y += vy * dt`, `z += vz * dt`, `yaw += yaw_rate * dt`.
  `yaw` is normalized to [-180, 180) degrees after integration.
- `GET /pose` on the same HTTP server returns JSON:
  `{"x": float, "y": float, "z": float, "yaw": float}`.
- The `FrameFeedServer` HTTP handler gains access to the pose object via
  constructor injection.

### 3. mock_server.py: Connection info printout

- At startup, detect local IP using the UDP socket trick
  (`socket.connect(("8.8.8.8", 80))`; fall back to `127.0.0.1`).
- After all servers are listening, print a block:

```
===========================================================
Mock drone ready. On the client machine run:

  python scripts/run_hardware.py \
      --preferred_server_host <LOCAL_IP> \
      --control_port <TCP_PORT> \
      --camera_url http://<LOCAL_IP>:<FRAME_PORT>/frame \
      --odom_http_url http://<LOCAL_IP>:<FRAME_PORT>/pose \
      --openvla_predict_url http://<OPENVLA_HOST>:5007/predict \
      --instruction "your instruction here"
===========================================================
```

- `<OPENVLA_HOST>` defaults to `<LOCAL_IP>` but can be overridden with a
  new `--openvla_host` flag for when the OpenVLA server is on a third
  machine.

### 4. interface.py: Mutually exclusive pose source

- New `--dead-reckoning` boolean flag (default False).
- Validation at startup:
  - If `--dead-reckoning` AND (`--odom_http_url` or `--odom_udp_port`):
    error out: "Cannot use both dead-reckoning and external odometry."
  - If neither `--dead-reckoning` NOR any odom source: error out:
    "No pose source. Use --odom_http_url / --odom_udp_port for external
    odometry, or --dead-reckoning for estimated poses."
- When `--dead-reckoning` is active, print red warning:
  `\033[91mWARNING: Dead-reckoning mode active. Pose will drift.\033[0m`
- Remove the current always-create `DeadReckoningPoseProvider` behavior;
  only instantiate it when `--dead-reckoning` is passed.
- `PoseManager` receives whichever single provider was selected.

## Files Changed

| File | Change |
|------|--------|
| `src/rvln/mininav/mock_server.py` | Webcam capture, SimulatedPose, /pose endpoint, connection info, CLI flags |
| `src/rvln/mininav/interface.py` | Mutually exclusive pose source, --dead-reckoning flag, red warning |

## Out of Scope

- Real odometry integration (ROS, MAVLink).
- Multi-client support for the TCP server.
- Audio or video streaming beyond single-frame HTTP pull.
