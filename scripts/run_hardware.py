#!/usr/bin/env python3
"""
Run the MiniNav real-drone integration pipeline.

Drives a real (or mocked) drone via the boieng wire format using the OpenVLA
server, LTL planner, and live goal adherence monitor. Results land under
results/hardware/run_<timestamp>/. See README "Running on Hardware (MiniNav)"
for the full terminal layout.

Wire format (matches boieng_mininav.py): each packet is 5 float32 values,
[frame_count, vx, vy, vz, yaw_rate], where vx/vy/vz are in cm/s and
yaw_rate is in rad/s. Per-step velocities are derived from OpenVLA's
predicted target pose and clipped to <=50 cm/s linear and <=20 deg/s
yaw before being sent.

Defaults target the Jetson + MiniNav drone: USB camera index 4, control
server at 192.168.0.101:8080, OpenVLA at 127.0.0.1:5007.

Usage (from repo root):
  # Live flight (USB camera + real drone):
  python scripts/run_hardware.py --instruction "take off and circle the red cone"

  # Fully simulated:
  #   terminal 1: python scripts/start_mock_hardware.py
  #   terminal 2: python scripts/start_server.py
  #   terminal 3:
  python scripts/run_hardware.py \
      --control_host 127.0.0.1 \
      --control_port 8080 \
      --camera_url http://127.0.0.1:8081/frame \
      --openvla_predict_url http://127.0.0.1:5007/predict \
      --initial_position 0,0,0,0 \
      --instruction "Move forward 10.0 meters, then turn toward the red car"

All flags are forwarded to rvln.mininav.interface.
"""

import sys
from pathlib import Path

# src/ layout: allow `python scripts/run_hardware.py` without `pip install -e .`
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.mininav.interface import main


if __name__ == "__main__":
    main()
