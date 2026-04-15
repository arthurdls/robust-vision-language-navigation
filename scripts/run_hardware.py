#!/usr/bin/env python3
"""
Run the MiniNav real-drone integration pipeline.

Drives a real (or mocked) drone via the boieng wire format using the OpenVLA
server, LTL planner, and live diary monitor. Results land under
results/hardware/run_<timestamp>/. See README "Running on Hardware (MiniNav)"
for the full terminal layout.

Usage (from repo root):
  # Live flight (real camera + real drone):
  python scripts/run_hardware.py --instruction "take off and circle the red cone"

  # Fully simulated (against scripts/start_simulate_hardware.py):
  python scripts/run_hardware.py \
      --preferred_server_host 127.0.0.1 \
      --control_port 8080 \
      --camera_url http://127.0.0.1:8081/frame \
      --openvla_predict_url http://127.0.0.1:5007/predict \
      --initial_position 0,0,0,0 \
      --command_is_velocity \
      --instruction "take off and circle the red cone"

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
