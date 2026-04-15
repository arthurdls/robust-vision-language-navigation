#!/usr/bin/env python3
"""
Start the MiniNav TCP mock drone control server.

Stands in for the real flight controller during dry runs. Accepts the
[frame_count, vx, vy, vz, yaw] float32 packets that the hardware pipeline
sends, logs every command to a CSV, and prints periodic summaries.

Usage (from repo root):
  python scripts/start_simulate_hardware.py
  python scripts/start_simulate_hardware.py --host 127.0.0.1 --port 8080
  python scripts/start_simulate_hardware.py --output_dir my_logs --print_every 5

All flags are forwarded to rvln.mininav.mock_server.
"""

import sys
from pathlib import Path

# src/ layout: allow `python scripts/start_simulate_hardware.py` without `pip install -e .`
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.mininav.mock_server import main


if __name__ == "__main__":
    main()
