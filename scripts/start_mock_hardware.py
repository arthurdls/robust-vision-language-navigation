#!/usr/bin/env python3
"""
Start the simulated MiniNav drone-side hardware.

Acts exactly like a real onboard companion, except the data is fake:
  * TCP control sink on --port: accepts [frame_count, vx, vy, vz, yaw]
    float32 packets and logs them to CSV.
  * HTTP frame feed on --frame_port (default 8081): serves GET /frame as
    image/jpeg, sourced from --frames_dir or auto-discovered from
    results/**/frames. Falls back to a generated white frame if no
    images are available, so the pipeline runs end-to-end with no real
    drone or USB camera attached.

Usage (from repo root):
  python scripts/start_mock_hardware.py
  python scripts/start_mock_hardware.py --host 127.0.0.1 --port 8080
  python scripts/start_mock_hardware.py --frames_dir results/ltl_results/run_xxx/frames
  python scripts/start_mock_hardware.py --frame_port 0   # disable frame feed

All flags are forwarded to rvln.mininav.mock_server.
"""

import sys
from pathlib import Path

# src/ layout: allow `python scripts/start_mock_hardware.py` without `pip install -e .`
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.mininav.mock_server import main


if __name__ == "__main__":
    main()
