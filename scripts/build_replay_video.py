#!/usr/bin/env python3
"""Build the synchronized replay video from a hardware run results directory.

Usage:
    python scripts/build_replay_video.py [--results <dir>] [--out <path>]

Defaults to results/hardware/ and writes to results/hardware/replay_synced.mp4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

from replay_video import composite


def main():
    parser = argparse.ArgumentParser(description="Build the hardware run replay video")
    parser.add_argument("--results", default=str(_REPO / "results" / "hardware"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    results_dir = Path(args.results)
    out_path = Path(args.out) if args.out else results_dir / "replay_synced.mp4"
    composite.build(results_dir, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
