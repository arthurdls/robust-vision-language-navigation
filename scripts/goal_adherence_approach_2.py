"""
Goal adherence approach 2.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from goal_adherence_utils import (
    analyze_temporal_frames,
    build_frame_grid,
    get_ordered_frames_from_dir,
    query_gpt4o_mini_temporal,
    sample_frames_every_n,
)

__all__ = [
    "sample_frames_every_n",
    "get_ordered_frames_from_dir",
    "build_frame_grid",
    "query_gpt4o_mini_temporal",
    "analyze_temporal_frames",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build frame grid and query gpt-4o-mini for temporal analysis."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Results directory (REPL or LTL frame layout).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Sample every N frames (default: 10).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional prompt; default asks what is happening across the frames.",
    )
    parser.add_argument(
        "--grid-only",
        type=Path,
        metavar="OUTPUT.png",
        default=None,
        help="Build the grid and save to this path (no API call). Use - to show in viewer.",
    )
    args = parser.parse_args()
    if args.grid_only is not None:
        paths = get_ordered_frames_from_dir(args.dir)
        if not paths:
            print("No frames found.", file=sys.stderr)
            sys.exit(1)
        sampled = sample_frames_every_n(paths, args.n)
        grid = build_frame_grid(sampled)
        if str(args.grid_only) == "-":
            fd, path = tempfile.mkstemp(suffix=".png", prefix="goal_grid_")
            os.close(fd)
            grid.save(path)
            try:
                if sys.platform == "darwin":
                    subprocess.run(["open", path], check=True)
                elif sys.platform == "win32":
                    os.startfile(path)
                else:
                    subprocess.run(["xdg-open", path], check=True)
            except (subprocess.CalledProcessError, OSError) as e:
                print(f"Could not open viewer: {e}. Grid saved to: {path}", file=sys.stderr)
            else:
                print("Opened grid (saved to", path + ")")
        else:
            grid.save(args.grid_only)
            print("Saved grid to", args.grid_only)
        sys.exit(0)
    response = analyze_temporal_frames(args.dir, args.n, prompt=args.prompt)
    print(response)
