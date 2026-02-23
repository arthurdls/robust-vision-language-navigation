#!/usr/bin/env python3
"""
Display FPV frames or trajectory plot images from a results directory.

Supports:
- LTL results: run dir with frames/ subdir (frame_000000.png, ...). Pass the run dir.
- UAV-Flow-Eval: dir with *_2d.png / *_3d.png trajectory plots. Pass that dir.

Usage (from repo root):
  python scripts/playback_fpv.py results/ltl_results/run_2026_02_23_03_17_00/
  python scripts/playback_fpv.py results
  python scripts/playback_fpv.py --pattern "*.png"

Controls:
  n / Space / Right  -> next image
  p / Left           -> previous image
  q / Esc            -> quit
"""

import argparse
import glob
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RESULTS = _REPO_ROOT / "results"


def _find_results_dir():
    """Return a results directory: first run with frames/, else flat/nested with *_2d.png / *_3d.png."""
    if not _DEFAULT_RESULTS.is_dir():
        return None
    # Prefer LTL layout: results/ltl_results/run_.../frames/*.png
    ltl_dir = _DEFAULT_RESULTS / "ltl_results"
    if ltl_dir.is_dir():
        for run_dir in sorted(ltl_dir.iterdir(), reverse=True):
            if run_dir.is_dir() and (run_dir / "frames").is_dir():
                if list((run_dir / "frames").glob("*.png")):
                    return str(run_dir)
    # Prefer flat layout: ./results contains *_2d.png / *_3d.png
    for pat in ("*_2d.png", "*_3d.png"):
        if glob.glob(str(_DEFAULT_RESULTS / pat)):
            return str(_DEFAULT_RESULTS)
    # Else look for nested subdirs (backward compatibility)
    for d in sorted(_DEFAULT_RESULTS.iterdir()):
        if d.is_dir():
            for sub in sorted(d.iterdir()):
                if sub.is_dir():
                    if (sub / "frames").is_dir() and list((sub / "frames").glob("*.png")):
                        return str(sub)
                    if glob.glob(str(sub / "*_2d.png")) or glob.glob(str(sub / "*_3d.png")):
                        return str(sub)
            if glob.glob(os.path.join(str(d), "*_2d.png")) or glob.glob(os.path.join(str(d), "*_3d.png")):
                return str(d)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Display FPV frames or trajectory plot images from a results directory",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=None,
        help="Run dir (LTL: with frames/ subdir) or dir with *_2d.png / *_3d.png (auto-detected if omitted)",
    )
    parser.add_argument(
        "--pattern",
        default="*_2d.png,*_3d.png",
        help="Comma-separated glob patterns (default: '*_2d.png,*_3d.png')",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Play frames as video (auto-advance at --fps).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second when using --video (default: 10).",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = _find_results_dir()
    if results_dir is None:
        print(
            "No results directory found. Run the simulator first, or pass a path.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isdir(results_dir):
        print(f"Error: not a directory: {results_dir}", file=sys.stderr)
        sys.exit(1)

    results_path = Path(results_dir)
    frames_dir = results_path / "frames"
    if frames_dir.is_dir():
        files = sorted(frames_dir.glob("*.png"))
        files = [str(p) for p in files]
    else:
        patterns = [p.strip() for p in args.pattern.split(",")]
        files = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(results_dir, pat)))
        files = sorted(set(files))

    if not files:
        print(
            f"No images found in {results_dir} (looked for frames/*.png or {args.pattern})",
            file=sys.stderr,
        )
        sys.exit(1)

    import cv2

    delay_ms = int(1000 / args.fps) if args.video else 0
    print(f"Found {len(files)} images in {results_dir}")
    if args.video:
        print(f"Video mode: {args.fps} fps. q/Esc=quit  Space=pause")
    else:
        print("Controls: n/Space/Right=next  p/Left=prev  q/Esc=quit")

    idx = 0
    win_name = "FPV playback"
    paused = False
    while True:
        path = files[idx]
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: could not read {path}", file=sys.stderr)
            idx = (idx + 1) % len(files)
            continue

        cv2.imshow(win_name, img)
        cv2.setWindowTitle(win_name, f"[{idx + 1}/{len(files)}] {os.path.basename(path)}")

        # In video mode when not paused, wait up to delay_ms then auto-advance; else wait for key
        wait_ms = delay_ms if (args.video and not paused) else 0
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            break
        elif key in (ord("p"), 81, 2):  # p, Left arrow
            idx = (idx - 1) % len(files)
        elif key == ord(" ") and args.video:  # Space = pause/unpause
            paused = not paused
        elif args.video and not paused:
            # Timeout or next key: advance
            idx = (idx + 1) % len(files)
        elif key != 255:  # non-video or paused: advance on any key except q/p
            idx = (idx + 1) % len(files)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
