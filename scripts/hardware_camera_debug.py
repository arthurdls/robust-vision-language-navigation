#!/usr/bin/env python3
"""
Cycle through the local cameras attached to the hardware to figure out
which OpenCV device index corresponds to which physical camera.

Probes indices [0, --max-index) at startup, keeps the ones that open and
produce a frame, and displays them one at a time in a single cv2 window
with the index, resolution, and a live FPS readout overlaid. Useful when
plugging the MiniNav drone's USB camera into a host with several
built-in or USB cameras and you need to figure out which `--camera N`
to pass to scripts/run_hardware.py.

Usage (from repo root):
  python scripts/hardware_camera_debug.py
  python scripts/hardware_camera_debug.py --max-index 8 --window-size 720

Controls:
  n / Space / Right  next camera
  p / Left           previous camera
  0-9                jump to that probed slot
  r                  re-probe (rescan all indices)
  s                  save a snapshot to /tmp/cam_<index>_<ts>.png
  q / Esc            quit
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


WINDOW_NAME = "hardware_camera_debug"


def probe_cameras(max_index: int, open_timeout_s: float) -> List[int]:
    """Return the indices in [0, max_index) that open and yield one frame."""
    found: List[int] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue

        ok = False
        deadline = time.time() + open_timeout_s
        while time.time() < deadline:
            grabbed, frame = cap.read()
            if grabbed and frame is not None:
                ok = True
                break
            time.sleep(0.05)
        cap.release()
        if ok:
            print(f"  [{idx}] producing frames")
            found.append(idx)
        else:
            print(f"  [{idx}] opens but no frames within {open_timeout_s:.1f}s")
    return found


def open_camera(idx: int) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def overlay_status(
    frame: np.ndarray,
    *,
    slot: int,
    index: int,
    total: int,
    fps: float,
    resolution: Tuple[int, int],
) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_h = 64
    canvas = frame.copy()
    cv2.rectangle(canvas, (0, 0), (w, bar_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(canvas, 0.55, frame, 0.45, 0.0, dst=frame)
    line1 = f"slot {slot + 1}/{total}  index {index}  {resolution[0]}x{resolution[1]}  {fps:.1f} fps"
    line2 = "n/Space=next  p=prev  0-9=jump  r=rescan  s=snapshot  q/Esc=quit"
    cv2.putText(
        frame, line1, (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, line2, (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
    )
    return frame


def fit_to_window(frame: np.ndarray, target: int) -> np.ndarray:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= target:
        return frame
    scale = target / float(longest)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--max-index", type=int, default=10,
        help="Probe indices [0, max_index) at startup. (default: %(default)s)",
    )
    parser.add_argument(
        "--probe-timeout", type=float, default=1.5,
        help=(
            "Seconds to wait for the first frame from a camera before "
            "considering the slot dead. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--window-size", type=int, default=800,
        help="Longest edge of the displayed frame in pixels. (default: %(default)s)",
    )
    parser.add_argument(
        "--snapshot-dir", type=str, default="/tmp",
        help="Where 's' writes snapshots. (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Probing camera indices 0..{args.max_index - 1}...")
    indices = probe_cameras(args.max_index, args.probe_timeout)
    if not indices:
        print(f"No working cameras found in [0, {args.max_index}).")
        sys.exit(1)
    print(f"Working cameras: {indices}")

    slot = 0
    cap = open_camera(indices[slot])
    if cap is None:
        print(f"Could not reopen camera {indices[slot]}.")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    last_frame_t = time.time()
    fps_estimate = 0.0
    fps_alpha = 0.1
    last_resolution = (0, 0)

    try:
        while True:
            grabbed, frame = cap.read()
            if not grabbed or frame is None:
                # Dropped frame, briefly retry rather than exiting.
                time.sleep(0.02)
                continue

            now = time.time()
            dt = now - last_frame_t
            last_frame_t = now
            if dt > 0:
                inst = 1.0 / dt
                fps_estimate = (
                    inst if fps_estimate == 0.0
                    else (1.0 - fps_alpha) * fps_estimate + fps_alpha * inst
                )
            last_resolution = (frame.shape[1], frame.shape[0])

            display = fit_to_window(frame, args.window_size)
            overlay_status(
                display,
                slot=slot,
                index=indices[slot],
                total=len(indices),
                fps=fps_estimate,
                resolution=last_resolution,
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 0xFF:
                continue

            if key in (ord("q"), 27):  # Esc
                break
            if key in (ord("n"), ord(" "), 83):  # Right arrow on most keymaps
                slot = (slot + 1) % len(indices)
                cap.release()
                cap = open_camera(indices[slot])
                fps_estimate = 0.0
                if cap is None:
                    print(f"Camera {indices[slot]} disappeared; rescanning.")
                    indices = probe_cameras(args.max_index, args.probe_timeout)
                    if not indices:
                        break
                    slot = 0
                    cap = open_camera(indices[slot])
                    if cap is None:
                        break
                continue
            if key in (ord("p"), 81):  # Left arrow
                slot = (slot - 1) % len(indices)
                cap.release()
                cap = open_camera(indices[slot])
                fps_estimate = 0.0
                if cap is None:
                    indices = probe_cameras(args.max_index, args.probe_timeout)
                    if not indices:
                        break
                    slot = 0
                    cap = open_camera(indices[slot])
                    if cap is None:
                        break
                continue
            if key == ord("r"):
                cap.release()
                print("Rescanning...")
                indices = probe_cameras(args.max_index, args.probe_timeout)
                if not indices:
                    print("No working cameras after rescan.")
                    break
                slot = min(slot, len(indices) - 1)
                cap = open_camera(indices[slot])
                fps_estimate = 0.0
                if cap is None:
                    break
                continue
            if key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                out = snapshot_dir / f"cam_{indices[slot]}_{ts}.png"
                cv2.imwrite(str(out), frame)
                print(f"Saved snapshot {out}")
                continue
            if ord("0") <= key <= ord("9"):
                target = key - ord("0")
                if target < len(indices):
                    slot = target
                    cap.release()
                    cap = open_camera(indices[slot])
                    fps_estimate = 0.0
                    if cap is None:
                        print(f"Camera {indices[slot]} did not reopen.")
                        break
                else:
                    print(f"Slot {target} out of range (have {len(indices)}).")
                continue
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
