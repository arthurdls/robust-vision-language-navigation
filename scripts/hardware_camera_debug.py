#!/usr/bin/env python3
"""
Cycle through the local cameras attached to the hardware to figure out
which OpenCV camera source corresponds to which physical camera.

At startup, this script probes:

 - **V4L2 nodes**: every ``/dev/video*`` device node, opened by integer
   index. Catches USB UVC webcams and any CSI bridge that exposes itself
   through V4L2 (e.g. the v4l2-loopback fallback on some Jetson images).
 - **Argus / CSI cameras** (Jetson onboard MIPI-CSI ports): probed via
   ``nvarguscamerasrc sensor-id=N`` GStreamer pipelines. Skipped silently
   when the host's OpenCV has no GStreamer support or when no Argus
   sensors are detected.

Each working source becomes a slot you can flip through with ``n`` / ``p``
in a single cv2 preview window, with the source label, resolution, and
live FPS overlaid. Use this to figure out which ``--camera N`` (or which
GStreamer pipeline) to pass to ``scripts/run_hardware.py`` when the host
has multiple cameras attached.

Usage (from repo root):
  python scripts/hardware_camera_debug.py
  python scripts/hardware_camera_debug.py --max-index 8 --max-csi 4
  python scripts/hardware_camera_debug.py --no-csi          # USB only
  python scripts/hardware_camera_debug.py --extra-pipeline "v4l2src device=/dev/video2 ! videoconvert ! appsink"

Controls:
  n / Space / Right  next camera
  p / Left           previous camera
  Enter              print the --camera flag for run_hardware.py
  0-9                jump to that probed slot
  r                  re-probe (rescan everything)
  s                  save a snapshot to /tmp/cam_<label>_<ts>.png
  q / Esc            quit
"""

import argparse
import glob
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


WINDOW_NAME = "hardware_camera_debug"


@dataclass
class Source:
    """A probed camera source we can open and read frames from."""
    label: str  # human-readable, e.g. "v4l2:/dev/video0" or "csi:sensor-id=0"
    kind: str   # "v4l2" | "csi" | "pipeline"
    open_fn: Callable[[], "cv2.VideoCapture"]
    # The cv2 integer index that maps to this source, when it exists. Only
    # set for V4L2 sources, since run_hardware.py's --camera flag is typed
    # as int and goes straight into cv2.VideoCapture(N).
    index: Optional[int] = None


# ---------------------------------------------------------------------------
# Source builders
# ---------------------------------------------------------------------------

def _v4l2_indices_from_dev() -> List[int]:
    """Return the integer indices behind /dev/videoN device nodes, sorted."""
    paths = sorted(glob.glob("/dev/video*"))
    out: List[int] = []
    for p in paths:
        suffix = p[len("/dev/video"):]
        if suffix.isdigit():
            out.append(int(suffix))
    return sorted(set(out))


def _build_v4l2_sources(max_index: int) -> List[Source]:
    """All V4L2 indices to try.

    Prefer the indices that actually have a /dev/video* node; fall back to
    [0, max_index) when /dev is empty (non-Linux hosts, containers, etc).
    """
    indices = _v4l2_indices_from_dev()
    if not indices:
        indices = list(range(max_index))
    sources: List[Source] = []
    for idx in indices:
        path = Path(f"/dev/video{idx}")
        label = f"v4l2:{path}" if path.exists() else f"v4l2:index={idx}"

        def opener(i: int = idx) -> cv2.VideoCapture:
            # Hint cv2 to use the V4L2 backend directly; otherwise it may
            # try gstreamer / ffmpeg first and fail silently on some builds.
            return cv2.VideoCapture(i, cv2.CAP_V4L2)
        sources.append(Source(label=label, kind="v4l2", open_fn=opener, index=idx))
    return sources


def _csi_pipeline(
    sensor_id: int,
    capture_w: int = 1920,
    capture_h: int = 1080,
    display_w: int = 960,
    display_h: int = 540,
    fps: int = 30,
    flip_method: int = 0,
) -> str:
    """Standard nvarguscamerasrc -> appsink pipeline for a CSI sensor."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_w}, "
        f"height=(int){capture_h}, format=(string)NV12, "
        f"framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_w}, height=(int){display_h}, "
        f"format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=1 sync=false"
    )


def _build_csi_sources(max_csi: int, fps: int) -> List[Source]:
    """One Source per candidate CSI sensor-id, opened via GStreamer."""
    sources: List[Source] = []
    for sid in range(max_csi):
        pipeline = _csi_pipeline(sid, fps=fps)
        label = f"csi:sensor-id={sid}"

        def opener(p: str = pipeline) -> cv2.VideoCapture:
            return cv2.VideoCapture(p, cv2.CAP_GSTREAMER)
        sources.append(Source(label=label, kind="csi", open_fn=opener))
    return sources


def _build_extra_sources(pipelines: List[str]) -> List[Source]:
    out: List[Source] = []
    for p in pipelines:
        label = f"pipeline:{_shorten(p, 40)}"

        def opener(pp: str = p) -> cv2.VideoCapture:
            return cv2.VideoCapture(pp, cv2.CAP_GSTREAMER)
        out.append(Source(label=label, kind="pipeline", open_fn=opener))
    return out


def _shorten(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n - 3] + "..."


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def probe(sources: List[Source], open_timeout_s: float) -> List[Source]:
    """Return the subset of sources that open *and* yield a frame in time."""
    found: List[Source] = []
    for src in sources:
        try:
            cap = src.open_fn()
        except Exception as exc:
            print(f"  [{src.label}] open raised: {exc}")
            continue

        if not cap or not cap.isOpened():
            if cap is not None:
                cap.release()
            print(f"  [{src.label}] did not open")
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
            print(f"  [{src.label}] producing frames")
            found.append(src)
        else:
            print(f"  [{src.label}] opens but no frames within {open_timeout_s:.1f}s")
    return found


def _gstreamer_supported() -> bool:
    """Best-effort check that this OpenCV build can use the GStreamer backend."""
    info = cv2.getBuildInformation()
    match = re.search(r"GStreamer\s*:\s*(\S+)", info)
    if not match:
        return False
    value = match.group(1).strip().lower()
    return value not in ("no", "off", "none")


def collect_sources(args: argparse.Namespace) -> List[Source]:
    candidates: List[Source] = []
    candidates += _build_v4l2_sources(args.max_index)
    if args.csi:
        if _gstreamer_supported():
            candidates += _build_csi_sources(args.max_csi, args.csi_fps)
        else:
            print("  GStreamer support not detected in cv2; skipping CSI probe.")
    if args.extra_pipeline:
        candidates += _build_extra_sources(args.extra_pipeline)
    return candidates


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def overlay_status(
    frame: np.ndarray,
    *,
    slot: int,
    source: Source,
    total: int,
    fps: float,
    resolution: Tuple[int, int],
) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_h = 64
    canvas = frame.copy()
    cv2.rectangle(canvas, (0, 0), (w, bar_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(canvas, 0.55, frame, 0.45, 0.0, dst=frame)
    line1 = (
        f"slot {slot + 1}/{total}  {source.label}  "
        f"{resolution[0]}x{resolution[1]}  {fps:.1f} fps"
    )
    line2 = "n=next  p=prev  Enter=print flag  0-9=jump  r=rescan  s=snap  q=quit"
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


def label_slug(label: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in label)
    return safe.strip("_")[:64] or "cam"


def report_run_hardware_flag(source: Source) -> None:
    """Print what to pass to run_hardware.py to use this source.

    ``run_hardware.py``'s ``--camera`` flag is typed as ``int`` and is fed
    directly into ``cv2.VideoCapture``. Only V4L2 sources map cleanly. CSI
    and arbitrary GStreamer pipelines aren't accepted by ``--camera`` today,
    so we print explicit guidance instead of pretending otherwise.
    """
    print()
    print("=" * 60)
    print(f"Selected source: {source.label}")
    if source.kind == "v4l2" and source.index is not None:
        print(f"  Use:  python scripts/run_hardware.py --camera {source.index}")
    elif source.kind == "csi":
        print(
            "  This is a CSI / Argus camera (nvarguscamerasrc). "
            "run_hardware.py's --camera flag is an integer cv2 index and "
            "does not accept a GStreamer pipeline. Either expose the "
            "camera as a V4L2 node first (e.g. nvv4l2camerasrc -> "
            "v4l2loopback) and use that index, or run a small HTTP "
            "frame server fed by this pipeline and pass --camera_url."
        )
    else:
        print(
            "  This source is a custom GStreamer pipeline. run_hardware.py's "
            "--camera flag only takes an integer cv2 device index. Run a "
            "frame server that consumes this pipeline and pass "
            "--camera_url instead."
        )
    print("=" * 60)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--max-index", type=int, default=10,
        help=(
            "When /dev/video* enumeration is empty (non-Linux), probe V4L2 "
            "indices [0, max_index). On Jetson this fallback is rarely "
            "used: real device nodes drive the probe. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max-csi", type=int, default=4,
        help=(
            "Probe CSI sensor-ids [0, max_csi) via nvarguscamerasrc. Jetson "
            "Nano supports 0/1; Xavier and Orin can support more. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--csi-fps", type=int, default=30,
        help="Capture FPS used in the nvarguscamerasrc pipeline. (default: %(default)s)",
    )
    parser.add_argument(
        "--no-csi", dest="csi", action="store_false", default=True,
        help="Skip the CSI / Argus probe (USB only).",
    )
    parser.add_argument(
        "--extra-pipeline", action="append", default=[],
        help=(
            "Extra GStreamer pipeline string to probe; can be repeated. "
            "Must end in 'appsink'. Useful for nvv4l2camerasrc, "
            "v4l2src device=/dev/videoN, RTSP feeds, etc."
        ),
    )
    parser.add_argument(
        "--probe-timeout", type=float, default=2.0,
        help=(
            "Seconds to wait for the first frame from a candidate source "
            "before considering the slot dead. CSI cameras sometimes need a "
            "moment for nvargus-daemon to warm up. (default: %(default)s)"
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


def open_or_die(src: Source) -> Optional[cv2.VideoCapture]:
    try:
        cap = src.open_fn()
    except Exception as exc:
        print(f"Failed to open {src.label}: {exc}")
        return None
    if not cap or not cap.isOpened():
        if cap is not None:
            cap.release()
        print(f"Failed to open {src.label}.")
        return None
    return cap


def main() -> None:
    args = parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    print("Probing camera sources...")
    candidates = collect_sources(args)
    if not candidates:
        print("No candidate sources to probe (V4L2 nodes empty, CSI disabled).")
        sys.exit(1)
    sources = probe(candidates, args.probe_timeout)
    if not sources:
        print("No working cameras found.")
        sys.exit(1)
    print(f"Working sources ({len(sources)}):")
    for i, s in enumerate(sources):
        print(f"  [{i}] {s.label}")

    slot = 0
    cap = open_or_die(sources[slot])
    if cap is None:
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
                source=sources[slot],
                total=len(sources),
                fps=fps_estimate,
                resolution=last_resolution,
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 0xFF:
                continue

            if key in (ord("q"), 27):  # Esc
                break
            if key in (10, 13):  # Enter / Return
                report_run_hardware_flag(sources[slot])
                continue
            if key in (ord("n"), ord(" "), 83):  # Right arrow
                slot = (slot + 1) % len(sources)
                cap.release()
                cap = open_or_die(sources[slot])
                fps_estimate = 0.0
                if cap is None:
                    print(f"Camera disappeared at slot {slot}; rescanning.")
                    sources = probe(collect_sources(args), args.probe_timeout)
                    if not sources:
                        break
                    slot = 0
                    cap = open_or_die(sources[slot])
                    if cap is None:
                        break
                continue
            if key in (ord("p"), 81):  # Left arrow
                slot = (slot - 1) % len(sources)
                cap.release()
                cap = open_or_die(sources[slot])
                fps_estimate = 0.0
                if cap is None:
                    sources = probe(collect_sources(args), args.probe_timeout)
                    if not sources:
                        break
                    slot = 0
                    cap = open_or_die(sources[slot])
                    if cap is None:
                        break
                continue
            if key == ord("r"):
                cap.release()
                print("Rescanning...")
                sources = probe(collect_sources(args), args.probe_timeout)
                if not sources:
                    print("No working cameras after rescan.")
                    break
                slot = min(slot, len(sources) - 1)
                cap = open_or_die(sources[slot])
                fps_estimate = 0.0
                if cap is None:
                    break
                continue
            if key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                out = snapshot_dir / f"cam_{label_slug(sources[slot].label)}_{ts}.png"
                cv2.imwrite(str(out), frame)
                print(f"Saved snapshot {out}")
                continue
            if ord("0") <= key <= ord("9"):
                target = key - ord("0")
                if target < len(sources):
                    slot = target
                    cap.release()
                    cap = open_or_die(sources[slot])
                    fps_estimate = 0.0
                    if cap is None:
                        break
                else:
                    print(f"Slot {target} out of range (have {len(sources)}).")
                continue
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
