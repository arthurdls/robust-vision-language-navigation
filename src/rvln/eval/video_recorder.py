"""Real-time playback video recorder shared by sim and hardware paths.

The recorder is decoupled from the control loop: a daemon thread reads
the latest captured camera frame from a buffer at the configured fps
and writes it to an mp4. On the hardware path the buffer is the
ThreadedCamera itself (frames stream at 30 fps from the real camera).
On the sim path the main thread is the only producer, so we use a
LatestFrameBuffer that the main thread updates whenever it captures
from the sim. The recorder still ticks at 30 fps and writes the most
recent buffered frame, so the video keeps growing during VLM polls and
other blocking phases without adding any sim-server load.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LatestFrameBuffer:
    """Thread-safe single-slot buffer holding the most recent frame.

    Quacks like the hardware ThreadedCamera so the existing VideoRecorder
    can read from it unchanged.
    """

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def update(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            return
        with self._lock:
            self._frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            frame = self._frame
        if frame is None:
            return False, None
        return True, frame


class VideoRecorder:
    """Stream camera frames into a single playback video on a dedicated thread.

    Runs independently of the control loop: the thread reads the latest frame
    from the camera at the configured fps and pushes it to disk. That way
    the video keeps growing during everything the main thread blocks on,
    convergence VLM polls, operator stdin prompts, OpenVLA round-trips,
    long checkpoints, so playback is continuous from camera-up to
    shutdown rather than freezing at the first checkpoint.

    The writer opens lazily on the first frame so we don't commit to a size
    before the camera has produced anything, and close() runs from the main
    finally block (or atexit) so SIGINT / errors all yield a playable file
    (the moov atom for mp4 is only written on release()).
    """

    def __init__(self, video_path: Path, fps: float, camera: Any):
        self.video_path = Path(video_path)
        # Some codecs reject fps <= 0; clamp to 1 fps as a safety floor.
        self.fps = float(fps) if fps and fps > 0 else 1.0
        self.camera = camera
        self._writer: Optional[cv2.VideoWriter] = None
        self._open_failed = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frames_written = 0

    def start(self) -> None:
        """Begin background capture. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="video-recorder", daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        period = 1.0 / self.fps
        next_t = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now < next_t:
                self._stop.wait(min(period, next_t - now))
                continue
            try:
                ok, frame = self.camera.read()
            except Exception as exc:
                logger.warning("VideoRecorder: camera.read raised (%s).", exc)
                ok, frame = False, None
            if ok and frame is not None:
                self._write_frame(frame)
            next_t += period
            if next_t < now - period:
                next_t = now + period

    def _open(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (w, h))
        if not writer.isOpened():
            logger.warning(
                "VideoRecorder: failed to open writer for %s (fourcc=mp4v, "
                "%dx%d @ %.2f fps). Skipping video output.",
                self.video_path, w, h, self.fps,
            )
            self._open_failed = True
            return
        self._writer = writer

    def _write_frame(self, frame_bgr: np.ndarray) -> None:
        if self._open_failed:
            return
        if self._writer is None:
            self._open(frame_bgr)
            if self._writer is None:
                return
        try:
            self._writer.write(frame_bgr)
            self._frames_written += 1
        except Exception as exc:
            logger.warning("VideoRecorder: write failed (%s).", exc)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("VideoRecorder thread did not exit within 2s.")
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._frames_written:
            logger.info(
                "VideoRecorder: wrote %d frames to %s",
                self._frames_written, self.video_path,
            )


# ---------------------------------------------------------------------------
# Sim-path episode playback: idempotent setup, atexit teardown.
# ---------------------------------------------------------------------------

# WeakKeyDictionary so closed-over recorder/buffer don't leak when the env is
# garbage-collected, and so equality-by-id is used instead of __hash__/eq
# (SimClient may not be hashable in the usual way).
_active_recorders: "weakref.WeakKeyDictionary[Any, VideoRecorder]" = (
    weakref.WeakKeyDictionary()
)
_atexit_registered = False
_atexit_lock = threading.Lock()


def _close_all_recorders() -> None:
    """atexit hook: flush every recorder we started."""
    for env, rec in list(_active_recorders.items()):
        try:
            rec.close()
        except Exception as exc:
            logger.warning("Failed to close VideoRecorder during atexit: %s", exc)
        try:
            if hasattr(env, "_playback_buffer"):
                env._playback_buffer = None
        except Exception:
            pass


def ensure_episode_playback(
    env: Any,
    run_dir: Path,
    fps: float,
) -> Optional[VideoRecorder]:
    """Idempotently start a real-time playback recorder for *env*.

    On first call attaches a `LatestFrameBuffer` to ``env._playback_buffer``
    and starts a `VideoRecorder` writing to ``run_dir / playback.mp4``.
    Subsequent calls for the same env are no-ops. The recorder is closed
    via atexit so SIGINT / normal exit yield a playable mp4.

    Callers that update the buffer (sim camera helpers in env_setup) check
    for ``env._playback_buffer`` and call ``.update(frame)`` on it.

    Returns the recorder on the first call, None on subsequent calls.
    """
    global _atexit_registered

    if env is None:
        return None
    if env in _active_recorders:
        return None

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    buffer = LatestFrameBuffer()
    recorder = VideoRecorder(
        video_path=run_dir / "playback.mp4",
        fps=fps,
        camera=buffer,
    )

    try:
        env._playback_buffer = buffer
    except Exception as exc:
        logger.warning(
            "Could not attach playback buffer to env (%s); skipping playback.", exc,
        )
        return None

    recorder.start()
    _active_recorders[env] = recorder

    with _atexit_lock:
        if not _atexit_registered:
            atexit.register(_close_all_recorders)
            _atexit_registered = True

    logger.info("Started playback recorder: %s @ %.1f fps", recorder.video_path, fps)
    return recorder


def stop_episode_playback(env: Any) -> None:
    """Explicitly stop and finalize the recorder for *env*, if any.

    Useful when a script wants the mp4 finalized before its own teardown
    (the atexit hook also covers this).
    """
    rec = _active_recorders.pop(env, None)
    if rec is not None:
        try:
            rec.close()
        finally:
            try:
                if hasattr(env, "_playback_buffer"):
                    env._playback_buffer = None
            except Exception:
                pass
