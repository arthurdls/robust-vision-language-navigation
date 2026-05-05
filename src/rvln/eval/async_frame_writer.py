"""Background frame writer.

Submits PNG writes to a single worker thread so the control loop never
blocks on disk I/O. close() drains the queue.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


_SENTINEL = object()


class AsyncFrameWriter:
    def __init__(self, frames_dir: Path, enabled: bool = True,
                 max_queue: int = 256) -> None:
        self._dir = Path(frames_dir)
        self._enabled = enabled
        if not enabled:
            self._q: Optional[queue.Queue] = None
            self._thread: Optional[threading.Thread] = None
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        self._q = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="frame-writer")
        self._thread.start()

    def write(self, filename: str, image: np.ndarray) -> None:
        if not self._enabled or self._q is None:
            return
        # block briefly if queue is saturated; avoids unbounded memory.
        self._q.put((filename, image))

    def _run(self) -> None:
        assert self._q is not None
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                return
            filename, image = item
            try:
                cv2.imwrite(str(self._dir / filename), image)
            except Exception:
                pass

    def close(self) -> None:
        if self._q is None or self._thread is None:
            return
        self._q.put(_SENTINEL)
        self._thread.join(timeout=30.0)
