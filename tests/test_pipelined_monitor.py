"""Tests for the pipelined goal-adherence monitor (time-based mode).

Mocks _timed_query_vlm with controllable latencies. No real VLM calls."""

import os
import sys
import threading
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.ai.goal_adherence_monitor import _atomic_write_text


def test_atomic_write_text_overwrites_existing_file(tmp_path):
    target = tmp_path / "diary.txt"
    target.write_text("old contents")
    _atomic_write_text(target, "new contents")
    assert target.read_text() == "new contents"


def test_atomic_write_text_creates_new_file(tmp_path):
    target = tmp_path / "diary.txt"
    _atomic_write_text(target, "hello")
    assert target.read_text() == "hello"


def test_atomic_write_text_no_partial_visible(tmp_path):
    """A reader holding the file open during the swap must always see either
    the old contents or the new contents, never a half-written byte sequence.

    Implementation uses os.replace which is atomic on POSIX.
    """
    target = tmp_path / "diary.txt"
    target.write_text("old")
    seen = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                seen.append(target.read_text())
            except FileNotFoundError:
                seen.append("__missing__")

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    for i in range(50):
        _atomic_write_text(target, f"new-{i}")
    stop.set()
    t.join(timeout=1.0)
    # Every observation must be either "old" or "new-NN" with NN in [0, 49].
    for s in seen:
        assert s == "old" or (s.startswith("new-") and 0 <= int(s.split("-")[1]) < 50), \
            f"saw partial write: {s!r}"
