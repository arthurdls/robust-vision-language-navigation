"""Lightweight per-step phase timer for the sim control loop.

Records phase durations (ms) for one control step and appends a JSON line
to the configured path on end_step(). Designed for negligible overhead so
it can stay enabled in production runs.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional


class StepTimer:
    def __init__(self, log_path: Path) -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self._log_path, "a", buffering=1)
        self._step: Optional[int] = None
        self._phase_ms: Dict[str, float] = {}
        self._step_start: Optional[float] = None

    def start_step(self, step: int) -> None:
        self._step = step
        self._phase_ms = {}
        self._step_start = time.perf_counter()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._phase_ms[f"{name}_ms"] = (time.perf_counter() - t0) * 1000.0

    def end_step(self) -> None:
        if self._step is None or self._step_start is None:
            return
        total_ms = (time.perf_counter() - self._step_start) * 1000.0
        rec = {"step": self._step, "total_ms": round(total_ms, 2)}
        for k, v in self._phase_ms.items():
            rec[k] = round(v, 2)
        self._fp.write(json.dumps(rec) + "\n")
        self._step = None
        self._step_start = None

    def flush(self) -> None:
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass
