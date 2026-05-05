import json
import time
from pathlib import Path

import pytest


def test_step_timer_records_phases(tmp_path: Path) -> None:
    from rvln.eval.step_timer import StepTimer

    log_path = tmp_path / "timings.jsonl"
    timer = StepTimer(log_path)

    timer.start_step(step=0)
    with timer.phase("get_frame"):
        time.sleep(0.01)
    with timer.phase("predict"):
        time.sleep(0.02)
    with timer.phase("apply_action"):
        time.sleep(0.005)
    timer.end_step()

    timer.flush()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["step"] == 0
    assert 8 <= rec["get_frame_ms"] <= 50
    assert 18 <= rec["predict_ms"] <= 60
    assert 4 <= rec["apply_action_ms"] <= 30
    assert "total_ms" in rec
