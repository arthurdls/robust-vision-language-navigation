import importlib.util
import json
import time
from pathlib import Path

import pytest


def _load_summarize():
    path = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_sim_loop.py"
    spec = importlib.util.spec_from_file_location("benchmark_sim_loop", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.summarize


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


def test_summarize_single_record_no_crash(tmp_path: Path) -> None:
    """statistics.quantiles requires >=2 data points; single-record JSONL must not crash."""
    summarize = _load_summarize()
    jsonl = tmp_path / "single.jsonl"
    jsonl.write_text(json.dumps({"step": 0, "get_frame_ms": 5.0, "total_ms": 5.0}) + "\n")
    summarize(jsonl)  # must not raise


def test_summarize_mixed_phase_records_no_crash(tmp_path: Path) -> None:
    """Phase columns discovered from records[0] only would drop keys present on later records."""
    summarize = _load_summarize()
    jsonl = tmp_path / "mixed.jsonl"
    records = [
        {"step": 0, "get_frame_ms": 3.0, "total_ms": 3.0},
        {"step": 1, "get_frame_ms": 4.0, "predict_ms": 20.0, "total_ms": 24.0},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    summarize(jsonl)  # must not raise; predict_ms must not be silently dropped
