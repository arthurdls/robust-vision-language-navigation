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


def test_apply_action_returns_post_step_image() -> None:
    """apply_action_poses must return the SimClient.step() image so the
    main loop can reuse it as next iteration's input frame."""
    from unittest.mock import MagicMock

    import numpy as np

    from rvln.sim.env_setup import apply_action_poses
    from rvln.sim.sim_client import SimClient

    fake_image = np.full((224, 224, 3), 42, dtype=np.uint8)
    client = MagicMock(spec=SimClient)
    client.step.return_value = (fake_image, [0, 0, 0], [0, 0, 0], 1)

    image, pose, steps = apply_action_poses(
        client,
        action_poses=[[0.1, 0.0, 0.0, 0.0]],
        initial_x=0.0, initial_y=0.0, initial_z=0.0, initial_yaw=0.0,
        sleep_s=0.0,
        drone_cam_id=0,
    )

    assert image is fake_image, "image returned must be the one from /step"
    assert steps == 1


def test_async_frame_writer_writes_in_background(tmp_path: Path) -> None:
    import numpy as np

    from rvln.eval.async_frame_writer import AsyncFrameWriter

    writer = AsyncFrameWriter(tmp_path, enabled=True)
    img = np.full((10, 10, 3), 7, dtype=np.uint8)

    t0 = time.perf_counter()
    for i in range(20):
        writer.write(f"frame_{i:06d}.png", img)
    submit_ms = (time.perf_counter() - t0) * 1000
    assert submit_ms < 20, "submit must be near-instant; was %.1fms" % submit_ms

    writer.close()  # blocks until queue drains

    assert sum(1 for _ in tmp_path.glob("frame_*.png")) == 20


def test_async_frame_writer_disabled_is_noop(tmp_path: Path) -> None:
    import numpy as np

    from rvln.eval.async_frame_writer import AsyncFrameWriter

    writer = AsyncFrameWriter(tmp_path, enabled=False)
    writer.write("x.png", np.zeros((4, 4, 3), dtype=np.uint8))
    writer.close()

    assert list(tmp_path.glob("*.png")) == []


def test_async_frame_writer_close_is_idempotent(tmp_path: Path) -> None:
    """Calling close() twice must not raise and must be a no-op after the first call."""
    from rvln.eval.async_frame_writer import AsyncFrameWriter

    writer = AsyncFrameWriter(tmp_path, enabled=True)
    writer.close()

    # Thread and queue should be cleared after the first close.
    assert writer._thread is None
    assert writer._q is None

    # Second close must not raise.
    writer.close()
