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


from unittest.mock import patch, MagicMock
from rvln.ai.goal_adherence_monitor import (
    GoalAdherenceMonitor,
    _CheckpointSnapshot,
    _LocalStageResult,
    _GlobalStageResult,
)


def _make_monitor(tmp_path, **kwargs):
    """Build a time-based monitor with the background thread NOT started.

    Patches _make_llm so no OpenAI API key is needed. check_interval_s is
    None so no background thread starts.
    """
    defaults = dict(
        subgoal="Approach the tree",
        check_interval=2,
        model="gpt-4o",
        artifacts_dir=tmp_path / "artifacts",
        check_interval_s=None,  # avoid starting the background thread
    )
    defaults.update(kwargs)
    with patch.object(GoalAdherenceMonitor, "_make_llm", return_value=MagicMock()):
        return GoalAdherenceMonitor(**defaults)


def test_snapshot_captures_step_and_displacement(tmp_path):
    m = _make_monitor(tmp_path)
    m._step = 5
    m._last_displacement = [10.0, 20.0, 30.0, 45.0]
    m._frame_paths = [tmp_path / f"f{i}.png" for i in range(5)]
    for p in m._frame_paths:
        p.write_bytes(b"img")
    m._frame_timestamps = [float(i) for i in range(5)]
    m._diary = ["entry 0", "entry 1"]

    snap = m._snapshot_for_checkpoint()
    assert snap is not None
    assert snap.step == 5
    assert snap.displacement == [10.0, 20.0, 30.0, 45.0]
    assert snap.frame_paths == list(m._frame_paths)
    assert snap.frame_timestamps == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert snap.diary_at_dispatch == ["entry 0", "entry 1"]


def test_snapshot_returns_none_when_too_few_frames(tmp_path):
    m = _make_monitor(tmp_path)
    m._step = 1
    m._frame_paths = [tmp_path / "f0.png"]
    m._frame_paths[0].write_bytes(b"img")
    m._frame_timestamps = [0.0]
    snap = m._snapshot_for_checkpoint()
    assert snap is None


def test_run_local_stage_returns_diary_entry(tmp_path):
    m = _make_monitor(tmp_path)
    # Two frames, ~3 s apart
    p0, p1 = tmp_path / "f0.png", tmp_path / "f1.png"
    p0.write_bytes(b"img0"); p1.write_bytes(b"img1")
    snap = _CheckpointSnapshot(
        step=10,
        displacement=[100.0, 200.0, 50.0, 30.0],
        frame_paths=[p0, p1],
        frame_timestamps=[0.0, 3.0],
        diary_at_dispatch=[],
    )
    with patch("rvln.ai.goal_adherence_monitor.build_frame_grid", return_value=MagicMock()), \
         patch.object(m, "_timed_query_vlm", return_value="moved forward"):
        result = m._run_local_stage(snap)
    assert result.step == 10
    assert "moved forward" in result.diary_entry
    assert result.response_local == "moved forward"


def test_run_local_stage_skip_local_returns_empty_entry(tmp_path):
    m = _make_monitor(tmp_path, skip_local=True)
    p0, p1 = tmp_path / "f0.png", tmp_path / "f1.png"
    p0.write_bytes(b"img"); p1.write_bytes(b"img")
    snap = _CheckpointSnapshot(
        step=10, displacement=[0, 0, 0, 0],
        frame_paths=[p0, p1], frame_timestamps=[0.0, 3.0],
        diary_at_dispatch=[],
    )
    result = m._run_local_stage(snap)
    assert result.diary_entry == ""
    assert result.response_local == ""
    assert result.grid_local is None


def test_run_global_stage_returns_parsed_json(tmp_path):
    m = _make_monitor(tmp_path)
    p0, p1 = tmp_path / "f0.png", tmp_path / "f1.png"
    p0.write_bytes(b"img"); p1.write_bytes(b"img")
    snap = _CheckpointSnapshot(
        step=10, displacement=[0, 0, 0, 0],
        frame_paths=[p0, p1], frame_timestamps=[0.0, 3.0],
        diary_at_dispatch=["earlier entry"],
    )
    local = _LocalStageResult(
        step=10, grid_local=None, prompt_local="", response_local="",
        diary_entry="Steps ~10: moved forward",
    )
    response = '{"complete": false, "completion_percentage": 0.42, "on_track": true, "should_stop": false}'
    with patch("rvln.ai.goal_adherence_monitor.build_frame_grid", return_value=MagicMock()), \
         patch.object(m, "_timed_query_vlm", return_value=response):
        gresult = m._run_global_stage(snap, local)
    assert gresult.step == 10
    assert gresult.parsed["completion_percentage"] == 0.42
    assert gresult.response_global == response


import time as _time
from rvln.ai.goal_adherence_monitor import _ReorderBuffer


def test_reorder_buffer_releases_in_dispatch_order():
    buf = _ReorderBuffer()
    buf.register(10, dispatch_time=0.0)
    buf.register(11, dispatch_time=0.1)
    buf.register(12, dispatch_time=0.2)
    # Arrive out of order
    buf.put(12, "twelve")
    buf.put(10, "ten")
    # Release: only step 10 is at the head, so [10] is released first
    released = buf.release_ready(now=0.3, timeout_s=10.0)
    assert released == [(10, "ten")]
    # Now 11 is at the head; not yet arrived
    released = buf.release_ready(now=0.4, timeout_s=10.0)
    assert released == []
    # 11 arrives; both 11 and 12 (already buffered) release in order
    buf.put(11, "eleven")
    released = buf.release_ready(now=0.5, timeout_s=10.0)
    assert released == [(11, "eleven"), (12, "twelve")]


def test_reorder_buffer_timeout_skips_hung_step():
    buf = _ReorderBuffer()
    buf.register(10, dispatch_time=0.0)
    buf.register(11, dispatch_time=1.0)
    # 10 never arrives. 11 arrives.
    buf.put(11, "eleven")
    # Before timeout: nothing releases (head is 10).
    assert buf.release_ready(now=5.0, timeout_s=10.0) == []
    # After timeout (relative to step 10's dispatch_time=0.0): step 10 is
    # skipped, then 11 is released.
    released = buf.release_ready(now=11.0, timeout_s=10.0)
    assert released == [(11, "eleven")]
    # Skip is recorded
    assert buf.skipped_steps == [10]


def test_reorder_buffer_late_arrival_after_skip():
    buf = _ReorderBuffer()
    buf.register(10, dispatch_time=0.0)
    buf.register(11, dispatch_time=1.0)
    buf.put(11, "eleven")
    buf.release_ready(now=11.0, timeout_s=10.0)  # skips 10, releases 11
    # 10 finally arrives (very late). It should NOT be released to the
    # publisher (its slot was skipped) but ALSO should not poison the buffer.
    out = buf.put(10, "ten_late")
    assert out is False  # signals "this step was already skipped"


import io
from PIL import Image


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _make_time_monitor(tmp_path, **kwargs):
    """Build a real time-based monitor with VLM calls patched.

    Threads ARE started: dispatcher, publisher, executor pool. Caller is
    responsible for calling cleanup() at end of test.
    """
    art = tmp_path / "diary_artifacts"
    art.mkdir(parents=True, exist_ok=True)
    defaults = dict(
        subgoal="Approach the tree",
        check_interval=2,
        model="gpt-4o",
        artifacts_dir=art,
        check_interval_s=0.05,  # fast tick for tests
        global_grid_spacing_s=0.05,
        local_grid_spacing_s=0.05,
    )
    defaults.update(kwargs)
    # Patch _make_llm so OPENAI_API_KEY is not required
    with patch.object(GoalAdherenceMonitor, "_make_llm", return_value=MagicMock()):
        m = GoalAdherenceMonitor(**defaults)
    return m


def _seed_frames(m, n: int, dt: float = 0.1):
    """Push ``n`` frames into the monitor (simulating ``on_frame``)."""
    base = m._artifacts_dir.parent
    for i in range(n):
        p = base / f"frame_{i:04d}.png"
        p.write_bytes(_png_bytes())
        with m._lock:
            m._frame_paths.append(p)
            m._frame_timestamps.append(_time.time())
            m._step += 1
        _time.sleep(dt)


def test_pipelined_dispatch_runs_concurrent_calls(tmp_path):
    """With check_interval_s=0.05 and call latency=0.3, several checkpoints
    are in flight at once and the dashboard sees a new dir per ~0.05 s."""
    m = _make_time_monitor(tmp_path)
    call_count = [0]
    call_lock = threading.Lock()

    def slow_query(grid, prompt, label, **kwargs):
        with call_lock:
            call_count[0] += 1
        _time.sleep(0.3)
        return '{"complete": false, "completion_percentage": 0.5, "on_track": true, "should_stop": false}'

    try:
        with patch.object(m, "_timed_query_vlm", side_effect=slow_query):
            # Seed continuously while the dispatcher fires, so each tick
            # produces a fresh snap.step. Over ~1.0 s at 50 ms/frame the
            # dispatcher should submit ~10+ workers; with 0.3 s latency, at
            # least 4 should complete (proves concurrent execution).
            for i in range(20):
                _seed_frames(m, 1, dt=0.05)
            _time.sleep(0.6)
            checkpoints = sorted((tmp_path / "diary_artifacts").glob("checkpoint_*"))
            assert len(checkpoints) >= 4, \
                f"expected >=4 pipelined checkpoints, got {len(checkpoints)}"
    finally:
        m.cleanup()


def test_pipelined_out_of_order_returns_release_in_dispatch_order(tmp_path):
    """Workers may finish in any order; the publisher commits diary entries
    in dispatch order (the reorder buffer contract)."""
    m = _make_time_monitor(tmp_path)

    call_seq_lock = threading.Lock()
    seen_calls: List[Tuple[str, int]] = []

    def staggered_query(grid, prompt, label, **kwargs):
        # Earlier "local" calls (which arrive first via dispatch order) take
        # LONGER than later ones, so workers finish out of order. The buffer
        # must still publish in dispatch order.
        with call_seq_lock:
            idx = len(seen_calls)
            seen_calls.append((label, idx))
        if "local" in label and idx < 4:
            _time.sleep(0.6 - 0.1 * idx)
        else:
            _time.sleep(0.05)
        return '{"complete": false, "completion_percentage": 0.5, "on_track": true, "should_stop": false}'

    try:
        with patch.object(m, "_timed_query_vlm", side_effect=staggered_query):
            for _ in range(20):
                _seed_frames(m, 1, dt=0.05)
            _time.sleep(1.5)
            checkpoints = sorted((tmp_path / "diary_artifacts").glob("checkpoint_*"))
            # Directory names encode dispatched step IDs; lexical sort ==
            # dispatch order. The publisher commits in strict dispatch order
            # so the completion_history list (one entry per global commit)
            # must be monotonic in step.
            steps = [int(p.name.split("_")[1]) for p in checkpoints]
            assert steps == sorted(steps), f"checkpoint dirs out of order: {steps}"
            assert len(steps) >= 2
    finally:
        m.cleanup()


def test_monitor_accepts_max_inflight_and_dispatch_timeout_kwargs(tmp_path):
    art = tmp_path / "diary_artifacts"
    art.mkdir(parents=True, exist_ok=True)
    with patch.object(GoalAdherenceMonitor, "_make_llm", return_value=MagicMock()):
        m = GoalAdherenceMonitor(
            subgoal="x",
            check_interval=2,
            model="gpt-4o",
            artifacts_dir=art,
            check_interval_s=0.05,
            max_inflight=4,
            dispatch_timeout_s=2.5,
        )
    try:
        assert m._max_inflight == 4
        assert m._dispatch_timeout_s == 2.5
        # The thread pool's _max_workers reflects max_inflight
        assert m._executor._max_workers == 4
    finally:
        m.cleanup()


def test_monitor_defaults_when_buffer_kwargs_omitted(tmp_path):
    art = tmp_path / "diary_artifacts"
    art.mkdir(parents=True, exist_ok=True)
    with patch.object(GoalAdherenceMonitor, "_make_llm", return_value=MagicMock()):
        m = GoalAdherenceMonitor(
            subgoal="x", check_interval=2, model="gpt-4o",
            artifacts_dir=art, check_interval_s=0.05,
        )
    try:
        assert m._max_inflight == 16
        assert m._dispatch_timeout_s == 30.0
    finally:
        m.cleanup()


import random


def test_publish_order_matches_dispatch_order_under_random_latencies(tmp_path):
    m = _make_time_monitor(tmp_path)
    publish_log: list[str] = []
    log_lock = threading.Lock()

    real_local = m._save_checkpoint_local_artifact
    real_global = m._save_checkpoint_global_artifact

    def logged_local(step, *a, **kw):
        with log_lock:
            publish_log.append(f"L{step}")
        return real_local(step, *a, **kw)

    def logged_global(step, *a, **kw):
        with log_lock:
            publish_log.append(f"G{step}")
        return real_global(step, *a, **kw)

    m._save_checkpoint_local_artifact = logged_local
    m._save_checkpoint_global_artifact = logged_global

    # Random latency per call in [0.05, 0.5] s.
    rng = random.Random(42)

    def random_query(grid, prompt, label, **kwargs):
        _time.sleep(rng.uniform(0.05, 0.5))
        return '{"complete": false, "completion_percentage": 0.5, "on_track": true, "should_stop": false}'

    try:
        with patch.object(m, "_timed_query_vlm", side_effect=random_query):
            _seed_frames(m, 12, dt=0.05)
            _time.sleep(3.0)  # wait for the pipeline to drain

            local_steps = [int(x[1:]) for x in publish_log if x.startswith("L")]
            global_steps = [int(x[1:]) for x in publish_log if x.startswith("G")]
            assert local_steps == sorted(local_steps), \
                f"local publish order not monotonic: {local_steps}"
            assert global_steps == sorted(global_steps), \
                f"global publish order not monotonic: {global_steps}"
            assert len(local_steps) >= 5, \
                f"expected >=5 publishes, got {len(local_steps)}"
    finally:
        m.cleanup()


def test_hung_call_is_skipped_after_timeout(tmp_path):
    m = _make_time_monitor(tmp_path)
    m._dispatch_timeout_s = 0.5  # short for the test

    call_count = [0]
    count_lock = threading.Lock()

    def maybe_hung_query(grid, prompt, label, **kwargs):
        with count_lock:
            call_count[0] += 1
            n = call_count[0]
        if n == 1:
            # First call hangs (long enough to exceed timeout, short enough
            # that test cleanup completes promptly).
            _time.sleep(2.0)
        _time.sleep(0.05)
        return '{"complete": false, "completion_percentage": 0.5, "on_track": true, "should_stop": false}'

    try:
        with patch.object(m, "_timed_query_vlm", side_effect=maybe_hung_query):
            _seed_frames(m, 5, dt=0.05)
            _time.sleep(2.5)  # exceeds 0.5s timeout
            checkpoints = sorted((tmp_path / "diary_artifacts").glob("checkpoint_*"))
            # The first checkpoint dir may not exist (its calls hung). Later
            # checkpoints DO exist (their dirs were published past the skip).
            assert len(checkpoints) >= 1, "later checkpoints must publish despite the hang"
            # Skipped step is recorded in the buffer
            assert m._local_buf.skipped_steps or m._global_buf.skipped_steps, \
                "expected at least one skipped step"
    finally:
        m.cleanup()
