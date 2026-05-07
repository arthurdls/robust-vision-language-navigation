"""Verify dashboard picks the latest checkpoint by directory name (zero-padded
step), not by filesystem mtime. Required for pipelined monitor: a late-returning
older call writes its checkpoint dir AFTER a newer one, giving it a newer mtime;
the dashboard must not regress to the older checkpoint."""

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.mininav.dashboard import build_run_state, _resolve_image_path


def _make_cp(run_dir: Path, step: int, *, mtime: float | None = None) -> Path:
    cp = run_dir / "subgoal_01_ascend" / "diary_artifacts" / f"checkpoint_{step:04d}"
    cp.mkdir(parents=True, exist_ok=True)
    (cp / "diary.txt").write_text(f"diary at step {step}")
    (cp / "response_global.txt").write_text(f"resp at step {step}")
    (cp / "grid_local.png").write_bytes(b"L")
    (cp / "grid_global.png").write_bytes(b"G")
    if mtime is not None:
        for child in cp.rglob("*"):
            os.utime(child, (mtime, mtime))
        os.utime(cp, (mtime, mtime))
    return cp


def test_build_run_state_picks_highest_step_even_when_older_has_newer_mtime(tmp_path):
    # Step 11 written first with old mtime; step 10 written second with new mtime
    # (simulating a late-returning older OpenAI call).
    _make_cp(tmp_path, 11, mtime=1000.0)
    _make_cp(tmp_path, 10, mtime=2000.0)  # newer mtime, lower step
    state = build_run_state(tmp_path)
    assert state.checkpoint_label == "checkpoint_0011", \
        f"expected step 11 (highest by name), got {state.checkpoint_label}"
    assert state.diary_text == "diary at step 11"
    assert state.response_text == "resp at step 11"


def test_resolve_image_path_picks_highest_step_by_name(tmp_path):
    _make_cp(tmp_path, 11, mtime=1000.0)
    _make_cp(tmp_path, 10, mtime=2000.0)
    p_local = _resolve_image_path(tmp_path, "local")
    p_global = _resolve_image_path(tmp_path, "global")
    assert p_local is not None and p_local.parent.name == "checkpoint_0011"
    assert p_global is not None and p_global.parent.name == "checkpoint_0011"
