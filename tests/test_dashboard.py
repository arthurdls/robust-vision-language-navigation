"""Tests for rvln.mininav.dashboard -- web dashboard backend."""

import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
from rvln.mininav.dashboard import RunState, build_run_state


def _make_artifacts(run_dir: Path, subgoal_idx: int, subgoal_name: str,
                    checkpoint_idx: int, *,
                    diary: str = "", response: str = "",
                    local_png: bytes = b"", global_png: bytes = b"",
                    convergence_idx: int | None = None,
                    convergence_response: str = "",
                    convergence_png: bytes = b"") -> Path:
    """Helper: write a synthetic subgoal/checkpoint tree under run_dir."""
    sg_dir = run_dir / f"subgoal_{subgoal_idx:02d}_{subgoal_name}"
    art = sg_dir / "diary_artifacts"
    cp = art / f"checkpoint_{checkpoint_idx:04d}"
    cp.mkdir(parents=True, exist_ok=True)
    (cp / "diary.txt").write_text(diary)
    (cp / "response_global.txt").write_text(response)
    (cp / "grid_local.png").write_bytes(local_png)
    (cp / "grid_global.png").write_bytes(global_png)
    if convergence_idx is not None:
        cv = art / f"convergence_{convergence_idx:03d}"
        cv.mkdir(parents=True, exist_ok=True)
        (cv / "response_00.txt").write_text(convergence_response)
        (cv / "grid_convergence_00.png").write_bytes(convergence_png)
    return sg_dir


class TestBuildRunState:
    def test_empty_run_dir(self, tmp_path):
        state = build_run_state(tmp_path)
        assert state.subgoals == []
        assert state.active_subgoal is None
        assert state.checkpoint_label is None
        assert state.run_complete is False

    def test_single_subgoal_single_checkpoint(self, tmp_path):
        _make_artifacts(tmp_path, 1, "ascend", 10,
                        diary="d", response="r",
                        local_png=b"L", global_png=b"G")
        state = build_run_state(tmp_path)
        assert [s["index"] for s in state.subgoals] == [1]
        assert state.subgoals[0]["name"] == "ascend"
        assert state.active_subgoal == {"index": 1, "name": "ascend"}
        assert state.checkpoint_label == "checkpoint_0010"
        assert state.checkpoint_count == 1
        assert state.diary_text == "d"
        assert state.response_text == "r"
        assert state.local_image_mtime > 0
        assert state.global_image_mtime > 0

    def test_checkpoint_count_in_active_subgoal(self, tmp_path):
        _make_artifacts(tmp_path, 1, "ascend", 10)
        _make_artifacts(tmp_path, 1, "ascend", 20)
        _make_artifacts(tmp_path, 1, "ascend", 30)
        state = build_run_state(tmp_path)
        assert state.checkpoint_count == 3

    def test_active_subgoal_is_one_with_newest_checkpoint(self, tmp_path):
        _make_artifacts(tmp_path, 1, "ascend", 10)
        time.sleep(0.02)
        _make_artifacts(tmp_path, 2, "cruise", 5)
        state = build_run_state(tmp_path)
        assert state.active_subgoal["index"] == 2
        assert state.active_subgoal["name"] == "cruise"

    def test_run_complete_when_run_info_exists(self, tmp_path):
        (tmp_path / "run_info.json").write_text("{}")
        state = build_run_state(tmp_path)
        assert state.run_complete is True

    def test_convergence_panel_populated(self, tmp_path):
        _make_artifacts(tmp_path, 1, "ascend", 10,
                        convergence_idx=2,
                        convergence_response="converged: yes",
                        convergence_png=b"C")
        state = build_run_state(tmp_path)
        assert state.convergence_label == "convergence_002"
        assert state.convergence_response == "converged: yes"
        assert state.convergence_image_mtime > 0
