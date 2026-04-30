"""
Tier 1 tests for negative constraint support in LiveDiaryMonitor.
No API calls, no GPU, no simulator. All LLM responses are mocked.
Run: conda run -n rvln-sim pytest tests/test_negative_constraints.py -v
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.ai.diary_monitor import LiveDiaryMonitor, DiaryCheckResult


def _make_monitor(constraints=None):
    return LiveDiaryMonitor(
        subgoal="Approach the tree",
        check_interval=2,
        model="gpt-4o",
        negative_constraints=constraints or [],
    )


def test_monitor_accepts_empty_constraints():
    m = _make_monitor()
    assert m._negative_constraints == []


def test_monitor_accepts_constraints():
    m = _make_monitor(["stay away from building B", "do not fly over zone C"])
    assert len(m._negative_constraints) == 2
    assert "building B" in m._negative_constraints[0]


def test_monitor_default_no_constraints():
    m = LiveDiaryMonitor(subgoal="Go forward", check_interval=2, model="gpt-4o")
    assert m._negative_constraints == []


def test_constraints_block_empty_when_none():
    m = _make_monitor()
    assert m._constraints_block() == ""


def test_constraints_block_renders_correctly():
    m = _make_monitor(["stay away from building B", "do not fly over zone C"])
    block = m._constraints_block()
    assert "Active constraints" in block
    assert "stay away from building B" in block
    assert "do not fly over zone C" in block


@patch("rvln.ai.diary_monitor.query_vlm")
@patch("rvln.ai.diary_monitor.build_frame_grid")
@patch("rvln.ai.diary_monitor.sample_frames_every_n")
def test_constraint_violation_force_converges(mock_sample, mock_grid, mock_vlm):
    m = _make_monitor(["stay away from building B"])
    m._frame_paths = [Path(f"/tmp/f{i}.png") for i in range(4)]
    m._frame_timestamps = [float(i) for i in range(4)]
    m._step = 4

    mock_grid.return_value = MagicMock()
    mock_sample.return_value = m._frame_paths[-4:]
    mock_vlm.side_effect = [
        "Drone moved closer to building B.",
        '{"complete": false, "completion_percentage": 0.3, "on_track": false, '
        '"should_stop": false, "constraint_violated": true}',
    ]

    result = m._run_checkpoint()
    assert result.action == "force_converge"
    assert "constraint" in result.reasoning.lower()


@patch("rvln.ai.diary_monitor.query_vlm")
@patch("rvln.ai.diary_monitor.build_frame_grid")
@patch("rvln.ai.diary_monitor.sample_frames_every_n")
def test_no_violation_continues(mock_sample, mock_grid, mock_vlm):
    m = _make_monitor(["stay away from building B"])
    m._frame_paths = [Path(f"/tmp/f{i}.png") for i in range(4)]
    m._frame_timestamps = [float(i) for i in range(4)]
    m._step = 4

    mock_grid.return_value = MagicMock()
    mock_sample.return_value = m._frame_paths[-4:]
    mock_vlm.side_effect = [
        "Drone moved toward the tree, building B not visible.",
        '{"complete": false, "completion_percentage": 0.4, "on_track": true, '
        '"should_stop": false, "constraint_violated": false}',
    ]

    result = m._run_checkpoint()
    assert result.action == "continue"
