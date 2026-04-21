"""Tests for LiveDiaryMonitor stall detection.

Covers the _is_stalled() heuristic (plateau detection across checkpoint
completion history) and its integration into _run_checkpoint() which
returns action="ask_help" when a stall is detected.
"""
from unittest.mock import patch, MagicMock
from pathlib import Path

from rvln.ai.diary_monitor import LiveDiaryMonitor


def _make_monitor_with_history(history, stall_window=3, stall_threshold=0.05):
    """Build a LiveDiaryMonitor with pre-loaded completion history for testing."""
    m = LiveDiaryMonitor(
        subgoal="move forward",
        check_interval=2,
        model="gpt-4o",
        stall_window=stall_window,
        stall_threshold=stall_threshold,
    )
    m._completion_history = list(history)
    return m


def test_no_stall_when_not_enough_history():
    """Stall detection needs at least stall_window checkpoints."""
    m = LiveDiaryMonitor.__new__(LiveDiaryMonitor)
    m._completion_history = [0.1, 0.12]
    m._stall_window = 3
    m._stall_threshold = 0.05
    m._stall_completion_floor = 0.8
    assert m._is_stalled() is False


def test_stall_detected_when_flat():
    m = LiveDiaryMonitor.__new__(LiveDiaryMonitor)
    m._completion_history = [0.30, 0.31, 0.32]
    m._stall_window = 3
    m._stall_threshold = 0.05
    m._stall_completion_floor = 0.8
    assert m._is_stalled() is True


def test_no_stall_when_progressing():
    m = LiveDiaryMonitor.__new__(LiveDiaryMonitor)
    m._completion_history = [0.30, 0.40, 0.50]
    m._stall_window = 3
    m._stall_threshold = 0.05
    m._stall_completion_floor = 0.8
    assert m._is_stalled() is False


def test_no_stall_when_completion_high():
    """Don't ask for help if already nearly done."""
    m = LiveDiaryMonitor.__new__(LiveDiaryMonitor)
    m._completion_history = [0.85, 0.86, 0.86]
    m._stall_window = 3
    m._stall_threshold = 0.05
    m._stall_completion_floor = 0.8
    assert m._is_stalled() is False


def test_stall_only_looks_at_last_window():
    """Earlier progress doesn't mask a recent plateau."""
    m = LiveDiaryMonitor.__new__(LiveDiaryMonitor)
    m._completion_history = [0.10, 0.20, 0.30, 0.31, 0.32]
    m._stall_window = 3
    m._stall_threshold = 0.05
    m._stall_completion_floor = 0.8
    assert m._is_stalled() is True


@patch("rvln.ai.diary_monitor.query_vlm")
@patch("rvln.ai.diary_monitor.build_frame_grid")
@patch("rvln.ai.diary_monitor.sample_frames_every_n")
def test_checkpoint_returns_ask_help_on_stall(mock_sample, mock_grid, mock_vlm):
    """When completion has plateaued, _run_checkpoint should return ask_help."""
    m = _make_monitor_with_history([0.30, 0.31])

    m._frame_paths = [Path(f"/tmp/f{i}.png") for i in range(4)]
    m._frame_timestamps = [float(i) for i in range(4)]
    m._step = 4

    mock_grid.return_value = MagicMock()
    mock_sample.return_value = m._frame_paths[-4:]
    mock_vlm.side_effect = [
        "No visible change.",
        '{"complete": false, "completion_percentage": 0.32, "on_track": true, "should_stop": false}',
    ]

    result = m._run_checkpoint()
    assert result.action == "ask_help"
    assert "stall" in result.reasoning.lower()


@patch("rvln.ai.diary_monitor.query_vlm")
@patch("rvln.ai.diary_monitor.build_frame_grid")
@patch("rvln.ai.diary_monitor.sample_frames_every_n")
def test_checkpoint_returns_continue_when_not_stalled(mock_sample, mock_grid, mock_vlm):
    """Normal progress should still return continue."""
    m = _make_monitor_with_history([0.10, 0.20])

    m._frame_paths = [Path(f"/tmp/f{i}.png") for i in range(4)]
    m._frame_timestamps = [float(i) for i in range(4)]
    m._step = 4

    mock_grid.return_value = MagicMock()
    mock_sample.return_value = m._frame_paths[-4:]
    mock_vlm.side_effect = [
        "Moved closer to target.",
        '{"complete": false, "completion_percentage": 0.35, "on_track": true, "should_stop": false}',
    ]

    result = m._run_checkpoint()
    assert result.action == "continue"
