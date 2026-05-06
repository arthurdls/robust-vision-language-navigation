"""
Unit tests for GoalAdherenceMonitor core logic: on_convergence, on_frame checkpoint
triggering, _parse_global_response, _parse_json_response, and _format_displacement.

All VLM/LLM responses are mocked. No API calls, no GPU, no simulator.
Run: conda run -n rvln-sim pytest tests/test_diary_monitor_core.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.ai.goal_adherence_monitor import GoalAdherenceMonitor, DiaryCheckResult


def _make_monitor(**kwargs):
    defaults = dict(
        subgoal="Approach the tree",
        check_interval=2,
        model="gpt-4o",
    )
    defaults.update(kwargs)
    return GoalAdherenceMonitor(**defaults)


# -------------------------------------------------------------------------
# _parse_json_response
# -------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_plain_json(self):
        result = GoalAdherenceMonitor._parse_json_response(
            '{"complete": true, "completion_percentage": 0.95}'
        )
        assert result is not None
        assert result["complete"] is True
        assert result["completion_percentage"] == 0.95

    def test_markdown_code_fence(self):
        result = GoalAdherenceMonitor._parse_json_response(
            '```json\n{"complete": false, "on_track": true}\n```'
        )
        assert result is not None
        assert result["complete"] is False

    def test_markdown_fence_no_lang(self):
        result = GoalAdherenceMonitor._parse_json_response(
            '```\n{"complete": true}\n```'
        )
        assert result is not None
        assert result["complete"] is True

    def test_json_with_surrounding_text(self):
        result = GoalAdherenceMonitor._parse_json_response(
            'Here is the assessment: {"complete": false, "completion_percentage": 0.4} end'
        )
        assert result is not None
        assert result["complete"] is False

    def test_garbage_returns_none(self):
        result = GoalAdherenceMonitor._parse_json_response("no json here at all")
        assert result is None

    def test_empty_string_returns_none(self):
        result = GoalAdherenceMonitor._parse_json_response("")
        assert result is None

    def test_nested_json(self):
        result = GoalAdherenceMonitor._parse_json_response(
            '{"outer": {"inner": 1}, "complete": false}'
        )
        assert result is not None
        assert result["outer"]["inner"] == 1

    def test_whitespace_around_json(self):
        result = GoalAdherenceMonitor._parse_json_response(
            '  \n  {"complete": true}  \n  '
        )
        assert result is not None
        assert result["complete"] is True


# -------------------------------------------------------------------------
# _parse_global_response
# -------------------------------------------------------------------------

class TestParseGlobalResponse:
    def _make_monitor_for_global(self):
        m = _make_monitor()
        m._last_completion_pct = 0.3
        return m

    def test_complete_triggers_force_converge(self):
        m = self._make_monitor_for_global()
        result = m._parse_global_response(
            '{"complete": true, "completion_percentage": 0.95, "on_track": true, "should_stop": false}',
            "diary entry",
        )
        assert result.action == "force_converge"
        assert result.completion_pct == 0.95

    def test_should_stop_triggers_force_converge(self):
        m = self._make_monitor_for_global()
        result = m._parse_global_response(
            '{"complete": false, "completion_percentage": 0.6, "on_track": false, "should_stop": true}',
            "diary entry",
        )
        assert result.action == "force_converge"

    def test_normal_progress_continues(self):
        m = self._make_monitor_for_global()
        result = m._parse_global_response(
            '{"complete": false, "completion_percentage": 0.5, "on_track": true, "should_stop": false}',
            "diary entry",
        )
        assert result.action == "continue"
        assert result.completion_pct == 0.5

    def test_completion_pct_clamped_to_0_1(self):
        m = self._make_monitor_for_global()
        result = m._parse_global_response(
            '{"complete": false, "completion_percentage": 1.5, "on_track": true, "should_stop": false}',
            "diary entry",
        )
        assert result.completion_pct == 1.0

        result2 = m._parse_global_response(
            '{"complete": false, "completion_percentage": -0.5, "on_track": true, "should_stop": false}',
            "diary entry",
        )
        assert result2.completion_pct == 0.0

    def test_complete_and_should_stop_both_trigger_force_converge(self):
        """Both complete and should_stop lead to force_converge for verification."""
        m = self._make_monitor_for_global()
        result = m._parse_global_response(
            '{"complete": true, "completion_percentage": 0.95, "should_stop": true}',
            "diary entry",
        )
        assert result.action == "force_converge"


# -------------------------------------------------------------------------
# _format_displacement
# -------------------------------------------------------------------------

class TestFormatDisplacement:
    def test_typical_values(self):
        m = _make_monitor()
        m._last_displacement = [350.0, 120.0, 0.0, 15.0]
        formatted = m._format_displacement()
        assert "3.50" in formatted
        assert "1.20" in formatted
        assert "0.00" in formatted
        assert "15.0" in formatted

    def test_all_zeros(self):
        m = _make_monitor()
        m._last_displacement = [0.0, 0.0, 0.0, 0.0]
        formatted = m._format_displacement()
        assert "0.00" in formatted

    def test_negative_values(self):
        m = _make_monitor()
        m._last_displacement = [-200.0, -50.0, 100.0, -45.0]
        formatted = m._format_displacement()
        assert "-2.00" in formatted
        assert "-0.50" in formatted
        assert "1.00" in formatted
        assert "-45.0" in formatted

    def test_large_values(self):
        m = _make_monitor()
        m._last_displacement = [10000.0, 5000.0, 2000.0, 359.0]
        formatted = m._format_displacement()
        assert "100.00" in formatted
        assert "50.00" in formatted


# -------------------------------------------------------------------------
# on_convergence (mocked VLM)
# -------------------------------------------------------------------------

class TestOnConvergence:
    def _setup_monitor(self):
        m = _make_monitor()
        m._frame_paths = [Path(f"/tmp/f{i}.png") for i in range(4)]
        m._frame_timestamps = [float(i) for i in range(4)]
        m._step = 4
        m._diary = ["Steps 0-2: moved forward"]
        m._last_displacement = [100.0, 0.0, 0.0, 0.0]
        return m

    @patch("rvln.ai.goal_adherence_monitor.query_vlm")
    @patch("rvln.ai.goal_adherence_monitor.build_frame_grid")
    @patch("rvln.ai.goal_adherence_monitor.sample_frames_every_n")
    def test_complete_returns_stop(self, mock_sample, mock_grid, mock_vlm):
        m = self._setup_monitor()
        mock_sample.return_value = m._frame_paths[-4:]
        mock_grid.return_value = MagicMock()
        mock_vlm.return_value = json.dumps({
            "complete": True,
            "completion_percentage": 0.95,
            "diagnosis": "complete",
            "corrective_instruction": None,
        })

        result = m.on_convergence(Path("/tmp/f3.png"), displacement=[100.0, 0.0, 0.0, 0.0])
        assert result.action == "stop"
        assert result.completion_pct == 0.95

    @patch("rvln.ai.goal_adherence_monitor.query_vlm")
    @patch("rvln.ai.goal_adherence_monitor.build_frame_grid")
    @patch("rvln.ai.goal_adherence_monitor.sample_frames_every_n")
    def test_stopped_short_returns_command(self, mock_sample, mock_grid, mock_vlm):
        m = self._setup_monitor()
        mock_sample.return_value = m._frame_paths[-4:]
        mock_grid.return_value = MagicMock()
        mock_vlm.return_value = json.dumps({
            "complete": False,
            "completion_percentage": 0.4,
            "diagnosis": "stopped_short",
            "corrective_instruction": "move forward toward the tree",
        })

        result = m.on_convergence(Path("/tmp/f3.png"), displacement=[50.0, 0.0, 0.0, 0.0])
        assert result.action == "command"
        assert "forward" in result.new_instruction
        assert m._corrections_used == 1

    @patch("rvln.ai.goal_adherence_monitor.query_vlm")
    @patch("rvln.ai.goal_adherence_monitor.build_frame_grid")
    @patch("rvln.ai.goal_adherence_monitor.sample_frames_every_n")
    def test_corrections_counter_increments(self, mock_sample, mock_grid, mock_vlm):
        m = self._setup_monitor()
        mock_sample.return_value = m._frame_paths[-4:]
        mock_grid.return_value = MagicMock()
        mock_vlm.return_value = json.dumps({
            "complete": False,
            "completion_percentage": 0.3,
            "diagnosis": "stopped_short",
            "corrective_instruction": "keep going",
        })

        assert m._corrections_used == 0
        m.on_convergence(Path("/tmp/f3.png"))
        assert m._corrections_used == 1
        m.on_convergence(Path("/tmp/f3.png"))
        assert m._corrections_used == 2

    def test_max_corrections_returns_ask_help(self):
        m = self._setup_monitor()
        m._corrections_used = 15
        m._max_corrections = 15

        result = m.on_convergence(Path("/tmp/f3.png"))
        assert result.action == "ask_help"
        assert "Max corrections" in result.reasoning

    @patch("rvln.ai.goal_adherence_monitor.query_vlm")
    @patch("rvln.ai.goal_adherence_monitor.build_frame_grid")
    @patch("rvln.ai.goal_adherence_monitor.sample_frames_every_n")
    def test_diagnosis_complete_with_no_corrective_asks_help(self, mock_sample, mock_grid, mock_vlm):
        """complete=False with diagnosis='complete' and no corrective should
        pull the operator in (after a retry that also yields no corrective).
        Previously this silently stopped the subgoal; now it returns ask_help
        with header CONVERGENCE GAVE NO CORRECTIVE so the operator gets a
        chance to override or replan."""
        m = self._setup_monitor()
        mock_sample.return_value = m._frame_paths[-4:]
        mock_grid.return_value = MagicMock()
        mock_vlm.return_value = json.dumps({
            "complete": False,
            "completion_percentage": 0.90,
            "diagnosis": "complete",
            "corrective_instruction": None,
        })

        result = m.on_convergence(Path("/tmp/f3.png"))
        assert result.action == "ask_help"
        assert result.ask_help_header == "CONVERGENCE GAVE NO CORRECTIVE"

    @patch("rvln.ai.goal_adherence_monitor.query_vlm")
    @patch("rvln.ai.goal_adherence_monitor.build_frame_grid")
    @patch("rvln.ai.goal_adherence_monitor.sample_frames_every_n")
    def test_unparseable_response_asks_help(self, mock_sample, mock_grid, mock_vlm):
        """If the convergence VLM emits unparseable JSON twice in a row, the
        old behavior was to silently stop the subgoal; now it must surface as
        ask_help so the operator can recover the run."""
        m = self._setup_monitor()
        mock_sample.return_value = m._frame_paths[-4:]
        mock_grid.return_value = MagicMock()
        mock_vlm.return_value = "not valid JSON, just prose"

        result = m.on_convergence(Path("/tmp/f3.png"))
        assert result.action == "ask_help"
        assert result.ask_help_header == "CONVERGENCE PARSE FAILURE"

    @patch("rvln.ai.goal_adherence_monitor.query_vlm")
    @patch("rvln.ai.goal_adherence_monitor.build_frame_grid")
    @patch("rvln.ai.goal_adherence_monitor.sample_frames_every_n")
    def test_diagnosis_complete_with_corrective_applies_corrective(
        self, mock_sample, mock_grid, mock_vlm,
    ):
        """Regression: the VLM can return contradictory output where
        complete=False but diagnosis='complete' alongside a real
        corrective_instruction. The corrective MUST be applied; previously the
        diagnosis short-circuited convergence into a premature stop and the
        operator would see 'task finished' even though a corrective had been
        proposed."""
        m = self._setup_monitor()
        mock_sample.return_value = m._frame_paths[-4:]
        mock_grid.return_value = MagicMock()
        mock_vlm.return_value = json.dumps({
            "complete": False,
            "completion_percentage": 0.85,
            "diagnosis": "complete",
            "corrective_instruction": "turn left 15 degrees toward the tree",
        })

        result = m.on_convergence(Path("/tmp/f3.png"))
        assert result.action == "command"
        assert "tree" in result.new_instruction
        assert m._corrections_used == 1


# -------------------------------------------------------------------------
# on_frame checkpoint triggering
# -------------------------------------------------------------------------

class TestOnFrameCheckpointTrigger:
    @patch("rvln.ai.goal_adherence_monitor.query_vlm")
    @patch("rvln.ai.goal_adherence_monitor.build_frame_grid")
    @patch("rvln.ai.goal_adherence_monitor.sample_frames_every_n")
    def test_non_checkpoint_returns_continue(self, mock_sample, mock_grid, mock_vlm):
        """Steps before the check_interval should return continue without VLM calls."""
        m = _make_monitor(check_interval=5)
        for i in range(4):
            result = m.on_frame(Path(f"/tmp/frame_{i}.png"))
            assert result.action == "continue"
            assert result.diary_entry == ""
        assert mock_vlm.call_count == 0

    @patch("rvln.ai.goal_adherence_monitor.query_vlm")
    @patch("rvln.ai.goal_adherence_monitor.build_frame_grid")
    @patch("rvln.ai.goal_adherence_monitor.sample_frames_every_n")
    def test_checkpoint_fires_at_interval(self, mock_sample, mock_grid, mock_vlm):
        """At exactly check_interval steps, a checkpoint should fire (VLM called)."""
        m = _make_monitor(check_interval=4)
        mock_grid.return_value = MagicMock()
        mock_sample.return_value = [Path(f"/tmp/frame_{i}.png") for i in range(4)]
        mock_vlm.side_effect = [
            "Drone moved toward tree.",
            '{"complete": false, "completion_percentage": 0.3, "on_track": true, "should_stop": false}',
        ]

        for i in range(3):
            result = m.on_frame(Path(f"/tmp/frame_{i}.png"))
            assert result.action == "continue"
            assert result.diary_entry == ""

        result = m.on_frame(Path("/tmp/frame_3.png"))
        # Checkpoint should have fired: VLM was called (local + global = 2 calls)
        assert mock_vlm.call_count == 2
        assert result.diary_entry != ""

    def test_step_counter_increments(self):
        m = _make_monitor(check_interval=100)
        for i in range(5):
            m.on_frame(Path(f"/tmp/frame_{i}.png"))
        assert m._step == 5

    def test_displacement_stored(self):
        m = _make_monitor(check_interval=100)
        m.on_frame(Path("/tmp/frame_0.png"), displacement=[10.0, 20.0, 30.0, 45.0])
        assert m._last_displacement == [10.0, 20.0, 30.0, 45.0]

    def test_frame_paths_accumulated(self):
        m = _make_monitor(check_interval=100)
        for i in range(3):
            m.on_frame(Path(f"/tmp/frame_{i}.png"))
        assert len(m._frame_paths) == 3
        assert m._frame_paths[0] == Path("/tmp/frame_0.png")
        assert m._frame_paths[2] == Path("/tmp/frame_2.png")
