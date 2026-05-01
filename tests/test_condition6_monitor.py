"""
Unit tests for TextOnlyGoalAdherenceMonitor (Condition 6).

The TextOnlyGoalAdherenceMonitor is defined inline in scripts/run_condition6_text_only.py.
It uses VLM for local (2-frame) diary entries but text-only LLM for global
assessment and convergence checks (no image grid).

No API calls: all LLM/VLM responses are mocked.
Run: conda run -n rvln-sim pytest tests/test_condition6_monitor.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if _SCRIPTS.is_dir() and str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_condition6_text_only import TextOnlyGoalAdherenceMonitor
from rvln.ai.ltl_planner import ConstraintInfo


def _make_monitor(constraints=None, stall_window=3, stall_threshold=0.05):
    with patch("rvln.ai.utils.llm_providers.LLMFactory") as mock_factory:
        mock_llm = MagicMock()
        mock_factory.create.return_value = mock_llm
        m = TextOnlyGoalAdherenceMonitor(
            subgoal="Approach the tree",
            check_interval=2,
            vlm_model="gpt-4o",
            llm_model="gpt-4o",
            constraints=constraints or [],
            stall_window=stall_window,
            stall_threshold=stall_threshold,
        )
    return m


class TestConstraintsBlock:
    def test_empty_constraints(self):
        m = _make_monitor()
        assert m._constraints_block() == ""

    def test_string_constraints(self):
        m = _make_monitor(["stay away from building B"])
        block = m._constraints_block()
        assert "stay away from building B" in block
        assert "Active constraints" in block

    def test_constraint_info_negative(self):
        m = _make_monitor([
            ConstraintInfo(description="Flying over building C", polarity="negative"),
        ])
        block = m._constraints_block()
        assert "AVOID: Flying over building C" in block

    def test_constraint_info_positive(self):
        m = _make_monitor([
            ConstraintInfo(description="Above 10 meters altitude", polarity="positive"),
        ])
        block = m._constraints_block()
        assert "MAINTAIN: Above 10 meters altitude" in block

    def test_mixed_constraints(self):
        m = _make_monitor([
            ConstraintInfo(description="Flying over highway", polarity="negative"),
            ConstraintInfo(description="Above treeline", polarity="positive"),
        ])
        block = m._constraints_block()
        assert "AVOID: Flying over highway" in block
        assert "MAINTAIN: Above treeline" in block


class TestStallDetection:
    def test_no_stall_insufficient_history(self):
        m = _make_monitor()
        m._completion_history = [0.1, 0.12]
        assert m._is_stalled() is False

    def test_stall_detected_when_flat(self):
        m = _make_monitor()
        m._completion_history = [0.30, 0.31, 0.32]
        assert m._is_stalled() is True

    def test_no_stall_when_progressing(self):
        m = _make_monitor()
        m._completion_history = [0.30, 0.40, 0.50]
        assert m._is_stalled() is False

    def test_no_stall_when_completion_high(self):
        m = _make_monitor()
        m._completion_history = [0.85, 0.86, 0.86]
        assert m._is_stalled() is False


class TestJsonParsing:
    def test_plain_json(self):
        result = TextOnlyGoalAdherenceMonitor._parse_json_response(
            '{"complete": false, "completion_percentage": 0.5}'
        )
        assert result is not None
        assert result["complete"] is False

    def test_json_in_markdown_fence(self):
        result = TextOnlyGoalAdherenceMonitor._parse_json_response(
            '```json\n{"complete": true}\n```'
        )
        assert result is not None
        assert result["complete"] is True

    def test_json_with_surrounding_text(self):
        result = TextOnlyGoalAdherenceMonitor._parse_json_response(
            'Here is the result: {"done": true} end'
        )
        assert result is not None
        assert result["done"] is True

    def test_garbage_returns_none(self):
        result = TextOnlyGoalAdherenceMonitor._parse_json_response("no json here")
        assert result is None


class TestDisplacementFormatting:
    def test_format(self):
        m = _make_monitor()
        m._last_displacement = [350.0, 120.0, 0.0, 15.0]
        formatted = m._format_displacement()
        assert "3.50" in formatted
        assert "1.20" in formatted
        assert "15.0" in formatted


class TestOnConvergence:
    def test_complete_returns_stop(self):
        m = _make_monitor()
        m._diary = ["Steps 0-2: moved forward"]
        m._last_displacement = [100.0, 0.0, 0.0, 0.0]

        m._llm = MagicMock()
        m._llm.make_request.return_value = json.dumps({
            "complete": True,
            "completion_percentage": 0.95,
            "diagnosis": "complete",
            "corrective_instruction": None,
            "constraint_violated": False,
        })

        result = m.on_convergence(Path("/tmp/fake.png"), displacement=[100.0, 0.0, 0.0, 0.0])
        assert result.action == "stop"

    def test_stopped_short_returns_command(self):
        m = _make_monitor()
        m._diary = ["Steps 0-2: moved forward slightly"]
        m._last_displacement = [50.0, 0.0, 0.0, 0.0]

        m._llm = MagicMock()
        m._llm.make_request.return_value = json.dumps({
            "complete": False,
            "completion_percentage": 0.4,
            "diagnosis": "stopped_short",
            "corrective_instruction": "keep moving forward toward the tree",
            "constraint_violated": False,
        })

        result = m.on_convergence(Path("/tmp/fake.png"), displacement=[50.0, 0.0, 0.0, 0.0])
        assert result.action == "command"
        assert "forward" in result.new_instruction

    def test_max_corrections_returns_ask_help(self):
        m = _make_monitor()
        m._corrections_used = 15
        m._max_corrections = 15

        result = m.on_convergence(Path("/tmp/fake.png"))
        assert result.action == "ask_help"
        assert "Max corrections" in result.reasoning
