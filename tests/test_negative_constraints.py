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


@patch("rvln.ai.diary_monitor.query_vlm")
@patch("rvln.ai.diary_monitor.build_frame_grid")
@patch("rvln.ai.diary_monitor.sample_frames_every_n")
def test_no_violation_field_without_constraints(mock_sample, mock_grid, mock_vlm):
    """Without constraints, missing constraint_violated field is fine."""
    m = _make_monitor()
    m._frame_paths = [Path(f"/tmp/f{i}.png") for i in range(4)]
    m._frame_timestamps = [float(i) for i in range(4)]
    m._step = 4

    mock_grid.return_value = MagicMock()
    mock_sample.return_value = m._frame_paths[-4:]
    mock_vlm.side_effect = [
        "Drone moved forward.",
        '{"complete": false, "completion_percentage": 0.5, "on_track": true, "should_stop": false}',
    ]

    result = m._run_checkpoint()
    assert result.action == "continue"


# ---------------------------------------------------------------------------
# End-to-end integration tests (planner -> monitor, no simulator)
# ---------------------------------------------------------------------------

import pytest


def test_end_to_end_constraint_flow():
    """Full flow: planner classifies + extracts constraints, monitor receives them."""
    spot = pytest.importorskip("spot", reason="spot not available")
    from rvln.ai.ltl_planner import LTLSymbolicPlanner

    class MockLLM:
        def __init__(self):
            self.ltl_nl_formula = {
                "pi_predicates": {
                    "pi_1": "Go to tree A",
                    "pi_2": "Go to streetlight B",
                    "pi_3": "Flying over building C",
                },
                "ltl_nl_formula": "F pi_2 & (!pi_2 U pi_1) & G(!pi_3)",
            }
        def make_natural_language_request(self, request):
            return ""

    planner = LTLSymbolicPlanner(MockLLM())
    planner.plan_from_natural_language("Go to A then B, never fly over C")

    assert "pi_3" in planner.constraint_predicates
    assert "pi_1" not in planner.constraint_predicates

    current = planner.get_next_predicate()
    assert current == "Go to tree A"
    constraints = planner.get_active_constraints()
    assert constraints == ["Flying over building C"]

    monitor = LiveDiaryMonitor(
        subgoal=current,
        check_interval=2,
        model="gpt-4o",
        negative_constraints=constraints,
    )
    assert monitor._negative_constraints == ["Flying over building C"]
    block = monitor._constraints_block()
    assert "Flying over building C" in block
    assert "Active constraints" in block
    monitor.cleanup()


def test_end_to_end_backward_compat():
    """Existing formulas without constraints work unchanged."""
    spot = pytest.importorskip("spot", reason="spot not available")
    from rvln.ai.ltl_planner import LTLSymbolicPlanner

    class MockLLM:
        def __init__(self):
            self.ltl_nl_formula = {
                "pi_predicates": {
                    "pi_1": "Go to A",
                    "pi_2": "Go to B",
                },
                "ltl_nl_formula": "F pi_2 & (!pi_2 U pi_1)",
            }
        def make_natural_language_request(self, request):
            return ""

    planner = LTLSymbolicPlanner(MockLLM())
    planner.plan_from_natural_language("A then B")

    assert planner.constraint_predicates == {}
    current = planner.get_next_predicate()
    assert planner.get_active_constraints() == []

    monitor = LiveDiaryMonitor(
        subgoal=current,
        check_interval=2,
        model="gpt-4o",
    )
    assert monitor._negative_constraints == []
    assert monitor._constraints_block() == ""
    monitor.cleanup()


def test_scoped_constraint_release():
    """Scoped constraint (!pi_3 U pi_2) releases after pi_2 is achieved."""
    spot = pytest.importorskip("spot", reason="spot not available")
    from rvln.ai.ltl_planner import LTLSymbolicPlanner

    class MockLLM:
        def __init__(self):
            self.ltl_nl_formula = {
                "pi_predicates": {
                    "pi_1": "Approach tree",
                    "pi_2": "Go to streetlight",
                    "pi_3": "Near the red car",
                },
                "ltl_nl_formula": "F pi_2 & (!pi_2 U pi_1) & (!pi_3 U pi_2)",
            }
        def make_natural_language_request(self, request):
            return ""

    planner = LTLSymbolicPlanner(MockLLM())
    planner.plan_from_natural_language("tree then streetlight, avoid car")

    assert "pi_3" in planner.constraint_predicates

    current = planner.get_next_predicate()
    c1 = planner.get_active_constraints()
    assert "Near the red car" in c1

    planner.advance_state(current)
    current = planner.get_next_predicate()
    c2 = planner.get_active_constraints()
    assert "Near the red car" in c2

    planner.advance_state(current)
    c3 = planner.get_active_constraints()
    assert c3 == []
