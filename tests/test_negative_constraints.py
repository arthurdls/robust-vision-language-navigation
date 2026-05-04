"""
Tier 1 tests for constraint support in GoalAdherenceMonitor.
No API calls, no GPU, no simulator. All LLM responses are mocked.
Run: conda run -n rvln-sim pytest tests/test_negative_constraints.py -v
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.ai.goal_adherence_monitor import GoalAdherenceMonitor, DiaryCheckResult
from rvln.ai.ltl_planner import ConstraintInfo


def _make_monitor(constraints=None):
    return GoalAdherenceMonitor(
        subgoal="Approach the tree",
        check_interval=2,
        model="gpt-4o",
        constraints=constraints or [],
    )


def test_monitor_empty_or_default_constraints():
    m = _make_monitor()
    assert m._constraints == []
    m2 = GoalAdherenceMonitor(subgoal="Go forward", check_interval=2, model="gpt-4o")
    assert m2._constraints == []


def test_monitor_accepts_constraints():
    m = _make_monitor(["stay away from building B", "do not fly over zone C"])
    assert len(m._constraints) == 2
    assert "building B" in m._constraints[0]


def test_constraints_block_empty_when_none():
    m = _make_monitor()
    assert m._constraints_block() == ""


def test_constraints_block_renders_correctly():
    m = _make_monitor(["stay away from building B", "do not fly over zone C"])
    block = m._constraints_block()
    assert "Active constraints" in block
    assert "stay away from building B" in block
    assert "do not fly over zone C" in block


def test_constraints_block_with_constraint_info():
    """ConstraintInfo objects render with AVOID/MAINTAIN labels."""
    m = _make_monitor([
        ConstraintInfo(description="Flying over building C", polarity="negative"),
        ConstraintInfo(description="Above 10 meters altitude", polarity="positive"),
    ])
    block = m._constraints_block()
    assert "AVOID: Flying over building C" in block
    assert "MAINTAIN: Above 10 meters altitude" in block


@patch("rvln.ai.goal_adherence_monitor.query_vlm")
@patch("rvln.ai.goal_adherence_monitor.build_frame_grid")
@patch("rvln.ai.goal_adherence_monitor.sample_frames_every_n")
def test_constraint_violation_force_converges(mock_sample, mock_grid, mock_vlm):
    m = _make_monitor([
        ConstraintInfo(description="stay away from building B", polarity="negative"),
    ])
    m._frame_paths = [Path(f"/tmp/f{i}.png") for i in range(4)]
    m._frame_timestamps = [float(i) for i in range(4)]
    m._step = 4

    mock_grid.return_value = MagicMock()
    mock_sample.return_value = m._frame_paths[-4:]
    mock_vlm.side_effect = [
        "Drone moved closer to building B.",
        '{"complete": false, "completion_percentage": 0.3, "should_stop": true}',
    ]

    result = m._run_checkpoint()
    assert result.action == "force_converge"


@patch("rvln.ai.goal_adherence_monitor.query_vlm")
@patch("rvln.ai.goal_adherence_monitor.build_frame_grid")
@patch("rvln.ai.goal_adherence_monitor.sample_frames_every_n")
def test_no_violation_continues(mock_sample, mock_grid, mock_vlm):
    m = _make_monitor([
        ConstraintInfo(description="stay away from building B", polarity="negative"),
    ])
    m._frame_paths = [Path(f"/tmp/f{i}.png") for i in range(4)]
    m._frame_timestamps = [float(i) for i in range(4)]
    m._step = 4

    mock_grid.return_value = MagicMock()
    mock_sample.return_value = m._frame_paths[-4:]
    mock_vlm.side_effect = [
        "Drone moved toward the tree, building B not visible.",
        '{"complete": false, "completion_percentage": 0.4, "should_stop": false}',
    ]

    result = m._run_checkpoint()
    assert result.action == "continue"


@patch("rvln.ai.goal_adherence_monitor.query_vlm")
@patch("rvln.ai.goal_adherence_monitor.build_frame_grid")
@patch("rvln.ai.goal_adherence_monitor.sample_frames_every_n")
def test_no_constraint_field_without_constraints(mock_sample, mock_grid, mock_vlm):
    """Without constraints, should_stop=false means continue."""
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


@patch("rvln.ai.goal_adherence_monitor.query_vlm")
@patch("rvln.ai.goal_adherence_monitor.build_frame_grid")
@patch("rvln.ai.goal_adherence_monitor.sample_frames_every_n")
def test_positive_constraint_violation_force_converges(mock_sample, mock_grid, mock_vlm):
    """Positive constraint violation also triggers force_converge."""
    m = _make_monitor([
        ConstraintInfo(description="Above 10 meters altitude", polarity="positive"),
    ])
    m._frame_paths = [Path(f"/tmp/f{i}.png") for i in range(4)]
    m._frame_timestamps = [float(i) for i in range(4)]
    m._step = 4

    mock_grid.return_value = MagicMock()
    mock_sample.return_value = m._frame_paths[-4:]
    mock_vlm.side_effect = [
        "Drone descended below 10 meters.",
        '{"complete": false, "completion_percentage": 0.3, "should_stop": true}',
    ]

    result = m._run_checkpoint()
    assert result.action == "force_converge"


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
    assert planner.constraint_predicates["pi_3"].polarity == "negative"
    assert "pi_1" not in planner.constraint_predicates

    current = planner.get_next_predicate()
    assert current == "Go to tree A"
    constraints = planner.get_active_constraints()
    assert len(constraints) == 1
    assert constraints[0].description == "Flying over building C"

    monitor = GoalAdherenceMonitor(
        subgoal=current,
        check_interval=2,
        model="gpt-4o",
        constraints=constraints,
    )
    assert len(monitor._constraints) == 1
    block = monitor._constraints_block()
    assert "AVOID: Flying over building C" in block
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

    monitor = GoalAdherenceMonitor(
        subgoal=current,
        check_interval=2,
        model="gpt-4o",
    )
    assert monitor._constraints == []
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
    assert planner.constraint_predicates["pi_3"].polarity == "negative"

    current = planner.get_next_predicate()
    c1 = planner.get_active_constraints()
    descs1 = [c.description for c in c1]
    assert "Near the red car" in descs1

    planner.advance_state(current)
    current = planner.get_next_predicate()
    c2 = planner.get_active_constraints()
    descs2 = [c.description for c in c2]
    assert "Near the red car" in descs2

    planner.advance_state(current)
    c3 = planner.get_active_constraints()
    assert c3 == []


def test_end_to_end_positive_constraint_flow():
    """Full flow with positive constraint: planner -> monitor."""
    spot = pytest.importorskip("spot", reason="spot not available")
    from rvln.ai.ltl_planner import LTLSymbolicPlanner

    class MockLLM:
        def __init__(self):
            self.ltl_nl_formula = {
                "pi_predicates": {
                    "pi_1": "Fly to the landmark",
                    "pi_2": "Above 10 meters altitude",
                },
                "ltl_nl_formula": "F pi_1 & G(pi_2)",
            }
        def make_natural_language_request(self, request):
            return ""

    planner = LTLSymbolicPlanner(MockLLM())
    planner.plan_from_natural_language("Fly to landmark, always above 10m")

    assert "pi_2" in planner.constraint_predicates
    assert planner.constraint_predicates["pi_2"].polarity == "positive"

    current = planner.get_next_predicate()
    assert current == "Fly to the landmark"
    constraints = planner.get_active_constraints()
    assert len(constraints) == 1
    assert constraints[0].polarity == "positive"

    monitor = GoalAdherenceMonitor(
        subgoal=current,
        check_interval=2,
        model="gpt-4o",
        constraints=constraints,
    )
    block = monitor._constraints_block()
    assert "MAINTAIN: Above 10 meters altitude" in block
    monitor.cleanup()


def test_end_to_end_mixed_constraints():
    """Full flow with mixed positive + negative constraints."""
    spot = pytest.importorskip("spot", reason="spot not available")
    from rvln.ai.ltl_planner import LTLSymbolicPlanner

    class MockLLM:
        def __init__(self):
            self.ltl_nl_formula = {
                "pi_predicates": {
                    "pi_1": "Navigate to the bridge",
                    "pi_2": "Above the treeline",
                    "pi_3": "Flying over the highway",
                },
                "ltl_nl_formula": "F pi_1 & G(pi_2) & G(!pi_3)",
            }
        def make_natural_language_request(self, request):
            return ""

    planner = LTLSymbolicPlanner(MockLLM())
    planner.plan_from_natural_language("bridge, stay above treeline, never over highway")

    constraints = planner.get_active_constraints()
    assert len(constraints) == 2

    monitor = GoalAdherenceMonitor(
        subgoal=planner.get_next_predicate(),
        check_interval=2,
        model="gpt-4o",
        constraints=constraints,
    )
    block = monitor._constraints_block()
    assert "MAINTAIN: Above the treeline" in block
    assert "AVOID: Flying over the highway" in block
    monitor.cleanup()


def test_scoped_positive_constraint_release():
    """Scoped positive constraint (pi_3 U pi_1) releases after pi_1 achieved."""
    spot = pytest.importorskip("spot", reason="spot not available")
    from rvln.ai.ltl_planner import LTLSymbolicPlanner

    class MockLLM:
        def __init__(self):
            self.ltl_nl_formula = {
                "pi_predicates": {
                    "pi_1": "Go to the tree",
                    "pi_2": "Go to the streetlight",
                    "pi_3": "River visible in frame",
                },
                "ltl_nl_formula": "F pi_2 & (!pi_2 U pi_1) & (pi_3 U pi_1)",
            }
        def make_natural_language_request(self, request):
            return ""

    planner = LTLSymbolicPlanner(MockLLM())
    planner.plan_from_natural_language("tree then streetlight, keep river visible until tree")

    assert planner.constraint_predicates["pi_3"].polarity == "positive"

    current = planner.get_next_predicate()
    c1 = planner.get_active_constraints()
    assert any(c.description == "River visible in frame" for c in c1)

    planner.advance_state(current)
    current = planner.get_next_predicate()
    c2 = planner.get_active_constraints()
    assert not any(c.description == "River visible in frame" for c in c2)

    planner.advance_state(current)
    assert planner.get_active_constraints() == []
