#!/usr/bin/env python3
"""
Robustness tests for LTLSymbolicPlanner using Spot (no LLM).
Run with conda env rvln-sim:  conda run -n rvln-sim python test_ltl_planner_robustness.py
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest

spot = pytest.importorskip("spot", reason="spot (conda-forge) not available outside rvln-sim")

from rvln.ai.ltl_planner import (
    LTLSymbolicPlanner,
    _predicate_key_to_index,
    _normalize_pi_predicates,
)


class MockLLM:
    """Minimal LLM interface for injecting LTL data without real API calls."""

    def __init__(self):
        self.ltl_nl_formula = None

    def make_natural_language_request(self, request: str) -> str:
        return ""


def test_helpers():
    """Test _predicate_key_to_index and _normalize_pi_predicates."""
    assert _predicate_key_to_index("pi_1") == 1
    assert _predicate_key_to_index("pi_10") == 10
    assert _predicate_key_to_index("p1") == 1
    assert _predicate_key_to_index("  pi_2  ") == 2
    for bad in ("", "x1", "predicate_1", "pi_", "p"):
        try:
            _predicate_key_to_index(bad)
            assert False, f"Expected ValueError for '{bad}'"
        except ValueError:
            pass

    assert _normalize_pi_predicates({}) == {}
    assert _normalize_pi_predicates(None) == {}
    assert _normalize_pi_predicates("not a dict") == {}
    d = _normalize_pi_predicates({"pi_1": "Task A", "pi_2": "Task B"})
    assert d == {"pi_1": "Task A", "pi_2": "Task B"}
    d2 = _normalize_pi_predicates({"p1": "Go", "pi_2": "Stop"})
    assert list(d2.keys()) == ["pi_1", "pi_2"] and d2["pi_1"] == "Go" and d2["pi_2"] == "Stop"
    # Non-string value skipped
    d3 = _normalize_pi_predicates({"pi_1": "A", "pi_2": 42})
    assert d3 == {"pi_1": "A"}


def test_plan_valid_simple():
    """Simple instruction: F pi_1."""
    mock = MockLLM()
    mock.ltl_nl_formula = {
        "pi_predicates": {"pi_1": "Deliver Coke to Location A"},
        "ltl_nl_formula": "F pi_1",
    }
    planner = LTLSymbolicPlanner(mock)
    planner.plan_from_natural_language("Deliver Coke to Location A")
    assert planner.pi_map == {"pi_1": "Deliver Coke to Location A"}
    assert planner.automaton is not None
    next_pred = planner.get_next_predicate()
    assert next_pred == "Deliver Coke to Location A"
    planner.advance_state("Deliver Coke to Location A")
    next_pred = planner.get_next_predicate()
    assert next_pred is None
    assert planner.finished


def test_plan_valid_sequence():
    """Sequence: F pi_3 & (!pi_3 U pi_2) & (!pi_2 U pi_1)."""
    mock = MockLLM()
    mock.ltl_nl_formula = {
        "pi_predicates": {
            "pi_1": "Go to Location A",
            "pi_2": "Go to Location B",
            "pi_3": "Go to Location C",
        },
        "ltl_nl_formula": "F pi_3 & (!pi_3 U pi_2) & (!pi_2 U pi_1)",
    }
    planner = LTLSymbolicPlanner(mock)
    planner.plan_from_natural_language("Go to A then B then C")
    assert len(planner.pi_map) == 3
    order = []
    while True:
        p = planner.get_next_predicate()
        if p is None:
            break
        order.append(p)
        planner.advance_state(p)
    assert order == [
        "Go to Location A",
        "Go to Location B",
        "Go to Location C",
    ]


def test_plan_valid_complex():
    """Complex: F pi_1 & (!pi_1 U (pi_2 | pi_3))."""
    mock = MockLLM()
    mock.ltl_nl_formula = {
        "pi_predicates": {
            "pi_1": "Deliver pen to Location D",
            "pi_2": "Deliver drink to Location E",
            "pi_3": "Deliver apple to Location E",
        },
        "ltl_nl_formula": "F pi_1 & (!pi_1 U (pi_2 | pi_3))",
    }
    planner = LTLSymbolicPlanner(mock)
    planner.plan_from_natural_language("Pen to D after drink or apple to E")
    assert len(planner.pi_map) == 3
    first = planner.get_next_predicate()
    assert first in ("Deliver drink to Location E", "Deliver apple to Location E")


def test_plan_invalid_instruction():
    """Reject empty or non-string instruction."""
    mock = MockLLM()
    mock.ltl_nl_formula = {"pi_predicates": {"pi_1": "X"}, "ltl_nl_formula": "F pi_1"}
    planner = LTLSymbolicPlanner(mock)
    for bad in ("", "   ", None, 123):
        try:
            planner.plan_from_natural_language(bad)
            assert False, f"Expected ValueError for instruction={bad!r}"
        except (ValueError, TypeError):
            pass


def test_plan_invalid_data():
    """Reject missing/empty LLM data."""
    mock = MockLLM()
    planner = LTLSymbolicPlanner(mock)
    # Empty data
    mock.ltl_nl_formula = None
    try:
        planner.plan_from_natural_language("do something")
        assert False
    except ValueError as e:
        assert "valid LTL formula" in str(e) or "empty" in str(e).lower()
    mock.ltl_nl_formula = {}
    try:
        planner.plan_from_natural_language("do something")
        assert False
    except ValueError as e:
        assert "valid" in str(e).lower() or "ltl_nl_formula" in str(e) or "pi_predicates" in str(e)
    # Missing keys
    mock.ltl_nl_formula = {"pi_predicates": {"pi_1": "X"}}
    try:
        planner.plan_from_natural_language("do something")
        assert False
    except ValueError:
        pass
    mock.ltl_nl_formula = {"ltl_nl_formula": "F pi_1"}
    try:
        planner.plan_from_natural_language("do something")
        assert False
    except ValueError:
        pass
    # Empty formula
    mock.ltl_nl_formula = {"pi_predicates": {"pi_1": "X"}, "ltl_nl_formula": ""}
    try:
        planner.plan_from_natural_language("do something")
        assert False
    except ValueError as e:
        assert "ltl_nl_formula" in str(e) or "non-empty" in str(e)
    # Empty pi_predicates (after normalize)
    mock.ltl_nl_formula = {"pi_predicates": {}, "ltl_nl_formula": "F pi_1"}
    try:
        planner.plan_from_natural_language("do something")
        assert False
    except ValueError as e:
        assert "predicate" in str(e).lower() or "valid" in str(e).lower()
    # Duplicate descriptions are allowed (tasks unique by pi_1, pi_2, ...)
    mock.ltl_nl_formula = {
        "pi_predicates": {"pi_1": "Same", "pi_2": "Same"},
        "ltl_nl_formula": "F pi_1 & F pi_2 & (!pi_2 U pi_1)",
    }
    planner.plan_from_natural_language("do something")
    first = planner.get_next_predicate()
    assert first == "Same"
    planner.advance_state("Same")
    second = planner.get_next_predicate()
    assert second == "Same"


def test_plan_invalid_formula():
    """Reject formula Spot cannot translate."""
    mock = MockLLM()
    mock.ltl_nl_formula = {
        "pi_predicates": {"pi_1": "X"},
        "ltl_nl_formula": "G pi_1",  # G (always) may not be supported for monitor
    }
    planner = LTLSymbolicPlanner(mock)
    try:
        planner.plan_from_natural_language("always X")
        # Spot may still accept G; if so, try a clearly invalid formula
        mock.ltl_nl_formula["ltl_nl_formula"] = "((("
        planner.plan_from_natural_language("bad")
        assert False
    except ValueError as e:
        assert "Spot" in str(e) or "formula" in str(e).lower() or "translate" in str(e).lower()


def test_get_next_without_plan():
    """get_next_predicate returns None when not planned."""
    mock = MockLLM()
    planner = LTLSymbolicPlanner(mock)
    assert planner.get_next_predicate() is None


def test_advance_unknown_task():
    """advance_state with unknown task does not crash."""
    mock = MockLLM()
    mock.ltl_nl_formula = {
        "pi_predicates": {"pi_1": "Task A"},
        "ltl_nl_formula": "F pi_1",
    }
    planner = LTLSymbolicPlanner(mock)
    planner.plan_from_natural_language("A")
    before = planner.current_automaton_state
    planner.advance_state("Unknown task description")
    assert planner.current_automaton_state == before


def test_advance_no_edge_sink():
    """When no edge matches, transition to sink if present."""
    mock = MockLLM()
    mock.ltl_nl_formula = {
        "pi_predicates": {"pi_1": "Only task"},
        "ltl_nl_formula": "F pi_1",
    }
    planner = LTLSymbolicPlanner(mock)
    planner.plan_from_natural_language("Only task")
    _ = planner.get_next_predicate()  # set _last_returned_predicate_key so advance_state knows which task
    planner.advance_state("Only task")
    # Should be in sink or finished
    assert planner.get_next_predicate() is None
    assert planner.finished


def test_normalize_key_variants():
    """Formula and predicates with p1 vs pi_1 both work after normalize."""
    mock = MockLLM()
    mock.ltl_nl_formula = {
        "pi_predicates": {"p1": "First", "p2": "Second"},
        "ltl_nl_formula": "F p1 & (!p2 U p1)",
    }
    planner = LTLSymbolicPlanner(mock)
    planner.plan_from_natural_language("First then Second")
    assert planner.pi_map.get("pi_1") == "First"
    assert planner.pi_map.get("pi_2") == "Second"
    next_p = planner.get_next_predicate()
    assert next_p == "First"


# -----------------------------------------------------------------------
# Complex automaton traversal
# -----------------------------------------------------------------------

def _run_full(planner):
    """Run automaton to completion, returning ordered subgoals."""
    order = []
    while True:
        p = planner.get_next_predicate()
        if p is None:
            break
        order.append(p)
        planner.advance_state(p)
    return order


def _step(planner, n):
    """Advance the planner through n steps."""
    for _ in range(n):
        p = planner.get_next_predicate()
        assert p is not None
        planner.advance_state(p)


class TestLongSequence:
    """Five-step ordered sequence."""

    def _make(self):
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {
                "pi_1": "Go to A",
                "pi_2": "Go to B",
                "pi_3": "Go to C",
                "pi_4": "Go to D",
                "pi_5": "Go to E",
            },
            "ltl_nl_formula": (
                "F pi_5 & (!pi_5 U pi_4) & (!pi_4 U pi_3) "
                "& (!pi_3 U pi_2) & (!pi_2 U pi_1)"
            ),
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language("A then B then C then D then E")
        return planner

    def test_correct_ordering(self):
        order = _run_full(self._make())
        assert order == ["Go to A", "Go to B", "Go to C", "Go to D", "Go to E"]

    def test_finished_after_completion(self):
        planner = self._make()
        _run_full(planner)
        assert planner.finished
        assert planner.get_next_predicate() is None

    def test_step_by_step_state_changes(self):
        planner = self._make()
        states = [planner.current_automaton_state]
        for _ in range(5):
            p = planner.get_next_predicate()
            assert p is not None
            planner.advance_state(p)
            states.append(planner.current_automaton_state)
        assert len(set(states)) == 6, f"Expected 6 distinct states, got {states}"


class TestEdgeCases:
    """Edge cases for automaton advancement."""

    def test_double_advance_same_task(self):
        """Advancing the same task twice doesn't crash or skip states."""
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {"pi_1": "Go to A", "pi_2": "Go to B"},
            "ltl_nl_formula": "F pi_2 & (!pi_2 U pi_1)",
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language("A then B")

        p = planner.get_next_predicate()
        planner.advance_state(p)
        state_after_first = planner.current_automaton_state
        planner.advance_state(p)  # double advance
        # Should either stay or go to sink, not crash
        assert planner.current_automaton_state >= 0

    def test_advance_after_finished(self):
        """Advancing after mission complete is a no-op."""
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {"pi_1": "Go to A"},
            "ltl_nl_formula": "F pi_1",
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language("go to A")

        _run_full(planner)
        assert planner.finished
        state = planner.current_automaton_state
        planner.advance_state("Go to A")  # after finished
        assert planner.finished

    def test_get_next_returns_none_consistently_after_done(self):
        """Repeated get_next_predicate() after finished always returns None."""
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {"pi_1": "Go to A", "pi_2": "Go to B"},
            "ltl_nl_formula": "F pi_2 & (!pi_2 U pi_1)",
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language("A then B")
        _run_full(planner)
        for _ in range(5):
            assert planner.get_next_predicate() is None


# ------------------------------------------------------------------
# Regression: last goal dropped when Spot produces fewer states
# ------------------------------------------------------------------

def _make_smaller_automaton_planner():
    """Build a planner whose automaton has fewer states.

    spot.postprocess(aut, "monitor", "small") merges states, producing
    the same structure that older/different Spot versions create natively.
    The result is a real Spot automaton where the last goal's only path
    goes through the sink edge.
    """
    mock = MockLLM()
    mock.ltl_nl_formula = {
        "pi_predicates": {
            "pi_1": "Go to A",
            "pi_2": "Go to B",
            "pi_3": "Go to C",
            "pi_4": "Go to D",
            "pi_5": "Go to E",
            "pi_6": "Go to F",
        },
        "ltl_nl_formula": (
            "F pi_6 & (!pi_6 U pi_5) & (!pi_5 U pi_4) "
            "& (!pi_4 U pi_3) & (!pi_3 U pi_2) & (!pi_2 U pi_1)"
        ),
    }
    planner = LTLSymbolicPlanner(mock)
    planner.plan_from_natural_language("A B C D E F")

    smaller = spot.postprocess(planner.automaton, "monitor", "small")
    planner.automaton = smaller
    planner._sink_state = None
    planner._add_sink_state()
    planner.current_automaton_state = planner.automaton.get_init_state_number()
    planner.finished = False
    planner._last_returned_predicate_key = None
    return planner


class TestLastGoalViaSinkEdge:
    """Regression: get_next_predicate must return all goals even when the
    Spot automaton has fewer states and the last goal's only outgoing edge
    leads to the sink.

    Uses spot.postprocess to produce a real smaller automaton (the same
    structure that different Spot versions create natively).
    """

    def test_normal_automaton_returns_all_six(self):
        """Baseline: the default automaton returns all 6 goals."""
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {
                "pi_1": "Go to A", "pi_2": "Go to B", "pi_3": "Go to C",
                "pi_4": "Go to D", "pi_5": "Go to E", "pi_6": "Go to F",
            },
            "ltl_nl_formula": (
                "F pi_6 & (!pi_6 U pi_5) & (!pi_5 U pi_4) "
                "& (!pi_4 U pi_3) & (!pi_3 U pi_2) & (!pi_2 U pi_1)"
            ),
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language("A B C D E F")
        order = _run_full(planner)
        assert order == [
            "Go to A", "Go to B", "Go to C",
            "Go to D", "Go to E", "Go to F",
        ]
        assert planner.finished

    def test_smaller_automaton_returns_all_six(self):
        """Bug: postprocessed (smaller) automaton drops the last goal."""
        planner = _make_smaller_automaton_planner()
        order = _run_full(planner)
        assert len(order) == 6, (
            f"Expected 6 goals but got {len(order)}: {order}"
        )
        assert order[-1] == "Go to F", (
            f"Last goal must be 'Go to F', got {order[-1]!r}"
        )
        assert planner.finished

    def test_no_double_return_after_last_goal(self):
        """After the last goal is returned and advanced, None must follow."""
        planner = _make_smaller_automaton_planner()
        order = _run_full(planner)
        assert len(order) == 6
        assert planner.get_next_predicate() is None
        assert planner.finished


# ------------------------------------------------------------------
# Regression: sink-edge fallback for the last goal
# ------------------------------------------------------------------

class TestSinkEdgeFallback:
    """Verify the sink-edge fallback in get_next_predicate returns the
    correct last predicate when the automaton's only outgoing edge from the
    current state leads to the added sink state.

    The test creates a standard three-step sequence, postprocesses the
    automaton to merge states, and walks through with get_next_predicate /
    advance_state to confirm all goals (including the last one) are returned.
    """

    def _make(self):
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {
                "pi_1": "Fly to landmark A",
                "pi_2": "Fly to landmark B",
                "pi_3": "Fly to landmark C",
            },
            "ltl_nl_formula": (
                "F pi_3 & (!pi_3 U pi_2) & (!pi_2 U pi_1)"
            ),
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language("A then B then C")

        smaller = spot.postprocess(planner.automaton, "monitor", "small")
        planner.automaton = smaller
        planner._sink_state = None
        planner._add_sink_state()
        planner.current_automaton_state = planner.automaton.get_init_state_number()
        planner.finished = False
        planner._last_returned_predicate_key = None
        return planner

    def test_all_three_goals_returned(self):
        """All three goals are returned in order, including via sink-edge."""
        planner = self._make()
        order = _run_full(planner)
        assert order == [
            "Fly to landmark A",
            "Fly to landmark B",
            "Fly to landmark C",
        ], f"Expected 3 goals in order, got {order}"
        assert planner.finished

    def test_last_goal_via_sink_edge(self):
        """Walk step-by-step and verify the last goal is returned."""
        planner = self._make()
        p1 = planner.get_next_predicate()
        assert p1 == "Fly to landmark A"
        planner.advance_state(p1)

        p2 = planner.get_next_predicate()
        assert p2 == "Fly to landmark B"
        planner.advance_state(p2)

        p3 = planner.get_next_predicate()
        assert p3 == "Fly to landmark C", (
            f"Last goal should be 'Fly to landmark C', got {p3!r}"
        )
        planner.advance_state(p3)

        assert planner.get_next_predicate() is None
        assert planner.finished

    def test_none_after_completion(self):
        """After completing all goals, repeated calls return None."""
        planner = self._make()
        _run_full(planner)
        for _ in range(5):
            assert planner.get_next_predicate() is None
        assert planner.finished


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
