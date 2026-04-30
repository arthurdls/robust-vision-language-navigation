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
    ConstraintInfo,
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


def test_classify_global_avoidance():
    """G(!pi_3) makes pi_3 a negative constraint, not a goal."""
    mock = MockLLM()
    mock.ltl_nl_formula = {
        "pi_predicates": {
            "pi_1": "Go to tree A",
            "pi_2": "Go to streetlight B",
            "pi_3": "Flying over building C",
        },
        "ltl_nl_formula": "F pi_2 & (!pi_2 U pi_1) & G(!pi_3)",
    }
    planner = LTLSymbolicPlanner(mock)
    planner.plan_from_natural_language("Go to A then B, never fly over C")

    assert "pi_3" in planner.constraint_predicates
    assert planner.constraint_predicates["pi_3"].polarity == "negative"
    assert "pi_1" not in planner.constraint_predicates
    assert "pi_2" not in planner.constraint_predicates


def test_multiple_global_constraints():
    """Multiple G(!pi_X) constraints all active simultaneously."""
    mock = MockLLM()
    mock.ltl_nl_formula = {
        "pi_predicates": {
            "pi_1": "Go to the park",
            "pi_2": "Near building A",
            "pi_3": "Near building B",
        },
        "ltl_nl_formula": "F pi_1 & G(!pi_2) & G(!pi_3)",
    }
    planner = LTLSymbolicPlanner(mock)
    planner.plan_from_natural_language("Go to park, avoid both buildings")

    assert "pi_2" in planner.constraint_predicates
    assert "pi_3" in planner.constraint_predicates
    assert "pi_1" not in planner.constraint_predicates

    _ = planner.get_next_predicate()
    constraints = planner.get_active_constraints()
    assert len(constraints) == 2
    descs = [c.description for c in constraints]
    assert "Near building A" in descs
    assert "Near building B" in descs


def test_classify_global_positive():
    """G(pi_2) makes pi_2 a positive constraint."""
    mock = MockLLM()
    mock.ltl_nl_formula = {
        "pi_predicates": {
            "pi_1": "Fly to the landmark",
            "pi_2": "Above 10 meters altitude",
        },
        "ltl_nl_formula": "F pi_1 & G(pi_2)",
    }
    planner = LTLSymbolicPlanner(mock)
    planner.plan_from_natural_language("Fly to landmark, always stay above 10m")

    assert "pi_2" in planner.constraint_predicates
    assert planner.constraint_predicates["pi_2"].polarity == "positive"
    assert "pi_1" not in planner.constraint_predicates


def test_positive_constraint_single_goal():
    """G(pi_2) with single goal: full sequence works."""
    mock = MockLLM()
    mock.ltl_nl_formula = {
        "pi_predicates": {
            "pi_1": "Reach the tower",
            "pi_2": "Altitude above 20m",
        },
        "ltl_nl_formula": "F pi_1 & G(pi_2)",
    }
    planner = LTLSymbolicPlanner(mock)
    planner.plan_from_natural_language("reach tower, always above 20m")

    order = []
    while True:
        p = planner.get_next_predicate()
        if p is None:
            break
        order.append(p)
        planner.advance_state(p)
    assert order == ["Reach the tower"]
    assert planner.finished


def test_positive_constraint_multi_step_sequence():
    """Positive constraint with three-step sequence: goals unaffected."""
    mock = MockLLM()
    mock.ltl_nl_formula = {
        "pi_predicates": {
            "pi_1": "Go to A",
            "pi_2": "Go to B",
            "pi_3": "Go to C",
            "pi_4": "Keep road visible",
        },
        "ltl_nl_formula": "F pi_3 & (!pi_3 U pi_2) & (!pi_2 U pi_1) & G(pi_4)",
    }
    planner = LTLSymbolicPlanner(mock)
    planner.plan_from_natural_language("A then B then C, always keep road visible")

    assert planner.constraint_predicates["pi_4"].polarity == "positive"
    assert "pi_1" not in planner.constraint_predicates
    assert "pi_2" not in planner.constraint_predicates
    assert "pi_3" not in planner.constraint_predicates

    order = []
    while True:
        p = planner.get_next_predicate()
        if p is None:
            break
        constraints = planner.get_active_constraints()
        assert any(c.description == "Keep road visible" for c in constraints)
        order.append(p)
        planner.advance_state(p)
    assert order == ["Go to A", "Go to B", "Go to C"]


def test_scoped_positive_and_negative_together():
    """Scoped positive (!pi_4 U pi_2) + scoped negative (!pi_3 U pi_1) together."""
    mock = MockLLM()
    mock.ltl_nl_formula = {
        "pi_predicates": {
            "pi_1": "Go to the tree",
            "pi_2": "Go to the streetlight",
            "pi_3": "Near the red car",
            "pi_4": "River visible",
        },
        "ltl_nl_formula": "F pi_2 & (!pi_2 U pi_1) & (!pi_3 U pi_1) & (pi_4 U pi_1)",
    }
    planner = LTLSymbolicPlanner(mock)
    planner.plan_from_natural_language("tree then streetlight, avoid car and keep river until tree")

    assert planner.constraint_predicates["pi_3"].polarity == "negative"
    assert planner.constraint_predicates["pi_4"].polarity == "positive"

    current = planner.get_next_predicate()
    assert current == "Go to the tree"
    constraints = planner.get_active_constraints()
    descs = {c.description: c.polarity for c in constraints}
    assert descs["Near the red car"] == "negative"
    assert descs["River visible"] == "positive"

    planner.advance_state(current)
    current = planner.get_next_predicate()
    assert current == "Go to the streetlight"
    constraints = planner.get_active_constraints()
    assert len(constraints) == 0


# -----------------------------------------------------------------------
# Complex automaton traversal and constraint lifecycle
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


def _active_descs(planner):
    """Return set of active constraint descriptions."""
    return {c.description for c in planner.get_active_constraints()}


def _active_map(planner):
    """Return dict of active constraint description -> polarity."""
    return {c.description: c.polarity for c in planner.get_active_constraints()}


class TestLongSequence:
    """Five-step ordered sequence with no constraints."""

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

    def test_no_constraints(self):
        planner = self._make()
        assert planner.constraint_predicates == {}

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


class TestGlobalConstraintThroughoutLongSequence:
    """Four-step sequence with one global avoidance active at every step."""

    def _make(self):
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {
                "pi_1": "Go to A",
                "pi_2": "Go to B",
                "pi_3": "Go to C",
                "pi_4": "Go to D",
                "pi_5": "Near the danger zone",
            },
            "ltl_nl_formula": (
                "F pi_4 & (!pi_4 U pi_3) & (!pi_3 U pi_2) "
                "& (!pi_2 U pi_1) & G(!pi_5)"
            ),
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language("A B C D, avoid danger zone")
        return planner

    def test_constraint_active_at_every_goal_step(self):
        planner = self._make()
        for i in range(4):
            p = planner.get_next_predicate()
            assert p is not None, f"Step {i+1} should return a goal"
            assert "Near the danger zone" in _active_descs(planner), (
                f"Global constraint should be active at step {i+1}"
            )
            planner.advance_state(p)
        assert planner.get_next_predicate() is None
        assert planner.finished

    def test_goals_not_in_constraints(self):
        planner = self._make()
        assert "pi_1" not in planner.constraint_predicates
        assert "pi_2" not in planner.constraint_predicates
        assert "pi_3" not in planner.constraint_predicates
        assert "pi_4" not in planner.constraint_predicates
        assert "pi_5" in planner.constraint_predicates

    def test_constraint_inactive_after_finish(self):
        planner = self._make()
        _run_full(planner)
        assert planner.get_active_constraints() == []


class TestMultipleGlobalConstraintsThroughSequence:
    """Three-step sequence with three simultaneous global avoidance constraints."""

    def _make(self):
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {
                "pi_1": "Go to A",
                "pi_2": "Go to B",
                "pi_3": "Go to C",
                "pi_4": "Near zone X",
                "pi_5": "Near zone Y",
                "pi_6": "Near zone Z",
            },
            "ltl_nl_formula": (
                "F pi_3 & (!pi_3 U pi_2) & (!pi_2 U pi_1) "
                "& G(!pi_4) & G(!pi_5) & G(!pi_6)"
            ),
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language("A B C, avoid X Y Z")
        return planner

    def test_all_three_constraints_active_throughout(self):
        planner = self._make()
        for i in range(3):
            p = planner.get_next_predicate()
            assert p is not None
            active = _active_descs(planner)
            assert "Near zone X" in active, f"Step {i+1}: X should be active"
            assert "Near zone Y" in active, f"Step {i+1}: Y should be active"
            assert "Near zone Z" in active, f"Step {i+1}: Z should be active"
            planner.advance_state(p)
        assert planner.get_next_predicate() is None
        assert planner.finished

    def test_correct_order(self):
        assert _run_full(self._make()) == ["Go to A", "Go to B", "Go to C"]


class TestScopedConstraintReleaseMidSequence:
    """Four-step sequence with scoped avoidance releasing after step 2."""

    def _make(self):
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {
                "pi_1": "Go to A",
                "pi_2": "Go to B",
                "pi_3": "Go to C",
                "pi_4": "Go to D",
                "pi_5": "Near the hazard",
            },
            # Avoid hazard until B is reached
            "ltl_nl_formula": (
                "F pi_4 & (!pi_4 U pi_3) & (!pi_3 U pi_2) "
                "& (!pi_2 U pi_1) & (!pi_5 U pi_2)"
            ),
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language("A B C D, avoid hazard until B")
        return planner

    def test_constraint_active_before_release_point(self):
        planner = self._make()
        # Step 1: Go to A - hazard should be active
        p1 = planner.get_next_predicate()
        assert p1 == "Go to A"
        assert "Near the hazard" in _active_descs(planner)
        planner.advance_state(p1)

        # Step 2: Go to B - hazard should still be active (releases AFTER B)
        p2 = planner.get_next_predicate()
        assert p2 == "Go to B"
        assert "Near the hazard" in _active_descs(planner)
        planner.advance_state(p2)

    def test_constraint_released_after_release_point(self):
        planner = self._make()
        _step(planner, 2)  # advance through A and B

        # Step 3: Go to C - hazard should be released
        p3 = planner.get_next_predicate()
        assert p3 == "Go to C"
        assert "Near the hazard" not in _active_descs(planner)
        planner.advance_state(p3)

        # Step 4: Go to D - still released
        p4 = planner.get_next_predicate()
        assert p4 == "Go to D"
        assert "Near the hazard" not in _active_descs(planner)


class TestTwoScopedConstraintsDifferentReleasePoints:
    """Four-step sequence with two scoped constraints releasing at steps 1 and 3."""

    def _make(self):
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {
                "pi_1": "Go to A",
                "pi_2": "Go to B",
                "pi_3": "Go to C",
                "pi_4": "Go to D",
                "pi_5": "Near hazard alpha",
                "pi_6": "Near hazard beta",
            },
            # alpha releases after A, beta releases after C
            "ltl_nl_formula": (
                "F pi_4 & (!pi_4 U pi_3) & (!pi_3 U pi_2) "
                "& (!pi_2 U pi_1) & (!pi_5 U pi_1) & (!pi_6 U pi_3)"
            ),
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language(
            "A B C D, avoid alpha until A, avoid beta until C"
        )
        return planner

    def test_both_active_initially(self):
        planner = self._make()
        p1 = planner.get_next_predicate()
        assert p1 == "Go to A"
        active = _active_descs(planner)
        assert "Near hazard alpha" in active
        assert "Near hazard beta" in active

    def test_alpha_releases_after_A(self):
        planner = self._make()
        _step(planner, 1)  # complete A

        p2 = planner.get_next_predicate()
        assert p2 == "Go to B"
        active = _active_descs(planner)
        assert "Near hazard alpha" not in active, "Alpha should release after A"
        assert "Near hazard beta" in active, "Beta should still be active"

    def test_beta_releases_after_C(self):
        planner = self._make()
        _step(planner, 3)  # complete A, B, C

        p4 = planner.get_next_predicate()
        assert p4 == "Go to D"
        active = _active_descs(planner)
        assert "Near hazard alpha" not in active
        assert "Near hazard beta" not in active, "Beta should release after C"

    def test_full_order(self):
        assert _run_full(self._make()) == [
            "Go to A", "Go to B", "Go to C", "Go to D"
        ]


class TestGlobalPlusScopedConstraints:
    """Global avoidance persists while scoped avoidance releases."""

    def _make(self):
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {
                "pi_1": "Go to A",
                "pi_2": "Go to B",
                "pi_3": "Go to C",
                "pi_4": "Near the wall",      # scoped: releases after B
                "pi_5": "Over the river",      # global: always active
            },
            "ltl_nl_formula": (
                "F pi_3 & (!pi_3 U pi_2) & (!pi_2 U pi_1) "
                "& (!pi_4 U pi_2) & G(!pi_5)"
            ),
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language("A B C, avoid wall until B, never over river")
        return planner

    def test_both_active_at_step_1(self):
        planner = self._make()
        planner.get_next_predicate()
        active = _active_descs(planner)
        assert "Near the wall" in active
        assert "Over the river" in active

    def test_both_active_at_step_2(self):
        planner = self._make()
        _step(planner, 1)
        planner.get_next_predicate()
        active = _active_descs(planner)
        assert "Near the wall" in active, "Scoped still active before B completes"
        assert "Over the river" in active

    def test_scoped_releases_global_persists_at_step_3(self):
        planner = self._make()
        _step(planner, 2)  # complete A and B
        planner.get_next_predicate()
        active = _active_descs(planner)
        assert "Near the wall" not in active, "Scoped should release after B"
        assert "Over the river" in active, "Global should still be active"


class TestPositiveConstraintThroughSequence:
    """Positive (maintenance) constraint stays active through entire 4-step sequence."""

    def _make(self):
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {
                "pi_1": "Go to A",
                "pi_2": "Go to B",
                "pi_3": "Go to C",
                "pi_4": "Go to D",
                "pi_5": "Above 10m altitude",
            },
            "ltl_nl_formula": (
                "F pi_4 & (!pi_4 U pi_3) & (!pi_3 U pi_2) "
                "& (!pi_2 U pi_1) & G(pi_5)"
            ),
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language("A B C D, always above 10m")
        return planner

    def test_positive_constraint_at_every_step(self):
        planner = self._make()
        for i in range(4):
            p = planner.get_next_predicate()
            assert p is not None
            active = _active_map(planner)
            assert "Above 10m altitude" in active, f"Step {i+1}: positive constraint missing"
            assert active["Above 10m altitude"] == "positive"
            planner.advance_state(p)
        assert planner.get_next_predicate() is None
        assert planner.finished

    def test_only_goals_in_execution_order(self):
        order = _run_full(self._make())
        assert order == ["Go to A", "Go to B", "Go to C", "Go to D"]


class TestScopedPositiveRelease:
    """Scoped positive constraint (pi_X U pi_Y) releases after Y is achieved."""

    def _make(self):
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {
                "pi_1": "Go to A",
                "pi_2": "Go to B",
                "pi_3": "Go to C",
                "pi_4": "Road visible",
            },
            # Keep road visible until B
            "ltl_nl_formula": (
                "F pi_3 & (!pi_3 U pi_2) & (!pi_2 U pi_1) & (pi_4 U pi_2)"
            ),
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language("A B C, keep road visible until B")
        return planner

    def test_active_before_release(self):
        planner = self._make()
        p1 = planner.get_next_predicate()
        assert p1 == "Go to A"
        active = _active_map(planner)
        assert "Road visible" in active
        assert active["Road visible"] == "positive"

        planner.advance_state(p1)
        p2 = planner.get_next_predicate()
        assert p2 == "Go to B"
        assert "Road visible" in _active_descs(planner)

    def test_released_after_B(self):
        planner = self._make()
        _step(planner, 2)  # complete A and B

        p3 = planner.get_next_predicate()
        assert p3 == "Go to C"
        assert "Road visible" not in _active_descs(planner)


class TestMixedScopedConstraintTypes:
    """Scoped negative + scoped positive releasing at different points."""

    def _make(self):
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {
                "pi_1": "Go to A",
                "pi_2": "Go to B",
                "pi_3": "Go to C",
                "pi_4": "Go to D",
                "pi_5": "Near the fence",     # negative, releases after A
                "pi_6": "Lake visible",        # positive, releases after C
            },
            "ltl_nl_formula": (
                "F pi_4 & (!pi_4 U pi_3) & (!pi_3 U pi_2) "
                "& (!pi_2 U pi_1) & (!pi_5 U pi_1) & (pi_6 U pi_3)"
            ),
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language(
            "A B C D, avoid fence until A, keep lake visible until C"
        )
        return planner

    def test_both_active_initially(self):
        planner = self._make()
        planner.get_next_predicate()
        m = _active_map(planner)
        assert m["Near the fence"] == "negative"
        assert m["Lake visible"] == "positive"

    def test_fence_released_lake_still_active_at_B(self):
        planner = self._make()
        _step(planner, 1)
        planner.get_next_predicate()
        active = _active_descs(planner)
        assert "Near the fence" not in active
        assert "Lake visible" in active

    def test_both_released_at_D(self):
        planner = self._make()
        _step(planner, 3)
        planner.get_next_predicate()
        assert _active_descs(planner) == set()

    def test_full_order(self):
        assert _run_full(self._make()) == [
            "Go to A", "Go to B", "Go to C", "Go to D"
        ]


class TestGlobalMixedWithScopedMixed:
    """Global negative + global positive + scoped negative, all together."""

    def _make(self):
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {
                "pi_1": "Go to A",
                "pi_2": "Go to B",
                "pi_3": "Go to C",
                "pi_4": "Above 20m",           # global positive
                "pi_5": "Over the highway",     # global negative
                "pi_6": "Near the crane",       # scoped negative, releases after B
            },
            "ltl_nl_formula": (
                "F pi_3 & (!pi_3 U pi_2) & (!pi_2 U pi_1) "
                "& G(pi_4) & G(!pi_5) & (!pi_6 U pi_2)"
            ),
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language(
            "A B C, always above 20m, never over highway, avoid crane until B"
        )
        return planner

    def test_all_three_active_at_step_1(self):
        planner = self._make()
        planner.get_next_predicate()
        m = _active_map(planner)
        assert m["Above 20m"] == "positive"
        assert m["Over the highway"] == "negative"
        assert m["Near the crane"] == "negative"

    def test_scoped_releases_globals_persist_at_step_3(self):
        planner = self._make()
        _step(planner, 2)
        planner.get_next_predicate()
        m = _active_map(planner)
        assert "Above 20m" in m, "Global positive should persist"
        assert "Over the highway" in m, "Global negative should persist"
        assert "Near the crane" not in m, "Scoped should release after B"

    def test_correct_order(self):
        assert _run_full(self._make()) == ["Go to A", "Go to B", "Go to C"]


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

    def test_constraints_empty_after_finished(self):
        """Active constraints list is empty once mission is finished."""
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {
                "pi_1": "Go to A",
                "pi_2": "Danger zone",
            },
            "ltl_nl_formula": "F pi_1 & G(!pi_2)",
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language("A, avoid danger")
        _run_full(planner)
        assert planner.get_active_constraints() == []

    def test_six_predicates_four_constraints(self):
        """Complex formula with 2 goals and 4 constraints (2 global, 2 scoped)."""
        mock = MockLLM()
        mock.ltl_nl_formula = {
            "pi_predicates": {
                "pi_1": "Go to A",
                "pi_2": "Go to B",
                "pi_3": "Near zone X",       # global negative
                "pi_4": "Near zone Y",       # global negative
                "pi_5": "Near zone W",       # scoped negative, releases after A
                "pi_6": "Path clear",        # global positive
            },
            "ltl_nl_formula": (
                "F pi_2 & (!pi_2 U pi_1) "
                "& G(!pi_3) & G(!pi_4) & (!pi_5 U pi_1) & G(pi_6)"
            ),
        }
        planner = LTLSymbolicPlanner(mock)
        planner.plan_from_natural_language("A then B with 4 constraints")

        assert len(planner.constraint_predicates) == 4
        assert planner.constraint_predicates["pi_3"].polarity == "negative"
        assert planner.constraint_predicates["pi_4"].polarity == "negative"
        assert planner.constraint_predicates["pi_5"].polarity == "negative"
        assert planner.constraint_predicates["pi_6"].polarity == "positive"

        # At step 1: all 4 active
        p1 = planner.get_next_predicate()
        assert p1 == "Go to A"
        assert len(planner.get_active_constraints()) == 4

        planner.advance_state(p1)

        # At step 2: scoped W released, 3 remain
        p2 = planner.get_next_predicate()
        assert p2 == "Go to B"
        active = _active_descs(planner)
        assert "Near zone X" in active
        assert "Near zone Y" in active
        assert "Path clear" in active
        assert "Near zone W" not in active


def _step(planner, n):
    """Advance the planner through n steps."""
    for _ in range(n):
        p = planner.get_next_predicate()
        assert p is not None
        planner.advance_state(p)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
