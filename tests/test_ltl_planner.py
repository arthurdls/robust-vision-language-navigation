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
    print("  [OK] helpers")


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
    print("  [OK] plan valid simple (F pi_1)")


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
    print("  [OK] plan valid sequence (A then B then C)")


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
    print("  [OK] plan valid complex (F pi_1 & (!pi_1 U (pi_2 | pi_3)))")


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
    print("  [OK] plan invalid instruction")


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
    print("  [OK] plan invalid data")


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
    print("  [OK] plan invalid formula")


def test_get_next_without_plan():
    """get_next_predicate returns None when not planned."""
    mock = MockLLM()
    planner = LTLSymbolicPlanner(mock)
    assert planner.get_next_predicate() is None
    print("  [OK] get_next without plan")


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
    print("  [OK] advance unknown task")


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
    print("  [OK] advance no edge / sink")


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
    print("  [OK] normalize key variants (p1/p2)")


def test_classify_global_avoidance():
    """G(!pi_3) makes pi_3 a constraint, not a goal."""
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
    assert "pi_1" not in planner.constraint_predicates
    assert "pi_2" not in planner.constraint_predicates


def test_active_constraints_global():
    """G(!pi_3) should be active at every state."""
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

    # During pi_1
    current = planner.get_next_predicate()
    assert current == "Go to tree A"
    constraints = planner.get_active_constraints()
    assert constraints == ["Flying over building C"]

    # During pi_2
    planner.advance_state(current)
    current = planner.get_next_predicate()
    assert current == "Go to streetlight B"
    constraints = planner.get_active_constraints()
    assert constraints == ["Flying over building C"]

    # After completion
    planner.advance_state(current)
    assert planner.get_next_predicate() is None
    assert planner.get_active_constraints() == []


def test_active_constraints_scoped():
    """!pi_3 U pi_2: pi_3 active before pi_2, released after."""
    mock = MockLLM()
    mock.ltl_nl_formula = {
        "pi_predicates": {
            "pi_1": "Approach the tree",
            "pi_2": "Go to the streetlight",
            "pi_3": "Near the red car",
        },
        "ltl_nl_formula": "F pi_2 & (!pi_2 U pi_1) & (!pi_3 U pi_2)",
    }
    planner = LTLSymbolicPlanner(mock)
    planner.plan_from_natural_language("Approach tree, then streetlight, avoid car")

    # During pi_1: pi_3 is active
    current = planner.get_next_predicate()
    assert current == "Approach the tree"
    constraints = planner.get_active_constraints()
    assert "Near the red car" in constraints

    # During pi_2: pi_3 still active (scoped until pi_2 done)
    planner.advance_state(current)
    current = planner.get_next_predicate()
    assert current == "Go to the streetlight"
    constraints = planner.get_active_constraints()
    assert "Near the red car" in constraints

    # After pi_2 done: pi_3 released
    planner.advance_state(current)
    assert planner.get_active_constraints() == []


def test_no_constraints_simple_sequence():
    """Pure sequence has no constraint predicates."""
    mock = MockLLM()
    mock.ltl_nl_formula = {
        "pi_predicates": {
            "pi_1": "Go to A",
            "pi_2": "Go to B",
            "pi_3": "Go to C",
        },
        "ltl_nl_formula": "F pi_3 & (!pi_3 U pi_2) & (!pi_2 U pi_1)",
    }
    planner = LTLSymbolicPlanner(mock)
    planner.plan_from_natural_language("Go A then B then C")

    assert planner.constraint_predicates == {}
    _ = planner.get_next_predicate()
    assert planner.get_active_constraints() == []


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
    assert "Near building A" in constraints
    assert "Near building B" in constraints


def main():
    print("LTL Planner robustness tests (Spot from rvln-sim)")
    print("-" * 50)
    test_helpers()
    test_plan_valid_simple()
    test_plan_valid_sequence()
    test_plan_valid_complex()
    test_plan_invalid_instruction()
    test_plan_invalid_data()
    test_plan_invalid_formula()
    test_get_next_without_plan()
    test_advance_unknown_task()
    test_advance_no_edge_sink()
    test_normalize_key_variants()
    print("-" * 50)
    print("All robustness tests passed.")


if __name__ == "__main__":
    main()
