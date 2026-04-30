"""
Tier 2 NL-to-LTL-NL conversion tests with real gpt-4o API calls.

Tests the full pipeline: natural language instruction -> LLM -> JSON with
pi_predicates and ltl_nl_formula -> Spot automaton -> constraint classification
-> correct execution order.

Run: conda run -n rvln-sim pytest tests/test_prompt_constraints.py -v -m "tier2"
Skip in CI: pytest tests/ -m "not tier2"
"""
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest

spot = pytest.importorskip("spot", reason="spot not available outside rvln-sim")

needs_api = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="No OPENAI_API_KEY"
)
tier2 = pytest.mark.tier2


def _make_planner():
    from rvln.ai.llm_interface import LLMUserInterface
    from rvln.ai.ltl_planner import LTLSymbolicPlanner

    llm = LLMUserInterface(model="gpt-4o")
    return LTLSymbolicPlanner(llm)


def _run_to_completion(planner):
    """Execute the automaton to completion, returning the ordered list of subgoals."""
    order = []
    while True:
        p = planner.get_next_predicate()
        if p is None:
            break
        order.append(p)
        planner.advance_state(p)
    return order


def _diag(planner):
    """Diagnostic string for assertion messages."""
    return (
        f"Formula: {planner._raw_formula}, "
        f"pi_map: {planner.pi_map}, "
        f"constraints: {planner.constraint_predicates}"
    )


# -----------------------------------------------------------------------
# Basic formula structure
# -----------------------------------------------------------------------

@needs_api
@tier2
def test_single_task():
    """Single instruction produces F pi_1 with one predicate."""
    planner = _make_planner()
    planner.plan_from_natural_language("Go to the tree.")

    assert len(planner.pi_map) == 1, f"Expected 1 predicate, got {planner.pi_map}"
    assert planner.constraint_predicates == {}, _diag(planner)

    order = _run_to_completion(planner)
    assert len(order) == 1, f"Expected 1 step, got {order}"
    assert planner.finished


@needs_api
@tier2
def test_two_step_sequence():
    """Two-step sequence: A then B, correct ordering."""
    planner = _make_planner()
    planner.plan_from_natural_language(
        "Go to the tree, then go to the streetlight."
    )

    assert len(planner.pi_map) == 2, f"Expected 2 predicates. {_diag(planner)}"
    assert planner.constraint_predicates == {}, _diag(planner)

    order = _run_to_completion(planner)
    assert len(order) == 2, f"Expected 2 steps, got {order}"
    assert "tree" in order[0].lower(), f"First step should be tree, got {order[0]}"
    assert "streetlight" in order[1].lower() or "light" in order[1].lower(), (
        f"Second step should be streetlight, got {order[1]}"
    )


@needs_api
@tier2
def test_three_step_sequence():
    """Three-step sequence: correct ordering and no constraints."""
    planner = _make_planner()
    planner.plan_from_natural_language(
        "Go to the tree, then go to the streetlight, then go to the building."
    )

    assert len(planner.pi_map) == 3, f"Expected 3 predicates. {_diag(planner)}"
    assert planner.constraint_predicates == {}, (
        f"Simple sequence should have no constraints. {_diag(planner)}"
    )

    order = _run_to_completion(planner)
    assert len(order) == 3, f"Expected 3 steps, got {order}"
    assert "tree" in order[0].lower(), f"First should be tree: {order}"
    assert planner.finished


@needs_api
@tier2
def test_parallel_independent_tasks():
    """Independent tasks (no ordering) should all be reachable."""
    planner = _make_planner()
    planner.plan_from_natural_language(
        "Deliver the pen to location A. Also deliver the apple to location B."
    )

    assert len(planner.pi_map) >= 2, f"Expected >=2 predicates. {_diag(planner)}"
    assert planner.constraint_predicates == {}, _diag(planner)

    order = _run_to_completion(planner)
    assert len(order) >= 2, f"Should complete all tasks, got {order}"
    assert planner.finished


# -----------------------------------------------------------------------
# Constraint classification
# -----------------------------------------------------------------------

@needs_api
@tier2
def test_global_avoidance():
    """G(!pi_X) for 'never fly over building C'."""
    planner = _make_planner()
    planner.plan_from_natural_language(
        "Go to tree A, then go to streetlight B, but never fly over building C."
    )

    assert len(planner.constraint_predicates) >= 1, (
        f"Expected at least 1 constraint. {_diag(planner)}"
    )

    neg = [c for c in planner.constraint_predicates.values() if c.polarity == "negative"]
    assert len(neg) >= 1, f"Expected negative constraint. {_diag(planner)}"

    _ = planner.get_next_predicate()
    constraints = planner.get_active_constraints()
    assert len(constraints) >= 1, f"Expected active constraint. {_diag(planner)}"

    building_c = [c for c in constraints if "building" in c.description.lower()]
    assert len(building_c) >= 1, (
        f"Expected constraint about building C. Got: {[c.description for c in constraints]}"
    )


@needs_api
@tier2
def test_scoped_avoidance():
    """Scoped avoidance: 'stay away from X until Y'."""
    planner = _make_planner()
    planner.plan_from_natural_language(
        "Approach the tree, then go to the streetlight, "
        "but stay away from the red car until you reach the streetlight."
    )

    assert len(planner.constraint_predicates) >= 1, (
        f"Expected at least 1 constraint. {_diag(planner)}"
    )

    _ = planner.get_next_predicate()
    constraints = planner.get_active_constraints()
    car_constraint = [
        c for c in constraints
        if "car" in c.description.lower() or "red" in c.description.lower()
    ]
    assert len(car_constraint) >= 1, (
        f"Expected constraint about 'red car'. Got: {[c.description for c in constraints]}"
    )


@needs_api
@tier2
def test_global_positive_constraint():
    """G(pi_X) for 'always stay above 10 meters altitude'."""
    planner = _make_planner()
    planner.plan_from_natural_language(
        "Fly to the landmark, but always stay above 10 meters altitude."
    )

    assert len(planner.constraint_predicates) >= 1, (
        f"Expected at least 1 constraint. {_diag(planner)}"
    )

    pos = [c for c in planner.constraint_predicates.values() if c.polarity == "positive"]
    assert len(pos) >= 1, (
        f"Expected positive constraint for altitude. {_diag(planner)}"
    )

    order = _run_to_completion(planner)
    assert len(order) == 1, f"Only goal task should execute, got {order}"
    assert "landmark" in order[0].lower(), f"Goal should be landmark, got {order[0]}"


@needs_api
@tier2
def test_scoped_positive_constraint():
    """Scoped maintenance: 'keep the river visible until you reach the tree'."""
    planner = _make_planner()
    planner.plan_from_natural_language(
        "Go to the tree, then the streetlight, "
        "but keep the river visible until you reach the tree."
    )

    assert len(planner.constraint_predicates) >= 1, (
        f"Expected at least 1 constraint for river visibility. {_diag(planner)}"
    )

    pos = [c for c in planner.constraint_predicates.values() if c.polarity == "positive"]
    assert len(pos) >= 1, (
        f"Expected positive constraint for river. {_diag(planner)}"
    )

    first = planner.get_next_predicate()
    assert "tree" in first.lower(), f"First goal should be tree, got {first}"
    c1 = planner.get_active_constraints()
    river = [c for c in c1 if "river" in c.description.lower()]
    assert len(river) >= 1, (
        f"River constraint should be active before tree. Got: {[c.description for c in c1]}"
    )

    planner.advance_state(first)
    second = planner.get_next_predicate()
    assert second is not None, "Streetlight should be next"
    c2 = planner.get_active_constraints()
    river_after = [c for c in c2 if "river" in c.description.lower()]
    assert len(river_after) == 0, (
        f"River constraint should release after tree. Still active: {[c.description for c in c2]}"
    )


@needs_api
@tier2
def test_mixed_positive_and_negative_constraints():
    """Mixed: G(pi_X) & G(!pi_Y) together."""
    planner = _make_planner()
    planner.plan_from_natural_language(
        "Navigate to the bridge, always stay above the treeline, "
        "and never fly over the highway."
    )

    assert len(planner.constraint_predicates) >= 2, (
        f"Expected at least 2 constraints (positive + negative). {_diag(planner)}"
    )

    polarities = {c.polarity for c in planner.constraint_predicates.values()}
    assert "positive" in polarities, f"Expected a positive constraint. {_diag(planner)}"
    assert "negative" in polarities, f"Expected a negative constraint. {_diag(planner)}"

    order = _run_to_completion(planner)
    assert len(order) == 1, f"Only goal task should execute. {_diag(planner)}"
    assert "bridge" in order[0].lower(), f"Goal should be bridge, got {order[0]}"


@needs_api
@tier2
def test_multiple_avoidance_constraints():
    """Multiple G(!pi_X) constraints active simultaneously."""
    planner = _make_planner()
    planner.plan_from_natural_language(
        "First go to the park, then navigate to the traffic light, "
        "and never go near building A or building B at any point."
    )

    assert len(planner.constraint_predicates) >= 2, (
        f"Expected at least 2 avoidance constraints. {_diag(planner)}"
    )

    neg = [c for c in planner.constraint_predicates.values() if c.polarity == "negative"]
    assert len(neg) >= 2, (
        f"Expected at least 2 negative constraints. {_diag(planner)}"
    )

    _ = planner.get_next_predicate()
    constraints = planner.get_active_constraints()
    assert len(constraints) >= 2, (
        f"Both building constraints should be active. Got: {[c.description for c in constraints]}"
    )


# -----------------------------------------------------------------------
# Branching / disjunction
# -----------------------------------------------------------------------

@needs_api
@tier2
def test_disjunction_before_goal():
    """'Do A or B, then C' should use OR logic."""
    planner = _make_planner()
    planner.plan_from_natural_language(
        "Eventually deliver the pen to location D, but only after you have "
        "delivered either a drink or an apple to location E."
    )

    assert len(planner.pi_map) == 3, f"Expected 3 predicates. {_diag(planner)}"
    assert planner.constraint_predicates == {}, _diag(planner)

    first = planner.get_next_predicate()
    assert first is not None
    assert "drink" in first.lower() or "apple" in first.lower(), (
        f"First task should be drink or apple, got {first}"
    )


# -----------------------------------------------------------------------
# Complex / long-horizon instructions
# -----------------------------------------------------------------------

@needs_api
@tier2
def test_five_step_sequence():
    """Five-step ordered sequence: all goals, no constraints, correct order."""
    planner = _make_planner()
    planner.plan_from_natural_language(
        "I need five things done in order: first go to the fridge, "
        "then deliver the water, then deliver the apple, "
        "then go to the living room, and finally go to the garage."
    )

    assert len(planner.pi_map) == 5, f"Expected 5 predicates. {_diag(planner)}"
    assert planner.constraint_predicates == {}, (
        f"Pure sequence should have no constraints. {_diag(planner)}"
    )

    order = _run_to_completion(planner)
    assert len(order) == 5, f"Expected 5 steps, got {len(order)}: {order}"
    assert "fridge" in order[0].lower(), f"First should be fridge: {order}"
    assert "garage" in order[-1].lower(), f"Last should be garage: {order}"
    assert planner.finished


@needs_api
@tier2
def test_long_sequence_with_global_avoidance():
    """Four-step sequence with a global avoidance constraint throughout."""
    planner = _make_planner()
    planner.plan_from_natural_language(
        "Go to the park, then the fountain, then the library, then the school, "
        "and never fly over the highway at any point during the mission."
    )

    assert len(planner.pi_map) >= 5, f"Expected >=5 predicates (4 goals + 1 constraint). {_diag(planner)}"

    neg = [c for c in planner.constraint_predicates.values() if c.polarity == "negative"]
    assert len(neg) >= 1, f"Expected highway avoidance constraint. {_diag(planner)}"

    order = _run_to_completion(planner)
    assert len(order) == 4, f"Expected 4 goal steps, got {order}"
    assert "park" in order[0].lower(), f"First should be park: {order}"
    assert "school" in order[-1].lower(), f"Last should be school: {order}"
    assert planner.finished


@needs_api
@tier2
def test_sequence_with_many_global_constraints():
    """Three-step sequence with three simultaneous avoidance constraints."""
    planner = _make_planner()
    planner.plan_from_natural_language(
        "Go to the tree, then the fountain, then the parking lot. "
        "Throughout the entire mission, never fly over the highway, "
        "never go near the construction zone, "
        "and never fly above the restricted airspace."
    )

    goals_count = len(planner.pi_map) - len(planner.constraint_predicates)
    assert goals_count == 3, (
        f"Expected 3 goal predicates, got {goals_count}. {_diag(planner)}"
    )
    assert len(planner.constraint_predicates) >= 3, (
        f"Expected at least 3 avoidance constraints. {_diag(planner)}"
    )

    neg = [c for c in planner.constraint_predicates.values() if c.polarity == "negative"]
    assert len(neg) >= 3, f"All constraints should be negative. {_diag(planner)}"

    first = planner.get_next_predicate()
    assert first is not None
    constraints = planner.get_active_constraints()
    assert len(constraints) >= 3, (
        f"All 3 avoidance constraints should be active at step 1. "
        f"Got: {[c.description for c in constraints]}"
    )

    planner.advance_state(first)
    order = _run_to_completion(planner)
    assert len(order) == 2, f"2 more steps after first, got {order}"
    assert planner.finished


@needs_api
@tier2
def test_sequence_with_mixed_global_constraints():
    """Three-step sequence with one positive and one negative global constraint."""
    planner = _make_planner()
    planner.plan_from_natural_language(
        "Go to the warehouse, then the dock, then the storage yard. "
        "Always stay above 15 meters altitude, "
        "and never fly over the residential area."
    )

    goals_count = len(planner.pi_map) - len(planner.constraint_predicates)
    assert goals_count == 3, (
        f"Expected 3 goal predicates. {_diag(planner)}"
    )
    assert len(planner.constraint_predicates) >= 2, (
        f"Expected at least 2 constraints. {_diag(planner)}"
    )

    pos = [c for c in planner.constraint_predicates.values() if c.polarity == "positive"]
    neg = [c for c in planner.constraint_predicates.values() if c.polarity == "negative"]
    assert len(pos) >= 1, f"Expected at least 1 positive constraint. {_diag(planner)}"
    assert len(neg) >= 1, f"Expected at least 1 negative constraint. {_diag(planner)}"

    order = _run_to_completion(planner)
    assert len(order) == 3, f"Expected 3 goal steps. {_diag(planner)}"
    assert "warehouse" in order[0].lower(), f"First should be warehouse: {order}"
    assert planner.finished


@needs_api
@tier2
def test_long_horizon_with_scoped_and_global_constraints():
    """Four goals with a scoped avoidance that releases mid-sequence plus a global avoidance."""
    planner = _make_planner()
    planner.plan_from_natural_language(
        "Go to the tree, then the bridge, then the tower, then the field. "
        "Stay away from the red car until you reach the bridge, "
        "and never fly over the river at any point."
    )

    assert len(planner.constraint_predicates) >= 2, (
        f"Expected at least 2 constraints (scoped + global). {_diag(planner)}"
    )

    first = planner.get_next_predicate()
    assert "tree" in first.lower(), f"First should be tree, got {first}"
    c1 = planner.get_active_constraints()
    car_c1 = [c for c in c1 if "car" in c.description.lower() or "red" in c.description.lower()]
    river_c1 = [c for c in c1 if "river" in c.description.lower()]
    assert len(car_c1) >= 1, (
        f"Red car constraint should be active at step 1. Got: {[c.description for c in c1]}"
    )
    assert len(river_c1) >= 1, (
        f"River constraint should be active at step 1. Got: {[c.description for c in c1]}"
    )

    planner.advance_state(first)
    second = planner.get_next_predicate()
    assert "bridge" in second.lower(), f"Second should be bridge, got {second}"

    planner.advance_state(second)
    third = planner.get_next_predicate()
    assert third is not None, "Tower should be next"
    c3 = planner.get_active_constraints()
    car_c3 = [c for c in c3 if "car" in c.description.lower() or "red" in c.description.lower()]
    river_c3 = [c for c in c3 if "river" in c.description.lower()]
    assert len(car_c3) == 0, (
        f"Red car constraint should have released after bridge. Still: {[c.description for c in c3]}"
    )
    assert len(river_c3) >= 1, (
        f"River constraint (global) should still be active. Got: {[c.description for c in c3]}"
    )

    planner.advance_state(third)
    remaining = _run_to_completion(planner)
    assert len(remaining) == 1, f"One more step after tower, got {remaining}"
    assert planner.finished
