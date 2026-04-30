"""
Tier 2 prompt smoke tests for negative constraint support.

These tests make real LLM API calls to verify the LTL planner prompt
correctly teaches the model to generate G(!pi_X) formulas for avoidance
constraints, and that the automaton-based classifier identifies them.

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


@needs_api
@tier2
def test_llm_generates_global_avoidance():
    """LLM should produce G(!pi_X) for 'never fly over building C'."""
    from rvln.ai.llm_interface import LLMUserInterface
    from rvln.ai.ltl_planner import LTLSymbolicPlanner

    llm = LLMUserInterface(model="gpt-4o")
    planner = LTLSymbolicPlanner(llm)
    planner.plan_from_natural_language(
        "Go to tree A, then go to streetlight B, but never fly over building C."
    )

    assert len(planner.constraint_predicates) >= 1, (
        f"Expected at least 1 constraint predicate, got {planner.constraint_predicates}. "
        f"Formula: {planner._raw_formula}, pi_map: {planner.pi_map}"
    )

    _ = planner.get_next_predicate()
    constraints = planner.get_active_constraints()
    assert len(constraints) >= 1, (
        f"Expected at least 1 active constraint, got {constraints}."
    )


@needs_api
@tier2
def test_llm_generates_scoped_avoidance():
    """LLM should encode 'stay away from X until you reach Y' as scoped constraint."""
    from rvln.ai.llm_interface import LLMUserInterface
    from rvln.ai.ltl_planner import LTLSymbolicPlanner

    llm = LLMUserInterface(model="gpt-4o")
    planner = LTLSymbolicPlanner(llm)
    planner.plan_from_natural_language(
        "Approach the tree, then go to the streetlight, "
        "but stay away from the red car until you reach the streetlight."
    )

    assert len(planner.constraint_predicates) >= 1, (
        f"Expected at least 1 constraint, got {planner.constraint_predicates}. "
        f"Formula: {planner._raw_formula}"
    )

    _ = planner.get_next_predicate()
    constraints = planner.get_active_constraints()
    car_constraint = [c for c in constraints if "car" in c.lower() or "red" in c.lower()]
    assert len(car_constraint) >= 1, (
        f"Expected constraint about 'red car', got {constraints}."
    )


@needs_api
@tier2
def test_llm_no_constraints_on_simple_sequence():
    """Simple sequences should produce no constraint predicates."""
    from rvln.ai.llm_interface import LLMUserInterface
    from rvln.ai.ltl_planner import LTLSymbolicPlanner

    llm = LLMUserInterface(model="gpt-4o")
    planner = LTLSymbolicPlanner(llm)
    planner.plan_from_natural_language(
        "Go to the tree, then go to the streetlight, then go to the building."
    )

    assert planner.constraint_predicates == {}, (
        f"Simple sequence should have no constraints, got {planner.constraint_predicates}. "
        f"Formula: {planner._raw_formula}"
    )
