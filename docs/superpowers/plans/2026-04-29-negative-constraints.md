# Negative Constraint Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add negative constraint support to the LTL-guided UAV system so the drone enforces avoidance constraints during subgoal execution, using the Spot automaton's edge structure as the single source of truth for constraint classification.

**Architecture:** After the planner builds the Spot automaton, each predicate is classified as goal or constraint by checking whether it ever produces a forward-progressing edge. Active constraints at each state are detected by checking whether the predicate has any valid edge at all. Constraints are injected into the diary monitor's VLM prompts, and violations trigger force-converge followed by the existing convergence correction flow (shared budget).

**Tech Stack:** Python, Spot (LTL automaton via conda-forge), pytest, OpenAI API (for Tier 2/3 smoke tests)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/rvln/ai/ltl_planner.py` | Modify | Add `_classify_predicates()` and `get_active_constraints()` using automaton BDD queries |
| `src/rvln/ai/prompts.py` | Modify | Add constraint predicate docs + examples to LTL prompts; add `{constraints_block}` and `constraint_violated` to diary global/convergence prompts |
| `src/rvln/ai/diary_monitor.py` | Modify | Accept `negative_constraints` parameter; build constraints block; handle `constraint_violated` in `_parse_global_response` |
| `src/rvln/ai/utils/parsing.py` | Modify | Add `G` (Globally) operator support to `parse_ltl_nl` |
| `scripts/run_integration.py` | Modify | Extract constraints from planner per subgoal; pass to `_run_subgoal`; log violations |
| `tests/test_ltl_planner.py` | Modify | Add constraint classification and active constraint tests |
| `tests/test_negative_constraints.py` | Create | Tier 1 tests for monitor constraint injection, violation parsing, end-to-end flow |
| `tests/test_prompt_constraints.py` | Create | Tier 2 tests (real LLM calls) for prompt engineering validation |
| `tests/test_vlm_constraint_prompts.py` | Create | Tier 3 tests (real VLM calls) for `constraint_violated` field validation |
| `tests/test_parsing.py` | Modify | Add tests for `G` operator in `parse_ltl_nl` |

---

### Task 1: Automaton-Based Constraint Classification in LTL Planner

**Files:**
- Modify: `src/rvln/ai/ltl_planner.py:39-103`
- Test: `tests/test_ltl_planner.py`

- [ ] **Step 1: Write failing tests for `_classify_predicates` and `get_active_constraints`**

Add these tests at the end of `tests/test_ltl_planner.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && conda run -n rvln-sim pytest tests/test_ltl_planner.py -v -k "constraint"`
Expected: FAIL with `AttributeError: 'LTLSymbolicPlanner' object has no attribute 'constraint_predicates'`

- [ ] **Step 3: Implement `_classify_predicates` and `get_active_constraints`**

In `src/rvln/ai/ltl_planner.py`, add to `__init__` (after line 52, `self.finished = False`):

```python
self.constraint_predicates: dict[str, str] = {}
```

In `plan_from_natural_language`, add after line 103 (`self.finished = False`):

```python
self.constraint_predicates = self._classify_predicates()
```

Add these two methods to the `LTLSymbolicPlanner` class:

```python
def _classify_predicates(self) -> dict[str, str]:
    """Classify predicates as constraints using the automaton's edge structure.

    A predicate is a constraint (not a goal) if it never has a
    forward-progressing edge (dst != src) at any state in the automaton.
    Goal predicates advance the automaton; constraint predicates only
    appear in negated contexts (G(!pi_X) or !pi_X U pi_Y).
    """
    if not self.pi_map or self.automaton is None:
        return {}

    bdd_false = spot.formula_to_bdd(
        spot.formula("0"), self.automaton.get_dict(), self.automaton
    )
    num_states = self.automaton.num_states()
    constraints: dict[str, str] = {}

    for key in self.pi_map:
        p_idx = _predicate_key_to_index(key)
        is_goal = False

        for state in range(num_states):
            try:
                test_bdd = self._get_bdd_for_single_task(p_idx)
            except ValueError:
                break
            for edge in self.automaton.out(state):
                if edge.dst == state:
                    continue
                if (test_bdd & edge.cond) != bdd_false:
                    is_goal = True
                    break
            if is_goal:
                break

        if not is_goal:
            constraints[key] = self.pi_map[key]

    return constraints

def get_active_constraints(self) -> list[str]:
    """Return NL descriptions of negative constraints active at the current state.

    A constraint is active if making it true produces no valid edge
    (not even a self-loop) from the current automaton state. When the
    constraint's scope expires (e.g., !pi_X U pi_Y after pi_Y is
    achieved), a self-loop edge appears and the constraint is released.
    """
    if self.finished or not self.constraint_predicates or self.automaton is None:
        return []

    bdd_false = spot.formula_to_bdd(
        spot.formula("0"), self.automaton.get_dict(), self.automaton
    )
    active: list[str] = []

    for key, nl_desc in self.constraint_predicates.items():
        p_idx = _predicate_key_to_index(key)
        try:
            test_bdd = self._get_bdd_for_single_task(p_idx)
        except ValueError:
            continue

        has_any_edge = False
        for edge in self.automaton.out(self.current_automaton_state):
            if (test_bdd & edge.cond) != bdd_false:
                has_any_edge = True
                break

        if not has_any_edge:
            active.append(nl_desc)

    return active
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && conda run -n rvln-sim pytest tests/test_ltl_planner.py -v -k "constraint"`
Expected: All 5 new constraint tests PASS

- [ ] **Step 5: Run full existing test suite for regressions**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && conda run -n rvln-sim pytest tests/test_ltl_planner.py -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add src/rvln/ai/ltl_planner.py tests/test_ltl_planner.py
git commit -m "feat: add automaton-based constraint classification to LTL planner"
```

---

### Task 2: Update LTL Prompts for Constraint Predicates

**Files:**
- Modify: `src/rvln/ai/prompts.py:230-367`
- Test: `tests/test_prompt_constraints.py` (new, Tier 2)

- [ ] **Step 1: Create the Tier 2 prompt smoke test file**

Create `tests/test_prompt_constraints.py`:

```python
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
```

- [ ] **Step 2: Add avoidance constraint documentation to `LTL_NL_SYSTEM_PROMPT`**

In `src/rvln/ai/prompts.py`, insert the following block inside `LTL_NL_SYSTEM_PROMPT` after the chain example (after the line `* * **Formula**: `F pi_3 & (!pi_2 U pi_1) & (!pi_3 U pi_2)`` around line 264) and before `### Your Task` (line 268):

```python
        * ---
        * ### Avoidance Constraints (G and Negative Predicates)
        * Some predicates represent CONDITIONS TO AVOID rather than goals to achieve.
        * These are called **constraint predicates**.
        *
        * **G (Globally/Always)**: `G(!pi_X)` means "pi_X must NEVER become true."
        * Use this for unconditional avoidance: things the robot must avoid at all times.
        *
        * * **Example**: "Go to A then B, but never fly over building C."
        *   * `pi_1` = "Go to A"
        *   * `pi_2` = "Go to B"
        *   * `pi_3` = "Flying over building C" (constraint predicate)
        *   * Formula: `F pi_2 & (!pi_2 U pi_1) & G(!pi_3)`
        *
        * **Scoped avoidance using Until**: `!pi_X U pi_Y` can also encode
        * "avoid pi_X until pi_Y is achieved" when pi_X is a spatial condition
        * rather than a goal.
        *
        * * **Example**: "Approach the tree, but stay away from building B until you reach the tree."
        *   * `pi_1` = "Approach the tree"
        *   * `pi_2` = "Near building B" (constraint predicate)
        *   * Formula: `F pi_1 & (!pi_2 U pi_1)`
        *
        * **How to decide if a predicate is a constraint vs. a goal**:
        * * If the instruction says "never", "avoid", "stay away from", "do not go near",
        *   "do not fly over" -- the predicate describes the VIOLATION CONDITION
        *   (what would be bad), and it gets negated with G(!) or placed on the left of U.
        * * If the instruction says "go to", "approach", "reach", "deliver" -- the
        *   predicate describes a GOAL to achieve.
        * * A constraint predicate should describe the state that must NOT occur
        *   (e.g., "Flying over building C", "Near the red car"), NOT the desired
        *   behavior (e.g., NOT "Stay away from building C").
```

- [ ] **Step 3: Add constraint examples to `LTL_NL_EXAMPLES_PROMPT`**

In `src/rvln/ai/prompts.py`, append these examples at the end of `LTL_NL_EXAMPLES_PROMPT` (before the closing `"""`):

```python

User: 'Go to the tree, then go to the streetlight, but never fly over the building.'
Assistant:
{
    "pi_predicates": {
        "pi_1": "Go to the tree",
        "pi_2": "Go to the streetlight",
        "pi_3": "Flying over the building"
    },
    "ltl_nl_formula": "F pi_2 & (!pi_2 U pi_1) & G(!pi_3)"
}

User: 'Approach the sculpture, but stay away from the red car until you reach the sculpture.'
Assistant:
{
    "pi_predicates": {
        "pi_1": "Approach the sculpture",
        "pi_2": "Near the red car"
    },
    "ltl_nl_formula": "F pi_1 & (!pi_2 U pi_1)"
}

User: 'First go to the park, then navigate to the traffic light, and never go near building A or building B at any point.'
Assistant:
{
    "pi_predicates": {
        "pi_1": "Go to the park",
        "pi_2": "Navigate to the traffic light",
        "pi_3": "Near building A",
        "pi_4": "Near building B"
    },
    "ltl_nl_formula": "F pi_2 & (!pi_2 U pi_1) & G(!pi_3) & G(!pi_4)"
}
```

- [ ] **Step 4: Store the raw formula in the planner for logging**

In `src/rvln/ai/ltl_planner.py`, add to `__init__` (after `self.finished = False` on line 52):

```python
self._raw_formula: str = ""
```

In `plan_from_natural_language`, add after `raw_formula = data["ltl_nl_formula"]` (line 77):

```python
self._raw_formula = raw_formula.strip()
```

- [ ] **Step 5: Run Tier 2 prompt smoke tests**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && conda run -n rvln-sim pytest tests/test_prompt_constraints.py -v -m "tier2"`
Expected: All 3 tests PASS. If `test_llm_generates_global_avoidance` fails, iterate on the prompt wording in Steps 2-3.

- [ ] **Step 6: Run existing tests for regression**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && conda run -n rvln-sim pytest tests/test_ltl_planner.py tests/test_parsing.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/rvln/ai/prompts.py src/rvln/ai/ltl_planner.py tests/test_prompt_constraints.py
git commit -m "feat: teach LTL planner prompt about G(!pi_X) avoidance constraints"
```

---

### Task 3: Add `G` Operator Support to LTL-NL Parser

**Files:**
- Modify: `src/rvln/ai/utils/parsing.py:32-88`
- Test: `tests/test_parsing.py`

- [ ] **Step 1: Write failing tests for `G` operator**

Add to `tests/test_parsing.py`:

```python
class TestParseLtlNlConstraints:
    def test_globally_not(self):
        pmap = {"pi_1": "go to A", "pi_2": "flying over building C"}
        result = parse_ltl_nl("G(!pi_2)", pmap)
        assert "ALWAYS" in result or "always" in result
        assert "flying over building C" in result

    def test_globally_bare(self):
        pmap = {"pi_1": "stay on course"}
        result = parse_ltl_nl("G pi_1", pmap)
        assert "ALWAYS" in result or "always" in result
        assert "stay on course" in result

    def test_formula_with_global_constraint(self):
        pmap = {
            "pi_1": "go to A",
            "pi_2": "go to B",
            "pi_3": "flying over building C",
        }
        result = parse_ltl_nl("F pi_2 & (!pi_2 U pi_1) & G(!pi_3)", pmap)
        assert "go to A" in result
        assert "go to B" in result
        assert "flying over building C" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && conda run -n rvln-sim pytest tests/test_parsing.py::TestParseLtlNlConstraints -v`
Expected: FAIL with `ValueError: Could not parse formula or sub-formula: 'G(!pi_2)'`

- [ ] **Step 3: Add `G` operator handling to `parse_ltl_nl`**

In `src/rvln/ai/utils/parsing.py`, add this block after the `F` handling (after line 76 `return f"(eventually {parse_ltl_nl(sub_formula, predicate_map)} must be accomplished)"`) and before the `!` handling (line 78):

```python
    if formula.startswith('G ') or formula.startswith('G('):
        sub_formula = formula[2:].strip() if formula.startswith('G ') else formula[1:].strip()
        return f"(it must ALWAYS be the case that {parse_ltl_nl(sub_formula, predicate_map)})"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && conda run -n rvln-sim pytest tests/test_parsing.py -v`
Expected: All PASS (new and existing)

- [ ] **Step 5: Commit**

```bash
git add src/rvln/ai/utils/parsing.py tests/test_parsing.py
git commit -m "feat: add G (globally) operator support to LTL-NL parser"
```

---

### Task 4: Update Diary Monitor to Accept and Handle Negative Constraints

**Files:**
- Modify: `src/rvln/ai/diary_monitor.py:128-179, 520-609, 611-721, 723-838, 886-923`
- Modify: `src/rvln/ai/prompts.py:64-151`
- Create: `tests/test_negative_constraints.py` (Tier 1)

This is the largest task. It modifies the diary prompts to include constraint context and the monitor to handle violations.

- [ ] **Step 1: Create Tier 1 test file**

Create `tests/test_negative_constraints.py`:

```python
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
    """Build a monitor with optional negative constraints."""
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
    """Existing code that omits negative_constraints still works."""
    m = LiveDiaryMonitor(
        subgoal="Go forward",
        check_interval=2,
        model="gpt-4o",
    )
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
    """When VLM reports constraint_violated, checkpoint returns force_converge."""
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
    """When VLM reports no violation, checkpoint returns continue."""
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
    m = _make_monitor()  # no constraints
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && conda run -n rvln-sim pytest tests/test_negative_constraints.py -v`
Expected: FAIL with `TypeError: LiveDiaryMonitor.__init__() got an unexpected keyword argument 'negative_constraints'`

- [ ] **Step 3: Add `negative_constraints` parameter to `LiveDiaryMonitor.__init__`**

In `src/rvln/ai/diary_monitor.py`, modify the constructor signature at line 128 to add the parameter after `stall_completion_floor`:

```python
    def __init__(
        self,
        subgoal: str,
        check_interval: int,
        model: str = DEFAULT_VLM_MODEL,
        artifacts_dir: Optional[Path] = None,
        max_corrections: int = 15,
        check_interval_s: Optional[float] = None,
        stall_window: int = 3,
        stall_threshold: float = 0.05,
        stall_completion_floor: float = 0.8,
        negative_constraints: Optional[List[str]] = None,
    ):
```

Add after line 165 (`self._completion_history: List[float] = []`):

```python
        self._negative_constraints: List[str] = list(negative_constraints or [])
```

- [ ] **Step 4: Add `_constraints_block` helper method**

Add to the `LiveDiaryMonitor` class (after `_format_displacement` at line 495):

```python
    def _constraints_block(self) -> str:
        """Build the constraint injection text for VLM prompts."""
        if not self._negative_constraints:
            return ""
        lines = [
            "Active constraints (must be maintained throughout):"
        ]
        for c in self._negative_constraints:
            lines.append(f"  - {c}")
        lines.append("")
        return "\n".join(lines)
```

- [ ] **Step 5: Modify `DIARY_GLOBAL_PROMPT` to include constraints block and `constraint_violated` field**

In `src/rvln/ai/prompts.py`, replace the `DIARY_GLOBAL_PROMPT` (lines 64-95) with:

```python
DIARY_GLOBAL_PROMPT = """\
Subgoal: {subgoal}
{constraints_block}
Previous estimated completion: {prev_completion_pct}
Current displacement from start: [x, y, z, yaw] = {displacement}

Diary of changes observed so far:
{diary}

The grid shows up to the 9 most recent sampled frames (left to right, top to
bottom, in temporal order). If there are more than 9 diary entries, earlier frames
are no longer visible in the grid -- rely on the diary text for that history.

Based on the diary and the grid of sampled frames, respond with EXACTLY ONE JSON
object (no markdown fences):

{{
  "complete": true/false,
  "completion_percentage": 0.0 to 1.0,
  "on_track": true/false,
  "should_stop": true/false,
  "constraint_violated": true/false
}}

- "complete": true ONLY if you are highly confident the subgoal has been fully
  accomplished. Do NOT mark complete for partial progress.
- "completion_percentage": your best estimate of how close the subgoal is to
  completion (0.0 = not started, 1.0 = fully done). NEVER set 1.0 unless you
  are highly confident -- use at most 0.95 when unsure.
- "on_track": true if the drone is making any progress toward the subgoal.
- "should_stop": true only if the drone is actively making things worse (e.g.,
  overshooting, moving away from target). The drone will be stopped and a
  correction issued. Do NOT set true for slow progress.
- "constraint_violated": true if any active constraint listed above has been
  violated or is about to be violated based on the visual evidence and diary.
  If true, also set "should_stop" to true. false if no constraints are listed
  or none have been violated."""
```

- [ ] **Step 6: Modify `DIARY_CONVERGENCE_PROMPT` to include constraints block and `constraint_violated` field**

In `src/rvln/ai/prompts.py`, replace `DIARY_CONVERGENCE_PROMPT` (lines 97-150) with:

```python
DIARY_CONVERGENCE_PROMPT = """\
Subgoal: {subgoal}
{constraints_block}
Previous estimated completion: {prev_completion_pct}
Current displacement from start: [x, y, z, yaw] = {displacement}

Diary of changes observed so far:
{diary}

The drone has stopped moving. The grid shows up to the 9 most recent sampled
frames (left to right, top to bottom, in temporal order). If there are more
than 9 diary entries, earlier frames are no longer visible in the grid -- rely
on the diary text for that history.

Given the diary and the sampled frames, is the subgoal complete? If not, did
the drone stop short or overshoot?

Respond with EXACTLY ONE JSON object (no markdown fences):

{{
  "complete": true/false,
  "completion_percentage": 0.0 to 1.0,
  "diagnosis": "stopped_short" or "overshot" or "complete" or "constraint_violated",
  "corrective_instruction": "..." or null,
  "constraint_violated": true/false
}}

- "complete": true ONLY if you are highly confident the subgoal has been fully
  accomplished. Do NOT mark complete for partial progress. When in doubt, keep
  it false and issue a corrective instruction.
- "completion_percentage": your best estimate of how close the subgoal is to
  completion (0.0 = not started, 1.0 = fully done). NEVER set 1.0 unless you
  are highly confident the subtask is fully complete -- use at most 0.95 if the
  result looks close but you are not certain.
- "diagnosis": "complete" if done, "stopped_short" if the drone needs to keep
  going, "overshot" if the drone went past the goal, "constraint_violated" if
  an active constraint was breached.
- "corrective_instruction": REQUIRED if not complete -- a single-action drone
  command to fix the biggest gap (not compound -- one action per correction).
  If a constraint was violated, the corrective instruction should move the drone
  AWAY from the constraint violation (e.g., "move away from building B").
  null only if complete.
- "constraint_violated": true if any active constraint listed above has been
  violated based on the visual evidence and diary. false if no constraints are
  listed or none have been violated.

  Useful corrective patterns:
    * "Turn toward <landmark>" -- re-orient the drone toward a visible or
      expected landmark so the policy can locate it.
    * "Turn right/left <N> degrees" -- precise yaw adjustment when the target
      is off-screen or partially visible.
    * "Move forward <N> meters" / "Move closer to <landmark>" -- close a gap.
    * "Ascend/Descend <N> meters" -- altitude correction.
    * "Move away from <landmark>" -- retreat from a constraint violation.
  Prefer a turn command when the target is not visible in the latest frame;
  the underlying policy needs to see the target to navigate toward it.

  IMPORTANT -- orientation tolerance: if the subgoal is about turning toward or
  facing a target and the target is already visible in the frame (even if
  off-center), mark the subgoal complete instead of issuing further turn
  corrections. Small yaw offsets are acceptable. Do NOT oscillate between
  left and right turn corrections trying to perfectly center the target."""
```

- [ ] **Step 7: Update all `GLOBAL_PROMPT_TEMPLATE.format()` calls to pass `constraints_block`**

In `src/rvln/ai/diary_monitor.py`, there are 3 places where `GLOBAL_PROMPT_TEMPLATE.format()` is called. Update each:

**`_run_checkpoint` (line 553):**
```python
        prompt_global = GLOBAL_PROMPT_TEMPLATE.format(
            subgoal=self._subgoal,
            constraints_block=self._constraints_block(),
            diary=diary_blob,
            prev_completion_pct=self._last_completion_pct,
            displacement=disp_str,
        )
```

**`_run_checkpoint_async` (line 664):**
```python
        prompt_global = GLOBAL_PROMPT_TEMPLATE.format(
            subgoal=self._subgoal,
            constraints_block=self._constraints_block(),
            diary=diary_blob,
            prev_completion_pct=self._last_completion_pct,
            displacement=disp_str,
        )
```

- [ ] **Step 8: Update all `CONVERGENCE_PROMPT_TEMPLATE.format()` calls to pass `constraints_block`**

There are 3 places where `CONVERGENCE_PROMPT_TEMPLATE.format()` is called. Update each:

**`on_convergence` (line 320):**
```python
        prompt = CONVERGENCE_PROMPT_TEMPLATE.format(
            subgoal=self._subgoal,
            constraints_block=self._constraints_block(),
            diary=diary_blob,
            prev_completion_pct=self._last_completion_pct,
            displacement=disp_str,
        )
```

**`_run_convergence_async` (line 743):**
```python
        prompt = CONVERGENCE_PROMPT_TEMPLATE.format(
            subgoal=self._subgoal,
            constraints_block=self._constraints_block(),
            diary=diary_blob,
            prev_completion_pct=self._last_completion_pct,
            displacement=disp_str,
        )
```

- [ ] **Step 9: Add constraint violation handling to `_parse_global_response`**

In `src/rvln/ai/diary_monitor.py`, in `_parse_global_response` (line 886), add this check immediately after the `pct` calculation (after line 897) and before the `complete` check (line 899):

```python
        if self._negative_constraints and parsed.get("constraint_violated", False):
            return DiaryCheckResult(
                action="force_converge",
                new_instruction="",
                reasoning=f"Constraint violated, stopping for correction. Raw: {response}",
                diary_entry=diary_entry,
                completion_pct=pct,
            )
```

- [ ] **Step 10: Run Tier 1 tests**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && conda run -n rvln-sim pytest tests/test_negative_constraints.py -v`
Expected: All 7 tests PASS

- [ ] **Step 11: Run all existing tests for regression**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && conda run -n rvln-sim pytest tests/ -v -m "not tier2 and not tier3"`
Expected: All PASS. Existing stall detection tests use `_make_monitor_with_history` which constructs monitors via `__new__` (bypassing `__init__`), so the new parameter doesn't affect them. Tests that call `__init__` directly without `negative_constraints` use the default `None`.

- [ ] **Step 12: Commit**

```bash
git add src/rvln/ai/diary_monitor.py src/rvln/ai/prompts.py tests/test_negative_constraints.py
git commit -m "feat: add negative constraint injection and violation handling to diary monitor"
```

---

### Task 5: Wire Constraints Through the Integration Runner

**Files:**
- Modify: `scripts/run_integration.py:190-543, 609-673`

- [ ] **Step 1: Add `negative_constraints` parameter to `_run_subgoal`**

In `scripts/run_integration.py`, add `negative_constraints: Optional[List[str]] = None` to the `_run_subgoal` signature. Insert after `max_seconds: Optional[float] = None` (line 209):

```python
    negative_constraints: Optional[List[str]] = None,
```

- [ ] **Step 2: Pass constraints to `LiveDiaryMonitor` constructor**

In `_run_subgoal`, modify the `LiveDiaryMonitor` constructor call (line 250):

```python
    monitor = LiveDiaryMonitor(
        subgoal=subgoal_nl,
        check_interval=check_interval,
        model=monitor_model,
        artifacts_dir=diary_artifacts,
        max_corrections=max_corrections,
        check_interval_s=check_interval_s,
        negative_constraints=negative_constraints,
    )
```

- [ ] **Step 3: Add `constraint_violated` to the return dict**

In `_run_subgoal`, modify the return dict (line 533) to include `constraint_violated`. Replace:

```python
    return {
        "subgoal": subgoal_nl,
        "converted_instruction": converted_instruction,
        "total_steps": total_steps,
        "stop_reason": stop_reason,
        "corrections_used": monitor.corrections_used,
        "last_completion_pct": monitor.last_completion_pct,
        "peak_completion": monitor.peak_completion,
        "vlm_calls": monitor.vlm_calls,
        "next_origin": [next_origin_x, next_origin_y, next_origin_z, next_origin_yaw],
    }
```

with:

```python
    return {
        "subgoal": subgoal_nl,
        "converted_instruction": converted_instruction,
        "total_steps": total_steps,
        "stop_reason": stop_reason,
        "corrections_used": monitor.corrections_used,
        "last_completion_pct": monitor.last_completion_pct,
        "peak_completion": monitor.peak_completion,
        "vlm_calls": monitor.vlm_calls,
        "negative_constraints": negative_constraints or [],
        "next_origin": [next_origin_x, next_origin_y, next_origin_z, next_origin_yaw],
    }
```

Also add the same field to the OOD early-return dict (line 238):

```python
        "negative_constraints": negative_constraints or [],
```

- [ ] **Step 4: Extract constraints from planner in `run_integrated_control_loop`**

In `run_integrated_control_loop`, modify the subgoal loop (starting at line 613). Replace:

```python
    while current_subgoal is not None:
        subgoal_index += 1
        safe_name = _sanitize_name(current_subgoal)
        subgoal_dir = run_dir / f"subgoal_{subgoal_index:02d}_{safe_name}"

        logger.info(
            "--- Subgoal %d: '%s' ---", subgoal_index, current_subgoal,
        )

        try:
            subgoal_result = _run_subgoal(
                env=env,
                batch=batch,
                server_url=server_url,
                subgoal_nl=current_subgoal,
                monitor_model=monitor_model,
                check_interval=check_interval,
                max_steps=max_steps_per_subgoal,
                max_corrections=max_corrections,
                origin_x=origin_x,
                origin_y=origin_y,
                origin_z=origin_z,
                origin_yaw=origin_yaw,
                drone_cam_id=drone_cam_id,
                frames_dir=frames_dir,
                subgoal_dir=subgoal_dir,
                frame_offset=total_frame_count,
                trajectory_log=trajectory_log,
                check_interval_s=check_interval_s,
                max_seconds=max_seconds,
            )
```

with:

```python
    while current_subgoal is not None:
        subgoal_index += 1
        safe_name = _sanitize_name(current_subgoal)
        subgoal_dir = run_dir / f"subgoal_{subgoal_index:02d}_{safe_name}"

        active_constraints = planner.get_active_constraints()
        if active_constraints:
            logger.info(
                "Active constraints for subgoal %d: %s",
                subgoal_index, active_constraints,
            )

        logger.info(
            "--- Subgoal %d: '%s' ---", subgoal_index, current_subgoal,
        )

        try:
            subgoal_result = _run_subgoal(
                env=env,
                batch=batch,
                server_url=server_url,
                subgoal_nl=current_subgoal,
                monitor_model=monitor_model,
                check_interval=check_interval,
                max_steps=max_steps_per_subgoal,
                max_corrections=max_corrections,
                origin_x=origin_x,
                origin_y=origin_y,
                origin_z=origin_z,
                origin_yaw=origin_yaw,
                drone_cam_id=drone_cam_id,
                frames_dir=frames_dir,
                subgoal_dir=subgoal_dir,
                frame_offset=total_frame_count,
                trajectory_log=trajectory_log,
                check_interval_s=check_interval_s,
                max_seconds=max_seconds,
                negative_constraints=active_constraints,
            )
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_integration.py
git commit -m "feat: wire negative constraints from planner through runner to monitor"
```

---

### Task 6: Tier 3 VLM Prompt Smoke Tests

**Files:**
- Create: `tests/test_vlm_constraint_prompts.py`

- [ ] **Step 1: Create Tier 3 test file**

Create `tests/test_vlm_constraint_prompts.py`:

```python
"""
Tier 3 VLM prompt smoke tests for constraint-aware diary monitoring.

These tests make real VLM API calls with synthetic diary context to verify
the VLM correctly returns constraint_violated in its JSON response.

Run: conda run -n rvln-sim pytest tests/test_vlm_constraint_prompts.py -v -m "tier3"
"""
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest

needs_api = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="No OPENAI_API_KEY",
)
tier3 = pytest.mark.tier3


def _render_global_prompt(subgoal, constraints, diary, displacement, prev_pct):
    from rvln.ai.prompts import DIARY_GLOBAL_PROMPT

    constraints_block = ""
    if constraints:
        lines = ["Active constraints (must be maintained throughout):"]
        for c in constraints:
            lines.append(f"  - {c}")
        lines.append("")
        constraints_block = "\n".join(lines)

    return DIARY_GLOBAL_PROMPT.format(
        subgoal=subgoal,
        constraints_block=constraints_block,
        diary=diary,
        prev_completion_pct=prev_pct,
        displacement=displacement,
    )


@needs_api
@tier3
def test_vlm_returns_constraint_violated_field():
    """VLM should include constraint_violated in JSON when constraints are present."""
    from rvln.ai.prompts import DIARY_SYSTEM_PROMPT
    from rvln.ai.utils.llm_providers import LLMFactory

    llm = LLMFactory.create("openai", model="gpt-4o")

    prompt = _render_global_prompt(
        subgoal="Approach the tree",
        constraints=["stay away from building B"],
        diary="Steps 0-20: Drone moved forward, building B visible on the right.\n"
              "Steps 20-40: Drone turned slightly toward building B.\n"
              "Checkpoint 40: completion = 0.30",
        displacement="[x: 3.50 m, y: 1.20 m, z: 0.00 m, yaw: 15.0 deg]",
        prev_pct=0.30,
    )

    messages = [
        {"role": "system", "content": DIARY_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    response = llm.make_request(messages, temperature=0.0)
    text = response.strip()

    start = text.find("{")
    end = text.rfind("}")
    assert start != -1 and end != -1, f"No JSON in VLM response: {text}"
    parsed = json.loads(text[start:end + 1])

    assert "constraint_violated" in parsed, (
        f"VLM response missing 'constraint_violated' field. Got: {parsed}"
    )
    assert isinstance(parsed["constraint_violated"], bool), (
        f"'constraint_violated' should be bool, got: {type(parsed['constraint_violated'])}"
    )


@needs_api
@tier3
def test_vlm_no_violation_without_constraints():
    """When no constraints are listed, constraint_violated should be false or absent."""
    from rvln.ai.prompts import DIARY_SYSTEM_PROMPT
    from rvln.ai.utils.llm_providers import LLMFactory

    llm = LLMFactory.create("openai", model="gpt-4o")

    prompt = _render_global_prompt(
        subgoal="Approach the tree",
        constraints=[],
        diary="Steps 0-20: Drone moved forward toward the tree.\n"
              "Checkpoint 20: completion = 0.50",
        displacement="[x: 5.00 m, y: 0.00 m, z: 0.00 m, yaw: 0.0 deg]",
        prev_pct=0.50,
    )

    messages = [
        {"role": "system", "content": DIARY_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    response = llm.make_request(messages, temperature=0.0)
    text = response.strip()
    start = text.find("{")
    end = text.rfind("}")
    assert start != -1 and end != -1, f"No JSON in VLM response: {text}"
    parsed = json.loads(text[start:end + 1])

    violated = parsed.get("constraint_violated", False)
    assert violated is False, (
        f"Expected no constraint violation with empty constraints, got: {parsed}"
    )
```

- [ ] **Step 2: Run Tier 3 tests**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && conda run -n rvln-sim pytest tests/test_vlm_constraint_prompts.py -v -m "tier3"`
Expected: Both tests PASS. If `test_vlm_returns_constraint_violated_field` fails (VLM doesn't return the field), iterate on the prompt wording in Task 4 Steps 5-6.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vlm_constraint_prompts.py
git commit -m "test: add Tier 3 VLM prompt smoke tests for constraint_violated field"
```

---

### Task 7: End-to-End Integration Tests (No Simulator)

**Files:**
- Modify: `tests/test_negative_constraints.py`

- [ ] **Step 1: Add full-flow integration tests**

Append to `tests/test_negative_constraints.py`:

```python
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

    # Planner classifies pi_3 as constraint
    assert "pi_3" in planner.constraint_predicates
    assert "pi_1" not in planner.constraint_predicates

    # Active constraints during pi_1
    current = planner.get_next_predicate()
    assert current == "Go to tree A"
    constraints = planner.get_active_constraints()
    assert constraints == ["Flying over building C"]

    # Monitor receives constraints
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

    # Violation detected in parsed response
    parsed = monitor._parse_json_response(
        '{"complete": false, "completion_percentage": 0.3, "on_track": false, '
        '"should_stop": true, "constraint_violated": true}'
    )
    assert parsed["constraint_violated"] is True

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

    # pi_3 is classified as constraint
    assert "pi_3" in planner.constraint_predicates

    # During pi_1: constraint active
    current = planner.get_next_predicate()
    c1 = planner.get_active_constraints()
    assert "Near the red car" in c1

    # During pi_2: constraint still active
    planner.advance_state(current)
    current = planner.get_next_predicate()
    c2 = planner.get_active_constraints()
    assert "Near the red car" in c2

    # After pi_2: constraint released
    planner.advance_state(current)
    c3 = planner.get_active_constraints()
    assert c3 == []
```

- [ ] **Step 2: Run all Tier 1 tests**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && conda run -n rvln-sim pytest tests/ -v -m "not tier2 and not tier3"`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_negative_constraints.py
git commit -m "test: add end-to-end integration tests for negative constraint flow"
```

---

## Test Execution Summary

After completing all tasks, run the full test suite in order:

```bash
# Tier 1 (free, fast, always):
conda run -n rvln-sim pytest tests/ -v -m "not tier2 and not tier3"

# Tier 2 (pennies, after prompt changes):
conda run -n rvln-sim pytest tests/test_prompt_constraints.py -v -m "tier2"

# Tier 3 (pennies, after prompt changes):
conda run -n rvln-sim pytest tests/test_vlm_constraint_prompts.py -v -m "tier3"
```

All three tiers must pass before running full-system experiments with the simulator.

| Task | Tier 1 | Tier 2 | Tier 3 | What it catches |
|------|--------|--------|--------|-----------------|
| 1 | Automaton classification + active detection | - | - | BDD query bugs, state tracking bugs |
| 2 | - | LLM prompt generates G(!pi_X) | - | Prompt engineering failures |
| 3 | G operator parsing | - | - | Human-readable output broken |
| 4 | Injection + violation handling | - | - | Monitor ignores constraints, bad JSON |
| 5 | (runner code, no unit test) | - | - | Constraints not wired through |
| 6 | - | - | VLM returns constraint_violated | VLM prompt wording bugs |
| 7 | Full planner+monitor flow | - | - | Integration bugs between components |
