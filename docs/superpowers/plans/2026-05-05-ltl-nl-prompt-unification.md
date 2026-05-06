# LTL-NL Prompt Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the dual constraint-aware vs sequential LTL-NL planner paths into a single sequential-only path. Each "until/while/for-N-meters" clause stays inside the predicate it scopes; the LTL formula is a strict sequential state machine. Remove constraint classification, active-constraints injection, `_WITH_CONSTRAINTS` prompt variants, the `use_constraints` toggle, related tests, and the constrained-task track from `experimental_design.txt`.

**Architecture:** The current `SequentialLTLPlanner` body becomes the new `LTLSymbolicPlanner` body (preserving the class name since the planner is still symbolic in the Spot-automaton sense). The constraint-aware planner, prompts, and runtime plumbing are deleted outright. No deprecation phase. The `cached_formulas/` directory is wiped because the prompt-version hash changes.

**Tech Stack:** Python 3, Spot (LTL → automaton), pytest, conda env `rvln-sim`.

---

## Architectural Notes for Implementers

- The Spot library is only available inside the `rvln-sim` conda env. Run tests with `conda run -n rvln-sim pytest <path>`.
- The new `LTLSymbolicPlanner` is the current `SequentialLTLPlanner` body PLUS the last-goal-via-sink-edge fallback fix from the current `LTLSymbolicPlanner.get_next_predicate`. Do not regress that bug.
- After this refactor, the input "turn left until you see the tree, then go toward the tree, then turn right until you see the building, and go to the building" must yield 4 predicates, each with its conditional clause preserved in the description, and the formula `F pi_4 & (!pi_4 U pi_3) & (!pi_3 U pi_2) & (!pi_2 U pi_1)`.
- `docs/` is gitignored in this repo. When committing files in `docs/`, use `git add -f`.
- **Never use em dashes (—)** in any text you write (CLAUDE.md rule). Use commas, parentheses, colons, or separate sentences.
- **Do not add `Co-Authored-By` lines** to commits (user's global rule).

---

### Task 1: Wipe `cached_formulas/`

**Files:**
- Delete: every regular file under `cached_formulas/`
- Keep: the directory itself (so `FORMULA_CACHE_DIR` from `paths.py` continues to resolve)

- [ ] **Step 1: Confirm directory contents**

Run: `ls cached_formulas/ | head -20 && ls cached_formulas/ | wc -l`
Expected: a list of `.json` files and a count.

- [ ] **Step 2: Delete all files inside the directory**

Run: `find cached_formulas/ -mindepth 1 -type f -delete`

- [ ] **Step 3: Verify the directory is empty**

Run: `ls cached_formulas/`
Expected: empty output.

- [ ] **Step 4: Commit**

```bash
git add -A cached_formulas/
git commit -m "chore: wipe formula cache ahead of LTL-NL prompt unification

Old entries are keyed by a prompt-version hash that no longer matches
the unified sequential prompt. Wiping prevents stale orphans."
```

---

### Task 2: Replace `src/rvln/ai/ltl_planner.py` with sequential body + bug fix; delete `sequential_ltl_planner.py`

**Files:**
- Modify (full rewrite): `src/rvln/ai/ltl_planner.py`
- Delete: `src/rvln/ai/sequential_ltl_planner.py`

The new `LTLSymbolicPlanner` is the current `SequentialLTLPlanner` (in `src/rvln/ai/sequential_ltl_planner.py`) with one addition: the last-goal-via-sink-edge fallback in `get_next_predicate`, ported from the current `src/rvln/ai/ltl_planner.py:294-312`.

- [ ] **Step 1: Rewrite `src/rvln/ai/ltl_planner.py` with the unified body**

Replace the entire file contents with:

```python
"""
LTL Symbolic Planner: parses natural language to LTL-NL via LLM, then uses Spot
to manage the automaton state and determine the next short-horizon subgoal.

Every predicate is a goal whose completion criterion is its full natural-
language description (including any "until ...", "while ...", "for N meters",
or "without ..." clauses). The Spot automaton drives sequential subgoal
advancement.
"""

from typing import Optional

try:
    import spot
except ImportError:
    spot = None

from .llm_interface import LLMUserInterface


def _predicate_key_to_index(key: str) -> int:
    """Parse predicate key to integer index (e.g. 'pi_1' -> 1, 'p1' -> 1)."""
    key = key.strip()
    if key.startswith("pi_"):
        return int(key[3:].strip())
    if key.startswith("p") and key[1:].strip().isdigit():
        return int(key[1:].strip())
    raise ValueError(f"Predicate key must be 'pi_N' or 'pN': got '{key}'")


def _normalize_pi_predicates(raw: dict) -> dict:
    """Normalize pi_predicates to canonical keys pi_1, pi_2, ... (ordered by index)."""
    if not raw or not isinstance(raw, dict):
        return {}
    result = {}
    for k, v in raw.items():
        if not isinstance(v, str):
            continue
        try:
            idx = _predicate_key_to_index(k)
        except (ValueError, TypeError):
            continue
        result[f"pi_{idx}"] = v.strip()
    return dict(sorted(result.items(), key=lambda x: _predicate_key_to_index(x[0])))


class LTLSymbolicPlanner:
    """
    Integrates LLMUserInterface to parse NL to LTL-NL, then uses Spot
    to manage the automaton state and determine the next action.
    """

    def __init__(self, llm_interface: LLMUserInterface):
        if spot is None:
            raise ImportError(
                "The 'spot' library is required for LTLSymbolicPlanner. "
                "Install via the rvln-sim conda environment."
            )
        self.llm_interface = llm_interface
        self.current_automaton_state = 0
        self.automaton = None
        self._sink_state: Optional[int] = None
        self.pi_map = {}
        self._last_returned_predicate_key: Optional[str] = None
        self.finished = False

    def plan_from_natural_language(self, instruction: str) -> None:
        """
        1. Query LLM for LTL formula.
        2. Convert LTL string to Spot Automaton.
        """
        if not instruction or not isinstance(instruction, str):
            raise ValueError("Instruction must be a non-empty string.")
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("Instruction must be a non-empty string.")

        print(f"[LTL Planner] Processing instruction: '{instruction}'")
        _ = self.llm_interface.make_natural_language_request(instruction)
        data = self.llm_interface.ltl_nl_formula

        if not data or not isinstance(data, dict):
            raise ValueError("LLM could not generate a valid LTL formula (empty or non-dict response).")
        if "ltl_nl_formula" not in data or "pi_predicates" not in data:
            raise ValueError(
                "LLM response must contain 'ltl_nl_formula' and 'pi_predicates'. "
                f"Keys received: {list(data.keys()) if isinstance(data, dict) else 'N/A'}."
            )

        raw_formula = data["ltl_nl_formula"]
        if not isinstance(raw_formula, str) or not raw_formula.strip():
            raise ValueError("'ltl_nl_formula' must be a non-empty string.")

        self.pi_map = _normalize_pi_predicates(data["pi_predicates"])
        if not self.pi_map:
            raise ValueError(
                "LLM returned no valid predicates (pi_predicates empty or not parseable). "
                "Use valid robot instructions."
            )
        self._last_returned_predicate_key = None

        print(f"[LTL Planner] Generated Formula: {raw_formula}")
        print(f"[LTL Planner] Predicates: {self.pi_map}")

        spot_formula = raw_formula.replace("pi_", "p")
        try:
            self.automaton = spot.translate(spot_formula, "monitor", "det")
        except Exception as e:
            raise ValueError(
                f"Spot could not translate LTL formula '{spot_formula}': {e}. "
                "Formula may be invalid or use unsupported operators."
            ) from e
        self._add_sink_state()
        self.current_automaton_state = self.automaton.get_init_state_number()
        self.finished = False

    def _add_sink_state(self) -> None:
        """Add a sink state and connect dead-end states to it.

        The edge condition is the BDD for the *last* predicate in pi_map
        order, so get_next_predicate() returns that predicate (not an
        earlier one) when in such a state.
        """
        self._sink_state = None
        if not self.pi_map or self.automaton is None:
            return
        try:
            n = self.automaton.num_states()
            self.automaton.new_states(1)
            self._sink_state = n
            last_key = list(self.pi_map.keys())[-1]
            last_p_idx = _predicate_key_to_index(last_key)
            bdd_sink_cond = self._get_bdd_for_single_task(last_p_idx)
            for s in range(n):
                has_outgoing_to_other = any(
                    edge.dst != s for edge in self.automaton.out(s)
                )
                if not has_outgoing_to_other:
                    self.automaton.new_edge(s, self._sink_state, bdd_sink_cond)
        except (ValueError, RuntimeError, AttributeError) as e:
            print(f"[LTL Planner] Could not add sink state: {e}. Continuing without sink.")
            self._sink_state = None

    def _get_bdd_for_single_task(self, active_p_idx: int):
        """BDD: the given predicate TRUE, all other known predicates FALSE."""
        if not self.pi_map or self.automaton is None:
            raise ValueError("Cannot build BDD: no predicates or automaton.")
        clauses = []
        for key in self.pi_map.keys():
            idx = _predicate_key_to_index(key)
            if idx == active_p_idx:
                clauses.append(f"p{idx}")
            else:
                clauses.append(f"!p{idx}")
        formula_str = " & ".join(clauses)
        f = spot.formula(formula_str)
        return spot.formula_to_bdd(f, self.automaton.get_dict(), self.automaton)

    def get_next_predicate(self) -> Optional[str]:
        """Find the next task by testing which predicate allows leaving the current state."""
        if self.finished:
            return None
        if not self.pi_map or self.automaton is None:
            return None
        if self._sink_state is not None and self.current_automaton_state == self._sink_state:
            self.finished = True
            return None

        bdd_false = spot.formula_to_bdd(
            spot.formula("0"), self.automaton.get_dict(), self.automaton
        )

        for key in self.pi_map.keys():
            p_idx = _predicate_key_to_index(key)
            try:
                test_world_bdd = self._get_bdd_for_single_task(p_idx)
            except ValueError:
                continue
            for edge in self.automaton.out(self.current_automaton_state):
                if edge.dst == self.current_automaton_state:
                    continue
                if edge.dst == self._sink_state:
                    continue
                if (test_world_bdd & edge.cond) != bdd_false:
                    self._last_returned_predicate_key = key
                    return self.pi_map[key]

        # Fallback: on some Spot versions (or after postprocessing) the
        # monitor automaton has fewer states, so the last goal's only
        # outgoing edge leads to the sink. The loop above skips sink edges,
        # which silently drops the final goal. Check sink edges for any
        # not-yet-returned predicate (skip _last_returned_predicate_key
        # to avoid re-returning an already-completed goal at a true dead-end).
        if self._sink_state is not None:
            for key in self.pi_map.keys():
                if key == self._last_returned_predicate_key:
                    continue
                p_idx = _predicate_key_to_index(key)
                try:
                    test_world_bdd = self._get_bdd_for_single_task(p_idx)
                except ValueError:
                    continue
                for edge in self.automaton.out(self.current_automaton_state):
                    if edge.dst == self.current_automaton_state:
                        continue
                    if edge.dst != self._sink_state:
                        continue
                    if (test_world_bdd & edge.cond) != bdd_false:
                        self._last_returned_predicate_key = key
                        return self.pi_map[key]

        # Sequence fallback: no edge fired; use state as index only if in range
        pred_keys = list(self.pi_map.keys())
        pred_values = list(self.pi_map.values())
        n_states = self.automaton.num_states()
        if 0 <= self.current_automaton_state < len(pred_values) and self.current_automaton_state < n_states:
            self._last_returned_predicate_key = pred_keys[self.current_automaton_state]
            return pred_values[self.current_automaton_state]

        if len(self.pi_map) == 1:
            k = next(iter(self.pi_map.keys()))
            self._last_returned_predicate_key = k
            return self.pi_map[k]
        print("[LTL Planner] No tasks trigger a state change. Mission Complete.")
        self.finished = True
        return None

    def advance_state(self, finished_task_nl: str) -> None:
        """Update automaton state when a subgoal is confirmed.

        Uses the predicate key from the last get_next_predicate() call so
        duplicate descriptions (e.g. 'turn 90 degrees' three times) work.
        """
        if not self.pi_map or self.automaton is None:
            return
        pi_key = self._last_returned_predicate_key
        if pi_key is None:
            print("[LTL Planner] Warning: no current predicate key (get_next_predicate not called or returned None).")
            return
        p_idx = _predicate_key_to_index(pi_key)
        try:
            current_world_bdd = self._get_bdd_for_single_task(p_idx)
        except ValueError:
            return
        bdd_false = spot.formula_to_bdd(
            spot.formula("0"), self.automaton.get_dict(), self.automaton
        )

        found_next = False
        for edge in self.automaton.out(self.current_automaton_state):
            if edge.dst == self.current_automaton_state:
                continue
            if (current_world_bdd & edge.cond) != bdd_false:
                print(f"[LTL Planner] Task '{pi_key}' satisfied edge condition.")
                print(
                    f"[LTL Planner] Transitioning State: {self.current_automaton_state} -> {edge.dst}"
                )
                self.current_automaton_state = edge.dst
                found_next = True
                break

        if not found_next:
            if self._sink_state is not None:
                self.current_automaton_state = self._sink_state
                print(
                    f"[LTL Planner] Task '{finished_task_nl}' completed. "
                    "Transitioning to sink (mission complete)."
                )
            else:
                self.finished = True
                print(
                    f"[LTL Planner] Task '{finished_task_nl}' completed but no outgoing edge; "
                    "marking mission complete."
                )
```

- [ ] **Step 2: Delete the now-redundant sequential planner module**

Run: `git rm src/rvln/ai/sequential_ltl_planner.py`

- [ ] **Step 3: Verify removed symbols are gone**

Run: `grep -n "ConstraintInfo\|constraint_predicates\|get_active_constraints\|_classify_predicates\|_active_positive_constraint_indices\|_get_bdd_constraint_violation\|_get_bdd_goal_check" src/rvln/ai/ltl_planner.py`
Expected: empty output.

- [ ] **Step 4: Sanity-import (skip if pytest not available outside `rvln-sim`)**

Run (inside `rvln-sim` conda env if Spot is needed): `python -c "from rvln.ai.ltl_planner import LTLSymbolicPlanner"`
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add src/rvln/ai/ltl_planner.py src/rvln/ai/sequential_ltl_planner.py
git commit -m "ai/ltl_planner: unify on sequential planner; drop constraint classifier

Replaces LTLSymbolicPlanner body with the SequentialLTLPlanner
implementation plus the last-goal-via-sink-edge fallback (the BUG FIX
branch from the previous constraint-aware planner). Deletes
sequential_ltl_planner.py. Removes ConstraintInfo, constraint_predicates,
get_active_constraints, _classify_predicates and related BDD helpers."
```

---

### Task 3: Edit `src/rvln/ai/prompts.py` (delete WITH_CONSTRAINTS variants, rename SEQUENTIAL → default)

**Files:**
- Modify: `src/rvln/ai/prompts.py`

The current file has both constraint-aware and sequential planner prompts plus six `_WITH_CONSTRAINTS` monitor prompts. After this task, only the sequential planner prompts and the non-constraint monitor prompts remain.

- [ ] **Step 1: Delete the constraint-aware planner prompts**

Delete the constants (and any header comments dedicated to them):
- `LTL_NL_SYSTEM_PROMPT` (the constraint-aware version, currently at lines 359-503)
- `LTL_NL_EXAMPLES_PROMPT` (the constraint-aware version, currently at lines 505-640)
- The comment block "Sequential-only LTL planner prompts (pre-constraint behavior)" comment block (currently lines 656-663) — replace with a plain `# LTL planner prompts` section header similar to the other section headers in the file.

- [ ] **Step 2: Rename the sequential planner prompts to drop the suffix**

Replace constant names:
- `LTL_NL_SYSTEM_PROMPT_SEQUENTIAL` -> `LTL_NL_SYSTEM_PROMPT`
- `LTL_NL_EXAMPLES_PROMPT_SEQUENTIAL` -> `LTL_NL_EXAMPLES_PROMPT`

Their string contents stay byte-for-byte identical.

- [ ] **Step 3: Delete the four `_WITH_CONSTRAINTS` monitor prompt constants**

- `DIARY_GLOBAL_PROMPT_WITH_CONSTRAINTS`
- `DIARY_CONVERGENCE_PROMPT_WITH_CONSTRAINTS`
- `TEXT_ONLY_GLOBAL_PROMPT_WITH_CONSTRAINTS`
- `TEXT_ONLY_CONVERGENCE_PROMPT_WITH_CONSTRAINTS`

- [ ] **Step 4: Strip `{constraints_block}` placeholder from two non-constraint text-only prompts**

In `TEXT_ONLY_GLOBAL_PROMPT` (currently at line ~992), remove the literal line containing `{constraints_block}` (currently line 994). The result starts:

```
Subgoal: {subgoal}
Previous estimated completion: {prev_completion_pct}
```

In `TEXT_ONLY_CONVERGENCE_PROMPT` (currently at line ~1062), remove the literal line containing `{constraints_block}` (currently line 1064). The result starts:

```
Subgoal: {subgoal}
Previous estimated completion: {prev_completion_pct}
Current displacement from start: [x, y, z, yaw] = {displacement}
{stop_reasoning_block}
```

- [ ] **Step 5: Verify deletions and renames**

Run: `grep -n "LTL_NL_SYSTEM_PROMPT_SEQUENTIAL\|LTL_NL_EXAMPLES_PROMPT_SEQUENTIAL\|WITH_CONSTRAINTS\|constraints_block" src/rvln/ai/prompts.py`
Expected: empty output.

Run: `grep -n "^LTL_NL_SYSTEM_PROMPT\b\|^LTL_NL_EXAMPLES_PROMPT\b" src/rvln/ai/prompts.py`
Expected: each constant shows up exactly once.

- [ ] **Step 6: Commit**

```bash
git add src/rvln/ai/prompts.py
git commit -m "ai/prompts: collapse to single sequential LTL prompt; drop WITH_CONSTRAINTS

Deletes LTL_NL_SYSTEM_PROMPT/EXAMPLES_PROMPT constraint-aware variants
and the four _WITH_CONSTRAINTS monitor prompts. Renames
LTL_NL_SYSTEM_PROMPT_SEQUENTIAL -> LTL_NL_SYSTEM_PROMPT and the
matching examples prompt. Strips {constraints_block} placeholders from
TEXT_ONLY_GLOBAL_PROMPT and TEXT_ONLY_CONVERGENCE_PROMPT."
```

---

### Task 4: Update `src/rvln/ai/llm_interface.py` (drop `use_constraints` branch)

**Files:**
- Modify: `src/rvln/ai/llm_interface.py`

- [ ] **Step 1: Remove the dropped imports**

In the import block at lines 25-33, delete `LTL_NL_SYSTEM_PROMPT_SEQUENTIAL` and `LTL_NL_EXAMPLES_PROMPT_SEQUENTIAL` (these no longer exist). Keep `LTL_NL_SYSTEM_PROMPT`, `LTL_NL_EXAMPLES_PROMPT`, `LTL_NL_RESTATED_TASK_PROMPT`, `LTL_NL_CHECK_PREDICATES_PROMPT`, `LTL_NL_CHECK_SEMANTICS_PROMPT`.

- [ ] **Step 2: Drop `use_constraints` from `__init__` signature and body**

Replace the constructor (currently lines 142-166) with:

```python
def __init__(self, model: str = DEFAULT_LLM_MODEL):
    self._model = model
    self._base_llm = LLMFactory.create(model=model, rate_limit_seconds=0.0)

    self._initial_context = [
        {"role": "system", "content": LTL_NL_SYSTEM_PROMPT},
        {"role": "system", "content": LTL_NL_EXAMPLES_PROMPT},
        {"role": "system", "content": LTL_NL_RESTATED_TASK_PROMPT},
    ]
    self._prompt_version = _prompt_version_for(
        (LTL_NL_SYSTEM_PROMPT, LTL_NL_EXAMPLES_PROMPT, LTL_NL_RESTATED_TASK_PROMPT)
    )

    self._history = list(self._initial_context)
    self.ltl_nl_formula = {}
    self._ltl_is_confirmed = False
    self.llm_call_records: List[Dict[str, Any]] = []
```

- [ ] **Step 3: Verify**

Run: `grep -n "use_constraints\|_use_constraints\|SEQUENTIAL" src/rvln/ai/llm_interface.py`
Expected: empty output.

Run: `python -c "from rvln.ai.llm_interface import LLMUserInterface"` (in any env that has the project on path; Spot not required).
Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add src/rvln/ai/llm_interface.py
git commit -m "ai/llm_interface: drop use_constraints toggle, unify on single prompt

Removes the dual-prompt branch and the use_constraints constructor arg.
Always uses LTL_NL_SYSTEM_PROMPT + LTL_NL_EXAMPLES_PROMPT +
LTL_NL_RESTATED_TASK_PROMPT."
```

---

### Task 5: Update `src/rvln/ai/goal_adherence_monitor.py` (drop constraint plumbing)

**Files:**
- Modify: `src/rvln/ai/goal_adherence_monitor.py`

- [ ] **Step 1: Remove `_WITH_CONSTRAINTS` imports and `_TEMPLATE_CONSTRAINTS` aliases**

In the import block (currently lines 73-81), delete the four lines that import `_WITH_CONSTRAINTS` prompts as `_TEMPLATE_CONSTRAINTS` aliases:
- `DIARY_GLOBAL_PROMPT_WITH_CONSTRAINTS as GLOBAL_PROMPT_TEMPLATE_CONSTRAINTS`
- `DIARY_CONVERGENCE_PROMPT_WITH_CONSTRAINTS as CONVERGENCE_PROMPT_TEMPLATE_CONSTRAINTS`
- `TEXT_ONLY_GLOBAL_PROMPT_WITH_CONSTRAINTS as TEXT_ONLY_GLOBAL_PROMPT_TEMPLATE_CONSTRAINTS`
- `TEXT_ONLY_CONVERGENCE_PROMPT_WITH_CONSTRAINTS as TEXT_ONLY_CONVERGENCE_PROMPT_TEMPLATE_CONSTRAINTS`

Keep the four non-constraint imports (`DIARY_GLOBAL_PROMPT`, `DIARY_CONVERGENCE_PROMPT`, `TEXT_ONLY_GLOBAL_PROMPT`, `TEXT_ONLY_CONVERGENCE_PROMPT`).

- [ ] **Step 2: Drop `constraints` and `negative_constraints` constructor params and `self._constraints`**

In `GoalAdherenceMonitor.__init__` (currently around line 176-188), remove:
- `constraints: Optional[List[Any]] = None,` parameter
- `negative_constraints: Optional[List[str]] = None,` parameter
- `self._constraints: List[Any] = list(constraints or negative_constraints or [])`

- [ ] **Step 3: Delete `_constraints_block` method**

Currently at line 861-872. Delete the entire method.

- [ ] **Step 4: Collapse the four prompt-build sites to their non-constraint branch**

Each site currently has the shape:
```python
if self._constraints:
    prompt = TEMPLATE_CONSTRAINTS.format(..., constraints_block=self._constraints_block())
else:
    prompt = TEMPLATE.format(..., constraints_block="")
```

Replace each with:
```python
prompt = TEMPLATE.format(...)
```

The `constraints_block=...` keyword argument must be removed from the `.format(...)` call. The four sites are approximately at lines 579-595 (text-only convergence), 876-887 (diary global), 931-944 (diary convergence), and 1158-1175 (text-only global).

After Task 3 the four non-constraint templates have no `{constraints_block}` placeholder, so passing it is unnecessary.

- [ ] **Step 5: Verify**

Run: `grep -n "constraints\|negative_constraints" src/rvln/ai/goal_adherence_monitor.py`
Expected: empty output.

Run: `python -c "from rvln.ai.goal_adherence_monitor import GoalAdherenceMonitor"`
Expected: no error.

- [ ] **Step 6: Commit**

```bash
git add src/rvln/ai/goal_adherence_monitor.py
git commit -m "ai/goal_adherence_monitor: drop constraints injection plumbing

Removes constraints / negative_constraints constructor params, the
_constraints_block helper, the _TEMPLATE_CONSTRAINTS prompt aliases,
and the per-site constraint conditional. The four non-constraint
prompt templates are now used unconditionally."
```

---

### Task 6: Update `src/rvln/eval/subgoal_runner.py` (drop `use_constraints` and constraint args)

**Files:**
- Modify: `src/rvln/eval/subgoal_runner.py`

- [ ] **Step 1: Drop `use_constraints` field from `SubgoalConfig`**

In the dataclass (currently at lines 36-46), remove `use_constraints: bool = True`.

- [ ] **Step 2: Delete the four ablation `_CONSTRAINTS` prompt variants**

Delete:
- `GRID_ONLY_GLOBAL_PROMPT_CONSTRAINTS` (lines 84-115)
- `GRID_ONLY_CONVERGENCE_PROMPT_CONSTRAINTS` (lines 144-174)
- `SINGLE_FRAME_GLOBAL_PROMPT_CONSTRAINTS` (lines 204-232)
- `SINGLE_FRAME_CONVERGENCE_PROMPT_CONSTRAINTS` (lines 256-279)

- [ ] **Step 3: Simplify `_patch_monitor_prompts`**

Replace the function body (currently lines 282-310) with a version that no longer assigns `_CONSTRAINTS` template attributes. The function now reads:

```python
def _patch_monitor_prompts(mode: MonitorMode) -> None:
    """Set GoalAdherenceMonitor prompt templates for the given mode.

    Must be called for every mode (including "full") to prevent stale
    templates from a previous call leaking into the next run.
    """
    import rvln.ai.goal_adherence_monitor as gam
    from rvln.ai.prompts import (
        DIARY_GLOBAL_PROMPT,
        DIARY_CONVERGENCE_PROMPT,
    )

    if mode == "grid_only":
        gam.GLOBAL_PROMPT_TEMPLATE = GRID_ONLY_GLOBAL_PROMPT
        gam.CONVERGENCE_PROMPT_TEMPLATE = GRID_ONLY_CONVERGENCE_PROMPT
    elif mode == "single_frame":
        gam.GLOBAL_PROMPT_TEMPLATE = SINGLE_FRAME_GLOBAL_PROMPT
        gam.CONVERGENCE_PROMPT_TEMPLATE = SINGLE_FRAME_CONVERGENCE_PROMPT
    else:
        gam.GLOBAL_PROMPT_TEMPLATE = DIARY_GLOBAL_PROMPT
        gam.CONVERGENCE_PROMPT_TEMPLATE = DIARY_CONVERGENCE_PROMPT
```

- [ ] **Step 4: Drop the `constraints` parameter from `run_subgoal`**

In `run_subgoal` signature (currently lines 317-336), remove the `constraints: Optional[List] = None,` parameter.

- [ ] **Step 5: Drop `effective_constraints` and the constraints arguments to `GoalAdherenceMonitor`**

Remove the line `effective_constraints = constraints if config.use_constraints else None` (currently line 377).

In every `GoalAdherenceMonitor(...)` call inside this file (there are at least two: initial creation and the override-subgoal recreation), remove `constraints=effective_constraints,`.

- [ ] **Step 6: Drop `_serialize_constraints` and the `constraints` field in result dict**

Delete the `_serialize_constraints` function (currently at lines 850-854).

In the final return dict of `run_subgoal` (currently around lines 833-847), remove the line `"constraints": _serialize_constraints(effective_constraints),`.

- [ ] **Step 7: Verify**

Run: `grep -n "use_constraints\|constraints\|negative_constraints" src/rvln/eval/subgoal_runner.py`
Expected: empty output.

Run: `python -c "from rvln.eval.subgoal_runner import SubgoalConfig, run_subgoal"`
Expected: no error.

- [ ] **Step 8: Commit**

```bash
git add src/rvln/eval/subgoal_runner.py
git commit -m "eval/subgoal_runner: drop use_constraints field and constraints arg

Removes SubgoalConfig.use_constraints, the constraints kwarg of
run_subgoal, the effective_constraints branch, the four ablation
_CONSTRAINTS prompt variants, the _patch_monitor_prompts assignments
to *_TEMPLATE_CONSTRAINTS, and the _serialize_constraints helper plus
its result-dict field."
```

---

### Task 7: Update `scripts/run_integration.py` (drop sequential planner import + constraint args)

**Files:**
- Modify: `scripts/run_integration.py`

- [ ] **Step 1: Update planner import**

Replace:
```python
from rvln.ai.sequential_ltl_planner import SequentialLTLPlanner
```
with:
```python
from rvln.ai.ltl_planner import LTLSymbolicPlanner
```

- [ ] **Step 2: Replace planner instantiation**

Replace `SequentialLTLPlanner(llm_interface)` with `LTLSymbolicPlanner(llm_interface)`.

- [ ] **Step 3: Drop `use_constraints=False` and constraints args**

Remove:
- `use_constraints=False` argument from any `LLMUserInterface(...)` call.
- The `active_constraints: List[Any] = []` declaration (currently around line 274).
- Any `planner.get_active_constraints()` call.
- `constraints=active_constraints,` argument from `run_subgoal(...)` calls.
- `use_constraints=False,` argument from `SubgoalConfig(...)` (currently line 283).

- [ ] **Step 4: Verify**

Run: `grep -n "SequentialLTLPlanner\|use_constraints\|active_constraints\|get_active_constraints\|constraints=" scripts/run_integration.py`
Expected: empty output.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_integration.py
git commit -m "scripts/run_integration: switch to unified LTLSymbolicPlanner

Drops SequentialLTLPlanner import, use_constraints kwarg,
active_constraints list, get_active_constraints calls, and the
constraints / use_constraints arguments to SubgoalConfig and
run_subgoal."
```

---

### Task 8: Update `scripts/run_hardware.py` (same cleanup as Task 7)

**Files:**
- Modify: `scripts/run_hardware.py`

- [ ] **Step 1: Audit current state**

Run: `grep -n "SequentialLTLPlanner\|LTLSymbolicPlanner\|use_constraints\|active_constraints\|get_active_constraints\|constraints=" scripts/run_hardware.py`
Note every line that comes up.

- [ ] **Step 2: Apply the same edits as Task 7**

- Replace any `from rvln.ai.sequential_ltl_planner import SequentialLTLPlanner` with `from rvln.ai.ltl_planner import LTLSymbolicPlanner`.
- Replace `SequentialLTLPlanner(...)` with `LTLSymbolicPlanner(...)`.
- Drop `use_constraints=False` from `LLMUserInterface(...)`.
- Drop any `active_constraints` declaration, `get_active_constraints` call, and `constraints=...` argument.
- Drop `use_constraints=False` from any `SubgoalConfig(...)` call.

- [ ] **Step 3: Verify**

Run: `grep -n "SequentialLTLPlanner\|use_constraints\|active_constraints\|get_active_constraints\|constraints=" scripts/run_hardware.py`
Expected: empty output.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_hardware.py
git commit -m "scripts/run_hardware: switch to unified LTLSymbolicPlanner

Same cleanup as run_integration: drop sequential planner import and
all constraint plumbing."
```

---

### Task 9: Update `scripts/run_condition2_llm_planner.py` (drop `use_constraints=False` only)

**Files:**
- Modify: `scripts/run_condition2_llm_planner.py`

C2 does not use the LTL planner; its only constraint reference is in `SubgoalConfig`.

- [ ] **Step 1: Drop `use_constraints=False` from `SubgoalConfig`**

Currently at line 285. Remove the line.

- [ ] **Step 2: Verify**

Run: `grep -n "use_constraints\|constraints" scripts/run_condition2_llm_planner.py`
Expected: empty output.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_condition2_llm_planner.py
git commit -m "scripts/run_condition2: drop use_constraints kwarg

C2 does not use the LTL planner; the only constraint reference was
SubgoalConfig(use_constraints=False), which no longer exists."
```

---

### Task 10: Update `scripts/run_condition3_open_loop.py`

**Files:**
- Modify: `scripts/run_condition3_open_loop.py`

- [ ] **Step 1: Drop constraint references**

Run: `grep -n "use_constraints\|active_constraints\|get_active_constraints\|constraints=" scripts/run_condition3_open_loop.py`
Note every match. (C3 does not call the monitor, so it likely has fewer references than C4-C6.)

- [ ] **Step 2: Apply the cleanup**

For each match found in Step 1:
- Drop `planner.get_active_constraints()` calls and any variable they assign to.
- Drop `use_constraints=True` (or `False`) arguments to `SubgoalConfig` or `LLMUserInterface`.
- Drop `constraints=active_constraints,` arguments to `run_subgoal`.

- [ ] **Step 3: Verify**

Run: `grep -n "use_constraints\|active_constraints\|get_active_constraints\|constraints=" scripts/run_condition3_open_loop.py`
Expected: empty output.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_condition3_open_loop.py
git commit -m "scripts/run_condition3: drop constraint plumbing

Same cleanup as run_integration."
```

---

### Task 11: Update `scripts/run_condition4_single_frame.py`, `run_condition5_grid_only.py`, `run_condition6_text_only.py`

**Files:**
- Modify: `scripts/run_condition4_single_frame.py`
- Modify: `scripts/run_condition5_grid_only.py`
- Modify: `scripts/run_condition6_text_only.py`

Each of these currently has the same shape:

```python
planner = LTLSymbolicPlanner(llm_interface)
...
active_constraints = planner.get_active_constraints()
...
config = SubgoalConfig(
    ...
    use_constraints=True,
)
result = run_subgoal(
    ...
    constraints=active_constraints,
)
```

- [ ] **Step 1: Drop `active_constraints` declaration and `get_active_constraints()` call (each file)**

- [ ] **Step 2: Drop `use_constraints=True` from each `SubgoalConfig(...)` call**

- [ ] **Step 3: Drop `constraints=active_constraints,` from each `run_subgoal(...)` call**

- [ ] **Step 4: Verify (each file)**

Run for each file:
```bash
grep -n "use_constraints\|active_constraints\|get_active_constraints\|constraints=" scripts/run_condition4_single_frame.py
grep -n "use_constraints\|active_constraints\|get_active_constraints\|constraints=" scripts/run_condition5_grid_only.py
grep -n "use_constraints\|active_constraints\|get_active_constraints\|constraints=" scripts/run_condition6_text_only.py
```
Expected: empty output for each.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_condition4_single_frame.py scripts/run_condition5_grid_only.py scripts/run_condition6_text_only.py
git commit -m "scripts/run_conditions 4-6: drop constraint plumbing

Same cleanup as run_integration."
```

---

### Task 12: Update `scripts/run_repl.py`, `scripts/verify_automaton_walkthrough.py`, `scripts/run_all_conditions.py`

**Files:**
- Modify: `scripts/run_repl.py`
- Modify: `scripts/verify_automaton_walkthrough.py`
- Modify: `scripts/run_all_conditions.py`

- [ ] **Step 1: Audit**

Run for each file:
```bash
grep -n "use_constraints\|active_constraints\|get_active_constraints\|constraint_predicates\|ConstraintInfo\|SequentialLTLPlanner" scripts/run_repl.py scripts/verify_automaton_walkthrough.py scripts/run_all_conditions.py
```

- [ ] **Step 2: `scripts/run_repl.py`**

If `LLMUserInterface(use_constraints=...)` appears, drop the kwarg. Otherwise no change needed.

- [ ] **Step 3: `scripts/verify_automaton_walkthrough.py`**

This script's stated purpose includes "Checks get_active_constraints() at each step" and "no constraint predicate leaked into the goal sequence". Now that the planner has no constraint surface, those checks are vacuous.

Either:
- (a) Delete the script entirely if it has no remaining value: `git rm scripts/verify_automaton_walkthrough.py`
- (b) Strip the constraint-specific parts: remove all references to `planner.constraint_predicates`, `planner.get_active_constraints()`, `active_constraints`, `leaked_constraints`, and `ConstraintInfo`. Keep only the goal-walk verification that the planner returns each predicate in order and reaches the sink.

Choose (a) unless the goal-walk part is being used by another tool. Verify with: `grep -rn "verify_automaton_walkthrough" .` — if only its own file shows up, delete it.

- [ ] **Step 4: `scripts/run_all_conditions.py`**

Apply any constraint-related cleanup the audit revealed (typically just dropping `use_constraints` flags or constraint references).

- [ ] **Step 5: Verify**

Run: `grep -n "use_constraints\|active_constraints\|get_active_constraints\|constraint_predicates\|ConstraintInfo\|SequentialLTLPlanner" scripts/run_repl.py scripts/run_all_conditions.py 2>&1; ls scripts/verify_automaton_walkthrough.py 2>&1`
Expected: empty output for the greps; the verify script is either gone or stripped.

- [ ] **Step 6: Commit**

```bash
git add scripts/
git commit -m "scripts: drop constraint plumbing from REPL, walkthrough, all-conditions"
```

---

### Task 13: Update `src/rvln/mininav/interface.py`

**Files:**
- Modify: `src/rvln/mininav/interface.py`

This module is a parallel integration runner with similar plumbing to `scripts/run_integration.py`.

- [ ] **Step 1: Audit**

Run: `grep -n "SequentialLTLPlanner\|use_constraints\|active_constraints\|get_active_constraints\|constraint_predicates\|constraints=\|negative_constraints\|ManualSequentialLTLPlanner" src/rvln/mininav/interface.py`

- [ ] **Step 2: Replace planner imports**

For both the conditional `from rvln.ai.no_ai_stubs import (... ManualSequentialLTLPlanner as SequentialLTLPlanner ...)` block (currently around line 2468) and the `from rvln.ai.sequential_ltl_planner import SequentialLTLPlanner` block (currently around line 2477), import `LTLSymbolicPlanner` from the unified module instead. Pick one of:
- Direct: `from rvln.ai.ltl_planner import LTLSymbolicPlanner` (and `from rvln.ai.no_ai_stubs import ManualLTLSymbolicPlanner as LTLSymbolicPlanner` for the stub branch — see Task 14 for the stub rename).
- Or, if the conditional was only there to pick stub-vs-real, simplify the block to a single import path.

Replace `SequentialLTLPlanner(llm_interface)` with `LTLSymbolicPlanner(llm_interface)`.

- [ ] **Step 3: Drop `use_constraints=False` from `LLMUserInterface(...)` calls**

Both occurrences (currently lines 2479 and 2555).

- [ ] **Step 4: Drop the `constraints` parameter from any helper function in this module**

The function at line 1163 has a `constraints: Optional[List[Any]] = None,` parameter and passes `constraints=constraints,` at line 1222. Remove both.

- [ ] **Step 5: Drop `active_constraints` declarations and `constraints=active_constraints,` arguments**

Currently at line 2505 and 2532. Remove.

- [ ] **Step 6: Verify**

Run: `grep -n "SequentialLTLPlanner\|use_constraints\|active_constraints\|get_active_constraints\|constraints=\|negative_constraints" src/rvln/mininav/interface.py`
Expected: empty output (or only the import alias of `LTLSymbolicPlanner` if you renamed in step 2).

- [ ] **Step 7: Commit**

```bash
git add src/rvln/mininav/interface.py
git commit -m "mininav/interface: switch to unified LTLSymbolicPlanner

Drops sequential planner imports, use_constraints kwargs,
active_constraints lists, and the constraints helper parameter."
```

---

### Task 14: Update `src/rvln/ai/no_ai_stubs.py` (rename `ManualSequentialLTLPlanner` → `ManualLTLSymbolicPlanner`; drop constraint params)

**Files:**
- Modify: `src/rvln/ai/no_ai_stubs.py`

- [ ] **Step 1: Drop `use_constraints` from the LLM stub**

In the stub at line 96, change `def __init__(self, model: str = "no-ai", use_constraints: bool = False):` to `def __init__(self, model: str = "no-ai"):` and delete the `self._use_constraints = use_constraints` line below it.

Update the module docstring at line 4 to drop the `SequentialLTLPlanner` reference (replace with `LTLSymbolicPlanner`).

- [ ] **Step 2: Rename the planner stub**

Rename `class ManualSequentialLTLPlanner` (line 135) to `class ManualLTLSymbolicPlanner`.
Update the docstring at line 136 to reference `rvln.ai.ltl_planner.LTLSymbolicPlanner` instead of the deleted sequential module.

- [ ] **Step 3: Drop `constraints` and `negative_constraints` from any monitor stub**

At line 229-230, remove:
- `constraints: Optional[List[Any]] = None,`
- `negative_constraints: Optional[List[str]] = None,`

If the stub stores them on `self`, drop those assignments too.

- [ ] **Step 4: Verify**

Run: `grep -n "ManualSequentialLTLPlanner\|use_constraints\|negative_constraints\|self._constraints" src/rvln/ai/no_ai_stubs.py`
Expected: empty output (the new class name `ManualLTLSymbolicPlanner` should not match this grep).

- [ ] **Step 5: Update consumer in `mininav/interface.py`**

If the import alias in `src/rvln/mininav/interface.py` (Task 13 step 2) used `ManualSequentialLTLPlanner`, update it to `ManualLTLSymbolicPlanner`.

- [ ] **Step 6: Commit**

```bash
git add src/rvln/ai/no_ai_stubs.py src/rvln/mininav/interface.py
git commit -m "ai/no_ai_stubs: rename ManualSequentialLTLPlanner -> ManualLTLSymbolicPlanner

Drops use_constraints / negative_constraints / constraints params
from the LLM and monitor stubs. Updates mininav/interface import alias
to match."
```

---

### Task 15: Update `src/rvln/eval/playback.py` and `src/rvln/eval/task_utils.py`

**Files:**
- Modify: `src/rvln/eval/playback.py`
- Modify: `src/rvln/eval/task_utils.py`

`playback.py` has a constraint-overlay feature: `load_constraint_text`, `draw_constraint_overlay`, and the `constraint_text` parameter pass-through. Now that constraints aren't tracked, the overlay always renders nothing.

`task_utils.py` checks for a `constraints_expected` field in task JSON, which is no longer populated.

- [ ] **Step 1: `playback.py` - delete constraint-overlay machinery**

Delete:
- `load_constraint_text` (lines 68-...).
- `draw_constraint_overlay` (line 135).
- The `constraint_text: Optional[str] = None,` parameter from any function that accepts it (line 197).
- The `constraint_text` docstring entry (line 213-214).
- The `if constraint_text: draw_constraint_overlay(...)` block (lines 246-247).
- The `constraint_text=...` calls (lines 273, 277).
- The `overlay` parameter ONLY if it solely controlled constraint overlay; if it also controlled other overlays, leave it but drop the constraint branch.

- [ ] **Step 2: `task_utils.py` - drop `constraints_expected` handling**

At line 86, the loader checks for `"constraints_expected"`. Remove the field from any allowed/expected key list and drop any branch that populates it.

- [ ] **Step 3: Verify**

Run: `grep -n "constraint" src/rvln/eval/playback.py src/rvln/eval/task_utils.py`
Expected: empty output.

- [ ] **Step 4: Commit**

```bash
git add src/rvln/eval/playback.py src/rvln/eval/task_utils.py
git commit -m "eval/playback,task_utils: drop constraint overlay and expected-constraints

Removes load_constraint_text, draw_constraint_overlay, the constraint_text
parameter pass-through in playback rendering, and the constraints_expected
handling in task_utils."
```

---

### Task 16: Delete the three constraint-only test files

**Files:**
- Delete: `tests/test_negative_constraints.py`
- Delete: `tests/test_prompt_constraints.py`
- Delete: `tests/test_vlm_constraint_prompts.py`

- [ ] **Step 1: Confirm none of these tests are referenced by other tests or fixtures**

Run: `grep -rn "test_negative_constraints\|test_prompt_constraints\|test_vlm_constraint_prompts" tests/ pyproject.toml`
Expected: only the three files themselves (or no matches outside their own files).

- [ ] **Step 2: Delete**

```bash
git rm tests/test_negative_constraints.py tests/test_prompt_constraints.py tests/test_vlm_constraint_prompts.py
```

- [ ] **Step 3: Commit**

```bash
git commit -m "tests: delete constraint-only test files

test_negative_constraints, test_prompt_constraints, and
test_vlm_constraint_prompts exercised behavior that no longer exists
(constraint classification, _WITH_CONSTRAINTS prompts)."
```

---

### Task 17: Rewrite `tests/test_condition_ablations.py`

**Files:**
- Modify (full rewrite of relevant cases): `tests/test_condition_ablations.py`

The current tests assert each condition script imports `LTLSymbolicPlanner`, calls `get_active_constraints()`, and passes `constraints=active_constraints`. After the refactor, the planner import remains but the constraint calls are gone.

- [ ] **Step 1: Read the current test file**

Run: `cat tests/test_condition_ablations.py`

Note the structure (it appears to be source-string scanning rather than runtime invocation).

- [ ] **Step 2: Rewrite the assertions**

For each test that previously asserted `assert "get_active_constraints" in source` or `assert "constraints=active_constraints" in source`, replace with:
```python
assert "get_active_constraints" not in source
assert "constraints=active_constraints" not in source
assert "use_constraints" not in source
```

For each test that asserted `assert "LTLSymbolicPlanner" in names`, keep that assertion (the class still exists in `rvln.ai.ltl_planner`). For tests asserting `assert "LTLSymbolicPlanner" not in names` for C1/C2, keep those.

For the test that asserted `assert "constraints=active_constraints" not in source` for C1/C2/C3, keep — the whole codebase now satisfies it.

- [ ] **Step 3: Run the test file**

Run: `pytest tests/test_condition_ablations.py -v`
Expected: PASS for all cases.

- [ ] **Step 4: Commit**

```bash
git add tests/test_condition_ablations.py
git commit -m "tests/test_condition_ablations: assert no constraint plumbing

After the LTL-NL prompt unification, every condition script must NOT
call get_active_constraints, pass constraints=active_constraints, or
use the use_constraints kwarg. C0, C3, C4, C5, C6 still import
LTLSymbolicPlanner; C1 and C2 still do not."
```

---

### Task 18: Update `tests/test_ltl_planner.py` and `tests/test_parsing.py`

**Files:**
- Modify: `tests/test_ltl_planner.py`
- Modify: `tests/test_parsing.py`

- [ ] **Step 1: Audit `test_ltl_planner.py`**

Run: `grep -n "ConstraintInfo\|constraint_predicates\|get_active_constraints\|G(\|negative\|positive_constraint" tests/test_ltl_planner.py`

- [ ] **Step 2: Strip constraint-specific cases from `test_ltl_planner.py`**

Remove the `ConstraintInfo` import. Delete any test that exercises `constraint_predicates`, `get_active_constraints`, polarity classification, or constraint-violation BDD paths. Keep tests that exercise:
- `_predicate_key_to_index`, `_normalize_pi_predicates`
- `plan_from_natural_language` happy path
- `get_next_predicate` ordering
- `advance_state` transitions
- The new last-goal-via-sink-edge fallback (write a fresh test that constructs a small monitor automaton where the last goal's only outgoing edge is the sink; verify `get_next_predicate` still returns it)

- [ ] **Step 3: Audit `test_parsing.py`**

Run: `grep -n "constraint\|G(\|negative\|positive" tests/test_parsing.py`

Strip any case that depends on the constraint-aware prompt's parsing behavior (e.g., expects the LTL parser to produce `G(!pi_X)` outputs). Keep general LTL-NL string parsing tests.

- [ ] **Step 4: Run the tests**

Run: `conda run -n rvln-sim pytest tests/test_ltl_planner.py tests/test_parsing.py -v`
Expected: PASS for all remaining cases.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ltl_planner.py tests/test_parsing.py
git commit -m "tests/test_ltl_planner,test_parsing: drop constraint-specific cases

Removes ConstraintInfo imports, constraint_predicates assertions,
polarity tests, and parsing tests for G() / U-scoped constraint
forms. Adds one regression test for the last-goal-via-sink-edge
fallback that this refactor preserved."
```

---

### Task 19: Update `experimental_design.txt`

**Files:**
- Modify: `experimental_design.txt`

- [ ] **Step 1: Delete Section 3 entirely**

Section 3 ("TEMPORAL CONSTRAINT SUPPORT", line 216) and its subsections 3a, 3b, 3c, 3d (running through approximately line 345). Delete all of it.

- [ ] **Step 2: Update Section 2 CONDITION 2 description**

Delete the "CONFOUND NOTE" paragraph that begins around line 102.

- [ ] **Step 3: Update Section 4b "TASK CATEGORIES"**

Delete the "Category: CONSTRAINED" block and its task list. Update the "Task distribution per map" line. Update "With 3 starting position variants per task" sentence to use the new total. Delete the "NOTE ON TASK BALANCE" paragraph.

- [ ] **Step 4: Update Section 4c**

In the "TASK JSON FORMAT" example, drop the `"category"` field. Drop the explanation paragraph about category.

- [ ] **Step 5: Delete Metric M2**

In Section 5, delete the entire M2 ("Constraint Adherence Rate") block, including its rationale paragraph.

- [ ] **Step 6: Delete Section 6f**

Delete "MANUAL CONSTRAINT ADHERENCE ANNOTATION" entirely.

- [ ] **Step 7: Update Section 6b episode count**

Update from `15 tasks x 3 starting variations = 45 episodes per condition` and `7 conditions x 45 = 315 total episodes` to whatever the new sequential-only task count yields. With 6 sequential tasks: `6 tasks x 3 variants = 18 episodes per condition; 7 conditions x 18 = 126 total episodes`.

- [ ] **Step 8: Delete the Constraint Enforcement row in Section 8c**

Delete the `Constraint Enforcement | Yes | No  | No  | No  | Yes | Yes | Yes` row from the ablation table.

- [ ] **Step 9: Update Section 11**

Delete the bullet about "the strongest story comes from the gap between Condition 2 and Condition 0 on constraint tasks". Delete the "Per-category" reporting bullet.

- [ ] **Step 10: Update Section 12**

Delete item 1 ("Temporal constraint support"). Renumber the remaining items.

- [ ] **Step 11: Verify**

Run: `grep -n -i "constraint\|constrained" experimental_design.txt`
Expected: no matches (or only matches in unrelated context).

- [ ] **Step 12: Commit**

```bash
git add experimental_design.txt
git commit -m "experimental_design: remove constrained-task track

Drops Section 3 (Temporal Constraint Support), the constrained task
category in 4b, M2 (Constraint Adherence Rate), Section 6f (manual
constraint adherence annotation), the Constraint Enforcement row in
the 8c ablation table, the C2 confound note, the constraint-related
paper-framing bullets in Section 11, and the constraint priority item
in Section 12. Updates Section 6b episode counts to reflect the
sequential-only task set."
```

---

### Task 20: Final verification

**Files:**
- No code changes; this task is pure verification.

- [ ] **Step 1: Repo-wide constraint-reference scan**

Run: `grep -rn "WITH_CONSTRAINTS\|use_constraints\|constraint_predicates\|get_active_constraints\|SequentialLTLPlanner\|ConstraintInfo\|_constraints_block\|negative_constraints" --include="*.py" --include="*.txt" .`
Expected: empty output.

If anything matches, return to the appropriate task and finish the cleanup before declaring done.

- [ ] **Step 2: Sequential-prompt scan**

Run: `grep -rn "LTL_NL_SYSTEM_PROMPT_SEQUENTIAL\|LTL_NL_EXAMPLES_PROMPT_SEQUENTIAL" --include="*.py" .`
Expected: empty output.

- [ ] **Step 3: Run the full pytest suite**

Run: `conda run -n rvln-sim pytest tests/ -v 2>&1 | tail -40`
Expected: all tests pass. If any fail, fix the cause before declaring done.

- [ ] **Step 4: Smoke test the unified prompt**

Run (only if an LLM is configured):
```bash
conda run -n rvln-sim python -c "
from rvln.ai.llm_interface import LLMUserInterface
from rvln.ai.ltl_planner import LTLSymbolicPlanner

llm = LLMUserInterface()
planner = LTLSymbolicPlanner(llm)
planner.plan_from_natural_language(
    'turn left until you see the tree, then go toward the tree, '
    'then turn right until you see the building, and go to the building'
)
print('PI MAP:', planner.pi_map)
print('FORMULA:', llm.ltl_nl_formula['ltl_nl_formula'])
print('FIRST:', planner.get_next_predicate())
"
```
Expected: `PI MAP` shows 4 predicates each containing the full conditional clause; `FORMULA` is `F pi_4 & (!pi_4 U pi_3) & (!pi_3 U pi_2) & (!pi_2 U pi_1)` (or equivalent ordering); `FIRST` is `"Turn left until you see the tree"`.

- [ ] **Step 5: Final commit (only if any cleanup edits were needed)**

If steps 1-4 surfaced any leftover references that needed fixing, commit them as a single fix-up. Otherwise, this task produces no commit.

---

## Self-Review

**Spec coverage:**
- Section 1 (module structure): Task 2.
- Section 2 (prompts): Task 3.
- Section 3 (`LLMUserInterface`): Task 4.
- Section 4 (`goal_adherence_monitor`): Task 5.
- Section 5 (`subgoal_runner`): Task 6.
- Section 6 (runner scripts): Tasks 7-13.
- Section 7 (tests): Tasks 16-18.
- Section 8 (`experimental_design.txt`): Task 19.
- Section 9 (`cached_formulas/`): Task 1.
- Other consumers (no_ai_stubs, playback, task_utils, mininav.interface, run_repl, verify_automaton_walkthrough): Tasks 12-15.
- Verification: Task 20.

**Placeholder scan:** No "TBD", "TODO", or "implement later" markers. Each step has either an exact diff target or a precise grep / verification command.

**Type consistency:** Class name stays `LTLSymbolicPlanner` throughout. Stub class is renamed to `ManualLTLSymbolicPlanner` consistently in Task 14 and consumed correctly in `mininav/interface.py` per Task 14 step 5. Renamed prompt constants are consumed only by `llm_interface.py`, updated in Task 4. Module-level template attributes on `goal_adherence_monitor` lose their `_CONSTRAINTS` siblings, and `subgoal_runner._patch_monitor_prompts` no longer assigns to those siblings (Task 5 + Task 6).

No issues found.
