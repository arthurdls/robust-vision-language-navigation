# LTL-NL Prompt Unification

Status: design accepted, pending implementation plan.

## Background

The codebase currently carries two parallel LTL-NL planning paths:

1. **Constraint-aware path.** `LTL_NL_SYSTEM_PROMPT` + `LTL_NL_EXAMPLES_PROMPT` instruct the LLM to lift instructions like "always stay above 10 m" or "never fly over building C" into separate constraint predicates connected by `G(...)`, `G(!...)`, or scoped `... U ...` clauses. `LTLSymbolicPlanner` then walks the formula tree at plan time to classify each predicate as a goal, a positive (maintenance) constraint, or a negative (avoidance) constraint, and exposes `get_active_constraints()` per automaton state. The `GoalAdherenceMonitor` accepts those `ConstraintInfo` objects and injects an "Active constraints (must be maintained throughout):" block into a parallel set of prompt templates (`*_WITH_CONSTRAINTS`).
2. **Sequential-only path.** `LTL_NL_SYSTEM_PROMPT_SEQUENTIAL` + `LTL_NL_EXAMPLES_PROMPT_SEQUENTIAL` tell the LLM to keep every "until / while / for N meters / without X" clause inside the predicate's natural-language description, treating each predicate purely as a goal whose completion criterion is the full clause. `SequentialLTLPlanner` walks the Spot automaton without any constraint classification.

The hardware runner and the full-system integration runner already use the sequential path. The constraint-aware path remains in place for the ablation conditions (C3-C6) and tests, but the dual code path imposes ongoing maintenance cost: two planner classes, two planner-prompt sets, six monitor-prompt variants, conditional plumbing in `LLMUserInterface`, `GoalAdherenceMonitor`, `SubgoalConfig`, and `subgoal_runner.run_subgoal`, plus four test files locking in constraint behavior.

The user has already deleted the constraint-bearing task JSONs (above_pergolas, sidewalk_to_crosswalk, person_to_cars_to_traffic_light, etc.) from `tasks/`, signaling that the constrained-task track is being retired from the experimental program.

## Goal

Collapse the two paths into one. Adopt the sequential prompt as the only planner prompt. Each natural-language constraint clause stays inside the predicate it scopes ("turn left until you see the tree" is one predicate, not two), and the LTL formula remains a strict sequential state machine over those predicates. The Spot automaton continues to drive subgoal advancement; the LLM still emits an LTL-NL formula; nothing about the planner-as-state-machine architecture changes. What goes away is the formal constraint enforcement layer (G(...) classification, active-constraints injection, `_WITH_CONSTRAINTS` prompt variants, and the `use_constraints` toggle that selected between the two paths).

After the refactor, the example "turn left until you see the tree, then go toward the tree, then turn right until you see the building, and go to it" produces exactly four predicates:

```
pi_1: "Turn left until you see the tree"
pi_2: "Go toward the tree"
pi_3: "Turn right until you see the building"
pi_4: "Go to the building"
ltl_nl_formula: F pi_4 & (!pi_4 U pi_3) & (!pi_3 U pi_2) & (!pi_2 U pi_1)
```

The "until you see X" stopping clauses are part of each predicate's completion criterion, evaluated by the goal-adherence monitor at runtime.

## Approach

Direct replacement, preserving the class name `LTLSymbolicPlanner` (the LTL planner remains symbolic in the Spot-automaton sense even after constraint classification is removed). The current `SequentialLTLPlanner` body becomes the new `LTLSymbolicPlanner` body. The constraint-aware planner, prompts, and plumbing are deleted outright; no shims, no deprecation phase. The `cached_formulas/` directory is wiped so that no stale entries from the constraint-aware prompt linger.

## Component changes

### 1. `src/rvln/ai/ltl_planner.py` and `src/rvln/ai/sequential_ltl_planner.py`

- Replace the body of `ltl_planner.py` with the current `SequentialLTLPlanner` implementation, renaming the class to `LTLSymbolicPlanner`.
- Bring forward the **last-goal sink-edge fallback** from the current `LTLSymbolicPlanner.get_next_predicate` (the "BUG FIX: on some Spot versions" branch). This compensates for monitor-automaton variants where the last goal's only outgoing edge leads to the sink, and is unrelated to constraint handling.
- Delete `src/rvln/ai/sequential_ltl_planner.py`.
- Removed from the class: `ConstraintInfo` dataclass, `constraint_predicates`, `get_active_constraints`, `_classify_predicates`, `_collect_goal_aps`, `_collect_all_aps`, `_walk_classify`, `_classify_under_g`, `_classify_until_left`, `_ap_description`, `_active_positive_constraint_indices`, `_get_bdd_goal_check`, `_get_bdd_constraint_violation`, the `_bdd_false` cache field.
- `_add_sink_state` uses the same sequential semantics the current `SequentialLTLPlanner` uses: last predicate's single-task BDD as the sink condition.

### 2. `src/rvln/ai/prompts.py`

Delete (constraint-aware planner prompts plus their constraint-injecting monitor siblings):
- `LTL_NL_SYSTEM_PROMPT` (constraint-aware version)
- `LTL_NL_EXAMPLES_PROMPT` (constraint-aware version)
- `DIARY_GLOBAL_PROMPT_WITH_CONSTRAINTS`
- `DIARY_CONVERGENCE_PROMPT_WITH_CONSTRAINTS`
- `TEXT_ONLY_GLOBAL_PROMPT_WITH_CONSTRAINTS`
- `TEXT_ONLY_CONVERGENCE_PROMPT_WITH_CONSTRAINTS`

Rename (drop the `_SEQUENTIAL` suffix; sequential is now the only prompt):
- `LTL_NL_SYSTEM_PROMPT_SEQUENTIAL` -> `LTL_NL_SYSTEM_PROMPT`
- `LTL_NL_EXAMPLES_PROMPT_SEQUENTIAL` -> `LTL_NL_EXAMPLES_PROMPT`

Untouched: `LTL_NL_RESTATED_TASK_PROMPT`, `SUBGOAL_CONVERSION_PROMPT`, the non-constraint monitor prompts (`DIARY_GLOBAL_PROMPT`, `DIARY_CONVERGENCE_PROMPT`, `TEXT_ONLY_GLOBAL_PROMPT`, `TEXT_ONLY_CONVERGENCE_PROMPT`), the OFFLINE EVALUATION-ONLY check prompts (`LTL_NL_CHECK_PREDICATES_PROMPT`, `LTL_NL_CHECK_SEMANTICS_PROMPT`), the goal-adherence helper prompts (`DEFAULT_TEMPORAL_PROMPT`, `DRONE_GOAL_MONITOR_CONTEXT`, `PROMPT_SUBTASK_COMPLETE`, `WHAT_CHANGED_PROMPT`, `SUBTASK_COMPLETE_DIARY_PROMPT`).

The "Sequential-only LTL planner prompts (pre-constraint behavior)" section comment block is removed; the renamed prompts move under a normal section header.

`TEXT_ONLY_GLOBAL_PROMPT` and `TEXT_ONLY_CONVERGENCE_PROMPT` currently contain `{constraints_block}` placeholders inside their template strings (lines 994 and 1064). Those placeholders are removed from the strings as part of this refactor.

### 3. `src/rvln/ai/llm_interface.py`

- Drop the `use_constraints: bool = True` constructor parameter.
- Drop the `if use_constraints / else` branch selecting between system/examples prompts.
- Always use the (single) renamed `LTL_NL_SYSTEM_PROMPT` + `LTL_NL_EXAMPLES_PROMPT` + `LTL_NL_RESTATED_TASK_PROMPT`.
- Remove imports of `LTL_NL_SYSTEM_PROMPT_SEQUENTIAL` / `LTL_NL_EXAMPLES_PROMPT_SEQUENTIAL` (no longer exist).
- The `_prompt_version_for(...)` hash and on-disk formula cache mechanism are unchanged. Old cache entries become orphans because their hash differs; the `cached_formulas/` directory is wiped (Section 6) to keep the repo clean.

### 4. `src/rvln/ai/goal_adherence_monitor.py`

- Remove imports of the four `_WITH_CONSTRAINTS` monitor prompts and the `_TEMPLATE_CONSTRAINTS` aliases.
- Remove module-level globals `GLOBAL_PROMPT_TEMPLATE_CONSTRAINTS`, `CONVERGENCE_PROMPT_TEMPLATE_CONSTRAINTS`, `TEXT_ONLY_GLOBAL_PROMPT_TEMPLATE_CONSTRAINTS`, `TEXT_ONLY_CONVERGENCE_PROMPT_TEMPLATE_CONSTRAINTS`.
- Drop `constraints` and `negative_constraints` constructor parameters; drop `self._constraints`.
- Delete `_constraints_block()`.
- Each prompt-build site (the four `if self._constraints: ... .format(constraints_block=...)` else branches at approximate lines 579, 876, 931, 1158) collapses to its non-constraint branch. Each `.format(...)` call drops the `constraints_block=` argument. The non-constraint prompt strings have no `{constraints_block}` placeholder after Section 2's edits to `TEXT_ONLY_GLOBAL_PROMPT` and `TEXT_ONLY_CONVERGENCE_PROMPT`.

### 5. `src/rvln/eval/subgoal_runner.py`

- Remove `use_constraints: bool = True` from `SubgoalConfig`.
- Remove the `constraints: Optional[List] = None` parameter from `run_subgoal`.
- Remove `effective_constraints = constraints if config.use_constraints else None` and stop passing it to `GoalAdherenceMonitor` (the kwarg there is gone).
- Delete the ablation prompt variants `GRID_ONLY_GLOBAL_PROMPT_CONSTRAINTS`, `GRID_ONLY_CONVERGENCE_PROMPT_CONSTRAINTS`, `SINGLE_FRAME_GLOBAL_PROMPT_CONSTRAINTS`, `SINGLE_FRAME_CONVERGENCE_PROMPT_CONSTRAINTS`.
- In `_patch_monitor_prompts`, drop the assignments to `gam.GLOBAL_PROMPT_TEMPLATE_CONSTRAINTS` / `gam.CONVERGENCE_PROMPT_TEMPLATE_CONSTRAINTS` (those module attributes no longer exist).
- Drop `_serialize_constraints` and the `"constraints": _serialize_constraints(...)` field from the result dict.

### 6. Runner scripts

`scripts/run_integration.py`:
- Change `from rvln.ai.sequential_ltl_planner import SequentialLTLPlanner` to `from rvln.ai.ltl_planner import LTLSymbolicPlanner`.
- Replace `SequentialLTLPlanner(...)` with `LTLSymbolicPlanner(...)`.
- Drop `use_constraints=False` argument to `LLMUserInterface(...)`.
- Drop `active_constraints: List[Any] = []`, drop any `planner.get_active_constraints()` call, drop `constraints=active_constraints` argument to `SubgoalConfig` / `run_subgoal`.
- Drop `use_constraints=False` argument to `SubgoalConfig`.

`scripts/run_condition3_open_loop.py`, `scripts/run_condition4_single_frame.py`, `scripts/run_condition5_grid_only.py`, `scripts/run_condition6_text_only.py`:
- Drop `planner.get_active_constraints()` calls.
- Drop `use_constraints=True` argument to `SubgoalConfig`.
- Drop `constraints=active_constraints` argument to `run_subgoal`.
- Class name `LTLSymbolicPlanner` and import path are unchanged.

`scripts/run_condition2_llm_planner.py`:
- Drop `use_constraints=False` from `SubgoalConfig`. C2 does not use the LTL planner, so no planner-related changes.

`scripts/run_hardware.py`:
- If it imports `SequentialLTLPlanner`, switch to `LTLSymbolicPlanner` from `rvln.ai.ltl_planner`. Drop `use_constraints=False` from `LLMUserInterface(...)` call.

`scripts/run_repl.py` and `scripts/verify_automaton_walkthrough.py`:
- Audit for any consumer of `LLMUserInterface(use_constraints=...)`, `get_active_constraints()`, or `constraint_predicates`. Drop those calls. The REPL only constructs `LLMUserInterface()`; the walkthrough script may walk the automaton, in which case no change.

Other consumers (`src/rvln/ai/no_ai_stubs.py`, `src/rvln/eval/playback.py`, `src/rvln/eval/task_utils.py`, `src/rvln/mininav/interface.py`): audit and remove constraint references. These are mostly diagnostic / serialization paths; remove the `constraints` field if any results dict or task structure carries it.

### 7. Tests

- Delete `tests/test_negative_constraints.py`. The behavior it tests no longer exists.
- Delete `tests/test_prompt_constraints.py`. The constraint-aware prompt no longer exists.
- Delete `tests/test_vlm_constraint_prompts.py`. The `_WITH_CONSTRAINTS` monitor prompts no longer exist.
- Rewrite `tests/test_condition_ablations.py`. The new assertions:
  - Each condition script (`run_integration.py`, `run_condition3_open_loop.py`, `run_condition4_single_frame.py`, `run_condition5_grid_only.py`, `run_condition6_text_only.py`) imports `LTLSymbolicPlanner` from `rvln.ai.ltl_planner`.
  - No condition script references `get_active_constraints`, `constraints=active_constraints`, `use_constraints=`, or `SequentialLTLPlanner`.
  - `run_condition1_naive.py` and `run_condition2_llm_planner.py` continue to NOT import `LTLSymbolicPlanner`.
- Audit `tests/test_ltl_planner.py` and `tests/test_parsing.py`. Remove any cases that exercise constraint-aware behavior; preserve cases that test sequential automaton walking.

### 8. `experimental_design.txt`

- Delete Section 3 ("TEMPORAL CONSTRAINT SUPPORT") in its entirety, including subsections 3a, 3b, 3c, 3d.
- In Section 4b ("TASK CATEGORIES"), delete the "CONSTRAINED" task category, the "Task distribution per map" line that splits sequential vs constrained, the "NOTE ON TASK BALANCE" paragraph, and the example list of constraint tasks. The remaining 6 sequential tasks become the full task set.
- In Section 4c, the example task JSON loses the `"category"` field discussion.
- Delete metric M2 ("Constraint Adherence Rate") from Section 5. Renumber subsequent metrics (M3 -> M2, M4 -> M3, etc.) only if renumbering is consistent across the doc; otherwise keep numeric labels and just remove M2.
- Delete Section 6f ("MANUAL CONSTRAINT ADHERENCE ANNOTATION").
- In Section 8c (ablation summary table), delete the "Constraint Enforcement" row.
- In Section 2's CONDITION 2 description, delete the "CONFOUND NOTE" paragraph that talks about C2 ablating both LTL planning and formal constraint enforcement.
- In Section 11 ("PAPER FRAMING NOTES"), delete the bullet about "the strongest story comes from the gap between Condition 2 and Condition 0 on constraint tasks" and the "per-category" breakdown bullet (sequential vs constrained no longer applies).
- In Section 12 ("IMPLEMENTATION PRIORITY ORDER"), delete item 1 ("Temporal constraint support") and renumber.
- Section 6b ("EPISODE COUNT"): update from 15 tasks * 3 = 45 episodes per condition to whatever the new sequential-only task count yields (currently 6 sequential tasks * 3 = 18 episodes per condition).

The doc edit is mechanical but extensive; the implementation plan should treat it as one task with a careful diff.

### 9. `cached_formulas/`

- Delete every file in `cached_formulas/` as part of the refactor. Old entries are keyed by a prompt-version hash that no longer matches the new prompt set; leaving them in-tree is dead weight.
- Keep the directory itself (recreated empty) so `FORMULA_CACHE_DIR` in `paths.py` continues to resolve.

## Components NOT changed

- `SubgoalConverter` and `SUBGOAL_CONVERSION_PROMPT`. The converter strips "until" / "near" clauses for OpenVLA's short-imperative input, downstream of the planner. That responsibility is unchanged: predicates from the planner still arrive with their full conditional clauses, and the converter still rewrites them for OpenVLA.
- `goal_adherence_monitor.py`'s diary-generation logic (local prompt, displacement tracking, completion-percentage tracking, stall detection, convergence detection). Constraints removal does not touch any of those; only the four prompt-build sites lose their constraint branches.
- The Spot translation path, automaton state advancement, and sink-state mechanics. The new `LTLSymbolicPlanner` is the current `SequentialLTLPlanner` plus the last-goal sink-edge fallback fix. No other automaton-walking changes.

## Verification

After implementation, the following should hold:

- `grep -r "WITH_CONSTRAINTS\|use_constraints\|constraint_predicates\|get_active_constraints\|SequentialLTLPlanner\|ConstraintInfo\|_constraints_block" --include="*.py"` returns no matches in `src/`, `scripts/`, or `tests/`.
- `pytest` runs cleanly with the rewritten `test_condition_ablations.py` and the surviving `test_ltl_planner.py` / `test_parsing.py`.
- A single-task end-to-end run with the example instruction "turn left until you see the tree, then go toward the tree, then turn right until you see the building, and go to the building" produces 4 predicates, each with its conditional clause preserved, and a sequential LTL formula `F pi_4 & (!pi_4 U pi_3) & (!pi_3 U pi_2) & (!pi_2 U pi_1)`.
- `cached_formulas/` is empty (the directory exists but contains no files).
- `experimental_design.txt` contains no mention of constraints, constrained tasks, M2, Section 3, or the constraint-enforcement ablation row.

## Out of scope

- Hardware safety layer (control barrier functions, geofencing, altitude bounds). Section 9 A1 of the experimental design called this out as orthogonal; that remains true and is not affected.
- The Spot library version, automaton determinization mode, or the `monitor det` translate options.
- The OpenVLA server, control loop frequency, async vs sync monitor mode, or any per-step timing instrumentation.
