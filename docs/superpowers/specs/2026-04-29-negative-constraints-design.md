# Negative Constraint Support: Design Spec

## Goal

Add negative constraint support to the LTL-guided UAV system so the drone can enforce avoidance constraints (e.g., "never fly over building C", "stay away from the red car until you reach the streetlight") during subgoal execution. This is the load-bearing feature for the paper's core contribution: demonstrating that LTL formalism provides strict constraint enforcement that ad-hoc LLM planning cannot.

## Architecture

The Spot automaton already encodes constraint semantics in its edge structure. Rather than parsing the LTL formula string, we query the automaton directly to classify predicates and determine which constraints are active at each state.

### Constraint Classification (One-Time)

After `plan_from_natural_language()` builds the automaton, classify each predicate:

- **Goal predicate**: has a forward-progressing edge (dst != src) at some reachable state. These are subgoals the drone must achieve.
- **Constraint predicate**: never has a forward-progressing edge at any state. These represent conditions that must NOT become true.

Algorithm:
```
for each pi_X in pi_map:
    is_goal = False
    for each state S in automaton:
        build BDD where pi_X=true, all others=false
        for each edge from S where dst != S:
            if BDD & edge.cond != false:
                is_goal = True
                break
    if not is_goal:
        mark pi_X as constraint predicate
```

This classification is stored once and does not change during execution.

### Active Constraint Detection (Per State)

Before each subgoal, `get_active_constraints()` checks which constraint predicates are actively enforced at the current automaton state:

- **Active**: making the predicate true produces NO matching edge at all (no self-loop, no forward edge). The automaton would reject this input. Example: `G(!p3)` at any state, or `!p3 U p2` before p2 is achieved.
- **Released**: making the predicate true matches a self-loop edge (dst == src). The constraint's scope has expired. Example: `!p3 U p2` after p2 is achieved.

Algorithm:
```
for each constraint predicate pi_X:
    build BDD where pi_X=true, all others=false
    has_any_edge = False
    for each edge from current_state:
        if BDD & edge.cond != false:
            has_any_edge = True
            break
    if not has_any_edge:
        pi_X is ACTIVE at this state
```

### Prompt Injection

Active constraint NL descriptions are passed to the `LiveDiaryMonitor` constructor via a new `negative_constraints: Optional[List[str]]` parameter (default `None`, empty list internally).

When constraints are present, they are injected into `DIARY_GLOBAL_PROMPT` and `DIARY_CONVERGENCE_PROMPT` after the subgoal line:

```
Active constraints (must be maintained throughout):
  - stay away from building B
  - do not fly over zone C
```

The expected JSON output gains a `constraint_violated: true/false` field. Documentation in the prompt explains when to set it true.

When no constraints are present, the block is empty and the field is omitted from the JSON spec (backward compatible).

### Violation Response Flow

When the VLM reports `constraint_violated: true` during a checkpoint:

1. Monitor returns `force_converge` (stop the drone).
2. Runner enters the convergence correction flow (same as premature stop).
3. The convergence prompt includes the constraint context.
4. VLM issues a corrective command (e.g., "move away from building B").
5. After correction, the drone resumes the original subgoal instruction.
6. Corrections count toward the shared `max_corrections` budget.
7. If budget exhausted, return `ask_help`.

This reuses the existing convergence correction infrastructure with no new control flow.

### Runner Integration

In `run_integrated_control_loop`, before each `_run_subgoal` call:

1. Call `planner.get_active_constraints()` to get NL descriptions.
2. Pass them to `_run_subgoal` via a new `negative_constraints` parameter.
3. `_run_subgoal` passes them to the `LiveDiaryMonitor` constructor.
4. The subgoal result dict gains a `constraint_violated: bool` field for logging.

### LTL Prompt Updates

`LTL_NL_SYSTEM_PROMPT` gains documentation about:
- Constraint predicates (conditions to avoid, not goals to achieve).
- The `G` (Globally) operator: `G(!pi_X)` means pi_X must never become true.
- Scoped avoidance: `!pi_X U pi_Y` where pi_X is a constraint (not a goal).
- How to decide if a predicate is a constraint vs. a goal.
- Constraint predicates should describe the violation condition ("flying over building C"), not the desired behavior ("stay away from building C").

`LTL_NL_EXAMPLES_PROMPT` gains 3 worked examples with avoidance constraints.

### LTL-NL Parser Update

`parse_ltl_nl` in `parsing.py` gains support for the `G` (Globally) unary operator so human-readable formula descriptions include constraint information in logs and artifacts.

## Components Modified

| Component | Change |
|-----------|--------|
| `ltl_planner.py` | Add `_classify_predicates()`, `get_active_constraints()` |
| `prompts.py` | Add constraint docs/examples to LTL prompts; add `{constraints_block}` and `constraint_violated` to diary prompts |
| `diary_monitor.py` | Accept `negative_constraints` param; inject into prompts; handle `constraint_violated` in `_parse_global_response` |
| `run_integration.py` | Extract constraints per subgoal; pass to monitor; log violations |
| `parsing.py` | Handle `G` operator in `parse_ltl_nl` |

## Testing Strategy (Three Tiers)

**Tier 1 (no API calls, free, fast):** Unit tests with mocked LLM responses. Verify automaton-based classification, active constraint detection at different states, prompt injection, JSON parsing of `constraint_violated`, backward compatibility. Run after every code change.

**Tier 2 (single LLM call, pennies):** Prompt smoke tests that send real instructions with avoidance language to the LLM and verify the returned formula contains `G(!pi_X)` or appropriate negation patterns, and that the planner's automaton-based classification identifies the correct constraints. Run once after prompt edits.

**Tier 3 (single VLM call, pennies):** Monitor prompt smoke tests that construct synthetic diary context with constraints, send to the VLM, and verify the response JSON includes `constraint_violated` as a boolean. Run once after prompt edits.

All three tiers must pass before running full-system experiments with the simulator.

## What This Does NOT Cover

- Post-hoc constraint violation analysis for Conditions 1 and 3 (separate script, Section 10 of experimental design).
- Changes to the SubgoalConverter (constraints are handled by the monitor, not embedded in OpenVLA instructions, as specified in Section 3c of the experimental design).
- Task JSON schema changes (the `negative_constraints_expected` field in task JSON is for evaluation bookkeeping, not consumed by the pipeline).
