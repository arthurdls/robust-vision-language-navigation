# Code Consolidation: Shared _run_subgoal Logic

## Problem

The condition runner scripts (`run_condition0_baseline.py`, `run_condition2_no_constraints.py`,
`run_condition5_no_monitor.py`) each contain a near-identical `_run_subgoal` function. Changes
to the subgoal loop (step budget, stall detection, correction logic) must be replicated across
all three files manually, which is error-prone.

## Proposed Fix

Extract the shared `_run_subgoal` logic into a single function in a shared module (e.g.,
`src/rvln/runners/subgoal_runner.py` or `scripts/shared_runner.py`). Each condition script
imports and calls it with condition-specific parameters.

## Key Differences Between Conditions

| Aspect | C0 (Baseline) | C2 (No Constraints) | C5 (No Monitor) |
|--------|--------------|---------------------|-----------------|
| Constraints passed to monitor | Yes | No | N/A |
| GoalAdherenceMonitor used | Yes | Yes | No |
| Convergence check | Yes | Yes | Yes |
| Subgoal source | LTL planner | LTL planner | LTL planner |

The differences are small enough to be controlled via flags or parameters rather than
separate implementations.

## Steps

1. Identify the canonical `_run_subgoal` (C0 is the most complete).
2. Parameterize the differences: `use_monitor: bool`, `use_constraints: bool`.
3. Move the function to a shared module.
4. Update C0, C2, and C5 to import and call the shared function.
5. Verify that all three conditions produce identical behavior to their current implementations
   by running a quick test on one task per condition.

## Scope

This is a refactor only. No behavioral changes. Test by diffing episode logs before/after.
