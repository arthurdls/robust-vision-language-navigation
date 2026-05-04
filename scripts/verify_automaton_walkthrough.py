#!/usr/bin/env python3
"""
Full automaton walkthrough verification for all 15 task instructions.

For each unique task, this script:
  1. Runs the LTL planner (LLM generates formula, Spot builds automaton)
  2. Classifies predicates via the formula-structural classifier
  3. Walks through the automaton by calling get_next_predicate() / advance_state()
     in a loop, simulating a perfect episode where every goal succeeds immediately
  4. Checks get_active_constraints() at each step
  5. Verifies: all goal predicates returned, correct ordering, mission completes,
     no constraint predicate leaked into the goal sequence

Usage:
  conda run -n rvln-sim python scripts/verify_automaton_walkthrough.py
  conda run -n rvln-sim python scripts/verify_automaton_walkthrough.py --model gpt-5.4
"""

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.ai.llm_interface import LLMUserInterface
from rvln.ai.ltl_planner import LTLSymbolicPlanner, _predicate_key_to_index
from rvln.config import DEFAULT_LLM_MODEL
from rvln.paths import REPO_ROOT


def walk_automaton(planner: LTLSymbolicPlanner) -> dict:
    """Step through the automaton, returning a detailed trace."""
    goal_keys = [k for k in planner.pi_map if k not in planner.constraint_predicates]
    constraint_keys = list(planner.constraint_predicates.keys())

    steps = []
    returned_keys = []
    max_iterations = len(planner.pi_map) + 5

    for i in range(max_iterations):
        state_before = planner.current_automaton_state
        active_constraints = planner.get_active_constraints()

        next_goal = planner.get_next_predicate()
        if next_goal is None:
            steps.append({
                "step": i,
                "state": state_before,
                "next_goal": None,
                "active_constraints": [c.description for c in active_constraints],
                "finished": planner.finished,
            })
            break

        current_key = planner._last_returned_predicate_key
        returned_keys.append(current_key)

        steps.append({
            "step": i,
            "state": state_before,
            "next_goal_key": current_key,
            "next_goal_desc": next_goal,
            "active_constraints": [c.description for c in active_constraints],
            "finished": False,
        })

        planner.advance_state(next_goal)

    leaked_constraints = [k for k in returned_keys if k in planner.constraint_predicates]
    missed_goals = [k for k in goal_keys if k not in returned_keys]
    completed = planner.finished

    return {
        "goal_keys_expected": goal_keys,
        "constraint_keys": constraint_keys,
        "returned_sequence": returned_keys,
        "steps": steps,
        "leaked_constraints": leaked_constraints,
        "missed_goals": missed_goals,
        "completed": completed,
        "n_goals_returned": len(returned_keys),
        "n_goals_expected": len(goal_keys),
    }


def verify_all_tasks(model: str):
    tasks_root = REPO_ROOT / "tasks"
    map_dirs = ["downtown_west", "greek_island", "suburb_neighborhood_day"]

    all_files: list[tuple[Path, dict]] = []
    for map_dir in map_dirs:
        task_dir = tasks_root / map_dir
        if not task_dir.is_dir():
            continue
        for jf in sorted(task_dir.glob("*.json")):
            with open(jf) as f:
                data = json.load(f)
            all_files.append((jf, data))

    unique_instructions = {}
    for jf, data in all_files:
        instruction = data["instruction"]
        if instruction not in unique_instructions:
            unique_instructions[instruction] = {
                "task_id": data.get("task_id", jf.stem),
                "expected_subgoal_count": data.get("expected_subgoal_count"),
                "category": data.get("category", "unknown"),
            }

    print(f"Found {len(all_files)} task files, {len(unique_instructions)} unique instructions")
    print(f"Model: {model}")
    print()

    all_results = []
    failures = []

    for instruction, info in unique_instructions.items():
        task_id = info["task_id"]
        expected_goals = info["expected_subgoal_count"]
        category = info["category"]

        print("=" * 80)
        print(f"TASK: {task_id}  (category: {category}, expected goals: {expected_goals})")
        print(f"Instruction: {instruction}")
        print()

        try:
            llm = LLMUserInterface(model=model)
            planner = LTLSymbolicPlanner(llm)
            planner.plan_from_natural_language(instruction)

            formula = planner._raw_formula
            pi_map = dict(planner.pi_map)
            constraints = {k: {"desc": v.description, "polarity": v.polarity}
                           for k, v in planner.constraint_predicates.items()}
            goal_keys = [k for k in pi_map if k not in planner.constraint_predicates]

            print(f"  Formula: {formula}")
            print(f"  Predicates ({len(pi_map)}):")
            for k, v in pi_map.items():
                role = "CONSTRAINT" if k in planner.constraint_predicates else "GOAL"
                extra = ""
                if k in planner.constraint_predicates:
                    extra = f" [{planner.constraint_predicates[k].polarity}]"
                print(f"    {k}: {v}  ({role}{extra})")
            print(f"  Goals: {len(goal_keys)}, Constraints: {len(constraints)}")
            print()

            print("  --- Automaton Walkthrough ---")
            trace = walk_automaton(planner)

            for step in trace["steps"]:
                if step.get("next_goal_key"):
                    ac = step["active_constraints"]
                    ac_str = f"  active_constraints={ac}" if ac else ""
                    print(f"    Step {step['step']}: state={step['state']} -> "
                          f"goal={step['next_goal_key']} "
                          f"(\"{step['next_goal_desc']}\")"
                          f"{ac_str}")
                else:
                    print(f"    Step {step['step']}: state={step['state']} -> "
                          f"None (finished={step['finished']})"
                          f"  active_constraints={step['active_constraints']}")

            print()

            issues = []
            warnings = []
            if trace["leaked_constraints"]:
                issues.append(f"CONSTRAINT LEAKED AS GOAL: {trace['leaked_constraints']}")
            if trace["missed_goals"]:
                issues.append(f"MISSED GOALS: {trace['missed_goals']}")
            if not trace["completed"]:
                issues.append("AUTOMATON DID NOT COMPLETE (finished=False)")
            if expected_goals is not None and trace["n_goals_returned"] != expected_goals:
                warnings.append(
                    f"GOAL COUNT VARIANCE: returned {trace['n_goals_returned']}, "
                    f"JSON expected {expected_goals} (LLM decomposition differs)")

            if issues:
                print(f"  RESULT: FAIL")
                for iss in issues:
                    print(f"    - {iss}")
                failures.append({"task_id": task_id, "issues": issues})
            elif warnings:
                print(f"  RESULT: PASS (returned {trace['n_goals_returned']} goals, "
                      f"mission completed)")
                for w in warnings:
                    print(f"    WARN: {w}")
            else:
                print(f"  RESULT: PASS (returned {trace['n_goals_returned']} goals, "
                      f"mission completed)")

            all_results.append({
                "task_id": task_id,
                "category": category,
                "instruction": instruction,
                "formula": formula,
                "pi_map": pi_map,
                "constraints": constraints,
                "goal_keys": goal_keys,
                "trace": trace,
                "issues": issues,
                "pass": len(issues) == 0,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            failures.append({"task_id": task_id, "issues": [f"EXCEPTION: {e}"]})
            all_results.append({
                "task_id": task_id,
                "category": category,
                "instruction": instruction,
                "error": str(e),
                "pass": False,
            })

        print()

    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    passed = sum(1 for r in all_results if r.get("pass"))
    total = len(all_results)
    print(f"  {passed}/{total} tasks PASSED")

    if failures:
        print()
        print("  FAILURES:")
        for f in failures:
            print(f"    {f['task_id']}:")
            for iss in f["issues"]:
                print(f"      - {iss}")

    output_path = REPO_ROOT / "results" / "automaton_walkthrough_verification.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify automaton walkthrough for all tasks")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="LLM model for planning")
    args = parser.parse_args()
    verify_all_tasks(model=args.model)
