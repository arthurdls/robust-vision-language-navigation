#!/usr/bin/env python3
"""
Validate expected_subgoal_count in task JSONs by running the LTL planner.

For each unique task instruction, runs the LTL planner (via LLMUserInterface +
LTLSymbolicPlanner) and counts how many goal predicates (non-constraint) the
automaton produces. Compares to the expected_subgoal_count in the JSON.

Usage:
  python scripts/validate_subgoal_counts.py
  python scripts/validate_subgoal_counts.py --model gpt-5.4
  python scripts/validate_subgoal_counts.py --fix   # update JSONs with actual counts
"""

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.ai.llm_interface import LLMUserInterface
from rvln.ai.ltl_planner import LTLSymbolicPlanner
from rvln.config import DEFAULT_LLM_MODEL
from rvln.paths import REPO_ROOT


def count_goal_predicates(planner: LTLSymbolicPlanner) -> int:
    """Count goal predicates (total predicates minus constraints)."""
    total = len(planner.pi_map)
    constraints = len(planner.constraint_predicates)
    return total - constraints


def validate_all_tasks(model: str, fix: bool = False):
    tasks_root = REPO_ROOT / "tasks"
    map_dirs = ["downtown_west", "greek_island", "suburb_neighborhood_day"]

    seen_instructions: dict[str, dict] = {}
    all_files: list[tuple[Path, dict]] = []

    for map_dir in map_dirs:
        task_dir = tasks_root / map_dir
        if not task_dir.is_dir():
            print(f"  [SKIP] {task_dir} not found")
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
                "expected": data.get("expected_subgoal_count"),
                "files": [],
            }
        unique_instructions[instruction]["files"].append((jf, data))

    print(f"Found {len(all_files)} task files, {len(unique_instructions)} unique instructions")
    print(f"Using model: {model}")
    print()

    mismatches = []
    results = []

    for instruction, info in unique_instructions.items():
        task_id = info["task_id"]
        expected = info["expected"]

        print(f"--- {task_id} (expected: {expected}) ---")
        print(f"  Instruction: {instruction[:100]}...")

        try:
            llm = LLMUserInterface(model=model)
            planner = LTLSymbolicPlanner(llm)
            planner.plan_from_natural_language(instruction)

            actual_goals = count_goal_predicates(planner)
            n_constraints = len(planner.constraint_predicates)
            n_total = len(planner.pi_map)
            formula = planner._raw_formula

            print(f"  Formula: {formula}")
            print(f"  Predicates: {planner.pi_map}")
            if planner.constraint_predicates:
                print(f"  Constraints: {list(planner.constraint_predicates.keys())}")
            print(f"  Total predicates: {n_total}, Goals: {actual_goals}, Constraints: {n_constraints}")

            status = "OK" if expected == actual_goals else "MISMATCH"
            if expected != actual_goals:
                mismatches.append({
                    "task_id": task_id,
                    "expected": expected,
                    "actual": actual_goals,
                    "formula": formula,
                })
            print(f"  Result: {status} (expected={expected}, actual={actual_goals})")

            results.append({
                "task_id": task_id,
                "instruction": instruction,
                "expected": expected,
                "actual_goals": actual_goals,
                "n_constraints": n_constraints,
                "formula": formula,
                "predicates": planner.pi_map,
                "constraint_keys": list(planner.constraint_predicates.keys()),
            })

            if fix and expected != actual_goals:
                for jf, data in info["files"]:
                    data["expected_subgoal_count"] = actual_goals
                    with open(jf, "w") as f:
                        json.dump(data, f, indent=2)
                        f.write("\n")
                    print(f"  FIXED: {jf.name} -> {actual_goals}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "task_id": task_id,
                "instruction": instruction,
                "expected": expected,
                "error": str(e),
            })

        print()

    print("=" * 80)
    print(f"SUMMARY: {len(results)} tasks validated, {len(mismatches)} mismatches")
    if mismatches:
        print()
        print("MISMATCHES:")
        for m in mismatches:
            direction = "TOO LOW" if m["expected"] < m["actual"] else "TOO HIGH"
            print(f"  {m['task_id']}: expected={m['expected']}, actual={m['actual']} ({direction})")
            print(f"    Formula: {m['formula']}")
    else:
        print("All expected_subgoal_count values match!")

    output_path = REPO_ROOT / "results" / "subgoal_count_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"results": results, "mismatches": mismatches}, f, indent=2)
    print(f"\nDetailed results written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate expected_subgoal_count in task JSONs")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="LLM model for planning")
    parser.add_argument("--fix", action="store_true", help="Update JSONs with actual counts")
    args = parser.parse_args()
    validate_all_tasks(model=args.model, fix=args.fix)
