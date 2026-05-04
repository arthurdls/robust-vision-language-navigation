#!/usr/bin/env python3
"""
Precompute LTL formulas for every task JSON under tasks/.

Reads each task's instruction, runs the LTL planner LLM call, and writes the
result into ``cached_formulas/{hash}.json``. Subsequent ``run_*`` invocations
will read from the cache instead of re-calling the planner LLM, which:
  - keeps the formula stable across the 3 starting-position variants of each
    task (otherwise mild OpenAI nondeterminism at temperature=0 can vary the
    formula between variants);
  - removes one LLM call per episode (45 per condition x 7 conditions);
  - makes experiments reproducible after the cache is committed to git.

Usage::

    python tools/precompute_formulas.py
    python tools/precompute_formulas.py --refresh         # ignore the cache
    python tools/precompute_formulas.py --tasks tasks/    # custom task root
    python tools/precompute_formulas.py --model gpt-4o    # override LLM model
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.config import DEFAULT_LLM_MODEL
from rvln.paths import FORMULA_CACHE_DIR, TASKS_DIR, load_env_vars

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(TASKS_DIR),
                        help="Root tasks directory (default: tasks/)")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL,
                        help="LLM model (must match the one used at runtime)")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore existing cache entries and recompute")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(name)s - %(message)s",
    )
    load_env_vars()

    if args.refresh:
        os.environ["RVLN_IGNORE_FORMULA_CACHE"] = "1"

    from rvln.ai.llm_interface import LLMUserInterface
    from rvln.ai.ltl_planner import LTLSymbolicPlanner

    tasks_root = Path(args.tasks).resolve()
    if not tasks_root.is_dir():
        raise SystemExit(f"Tasks directory not found: {tasks_root}")

    paths = sorted(glob.glob(str(tasks_root / "**" / "*.json"), recursive=True))
    if not paths:
        raise SystemExit(f"No task JSON files under {tasks_root}")

    # Deduplicate by instruction so the 3 starting-position variants of each
    # task contribute one cache entry, not three.
    seen = set()
    instructions = []
    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
            instr = (data.get("instruction") or "").strip()
        except Exception as e:
            logger.warning("Skipping %s: %s", p, e)
            continue
        if not instr or instr in seen:
            continue
        seen.add(instr)
        instructions.append((p, instr))

    logger.info(
        "Pre-computing %d unique instructions from %d task files (model=%s)",
        len(instructions), len(paths), args.model,
    )

    FORMULA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n_ok, n_fail = 0, 0
    for path, instr in instructions:
        try:
            iface = LLMUserInterface(model=args.model)
            planner = LTLSymbolicPlanner(iface)
            planner.plan_from_natural_language(instr)
            n_ok += 1
            logger.info("  OK  %s -> %s", Path(path).name,
                        iface.ltl_nl_formula.get("ltl_nl_formula", ""))
        except Exception as e:
            n_fail += 1
            logger.error("  FAIL %s: %s", Path(path).name, e)

    logger.info("Done: %d cached, %d failed. Cache dir: %s",
                n_ok, n_fail, FORMULA_CACHE_DIR)


if __name__ == "__main__":
    main()
