"""
Shared utilities for condition evaluation scripts.

Consolidates helpers that were previously duplicated across
run_condition*.py and run_all_conditions.py.
"""

import glob
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import (
    DEFAULT_DIARY_CHECK_INTERVAL,
    DEFAULT_MAX_CORRECTIONS,
    DEFAULT_MAX_STEPS_PER_SUBGOAL,
)

logger = logging.getLogger(__name__)


def get_completed_task_ids(results_dir: Path) -> set:
    """Scan a results directory for completed task IDs (runs with run_info.json)."""
    completed = set()
    if not results_dir.is_dir():
        return completed
    for entry in results_dir.iterdir():
        if not entry.is_dir():
            continue
        run_info_path = entry / "run_info.json"
        if not run_info_path.exists():
            continue
        try:
            with open(run_info_path, "r") as f:
                run_info = json.load(f)
            task_id = run_info.get("task", {}).get("task_id", "")
            if task_id:
                completed.add(task_id)
        except Exception:
            continue
    return completed


def load_eval_task(path: Path) -> Dict[str, Any]:
    """Load a task JSON with all fields needed by any condition.

    Returns a dict with unified keys. Each condition script reads
    only the keys it needs; extra keys are harmless.
    """
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Task JSON must be an object")
    instruction = data.get("instruction", "").strip()
    initial_pos = data.get("initial_pos")
    if not instruction:
        raise ValueError("Task JSON must have 'instruction'")
    if not initial_pos or not isinstance(initial_pos, (list, tuple)) or len(initial_pos) < 4:
        raise ValueError("Task JSON must have 'initial_pos' with at least 4 numbers")

    per_subgoal = int(data.get("max_steps_per_subgoal", DEFAULT_MAX_STEPS_PER_SUBGOAL))
    expected_subgoals = int(data.get("expected_subgoal_count", 3))

    result = {
        "instruction": instruction,
        "initial_pos": [float(x) for x in initial_pos],
        "max_steps": per_subgoal * expected_subgoals,
        "max_steps_per_subgoal": per_subgoal,
        "diary_check_interval": int(data.get("diary_check_interval", DEFAULT_DIARY_CHECK_INTERVAL)),
        "max_corrections": int(data.get("max_corrections", DEFAULT_MAX_CORRECTIONS)),
        "expected_subgoal_count": expected_subgoals,
    }
    for passthrough_key in ("task_id", "category", "difficulty", "region", "notes",
                            "constraints_expected"):
        if passthrough_key in data:
            result[passthrough_key] = data[passthrough_key]
    return result


def discover_tasks(tasks_dir: Path) -> List[Dict[str, Any]]:
    """Discover and load all task JSONs from a directory."""
    if not tasks_dir.is_dir():
        logger.warning("No task directory: %s", tasks_dir)
        return []
    json_files = sorted(glob.glob(str(tasks_dir / "*.json")))
    tasks = []
    for jf in json_files:
        try:
            tasks.append(load_eval_task(Path(jf)))
        except Exception as e:
            logger.warning("Skipping %s: %s", jf, e)
    return tasks


def resolve_eval_tasks(
    args,
    map_info,
    tasks_root: Path,
    overrides: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Resolve tasks from CLI args (command, task file, or run_all_tasks).

    ``overrides`` is a dict mapping task-dict keys to argparse attribute names
    for CLI overrides (e.g. {"max_steps_per_subgoal": "max_steps_per_subgoal"}).
    """
    from ..sim.transforms import parse_position
    from ..maps import validate_task_map

    cmd = getattr(args, "command", None)
    task_file = getattr(args, "task", None)
    run_all = getattr(args, "run_all_tasks", False)

    count = sum([1 if cmd else 0, 1 if task_file else 0, 1 if run_all else 0])
    if count == 0:
        raise SystemExit(
            "Specify a task source:\n"
            "  -c \"instruction\" --initial-position x,y,z,yaw\n"
            "  --task TASK.json\n"
            "  --run_all_tasks"
        )
    if count > 1:
        raise SystemExit("At most one of -c/--command, --task, or --run_all_tasks is allowed.")

    overrides = overrides or {}
    tasks_dir = tasks_root / map_info.task_dir_name

    if cmd is not None:
        initial_pos_str = getattr(args, "initial_position", None) or map_info.default_position
        task = {
            "instruction": cmd.strip(),
            "initial_pos": parse_position(initial_pos_str),
        }
        for key, attr in overrides.items():
            val = getattr(args, attr, None)
            if val is not None:
                task[key] = val
        return [task]

    if task_file is not None:
        validate_task_map(task_file, map_info)
        path = Path(task_file)
        if not path.is_absolute():
            if len(path.parts) > 1:
                path = tasks_root / path
            else:
                path = tasks_dir / path.name
        if not path.exists():
            raise SystemExit(f"Task file not found: {path}")
        task = load_eval_task(path)
        _apply_overrides(task, args, overrides)
        return [task]

    tasks_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(glob.glob(str(tasks_dir / "*.json")))
    if not json_files:
        raise SystemExit(f"No JSON files found in {tasks_dir}")
    tasks = []
    for jf in json_files:
        try:
            task = load_eval_task(Path(jf))
            _apply_overrides(task, args, overrides)
            tasks.append(task)
        except Exception as e:
            logger.warning("Skipping %s: %s", jf, e)
    return tasks


def _apply_overrides(task: dict, args, overrides: Dict[str, str]) -> None:
    """Apply CLI argument overrides to a task dict (only if arg value is truthy)."""
    for key, attr in overrides.items():
        val = getattr(args, attr, None)
        if val:
            task[key] = val


def sanitize_run_label(text: str, max_len: int = 40, fallback: str = "task") -> str:
    """Sanitize a string into a filesystem-safe run label."""
    clean = text.lower().replace(" ", "_")
    safe = "".join(c for c in clean if c.isalnum() or c == "_")
    return safe[:max_len] or fallback
