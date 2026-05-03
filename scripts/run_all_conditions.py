#!/usr/bin/env python3
"""
Run all conditions back-to-back on a single map with a persistent simulator.

Tasks are loaded from a SHARED directory: tasks/<map_dir>/
Results go to: results/condition<N>/<map_dir>/<run_name>/

The simulator (run_simulator.py) must already be running with the matching map.

Usage:
  python scripts/run_all_conditions.py --map greek_island
  python scripts/run_all_conditions.py --map downtown_west --conditions 0,1,3
  python scripts/run_all_conditions.py --map greek_island --save-mp4
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from rvln.config import (
    DEFAULT_DIARY_CHECK_INTERVAL,
    DEFAULT_DIARY_MODE,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_CORRECTIONS,
    DEFAULT_MAX_STEPS_PER_SUBGOAL,
    DEFAULT_SEED,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_SIM_API_PORT,
    DEFAULT_SIM_HOST,
    DEFAULT_SIM_PORT,
    DEFAULT_TIME_DILATION,
    DEFAULT_VLM_MODEL,
)
from rvln.maps import SUPPORTED_MAPS
from rvln.paths import BATCH_SCRIPT, REPO_ROOT, UAV_FLOW_EVAL
from rvln.sim.env_setup import (
    import_batch_module,
    load_env_vars,
    setup_sim_env,
)

logger = logging.getLogger(__name__)

ALL_CONDITIONS = list(range(7))

CONDITION_MODULES = {
    0: ("run_integration", "run_integrated_control_loop"),
    1: ("run_condition1_naive", "run_naive_control_loop"),
    2: ("run_condition2_llm_planner", "run_llm_planner_control_loop"),
    3: ("run_condition3_open_loop", "run_open_loop_control_loop"),
    4: ("run_condition4_single_frame", "run_single_frame_control_loop"),
    5: ("run_condition5_grid_only", "run_grid_only_control_loop"),
    6: ("run_condition6_text_only", "run_text_only_control_loop"),
}

CONDITION_PREFIXES = {
    0: "c0_full_system",
    1: "c1_naive",
    2: "c2_llm_planner",
    3: "c3_open_loop",
    4: "c4_single_frame",
    5: "c5_grid_only",
    6: "c6_text_only",
}


def _get_completed_task_ids(results_dir: Path) -> set:
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


def _load_task_full(path: Path) -> Dict[str, Any]:
    """Load a task JSON with all optional fields for any condition."""
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
    result = {
        "instruction": instruction,
        "initial_pos": [float(x) for x in initial_pos],
        "max_steps": int(data.get("max_steps_per_subgoal", DEFAULT_MAX_STEPS_PER_SUBGOAL)),
        "max_steps_per_subgoal": int(data.get("max_steps_per_subgoal", DEFAULT_MAX_STEPS_PER_SUBGOAL)),
        "diary_check_interval": int(data.get("diary_check_interval", DEFAULT_DIARY_CHECK_INTERVAL)),
        "max_corrections": int(data.get("max_corrections", DEFAULT_MAX_CORRECTIONS)),
        "expected_subgoal_count": int(data.get("expected_subgoal_count", 3)),
    }
    for passthrough_key in ("task_id", "category", "difficulty", "region", "notes",
                            "constraints_expected"):
        if passthrough_key in data:
            result[passthrough_key] = data[passthrough_key]
    return result


def _discover_tasks(task_dir_name: str) -> List[Dict[str, Any]]:
    """Discover tasks from the shared tasks/<map_dir>/ directory."""
    tasks_dir = REPO_ROOT / "tasks" / task_dir_name
    if not tasks_dir.is_dir():
        logger.warning("No shared task directory: %s", tasks_dir)
        return []
    json_files = sorted(glob.glob(str(tasks_dir / "*.json")))
    tasks = []
    for jf in json_files:
        try:
            tasks.append(_load_task_full(Path(jf)))
        except Exception as e:
            logger.warning("Skipping %s: %s", jf, e)
    return tasks


def _sanitize_name(text: str, max_len: int = 40) -> str:
    clean = text.lower().replace(" ", "_")
    safe = "".join(c for c in clean if c.isalnum() or c == "_")
    return safe[:max_len] or "task"


def _import_control_loop(condition: int):
    """Import and return the control loop function for a condition."""
    module_name, func_name = CONDITION_MODULES[condition]
    mod = __import__(module_name)
    return getattr(mod, func_name)


def _run_condition_tasks(
    condition: int,
    tasks: List[Dict[str, Any]],
    env,
    batch,
    server_url: str,
    map_info,
    args,
):
    control_loop = _import_control_loop(condition)
    results_dir = Path(args.results_dir) / f"condition{condition}" / map_info.task_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = CONDITION_PREFIXES[condition]
    drone_cam_id = env.drone_cam_id

    completed_ids = _get_completed_task_ids(results_dir)
    if completed_ids:
        logger.info("  Found %d completed task(s) in %s", len(completed_ids), results_dir)

    for idx, task in enumerate(tasks):
        task_id = task.get("task_id", "")
        if task_id and task_id in completed_ids:
            logger.info("  Skipping already completed task: %s", task_id)
            continue

        logger.info(
            "  Task %d/%d: '%s'",
            idx + 1, len(tasks), task["instruction"][:80],
        )
        ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        task_label = task.get("task_id") or _sanitize_name(task["instruction"], max_len=30)
        run_name = f"{prefix}__{task_label}__{ts}"
        run_dir = results_dir / run_name

        common_kwargs = dict(
            env=env,
            batch=batch,
            task=task,
            server_url=server_url,
            run_dir=run_dir,
            drone_cam_id=drone_cam_id,
            save_mp4=args.save_mp4,
            mp4_fps=args.mp4_fps,
            seed=args.seed,
            time_dilation=args.time_dilation,
            env_id=map_info.env_id,
        )

        try:
            if condition == 0:
                run_info = control_loop(
                    **common_kwargs,
                    llm_model=args.llm_model,
                    monitor_model=args.monitor_model,
                    diary_mode=args.diary_mode,
                )
            elif condition == 1:
                run_info = control_loop(**common_kwargs)
            elif condition == 2:
                run_info = control_loop(
                    **common_kwargs,
                    llm_model=args.llm_model,
                    monitor_model=args.monitor_model,
                    diary_mode=args.diary_mode,
                )
            elif condition == 3:
                run_info = control_loop(
                    **common_kwargs,
                    llm_model=args.llm_model,
                    converter_model=args.llm_model,
                )
            elif condition == 4:
                run_info = control_loop(
                    **common_kwargs,
                    llm_model=args.llm_model,
                    monitor_model=args.monitor_model,
                )
            elif condition == 5:
                run_info = control_loop(
                    **common_kwargs,
                    llm_model=args.llm_model,
                    monitor_model=args.monitor_model,
                    diary_mode=args.diary_mode,
                )
            elif condition == 6:
                run_info = control_loop(
                    **common_kwargs,
                    llm_model=args.llm_model,
                    vlm_model=args.vlm_model,
                )

            logger.info(
                "  Run saved to %s (steps=%s, stop=%s)",
                run_dir,
                run_info.get("total_steps", "?"),
                run_info.get("stop_reason", "?"),
            )
        except KeyboardInterrupt:
            logger.info("  Task interrupted.")
            raise
        except Exception as e:
            logger.error("  Task failed: %s", e, exc_info=True)


def main():
    load_env_vars()
    parser = argparse.ArgumentParser(
        description="Run all conditions back-to-back on a single map",
    )
    parser.add_argument("--map", type=str, required=True,
                        help="Map task_dir_name (e.g. greek_island, downtown_west)")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated condition numbers to run (default: all 0-6)")
    parser.add_argument("-t", "--time_dilation", type=int, default=DEFAULT_TIME_DILATION)
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("-p", "--server_port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--server_host", type=str, default=DEFAULT_SERVER_HOST)
    parser.add_argument("--sim_host", type=str, default=DEFAULT_SIM_HOST)
    parser.add_argument("--sim_port", type=int, default=DEFAULT_SIM_PORT)
    parser.add_argument("--sim_api_port", type=int, default=DEFAULT_SIM_API_PORT)
    parser.add_argument("--llm_model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--monitor_model", type=str, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--vlm_model", type=str, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--diary_mode", type=str, default=DEFAULT_DIARY_MODE)
    parser.add_argument("--results_dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--mp4-fps", type=float, default=10.0)
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    if args.conditions:
        conditions = [int(c.strip()) for c in args.conditions.split(",")]
    else:
        conditions = ALL_CONDITIONS

    map_task_dir = args.map
    matching_map = None
    for m in SUPPORTED_MAPS.values():
        if m.task_dir_name == map_task_dir:
            matching_map = m
            break
    if matching_map is None:
        valid = ", ".join(m.task_dir_name for m in SUPPORTED_MAPS.values())
        raise SystemExit(f"Unknown map '{map_task_dir}'. Valid task_dir_names: {valid}")

    if not BATCH_SCRIPT.exists():
        logger.error("batch_run_act_all.py not found at %s", BATCH_SCRIPT)
        sys.exit(1)

    batch = import_batch_module()
    os.chdir(str(UAV_FLOW_EVAL))

    server_url = f"http://{args.server_host}:{args.server_port}/predict"

    env = setup_sim_env(
        int(args.time_dilation), int(args.seed), batch,
        sim_host=args.sim_host, sim_api_port=args.sim_api_port,
    )
    map_info = env.get_map_info()

    if map_info.task_dir_name != map_task_dir:
        raise SystemExit(
            f"Map mismatch: --map {map_task_dir} but simulator is running "
            f"'{map_info.name}' (task_dir={map_info.task_dir_name}).\n"
            f"Restart the simulator with: python scripts/run_simulator.py --scene {matching_map.name}"
        )

    logger.info("Connected to simulator: map=%s", map_info.name)
    logger.info("Conditions to run: %s", conditions)

    # All conditions use the SAME shared tasks
    tasks = _discover_tasks(map_task_dir)
    if not tasks:
        raise SystemExit(f"No tasks found in tasks/{map_task_dir}/")
    logger.info("Found %d shared task(s) for map '%s'", len(tasks), map_task_dir)

    try:
        for cond in conditions:
            if cond not in CONDITION_MODULES:
                logger.warning("Unknown condition %d, skipping", cond)
                continue

            logger.info(
                "\n===== Condition %d (%s): %d task(s) =====",
                cond, CONDITION_PREFIXES[cond], len(tasks),
            )

            _run_condition_tasks(
                condition=cond,
                tasks=tasks,
                env=env,
                batch=batch,
                server_url=server_url,
                map_info=map_info,
                args=args,
            )

            logger.info("===== Condition %d finished =====\n", cond)
    except KeyboardInterrupt:
        logger.info("\nInterrupted. Exiting.")


if __name__ == "__main__":
    main()
