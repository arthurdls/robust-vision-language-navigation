#!/usr/bin/env python3
"""
Integrated LTL planner + GoalAdherenceMonitor control loop (simulation).

Combines the LTL-NL neuro-symbolic planner (multi-step instruction decomposition
via Spot automaton) with the GoalAdherenceMonitor (goal adherence subgoal supervision
with convergence corrections and stall detection).

For each subgoal produced by the planner:
  1. SubgoalConverter rewrites the NL predicate into a short OpenVLA instruction.
  2. A fresh GoalAdherenceMonitor supervises execution with periodic VLM checkpoints,
     completion tracking (including peak_completion), and corrective commands
     on convergence.
  3. When the monitor confirms completion (or the step budget is exhausted),
     the planner advances to the next subgoal.

This replaces the simple GoalAdherenceMonitor used in scripts/run_ltl.py with
the full goal adherence supervision pipeline from scripts/run_goal_adherence.py.

OpenVLA server must be running: python scripts/start_server.py

Usage (from repo root):
  # Single task from tasks/condition0/
  python scripts/run_integration.py --task task.json
  # All tasks in tasks/condition0/ (skips already completed)
  python scripts/run_integration.py --run_all_tasks
  # Ad-hoc command
  python scripts/run_integration.py -c "Go to the tree then land" --initial-position -181,7331,876,-89
  # Custom models and diary parameters
  python scripts/run_integration.py --task first_task.json --llm_model gpt-4o --monitor_model gpt-4o \\
      --diary-check-interval 10 --max-steps-per-subgoal 200 --max-corrections 10
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.config import (
    ACTION_SMALL_DELTA_POS,
    ACTION_SMALL_DELTA_YAW,
    DEFAULT_DIARY_CHECK_INTERVAL,
    DEFAULT_DIARY_CHECK_INTERVAL_S,
    DEFAULT_DIARY_MODE,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_CORRECTIONS,
    DEFAULT_MAX_SECONDS_PER_SUBGOAL,
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
from rvln.eval.subgoal_runner import SubgoalConfig, run_subgoal
from rvln.eval.task_utils import (
    get_completed_task_ids,
    load_eval_task,
    make_ask_help_callback,
    resolve_eval_tasks,
    sanitize_run_label,
)
from rvln.paths import (
    BATCH_SCRIPT,
    REPO_ROOT,
    UAV_FLOW_EVAL,
)
from rvln.sim.env_setup import (
    import_batch_module,
    interactive_camera_select,
    load_env_vars,
    normalize_initial_pos,
    parse_position,
    setup_sim_env,
)

SHARED_TASKS_DIR = REPO_ROOT / "tasks"
CONDITION0_RESULTS_DIR = REPO_ROOT / "results" / "condition0"

logger = logging.getLogger(__name__)


class _SafeEncoder(json.JSONEncoder):
    """Encode dataclass instances and other non-serializable objects."""

    def default(self, o):
        if hasattr(o, "__dataclass_fields__"):
            from dataclasses import asdict
            return asdict(o)
        return super().default(o)




# ---------------------------------------------------------------------------
# Integrated control loop
# ---------------------------------------------------------------------------

def run_integrated_control_loop(
    env: Any,
    batch: Any,
    task: Dict[str, Any],
    server_url: str,
    run_dir: Path,
    llm_model: str,
    monitor_model: str,
    drone_cam_id: int,
    save_mp4: bool = False,
    mp4_fps: float = 10.0,
    check_interval_s: Optional[float] = None,
    max_seconds: Optional[float] = None,
    seed: int = DEFAULT_SEED,
    time_dilation: int = DEFAULT_TIME_DILATION,
    env_id: str = "",
    diary_mode: str = DEFAULT_DIARY_MODE,
) -> Dict[str, Any]:
    """Plan with LTL, then execute each subgoal with goal adherence monitoring.

    Returns the run_info dict written to disk.
    """
    from rvln.ai.llm_interface import LLMUserInterface
    from rvln.ai.ltl_planner import LTLSymbolicPlanner
    from rvln.eval.batch_runner import CUDAOutOfMemoryError

    instruction = task["instruction"]
    initial_pos = normalize_initial_pos(task["initial_pos"])
    max_steps_per_subgoal = task["max_steps_per_subgoal"]
    check_interval = task["diary_check_interval"]
    max_corrections = task["max_corrections"]

    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # --- Teleport drone to initial position ---
    env.teleport(initial_pos[0:3], initial_pos[3])
    time.sleep(batch.SLEEP_AFTER_RESET_S)

    start_ts = datetime.now().isoformat()

    # --- Planning and execution (with replan support) ---
    subgoal_summaries: List[Dict[str, Any]] = []
    trajectory_log: List[Dict[str, Any]] = []
    ltl_plans: List[Dict[str, Any]] = []
    total_frame_count = 0
    subgoal_index = 0
    replan_count = 0
    aborted = False
    completed = False

    origin_x, origin_y, origin_z = initial_pos[0], initial_pos[1], initial_pos[2]
    origin_yaw = initial_pos[3]

    try:
        while True:
            # --- LTL planning phase ---
            logger.info(
                "%sPlanning instruction: '%s'",
                f"[REPLAN #{replan_count}] " if replan_count > 0 else "",
                instruction,
            )
            llm_interface = LLMUserInterface(model=llm_model)
            planner = LTLSymbolicPlanner(llm_interface)
            planner.plan_from_natural_language(instruction)

            ltl_plan = {
                "ltl_nl_formula": llm_interface.ltl_nl_formula.get("ltl_nl_formula", ""),
                "pi_predicates": dict(planner.pi_map),
                "replan_index": replan_count,
                "instruction": instruction,
            }
            ltl_plans.append(ltl_plan)
            logger.info("LTL plan: %s", json.dumps(ltl_plan, indent=2))

            current_subgoal = planner.get_next_predicate()
            if current_subgoal is None:
                logger.error("LTL planning produced no subgoals for instruction: '%s'", instruction)
                if replan_count > 0:
                    logger.warning("Replan produced no subgoals, ending run.")
                    break
                raise RuntimeError("LTL planning produced no subgoals.")

            replan_requested = False

            while current_subgoal is not None:
                subgoal_index += 1
                safe_name = sanitize_run_label(current_subgoal, fallback="subgoal")
                subgoal_dir = run_dir / f"subgoal_{subgoal_index:02d}_{safe_name}"

                logger.info(
                    "--- Subgoal %d: '%s' ---", subgoal_index, current_subgoal,
                )

                try:
                    sg_config = SubgoalConfig(
                        monitor_mode="full",
                        check_interval=check_interval,
                        max_steps=max_steps_per_subgoal,
                        max_corrections=max_corrections,
                        check_interval_s=check_interval_s,
                        max_seconds=max_seconds,
                    )
                    subgoal_result = run_subgoal(
                        env=env,
                        batch=batch,
                        server_url=server_url,
                        subgoal_nl=current_subgoal,
                        monitor_model=monitor_model,
                        llm_model=llm_model,
                        config=sg_config,
                        origin_x=origin_x,
                        origin_y=origin_y,
                        origin_z=origin_z,
                        origin_yaw=origin_yaw,
                        drone_cam_id=drone_cam_id,
                        frames_dir=frames_dir,
                        subgoal_dir=subgoal_dir,
                        frame_offset=total_frame_count,
                        trajectory_log=trajectory_log,
                        ask_help_callback=make_ask_help_callback(),
                    )
                except CUDAOutOfMemoryError as e:
                    logger.error(
                        "Aborting run: OpenVLA server ran out of GPU memory. %s", e,
                    )
                    raise

                total_frame_count += subgoal_result["total_steps"]
                subgoal_summaries.append(subgoal_result)

                sr = subgoal_result["stop_reason"]
                logger.info(
                    "Subgoal %d finished: stop_reason=%s, steps=%d, completion=%.2f",
                    subgoal_index, sr,
                    subgoal_result["total_steps"],
                    subgoal_result.get("last_completion_pct", 0.0),
                )

                # Update world-space origin from wherever the drone ended up
                next_origin = subgoal_result["next_origin"]
                origin_x, origin_y, origin_z, origin_yaw = (
                    next_origin[0], next_origin[1], next_origin[2], next_origin[3],
                )

                if sr in ("abort", "ask_help"):
                    logger.info(
                        "Mission aborted (stop_reason=%s) at subgoal '%s'.",
                        sr, current_subgoal,
                    )
                    aborted = True
                    break

                if sr == "replan":
                    new_instr = subgoal_result["replan_instruction"]
                    replan_count += 1
                    logger.info(
                        "Full replan requested (replan #%d). "
                        "Old instruction: '%s'. New instruction: '%s'. "
                        "Drone origin for replan: [%.1f, %.1f, %.1f, %.1f]",
                        replan_count, instruction, new_instr,
                        origin_x, origin_y, origin_z, origin_yaw,
                    )
                    instruction = new_instr
                    replan_requested = True
                    break

                if sr == "skipped":
                    logger.info("Subgoal '%s' skipped by user, advancing planner.", current_subgoal)

                # Advance planner state (for all non-abort/non-replan outcomes)
                planner.advance_state(current_subgoal)

                current_subgoal = planner.get_next_predicate()
                if current_subgoal is not None:
                    logger.info("Advancing to next subgoal: '%s'", current_subgoal)

            if aborted or not replan_requested:
                break

        completed = not aborted

    finally:
        logger.info(
            "Run finished: %d subgoals processed, %d replan(s), aborted=%s.",
            subgoal_index, replan_count, aborted,
        )

        end_ts = datetime.now().isoformat()

        # --- Save trajectory log ---
        with open(run_dir / "trajectory_log.json", "w") as f:
            json.dump(trajectory_log, f, indent=2)

        # --- Optional playback mp4 ---
        playback_mp4: Optional[Path] = None
        if save_mp4:
            try:
                from rvln.eval.playback import save_run_directory_mp4
                playback_mp4 = save_run_directory_mp4(run_dir, fps=mp4_fps)
            except Exception as e:
                logger.warning("Could not write playback.mp4: %s", e)

        # --- Save run info ---
        total_steps_all = sum(s["total_steps"] for s in subgoal_summaries)
        total_vlm_calls = sum(s.get("vlm_call_count", 0) for s in subgoal_summaries)
        total_corrections = sum(s.get("corrections_used", 0) for s in subgoal_summaries)
        all_vlm_records = []
        for s in subgoal_summaries:
            all_vlm_records.extend(s.get("vlm_call_records", []))
        total_input_tokens = sum(r.get("input_tokens", 0) for r in all_vlm_records)
        total_output_tokens = sum(r.get("output_tokens", 0) for r in all_vlm_records)
        end_dt = datetime.fromisoformat(end_ts)
        start_dt = datetime.fromisoformat(start_ts)
        wall_clock_seconds = (end_dt - start_dt).total_seconds()

        run_info: Dict[str, Any] = {
            "condition": "condition0_full_system",
            "task": task,
            "seed": seed,
            "time_dilation": time_dilation,
            "env_id": env_id,
            "server_url": server_url,
            "drone_cam_id": drone_cam_id,
            "diary_mode": diary_mode,
            "llm_model": llm_model,
            "monitor_model": monitor_model,
            "models": {
                "ltl_nl_planning": llm_model,
                "subgoal_converter": llm_model,
                "goal_adherence_monitor": monitor_model,
                "openvla_predict_url": server_url,
            },
            "config": {
                "max_steps_per_subgoal": task["max_steps_per_subgoal"],
                "diary_check_interval": task["diary_check_interval"],
                "max_corrections": task["max_corrections"],
                "convergence_thresholds": {
                    "position_delta": ACTION_SMALL_DELTA_POS,
                    "yaw_delta": ACTION_SMALL_DELTA_YAW,
                    "consecutive_steps": batch.ACTION_SMALL_STEPS,
                },
            },
            "ltl_plan": ltl_plans[0] if ltl_plans else {},
            "ltl_plans": ltl_plans,
            "replan_count": replan_count,
            "aborted": aborted,
            "completed": completed,
            "subgoal_count": subgoal_index,
            "subgoal_summaries": subgoal_summaries,
            "total_steps": total_steps_all,
            "total_vlm_calls": total_vlm_calls,
            "total_corrections": total_corrections,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "vlm_call_records": all_vlm_records,
            "playback_mp4": str(playback_mp4) if playback_mp4 else None,
            "start_time": start_ts,
            "end_time": end_ts,
            "wall_clock_seconds": round(wall_clock_seconds, 2),
        }
        with open(run_dir / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2, cls=_SafeEncoder)

    return run_info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    load_env_vars()
    parser = argparse.ArgumentParser(
        description="Integrated LTL planner + goal adherence monitor control loop",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "-c", "--command",
        type=str,
        default=None,
        help="Natural language instruction (use with --initial-position)",
    )
    mode.add_argument(
        "--task",
        type=str,
        default=None,
        metavar="TASK.json",
        help="Run single task from tasks/system/",
    )
    mode.add_argument(
        "--run_all_tasks",
        action="store_true",
        help="Run all JSON tasks in tasks/system/",
    )

    parser.add_argument(
        "--initial-position",
        type=str,
        default=None,
        metavar="x,y,z,yaw",
        help="Initial position (for -c mode). Default: %(default)s",
    )

    # Sim / server
    parser.add_argument("-t", "--time_dilation", type=int, default=DEFAULT_TIME_DILATION)
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("-p", "--server_port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--server_host", type=str, default=DEFAULT_SERVER_HOST,
                        help=f"OpenVLA server host (default: {DEFAULT_SERVER_HOST})")
    parser.add_argument("--sim_host", type=str, default=DEFAULT_SIM_HOST,
                        help=f"Simulator host (default: {DEFAULT_SIM_HOST})")
    parser.add_argument("--sim_port", type=int, default=DEFAULT_SIM_PORT,
                        help=f"Simulator UnrealCV port (default: {DEFAULT_SIM_PORT})")
    parser.add_argument("--sim_api_port", type=int, default=DEFAULT_SIM_API_PORT,
                        help=f"Sim API server port (default: {DEFAULT_SIM_API_PORT})")

    # LLM / monitor models
    parser.add_argument(
        "--llm_model",
        default=DEFAULT_LLM_MODEL,
        help=f"LLM for LTL decomposition (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--monitor_model",
        default=DEFAULT_VLM_MODEL,
        help=f"VLM for goal adherence monitor and subgoal converter (default: {DEFAULT_VLM_MODEL})",
    )

    # Diary mode and parameters (override task JSON values)
    parser.add_argument(
        "--diary-mode", choices=("frame", "time"), default=DEFAULT_DIARY_MODE,
        help="Checkpoint mode: 'frame' (sync, every N steps) or 'time' (async, every N seconds). "
             f"Default: {DEFAULT_DIARY_MODE}.",
    )
    parser.add_argument(
        "--diary-check-interval",
        type=int,
        default=None,
        help=f"Steps between diary checkpoints, frame mode (default from task JSON or {DEFAULT_DIARY_CHECK_INTERVAL})",
    )
    parser.add_argument(
        "--diary-check-interval-s",
        type=float,
        default=DEFAULT_DIARY_CHECK_INTERVAL_S,
        help=f"Seconds between diary checkpoints, time mode (default: {DEFAULT_DIARY_CHECK_INTERVAL_S})",
    )
    parser.add_argument(
        "--max-steps-per-subgoal",
        type=int,
        default=None,
        help=f"Max steps per subgoal (default from task JSON or {DEFAULT_MAX_STEPS_PER_SUBGOAL})",
    )
    parser.add_argument(
        "--max-seconds-per-subgoal",
        type=float,
        default=DEFAULT_MAX_SECONDS_PER_SUBGOAL,
        help=f"Max seconds per subgoal, time mode (default: {DEFAULT_MAX_SECONDS_PER_SUBGOAL})",
    )
    parser.add_argument(
        "--max-corrections",
        type=int,
        default=None,
        help=f"Max corrective commands per subgoal (default from task JSON or {DEFAULT_MAX_CORRECTIONS})",
    )

    # Output
    parser.add_argument(
        "-o", "--results_dir",
        default=str(CONDITION0_RESULTS_DIR),
        help="Base directory for results (default: results/condition0)",
    )
    parser.add_argument(
        "--save-mp4",
        action="store_true",
        help="Encode frames to playback.mp4 after each run",
    )
    parser.add_argument(
        "--mp4-fps",
        type=float,
        default=10.0,
        help="Frame rate for --save-mp4 (default: 10)",
    )

    # Camera / logging
    parser.add_argument(
        "--select-cam",
        action="store_true",
        help="Interactively pick the drone camera instead of auto-detecting",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    if not BATCH_SCRIPT.exists():
        logger.error("batch_run_act_all.py not found at %s", BATCH_SCRIPT)
        sys.exit(1)

    batch = import_batch_module()
    os.chdir(str(UAV_FLOW_EVAL))

    server_url = f"http://{args.server_host}:{args.server_port}/predict"

    env = setup_sim_env(int(args.time_dilation), int(args.seed), batch,
                        sim_host=args.sim_host, sim_api_port=args.sim_api_port)
    map_info = env.get_map_info()
    results_base = Path(args.results_dir) / map_info.task_dir_name
    results_base.mkdir(parents=True, exist_ok=True)
    tasks = resolve_eval_tasks(args, map_info, SHARED_TASKS_DIR, overrides={
        "max_steps_per_subgoal": "max_steps_per_subgoal",
        "diary_check_interval": "diary_check_interval",
        "max_corrections": "max_corrections",
    })

    try:
        drone_cam_id = env.drone_cam_id
        if args.select_cam:
            logger.info("Camera selection: pick the camera for OpenVLA.")
            initial_pos_for_cam = (
                tasks[0]["initial_pos"] if tasks
                else normalize_initial_pos(parse_position(map_info.default_position))
            )
            drone_cam_id = interactive_camera_select(env, initial_pos_for_cam, batch)

        completed_ids = get_completed_task_ids(results_base)
        if completed_ids:
            logger.info("Found %d completed task(s) in %s", len(completed_ids), results_base)

        for idx, task in enumerate(tasks):
            task_id = task.get("task_id", "")
            if task_id and task_id in completed_ids:
                logger.info("Skipping already completed task: %s", task_id)
                continue
            logger.info(
                "\n===== Task %d/%d: '%s' =====",
                idx + 1, len(tasks), task["instruction"][:80],
            )
            ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            task_label = task.get("task_id") or sanitize_run_label(task["instruction"], max_len=30, fallback="subgoal")
            run_name = f"c0_full_system__{task_label}__{ts}"
            try:
                run_dir = results_base / run_name
                use_time_mode = args.diary_mode == "time"
                run_info = run_integrated_control_loop(
                    env=env,
                    batch=batch,
                    task=task,
                    server_url=server_url,
                    run_dir=run_dir,
                    llm_model=args.llm_model,
                    monitor_model=args.monitor_model,
                    drone_cam_id=drone_cam_id,
                    save_mp4=args.save_mp4,
                    mp4_fps=args.mp4_fps,
                    check_interval_s=args.diary_check_interval_s if use_time_mode else None,
                    max_seconds=args.max_seconds_per_subgoal if use_time_mode else None,
                    seed=args.seed,
                    time_dilation=args.time_dilation,
                    env_id=map_info.env_id,
                    diary_mode=args.diary_mode,
                )
                logger.info(
                    "Run saved to %s (%d subgoals, %d total steps)",
                    run_dir, run_info["subgoal_count"], run_info["total_steps"],
                )
            except KeyboardInterrupt:
                logger.info("Task interrupted.")
                break
            except Exception as e:
                logger.error("Task failed: %s", e, exc_info=True)
            logger.info("===== Task %d finished =====\n", idx + 1)
    except KeyboardInterrupt:
        logger.info("Interrupted. Exiting.")


if __name__ == "__main__":
    main()
