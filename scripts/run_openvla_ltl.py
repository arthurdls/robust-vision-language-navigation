#!/usr/bin/env python3
"""
Run OpenVLA simulation with LTL-based symbolic planning and goal adherence monitoring.

Uses the same env/setup as start_openvla_sim.py but runs a custom control loop that:
- Breaks the task instruction into LTL subgoals and feeds one subgoal at a time to OpenVLA.
- When OpenVLA reports done, verifies the subgoal via a goal monitor before advancing.
- Saves every camera frame sent to the model and run metadata under results/ltl_results/.

OpenVLA server must be running: python scripts/start_openvla_server.py

Usage (from repo root):
  # Ad-hoc command (require --initial-position)
  python scripts/run_openvla_ltl.py -c "Go to the red building..." --initial-position 100,100,100,61
  # Single task from tasks/ltl_tasks/
  python scripts/run_openvla_ltl.py --task first_task.json
  # All tasks in tasks/ltl_tasks/
  python scripts/run_openvla_ltl.py --run_all_tasks
  # Skip camera selection and use default camera:
  python scripts/run_openvla_ltl.py --task first_task.json --use-default-cam
  # Interactive: prompt for task name in CLI (no initial task). Use --use-default-cam to skip camera selection:
  python scripts/run_openvla_ltl.py --interactive --use-default-cam
  # Or run one task then stay in interactive mode to rerun or run another:
  python scripts/run_openvla_ltl.py --task third_task.json --interactive --use-default-cam
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from sim_common import (
    BATCH_SCRIPT,
    DEFAULT_INITIAL_POSITION,
    DEFAULT_SERVER_PORT,
    DEFAULT_SEED,
    DEFAULT_TIME_DILATION,
    DOWNTOWN_ENV_ID,
    DRONE_CAM_ID,
    REPO_ROOT,
    UAV_FLOW_EVAL,
    apply_action_poses,
    import_batch_module,
    interactive_camera_select,
    load_env_vars,
    normalize_initial_pos,
    parse_position,
    relative_pose_to_world,
    set_drone_cam_and_get_image,
    setup_env_and_imports,
    setup_sim_env,
    state_for_openvla,
)

LTL_TASKS_DIR = REPO_ROOT / "tasks" / "ltl_tasks"
LTL_RESULTS_DIR = REPO_ROOT / "results" / "ltl_results"

_TASK_NOT_FOUND = object()

DEFAULT_MAX_STEPS = 100
IMAGE_HISTORY_LEN = 10
GOAL_MONITOR_PERIODIC_STEPS = 30
SMALL_DELTA_POS = 3.0
SMALL_DELTA_YAW = 1.0

logger = logging.getLogger(__name__)


def run_ltl_control_loop(
    initial_pos: List[float],
    env: Any,
    full_instruction: str,
    max_steps: Optional[int],
    trajectory_log: List[Dict[str, Any]],
    server_url: str,
    batch: Any,
    planner: Any,
    goal_monitor: Any,
    reset_model_fn: Any,
    run_dir: Optional[Path] = None,
    subgoals_out: Optional[List[str]] = None,
    goal_adherence_on: bool = False,
    drone_cam_id: Optional[int] = None,
    set_cam_at_start: bool = True,
) -> None:
    """
    LTL-aware control loop: plan from full_instruction, run subgoals one at a time,
    verify with goal monitor when OpenVLA reports done, advance or retry.
    If run_dir is set, saves every frame sent to the model under run_dir/frames/.
    """
    full_instruction = full_instruction.strip().lower()
    initial_pos = normalize_initial_pos(initial_pos)
    initial_x, initial_y, initial_z = initial_pos[0:3]
    initial_yaw = initial_pos[4]
    env.unwrapped.unrealcv.set_obj_location(
        env.unwrapped.player_list[0], initial_pos[0:3]
    )
    env.unwrapped.unrealcv.set_rotation(
        env.unwrapped.player_list[0], initial_pos[4] - 180
    )
    if set_cam_at_start:
        batch.set_cam(env)
        time.sleep(batch.SLEEP_AFTER_RESET_S)
    cam_id = drone_cam_id if drone_cam_id is not None else DRONE_CAM_ID
    image = set_drone_cam_and_get_image(env, cam_id)

    subgoal_origin_x, subgoal_origin_y = initial_x, initial_y
    subgoal_origin_z, subgoal_origin_yaw = initial_z, initial_yaw
    current_pose: List[float] = [0.0, 0.0, 0.0, 0.0]
    last_pose = None
    small_count = 0
    step_count = 0
    image_history = deque(maxlen=IMAGE_HISTORY_LEN)
    frame_index = 0
    subgoals_used = subgoals_out if subgoals_out is not None else []

    planner.plan_from_natural_language(full_instruction)
    current_subgoal = planner.get_next_predicate()
    if current_subgoal is None:
        raise RuntimeError(
            "LTL planning produced no subgoals. Fix the instruction or planner "
            "(e.g. ensure predicate descriptions are unique)."
        )

    logger.info("Start LTL control loop. First subgoal: %s", current_subgoal)

    while current_subgoal is not None:
        batch.set_cam(env)
        if image is not None:
            image_history.append(image.copy())

        if image is None:
            logger.warning("No image, ending control loop.")
            break

        if run_dir is not None and image is not None:
            frames_dir = run_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frames_dir / "frame_{:06d}.png".format(frame_index)
            try:
                import cv2
                cv2.imwrite(str(frame_path), image)
            except Exception as e:
                logger.debug("Failed to save frame %s: %s", frame_path, e)
            frame_index += 1

        response = batch.send_prediction_request(
            image=Image.fromarray(image),
            proprio=state_for_openvla(current_pose),
            instr=(current_subgoal or "").lower(),
            server_url=server_url,
        )

        if response is None:
            logger.warning("No valid response, ending control.")
            break

        action_poses_data = response.get("action")
        if not isinstance(action_poses_data, list) or len(action_poses_data) == 0:
            logger.warning("Response 'action' empty or invalid, stopping.")
            break

        try:
            new_image, current_pose, steps_added = apply_action_poses(
                env,
                action_poses_data,
                subgoal_origin_x,
                subgoal_origin_y,
                subgoal_origin_z,
                subgoal_origin_yaw,
                batch.set_cam,
                trajectory_log=trajectory_log,
                sleep_s=0.1,
                drone_cam_id=cam_id,
            )
        except Exception as e:
            logger.error("Error executing action: %s", e)
            break

        step_count += steps_added
        if new_image is not None:
            image = new_image

        pose_now = current_pose
        advanced_this_iteration = False
        if last_pose is not None:
            diffs = [abs(a - b) for a, b in zip(pose_now, last_pose)]
            if (
                all(d < SMALL_DELTA_POS for d in diffs[:3])
                and diffs[3] < SMALL_DELTA_YAW
            ):
                small_count += 1
            else:
                small_count = 0
            if small_count >= batch.ACTION_SMALL_STEPS:
                logger.info(
                    "Detected %d steps with very small change, ending current subgoal.",
                    batch.ACTION_SMALL_STEPS,
                )
                subgoals_used.append(current_subgoal)
                planner.advance_state(current_subgoal)
                current_subgoal = planner.get_next_predicate()
                reset_model_fn(server_url)
                subgoal_origin_x, subgoal_origin_y, subgoal_origin_z, subgoal_origin_yaw = relative_pose_to_world(
                    subgoal_origin_x, subgoal_origin_y, subgoal_origin_z, subgoal_origin_yaw, current_pose
                )
                current_pose = [0.0, 0.0, 0.0, 0.0]
                small_count = 0
                step_count = 0
                advanced_this_iteration = True
                if current_subgoal is None:
                    logger.info("No more subgoals. Task complete.")
                    break
                logger.info("Next subgoal: %s", current_subgoal)
        last_pose = pose_now

        if not advanced_this_iteration and max_steps is not None and step_count >= max_steps:
            logger.info("Reached max_steps %d for current subgoal, advancing.", max_steps)
            subgoals_used.append(current_subgoal)
            planner.advance_state(current_subgoal)
            current_subgoal = planner.get_next_predicate()
            reset_model_fn(server_url)
            subgoal_origin_x, subgoal_origin_y, subgoal_origin_z, subgoal_origin_yaw = relative_pose_to_world(
                subgoal_origin_x, subgoal_origin_y, subgoal_origin_z, subgoal_origin_yaw, current_pose
            )
            current_pose = [0.0, 0.0, 0.0, 0.0]
            small_count = 0
            step_count = 0
            advanced_this_iteration = True
            if current_subgoal is None:
                logger.info("No more subgoals. Task complete.")
                break
            logger.info("Next subgoal: %s", current_subgoal)

        if not advanced_this_iteration and goal_adherence_on and step_count % GOAL_MONITOR_PERIODIC_STEPS == 0 and step_count > 0:
            result = goal_monitor.check(
                list(image_history),
                current_subgoal,
                full_goal=full_instruction,
                model_claimed_done=False,
            )
            if result.goal_achieved:
                logger.info("Periodic check: full goal achieved.")
                break

        if not advanced_this_iteration and response.get("done") is True:
            if goal_adherence_on:
                result = goal_monitor.check(
                    list(image_history),
                    current_subgoal,
                    full_goal=full_instruction,
                    model_claimed_done=True,
                )
                if result.subgoal_achieved:
                    logger.info("Subgoal verified: %s. Advancing planner.", current_subgoal)
                    subgoals_used.append(current_subgoal)
                    planner.advance_state(current_subgoal)
                    current_subgoal = planner.get_next_predicate()
                    reset_model_fn(server_url)
                    subgoal_origin_x, subgoal_origin_y, subgoal_origin_z, subgoal_origin_yaw = relative_pose_to_world(
                        subgoal_origin_x, subgoal_origin_y, subgoal_origin_z, subgoal_origin_yaw, current_pose
                    )
                    current_pose = [0.0, 0.0, 0.0, 0.0]
                    step_count = 0
                    if current_subgoal is None:
                        logger.info("No more subgoals. Task complete.")
                        break
                else:
                    if result.suggest_retry:
                        logger.info(
                            "Goal monitor: subgoal not achieved, suggest retry. Continuing with same subgoal."
                        )
            else:
                logger.info("Model reported done. Advancing planner (goal adherence off).")
                subgoals_used.append(current_subgoal)
                planner.advance_state(current_subgoal)
                current_subgoal = planner.get_next_predicate()
                reset_model_fn(server_url)
                subgoal_origin_x, subgoal_origin_y, subgoal_origin_z, subgoal_origin_yaw = relative_pose_to_world(
                    subgoal_origin_x, subgoal_origin_y, subgoal_origin_z, subgoal_origin_yaw, current_pose
                )
                current_pose = [0.0, 0.0, 0.0, 0.0]
                step_count = 0
                if current_subgoal is None:
                    logger.info("No more subgoals. Task complete.")
                    break


def _load_task_from_json(path: Path) -> Dict[str, Any]:
    """Load ltl task JSON: instruction and initial_pos (4 or 5 elements)."""
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Task JSON must be an object")
    instruction = data.get("instruction", "").strip().lower()
    initial_pos = data.get("initial_pos")
    if not instruction:
        raise ValueError("Task JSON must have 'instruction'")
    if not initial_pos or not isinstance(initial_pos, (list, tuple)) or len(initial_pos) < 4:
        raise ValueError("Task JSON must have 'initial_pos' with 4 or 5 numbers (x,y,z,yaw)")
    return {"instruction": instruction, "initial_pos": [float(x) for x in initial_pos]}


def _resolve_task_name_to_task(cli_input: str) -> Any:
    """
    Resolve CLI input to a single task dict for interactive rerun.
    Returns None if user chose quit (empty, 'quit', 'exit', 'q').
    Returns _TASK_NOT_FOUND if the task file does not exist (caller should re-prompt).
    Otherwise returns the task dict from _load_task_from_json.
    """
    raw = (cli_input or "").strip()
    if not raw or raw.lower() in ("quit", "exit", "q"):
        return None
    name = raw if raw.endswith(".json") else raw + ".json"
    path = LTL_TASKS_DIR / name
    if not path.exists():
        logger.error("Task file not found: %s", path)
        return _TASK_NOT_FOUND
    try:
        return _load_task_from_json(path)
    except Exception as e:
        logger.warning("Failed to load task from %s: %s", path, e)
        return _TASK_NOT_FOUND


def _resolve_tasks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Build list of tasks from -c, --task, --run_all_tasks (at most one), or --interactive with no task."""
    cmd = getattr(args, "command", None)
    task_file = getattr(args, "task", None)
    run_all = getattr(args, "run_all_tasks", False)
    interactive = getattr(args, "interactive", False)

    count = sum([1 if cmd else 0, 1 if task_file else 0, 1 if run_all else 0])
    if count == 0:
        if interactive:
            return []
        raise SystemExit(
            "Specify a task or use --interactive to enter the task at the prompt.\n"
            "  -c \"instruction\" [--initial-position x,y,z,yaw]  run one ad-hoc task\n"
            "  --task first_task.json                           run one task from tasks/ltl_tasks/\n"
            "  --run_all_tasks                                  run all JSONs in tasks/ltl_tasks/\n"
            "  --interactive                                    no initial task; prompt for task name in CLI"
        )
    if count > 1:
        raise SystemExit(
            "At most one of -c/--command, --task, or --run_all_tasks is allowed.\n"
            "  -c \"instruction\" [--initial-position x,y,z,yaw]  run one ad-hoc task (position optional)\n"
            "  --task first_task.json                           run one task from tasks/ltl_tasks/\n"
            "  --run_all_tasks                                  run all JSONs in tasks/ltl_tasks/"
        )

    if cmd is not None:
        initial_pos_str = getattr(args, "initial_position", None) or DEFAULT_INITIAL_POSITION
        return [{"instruction": cmd.strip().lower(), "initial_pos": parse_position(initial_pos_str)}]

    if task_file is not None:
        path = Path(task_file)
        if not path.is_absolute():
            path = LTL_TASKS_DIR / path.name
        if not path.exists():
            raise SystemExit("Task file not found: {}".format(path))
        return [_load_task_from_json(path)]

    LTL_TASKS_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(glob.glob(str(LTL_TASKS_DIR / "*.json")))
    if not json_files:
        raise SystemExit("No JSON files found in {}".format(LTL_TASKS_DIR))
    tasks = []
    for jf in json_files:
        try:
            tasks.append(_load_task_from_json(Path(jf)))
        except Exception as e:
            logger.warning("Skipping %s: %s", jf, e)
    return tasks


def _run_single_task(
    env: Any,
    task: Dict[str, Any],
    batch: Any,
    planner: Any,
    goal_monitor: Any,
    server_url: str,
    results_base: Path,
    args: argparse.Namespace,
    run_name: str,
    drone_cam_id: int,
    set_cam_at_start: bool = True,
) -> Path:
    """Run one task (LTL loop + save trajectory_log and run_info). Returns run_dir."""
    instruction = task["instruction"]
    initial_pos = task["initial_pos"]
    start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = results_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    batch.reset_model(server_url)
    logger.info("instruction: %s", instruction)

    trajectory_log: List[Dict[str, Any]] = []
    subgoals_used: List[str] = []

    run_ltl_control_loop(
        initial_pos,
        env,
        instruction,
        args.max_steps,
        trajectory_log,
        server_url,
        batch,
        planner,
        goal_monitor,
        batch.reset_model,
        run_dir=run_dir,
        subgoals_out=subgoals_used,
        goal_adherence_on=args.goal_adherence_on,
        drone_cam_id=drone_cam_id,
        set_cam_at_start=set_cam_at_start,
    )

    with open(run_dir / "trajectory_log.json", "w") as f:
        json.dump(trajectory_log, f, indent=2)
    end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    run_info = {
        "instruction": instruction,
        "initial_pos": initial_pos,
        "start_time": start_time,
        "end_time": end_time,
        "subgoals": subgoals_used,
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    return run_dir


def main():
    load_env_vars()
    parser = argparse.ArgumentParser(
        description="Run OpenVLA with LTL planning and goal adherence monitoring"
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "-c",
        "--command",
        type=str,
        default=None,
        help="Natural language instruction (optional --initial-position)",
    )
    mode.add_argument(
        "--task",
        type=str,
        default=None,
        metavar="TASK.json",
        help="Run single task from tasks/ltl_tasks/ (e.g. first_task.json)",
    )
    mode.add_argument(
        "--run_all_tasks",
        action="store_true",
        help="Run all JSON tasks in tasks/ltl_tasks/",
    )
    parser.add_argument(
        "--initial-position",
        type=str,
        default=DEFAULT_INITIAL_POSITION,
        metavar="x,y,z,yaw",
        help="Initial position as comma-separated x,y,z,yaw (supports negatives). Default: %(default)s",
    )
    parser.add_argument(
        "-e",
        "--env_id",
        default=DOWNTOWN_ENV_ID,
        help="Environment ID",
    )
    parser.add_argument(
        "-t",
        "--time_dilation",
        type=int,
        default=DEFAULT_TIME_DILATION,
        help="Time dilation for simulator",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed",
    )
    parser.add_argument(
        "-o",
        "--results_dir",
        default=str(LTL_RESULTS_DIR),
        help="Base directory for run outputs (default: results/ltl_results)",
    )
    parser.add_argument(
        "-p",
        "--server_port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="OpenVLA server port",
    )
    parser.add_argument(
        "-m",
        "--max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Max inference steps per task",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--llm_model",
        default="gpt-4o-mini",
        help="LLM for LTL decomposition (and goal monitor if not set)",
    )
    parser.add_argument(
        "--goal_monitor_model",
        default=None,
        help="VLM for goal monitor (default: same as --llm_model)",
    )
    parser.add_argument(
        "--goal-adherence-on",
        dest="goal_adherence_on",
        action="store_true",
        default=False,
        help="Enable goal adherence monitor (off by default).",
    )
    parser.add_argument(
        "--use-default-cam",
        action="store_true",
        help="Use the default drone camera (ID %d) and skip interactive camera selection." % DRONE_CAM_ID,
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Wait for CLI input to run tasks: enter a task name, or 'quit' to exit.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    tasks = _resolve_tasks(args)

    if not BATCH_SCRIPT.exists():
        logger.error("batch_run_act_all.py not found at %s", BATCH_SCRIPT)
        sys.exit(1)

    setup_env_and_imports()
    batch = import_batch_module()

    from modules.goal_monitor import GoalAdherenceMonitor
    from modules.llm_user_interface import LLM_User_Interface
    from modules.ltl_planner import LTL_Symbolic_Planner

    os.chdir(str(UAV_FLOW_EVAL))
    server_url = "http://127.0.0.1:{}".format(args.server_port) + "/predict"
    results_base = Path(args.results_dir)
    results_base.mkdir(parents=True, exist_ok=True)

    env = setup_sim_env(args.env_id, int(args.time_dilation), int(args.seed), batch)

    llm_interface = LLM_User_Interface(model=args.llm_model)
    planner = LTL_Symbolic_Planner(llm_interface)
    goal_monitor_model = args.goal_monitor_model or args.llm_model
    goal_monitor = GoalAdherenceMonitor(model=goal_monitor_model)

    drone_cam_id = DRONE_CAM_ID
    if not args.use_default_cam:
        logger.info("Camera selection: stopping before main loop. Use the window to pick the camera.")
        initial_pos_for_cam = (
            tasks[0]["initial_pos"] if tasks
            else normalize_initial_pos(parse_position(DEFAULT_INITIAL_POSITION))
        )
        drone_cam_id = interactive_camera_select(env, initial_pos_for_cam, batch)

    for idx, task in enumerate(tasks):
        logger.info(
            "\n===== Task %d/%d =====",
            idx + 1,
            len(tasks),
        )
        start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_name = "run_{}".format(start_time) if len(tasks) == 1 else "run_{}_{:02d}".format(start_time, idx)
        try:
            run_dir = _run_single_task(
                env,
                task,
                batch,
                planner,
                goal_monitor,
                server_url,
                results_base,
                args,
                run_name,
                drone_cam_id,
                set_cam_at_start=(idx == 0),
            )
            logger.info("Run saved to %s", run_dir)
        except KeyboardInterrupt:
            logger.info("Task interrupted.")
            if args.interactive:
                break
            raise
        logger.info("===== Task %d finished =====\n", idx + 1)

    if args.interactive:
        while True:
            try:
                user_input = input(
                    "Enter task name (e.g. third_task or third_task.json), or 'quit' to exit: "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                logger.info("Exiting interactive loop.")
                break
            result = _resolve_task_name_to_task(user_input)
            if result is None:
                break
            if result is _TASK_NOT_FOUND:
                continue
            start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            run_name = "run_{}".format(start_time)
            try:
                run_dir = _run_single_task(
                    env,
                    result,
                    batch,
                    planner,
                    goal_monitor,
                    server_url,
                    results_base,
                    args,
                    run_name,
                    drone_cam_id,
                    set_cam_at_start=False,
                )
                logger.info("Run saved to %s", run_dir)
            except KeyboardInterrupt:
                logger.info("Task interrupted. Enter another task or 'quit' to exit.")
                continue

    env.close()


if __name__ == "__main__":
    main()
